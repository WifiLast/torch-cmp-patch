"""SD1.5 / SDXL UNet step benchmark at the webui's real shapes (CMP 50HX).

    --arch sdxl: SDXL UNet, 1024x1024 (latent 128x128), CFG batch 2, 154-token
    context (2 prompt chunks). Pass --h/--w/--ctx to change.

Mirrors a stable-diffusion-webui-amdgpu generation: 512x1024 -> latent
4x128x64, CFG batch 2 (cond + uncond), 308-token context (4 prompt chunks),
fp16. Random weights -- this measures speed, not images.

    CMPEXT3_AUTOPATCH=0 python tools/bench_sd15_unet.py   # stock torch
    python tools/bench_sd15_unet.py                       # with cmpext3
    python tools/bench_sd15_unet.py --profile             # + top ops by CUDA time
"""
import argparse
import os
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--arch", choices=["sd15", "sdxl"], default="sd15")
p.add_argument("--h", type=int, default=128, help="latent height (1024 px / 8)")
p.add_argument("--w", type=int, default=None, help="latent width (sd15: 64 = 512 px, sdxl: 128)")
p.add_argument("--batch", type=int, default=2)
p.add_argument("--ctx", type=int, default=None, help="context tokens (sd15: 308, sdxl: 154)")
p.add_argument("--warmup", type=int, default=4)
p.add_argument("--iters", type=int, default=10)
p.add_argument("--profile", action="store_true")
p.add_argument("--top", type=int, default=25)
args, _ = p.parse_known_args()
if args.w is None:
    args.w = 64 if args.arch == "sd15" else 128
if args.ctx is None:
    args.ctx = 308 if args.arch == "sd15" else 154

import torch.nn.functional as F  # noqa: E402
from diffusers import UNet2DConditionModel  # noqa: E402

patched = os.environ.get("CMPEXT3_AUTOPATCH", "1") != "0"
try:
    import cmpext3
    patched = patched and cmpext3.is_enabled()
except ImportError:
    cmpext3 = None
    patched = False

dev = "cuda"
torch.manual_seed(0)
if args.arch == "sd15":
    unet = UNet2DConditionModel(   # SD1.5 config
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280), cross_attention_dim=768,
        attention_head_dim=8,
        down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
        up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
    ).to(dev, torch.float16).eval()
    ctx_dim, extra = 768, {}
else:
    unet = UNet2DConditionModel(   # SDXL base config
        sample_size=128, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(320, 640, 1280), cross_attention_dim=2048,
        attention_head_dim=(5, 10, 20), transformer_layers_per_block=(1, 2, 10),
        use_linear_projection=True, addition_embed_type="text_time",
        addition_time_embed_dim=256, projection_class_embeddings_input_dim=2816,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    ).to(dev, torch.float16).eval()
    ctx_dim = 2048
    extra = {"added_cond_kwargs": {
        "text_embeds": torch.randn(args.batch, 1280, device=dev, dtype=torch.float16),
        "time_ids": torch.tensor([[args.h * 8, args.w * 8, 0, 0, args.h * 8, args.w * 8]] * args.batch,
                                 device=dev, dtype=torch.float16)}}

x = torch.randn(args.batch, 4, args.h, args.w, device=dev, dtype=torch.float16)
ctx = torch.randn(args.batch, args.ctx, ctx_dim, device=dev, dtype=torch.float16)
t = torch.tensor([500] * args.batch, device=dev)


def step():
    with torch.no_grad():
        return unet(x, t, encoder_hidden_states=ctx, **extra).sample


for _ in range(args.warmup):
    step()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(args.iters):
    out = step()
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / args.iters * 1000
print(f"{'cmpext3' if patched else 'stock'}: {ms:.1f} ms/UNet call "
      f"(B{args.batch} {args.h}x{args.w} ctx{args.ctx}) -> "
      f"{1000 / ms:.2f} it/s, 20 steps = {ms * 20 / 1000:.1f} s; "
      f"finite={torch.isfinite(out).all().item()}")

if args.profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.top,
                                    max_name_column_width=70))
if patched and os.environ.get("BENCH_REPORT"):
    print(cmpext3.autoselect_report())
