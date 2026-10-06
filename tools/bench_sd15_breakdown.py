"""Per-op-class time breakdown of one SD1.5 UNet call (CUPTI is unavailable
on this box, so each patched F/torch op is wrapped with cuda synchronize +
timer; absolute numbers include sync overhead, the split is what matters)."""
import collections, os, sys, time
sys.argv = [sys.argv[0], "--iters", "1", "--warmup", "3"] + sys.argv[1:]
import torch, torch.nn.functional as F
exec(open(os.path.join(os.path.dirname(__file__), "bench_sd15_unet.py")).read().split("for _ in range(args.warmup)")[0])
for _ in range(args.warmup): step()
acc = collections.defaultdict(float); cnt = collections.Counter(); shapes = collections.defaultdict(float)
def wrap(mod, name):
    orig = getattr(mod, name)
    def w(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig(*a, **k)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
        acc[name] += dt; cnt[name] += 1
        if name in ("conv2d", "scaled_dot_product_attention", "linear"):
            shapes[(name, tuple(a[0].shape), tuple(a[1].shape))] += dt
        return r
    setattr(mod, name, w)
for n in ["conv2d", "linear", "scaled_dot_product_attention", "group_norm", "layer_norm", "gelu", "silu", "interpolate"]:
    wrap(F, n)
torch.cuda.synchronize(); t0 = time.perf_counter(); step(); torch.cuda.synchronize()
total = (time.perf_counter() - t0) * 1000
print(f"total {total:.1f} ms (with per-op syncs)")
for n, v in sorted(acc.items(), key=lambda x: -x[1]):
    print(f"  {n:30s} {v:8.1f} ms  {100*v/total:5.1f}%  ({cnt[n]} calls)")
print(f"  {'other (adds, reshapes, ...)':30s} {total-sum(acc.values()):8.1f} ms")
print("top shapes:")
for (n, a, b), v in sorted(shapes.items(), key=lambda x: -x[1])[:14]:
    print(f"  {v:7.1f} ms  {n:28s} {a} x {b}")
