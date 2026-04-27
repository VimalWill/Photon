import torch
import photon
from torch.profiler import profile, record_function, ProfilerActivity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE  = "cuda"
DTYPE   = torch.float32
REPEATS = 200
WARMUP  = 20

configs = [
    # (batch, seq_len, head_dim)
    # flash path:    d<=64 on sm<90,  d<=128 on sm90 (H100/H200)
    # fallback path: d>64  on sm<90,  d>128  on sm90
    (8,  512,  64),
    (8,  512,  128),
    (8,  512,  256),
    (8,  2048, 64),
    (8,  2048, 128),
    (8,  2048, 256),
    (32, 512,  64),
    (32, 512,  128),
]


def make_inputs(b, n, d):
    return (
        torch.randn(b, n, d, device=DEVICE, dtype=DTYPE),
        torch.randn(b, n, d, device=DEVICE, dtype=DTYPE),
        torch.randn(b, n, d, device=DEVICE, dtype=DTYPE),
    )


def benchmark(b, n, d):
    q, k, v = make_inputs(b, n, d)

    # Warmup
    for _ in range(WARMUP):
        _ = photon.linear_attention(q, k, v)
    torch.cuda.synchronize()

    # Timing with CUDA events (most accurate for GPU kernels)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(REPEATS):
        _ = photon.linear_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / REPEATS
    tflops = (2 * b * n * d * d) / (ms * 1e-3) / 1e12  # FLOPs for two d×d matmuls
    gb_s   = (3 * b * n * d * 4) / (ms * 1e-3) / 1e9   # bytes read (Q, K, V)
    path   = "flash" if d <= 64 else "fallback"
    print(f"  b={b:<3} n={n:<5} d={d:<4} [{path:>8}]  {ms:.3f} ms   {tflops:.2f} TFLOPS   {gb_s:.1f} GB/s")


# ---------------------------------------------------------------------------
# 1. Latency table
# ---------------------------------------------------------------------------
print("=" * 75)
print("Latency")
print("=" * 75)
for cfg in configs:
    benchmark(*cfg)

# ---------------------------------------------------------------------------
# 2. Correctness check against naive reference
# ---------------------------------------------------------------------------
print()
print("=" * 75)
print("Correctness (vs naive kernel-feature-map attention)")
print("=" * 75)

def naive_linear_attn(q, k, v):
    phi_q = 1 + q
    phi_k = 1 + k
    S = torch.einsum("bnd,bnv->bdv", phi_k, v)   # [b, d, d_v]
    z = phi_k.sum(dim=1)                           # [b, d]
    num   = torch.einsum("bnd,bdv->bnv", phi_q, S)
    denom = (phi_q * z.unsqueeze(1)).sum(dim=-1, keepdim=True)
    return num / (denom + 1e-6)

for b, n, d in configs:
    q, k, v = make_inputs(b, n, d)
    ref = naive_linear_attn(q, k, v)
    out = photon.linear_attention(q, k, v)
    err = (ref - out).abs().max().item()
    ok  = "PASS" if err < 1e-3 else "FAIL"
    print(f"  b={b:<3} n={n:<5} d={d:<4}  max_abs_err={err:.2e}  [{ok}]")

# ---------------------------------------------------------------------------
# 3. Torch profiler trace (writes Chrome trace JSON)
# ---------------------------------------------------------------------------
print()
print("=" * 75)
print("Torch profiler trace  →  /tmp/photon_trace.json")
print("=" * 75)

b, n, d = 8, 2048, 64
q, k, v = make_inputs(b, n, d)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(10):
        with record_function("photon.linear_attention"):
            _ = photon.linear_attention(q, k, v)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("/tmp/photon_trace.json")
print("Trace written to /tmp/photon_trace.json")
print('Open in Chrome at  chrome://tracing  or  https://ui.perfetto.dev')
