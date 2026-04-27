import torch
import photon

DEVICE = "cuda"
DTYPE  = torch.float32

BATCH_SIZES = [8, 16, 32, 64]
SEQ_LENS    = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
HEAD_DIMS   = [64, 128, 256]

# Determine flash/fallback path label at runtime based on actual device.
def _flash_max_d():
    major = torch.cuda.get_device_properties(0).major
    return 128 if major >= 9 else 64

FLASH_MAX_D = _flash_max_d()


def _repeats(n):
    if n <=  4_096: return 20, 100
    if n <= 16_384: return 10,  50
    if n <= 65_536: return  3,  10
    return 2, 5


def benchmark(b, n, d):
    q = torch.randn(b, n, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn(b, n, d, device=DEVICE, dtype=DTYPE)
    v = torch.randn(b, n, d, device=DEVICE, dtype=DTYPE)

    q = q / d ** 0.5

    warmup, repeats = _repeats(n)
    for _ in range(warmup):
        photon.linear_attention(q, k, v)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(repeats):
        photon.linear_attention(q, k, v)
    t1.record()
    torch.cuda.synchronize()

    ms    = t0.elapsed_time(t1) / repeats
    gb_s  = (4 * b * n * d * 4) / (ms * 1e-3) / 1e9   # Q+K+V reads + Out write
    tflop = (4 * b * n * d * d) / (ms * 1e-3) / 1e12  # S accumulation + output projection
    return ms, gb_s, tflop


def fmt(ms, gb_s, tflop):
    return f"{ms:7.1f}ms {gb_s:6.1f}GB/s {tflop:5.2f}T"


def run_sweep():
    sm = torch.cuda.get_device_properties(0)
    print(f"Device : {sm.name}")
    print(f"Memory : {sm.total_memory / 1e9:.0f} GB")
    print(f"Flash path covers d <= {FLASH_MAX_D} (sm{sm.major}{sm.minor})")

    for d in HEAD_DIMS:
        path = "flash" if d <= FLASH_MAX_D else "fallback"
        col_w = 26
        sep   = "=" * (10 + col_w * len(BATCH_SIZES))

        print()
        print(sep)
        print(f"  d={d}  [{path}]          " +
              "".join(f"{'b='+str(b):>{col_w}}" for b in BATCH_SIZES))
        print(f"  {'n':>6}  " +
              "".join(f"{'ms':>7}  {'GB/s':>6}  {'TFLOPS':>5}  " for _ in BATCH_SIZES))
        print("-" * (10 + col_w * len(BATCH_SIZES)))

        for n in SEQ_LENS:
            row = f"  {n:>6}  "
            for b in BATCH_SIZES:
                ms, gb_s, tflop = benchmark(b, n, d)
                row += fmt(ms, gb_s, tflop) + "  "
            print(row)

        print(sep)


def run_correctness():
    def naive(q, k, v):
        phi_q = 1 + q
        phi_k = 1 + k
        S     = torch.einsum("bnd,bnv->bdv", phi_k, v)
        z     = phi_k.sum(1)
        num   = torch.einsum("bnd,bdv->bnv", phi_q, S)
        denom = (phi_q * z.unsqueeze(1)).sum(-1, keepdim=True)
        return num / (denom + 1e-6)

    print()
    print("=" * 60)
    print("Correctness")
    print("=" * 60)
    checks = [(8, 2048, d) for d in HEAD_DIMS]
    for b, n, d in checks:
        q = torch.randn(b, n, d, device=DEVICE, dtype=DTYPE) / d ** 0.5
        k = torch.randn(b, n, d, device=DEVICE, dtype=DTYPE)
        v = torch.randn(b, n, d, device=DEVICE, dtype=DTYPE)
        ref = naive(q, k, v)
        out = photon.linear_attention(q, k, v)
        err = (ref - out).abs().max().item()
        status = "PASS" if err < 1e-3 else "FAIL"
        print(f"  b={b} n={n} d={d:<4}  max_abs_err={err:.2e}  [{status}]")


if __name__ == "__main__":
    run_correctness()
    run_sweep()
