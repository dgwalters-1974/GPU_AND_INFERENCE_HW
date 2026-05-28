import torch
import numpy as np

# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # TODO (1 line): implement a lowest-AI op
    ans = x.clone()
    return ans


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        ans = x
        for _ in range(num_ops):
            ans = ans * x + x
        return ans

    # TODO (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    if compiled:
        return torch.compile(fn)
    else:
        return fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # TODO: time `rep` runs using CUDA events and return median latency (ms)
    
    times = []
    for _ in range(rep):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times)) # follow type hint
    


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # TODO: compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    
    if variant == "compiled":
        bytes_moved = num_elements * 2 * bytes_per_element
    else:
        bytes_moved = num_elements * bytes_per_element * 6 * num_ops 
    
    total_flops = num_ops * num_elements * 2
    ai = total_flops / bytes_moved
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops
    

# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
# 
#Ans:
# Compiled runtime is dominated by memory traffic to and from the  HBM.The fused kernel reads
# x once and writes ans once per element. So bytes moved is fixed by tensor size and independent 
# of num_ops. At HBM bandwidth, that sets a roughly constant wall-clock time (~0.21 ms here). 
# Meanwhile total FLOPs = (2N * num_ops) grows linearly with num_ops. Since FLOP/s =  FLOPs / runtime, 
# the numerator grows while the denominator stays flat, so achieved FLOP/s scales linearly with num_ops. 
# This only holds while we're memory-bound — once the register calc time catches up with the memory
# time (past the ridge point), runtime starts increasing too, and FLOP/s saturates near peak.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
#Ans:
# The 1024×1024 matmul is too small to saturate the H100's 132 SMs: it produces only ~64 output    
# "tiles" (one per 128×128 block), so at most half the SM's are doing useful work at any moment.
# Kernel launch and sync overheads (~10 µs) also become a non-trivial fraction of the 67 µs
# runtime. Both effects shrink (in relative terms) as the matmul grows, which is why 2048 and 4096 
# matmuls climb back toward peak FP32 throughput.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
#Ans:
# Up to 64 ops, AI stayed below the ridge of 20 FLOP/B, so the kernel was memory-bound — the SMs   
# had slack capacity sitting idle while waiting on HBM. Adding more arithmetic was free because it 
# filled in the idle time. At 128 ops, AI = 32 puts us above the ridge, so the kernel is now       
# compute-bound and in a different region. The SMs are running at their FP32 throughput limit,
# and any additional arithmetic translates directly into more time on the compute units.
# This is why runtime first ticks up noticeably (0.21 to 0.32 ms) — we've stopped getting
# "free" FLOPs and are now paying for them.


# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
#Ans:
# In eager mode each operator runs as its own kernel, so every multiply and add in the acc 
# = acc * x + x loop forces a round trip through HBM. Bytes moved grows linearly with 
# num_ops — and so do FLOPs, so the ratio (AI) is pinned at a constant ~0.083 FLOP/B regardless
# of how much arithmetic we do. All the eager points therefore pile up at the same (AI, TFLOP/s)
# location on the roofline, sitting right on the memory-bandwidth ceiling. 
# Runtime grows linearly with num_ops because we pay full HBM traffic on every iteration while 
# achieved FLOP/s is constant at ≈ 0.083*3.35 TB/s ≈ 0.28 TFLOP/s, two orders of magnitude below 
# what the compiled fused kernel reaches.