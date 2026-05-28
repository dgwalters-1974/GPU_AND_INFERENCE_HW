import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)
from transformers import DynamicCache

def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`
    
    kache = DynamicCache()
    generated_tokens = []
    
    with torch.inference_mode():
        outputs = model(input_ids = input_ids,
                        past_key_values = kache,
                        use_cache = True)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id)
    
        for _ in range(n_steps - 1):
            outputs = model(input_ids=next_token_id.unsqueeze(0),
                            past_key_values = kache,
                            use_cache = True)
            
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token_id)
        
        return torch.cat(generated_tokens).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    with torch.profiler.profile(
        activities = [torch.profiler.ProfilerActivity.CPU, 
                      torch.profiler.ProfilerActivity.CUDA],
        record_shapes = True,
        profile_memory = True,
        with_stack = True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.
    model = build_model(torch.bfloat16)  # Not float16!                    
    input_ids = get_input_ids()                                            
    profile(optimized_loop, model, input_ids, optimized_trace_name)               
    t_elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")    
    del model                                                                
    torch.cuda.empty_cache()
    return t_elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix: 
#                          
#   1. KV cache  + only feeding the new token each step.                                                
#      The slow loop re-encoded the full (growing) context every iteration (O(L^2) work over the run). 
#      With KV cache, each decoder step only does new work for the single new
#     token — past K/V are reused from cache and so are the past tokens' Q/FFN/output 
#     projections (they simply aren't recomputed). Matmul traffic drops from 
#     2.3 GB to 97 MB. Largest single contributor.
  
#   │           Metric           │  Slow   │ Optimized │                       
#   ├────────────────────────────┼─────────┼───────────┤                                    
#   │ Wall time                  │ 0.95s   │ 0.19s     │
#   ├────────────────────────────┼─────────┼───────────┤                                    
#   │ Self CUDA time             │ 80 ms   │ 4.6 ms    │                                  
#   ├────────────────────────────┼─────────┼───────────┤                                    
#   │ Matmul CUDA output volume  │ 2.26 GB │ 97 MB     │  
#
#     Matmul output volume drops from 2.26 GB to 97 MB (~24x less work to do)
#     Per fix speed up not measured in isolation but this change is responsible for
#     a significant portion of the overall speedup.
#                         
#   2. BF16 instead of FP32. Three effects:
#       - 2x less HBM bandwidth per matmul (smaller dtype - 2 vs. 4 bytes per element).   
#       - Unlocks H100 tensor cores: ~67 TFLOP/s FP32 (CUDA cores) vs  around 989 TFLOP/s 
#     BF16 (tensor cores) -> around 15x per-FLOP throughput.
#       - Unlocks Flash Attention.    
#                                                             
#   3. torch.inference_mode(). Disables autograd. Small but free CPU-side win.                      
#                      
# Measured end-to-end:                        
#   Wall time:       0.95s to 0.19s   (5.09 times)     
#   Self CUDA time:  80 ms to 4.6 ms  (~17 times)    
#                                                 
# The gap between 17x CUDA time and 5x wall time tells us the optimized                            
# version is now CPU-bound on Python kernel side.

                                       
# Biggest impact and why:    
#                               
#   KV cache. It changes the amount of work the model has to do, not just how 
#   fast each unit of work runs. Without it, even BF16 + tensor cores + Flash Attention 
#   would still be re-encoding the full prompt every decode step. The KV cache 
#   eliminates that redundant work, BF16 then compounds things by making the remaining, 
#   smaller workload run on faster hardware routes. The two effects multiply: less work to 
#   do, and faster per-FLOP processing.
