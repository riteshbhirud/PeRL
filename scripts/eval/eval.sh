# VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=6,7 lm_eval --model vllm \
#     --model_args pretrained="outputs/grpo_full_qwen2_5_3b_20251121_111716/checkpoint-1024",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8 \
#     --tasks gpqa_diamond_zeroshot \
#     --batch_size auto \
#     --apply_chat_template \
#     --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=32768,n=32" \
#     --log_samples \
#     --output_path ./results_deepseek_aime24 \
#     --seed 42 \
#     --trust_remote_code

# VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=6,7 lm_eval --model vllm \
#     --model_args pretrained="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8 \
#     --tasks gpqa_diamond_zeroshot \
#     --batch_size auto \
#     --apply_chat_template \
#     --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=32768,n=32" \
#     --log_samples \
#     --output_path ./results_deepseek_aime24 \
#     --seed 42 \
#     --trust_remote_code \
#     --enable_thinking

CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn && lighteval vllm \
    "model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,tensor_parallel_size=4" \
    "aime24" \
    --save-details \
    --output-dir "outputs"