from lora.adapter import apply_peft, PEFT_TYPE_TO_FUNCTION_MAPPING
from transformers import AutoModelForCausalLM

from argparse import Namespace

def param_test(
    r: int = 32,
):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    results = {}
    
    # 加载一次模型来计算总参数数
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    total_params = sum(p.numel() for p in base_model.parameters())
    total_params_m = total_params / 1_000_000
    print(f"\n模型总参数: {total_params_m:.4f}M ({total_params:,})\n")
    print("=" * 80)
     
    for peft_type in PEFT_TYPE_TO_FUNCTION_MAPPING.keys():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        args = Namespace(
            peft=Namespace(
                type=peft_type,
                r=r,
                lora_alpha=64,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
            ),
            training=Namespace(
                learning_rate=1e-5,
                beta=0.0,
                max_steps=1000,
            )
        )
        
        try:
            optimizer, peft_model = PEFT_TYPE_TO_FUNCTION_MAPPING[peft_type](model, args)
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            trainable_params_m = trainable_params / 1_000_000
            percentage = (trainable_params / total_params) * 100
            results[peft_type] = {
                'trainable_params_m': trainable_params_m,
                'trainable_params': trainable_params,
                'percentage': percentage
            }
            print(f"{peft_type:30s}: {trainable_params_m:8.4f}M ({percentage:6.2f}%)")
        except Exception as e:
            print(f"{peft_type:30s}: Error - {str(e)}")
            results[peft_type] = None
             
    # 输出总结
    print("=" * 80)
    print("\n总结:")
    print("-" * 80)
    successful_results = {k: v for k, v in results.items() if v is not None}
    if successful_results:
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['trainable_params'])
        print(f"{'方法':<30s} {'可训练参数(M)':>15s} {'百分比':>10s}")
        print("-" * 80)
        for peft_type, data in sorted_results:
            print(f"{peft_type:<30s} {data['trainable_params_m']:>15.4f} {data['percentage']:>9.2f}%")
        print("-" * 80)
        min_params_type = min(successful_results.items(), key=lambda x: x[1]['trainable_params'])
        max_params_type = max(successful_results.items(), key=lambda x: x[1]['trainable_params'])
        print(f"\n最少可训练参数: {min_params_type[0]} ({min_params_type[1]['trainable_params_m']:.4f}M, {min_params_type[1]['percentage']:.2f}%)")
        print(f"最多可训练参数: {max_params_type[0]} ({max_params_type[1]['trainable_params_m']:.4f}M, {max_params_type[1]['percentage']:.2f}%)")
    else:
        print("所有方法都失败了，没有可用的结果。")
    
    return results

if __name__ == "__main__":
    results = param_test()