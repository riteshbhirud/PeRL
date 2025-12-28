#!/bin/bash

set -exo pipefail

PROJECT_DIR="${PROJECT_DIR:-.}"
export HF_ENDPOINT="https://hf-mirror.com"

# 定义要下载的checkpoint列表
# 格式: "MODEL_NAME/CHECKPOINT_NAME"
CHECKPOINTS=(
    "dapo_lora_r16_qwen2_5_3b_20251124_104900/checkpoint-1024"
    "dapo_lora_lr5_20251129_222821/checkpoint-1024"
    "dapo_lora_r8_20251129_135342/checkpoint-1024"
    "grpo_full_qwen2_5_3b_20251121_111716"
    "dr_grpo_lora_20251129_132413/checkpoint-1024"
    "dr_grpo_lora_20251130_192918/checkpoint-1024"
)

# 如果提供了参数，使用参数指定的路径
if [ $# -ge 2 ]; then
    hf download MikaStars39/PeRL \
        --repo-type model \
        --local-dir "${PROJECT_DIR}/ckpts/" \
        --include "$1/$2/*"
else
    # 批量下载所有checkpoint
    echo "=========================================="
    echo "开始下载所有checkpoint..."
    echo "=========================================="
    
    for checkpoint_path in "${CHECKPOINTS[@]}"; do
        echo ""
        echo "下载: ${checkpoint_path}"
        echo "----------------------------------------"
        
        hf download MikaStars39/PeRL \
            --repo-type model \
            --local-dir "${PROJECT_DIR}/ckpts/" \
            --include "${checkpoint_path}/*" || {
                echo "警告: ${checkpoint_path} 下载失败"
                continue
            }
        
        # 检查下载是否成功
        if [[ "${checkpoint_path}" == *"/"* ]]; then
            model_name="${checkpoint_path%%/*}"
            checkpoint_name="${checkpoint_path#*/}"
            if [[ -d "${PROJECT_DIR}/ckpts/${model_name}/${checkpoint_name}" ]]; then
                echo "✓ 成功下载: ${checkpoint_path}"
            else
                echo "✗ 下载失败: ${checkpoint_path}"
            fi
        else
            if [[ -d "${PROJECT_DIR}/ckpts/${checkpoint_path}" ]]; then
                echo "✓ 成功下载: ${checkpoint_path}"
            else
                echo "✗ 下载失败: ${checkpoint_path}"
            fi
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "所有checkpoint下载完成！"
    echo "=========================================="
fi

hf download 5456es/perl_results \
    --repo-type model \
    --local-dir "ckpts/" \
    --include "dapo_lora_fa_20251204_152725/checkpoint-1024/*"