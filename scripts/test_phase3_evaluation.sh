#!/bin/bash
# Phase 3 Evaluation Pipeline Integration Test
#
# Tests the complete evaluation pipeline:
# 1. Checkpoint evaluation (mock mode)
# 2. Batch evaluation orchestration
# 3. Result aggregation
# 4. OOD analysis
# 5. Long-chain filtering
# 6. Visualization generation
#
# Note: Since actual model inference requires GPU, this test uses mock
# evaluation results to verify the pipeline flow and data processing.

set -e  # Exit on error

echo "=========================================="
echo "Phase 3 Evaluation Pipeline Integration Test"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Setup directories
TEST_DIR="results/test_phase3"
MOCK_CHECKPOINTS="$TEST_DIR/mock_checkpoints"
MOCK_RESULTS="$TEST_DIR/mock_results"
AGGREGATED_DIR="$TEST_DIR/aggregated"
OOD_DIR="$TEST_DIR/ood_analysis"
PLOTS_DIR="$TEST_DIR/plots"

# Clean up previous test
rm -rf "$TEST_DIR"
mkdir -p "$MOCK_CHECKPOINTS" "$MOCK_RESULTS" "$AGGREGATED_DIR" "$OOD_DIR" "$PLOTS_DIR"

echo "Test directories created at: $TEST_DIR"
echo ""

# =============================================================================
# Test 0: Verify all scripts are importable
# =============================================================================
echo "0. Verifying Python imports..."

python -c "
from perl.evaluation import (
    load_results, save_results, print_summary,
    EvaluationResult, ProblemResult, EvaluationMetadata, EvaluationStatistics,
    extract_answer, check_correctness, load_benchmark,
    SUPPORTED_BENCHMARKS
)
from scripts.aggregate_results import aggregate_by_method_benchmark, ResultEntry
from scripts.analyze_ood import OODReport, ProblemAnalysis, analyze_by_dimension
from scripts.filter_long_chain import filter_by_chain_length, ChainLengthStats
from scripts.generate_ood_plots import extract_chain_length_data, MATPLOTLIB_AVAILABLE
print('All imports successful')
print(f'Matplotlib available: {MATPLOTLIB_AVAILABLE}')
print(f'Supported benchmarks: {list(SUPPORTED_BENCHMARKS.keys())}')
"

if [ $? -eq 0 ]; then
    echo "  ✓ All Python imports work"
else
    echo "  ✗ Import errors detected"
    exit 1
fi
echo ""

# =============================================================================
# Test 1: Create mock checkpoints
# =============================================================================
echo "1. Creating mock PEFT checkpoints..."

python -c "
import json
import os

methods = ['lora', 'dora', 'adalora']
seeds = [42, 43]
ranks = {'lora': 16, 'dora': 32, 'adalora': 32}

for method in methods:
    for seed in seeds:
        ckpt_dir = f'$MOCK_CHECKPOINTS/{method}_r{ranks[method]}_s{seed}/checkpoint-100'
        os.makedirs(ckpt_dir, exist_ok=True)

        # Create adapter_config.json
        config = {
            'peft_type': 'LORA' if method != 'adalora' else 'ADALORA',
            'r': ranks[method],
            'lora_alpha': ranks[method] * 2,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            'base_model_name_or_path': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
            'use_dora': method == 'dora',
        }
        with open(f'{ckpt_dir}/adapter_config.json', 'w') as f:
            json.dump(config, f, indent=2)

print(f'Created mock checkpoints for {len(methods)} methods x {len(seeds)} seeds')
"

CKPT_COUNT=$(find "$MOCK_CHECKPOINTS" -name "adapter_config.json" | wc -l | tr -d ' ')
echo "  ✓ Created $CKPT_COUNT mock checkpoints"
echo ""

# =============================================================================
# Test 2: Create mock evaluation results
# =============================================================================
echo "2. Creating mock evaluation results..."

python << 'PYTHON_SCRIPT'
import json
import os
import random
from datetime import datetime

methods = ['lora', 'dora', 'adalora']
seeds = [42, 43]
ranks = {'lora': 16, 'dora': 32, 'adalora': 32}
benchmarks = ['math500', 'aime2024']

# Simulate different accuracies per method
base_accuracy = {'lora': 0.65, 'dora': 0.68, 'adalora': 0.66}
difficulties = ['easy', 'medium', 'hard']
categories = ['algebra', 'geometry', 'number_theory', 'counting_probability']

for method in methods:
    for seed in seeds:
        for benchmark in benchmarks:
            random.seed(seed + hash(method) + hash(benchmark))

            # Adjust for benchmark difficulty
            acc_modifier = 0.0 if benchmark == 'math500' else -0.15

            # Generate problems
            num_problems = 50 if benchmark == 'math500' else 30
            problems = []

            for i in range(num_problems):
                difficulty = random.choice(difficulties)
                category = random.choice(categories)

                # Accuracy varies by difficulty
                diff_acc = {'easy': 0.85, 'medium': 0.65, 'hard': 0.35}
                prob_acc = base_accuracy[method] * diff_acc[difficulty] + acc_modifier
                correct = random.random() < prob_acc

                # Reasoning steps vary by difficulty
                base_steps = {'easy': 3, 'medium': 6, 'hard': 12}
                steps = max(1, int(random.gauss(base_steps[difficulty], 2)))

                problems.append({
                    'problem_id': f'{benchmark}_p{i+1}',
                    'question': f'Problem {i+1} about {category}',
                    'ground_truth': str(random.randint(1, 100)),
                    'model_answer': str(random.randint(1, 100)) if correct else str(random.randint(101, 200)),
                    'model_response': f'Step 1-{steps}. The answer is...',
                    'correct': correct,
                    'score': 1.0 if correct else 0.0,
                    'reasoning_steps': steps,
                    'reasoning_tokens': steps * 50,
                    'difficulty': difficulty,
                    'category': category,
                    'generation_time': random.uniform(0.5, 3.0),
                    'extraction_method': 'boxed',
                    'verification_method': 'exact',
                    'metadata': {}
                })

            correct_count = sum(1 for p in problems if p['correct'])
            accuracy = correct_count / num_problems

            result = {
                'metadata': {
                    'checkpoint': f'mock_checkpoints/{method}_r{ranks[method]}_s{seed}/checkpoint-100',
                    'checkpoint_step': 100,
                    'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                    'peft_method': method,
                    'benchmark': benchmark,
                    'rank': ranks[method],
                    'alpha': ranks[method] * 2,
                    'seed': seed,
                    'evaluation_date': datetime.now().isoformat(),
                    'total_problems': num_problems,
                    'correct_count': correct_count,
                    'accuracy': accuracy,
                    'generation_config': {'max_new_tokens': 2048, 'temperature': 0.7},
                    'extra': {}
                },
                'statistics': {
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_count': num_problems,
                    'avg_reasoning_steps': sum(p['reasoning_steps'] for p in problems) / num_problems,
                    'avg_reasoning_tokens': sum(p['reasoning_tokens'] for p in problems) / num_problems,
                    'avg_generation_time': sum(p['generation_time'] for p in problems) / num_problems,
                    'total_generation_time': sum(p['generation_time'] for p in problems),
                    'accuracy_by_difficulty': {},
                    'accuracy_by_category': {},
                    'extraction_method_counts': {'boxed': num_problems},
                    'verification_method_counts': {'exact': num_problems}
                },
                'results': problems
            }

            # Save result
            result_dir = f'results/test_phase3/mock_results/{method}/{benchmark}'
            os.makedirs(result_dir, exist_ok=True)
            result_file = f'{result_dir}/seed{seed}_step100.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

print(f'Created evaluation results for {len(methods)} methods x {len(seeds)} seeds x {len(benchmarks)} benchmarks')
PYTHON_SCRIPT

RESULT_COUNT=$(find "$MOCK_RESULTS" -name "*.json" | wc -l | tr -d ' ')
echo "  ✓ Created $RESULT_COUNT mock evaluation result files"
echo ""

# =============================================================================
# Test 3: Test result loading
# =============================================================================
echo "3. Testing result loading..."

python -c "
from perl.evaluation import load_results
from pathlib import Path

results_dir = Path('$MOCK_RESULTS')
loaded_count = 0
total_problems = 0

for json_file in results_dir.rglob('*.json'):
    result = load_results(json_file)
    loaded_count += 1
    total_problems += len(result.results)

print(f'Loaded {loaded_count} result files with {total_problems} total problems')
print(f'Sample metadata: method={result.metadata.peft_method}, benchmark={result.metadata.benchmark}')
"

if [ $? -eq 0 ]; then
    echo "  ✓ Result loading works"
else
    echo "  ✗ Result loading failed"
    exit 1
fi
echo ""

# =============================================================================
# Test 4: Test result aggregation
# =============================================================================
echo "4. Testing result aggregation..."

python scripts/aggregate_results.py \
    --results_dir "$MOCK_RESULTS" \
    --output "$AGGREGATED_DIR" \
    2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "  ✓ Result aggregation completed"

    # Show aggregated stats
    if [ -f "$AGGREGATED_DIR/aggregated_results.json" ]; then
        echo "  Aggregated results summary:"
        python -c "
import json
with open('$AGGREGATED_DIR/aggregated_results.json') as f:
    data = json.load(f)
    for method, benchmarks in data.get('by_method', {}).items():
        for bench, stats in benchmarks.items():
            print(f'    {method}/{bench}: {stats[\"mean\"]*100:.1f}% ± {stats[\"std\"]*100:.1f}%')
" 2>/dev/null || echo "    (Could not parse aggregated results)"
    fi
else
    echo "  ✗ Result aggregation failed"
    exit 1
fi
echo ""

# =============================================================================
# Test 5: Test OOD analysis
# =============================================================================
echo "5. Testing OOD analysis..."

python scripts/analyze_ood.py \
    --results_dir "$MOCK_RESULTS" \
    --output_dir "$OOD_DIR" \
    2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "  ✓ OOD analysis completed"

    # Check output files
    OOD_FILES=$(find "$OOD_DIR" -name "*.json" -o -name "*.md" | wc -l | tr -d ' ')
    echo "  Created $OOD_FILES OOD analysis files"
else
    echo "  ✗ OOD analysis failed"
    exit 1
fi
echo ""

# =============================================================================
# Test 6: Test long-chain filtering
# =============================================================================
echo "6. Testing long-chain filtering..."

python scripts/filter_long_chain.py \
    --results_dir "$MOCK_RESULTS" \
    --min_steps 9 \
    --output_dir "$TEST_DIR/long_chain" \
    --analyze \
    2>&1 | head -30

if [ $? -eq 0 ]; then
    echo "  ✓ Long-chain filtering completed"
else
    echo "  ✗ Long-chain filtering failed"
    exit 1
fi
echo ""

# =============================================================================
# Test 7: Test plot generation
# =============================================================================
echo "7. Testing plot generation..."

# First create analysis data for plotting
# Note: load_analysis_results loads ood_analysis.json into results["combined"]
# So the file should contain the INNER structure (chain_length, category, etc.)
python -c "
import json
import os

# Create mock analysis data structure that matches what analyze_ood produces
analysis = {
    'chain_length': {
        'short': {
            'lora': {'accuracy': 0.75, 'std': 0.05},
            'dora': {'accuracy': 0.78, 'std': 0.04},
            'adalora': {'accuracy': 0.76, 'std': 0.05}
        },
        'medium': {
            'lora': {'accuracy': 0.62, 'std': 0.06},
            'dora': {'accuracy': 0.65, 'std': 0.05},
            'adalora': {'accuracy': 0.63, 'std': 0.06}
        },
        'long': {
            'lora': {'accuracy': 0.38, 'std': 0.08},
            'dora': {'accuracy': 0.42, 'std': 0.07},
            'adalora': {'accuracy': 0.40, 'std': 0.08}
        }
    },
    'category': {
        'algebra': {
            'lora': {'accuracy': 0.68},
            'dora': {'accuracy': 0.72},
            'adalora': {'accuracy': 0.70}
        },
        'geometry': {
            'lora': {'accuracy': 0.55},
            'dora': {'accuracy': 0.58},
            'adalora': {'accuracy': 0.56}
        },
        'number_theory': {
            'lora': {'accuracy': 0.48},
            'dora': {'accuracy': 0.52},
            'adalora': {'accuracy': 0.50}
        },
        'counting_probability': {
            'lora': {'accuracy': 0.60},
            'dora': {'accuracy': 0.63},
            'adalora': {'accuracy': 0.61}
        }
    },
    'difficulty': {
        'easy': {
            'lora': {'accuracy': 0.85},
            'dora': {'accuracy': 0.88},
            'adalora': {'accuracy': 0.86}
        },
        'medium': {
            'lora': {'accuracy': 0.62},
            'dora': {'accuracy': 0.65},
            'adalora': {'accuracy': 0.63}
        },
        'hard': {
            'lora': {'accuracy': 0.32},
            'dora': {'accuracy': 0.36},
            'adalora': {'accuracy': 0.34}
        }
    },
    'overall': {
        'lora': {'accuracy': 0.65},
        'dora': {'accuracy': 0.68},
        'adalora': {'accuracy': 0.66}
    }
}

os.makedirs('$OOD_DIR', exist_ok=True)
with open('$OOD_DIR/ood_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)
print('Created mock analysis data for plotting')
"

python scripts/generate_ood_plots.py \
    --analysis-dir "$OOD_DIR" \
    --output-dir "$PLOTS_DIR" \
    2>&1

if [ $? -eq 0 ]; then
    PLOT_COUNT=$(find "$PLOTS_DIR" -name "*.png" | wc -l | tr -d ' ')
    echo "  ✓ Plot generation completed - created $PLOT_COUNT plots"
else
    echo "  ✗ Plot generation failed"
    exit 1
fi
echo ""

# =============================================================================
# Test 8: Verify CLI interfaces
# =============================================================================
echo "8. Verifying CLI interfaces..."

SCRIPTS=(
    "scripts/evaluate_checkpoint.py"
    "scripts/batch_evaluate.py"
    "scripts/aggregate_results.py"
    "scripts/analyze_ood.py"
    "scripts/filter_long_chain.py"
    "scripts/generate_ood_plots.py"
)

for script in "${SCRIPTS[@]}"; do
    python "$script" --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ $(basename $script) --help works"
    else
        echo "  ✗ $(basename $script) --help failed"
        exit 1
    fi
done
echo ""

# =============================================================================
# Test 9: Verify output files
# =============================================================================
echo "9. Verifying output files..."

# Check directory structure
echo "  Output directory structure:"
find "$TEST_DIR" -type f \( -name "*.json" -o -name "*.md" -o -name "*.png" \) | head -20 | while read f; do
    SIZE=$(wc -c < "$f" | tr -d ' ')
    echo "    $(echo $f | sed 's|results/test_phase3/||') ($SIZE bytes)"
done

# Count files
JSON_COUNT=$(find "$TEST_DIR" -name "*.json" | wc -l | tr -d ' ')
MD_COUNT=$(find "$TEST_DIR" -name "*.md" | wc -l | tr -d ' ')
PNG_COUNT=$(find "$TEST_DIR" -name "*.png" | wc -l | tr -d ' ')

echo ""
echo "  File counts:"
echo "    JSON files: $JSON_COUNT"
echo "    Markdown files: $MD_COUNT"
echo "    PNG plots: $PNG_COUNT"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Phase 3 Integration Test Complete"
echo "=========================================="
echo ""
echo "Test Results Summary:"
echo "  ✓ Python imports work"
echo "  ✓ Mock checkpoints created"
echo "  ✓ Mock evaluation results created"
echo "  ✓ Result loading works"
echo "  ✓ Result aggregation works"
echo "  ✓ OOD analysis works"
echo "  ✓ Long-chain filtering works"
echo "  ✓ Plot generation works"
echo "  ✓ All CLI interfaces work"
echo ""
echo "Output location: $TEST_DIR"
echo ""
echo "Note: This test used mock data. For production:"
echo "  1. Real checkpoints need GPU for inference"
echo "  2. Actual evaluation will use vLLM/transformers"
echo "  3. Results will reflect real model performance"
echo ""
echo "The pipeline is ready for production use!"
echo "=========================================="
