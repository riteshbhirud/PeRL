# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os

import datasets

SYSTEM_PROMPT="You are JoyAI, a large language model trained by JD (京东). For every response, please provide a step-by-step reasoning process enclosed in <think> and </think> tags. After the thinking, you need to output the final answer."

def make_map_fn(split):
    def process_fn(example, idx):
        prompt = example.pop("prompt")
        prompt.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        data = {
            "data_source": example.pop("data_source"),
            "prompt": prompt,
            "ability": "math",
            "reward_model": example.pop("reward_model"),
            "extra_info": example.pop("extra_info")
        }
        return data

    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]

    train_dataset = train_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True,
        num_proc=16,
    )
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
