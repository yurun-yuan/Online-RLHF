import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForSequenceClassification
from accelerate import Accelerator
from typing import Dict, List

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of responses per prompt"},
    )

class ArmoRMPipeline:
    def __init__(self, model_id, device, torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_flash_attention_2=False,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}


accelerator = Accelerator()
local_rank = Accelerator().local_process_index

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device
print(f"local_rank: {local_rank}\tdevice: {device}")

reward_model = script_args.reward_name_or_path
rm_pipe = ArmoRMPipeline(reward_model, device, trust_remote_code=True)


ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")


data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

"""
We process the data format here and query the reward model to get the rewards.
"""


def get_reward(test_texts):
    rewards = [rm_pipe(chat)["score"] for chat in test_texts]
    return rewards


def change_of_format(prom, resp):
    # To be modified according to the reward model and the LLM you use
    # Be careful about multi-turn conversions
    """
    prom = prom.replace("<s>GPT4 Correct User: ", "").replace("<|end_of_turn|>GPT4 Correct Assistant:", "")

    final_resp = resp.split("GPT4 Correct User")[0]
    """
    prom = prom
    final_resp = resp

    message = [
        {"role": "user", "content": prom},
        {"role": "assistant", "content": final_resp},
    ]
    return message


data = []

import itertools

# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["responses"]) < script_args.K:
            continue
        test_texts = [change_of_format(sample['prompt'], response) for response in sample['responses']]
        rewards = get_reward(test_texts)
        data.append({"prompt": sample["prompt"], "responses": sample["responses"], "rewards": rewards})

partial_data = {
    "type": "text_only",
    "instances": data,
}

with open(script_args.output_dir + str(local_rank) + ".json", "w", encoding="utf8") as f:
        json.dump(partial_data, f, ensure_ascii=False)
