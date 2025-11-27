import torch
import os
import json
from typing import Callable, List
from vllm import LLM
import torch
import os
import json
from typing import Callable, List
from vllm import LLM, SamplingParams
from pathlib import Path
from .drgrpo_grader import r1_zero_reward_fn

DATA_PATH = r"E:\Study\assignment5-alignment\data\gsm8k\test.jsonl"


def load_data(data_path):
    """
    从本地加载数据返回data
    """
    data_path = Path(data_path)
    if data_path.exists():
        print(f"load data from disk")
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    return data


def create_prompt(question):
    return f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer> User: {question} Assistant: <think>"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    gt: List[str],
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    scores = []
    serializad = []
    for output, g in zip(outputs, gt):
        text = output.outputs[0].text
        score_dict = reward_fn(text, g)
        scores.append(score_dict)
        serializad.append(
            {
                "prompt": output.prompt,
                "generation": text,
                "ground_truth": g,
                "score": score_dict,
            }
        )

    result_path = Path("evaluation") / "eval_result.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as f:
        for item in serializad:
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )  # ensure_ascii =false 让dumps输出非ascii字符，比如中文


def main():
    # 1.加载数据
    dataset = load_data(DATA_PATH)
    prompts = [create_prompt(data["question"]) for data in dataset]
    gt = [data["answer"] for data in dataset]
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        gt=gt,
    )
