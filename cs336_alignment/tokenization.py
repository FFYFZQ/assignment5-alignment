from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import Callable, List, Dict
import torch


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """Tokenize prompt/output pairs and build teacher-forcing tensors."""
    assert len(prompt_strs) == len(output_strs)
    batch_size = len(prompt_strs)
    prompt_ids = tokenizer(
        prompt_strs,
        padding=False,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    output_ids = tokenizer(
        output_strs,
        padding=False,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    ids = []
    prompt_lens = []  # 用于后续计算mask的时候
    for i in range(batch_size):
        p_ids = prompt_ids["input_ids"][i]
        o_ids = output_ids["input_ids"][i]
        ids.append(p_ids + o_ids)
        prompt_lens.append(len(p_ids))

    # 手动填充长度不够的部分
    max_len = 1024
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_input_ids = []
    response_input_mask = []
    for i in range(batch_size):
        i_ids = ids[i]
        i_len = len(i_ids)
        pad_len = max_len - i_len
        pad_input_ids.append(i_ids + pad_token_ids * pad_len)

        mask = [0] * prompt_lens[i] + [1] * (i_len - prompt_lens) + [0] * pad_len
        response_input_mask.append(mask)

    pad_input_ids = torch.tensor(pad_input_ids, dtype=torch.long)
    response_input_mask = torch.tensor(response_input_mask, dtype=torch.long)

    input_ids = pad_input_ids[:, :-1]
    labels = pad_input_ids[:, 1:]
    response_mask = response_input_mask[:, 1:]

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
