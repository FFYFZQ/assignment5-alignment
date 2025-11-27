from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from typing import Callable, List, Dict
import torch
import torch.nn.functional as F


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
    )
    output_ids = tokenizer(
        output_strs,
        padding=False,
        truncation=True,
        max_length=1024,
    )  # 长短不一的话不能直接转换为tensor
    ids = []
    prompt_lens = []  # 用于后续计算mask的时候
    for i in range(batch_size):
        p_ids = prompt_ids["input_ids"][i]
        o_ids = output_ids["input_ids"][i]
        ids.append(p_ids + o_ids)
        prompt_lens.append(len(p_ids))

    # 手动填充长度不够的部分
    max_len = max(len(x) for x in ids)
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_input_ids = []
    response_input_mask = []
    for i in range(batch_size):
        i_ids = ids[i]
        i_len = len(i_ids)
        pad_len = max_len - i_len
        pad_input_ids.append(
            i_ids + [pad_token_ids] * pad_len
        )  # 注意只能是列表和列表进行相加

        mask = [0] * prompt_lens[i] + [1] * (i_len - prompt_lens[i]) + [0] * pad_len
        response_input_mask.append(mask)

    pad_input_ids = torch.tensor(pad_input_ids, dtype=torch.long)
    response_input_mask = torch.tensor(response_input_mask, dtype=torch.long)

    input_ids = pad_input_ids[:, :-1]
    labels = pad_input_ids[:, 1:]
    response_mask = response_input_mask[:, 1:]

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    give a logits and compute the sum of entropy for every unit in the last dim
    """
    # logits [bs, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    probs = torch.exp(log_probs)

    p_log_p = -probs * log_probs
    entropy = p_log_p.sum(dim=-1)

    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    model.to(input_ids.device)
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)
        respoonse_logits = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
        result = {"log_probs": respoonse_logits}

    if return_token_entropy:
        entropy = compute_entropy(logits)
        result["token_entropy"] = entropy

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    return a masked normalization sum of the input tensor
    """
    mask_tensor = tensor * mask
    total_sum = torch.sum(mask_tensor, dim=dim)
    norm_sum = total_sum / normalize_constant
    return norm_sum


def cross_entropy(logits: torch.Tensor) -> float:
    """
    compute the cross_entropy loss of the input
    """
    probs = torch.exp(logits)
    p_log_p = -probs * logits
    loss = torch.sum(p_log_p)

    return loss


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    return loss and other metadata for a microbatch of SFT
    """
    mask_log_probs = policy_log_probs * response_mask
    bs = response_mask.shape[0]

    loss = -torch.sum(mask_log_probs) / normalize_constant / bs

    loss_for_backward = loss / gradient_accumulation_steps

    # 反向传播
    loss_for_backward.backward()

    # 第一个返回值是用于 backward 的 loss (虽然在这之后已经不需要再 backward 了，但接口要求返回它)
    # Metadata 中通常记录未经过梯度累积缩放的 loss，便于观察真实的 loss 曲线
    return loss_for_backward, {
        "microbatch_loss": loss.detach(),
    }
