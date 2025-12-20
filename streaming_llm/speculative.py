"""
Speculative decoding helpers (exact + greedy_match).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, List

import torch


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for probabilistic sampling.")
    return torch.softmax(logits / temperature, dim=-1)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def propose_tokens(
    model,
    first_logits: torch.Tensor,
    past_key_values,
    k: int,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], object, torch.Tensor]:
    tokens = []
    logits_list = []
    past = past_key_values
    logits = first_logits

    for _ in range(k):
        token = sample_from_logits(logits, temperature, generator=generator)
        logits_list.append(logits)
        tokens.append(token)

        outputs = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    if tokens:
        proposed = torch.cat(tokens, dim=1)
    else:
        proposed = first_logits.new_empty(first_logits.shape[0], 0, dtype=torch.long)
    return proposed, logits_list, past, logits


def greedy_match(
    proposed: torch.Tensor,
    target_next_logits: torch.Tensor,
    target_verify_logits: torch.Tensor,
) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    k = proposed.shape[1]
    accepted = 0
    fallback_token = None
    bonus_token = None

    for i in range(k):
        if i == 0:
            p_logits = target_next_logits
        else:
            p_logits = target_verify_logits[:, i - 1, :]
        greedy = p_logits.argmax(dim=-1, keepdim=True)
        if proposed[:, i : i + 1].equal(greedy):
            accepted += 1
            continue
        fallback_token = greedy
        break

    if accepted == k:
        bonus_logits = target_verify_logits[:, -1, :]
        bonus_token = bonus_logits.argmax(dim=-1, keepdim=True)

    return accepted, fallback_token, bonus_token


def exact_accept(
    proposed: torch.Tensor,
    draft_logits: Iterable[torch.Tensor],
    target_next_logits: torch.Tensor,
    target_verify_logits: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    if temperature <= 0:
        return greedy_match(proposed, target_next_logits, target_verify_logits)

    draft_logits_list = list(draft_logits)
    k = proposed.shape[1]
    accepted = 0
    fallback_token = None
    bonus_token = None
    eps = 1e-8

    for i in range(k):
        if i == 0:
            p_logits = target_next_logits
        else:
            p_logits = target_verify_logits[:, i - 1, :]
        q_logits = draft_logits_list[i]

        p_probs = logits_to_probs(p_logits, temperature)
        q_probs = logits_to_probs(q_logits, temperature)

        token = proposed[:, i]
        p_token = p_probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
        q_token = q_probs.gather(-1, token.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        accept_prob = torch.minimum(torch.ones_like(p_token), p_token / q_token)

        u = torch.rand(accept_prob.shape, device=accept_prob.device, generator=generator)
        if torch.all(u <= accept_prob):
            accepted += 1
            continue

        correction = (p_probs - q_probs).clamp_min(0)
        denom = correction.sum(dim=-1, keepdim=True)
        if torch.all(denom <= eps):
            correction = p_probs
            denom = correction.sum(dim=-1, keepdim=True)
        correction = correction / denom
        fallback_token = torch.multinomial(
            correction, num_samples=1, generator=generator
        )
        break

    if accepted == k:
        bonus_logits = target_verify_logits[:, -1, :]
        bonus_token = sample_from_logits(
            bonus_logits, temperature, generator=generator
        )

    return accepted, fallback_token, bonus_token
