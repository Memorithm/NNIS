#!/usr/bin/env python3
"""Build the pinned TinyLlama fixture using the qualified F16 LM-head oracle.

The base converter and checkpoint identity remain unchanged. Only greedy oracle
selection changes: before every argmax, Transformers F32 logits are rounded through
IEEE F16 and widened back to F32, matching the documented NNIS F16 LM-head boundary.
"""

from __future__ import annotations

import torch

import tinyllama_1p1b_chat_fixture as base
from validate_tinyllama_reference_artifact_v2 import (
    CHECKPOINT_SPEC_VERSION,
    REFERENCE_ORACLE_SEMANTICS,
    validate_fixture,
)


def greedy_tokens_f16_lm_head_contract(
    model,
    input_ids: list[int],
    steps: int,
) -> list[int]:
    current = torch.tensor([input_ids], dtype=torch.long, device="cpu")
    generated: list[int] = []
    with torch.inference_mode():
        result = model(input_ids=current, use_cache=True)
        logits = result.logits[0, -1].float().cpu()
        past = result.past_key_values
        for step in range(steps):
            qualified_logits = logits.to(torch.float16).to(torch.float32)
            token = int(torch.argmax(qualified_logits).item())
            generated.append(token)
            if step + 1 == steps:
                break
            result = model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device="cpu"),
                past_key_values=past,
                use_cache=True,
            )
            logits = result.logits[0, -1].float().cpu()
            past = result.past_key_values
    return generated


def main() -> None:
    base.CHECKPOINT_SPEC_VERSION = CHECKPOINT_SPEC_VERSION
    base.REFERENCE_ORACLE_SEMANTICS = REFERENCE_ORACLE_SEMANTICS
    base.validate_fixture = validate_fixture
    base.greedy_tokens = greedy_tokens_f16_lm_head_contract
    base.main()


if __name__ == "__main__":
    main()
