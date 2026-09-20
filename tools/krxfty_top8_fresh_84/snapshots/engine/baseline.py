"""The organizers' native engine, kept verbatim as the fallback and the self-check reference."""

import torch
from transformers import AutoModelForCausalLM


class BaselineEngine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits at the last position only, [B, V] float32 (a
        full [B, T, V] tensor would be gigabytes at batch 16)."""
        return self.model(input_ids=input_ids, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float().clone()

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        current = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        cache = None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                output = self.model(
                    input_ids=current,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist()
