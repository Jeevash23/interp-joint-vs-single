"""Pilot diagnostic, 17 October 2026 (DEVELOPMENT-NOTES.md): our GPT-2 forward pass against transformers'
GPT2LMHeadModel in float64 on the CPU, on the pilot's two development draws. In float32 the two differed by up to
1.4e-4 in a margin (pilot.json); in float64 an implementation difference would remain and rounding would not.
Writes study/results/pilot-parity-float64.json. No outcome is computed.
"""
from __future__ import annotations

import json

import torch
from safetensors.torch import load_file

import ai10_common as common
import prompts
from gpt2 import END, GPT2, margins_from_logits

torch.set_num_threads(common.CONFIG["cpu_threads"])


def main() -> None:
    from transformers import GPT2LMHeadModel
    tok = prompts.load_tokenizer()
    draws = json.loads((common.PROMPTS / "development-draws.json").read_text())[:2]
    out = {"written_at_utc": common.utc_now(), "draws": [d["draw"] for d in draws], "float64": {}, "float32_cpu": {}}
    for dtype, key in ((torch.float64, "float64"), (torch.float32, "float32_cpu")):
        ref = GPT2LMHeadModel.from_pretrained(str(common.model_dir()), local_files_only=True).to(dtype).eval()
        ours = GPT2(load_file(str(common.model_dir() / "model.safetensors")), "cpu", dtype=dtype)
        for spec in draws:
            tokens = torch.tensor(prompts.task_token_ids(tok, spec))
            ids = list(prompts.class_token_ids(tok, spec))
            x = prompts.inputs(len(spec["templates"]))
            cls = prompts.answer_class(x[:, 0], x[:, 1])
            with torch.no_grad():
                hf = ref(tokens).logits[:, END]
                mine = ours.ln(ours.natural(tokens)["resid"][:, -1, END], *ours.lnf) @ ours.wte.T
            out[key][str(spec["draw"])] = {
                "max_logit_gap": float((hf - mine).abs().max()),
                "max_margin_gap": float((margins_from_logits(hf[:, ids], cls) - margins_from_logits(mine[:, ids], cls))
                                        .abs().max()),
                "max_abs_logit": float(hf.abs().max())}
        del ref, ours
    common.save_json(common.RESULTS / "pilot-parity-float64.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
