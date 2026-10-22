# Model checkpoints

No weights are included. The study code reads GPT-2 Small from `models/gpt2-small/` (or from the folder named by
`MODELS_DIR`); `tools/get_models.py` downloads it at the revision the study used and stops if any file's SHA-256
differs from the record:

| Name and folder | Hugging Face repository | Revision | Size | Used by |
|---|---|---|---|---|
| `gpt2-small` | `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 0.55 GB | the study (GPT-2 Small, on the local GPU); the post hoc NumPy verifier |
| `gpt2-medium` | `openai-community/gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | 1.52 GB | the bigger-model arm (downloaded inside its cloud image instead) |

```sh
python tools/get_models.py --list          # the table above
python tools/get_models.py gpt2-small      # what the study and the post hoc verifier need (make models)
python tools/get_models.py --check         # check what is here, without network access
```

The study recorded the checkpoint's safetensors digest in its configuration and freeze record, and checks it before a
run. GPT-2 Medium was used only by the bigger-model arm, which downloads it inside its cloud image
(`study/cloud/ai10_modal.py`) and checks it against the same digests. The checkpoints are distributed under their own
licence (MIT for GPT-2).
