# Environment

The study ran in one Python environment, recorded exactly in `uv.lock` (180 locked packages, including this release as
a virtual project); the cloud image of the bigger-model arm was built from the same lock. `pyproject.toml` declares it
as the original project did: the `experiment` extra is the full research environment and the `dev` group adds pytest.

| Package | Version |
|---|---|
| Python | 3.12.14 (`uv` installs it if it is missing) |
| torch | 2.13.0 (CUDA 13 build from PyPI) |
| transformers | 5.16.1 |
| tokenizers | 0.23.1 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| safetensors | 0.8.0 |

Every script ran with `python -B`. The quick check needs only NumPy, SciPy and tokenizers (the number script imports the
prompt module, which imports tokenizers, without loading a tokeniser). The study code imports torch and transformers.

## A fresh install

```sh
uv sync --frozen --extra experiment --group dev        # what make setup runs
```

On Linux x86-64 this downloads 176 packages, about 3.0 GB of wheels. The largest are torch (527 MB) and the
NVIDIA CUDA libraries it depends on (about 2.0 GB together), then triton (198 MB). The sizes come from `uv.lock`;
installing creates `.venv/` in this folder.

## A smaller CPU-only environment

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv numpy==2.5.2 scipy==1.18.1 tokenizers==0.23.1
```

This is enough for `make quick-check` and `make numbers`. It is not the locked environment; for model reruns use
`make setup`.
