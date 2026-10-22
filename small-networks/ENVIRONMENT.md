# Environment

Every study ran on a CPU, in one Python environment: CPython 3.12.14 with the 35 packages pinned in Study 1's
`studies/1-compiled-programs/02-Code/requirements-v2-portable.txt` (among them NumPy 2.2.0, SciPy 1.16.0, JAX and
jaxlib 0.8.1, dm-haiku 0.0.15 and Matplotlib 3.10.3), the CPU build of PyTorch 2.7.1, and two libraries installed
from their repositories at fixed commits:

| Library | Repository | Commit | Licence |
|---|---|---|---|
| Tracr | `https://github.com/google-deepmind/tracr` | `9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd` | Apache 2.0 |
| CAE (`causal_abstraction_eval`) | `https://github.com/MelouxM/CAE` | `d91f65b5620423d3eae98478bc9843f4a40e361a` | MIT |

`requirements.txt` lists all 38 with their versions. The original environment's complete package list is in
`studies/1-compiled-programs/06-Study-History/V2-SOURCE-ENVIRONMENT-MANIFEST.json` (`installed_packages`).

## A fresh install

```sh
make setup
```

runs the commands with which the environment was built (uv's managed CPython 3.12.14):

```sh
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python -r studies/1-compiled-programs/02-Code/requirements-v2-portable.txt
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  'tracr @ git+https://github.com/google-deepmind/tracr@9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd' \
  'causal_abstraction_eval @ git+https://github.com/MelouxM/CAE@d91f65b5620423d3eae98478bc9843f4a40e361a'
```

Installed, the environment takes about 1.3 GB, most of it PyTorch (650 MB) and jaxlib (320 MB).

## Smaller environments

The quick check needs only NumPy and SciPy, and the figures Matplotlib:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv numpy==2.2.0 scipy==1.16.0 matplotlib==3.10.3
```

Study 1's census and two of its post hoc analyses import Tracr (with JAX and dm-haiku), and the code of Studies 2 and
3 imports PyTorch; `make verify` needs the full environment.

## The libraries' own repositories

Three scripts of Study 1 read files from clones of the libraries' repositories in
`studies/1-compiled-programs/02-Code/vendor/`, which are not included (third-party code): `posthoc/cae_mediator.py`
and `02-Code/reproduce_official_v2.py` read CAE's test suite, and `02-Code/v2_official_grant.py` reads the
`rep_divergence` repository (`https://github.com/grantsrb/rep_divergence`, commit
`f2548d2ea9b4f4b87a87ba5d53db43838d15c521`). `make vendor` fetches the three at the recorded commits, without
their history (network access, about 25 MB, nearly all of it `rep_divergence`).
