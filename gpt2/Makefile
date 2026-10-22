# Joint and single interchange tests in GPT-2 on indirect-object identification: code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` (uv sync) creates. The study's
# own Makefile, part of its freeze, is kept as provenance/Makefile-at-freeze.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
CPU := CUDA_VISIBLE_DEVICES= $(PY) -B
export PYTHONDONTWRITEBYTECODE := 1
export OMP_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
export OPENBLAS_NUM_THREADS ?= 2
export HF_HUB_OFFLINE := 1
.NOTPARALLEL:

.PHONY: help setup python-check unpack check-freezes quick-check test numbers tokenisers models overlap-final

help:
	@echo "make setup          create .venv with the pinned environment (uv sync; downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check    CPU, under a minute, no model and no network: freeze records, every number and table"
	@echo "                    of the manuscript, and the study's four analyses rerun from the saved records"
	@echo "make test           the study's unit tests (needs PyTorch, transformers and GPT-2 Small's tokeniser: make tokenisers)"
	@echo "make numbers        rewrite paper/generated/ from the saved outputs"
	@echo "make tokenisers     download GPT-2 Small's configuration and tokeniser files (a few MB)"
	@echo "make models         download GPT-2 Small (about 0.55 GB), for model reruns and the post hoc verifier"
	@echo "make overlap-final  rerun arm d on the companion study's 200 networks (CPU, about 4 hours; writes into a copy)"

setup:
	uv sync --frozen --extra experiment --group dev

python-check:
	@$(PY) -c "import numpy, scipy, tokenizers" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

quick-check: unpack
	$(CPU) tools/check_freezes.py
	$(CPU) paper/make_numbers.py --check
	$(CPU) tools/replay_analyses.py

test: unpack
	cd study/code && $(CPU) test_ai10.py

numbers: unpack
	$(CPU) paper/make_numbers.py

tokenisers: python-check
	$(PY) tools/get_models.py --no-weights gpt2-small

models: python-check
	$(PY) tools/get_models.py gpt2-small

# Arm d writes its records once and refuses to overwrite them, so it runs in a fresh copy of the release.
overlap-final: unpack
	@test -n "$(COPY)" || { echo "usage: make overlap-final COPY=<new folder> (the release is copied there, without arm d's results)"; exit 1; }
	@test ! -e "$(COPY)" || { echo "$(COPY) exists"; exit 1; }
	cp -a . "$(COPY)" && rm -rf "$(COPY)/study/results/overlap-ai05/final" "$(COPY)/study/results/analysis/overlap-final"
	cd "$(COPY)/study/code" && $(CPU) overlap_ai05.py final && $(CPU) analyse.py overlap-final
