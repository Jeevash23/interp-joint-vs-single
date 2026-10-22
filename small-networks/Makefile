# When can a joint intervention reject an alignment that single interventions accept? Code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
.NOTPARALLEL:

S1 := studies/1-compiled-programs
VENDOR := $(S1)/02-Code/vendor
TRACR := 9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd
CAE := d91f65b5620423d3eae98478bc9843f4a40e361a
REPDIV := f2548d2ea9b4f4b87a87ba5d53db43838d15c521

.PHONY: help setup python-check unpack check-freezes quick-check verify numbers figures vendor

help:
	@echo "make setup          create .venv with the recorded environment (downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check    CPU, about a minute, NumPy and SciPy only, no network: freeze records, every number"
	@echo "                    and table of the manuscript, and seven saved-output analyses rerun and compared"
	@echo "make verify         quick-check plus Study 1's census recomputed from the compiled networks and three"
	@echo "                    more analyses rerun (needs the full environment: make setup)"
	@echo "make numbers        rewrite paper/generated/ from the saved outputs"
	@echo "make figures        draw Figures 2 and 3 into paper/figures/"
	@echo "make vendor         fetch the three third-party repositories Study 1 kept clones of (about 25 MB)"

setup:
	uv venv --python 3.12.14 .venv
	uv pip install --python .venv/bin/python -r $(S1)/02-Code/requirements-v2-portable.txt
	uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
	uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
	  'tracr @ git+https://github.com/google-deepmind/tracr@$(TRACR)' \
	  'causal_abstraction_eval @ git+https://github.com/MelouxM/CAE@$(CAE)'

python-check:
	@$(PY) -c "import numpy, scipy" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

quick-check: unpack
	$(PY) tools/check_freezes.py
	$(PY) paper/make_numbers.py --check
	$(PY) tools/replay_analyses.py

verify: quick-check
	$(PY) $(S1)/posthoc/census.py --check
	$(PY) tools/replay_analyses.py --all

numbers: unpack
	$(PY) paper/make_numbers.py

figures: unpack
	$(PY) paper/make_figures.py

# Each repository at its recorded commit, without history (GitHub serves a commit by its full hash).
vendor:
	for r in tracr:https://github.com/google-deepmind/tracr:$(TRACR) cae:https://github.com/MelouxM/CAE:$(CAE) \
	         rep_divergence:https://github.com/grantsrb/rep_divergence:$(REPDIV); do \
	  name=$${r%%:*}; rest=$${r#*:}; commit=$${rest##*:}; url=$${rest%:*}; \
	  [ -d $(VENDOR)/$$name/.git ] && continue; \
	  git init -q $(VENDOR)/$$name && git -C $(VENDOR)/$$name fetch -q --depth 1 $$url $$commit && \
	  git -C $(VENDOR)/$$name checkout -q FETCH_HEAD || exit 1; done
