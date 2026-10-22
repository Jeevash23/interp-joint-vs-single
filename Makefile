# Joint and single interchange tests, from small networks to GPT-2: code release of one paper in two parts.
# small-networks/ holds Studies 1-3 and gpt2/ holds Study 4; each is a complete release with its own Makefile,
# environment notes, freeze checks and checksums. `make help` lists the targets.
# PYTHON_SMALL and PYTHON_GPT2 are the Pythons of the two parts (by default each part's own .venv, made by
# `make setup`); PYTHON, if given, is used for both. Paths are taken from the top folder.
PYTHON ?=
PYTHON_SMALL ?= $(if $(PYTHON),$(PYTHON),small-networks/.venv/bin/python)
PYTHON_GPT2 ?= $(if $(PYTHON),$(PYTHON),gpt2/.venv/bin/python)
PY_SMALL := $(if $(findstring /,$(PYTHON_SMALL)),$(abspath $(PYTHON_SMALL)),$(PYTHON_SMALL))
PY_GPT2 := $(if $(findstring /,$(PYTHON_GPT2)),$(abspath $(PYTHON_GPT2)),$(PYTHON_GPT2))
export PYTHONDONTWRITEBYTECODE := 1
# Variables given on this command line are not passed on to the parts; each part gets its own PYTHON below.
MAKEOVERRIDES :=
.NOTPARALLEL:

.PHONY: help setup checksums check-freezes quick-check paper

help:
	@echo "make setup          create each part's environment (downloads packages; see each part's ENVIRONMENT.md)"
	@echo "make checksums      every file listed in SHA256SUMS, here and in both parts"
	@echo "make check-freezes  both parts' freeze records against the files they name"
	@echo "make quick-check    CPU, a few minutes, no model and no network: checksums, both parts' quick-checks"
	@echo "                    (freeze records, every number and table recomputed from the saved outputs), and the"
	@echo "                    paper's generated files against the two parts"
	@echo "make paper          the figures (small-networks, needs matplotlib), then paper/build/main.pdf (latexmk),"
	@echo "                    when the manuscript's LaTeX source is in paper/"

setup:
	$(MAKE) -C small-networks setup
	$(MAKE) -C gpt2 setup

checksums:
	$(PY_SMALL) tools/check_sums.py

check-freezes:
	$(MAKE) -C small-networks check-freezes PYTHON=$(PY_SMALL)
	$(MAKE) -C gpt2 check-freezes PYTHON=$(PY_GPT2)

quick-check: checksums
	$(MAKE) -C small-networks quick-check PYTHON=$(PY_SMALL)
	$(MAKE) -C gpt2 quick-check PYTHON=$(PY_GPT2)
	$(PY_SMALL) paper/check_numbers.py

paper:
	@test -f paper/main.tex || { echo "paper/main.tex is missing: the manuscript's LaTeX source is not in this folder"; exit 1; }
	$(MAKE) -C small-networks figures PYTHON=$(PY_SMALL)
	mkdir -p paper/figures
	cp small-networks/paper/figures/fig-compiled-panels.pdf small-networks/paper/figures/fig-trained.pdf paper/figures/
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
