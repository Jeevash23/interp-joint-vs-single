# The preprint's source

`main.tex`, `sections/` and `references.bib` are the LaTeX source of the preprint *When can a joint intervention
reject an alignment that single interventions accept? From small networks to GPT-2*. Files that differ from the
preprint's source, by the same wording rules as the rest of the release: `main.tex` (a comment on how the statement
was decided x1); `sections/appendix.tex` (machine details x1). The preprint types no study number of its own: every
number is a macro in `generated/numbers.tex`, and every generated table is a file in `generated/`.

`check_numbers.py` checks that these are the two parts' own generated files (`../small-networks/paper/generated/`
for Studies 1–3, `../gpt2/paper/generated/` for Study 4), concatenated and copied unchanged except for one wording
change in Study 4's predictions table ("the companion study" becomes "Study 2"). Each part's `make quick-check`
recomputes its generated files from its saved outputs.

The figures are not shipped as PDFs. `make paper` at the top level draws them from the saved outputs with
`small-networks/paper/make_figures.py` (needs matplotlib) and builds `build/main.pdf` with latexmk.
