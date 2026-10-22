# Pilot (superseded, 29 September 2026)

The first, preliminary version of AI-05: two hand-built Boolean program templates, four constructed explanations
each, and four target queries per test. It found that restricted single-variable tests accepted the constructed
overclaims and that held-out singles, joint edits and an unmapped-variable control rejected them, with joint
edits adding nothing beyond well-chosen held-out singles. It was first reported as the completed project; that
claim was withdrawn the same day (`../../provenance/STATUS-CORRECTION.md`), and Study 1 replaced it. It is not
used in the paper.

The folder keeps the pilot's original internal layout because its code locates files relative to its own parent
folder (`02-Code/study.py` writes `04-Results/final.json` and reads `06-Study-History/PROTOCOL-FROZEN-2026-09-29.md`).

| Path | Contents |
|---|---|
| `01-Papers/AI-05-REPORT.md`, `.pdf` | The pilot report (corrected status) and its literature record |
| `02-Code/` | `study.py` (standard library only), `verify_records.py`, `build_report_pdf.py`, `requirements.lock` |
| `03-Data/MANIFEST.json` | Input manifest |
| `04-Results/` | `final.json` (all records), independent verification, clean replay |
| `05-Notes/` | Claim–evidence map, reproduction notes, comparator memo |
| `06-Study-History/` | Frozen pilot protocol, source manifest, and `INITIAL-COMPLETION-CLAIM-SUPERSEDED/` (the report, handoff and manifest as they were before the status correction; their relative links were written for the original location) |
