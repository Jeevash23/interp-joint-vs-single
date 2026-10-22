# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `companion-study2-networks.tar.xz` | 928 | 36.0 | 36.9 | the companion study's Study 2 networks that arm d loads: for each of its 32 development and 200 final networks, the trained weights, the census of every interchange decision and the task and training records (byte copies) |
| `study-results.tar.xz` | 162 | 0.3 | 1.9 | the study's records and analyses: per-task JSON records of the development and confirmation runs of GPT-2 Small, the cloud run of GPT-2 Medium, arm d on the companion study's networks, the gates and settings, and reduced copies of the 43 `arm-a.npz` arrays (the four arrays the number script reads) |

Total packed size: 36.3 MB. Rendered figures are not included; `make figures` draws them from these outputs.
