# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `study0-pilot.tar.xz` | 3 | 0.0 | 0.1 | the superseded pilot (not used in the manuscript): its final record and verification |
| `study1-compiled-programs.tar.xz` | 180 | 4.1 | 16.2 | Study 1: the eight frozen panels (per-program query records), analyses, independent verification, clean-environment replay, official reproductions, and the post hoc census |
| `study2-trained-networks.tar.xz` | 906 | 20.5 | 24.1 | Study 2: per-network task, training record and census of every single and joint interchange (232 networks), the frozen analysis, verification, post hoc results and the saved held-out refits (trained weights and fitted alignments not included) |
| `study3-multi-site.tar.xz` | 705 | 8.8 | 9.7 | Study 3: the same for its 232 networks, with its margin analysis and post hoc results (trained weights and fitted alignments not included) |

Total packed size: 33.5 MB. Rendered figures are not included; `make figures` draws them from these outputs.
