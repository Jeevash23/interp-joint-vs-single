# Third-party repositories (not included)

Study 1 ran with clones of three repositories in this folder. They are not part of the release; `make vendor`
(from the release root) fetches each at the commit the study recorded:

| Folder | Repository | Commit | Licence |
|---|---|---|---|
| `tracr` | `https://github.com/google-deepmind/tracr` | `9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd` | Apache 2.0 |
| `cae` | `https://github.com/MelouxM/CAE` | `d91f65b5620423d3eae98478bc9843f4a40e361a` | MIT |
| `rep_divergence` | `https://github.com/grantsrb/rep_divergence` | `f2548d2ea9b4f4b87a87ba5d53db43838d15c521` | Apache 2.0 |

The study's environment manifest (`../06-Study-History/V2-SOURCE-ENVIRONMENT-MANIFEST.json`) records the Tracr and
CAE commits and that the clones had no local changes. Tracr and CAE are also installed into the environment by
`make setup`; the clones are needed only by `posthoc/cae_mediator.py`, `02-Code/reproduce_official_v2.py` and
`02-Code/v2_official_grant.py`.
