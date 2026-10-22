# Changes from the original project

This release was exported from the project folder in which the studies ran. Files are copied byte for byte except
those listed below. There are two kinds of change:

- **redacted:** records in which personal or machine-specific strings were replaced: absolute paths and other
  projects' folder names by placeholders (`<project>`, `<portfolio>`, `<portfolio-root>`, `<workspace>`,
  `<research>`, `<sibling-project>`, `<uv-data>`, `<home>`), the operating system's name by `[distro]`, local
  timestamps by the same instant in UTC, and the names of the writing and coding tools in working records by the placeholder `[redacted]` (the assistants used are named in the top-level README.md, under "Use of AI assistants"). Nothing
  else in these files changed; the counts show how many strings of each kind were replaced.
- **not shipped:** files named in a freeze manifest but left out (listed at the end with their SHA-256).

Inputs larger than 1 MB are shipped compressed (`.xz`) and saved outputs as archives under `data/saved-outputs/`;
`tools/unpack.py` restores them byte for byte and checks each SHA-256 against `data/PACKED-FILES.json`.
Compression is not a change and is not listed below.

To check every freeze manifest against this release: `make check-freezes`.

## Files that differ from the project originals

| File | Change | Original SHA-256 | Frozen in |
|---|---|---|---|
| `studies/0-pilot/04-Results/final.json` | redacted (platform string x1); shipped inside `data/saved-outputs/study0-pilot.tar.xz` | `cccfe6bdd02ac19a5931a66b0d69cb0cd317855777ba4c65ad29216d1f361eaf` | - |
| `studies/0-pilot/06-Study-History/SOURCE-ENVIRONMENT-MANIFEST.json` | redacted (AI tool x1, platform string x1) | `b8576ec58df822dc015295fa71d4ca971be6b8b4bd09b74dcf3e562587009edf` | - |
| `studies/1-compiled-programs/02-Code/requirements-v2.lock` | redacted (absolute path x2) | `01b2e4e5e2de3e23edfc2dce00932560ab7effc90b2c93e3b010a44a946b124d` | - |
| `studies/1-compiled-programs/04-Results/v2/clean-environment-replay.json` | redacted (absolute path x1); shipped inside `data/saved-outputs/study1-compiled-programs.tar.xz` | `21a9237fe2962d5f8c562a76ec7d85ae0a0ce4ae92e335d578c70adae319dd79` | - |
| `studies/1-compiled-programs/06-Study-History/V2-SOURCE-ENVIRONMENT-MANIFEST.json` | redacted (absolute path x1, platform string x1) | `1af178d9de2430afb7579d6c95bbff38ebeeab841191ee90cf6416b6e50a5e54` | - |
