# CI merge validation

The reconstructed CI inputs passed fresh validation with actionlint 1.7.12 and
ShellCheck 0.11.0. All ten input SHA-256 values matched the retained manifest
before and after the isolated mutation. This evidence validates workflow
structure and shell behavior; it does not claim a compiler build or a full Go
regression run.

## Input and tool provenance

The fresh run used checkout HEAD
`e82c916ff4d8c2befc6f1891decb0f623401a94b` with uncommitted merge parent
`13a968f350564e739bf700df698a9a635dcdddcd`. The checked bytes are identified by
`recovered-inputs.sha256` and its frozen copy `fresh/validated-inputs.sha256`,
rather than HEAD alone. The manifest covers seven workflows, the actionlint
configuration, and both installer scripts.

The exact validator archives were downloaded again after the environment reset.
Both archive hashes matched retained pins and fresh official GitHub release
metadata. Both extracted executable hashes matched parent-retained values.
`fresh-tool-provenance.json` records the URLs, hashes, verification results, and
observed versions. ShellCheck 0.11.0 is the local validator version; it is not
claimed as a hosted workflow pin.

## Fresh validation results

| Check | Exit | Result |
| --- | ---: | --- |
| Ten original input hashes, before validation | 0 | All matched |
| actionlint with the task-local ShellCheck in PATH | 0 | No diagnostics |
| Direct ShellCheck on both installer scripts | 0 | No diagnostics |
| YAML parsing and Bash syntax | 0 | Eight YAML documents, 67 run blocks, two installer scripts; no failures |
| Masked bootstrap export in an isolated workflow copy | 1 | Exactly one SC2155 diagnostic |
| Ten original input hashes, after mutation | 0 | All matched |
| actionlint on the unchanged originals | 0 | No diagnostics |

`fresh/` contains each exact command, exit status, duration, source parents,
stdout/stderr hashes, and raw stdout/stderr. The mutation copy restores only the
combined bootstrap assignment and export. It never replaces an original file.
`fresh/mutation-normalized.txt` is an additional ANSI-stripped rendering; the
colored raw stream is retained separately.

## Independent shell behavior probes

Root supplied `fresh-shell-probes.json` and `check-shell-semantics.py`:

- A stubbed bootstrap lookup returning 37 was masked by the original export:
  that script exited 0 and printed `BUILD_REACHED`. The corrected assignment
  followed by export exited 37 and never reached that marker.
- The original SIMD word list and corrected Bash array expanded to the same
  17 paths in the same order. Both probes exited 0. The array preserves the
  formatter file inventory.

These are isolated Bash probes. The `go` command in the bootstrap probe is a
shell stub; no Go tool was run.

## Fresh baseline replay

`fresh-baseline-replay/` is a fresh replay of the pre-repair diagnostic
conditions. It is not recovered historical raw output. Five workflow files
come directly from public parent `e82c916f`; the configuration comes from
`9a628f5b`. The SIMD workflow starts from `e82c916f` and receives only the exact
bootstrap block already present in `9a628f5b`. The hash-verified merged main
workflow and both installers are retained. `replay-baseline.py` records that
construction, and `fresh-baseline-replay/inputs.sha256` identifies its bytes.

The replay exited 1 with exactly eleven diagnostics: five SC2155, three SC1007,
two SC2086, and the unrecognized ARM runner label. Raw streams, normalized
diagnostics, and the command/result record are preserved.

## Recovered historical records

`recovered/` contains the five historical command-result JSON objects, the YAML
result body, and the normalized mutation diagnostic copied from returned tool
output. Its README identifies unavailable historical raw streams and the
limits of recovery. Fresh records are kept outside that directory.

## Reproduction and copying

`run-fresh-validation.sh` and `check-yaml-shell.py` are the fresh validation
helpers. They use the task-local tools and existing Python 3.12.14/PyYAML 6.0.3
through offline, cache-disabled uv. Original repository files are read only.

`copy-manifest.sha256` lists the evidence files to preserve, with paths relative
to this directory. It excludes itself. Validator binaries, downloaded archives,
private caches, and build products are outside this evidence directory and are
not part of the copy.
