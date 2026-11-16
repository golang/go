# Initial differential corpus evidence

This bundle preserves three completed initial campaigns. It records failures as
well as successful cells. Final compiler repairs, permission follow-ups, and
runtime checks have their own evidence; this archive does not claim that the
final toolchain passed this entire matrix.

## Recorded outcomes

| Initial campaign | Cases | Stock Go | Fork Go | LoGOs build | Lint report outcomes |
|---|---:|---|---|---|---|
| Default policy, candidate3 | 26 | 25 pass; 1 missing prerequisite | 24 pass; 1 compiler error; 1 missing prerequisite | 16 pass; 9 capability denials; 1 blocked | 16 findings; 1 clean; 8 pre-lint capability denials; 1 blocked |
| Exact-package permissions, candidate3 | 10 | 9 pass; 1 upstream-incompatible | 8 pass; 1 compiler error; 1 uncompared | 5 pass; 4 failed; 1 uncompared | 8 findings; 1 compiler error; 1 ordinary compiler-error report |
| gglint, basic/standard/strict | 26 | Host oracle metadata only | not selected | not selected | 53 findings; 3 clean; 22 load-failed, across 78 cells |

The default corpus samples **24 external sources across 26 configurations**.
The permission campaign repeats ten of those sources with explicit reviewed
capability grants. It adds no new upstream sources. Origins cover GitHub,
GitLab, Codeberg, SourceHut, and go.googlesource.com. The source manifest records
selection cohorts, package patterns, cgo settings, build tags, and exact pins.

Every compiler matrix command here is a **build**. These three campaigns
executed **zero upstream test commands and zero upstream test functions**.
The separate harness verification logs record 35 runner unit tests. They do
not establish upstream runtime coverage. Runtime and final compiler evidence
must be assessed separately.

A valid lint report establishes its shape, exit contract, and repeated result.
It does not establish that its findings are true positives. Partial findings
from load-failed cells are not counted as complete lint coverage.

## Failures retained

- `golang.org/x/tools/cmd/digraph` triggered the candidate3 internal compiler
  error about one implicit type parameter. The permission replay retained it
  in fork Go, LoGOs build, and native lint.
- The permitted modernc SQLite build reached compilation and its compiler
  subprocess was killed. Native lint completed with 1,566 primary findings and
  a matching repeat. The killed build remains failed.
- errorlint and exhaustive reached linker checks after their unsafe grants and
  were denied process-launch capability. The later reviewed linker policies
  are included only as pending inputs, not as completed replay results.
- decorder initially lacked two cached dependencies. After their download,
  its pinned `golang.org/x/tools` v0.20.0 dependency failed the same internal
  layout assertion with stock Go and both candidates. This is an
  upstream-incompatible baseline. Its ordinary native compiler JSON has an
  intentionally empty `rule_id`; the archived older runner recorded that as
  `invalid-report`. The later validated classifier uses `load-failed` for this
  shape. Historical report fields are unchanged.
- Broad gglint retained 22 incomplete cells, including missing dependencies and
  missing vetx imports. Their per-case and per-mode statuses are in
  [summary.json](summary.json), with complete raw evidence in the archive.

No result from the interrupted candidate2 `corpus-compiler-final` campaign is
included or added to these counts.

## Archive and provenance

[initial-corpus.tar.gz](initial-corpus.tar.gz) contains 723 regular
files, totaling 12,367,885 content bytes before compression. Its SHA-256 is:

```
db8283ae871b509a5924e495b3b417531a30118af159bd2088c982961808526d
```

The archive contains:

- `runs/<campaign>/`: original report JSON, stdout, stderr, generated configs,
  saved policy inputs, and tool metadata. Commands retain their original
  absolute paths; the corresponding archived log is under this campaign
  prefix with the same relative path.
- `inputs/<campaign>/`: runner, manifest, and policy snapshots obtained from
  Git and matched byte-for-byte to the SHA-256 values recorded by that run.
- `inputs/runner-validation-35/`: the later strict runner, its tests, exact
  captured compiler-error record, and pending linker-policy inputs at commit
  `a954f0bcd7d8da06733bbf651a9b38da32a705db`. These files were not used by the
  earlier initial campaigns.
- `metadata/`: dependency download metadata, native host/compiler/libc facts,
  source verification and runner test logs, and script README RED/regeneration/
  GREEN logs. Dependency repairs do not alter earlier outcomes.
- `input-snapshots.json`, `campaign-summary.json`, and `SHA256SUMS`: provenance,
  concise results, and hashes for every payload file except `SHA256SUMS` itself.

The package excludes work directories, caches, source checkouts, and binaries.
Tool binaries are identified by the hashes captured during each original run;
packaging did not inspect the later partially rebuilt toolchain.

Archive members are sorted, ownership and timestamps are zeroed, file modes
are 0644, and gzip has no filename or timestamp. Two independent constructions
produced identical bytes. Every member and 642 recorded log hashes
were checked before the archive was written.

## Verify

From this directory:

```sh
sha256sum -c files.sha256
mkdir unpacked
cd unpacked
tar -xzf ../initial-corpus.tar.gz
sha256sum -c SHA256SUMS
```

[raw-files.sha256](raw-files.sha256) is the external copy of the archive's
payload hash manifest. [summary.json](summary.json) keeps source identities,
per-stage and per-mode outcomes, valid-report counts, and pending follow-ups
machine-readable. Later reruns must retain their own source, tool, input, and
policy identities; they must not replace these initial records.
