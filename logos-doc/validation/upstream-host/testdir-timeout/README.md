# Separate the complete compiler-fixture partition

Hosted run [43951103189](https://github.com/gosuda/LoGOs/actions/runs/43951103189)
recorded 666 unfiltered dist names at source commit
`9909feb26fe92b124f17c11fdf0022be7a643188`. The literal `cmd/go` suite passed in
1533.587 seconds. The complement failed existing SIMD assertions and
`cmd/internal/testdir` reached its 600.050-second package deadline. Both original
exits, complete logs, inventory, and partition receipt are retained unchanged in
`evidence/`. The original helper, tests, and workflow from base
`ed3b743476c4665e89f18f6c2f3d090bbf3610b8` are retained in `old/`.

The new contract has three disjoint groups whose union is that same unfiltered
inventory: exactly `cmd/go`, every canonical numbered
`cmd/internal/testdir:I_N` shard, and the complete complement. All shard names
must share a positive count N and cover exactly indices 0 through N-1. The
observed count is checked before any count-based allocation. Canonical decimal
spelling matches dist's `%d_%d` producer. Prefixes, suffixes, literal package
names, other variants, and noncanonical numbers remain in the complement; no
test disappears because its name resembles a selected shard.

The frozen inventory contains one shard, `cmd/internal/testdir:0_1`, leaving 664
complement names. `src/cmd/dist/test.go` registers the shards without an explicit
package timeout; its scale-1 default is 10 minutes. The testdir runner queues its
fixtures with `t.Parallel`. The timeout stack has 37 waiting Parallel frames and
four active fixture ages of 1, 17, 37, and 76 seconds. The separately selected
LoGOs fixture suite passed in 355.656 seconds. These observations justify a
bounded aggregate-capacity retry: testdir scale 2 gives a 20-minute package
deadline within a 25-minute step. The complement stays at scale 1, `cmd/go` stays
at scale 4, and the job keeps its 90-minute limit. No fixture assertion or Go
source changes, and a repeated timeout remains a failure.

The new workflow must bind each actual `run.bash -run` expression to the same
canonical grammar used in its receipt. All three suites run independently after
a successful inventory, including after an ordinary sibling failure. A new
upstream-only receipt boundary records both installed drivers, compile/link/asm/
cgo, exact HEAD and src tree, and clean source status before and after all three
partitions. It uses an explicit repository root even though the steps run from
`src`. Cancellation or build failure does not invoke the after check. These
installed-tool receipts did not exist in the old run and are not inferred from
its logs.

`frozen.json` pins the original evidence and the unchanged native inputs. The
older `../frozen.json` remains the historical two-way plan and keeps its Wasm and
Go source pins unchanged. This slice supersedes only its partition plan. New
tests are frozen before implementation; local checks use Python, Bash, and
temporary Git repositories with inert tool files. No Go command runs locally.
Hosted three-way execution and installed-tool stability remain pending.
