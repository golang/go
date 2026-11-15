---
name: logos-concurrency
description: Structure goroutine groups and shared state in loGo/s. Use when the user says go group, goroutine groups, cancellation, Locked, Hot, or data races in .lgo code.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH.
metadata:
  version: "0.22.0"
  milestone: "M8"
---

# loGo/s concurrency

loGo/s provides structured concurrency through goroutine groups and panic-safe synchronization primitives. Goroutines in `.lgo` files must run inside a `go group` block. Shared state is coordinated through `logos/sync.Locked[T]` and `logos/sync.Hot[T]`.

## Procedure

1. Identify concurrency scope: verify whether work can run in parallel and requires context cancellation or worker limits.
2. Structure goroutines with `go group`: define child tasks inside `go group (ctx, limit = N) { ... }` blocks.
3. Propagate errors: use `?` on fallible calls inside child tasks or invoke an error-last child call to yield errors to the group (direct `return` inside a child is rejected with `correctness::return_in_group_child`). Ensure the function enclosing the `go group` returns an `error`.
4. Guard shared state: wrap mutable state with `logos/sync.Locked[T]` for mutex protection or `logos/sync.Hot[T]` for atomic read-mostly pointers.
5. Verify with race detector: run `logos test -race ./...` to detect exercised races; `-race` reports only exercised races and gives no universal data-race freedom. Fix any diagnostics reported by `logos build` or `logos lint`.

## Group syntax

Each row contrasts the loGo/s form with the plain Go pattern it replaces.

| loGo/s | Plain Go |
|---|---|
| `go group (ctx, limit = N) { go { work()? } }` | `g, gctx := rt.NewGroup(ctx, N)` followed by `g.Go(func(ctx context.Context) error { return work() })` and `g.Wait()` |
| `go group (ctx) { go { work()? } }` | `g, gctx := rt.NewGroup(ctx, 0)` followed by `g.Go(func(ctx context.Context) error { return work() })` and `g.Wait()` |

## Group semantics

The compiler lowers `go group` blocks to `logos/rt.Group`. The following invariants govern group execution:

- Cancellation: the group context cancels upon the first child error or panic.
- Joining: the group statement joins all child goroutines before proceeding. The enclosing function propagates any group error like `?`.
- Concurrency limit: when `limit = N` is specified, an internal semaphore bounds concurrent execution to at most N child goroutines.
- Child block contract: using `?` inside a child returns the error directly to the group.
- Disallowed return: a bare `return` inside a child block is rejected with `correctness::return_in_group_child`.
- Channel operations: direct channel send, receive, and range operations inside a child block automatically select on context cancellation.
- Netpoll integration: children blocked on netpoll-backed network I/O wake within 100 ms of group cancellation with a timeout error.
- Panic handling: a panicking child goroutine terminates the group and wraps the panic value and stack into `*logos/rt.PanicError`.

## Shared state

The `logos/sync` package provides concurrency containers:

- `Locked[T]`: pairs a value of type `T` with a mutex. Call `l.WithLock(func(v *T) { ... })` to access the value. `WithLock` acquires the lock, executes the callback, and guarantees unlock on return or panic. Callbacks must not return or leak the pointer `*T`.
- `Hot[T]`: holds a read-mostly pointer managed with atomic pointer swaps. `Load()` returns a read-only pointer `const *T`. `Store(*T)` publishes a new pointer. `Update(func(old const *T) *T)` applies updates inside a compare-and-swap loop without maintaining a retire list.

## Diagnostics

| Rule | Severity | Fix |
|---|---|---|
| `correctness::return_in_group_child` | forbid | Remove return statement; use ? or expression outcome to yield an error to the group. |
| `correctness::child_outside_group` | forbid | Move the child go block inside an enclosing go group statement. |
| `correctness::group_needs_error_result` | forbid | Add an error return type to the function enclosing the go group statement. |
| `concurrency::unsupervised_goroutine` | forbid | Replace bare go f() with a managed child block inside go group. |
| `concurrency::reentrant_withlock_deadlock` | forbid | Do not invoke WithLock on the same Locked receiver inside a WithLock callback. |
| `concurrency::concurrent_multiword_race` | forbid | Protect multi-word structures accessed across goroutines with Locked or Hot. |
