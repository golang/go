# `runtime/executor` — Single-Threaded Cooperative Scheduler

A new stdlib package, `runtime/executor`, provides cooperative tasks that
are invisible to Go's global scheduler: they never appear on a P's
runqueue, never migrate between OS threads, and are not counted by
`runtime.NumGoroutine()`.

## API

```go
package executor

type Executor struct{ /* opaque */ }

func New() *Executor
func (e *Executor) Co(fn func())                    // safe from any goroutine
func (e *Executor) Pulse(ctx context.Context) error // single-owner; UB if concurrent
func Yield()                                         // executor-task only
```

## Concepts

- **Owner thread** of an `Executor` is the OS thread that most recently
  entered `Pulse` on that instance. Sticky between Pulses; `nil` before
  the first Pulse.
- **`Co(fn)`**: from the owner thread, runs `fn` synchronously inline
  until it suspends or completes. From any other goroutine, `fn` is
  enqueued and runs on the owner thread at the next `Pulse`.
- **`Pulse(ctx)`**: drains all currently-resumable tasks synchronously
  on the calling thread. Returns `ctx.Err()` if the context fires,
  otherwise `nil`.
- **`Yield()`**: suspends the calling task; resumed at the *next*
  `Pulse` (not the current one).
- **Suspension**: standard primitives (channels, `sync.Mutex`,
  `sync.Cond`, …) all work — the runtime routes their `gopark`/`goready`
  back to the owning Executor. Netpoll FDs are **not supported in v1**.

## Threading

- `Pulse` does not internally `LockOSThread`. For tasks to stay pinned
  to a single OS thread across Pulses, the caller must `LockOSThread`
  before any `Co`/`Pulse` and stay locked.
- Concurrent `Pulse` calls on the same Executor are undefined behavior.
  Re-entrant `Pulse` from inside a Pulse-driven task panics.
- Multiple `Executor` instances are independent and may be driven from
  different goroutines on different threads.

## Examples

### Frame loop

```go
import (
    "context"
    "runtime"
    "runtime/executor"
    "time"
)

runtime.LockOSThread()
defer runtime.UnlockOSThread()

ex := executor.New()

for i := 0; i < 3; i++ {
    i := i
    ready := make(chan struct{}, 1)
    ready <- struct{}{}

    ex.Co(func() {
        for tick := 0; ; tick++ {
            <-ready
            update(i, tick)
            go func() { ready <- struct{}{} }()
            executor.Yield()
        }
    })
}

for {
    ctx, cancel := context.WithTimeout(context.Background(), 16*time.Millisecond)
    _ = ex.Pulse(ctx) // returns DeadlineExceeded if work overruns the frame
    cancel()
    render()
}
```

### Cross-thread work submission

```go
runtime.LockOSThread()
defer runtime.UnlockOSThread()
ex := executor.New()
_ = ex.Pulse(context.Background()) // establishes ownership

// Producer on a different goroutine — Co is queued, not inline.
go func() {
    ex.Co(func() {
        // Runs on ex's owner thread at the next Pulse.
    })
}()

// Drive submitted work.
_ = ex.Pulse(context.Background())
```

### Yield-driven cooperative loop

```go
ex.Co(func() {
    for !done() {
        step()
        executor.Yield() // hand back to the Pulse caller; resume on next Pulse
    }
})
```

## Invariants worth knowing

- `runtime.NumGoroutine()` does not count executor tasks.
- An executor task never runs on a goroutine other than its owner
  thread's M (assuming the caller follows the `LockOSThread` convention).
- A panic that escapes the user `fn` propagates to whichever caller is
  driving it (`Co` for inline runs, `Pulse` for resumed runs); the
  executor's bookkeeping remains coherent for other tasks.
- `Co` is safe to call concurrently from many goroutines.

## When to use it

- Game frame loops, deterministic simulation steps, request pumps with a
  latency budget, anything that needs cooperative multitasking
  on a fixed thread without interference from the global scheduler.

## When *not* to use it

- Workloads that need to park on `net.Conn` or other netpoll FDs (v1
  doesn't support those).
- Anything that benefits from being scheduled across multiple cores —
  this is single-threaded by design.
