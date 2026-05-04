// Package executor provides a single-threaded cooperative scheduler.
//
// An [Executor] owns a private set of "executor tasks" — goroutines
// created via [Executor.Co] that are invisible to Go's global
// scheduler: they never appear on any P's runqueue, never migrate
// between OS threads, and are not counted by [runtime.NumGoroutine].
//
// All progress on executor tasks happens synchronously, on the
// owner thread, under a [Executor.Pulse] call (with the exception
// of the first slice of a Co call made from the owner thread, which
// runs inline in Co itself).
//
// The owner thread is the OS thread that most recently entered
// Pulse on this Executor; it is sticky between Pulses. Before the
// first Pulse, the owner thread is unestablished and all Co calls
// are treated as cross-thread (queued, run at the next Pulse).
//
// Multiple Executors may coexist; their task sets are disjoint.
//
// # Suspension primitives
//
// Executor tasks suspend and resume via Go's standard
// synchronization primitives — channels, sync.Mutex, sync.Cond, and
// so on. The runtime cooperates by routing parks and wakeups for
// executor-owned goroutines back to their owning Executor instead
// of the global scheduler. The package additionally exposes [Yield]
// for explicit "yield until the next Pulse" behavior.
//
// Parking on a netpoll file descriptor (for example, blocking I/O
// on net.Conn) is NOT supported in this version. A future revision
// is planned.
//
// # Concurrency model
//
// Co is safe to call from any goroutine. Pulse and Yield are
// single-owner: concurrent Pulse calls on the same Executor are
// undefined behavior, and a re-entrant Pulse from inside a task
// driven by an outer Pulse on the same Executor panics.
//
// # Status
//
// Experimental. The API surface is stable per the contract under
// specs/001-runtime-executor/contracts/executor.go but the
// implementation may evolve.
package executor
