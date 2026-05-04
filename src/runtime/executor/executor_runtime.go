package executor

import (
	"internal/race"
	"unsafe"
	_ "unsafe" // for go:linkname
)

// Imports of runtime helpers via //go:linkname. The runtime side
// (src/runtime/exec.go) publishes each helper with a self-linkname
// directive. All cross-package values are passed as unsafe.Pointer
// so the executor package never references runtime's *g or *m
// types directly.
//
// For the synchronous-switch primitives (newcoro / coroswitch) we
// follow the same trick the iter package uses: declare a local
// opaque coro type and linkname the runtime functions directly.
// Heap-allocated closures are forbidden inside the runtime
// package, so the closure that wraps a Co'd user function (with
// its panic-recovery defer chain) is constructed here, where heap
// closures are fine, and passed through newcoro.

type coro struct{}

//go:linkname runtime_newcoro runtime.newcoro
func runtime_newcoro(func(*coro)) *coro

//go:linknamestd runtime_coroswitch runtime.coroswitch
func runtime_coroswitch(*coro)

//go:linkname runtime_execSetOwner runtime.execSetOwner
func runtime_execSetOwner(gp unsafe.Pointer, owner unsafe.Pointer)

//go:linkname runtime_execGetOwner runtime.execGetOwner
func runtime_execGetOwner(gp unsafe.Pointer) unsafe.Pointer

//go:linkname runtime_execCoroG runtime.execCoroG
func runtime_execCoroG(c unsafe.Pointer) unsafe.Pointer

//go:linkname runtime_execCurg runtime.execCurg
func runtime_execCurg() unsafe.Pointer

//go:linkname runtime_execCurM runtime.execCurM
func runtime_execCurM() unsafe.Pointer

//go:linkname runtime_execAddNExec runtime.execAddNExec
func runtime_execAddNExec(delta int64)

//go:linkname runtime_execInstallHooks runtime.execInstallHooks
func runtime_execInstallHooks(park, ready func(gp unsafe.Pointer))

// init installs the runtime hooks. Called exactly once per process.
func init() {
	runtime_execInstallHooks(parkHook, readyHook)
}

// task is the per-executor-task record. A pointer to one is stored
// in the corresponding goroutine's g.execOwner field (set via
// runtime_execSetOwner). The runtime hooks recover the *task by
// reading g.execOwner.
type task struct {
	owner *Executor

	// c is the local-typed *coro produced by runtime_newcoro. We
	// coroswitch on this to enter/leave the task.
	c *coro

	// g is the underlying *runtime.g (cached after creation),
	// stored as an opaque pointer.
	g unsafe.Pointer

	// next links the task into one of the Executor's intrusive
	// lists (runnable, parked, or yielded). Owner-thread access
	// only; non-owner wakeups go through the lock-free wakeQ in
	// mpsc.go and merge into runnable on the owner thread.
	next *task

	// state records which list the task is currently on. Owner-
	// thread access only.
	state taskState

	// panicVal, if non-nil, is a panic captured by the deferred
	// recover wrapping the user fn. The driver re-raises it on its
	// own stack after the task switches back. Owner-thread access
	// only.
	panicVal any

	// done is set true after the user fn (and any deferred
	// panic-handling) completes, just before the coro exits.
	done bool
}

type taskState uint8

const (
	stateRunnable taskState = iota
	stateParked
	stateYielded
	stateRunning
	stateDone
)

// parkHook is invoked from gopark when the running G is an executor
// task. The runtime has already executed the unlock callback; we
// only need to record the task as parked and switch back to its
// driver via coroswitch.
//
// parkHook runs on the executor task's goroutine (the parking g).
func parkHook(gp unsafe.Pointer) {
	t := (*task)(runtime_execGetOwner(gp))
	if t == nil {
		// Should not happen — gopark already verified gp.execOwner
		// non-nil, but stay defensive.
		return
	}
	e := t.owner
	// The task is currently 'running'. Move it onto the parked list.
	t.state = stateParked
	t.next = e.parkedHead
	e.parkedHead = t
	// Switch back to the driver. Control returns here when the
	// driver next coroswitches into this task (i.e. after a goready
	// makes us runnable and Pulse picks us up).
	race.Release(unsafe.Pointer(&t.c))
	runtime_coroswitch(t.c)
	race.Acquire(unsafe.Pointer(&t.c))
	// Resumed by the driver. Restore "running" state. The task is
	// no longer on parked or runnable; the driver detached it.
	t.state = stateRunning
}

// readyHook is invoked from goready when the woken G is an executor
// task. It may be called from any OS thread, not just the owner.
// We push the task onto the lock-free wakeQ; the owner thread's
// next drive-loop iteration will merge it into runnable.
//
// readyHook does not yet remove t from the parked list — that
// happens lazily on the owner thread when the wakeQ entry is
// processed, because parked-list mutation is owner-thread-only.
func readyHook(gp unsafe.Pointer) {
	t := (*task)(runtime_execGetOwner(gp))
	if t == nil {
		return
	}
	w := &wakeEntry{t: t}
	pushWake(t.owner, w)
}
