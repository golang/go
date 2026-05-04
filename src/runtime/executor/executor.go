package executor

import (
	"context"
	"internal/race"
	"sync/atomic"
	"unsafe"
)

// Executor is a single-threaded cooperative scheduler. See the
// package documentation for the complete behavior contract.
//
// The zero value is not usable; obtain instances from [New]. An
// Executor must not be copied after first use.
type Executor struct {
	_ noCopy

	// runnableHead, runnableTail are owner-thread-access linked
	// lists of tasks ready to run on the next drive-loop iteration.
	// Pulse pops from runnableHead; submitQ-derived and
	// wakeQ-derived tasks are appended at runnableTail.
	runnableHead *task
	runnableTail *task

	// parkedHead is the owner-thread-access linked list of tasks
	// currently parked (suspended on a standard primitive). Tasks
	// move between parkedHead and runnableHead via the wakeQ when
	// goready fires from any M.
	parkedHead *task

	// yieldedHead is the owner-thread-access linked list of tasks
	// that called Yield(). They are NOT picked up by the current
	// Pulse; the drive routine appends them to runnable only after
	// the current Pulse returns and a subsequent Pulse begins.
	// (FR-013a: a yielded task becomes resumable at the next Pulse.)
	yieldedHead *task

	// wakeQ is the lock-free MPSC queue onto which non-owner
	// (and owner) goroutines push wake events when goready fires
	// for one of this Executor's tasks.
	wakeQ atomic.Pointer[wakeEntry]

	// submitQ is the lock-free MPSC queue for cross-thread Co
	// submissions.
	submitQ atomic.Pointer[submitEntry]

	// ownerM is the M (OS thread) that most recently drove Pulse on
	// this Executor. Sticky; nil before first Pulse. Used to detect
	// owner-thread Co calls.
	ownerM atomic.Pointer[byte]

	// inPulse is the re-entrancy flag.
	inPulse uint32
}

// New returns a fresh Executor with no tasks.
func New() *Executor {
	return &Executor{}
}

// Co submits fn for execution as an executor task. See the package
// documentation for the full contract.
func (e *Executor) Co(fn func()) {
	if fn == nil {
		panic("runtime/executor: Co called with nil function")
	}
	// Owner-thread check: if we are currently on the same M as the
	// most recent Pulse, run fn inline. Otherwise enqueue.
	owner := e.ownerM.Load()
	if owner != nil && (*byte)(runtime_execCurM()) == owner {
		e.runInline(fn)
		return
	}
	pushSubmit(e, fn)
}

// runInline creates a task for fn and synchronously coroswitches
// into it. The caller must be on the owner thread. Returns when fn
// completes or first suspends.
func (e *Executor) runInline(fn func()) {
	t := e.newTask(fn)
	t.state = stateRunning
	// race.Release/Acquire around coroswitch so the race detector
	// sees the implicit happens-before that coroswitch establishes
	// (control transfer is single-threaded on the M but the detector
	// tracks goroutine identity). Same pattern iter.Pull uses.
	race.Release(unsafe.Pointer(&t.c))
	runtime_coroswitch(t.c)
	race.Acquire(unsafe.Pointer(&t.c))
	if t.panicVal != nil {
		v := t.panicVal
		t.panicVal = nil
		panic(v)
	}
}

// newTask allocates a task record, creates its coro (whose body
// runs fn under a deferred recover that captures any panic into
// t.panicVal and an exit hook that decrements the runtime's
// executor count), stamps the task pointer onto the underlying
// g.execOwner field, and bumps the executor goroutine counter.
func (e *Executor) newTask(fn func()) *task {
	t := &task{owner: e}
	// The closure below is constructed in runtime/executor (not in
	// runtime), where heap-allocated closures are permitted. It
	// captures t and fn so the runtime side does not need to.
	//
	// race.Acquire on entry and race.Release on exit (deferred to
	// run last, after the panic-capture and exit hooks) bracket the
	// task body so the race detector sees the implicit happens-
	// before relationship that coroswitch establishes between the
	// task and its driver. Without these, fields like t.done and
	// t.panicVal would appear racy to -race even though they are
	// safe (synchronized by OS-thread serialization through the M).
	body := func(c *coro) {
		race.Acquire(unsafe.Pointer(&t.c))
		defer race.Release(unsafe.Pointer(&t.c))
		defer func() {
			t.done = true
			runtime_execAddNExec(-1)
		}()
		defer func() {
			if r := recover(); r != nil {
				t.panicVal = r
			}
		}()
		fn()
	}
	t.c = runtime_newcoro(body)
	t.g = runtime_execCoroG(unsafe.Pointer(t.c))
	runtime_execSetOwner(t.g, unsafe.Pointer(t))
	runtime_execAddNExec(+1)
	return t
}

// Pulse drives currently-resumable executor tasks forward
// synchronously on the calling OS thread. See the package
// documentation for the full contract.
func (e *Executor) Pulse(ctx context.Context) error {
	if !atomic.CompareAndSwapUint32(&e.inPulse, 0, 1) {
		// Re-entrant Pulse on the same instance (FR-016).
		panic("runtime/executor: re-entrant Pulse")
	}
	defer atomic.StoreUint32(&e.inPulse, 0)

	// Record the M this Pulse runs on. This is used by Co's
	// owner-thread detection. We deliberately do NOT call
	// LockOSThread here: doing so would change the goroutine's
	// lockedExt counter relative to its state when an executor
	// task's coro was created, and the runtime's coro mechanism
	// requires the lock state to be invariant across the coro's
	// lifetime. Callers who want strict thread-pinning of executor
	// tasks should LockOSThread themselves before any Co/Pulse on
	// this Executor and remain locked until they are done.
	e.ownerM.Store((*byte)(runtime_execCurM()))

	// FR-003 fast-path: if ctx is already done, return immediately.
	if err := ctx.Err(); err != nil {
		return err
	}

	// Promote any tasks that yielded during a previous Pulse onto
	// runnable. They were intentionally held back so that within a
	// single Pulse a yielded task does not loop instantly.
	e.appendYieldedToRunnable()

	// Drain submitQ into runnable: each pending Co produces a fresh
	// task, runnable from this Pulse onward.
	e.drainSubmitToRunnable()

	// Drain wakeQ once before entering the drive loop, then again
	// after each task slice (to catch wakeups that fired during the
	// slice from another M).
	e.drainWakeToRunnable()

	for {
		// Context check between resumes (FR-008).
		if err := ctx.Err(); err != nil {
			return err
		}
		t := e.popRunnable()
		if t == nil {
			// Nothing currently runnable. Honor FR-015: return nil.
			// We deliberately do not block awaiting a wakeup; that
			// is the caller's responsibility (typically a frame loop).
			return nil
		}
		t.state = stateRunning
		race.Release(unsafe.Pointer(&t.c))
		runtime_coroswitch(t.c)
		race.Acquire(unsafe.Pointer(&t.c))
		// The task either suspended (parkHook routed it to parked or
		// yieldedHead) or completed (state will be stateDone via the
		// onExit callback). If it panicked, re-raise on the driver.
		if t.panicVal != nil {
			v := t.panicVal
			t.panicVal = nil
			panic(v)
		}
		// Merge any wakeups that arrived during the slice.
		e.drainWakeToRunnable()
	}
}

// Yield suspends the calling executor task until the next Pulse on
// the owning Executor.
func Yield() {
	gp := runtime_execCurg()
	owner := runtime_execGetOwner(gp)
	if owner == nil {
		panic("runtime/executor: Yield called outside an executor task")
	}
	t := (*task)(owner)
	// Park onto the yielded list (NOT runnable) so the current
	// Pulse does not pick us up again before returning. The next
	// Pulse will promote yieldedHead to runnable at entry.
	t.state = stateYielded
	t.next = t.owner.yieldedHead
	t.owner.yieldedHead = t
	// Switch back to the driver. Control returns here when a
	// future Pulse selects this task from runnable and switches in.
	race.Release(unsafe.Pointer(&t.c))
	runtime_coroswitch(t.c)
	race.Acquire(unsafe.Pointer(&t.c))
	t.state = stateRunning
}

// popRunnable removes and returns the head of the runnable list,
// or nil if empty. Owner-thread only.
func (e *Executor) popRunnable() *task {
	t := e.runnableHead
	if t == nil {
		return nil
	}
	e.runnableHead = t.next
	if e.runnableHead == nil {
		e.runnableTail = nil
	}
	t.next = nil
	return t
}

// appendRunnable appends t at the tail of runnable. Owner-thread only.
func (e *Executor) appendRunnable(t *task) {
	t.next = nil
	t.state = stateRunnable
	if e.runnableTail == nil {
		e.runnableHead = t
		e.runnableTail = t
		return
	}
	e.runnableTail.next = t
	e.runnableTail = t
}

// appendYieldedToRunnable transfers the entire yieldedHead list onto
// runnable in FIFO-of-yields order. Yields were prepended to
// yieldedHead, so iterate and append.
func (e *Executor) appendYieldedToRunnable() {
	t := e.yieldedHead
	e.yieldedHead = nil
	for t != nil {
		next := t.next
		e.appendRunnable(t)
		t = next
	}
}

// drainSubmitToRunnable consumes the submitQ, materializing a task
// per submission and appending it to runnable.
func (e *Executor) drainSubmitToRunnable() {
	s := drainSubmit(e)
	for s != nil {
		next := s.next
		t := e.newTask(s.fn)
		e.appendRunnable(t)
		s = next
	}
}

// drainWakeToRunnable consumes the wakeQ, removing each task from
// the parked list and appending to runnable. Owner-thread only.
func (e *Executor) drainWakeToRunnable() {
	w := drainWake(e)
	for w != nil {
		next := w.next
		t := w.t
		// Remove t from parkedHead (linear scan; the list is
		// expected small for typical executor workloads).
		e.unlinkFromParked(t)
		e.appendRunnable(t)
		w = next
	}
}

// unlinkFromParked removes t from e.parkedHead. No-op if not on the
// list (defensive — could happen if a wake fires twice).
func (e *Executor) unlinkFromParked(t *task) {
	if t.state != stateParked {
		// Already moved (e.g. duplicate wake).
		return
	}
	if e.parkedHead == t {
		e.parkedHead = t.next
		t.next = nil
		return
	}
	for p := e.parkedHead; p != nil; p = p.next {
		if p.next == t {
			p.next = t.next
			t.next = nil
			return
		}
	}
	// Not found — should not happen, but stay defensive.
}

// noCopy is a vet-only marker that prevents accidental copying of
// an Executor value (cf. sync.Mutex, strings.Builder).
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}
