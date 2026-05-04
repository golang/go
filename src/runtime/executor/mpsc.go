package executor

// submitEntry is one element of the lock-free MPSC queue used to
// receive cross-thread Co submissions. Producers (any goroutine on
// any M) atomically CAS-prepend a fresh entry. The single consumer
// (the owner thread, at the start of every Pulse) atomically swaps
// the head to nil and reverses the resulting list to recover FIFO
// submission order.
type submitEntry struct {
	fn   func()
	next *submitEntry
}

// pushSubmit atomically prepends a new submission carrying fn onto
// e.submitQ.
func pushSubmit(e *Executor, fn func()) {
	entry := &submitEntry{fn: fn}
	for {
		head := e.submitQ.Load()
		entry.next = head
		if e.submitQ.CompareAndSwap(head, entry) {
			return
		}
	}
}

// drainSubmit atomically swaps e.submitQ's head with nil and
// returns the queued submissions in FIFO submission order. Returns
// nil if the queue was empty. Owner-thread only.
func drainSubmit(e *Executor) *submitEntry {
	head := e.submitQ.Swap(nil)
	if head == nil {
		return nil
	}
	var prev *submitEntry
	for cur := head; cur != nil; {
		next := cur.next
		cur.next = prev
		prev = cur
		cur = next
	}
	return prev
}

// wakeEntry is one element of the lock-free MPSC wake queue used
// when goready is invoked on an executor task from any M. The
// owner thread drains this queue at each iteration of the drive
// loop and merges entries into the runnable list.
type wakeEntry struct {
	t    *task
	next *wakeEntry
}

// pushWake atomically prepends w onto e.wakeQ.
func pushWake(e *Executor, w *wakeEntry) {
	for {
		head := e.wakeQ.Load()
		w.next = head
		if e.wakeQ.CompareAndSwap(head, w) {
			return
		}
	}
}

// drainWake atomically swaps e.wakeQ's head with nil and returns
// the wake entries in FIFO order. Owner-thread only.
func drainWake(e *Executor) *wakeEntry {
	head := e.wakeQ.Swap(nil)
	if head == nil {
		return nil
	}
	var prev *wakeEntry
	for cur := head; cur != nil; {
		next := cur.next
		cur.next = prev
		prev = cur
		cur = next
	}
	return prev
}
