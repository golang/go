// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"strings"

	"internal/trace/v2/event"
	"internal/trace/v2/event/go122"
	"internal/trace/v2/version"
)

// ordering emulates Go scheduler state for both validation and
// for putting events in the right order.
//
// The interface to ordering consists of two methods: Advance
// and Next. Advance is called to try and advance an event and
// add completed events to the ordering. Next is used to pick
// off events in the ordering.
type ordering struct {
	gStates     map[GoID]*gState
	pStates     map[ProcID]*pState // TODO: The keys are dense, so this can be a slice.
	mStates     map[ThreadID]*mState
	activeTasks map[TaskID]taskState
	gcSeq       uint64
	gcState     gcState
	initialGen  uint64
	queue       queue[Event]
}

// Advance checks if it's valid to proceed with ev which came from thread m.
//
// It assumes the gen value passed to it is monotonically increasing across calls.
//
// If any error is returned, then the trace is broken and trace parsing must cease.
// If it's not valid to advance with ev, but no error was encountered, the caller
// should attempt to advance with other candidate events from other threads. If the
// caller runs out of candidates, the trace is invalid.
//
// If this returns true, Next is guaranteed to return a complete event. However,
// multiple events may be added to the ordering, so the caller should (but is not
// required to) continue to call Next until it is exhausted.
func (o *ordering) Advance(ev *baseEvent, evt *evTable, m ThreadID, gen uint64) (bool, error) {
	if o.initialGen == 0 {
		// Set the initial gen if necessary.
		o.initialGen = gen
	}

	var curCtx, newCtx schedCtx
	curCtx.M = m
	newCtx.M = m

	var ms *mState
	if m == NoThread {
		curCtx.P = NoProc
		curCtx.G = NoGoroutine
		newCtx = curCtx
	} else {
		// Pull out or create the mState for this event.
		var ok bool
		ms, ok = o.mStates[m]
		if !ok {
			ms = &mState{
				g: NoGoroutine,
				p: NoProc,
			}
			o.mStates[m] = ms
		}
		curCtx.P = ms.p
		curCtx.G = ms.g
		newCtx = curCtx
	}

	// Generates an event from the current context.
	currentEvent := func() Event {
		return Event{table: evt, ctx: curCtx, base: *ev}
	}

	switch typ := ev.typ; typ {
	// Handle procs.
	case go122.EvProcStatus:
		pid := ProcID(ev.args[0])
		status := go122.ProcStatus(ev.args[1])
		if int(status) >= len(go122ProcStatus2ProcState) {
			return false, fmt.Errorf("invalid status for proc %d: %d", pid, status)
		}
		oldState := go122ProcStatus2ProcState[status]
		if s, ok := o.pStates[pid]; ok {
			if status == go122.ProcSyscallAbandoned && s.status == go122.ProcSyscall {
				// ProcSyscallAbandoned is a special case of ProcSyscall. It indicates a
				// potential loss of information, but if we're already in ProcSyscall,
				// we haven't lost the relevant information. Promote the status and advance.
				oldState = ProcRunning
				ev.args[1] = uint64(go122.ProcSyscall)
			} else if status == go122.ProcSyscallAbandoned && s.status == go122.ProcSyscallAbandoned {
				// If we're passing through ProcSyscallAbandoned, then there's no promotion
				// to do. We've lost the M that this P is associated with. However it got there,
				// it's going to appear as idle in the API, so pass through as idle.
				oldState = ProcIdle
				ev.args[1] = uint64(go122.ProcSyscallAbandoned)
			} else if s.status != status {
				return false, fmt.Errorf("inconsistent status for proc %d: old %v vs. new %v", pid, s.status, status)
			}
			s.seq = makeSeq(gen, 0) // Reset seq.
		} else {
			o.pStates[pid] = &pState{id: pid, status: status, seq: makeSeq(gen, 0)}
			if gen == o.initialGen {
				oldState = ProcUndetermined
			} else {
				oldState = ProcNotExist
			}
		}
		ev.extra(version.Go122)[0] = uint64(oldState) // Smuggle in the old state for StateTransition.

		// Bind the proc to the new context, if it's running.
		if status == go122.ProcRunning || status == go122.ProcSyscall {
			newCtx.P = pid
		}
		// If we're advancing through ProcSyscallAbandoned *but* oldState is running then we've
		// promoted it to ProcSyscall. However, because it's ProcSyscallAbandoned, we know this
		// P is about to get stolen and its status very likely isn't being emitted by the same
		// thread it was bound to. Since this status is Running -> Running and Running is binding,
		// we need to make sure we emit it in the right context: the context to which it is bound.
		// Find it, and set our current context to it.
		if status == go122.ProcSyscallAbandoned && oldState == ProcRunning {
			// N.B. This is slow but it should be fairly rare.
			found := false
			for mid, ms := range o.mStates {
				if ms.p == pid {
					curCtx.M = mid
					curCtx.P = pid
					curCtx.G = ms.g
					found = true
				}
			}
			if !found {
				return false, fmt.Errorf("failed to find sched context for proc %d that's about to be stolen", pid)
			}
		}
		o.queue.push(currentEvent())
	case go122.EvProcStart:
		pid := ProcID(ev.args[0])
		seq := makeSeq(gen, ev.args[1])

		// Try to advance. We might fail here due to sequencing, because the P hasn't
		// had a status emitted, or because we already have a P and we're in a syscall,
		// and we haven't observed that it was stolen from us yet.
		state, ok := o.pStates[pid]
		if !ok || state.status != go122.ProcIdle || !seq.succeeds(state.seq) || curCtx.P != NoProc {
			// We can't make an inference as to whether this is bad. We could just be seeing
			// a ProcStart on a different M before the proc's state was emitted, or before we
			// got to the right point in the trace.
			//
			// Note that we also don't advance here if we have a P and we're in a syscall.
			return false, nil
		}
		// We can advance this P. Check some invariants.
		//
		// We might have a goroutine if a goroutine is exiting a syscall.
		reqs := event.SchedReqs{Thread: event.MustHave, Proc: event.MustNotHave, Goroutine: event.MayHave}
		if err := validateCtx(curCtx, reqs); err != nil {
			return false, err
		}
		state.status = go122.ProcRunning
		state.seq = seq
		newCtx.P = pid
		o.queue.push(currentEvent())
	case go122.EvProcStop:
		// We must be able to advance this P.
		//
		// There are 2 ways a P can stop: ProcStop and ProcSteal. ProcStop is used when the P
		// is stopped by the same M that started it, while ProcSteal is used when another M
		// steals the P by stopping it from a distance.
		//
		// Since a P is bound to an M, and we're stopping on the same M we started, it must
		// always be possible to advance the current M's P from a ProcStop. This is also why
		// ProcStop doesn't need a sequence number.
		state, ok := o.pStates[curCtx.P]
		if !ok {
			return false, fmt.Errorf("event %s for proc (%v) that doesn't exist", go122.EventString(typ), curCtx.P)
		}
		if state.status != go122.ProcRunning && state.status != go122.ProcSyscall {
			return false, fmt.Errorf("%s event for proc that's not %s or %s", go122.EventString(typ), go122.ProcRunning, go122.ProcSyscall)
		}
		reqs := event.SchedReqs{Thread: event.MustHave, Proc: event.MustHave, Goroutine: event.MayHave}
		if err := validateCtx(curCtx, reqs); err != nil {
			return false, err
		}
		state.status = go122.ProcIdle
		newCtx.P = NoProc
		o.queue.push(currentEvent())
	case go122.EvProcSteal:
		pid := ProcID(ev.args[0])
		seq := makeSeq(gen, ev.args[1])
		state, ok := o.pStates[pid]
		if !ok || (state.status != go122.ProcSyscall && state.status != go122.ProcSyscallAbandoned) || !seq.succeeds(state.seq) {
			// We can't make an inference as to whether this is bad. We could just be seeing
			// a ProcStart on a different M before the proc's state was emitted, or before we
			// got to the right point in the trace.
			return false, nil
		}
		// We can advance this P. Check some invariants.
		reqs := event.SchedReqs{Thread: event.MustHave, Proc: event.MayHave, Goroutine: event.MayHave}
		if err := validateCtx(curCtx, reqs); err != nil {
			return false, err
		}
		// Smuggle in the P state that let us advance so we can surface information to the event.
		// Specifically, we need to make sure that the event is interpreted not as a transition of
		// ProcRunning -> ProcIdle but ProcIdle -> ProcIdle instead.
		//
		// ProcRunning is binding, but we may be running with a P on the current M and we can't
		// bind another P. This P is about to go ProcIdle anyway.
		oldStatus := state.status
		ev.extra(version.Go122)[0] = uint64(oldStatus)

		// Update the P's status and sequence number.
		state.status = go122.ProcIdle
		state.seq = seq

		// If we've lost information then don't try to do anything with the M.
		// It may have moved on and we can't be sure.
		if oldStatus == go122.ProcSyscallAbandoned {
			o.queue.push(currentEvent())
			break
		}

		// Validate that the M we're stealing from is what we expect.
		mid := ThreadID(ev.args[2]) // The M we're stealing from.

		if mid == curCtx.M {
			// We're stealing from ourselves. This behaves like a ProcStop.
			if curCtx.P != pid {
				return false, fmt.Errorf("tried to self-steal proc %d (thread %d), but got proc %d instead", pid, mid, curCtx.P)
			}
			newCtx.P = NoProc
			o.queue.push(currentEvent())
			break
		}

		// We're stealing from some other M.
		mState, ok := o.mStates[mid]
		if !ok {
			return false, fmt.Errorf("stole proc from non-existent thread %d", mid)
		}

		// Make sure we're actually stealing the right P.
		if mState.p != pid {
			return false, fmt.Errorf("tried to steal proc %d from thread %d, but got proc %d instead", pid, mid, mState.p)
		}

		// Tell the M it has no P so it can proceed.
		//
		// This is safe because we know the P was in a syscall and
		// the other M must be trying to get out of the syscall.
		// GoSyscallEndBlocked cannot advance until the corresponding
		// M loses its P.
		mState.p = NoProc
		o.queue.push(currentEvent())

	// Handle goroutines.
	case go122.EvGoStatus:
		gid := GoID(ev.args[0])
		mid := ThreadID(ev.args[1])
		status := go122.GoStatus(ev.args[2])

		if int(status) >= len(go122GoStatus2GoState) {
			return false, fmt.Errorf("invalid status for goroutine %d: %d", gid, status)
		}
		oldState := go122GoStatus2GoState[status]
		if s, ok := o.gStates[gid]; ok {
			if s.status != status {
				return false, fmt.Errorf("inconsistent status for goroutine %d: old %v vs. new %v", gid, s.status, status)
			}
			s.seq = makeSeq(gen, 0) // Reset seq.
		} else if gen == o.initialGen {
			// Set the state.
			o.gStates[gid] = &gState{id: gid, status: status, seq: makeSeq(gen, 0)}
			oldState = GoUndetermined
		} else {
			return false, fmt.Errorf("found goroutine status for new goroutine after the first generation: id=%v status=%v", gid, status)
		}
		ev.extra(version.Go122)[0] = uint64(oldState) // Smuggle in the old state for StateTransition.

		switch status {
		case go122.GoRunning:
			// Bind the goroutine to the new context, since it's running.
			newCtx.G = gid
		case go122.GoSyscall:
			if mid == NoThread {
				return false, fmt.Errorf("found goroutine %d in syscall without a thread", gid)
			}
			// Is the syscall on this thread? If so, bind it to the context.
			// Otherwise, we're talking about a G sitting in a syscall on an M.
			// Validate the named M.
			if mid == curCtx.M {
				if gen != o.initialGen && curCtx.G != gid {
					// If this isn't the first generation, we *must* have seen this
					// binding occur already. Even if the G was blocked in a syscall
					// for multiple generations since trace start, we would have seen
					// a previous GoStatus event that bound the goroutine to an M.
					return false, fmt.Errorf("inconsistent thread for syscalling goroutine %d: thread has goroutine %d", gid, curCtx.G)
				}
				newCtx.G = gid
				break
			}
			// Now we're talking about a thread and goroutine that have been
			// blocked on a syscall for the entire generation. This case must
			// not have a P; the runtime makes sure that all Ps are traced at
			// the beginning of a generation, which involves taking a P back
			// from every thread.
			ms, ok := o.mStates[mid]
			if ok {
				// This M has been seen. That means we must have seen this
				// goroutine go into a syscall on this thread at some point.
				if ms.g != gid {
					// But the G on the M doesn't match. Something's wrong.
					return false, fmt.Errorf("inconsistent thread for syscalling goroutine %d: thread has goroutine %d", gid, ms.g)
				}
				// This case is just a Syscall->Syscall event, which needs to
				// appear as having the G currently bound to this M.
				curCtx.G = ms.g
			} else if !ok {
				// The M hasn't been seen yet. That means this goroutine
				// has just been sitting in a syscall on this M. Create
				// a state for it.
				o.mStates[mid] = &mState{g: gid, p: NoProc}
				// Don't set curCtx.G in this case because this event is the
				// binding event (and curCtx represents the "before" state).
			}
			// Update the current context to the M we're talking about.
			curCtx.M = mid
		}
		o.queue.push(currentEvent())
	case go122.EvGoCreate:
		// Goroutines must be created on a running P, but may or may not be created
		// by a running goroutine.
		reqs := event.SchedReqs{Thread: event.MustHave, Proc: event.MustHave, Goroutine: event.MayHave}
		if err := validateCtx(curCtx, reqs); err != nil {
			return false, err
		}
		// If we have a goroutine, it must be running.
		if state, ok := o.gStates[curCtx.G]; ok && state.status != go122.GoRunning {
			return false, fmt.Errorf("%s event for goroutine that's not %s", go122.EventString(typ), GoRunning)
		}
		// This goroutine created another. Add a state for it.
		newgid := GoID(ev.args[0])
		if _, ok := o.gStates[newgid]; ok {
			return false, fmt.Errorf("tried to create goroutine (%v) that already exists", newgid)
		}
		o.gStates[newgid] = &gState{id: newgid, status: go122.GoRunnable, seq: makeSeq(gen, 0)}
		o.queue.push(currentEvent())
	case go122.EvGoDestroy, go122.EvGoStop, go122.EvGoBlock:
		// These are goroutine events that all require an active running
		// goroutine on some thread. They must *always* be advance-able,
		// since running goroutines are bound to their M.
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		state, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("event %s for goroutine (%v) that doesn't exist", go122.EventString(typ), curCtx.G)
		}
		if state.status != go122.GoRunning {
			return false, fmt.Errorf("%s event for goroutine that's not %s", go122.EventString(typ), GoRunning)
		}
		// Handle each case slightly differently; we just group them together
		// because they have shared preconditions.
		switch typ {
		case go122.EvGoDestroy:
			// This goroutine is exiting itself.
			delete(o.gStates, curCtx.G)
			newCtx.G = NoGoroutine
		case go122.EvGoStop:
			// Goroutine stopped (yielded). It's runnable but not running on this M.
			state.status = go122.GoRunnable
			newCtx.G = NoGoroutine
		case go122.EvGoBlock:
			// Goroutine blocked. It's waiting now and not running on this M.
			state.status = go122.GoWaiting
			newCtx.G = NoGoroutine
		}
		o.queue.push(currentEvent())
	case go122.EvGoStart:
		gid := GoID(ev.args[0])
		seq := makeSeq(gen, ev.args[1])
		state, ok := o.gStates[gid]
		if !ok || state.status != go122.GoRunnable || !seq.succeeds(state.seq) {
			// We can't make an inference as to whether this is bad. We could just be seeing
			// a GoStart on a different M before the goroutine was created, before it had its
			// state emitted, or before we got to the right point in the trace yet.
			return false, nil
		}
		// We can advance this goroutine. Check some invariants.
		reqs := event.SchedReqs{Thread: event.MustHave, Proc: event.MustHave, Goroutine: event.MustNotHave}
		if err := validateCtx(curCtx, reqs); err != nil {
			return false, err
		}
		state.status = go122.GoRunning
		state.seq = seq
		newCtx.G = gid
		o.queue.push(currentEvent())
	case go122.EvGoUnblock:
		// N.B. These both reference the goroutine to unblock, not the current goroutine.
		gid := GoID(ev.args[0])
		seq := makeSeq(gen, ev.args[1])
		state, ok := o.gStates[gid]
		if !ok || state.status != go122.GoWaiting || !seq.succeeds(state.seq) {
			// We can't make an inference as to whether this is bad. We could just be seeing
			// a GoUnblock on a different M before the goroutine was created and blocked itself,
			// before it had its state emitted, or before we got to the right point in the trace yet.
			return false, nil
		}
		state.status = go122.GoRunnable
		state.seq = seq
		// N.B. No context to validate. Basically anything can unblock
		// a goroutine (e.g. sysmon).
		o.queue.push(currentEvent())
	case go122.EvGoSyscallBegin:
		// Entering a syscall requires an active running goroutine with a
		// proc on some thread. It is always advancable.
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		state, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("event %s for goroutine (%v) that doesn't exist", go122.EventString(typ), curCtx.G)
		}
		if state.status != go122.GoRunning {
			return false, fmt.Errorf("%s event for goroutine that's not %s", go122.EventString(typ), GoRunning)
		}
		// Goroutine entered a syscall. It's still running on this P and M.
		state.status = go122.GoSyscall
		pState, ok := o.pStates[curCtx.P]
		if !ok {
			return false, fmt.Errorf("uninitialized proc %d found during %s", curCtx.P, go122.EventString(typ))
		}
		pState.status = go122.ProcSyscall
		// Validate the P sequence number on the event and advance it.
		//
		// We have a P sequence number for what is supposed to be a goroutine event
		// so that we can correctly model P stealing. Without this sequence number here,
		// the syscall from which a ProcSteal event is stealing can be ambiguous in the
		// face of broken timestamps. See the go122-syscall-steal-proc-ambiguous test for
		// more details.
		//
		// Note that because this sequence number only exists as a tool for disambiguation,
		// we can enforce that we have the right sequence number at this point; we don't need
		// to back off and see if any other events will advance. This is a running P.
		pSeq := makeSeq(gen, ev.args[0])
		if !pSeq.succeeds(pState.seq) {
			return false, fmt.Errorf("failed to advance %s: can't make sequence: %s -> %s", go122.EventString(typ), pState.seq, pSeq)
		}
		pState.seq = pSeq
		o.queue.push(currentEvent())
	case go122.EvGoSyscallEnd:
		// This event is always advance-able because it happens on the same
		// thread that EvGoSyscallStart happened, and the goroutine can't leave
		// that thread until its done.
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		state, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("event %s for goroutine (%v) that doesn't exist", go122.EventString(typ), curCtx.G)
		}
		if state.status != go122.GoSyscall {
			return false, fmt.Errorf("%s event for goroutine that's not %s", go122.EventString(typ), GoRunning)
		}
		state.status = go122.GoRunning

		// Transfer the P back to running from syscall.
		pState, ok := o.pStates[curCtx.P]
		if !ok {
			return false, fmt.Errorf("uninitialized proc %d found during %s", curCtx.P, go122.EventString(typ))
		}
		if pState.status != go122.ProcSyscall {
			return false, fmt.Errorf("expected proc %d in state %v, but got %v instead", curCtx.P, go122.ProcSyscall, pState.status)
		}
		pState.status = go122.ProcRunning
		o.queue.push(currentEvent())
	case go122.EvGoSyscallEndBlocked:
		// This event becomes advanceable when its P is not in a syscall state
		// (lack of a P altogether is also acceptable for advancing).
		// The transfer out of ProcSyscall can happen either voluntarily via
		// ProcStop or involuntarily via ProcSteal. We may also acquire a new P
		// before we get here (after the transfer out) but that's OK: that new
		// P won't be in the ProcSyscall state anymore.
		//
		// Basically: while we have a preemptible P, don't advance, because we
		// *know* from the event that we're going to lose it at some point during
		// the syscall. We shouldn't advance until that happens.
		if curCtx.P != NoProc {
			pState, ok := o.pStates[curCtx.P]
			if !ok {
				return false, fmt.Errorf("uninitialized proc %d found during %s", curCtx.P, go122.EventString(typ))
			}
			if pState.status == go122.ProcSyscall {
				return false, nil
			}
		}
		// As mentioned above, we may have a P here if we ProcStart
		// before this event.
		if err := validateCtx(curCtx, event.SchedReqs{Thread: event.MustHave, Proc: event.MayHave, Goroutine: event.MustHave}); err != nil {
			return false, err
		}
		state, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("event %s for goroutine (%v) that doesn't exist", go122.EventString(typ), curCtx.G)
		}
		if state.status != go122.GoSyscall {
			return false, fmt.Errorf("%s event for goroutine that's not %s", go122.EventString(typ), GoRunning)
		}
		newCtx.G = NoGoroutine
		state.status = go122.GoRunnable
		o.queue.push(currentEvent())
	case go122.EvGoCreateSyscall:
		// This event indicates that a goroutine is effectively
		// being created out of a cgo callback. Such a goroutine
		// is 'created' in the syscall state.
		if err := validateCtx(curCtx, event.SchedReqs{Thread: event.MustHave, Proc: event.MayHave, Goroutine: event.MustNotHave}); err != nil {
			return false, err
		}
		// This goroutine is effectively being created. Add a state for it.
		newgid := GoID(ev.args[0])
		if _, ok := o.gStates[newgid]; ok {
			return false, fmt.Errorf("tried to create goroutine (%v) in syscall that already exists", newgid)
		}
		o.gStates[newgid] = &gState{id: newgid, status: go122.GoSyscall, seq: makeSeq(gen, 0)}
		// Goroutine is executing. Bind it to the context.
		newCtx.G = newgid
		o.queue.push(currentEvent())
	case go122.EvGoDestroySyscall:
		// This event indicates that a goroutine created for a
		// cgo callback is disappearing, either because the callback
		// ending or the C thread that called it is being destroyed.
		//
		// Also, treat this as if we lost our P too.
		// The thread ID may be reused by the platform and we'll get
		// really confused if we try to steal the P is this is running
		// with later. The new M with the same ID could even try to
		// steal back this P from itself!
		//
		// The runtime is careful to make sure that any GoCreateSyscall
		// event will enter the runtime emitting events for reacquiring a P.
		//
		// Note: we might have a P here. The P might not be released
		// eagerly by the runtime, and it might get stolen back later
		// (or never again, if the program is going to exit).
		if err := validateCtx(curCtx, event.SchedReqs{Thread: event.MustHave, Proc: event.MayHave, Goroutine: event.MustHave}); err != nil {
			return false, err
		}
		// Check to make sure the goroutine exists in the right state.
		state, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("event %s for goroutine (%v) that doesn't exist", go122.EventString(typ), curCtx.G)
		}
		if state.status != go122.GoSyscall {
			return false, fmt.Errorf("%s event for goroutine that's not %v", go122.EventString(typ), GoSyscall)
		}
		// This goroutine is exiting itself.
		delete(o.gStates, curCtx.G)
		newCtx.G = NoGoroutine

		// If we have a proc, then we're dissociating from it now. See the comment at the top of the case.
		if curCtx.P != NoProc {
			pState, ok := o.pStates[curCtx.P]
			if !ok {
				return false, fmt.Errorf("found invalid proc %d during %s", curCtx.P, go122.EventString(typ))
			}
			if pState.status != go122.ProcSyscall {
				return false, fmt.Errorf("proc %d in unexpected state %s during %s", curCtx.P, pState.status, go122.EventString(typ))
			}
			// See the go122-create-syscall-reuse-thread-id test case for more details.
			pState.status = go122.ProcSyscallAbandoned
			newCtx.P = NoProc

			// Queue an extra self-ProcSteal event.
			extra := Event{
				table: evt,
				ctx:   curCtx,
				base: baseEvent{
					typ:  go122.EvProcSteal,
					time: ev.time,
				},
			}
			extra.base.args[0] = uint64(curCtx.P)
			extra.base.extra(version.Go122)[0] = uint64(go122.ProcSyscall)
			o.queue.push(extra)
		}
		o.queue.push(currentEvent())

	// Handle tasks. Tasks are interesting because:
	// - There's no Begin event required to reference a task.
	// - End for a particular task ID can appear multiple times.
	// As a result, there's very little to validate. The only
	// thing we have to be sure of is that a task didn't begin
	// after it had already begun. Task IDs are allowed to be
	// reused, so we don't care about a Begin after an End.
	case go122.EvUserTaskBegin:
		id := TaskID(ev.args[0])
		if _, ok := o.activeTasks[id]; ok {
			return false, fmt.Errorf("task ID conflict: %d", id)
		}
		// Get the parent ID, but don't validate it. There's no guarantee
		// we actually have information on whether it's active.
		parentID := TaskID(ev.args[1])
		if parentID == BackgroundTask {
			// Note: a value of 0 here actually means no parent, *not* the
			// background task. Automatic background task attachment only
			// applies to regions.
			parentID = NoTask
			ev.args[1] = uint64(NoTask)
		}

		// Validate the name and record it. We'll need to pass it through to
		// EvUserTaskEnd.
		nameID := stringID(ev.args[2])
		name, ok := evt.strings.get(nameID)
		if !ok {
			return false, fmt.Errorf("invalid string ID %v for %v event", nameID, typ)
		}
		o.activeTasks[id] = taskState{name: name, parentID: parentID}
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvUserTaskEnd:
		id := TaskID(ev.args[0])
		if ts, ok := o.activeTasks[id]; ok {
			// Smuggle the task info. This may happen in a different generation,
			// which may not have the name in its string table. Add it to the extra
			// strings table so we can look it up later.
			ev.extra(version.Go122)[0] = uint64(ts.parentID)
			ev.extra(version.Go122)[1] = uint64(evt.addExtraString(ts.name))
			delete(o.activeTasks, id)
		} else {
			// Explicitly clear the task info.
			ev.extra(version.Go122)[0] = uint64(NoTask)
			ev.extra(version.Go122)[1] = uint64(evt.addExtraString(""))
		}
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())

	// Handle user regions.
	case go122.EvUserRegionBegin:
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		tid := TaskID(ev.args[0])
		nameID := stringID(ev.args[1])
		name, ok := evt.strings.get(nameID)
		if !ok {
			return false, fmt.Errorf("invalid string ID %v for %v event", nameID, typ)
		}
		gState, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("encountered EvUserRegionBegin without known state for current goroutine %d", curCtx.G)
		}
		if err := gState.beginRegion(userRegion{tid, name}); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvUserRegionEnd:
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		tid := TaskID(ev.args[0])
		nameID := stringID(ev.args[1])
		name, ok := evt.strings.get(nameID)
		if !ok {
			return false, fmt.Errorf("invalid string ID %v for %v event", nameID, typ)
		}
		gState, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("encountered EvUserRegionEnd without known state for current goroutine %d", curCtx.G)
		}
		if err := gState.endRegion(userRegion{tid, name}); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())

	// Handle the GC mark phase.
	//
	// We have sequence numbers for both start and end because they
	// can happen on completely different threads. We want an explicit
	// partial order edge between start and end here, otherwise we're
	// relying entirely on timestamps to make sure we don't advance a
	// GCEnd for a _different_ GC cycle if timestamps are wildly broken.
	case go122.EvGCActive:
		seq := ev.args[0]
		if gen == o.initialGen {
			if o.gcState != gcUndetermined {
				return false, fmt.Errorf("GCActive in the first generation isn't first GC event")
			}
			o.gcSeq = seq
			o.gcState = gcRunning
			o.queue.push(currentEvent())
			break
		}
		if seq != o.gcSeq+1 {
			// This is not the right GC cycle.
			return false, nil
		}
		if o.gcState != gcRunning {
			return false, fmt.Errorf("encountered GCActive while GC was not in progress")
		}
		o.gcSeq = seq
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvGCBegin:
		seq := ev.args[0]
		if o.gcState == gcUndetermined {
			o.gcSeq = seq
			o.gcState = gcRunning
			o.queue.push(currentEvent())
			break
		}
		if seq != o.gcSeq+1 {
			// This is not the right GC cycle.
			return false, nil
		}
		if o.gcState == gcRunning {
			return false, fmt.Errorf("encountered GCBegin while GC was already in progress")
		}
		o.gcSeq = seq
		o.gcState = gcRunning
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvGCEnd:
		seq := ev.args[0]
		if seq != o.gcSeq+1 {
			// This is not the right GC cycle.
			return false, nil
		}
		if o.gcState == gcNotRunning {
			return false, fmt.Errorf("encountered GCEnd when GC was not in progress")
		}
		if o.gcState == gcUndetermined {
			return false, fmt.Errorf("encountered GCEnd when GC was in an undetermined state")
		}
		o.gcSeq = seq
		o.gcState = gcNotRunning
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())

	// Handle simple instantaneous events that require a G.
	case go122.EvGoLabel, go122.EvProcsChange, go122.EvUserLog:
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())

	// Handle allocation states, which don't require a G.
	case go122.EvHeapAlloc, go122.EvHeapGoal:
		if err := validateCtx(curCtx, event.SchedReqs{Thread: event.MustHave, Proc: event.MustHave, Goroutine: event.MayHave}); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())

	// Handle sweep, which is bound to a P and doesn't require a G.
	case go122.EvGCSweepBegin:
		if err := validateCtx(curCtx, event.SchedReqs{Thread: event.MustHave, Proc: event.MustHave, Goroutine: event.MayHave}); err != nil {
			return false, err
		}
		if err := o.pStates[curCtx.P].beginRange(makeRangeType(typ, 0)); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvGCSweepActive:
		pid := ProcID(ev.args[0])
		// N.B. In practice Ps can't block while they're sweeping, so this can only
		// ever reference curCtx.P. However, be lenient about this like we are with
		// GCMarkAssistActive; there's no reason the runtime couldn't change to block
		// in the middle of a sweep.
		pState, ok := o.pStates[pid]
		if !ok {
			return false, fmt.Errorf("encountered GCSweepActive for unknown proc %d", pid)
		}
		if err := pState.activeRange(makeRangeType(typ, 0), gen == o.initialGen); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvGCSweepEnd:
		if err := validateCtx(curCtx, event.SchedReqs{Thread: event.MustHave, Proc: event.MustHave, Goroutine: event.MayHave}); err != nil {
			return false, err
		}
		_, err := o.pStates[curCtx.P].endRange(typ)
		if err != nil {
			return false, err
		}
		o.queue.push(currentEvent())

	// Handle special goroutine-bound event ranges.
	case go122.EvSTWBegin, go122.EvGCMarkAssistBegin:
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		desc := stringID(0)
		if typ == go122.EvSTWBegin {
			desc = stringID(ev.args[0])
		}
		gState, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("encountered event of type %d without known state for current goroutine %d", typ, curCtx.G)
		}
		if err := gState.beginRange(makeRangeType(typ, desc)); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvGCMarkAssistActive:
		gid := GoID(ev.args[0])
		// N.B. Like GoStatus, this can happen at any time, because it can
		// reference a non-running goroutine. Don't check anything about the
		// current scheduler context.
		gState, ok := o.gStates[gid]
		if !ok {
			return false, fmt.Errorf("uninitialized goroutine %d found during %s", gid, go122.EventString(typ))
		}
		if err := gState.activeRange(makeRangeType(typ, 0), gen == o.initialGen); err != nil {
			return false, err
		}
		o.queue.push(currentEvent())
	case go122.EvSTWEnd, go122.EvGCMarkAssistEnd:
		if err := validateCtx(curCtx, event.UserGoReqs); err != nil {
			return false, err
		}
		gState, ok := o.gStates[curCtx.G]
		if !ok {
			return false, fmt.Errorf("encountered event of type %d without known state for current goroutine %d", typ, curCtx.G)
		}
		desc, err := gState.endRange(typ)
		if err != nil {
			return false, err
		}
		if typ == go122.EvSTWEnd {
			// Smuggle the kind into the event.
			// Don't use ev.extra here so we have symmetry with STWBegin.
			ev.args[0] = uint64(desc)
		}
		o.queue.push(currentEvent())
	default:
		return false, fmt.Errorf("bad event type found while ordering: %v", ev.typ)
	}
	if ms != nil {
		// Update the mState for this event.
		ms.p = newCtx.P
		ms.g = newCtx.G
	}
	return true, nil
}

// Next returns the next event in the ordering.
func (o *ordering) Next() (Event, bool) {
	return o.queue.pop()
}

// schedCtx represents the scheduling resources associated with an event.
type schedCtx struct {
	G GoID
	P ProcID
	M ThreadID
}

// validateCtx ensures that ctx conforms to some reqs, returning an error if
// it doesn't.
func validateCtx(ctx schedCtx, reqs event.SchedReqs) error {
	// Check thread requirements.
	if reqs.Thread == event.MustHave && ctx.M == NoThread {
		return fmt.Errorf("expected a thread but didn't have one")
	} else if reqs.Thread == event.MustNotHave && ctx.M != NoThread {
		return fmt.Errorf("expected no thread but had one")
	}

	// Check proc requirements.
	if reqs.Proc == event.MustHave && ctx.P == NoProc {
		return fmt.Errorf("expected a proc but didn't have one")
	} else if reqs.Proc == event.MustNotHave && ctx.P != NoProc {
		return fmt.Errorf("expected no proc but had one")
	}

	// Check goroutine requirements.
	if reqs.Goroutine == event.MustHave && ctx.G == NoGoroutine {
		return fmt.Errorf("expected a goroutine but didn't have one")
	} else if reqs.Goroutine == event.MustNotHave && ctx.G != NoGoroutine {
		return fmt.Errorf("expected no goroutine but had one")
	}
	return nil
}

// gcState is a trinary variable for the current state of the GC.
//
// The third state besides "enabled" and "disabled" is "undetermined."
type gcState uint8

const (
	gcUndetermined gcState = iota
	gcNotRunning
	gcRunning
)

// String returns a human-readable string for the GC state.
func (s gcState) String() string {
	switch s {
	case gcUndetermined:
		return "Undetermined"
	case gcNotRunning:
		return "NotRunning"
	case gcRunning:
		return "Running"
	}
	return "Bad"
}

// userRegion represents a unique user region when attached to some gState.
type userRegion struct {
	// name must be a resolved string because the string ID for the same
	// string may change across generations, but we care about checking
	// the value itself.
	taskID TaskID
	name   string
}

// rangeType is a way to classify special ranges of time.
//
// These typically correspond 1:1 with "Begin" events, but
// they may have an optional subtype that describes the range
// in more detail.
type rangeType struct {
	typ  event.Type // "Begin" event.
	desc stringID   // Optional subtype.
}

// makeRangeType constructs a new rangeType.
func makeRangeType(typ event.Type, desc stringID) rangeType {
	if styp := go122.Specs()[typ].StartEv; styp != go122.EvNone {
		typ = styp
	}
	return rangeType{typ, desc}
}

// gState is the state of a goroutine at a point in the trace.
type gState struct {
	id     GoID
	status go122.GoStatus
	seq    seqCounter

	// regions are the active user regions for this goroutine.
	regions []userRegion

	// rangeState is the state of special time ranges bound to this goroutine.
	rangeState
}

// beginRegion starts a user region on the goroutine.
func (s *gState) beginRegion(r userRegion) error {
	s.regions = append(s.regions, r)
	return nil
}

// endRegion ends a user region on the goroutine.
func (s *gState) endRegion(r userRegion) error {
	if len(s.regions) == 0 {
		// We do not know about regions that began before tracing started.
		return nil
	}
	if next := s.regions[len(s.regions)-1]; next != r {
		return fmt.Errorf("misuse of region in goroutine %v: region end %v when the inner-most active region start event is %v", s.id, r, next)
	}
	s.regions = s.regions[:len(s.regions)-1]
	return nil
}

// pState is the state of a proc at a point in the trace.
type pState struct {
	id     ProcID
	status go122.ProcStatus
	seq    seqCounter

	// rangeState is the state of special time ranges bound to this proc.
	rangeState
}

// mState is the state of a thread at a point in the trace.
type mState struct {
	g GoID   // Goroutine bound to this M. (The goroutine's state is Executing.)
	p ProcID // Proc bound to this M. (The proc's state is Executing.)
}

// rangeState represents the state of special time ranges.
type rangeState struct {
	// inFlight contains the rangeTypes of any ranges bound to a resource.
	inFlight []rangeType
}

// beginRange begins a special range in time on the goroutine.
//
// Returns an error if the range is already in progress.
func (s *rangeState) beginRange(typ rangeType) error {
	if s.hasRange(typ) {
		return fmt.Errorf("discovered event already in-flight for when starting event %v", go122.Specs()[typ.typ].Name)
	}
	s.inFlight = append(s.inFlight, typ)
	return nil
}

// activeRange marks special range in time on the goroutine as active in the
// initial generation, or confirms that it is indeed active in later generations.
func (s *rangeState) activeRange(typ rangeType, isInitialGen bool) error {
	if isInitialGen {
		if s.hasRange(typ) {
			return fmt.Errorf("found named active range already in first gen: %v", typ)
		}
		s.inFlight = append(s.inFlight, typ)
	} else if !s.hasRange(typ) {
		return fmt.Errorf("resource is missing active range: %v %v", go122.Specs()[typ.typ].Name, s.inFlight)
	}
	return nil
}

// hasRange returns true if a special time range on the goroutine as in progress.
func (s *rangeState) hasRange(typ rangeType) bool {
	for _, ftyp := range s.inFlight {
		if ftyp == typ {
			return true
		}
	}
	return false
}

// endsRange ends a special range in time on the goroutine.
//
// This must line up with the start event type  of the range the goroutine is currently in.
func (s *rangeState) endRange(typ event.Type) (stringID, error) {
	st := go122.Specs()[typ].StartEv
	idx := -1
	for i, r := range s.inFlight {
		if r.typ == st {
			idx = i
			break
		}
	}
	if idx < 0 {
		return 0, fmt.Errorf("tried to end event %v, but not in-flight", go122.Specs()[st].Name)
	}
	// Swap remove.
	desc := s.inFlight[idx].desc
	s.inFlight[idx], s.inFlight[len(s.inFlight)-1] = s.inFlight[len(s.inFlight)-1], s.inFlight[idx]
	s.inFlight = s.inFlight[:len(s.inFlight)-1]
	return desc, nil
}

// seqCounter represents a global sequence counter for a resource.
type seqCounter struct {
	gen uint64 // The generation for the local sequence counter seq.
	seq uint64 // The sequence number local to the generation.
}

// makeSeq creates a new seqCounter.
func makeSeq(gen, seq uint64) seqCounter {
	return seqCounter{gen: gen, seq: seq}
}

// succeeds returns true if a is the immediate successor of b.
func (a seqCounter) succeeds(b seqCounter) bool {
	return a.gen == b.gen && a.seq == b.seq+1
}

// String returns a debug string representation of the seqCounter.
func (c seqCounter) String() string {
	return fmt.Sprintf("%d (gen=%d)", c.seq, c.gen)
}

func dumpOrdering(order *ordering) string {
	var sb strings.Builder
	for id, state := range order.gStates {
		fmt.Fprintf(&sb, "G %d [status=%s seq=%s]\n", id, state.status, state.seq)
	}
	fmt.Fprintln(&sb)
	for id, state := range order.pStates {
		fmt.Fprintf(&sb, "P %d [status=%s seq=%s]\n", id, state.status, state.seq)
	}
	fmt.Fprintln(&sb)
	for id, state := range order.mStates {
		fmt.Fprintf(&sb, "M %d [g=%d p=%d]\n", id, state.g, state.p)
	}
	fmt.Fprintln(&sb)
	fmt.Fprintf(&sb, "GC %d %s\n", order.gcSeq, order.gcState)
	return sb.String()
}

// taskState represents an active task.
type taskState struct {
	// name is the type of the active task.
	name string

	// parentID is the parent ID of the active task.
	parentID TaskID
}

// queue implements a growable ring buffer with a queue API.
type queue[T any] struct {
	start, end int
	buf        []T
}

// push adds a new event to the back of the queue.
func (q *queue[T]) push(value T) {
	if q.end-q.start == len(q.buf) {
		q.grow()
	}
	q.buf[q.end%len(q.buf)] = value
	q.end++
}

// grow increases the size of the queue.
func (q *queue[T]) grow() {
	if len(q.buf) == 0 {
		q.buf = make([]T, 2)
		return
	}

	// Create new buf and copy data over.
	newBuf := make([]T, len(q.buf)*2)
	pivot := q.start % len(q.buf)
	first, last := q.buf[pivot:], q.buf[:pivot]
	copy(newBuf[:len(first)], first)
	copy(newBuf[len(first):], last)

	// Update the queue state.
	q.start = 0
	q.end = len(q.buf)
	q.buf = newBuf
}

// pop removes an event from the front of the queue. If the
// queue is empty, it returns an EventBad event.
func (q *queue[T]) pop() (T, bool) {
	if q.end-q.start == 0 {
		return *new(T), false
	}
	elem := &q.buf[q.start%len(q.buf)]
	value := *elem
	*elem = *new(T) // Clear the entry before returning, so we don't hold onto old tables.
	q.start++
	return value, true
}
