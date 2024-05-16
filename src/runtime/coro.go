// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// A coro represents extra concurrency without extra parallelism,
// as would be needed for a coroutine implementation.
// The coro does not represent a specific coroutine, only the ability
// to do coroutine-style control transfers.
// It can be thought of as like a special channel that always has
// a goroutine blocked on it. If another goroutine calls coroswitch(c),
// the caller becomes the goroutine blocked in c, and the goroutine
// formerly blocked in c starts running.
// These switches continue until a call to coroexit(c),
// which ends the use of the coro by releasing the blocked
// goroutine in c and exiting the current goroutine.
//
// Coros are heap allocated and garbage collected, so that user code
// can hold a pointer to a coro without causing potential dangling
// pointer errors.
type coro struct {
	gp guintptr
	f  func(*coro)
}

//go:linkname newcoro

// newcoro creates a new coro containing a
// goroutine blocked waiting to run f
// and returns that coro.
func newcoro(f func(*coro)) *coro {
	c := new(coro)
	c.f = f
	pc := getcallerpc()
	gp := getg()
	systemstack(func() {
		start := corostart
		startfv := *(**funcval)(unsafe.Pointer(&start))
		gp = newproc1(startfv, gp, pc, true, waitReasonCoroutine)
	})
	gp.coroarg = c
	c.gp.set(gp)
	return c
}

// corostart is the entry func for a new coroutine.
// It runs the coroutine user function f passed to corostart
// and then calls coroexit to remove the extra concurrency.
func corostart() {
	gp := getg()
	c := gp.coroarg
	gp.coroarg = nil

	c.f(c)
	coroexit(c)
}

// coroexit is like coroswitch but closes the coro
// and exits the current goroutine
func coroexit(c *coro) {
	gp := getg()
	gp.coroarg = c
	gp.coroexit = true
	mcall(coroswitch_m)
}

//go:linkname coroswitch

// coroswitch switches to the goroutine blocked on c
// and then blocks the current goroutine on c.
func coroswitch(c *coro) {
	gp := getg()
	gp.coroarg = c
	mcall(coroswitch_m)
}

// coroswitch_m is the implementation of coroswitch
// that runs on the m stack.
//
// Note: Coroutine switches are expected to happen at
// an order of magnitude (or more) higher frequency
// than regular goroutine switches, so this path is heavily
// optimized to remove unnecessary work.
// The fast path here is three CAS: the one at the top on gp.atomicstatus,
// the one in the middle to choose the next g,
// and the one at the bottom on gnext.atomicstatus.
// It is important not to add more atomic operations or other
// expensive operations to the fast path.
func coroswitch_m(gp *g) {
	// TODO(go.dev/issue/65889): Something really nasty will happen if either
	// goroutine in this handoff tries to lock itself to an OS thread.
	// There's an explicit multiplexing going on here that needs to be
	// disabled if either the consumer or the iterator ends up in such
	// a state.
	c := gp.coroarg
	gp.coroarg = nil
	exit := gp.coroexit
	gp.coroexit = false
	mp := gp.m

	// Acquire tracer for writing for the duration of this call.
	//
	// There's a lot of state manipulation performed with shortcuts
	// but we need to make sure the tracer can only observe the
	// start and end states to maintain a coherent model and avoid
	// emitting an event for every single transition.
	trace := traceAcquire()

	if exit {
		// TODO(65889): If we're locked to the current OS thread and
		// we exit here while tracing is enabled, we're going to end up
		// in a really bad place (traceAcquire also calls acquirem; there's
		// no releasem before the thread exits).
		gdestroy(gp)
		gp = nil
	} else {
		// If we can CAS ourselves directly from running to waiting, so do,
		// keeping the control transfer as lightweight as possible.
		gp.waitreason = waitReasonCoroutine
		if !gp.atomicstatus.CompareAndSwap(_Grunning, _Gwaiting) {
			// The CAS failed: use casgstatus, which will take care of
			// coordinating with the garbage collector about the state change.
			casgstatus(gp, _Grunning, _Gwaiting)
		}

		// Clear gp.m.
		setMNoWB(&gp.m, nil)
	}

	// The goroutine stored in c is the one to run next.
	// Swap it with ourselves.
	var gnext *g
	for {
		// Note: this is a racy load, but it will eventually
		// get the right value, and if it gets the wrong value,
		// the c.gp.cas will fail, so no harm done other than
		// a wasted loop iteration.
		// The cas will also sync c.gp's
		// memory enough that the next iteration of the racy load
		// should see the correct value.
		// We are avoiding the atomic load to keep this path
		// as lightweight as absolutely possible.
		// (The atomic load is free on x86 but not free elsewhere.)
		next := c.gp
		if next.ptr() == nil {
			throw("coroswitch on exited coro")
		}
		var self guintptr
		self.set(gp)
		if c.gp.cas(next, self) {
			gnext = next.ptr()
			break
		}
	}

	// Emit the trace event after getting gnext but before changing curg.
	// GoSwitch expects that the current G is running and that we haven't
	// switched yet for correct status emission.
	if trace.ok() {
		trace.GoSwitch(gnext, exit)
	}

	// Start running next, without heavy scheduling machinery.
	// Set mp.curg and gnext.m and then update scheduling state
	// directly if possible.
	setGNoWB(&mp.curg, gnext)
	setMNoWB(&gnext.m, mp)
	if !gnext.atomicstatus.CompareAndSwap(_Gwaiting, _Grunning) {
		// The CAS failed: use casgstatus, which will take care of
		// coordinating with the garbage collector about the state change.
		casgstatus(gnext, _Gwaiting, _Grunnable)
		casgstatus(gnext, _Grunnable, _Grunning)
	}

	// Release the trace locker. We've completed all the necessary transitions..
	if trace.ok() {
		traceRelease(trace)
	}

	// Switch to gnext. Does not return.
	gogo(&gnext.sched)
}
