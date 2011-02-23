// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import "runtime"

// Cond implements a condition variable, a rendezvous point
// for goroutines waiting for or announcing the occurrence
// of an event.
//
// Each Cond has an associated Locker L (often a *Mutex or *RWMutex),
// which must be held when changing the condition and
// when calling the Wait method.
type Cond struct {
	L       Locker // held while observing or changing the condition
	m       Mutex  // held to avoid internal races
	waiters int    // number of goroutines blocked on Wait
	sema    *uint32
}

// NewCond returns a new Cond with Locker l.
func NewCond(l Locker) *Cond {
	return &Cond{L: l}
}

// Wait atomically unlocks c.L and suspends execution
// of the calling goroutine.  After later resuming execution,
// Wait locks c.L before returning.
//
// Because L is not locked when Wait first resumes, the caller
// typically cannot assume that the condition is true when
// Wait returns.  Instead, the caller should Wait in a loop:
//
//    c.L.Lock()
//    for !condition() {
//        c.Wait()
//    }
//    ... make use of condition ...
//    c.L.Unlock()
//
func (c *Cond) Wait() {
	c.m.Lock()
	if c.sema == nil {
		c.sema = new(uint32)
	}
	s := c.sema
	c.waiters++
	c.m.Unlock()
	c.L.Unlock()
	runtime.Semacquire(s)
	c.L.Lock()
}

// Signal wakes one goroutine waiting on c, if there is any.
//
// It is allowed but not required for the caller to hold c.L
// during the call.
func (c *Cond) Signal() {
	c.m.Lock()
	if c.waiters > 0 {
		c.waiters--
		runtime.Semrelease(c.sema)
	}
	c.m.Unlock()
}

// Broadcast wakes all goroutines waiting on c.
//
// It is allowed but not required for the caller to hold c.L
// during the call.
func (c *Cond) Broadcast() {
	c.m.Lock()
	if c.waiters > 0 {
		s := c.sema
		n := c.waiters
		for i := 0; i < n; i++ {
			runtime.Semrelease(s)
		}
		// We just issued n wakeups via the semaphore s.
		// To ensure that they wake up the existing waiters
		// and not waiters that arrive after Broadcast returns,
		// clear c.sema.  The next operation will allocate
		// a new one.
		c.sema = nil
		c.waiters = 0
	}
	c.m.Unlock()
}
