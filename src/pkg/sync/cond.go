// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

// Cond implements a condition variable, a rendezvous point
// for goroutines waiting for or announcing the occurrence
// of an event.
//
// Each Cond has an associated Locker L (often a *Mutex or *RWMutex),
// which must be held when changing the condition and
// when calling the Wait method.
type Cond struct {
	L Locker // held while observing or changing the condition
	m Mutex  // held to avoid internal races

	// We must be careful to make sure that when Signal
	// releases a semaphore, the corresponding acquire is
	// executed by a goroutine that was already waiting at
	// the time of the call to Signal, not one that arrived later.
	// To ensure this, we segment waiting goroutines into
	// generations punctuated by calls to Signal.  Each call to
	// Signal begins another generation if there are no goroutines
	// left in older generations for it to wake.  Because of this
	// optimization (only begin another generation if there
	// are no older goroutines left), we only need to keep track
	// of the two most recent generations, which we call old
	// and new.
	oldWaiters int     // number of waiters in old generation...
	oldSema    *uint32 // ... waiting on this semaphore

	newWaiters int     // number of waiters in new generation...
	newSema    *uint32 // ... waiting on this semaphore
}

// NewCond returns a new Cond with Locker l.
func NewCond(l Locker) *Cond {
	return &Cond{L: l}
}

// Wait atomically unlocks c.L and suspends execution
// of the calling goroutine.  After later resuming execution,
// Wait locks c.L before returning.  Unlike in other systems,
// Wait cannot return unless awoken by Broadcast or Signal.
//
// Because c.L is not locked when Wait first resumes, the caller
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
	if raceenabled {
		raceDisable()
	}
	c.m.Lock()
	if c.newSema == nil {
		c.newSema = new(uint32)
	}
	s := c.newSema
	c.newWaiters++
	c.m.Unlock()
	if raceenabled {
		raceEnable()
	}
	c.L.Unlock()
	runtime_Semacquire(s)
	c.L.Lock()
}

// Signal wakes one goroutine waiting on c, if there is any.
//
// It is allowed but not required for the caller to hold c.L
// during the call.
func (c *Cond) Signal() {
	if raceenabled {
		raceDisable()
	}
	c.m.Lock()
	if c.oldWaiters == 0 && c.newWaiters > 0 {
		// Retire old generation; rename new to old.
		c.oldWaiters = c.newWaiters
		c.oldSema = c.newSema
		c.newWaiters = 0
		c.newSema = nil
	}
	if c.oldWaiters > 0 {
		c.oldWaiters--
		runtime_Semrelease(c.oldSema)
	}
	c.m.Unlock()
	if raceenabled {
		raceEnable()
	}
}

// Broadcast wakes all goroutines waiting on c.
//
// It is allowed but not required for the caller to hold c.L
// during the call.
func (c *Cond) Broadcast() {
	if raceenabled {
		raceDisable()
	}
	c.m.Lock()
	// Wake both generations.
	if c.oldWaiters > 0 {
		for i := 0; i < c.oldWaiters; i++ {
			runtime_Semrelease(c.oldSema)
		}
		c.oldWaiters = 0
	}
	if c.newWaiters > 0 {
		for i := 0; i < c.newWaiters; i++ {
			runtime_Semrelease(c.newSema)
		}
		c.newWaiters = 0
		c.newSema = nil
	}
	c.m.Unlock()
	if raceenabled {
		raceEnable()
	}
}
