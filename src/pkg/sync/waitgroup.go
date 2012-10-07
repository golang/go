// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"sync/atomic"
	"unsafe"
)

// A WaitGroup waits for a collection of goroutines to finish.
// The main goroutine calls Add to set the number of
// goroutines to wait for.  Then each of the goroutines
// runs and calls Done when finished.  At the same time,
// Wait can be used to block until all goroutines have finished.
type WaitGroup struct {
	m       Mutex
	counter int32
	waiters int32
	sema    *uint32
}

// WaitGroup creates a new semaphore each time the old semaphore
// is released. This is to avoid the following race:
//
// G1: Add(1)
// G1: go G2()
// G1: Wait() // Context switch after Unlock() and before Semacquire().
// G2: Done() // Release semaphore: sema == 1, waiters == 0. G1 doesn't run yet.
// G3: Wait() // Finds counter == 0, waiters == 0, doesn't block.
// G3: Add(1) // Makes counter == 1, waiters == 0.
// G3: go G4()
// G3: Wait() // G1 still hasn't run, G3 finds sema == 1, unblocked! Bug.

// Add adds delta, which may be negative, to the WaitGroup counter.
// If the counter becomes zero, all goroutines blocked on Wait() are released.
// If the counter goes negative, Add panics.
func (wg *WaitGroup) Add(delta int) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(wg))
		raceDisable()
		defer raceEnable()
	}
	v := atomic.AddInt32(&wg.counter, int32(delta))
	if v < 0 {
		panic("sync: negative WaitGroup counter")
	}
	if v > 0 || atomic.LoadInt32(&wg.waiters) == 0 {
		return
	}
	wg.m.Lock()
	for i := int32(0); i < wg.waiters; i++ {
		runtime_Semrelease(wg.sema)
	}
	wg.waiters = 0
	wg.sema = nil
	wg.m.Unlock()
}

// Done decrements the WaitGroup counter.
func (wg *WaitGroup) Done() {
	wg.Add(-1)
}

// Wait blocks until the WaitGroup counter is zero.
func (wg *WaitGroup) Wait() {
	if raceenabled {
		raceDisable()
	}
	if atomic.LoadInt32(&wg.counter) == 0 {
		if raceenabled {
			raceEnable()
			raceAcquire(unsafe.Pointer(wg))
		}
		return
	}
	wg.m.Lock()
	atomic.AddInt32(&wg.waiters, 1)
	// This code is racing with the unlocked path in Add above.
	// The code above modifies counter and then reads waiters.
	// We must modify waiters and then read counter (the opposite order)
	// to avoid missing an Add.
	if atomic.LoadInt32(&wg.counter) == 0 {
		atomic.AddInt32(&wg.waiters, -1)
		if raceenabled {
			raceEnable()
			raceAcquire(unsafe.Pointer(wg))
			raceDisable()
		}
		wg.m.Unlock()
		if raceenabled {
			raceEnable()
		}
		return
	}
	if wg.sema == nil {
		wg.sema = new(uint32)
	}
	s := wg.sema
	wg.m.Unlock()
	runtime_Semacquire(s)
	if raceenabled {
		raceEnable()
		raceAcquire(unsafe.Pointer(wg))
	}
}
