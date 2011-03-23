// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import "runtime"

// A WaitGroup waits for a collection of goroutines to finish.
// The main goroutine calls Add to set the number of
// goroutines to wait for.  Then each of the goroutines
// runs and calls Done when finished.  At the same time,
// Wait can be used to block until all goroutines have finished.
//
// For example:
//
//   for i := 0; i < n; i++ {
//       if !condition(i) {
//           continue
//       }
//       wg.Add(1)
//       go func() {
//           // Do something.
//           wg.Done()
//       }()
//   }
//   wg.Wait()
// 
type WaitGroup struct {
	m       Mutex
	counter int
	waiters int
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
func (wg *WaitGroup) Add(delta int) {
	wg.m.Lock()
	if delta < -wg.counter {
		wg.m.Unlock()
		panic("sync: negative WaitGroup count")
	}
	wg.counter += delta
	if wg.counter == 0 && wg.waiters > 0 {
		for i := 0; i < wg.waiters; i++ {
			runtime.Semrelease(wg.sema)
		}
		wg.waiters = 0
		wg.sema = nil
	}
	wg.m.Unlock()
}

// Done decrements the WaitGroup counter.
func (wg *WaitGroup) Done() {
	wg.Add(-1)
}

// Wait blocks until the WaitGroup counter is zero.
func (wg *WaitGroup) Wait() {
	wg.m.Lock()
	if wg.counter == 0 {
		wg.m.Unlock()
		return
	}
	wg.waiters++
	if wg.sema == nil {
		wg.sema = new(uint32)
	}
	s := wg.sema
	wg.m.Unlock()
	runtime.Semacquire(s)
}
