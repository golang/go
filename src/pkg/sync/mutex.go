// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The sync package provides basic synchronization primitives
// such as mutual exclusion locks.  Other than the Once type,
// most are intended for use by low-level library routines.
// Higher-level synchronization  is better done via channels
// and communication.
package sync

import "runtime"

func cas(val *uint32, old, new uint32) bool

// A Mutex is a mutual exclusion lock.
// Mutexes can be created as part of other structures;
// the zero value for a Mutex is an unlocked mutex.
type Mutex struct {
	key  uint32
	sema uint32
}

// Add delta to *val, and return the new *val in a thread-safe way. If multiple
// goroutines call xadd on the same val concurrently, the changes will be
// serialized, and all the deltas will be added in an undefined order.
func xadd(val *uint32, delta int32) (new uint32) {
	for {
		v := *val
		nv := v + uint32(delta)
		if cas(val, v, nv) {
			return nv
		}
	}
	panic("unreached")
}

// Lock locks m.
// If the lock is already in use, the calling goroutine
// blocks until the mutex is available.
func (m *Mutex) Lock() {
	if xadd(&m.key, 1) == 1 {
		// changed from 0 to 1; we hold lock
		return
	}
	runtime.Semacquire(&m.sema)
}

// Unlock unlocks m.
// It is a run-time error if m is not locked on entry to Unlock.
//
// A locked Mutex is not associated with a particular goroutine.
// It is allowed for one goroutine to lock a Mutex and then
// arrange for another goroutine to unlock it.
func (m *Mutex) Unlock() {
	if xadd(&m.key, -1) == 0 {
		// changed from 1 to 0; no contention
		return
	}
	runtime.Semrelease(&m.sema)
}
