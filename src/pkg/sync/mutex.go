// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sync provides basic synchronization primitives such as mutual
// exclusion locks.  Other than the Once and WaitGroup types, most are intended
// for use by low-level library routines.  Higher-level synchronization is
// better done via channels and communication.
package sync

import (
	"runtime"
	"sync/atomic"
)

// A Mutex is a mutual exclusion lock.
// Mutexes can be created as part of other structures;
// the zero value for a Mutex is an unlocked mutex.
type Mutex struct {
	key  int32
	sema uint32
}

// A Locker represents an object that can be locked and unlocked.
type Locker interface {
	Lock()
	Unlock()
}

// Lock locks m.
// If the lock is already in use, the calling goroutine
// blocks until the mutex is available.
func (m *Mutex) Lock() {
	if atomic.AddInt32(&m.key, 1) == 1 {
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
	switch v := atomic.AddInt32(&m.key, -1); {
	case v == 0:
		// changed from 1 to 0; no contention
		return
	case v == -1:
		// changed from 0 to -1: wasn't locked
		// (or there are 4 billion goroutines waiting)
		panic("sync: unlock of unlocked mutex")
	}
	runtime.Semrelease(&m.sema)
}
