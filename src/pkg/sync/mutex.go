// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The sync package provides basic synchronization primitives
// such as mutual exclusion locks.  These are intended for use
// by low-level library routines.  Higher-level synchronization
// is better done via channels and communication.
package sync

import "runtime"

func cas(val *uint32, old, new uint32) bool

// A Mutex is a mutual exclusion lock.
// Mutexes can be created as part of other structures;
// the zero value for a Mutex is an unlocked mutex.
type Mutex struct {
	key	uint32;
	sema	uint32;
}

func xadd(val *uint32, delta int32) (new uint32) {
	for {
		v := *val;
		nv := v+uint32(delta);
		if cas(val, v, nv) {
			return nv;
		}
	}
	panic("unreached");
}

// Lock locks m.
// If the lock is already in use, the calling goroutine
// blocks until the mutex is available.
func (m *Mutex) Lock() {
	if xadd(&m.key, 1) == 1 {
		// changed from 0 to 1; we hold lock
		return;
	}
	runtime.Semacquire(&m.sema);
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
		return;
	}
	runtime.Semrelease(&m.sema);
}

// Stub implementation of r/w locks.
// This satisfies the semantics but
// is not terribly efficient.

// The next comment goes in the BUGS section of the document,
// in its own paragraph, without the (rsc) tag.

// BUG(rsc): RWMutex does not (yet) allow multiple readers;
// instead it behaves as if RLock and RUnlock were Lock and Unlock.

// An RWMutex is a reader/writer mutual exclusion lock.
// The lock can be held by an arbitrary number of readers
// or a single writer.
// RWMutexes can be created as part of other
// structures; the zero value for a RWMutex is
// an unlocked mutex.
type RWMutex struct {
	m Mutex;
}

// RLock locks rw for reading.
// If the lock is already locked for writing or there is a writer already waiting
// to acquire the lock, RLock blocks until the writer has released the lock.
func (rw *RWMutex) RLock()	{ rw.m.Lock() }

// RUnlock undoes a single RLock call;
// it does not affect other simultaneous readers.
// It is a run-time error if rw is not locked for reading
// on entry to RUnlock.
func (rw *RWMutex) RUnlock()	{ rw.m.Unlock() }

// Lock locks rw for writing.
// If the lock is already locked for reading or writing,
// Lock blocks until the lock is available.
// To ensure that the lock eventually becomes available,
// a blocked Lock call excludes new readers from acquiring
// the lock.
func (rw *RWMutex) Lock()	{ rw.m.Lock() }

// Unlock unlocks rw for writing.
// It is a run-time error if rw is not locked for writing
// on entry to Unlock.
//
// Like for Mutexes,
// a locked RWMutex is not associated with a particular goroutine.
// It is allowed for one goroutine to RLock (Lock) an RWMutex and then
// arrange for another goroutine to RUnlock (Unlock) it.
func (rw *RWMutex) Unlock()	{ rw.m.Unlock() }
