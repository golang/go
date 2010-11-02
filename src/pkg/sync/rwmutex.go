// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

// An RWMutex is a reader/writer mutual exclusion lock.
// The lock can be held by an arbitrary number of readers
// or a single writer.
// RWMutexes can be created as part of other
// structures; the zero value for a RWMutex is
// an unlocked mutex.
//
// Writers take priority over Readers: no new RLocks
// are granted while a blocked Lock call is waiting.
type RWMutex struct {
	w           Mutex  // held if there are pending readers or writers
	r           Mutex  // held if the w is being rd
	readerCount uint32 // number of pending readers
}

// RLock locks rw for reading.
// If the lock is already locked for writing or there is a writer already waiting
// to release the lock, RLock blocks until the writer has released the lock.
func (rw *RWMutex) RLock() {
	// Use rw.r.Lock() to block granting the RLock if a goroutine
	// is waiting for its Lock. This is the prevent starvation of W in
	// this situation:
	//   A: rw.RLock()  // granted
	//   W: rw.Lock()   // waiting for rw.w().Lock()
	//   B: rw.RLock()  // granted
	//   C: rw.RLock()  // granted
	//   B: rw.RUnlock()
	//   ... (new readers come and go indefinitely, W is starving)
	rw.r.Lock()
	if xadd(&rw.readerCount, 1) == 1 {
		// The first reader locks rw.w, so writers will be blocked
		// while the readers have the RLock.
		rw.w.Lock()
	}
	rw.r.Unlock()
}

// RUnlock undoes a single RLock call;
// it does not affect other simultaneous readers.
// It is a run-time error if rw is not locked for reading
// on entry to RUnlock.
func (rw *RWMutex) RUnlock() {
	if xadd(&rw.readerCount, -1) == 0 {
		// last reader finished, enable writers
		rw.w.Unlock()
	}
}

// Lock locks rw for writing.
// If the lock is already locked for reading or writing,
// Lock blocks until the lock is available.
// To ensure that the lock eventually becomes available,
// a blocked Lock call excludes new readers from acquiring
// the lock.
func (rw *RWMutex) Lock() {
	rw.r.Lock()
	rw.w.Lock()
	rw.r.Unlock()
}

// Unlock unlocks rw for writing.
// It is a run-time error if rw is not locked for writing
// on entry to Unlock.
//
// Like for Mutexes,
// a locked RWMutex is not associated with a particular goroutine.
// It is allowed for one goroutine to RLock (Lock) an RWMutex and then
// arrange for another goroutine to RUnlock (Unlock) it.
func (rw *RWMutex) Unlock() { rw.w.Unlock() }
