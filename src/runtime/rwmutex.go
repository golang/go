// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
)

// This is a copy of sync/rwmutex.go rewritten to work in the runtime.

// An rwmutex is a reader/writer mutual exclusion lock.
// The lock can be held by an arbitrary number of readers or a single writer.
// This is a variant of sync.RWMutex, for the runtime package.
// This is less convenient than sync.RWMutex, because it must be
// initialized before use. Sorry.
type rwmutex struct {
	w           uint32 // semaphore for pending writers
	writerSem   uint32 // semaphore for writers to wait for completing readers
	readerSem   uint32 // semaphore for readers to wait for completing writers
	readerCount uint32 // number of pending readers
	readerWait  uint32 // number of departing readers
}

const rwmutexMaxReaders = 1 << 30

// init initializes rw. This must be called before any other methods.
func (rw *rwmutex) init() {
	rw.w = 1
}

// rlock locks rw for reading.
func (rw *rwmutex) rlock() {
	if int32(atomic.Xadd(&rw.readerCount, 1)) < 0 {
		// A writer is pending.
		semacquire(&rw.readerSem)
	}
}

// runlock undoes a single rlock call on rw.
func (rw *rwmutex) runlock() {
	if r := int32(atomic.Xadd(&rw.readerCount, -1)); r < 0 {
		if r+1 == 0 || r+1 == -rwmutexMaxReaders {
			throw("runlock of unlocked rwmutex")
		}
		// A writer is pending.
		if atomic.Xadd(&rw.readerWait, -1) == 0 {
			// The last reader unblocks the writer.
			semrelease(&rw.writerSem)
		}
	}
}

// lock locks rw for writing.
func (rw *rwmutex) lock() {
	// Resolve competition with other writers.
	semacquire(&rw.w)
	// Announce that there is a pending writer.
	r := int32(atomic.Xadd(&rw.readerCount, -rwmutexMaxReaders)) + rwmutexMaxReaders
	// Wait for any active readers to complete.
	if r != 0 && atomic.Xadd(&rw.readerWait, r) != 0 {
		semacquire(&rw.writerSem)
	}
}

// unlock unlocks rw for writing.
func (rw *rwmutex) unlock() {
	// Announce to readers that there is no active writer.
	r := int32(atomic.Xadd(&rw.readerCount, rwmutexMaxReaders))
	if r >= rwmutexMaxReaders {
		throw("unlock of unlocked rwmutex")
	}
	// Unblock blocked readers, if any.
	for i := int32(0); i < r; i++ {
		semrelease(&rw.readerSem)
	}
	// Allow other writers to proceed.
	semrelease(&rw.w)
}
