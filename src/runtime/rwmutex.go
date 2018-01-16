// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
)

// This is a copy of sync/rwmutex.go rewritten to work in the runtime.

// A rwmutex is a reader/writer mutual exclusion lock.
// The lock can be held by an arbitrary number of readers or a single writer.
// This is a variant of sync.RWMutex, for the runtime package.
// Like mutex, rwmutex blocks the calling M.
// It does not interact with the goroutine scheduler.
type rwmutex struct {
	rLock      mutex    // protects readers, readerPass, writer
	readers    muintptr // list of pending readers
	readerPass uint32   // number of pending readers to skip readers list

	wLock  mutex    // serializes writers
	writer muintptr // pending writer waiting for completing readers

	readerCount uint32 // number of pending readers
	readerWait  uint32 // number of departing readers
}

const rwmutexMaxReaders = 1 << 30

// rlock locks rw for reading.
func (rw *rwmutex) rlock() {
	// The reader must not be allowed to lose its P or else other
	// things blocking on the lock may consume all of the Ps and
	// deadlock (issue #20903). Alternatively, we could drop the P
	// while sleeping.
	acquirem()
	if int32(atomic.Xadd(&rw.readerCount, 1)) < 0 {
		// A writer is pending. Park on the reader queue.
		systemstack(func() {
			lock(&rw.rLock)
			if rw.readerPass > 0 {
				// Writer finished.
				rw.readerPass -= 1
				unlock(&rw.rLock)
			} else {
				// Queue this reader to be woken by
				// the writer.
				m := getg().m
				m.schedlink = rw.readers
				rw.readers.set(m)
				unlock(&rw.rLock)
				notesleep(&m.park)
				noteclear(&m.park)
			}
		})
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
			lock(&rw.rLock)
			w := rw.writer.ptr()
			if w != nil {
				notewakeup(&w.park)
			}
			unlock(&rw.rLock)
		}
	}
	releasem(getg().m)
}

// lock locks rw for writing.
func (rw *rwmutex) lock() {
	// Resolve competition with other writers and stick to our P.
	lock(&rw.wLock)
	m := getg().m
	// Announce that there is a pending writer.
	r := int32(atomic.Xadd(&rw.readerCount, -rwmutexMaxReaders)) + rwmutexMaxReaders
	// Wait for any active readers to complete.
	lock(&rw.rLock)
	if r != 0 && atomic.Xadd(&rw.readerWait, r) != 0 {
		// Wait for reader to wake us up.
		systemstack(func() {
			rw.writer.set(m)
			unlock(&rw.rLock)
			notesleep(&m.park)
			noteclear(&m.park)
		})
	} else {
		unlock(&rw.rLock)
	}
}

// unlock unlocks rw for writing.
func (rw *rwmutex) unlock() {
	// Announce to readers that there is no active writer.
	r := int32(atomic.Xadd(&rw.readerCount, rwmutexMaxReaders))
	if r >= rwmutexMaxReaders {
		throw("unlock of unlocked rwmutex")
	}
	// Unblock blocked readers.
	lock(&rw.rLock)
	for rw.readers.ptr() != nil {
		reader := rw.readers.ptr()
		rw.readers = reader.schedlink
		reader.schedlink.set(nil)
		notewakeup(&reader.park)
		r -= 1
	}
	// If r > 0, there are pending readers that aren't on the
	// queue. Tell them to skip waiting.
	rw.readerPass += uint32(r)
	unlock(&rw.rLock)
	// Allow other writers to proceed.
	unlock(&rw.wLock)
}
