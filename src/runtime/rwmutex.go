// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
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

	readerCount atomic.Int32 // number of pending readers
	readerWait  atomic.Int32 // number of departing readers

	readRank lockRank // semantic lock rank for read locking
}

// Lock ranking an rwmutex has two aspects:
//
// Semantic ranking: this rwmutex represents some higher level lock that
// protects some resource (e.g., allocmLock protects creation of new Ms). The
// read and write locks of that resource need to be represented in the lock
// rank.
//
// Internal ranking: as an implementation detail, rwmutex uses two mutexes:
// rLock and wLock. These have lock order requirements: wLock must be locked
// before rLock. This also needs to be represented in the lock rank.
//
// Semantic ranking is represented by acquiring readRank during read lock and
// writeRank during write lock.
//
// wLock is held for the duration of a write lock, so it uses writeRank
// directly, both for semantic and internal ranking. rLock is only held
// temporarily inside the rlock/lock methods, so it uses readRankInternal to
// represent internal ranking. Semantic ranking is represented by a separate
// acquire of readRank for the duration of a read lock.
//
// The lock ranking must document this ordering:
//   - readRankInternal is a leaf lock.
//   - readRank is taken before readRankInternal.
//   - writeRank is taken before readRankInternal.
//   - readRank is placed in the lock order wherever a read lock of this rwmutex
//     belongs.
//   - writeRank is placed in the lock order wherever a write lock of this
//     rwmutex belongs.
func (rw *rwmutex) init(readRank, readRankInternal, writeRank lockRank) {
	rw.readRank = readRank

	lockInit(&rw.rLock, readRankInternal)
	lockInit(&rw.wLock, writeRank)
}

const rwmutexMaxReaders = 1 << 30

// rlock locks rw for reading.
func (rw *rwmutex) rlock() {
	// The reader must not be allowed to lose its P or else other
	// things blocking on the lock may consume all of the Ps and
	// deadlock (issue #20903). Alternatively, we could drop the P
	// while sleeping.
	acquireLockRankAndM(rw.readRank)
	lockWithRankMayAcquire(&rw.rLock, getLockRank(&rw.rLock))

	if rw.readerCount.Add(1) < 0 {
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
	if r := rw.readerCount.Add(-1); r < 0 {
		if r+1 == 0 || r+1 == -rwmutexMaxReaders {
			throw("runlock of unlocked rwmutex")
		}
		// A writer is pending.
		if rw.readerWait.Add(-1) == 0 {
			// The last reader unblocks the writer.
			lock(&rw.rLock)
			w := rw.writer.ptr()
			if w != nil {
				notewakeup(&w.park)
			}
			unlock(&rw.rLock)
		}
	}
	releaseLockRankAndM(rw.readRank)
}

// lock locks rw for writing.
func (rw *rwmutex) lock() {
	// Resolve competition with other writers and stick to our P.
	lock(&rw.wLock)
	m := getg().m
	// Announce that there is a pending writer.
	r := rw.readerCount.Add(-rwmutexMaxReaders) + rwmutexMaxReaders
	// Wait for any active readers to complete.
	lock(&rw.rLock)
	if r != 0 && rw.readerWait.Add(r) != 0 {
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
	r := rw.readerCount.Add(rwmutexMaxReaders)
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
