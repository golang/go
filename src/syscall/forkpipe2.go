// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd || solaris

package syscall

import "sync"

// forkExecPipe atomically opens a pipe with O_CLOEXEC set on both file
// descriptors.
func forkExecPipe(p []int) error {
	return Pipe2(p, O_CLOEXEC)
}

var (
	// Guard the forking variable.
	forkingLock sync.Mutex
	// Number of goroutines currently forking, and thus the
	// number of goroutines holding a conceptual write lock
	// on ForkLock.
	forking int
)

// hasWaitingReaders reports whether any goroutine is waiting
// to acquire a read lock on rw. It is defined in the sync package.
func hasWaitingReaders(rw *sync.RWMutex) bool

// acquireForkLock acquires a write lock on ForkLock.
// ForkLock is exported and we've promised that during a fork
// we will call ForkLock.Lock, so that no other threads create
// new fds that are not yet close-on-exec before we fork.
// But that forces all fork calls to be serialized, which is bad.
// But we haven't promised that serialization, and it is essentially
// undetectable by other users of ForkLock, which is good.
// Avoid the serialization by ensuring that ForkLock is locked
// at the first fork and unlocked when there are no more forks.
func acquireForkLock() {
	forkingLock.Lock()
	defer forkingLock.Unlock()

	if forking == 0 {
		// There is no current write lock on ForkLock.
		ForkLock.Lock()
		forking++
		return
	}

	// ForkLock is currently locked for writing.

	if hasWaitingReaders(&ForkLock) {
		// ForkLock is locked for writing, and at least one
		// goroutine is waiting to read from it.
		// To avoid lock starvation, allow readers to proceed.
		// The simple way to do this is for us to acquire a
		// read lock. That will block us until all current
		// conceptual write locks are released.
		//
		// Note that this case is unusual on modern systems
		// with O_CLOEXEC and SOCK_CLOEXEC. On those systems
		// the standard library should never take a read
		// lock on ForkLock.

		forkingLock.Unlock()

		ForkLock.RLock()
		ForkLock.RUnlock()

		forkingLock.Lock()

		// Readers got a chance, so now take the write lock.

		if forking == 0 {
			ForkLock.Lock()
		}
	}

	forking++
}

// releaseForkLock releases the conceptual write lock on ForkLock
// acquired by acquireForkLock.
func releaseForkLock() {
	forkingLock.Lock()
	defer forkingLock.Unlock()

	if forking <= 0 {
		panic("syscall.releaseForkLock: negative count")
	}

	forking--

	if forking == 0 {
		// No more conceptual write locks.
		ForkLock.Unlock()
	}
}
