// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64) && linux

package runtime

import (
	"internal/goarch"
	"unsafe"
)

//go:linkname secret_count runtime/secret.count
func secret_count() int32 {
	return getg().secret
}

//go:linkname secret_inc runtime/secret.inc
func secret_inc() {
	gp := getg()
	gp.secret++
}

//go:linkname secret_dec runtime/secret.dec
func secret_dec() {
	gp := getg()
	gp.secret--
}

//go:linkname secret_eraseSecrets runtime/secret.eraseSecrets
func secret_eraseSecrets() {
	// zero all the stack memory that might be dirtied with
	// secrets. We do this from the systemstack so that we
	// don't have to figure out which holes we have to keep
	// to ensure that we can return from memclr. gp.sched will
	// act as a pigeonhole for our actual return.
	lo := getg().stack.lo
	systemstack(func() {
		// Note, this systemstack call happens within the secret mode,
		// so we don't have to call out to erase our registers, the systemstack
		// code will do that.
		mp := acquirem()
		sp := mp.curg.sched.sp
		// we need to keep systemstack return on top of the stack being cleared
		// for traceback
		sp -= goarch.PtrSize
		// TODO: keep some sort of low water mark so that we don't have
		// to zero a potentially large stack if we used just a little
		// bit of it. That will allow us to use a higher value for
		// lo than gp.stack.lo.
		memclrNoHeapPointers(unsafe.Pointer(lo), sp-lo)
		releasem(mp)
	})
	// Don't put any code here: the stack frame's contents are gone!
}

// addSecret records the fact that we need to zero p immediately
// when it is freed.
func addSecret(p unsafe.Pointer, size uintptr) {
	// TODO(dmo): figure out the cost of these. These are mostly
	// intended to catch allocations that happen via the runtime
	// that the user has no control over and not big buffers that user
	// code is allocating. The cost should be relatively low,
	// but we have run into a wall with other special allocations before.
	lock(&mheap_.speciallock)
	s := (*specialSecret)(mheap_.specialSecretAlloc.alloc())
	s.special.kind = _KindSpecialSecret
	s.size = size
	unlock(&mheap_.speciallock)
	addspecial(p, &s.special, false)
}

// send a no-op signal to an M for the purposes of
// clobbering the signal stack
//
// Use sigpreempt. If we don't have a preemption queued, this just
// turns into a no-op
func noopSignal(mp *m) {
	signalM(mp, sigPreempt)
}

// secret_getStack returns the memory range of the
// current goroutine's stack.
// For testing only.
// Note that this is kind of tricky, as the goroutine can
// be copied and/or exit before the result is used, at which
// point it may no longer be valid.
//
//go:linkname secret_getStack runtime/secret.getStack
func secret_getStack() (uintptr, uintptr) {
	gp := getg()
	return gp.stack.lo, gp.stack.hi
}

// return a slice of all Ms signal stacks
// For testing only.
//
//go:linkname secret_appendSignalStacks runtime/secret.appendSignalStacks
func secret_appendSignalStacks(sigstacks []stack) []stack {
	// This is probably overkill, but it's what
	// doAllThreadsSyscall does
	stw := stopTheWorld(stwAllThreadsSyscall)
	allocmLock.lock()
	acquirem()
	for mp := allm; mp != nil; mp = mp.alllink {
		sigstacks = append(sigstacks, mp.gsignal.stack)
	}
	releasem(getg().m)
	allocmLock.unlock()
	startTheWorld(stw)
	return sigstacks
}
