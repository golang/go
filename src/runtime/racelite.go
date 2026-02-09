// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: //go:build racelite

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

// TODO(thepudds): we currently have a ~3 level scheme of lighter checks before doing more work.
// The intent is the first check could be emitted by the compiler.
// And in general, we certainly can improve from this very first cut.

const (
	// raceliteIdle is the state of a virtual register when it is not being written to.
	raceliteIdle = 0
	// raceliteWriting is the state of a virtual register when it is being written to.
	raceliteWriting = 1
)

var (
	// raceliteCheckAddrMask controls how often we check addresses for racelite.
	// Currently, for proof-of-concept, we set to check 1 out of 16 addresses to make
	// results come back very quickly, but a "real" system could be much higher.
	// We would pick a higher value to keep overhead manageable.
	raceliteCheckAddrMask = (1 << 4) - 1

	// raceliteCheckAddrRand allows us to randomize which addresses we check for racelite,
	// but to do so in a way we get a consistent answer per address across concurrent readers
	// and writers.
	//
	// TODO(thepudds): for now, for convenience we only check this in runtime, but we
	// could have the compiler emit the check -- probably important to do or at least
	// try if were were to pursue this general approach.
	raceliteCheckAddrRand uint32 = 0

	// raceliteCheckWordRand allows us to randomize which offset we check for racelite within
	// a given object, which helps prevent us from having different words from the same object
	// interfere with each other.
	raceliteCheckWordRand uint32 = 0

	// raceliteCheckAddrMask = (1 << 10) - 1 // check 1 out of 1024 addresses
)

// raceliteVirtualRegister is a virtual register that can be used to store an address.
type raceliteVirtualRegister struct {
	lock  mutex                // lock ensures that the virtual register maintains a consistent state
	addr  atomic.Uintptr       // the address currently being monitored in the virtual register
	owner atomic.UnsafePointer // the writer currently claiming the virtual register
}

var raceliteReg *raceliteVirtualRegister = new(raceliteVirtualRegister)

func (r *raceliteVirtualRegister) goid() uint64 {
	gp := (*g)(r.owner.Load())
	if gp == nil {
		return 0
	}
	return gp.goid
}

func micropause() {
	// Now record our write, followed by a small delay to let someone see it.
	// We either do a ~nano delay or a ~1 in 100 chance of a ~micro delay.
	// The ~nano delay is from just checking cheaprand.
	// good with 100k samples!
	if diag() || cheaprandn(10_000) == 0 {
		// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
		// let's us see it work for the first time. ;)
		usleep(2)
	}
	if diag() {
		Gosched() // FIXME: Remove this after testing.
	}
}

// claim allows the writer goroutine as the owner of the virtual register.
func (r *raceliteVirtualRegister) claim(addr uintptr, gp *g) bool {

	lock(&r.lock)
	switch r.addr.Load() {
	case 0:
		// The virtual register is not occupied.
		// The writer can claim it.
		r.addr.Store(addr)
		r.owner.Store(unsafe.Pointer(gp))
		unlock(&r.lock)
		return true

	case addr:
		// The virtual register is already occupied by the same address.
		// This only happens if another writer intercepted the write.
		// We are, therefore, dealing with a write-write race.
		unlock(&r.lock)

		// Write the stacks of the current goroutine and the
		// virtual register claimant on the system stack.
		//
		// We stop the world to prevent jumbling the stacks.
		stw := stopTheWorld(stwRacelite)

		// We need to move to the system stack to walk
		// the stack of the current goroutine.
		systemstack(func() {
			print("RACELITE TRIGGERED: write-write race at",
				" addr= ", hex(addr),
				" goid=", gp.goid,
				"\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
			print("\n")
			if gp2 := (*g)(raceliteReg.owner.Load()); gp2 != nil {
				traceback(^uintptr(0), ^uintptr(0), 0, gp2)
			} else {
				print("writer: stack lost\n")
			}
			println("RACELITE END")
		})

		startTheWorld(stw)
		return false

	default:
		// The virtual register is already occupied by another address.
		unlock(&r.lock)
		return false
	}
}

// monitor allows a reader goroutine to check whether the virtual
// register is occupied by a writer with the given address.
func (r *raceliteVirtualRegister) monitor(addr uintptr, gp *g, readfirst bool) bool {
	lock(&r.lock)

	// Check the status of the virtual register.
	if raceliteReg.addr.Load() == addr {
		// The virtual register is already occupied by the same address.
		// This can only occurr if another writer intercepted the write.
		// Check whether we are being written to.
		unlock(&r.lock)

		// Write the stacks of the current goroutine and the
		// virtual register claimant on the system stack.
		//
		// We stop the world to prevent jumbling the stacks.
		stw := stopTheWorld(stwRacelite)

		// We need to move to the system stack to walk
		// the stack of the current goroutine.
		systemstack(func() {
			if readfirst {
				print("RACELITE TRIGGERED: read-write race at",
					" addr= ", hex(addr),
					" goid= ", gp.goid,
					"\n")
			} else {
				print("RACELITE TRIGGERED: write-read race at",
					" addr= ", hex(addr),
					" goid= ", gp.goid,
					"\n")
			}
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
			println()
			if gp2 := (*g)(raceliteReg.owner.Load()); gp2 != nil {
				traceback(^uintptr(0), ^uintptr(0), 0, gp2)
			} else {
				print("writer: stack lost\n")
			}
			println("RACELITE END")
		})

		startTheWorld(stw)
		return true
	}
	unlock(&r.lock)
	return false
}

// release allows the writer goroutine to release the virtual register.
func (r *raceliteVirtualRegister) release() {
	lock(&r.lock)
	r.addr.Store(0)
	r.owner.Store(nil)
	unlock(&r.lock)
}

// racelitewrite instruments a store operation with data race detection,
// according to the following logic:
//
//	Let A be the stored address
//	Let V be a virtual register that can track A, where
//	- V.claimed(A) checks if the virtual register is already claimed for A
//	- V.claim(A) claims the virtual register for the writer
//	- V.release() releases the virtual register
//
//	if V.claimed(A):
//		Report write-write race
//		return
//	V.claim(A)
//	micropause()
//	V.release()
func racelitewrite(addr uintptr) {
	if !raceliteCheckAddr(addr) {
		// We are not sampling this address.
		return
	}
	gp := getg()

	// Check the status of the virtual register.
	if !raceliteReg.claim(addr, gp) {
		// We did not claim the virtual register.
		// It was either claimed by another writer, or
		// we had a write-write race.
		return
	}

	if diag() {
		print("RACELITE write: claimed virtual register\n",
			"Virtual register:",
			" addr=", hex(addr),
			" reg.addr=", hex(raceliteReg.addr.Load()),
			" owner=", raceliteReg.goid(),
			" goid=", gp.goid,
			"\n")
	}

	micropause()

	// Release the virtual register.
	raceliteReg.release()
	if diag() {
		print("RACELITE write: released virtual register\n",
			"Virtual register:",
			" addr=", hex(addr),
			" reg.addr=", hex(raceliteReg.addr.Load()),
			" owner=", raceliteReg.goid(),
			" goid=", gp.goid,
			"\n")
	}
}

// raceliteread instruments a load operation with data race detection,
// according to the following logic:
//
//	Let A be the loaded address
//	Let V be a virtual register that can track A, where
//	- V.claimed(A) checks if the virtual register is already claimed for A
//
//	if V.claimed(A):
//		Report write-read race
//		return
//	micropause()
//	if V.claimed(A):
//		Report read-write race
func raceliteread(addr uintptr) {
	if !raceliteCheckAddr(addr) {
		// We are not sampling this address.
		return
	}
	gp := getg()

	if raceliteReg.monitor(addr, gp, false) {
		// We had a write-read race.
		return
	}

	micropause()

	// Check for a read-write race.
	raceliteReg.monitor(addr, gp, true)
}

func raceliteCheckAddr(addr uintptr) bool {
	// Check that this address is not on our stack.
	if inStack(addr) {
		// Ignore stack addresses
		return false
	}

	// Check that we are sampling this address.
	//
	// FIXME(vsaioc): At the moment, we always sample when debugging.
	if !diag() && !(uint32(addr>>3)^raceliteCheckAddrRand == 0) {
		// Not checking this addr. This is the common case.
		if diag() {
			print("RACELITE write: not checking addr=", hex(addr), "\n")
		}
		return false
	}

	// Extract heap object.
	base, span, _ := findObject(uintptr(addr), 0, 0)
	if span == nil && base == 0 {
		// We got some bad pointer. We don't care.
		return false
	}

	return true
}

// diag reports true is we should print extra info.
func diag() bool {
	return debug.racelite >= 2
}

// inStack checks if addr is in the current goroutine's stack.
func inStack(addr uintptr) bool {
	// TODO(thepudds): probably want to make sure we are not preempted.
	// TODO(vsaioc): is the check through mp.preemptoff or mp.curg.preempt?
	mp := acquirem()
	inStack := mp.curg.stack.lo <= addr && addr < mp.curg.stack.hi
	releasem(mp)
	return inStack
}
