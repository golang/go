// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: //go:build racelite

package runtime

import (
	"unsafe"
)

// TODO(thepudds): we currently have a ~3 level scheme of lighter checks before doing more work.
// The intent is the first check could be emitted by the compiler.
// And in general, we certainly can improve from this very first cut.

const (
	// raceliteShift is the number of bits to shift the
	// address. We use raceliteShift to compute the number of
	// virtual registers we have.
	raceliteShift = 4

	// raceliteRegNum is the number of virtual registers we have.
	// It is a power of 2.
	raceliteRegNum = 1 << raceliteShift

	// raceliteCheckAddrMask selects an address suffix which
	// can be monitored for racelite.
	raceliteCheckAddrMask = (1 << (4 * raceliteShift)) - 1
)

var (
	// raceliteCheckAddrRand allows us to randomize which addresses we check for racelite,
	// but to do so in a way we get a consistent answer per address across concurrent readers
	// and writers.
	//
	// TODO(thepudds): for now, for convenience we only check this in runtime, but we
	// could have the compiler emit the check -- probably important to do or at least
	// try if were were to pursue this general approach.
	raceliteCheckAddrRand uint32 = 0
)

// raceliteVirtualRegister is a virtual register that can be used to
// dynamically monitor an address for data races.
type raceliteVirtualRegister struct {
	lock       mutex          // lock ensures that the virtual register maintains a consistent state
	addr       uintptr        // the address currently being monitored in the virtual register
	owner      unsafe.Pointer // the writer currently claiming the virtual register
	count      uint64         // the number of times the virtual register has been claimed
	identifier uint32         // the identifier of the virtual register
}

var raceliteReg [raceliteRegNum]*raceliteVirtualRegister

func raceliteinit() {
	if debug.racelite <= 0 {
		return
	}
	for i := 0; i < raceliteRegNum; i++ {
		raceliteReg[i] = new(raceliteVirtualRegister)
		raceliteReg[i].identifier = uint32(i)
	}
}

func raceliteget(addr uintptr) *raceliteVirtualRegister {
	return raceliteReg[(addr>>3)&(raceliteRegNum-1)]
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
	switch r.addr {
	case 0:
		// The virtual register is not occupied.
		// The writer can claim it.
		r.addr = addr
		r.owner = unsafe.Pointer(gp)
		r.count++
		unlock(&r.lock)
		return true

	case addr:
		// The virtual register is already occupied by the same address.
		// This only happens if another writer intercepted the write.
		// We are, therefore, dealing with a write-write race.

		// Store up the owner of the virtual register before unlocking.
		gp2 := (*g)(r.owner)
		// We must release the lock in order to stop the world.
		unlock(&r.lock)

		// Write the stacks of the current goroutine and the
		// virtual register claimant on the system stack.
		//
		// We stop the world to prevent jumbling the stacks.
		stw := stopTheWorld(stwRacelite)

		// We need to move to the system stack to walk
		// the stack of the current goroutine.
		systemstack(func() {
			print("VR", r.identifier, " RACELITE TRIGGERED: write-write race at",
				" addr= ", hex(addr),
				"\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
			print("\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp2)
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
	if r.addr != addr {
		// The virtual register is occupied by another address.
		unlock(&r.lock)
		return false
	}

	// The virtual register is occupied by the same address.
	// Store up the owner of the virtual register before unlocking.
	gp2 := (*g)(r.owner)

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
			print("VR", r.identifier, " RACELITE TRIGGERED: read-write race at",
				" addr= ", hex(addr),
				"\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
			println()
			traceback(^uintptr(0), ^uintptr(0), 0, gp2)
		} else {
			print("VR", r.identifier, " RACELITE TRIGGERED: write-read race at",
				" addr= ", hex(addr),
				"\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp2)
			println()
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
		}
		println("RACELITE END")
	})

	startTheWorld(stw)
	return true
}

// release allows the writer goroutine to release the virtual register.
func (r *raceliteVirtualRegister) release() {
	lock(&r.lock)
	r.addr = 0
	r.owner = nil
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
	gp, r := getg(), raceliteget(addr)

	// Check the status of the virtual register.
	if !r.claim(addr, gp) {
		// We did not claim the virtual register.
		// It was either claimed by another writer, or
		// we had a write-write race.
		return
	}

	micropause()

	// Release the virtual register.
	r.release()
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
	gp, r := getg(), raceliteget(addr)

	if r.monitor(addr, gp, false) {
		// We had a write-read race.
		return
	}

	micropause()

	// Check for a read-write race.
	r.monitor(addr, gp, true)
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
	if !diag() && uint32(addr>>(3+raceliteShift))&raceliteCheckAddrMask != raceliteCheckAddrRand {
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
	mp := acquirem()
	inStack := mp.curg.stack.lo <= addr && addr < mp.curg.stack.hi
	releasem(mp)
	return inStack
}
