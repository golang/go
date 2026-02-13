// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: //go:build racelite

package runtime

import (
	"internal/runtime/atomic"
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

	raceliteSeparator = "==================\n"
	dataRaceHeader    = raceliteSeparator + "WARNING: DATA RACE\n"
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
	// Mutual exclusion ensures that the virtual register maintains a consistent state.
	// We do not use a runtime mutex here because we want to stop the world.
	// mu and rmu are used for a custom read-write spin-lock.
	mu, rmu atomic.Int32

	// addr is the address currently being monitored in the virtual register
	addr uintptr
	// owner is the writer goroutine currently claiming the virtual register
	owner *g

	// Diagnostic information.
	// TODO(vsaioc): Add more for experimentation purposes and remove
	// for the polished release version.
	count      uint64 // the number of times the virtual register has been claimed
	identifier uint32 // the identifier of the virtual register
}

// raceliteReg is the globalarray of virtual registers.
var raceliteReg *[raceliteRegNum]raceliteVirtualRegister

func raceliteinit() {
	if debug.racelite <= 0 {
		// No-op if racelite is not enabled.
		return
	}

	raceliteReg = new([raceliteRegNum]raceliteVirtualRegister)

	// Initialize all virtual registers.
	for i := 0; i < raceliteRegNum; i++ {
		// Assign an identifier to the virtual register.
		// This is used for debugging and experimentation.
		raceliteReg[i].identifier = uint32(i)
	}
}

func raceliteget(addr uintptr) *raceliteVirtualRegister {
	// Compile-time optimized to bitwise AND.
	return &raceliteReg[(addr>>3)%raceliteRegNum]
}

// micropause injects a small delay to a load
// or store operation, allowing Racelite to see
// whether another thread accessed the same address.
func micropause() {
	if diag() || cheaprandn(10_000) == 0 {
		// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
		// let's us see it work for the first time. ;)
		usleep(2)
	}
	// NOTE(vsaioc): We may want to experiment here.
	Gosched()
}

// report prints a data race report to the console.
// The reports match the format of the data race warnings
// issued by TSan (omitting ancestry information).
//
// Example:
//
//	==================
//	WARNING: DATA RACE
//	Write at 0x1234567890 by goroutine 123
//	  [stack trace of the writer]
//	Previous write at 0x1234567890 by goroutine 456
//	  [stack trace of the previous writer]
//
//	Diagnostic information:
//	VR(4): count=40
//	===^=========^====
//	   |         '-Number of times the virtual register has been claimed
//	   '-Virtual register identifier
func (r *raceliteVirtualRegister) report(addr uintptr, gp, g2 *g, op1, op2 string) {
	// Write the stacks of the current goroutine and the
	// virtual register claimant on the system stack.
	//
	// We stop the world to prevent jumbling the stacks.
	stw := stopTheWorld(stwRacelite)

	// We need to move to the system stack to walk
	// the stack of the current goroutine.
	systemstack(func() {
		print(dataRaceHeader)
		print(op1, " at ", hex(addr), " by goroutine ", gp.goid, "\n")
		traceback(^uintptr(0), ^uintptr(0), 0, gp)
		print("\n")
		print("Previous ", op2, " at ", hex(addr), " by goroutine ", gp2.goid, "\n")
		traceback(^uintptr(0), ^uintptr(0), 0, gp2)
		print("\n",
			"Diagnostic information:\n",
			"VR(", r.identifier, "): count=", r.count,
			"\n")
		print(raceliteSeparator)
	})

	// We can restart the world now.
	startTheWorld(stw)
}

// lock acquires the write lock on the virtual register.
func (r *raceliteVirtualRegister) lock() {
	// Atomically acquire the write lock.
	for !r.mu.CompareAndSwap(0, 1) {
		// Allow pre-emption by the other scheduler.
		Gosched()
	}
	// Wait for any read locks to be released before
	// proceeding.
	for r.rmu.Load() != 0 {
		// Allow pre-emption by the other scheduler.
		Gosched()
	}
}

// unlock releases the write lock on the virtual register.
func (r *raceliteVirtualRegister) unlock() {
	// Atomically release the write lock. If the lock is not held,
	// crash and burn. We are dealing with a bug.
	if !r.mu.CompareAndSwap(1, 0) {
		throw("raceliteVirtualRegister: unlock: lock is not held")
	}
}

// rlock increments the read lock on the virtual register.
func (r *raceliteVirtualRegister) rlock() {
	for {
		// Wait for the write lock to be released before
		for r.mu.Load() != 0 {
			Gosched()
		}
		// Atomically increment the read lock.
		r.rmu.Add(1)
		// If the write lock is not held, we can proceed.
		if r.mu.Load() == 0 {
			break
		}
		// Otherwise, we need to release the read lock and try again.
		r.rmu.Add(-1)
	}
}

// runlock decrements the read lock on the virtual register.
func (r *raceliteVirtualRegister) runlock() {
	// Atomically decrement the read lock. If the read lock is not held,
	// crash and burn. We are dealing with a bug.
	if r.rmu.Add(-1) < 0 {
		throw("raceliteVirtualRegister: runlock: rmu is below 0")
	}
}

// demote atommically switches the virtual register from holding
// a write lock to a read lock without interrupting the critical
// section.
//
// We use this to improve performance when write-write and read-write
// races are detected simultaneously.
func (r *raceliteVirtualRegister) demote() {
	// Atomically check if the write lock is held. If not,
	// crash and burn. We are dealing with a bug.
	if r.mu.Load() == 0 {
		throw("raceliteVirtualRegister: demote: lock is not held")
	}
	r.rmu.Add(1)
	r.unlock()
}

// claim allows the writer goroutine as the owner of the virtual register.
func (r *raceliteVirtualRegister) claim(addr uintptr, gp *g) bool {
	r.lock()
	switch r.addr {
	case 0:
		// The virtual register is not occupied.
		// The writer can claim it.
		r.addr, r.owner = addr, gp
		r.count++

		r.unlock()
		return true

	case addr:
		// The virtual register is already occupied by the same address.
		// This only happens if another writer intercepted the write.
		// We are, therefore, dealing with a write-write race.

		// We demote the goroutine to virtual register reader
		// without interrupting the critical section. Other
		// readers can issue reports virtual register.
		r.demote()

		// Report the write-write race,
		r.report(addr, gp, r.owner, "Write", "write")

		startTheWorld(stw)

		r.runlock()
		return false

	default:
		// The virtual register is already occupied by another address.
		r.unlock()
		return false
	}
}

// monitor allows a reader goroutine to check whether the virtual
// register is occupied by a writer with the given address.
func (r *raceliteVirtualRegister) monitor(addr uintptr, gp *g, writefirst bool) bool {
	r.rlock()
	// Check the status of the virtual register.
	if r.addr != addr {
		// The virtual register is occupied by another address.
		r.runlock()
		return false
	}

	// The virtual register is occupied by the same address.
	if writefirst {
		// If the writer stepped in first, we have a write-read race.
		r.report(addr, gp, r.owner, "Read", "write")
	} else {
		// If the reader stepped in first, we have a read-write race.
		// Invert the order of the claimant writer and reader in the report.
		r.report(addr, r.owner, gp, "Write", "read")
	}

	// Reader reporting is done.
	r.runlock()
	return true
}

// release detaches the virtual register from the writer goroutine.
func (r *raceliteVirtualRegister) release() {
	r.lock()
	r.addr, r.owner = 0, nil
	r.unlock()
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
//	else:
//		V.claim(A)
//		micropause()
//		V.release()
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
//	else:
//		micropause()
//		if V.claimed(A):
//			Report read-write race
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
