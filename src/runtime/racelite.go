// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: //go:build racelite

package runtime

import (
	"internal/runtime/atomic"
)

const (
	// raceliteShift is the number of bits to shift the
	// address. We use raceliteShift to compute the number of
	// virtual registers we have.
	raceliteShift uint8 = 4

	// raceliteRecordNum is the number of data race records we have
	// available for reporting.
	raceliteRecordNum uint32 = 256

	// racelitePCDepth is the number of program counters to store
	// in data race record stacks.
	racelitePCDepth = 16

	// raceliteRegNum is the number of virtual registers we have.
	// Keep as power of 2 for efficient modulo operation.
	raceliteRegNum = 1 << raceliteShift

	// raceliteCheckAddrMask selects an address suffix which
	// can be monitored for racelite.
	raceliteCheckAddrMask = (1 << (4 * raceliteShift)) - 1

	// Operation masks are used to determine the type of the operation
	// performed at the data race

	// writeOp1Mask is used to determine whether
	// the first operation was a write.
	writeOp1Mask uint8 = 0x3
	// writeOp2Mask is used to determine whether
	// the second operation was a write.
	writeOp2Mask uint8 = 0xc

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

// diag reports true is we should print extra info.
func diag() bool {
	return debug.racelite >= 2
}

// raceliteReg is the globalarray of virtual registers.
var raceliteReg *[raceliteRegNum]raceliteVirtualRegister

// Initialize Racelite tooling
func raceliteinit() {
	if debug.racelite <= 0 {
		// No-op if racelite is not enabled.
		return
	}

	// Initialize the random address sampler.
	raceliteCheckAddrRand = cheaprand()

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

// inStack checks if addr is in the current goroutine's stack.
func inStack(addr uintptr) bool {
	gp := getg()
	return gp.stack.lo <= addr && addr < gp.stack.hi
}

// micropause injects a small delay to a load
// or store operation, allowing Racelite to see
// whether another thread accessed the same address.
func micropause() {
	if cheaprandn(100_000) == 0 {
		// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
		// let's us see it work for the first time. ;)
		usleep(1)
	}
	// NOTE(vsaioc): We may want to experiment here.
	// Gosched()
}

// raceliteCheckAddr checks if we should sample the given address for data race detection.
func raceliteCheckAddr(addr uintptr) bool {
	// Check that we are sampling this address.
	//
	// FIXME(vsaioc): At the moment, we always sample when debugging.
	if !diag() && uint32(addr>>(3+raceliteShift))&raceliteCheckAddrMask != raceliteCheckAddrRand {
		return false
	}

	// Check that this address is not on our stack.
	if inStack(addr) {
		// Ignore stack addresses
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

// raceliteLock is a custom read-write spin-lock that is only
// used for locking virtual registers and data race records.
//
// TOOD(vsaioc): better integrate this with the rest of the runtime.
type raceliteLock struct {
	mu, rmu atomic.Int32
}

// raceliteVirtualRegister is a virtual register that can be used to
// dynamically monitor an address for data races.
type raceliteVirtualRegister struct {
	// Mutual exclusion ensures that the virtual register maintains a consistent state.
	// We do not use a runtime mutex here because we want to stop the world.
	// mu and rmu are used for a custom read-write spin-lock.
	raceliteLock

	// addr is the address currently being monitored in the virtual register
	addr uintptr
	// pcs contains the program counters of the writer goroutine
	// that claimed the virtual register.
	pcs [racelitePCDepth]uintptr
	// n is the stack depth of the writer goroutine
	// that claimed the virtual register.
	n int
	// goid is the goroutine ID of the writer goroutine
	// that claimed the virtual register.
	goid uint64

	// Diagnostic information.
	//
	// TODO(vsaioc): Add more for experimentation purposes and remove
	// for the polished release version.
	count      uint64 // the number of times the virtual register has been claimed
	identifier uint32 // the identifier of the virtual register
}

// raceliteRec is a preserved data race record. We use this
// to avoid flooding I/O with data race reports every time a data race occurs.
type raceliteRec struct {
	// The address of the data race. This is just representative,
	// as de-duplication is handled at the source location of the operations.
	addr uintptr

	// The program counters where the data race occurred.
	pcs1, pcs2 [racelitePCDepth]uintptr
	// The stack depth for each racy goroutine.
	n1, n2 int
	// The goroutine IDs of the two goroutines involved in the data race.
	goid1, goid2 uint64

	// The operation types that caused the data race.
	// The last 4 LSB have the following meanings:
	//
	//	- 0000: read-read (impossible)
	//	- 0011: write-read
	//	- 1100: read-write
	//	- 1111: write-write
	ops uint8

	count uint64 // the number of times the data race has been reported
}

var (
	// raceliteReg is the globalarray of virtual registers.
	raceliteReg *[raceliteRegNum]raceliteVirtualRegister

	// raceliteRecs is a global structure that contains the array of data race records
	// and a mutex to protect access to the array.
	raceliteRecs *struct {
		raceliteLock
		recs [raceliteRecordNum]raceliteRec
	}
)

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

	// Initialize the data race record structure.
	raceliteRecs = new(struct {
		raceliteLock
		recs [raceliteRecordNum]raceliteRec
	})
}

// raceliteReport prints a data race report to the console.
// The reports match the format of the data race warnings
// issued by TSan (omitting ancestry information).
//
// Example:
//
//	==================
//	WARNING: DATA RACE
//	Write at 0x1234567890 by goroutine 123
//	[stack trace of the writer]
//
//	Previous write at 0x1234567890 by goroutine 456
//	[stack trace of the previous writer]
//
//	Race discovered 40 times.
//	==================
func (rec raceliteRec) report() {
	if rec.addr == 0 {
		return
	}

	print(dataRaceHeader)
	var op1, op2 string
	if rec.ops&writeOp1Mask != 0 {
		op1 = "Write"
	} else {
		op1 = "Read"
	}
	if rec.ops&writeOp2Mask != 0 {
		op2 = "write"
	} else {
		op2 = "read"
	}
	print(op1, " at ", hex(rec.addr), " by goroutine ", rec.goid1, "\n")
	for _, pc := range rec.pcs1[:rec.n1] {
		if f := findfunc(pc); f.valid() {
			if pc > f.entry() {
				pc--
			}
			printAncestorTracebackFuncInfo(f, pc)
		}
	}
	print("\n")
	print("Previous ", op2, " at ", hex(rec.addr), " by goroutine ", rec.goid2, "\n")
	for _, pc := range rec.pcs2[:rec.n2] {
		if f := findfunc(pc); f.valid() {
			if pc > f.entry() {
				pc--
			}
			printAncestorTracebackFuncInfo(f, pc)
		}
	}
	print("\n", "Race discovered ", rec.count, " times.\n")
	print(raceliteSeparator)
}

func raceliteReportAll() {
	if debug.racelite <= 0 {
		// No-op if racelite is not enabled.
		return
	}

	stw := stopTheWorld(stwRacelite)
	for _, rec := range raceliteRecs.recs {
		rec.report()
	}
	startTheWorld(stw)
}

// raceliteget returns the virtual register for the given address.
func raceliteget(addr uintptr) *raceliteVirtualRegister {
	// Compile-time optimized to bitwise AND.
	return &raceliteReg[(addr>>3)%raceliteRegNum]
}

// record checks whether a data race can be recorded in
// the global data race record data structure.
func record(rec raceliteRec) {
	// Compute fingerprint from PCs
	fp := uint64(0)
	for i := 0; i < rec.n1; i++ {
		fp = fp*31 + uint64(rec.pcs1[i])
	}
	for i := 0; i < rec.n2; i++ {
		fp = fp*31 + uint64(rec.pcs2[i])
	}

	slot := fp % uint64(raceliteRecordNum)

	raceliteRecs.lock()
	switch rec2 := raceliteRecs.recs[slot]; {
	case rec2.addr == 0:
		// We have discovered a new data race. Store it.
		raceliteRecs.recs[slot] = rec
	case rec2.pcs1[0] == rec.pcs1[0] && rec2.pcs2[0] == rec.pcs2[0]:
		// We have discovered a duplicate data race. Increment the count.
		raceliteRecs.recs[slot].count++
	default:
		// We have encountered a hash collision.
		// Report it without overriding the existing record.
		if false && diag() {
			println("Hash collision detected for data race at", rec.addr)
		}
	}
	raceliteRecs.unlock()
}

// lock acquires the write lock on the virtual register.
func (r *raceliteLock) lock() {
	// Atomically acquire the write lock.
	for !r.mu.CompareAndSwap(0, 1) {
		Gosched() // Allow pre-emption
	}

	// Wait for any read locks to be released before
	// proceeding.
	for r.rmu.Load() != 0 {
		Gosched() // Allow pre-emption
	}
}

// unlock releases the write lock on the virtual register.
func (r *raceliteLock) unlock() {
	// Atomically release the write lock. If the lock is not held,
	// crash and burn. We are dealing with a bug.
	if !r.mu.CompareAndSwap(1, 0) {
		throw("raceliteVirtualRegister: unlock: lock is not held")
	}
}

// rlock increments the read lock on the virtual register.
func (r *raceliteLock) rlock() {
	for {
		// Wait for the write lock to be released before
		for r.mu.Load() != 0 {
			Gosched() // Allow pre-emption
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
func (r *raceliteLock) runlock() {
	// Atomically decrement the read lock. If the read lock is not held,
	// crash and burn. We are dealing with a bug.
	if r.rmu.Add(-1) < 0 {
		throw("raceliteVirtualRegister: runlock: rmu is below 0")
	}
}

// claim allows the writer goroutine as the owner of the virtual register.
func (r *raceliteVirtualRegister) claim(addr uintptr) bool {
	r.lock()
	switch r.addr {
	case 0:
		// The virtual register is not occupied.
		// The writer can claim it.
		r.addr = addr

		// Record the current goroutine as the owner.
		r.goid = getg().goid       // Get its goroutine ID.
		r.n = callers(2, r.pcs[:]) // Copy its PC stack

		if false && diag() {
			r.count++ // Increment the claim count.
			print("Diagnostic information:\n",
				"VR(", r.identifier, "): claimed=", r.count, "\n")
		}

		r.unlock()
		return true

	case addr:
		// The virtual register is already occupied by the same address.
		// This only happens if another writer intercepted the write.
		// We are, therefore, dealing with a write-write race.

		// Construct a data race record.
		// Fill it with all information about the writer
		// goroutine before releasing the lock on the
		// virtual registers.
		rec := raceliteRec{
			n2:    r.n,    // Get stack depth of writer goroutine.
			goid2: r.goid, // Get goroutine ID of writer goroutine.
		}
		// Copy its PC stack.
		copy(rec.pcs2[:], r.pcs[:])

		// We can now release the lock on the virtual register.
		r.unlock()

		// Record the remaining information
		rec.addr = addr
		// Get the PCs and stack depth of the current goroutine.
		rec.n1 = callers(2, rec.pcs1[:racelitePCDepth])

		rec.goid1 = getg().goid

		rec.ops = writeOp1Mask | writeOp2Mask // write-write race
		rec.count = 1

		record(rec)
		return false

	default:
		// The virtual register is already occupied by another address.
		r.unlock()
		return false
	}
}

// monitor allows a reader goroutine to check whether the virtual
// register is occupied by a writer with the given address.
func (r *raceliteVirtualRegister) monitor(addr uintptr, op uint8) bool {
	r.rlock()
	// Check the status of the virtual register.
	if r.addr != addr {
		// The virtual register is occupied by another address.
		// This is not a data race.
		r.runlock()
		return false
	}

	// The register is oocupied by the same address.
	// Construct a data race record with information
	// about the writer goroutine.
	var rec raceliteRec

	if op&writeOp1Mask != 0 {
		// The writer occurred second, so we have a read-write race
		// Place the writer as the first goroutine.
		rec = raceliteRec{
			n1:    r.n,    // Get stack depth of writer goroutine.
			goid1: r.goid, // Get goroutine ID of writer goroutine.
		}
		// Copy its PC stack.
		copy(rec.pcs1[:], r.pcs[:])
	} else if op&writeOp2Mask != 0 {
		// The writer occurred first, so we have a write-read race
		// Place the writer as the second goroutine.
		rec = raceliteRec{
			n2:    r.n,    // Get stack depth of writer goroutine.
			goid2: r.goid, // Get goroutine ID of writer goroutine.
		}
		// Copy its PC stack.
		copy(rec.pcs2[:], r.pcs[:])
	} else {
		// Something strange happened. Tolerate the error,
		// but drop the data race record.
		r.runlock()
		return true
	}

	// We can now release the lock on the virtual register.
	r.runlock()

	if op&writeOp1Mask != 0 {
		// The writer occurred second, so we have a read-write race.
		// Place the reader as the second goroutine.
		rec.n2 = callers(2, rec.pcs2[:])
		rec.goid2 = getg().goid // Get goroutine ID of reader goroutine.
	} else if op&writeOp2Mask != 0 {
		// The writer occurred first, so we have a write-read race.
		// Place the reader as the first goroutine.
		rec.n1 = callers(2, rec.pcs1[:])
		rec.goid1 = getg().goid // Get goroutine ID of writer goroutine.
	} else {
		// Something strange happened. Tolerate the error,
		// but drop the data race record.
		return true
	}

	// Record the remaining information
	rec.addr = addr

	rec.ops = op // write-read or read-write race
	rec.count = 1

	record(rec)

	return true
}

// release detaches the virtual register from the writer goroutine.
func (r *raceliteVirtualRegister) release() {
	r.lock()
	// We only need to clear the address.
	// All other properties are updated on the next claim.
	r.addr = 0
	r.unlock()
}

// racelitewrite instruments a store operation with data race detection,
// according to the following logic:
//
//	Let A be the stored address and V the virtual register
//
//	if V is claimed for A:
//		Report write-write race and exit
//	claim V for A
//	pause
//	release V
func racelitewrite(addr uintptr) {
	if !raceliteCheckAddr(addr) {
		// We are not sampling this address.
		return
	}
	r := raceliteget(addr)

	// Check the status of the virtual register.
	if !r.claim(addr) {
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
//	Let A be the loaded address and V the virtual register
//
//	if V is claimed for A:
//		Report write-read race and exit
//	pause
//	if V is claimed for A:
//		Report read-write race and exit
func raceliteread(addr uintptr) {
	if !raceliteCheckAddr(addr) {
		// We are not sampling this address.
		return
	}
	r := raceliteget(addr)

	// Check for a write-read race.
	if r.monitor(addr, writeOp2Mask) {
		return
	}

	micropause()

	// Check for a read-write race.
	r.monitor(addr, writeOp1Mask)
}
