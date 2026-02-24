// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: //go:build racelite

package runtime

import (
	"internal/goarch"
	"internal/runtime/sys"
)

const (
	// raceliteVRShift is log2(raceliteVRNum).
	raceliteVRShift uint8 = 4
	// raceliteVRNum is the number of virtual registers we have.
	// Keep as power of 2 for efficient modulo operation.
	raceliteVRNum = 1 << raceliteVRShift

	// raceliteRecordShift is log2(raceliteRecordNum).
	raceliteRecordShift uint8 = 8
	// raceliteRecordNum is the number of data race records we have.
	raceliteRecordNum = 1 << raceliteRecordShift

	// racelitePCDepth is the number of program counters to store
	// in data race record stacks.
	racelitePCDepth = 16

	// These values denote the type of the accesses involved
	// in a race as follows (assume BE):
	//
	//	 .----> Second operation (latest)
	//	 | .--> First operation (previous)
	//	.-+-.
	//	|0|0: write-write (default)
	//	|0|1: read-write
	//	|1|0: write-read
	//	|1|1: read-read (impossible case)
	//
	// They are used to determine the access type pair.
	raceliteOp1Read, raceliteOp2Read uint8 = 0b01, 0b10

	// raceliteDisarmedPCsShift is log2(raceliteDisarmedPCsSize).
	raceliteDisarmedPCsShift uint8 = 12
	// raceliteDisarmedPCsSize is the size of the array that
	// monitors disarmed PCs.
	raceliteDisarmedPCsSize = 1 << raceliteDisarmedPCsShift
)

var (
	// raceliteSamplingRand allows us to randomize which addresses we check for racelite,
	// but to do so in a way we get a consistent answer per address across concurrent readers
	// and writers.
	//
	// TODO(thepudds): for now, for convenience we only check this in runtime, but we
	// could have the compiler emit the check -- probably important to do or at least
	// try if were were to pursue this general approach.
	raceliteSamplingRand uint32 = 0

	// raceliteReg is the globalarray of virtual registers.
	raceliteReg *[raceliteVRNum]raceliteVirtualRegister

	// raceliteDisarmedPCs is the array that monitors disarmed PCs.
	// Each PC is represented by a bit in the array.
	// We use Fibonacci hashing to map the PC unto the array index.
	raceliteDisarmedPCs *[raceliteDisarmedPCsSize]bool

	// raceliteRecords is the array that records data races.
	// This prevents duplicate data races from being reported.
	raceliteRecords *[raceliteRecordNum]bool

	// racliteSamplingShift is log2(raceliteSamplingMask+1)
	//
	// Initialized in raceliteinit based on the value of debug.racelite.
	raceliteSamplingShift uint8
	// raceliteSamplingMask is used when determining whether to sample an address.
	//
	// Initialized in raceliteinit based on the value of debug.racelite.
	raceliteSamplingMask uint32
)

// diag reports true is we should print extra info.
func diag() bool {
	return debug.racelite >= 2
}

// Initialize Racelite tooling
func raceliteinit() {
	if debug.racelite <= 0 {
		// No-op if racelite is not enabled.
		return
	}

	// Initialize the random address sampler.
	raceliteSamplingRand = cheaprand()

	raceliteReg = new([raceliteVRNum]raceliteVirtualRegister)
	// Initialize all virtual registers.
	for i := 0; i < raceliteVRNum; i++ {
		// Assign an identifier to the virtual register.
		// This is used for debugging and experimentation.
		raceliteReg[i].identifier = uint32(i)
	}

	raceliteDisarmedPCs = new([raceliteDisarmedPCsSize]bool)
	raceliteRecords = new([raceliteRecordNum]bool)

	switch debug.racelite {
	case 1:
		// Standard Racelite
		// Sample one address (8-byte aligned) in every 4096.
		raceliteSamplingShift = 11
	case 2:
		// Debug Racelite
		// Sample one address (8-byte aligned) in every 2.
		raceliteSamplingShift = 1
	}

	raceliteSamplingMask = (1 << raceliteSamplingShift) - 1
}

// raceliteget returns the virtual register for the given address.
func raceliteget(addr uintptr) *raceliteVirtualRegister {
	// Compile-time optimized to bitwise AND.
	return &raceliteReg[(addr>>3)%raceliteVRNum]
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
}

// raceliteHash computes the Fibonacci hash of the given value,
// then compressing it into a range of [0, 2^shift).
func raceliteHash(v uintptr, shift uint8) uintptr {
	const k = 0x9e3779b97f4a7c15 // golden ratio
	return (v * k) >> (goarch.PtrSize*8 - shift)
}

// raceliteDisarm marks the given PC as disarmed.
// Is racy, but that is ok.
func raceliteDisarm(v uintptr) {
	raceliteDisarmedPCs[raceliteHash(v, raceliteDisarmedPCsShift)] = true
}

// raceliteDisarmed checks if the given PC is disarmed.
// Is racy, but that is ok.
func raceliteDisarmed(v uintptr) bool {
	return raceliteDisarmedPCs[raceliteHash(v, raceliteDisarmedPCsShift)]
}

// raceliteSampling checks if we should sample the given address for data race detection.
func raceliteSampling(addr uintptr) bool {
	// Check that we are sampling this address, stripped of
	// a (3+raceliteVRShift)-bit suffix for 8-byte alignment
	// and because we have enough virutal registers.
	if (uint32(addr>>(3+raceliteVRShift))^raceliteSamplingRand)&raceliteSamplingMask != 0 {
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

// raceliteVirtualRegister is a virtual register that can be used to
// dynamically monitor an address for data races.
type raceliteVirtualRegister struct {
	// Mutual exclusion ensures that the virtual register maintains a consistent state.
	rwmutex

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

// raceliteRec is a preserved data race record. Aggregate
// data races matching the same signature into a single report.
type raceliteRec struct {
	// The address with racy accesses. This is just representative,
	// as de-duplication is relative to access operation source locations.
	addr uintptr

	// Program counters of racy stacks.
	//
	// pcs1 denotes the previous operation
	// pcs2 denotes the latest operation
	pcs1, pcs2 [racelitePCDepth]uintptr
	// Stack depth of racy goroutines (cannot exceed racelitePCDepth).
	//
	// n1 denotes the stack depth of the previous operation
	// n2 denotes the stack depth of the latest operation
	n1, n2 int
	// The racy goroutine IDs. goid1 denotes the goroutine ID of the previous operation
	// goid2 denotes the goroutine ID of the latest operation
	goid1, goid2 uint64

	// The operation types that caused the data race.
	// For some prefix p, op carries the following meanings:
	//
	//	01: write-read
	//	10: read-write
	//	00: write-write (default)
	//	11: read-read (impossible)
	ops uint8
}

// reportPC prints a stack trace to the console.
func (rec raceliteRec) reportPC(pcs [racelitePCDepth]uintptr, n int) {
	for _, pc := range pcs[:n] {
		if f := findfunc(pc); f.valid() {
			if pc > f.entry() {
				pc--
			}
			printAncestorTracebackFuncInfo(f, pc)
		}
	}
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
//	Race discovered 3 times.
//	==================
func (rec raceliteRec) report() {
	if rec.addr == 0 {
		return
	}

	slot := raceliteHash(((rec.pcs1[0] + rec.pcs2[0]) ^ (rec.pcs1[0] * rec.pcs2[0])), raceliteRecordShift)
	if raceliteRecords[slot] {
		// We already recorded this data race.
		return
	}

	raceliteRecords[slot] = true

	var op1, op2 string
	// Get the type of the previous operation.
	if rec.ops&raceliteOp1Read != 0 {
		op1 = "read"
	} else {
		op1 = "write"
	}
	// Get the type of the latest operation.
	if rec.ops&raceliteOp2Read != 0 {
		op2 = "Read"
	} else {
		op2 = "Write"
	}

	stw := stopTheWorld(stwRacelite)
	// Print the latest operation first (like TSan).
	print("==================\n",
		"WARNING: DATA RACE\n",
		op2, " at ", hex(rec.addr), " by goroutine ", rec.goid2, "\n")
	rec.reportPC(rec.pcs2, rec.n2)
	// Print the previous operation second.
	print("\n",
		"Previous ", op1, " at ", hex(rec.addr), " by goroutine ", rec.goid1, "\n")
	rec.reportPC(rec.pcs1, rec.n1)
	print("==================\n")
	startTheWorld(stw)
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
			n1:    r.n,    // Get stack depth of writer goroutine.
			goid1: r.goid, // Get goroutine ID of writer goroutine.
		}
		// Copy its PC stack.
		copy(rec.pcs1[:], r.pcs[:])

		// We obtained all the information we need about the claiming writer
		// goroutine and can now release the lock on the virtual register.
		r.unlock()

		// Record the remaining information
		rec.addr = addr
		// Get the PCs and stack depth of the current goroutine.
		rec.n2 = callers(2, rec.pcs2[:racelitePCDepth])

		// Disarm the instrumentation for this writer.
		raceliteDisarm(rec.pcs2[0])

		rec.goid2 = getg().goid
		rec.report()
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

	switch {
	case op&raceliteOp1Read != 0:
		// The writer occurred second, so we have a read-write race
		// Place the writer as the second goroutine.
		rec = raceliteRec{
			n2:    r.n,    // Get stack depth of writer goroutine.
			goid2: r.goid, // Get goroutine ID of writer goroutine.
		}
		copy(rec.pcs2[:], r.pcs[:r.n]) // Copy the stack.
	case op^raceliteOp2Read != 0:
		// The writer occurred first, so we have a write-read race
		// Place the writer as the first goroutine.
		rec = raceliteRec{
			n1:    r.n,    // Get stack depth of writer goroutine.
			goid1: r.goid, // Get goroutine ID of writer goroutine.
		}
		copy(rec.pcs1[:], r.pcs[:r.n]) // Copy the stack.
	default:
		// Something strange happened. Tolerate, but drop the report.
		r.runlock()
		return true
	}

	// We obtained all the information we need about the writer goroutine,
	// and can now release the lock on the virtual register.
	r.runlock()

	switch {
	case op&raceliteOp1Read != 0:
		// The writer occurred second, so we have a read-write race.
		// Place the reader as the first goroutine.
		rec.n1 = callers(2, rec.pcs1[:])
		rec.goid1 = getg().goid // Get goroutine ID of reader goroutine.

		// Disarm reader instrumentation
		raceliteDisarm(rec.pcs1[0])
	case op&raceliteOp2Read != 0:
		// The writer occurred first, so we have a write-read race.
		// Place the reader as the second goroutine.
		rec.n2 = callers(2, rec.pcs2[:])
		rec.goid2 = getg().goid // Get goroutine ID of writer goroutine.

		// Disarm reader instrumentation
		raceliteDisarm(rec.pcs2[0])
	default:
		// Something strange happened. Tolerate, but drop the report.
		return true
	}

	// Record the remaining information
	rec.addr = addr
	rec.ops = op // write-read or read-write race

	rec.report()

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
	if !raceliteSampling(addr) {
		// We are not sampling this address.
		return
	}

	if raceliteDisarmed(sys.GetCallerPC()) {
		// We already detected a data race at this PC.
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
	if !raceliteSampling(addr) {
		// We are not sampling this address.
		return
	}

	if raceliteDisarmed(sys.GetCallerPC()) {
		// We already detected a data race at this PC.
		return
	}

	r := raceliteget(addr)
	// Check for a write-read race.
	if r.monitor(addr, raceliteOp2Read) {
		return
	}

	micropause()

	// Check for a read-write race.
	r.monitor(addr, raceliteOp1Read)
}
