// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Declarations for runtime services implemented in C or assembly.
// C implementations of these functions are in stubs.goc.
// Assembly implementations are in various files, see comments with
// each function.

const (
	ptrSize = unsafe.Sizeof((*byte)(nil))
)

//go:noescape
func gogetcallerpc(p unsafe.Pointer) uintptr

//go:noescape
func racereadpc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racewritepc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racereadrangepc(addr unsafe.Pointer, len int, callpc, pc uintptr)

// Should be a built-in for unsafe.Pointer?
func add(p unsafe.Pointer, x uintptr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + x)
}

// n must be a power of 2
func roundup(p unsafe.Pointer, n uintptr) unsafe.Pointer {
	return unsafe.Pointer((uintptr(p) + n - 1) &^ (n - 1))
}

// in stubs.goc
func acquirem() *m
func releasem(mp *m)

// in asm_*.s
func mcall(fn *byte)
func onM(fn *byte)

// C routines that run on the M stack.  Call these like
//   mcall(&mcacheRefill)
// Arguments should be passed in m->scalararg[x] and
// m->ptrarg[x].  Return values can be passed in those
// same slots.
var mcacheRefill byte
var largeAlloc byte
var mprofMalloc byte
var mgc2 byte
var setFinalizer byte
var markallocated_m byte

// memclr clears n bytes starting at ptr.
// in memclr_*.s
func memclr(ptr unsafe.Pointer, n uintptr)

func racemalloc(p unsafe.Pointer, size uintptr)
func tracealloc(p unsafe.Pointer, size uintptr, typ *_type)

// memmove copies n bytes from "from" to "to".
// in memmove_*.s
func memmove(to unsafe.Pointer, from unsafe.Pointer, n uintptr)

// in asm_*.s
func fastrand2() uint32

const (
	gcpercentUnknown = -2
	concurrentSweep  = true
)

// in asm_*.s
// if *p == x { *p = y; return true } else { return false }, atomically
//go:noescape
func gocas(p *uint32, x uint32, y uint32) bool

//go:noescape
func gocasx(p *uintptr, x uintptr, y uintptr) bool

func goreadgogc() int32
func gonanotime() int64
func gosched()
func starttheworld()
func stoptheworld()
func clearpools()

// in asm_*.s
//go:noescape
func gohash(a *alg, p unsafe.Pointer, size uintptr, seed uintptr) uintptr

// in asm_*.s
//go:noescape
func goeq(alg *alg, p, q unsafe.Pointer, size uintptr) bool

// exported value for testing
var hashLoad = loadFactor

// in asm_*.s
//go:noescape
func gomemeq(a, b unsafe.Pointer, size uintptr) bool

// Code pointer for the nohash algorithm. Used for producing better error messages.
var nohashcode uintptr

// Go version of runtime.throw.
// in panic.c
func gothrow(s string)

func golock(x *lock)
func gounlock(x *lock)
func semacquire(*uint32, bool)
func semrelease(*uint32)
