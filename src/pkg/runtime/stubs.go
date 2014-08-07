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

//go:noescape
func racewriterangepc(addr unsafe.Pointer, len int, callpc, pc uintptr)

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

// An mFunction represents a C function that runs on the M stack.  It
// can be called from Go using mcall or onM.  Through the magic of
// linking, an mFunction variable and the corresponding C code entry
// point live at the same address.
type mFunction byte

// in asm_*.s
func mcall(fn *mFunction)
func onM(fn *mFunction)

// C functions that run on the M stack.  Call these like
//   mcall(&mcacheRefill_m)
// Arguments should be passed in m->scalararg[x] and
// m->ptrarg[x].  Return values can be passed in those
// same slots.
var (
	mcacheRefill_m,
	largeAlloc_m,
	mprofMalloc_m,
	gc_m,
	setFinalizer_m,
	markallocated_m,
	unrollgcprog_m,
	unrollgcproginplace_m mFunction
)

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

// Atomic operations to read/write a pointer.
// in stubs.goc
func goatomicloadp(p unsafe.Pointer) unsafe.Pointer     // return *p
func goatomicstorep(p unsafe.Pointer, v unsafe.Pointer) // *p = v

// in stubs.goc
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

// Return the Go equivalent of the C Alg structure.
// TODO: at some point Go will hold the truth for the layout
// of runtime structures and C will be derived from it (if
// needed at all).  At that point this function can go away.
type goalgtype struct {
	// function for hashing objects of this type
	// (ptr to object, size, seed) -> hash
	hash func(unsafe.Pointer, uintptr, uintptr) uintptr
}

func goalg(a *alg) *goalgtype {
	return (*goalgtype)(unsafe.Pointer(a))
}

// noescape hides a pointer from escape analysis.  noescape is
// the identity function but escape analysis doesn't think the
// output depends on the input.  noescape is inlined and currently
// compiles down to a single xor instruction.
// USE CAREFULLY!
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}
