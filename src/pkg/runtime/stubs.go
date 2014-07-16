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

// rawstring allocates storage for a new string. The returned
// string and byte slice both refer to the same storage.
// The storage is not zeroed. Callers should use
// b to set the string contents and then drop b.
func rawstring(size int) (string, []byte)

// rawbyteslice allocates a new byte slice. The byte slice is not zeroed.
func rawbyteslice(size int) []byte

// rawruneslice allocates a new rune slice. The rune slice is not zeroed.
func rawruneslice(size int) []rune

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

// Make a new object of the given type
// in stubs.goc
func unsafe_New(t *_type) unsafe.Pointer
func unsafe_NewArray(t *_type, n uintptr) unsafe.Pointer

// memclr clears n bytes starting at ptr.
// in memclr_*.s
func memclr(ptr unsafe.Pointer, n uintptr)

// memmove copies n bytes from "from" to "to".
// in memmove_*.s
func memmove(to unsafe.Pointer, from unsafe.Pointer, n uintptr)

// in asm_*.s
func fastrand2() uint32

// in asm_*.s
// if *p == x { *p = y; return true } else { return false }, atomically
//go:noescape
func gocas(p *uint32, x uint32, y uint32) bool

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
