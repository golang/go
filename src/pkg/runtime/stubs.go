// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Declarations for runtime services implemented in C or assembly.
// C implementations of these functions are in stubs.goc.
// Assembly implementations are in various files, see comments with
// each function.

const ptrSize = 4 << (^uintptr(0) >> 63) // unsafe.Sizeof(uintptr(0)) but an ideal const

//go:noescape
func racereadpc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racewritepc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racereadrangepc(addr unsafe.Pointer, len int, callpc, pc uintptr)

//go:noescape
func racewriterangepc(addr unsafe.Pointer, len int, callpc, pc uintptr)

//go:noescape
func raceacquire(addr unsafe.Pointer)

//go:noescape
func racerelease(addr unsafe.Pointer)

//go:noescape
func raceacquireg(gp *g, addr unsafe.Pointer)

//go:noescape
func racereleaseg(gp *g, addr unsafe.Pointer)

func racefingo()

// Should be a built-in for unsafe.Pointer?
func add(p unsafe.Pointer, x uintptr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + x)
}

// n must be a power of 2
func roundup(p unsafe.Pointer, n uintptr) unsafe.Pointer {
	delta := -uintptr(p) & (n - 1)
	return unsafe.Pointer(uintptr(p) + delta)
}

// in stubs.goc
func getg() *g
func acquirem() *m
func releasem(mp *m)
func gomcache() *mcache

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
	gc_m,
	scavenge_m,
	setFinalizer_m,
	removeFinalizer_m,
	markallocated_m,
	unrollgcprog_m,
	unrollgcproginplace_m,
	gosched_m,
	setgcpercent_m,
	setmaxthreads_m,
	ready_m,
	park_m mFunction
)

// memclr clears n bytes starting at ptr.
// in memclr_*.s
//go:noescape
func memclr(ptr unsafe.Pointer, n uintptr)

func racemalloc(p unsafe.Pointer, size uintptr)

// memmove copies n bytes from "from" to "to".
// in memmove_*.s
//go:noescape
func memmove(to unsafe.Pointer, from unsafe.Pointer, n uintptr)

// in asm_*.s
func fastrand2() uint32

const (
	concurrentSweep = true
)

func gosched()
func starttheworld()
func stoptheworld()

// exported value for testing
var hashLoad = loadFactor

// in asm_*.s
//go:noescape
func memeq(a, b unsafe.Pointer, size uintptr) bool

// Code pointers for the nohash/noequal algorithms. Used for producing better error messages.
var nohashcode uintptr
var noequalcode uintptr

// Go version of runtime.throw.
// in panic.c
func gothrow(s string)

// noescape hides a pointer from escape analysis.  noescape is
// the identity function but escape analysis doesn't think the
// output depends on the input.  noescape is inlined and currently
// compiles down to a single xor instruction.
// USE CAREFULLY!
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

func entersyscall()
func entersyscallblock()
func exitsyscall()

func goroutineheader(gp *g)
func traceback(pc, sp, lr uintptr, gp *g)
func tracebackothers(gp *g)

func cgocallback(fn, frame unsafe.Pointer, framesize uintptr)
func gogo(buf *gobuf)
func gosave(buf *gobuf)
func read(fd int32, p unsafe.Pointer, n int32) int32
func close(fd int32) int32
func mincore(addr unsafe.Pointer, n uintptr, dst *byte) int32
func jmpdefer(fv *funcval, argp unsafe.Pointer)
func exit1(code int32)
func asminit()
func setg(gg *g)
func exit(code int32)
func breakpoint()
func nanotime() int64
func usleep(usec uint32)
func cputicks() int64
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer
func munmap(addr unsafe.Pointer, n uintptr)
func madvise(addr unsafe.Pointer, n uintptr, flags int32)
func newstackcall(fv *funcval, addr unsafe.Pointer, size uint32)
func reflectcall(fn, arg unsafe.Pointer, n uint32, retoffset uint32)
func procyield(cycles uint32)
func osyield()
func cgocallback_gofunc(fv *funcval, frame unsafe.Pointer, framesize uintptr)
func persistentalloc(size, align uintptr, stat *uint64) unsafe.Pointer
func readgogc() int32
func purgecachedstats(c *mcache)
func gostringnocopy(b *byte) string

//go:noescape
func write(fd uintptr, p unsafe.Pointer, n int32) int32

//go:noescape
func cas(ptr *uint32, old, new uint32) bool

//go:noescape
func cas64(ptr *uint64, old, new uint64) bool

//go:noescape
func casp(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool

//go:noescape
func casuintptr(ptr *uintptr, old, new uintptr) bool

//go:noescape
func xadd(ptr *uint32, delta int32) uint32

//go:noescape
func xadd64(ptr *uint64, delta int64) uint64

//go:noescape
func xchg(ptr *uint32, new uint32) uint32

//go:noescape
func xchg64(ptr *uint64, new uint64) uint64

//go:noescape
func xchgp(ptr unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer

//go:noescape
func atomicstore(ptr *uint32, val uint32)

//go:noescape
func atomicstore64(ptr *uint64, val uint64)

//go:noescape
func atomicstorep(ptr unsafe.Pointer, val unsafe.Pointer)

//go:noescape
func atomicstoreuintptr(ptr *uintptr, new uintptr)

//go:noescape
func atomicload(ptr *uint32) uint32

//go:noescape
func atomicload64(ptr *uint64) uint64

//go:noescape
func atomicloadp(ptr unsafe.Pointer) unsafe.Pointer

//go:noescape
func atomicloaduintptr(ptr *uintptr) uintptr

//go:noescape
func atomicloaduint(ptr *uint) uint

//go:noescape
func atomicor8(ptr *uint8, val uint8)

//go:noescape
func setcallerpc(argp unsafe.Pointer, pc uintptr)

//go:noescape
func getcallerpc(argp unsafe.Pointer) uintptr

//go:noescape
func getcallersp(argp unsafe.Pointer) uintptr

//go:noescape
func asmcgocall(fn, arg unsafe.Pointer)

//go:noescape
func open(name *byte, mode, perm int32) int32
