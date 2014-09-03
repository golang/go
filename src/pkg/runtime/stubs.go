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
const regSize = 4 << (^uintreg(0) >> 63) // unsafe.Sizeof(uintreg(0)) but an ideal const

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
//go:nosplit
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

// in asm_*.s
func mcall(func(*g))
func onM(fn func())

// C functions that run on the M stack.
// Call using mcall.
// These functions need to be written to arrange explicitly
// for the goroutine to continue execution.
func gosched_m(*g)
func park_m(*g)

// More C functions that run on the M stack.
// Call using onM.
// Arguments should be passed in m->scalararg[x] and m->ptrarg[x].
// Return values can be passed in those same slots.
// These functions return to the goroutine when they return.
func mcacheRefill_m()
func largeAlloc_m()
func gc_m()
func scavenge_m()
func setFinalizer_m()
func removeFinalizer_m()
func markallocated_m()
func unrollgcprog_m()
func unrollgcproginplace_m()
func setgcpercent_m()
func setmaxthreads_m()
func ready_m()
func deferproc_m()
func goexit_m()

// memclr clears n bytes starting at ptr.
// in memclr_*.s
//go:noescape
func memclr(ptr unsafe.Pointer, n uintptr)

func racemalloc(p unsafe.Pointer, size uintptr)

// memmove copies n bytes from "from" to "to".
// in memmove_*.s
//go:noescape
func memmove(to unsafe.Pointer, from unsafe.Pointer, n uintptr)

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
//go:nosplit
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

func entersyscall()
func entersyscallblock()
func exitsyscall()

func cgocallback(fn, frame unsafe.Pointer, framesize uintptr)
func gogo(buf *gobuf)
func gosave(buf *gobuf)
func read(fd int32, p unsafe.Pointer, n int32) int32
func close(fd int32) int32
func mincore(addr unsafe.Pointer, n uintptr, dst *byte) int32

//go:noescape
func jmpdefer(fv *funcval, argp uintptr)
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
func goexit()

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

//go:noescape
func gotraceback(*bool) int32

func funcname(*_func) *byte

func gofuncname(f *_func) string {
	return gostringnocopy(funcname(f))
}

const _NoArgs = ^uintptr(0)

var newproc, lessstack struct{} // C/assembly functions

func funcspdelta(*_func, uintptr) int32 // symtab.c
func funcarglen(*_func, uintptr) int32  // symtab.c
const _ArgsSizeUnknown = -0x80000000    // funcdata.h

// return0 is a stub used to return 0 from deferproc.
// It is called at the very end of deferproc to signal
// the calling Go function that it should not jump
// to deferreturn.
// in asm_*.s
func return0()
