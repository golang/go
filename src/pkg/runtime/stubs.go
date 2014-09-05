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

// mcall switches from the g to the g0 stack and invokes fn(g),
// where g is the goroutine that made the call.
// mcall saves g's current PC/SP in g->sched so that it can be restored later.
// It is up to fn to arrange for that later execution, typically by recording
// g in a data structure, causing something to call ready(g) later.
// mcall returns to the original goroutine g later, when g has been rescheduled.
// fn must not return at all; typically it ends by calling schedule, to let the m
// run other goroutines.
//
// mcall can only be called from g stacks (not g0, not gsignal).
//go:noescape
func mcall(fn func(*g))

// onM switches from the g to the g0 stack and invokes fn().
// When fn returns, onM switches back to the g and returns,
// continuing execution on the g stack.
// If arguments must be passed to fn, they can be written to
// g->m->ptrarg (pointers) and g->m->scalararg (non-pointers)
// before the call and then consulted during fn.
// Similarly, fn can pass return values back in those locations.
// If fn is written in Go, it can be a closure, which avoids the need for
// ptrarg and scalararg entirely.
// After reading values out of ptrarg and scalararg it is conventional
// to zero them to avoid (memory or information) leaks.
//
// If onM is called from a g0 stack, it invokes fn and returns,
// without any stack switches.
//
// If onM is called from a gsignal stack, it crashes the program.
// The implication is that functions used in signal handlers must
// not use onM.
//
// NOTE(rsc): We could introduce a separate onMsignal that is
// like onM but if called from a gsignal stack would just run fn on
// that stack. The caller of onMsignal would be required to save the
// old values of ptrarg/scalararg and restore them when the call
// was finished, in case the signal interrupted an onM sequence
// in progress on the g or g0 stacks. Until there is a clear need for this,
// we just reject onM in signal handling contexts entirely.
//
//go:noescape
func onM(fn func())

func badonm() {
	gothrow("onM called from signal goroutine")
}

// C functions that run on the M stack.
// Call using mcall.
func gosched_m(*g)
func park_m(*g)

// More C functions that run on the M stack.
// Call using onM.
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
func newextram()
func lockOSThread()
func unlockOSThread()

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
func osyield()
func procyield(cycles uint32)
func cgocallback_gofunc(fv *funcval, frame unsafe.Pointer, framesize uintptr)
func readgogc() int32
func purgecachedstats(c *mcache)
func gostringnocopy(b *byte) string
func goexit()

//go:noescape
func write(fd uintptr, p unsafe.Pointer, n int32) int32

//go:noescape
func cas(ptr *uint32, old, new uint32) bool

//go:noescape
func casp(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool

//go:noescape
func casuintptr(ptr *uintptr, old, new uintptr) bool

//go:noescape
func atomicstoreuintptr(ptr *uintptr, new uintptr)

//go:noescape
func atomicloaduintptr(ptr *uintptr) uintptr

//go:noescape
func atomicloaduint(ptr *uint) uint

//go:noescape
func setcallerpc(argp unsafe.Pointer, pc uintptr)

//go:noescape
func getcallerpc(argp unsafe.Pointer) uintptr

//go:noescape
func getcallersp(argp unsafe.Pointer) uintptr

//go:noescape
func asmcgocall(fn, arg unsafe.Pointer)

//go:noescape
func asmcgocall_errno(fn, arg unsafe.Pointer) int32

//go:noescape
func open(name *byte, mode, perm int32) int32

//go:noescape
func gotraceback(*bool) int32

const _NoArgs = ^uintptr(0)

func newstack()
func newproc()
func lessstack()
func morestack()
func mstart()
func rt0_go()
func sigpanic()

// return0 is a stub used to return 0 from deferproc.
// It is called at the very end of deferproc to signal
// the calling Go function that it should not jump
// to deferreturn.
// in asm_*.s
func return0()
