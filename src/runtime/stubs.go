// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Declarations for runtime services implemented in C or assembly.

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

// in runtime.c
func getg() *g
func acquirem() *m
func releasem(mp *m)
func gomcache() *mcache
func readgstatus(*g) uint32 // proc.c

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

// onMsignal is like onM but is allowed to be used in code that
// might run on the gsignal stack. Code running on a signal stack
// may be interrupting an onM sequence on the main stack, so
// if the onMsignal calling sequence writes to ptrarg/scalararg,
// it must first save the old values and then restore them when
// finished. As an exception to the rule, it is fine not to save and
// restore the values if the program is trying to crash rather than
// return from the signal handler.
// Once all the runtime is written in Go, there will be no ptrarg/scalararg
// and the distinction between onM and onMsignal (and perhaps mcall)
// can go away.
//
// If onMsignal is called from a gsignal stack, it invokes fn directly,
// without a stack switch. Otherwise onMsignal behaves like onM.
//
//go:noescape
func onM_signalok(fn func())

func badonm() {
	gothrow("onM called from signal goroutine")
}

// C functions that run on the M stack.
// Call using mcall.
func gosched_m(*g)
func park_m(*g)
func recovery_m(*g)

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
func startpanic_m()
func dopanic_m()
func readmemstats_m()
func writeheapdump_m()

// memclr clears n bytes starting at ptr.
// in memclr_*.s
//go:noescape
func memclr(ptr unsafe.Pointer, n uintptr)

// memmove copies n bytes from "from" to "to".
// in memmove_*.s
//go:noescape
func memmove(to unsafe.Pointer, from unsafe.Pointer, n uintptr)

func starttheworld()
func stoptheworld()
func newextram()
func lockOSThread()
func unlockOSThread()

// exported value for testing
var hashLoad = loadFactor

// in asm_*.s
func fastrand1() uint32

// in asm_*.s
//go:noescape
func memeq(a, b unsafe.Pointer, size uintptr) bool

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
func reentersyscall(pc uintptr, sp unsafe.Pointer)
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

// careful: cputicks is not guaranteed to be monotonic!  In particular, we have
// noticed drift between cpus on certain os/arch combinations.  See issue 8976.
func cputicks() int64

func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer
func munmap(addr unsafe.Pointer, n uintptr)
func madvise(addr unsafe.Pointer, n uintptr, flags int32)
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

// getcallerpc returns the program counter (PC) of its caller's caller.
// getcallersp returns the stack pointer (SP) of its caller's caller.
// For both, the argp must be a pointer to the caller's first function argument.
// The implementation may or may not use argp, depending on
// the architecture.
//
// For example:
//
//	func f(arg1, arg2, arg3 int) {
//		pc := getcallerpc(unsafe.Pointer(&arg1))
//		sp := getcallerpc(unsafe.Pointer(&arg2))
//	}
//
// These two lines find the PC and SP immediately following
// the call to f (where f will return).
//
// The call to getcallerpc and getcallersp must be done in the
// frame being asked about. It would not be correct for f to pass &arg1
// to another function g and let g call getcallerpc/getcallersp.
// The call inside g might return information about g's caller or
// information about f's caller or complete garbage.
//
// The result of getcallersp is correct at the time of the return,
// but it may be invalidated by any subsequent call to a function
// that might relocate the stack in order to grow or shrink it.
// A general rule is that the result of getcallersp should be used
// immediately and can only be passed to nosplit functions.

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
func morestack()
func mstart()
func rt0_go()

// return0 is a stub used to return 0 from deferproc.
// It is called at the very end of deferproc to signal
// the calling Go function that it should not jump
// to deferreturn.
// in asm_*.s
func return0()

// thunk to call time.now.
func timenow() (sec int64, nsec int32)

// in asm_*.s
// not called directly; definitions here supply type information for traceback.
func call16(fn, arg unsafe.Pointer, n, retoffset uint32)
func call32(fn, arg unsafe.Pointer, n, retoffset uint32)
func call64(fn, arg unsafe.Pointer, n, retoffset uint32)
func call128(fn, arg unsafe.Pointer, n, retoffset uint32)
func call256(fn, arg unsafe.Pointer, n, retoffset uint32)
func call512(fn, arg unsafe.Pointer, n, retoffset uint32)
func call1024(fn, arg unsafe.Pointer, n, retoffset uint32)
func call2048(fn, arg unsafe.Pointer, n, retoffset uint32)
func call4096(fn, arg unsafe.Pointer, n, retoffset uint32)
func call8192(fn, arg unsafe.Pointer, n, retoffset uint32)
func call16384(fn, arg unsafe.Pointer, n, retoffset uint32)
func call32768(fn, arg unsafe.Pointer, n, retoffset uint32)
func call65536(fn, arg unsafe.Pointer, n, retoffset uint32)
func call131072(fn, arg unsafe.Pointer, n, retoffset uint32)
func call262144(fn, arg unsafe.Pointer, n, retoffset uint32)
func call524288(fn, arg unsafe.Pointer, n, retoffset uint32)
func call1048576(fn, arg unsafe.Pointer, n, retoffset uint32)
func call2097152(fn, arg unsafe.Pointer, n, retoffset uint32)
func call4194304(fn, arg unsafe.Pointer, n, retoffset uint32)
func call8388608(fn, arg unsafe.Pointer, n, retoffset uint32)
func call16777216(fn, arg unsafe.Pointer, n, retoffset uint32)
func call33554432(fn, arg unsafe.Pointer, n, retoffset uint32)
func call67108864(fn, arg unsafe.Pointer, n, retoffset uint32)
func call134217728(fn, arg unsafe.Pointer, n, retoffset uint32)
func call268435456(fn, arg unsafe.Pointer, n, retoffset uint32)
func call536870912(fn, arg unsafe.Pointer, n, retoffset uint32)
func call1073741824(fn, arg unsafe.Pointer, n, retoffset uint32)
