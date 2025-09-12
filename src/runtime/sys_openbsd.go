// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/runtime/atomic"
	"unsafe"
)

// The *_trampoline functions convert from the Go calling convention to the C calling convention
// and then call the underlying libc function. These are defined in sys_openbsd_$ARCH.s.

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_init(attr *pthreadattr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_init_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_init_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_destroy(attr *pthreadattr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_destroy_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_destroy_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_getstacksize(attr *pthreadattr, size *uintptr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_getstacksize_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	KeepAlive(size)
	return ret
}
func pthread_attr_getstacksize_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_setdetachstate(attr *pthreadattr, state int) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_setdetachstate_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_setdetachstate_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_create(attr *pthreadattr, start uintptr, arg unsafe.Pointer) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_create_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	KeepAlive(arg) // Just for consistency. Arg of course needs to be kept alive for the start function.
	return ret
}
func pthread_create_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func thrsleep(ident uintptr, clock_id int32, tsp *timespec, lock uintptr, abort *uint32) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(thrsleep_trampoline)), unsafe.Pointer(&ident))
	KeepAlive(tsp)
	KeepAlive(abort)
	return ret
}
func thrsleep_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func thrwakeup(ident uintptr, n int32) int32 {
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(thrwakeup_trampoline)), unsafe.Pointer(&ident))
}
func thrwakeup_trampoline()

//go:nosplit
func osyield() {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(sched_yield_trampoline)), unsafe.Pointer(nil))
}
func sched_yield_trampoline()

//go:nosplit
func osyield_no_g() {
	asmcgocall_no_g(unsafe.Pointer(abi.FuncPCABI0(sched_yield_trampoline)), unsafe.Pointer(nil))
}

// This is exported via linkname to assembly in runtime/cgo.
//
//go:linkname exit
//go:nosplit
//go:cgo_unsafe_args
func exit(code int32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(exit_trampoline)), unsafe.Pointer(&code))
}
func exit_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func getthrid() (tid int32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(getthrid_trampoline)), unsafe.Pointer(&tid))
	return
}
func getthrid_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func raiseproc(sig uint32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(raiseproc_trampoline)), unsafe.Pointer(&sig))
}
func raiseproc_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func thrkill(tid int32, sig int) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(thrkill_trampoline)), unsafe.Pointer(&tid))
}
func thrkill_trampoline()

// mmap is used to do low-level memory allocation via mmap. Don't allow stack
// splits, since this function (used by sysAlloc) is called in a lot of low-level
// parts of the runtime and callers often assume it won't acquire any locks.
//
//go:nosplit
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (unsafe.Pointer, int) {
	args := struct {
		addr            unsafe.Pointer
		n               uintptr
		prot, flags, fd int32
		off             uint32
		ret1            unsafe.Pointer
		ret2            int
	}{addr, n, prot, flags, fd, off, nil, 0}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(mmap_trampoline)), unsafe.Pointer(&args))
	KeepAlive(addr) // Just for consistency. Hopefully addr is not a Go address.
	return args.ret1, args.ret2
}
func mmap_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func munmap(addr unsafe.Pointer, n uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(munmap_trampoline)), unsafe.Pointer(&addr))
	KeepAlive(addr) // Just for consistency. Hopefully addr is not a Go address.
}
func munmap_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func madvise(addr unsafe.Pointer, n uintptr, flags int32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(madvise_trampoline)), unsafe.Pointer(&addr))
	KeepAlive(addr) // Just for consistency. Hopefully addr is not a Go address.
}
func madvise_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func open(name *byte, mode, perm int32) (ret int32) {
	ret = libcCall(unsafe.Pointer(abi.FuncPCABI0(open_trampoline)), unsafe.Pointer(&name))
	KeepAlive(name)
	return
}
func open_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func closefd(fd int32) int32 {
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(close_trampoline)), unsafe.Pointer(&fd))
}
func close_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func read(fd int32, p unsafe.Pointer, n int32) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(read_trampoline)), unsafe.Pointer(&fd))
	KeepAlive(p)
	return ret
}
func read_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func write1(fd uintptr, p unsafe.Pointer, n int32) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(write_trampoline)), unsafe.Pointer(&fd))
	KeepAlive(p)
	return ret
}
func write_trampoline()

func pipe2(flags int32) (r, w int32, errno int32) {
	var p [2]int32
	args := struct {
		p     unsafe.Pointer
		flags int32
	}{noescape(unsafe.Pointer(&p)), flags}
	errno = libcCall(unsafe.Pointer(abi.FuncPCABI0(pipe2_trampoline)), unsafe.Pointer(&args))
	return p[0], p[1], errno
}
func pipe2_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func setitimer(mode int32, new, old *itimerval) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(setitimer_trampoline)), unsafe.Pointer(&mode))
	KeepAlive(new)
	KeepAlive(old)
}
func setitimer_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func usleep(usec uint32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(usleep_trampoline)), unsafe.Pointer(&usec))
}
func usleep_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func usleep_no_g(usec uint32) {
	asmcgocall_no_g(unsafe.Pointer(abi.FuncPCABI0(usleep_trampoline)), unsafe.Pointer(&usec))
}

//go:nosplit
//go:cgo_unsafe_args
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(sysctl_trampoline)), unsafe.Pointer(&mib))
	KeepAlive(mib)
	KeepAlive(out)
	KeepAlive(size)
	KeepAlive(dst)
	return ret
}
func sysctl_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func fcntl(fd, cmd, arg int32) (ret int32, errno int32) {
	args := struct {
		fd, cmd, arg int32
		ret, errno   int32
	}{fd, cmd, arg, 0, 0}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(fcntl_trampoline)), unsafe.Pointer(&args))
	return args.ret, args.errno
}
func fcntl_trampoline()

//go:nosplit
func nanotime1() int64 {
	var ts timespec
	args := struct {
		clock_id int32
		tp       unsafe.Pointer
	}{_CLOCK_MONOTONIC, unsafe.Pointer(&ts)}
	if errno := libcCall(unsafe.Pointer(abi.FuncPCABI0(clock_gettime_trampoline)), unsafe.Pointer(&args)); errno < 0 {
		// Avoid growing the nosplit stack.
		systemstack(func() {
			println("runtime: errno", -errno)
			throw("clock_gettime failed")
		})
	}
	return ts.tv_sec*1e9 + int64(ts.tv_nsec)
}
func clock_gettime_trampoline()

//go:nosplit
func walltime() (int64, int32) {
	var ts timespec
	args := struct {
		clock_id int32
		tp       unsafe.Pointer
	}{_CLOCK_REALTIME, unsafe.Pointer(&ts)}
	if errno := libcCall(unsafe.Pointer(abi.FuncPCABI0(clock_gettime_trampoline)), unsafe.Pointer(&args)); errno < 0 {
		// Avoid growing the nosplit stack.
		systemstack(func() {
			println("runtime: errno", -errno)
			throw("clock_gettime failed")
		})
	}
	return ts.tv_sec, int32(ts.tv_nsec)
}

//go:nosplit
//go:cgo_unsafe_args
func kqueue() int32 {
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(kqueue_trampoline)), nil)
}
func kqueue_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(kevent_trampoline)), unsafe.Pointer(&kq))
	KeepAlive(ch)
	KeepAlive(ev)
	KeepAlive(ts)
	return ret
}
func kevent_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigaction(sig uint32, new *sigactiont, old *sigactiont) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(sigaction_trampoline)), unsafe.Pointer(&sig))
	KeepAlive(new)
	KeepAlive(old)
}
func sigaction_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigprocmask(how uint32, new *sigset, old *sigset) {
	// sigprocmask is called from sigsave, which is called from needm.
	// As such, we have to be able to run with no g here.
	asmcgocall_no_g(unsafe.Pointer(abi.FuncPCABI0(sigprocmask_trampoline)), unsafe.Pointer(&how))
	KeepAlive(new)
	KeepAlive(old)
}
func sigprocmask_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigaltstack(new *stackt, old *stackt) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(sigaltstack_trampoline)), unsafe.Pointer(&new))
	KeepAlive(new)
	KeepAlive(old)
}
func sigaltstack_trampoline()

// Not used on OpenBSD, but must be defined.
func exitThread(wait *atomic.Uint32) {
	throw("exitThread")
}

//go:nosplit
//go:cgo_unsafe_args
func issetugid() (ret int32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(issetugid_trampoline)), unsafe.Pointer(&ret))
	return
}
func issetugid_trampoline()

// The X versions of syscall expect the libc call to return a 64-bit result.
// Otherwise (the non-X version) expects a 32-bit result.
// This distinction is required because an error is indicated by returning -1,
// and we need to know whether to check 32 or 64 bits of the result.
// (Some libc functions that return 32 bits put junk in the upper 32 bits of AX.)

// golang.org/x/sys linknames syscall_syscall
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_syscall syscall.syscall
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscall(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscall()

//go:linkname syscall_syscallX syscall.syscallX
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscallX(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscallX)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscallX()

// golang.org/x/sys linknames syscall.syscall6
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_syscall6 syscall.syscall6
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall6)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscall6()

//go:linkname syscall_syscall6X syscall.syscall6X
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscall6X(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall6X)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscall6X()

// golang.org/x/sys linknames syscall.syscall10
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_syscall10 syscall.syscall10
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscall10(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 uintptr) (r1, r2, err uintptr) {
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall10)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscall10()

//go:linkname syscall_syscall10X syscall.syscall10X
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscall10X(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 uintptr) (r1, r2, err uintptr) {
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall10X)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscall10X()

// golang.org/x/sys linknames syscall_rawSyscall
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_rawSyscall syscall.rawSyscall
//go:nosplit
//go:cgo_unsafe_args
func syscall_rawSyscall(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall)), unsafe.Pointer(&fn))
	return
}

// golang.org/x/sys linknames syscall_rawSyscall6
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_rawSyscall6 syscall.rawSyscall6
//go:nosplit
//go:cgo_unsafe_args
func syscall_rawSyscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall6)), unsafe.Pointer(&fn))
	return
}

//go:linkname syscall_rawSyscall6X syscall.rawSyscall6X
//go:nosplit
//go:cgo_unsafe_args
func syscall_rawSyscall6X(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall6X)), unsafe.Pointer(&fn))
	return
}

//go:linkname syscall_rawSyscall10X syscall.rawSyscall10X
//go:nosplit
//go:cgo_unsafe_args
func syscall_rawSyscall10X(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 uintptr) (r1, r2, err uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall10X)), unsafe.Pointer(&fn))
	return
}

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "libpthread.so"
//go:cgo_import_dynamic libc_pthread_attr_destroy pthread_attr_destroy "libpthread.so"
//go:cgo_import_dynamic libc_pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.so"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.so"
//go:cgo_import_dynamic libc_pthread_create pthread_create "libpthread.so"
//go:cgo_import_dynamic libc_pthread_sigmask pthread_sigmask "libpthread.so"

//go:cgo_import_dynamic libc_thrsleep __thrsleep "libc.so"
//go:cgo_import_dynamic libc_thrwakeup __thrwakeup "libc.so"
//go:cgo_import_dynamic libc_sched_yield sched_yield "libc.so"

//go:cgo_import_dynamic libc_errno __errno "libc.so"
//go:cgo_import_dynamic libc_exit exit "libc.so"
//go:cgo_import_dynamic libc_getthrid getthrid "libc.so"
//go:cgo_import_dynamic libc_sched_yield sched_yield "libc.so"
//go:cgo_import_dynamic libc_thrkill thrkill "libc.so"

//go:cgo_import_dynamic libc_mmap mmap "libc.so"
//go:cgo_import_dynamic libc_munmap munmap "libc.so"
//go:cgo_import_dynamic libc_madvise madvise "libc.so"

//go:cgo_import_dynamic libc_open open "libc.so"
//go:cgo_import_dynamic libc_close close "libc.so"
//go:cgo_import_dynamic libc_read read "libc.so"
//go:cgo_import_dynamic libc_write write "libc.so"
//go:cgo_import_dynamic libc_pipe2 pipe2 "libc.so"

//go:cgo_import_dynamic libc_clock_gettime clock_gettime "libc.so"
//go:cgo_import_dynamic libc_setitimer setitimer "libc.so"
//go:cgo_import_dynamic libc_usleep usleep "libc.so"
//go:cgo_import_dynamic libc_sysctl sysctl "libc.so"
//go:cgo_import_dynamic libc_fcntl fcntl "libc.so"
//go:cgo_import_dynamic libc_getpid getpid "libc.so"
//go:cgo_import_dynamic libc_kill kill "libc.so"
//go:cgo_import_dynamic libc_kqueue kqueue "libc.so"
//go:cgo_import_dynamic libc_kevent kevent "libc.so"

//go:cgo_import_dynamic libc_sigaction sigaction "libc.so"
//go:cgo_import_dynamic libc_sigaltstack sigaltstack "libc.so"

//go:cgo_import_dynamic libc_issetugid issetugid "libc.so"

//go:cgo_import_dynamic _ _ "libpthread.so"
//go:cgo_import_dynamic _ _ "libc.so"
