// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

package runtime

import (
	"internal/abi"
	"unsafe"
)

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
func fcntl(fd, cmd, arg int32) int32 {
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(fcntl_trampoline)), unsafe.Pointer(&fd))
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
func exitThread(wait *uint32) {
}

//go:nosplit
func closeonexec(fd int32) {
	fcntl(fd, _F_SETFD, _FD_CLOEXEC)
}

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

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

//go:cgo_import_dynamic _ _ "libc.so"
