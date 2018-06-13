// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Call fn with arg as its argument. Return what fn returns.
// fn is the raw pc value of the entry point of the desired function.
// Switches to the system stack, if not already there.
// Preserves the calling point as the location where a profiler traceback will begin.
//go:nosplit
func libcCall(fn, arg unsafe.Pointer) int32 {
	// Leave caller's PC/SP/G around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil {
		mp.libcallg.set(gp)
		mp.libcallpc = getcallerpc()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = getcallersp()
	}
	res := asmcgocall(fn, arg)
	if mp != nil {
		mp.libcallsp = 0
	}
	return res
}

// The *_trampoline functions convert from the Go calling convention to the C calling convention
// and then call the underlying libc function.  They are defined in sys_darwin_$ARCH.s.

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_init(attr *pthreadattr) int32 {
	return libcCall(unsafe.Pointer(funcPC(pthread_attr_init_trampoline)), unsafe.Pointer(&attr))
}
func pthread_attr_init_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_setstacksize(attr *pthreadattr, size uintptr) int32 {
	return libcCall(unsafe.Pointer(funcPC(pthread_attr_setstacksize_trampoline)), unsafe.Pointer(&attr))
}
func pthread_attr_setstacksize_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_setdetachstate(attr *pthreadattr, state int) int32 {
	return libcCall(unsafe.Pointer(funcPC(pthread_attr_setdetachstate_trampoline)), unsafe.Pointer(&attr))
}
func pthread_attr_setdetachstate_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_create(attr *pthreadattr, start uintptr, arg unsafe.Pointer) int32 {
	return libcCall(unsafe.Pointer(funcPC(pthread_create_trampoline)), unsafe.Pointer(&attr))
}
func pthread_create_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func raise(sig uint32) {
	libcCall(unsafe.Pointer(funcPC(raise_trampoline)), unsafe.Pointer(&sig))
}
func raise_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_self() (t pthread) {
	libcCall(unsafe.Pointer(funcPC(pthread_self_trampoline)), unsafe.Pointer(&t))
	return
}
func pthread_self_trampoline()

func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (unsafe.Pointer, int) {
	args := struct {
		addr            unsafe.Pointer
		n               uintptr
		prot, flags, fd int32
		off             uint32
		ret1            unsafe.Pointer
		ret2            int
	}{addr, n, prot, flags, fd, off, nil, 0}
	libcCall(unsafe.Pointer(funcPC(mmap_trampoline)), unsafe.Pointer(&args))
	return args.ret1, args.ret2
}
func mmap_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func munmap(addr unsafe.Pointer, n uintptr) {
	libcCall(unsafe.Pointer(funcPC(munmap_trampoline)), unsafe.Pointer(&addr))
}
func munmap_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func madvise(addr unsafe.Pointer, n uintptr, flags int32) {
	libcCall(unsafe.Pointer(funcPC(madvise_trampoline)), unsafe.Pointer(&addr))
}
func madvise_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func read(fd int32, p unsafe.Pointer, n int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(read_trampoline)), unsafe.Pointer(&fd))
}
func read_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func closefd(fd int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(close_trampoline)), unsafe.Pointer(&fd))
}
func close_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func exit(code int32) {
	libcCall(unsafe.Pointer(funcPC(exit_trampoline)), unsafe.Pointer(&code))
}
func exit_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func usleep(usec uint32) {
	libcCall(unsafe.Pointer(funcPC(usleep_trampoline)), unsafe.Pointer(&usec))
}
func usleep_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func write(fd uintptr, p unsafe.Pointer, n int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(write_trampoline)), unsafe.Pointer(&fd))
}
func write_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func open(name *byte, mode, perm int32) (ret int32) {
	return libcCall(unsafe.Pointer(funcPC(open_trampoline)), unsafe.Pointer(&name))
}
func open_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func nanotime() int64 {
	var r struct {
		t            int64  // raw timer
		numer, denom uint32 // conversion factors. nanoseconds = t * numer / denom.
	}
	libcCall(unsafe.Pointer(funcPC(nanotime_trampoline)), unsafe.Pointer(&r))
	// Note: Apple seems unconcerned about overflow here. See
	// https://developer.apple.com/library/content/qa/qa1398/_index.html
	// Note also, numer == denom == 1 is common.
	t := r.t
	if r.numer != 1 {
		t *= int64(r.numer)
	}
	if r.denom != 1 {
		t /= int64(r.denom)
	}
	return t
}
func nanotime_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func walltime() (int64, int32) {
	var t timeval
	libcCall(unsafe.Pointer(funcPC(walltime_trampoline)), unsafe.Pointer(&t))
	return int64(t.tv_sec), 1000 * t.tv_usec
}
func walltime_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigaction(sig uint32, new *usigactiont, old *usigactiont) {
	asmcgocall(unsafe.Pointer(funcPC(sigaction_trampoline)), unsafe.Pointer(&sig))
}
func sigaction_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigprocmask(how uint32, new *sigset, old *sigset) {
	asmcgocall(unsafe.Pointer(funcPC(sigprocmask_trampoline)), unsafe.Pointer(&how))
}
func sigprocmask_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigaltstack(new *stackt, old *stackt) {
	if new != nil && new.ss_flags&_SS_DISABLE != 0 && new.ss_size == 0 {
		// Despite the fact that Darwin's sigaltstack man page says it ignores the size
		// when SS_DISABLE is set, it doesn't. sigaltstack returns ENOMEM
		// if we don't give it a reasonable size.
		// ref: http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140421/214296.html
		new.ss_size = 32768
	}
	asmcgocall(unsafe.Pointer(funcPC(sigaltstack_trampoline)), unsafe.Pointer(&new))
}
func sigaltstack_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func raiseproc(sig uint32) {
	asmcgocall(unsafe.Pointer(funcPC(raiseproc_trampoline)), unsafe.Pointer(&sig))
}
func raiseproc_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func setitimer(mode int32, new, old *itimerval) {
	asmcgocall(unsafe.Pointer(funcPC(setitimer_trampoline)), unsafe.Pointer(&mode))
}
func setitimer_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32 {
	return asmcgocall(unsafe.Pointer(funcPC(sysctl_trampoline)), unsafe.Pointer(&mib))
}
func sysctl_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func fcntl(fd, cmd, arg int32) int32 {
	return asmcgocall(unsafe.Pointer(funcPC(fcntl_trampoline)), unsafe.Pointer(&fd))
}
func fcntl_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func kqueue() int32 {
	v := asmcgocall(unsafe.Pointer(funcPC(kqueue_trampoline)), nil)
	return v
}
func kqueue_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32 {
	return asmcgocall(unsafe.Pointer(funcPC(kevent_trampoline)), unsafe.Pointer(&kq))
}
func kevent_trampoline()

// Not used on Darwin, but must be defined.
func exitThread(wait *uint32) {
}

//go:nosplit
func closeonexec(fd int32) {
	fcntl(fd, _F_SETFD, _FD_CLOEXEC)
}

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_setstacksize pthread_attr_setstacksize "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_create pthread_create "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_exit exit "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_raise raise "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_open open "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_close close "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_read read "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_write write "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_mmap mmap "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_munmap munmap "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_madvise madvise "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_error __error "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_usleep usleep "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_mach_timebase_info mach_timebase_info "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_mach_absolute_time mach_absolute_time "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_gettimeofday gettimeofday "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sigaction sigaction "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_sigmask pthread_sigmask "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sigaltstack sigaltstack "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_getpid getpid "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_kill kill "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_setitimer setitimer "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sysctl sysctl "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_fcntl fcntl "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_kqueue kqueue "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_kevent kevent "/usr/lib/libSystem.B.dylib"

// Magic incantation to get libSystem actually dynamically linked.
// TODO: Why does the code require this?  See cmd/compile/internal/ld/go.go:210
//go:cgo_import_dynamic _ _ "/usr/lib/libSystem.B.dylib"
