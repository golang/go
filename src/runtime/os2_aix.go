// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains main runtime AIX syscalls.
// Pollset syscalls are in netpoll_aix.go.
// The implementation is based on Solaris and Windows.
// Each syscall is made by calling its libc symbol using asmcgocall and asmsyscall6
// asssembly functions.

package runtime

import (
	"unsafe"
)

// Symbols imported for __start function.

//go:cgo_import_dynamic libc___n_pthreads __n_pthreads "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libc___mod_init __mod_init "libc.a/shr_64.o"
//go:linkname libc___n_pthreads libc___n_pthread
//go:linkname libc___mod_init libc___mod_init

var (
	libc___n_pthread,
	libc___mod_init libFunc
)

// Syscalls

//go:cgo_import_dynamic libc__Errno _Errno "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_clock_gettime clock_gettime "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_close close "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_exit exit "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_getpid getpid "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_getsystemcfg getsystemcfg "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_kill kill "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_madvise madvise "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_malloc malloc "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_mmap mmap "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_munmap munmap "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_open open "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_pipe pipe "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_raise raise "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_read read "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sched_yield sched_yield "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sem_init sem_init "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sem_post sem_post "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sem_timedwait sem_timedwait "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sem_wait sem_wait "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_setitimer setitimer "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sigaction sigaction "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sigaltstack sigaltstack "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_sysconf sysconf "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_usleep usleep "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_write write "libc.a/shr_64.o"

//go:cgo_import_dynamic libpthread___pth_init __pth_init "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_attr_destroy pthread_attr_destroy "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_attr_init pthread_attr_init "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_attr_getstacksize pthread_attr_getstacksize "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_attr_setstacksize pthread_attr_setstacksize "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_attr_setstackaddr pthread_attr_setstackaddr "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_create pthread_create "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic libpthread_sigthreadmask sigthreadmask "libpthread.a/shr_xpg5_64.o"

//go:linkname libc__Errno libc__Errno
//go:linkname libc_clock_gettime libc_clock_gettime
//go:linkname libc_close libc_close
//go:linkname libc_exit libc_exit
//go:linkname libc_getpid libc_getpid
//go:linkname libc_getsystemcfg libc_getsystemcfg
//go:linkname libc_kill libc_kill
//go:linkname libc_madvise libc_madvise
//go:linkname libc_malloc libc_malloc
//go:linkname libc_mmap libc_mmap
//go:linkname libc_munmap libc_munmap
//go:linkname libc_open libc_open
//go:linkname libc_pipe libc_pipe
//go:linkname libc_raise libc_raise
//go:linkname libc_read libc_read
//go:linkname libc_sched_yield libc_sched_yield
//go:linkname libc_sem_init libc_sem_init
//go:linkname libc_sem_post libc_sem_post
//go:linkname libc_sem_timedwait libc_sem_timedwait
//go:linkname libc_sem_wait libc_sem_wait
//go:linkname libc_setitimer libc_setitimer
//go:linkname libc_sigaction libc_sigaction
//go:linkname libc_sigaltstack libc_sigaltstack
//go:linkname libc_sysconf libc_sysconf
//go:linkname libc_usleep libc_usleep
//go:linkname libc_write libc_write

//go:linkname libpthread___pth_init libpthread___pth_init
//go:linkname libpthread_attr_destroy libpthread_attr_destroy
//go:linkname libpthread_attr_init libpthread_attr_init
//go:linkname libpthread_attr_getstacksize libpthread_attr_getstacksize
//go:linkname libpthread_attr_setstacksize libpthread_attr_setstacksize
//go:linkname libpthread_attr_setdetachstate libpthread_attr_setdetachstate
//go:linkname libpthread_attr_setstackaddr libpthread_attr_setstackaddr
//go:linkname libpthread_create libpthread_create
//go:linkname libpthread_sigthreadmask libpthread_sigthreadmask

var (
	//libc
	libc__Errno,
	libc_clock_gettime,
	libc_close,
	libc_exit,
	libc_getpid,
	libc_getsystemcfg,
	libc_kill,
	libc_madvise,
	libc_malloc,
	libc_mmap,
	libc_munmap,
	libc_open,
	libc_pipe,
	libc_raise,
	libc_read,
	libc_sched_yield,
	libc_sem_init,
	libc_sem_post,
	libc_sem_timedwait,
	libc_sem_wait,
	libc_setitimer,
	libc_sigaction,
	libc_sigaltstack,
	libc_sysconf,
	libc_usleep,
	libc_write,
	//libpthread
	libpthread___pth_init,
	libpthread_attr_destroy,
	libpthread_attr_init,
	libpthread_attr_getstacksize,
	libpthread_attr_setstacksize,
	libpthread_attr_setdetachstate,
	libpthread_attr_setstackaddr,
	libpthread_create,
	libpthread_sigthreadmask libFunc
)

type libFunc uintptr

// asmsyscall6 calls the libc symbol using a C convention.
// It's defined in sys_aix_ppc64.go.
var asmsyscall6 libFunc

//go:nowritebarrier
//go:nosplit
func syscall0(fn *libFunc) (r, err uintptr) {
	c := &getg().m.libcall
	c.fn = uintptr(unsafe.Pointer(fn))
	c.n = 0
	c.args = uintptr(noescape(unsafe.Pointer(&fn))) // it's unused but must be non-nil, otherwise crashes

	asmcgocall(unsafe.Pointer(&asmsyscall6), unsafe.Pointer(c))

	return c.r1, c.err
}

//go:nowritebarrier
//go:nosplit
func syscall1(fn *libFunc, a0 uintptr) (r, err uintptr) {
	c := &getg().m.libcall
	c.fn = uintptr(unsafe.Pointer(fn))
	c.n = 1
	c.args = uintptr(noescape(unsafe.Pointer(&a0)))

	asmcgocall(unsafe.Pointer(&asmsyscall6), unsafe.Pointer(c))

	return c.r1, c.err
}

//go:nowritebarrier
//go:nosplit
func syscall2(fn *libFunc, a0, a1 uintptr) (r, err uintptr) {
	c := &getg().m.libcall
	c.fn = uintptr(unsafe.Pointer(fn))
	c.n = 2
	c.args = uintptr(noescape(unsafe.Pointer(&a0)))

	asmcgocall(unsafe.Pointer(&asmsyscall6), unsafe.Pointer(c))

	return c.r1, c.err
}

//go:nowritebarrier
//go:nosplit
func syscall3(fn *libFunc, a0, a1, a2 uintptr) (r, err uintptr) {
	c := &getg().m.libcall
	c.fn = uintptr(unsafe.Pointer(fn))
	c.n = 3
	c.args = uintptr(noescape(unsafe.Pointer(&a0)))

	asmcgocall(unsafe.Pointer(&asmsyscall6), unsafe.Pointer(c))

	return c.r1, c.err
}

//go:nowritebarrier
//go:nosplit
func syscall4(fn *libFunc, a0, a1, a2, a3 uintptr) (r, err uintptr) {
	c := &getg().m.libcall
	c.fn = uintptr(unsafe.Pointer(fn))
	c.n = 4
	c.args = uintptr(noescape(unsafe.Pointer(&a0)))

	asmcgocall(unsafe.Pointer(&asmsyscall6), unsafe.Pointer(c))

	return c.r1, c.err
}

//go:nowritebarrier
//go:nosplit
func syscall5(fn *libFunc, a0, a1, a2, a3, a4 uintptr) (r, err uintptr) {
	c := &getg().m.libcall
	c.fn = uintptr(unsafe.Pointer(fn))
	c.n = 5
	c.args = uintptr(noescape(unsafe.Pointer(&a0)))

	asmcgocall(unsafe.Pointer(&asmsyscall6), unsafe.Pointer(c))

	return c.r1, c.err
}

//go:nowritebarrier
//go:nosplit
func syscall6(fn *libFunc, a0, a1, a2, a3, a4, a5 uintptr) (r, err uintptr) {
	c := &getg().m.libcall
	c.fn = uintptr(unsafe.Pointer(fn))
	c.n = 6
	c.args = uintptr(noescape(unsafe.Pointer(&a0)))

	asmcgocall(unsafe.Pointer(&asmsyscall6), unsafe.Pointer(c))

	return c.r1, c.err
}

//go:nosplit
func exit(code int32) {
	syscall1(&libc_exit, uintptr(code))
}

//go:nosplit
func write(fd uintptr, p unsafe.Pointer, n int32) int32 {
	r, _ := syscall3(&libc_write, uintptr(fd), uintptr(p), uintptr(n))
	return int32(r)

}

//go:nosplit
func read(fd int32, p unsafe.Pointer, n int32) int32 {
	r, _ := syscall3(&libc_read, uintptr(fd), uintptr(p), uintptr(n))
	return int32(r)
}

//go:nosplit
func open(name *byte, mode, perm int32) int32 {
	r, _ := syscall3(&libc_open, uintptr(unsafe.Pointer(name)), uintptr(mode), uintptr(perm))
	return int32(r)
}

//go:nosplit
func closefd(fd int32) int32 {
	r, _ := syscall1(&libc_close, uintptr(fd))
	return int32(r)
}

//go:nosplit
func pipe(fd *int32) int32 {
	r, _ := syscall1(&libc_pipe, uintptr(unsafe.Pointer(fd)))
	return int32(r)
}

// mmap calls the mmap system call.
// We only pass the lower 32 bits of file offset to the
// assembly routine; the higher bits (if required), should be provided
// by the assembly routine as 0.
// The err result is an OS error code such as ENOMEM.
//go:nosplit
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (p unsafe.Pointer, err int) {
	r, err0 := syscall6(&libc_mmap, uintptr(addr), uintptr(n), uintptr(prot), uintptr(flags), uintptr(fd), uintptr(off))
	return unsafe.Pointer(r), int(err0)
}

//go:nosplit
func munmap(addr unsafe.Pointer, n uintptr) {
	r, err := syscall2(&libc_munmap, uintptr(addr), uintptr(n))
	if int32(r) == -1 {
		println("syscall munmap failed: ", hex(err))
		throw("syscall munmap")
	}
}

//go:nosplit
func madvise(addr unsafe.Pointer, n uintptr, flags int32) {
	r, err := syscall3(&libc_madvise, uintptr(addr), uintptr(n), uintptr(flags))
	if int32(r) == -1 {
		println("syscall madvise failed: ", hex(err))
		throw("syscall madvise")
	}
}

//go:nosplit
func sigaction(sig uintptr, new, old *sigactiont) {
	r, err := syscall3(&libc_sigaction, sig, uintptr(unsafe.Pointer(new)), uintptr(unsafe.Pointer(old)))
	if int32(r) == -1 {
		println("Sigaction failed for sig: ", sig, " with error:", hex(err))
		throw("syscall sigaction")
	}
}

//go:nosplit
func sigaltstack(new, old *stackt) {
	r, err := syscall2(&libc_sigaltstack, uintptr(unsafe.Pointer(new)), uintptr(unsafe.Pointer(old)))
	if int32(r) == -1 {
		println("syscall sigaltstack failed: ", hex(err))
		throw("syscall sigaltstack")
	}
}

//go:nosplit
func getsystemcfg(label uint) uintptr {
	r, _ := syscall1(&libc_getsystemcfg, uintptr(label))
	return r
}

//go:nosplit
func usleep(us uint32) {
	r, err := syscall1(&libc_usleep, uintptr(us))
	if int32(r) == -1 {
		println("syscall usleep failed: ", hex(err))
		throw("syscall usleep")
	}
}

//go:nosplit
func clock_gettime(clockid int32, tp *timespec) int32 {
	r, _ := syscall2(&libc_clock_gettime, uintptr(clockid), uintptr(unsafe.Pointer(tp)))
	return int32(r)
}

//go:nosplit
func setitimer(mode int32, new, old *itimerval) {
	r, err := syscall3(&libc_setitimer, uintptr(mode), uintptr(unsafe.Pointer(new)), uintptr(unsafe.Pointer(old)))
	if int32(r) == -1 {
		println("syscall setitimer failed: ", hex(err))
		throw("syscall setitimer")
	}
}

//go:nosplit
func malloc(size uintptr) unsafe.Pointer {
	r, _ := syscall1(&libc_malloc, size)
	return unsafe.Pointer(r)
}

//go:nosplit
func sem_init(sem *semt, pshared int32, value uint32) int32 {
	r, _ := syscall3(&libc_sem_init, uintptr(unsafe.Pointer(sem)), uintptr(pshared), uintptr(value))
	return int32(r)
}

//go:nosplit
func sem_wait(sem *semt) (int32, int32) {
	r, err := syscall1(&libc_sem_wait, uintptr(unsafe.Pointer(sem)))
	return int32(r), int32(err)
}

//go:nosplit
func sem_post(sem *semt) int32 {
	r, _ := syscall1(&libc_sem_post, uintptr(unsafe.Pointer(sem)))
	return int32(r)
}

//go:nosplit
func sem_timedwait(sem *semt, timeout *timespec) (int32, int32) {
	r, err := syscall2(&libc_sem_timedwait, uintptr(unsafe.Pointer(sem)), uintptr(unsafe.Pointer(timeout)))
	return int32(r), int32(err)
}

//go:nosplit
func raise(sig uint32) {
	r, err := syscall1(&libc_raise, uintptr(sig))
	if int32(r) == -1 {
		println("syscall raise failed: ", hex(err))
		throw("syscall raise")
	}
}

//go:nosplit
func raiseproc(sig uint32) {
	pid, err := syscall0(&libc_getpid)
	if int32(pid) == -1 {
		println("syscall getpid failed: ", hex(err))
		throw("syscall raiseproc")
	}

	syscall2(&libc_kill, pid, uintptr(sig))
}

func osyield1()

//go:nosplit
func osyield() {
	_g_ := getg()

	// Check the validity of m because we might be called in cgo callback
	// path early enough where there isn't a m available yet.
	if _g_ != nil && _g_.m != nil {
		r, err := syscall0(&libc_sched_yield)
		if int32(r) == -1 {
			println("syscall osyield failed: ", hex(err))
			throw("syscall osyield")
		}
		return
	}
	osyield1()
}

//go:nosplit
func sysconf(name int32) uintptr {
	r, _ := syscall1(&libc_sysconf, uintptr(name))
	if int32(r) == -1 {
		throw("syscall sysconf")
	}
	return r

}

// pthread functions returns its error code in the main return value
// Therefore, err returns by syscall means nothing and must not be used

//go:nosplit
func pthread_attr_destroy(attr *pthread_attr) int32 {
	r, _ := syscall1(&libpthread_attr_destroy, uintptr(unsafe.Pointer(attr)))
	return int32(r)
}

//go:nosplit
func pthread_attr_init(attr *pthread_attr) int32 {
	r, _ := syscall1(&libpthread_attr_init, uintptr(unsafe.Pointer(attr)))
	return int32(r)
}

//go:nosplit
func pthread_attr_setdetachstate(attr *pthread_attr, state int32) int32 {
	r, _ := syscall2(&libpthread_attr_setdetachstate, uintptr(unsafe.Pointer(attr)), uintptr(state))
	return int32(r)
}

//go:nosplit
func pthread_attr_setstackaddr(attr *pthread_attr, stk unsafe.Pointer) int32 {
	r, _ := syscall2(&libpthread_attr_setstackaddr, uintptr(unsafe.Pointer(attr)), uintptr(stk))
	return int32(r)
}

//go:nosplit
func pthread_attr_getstacksize(attr *pthread_attr, size *uint64) int32 {
	r, _ := syscall2(&libpthread_attr_getstacksize, uintptr(unsafe.Pointer(attr)), uintptr(unsafe.Pointer(size)))
	return int32(r)
}

//go:nosplit
func pthread_attr_setstacksize(attr *pthread_attr, size uint64) int32 {
	r, _ := syscall2(&libpthread_attr_setstacksize, uintptr(unsafe.Pointer(attr)), uintptr(size))
	return int32(r)
}

//go:nosplit
func pthread_create(tid *pthread, attr *pthread_attr, fn *funcDescriptor, arg unsafe.Pointer) int32 {
	r, _ := syscall4(&libpthread_create, uintptr(unsafe.Pointer(tid)), uintptr(unsafe.Pointer(attr)), uintptr(unsafe.Pointer(fn)), uintptr(arg))
	return int32(r)
}

// On multi-thread program, sigprocmask must not be called.
// It's replaced by sigthreadmask.
//go:nosplit
func sigprocmask(how int32, new, old *sigset) {
	r, err := syscall3(&libpthread_sigthreadmask, uintptr(how), uintptr(unsafe.Pointer(new)), uintptr(unsafe.Pointer(old)))
	if int32(r) != 0 {
		println("syscall sigthreadmask failed: ", hex(err))
		throw("syscall sigthreadmask")
	}
}
