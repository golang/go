// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix

package runtime

import (
	"unsafe"
)

const (
	threadStackSize = 0x100000 // size of a thread stack allocated by OS
)

// funcDescriptor is a structure representing a function descriptor
// A variable with this type is always created in assembler
type funcDescriptor struct {
	fn         uintptr
	toc        uintptr
	envPointer uintptr // unused in Golang
}

type mOS struct {
	waitsema uintptr // semaphore for parking on locks
	perrno   uintptr // pointer to tls errno
}

//go:nosplit
func semacreate(mp *m) {
	if mp.waitsema != 0 {
		return
	}

	var sem *semt

	// Call libc's malloc rather than malloc. This will
	// allocate space on the C heap. We can't call mallocgc
	// here because it could cause a deadlock.
	sem = (*semt)(malloc(unsafe.Sizeof(*sem)))
	if sem_init(sem, 0, 0) != 0 {
		throw("sem_init")
	}
	mp.waitsema = uintptr(unsafe.Pointer(sem))
}

//go:nosplit
func semasleep(ns int64) int32 {
	_m_ := getg().m
	if ns >= 0 {
		var ts timespec

		if clock_gettime(_CLOCK_REALTIME, &ts) != 0 {
			throw("clock_gettime")
		}
		ts.tv_sec += ns / 1e9
		ts.tv_nsec += ns % 1e9
		if ts.tv_nsec >= 1e9 {
			ts.tv_sec++
			ts.tv_nsec -= 1e9
		}

		if r, err := sem_timedwait((*semt)(unsafe.Pointer(_m_.waitsema)), &ts); r != 0 {
			if err == _ETIMEDOUT || err == _EAGAIN || err == _EINTR {
				return -1
			}
			println("sem_timedwait err ", err, " ts.tv_sec ", ts.tv_sec, " ts.tv_nsec ", ts.tv_nsec, " ns ", ns, " id ", _m_.id)
			throw("sem_timedwait")
		}
		return 0
	}
	for {
		r1, err := sem_wait((*semt)(unsafe.Pointer(_m_.waitsema)))
		if r1 == 0 {
			break
		}
		if err == _EINTR {
			continue
		}
		throw("sem_wait")
	}
	return 0
}

//go:nosplit
func semawakeup(mp *m) {
	if sem_post((*semt)(unsafe.Pointer(mp.waitsema))) != 0 {
		throw("sem_post")
	}
}

func osinit() {
	ncpu = int32(sysconf(__SC_NPROCESSORS_ONLN))
	physPageSize = sysconf(__SC_PAGE_SIZE)
}

// newosproc0 is a version of newosproc that can be called before the runtime
// is initialized.
//
// This function is not safe to use after initialization as it does not pass an M as fnarg.
//
//go:nosplit
func newosproc0(stacksize uintptr, fn *funcDescriptor) {
	var (
		attr pthread_attr
		oset sigset
		tid  pthread
	)

	if pthread_attr_init(&attr) != 0 {
		write(2, unsafe.Pointer(&failthreadcreate[0]), int32(len(failthreadcreate)))
		exit(1)
	}

	if pthread_attr_setstacksize(&attr, threadStackSize) != 0 {
		write(2, unsafe.Pointer(&failthreadcreate[0]), int32(len(failthreadcreate)))
		exit(1)
	}

	if pthread_attr_setdetachstate(&attr, _PTHREAD_CREATE_DETACHED) != 0 {
		write(2, unsafe.Pointer(&failthreadcreate[0]), int32(len(failthreadcreate)))
		exit(1)
	}

	// Disable signals during create, so that the new thread starts
	// with signals disabled. It will enable them in minit.
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	var ret int32
	for tries := 0; tries < 20; tries++ {
		// pthread_create can fail with EAGAIN for no reasons
		// but it will be ok if it retries.
		ret = pthread_create(&tid, &attr, fn, nil)
		if ret != _EAGAIN {
			break
		}
		usleep(uint32(tries+1) * 1000) // Milliseconds.
	}
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if ret != 0 {
		write(2, unsafe.Pointer(&failthreadcreate[0]), int32(len(failthreadcreate)))
		exit(1)
	}

}

var failthreadcreate = []byte("runtime: failed to create new OS thread\n")

// Called to do synchronous initialization of Go code built with
// -buildmode=c-archive or -buildmode=c-shared.
// None of the Go runtime is initialized.
//go:nosplit
//go:nowritebarrierrec
func libpreinit() {
	initsig(true)
}

// Ms related functions
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024) // AIX wants >= 8K
	mp.gsignal.m = mp
}

// errno address must be retrieved by calling _Errno libc function.
// This will return a pointer to errno
func miniterrno() {
	mp := getg().m
	r, _ := syscall0(&libc__Errno)
	mp.perrno = r

}

func minit() {
	miniterrno()
	minitSignals()
	getg().m.procid = uint64(pthread_self())
}

func unminit() {
	unminitSignals()
}

// Called from exitm, but not from drop, to undo the effect of thread-owned
// resources in minit, semacreate, or elsewhere. Do not take locks after calling this.
func mdestroy(mp *m) {
}

// tstart is a function descriptor to _tstart defined in assembly.
var tstart funcDescriptor

func newosproc(mp *m) {
	var (
		attr pthread_attr
		oset sigset
		tid  pthread
	)

	if pthread_attr_init(&attr) != 0 {
		throw("pthread_attr_init")
	}

	if pthread_attr_setstacksize(&attr, threadStackSize) != 0 {
		throw("pthread_attr_getstacksize")
	}

	if pthread_attr_setdetachstate(&attr, _PTHREAD_CREATE_DETACHED) != 0 {
		throw("pthread_attr_setdetachstate")
	}

	// Disable signals during create, so that the new thread starts
	// with signals disabled. It will enable them in minit.
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	var ret int32
	for tries := 0; tries < 20; tries++ {
		// pthread_create can fail with EAGAIN for no reasons
		// but it will be ok if it retries.
		ret = pthread_create(&tid, &attr, &tstart, unsafe.Pointer(mp))
		if ret != _EAGAIN {
			break
		}
		usleep(uint32(tries+1) * 1000) // Milliseconds.
	}
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if ret != 0 {
		print("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", ret, ")\n")
		if ret == _EAGAIN {
			println("runtime: may need to increase max user processes (ulimit -u)")
		}
		throw("newosproc")
	}

}

func exitThread(wait *uint32) {
	// We should never reach exitThread on AIX because we let
	// libc clean up threads.
	throw("exitThread")
}

var urandom_dev = []byte("/dev/urandom\x00")

//go:nosplit
func getRandomData(r []byte) {
	fd := open(&urandom_dev[0], 0 /* O_RDONLY */, 0)
	n := read(fd, unsafe.Pointer(&r[0]), int32(len(r)))
	closefd(fd)
	extendRandom(r, int(n))
}

func goenvs() {
	goenvs_unix()
}

/* SIGNAL */

const (
	_NSIG = 256
)

// sigtramp is a function descriptor to _sigtramp defined in assembly
var sigtramp funcDescriptor

//go:nosplit
//go:nowritebarrierrec
func setsig(i uint32, fn uintptr) {
	var sa sigactiont
	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK | _SA_RESTART
	sa.sa_mask = sigset_all
	if fn == funcPC(sighandler) {
		fn = uintptr(unsafe.Pointer(&sigtramp))
	}
	sa.sa_handler = fn
	sigaction(uintptr(i), &sa, nil)

}

//go:nosplit
//go:nowritebarrierrec
func setsigstack(i uint32) {
	var sa sigactiont
	sigaction(uintptr(i), nil, &sa)
	if sa.sa_flags&_SA_ONSTACK != 0 {
		return
	}
	sa.sa_flags |= _SA_ONSTACK
	sigaction(uintptr(i), &sa, nil)
}

//go:nosplit
//go:nowritebarrierrec
func getsig(i uint32) uintptr {
	var sa sigactiont
	sigaction(uintptr(i), nil, &sa)
	return sa.sa_handler
}

// setSignaltstackSP sets the ss_sp field of a stackt.
//go:nosplit
func setSignalstackSP(s *stackt, sp uintptr) {
	*(*uintptr)(unsafe.Pointer(&s.ss_sp)) = sp
}

//go:nosplit
func (c *sigctxt) fixsigcode(sig uint32) {
	switch sig {
	case _SIGPIPE:
		// For SIGPIPE, c.sigcode() isn't set to _SI_USER as on Linux.
		// Therefore, raisebadsignal won't raise SIGPIPE again if
		// it was deliver in a non-Go thread.
		c.set_sigcode(_SI_USER)
	}
}

//go:nosplit
//go:nowritebarrierrec
func sigaddset(mask *sigset, i int) {
	(*mask)[(i-1)/64] |= 1 << ((uint32(i) - 1) & 63)
}

func sigdelset(mask *sigset, i int) {
	(*mask)[(i-1)/64] &^= 1 << ((uint32(i) - 1) & 63)
}

const (
	_CLOCK_REALTIME  = 9
	_CLOCK_MONOTONIC = 10
)

//go:nosplit
func nanotime1() int64 {
	tp := &timespec{}
	if clock_gettime(_CLOCK_REALTIME, tp) != 0 {
		throw("syscall clock_gettime failed")
	}
	return tp.tv_sec*1000000000 + tp.tv_nsec
}

func walltime1() (sec int64, nsec int32) {
	ts := &timespec{}
	if clock_gettime(_CLOCK_REALTIME, ts) != 0 {
		throw("syscall clock_gettime failed")
	}
	return ts.tv_sec, int32(ts.tv_nsec)
}

//go:nosplit
func fcntl(fd, cmd, arg int32) int32 {
	r, _ := syscall3(&libc_fcntl, uintptr(fd), uintptr(cmd), uintptr(arg))
	return int32(r)
}

//go:nosplit
func closeonexec(fd int32) {
	fcntl(fd, _F_SETFD, _FD_CLOEXEC)
}

//go:nosplit
func setNonblock(fd int32) {
	flags := fcntl(fd, _F_GETFL, 0)
	fcntl(fd, _F_SETFL, flags|_O_NONBLOCK)
}
