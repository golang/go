// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

const (
	_SS_DISABLE  = 4
	_SIG_BLOCK   = 1
	_SIG_UNBLOCK = 2
	_SIG_SETMASK = 3
	_NSIG        = 33
	_SI_USER     = 0

	// From NetBSD's <sys/ucontext.h>
	_UC_SIGMASK = 0x01
	_UC_CPU     = 0x04

	// From <sys/lwp.h>
	_LWP_DETACHED = 0x00000040
)

type mOS struct {
	waitsemacount uint32
}

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func sigaction(sig uint32, new, old *sigactiont)

//go:noescape
func sigaltstack(new, old *stackt)

//go:noescape
func sigprocmask(how int32, new, old *sigset)

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

func lwp_tramp()

func raiseproc(sig uint32)

func lwp_kill(tid int32, sig int)

//go:noescape
func getcontext(ctxt unsafe.Pointer)

//go:noescape
func lwp_create(ctxt unsafe.Pointer, flags uintptr, lwpid unsafe.Pointer) int32

//go:noescape
func lwp_park(clockid, flags int32, ts *timespec, unpark int32, hint, unparkhint unsafe.Pointer) int32

//go:noescape
func lwp_unpark(lwp int32, hint unsafe.Pointer) int32

func lwp_self() int32

func osyield()

func kqueue() int32

//go:noescape
func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32

func pipe() (r, w int32, errno int32)
func pipe2(flags int32) (r, w int32, errno int32)
func closeonexec(fd int32)
func setNonblock(fd int32)

const (
	_ESRCH     = 3
	_ETIMEDOUT = 60

	// From NetBSD's <sys/time.h>
	_CLOCK_REALTIME  = 0
	_CLOCK_VIRTUAL   = 1
	_CLOCK_PROF      = 2
	_CLOCK_MONOTONIC = 3

	_TIMER_RELTIME = 0
	_TIMER_ABSTIME = 1
)

var sigset_all = sigset{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}

// From NetBSD's <sys/sysctl.h>
const (
	_CTL_HW      = 6
	_HW_NCPU     = 3
	_HW_PAGESIZE = 7
)

func getncpu() int32 {
	mib := [2]uint32{_CTL_HW, _HW_NCPU}
	out := uint32(0)
	nout := unsafe.Sizeof(out)
	ret := sysctl(&mib[0], 2, (*byte)(unsafe.Pointer(&out)), &nout, nil, 0)
	if ret >= 0 {
		return int32(out)
	}
	return 1
}

func getPageSize() uintptr {
	mib := [2]uint32{_CTL_HW, _HW_PAGESIZE}
	out := uint32(0)
	nout := unsafe.Sizeof(out)
	ret := sysctl(&mib[0], 2, (*byte)(unsafe.Pointer(&out)), &nout, nil, 0)
	if ret >= 0 {
		return uintptr(out)
	}
	return 0
}

//go:nosplit
func semacreate(mp *m) {
}

//go:nosplit
func semasleep(ns int64) int32 {
	_g_ := getg()
	var deadline int64
	if ns >= 0 {
		deadline = nanotime() + ns
	}

	for {
		v := atomic.Load(&_g_.m.waitsemacount)
		if v > 0 {
			if atomic.Cas(&_g_.m.waitsemacount, v, v-1) {
				return 0 // semaphore acquired
			}
			continue
		}

		// Sleep until unparked by semawakeup or timeout.
		var tsp *timespec
		var ts timespec
		if ns >= 0 {
			wait := deadline - nanotime()
			if wait <= 0 {
				return -1
			}
			ts.setNsec(wait)
			tsp = &ts
		}
		ret := lwp_park(_CLOCK_MONOTONIC, _TIMER_RELTIME, tsp, 0, unsafe.Pointer(&_g_.m.waitsemacount), nil)
		if ret == _ETIMEDOUT {
			return -1
		}
	}
}

//go:nosplit
func semawakeup(mp *m) {
	atomic.Xadd(&mp.waitsemacount, 1)
	// From NetBSD's _lwp_unpark(2) manual:
	// "If the target LWP is not currently waiting, it will return
	// immediately upon the next call to _lwp_park()."
	ret := lwp_unpark(int32(mp.procid), unsafe.Pointer(&mp.waitsemacount))
	if ret != 0 && ret != _ESRCH {
		// semawakeup can be called on signal stack.
		systemstack(func() {
			print("thrwakeup addr=", &mp.waitsemacount, " sem=", mp.waitsemacount, " ret=", ret, "\n")
		})
	}
}

// May run with m.p==nil, so write barriers are not allowed.
//go:nowritebarrier
func newosproc(mp *m) {
	stk := unsafe.Pointer(mp.g0.stack.hi)
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " id=", mp.id, " ostk=", &mp, "\n")
	}

	var uc ucontextt
	getcontext(unsafe.Pointer(&uc))

	// _UC_SIGMASK does not seem to work here.
	// It would be nice if _UC_SIGMASK and _UC_STACK
	// worked so that we could do all the work setting
	// the sigmask and the stack here, instead of setting
	// the mask here and the stack in netbsdMstart.
	// For now do the blocking manually.
	uc.uc_flags = _UC_SIGMASK | _UC_CPU
	uc.uc_link = nil
	uc.uc_sigmask = sigset_all

	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)

	lwp_mcontext_init(&uc.uc_mcontext, stk, mp, mp.g0, funcPC(netbsdMstart))

	ret := lwp_create(unsafe.Pointer(&uc), _LWP_DETACHED, unsafe.Pointer(&mp.procid))
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if ret < 0 {
		print("runtime: failed to create new OS thread (have ", mcount()-1, " already; errno=", -ret, ")\n")
		if ret == -_EAGAIN {
			println("runtime: may need to increase max user processes (ulimit -p)")
		}
		throw("runtime.newosproc")
	}
}

// netbsdMStart is the function call that starts executing a newly
// created thread. On NetBSD, a new thread inherits the signal stack
// of the creating thread. That confuses minit, so we remove that
// signal stack here before calling the regular mstart. It's a bit
// baroque to remove a signal stack here only to add one in minit, but
// it's a simple change that keeps NetBSD working like other OS's.
// At this point all signals are blocked, so there is no race.
//go:nosplit
func netbsdMstart() {
	st := stackt{ss_flags: _SS_DISABLE}
	sigaltstack(&st, nil)
	mstart()
}

func osinit() {
	ncpu = getncpu()
	if physPageSize == 0 {
		physPageSize = getPageSize()
	}
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

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024)
	mp.gsignal.m = mp
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
	_g_ := getg()
	_g_.m.procid = uint64(lwp_self())

	// On NetBSD a thread created by pthread_create inherits the
	// signal stack of the creating thread. We always create a
	// new signal stack here, to avoid having two Go threads using
	// the same signal stack. This breaks the case of a thread
	// created in C that calls sigaltstack and then calls a Go
	// function, because we will lose track of the C code's
	// sigaltstack, but it's the best we can do.
	signalstack(&_g_.m.gsignal.stack)
	_g_.m.newSigstack = true

	minitSignalMask()
}

// Called from dropm to undo the effect of an minit.
//go:nosplit
func unminit() {
	unminitSignals()
}

func sigtramp()

type sigactiont struct {
	sa_sigaction uintptr
	sa_mask      sigset
	sa_flags     int32
}

//go:nosplit
//go:nowritebarrierrec
func setsig(i uint32, fn uintptr) {
	var sa sigactiont
	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK | _SA_RESTART
	sa.sa_mask = sigset_all
	if fn == funcPC(sighandler) {
		fn = funcPC(sigtramp)
	}
	sa.sa_sigaction = fn
	sigaction(i, &sa, nil)
}

//go:nosplit
//go:nowritebarrierrec
func setsigstack(i uint32) {
	throw("setsigstack")
}

//go:nosplit
//go:nowritebarrierrec
func getsig(i uint32) uintptr {
	var sa sigactiont
	sigaction(i, nil, &sa)
	return sa.sa_sigaction
}

// setSignaltstackSP sets the ss_sp field of a stackt.
//go:nosplit
func setSignalstackSP(s *stackt, sp uintptr) {
	s.ss_sp = sp
}

//go:nosplit
//go:nowritebarrierrec
func sigaddset(mask *sigset, i int) {
	mask.__bits[(i-1)/32] |= 1 << ((uint32(i) - 1) & 31)
}

func sigdelset(mask *sigset, i int) {
	mask.__bits[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
}

//go:nosplit
func (c *sigctxt) fixsigcode(sig uint32) {
}

func sysargs(argc int32, argv **byte) {
	n := argc + 1

	// skip over argv, envp to get to auxv
	for argv_index(argv, n) != nil {
		n++
	}

	// skip NULL separator
	n++

	// now argv+n is auxv
	auxv := (*[1 << 28]uintptr)(add(unsafe.Pointer(argv), uintptr(n)*sys.PtrSize))
	sysauxv(auxv[:])
}

const (
	_AT_NULL   = 0 // Terminates the vector
	_AT_PAGESZ = 6 // Page size in bytes
)

func sysauxv(auxv []uintptr) {
	for i := 0; auxv[i] != _AT_NULL; i += 2 {
		tag, val := auxv[i], auxv[i+1]
		switch tag {
		case _AT_PAGESZ:
			physPageSize = val
		}
	}
}

// raise sends signal to the calling thread.
//
// It must be nosplit because it is used by the signal handler before
// it definitely has a Go stack.
//
//go:nosplit
func raise(sig uint32) {
	lwp_kill(lwp_self(), int(sig))
}

func signalM(mp *m, sig int) {
	lwp_kill(int32(mp.procid), sig)
}
