// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

const (
	_ESRCH     = 3
	_ETIMEDOUT = 60

	// From NetBSD's <sys/time.h>
	_CLOCK_REALTIME  = 0
	_CLOCK_VIRTUAL   = 1
	_CLOCK_PROF      = 2
	_CLOCK_MONOTONIC = 3
)

var sigset_all = sigset{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}

// From NetBSD's <sys/sysctl.h>
const (
	_CTL_HW  = 6
	_HW_NCPU = 3
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

//go:nosplit
func semacreate(mp *m) {
}

//go:nosplit
func semasleep(ns int64) int32 {
	_g_ := getg()

	// Compute sleep deadline.
	var tsp *timespec
	if ns >= 0 {
		var ts timespec
		var nsec int32
		ns += nanotime()
		ts.set_sec(timediv(ns, 1000000000, &nsec))
		ts.set_nsec(nsec)
		tsp = &ts
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
		ret := lwp_park(tsp, 0, unsafe.Pointer(&_g_.m.waitsemacount), nil)
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
func newosproc(mp *m, stk unsafe.Pointer) {
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " id=", mp.id, " ostk=", &mp, "\n")
	}

	var uc ucontextt
	getcontext(unsafe.Pointer(&uc))

	uc.uc_flags = _UC_SIGMASK | _UC_CPU
	uc.uc_link = nil
	uc.uc_sigmask = sigset_all

	lwp_mcontext_init(&uc.uc_mcontext, stk, mp, mp.g0, funcPC(mstart))

	ret := lwp_create(unsafe.Pointer(&uc), 0, unsafe.Pointer(&mp.procid))
	if ret < 0 {
		print("runtime: failed to create new OS thread (have ", mcount()-1, " already; errno=", -ret, ")\n")
		throw("runtime.newosproc")
	}
}

func osinit() {
	ncpu = getncpu()
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

//go:nosplit
func msigsave(mp *m) {
	sigprocmask(_SIG_SETMASK, nil, &mp.sigmask)
}

//go:nosplit
func msigrestore(sigmask sigset) {
	sigprocmask(_SIG_SETMASK, &sigmask, nil)
}

//go:nosplit
func sigblock() {
	sigprocmask(_SIG_SETMASK, &sigset_all, nil)
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
func minit() {
	_g_ := getg()
	_g_.m.procid = uint64(lwp_self())

	// Initialize signal handling
	var st sigaltstackt
	sigaltstack(nil, &st)
	if st.ss_flags&_SS_DISABLE != 0 {
		signalstack(&_g_.m.gsignal.stack)
		_g_.m.newSigstack = true
	} else {
		// Use existing signal stack.
		stsp := uintptr(unsafe.Pointer(st.ss_sp))
		_g_.m.gsignal.stack.lo = stsp
		_g_.m.gsignal.stack.hi = stsp + st.ss_size
		_g_.m.gsignal.stackguard0 = stsp + _StackGuard
		_g_.m.gsignal.stackguard1 = stsp + _StackGuard
		_g_.m.gsignal.stackAlloc = st.ss_size
		_g_.m.newSigstack = false
	}

	// restore signal mask from m.sigmask and unblock essential signals
	nmask := _g_.m.sigmask
	for i := range sigtable {
		if sigtable[i].flags&_SigUnblock != 0 {
			nmask.__bits[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
		}
	}
	sigprocmask(_SIG_SETMASK, &nmask, nil)
}

// Called from dropm to undo the effect of an minit.
//go:nosplit
func unminit() {
	if getg().m.newSigstack {
		signalstack(nil)
	}
}

func memlimit() uintptr {
	return 0
}

func sigtramp()

type sigactiont struct {
	sa_sigaction uintptr
	sa_mask      sigset
	sa_flags     int32
}

//go:nosplit
//go:nowritebarrierrec
func setsig(i int32, fn uintptr, restart bool) {
	var sa sigactiont
	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK
	if restart {
		sa.sa_flags |= _SA_RESTART
	}
	sa.sa_mask = sigset_all
	if fn == funcPC(sighandler) {
		fn = funcPC(sigtramp)
	}
	sa.sa_sigaction = fn
	sigaction(i, &sa, nil)
}

//go:nosplit
//go:nowritebarrierrec
func setsigstack(i int32) {
	throw("setsigstack")
}

//go:nosplit
//go:nowritebarrierrec
func getsig(i int32) uintptr {
	var sa sigactiont
	sigaction(i, nil, &sa)
	if sa.sa_sigaction == funcPC(sigtramp) {
		return funcPC(sighandler)
	}
	return sa.sa_sigaction
}

//go:nosplit
func signalstack(s *stack) {
	var st sigaltstackt
	if s == nil {
		st.ss_flags = _SS_DISABLE
	} else {
		st.ss_sp = s.lo
		st.ss_size = s.hi - s.lo
		st.ss_flags = 0
	}
	sigaltstack(&st, nil)
}

//go:nosplit
//go:nowritebarrierrec
func updatesigmask(m sigmask) {
	var mask sigset
	copy(mask.__bits[:], m[:])
	sigprocmask(_SIG_SETMASK, &mask, nil)
}

func unblocksig(sig int32) {
	var mask sigset
	mask.__bits[(sig-1)/32] |= 1 << ((uint32(sig) - 1) & 31)
	sigprocmask(_SIG_UNBLOCK, &mask, nil)
}
