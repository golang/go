// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	ESRCH       = 3
	EAGAIN      = 35
	EWOULDBLOCK = EAGAIN
	ENOTSUP     = 91

	// From OpenBSD's sys/time.h
	CLOCK_REALTIME  = 0
	CLOCK_VIRTUAL   = 1
	CLOCK_PROF      = 2
	CLOCK_MONOTONIC = 3
)

var sigset_none = uint32(0)
var sigset_all = ^sigset_none

// From OpenBSD's <sys/sysctl.h>
const (
	CTL_HW  = 6
	HW_NCPU = 3
)

func getncpu() int32 {
	mib := [2]uint32{CTL_HW, HW_NCPU}
	out := uint32(0)
	nout := unsafe.Sizeof(out)

	// Fetch hw.ncpu via sysctl.
	ret := sysctl(&mib[0], 2, (*byte)(unsafe.Pointer(&out)), &nout, nil, 0)
	if ret >= 0 {
		return int32(out)
	}
	return 1
}

//go:nosplit
func semacreate() uintptr {
	return 1
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
		ts.set_sec(int64(timediv(ns, 1000000000, &nsec)))
		ts.set_nsec(nsec)
		tsp = &ts
	}

	for {
		// spin-mutex lock
		for {
			if xchg(&_g_.m.waitsemalock, 1) == 0 {
				break
			}
			osyield()
		}

		if _g_.m.waitsemacount != 0 {
			// semaphore is available.
			_g_.m.waitsemacount--
			// spin-mutex unlock
			atomicstore(&_g_.m.waitsemalock, 0)
			return 0 // semaphore acquired
		}

		// sleep until semaphore != 0 or timeout.
		// thrsleep unlocks m.waitsemalock.
		ret := thrsleep((uintptr)(unsafe.Pointer(&_g_.m.waitsemacount)), CLOCK_MONOTONIC, tsp, (uintptr)(unsafe.Pointer(&_g_.m.waitsemalock)), (*int32)(unsafe.Pointer(&_g_.m.waitsemacount)))
		if ret == EWOULDBLOCK {
			return -1
		}
	}
}

//go:nosplit
func semawakeup(mp *m) {
	// spin-mutex lock
	for {
		if xchg(&mp.waitsemalock, 1) == 0 {
			break
		}
		osyield()
	}
	mp.waitsemacount++
	ret := thrwakeup(uintptr(unsafe.Pointer(&mp.waitsemacount)), 1)
	if ret != 0 && ret != ESRCH {
		// semawakeup can be called on signal stack.
		systemstack(func() {
			print("thrwakeup addr=", &mp.waitsemacount, " sem=", mp.waitsemacount, " ret=", ret, "\n")
		})
	}
	// spin-mutex unlock
	atomicstore(&mp.waitsemalock, 0)
}

func newosproc(mp *m, stk unsafe.Pointer) {
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " id=", mp.id, "/", int32(mp.tls[0]), " ostk=", &mp, "\n")
	}

	mp.tls[0] = uintptr(mp.id) // so 386 asm can find it

	param := tforkt{
		tf_tcb:   unsafe.Pointer(&mp.tls[0]),
		tf_tid:   (*int32)(unsafe.Pointer(&mp.procid)),
		tf_stack: uintptr(stk),
	}

	oset := sigprocmask(_SIG_SETMASK, sigset_all)
	ret := tfork(&param, unsafe.Sizeof(param), mp, mp.g0, funcPC(mstart))
	sigprocmask(_SIG_SETMASK, oset)

	if ret < 0 {
		print("runtime: failed to create new OS thread (have ", mcount()-1, " already; errno=", -ret, ")\n")
		if ret == -ENOTSUP {
			print("runtime: is kern.rthreads disabled?\n")
		}
		gothrow("runtime.newosproc")
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
	close(fd)
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
// Called on the new thread, can not allocate memory.
func minit() {
	_g_ := getg()

	// m.procid is a uint64, but tfork writes an int32. Fix it up.
	_g_.m.procid = uint64(*(*int32)(unsafe.Pointer(&_g_.m.procid)))

	// Initialize signal handling
	signalstack((*byte)(unsafe.Pointer(_g_.m.gsignal.stack.lo)), 32*1024)
	sigprocmask(_SIG_SETMASK, sigset_none)
}

// Called from dropm to undo the effect of an minit.
func unminit() {
	signalstack(nil, 0)
}

func memlimit() uintptr {
	return 0
}

func sigtramp()

type sigactiont struct {
	sa_sigaction uintptr
	sa_mask      uint32
	sa_flags     int32
}

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

func setsigstack(i int32) {
	gothrow("setsigstack")
}

func getsig(i int32) uintptr {
	var sa sigactiont
	sigaction(i, nil, &sa)
	if sa.sa_sigaction == funcPC(sigtramp) {
		return funcPC(sighandler)
	}
	return sa.sa_sigaction
}

func signalstack(p *byte, n int32) {
	var st stackt

	st.ss_sp = uintptr(unsafe.Pointer(p))
	st.ss_size = uintptr(n)
	st.ss_flags = 0
	if p == nil {
		st.ss_flags = _SS_DISABLE
	}
	sigaltstack(&st, nil)
}

func unblocksignals() {
	sigprocmask(_SIG_SETMASK, sigset_none)
}
