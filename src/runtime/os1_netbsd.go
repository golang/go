// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	_ESRCH   = 3
	_ENOTSUP = 91

	// From NetBSD's <sys/time.h>
	_CLOCK_REALTIME  = 0
	_CLOCK_VIRTUAL   = 1
	_CLOCK_PROF      = 2
	_CLOCK_MONOTONIC = 3
)

var sigset_none = sigset{}
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
func semacreate() uintptr {
	return 1
}

//go:nosplit
func semasleep(ns int64) int32 {
	_g_ := getg()

	// spin-mutex lock
	for {
		if xchg(&_g_.m.waitsemalock, 1) == 0 {
			break
		}
		osyield()
	}

	for {
		// lock held
		if _g_.m.waitsemacount == 0 {
			// sleep until semaphore != 0 or timeout.
			// thrsleep unlocks m.waitsemalock.
			if ns < 0 {
				// TODO(jsing) - potential deadlock!
				//
				// There is a potential deadlock here since we
				// have to release the waitsemalock mutex
				// before we call lwp_park() to suspend the
				// thread. This allows another thread to
				// release the lock and call lwp_unpark()
				// before the thread is actually suspended.
				// If this occurs the current thread will end
				// up sleeping indefinitely. Unfortunately
				// the NetBSD kernel does not appear to provide
				// a mechanism for unlocking the userspace
				// mutex once the thread is actually parked.
				atomicstore(&_g_.m.waitsemalock, 0)
				lwp_park(nil, 0, unsafe.Pointer(&_g_.m.waitsemacount), nil)
			} else {
				var ts timespec
				var nsec int32
				ns += nanotime()
				ts.set_sec(timediv(ns, 1000000000, &nsec))
				ts.set_nsec(nsec)
				// TODO(jsing) - potential deadlock!
				// See above for details.
				atomicstore(&_g_.m.waitsemalock, 0)
				lwp_park(&ts, 0, unsafe.Pointer(&_g_.m.waitsemacount), nil)
			}
			// reacquire lock
			for {
				if xchg(&_g_.m.waitsemalock, 1) == 0 {
					break
				}
				osyield()
			}
		}

		// lock held (again)
		if _g_.m.waitsemacount != 0 {
			// semaphore is available.
			_g_.m.waitsemacount--
			// spin-mutex unlock
			atomicstore(&_g_.m.waitsemalock, 0)
			return 0
		}

		// semaphore not available.
		// if there is a timeout, stop now.
		// otherwise keep trying.
		if ns >= 0 {
			break
		}
	}

	// lock held but giving up
	// spin-mutex unlock
	atomicstore(&_g_.m.waitsemalock, 0)
	return -1
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
	// TODO(jsing) - potential deadlock, see semasleep() for details.
	// Confirm that LWP is parked before unparking...
	ret := lwp_unpark(int32(mp.procid), unsafe.Pointer(&mp.waitsemacount))
	if ret != 0 && ret != _ESRCH {
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
	_g_.m.procid = uint64(lwp_self())

	// Initialize signal handling
	signalstack((*byte)(unsafe.Pointer(_g_.m.gsignal.stack.lo)), 32*1024)
	sigprocmask(_SIG_SETMASK, &sigset_none, nil)
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
	sa_mask      sigset
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
	throw("setsigstack")
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
	var st sigaltstackt

	st.ss_sp = uintptr(unsafe.Pointer(p))
	st.ss_size = uintptr(n)
	st.ss_flags = 0
	if p == nil {
		st.ss_flags = _SS_DISABLE
	}
	sigaltstack(&st, nil)
}

func unblocksignals() {
	sigprocmask(_SIG_SETMASK, &sigset_none, nil)
}
