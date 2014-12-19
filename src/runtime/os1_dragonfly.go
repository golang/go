// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// From DragonFly's <sys/sysctl.h>
const (
	_CTL_HW  = 6
	_HW_NCPU = 3
)

var sigset_none = sigset{}
var sigset_all = sigset{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}

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
func futexsleep(addr *uint32, val uint32, ns int64) {
	systemstack(func() {
		futexsleep1(addr, val, ns)
	})
}

func futexsleep1(addr *uint32, val uint32, ns int64) {
	var timeout int32
	if ns >= 0 {
		// The timeout is specified in microseconds - ensure that we
		// do not end up dividing to zero, which would put us to sleep
		// indefinitely...
		timeout = timediv(ns, 1000, nil)
		if timeout == 0 {
			timeout = 1
		}
	}

	// sys_umtx_sleep will return EWOULDBLOCK (EAGAIN) when the timeout
	// expires or EBUSY if the mutex value does not match.
	ret := sys_umtx_sleep(addr, int32(val), timeout)
	if ret >= 0 || ret == -_EINTR || ret == -_EAGAIN || ret == -_EBUSY {
		return
	}

	print("umtx_sleep addr=", addr, " val=", val, " ret=", ret, "\n")
	*(*int32)(unsafe.Pointer(uintptr(0x1005))) = 0x1005
}

//go:nosplit
func futexwakeup(addr *uint32, cnt uint32) {
	ret := sys_umtx_wakeup(addr, int32(cnt))
	if ret >= 0 {
		return
	}

	systemstack(func() {
		print("umtx_wake_addr=", addr, " ret=", ret, "\n")
		*(*int32)(unsafe.Pointer(uintptr(0x1006))) = 0x1006
	})
}

func lwp_start(uintptr)

func newosproc(mp *m, stk unsafe.Pointer) {
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " lwp_start=", funcPC(lwp_start), " id=", mp.id, "/", mp.tls[0], " ostk=", &mp, "\n")
	}

	var oset sigset
	sigprocmask(&sigset_all, &oset)

	params := lwpparams{
		start_func: funcPC(lwp_start),
		arg:        unsafe.Pointer(mp),
		stack:      uintptr(stk),
		tid1:       unsafe.Pointer(&mp.procid),
		tid2:       nil,
	}

	mp.tls[0] = uintptr(mp.id) // so 386 asm can find it

	lwp_create(&params)
	sigprocmask(&oset, nil)
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

	// m.procid is a uint64, but lwp_start writes an int32. Fix it up.
	_g_.m.procid = uint64(*(*int32)(unsafe.Pointer(&_g_.m.procid)))

	// Initialize signal handling
	signalstack((*byte)(unsafe.Pointer(_g_.m.gsignal.stack.lo)), 32*1024)
	sigprocmask(&sigset_none, nil)
}

// Called from dropm to undo the effect of an minit.
func unminit() {
	signalstack(nil, 0)
}

func memlimit() uintptr {
	/*
		                TODO: Convert to Go when something actually uses the result.

				Rlimit rl;
				extern byte runtime·text[], runtime·end[];
				uintptr used;

				if(runtime·getrlimit(RLIMIT_AS, &rl) != 0)
					return 0;
				if(rl.rlim_cur >= 0x7fffffff)
					return 0;

				// Estimate our VM footprint excluding the heap.
				// Not an exact science: use size of binary plus
				// some room for thread stacks.
				used = runtime·end - runtime·text + (64<<20);
				if(used >= rl.rlim_cur)
					return 0;

				// If there's not at least 16 MB left, we're probably
				// not going to be able to do much.  Treat as no limit.
				rl.rlim_cur -= used;
				if(rl.rlim_cur < (16<<20))
					return 0;

				return rl.rlim_cur - used;
	*/
	return 0
}

func sigtramp()

type sigactiont struct {
	sa_sigaction uintptr
	sa_flags     int32
	sa_mask      sigset
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
	sigprocmask(&sigset_none, nil)
}
