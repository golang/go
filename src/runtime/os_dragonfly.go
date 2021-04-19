// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	_NSIG        = 33
	_SI_USER     = 0
	_SS_DISABLE  = 4
	_RLIMIT_AS   = 10
	_SIG_BLOCK   = 1
	_SIG_UNBLOCK = 2
	_SIG_SETMASK = 3
)

type mOS struct{}

//go:noescape
func lwp_create(param *lwpparams) int32

//go:noescape
func sigaltstack(new, old *stackt)

//go:noescape
func sigaction(sig uint32, new, old *sigactiont)

//go:noescape
func sigprocmask(how int32, new, old *sigset)

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

//go:noescape
func getrlimit(kind int32, limit unsafe.Pointer) int32

func raise(sig uint32)
func raiseproc(sig uint32)

//go:noescape
func sys_umtx_sleep(addr *uint32, val, timeout int32) int32

//go:noescape
func sys_umtx_wakeup(addr *uint32, val int32) int32

func osyield()

const stackSystem = 0

// From DragonFly's <sys/sysctl.h>
const (
	_CTL_HW      = 6
	_HW_NCPU     = 3
	_HW_PAGESIZE = 7
)

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

// May run with m.p==nil, so write barriers are not allowed.
//go:nowritebarrier
func newosproc(mp *m, stk unsafe.Pointer) {
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " lwp_start=", funcPC(lwp_start), " id=", mp.id, " ostk=", &mp, "\n")
	}

	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)

	params := lwpparams{
		start_func: funcPC(lwp_start),
		arg:        unsafe.Pointer(mp),
		stack:      uintptr(stk),
		tid1:       unsafe.Pointer(&mp.procid),
		tid2:       nil,
	}

	// TODO: Check for error.
	lwp_create(&params)
	sigprocmask(_SIG_SETMASK, &oset, nil)
}

func osinit() {
	ncpu = getncpu()
	physPageSize = getPageSize()
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
	// m.procid is a uint64, but lwp_start writes an int32. Fix it up.
	_g_ := getg()
	_g_.m.procid = uint64(*(*int32)(unsafe.Pointer(&_g_.m.procid)))

	minitSignals()
}

// Called from dropm to undo the effect of an minit.
//go:nosplit
func unminit() {
	unminitSignals()
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
				// not going to be able to do much. Treat as no limit.
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

func (c *sigctxt) fixsigcode(sig uint32) {
}
