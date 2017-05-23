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
func sigaltstack(new, old *sigaltstackt)

//go:noescape
func sigaction(sig int32, new, old *sigactiont)

//go:noescape
func sigprocmask(how int32, new, old *sigset)

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

//go:noescape
func getrlimit(kind int32, limit unsafe.Pointer) int32

func raise(sig int32)
func raiseproc(sig int32)

//go:noescape
func sys_umtx_sleep(addr *uint32, val, timeout int32) int32

//go:noescape
func sys_umtx_wakeup(addr *uint32, val int32) int32

func osyield()

const stackSystem = 0

// From DragonFly's <sys/sysctl.h>
const (
	_CTL_HW  = 6
	_HW_NCPU = 3
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
// Called on the new thread, cannot allocate memory.
func minit() {
	_g_ := getg()

	// m.procid is a uint64, but lwp_start writes an int32. Fix it up.
	_g_.m.procid = uint64(*(*int32)(unsafe.Pointer(&_g_.m.procid)))

	// Initialize signal handling.

	// On DragonFly a thread created by pthread_create inherits
	// the signal stack of the creating thread. We always create
	// a new signal stack here, to avoid having two Go threads
	// using the same signal stack. This breaks the case of a
	// thread created in C that calls sigaltstack and then calls a
	// Go function, because we will lose track of the C code's
	// sigaltstack, but it's the best we can do.
	signalstack(&_g_.m.gsignal.stack)
	_g_.m.newSigstack = true

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
