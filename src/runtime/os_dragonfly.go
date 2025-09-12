// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

const (
	_NSIG        = 33
	_SI_USER     = 0
	_SS_DISABLE  = 4
	_SIG_BLOCK   = 1
	_SIG_UNBLOCK = 2
	_SIG_SETMASK = 3
)

type mOS struct {
	waitsema uint32 // semaphore for parking on locks
}

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

func raiseproc(sig uint32)

func lwp_gettid() int32
func lwp_kill(pid, tid int32, sig int)

//go:noescape
func sys_umtx_sleep(addr *uint32, val, timeout int32) int32

//go:noescape
func sys_umtx_wakeup(addr *uint32, val int32) int32

func osyield()

//go:nosplit
func osyield_no_g() {
	osyield()
}

func kqueue() int32

//go:noescape
func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32

func pipe2(flags int32) (r, w int32, errno int32)
func fcntl(fd, cmd, arg int32) (ret int32, errno int32)

func issetugid() int32

// From DragonFly's <sys/sysctl.h>
const (
	_CTL_HW      = 6
	_HW_NCPU     = 3
	_HW_PAGESIZE = 7
)

var sigset_all = sigset{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}

func getCPUCount() int32 {
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
//
//go:nowritebarrier
func newosproc(mp *m) {
	stk := unsafe.Pointer(mp.g0.stack.hi)
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " lwp_start=", abi.FuncPCABI0(lwp_start), " id=", mp.id, " ostk=", &mp, "\n")
	}

	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)

	params := lwpparams{
		start_func: abi.FuncPCABI0(lwp_start),
		arg:        unsafe.Pointer(mp),
		stack:      uintptr(stk),
		tid1:       nil, // minit will record tid
		tid2:       nil,
	}

	// TODO: Check for error.
	retryOnEAGAIN(func() int32 {
		lwp_create(&params)
		return 0
	})
	sigprocmask(_SIG_SETMASK, &oset, nil)
}

func osinit() {
	numCPUStartup = getCPUCount()
	if physPageSize == 0 {
		physPageSize = getPageSize()
	}
}

var urandom_dev = []byte("/dev/urandom\x00")

//go:nosplit
func readRandom(r []byte) int {
	fd := open(&urandom_dev[0], 0 /* O_RDONLY */, 0)
	n := read(fd, unsafe.Pointer(&r[0]), int32(len(r)))
	closefd(fd)
	return int(n)
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
	getg().m.procid = uint64(lwp_gettid())
	minitSignals()
}

// Called from dropm to undo the effect of an minit.
//
//go:nosplit
func unminit() {
	unminitSignals()
	getg().m.procid = 0
}

// Called from mexit, but not from dropm, to undo the effect of thread-owned
// resources in minit, semacreate, or elsewhere. Do not take locks after calling this.
//
// This always runs without a P, so //go:nowritebarrierrec is required.
//
//go:nowritebarrierrec
func mdestroy(mp *m) {
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
	if fn == abi.FuncPCABIInternal(sighandler) { // abi.FuncPCABIInternal(sighandler) matches the callers in signal_unix.go
		fn = abi.FuncPCABI0(sigtramp)
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

// setSignalstackSP sets the ss_sp field of a stackt.
//
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

func setProcessCPUProfiler(hz int32) {
	setProcessCPUProfilerTimer(hz)
}

func setThreadCPUProfiler(hz int32) {
	setThreadCPUProfilerHz(hz)
}

//go:nosplit
func validSIGPROF(mp *m, c *sigctxt) bool {
	return true
}

func sysargs(argc int32, argv **byte) {
	n := argc + 1

	// skip over argv, envp to get to auxv
	for argv_index(argv, n) != nil {
		n++
	}

	// skip NULL separator
	n++

	auxvp := (*[1 << 28]uintptr)(add(unsafe.Pointer(argv), uintptr(n)*goarch.PtrSize))
	pairs := sysauxv(auxvp[:])
	auxv = auxvp[: pairs*2 : pairs*2]
}

const (
	_AT_NULL   = 0
	_AT_PAGESZ = 6
)

func sysauxv(auxv []uintptr) (pairs int) {
	var i int
	for i = 0; auxv[i] != _AT_NULL; i += 2 {
		tag, val := auxv[i], auxv[i+1]
		switch tag {
		case _AT_PAGESZ:
			physPageSize = val
		}
	}
	return i / 2
}

// raise sends a signal to the calling thread.
//
// It must be nosplit because it is used by the signal handler before
// it definitely has a Go stack.
//
//go:nosplit
func raise(sig uint32) {
	lwp_kill(-1, lwp_gettid(), int(sig))
}

func signalM(mp *m, sig int) {
	lwp_kill(-1, int32(mp.procid), sig)
}

// sigPerThreadSyscall is only used on linux, so we assign a bogus signal
// number.
const sigPerThreadSyscall = 1 << 31

//go:nosplit
func runPerThreadSyscall() {
	throw("runPerThreadSyscall only valid on linux")
}
