// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

type mOS struct{}

//go:noescape
func thr_new(param *thrparam, size int32)

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
func sys_umtx_op(addr *uint32, mode int32, val uint32, uaddr1 uintptr, ut *umtx_time) int32

func osyield()

// From FreeBSD's <sys/sysctl.h>
const (
	_CTL_HW      = 6
	_HW_PAGESIZE = 7
)

var sigset_all = sigset{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}

// Undocumented numbers from FreeBSD's lib/libc/gen/sysctlnametomib.c.
const (
	_CTL_QUERY     = 0
	_CTL_QUERY_MIB = 3
)

// sysctlnametomib fill mib with dynamically assigned sysctl entries of name,
// return count of effected mib slots, return 0 on error.
func sysctlnametomib(name []byte, mib *[_CTL_MAXNAME]uint32) uint32 {
	oid := [2]uint32{_CTL_QUERY, _CTL_QUERY_MIB}
	miblen := uintptr(_CTL_MAXNAME)
	if sysctl(&oid[0], 2, (*byte)(unsafe.Pointer(mib)), &miblen, (*byte)(unsafe.Pointer(&name[0])), (uintptr)(len(name))) < 0 {
		return 0
	}
	miblen /= unsafe.Sizeof(uint32(0))
	if miblen <= 0 {
		return 0
	}
	return uint32(miblen)
}

const (
	_CPU_SETSIZE_MAX = 32 // Limited by _MaxGomaxprocs(256) in runtime2.go.
	_CPU_CURRENT_PID = -1 // Current process ID.
)

//go:noescape
func cpuset_getaffinity(level int, which int, id int64, size int, mask *byte) int32

func getncpu() int32 {
	var mask [_CPU_SETSIZE_MAX]byte
	var mib [_CTL_MAXNAME]uint32

	// According to FreeBSD's /usr/src/sys/kern/kern_cpuset.c,
	// cpuset_getaffinity return ERANGE when provided buffer size exceed the limits in kernel.
	// Querying kern.smp.maxcpus to calculate maximum buffer size.
	// See https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=200802

	// Variable kern.smp.maxcpus introduced at Dec 23 2003, revision 123766,
	// with dynamically assigned sysctl entries.
	miblen := sysctlnametomib([]byte("kern.smp.maxcpus"), &mib)
	if miblen == 0 {
		return 1
	}

	// Query kern.smp.maxcpus.
	dstsize := uintptr(4)
	maxcpus := uint32(0)
	if sysctl(&mib[0], miblen, (*byte)(unsafe.Pointer(&maxcpus)), &dstsize, nil, 0) != 0 {
		return 1
	}

	size := maxcpus / _NBBY
	ptrsize := uint32(unsafe.Sizeof(uintptr(0)))
	if size < ptrsize {
		size = ptrsize
	}
	if size > _CPU_SETSIZE_MAX {
		return 1
	}

	if cpuset_getaffinity(_CPU_LEVEL_WHICH, _CPU_WHICH_PID, _CPU_CURRENT_PID,
		int(size), (*byte)(unsafe.Pointer(&mask[0]))) != 0 {
		return 1
	}
	n := int32(0)
	for _, v := range mask[:size] {
		for v != 0 {
			n += int32(v & 1)
			v >>= 1
		}
	}
	if n == 0 {
		return 1
	}
	return n
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

// FreeBSD's umtx_op syscall is effectively the same as Linux's futex, and
// thus the code is largely similar. See Linux implementation
// and lock_futex.go for comments.

//go:nosplit
func futexsleep(addr *uint32, val uint32, ns int64) {
	systemstack(func() {
		futexsleep1(addr, val, ns)
	})
}

func futexsleep1(addr *uint32, val uint32, ns int64) {
	var utp *umtx_time
	if ns >= 0 {
		var ut umtx_time
		ut._clockid = _CLOCK_MONOTONIC
		ut._timeout.set_sec(int64(timediv(ns, 1000000000, (*int32)(unsafe.Pointer(&ut._timeout.tv_nsec)))))
		utp = &ut
	}
	ret := sys_umtx_op(addr, _UMTX_OP_WAIT_UINT_PRIVATE, val, unsafe.Sizeof(*utp), utp)
	if ret >= 0 || ret == -_EINTR {
		return
	}
	print("umtx_wait addr=", addr, " val=", val, " ret=", ret, "\n")
	*(*int32)(unsafe.Pointer(uintptr(0x1005))) = 0x1005
}

//go:nosplit
func futexwakeup(addr *uint32, cnt uint32) {
	ret := sys_umtx_op(addr, _UMTX_OP_WAKE_PRIVATE, cnt, 0, nil)
	if ret >= 0 {
		return
	}

	systemstack(func() {
		print("umtx_wake_addr=", addr, " ret=", ret, "\n")
	})
}

func thr_start()

// May run with m.p==nil, so write barriers are not allowed.
//go:nowritebarrier
func newosproc(mp *m, stk unsafe.Pointer) {
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " thr_start=", funcPC(thr_start), " id=", mp.id, " ostk=", &mp, "\n")
	}

	// NOTE(rsc): This code is confused. stackbase is the top of the stack
	// and is equal to stk. However, it's working, so I'm not changing it.
	param := thrparam{
		start_func: funcPC(thr_start),
		arg:        unsafe.Pointer(mp),
		stack_base: mp.g0.stack.hi,
		stack_size: uintptr(stk) - mp.g0.stack.hi,
		child_tid:  unsafe.Pointer(&mp.procid),
		parent_tid: nil,
		tls_base:   unsafe.Pointer(&mp.tls[0]),
		tls_size:   unsafe.Sizeof(mp.tls),
	}

	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	// TODO: Check for error.
	thr_new(&param, int32(unsafe.Sizeof(param)))
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
	// m.procid is a uint64, but thr_new writes a uint32 on 32-bit systems.
	// Fix it up. (Only matters on big-endian, but be clean anyway.)
	if sys.PtrSize == 4 {
		_g_ := getg()
		_g_.m.procid = uint64(*(*uint32)(unsafe.Pointer(&_g_.m.procid)))
	}

	// On FreeBSD before about April 2017 there was a bug such
	// that calling execve from a thread other than the main
	// thread did not reset the signal stack. That would confuse
	// minitSignals, which calls minitSignalStack, which checks
	// whether there is currently a signal stack and uses it if
	// present. To avoid this confusion, explicitly disable the
	// signal stack on the main thread when not running in a
	// library. This can be removed when we are confident that all
	// FreeBSD users are running a patched kernel. See issue #15658.
	if gp := getg(); !isarchive && !islibrary && gp.m == &m0 && gp == gp.m.g0 {
		st := stackt{ss_flags: _SS_DISABLE}
		sigaltstack(&st, nil)
	}

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
	sa_handler uintptr
	sa_flags   int32
	sa_mask    sigset
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
	sa.sa_handler = fn
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
	return sa.sa_handler
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
