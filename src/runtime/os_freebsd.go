// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

type mOS struct {
	waitsema uint32 // semaphore for parking on locks
}

//go:noescape
func thr_new(param *thrparam, size int32) int32

//go:noescape
func sigaltstack(new, old *stackt)

//go:noescape
func sigprocmask(how int32, new, old *sigset)

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

func raiseproc(sig uint32)

func thr_self() thread
func thr_kill(tid thread, sig int)

//go:noescape
func sys_umtx_op(addr *uint32, mode int32, val uint32, uaddr1 uintptr, ut *umtx_time) int32

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
	_CPU_CURRENT_PID = -1 // Current process ID.
)

//go:noescape
func cpuset_getaffinity(level int, which int, id int64, size int, mask *byte) int32

//go:systemstack
func getncpu() int32 {
	// Use a large buffer for the CPU mask. We're on the system
	// stack, so this is fine, and we can't allocate memory for a
	// dynamically-sized buffer at this point.
	const maxCPUs = 64 * 1024
	var mask [maxCPUs / 8]byte
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

	maskSize := int(maxcpus+7) / 8
	if maskSize < goarch.PtrSize {
		maskSize = goarch.PtrSize
	}
	if maskSize > len(mask) {
		maskSize = len(mask)
	}

	if cpuset_getaffinity(_CPU_LEVEL_WHICH, _CPU_WHICH_PID, _CPU_CURRENT_PID,
		maskSize, (*byte)(unsafe.Pointer(&mask[0]))) != 0 {
		return 1
	}
	n := int32(0)
	for _, v := range mask[:maskSize] {
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
		ut._timeout.setNsec(ns)
		utp = &ut
	}
	ret := sys_umtx_op(addr, _UMTX_OP_WAIT_UINT_PRIVATE, val, unsafe.Sizeof(*utp), utp)
	if ret >= 0 || ret == -_EINTR || ret == -_ETIMEDOUT {
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
//
//go:nowritebarrier
func newosproc(mp *m) {
	stk := unsafe.Pointer(mp.g0.stack.hi)
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " thr_start=", abi.FuncPCABI0(thr_start), " id=", mp.id, " ostk=", &mp, "\n")
	}

	param := thrparam{
		start_func: abi.FuncPCABI0(thr_start),
		arg:        unsafe.Pointer(mp),
		stack_base: mp.g0.stack.lo,
		stack_size: uintptr(stk) - mp.g0.stack.lo,
		child_tid:  nil, // minit will record tid
		parent_tid: nil,
		tls_base:   unsafe.Pointer(&mp.tls[0]),
		tls_size:   unsafe.Sizeof(mp.tls),
	}

	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	ret := retryOnEAGAIN(func() int32 {
		errno := thr_new(&param, int32(unsafe.Sizeof(param)))
		// thr_new returns negative errno
		return -errno
	})
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if ret != 0 {
		print("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", ret, ")\n")
		throw("newosproc")
	}
}

// Version of newosproc that doesn't require a valid G.
//
//go:nosplit
func newosproc0(stacksize uintptr, fn unsafe.Pointer) {
	stack := sysAlloc(stacksize, &memstats.stacks_sys)
	if stack == nil {
		writeErrStr(failallocatestack)
		exit(1)
	}
	// This code "knows" it's being called once from the library
	// initialization code, and so it's using the static m0 for the
	// tls and procid (thread) pointers. thr_new() requires the tls
	// pointers, though the tid pointers can be nil.
	// However, newosproc0 is currently unreachable because builds
	// utilizing c-shared/c-archive force external linking.
	param := thrparam{
		start_func: uintptr(fn),
		arg:        nil,
		stack_base: uintptr(stack), //+stacksize?
		stack_size: stacksize,
		child_tid:  nil, // minit will record tid
		parent_tid: nil,
		tls_base:   unsafe.Pointer(&m0.tls[0]),
		tls_size:   unsafe.Sizeof(m0.tls),
	}

	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	ret := thr_new(&param, int32(unsafe.Sizeof(param)))
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if ret < 0 {
		writeErrStr(failthreadcreate)
		exit(1)
	}
}

// Called to do synchronous initialization of Go code built with
// -buildmode=c-archive or -buildmode=c-shared.
// None of the Go runtime is initialized.
//
//go:nosplit
//go:nowritebarrierrec
func libpreinit() {
	initsig(true)
}

func osinit() {
	ncpu = getncpu()
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
	getg().m.procid = uint64(thr_self())

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
//
//go:nosplit
func unminit() {
	unminitSignals()
	getg().m.procid = 0
}

// Called from exitm, but not from drop, to undo the effect of thread-owned
// resources in minit, semacreate, or elsewhere. Do not take locks after calling this.
func mdestroy(mp *m) {
}

func sigtramp()

type sigactiont struct {
	sa_handler uintptr
	sa_flags   int32
	sa_mask    sigset
}

// See os_freebsd2.go, os_freebsd_amd64.go for setsig function

//go:nosplit
//go:nowritebarrierrec
func setsigstack(i uint32) {
	var sa sigactiont
	sigaction(i, nil, &sa)
	if sa.sa_flags&_SA_ONSTACK != 0 {
		return
	}
	sa.sa_flags |= _SA_ONSTACK
	sigaction(i, &sa, nil)
}

//go:nosplit
//go:nowritebarrierrec
func getsig(i uint32) uintptr {
	var sa sigactiont
	sigaction(i, nil, &sa)
	return sa.sa_handler
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

	// now argv+n is auxv
	auxvp := (*[1 << 28]uintptr)(add(unsafe.Pointer(argv), uintptr(n)*goarch.PtrSize))
	pairs := sysauxv(auxvp[:])
	auxv = auxvp[: pairs*2 : pairs*2]
}

const (
	_AT_NULL     = 0  // Terminates the vector
	_AT_PAGESZ   = 6  // Page size in bytes
	_AT_PLATFORM = 15 // string identifying platform
	_AT_TIMEKEEP = 22 // Pointer to timehands.
	_AT_HWCAP    = 25 // CPU feature flags
	_AT_HWCAP2   = 26 // CPU feature flags 2
)

func sysauxv(auxv []uintptr) (pairs int) {
	var i int
	for i = 0; auxv[i] != _AT_NULL; i += 2 {
		tag, val := auxv[i], auxv[i+1]
		switch tag {
		// _AT_NCPUS from auxv shouldn't be used due to golang.org/issue/15206
		case _AT_PAGESZ:
			physPageSize = val
		case _AT_TIMEKEEP:
			timekeepSharedPage = (*vdsoTimekeep)(unsafe.Pointer(val))
		}

		archauxv(tag, val)
	}
	return i / 2
}

// sysSigaction calls the sigaction system call.
//
//go:nosplit
func sysSigaction(sig uint32, new, old *sigactiont) {
	// Use system stack to avoid split stack overflow on amd64
	if asmSigaction(uintptr(sig), new, old) != 0 {
		systemstack(func() {
			throw("sigaction failed")
		})
	}
}

// asmSigaction is implemented in assembly.
//
//go:noescape
func asmSigaction(sig uintptr, new, old *sigactiont) int32

// raise sends a signal to the calling thread.
//
// It must be nosplit because it is used by the signal handler before
// it definitely has a Go stack.
//
//go:nosplit
func raise(sig uint32) {
	thr_kill(thr_self(), int(sig))
}

func signalM(mp *m, sig int) {
	thr_kill(thread(mp.procid), sig)
}

// sigPerThreadSyscall is only used on linux, so we assign a bogus signal
// number.
const sigPerThreadSyscall = 1 << 31

//go:nosplit
func runPerThreadSyscall() {
	throw("runPerThreadSyscall only valid on linux")
}
