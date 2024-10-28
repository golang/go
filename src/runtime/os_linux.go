// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/runtime/atomic"
	"internal/runtime/syscall"
	"unsafe"
)

// sigPerThreadSyscall is the same signal (SIGSETXID) used by glibc for
// per-thread syscalls on Linux. We use it for the same purpose in non-cgo
// binaries.
const sigPerThreadSyscall = _SIGRTMIN + 1

type mOS struct {
	// profileTimer holds the ID of the POSIX interval timer for profiling CPU
	// usage on this thread.
	//
	// It is valid when the profileTimerValid field is true. A thread
	// creates and manages its own timer, and these fields are read and written
	// only by this thread. But because some of the reads on profileTimerValid
	// are in signal handling code, this field should be atomic type.
	profileTimer      int32
	profileTimerValid atomic.Bool

	// needPerThreadSyscall indicates that a per-thread syscall is required
	// for doAllThreadsSyscall.
	needPerThreadSyscall atomic.Uint8

	// This is a pointer to a chunk of memory allocated with a special
	// mmap invocation in vgetrandomGetState().
	vgetrandomState uintptr

	waitsema uint32 // semaphore for parking on locks
}

//go:noescape
func futex(addr unsafe.Pointer, op int32, val uint32, ts, addr2 unsafe.Pointer, val3 uint32) int32

// Linux futex.
//
//	futexsleep(uint32 *addr, uint32 val)
//	futexwakeup(uint32 *addr)
//
// Futexsleep atomically checks if *addr == val and if so, sleeps on addr.
// Futexwakeup wakes up threads sleeping on addr.
// Futexsleep is allowed to wake up spuriously.

const (
	_FUTEX_PRIVATE_FLAG = 128
	_FUTEX_WAIT_PRIVATE = 0 | _FUTEX_PRIVATE_FLAG
	_FUTEX_WAKE_PRIVATE = 1 | _FUTEX_PRIVATE_FLAG
)

// Atomically,
//
//	if(*addr == val) sleep
//
// Might be woken up spuriously; that's allowed.
// Don't sleep longer than ns; ns < 0 means forever.
//
//go:nosplit
func futexsleep(addr *uint32, val uint32, ns int64) {
	// Some Linux kernels have a bug where futex of
	// FUTEX_WAIT returns an internal error code
	// as an errno. Libpthread ignores the return value
	// here, and so can we: as it says a few lines up,
	// spurious wakeups are allowed.
	if ns < 0 {
		futex(unsafe.Pointer(addr), _FUTEX_WAIT_PRIVATE, val, nil, nil, 0)
		return
	}

	var ts timespec
	ts.setNsec(ns)
	futex(unsafe.Pointer(addr), _FUTEX_WAIT_PRIVATE, val, unsafe.Pointer(&ts), nil, 0)
}

// If any procs are sleeping on addr, wake up at most cnt.
//
//go:nosplit
func futexwakeup(addr *uint32, cnt uint32) {
	ret := futex(unsafe.Pointer(addr), _FUTEX_WAKE_PRIVATE, cnt, nil, nil, 0)
	if ret >= 0 {
		return
	}

	// I don't know that futex wakeup can return
	// EAGAIN or EINTR, but if it does, it would be
	// safe to loop and call futex again.
	systemstack(func() {
		print("futexwakeup addr=", addr, " returned ", ret, "\n")
	})

	*(*int32)(unsafe.Pointer(uintptr(0x1006))) = 0x1006
}

func getproccount() int32 {
	// This buffer is huge (8 kB) but we are on the system stack
	// and there should be plenty of space (64 kB).
	// Also this is a leaf, so we're not holding up the memory for long.
	// See golang.org/issue/11823.
	// The suggested behavior here is to keep trying with ever-larger
	// buffers, but we don't have a dynamic memory allocator at the
	// moment, so that's a bit tricky and seems like overkill.
	const maxCPUs = 64 * 1024
	var buf [maxCPUs / 8]byte
	r := sched_getaffinity(0, unsafe.Sizeof(buf), &buf[0])
	if r < 0 {
		return 1
	}
	n := int32(0)
	for _, v := range buf[:r] {
		for v != 0 {
			n += int32(v & 1)
			v >>= 1
		}
	}
	if n == 0 {
		n = 1
	}
	return n
}

// Clone, the Linux rfork.
const (
	_CLONE_VM             = 0x100
	_CLONE_FS             = 0x200
	_CLONE_FILES          = 0x400
	_CLONE_SIGHAND        = 0x800
	_CLONE_PTRACE         = 0x2000
	_CLONE_VFORK          = 0x4000
	_CLONE_PARENT         = 0x8000
	_CLONE_THREAD         = 0x10000
	_CLONE_NEWNS          = 0x20000
	_CLONE_SYSVSEM        = 0x40000
	_CLONE_SETTLS         = 0x80000
	_CLONE_PARENT_SETTID  = 0x100000
	_CLONE_CHILD_CLEARTID = 0x200000
	_CLONE_UNTRACED       = 0x800000
	_CLONE_CHILD_SETTID   = 0x1000000
	_CLONE_STOPPED        = 0x2000000
	_CLONE_NEWUTS         = 0x4000000
	_CLONE_NEWIPC         = 0x8000000

	// As of QEMU 2.8.0 (5ea2fc84d), user emulation requires all six of these
	// flags to be set when creating a thread; attempts to share the other
	// five but leave SYSVSEM unshared will fail with -EINVAL.
	//
	// In non-QEMU environments CLONE_SYSVSEM is inconsequential as we do not
	// use System V semaphores.

	cloneFlags = _CLONE_VM | /* share memory */
		_CLONE_FS | /* share cwd, etc */
		_CLONE_FILES | /* share fd table */
		_CLONE_SIGHAND | /* share sig handler table */
		_CLONE_SYSVSEM | /* share SysV semaphore undo lists (see issue #20763) */
		_CLONE_THREAD /* revisit - okay for now */
)

//go:noescape
func clone(flags int32, stk, mp, gp, fn unsafe.Pointer) int32

// May run with m.p==nil, so write barriers are not allowed.
//
//go:nowritebarrier
func newosproc(mp *m) {
	stk := unsafe.Pointer(mp.g0.stack.hi)
	/*
	 * note: strace gets confused if we use CLONE_PTRACE here.
	 */
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " clone=", abi.FuncPCABI0(clone), " id=", mp.id, " ostk=", &mp, "\n")
	}

	// Disable signals during clone, so that the new thread starts
	// with signals disabled. It will enable them in minit.
	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	ret := retryOnEAGAIN(func() int32 {
		r := clone(cloneFlags, stk, unsafe.Pointer(mp), unsafe.Pointer(mp.g0), unsafe.Pointer(abi.FuncPCABI0(mstart)))
		// clone returns positive TID, negative errno.
		// We don't care about the TID.
		if r >= 0 {
			return 0
		}
		return -r
	})
	sigprocmask(_SIG_SETMASK, &oset, nil)

	if ret != 0 {
		print("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", ret, ")\n")
		if ret == _EAGAIN {
			println("runtime: may need to increase max user processes (ulimit -u)")
		}
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
	ret := clone(cloneFlags, unsafe.Pointer(uintptr(stack)+stacksize), nil, nil, fn)
	if ret < 0 {
		writeErrStr(failthreadcreate)
		exit(1)
	}
}

const (
	_AT_NULL     = 0  // End of vector
	_AT_PAGESZ   = 6  // System physical page size
	_AT_PLATFORM = 15 // string identifying platform
	_AT_HWCAP    = 16 // hardware capability bit vector
	_AT_SECURE   = 23 // secure mode boolean
	_AT_RANDOM   = 25 // introduced in 2.6.29
	_AT_HWCAP2   = 26 // hardware capability bit vector 2
)

var procAuxv = []byte("/proc/self/auxv\x00")

var addrspace_vec [1]byte

func mincore(addr unsafe.Pointer, n uintptr, dst *byte) int32

var auxvreadbuf [128]uintptr

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

	if pairs := sysauxv(auxvp[:]); pairs != 0 {
		auxv = auxvp[: pairs*2 : pairs*2]
		return
	}
	// In some situations we don't get a loader-provided
	// auxv, such as when loaded as a library on Android.
	// Fall back to /proc/self/auxv.
	fd := open(&procAuxv[0], 0 /* O_RDONLY */, 0)
	if fd < 0 {
		// On Android, /proc/self/auxv might be unreadable (issue 9229), so we fallback to
		// try using mincore to detect the physical page size.
		// mincore should return EINVAL when address is not a multiple of system page size.
		const size = 256 << 10 // size of memory region to allocate
		p, err := mmap(nil, size, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
		if err != 0 {
			return
		}
		var n uintptr
		for n = 4 << 10; n < size; n <<= 1 {
			err := mincore(unsafe.Pointer(uintptr(p)+n), 1, &addrspace_vec[0])
			if err == 0 {
				physPageSize = n
				break
			}
		}
		if physPageSize == 0 {
			physPageSize = size
		}
		munmap(p, size)
		return
	}

	n = read(fd, noescape(unsafe.Pointer(&auxvreadbuf[0])), int32(unsafe.Sizeof(auxvreadbuf)))
	closefd(fd)
	if n < 0 {
		return
	}
	// Make sure buf is terminated, even if we didn't read
	// the whole file.
	auxvreadbuf[len(auxvreadbuf)-2] = _AT_NULL
	pairs := sysauxv(auxvreadbuf[:])
	auxv = auxvreadbuf[: pairs*2 : pairs*2]
}

// secureMode holds the value of AT_SECURE passed in the auxiliary vector.
var secureMode bool

func sysauxv(auxv []uintptr) (pairs int) {
	// Process the auxiliary vector entries provided by the kernel when the
	// program is executed. See getauxval(3).
	var i int
	for ; auxv[i] != _AT_NULL; i += 2 {
		tag, val := auxv[i], auxv[i+1]
		switch tag {
		case _AT_RANDOM:
			// The kernel provides a pointer to 16 bytes of cryptographically
			// random data. Note that in cgo programs this value may have
			// already been used by libc at this point, and in particular glibc
			// and musl use the value as-is for stack and pointer protector
			// cookies from libc_start_main and/or dl_start. Also, cgo programs
			// may use the value after we do.
			startupRand = (*[16]byte)(unsafe.Pointer(val))[:]

		case _AT_PAGESZ:
			physPageSize = val

		case _AT_SECURE:
			secureMode = val == 1
		}

		archauxv(tag, val)
		vdsoauxv(tag, val)
	}
	return i / 2
}

var sysTHPSizePath = []byte("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size\x00")

func getHugePageSize() uintptr {
	var numbuf [20]byte
	fd := open(&sysTHPSizePath[0], 0 /* O_RDONLY */, 0)
	if fd < 0 {
		return 0
	}
	ptr := noescape(unsafe.Pointer(&numbuf[0]))
	n := read(fd, ptr, int32(len(numbuf)))
	closefd(fd)
	if n <= 0 {
		return 0
	}
	n-- // remove trailing newline
	v, ok := atoi(slicebytetostringtmp((*byte)(ptr), int(n)))
	if !ok || v < 0 {
		v = 0
	}
	if v&(v-1) != 0 {
		// v is not a power of 2
		return 0
	}
	return uintptr(v)
}

func osinit() {
	ncpu = getproccount()
	physHugePageSize = getHugePageSize()
	osArchInit()
	vgetrandomInit()
}

var urandom_dev = []byte("/dev/urandom\x00")

func readRandom(r []byte) int {
	// Note that all supported Linux kernels should provide AT_RANDOM which
	// populates startupRand, so this fallback should be unreachable.
	fd := open(&urandom_dev[0], 0 /* O_RDONLY */, 0)
	n := read(fd, unsafe.Pointer(&r[0]), int32(len(r)))
	closefd(fd)
	return int(n)
}

func goenvs() {
	goenvs_unix()
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

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024) // Linux wants >= 2K
	mp.gsignal.m = mp
}

func gettid() uint32

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
	minitSignals()

	// Cgo-created threads and the bootstrap m are missing a
	// procid. We need this for asynchronous preemption and it's
	// useful in debuggers.
	getg().m.procid = uint64(gettid())
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
	if mp.vgetrandomState != 0 {
		vgetrandomPutState(mp.vgetrandomState)
		mp.vgetrandomState = 0
	}
}

// #ifdef GOARCH_386
// #define sa_handler k_sa_handler
// #endif

func sigreturn__sigaction()
func sigtramp() // Called via C ABI
func cgoSigtramp()

//go:noescape
func sigaltstack(new, old *stackt)

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func timer_create(clockid int32, sevp *sigevent, timerid *int32) int32

//go:noescape
func timer_settime(timerid int32, flags int32, new, old *itimerspec) int32

//go:noescape
func timer_delete(timerid int32) int32

//go:noescape
func rtsigprocmask(how int32, new, old *sigset, size int32)

//go:nosplit
//go:nowritebarrierrec
func sigprocmask(how int32, new, old *sigset) {
	rtsigprocmask(how, new, old, int32(unsafe.Sizeof(*new)))
}

func raise(sig uint32)
func raiseproc(sig uint32)

//go:noescape
func sched_getaffinity(pid, len uintptr, buf *byte) int32
func osyield()

//go:nosplit
func osyield_no_g() {
	osyield()
}

func pipe2(flags int32) (r, w int32, errno int32)

//go:nosplit
func fcntl(fd, cmd, arg int32) (ret int32, errno int32) {
	r, _, err := syscall.Syscall6(syscall.SYS_FCNTL, uintptr(fd), uintptr(cmd), uintptr(arg), 0, 0, 0)
	return int32(r), int32(err)
}

const (
	_si_max_size    = 128
	_sigev_max_size = 64
)

//go:nosplit
//go:nowritebarrierrec
func setsig(i uint32, fn uintptr) {
	var sa sigactiont
	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK | _SA_RESTORER | _SA_RESTART
	sigfillset(&sa.sa_mask)
	// Although Linux manpage says "sa_restorer element is obsolete and
	// should not be used". x86_64 kernel requires it. Only use it on
	// x86.
	if GOARCH == "386" || GOARCH == "amd64" {
		sa.sa_restorer = abi.FuncPCABI0(sigreturn__sigaction)
	}
	if fn == abi.FuncPCABIInternal(sighandler) { // abi.FuncPCABIInternal(sighandler) matches the callers in signal_unix.go
		if iscgo {
			fn = abi.FuncPCABI0(cgoSigtramp)
		} else {
			fn = abi.FuncPCABI0(sigtramp)
		}
	}
	sa.sa_handler = fn
	sigaction(i, &sa, nil)
}

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
	*(*uintptr)(unsafe.Pointer(&s.ss_sp)) = sp
}

//go:nosplit
func (c *sigctxt) fixsigcode(sig uint32) {
}

// sysSigaction calls the rt_sigaction system call.
//
//go:nosplit
func sysSigaction(sig uint32, new, old *sigactiont) {
	if rt_sigaction(uintptr(sig), new, old, unsafe.Sizeof(sigactiont{}.sa_mask)) != 0 {
		// Workaround for bugs in QEMU user mode emulation.
		//
		// QEMU turns calls to the sigaction system call into
		// calls to the C library sigaction call; the C
		// library call rejects attempts to call sigaction for
		// SIGCANCEL (32) or SIGSETXID (33).
		//
		// QEMU rejects calling sigaction on SIGRTMAX (64).
		//
		// Just ignore the error in these case. There isn't
		// anything we can do about it anyhow.
		if sig != 32 && sig != 33 && sig != 64 {
			// Use system stack to avoid split stack overflow on ppc64/ppc64le.
			systemstack(func() {
				throw("sigaction failed")
			})
		}
	}
}

// rt_sigaction is implemented in assembly.
//
//go:noescape
func rt_sigaction(sig uintptr, new, old *sigactiont, size uintptr) int32

func getpid() int
func tgkill(tgid, tid, sig int)

// signalM sends a signal to mp.
func signalM(mp *m, sig int) {
	tgkill(getpid(), int(mp.procid), sig)
}

// validSIGPROF compares this signal delivery's code against the signal sources
// that the profiler uses, returning whether the delivery should be processed.
// To be processed, a signal delivery from a known profiling mechanism should
// correspond to the best profiling mechanism available to this thread. Signals
// from other sources are always considered valid.
//
//go:nosplit
func validSIGPROF(mp *m, c *sigctxt) bool {
	code := int32(c.sigcode())
	setitimer := code == _SI_KERNEL
	timer_create := code == _SI_TIMER

	if !(setitimer || timer_create) {
		// The signal doesn't correspond to a profiling mechanism that the
		// runtime enables itself. There's no reason to process it, but there's
		// no reason to ignore it either.
		return true
	}

	if mp == nil {
		// Since we don't have an M, we can't check if there's an active
		// per-thread timer for this thread. We don't know how long this thread
		// has been around, and if it happened to interact with the Go scheduler
		// at a time when profiling was active (causing it to have a per-thread
		// timer). But it may have never interacted with the Go scheduler, or
		// never while profiling was active. To avoid double-counting, process
		// only signals from setitimer.
		//
		// When a custom cgo traceback function has been registered (on
		// platforms that support runtime.SetCgoTraceback), SIGPROF signals
		// delivered to a thread that cannot find a matching M do this check in
		// the assembly implementations of runtime.cgoSigtramp.
		return setitimer
	}

	// Having an M means the thread interacts with the Go scheduler, and we can
	// check whether there's an active per-thread timer for this thread.
	if mp.profileTimerValid.Load() {
		// If this M has its own per-thread CPU profiling interval timer, we
		// should track the SIGPROF signals that come from that timer (for
		// accurate reporting of its CPU usage; see issue 35057) and ignore any
		// that it gets from the process-wide setitimer (to not over-count its
		// CPU consumption).
		return timer_create
	}

	// No active per-thread timer means the only valid profiler is setitimer.
	return setitimer
}

func setProcessCPUProfiler(hz int32) {
	setProcessCPUProfilerTimer(hz)
}

func setThreadCPUProfiler(hz int32) {
	mp := getg().m
	mp.profilehz = hz

	// destroy any active timer
	if mp.profileTimerValid.Load() {
		timerid := mp.profileTimer
		mp.profileTimerValid.Store(false)
		mp.profileTimer = 0

		ret := timer_delete(timerid)
		if ret != 0 {
			print("runtime: failed to disable profiling timer; timer_delete(", timerid, ") errno=", -ret, "\n")
			throw("timer_delete")
		}
	}

	if hz == 0 {
		// If the goal was to disable profiling for this thread, then the job's done.
		return
	}

	// The period of the timer should be 1/Hz. For every "1/Hz" of additional
	// work, the user should expect one additional sample in the profile.
	//
	// But to scale down to very small amounts of application work, to observe
	// even CPU usage of "one tenth" of the requested period, set the initial
	// timing delay in a different way: So that "one tenth" of a period of CPU
	// spend shows up as a 10% chance of one sample (for an expected value of
	// 0.1 samples), and so that "two and six tenths" periods of CPU spend show
	// up as a 60% chance of 3 samples and a 40% chance of 2 samples (for an
	// expected value of 2.6). Set the initial delay to a value in the uniform
	// random distribution between 0 and the desired period. And because "0"
	// means "disable timer", add 1 so the half-open interval [0,period) turns
	// into (0,period].
	//
	// Otherwise, this would show up as a bias away from short-lived threads and
	// from threads that are only occasionally active: for example, when the
	// garbage collector runs on a mostly-idle system, the additional threads it
	// activates may do a couple milliseconds of GC-related work and nothing
	// else in the few seconds that the profiler observes.
	spec := new(itimerspec)
	spec.it_value.setNsec(1 + int64(cheaprandn(uint32(1e9/hz))))
	spec.it_interval.setNsec(1e9 / int64(hz))

	var timerid int32
	var sevp sigevent
	sevp.notify = _SIGEV_THREAD_ID
	sevp.signo = _SIGPROF
	sevp.sigev_notify_thread_id = int32(mp.procid)
	ret := timer_create(_CLOCK_THREAD_CPUTIME_ID, &sevp, &timerid)
	if ret != 0 {
		// If we cannot create a timer for this M, leave profileTimerValid false
		// to fall back to the process-wide setitimer profiler.
		return
	}

	ret = timer_settime(timerid, 0, spec, nil)
	if ret != 0 {
		print("runtime: failed to configure profiling timer; timer_settime(", timerid,
			", 0, {interval: {",
			spec.it_interval.tv_sec, "s + ", spec.it_interval.tv_nsec, "ns} value: {",
			spec.it_value.tv_sec, "s + ", spec.it_value.tv_nsec, "ns}}, nil) errno=", -ret, "\n")
		throw("timer_settime")
	}

	mp.profileTimer = timerid
	mp.profileTimerValid.Store(true)
}

// perThreadSyscallArgs contains the system call number, arguments, and
// expected return values for a system call to be executed on all threads.
type perThreadSyscallArgs struct {
	trap uintptr
	a1   uintptr
	a2   uintptr
	a3   uintptr
	a4   uintptr
	a5   uintptr
	a6   uintptr
	r1   uintptr
	r2   uintptr
}

// perThreadSyscall is the system call to execute for the ongoing
// doAllThreadsSyscall.
//
// perThreadSyscall may only be written while mp.needPerThreadSyscall == 0 on
// all Ms.
var perThreadSyscall perThreadSyscallArgs

// syscall_runtime_doAllThreadsSyscall and executes a specified system call on
// all Ms.
//
// The system call is expected to succeed and return the same value on every
// thread. If any threads do not match, the runtime throws.
//
//go:linkname syscall_runtime_doAllThreadsSyscall syscall.runtime_doAllThreadsSyscall
//go:uintptrescapes
func syscall_runtime_doAllThreadsSyscall(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	if iscgo {
		// In cgo, we are not aware of threads created in C, so this approach will not work.
		panic("doAllThreadsSyscall not supported with cgo enabled")
	}

	// STW to guarantee that user goroutines see an atomic change to thread
	// state. Without STW, goroutines could migrate Ms while change is in
	// progress and e.g., see state old -> new -> old -> new.
	//
	// N.B. Internally, this function does not depend on STW to
	// successfully change every thread. It is only needed for user
	// expectations, per above.
	stw := stopTheWorld(stwAllThreadsSyscall)

	// This function depends on several properties:
	//
	// 1. All OS threads that already exist are associated with an M in
	//    allm. i.e., we won't miss any pre-existing threads.
	// 2. All Ms listed in allm will eventually have an OS thread exist.
	//    i.e., they will set procid and be able to receive signals.
	// 3. OS threads created after we read allm will clone from a thread
	//    that has executed the system call. i.e., they inherit the
	//    modified state.
	//
	// We achieve these through different mechanisms:
	//
	// 1. Addition of new Ms to allm in allocm happens before clone of its
	//    OS thread later in newm.
	// 2. newm does acquirem to avoid being preempted, ensuring that new Ms
	//    created in allocm will eventually reach OS thread clone later in
	//    newm.
	// 3. We take allocmLock for write here to prevent allocation of new Ms
	//    while this function runs. Per (1), this prevents clone of OS
	//    threads that are not yet in allm.
	allocmLock.lock()

	// Disable preemption, preventing us from changing Ms, as we handle
	// this M specially.
	//
	// N.B. STW and lock() above do this as well, this is added for extra
	// clarity.
	acquirem()

	// N.B. allocmLock also prevents concurrent execution of this function,
	// serializing use of perThreadSyscall, mp.needPerThreadSyscall, and
	// ensuring all threads execute system calls from multiple calls in the
	// same order.

	r1, r2, errno := syscall.Syscall6(trap, a1, a2, a3, a4, a5, a6)
	if GOARCH == "ppc64" || GOARCH == "ppc64le" {
		// TODO(https://go.dev/issue/51192 ): ppc64 doesn't use r2.
		r2 = 0
	}
	if errno != 0 {
		releasem(getg().m)
		allocmLock.unlock()
		startTheWorld(stw)
		return r1, r2, errno
	}

	perThreadSyscall = perThreadSyscallArgs{
		trap: trap,
		a1:   a1,
		a2:   a2,
		a3:   a3,
		a4:   a4,
		a5:   a5,
		a6:   a6,
		r1:   r1,
		r2:   r2,
	}

	// Wait for all threads to start.
	//
	// As described above, some Ms have been added to allm prior to
	// allocmLock, but not yet completed OS clone and set procid.
	//
	// At minimum we must wait for a thread to set procid before we can
	// send it a signal.
	//
	// We take this one step further and wait for all threads to start
	// before sending any signals. This prevents system calls from getting
	// applied twice: once in the parent and once in the child, like so:
	//
	//          A                     B                  C
	//                         add C to allm
	// doAllThreadsSyscall
	//   allocmLock.lock()
	//   signal B
	//                         <receive signal>
	//                         execute syscall
	//                         <signal return>
	//                         clone C
	//                                             <thread start>
	//                                             set procid
	//   signal C
	//                                             <receive signal>
	//                                             execute syscall
	//                                             <signal return>
	//
	// In this case, thread C inherited the syscall-modified state from
	// thread B and did not need to execute the syscall, but did anyway
	// because doAllThreadsSyscall could not be sure whether it was
	// required.
	//
	// Some system calls may not be idempotent, so we ensure each thread
	// executes the system call exactly once.
	for mp := allm; mp != nil; mp = mp.alllink {
		for atomic.Load64(&mp.procid) == 0 {
			// Thread is starting.
			osyield()
		}
	}

	// Signal every other thread, where they will execute perThreadSyscall
	// from the signal handler.
	gp := getg()
	tid := gp.m.procid
	for mp := allm; mp != nil; mp = mp.alllink {
		if atomic.Load64(&mp.procid) == tid {
			// Our thread already performed the syscall.
			continue
		}
		mp.needPerThreadSyscall.Store(1)
		signalM(mp, sigPerThreadSyscall)
	}

	// Wait for all threads to complete.
	for mp := allm; mp != nil; mp = mp.alllink {
		if mp.procid == tid {
			continue
		}
		for mp.needPerThreadSyscall.Load() != 0 {
			osyield()
		}
	}

	perThreadSyscall = perThreadSyscallArgs{}

	releasem(getg().m)
	allocmLock.unlock()
	startTheWorld(stw)

	return r1, r2, errno
}

// runPerThreadSyscall runs perThreadSyscall for this M if required.
//
// This function throws if the system call returns with anything other than the
// expected values.
//
//go:nosplit
func runPerThreadSyscall() {
	gp := getg()
	if gp.m.needPerThreadSyscall.Load() == 0 {
		return
	}

	args := perThreadSyscall
	r1, r2, errno := syscall.Syscall6(args.trap, args.a1, args.a2, args.a3, args.a4, args.a5, args.a6)
	if GOARCH == "ppc64" || GOARCH == "ppc64le" {
		// TODO(https://go.dev/issue/51192 ): ppc64 doesn't use r2.
		r2 = 0
	}
	if errno != 0 || r1 != args.r1 || r2 != args.r2 {
		print("trap:", args.trap, ", a123456=[", args.a1, ",", args.a2, ",", args.a3, ",", args.a4, ",", args.a5, ",", args.a6, "]\n")
		print("results: got {r1=", r1, ",r2=", r2, ",errno=", errno, "}, want {r1=", args.r1, ",r2=", args.r2, ",errno=0}\n")
		fatal("AllThreadsSyscall6 results differ between threads; runtime corrupted")
	}

	gp.m.needPerThreadSyscall.Store(0)
}

const (
	_SI_USER     = 0
	_SI_TKILL    = -6
	_SYS_SECCOMP = 1
)

// sigFromUser reports whether the signal was sent because of a call
// to kill or tgkill.
//
//go:nosplit
func (c *sigctxt) sigFromUser() bool {
	code := int32(c.sigcode())
	return code == _SI_USER || code == _SI_TKILL
}

// sigFromSeccomp reports whether the signal was sent from seccomp.
//
//go:nosplit
func (c *sigctxt) sigFromSeccomp() bool {
	code := int32(c.sigcode())
	return code == _SYS_SECCOMP
}

//go:nosplit
func mprotect(addr unsafe.Pointer, n uintptr, prot int32) (ret int32, errno int32) {
	r, _, err := syscall.Syscall6(syscall.SYS_MPROTECT, uintptr(addr), n, uintptr(prot), 0, 0, 0)
	return int32(r), int32(err)
}
