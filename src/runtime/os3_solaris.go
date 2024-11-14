// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/runtime/atomic"
	"unsafe"
)

//go:cgo_export_dynamic runtime.end _end
//go:cgo_export_dynamic runtime.etext _etext
//go:cgo_export_dynamic runtime.edata _edata

//go:cgo_import_dynamic libc____errno ___errno "libc.so"
//go:cgo_import_dynamic libc_clock_gettime clock_gettime "libc.so"
//go:cgo_import_dynamic libc_exit _exit "libc.so"
//go:cgo_import_dynamic libc_getcontext getcontext "libc.so"
//go:cgo_import_dynamic libc_kill kill "libc.so"
//go:cgo_import_dynamic libc_madvise madvise "libc.so"
//go:cgo_import_dynamic libc_malloc malloc "libc.so"
//go:cgo_import_dynamic libc_mmap mmap "libc.so"
//go:cgo_import_dynamic libc_munmap munmap "libc.so"
//go:cgo_import_dynamic libc_open open "libc.so"
//go:cgo_import_dynamic libc_pthread_attr_destroy pthread_attr_destroy "libc.so"
//go:cgo_import_dynamic libc_pthread_attr_getstack pthread_attr_getstack "libc.so"
//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "libc.so"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libc.so"
//go:cgo_import_dynamic libc_pthread_attr_setstack pthread_attr_setstack "libc.so"
//go:cgo_import_dynamic libc_pthread_create pthread_create "libc.so"
//go:cgo_import_dynamic libc_pthread_self pthread_self "libc.so"
//go:cgo_import_dynamic libc_pthread_kill pthread_kill "libc.so"
//go:cgo_import_dynamic libc_raise raise "libc.so"
//go:cgo_import_dynamic libc_read read "libc.so"
//go:cgo_import_dynamic libc_select select "libc.so"
//go:cgo_import_dynamic libc_sched_yield sched_yield "libc.so"
//go:cgo_import_dynamic libc_sem_init sem_init "libc.so"
//go:cgo_import_dynamic libc_sem_post sem_post "libc.so"
//go:cgo_import_dynamic libc_sem_reltimedwait_np sem_reltimedwait_np "libc.so"
//go:cgo_import_dynamic libc_sem_wait sem_wait "libc.so"
//go:cgo_import_dynamic libc_setitimer setitimer "libc.so"
//go:cgo_import_dynamic libc_sigaction sigaction "libc.so"
//go:cgo_import_dynamic libc_sigaltstack sigaltstack "libc.so"
//go:cgo_import_dynamic libc_sigprocmask sigprocmask "libc.so"
//go:cgo_import_dynamic libc_sysconf sysconf "libc.so"
//go:cgo_import_dynamic libc_usleep usleep "libc.so"
//go:cgo_import_dynamic libc_write write "libc.so"
//go:cgo_import_dynamic libc_pipe2 pipe2 "libc.so"

//go:linkname libc____errno libc____errno
//go:linkname libc_clock_gettime libc_clock_gettime
//go:linkname libc_exit libc_exit
//go:linkname libc_getcontext libc_getcontext
//go:linkname libc_kill libc_kill
//go:linkname libc_madvise libc_madvise
//go:linkname libc_malloc libc_malloc
//go:linkname libc_mmap libc_mmap
//go:linkname libc_munmap libc_munmap
//go:linkname libc_open libc_open
//go:linkname libc_pthread_attr_destroy libc_pthread_attr_destroy
//go:linkname libc_pthread_attr_getstack libc_pthread_attr_getstack
//go:linkname libc_pthread_attr_init libc_pthread_attr_init
//go:linkname libc_pthread_attr_setdetachstate libc_pthread_attr_setdetachstate
//go:linkname libc_pthread_attr_setstack libc_pthread_attr_setstack
//go:linkname libc_pthread_create libc_pthread_create
//go:linkname libc_pthread_self libc_pthread_self
//go:linkname libc_pthread_kill libc_pthread_kill
//go:linkname libc_raise libc_raise
//go:linkname libc_read libc_read
//go:linkname libc_select libc_select
//go:linkname libc_sched_yield libc_sched_yield
//go:linkname libc_sem_init libc_sem_init
//go:linkname libc_sem_post libc_sem_post
//go:linkname libc_sem_reltimedwait_np libc_sem_reltimedwait_np
//go:linkname libc_sem_wait libc_sem_wait
//go:linkname libc_setitimer libc_setitimer
//go:linkname libc_sigaction libc_sigaction
//go:linkname libc_sigaltstack libc_sigaltstack
//go:linkname libc_sigprocmask libc_sigprocmask
//go:linkname libc_sysconf libc_sysconf
//go:linkname libc_usleep libc_usleep
//go:linkname libc_write libc_write
//go:linkname libc_pipe2 libc_pipe2

var (
	libc____errno,
	libc_clock_gettime,
	libc_exit,
	libc_getcontext,
	libc_kill,
	libc_madvise,
	libc_malloc,
	libc_mmap,
	libc_munmap,
	libc_open,
	libc_pthread_attr_destroy,
	libc_pthread_attr_getstack,
	libc_pthread_attr_init,
	libc_pthread_attr_setdetachstate,
	libc_pthread_attr_setstack,
	libc_pthread_create,
	libc_pthread_self,
	libc_pthread_kill,
	libc_raise,
	libc_read,
	libc_sched_yield,
	libc_select,
	libc_sem_init,
	libc_sem_post,
	libc_sem_reltimedwait_np,
	libc_sem_wait,
	libc_setitimer,
	libc_sigaction,
	libc_sigaltstack,
	libc_sigprocmask,
	libc_sysconf,
	libc_usleep,
	libc_write,
	libc_pipe2 libcFunc
)

var sigset_all = sigset{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}

func getPageSize() uintptr {
	n := int32(sysconf(__SC_PAGESIZE))
	if n <= 0 {
		return 0
	}
	return uintptr(n)
}

func osinit() {
	// Call miniterrno so that we can safely make system calls
	// before calling minit on m0.
	asmcgocall(unsafe.Pointer(abi.FuncPCABI0(miniterrno)), unsafe.Pointer(&libc____errno))

	ncpu = getncpu()
	if physPageSize == 0 {
		physPageSize = getPageSize()
	}
}

func tstart_sysvicall(newm *m) uint32

// May run with m.p==nil, so write barriers are not allowed.
//
//go:nowritebarrier
func newosproc(mp *m) {
	var (
		attr pthreadattr
		oset sigset
		tid  pthread
		ret  int32
		size uint64
	)

	if pthread_attr_init(&attr) != 0 {
		throw("pthread_attr_init")
	}
	// Allocate a new 2MB stack.
	if pthread_attr_setstack(&attr, 0, 0x200000) != 0 {
		throw("pthread_attr_setstack")
	}
	// Read back the allocated stack.
	if pthread_attr_getstack(&attr, unsafe.Pointer(&mp.g0.stack.hi), &size) != 0 {
		throw("pthread_attr_getstack")
	}
	mp.g0.stack.lo = mp.g0.stack.hi - uintptr(size)
	if pthread_attr_setdetachstate(&attr, _PTHREAD_CREATE_DETACHED) != 0 {
		throw("pthread_attr_setdetachstate")
	}

	// Disable signals during create, so that the new thread starts
	// with signals disabled. It will enable them in minit.
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	ret = retryOnEAGAIN(func { pthread_create(&tid, &attr, abi.FuncPCABI0(tstart_sysvicall), unsafe.Pointer(mp)) })
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if ret != 0 {
		print("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", ret, ")\n")
		if ret == _EAGAIN {
			println("runtime: may need to increase max user processes (ulimit -u)")
		}
		throw("newosproc")
	}
}

func exitThread(wait *atomic.Uint32) {
	// We should never reach exitThread on Solaris because we let
	// libc clean up threads.
	throw("exitThread")
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

func miniterrno()

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
	asmcgocall(unsafe.Pointer(abi.FuncPCABI0(miniterrno)), unsafe.Pointer(&libc____errno))

	minitSignals()

	getg().m.procid = uint64(pthread_self())
}

// Called from dropm to undo the effect of an minit.
func unminit() {
	unminitSignals()
	getg().m.procid = 0
}

// Called from exitm, but not from drop, to undo the effect of thread-owned
// resources in minit, semacreate, or elsewhere. Do not take locks after calling this.
func mdestroy(mp *m) {
}

func sigtramp()

//go:nosplit
//go:nowritebarrierrec
func setsig(i uint32, fn uintptr) {
	var sa sigactiont

	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK | _SA_RESTART
	sa.sa_mask = sigset_all
	if fn == abi.FuncPCABIInternal(sighandler) { // abi.FuncPCABIInternal(sighandler) matches the callers in signal_unix.go
		fn = abi.FuncPCABI0(sigtramp)
	}
	*((*uintptr)(unsafe.Pointer(&sa._funcptr))) = fn
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
	return *((*uintptr)(unsafe.Pointer(&sa._funcptr)))
}

// setSignalstackSP sets the ss_sp field of a stackt.
//
//go:nosplit
func setSignalstackSP(s *stackt, sp uintptr) {
	*(*uintptr)(unsafe.Pointer(&s.ss_sp)) = sp
}

//go:nosplit
//go:nowritebarrierrec
func sigaddset(mask *sigset, i int) {
	mask.__sigbits[(i-1)/32] |= 1 << ((uint32(i) - 1) & 31)
}

func sigdelset(mask *sigset, i int) {
	mask.__sigbits[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
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

//go:nosplit
func semacreate(mp *m) {
	if mp.waitsema != 0 {
		return
	}

	var sem *semt

	// Call libc's malloc rather than malloc. This will
	// allocate space on the C heap. We can't call malloc
	// here because it could cause a deadlock.
	mp.libcall.fn = uintptr(unsafe.Pointer(&libc_malloc))
	mp.libcall.n = 1
	mp.scratch = mscratch{}
	mp.scratch.v[0] = unsafe.Sizeof(*sem)
	mp.libcall.args = uintptr(unsafe.Pointer(&mp.scratch))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&mp.libcall))
	sem = (*semt)(unsafe.Pointer(mp.libcall.r1))
	if sem_init(sem, 0, 0) != 0 {
		throw("sem_init")
	}
	mp.waitsema = uintptr(unsafe.Pointer(sem))
}

//go:nosplit
func semasleep(ns int64) int32 {
	mp := getg().m
	if ns >= 0 {
		mp.ts.tv_sec = ns / 1000000000
		mp.ts.tv_nsec = ns % 1000000000

		mp.libcall.fn = uintptr(unsafe.Pointer(&libc_sem_reltimedwait_np))
		mp.libcall.n = 2
		mp.scratch = mscratch{}
		mp.scratch.v[0] = mp.waitsema
		mp.scratch.v[1] = uintptr(unsafe.Pointer(&mp.ts))
		mp.libcall.args = uintptr(unsafe.Pointer(&mp.scratch))
		asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&mp.libcall))
		if *mp.perrno != 0 {
			if *mp.perrno == _ETIMEDOUT || *mp.perrno == _EAGAIN || *mp.perrno == _EINTR {
				return -1
			}
			throw("sem_reltimedwait_np")
		}
		return 0
	}
	for {
		mp.libcall.fn = uintptr(unsafe.Pointer(&libc_sem_wait))
		mp.libcall.n = 1
		mp.scratch = mscratch{}
		mp.scratch.v[0] = mp.waitsema
		mp.libcall.args = uintptr(unsafe.Pointer(&mp.scratch))
		asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&mp.libcall))
		if mp.libcall.r1 == 0 {
			break
		}
		if *mp.perrno == _EINTR {
			continue
		}
		throw("sem_wait")
	}
	return 0
}

//go:nosplit
func semawakeup(mp *m) {
	if sem_post((*semt)(unsafe.Pointer(mp.waitsema))) != 0 {
		throw("sem_post")
	}
}

//go:nosplit
func closefd(fd int32) int32 {
	return int32(sysvicall1(&libc_close, uintptr(fd)))
}

//go:nosplit
func exit(r int32) {
	sysvicall1(&libc_exit, uintptr(r))
}

//go:nosplit
func getcontext(context *ucontext) /* int32 */ {
	sysvicall1(&libc_getcontext, uintptr(unsafe.Pointer(context)))
}

//go:nosplit
func madvise(addr unsafe.Pointer, n uintptr, flags int32) {
	sysvicall3(&libc_madvise, uintptr(addr), uintptr(n), uintptr(flags))
}

//go:nosplit
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (unsafe.Pointer, int) {
	p, err := doMmap(uintptr(addr), n, uintptr(prot), uintptr(flags), uintptr(fd), uintptr(off))
	if p == ^uintptr(0) {
		return nil, int(err)
	}
	return unsafe.Pointer(p), 0
}

//go:nosplit
//go:cgo_unsafe_args
func doMmap(addr, n, prot, flags, fd, off uintptr) (uintptr, uintptr) {
	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(&libc_mmap))
	libcall.n = 6
	libcall.args = uintptr(noescape(unsafe.Pointer(&addr)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	return libcall.r1, libcall.err
}

//go:nosplit
func munmap(addr unsafe.Pointer, n uintptr) {
	sysvicall2(&libc_munmap, uintptr(addr), uintptr(n))
}

const (
	_CLOCK_REALTIME  = 3
	_CLOCK_MONOTONIC = 4
)

//go:nosplit
func nanotime1() int64 {
	var ts mts
	sysvicall2(&libc_clock_gettime, _CLOCK_MONOTONIC, uintptr(unsafe.Pointer(&ts)))
	return ts.tv_sec*1e9 + ts.tv_nsec
}

//go:nosplit
func open(path *byte, mode, perm int32) int32 {
	return int32(sysvicall3(&libc_open, uintptr(unsafe.Pointer(path)), uintptr(mode), uintptr(perm)))
}

func pthread_attr_destroy(attr *pthreadattr) int32 {
	return int32(sysvicall1(&libc_pthread_attr_destroy, uintptr(unsafe.Pointer(attr))))
}

func pthread_attr_getstack(attr *pthreadattr, addr unsafe.Pointer, size *uint64) int32 {
	return int32(sysvicall3(&libc_pthread_attr_getstack, uintptr(unsafe.Pointer(attr)), uintptr(addr), uintptr(unsafe.Pointer(size))))
}

func pthread_attr_init(attr *pthreadattr) int32 {
	return int32(sysvicall1(&libc_pthread_attr_init, uintptr(unsafe.Pointer(attr))))
}

func pthread_attr_setdetachstate(attr *pthreadattr, state int32) int32 {
	return int32(sysvicall2(&libc_pthread_attr_setdetachstate, uintptr(unsafe.Pointer(attr)), uintptr(state)))
}

func pthread_attr_setstack(attr *pthreadattr, addr uintptr, size uint64) int32 {
	return int32(sysvicall3(&libc_pthread_attr_setstack, uintptr(unsafe.Pointer(attr)), uintptr(addr), uintptr(size)))
}

func pthread_create(thread *pthread, attr *pthreadattr, fn uintptr, arg unsafe.Pointer) int32 {
	return int32(sysvicall4(&libc_pthread_create, uintptr(unsafe.Pointer(thread)), uintptr(unsafe.Pointer(attr)), uintptr(fn), uintptr(arg)))
}

func pthread_self() pthread {
	return pthread(sysvicall0(&libc_pthread_self))
}

func signalM(mp *m, sig int) {
	sysvicall2(&libc_pthread_kill, uintptr(pthread(mp.procid)), uintptr(sig))
}

//go:nosplit
//go:nowritebarrierrec
func raise(sig uint32) /* int32 */ {
	sysvicall1(&libc_raise, uintptr(sig))
}

func raiseproc(sig uint32) /* int32 */ {
	pid := sysvicall0(&libc_getpid)
	sysvicall2(&libc_kill, pid, uintptr(sig))
}

//go:nosplit
func read(fd int32, buf unsafe.Pointer, nbyte int32) int32 {
	r1, err := sysvicall3Err(&libc_read, uintptr(fd), uintptr(buf), uintptr(nbyte))
	if c := int32(r1); c >= 0 {
		return c
	}
	return -int32(err)
}

//go:nosplit
func sem_init(sem *semt, pshared int32, value uint32) int32 {
	return int32(sysvicall3(&libc_sem_init, uintptr(unsafe.Pointer(sem)), uintptr(pshared), uintptr(value)))
}

//go:nosplit
func sem_post(sem *semt) int32 {
	return int32(sysvicall1(&libc_sem_post, uintptr(unsafe.Pointer(sem))))
}

//go:nosplit
func sem_reltimedwait_np(sem *semt, timeout *timespec) int32 {
	return int32(sysvicall2(&libc_sem_reltimedwait_np, uintptr(unsafe.Pointer(sem)), uintptr(unsafe.Pointer(timeout))))
}

//go:nosplit
func sem_wait(sem *semt) int32 {
	return int32(sysvicall1(&libc_sem_wait, uintptr(unsafe.Pointer(sem))))
}

func setitimer(which int32, value *itimerval, ovalue *itimerval) /* int32 */ {
	sysvicall3(&libc_setitimer, uintptr(which), uintptr(unsafe.Pointer(value)), uintptr(unsafe.Pointer(ovalue)))
}

//go:nosplit
//go:nowritebarrierrec
func sigaction(sig uint32, act *sigactiont, oact *sigactiont) /* int32 */ {
	sysvicall3(&libc_sigaction, uintptr(sig), uintptr(unsafe.Pointer(act)), uintptr(unsafe.Pointer(oact)))
}

//go:nosplit
//go:nowritebarrierrec
func sigaltstack(ss *stackt, oss *stackt) /* int32 */ {
	sysvicall2(&libc_sigaltstack, uintptr(unsafe.Pointer(ss)), uintptr(unsafe.Pointer(oss)))
}

//go:nosplit
//go:nowritebarrierrec
func sigprocmask(how int32, set *sigset, oset *sigset) /* int32 */ {
	sysvicall3(&libc_sigprocmask, uintptr(how), uintptr(unsafe.Pointer(set)), uintptr(unsafe.Pointer(oset)))
}

func sysconf(name int32) int64 {
	return int64(sysvicall1(&libc_sysconf, uintptr(name)))
}

func usleep1(usec uint32)

//go:nosplit
func usleep_no_g(µs uint32) {
	usleep1(µs)
}

//go:nosplit
func usleep(µs uint32) {
	usleep1(µs)
}

func walltime() (sec int64, nsec int32) {
	var ts mts
	sysvicall2(&libc_clock_gettime, _CLOCK_REALTIME, uintptr(unsafe.Pointer(&ts)))
	return ts.tv_sec, int32(ts.tv_nsec)
}

//go:nosplit
func write1(fd uintptr, buf unsafe.Pointer, nbyte int32) int32 {
	r1, err := sysvicall3Err(&libc_write, fd, uintptr(buf), uintptr(nbyte))
	if c := int32(r1); c >= 0 {
		return c
	}
	return -int32(err)
}

//go:nosplit
func pipe2(flags int32) (r, w int32, errno int32) {
	var p [2]int32
	_, e := sysvicall2Err(&libc_pipe2, uintptr(noescape(unsafe.Pointer(&p))), uintptr(flags))
	return p[0], p[1], int32(e)
}

//go:nosplit
func fcntl(fd, cmd, arg int32) (ret int32, errno int32) {
	r1, err := sysvicall3Err(&libc_fcntl, uintptr(fd), uintptr(cmd), uintptr(arg))
	return int32(r1), int32(err)
}

func osyield1()

//go:nosplit
func osyield_no_g() {
	osyield1()
}

//go:nosplit
func osyield() {
	sysvicall0(&libc_sched_yield)
}

//go:linkname executablePath os.executablePath
var executablePath string

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
	_AT_NULL         = 0    // Terminates the vector
	_AT_PAGESZ       = 6    // Page size in bytes
	_AT_SUN_EXECNAME = 2014 // exec() path name
)

func sysauxv(auxv []uintptr) (pairs int) {
	var i int
	for i = 0; auxv[i] != _AT_NULL; i += 2 {
		tag, val := auxv[i], auxv[i+1]
		switch tag {
		case _AT_PAGESZ:
			physPageSize = val
		case _AT_SUN_EXECNAME:
			executablePath = gostringnocopy((*byte)(unsafe.Pointer(val)))
		}
	}
	return i / 2
}

// sigPerThreadSyscall is only used on linux, so we assign a bogus signal
// number.
const sigPerThreadSyscall = 1 << 31

//go:nosplit
func runPerThreadSyscall() {
	throw("runPerThreadSyscall only valid on linux")
}
