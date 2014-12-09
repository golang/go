// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:cgo_export_dynamic runtime.end _end
//go:cgo_export_dynamic runtime.etext _etext
//go:cgo_export_dynamic runtime.edata _edata

//go:cgo_import_dynamic libc____errno ___errno "libc.so"
//go:cgo_import_dynamic libc_clock_gettime clock_gettime "libc.so"
//go:cgo_import_dynamic libc_close close "libc.so"
//go:cgo_import_dynamic libc_exit exit "libc.so"
//go:cgo_import_dynamic libc_fstat fstat "libc.so"
//go:cgo_import_dynamic libc_getcontext getcontext "libc.so"
//go:cgo_import_dynamic libc_getrlimit getrlimit "libc.so"
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

//go:linkname libc____errno libc____errno
//go:linkname libc_clock_gettime libc_clock_gettime
//go:linkname libc_close libc_close
//go:linkname libc_exit libc_exit
//go:linkname libc_fstat libc_fstat
//go:linkname libc_getcontext libc_getcontext
//go:linkname libc_getrlimit libc_getrlimit
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

var (
	libc____errno,
	libc_clock_gettime,
	libc_close,
	libc_exit,
	libc_fstat,
	libc_getcontext,
	libc_getrlimit,
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
	libc_write libcFunc
)

var sigset_none = sigset{}
var sigset_all = sigset{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}

func getncpu() int32 {
	n := int32(sysconf(__SC_NPROCESSORS_ONLN))
	if n < 1 {
		return 1
	}
	return n
}

func osinit() {
	ncpu = getncpu()
}

func tstart_sysvicall()

func newosproc(mp *m, _ unsafe.Pointer) {
	var (
		attr pthreadattr
		oset sigset
		tid  pthread
		ret  int32
		size uint64
	)

	if pthread_attr_init(&attr) != 0 {
		gothrow("pthread_attr_init")
	}
	if pthread_attr_setstack(&attr, 0, 0x200000) != 0 {
		gothrow("pthread_attr_setstack")
	}
	if pthread_attr_getstack(&attr, unsafe.Pointer(&mp.g0.stack.hi), &size) != 0 {
		gothrow("pthread_attr_getstack")
	}
	mp.g0.stack.lo = mp.g0.stack.hi - uintptr(size)
	if pthread_attr_setdetachstate(&attr, _PTHREAD_CREATE_DETACHED) != 0 {
		gothrow("pthread_attr_setdetachstate")
	}

	// Disable signals during create, so that the new thread starts
	// with signals disabled.  It will enable them in minit.
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	ret = pthread_create(&tid, &attr, funcPC(tstart_sysvicall), unsafe.Pointer(mp))
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if ret != 0 {
		print("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", ret, ")\n")
		gothrow("newosproc")
	}
}

var urandom_dev = []byte("/dev/random\x00")

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

func miniterrno()

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
func minit() {
	_g_ := getg()
	asmcgocall(unsafe.Pointer(funcPC(miniterrno)), unsafe.Pointer(libc____errno))
	// Initialize signal handling
	signalstack((*byte)(unsafe.Pointer(_g_.m.gsignal.stack.lo)), 32*1024)
	sigprocmask(_SIG_SETMASK, &sigset_none, nil)
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

func setsig(i int32, fn uintptr, restart bool) {
	var sa sigactiont

	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK
	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK
	if restart {
		sa.sa_flags |= _SA_RESTART
	}
	sa.sa_mask = sigset_all
	if fn == funcPC(sighandler) {
		fn = funcPC(sigtramp)
	}
	*((*uintptr)(unsafe.Pointer(&sa._funcptr))) = fn
	sigaction(i, &sa, nil)
}

func getsig(i int32) uintptr {
	var sa sigactiont
	sigaction(i, nil, &sa)
	if *((*uintptr)(unsafe.Pointer(&sa._funcptr))) == funcPC(sigtramp) {
		return funcPC(sighandler)
	}
	return *((*uintptr)(unsafe.Pointer(&sa._funcptr)))
}

func signalstack(p *byte, n int32) {
	var st sigaltstackt
	st.ss_sp = (*byte)(unsafe.Pointer(p))
	st.ss_size = uint64(n)
	st.ss_flags = 0
	if p == nil {
		st.ss_flags = _SS_DISABLE
	}
	sigaltstack(&st, nil)
}

func unblocksignals() {
	sigprocmask(_SIG_SETMASK, &sigset_none, nil)
}

//go:nosplit
func semacreate() uintptr {
	var sem *semt
	_g_ := getg()

	// Call libc's malloc rather than malloc.  This will
	// allocate space on the C heap.  We can't call malloc
	// here because it could cause a deadlock.
	_g_.m.libcall.fn = uintptr(libc_malloc)
	_g_.m.libcall.n = 1
	memclr(unsafe.Pointer(&_g_.m.scratch), uintptr(len(_g_.m.scratch.v)))
	_g_.m.scratch.v[0] = unsafe.Sizeof(*sem)
	_g_.m.libcall.args = uintptr(unsafe.Pointer(&_g_.m.scratch))
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&_g_.m.libcall))
	sem = (*semt)(unsafe.Pointer(_g_.m.libcall.r1))
	if sem_init(sem, 0, 0) != 0 {
		gothrow("sem_init")
	}
	return uintptr(unsafe.Pointer(sem))
}

//go:nosplit
func semasleep(ns int64) int32 {
	_m_ := getg().m
	if ns >= 0 {
		_m_.ts.tv_sec = ns / 1000000000
		_m_.ts.tv_nsec = ns % 1000000000

		_m_.libcall.fn = uintptr(unsafe.Pointer(libc_sem_reltimedwait_np))
		_m_.libcall.n = 2
		memclr(unsafe.Pointer(&_m_.scratch), uintptr(len(_m_.scratch.v)))
		_m_.scratch.v[0] = _m_.waitsema
		_m_.scratch.v[1] = uintptr(unsafe.Pointer(&_m_.ts))
		_m_.libcall.args = uintptr(unsafe.Pointer(&_m_.scratch))
		asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&_m_.libcall))
		if *_m_.perrno != 0 {
			if *_m_.perrno == _ETIMEDOUT || *_m_.perrno == _EAGAIN || *_m_.perrno == _EINTR {
				return -1
			}
			gothrow("sem_reltimedwait_np")
		}
		return 0
	}
	for {
		_m_.libcall.fn = uintptr(unsafe.Pointer(libc_sem_wait))
		_m_.libcall.n = 1
		memclr(unsafe.Pointer(&_m_.scratch), uintptr(len(_m_.scratch.v)))
		_m_.scratch.v[0] = _m_.waitsema
		_m_.libcall.args = uintptr(unsafe.Pointer(&_m_.scratch))
		asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&_m_.libcall))
		if _m_.libcall.r1 == 0 {
			break
		}
		if *_m_.perrno == _EINTR {
			continue
		}
		gothrow("sem_wait")
	}
	return 0
}

//go:nosplit
func semawakeup(mp *m) {
	if sem_post((*semt)(unsafe.Pointer(mp.waitsema))) != 0 {
		gothrow("sem_post")
	}
}

//go:nosplit
func close(fd int32) int32 {
	return int32(sysvicall1(libc_close, uintptr(fd)))
}

//go:nosplit
func exit(r int32) {
	sysvicall1(libc_exit, uintptr(r))
}

//go:nosplit
func getcontext(context *ucontext) /* int32 */ {
	sysvicall1(libc_getcontext, uintptr(unsafe.Pointer(context)))
}

//go:nosplit
func madvise(addr unsafe.Pointer, n uintptr, flags int32) {
	sysvicall3(libc_madvise, uintptr(addr), uintptr(n), uintptr(flags))
}

//go:nosplit
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer {
	return unsafe.Pointer(sysvicall6(libc_mmap, uintptr(addr), uintptr(n), uintptr(prot), uintptr(flags), uintptr(fd), uintptr(off)))
}

//go:nosplit
func munmap(addr unsafe.Pointer, n uintptr) {
	sysvicall2(libc_munmap, uintptr(addr), uintptr(n))
}

func nanotime1()

//go:nosplit
func nanotime() int64 {
	return int64(sysvicall0(libcFunc(funcPC(nanotime1))))
}

//go:nosplit
func open(path *byte, mode, perm int32) int32 {
	return int32(sysvicall3(libc_open, uintptr(unsafe.Pointer(path)), uintptr(mode), uintptr(perm)))
}

func pthread_attr_destroy(attr *pthreadattr) int32 {
	return int32(sysvicall1(libc_pthread_attr_destroy, uintptr(unsafe.Pointer(attr))))
}

func pthread_attr_getstack(attr *pthreadattr, addr unsafe.Pointer, size *uint64) int32 {
	return int32(sysvicall3(libc_pthread_attr_getstack, uintptr(unsafe.Pointer(attr)), uintptr(addr), uintptr(unsafe.Pointer(size))))
}

func pthread_attr_init(attr *pthreadattr) int32 {
	return int32(sysvicall1(libc_pthread_attr_init, uintptr(unsafe.Pointer(attr))))
}

func pthread_attr_setdetachstate(attr *pthreadattr, state int32) int32 {
	return int32(sysvicall2(libc_pthread_attr_setdetachstate, uintptr(unsafe.Pointer(attr)), uintptr(state)))
}

func pthread_attr_setstack(attr *pthreadattr, addr uintptr, size uint64) int32 {
	return int32(sysvicall3(libc_pthread_attr_setstack, uintptr(unsafe.Pointer(attr)), uintptr(addr), uintptr(size)))
}

func pthread_create(thread *pthread, attr *pthreadattr, fn uintptr, arg unsafe.Pointer) int32 {
	return int32(sysvicall4(libc_pthread_create, uintptr(unsafe.Pointer(thread)), uintptr(unsafe.Pointer(attr)), uintptr(fn), uintptr(arg)))
}

func raise(sig int32) /* int32 */ {
	sysvicall1(libc_raise, uintptr(sig))
}

//go:nosplit
func read(fd int32, buf unsafe.Pointer, nbyte int32) int32 {
	return int32(sysvicall3(libc_read, uintptr(fd), uintptr(buf), uintptr(nbyte)))
}

//go:nosplit
func sem_init(sem *semt, pshared int32, value uint32) int32 {
	return int32(sysvicall3(libc_sem_init, uintptr(unsafe.Pointer(sem)), uintptr(pshared), uintptr(value)))
}

//go:nosplit
func sem_post(sem *semt) int32 {
	return int32(sysvicall1(libc_sem_post, uintptr(unsafe.Pointer(sem))))
}

//go:nosplit
func sem_reltimedwait_np(sem *semt, timeout *timespec) int32 {
	return int32(sysvicall2(libc_sem_reltimedwait_np, uintptr(unsafe.Pointer(sem)), uintptr(unsafe.Pointer(timeout))))
}

//go:nosplit
func sem_wait(sem *semt) int32 {
	return int32(sysvicall1(libc_sem_wait, uintptr(unsafe.Pointer(sem))))
}

func setitimer(which int32, value *itimerval, ovalue *itimerval) /* int32 */ {
	sysvicall3(libc_setitimer, uintptr(which), uintptr(unsafe.Pointer(value)), uintptr(unsafe.Pointer(ovalue)))
}

func sigaction(sig int32, act *sigactiont, oact *sigactiont) /* int32 */ {
	sysvicall3(libc_sigaction, uintptr(sig), uintptr(unsafe.Pointer(act)), uintptr(unsafe.Pointer(oact)))
}

func sigaltstack(ss *sigaltstackt, oss *sigaltstackt) /* int32 */ {
	sysvicall2(libc_sigaltstack, uintptr(unsafe.Pointer(ss)), uintptr(unsafe.Pointer(oss)))
}

func sigprocmask(how int32, set *sigset, oset *sigset) /* int32 */ {
	sysvicall3(libc_sigprocmask, uintptr(how), uintptr(unsafe.Pointer(set)), uintptr(unsafe.Pointer(oset)))
}

func sysconf(name int32) int64 {
	return int64(sysvicall1(libc_sysconf, uintptr(name)))
}

func usleep1(uint32)

//go:nosplit
func usleep(µs uint32) {
	usleep1(µs)
}

//go:nosplit
func write(fd uintptr, buf unsafe.Pointer, nbyte int32) int32 {
	return int32(sysvicall3(libc_write, uintptr(fd), uintptr(buf), uintptr(nbyte)))
}

func osyield1()

//go:nosplit
func osyield() {
	_g_ := getg()

	// Check the validity of m because we might be called in cgo callback
	// path early enough where there isn't a m available yet.
	if _g_ != nil && _g_.m != nil {
		sysvicall0(libc_sched_yield)
		return
	}
	osyield1()
}
