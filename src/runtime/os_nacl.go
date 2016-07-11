// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type mOS struct {
	waitsema      int32 // semaphore for parking on locks
	waitsemacount int32
	waitsemalock  int32
}

func nacl_exception_stack(p uintptr, size int32) int32
func nacl_exception_handler(fn uintptr, arg unsafe.Pointer) int32
func nacl_sem_create(flag int32) int32
func nacl_sem_wait(sem int32) int32
func nacl_sem_post(sem int32) int32
func nacl_mutex_create(flag int32) int32
func nacl_mutex_lock(mutex int32) int32
func nacl_mutex_trylock(mutex int32) int32
func nacl_mutex_unlock(mutex int32) int32
func nacl_cond_create(flag int32) int32
func nacl_cond_wait(cond, n int32) int32
func nacl_cond_signal(cond int32) int32
func nacl_cond_broadcast(cond int32) int32

//go:noescape
func nacl_cond_timed_wait_abs(cond, lock int32, ts *timespec) int32
func nacl_thread_create(fn uintptr, stk, tls, xx unsafe.Pointer) int32

//go:noescape
func nacl_nanosleep(ts, extra *timespec) int32
func nanotime() int64
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer
func exit(code int32)
func osyield()

//go:noescape
func write(fd uintptr, p unsafe.Pointer, n int32) int32

//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {
	throw("too many writes on closed pipe")
}

func dieFromSignal(sig int32) {
	exit(2)
}

func sigpanic() {
	g := getg()
	if !canpanic(g) {
		throw("unexpected signal during runtime execution")
	}

	// Native Client only invokes the exception handler for memory faults.
	g.sig = _SIGSEGV
	panicmem()
}

func raiseproc(sig int32) {
}

// Stubs so tests can link correctly. These should never be called.
func open(name *byte, mode, perm int32) int32
func closefd(fd int32) int32
func read(fd int32, p unsafe.Pointer, n int32) int32

type sigset struct{}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024)
	mp.gsignal.m = mp
}

func sigtramp()

//go:nosplit
func msigsave(mp *m) {
}

//go:nosplit
func msigrestore(sigmask sigset) {
}

//go:nosplit
func sigblock() {
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
	_g_ := getg()

	// Initialize signal handling
	ret := nacl_exception_stack(_g_.m.gsignal.stack.lo, 32*1024)
	if ret < 0 {
		print("runtime: nacl_exception_stack: error ", -ret, "\n")
	}

	ret = nacl_exception_handler(funcPC(sigtramp), nil)
	if ret < 0 {
		print("runtime: nacl_exception_handler: error ", -ret, "\n")
	}
}

// Called from dropm to undo the effect of an minit.
func unminit() {
}

func osinit() {
	ncpu = 1
	getg().m.procid = 2
	//nacl_exception_handler(funcPC(sigtramp), nil);
}

func signame(sig uint32) string {
	if sig >= uint32(len(sigtable)) {
		return ""
	}
	return sigtable[sig].name
}

func crash() {
	*(*int32)(nil) = 0
}

//go:noescape
func getRandomData([]byte)

func goenvs() {
	goenvs_unix()
}

func initsig(preinit bool) {
}

//go:nosplit
func usleep(us uint32) {
	var ts timespec

	ts.tv_sec = int64(us / 1e6)
	ts.tv_nsec = int32(us%1e6) * 1e3
	nacl_nanosleep(&ts, nil)
}

func mstart_nacl()

// May run with m.p==nil, so write barriers are not allowed.
//go:nowritebarrier
func newosproc(mp *m, stk unsafe.Pointer) {
	mp.tls[0] = uintptr(unsafe.Pointer(mp.g0))
	mp.tls[1] = uintptr(unsafe.Pointer(mp))
	ret := nacl_thread_create(funcPC(mstart_nacl), stk, unsafe.Pointer(&mp.tls[2]), nil)
	if ret < 0 {
		print("nacl_thread_create: error ", -ret, "\n")
		throw("newosproc")
	}
}

//go:nosplit
func semacreate(mp *m) {
	if mp.waitsema != 0 {
		return
	}
	systemstack(func() {
		mu := nacl_mutex_create(0)
		if mu < 0 {
			print("nacl_mutex_create: error ", -mu, "\n")
			throw("semacreate")
		}
		c := nacl_cond_create(0)
		if c < 0 {
			print("nacl_cond_create: error ", -c, "\n")
			throw("semacreate")
		}
		mp.waitsema = c
		mp.waitsemalock = mu
	})
}

//go:nosplit
func semasleep(ns int64) int32 {
	var ret int32

	systemstack(func() {
		_g_ := getg()
		if nacl_mutex_lock(_g_.m.waitsemalock) < 0 {
			throw("semasleep")
		}

		for _g_.m.waitsemacount == 0 {
			if ns < 0 {
				if nacl_cond_wait(_g_.m.waitsema, _g_.m.waitsemalock) < 0 {
					throw("semasleep")
				}
			} else {
				var ts timespec
				end := ns + nanotime()
				ts.tv_sec = end / 1e9
				ts.tv_nsec = int32(end % 1e9)
				r := nacl_cond_timed_wait_abs(_g_.m.waitsema, _g_.m.waitsemalock, &ts)
				if r == -_ETIMEDOUT {
					nacl_mutex_unlock(_g_.m.waitsemalock)
					ret = -1
					return
				}
				if r < 0 {
					throw("semasleep")
				}
			}
		}

		_g_.m.waitsemacount = 0
		nacl_mutex_unlock(_g_.m.waitsemalock)
		ret = 0
	})
	return ret
}

//go:nosplit
func semawakeup(mp *m) {
	systemstack(func() {
		if nacl_mutex_lock(mp.waitsemalock) < 0 {
			throw("semawakeup")
		}
		if mp.waitsemacount != 0 {
			throw("semawakeup")
		}
		mp.waitsemacount = 1
		nacl_cond_signal(mp.waitsema)
		nacl_mutex_unlock(mp.waitsemalock)
	})
}

func memlimit() uintptr {
	return 0
}

// This runs on a foreign stack, without an m or a g. No stack split.
//go:nosplit
//go:norace
//go:nowritebarrierrec
func badsignal(sig uintptr) {
	cgocallback(unsafe.Pointer(funcPC(badsignalgo)), noescape(unsafe.Pointer(&sig)), unsafe.Sizeof(sig), 0)
}

func badsignalgo(sig uintptr) {
	if !sigsend(uint32(sig)) {
		// A foreign thread received the signal sig, and the
		// Go code does not want to handle it.
		raisebadsignal(int32(sig))
	}
}

// This runs on a foreign stack, without an m or a g. No stack split.
//go:nosplit
func badsignal2() {
	write(2, unsafe.Pointer(&badsignal1[0]), int32(len(badsignal1)))
	exit(2)
}

var badsignal1 = []byte("runtime: signal received on thread not created by Go.\n")

func raisebadsignal(sig int32) {
	badsignal2()
}

func madvise(addr unsafe.Pointer, n uintptr, flags int32) {}
func munmap(addr unsafe.Pointer, n uintptr)               {}
func resetcpuprofiler(hz int32)                           {}
func sigdisable(uint32)                                   {}
func sigenable(uint32)                                    {}
func sigignore(uint32)                                    {}
func closeonexec(int32)                                   {}

var writelock uint32 // test-and-set spin lock for write

/*
An attempt at IRT. Doesn't work. See end of sys_nacl_amd64.s.

void (*nacl_irt_query)(void);

int8 nacl_irt_basic_v0_1_str[] = "nacl-irt-basic-0.1";
void *nacl_irt_basic_v0_1[6]; // exit, gettod, clock, nanosleep, sched_yield, sysconf
int32 nacl_irt_basic_v0_1_size = sizeof(nacl_irt_basic_v0_1);

int8 nacl_irt_memory_v0_3_str[] = "nacl-irt-memory-0.3";
void *nacl_irt_memory_v0_3[3]; // mmap, munmap, mprotect
int32 nacl_irt_memory_v0_3_size = sizeof(nacl_irt_memory_v0_3);

int8 nacl_irt_thread_v0_1_str[] = "nacl-irt-thread-0.1";
void *nacl_irt_thread_v0_1[3]; // thread_create, thread_exit, thread_nice
int32 nacl_irt_thread_v0_1_size = sizeof(nacl_irt_thread_v0_1);
*/
