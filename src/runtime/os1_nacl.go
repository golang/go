// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024)
	mp.gsignal.m = mp
}

func sigtramp()

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
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

func crash() {
	*(*int32)(nil) = 0
}

//go:nosplit
func getRandomData(r []byte) {
	// TODO: does nacl have a random source we can use?
	extendRandom(r, 0)
}

func goenvs() {
	goenvs_unix()
}

func initsig() {
}

//go:nosplit
func usleep(us uint32) {
	var ts timespec

	ts.tv_sec = int64(us / 1e6)
	ts.tv_nsec = int32(us%1e6) * 1e3
	nacl_nanosleep(&ts, nil)
}

func mstart_nacl()

func newosproc(mp *m, stk unsafe.Pointer) {
	tls := (*[3]unsafe.Pointer)(unsafe.Pointer(&mp.tls))
	tls[0] = unsafe.Pointer(mp.g0)
	tls[1] = unsafe.Pointer(mp)
	ret := nacl_thread_create(funcPC(mstart_nacl), stk, unsafe.Pointer(&tls[2]), nil)
	if ret < 0 {
		print("nacl_thread_create: error ", -ret, "\n")
		gothrow("newosproc")
	}
}

//go:nosplit
func semacreate() uintptr {
	var cond uintptr
	systemstack(func() {
		mu := nacl_mutex_create(0)
		if mu < 0 {
			print("nacl_mutex_create: error ", -mu, "\n")
			gothrow("semacreate")
		}
		c := nacl_cond_create(0)
		if c < 0 {
			print("nacl_cond_create: error ", -cond, "\n")
			gothrow("semacreate")
		}
		cond = uintptr(c)
		_g_ := getg()
		_g_.m.waitsemalock = uint32(mu)
	})
	return cond
}

//go:nosplit
func semasleep(ns int64) int32 {
	var ret int32

	systemstack(func() {
		_g_ := getg()
		if nacl_mutex_lock(int32(_g_.m.waitsemalock)) < 0 {
			gothrow("semasleep")
		}

		for _g_.m.waitsemacount == 0 {
			if ns < 0 {
				if nacl_cond_wait(int32(_g_.m.waitsema), int32(_g_.m.waitsemalock)) < 0 {
					gothrow("semasleep")
				}
			} else {
				var ts timespec
				end := ns + nanotime()
				ts.tv_sec = end / 1e9
				ts.tv_nsec = int32(end % 1e9)
				r := nacl_cond_timed_wait_abs(int32(_g_.m.waitsema), int32(_g_.m.waitsemalock), &ts)
				if r == -_ETIMEDOUT {
					nacl_mutex_unlock(int32(_g_.m.waitsemalock))
					ret = -1
					return
				}
				if r < 0 {
					gothrow("semasleep")
				}
			}
		}

		_g_.m.waitsemacount = 0
		nacl_mutex_unlock(int32(_g_.m.waitsemalock))
		ret = 0
	})
	return ret
}

//go:nosplit
func semawakeup(mp *m) {
	systemstack(func() {
		if nacl_mutex_lock(int32(mp.waitsemalock)) < 0 {
			gothrow("semawakeup")
		}
		if mp.waitsemacount != 0 {
			gothrow("semawakeup")
		}
		mp.waitsemacount = 1
		nacl_cond_signal(int32(mp.waitsema))
		nacl_mutex_unlock(int32(mp.waitsemalock))
	})
}

func memlimit() uintptr {
	return 0
}

// This runs on a foreign stack, without an m or a g.  No stack split.
//go:nosplit
func badsignal2() {
	write(2, unsafe.Pointer(&badsignal1[0]), int32(len(badsignal1)))
	exit(2)
}

var badsignal1 = []byte("runtime: signal received on thread not created by Go.\n")

func madvise(addr unsafe.Pointer, n uintptr, flags int32) {}
func munmap(addr unsafe.Pointer, n uintptr)               {}
func resetcpuprofiler(hz int32)                           {}
func sigdisable(uint32)                                   {}
func sigenable(uint32)                                    {}
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
