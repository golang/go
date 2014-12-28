// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

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

func sigpanic() {
	g := getg()
	if !canpanic(g) {
		throw("unexpected signal during runtime execution")
	}

	// Native Client only invokes the exception handler for memory faults.
	g.sig = _SIGSEGV
	panicmem()
}
