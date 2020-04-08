// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package runtime

import (
	"unsafe"
)

func exit(code int32)

func write1(fd uintptr, p unsafe.Pointer, n int32) int32 {
	if fd > 2 {
		throw("runtime.write to fd > 2 is unsupported")
	}
	wasmWrite(fd, p, n)
	return n
}

// Stubs so tests can link correctly. These should never be called.
func open(name *byte, mode, perm int32) int32        { panic("not implemented") }
func closefd(fd int32) int32                         { panic("not implemented") }
func read(fd int32, p unsafe.Pointer, n int32) int32 { panic("not implemented") }

//go:noescape
func wasmWrite(fd uintptr, p unsafe.Pointer, n int32)

func usleep(usec uint32)

func exitThread(wait *uint32)

type mOS struct{}

func osyield()

const _SIGSEGV = 0xb

func sigpanic() {
	g := getg()
	if !canpanic(g) {
		throw("unexpected signal during runtime execution")
	}

	// js only invokes the exception handler for memory faults.
	g.sig = _SIGSEGV
	panicmem()
}

type sigset struct{}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024)
	mp.gsignal.m = mp
}

//go:nosplit
func msigsave(mp *m) {
}

//go:nosplit
func msigrestore(sigmask sigset) {
}

//go:nosplit
//go:nowritebarrierrec
func clearSignalHandlers() {
}

//go:nosplit
func sigblock() {
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
}

// Called from dropm to undo the effect of an minit.
func unminit() {
}

func osinit() {
	ncpu = 1
	getg().m.procid = 2
	physPageSize = 64 * 1024
}

// wasm has no signals
const _NSIG = 0

func signame(sig uint32) string {
	return ""
}

func crash() {
	*(*int32)(nil) = 0
}

func getRandomData(r []byte)

func goenvs() {
	goenvs_unix()
}

func initsig(preinit bool) {
}

// May run with m.p==nil, so write barriers are not allowed.
//go:nowritebarrier
func newosproc(mp *m) {
	panic("newosproc: not implemented")
}

func setProcessCPUProfiler(hz int32) {}
func setThreadCPUProfiler(hz int32)  {}
func sigdisable(uint32)              {}
func sigenable(uint32)               {}
func sigignore(uint32)               {}

//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {
	throw("too many writes on closed pipe")
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}

//go:linkname syscall_now syscall.now
func syscall_now() (sec int64, nsec int32) {
	sec, nsec, _ = time_now()
	return
}

// gsignalStack is unused on js.
type gsignalStack struct{}

const preemptMSupported = false

func preemptM(mp *m) {
	// No threads, so nothing to do.
}
