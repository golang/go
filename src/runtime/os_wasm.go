// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

func osinit() {
	// https://webassembly.github.io/spec/core/exec/runtime.html#memory-instances
	physPageSize = 64 * 1024
	initBloc()
	blocMax = uintptr(currentMemory()) * physPageSize // record the initial linear memory size
	ncpu = 1
	getg().m.procid = 2
}

const _SIGSEGV = 0xb

func sigpanic() {
	gp := getg()
	if !canpanic() {
		throw("unexpected signal during runtime execution")
	}

	// js only invokes the exception handler for memory faults.
	gp.sig = _SIGSEGV
	panicmem()
}

// func exitThread(wait *uint32)
// FIXME: wasm doesn't have atomic yet
func exitThread(wait *atomic.Uint32)

type mOS struct{}

func osyield()

//go:nosplit
func osyield_no_g() {
	osyield()
}

type sigset struct{}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024)
	mp.gsignal.m = mp
}

//go:nosplit
func usleep_no_g(usec uint32) {
	usleep(usec)
}

//go:nosplit
func sigsave(p *sigset) {
}

//go:nosplit
func msigrestore(sigmask sigset) {
}

//go:nosplit
//go:nowritebarrierrec
func clearSignalHandlers() {
}

//go:nosplit
func sigblock(exiting bool) {
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
}

// Called from dropm to undo the effect of an minit.
func unminit() {
}

// Called from exitm, but not from drop, to undo the effect of thread-owned
// resources in minit, semacreate, or elsewhere. Do not take locks after calling this.
func mdestroy(mp *m) {
}

// wasm has no signals
const _NSIG = 0

func signame(sig uint32) string {
	return ""
}

func crash() {
	abort()
}

func initsig(preinit bool) {
}

// May run with m.p==nil, so write barriers are not allowed.
//
//go:nowritebarrier
func newosproc(mp *m) {
	throw("newosproc: not implemented")
}

// Do nothing on WASM platform, always return EPIPE to caller.
//
//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {}

//go:linkname syscall_now syscall.now
func syscall_now() (sec int64, nsec int32) {
	sec, nsec, _ = time_now()
	return
}

//go:nosplit
func cputicks() int64 {
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}

// gsignalStack is unused on js.
type gsignalStack struct{}

const preemptMSupported = false

func preemptM(mp *m) {
	// No threads, so nothing to do.
}

// getfp returns the frame pointer register of its caller or 0 if not implemented.
// TODO: Make this a compiler intrinsic
func getfp() uintptr { return 0 }

func setProcessCPUProfiler(hz int32) {}
func setThreadCPUProfiler(hz int32)  {}
func sigdisable(uint32)              {}
func sigenable(uint32)               {}
func sigignore(uint32)               {}

// Stubs so tests can link correctly. These should never be called.
func open(name *byte, mode, perm int32) int32        { panic("not implemented") }
func closefd(fd int32) int32                         { panic("not implemented") }
func read(fd int32, p unsafe.Pointer, n int32) int32 { panic("not implemented") }
