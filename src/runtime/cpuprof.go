// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CPU profiling.
//
// The signal handler for the profiling clock tick adds a new stack trace
// to a log of recent traces. The log is read by a user goroutine that
// turns it into formatted profile data. If the reader does not keep up
// with the log, those writes will be recorded as a count of lost records.
// The actual profile buffer is in profbuf.go.

package runtime

import (
	"internal/abi"
	"runtime/internal/sys"
	"unsafe"
)

const (
	maxCPUProfStack = 64

	// profBufWordCount is the size of the CPU profile buffer's storage for the
	// header and stack of each sample, measured in 64-bit words. Every sample
	// has a required header of two words. With a small additional header (a
	// word or two) and stacks at the profiler's maximum length of 64 frames,
	// that capacity can support 1900 samples or 19 thread-seconds at a 100 Hz
	// sample rate, at a cost of 1 MiB.
	profBufWordCount = 1 << 17
	// profBufTagCount is the size of the CPU profile buffer's storage for the
	// goroutine tags associated with each sample. A capacity of 1<<14 means
	// room for 16k samples, or 160 thread-seconds at a 100 Hz sample rate.
	profBufTagCount = 1 << 14
)

type cpuProfile struct {
	lock mutex
	on   bool     // profiling is on
	log  *profBuf // profile events written here

	// extra holds extra stacks accumulated in addNonGo
	// corresponding to profiling signals arriving on
	// non-Go-created threads. Those stacks are written
	// to log the next time a normal Go thread gets the
	// signal handler.
	// Assuming the stacks are 2 words each (we don't get
	// a full traceback from those threads), plus one word
	// size for framing, 100 Hz profiling would generate
	// 300 words per second.
	// Hopefully a normal Go thread will get the profiling
	// signal at least once every few seconds.
	extra      [1000]uintptr
	numExtra   int
	lostExtra  uint64 // count of frames lost because extra is full
	lostAtomic uint64 // count of frames lost because of being in atomic64 on mips/arm; updated racily
}

var cpuprof cpuProfile

// SetCPUProfileRate sets the CPU profiling rate to hz samples per second.
// If hz <= 0, SetCPUProfileRate turns off profiling.
// If the profiler is on, the rate cannot be changed without first turning it off.
//
// Most clients should use the [runtime/pprof] package or
// the [testing] package's -test.cpuprofile flag instead of calling
// SetCPUProfileRate directly.
func SetCPUProfileRate(hz int) {
	// Clamp hz to something reasonable.
	if hz < 0 {
		hz = 0
	}
	if hz > 1000000 {
		hz = 1000000
	}

	lock(&cpuprof.lock)
	if hz > 0 {
		if cpuprof.on || cpuprof.log != nil {
			print("runtime: cannot set cpu profile rate until previous profile has finished.\n")
			unlock(&cpuprof.lock)
			return
		}

		cpuprof.on = true
		cpuprof.log = newProfBuf(1, profBufWordCount, profBufTagCount)
		hdr := [1]uint64{uint64(hz)}
		cpuprof.log.write(nil, nanotime(), hdr[:], nil)
		setcpuprofilerate(int32(hz))
	} else if cpuprof.on {
		setcpuprofilerate(0)
		cpuprof.on = false
		cpuprof.addExtra()
		cpuprof.log.close()
	}
	unlock(&cpuprof.lock)
}

// add adds the stack trace to the profile.
// It is called from signal handlers and other limited environments
// and cannot allocate memory or acquire locks that might be
// held at the time of the signal, nor can it use substantial amounts
// of stack.
//
//go:nowritebarrierrec
func (p *cpuProfile) add(tagPtr *unsafe.Pointer, stk []uintptr) {
	// Simple cas-lock to coordinate with setcpuprofilerate.
	for !prof.signalLock.CompareAndSwap(0, 1) {
		// TODO: Is it safe to osyield here? https://go.dev/issue/52672
		osyield()
	}

	if prof.hz.Load() != 0 { // implies cpuprof.log != nil
		if p.numExtra > 0 || p.lostExtra > 0 || p.lostAtomic > 0 {
			p.addExtra()
		}
		hdr := [1]uint64{1}
		// Note: write "knows" that the argument is &gp.labels,
		// because otherwise its write barrier behavior may not
		// be correct. See the long comment there before
		// changing the argument here.
		cpuprof.log.write(tagPtr, nanotime(), hdr[:], stk)
	}

	prof.signalLock.Store(0)
}

// addNonGo adds the non-Go stack trace to the profile.
// It is called from a non-Go thread, so we cannot use much stack at all,
// nor do anything that needs a g or an m.
// In particular, we can't call cpuprof.log.write.
// Instead, we copy the stack into cpuprof.extra,
// which will be drained the next time a Go thread
// gets the signal handling event.
//
//go:nosplit
//go:nowritebarrierrec
func (p *cpuProfile) addNonGo(stk []uintptr) {
	// Simple cas-lock to coordinate with SetCPUProfileRate.
	// (Other calls to add or addNonGo should be blocked out
	// by the fact that only one SIGPROF can be handled by the
	// process at a time. If not, this lock will serialize those too.
	// The use of timer_create(2) on Linux to request process-targeted
	// signals may have changed this.)
	for !prof.signalLock.CompareAndSwap(0, 1) {
		// TODO: Is it safe to osyield here? https://go.dev/issue/52672
		osyield()
	}

	if cpuprof.numExtra+1+len(stk) < len(cpuprof.extra) {
		i := cpuprof.numExtra
		cpuprof.extra[i] = uintptr(1 + len(stk))
		copy(cpuprof.extra[i+1:], stk)
		cpuprof.numExtra += 1 + len(stk)
	} else {
		cpuprof.lostExtra++
	}

	prof.signalLock.Store(0)
}

// addExtra adds the "extra" profiling events,
// queued by addNonGo, to the profile log.
// addExtra is called either from a signal handler on a Go thread
// or from an ordinary goroutine; either way it can use stack
// and has a g. The world may be stopped, though.
func (p *cpuProfile) addExtra() {
	// Copy accumulated non-Go profile events.
	hdr := [1]uint64{1}
	for i := 0; i < p.numExtra; {
		p.log.write(nil, 0, hdr[:], p.extra[i+1:i+int(p.extra[i])])
		i += int(p.extra[i])
	}
	p.numExtra = 0

	// Report any lost events.
	if p.lostExtra > 0 {
		hdr := [1]uint64{p.lostExtra}
		lostStk := [2]uintptr{
			abi.FuncPCABIInternal(_LostExternalCode) + sys.PCQuantum,
			abi.FuncPCABIInternal(_ExternalCode) + sys.PCQuantum,
		}
		p.log.write(nil, 0, hdr[:], lostStk[:])
		p.lostExtra = 0
	}

	if p.lostAtomic > 0 {
		hdr := [1]uint64{p.lostAtomic}
		lostStk := [2]uintptr{
			abi.FuncPCABIInternal(_LostSIGPROFDuringAtomic64) + sys.PCQuantum,
			abi.FuncPCABIInternal(_System) + sys.PCQuantum,
		}
		p.log.write(nil, 0, hdr[:], lostStk[:])
		p.lostAtomic = 0
	}

}

// CPUProfile panics.
// It formerly provided raw access to chunks of
// a pprof-format profile generated by the runtime.
// The details of generating that format have changed,
// so this functionality has been removed.
//
// Deprecated: Use the [runtime/pprof] package,
// or the handlers in the [net/http/pprof] package,
// or the [testing] package's -test.cpuprofile flag instead.
func CPUProfile() []byte {
	panic("CPUProfile no longer available")
}

//go:linkname runtime_pprof_runtime_cyclesPerSecond runtime/pprof.runtime_cyclesPerSecond
func runtime_pprof_runtime_cyclesPerSecond() int64 {
	return ticksPerSecond()
}

// readProfile, provided to runtime/pprof, returns the next chunk of
// binary CPU profiling stack trace data, blocking until data is available.
// If profiling is turned off and all the profile data accumulated while it was
// on has been returned, readProfile returns eof=true.
// The caller must save the returned data and tags before calling readProfile again.
// The returned data contains a whole number of records, and tags contains
// exactly one entry per record.
//
//go:linkname runtime_pprof_readProfile runtime/pprof.readProfile
func runtime_pprof_readProfile() ([]uint64, []unsafe.Pointer, bool) {
	lock(&cpuprof.lock)
	log := cpuprof.log
	unlock(&cpuprof.lock)
	readMode := profBufBlocking
	if GOOS == "darwin" || GOOS == "ios" {
		readMode = profBufNonBlocking // For #61768; on Darwin notes are not async-signal-safe.  See sigNoteSetup in os_darwin.go.
	}
	data, tags, eof := log.read(readMode)
	if len(data) == 0 && eof {
		lock(&cpuprof.lock)
		cpuprof.log = nil
		unlock(&cpuprof.lock)
	}
	return data, tags, eof
}
