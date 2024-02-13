// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.exectracer2

// Go execution tracer.
// The tracer captures a wide range of execution events like goroutine
// creation/blocking/unblocking, syscall enter/exit/block, GC-related events,
// changes of heap size, processor start/stop, etc and writes them to a buffer
// in a compact form. A precise nanosecond-precision timestamp and a stack
// trace is captured for most events.
// See https://golang.org/s/go15trace for more info.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/goos"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Event types in the trace, args are given in square brackets.
const (
	traceEvNone              = 0  // unused
	traceEvBatch             = 1  // start of per-P batch of events [pid, timestamp]
	traceEvFrequency         = 2  // contains tracer timer frequency [frequency (ticks per second)]
	traceEvStack             = 3  // stack [stack id, number of PCs, array of {PC, func string ID, file string ID, line}]
	traceEvGomaxprocs        = 4  // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack id]
	traceEvProcStart         = 5  // start of P [timestamp, thread id]
	traceEvProcStop          = 6  // stop of P [timestamp]
	traceEvGCStart           = 7  // GC start [timestamp, seq, stack id]
	traceEvGCDone            = 8  // GC done [timestamp]
	traceEvSTWStart          = 9  // STW start [timestamp, kind]
	traceEvSTWDone           = 10 // STW done [timestamp]
	traceEvGCSweepStart      = 11 // GC sweep start [timestamp, stack id]
	traceEvGCSweepDone       = 12 // GC sweep done [timestamp, swept, reclaimed]
	traceEvGoCreate          = 13 // goroutine creation [timestamp, new goroutine id, new stack id, stack id]
	traceEvGoStart           = 14 // goroutine starts running [timestamp, goroutine id, seq]
	traceEvGoEnd             = 15 // goroutine ends [timestamp]
	traceEvGoStop            = 16 // goroutine stops (like in select{}) [timestamp, stack]
	traceEvGoSched           = 17 // goroutine calls Gosched [timestamp, stack]
	traceEvGoPreempt         = 18 // goroutine is preempted [timestamp, stack]
	traceEvGoSleep           = 19 // goroutine calls Sleep [timestamp, stack]
	traceEvGoBlock           = 20 // goroutine blocks [timestamp, stack]
	traceEvGoUnblock         = 21 // goroutine is unblocked [timestamp, goroutine id, seq, stack]
	traceEvGoBlockSend       = 22 // goroutine blocks on chan send [timestamp, stack]
	traceEvGoBlockRecv       = 23 // goroutine blocks on chan recv [timestamp, stack]
	traceEvGoBlockSelect     = 24 // goroutine blocks on select [timestamp, stack]
	traceEvGoBlockSync       = 25 // goroutine blocks on Mutex/RWMutex [timestamp, stack]
	traceEvGoBlockCond       = 26 // goroutine blocks on Cond [timestamp, stack]
	traceEvGoBlockNet        = 27 // goroutine blocks on network [timestamp, stack]
	traceEvGoSysCall         = 28 // syscall enter [timestamp, stack]
	traceEvGoSysExit         = 29 // syscall exit [timestamp, goroutine id, seq, real timestamp]
	traceEvGoSysBlock        = 30 // syscall blocks [timestamp]
	traceEvGoWaiting         = 31 // denotes that goroutine is blocked when tracing starts [timestamp, goroutine id]
	traceEvGoInSyscall       = 32 // denotes that goroutine is in syscall when tracing starts [timestamp, goroutine id]
	traceEvHeapAlloc         = 33 // gcController.heapLive change [timestamp, heap_alloc]
	traceEvHeapGoal          = 34 // gcController.heapGoal() (formerly next_gc) change [timestamp, heap goal in bytes]
	traceEvTimerGoroutine    = 35 // not currently used; previously denoted timer goroutine [timer goroutine id]
	traceEvFutileWakeup      = 36 // not currently used; denotes that the previous wakeup of this goroutine was futile [timestamp]
	traceEvString            = 37 // string dictionary entry [ID, length, string]
	traceEvGoStartLocal      = 38 // goroutine starts running on the same P as the last event [timestamp, goroutine id]
	traceEvGoUnblockLocal    = 39 // goroutine is unblocked on the same P as the last event [timestamp, goroutine id, stack]
	traceEvGoSysExitLocal    = 40 // syscall exit on the same P as the last event [timestamp, goroutine id, real timestamp]
	traceEvGoStartLabel      = 41 // goroutine starts running with label [timestamp, goroutine id, seq, label string id]
	traceEvGoBlockGC         = 42 // goroutine blocks on GC assist [timestamp, stack]
	traceEvGCMarkAssistStart = 43 // GC mark assist start [timestamp, stack]
	traceEvGCMarkAssistDone  = 44 // GC mark assist done [timestamp]
	traceEvUserTaskCreate    = 45 // trace.NewTask [timestamp, internal task id, internal parent task id, name string, stack]
	traceEvUserTaskEnd       = 46 // end of a task [timestamp, internal task id, stack]
	traceEvUserRegion        = 47 // trace.WithRegion [timestamp, internal task id, mode(0:start, 1:end), name string, stack]
	traceEvUserLog           = 48 // trace.Log [timestamp, internal task id, key string id, stack, value string]
	traceEvCPUSample         = 49 // CPU profiling sample [timestamp, real timestamp, real P id (-1 when absent), goroutine id, stack]
	traceEvCount             = 50
	// Byte is used but only 6 bits are available for event type.
	// The remaining 2 bits are used to specify the number of arguments.
	// That means, the max event type value is 63.
)

// traceBlockReason is an enumeration of reasons a goroutine might block.
// This is the interface the rest of the runtime uses to tell the
// tracer why a goroutine blocked. The tracer then propagates this information
// into the trace however it sees fit.
//
// Note that traceBlockReasons should not be compared, since reasons that are
// distinct by name may *not* be distinct by value.
type traceBlockReason uint8

// For maximal efficiency, just map the trace block reason directly to a trace
// event.
const (
	traceBlockGeneric         traceBlockReason = traceEvGoBlock
	traceBlockForever                          = traceEvGoStop
	traceBlockNet                              = traceEvGoBlockNet
	traceBlockSelect                           = traceEvGoBlockSelect
	traceBlockCondWait                         = traceEvGoBlockCond
	traceBlockSync                             = traceEvGoBlockSync
	traceBlockChanSend                         = traceEvGoBlockSend
	traceBlockChanRecv                         = traceEvGoBlockRecv
	traceBlockGCMarkAssist                     = traceEvGoBlockGC
	traceBlockGCSweep                          = traceEvGoBlock
	traceBlockSystemGoroutine                  = traceEvGoBlock
	traceBlockPreempted                        = traceEvGoBlock
	traceBlockDebugCall                        = traceEvGoBlock
	traceBlockUntilGCEnds                      = traceEvGoBlock
	traceBlockSleep                            = traceEvGoSleep
)

const (
	// Timestamps in trace are cputicks/traceTickDiv.
	// This makes absolute values of timestamp diffs smaller,
	// and so they are encoded in less number of bytes.
	// 64 on x86 is somewhat arbitrary (one tick is ~20ns on a 3GHz machine).
	// The suggested increment frequency for PowerPC's time base register is
	// 512 MHz according to Power ISA v2.07 section 6.2, so we use 16 on ppc64
	// and ppc64le.
	traceTimeDiv = 16 + 48*(goarch.Is386|goarch.IsAmd64)
	// Maximum number of PCs in a single stack trace.
	// Since events contain only stack id rather than whole stack trace,
	// we can allow quite large values here.
	traceStackSize = 128
	// Identifier of a fake P that is used when we trace without a real P.
	traceGlobProc = -1
	// Maximum number of bytes to encode uint64 in base-128.
	traceBytesPerNumber = 10
	// Shift of the number of arguments in the first event byte.
	traceArgCountShift = 6
)

// trace is global tracing context.
var trace struct {
	// trace.lock must only be acquired on the system stack where
	// stack splits cannot happen while it is held.
	lock          mutex       // protects the following members
	enabled       bool        // when set runtime traces events
	shutdown      bool        // set when we are waiting for trace reader to finish after setting enabled to false
	headerWritten bool        // whether ReadTrace has emitted trace header
	footerWritten bool        // whether ReadTrace has emitted trace footer
	shutdownSema  uint32      // used to wait for ReadTrace completion
	seqStart      uint64      // sequence number when tracing was started
	startTicks    int64       // cputicks when tracing was started
	endTicks      int64       // cputicks when tracing was stopped
	startNanotime int64       // nanotime when tracing was started
	endNanotime   int64       // nanotime when tracing was stopped
	startTime     traceTime   // traceClockNow when tracing started
	endTime       traceTime   // traceClockNow when tracing stopped
	seqGC         uint64      // GC start/done sequencer
	reading       traceBufPtr // buffer currently handed off to user
	empty         traceBufPtr // stack of empty buffers
	fullHead      traceBufPtr // queue of full buffers
	fullTail      traceBufPtr
	stackTab      traceStackTable // maps stack traces to unique ids
	// cpuLogRead accepts CPU profile samples from the signal handler where
	// they're generated. It uses a two-word header to hold the IDs of the P and
	// G (respectively) that were active at the time of the sample. Because
	// profBuf uses a record with all zeros in its header to indicate overflow,
	// we make sure to make the P field always non-zero: The ID of a real P will
	// start at bit 1, and bit 0 will be set. Samples that arrive while no P is
	// running (such as near syscalls) will set the first header field to 0b10.
	// This careful handling of the first header field allows us to store ID of
	// the active G directly in the second field, even though that will be 0
	// when sampling g0.
	cpuLogRead *profBuf
	// cpuLogBuf is a trace buffer to hold events corresponding to CPU profile
	// samples, which arrive out of band and not directly connected to a
	// specific P.
	cpuLogBuf traceBufPtr

	reader atomic.Pointer[g] // goroutine that called ReadTrace, or nil

	signalLock  atomic.Uint32 // protects use of the following member, only usable in signal handlers
	cpuLogWrite *profBuf      // copy of cpuLogRead for use in signal handlers, set without signalLock

	// Dictionary for traceEvString.
	//
	// TODO: central lock to access the map is not ideal.
	//   option: pre-assign ids to all user annotation region names and tags
	//   option: per-P cache
	//   option: sync.Map like data structure
	stringsLock mutex
	strings     map[string]uint64
	stringSeq   uint64

	// markWorkerLabels maps gcMarkWorkerMode to string ID.
	markWorkerLabels [len(gcMarkWorkerModeStrings)]uint64

	bufLock mutex       // protects buf
	buf     traceBufPtr // global trace buffer, used when running without a p
}

// gTraceState is per-G state for the tracer.
type gTraceState struct {
	sysExitTime        traceTime // timestamp when syscall has returned
	tracedSyscallEnter bool      // syscall or cgo was entered while trace was enabled or StartTrace has emitted EvGoInSyscall about this goroutine
	seq                uint64    // trace event sequencer
	lastP              puintptr  // last P emitted an event for this goroutine
}

// Unused; for compatibility with the new tracer.
func (s *gTraceState) reset() {}

// mTraceState is per-M state for the tracer.
type mTraceState struct {
	startingTrace  bool // this M is in TraceStart, potentially before traceEnabled is true
	tracedSTWStart bool // this M traced a STW start, so it should trace an end
}

// pTraceState is per-P state for the tracer.
type pTraceState struct {
	buf traceBufPtr

	// inSweep indicates the sweep events should be traced.
	// This is used to defer the sweep start event until a span
	// has actually been swept.
	inSweep bool

	// swept and reclaimed track the number of bytes swept and reclaimed
	// by sweeping in the current sweep loop (while inSweep was true).
	swept, reclaimed uintptr
}

// traceLockInit initializes global trace locks.
func traceLockInit() {
	lockInit(&trace.bufLock, lockRankTraceBuf)
	lockInit(&trace.stringsLock, lockRankTraceStrings)
	lockInit(&trace.lock, lockRankTrace)
	lockInit(&trace.stackTab.lock, lockRankTraceStackTab)
}

// traceBufHeader is per-P tracing buffer.
type traceBufHeader struct {
	link     traceBufPtr             // in trace.empty/full
	lastTime traceTime               // when we wrote the last event
	pos      int                     // next write offset in arr
	stk      [traceStackSize]uintptr // scratch buffer for traceback
}

// traceBuf is per-P tracing buffer.
type traceBuf struct {
	_ sys.NotInHeap
	traceBufHeader
	arr [64<<10 - unsafe.Sizeof(traceBufHeader{})]byte // underlying buffer for traceBufHeader.buf
}

// traceBufPtr is a *traceBuf that is not traced by the garbage
// collector and doesn't have write barriers. traceBufs are not
// allocated from the GC'd heap, so this is safe, and are often
// manipulated in contexts where write barriers are not allowed, so
// this is necessary.
//
// TODO: Since traceBuf is now embedded runtime/internal/sys.NotInHeap, this isn't necessary.
type traceBufPtr uintptr

func (tp traceBufPtr) ptr() *traceBuf   { return (*traceBuf)(unsafe.Pointer(tp)) }
func (tp *traceBufPtr) set(b *traceBuf) { *tp = traceBufPtr(unsafe.Pointer(b)) }
func traceBufPtrOf(b *traceBuf) traceBufPtr {
	return traceBufPtr(unsafe.Pointer(b))
}

// traceEnabled returns true if the trace is currently enabled.
//
// nosplit because it's called on the syscall path when stack movement is forbidden.
//
//go:nosplit
func traceEnabled() bool {
	return trace.enabled
}

// traceShuttingDown returns true if the trace is currently shutting down.
//
//go:nosplit
func traceShuttingDown() bool {
	return trace.shutdown
}

// traceLocker represents an M writing trace events. While a traceLocker value
// is valid, the tracer observes all operations on the G/M/P or trace events being
// written as happening atomically.
//
// This doesn't do much for the current tracer, because the current tracer doesn't
// need atomicity around non-trace runtime operations. All the state it needs it
// collects carefully during a STW.
type traceLocker struct {
	enabled bool
}

// traceAcquire prepares this M for writing one or more trace events.
//
// This exists for compatibility with the upcoming new tracer; it doesn't do much
// in the current tracer.
//
// nosplit because it's called on the syscall path when stack movement is forbidden.
//
//go:nosplit
func traceAcquire() traceLocker {
	if !traceEnabled() {
		return traceLocker{false}
	}
	return traceLocker{true}
}

// ok returns true if the traceLocker is valid (i.e. tracing is enabled).
//
// nosplit because it's called on the syscall path when stack movement is forbidden.
//
//go:nosplit
func (tl traceLocker) ok() bool {
	return tl.enabled
}

// traceRelease indicates that this M is done writing trace events.
//
// This exists for compatibility with the upcoming new tracer; it doesn't do anything
// in the current tracer.
//
// nosplit because it's called on the syscall path when stack movement is forbidden.
//
//go:nosplit
func traceRelease(tl traceLocker) {
}

// StartTrace enables tracing for the current process.
// While tracing, the data will be buffered and available via [ReadTrace].
// StartTrace returns an error if tracing is already enabled.
// Most clients should use the [runtime/trace] package or the [testing] package's
// -test.trace flag instead of calling StartTrace directly.
func StartTrace() error {
	// Stop the world so that we can take a consistent snapshot
	// of all goroutines at the beginning of the trace.
	// Do not stop the world during GC so we ensure we always see
	// a consistent view of GC-related events (e.g. a start is always
	// paired with an end).
	stw := stopTheWorldGC(stwStartTrace)

	// Prevent sysmon from running any code that could generate events.
	lock(&sched.sysmonlock)

	// We are in stop-the-world, but syscalls can finish and write to trace concurrently.
	// Exitsyscall could check trace.enabled long before and then suddenly wake up
	// and decide to write to trace at a random point in time.
	// However, such syscall will use the global trace.buf buffer, because we've
	// acquired all p's by doing stop-the-world. So this protects us from such races.
	lock(&trace.bufLock)

	if trace.enabled || trace.shutdown {
		unlock(&trace.bufLock)
		unlock(&sched.sysmonlock)
		startTheWorldGC(stw)
		return errorString("tracing is already enabled")
	}

	// Can't set trace.enabled yet. While the world is stopped, exitsyscall could
	// already emit a delayed event (see exitTicks in exitsyscall) if we set trace.enabled here.
	// That would lead to an inconsistent trace:
	// - either GoSysExit appears before EvGoInSyscall,
	// - or GoSysExit appears for a goroutine for which we don't emit EvGoInSyscall below.
	// To instruct traceEvent that it must not ignore events below, we set trace.startingTrace.
	// trace.enabled is set afterwards once we have emitted all preliminary events.
	mp := getg().m
	mp.trace.startingTrace = true

	// Obtain current stack ID to use in all traceEvGoCreate events below.
	stkBuf := make([]uintptr, traceStackSize)
	stackID := traceStackID(mp, stkBuf, 2)

	profBuf := newProfBuf(2, profBufWordCount, profBufTagCount) // after the timestamp, header is [pp.id, gp.goid]
	trace.cpuLogRead = profBuf

	// We must not acquire trace.signalLock outside of a signal handler: a
	// profiling signal may arrive at any time and try to acquire it, leading to
	// deadlock. Because we can't use that lock to protect updates to
	// trace.cpuLogWrite (only use of the structure it references), reads and
	// writes of the pointer must be atomic. (And although this field is never
	// the sole pointer to the profBuf value, it's best to allow a write barrier
	// here.)
	atomicstorep(unsafe.Pointer(&trace.cpuLogWrite), unsafe.Pointer(profBuf))

	// World is stopped, no need to lock.
	forEachGRace(func(gp *g) {
		status := readgstatus(gp)
		if status != _Gdead {
			gp.trace.seq = 0
			gp.trace.lastP = getg().m.p
			// +PCQuantum because traceFrameForPC expects return PCs and subtracts PCQuantum.
			id := trace.stackTab.put([]uintptr{logicalStackSentinel, startPCforTrace(gp.startpc) + sys.PCQuantum})
			traceEvent(traceEvGoCreate, -1, gp.goid, uint64(id), stackID)
		}
		if status == _Gwaiting {
			// traceEvGoWaiting is implied to have seq=1.
			gp.trace.seq++
			traceEvent(traceEvGoWaiting, -1, gp.goid)
		}
		if status == _Gsyscall {
			gp.trace.seq++
			gp.trace.tracedSyscallEnter = true
			traceEvent(traceEvGoInSyscall, -1, gp.goid)
		} else if status == _Gdead && gp.m != nil && gp.m.isextra {
			// Trigger two trace events for the dead g in the extra m,
			// since the next event of the g will be traceEvGoSysExit in exitsyscall,
			// while calling from C thread to Go.
			gp.trace.seq = 0
			gp.trace.lastP = getg().m.p
			// +PCQuantum because traceFrameForPC expects return PCs and subtracts PCQuantum.
			id := trace.stackTab.put([]uintptr{logicalStackSentinel, startPCforTrace(0) + sys.PCQuantum}) // no start pc
			traceEvent(traceEvGoCreate, -1, gp.goid, uint64(id), stackID)
			gp.trace.seq++
			gp.trace.tracedSyscallEnter = true
			traceEvent(traceEvGoInSyscall, -1, gp.goid)
		} else {
			// We need to explicitly clear the flag. A previous trace might have ended with a goroutine
			// not emitting a GoSysExit and clearing the flag, leaving it in a stale state. Clearing
			// it here makes it unambiguous to any goroutine exiting a syscall racing with us that
			// no EvGoInSyscall event was emitted for it. (It's not racy to set this flag here, because
			// it'll only get checked when the goroutine runs again, which will be after the world starts
			// again.)
			gp.trace.tracedSyscallEnter = false
		}
	})
	// Use a dummy traceLocker. The trace isn't enabled yet, but we can still write events.
	tl := traceLocker{}
	tl.ProcStart()
	tl.GoStart()
	// Note: startTicks needs to be set after we emit traceEvGoInSyscall events.
	// If we do it the other way around, it is possible that exitsyscall will
	// query sysExitTime after startTicks but before traceEvGoInSyscall timestamp.
	// It will lead to a false conclusion that cputicks is broken.
	trace.startTime = traceClockNow()
	trace.startTicks = cputicks()
	trace.startNanotime = nanotime()
	trace.headerWritten = false
	trace.footerWritten = false

	// string to id mapping
	//  0 : reserved for an empty string
	//  remaining: other strings registered by traceString
	trace.stringSeq = 0
	trace.strings = make(map[string]uint64)

	trace.seqGC = 0
	mp.trace.startingTrace = false
	trace.enabled = true

	// Register runtime goroutine labels.
	_, pid, bufp := traceAcquireBuffer()
	for i, label := range gcMarkWorkerModeStrings[:] {
		trace.markWorkerLabels[i], bufp = traceString(bufp, pid, label)
	}
	traceReleaseBuffer(mp, pid)

	unlock(&trace.bufLock)

	unlock(&sched.sysmonlock)

	// Record the current state of HeapGoal to avoid information loss in trace.
	//
	// Use the same dummy trace locker. The trace can't end until after we start
	// the world, and we can safely trace from here.
	tl.HeapGoal()

	startTheWorldGC(stw)
	return nil
}

// StopTrace stops tracing, if it was previously enabled.
// StopTrace only returns after all the reads for the trace have completed.
func StopTrace() {
	// Stop the world so that we can collect the trace buffers from all p's below,
	// and also to avoid races with traceEvent.
	stw := stopTheWorldGC(stwStopTrace)

	// See the comment in StartTrace.
	lock(&sched.sysmonlock)

	// See the comment in StartTrace.
	lock(&trace.bufLock)

	if !trace.enabled {
		unlock(&trace.bufLock)
		unlock(&sched.sysmonlock)
		startTheWorldGC(stw)
		return
	}

	// Trace GoSched for us, and use a dummy locker. The world is stopped
	// and we control whether the trace is enabled, so this is safe.
	tl := traceLocker{}
	tl.GoSched()

	atomicstorep(unsafe.Pointer(&trace.cpuLogWrite), nil)
	trace.cpuLogRead.close()
	traceReadCPU()

	// Loop over all allocated Ps because dead Ps may still have
	// trace buffers.
	for _, p := range allp[:cap(allp)] {
		buf := p.trace.buf
		if buf != 0 {
			traceFullQueue(buf)
			p.trace.buf = 0
		}
	}
	if trace.buf != 0 {
		buf := trace.buf
		trace.buf = 0
		if buf.ptr().pos != 0 {
			traceFullQueue(buf)
		}
	}
	if trace.cpuLogBuf != 0 {
		buf := trace.cpuLogBuf
		trace.cpuLogBuf = 0
		if buf.ptr().pos != 0 {
			traceFullQueue(buf)
		}
	}

	// Wait for startNanotime != endNanotime. On Windows the default interval between
	// system clock ticks is typically between 1 and 15 milliseconds, which may not
	// have passed since the trace started. Without nanotime moving forward, trace
	// tooling has no way of identifying how much real time each cputicks time deltas
	// represent.
	for {
		trace.endTime = traceClockNow()
		trace.endTicks = cputicks()
		trace.endNanotime = nanotime()

		if trace.endNanotime != trace.startNanotime || faketime != 0 {
			break
		}
		osyield()
	}

	trace.enabled = false
	trace.shutdown = true
	unlock(&trace.bufLock)

	unlock(&sched.sysmonlock)

	startTheWorldGC(stw)

	// The world is started but we've set trace.shutdown, so new tracing can't start.
	// Wait for the trace reader to flush pending buffers and stop.
	semacquire(&trace.shutdownSema)
	if raceenabled {
		raceacquire(unsafe.Pointer(&trace.shutdownSema))
	}

	systemstack(func() {
		// The lock protects us from races with StartTrace/StopTrace because they do stop-the-world.
		lock(&trace.lock)
		for _, p := range allp[:cap(allp)] {
			if p.trace.buf != 0 {
				throw("trace: non-empty trace buffer in proc")
			}
		}
		if trace.buf != 0 {
			throw("trace: non-empty global trace buffer")
		}
		if trace.fullHead != 0 || trace.fullTail != 0 {
			throw("trace: non-empty full trace buffer")
		}
		if trace.reading != 0 || trace.reader.Load() != nil {
			throw("trace: reading after shutdown")
		}
		for trace.empty != 0 {
			buf := trace.empty
			trace.empty = buf.ptr().link
			sysFree(unsafe.Pointer(buf), unsafe.Sizeof(*buf.ptr()), &memstats.other_sys)
		}
		trace.strings = nil
		trace.shutdown = false
		trace.cpuLogRead = nil
		unlock(&trace.lock)
	})
}

// traceAdvance is called from panic, it does nothing for the legacy tracer.
func traceAdvance(stopTrace bool) {}

// ReadTrace returns the next chunk of binary tracing data, blocking until data
// is available. If tracing is turned off and all the data accumulated while it
// was on has been returned, ReadTrace returns nil. The caller must copy the
// returned data before calling ReadTrace again.
// ReadTrace must be called from one goroutine at a time.
func ReadTrace() []byte {
top:
	var buf []byte
	var park bool
	systemstack(func() {
		buf, park = readTrace0()
	})
	if park {
		gopark(func(gp *g, _ unsafe.Pointer) bool {
			if !trace.reader.CompareAndSwapNoWB(nil, gp) {
				// We're racing with another reader.
				// Wake up and handle this case.
				return false
			}

			if g2 := traceReader(); gp == g2 {
				// New data arrived between unlocking
				// and the CAS and we won the wake-up
				// race, so wake up directly.
				return false
			} else if g2 != nil {
				printlock()
				println("runtime: got trace reader", g2, g2.goid)
				throw("unexpected trace reader")
			}

			return true
		}, nil, waitReasonTraceReaderBlocked, traceBlockSystemGoroutine, 2)
		goto top
	}

	return buf
}

// readTrace0 is ReadTrace's continuation on g0. This must run on the
// system stack because it acquires trace.lock.
//
//go:systemstack
func readTrace0() (buf []byte, park bool) {
	if raceenabled {
		// g0 doesn't have a race context. Borrow the user G's.
		if getg().racectx != 0 {
			throw("expected racectx == 0")
		}
		getg().racectx = getg().m.curg.racectx
		// (This defer should get open-coded, which is safe on
		// the system stack.)
		defer func() { getg().racectx = 0 }()
	}

	// Optimistically look for CPU profile samples. This may write new stack
	// records, and may write new tracing buffers. This must be done with the
	// trace lock not held. footerWritten and shutdown are safe to access
	// here. They are only mutated by this goroutine or during a STW.
	if !trace.footerWritten && !trace.shutdown {
		traceReadCPU()
	}

	// This function must not allocate while holding trace.lock:
	// allocation can call heap allocate, which will try to emit a trace
	// event while holding heap lock.
	lock(&trace.lock)

	if trace.reader.Load() != nil {
		// More than one goroutine reads trace. This is bad.
		// But we rather do not crash the program because of tracing,
		// because tracing can be enabled at runtime on prod servers.
		unlock(&trace.lock)
		println("runtime: ReadTrace called from multiple goroutines simultaneously")
		return nil, false
	}
	// Recycle the old buffer.
	if buf := trace.reading; buf != 0 {
		buf.ptr().link = trace.empty
		trace.empty = buf
		trace.reading = 0
	}
	// Write trace header.
	if !trace.headerWritten {
		trace.headerWritten = true
		unlock(&trace.lock)
		return []byte("go 1.21 trace\x00\x00\x00"), false
	}
	// Wait for new data.
	if trace.fullHead == 0 && !trace.shutdown {
		// We don't simply use a note because the scheduler
		// executes this goroutine directly when it wakes up
		// (also a note would consume an M).
		unlock(&trace.lock)
		return nil, true
	}
newFull:
	assertLockHeld(&trace.lock)
	// Write a buffer.
	if trace.fullHead != 0 {
		buf := traceFullDequeue()
		trace.reading = buf
		unlock(&trace.lock)
		return buf.ptr().arr[:buf.ptr().pos], false
	}

	// Write footer with timer frequency.
	if !trace.footerWritten {
		trace.footerWritten = true
		freq := (float64(trace.endTicks-trace.startTicks) / traceTimeDiv) / (float64(trace.endNanotime-trace.startNanotime) / 1e9)
		if freq <= 0 {
			throw("trace: ReadTrace got invalid frequency")
		}
		unlock(&trace.lock)

		// Write frequency event.
		bufp := traceFlush(0, 0)
		buf := bufp.ptr()
		buf.byte(traceEvFrequency | 0<<traceArgCountShift)
		buf.varint(uint64(freq))

		// Dump stack table.
		// This will emit a bunch of full buffers, we will pick them up
		// on the next iteration.
		bufp = trace.stackTab.dump(bufp)

		// Flush final buffer.
		lock(&trace.lock)
		traceFullQueue(bufp)
		goto newFull // trace.lock should be held at newFull
	}
	// Done.
	if trace.shutdown {
		unlock(&trace.lock)
		if raceenabled {
			// Model synchronization on trace.shutdownSema, which race
			// detector does not see. This is required to avoid false
			// race reports on writer passed to trace.Start.
			racerelease(unsafe.Pointer(&trace.shutdownSema))
		}
		// trace.enabled is already reset, so can call traceable functions.
		semrelease(&trace.shutdownSema)
		return nil, false
	}
	// Also bad, but see the comment above.
	unlock(&trace.lock)
	println("runtime: spurious wakeup of trace reader")
	return nil, false
}

// traceReader returns the trace reader that should be woken up, if any.
// Callers should first check that trace.enabled or trace.shutdown is set.
//
// This must run on the system stack because it acquires trace.lock.
//
//go:systemstack
func traceReader() *g {
	// Optimistic check first
	if traceReaderAvailable() == nil {
		return nil
	}
	lock(&trace.lock)
	gp := traceReaderAvailable()
	if gp == nil || !trace.reader.CompareAndSwapNoWB(gp, nil) {
		unlock(&trace.lock)
		return nil
	}
	unlock(&trace.lock)
	return gp
}

// traceReaderAvailable returns the trace reader if it is not currently
// scheduled and should be. Callers should first check that trace.enabled
// or trace.shutdown is set.
func traceReaderAvailable() *g {
	if trace.fullHead != 0 || trace.shutdown {
		return trace.reader.Load()
	}
	return nil
}

// traceProcFree frees trace buffer associated with pp.
//
// This must run on the system stack because it acquires trace.lock.
//
//go:systemstack
func traceProcFree(pp *p) {
	buf := pp.trace.buf
	pp.trace.buf = 0
	if buf == 0 {
		return
	}
	lock(&trace.lock)
	traceFullQueue(buf)
	unlock(&trace.lock)
}

// ThreadDestroy is a no-op. It exists as a stub to support the new tracer.
//
// This must run on the system stack, just to match the new tracer.
func traceThreadDestroy(_ *m) {
	// No-op in old tracer.
}

// traceFullQueue queues buf into queue of full buffers.
func traceFullQueue(buf traceBufPtr) {
	buf.ptr().link = 0
	if trace.fullHead == 0 {
		trace.fullHead = buf
	} else {
		trace.fullTail.ptr().link = buf
	}
	trace.fullTail = buf
}

// traceFullDequeue dequeues from queue of full buffers.
func traceFullDequeue() traceBufPtr {
	buf := trace.fullHead
	if buf == 0 {
		return 0
	}
	trace.fullHead = buf.ptr().link
	if trace.fullHead == 0 {
		trace.fullTail = 0
	}
	buf.ptr().link = 0
	return buf
}

// traceEvent writes a single event to trace buffer, flushing the buffer if necessary.
// ev is event type.
// If skip > 0, write current stack id as the last argument (skipping skip top frames).
// If skip = 0, this event type should contain a stack, but we don't want
// to collect and remember it for this particular call.
func traceEvent(ev byte, skip int, args ...uint64) {
	mp, pid, bufp := traceAcquireBuffer()
	// Double-check trace.enabled now that we've done m.locks++ and acquired bufLock.
	// This protects from races between traceEvent and StartTrace/StopTrace.

	// The caller checked that trace.enabled == true, but trace.enabled might have been
	// turned off between the check and now. Check again. traceLockBuffer did mp.locks++,
	// StopTrace does stopTheWorld, and stopTheWorld waits for mp.locks to go back to zero,
	// so if we see trace.enabled == true now, we know it's true for the rest of the function.
	// Exitsyscall can run even during stopTheWorld. The race with StartTrace/StopTrace
	// during tracing in exitsyscall is resolved by locking trace.bufLock in traceLockBuffer.
	//
	// Note trace_userTaskCreate runs the same check.
	if !trace.enabled && !mp.trace.startingTrace {
		traceReleaseBuffer(mp, pid)
		return
	}

	if skip > 0 {
		if getg() == mp.curg {
			skip++ // +1 because stack is captured in traceEventLocked.
		}
	}
	traceEventLocked(0, mp, pid, bufp, ev, 0, skip, args...)
	traceReleaseBuffer(mp, pid)
}

// traceEventLocked writes a single event of type ev to the trace buffer bufp,
// flushing the buffer if necessary. pid is the id of the current P, or
// traceGlobProc if we're tracing without a real P.
//
// Preemption is disabled, and if running without a real P the global tracing
// buffer is locked.
//
// Events types that do not include a stack set skip to -1. Event types that
// include a stack may explicitly reference a stackID from the trace.stackTab
// (obtained by an earlier call to traceStackID). Without an explicit stackID,
// this function will automatically capture the stack of the goroutine currently
// running on mp, skipping skip top frames or, if skip is 0, writing out an
// empty stack record.
//
// It records the event's args to the traceBuf, and also makes an effort to
// reserve extraBytes bytes of additional space immediately following the event,
// in the same traceBuf.
func traceEventLocked(extraBytes int, mp *m, pid int32, bufp *traceBufPtr, ev byte, stackID uint32, skip int, args ...uint64) {
	buf := bufp.ptr()
	// TODO: test on non-zero extraBytes param.
	maxSize := 2 + 5*traceBytesPerNumber + extraBytes // event type, length, sequence, timestamp, stack id and two add params
	if buf == nil || len(buf.arr)-buf.pos < maxSize {
		systemstack(func() {
			buf = traceFlush(traceBufPtrOf(buf), pid).ptr()
		})
		bufp.set(buf)
	}

	ts := traceClockNow()
	if ts <= buf.lastTime {
		ts = buf.lastTime + 1
	}
	tsDiff := uint64(ts - buf.lastTime)
	buf.lastTime = ts
	narg := byte(len(args))
	if stackID != 0 || skip >= 0 {
		narg++
	}
	// We have only 2 bits for number of arguments.
	// If number is >= 3, then the event type is followed by event length in bytes.
	if narg > 3 {
		narg = 3
	}
	startPos := buf.pos
	buf.byte(ev | narg<<traceArgCountShift)
	var lenp *byte
	if narg == 3 {
		// Reserve the byte for length assuming that length < 128.
		buf.varint(0)
		lenp = &buf.arr[buf.pos-1]
	}
	buf.varint(tsDiff)
	for _, a := range args {
		buf.varint(a)
	}
	if stackID != 0 {
		buf.varint(uint64(stackID))
	} else if skip == 0 {
		buf.varint(0)
	} else if skip > 0 {
		buf.varint(traceStackID(mp, buf.stk[:], skip))
	}
	evSize := buf.pos - startPos
	if evSize > maxSize {
		throw("invalid length of trace event")
	}
	if lenp != nil {
		// Fill in actual length.
		*lenp = byte(evSize - 2)
	}
}

// traceCPUSample writes a CPU profile sample stack to the execution tracer's
// profiling buffer. It is called from a signal handler, so is limited in what
// it can do.
func traceCPUSample(gp *g, _ *m, pp *p, stk []uintptr) {
	if !traceEnabled() {
		// Tracing is usually turned off; don't spend time acquiring the signal
		// lock unless it's active.
		return
	}

	// Match the clock used in traceEventLocked
	now := traceClockNow()
	// The "header" here is the ID of the P that was running the profiled code,
	// followed by the ID of the goroutine. (For normal CPU profiling, it's
	// usually the number of samples with the given stack.) Near syscalls, pp
	// may be nil. Reporting goid of 0 is fine for either g0 or a nil gp.
	var hdr [2]uint64
	if pp != nil {
		// Overflow records in profBuf have all header values set to zero. Make
		// sure that real headers have at least one bit set.
		hdr[0] = uint64(pp.id)<<1 | 0b1
	} else {
		hdr[0] = 0b10
	}
	if gp != nil {
		hdr[1] = gp.goid
	}

	// Allow only one writer at a time
	for !trace.signalLock.CompareAndSwap(0, 1) {
		// TODO: Is it safe to osyield here? https://go.dev/issue/52672
		osyield()
	}

	if log := (*profBuf)(atomic.Loadp(unsafe.Pointer(&trace.cpuLogWrite))); log != nil {
		// Note: we don't pass a tag pointer here (how should profiling tags
		// interact with the execution tracer?), but if we did we'd need to be
		// careful about write barriers. See the long comment in profBuf.write.
		log.write(nil, int64(now), hdr[:], stk)
	}

	trace.signalLock.Store(0)
}

func traceReadCPU() {
	bufp := &trace.cpuLogBuf

	for {
		data, tags, _ := trace.cpuLogRead.read(profBufNonBlocking)
		if len(data) == 0 {
			break
		}
		for len(data) > 0 {
			if len(data) < 4 || data[0] > uint64(len(data)) {
				break // truncated profile
			}
			if data[0] < 4 || tags != nil && len(tags) < 1 {
				break // malformed profile
			}
			if len(tags) < 1 {
				break // mismatched profile records and tags
			}
			timestamp := data[1]
			ppid := data[2] >> 1
			if hasP := (data[2] & 0b1) != 0; !hasP {
				ppid = ^uint64(0)
			}
			goid := data[3]
			stk := data[4:data[0]]
			empty := len(stk) == 1 && data[2] == 0 && data[3] == 0
			data = data[data[0]:]
			// No support here for reporting goroutine tags at the moment; if
			// that information is to be part of the execution trace, we'd
			// probably want to see when the tags are applied and when they
			// change, instead of only seeing them when we get a CPU sample.
			tags = tags[1:]

			if empty {
				// Looks like an overflow record from the profBuf. Not much to
				// do here, we only want to report full records.
				//
				// TODO: should we start a goroutine to drain the profBuf,
				// rather than relying on a high-enough volume of tracing events
				// to keep ReadTrace busy? https://go.dev/issue/52674
				continue
			}

			buf := bufp.ptr()
			if buf == nil {
				systemstack(func() {
					*bufp = traceFlush(*bufp, 0)
				})
				buf = bufp.ptr()
			}
			nstk := 1
			buf.stk[0] = logicalStackSentinel
			for ; nstk < len(buf.stk) && nstk-1 < len(stk); nstk++ {
				buf.stk[nstk] = uintptr(stk[nstk-1])
			}
			stackID := trace.stackTab.put(buf.stk[:nstk])

			traceEventLocked(0, nil, 0, bufp, traceEvCPUSample, stackID, 1, timestamp, ppid, goid)
		}
	}
}

// logicalStackSentinel is a sentinel value at pcBuf[0] signifying that
// pcBuf[1:] holds a logical stack requiring no further processing. Any other
// value at pcBuf[0] represents a skip value to apply to the physical stack in
// pcBuf[1:] after inline expansion.
const logicalStackSentinel = ^uintptr(0)

// traceStackID captures a stack trace into pcBuf, registers it in the trace
// stack table, and returns its unique ID. pcBuf should have a length equal to
// traceStackSize. skip controls the number of leaf frames to omit in order to
// hide tracer internals from stack traces, see CL 5523.
func traceStackID(mp *m, pcBuf []uintptr, skip int) uint64 {
	gp := getg()
	curgp := mp.curg
	nstk := 1
	if tracefpunwindoff() || mp.hasCgoOnStack() {
		// Slow path: Unwind using default unwinder. Used when frame pointer
		// unwinding is unavailable or disabled (tracefpunwindoff), or might
		// produce incomplete results or crashes (hasCgoOnStack). Note that no
		// cgo callback related crashes have been observed yet. The main
		// motivation is to take advantage of a potentially registered cgo
		// symbolizer.
		pcBuf[0] = logicalStackSentinel
		if curgp == gp {
			nstk += callers(skip+1, pcBuf[1:])
		} else if curgp != nil {
			nstk += gcallers(curgp, skip, pcBuf[1:])
		}
	} else {
		// Fast path: Unwind using frame pointers.
		pcBuf[0] = uintptr(skip)
		if curgp == gp {
			nstk += fpTracebackPCs(unsafe.Pointer(getfp()), pcBuf[1:])
		} else if curgp != nil {
			// We're called on the g0 stack through mcall(fn) or systemstack(fn). To
			// behave like gcallers above, we start unwinding from sched.bp, which
			// points to the caller frame of the leaf frame on g's stack. The return
			// address of the leaf frame is stored in sched.pc, which we manually
			// capture here.
			pcBuf[1] = curgp.sched.pc
			nstk += 1 + fpTracebackPCs(unsafe.Pointer(curgp.sched.bp), pcBuf[2:])
		}
	}
	if nstk > 0 {
		nstk-- // skip runtime.goexit
	}
	if nstk > 0 && curgp.goid == 1 {
		nstk-- // skip runtime.main
	}
	id := trace.stackTab.put(pcBuf[:nstk])
	return uint64(id)
}

// tracefpunwindoff returns true if frame pointer unwinding for the tracer is
// disabled via GODEBUG or not supported by the architecture.
// TODO(#60254): support frame pointer unwinding on plan9/amd64.
func tracefpunwindoff() bool {
	return debug.tracefpunwindoff != 0 || (goarch.ArchFamily != goarch.AMD64 && goarch.ArchFamily != goarch.ARM64) || goos.IsPlan9 == 1
}

// fpTracebackPCs populates pcBuf with the return addresses for each frame and
// returns the number of PCs written to pcBuf. The returned PCs correspond to
// "physical frames" rather than "logical frames"; that is if A is inlined into
// B, this will return a PC for only B.
func fpTracebackPCs(fp unsafe.Pointer, pcBuf []uintptr) (i int) {
	for i = 0; i < len(pcBuf) && fp != nil; i++ {
		// return addr sits one word above the frame pointer
		pcBuf[i] = *(*uintptr)(unsafe.Pointer(uintptr(fp) + goarch.PtrSize))
		// follow the frame pointer to the next one
		fp = unsafe.Pointer(*(*uintptr)(fp))
	}
	return i
}

// traceAcquireBuffer returns trace buffer to use and, if necessary, locks it.
func traceAcquireBuffer() (mp *m, pid int32, bufp *traceBufPtr) {
	// Any time we acquire a buffer, we may end up flushing it,
	// but flushes are rare. Record the lock edge even if it
	// doesn't happen this time.
	lockRankMayTraceFlush()

	mp = acquirem()
	if p := mp.p.ptr(); p != nil {
		return mp, p.id, &p.trace.buf
	}
	lock(&trace.bufLock)
	return mp, traceGlobProc, &trace.buf
}

// traceReleaseBuffer releases a buffer previously acquired with traceAcquireBuffer.
func traceReleaseBuffer(mp *m, pid int32) {
	if pid == traceGlobProc {
		unlock(&trace.bufLock)
	}
	releasem(mp)
}

// lockRankMayTraceFlush records the lock ranking effects of a
// potential call to traceFlush.
func lockRankMayTraceFlush() {
	lockWithRankMayAcquire(&trace.lock, getLockRank(&trace.lock))
}

// traceFlush puts buf onto stack of full buffers and returns an empty buffer.
//
// This must run on the system stack because it acquires trace.lock.
//
//go:systemstack
func traceFlush(buf traceBufPtr, pid int32) traceBufPtr {
	lock(&trace.lock)
	if buf != 0 {
		traceFullQueue(buf)
	}
	if trace.empty != 0 {
		buf = trace.empty
		trace.empty = buf.ptr().link
	} else {
		buf = traceBufPtr(sysAlloc(unsafe.Sizeof(traceBuf{}), &memstats.other_sys))
		if buf == 0 {
			throw("trace: out of memory")
		}
	}
	bufp := buf.ptr()
	bufp.link.set(nil)
	bufp.pos = 0

	// initialize the buffer for a new batch
	ts := traceClockNow()
	if ts <= bufp.lastTime {
		ts = bufp.lastTime + 1
	}
	bufp.lastTime = ts
	bufp.byte(traceEvBatch | 1<<traceArgCountShift)
	bufp.varint(uint64(pid))
	bufp.varint(uint64(ts))

	unlock(&trace.lock)
	return buf
}

// traceString adds a string to the trace.strings and returns the id.
func traceString(bufp *traceBufPtr, pid int32, s string) (uint64, *traceBufPtr) {
	if s == "" {
		return 0, bufp
	}

	lock(&trace.stringsLock)
	if raceenabled {
		// raceacquire is necessary because the map access
		// below is race annotated.
		raceacquire(unsafe.Pointer(&trace.stringsLock))
	}

	if id, ok := trace.strings[s]; ok {
		if raceenabled {
			racerelease(unsafe.Pointer(&trace.stringsLock))
		}
		unlock(&trace.stringsLock)

		return id, bufp
	}

	trace.stringSeq++
	id := trace.stringSeq
	trace.strings[s] = id

	if raceenabled {
		racerelease(unsafe.Pointer(&trace.stringsLock))
	}
	unlock(&trace.stringsLock)

	// memory allocation in above may trigger tracing and
	// cause *bufp changes. Following code now works with *bufp,
	// so there must be no memory allocation or any activities
	// that causes tracing after this point.

	buf := bufp.ptr()
	size := 1 + 2*traceBytesPerNumber + len(s)
	if buf == nil || len(buf.arr)-buf.pos < size {
		systemstack(func() {
			buf = traceFlush(traceBufPtrOf(buf), pid).ptr()
			bufp.set(buf)
		})
	}
	buf.byte(traceEvString)
	buf.varint(id)

	// double-check the string and the length can fit.
	// Otherwise, truncate the string.
	slen := len(s)
	if room := len(buf.arr) - buf.pos; room < slen+traceBytesPerNumber {
		slen = room
	}

	buf.varint(uint64(slen))
	buf.pos += copy(buf.arr[buf.pos:], s[:slen])

	bufp.set(buf)
	return id, bufp
}

// varint appends v to buf in little-endian-base-128 encoding.
func (buf *traceBuf) varint(v uint64) {
	pos := buf.pos
	for ; v >= 0x80; v >>= 7 {
		buf.arr[pos] = 0x80 | byte(v)
		pos++
	}
	buf.arr[pos] = byte(v)
	pos++
	buf.pos = pos
}

// varintAt writes varint v at byte position pos in buf. This always
// consumes traceBytesPerNumber bytes. This is intended for when the
// caller needs to reserve space for a varint but can't populate it
// until later.
func (buf *traceBuf) varintAt(pos int, v uint64) {
	for i := 0; i < traceBytesPerNumber; i++ {
		if i < traceBytesPerNumber-1 {
			buf.arr[pos] = 0x80 | byte(v)
		} else {
			buf.arr[pos] = byte(v)
		}
		v >>= 7
		pos++
	}
}

// byte appends v to buf.
func (buf *traceBuf) byte(v byte) {
	buf.arr[buf.pos] = v
	buf.pos++
}

// traceStackTable maps stack traces (arrays of PC's) to unique uint32 ids.
// It is lock-free for reading.
type traceStackTable struct {
	lock mutex // Must be acquired on the system stack
	seq  uint32
	mem  traceAlloc
	tab  [1 << 13]traceStackPtr
}

// traceStack is a single stack in traceStackTable.
type traceStack struct {
	link traceStackPtr
	hash uintptr
	id   uint32
	n    int
	stk  [0]uintptr // real type [n]uintptr
}

type traceStackPtr uintptr

func (tp traceStackPtr) ptr() *traceStack { return (*traceStack)(unsafe.Pointer(tp)) }

// stack returns slice of PCs.
func (ts *traceStack) stack() []uintptr {
	return (*[traceStackSize]uintptr)(unsafe.Pointer(&ts.stk))[:ts.n]
}

// put returns a unique id for the stack trace pcs and caches it in the table,
// if it sees the trace for the first time.
func (tab *traceStackTable) put(pcs []uintptr) uint32 {
	if len(pcs) == 0 {
		return 0
	}
	hash := memhash(unsafe.Pointer(&pcs[0]), 0, uintptr(len(pcs))*unsafe.Sizeof(pcs[0]))
	// First, search the hashtable w/o the mutex.
	if id := tab.find(pcs, hash); id != 0 {
		return id
	}
	// Now, double check under the mutex.
	// Switch to the system stack so we can acquire tab.lock
	var id uint32
	systemstack(func() {
		lock(&tab.lock)
		if id = tab.find(pcs, hash); id != 0 {
			unlock(&tab.lock)
			return
		}
		// Create new record.
		tab.seq++
		stk := tab.newStack(len(pcs))
		stk.hash = hash
		stk.id = tab.seq
		id = stk.id
		stk.n = len(pcs)
		stkpc := stk.stack()
		copy(stkpc, pcs)
		part := int(hash % uintptr(len(tab.tab)))
		stk.link = tab.tab[part]
		atomicstorep(unsafe.Pointer(&tab.tab[part]), unsafe.Pointer(stk))
		unlock(&tab.lock)
	})
	return id
}

// find checks if the stack trace pcs is already present in the table.
func (tab *traceStackTable) find(pcs []uintptr, hash uintptr) uint32 {
	part := int(hash % uintptr(len(tab.tab)))
Search:
	for stk := tab.tab[part].ptr(); stk != nil; stk = stk.link.ptr() {
		if stk.hash == hash && stk.n == len(pcs) {
			for i, stkpc := range stk.stack() {
				if stkpc != pcs[i] {
					continue Search
				}
			}
			return stk.id
		}
	}
	return 0
}

// newStack allocates a new stack of size n.
func (tab *traceStackTable) newStack(n int) *traceStack {
	return (*traceStack)(tab.mem.alloc(unsafe.Sizeof(traceStack{}) + uintptr(n)*goarch.PtrSize))
}

// traceFrames returns the frames corresponding to pcs. It may
// allocate and may emit trace events.
func traceFrames(bufp traceBufPtr, pcs []uintptr) ([]traceFrame, traceBufPtr) {
	frames := make([]traceFrame, 0, len(pcs))
	ci := CallersFrames(pcs)
	for {
		var frame traceFrame
		f, more := ci.Next()
		frame, bufp = traceFrameForPC(bufp, 0, f)
		frames = append(frames, frame)
		if !more {
			return frames, bufp
		}
	}
}

// dump writes all previously cached stacks to trace buffers,
// releases all memory and resets state.
//
// This must run on the system stack because it calls traceFlush.
//
//go:systemstack
func (tab *traceStackTable) dump(bufp traceBufPtr) traceBufPtr {
	for i := range tab.tab {
		stk := tab.tab[i].ptr()
		for ; stk != nil; stk = stk.link.ptr() {
			var frames []traceFrame
			frames, bufp = traceFrames(bufp, fpunwindExpand(stk.stack()))

			// Estimate the size of this record. This
			// bound is pretty loose, but avoids counting
			// lots of varint sizes.
			maxSize := 1 + traceBytesPerNumber + (2+4*len(frames))*traceBytesPerNumber
			// Make sure we have enough buffer space.
			if buf := bufp.ptr(); len(buf.arr)-buf.pos < maxSize {
				bufp = traceFlush(bufp, 0)
			}

			// Emit header, with space reserved for length.
			buf := bufp.ptr()
			buf.byte(traceEvStack | 3<<traceArgCountShift)
			lenPos := buf.pos
			buf.pos += traceBytesPerNumber

			// Emit body.
			recPos := buf.pos
			buf.varint(uint64(stk.id))
			buf.varint(uint64(len(frames)))
			for _, frame := range frames {
				buf.varint(uint64(frame.PC))
				buf.varint(frame.funcID)
				buf.varint(frame.fileID)
				buf.varint(frame.line)
			}

			// Fill in size header.
			buf.varintAt(lenPos, uint64(buf.pos-recPos))
		}
	}

	tab.mem.drop()
	*tab = traceStackTable{}
	lockInit(&((*tab).lock), lockRankTraceStackTab)

	return bufp
}

// fpunwindExpand checks if pcBuf contains logical frames (which include inlined
// frames) or physical frames (produced by frame pointer unwinding) using a
// sentinel value in pcBuf[0]. Logical frames are simply returned without the
// sentinel. Physical frames are turned into logical frames via inline unwinding
// and by applying the skip value that's stored in pcBuf[0].
func fpunwindExpand(pcBuf []uintptr) []uintptr {
	if len(pcBuf) > 0 && pcBuf[0] == logicalStackSentinel {
		// pcBuf contains logical rather than inlined frames, skip has already been
		// applied, just return it without the sentinel value in pcBuf[0].
		return pcBuf[1:]
	}

	var (
		lastFuncID = abi.FuncIDNormal
		newPCBuf   = make([]uintptr, 0, traceStackSize)
		skip       = pcBuf[0]
		// skipOrAdd skips or appends retPC to newPCBuf and returns true if more
		// pcs can be added.
		skipOrAdd = func(retPC uintptr) bool {
			if skip > 0 {
				skip--
			} else {
				newPCBuf = append(newPCBuf, retPC)
			}
			return len(newPCBuf) < cap(newPCBuf)
		}
	)

outer:
	for _, retPC := range pcBuf[1:] {
		callPC := retPC - 1
		fi := findfunc(callPC)
		if !fi.valid() {
			// There is no funcInfo if callPC belongs to a C function. In this case
			// we still keep the pc, but don't attempt to expand inlined frames.
			if more := skipOrAdd(retPC); !more {
				break outer
			}
			continue
		}

		u, uf := newInlineUnwinder(fi, callPC)
		for ; uf.valid(); uf = u.next(uf) {
			sf := u.srcFunc(uf)
			if sf.funcID == abi.FuncIDWrapper && elideWrapperCalling(lastFuncID) {
				// ignore wrappers
			} else if more := skipOrAdd(uf.pc + 1); !more {
				break outer
			}
			lastFuncID = sf.funcID
		}
	}
	return newPCBuf
}

type traceFrame struct {
	PC     uintptr
	funcID uint64
	fileID uint64
	line   uint64
}

// traceFrameForPC records the frame information.
// It may allocate memory.
func traceFrameForPC(buf traceBufPtr, pid int32, f Frame) (traceFrame, traceBufPtr) {
	bufp := &buf
	var frame traceFrame
	frame.PC = f.PC

	fn := f.Function
	const maxLen = 1 << 10
	if len(fn) > maxLen {
		fn = fn[len(fn)-maxLen:]
	}
	frame.funcID, bufp = traceString(bufp, pid, fn)
	frame.line = uint64(f.Line)
	file := f.File
	if len(file) > maxLen {
		file = file[len(file)-maxLen:]
	}
	frame.fileID, bufp = traceString(bufp, pid, file)
	return frame, (*bufp)
}

// traceAlloc is a non-thread-safe region allocator.
// It holds a linked list of traceAllocBlock.
type traceAlloc struct {
	head traceAllocBlockPtr
	off  uintptr
}

// traceAllocBlock is a block in traceAlloc.
//
// traceAllocBlock is allocated from non-GC'd memory, so it must not
// contain heap pointers. Writes to pointers to traceAllocBlocks do
// not need write barriers.
type traceAllocBlock struct {
	_    sys.NotInHeap
	next traceAllocBlockPtr
	data [64<<10 - goarch.PtrSize]byte
}

// TODO: Since traceAllocBlock is now embedded runtime/internal/sys.NotInHeap, this isn't necessary.
type traceAllocBlockPtr uintptr

func (p traceAllocBlockPtr) ptr() *traceAllocBlock   { return (*traceAllocBlock)(unsafe.Pointer(p)) }
func (p *traceAllocBlockPtr) set(x *traceAllocBlock) { *p = traceAllocBlockPtr(unsafe.Pointer(x)) }

// alloc allocates n-byte block.
func (a *traceAlloc) alloc(n uintptr) unsafe.Pointer {
	n = alignUp(n, goarch.PtrSize)
	if a.head == 0 || a.off+n > uintptr(len(a.head.ptr().data)) {
		if n > uintptr(len(a.head.ptr().data)) {
			throw("trace: alloc too large")
		}
		block := (*traceAllocBlock)(sysAlloc(unsafe.Sizeof(traceAllocBlock{}), &memstats.other_sys))
		if block == nil {
			throw("trace: out of memory")
		}
		block.next.set(a.head.ptr())
		a.head.set(block)
		a.off = 0
	}
	p := &a.head.ptr().data[a.off]
	a.off += n
	return unsafe.Pointer(p)
}

// drop frees all previously allocated memory and resets the allocator.
func (a *traceAlloc) drop() {
	for a.head != 0 {
		block := a.head.ptr()
		a.head.set(block.next.ptr())
		sysFree(unsafe.Pointer(block), unsafe.Sizeof(traceAllocBlock{}), &memstats.other_sys)
	}
}

// The following functions write specific events to trace.

func (_ traceLocker) Gomaxprocs(procs int32) {
	traceEvent(traceEvGomaxprocs, 1, uint64(procs))
}

func (_ traceLocker) ProcStart() {
	traceEvent(traceEvProcStart, -1, uint64(getg().m.id))
}

func (_ traceLocker) ProcStop(pp *p) {
	// Sysmon and stopTheWorld can stop Ps blocked in syscalls,
	// to handle this we temporary employ the P.
	mp := acquirem()
	oldp := mp.p
	mp.p.set(pp)
	traceEvent(traceEvProcStop, -1)
	mp.p = oldp
	releasem(mp)
}

func (_ traceLocker) GCStart() {
	traceEvent(traceEvGCStart, 3, trace.seqGC)
	trace.seqGC++
}

func (_ traceLocker) GCDone() {
	traceEvent(traceEvGCDone, -1)
}

func (_ traceLocker) STWStart(reason stwReason) {
	// Don't trace if this STW is for trace start/stop, since traceEnabled
	// switches during a STW.
	if reason == stwStartTrace || reason == stwStopTrace {
		return
	}
	getg().m.trace.tracedSTWStart = true
	traceEvent(traceEvSTWStart, -1, uint64(reason))
}

func (_ traceLocker) STWDone() {
	mp := getg().m
	if !mp.trace.tracedSTWStart {
		return
	}
	mp.trace.tracedSTWStart = false
	traceEvent(traceEvSTWDone, -1)
}

// traceGCSweepStart prepares to trace a sweep loop. This does not
// emit any events until traceGCSweepSpan is called.
//
// traceGCSweepStart must be paired with traceGCSweepDone and there
// must be no preemption points between these two calls.
func (_ traceLocker) GCSweepStart() {
	// Delay the actual GCSweepStart event until the first span
	// sweep. If we don't sweep anything, don't emit any events.
	pp := getg().m.p.ptr()
	if pp.trace.inSweep {
		throw("double traceGCSweepStart")
	}
	pp.trace.inSweep, pp.trace.swept, pp.trace.reclaimed = true, 0, 0
}

// traceGCSweepSpan traces the sweep of a single page.
//
// This may be called outside a traceGCSweepStart/traceGCSweepDone
// pair; however, it will not emit any trace events in this case.
func (_ traceLocker) GCSweepSpan(bytesSwept uintptr) {
	pp := getg().m.p.ptr()
	if pp.trace.inSweep {
		if pp.trace.swept == 0 {
			traceEvent(traceEvGCSweepStart, 1)
		}
		pp.trace.swept += bytesSwept
	}
}

func (_ traceLocker) GCSweepDone() {
	pp := getg().m.p.ptr()
	if !pp.trace.inSweep {
		throw("missing traceGCSweepStart")
	}
	if pp.trace.swept != 0 {
		traceEvent(traceEvGCSweepDone, -1, uint64(pp.trace.swept), uint64(pp.trace.reclaimed))
	}
	pp.trace.inSweep = false
}

func (_ traceLocker) GCMarkAssistStart() {
	traceEvent(traceEvGCMarkAssistStart, 1)
}

func (_ traceLocker) GCMarkAssistDone() {
	traceEvent(traceEvGCMarkAssistDone, -1)
}

func (_ traceLocker) GoCreate(newg *g, pc uintptr) {
	newg.trace.seq = 0
	newg.trace.lastP = getg().m.p
	// +PCQuantum because traceFrameForPC expects return PCs and subtracts PCQuantum.
	id := trace.stackTab.put([]uintptr{logicalStackSentinel, startPCforTrace(pc) + sys.PCQuantum})
	traceEvent(traceEvGoCreate, 2, newg.goid, uint64(id))
}

func (_ traceLocker) GoStart() {
	gp := getg().m.curg
	pp := gp.m.p
	gp.trace.seq++
	if pp.ptr().gcMarkWorkerMode != gcMarkWorkerNotWorker {
		traceEvent(traceEvGoStartLabel, -1, gp.goid, gp.trace.seq, trace.markWorkerLabels[pp.ptr().gcMarkWorkerMode])
	} else if gp.trace.lastP == pp {
		traceEvent(traceEvGoStartLocal, -1, gp.goid)
	} else {
		gp.trace.lastP = pp
		traceEvent(traceEvGoStart, -1, gp.goid, gp.trace.seq)
	}
}

func (_ traceLocker) GoEnd() {
	traceEvent(traceEvGoEnd, -1)
}

func (_ traceLocker) GoSched() {
	gp := getg()
	gp.trace.lastP = gp.m.p
	traceEvent(traceEvGoSched, 1)
}

func (_ traceLocker) GoPreempt() {
	gp := getg()
	gp.trace.lastP = gp.m.p
	traceEvent(traceEvGoPreempt, 1)
}

func (_ traceLocker) GoPark(reason traceBlockReason, skip int) {
	// Convert the block reason directly to a trace event type.
	// See traceBlockReason for more information.
	traceEvent(byte(reason), skip)
}

func (_ traceLocker) GoUnpark(gp *g, skip int) {
	pp := getg().m.p
	gp.trace.seq++
	if gp.trace.lastP == pp {
		traceEvent(traceEvGoUnblockLocal, skip, gp.goid)
	} else {
		gp.trace.lastP = pp
		traceEvent(traceEvGoUnblock, skip, gp.goid, gp.trace.seq)
	}
}

func (_ traceLocker) GoSysCall() {
	var skip int
	switch {
	case tracefpunwindoff():
		// Unwind by skipping 1 frame relative to gp.syscallsp which is captured 3
		// frames above this frame. For frame pointer unwinding we produce the same
		// results by hard coding the number of frames in between our caller and the
		// actual syscall, see cases below.
		// TODO(felixge): Implement gp.syscallbp to avoid this workaround?
		skip = 1
	case GOOS == "solaris" || GOOS == "illumos":
		// These platforms don't use a libc_read_trampoline.
		skip = 3
	default:
		// Skip the extra trampoline frame used on most systems.
		skip = 4
	}
	getg().m.curg.trace.tracedSyscallEnter = true
	traceEvent(traceEvGoSysCall, skip)
}

func (_ traceLocker) GoSysExit(lostP bool) {
	if !lostP {
		throw("lostP must always be true in the old tracer for GoSysExit")
	}
	gp := getg().m.curg
	if !gp.trace.tracedSyscallEnter {
		// There was no syscall entry traced for us at all, so there's definitely
		// no EvGoSysBlock or EvGoInSyscall before us, which EvGoSysExit requires.
		return
	}
	gp.trace.tracedSyscallEnter = false
	ts := gp.trace.sysExitTime
	if ts != 0 && ts < trace.startTime {
		// There is a race between the code that initializes sysExitTimes
		// (in exitsyscall, which runs without a P, and therefore is not
		// stopped with the rest of the world) and the code that initializes
		// a new trace. The recorded sysExitTime must therefore be treated
		// as "best effort". If they are valid for this trace, then great,
		// use them for greater accuracy. But if they're not valid for this
		// trace, assume that the trace was started after the actual syscall
		// exit (but before we actually managed to start the goroutine,
		// aka right now), and assign a fresh time stamp to keep the log consistent.
		ts = 0
	}
	gp.trace.sysExitTime = 0
	gp.trace.seq++
	gp.trace.lastP = gp.m.p
	traceEvent(traceEvGoSysExit, -1, gp.goid, gp.trace.seq, uint64(ts))
}

// nosplit because it's called from exitsyscall without a P.
//
//go:nosplit
func (_ traceLocker) RecordSyscallExitedTime(gp *g, oldp *p) {
	// Wait till traceGoSysBlock event is emitted.
	// This ensures consistency of the trace (the goroutine is started after it is blocked).
	for oldp != nil && oldp.syscalltick == gp.m.syscalltick {
		osyield()
	}
	// We can't trace syscall exit right now because we don't have a P.
	// Tracing code can invoke write barriers that cannot run without a P.
	// So instead we remember the syscall exit time and emit the event
	// in execute when we have a P.
	gp.trace.sysExitTime = traceClockNow()
}

func (_ traceLocker) GoSysBlock(pp *p) {
	// Sysmon and stopTheWorld can declare syscalls running on remote Ps as blocked,
	// to handle this we temporary employ the P.
	mp := acquirem()
	oldp := mp.p
	mp.p.set(pp)
	traceEvent(traceEvGoSysBlock, -1)
	mp.p = oldp
	releasem(mp)
}

func (t traceLocker) ProcSteal(pp *p, forMe bool) {
	t.ProcStop(pp)
}

func (_ traceLocker) HeapAlloc(live uint64) {
	traceEvent(traceEvHeapAlloc, -1, live)
}

func (_ traceLocker) HeapGoal() {
	heapGoal := gcController.heapGoal()
	if heapGoal == ^uint64(0) {
		// Heap-based triggering is disabled.
		traceEvent(traceEvHeapGoal, -1, 0)
	} else {
		traceEvent(traceEvHeapGoal, -1, heapGoal)
	}
}

// To access runtime functions from runtime/trace.
// See runtime/trace/annotation.go

//go:linkname trace_userTaskCreate runtime/trace.userTaskCreate
func trace_userTaskCreate(id, parentID uint64, taskType string) {
	if !trace.enabled {
		return
	}

	// Same as in traceEvent.
	mp, pid, bufp := traceAcquireBuffer()
	if !trace.enabled && !mp.trace.startingTrace {
		traceReleaseBuffer(mp, pid)
		return
	}

	typeStringID, bufp := traceString(bufp, pid, taskType)
	traceEventLocked(0, mp, pid, bufp, traceEvUserTaskCreate, 0, 3, id, parentID, typeStringID)
	traceReleaseBuffer(mp, pid)
}

//go:linkname trace_userTaskEnd runtime/trace.userTaskEnd
func trace_userTaskEnd(id uint64) {
	traceEvent(traceEvUserTaskEnd, 2, id)
}

//go:linkname trace_userRegion runtime/trace.userRegion
func trace_userRegion(id, mode uint64, name string) {
	if !trace.enabled {
		return
	}

	mp, pid, bufp := traceAcquireBuffer()
	if !trace.enabled && !mp.trace.startingTrace {
		traceReleaseBuffer(mp, pid)
		return
	}

	nameStringID, bufp := traceString(bufp, pid, name)
	traceEventLocked(0, mp, pid, bufp, traceEvUserRegion, 0, 3, id, mode, nameStringID)
	traceReleaseBuffer(mp, pid)
}

//go:linkname trace_userLog runtime/trace.userLog
func trace_userLog(id uint64, category, message string) {
	if !trace.enabled {
		return
	}

	mp, pid, bufp := traceAcquireBuffer()
	if !trace.enabled && !mp.trace.startingTrace {
		traceReleaseBuffer(mp, pid)
		return
	}

	categoryID, bufp := traceString(bufp, pid, category)

	// The log message is recorded after all of the normal trace event
	// arguments, including the task, category, and stack IDs. We must ask
	// traceEventLocked to reserve extra space for the length of the message
	// and the message itself.
	extraSpace := traceBytesPerNumber + len(message)
	traceEventLocked(extraSpace, mp, pid, bufp, traceEvUserLog, 0, 3, id, categoryID)
	buf := bufp.ptr()

	// double-check the message and its length can fit.
	// Otherwise, truncate the message.
	slen := len(message)
	if room := len(buf.arr) - buf.pos; room < slen+traceBytesPerNumber {
		slen = room
	}
	buf.varint(uint64(slen))
	buf.pos += copy(buf.arr[buf.pos:], message[:slen])

	traceReleaseBuffer(mp, pid)
}

// the start PC of a goroutine for tracing purposes. If pc is a wrapper,
// it returns the PC of the wrapped function. Otherwise it returns pc.
func startPCforTrace(pc uintptr) uintptr {
	f := findfunc(pc)
	if !f.valid() {
		return pc // may happen for locked g in extra M since its pc is 0.
	}
	w := funcdata(f, abi.FUNCDATA_WrapInfo)
	if w == nil {
		return pc // not a wrapper
	}
	return f.datap.textAddr(*(*uint32)(w))
}

// OneNewExtraM registers the fact that a new extra M was created with
// the tracer. This matters if the M (which has an attached G) is used while
// the trace is still active because if it is, we need the fact that it exists
// to show up in the final trace.
func (tl traceLocker) OneNewExtraM(gp *g) {
	// Trigger two trace events for the locked g in the extra m,
	// since the next event of the g will be traceEvGoSysExit in exitsyscall,
	// while calling from C thread to Go.
	tl.GoCreate(gp, 0) // no start pc
	gp.trace.seq++
	traceEvent(traceEvGoInSyscall, -1, gp.goid)
}

// Used only in the new tracer.
func (tl traceLocker) GoCreateSyscall(gp *g) {
}

// Used only in the new tracer.
func (tl traceLocker) GoDestroySyscall() {
}

// traceTime represents a timestamp for the trace.
type traceTime uint64

// traceClockNow returns a monotonic timestamp. The clock this function gets
// the timestamp from is specific to tracing, and shouldn't be mixed with other
// clock sources.
//
// nosplit because it's called from exitsyscall, which is nosplit.
//
//go:nosplit
func traceClockNow() traceTime {
	return traceTime(cputicks() / traceTimeDiv)
}

func traceExitingSyscall() {
}

func traceExitedSyscall() {
}

// Not used in the old tracer. Defined for compatibility.
const defaultTraceAdvancePeriod = 0
