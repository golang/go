// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go execution tracer.
// The tracer captures a wide range of execution events like goroutine
// creation/blocking/unblocking, syscall enter/exit/block, GC-related events,
// changes of heap size, processor start/stop, etc and writes them to a buffer
// in a compact form. A precise nanosecond-precision timestamp and a stack
// trace is captured for most events.
// See http://golang.org/s/go15trace for more info.

package runtime

import "unsafe"

// Event types in the trace, args are given in square brackets.
const (
	traceEvNone           = 0  // unused
	traceEvBatch          = 1  // start of per-P batch of events [pid, timestamp]
	traceEvFrequency      = 2  // contains tracer timer frequency [frequency (ticks per second)]
	traceEvStack          = 3  // stack [stack id, number of PCs, array of PCs]
	traceEvGomaxprocs     = 4  // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack id]
	traceEvProcStart      = 5  // start of P [timestamp]
	traceEvProcStop       = 6  // stop of P [timestamp]
	traceEvGCStart        = 7  // GC start [timestamp, stack id]
	traceEvGCDone         = 8  // GC done [timestamp]
	traceEvGCScanStart    = 9  // GC scan start [timestamp]
	traceEvGCScanDone     = 10 // GC scan done [timestamp]
	traceEvGCSweepStart   = 11 // GC sweep start [timestamp, stack id]
	traceEvGCSweepDone    = 12 // GC sweep done [timestamp]
	traceEvGoCreate       = 13 // goroutine creation [timestamp, new goroutine id, start PC, stack id]
	traceEvGoStart        = 14 // goroutine starts running [timestamp, goroutine id]
	traceEvGoEnd          = 15 // goroutine ends [timestamp]
	traceEvGoStop         = 16 // goroutine stops (like in select{}) [timestamp, stack]
	traceEvGoSched        = 17 // goroutine calls Gosched [timestamp, stack]
	traceEvGoPreempt      = 18 // goroutine is preempted [timestamp, stack]
	traceEvGoSleep        = 19 // goroutine calls Sleep [timestamp, stack]
	traceEvGoBlock        = 20 // goroutine blocks [timestamp, stack]
	traceEvGoUnblock      = 21 // goroutine is unblocked [timestamp, goroutine id, stack]
	traceEvGoBlockSend    = 22 // goroutine blocks on chan send [timestamp, stack]
	traceEvGoBlockRecv    = 23 // goroutine blocks on chan recv [timestamp, stack]
	traceEvGoBlockSelect  = 24 // goroutine blocks on select [timestamp, stack]
	traceEvGoBlockSync    = 25 // goroutine blocks on Mutex/RWMutex [timestamp, stack]
	traceEvGoBlockCond    = 26 // goroutine blocks on Cond [timestamp, stack]
	traceEvGoBlockNet     = 27 // goroutine blocks on network [timestamp, stack]
	traceEvGoSysCall      = 28 // syscall enter [timestamp, stack]
	traceEvGoSysExit      = 29 // syscall exit [timestamp, goroutine id]
	traceEvGoSysBlock     = 30 // syscall blocks [timestamp, stack]
	traceEvGoWaiting      = 31 // denotes that goroutine is blocked when tracing starts [goroutine id]
	traceEvGoInSyscall    = 32 // denotes that goroutine is in syscall when tracing starts [goroutine id]
	traceEvHeapAlloc      = 33 // memstats.heap_alloc change [timestamp, heap_alloc]
	traceEvNextGC         = 34 // memstats.next_gc change [timestamp, next_gc]
	traceEvTimerGoroutine = 35 // denotes timer goroutine [timer goroutine id]
	traceEvCount          = 36
)

const (
	// Timestamps in trace are cputicks/traceTickDiv.
	// This makes absolute values of timestamp diffs smaller,
	// and so they are encoded in less number of bytes.
	// 64 is somewhat arbitrary (one tick is ~20ns on a 3GHz machine).
	traceTickDiv = 64
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
	lock          mutex     // protects the following members
	lockOwner     *g        // to avoid deadlocks during recursive lock locks
	enabled       bool      // when set runtime traces events
	shutdown      bool      // set when we are waiting for trace reader to finish after setting enabled to false
	headerWritten bool      // whether ReadTrace has emitted trace header
	footerWritten bool      // whether ReadTrace has emitted trace footer
	shutdownSema  uint32    // used to wait for ReadTrace completion
	ticksStart    int64     // cputicks when tracing was started
	ticksEnd      int64     // cputicks when tracing was stopped
	timeStart     int64     // nanotime when tracing was started
	timeEnd       int64     // nanotime when tracing was stopped
	reading       *traceBuf // buffer currently handed off to user
	empty         *traceBuf // stack of empty buffers
	fullHead      *traceBuf // queue of full buffers
	fullTail      *traceBuf
	reader        *g              // goroutine that called ReadTrace, or nil
	stackTab      traceStackTable // maps stack traces to unique ids

	bufLock mutex     // protects buf
	buf     *traceBuf // global trace buffer, used when running without a p
}

// traceBufHeader is per-P tracing buffer.
type traceBufHeader struct {
	link      *traceBuf               // in trace.empty/full
	lastTicks uint64                  // when we wrote the last event
	buf       []byte                  // trace data, always points to traceBuf.arr
	stk       [traceStackSize]uintptr // scratch buffer for traceback
}

// traceBuf is per-P tracing buffer.
type traceBuf struct {
	traceBufHeader
	arr [64<<10 - unsafe.Sizeof(traceBufHeader{})]byte // underlying buffer for traceBufHeader.buf
}

// StartTrace enables tracing for the current process.
// While tracing, the data will be buffered and available via ReadTrace.
// StartTrace returns an error if tracing is already enabled.
// Most clients should use the runtime/pprof package or the testing package's
// -test.trace flag instead of calling StartTrace directly.
func StartTrace() error {
	// Stop the world, so that we can take a consistent snapshot
	// of all goroutines at the beginning of the trace.
	semacquire(&worldsema, false)
	_g_ := getg()
	_g_.m.preemptoff = "start tracing"
	systemstack(stoptheworld)

	// We are in stop-the-world, but syscalls can finish and write to trace concurrently.
	// Exitsyscall could check trace.enabled long before and then suddenly wake up
	// and decide to write to trace at a random point in time.
	// However, such syscall will use the global trace.buf buffer, because we've
	// acquired all p's by doing stop-the-world. So this protects us from such races.
	lock(&trace.bufLock)

	if trace.enabled || trace.shutdown {
		unlock(&trace.bufLock)
		_g_.m.preemptoff = ""
		semrelease(&worldsema)
		systemstack(starttheworld)
		return errorString("tracing is already enabled")
	}

	trace.ticksStart = cputicks()
	trace.timeStart = nanotime()
	trace.enabled = true
	trace.headerWritten = false
	trace.footerWritten = false

	for _, gp := range allgs {
		status := readgstatus(gp)
		if status != _Gdead {
			traceGoCreate(gp, gp.startpc)
		}
		if status == _Gwaiting {
			traceEvent(traceEvGoWaiting, false, uint64(gp.goid))
		}
		if status == _Gsyscall {
			traceEvent(traceEvGoInSyscall, false, uint64(gp.goid))
		}
	}
	traceProcStart()
	traceGoStart()

	unlock(&trace.bufLock)

	_g_.m.preemptoff = ""
	semrelease(&worldsema)
	systemstack(starttheworld)
	return nil
}

// StopTrace stops tracing, if it was previously enabled.
// StopTrace only returns after all the reads for the trace have completed.
func StopTrace() {
	// Stop the world so that we can collect the trace buffers from all p's below,
	// and also to avoid races with traceEvent.
	semacquire(&worldsema, false)
	_g_ := getg()
	_g_.m.preemptoff = "stop tracing"
	systemstack(stoptheworld)

	// See the comment in StartTrace.
	lock(&trace.bufLock)

	if !trace.enabled {
		unlock(&trace.bufLock)
		_g_.m.preemptoff = ""
		semrelease(&worldsema)
		systemstack(starttheworld)
		return
	}

	traceGoSched()
	traceGoStart()

	for _, p := range &allp {
		if p == nil {
			break
		}
		buf := p.tracebuf
		if buf != nil {
			traceFullQueue(buf)
			p.tracebuf = nil
		}
	}
	if trace.buf != nil && len(trace.buf.buf) != 0 {
		buf := trace.buf
		trace.buf = nil
		traceFullQueue(buf)
	}

	for {
		trace.ticksEnd = cputicks()
		trace.timeEnd = nanotime()
		// Windows time can tick only every 15ms, wait for at least one tick.
		if trace.timeEnd != trace.timeStart {
			break
		}
		osyield()
	}

	trace.enabled = false
	trace.shutdown = true
	trace.stackTab.dump()

	unlock(&trace.bufLock)

	_g_.m.preemptoff = ""
	semrelease(&worldsema)
	systemstack(starttheworld)

	// The world is started but we've set trace.shutdown, so new tracing can't start.
	// Wait for the trace reader to flush pending buffers and stop.
	semacquire(&trace.shutdownSema, false)
	if raceenabled {
		raceacquire(unsafe.Pointer(&trace.shutdownSema))
	}

	// The lock protects us from races with StartTrace/StopTrace because they do stop-the-world.
	lock(&trace.lock)
	for _, p := range &allp {
		if p == nil {
			break
		}
		if p.tracebuf != nil {
			throw("trace: non-empty trace buffer in proc")
		}
	}
	if trace.buf != nil {
		throw("trace: non-empty global trace buffer")
	}
	if trace.fullHead != nil || trace.fullTail != nil {
		throw("trace: non-empty full trace buffer")
	}
	if trace.reading != nil || trace.reader != nil {
		throw("trace: reading after shutdown")
	}
	for trace.empty != nil {
		buf := trace.empty
		trace.empty = buf.link
		sysFree(unsafe.Pointer(buf), unsafe.Sizeof(*buf), &memstats.other_sys)
	}
	trace.shutdown = false
	unlock(&trace.lock)
}

// ReadTrace returns the next chunk of binary tracing data, blocking until data
// is available. If tracing is turned off and all the data accumulated while it
// was on has been returned, ReadTrace returns nil. The caller must copy the
// returned data before calling ReadTrace again.
// ReadTrace must be called from one goroutine at a time.
func ReadTrace() []byte {
	// This function may need to lock trace.lock recursively
	// (goparkunlock -> traceGoPark -> traceEvent -> traceFlush).
	// To allow this we use trace.lockOwner.
	// Also this function must not allocate while holding trace.lock:
	// allocation can call heap allocate, which will try to emit a trace
	// event while holding heap lock.
	lock(&trace.lock)
	trace.lockOwner = getg()

	if trace.reader != nil {
		// More than one goroutine reads trace. This is bad.
		// But we rather do not crash the program because of tracing,
		// because tracing can be enabled at runtime on prod servers.
		trace.lockOwner = nil
		unlock(&trace.lock)
		println("runtime: ReadTrace called from multiple goroutines simultaneously")
		return nil
	}
	// Recycle the old buffer.
	if buf := trace.reading; buf != nil {
		buf.link = trace.empty
		trace.empty = buf
		trace.reading = nil
	}
	// Write trace header.
	if !trace.headerWritten {
		trace.headerWritten = true
		trace.lockOwner = nil
		unlock(&trace.lock)
		return []byte("gotrace\x00")
	}
	// Wait for new data.
	if trace.fullHead == nil && !trace.shutdown {
		trace.reader = getg()
		goparkunlock(&trace.lock, "trace reader (blocked)", traceEvGoBlock)
		lock(&trace.lock)
	}
	// Write a buffer.
	if trace.fullHead != nil {
		buf := traceFullDequeue()
		trace.reading = buf
		trace.lockOwner = nil
		unlock(&trace.lock)
		return buf.buf
	}
	// Write footer with timer frequency.
	if !trace.footerWritten {
		trace.footerWritten = true
		// Use float64 because (trace.ticksEnd - trace.ticksStart) * 1e9 can overflow int64.
		freq := float64(trace.ticksEnd-trace.ticksStart) * 1e9 / float64(trace.timeEnd-trace.timeStart) / traceTickDiv
		trace.lockOwner = nil
		unlock(&trace.lock)
		var data []byte
		data = append(data, traceEvFrequency|0<<traceArgCountShift)
		data = traceAppend(data, uint64(freq))
		if timers.gp != nil {
			data = append(data, traceEvTimerGoroutine|0<<traceArgCountShift)
			data = traceAppend(data, uint64(timers.gp.goid))
		}
		return data
	}
	// Done.
	if trace.shutdown {
		trace.lockOwner = nil
		unlock(&trace.lock)
		if raceenabled {
			// Model synchronization on trace.shutdownSema, which race
			// detector does not see. This is required to avoid false
			// race reports on writer passed to pprof.StartTrace.
			racerelease(unsafe.Pointer(&trace.shutdownSema))
		}
		// trace.enabled is already reset, so can call traceable functions.
		semrelease(&trace.shutdownSema)
		return nil
	}
	// Also bad, but see the comment above.
	trace.lockOwner = nil
	unlock(&trace.lock)
	println("runtime: spurious wakeup of trace reader")
	return nil
}

// traceReader returns the trace reader that should be woken up, if any.
func traceReader() *g {
	if trace.reader == nil || (trace.fullHead == nil && !trace.shutdown) {
		return nil
	}
	lock(&trace.lock)
	if trace.reader == nil || (trace.fullHead == nil && !trace.shutdown) {
		unlock(&trace.lock)
		return nil
	}
	gp := trace.reader
	trace.reader = nil
	unlock(&trace.lock)
	return gp
}

// traceProcFree frees trace buffer associated with pp.
func traceProcFree(pp *p) {
	buf := pp.tracebuf
	pp.tracebuf = nil
	if buf == nil {
		return
	}
	lock(&trace.lock)
	traceFullQueue(buf)
	unlock(&trace.lock)
}

// traceFullQueue queues buf into queue of full buffers.
func traceFullQueue(buf *traceBuf) {
	buf.link = nil
	if trace.fullHead == nil {
		trace.fullHead = buf
	} else {
		trace.fullTail.link = buf
	}
	trace.fullTail = buf
}

// traceFullDequeue dequeues from queue of full buffers.
func traceFullDequeue() *traceBuf {
	buf := trace.fullHead
	if buf == nil {
		return nil
	}
	trace.fullHead = buf.link
	if trace.fullHead == nil {
		trace.fullTail = nil
	}
	buf.link = nil
	return buf
}

// traceEvent writes a single event to trace buffer, flushing the buffer if necessary.
// ev is event type.
// If stack, write current stack id as the last argument.
func traceEvent(ev byte, stack bool, args ...uint64) {
	mp, pid, bufp := traceAcquireBuffer()
	// Double-check trace.enabled now that we've done m.locks++ and acquired bufLock.
	// This protects from races between traceEvent and StartTrace/StopTrace.

	// The caller checked that trace.enabled == true, but trace.enabled might have been
	// turned off between the check and now. Check again. traceLockBuffer did mp.locks++,
	// StopTrace does stoptheworld, and stoptheworld waits for mp.locks to go back to zero,
	// so if we see trace.enabled == true now, we know it's true for the rest of the function.
	// Exitsyscall can run even during stoptheworld. The race with StartTrace/StopTrace
	// during tracing in exitsyscall is resolved by locking trace.bufLock in traceLockBuffer.
	if !trace.enabled {
		traceReleaseBuffer(pid)
		return
	}
	buf := *bufp
	const maxSize = 2 + 4*traceBytesPerNumber // event type, length, timestamp, stack id and two add params
	if buf == nil || cap(buf.buf)-len(buf.buf) < maxSize {
		buf = traceFlush(buf)
		*bufp = buf
	}

	ticks := uint64(cputicks()) / traceTickDiv
	tickDiff := ticks - buf.lastTicks
	if len(buf.buf) == 0 {
		data := buf.buf
		data = append(data, traceEvBatch|1<<traceArgCountShift)
		data = traceAppend(data, uint64(pid))
		data = traceAppend(data, ticks)
		buf.buf = data
		tickDiff = 0
	}
	buf.lastTicks = ticks
	narg := byte(len(args))
	if stack {
		narg++
	}
	// We have only 2 bits for number of arguments.
	// If number is >= 3, then the event type is followed by event length in bytes.
	if narg > 3 {
		narg = 3
	}
	data := buf.buf
	data = append(data, ev|narg<<traceArgCountShift)
	var lenp *byte
	if narg == 3 {
		// Reserve the byte for length assuming that length < 128.
		data = append(data, 0)
		lenp = &data[len(data)-1]
	}
	data = traceAppend(data, tickDiff)
	for _, a := range args {
		data = traceAppend(data, a)
	}
	if stack {
		_g_ := getg()
		gp := mp.curg
		if gp == nil && ev == traceEvGoSysBlock {
			gp = _g_
		}
		var nstk int
		if gp == _g_ {
			nstk = callers(1, &buf.stk[0], len(buf.stk))
		} else if gp != nil {
			nstk = gcallers(mp.curg, 1, &buf.stk[0], len(buf.stk))
		}
		id := trace.stackTab.put(buf.stk[:nstk])
		data = traceAppend(data, uint64(id))
	}
	evSize := len(data) - len(buf.buf)
	if evSize > maxSize {
		throw("invalid length of trace event")
	}
	if lenp != nil {
		// Fill in actual length.
		*lenp = byte(evSize - 2)
	}
	buf.buf = data
	traceReleaseBuffer(pid)
}

// traceAcquireBuffer returns trace buffer to use and, if necessary, locks it.
func traceAcquireBuffer() (mp *m, pid int32, bufp **traceBuf) {
	mp = acquirem()
	if p := mp.p; p != nil {
		return mp, p.id, &p.tracebuf
	}
	lock(&trace.bufLock)
	return mp, traceGlobProc, &trace.buf
}

// traceReleaseBuffer releases a buffer previously acquired with traceAcquireBuffer.
func traceReleaseBuffer(pid int32) {
	if pid == traceGlobProc {
		unlock(&trace.bufLock)
	}
	releasem(getg().m)
}

// traceFlush puts buf onto stack of full buffers and returns an empty buffer.
func traceFlush(buf *traceBuf) *traceBuf {
	owner := trace.lockOwner
	dolock := owner == nil || owner != getg().m.curg
	if dolock {
		lock(&trace.lock)
	}
	if buf != nil {
		if &buf.buf[0] != &buf.arr[0] {
			throw("trace buffer overflow")
		}
		traceFullQueue(buf)
	}
	if trace.empty != nil {
		buf = trace.empty
		trace.empty = buf.link
	} else {
		buf = (*traceBuf)(sysAlloc(unsafe.Sizeof(traceBuf{}), &memstats.other_sys))
		if buf == nil {
			throw("trace: out of memory")
		}
	}
	buf.link = nil
	buf.buf = buf.arr[:0]
	buf.lastTicks = 0
	if dolock {
		unlock(&trace.lock)
	}
	return buf
}

// traceAppend appends v to buf in little-endian-base-128 encoding.
func traceAppend(buf []byte, v uint64) []byte {
	for ; v >= 0x80; v >>= 7 {
		buf = append(buf, 0x80|byte(v))
	}
	buf = append(buf, byte(v))
	return buf
}

// traceStackTable maps stack traces (arrays of PC's) to unique uint32 ids.
// It is lock-free for reading.
type traceStackTable struct {
	lock mutex
	seq  uint32
	mem  traceAlloc
	tab  [1 << 13]*traceStack
}

// traceStack is a single stack in traceStackTable.
type traceStack struct {
	link *traceStack
	hash uintptr
	id   uint32
	n    int
	stk  [0]uintptr // real type [n]uintptr
}

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
	hash := memhash(unsafe.Pointer(&pcs[0]), uintptr(len(pcs))*unsafe.Sizeof(pcs[0]), 0)
	// First, search the hashtable w/o the mutex.
	if id := tab.find(pcs, hash); id != 0 {
		return id
	}
	// Now, double check under the mutex.
	lock(&tab.lock)
	if id := tab.find(pcs, hash); id != 0 {
		unlock(&tab.lock)
		return id
	}
	// Create new record.
	tab.seq++
	stk := tab.newStack(len(pcs))
	stk.hash = hash
	stk.id = tab.seq
	stk.n = len(pcs)
	stkpc := stk.stack()
	for i, pc := range pcs {
		stkpc[i] = pc
	}
	part := int(hash % uintptr(len(tab.tab)))
	stk.link = tab.tab[part]
	atomicstorep(unsafe.Pointer(&tab.tab[part]), unsafe.Pointer(stk))
	unlock(&tab.lock)
	return stk.id
}

// find checks if the stack trace pcs is already present in the table.
func (tab *traceStackTable) find(pcs []uintptr, hash uintptr) uint32 {
	part := int(hash % uintptr(len(tab.tab)))
Search:
	for stk := tab.tab[part]; stk != nil; stk = stk.link {
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
	return (*traceStack)(tab.mem.alloc(unsafe.Sizeof(traceStack{}) + uintptr(n)*ptrSize))
}

// dump writes all previously cached stacks to trace buffers,
// releases all memory and resets state.
func (tab *traceStackTable) dump() {
	var tmp [(2 + traceStackSize) * traceBytesPerNumber]byte
	buf := traceFlush(nil)
	for _, stk := range tab.tab {
		for ; stk != nil; stk = stk.link {
			maxSize := 1 + (3+stk.n)*traceBytesPerNumber
			if cap(buf.buf)-len(buf.buf) < maxSize {
				buf = traceFlush(buf)
			}
			// Form the event in the temp buffer, we need to know the actual length.
			tmpbuf := tmp[:0]
			tmpbuf = traceAppend(tmpbuf, uint64(stk.id))
			tmpbuf = traceAppend(tmpbuf, uint64(stk.n))
			for _, pc := range stk.stack() {
				tmpbuf = traceAppend(tmpbuf, uint64(pc))
			}
			// Now copy to the buffer.
			data := buf.buf
			data = append(data, traceEvStack|3<<traceArgCountShift)
			data = traceAppend(data, uint64(len(tmpbuf)))
			data = append(data, tmpbuf...)
			buf.buf = data
		}
	}

	lock(&trace.lock)
	traceFullQueue(buf)
	unlock(&trace.lock)

	tab.mem.drop()
	*tab = traceStackTable{}
}

// traceAlloc is a non-thread-safe region allocator.
// It holds a linked list of traceAllocBlock.
type traceAlloc struct {
	head *traceAllocBlock
	off  uintptr
}

// traceAllocBlock is a block in traceAlloc.
type traceAllocBlock struct {
	next *traceAllocBlock
	data [64<<10 - ptrSize]byte
}

// alloc allocates n-byte block.
func (a *traceAlloc) alloc(n uintptr) unsafe.Pointer {
	n = round(n, ptrSize)
	if a.head == nil || a.off+n > uintptr(len(a.head.data)) {
		if n > uintptr(len(a.head.data)) {
			throw("trace: alloc too large")
		}
		block := (*traceAllocBlock)(sysAlloc(unsafe.Sizeof(traceAllocBlock{}), &memstats.other_sys))
		if block == nil {
			throw("trace: out of memory")
		}
		block.next = a.head
		a.head = block
		a.off = 0
	}
	p := &a.head.data[a.off]
	a.off += n
	return unsafe.Pointer(p)
}

// drop frees all previously allocated memory and resets the allocator.
func (a *traceAlloc) drop() {
	for a.head != nil {
		block := a.head
		a.head = block.next
		sysFree(unsafe.Pointer(block), unsafe.Sizeof(traceAllocBlock{}), &memstats.other_sys)
	}
}

// The following functions write specific events to trace.

func traceGomaxprocs(procs int32) {
	traceEvent(traceEvGomaxprocs, true, uint64(procs))
}

func traceProcStart() {
	traceEvent(traceEvProcStart, false)
}

func traceProcStop(pp *p) {
	// Sysmon and stoptheworld can stop Ps blocked in syscalls,
	// to handle this we temporary employ the P.
	mp := acquirem()
	oldp := mp.p
	mp.p = pp
	traceEvent(traceEvProcStop, false)
	mp.p = oldp
	releasem(mp)
}

func traceGCStart() {
	traceEvent(traceEvGCStart, true)
}

func traceGCDone() {
	traceEvent(traceEvGCDone, false)
}

func traceGCScanStart() {
	traceEvent(traceEvGCScanStart, false)
}

func traceGCScanDone() {
	traceEvent(traceEvGCScanDone, false)
}

func traceGCSweepStart() {
	traceEvent(traceEvGCSweepStart, true)
}

func traceGCSweepDone() {
	traceEvent(traceEvGCSweepDone, false)
}

func traceGoCreate(newg *g, pc uintptr) {
	traceEvent(traceEvGoCreate, true, uint64(newg.goid), uint64(pc))
}

func traceGoStart() {
	traceEvent(traceEvGoStart, false, uint64(getg().m.curg.goid))
}

func traceGoEnd() {
	traceEvent(traceEvGoEnd, false)
}

func traceGoSched() {
	traceEvent(traceEvGoSched, true)
}

func traceGoPreempt() {
	traceEvent(traceEvGoPreempt, true)
}

func traceGoStop() {
	traceEvent(traceEvGoStop, true)
}

func traceGoPark(traceEv byte, gp *g) {
	traceEvent(traceEv, true)
}

func traceGoUnpark(gp *g) {
	traceEvent(traceEvGoUnblock, true, uint64(gp.goid))
}

func traceGoSysCall() {
	traceEvent(traceEvGoSysCall, true)
}

func traceGoSysExit() {
	traceEvent(traceEvGoSysExit, false, uint64(getg().m.curg.goid))
}

func traceGoSysBlock(pp *p) {
	// Sysmon and stoptheworld can declare syscalls running on remote Ps as blocked,
	// to handle this we temporary employ the P.
	mp := acquirem()
	oldp := mp.p
	mp.p = pp
	traceEvent(traceEvGoSysBlock, true)
	mp.p = oldp
	releasem(mp)
}

func traceHeapAlloc() {
	traceEvent(traceEvHeapAlloc, false, memstats.heap_alloc)
}

func traceNextGC() {
	traceEvent(traceEvNextGC, false, memstats.next_gc)
}
