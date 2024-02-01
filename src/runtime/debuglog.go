// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides an internal debug logging facility. The debug
// log is a lightweight, in-memory, per-M ring buffer. By default, the
// runtime prints the debug log on panic.
//
// To print something to the debug log, call dlog to obtain a dlogger
// and use the methods on that to add values. The values will be
// space-separated in the output (much like println).
//
// This facility can be enabled by passing -tags debuglog when
// building. Without this tag, dlog calls compile to nothing.

package runtime

import (
	"internal/runtime/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// debugLogBytes is the size of each per-M ring buffer. This is
// allocated off-heap to avoid blowing up the M and hence the GC'd
// heap size.
const debugLogBytes = 16 << 10

// debugLogStringLimit is the maximum number of bytes in a string.
// Above this, the string will be truncated with "..(n more bytes).."
const debugLogStringLimit = debugLogBytes / 8

// dlog returns a debug logger. The caller can use methods on the
// returned logger to add values, which will be space-separated in the
// final output, much like println. The caller must call end() to
// finish the message.
//
// dlog can be used from highly-constrained corners of the runtime: it
// is safe to use in the signal handler, from within the write
// barrier, from within the stack implementation, and in places that
// must be recursively nosplit.
//
// This will be compiled away if built without the debuglog build tag.
// However, argument construction may not be. If any of the arguments
// are not literals or trivial expressions, consider protecting the
// call with "if dlogEnabled".
//
//go:nosplit
//go:nowritebarrierrec
func dlog() *dlogger {
	if !dlogEnabled {
		return nil
	}

	// Get the time.
	tick, nano := uint64(cputicks()), uint64(nanotime())

	// Try to get a cached logger.
	l := getCachedDlogger()

	// If we couldn't get a cached logger, try to get one from the
	// global pool.
	if l == nil {
		allp := (*uintptr)(unsafe.Pointer(&allDloggers))
		all := (*dlogger)(unsafe.Pointer(atomic.Loaduintptr(allp)))
		for l1 := all; l1 != nil; l1 = l1.allLink {
			if l1.owned.Load() == 0 && l1.owned.CompareAndSwap(0, 1) {
				l = l1
				break
			}
		}
	}

	// If that failed, allocate a new logger.
	if l == nil {
		// Use sysAllocOS instead of sysAlloc because we want to interfere
		// with the runtime as little as possible, and sysAlloc updates accounting.
		l = (*dlogger)(sysAllocOS(unsafe.Sizeof(dlogger{})))
		if l == nil {
			throw("failed to allocate debug log")
		}
		l.w.r.data = &l.w.data
		l.owned.Store(1)

		// Prepend to allDloggers list.
		headp := (*uintptr)(unsafe.Pointer(&allDloggers))
		for {
			head := atomic.Loaduintptr(headp)
			l.allLink = (*dlogger)(unsafe.Pointer(head))
			if atomic.Casuintptr(headp, head, uintptr(unsafe.Pointer(l))) {
				break
			}
		}
	}

	// If the time delta is getting too high, write a new sync
	// packet. We set the limit so we don't write more than 6
	// bytes of delta in the record header.
	const deltaLimit = 1<<(3*7) - 1 // ~2ms between sync packets
	if tick-l.w.tick > deltaLimit || nano-l.w.nano > deltaLimit {
		l.w.writeSync(tick, nano)
	}

	// Reserve space for framing header.
	l.w.ensure(debugLogHeaderSize)
	l.w.write += debugLogHeaderSize

	// Write record header.
	l.w.uvarint(tick - l.w.tick)
	l.w.uvarint(nano - l.w.nano)
	gp := getg()
	if gp != nil && gp.m != nil && gp.m.p != 0 {
		l.w.varint(int64(gp.m.p.ptr().id))
	} else {
		l.w.varint(-1)
	}

	return l
}

// A dlogger writes to the debug log.
//
// To obtain a dlogger, call dlog(). When done with the dlogger, call
// end().
type dlogger struct {
	_ sys.NotInHeap
	w debugLogWriter

	// allLink is the next dlogger in the allDloggers list.
	allLink *dlogger

	// owned indicates that this dlogger is owned by an M. This is
	// accessed atomically.
	owned atomic.Uint32
}

// allDloggers is a list of all dloggers, linked through
// dlogger.allLink. This is accessed atomically. This is prepend only,
// so it doesn't need to protect against ABA races.
var allDloggers *dlogger

//go:nosplit
func (l *dlogger) end() {
	if !dlogEnabled {
		return
	}

	// Fill in framing header.
	size := l.w.write - l.w.r.end
	if !l.w.writeFrameAt(l.w.r.end, size) {
		throw("record too large")
	}

	// Commit the record.
	l.w.r.end = l.w.write

	// Attempt to return this logger to the cache.
	if putCachedDlogger(l) {
		return
	}

	// Return the logger to the global pool.
	l.owned.Store(0)
}

const (
	debugLogUnknown = 1 + iota
	debugLogBoolTrue
	debugLogBoolFalse
	debugLogInt
	debugLogUint
	debugLogHex
	debugLogPtr
	debugLogString
	debugLogConstString
	debugLogStringOverflow

	debugLogPC
	debugLogTraceback
)

//go:nosplit
func (l *dlogger) b(x bool) *dlogger {
	if !dlogEnabled {
		return l
	}
	if x {
		l.w.byte(debugLogBoolTrue)
	} else {
		l.w.byte(debugLogBoolFalse)
	}
	return l
}

//go:nosplit
func (l *dlogger) i(x int) *dlogger {
	return l.i64(int64(x))
}

//go:nosplit
func (l *dlogger) i8(x int8) *dlogger {
	return l.i64(int64(x))
}

//go:nosplit
func (l *dlogger) i16(x int16) *dlogger {
	return l.i64(int64(x))
}

//go:nosplit
func (l *dlogger) i32(x int32) *dlogger {
	return l.i64(int64(x))
}

//go:nosplit
func (l *dlogger) i64(x int64) *dlogger {
	if !dlogEnabled {
		return l
	}
	l.w.byte(debugLogInt)
	l.w.varint(x)
	return l
}

//go:nosplit
func (l *dlogger) u(x uint) *dlogger {
	return l.u64(uint64(x))
}

//go:nosplit
func (l *dlogger) uptr(x uintptr) *dlogger {
	return l.u64(uint64(x))
}

//go:nosplit
func (l *dlogger) u8(x uint8) *dlogger {
	return l.u64(uint64(x))
}

//go:nosplit
func (l *dlogger) u16(x uint16) *dlogger {
	return l.u64(uint64(x))
}

//go:nosplit
func (l *dlogger) u32(x uint32) *dlogger {
	return l.u64(uint64(x))
}

//go:nosplit
func (l *dlogger) u64(x uint64) *dlogger {
	if !dlogEnabled {
		return l
	}
	l.w.byte(debugLogUint)
	l.w.uvarint(x)
	return l
}

//go:nosplit
func (l *dlogger) hex(x uint64) *dlogger {
	if !dlogEnabled {
		return l
	}
	l.w.byte(debugLogHex)
	l.w.uvarint(x)
	return l
}

//go:nosplit
func (l *dlogger) p(x any) *dlogger {
	if !dlogEnabled {
		return l
	}
	l.w.byte(debugLogPtr)
	if x == nil {
		l.w.uvarint(0)
	} else {
		v := efaceOf(&x)
		switch v._type.Kind_ & kindMask {
		case kindChan, kindFunc, kindMap, kindPtr, kindUnsafePointer:
			l.w.uvarint(uint64(uintptr(v.data)))
		default:
			throw("not a pointer type")
		}
	}
	return l
}

//go:nosplit
func (l *dlogger) s(x string) *dlogger {
	if !dlogEnabled {
		return l
	}

	strData := unsafe.StringData(x)
	datap := &firstmoduledata
	if len(x) > 4 && datap.etext <= uintptr(unsafe.Pointer(strData)) && uintptr(unsafe.Pointer(strData)) < datap.end {
		// String constants are in the rodata section, which
		// isn't recorded in moduledata. But it has to be
		// somewhere between etext and end.
		l.w.byte(debugLogConstString)
		l.w.uvarint(uint64(len(x)))
		l.w.uvarint(uint64(uintptr(unsafe.Pointer(strData)) - datap.etext))
	} else {
		l.w.byte(debugLogString)
		// We can't use unsafe.Slice as it may panic, which isn't safe
		// in this (potentially) nowritebarrier context.
		var b []byte
		bb := (*slice)(unsafe.Pointer(&b))
		bb.array = unsafe.Pointer(strData)
		bb.len, bb.cap = len(x), len(x)
		if len(b) > debugLogStringLimit {
			b = b[:debugLogStringLimit]
		}
		l.w.uvarint(uint64(len(b)))
		l.w.bytes(b)
		if len(b) != len(x) {
			l.w.byte(debugLogStringOverflow)
			l.w.uvarint(uint64(len(x) - len(b)))
		}
	}
	return l
}

//go:nosplit
func (l *dlogger) pc(x uintptr) *dlogger {
	if !dlogEnabled {
		return l
	}
	l.w.byte(debugLogPC)
	l.w.uvarint(uint64(x))
	return l
}

//go:nosplit
func (l *dlogger) traceback(x []uintptr) *dlogger {
	if !dlogEnabled {
		return l
	}
	l.w.byte(debugLogTraceback)
	l.w.uvarint(uint64(len(x)))
	for _, pc := range x {
		l.w.uvarint(uint64(pc))
	}
	return l
}

// A debugLogWriter is a ring buffer of binary debug log records.
//
// A log record consists of a 2-byte framing header and a sequence of
// fields. The framing header gives the size of the record as a little
// endian 16-bit value. Each field starts with a byte indicating its
// type, followed by type-specific data. If the size in the framing
// header is 0, it's a sync record consisting of two little endian
// 64-bit values giving a new time base.
//
// Because this is a ring buffer, new records will eventually
// overwrite old records. Hence, it maintains a reader that consumes
// the log as it gets overwritten. That reader state is where an
// actual log reader would start.
type debugLogWriter struct {
	_     sys.NotInHeap
	write uint64
	data  debugLogBuf

	// tick and nano are the time bases from the most recently
	// written sync record.
	tick, nano uint64

	// r is a reader that consumes records as they get overwritten
	// by the writer. It also acts as the initial reader state
	// when printing the log.
	r debugLogReader

	// buf is a scratch buffer for encoding. This is here to
	// reduce stack usage.
	buf [10]byte
}

type debugLogBuf struct {
	_ sys.NotInHeap
	b [debugLogBytes]byte
}

const (
	// debugLogHeaderSize is the number of bytes in the framing
	// header of every dlog record.
	debugLogHeaderSize = 2

	// debugLogSyncSize is the number of bytes in a sync record.
	debugLogSyncSize = debugLogHeaderSize + 2*8
)

//go:nosplit
func (l *debugLogWriter) ensure(n uint64) {
	for l.write+n >= l.r.begin+uint64(len(l.data.b)) {
		// Consume record at begin.
		if l.r.skip() == ^uint64(0) {
			// Wrapped around within a record.
			//
			// TODO(austin): It would be better to just
			// eat the whole buffer at this point, but we
			// have to communicate that to the reader
			// somehow.
			throw("record wrapped around")
		}
	}
}

//go:nosplit
func (l *debugLogWriter) writeFrameAt(pos, size uint64) bool {
	l.data.b[pos%uint64(len(l.data.b))] = uint8(size)
	l.data.b[(pos+1)%uint64(len(l.data.b))] = uint8(size >> 8)
	return size <= 0xFFFF
}

//go:nosplit
func (l *debugLogWriter) writeSync(tick, nano uint64) {
	l.tick, l.nano = tick, nano
	l.ensure(debugLogHeaderSize)
	l.writeFrameAt(l.write, 0)
	l.write += debugLogHeaderSize
	l.writeUint64LE(tick)
	l.writeUint64LE(nano)
	l.r.end = l.write
}

//go:nosplit
func (l *debugLogWriter) writeUint64LE(x uint64) {
	var b [8]byte
	b[0] = byte(x)
	b[1] = byte(x >> 8)
	b[2] = byte(x >> 16)
	b[3] = byte(x >> 24)
	b[4] = byte(x >> 32)
	b[5] = byte(x >> 40)
	b[6] = byte(x >> 48)
	b[7] = byte(x >> 56)
	l.bytes(b[:])
}

//go:nosplit
func (l *debugLogWriter) byte(x byte) {
	l.ensure(1)
	pos := l.write
	l.write++
	l.data.b[pos%uint64(len(l.data.b))] = x
}

//go:nosplit
func (l *debugLogWriter) bytes(x []byte) {
	l.ensure(uint64(len(x)))
	pos := l.write
	l.write += uint64(len(x))
	for len(x) > 0 {
		n := copy(l.data.b[pos%uint64(len(l.data.b)):], x)
		pos += uint64(n)
		x = x[n:]
	}
}

//go:nosplit
func (l *debugLogWriter) varint(x int64) {
	var u uint64
	if x < 0 {
		u = (^uint64(x) << 1) | 1 // complement i, bit 0 is 1
	} else {
		u = (uint64(x) << 1) // do not complement i, bit 0 is 0
	}
	l.uvarint(u)
}

//go:nosplit
func (l *debugLogWriter) uvarint(u uint64) {
	i := 0
	for u >= 0x80 {
		l.buf[i] = byte(u) | 0x80
		u >>= 7
		i++
	}
	l.buf[i] = byte(u)
	i++
	l.bytes(l.buf[:i])
}

type debugLogReader struct {
	data *debugLogBuf

	// begin and end are the positions in the log of the beginning
	// and end of the log data, modulo len(data).
	begin, end uint64

	// tick and nano are the current time base at begin.
	tick, nano uint64
}

//go:nosplit
func (r *debugLogReader) skip() uint64 {
	// Read size at pos.
	if r.begin+debugLogHeaderSize > r.end {
		return ^uint64(0)
	}
	size := uint64(r.readUint16LEAt(r.begin))
	if size == 0 {
		// Sync packet.
		r.tick = r.readUint64LEAt(r.begin + debugLogHeaderSize)
		r.nano = r.readUint64LEAt(r.begin + debugLogHeaderSize + 8)
		size = debugLogSyncSize
	}
	if r.begin+size > r.end {
		return ^uint64(0)
	}
	r.begin += size
	return size
}

//go:nosplit
func (r *debugLogReader) readUint16LEAt(pos uint64) uint16 {
	return uint16(r.data.b[pos%uint64(len(r.data.b))]) |
		uint16(r.data.b[(pos+1)%uint64(len(r.data.b))])<<8
}

//go:nosplit
func (r *debugLogReader) readUint64LEAt(pos uint64) uint64 {
	var b [8]byte
	for i := range b {
		b[i] = r.data.b[pos%uint64(len(r.data.b))]
		pos++
	}
	return uint64(b[0]) | uint64(b[1])<<8 |
		uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 |
		uint64(b[6])<<48 | uint64(b[7])<<56
}

func (r *debugLogReader) peek() (tick uint64) {
	// Consume any sync records.
	size := uint64(0)
	for size == 0 {
		if r.begin+debugLogHeaderSize > r.end {
			return ^uint64(0)
		}
		size = uint64(r.readUint16LEAt(r.begin))
		if size != 0 {
			break
		}
		if r.begin+debugLogSyncSize > r.end {
			return ^uint64(0)
		}
		// Sync packet.
		r.tick = r.readUint64LEAt(r.begin + debugLogHeaderSize)
		r.nano = r.readUint64LEAt(r.begin + debugLogHeaderSize + 8)
		r.begin += debugLogSyncSize
	}

	// Peek tick delta.
	if r.begin+size > r.end {
		return ^uint64(0)
	}
	pos := r.begin + debugLogHeaderSize
	var u uint64
	for i := uint(0); ; i += 7 {
		b := r.data.b[pos%uint64(len(r.data.b))]
		pos++
		u |= uint64(b&^0x80) << i
		if b&0x80 == 0 {
			break
		}
	}
	if pos > r.begin+size {
		return ^uint64(0)
	}
	return r.tick + u
}

func (r *debugLogReader) header() (end, tick, nano uint64, p int) {
	// Read size. We've already skipped sync packets and checked
	// bounds in peek.
	size := uint64(r.readUint16LEAt(r.begin))
	end = r.begin + size
	r.begin += debugLogHeaderSize

	// Read tick, nano, and p.
	tick = r.uvarint() + r.tick
	nano = r.uvarint() + r.nano
	p = int(r.varint())

	return
}

func (r *debugLogReader) uvarint() uint64 {
	var u uint64
	for i := uint(0); ; i += 7 {
		b := r.data.b[r.begin%uint64(len(r.data.b))]
		r.begin++
		u |= uint64(b&^0x80) << i
		if b&0x80 == 0 {
			break
		}
	}
	return u
}

func (r *debugLogReader) varint() int64 {
	u := r.uvarint()
	var v int64
	if u&1 == 0 {
		v = int64(u >> 1)
	} else {
		v = ^int64(u >> 1)
	}
	return v
}

func (r *debugLogReader) printVal() bool {
	typ := r.data.b[r.begin%uint64(len(r.data.b))]
	r.begin++

	switch typ {
	default:
		print("<unknown field type ", hex(typ), " pos ", r.begin-1, " end ", r.end, ">\n")
		return false

	case debugLogUnknown:
		print("<unknown kind>")

	case debugLogBoolTrue:
		print(true)

	case debugLogBoolFalse:
		print(false)

	case debugLogInt:
		print(r.varint())

	case debugLogUint:
		print(r.uvarint())

	case debugLogHex, debugLogPtr:
		print(hex(r.uvarint()))

	case debugLogString:
		sl := r.uvarint()
		if r.begin+sl > r.end {
			r.begin = r.end
			print("<string length corrupted>")
			break
		}
		for sl > 0 {
			b := r.data.b[r.begin%uint64(len(r.data.b)):]
			if uint64(len(b)) > sl {
				b = b[:sl]
			}
			r.begin += uint64(len(b))
			sl -= uint64(len(b))
			gwrite(b)
		}

	case debugLogConstString:
		len, ptr := int(r.uvarint()), uintptr(r.uvarint())
		ptr += firstmoduledata.etext
		// We can't use unsafe.String as it may panic, which isn't safe
		// in this (potentially) nowritebarrier context.
		str := stringStruct{
			str: unsafe.Pointer(ptr),
			len: len,
		}
		s := *(*string)(unsafe.Pointer(&str))
		print(s)

	case debugLogStringOverflow:
		print("..(", r.uvarint(), " more bytes)..")

	case debugLogPC:
		printDebugLogPC(uintptr(r.uvarint()), false)

	case debugLogTraceback:
		n := int(r.uvarint())
		for i := 0; i < n; i++ {
			print("\n\t")
			// gentraceback PCs are always return PCs.
			// Convert them to call PCs.
			//
			// TODO(austin): Expand inlined frames.
			printDebugLogPC(uintptr(r.uvarint()), true)
		}
	}

	return true
}

// printDebugLog prints the debug log.
func printDebugLog() {
	if !dlogEnabled {
		return
	}

	// This function should not panic or throw since it is used in
	// the fatal panic path and this may deadlock.

	printlock()

	// Get the list of all debug logs.
	allp := (*uintptr)(unsafe.Pointer(&allDloggers))
	all := (*dlogger)(unsafe.Pointer(atomic.Loaduintptr(allp)))

	// Count the logs.
	n := 0
	for l := all; l != nil; l = l.allLink {
		n++
	}
	if n == 0 {
		printunlock()
		return
	}

	// Prepare read state for all logs.
	type readState struct {
		debugLogReader
		first    bool
		lost     uint64
		nextTick uint64
	}
	// Use sysAllocOS instead of sysAlloc because we want to interfere
	// with the runtime as little as possible, and sysAlloc updates accounting.
	state1 := sysAllocOS(unsafe.Sizeof(readState{}) * uintptr(n))
	if state1 == nil {
		println("failed to allocate read state for", n, "logs")
		printunlock()
		return
	}
	state := (*[1 << 20]readState)(state1)[:n]
	{
		l := all
		for i := range state {
			s := &state[i]
			s.debugLogReader = l.w.r
			s.first = true
			s.lost = l.w.r.begin
			s.nextTick = s.peek()
			l = l.allLink
		}
	}

	// Print records.
	for {
		// Find the next record.
		var best struct {
			tick uint64
			i    int
		}
		best.tick = ^uint64(0)
		for i := range state {
			if state[i].nextTick < best.tick {
				best.tick = state[i].nextTick
				best.i = i
			}
		}
		if best.tick == ^uint64(0) {
			break
		}

		// Print record.
		s := &state[best.i]
		if s.first {
			print(">> begin log ", best.i)
			if s.lost != 0 {
				print("; lost first ", s.lost>>10, "KB")
			}
			print(" <<\n")
			s.first = false
		}

		end, _, nano, p := s.header()
		oldEnd := s.end
		s.end = end

		print("[")
		var tmpbuf [21]byte
		pnano := int64(nano) - runtimeInitTime
		if pnano < 0 {
			// Logged before runtimeInitTime was set.
			pnano = 0
		}
		pnanoBytes := itoaDiv(tmpbuf[:], uint64(pnano), 9)
		print(slicebytetostringtmp((*byte)(noescape(unsafe.Pointer(&pnanoBytes[0]))), len(pnanoBytes)))
		print(" P ", p, "] ")

		for i := 0; s.begin < s.end; i++ {
			if i > 0 {
				print(" ")
			}
			if !s.printVal() {
				// Abort this P log.
				print("<aborting P log>")
				end = oldEnd
				break
			}
		}
		println()

		// Move on to the next record.
		s.begin = end
		s.end = oldEnd
		s.nextTick = s.peek()
	}

	printunlock()
}

// printDebugLogPC prints a single symbolized PC. If returnPC is true,
// pc is a return PC that must first be converted to a call PC.
func printDebugLogPC(pc uintptr, returnPC bool) {
	fn := findfunc(pc)
	if returnPC && (!fn.valid() || pc > fn.entry()) {
		// TODO(austin): Don't back up if the previous frame
		// was a sigpanic.
		pc--
	}

	print(hex(pc))
	if !fn.valid() {
		print(" [unknown PC]")
	} else {
		name := funcname(fn)
		file, line := funcline(fn, pc)
		print(" [", name, "+", hex(pc-fn.entry()),
			" ", file, ":", line, "]")
	}
}
