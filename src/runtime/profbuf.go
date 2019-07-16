// Copyright 2017 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

// A profBuf is a lock-free buffer for profiling events,
// safe for concurrent use by one reader and one writer.
// The writer may be a signal handler running without a user g.
// The reader is assumed to be a user g.
//
// Each logged event corresponds to a fixed size header, a list of
// uintptrs (typically a stack), and exactly one unsafe.Pointer tag.
// The header and uintptrs are stored in the circular buffer data and the
// tag is stored in a circular buffer tags, running in parallel.
// In the circular buffer data, each event takes 2+hdrsize+len(stk)
// words: the value 2+hdrsize+len(stk), then the time of the event, then
// hdrsize words giving the fixed-size header, and then len(stk) words
// for the stack.
//
// The current effective offsets into the tags and data circular buffers
// for reading and writing are stored in the high 30 and low 32 bits of r and w.
// The bottom bits of the high 32 are additional flag bits in w, unused in r.
// "Effective" offsets means the total number of reads or writes, mod 2^length.
// The offset in the buffer is the effective offset mod the length of the buffer.
// To make wraparound mod 2^length match wraparound mod length of the buffer,
// the length of the buffer must be a power of two.
//
// If the reader catches up to the writer, a flag passed to read controls
// whether the read blocks until more data is available. A read returns a
// pointer to the buffer data itself; the caller is assumed to be done with
// that data at the next read. The read offset rNext tracks the next offset to
// be returned by read. By definition, r ≤ rNext ≤ w (before wraparound),
// and rNext is only used by the reader, so it can be accessed without atomics.
//
// If the writer gets ahead of the reader, so that the buffer fills,
// future writes are discarded and replaced in the output stream by an
// overflow entry, which has size 2+hdrsize+1, time set to the time of
// the first discarded write, a header of all zeroed words, and a "stack"
// containing one word, the number of discarded writes.
//
// Between the time the buffer fills and the buffer becomes empty enough
// to hold more data, the overflow entry is stored as a pending overflow
// entry in the fields overflow and overflowTime. The pending overflow
// entry can be turned into a real record by either the writer or the
// reader. If the writer is called to write a new record and finds that
// the output buffer has room for both the pending overflow entry and the
// new record, the writer emits the pending overflow entry and the new
// record into the buffer. If the reader is called to read data and finds
// that the output buffer is empty but that there is a pending overflow
// entry, the reader will return a synthesized record for the pending
// overflow entry.
//
// Only the writer can create or add to a pending overflow entry, but
// either the reader or the writer can clear the pending overflow entry.
// A pending overflow entry is indicated by the low 32 bits of 'overflow'
// holding the number of discarded writes, and overflowTime holding the
// time of the first discarded write. The high 32 bits of 'overflow'
// increment each time the low 32 bits transition from zero to non-zero
// or vice versa. This sequence number avoids ABA problems in the use of
// compare-and-swap to coordinate between reader and writer.
// The overflowTime is only written when the low 32 bits of overflow are
// zero, that is, only when there is no pending overflow entry, in
// preparation for creating a new one. The reader can therefore fetch and
// clear the entry atomically using
//
//	for {
//		overflow = load(&b.overflow)
//		if uint32(overflow) == 0 {
//			// no pending entry
//			break
//		}
//		time = load(&b.overflowTime)
//		if cas(&b.overflow, overflow, ((overflow>>32)+1)<<32) {
//			// pending entry cleared
//			break
//		}
//	}
//	if uint32(overflow) > 0 {
//		emit entry for uint32(overflow), time
//	}
//
type profBuf struct {
	// accessed atomically
	r, w         profAtomic
	overflow     uint64
	overflowTime uint64
	eof          uint32

	// immutable (excluding slice content)
	hdrsize uintptr
	data    []uint64
	tags    []unsafe.Pointer

	// owned by reader
	rNext       profIndex
	overflowBuf []uint64 // for use by reader to return overflow record
	wait        note
}

// A profAtomic is the atomically-accessed word holding a profIndex.
type profAtomic uint64

// A profIndex is the packet tag and data counts and flags bits, described above.
type profIndex uint64

const (
	profReaderSleeping profIndex = 1 << 32 // reader is sleeping and must be woken up
	profWriteExtra     profIndex = 1 << 33 // overflow or eof waiting
)

func (x *profAtomic) load() profIndex {
	return profIndex(atomic.Load64((*uint64)(x)))
}

func (x *profAtomic) store(new profIndex) {
	atomic.Store64((*uint64)(x), uint64(new))
}

func (x *profAtomic) cas(old, new profIndex) bool {
	return atomic.Cas64((*uint64)(x), uint64(old), uint64(new))
}

func (x profIndex) dataCount() uint32 {
	return uint32(x)
}

func (x profIndex) tagCount() uint32 {
	return uint32(x >> 34)
}

// countSub subtracts two counts obtained from profIndex.dataCount or profIndex.tagCount,
// assuming that they are no more than 2^29 apart (guaranteed since they are never more than
// len(data) or len(tags) apart, respectively).
// tagCount wraps at 2^30, while dataCount wraps at 2^32.
// This function works for both.
func countSub(x, y uint32) int {
	// x-y is 32-bit signed or 30-bit signed; sign-extend to 32 bits and convert to int.
	return int(int32(x-y) << 2 >> 2)
}

// addCountsAndClearFlags returns the packed form of "x + (data, tag) - all flags".
func (x profIndex) addCountsAndClearFlags(data, tag int) profIndex {
	return profIndex((uint64(x)>>34+uint64(uint32(tag)<<2>>2))<<34 | uint64(uint32(x)+uint32(data)))
}

// hasOverflow reports whether b has any overflow records pending.
func (b *profBuf) hasOverflow() bool {
	return uint32(atomic.Load64(&b.overflow)) > 0
}

// takeOverflow consumes the pending overflow records, returning the overflow count
// and the time of the first overflow.
// When called by the reader, it is racing against incrementOverflow.
func (b *profBuf) takeOverflow() (count uint32, time uint64) {
	overflow := atomic.Load64(&b.overflow)
	time = atomic.Load64(&b.overflowTime)
	for {
		count = uint32(overflow)
		if count == 0 {
			time = 0
			break
		}
		// Increment generation, clear overflow count in low bits.
		if atomic.Cas64(&b.overflow, overflow, ((overflow>>32)+1)<<32) {
			break
		}
		overflow = atomic.Load64(&b.overflow)
		time = atomic.Load64(&b.overflowTime)
	}
	return uint32(overflow), time
}

// incrementOverflow records a single overflow at time now.
// It is racing against a possible takeOverflow in the reader.
func (b *profBuf) incrementOverflow(now int64) {
	for {
		overflow := atomic.Load64(&b.overflow)

		// Once we see b.overflow reach 0, it's stable: no one else is changing it underfoot.
		// We need to set overflowTime if we're incrementing b.overflow from 0.
		if uint32(overflow) == 0 {
			// Store overflowTime first so it's always available when overflow != 0.
			atomic.Store64(&b.overflowTime, uint64(now))
			atomic.Store64(&b.overflow, (((overflow>>32)+1)<<32)+1)
			break
		}
		// Otherwise we're racing to increment against reader
		// who wants to set b.overflow to 0.
		// Out of paranoia, leave 2³²-1 a sticky overflow value,
		// to avoid wrapping around. Extremely unlikely.
		if int32(overflow) == -1 {
			break
		}
		if atomic.Cas64(&b.overflow, overflow, overflow+1) {
			break
		}
	}
}

// newProfBuf returns a new profiling buffer with room for
// a header of hdrsize words and a buffer of at least bufwords words.
func newProfBuf(hdrsize, bufwords, tags int) *profBuf {
	if min := 2 + hdrsize + 1; bufwords < min {
		bufwords = min
	}

	// Buffer sizes must be power of two, so that we don't have to
	// worry about uint32 wraparound changing the effective position
	// within the buffers. We store 30 bits of count; limiting to 28
	// gives us some room for intermediate calculations.
	if bufwords >= 1<<28 || tags >= 1<<28 {
		throw("newProfBuf: buffer too large")
	}
	var i int
	for i = 1; i < bufwords; i <<= 1 {
	}
	bufwords = i
	for i = 1; i < tags; i <<= 1 {
	}
	tags = i

	b := new(profBuf)
	b.hdrsize = uintptr(hdrsize)
	b.data = make([]uint64, bufwords)
	b.tags = make([]unsafe.Pointer, tags)
	b.overflowBuf = make([]uint64, 2+b.hdrsize+1)
	return b
}

// canWriteRecord reports whether the buffer has room
// for a single contiguous record with a stack of length nstk.
func (b *profBuf) canWriteRecord(nstk int) bool {
	br := b.r.load()
	bw := b.w.load()

	// room for tag?
	if countSub(br.tagCount(), bw.tagCount())+len(b.tags) < 1 {
		return false
	}

	// room for data?
	nd := countSub(br.dataCount(), bw.dataCount()) + len(b.data)
	want := 2 + int(b.hdrsize) + nstk
	i := int(bw.dataCount() % uint32(len(b.data)))
	if i+want > len(b.data) {
		// Can't fit in trailing fragment of slice.
		// Skip over that and start over at beginning of slice.
		nd -= len(b.data) - i
	}
	return nd >= want
}

// canWriteTwoRecords reports whether the buffer has room
// for two records with stack lengths nstk1, nstk2, in that order.
// Each record must be contiguous on its own, but the two
// records need not be contiguous (one can be at the end of the buffer
// and the other can wrap around and start at the beginning of the buffer).
func (b *profBuf) canWriteTwoRecords(nstk1, nstk2 int) bool {
	br := b.r.load()
	bw := b.w.load()

	// room for tag?
	if countSub(br.tagCount(), bw.tagCount())+len(b.tags) < 2 {
		return false
	}

	// room for data?
	nd := countSub(br.dataCount(), bw.dataCount()) + len(b.data)

	// first record
	want := 2 + int(b.hdrsize) + nstk1
	i := int(bw.dataCount() % uint32(len(b.data)))
	if i+want > len(b.data) {
		// Can't fit in trailing fragment of slice.
		// Skip over that and start over at beginning of slice.
		nd -= len(b.data) - i
		i = 0
	}
	i += want
	nd -= want

	// second record
	want = 2 + int(b.hdrsize) + nstk2
	if i+want > len(b.data) {
		// Can't fit in trailing fragment of slice.
		// Skip over that and start over at beginning of slice.
		nd -= len(b.data) - i
		i = 0
	}
	return nd >= want
}

// write writes an entry to the profiling buffer b.
// The entry begins with a fixed hdr, which must have
// length b.hdrsize, followed by a variable-sized stack
// and a single tag pointer *tagPtr (or nil if tagPtr is nil).
// No write barriers allowed because this might be called from a signal handler.
func (b *profBuf) write(tagPtr *unsafe.Pointer, now int64, hdr []uint64, stk []uintptr) {
	if b == nil {
		return
	}
	if len(hdr) > int(b.hdrsize) {
		throw("misuse of profBuf.write")
	}

	if hasOverflow := b.hasOverflow(); hasOverflow && b.canWriteTwoRecords(1, len(stk)) {
		// Room for both an overflow record and the one being written.
		// Write the overflow record if the reader hasn't gotten to it yet.
		// Only racing against reader, not other writers.
		count, time := b.takeOverflow()
		if count > 0 {
			var stk [1]uintptr
			stk[0] = uintptr(count)
			b.write(nil, int64(time), nil, stk[:])
		}
	} else if hasOverflow || !b.canWriteRecord(len(stk)) {
		// Pending overflow without room to write overflow and new records
		// or no overflow but also no room for new record.
		b.incrementOverflow(now)
		b.wakeupExtra()
		return
	}

	// There's room: write the record.
	br := b.r.load()
	bw := b.w.load()

	// Profiling tag
	//
	// The tag is a pointer, but we can't run a write barrier here.
	// We have interrupted the OS-level execution of gp, but the
	// runtime still sees gp as executing. In effect, we are running
	// in place of the real gp. Since gp is the only goroutine that
	// can overwrite gp.labels, the value of gp.labels is stable during
	// this signal handler: it will still be reachable from gp when
	// we finish executing. If a GC is in progress right now, it must
	// keep gp.labels alive, because gp.labels is reachable from gp.
	// If gp were to overwrite gp.labels, the deletion barrier would
	// still shade that pointer, which would preserve it for the
	// in-progress GC, so all is well. Any future GC will see the
	// value we copied when scanning b.tags (heap-allocated).
	// We arrange that the store here is always overwriting a nil,
	// so there is no need for a deletion barrier on b.tags[wt].
	wt := int(bw.tagCount() % uint32(len(b.tags)))
	if tagPtr != nil {
		*(*uintptr)(unsafe.Pointer(&b.tags[wt])) = uintptr(unsafe.Pointer(*tagPtr))
	}

	// Main record.
	// It has to fit in a contiguous section of the slice, so if it doesn't fit at the end,
	// leave a rewind marker (0) and start over at the beginning of the slice.
	wd := int(bw.dataCount() % uint32(len(b.data)))
	nd := countSub(br.dataCount(), bw.dataCount()) + len(b.data)
	skip := 0
	if wd+2+int(b.hdrsize)+len(stk) > len(b.data) {
		b.data[wd] = 0
		skip = len(b.data) - wd
		nd -= skip
		wd = 0
	}
	data := b.data[wd:]
	data[0] = uint64(2 + b.hdrsize + uintptr(len(stk))) // length
	data[1] = uint64(now)                               // time stamp
	// header, zero-padded
	i := uintptr(copy(data[2:2+b.hdrsize], hdr))
	for ; i < b.hdrsize; i++ {
		data[2+i] = 0
	}
	for i, pc := range stk {
		data[2+b.hdrsize+uintptr(i)] = uint64(pc)
	}

	for {
		// Commit write.
		// Racing with reader setting flag bits in b.w, to avoid lost wakeups.
		old := b.w.load()
		new := old.addCountsAndClearFlags(skip+2+len(stk)+int(b.hdrsize), 1)
		if !b.w.cas(old, new) {
			continue
		}
		// If there was a reader, wake it up.
		if old&profReaderSleeping != 0 {
			notewakeup(&b.wait)
		}
		break
	}
}

// close signals that there will be no more writes on the buffer.
// Once all the data has been read from the buffer, reads will return eof=true.
func (b *profBuf) close() {
	if atomic.Load(&b.eof) > 0 {
		throw("runtime: profBuf already closed")
	}
	atomic.Store(&b.eof, 1)
	b.wakeupExtra()
}

// wakeupExtra must be called after setting one of the "extra"
// atomic fields b.overflow or b.eof.
// It records the change in b.w and wakes up the reader if needed.
func (b *profBuf) wakeupExtra() {
	for {
		old := b.w.load()
		new := old | profWriteExtra
		if !b.w.cas(old, new) {
			continue
		}
		if old&profReaderSleeping != 0 {
			notewakeup(&b.wait)
		}
		break
	}
}

// profBufReadMode specifies whether to block when no data is available to read.
type profBufReadMode int

const (
	profBufBlocking profBufReadMode = iota
	profBufNonBlocking
)

var overflowTag [1]unsafe.Pointer // always nil

func (b *profBuf) read(mode profBufReadMode) (data []uint64, tags []unsafe.Pointer, eof bool) {
	if b == nil {
		return nil, nil, true
	}

	br := b.rNext

	// Commit previous read, returning that part of the ring to the writer.
	// First clear tags that have now been read, both to avoid holding
	// up the memory they point at for longer than necessary
	// and so that b.write can assume it is always overwriting
	// nil tag entries (see comment in b.write).
	rPrev := b.r.load()
	if rPrev != br {
		ntag := countSub(br.tagCount(), rPrev.tagCount())
		ti := int(rPrev.tagCount() % uint32(len(b.tags)))
		for i := 0; i < ntag; i++ {
			b.tags[ti] = nil
			if ti++; ti == len(b.tags) {
				ti = 0
			}
		}
		b.r.store(br)
	}

Read:
	bw := b.w.load()
	numData := countSub(bw.dataCount(), br.dataCount())
	if numData == 0 {
		if b.hasOverflow() {
			// No data to read, but there is overflow to report.
			// Racing with writer flushing b.overflow into a real record.
			count, time := b.takeOverflow()
			if count == 0 {
				// Lost the race, go around again.
				goto Read
			}
			// Won the race, report overflow.
			dst := b.overflowBuf
			dst[0] = uint64(2 + b.hdrsize + 1)
			dst[1] = uint64(time)
			for i := uintptr(0); i < b.hdrsize; i++ {
				dst[2+i] = 0
			}
			dst[2+b.hdrsize] = uint64(count)
			return dst[:2+b.hdrsize+1], overflowTag[:1], false
		}
		if atomic.Load(&b.eof) > 0 {
			// No data, no overflow, EOF set: done.
			return nil, nil, true
		}
		if bw&profWriteExtra != 0 {
			// Writer claims to have published extra information (overflow or eof).
			// Attempt to clear notification and then check again.
			// If we fail to clear the notification it means b.w changed,
			// so we still need to check again.
			b.w.cas(bw, bw&^profWriteExtra)
			goto Read
		}

		// Nothing to read right now.
		// Return or sleep according to mode.
		if mode == profBufNonBlocking {
			return nil, nil, false
		}
		if !b.w.cas(bw, bw|profReaderSleeping) {
			goto Read
		}
		// Committed to sleeping.
		notetsleepg(&b.wait, -1)
		noteclear(&b.wait)
		goto Read
	}
	data = b.data[br.dataCount()%uint32(len(b.data)):]
	if len(data) > numData {
		data = data[:numData]
	} else {
		numData -= len(data) // available in case of wraparound
	}
	skip := 0
	if data[0] == 0 {
		// Wraparound record. Go back to the beginning of the ring.
		skip = len(data)
		data = b.data
		if len(data) > numData {
			data = data[:numData]
		}
	}

	ntag := countSub(bw.tagCount(), br.tagCount())
	if ntag == 0 {
		throw("runtime: malformed profBuf buffer - tag and data out of sync")
	}
	tags = b.tags[br.tagCount()%uint32(len(b.tags)):]
	if len(tags) > ntag {
		tags = tags[:ntag]
	}

	// Count out whole data records until either data or tags is done.
	// They are always in sync in the buffer, but due to an end-of-slice
	// wraparound we might need to stop early and return the rest
	// in the next call.
	di := 0
	ti := 0
	for di < len(data) && data[di] != 0 && ti < len(tags) {
		if uintptr(di)+uintptr(data[di]) > uintptr(len(data)) {
			throw("runtime: malformed profBuf buffer - invalid size")
		}
		di += int(data[di])
		ti++
	}

	// Remember how much we returned, to commit read on next call.
	b.rNext = br.addCountsAndClearFlags(skip+di, ti)

	if raceenabled {
		// Match racereleasemerge in runtime_setProfLabel,
		// so that the setting of the labels in runtime_setProfLabel
		// is treated as happening before any use of the labels
		// by our caller. The synchronization on labelSync itself is a fiction
		// for the race detector. The actual synchronization is handled
		// by the fact that the signal handler only reads from the current
		// goroutine and uses atomics to write the updated queue indices,
		// and then the read-out from the signal handler buffer uses
		// atomics to read those queue indices.
		raceacquire(unsafe.Pointer(&labelSync))
	}

	return data[:di], tags[:ti], false
}
