// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.pagetrace

// Page tracer.
//
// This file contains an implementation of page trace instrumentation for tracking
// the way the Go runtime manages pages of memory. The trace may be enabled at program
// startup with the GODEBUG option pagetrace.
//
// Each page trace event is either 8 or 16 bytes wide. The first
// 8 bytes follow this format for non-sync events:
//
//     [16 timestamp delta][35 base address][10 npages][1 isLarge][2 pageTraceEventType]
//
// If the "large" bit is set then the event is 16 bytes wide with the second 8 byte word
// containing the full npages value (the npages bitfield is 0).
//
// The base address's bottom pageShift bits are always zero hence why we can pack other
// data in there. We ignore the top 16 bits, assuming a 48 bit address space for the
// heap.
//
// The timestamp delta is computed from the difference between the current nanotime
// timestamp and the last sync event's timestamp. The bottom pageTraceTimeLostBits of
// this delta is removed and only the next pageTraceTimeDeltaBits are kept.
//
// A sync event is emitted at the beginning of each trace buffer and whenever the
// timestamp delta would not fit in an event.
//
// Sync events have the following structure:
//
//    [61 timestamp or P ID][1 isPID][2 pageTraceSyncEvent]
//
// In essence, the "large" bit repurposed to indicate whether it's a timestamp or a P ID
// (these are typically uint32). Note that we only have 61 bits for the 64-bit timestamp,
// but like for the delta we drop the bottom pageTraceTimeLostBits here as well.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// pageTraceAlloc records a page trace allocation event.
// pp may be nil. Call only if debug.pagetracefd != 0.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func pageTraceAlloc(pp *p, now int64, base, npages uintptr) {
	if pageTrace.enabled {
		if now == 0 {
			now = nanotime()
		}
		pageTraceEmit(pp, now, base, npages, pageTraceAllocEvent)
	}
}

// pageTraceFree records a page trace free event.
// pp may be nil. Call only if debug.pagetracefd != 0.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func pageTraceFree(pp *p, now int64, base, npages uintptr) {
	if pageTrace.enabled {
		if now == 0 {
			now = nanotime()
		}
		pageTraceEmit(pp, now, base, npages, pageTraceFreeEvent)
	}
}

// pageTraceScav records a page trace scavenge event.
// pp may be nil. Call only if debug.pagetracefd != 0.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func pageTraceScav(pp *p, now int64, base, npages uintptr) {
	if pageTrace.enabled {
		if now == 0 {
			now = nanotime()
		}
		pageTraceEmit(pp, now, base, npages, pageTraceScavEvent)
	}
}

// pageTraceEventType is a page trace event type.
type pageTraceEventType uint8

const (
	pageTraceSyncEvent  pageTraceEventType = iota // Timestamp emission.
	pageTraceAllocEvent                           // Allocation of pages.
	pageTraceFreeEvent                            // Freeing pages.
	pageTraceScavEvent                            // Scavenging pages.
)

// pageTraceEmit emits a page trace event.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func pageTraceEmit(pp *p, now int64, base, npages uintptr, typ pageTraceEventType) {
	// Get a buffer.
	var tbp *pageTraceBuf
	pid := int32(-1)
	if pp == nil {
		// We have no P, so take the global buffer.
		lock(&pageTrace.lock)
		tbp = &pageTrace.buf
	} else {
		tbp = &pp.pageTraceBuf
		pid = pp.id
	}

	// Initialize the buffer if necessary.
	tb := *tbp
	if tb.buf == nil {
		tb.buf = (*pageTraceEvents)(sysAlloc(pageTraceBufSize, &memstats.other_sys))
		tb = tb.writePid(pid)
	}

	// Handle timestamp and emit a sync event if necessary.
	if now < tb.timeBase {
		now = tb.timeBase
	}
	if now-tb.timeBase >= pageTraceTimeMaxDelta {
		tb.timeBase = now
		tb = tb.writeSync(pid)
	}

	// Emit the event.
	tb = tb.writeEvent(pid, now, base, npages, typ)

	// Write back the buffer.
	*tbp = tb
	if pp == nil {
		unlock(&pageTrace.lock)
	}
}

const (
	pageTraceBufSize = 32 << 10

	// These constants describe the per-event timestamp delta encoding.
	pageTraceTimeLostBits  = 7  // How many bits of precision we lose in the delta.
	pageTraceTimeDeltaBits = 16 // Size of the delta in bits.
	pageTraceTimeMaxDelta  = 1 << (pageTraceTimeLostBits + pageTraceTimeDeltaBits)
)

// pageTraceEvents is the low-level buffer containing the trace data.
type pageTraceEvents struct {
	_      sys.NotInHeap
	events [pageTraceBufSize / 8]uint64
}

// pageTraceBuf is a wrapper around pageTraceEvents that knows how to write events
// to the buffer. It tracks state necessary to do so.
type pageTraceBuf struct {
	buf      *pageTraceEvents
	len      int   // How many events have been written so far.
	timeBase int64 // The current timestamp base from which deltas are produced.
	finished bool  // Whether this trace buf should no longer flush anything out.
}

// writePid writes a P ID event indicating which P we're running on.
//
// Assumes there's always space in the buffer since this is only called at the
// beginning of a new buffer.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func (tb pageTraceBuf) writePid(pid int32) pageTraceBuf {
	e := uint64(int64(pid))<<3 | 0b100 | uint64(pageTraceSyncEvent)
	tb.buf.events[tb.len] = e
	tb.len++
	return tb
}

// writeSync writes a sync event, which is just a timestamp. Handles flushing.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func (tb pageTraceBuf) writeSync(pid int32) pageTraceBuf {
	if tb.len+1 > len(tb.buf.events) {
		// N.B. flush will writeSync again.
		return tb.flush(pid, tb.timeBase)
	}
	e := ((uint64(tb.timeBase) >> pageTraceTimeLostBits) << 3) | uint64(pageTraceSyncEvent)
	tb.buf.events[tb.len] = e
	tb.len++
	return tb
}

// writeEvent handles writing all non-sync and non-pid events. Handles flushing if necessary.
//
// pid indicates the P we're currently running on. Necessary in case we need to flush.
// now is the current nanotime timestamp.
// base is the base address of whatever group of pages this event is happening to.
// npages is the length of the group of pages this event is happening to.
// typ is the event that's happening to these pages.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func (tb pageTraceBuf) writeEvent(pid int32, now int64, base, npages uintptr, typ pageTraceEventType) pageTraceBuf {
	large := 0
	np := npages
	if npages >= 1024 {
		large = 1
		np = 0
	}
	if tb.len+1+large > len(tb.buf.events) {
		tb = tb.flush(pid, now)
	}
	if base%pageSize != 0 {
		throw("base address not page aligned")
	}
	e := uint64(base)
	// The pageShift low-order bits are zero.
	e |= uint64(typ)        // 2 bits
	e |= uint64(large) << 2 // 1 bit
	e |= uint64(np) << 3    // 10 bits
	// Write the timestamp delta in the upper pageTraceTimeDeltaBits.
	e |= uint64((now-tb.timeBase)>>pageTraceTimeLostBits) << (64 - pageTraceTimeDeltaBits)
	tb.buf.events[tb.len] = e
	if large != 0 {
		// npages doesn't fit in 10 bits, so write an additional word with that data.
		tb.buf.events[tb.len+1] = uint64(npages)
	}
	tb.len += 1 + large
	return tb
}

// flush writes out the contents of the buffer to pageTrace.fd and resets the buffer.
// It then writes out a P ID event and the first sync event for the new buffer.
//
// Must run on the system stack as a crude way to prevent preemption.
//
//go:systemstack
func (tb pageTraceBuf) flush(pid int32, now int64) pageTraceBuf {
	if !tb.finished {
		lock(&pageTrace.fdLock)
		writeFull(uintptr(pageTrace.fd), (*byte)(unsafe.Pointer(&tb.buf.events[0])), tb.len*8)
		unlock(&pageTrace.fdLock)
	}
	tb.len = 0
	tb.timeBase = now
	return tb.writePid(pid).writeSync(pid)
}

var pageTrace struct {
	// enabled indicates whether tracing is enabled. If true, fd >= 0.
	//
	// Safe to read without synchronization because it's only set once
	// at program initialization.
	enabled bool

	// buf is the page trace buffer used if there is no P.
	//
	// lock protects buf.
	lock mutex
	buf  pageTraceBuf

	// fdLock protects writing to fd.
	//
	// fd is the file to write the page trace to.
	fdLock mutex
	fd     int32
}

// initPageTrace initializes the page tracing infrastructure from GODEBUG.
//
// env must be the value of the GODEBUG environment variable.
func initPageTrace(env string) {
	var value string
	for env != "" {
		elt, rest := env, ""
		for i := 0; i < len(env); i++ {
			if env[i] == ',' {
				elt, rest = env[:i], env[i+1:]
				break
			}
		}
		env = rest
		if hasPrefix(elt, "pagetrace=") {
			value = elt[len("pagetrace="):]
			break
		}
	}
	pageTrace.fd = -1
	if canCreateFile && value != "" {
		var tmp [4096]byte
		if len(value) != 0 && len(value) < 4096 {
			copy(tmp[:], value)
			pageTrace.fd = create(&tmp[0], 0o664)
		}
	}
	pageTrace.enabled = pageTrace.fd >= 0
}

// finishPageTrace flushes all P's trace buffers and disables page tracing.
func finishPageTrace() {
	if !pageTrace.enabled {
		return
	}
	// Grab worldsema as we're about to execute a ragged barrier.
	semacquire(&worldsema)
	systemstack(func() {
		// Disable tracing. This isn't strictly necessary and it's best-effort.
		pageTrace.enabled = false

		// Execute a ragged barrier, flushing each trace buffer.
		forEachP(waitReasonPageTraceFlush, func(pp *p) {
			if pp.pageTraceBuf.buf != nil {
				pp.pageTraceBuf = pp.pageTraceBuf.flush(pp.id, nanotime())
			}
			pp.pageTraceBuf.finished = true
		})

		// Write the global have-no-P buffer.
		lock(&pageTrace.lock)
		if pageTrace.buf.buf != nil {
			pageTrace.buf = pageTrace.buf.flush(-1, nanotime())
		}
		pageTrace.buf.finished = true
		unlock(&pageTrace.lock)

		// Safely close the file as nothing else should be allowed to write to the fd.
		lock(&pageTrace.fdLock)
		closefd(pageTrace.fd)
		pageTrace.fd = -1
		unlock(&pageTrace.fdLock)
	})
	semrelease(&worldsema)
}

// writeFull ensures that a complete write of bn bytes from b is made to fd.
func writeFull(fd uintptr, b *byte, bn int) {
	for bn > 0 {
		n := write(fd, unsafe.Pointer(b), int32(bn))
		if n == -_EINTR || n == -_EAGAIN {
			continue
		}
		if n < 0 {
			print("errno=", -n, "\n")
			throw("writeBytes: bad write")
		}
		bn -= int(n)
		b = addb(b, uintptr(n))
	}
}
