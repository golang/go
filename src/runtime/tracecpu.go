// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CPU profile -> trace

package runtime

// traceInitReadCPU initializes CPU profile -> tracer state for tracing.
//
// Returns a profBuf for reading from.
func traceInitReadCPU() {
	if traceEnabled() {
		throw("traceInitReadCPU called with trace enabled")
	}
	// Create new profBuf for CPU samples that will be emitted as events.
	// Format: after the timestamp, header is [pp.id, gp.goid, mp.procid].
	trace.cpuLogRead[0] = newProfBuf(3, profBufWordCount, profBufTagCount)
	trace.cpuLogRead[1] = newProfBuf(3, profBufWordCount, profBufTagCount)
	// We must not acquire trace.signalLock outside of a signal handler: a
	// profiling signal may arrive at any time and try to acquire it, leading to
	// deadlock. Because we can't use that lock to protect updates to
	// trace.cpuLogWrite (only use of the structure it references), reads and
	// writes of the pointer must be atomic. (And although this field is never
	// the sole pointer to the profBuf value, it's best to allow a write barrier
	// here.)
	trace.cpuLogWrite[0].Store(trace.cpuLogRead[0])
	trace.cpuLogWrite[1].Store(trace.cpuLogRead[1])
}

// traceStartReadCPU creates a goroutine to start reading CPU profile
// data into an active trace.
//
// traceAdvanceSema must be held.
func traceStartReadCPU() {
	if !traceEnabled() {
		throw("traceStartReadCPU called with trace disabled")
	}
	// Spin up the logger goroutine.
	trace.cpuSleep = newWakeableSleep()
	done := make(chan struct{}, 1)
	go func() {
		for traceEnabled() {
			// Sleep here because traceReadCPU is non-blocking. This mirrors
			// how the runtime/pprof package obtains CPU profile data.
			//
			// We can't do a blocking read here because Darwin can't do a
			// wakeup from a signal handler, so all CPU profiling is just
			// non-blocking. See #61768 for more details.
			//
			// Like the runtime/pprof package, even if that bug didn't exist
			// we would still want to do a goroutine-level sleep in between
			// reads to avoid frequent wakeups.
			trace.cpuSleep.sleep(100_000_000)

			tl := traceAcquire()
			if !tl.ok() {
				// Tracing disabled.
				break
			}
			keepGoing := traceReadCPU(tl.gen)
			traceRelease(tl)
			if !keepGoing {
				break
			}
		}
		done <- struct{}{}
	}()
	trace.cpuLogDone = done
}

// traceStopReadCPU blocks until the trace CPU reading goroutine exits.
//
// traceAdvanceSema must be held, and tracing must be disabled.
func traceStopReadCPU() {
	if traceEnabled() {
		throw("traceStopReadCPU called with trace enabled")
	}

	// Once we close the profbuf, we'll be in one of two situations:
	// - The logger goroutine has already exited because it observed
	//   that the trace is disabled.
	// - The logger goroutine is asleep.
	//
	// Wake the goroutine so it can observe that their the buffer is
	// closed an exit.
	trace.cpuLogWrite[0].Store(nil)
	trace.cpuLogWrite[1].Store(nil)
	trace.cpuLogRead[0].close()
	trace.cpuLogRead[1].close()
	trace.cpuSleep.wake()

	// Wait until the logger goroutine exits.
	<-trace.cpuLogDone

	// Clear state for the next trace.
	trace.cpuLogDone = nil
	trace.cpuLogRead[0] = nil
	trace.cpuLogRead[1] = nil
	trace.cpuSleep.close()
}

// traceReadCPU attempts to read from the provided profBuf[gen%2] and write
// into the trace. Returns true if there might be more to read or false
// if the profBuf is closed or the caller should otherwise stop reading.
//
// The caller is responsible for ensuring that gen does not change. Either
// the caller must be in a traceAcquire/traceRelease block, or must be calling
// with traceAdvanceSema held.
//
// No more than one goroutine may be in traceReadCPU for the same
// profBuf at a time.
//
// Must not run on the system stack because profBuf.read performs race
// operations.
func traceReadCPU(gen uintptr) bool {
	var pcBuf [traceStackSize]uintptr

	data, tags, eof := trace.cpuLogRead[gen%2].read(profBufNonBlocking)
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

		// Deserialize the data in the profile buffer.
		recordLen := data[0]
		timestamp := data[1]
		ppid := data[2] >> 1
		if hasP := (data[2] & 0b1) != 0; !hasP {
			ppid = ^uint64(0)
		}
		goid := data[3]
		mpid := data[4]
		stk := data[5:recordLen]

		// Overflow records always have their headers contain
		// all zeroes.
		isOverflowRecord := len(stk) == 1 && data[2] == 0 && data[3] == 0 && data[4] == 0

		// Move the data iterator forward.
		data = data[recordLen:]
		// No support here for reporting goroutine tags at the moment; if
		// that information is to be part of the execution trace, we'd
		// probably want to see when the tags are applied and when they
		// change, instead of only seeing them when we get a CPU sample.
		tags = tags[1:]

		if isOverflowRecord {
			// Looks like an overflow record from the profBuf. Not much to
			// do here, we only want to report full records.
			continue
		}

		// Construct the stack for insertion to the stack table.
		nstk := 1
		pcBuf[0] = logicalStackSentinel
		for ; nstk < len(pcBuf) && nstk-1 < len(stk); nstk++ {
			pcBuf[nstk] = uintptr(stk[nstk-1])
		}

		// Write out a trace event.
		w := unsafeTraceWriter(gen, trace.cpuBuf[gen%2])

		// Ensure we have a place to write to.
		var flushed bool
		w, flushed = w.ensure(2 + 5*traceBytesPerNumber /* traceEvCPUSamples + traceEvCPUSample + timestamp + g + m + p + stack ID */)
		if flushed {
			// Annotate the batch as containing strings.
			w.byte(byte(traceEvCPUSamples))
		}

		// Add the stack to the table.
		stackID := trace.stackTab[gen%2].put(pcBuf[:nstk])

		// Write out the CPU sample.
		w.byte(byte(traceEvCPUSample))
		w.varint(timestamp)
		w.varint(mpid)
		w.varint(ppid)
		w.varint(goid)
		w.varint(stackID)

		trace.cpuBuf[gen%2] = w.traceBuf
	}
	return !eof
}

// traceCPUFlush flushes trace.cpuBuf[gen%2]. The caller must be certain that gen
// has completed and that there are no more writers to it.
func traceCPUFlush(gen uintptr) {
	// Flush any remaining trace buffers containing CPU samples.
	if buf := trace.cpuBuf[gen%2]; buf != nil {
		systemstack(func() {
			lock(&trace.lock)
			traceBufFlush(buf, gen)
			unlock(&trace.lock)
			trace.cpuBuf[gen%2] = nil
		})
	}
}

// traceCPUSample writes a CPU profile sample stack to the execution tracer's
// profiling buffer. It is called from a signal handler, so is limited in what
// it can do. mp must be the thread that is currently stopped in a signal.
func traceCPUSample(gp *g, mp *m, pp *p, stk []uintptr) {
	if !traceEnabled() {
		// Tracing is usually turned off; don't spend time acquiring the signal
		// lock unless it's active.
		return
	}
	if mp == nil {
		// Drop samples that don't have an identifiable thread. We can't render
		// this in any useful way anyway.
		return
	}

	// We're going to conditionally write to one of two buffers based on the
	// generation. To make sure we write to the correct one, we need to make
	// sure this thread's trace seqlock is held. If it already is, then we're
	// in the tracer and we can just take advantage of that. If it isn't, then
	// we need to acquire it and read the generation.
	locked := false
	if mp.trace.seqlock.Load()%2 == 0 {
		mp.trace.seqlock.Add(1)
		locked = true
	}
	gen := trace.gen.Load()
	if gen == 0 {
		// Tracing is disabled, as it turns out. Release the seqlock if necessary
		// and exit.
		if locked {
			mp.trace.seqlock.Add(1)
		}
		return
	}

	now := traceClockNow()
	// The "header" here is the ID of the M that was running the profiled code,
	// followed by the IDs of the P and goroutine. (For normal CPU profiling, it's
	// usually the number of samples with the given stack.) Near syscalls, pp
	// may be nil. Reporting goid of 0 is fine for either g0 or a nil gp.
	var hdr [3]uint64
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
	hdr[2] = uint64(mp.procid)

	// Allow only one writer at a time
	for !trace.signalLock.CompareAndSwap(0, 1) {
		// TODO: Is it safe to osyield here? https://go.dev/issue/52672
		osyield()
	}

	if log := trace.cpuLogWrite[gen%2].Load(); log != nil {
		// Note: we don't pass a tag pointer here (how should profiling tags
		// interact with the execution tracer?), but if we did we'd need to be
		// careful about write barriers. See the long comment in profBuf.write.
		log.write(nil, int64(now), hdr[:], stk)
	}

	trace.signalLock.Store(0)

	// Release the seqlock if we acquired it earlier.
	if locked {
		mp.trace.seqlock.Add(1)
	}
}
