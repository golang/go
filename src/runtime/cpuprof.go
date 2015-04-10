// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CPU profiling.
// Based on algorithms and data structures used in
// http://code.google.com/p/google-perftools/.
//
// The main difference between this code and the google-perftools
// code is that this code is written to allow copying the profile data
// to an arbitrary io.Writer, while the google-perftools code always
// writes to an operating system file.
//
// The signal handler for the profiling clock tick adds a new stack trace
// to a hash table tracking counts for recent traces.  Most clock ticks
// hit in the cache.  In the event of a cache miss, an entry must be
// evicted from the hash table, copied to a log that will eventually be
// written as profile data.  The google-perftools code flushed the
// log itself during the signal handler.  This code cannot do that, because
// the io.Writer might block or need system calls or locks that are not
// safe to use from within the signal handler.  Instead, we split the log
// into two halves and let the signal handler fill one half while a goroutine
// is writing out the other half.  When the signal handler fills its half, it
// offers to swap with the goroutine.  If the writer is not done with its half,
// we lose the stack trace for this clock tick (and record that loss).
// The goroutine interacts with the signal handler by calling getprofile() to
// get the next log piece to write, implicitly handing back the last log
// piece it obtained.
//
// The state of this dance between the signal handler and the goroutine
// is encoded in the Profile.handoff field.  If handoff == 0, then the goroutine
// is not using either log half and is waiting (or will soon be waiting) for
// a new piece by calling notesleep(&p.wait).  If the signal handler
// changes handoff from 0 to non-zero, it must call notewakeup(&p.wait)
// to wake the goroutine.  The value indicates the number of entries in the
// log half being handed off.  The goroutine leaves the non-zero value in
// place until it has finished processing the log half and then flips the number
// back to zero.  Setting the high bit in handoff means that the profiling is over,
// and the goroutine is now in charge of flushing the data left in the hash table
// to the log and returning that data.
//
// The handoff field is manipulated using atomic operations.
// For the most part, the manipulation of handoff is orderly: if handoff == 0
// then the signal handler owns it and can change it to non-zero.
// If handoff != 0 then the goroutine owns it and can change it to zero.
// If that were the end of the story then we would not need to manipulate
// handoff using atomic operations.  The operations are needed, however,
// in order to let the log closer set the high bit to indicate "EOF" safely
// in the situation when normally the goroutine "owns" handoff.

package runtime

import "unsafe"

const (
	numBuckets      = 1 << 10
	logSize         = 1 << 17
	assoc           = 4
	maxCPUProfStack = 64
)

type cpuprofEntry struct {
	count uintptr
	depth int
	stack [maxCPUProfStack]uintptr
}

type cpuProfile struct {
	on     bool    // profiling is on
	wait   note    // goroutine waits here
	count  uintptr // tick count
	evicts uintptr // eviction count
	lost   uintptr // lost ticks that need to be logged

	// Active recent stack traces.
	hash [numBuckets]struct {
		entry [assoc]cpuprofEntry
	}

	// Log of traces evicted from hash.
	// Signal handler has filled log[toggle][:nlog].
	// Goroutine is writing log[1-toggle][:handoff].
	log     [2][logSize / 2]uintptr
	nlog    int
	toggle  int32
	handoff uint32

	// Writer state.
	// Writer maintains its own toggle to avoid races
	// looking at signal handler's toggle.
	wtoggle  uint32
	wholding bool // holding & need to release a log half
	flushing bool // flushing hash table - profile is over
	eodSent  bool // special end-of-data record sent; => flushing
}

var (
	cpuprofLock mutex
	cpuprof     *cpuProfile

	eod = [3]uintptr{0, 1, 0}
)

func setcpuprofilerate(hz int32) {
	systemstack(func() {
		setcpuprofilerate_m(hz)
	})
}

// lostProfileData is a no-op function used in profiles
// to mark the number of profiling stack traces that were
// discarded due to slow data writers.
func lostProfileData() {}

// SetCPUProfileRate sets the CPU profiling rate to hz samples per second.
// If hz <= 0, SetCPUProfileRate turns off profiling.
// If the profiler is on, the rate cannot be changed without first turning it off.
//
// Most clients should use the runtime/pprof package or
// the testing package's -test.cpuprofile flag instead of calling
// SetCPUProfileRate directly.
func SetCPUProfileRate(hz int) {
	// Clamp hz to something reasonable.
	if hz < 0 {
		hz = 0
	}
	if hz > 1000000 {
		hz = 1000000
	}

	lock(&cpuprofLock)
	if hz > 0 {
		if cpuprof == nil {
			cpuprof = (*cpuProfile)(sysAlloc(unsafe.Sizeof(cpuProfile{}), &memstats.other_sys))
			if cpuprof == nil {
				print("runtime: cpu profiling cannot allocate memory\n")
				unlock(&cpuprofLock)
				return
			}
		}
		if cpuprof.on || cpuprof.handoff != 0 {
			print("runtime: cannot set cpu profile rate until previous profile has finished.\n")
			unlock(&cpuprofLock)
			return
		}

		cpuprof.on = true
		// pprof binary header format.
		// http://code.google.com/p/google-perftools/source/browse/trunk/src/profiledata.cc#117
		p := &cpuprof.log[0]
		p[0] = 0                 // count for header
		p[1] = 3                 // depth for header
		p[2] = 0                 // version number
		p[3] = uintptr(1e6 / hz) // period (microseconds)
		p[4] = 0
		cpuprof.nlog = 5
		cpuprof.toggle = 0
		cpuprof.wholding = false
		cpuprof.wtoggle = 0
		cpuprof.flushing = false
		cpuprof.eodSent = false
		noteclear(&cpuprof.wait)

		setcpuprofilerate(int32(hz))
	} else if cpuprof != nil && cpuprof.on {
		setcpuprofilerate(0)
		cpuprof.on = false

		// Now add is not running anymore, and getprofile owns the entire log.
		// Set the high bit in cpuprof.handoff to tell getprofile.
		for {
			n := cpuprof.handoff
			if n&0x80000000 != 0 {
				print("runtime: setcpuprofile(off) twice\n")
			}
			if cas(&cpuprof.handoff, n, n|0x80000000) {
				if n == 0 {
					// we did the transition from 0 -> nonzero so we wake getprofile
					notewakeup(&cpuprof.wait)
				}
				break
			}
		}
	}
	unlock(&cpuprofLock)
}

// add adds the stack trace to the profile.
// It is called from signal handlers and other limited environments
// and cannot allocate memory or acquire locks that might be
// held at the time of the signal, nor can it use substantial amounts
// of stack.  It is allowed to call evict.
func (p *cpuProfile) add(pc []uintptr) {
	if len(pc) > maxCPUProfStack {
		pc = pc[:maxCPUProfStack]
	}

	// Compute hash.
	h := uintptr(0)
	for _, x := range pc {
		h = h<<8 | (h >> (8 * (unsafe.Sizeof(h) - 1)))
		h += x * 41
	}
	p.count++

	// Add to entry count if already present in table.
	b := &p.hash[h%numBuckets]
Assoc:
	for i := range b.entry {
		e := &b.entry[i]
		if e.depth != len(pc) {
			continue
		}
		for j := range pc {
			if e.stack[j] != pc[j] {
				continue Assoc
			}
		}
		e.count++
		return
	}

	// Evict entry with smallest count.
	var e *cpuprofEntry
	for i := range b.entry {
		if e == nil || b.entry[i].count < e.count {
			e = &b.entry[i]
		}
	}
	if e.count > 0 {
		if !p.evict(e) {
			// Could not evict entry.  Record lost stack.
			p.lost++
			return
		}
		p.evicts++
	}

	// Reuse the newly evicted entry.
	e.depth = len(pc)
	e.count = 1
	copy(e.stack[:], pc)
}

// evict copies the given entry's data into the log, so that
// the entry can be reused.  evict is called from add, which
// is called from the profiling signal handler, so it must not
// allocate memory or block.  It is safe to call flushlog.
// evict returns true if the entry was copied to the log,
// false if there was no room available.
func (p *cpuProfile) evict(e *cpuprofEntry) bool {
	d := e.depth
	nslot := d + 2
	log := &p.log[p.toggle]
	if p.nlog+nslot > len(log) {
		if !p.flushlog() {
			return false
		}
		log = &p.log[p.toggle]
	}

	q := p.nlog
	log[q] = e.count
	q++
	log[q] = uintptr(d)
	q++
	copy(log[q:], e.stack[:d])
	q += d
	p.nlog = q
	e.count = 0
	return true
}

// flushlog tries to flush the current log and switch to the other one.
// flushlog is called from evict, called from add, called from the signal handler,
// so it cannot allocate memory or block.  It can try to swap logs with
// the writing goroutine, as explained in the comment at the top of this file.
func (p *cpuProfile) flushlog() bool {
	if !cas(&p.handoff, 0, uint32(p.nlog)) {
		return false
	}
	notewakeup(&p.wait)

	p.toggle = 1 - p.toggle
	log := &p.log[p.toggle]
	q := 0
	if p.lost > 0 {
		lostPC := funcPC(lostProfileData)
		log[0] = p.lost
		log[1] = 1
		log[2] = lostPC
		q = 3
		p.lost = 0
	}
	p.nlog = q
	return true
}

// getprofile blocks until the next block of profiling data is available
// and returns it as a []byte.  It is called from the writing goroutine.
func (p *cpuProfile) getprofile() []byte {
	if p == nil {
		return nil
	}

	if p.wholding {
		// Release previous log to signal handling side.
		// Loop because we are racing against SetCPUProfileRate(0).
		for {
			n := p.handoff
			if n == 0 {
				print("runtime: phase error during cpu profile handoff\n")
				return nil
			}
			if n&0x80000000 != 0 {
				p.wtoggle = 1 - p.wtoggle
				p.wholding = false
				p.flushing = true
				goto Flush
			}
			if cas(&p.handoff, n, 0) {
				break
			}
		}
		p.wtoggle = 1 - p.wtoggle
		p.wholding = false
	}

	if p.flushing {
		goto Flush
	}

	if !p.on && p.handoff == 0 {
		return nil
	}

	// Wait for new log.
	notetsleepg(&p.wait, -1)
	noteclear(&p.wait)

	switch n := p.handoff; {
	case n == 0:
		print("runtime: phase error during cpu profile wait\n")
		return nil
	case n == 0x80000000:
		p.flushing = true
		goto Flush
	default:
		n &^= 0x80000000

		// Return new log to caller.
		p.wholding = true

		return uintptrBytes(p.log[p.wtoggle][:n])
	}

	// In flush mode.
	// Add is no longer being called.  We own the log.
	// Also, p.handoff is non-zero, so flushlog will return false.
	// Evict the hash table into the log and return it.
Flush:
	for i := range p.hash {
		b := &p.hash[i]
		for j := range b.entry {
			e := &b.entry[j]
			if e.count > 0 && !p.evict(e) {
				// Filled the log.  Stop the loop and return what we've got.
				break Flush
			}
		}
	}

	// Return pending log data.
	if p.nlog > 0 {
		// Note that we're using toggle now, not wtoggle,
		// because we're working on the log directly.
		n := p.nlog
		p.nlog = 0
		return uintptrBytes(p.log[p.toggle][:n])
	}

	// Made it through the table without finding anything to log.
	if !p.eodSent {
		// We may not have space to append this to the partial log buf,
		// so we always return a new slice for the end-of-data marker.
		p.eodSent = true
		return uintptrBytes(eod[:])
	}

	// Finally done.  Clean up and return nil.
	p.flushing = false
	if !cas(&p.handoff, p.handoff, 0) {
		print("runtime: profile flush racing with something\n")
	}
	return nil
}

func uintptrBytes(p []uintptr) (ret []byte) {
	pp := (*slice)(unsafe.Pointer(&p))
	rp := (*slice)(unsafe.Pointer(&ret))

	rp.array = pp.array
	rp.len = pp.len * int(unsafe.Sizeof(p[0]))
	rp.cap = rp.len

	return
}

// CPUProfile returns the next chunk of binary CPU profiling stack trace data,
// blocking until data is available.  If profiling is turned off and all the profile
// data accumulated while it was on has been returned, CPUProfile returns nil.
// The caller must save the returned data before calling CPUProfile again.
//
// Most clients should use the runtime/pprof package or
// the testing package's -test.cpuprofile flag instead of calling
// CPUProfile directly.
func CPUProfile() []byte {
	return cpuprof.getprofile()
}

//go:linkname runtime_pprof_runtime_cyclesPerSecond runtime/pprof.runtime_cyclesPerSecond
func runtime_pprof_runtime_cyclesPerSecond() int64 {
	return tickspersecond()
}
