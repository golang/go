// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package oldtrace implements a parser for Go execution traces from versions
// 1.11–1.21.
//
// The package started as a copy of Go 1.19's internal/trace, but has been
// optimized to be faster while using less memory and fewer allocations. It has
// been further modified for the specific purpose of converting traces to the
// new 1.22+ format.
package oldtrace

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"internal/trace/event"
	"internal/trace/version"
	"io"
	"math"
	"slices"
	"sort"
)

// Timestamp represents a count of nanoseconds since the beginning of the trace.
// They can only be meaningfully compared with other timestamps from the same
// trace.
type Timestamp int64

// Event describes one event in the trace.
type Event struct {
	// The Event type is carefully laid out to optimize its size and to avoid
	// pointers, the latter so that the garbage collector won't have to scan any
	// memory of our millions of events.

	Ts    Timestamp  // timestamp in nanoseconds
	G     uint64     // G on which the event happened
	Args  [4]uint64  // event-type-specific arguments
	StkID uint32     // unique stack ID
	P     int32      // P on which the event happened (can be a real P or one of TimerP, NetpollP, SyscallP)
	Type  event.Type // one of Ev*
}

// Frame is a frame in stack traces.
type Frame struct {
	PC uint64
	// string ID of the function name
	Fn uint64
	// string ID of the file name
	File uint64
	Line int
}

const (
	// Special P identifiers:
	FakeP    = 1000000 + iota
	TimerP   // contains timer unblocks
	NetpollP // contains network unblocks
	SyscallP // contains returns from syscalls
	GCP      // contains GC state
	ProfileP // contains recording of CPU profile samples
)

// Trace is the result of Parse.
type Trace struct {
	Version version.Version

	// Events is the sorted list of Events in the trace.
	Events Events
	// Stacks is the stack traces (stored as slices of PCs), keyed by stack IDs
	// from the trace.
	Stacks        map[uint32][]uint64
	PCs           map[uint64]Frame
	Strings       map[uint64]string
	InlineStrings []string
}

// batchOffset records the byte offset of, and number of events in, a batch. A
// batch is a sequence of events emitted by a P. Events within a single batch
// are sorted by time.
type batchOffset struct {
	offset    int
	numEvents int
}

type parser struct {
	ver  version.Version
	data []byte
	off  int

	strings              map[uint64]string
	inlineStrings        []string
	inlineStringsMapping map[string]int
	// map from Ps to their batch offsets
	batchOffsets map[int32][]batchOffset
	stacks       map[uint32][]uint64
	stacksData   []uint64
	ticksPerSec  int64
	pcs          map[uint64]Frame
	cpuSamples   []Event
	timerGoids   map[uint64]bool

	// state for readRawEvent
	args []uint64

	// state for parseEvent
	lastTs Timestamp
	lastG  uint64
	// map from Ps to the last Gs that ran on them
	lastGs map[int32]uint64
	lastP  int32
}

func (p *parser) discard(n uint64) bool {
	if n > math.MaxInt {
		return false
	}
	if noff := p.off + int(n); noff < p.off || noff > len(p.data) {
		return false
	} else {
		p.off = noff
	}
	return true
}

func newParser(r io.Reader, ver version.Version) (*parser, error) {
	var buf []byte
	if seeker, ok := r.(io.Seeker); ok {
		// Determine the size of the reader so that we can allocate a buffer
		// without having to grow it later.
		cur, err := seeker.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, err
		}
		end, err := seeker.Seek(0, io.SeekEnd)
		if err != nil {
			return nil, err
		}
		_, err = seeker.Seek(cur, io.SeekStart)
		if err != nil {
			return nil, err
		}

		buf = make([]byte, end-cur)
		_, err = io.ReadFull(r, buf)
		if err != nil {
			return nil, err
		}
	} else {
		var err error
		buf, err = io.ReadAll(r)
		if err != nil {
			return nil, err
		}
	}
	return &parser{data: buf, ver: ver, timerGoids: make(map[uint64]bool)}, nil
}

// Parse parses Go execution traces from versions 1.11–1.21. The provided reader
// will be read to completion and the entire trace will be materialized in
// memory. That is, this function does not allow incremental parsing.
//
// The reader has to be positioned just after the trace header and vers needs to
// be the version of the trace. This can be achieved by using
// version.ReadHeader.
func Parse(r io.Reader, vers version.Version) (Trace, error) {
	// We accept the version as an argument because internal/trace/v2 will have
	// already read the version to determine which parser to use.
	p, err := newParser(r, vers)
	if err != nil {
		return Trace{}, err
	}
	return p.parse()
}

// parse parses, post-processes and verifies the trace.
func (p *parser) parse() (Trace, error) {
	defer func() {
		p.data = nil
	}()

	// We parse a trace by running the following steps in order:
	//
	// 1. In the initial pass we collect information about batches (their
	//    locations and sizes.) We also parse CPU profiling samples in this
	//    step, simply to reduce the number of full passes that we need.
	//
	// 2. In the second pass we parse batches and merge them into a globally
	//    ordered event stream. This uses the batch information from the first
	//    pass to quickly find batches.
	//
	// 3. After all events have been parsed we convert their timestamps from CPU
	//    ticks to wall time. Furthermore we move timers and syscalls to
	//    dedicated, fake Ps.
	//
	// 4. Finally, we validate the trace.

	p.strings = make(map[uint64]string)
	p.batchOffsets = make(map[int32][]batchOffset)
	p.lastGs = make(map[int32]uint64)
	p.stacks = make(map[uint32][]uint64)
	p.pcs = make(map[uint64]Frame)
	p.inlineStringsMapping = make(map[string]int)

	if err := p.collectBatchesAndCPUSamples(); err != nil {
		return Trace{}, err
	}

	events, err := p.parseEventBatches()
	if err != nil {
		return Trace{}, err
	}

	if p.ticksPerSec == 0 {
		return Trace{}, errors.New("no EvFrequency event")
	}

	if events.Len() > 0 {
		// Translate cpu ticks to real time.
		minTs := events.Ptr(0).Ts
		// Use floating point to avoid integer overflows.
		freq := 1e9 / float64(p.ticksPerSec)
		for i := 0; i < events.Len(); i++ {
			ev := events.Ptr(i)
			ev.Ts = Timestamp(float64(ev.Ts-minTs) * freq)
			// Move timers and syscalls to separate fake Ps.
			if p.timerGoids[ev.G] && ev.Type == EvGoUnblock {
				ev.P = TimerP
			}
			if ev.Type == EvGoSysExit {
				ev.P = SyscallP
			}
		}
	}

	if err := p.postProcessTrace(events); err != nil {
		return Trace{}, err
	}

	res := Trace{
		Version:       p.ver,
		Events:        events,
		Stacks:        p.stacks,
		Strings:       p.strings,
		InlineStrings: p.inlineStrings,
		PCs:           p.pcs,
	}
	return res, nil
}

// rawEvent is a helper type used during parsing.
type rawEvent struct {
	typ   event.Type
	args  []uint64
	sargs []string

	// if typ == EvBatch, these fields describe the batch.
	batchPid    int32
	batchOffset int
}

type proc struct {
	pid int32
	// the remaining events in the current batch
	events []Event
	// buffer for reading batches into, aliased by proc.events
	buf []Event

	// there are no more batches left
	done bool
}

const eventsBucketSize = 524288 // 32 MiB of events

type Events struct {
	// Events is a slice of slices that grows one slice of size eventsBucketSize
	// at a time. This avoids the O(n) cost of slice growth in append, and
	// additionally allows consumers to drop references to parts of the data,
	// freeing memory piecewise.
	n       int
	buckets []*[eventsBucketSize]Event
	off     int
}

// grow grows the slice by one and returns a pointer to the new element, without
// overwriting it.
func (l *Events) grow() *Event {
	a, b := l.index(l.n)
	if a >= len(l.buckets) {
		l.buckets = append(l.buckets, new([eventsBucketSize]Event))
	}
	ptr := &l.buckets[a][b]
	l.n++
	return ptr
}

// append appends v to the slice and returns a pointer to the new element.
func (l *Events) append(v Event) *Event {
	ptr := l.grow()
	*ptr = v
	return ptr
}

func (l *Events) Ptr(i int) *Event {
	a, b := l.index(i + l.off)
	return &l.buckets[a][b]
}

func (l *Events) index(i int) (int, int) {
	// Doing the division on uint instead of int compiles this function to a
	// shift and an AND (for power of 2 bucket sizes), versus a whole bunch of
	// instructions for int.
	return int(uint(i) / eventsBucketSize), int(uint(i) % eventsBucketSize)
}

func (l *Events) Len() int {
	return l.n - l.off
}

func (l *Events) Less(i, j int) bool {
	return l.Ptr(i).Ts < l.Ptr(j).Ts
}

func (l *Events) Swap(i, j int) {
	*l.Ptr(i), *l.Ptr(j) = *l.Ptr(j), *l.Ptr(i)
}

func (l *Events) Pop() (*Event, bool) {
	if l.off == l.n {
		return nil, false
	}
	a, b := l.index(l.off)
	ptr := &l.buckets[a][b]
	l.off++
	if b == eventsBucketSize-1 || l.off == l.n {
		// We've consumed the last event from the bucket, so drop the bucket and
		// allow GC to collect it.
		l.buckets[a] = nil
	}
	return ptr, true
}

func (l *Events) All() func(yield func(ev *Event) bool) {
	return func(yield func(ev *Event) bool) {
		for i := 0; i < l.Len(); i++ {
			a, b := l.index(i + l.off)
			ptr := &l.buckets[a][b]
			if !yield(ptr) {
				return
			}
		}
	}
}

// parseEventBatches reads per-P event batches and merges them into a single, consistent
// stream. The high level idea is as follows. Events within an individual batch
// are in correct order, because they are emitted by a single P. So we need to
// produce a correct interleaving of the batches. To do this we take first
// unmerged event from each batch (frontier). Then choose subset that is "ready"
// to be merged, that is, events for which all dependencies are already merged.
// Then we choose event with the lowest timestamp from the subset, merge it and
// repeat. This approach ensures that we form a consistent stream even if
// timestamps are incorrect (condition observed on some machines).
func (p *parser) parseEventBatches() (Events, error) {
	// The ordering of CPU profile sample events in the data stream is based on
	// when each run of the signal handler was able to acquire the spinlock,
	// with original timestamps corresponding to when ReadTrace pulled the data
	// off of the profBuf queue. Re-sort them by the timestamp we captured
	// inside the signal handler.
	slices.SortFunc(p.cpuSamples, func(a, b Event) int {
		return cmp.Compare(a.Ts, b.Ts)
	})

	allProcs := make([]proc, 0, len(p.batchOffsets))
	for pid := range p.batchOffsets {
		allProcs = append(allProcs, proc{pid: pid})
	}
	allProcs = append(allProcs, proc{pid: ProfileP, events: p.cpuSamples})

	events := Events{}

	// Merge events as long as at least one P has more events
	gs := make(map[uint64]gState)
	// Note: technically we don't need a priority queue here. We're only ever
	// interested in the earliest elligible event, which means we just have to
	// track the smallest element. However, in practice, the priority queue
	// performs better, because for each event we only have to compute its state
	// transition once, not on each iteration. If it was elligible before, it'll
	// already be in the queue. Furthermore, on average, we only have one P to
	// look at in each iteration, because all other Ps are already in the queue.
	var frontier orderEventList

	availableProcs := make([]*proc, len(allProcs))
	for i := range allProcs {
		availableProcs[i] = &allProcs[i]
	}
	for {
	pidLoop:
		for i := 0; i < len(availableProcs); i++ {
			proc := availableProcs[i]

			for len(proc.events) == 0 {
				// Call loadBatch in a loop because sometimes batches are empty
				evs, err := p.loadBatch(proc.pid, proc.buf[:0])
				proc.buf = evs[:0]
				if err == io.EOF {
					// This P has no more events
					proc.done = true
					availableProcs[i], availableProcs[len(availableProcs)-1] = availableProcs[len(availableProcs)-1], availableProcs[i]
					availableProcs = availableProcs[:len(availableProcs)-1]
					// We swapped the element at i with another proc, so look at
					// the index again
					i--
					continue pidLoop
				} else if err != nil {
					return Events{}, err
				} else {
					proc.events = evs
				}
			}

			ev := &proc.events[0]
			g, init, _ := stateTransition(ev)

			// TODO(dh): This implementation matches the behavior of the
			// upstream 'go tool trace', and works in practice, but has run into
			// the following inconsistency during fuzzing: what happens if
			// multiple Ps have events for the same G? While building the
			// frontier we will check all of the events against the current
			// state of the G. However, when we process the frontier, the state
			// of the G changes, and a transition that was valid while building
			// the frontier may no longer be valid when processing the frontier.
			// Is this something that can happen for real, valid traces, or is
			// this only possible with corrupt data?
			if !transitionReady(g, gs[g], init) {
				continue
			}
			proc.events = proc.events[1:]
			availableProcs[i], availableProcs[len(availableProcs)-1] = availableProcs[len(availableProcs)-1], availableProcs[i]
			availableProcs = availableProcs[:len(availableProcs)-1]
			frontier.Push(orderEvent{*ev, proc})

			// We swapped the element at i with another proc, so look at the
			// index again
			i--
		}

		if len(frontier) == 0 {
			for i := range allProcs {
				if !allProcs[i].done {
					return Events{}, fmt.Errorf("no consistent ordering of events possible")
				}
			}
			break
		}
		f := frontier.Pop()

		// We're computing the state transition twice, once when computing the
		// frontier, and now to apply the transition. This is fine because
		// stateTransition is a pure function. Computing it again is cheaper
		// than storing large items in the frontier.
		g, init, next := stateTransition(&f.ev)

		// Get rid of "Local" events, they are intended merely for ordering.
		switch f.ev.Type {
		case EvGoStartLocal:
			f.ev.Type = EvGoStart
		case EvGoUnblockLocal:
			f.ev.Type = EvGoUnblock
		case EvGoSysExitLocal:
			f.ev.Type = EvGoSysExit
		}
		events.append(f.ev)

		if err := transition(gs, g, init, next); err != nil {
			return Events{}, err
		}
		availableProcs = append(availableProcs, f.proc)
	}

	// At this point we have a consistent stream of events. Make sure time
	// stamps respect the ordering. The tests will skip (not fail) the test case
	// if they see this error.
	if !sort.IsSorted(&events) {
		return Events{}, ErrTimeOrder
	}

	// The last part is giving correct timestamps to EvGoSysExit events. The
	// problem with EvGoSysExit is that actual syscall exit timestamp
	// (ev.Args[2]) is potentially acquired long before event emission. So far
	// we've used timestamp of event emission (ev.Ts). We could not set ev.Ts =
	// ev.Args[2] earlier, because it would produce seemingly broken timestamps
	// (misplaced event). We also can't simply update the timestamp and resort
	// events, because if timestamps are broken we will misplace the event and
	// later report logically broken trace (instead of reporting broken
	// timestamps).
	lastSysBlock := make(map[uint64]Timestamp)
	for i := 0; i < events.Len(); i++ {
		ev := events.Ptr(i)
		switch ev.Type {
		case EvGoSysBlock, EvGoInSyscall:
			lastSysBlock[ev.G] = ev.Ts
		case EvGoSysExit:
			ts := Timestamp(ev.Args[2])
			if ts == 0 {
				continue
			}
			block := lastSysBlock[ev.G]
			if block == 0 {
				return Events{}, fmt.Errorf("stray syscall exit")
			}
			if ts < block {
				return Events{}, ErrTimeOrder
			}
			ev.Ts = ts
		}
	}
	sort.Stable(&events)

	return events, nil
}

// collectBatchesAndCPUSamples records the offsets of batches and parses CPU samples.
func (p *parser) collectBatchesAndCPUSamples() error {
	// Read events.
	var raw rawEvent
	var curP int32
	for n := uint64(0); ; n++ {
		err := p.readRawEvent(skipArgs|skipStrings, &raw)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if raw.typ == EvNone {
			continue
		}

		if raw.typ == EvBatch {
			bo := batchOffset{offset: raw.batchOffset}
			p.batchOffsets[raw.batchPid] = append(p.batchOffsets[raw.batchPid], bo)
			curP = raw.batchPid
		}

		batches := p.batchOffsets[curP]
		if len(batches) == 0 {
			return fmt.Errorf("read event %d with current P of %d, but P has no batches yet",
				raw.typ, curP)
		}
		batches[len(batches)-1].numEvents++

		if raw.typ == EvCPUSample {
			e := Event{Type: raw.typ}

			argOffset := 1
			narg := raw.argNum()
			if len(raw.args) != narg {
				return fmt.Errorf("CPU sample has wrong number of arguments: want %d, got %d", narg, len(raw.args))
			}
			for i := argOffset; i < narg; i++ {
				if i == narg-1 {
					e.StkID = uint32(raw.args[i])
				} else {
					e.Args[i-argOffset] = raw.args[i]
				}
			}

			e.Ts = Timestamp(e.Args[0])
			e.P = int32(e.Args[1])
			e.G = e.Args[2]
			e.Args[0] = 0

			// Most events are written out by the active P at the exact moment
			// they describe. CPU profile samples are different because they're
			// written to the tracing log after some delay, by a separate worker
			// goroutine, into a separate buffer.
			//
			// We keep these in their own batch until all of the batches are
			// merged in timestamp order. We also (right before the merge)
			// re-sort these events by the timestamp captured in the profiling
			// signal handler.
			//
			// Note that we're not concerned about the memory usage of storing
			// all CPU samples during the indexing phase. There are orders of
			// magnitude fewer CPU samples than runtime events.
			p.cpuSamples = append(p.cpuSamples, e)
		}
	}

	return nil
}

const (
	skipArgs = 1 << iota
	skipStrings
)

func (p *parser) readByte() (byte, bool) {
	if p.off < len(p.data) && p.off >= 0 {
		b := p.data[p.off]
		p.off++
		return b, true
	} else {
		return 0, false
	}
}

func (p *parser) readFull(n int) ([]byte, error) {
	if p.off >= len(p.data) || p.off < 0 || p.off+n > len(p.data) {
		// p.off < 0 is impossible but makes BCE happy.
		//
		// We do fail outright if there's not enough data, we don't care about
		// partial results.
		return nil, io.ErrUnexpectedEOF
	}
	buf := p.data[p.off : p.off+n]
	p.off += n
	return buf, nil
}

// readRawEvent reads a raw event into ev. The slices in ev are only valid until
// the next call to readRawEvent, even when storing to a different location.
func (p *parser) readRawEvent(flags uint, ev *rawEvent) error {
	// The number of arguments is encoded using two bits and can thus only
	// represent the values 0–3. The value 3 (on the wire) indicates that
	// arguments are prefixed by their byte length, to encode >=3 arguments.
	const inlineArgs = 3

	// Read event type and number of arguments (1 byte).
	b, ok := p.readByte()
	if !ok {
		return io.EOF
	}
	typ := event.Type(b << 2 >> 2)
	// Most events have a timestamp before the actual arguments, so we add 1 and
	// parse it like it's the first argument. EvString has a special format and
	// the number of arguments doesn't matter. EvBatch writes '1' as the number
	// of arguments, but actually has two: a pid and a timestamp, but here the
	// timestamp is the second argument, not the first; adding 1 happens to come
	// up with the correct number, but it doesn't matter, because EvBatch has
	// custom logic for parsing.
	//
	// Note that because we're adding 1, inlineArgs == 3 describes the largest
	// number of logical arguments that isn't length-prefixed, even though the
	// value 3 on the wire indicates length-prefixing. For us, that becomes narg
	// == 4.
	narg := b>>6 + 1
	if typ == EvNone || typ >= EvCount || EventDescriptions[typ].minVersion > p.ver {
		return fmt.Errorf("unknown event type %d", typ)
	}

	switch typ {
	case EvString:
		if flags&skipStrings != 0 {
			// String dictionary entry [ID, length, string].
			if _, err := p.readVal(); err != nil {
				return errMalformedVarint
			}
			ln, err := p.readVal()
			if err != nil {
				return err
			}
			if !p.discard(ln) {
				return fmt.Errorf("failed to read trace: %w", io.EOF)
			}
		} else {
			// String dictionary entry [ID, length, string].
			id, err := p.readVal()
			if err != nil {
				return err
			}
			if id == 0 {
				return errors.New("string has invalid id 0")
			}
			if p.strings[id] != "" {
				return fmt.Errorf("string has duplicate id %d", id)
			}
			var ln uint64
			ln, err = p.readVal()
			if err != nil {
				return err
			}
			if ln == 0 {
				return errors.New("string has invalid length 0")
			}
			if ln > 1e6 {
				return fmt.Errorf("string has too large length %d", ln)
			}
			buf, err := p.readFull(int(ln))
			if err != nil {
				return fmt.Errorf("failed to read trace: %w", err)
			}
			p.strings[id] = string(buf)
		}

		ev.typ = EvNone
		return nil
	case EvBatch:
		if want := byte(2); narg != want {
			return fmt.Errorf("EvBatch has wrong number of arguments: got %d, want %d", narg, want)
		}

		// -1 because we've already read the first byte of the batch
		off := p.off - 1

		pid, err := p.readVal()
		if err != nil {
			return err
		}
		if pid != math.MaxUint64 && pid > math.MaxInt32 {
			return fmt.Errorf("processor ID %d is larger than maximum of %d", pid, uint64(math.MaxUint))
		}

		var pid32 int32
		if pid == math.MaxUint64 {
			pid32 = -1
		} else {
			pid32 = int32(pid)
		}

		v, err := p.readVal()
		if err != nil {
			return err
		}

		*ev = rawEvent{
			typ:         EvBatch,
			args:        p.args[:0],
			batchPid:    pid32,
			batchOffset: off,
		}
		ev.args = append(ev.args, pid, v)
		return nil
	default:
		*ev = rawEvent{typ: typ, args: p.args[:0]}
		if narg <= inlineArgs {
			if flags&skipArgs == 0 {
				for i := 0; i < int(narg); i++ {
					v, err := p.readVal()
					if err != nil {
						return fmt.Errorf("failed to read event %d argument: %w", typ, err)
					}
					ev.args = append(ev.args, v)
				}
			} else {
				for i := 0; i < int(narg); i++ {
					if _, err := p.readVal(); err != nil {
						return fmt.Errorf("failed to read event %d argument: %w", typ, errMalformedVarint)
					}
				}
			}
		} else {
			// More than inlineArgs args, the first value is length of the event
			// in bytes.
			v, err := p.readVal()
			if err != nil {
				return fmt.Errorf("failed to read event %d argument: %w", typ, err)
			}

			if limit := uint64(2048); v > limit {
				// At the time of Go 1.19, v seems to be at most 128. Set 2048
				// as a generous upper limit and guard against malformed traces.
				return fmt.Errorf("failed to read event %d argument: length-prefixed argument too big: %d bytes, limit is %d", typ, v, limit)
			}

			if flags&skipArgs == 0 || typ == EvCPUSample {
				buf, err := p.readFull(int(v))
				if err != nil {
					return fmt.Errorf("failed to read trace: %w", err)
				}
				for len(buf) > 0 {
					var v uint64
					v, buf, err = readValFrom(buf)
					if err != nil {
						return err
					}
					ev.args = append(ev.args, v)
				}
			} else {
				// Skip over arguments
				if !p.discard(v) {
					return fmt.Errorf("failed to read trace: %w", io.EOF)
				}
			}
			if typ == EvUserLog {
				// EvUserLog records are followed by a value string
				if flags&skipArgs == 0 {
					// Read string
					s, err := p.readStr()
					if err != nil {
						return err
					}
					ev.sargs = append(ev.sargs, s)
				} else {
					// Skip string
					v, err := p.readVal()
					if err != nil {
						return err
					}
					if !p.discard(v) {
						return io.EOF
					}
				}
			}
		}

		p.args = ev.args[:0]
		return nil
	}
}

// loadBatch loads the next batch for pid and appends its contents to to events.
func (p *parser) loadBatch(pid int32, events []Event) ([]Event, error) {
	offsets := p.batchOffsets[pid]
	if len(offsets) == 0 {
		return nil, io.EOF
	}
	n := offsets[0].numEvents
	offset := offsets[0].offset
	offsets = offsets[1:]
	p.batchOffsets[pid] = offsets

	p.off = offset

	if cap(events) < n {
		events = make([]Event, 0, n)
	}

	gotHeader := false
	var raw rawEvent
	var ev Event
	for {
		err := p.readRawEvent(0, &raw)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if raw.typ == EvNone || raw.typ == EvCPUSample {
			continue
		}
		if raw.typ == EvBatch {
			if gotHeader {
				break
			} else {
				gotHeader = true
			}
		}

		err = p.parseEvent(&raw, &ev)
		if err != nil {
			return nil, err
		}
		if ev.Type != EvNone {
			events = append(events, ev)
		}
	}

	return events, nil
}

func (p *parser) readStr() (s string, err error) {
	sz, err := p.readVal()
	if err != nil {
		return "", err
	}
	if sz == 0 {
		return "", nil
	}
	if sz > 1e6 {
		return "", fmt.Errorf("string is too large (len=%d)", sz)
	}
	buf, err := p.readFull(int(sz))
	if err != nil {
		return "", fmt.Errorf("failed to read trace: %w", err)
	}
	return string(buf), nil
}

// parseEvent transforms raw events into events.
// It does analyze and verify per-event-type arguments.
func (p *parser) parseEvent(raw *rawEvent, ev *Event) error {
	desc := &EventDescriptions[raw.typ]
	if desc.Name == "" {
		return fmt.Errorf("missing description for event type %d", raw.typ)
	}
	narg := raw.argNum()
	if len(raw.args) != narg {
		return fmt.Errorf("%s has wrong number of arguments: want %d, got %d", desc.Name, narg, len(raw.args))
	}
	switch raw.typ {
	case EvBatch:
		p.lastGs[p.lastP] = p.lastG
		if raw.args[0] != math.MaxUint64 && raw.args[0] > math.MaxInt32 {
			return fmt.Errorf("processor ID %d is larger than maximum of %d", raw.args[0], uint64(math.MaxInt32))
		}
		if raw.args[0] == math.MaxUint64 {
			p.lastP = -1
		} else {
			p.lastP = int32(raw.args[0])
		}
		p.lastG = p.lastGs[p.lastP]
		p.lastTs = Timestamp(raw.args[1])
	case EvFrequency:
		p.ticksPerSec = int64(raw.args[0])
		if p.ticksPerSec <= 0 {
			// The most likely cause for this is tick skew on different CPUs.
			// For example, solaris/amd64 seems to have wildly different
			// ticks on different CPUs.
			return ErrTimeOrder
		}
	case EvTimerGoroutine:
		p.timerGoids[raw.args[0]] = true
	case EvStack:
		if len(raw.args) < 2 {
			return fmt.Errorf("EvStack has wrong number of arguments: want at least 2, got %d", len(raw.args))
		}
		size := raw.args[1]
		if size > 1000 {
			return fmt.Errorf("EvStack has bad number of frames: %d", size)
		}
		want := 2 + 4*size
		if uint64(len(raw.args)) != want {
			return fmt.Errorf("EvStack has wrong number of arguments: want %d, got %d", want, len(raw.args))
		}
		id := uint32(raw.args[0])
		if id != 0 && size > 0 {
			stk := p.allocateStack(size)
			for i := 0; i < int(size); i++ {
				pc := raw.args[2+i*4+0]
				fn := raw.args[2+i*4+1]
				file := raw.args[2+i*4+2]
				line := raw.args[2+i*4+3]
				stk[i] = pc

				if _, ok := p.pcs[pc]; !ok {
					p.pcs[pc] = Frame{PC: pc, Fn: fn, File: file, Line: int(line)}
				}
			}
			p.stacks[id] = stk
		}
	case EvCPUSample:
		// These events get parsed during the indexing step and don't strictly
		// belong to the batch.
	default:
		*ev = Event{Type: raw.typ, P: p.lastP, G: p.lastG}
		var argOffset int
		ev.Ts = p.lastTs + Timestamp(raw.args[0])
		argOffset = 1
		p.lastTs = ev.Ts
		for i := argOffset; i < narg; i++ {
			if i == narg-1 && desc.Stack {
				ev.StkID = uint32(raw.args[i])
			} else {
				ev.Args[i-argOffset] = raw.args[i]
			}
		}
		switch raw.typ {
		case EvGoStart, EvGoStartLocal, EvGoStartLabel:
			p.lastG = ev.Args[0]
			ev.G = p.lastG
		case EvGoEnd, EvGoStop, EvGoSched, EvGoPreempt,
			EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
			EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet,
			EvGoSysBlock, EvGoBlockGC:
			p.lastG = 0
		case EvGoSysExit, EvGoWaiting, EvGoInSyscall:
			ev.G = ev.Args[0]
		case EvUserTaskCreate:
			// e.Args 0: taskID, 1:parentID, 2:nameID
		case EvUserRegion:
			// e.Args 0: taskID, 1: mode, 2:nameID
		case EvUserLog:
			// e.Args 0: taskID, 1:keyID, 2: stackID, 3: messageID
			// raw.sargs 0: message

			if id, ok := p.inlineStringsMapping[raw.sargs[0]]; ok {
				ev.Args[3] = uint64(id)
			} else {
				id := len(p.inlineStrings)
				p.inlineStringsMapping[raw.sargs[0]] = id
				p.inlineStrings = append(p.inlineStrings, raw.sargs[0])
				ev.Args[3] = uint64(id)
			}
		}

		return nil
	}

	ev.Type = EvNone
	return nil
}

// ErrTimeOrder is returned by Parse when the trace contains
// time stamps that do not respect actual event ordering.
var ErrTimeOrder = errors.New("time stamps out of order")

// postProcessTrace does inter-event verification and information restoration.
// The resulting trace is guaranteed to be consistent
// (for example, a P does not run two Gs at the same time, or a G is indeed
// blocked before an unblock event).
func (p *parser) postProcessTrace(events Events) error {
	const (
		gDead = iota
		gRunnable
		gRunning
		gWaiting
	)
	type gdesc struct {
		state        int
		ev           *Event
		evStart      *Event
		evCreate     *Event
		evMarkAssist *Event
	}
	type pdesc struct {
		running bool
		g       uint64
		evSweep *Event
	}

	gs := make(map[uint64]gdesc)
	ps := make(map[int32]pdesc)
	tasks := make(map[uint64]*Event)           // task id to task creation events
	activeRegions := make(map[uint64][]*Event) // goroutine id to stack of regions
	gs[0] = gdesc{state: gRunning}
	var evGC, evSTW *Event

	checkRunning := func(p pdesc, g gdesc, ev *Event, allowG0 bool) error {
		name := EventDescriptions[ev.Type].Name
		if g.state != gRunning {
			return fmt.Errorf("g %d is not running while %s (time %d)", ev.G, name, ev.Ts)
		}
		if p.g != ev.G {
			return fmt.Errorf("p %d is not running g %d while %s (time %d)", ev.P, ev.G, name, ev.Ts)
		}
		if !allowG0 && ev.G == 0 {
			return fmt.Errorf("g 0 did %s (time %d)", name, ev.Ts)
		}
		return nil
	}

	for evIdx := 0; evIdx < events.Len(); evIdx++ {
		ev := events.Ptr(evIdx)

		switch ev.Type {
		case EvProcStart:
			p := ps[ev.P]
			if p.running {
				return fmt.Errorf("p %d is running before start (time %d)", ev.P, ev.Ts)
			}
			p.running = true

			ps[ev.P] = p
		case EvProcStop:
			p := ps[ev.P]
			if !p.running {
				return fmt.Errorf("p %d is not running before stop (time %d)", ev.P, ev.Ts)
			}
			if p.g != 0 {
				return fmt.Errorf("p %d is running a goroutine %d during stop (time %d)", ev.P, p.g, ev.Ts)
			}
			p.running = false

			ps[ev.P] = p
		case EvGCStart:
			if evGC != nil {
				return fmt.Errorf("previous GC is not ended before a new one (time %d)", ev.Ts)
			}
			evGC = ev
			// Attribute this to the global GC state.
			ev.P = GCP
		case EvGCDone:
			if evGC == nil {
				return fmt.Errorf("bogus GC end (time %d)", ev.Ts)
			}
			evGC = nil
		case EvSTWStart:
			evp := &evSTW
			if *evp != nil {
				return fmt.Errorf("previous STW is not ended before a new one (time %d)", ev.Ts)
			}
			*evp = ev
		case EvSTWDone:
			evp := &evSTW
			if *evp == nil {
				return fmt.Errorf("bogus STW end (time %d)", ev.Ts)
			}
			*evp = nil
		case EvGCSweepStart:
			p := ps[ev.P]
			if p.evSweep != nil {
				return fmt.Errorf("previous sweeping is not ended before a new one (time %d)", ev.Ts)
			}
			p.evSweep = ev

			ps[ev.P] = p
		case EvGCMarkAssistStart:
			g := gs[ev.G]
			if g.evMarkAssist != nil {
				return fmt.Errorf("previous mark assist is not ended before a new one (time %d)", ev.Ts)
			}
			g.evMarkAssist = ev

			gs[ev.G] = g
		case EvGCMarkAssistDone:
			// Unlike most events, mark assists can be in progress when a
			// goroutine starts tracing, so we can't report an error here.
			g := gs[ev.G]
			if g.evMarkAssist != nil {
				g.evMarkAssist = nil
			}

			gs[ev.G] = g
		case EvGCSweepDone:
			p := ps[ev.P]
			if p.evSweep == nil {
				return fmt.Errorf("bogus sweeping end (time %d)", ev.Ts)
			}
			p.evSweep = nil

			ps[ev.P] = p
		case EvGoWaiting:
			g := gs[ev.G]
			if g.state != gRunnable {
				return fmt.Errorf("g %d is not runnable before EvGoWaiting (time %d)", ev.G, ev.Ts)
			}
			g.state = gWaiting
			g.ev = ev

			gs[ev.G] = g
		case EvGoInSyscall:
			g := gs[ev.G]
			if g.state != gRunnable {
				return fmt.Errorf("g %d is not runnable before EvGoInSyscall (time %d)", ev.G, ev.Ts)
			}
			g.state = gWaiting
			g.ev = ev

			gs[ev.G] = g
		case EvGoCreate:
			g := gs[ev.G]
			p := ps[ev.P]
			if err := checkRunning(p, g, ev, true); err != nil {
				return err
			}
			if _, ok := gs[ev.Args[0]]; ok {
				return fmt.Errorf("g %d already exists (time %d)", ev.Args[0], ev.Ts)
			}
			gs[ev.Args[0]] = gdesc{state: gRunnable, ev: ev, evCreate: ev}

		case EvGoStart, EvGoStartLabel:
			g := gs[ev.G]
			p := ps[ev.P]
			if g.state != gRunnable {
				return fmt.Errorf("g %d is not runnable before start (time %d)", ev.G, ev.Ts)
			}
			if p.g != 0 {
				return fmt.Errorf("p %d is already running g %d while start g %d (time %d)", ev.P, p.g, ev.G, ev.Ts)
			}
			g.state = gRunning
			g.evStart = ev
			p.g = ev.G
			if g.evCreate != nil {
				ev.StkID = uint32(g.evCreate.Args[1])
				g.evCreate = nil
			}

			if g.ev != nil {
				g.ev = nil
			}

			gs[ev.G] = g
			ps[ev.P] = p
		case EvGoEnd, EvGoStop:
			g := gs[ev.G]
			p := ps[ev.P]
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.evStart = nil
			g.state = gDead
			p.g = 0

			if ev.Type == EvGoEnd { // flush all active regions
				delete(activeRegions, ev.G)
			}

			gs[ev.G] = g
			ps[ev.P] = p
		case EvGoSched, EvGoPreempt:
			g := gs[ev.G]
			p := ps[ev.P]
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.state = gRunnable
			g.evStart = nil
			p.g = 0
			g.ev = ev

			gs[ev.G] = g
			ps[ev.P] = p
		case EvGoUnblock:
			g := gs[ev.G]
			p := ps[ev.P]
			if g.state != gRunning {
				return fmt.Errorf("g %d is not running while unpark (time %d)", ev.G, ev.Ts)
			}
			if ev.P != TimerP && p.g != ev.G {
				return fmt.Errorf("p %d is not running g %d while unpark (time %d)", ev.P, ev.G, ev.Ts)
			}
			g1 := gs[ev.Args[0]]
			if g1.state != gWaiting {
				return fmt.Errorf("g %d is not waiting before unpark (time %d)", ev.Args[0], ev.Ts)
			}
			if g1.ev != nil && g1.ev.Type == EvGoBlockNet {
				ev.P = NetpollP
			}
			g1.state = gRunnable
			g1.ev = ev
			gs[ev.Args[0]] = g1

		case EvGoSysCall:
			g := gs[ev.G]
			p := ps[ev.P]
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.ev = ev

			gs[ev.G] = g
		case EvGoSysBlock:
			g := gs[ev.G]
			p := ps[ev.P]
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.state = gWaiting
			g.evStart = nil
			p.g = 0

			gs[ev.G] = g
			ps[ev.P] = p
		case EvGoSysExit:
			g := gs[ev.G]
			if g.state != gWaiting {
				return fmt.Errorf("g %d is not waiting during syscall exit (time %d)", ev.G, ev.Ts)
			}
			g.state = gRunnable
			g.ev = ev

			gs[ev.G] = g
		case EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
			EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet, EvGoBlockGC:
			g := gs[ev.G]
			p := ps[ev.P]
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.state = gWaiting
			g.ev = ev
			g.evStart = nil
			p.g = 0

			gs[ev.G] = g
			ps[ev.P] = p
		case EvUserTaskCreate:
			taskid := ev.Args[0]
			if prevEv, ok := tasks[taskid]; ok {
				return fmt.Errorf("task id conflicts (id:%d), %q vs %q", taskid, ev, prevEv)
			}
			tasks[ev.Args[0]] = ev

		case EvUserTaskEnd:
			taskid := ev.Args[0]
			delete(tasks, taskid)

		case EvUserRegion:
			mode := ev.Args[1]
			regions := activeRegions[ev.G]
			if mode == 0 { // region start
				activeRegions[ev.G] = append(regions, ev) // push
			} else if mode == 1 { // region end
				n := len(regions)
				if n > 0 { // matching region start event is in the trace.
					s := regions[n-1]
					if s.Args[0] != ev.Args[0] || s.Args[2] != ev.Args[2] { // task id, region name mismatch
						return fmt.Errorf("misuse of region in goroutine %d: span end %q when the inner-most active span start event is %q", ev.G, ev, s)
					}

					if n > 1 {
						activeRegions[ev.G] = regions[:n-1]
					} else {
						delete(activeRegions, ev.G)
					}
				}
			} else {
				return fmt.Errorf("invalid user region mode: %q", ev)
			}
		}

		if ev.StkID != 0 && len(p.stacks[ev.StkID]) == 0 {
			// Make sure events don't refer to stacks that don't exist or to
			// stacks with zero frames. Neither of these should be possible, but
			// better be safe than sorry.

			ev.StkID = 0
		}

	}

	// TODO(mknyszek): restore stacks for EvGoStart events.
	return nil
}

var errMalformedVarint = errors.New("malformatted base-128 varint")

// readVal reads unsigned base-128 value from r.
func (p *parser) readVal() (uint64, error) {
	v, n := binary.Uvarint(p.data[p.off:])
	if n <= 0 {
		return 0, errMalformedVarint
	}
	p.off += n
	return v, nil
}

func readValFrom(buf []byte) (v uint64, rem []byte, err error) {
	v, n := binary.Uvarint(buf)
	if n <= 0 {
		return 0, nil, errMalformedVarint
	}
	return v, buf[n:], nil
}

func (ev *Event) String() string {
	desc := &EventDescriptions[ev.Type]
	w := new(bytes.Buffer)
	fmt.Fprintf(w, "%d %s p=%d g=%d stk=%d", ev.Ts, desc.Name, ev.P, ev.G, ev.StkID)
	for i, a := range desc.Args {
		fmt.Fprintf(w, " %s=%d", a, ev.Args[i])
	}
	return w.String()
}

// argNum returns total number of args for the event accounting for timestamps,
// sequence numbers and differences between trace format versions.
func (raw *rawEvent) argNum() int {
	desc := &EventDescriptions[raw.typ]
	if raw.typ == EvStack {
		return len(raw.args)
	}
	narg := len(desc.Args)
	if desc.Stack {
		narg++
	}
	switch raw.typ {
	case EvBatch, EvFrequency, EvTimerGoroutine:
		return narg
	}
	narg++ // timestamp
	return narg
}

// Event types in the trace.
// Verbatim copy from src/runtime/trace.go with the "trace" prefix removed.
const (
	EvNone              event.Type = 0  // unused
	EvBatch             event.Type = 1  // start of per-P batch of events [pid, timestamp]
	EvFrequency         event.Type = 2  // contains tracer timer frequency [frequency (ticks per second)]
	EvStack             event.Type = 3  // stack [stack id, number of PCs, array of {PC, func string ID, file string ID, line}]
	EvGomaxprocs        event.Type = 4  // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack id]
	EvProcStart         event.Type = 5  // start of P [timestamp, thread id]
	EvProcStop          event.Type = 6  // stop of P [timestamp]
	EvGCStart           event.Type = 7  // GC start [timestamp, seq, stack id]
	EvGCDone            event.Type = 8  // GC done [timestamp]
	EvSTWStart          event.Type = 9  // GC mark termination start [timestamp, kind]
	EvSTWDone           event.Type = 10 // GC mark termination done [timestamp]
	EvGCSweepStart      event.Type = 11 // GC sweep start [timestamp, stack id]
	EvGCSweepDone       event.Type = 12 // GC sweep done [timestamp, swept, reclaimed]
	EvGoCreate          event.Type = 13 // goroutine creation [timestamp, new goroutine id, new stack id, stack id]
	EvGoStart           event.Type = 14 // goroutine starts running [timestamp, goroutine id, seq]
	EvGoEnd             event.Type = 15 // goroutine ends [timestamp]
	EvGoStop            event.Type = 16 // goroutine stops (like in select{}) [timestamp, stack]
	EvGoSched           event.Type = 17 // goroutine calls Gosched [timestamp, stack]
	EvGoPreempt         event.Type = 18 // goroutine is preempted [timestamp, stack]
	EvGoSleep           event.Type = 19 // goroutine calls Sleep [timestamp, stack]
	EvGoBlock           event.Type = 20 // goroutine blocks [timestamp, stack]
	EvGoUnblock         event.Type = 21 // goroutine is unblocked [timestamp, goroutine id, seq, stack]
	EvGoBlockSend       event.Type = 22 // goroutine blocks on chan send [timestamp, stack]
	EvGoBlockRecv       event.Type = 23 // goroutine blocks on chan recv [timestamp, stack]
	EvGoBlockSelect     event.Type = 24 // goroutine blocks on select [timestamp, stack]
	EvGoBlockSync       event.Type = 25 // goroutine blocks on Mutex/RWMutex [timestamp, stack]
	EvGoBlockCond       event.Type = 26 // goroutine blocks on Cond [timestamp, stack]
	EvGoBlockNet        event.Type = 27 // goroutine blocks on network [timestamp, stack]
	EvGoSysCall         event.Type = 28 // syscall enter [timestamp, stack]
	EvGoSysExit         event.Type = 29 // syscall exit [timestamp, goroutine id, seq, real timestamp]
	EvGoSysBlock        event.Type = 30 // syscall blocks [timestamp]
	EvGoWaiting         event.Type = 31 // denotes that goroutine is blocked when tracing starts [timestamp, goroutine id]
	EvGoInSyscall       event.Type = 32 // denotes that goroutine is in syscall when tracing starts [timestamp, goroutine id]
	EvHeapAlloc         event.Type = 33 // gcController.heapLive change [timestamp, heap live bytes]
	EvHeapGoal          event.Type = 34 // gcController.heapGoal change [timestamp, heap goal bytes]
	EvTimerGoroutine    event.Type = 35 // denotes timer goroutine [timer goroutine id]
	EvFutileWakeup      event.Type = 36 // denotes that the previous wakeup of this goroutine was futile [timestamp]
	EvString            event.Type = 37 // string dictionary entry [ID, length, string]
	EvGoStartLocal      event.Type = 38 // goroutine starts running on the same P as the last event [timestamp, goroutine id]
	EvGoUnblockLocal    event.Type = 39 // goroutine is unblocked on the same P as the last event [timestamp, goroutine id, stack]
	EvGoSysExitLocal    event.Type = 40 // syscall exit on the same P as the last event [timestamp, goroutine id, real timestamp]
	EvGoStartLabel      event.Type = 41 // goroutine starts running with label [timestamp, goroutine id, seq, label string id]
	EvGoBlockGC         event.Type = 42 // goroutine blocks on GC assist [timestamp, stack]
	EvGCMarkAssistStart event.Type = 43 // GC mark assist start [timestamp, stack]
	EvGCMarkAssistDone  event.Type = 44 // GC mark assist done [timestamp]
	EvUserTaskCreate    event.Type = 45 // trace.NewTask [timestamp, internal task id, internal parent id, stack, name string]
	EvUserTaskEnd       event.Type = 46 // end of task [timestamp, internal task id, stack]
	EvUserRegion        event.Type = 47 // trace.WithRegion [timestamp, internal task id, mode(0:start, 1:end), name string]
	EvUserLog           event.Type = 48 // trace.Log [timestamp, internal id, key string id, stack, value string]
	EvCPUSample         event.Type = 49 // CPU profiling sample [timestamp, stack, real timestamp, real P id (-1 when absent), goroutine id]
	EvCount             event.Type = 50
)

var EventDescriptions = [256]struct {
	Name       string
	minVersion version.Version
	Stack      bool
	Args       []string
	SArgs      []string // string arguments
}{
	EvNone:              {"None", 5, false, []string{}, nil},
	EvBatch:             {"Batch", 5, false, []string{"p", "ticks"}, nil}, // in 1.5 format it was {"p", "seq", "ticks"}
	EvFrequency:         {"Frequency", 5, false, []string{"freq"}, nil},   // in 1.5 format it was {"freq", "unused"}
	EvStack:             {"Stack", 5, false, []string{"id", "siz"}, nil},
	EvGomaxprocs:        {"Gomaxprocs", 5, true, []string{"procs"}, nil},
	EvProcStart:         {"ProcStart", 5, false, []string{"thread"}, nil},
	EvProcStop:          {"ProcStop", 5, false, []string{}, nil},
	EvGCStart:           {"GCStart", 5, true, []string{"seq"}, nil}, // in 1.5 format it was {}
	EvGCDone:            {"GCDone", 5, false, []string{}, nil},
	EvSTWStart:          {"GCSTWStart", 5, false, []string{"kindid"}, []string{"kind"}}, // <= 1.9, args was {} (implicitly {0})
	EvSTWDone:           {"GCSTWDone", 5, false, []string{}, nil},
	EvGCSweepStart:      {"GCSweepStart", 5, true, []string{}, nil},
	EvGCSweepDone:       {"GCSweepDone", 5, false, []string{"swept", "reclaimed"}, nil}, // before 1.9, format was {}
	EvGoCreate:          {"GoCreate", 5, true, []string{"g", "stack"}, nil},
	EvGoStart:           {"GoStart", 5, false, []string{"g", "seq"}, nil}, // in 1.5 format it was {"g"}
	EvGoEnd:             {"GoEnd", 5, false, []string{}, nil},
	EvGoStop:            {"GoStop", 5, true, []string{}, nil},
	EvGoSched:           {"GoSched", 5, true, []string{}, nil},
	EvGoPreempt:         {"GoPreempt", 5, true, []string{}, nil},
	EvGoSleep:           {"GoSleep", 5, true, []string{}, nil},
	EvGoBlock:           {"GoBlock", 5, true, []string{}, nil},
	EvGoUnblock:         {"GoUnblock", 5, true, []string{"g", "seq"}, nil}, // in 1.5 format it was {"g"}
	EvGoBlockSend:       {"GoBlockSend", 5, true, []string{}, nil},
	EvGoBlockRecv:       {"GoBlockRecv", 5, true, []string{}, nil},
	EvGoBlockSelect:     {"GoBlockSelect", 5, true, []string{}, nil},
	EvGoBlockSync:       {"GoBlockSync", 5, true, []string{}, nil},
	EvGoBlockCond:       {"GoBlockCond", 5, true, []string{}, nil},
	EvGoBlockNet:        {"GoBlockNet", 5, true, []string{}, nil},
	EvGoSysCall:         {"GoSysCall", 5, true, []string{}, nil},
	EvGoSysExit:         {"GoSysExit", 5, false, []string{"g", "seq", "ts"}, nil},
	EvGoSysBlock:        {"GoSysBlock", 5, false, []string{}, nil},
	EvGoWaiting:         {"GoWaiting", 5, false, []string{"g"}, nil},
	EvGoInSyscall:       {"GoInSyscall", 5, false, []string{"g"}, nil},
	EvHeapAlloc:         {"HeapAlloc", 5, false, []string{"mem"}, nil},
	EvHeapGoal:          {"HeapGoal", 5, false, []string{"mem"}, nil},
	EvTimerGoroutine:    {"TimerGoroutine", 5, false, []string{"g"}, nil}, // in 1.5 format it was {"g", "unused"}
	EvFutileWakeup:      {"FutileWakeup", 5, false, []string{}, nil},
	EvString:            {"String", 7, false, []string{}, nil},
	EvGoStartLocal:      {"GoStartLocal", 7, false, []string{"g"}, nil},
	EvGoUnblockLocal:    {"GoUnblockLocal", 7, true, []string{"g"}, nil},
	EvGoSysExitLocal:    {"GoSysExitLocal", 7, false, []string{"g", "ts"}, nil},
	EvGoStartLabel:      {"GoStartLabel", 8, false, []string{"g", "seq", "labelid"}, []string{"label"}},
	EvGoBlockGC:         {"GoBlockGC", 8, true, []string{}, nil},
	EvGCMarkAssistStart: {"GCMarkAssistStart", 9, true, []string{}, nil},
	EvGCMarkAssistDone:  {"GCMarkAssistDone", 9, false, []string{}, nil},
	EvUserTaskCreate:    {"UserTaskCreate", 11, true, []string{"taskid", "pid", "typeid"}, []string{"name"}},
	EvUserTaskEnd:       {"UserTaskEnd", 11, true, []string{"taskid"}, nil},
	EvUserRegion:        {"UserRegion", 11, true, []string{"taskid", "mode", "typeid"}, []string{"name"}},
	EvUserLog:           {"UserLog", 11, true, []string{"id", "keyid"}, []string{"category", "message"}},
	EvCPUSample:         {"CPUSample", 19, true, []string{"ts", "p", "g"}, nil},
}

//gcassert:inline
func (p *parser) allocateStack(size uint64) []uint64 {
	if size == 0 {
		return nil
	}

	// Stacks are plentiful but small. For our "Staticcheck on std" trace with
	// 11e6 events, we have roughly 500,000 stacks, using 200 MiB of memory. To
	// avoid making 500,000 small allocations we allocate backing arrays 1 MiB
	// at a time.
	out := p.stacksData
	if uint64(len(out)) < size {
		out = make([]uint64, 1024*128)
	}
	p.stacksData = out[size:]
	return out[:size:size]
}

func (tr *Trace) STWReason(kindID uint64) STWReason {
	if tr.Version < 21 {
		if kindID == 0 || kindID == 1 {
			return STWReason(kindID + 1)
		} else {
			return STWUnknown
		}
	} else if tr.Version == 21 {
		if kindID < NumSTWReasons {
			return STWReason(kindID)
		} else {
			return STWUnknown
		}
	} else {
		return STWUnknown
	}
}

type STWReason int

const (
	STWUnknown                 STWReason = 0
	STWGCMarkTermination       STWReason = 1
	STWGCSweepTermination      STWReason = 2
	STWWriteHeapDump           STWReason = 3
	STWGoroutineProfile        STWReason = 4
	STWGoroutineProfileCleanup STWReason = 5
	STWAllGoroutinesStackTrace STWReason = 6
	STWReadMemStats            STWReason = 7
	STWAllThreadsSyscall       STWReason = 8
	STWGOMAXPROCS              STWReason = 9
	STWStartTrace              STWReason = 10
	STWStopTrace               STWReason = 11
	STWCountPagesInUse         STWReason = 12
	STWReadMetricsSlow         STWReason = 13
	STWReadMemStatsSlow        STWReason = 14
	STWPageCachePagesLeaked    STWReason = 15
	STWResetDebugLog           STWReason = 16

	NumSTWReasons = 17
)
