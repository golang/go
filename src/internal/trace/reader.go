// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"internal/trace/internal/tracev1"
	"internal/trace/tracev2"
	"internal/trace/version"
)

// Reader reads a byte stream, validates it, and produces trace events.
//
// Provided the trace is non-empty the Reader always produces a Sync
// event as the first event, and a Sync event as the last event.
// (There may also be any number of Sync events in the middle, too.)
type Reader struct {
	version    version.Version
	r          *bufio.Reader
	lastTs     Time
	gen        *generation
	frontier   []*batchCursor
	cpuSamples []cpuSample
	order      ordering
	syncs      int
	readGenErr error
	done       bool

	// Spill state.
	//
	// Traces before Go 1.26 had no explicit end-of-generation signal, and
	// so the first batch of the next generation needed to be parsed to identify
	// a new generation. This batch is the "spilled" so we don't lose track
	// of it when parsing the next generation.
	//
	// This is unnecessary after Go 1.26 because of an explicit end-of-generation
	// signal.
	spill        *spilledBatch
	spillErr     error // error from reading spill
	spillErrSync bool  // whether we emitted a Sync before reporting spillErr

	v1Events *traceV1Converter
}

// NewReader creates a new trace reader.
func NewReader(r io.Reader) (*Reader, error) {
	br := bufio.NewReader(r)
	v, err := version.ReadHeader(br)
	if err != nil {
		return nil, err
	}
	switch v {
	case version.Go111, version.Go119, version.Go121:
		tr, err := tracev1.Parse(br, v)
		if err != nil {
			return nil, err
		}
		return &Reader{
			v1Events: convertV1Trace(tr),
		}, nil
	case version.Go122, version.Go123, version.Go125, version.Go126:
		return &Reader{
			version: v,
			r:       br,
			order: ordering{
				traceVer:    v,
				mStates:     make(map[ThreadID]*mState),
				pStates:     make(map[ProcID]*pState),
				gStates:     make(map[GoID]*gState),
				activeTasks: make(map[TaskID]taskState),
			},
		}, nil
	default:
		return nil, fmt.Errorf("unknown or unsupported version go 1.%d", v)
	}
}

// ReadEvent reads a single event from the stream.
//
// If the stream has been exhausted, it returns an invalid event and io.EOF.
func (r *Reader) ReadEvent() (e Event, err error) {
	// Return only io.EOF if we're done.
	if r.done {
		return Event{}, io.EOF
	}

	// Handle v1 execution traces.
	if r.v1Events != nil {
		if r.syncs == 0 {
			// Always emit a sync event first, if we have any events at all.
			ev, ok := r.v1Events.events.Peek()
			if ok {
				r.syncs++
				return syncEvent(r.v1Events.evt, Time(ev.Ts-1), r.syncs), nil
			}
		}
		ev, err := r.v1Events.next()
		if err == io.EOF {
			// Always emit a sync event at the end.
			r.done = true
			r.syncs++
			return syncEvent(nil, r.v1Events.lastTs+1, r.syncs), nil
		} else if err != nil {
			return Event{}, err
		}
		return ev, nil
	}

	// Trace v2 parsing algorithm.
	//
	// (1) Read in all the batches for the next generation from the stream.
	//   (a) Use the size field in the header to quickly find all batches.
	// (2) Parse out the strings, stacks, CPU samples, and timestamp conversion data.
	// (3) Group each event batch by M, sorted by timestamp. (batchCursor contains the groups.)
	// (4) Organize batchCursors in a min-heap, ordered by the timestamp of the next event for each M.
	// (5) Try to advance the next event for the M at the top of the min-heap.
	//   (a) On success, select that M.
	//   (b) On failure, sort the min-heap and try to advance other Ms. Select the first M that advances.
	//   (c) If there's nothing left to advance, goto (1).
	// (6) Select the latest event for the selected M and get it ready to be returned.
	// (7) Read the next event for the selected M and update the min-heap.
	// (8) Return the selected event, goto (5) on the next call.

	// Set us up to track the last timestamp and fix up
	// the timestamp of any event that comes through.
	defer func() {
		if err != nil {
			return
		}
		if err = e.validateTableIDs(); err != nil {
			return
		}
		if e.base.time <= r.lastTs {
			e.base.time = r.lastTs + 1
		}
		r.lastTs = e.base.time
	}()

	// Consume any events in the ordering first.
	if ev, ok := r.order.Next(); ok {
		return ev, nil
	}

	// Check if we need to refresh the generation.
	if len(r.frontier) == 0 && len(r.cpuSamples) == 0 {
		if r.version < version.Go126 {
			return r.nextGenWithSpill()
		}
		if r.readGenErr != nil {
			return Event{}, r.readGenErr
		}
		gen, err := readGeneration(r.r, r.version)
		if err != nil {
			// Before returning an error, emit the sync event
			// for the current generation and queue up the error
			// for the next call.
			r.readGenErr = err
			r.gen = nil
			r.syncs++
			return syncEvent(nil, r.lastTs, r.syncs), nil
		}
		return r.installGen(gen)
	}
	tryAdvance := func(i int) (bool, error) {
		bc := r.frontier[i]

		if ok, err := r.order.Advance(&bc.ev, r.gen.evTable, bc.m, r.gen.gen); !ok || err != nil {
			return ok, err
		}

		// Refresh the cursor's event.
		ok, err := bc.nextEvent(r.gen.batches[bc.m], r.gen.freq)
		if err != nil {
			return false, err
		}
		if ok {
			// If we successfully refreshed, update the heap.
			heapUpdate(r.frontier, i)
		} else {
			// There's nothing else to read. Delete this cursor from the frontier.
			r.frontier = heapRemove(r.frontier, i)
		}
		return true, nil
	}
	// Inject a CPU sample if it comes next.
	if len(r.cpuSamples) != 0 {
		if len(r.frontier) == 0 || r.cpuSamples[0].time < r.frontier[0].ev.time {
			e := r.cpuSamples[0].asEvent(r.gen.evTable)
			r.cpuSamples = r.cpuSamples[1:]
			return e, nil
		}
	}
	// Try to advance the head of the frontier, which should have the minimum timestamp.
	// This should be by far the most common case
	if len(r.frontier) == 0 {
		return Event{}, fmt.Errorf("broken trace: frontier is empty:\n[gen=%d]\n\n%s\n%s\n", r.gen.gen, dumpFrontier(r.frontier), dumpOrdering(&r.order))
	}
	if ok, err := tryAdvance(0); err != nil {
		return Event{}, err
	} else if !ok {
		// Try to advance the rest of the frontier, in timestamp order.
		//
		// To do this, sort the min-heap. A sorted min-heap is still a
		// min-heap, but now we can iterate over the rest and try to
		// advance in order. This path should be rare.
		slices.SortFunc(r.frontier, (*batchCursor).compare)
		success := false
		for i := 1; i < len(r.frontier); i++ {
			if ok, err = tryAdvance(i); err != nil {
				return Event{}, err
			} else if ok {
				success = true
				break
			}
		}
		if !success {
			return Event{}, fmt.Errorf("broken trace: failed to advance: frontier:\n[gen=%d]\n\n%s\n%s\n", r.gen.gen, dumpFrontier(r.frontier), dumpOrdering(&r.order))
		}
	}

	// Pick off the next event on the queue. At this point, one must exist.
	ev, ok := r.order.Next()
	if !ok {
		panic("invariant violation: advance successful, but queue is empty")
	}
	return ev, nil
}

// nextGenWithSpill reads the generation and calls nextGen while
// also handling any spilled batches.
func (r *Reader) nextGenWithSpill() (Event, error) {
	if r.version >= version.Go126 {
		return Event{}, errors.New("internal error: nextGenWithSpill called for Go 1.26+ trace")
	}
	if r.spillErr != nil {
		if r.spillErrSync {
			return Event{}, r.spillErr
		}
		r.spillErrSync = true
		r.syncs++
		return syncEvent(nil, r.lastTs, r.syncs), nil
	}
	if r.gen != nil && r.spill == nil {
		// If we have a generation from the last read,
		// and there's nothing left in the frontier, and
		// there's no spilled batch, indicating that there's
		// no further generation, it means we're done.
		// Emit the final sync event.
		r.done = true
		r.syncs++
		return syncEvent(nil, r.lastTs, r.syncs), nil
	}

	// Read the next generation.
	var gen *generation
	gen, r.spill, r.spillErr = readGenerationWithSpill(r.r, r.spill, r.version)
	if gen == nil {
		r.gen = nil
		r.spillErrSync = true
		r.syncs++
		return syncEvent(nil, r.lastTs, r.syncs), nil
	}
	return r.installGen(gen)
}

// installGen installs the new generation into the Reader and returns
// a Sync event for the new generation.
func (r *Reader) installGen(gen *generation) (Event, error) {
	if gen == nil {
		// Emit the final sync event.
		r.gen = nil
		r.done = true
		r.syncs++
		return syncEvent(nil, r.lastTs, r.syncs), nil
	}
	r.gen = gen

	// Reset CPU samples cursor.
	r.cpuSamples = r.gen.cpuSamples

	// Reset frontier.
	for _, m := range r.gen.batchMs {
		batches := r.gen.batches[m]
		bc := &batchCursor{m: m}
		ok, err := bc.nextEvent(batches, r.gen.freq)
		if err != nil {
			return Event{}, err
		}
		if !ok {
			// Turns out there aren't actually any events in these batches.
			continue
		}
		r.frontier = heapInsert(r.frontier, bc)
	}
	r.syncs++

	// Always emit a sync event at the beginning of the generation.
	return syncEvent(r.gen.evTable, r.gen.freq.mul(r.gen.minTs), r.syncs), nil
}

func dumpFrontier(frontier []*batchCursor) string {
	var sb strings.Builder
	for _, bc := range frontier {
		spec := tracev2.Specs()[bc.ev.typ]
		fmt.Fprintf(&sb, "M %d [%s time=%d", bc.m, spec.Name, bc.ev.time)
		for i, arg := range spec.Args[1:] {
			fmt.Fprintf(&sb, " %s=%d", arg, bc.ev.args[i])
		}
		fmt.Fprintf(&sb, "]\n")
	}
	return sb.String()
}
