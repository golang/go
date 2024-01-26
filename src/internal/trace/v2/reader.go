// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bufio"
	"fmt"
	"io"
	"slices"
	"strings"

	"internal/trace/v2/event/go122"
	"internal/trace/v2/version"
)

// Reader reads a byte stream, validates it, and produces trace events.
type Reader struct {
	r           *bufio.Reader
	lastTs      Time
	gen         *generation
	spill       *spilledBatch
	frontier    []*batchCursor
	cpuSamples  []cpuSample
	order       ordering
	emittedSync bool
}

// NewReader creates a new trace reader.
func NewReader(r io.Reader) (*Reader, error) {
	br := bufio.NewReader(r)
	v, err := version.ReadHeader(br)
	if err != nil {
		return nil, err
	}
	if v != version.Go122 {
		return nil, fmt.Errorf("unknown or unsupported version go 1.%d", v)
	}
	return &Reader{
		r: br,
		order: ordering{
			mStates:     make(map[ThreadID]*mState),
			pStates:     make(map[ProcID]*pState),
			gStates:     make(map[GoID]*gState),
			activeTasks: make(map[TaskID]taskState),
		},
		// Don't emit a sync event when we first go to emit events.
		emittedSync: true,
	}, nil
}

// ReadEvent reads a single event from the stream.
//
// If the stream has been exhausted, it returns an invalid
// event and io.EOF.
func (r *Reader) ReadEvent() (e Event, err error) {
	// Go 1.22+ trace parsing algorithm.
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

	// Consume any extra events produced during parsing.
	if ev := r.order.consumeExtraEvent(); ev.Kind() != EventBad {
		return ev, nil
	}

	// Check if we need to refresh the generation.
	if len(r.frontier) == 0 && len(r.cpuSamples) == 0 {
		if !r.emittedSync {
			r.emittedSync = true
			return syncEvent(r.gen.evTable, r.lastTs), nil
		}
		if r.gen != nil && r.spill == nil {
			// If we have a generation from the last read,
			// and there's nothing left in the frontier, and
			// there's no spilled batch, indicating that there's
			// no further generation, it means we're done.
			// Return io.EOF.
			return Event{}, io.EOF
		}
		// Read the next generation.
		r.gen, r.spill, err = readGeneration(r.r, r.spill)
		if err != nil {
			return Event{}, err
		}

		// Reset CPU samples cursor.
		r.cpuSamples = r.gen.cpuSamples

		// Reset frontier.
		for m, batches := range r.gen.batches {
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

		// Reset emittedSync.
		r.emittedSync = false
	}
	refresh := func(i int) error {
		bc := r.frontier[i]

		// Refresh the cursor's event.
		ok, err := bc.nextEvent(r.gen.batches[bc.m], r.gen.freq)
		if err != nil {
			return err
		}
		if ok {
			// If we successfully refreshed, update the heap.
			heapUpdate(r.frontier, i)
		} else {
			// There's nothing else to read. Delete this cursor from the frontier.
			r.frontier = heapRemove(r.frontier, i)
		}
		return nil
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
	bc := r.frontier[0]
	if ctx, ok, err := r.order.advance(&bc.ev, r.gen.evTable, bc.m, r.gen.gen); err != nil {
		return Event{}, err
	} else if ok {
		e := Event{table: r.gen.evTable, ctx: ctx, base: bc.ev}
		return e, refresh(0)
	}
	// Sort the min-heap. A sorted min-heap is still a min-heap,
	// but now we can iterate over the rest and try to advance in
	// order. This path should be rare.
	slices.SortFunc(r.frontier, (*batchCursor).compare)
	// Try to advance the rest of the frontier, in timestamp order.
	for i := 1; i < len(r.frontier); i++ {
		bc := r.frontier[i]
		if ctx, ok, err := r.order.advance(&bc.ev, r.gen.evTable, bc.m, r.gen.gen); err != nil {
			return Event{}, err
		} else if ok {
			e := Event{table: r.gen.evTable, ctx: ctx, base: bc.ev}
			return e, refresh(i)
		}
	}
	return Event{}, fmt.Errorf("broken trace: failed to advance: frontier:\n[gen=%d]\n\n%s\n%s\n", r.gen.gen, dumpFrontier(r.frontier), dumpOrdering(&r.order))
}

func dumpFrontier(frontier []*batchCursor) string {
	var sb strings.Builder
	for _, bc := range frontier {
		spec := go122.Specs()[bc.ev.typ]
		fmt.Fprintf(&sb, "M %d [%s time=%d", bc.m, spec.Name, bc.ev.time)
		for i, arg := range spec.Args[1:] {
			fmt.Fprintf(&sb, " %s=%d", arg, bc.ev.args[i])
		}
		fmt.Fprintf(&sb, "]\n")
	}
	return sb.String()
}
