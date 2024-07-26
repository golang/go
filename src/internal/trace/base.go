// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains data types that all implementations of the trace format
// parser need to provide to the rest of the package.

package trace

import (
	"fmt"
	"math"
	"strings"

	"internal/trace/event"
	"internal/trace/event/go122"
	"internal/trace/version"
)

// maxArgs is the maximum number of arguments for "plain" events,
// i.e. anything that could reasonably be represented as a baseEvent.
const maxArgs = 5

// timedEventArgs is an array that is able to hold the arguments for any
// timed event.
type timedEventArgs [maxArgs - 1]uint64

// baseEvent is the basic unprocessed event. This serves as a common
// fundamental data structure across.
type baseEvent struct {
	typ  event.Type
	time Time
	args timedEventArgs
}

// extra returns a slice representing extra available space in args
// that the parser can use to pass data up into Event.
func (e *baseEvent) extra(v version.Version) []uint64 {
	switch v {
	case version.Go122:
		return e.args[len(go122.Specs()[e.typ].Args)-1:]
	}
	panic(fmt.Sprintf("unsupported version: go 1.%d", v))
}

// evTable contains the per-generation data necessary to
// interpret an individual event.
type evTable struct {
	freq    frequency
	strings dataTable[stringID, string]
	stacks  dataTable[stackID, stack]
	pcs     map[uint64]frame

	// extraStrings are strings that get generated during
	// parsing but haven't come directly from the trace, so
	// they don't appear in strings.
	extraStrings   []string
	extraStringIDs map[string]extraStringID
	nextExtra      extraStringID

	// expData contains extra unparsed data that is accessible
	// only to ExperimentEvent via an EventExperimental event.
	expData map[event.Experiment]*ExperimentalData
}

// addExtraString adds an extra string to the evTable and returns
// a unique ID for the string in the table.
func (t *evTable) addExtraString(s string) extraStringID {
	if s == "" {
		return 0
	}
	if t.extraStringIDs == nil {
		t.extraStringIDs = make(map[string]extraStringID)
	}
	if id, ok := t.extraStringIDs[s]; ok {
		return id
	}
	t.nextExtra++
	id := t.nextExtra
	t.extraStrings = append(t.extraStrings, s)
	t.extraStringIDs[s] = id
	return id
}

// getExtraString returns the extra string for the provided ID.
// The ID must have been produced by addExtraString for this evTable.
func (t *evTable) getExtraString(id extraStringID) string {
	if id == 0 {
		return ""
	}
	return t.extraStrings[id-1]
}

// dataTable is a mapping from EIs to Es.
type dataTable[EI ~uint64, E any] struct {
	present []uint8
	dense   []E
	sparse  map[EI]E
}

// insert tries to add a mapping from id to s.
//
// Returns an error if a mapping for id already exists, regardless
// of whether or not s is the same in content. This should be used
// for validation during parsing.
func (d *dataTable[EI, E]) insert(id EI, data E) error {
	if d.sparse == nil {
		d.sparse = make(map[EI]E)
	}
	if existing, ok := d.get(id); ok {
		return fmt.Errorf("multiple %Ts with the same ID: id=%d, new=%v, existing=%v", data, id, data, existing)
	}
	d.sparse[id] = data
	return nil
}

// compactify attempts to compact sparse into dense.
//
// This is intended to be called only once after insertions are done.
func (d *dataTable[EI, E]) compactify() {
	if d.sparse == nil || len(d.dense) != 0 {
		// Already compactified.
		return
	}
	// Find the range of IDs.
	maxID := EI(0)
	minID := ^EI(0)
	for id := range d.sparse {
		if id > maxID {
			maxID = id
		}
		if id < minID {
			minID = id
		}
	}
	if maxID >= math.MaxInt {
		// We can't create a slice big enough to hold maxID elements
		return
	}
	// We're willing to waste at most 2x memory.
	if int(maxID-minID) > max(len(d.sparse), 2*len(d.sparse)) {
		return
	}
	if int(minID) > len(d.sparse) {
		return
	}
	size := int(maxID) + 1
	d.present = make([]uint8, (size+7)/8)
	d.dense = make([]E, size)
	for id, data := range d.sparse {
		d.dense[id] = data
		d.present[id/8] |= uint8(1) << (id % 8)
	}
	d.sparse = nil
}

// get returns the E for id or false if it doesn't
// exist. This should be used for validation during parsing.
func (d *dataTable[EI, E]) get(id EI) (E, bool) {
	if id == 0 {
		return *new(E), true
	}
	if uint64(id) < uint64(len(d.dense)) {
		if d.present[id/8]&(uint8(1)<<(id%8)) != 0 {
			return d.dense[id], true
		}
	} else if d.sparse != nil {
		if data, ok := d.sparse[id]; ok {
			return data, true
		}
	}
	return *new(E), false
}

// forEach iterates over all ID/value pairs in the data table.
func (d *dataTable[EI, E]) forEach(yield func(EI, E) bool) bool {
	for id, value := range d.dense {
		if d.present[id/8]&(uint8(1)<<(id%8)) == 0 {
			continue
		}
		if !yield(EI(id), value) {
			return false
		}
	}
	if d.sparse == nil {
		return true
	}
	for id, value := range d.sparse {
		if !yield(id, value) {
			return false
		}
	}
	return true
}

// mustGet returns the E for id or panics if it fails.
//
// This should only be used if id has already been validated.
func (d *dataTable[EI, E]) mustGet(id EI) E {
	data, ok := d.get(id)
	if !ok {
		panic(fmt.Sprintf("expected id %d in %T table", id, data))
	}
	return data
}

// frequency is nanoseconds per timestamp unit.
type frequency float64

// mul multiplies an unprocessed to produce a time in nanoseconds.
func (f frequency) mul(t timestamp) Time {
	return Time(float64(t) * float64(f))
}

// stringID is an index into the string table for a generation.
type stringID uint64

// extraStringID is an index into the extra string table for a generation.
type extraStringID uint64

// stackID is an index into the stack table for a generation.
type stackID uint64

// cpuSample represents a CPU profiling sample captured by the trace.
type cpuSample struct {
	schedCtx
	time  Time
	stack stackID
}

// asEvent produces a complete Event from a cpuSample. It needs
// the evTable from the generation that created it.
//
// We don't just store it as an Event in generation to minimize
// the amount of pointer data floating around.
func (s cpuSample) asEvent(table *evTable) Event {
	// TODO(mknyszek): This is go122-specific, but shouldn't be.
	// Generalize this in the future.
	e := Event{
		table: table,
		ctx:   s.schedCtx,
		base: baseEvent{
			typ:  go122.EvCPUSample,
			time: s.time,
		},
	}
	e.base.args[0] = uint64(s.stack)
	return e
}

// stack represents a goroutine stack sample.
type stack struct {
	pcs []uint64
}

func (s stack) String() string {
	var sb strings.Builder
	for _, frame := range s.pcs {
		fmt.Fprintf(&sb, "\t%#v\n", frame)
	}
	return sb.String()
}

// frame represents a single stack frame.
type frame struct {
	pc     uint64
	funcID stringID
	fileID stringID
	line   uint64
}
