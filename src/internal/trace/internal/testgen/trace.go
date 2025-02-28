// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testgen

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"internal/trace"
	"internal/trace/raw"
	"internal/trace/tracev2"
	"internal/trace/version"
	"internal/txtar"
)

func Main(ver version.Version, f func(*Trace)) {
	// Create an output file.
	out, err := os.Create(os.Args[1])
	if err != nil {
		panic(err.Error())
	}
	defer out.Close()

	// Create a new trace.
	trace := NewTrace(ver)

	// Call the generator.
	f(trace)

	// Write out the generator's state.
	if _, err := out.Write(trace.Generate()); err != nil {
		panic(err.Error())
	}
}

// Trace represents an execution trace for testing.
//
// It does a little bit of work to ensure that the produced trace is valid,
// just for convenience. It mainly tracks batches and batch sizes (so they're
// trivially correct), tracks strings and stacks, and makes sure emitted string
// and stack batches are valid. That last part can be controlled by a few options.
//
// Otherwise, it performs no validation on the trace at all.
type Trace struct {
	// Trace data state.
	ver             version.Version
	names           map[string]tracev2.EventType
	specs           []tracev2.EventSpec
	events          []raw.Event
	gens            []*Generation
	validTimestamps bool
	lastTs          Time

	// Expectation state.
	bad      bool
	badMatch *regexp.Regexp
}

// NewTrace creates a new trace.
func NewTrace(ver version.Version) *Trace {
	return &Trace{
		names:           tracev2.EventNames(ver.Specs()),
		specs:           ver.Specs(),
		ver:             ver,
		validTimestamps: true,
	}
}

// ExpectFailure writes down that the trace should be broken. The caller
// must provide a pattern matching the expected error produced by the parser.
func (t *Trace) ExpectFailure(pattern string) {
	t.bad = true
	t.badMatch = regexp.MustCompile(pattern)
}

// ExpectSuccess writes down that the trace should successfully parse.
func (t *Trace) ExpectSuccess() {
	t.bad = false
}

// RawEvent emits an event into the trace. name must correspond to one
// of the names in Specs() result for the version that was passed to
// this trace.
func (t *Trace) RawEvent(typ tracev2.EventType, data []byte, args ...uint64) {
	t.events = append(t.events, t.createEvent(typ, data, args...))
}

// DisableTimestamps makes the timestamps for all events generated after
// this call zero. Raw events are exempted from this because the caller
// has to pass their own timestamp into those events anyway.
func (t *Trace) DisableTimestamps() {
	t.validTimestamps = false
}

// Generation creates a new trace generation.
//
// This provides more structure than Event to allow for more easily
// creating complex traces that are mostly or completely correct.
func (t *Trace) Generation(gen uint64) *Generation {
	g := &Generation{
		trace:   t,
		gen:     gen,
		strings: make(map[string]uint64),
		stacks:  make(map[stack]uint64),
		sync:    sync{freq: 15625000},
	}
	t.gens = append(t.gens, g)
	return g
}

// Generate creates a test file for the trace.
func (t *Trace) Generate() []byte {
	// Trace file contents.
	var buf bytes.Buffer
	tw, err := raw.NewTextWriter(&buf, t.ver)
	if err != nil {
		panic(err.Error())
	}

	// Write raw top-level events.
	for _, e := range t.events {
		tw.WriteEvent(e)
	}

	// Write generations.
	for _, g := range t.gens {
		g.writeEventsTo(tw)
	}

	// Expectation file contents.
	expect := []byte("SUCCESS\n")
	if t.bad {
		expect = []byte(fmt.Sprintf("FAILURE %q\n", t.badMatch))
	}

	// Create the test file's contents.
	return txtar.Format(&txtar.Archive{
		Files: []txtar.File{
			{Name: "expect", Data: expect},
			{Name: "trace", Data: buf.Bytes()},
		},
	})
}

func (t *Trace) createEvent(ev tracev2.EventType, data []byte, args ...uint64) raw.Event {
	spec := t.specs[ev]
	if ev != tracev2.EvStack {
		if arity := len(spec.Args); len(args) != arity {
			panic(fmt.Sprintf("expected %d args for %s, got %d", arity, spec.Name, len(args)))
		}
	}
	return raw.Event{
		Version: t.ver,
		Ev:      ev,
		Args:    args,
		Data:    data,
	}
}

type stack struct {
	stk [32]trace.StackFrame
	len int
}

var (
	NoString = ""
	NoStack  = []trace.StackFrame{}
)

// Generation represents a single generation in the trace.
type Generation struct {
	trace   *Trace
	gen     uint64
	batches []*Batch
	strings map[string]uint64
	stacks  map[stack]uint64
	sync    sync

	// Options applied when Trace.Generate is called.
	ignoreStringBatchSizeLimit bool
	ignoreStackBatchSizeLimit  bool
}

// Batch starts a new event batch in the trace data.
//
// This is convenience function for generating correct batches.
func (g *Generation) Batch(thread trace.ThreadID, time Time) *Batch {
	b := &Batch{
		gen:    g,
		thread: thread,
	}
	b.setTimestamp(time)
	g.batches = append(g.batches, b)
	return b
}

// String registers a string with the trace.
//
// This is a convenience function for easily adding correct
// strings to traces.
func (g *Generation) String(s string) uint64 {
	if len(s) == 0 {
		return 0
	}
	if id, ok := g.strings[s]; ok {
		return id
	}
	id := uint64(len(g.strings) + 1)
	g.strings[s] = id
	return id
}

// Stack registers a stack with the trace.
//
// This is a convenience function for easily adding correct
// stacks to traces.
func (g *Generation) Stack(stk []trace.StackFrame) uint64 {
	if len(stk) == 0 {
		return 0
	}
	if len(stk) > 32 {
		panic("stack too big for test")
	}
	var stkc stack
	copy(stkc.stk[:], stk)
	stkc.len = len(stk)
	if id, ok := g.stacks[stkc]; ok {
		return id
	}
	id := uint64(len(g.stacks) + 1)
	g.stacks[stkc] = id
	return id
}

// Sync configures the sync batch for the generation. For go1.25 and later,
// the time value is the timestamp of the EvClockSnapshot event. For earlier
// version, the time value is the timestamp of the batch containing a lone
// EvFrequency event.
func (g *Generation) Sync(freq uint64, time Time, mono uint64, wall time.Time) {
	if g.trace.ver < version.Go125 && (mono != 0 || !wall.IsZero()) {
		panic(fmt.Sprintf("mono and wall args are not supported in go1.%d traces", g.trace.ver))
	}
	g.sync = sync{
		freq:     freq,
		time:     time,
		mono:     mono,
		walltime: wall,
	}
}

type sync struct {
	freq     uint64
	time     Time
	mono     uint64
	walltime time.Time
}

// writeEventsTo emits event batches in the generation to tw.
func (g *Generation) writeEventsTo(tw *raw.TextWriter) {
	// go1.25+ sync batches are emitted at the start of the generation.
	if g.trace.ver >= version.Go125 {
		b := g.newStructuralBatch()
		// Arrange for EvClockSnapshot's ts to be exactly g.sync.time.
		b.setTimestamp(g.sync.time - 1)
		b.RawEvent(tracev2.EvSync, nil)
		b.RawEvent(tracev2.EvFrequency, nil, g.sync.freq)
		sec := uint64(g.sync.walltime.Unix())
		nsec := uint64(g.sync.walltime.Nanosecond())
		b.Event("ClockSnapshot", g.sync.mono, sec, nsec)
		b.writeEventsTo(tw)
	}

	// Write event batches for the generation.
	for _, b := range g.batches {
		b.writeEventsTo(tw)
	}

	// Write lone EvFrequency sync batch for older traces.
	if g.trace.ver < version.Go125 {
		b := g.newStructuralBatch()
		b.setTimestamp(g.sync.time)
		b.RawEvent(tracev2.EvFrequency, nil, g.sync.freq)
		b.writeEventsTo(tw)
	}

	// Write stacks.
	b := g.newStructuralBatch()
	b.RawEvent(tracev2.EvStacks, nil)
	for stk, id := range g.stacks {
		stk := stk.stk[:stk.len]
		args := []uint64{id}
		for _, f := range stk {
			args = append(args, f.PC, g.String(f.Func), g.String(f.File), f.Line)
		}
		b.RawEvent(tracev2.EvStack, nil, args...)

		// Flush the batch if necessary.
		if !g.ignoreStackBatchSizeLimit && b.size > tracev2.MaxBatchSize/2 {
			b.writeEventsTo(tw)
			b = g.newStructuralBatch()
		}
	}
	b.writeEventsTo(tw)

	// Write strings.
	b = g.newStructuralBatch()
	b.RawEvent(tracev2.EvStrings, nil)
	for s, id := range g.strings {
		b.RawEvent(tracev2.EvString, []byte(s), id)

		// Flush the batch if necessary.
		if !g.ignoreStringBatchSizeLimit && b.size > tracev2.MaxBatchSize/2 {
			b.writeEventsTo(tw)
			b = g.newStructuralBatch()
		}
	}
	b.writeEventsTo(tw)
}

func (g *Generation) newStructuralBatch() *Batch {
	b := &Batch{gen: g, thread: trace.NoThread}
	b.setTimestamp(g.trace.lastTs + 1)
	return b
}

// Batch represents an event batch.
type Batch struct {
	gen       *Generation
	thread    trace.ThreadID
	timestamp Time
	size      uint64
	events    []raw.Event
}

// Event emits an event into a batch. name must correspond to one
// of the names in Specs() result for the version that was passed to
// this trace. Callers must omit the timestamp delta.
func (b *Batch) Event(name string, args ...any) {
	ev, ok := b.gen.trace.names[name]
	if !ok {
		panic(fmt.Sprintf("invalid or unknown event %s", name))
	}
	var uintArgs []uint64
	argOff := 0
	if b.gen.trace.specs[ev].IsTimedEvent {
		if b.gen.trace.validTimestamps {
			uintArgs = []uint64{1}
			b.gen.trace.lastTs += 1
		} else {
			uintArgs = []uint64{0}
		}
		argOff = 1
	}
	spec := b.gen.trace.specs[ev]
	if arity := len(spec.Args) - argOff; len(args) != arity {
		panic(fmt.Sprintf("expected %d args for %s, got %d", arity, spec.Name, len(args)))
	}
	for i, arg := range args {
		uintArgs = append(uintArgs, b.uintArgFor(arg, spec.Args[i+argOff]))
	}
	b.RawEvent(ev, nil, uintArgs...)
}

func (b *Batch) uintArgFor(arg any, argSpec string) uint64 {
	components := strings.SplitN(argSpec, "_", 2)
	typStr := components[0]
	if len(components) == 2 {
		typStr = components[1]
	}
	var u uint64
	switch typStr {
	case "value", "mono", "sec", "nsec":
		u = arg.(uint64)
	case "stack":
		u = b.gen.Stack(arg.([]trace.StackFrame))
	case "seq":
		u = uint64(arg.(Seq))
	case "pstatus":
		u = uint64(arg.(tracev2.ProcStatus))
	case "gstatus":
		u = uint64(arg.(tracev2.GoStatus))
	case "g":
		u = uint64(arg.(trace.GoID))
	case "m":
		u = uint64(arg.(trace.ThreadID))
	case "p":
		u = uint64(arg.(trace.ProcID))
	case "string":
		u = b.gen.String(arg.(string))
	case "task":
		u = uint64(arg.(trace.TaskID))
	default:
		panic(fmt.Sprintf("unsupported arg type %q for spec %q", typStr, argSpec))
	}
	return u
}

// RawEvent emits an event into a batch. name must correspond to one
// of the names in Specs() result for the version that was passed to
// this trace.
func (b *Batch) RawEvent(typ tracev2.EventType, data []byte, args ...uint64) {
	ev := b.gen.trace.createEvent(typ, data, args...)

	// Compute the size of the event and add it to the batch.
	b.size += 1 // One byte for the event header.
	var buf [binary.MaxVarintLen64]byte
	for _, arg := range args {
		b.size += uint64(binary.PutUvarint(buf[:], arg))
	}
	if len(data) != 0 {
		b.size += uint64(binary.PutUvarint(buf[:], uint64(len(data))))
		b.size += uint64(len(data))
	}

	// Add the event.
	b.events = append(b.events, ev)
}

// writeEventsTo emits events in the batch, including the batch header, to tw.
func (b *Batch) writeEventsTo(tw *raw.TextWriter) {
	tw.WriteEvent(raw.Event{
		Version: b.gen.trace.ver,
		Ev:      tracev2.EvEventBatch,
		Args:    []uint64{b.gen.gen, uint64(b.thread), uint64(b.timestamp), b.size},
	})
	for _, e := range b.events {
		tw.WriteEvent(e)
	}
}

// setTimestamp sets the timestamp for the batch.
func (b *Batch) setTimestamp(t Time) {
	if b.gen.trace.validTimestamps {
		b.timestamp = t
		b.gen.trace.lastTs = t
	}
}

// Seq represents a sequence counter.
type Seq uint64

// Time represents a low-level trace timestamp (which does not necessarily
// correspond to nanoseconds, like trace.Time does).
type Time uint64
