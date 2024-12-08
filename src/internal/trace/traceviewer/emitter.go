// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceviewer

import (
	"encoding/json"
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer/format"
	"io"
	"strconv"
	"time"
)

type TraceConsumer struct {
	ConsumeTimeUnit    func(unit string)
	ConsumeViewerEvent func(v *format.Event, required bool)
	ConsumeViewerFrame func(key string, f format.Frame)
	Flush              func()
}

// ViewerDataTraceConsumer returns a TraceConsumer that writes to w. The
// startIdx and endIdx are used for splitting large traces. They refer to
// indexes in the traceEvents output array, not the events in the trace input.
func ViewerDataTraceConsumer(w io.Writer, startIdx, endIdx int64) TraceConsumer {
	allFrames := make(map[string]format.Frame)
	requiredFrames := make(map[string]format.Frame)
	enc := json.NewEncoder(w)
	written := 0
	index := int64(-1)

	io.WriteString(w, "{")
	return TraceConsumer{
		ConsumeTimeUnit: func(unit string) {
			io.WriteString(w, `"displayTimeUnit":`)
			enc.Encode(unit)
			io.WriteString(w, ",")
		},
		ConsumeViewerEvent: func(v *format.Event, required bool) {
			index++
			if !required && (index < startIdx || index > endIdx) {
				// not in the range. Skip!
				return
			}
			WalkStackFrames(allFrames, v.Stack, func(id int) {
				s := strconv.Itoa(id)
				requiredFrames[s] = allFrames[s]
			})
			WalkStackFrames(allFrames, v.EndStack, func(id int) {
				s := strconv.Itoa(id)
				requiredFrames[s] = allFrames[s]
			})
			if written == 0 {
				io.WriteString(w, `"traceEvents": [`)
			}
			if written > 0 {
				io.WriteString(w, ",")
			}
			enc.Encode(v)
			// TODO(mknyszek): get rid of the extra \n inserted by enc.Encode.
			// Same should be applied to splittingTraceConsumer.
			written++
		},
		ConsumeViewerFrame: func(k string, v format.Frame) {
			allFrames[k] = v
		},
		Flush: func() {
			io.WriteString(w, `], "stackFrames":`)
			enc.Encode(requiredFrames)
			io.WriteString(w, `}`)
		},
	}
}

func SplittingTraceConsumer(max int) (*splitter, TraceConsumer) {
	type eventSz struct {
		Time   float64
		Sz     int
		Frames []int
	}

	var (
		// data.Frames contains only the frames for required events.
		data = format.Data{Frames: make(map[string]format.Frame)}

		allFrames = make(map[string]format.Frame)

		sizes []eventSz
		cw    countingWriter
	)

	s := new(splitter)

	return s, TraceConsumer{
		ConsumeTimeUnit: func(unit string) {
			data.TimeUnit = unit
		},
		ConsumeViewerEvent: func(v *format.Event, required bool) {
			if required {
				// Store required events inside data so flush
				// can include them in the required part of the
				// trace.
				data.Events = append(data.Events, v)
				WalkStackFrames(allFrames, v.Stack, func(id int) {
					s := strconv.Itoa(id)
					data.Frames[s] = allFrames[s]
				})
				WalkStackFrames(allFrames, v.EndStack, func(id int) {
					s := strconv.Itoa(id)
					data.Frames[s] = allFrames[s]
				})
				return
			}
			enc := json.NewEncoder(&cw)
			enc.Encode(v)
			size := eventSz{Time: v.Time, Sz: cw.size + 1} // +1 for ",".
			// Add referenced stack frames. Their size is computed
			// in flush, where we can dedup across events.
			WalkStackFrames(allFrames, v.Stack, func(id int) {
				size.Frames = append(size.Frames, id)
			})
			WalkStackFrames(allFrames, v.EndStack, func(id int) {
				size.Frames = append(size.Frames, id) // This may add duplicates. We'll dedup later.
			})
			sizes = append(sizes, size)
			cw.size = 0
		},
		ConsumeViewerFrame: func(k string, v format.Frame) {
			allFrames[k] = v
		},
		Flush: func() {
			// Calculate size of the mandatory part of the trace.
			// This includes thread names and stack frames for
			// required events.
			cw.size = 0
			enc := json.NewEncoder(&cw)
			enc.Encode(data)
			requiredSize := cw.size

			// Then calculate size of each individual event and
			// their stack frames, grouping them into ranges. We
			// only include stack frames relevant to the events in
			// the range to reduce overhead.

			var (
				start = 0

				eventsSize = 0

				frames     = make(map[string]format.Frame)
				framesSize = 0
			)
			for i, ev := range sizes {
				eventsSize += ev.Sz

				// Add required stack frames. Note that they
				// may already be in the map.
				for _, id := range ev.Frames {
					s := strconv.Itoa(id)
					_, ok := frames[s]
					if ok {
						continue
					}
					f := allFrames[s]
					frames[s] = f
					framesSize += stackFrameEncodedSize(uint(id), f)
				}

				total := requiredSize + framesSize + eventsSize
				if total < max {
					continue
				}

				// Reached max size, commit this range and
				// start a new range.
				startTime := time.Duration(sizes[start].Time * 1000)
				endTime := time.Duration(ev.Time * 1000)
				s.Ranges = append(s.Ranges, Range{
					Name:      fmt.Sprintf("%v-%v", startTime, endTime),
					Start:     start,
					End:       i + 1,
					StartTime: int64(startTime),
					EndTime:   int64(endTime),
				})
				start = i + 1
				frames = make(map[string]format.Frame)
				framesSize = 0
				eventsSize = 0
			}
			if len(s.Ranges) <= 1 {
				s.Ranges = nil
				return
			}

			if end := len(sizes) - 1; start < end {
				s.Ranges = append(s.Ranges, Range{
					Name:      fmt.Sprintf("%v-%v", time.Duration(sizes[start].Time*1000), time.Duration(sizes[end].Time*1000)),
					Start:     start,
					End:       end,
					StartTime: int64(sizes[start].Time * 1000),
					EndTime:   int64(sizes[end].Time * 1000),
				})
			}
		},
	}
}

type splitter struct {
	Ranges []Range
}

type countingWriter struct {
	size int
}

func (cw *countingWriter) Write(data []byte) (int, error) {
	cw.size += len(data)
	return len(data), nil
}

func stackFrameEncodedSize(id uint, f format.Frame) int {
	// We want to know the marginal size of traceviewer.Data.Frames for
	// each event. Running full JSON encoding of the map for each event is
	// far too slow.
	//
	// Since the format is fixed, we can easily compute the size without
	// encoding.
	//
	// A single entry looks like one of the following:
	//
	//   "1":{"name":"main.main:30"},
	//   "10":{"name":"pkg.NewSession:173","parent":9},
	//
	// The parent is omitted if 0. The trailing comma is omitted from the
	// last entry, but we don't need that much precision.
	const (
		baseSize = len(`"`) + len(`":{"name":"`) + len(`"},`)

		// Don't count the trailing quote on the name, as that is
		// counted in baseSize.
		parentBaseSize = len(`,"parent":`)
	)

	size := baseSize

	size += len(f.Name)

	// Bytes for id (always positive).
	for id > 0 {
		size += 1
		id /= 10
	}

	if f.Parent > 0 {
		size += parentBaseSize
		// Bytes for parent (always positive).
		for f.Parent > 0 {
			size += 1
			f.Parent /= 10
		}
	}

	return size
}

// WalkStackFrames calls fn for id and all of its parent frames from allFrames.
func WalkStackFrames(allFrames map[string]format.Frame, id int, fn func(id int)) {
	for id != 0 {
		f, ok := allFrames[strconv.Itoa(id)]
		if !ok {
			break
		}
		fn(id)
		id = f.Parent
	}
}

type Mode int

const (
	ModeGoroutineOriented Mode = 1 << iota
	ModeTaskOriented
	ModeThreadOriented // Mutually exclusive with ModeGoroutineOriented.
)

// NewEmitter returns a new Emitter that writes to c. The rangeStart and
// rangeEnd args are used for splitting large traces.
func NewEmitter(c TraceConsumer, rangeStart, rangeEnd time.Duration) *Emitter {
	c.ConsumeTimeUnit("ns")

	return &Emitter{
		c:          c,
		rangeStart: rangeStart,
		rangeEnd:   rangeEnd,
		frameTree:  frameNode{children: make(map[uint64]frameNode)},
		resources:  make(map[uint64]string),
		tasks:      make(map[uint64]task),
	}
}

type Emitter struct {
	c          TraceConsumer
	rangeStart time.Duration
	rangeEnd   time.Duration

	heapStats, prevHeapStats     heapStats
	gstates, prevGstates         [gStateCount]int64
	threadStats, prevThreadStats [threadStateCount]int64
	gomaxprocs                   uint64
	frameTree                    frameNode
	frameSeq                     int
	arrowSeq                     uint64
	filter                       func(uint64) bool
	resourceType                 string
	resources                    map[uint64]string
	focusResource                uint64
	tasks                        map[uint64]task
	asyncSliceSeq                uint64
}

type task struct {
	name      string
	sortIndex int
}

func (e *Emitter) Gomaxprocs(v uint64) {
	if v > e.gomaxprocs {
		e.gomaxprocs = v
	}
}

func (e *Emitter) Resource(id uint64, name string) {
	if e.filter != nil && !e.filter(id) {
		return
	}
	e.resources[id] = name
}

func (e *Emitter) SetResourceType(name string) {
	e.resourceType = name
}

func (e *Emitter) SetResourceFilter(filter func(uint64) bool) {
	e.filter = filter
}

func (e *Emitter) Task(id uint64, name string, sortIndex int) {
	e.tasks[id] = task{name, sortIndex}
}

func (e *Emitter) Slice(s SliceEvent) {
	if e.filter != nil && !e.filter(s.Resource) {
		return
	}
	e.slice(s, format.ProcsSection, "")
}

func (e *Emitter) TaskSlice(s SliceEvent) {
	e.slice(s, format.TasksSection, pickTaskColor(s.Resource))
}

func (e *Emitter) slice(s SliceEvent, sectionID uint64, cname string) {
	if !e.tsWithinRange(s.Ts) && !e.tsWithinRange(s.Ts+s.Dur) {
		return
	}
	e.OptionalEvent(&format.Event{
		Name:     s.Name,
		Phase:    "X",
		Time:     viewerTime(s.Ts),
		Dur:      viewerTime(s.Dur),
		PID:      sectionID,
		TID:      s.Resource,
		Stack:    s.Stack,
		EndStack: s.EndStack,
		Arg:      s.Arg,
		Cname:    cname,
	})
}

type SliceEvent struct {
	Name     string
	Ts       time.Duration
	Dur      time.Duration
	Resource uint64
	Stack    int
	EndStack int
	Arg      any
}

func (e *Emitter) AsyncSlice(s AsyncSliceEvent) {
	if !e.tsWithinRange(s.Ts) && !e.tsWithinRange(s.Ts+s.Dur) {
		return
	}
	if e.filter != nil && !e.filter(s.Resource) {
		return
	}
	cname := ""
	if s.TaskColorIndex != 0 {
		cname = pickTaskColor(s.TaskColorIndex)
	}
	e.asyncSliceSeq++
	e.OptionalEvent(&format.Event{
		Category: s.Category,
		Name:     s.Name,
		Phase:    "b",
		Time:     viewerTime(s.Ts),
		TID:      s.Resource,
		ID:       e.asyncSliceSeq,
		Scope:    s.Scope,
		Stack:    s.Stack,
		Cname:    cname,
	})
	e.OptionalEvent(&format.Event{
		Category: s.Category,
		Name:     s.Name,
		Phase:    "e",
		Time:     viewerTime(s.Ts + s.Dur),
		TID:      s.Resource,
		ID:       e.asyncSliceSeq,
		Scope:    s.Scope,
		Stack:    s.EndStack,
		Arg:      s.Arg,
		Cname:    cname,
	})
}

type AsyncSliceEvent struct {
	SliceEvent
	Category       string
	Scope          string
	TaskColorIndex uint64 // Take on the same color as the task with this ID.
}

func (e *Emitter) Instant(i InstantEvent) {
	if !e.tsWithinRange(i.Ts) {
		return
	}
	if e.filter != nil && !e.filter(i.Resource) {
		return
	}
	cname := ""
	e.OptionalEvent(&format.Event{
		Name:     i.Name,
		Category: i.Category,
		Phase:    "I",
		Scope:    "t",
		Time:     viewerTime(i.Ts),
		PID:      format.ProcsSection,
		TID:      i.Resource,
		Stack:    i.Stack,
		Cname:    cname,
		Arg:      i.Arg,
	})
}

type InstantEvent struct {
	Ts       time.Duration
	Name     string
	Category string
	Resource uint64
	Stack    int
	Arg      any
}

func (e *Emitter) Arrow(a ArrowEvent) {
	if e.filter != nil && (!e.filter(a.FromResource) || !e.filter(a.ToResource)) {
		return
	}
	e.arrow(a, format.ProcsSection)
}

func (e *Emitter) TaskArrow(a ArrowEvent) {
	e.arrow(a, format.TasksSection)
}

func (e *Emitter) arrow(a ArrowEvent, sectionID uint64) {
	if !e.tsWithinRange(a.Start) || !e.tsWithinRange(a.End) {
		return
	}
	e.arrowSeq++
	e.OptionalEvent(&format.Event{
		Name:  a.Name,
		Phase: "s",
		TID:   a.FromResource,
		PID:   sectionID,
		ID:    e.arrowSeq,
		Time:  viewerTime(a.Start),
		Stack: a.FromStack,
	})
	e.OptionalEvent(&format.Event{
		Name:  a.Name,
		Phase: "t",
		TID:   a.ToResource,
		PID:   sectionID,
		ID:    e.arrowSeq,
		Time:  viewerTime(a.End),
	})
}

type ArrowEvent struct {
	Name         string
	Start        time.Duration
	End          time.Duration
	FromResource uint64
	FromStack    int
	ToResource   uint64
}

func (e *Emitter) Event(ev *format.Event) {
	e.c.ConsumeViewerEvent(ev, true)
}

func (e *Emitter) HeapAlloc(ts time.Duration, v uint64) {
	e.heapStats.heapAlloc = v
	e.emitHeapCounters(ts)
}

func (e *Emitter) Focus(id uint64) {
	e.focusResource = id
}

func (e *Emitter) GoroutineTransition(ts time.Duration, from, to GState) {
	e.gstates[from]--
	e.gstates[to]++
	if e.prevGstates == e.gstates {
		return
	}
	if e.tsWithinRange(ts) {
		e.OptionalEvent(&format.Event{
			Name:  "Goroutines",
			Phase: "C",
			Time:  viewerTime(ts),
			PID:   1,
			Arg: &format.GoroutineCountersArg{
				Running:   uint64(e.gstates[GRunning]),
				Runnable:  uint64(e.gstates[GRunnable]),
				GCWaiting: uint64(e.gstates[GWaitingGC]),
			},
		})
	}
	e.prevGstates = e.gstates
}

func (e *Emitter) IncThreadStateCount(ts time.Duration, state ThreadState, delta int64) {
	e.threadStats[state] += delta
	if e.prevThreadStats == e.threadStats {
		return
	}
	if e.tsWithinRange(ts) {
		e.OptionalEvent(&format.Event{
			Name:  "Threads",
			Phase: "C",
			Time:  viewerTime(ts),
			PID:   1,
			Arg: &format.ThreadCountersArg{
				Running:   int64(e.threadStats[ThreadStateRunning]),
				InSyscall: int64(e.threadStats[ThreadStateInSyscall]),
				// TODO(mknyszek): Why is InSyscallRuntime not included here?
			},
		})
	}
	e.prevThreadStats = e.threadStats
}

func (e *Emitter) HeapGoal(ts time.Duration, v uint64) {
	// This cutoff at 1 PiB is a Workaround for https://github.com/golang/go/issues/63864.
	//
	// TODO(mknyszek): Remove this once the problem has been fixed.
	const PB = 1 << 50
	if v > PB {
		v = 0
	}
	e.heapStats.nextGC = v
	e.emitHeapCounters(ts)
}

func (e *Emitter) emitHeapCounters(ts time.Duration) {
	if e.prevHeapStats == e.heapStats {
		return
	}
	diff := uint64(0)
	if e.heapStats.nextGC > e.heapStats.heapAlloc {
		diff = e.heapStats.nextGC - e.heapStats.heapAlloc
	}
	if e.tsWithinRange(ts) {
		e.OptionalEvent(&format.Event{
			Name:  "Heap",
			Phase: "C",
			Time:  viewerTime(ts),
			PID:   1,
			Arg:   &format.HeapCountersArg{Allocated: e.heapStats.heapAlloc, NextGC: diff},
		})
	}
	e.prevHeapStats = e.heapStats
}

// Err returns an error if the emitter is in an invalid state.
func (e *Emitter) Err() error {
	if e.gstates[GRunnable] < 0 || e.gstates[GRunning] < 0 || e.threadStats[ThreadStateInSyscall] < 0 || e.threadStats[ThreadStateInSyscallRuntime] < 0 {
		return fmt.Errorf(
			"runnable=%d running=%d insyscall=%d insyscallRuntime=%d",
			e.gstates[GRunnable],
			e.gstates[GRunning],
			e.threadStats[ThreadStateInSyscall],
			e.threadStats[ThreadStateInSyscallRuntime],
		)
	}
	return nil
}

func (e *Emitter) tsWithinRange(ts time.Duration) bool {
	return e.rangeStart <= ts && ts <= e.rangeEnd
}

// OptionalEvent emits ev if it's within the time range of the consumer, i.e.
// the selected trace split range.
func (e *Emitter) OptionalEvent(ev *format.Event) {
	e.c.ConsumeViewerEvent(ev, false)
}

func (e *Emitter) Flush() {
	e.processMeta(format.StatsSection, "STATS", 0)

	if len(e.tasks) != 0 {
		e.processMeta(format.TasksSection, "TASKS", 1)
	}
	for id, task := range e.tasks {
		e.threadMeta(format.TasksSection, id, task.name, task.sortIndex)
	}

	e.processMeta(format.ProcsSection, e.resourceType, 2)

	e.threadMeta(format.ProcsSection, trace.GCP, "GC", -6)
	e.threadMeta(format.ProcsSection, trace.NetpollP, "Network", -5)
	e.threadMeta(format.ProcsSection, trace.TimerP, "Timers", -4)
	e.threadMeta(format.ProcsSection, trace.SyscallP, "Syscalls", -3)

	for id, name := range e.resources {
		priority := int(id)
		if e.focusResource != 0 && id == e.focusResource {
			// Put the focus goroutine on top.
			priority = -2
		}
		e.threadMeta(format.ProcsSection, id, name, priority)
	}

	e.c.Flush()
}

func (e *Emitter) threadMeta(sectionID, tid uint64, name string, priority int) {
	e.Event(&format.Event{
		Name:  "thread_name",
		Phase: "M",
		PID:   sectionID,
		TID:   tid,
		Arg:   &format.NameArg{Name: name},
	})
	e.Event(&format.Event{
		Name:  "thread_sort_index",
		Phase: "M",
		PID:   sectionID,
		TID:   tid,
		Arg:   &format.SortIndexArg{Index: priority},
	})
}

func (e *Emitter) processMeta(sectionID uint64, name string, priority int) {
	e.Event(&format.Event{
		Name:  "process_name",
		Phase: "M",
		PID:   sectionID,
		Arg:   &format.NameArg{Name: name},
	})
	e.Event(&format.Event{
		Name:  "process_sort_index",
		Phase: "M",
		PID:   sectionID,
		Arg:   &format.SortIndexArg{Index: priority},
	})
}

// Stack emits the given frames and returns a unique id for the stack. No
// pointers to the given data are being retained beyond the call to Stack.
func (e *Emitter) Stack(stk []*trace.Frame) int {
	return e.buildBranch(e.frameTree, stk)
}

// buildBranch builds one branch in the prefix tree rooted at ctx.frameTree.
func (e *Emitter) buildBranch(parent frameNode, stk []*trace.Frame) int {
	if len(stk) == 0 {
		return parent.id
	}
	last := len(stk) - 1
	frame := stk[last]
	stk = stk[:last]

	node, ok := parent.children[frame.PC]
	if !ok {
		e.frameSeq++
		node.id = e.frameSeq
		node.children = make(map[uint64]frameNode)
		parent.children[frame.PC] = node
		e.c.ConsumeViewerFrame(strconv.Itoa(node.id), format.Frame{Name: fmt.Sprintf("%v:%v", frame.Fn, frame.Line), Parent: parent.id})
	}
	return e.buildBranch(node, stk)
}

type heapStats struct {
	heapAlloc uint64
	nextGC    uint64
}

func viewerTime(t time.Duration) float64 {
	return float64(t) / float64(time.Microsecond)
}

type GState int

const (
	GDead GState = iota
	GRunnable
	GRunning
	GWaiting
	GWaitingGC

	gStateCount
)

type ThreadState int

const (
	ThreadStateInSyscall ThreadState = iota
	ThreadStateInSyscallRuntime
	ThreadStateRunning

	threadStateCount
)

type frameNode struct {
	id       int
	children map[uint64]frameNode
}

// Mapping from more reasonable color names to the reserved color names in
// https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html#L50
// The chrome trace viewer allows only those as cname values.
const (
	colorLightMauve     = "thread_state_uninterruptible" // 182, 125, 143
	colorOrange         = "thread_state_iowait"          // 255, 140, 0
	colorSeafoamGreen   = "thread_state_running"         // 126, 200, 148
	colorVistaBlue      = "thread_state_runnable"        // 133, 160, 210
	colorTan            = "thread_state_unknown"         // 199, 155, 125
	colorIrisBlue       = "background_memory_dump"       // 0, 180, 180
	colorMidnightBlue   = "light_memory_dump"            // 0, 0, 180
	colorDeepMagenta    = "detailed_memory_dump"         // 180, 0, 180
	colorBlue           = "vsync_highlight_color"        // 0, 0, 255
	colorGrey           = "generic_work"                 // 125, 125, 125
	colorGreen          = "good"                         // 0, 125, 0
	colorDarkGoldenrod  = "bad"                          // 180, 125, 0
	colorPeach          = "terrible"                     // 180, 0, 0
	colorBlack          = "black"                        // 0, 0, 0
	colorLightGrey      = "grey"                         // 221, 221, 221
	colorWhite          = "white"                        // 255, 255, 255
	colorYellow         = "yellow"                       // 255, 255, 0
	colorOlive          = "olive"                        // 100, 100, 0
	colorCornflowerBlue = "rail_response"                // 67, 135, 253
	colorSunsetOrange   = "rail_animation"               // 244, 74, 63
	colorTangerine      = "rail_idle"                    // 238, 142, 0
	colorShamrockGreen  = "rail_load"                    // 13, 168, 97
	colorGreenishYellow = "startup"                      // 230, 230, 0
	colorDarkGrey       = "heap_dump_stack_frame"        // 128, 128, 128
	colorTawny          = "heap_dump_child_node_arrow"   // 204, 102, 0
	colorLemon          = "cq_build_running"             // 255, 255, 119
	colorLime           = "cq_build_passed"              // 153, 238, 102
	colorPink           = "cq_build_failed"              // 238, 136, 136
	colorSilver         = "cq_build_abandoned"           // 187, 187, 187
	colorManzGreen      = "cq_build_attempt_runnig"      // 222, 222, 75
	colorKellyGreen     = "cq_build_attempt_passed"      // 108, 218, 35
	colorAnotherGrey    = "cq_build_attempt_failed"      // 187, 187, 187
)

var colorForTask = []string{
	colorLightMauve,
	colorOrange,
	colorSeafoamGreen,
	colorVistaBlue,
	colorTan,
	colorMidnightBlue,
	colorIrisBlue,
	colorDeepMagenta,
	colorGreen,
	colorDarkGoldenrod,
	colorPeach,
	colorOlive,
	colorCornflowerBlue,
	colorSunsetOrange,
	colorTangerine,
	colorShamrockGreen,
	colorTawny,
	colorLemon,
	colorLime,
	colorPink,
	colorSilver,
	colorManzGreen,
	colorKellyGreen,
}

func pickTaskColor(id uint64) string {
	idx := id % uint64(len(colorForTask))
	return colorForTask[idx]
}
