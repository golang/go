// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	"log"
	"math"
	"net/http"
	"runtime/debug"
	"sort"
	"strconv"
	"time"

	"internal/trace/traceviewer/format"
)

func init() {
	http.HandleFunc("/trace", httpTrace)
	http.HandleFunc("/jsontrace", httpJsonTrace)
	http.Handle("/static/", traceviewer.StaticHandler())
}

// httpTrace serves either whole trace (goid==0) or trace for goid goroutine.
func httpTrace(w http.ResponseWriter, r *http.Request) {
	_, err := parseTrace()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	traceviewer.TraceHandler().ServeHTTP(w, r)
}

// httpJsonTrace serves json trace, requested from within templTrace HTML.
func httpJsonTrace(w http.ResponseWriter, r *http.Request) {
	defer debug.FreeOSMemory()
	defer reportMemoryUsage("after httpJsonTrace")
	// This is an AJAX handler, so instead of http.Error we use log.Printf to log errors.
	res, err := parseTrace()
	if err != nil {
		log.Printf("failed to parse trace: %v", err)
		return
	}

	params := &traceParams{
		parsed:  res,
		endTime: math.MaxInt64,
	}

	if goids := r.FormValue("goid"); goids != "" {
		// If goid argument is present, we are rendering a trace for this particular goroutine.
		goid, err := strconv.ParseUint(goids, 10, 64)
		if err != nil {
			log.Printf("failed to parse goid parameter %q: %v", goids, err)
			return
		}
		analyzeGoroutines(res.Events)
		g, ok := gs[goid]
		if !ok {
			log.Printf("failed to find goroutine %d", goid)
			return
		}
		params.mode = traceviewer.ModeGoroutineOriented
		params.startTime = g.StartTime
		if g.EndTime != 0 {
			params.endTime = g.EndTime
		} else { // The goroutine didn't end.
			params.endTime = lastTimestamp()
		}
		params.maing = goid
		params.gs = trace.RelatedGoroutines(res.Events, goid)
	} else if taskids := r.FormValue("taskid"); taskids != "" {
		taskid, err := strconv.ParseUint(taskids, 10, 64)
		if err != nil {
			log.Printf("failed to parse taskid parameter %q: %v", taskids, err)
			return
		}
		annotRes, _ := analyzeAnnotations()
		task, ok := annotRes.tasks[taskid]
		if !ok || len(task.events) == 0 {
			log.Printf("failed to find task with id %d", taskid)
			return
		}
		goid := task.events[0].G
		params.mode = traceviewer.ModeGoroutineOriented | traceviewer.ModeTaskOriented
		params.startTime = task.firstTimestamp() - 1
		params.endTime = task.lastTimestamp() + 1
		params.maing = goid
		params.tasks = task.descendants()
		gs := map[uint64]bool{}
		for _, t := range params.tasks {
			// find only directly involved goroutines
			for k, v := range t.RelatedGoroutines(res.Events, 0) {
				gs[k] = v
			}
		}
		params.gs = gs
	} else if taskids := r.FormValue("focustask"); taskids != "" {
		taskid, err := strconv.ParseUint(taskids, 10, 64)
		if err != nil {
			log.Printf("failed to parse focustask parameter %q: %v", taskids, err)
			return
		}
		annotRes, _ := analyzeAnnotations()
		task, ok := annotRes.tasks[taskid]
		if !ok || len(task.events) == 0 {
			log.Printf("failed to find task with id %d", taskid)
			return
		}
		params.mode = traceviewer.ModeTaskOriented
		params.startTime = task.firstTimestamp() - 1
		params.endTime = task.lastTimestamp() + 1
		params.tasks = task.descendants()
	}

	start := int64(0)
	end := int64(math.MaxInt64)
	if startStr, endStr := r.FormValue("start"), r.FormValue("end"); startStr != "" && endStr != "" {
		// If start/end arguments are present, we are rendering a range of the trace.
		start, err = strconv.ParseInt(startStr, 10, 64)
		if err != nil {
			log.Printf("failed to parse start parameter %q: %v", startStr, err)
			return
		}
		end, err = strconv.ParseInt(endStr, 10, 64)
		if err != nil {
			log.Printf("failed to parse end parameter %q: %v", endStr, err)
			return
		}
	}

	c := traceviewer.ViewerDataTraceConsumer(w, start, end)
	if err := generateTrace(params, c); err != nil {
		log.Printf("failed to generate trace: %v", err)
		return
	}
}

// splitTrace splits the trace into a number of ranges,
// each resulting in approx 100MB of json output
// (trace viewer can hardly handle more).
func splitTrace(res trace.ParseResult) []traceviewer.Range {
	params := &traceParams{
		parsed:  res,
		endTime: math.MaxInt64,
	}
	s, c := traceviewer.SplittingTraceConsumer(100 << 20) // 100M
	if err := generateTrace(params, c); err != nil {
		dief("%v\n", err)
	}
	return s.Ranges
}

type traceParams struct {
	parsed    trace.ParseResult
	mode      traceviewer.Mode
	startTime int64
	endTime   int64
	maing     uint64          // for goroutine-oriented view, place this goroutine on the top row
	gs        map[uint64]bool // Goroutines to be displayed for goroutine-oriented or task-oriented view
	tasks     []*taskDesc     // Tasks to be displayed. tasks[0] is the top-most task
}

type traceContext struct {
	*traceParams
	consumer traceviewer.TraceConsumer
	emitter  *traceviewer.Emitter
	arrowSeq uint64
	gcount   uint64
	regionID int // last emitted region id. incremented in each emitRegion call.
}

type gInfo struct {
	state      traceviewer.GState // current state
	name       string             // name chosen for this goroutine at first EvGoStart
	isSystemG  bool
	start      *trace.Event // most recent EvGoStart
	markAssist *trace.Event // if non-nil, the mark assist currently running.
}

type NameArg struct {
	Name string `json:"name"`
}

type TaskArg struct {
	ID     uint64 `json:"id"`
	StartG uint64 `json:"start_g,omitempty"`
	EndG   uint64 `json:"end_g,omitempty"`
}

type RegionArg struct {
	TaskID uint64 `json:"taskid,omitempty"`
}

type SortIndexArg struct {
	Index int `json:"sort_index"`
}

// generateTrace generates json trace for trace-viewer:
// https://github.com/google/trace-viewer
// Trace format is described at:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/view
// If mode==goroutineMode, generate trace for goroutine goid, otherwise whole trace.
// startTime, endTime determine part of the trace that we are interested in.
// gset restricts goroutines that are included in the resulting trace.
func generateTrace(params *traceParams, consumer traceviewer.TraceConsumer) error {
	emitter := traceviewer.NewEmitter(
		consumer,
		time.Duration(params.startTime),
		time.Duration(params.endTime),
	)
	if params.mode&traceviewer.ModeGoroutineOriented != 0 {
		emitter.SetResourceType("G")
	} else {
		emitter.SetResourceType("PROCS")
	}
	defer emitter.Flush()

	ctx := &traceContext{traceParams: params, emitter: emitter}
	ctx.consumer = consumer

	maxProc := 0
	ginfos := make(map[uint64]*gInfo)
	stacks := params.parsed.Stacks

	getGInfo := func(g uint64) *gInfo {
		info, ok := ginfos[g]
		if !ok {
			info = &gInfo{}
			ginfos[g] = info
		}
		return info
	}

	// Since we make many calls to setGState, we record a sticky
	// error in setGStateErr and check it after every event.
	var setGStateErr error
	setGState := func(ev *trace.Event, g uint64, oldState, newState traceviewer.GState) {
		info := getGInfo(g)
		if oldState == traceviewer.GWaiting && info.state == traceviewer.GWaitingGC {
			// For checking, traceviewer.GWaiting counts as any traceviewer.GWaiting*.
			oldState = info.state
		}
		if info.state != oldState && setGStateErr == nil {
			setGStateErr = fmt.Errorf("expected G %d to be in state %d, but got state %d", g, oldState, info.state)
		}

		emitter.GoroutineTransition(time.Duration(ev.Ts), info.state, newState)
		info.state = newState
	}

	for _, ev := range ctx.parsed.Events {
		// Handle state transitions before we filter out events.
		switch ev.Type {
		case trace.EvGoStart, trace.EvGoStartLabel:
			setGState(ev, ev.G, traceviewer.GRunnable, traceviewer.GRunning)
			info := getGInfo(ev.G)
			info.start = ev
		case trace.EvProcStart:
			emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateRunning, 1)
		case trace.EvProcStop:
			emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateRunning, -1)
		case trace.EvGoCreate:
			newG := ev.Args[0]
			info := getGInfo(newG)
			if info.name != "" {
				return fmt.Errorf("duplicate go create event for go id=%d detected at offset %d", newG, ev.Off)
			}

			stk, ok := stacks[ev.Args[1]]
			if !ok || len(stk) == 0 {
				return fmt.Errorf("invalid go create event: missing stack information for go id=%d at offset %d", newG, ev.Off)
			}

			fname := stk[0].Fn
			info.name = fmt.Sprintf("G%v %s", newG, fname)
			info.isSystemG = trace.IsSystemGoroutine(fname)

			ctx.gcount++
			setGState(ev, newG, traceviewer.GDead, traceviewer.GRunnable)
		case trace.EvGoEnd:
			ctx.gcount--
			setGState(ev, ev.G, traceviewer.GRunning, traceviewer.GDead)
		case trace.EvGoUnblock:
			setGState(ev, ev.Args[0], traceviewer.GWaiting, traceviewer.GRunnable)
		case trace.EvGoSysExit:
			setGState(ev, ev.G, traceviewer.GWaiting, traceviewer.GRunnable)
			if getGInfo(ev.G).isSystemG {
				emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateInSyscallRuntime, -1)
			} else {
				emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateInSyscall, -1)
			}
		case trace.EvGoSysBlock:
			setGState(ev, ev.G, traceviewer.GRunning, traceviewer.GWaiting)
			if getGInfo(ev.G).isSystemG {
				emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateInSyscallRuntime, 1)
			} else {
				emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateInSyscall, 1)
			}
		case trace.EvGoSched, trace.EvGoPreempt:
			setGState(ev, ev.G, traceviewer.GRunning, traceviewer.GRunnable)
		case trace.EvGoStop,
			trace.EvGoSleep, trace.EvGoBlock, trace.EvGoBlockSend, trace.EvGoBlockRecv,
			trace.EvGoBlockSelect, trace.EvGoBlockSync, trace.EvGoBlockCond, trace.EvGoBlockNet:
			setGState(ev, ev.G, traceviewer.GRunning, traceviewer.GWaiting)
		case trace.EvGoBlockGC:
			setGState(ev, ev.G, traceviewer.GRunning, traceviewer.GWaitingGC)
		case trace.EvGCMarkAssistStart:
			getGInfo(ev.G).markAssist = ev
		case trace.EvGCMarkAssistDone:
			getGInfo(ev.G).markAssist = nil
		case trace.EvGoWaiting:
			setGState(ev, ev.G, traceviewer.GRunnable, traceviewer.GWaiting)
		case trace.EvGoInSyscall:
			// Cancel out the effect of EvGoCreate at the beginning.
			setGState(ev, ev.G, traceviewer.GRunnable, traceviewer.GWaiting)
			if getGInfo(ev.G).isSystemG {
				emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateInSyscallRuntime, 1)
			} else {
				emitter.IncThreadStateCount(time.Duration(ev.Ts), traceviewer.ThreadStateInSyscall, 1)
			}
		case trace.EvHeapAlloc:
			emitter.HeapAlloc(time.Duration(ev.Ts), ev.Args[0])
		case trace.EvHeapGoal:
			emitter.HeapGoal(time.Duration(ev.Ts), ev.Args[0])
		}
		if setGStateErr != nil {
			return setGStateErr
		}

		if err := emitter.Err(); err != nil {
			return fmt.Errorf("invalid state after processing %v: %s", ev, err)
		}

		// Ignore events that are from uninteresting goroutines
		// or outside of the interesting timeframe.
		if ctx.gs != nil && ev.P < trace.FakeP && !ctx.gs[ev.G] {
			continue
		}
		if !withinTimeRange(ev, ctx.startTime, ctx.endTime) {
			continue
		}

		if ev.P < trace.FakeP && ev.P > maxProc {
			maxProc = ev.P
		}

		// Emit trace objects.
		switch ev.Type {
		case trace.EvProcStart:
			if ctx.mode&traceviewer.ModeGoroutineOriented != 0 {
				continue
			}
			ctx.emitInstant(ev, "proc start", "")
		case trace.EvProcStop:
			if ctx.mode&traceviewer.ModeGoroutineOriented != 0 {
				continue
			}
			ctx.emitInstant(ev, "proc stop", "")
		case trace.EvGCStart:
			ctx.emitSlice(ev, "GC")
		case trace.EvGCDone:
		case trace.EvSTWStart:
			if ctx.mode&traceviewer.ModeGoroutineOriented != 0 {
				continue
			}
			ctx.emitSlice(ev, fmt.Sprintf("STW (%s)", ev.SArgs[0]))
		case trace.EvSTWDone:
		case trace.EvGCMarkAssistStart:
			// Mark assists can continue past preemptions, so truncate to the
			// whichever comes first. We'll synthesize another slice if
			// necessary in EvGoStart.
			markFinish := ev.Link
			goFinish := getGInfo(ev.G).start.Link
			fakeMarkStart := *ev
			text := "MARK ASSIST"
			if markFinish == nil || markFinish.Ts > goFinish.Ts {
				fakeMarkStart.Link = goFinish
				text = "MARK ASSIST (unfinished)"
			}
			ctx.emitSlice(&fakeMarkStart, text)
		case trace.EvGCSweepStart:
			slice := ctx.makeSlice(ev, "SWEEP")
			if done := ev.Link; done != nil && done.Args[0] != 0 {
				slice.Arg = struct {
					Swept     uint64 `json:"Swept bytes"`
					Reclaimed uint64 `json:"Reclaimed bytes"`
				}{done.Args[0], done.Args[1]}
			}
			ctx.emit(slice)
		case trace.EvGoStart, trace.EvGoStartLabel:
			info := getGInfo(ev.G)
			if ev.Type == trace.EvGoStartLabel {
				ctx.emitSlice(ev, ev.SArgs[0])
			} else {
				ctx.emitSlice(ev, info.name)
			}
			if info.markAssist != nil {
				// If we're in a mark assist, synthesize a new slice, ending
				// either when the mark assist ends or when we're descheduled.
				markFinish := info.markAssist.Link
				goFinish := ev.Link
				fakeMarkStart := *ev
				text := "MARK ASSIST (resumed, unfinished)"
				if markFinish != nil && markFinish.Ts < goFinish.Ts {
					fakeMarkStart.Link = markFinish
					text = "MARK ASSIST (resumed)"
				}
				ctx.emitSlice(&fakeMarkStart, text)
			}
		case trace.EvGoCreate:
			ctx.emitArrow(ev, "go")
		case trace.EvGoUnblock:
			ctx.emitArrow(ev, "unblock")
		case trace.EvGoSysCall:
			ctx.emitInstant(ev, "syscall", "")
		case trace.EvGoSysExit:
			ctx.emitArrow(ev, "sysexit")
		case trace.EvUserLog:
			ctx.emitInstant(ev, formatUserLog(ev), "user event")
		case trace.EvUserTaskCreate:
			ctx.emitInstant(ev, "task start", "user event")
		case trace.EvUserTaskEnd:
			ctx.emitInstant(ev, "task end", "user event")
		case trace.EvCPUSample:
			if ev.P >= 0 {
				// only show in this UI when there's an associated P
				ctx.emitInstant(ev, "CPU profile sample", "")
			}
		}
	}

	// Display task and its regions if we are in task-oriented presentation mode.
	if ctx.mode&traceviewer.ModeTaskOriented != 0 {
		// sort tasks based on the task start time.
		sortedTask := make([]*taskDesc, len(ctx.tasks))
		copy(sortedTask, ctx.tasks)
		sort.SliceStable(sortedTask, func(i, j int) bool {
			ti, tj := sortedTask[i], sortedTask[j]
			if ti.firstTimestamp() == tj.firstTimestamp() {
				return ti.lastTimestamp() < tj.lastTimestamp()
			}
			return ti.firstTimestamp() < tj.firstTimestamp()
		})

		for i, task := range sortedTask {
			ctx.emitTask(task, i)

			// If we are in goroutine-oriented mode, we draw regions.
			// TODO(hyangah): add this for task/P-oriented mode (i.e., focustask view) too.
			if ctx.mode&traceviewer.ModeGoroutineOriented != 0 {
				for _, s := range task.regions {
					ctx.emitRegion(s)
				}
			}
		}
	}

	// Display goroutine rows if we are either in goroutine-oriented mode.
	if ctx.mode&traceviewer.ModeGoroutineOriented != 0 {
		for k, v := range ginfos {
			if !ctx.gs[k] {
				continue
			}
			emitter.Resource(k, v.name)
		}
		emitter.Focus(ctx.maing)

		// Row for GC or global state (specified with G=0)
		ctx.emitFooter(&format.Event{Name: "thread_sort_index", Phase: "M", PID: format.ProcsSection, TID: 0, Arg: &SortIndexArg{-1}})
	} else {
		// Display rows for Ps if we are in the default trace view mode.
		for i := 0; i <= maxProc; i++ {
			emitter.Resource(uint64(i), fmt.Sprintf("Proc %v", i))
		}
	}

	return nil
}

func (ctx *traceContext) emit(e *format.Event) {
	ctx.consumer.ConsumeViewerEvent(e, false)
}

func (ctx *traceContext) emitFooter(e *format.Event) {
	ctx.consumer.ConsumeViewerEvent(e, true)
}
func (ctx *traceContext) time(ev *trace.Event) float64 {
	// Trace viewer wants timestamps in microseconds.
	return float64(ev.Ts) / 1000
}

func withinTimeRange(ev *trace.Event, s, e int64) bool {
	if evEnd := ev.Link; evEnd != nil {
		return ev.Ts <= e && evEnd.Ts >= s
	}
	return ev.Ts >= s && ev.Ts <= e
}

func tsWithinRange(ts, s, e int64) bool {
	return s <= ts && ts <= e
}

func (ctx *traceContext) proc(ev *trace.Event) uint64 {
	if ctx.mode&traceviewer.ModeGoroutineOriented != 0 && ev.P < trace.FakeP {
		return ev.G
	} else {
		return uint64(ev.P)
	}
}

func (ctx *traceContext) emitSlice(ev *trace.Event, name string) {
	ctx.emit(ctx.makeSlice(ev, name))
}

func (ctx *traceContext) makeSlice(ev *trace.Event, name string) *format.Event {
	// If ViewerEvent.Dur is not a positive value,
	// trace viewer handles it as a non-terminating time interval.
	// Avoid it by setting the field with a small value.
	durationUsec := ctx.time(ev.Link) - ctx.time(ev)
	if ev.Link.Ts-ev.Ts <= 0 {
		durationUsec = 0.0001 // 0.1 nanoseconds
	}
	sl := &format.Event{
		Name:     name,
		Phase:    "X",
		Time:     ctx.time(ev),
		Dur:      durationUsec,
		TID:      ctx.proc(ev),
		Stack:    ctx.emitter.Stack(ev.Stk),
		EndStack: ctx.emitter.Stack(ev.Link.Stk),
	}

	// grey out non-overlapping events if the event is not a global event (ev.G == 0)
	if ctx.mode&traceviewer.ModeTaskOriented != 0 && ev.G != 0 {
		// include P information.
		if t := ev.Type; t == trace.EvGoStart || t == trace.EvGoStartLabel {
			type Arg struct {
				P int
			}
			sl.Arg = &Arg{P: ev.P}
		}
		// grey out non-overlapping events.
		overlapping := false
		for _, task := range ctx.tasks {
			if _, overlapped := task.overlappingDuration(ev); overlapped {
				overlapping = true
				break
			}
		}
		if !overlapping {
			sl.Cname = colorLightGrey
		}
	}
	return sl
}

func (ctx *traceContext) emitTask(task *taskDesc, sortIndex int) {
	taskRow := uint64(task.id)
	taskName := task.name
	durationUsec := float64(task.lastTimestamp()-task.firstTimestamp()) / 1e3

	ctx.emitter.Task(taskRow, taskName, sortIndex)
	ts := float64(task.firstTimestamp()) / 1e3
	sl := &format.Event{
		Name:  taskName,
		Phase: "X",
		Time:  ts,
		Dur:   durationUsec,
		PID:   format.TasksSection,
		TID:   taskRow,
		Cname: pickTaskColor(task.id),
	}
	targ := TaskArg{ID: task.id}
	if task.create != nil {
		sl.Stack = ctx.emitter.Stack(task.create.Stk)
		targ.StartG = task.create.G
	}
	if task.end != nil {
		sl.EndStack = ctx.emitter.Stack(task.end.Stk)
		targ.EndG = task.end.G
	}
	sl.Arg = targ
	ctx.emit(sl)

	if task.create != nil && task.create.Type == trace.EvUserTaskCreate && task.create.Args[1] != 0 {
		ctx.arrowSeq++
		ctx.emit(&format.Event{Name: "newTask", Phase: "s", TID: task.create.Args[1], ID: ctx.arrowSeq, Time: ts, PID: format.TasksSection})
		ctx.emit(&format.Event{Name: "newTask", Phase: "t", TID: taskRow, ID: ctx.arrowSeq, Time: ts, PID: format.TasksSection})
	}
}

func (ctx *traceContext) emitRegion(s regionDesc) {
	if s.Name == "" {
		return
	}

	if !tsWithinRange(s.firstTimestamp(), ctx.startTime, ctx.endTime) &&
		!tsWithinRange(s.lastTimestamp(), ctx.startTime, ctx.endTime) {
		return
	}

	ctx.regionID++
	regionID := ctx.regionID

	id := s.TaskID
	scopeID := fmt.Sprintf("%x", id)
	name := s.Name

	sl0 := &format.Event{
		Category: "Region",
		Name:     name,
		Phase:    "b",
		Time:     float64(s.firstTimestamp()) / 1e3,
		TID:      s.G, // only in goroutine-oriented view
		ID:       uint64(regionID),
		Scope:    scopeID,
		Cname:    pickTaskColor(s.TaskID),
	}
	if s.Start != nil {
		sl0.Stack = ctx.emitter.Stack(s.Start.Stk)
	}
	ctx.emit(sl0)

	sl1 := &format.Event{
		Category: "Region",
		Name:     name,
		Phase:    "e",
		Time:     float64(s.lastTimestamp()) / 1e3,
		TID:      s.G,
		ID:       uint64(regionID),
		Scope:    scopeID,
		Cname:    pickTaskColor(s.TaskID),
		Arg:      RegionArg{TaskID: s.TaskID},
	}
	if s.End != nil {
		sl1.Stack = ctx.emitter.Stack(s.End.Stk)
	}
	ctx.emit(sl1)
}

func (ctx *traceContext) emitInstant(ev *trace.Event, name, category string) {
	if !tsWithinRange(ev.Ts, ctx.startTime, ctx.endTime) {
		return
	}

	cname := ""
	if ctx.mode&traceviewer.ModeTaskOriented != 0 {
		taskID, isUserAnnotation := isUserAnnotationEvent(ev)

		show := false
		for _, task := range ctx.tasks {
			if isUserAnnotation && task.id == taskID || task.overlappingInstant(ev) {
				show = true
				break
			}
		}
		// grey out or skip if non-overlapping instant.
		if !show {
			if isUserAnnotation {
				return // don't display unrelated user annotation events.
			}
			cname = colorLightGrey
		}
	}
	var arg any
	if ev.Type == trace.EvProcStart {
		type Arg struct {
			ThreadID uint64
		}
		arg = &Arg{ev.Args[0]}
	}
	ctx.emit(&format.Event{
		Name:     name,
		Category: category,
		Phase:    "I",
		Scope:    "t",
		Time:     ctx.time(ev),
		TID:      ctx.proc(ev),
		Stack:    ctx.emitter.Stack(ev.Stk),
		Cname:    cname,
		Arg:      arg})
}

func (ctx *traceContext) emitArrow(ev *trace.Event, name string) {
	if ev.Link == nil {
		// The other end of the arrow is not captured in the trace.
		// For example, a goroutine was unblocked but was not scheduled before trace stop.
		return
	}
	if ctx.mode&traceviewer.ModeGoroutineOriented != 0 && (!ctx.gs[ev.Link.G] || ev.Link.Ts < ctx.startTime || ev.Link.Ts > ctx.endTime) {
		return
	}

	if ev.P == trace.NetpollP || ev.P == trace.TimerP || ev.P == trace.SyscallP {
		// Trace-viewer discards arrows if they don't start/end inside of a slice or instant.
		// So emit a fake instant at the start of the arrow.
		ctx.emitInstant(&trace.Event{P: ev.P, Ts: ev.Ts}, "unblock", "")
	}

	color := ""
	if ctx.mode&traceviewer.ModeTaskOriented != 0 {
		overlapping := false
		// skip non-overlapping arrows.
		for _, task := range ctx.tasks {
			if _, overlapped := task.overlappingDuration(ev); overlapped {
				overlapping = true
				break
			}
		}
		if !overlapping {
			return
		}
	}

	ctx.arrowSeq++
	ctx.emit(&format.Event{Name: name, Phase: "s", TID: ctx.proc(ev), ID: ctx.arrowSeq, Time: ctx.time(ev), Stack: ctx.emitter.Stack(ev.Stk), Cname: color})
	ctx.emit(&format.Event{Name: name, Phase: "t", TID: ctx.proc(ev.Link), ID: ctx.arrowSeq, Time: ctx.time(ev.Link), Cname: color})
}

// firstTimestamp returns the timestamp of the first event record.
func firstTimestamp() int64 {
	res, _ := parseTrace()
	if len(res.Events) > 0 {
		return res.Events[0].Ts
	}
	return 0
}

// lastTimestamp returns the timestamp of the last event record.
func lastTimestamp() int64 {
	res, _ := parseTrace()
	if n := len(res.Events); n > 1 {
		return res.Events[n-1].Ts
	}
	return 0
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
