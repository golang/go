// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"internal/trace"
	"io"
	"log"
	"math"
	"net/http"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"
)

func init() {
	http.HandleFunc("/trace", httpTrace)
	http.HandleFunc("/jsontrace", httpJsonTrace)
	http.HandleFunc("/trace_viewer_html", httpTraceViewerHTML)
	http.HandleFunc("/webcomponents.min.js", webcomponentsJS)
}

// httpTrace serves either whole trace (goid==0) or trace for goid goroutine.
func httpTrace(w http.ResponseWriter, r *http.Request) {
	_, err := parseTrace()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	html := strings.ReplaceAll(templTrace, "{{PARAMS}}", r.Form.Encode())
	w.Write([]byte(html))

}

// https://chromium.googlesource.com/catapult/+/9508452e18f130c98499cb4c4f1e1efaedee8962/tracing/docs/embedding-trace-viewer.md
// This is almost verbatim copy of https://chromium-review.googlesource.com/c/catapult/+/2062938/2/tracing/bin/index.html
var templTrace = `
<html>
<head>
<script src="/webcomponents.min.js"></script>
<script>
'use strict';

function onTraceViewerImportFail() {
  document.addEventListener('DOMContentLoaded', function() {
    document.body.textContent =
    '/trace_viewer_full.html is missing. File a bug in https://golang.org/issue';
  });
}
</script>

<link rel="import" href="/trace_viewer_html"
      onerror="onTraceViewerImportFail(event)">

<style type="text/css">
  html, body {
    box-sizing: border-box;
    overflow: hidden;
    margin: 0px;
    padding: 0;
    width: 100%;
    height: 100%;
  }
  #trace-viewer {
    width: 100%;
    height: 100%;
  }
  #trace-viewer:focus {
    outline: none;
  }
</style>
<script>
'use strict';
(function() {
  var viewer;
  var url;
  var model;

  function load() {
    var req = new XMLHttpRequest();
    var isBinary = /[.]gz$/.test(url) || /[.]zip$/.test(url);
    req.overrideMimeType('text/plain; charset=x-user-defined');
    req.open('GET', url, true);
    if (isBinary)
      req.responseType = 'arraybuffer';

    req.onreadystatechange = function(event) {
      if (req.readyState !== 4)
        return;

      window.setTimeout(function() {
        if (req.status === 200)
          onResult(isBinary ? req.response : req.responseText);
        else
          onResultFail(req.status);
      }, 0);
    };
    req.send(null);
  }

  function onResultFail(err) {
    var overlay = new tr.ui.b.Overlay();
    overlay.textContent = err + ': ' + url + ' could not be loaded';
    overlay.title = 'Failed to fetch data';
    overlay.visible = true;
  }

  function onResult(result) {
    model = new tr.Model();
    var opts = new tr.importer.ImportOptions();
    opts.shiftWorldToZero = false;
    var i = new tr.importer.Import(model, opts);
    var p = i.importTracesWithProgressDialog([result]);
    p.then(onModelLoaded, onImportFail);
  }

  function onModelLoaded() {
    viewer.model = model;
    viewer.viewTitle = "trace";

    if (!model || model.bounds.isEmpty)
      return;
    var sel = window.location.hash.substr(1);
    if (sel === '')
      return;
    var parts = sel.split(':');
    var range = new (tr.b.Range || tr.b.math.Range)();
    range.addValue(parseFloat(parts[0]));
    range.addValue(parseFloat(parts[1]));
    viewer.trackView.viewport.interestRange.set(range);
  }

  function onImportFail(err) {
    var overlay = new tr.ui.b.Overlay();
    overlay.textContent = tr.b.normalizeException(err).message;
    overlay.title = 'Import error';
    overlay.visible = true;
  }

  document.addEventListener('WebComponentsReady', function() {
    var container = document.createElement('track-view-container');
    container.id = 'track_view_container';

    viewer = document.createElement('tr-ui-timeline-view');
    viewer.track_view_container = container;
    Polymer.dom(viewer).appendChild(container);

    viewer.id = 'trace-viewer';
    viewer.globalMode = true;
    Polymer.dom(document.body).appendChild(viewer);

    url = '/jsontrace?{{PARAMS}}';
    load();
  });
}());
</script>
</head>
<body>
</body>
</html>
`

// httpTraceViewerHTML serves static part of trace-viewer.
// This URL is queried from templTrace HTML.
func httpTraceViewerHTML(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, filepath.Join(runtime.GOROOT(), "misc", "trace", "trace_viewer_full.html"))
}

func webcomponentsJS(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, filepath.Join(runtime.GOROOT(), "misc", "trace", "webcomponents.min.js"))
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
		params.mode = modeGoroutineOriented
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
		params.mode = modeGoroutineOriented | modeTaskOriented
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
		params.mode = modeTaskOriented
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

	c := viewerDataTraceConsumer(w, start, end)
	if err := generateTrace(params, c); err != nil {
		log.Printf("failed to generate trace: %v", err)
		return
	}
}

type Range struct {
	Name      string
	Start     int
	End       int
	StartTime int64
	EndTime   int64
}

func (r Range) URL() string {
	return fmt.Sprintf("/trace?start=%d&end=%d", r.Start, r.End)
}

// splitTrace splits the trace into a number of ranges,
// each resulting in approx 100MB of json output
// (trace viewer can hardly handle more).
func splitTrace(res trace.ParseResult) []Range {
	params := &traceParams{
		parsed:  res,
		endTime: math.MaxInt64,
	}
	s, c := splittingTraceConsumer(100 << 20) // 100M
	if err := generateTrace(params, c); err != nil {
		dief("%v\n", err)
	}
	return s.Ranges
}

type splitter struct {
	Ranges []Range
}

func splittingTraceConsumer(max int) (*splitter, traceConsumer) {
	type eventSz struct {
		Time float64
		Sz   int
	}

	var (
		data = ViewerData{Frames: make(map[string]ViewerFrame)}

		sizes []eventSz
		cw    countingWriter
	)

	s := new(splitter)

	return s, traceConsumer{
		consumeTimeUnit: func(unit string) {
			data.TimeUnit = unit
		},
		consumeViewerEvent: func(v *ViewerEvent, required bool) {
			if required {
				// Store required events inside data
				// so flush can include them in the required
				// part of the trace.
				data.Events = append(data.Events, v)
				return
			}
			enc := json.NewEncoder(&cw)
			enc.Encode(v)
			sizes = append(sizes, eventSz{v.Time, cw.size + 1}) // +1 for ",".
			cw.size = 0
		},
		consumeViewerFrame: func(k string, v ViewerFrame) {
			data.Frames[k] = v
		},
		flush: func() {
			// Calculate size of the mandatory part of the trace.
			// This includes stack traces and thread names.
			cw.size = 0
			enc := json.NewEncoder(&cw)
			enc.Encode(data)
			minSize := cw.size

			// Then calculate size of each individual event
			// and group them into ranges.
			sum := minSize
			start := 0
			for i, ev := range sizes {
				if sum+ev.Sz > max {
					startTime := time.Duration(sizes[start].Time * 1000)
					endTime := time.Duration(ev.Time * 1000)
					ranges = append(ranges, Range{
						Name:      fmt.Sprintf("%v-%v", startTime, endTime),
						Start:     start,
						End:       i + 1,
						StartTime: int64(startTime),
						EndTime:   int64(endTime),
					})
					start = i + 1
					sum = minSize
				} else {
					sum += ev.Sz + 1
				}
			}
			if len(ranges) <= 1 {
				s.Ranges = nil
				return
			}

			if end := len(sizes) - 1; start < end {
				ranges = append(ranges, Range{
					Name:      fmt.Sprintf("%v-%v", time.Duration(sizes[start].Time*1000), time.Duration(sizes[end].Time*1000)),
					Start:     start,
					End:       end,
					StartTime: int64(sizes[start].Time * 1000),
					EndTime:   int64(sizes[end].Time * 1000),
				})
			}
			s.Ranges = ranges
		},
	}
}

type countingWriter struct {
	size int
}

func (cw *countingWriter) Write(data []byte) (int, error) {
	cw.size += len(data)
	return len(data), nil
}

type traceParams struct {
	parsed    trace.ParseResult
	mode      traceviewMode
	startTime int64
	endTime   int64
	maing     uint64          // for goroutine-oriented view, place this goroutine on the top row
	gs        map[uint64]bool // Goroutines to be displayed for goroutine-oriented or task-oriented view
	tasks     []*taskDesc     // Tasks to be displayed. tasks[0] is the top-most task
}

type traceviewMode uint

const (
	modeGoroutineOriented traceviewMode = 1 << iota
	modeTaskOriented
)

type traceContext struct {
	*traceParams
	consumer  traceConsumer
	frameTree frameNode
	frameSeq  int
	arrowSeq  uint64
	gcount    uint64

	heapStats, prevHeapStats     heapStats
	threadStats, prevThreadStats threadStats
	gstates, prevGstates         [gStateCount]int64

	regionID int // last emitted region id. incremented in each emitRegion call.
}

type heapStats struct {
	heapAlloc uint64
	nextGC    uint64
}

type threadStats struct {
	insyscallRuntime int64 // system goroutine in syscall
	insyscall        int64 // user goroutine in syscall
	prunning         int64 // thread running P
}

type frameNode struct {
	id       int
	children map[uint64]frameNode
}

type gState int

const (
	gDead gState = iota
	gRunnable
	gRunning
	gWaiting
	gWaitingGC

	gStateCount
)

type gInfo struct {
	state      gState // current state
	name       string // name chosen for this goroutine at first EvGoStart
	isSystemG  bool
	start      *trace.Event // most recent EvGoStart
	markAssist *trace.Event // if non-nil, the mark assist currently running.
}

type ViewerData struct {
	Events   []*ViewerEvent         `json:"traceEvents"`
	Frames   map[string]ViewerFrame `json:"stackFrames"`
	TimeUnit string                 `json:"displayTimeUnit"`

	// This is where mandatory part of the trace starts (e.g. thread names)
	footer int
}

type ViewerEvent struct {
	Name     string      `json:"name,omitempty"`
	Phase    string      `json:"ph"`
	Scope    string      `json:"s,omitempty"`
	Time     float64     `json:"ts"`
	Dur      float64     `json:"dur,omitempty"`
	Pid      uint64      `json:"pid"`
	Tid      uint64      `json:"tid"`
	ID       uint64      `json:"id,omitempty"`
	Stack    int         `json:"sf,omitempty"`
	EndStack int         `json:"esf,omitempty"`
	Arg      interface{} `json:"args,omitempty"`
	Cname    string      `json:"cname,omitempty"`
	Category string      `json:"cat,omitempty"`
}

type ViewerFrame struct {
	Name   string `json:"name"`
	Parent int    `json:"parent,omitempty"`
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

type traceConsumer struct {
	consumeTimeUnit    func(unit string)
	consumeViewerEvent func(v *ViewerEvent, required bool)
	consumeViewerFrame func(key string, f ViewerFrame)
	flush              func()
}

const (
	procsSection = 0 // where Goroutines or per-P timelines are presented.
	statsSection = 1 // where counters are presented.
	tasksSection = 2 // where Task hierarchy & timeline is presented.
)

// generateTrace generates json trace for trace-viewer:
// https://github.com/google/trace-viewer
// Trace format is described at:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/view
// If mode==goroutineMode, generate trace for goroutine goid, otherwise whole trace.
// startTime, endTime determine part of the trace that we are interested in.
// gset restricts goroutines that are included in the resulting trace.
func generateTrace(params *traceParams, consumer traceConsumer) error {
	defer consumer.flush()

	ctx := &traceContext{traceParams: params}
	ctx.frameTree.children = make(map[uint64]frameNode)
	ctx.consumer = consumer

	ctx.consumer.consumeTimeUnit("ns")
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
	setGState := func(ev *trace.Event, g uint64, oldState, newState gState) {
		info := getGInfo(g)
		if oldState == gWaiting && info.state == gWaitingGC {
			// For checking, gWaiting counts as any gWaiting*.
			oldState = info.state
		}
		if info.state != oldState && setGStateErr == nil {
			setGStateErr = fmt.Errorf("expected G %d to be in state %d, but got state %d", g, oldState, newState)
		}
		ctx.gstates[info.state]--
		ctx.gstates[newState]++
		info.state = newState
	}

	for _, ev := range ctx.parsed.Events {
		// Handle state transitions before we filter out events.
		switch ev.Type {
		case trace.EvGoStart, trace.EvGoStartLabel:
			setGState(ev, ev.G, gRunnable, gRunning)
			info := getGInfo(ev.G)
			info.start = ev
		case trace.EvProcStart:
			ctx.threadStats.prunning++
		case trace.EvProcStop:
			ctx.threadStats.prunning--
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
			info.isSystemG = isSystemGoroutine(fname)

			ctx.gcount++
			setGState(ev, newG, gDead, gRunnable)
		case trace.EvGoEnd:
			ctx.gcount--
			setGState(ev, ev.G, gRunning, gDead)
		case trace.EvGoUnblock:
			setGState(ev, ev.Args[0], gWaiting, gRunnable)
		case trace.EvGoSysExit:
			setGState(ev, ev.G, gWaiting, gRunnable)
			if getGInfo(ev.G).isSystemG {
				ctx.threadStats.insyscallRuntime--
			} else {
				ctx.threadStats.insyscall--
			}
		case trace.EvGoSysBlock:
			setGState(ev, ev.G, gRunning, gWaiting)
			if getGInfo(ev.G).isSystemG {
				ctx.threadStats.insyscallRuntime++
			} else {
				ctx.threadStats.insyscall++
			}
		case trace.EvGoSched, trace.EvGoPreempt:
			setGState(ev, ev.G, gRunning, gRunnable)
		case trace.EvGoStop,
			trace.EvGoSleep, trace.EvGoBlock, trace.EvGoBlockSend, trace.EvGoBlockRecv,
			trace.EvGoBlockSelect, trace.EvGoBlockSync, trace.EvGoBlockCond, trace.EvGoBlockNet:
			setGState(ev, ev.G, gRunning, gWaiting)
		case trace.EvGoBlockGC:
			setGState(ev, ev.G, gRunning, gWaitingGC)
		case trace.EvGCMarkAssistStart:
			getGInfo(ev.G).markAssist = ev
		case trace.EvGCMarkAssistDone:
			getGInfo(ev.G).markAssist = nil
		case trace.EvGoWaiting:
			setGState(ev, ev.G, gRunnable, gWaiting)
		case trace.EvGoInSyscall:
			// Cancel out the effect of EvGoCreate at the beginning.
			setGState(ev, ev.G, gRunnable, gWaiting)
			if getGInfo(ev.G).isSystemG {
				ctx.threadStats.insyscallRuntime++
			} else {
				ctx.threadStats.insyscall++
			}
		case trace.EvHeapAlloc:
			ctx.heapStats.heapAlloc = ev.Args[0]
		case trace.EvNextGC:
			ctx.heapStats.nextGC = ev.Args[0]
		}
		if setGStateErr != nil {
			return setGStateErr
		}
		if ctx.gstates[gRunnable] < 0 || ctx.gstates[gRunning] < 0 || ctx.threadStats.insyscall < 0 || ctx.threadStats.insyscallRuntime < 0 {
			return fmt.Errorf("invalid state after processing %v: runnable=%d running=%d insyscall=%d insyscallRuntime=%d", ev, ctx.gstates[gRunnable], ctx.gstates[gRunning], ctx.threadStats.insyscall, ctx.threadStats.insyscallRuntime)
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
			if ctx.mode&modeGoroutineOriented != 0 {
				continue
			}
			ctx.emitInstant(ev, "proc start", "")
		case trace.EvProcStop:
			if ctx.mode&modeGoroutineOriented != 0 {
				continue
			}
			ctx.emitInstant(ev, "proc stop", "")
		case trace.EvGCStart:
			ctx.emitSlice(ev, "GC")
		case trace.EvGCDone:
		case trace.EvGCSTWStart:
			if ctx.mode&modeGoroutineOriented != 0 {
				continue
			}
			ctx.emitSlice(ev, fmt.Sprintf("STW (%s)", ev.SArgs[0]))
		case trace.EvGCSTWDone:
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
		}
		// Emit any counter updates.
		ctx.emitThreadCounters(ev)
		ctx.emitHeapCounters(ev)
		ctx.emitGoroutineCounters(ev)
	}

	ctx.emitSectionFooter(statsSection, "STATS", 0)

	if ctx.mode&modeTaskOriented != 0 {
		ctx.emitSectionFooter(tasksSection, "TASKS", 1)
	}

	if ctx.mode&modeGoroutineOriented != 0 {
		ctx.emitSectionFooter(procsSection, "G", 2)
	} else {
		ctx.emitSectionFooter(procsSection, "PROCS", 2)
	}

	ctx.emitFooter(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: procsSection, Tid: trace.GCP, Arg: &NameArg{"GC"}})
	ctx.emitFooter(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: procsSection, Tid: trace.GCP, Arg: &SortIndexArg{-6}})

	ctx.emitFooter(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: procsSection, Tid: trace.NetpollP, Arg: &NameArg{"Network"}})
	ctx.emitFooter(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: procsSection, Tid: trace.NetpollP, Arg: &SortIndexArg{-5}})

	ctx.emitFooter(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: procsSection, Tid: trace.TimerP, Arg: &NameArg{"Timers"}})
	ctx.emitFooter(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: procsSection, Tid: trace.TimerP, Arg: &SortIndexArg{-4}})

	ctx.emitFooter(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: procsSection, Tid: trace.SyscallP, Arg: &NameArg{"Syscalls"}})
	ctx.emitFooter(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: procsSection, Tid: trace.SyscallP, Arg: &SortIndexArg{-3}})

	// Display rows for Ps if we are in the default trace view mode (not goroutine-oriented presentation)
	if ctx.mode&modeGoroutineOriented == 0 {
		for i := 0; i <= maxProc; i++ {
			ctx.emitFooter(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: procsSection, Tid: uint64(i), Arg: &NameArg{fmt.Sprintf("Proc %v", i)}})
			ctx.emitFooter(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: procsSection, Tid: uint64(i), Arg: &SortIndexArg{i}})
		}
	}

	// Display task and its regions if we are in task-oriented presentation mode.
	if ctx.mode&modeTaskOriented != 0 {
		// sort tasks based on the task start time.
		sortedTask := make([]*taskDesc, 0, len(ctx.tasks))
		for _, task := range ctx.tasks {
			sortedTask = append(sortedTask, task)
		}
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
			if ctx.mode&modeGoroutineOriented != 0 {
				for _, s := range task.regions {
					ctx.emitRegion(s)
				}
			}
		}
	}

	// Display goroutine rows if we are either in goroutine-oriented mode.
	if ctx.mode&modeGoroutineOriented != 0 {
		for k, v := range ginfos {
			if !ctx.gs[k] {
				continue
			}
			ctx.emitFooter(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: procsSection, Tid: k, Arg: &NameArg{v.name}})
		}
		// Row for the main goroutine (maing)
		ctx.emitFooter(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: procsSection, Tid: ctx.maing, Arg: &SortIndexArg{-2}})
		// Row for GC or global state (specified with G=0)
		ctx.emitFooter(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: procsSection, Tid: 0, Arg: &SortIndexArg{-1}})
	}

	return nil
}

func (ctx *traceContext) emit(e *ViewerEvent) {
	ctx.consumer.consumeViewerEvent(e, false)
}

func (ctx *traceContext) emitFooter(e *ViewerEvent) {
	ctx.consumer.consumeViewerEvent(e, true)
}
func (ctx *traceContext) emitSectionFooter(sectionID uint64, name string, priority int) {
	ctx.emitFooter(&ViewerEvent{Name: "process_name", Phase: "M", Pid: sectionID, Arg: &NameArg{name}})
	ctx.emitFooter(&ViewerEvent{Name: "process_sort_index", Phase: "M", Pid: sectionID, Arg: &SortIndexArg{priority}})
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
	if ctx.mode&modeGoroutineOriented != 0 && ev.P < trace.FakeP {
		return ev.G
	} else {
		return uint64(ev.P)
	}
}

func (ctx *traceContext) emitSlice(ev *trace.Event, name string) {
	ctx.emit(ctx.makeSlice(ev, name))
}

func (ctx *traceContext) makeSlice(ev *trace.Event, name string) *ViewerEvent {
	// If ViewerEvent.Dur is not a positive value,
	// trace viewer handles it as a non-terminating time interval.
	// Avoid it by setting the field with a small value.
	durationUsec := ctx.time(ev.Link) - ctx.time(ev)
	if ev.Link.Ts-ev.Ts <= 0 {
		durationUsec = 0.0001 // 0.1 nanoseconds
	}
	sl := &ViewerEvent{
		Name:     name,
		Phase:    "X",
		Time:     ctx.time(ev),
		Dur:      durationUsec,
		Tid:      ctx.proc(ev),
		Stack:    ctx.stack(ev.Stk),
		EndStack: ctx.stack(ev.Link.Stk),
	}

	// grey out non-overlapping events if the event is not a global event (ev.G == 0)
	if ctx.mode&modeTaskOriented != 0 && ev.G != 0 {
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

	ctx.emitFooter(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: tasksSection, Tid: taskRow, Arg: &NameArg{fmt.Sprintf("T%d %s", task.id, taskName)}})
	ctx.emit(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: tasksSection, Tid: taskRow, Arg: &SortIndexArg{sortIndex}})
	ts := float64(task.firstTimestamp()) / 1e3
	sl := &ViewerEvent{
		Name:  taskName,
		Phase: "X",
		Time:  ts,
		Dur:   durationUsec,
		Pid:   tasksSection,
		Tid:   taskRow,
		Cname: pickTaskColor(task.id),
	}
	targ := TaskArg{ID: task.id}
	if task.create != nil {
		sl.Stack = ctx.stack(task.create.Stk)
		targ.StartG = task.create.G
	}
	if task.end != nil {
		sl.EndStack = ctx.stack(task.end.Stk)
		targ.EndG = task.end.G
	}
	sl.Arg = targ
	ctx.emit(sl)

	if task.create != nil && task.create.Type == trace.EvUserTaskCreate && task.create.Args[1] != 0 {
		ctx.arrowSeq++
		ctx.emit(&ViewerEvent{Name: "newTask", Phase: "s", Tid: task.create.Args[1], ID: ctx.arrowSeq, Time: ts, Pid: tasksSection})
		ctx.emit(&ViewerEvent{Name: "newTask", Phase: "t", Tid: taskRow, ID: ctx.arrowSeq, Time: ts, Pid: tasksSection})
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

	sl0 := &ViewerEvent{
		Category: "Region",
		Name:     name,
		Phase:    "b",
		Time:     float64(s.firstTimestamp()) / 1e3,
		Tid:      s.G, // only in goroutine-oriented view
		ID:       uint64(regionID),
		Scope:    scopeID,
		Cname:    pickTaskColor(s.TaskID),
	}
	if s.Start != nil {
		sl0.Stack = ctx.stack(s.Start.Stk)
	}
	ctx.emit(sl0)

	sl1 := &ViewerEvent{
		Category: "Region",
		Name:     name,
		Phase:    "e",
		Time:     float64(s.lastTimestamp()) / 1e3,
		Tid:      s.G,
		ID:       uint64(regionID),
		Scope:    scopeID,
		Cname:    pickTaskColor(s.TaskID),
		Arg:      RegionArg{TaskID: s.TaskID},
	}
	if s.End != nil {
		sl1.Stack = ctx.stack(s.End.Stk)
	}
	ctx.emit(sl1)
}

type heapCountersArg struct {
	Allocated uint64
	NextGC    uint64
}

func (ctx *traceContext) emitHeapCounters(ev *trace.Event) {
	if ctx.prevHeapStats == ctx.heapStats {
		return
	}
	diff := uint64(0)
	if ctx.heapStats.nextGC > ctx.heapStats.heapAlloc {
		diff = ctx.heapStats.nextGC - ctx.heapStats.heapAlloc
	}
	if tsWithinRange(ev.Ts, ctx.startTime, ctx.endTime) {
		ctx.emit(&ViewerEvent{Name: "Heap", Phase: "C", Time: ctx.time(ev), Pid: 1, Arg: &heapCountersArg{ctx.heapStats.heapAlloc, diff}})
	}
	ctx.prevHeapStats = ctx.heapStats
}

type goroutineCountersArg struct {
	Running   uint64
	Runnable  uint64
	GCWaiting uint64
}

func (ctx *traceContext) emitGoroutineCounters(ev *trace.Event) {
	if ctx.prevGstates == ctx.gstates {
		return
	}
	if tsWithinRange(ev.Ts, ctx.startTime, ctx.endTime) {
		ctx.emit(&ViewerEvent{Name: "Goroutines", Phase: "C", Time: ctx.time(ev), Pid: 1, Arg: &goroutineCountersArg{uint64(ctx.gstates[gRunning]), uint64(ctx.gstates[gRunnable]), uint64(ctx.gstates[gWaitingGC])}})
	}
	ctx.prevGstates = ctx.gstates
}

type threadCountersArg struct {
	Running   int64
	InSyscall int64
}

func (ctx *traceContext) emitThreadCounters(ev *trace.Event) {
	if ctx.prevThreadStats == ctx.threadStats {
		return
	}
	if tsWithinRange(ev.Ts, ctx.startTime, ctx.endTime) {
		ctx.emit(&ViewerEvent{Name: "Threads", Phase: "C", Time: ctx.time(ev), Pid: 1, Arg: &threadCountersArg{
			Running:   ctx.threadStats.prunning,
			InSyscall: ctx.threadStats.insyscall}})
	}
	ctx.prevThreadStats = ctx.threadStats
}

func (ctx *traceContext) emitInstant(ev *trace.Event, name, category string) {
	if !tsWithinRange(ev.Ts, ctx.startTime, ctx.endTime) {
		return
	}

	cname := ""
	if ctx.mode&modeTaskOriented != 0 {
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
	var arg interface{}
	if ev.Type == trace.EvProcStart {
		type Arg struct {
			ThreadID uint64
		}
		arg = &Arg{ev.Args[0]}
	}
	ctx.emit(&ViewerEvent{
		Name:     name,
		Category: category,
		Phase:    "I",
		Scope:    "t",
		Time:     ctx.time(ev),
		Tid:      ctx.proc(ev),
		Stack:    ctx.stack(ev.Stk),
		Cname:    cname,
		Arg:      arg})
}

func (ctx *traceContext) emitArrow(ev *trace.Event, name string) {
	if ev.Link == nil {
		// The other end of the arrow is not captured in the trace.
		// For example, a goroutine was unblocked but was not scheduled before trace stop.
		return
	}
	if ctx.mode&modeGoroutineOriented != 0 && (!ctx.gs[ev.Link.G] || ev.Link.Ts < ctx.startTime || ev.Link.Ts > ctx.endTime) {
		return
	}

	if ev.P == trace.NetpollP || ev.P == trace.TimerP || ev.P == trace.SyscallP {
		// Trace-viewer discards arrows if they don't start/end inside of a slice or instant.
		// So emit a fake instant at the start of the arrow.
		ctx.emitInstant(&trace.Event{P: ev.P, Ts: ev.Ts}, "unblock", "")
	}

	color := ""
	if ctx.mode&modeTaskOriented != 0 {
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
	ctx.emit(&ViewerEvent{Name: name, Phase: "s", Tid: ctx.proc(ev), ID: ctx.arrowSeq, Time: ctx.time(ev), Stack: ctx.stack(ev.Stk), Cname: color})
	ctx.emit(&ViewerEvent{Name: name, Phase: "t", Tid: ctx.proc(ev.Link), ID: ctx.arrowSeq, Time: ctx.time(ev.Link), Cname: color})
}

func (ctx *traceContext) stack(stk []*trace.Frame) int {
	return ctx.buildBranch(ctx.frameTree, stk)
}

// buildBranch builds one branch in the prefix tree rooted at ctx.frameTree.
func (ctx *traceContext) buildBranch(parent frameNode, stk []*trace.Frame) int {
	if len(stk) == 0 {
		return parent.id
	}
	last := len(stk) - 1
	frame := stk[last]
	stk = stk[:last]

	node, ok := parent.children[frame.PC]
	if !ok {
		ctx.frameSeq++
		node.id = ctx.frameSeq
		node.children = make(map[uint64]frameNode)
		parent.children[frame.PC] = node
		ctx.consumer.consumeViewerFrame(strconv.Itoa(node.id), ViewerFrame{fmt.Sprintf("%v:%v", frame.Fn, frame.Line), parent.id})
	}
	return ctx.buildBranch(node, stk)
}

func isSystemGoroutine(entryFn string) bool {
	// This mimics runtime.isSystemGoroutine as closely as
	// possible.
	return entryFn != "runtime.main" && strings.HasPrefix(entryFn, "runtime.")
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

type jsonWriter struct {
	w   io.Writer
	enc *json.Encoder
}

func viewerDataTraceConsumer(w io.Writer, start, end int64) traceConsumer {
	frames := make(map[string]ViewerFrame)
	enc := json.NewEncoder(w)
	written := 0
	index := int64(-1)

	io.WriteString(w, "{")
	return traceConsumer{
		consumeTimeUnit: func(unit string) {
			io.WriteString(w, `"displayTimeUnit":`)
			enc.Encode(unit)
			io.WriteString(w, ",")
		},
		consumeViewerEvent: func(v *ViewerEvent, required bool) {
			index++
			if !required && (index < start || index > end) {
				// not in the range. Skip!
				return
			}
			if written == 0 {
				io.WriteString(w, `"traceEvents": [`)
			}
			if written > 0 {
				io.WriteString(w, ",")
			}
			enc.Encode(v)
			// TODO: get rid of the extra \n inserted by enc.Encode.
			// Same should be applied to splittingTraceConsumer.
			written++
		},
		consumeViewerFrame: func(k string, v ViewerFrame) {
			frames[k] = v
		},
		flush: func() {
			io.WriteString(w, `], "stackFrames":`)
			enc.Encode(frames)
			io.WriteString(w, `}`)
		},
	}
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
