// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"internal/trace"
	"log"
	"net/http"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

func init() {
	http.HandleFunc("/trace", httpTrace)
	http.HandleFunc("/jsontrace", httpJsonTrace)
	http.HandleFunc("/trace_viewer_html", httpTraceViewerHTML)
}

// httpTrace serves either whole trace (goid==0) or trace for goid goroutine.
func httpTrace(w http.ResponseWriter, r *http.Request) {
	_, err := parseEvents()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	html := strings.Replace(templTrace, "{{PARAMS}}", r.Form.Encode(), -1)
	w.Write([]byte(html))

}

// See https://github.com/catapult-project/catapult/blob/master/tracing/docs/embedding-trace-viewer.md
// This is almost verbatim copy of:
// https://github.com/catapult-project/catapult/blob/master/tracing/bin/index.html
// on revision 623a005a3ffa9de13c4b92bc72290e7bcd1ca591.
var templTrace = `
<html>
<head>
<link href="/trace_viewer_html" rel="import">
<script>
(function() {
  var viewer;
  var url;
  var model;

  function load() {
    var req = new XMLHttpRequest();
    var is_binary = /[.]gz$/.test(url) || /[.]zip$/.test(url);
    req.overrideMimeType('text/plain; charset=x-user-defined');
    req.open('GET', url, true);
    if (is_binary)
      req.responseType = 'arraybuffer';

    req.onreadystatechange = function(event) {
      if (req.readyState !== 4)
        return;

      window.setTimeout(function() {
        if (req.status === 200)
          onResult(is_binary ? req.response : req.responseText);
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
    var i = new tr.importer.Import(model);
    var p = i.importTracesWithProgressDialog([result]);
    p.then(onModelLoaded, onImportFail);
  }

  function onModelLoaded() {
    viewer.model = model;
    viewer.viewTitle = "trace";
  }

  function onImportFail() {
    var overlay = new tr.ui.b.Overlay();
    overlay.textContent = tr.b.normalizeException(err).message;
    overlay.title = 'Import error';
    overlay.visible = true;
  }

  document.addEventListener('DOMContentLoaded', function() {
    var container = document.createElement('track-view-container');
    container.id = 'track_view_container';

    viewer = document.createElement('tr-ui-timeline-view');
    viewer.track_view_container = container;
    viewer.appendChild(container);

    viewer.id = 'trace-viewer';
    viewer.globalMode = true;
    document.body.appendChild(viewer);

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
	http.ServeFile(w, r, filepath.Join(runtime.GOROOT(), "misc", "trace", "trace_viewer_lean.html"))
}

// httpJsonTrace serves json trace, requested from within templTrace HTML.
func httpJsonTrace(w http.ResponseWriter, r *http.Request) {
	// This is an AJAX handler, so instead of http.Error we use log.Printf to log errors.
	events, err := parseEvents()
	if err != nil {
		log.Printf("failed to parse trace: %v", err)
		return
	}

	params := &traceParams{
		events:  events,
		endTime: int64(1<<63 - 1),
	}

	if goids := r.FormValue("goid"); goids != "" {
		// If goid argument is present, we are rendering a trace for this particular goroutine.
		goid, err := strconv.ParseUint(goids, 10, 64)
		if err != nil {
			log.Printf("failed to parse goid parameter '%v': %v", goids, err)
			return
		}
		analyzeGoroutines(events)
		g := gs[goid]
		params.gtrace = true
		params.startTime = g.StartTime
		params.endTime = g.EndTime
		params.maing = goid
		params.gs = trace.RelatedGoroutines(events, goid)
	}

	data := generateTrace(params)

	if startStr, endStr := r.FormValue("start"), r.FormValue("end"); startStr != "" && endStr != "" {
		// If start/end arguments are present, we are rendering a range of the trace.
		start, err := strconv.ParseUint(startStr, 10, 64)
		if err != nil {
			log.Printf("failed to parse start parameter '%v': %v", startStr, err)
			return
		}
		end, err := strconv.ParseUint(endStr, 10, 64)
		if err != nil {
			log.Printf("failed to parse end parameter '%v': %v", endStr, err)
			return
		}
		if start >= uint64(len(data.Events)) || end <= start || end > uint64(len(data.Events)) {
			log.Printf("bogus start/end parameters: %v/%v, trace size %v", start, end, len(data.Events))
			return
		}
		data.Events = append(data.Events[start:end], data.Events[data.footer:]...)
	}
	err = json.NewEncoder(w).Encode(data)
	if err != nil {
		log.Printf("failed to serialize trace: %v", err)
		return
	}
}

type Range struct {
	Name  string
	Start int
	End   int
}

// splitTrace splits the trace into a number of ranges,
// each resulting in approx 100MB of json output (trace viewer can hardly handle more).
func splitTrace(data ViewerData) []Range {
	const rangeSize = 100 << 20
	var ranges []Range
	cw := new(countingWriter)
	enc := json.NewEncoder(cw)
	// First calculate size of the mandatory part of the trace.
	// This includes stack traces and thread names.
	data1 := data
	data1.Events = data.Events[data.footer:]
	enc.Encode(data1)
	auxSize := cw.size
	cw.size = 0
	// Then calculate size of each individual event and group them into ranges.
	for i, start := 0, 0; i < data.footer; i++ {
		enc.Encode(data.Events[i])
		if cw.size+auxSize > rangeSize || i == data.footer-1 {
			ranges = append(ranges, Range{
				Name:  fmt.Sprintf("%v-%v", time.Duration(data.Events[start].Time*1000), time.Duration(data.Events[i].Time*1000)),
				Start: start,
				End:   i + 1,
			})
			start = i + 1
			cw.size = 0
		}
	}
	if len(ranges) == 1 {
		ranges = nil
	}
	return ranges
}

type countingWriter struct {
	size int
}

func (cw *countingWriter) Write(data []byte) (int, error) {
	cw.size += len(data)
	return len(data), nil
}

type traceParams struct {
	events    []*trace.Event
	gtrace    bool
	startTime int64
	endTime   int64
	maing     uint64
	gs        map[uint64]bool
}

type traceContext struct {
	*traceParams
	data      ViewerData
	frameTree frameNode
	frameSeq  int
	arrowSeq  uint64
	heapAlloc uint64
	nextGC    uint64
	gcount    uint64
	grunnable uint64
	grunning  uint64
	insyscall uint64
	prunning  uint64
}

type frameNode struct {
	id       int
	children map[uint64]frameNode
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
}

type ViewerFrame struct {
	Name   string `json:"name"`
	Parent int    `json:"parent,omitempty"`
}

type NameArg struct {
	Name string `json:"name"`
}

type SortIndexArg struct {
	Index int `json:"sort_index"`
}

// generateTrace generates json trace for trace-viewer:
// https://github.com/google/trace-viewer
// Trace format is described at:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/view
// If gtrace=true, generate trace for goroutine goid, otherwise whole trace.
// startTime, endTime determine part of the trace that we are interested in.
// gset restricts goroutines that are included in the resulting trace.
func generateTrace(params *traceParams) ViewerData {
	ctx := &traceContext{traceParams: params}
	ctx.frameTree.children = make(map[uint64]frameNode)
	ctx.data.Frames = make(map[string]ViewerFrame)
	ctx.data.TimeUnit = "ns"
	maxProc := 0
	gnames := make(map[uint64]string)
	for _, ev := range ctx.events {
		// Handle trace.EvGoStart separately, because we need the goroutine name
		// even if ignore the event otherwise.
		if ev.Type == trace.EvGoStart {
			if _, ok := gnames[ev.G]; !ok {
				if len(ev.Stk) > 0 {
					gnames[ev.G] = fmt.Sprintf("G%v %s", ev.G, ev.Stk[0].Fn)
				} else {
					gnames[ev.G] = fmt.Sprintf("G%v", ev.G)
				}
			}
		}

		// Ignore events that are from uninteresting goroutines
		// or outside of the interesting timeframe.
		if ctx.gs != nil && ev.P < trace.FakeP && !ctx.gs[ev.G] {
			continue
		}
		if ev.Ts < ctx.startTime || ev.Ts > ctx.endTime {
			continue
		}

		if ev.P < trace.FakeP && ev.P > maxProc {
			maxProc = ev.P
		}

		switch ev.Type {
		case trace.EvProcStart:
			if ctx.gtrace {
				continue
			}
			ctx.prunning++
			ctx.emitThreadCounters(ev)
			ctx.emitInstant(ev, "proc start")
		case trace.EvProcStop:
			if ctx.gtrace {
				continue
			}
			ctx.prunning--
			ctx.emitThreadCounters(ev)
			ctx.emitInstant(ev, "proc stop")
		case trace.EvGCStart:
			ctx.emitSlice(ev, "GC")
		case trace.EvGCDone:
		case trace.EvGCScanStart:
			if ctx.gtrace {
				continue
			}
			ctx.emitSlice(ev, "MARK")
		case trace.EvGCScanDone:
		case trace.EvGCSweepStart:
			ctx.emitSlice(ev, "SWEEP")
		case trace.EvGCSweepDone:
		case trace.EvGoStart:
			ctx.grunnable--
			ctx.grunning++
			ctx.emitGoroutineCounters(ev)
			ctx.emitSlice(ev, gnames[ev.G])
		case trace.EvGoCreate:
			ctx.gcount++
			ctx.grunnable++
			ctx.emitGoroutineCounters(ev)
			ctx.emitArrow(ev, "go")
		case trace.EvGoEnd:
			ctx.gcount--
			ctx.grunning--
			ctx.emitGoroutineCounters(ev)
		case trace.EvGoUnblock:
			ctx.grunnable++
			ctx.emitGoroutineCounters(ev)
			ctx.emitArrow(ev, "unblock")
		case trace.EvGoSysCall:
			ctx.emitInstant(ev, "syscall")
		case trace.EvGoSysExit:
			ctx.grunnable++
			ctx.emitGoroutineCounters(ev)
			ctx.insyscall--
			ctx.emitThreadCounters(ev)
			ctx.emitArrow(ev, "sysexit")
		case trace.EvGoSysBlock:
			ctx.grunning--
			ctx.emitGoroutineCounters(ev)
			ctx.insyscall++
			ctx.emitThreadCounters(ev)
		case trace.EvGoSched, trace.EvGoPreempt:
			ctx.grunnable++
			ctx.grunning--
			ctx.emitGoroutineCounters(ev)
		case trace.EvGoStop,
			trace.EvGoSleep, trace.EvGoBlock, trace.EvGoBlockSend, trace.EvGoBlockRecv,
			trace.EvGoBlockSelect, trace.EvGoBlockSync, trace.EvGoBlockCond, trace.EvGoBlockNet:
			ctx.grunning--
			ctx.emitGoroutineCounters(ev)
		case trace.EvGoWaiting:
			ctx.grunnable--
			ctx.emitGoroutineCounters(ev)
		case trace.EvGoInSyscall:
			ctx.insyscall++
			ctx.emitThreadCounters(ev)
		case trace.EvHeapAlloc:
			ctx.heapAlloc = ev.Args[0]
			ctx.emitHeapCounters(ev)
		case trace.EvNextGC:
			ctx.nextGC = ev.Args[0]
			ctx.emitHeapCounters(ev)
		}
	}

	ctx.data.footer = len(ctx.data.Events)
	ctx.emit(&ViewerEvent{Name: "process_name", Phase: "M", Pid: 0, Arg: &NameArg{"PROCS"}})
	ctx.emit(&ViewerEvent{Name: "process_sort_index", Phase: "M", Pid: 0, Arg: &SortIndexArg{1}})

	ctx.emit(&ViewerEvent{Name: "process_name", Phase: "M", Pid: 1, Arg: &NameArg{"STATS"}})
	ctx.emit(&ViewerEvent{Name: "process_sort_index", Phase: "M", Pid: 1, Arg: &SortIndexArg{0}})

	ctx.emit(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: 0, Tid: trace.NetpollP, Arg: &NameArg{"Network"}})
	ctx.emit(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: 0, Tid: trace.NetpollP, Arg: &SortIndexArg{-5}})

	ctx.emit(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: 0, Tid: trace.TimerP, Arg: &NameArg{"Timers"}})
	ctx.emit(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: 0, Tid: trace.TimerP, Arg: &SortIndexArg{-4}})

	ctx.emit(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: 0, Tid: trace.SyscallP, Arg: &NameArg{"Syscalls"}})
	ctx.emit(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: 0, Tid: trace.SyscallP, Arg: &SortIndexArg{-3}})

	if !ctx.gtrace {
		for i := 0; i <= maxProc; i++ {
			ctx.emit(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: 0, Tid: uint64(i), Arg: &NameArg{fmt.Sprintf("Proc %v", i)}})
			ctx.emit(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: 0, Tid: uint64(i), Arg: &SortIndexArg{i}})
		}
	}

	if ctx.gtrace && ctx.gs != nil {
		for k, v := range gnames {
			if !ctx.gs[k] {
				continue
			}
			ctx.emit(&ViewerEvent{Name: "thread_name", Phase: "M", Pid: 0, Tid: k, Arg: &NameArg{v}})
		}
		ctx.emit(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: 0, Tid: ctx.maing, Arg: &SortIndexArg{-2}})
		ctx.emit(&ViewerEvent{Name: "thread_sort_index", Phase: "M", Pid: 0, Tid: 0, Arg: &SortIndexArg{-1}})
	}

	return ctx.data
}

func (ctx *traceContext) emit(e *ViewerEvent) {
	ctx.data.Events = append(ctx.data.Events, e)
}

func (ctx *traceContext) time(ev *trace.Event) float64 {
	// Trace viewer wants timestamps in microseconds.
	return float64(ev.Ts-ctx.startTime) / 1000
}

func (ctx *traceContext) proc(ev *trace.Event) uint64 {
	if ctx.gtrace && ev.P < trace.FakeP {
		return ev.G
	} else {
		return uint64(ev.P)
	}
}

func (ctx *traceContext) emitSlice(ev *trace.Event, name string) {
	ctx.emit(&ViewerEvent{
		Name:     name,
		Phase:    "X",
		Time:     ctx.time(ev),
		Dur:      ctx.time(ev.Link) - ctx.time(ev),
		Tid:      ctx.proc(ev),
		Stack:    ctx.stack(ev.Stk),
		EndStack: ctx.stack(ev.Link.Stk),
	})
}

func (ctx *traceContext) emitHeapCounters(ev *trace.Event) {
	type Arg struct {
		Allocated uint64
		NextGC    uint64
	}
	if ctx.gtrace {
		return
	}
	diff := uint64(0)
	if ctx.nextGC > ctx.heapAlloc {
		diff = ctx.nextGC - ctx.heapAlloc
	}
	ctx.emit(&ViewerEvent{Name: "Heap", Phase: "C", Time: ctx.time(ev), Pid: 1, Arg: &Arg{ctx.heapAlloc, diff}})
}

func (ctx *traceContext) emitGoroutineCounters(ev *trace.Event) {
	type Arg struct {
		Running  uint64
		Runnable uint64
	}
	if ctx.gtrace {
		return
	}
	ctx.emit(&ViewerEvent{Name: "Goroutines", Phase: "C", Time: ctx.time(ev), Pid: 1, Arg: &Arg{ctx.grunning, ctx.grunnable}})
}

func (ctx *traceContext) emitThreadCounters(ev *trace.Event) {
	type Arg struct {
		Running   uint64
		InSyscall uint64
	}
	if ctx.gtrace {
		return
	}
	ctx.emit(&ViewerEvent{Name: "Threads", Phase: "C", Time: ctx.time(ev), Pid: 1, Arg: &Arg{ctx.prunning, ctx.insyscall}})
}

func (ctx *traceContext) emitInstant(ev *trace.Event, name string) {
	var arg interface{}
	if ev.Type == trace.EvProcStart {
		type Arg struct {
			ThreadID uint64
		}
		arg = &Arg{ev.Args[0]}
	}
	ctx.emit(&ViewerEvent{Name: name, Phase: "I", Scope: "t", Time: ctx.time(ev), Tid: ctx.proc(ev), Stack: ctx.stack(ev.Stk), Arg: arg})
}

func (ctx *traceContext) emitArrow(ev *trace.Event, name string) {
	if ev.Link == nil {
		// The other end of the arrow is not captured in the trace.
		// For example, a goroutine was unblocked but was not scheduled before trace stop.
		return
	}
	if ctx.gtrace && (!ctx.gs[ev.Link.G] || ev.Link.Ts < ctx.startTime || ev.Link.Ts > ctx.endTime) {
		return
	}

	if ev.P == trace.NetpollP || ev.P == trace.TimerP || ev.P == trace.SyscallP {
		// Trace-viewer discards arrows if they don't start/end inside of a slice or instant.
		// So emit a fake instant at the start of the arrow.
		ctx.emitInstant(&trace.Event{P: ev.P, Ts: ev.Ts}, "unblock")
	}

	ctx.arrowSeq++
	ctx.emit(&ViewerEvent{Name: name, Phase: "s", Tid: ctx.proc(ev), ID: ctx.arrowSeq, Time: ctx.time(ev), Stack: ctx.stack(ev.Stk)})
	ctx.emit(&ViewerEvent{Name: name, Phase: "t", Tid: ctx.proc(ev.Link), ID: ctx.arrowSeq, Time: ctx.time(ev.Link)})
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
		ctx.data.Frames[strconv.Itoa(node.id)] = ViewerFrame{fmt.Sprintf("%v:%v", frame.Fn, frame.Line), parent.id}
	}
	return ctx.buildBranch(node, stk)
}
