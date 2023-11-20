// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceviewer

import (
	"embed"
	"fmt"
	"html/template"
	"net/http"
	"strings"
)

func MainHandler(views []View) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if err := templMain.Execute(w, views); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	})
}

const CommonStyle = `
/* See https://github.com/golang/pkgsite/blob/master/static/shared/typography/typography.css */
body {
  font-family:	-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji';
  font-size:	1rem;
  line-height:	normal;
  max-width:	9in;
  margin:	1em;
}
h1 { font-size: 1.5rem; }
h2 { font-size: 1.375rem; }
h1,h2 {
  font-weight: 600;
  line-height: 1.25em;
  word-break: break-word;
}
p  { color: grey85; font-size:85%; }
code,
pre,
textarea.code {
  font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 0.875rem;
  line-height: 1.5em;
}

pre,
textarea.code {
  background-color: var(--color-background-accented);
  border: var(--border);
  border-radius: var(--border-radius);
  color: var(--color-text);
  overflow-x: auto;
  padding: 0.625rem;
  tab-size: 4;
  white-space: pre;
}
`

var templMain = template.Must(template.New("").Parse(`
<html>
<style>` + CommonStyle + `</style>
<body>
<h1>cmd/trace: the Go trace event viewer</h1>
<p>
  This web server provides various visualizations of an event log gathered during
  the execution of a Go program that uses the <a href='https://pkg.go.dev/runtime/trace'>runtime/trace</a> package.
</p>

<h2>Event timelines for running goroutines</h2>
{{range $i, $view := $}}
{{if $view.Ranges}}
{{if eq $i 0}}
<p>
  Large traces are split into multiple sections of equal data size
  (not duration) to avoid overwhelming the visualizer.
</p>
{{end}}
<ul>
	{{range $index, $e := $view.Ranges}}
		<li><a href="{{$view.URL $index}}">View trace by {{$view.Type}} ({{$e.Name}})</a></li>
	{{end}}
</ul>
{{else}}
<ul>
	<li><a href="{{$view.URL -1}}">View trace by {{$view.Type}}</a></li>
</ul>
{{end}}
{{end}}
<p>
  This view displays a series of timelines for a type of resource.
  The "by proc" view consists of a timeline for each of the GOMAXPROCS
  logical processors, showing which goroutine (if any) was running on that
  logical processor at each moment.
  The "by thread" view (if available) consists of a similar timeline for each
  OS thread.

  Each goroutine has an identifying number (e.g. G123), main function,
  and color.

  A colored bar represents an uninterrupted span of execution.

  Execution of a goroutine may migrate from one logical processor to another,
  causing a single colored bar to be horizontally continuous but
  vertically displaced.
</p>
<p>
  Clicking on a span reveals information about it, such as its
  duration, its causal predecessors and successors, and the stack trace
  at the final moment when it yielded the logical processor, for example
  because it made a system call or tried to acquire a mutex.

  Directly underneath each bar, a smaller bar or more commonly a fine
  vertical line indicates an event occurring during its execution.
  Some of these are related to garbage collection; most indicate that
  a goroutine yielded its logical processor but then immediately resumed execution
  on the same logical processor. Clicking on the event displays the stack trace
  at the moment it occurred.
</p>
<p>
  The causal relationships between spans of goroutine execution
  can be displayed by clicking the Flow Events button at the top.
</p>
<p>
  At the top ("STATS"), there are three additional timelines that
  display statistical information.

  "Goroutines" is a time series of the count of existing goroutines;
  clicking on it displays their breakdown by state at that moment:
  running, runnable, or waiting.

  "Heap" is a time series of the amount of heap memory allocated (in orange)
  and (in green) the allocation limit at which the next GC cycle will begin.

  "Threads" shows the number of kernel threads in existence: there is
  always one kernel thread per logical processor, and additional threads
  are created for calls to non-Go code such as a system call or a
  function written in C.
</p>
<p>
  Above the event trace for the first logical processor are
  traces for various runtime-internal events.

  The "GC" bar shows when the garbage collector is running, and in which stage.
  Garbage collection may temporarily affect all the logical processors
  and the other metrics.

  The "Network", "Timers", and "Syscalls" traces indicate events in
  the runtime that cause goroutines to wake up.
</p>
<p>
  The visualization allows you to navigate events at scales ranging from several
  seconds to a handful of nanoseconds.

  Consult the documentation for the Chromium <a href='https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/'>Trace Event Profiling Tool<a/>
  for help navigating the view.
</p>

<ul>
<li><a href="/goroutines">Goroutine analysis</a></li>
</ul>
<p>
  This view displays information about each set of goroutines that
  shares the same main function.

  Clicking on a main function shows links to the four types of
  blocking profile (see below) applied to that subset of goroutines.

  It also shows a table of specific goroutine instances, with various
  execution statistics and a link to the event timeline for each one.

  The timeline displays only the selected goroutine and any others it
  interacts with via block/unblock events. (The timeline is
  goroutine-oriented rather than logical processor-oriented.)
</p>

<h2>Profiles</h2>
<p>
  Each link below displays a global profile in zoomable graph form as
  produced by <a href='https://go.dev/blog/pprof'>pprof</a>'s "web" command.

  In addition there is a link to download the profile for offline
  analysis with pprof.

  All four profiles represent causes of delay that prevent a goroutine
  from running on a logical processor: because it was waiting for the network,
  for a synchronization operation on a mutex or channel, for a system call,
  or for a logical processor to become available.
</p>
<ul>
<li><a href="/io">Network blocking profile</a> (<a href="/io?raw=1" download="io.profile">⬇</a>)</li>
<li><a href="/block">Synchronization blocking profile</a> (<a href="/block?raw=1" download="block.profile">⬇</a>)</li>
<li><a href="/syscall">Syscall profile</a> (<a href="/syscall?raw=1" download="syscall.profile">⬇</a>)</li>
<li><a href="/sched">Scheduler latency profile</a> (<a href="/sched?raw=1" download="sched.profile">⬇</a>)</li>
</ul>

<h2>User-defined tasks and regions</h2>
<p>
  The trace API allows a target program to annotate a <a
  href='https://pkg.go.dev/runtime/trace#Region'>region</a> of code
  within a goroutine, such as a key function, so that its performance
  can be analyzed.

  <a href='https://pkg.go.dev/runtime/trace#Log'>Log events</a> may be
  associated with a region to record progress and relevant values.

  The API also allows annotation of higher-level
  <a href='https://pkg.go.dev/runtime/trace#Task'>tasks</a>,
  which may involve work across many goroutines.
</p>
<p>
  The links below display, for each region and task, a histogram of its execution times.

  Each histogram bucket contains a sample trace that records the
  sequence of events such as goroutine creations, log events, and
  subregion start/end times.

  For each task, you can click through to a logical-processor or
  goroutine-oriented view showing the tasks and regions on the
  timeline.

  Such information may help uncover which steps in a region are
  unexpectedly slow, or reveal relationships between the data values
  logged in a request and its running time.
</p>
<ul>
<li><a href="/usertasks">User-defined tasks</a></li>
<li><a href="/userregions">User-defined regions</a></li>
</ul>

<h2>Garbage collection metrics</h2>
<ul>
<li><a href="/mmu">Minimum mutator utilization</a></li>
</ul>
<p>
  This chart indicates the maximum GC pause time (the largest x value
  for which y is zero), and more generally, the fraction of time that
  the processors are available to application goroutines ("mutators"),
  for any time window of a specified size, in the worst case.
</p>
</body>
</html>
`))

type View struct {
	Type   ViewType
	Ranges []Range
}

type ViewType string

const (
	ViewProc   ViewType = "proc"
	ViewThread ViewType = "thread"
)

func (v View) URL(rangeIdx int) string {
	if rangeIdx < 0 {
		return fmt.Sprintf("/trace?view=%s", v.Type)
	}
	return v.Ranges[rangeIdx].URL(v.Type)
}

type Range struct {
	Name      string
	Start     int
	End       int
	StartTime int64
	EndTime   int64
}

func (r Range) URL(viewType ViewType) string {
	return fmt.Sprintf("/trace?view=%s&start=%d&end=%d", viewType, r.Start, r.End)
}

func TraceHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		html := strings.ReplaceAll(templTrace, "{{PARAMS}}", r.Form.Encode())
		w.Write([]byte(html))
	})
}

// https://chromium.googlesource.com/catapult/+/9508452e18f130c98499cb4c4f1e1efaedee8962/tracing/docs/embedding-trace-viewer.md
// This is almost verbatim copy of https://chromium-review.googlesource.com/c/catapult/+/2062938/2/tracing/bin/index.html
var templTrace = `
<html>
<head>
<script src="/static/webcomponents.min.js"></script>
<script>
'use strict';

function onTraceViewerImportFail() {
  document.addEventListener('DOMContentLoaded', function() {
    document.body.textContent =
    '/static/trace_viewer_full.html is missing. File a bug in https://golang.org/issue';
  });
}
</script>

<link rel="import" href="/static/trace_viewer_full.html"
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

//go:embed static/trace_viewer_full.html static/webcomponents.min.js
var staticContent embed.FS

func StaticHandler() http.Handler {
	return http.FileServer(http.FS(staticContent))
}
