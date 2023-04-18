// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"cmd/internal/browser"
	"flag"
	"fmt"
	"html/template"
	"internal/trace"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"sync"

	_ "net/http/pprof" // Required to use pprof
)

const usageMessage = "" +
	`Usage of 'go tool trace':
Given a trace file produced by 'go test':
	go test -trace=trace.out pkg

Open a web browser displaying trace:
	go tool trace [flags] [pkg.test] trace.out

Generate a pprof-like profile from the trace:
    go tool trace -pprof=TYPE [pkg.test] trace.out

[pkg.test] argument is required for traces produced by Go 1.6 and below.
Go 1.7 does not require the binary argument.

Supported profile types are:
    - net: network blocking profile
    - sync: synchronization blocking profile
    - syscall: syscall blocking profile
    - sched: scheduler latency profile

Flags:
	-http=addr: HTTP service address (e.g., ':6060')
	-pprof=type: print a pprof-like profile instead
	-d: print debug info such as parsed events

Note that while the various profiles available when launching
'go tool trace' work on every browser, the trace viewer itself
(the 'view trace' page) comes from the Chrome/Chromium project
and is only actively tested on that browser.
`

var (
	httpFlag  = flag.String("http", "localhost:0", "HTTP service address (e.g., ':6060')")
	pprofFlag = flag.String("pprof", "", "print a pprof-like profile instead")
	debugFlag = flag.Bool("d", false, "print debug information such as parsed events list")

	// The binary file name, left here for serveSVGProfile.
	programBinary string
	traceFile     string
)

func main() {
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usageMessage)
		os.Exit(2)
	}
	flag.Parse()

	// Go 1.7 traces embed symbol info and does not require the binary.
	// But we optionally accept binary as first arg for Go 1.5 traces.
	switch flag.NArg() {
	case 1:
		traceFile = flag.Arg(0)
	case 2:
		programBinary = flag.Arg(0)
		traceFile = flag.Arg(1)
	default:
		flag.Usage()
	}

	var pprofFunc func(io.Writer, *http.Request) error
	switch *pprofFlag {
	case "net":
		pprofFunc = pprofByGoroutine(computePprofIO)
	case "sync":
		pprofFunc = pprofByGoroutine(computePprofBlock)
	case "syscall":
		pprofFunc = pprofByGoroutine(computePprofSyscall)
	case "sched":
		pprofFunc = pprofByGoroutine(computePprofSched)
	}
	if pprofFunc != nil {
		if err := pprofFunc(os.Stdout, &http.Request{}); err != nil {
			dief("failed to generate pprof: %v\n", err)
		}
		os.Exit(0)
	}
	if *pprofFlag != "" {
		dief("unknown pprof type %s\n", *pprofFlag)
	}

	ln, err := net.Listen("tcp", *httpFlag)
	if err != nil {
		dief("failed to create server socket: %v\n", err)
	}

	log.Print("Parsing trace...")
	res, err := parseTrace()
	if err != nil {
		dief("%v\n", err)
	}

	if *debugFlag {
		trace.Print(res.Events)
		os.Exit(0)
	}
	reportMemoryUsage("after parsing trace")
	debug.FreeOSMemory()

	log.Print("Splitting trace...")
	ranges = splitTrace(res)
	reportMemoryUsage("after spliting trace")
	debug.FreeOSMemory()

	addr := "http://" + ln.Addr().String()
	log.Printf("Opening browser. Trace viewer is listening on %s", addr)
	browser.Open(addr)

	// Start http server.
	http.HandleFunc("/", httpMain)
	err = http.Serve(ln, nil)
	dief("failed to start http server: %v\n", err)
}

var ranges []Range

var loader struct {
	once sync.Once
	res  trace.ParseResult
	err  error
}

// parseEvents is a compatibility wrapper that returns only
// the Events part of trace.ParseResult returned by parseTrace.
func parseEvents() ([]*trace.Event, error) {
	res, err := parseTrace()
	if err != nil {
		return nil, err
	}
	return res.Events, err
}

func parseTrace() (trace.ParseResult, error) {
	loader.once.Do(func() {
		tracef, err := os.Open(traceFile)
		if err != nil {
			loader.err = fmt.Errorf("failed to open trace file: %v", err)
			return
		}
		defer tracef.Close()

		// Parse and symbolize.
		res, err := trace.Parse(bufio.NewReader(tracef), programBinary)
		if err != nil {
			loader.err = fmt.Errorf("failed to parse trace: %v", err)
			return
		}
		loader.res = res
	})
	return loader.res, loader.err
}

// httpMain serves the starting page.
func httpMain(w http.ResponseWriter, r *http.Request) {
	if err := templMain.Execute(w, ranges); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

var templMain = template.Must(template.New("").Parse(`
<html>
<style>
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
</style>
<body>
<h1>cmd/trace: the Go trace event viewer</h1>
<p>
  This web server provides various visualizations of an event log gathered during
  the execution of a Go program that uses the <a href='https://pkg.go.dev/runtime/trace'>runtime/trace</a> package.
</p>

<h2>Event timelines for running goroutines</h2>
{{if $}}
<p>
  Large traces are split into multiple sections of equal data size
  (not duration) to avoid overwhelming the visualizer.
</p>
<ul>
	{{range $e := $}}
		<li><a href="{{$e.URL}}">View trace ({{$e.Name}})</a></li>
	{{end}}
</ul>
{{else}}
<ul>
	<li><a href="/trace">View trace</a></li>
</ul>
{{end}}
<p>
  This view displays a timeline for each of the GOMAXPROCS logical
  processors, showing which goroutine (if any) was running on that
  logical processor at each moment.

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
<li><a href="/syscall">Syscall blocking profile</a> (<a href="/syscall?raw=1" download="syscall.profile">⬇</a>)</li>
<li><a href="/sched">Scheduler latency profile</a> (<a href="/sche?raw=1" download="sched.profile">⬇</a>)</li>
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

func dief(msg string, args ...any) {
	fmt.Fprintf(os.Stderr, msg, args...)
	os.Exit(1)
}

var debugMemoryUsage bool

func init() {
	v := os.Getenv("DEBUG_MEMORY_USAGE")
	debugMemoryUsage = v != ""
}

func reportMemoryUsage(msg string) {
	if !debugMemoryUsage {
		return
	}
	var s runtime.MemStats
	runtime.ReadMemStats(&s)
	w := os.Stderr
	fmt.Fprintf(w, "%s\n", msg)
	fmt.Fprintf(w, " Alloc:\t%d Bytes\n", s.Alloc)
	fmt.Fprintf(w, " Sys:\t%d Bytes\n", s.Sys)
	fmt.Fprintf(w, " HeapReleased:\t%d Bytes\n", s.HeapReleased)
	fmt.Fprintf(w, " HeapSys:\t%d Bytes\n", s.HeapSys)
	fmt.Fprintf(w, " HeapInUse:\t%d Bytes\n", s.HeapInuse)
	fmt.Fprintf(w, " HeapAlloc:\t%d Bytes\n", s.HeapAlloc)
	var dummy string
	fmt.Printf("Enter to continue...")
	fmt.Scanf("%s", &dummy)
}
