// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"cmd/internal/browser"
	"cmd/internal/telemetry"
	cmdv2 "cmd/trace/v2"
	"flag"
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
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
	-d=int: print debug info such as parsed events (1 for high-level, 2 for low-level)

Note that while the various profiles available when launching
'go tool trace' work on every browser, the trace viewer itself
(the 'view trace' page) comes from the Chrome/Chromium project
and is only actively tested on that browser.
`

var (
	httpFlag  = flag.String("http", "localhost:0", "HTTP service address (e.g., ':6060')")
	pprofFlag = flag.String("pprof", "", "print a pprof-like profile instead")
	debugFlag = flag.Int("d", 0, "print debug information (1 for basic debug info, 2 for lower-level info)")

	// The binary file name, left here for serveSVGProfile.
	programBinary string
	traceFile     string
)

func main() {
	telemetry.Start()
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usageMessage)
		os.Exit(2)
	}
	flag.Parse()
	telemetry.Inc("trace/invocations")
	telemetry.CountFlags("trace/flag:", *flag.CommandLine)

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

	if isTraceV2(traceFile) {
		if err := cmdv2.Main(traceFile, *httpFlag, *pprofFlag, *debugFlag); err != nil {
			dief("%s\n", err)
		}
		return
	}

	var pprofFunc traceviewer.ProfileFunc
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
		records, err := pprofFunc(&http.Request{})
		if err != nil {
			dief("failed to generate pprof: %v\n", err)
		}
		if err := traceviewer.BuildProfile(records).Write(os.Stdout); err != nil {
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

	if *debugFlag != 0 {
		trace.Print(res.Events)
		os.Exit(0)
	}
	reportMemoryUsage("after parsing trace")
	debug.FreeOSMemory()

	log.Print("Splitting trace...")
	ranges = splitTrace(res)
	reportMemoryUsage("after splitting trace")
	debug.FreeOSMemory()

	addr := "http://" + ln.Addr().String()
	log.Printf("Opening browser. Trace viewer is listening on %s", addr)
	browser.Open(addr)

	// Install MMU handler.
	http.HandleFunc("/mmu", traceviewer.MMUHandlerFunc(ranges, mutatorUtil))

	// Install main handler.
	http.Handle("/", traceviewer.MainHandler([]traceviewer.View{
		{Type: traceviewer.ViewProc, Ranges: ranges},
	}))

	// Start http server.
	err = http.Serve(ln, nil)
	dief("failed to start http server: %v\n", err)
}

// isTraceV2 returns true if filename holds a v2 trace.
func isTraceV2(filename string) bool {
	file, err := os.Open(filename)
	if err != nil {
		return false
	}
	defer file.Close()

	ver, _, err := trace.ReadVersion(file)
	if err != nil {
		return false
	}
	return ver >= 1022
}

var ranges []traceviewer.Range

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

func mutatorUtil(flags trace.UtilFlags) ([][]trace.MutatorUtil, error) {
	events, err := parseEvents()
	if err != nil {
		return nil, err
	}
	return trace.MutatorUtilization(events, flags), nil
}
