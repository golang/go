// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/browser"
	"cmd/internal/telemetry/counter"
	"cmp"
	"flag"
	"fmt"
	"internal/trace"
	"internal/trace/raw"
	"internal/trace/tracev2/event"
	"internal/trace/traceviewer"
	"io"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof" // Required to use pprof
	"os"
	"slices"
	"sync/atomic"
	"text/tabwriter"
	"time"
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
	-d=mode: print debug info and exit (modes: wire, parsed, footprint)

Note that while the various profiles available when launching
'go tool trace' work on every browser, the trace viewer itself
(the 'view trace' page) comes from the Chrome/Chromium project
and is only actively tested on that browser.
`

var (
	httpFlag  = flag.String("http", "localhost:0", "HTTP service address (e.g., ':6060')")
	pprofFlag = flag.String("pprof", "", "print a pprof-like profile instead")
	debugFlag = flag.String("d", "", "print debug info and exit (modes: wire, parsed, footprint)")

	// The binary file name, left here for serveSVGProfile.
	programBinary string
	traceFile     string
)

func main() {
	counter.Open()
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usageMessage)
		os.Exit(2)
	}
	flag.Parse()
	counter.Inc("trace/invocations")
	counter.CountFlags("trace/flag:", *flag.CommandLine)

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

	tracef, err := os.Open(traceFile)
	if err != nil {
		logAndDie(fmt.Errorf("failed to read trace file: %w", err))
	}
	defer tracef.Close()

	// Get the size of the trace file.
	fi, err := tracef.Stat()
	if err != nil {
		logAndDie(fmt.Errorf("failed to stat trace file: %v", err))
	}
	traceSize := fi.Size()

	// Handle requests for profiles.
	if *pprofFlag != "" {
		parsed, err := parseTrace(tracef, traceSize)
		if err != nil {
			logAndDie(err)
		}
		var f traceviewer.ProfileFunc
		switch *pprofFlag {
		case "net":
			f = pprofByGoroutine(computePprofIO(), parsed)
		case "sync":
			f = pprofByGoroutine(computePprofBlock(), parsed)
		case "syscall":
			f = pprofByGoroutine(computePprofSyscall(), parsed)
		case "sched":
			f = pprofByGoroutine(computePprofSched(), parsed)
		default:
			logAndDie(fmt.Errorf("unknown pprof type %s\n", *pprofFlag))
		}
		records, err := f(&http.Request{})
		if err != nil {
			logAndDie(fmt.Errorf("failed to generate pprof: %v\n", err))
		}
		if err := traceviewer.BuildProfile(records).Write(os.Stdout); err != nil {
			logAndDie(fmt.Errorf("failed to generate pprof: %v\n", err))
		}
		logAndDie(nil)
	}

	// Debug flags.
	if *debugFlag != "" {
		switch *debugFlag {
		case "parsed":
			logAndDie(debugProcessedEvents(tracef))
		case "wire":
			logAndDie(debugRawEvents(tracef))
		case "footprint":
			logAndDie(debugEventsFootprint(tracef))
		default:
			logAndDie(fmt.Errorf("invalid debug mode %s, want one of: parsed, wire, footprint", *debugFlag))
		}
	}

	ln, err := net.Listen("tcp", *httpFlag)
	if err != nil {
		logAndDie(fmt.Errorf("failed to create server socket: %w", err))
	}
	addr := "http://" + ln.Addr().String()

	log.Print("Preparing trace for viewer...")
	parsed, err := parseTraceInteractive(tracef, traceSize)
	if err != nil {
		logAndDie(err)
	}
	// N.B. tracef not needed after this point.
	// We might double-close, but that's fine; we ignore the error.
	tracef.Close()

	// Print a nice message for a partial trace.
	if parsed.err != nil {
		log.Printf("Encountered error, but able to proceed. Error: %v", parsed.err)

		lost := parsed.size - parsed.valid
		pct := float64(lost) / float64(parsed.size) * 100
		log.Printf("Lost %.2f%% of the latest trace data due to error (%s of %s)", pct, byteCount(lost), byteCount(parsed.size))
	}

	log.Print("Splitting trace for viewer...")
	ranges, err := splitTrace(parsed)
	if err != nil {
		logAndDie(err)
	}

	log.Printf("Opening browser. Trace viewer is listening on %s", addr)
	browser.Open(addr)

	mutatorUtil := func(flags trace.UtilFlags) ([][]trace.MutatorUtil, error) {
		return trace.MutatorUtilizationV2(parsed.events, flags), nil
	}

	mux := http.NewServeMux()

	// Main endpoint.
	mux.Handle("/", traceviewer.MainHandler([]traceviewer.View{
		{Type: traceviewer.ViewProc, Ranges: ranges},
		// N.B. Use the same ranges for threads. It takes a long time to compute
		// the split a second time, but the makeup of the events are similar enough
		// that this is still a good split.
		{Type: traceviewer.ViewThread, Ranges: ranges},
	}))

	// Catapult handlers.
	mux.Handle("/trace", traceviewer.TraceHandler())
	mux.Handle("/jsontrace", JSONTraceHandler(parsed))
	mux.Handle("/static/", traceviewer.StaticHandler())

	// Goroutines handlers.
	mux.HandleFunc("/goroutines", GoroutinesHandlerFunc(parsed.summary.Goroutines))
	mux.HandleFunc("/goroutine", GoroutineHandler(parsed.summary.Goroutines))

	// MMU handler.
	mux.HandleFunc("/mmu", traceviewer.MMUHandlerFunc(ranges, mutatorUtil))

	// Basic pprof endpoints.
	mux.HandleFunc("/io", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofIO(), parsed)))
	mux.HandleFunc("/block", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofBlock(), parsed)))
	mux.HandleFunc("/syscall", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofSyscall(), parsed)))
	mux.HandleFunc("/sched", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofSched(), parsed)))

	// Region-based pprof endpoints.
	mux.HandleFunc("/regionio", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofIO(), parsed)))
	mux.HandleFunc("/regionblock", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofBlock(), parsed)))
	mux.HandleFunc("/regionsyscall", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofSyscall(), parsed)))
	mux.HandleFunc("/regionsched", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofSched(), parsed)))

	// Region endpoints.
	mux.HandleFunc("/userregions", UserRegionsHandlerFunc(parsed))
	mux.HandleFunc("/userregion", UserRegionHandlerFunc(parsed))

	// Task endpoints.
	mux.HandleFunc("/usertasks", UserTasksHandlerFunc(parsed))
	mux.HandleFunc("/usertask", UserTaskHandlerFunc(parsed))

	err = http.Serve(ln, mux)
	logAndDie(fmt.Errorf("failed to start http server: %w", err))
}

func logAndDie(err error) {
	if err == nil {
		os.Exit(0)
	}
	fmt.Fprintf(os.Stderr, "%s\n", err)
	os.Exit(1)
}

func parseTraceInteractive(tr io.Reader, size int64) (parsed *parsedTrace, err error) {
	done := make(chan struct{})
	cr := countingReader{r: tr}
	go func() {
		parsed, err = parseTrace(&cr, size)
		done <- struct{}{}
	}()
	ticker := time.NewTicker(5 * time.Second)
progressLoop:
	for {
		select {
		case <-ticker.C:
		case <-done:
			ticker.Stop()
			break progressLoop
		}
		progress := cr.bytesRead.Load()
		pct := float64(progress) / float64(size) * 100
		log.Printf("%s of %s (%.1f%%) processed...", byteCount(progress), byteCount(size), pct)
	}
	return
}

type parsedTrace struct {
	events      []trace.Event
	summary     *trace.Summary
	size, valid int64
	err         error
}

func parseTrace(rr io.Reader, size int64) (*parsedTrace, error) {
	// Set up the reader.
	cr := countingReader{r: rr}
	r, err := trace.NewReader(&cr)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace reader: %w", err)
	}

	// Set up state.
	s := trace.NewSummarizer()
	t := new(parsedTrace)
	var validBytes int64
	var validEvents int
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			validBytes = cr.bytesRead.Load()
			validEvents = len(t.events)
			break
		}
		if err != nil {
			t.err = err
			break
		}
		t.events = append(t.events, ev)
		s.Event(&t.events[len(t.events)-1])

		if ev.Kind() == trace.EventSync {
			validBytes = cr.bytesRead.Load()
			validEvents = len(t.events)
		}
	}

	// Check to make sure we got at least one good generation.
	if validEvents == 0 {
		return nil, fmt.Errorf("failed to parse any useful part of the trace: %v", t.err)
	}

	// Finish off the parsedTrace.
	t.summary = s.Finalize()
	t.valid = validBytes
	t.size = size
	t.events = t.events[:validEvents]
	return t, nil
}

func (t *parsedTrace) startTime() trace.Time {
	return t.events[0].Time()
}

func (t *parsedTrace) endTime() trace.Time {
	return t.events[len(t.events)-1].Time()
}

// splitTrace splits the trace into a number of ranges, each resulting in approx 100 MiB of
// json output (the trace viewer can hardly handle more).
func splitTrace(parsed *parsedTrace) ([]traceviewer.Range, error) {
	// TODO(mknyszek): Split traces by generation by doing a quick first pass over the
	// trace to identify all the generation boundaries.
	s, c := traceviewer.SplittingTraceConsumer(100 << 20) // 100 MiB
	if err := generateTrace(parsed, defaultGenOpts(), c); err != nil {
		return nil, err
	}
	return s.Ranges, nil
}

func debugProcessedEvents(trc io.Reader) error {
	tr, err := trace.NewReader(trc)
	if err != nil {
		return err
	}
	for {
		ev, err := tr.ReadEvent()
		if err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
		fmt.Println(ev.String())
	}
}

func debugRawEvents(trc io.Reader) error {
	rr, err := raw.NewReader(trc)
	if err != nil {
		return err
	}
	for {
		ev, err := rr.ReadEvent()
		if err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
		fmt.Println(ev.String())
	}
}

func debugEventsFootprint(trc io.Reader) error {
	cr := countingReader{r: trc}
	tr, err := raw.NewReader(&cr)
	if err != nil {
		return err
	}
	type eventStats struct {
		typ   event.Type
		count int
		bytes int
	}
	var stats [256]eventStats
	for i := range stats {
		stats[i].typ = event.Type(i)
	}
	eventsRead := 0
	for {
		e, err := tr.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		s := &stats[e.Ev]
		s.count++
		s.bytes += e.EncodedSize()
		eventsRead++
	}
	slices.SortFunc(stats[:], func(a, b eventStats) int {
		return cmp.Compare(b.bytes, a.bytes)
	})
	specs := tr.Version().Specs()
	w := tabwriter.NewWriter(os.Stdout, 3, 8, 2, ' ', 0)
	fmt.Fprintf(w, "Event\tBytes\t%%\tCount\t%%\n")
	fmt.Fprintf(w, "-\t-\t-\t-\t-\n")
	for i := range stats {
		stat := &stats[i]
		name := ""
		if int(stat.typ) >= len(specs) {
			name = fmt.Sprintf("<unknown (%d)>", stat.typ)
		} else {
			name = specs[stat.typ].Name
		}
		bytesPct := float64(stat.bytes) / float64(cr.bytesRead.Load()) * 100
		countPct := float64(stat.count) / float64(eventsRead) * 100
		fmt.Fprintf(w, "%s\t%d\t%.2f%%\t%d\t%.2f%%\n", name, stat.bytes, bytesPct, stat.count, countPct)
	}
	w.Flush()
	return nil
}

type countingReader struct {
	r         io.Reader
	bytesRead atomic.Int64
}

func (c *countingReader) Read(buf []byte) (n int, err error) {
	n, err = c.r.Read(buf)
	c.bytesRead.Add(int64(n))
	return n, err
}

type byteCount int64

func (b byteCount) String() string {
	var suffix string
	var divisor int64
	switch {
	case b < 1<<10:
		suffix = "B"
		divisor = 1
	case b < 1<<20:
		suffix = "KiB"
		divisor = 1 << 10
	case b < 1<<30:
		suffix = "MiB"
		divisor = 1 << 20
	case b < 1<<40:
		suffix = "GiB"
		divisor = 1 << 30
	}
	if divisor == 1 {
		return fmt.Sprintf("%d %s", b, suffix)
	}
	return fmt.Sprintf("%.1f %s", float64(b)/float64(divisor), suffix)
}
