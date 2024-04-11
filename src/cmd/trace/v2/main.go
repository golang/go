// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	tracev2 "internal/trace/v2"
	"io"
	"log"
	"net"
	"net/http"
	"os"

	"internal/trace/v2/raw"

	"cmd/internal/browser"
)

// Main is the main function for cmd/trace v2.
func Main(traceFile, httpAddr, pprof string, debug int) error {
	tracef, err := os.Open(traceFile)
	if err != nil {
		return fmt.Errorf("failed to read trace file: %w", err)
	}
	defer tracef.Close()

	// Handle requests for profiles.
	if pprof != "" {
		parsed, err := parseTrace(tracef)
		if err != nil {
			return err
		}
		var f traceviewer.ProfileFunc
		switch pprof {
		case "net":
			f = pprofByGoroutine(computePprofIO(), parsed)
		case "sync":
			f = pprofByGoroutine(computePprofBlock(), parsed)
		case "syscall":
			f = pprofByGoroutine(computePprofSyscall(), parsed)
		case "sched":
			f = pprofByGoroutine(computePprofSched(), parsed)
		default:
			return fmt.Errorf("unknown pprof type %s\n", pprof)
		}
		records, err := f(&http.Request{})
		if err != nil {
			return fmt.Errorf("failed to generate pprof: %v\n", err)
		}
		if err := traceviewer.BuildProfile(records).Write(os.Stdout); err != nil {
			return fmt.Errorf("failed to generate pprof: %v\n", err)
		}
		return nil
	}

	// Debug flags.
	switch debug {
	case 1:
		return debugProcessedEvents(tracef)
	case 2:
		return debugRawEvents(tracef)
	}

	ln, err := net.Listen("tcp", httpAddr)
	if err != nil {
		return fmt.Errorf("failed to create server socket: %w", err)
	}
	addr := "http://" + ln.Addr().String()

	log.Print("Preparing trace for viewer...")
	parsed, err := parseTrace(tracef)
	if err != nil {
		return err
	}
	// N.B. tracef not needed after this point.
	// We might double-close, but that's fine; we ignore the error.
	tracef.Close()

	log.Print("Splitting trace for viewer...")
	ranges, err := splitTrace(parsed)
	if err != nil {
		return err
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
	return fmt.Errorf("failed to start http server: %w", err)
}

type parsedTrace struct {
	events  []tracev2.Event
	summary *trace.Summary
}

func parseTrace(tr io.Reader) (*parsedTrace, error) {
	r, err := tracev2.NewReader(tr)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace reader: %w", err)
	}
	s := trace.NewSummarizer()
	t := new(parsedTrace)
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("failed to read event: %w", err)
		}
		t.events = append(t.events, ev)
		s.Event(&t.events[len(t.events)-1])
	}
	t.summary = s.Finalize()
	return t, nil
}

func (t *parsedTrace) startTime() tracev2.Time {
	return t.events[0].Time()
}

func (t *parsedTrace) endTime() tracev2.Time {
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

func debugProcessedEvents(trace io.Reader) error {
	tr, err := tracev2.NewReader(trace)
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

func debugRawEvents(trace io.Reader) error {
	rr, err := raw.NewReader(trace)
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
