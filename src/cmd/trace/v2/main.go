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
	log.Printf("Analyzing goroutines...")
	gSummaries := trace.SummarizeGoroutines(parsed.events)

	log.Printf("Opening browser. Trace viewer is listening on %s", addr)
	browser.Open(addr)

	mux := http.NewServeMux()
	mux.Handle("/", traceviewer.MainHandler(ranges))
	mux.Handle("/trace", traceviewer.TraceHandler())
	mux.Handle("/jsontrace", JSONTraceHandler(parsed))
	mux.Handle("/static/", traceviewer.StaticHandler())
	mux.HandleFunc("/goroutines", GoroutinesHandlerFunc(gSummaries))
	mux.HandleFunc("/goroutine", GoroutineHandler(gSummaries))

	// Install MMU handlers.
	mutatorUtil := func(flags trace.UtilFlags) ([][]trace.MutatorUtil, error) {
		return trace.MutatorUtilizationV2(parsed.events, flags), nil
	}
	traceviewer.InstallMMUHandlers(mux, ranges, mutatorUtil)

	err = http.Serve(ln, mux)
	return fmt.Errorf("failed to start http server: %w", err)
}

type parsedTrace struct {
	events []tracev2.Event
}

func parseTrace(trace io.Reader) (*parsedTrace, error) {
	r, err := tracev2.NewReader(trace)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace reader: %w", err)
	}
	var t parsedTrace
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("failed to read event: %w", err)
		}
		t.events = append(t.events, ev)
	}
	return &t, nil
}

// splitTrace splits the trace into a number of ranges, each resulting in approx 100 MiB of
// json output (the trace viewer can hardly handle more).
func splitTrace(parsed *parsedTrace) ([]traceviewer.Range, error) {
	// TODO(mknyszek): Split traces by generation by doing a quick first pass over the
	// trace to identify all the generation boundaries.
	s, c := traceviewer.SplittingTraceConsumer(100 << 20) // 100 MiB
	if err := generateTrace(parsed, c); err != nil {
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
