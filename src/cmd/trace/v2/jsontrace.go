// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"log"
	"math"
	"net/http"
	"strconv"
	"time"

	"internal/trace"
	"internal/trace/traceviewer"
	tracev2 "internal/trace/v2"
)

func JSONTraceHandler(parsed *parsedTrace) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		opts := defaultGenOpts()

		if goids := r.FormValue("goid"); goids != "" {
			// Render trace focused on a particular goroutine.

			id, err := strconv.ParseUint(goids, 10, 64)
			if err != nil {
				log.Printf("failed to parse goid parameter %q: %v", goids, err)
				return
			}
			goid := tracev2.GoID(id)
			g, ok := parsed.summary.Goroutines[goid]
			if !ok {
				log.Printf("failed to find goroutine %d", goid)
				return
			}
			opts.mode = traceviewer.ModeGoroutineOriented
			if g.StartTime != 0 {
				opts.startTime = g.StartTime.Sub(parsed.startTime())
			} else {
				opts.startTime = 0
			}
			if g.EndTime != 0 {
				opts.endTime = g.EndTime.Sub(parsed.startTime())
			} else { // The goroutine didn't end.
				opts.endTime = parsed.endTime().Sub(parsed.startTime())
			}
			opts.focusGoroutine = goid
			opts.goroutines = trace.RelatedGoroutinesV2(parsed.events, goid)
		}

		// Parse start and end options. Both or none must be present.
		start := int64(0)
		end := int64(math.MaxInt64)
		if startStr, endStr := r.FormValue("start"), r.FormValue("end"); startStr != "" && endStr != "" {
			var err error
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
		if err := generateTrace(parsed, opts, c); err != nil {
			log.Printf("failed to generate trace: %v", err)
		}
	})
}

// traceContext is a wrapper around a traceviewer.Emitter with some additional
// information that's useful to most parts of trace viewer JSON emission.
type traceContext struct {
	*traceviewer.Emitter
	startTime tracev2.Time
}

// elapsed returns the elapsed time between the trace time and the start time
// of the trace.
func (ctx *traceContext) elapsed(now tracev2.Time) time.Duration {
	return now.Sub(ctx.startTime)
}

type genOpts struct {
	mode      traceviewer.Mode
	startTime time.Duration
	endTime   time.Duration

	// Used if mode != 0.
	focusGoroutine tracev2.GoID
	goroutines     map[tracev2.GoID]struct{} // Goroutines to be displayed for goroutine-oriented or task-oriented view. goroutines[0] is the main goroutine.
}

func defaultGenOpts() *genOpts {
	return &genOpts{
		startTime: time.Duration(0),
		endTime:   time.Duration(math.MaxInt64),
	}
}

func generateTrace(parsed *parsedTrace, opts *genOpts, c traceviewer.TraceConsumer) error {
	ctx := &traceContext{
		Emitter:   traceviewer.NewEmitter(c, 0, opts.startTime, opts.endTime),
		startTime: parsed.events[0].Time(),
	}
	defer ctx.Flush()

	var g generator
	if opts.mode&traceviewer.ModeGoroutineOriented != 0 {
		g = newGoroutineGenerator(ctx, opts.focusGoroutine, opts.goroutines)
	} else {
		g = newProcGenerator()
	}
	runGenerator(ctx, g, parsed)
	return nil
}
