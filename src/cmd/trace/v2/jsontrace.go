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

	"internal/trace/traceviewer"
	tracev2 "internal/trace/v2"
)

func JSONTraceHandler(parsed *parsedTrace) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		if err := generateTrace(parsed, c); err != nil {
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

func generateTrace(parsed *parsedTrace, c traceviewer.TraceConsumer) error {
	ctx := &traceContext{
		Emitter:   traceviewer.NewEmitter(c, 0, time.Duration(0), time.Duration(math.MaxInt64)),
		startTime: parsed.events[0].Time(),
	}
	defer ctx.Flush()

	runGenerator(ctx, newProcGenerator(), parsed)
	return nil
}
