// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"cmp"
	"log"
	"math"
	"net/http"
	"slices"
	"strconv"
	"time"

	"internal/trace"
	"internal/trace/traceviewer"
	tracev2 "internal/trace/v2"
)

func JSONTraceHandler(parsed *parsedTrace) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		opts := defaultGenOpts()

		switch r.FormValue("view") {
		case "thread":
			opts.mode = traceviewer.ModeThreadOriented
		}
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
		} else if taskids := r.FormValue("focustask"); taskids != "" {
			taskid, err := strconv.ParseUint(taskids, 10, 64)
			if err != nil {
				log.Printf("failed to parse focustask parameter %q: %v", taskids, err)
				return
			}
			task, ok := parsed.summary.Tasks[tracev2.TaskID(taskid)]
			if !ok || (task.Start == nil && task.End == nil) {
				log.Printf("failed to find task with id %d", taskid)
				return
			}
			opts.setTask(parsed, task)
		} else if taskids := r.FormValue("taskid"); taskids != "" {
			taskid, err := strconv.ParseUint(taskids, 10, 64)
			if err != nil {
				log.Printf("failed to parse taskid parameter %q: %v", taskids, err)
				return
			}
			task, ok := parsed.summary.Tasks[tracev2.TaskID(taskid)]
			if !ok {
				log.Printf("failed to find task with id %d", taskid)
				return
			}
			// This mode is goroutine-oriented.
			opts.mode = traceviewer.ModeGoroutineOriented
			opts.setTask(parsed, task)

			// Pick the goroutine to orient ourselves around by just
			// trying to pick the earliest event in the task that makes
			// any sense. Though, we always want the start if that's there.
			var firstEv *tracev2.Event
			if task.Start != nil {
				firstEv = task.Start
			} else {
				for _, logEv := range task.Logs {
					if firstEv == nil || logEv.Time() < firstEv.Time() {
						firstEv = logEv
					}
				}
				if task.End != nil && (firstEv == nil || task.End.Time() < firstEv.Time()) {
					firstEv = task.End
				}
			}
			if firstEv == nil || firstEv.Goroutine() == tracev2.NoGoroutine {
				log.Printf("failed to find task with id %d", taskid)
				return
			}

			// Set the goroutine filtering options.
			goid := firstEv.Goroutine()
			opts.focusGoroutine = goid
			goroutines := make(map[tracev2.GoID]struct{})
			for _, task := range opts.tasks {
				// Find only directly involved goroutines.
				for id := range task.Goroutines {
					goroutines[id] = struct{}{}
				}
			}
			opts.goroutines = goroutines
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
	endTime   tracev2.Time
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
	tasks          []*trace.UserTaskSummary
}

// setTask sets a task to focus on.
func (opts *genOpts) setTask(parsed *parsedTrace, task *trace.UserTaskSummary) {
	opts.mode |= traceviewer.ModeTaskOriented
	if task.Start != nil {
		opts.startTime = task.Start.Time().Sub(parsed.startTime())
	} else { // The task started before the trace did.
		opts.startTime = 0
	}
	if task.End != nil {
		opts.endTime = task.End.Time().Sub(parsed.startTime())
	} else { // The task didn't end.
		opts.endTime = parsed.endTime().Sub(parsed.startTime())
	}
	opts.tasks = task.Descendents()
	slices.SortStableFunc(opts.tasks, func(a, b *trace.UserTaskSummary) int {
		aStart, bStart := parsed.startTime(), parsed.startTime()
		if a.Start != nil {
			aStart = a.Start.Time()
		}
		if b.Start != nil {
			bStart = b.Start.Time()
		}
		if a.Start != b.Start {
			return cmp.Compare(aStart, bStart)
		}
		// Break ties with the end time.
		aEnd, bEnd := parsed.endTime(), parsed.endTime()
		if a.End != nil {
			aEnd = a.End.Time()
		}
		if b.End != nil {
			bEnd = b.End.Time()
		}
		return cmp.Compare(aEnd, bEnd)
	})
}

func defaultGenOpts() *genOpts {
	return &genOpts{
		startTime: time.Duration(0),
		endTime:   time.Duration(math.MaxInt64),
	}
}

func generateTrace(parsed *parsedTrace, opts *genOpts, c traceviewer.TraceConsumer) error {
	ctx := &traceContext{
		Emitter:   traceviewer.NewEmitter(c, opts.startTime, opts.endTime),
		startTime: parsed.events[0].Time(),
		endTime:   parsed.events[len(parsed.events)-1].Time(),
	}
	defer ctx.Flush()

	var g generator
	if opts.mode&traceviewer.ModeGoroutineOriented != 0 {
		g = newGoroutineGenerator(ctx, opts.focusGoroutine, opts.goroutines)
	} else if opts.mode&traceviewer.ModeThreadOriented != 0 {
		g = newThreadGenerator()
	} else {
		g = newProcGenerator()
	}
	runGenerator(ctx, g, parsed, opts)
	return nil
}
