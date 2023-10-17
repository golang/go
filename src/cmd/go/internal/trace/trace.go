// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"cmd/internal/traceviewer"
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

// Constants used in event fields.
// See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
// for more details.
const (
	phaseDurationBegin = "B"
	phaseDurationEnd   = "E"
	phaseFlowStart     = "s"
	phaseFlowEnd       = "f"

	bindEnclosingSlice = "e"
)

var traceStarted atomic.Bool

func getTraceContext(ctx context.Context) (traceContext, bool) {
	if !traceStarted.Load() {
		return traceContext{}, false
	}
	v := ctx.Value(traceKey{})
	if v == nil {
		return traceContext{}, false
	}
	return v.(traceContext), true
}

// StartSpan starts a trace event with the given name. The Span ends when its Done method is called.
func StartSpan(ctx context.Context, name string) (context.Context, *Span) {
	tc, ok := getTraceContext(ctx)
	if !ok {
		return ctx, nil
	}
	childSpan := &Span{t: tc.t, name: name, tid: tc.tid, start: time.Now()}
	tc.t.writeEvent(&traceviewer.Event{
		Name:  childSpan.name,
		Time:  float64(childSpan.start.UnixNano()) / float64(time.Microsecond),
		TID:   childSpan.tid,
		Phase: phaseDurationBegin,
	})
	ctx = context.WithValue(ctx, traceKey{}, traceContext{tc.t, tc.tid})
	return ctx, childSpan
}

// StartGoroutine associates the context with a new Thread ID. The Chrome trace viewer associates each
// trace event with a thread, and doesn't expect events with the same thread id to happen at the
// same time.
func StartGoroutine(ctx context.Context) context.Context {
	tc, ok := getTraceContext(ctx)
	if !ok {
		return ctx
	}
	return context.WithValue(ctx, traceKey{}, traceContext{tc.t, tc.t.getNextTID()})
}

// Flow marks a flow indicating that the 'to' span depends on the 'from' span.
// Flow should be called while the 'to' span is in progress.
func Flow(ctx context.Context, from *Span, to *Span) {
	tc, ok := getTraceContext(ctx)
	if !ok || from == nil || to == nil {
		return
	}

	id := tc.t.getNextFlowID()
	tc.t.writeEvent(&traceviewer.Event{
		Name:     from.name + " -> " + to.name,
		Category: "flow",
		ID:       id,
		Time:     float64(from.end.UnixNano()) / float64(time.Microsecond),
		Phase:    phaseFlowStart,
		TID:      from.tid,
	})
	tc.t.writeEvent(&traceviewer.Event{
		Name:      from.name + " -> " + to.name,
		Category:  "flow", // TODO(matloob): Add Category to Flow?
		ID:        id,
		Time:      float64(to.start.UnixNano()) / float64(time.Microsecond),
		Phase:     phaseFlowEnd,
		TID:       to.tid,
		BindPoint: bindEnclosingSlice,
	})
}

type Span struct {
	t *tracer

	name  string
	tid   uint64
	start time.Time
	end   time.Time
}

func (s *Span) Done() {
	if s == nil {
		return
	}
	s.end = time.Now()
	s.t.writeEvent(&traceviewer.Event{
		Name:  s.name,
		Time:  float64(s.end.UnixNano()) / float64(time.Microsecond),
		TID:   s.tid,
		Phase: phaseDurationEnd,
	})
}

type tracer struct {
	file chan traceFile // 1-buffered

	nextTID    uint64
	nextFlowID uint64
}

func (t *tracer) writeEvent(ev *traceviewer.Event) error {
	f := <-t.file
	defer func() { t.file <- f }()
	var err error
	if f.entries == 0 {
		_, err = f.sb.WriteString("[\n")
	} else {
		_, err = f.sb.WriteString(",")
	}
	f.entries++
	if err != nil {
		return nil
	}

	if err := f.enc.Encode(ev); err != nil {
		return err
	}

	// Write event string to output file.
	_, err = f.f.WriteString(f.sb.String())
	f.sb.Reset()
	return err
}

func (t *tracer) Close() error {
	f := <-t.file
	defer func() { t.file <- f }()

	_, firstErr := f.f.WriteString("]")
	if err := f.f.Close(); firstErr == nil {
		firstErr = err
	}
	return firstErr
}

func (t *tracer) getNextTID() uint64 {
	return atomic.AddUint64(&t.nextTID, 1)
}

func (t *tracer) getNextFlowID() uint64 {
	return atomic.AddUint64(&t.nextFlowID, 1)
}

// traceKey is the context key for tracing information. It is unexported to prevent collisions with context keys defined in
// other packages.
type traceKey struct{}

type traceContext struct {
	t   *tracer
	tid uint64
}

// Start starts a trace which writes to the given file.
func Start(ctx context.Context, file string) (context.Context, func() error, error) {
	traceStarted.Store(true)
	if file == "" {
		return nil, nil, errors.New("no trace file supplied")
	}
	f, err := os.Create(file)
	if err != nil {
		return nil, nil, err
	}
	t := &tracer{file: make(chan traceFile, 1)}
	sb := new(strings.Builder)
	t.file <- traceFile{
		f:   f,
		sb:  sb,
		enc: json.NewEncoder(sb),
	}
	ctx = context.WithValue(ctx, traceKey{}, traceContext{t: t})
	return ctx, t.Close, nil
}

type traceFile struct {
	f       *os.File
	sb      *strings.Builder
	enc     *json.Encoder
	entries int64
}
