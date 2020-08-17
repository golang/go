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

var traceStarted int32

func getTraceContext(ctx context.Context) (traceContext, bool) {
	if atomic.LoadInt32(&traceStarted) == 0 {
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
	childSpan := &Span{t: tc.t, name: name, start: time.Now()}
	tc.t.writeEvent(&traceviewer.Event{
		Name:  childSpan.name,
		Time:  float64(childSpan.start.UnixNano()) / float64(time.Microsecond),
		Phase: "B",
	})
	ctx = context.WithValue(ctx, traceKey{}, traceContext{tc.t})
	return ctx, childSpan
}

type Span struct {
	t *tracer

	name  string
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
		Phase: "E",
	})
}

type tracer struct {
	file chan traceFile // 1-buffered
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

// traceKey is the context key for tracing information. It is unexported to prevent collisions with context keys defined in
// other packages.
type traceKey struct{}

type traceContext struct {
	t *tracer
}

// Start starts a trace which writes to the given file.
func Start(ctx context.Context, file string) (context.Context, func() error, error) {
	atomic.StoreInt32(&traceStarted, 1)
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
