// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"
	"sync"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
)

type SpanContext struct {
	TraceID TraceID
	SpanID  SpanID
}

type Span struct {
	Name     string
	ID       SpanContext
	ParentID SpanID
	mu       sync.Mutex
	start    core.Event
	finish   core.Event
	events   []core.Event
}

type contextKeyType int

const (
	spanContextKey = contextKeyType(iota)
	labelContextKey
)

func GetSpan(ctx context.Context) *Span {
	v := ctx.Value(spanContextKey)
	if v == nil {
		return nil
	}
	return v.(*Span)
}

// Spans creates an exporter that maintains hierarchical span structure in the
// context.
// It creates new spans on start events, adds events to the current span on
// log or label, and closes the span on end events.
// The span structure can then be used by other exporters.
func Spans(output event.Exporter) event.Exporter {
	return func(ctx context.Context, ev core.Event, lm label.Map) context.Context {
		switch {
		case event.IsLog(ev), event.IsLabel(ev):
			if span := GetSpan(ctx); span != nil {
				span.mu.Lock()
				span.events = append(span.events, ev)
				span.mu.Unlock()
			}
		case event.IsStart(ev):
			span := &Span{
				Name:  keys.Start.Get(lm),
				start: ev,
			}
			if parent := GetSpan(ctx); parent != nil {
				span.ID.TraceID = parent.ID.TraceID
				span.ParentID = parent.ID.SpanID
			} else {
				span.ID.TraceID = newTraceID()
			}
			span.ID.SpanID = newSpanID()
			ctx = context.WithValue(ctx, spanContextKey, span)
		case event.IsEnd(ev):
			if span := GetSpan(ctx); span != nil {
				span.mu.Lock()
				span.finish = ev
				span.mu.Unlock()
			}
		case event.IsDetach(ev):
			ctx = context.WithValue(ctx, spanContextKey, nil)
		}
		return output(ctx, ev, lm)
	}
}

func (s *SpanContext) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "%v:%v", s.TraceID, s.SpanID)
}

func (s *Span) Start() core.Event {
	// start never changes after construction, so we don't need to hold the mutex
	return s.start
}

func (s *Span) Finish() core.Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.finish
}

func (s *Span) Events() []core.Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.events
}

func (s *Span) Format(f fmt.State, r rune) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(f, "%v %v", s.Name, s.ID)
	if s.ParentID.IsValid() {
		fmt.Fprintf(f, "[%v]", s.ParentID)
	}
	fmt.Fprintf(f, " %v->%v", s.start, s.finish)
}
