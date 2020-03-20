// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/telemetry/event"
)

type SpanContext struct {
	TraceID TraceID
	SpanID  SpanID
}

type Span struct {
	Name     string
	ID       SpanContext
	ParentID SpanID
	Start    event.Event
	Finish   event.Event
	Events   []event.Event
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
// It creates new spans on EventStartSpan, adds events to the current span on
// EventLog or EventTag, and closes the span on EventEndSpan.
// The span structure can then be used by other exporters.
func Spans(output event.Exporter) event.Exporter {
	return func(ctx context.Context, ev event.Event, tagMap event.TagMap) context.Context {
		switch {
		case ev.IsLog(), ev.IsLabel():
			if span := GetSpan(ctx); span != nil {
				span.Events = append(span.Events, ev)
			}
		case ev.IsStartSpan():
			span := &Span{
				Name:  event.Name.Get(tagMap),
				Start: ev,
			}
			if parent := GetSpan(ctx); parent != nil {
				span.ID.TraceID = parent.ID.TraceID
				span.ParentID = parent.ID.SpanID
			} else {
				span.ID.TraceID = newTraceID()
			}
			span.ID.SpanID = newSpanID()
			ctx = context.WithValue(ctx, spanContextKey, span)
		case ev.IsEndSpan():
			if span := GetSpan(ctx); span != nil {
				span.Finish = ev
			}
		case ev.IsDetach():
			ctx = context.WithValue(ctx, spanContextKey, nil)
		}
		return output(ctx, ev, tagMap)
	}
}

func (s *SpanContext) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "%v:%v", s.TraceID, s.SpanID)
}

func (s *Span) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "%v %v", s.Name, s.ID)
	if s.ParentID.IsValid() {
		fmt.Fprintf(f, "[%v]", s.ParentID)
	}
	fmt.Fprintf(f, " %v->%v", s.Start, s.Finish)
}
