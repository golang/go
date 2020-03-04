// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/tools/internal/telemetry"
)

type SpanContext struct {
	TraceID TraceID
	SpanID  SpanID
}

type Span struct {
	Name     string
	ID       SpanContext
	ParentID SpanID
	Start    time.Time
	Finish   time.Time
	Tags     telemetry.TagList
	Events   []telemetry.Event
}

type contextKeyType int

const (
	spanContextKey = contextKeyType(iota)
)

func GetSpan(ctx context.Context) *Span {
	v := ctx.Value(spanContextKey)
	if v == nil {
		return nil
	}
	return v.(*Span)
}

// ContextSpan is an exporter that maintains hierarchical span structure in the
// context.
// It creates new spans on EventStartSpan, adds events to the current span on
// EventLog or EventTag, and closes the span on EventEndSpan.
// The span structure can then be used by other exporters.
func ContextSpan(ctx context.Context, event telemetry.Event) context.Context {
	switch event.Type {
	case telemetry.EventLog, telemetry.EventTag:
		if span := GetSpan(ctx); span != nil {
			span.Events = append(span.Events, event)
		}
	case telemetry.EventStartSpan:
		span := &Span{
			Name:  event.Message,
			Start: event.At,
			Tags:  event.Tags,
		}
		if parent := GetSpan(ctx); parent != nil {
			span.ID.TraceID = parent.ID.TraceID
			span.ParentID = parent.ID.SpanID
		} else {
			span.ID.TraceID = newTraceID()
		}
		span.ID.SpanID = newSpanID()
		ctx = context.WithValue(ctx, spanContextKey, span)
	case telemetry.EventEndSpan:
		if span := GetSpan(ctx); span != nil {
			span.Finish = event.At
		}
	}
	return ctx
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return context.WithValue(ctx, spanContextKey, nil)
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
