// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package trace adds support for telemetry tracing.
package trace

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/lsp/telemetry/worker"
)

type Span struct {
	Name     string
	TraceID  TraceID
	SpanID   SpanID
	ParentID SpanID
	Start    time.Time
	Finish   time.Time
	Tags     tag.List
	Events   []Event

	ready bool
}

type Event struct {
	Time time.Time
	Tags tag.List
}

type Observer func(*Span)

func RegisterObservers(o ...Observer) {
	worker.Do(func() {
		if !registered {
			registered = true
			tag.Observe(tagObserver)
			log.AddLogger(logger)
		}
		observers = append(observers, o...)
	})
}

func StartSpan(ctx context.Context, name string, tags ...tag.Tag) (context.Context, func()) {
	span := &Span{
		Name:  name,
		Start: time.Now(),
	}
	if parent := fromContext(ctx); parent != nil {
		span.TraceID = parent.TraceID
		span.ParentID = parent.SpanID
	} else {
		span.TraceID = newTraceID()
	}
	span.SpanID = newSpanID()
	ctx = context.WithValue(ctx, contextKey, span)
	if len(tags) > 0 {
		ctx = tag.With(ctx, tags...)
	}
	worker.Do(func() {
		span.ready = true
		for _, o := range observers {
			o(span)
		}
	})
	return ctx, span.close
}

func (s *Span) close() {
	now := time.Now()
	worker.Do(func() {
		s.Finish = now
		for _, o := range observers {
			o(s)
		}
	})
}

func (s *Span) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "%v %v:%v", s.Name, s.TraceID, s.SpanID)
	if s.ParentID.IsValid() {
		fmt.Fprintf(f, "[%v]", s.ParentID)
	}
	fmt.Fprintf(f, " %v->%v", s.Start, s.Finish)
}

type contextKeyType int

var contextKey contextKeyType

func fromContext(ctx context.Context) *Span {
	v := ctx.Value(contextKey)
	if v == nil {
		return nil
	}
	return v.(*Span)
}

var (
	observers  []Observer
	registered bool
)

func tagObserver(ctx context.Context, at time.Time, tags tag.List) {
	span := fromContext(ctx)
	if span == nil {
		return
	}
	if !span.ready {
		span.Tags = append(span.Tags, tags...)
		return
	}
	span.Events = append(span.Events, Event{
		Time: at,
		Tags: tags,
	})
}

func logger(ctx context.Context, at time.Time, tags tag.List) bool {
	span := fromContext(ctx)
	if span == nil {
		return false
	}
	span.Events = append(span.Events, Event{
		Time: at,
		Tags: tags,
	})
	return false
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return context.WithValue(ctx, contextKey, nil)
}
