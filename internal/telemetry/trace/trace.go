// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package trace adds support for telemetry tracing.
package trace

import (
	"context"
	"time"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/tag"
)

func StartSpan(ctx context.Context, name string, tags ...telemetry.Tag) (context.Context, func()) {
	start := time.Now()
	span := &telemetry.Span{Name: name}
	if parent := telemetry.GetSpan(ctx); parent != nil {
		span.ID.TraceID = parent.ID.TraceID
		span.ParentID = parent.ID.SpanID
	} else {
		span.ID.TraceID = telemetry.NewTraceID()
	}
	span.ID.SpanID = telemetry.NewSpanID()
	ctx = telemetry.WithSpan(ctx, span)
	if len(tags) > 0 {
		ctx = tag.With(ctx, tags...)
	}
	export.StartSpan(ctx, span, start)
	return ctx, func() { export.FinishSpan(ctx, span, time.Now()) }
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return telemetry.WithSpan(ctx, nil)
}
