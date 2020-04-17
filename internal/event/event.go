// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"

	"golang.org/x/tools/internal/event/core"
)

// Exporter is a function that handles events.
// It may return a modified context and event.
type Exporter func(context.Context, core.Event, core.TagMap) context.Context

// SetExporter sets the global exporter function that handles all events.
// The exporter is called synchronously from the event call site, so it should
// return quickly so as not to hold up user code.
func SetExporter(e Exporter) {
	core.SetExporter(core.Exporter(e))
}

// Log sends a log event with the supplied tag list to the exporter.
func Log(ctx context.Context, tags ...core.Tag) {
	core.Log(ctx, tags...)
}

// Print takes a message and a tag list and combines them into a single event
// before delivering them to the exporter.
func Print(ctx context.Context, message string, tags ...core.Tag) {
	core.Print(ctx, message, tags...)
}

// Error takes a message and a tag list and combines them into a single event
// before delivering them to the exporter. It captures the error in the
// delivered event.
func Error(ctx context.Context, message string, err error, tags ...core.Tag) {
	core.Error(ctx, message, err, tags...)
}

// Record sends a label event to the exporter with the supplied tags.
func Record(ctx context.Context, tags ...core.Tag) context.Context {
	return core.Record(ctx, tags...)
}

// Label sends a label event to the exporter with the supplied tags.
func Label(ctx context.Context, tags ...core.Tag) context.Context {
	return core.Label(ctx, tags...)
}

// StartSpan sends a span start event with the supplied tag list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func StartSpan(ctx context.Context, name string, tags ...core.Tag) (context.Context, func()) {
	return core.StartSpan(ctx, name, tags...)
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return core.Detach(ctx)
}
