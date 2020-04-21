// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"

	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
)

// Exporter is a function that handles events.
// It may return a modified context and event.
type Exporter func(context.Context, core.Event, label.Map) context.Context

// SetExporter sets the global exporter function that handles all events.
// The exporter is called synchronously from the event call site, so it should
// return quickly so as not to hold up user code.
func SetExporter(e Exporter) {
	core.SetExporter(core.Exporter(e))
}

// Log takes a message and a label list and combines them into a single event
// before delivering them to the exporter.
func Log(ctx context.Context, message string, labels ...label.Label) {
	core.Export(ctx, core.MakeEvent(core.LogType, [3]label.Label{
		keys.Msg.Of(message),
	}, labels))
}

// Error takes a message and a label list and combines them into a single event
// before delivering them to the exporter. It captures the error in the
// delivered event.
func Error(ctx context.Context, message string, err error, labels ...label.Label) {
	core.Export(ctx, core.MakeEvent(core.LogType, [3]label.Label{
		keys.Msg.Of(message),
		keys.Err.Of(err),
	}, labels))
}

// Metric sends a label event to the exporter with the supplied labels.
func Metric(ctx context.Context, labels ...label.Label) {
	core.Export(ctx, core.MakeEvent(core.RecordType, [3]label.Label{}, labels))
}

// Label sends a label event to the exporter with the supplied labels.
func Label(ctx context.Context, labels ...label.Label) context.Context {
	return core.Export(ctx, core.MakeEvent(core.LabelType, [3]label.Label{}, labels))
}

// Start sends a span start event with the supplied label list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func Start(ctx context.Context, name string, labels ...label.Label) (context.Context, func()) {
	return core.ExportPair(ctx,
		core.MakeEvent(core.StartSpanType, [3]label.Label{
			keys.Name.Of(name),
		}, labels),
		core.MakeEvent(core.EndSpanType, [3]label.Label{}, nil))
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return core.Export(ctx, core.MakeEvent(core.DetachType, [3]label.Label{}, nil))
}
