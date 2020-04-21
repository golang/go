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
	core.Export(ctx, core.MakeEvent([3]label.Label{
		keys.Msg.Of(message),
	}, labels))
}

// IsLog returns true if the event was built by the Log function.
// It is intended to be used in exporters to identify the semantics of the
// event when deciding what to do with it.
func IsLog(ev core.Event) bool {
	return ev.Label(0).Key() == keys.Msg
}

// Error takes a message and a label list and combines them into a single event
// before delivering them to the exporter. It captures the error in the
// delivered event.
func Error(ctx context.Context, message string, err error, labels ...label.Label) {
	core.Export(ctx, core.MakeEvent([3]label.Label{
		keys.Msg.Of(message),
		keys.Err.Of(err),
	}, labels))
}

// IsError returns true if the event was built by the Error function.
// It is intended to be used in exporters to identify the semantics of the
// event when deciding what to do with it.
func IsError(ev core.Event) bool {
	return ev.Label(0).Key() == keys.Msg &&
		ev.Label(1).Key() == keys.Err
}

// Metric sends a label event to the exporter with the supplied labels.
func Metric(ctx context.Context, labels ...label.Label) {
	core.Export(ctx, core.MakeEvent([3]label.Label{
		keys.Metric.New(),
	}, labels))
}

// IsMetric returns true if the event was built by the Metric function.
// It is intended to be used in exporters to identify the semantics of the
// event when deciding what to do with it.
func IsMetric(ev core.Event) bool {
	return ev.Label(0).Key() == keys.Metric
}

// Label sends a label event to the exporter with the supplied labels.
func Label(ctx context.Context, labels ...label.Label) context.Context {
	return core.Export(ctx, core.MakeEvent([3]label.Label{
		keys.Label.New(),
	}, labels))
}

// IsLabel returns true if the event was built by the Label function.
// It is intended to be used in exporters to identify the semantics of the
// event when deciding what to do with it.
func IsLabel(ev core.Event) bool {
	return ev.Label(0).Key() == keys.Label
}

// Start sends a span start event with the supplied label list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func Start(ctx context.Context, name string, labels ...label.Label) (context.Context, func()) {
	return core.ExportPair(ctx,
		core.MakeEvent([3]label.Label{
			keys.Start.Of(name),
		}, labels),
		core.MakeEvent([3]label.Label{
			keys.End.New(),
		}, nil))
}

// IsStart returns true if the event was built by the Start function.
// It is intended to be used in exporters to identify the semantics of the
// event when deciding what to do with it.
func IsStart(ev core.Event) bool {
	return ev.Label(0).Key() == keys.Start
}

// IsEnd returns true if the event was built by the End function.
// It is intended to be used in exporters to identify the semantics of the
// event when deciding what to do with it.
func IsEnd(ev core.Event) bool {
	return ev.Label(0).Key() == keys.End
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return core.Export(ctx, core.MakeEvent([3]label.Label{
		keys.Detach.New(),
	}, nil))
}

// IsDetach returns true if the event was built by the Detach function.
// It is intended to be used in exporters to identify the semantics of the
// event when deciding what to do with it.
func IsDetach(ev core.Event) bool {
	return ev.Label(0).Key() == keys.Detach
}
