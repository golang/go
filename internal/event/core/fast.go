// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package core

import (
	"context"
)

// Log1 takes a message and one tag delivers a log event to the exporter.
// It is a customized version of Print that is faster and does no allocation.
func Log1(ctx context.Context, message string, t1 Tag) {
	Export(ctx, MakeEvent(LogType, sTags{Msg.Of(message), t1}, nil))
}

// Log2 takes a message and two tags and delivers a log event to the exporter.
// It is a customized version of Print that is faster and does no allocation.
func Log2(ctx context.Context, message string, t1 Tag, t2 Tag) {
	Export(ctx, MakeEvent(LogType, sTags{Msg.Of(message), t1, t2}, nil))
}

// Metric1 sends a label event to the exporter with the supplied tags.
func Metric1(ctx context.Context, t1 Tag) context.Context {
	return Export(ctx, MakeEvent(RecordType, sTags{t1}, nil))
}

// Metric2 sends a label event to the exporter with the supplied tags.
func Metric2(ctx context.Context, t1, t2 Tag) context.Context {
	return Export(ctx, MakeEvent(RecordType, sTags{t1, t2}, nil))
}

// Metric3 sends a label event to the exporter with the supplied tags.
func Metric3(ctx context.Context, t1, t2, t3 Tag) context.Context {
	return Export(ctx, MakeEvent(RecordType, sTags{t1, t2, t3}, nil))
}

// Start1 sends a span start event with the supplied tag list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func Start1(ctx context.Context, name string, t1 Tag) (context.Context, func()) {
	return ExportPair(ctx,
		MakeEvent(StartSpanType, sTags{Name.Of(name), t1}, nil),
		MakeEvent(EndSpanType, sTags{}, nil))
}

// Start2 sends a span start event with the supplied tag list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func Start2(ctx context.Context, name string, t1, t2 Tag) (context.Context, func()) {
	return ExportPair(ctx,
		MakeEvent(StartSpanType, sTags{Name.Of(name), t1, t2}, nil),
		MakeEvent(EndSpanType, sTags{}, nil))
}
