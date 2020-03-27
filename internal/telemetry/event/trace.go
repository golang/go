// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
)

// StartSpan sends a span start event with the supplied tag list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func StartSpan(ctx context.Context, name string, tags ...Tag) (context.Context, func()) {
	return dispatchPair(ctx,
		makeEvent(StartSpanType, sTags{Name.Of(name)}, tags),
		makeEvent(EndSpanType, sTags{}, nil))
}

// StartSpan1 sends a span start event with the supplied tag list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func StartSpan1(ctx context.Context, name string, t1 Tag) (context.Context, func()) {
	return dispatchPair(ctx,
		makeEvent(StartSpanType, sTags{Name.Of(name), t1}, nil),
		makeEvent(EndSpanType, sTags{}, nil))
}

// StartSpan2 sends a span start event with the supplied tag list to the exporter.
// It also returns a function that will end the span, which should normally be
// deferred.
func StartSpan2(ctx context.Context, name string, t1, t2 Tag) (context.Context, func()) {
	return dispatchPair(ctx,
		makeEvent(StartSpanType, sTags{Name.Of(name), t1, t2}, nil),
		makeEvent(EndSpanType, sTags{}, nil))
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return dispatch(ctx, makeEvent(DetachType, sTags{}, nil))
}
