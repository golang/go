// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
)

func StartSpan(ctx context.Context, name string, tags ...Tag) (context.Context, func()) {
	ctx = dispatch(ctx, makeEvent(StartSpanType, sTags{Name.Of(name)}, tags))
	return ctx, func() { dispatch(ctx, makeEvent(EndSpanType, sTags{}, nil)) }
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return dispatch(ctx, makeEvent(DetachType, sTags{}, nil))
}
