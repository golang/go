// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import "context"

type contextKeyType int

const (
	spanContextKey = contextKeyType(iota)
)

func WithSpan(ctx context.Context, span *Span) context.Context {
	return context.WithValue(ctx, spanContextKey, span)
}

func GetSpan(ctx context.Context) *Span {
	v := ctx.Value(spanContextKey)
	if v == nil {
		return nil
	}
	return v.(*Span)
}
