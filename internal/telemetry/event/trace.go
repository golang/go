// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"time"
)

func StartSpan(ctx context.Context, name string, tags ...Tag) (context.Context, func()) {
	ctx = ProcessEvent(ctx, Event{
		Type:    StartSpanType,
		Message: name,
		At:      time.Now(),
		Tags:    tags,
	})
	return ctx, func() {
		ProcessEvent(ctx, Event{
			Type: EndSpanType,
			At:   time.Now(),
		})
	}
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return ProcessEvent(ctx, Event{
		Type: DetachType,
		At:   time.Now(),
	})
}
