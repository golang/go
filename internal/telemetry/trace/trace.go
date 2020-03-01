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
)

func StartSpan(ctx context.Context, name string, tags ...telemetry.Tag) (context.Context, func()) {
	ctx = export.ProcessEvent(ctx, telemetry.Event{
		Type:    telemetry.EventStartSpan,
		Message: name,
		At:      time.Now(),
		Tags:    tags,
	})
	return ctx, func() {
		export.ProcessEvent(ctx, telemetry.Event{
			Type: telemetry.EventEndSpan,
			At:   time.Now(),
		})
	}
}

// Detach returns a context without an associated span.
// This allows the creation of spans that are not children of the current span.
func Detach(ctx context.Context) context.Context {
	return export.Detach(ctx)
}
