// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"

	"golang.org/x/tools/internal/telemetry"
)

// Tag returns a context updated with tag values from the event.
// It ignores events that are not or type EventTag or EventStartSpan.
func Tag(ctx context.Context, event telemetry.Event) context.Context {
	//TODO: Do we need to do something more efficient than just store tags
	//TODO: directly on the context?
	switch event.Type {
	case telemetry.EventTag, telemetry.EventStartSpan:
		for _, t := range event.Tags {
			ctx = context.WithValue(ctx, t.Key, t.Value)
		}
	}
	return ctx
}
