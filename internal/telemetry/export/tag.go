// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"

	"golang.org/x/tools/internal/telemetry/event"
)

// Tag manipulates the context using the event.
// If the event is type IsTag or IsStartSpan then it returns a context updated
// with tag values from the event.
// For all other event types the event tags will be updated with values from the
// context if they are missing.
func Tag(ctx context.Context, ev event.Event) (context.Context, event.Event) {
	//TODO: Do we need to do something more efficient than just store tags
	//TODO: directly on the context?
	switch {
	case ev.IsLabel(), ev.IsStartSpan():
		for _, t := range ev.Tags {
			ctx = context.WithValue(ctx, t.Key(), t.Value())
		}
	default:
		// all other types want the tags filled in if needed
		for i := range ev.Tags {
			if ev.Tags[i].Value() == nil {
				key := ev.Tags[i].Key()
				ev.Tags[i] = key.OfValue(ctx.Value(key.Identity()))
			}
		}
	}
	return ctx, ev
}
