// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"

	"golang.org/x/tools/internal/telemetry/event"
)

// Tag returns a context updated with tag values from the event.
// It ignores events that are not or type IsTag or IsStartSpan.
func Tag(ctx context.Context, ev event.Event) context.Context {
	//TODO: Do we need to do something more efficient than just store tags
	//TODO: directly on the context?
	if ev.IsTag() || ev.IsStartSpan() {
		for _, t := range ev.Tags {
			ctx = context.WithValue(ctx, t.Key, t.Value)
		}
	}
	return ctx
}
