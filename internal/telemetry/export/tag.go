// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"

	"golang.org/x/tools/internal/telemetry/event"
)

// Labels builds an exporter that manipulates the context using the event.
// If the event is type IsTag or IsStartSpan then it returns a context updated
// with tag values from the event.
// For all other event types the event tags will be updated with values from the
// context if they are missing.
func Labels(output event.Exporter) event.Exporter {
	return func(ctx context.Context, ev event.Event, tagMap event.TagMap) context.Context {
		stored, _ := ctx.Value(labelContextKey).(event.TagMap)
		if ev.IsLabel() || ev.IsStartSpan() {
			// update the tag source stored in the context
			fromEvent := event.TagMap(ev)
			if stored == nil {
				stored = fromEvent
			} else {
				stored = event.MergeTagMaps(fromEvent, stored)
			}
			ctx = context.WithValue(ctx, labelContextKey, stored)
		}
		// add the stored tag context to the tag source
		tagMap = event.MergeTagMaps(tagMap, stored)
		return output(ctx, ev, tagMap)
	}
}
