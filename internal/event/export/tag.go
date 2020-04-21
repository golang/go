// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/label"
)

// Labels builds an exporter that manipulates the context using the event.
// If the event is type IsLabel or IsStartSpan then it returns a context updated
// with label values from the event.
// For all other event types the event labels will be updated with values from the
// context if they are missing.
func Labels(output event.Exporter) event.Exporter {
	return func(ctx context.Context, ev core.Event, lm label.Map) context.Context {
		stored, _ := ctx.Value(labelContextKey).(label.Map)
		if event.IsLabel(ev) || event.IsStart(ev) {
			// update the label map stored in the context
			fromEvent := label.Map(ev)
			if stored == nil {
				stored = fromEvent
			} else {
				stored = label.MergeMaps(fromEvent, stored)
			}
			ctx = context.WithValue(ctx, labelContextKey, stored)
		}
		// add the stored label context to the label map
		lm = label.MergeMaps(lm, stored)
		return output(ctx, ev, lm)
	}
}
