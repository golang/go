// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"time"
)

// Label sends a label event to the exporter with the supplied tags.
func Label(ctx context.Context, tags ...Tag) context.Context {
	ctx, _ = ProcessEvent(ctx, Event{
		Type: LabelType,
		At:   time.Now(),
		Tags: newTagSet(tags),
	})
	return ctx
}

// Query sends a query event to the exporter with the supplied keys.
// The returned tags will have up to date values if the exporter supports it.
func Query(ctx context.Context, keys ...Key) TagSet {
	tags := make([]Tag, len(keys))
	for i, k := range keys {
		tags[i] = k.OfValue(nil)
	}
	_, ev := ProcessEvent(ctx, Event{
		Type: QueryType,
		Tags: newTagSet(tags),
	})
	return ev.Tags
}
