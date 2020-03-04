// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag provides support for telemetry tagging.
// This package is a thin shim over contexts with the main addition being the
// the ability to observe when contexts get tagged with new values.
package tag

import (
	"context"
	"time"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export"
)

// With delivers the tag list to the telemetry exporter.
func With(ctx context.Context, tags ...telemetry.Tag) context.Context {
	return export.ProcessEvent(ctx, telemetry.Event{
		Type: telemetry.EventTag,
		At:   time.Now(),
		Tags: tags,
	})
}

// Get collects a set of values from the context and returns them as a tag list.
func Get(ctx context.Context, keys ...interface{}) telemetry.TagList {
	tags := make(telemetry.TagList, len(keys))
	for i, key := range keys {
		tags[i] = telemetry.Tag{Key: key, Value: ctx.Value(key)}
	}
	return tags
}
