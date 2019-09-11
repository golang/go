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

//TODO: Do we need to do something more efficient than just store tags
//TODO: directly on the context?

// Tagger is the interface to something that returns a Tag given a context.
// Both Tag itself and Key support this interface, allowing methods that can
// take either (and other implementations as well)
type Tagger interface {
	// Tag returns a Tag potentially using information from the Context.
	Tag(context.Context) telemetry.Tag
}

// With is roughly equivalent to context.WithValue except that it also notifies
// registered observers.
// Unlike WithValue, it takes a list of tags so that you can set many values
// at once if needed. Each call to With results in one invocation of each
// observer.
func With(ctx context.Context, tags ...telemetry.Tag) context.Context {
	at := time.Now()
	for _, t := range tags {
		ctx = context.WithValue(ctx, t.Key, t.Value)
	}
	export.Tag(ctx, at, tags)
	return ctx
}

// Get collects a set of values from the context and returns them as a tag list.
func Get(ctx context.Context, keys ...interface{}) telemetry.TagList {
	tags := make(telemetry.TagList, len(keys))
	for i, key := range keys {
		tags[i] = telemetry.Tag{Key: key, Value: ctx.Value(key)}
	}
	return tags
}

// Tags collects a list of tags for the taggers from the context.
func Tags(ctx context.Context, taggers ...Tagger) telemetry.TagList {
	tags := make(telemetry.TagList, len(taggers))
	for i, t := range taggers {
		tags[i] = t.Tag(ctx)
	}
	return tags
}
