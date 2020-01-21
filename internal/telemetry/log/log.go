// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package log is a context based logging package, designed to interact well
// with both the lsp protocol and the other telemetry packages.
package log

import (
	"context"
	"time"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/tag"
)

type Event telemetry.Event

// With sends a tag list to the installed loggers.
func With(ctx context.Context, tags ...telemetry.Tag) {
	export.Log(ctx, telemetry.Event{
		At:   time.Now(),
		Tags: tags,
	})
}

// Print takes a message and a tag list and combines them into a single tag
// list before delivering them to the loggers.
func Print(ctx context.Context, message string, tags ...tag.Tagger) {
	export.Log(ctx, telemetry.Event{
		At:      time.Now(),
		Message: message,
		Tags:    tag.Tags(ctx, tags...),
	})
}

// Error takes a message and a tag list and combines them into a single tag
// list before delivering them to the loggers. It captures the error in the
// delivered event.
func Error(ctx context.Context, message string, err error, tags ...tag.Tagger) {
	if err == nil {
		err = errorString(message)
		message = ""
	}
	export.Log(ctx, telemetry.Event{
		At:      time.Now(),
		Message: message,
		Error:   err,
		Tags:    tag.Tags(ctx, tags...),
	})
}

type errorString string

// Error allows errorString to conform to the error interface.
func (err errorString) Error() string { return string(err) }
