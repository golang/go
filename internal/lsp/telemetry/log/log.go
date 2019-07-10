// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package log is a context based logging package, designed to interact well
// with both the lsp protocol and the other telemetry packages.
package log

import (
	"context"
	"fmt"
	"os"
	"time"

	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/lsp/telemetry/worker"
)

const (
	// The well known tag keys for the logging system.
	MessageTag = tag.Key("message")
	ErrorTag   = tag.Key("error")
)

// Logger is a function that handles logging messages.
// Loggers are registered at start up, and may use information in the context
// to decide what to do with a given log message.
type Logger func(ctx context.Context, at time.Time, tags tag.List) bool

// With sends a tag list to the installed loggers.
func With(ctx context.Context, tags ...tag.Tag) {
	at := time.Now()
	worker.Do(func() {
		deliver(ctx, at, tags)
	})
}

// Print takes a message and a tag list and combines them into a single tag
// list before delivering them to the loggers.
func Print(ctx context.Context, message string, tags ...tag.Tagger) {
	at := time.Now()
	worker.Do(func() {
		tags := append(tag.Tags(ctx, tags...), MessageTag.Of(message))
		deliver(ctx, at, tags)
	})
}

type errorString string

// Error allows errorString to conform to the error interface.
func (err errorString) Error() string { return string(err) }

// Print takes a message and a tag list and combines them into a single tag
// list before delivering them to the loggers.
func Error(ctx context.Context, message string, err error, tags ...tag.Tagger) {
	at := time.Now()
	worker.Do(func() {
		if err == nil {
			err = errorString(message)
			message = ""
		}
		tags := append(tag.Tags(ctx, tags...), MessageTag.Of(message), ErrorTag.Of(err))
		deliver(ctx, at, tags)
	})
}

func deliver(ctx context.Context, at time.Time, tags tag.List) {
	delivered := false
	for _, logger := range loggers {
		if logger(ctx, at, tags) {
			delivered = true
		}
	}
	if !delivered {
		// no logger processed the message, so we log to stderr just in case
		Stderr(ctx, at, tags)
	}
}

var loggers = []Logger{}

func AddLogger(logger Logger) {
	worker.Do(func() {
		loggers = append(loggers, logger)
	})
}

// Stderr is a logger that logs to stderr in the standard format.
func Stderr(ctx context.Context, at time.Time, tags tag.List) bool {
	fmt.Fprintf(os.Stderr, "%v\n", ToEntry(ctx, at, tags))
	return true
}
