// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package log provides helper methods for exporting log events to the
// internal/event package.
package log

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/event/tag"
)

// Level parameterizes log severity.
type Level int

const (
	_ Level = iota
	Error
	Warning
	Info
	Debug
	Trace
)

// Log exports a log event labeled with level l.
func (l Level) Log(ctx context.Context, msg string) {
	event.Log(ctx, msg, tag.Level.Of(int(l)))
}

// Logf formats and exports a log event labeled with level l.
func (l Level) Logf(ctx context.Context, format string, args ...interface{}) {
	l.Log(ctx, fmt.Sprintf(format, args...))
}

// LabeledLevel extracts the labeled log l
func LabeledLevel(lm label.Map) Level {
	return Level(tag.Level.Get(lm))
}
