// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"errors"
	"time"
)

// Log sends a log event with the supplied tag list to the exporter.
func Log(ctx context.Context, tags ...Tag) {
	ProcessEvent(ctx, Event{
		Type: LogType,
		At:   time.Now(),
		Tags: tags,
	})
}

// Print takes a message and a tag list and combines them into a single event
// before delivering them to the exporter.
func Print(ctx context.Context, message string, tags ...Tag) {
	ProcessEvent(ctx, Event{
		Type:    LogType,
		At:      time.Now(),
		Message: message,
		Tags:    tags,
	})
}

// Error takes a message and a tag list and combines them into a single event
// before delivering them to the exporter. It captures the error in the
// delivered event.
func Error(ctx context.Context, message string, err error, tags ...Tag) {
	if err == nil {
		err = errors.New(message)
		message = ""
	}
	ProcessEvent(ctx, Event{
		Type:    LogType,
		At:      time.Now(),
		Message: message,
		Error:   err,
		Tags:    tags,
	})
}
