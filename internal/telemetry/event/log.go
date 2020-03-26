// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"errors"
)

// Log sends a log event with the supplied tag list to the exporter.
func Log(ctx context.Context, tags ...Tag) {
	dispatch(ctx, makeEvent(LogType, sTags{}, tags))
}

// Log1 sends a label event to the exporter with the supplied tags.
func Log1(ctx context.Context, t1 Tag) context.Context {
	return dispatch(ctx, makeEvent(LogType, sTags{t1}, nil))
}

// Log2 sends a label event to the exporter with the supplied tags.
func Log2(ctx context.Context, t1, t2 Tag) context.Context {
	return dispatch(ctx, makeEvent(LogType, sTags{t1, t2}, nil))
}

// Log3 sends a label event to the exporter with the supplied tags.
func Log3(ctx context.Context, t1, t2, t3 Tag) context.Context {
	return dispatch(ctx, makeEvent(LogType, sTags{t1, t2, t3}, nil))
}

// Print takes a message and a tag list and combines them into a single event
// before delivering them to the exporter.
func Print(ctx context.Context, message string, tags ...Tag) {
	dispatch(ctx, makeEvent(LogType, sTags{Msg.Of(message)}, tags))
}

// Print1 takes a message and one tag delivers a log event to the exporter.
// It is a customized version of Print that is faster and does no allocation.
func Print1(ctx context.Context, message string, t1 Tag) {
	dispatch(ctx, makeEvent(LogType, sTags{Msg.Of(message), t1}, nil))
}

// Print2 takes a message and two tags and delivers a log event to the exporter.
// It is a customized version of Print that is faster and does no allocation.
func Print2(ctx context.Context, message string, t1 Tag, t2 Tag) {
	dispatch(ctx, makeEvent(LogType, sTags{Msg.Of(message), t1, t2}, nil))
}

// Error takes a message and a tag list and combines them into a single event
// before delivering them to the exporter. It captures the error in the
// delivered event.
func Error(ctx context.Context, message string, err error, tags ...Tag) {
	if err == nil {
		err = errors.New(message)
		message = ""
	}
	dispatch(ctx, makeEvent(LogType, sTags{Msg.Of(message), Err.Of(err)}, tags))
}
