// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xlog

import (
	"context"
	"fmt"
	"log"
)

// logger is a wrapper over a sink to provide a clean API over the core log
// function.
type logger struct {
	sink Sink
}

// Level indicates the severity of the logging message.
type Level int

const (
	ErrorLevel = Level(iota)
	InfoLevel
	DebugLevel
)

// Sink is the interface to something that consumes logging messages.
// This can be implemented and then registered with a context to control the
// destination or formatting of logging.
type Sink interface {
	Log(ctx context.Context, level Level, message string)
}

// StdSink is a Sink that writes to the standard log package.
type StdSink struct{}

// Errorf is intended for the logging of errors that we could not easily return
// to the client but that caused problems internally.
func (l logger) Errorf(ctx context.Context, format string, args ...interface{}) {
	l.sink.Log(ctx, ErrorLevel, fmt.Sprintf(format, args...))
}

// Infof is intended for logging of messages that may help the user understand
// the behavior or be useful in a bug report.
func (l logger) Infof(ctx context.Context, format string, args ...interface{}) {
	l.sink.Log(ctx, InfoLevel, fmt.Sprintf(format, args...))
}

// Debugf is intended to be used only while debugging.
func (l logger) Debugf(ctx context.Context, format string, args ...interface{}) {
	l.sink.Log(ctx, DebugLevel, fmt.Sprintf(format, args...))
}

// Log implements Sink for the StdSink.
// It writes the message using log.Print with a level based prefix.
func (StdSink) Log(ctx context.Context, level Level, message string) {
	switch level {
	case ErrorLevel:
		log.Print("Error: ", message)
	case InfoLevel:
		log.Print("Info: ", message)
	case DebugLevel:
		log.Print("Debug: ", message)
	}
}

type contextKeyType int

const contextKey = contextKeyType(0)

func With(ctx context.Context, sink Sink) context.Context {
	return context.WithValue(ctx, contextKey, sink)
}

func From(ctx context.Context) logger {
	return logger{sink: ctx.Value(contextKey).(Sink)}
}

func Errorf(ctx context.Context, format string, args ...interface{}) {
	From(ctx).Errorf(ctx, format, args...)
}

func Infof(ctx context.Context, format string, args ...interface{}) {
	From(ctx).Infof(ctx, format, args...)
}

func Debugf(ctx context.Context, format string, args ...interface{}) {
	From(ctx).Debugf(ctx, format, args...)
}
