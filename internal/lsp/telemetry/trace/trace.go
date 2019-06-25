// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag adds support for telemetry tracins.
package trace

import (
	"context"
)

type Span interface {
	AddAttributes(attributes ...Attribute)
	AddMessageReceiveEvent(messageID, uncompressedByteSize, compressedByteSize int64)
	AddMessageSendEvent(messageID, uncompressedByteSize, compressedByteSize int64)
	Annotate(attributes []Attribute, str string)
	Annotatef(attributes []Attribute, format string, a ...interface{})
	End()
	IsRecordingEvents() bool
	SetName(name string)
	SetStatus(status Status)
}

type Attribute interface{}

type Status struct {
	Code    int32
	Message string
}

type nullSpan struct{}

func (nullSpan) AddAttributes(attributes ...Attribute)                                            {}
func (nullSpan) AddMessageReceiveEvent(messageID, uncompressedByteSize, compressedByteSize int64) {}
func (nullSpan) AddMessageSendEvent(messageID, uncompressedByteSize, compressedByteSize int64)    {}
func (nullSpan) Annotate(attributes []Attribute, str string)                                      {}
func (nullSpan) Annotatef(attributes []Attribute, format string, a ...interface{})                {}
func (nullSpan) End()                                                                             {}
func (nullSpan) IsRecordingEvents() bool                                                          { return false }
func (nullSpan) SetName(name string)                                                              {}
func (nullSpan) SetStatus(status Status)                                                          {}

var (
	FromContext = func(ctx context.Context) Span { return nullSpan{} }
	NewContext  = func(ctx context.Context, span Span) context.Context { return ctx }
	StartSpan   = func(ctx context.Context, name string, options ...interface{}) (context.Context, Span) {
		return ctx, nullSpan{}
	}
	BoolAttribute    = func(key string, value bool) Attribute { return nil }
	Float64Attribute = func(key string, value float64) Attribute { return nil }
	Int64Attribute   = func(key string, value int64) Attribute { return nil }
	StringAttribute  = func(key string, value string) Attribute { return nil }
	WithSpanKind     = func(spanKind int) interface{} { return nil }
)

const (
	SpanKindUnspecified = iota
	SpanKindServer
	SpanKindClient
)
