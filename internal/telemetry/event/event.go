// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package event provides support for event based telemetry.
package event

import (
	"fmt"
	"time"
)

type eventType uint8

const (
	LogType = eventType(iota)
	StartSpanType
	EndSpanType
	LabelType
	DetachType
)

type Event struct {
	Type    eventType
	At      time.Time
	Message string
	Error   error
	Tags    TagList
}

func (e Event) IsLog() bool       { return e.Type == LogType }
func (e Event) IsEndSpan() bool   { return e.Type == EndSpanType }
func (e Event) IsStartSpan() bool { return e.Type == StartSpanType }
func (e Event) IsTag() bool       { return e.Type == LabelType }
func (e Event) IsDetach() bool    { return e.Type == DetachType }

func (e Event) Format(f fmt.State, r rune) {
	if !e.At.IsZero() {
		fmt.Fprint(f, e.At.Format("2006/01/02 15:04:05 "))
	}
	fmt.Fprint(f, e.Message)
	if e.Error != nil {
		if f.Flag('+') {
			fmt.Fprintf(f, ": %+v", e.Error)
		} else {
			fmt.Fprintf(f, ": %v", e.Error)
		}
	}
	for _, tag := range e.Tags {
		fmt.Fprintf(f, "\n\t%s = %v", tag.Key.Name, tag.Value)
	}
}
