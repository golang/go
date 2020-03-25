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
	RecordType
)

type Event struct {
	typ     eventType
	At      time.Time
	Message string
	Error   error

	tags []Tag
}

func (ev Event) IsLog() bool       { return ev.typ == LogType }
func (ev Event) IsEndSpan() bool   { return ev.typ == EndSpanType }
func (ev Event) IsStartSpan() bool { return ev.typ == StartSpanType }
func (ev Event) IsLabel() bool     { return ev.typ == LabelType }
func (ev Event) IsDetach() bool    { return ev.typ == DetachType }
func (ev Event) IsRecord() bool    { return ev.typ == RecordType }

func (ev Event) Format(f fmt.State, r rune) {
	if !ev.At.IsZero() {
		fmt.Fprint(f, ev.At.Format("2006/01/02 15:04:05 "))
	}
	fmt.Fprint(f, ev.Message)
	if ev.Error != nil {
		if f.Flag('+') {
			fmt.Fprintf(f, ": %+v", ev.Error)
		} else {
			fmt.Fprintf(f, ": %v", ev.Error)
		}
	}
	for it := ev.Tags(); it.Valid(); it.Advance() {
		tag := it.Tag()
		fmt.Fprintf(f, "\n\t%v", tag)
	}
}

func (ev Event) Tags() TagIterator {
	if len(ev.tags) == 0 {
		return TagIterator{}
	}
	return NewTagIterator(ev.tags...)
}

func (ev Event) Map() TagMap {
	return NewTagMap(ev.tags...)
}
