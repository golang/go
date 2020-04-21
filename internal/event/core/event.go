// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package core provides support for event based telemetry.
package core

import (
	"fmt"
	"time"

	"golang.org/x/tools/internal/event/label"
)

// Event holds the information about an event of note that ocurred.
type Event struct {
	at time.Time

	// As events are often on the stack, storing the first few labels directly
	// in the event can avoid an allocation at all for the very common cases of
	// simple events.
	// The length needs to be large enough to cope with the majority of events
	// but no so large as to cause undue stack pressure.
	// A log message with two values will use 3 labels (one for each value and
	// one for the message itself).

	static  [3]label.Label // inline storage for the first few labels
	dynamic []label.Label  // dynamically sized storage for remaining labels
}

// eventLabelMap implements label.Map for a the labels of an Event.
type eventLabelMap struct {
	event Event
}

func (ev Event) At() time.Time { return ev.at }

func (ev Event) Format(f fmt.State, r rune) {
	if !ev.at.IsZero() {
		fmt.Fprint(f, ev.at.Format("2006/01/02 15:04:05 "))
	}
	for index := 0; ev.Valid(index); index++ {
		if l := ev.Label(index); l.Valid() {
			fmt.Fprintf(f, "\n\t%v", l)
		}
	}
}

func (ev Event) Valid(index int) bool {
	return index >= 0 && index < len(ev.static)+len(ev.dynamic)
}

func (ev Event) Label(index int) label.Label {
	if index < len(ev.static) {
		return ev.static[index]
	}
	return ev.dynamic[index-len(ev.static)]
}

func (ev Event) Find(key label.Key) label.Label {
	for _, l := range ev.static {
		if l.Key() == key {
			return l
		}
	}
	for _, l := range ev.dynamic {
		if l.Key() == key {
			return l
		}
	}
	return label.Label{}
}

func MakeEvent(static [3]label.Label, labels []label.Label) Event {
	return Event{
		static:  static,
		dynamic: labels,
	}
}

// CloneEvent event returns a copy of the event with the time adjusted to at.
func CloneEvent(ev Event, at time.Time) Event {
	ev.at = at
	return ev
}
