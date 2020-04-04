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
	invalidType   = eventType(iota)
	LogType       // an event that should be recorded in a log
	StartSpanType // the start of a span of time
	EndSpanType   // the end of a span of time
	LabelType     // some values that should be noted for later events
	DetachType    // an event that causes a context to detach
	RecordType    // a value that should be tracked
)

// sTags is used to hold a small number of tags inside an event whichout
// requiring a separate allocation.
// As tags are often on the stack, this avoids an allocation at all for
// the very common cases of simple events.
// The length needs to be large enough to cope with the majority of events
// but no so large as to cause undue stack pressure.
// A log message with two values will use 3 tags (one for each value and
// one for the message itself).
type sTags [3]Tag

// Event holds the information about an event of note that ocurred.
type Event struct {
	At time.Time

	typ     eventType
	static  sTags // inline storage for the first few tags
	dynamic []Tag // dynamically sized storage for remaining tags
}

// eventTagMap implements TagMap for a the tags of an Event.
type eventTagMap struct {
	event Event
}

func (ev Event) IsLog() bool       { return ev.typ == LogType }
func (ev Event) IsEndSpan() bool   { return ev.typ == EndSpanType }
func (ev Event) IsStartSpan() bool { return ev.typ == StartSpanType }
func (ev Event) IsLabel() bool     { return ev.typ == LabelType }
func (ev Event) IsDetach() bool    { return ev.typ == DetachType }
func (ev Event) IsRecord() bool    { return ev.typ == RecordType }

func (ev Event) Format(f fmt.State, r rune) {
	tagMap := TagMap(ev)
	if !ev.At.IsZero() {
		fmt.Fprint(f, ev.At.Format("2006/01/02 15:04:05 "))
	}
	msg := Msg.Get(tagMap)
	err := Err.Get(tagMap)
	fmt.Fprint(f, msg)
	if err != nil {
		if f.Flag('+') {
			fmt.Fprintf(f, ": %+v", err)
		} else {
			fmt.Fprintf(f, ": %v", err)
		}
	}
	for it := ev.Tags(); it.Valid(); it.Advance() {
		tag := it.Tag()
		// msg and err were both already printed above, so we skip them to avoid
		// double printing
		if tag.Key == Msg || tag.Key == Err {
			continue
		}
		fmt.Fprintf(f, "\n\t%v", tag)
	}
}

func (ev Event) Tags() TagIterator {
	return ChainTagIterators(
		NewTagIterator(ev.static[:]...),
		NewTagIterator(ev.dynamic...))
}

func (ev Event) Find(key interface{}) Tag {
	for _, tag := range ev.static {
		if tag.Key == key {
			return tag
		}
	}
	for _, tag := range ev.dynamic {
		if tag.Key == key {
			return tag
		}
	}
	return Tag{}
}

func makeEvent(typ eventType, static sTags, tags []Tag) Event {
	return Event{
		typ:     typ,
		static:  static,
		dynamic: tags,
	}
}
