// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package version

import (
	"fmt"
	"io"

	"internal/trace/tracev2"
	"internal/trace/tracev2/event"
)

// Version represents the version of a trace file.
type Version uint32

const (
	Go111   Version = 11 // v1
	Go119   Version = 19 // v1
	Go121   Version = 21 // v1
	Go122   Version = 22 // v2
	Go123   Version = 23 // v2
	Current         = Go123
)

var versions = map[Version][]event.Spec{
	// Go 1.11–1.21 use a different parser and are only set here for the sake of
	// Version.Valid.
	Go111: nil,
	Go119: nil,
	Go121: nil,

	Go122: tracev2.Specs()[:tracev2.EvUserLog+1], // All events after are Go 1.23+.
	Go123: tracev2.Specs(),
}

// Specs returns the set of event.Specs for this version.
func (v Version) Specs() []event.Spec {
	return versions[v]
}

// EventName returns a string name of a wire format event
// for a particular trace version.
func (v Version) EventName(typ event.Type) string {
	if !v.Valid() {
		return "<invalid trace version>"
	}
	s := v.Specs()
	if len(s) == 0 {
		return "<v1 trace event type>"
	}
	if int(typ) < len(s) && s[typ].Name != "" {
		return s[typ].Name
	}
	return fmt.Sprintf("Invalid(%d)", typ)
}

func (v Version) Valid() bool {
	_, ok := versions[v]
	return ok
}

// headerFmt is the format of the header of all Go execution traces.
const headerFmt = "go 1.%d trace\x00\x00\x00"

// ReadHeader reads the version of the trace out of the trace file's
// header, whose prefix must be present in v.
func ReadHeader(r io.Reader) (Version, error) {
	var v Version
	_, err := fmt.Fscanf(r, headerFmt, &v)
	if err != nil {
		return v, fmt.Errorf("bad file format: not a Go execution trace?")
	}
	if !v.Valid() {
		return v, fmt.Errorf("unknown or unsupported trace version go 1.%d", v)
	}
	return v, nil
}

// WriteHeader writes a header for a trace version v to w.
func WriteHeader(w io.Writer, v Version) (int, error) {
	return fmt.Fprintf(w, headerFmt, v)
}
