// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package version

import (
	"fmt"
	"io"

	"internal/trace/v2/event"
	"internal/trace/v2/event/go122"
)

// Version represents the version of a trace file.
type Version uint32

const (
	Go122   Version = 22
	Current         = Go122
)

var versions = map[Version][]event.Spec{
	Go122: go122.Specs(),
}

// Specs returns the set of event.Specs for this version.
func (v Version) Specs() []event.Spec {
	return versions[v]
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
