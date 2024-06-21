// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package raw

import (
	"encoding/binary"
	"strconv"
	"strings"

	"internal/trace/event"
	"internal/trace/version"
)

// Event is a simple representation of a trace event.
//
// Note that this typically includes much more than just
// timestamped events, and it also represents parts of the
// trace format's framing. (But not interpreted.)
type Event struct {
	Version version.Version
	Ev      event.Type
	Args    []uint64
	Data    []byte
}

// String returns the canonical string representation of the event.
//
// This format is the same format that is parsed by the TextReader
// and emitted by the TextWriter.
func (e *Event) String() string {
	spec := e.Version.Specs()[e.Ev]

	var s strings.Builder
	s.WriteString(spec.Name)
	for i := range spec.Args {
		s.WriteString(" ")
		s.WriteString(spec.Args[i])
		s.WriteString("=")
		s.WriteString(strconv.FormatUint(e.Args[i], 10))
	}
	if spec.IsStack {
		frames := e.Args[len(spec.Args):]
		for i := 0; i < len(frames); i++ {
			if i%4 == 0 {
				s.WriteString("\n\t")
			} else {
				s.WriteString(" ")
			}
			s.WriteString(frameFields[i%4])
			s.WriteString("=")
			s.WriteString(strconv.FormatUint(frames[i], 10))
		}
	}
	if e.Data != nil {
		s.WriteString("\n\tdata=")
		s.WriteString(strconv.Quote(string(e.Data)))
	}
	return s.String()
}

// EncodedSize returns the canonical encoded size of an event.
func (e *Event) EncodedSize() int {
	size := 1
	var buf [binary.MaxVarintLen64]byte
	for _, arg := range e.Args {
		size += binary.PutUvarint(buf[:], arg)
	}
	spec := e.Version.Specs()[e.Ev]
	if spec.HasData {
		size += binary.PutUvarint(buf[:], uint64(len(e.Data)))
		size += len(e.Data)
	}
	return size
}
