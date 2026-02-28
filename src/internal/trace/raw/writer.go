// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package raw

import (
	"encoding/binary"
	"fmt"
	"io"

	"internal/trace/tracev2"
	"internal/trace/version"
)

// Writer emits the wire format of a trace.
//
// It may not produce a byte-for-byte compatible trace from what is
// produced by the runtime, because it may be missing extra padding
// in the LEB128 encoding that the runtime adds but isn't necessary
// when you know the data up-front.
type Writer struct {
	w     io.Writer
	buf   []byte
	v     version.Version
	specs []tracev2.EventSpec
}

// NewWriter creates a new byte format writer.
func NewWriter(w io.Writer, v version.Version) (*Writer, error) {
	_, err := version.WriteHeader(w, v)
	return &Writer{w: w, v: v, specs: v.Specs()}, err
}

// WriteEvent writes a single event to the trace wire format stream.
func (w *Writer) WriteEvent(e Event) error {
	// Check version.
	if e.Version != w.v {
		return fmt.Errorf("mismatched version between writer (go 1.%d) and event (go 1.%d)", w.v, e.Version)
	}

	// Write event header byte.
	w.buf = append(w.buf, uint8(e.Ev))

	// Write out all arguments.
	spec := w.specs[e.Ev]
	for _, arg := range e.Args[:len(spec.Args)] {
		w.buf = binary.AppendUvarint(w.buf, arg)
	}
	if spec.IsStack {
		frameArgs := e.Args[len(spec.Args):]
		for i := 0; i < len(frameArgs); i++ {
			w.buf = binary.AppendUvarint(w.buf, frameArgs[i])
		}
	}

	// Write out the length of the data.
	if spec.HasData {
		w.buf = binary.AppendUvarint(w.buf, uint64(len(e.Data)))
	}

	// Write out varint events.
	_, err := w.w.Write(w.buf)
	w.buf = w.buf[:0]
	if err != nil {
		return err
	}

	// Write out data.
	if spec.HasData {
		_, err := w.w.Write(e.Data)
		return err
	}
	return nil
}
