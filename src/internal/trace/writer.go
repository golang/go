// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import "bytes"

// Writer is a test trace writer.
type Writer struct {
	bytes.Buffer
}

func NewWriter() *Writer {
	w := new(Writer)
	w.Write([]byte("go 1.9 trace\x00\x00\x00\x00"))
	return w
}

// Emit writes an event record to the trace.
// See Event types for valid types and required arguments.
func (w *Writer) Emit(typ byte, args ...uint64) {
	nargs := byte(len(args)) - 1
	if nargs > 3 {
		nargs = 3
	}
	buf := []byte{typ | nargs<<6}
	if nargs == 3 {
		buf = append(buf, 0)
	}
	for _, a := range args {
		buf = appendVarint(buf, a)
	}
	if nargs == 3 {
		buf[1] = byte(len(buf) - 2)
	}
	n, err := w.Write(buf)
	if n != len(buf) || err != nil {
		panic("failed to write")
	}
}

func appendVarint(buf []byte, v uint64) []byte {
	for ; v >= 0x80; v >>= 7 {
		buf = append(buf, 0x80|byte(v))
	}
	buf = append(buf, byte(v))
	return buf
}
