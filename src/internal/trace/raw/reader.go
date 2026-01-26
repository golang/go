// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package raw

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"

	"internal/trace/tracev2"
	"internal/trace/version"
)

// Reader parses trace bytes with only very basic validation
// into an event stream.
type Reader struct {
	r     *bufio.Reader
	v     version.Version
	specs []tracev2.EventSpec
}

// NewReader creates a new reader for the trace wire format.
func NewReader(r io.Reader) (*Reader, error) {
	br := bufio.NewReader(r)
	v, err := version.ReadHeader(br)
	if err != nil {
		return nil, err
	}
	return &Reader{r: br, v: v, specs: v.Specs()}, nil
}

// Version returns the version of the trace that we're reading.
func (r *Reader) Version() version.Version {
	return r.v
}

// ReadEvent reads and returns the next trace event in the byte stream.
func (r *Reader) ReadEvent() (Event, error) {
	evb, err := r.r.ReadByte()
	if err == io.EOF {
		return Event{}, io.EOF
	}
	if err != nil {
		return Event{}, err
	}
	if int(evb) >= len(r.specs) || evb == 0 {
		return Event{}, fmt.Errorf("invalid event type: %d", evb)
	}
	ev := tracev2.EventType(evb)
	spec := r.specs[ev]
	args, err := r.readArgs(len(spec.Args))
	if err != nil {
		return Event{}, err
	}
	if spec.IsStack {
		len := int(args[1])
		for i := 0; i < len; i++ {
			// Each stack frame has four args: pc, func ID, file ID, line number.
			frame, err := r.readArgs(4)
			if err != nil {
				return Event{}, err
			}
			args = append(args, frame...)
		}
	}
	var data []byte
	if spec.HasData {
		data, err = r.readData()
		if err != nil {
			return Event{}, err
		}
	}
	return Event{
		Version: r.v,
		Ev:      ev,
		Args:    args,
		Data:    data,
	}, nil
}

func (r *Reader) readArgs(n int) ([]uint64, error) {
	var args []uint64
	for i := 0; i < n; i++ {
		val, err := binary.ReadUvarint(r.r)
		if err != nil {
			return nil, err
		}
		args = append(args, val)
	}
	return args, nil
}

func (r *Reader) readData() ([]byte, error) {
	len, err := binary.ReadUvarint(r.r)
	if err != nil {
		return nil, err
	}
	var data []byte
	for i := 0; i < int(len); i++ {
		b, err := r.r.ReadByte()
		if err != nil {
			return nil, err
		}
		data = append(data, b)
	}
	return data, nil
}
