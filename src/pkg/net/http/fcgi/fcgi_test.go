// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fcgi

import (
	"bytes"
	"io"
	"testing"
)

var sizeTests = []struct {
	size  uint32
	bytes []byte
}{
	{0, []byte{0x00}},
	{127, []byte{0x7F}},
	{128, []byte{0x80, 0x00, 0x00, 0x80}},
	{1000, []byte{0x80, 0x00, 0x03, 0xE8}},
	{33554431, []byte{0x81, 0xFF, 0xFF, 0xFF}},
}

func TestSize(t *testing.T) {
	b := make([]byte, 4)
	for i, test := range sizeTests {
		n := encodeSize(b, test.size)
		if !bytes.Equal(b[:n], test.bytes) {
			t.Errorf("%d expected %x, encoded %x", i, test.bytes, b)
		}
		size, n := readSize(test.bytes)
		if size != test.size {
			t.Errorf("%d expected %d, read %d", i, test.size, size)
		}
		if len(test.bytes) != n {
			t.Errorf("%d did not consume all the bytes", i)
		}
	}
}

var streamTests = []struct {
	desc    string
	recType uint8
	reqId   uint16
	content []byte
	raw     []byte
}{
	{"single record", typeStdout, 1, nil,
		[]byte{1, typeStdout, 0, 1, 0, 0, 0, 0},
	},
	// this data will have to be split into two records
	{"two records", typeStdin, 300, make([]byte, 66000),
		bytes.Join([][]byte{
			// header for the first record
			{1, typeStdin, 0x01, 0x2C, 0xFF, 0xFF, 1, 0},
			make([]byte, 65536),
			// header for the second
			{1, typeStdin, 0x01, 0x2C, 0x01, 0xD1, 7, 0},
			make([]byte, 472),
			// header for the empty record
			{1, typeStdin, 0x01, 0x2C, 0, 0, 0, 0},
		},
			nil),
	},
}

type nilCloser struct {
	io.ReadWriter
}

func (c *nilCloser) Close() error { return nil }

func TestStreams(t *testing.T) {
	var rec record
outer:
	for _, test := range streamTests {
		buf := bytes.NewBuffer(test.raw)
		var content []byte
		for buf.Len() > 0 {
			if err := rec.read(buf); err != nil {
				t.Errorf("%s: error reading record: %v", test.desc, err)
				continue outer
			}
			content = append(content, rec.content()...)
		}
		if rec.h.Type != test.recType {
			t.Errorf("%s: got type %d expected %d", test.desc, rec.h.Type, test.recType)
			continue
		}
		if rec.h.Id != test.reqId {
			t.Errorf("%s: got request ID %d expected %d", test.desc, rec.h.Id, test.reqId)
			continue
		}
		if !bytes.Equal(content, test.content) {
			t.Errorf("%s: read wrong content", test.desc)
			continue
		}
		buf.Reset()
		c := newConn(&nilCloser{buf})
		w := newWriter(c, test.recType, test.reqId)
		if _, err := w.Write(test.content); err != nil {
			t.Errorf("%s: error writing record: %v", test.desc, err)
			continue
		}
		if err := w.Close(); err != nil {
			t.Errorf("%s: error closing stream: %v", test.desc, err)
			continue
		}
		if !bytes.Equal(buf.Bytes(), test.raw) {
			t.Errorf("%s: wrote wrong content", test.desc)
		}
	}
}
