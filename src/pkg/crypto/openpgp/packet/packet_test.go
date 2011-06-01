// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto/openpgp/error"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"testing"
)

func TestReadFull(t *testing.T) {
	var out [4]byte

	b := bytes.NewBufferString("foo")
	n, err := readFull(b, out[:3])
	if n != 3 || err != nil {
		t.Errorf("full read failed n:%d err:%s", n, err)
	}

	b = bytes.NewBufferString("foo")
	n, err = readFull(b, out[:4])
	if n != 3 || err != io.ErrUnexpectedEOF {
		t.Errorf("partial read failed n:%d err:%s", n, err)
	}

	b = bytes.NewBuffer(nil)
	n, err = readFull(b, out[:3])
	if n != 0 || err != io.ErrUnexpectedEOF {
		t.Errorf("empty read failed n:%d err:%s", n, err)
	}
}

func readerFromHex(s string) io.Reader {
	data, err := hex.DecodeString(s)
	if err != nil {
		panic("readerFromHex: bad input")
	}
	return bytes.NewBuffer(data)
}

var readLengthTests = []struct {
	hexInput  string
	length    int64
	isPartial bool
	err       os.Error
}{
	{"", 0, false, io.ErrUnexpectedEOF},
	{"1f", 31, false, nil},
	{"c0", 0, false, io.ErrUnexpectedEOF},
	{"c101", 256 + 1 + 192, false, nil},
	{"e0", 1, true, nil},
	{"e1", 2, true, nil},
	{"e2", 4, true, nil},
	{"ff", 0, false, io.ErrUnexpectedEOF},
	{"ff00", 0, false, io.ErrUnexpectedEOF},
	{"ff0000", 0, false, io.ErrUnexpectedEOF},
	{"ff000000", 0, false, io.ErrUnexpectedEOF},
	{"ff00000000", 0, false, nil},
	{"ff01020304", 16909060, false, nil},
}

func TestReadLength(t *testing.T) {
	for i, test := range readLengthTests {
		length, isPartial, err := readLength(readerFromHex(test.hexInput))
		if test.err != nil {
			if err != test.err {
				t.Errorf("%d: expected different error got:%s want:%s", i, err, test.err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%d: unexpected error: %s", i, err)
			continue
		}
		if length != test.length || isPartial != test.isPartial {
			t.Errorf("%d: bad result got:(%d,%t) want:(%d,%t)", i, length, isPartial, test.length, test.isPartial)
		}
	}
}

var partialLengthReaderTests = []struct {
	hexInput  string
	err       os.Error
	hexOutput string
}{
	{"e0", io.ErrUnexpectedEOF, ""},
	{"e001", io.ErrUnexpectedEOF, ""},
	{"e0010102", nil, "0102"},
	{"ff00000000", nil, ""},
	{"e10102e1030400", nil, "01020304"},
	{"e101", io.ErrUnexpectedEOF, ""},
}

func TestPartialLengthReader(t *testing.T) {
	for i, test := range partialLengthReaderTests {
		r := &partialLengthReader{readerFromHex(test.hexInput), 0, true}
		out, err := ioutil.ReadAll(r)
		if test.err != nil {
			if err != test.err {
				t.Errorf("%d: expected different error got:%s want:%s", i, err, test.err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%d: unexpected error: %s", i, err)
			continue
		}

		got := fmt.Sprintf("%x", out)
		if got != test.hexOutput {
			t.Errorf("%d: got:%s want:%s", i, test.hexOutput, got)
		}
	}
}

var readHeaderTests = []struct {
	hexInput        string
	structuralError bool
	unexpectedEOF   bool
	tag             int
	length          int64
	hexOutput       string
}{
	{"", false, false, 0, 0, ""},
	{"7f", true, false, 0, 0, ""},

	// Old format headers
	{"80", false, true, 0, 0, ""},
	{"8001", false, true, 0, 1, ""},
	{"800102", false, false, 0, 1, "02"},
	{"81000102", false, false, 0, 1, "02"},
	{"820000000102", false, false, 0, 1, "02"},
	{"860000000102", false, false, 1, 1, "02"},
	{"83010203", false, false, 0, -1, "010203"},

	// New format headers
	{"c0", false, true, 0, 0, ""},
	{"c000", false, false, 0, 0, ""},
	{"c00102", false, false, 0, 1, "02"},
	{"c0020203", false, false, 0, 2, "0203"},
	{"c00202", false, true, 0, 2, ""},
	{"c3020203", false, false, 3, 2, "0203"},
}

func TestReadHeader(t *testing.T) {
	for i, test := range readHeaderTests {
		tag, length, contents, err := readHeader(readerFromHex(test.hexInput))
		if test.structuralError {
			if _, ok := err.(error.StructuralError); ok {
				continue
			}
			t.Errorf("%d: expected StructuralError, got:%s", i, err)
			continue
		}
		if err != nil {
			if len(test.hexInput) == 0 && err == os.EOF {
				continue
			}
			if !test.unexpectedEOF || err != io.ErrUnexpectedEOF {
				t.Errorf("%d: unexpected error from readHeader: %s", i, err)
			}
			continue
		}
		if int(tag) != test.tag || length != test.length {
			t.Errorf("%d: got:(%d,%d) want:(%d,%d)", i, int(tag), length, test.tag, test.length)
			continue
		}

		body, err := ioutil.ReadAll(contents)
		if err != nil {
			if !test.unexpectedEOF || err != io.ErrUnexpectedEOF {
				t.Errorf("%d: unexpected error from contents: %s", i, err)
			}
			continue
		}
		if test.unexpectedEOF {
			t.Errorf("%d: expected ErrUnexpectedEOF from contents but got no error", i)
			continue
		}
		got := fmt.Sprintf("%x", body)
		if got != test.hexOutput {
			t.Errorf("%d: got:%s want:%s", i, got, test.hexOutput)
		}
	}
}

func TestSerializeHeader(t *testing.T) {
	tag := packetTypePublicKey
	lengths := []int{0, 1, 2, 64, 192, 193, 8000, 8384, 8385, 10000}

	for _, length := range lengths {
		buf := bytes.NewBuffer(nil)
		serializeHeader(buf, tag, length)
		tag2, length2, _, err := readHeader(buf)
		if err != nil {
			t.Errorf("length %d, err: %s", length, err)
		}
		if tag2 != tag {
			t.Errorf("length %d, tag incorrect (got %d, want %d)", length, tag2, tag)
		}
		if int(length2) != length {
			t.Errorf("length %d, length incorrect (got %d)", length, length2)
		}
	}
}

func TestPartialLengths(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	w := new(partialLengthWriter)
	w.w = noOpCloser{buf}

	const maxChunkSize = 64

	var b [maxChunkSize]byte
	var n uint8
	for l := 1; l <= maxChunkSize; l++ {
		for i := 0; i < l; i++ {
			b[i] = n
			n++
		}
		m, err := w.Write(b[:l])
		if m != l {
			t.Errorf("short write got: %d want: %d", m, l)
		}
		if err != nil {
			t.Errorf("error from write: %s", err)
		}
	}
	w.Close()

	want := (maxChunkSize * (maxChunkSize + 1)) / 2
	copyBuf := bytes.NewBuffer(nil)
	r := &partialLengthReader{buf, 0, true}
	m, err := io.Copy(copyBuf, r)
	if m != int64(want) {
		t.Errorf("short copy got: %d want: %d", m, want)
	}
	if err != nil {
		t.Errorf("error from copy: %s", err)
	}

	copyBytes := copyBuf.Bytes()
	for i := 0; i < want; i++ {
		if copyBytes[i] != uint8(i) {
			t.Errorf("bad pattern in copy at %d", i)
			break
		}
	}
}
