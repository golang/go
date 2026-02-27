// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"
)

func fmtDataChunk(chunk []byte) string {
	out := ""
	var last byte
	var count int
	for _, c := range chunk {
		if c != last {
			if count > 0 {
				out += fmt.Sprintf(" x %d ", count)
				count = 0
			}
			out += string([]byte{c})
			last = c
		}
		count++
	}
	if count > 0 {
		out += fmt.Sprintf(" x %d", count)
	}
	return out
}

func fmtDataChunks(chunks [][]byte) string {
	var out string
	for _, chunk := range chunks {
		out += fmt.Sprintf("{%q}", fmtDataChunk(chunk))
	}
	return out
}

func testDataBuffer(t *testing.T, wantBytes []byte, setup func(t *testing.T) *dataBuffer) {
	// Run setup, then read the remaining bytes from the dataBuffer and check
	// that they match wantBytes. We use different read sizes to check corner
	// cases in Read.
	for _, readSize := range []int{1, 2, 1 * 1024, 32 * 1024} {
		t.Run(fmt.Sprintf("ReadSize=%d", readSize), func(t *testing.T) {
			b := setup(t)
			buf := make([]byte, readSize)
			var gotRead bytes.Buffer
			for {
				n, err := b.Read(buf)
				gotRead.Write(buf[:n])
				if err == errReadEmpty {
					break
				}
				if err != nil {
					t.Fatalf("error after %v bytes: %v", gotRead.Len(), err)
				}
			}
			if got, want := gotRead.Bytes(), wantBytes; !bytes.Equal(got, want) {
				t.Errorf("FinalRead=%q, want %q", fmtDataChunk(got), fmtDataChunk(want))
			}
		})
	}
}

func TestDataBufferAllocation(t *testing.T) {
	writes := [][]byte{
		bytes.Repeat([]byte("a"), 1*1024-1),
		[]byte("a"),
		bytes.Repeat([]byte("b"), 4*1024-1),
		[]byte("b"),
		bytes.Repeat([]byte("c"), 8*1024-1),
		[]byte("c"),
		bytes.Repeat([]byte("d"), 16*1024-1),
		[]byte("d"),
		bytes.Repeat([]byte("e"), 32*1024),
	}
	var wantRead bytes.Buffer
	for _, p := range writes {
		wantRead.Write(p)
	}

	testDataBuffer(t, wantRead.Bytes(), func(t *testing.T) *dataBuffer {
		b := &dataBuffer{}
		for _, p := range writes {
			if n, err := b.Write(p); n != len(p) || err != nil {
				t.Fatalf("Write(%q x %d)=%v,%v want %v,nil", p[:1], len(p), n, err, len(p))
			}
		}
		want := [][]byte{
			bytes.Repeat([]byte("a"), 1*1024),
			bytes.Repeat([]byte("b"), 4*1024),
			bytes.Repeat([]byte("c"), 8*1024),
			bytes.Repeat([]byte("d"), 16*1024),
			bytes.Repeat([]byte("e"), 16*1024),
			bytes.Repeat([]byte("e"), 16*1024),
		}
		if !reflect.DeepEqual(b.chunks, want) {
			t.Errorf("dataBuffer.chunks\ngot:  %s\nwant: %s", fmtDataChunks(b.chunks), fmtDataChunks(want))
		}
		return b
	})
}

func TestDataBufferAllocationWithExpected(t *testing.T) {
	writes := [][]byte{
		bytes.Repeat([]byte("a"), 1*1024), // allocates 16KB
		bytes.Repeat([]byte("b"), 14*1024),
		bytes.Repeat([]byte("c"), 15*1024), // allocates 16KB more
		bytes.Repeat([]byte("d"), 2*1024),
		bytes.Repeat([]byte("e"), 1*1024), // overflows 32KB expectation, allocates just 1KB
	}
	var wantRead bytes.Buffer
	for _, p := range writes {
		wantRead.Write(p)
	}

	testDataBuffer(t, wantRead.Bytes(), func(t *testing.T) *dataBuffer {
		b := &dataBuffer{expected: 32 * 1024}
		for _, p := range writes {
			if n, err := b.Write(p); n != len(p) || err != nil {
				t.Fatalf("Write(%q x %d)=%v,%v want %v,nil", p[:1], len(p), n, err, len(p))
			}
		}
		want := [][]byte{
			append(bytes.Repeat([]byte("a"), 1*1024), append(bytes.Repeat([]byte("b"), 14*1024), bytes.Repeat([]byte("c"), 1*1024)...)...),
			append(bytes.Repeat([]byte("c"), 14*1024), bytes.Repeat([]byte("d"), 2*1024)...),
			bytes.Repeat([]byte("e"), 1*1024),
		}
		if !reflect.DeepEqual(b.chunks, want) {
			t.Errorf("dataBuffer.chunks\ngot:  %s\nwant: %s", fmtDataChunks(b.chunks), fmtDataChunks(want))
		}
		return b
	})
}

func TestDataBufferWriteAfterPartialRead(t *testing.T) {
	testDataBuffer(t, []byte("cdxyz"), func(t *testing.T) *dataBuffer {
		b := &dataBuffer{}
		if n, err := b.Write([]byte("abcd")); n != 4 || err != nil {
			t.Fatalf("Write(\"abcd\")=%v,%v want 4,nil", n, err)
		}
		p := make([]byte, 2)
		if n, err := b.Read(p); n != 2 || err != nil || !bytes.Equal(p, []byte("ab")) {
			t.Fatalf("Read()=%q,%v,%v want \"ab\",2,nil", p, n, err)
		}
		if n, err := b.Write([]byte("xyz")); n != 3 || err != nil {
			t.Fatalf("Write(\"xyz\")=%v,%v want 3,nil", n, err)
		}
		return b
	})
}
