// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code is a duplicate of ../chunked_test.go with these edits:
//	s/newChunked/NewChunked/g
//	s/package http/package httputil/
// Please make any changes in both files.

package httputil

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"testing"
)

func TestChunk(t *testing.T) {
	var b bytes.Buffer

	w := NewChunkedWriter(&b)
	const chunk1 = "hello, "
	const chunk2 = "world! 0123456789abcdef"
	w.Write([]byte(chunk1))
	w.Write([]byte(chunk2))
	w.Close()

	if g, e := b.String(), "7\r\nhello, \r\n17\r\nworld! 0123456789abcdef\r\n0\r\n"; g != e {
		t.Fatalf("chunk writer wrote %q; want %q", g, e)
	}

	r := NewChunkedReader(&b)
	data, err := ioutil.ReadAll(r)
	if err != nil {
		t.Logf(`data: "%s"`, data)
		t.Fatalf("ReadAll from reader: %v", err)
	}
	if g, e := string(data), chunk1+chunk2; g != e {
		t.Errorf("chunk reader read %q; want %q", g, e)
	}
}

func TestChunkReaderAllocs(t *testing.T) {
	var buf bytes.Buffer
	w := NewChunkedWriter(&buf)
	a, b, c := []byte("aaaaaa"), []byte("bbbbbbbbbbbb"), []byte("cccccccccccccccccccccccc")
	w.Write(a)
	w.Write(b)
	w.Write(c)
	w.Close()

	readBuf := make([]byte, len(a)+len(b)+len(c)+1)
	byter := bytes.NewReader(buf.Bytes())
	bufr := bufio.NewReader(byter)
	mallocs := testing.AllocsPerRun(10, func() {
		byter.Seek(0, 0)
		bufr.Reset(byter)
		r := NewChunkedReader(bufr)
		n, err := io.ReadFull(r, readBuf)
		if n != len(readBuf)-1 {
			t.Fatalf("read %d bytes; want %d", n, len(readBuf)-1)
		}
		if err != io.ErrUnexpectedEOF {
			t.Fatalf("read error = %v; want ErrUnexpectedEOF", err)
		}
	})
	if mallocs > 1.5 {
		t.Logf("mallocs = %v; want 1", mallocs)
	}
}

func TestParseHexUint(t *testing.T) {
	for i := uint64(0); i <= 1234; i++ {
		line := []byte(fmt.Sprintf("%x", i))
		got, err := parseHexUint(line)
		if err != nil {
			t.Fatalf("on %d: %v", i, err)
		}
		if got != i {
			t.Errorf("for input %q = %d; want %d", line, got, i)
		}
	}
	_, err := parseHexUint([]byte("bogus"))
	if err == nil {
		t.Error("expected error on bogus input")
	}
}
