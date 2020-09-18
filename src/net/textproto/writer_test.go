// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textproto

import (
	"bufio"
	"bytes"
	"testing"
)

func TestPrintfLine(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(bufio.NewWriter(&buf))
	err := w.PrintfLine("foo %d", 123)
	if s := buf.String(); s != "foo 123\r\n" || err != nil {
		t.Fatalf("s=%q; err=%s", s, err)
	}
}

func TestDotWriter(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(bufio.NewWriter(&buf))
	d := w.DotWriter()
	n, err := d.Write([]byte("abc\n.def\n..ghi\n.jkl\n."))
	if n != 21 || err != nil {
		t.Fatalf("Write: %d, %s", n, err)
	}
	d.Close()
	want := "abc\r\n..def\r\n...ghi\r\n..jkl\r\n..\r\n.\r\n"
	if s := buf.String(); s != want {
		t.Fatalf("wrote %q", s)
	}
}

func TestDotWriterCloseEmptyWrite(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(bufio.NewWriter(&buf))
	d := w.DotWriter()
	n, err := d.Write([]byte{})
	if n != 0 || err != nil {
		t.Fatalf("Write: %d, %s", n, err)
	}
	d.Close()
	want := "\r\n.\r\n"
	if s := buf.String(); s != want {
		t.Fatalf("wrote %q; want %q", s, want)
	}
}

func TestDotWriterCloseNoWrite(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(bufio.NewWriter(&buf))
	d := w.DotWriter()
	d.Close()
	want := "\r\n.\r\n"
	if s := buf.String(); s != want {
		t.Fatalf("wrote %q; want %q", s, want)
	}
}
