// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"io"
	"os"
	"testing"
)

func checkWrite(t *testing.T, w io.Writer, data []byte, c chan int) {
	n, err := w.Write(data)
	if err != nil {
		t.Errorf("write: %v", err)
	}
	if n != len(data) {
		t.Errorf("short write: %d != %d", n, len(data))
	}
	c <- 0
}

func checkRead(t *testing.T, r io.Reader, data []byte, wantErr os.Error) {
	buf := make([]byte, len(data)+10)
	n, err := r.Read(buf)
	if err != wantErr {
		t.Errorf("read: %v", err)
		return
	}
	if n != len(data) || !bytes.Equal(buf[0:n], data) {
		t.Errorf("bad read: got %q", buf[0:n])
		return
	}
}

// Test a simple read/write/close sequence.
// Assumes that the underlying io.Pipe implementation
// is solid and we're just testing the net wrapping.

func TestPipe(t *testing.T) {
	c := make(chan int)
	cli, srv := Pipe()
	go checkWrite(t, cli, []byte("hello, world"), c)
	checkRead(t, srv, []byte("hello, world"), nil)
	<-c
	go checkWrite(t, srv, []byte("line 2"), c)
	checkRead(t, cli, []byte("line 2"), nil)
	<-c
	go checkWrite(t, cli, []byte("a third line"), c)
	checkRead(t, srv, []byte("a third line"), nil)
	<-c
	go srv.Close()
	checkRead(t, cli, nil, os.EOF)
	cli.Close()
}
