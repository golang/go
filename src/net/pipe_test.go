// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"io"
	"testing"
)

func checkPipeWrite(t *testing.T, w io.Writer, data []byte, c chan int) {
	n, err := w.Write(data)
	if err != nil {
		t.Error(err)
	}
	if n != len(data) {
		t.Errorf("short write: %d != %d", n, len(data))
	}
	c <- 0
}

func checkPipeRead(t *testing.T, r io.Reader, data []byte, wantErr error) {
	buf := make([]byte, len(data)+10)
	n, err := r.Read(buf)
	if err != wantErr {
		t.Error(err)
		return
	}
	if n != len(data) || !bytes.Equal(buf[0:n], data) {
		t.Errorf("bad read: got %q", buf[0:n])
		return
	}
}

// TestPipe tests a simple read/write/close sequence.
// Assumes that the underlying io.Pipe implementation
// is solid and we're just testing the net wrapping.
func TestPipe(t *testing.T) {
	c := make(chan int)
	cli, srv := Pipe()
	go checkPipeWrite(t, cli, []byte("hello, world"), c)
	checkPipeRead(t, srv, []byte("hello, world"), nil)
	<-c
	go checkPipeWrite(t, srv, []byte("line 2"), c)
	checkPipeRead(t, cli, []byte("line 2"), nil)
	<-c
	go checkPipeWrite(t, cli, []byte("a third line"), c)
	checkPipeRead(t, srv, []byte("a third line"), nil)
	<-c
	go srv.Close()
	checkPipeRead(t, cli, nil, io.EOF)
	cli.Close()
}
