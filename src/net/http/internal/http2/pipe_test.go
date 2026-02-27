// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func TestPipeClose(t *testing.T) {
	var p pipe
	p.b = new(bytes.Buffer)
	a := errors.New("a")
	b := errors.New("b")
	p.CloseWithError(a)
	p.CloseWithError(b)
	_, err := p.Read(make([]byte, 1))
	if err != a {
		t.Errorf("err = %v want %v", err, a)
	}
}

func TestPipeDoneChan(t *testing.T) {
	var p pipe
	done := p.Done()
	select {
	case <-done:
		t.Fatal("done too soon")
	default:
	}
	p.CloseWithError(io.EOF)
	select {
	case <-done:
	default:
		t.Fatal("should be done")
	}
}

func TestPipeDoneChan_ErrFirst(t *testing.T) {
	var p pipe
	p.CloseWithError(io.EOF)
	done := p.Done()
	select {
	case <-done:
	default:
		t.Fatal("should be done")
	}
}

func TestPipeDoneChan_Break(t *testing.T) {
	var p pipe
	done := p.Done()
	select {
	case <-done:
		t.Fatal("done too soon")
	default:
	}
	p.BreakWithError(io.EOF)
	select {
	case <-done:
	default:
		t.Fatal("should be done")
	}
}

func TestPipeDoneChan_Break_ErrFirst(t *testing.T) {
	var p pipe
	p.BreakWithError(io.EOF)
	done := p.Done()
	select {
	case <-done:
	default:
		t.Fatal("should be done")
	}
}

func TestPipeCloseWithError(t *testing.T) {
	p := &pipe{b: new(bytes.Buffer)}
	const body = "foo"
	io.WriteString(p, body)
	a := errors.New("test error")
	p.CloseWithError(a)
	all, err := io.ReadAll(p)
	if string(all) != body {
		t.Errorf("read bytes = %q; want %q", all, body)
	}
	if err != a {
		t.Logf("read error = %v, %v", err, a)
	}
	if p.Len() != 0 {
		t.Errorf("pipe should have 0 unread bytes")
	}
	// Read and Write should fail.
	if n, err := p.Write([]byte("abc")); err != errClosedPipeWrite || n != 0 {
		t.Errorf("Write(abc) after close\ngot %v, %v\nwant 0, %v", n, err, errClosedPipeWrite)
	}
	if n, err := p.Read(make([]byte, 1)); err == nil || n != 0 {
		t.Errorf("Read() after close\ngot %v, nil\nwant 0, %v", n, errClosedPipeWrite)
	}
	if p.Len() != 0 {
		t.Errorf("pipe should have 0 unread bytes")
	}
}

func TestPipeBreakWithError(t *testing.T) {
	p := &pipe{b: new(bytes.Buffer)}
	io.WriteString(p, "foo")
	a := errors.New("test err")
	p.BreakWithError(a)
	all, err := io.ReadAll(p)
	if string(all) != "" {
		t.Errorf("read bytes = %q; want empty string", all)
	}
	if err != a {
		t.Logf("read error = %v, %v", err, a)
	}
	if p.b != nil {
		t.Errorf("buffer should be nil after BreakWithError")
	}
	if p.Len() != 3 {
		t.Errorf("pipe should have 3 unread bytes")
	}
	// Write should fail.
	if n, err := p.Write([]byte("abc")); err != errClosedPipeWrite || n != 0 {
		t.Errorf("Write(abc) after break\ngot %v, %v\nwant 0, errClosedPipeWrite", n, err)
	}
	if p.b != nil {
		t.Errorf("buffer should be nil after Write")
	}
	if p.Len() != 3 {
		t.Errorf("pipe should have 6 unread bytes")
	}
	// Read should fail.
	if n, err := p.Read(make([]byte, 1)); err == nil || n != 0 {
		t.Errorf("Read() after close\ngot %v, nil\nwant 0, not nil", n)
	}
}
