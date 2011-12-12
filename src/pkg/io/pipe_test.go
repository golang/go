// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"fmt"
	. "io"
	"testing"
	"time"
)

func checkWrite(t *testing.T, w Writer, data []byte, c chan int) {
	n, err := w.Write(data)
	if err != nil {
		t.Errorf("write: %v", err)
	}
	if n != len(data) {
		t.Errorf("short write: %d != %d", n, len(data))
	}
	c <- 0
}

// Test a single read/write pair.
func TestPipe1(t *testing.T) {
	c := make(chan int)
	r, w := Pipe()
	var buf = make([]byte, 64)
	go checkWrite(t, w, []byte("hello, world"), c)
	n, err := r.Read(buf)
	if err != nil {
		t.Errorf("read: %v", err)
	} else if n != 12 || string(buf[0:12]) != "hello, world" {
		t.Errorf("bad read: got %q", buf[0:n])
	}
	<-c
	r.Close()
	w.Close()
}

func reader(t *testing.T, r Reader, c chan int) {
	var buf = make([]byte, 64)
	for {
		n, err := r.Read(buf)
		if err == EOF {
			c <- 0
			break
		}
		if err != nil {
			t.Errorf("read: %v", err)
		}
		c <- n
	}
}

// Test a sequence of read/write pairs.
func TestPipe2(t *testing.T) {
	c := make(chan int)
	r, w := Pipe()
	go reader(t, r, c)
	var buf = make([]byte, 64)
	for i := 0; i < 5; i++ {
		p := buf[0 : 5+i*10]
		n, err := w.Write(p)
		if n != len(p) {
			t.Errorf("wrote %d, got %d", len(p), n)
		}
		if err != nil {
			t.Errorf("write: %v", err)
		}
		nn := <-c
		if nn != n {
			t.Errorf("wrote %d, read got %d", n, nn)
		}
	}
	w.Close()
	nn := <-c
	if nn != 0 {
		t.Errorf("final read got %d", nn)
	}
}

type pipeReturn struct {
	n   int
	err error
}

// Test a large write that requires multiple reads to satisfy.
func writer(w WriteCloser, buf []byte, c chan pipeReturn) {
	n, err := w.Write(buf)
	w.Close()
	c <- pipeReturn{n, err}
}

func TestPipe3(t *testing.T) {
	c := make(chan pipeReturn)
	r, w := Pipe()
	var wdat = make([]byte, 128)
	for i := 0; i < len(wdat); i++ {
		wdat[i] = byte(i)
	}
	go writer(w, wdat, c)
	var rdat = make([]byte, 1024)
	tot := 0
	for n := 1; n <= 256; n *= 2 {
		nn, err := r.Read(rdat[tot : tot+n])
		if err != nil && err != EOF {
			t.Fatalf("read: %v", err)
		}

		// only final two reads should be short - 1 byte, then 0
		expect := n
		if n == 128 {
			expect = 1
		} else if n == 256 {
			expect = 0
			if err != EOF {
				t.Fatalf("read at end: %v", err)
			}
		}
		if nn != expect {
			t.Fatalf("read %d, expected %d, got %d", n, expect, nn)
		}
		tot += nn
	}
	pr := <-c
	if pr.n != 128 || pr.err != nil {
		t.Fatalf("write 128: %d, %v", pr.n, pr.err)
	}
	if tot != 128 {
		t.Fatalf("total read %d != 128", tot)
	}
	for i := 0; i < 128; i++ {
		if rdat[i] != byte(i) {
			t.Fatalf("rdat[%d] = %d", i, rdat[i])
		}
	}
}

// Test read after/before writer close.

type closer interface {
	CloseWithError(error) error
	Close() error
}

type pipeTest struct {
	async          bool
	err            error
	closeWithError bool
}

func (p pipeTest) String() string {
	return fmt.Sprintf("async=%v err=%v closeWithError=%v", p.async, p.err, p.closeWithError)
}

var pipeTests = []pipeTest{
	{true, nil, false},
	{true, nil, true},
	{true, ErrShortWrite, true},
	{false, nil, false},
	{false, nil, true},
	{false, ErrShortWrite, true},
}

func delayClose(t *testing.T, cl closer, ch chan int, tt pipeTest) {
	time.Sleep(1 * time.Millisecond)
	var err error
	if tt.closeWithError {
		err = cl.CloseWithError(tt.err)
	} else {
		err = cl.Close()
	}
	if err != nil {
		t.Errorf("delayClose: %v", err)
	}
	ch <- 0
}

func TestPipeReadClose(t *testing.T) {
	for _, tt := range pipeTests {
		c := make(chan int, 1)
		r, w := Pipe()
		if tt.async {
			go delayClose(t, w, c, tt)
		} else {
			delayClose(t, w, c, tt)
		}
		var buf = make([]byte, 64)
		n, err := r.Read(buf)
		<-c
		want := tt.err
		if want == nil {
			want = EOF
		}
		if err != want {
			t.Errorf("read from closed pipe: %v want %v", err, want)
		}
		if n != 0 {
			t.Errorf("read on closed pipe returned %d", n)
		}
		if err = r.Close(); err != nil {
			t.Errorf("r.Close: %v", err)
		}
	}
}

// Test close on Read side during Read.
func TestPipeReadClose2(t *testing.T) {
	c := make(chan int, 1)
	r, _ := Pipe()
	go delayClose(t, r, c, pipeTest{})
	n, err := r.Read(make([]byte, 64))
	<-c
	if n != 0 || err != ErrClosedPipe {
		t.Errorf("read from closed pipe: %v, %v want %v, %v", n, err, 0, ErrClosedPipe)
	}
}

// Test write after/before reader close.

func TestPipeWriteClose(t *testing.T) {
	for _, tt := range pipeTests {
		c := make(chan int, 1)
		r, w := Pipe()
		if tt.async {
			go delayClose(t, r, c, tt)
		} else {
			delayClose(t, r, c, tt)
		}
		n, err := WriteString(w, "hello, world")
		<-c
		expect := tt.err
		if expect == nil {
			expect = ErrClosedPipe
		}
		if err != expect {
			t.Errorf("write on closed pipe: %v want %v", err, expect)
		}
		if n != 0 {
			t.Errorf("write on closed pipe returned %d", n)
		}
		if err = w.Close(); err != nil {
			t.Errorf("w.Close: %v", err)
		}
	}
}

func TestWriteEmpty(t *testing.T) {
	r, w := Pipe()
	go func() {
		w.Write([]byte{})
		w.Close()
	}()
	var b [2]byte
	ReadFull(r, b[0:2])
	r.Close()
}

func TestWriteNil(t *testing.T) {
	r, w := Pipe()
	go func() {
		w.Write(nil)
		w.Close()
	}()
	var b [2]byte
	ReadFull(r, b[0:2])
	r.Close()
}
