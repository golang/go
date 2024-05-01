// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"bytes"
	"fmt"
	. "io"
	"sort"
	"strings"
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

// Test close on Write side during Write.
func TestPipeWriteClose2(t *testing.T) {
	c := make(chan int, 1)
	_, w := Pipe()
	go delayClose(t, w, c, pipeTest{})
	n, err := w.Write(make([]byte, 64))
	<-c
	if n != 0 || err != ErrClosedPipe {
		t.Errorf("write to closed pipe: %v, %v want %v, %v", n, err, 0, ErrClosedPipe)
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

func TestWriteAfterWriterClose(t *testing.T) {
	r, w := Pipe()
	defer r.Close()
	done := make(chan bool)
	var writeErr error
	go func() {
		_, err := w.Write([]byte("hello"))
		if err != nil {
			t.Errorf("got error: %q; expected none", err)
		}
		w.Close()
		_, writeErr = w.Write([]byte("world"))
		done <- true
	}()

	buf := make([]byte, 100)
	var result string
	n, err := ReadFull(r, buf)
	if err != nil && err != ErrUnexpectedEOF {
		t.Fatalf("got: %q; want: %q", err, ErrUnexpectedEOF)
	}
	result = string(buf[0:n])
	<-done

	if result != "hello" {
		t.Errorf("got: %q; want: %q", result, "hello")
	}
	if writeErr != ErrClosedPipe {
		t.Errorf("got: %q; want: %q", writeErr, ErrClosedPipe)
	}
}

func TestPipeCloseError(t *testing.T) {
	type testError1 struct{ error }
	type testError2 struct{ error }

	r, w := Pipe()
	r.CloseWithError(testError1{})
	if _, err := w.Write(nil); err != (testError1{}) {
		t.Errorf("Write error: got %T, want testError1", err)
	}
	r.CloseWithError(testError2{})
	if _, err := w.Write(nil); err != (testError1{}) {
		t.Errorf("Write error: got %T, want testError1", err)
	}

	r, w = Pipe()
	w.CloseWithError(testError1{})
	if _, err := r.Read(nil); err != (testError1{}) {
		t.Errorf("Read error: got %T, want testError1", err)
	}
	w.CloseWithError(testError2{})
	if _, err := r.Read(nil); err != (testError1{}) {
		t.Errorf("Read error: got %T, want testError1", err)
	}
}

func TestPipeConcurrent(t *testing.T) {
	const (
		input    = "0123456789abcdef"
		count    = 8
		readSize = 2
	)

	t.Run("Write", func(t *testing.T) {
		r, w := Pipe()

		for i := 0; i < count; i++ {
			go func() {
				time.Sleep(time.Millisecond) // Increase probability of race
				if n, err := w.Write([]byte(input)); n != len(input) || err != nil {
					t.Errorf("Write() = (%d, %v); want (%d, nil)", n, err, len(input))
				}
			}()
		}

		buf := make([]byte, count*len(input))
		for i := 0; i < len(buf); i += readSize {
			if n, err := r.Read(buf[i : i+readSize]); n != readSize || err != nil {
				t.Errorf("Read() = (%d, %v); want (%d, nil)", n, err, readSize)
			}
		}

		// Since each Write is fully gated, if multiple Read calls were needed,
		// the contents of Write should still appear together in the output.
		got := string(buf)
		want := strings.Repeat(input, count)
		if got != want {
			t.Errorf("got: %q; want: %q", got, want)
		}
	})

	t.Run("Read", func(t *testing.T) {
		r, w := Pipe()

		c := make(chan []byte, count*len(input)/readSize)
		for i := 0; i < cap(c); i++ {
			go func() {
				time.Sleep(time.Millisecond) // Increase probability of race
				buf := make([]byte, readSize)
				if n, err := r.Read(buf); n != readSize || err != nil {
					t.Errorf("Read() = (%d, %v); want (%d, nil)", n, err, readSize)
				}
				c <- buf
			}()
		}

		for i := 0; i < count; i++ {
			if n, err := w.Write([]byte(input)); n != len(input) || err != nil {
				t.Errorf("Write() = (%d, %v); want (%d, nil)", n, err, len(input))
			}
		}

		// Since each read is independent, the only guarantee about the output
		// is that it is a permutation of the input in readSized groups.
		got := make([]byte, 0, count*len(input))
		for i := 0; i < cap(c); i++ {
			got = append(got, (<-c)...)
		}
		got = sortBytesInGroups(got, readSize)
		want := bytes.Repeat([]byte(input), count)
		want = sortBytesInGroups(want, readSize)
		if string(got) != string(want) {
			t.Errorf("got: %q; want: %q", got, want)
		}
	})
}

func sortBytesInGroups(b []byte, n int) []byte {
	var groups [][]byte
	for len(b) > 0 {
		groups = append(groups, b[:n])
		b = b[n:]
	}
	sort.Slice(groups, func(i, j int) bool { return bytes.Compare(groups[i], groups[j]) < 0 })
	return bytes.Join(groups, nil)
}
