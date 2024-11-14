// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestOneByteReader_nonEmptyReader(t *testing.T) {
	msg := "Hello, World!"
	buf := new(bytes.Buffer)
	buf.WriteString(msg)

	obr := OneByteReader(buf)
	var b []byte
	n, err := obr.Read(b)
	if err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}

	b = make([]byte, 3)
	// Read from obr until EOF.
	got := new(strings.Builder)
	for i := 0; ; i++ {
		n, err = obr.Read(b)
		if err != nil {
			break
		}
		if g, w := n, 1; g != w {
			t.Errorf("Iteration #%d read %d bytes, want %d", i, g, w)
		}
		got.Write(b[:n])
	}
	if g, w := err, io.EOF; g != w {
		t.Errorf("Unexpected error after reading all bytes\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := got.String(), "Hello, World!"; g != w {
		t.Errorf("Read mismatch\n\tGot:  %q\n\tWant: %q", g, w)
	}
}

func TestOneByteReader_emptyReader(t *testing.T) {
	r := new(bytes.Buffer)

	obr := OneByteReader(r)
	var b []byte
	if n, err := obr.Read(b); err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}

	b = make([]byte, 5)
	n, err := obr.Read(b)
	if g, w := err, io.EOF; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
}

func TestHalfReader_nonEmptyReader(t *testing.T) {
	msg := "Hello, World!"
	buf := new(bytes.Buffer)
	buf.WriteString(msg)
	// empty read buffer
	hr := HalfReader(buf)
	var b []byte
	n, err := hr.Read(b)
	if err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}
	// non empty read buffer
	b = make([]byte, 2)
	got := new(strings.Builder)
	for i := 0; ; i++ {
		n, err = hr.Read(b)
		if err != nil {
			break
		}
		if g, w := n, 1; g != w {
			t.Errorf("Iteration #%d read %d bytes, want %d", i, g, w)
		}
		got.Write(b[:n])
	}
	if g, w := err, io.EOF; g != w {
		t.Errorf("Unexpected error after reading all bytes\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := got.String(), "Hello, World!"; g != w {
		t.Errorf("Read mismatch\n\tGot:  %q\n\tWant: %q", g, w)
	}
}

func TestHalfReader_emptyReader(t *testing.T) {
	r := new(bytes.Buffer)

	hr := HalfReader(r)
	var b []byte
	if n, err := hr.Read(b); err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}

	b = make([]byte, 5)
	n, err := hr.Read(b)
	if g, w := err, io.EOF; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
}

func TestTimeOutReader_nonEmptyReader(t *testing.T) {
	msg := "Hello, World!"
	buf := new(bytes.Buffer)
	buf.WriteString(msg)
	// empty read buffer
	tor := TimeoutReader(buf)
	var b []byte
	n, err := tor.Read(b)
	if err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}
	// Second call should timeout
	n, err = tor.Read(b)
	if g, w := err, ErrTimeout; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
	// non empty read buffer
	tor2 := TimeoutReader(buf)
	b = make([]byte, 3)
	if n, err := tor2.Read(b); err != nil || n == 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}
	// Second call should timeout
	n, err = tor2.Read(b)
	if g, w := err, ErrTimeout; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
}

func TestTimeOutReader_emptyReader(t *testing.T) {
	r := new(bytes.Buffer)
	// empty read buffer
	tor := TimeoutReader(r)
	var b []byte
	if n, err := tor.Read(b); err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}
	// Second call should timeout
	n, err := tor.Read(b)
	if g, w := err, ErrTimeout; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
	// non empty read buffer
	tor2 := TimeoutReader(r)
	b = make([]byte, 5)
	if n, err := tor2.Read(b); err != io.EOF || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}
	// Second call should timeout
	n, err = tor2.Read(b)
	if g, w := err, ErrTimeout; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
}

func TestDataErrReader_nonEmptyReader(t *testing.T) {
	msg := "Hello, World!"
	buf := new(bytes.Buffer)
	buf.WriteString(msg)

	der := DataErrReader(buf)

	b := make([]byte, 3)
	got := new(strings.Builder)
	var n int
	var err error
	for {
		n, err = der.Read(b)
		got.Write(b[:n])
		if err != nil {
			break
		}
	}
	if err != io.EOF || n == 0 {
		t.Errorf("Last Read returned n=%d err=%v", n, err)
	}
	if g, w := got.String(), "Hello, World!"; g != w {
		t.Errorf("Read mismatch\n\tGot:  %q\n\tWant: %q", g, w)
	}
}

func TestDataErrReader_emptyReader(t *testing.T) {
	r := new(bytes.Buffer)

	der := DataErrReader(r)
	var b []byte
	if n, err := der.Read(b); err != io.EOF || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}

	b = make([]byte, 5)
	n, err := der.Read(b)
	if g, w := err, io.EOF; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
}

func TestErrReader(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"nil error", nil},
		{"non-nil error", errors.New("io failure")},
		{"io.EOF", io.EOF},
	}

	for _, tt := range cases {
		tt := tt
		t.Run(tt.name, func { t ->
			n, err := ErrReader(tt.err).Read(nil)
			if err != tt.err {
				t.Fatalf("Error mismatch\nGot:  %v\nWant: %v", err, tt.err)
			}
			if n != 0 {
				t.Fatalf("Byte count mismatch: got %d want 0", n)
			}
		})
	}
}

func TestStringsReader(t *testing.T) {
	const msg = "Now is the time for all good gophers."

	r := strings.NewReader(msg)
	if err := TestReader(r, []byte(msg)); err != nil {
		t.Fatal(err)
	}
}
