// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"fmt"
	"io"
	"math/rand"
	"runtime"
	"testing"
)

func BenchmarkEncode(b *testing.B) {
	doBench(b, func(b *testing.B, buf0 []byte, level, n int) {
		b.StopTimer()
		b.SetBytes(int64(n))

		buf1 := make([]byte, n)
		for i := 0; i < n; i += len(buf0) {
			if len(buf0) > n-i {
				buf0 = buf0[:n-i]
			}
			copy(buf1[i:], buf0)
		}
		buf0 = nil
		w, err := NewWriter(io.Discard, level)
		if err != nil {
			b.Fatal(err)
		}
		runtime.GC()
		b.StartTimer()
		for i := 0; i < b.N; i++ {
			w.Reset(io.Discard)
			w.Write(buf1)
			w.Close()
		}
	})
}

// errorWriter is a writer that fails after N writes.
type errorWriter struct {
	N int
}

func (e *errorWriter) Write(b []byte) (int, error) {
	if e.N <= 0 {
		return 0, io.ErrClosedPipe
	}
	e.N--
	return len(b), nil
}

// Test if errors from the underlying writer is passed upwards.
func TestWriteError(t *testing.T) {
	t.Parallel()
	buf := new(bytes.Buffer)
	n := 65536
	if !testing.Short() {
		n *= 4
	}
	for i := 0; i < n; i++ {
		fmt.Fprintf(buf, "asdasfasf%d%dfghfgujyut%dyutyu\n", i, i, i)
	}
	in := buf.Bytes()
	// We create our own buffer to control number of writes.
	copyBuffer := make([]byte, 128)
	for l := 0; l < 10; l++ {
		for fail := 1; fail <= 256; fail *= 2 {
			// Fail after 'fail' writes
			ew := &errorWriter{N: fail}
			w, err := NewWriter(ew, l)
			if err != nil {
				t.Fatalf("NewWriter: level %d: %v", l, err)
			}
			n, err := io.CopyBuffer(w, struct{ io.Reader }{bytes.NewBuffer(in)}, copyBuffer)
			if err == nil {
				t.Fatalf("Level %d: Expected an error, writer was %#v", l, ew)
			}
			n2, err := w.Write([]byte{1, 2, 2, 3, 4, 5})
			if n2 != 0 {
				t.Fatal("Level", l, "Expected 0 length write, got", n)
			}
			if err == nil {
				t.Fatal("Level", l, "Expected an error")
			}
			err = w.Flush()
			if err == nil {
				t.Fatal("Level", l, "Expected an error on flush")
			}
			err = w.Close()
			if err == nil {
				t.Fatal("Level", l, "Expected an error on close")
			}

			w.Reset(io.Discard)
			n2, err = w.Write([]byte{1, 2, 3, 4, 5, 6})
			if err != nil {
				t.Fatal("Level", l, "Got unexpected error after reset:", err)
			}
			if n2 == 0 {
				t.Fatal("Level", l, "Got 0 length write, expected > 0")
			}
			if testing.Short() {
				return
			}
		}
	}
}

// Test if two runs produce identical results
// even when writing different sizes to the Writer.
func TestDeterministic(t *testing.T) {
	t.Parallel()
	for i := 0; i <= 9; i++ {
		t.Run(fmt.Sprint("L", i), func(t *testing.T) { testDeterministic(i, t) })
	}
	t.Run("LM2", func(t *testing.T) { testDeterministic(-2, t) })
}

func testDeterministic(i int, t *testing.T) {
	t.Parallel()
	// Test so much we cross a good number of block boundaries.
	var length = maxStoreBlockSize*30 + 500
	if testing.Short() {
		length /= 10
	}

	// Create a random, but compressible stream.
	rng := rand.New(rand.NewSource(1))
	t1 := make([]byte, length)
	for i := range t1 {
		t1[i] = byte(rng.Int63() & 7)
	}

	// Do our first encode.
	var b1 bytes.Buffer
	br := bytes.NewBuffer(t1)
	w, err := NewWriter(&b1, i)
	if err != nil {
		t.Fatal(err)
	}
	// Use a very small prime sized buffer.
	cbuf := make([]byte, 787)
	_, err = io.CopyBuffer(w, struct{ io.Reader }{br}, cbuf)
	if err != nil {
		t.Fatal(err)
	}
	w.Close()

	// We choose a different buffer size,
	// bigger than a maximum block, and also a prime.
	var b2 bytes.Buffer
	cbuf = make([]byte, 81761)
	br2 := bytes.NewBuffer(t1)
	w2, err := NewWriter(&b2, i)
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.CopyBuffer(w2, struct{ io.Reader }{br2}, cbuf)
	if err != nil {
		t.Fatal(err)
	}
	w2.Close()

	b1b := b1.Bytes()
	b2b := b2.Bytes()

	if !bytes.Equal(b1b, b2b) {
		t.Errorf("level %d did not produce deterministic result, result mismatch, len(a) = %d, len(b) = %d", i, len(b1b), len(b2b))
	}
}

// TestDeflateFast_Reset will test that encoding is consistent
// across a warparound of the table offset.
// See https://github.com/golang/go/issues/34121
func TestDeflateFast_Reset(t *testing.T) {
	buf := new(bytes.Buffer)
	n := 65536

	for i := 0; i < n; i++ {
		fmt.Fprintf(buf, "asdfasdfasdfasdf%d%dfghfgujyut%dyutyu\n", i, i, i)
	}
	// This is specific to level 1.
	const level = 1
	in := buf.Bytes()
	offset := 1
	if testing.Short() {
		offset = 256
	}

	// We do an encode with a clean buffer to compare.
	var want bytes.Buffer
	w, err := NewWriter(&want, level)
	if err != nil {
		t.Fatalf("NewWriter: level %d: %v", level, err)
	}

	// Output written 3 times.
	w.Write(in)
	w.Write(in)
	w.Write(in)
	w.Close()

	for ; offset <= 256; offset *= 2 {
		w, err := NewWriter(io.Discard, level)
		if err != nil {
			t.Fatalf("NewWriter: level %d: %v", level, err)
		}

		// Reset until we are right before the wraparound.
		// Each reset adds maxMatchOffset to the offset.
		for i := 0; i < (bufferReset-len(in)-offset-maxMatchOffset)/maxMatchOffset; i++ {
			// skip ahead to where we are close to wrap around...
			w.d.reset(nil)
		}
		var got bytes.Buffer
		w.Reset(&got)

		// Write 3 times, close.
		for i := 0; i < 3; i++ {
			_, err = w.Write(in)
			if err != nil {
				t.Fatal(err)
			}
		}
		err = w.Close()
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got.Bytes(), want.Bytes()) {
			t.Fatalf("output did not match at wraparound, len(want)  = %d, len(got) = %d", want.Len(), got.Len())
		}
	}
}
