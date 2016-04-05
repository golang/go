// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"runtime"
	"testing"
)

func benchmarkEncoder(b *testing.B, testfile, level, n int) {
	b.StopTimer()
	b.SetBytes(int64(n))
	buf0, err := ioutil.ReadFile(testfiles[testfile])
	if err != nil {
		b.Fatal(err)
	}
	if len(buf0) == 0 {
		b.Fatalf("test file %q has no data", testfiles[testfile])
	}
	buf1 := make([]byte, n)
	for i := 0; i < n; i += len(buf0) {
		if len(buf0) > n-i {
			buf0 = buf0[:n-i]
		}
		copy(buf1[i:], buf0)
	}
	buf0 = nil
	w, err := NewWriter(ioutil.Discard, level)
	if err != nil {
		b.Fatal(err)
	}
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		w.Reset(ioutil.Discard)
		w.Write(buf1)
		w.Close()
	}
}

func BenchmarkEncodeDigitsHuffman1e4(b *testing.B)  { benchmarkEncoder(b, digits, huffman, 1e4) }
func BenchmarkEncodeDigitsHuffman1e5(b *testing.B)  { benchmarkEncoder(b, digits, huffman, 1e5) }
func BenchmarkEncodeDigitsHuffman1e6(b *testing.B)  { benchmarkEncoder(b, digits, huffman, 1e6) }
func BenchmarkEncodeDigitsSpeed1e4(b *testing.B)    { benchmarkEncoder(b, digits, speed, 1e4) }
func BenchmarkEncodeDigitsSpeed1e5(b *testing.B)    { benchmarkEncoder(b, digits, speed, 1e5) }
func BenchmarkEncodeDigitsSpeed1e6(b *testing.B)    { benchmarkEncoder(b, digits, speed, 1e6) }
func BenchmarkEncodeDigitsDefault1e4(b *testing.B)  { benchmarkEncoder(b, digits, default_, 1e4) }
func BenchmarkEncodeDigitsDefault1e5(b *testing.B)  { benchmarkEncoder(b, digits, default_, 1e5) }
func BenchmarkEncodeDigitsDefault1e6(b *testing.B)  { benchmarkEncoder(b, digits, default_, 1e6) }
func BenchmarkEncodeDigitsCompress1e4(b *testing.B) { benchmarkEncoder(b, digits, compress, 1e4) }
func BenchmarkEncodeDigitsCompress1e5(b *testing.B) { benchmarkEncoder(b, digits, compress, 1e5) }
func BenchmarkEncodeDigitsCompress1e6(b *testing.B) { benchmarkEncoder(b, digits, compress, 1e6) }
func BenchmarkEncodeTwainHuffman1e4(b *testing.B)   { benchmarkEncoder(b, twain, huffman, 1e4) }
func BenchmarkEncodeTwainHuffman1e5(b *testing.B)   { benchmarkEncoder(b, twain, huffman, 1e5) }
func BenchmarkEncodeTwainHuffman1e6(b *testing.B)   { benchmarkEncoder(b, twain, huffman, 1e6) }
func BenchmarkEncodeTwainSpeed1e4(b *testing.B)     { benchmarkEncoder(b, twain, speed, 1e4) }
func BenchmarkEncodeTwainSpeed1e5(b *testing.B)     { benchmarkEncoder(b, twain, speed, 1e5) }
func BenchmarkEncodeTwainSpeed1e6(b *testing.B)     { benchmarkEncoder(b, twain, speed, 1e6) }
func BenchmarkEncodeTwainDefault1e4(b *testing.B)   { benchmarkEncoder(b, twain, default_, 1e4) }
func BenchmarkEncodeTwainDefault1e5(b *testing.B)   { benchmarkEncoder(b, twain, default_, 1e5) }
func BenchmarkEncodeTwainDefault1e6(b *testing.B)   { benchmarkEncoder(b, twain, default_, 1e6) }
func BenchmarkEncodeTwainCompress1e4(b *testing.B)  { benchmarkEncoder(b, twain, compress, 1e4) }
func BenchmarkEncodeTwainCompress1e5(b *testing.B)  { benchmarkEncoder(b, twain, compress, 1e5) }
func BenchmarkEncodeTwainCompress1e6(b *testing.B)  { benchmarkEncoder(b, twain, compress, 1e6) }

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
	buf := new(bytes.Buffer)
	for i := 0; i < 1024*1024; i++ {
		buf.WriteString(fmt.Sprintf("asdasfasf%d%dfghfgujyut%dyutyu\n", i, i, i))
	}
	in := buf.Bytes()
	// We create our own buffer to control number of writes.
	copyBuffer := make([]byte, 1024)
	for l := 0; l < 10; l++ {
		for fail := 1; fail <= 512; fail *= 2 {
			// Fail after 'fail' writes
			ew := &errorWriter{N: fail}
			w, err := NewWriter(ew, l)
			if err != nil {
				t.Fatalf("NewWriter: level %d: %v", l, err)
			}
			n, err := io.CopyBuffer(w, bytes.NewBuffer(in), copyBuffer)
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

			w.Reset(ioutil.Discard)
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
func TestDeterministicL0(t *testing.T)  { testDeterministic(0, t) }
func TestDeterministicL1(t *testing.T)  { testDeterministic(1, t) }
func TestDeterministicL2(t *testing.T)  { testDeterministic(2, t) }
func TestDeterministicL3(t *testing.T)  { testDeterministic(3, t) }
func TestDeterministicL4(t *testing.T)  { testDeterministic(4, t) }
func TestDeterministicL5(t *testing.T)  { testDeterministic(5, t) }
func TestDeterministicL6(t *testing.T)  { testDeterministic(6, t) }
func TestDeterministicL7(t *testing.T)  { testDeterministic(7, t) }
func TestDeterministicL8(t *testing.T)  { testDeterministic(8, t) }
func TestDeterministicL9(t *testing.T)  { testDeterministic(9, t) }
func TestDeterministicLM2(t *testing.T) { testDeterministic(-2, t) }

func testDeterministic(i int, t *testing.T) {
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
	_, err = io.CopyBuffer(w, br, cbuf)
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
	_, err = io.CopyBuffer(w2, br2, cbuf)
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
