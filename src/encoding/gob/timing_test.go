// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"io"
	"os"
	"runtime"
	"testing"
)

type Bench struct {
	A int
	B float64
	C string
	D []byte
}

func benchmarkEndToEnd(b *testing.B, ctor func() interface{}, pipe func() (r io.Reader, w io.Writer, err error)) {
	b.RunParallel(func(pb *testing.PB) {
		r, w, err := pipe()
		if err != nil {
			b.Fatal("can't get pipe:", err)
		}
		v := ctor()
		enc := NewEncoder(w)
		dec := NewDecoder(r)
		for pb.Next() {
			if err := enc.Encode(v); err != nil {
				b.Fatal("encode error:", err)
			}
			if err := dec.Decode(v); err != nil {
				b.Fatal("decode error:", err)
			}
		}
	})
}

func BenchmarkEndToEndPipe(b *testing.B) {
	benchmarkEndToEnd(b, func() interface{} {
		return &Bench{7, 3.2, "now is the time", bytes.Repeat([]byte("for all good men"), 100)}
	}, func() (r io.Reader, w io.Writer, err error) {
		r, w, err = os.Pipe()
		return
	})
}

func BenchmarkEndToEndByteBuffer(b *testing.B) {
	benchmarkEndToEnd(b, func() interface{} {
		return &Bench{7, 3.2, "now is the time", bytes.Repeat([]byte("for all good men"), 100)}
	}, func() (r io.Reader, w io.Writer, err error) {
		var buf bytes.Buffer
		return &buf, &buf, nil
	})
}

func BenchmarkEndToEndSliceByteBuffer(b *testing.B) {
	benchmarkEndToEnd(b, func() interface{} {
		v := &Bench{7, 3.2, "now is the time", nil}
		Register(v)
		arr := make([]interface{}, 100)
		for i := range arr {
			arr[i] = v
		}
		return &arr
	}, func() (r io.Reader, w io.Writer, err error) {
		var buf bytes.Buffer
		return &buf, &buf, nil
	})
}

func TestCountEncodeMallocs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Skip("skipping; GOMAXPROCS>1")
	}

	const N = 1000

	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	bench := &Bench{7, 3.2, "now is the time", []byte("for all good men")}

	allocs := testing.AllocsPerRun(N, func() {
		err := enc.Encode(bench)
		if err != nil {
			t.Fatal("encode:", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("mallocs per encode of type Bench: %v; wanted 0\n", allocs)
	}
}

func TestCountDecodeMallocs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Skip("skipping; GOMAXPROCS>1")
	}

	const N = 1000

	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	bench := &Bench{7, 3.2, "now is the time", []byte("for all good men")}

	// Fill the buffer with enough to decode
	testing.AllocsPerRun(N, func() {
		err := enc.Encode(bench)
		if err != nil {
			t.Fatal("encode:", err)
		}
	})

	dec := NewDecoder(&buf)
	allocs := testing.AllocsPerRun(N, func() {
		*bench = Bench{}
		err := dec.Decode(&bench)
		if err != nil {
			t.Fatal("decode:", err)
		}
	})
	if allocs != 4 {
		t.Fatalf("mallocs per decode of type Bench: %v; wanted 4\n", allocs)
	}
}

func BenchmarkEncodeComplex128Slice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]complex128, 1000)
	for i := range a {
		a[i] = 1.2 + 3.4i
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		err := enc.Encode(a)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodeFloat64Slice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]float64, 1000)
	for i := range a {
		a[i] = 1.23e4
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		err := enc.Encode(a)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodeInt32Slice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]int32, 1000)
	for i := range a {
		a[i] = 1234
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		err := enc.Encode(a)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodeStringSlice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]string, 1000)
	for i := range a {
		a[i] = "now is the time"
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		err := enc.Encode(a)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// benchmarkBuf is a read buffer we can reset
type benchmarkBuf struct {
	offset int
	data   []byte
}

func (b *benchmarkBuf) Read(p []byte) (n int, err error) {
	n = copy(p, b.data[b.offset:])
	if n == 0 {
		return 0, io.EOF
	}
	b.offset += n
	return
}

func (b *benchmarkBuf) ReadByte() (c byte, err error) {
	if b.offset >= len(b.data) {
		return 0, io.EOF
	}
	c = b.data[b.offset]
	b.offset++
	return
}

func (b *benchmarkBuf) reset() {
	b.offset = 0
}

func BenchmarkDecodeComplex128Slice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]complex128, 1000)
	for i := range a {
		a[i] = 1.2 + 3.4i
	}
	err := enc.Encode(a)
	if err != nil {
		b.Fatal(err)
	}
	x := make([]complex128, 1000)
	bbuf := benchmarkBuf{data: buf.Bytes()}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bbuf.reset()
		dec := NewDecoder(&bbuf)
		err := dec.Decode(&x)
		if err != nil {
			b.Fatal(i, err)
		}
	}
}

func BenchmarkDecodeFloat64Slice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]float64, 1000)
	for i := range a {
		a[i] = 1.23e4
	}
	err := enc.Encode(a)
	if err != nil {
		b.Fatal(err)
	}
	x := make([]float64, 1000)
	bbuf := benchmarkBuf{data: buf.Bytes()}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bbuf.reset()
		dec := NewDecoder(&bbuf)
		err := dec.Decode(&x)
		if err != nil {
			b.Fatal(i, err)
		}
	}
}

func BenchmarkDecodeInt32Slice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]int32, 1000)
	for i := range a {
		a[i] = 1234
	}
	err := enc.Encode(a)
	if err != nil {
		b.Fatal(err)
	}
	x := make([]int32, 1000)
	bbuf := benchmarkBuf{data: buf.Bytes()}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bbuf.reset()
		dec := NewDecoder(&bbuf)
		err := dec.Decode(&x)
		if err != nil {
			b.Fatal(i, err)
		}
	}
}

func BenchmarkDecodeStringSlice(b *testing.B) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	a := make([]string, 1000)
	for i := range a {
		a[i] = "now is the time"
	}
	err := enc.Encode(a)
	if err != nil {
		b.Fatal(err)
	}
	x := make([]string, 1000)
	bbuf := benchmarkBuf{data: buf.Bytes()}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bbuf.reset()
		dec := NewDecoder(&bbuf)
		err := dec.Decode(&x)
		if err != nil {
			b.Fatal(i, err)
		}
	}
}
