// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"io"
	"os"
	"reflect"
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
	if allocs != 3 {
		t.Fatalf("mallocs per decode of type Bench: %v; wanted 3\n", allocs)
	}
}

func benchmarkEncodeSlice(b *testing.B, a interface{}) {
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var buf bytes.Buffer
		enc := NewEncoder(&buf)

		for pb.Next() {
			buf.Reset()
			err := enc.Encode(a)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkEncodeComplex128Slice(b *testing.B) {
	a := make([]complex128, 1000)
	for i := range a {
		a[i] = 1.2 + 3.4i
	}
	benchmarkEncodeSlice(b, a)
}

func BenchmarkEncodeFloat64Slice(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = 1.23e4
	}
	benchmarkEncodeSlice(b, a)
}

func BenchmarkEncodeInt32Slice(b *testing.B) {
	a := make([]int32, 1000)
	for i := range a {
		a[i] = int32(i * 100)
	}
	benchmarkEncodeSlice(b, a)
}

func BenchmarkEncodeStringSlice(b *testing.B) {
	a := make([]string, 1000)
	for i := range a {
		a[i] = "now is the time"
	}
	benchmarkEncodeSlice(b, a)
}

func BenchmarkEncodeInterfaceSlice(b *testing.B) {
	a := make([]interface{}, 1000)
	for i := range a {
		a[i] = "now is the time"
	}
	benchmarkEncodeSlice(b, a)
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

func benchmarkDecodeSlice(b *testing.B, a interface{}) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	err := enc.Encode(a)
	if err != nil {
		b.Fatal(err)
	}

	ra := reflect.ValueOf(a)
	rt := ra.Type()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		// TODO(#19025): Move per-thread allocation before ResetTimer.
		rp := reflect.New(rt)
		rp.Elem().Set(reflect.MakeSlice(rt, ra.Len(), ra.Cap()))
		p := rp.Interface()

		bbuf := benchmarkBuf{data: buf.Bytes()}

		for pb.Next() {
			bbuf.reset()
			dec := NewDecoder(&bbuf)
			err := dec.Decode(p)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkDecodeComplex128Slice(b *testing.B) {
	a := make([]complex128, 1000)
	for i := range a {
		a[i] = 1.2 + 3.4i
	}
	benchmarkDecodeSlice(b, a)
}

func BenchmarkDecodeFloat64Slice(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = 1.23e4
	}
	benchmarkDecodeSlice(b, a)
}

func BenchmarkDecodeInt32Slice(b *testing.B) {
	a := make([]int32, 1000)
	for i := range a {
		a[i] = 1234
	}
	benchmarkDecodeSlice(b, a)
}

func BenchmarkDecodeStringSlice(b *testing.B) {
	a := make([]string, 1000)
	for i := range a {
		a[i] = "now is the time"
	}
	benchmarkDecodeSlice(b, a)
}
func BenchmarkDecodeStringsSlice(b *testing.B) {
	a := make([][]string, 1000)
	for i := range a {
		a[i] = []string{"now is the time"}
	}
	benchmarkDecodeSlice(b, a)
}
func BenchmarkDecodeBytesSlice(b *testing.B) {
	a := make([][]byte, 1000)
	for i := range a {
		a[i] = []byte("now is the time")
	}
	benchmarkDecodeSlice(b, a)
}

func BenchmarkDecodeInterfaceSlice(b *testing.B) {
	a := make([]interface{}, 1000)
	for i := range a {
		a[i] = "now is the time"
	}
	benchmarkDecodeSlice(b, a)
}

func BenchmarkDecodeMap(b *testing.B) {
	count := 1000
	m := make(map[int]int, count)
	for i := 0; i < count; i++ {
		m[i] = i
	}
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	err := enc.Encode(m)
	if err != nil {
		b.Fatal(err)
	}
	bbuf := benchmarkBuf{data: buf.Bytes()}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var rm map[int]int
		bbuf.reset()
		dec := NewDecoder(&bbuf)
		err := dec.Decode(&rm)
		if err != nil {
			b.Fatal(i, err)
		}
	}
}
