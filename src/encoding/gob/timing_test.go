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
