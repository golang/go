// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
)

type Bench struct {
	A int
	B float64
	C string
	D []byte
}

var memprofile = flag.String("memprofile", "", "write the memory profile in Test*Mallocs to the named file")

func benchmarkEndToEnd(r io.Reader, w io.Writer, b *testing.B) {
	b.StopTimer()
	enc := NewEncoder(w)
	dec := NewDecoder(r)
	bench := &Bench{7, 3.2, "now is the time", []byte("for all good men")}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if enc.Encode(bench) != nil {
			panic("encode error")
		}
		if dec.Decode(bench) != nil {
			panic("decode error")
		}
	}
}

func BenchmarkEndToEndPipe(b *testing.B) {
	r, w, err := os.Pipe()
	if err != nil {
		panic("can't get pipe:" + err.String())
	}
	benchmarkEndToEnd(r, w, b)
}

func BenchmarkEndToEndByteBuffer(b *testing.B) {
	var buf bytes.Buffer
	benchmarkEndToEnd(&buf, &buf, b)
}

func TestCountEncodeMallocs(t *testing.T) {
	runtime.MemProfileRate = 1
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	bench := &Bench{7, 3.2, "now is the time", []byte("for all good men")}
	mallocs := 0 - runtime.MemStats.Mallocs
	const count = 1000
	for i := 0; i < count; i++ {
		err := enc.Encode(bench)
		if err != nil {
			t.Fatal("encode:", err)
		}
	}
	mallocs += runtime.MemStats.Mallocs
	if *memprofile != "" {
		if fd, err := os.Open(*memprofile, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0666); err != nil {
			t.Errorf("can't open %s: %s", *memprofile, err)
		} else {
			if err = pprof.WriteHeapProfile(fd); err != nil {
				t.Errorf("can't write %s: %s", *memprofile, err)
			}
			fd.Close()
		}
	}
	fmt.Printf("mallocs per encode of type Bench: %d\n", mallocs/count)
}

func TestCountDecodeMallocs(t *testing.T) {
	runtime.MemProfileRate = 1
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	bench := &Bench{7, 3.2, "now is the time", []byte("for all good men")}
	const count = 1000
	for i := 0; i < count; i++ {
		err := enc.Encode(bench)
		if err != nil {
			t.Fatal("encode:", err)
		}
	}
	dec := NewDecoder(&buf)
	mallocs := 0 - runtime.MemStats.Mallocs
	for i := 0; i < count; i++ {
		*bench = Bench{}
		err := dec.Decode(&bench)
		if err != nil {
			t.Fatal("decode:", err)
		}
	}
	mallocs += runtime.MemStats.Mallocs
	if *memprofile != "" {
		if fd, err := os.Open(*memprofile, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0666); err != nil {
			t.Errorf("can't open %s: %s", *memprofile, err)
		} else {
			if err = pprof.WriteHeapProfile(fd); err != nil {
				t.Errorf("can't write %s: %s", *memprofile, err)
			}
			fd.Close()
		}
	}
	fmt.Printf("mallocs per decode of type Bench: %d\n", mallocs/count)
}
