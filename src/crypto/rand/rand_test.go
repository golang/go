// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"bytes"
	"compress/flate"
	"crypto/internal/boring"
	"internal/race"
	"io"
	"runtime"
	"testing"
)

func TestRead(t *testing.T) {
	var n int = 4e6
	if testing.Short() {
		n = 1e5
	}
	b := make([]byte, n)
	n, err := io.ReadFull(Reader, b)
	if n != len(b) || err != nil {
		t.Fatalf("ReadFull(buf) = %d, %s", n, err)
	}

	var z bytes.Buffer
	f, _ := flate.NewWriter(&z, 5)
	f.Write(b)
	f.Close()
	if z.Len() < len(b)*99/100 {
		t.Fatalf("Compressed %d -> %d", len(b), z.Len())
	}
}

func TestReadEmpty(t *testing.T) {
	n, err := Reader.Read(make([]byte, 0))
	if n != 0 || err != nil {
		t.Fatalf("Read(make([]byte, 0)) = %d, %v", n, err)
	}
	n, err = Reader.Read(nil)
	if n != 0 || err != nil {
		t.Fatalf("Read(nil) = %d, %v", n, err)
	}
}

func BenchmarkRead(b *testing.B) {
	b.Run("32", func(b *testing.B) {
		benchmarkRead(b, 32)
	})
	b.Run("4K", func(b *testing.B) {
		benchmarkRead(b, 4<<10)
	})
}

func benchmarkRead(b *testing.B, size int) {
	b.SetBytes(int64(size))
	buf := make([]byte, size)
	for i := 0; i < b.N; i++ {
		if _, err := Read(buf); err != nil {
			b.Fatal(err)
		}
	}
}

func TestReadAllocs(t *testing.T) {
	if boring.Enabled || race.Enabled || (runtime.GOOS == "js" && runtime.GOARCH == "wasm") {
		t.Skip("zero-allocs unsupported")
	}

	allocs := testing.AllocsPerRun(100, func() {
		buf := make([]byte, 32)
		Read(buf)
	})
	if allocs != 0 {
		t.Fatalf("allocs = %v; want = 0", allocs)
	}
}
