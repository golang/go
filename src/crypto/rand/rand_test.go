// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"bytes"
	"compress/flate"
	. "crypto/rand"
	"io"
	"sync"
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

func TestReadLoops(t *testing.T) {
	b := make([]byte, 1)
	for {
		n, err := Read(b)
		if n != 1 || err != nil {
			t.Fatalf("Read(b) = %d, %v", n, err)
		}
		if b[0] == 42 {
			break
		}
	}
	for {
		n, err := Read(b)
		if n != 1 || err != nil {
			t.Fatalf("Read(b) = %d, %v", n, err)
		}
		if b[0] == 0 {
			break
		}
	}
}

func TestLargeRead(t *testing.T) {
	// 40MiB, more than the documented maximum of 32Mi-1 on Linux 32-bit.
	b := make([]byte, 40<<20)
	if n, err := Read(b); err != nil {
		t.Fatal(err)
	} else if n != len(b) {
		t.Fatalf("Read(b) = %d, want %d", n, len(b))
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

type readerFunc func([]byte) (int, error)

func (f readerFunc) Read(b []byte) (int, error) {
	return f(b)
}

func TestReadUsesReader(t *testing.T) {
	var called bool
	defer func(r io.Reader) { Reader = r }(Reader)
	Reader = readerFunc(func(b []byte) (int, error) {
		called = true
		return len(b), nil
	})
	n, err := Read(make([]byte, 32))
	if n != 32 || err != nil {
		t.Fatalf("Read(make([]byte, 32)) = %d, %v", n, err)
	}
	if !called {
		t.Error("Read did not use Reader")
	}
}

func TestConcurrentRead(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	const N = 100
	const M = 1000
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			for i := 0; i < M; i++ {
				b := make([]byte, 32)
				n, err := Read(b)
				if n != 32 || err != nil {
					t.Errorf("Read = %d, %v", n, err)
				}
			}
		}()
	}
	wg.Wait()
}

func BenchmarkRead(b *testing.B) {
	b.Run("4", func(b *testing.B) {
		benchmarkRead(b, 4)
	})
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
