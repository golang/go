// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maphash

import (
	"hash"
	"testing"
)

func TestUnseededHash(t *testing.T) {
	m := map[uint64]struct{}{}
	for i := 0; i < 1000; i++ {
		h := new(Hash)
		m[h.Sum64()] = struct{}{}
	}
	if len(m) < 900 {
		t.Errorf("empty hash not sufficiently random: got %d, want 1000", len(m))
	}
}

func TestSeededHash(t *testing.T) {
	s := MakeSeed()
	m := map[uint64]struct{}{}
	for i := 0; i < 1000; i++ {
		h := new(Hash)
		h.SetSeed(s)
		m[h.Sum64()] = struct{}{}
	}
	if len(m) != 1 {
		t.Errorf("seeded hash is random: got %d, want 1", len(m))
	}
}

func TestHashGrouping(t *testing.T) {
	b := []byte("foo")
	h1 := new(Hash)
	h2 := new(Hash)
	h2.SetSeed(h1.Seed())
	h1.Write(b)
	for _, x := range b {
		err := h2.WriteByte(x)
		if err != nil {
			t.Fatalf("WriteByte: %v", err)
		}
	}
	if h1.Sum64() != h2.Sum64() {
		t.Errorf("hash of \"foo\" and \"f\",\"o\",\"o\" not identical")
	}
}

func TestHashBytesVsString(t *testing.T) {
	s := "foo"
	b := []byte(s)
	h1 := new(Hash)
	h2 := new(Hash)
	h2.SetSeed(h1.Seed())
	n1, err1 := h1.WriteString(s)
	if n1 != len(s) || err1 != nil {
		t.Fatalf("WriteString(s) = %d, %v, want %d, nil", n1, err1, len(s))
	}
	n2, err2 := h2.Write(b)
	if n2 != len(b) || err2 != nil {
		t.Fatalf("Write(b) = %d, %v, want %d, nil", n2, err2, len(b))
	}
	if h1.Sum64() != h2.Sum64() {
		t.Errorf("hash of string and bytes not identical")
	}
}

func TestHashHighBytes(t *testing.T) {
	// See issue 34925.
	const N = 10
	m := map[uint64]struct{}{}
	for i := 0; i < N; i++ {
		h := new(Hash)
		h.WriteString("foo")
		m[h.Sum64()>>32] = struct{}{}
	}
	if len(m) < N/2 {
		t.Errorf("from %d seeds, wanted at least %d different hashes; got %d", N, N/2, len(m))
	}
}

func TestRepeat(t *testing.T) {
	h1 := new(Hash)
	h1.WriteString("testing")
	sum1 := h1.Sum64()

	h1.Reset()
	h1.WriteString("testing")
	sum2 := h1.Sum64()

	if sum1 != sum2 {
		t.Errorf("different sum after reseting: %#x != %#x", sum1, sum2)
	}

	h2 := new(Hash)
	h2.SetSeed(h1.Seed())
	h2.WriteString("testing")
	sum3 := h2.Sum64()

	if sum1 != sum3 {
		t.Errorf("different sum on the same seed: %#x != %#x", sum1, sum3)
	}
}

func TestSeedFromSum64(t *testing.T) {
	h1 := new(Hash)
	h1.WriteString("foo")
	x := h1.Sum64() // seed generated here
	h2 := new(Hash)
	h2.SetSeed(h1.Seed())
	h2.WriteString("foo")
	y := h2.Sum64()
	if x != y {
		t.Errorf("hashes don't match: want %x, got %x", x, y)
	}
}

func TestSeedFromSeed(t *testing.T) {
	h1 := new(Hash)
	h1.WriteString("foo")
	_ = h1.Seed() // seed generated here
	x := h1.Sum64()
	h2 := new(Hash)
	h2.SetSeed(h1.Seed())
	h2.WriteString("foo")
	y := h2.Sum64()
	if x != y {
		t.Errorf("hashes don't match: want %x, got %x", x, y)
	}
}

func TestSeedFromFlush(t *testing.T) {
	b := make([]byte, 65)
	h1 := new(Hash)
	h1.Write(b) // seed generated here
	x := h1.Sum64()
	h2 := new(Hash)
	h2.SetSeed(h1.Seed())
	h2.Write(b)
	y := h2.Sum64()
	if x != y {
		t.Errorf("hashes don't match: want %x, got %x", x, y)
	}
}

func TestSeedFromReset(t *testing.T) {
	h1 := new(Hash)
	h1.WriteString("foo")
	h1.Reset() // seed generated here
	h1.WriteString("foo")
	x := h1.Sum64()
	h2 := new(Hash)
	h2.SetSeed(h1.Seed())
	h2.WriteString("foo")
	y := h2.Sum64()
	if x != y {
		t.Errorf("hashes don't match: want %x, got %x", x, y)
	}
}

// Make sure a Hash implements the hash.Hash and hash.Hash64 interfaces.
var _ hash.Hash = &Hash{}
var _ hash.Hash64 = &Hash{}

func benchmarkSize(b *testing.B, size int) {
	h := &Hash{}
	buf := make([]byte, size)
	b.SetBytes(int64(size))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		h.Reset()
		h.Write(buf)
		h.Sum64()
	}
}

func BenchmarkHash8Bytes(b *testing.B) {
	benchmarkSize(b, 8)
}

func BenchmarkHash320Bytes(b *testing.B) {
	benchmarkSize(b, 320)
}

func BenchmarkHash1K(b *testing.B) {
	benchmarkSize(b, 1024)
}

func BenchmarkHash8K(b *testing.B) {
	benchmarkSize(b, 8192)
}
