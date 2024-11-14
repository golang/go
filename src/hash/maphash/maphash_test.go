// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maphash

import (
	"bytes"
	"fmt"
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
	b := bytes.Repeat([]byte("foo"), 100)
	hh := make([]*Hash, 7)
	for i := range hh {
		hh[i] = new(Hash)
	}
	for _, h := range hh[1:] {
		h.SetSeed(hh[0].Seed())
	}
	hh[0].Write(b)
	hh[1].WriteString(string(b))

	writeByte := func(h *Hash, b byte) {
		err := h.WriteByte(b)
		if err != nil {
			t.Fatalf("WriteByte: %v", err)
		}
	}
	writeSingleByte := func(h *Hash, b byte) {
		_, err := h.Write([]byte{b})
		if err != nil {
			t.Fatalf("Write single byte: %v", err)
		}
	}
	writeStringSingleByte := func(h *Hash, b byte) {
		_, err := h.WriteString(string([]byte{b}))
		if err != nil {
			t.Fatalf("WriteString single byte: %v", err)
		}
	}

	for i, x := range b {
		writeByte(hh[2], x)
		writeSingleByte(hh[3], x)
		if i == 0 {
			writeByte(hh[4], x)
		} else {
			writeSingleByte(hh[4], x)
		}
		writeStringSingleByte(hh[5], x)
		if i == 0 {
			writeByte(hh[6], x)
		} else {
			writeStringSingleByte(hh[6], x)
		}
	}

	sum := hh[0].Sum64()
	for i, h := range hh {
		if sum != h.Sum64() {
			t.Errorf("hash %d not identical to a single Write", i)
		}
	}

	if sum1 := Bytes(hh[0].Seed(), b); sum1 != hh[0].Sum64() {
		t.Errorf("hash using Bytes not identical to a single Write")
	}

	if sum1 := String(hh[0].Seed(), string(b)); sum1 != hh[0].Sum64() {
		t.Errorf("hash using String not identical to a single Write")
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
		t.Errorf("different sum after resetting: %#x != %#x", sum1, sum2)
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
	s := string(buf)

	b.Run("Write", func { b ->
		b.SetBytes(int64(size))
		for i := 0; i < b.N; i++ {
			h.Reset()
			h.Write(buf)
			h.Sum64()
		}
	})

	b.Run("Bytes", func { b ->
		b.SetBytes(int64(size))
		seed := h.Seed()
		for i := 0; i < b.N; i++ {
			Bytes(seed, buf)
		}
	})

	b.Run("String", func { b ->
		b.SetBytes(int64(size))
		seed := h.Seed()
		for i := 0; i < b.N; i++ {
			String(seed, s)
		}
	})
}

func BenchmarkHash(b *testing.B) {
	sizes := []int{4, 8, 16, 32, 64, 256, 320, 1024, 4096, 16384}
	for _, size := range sizes {
		b.Run(fmt.Sprint("n=", size), func { b -> benchmarkSize(b, size) })
	}
}
