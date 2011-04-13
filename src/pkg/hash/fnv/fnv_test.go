// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fnv

import (
	"bytes"
	"encoding/binary"
	"hash"
	"testing"
)

const testDataSize = 40

type golden struct {
	sum  []byte
	text string
}

var golden32 = []golden{
	{[]byte{0x81, 0x1c, 0x9d, 0xc5}, ""},
	{[]byte{0x05, 0x0c, 0x5d, 0x7e}, "a"},
	{[]byte{0x70, 0x77, 0x2d, 0x38}, "ab"},
	{[]byte{0x43, 0x9c, 0x2f, 0x4b}, "abc"},
}

var golden32a = []golden{
	{[]byte{0x81, 0x1c, 0x9d, 0xc5}, ""},
	{[]byte{0xe4, 0x0c, 0x29, 0x2c}, "a"},
	{[]byte{0x4d, 0x25, 0x05, 0xca}, "ab"},
	{[]byte{0x1a, 0x47, 0xe9, 0x0b}, "abc"},
}

var golden64 = []golden{
	{[]byte{0xcb, 0xf2, 0x9c, 0xe4, 0x84, 0x22, 0x23, 0x25}, ""},
	{[]byte{0xaf, 0x63, 0xbd, 0x4c, 0x86, 0x01, 0xb7, 0xbe}, "a"},
	{[]byte{0x08, 0x32, 0x67, 0x07, 0xb4, 0xeb, 0x37, 0xb8}, "ab"},
	{[]byte{0xd8, 0xdc, 0xca, 0x18, 0x6b, 0xaf, 0xad, 0xcb}, "abc"},
}

var golden64a = []golden{
	{[]byte{0xcb, 0xf2, 0x9c, 0xe4, 0x84, 0x22, 0x23, 0x25}, ""},
	{[]byte{0xaf, 0x63, 0xdc, 0x4c, 0x86, 0x01, 0xec, 0x8c}, "a"},
	{[]byte{0x08, 0x9c, 0x44, 0x07, 0xb5, 0x45, 0x98, 0x6a}, "ab"},
	{[]byte{0xe7, 0x1f, 0xa2, 0x19, 0x05, 0x41, 0x57, 0x4b}, "abc"},
}

func TestGolden32(t *testing.T) {
	testGolden(t, New32(), golden32)
}

func TestGolden32a(t *testing.T) {
	testGolden(t, New32a(), golden32a)
}

func TestGolden64(t *testing.T) {
	testGolden(t, New64(), golden64)
}

func TestGolden64a(t *testing.T) {
	testGolden(t, New64a(), golden64a)
}

func testGolden(t *testing.T, hash hash.Hash, gold []golden) {
	for _, g := range gold {
		hash.Reset()
		done, error := hash.Write([]byte(g.text))
		if error != nil {
			t.Fatalf("write error: %s", error)
		}
		if done != len(g.text) {
			t.Fatalf("wrote only %d out of %d bytes", done, len(g.text))
		}
		if actual := hash.Sum(); !bytes.Equal(g.sum, actual) {
			t.Errorf("hash(%q) = 0x%x want 0x%x", g.text, actual, g.sum)
		}
	}
}

func TestIntegrity32(t *testing.T) {
	testIntegrity(t, New32())
}

func TestIntegrity32a(t *testing.T) {
	testIntegrity(t, New32a())
}

func TestIntegrity64(t *testing.T) {
	testIntegrity(t, New64())
}

func TestIntegrity64a(t *testing.T) {
	testIntegrity(t, New64a())
}

func testIntegrity(t *testing.T, h hash.Hash) {
	data := []byte{'1', '2', 3, 4, 5}
	h.Write(data)
	sum := h.Sum()

	if size := h.Size(); size != len(sum) {
		t.Fatalf("Size()=%d but len(Sum())=%d", size, len(sum))
	}

	if a := h.Sum(); !bytes.Equal(sum, a) {
		t.Fatalf("first Sum()=0x%x, second Sum()=0x%x", sum, a)
	}

	h.Reset()
	h.Write(data)
	if a := h.Sum(); !bytes.Equal(sum, a) {
		t.Fatalf("Sum()=0x%x, but after Reset() Sum()=0x%x", sum, a)
	}

	h.Reset()
	h.Write(data[:2])
	h.Write(data[2:])
	if a := h.Sum(); !bytes.Equal(sum, a) {
		t.Fatalf("Sum()=0x%x, but with partial writes, Sum()=0x%x", sum, a)
	}

	switch h.Size() {
	case 4:
		sum32 := h.(hash.Hash32).Sum32()
		if sum32 != binary.BigEndian.Uint32(sum) {
			t.Fatalf("Sum()=0x%x, but Sum32()=0x%x", sum, sum32)
		}
	case 8:
		sum64 := h.(hash.Hash64).Sum64()
		if sum64 != binary.BigEndian.Uint64(sum) {
			t.Fatalf("Sum()=0x%x, but Sum64()=0x%x", sum, sum64)
		}
	}
}

func Benchmark32(b *testing.B) {
	benchmark(b, New32())
}

func Benchmark32a(b *testing.B) {
	benchmark(b, New32a())
}

func Benchmark64(b *testing.B) {
	benchmark(b, New64())
}

func Benchmark64a(b *testing.B) {
	benchmark(b, New64a())
}

func benchmark(b *testing.B, h hash.Hash) {
	b.ResetTimer()
	b.SetBytes(testDataSize)
	data := make([]byte, testDataSize)
	for i := range data {
		data[i] = byte(i + 'a')
	}

	b.StartTimer()
	for todo := b.N; todo != 0; todo-- {
		h.Reset()
		h.Write(data)
		h.Sum()
	}
}
