// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"hash"
	"io"
	"testing"
)

type test struct {
	ieee, castagnoli uint32
	in               string
}

var golden = []test{
	{0x0, 0x0, ""},
	{0xe8b7be43, 0xc1d04330, "a"},
	{0x9e83486d, 0xe2a22936, "ab"},
	{0x352441c2, 0x364b3fb7, "abc"},
	{0xed82cd11, 0x92c80a31, "abcd"},
	{0x8587d865, 0xc450d697, "abcde"},
	{0x4b8e39ef, 0x53bceff1, "abcdef"},
	{0x312a6aa6, 0xe627f441, "abcdefg"},
	{0xaeef2a50, 0xa9421b7, "abcdefgh"},
	{0x8da988af, 0x2ddc99fc, "abcdefghi"},
	{0x3981703a, 0xe6599437, "abcdefghij"},
	{0x6b9cdfe7, 0xb2cc01fe, "Discard medicine more than two years old."},
	{0xc90ef73f, 0xe28207f, "He who has a shady past knows that nice guys finish last."},
	{0xb902341f, 0xbe93f964, "I wouldn't marry him with a ten foot pole."},
	{0x42080e8, 0x9e3be0c3, "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave"},
	{0x154c6d11, 0xf505ef04, "The days of the digital watch are numbered.  -Tom Stoppard"},
	{0x4c418325, 0x85d3dc82, "Nepal premier won't resign."},
	{0x33955150, 0xc5142380, "For every action there is an equal and opposite government program."},
	{0x26216a4b, 0x75eb77dd, "His money is twice tainted: 'taint yours and 'taint mine."},
	{0x1abbe45e, 0x91ebe9f7, "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977"},
	{0xc89a94f7, 0xf0b1168e, "It's a tiny change to the code and not completely disgusting. - Bob Manchek"},
	{0xab3abe14, 0x572b74e2, "size:  a.out:  bad magic"},
	{0xbab102b6, 0x8a58a6d5, "The major problem is with sendmail.  -Mark Horton"},
	{0x999149d7, 0x9c426c50, "Give me a rock, paper and scissors and I will move the world.  CCFestoon"},
	{0x6d52a33c, 0x735400a4, "If the enemy is within range, then so are you."},
	{0x90631e8d, 0xbec49c95, "It's well we cannot hear the screams/That we create in others' dreams."},
	{0x78309130, 0xa95a2079, "You remind me of a TV show, but that's all right: I watch it anyway."},
	{0x7d0a377f, 0xde2e65c5, "C is as portable as Stonehedge!!"},
	{0x8c79fd79, 0x297a88ed, "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley"},
	{0xa20b7167, 0x66ed1d8b, "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule"},
	{0x8e0bb443, 0xdcded527, "How can you write a big system without C++?  -Paul Glick"},
}

func TestGolden(t *testing.T) {
	castagnoliTab := MakeTable(Castagnoli)

	for _, g := range golden {
		ieee := NewIEEE()
		io.WriteString(ieee, g.in)
		s := ieee.Sum32()
		if s != g.ieee {
			t.Errorf("IEEE(%s) = 0x%x want 0x%x", g.in, s, g.ieee)
		}

		castagnoli := New(castagnoliTab)
		io.WriteString(castagnoli, g.in)
		s = castagnoli.Sum32()
		if s != g.castagnoli {
			t.Errorf("Castagnoli(%s) = 0x%x want 0x%x", g.in, s, g.castagnoli)
		}

		if len(g.in) > 0 {
			// The SSE4.2 implementation of this has code to deal
			// with misaligned data so we ensure that we test that
			// too.
			castagnoli = New(castagnoliTab)
			io.WriteString(castagnoli, g.in[:1])
			io.WriteString(castagnoli, g.in[1:])
			s = castagnoli.Sum32()
			if s != g.castagnoli {
				t.Errorf("Castagnoli[misaligned](%s) = 0x%x want 0x%x", g.in, s, g.castagnoli)
			}
		}
	}
}

func BenchmarkIEEECrc40B(b *testing.B) {
	benchmark(b, NewIEEE(), 40)
}

func BenchmarkIEEECrc1KB(b *testing.B) {
	benchmark(b, NewIEEE(), 1<<10)
}

func BenchmarkIEEECrc4KB(b *testing.B) {
	benchmark(b, NewIEEE(), 4<<10)
}

func BenchmarkIEEECrc32KB(b *testing.B) {
	benchmark(b, NewIEEE(), 32<<10)
}

func BenchmarkCastagnoliCrc40B(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 40)
}

func BenchmarkCastagnoliCrc1KB(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 1<<10)
}

func BenchmarkCastagnoliCrc4KB(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 4<<10)
}

func BenchmarkCastagnoliCrc32KB(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 32<<10)
}

func benchmark(b *testing.B, h hash.Hash32, n int64) {
	b.SetBytes(n)
	data := make([]byte, n)
	for i := range data {
		data[i] = byte(i)
	}
	in := make([]byte, 0, h.Size())

	// Warm up
	h.Reset()
	h.Write(data)
	h.Sum(in)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Reset()
		h.Write(data)
		h.Sum(in)
	}
}
