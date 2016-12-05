// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"hash"
	"math/rand"
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

// testGoldenIEEE verifies that the given function returns
// correct IEEE checksums.
func testGoldenIEEE(t *testing.T, crcFunc func(b []byte) uint32) {
	for _, g := range golden {
		if crc := crcFunc([]byte(g.in)); crc != g.ieee {
			t.Errorf("IEEE(%s) = 0x%x want 0x%x", g.in, crc, g.ieee)
		}
	}
}

// testGoldenCastagnoli verifies that the given function returns
// correct IEEE checksums.
func testGoldenCastagnoli(t *testing.T, crcFunc func(b []byte) uint32) {
	for _, g := range golden {
		if crc := crcFunc([]byte(g.in)); crc != g.castagnoli {
			t.Errorf("Castagnoli(%s) = 0x%x want 0x%x", g.in, crc, g.castagnoli)
		}
	}
}

// testCrossCheck generates random buffers of various lengths and verifies that
// the two "update" functions return the same result.
func testCrossCheck(t *testing.T, crcFunc1, crcFunc2 func(crc uint32, b []byte) uint32) {
	// The AMD64 implementation has some cutoffs at lengths 168*3=504 and
	// 1344*3=4032. We should make sure lengths around these values are in the
	// list.
	lengths := []int{0, 1, 2, 3, 4, 5, 10, 16, 50, 100, 128,
		500, 501, 502, 503, 504, 505, 512, 1000, 1024, 2000,
		4030, 4031, 4032, 4033, 4036, 4040, 4048, 4096, 5000, 10000}
	for _, length := range lengths {
		p := make([]byte, length)
		_, _ = rand.Read(p)
		crcInit := uint32(rand.Int63())
		crc1 := crcFunc1(crcInit, p)
		crc2 := crcFunc2(crcInit, p)
		if crc1 != crc2 {
			t.Errorf("mismatch: 0x%x vs 0x%x (buffer length %d)", crc1, crc2, length)
		}
	}
}

// TestSimple tests the simple generic algorithm.
func TestSimple(t *testing.T) {
	tab := simpleMakeTable(IEEE)
	testGoldenIEEE(t, func(b []byte) uint32 {
		return simpleUpdate(0, tab, b)
	})

	tab = simpleMakeTable(Castagnoli)
	testGoldenCastagnoli(t, func(b []byte) uint32 {
		return simpleUpdate(0, tab, b)
	})
}

// TestSimple tests the slicing-by-8 algorithm.
func TestSlicing(t *testing.T) {
	tab := slicingMakeTable(IEEE)
	testGoldenIEEE(t, func(b []byte) uint32 {
		return slicingUpdate(0, tab, b)
	})

	tab = slicingMakeTable(Castagnoli)
	testGoldenCastagnoli(t, func(b []byte) uint32 {
		return slicingUpdate(0, tab, b)
	})

	// Cross-check various polys against the simple algorithm.
	for _, poly := range []uint32{IEEE, Castagnoli, Koopman, 0xD5828281} {
		t1 := simpleMakeTable(poly)
		f1 := func(crc uint32, b []byte) uint32 {
			return simpleUpdate(crc, t1, b)
		}
		t2 := slicingMakeTable(poly)
		f2 := func(crc uint32, b []byte) uint32 {
			return slicingUpdate(crc, t2, b)
		}
		testCrossCheck(t, f1, f2)
	}
}

func TestArchIEEE(t *testing.T) {
	if !archAvailableIEEE() {
		t.Skip("Arch-specific IEEE not available.")
	}
	archInitIEEE()
	slicingTable := slicingMakeTable(IEEE)
	testCrossCheck(t, archUpdateIEEE, func(crc uint32, b []byte) uint32 {
		return slicingUpdate(crc, slicingTable, b)
	})
}

func TestArchCastagnoli(t *testing.T) {
	if !archAvailableCastagnoli() {
		t.Skip("Arch-specific Castagnoli not available.")
	}
	archInitCastagnoli()
	slicingTable := slicingMakeTable(Castagnoli)
	testCrossCheck(t, archUpdateCastagnoli, func(crc uint32, b []byte) uint32 {
		return slicingUpdate(crc, slicingTable, b)
	})
}

func TestGolden(t *testing.T) {
	testGoldenIEEE(t, ChecksumIEEE)

	// Some implementations have special code to deal with misaligned
	// data; test that as well.
	for delta := 1; delta <= 7; delta++ {
		testGoldenIEEE(t, func(b []byte) uint32 {
			ieee := NewIEEE()
			d := delta
			if d >= len(b) {
				d = len(b)
			}
			ieee.Write(b[:d])
			ieee.Write(b[d:])
			return ieee.Sum32()
		})
	}

	castagnoliTab := MakeTable(Castagnoli)
	if castagnoliTab == nil {
		t.Errorf("nil Castagnoli Table")
	}

	testGoldenCastagnoli(t, func(b []byte) uint32 {
		castagnoli := New(castagnoliTab)
		castagnoli.Write(b)
		return castagnoli.Sum32()
	})

	// Some implementations have special code to deal with misaligned
	// data; test that as well.
	for delta := 1; delta <= 7; delta++ {
		testGoldenCastagnoli(t, func(b []byte) uint32 {
			castagnoli := New(castagnoliTab)
			d := delta
			if d >= len(b) {
				d = len(b)
			}
			castagnoli.Write(b[:d])
			castagnoli.Write(b[d:])
			return castagnoli.Sum32()
		})
	}
}

func BenchmarkIEEECrc40B(b *testing.B) {
	benchmark(b, NewIEEE(), 40, 0)
}

func BenchmarkIEEECrc1KB(b *testing.B) {
	benchmark(b, NewIEEE(), 1<<10, 0)
}

func BenchmarkIEEECrc4KB(b *testing.B) {
	benchmark(b, NewIEEE(), 4<<10, 0)
}

func BenchmarkIEEECrc32KB(b *testing.B) {
	benchmark(b, NewIEEE(), 32<<10, 0)
}

func BenchmarkCastagnoliCrc15B(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 15, 0)
}

func BenchmarkCastagnoliCrc15BMisaligned(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 15, 1)
}

func BenchmarkCastagnoliCrc40B(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 40, 0)
}

func BenchmarkCastagnoliCrc40BMisaligned(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 40, 1)
}

func BenchmarkCastagnoliCrc512(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 512, 0)
}

func BenchmarkCastagnoliCrc512Misaligned(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 512, 1)
}

func BenchmarkCastagnoliCrc1KB(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 1<<10, 0)
}

func BenchmarkCastagnoliCrc1KBMisaligned(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 1<<10, 1)
}

func BenchmarkCastagnoliCrc4KB(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 4<<10, 0)
}

func BenchmarkCastagnoliCrc4KBMisaligned(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 4<<10, 1)
}

func BenchmarkCastagnoliCrc32KB(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 32<<10, 0)
}

func BenchmarkCastagnoliCrc32KBMisaligned(b *testing.B) {
	benchmark(b, New(MakeTable(Castagnoli)), 32<<10, 1)
}

func benchmark(b *testing.B, h hash.Hash32, n, alignment int64) {
	b.SetBytes(n)
	data := make([]byte, n+alignment)
	data = data[alignment:]
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
