// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"encoding"
	"fmt"
	"hash"
	"internal/testhash"
	"io"
	"math/rand"
	"testing"
)

// First test, so that it can be the one to initialize castagnoliTable.
func TestCastagnoliRace(t *testing.T) {
	// The MakeTable(Castagnoli) lazily initializes castagnoliTable,
	// which races with the switch on tab during Write to check
	// whether tab == castagnoliTable.
	ieee := NewIEEE()
	go MakeTable(Castagnoli)
	ieee.Write([]byte("hello"))
}

func TestHashInterface(t *testing.T) {
	testhash.TestHash(t, func() hash.Hash { return NewIEEE() })
}

type test struct {
	ieee, castagnoli    uint32
	in                  string
	halfStateIEEE       string // IEEE marshaled hash state after first half of in written, used by TestGoldenMarshal
	halfStateCastagnoli string // Castagnoli marshaled hash state after first half of in written, used by TestGoldenMarshal
}

var golden = []test{
	{0x0, 0x0, "", "crc\x01ʇ\x91M\x00\x00\x00\x00", "crc\x01wB\x84\x81\x00\x00\x00\x00"},
	{0xe8b7be43, 0xc1d04330, "a", "crc\x01ʇ\x91M\x00\x00\x00\x00", "crc\x01wB\x84\x81\x00\x00\x00\x00"},
	{0x9e83486d, 0xe2a22936, "ab", "crc\x01ʇ\x91M跾C", "crc\x01wB\x84\x81\xc1\xd0C0"},
	{0x352441c2, 0x364b3fb7, "abc", "crc\x01ʇ\x91M跾C", "crc\x01wB\x84\x81\xc1\xd0C0"},
	{0xed82cd11, 0x92c80a31, "abcd", "crc\x01ʇ\x91M\x9e\x83Hm", "crc\x01wB\x84\x81\xe2\xa2)6"},
	{0x8587d865, 0xc450d697, "abcde", "crc\x01ʇ\x91M\x9e\x83Hm", "crc\x01wB\x84\x81\xe2\xa2)6"},
	{0x4b8e39ef, 0x53bceff1, "abcdef", "crc\x01ʇ\x91M5$A\xc2", "crc\x01wB\x84\x816K?\xb7"},
	{0x312a6aa6, 0xe627f441, "abcdefg", "crc\x01ʇ\x91M5$A\xc2", "crc\x01wB\x84\x816K?\xb7"},
	{0xaeef2a50, 0xa9421b7, "abcdefgh", "crc\x01ʇ\x91M\xed\x82\xcd\x11", "crc\x01wB\x84\x81\x92\xc8\n1"},
	{0x8da988af, 0x2ddc99fc, "abcdefghi", "crc\x01ʇ\x91M\xed\x82\xcd\x11", "crc\x01wB\x84\x81\x92\xc8\n1"},
	{0x3981703a, 0xe6599437, "abcdefghij", "crc\x01ʇ\x91M\x85\x87\xd8e", "crc\x01wB\x84\x81\xc4P֗"},
	{0x6b9cdfe7, 0xb2cc01fe, "Discard medicine more than two years old.", "crc\x01ʇ\x91M\xfd\xe5\xc2J", "crc\x01wB\x84\x81S\"(\xe0"},
	{0xc90ef73f, 0xe28207f, "He who has a shady past knows that nice guys finish last.", "crc\x01ʇ\x91M\x01ǋ+", "crc\x01wB\x84\x81'\xdaR\x15"},
	{0xb902341f, 0xbe93f964, "I wouldn't marry him with a ten foot pole.", "crc\x01ʇ\x91M\x9d\x13\xce\x10", "crc\x01wB\x84\x81\xc3\xed\xabG"},
	{0x42080e8, 0x9e3be0c3, "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave", "crc\x01ʇ\x91M-\xed\xf7\x94", "crc\x01wB\x84\x81\xce\xceb\x81"},
	{0x154c6d11, 0xf505ef04, "The days of the digital watch are numbered.  -Tom Stoppard", "crc\x01ʇ\x91MOa\xa5\r", "crc\x01wB\x84\x81\xd3s\x9dP"},
	{0x4c418325, 0x85d3dc82, "Nepal premier won't resign.", "crc\x01ʇ\x91M\xa8S9\x85", "crc\x01wB\x84\x81{\x90\x8a\x14"},
	{0x33955150, 0xc5142380, "For every action there is an equal and opposite government program.", "crc\x01ʇ\x91Ma\xe9>\x86", "crc\x01wB\x84\x81\xaa@\xc4\x1c"},
	{0x26216a4b, 0x75eb77dd, "His money is twice tainted: 'taint yours and 'taint mine.", "crc\x01ʇ\x91M\\\x1an\x88", "crc\x01wB\x84\x81W\a8Z"},
	{0x1abbe45e, 0x91ebe9f7, "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977", "crc\x01ʇ\x91M\xb7\xf5\xf2\xca", "crc\x01wB\x84\x81\xc4o\x9d\x85"},
	{0xc89a94f7, 0xf0b1168e, "It's a tiny change to the code and not completely disgusting. - Bob Manchek", "crc\x01ʇ\x91M\x84g1\xe8", "crc\x01wB\x84\x81#\x98\f\xab"},
	{0xab3abe14, 0x572b74e2, "size:  a.out:  bad magic", "crc\x01ʇ\x91M\x8a\x0f\xad\b", "crc\x01wB\x84\x81\x80\xc9n\xd8"},
	{0xbab102b6, 0x8a58a6d5, "The major problem is with sendmail.  -Mark Horton", "crc\x01ʇ\x91M\a\xf0\xb3\x15", "crc\x01wB\x84\x81liS\xcc"},
	{0x999149d7, 0x9c426c50, "Give me a rock, paper and scissors and I will move the world.  CCFestoon", "crc\x01ʇ\x91M\x0fa\xbc.", "crc\x01wB\x84\x81\xdb͏C"},
	{0x6d52a33c, 0x735400a4, "If the enemy is within range, then so are you.", "crc\x01ʇ\x91My\x1b\x99\xf8", "crc\x01wB\x84\x81\xaaB\x037"},
	{0x90631e8d, 0xbec49c95, "It's well we cannot hear the screams/That we create in others' dreams.", "crc\x01ʇ\x91M\bqfY", "crc\x01wB\x84\x81\x16y\xa1\xd2"},
	{0x78309130, 0xa95a2079, "You remind me of a TV show, but that's all right: I watch it anyway.", "crc\x01ʇ\x91M\xbdO,\xc2", "crc\x01wB\x84\x81f&\xc5\xe4"},
	{0x7d0a377f, 0xde2e65c5, "C is as portable as Stonehedge!!", "crc\x01ʇ\x91M\xf7\xd6\x00\xd5", "crc\x01wB\x84\x81de\\\xf8"},
	{0x8c79fd79, 0x297a88ed, "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley", "crc\x01ʇ\x91Ml+\xb8\xa7", "crc\x01wB\x84\x81\xbf\xd6S\xdd"},
	{0xa20b7167, 0x66ed1d8b, "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule", "crc\x01ʇ\x91M<lR[", "crc\x01wB\x84\x81{\xaco\xb1"},
	{0x8e0bb443, 0xdcded527, "How can you write a big system without C++?  -Paul Glick", "crc\x01ʇ\x91M\x0e\x88\x89\xed", "crc\x01wB\x84\x813\xd7C\u007f"},
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
	lengths := []int{0, 1, 2, 3, 4, 5, 10, 16, 50, 63, 64, 65, 100,
		127, 128, 129, 255, 256, 257, 300, 312, 384, 416, 448, 480,
		500, 501, 502, 503, 504, 505, 512, 513, 1000, 1024, 2000,
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

func TestGoldenMarshal(t *testing.T) {
	t.Run("IEEE", func(t *testing.T) {
		for _, g := range golden {
			h := New(IEEETable)
			h2 := New(IEEETable)

			io.WriteString(h, g.in[:len(g.in)/2])

			state, err := h.(encoding.BinaryMarshaler).MarshalBinary()
			if err != nil {
				t.Errorf("could not marshal: %v", err)
				continue
			}

			stateAppend, err := h.(encoding.BinaryAppender).AppendBinary(make([]byte, 4, 32))
			if err != nil {
				t.Errorf("could not marshal: %v", err)
				continue
			}
			stateAppend = stateAppend[4:]

			if string(state) != g.halfStateIEEE {
				t.Errorf("IEEE(%q) state = %q, want %q", g.in, state, g.halfStateIEEE)
				continue
			}

			if string(stateAppend) != g.halfStateIEEE {
				t.Errorf("IEEE(%q) state = %q, want %q", g.in, stateAppend, g.halfStateIEEE)
				continue
			}

			if err := h2.(encoding.BinaryUnmarshaler).UnmarshalBinary(state); err != nil {
				t.Errorf("could not unmarshal: %v", err)
				continue
			}

			io.WriteString(h, g.in[len(g.in)/2:])
			io.WriteString(h2, g.in[len(g.in)/2:])

			if h.Sum32() != h2.Sum32() {
				t.Errorf("IEEE(%s) = 0x%x != marshaled 0x%x", g.in, h.Sum32(), h2.Sum32())
			}
		}
	})
	t.Run("Castagnoli", func(t *testing.T) {
		table := MakeTable(Castagnoli)
		for _, g := range golden {
			h := New(table)
			h2 := New(table)

			io.WriteString(h, g.in[:len(g.in)/2])

			state, err := h.(encoding.BinaryMarshaler).MarshalBinary()
			if err != nil {
				t.Errorf("could not marshal: %v", err)
				continue
			}

			stateAppend, err := h.(encoding.BinaryAppender).AppendBinary(make([]byte, 4, 32))
			if err != nil {
				t.Errorf("could not marshal: %v", err)
				continue
			}
			stateAppend = stateAppend[4:]

			if string(state) != g.halfStateCastagnoli {
				t.Errorf("Castagnoli(%q) state = %q, want %q", g.in, state, g.halfStateCastagnoli)
				continue
			}

			if string(stateAppend) != g.halfStateCastagnoli {
				t.Errorf("Castagnoli(%q) state = %q, want %q", g.in, stateAppend, g.halfStateCastagnoli)
				continue
			}

			if err := h2.(encoding.BinaryUnmarshaler).UnmarshalBinary(state); err != nil {
				t.Errorf("could not unmarshal: %v", err)
				continue
			}

			io.WriteString(h, g.in[len(g.in)/2:])
			io.WriteString(h2, g.in[len(g.in)/2:])

			if h.Sum32() != h2.Sum32() {
				t.Errorf("Castagnoli(%s) = 0x%x != marshaled 0x%x", g.in, h.Sum32(), h2.Sum32())
			}
		}
	})
}

func TestMarshalTableMismatch(t *testing.T) {
	h1 := New(IEEETable)
	h2 := New(MakeTable(Castagnoli))

	state1, err := h1.(encoding.BinaryMarshaler).MarshalBinary()
	if err != nil {
		t.Errorf("could not marshal: %v", err)
	}

	if err := h2.(encoding.BinaryUnmarshaler).UnmarshalBinary(state1); err == nil {
		t.Errorf("no error when one was expected")
	}
}

// TestSlicing tests the slicing-by-8 algorithm.
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

func BenchmarkCRC32(b *testing.B) {
	b.Run("poly=IEEE", benchmarkAll(NewIEEE()))
	b.Run("poly=Castagnoli", benchmarkAll(New(MakeTable(Castagnoli))))
	b.Run("poly=Koopman", benchmarkAll(New(MakeTable(Koopman))))
}

func benchmarkAll(h hash.Hash32) func(b *testing.B) {
	return func(b *testing.B) {
		for _, size := range []int{15, 40, 512, 1 << 10, 4 << 10, 32 << 10} {
			name := fmt.Sprint(size)
			if size >= 1024 {
				name = fmt.Sprintf("%dkB", size/1024)
			}
			b.Run("size="+name, func(b *testing.B) {
				for align := 0; align <= 1; align++ {
					b.Run(fmt.Sprintf("align=%d", align), func(b *testing.B) {
						benchmark(b, h, int64(size), int64(align))
					})
				}
			})
		}
	}
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
	// Avoid further allocations
	in = in[:0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Reset()
		h.Write(data)
		h.Sum(in)
		in = in[:0]
	}
}
