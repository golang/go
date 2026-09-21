// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adler32

import (
	"encoding"
	"fmt"
	"hash"
	"internal/testhash"
	"io"
	"strings"
	"testing"
)

func TestHashInterface(t *testing.T) {
	testhash.TestHash(t, func() hash.Hash { return New() })
}

var golden = []struct {
	out       uint32
	in        string
	halfState string // marshaled hash state after first half of in written, used by TestGoldenMarshal
}{
	{0x00000001, "", "adl\x01\x00\x00\x00\x01"},
	{0x00620062, "a", "adl\x01\x00\x00\x00\x01"},
	{0x012600c4, "ab", "adl\x01\x00b\x00b"},
	{0x024d0127, "abc", "adl\x01\x00b\x00b"},
	{0x03d8018b, "abcd", "adl\x01\x01&\x00\xc4"},
	{0x05c801f0, "abcde", "adl\x01\x01&\x00\xc4"},
	{0x081e0256, "abcdef", "adl\x01\x02M\x01'"},
	{0x0adb02bd, "abcdefg", "adl\x01\x02M\x01'"},
	{0x0e000325, "abcdefgh", "adl\x01\x03\xd8\x01\x8b"},
	{0x118e038e, "abcdefghi", "adl\x01\x03\xd8\x01\x8b"},
	{0x158603f8, "abcdefghij", "adl\x01\x05\xc8\x01\xf0"},
	{0x3f090f02, "Discard medicine more than two years old.", "adl\x01NU\a\x87"},
	{0x46d81477, "He who has a shady past knows that nice guys finish last.", "adl\x01\x89\x8e\t\xe9"},
	{0x40ee0ee1, "I wouldn't marry him with a ten foot pole.", "adl\x01R\t\ag"},
	{0x16661315, "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave", "adl\x01\u007f\xbb\t\x10"},
	{0x5b2e1480, "The days of the digital watch are numbered.  -Tom Stoppard", "adl\x01\x99:\n~"},
	{0x8c3c09ea, "Nepal premier won't resign.", "adl\x01\"\x05\x05\x05"},
	{0x45ac18fd, "For every action there is an equal and opposite government program.", "adl\x01\xcc\xfa\f\x00"},
	{0x53c61462, "His money is twice tainted: 'taint yours and 'taint mine.", "adl\x01\x93\xa9\n\b"},
	{0x7e511e63, "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977", "adl\x01e\xf5\x10\x14"},
	{0xe4801a6a, "It's a tiny change to the code and not completely disgusting. - Bob Manchek", "adl\x01\xee\x00\f\xb2"},
	{0x61b507df, "size:  a.out:  bad magic", "adl\x01\x1a\xfc\x04\x1d"},
	{0xb8631171, "The major problem is with sendmail.  -Mark Horton", "adl\x01mi\b\xdc"},
	{0x8b5e1904, "Give me a rock, paper and scissors and I will move the world.  CCFestoon", "adl\x01\xe3\n\f\x9f"},
	{0x7cc6102b, "If the enemy is within range, then so are you.", "adl\x01_\xe0\b\x1e"},
	{0x700318e7, "It's well we cannot hear the screams/That we create in others' dreams.", "adl\x01ۘ\f\x87"},
	{0x1e601747, "You remind me of a TV show, but that's all right: I watch it anyway.", "adl\x01\xcc}\v\x83"},
	{0xb55b0b09, "C is as portable as Stonehedge!!", "adl\x01,^\x05\xad"},
	{0x39111dd0, "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley", "adl\x01M\xd1\x0e\xc8"},
	{0x91dd304f, "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule", "adl\x01#\xd8\x17\xd7"},
	{0x2e5d1316, "How can you write a big system without C++?  -Paul Glick", "adl\x01\x8fU\n\x0f"},
	{0xd0201df6, "'Invariant assertions' is the most elegant programming technique!  -Tom Szymanski", "adl\x01/\x98\x0e\xc4"},
	{0x211297c8, strings.Repeat("\xff", 5548) + "8", "adl\x01\x9a\xa6\xcb\xc1"},
	{0xbaa198c8, strings.Repeat("\xff", 5549) + "9", "adl\x01gu\xcc\xc0"},
	{0x553499be, strings.Repeat("\xff", 5550) + "0", "adl\x01gu\xcc\xc0"},
	{0xf0c19abe, strings.Repeat("\xff", 5551) + "1", "adl\x015CͿ"},
	{0x8d5c9bbe, strings.Repeat("\xff", 5552) + "2", "adl\x015CͿ"},
	{0x2af69cbe, strings.Repeat("\xff", 5553) + "3", "adl\x01\x04\x10ξ"},
	{0xc9809dbe, strings.Repeat("\xff", 5554) + "4", "adl\x01\x04\x10ξ"},
	{0x69189ebe, strings.Repeat("\xff", 5555) + "5", "adl\x01\xd3\xcdϽ"},
	{0x86af0001, strings.Repeat("\x00", 1e5), "adl\x01\xc3P\x00\x01"},
	{0x79660b4d, strings.Repeat("a", 1e5), "adl\x01\x81k\x05\xa7"},
	{0x110588ee, strings.Repeat("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 1e4), "adl\x01e\xd2\xc4p"},
}

// checksum is a slow but simple implementation of the Adler-32 checksum.
// It is a straight port of the sample code in RFC 1950 section 9.
func checksum(p []byte) uint32 {
	s1, s2 := uint32(1), uint32(0)
	for _, x := range p {
		s1 = (s1 + uint32(x)) % mod
		s2 = (s2 + s1) % mod
	}
	return s2<<16 | s1
}

func TestGolden(t *testing.T) {
	for _, g := range golden {
		in := g.in
		if len(in) > 220 {
			in = in[:100] + "..." + in[len(in)-100:]
		}
		p := []byte(g.in)
		if got := checksum(p); got != g.out {
			t.Errorf("simple implementation: checksum(%q) = 0x%x want 0x%x", in, got, g.out)
			continue
		}
		if got := Checksum(p); got != g.out {
			t.Errorf("optimized implementation: Checksum(%q) = 0x%x want 0x%x", in, got, g.out)
			continue
		}
	}
}

// TestUpdate cross-checks update against the reference
// implementation across lengths, alignments, initial states, and the
// block boundaries of vectorized implementations.
func TestUpdate(t *testing.T) {
	// Deterministic pseudo-random input.
	data := make([]byte, 2*nmax+256)
	seed := uint32(0x1234567)
	for i := range data {
		seed = seed*1664525 + 1013904223
		data[i] = byte(seed >> 24)
	}

	lengths := []int{
		0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65,
		95, 96, 127, 128, 129, 255, 256, 511, 512, 1024,
		nmax - nmax%16 - 1, nmax - nmax%16, nmax - nmax%16 + 1,
		nmax - nmax%32 - 1, nmax - nmax%32, nmax - nmax%32 + 1,
		nmax - nmax%64 - 1, nmax - nmax%64, nmax - nmax%64 + 1,
		nmax - 1, nmax, nmax + 1, nmax + 31, nmax + 32,
		2*nmax - 1, 2 * nmax, 2*nmax + 17,
	}

	// All-0xff input maximizes the intermediate sums, exercising
	// overflow handling.
	maxData := make([]byte, 2*nmax+256)
	for i := range maxData {
		maxData[i] = 0xff
	}

	// worst is the largest valid initial state: s1 = s2 = mod-1.
	// Together with all-0xff input it maximizes the intermediate
	// sums of an implementation that defers modular reduction.
	const worst = digest((mod-1)<<16 | (mod - 1))

	for _, in := range [][]byte{data, maxData} {
		for _, n := range lengths {
			for _, offset := range []int{0, 1, 7} {
				p := in[offset : offset+n]
				if got, want := uint32(update(1, p)), checksum(p); got != want {
					t.Errorf("update(1, data[%d:%d]) = 0x%x, want 0x%x", offset, offset+n, got, want)
				}
				if got, want := update(worst, p), updateGeneric(worst, p); got != want {
					t.Errorf("update(worst, data[%d:%d]) = 0x%x, want 0x%x", offset, offset+n, uint32(got), uint32(want))
				}
			}
		}
	}

	// Nonzero initial states: check that splitting the input at an
	// arbitrary point gives the same result as a single update.
	p := data[:nmax+1024]
	want := checksum(p)
	for _, split := range []int{1, 15, 16, 63, 255, 4096, nmax - 1, nmax, len(p) - 1} {
		if got := uint32(update(update(1, p[:split]), p[split:])); got != want {
			t.Errorf("update split at %d = 0x%x, want 0x%x", split, got, want)
		}
	}
}

func TestGoldenMarshal(t *testing.T) {
	for _, g := range golden {
		h := New()
		h2 := New()

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

		if string(state) != g.halfState {
			t.Errorf("checksum(%q) state = %q, want %q", g.in, state, g.halfState)
			continue
		}

		if string(stateAppend) != g.halfState {
			t.Errorf("checksum(%q) state = %q, want %q", g.in, stateAppend, g.halfState)
			continue
		}

		if err := h2.(encoding.BinaryUnmarshaler).UnmarshalBinary(state); err != nil {
			t.Errorf("could not unmarshal: %v", err)
			continue
		}

		io.WriteString(h, g.in[len(g.in)/2:])
		io.WriteString(h2, g.in[len(g.in)/2:])

		if h.Sum32() != h2.Sum32() {
			t.Errorf("checksum(%q) = 0x%x != marshaled (0x%x)", g.in, h.Sum32(), h2.Sum32())
		}
	}
}

func BenchmarkAdler32(b *testing.B) {
	sizes := []int{16, 64, 256, 1 << 10, 4 << 10, 32 << 10, 512 << 10}
	for _, size := range sizes {
		data := make([]byte, size)
		for i := range data {
			data[i] = byte(i)
		}
		name := fmt.Sprintf("%dB", size)
		if size >= 1<<10 {
			name = fmt.Sprintf("%dKB", size>>10)
		}
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(size))
			for b.Loop() {
				Checksum(data)
			}
		})
	}
}

func BenchmarkAdler32KB(b *testing.B) {
	b.SetBytes(1024)
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i)
	}
	h := New()
	in := make([]byte, 0, h.Size())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Reset()
		h.Write(data)
		h.Sum(in)
	}
}
