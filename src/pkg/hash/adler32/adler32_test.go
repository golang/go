// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adler32

import (
	"strings"
	"testing"
)

var golden = []struct {
	out uint32
	in  string
}{
	{0x00000001, ""},
	{0x00620062, "a"},
	{0x012600c4, "ab"},
	{0x024d0127, "abc"},
	{0x03d8018b, "abcd"},
	{0x05c801f0, "abcde"},
	{0x081e0256, "abcdef"},
	{0x0adb02bd, "abcdefg"},
	{0x0e000325, "abcdefgh"},
	{0x118e038e, "abcdefghi"},
	{0x158603f8, "abcdefghij"},
	{0x3f090f02, "Discard medicine more than two years old."},
	{0x46d81477, "He who has a shady past knows that nice guys finish last."},
	{0x40ee0ee1, "I wouldn't marry him with a ten foot pole."},
	{0x16661315, "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave"},
	{0x5b2e1480, "The days of the digital watch are numbered.  -Tom Stoppard"},
	{0x8c3c09ea, "Nepal premier won't resign."},
	{0x45ac18fd, "For every action there is an equal and opposite government program."},
	{0x53c61462, "His money is twice tainted: 'taint yours and 'taint mine."},
	{0x7e511e63, "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977"},
	{0xe4801a6a, "It's a tiny change to the code and not completely disgusting. - Bob Manchek"},
	{0x61b507df, "size:  a.out:  bad magic"},
	{0xb8631171, "The major problem is with sendmail.  -Mark Horton"},
	{0x8b5e1904, "Give me a rock, paper and scissors and I will move the world.  CCFestoon"},
	{0x7cc6102b, "If the enemy is within range, then so are you."},
	{0x700318e7, "It's well we cannot hear the screams/That we create in others' dreams."},
	{0x1e601747, "You remind me of a TV show, but that's all right: I watch it anyway."},
	{0xb55b0b09, "C is as portable as Stonehedge!!"},
	{0x39111dd0, "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley"},
	{0x91dd304f, "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule"},
	{0x2e5d1316, "How can you write a big system without C++?  -Paul Glick"},
	{0xd0201df6, "'Invariant assertions' is the most elegant programming technique!  -Tom Szymanski"},
	{0x211297c8, strings.Repeat("\xff", 5548) + "8"},
	{0xbaa198c8, strings.Repeat("\xff", 5549) + "9"},
	{0x553499be, strings.Repeat("\xff", 5550) + "0"},
	{0xf0c19abe, strings.Repeat("\xff", 5551) + "1"},
	{0x8d5c9bbe, strings.Repeat("\xff", 5552) + "2"},
	{0x2af69cbe, strings.Repeat("\xff", 5553) + "3"},
	{0xc9809dbe, strings.Repeat("\xff", 5554) + "4"},
	{0x69189ebe, strings.Repeat("\xff", 5555) + "5"},
	{0x86af0001, strings.Repeat("\x00", 1e5)},
	{0x79660b4d, strings.Repeat("a", 1e5)},
	{0x110588ee, strings.Repeat("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 1e4)},
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
