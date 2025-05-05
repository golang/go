// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc64

import (
	"encoding"
	"hash"
	"internal/testhash"
	"io"
	"testing"
)

func TestCRC64Hash(t *testing.T) {
	testhash.TestHash(t, func() hash.Hash { return New(MakeTable(ISO)) })
}

type test struct {
	outISO        uint64
	outECMA       uint64
	in            string
	halfStateISO  string // ISO marshaled hash state after first half of in written, used by TestGoldenMarshal
	halfStateECMA string // ECMA marshaled hash state after first half of in written, used by TestGoldenMarshal
}

var golden = []test{
	{0x0, 0x0, "", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\x00\x00\x00\x00\x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee\x00\x00\x00\x00\x00\x00\x00\x00"},
	{0x3420000000000000, 0x330284772e652b05, "a", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\x00\x00\x00\x00\x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee\x00\x00\x00\x00\x00\x00\x00\x00"},
	{0x36c4200000000000, 0xbc6573200e84b046, "ab", "crc\x02s\xba\x84\x84\xbb\xcd]\xef4 \x00\x00\x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee3\x02\x84w.e+\x05"},
	{0x3776c42000000000, 0x2cd8094a1a277627, "abc", "crc\x02s\xba\x84\x84\xbb\xcd]\xef4 \x00\x00\x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee3\x02\x84w.e+\x05"},
	{0x336776c420000000, 0x3c9d28596e5960ba, "abcd", "crc\x02s\xba\x84\x84\xbb\xcd]\xef6\xc4 \x00\x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee\xbces \x0e\x84\xb0F"},
	{0x32d36776c4200000, 0x40bdf58fb0895f2, "abcde", "crc\x02s\xba\x84\x84\xbb\xcd]\xef6\xc4 \x00\x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee\xbces \x0e\x84\xb0F"},
	{0x3002d36776c42000, 0xd08e9f8545a700f4, "abcdef", "crc\x02s\xba\x84\x84\xbb\xcd]\xef7v\xc4 \x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee,\xd8\tJ\x1a'v'"},
	{0x31b002d36776c420, 0xec20a3a8cc710e66, "abcdefg", "crc\x02s\xba\x84\x84\xbb\xcd]\xef7v\xc4 \x00\x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee,\xd8\tJ\x1a'v'"},
	{0xe21b002d36776c4, 0x67b4f30a647a0c59, "abcdefgh", "crc\x02s\xba\x84\x84\xbb\xcd]\xef3gv\xc4 \x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee<\x9d(YnY`\xba"},
	{0x8b6e21b002d36776, 0x9966f6c89d56ef8e, "abcdefghi", "crc\x02s\xba\x84\x84\xbb\xcd]\xef3gv\xc4 \x00\x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee<\x9d(YnY`\xba"},
	{0x7f5b6e21b002d367, 0x32093a2ecd5773f4, "abcdefghij", "crc\x02s\xba\x84\x84\xbb\xcd]\xef2\xd3gv\xc4 \x00\x00", "crc\x02`&\x9aR\xe1\xb7\xfee\x04\v\xdfX\xfb\b\x95\xf2"},
	{0x8ec0e7c835bf9cdf, 0x8a0825223ea6d221, "Discard medicine more than two years old.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\xc6\xc0\f\xac'\x11\x12\xd5", "crc\x02`&\x9aR\xe1\xb7\xfee\xfd%\xc0&\xa0R\xef\x95"},
	{0xc7db1759e2be5ab4, 0x8562c0ac2ab9a00d, "He who has a shady past knows that nice guys finish last.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\t\xcb\xd15X[r\t", "crc\x02`&\x9aR\xe1\xb7\xfee\a\x02\xe8|+\xc1\x06\xe3"},
	{0xfbf9d9603a6fa020, 0x3ee2a39c083f38b4, "I wouldn't marry him with a ten foot pole.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\x19\xc8d\xbe\x84\x14\x87_", "crc\x02`&\x9aR\xe1\xb7\xfee˷\xd3\xeeG\xdcE\x8c"},
	{0xeafc4211a6daa0ef, 0x1f603830353e518a, "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\xad\x1b*\xc0\xb1\xf3i(", "crc\x02`&\x9aR\xe1\xb7\xfee\xa7\x8a\xdb\xf6\xd2R\t\x96"},
	{0x3e05b21c7a4dc4da, 0x2fd681d7b2421fd, "The days of the digital watch are numbered.  -Tom Stoppard", "crc\x02s\xba\x84\x84\xbb\xcd]\xefv78\x1ak\x02\x8f\xff", "crc\x02`&\x9aR\xe1\xb7\xfeeT\xcbl\x10\xfb\x87K*"},
	{0x5255866ad6ef28a6, 0x790ef2b16a745a41, "Nepal premier won't resign.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\xcbf\x11R\xbfh\xde\xc9", "crc\x02`&\x9aR\xe1\xb7\xfee6\x13ُ\x06_\xbd\x9a"},
	{0x8a79895be1e9c361, 0x3ef8f06daccdcddf, "For every action there is an equal and opposite government program.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\xf3pV\x01c_Wu", "crc\x02`&\x9aR\xe1\xb7\xfee\xe7\xc6\n\b\x12FL\xa0"},
	{0x8878963a649d4916, 0x49e41b2660b106d, "His money is twice tainted: 'taint yours and 'taint mine.", "crc\x02s\xba\x84\x84\xbb\xcd]\xefñ\xff\xf1\xe0/Δ", "crc\x02`&\x9aR\xe1\xb7\xfeeOL/\xb1\xec\xa2\x14\x87"},
	{0xa7b9d53ea87eb82f, 0x561cc0cfa235ac68, "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977", "crc\x02s\xba\x84\x84\xbb\xcd]\xefݸa\xe1\xb5\xf8\xb9W", "crc\x02`&\x9aR\xe1\xb7\xfee\x87)GQ\x03\xf4K\t"},
	{0xdb6805c0966a2f9c, 0xd4fe9ef082e69f59, "It's a tiny change to the code and not completely disgusting. - Bob Manchek", "crc\x02s\xba\x84\x84\xbb\xcd]\xefV\xba\x12\x91\x81\x1fNU", "crc\x02`&\x9aR\xe1\xb7\xfee\n\xb8\x81v?\xdeL\xcb"},
	{0xf3553c65dacdadd2, 0xe3b5e46cd8d63a4d, "size:  a.out:  bad magic", "crc\x02s\xba\x84\x84\xbb\xcd]\xefG\xad\xbc\xb2\xa8y\xc9\xdc", "crc\x02`&\x9aR\xe1\xb7\xfee\xcc\xce\xe5\xe6\x89p\x01\xb8"},
	{0x9d5e034087a676b9, 0x865aaf6b94f2a051, "The major problem is with sendmail.  -Mark Horton", "crc\x02s\xba\x84\x84\xbb\xcd]\uf8acn\x8aT;&\xd5", "crc\x02`&\x9aR\xe1\xb7\xfeeFf\x9c\x1f\xc9x\xbfa"},
	{0xa6db2d7f8da96417, 0x7eca10d2f8136eb4, "Give me a rock, paper and scissors and I will move the world.  CCFestoon", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\xeb\x18\xbf\xf9}\x91\xe5|", "crc\x02`&\x9aR\xe1\xb7\xfeea\x9e\x05:\xce[\xe7\x19"},
	{0x325e00cd2fe819f9, 0xd7dd118c98e98727, "If the enemy is within range, then so are you.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef^5k\xd0Aj_{", "crc\x02`&\x9aR\xe1\xb7\xfee\v#\x99\xa8r\x83YR"},
	{0x88c6600ce58ae4c6, 0x70fb33c119c29318, "It's well we cannot hear the screams/That we create in others' dreams.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef|\xb5\x02\xdcw\x18/\x86", "crc\x02`&\x9aR\xe1\xb7\xfee]\x9d-\xed\x8c\xf9r9"},
	{0x28c4a3f3b769e078, 0x57c891e39a97d9b7, "You remind me of a TV show, but that's all right: I watch it anyway.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\x03\x8bd\x1c\xb0_\x16\x98", "crc\x02`&\x9aR\xe1\xb7\xfee\xafW\x98\xaa\"\xe7\xd7|"},
	{0xa698a34c9d9f1dca, 0xa1f46ba20ad06eb7, "C is as portable as Stonehedge!!", "crc\x02s\xba\x84\x84\xbb\xcd]\xef.P\xe1I\xc6pi\xdc", "crc\x02`&\x9aR\xe1\xb7\xfee֚\x06\x01(\xc0\x1e\x8b"},
	{0xf6c1e2a8c26c5cfc, 0x7ad25fafa1710407, "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\xf7\xa04\x8a\xf2o\xe0;", "crc\x02`&\x9aR\xe1\xb7\xfee<[\xd2%\x9em\x94\x04"},
	{0xd402559dfe9b70c, 0x73cef1666185c13f, "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule", "crc\x02s\xba\x84\x84\xbb\xcd]\xef\u007f\xae\xb9\xbaX=\x19v", "crc\x02`&\x9aR\xe1\xb7\xfee\xb2˦Y\xc5\xd0G\x03"},
	{0xdb6efff26aa94946, 0xb41858f73c389602, "How can you write a big system without C++?  -Paul Glick", "crc\x02s\xba\x84\x84\xbb\xcd]\xefa\xed$js\xb9\xa5A", "crc\x02`&\x9aR\xe1\xb7\xfeeZm\x96\x8a\xe2\xaf\x13p"},
	{0xe7fcf1006b503b61, 0x27db187fc15bbc72, "This is a test of the emergency broadcast system.", "crc\x02s\xba\x84\x84\xbb\xcd]\xef}\xee[q\x16\xcb\xe4\x8d", "crc\x02`&\x9aR\xe1\xb7\xfee\xb1\x93] \xeb\xa9am"},
}

func TestGolden(t *testing.T) {
	tabISO := MakeTable(ISO)
	tabECMA := MakeTable(ECMA)
	for i := 0; i < len(golden); i++ {
		g := golden[i]
		c := New(tabISO)
		io.WriteString(c, g.in)
		s := c.Sum64()
		if s != g.outISO {
			t.Fatalf("ISO crc64(%s) = 0x%x want 0x%x", g.in, s, g.outISO)
		}
		c = New(tabECMA)
		io.WriteString(c, g.in)
		s = c.Sum64()
		if s != g.outECMA {
			t.Fatalf("ECMA crc64(%s) = 0x%x want 0x%x", g.in, s, g.outECMA)
		}
	}
}

func TestGoldenMarshal(t *testing.T) {
	t.Run("ISO", func(t *testing.T) {
		table := MakeTable(ISO)
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

			if string(state) != g.halfStateISO {
				t.Errorf("ISO crc64(%q) state = %q, want %q", g.in, state, g.halfStateISO)
				continue
			}

			if string(stateAppend) != g.halfStateISO {
				t.Errorf("ISO crc64(%q) state = %q, want %q", g.in, stateAppend, g.halfStateISO)
				continue
			}

			if err := h2.(encoding.BinaryUnmarshaler).UnmarshalBinary(state); err != nil {
				t.Errorf("could not unmarshal: %v", err)
				continue
			}

			io.WriteString(h, g.in[len(g.in)/2:])
			io.WriteString(h2, g.in[len(g.in)/2:])

			if h.Sum64() != h2.Sum64() {
				t.Errorf("ISO crc64(%s) = 0x%x != marshaled (0x%x)", g.in, h.Sum64(), h2.Sum64())
			}
		}
	})
	t.Run("ECMA", func(t *testing.T) {
		table := MakeTable(ECMA)
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

			if string(state) != g.halfStateECMA {
				t.Errorf("ECMA crc64(%q) state = %q, want %q", g.in, state, g.halfStateECMA)
				continue
			}

			if string(stateAppend) != g.halfStateECMA {
				t.Errorf("ECMA crc64(%q) state = %q, want %q", g.in, stateAppend, g.halfStateECMA)
				continue
			}

			if err := h2.(encoding.BinaryUnmarshaler).UnmarshalBinary(state); err != nil {
				t.Errorf("could not unmarshal: %v", err)
				continue
			}

			io.WriteString(h, g.in[len(g.in)/2:])
			io.WriteString(h2, g.in[len(g.in)/2:])

			if h.Sum64() != h2.Sum64() {
				t.Errorf("ECMA crc64(%s) = 0x%x != marshaled (0x%x)", g.in, h.Sum64(), h2.Sum64())
			}
		}
	})
}

func TestMarshalTableMismatch(t *testing.T) {
	h1 := New(MakeTable(ISO))
	h2 := New(MakeTable(ECMA))

	state1, err := h1.(encoding.BinaryMarshaler).MarshalBinary()
	if err != nil {
		t.Errorf("could not marshal: %v", err)
	}

	if err := h2.(encoding.BinaryUnmarshaler).UnmarshalBinary(state1); err == nil {
		t.Errorf("no error when one was expected")
	}
}

func bench(b *testing.B, poly uint64, size int64) {
	b.SetBytes(size)
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i)
	}
	h := New(MakeTable(poly))
	in := make([]byte, 0, h.Size())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Reset()
		h.Write(data)
		h.Sum(in)
	}
}

func BenchmarkCrc64(b *testing.B) {
	b.Run("ISO64KB", func(b *testing.B) {
		bench(b, ISO, 64<<10)
	})
	b.Run("ISO4KB", func(b *testing.B) {
		bench(b, ISO, 4<<10)
	})
	b.Run("ISO1KB", func(b *testing.B) {
		bench(b, ISO, 1<<10)
	})
	b.Run("ECMA64KB", func(b *testing.B) {
		bench(b, ECMA, 64<<10)
	})
	b.Run("Random64KB", func(b *testing.B) {
		bench(b, 0x777, 64<<10)
	})
	b.Run("Random16KB", func(b *testing.B) {
		bench(b, 0x777, 16<<10)
	})
}
