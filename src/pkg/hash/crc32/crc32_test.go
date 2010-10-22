// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"io"
	"testing"
)

type test struct {
	out uint32
	in  string
}

var golden = []test{
	{0x0, ""},
	{0xe8b7be43, "a"},
	{0x9e83486d, "ab"},
	{0x352441c2, "abc"},
	{0xed82cd11, "abcd"},
	{0x8587d865, "abcde"},
	{0x4b8e39ef, "abcdef"},
	{0x312a6aa6, "abcdefg"},
	{0xaeef2a50, "abcdefgh"},
	{0x8da988af, "abcdefghi"},
	{0x3981703a, "abcdefghij"},
	{0x6b9cdfe7, "Discard medicine more than two years old."},
	{0xc90ef73f, "He who has a shady past knows that nice guys finish last."},
	{0xb902341f, "I wouldn't marry him with a ten foot pole."},
	{0x42080e8, "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave"},
	{0x154c6d11, "The days of the digital watch are numbered.  -Tom Stoppard"},
	{0x4c418325, "Nepal premier won't resign."},
	{0x33955150, "For every action there is an equal and opposite government program."},
	{0x26216a4b, "His money is twice tainted: 'taint yours and 'taint mine."},
	{0x1abbe45e, "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977"},
	{0xc89a94f7, "It's a tiny change to the code and not completely disgusting. - Bob Manchek"},
	{0xab3abe14, "size:  a.out:  bad magic"},
	{0xbab102b6, "The major problem is with sendmail.  -Mark Horton"},
	{0x999149d7, "Give me a rock, paper and scissors and I will move the world.  CCFestoon"},
	{0x6d52a33c, "If the enemy is within range, then so are you."},
	{0x90631e8d, "It's well we cannot hear the screams/That we create in others' dreams."},
	{0x78309130, "You remind me of a TV show, but that's all right: I watch it anyway."},
	{0x7d0a377f, "C is as portable as Stonehedge!!"},
	{0x8c79fd79, "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley"},
	{0xa20b7167, "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule"},
	{0x8e0bb443, "How can you write a big system without C++?  -Paul Glick"},
}

func TestGolden(t *testing.T) {
	for i := 0; i < len(golden); i++ {
		g := golden[i]
		c := NewIEEE()
		io.WriteString(c, g.in)
		s := c.Sum32()
		if s != g.out {
			t.Errorf("crc32(%s) = 0x%x want 0x%x", g.in, s, g.out)
			t.FailNow()
		}
	}
}

func BenchmarkCrc32KB(b *testing.B) {
	b.StopTimer()
	data := make([]uint8, 1024)
	for i := 0; i < 1024; i++ {
		data[i] = uint8(i)
	}
	c := NewIEEE()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		c.Write(data)
	}
}
