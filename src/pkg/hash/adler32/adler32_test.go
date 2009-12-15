// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adler32

import (
	"io"
	"testing"
)

type _Adler32Test struct {
	out uint32
	in  string
}

var golden = []_Adler32Test{
	_Adler32Test{0x1, ""},
	_Adler32Test{0x620062, "a"},
	_Adler32Test{0x12600c4, "ab"},
	_Adler32Test{0x24d0127, "abc"},
	_Adler32Test{0x3d8018b, "abcd"},
	_Adler32Test{0x5c801f0, "abcde"},
	_Adler32Test{0x81e0256, "abcdef"},
	_Adler32Test{0xadb02bd, "abcdefg"},
	_Adler32Test{0xe000325, "abcdefgh"},
	_Adler32Test{0x118e038e, "abcdefghi"},
	_Adler32Test{0x158603f8, "abcdefghij"},
	_Adler32Test{0x3f090f02, "Discard medicine more than two years old."},
	_Adler32Test{0x46d81477, "He who has a shady past knows that nice guys finish last."},
	_Adler32Test{0x40ee0ee1, "I wouldn't marry him with a ten foot pole."},
	_Adler32Test{0x16661315, "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave"},
	_Adler32Test{0x5b2e1480, "The days of the digital watch are numbered.  -Tom Stoppard"},
	_Adler32Test{0x8c3c09ea, "Nepal premier won't resign."},
	_Adler32Test{0x45ac18fd, "For every action there is an equal and opposite government program."},
	_Adler32Test{0x53c61462, "His money is twice tainted: 'taint yours and 'taint mine."},
	_Adler32Test{0x7e511e63, "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977"},
	_Adler32Test{0xe4801a6a, "It's a tiny change to the code and not completely disgusting. - Bob Manchek"},
	_Adler32Test{0x61b507df, "size:  a.out:  bad magic"},
	_Adler32Test{0xb8631171, "The major problem is with sendmail.  -Mark Horton"},
	_Adler32Test{0x8b5e1904, "Give me a rock, paper and scissors and I will move the world.  CCFestoon"},
	_Adler32Test{0x7cc6102b, "If the enemy is within range, then so are you."},
	_Adler32Test{0x700318e7, "It's well we cannot hear the screams/That we create in others' dreams."},
	_Adler32Test{0x1e601747, "You remind me of a TV show, but that's all right: I watch it anyway."},
	_Adler32Test{0xb55b0b09, "C is as portable as Stonehedge!!"},
	_Adler32Test{0x39111dd0, "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley"},
	_Adler32Test{0x91dd304f, "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule"},
	_Adler32Test{0x2e5d1316, "How can you write a big system without C++?  -Paul Glick"},
	_Adler32Test{0xd0201df6, "'Invariant assertions' is the most elegant programming technique!  -Tom Szymanski"},
}

func TestGolden(t *testing.T) {
	for i := 0; i < len(golden); i++ {
		g := golden[i]
		c := New()
		io.WriteString(c, g.in)
		s := c.Sum32()
		if s != g.out {
			t.Errorf("adler32(%s) = 0x%x want 0x%x", g.in, s, g.out)
			t.FailNow()
		}
	}
}
