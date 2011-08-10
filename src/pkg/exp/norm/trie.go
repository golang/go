// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

type trie struct {
	index  []uint8
	values []uint16
}

const (
	t1 = 0x00 // 0000 0000
	tx = 0x80 // 1000 0000
	t2 = 0xC0 // 1100 0000
	t3 = 0xE0 // 1110 0000
	t4 = 0xF0 // 1111 0000
	t5 = 0xF8 // 1111 1000
	t6 = 0xFC // 1111 1100
	te = 0xFE // 1111 1110

	maskx = 0x3F // 0011 1111
	mask2 = 0x1F // 0001 1111
	mask3 = 0x0F // 0000 1111
	mask4 = 0x07 // 0000 0111
)

// lookup returns the trie value for the first UTF-8 encoding in s and
// the width in bytes of this encoding. The size will be 0 if s does not
// hold enough bytes to complete the encoding. len(s) must be greater than 0.
func (t *trie) lookup(s []byte) (v uint16, sz int) {
	c0 := s[0]
	switch {
	case c0 < tx:
		return t.values[c0], 1
	case c0 < t2:
		return 0, 1
	case c0 < t3:
		if len(s) < 2 {
			return 0, 0
		}
		i := t.index[c0]
		c1 := s[1]
		if c1 < tx || t2 <= c1 {
			return 0, 1
		}
		o := uint16(i)<<6 + uint16(c1)&maskx
		return t.values[o], 2
	case c0 < t4:
		if len(s) < 3 {
			return 0, 0
		}
		i := t.index[c0]
		c1 := s[1]
		if c1 < tx || t2 <= c1 {
			return 0, 1
		}
		o := uint16(i)<<6 + uint16(c1)&maskx
		i = t.index[o]
		c2 := s[2]
		if c2 < tx || t2 <= c2 {
			return 0, 2
		}
		o = uint16(i)<<6 + uint16(c2)&maskx
		return t.values[o], 3
	case c0 < t5:
		if len(s) < 4 {
			return 0, 0
		}
		i := t.index[c0]
		c1 := s[1]
		if c1 < tx || t2 <= c1 {
			return 0, 1
		}
		o := uint16(i)<<6 + uint16(c1)&maskx
		i = t.index[o]
		c2 := s[2]
		if c2 < tx || t2 <= c2 {
			return 0, 2
		}
		o = uint16(i)<<6 + uint16(c2)&maskx
		i = t.index[o]
		c3 := s[3]
		if c3 < tx || t2 <= c3 {
			return 0, 3
		}
		o = uint16(i)<<6 + uint16(c3)&maskx
		return t.values[o], 4
	case c0 < t6:
		if len(s) < 5 {
			return 0, 0
		}
		return 0, 5
	case c0 < te:
		if len(s) < 6 {
			return 0, 0
		}
		return 0, 6
	}
	// Illegal rune
	return 0, 1
}

// lookupString returns the trie value for the first UTF-8 encoding in s and
// the width in bytes of this encoding. The size will be 0 if s does not
// hold enough bytes to complete the encoding. len(s) must be greater than 0.
func (t *trie) lookupString(s string) (v uint16, sz int) {
	c0 := s[0]
	switch {
	case c0 < tx:
		return t.values[c0], 1
	case c0 < t2:
		return 0, 1
	case c0 < t3:
		if len(s) < 2 {
			return 0, 0
		}
		i := t.index[c0]
		c1 := s[1]
		if c1 < tx || t2 <= c1 {
			return 0, 1
		}
		o := uint16(i)<<6 + uint16(c1)&maskx
		return t.values[o], 2
	case c0 < t4:
		if len(s) < 3 {
			return 0, 0
		}
		i := t.index[c0]
		c1 := s[1]
		if c1 < tx || t2 <= c1 {
			return 0, 1
		}
		o := uint16(i)<<6 + uint16(c1)&maskx
		i = t.index[o]
		c2 := s[2]
		if c2 < tx || t2 <= c2 {
			return 0, 2
		}
		o = uint16(i)<<6 + uint16(c2)&maskx
		return t.values[o], 3
	case c0 < t5:
		if len(s) < 4 {
			return 0, 0
		}
		i := t.index[c0]
		c1 := s[1]
		if c1 < tx || t2 <= c1 {
			return 0, 1
		}
		o := uint16(i)<<6 + uint16(c1)&maskx
		i = t.index[o]
		c2 := s[2]
		if c2 < tx || t2 <= c2 {
			return 0, 2
		}
		o = uint16(i)<<6 + uint16(c2)&maskx
		i = t.index[o]
		c3 := s[3]
		if c3 < tx || t2 <= c3 {
			return 0, 3
		}
		o = uint16(i)<<6 + uint16(c3)&maskx
		return t.values[o], 4
	case c0 < t6:
		if len(s) < 5 {
			return 0, 0
		}
		return 0, 5
	case c0 < te:
		if len(s) < 6 {
			return 0, 0
		}
		return 0, 6
	}
	// Illegal rune
	return 0, 1
}

// lookupUnsafe returns the trie value for the first UTF-8 encoding in s.
// s must hold a full encoding.
func (t *trie) lookupUnsafe(s []byte) uint16 {
	c0 := s[0]
	if c0 < tx {
		return t.values[c0]
	}
	if c0 < t2 {
		return 0
	}
	i := t.index[c0]
	o := uint16(i)<<6 + uint16(s[1])&maskx
	if c0 < t3 {
		return t.values[o]
	}
	i = t.index[o]
	o = uint16(i)<<6 + uint16(s[2])&maskx
	if c0 < t4 {
		return t.values[o]
	}
	i = t.index[o]
	o = uint16(i)<<6 + uint16(s[3])&maskx
	if c0 < t5 {
		return t.values[o]
	}
	return 0
}

// lookupStringUnsafe returns the trie value for the first UTF-8 encoding in s.
// s must hold a full encoding.
func (t *trie) lookupStringUnsafe(s string) uint16 {
	c0 := s[0]
	if c0 < tx {
		return t.values[c0]
	}
	if c0 < t2 {
		return 0
	}
	i := t.index[c0]
	o := uint16(i)<<6 + uint16(s[1])&maskx
	if c0 < t3 {
		return t.values[o]
	}
	i = t.index[o]
	o = uint16(i)<<6 + uint16(s[2])&maskx
	if c0 < t4 {
		return t.values[o]
	}
	i = t.index[o]
	o = uint16(i)<<6 + uint16(s[3])&maskx
	if c0 < t5 {
		return t.values[o]
	}
	return 0
}
