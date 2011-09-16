// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

type valueRange struct {
	value  uint16 // header: value:stride
	lo, hi byte   // header: lo:n
}

type trie struct {
	index        []uint8
	values       []uint16
	sparse       []valueRange
	sparseOffset []uint16
	cutoff       uint8 // indices >= cutoff are sparse
}

// lookupValue determines the type of block n and looks up the value for b.
// For n < t.cutoff, the block is a simple lookup table. Otherwise, the block
// is a list of ranges with an accompanying value. Given a matching range r,
// the value for b is by r.value + (b - r.lo) * stride.
func (t *trie) lookupValue(n uint8, b byte) uint16 {
	if n < t.cutoff {
		return t.values[uint16(n)<<6+uint16(b&maskx)]
	}
	offset := t.sparseOffset[n-t.cutoff]
	header := t.sparse[offset]
	lo := offset + 1
	hi := lo + uint16(header.lo)
	for lo < hi {
		m := lo + (hi-lo)/2
		r := t.sparse[m]
		if r.lo <= b && b <= r.hi {
			return r.value + uint16(b-r.lo)*header.value
		}
		if b < r.lo {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return 0
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
		return t.lookupValue(i, c1), 2
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
		return t.lookupValue(i, c2), 3
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
		return t.lookupValue(i, c3), 4
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
		return t.lookupValue(i, c1), 2
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
		return t.lookupValue(i, c2), 3
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
		return t.lookupValue(i, c3), 4
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
	if c0 < t3 {
		return t.lookupValue(i, s[1])
	}
	i = t.index[uint16(i)<<6+uint16(s[1])&maskx]
	if c0 < t4 {
		return t.lookupValue(i, s[2])
	}
	i = t.index[uint16(i)<<6+uint16(s[2])&maskx]
	if c0 < t5 {
		return t.lookupValue(i, s[3])
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
	if c0 < t3 {
		return t.lookupValue(i, s[1])
	}
	i = t.index[uint16(i)<<6+uint16(s[1])&maskx]
	if c0 < t4 {
		return t.lookupValue(i, s[2])
	}
	i = t.index[uint16(i)<<6+uint16(s[2])&maskx]
	if c0 < t5 {
		return t.lookupValue(i, s[3])
	}
	return 0
}
