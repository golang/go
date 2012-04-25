// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The trie in this file is used to associate the first full character
// in an UTF-8 string to a collation element.
// All but the last byte in a UTF-8 byte sequence are 
// used to lookup offsets in the index table to be used for the next byte.
// The last byte is used to index into a table of collation elements.
// For a full description, see exp/locale/collate/build/trie.go.

package collate

const blockSize = 64

type trie struct {
	index  []uint16
	values []uint32
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

func (t *trie) lookupValue(n uint16, b byte) colElem {
	return colElem(t.values[int(n)<<6+int(b&maskx)])
}

// lookup returns the trie value for the first UTF-8 encoding in s and
// the width in bytes of this encoding. The size will be 0 if s does not
// hold enough bytes to complete the encoding. len(s) must be greater than 0.
func (t *trie) lookup(s []byte) (v colElem, sz int) {
	c0 := s[0]
	switch {
	case c0 < tx:
		return colElem(t.values[c0]), 1
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
		o := int(i)<<6 + int(c1)&maskx
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
		o := int(i)<<6 + int(c1)&maskx
		i = t.index[o]
		c2 := s[2]
		if c2 < tx || t2 <= c2 {
			return 0, 2
		}
		o = int(i)<<6 + int(c2)&maskx
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
