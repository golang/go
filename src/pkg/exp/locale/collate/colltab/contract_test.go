// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colltab

import (
	"testing"
)

type lookupStrings struct {
	str    string
	offset int
	n      int // bytes consumed from input
}

type LookupTest struct {
	lookup []lookupStrings
	n      int
	tries  contractTrieSet
}

var lookupTests = []LookupTest{
	{[]lookupStrings{
		{"abc", 1, 3},
		{"a", 0, 0},
		{"b", 0, 0},
		{"c", 0, 0},
		{"d", 0, 0},
	},
		1,
		contractTrieSet{
			{'a', 0, 1, 0xFF},
			{'b', 0, 1, 0xFF},
			{'c', 'c', 0, 1},
		},
	},
	{[]lookupStrings{
		{"abc", 1, 3},
		{"abd", 2, 3},
		{"abe", 3, 3},
		{"a", 0, 0},
		{"ab", 0, 0},
		{"d", 0, 0},
		{"f", 0, 0},
	},
		1,
		contractTrieSet{
			{'a', 0, 1, 0xFF},
			{'b', 0, 1, 0xFF},
			{'c', 'e', 0, 1},
		},
	},
	{[]lookupStrings{
		{"abc", 1, 3},
		{"ab", 2, 2},
		{"a", 3, 1},
		{"abcd", 1, 3},
		{"abe", 2, 2},
	},
		1,
		contractTrieSet{
			{'a', 0, 1, 3},
			{'b', 0, 1, 2},
			{'c', 'c', 0, 1},
		},
	},
	{[]lookupStrings{
		{"abc", 1, 3},
		{"abd", 2, 3},
		{"ab", 3, 2},
		{"ac", 4, 2},
		{"a", 5, 1},
		{"b", 6, 1},
		{"ba", 6, 1},
	},
		2,
		contractTrieSet{
			{'b', 'b', 0, 6},
			{'a', 0, 2, 5},
			{'c', 'c', 0, 4},
			{'b', 0, 1, 3},
			{'c', 'd', 0, 1},
		},
	},
	{[]lookupStrings{
		{"bcde", 2, 4},
		{"bc", 7, 2},
		{"ab", 6, 2},
		{"bcd", 5, 3},
		{"abcd", 1, 4},
		{"abc", 4, 3},
		{"bcdf", 3, 4},
	},
		2,
		contractTrieSet{
			{'b', 3, 1, 0xFF},
			{'a', 0, 1, 0xFF},
			{'b', 0, 1, 6},
			{'c', 0, 1, 4},
			{'d', 'd', 0, 1},
			{'c', 0, 1, 7},
			{'d', 0, 1, 5},
			{'e', 'f', 0, 2},
		},
	},
}

func lookup(c *contractTrieSet, nnode int, s []uint8) (i, n int) {
	scan := c.scanner(0, nnode, s)
	scan.scan(0)
	return scan.result()
}

func TestLookupContraction(t *testing.T) {
	for i, tt := range lookupTests {
		cts := contractTrieSet(tt.tries)
		for j, lu := range tt.lookup {
			str := lu.str
			for _, s := range []string{str, str + "X"} {
				const msg = `%d:%d: %s of "%s" %v; want %v`
				offset, n := lookup(&cts, tt.n, []byte(s))
				if offset != lu.offset {
					t.Errorf(msg, i, j, "offset", s, offset, lu.offset)
				}
				if n != lu.n {
					t.Errorf(msg, i, j, "bytes consumed", s, n, len(str))
				}
			}
		}
	}
}
