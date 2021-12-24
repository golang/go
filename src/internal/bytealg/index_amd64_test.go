// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg_test

import (
	"internal/bytealg"
	"internal/cpu"
	"testing"
)

type BinOpTest struct {
	a string
	b string
	i int
}

var indexTests = []BinOpTest{
	{"012345ab", "ab", 6},
	{"012345ab", "bc", -1},
	// evex prefx match, needle =< 32
	{"xxxxxxxxxxxxxxxxxxxx", "ab", -1},
	{"xxxxxxxxxxxxxxabaaaa", "ab", 14},
	{"abcxxxxxxxxxxxxxxxxx", "abc", 0},
	{"xxxxxxxxxxxxxxxxxxxx", "abcd", -1},
	{"abababababababababab", "abcd", -1},
	{"abababababababxdabxd", "abcd", -1},
	{"abxdabababababcdabxd", "abcd", 12},
	{"abxdabababababddabcd", "abcd", 16},
	{"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "ab", -1},
	{"xabxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "ab", 1},
	{"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "abcd", -1},
	{"abababababababababababxxxxxxxxxxxxxxbaba", "abcd", -1},
	{"ababababababxdabaabxdaxxxxxxxxxxxxxxabxd", "abcd", -1},
	{"ababababababxdabaabcdaxxxxxxxxxxxxxxbaba", "abcd", 17},
	{"ababababababxdabaabxdxxxxxxxxxxxxxxxabcd", "abcd", 36},
	{"ababababababxdabaabxdxxxxxxxxxxxababcdcd", "abcd", 34},
	{"ababababababxdabaabxdxxxxxxxxxxxxxabxdxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxababxdxx", "abcd", -1},
	{"ababababababxdabaabxdxxxxxxxxxxxxxabcdxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxababxdxx", "abcd", 34},
	// len(needle) = 32
	{"ababeeeeeeexeeeeeeeeeeeeeeeeeeeecdxxxxxx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", -1},
	{"ababeeeeeeeeeeeeeeeeeeeeeeeeeeeecdxxxxxx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 2},
	// evex prefix match, needle > 32
	{"ababeeeeeeeeeeeeeexxeeeeeeeeeeeeeeecdxxxxxx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", -1},
	{"ababeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecdxxxxxx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 2},
	{"abxxxxxxabeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 8},
	{"ababeeeeeeeeeeeeeexxeeeeeeeeeeeeeeecdxxxxxxxxxxxxxxxxxxxxxabeeeeeeeeeeeeeexxeeeeeeeeeeeeeeecd", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", -1},
	{"ababeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecdxxxxxxxxxxxxxxxxxxxxxabeeeeeeeeeeeeeexxeeeeeeeeeeeeeeecd", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 2},
	{"ababeeeeeeeeeeeeeexxeeeeeeeeeeeeeeecdxxxxxxxxxxxxxxxxxxxxxabeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 58},
	{"ababeeeeeeeeeeeeeexxeeeeeeeeeeeeeeecdxxxxxxxxxxxxxxxxxxxxxabeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecdx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 58},
	{"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxabeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecdxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 33},
	// len(needle) = 63
	{"xxxxabeeeeeeeeeeeeeeeeeeeeeexeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecdxxx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", -1},
	{"xxxxabeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecdxxx", "abeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecd", 4},
}

func TestIndexSIMDPrefixMatch(t *testing.T) {
	if cpu.X86.HasAVX512VL && cpu.X86.HasAVX512BW {
		for _, test := range indexTests {
			a := []byte(test.a)
			b := []byte(test.b)
			// Valid input length of b is [2, 63]
			actual := bytealg.Index(a, b)
			if actual != test.i {
				t.Errorf("Index(%q,%q) = %v; want %v", a, b, actual, test.i)
			}
		}
	}
}
