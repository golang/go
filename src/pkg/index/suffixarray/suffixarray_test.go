// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package suffixarray

import (
	"container/vector"
	"sort"
	"strings"
	"testing"
)


type testCase struct {
	name    string   // name of test case
	source  string   // source to index
	lookups []string // strings to lookup
}


var testCases = []testCase{
	{
		"empty string",
		"",
		[]string{
			"",
			"foo",
		},
	},

	{
		"all a's",
		"aaaaaaaaaa", // 10 a's
		[]string{
			"",
			"a",
			"aa",
			"aaa",
			"aaaa",
			"aaaaa",
			"aaaaaa",
			"aaaaaaa",
			"aaaaaaaa",
			"aaaaaaaaa",
			"aaaaaaaaaa",
			"aaaaaaaaaaa", // 11 a's
		},
	},

	{
		"abc",
		"abc",
		[]string{
			"a",
			"b",
			"c",
			"ab",
			"bc",
			"abc",
		},
	},

	{
		"barbara*3",
		"barbarabarbarabarbara",
		[]string{
			"a",
			"bar",
			"rab",
			"arab",
			"barbar",
		},
	},

	{
		"typing drill",
		"Now is the time for all good men to come to the aid of their country.",
		[]string{
			"Now",
			"the time",
			"to come the aid",
			"is the time for all good men to come to the aid of their",
		},
	},
}


// find all occurences of s in source; report at most n occurences
func find(src, s string, n int) []int {
	var res vector.IntVector
	if s != "" && n != 0 {
		// find at most n occurences of s in src
		for i := -1; n < 0 || len(res) < n; {
			j := strings.Index(src[i+1:], s)
			if j < 0 {
				break
			}
			i += j + 1
			res.Push(i)
		}
	}
	return res
}


func testLookups(t *testing.T, src string, x *Index, tc *testCase, n int) {
	for _, s := range tc.lookups {
		res := x.Lookup([]byte(s), n)
		exp := find(tc.source, s, n)

		// check that the lengths match
		if len(res) != len(exp) {
			t.Errorf("test %q, lookup %q (n = %d): expected %d results; got %d", tc.name, s, n, len(exp), len(res))
		}

		// if n >= 0 the number of results is limited --- unless n >= all results,
		// we may obtain different positions from the Index and from find (because
		// Index may not find the results in the same order as find) => in general
		// we cannot simply check that the res and exp lists are equal

		// check that there are no duplicates
		sort.SortInts(res)
		for i, r := range res {
			if i > 0 && res[i-1] == r {
				t.Errorf("test %q, lookup %q, result %d (n = %d): found duplicate index %d", tc.name, s, i, n, r)
			}
		}

		// check that each result is in fact a correct match
		for i, r := range res {
			if r < 0 || len(src) <= r {
				t.Errorf("test %q, lookup %q, result %d (n = %d): index %d out of range [0, %d[", tc.name, s, i, n, r, len(src))
			} else if !strings.HasPrefix(src[r:], s) {
				t.Errorf("test %q, lookup %q, result %d (n = %d): index %d not a match", tc.name, s, i, n, r)
			}
		}

		if n < 0 {
			// all results computed - sorted res and exp must be equal
			for i, r := range res {
				e := exp[i]
				if r != e {
					t.Errorf("test %q, lookup %q, result %d: expected index %d; got %d", tc.name, s, i, e, r)
					continue
				}
			}
		}
	}
}


func TestIndex(t *testing.T) {
	for _, tc := range testCases {
		x := New([]byte(tc.source))
		testLookups(t, tc.source, x, &tc, 0)
		testLookups(t, tc.source, x, &tc, 1)
		testLookups(t, tc.source, x, &tc, 10)
		testLookups(t, tc.source, x, &tc, -1)
	}
}
