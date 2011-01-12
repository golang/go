// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package suffixarray

import (
	"bytes"
	"container/vector"
	"regexp"
	"sort"
	"strings"
	"testing"
)


type testCase struct {
	name     string   // name of test case
	source   string   // source to index
	patterns []string // patterns to lookup
}


var testCases = []testCase{
	{
		"empty string",
		"",
		[]string{
			"",
			"foo",
			"(foo)",
			".*",
			"a*",
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
			".",
			".*",
			"a+",
			"aa+",
			"aaaa[b]?",
			"aaa*",
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
			"a.c",
			"a(b|c)",
			"abc?",
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
			"bara?bar",
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
			"to (come|the)?",
		},
	},
}


// find all occurrences of s in source; report at most n occurences
func find(src, s string, n int) []int {
	var res vector.IntVector
	if s != "" && n != 0 {
		// find at most n occurrences of s in src
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


func testLookup(t *testing.T, tc *testCase, x *Index, s string, n int) {
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

	// check that each result is in fact a correct match and there are no duplicates
	sort.SortInts(res)
	for i, r := range res {
		if r < 0 || len(tc.source) <= r {
			t.Errorf("test %q, lookup %q, result %d (n = %d): index %d out of range [0, %d[", tc.name, s, i, n, r, len(tc.source))
		} else if !strings.HasPrefix(tc.source[r:], s) {
			t.Errorf("test %q, lookup %q, result %d (n = %d): index %d not a match", tc.name, s, i, n, r)
		}
		if i > 0 && res[i-1] == r {
			t.Errorf("test %q, lookup %q, result %d (n = %d): found duplicate index %d", tc.name, s, i, n, r)
		}
	}

	if n < 0 {
		// all results computed - sorted res and exp must be equal
		for i, r := range res {
			e := exp[i]
			if r != e {
				t.Errorf("test %q, lookup %q, result %d: expected index %d; got %d", tc.name, s, i, e, r)
			}
		}
	}
}


func testFindAllIndex(t *testing.T, tc *testCase, x *Index, rx *regexp.Regexp, n int) {
	res := x.FindAllIndex(rx, n)
	exp := rx.FindAllStringIndex(tc.source, n)

	// check that the lengths match
	if len(res) != len(exp) {
		t.Errorf("test %q, FindAllIndex %q (n = %d): expected %d results; got %d", tc.name, rx, n, len(exp), len(res))
	}

	// if n >= 0 the number of results is limited --- unless n >= all results,
	// we may obtain different positions from the Index and from regexp (because
	// Index may not find the results in the same order as regexp) => in general
	// we cannot simply check that the res and exp lists are equal

	// check that each result is in fact a correct match and the result is sorted
	for i, r := range res {
		if r[0] < 0 || r[0] > r[1] || len(tc.source) < r[1] {
			t.Errorf("test %q, FindAllIndex %q, result %d (n == %d): illegal match [%d, %d]", tc.name, rx, i, n, r[0], r[1])
		} else if !rx.MatchString(tc.source[r[0]:r[1]]) {
			t.Errorf("test %q, FindAllIndex %q, result %d (n = %d): [%d, %d] not a match", tc.name, rx, i, n, r[0], r[1])
		}
	}

	if n < 0 {
		// all results computed - sorted res and exp must be equal
		for i, r := range res {
			e := exp[i]
			if r[0] != e[0] || r[1] != e[1] {
				t.Errorf("test %q, FindAllIndex %q, result %d: expected match [%d, %d]; got [%d, %d]",
					tc.name, rx, i, e[0], e[1], r[0], r[1])
			}
		}
	}
}


func testLookups(t *testing.T, tc *testCase, x *Index, n int) {
	for _, pat := range tc.patterns {
		testLookup(t, tc, x, pat, n)
		if rx, err := regexp.Compile(pat); err == nil {
			testFindAllIndex(t, tc, x, rx, n)
		}
	}
}


// index is used to hide the sort.Interface
type index Index

func (x *index) Len() int           { return len(x.sa) }
func (x *index) Less(i, j int) bool { return bytes.Compare(x.at(i), x.at(j)) < 0 }
func (x *index) Swap(i, j int)      { x.sa[i], x.sa[j] = x.sa[j], x.sa[i] }
func (a *index) at(i int) []byte    { return a.data[a.sa[i]:] }


func testConstruction(t *testing.T, tc *testCase, x *Index) {
	if !sort.IsSorted((*index)(x)) {
		t.Errorf("testConstruction failed %s", tc.name)
	}
}


func TestIndex(t *testing.T) {
	for _, tc := range testCases {
		x := New([]byte(tc.source))
		testConstruction(t, &tc, x)
		testLookups(t, &tc, x, 0)
		testLookups(t, &tc, x, 1)
		testLookups(t, &tc, x, 10)
		testLookups(t, &tc, x, 2e9)
		testLookups(t, &tc, x, -1)
	}
}
