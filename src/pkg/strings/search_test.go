// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	"reflect"
	. "strings"
	"testing"
)

func TestFinderNext(t *testing.T) {
	testCases := []struct {
		pat, text string
		index     int
	}{
		{"", "", 0},
		{"", "abc", 0},
		{"abc", "", -1},
		{"abc", "abc", 0},
		{"d", "abcdefg", 3},
		{"nan", "banana", 2},
		{"pan", "anpanman", 2},
		{"nnaaman", "anpanmanam", -1},
		{"abcd", "abc", -1},
		{"abcd", "bcd", -1},
		{"bcd", "abcd", 1},
		{"abc", "acca", -1},
		{"aa", "aaa", 0},
		{"baa", "aaaaa", -1},
		{"at that", "which finally halts.  at that point", 22},
	}

	for _, tc := range testCases {
		got := StringFind(tc.pat, tc.text)
		want := tc.index
		if got != want {
			t.Errorf("stringFind(%q, %q) got %d, want %d\n", tc.pat, tc.text, got, want)
		}
	}
}

func TestFinderCreation(t *testing.T) {
	testCases := []struct {
		pattern string
		bad     [256]int
		suf     []int
	}{
		{
			"abc",
			[256]int{'a': 2, 'b': 1, 'c': 3},
			[]int{5, 4, 1},
		},
		{
			"mississi",
			[256]int{'i': 3, 'm': 7, 's': 1},
			[]int{15, 14, 13, 7, 11, 10, 7, 1},
		},
		// From http://www.cs.utexas.edu/~moore/publications/fstrpos.pdf
		{
			"abcxxxabc",
			[256]int{'a': 2, 'b': 1, 'c': 6, 'x': 3},
			[]int{14, 13, 12, 11, 10, 9, 11, 10, 1},
		},
		{
			"abyxcdeyx",
			[256]int{'a': 8, 'b': 7, 'c': 4, 'd': 3, 'e': 2, 'y': 1, 'x': 5},
			[]int{17, 16, 15, 14, 13, 12, 7, 10, 1},
		},
	}

	for _, tc := range testCases {
		bad, good := DumpTables(tc.pattern)

		for i, got := range bad {
			want := tc.bad[i]
			if want == 0 {
				want = len(tc.pattern)
			}
			if got != want {
				t.Errorf("boyerMoore(%q) bad['%c']: got %d want %d", tc.pattern, i, got, want)
			}
		}

		if !reflect.DeepEqual(good, tc.suf) {
			t.Errorf("boyerMoore(%q) got %v want %v", tc.pattern, good, tc.suf)
		}
	}
}
