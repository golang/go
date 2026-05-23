// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"fmt"
	"strings"
	"testing"
)

// For each pattern/text pair, what is the expected output of each function?
// We can derive the textual results from the indexed results, the non-submatch
// results from the submatched results, the single results from the 'all' results,
// and the byte results from the string results. Therefore the table includes
// only the FindAllStringSubmatchIndex result.
type FindTest struct {
	pat     string
	text    string
	max     int
	matches [][]int
}

func (t FindTest) String() string {
	return fmt.Sprintf("pat: %#q text: %#q", t.pat, t.text)
}

var findTests = []FindTest{
	{``, ``, -1, build(1, 0, 0)},
	{`^abcdefg`, "abcdefg", -1, build(1, 0, 7)},
	{`a+`, "baaab", -1, build(1, 1, 4)},
	{`a`, "bababaab", -1, build(4, 1, 2, 3, 4, 5, 6, 6, 7)},
	{`a`, "bababaab", 0, nil},
	{`a`, "bababaab", 1, build(1, 1, 2)},
	{`a`, "bababaab", 2, build(2, 1, 2, 3, 4)},
	{`a`, "bababaab", 3, build(3, 1, 2, 3, 4, 5, 6)},
	{`a`, "bababaab", 4, build(4, 1, 2, 3, 4, 5, 6, 6, 7)},
	{`a`, "bababaab", 5, build(4, 1, 2, 3, 4, 5, 6, 6, 7)},
	{"abcd..", "abcdef", -1, build(1, 0, 6)},
	{`a`, "a", -1, build(1, 0, 1)},
	{`x`, "y", -1, nil},
	{`b`, "abc", -1, build(1, 1, 2)},
	{`.`, "a", -1, build(1, 0, 1)},
	{`.*`, "abcdef", -1, build(1, 0, 6)},
	{`^`, "abcde", -1, build(1, 0, 0)},
	{`$`, "abcde", -1, build(1, 5, 5)},
	{`^abcd$`, "abcd", -1, build(1, 0, 4)},
	{`^bcd'`, "abcdef", -1, nil},
	{`^abcd$`, "abcde", -1, nil},
	{`a+`, "baaab", -1, build(1, 1, 4)},
	{`a*`, "baaab", -1, build(3, 0, 0, 1, 4, 5, 5)},
	{`[a-z]+`, "abcd", -1, build(1, 0, 4)},
	{`[^a-z]+`, "ab1234cd", -1, build(1, 2, 6)},
	{`[a\-\]z]+`, "az]-bcz", -1, build(2, 0, 4, 6, 7)},
	{`[^\n]+`, "abcd\n", -1, build(1, 0, 4)},
	{`[日本語]+`, "日本語日本語", -1, build(1, 0, 18)},
	{`日本語+`, "日本語", -1, build(1, 0, 9)},
	{`日本語+`, "日本語語語語", -1, build(1, 0, 18)},
	{`()`, "", -1, build(1, 0, 0, 0, 0)},
	{`(a)`, "a", -1, build(1, 0, 1, 0, 1)},
	{`(.)(.)`, "日a", -1, build(1, 0, 4, 0, 3, 3, 4)},
	{`(.*)`, "", -1, build(1, 0, 0, 0, 0)},
	{`(.*)`, "abcd", -1, build(1, 0, 4, 0, 4)},
	{`(..)(..)`, "abcd", -1, build(1, 0, 4, 0, 2, 2, 4)},
	{`(([^xyz]*)(d))`, "abcd", -1, build(1, 0, 4, 0, 4, 0, 3, 3, 4)},
	{`((a|b|c)*(d))`, "abcd", -1, build(1, 0, 4, 0, 4, 2, 3, 3, 4)},
	{`(((a|b|c)*)(d))`, "abcd", -1, build(1, 0, 4, 0, 4, 0, 3, 2, 3, 3, 4)},
	{`\a\f\n\r\t\v`, "\a\f\n\r\t\v", -1, build(1, 0, 6)},
	{`[\a\f\n\r\t\v]+`, "\a\f\n\r\t\v", -1, build(1, 0, 6)},

	{`a*(|(b))c*`, "aacc", -1, build(1, 0, 4, 2, 2, -1, -1)},
	{`(.*).*`, "ab", -1, build(1, 0, 2, 0, 2)},
	{`[.]`, ".", -1, build(1, 0, 1)},
	{`/$`, "/abc/", -1, build(1, 4, 5)},
	{`/$`, "/abc", -1, nil},

	// multiple matches
	{`.`, "abc", -1, build(3, 0, 1, 1, 2, 2, 3)},
	{`(.)`, "abc", -1, build(3, 0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 2, 3)},
	{`.(.)`, "abcd", -1, build(2, 0, 2, 1, 2, 2, 4, 3, 4)},
	{`ab*`, "abbaab", -1, build(3, 0, 3, 3, 4, 4, 6)},
	{`a(b*)`, "abbaab", -1, build(3, 0, 3, 1, 3, 3, 4, 4, 4, 4, 6, 5, 6)},

	// fixed bugs
	{`ab$`, "cab", -1, build(1, 1, 3)},
	{`axxb$`, "axxcb", -1, nil},
	{`data`, "daXY data", -1, build(1, 5, 9)},
	{`da(.)a$`, "daXY data", -1, build(1, 5, 9, 7, 8)},
	{`zx+`, "zzx", -1, build(1, 1, 3)},
	{`ab$`, "abcab", -1, build(1, 3, 5)},
	{`(aa)*$`, "a", -1, build(1, 1, 1, -1, -1)},
	{`(?:.|(?:.a))`, "", -1, nil},
	{`(?:A(?:A|a))`, "Aa", -1, build(1, 0, 2)},
	{`(?:A|(?:A|a))`, "a", -1, build(1, 0, 1)},
	{`(a){0}`, "", -1, build(1, 0, 0, -1, -1)},
	{`(?-s)(?:(?:^).)`, "\n", -1, nil},
	{`(?s)(?:(?:^).)`, "\n", -1, build(1, 0, 1)},
	{`(?:(?:^).)`, "\n", -1, nil},
	{`\b`, "x", -1, build(2, 0, 0, 1, 1)},
	{`\b`, "xx", -1, build(2, 0, 0, 2, 2)},
	{`\b`, "x y", -1, build(4, 0, 0, 1, 1, 2, 2, 3, 3)},
	{`\b`, "xx yy", -1, build(4, 0, 0, 2, 2, 3, 3, 5, 5)},
	{`\B`, "x", -1, nil},
	{`\B`, "xx", -1, build(1, 1, 1)},
	{`\B`, "x y", -1, nil},
	{`\B`, "xx yy", -1, build(2, 1, 1, 4, 4)},
	{`(|a)*`, "aa", -1, build(3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2)},
	{`0A|0[aA]`, "0a", -1, build(1, 0, 2)},
	{`0[aA]|0A`, "0a", -1, build(1, 0, 2)},

	// RE2 tests
	{`[^\S\s]`, "abcd", -1, nil},
	{`[^\S[:space:]]`, "abcd", -1, nil},
	{`[^\D\d]`, "abcd", -1, nil},
	{`[^\D[:digit:]]`, "abcd", -1, nil},
	{`(?i)\W`, "x", -1, nil},
	{`(?i)\W`, "k", -1, nil},
	{`(?i)\W`, "s", -1, nil},

	// can backslash-escape any punctuation
	{`\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\{\|\}\~`,
		`!"#$%&'()*+,-./:;<=>?@[\]^_{|}~`, -1, build(1, 0, 31)},
	{`[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\{\|\}\~]+`,
		`!"#$%&'()*+,-./:;<=>?@[\]^_{|}~`, -1, build(1, 0, 31)},
	{"\\`", "`", -1, build(1, 0, 1)},
	{"[\\`]+", "`", -1, build(1, 0, 1)},

	{"\ufffd", "\xff", -1, build(1, 0, 1)},
	{"\ufffd", "hello\xffworld", -1, build(1, 5, 6)},
	{`.*`, "hello\xffworld", -1, build(1, 0, 11)},
	{`\x{fffd}`, "\xc2\x00", -1, build(1, 0, 1)},
	{"[\ufffd]", "\xff", -1, build(1, 0, 1)},
	{`[\x{fffd}]`, "\xc2\x00", -1, build(1, 0, 1)},

	// long set of matches (longer than startSize)
	{
		".",
		"qwertyuiopasdfghjklzxcvbnm1234567890",
		-1,
		build(36, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
			10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
			20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30,
			30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36),
	},
}

// build is a helper to construct a [][]int by extracting n sequences from x.
// This represents n matches with len(x)/n submatches each.
func build(n int, x ...int) [][]int {
	ret := make([][]int, n)
	runLength := len(x) / n
	j := 0
	for i := range ret {
		ret[i] = make([]int, runLength)
		copy(ret[i], x[j:])
		j += runLength
		if j > len(x) {
			panic("invalid build entry")
		}
	}
	return ret
}

// First the simple cases.

func TestFind(t *testing.T) {
	for _, test := range findTests {
		re := MustCompile(test.pat)
		if re.String() != test.pat {
			t.Errorf("re.String() = %q, want %q", re.String(), test.pat)
		}
		result := re.Find([]byte(test.text))
		switch {
		case test.max == 0:
			// do not know whether to match or not; skip
		case len(test.matches) == 0 && len(result) == 0:
			// ok
		case test.matches == nil && result != nil:
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == nil:
			t.Errorf("got no match, want one: %s", test)
		case test.matches != nil && result != nil:
			want := test.text[test.matches[0][0]:test.matches[0][1]]
			if len(result) != cap(result) {
				t.Errorf("got capacity %d, want %d: %s", cap(result), len(result), test)
			}
			if want != string(result) {
				t.Errorf("got %q, want %q: %s", result, want, test)
			}
		}
	}
}

func TestFindString(t *testing.T) {
	for _, test := range findTests {
		result := MustCompile(test.pat).FindString(test.text)
		switch {
		case test.max == 0:
			// do not know whether to match or not; skip
		case len(test.matches) == 0 && len(result) == 0:
			// ok
		case test.matches == nil && result != "":
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == "":
			// Tricky because an empty result has two meanings: no match or empty match.
			if test.matches[0][0] != test.matches[0][1] {
				t.Errorf("got no match, want one: %s", test)
			}
		case test.matches != nil && result != "":
			want := test.text[test.matches[0][0]:test.matches[0][1]]
			if want != result {
				t.Errorf("got %q, want %q: %s", result, want, test)
			}
		}
	}
}

func testFindIndex(test *FindTest, result []int, t *testing.T) {
	switch {
	case test.max == 0:
		// do not know whether to match or not; skip
	case len(test.matches) == 0 && len(result) == 0:
		// ok
	case test.matches == nil && result != nil:
		t.Errorf("got match %v, want none: %s", result, test)
	case test.matches != nil && result == nil:
		t.Errorf("got no match, want one: %s", test)
	case test.matches != nil && result != nil:
		want := test.matches[0]
		if want[0] != result[0] || want[1] != result[1] {
			t.Errorf("got %v, want %v: %s", result, want, test)
		}
	}
}

func TestFindIndex(t *testing.T) {
	for _, test := range findTests {
		testFindIndex(&test, MustCompile(test.pat).FindIndex([]byte(test.text)), t)
	}
}

func TestFindStringIndex(t *testing.T) {
	for _, test := range findTests {
		testFindIndex(&test, MustCompile(test.pat).FindStringIndex(test.text), t)
	}
}

func TestFindReaderIndex(t *testing.T) {
	for _, test := range findTests {
		testFindIndex(&test, MustCompile(test.pat).FindReaderIndex(strings.NewReader(test.text)), t)
	}
}

// Now come the simple All cases.

func TestFindAll(t *testing.T) {
	for _, test := range findTests {
		result := MustCompile(test.pat).FindAll([]byte(test.text), test.max)
		switch {
		case test.matches == nil && result == nil:
			// ok
		case test.matches == nil && result != nil:
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == nil:
			t.Fatalf("got no match, want one: %s", test)
		case test.matches != nil && result != nil:
			if len(test.matches) != len(result) {
				t.Errorf("got %d matches, want %d: %s", len(result), len(test.matches), test)
				continue
			}
			for k, e := range test.matches {
				got := result[k]
				if len(got) != cap(got) {
					t.Errorf("match %d: got capacity %d, want %d: %s", k, cap(got), len(got), test)
				}
				want := test.text[e[0]:e[1]]
				if want != string(got) {
					t.Errorf("match %d: got %q, want %q: %s", k, got, want, test)
				}
			}
		}
	}
}

func TestFindAllString(t *testing.T) {
	for _, test := range findTests {
		result := MustCompile(test.pat).FindAllString(test.text, test.max)
		switch {
		case test.matches == nil && result == nil:
			// ok
		case test.matches == nil && result != nil:
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == nil:
			t.Errorf("got no match, want one: %s", test)
		case test.matches != nil && result != nil:
			if len(test.matches) != len(result) {
				t.Errorf("got %d matches, want %d: %s", len(result), len(test.matches), test)
				continue
			}
			for k, e := range test.matches {
				want := test.text[e[0]:e[1]]
				if want != result[k] {
					t.Errorf("got %q, want %q: %s", result[k], want, test)
				}
			}
		}
	}
}

func testFindAllIndex(test *FindTest, result [][]int, t *testing.T) {
	switch {
	case test.matches == nil && result == nil:
		// ok
	case test.matches == nil && result != nil:
		t.Errorf("got match %v, want none: %s", result, test)
	case test.matches != nil && result == nil:
		t.Errorf("got no match, want one: %s", test)
	case test.matches != nil && result != nil:
		if len(test.matches) != len(result) {
			t.Errorf("got %d matches, want %d: %s", len(result), len(test.matches), test)
			return
		}
		for k, e := range test.matches {
			if e[0] != result[k][0] || e[1] != result[k][1] {
				t.Errorf("match %d: got %v, want %v: %s", k, result[k], e, test)
			}
		}
	}
}

func TestFindAllIndex(t *testing.T) {
	for _, test := range findTests {
		testFindAllIndex(&test, MustCompile(test.pat).FindAllIndex([]byte(test.text), test.max), t)
	}
}

func TestFindAllStringIndex(t *testing.T) {
	for _, test := range findTests {
		testFindAllIndex(&test, MustCompile(test.pat).FindAllStringIndex(test.text, test.max), t)
	}
}

// Now come the Submatch cases.

func testSubmatchBytes(test *FindTest, n int, submatches []int, result [][]byte, t *testing.T) {
	if len(submatches) != len(result)*2 {
		t.Errorf("match %d: got %d submatches, want %d: %s", n, len(result), len(submatches)/2, test)
		return
	}
	for k := 0; k < len(submatches); k += 2 {
		if submatches[k] == -1 {
			if result[k/2] != nil {
				t.Errorf("match %d: got %q, want nil: %s", n, result, test)
			}
			continue
		}
		got := result[k/2]
		if len(got) != cap(got) {
			t.Errorf("match %d: got capacity %d, want %d: %s", n, cap(got), len(got), test)
			return
		}
		want := test.text[submatches[k]:submatches[k+1]]
		if want != string(got) {
			t.Errorf("match %d: got %q, want %q: %s", n, got, want, test)
			return
		}
	}
}

func TestFindSubmatch(t *testing.T) {
	for _, test := range findTests {
		result := MustCompile(test.pat).FindSubmatch([]byte(test.text))
		switch {
		case test.max == 0:
			// do not know whether to match or not; skip
		case test.matches == nil && result == nil:
			// ok
		case test.matches == nil && result != nil:
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == nil:
			t.Errorf("got no match, want one: %s", test)
		case test.matches != nil && result != nil:
			testSubmatchBytes(&test, 0, test.matches[0], result, t)
		}
	}
}

func testSubmatchString(test *FindTest, n int, submatches []int, result []string, t *testing.T) {
	if len(submatches) != len(result)*2 {
		t.Errorf("match %d: got %d submatches, want %d: %s", n, len(result), len(submatches)/2, test)
		return
	}
	for k := 0; k < len(submatches); k += 2 {
		if submatches[k] == -1 {
			if result[k/2] != "" {
				t.Errorf("match %d: got %q, want empty string: %s", n, result, test)
			}
			continue
		}
		want := test.text[submatches[k]:submatches[k+1]]
		if want != result[k/2] {
			t.Errorf("match %d: got %q, want %q: %s", n, result[k/2], want, test)
			return
		}
	}
}

func TestFindStringSubmatch(t *testing.T) {
	for _, test := range findTests {
		result := MustCompile(test.pat).FindStringSubmatch(test.text)
		switch {
		case test.max == 0:
			// do not know whether to match or not; skip
		case test.matches == nil && result == nil:
			// ok
		case test.matches == nil && result != nil:
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == nil:
			t.Errorf("got no match, want one: %s", test)
		case test.matches != nil && result != nil:
			testSubmatchString(&test, 0, test.matches[0], result, t)
		}
	}
}

func testSubmatchIndices(test *FindTest, n int, want, result []int, t *testing.T) {
	if len(want) != len(result) {
		t.Errorf("match %d: got %d matches, want %d: %s", n, len(result)/2, len(want)/2, test)
		return
	}
	for k, e := range want {
		if e != result[k] {
			t.Errorf("match %d: submatch error: got %v, want %v: %s", n, result, want, test)
		}
	}
}

func testFindSubmatchIndex(test *FindTest, result []int, t *testing.T) {
	switch {
	case test.max == 0:
		// do not know whether to match or not; skip
	case test.matches == nil && result == nil:
		// ok
	case test.matches == nil && result != nil:
		t.Errorf("got match %v, want none: %s", result, test)
	case test.matches != nil && result == nil:
		t.Errorf("got no match, want one: %s", test)
	case test.matches != nil && result != nil:
		testSubmatchIndices(test, 0, test.matches[0], result, t)
	}
}

func TestFindSubmatchIndex(t *testing.T) {
	for _, test := range findTests {
		testFindSubmatchIndex(&test, MustCompile(test.pat).FindSubmatchIndex([]byte(test.text)), t)
	}
}

func TestFindStringSubmatchIndex(t *testing.T) {
	for _, test := range findTests {
		testFindSubmatchIndex(&test, MustCompile(test.pat).FindStringSubmatchIndex(test.text), t)
	}
}

func TestFindReaderSubmatchIndex(t *testing.T) {
	for _, test := range findTests {
		testFindSubmatchIndex(&test, MustCompile(test.pat).FindReaderSubmatchIndex(strings.NewReader(test.text)), t)
	}
}

// Now come the monster AllSubmatch cases.

func TestFindAllSubmatch(t *testing.T) {
	for _, test := range findTests {
		result := MustCompile(test.pat).FindAllSubmatch([]byte(test.text), test.max)
		switch {
		case test.matches == nil && result == nil:
			// ok
		case test.matches == nil && result != nil:
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == nil:
			t.Errorf("got no match, want one: %s", test)
		case len(test.matches) != len(result):
			t.Errorf("got %d matches, want %d: %s", len(result), len(test.matches), test)
		case test.matches != nil && result != nil:
			for k, match := range test.matches {
				testSubmatchBytes(&test, k, match, result[k], t)
			}
		}
	}
}

func TestFindAllStringSubmatch(t *testing.T) {
	for _, test := range findTests {
		result := MustCompile(test.pat).FindAllStringSubmatch(test.text, test.max)
		switch {
		case test.matches == nil && result == nil:
			// ok
		case test.matches == nil && result != nil:
			t.Errorf("got match %q, want none: %s", result, test)
		case test.matches != nil && result == nil:
			t.Errorf("got no match, want one: %s", test)
		case len(test.matches) != len(result):
			t.Errorf("got %d matches, want %d: %s", len(result), len(test.matches), test)
		case test.matches != nil && result != nil:
			for k, match := range test.matches {
				testSubmatchString(&test, k, match, result[k], t)
			}
		}
	}
}

func testFindAllSubmatchIndex(test *FindTest, result [][]int, t *testing.T) {
	switch {
	case test.matches == nil && result == nil:
		// ok
	case test.matches == nil && result != nil:
		t.Errorf("got match %v, want none: %s", result, test)
	case test.matches != nil && result == nil:
		t.Errorf("got no match, want one: %s", test)
	case len(test.matches) != len(result):
		t.Errorf("got %d matches, want %d: %s", len(result), len(test.matches), test)
	case test.matches != nil && result != nil:
		for k, match := range test.matches {
			testSubmatchIndices(test, k, match, result[k], t)
		}
	}
}

func TestFindAllSubmatchIndex(t *testing.T) {
	for _, test := range findTests {
		testFindAllSubmatchIndex(&test, MustCompile(test.pat).FindAllSubmatchIndex([]byte(test.text), test.max), t)
	}
}

func TestFindAllStringSubmatchIndex(t *testing.T) {
	for _, test := range findTests {
		testFindAllSubmatchIndex(&test, MustCompile(test.pat).FindAllStringSubmatchIndex(test.text, test.max), t)
	}
}
