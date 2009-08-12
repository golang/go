// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	. "bytes";
	"strings";
	"testing";
)

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false;
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false;
		}
	}
	return true;
}

func arrayOfString(a [][]byte) []string {
	result := make([]string, len(a));
	for j := 0; j < len(a); j++ {
		result[j] = string(a[j])
	}
	return result
}

// For ease of reading, the test cases use strings that are converted to byte
// arrays before invoking the functions.

var abcd = "abcd"
var faces = "☺☻☹"
var commas = "1,2,3,4"
var dots = "1....2....3....4"

type CompareTest struct {
	a string;
	b string;
	cmp int;
}
var comparetests = []CompareTest {
	CompareTest{ "", "", 0 },
	CompareTest{ "a", "", 1 },
	CompareTest{ "", "a", -1 },
	CompareTest{ "abc", "abc", 0 },
	CompareTest{ "ab", "abc", -1 },
	CompareTest{ "abc", "ab", 1 },
	CompareTest{ "x", "ab", 1 },
	CompareTest{ "ab", "x", -1 },
	CompareTest{ "x", "a", 1 },
	CompareTest{ "b", "x", -1 },
}

func TestCompare(t *testing.T) {
	for i := 0; i < len(comparetests); i++ {
		tt := comparetests[i];
		a := strings.Bytes(tt.a);
		b := strings.Bytes(tt.b);
		cmp := Compare(a, b);
		eql := Equal(a, b);
		if cmp != tt.cmp {
			t.Errorf(`Compare(%q, %q) = %v`, tt.a, tt.b, cmp);
		}
		if eql != (tt.cmp==0) {
			t.Errorf(`Equal(%q, %q) = %v`, tt.a, tt.b, eql);
		}
	}
}


type ExplodeTest struct {
	s string;
	n int;
	a []string;
}
var explodetests = []ExplodeTest {
	ExplodeTest{ abcd,	0, []string{"a", "b", "c", "d"} },
	ExplodeTest{ faces,	0, []string{"☺", "☻", "☹"} },
	ExplodeTest{ abcd,	2, []string{"a", "bcd"} },
}
func TestExplode(t *testing.T) {
	for _, tt := range(explodetests) {
		a := Split(strings.Bytes(tt.s), nil, tt.n);
		result := arrayOfString(a);
		if !eq(result, tt.a) {
			t.Errorf(`Explode("%s", %d) = %v; want %v`, tt.s, tt.n, result, tt.a);
			continue;
		}
		s := Join(a, []byte{});
		if string(s) != tt.s {
			t.Errorf(`Join(Explode("%s", %d), "") = "%s"`, tt.s, tt.n, s);
		}
	}
}


type SplitTest struct {
	s string;
	sep string;
	n int;
	a []string;
}
var splittests = []SplitTest {
	SplitTest{ abcd,	"a",	0, []string{"", "bcd"} },
	SplitTest{ abcd,	"z",	0, []string{"abcd"} },
	SplitTest{ abcd,	"",	0, []string{"a", "b", "c", "d"} },
	SplitTest{ commas,	",",	0, []string{"1", "2", "3", "4"} },
	SplitTest{ dots,	"...",	0, []string{"1", ".2", ".3", ".4"} },
	SplitTest{ faces,	"☹",	0, []string{"☺☻", ""} },
	SplitTest{ faces,	"~",	0, []string{faces} },
	SplitTest{ faces,	"",	0, []string{"☺", "☻", "☹"} },
	SplitTest{ "1 2 3 4",	" ",	3, []string{"1", "2", "3 4"} },
	SplitTest{ "1 2 3",	" ",	3, []string{"1", "2", "3"} },
	SplitTest{ "1 2",	" ",	3, []string{"1", "2"} },
	SplitTest{ "123",	"",	2, []string{"1", "23"} },
	SplitTest{ "123",	"",	17, []string{"1", "2", "3"} },
}
func TestSplit(t *testing.T) {
	for _, tt := range splittests {
		a := Split(strings.Bytes(tt.s), strings.Bytes(tt.sep), tt.n);
		result := arrayOfString(a);
		if !eq(result, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, result, tt.a);
			continue;
		}
		s := Join(a, strings.Bytes(tt.sep));
		if string(s) != tt.s {
			t.Errorf(`Join(Split(%q, %q, %d), %q) = %q`, tt.s, tt.sep, tt.n, tt.sep, s);
		}
	}
}

type CopyTest struct {
	a	string;
	b	string;
	n	int;
	res	string;
}
var copytests = []CopyTest {
	CopyTest{ "", "", 0, "" },
	CopyTest{ "a", "", 0, "a" },
	CopyTest{ "a", "a", 1, "a" },
	CopyTest{ "a", "b", 1, "b" },
	CopyTest{ "xyz", "abc", 3, "abc" },
	CopyTest{ "wxyz", "abc", 3, "abcz" },
	CopyTest{ "xyz", "abcd", 3, "abc" },
}

func TestCopy(t *testing.T) {
	for i := 0; i < len(copytests); i++ {
		tt := copytests[i];
		dst := strings.Bytes(tt.a);
		n := Copy(dst, strings.Bytes(tt.b));
		result := string(dst);
		if result != tt.res || n != tt.n {
			t.Errorf(`Copy(%q, %q) = %d, %q; want %d, %q`, tt.a, tt.b, n, result, tt.n, tt.res);
			continue;
		}
	}
}
