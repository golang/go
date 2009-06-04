// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes

import (
	"bytes";
	"io";
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
		a := io.StringBytes(tt.a);
		b := io.StringBytes(tt.b);
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
	a []string;
}
var explodetests = []ExplodeTest {
	ExplodeTest{ abcd,	[]string{"a", "b", "c", "d"} },
	ExplodeTest{ faces,	[]string{"☺", "☻", "☹" } },
}
func TestExplode(t *testing.T) {
	for i := 0; i < len(explodetests); i++ {
		tt := explodetests[i];
		a := Explode(io.StringBytes(tt.s));
		result := arrayOfString(a);
		if !eq(result, tt.a) {
			t.Errorf(`Explode("%s") = %v; want %v`, tt.s, result, tt.a);
			continue;
		}
		s := Join(a, []byte{});
		if string(s) != tt.s {
			t.Errorf(`Join(Explode("%s"), "") = "%s"`, tt.s, s);
		}
	}
}


type SplitTest struct {
	s string;
	sep string;
	a []string;
}
var splittests = []SplitTest {
	SplitTest{ abcd,	"a",	[]string{"", "bcd"} },
	SplitTest{ abcd,	"z",	[]string{"abcd"} },
	SplitTest{ abcd,	"",	[]string{"a", "b", "c", "d"} },
	SplitTest{ commas,	",",	[]string{"1", "2", "3", "4"} },
	SplitTest{ dots,	"...",	[]string{"1", ".2", ".3", ".4"} },
	SplitTest{ faces,	"☹",	[]string{"☺☻", ""} },
	SplitTest{ faces,	"~",	[]string{faces} },
	SplitTest{ faces,	"",	[]string{"☺", "☻", "☹"} },
}
func TestSplit(t *testing.T) {
	for i := 0; i < len(splittests); i++ {
		tt := splittests[i];
		a := Split(io.StringBytes(tt.s), io.StringBytes(tt.sep));
		result := arrayOfString(a);
		if !eq(result, tt.a) {
			t.Errorf(`Split("%s", "%s") = %v; want %v`, tt.s, tt.sep, result, tt.a);
			continue;
		}
		s := Join(a, io.StringBytes(tt.sep));
		if string(s) != tt.s {
			t.Errorf(`Join(Split("%s", "%s"), "%s") = "%s"`, tt.s, tt.sep, tt.sep, s);
		}
	}
}

type CopyTest struct {
	a	string;
	b	string;
	res	string;
}
var copytests = []CopyTest {
	CopyTest{ "", "", "" },
	CopyTest{ "a", "", "a" },
	CopyTest{ "a", "a", "a" },
	CopyTest{ "a", "b", "b" },
	CopyTest{ "xyz", "abc", "abc" },
	CopyTest{ "wxyz", "abc", "abcz" },
}

func TestCopy(t *testing.T) {
	for i := 0; i < len(copytests); i++ {
		tt := copytests[i];
		dst := io.StringBytes(tt.a);
		Copy(dst, io.StringBytes(tt.b));
		result := string(dst);
		if result != tt.res {
			t.Errorf(`Copy("%s", "%s") = "%s"; want "%s"`, tt.a, tt.b, result, tt.res);
			continue;
		}
	}
}
