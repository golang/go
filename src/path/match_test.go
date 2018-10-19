// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package path

import "testing"

type MatchTest struct {
	pattern, s string
	match      bool
	valid      bool
	err        error
}

var matchTests = []MatchTest{
	{"abc", "abc", true, true, nil},
	{"*", "abc", true, true, nil},
	{"*c", "abc", true, true, nil},
	{"a*", "a", true, true, nil},
	{"a*", "abc", true, true, nil},
	{"a*", "ab/c", false, true, nil},
	{"a*/b", "abc/b", true, true, nil},
	{"a*/b", "a/c/b", false, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxe/f", true, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxexxx/f", true, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxe/xxx/f", false, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxexxx/fff", false, true, nil},
	{"a*b?c*x", "abxbbxdbxebxczzx", true, true, nil},
	{"a*b?c*x", "abxbbxdbxebxczzy", false, true, nil},
	{"ab[c]", "abc", true, true, nil},
	{"ab[b-d]", "abc", true, true, nil},
	{"ab[e-g]", "abc", false, true, nil},
	{"ab[^c]", "abc", false, true, nil},
	{"ab[^b-d]", "abc", false, true, nil},
	{"ab[^e-g]", "abc", true, true, nil},
	{"a\\*b", "a*b", true, true, nil},
	{"a\\*b", "ab", false, true, nil},
	{"a?b", "a☺b", true, true, nil},
	{"a[^a]b", "a☺b", true, true, nil},
	{"a???b", "a☺b", false, true, nil},
	{"a[^a][^a][^a]b", "a☺b", false, true, nil},
	{"[a-ζ]*", "α", true, true, nil},
	{"*[a-ζ]", "A", false, true, nil},
	{"a?b", "a/b", false, true, nil},
	{"a*b", "a/b", false, true, nil},
	{"[\\]a]", "]", true, true, nil},
	{"[\\-]", "-", true, true, nil},
	{"[x\\-]", "x", true, true, nil},
	{"[x\\-]", "-", true, true, nil},
	{"[x\\-]", "z", false, true, nil},
	{"[\\-x]", "x", true, true, nil},
	{"[\\-x]", "-", true, true, nil},
	{"[\\-x]", "a", false, true, nil},
	{"[]a]", "]", false, false, ErrBadPattern},
	{"[-]", "-", false, false, ErrBadPattern},
	{"[x-]", "x", false, false, ErrBadPattern},
	{"[x-]", "-", false, false, ErrBadPattern},
	{"[x-]", "z", false, false, ErrBadPattern},
	{"[-x]", "x", false, false, ErrBadPattern},
	{"[-x]", "-", false, false, ErrBadPattern},
	{"[-x]", "a", false, false, ErrBadPattern},
	{"\\", "a", false, false, ErrBadPattern},
	{"[a-b-c]", "a", false, false, ErrBadPattern},
	{"[", "a", false, false, ErrBadPattern},
	{"[^", "a", false, false, ErrBadPattern},
	{"[^bc", "a", false, false, ErrBadPattern},
	{"a[", "a", false, false, nil},
	{"a[", "ab", false, false, ErrBadPattern},
	{"*x", "xxx", true, true, nil},
}

func TestMatch(t *testing.T) {
	for _, tt := range matchTests {
		ok, err := Match(tt.pattern, tt.s)
		if ok != tt.match || err != tt.err {
			t.Errorf("Match(%#q, %#q) = %v, %v want %v, %v", tt.pattern, tt.s, ok, err, tt.match, tt.err)
		}
	}
}

func TestIsPatternValid(t *testing.T) {
	for _, tt := range matchTests {
		valid := IsPatternValid(tt.pattern)
		if valid && !tt.valid || !valid && tt.valid {
			t.Errorf("IsPatternValid(%#q) returned %t", tt.pattern, tt.valid)
		}
	}
}
