// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package path

import (
	"os"
	"testing"
)

type MatchTest struct {
	pattern, s string
	match      bool
	err        os.Error
}

var matchTests = []MatchTest{
	MatchTest{"abc", "abc", true, nil},
	MatchTest{"*", "abc", true, nil},
	MatchTest{"*c", "abc", true, nil},
	MatchTest{"a*", "a", true, nil},
	MatchTest{"a*", "abc", true, nil},
	MatchTest{"a*", "ab/c", false, nil},
	MatchTest{"a*/b", "abc/b", true, nil},
	MatchTest{"a*/b", "a/c/b", false, nil},
	MatchTest{"a*b*c*d*e*/f", "axbxcxdxe/f", true, nil},
	MatchTest{"a*b*c*d*e*/f", "axbxcxdxexxx/f", true, nil},
	MatchTest{"a*b*c*d*e*/f", "axbxcxdxe/xxx/f", false, nil},
	MatchTest{"a*b*c*d*e*/f", "axbxcxdxexxx/fff", false, nil},
	MatchTest{"a*b?c*x", "abxbbxdbxebxczzx", true, nil},
	MatchTest{"a*b?c*x", "abxbbxdbxebxczzy", false, nil},
	MatchTest{"ab[c]", "abc", true, nil},
	MatchTest{"ab[b-d]", "abc", true, nil},
	MatchTest{"ab[e-g]", "abc", false, nil},
	MatchTest{"ab[^c]", "abc", false, nil},
	MatchTest{"ab[^b-d]", "abc", false, nil},
	MatchTest{"ab[^e-g]", "abc", true, nil},
	MatchTest{"a\\*b", "a*b", true, nil},
	MatchTest{"a\\*b", "ab", false, nil},
	MatchTest{"a?b", "a☺b", true, nil},
	MatchTest{"a[^a]b", "a☺b", true, nil},
	MatchTest{"a???b", "a☺b", false, nil},
	MatchTest{"a[^a][^a][^a]b", "a☺b", false, nil},
	MatchTest{"[a-ζ]*", "α", true, nil},
	MatchTest{"*[a-ζ]", "A", false, nil},
	MatchTest{"a?b", "a/b", false, nil},
	MatchTest{"a*b", "a/b", false, nil},
	MatchTest{"[\\]a]", "]", true, nil},
	MatchTest{"[\\-]", "-", true, nil},
	MatchTest{"[x\\-]", "x", true, nil},
	MatchTest{"[x\\-]", "-", true, nil},
	MatchTest{"[x\\-]", "z", false, nil},
	MatchTest{"[\\-x]", "x", true, nil},
	MatchTest{"[\\-x]", "-", true, nil},
	MatchTest{"[\\-x]", "a", false, nil},
	MatchTest{"[]a]", "]", false, ErrBadPattern},
	MatchTest{"[-]", "-", false, ErrBadPattern},
	MatchTest{"[x-]", "x", false, ErrBadPattern},
	MatchTest{"[x-]", "-", false, ErrBadPattern},
	MatchTest{"[x-]", "z", false, ErrBadPattern},
	MatchTest{"[-x]", "x", false, ErrBadPattern},
	MatchTest{"[-x]", "-", false, ErrBadPattern},
	MatchTest{"[-x]", "a", false, ErrBadPattern},
	MatchTest{"\\", "a", false, ErrBadPattern},
	MatchTest{"[a-b-c]", "a", false, ErrBadPattern},
}

func TestMatch(t *testing.T) {
	for _, tt := range matchTests {
		ok, err := Match(tt.pattern, tt.s)
		if ok != tt.match || err != tt.err {
			t.Errorf("Match(%#q, %#q) = %v, %v want %v, nil\n", tt.pattern, tt.s, ok, err, tt.match)
		}
	}
}
