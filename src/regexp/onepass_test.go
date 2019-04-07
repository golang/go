// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"reflect"
	"regexp/syntax"
	"strings"
	"testing"
)

var runeMergeTests = []struct {
	left, right, merged []rune
	next                []uint32
	leftPC, rightPC     uint32
}{
	{
		// empty rhs
		[]rune{69, 69},
		[]rune{},
		[]rune{69, 69},
		[]uint32{1},
		1, 2,
	},
	{
		// identical runes, identical targets
		[]rune{69, 69},
		[]rune{69, 69},
		[]rune{},
		[]uint32{mergeFailed},
		1, 1,
	},
	{
		// identical runes, different targets
		[]rune{69, 69},
		[]rune{69, 69},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
	{
		// append right-first
		[]rune{69, 69},
		[]rune{71, 71},
		[]rune{69, 69, 71, 71},
		[]uint32{1, 2},
		1, 2,
	},
	{
		// append, left-first
		[]rune{71, 71},
		[]rune{69, 69},
		[]rune{69, 69, 71, 71},
		[]uint32{2, 1},
		1, 2,
	},
	{
		// successful interleave
		[]rune{60, 60, 71, 71, 101, 101},
		[]rune{69, 69, 88, 88},
		[]rune{60, 60, 69, 69, 71, 71, 88, 88, 101, 101},
		[]uint32{1, 2, 1, 2, 1},
		1, 2,
	},
	{
		// left surrounds right
		[]rune{69, 74},
		[]rune{71, 71},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
	{
		// right surrounds left
		[]rune{69, 74},
		[]rune{68, 75},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
	{
		// overlap at interval begin
		[]rune{69, 74},
		[]rune{74, 75},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
	{
		// overlap ar interval end
		[]rune{69, 74},
		[]rune{65, 69},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
	{
		// overlap from above
		[]rune{69, 74},
		[]rune{71, 74},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
	{
		// overlap from below
		[]rune{69, 74},
		[]rune{65, 71},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
	{
		// out of order []rune
		[]rune{69, 74, 60, 65},
		[]rune{66, 67},
		[]rune{},
		[]uint32{mergeFailed},
		1, 2,
	},
}

func TestMergeRuneSet(t *testing.T) {
	for ix, test := range runeMergeTests {
		merged, next := mergeRuneSets(&test.left, &test.right, test.leftPC, test.rightPC)
		if !reflect.DeepEqual(merged, test.merged) {
			t.Errorf("mergeRuneSet :%d (%v, %v) merged\n have\n%v\nwant\n%v", ix, test.left, test.right, merged, test.merged)
		}
		if !reflect.DeepEqual(next, test.next) {
			t.Errorf("mergeRuneSet :%d(%v, %v) next\n have\n%v\nwant\n%v", ix, test.left, test.right, next, test.next)
		}
	}
}

var onePassTests = []struct {
	re        string
	isOnePass bool
}{
	{`^(?:a|(?:a*))$`, false},
	{`^(?:(a)|(?:a*))$`, false},
	{`^(?:(?:(?:.(?:$))?))$`, true},
	{`^abcd$`, true},
	{`^(?:(?:a{0,})*?)$`, true},
	{`^(?:(?:a+)*)$`, true},
	{`^(?:(?:a|(?:aa)))$`, true},
	{`^(?:[^\s\S])$`, true},
	{`^(?:(?:a{3,4}){0,})$`, false},
	{`^(?:(?:(?:a*)+))$`, true},
	{`^[a-c]+$`, true},
	{`^[a-c]*$`, true},
	{`^(?:a*)$`, true},
	{`^(?:(?:aa)|a)$`, true},
	{`^[a-c]*`, false},
	{`^...$`, true},
	{`^(?:a|(?:aa))$`, true},
	{`^a((b))c$`, true},
	{`^a.[l-nA-Cg-j]?e$`, true},
	{`^a((b))$`, true},
	{`^a(?:(b)|(c))c$`, true},
	{`^a(?:(b*)|(c))c$`, false},
	{`^a(?:b|c)$`, true},
	{`^a(?:b?|c)$`, true},
	{`^a(?:b?|c?)$`, false},
	{`^a(?:b?|c+)$`, true},
	{`^a(?:b+|(bc))d$`, false},
	{`^a(?:bc)+$`, true},
	{`^a(?:[bcd])+$`, true},
	{`^a((?:[bcd])+)$`, true},
	{`^a(:?b|c)*d$`, true},
	{`^.bc(d|e)*$`, true},
	{`^(?:(?:aa)|.)$`, false},
	{`^(?:(?:a{1,2}){1,2})$`, false},
	{`^l` + strings.Repeat("o", 2<<8) + `ng$`, true},
}

func TestCompileOnePass(t *testing.T) {
	var (
		p   *syntax.Prog
		re  *syntax.Regexp
		err error
	)
	for _, test := range onePassTests {
		if re, err = syntax.Parse(test.re, syntax.Perl); err != nil {
			t.Errorf("Parse(%q) got err:%s, want success", test.re, err)
			continue
		}
		// needs to be done before compile...
		re = re.Simplify()
		if p, err = syntax.Compile(re); err != nil {
			t.Errorf("Compile(%q) got err:%s, want success", test.re, err)
			continue
		}
		isOnePass := compileOnePass(p) != nil
		if isOnePass != test.isOnePass {
			t.Errorf("CompileOnePass(%q) got isOnePass=%v, expected %v", test.re, isOnePass, test.isOnePass)
		}
	}
}

// TODO(cespare): Unify with onePassTests and rationalize one-pass test cases.
var onePassTests1 = []struct {
	re    string
	match string
}{
	{`^a(/b+(#c+)*)*$`, "a/b#c"}, // golang.org/issue/11905
}

func TestRunOnePass(t *testing.T) {
	for _, test := range onePassTests1 {
		re, err := Compile(test.re)
		if err != nil {
			t.Errorf("Compile(%q): got err: %s", test.re, err)
			continue
		}
		if re.onepass == nil {
			t.Errorf("Compile(%q): got nil, want one-pass", test.re)
			continue
		}
		if !re.MatchString(test.match) {
			t.Errorf("onepass %q did not match %q", test.re, test.match)
		}
	}
}
