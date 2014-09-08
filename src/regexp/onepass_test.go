// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"reflect"
	"regexp/syntax"
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

const noStr = `!`

var onePass = &onePassProg{}

var onePassTests = []struct {
	re      string
	onePass *onePassProg
	prog    string
}{
	{`^(?:a|(?:a*))$`, notOnePass, noStr},
	{`^(?:(a)|(?:a*))$`, notOnePass, noStr},
	{`^(?:(?:(?:.(?:$))?))$`, onePass, `a`},
	{`^abcd$`, onePass, `abcd`},
	{`^abcd$`, onePass, `abcde`},
	{`^(?:(?:a{0,})*?)$`, onePass, `a`},
	{`^(?:(?:a+)*)$`, onePass, ``},
	{`^(?:(?:a|(?:aa)))$`, onePass, ``},
	{`^(?:[^\s\S])$`, onePass, ``},
	{`^(?:(?:a{3,4}){0,})$`, notOnePass, `aaaaaa`},
	{`^(?:(?:a+)*)$`, onePass, `a`},
	{`^(?:(?:(?:a*)+))$`, onePass, noStr},
	{`^(?:(?:a+)*)$`, onePass, ``},
	{`^[a-c]+$`, onePass, `abc`},
	{`^[a-c]*$`, onePass, `abcdabc`},
	{`^(?:a*)$`, onePass, `aaaaaaa`},
	{`^(?:(?:aa)|a)$`, onePass, `a`},
	{`^[a-c]*`, notOnePass, `abcdabc`},
	{`^[a-c]*$`, onePass, `abc`},
	{`^...$`, onePass, ``},
	{`^(?:a|(?:aa))$`, onePass, `a`},
	{`^[a-c]*`, notOnePass, `abcabc`},
	{`^a((b))c$`, onePass, noStr},
	{`^a.[l-nA-Cg-j]?e$`, onePass, noStr},
	{`^a((b))$`, onePass, noStr},
	{`^a(?:(b)|(c))c$`, onePass, noStr},
	{`^a(?:(b*)|(c))c$`, notOnePass, noStr},
	{`^a(?:b|c)$`, onePass, noStr},
	{`^a(?:b?|c)$`, onePass, noStr},
	{`^a(?:b?|c?)$`, notOnePass, noStr},
	{`^a(?:b?|c+)$`, onePass, noStr},
	{`^a(?:b+|(bc))d$`, notOnePass, noStr},
	{`^a(?:bc)+$`, onePass, noStr},
	{`^a(?:[bcd])+$`, onePass, noStr},
	{`^a((?:[bcd])+)$`, onePass, noStr},
	{`^a(:?b|c)*d$`, onePass, `abbbccbbcbbd"`},
	{`^.bc(d|e)*$`, onePass, `abcddddddeeeededd`},
	{`^(?:(?:aa)|.)$`, notOnePass, `a`},
	{`^(?:(?:a{1,2}){1,2})$`, notOnePass, `aaaa`},
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
		onePass = compileOnePass(p)
		if (onePass == notOnePass) != (test.onePass == notOnePass) {
			t.Errorf("CompileOnePass(%q) got %v, expected %v", test.re, onePass, test.onePass)
		}
	}
}
