// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"reflect"
	"testing"
)

var compileTests = []struct {
	Regexp string
	Prog   string
}{
	{"a", `  0	fail
  1*	rune1 "a" -> 2
  2	match
`},
	{"[A-M][n-z]", `  0	fail
  1*	rune "AM" -> 2
  2	rune "nz" -> 3
  3	match
`},
	{"", `  0	fail
  1*	nop -> 2
  2	match
`},
	{"a?", `  0	fail
  1	rune1 "a" -> 3
  2*	alt -> 1, 3
  3	match
`},
	{"a??", `  0	fail
  1	rune1 "a" -> 3
  2*	alt -> 3, 1
  3	match
`},
	{"a+", `  0	fail
  1*	rune1 "a" -> 2
  2	alt -> 1, 3
  3	match
`},
	{"a+?", `  0	fail
  1*	rune1 "a" -> 2
  2	alt -> 3, 1
  3	match
`},
	{"a*", `  0	fail
  1	rune1 "a" -> 2
  2*	alt -> 1, 3
  3	match
`},
	{"a*?", `  0	fail
  1	rune1 "a" -> 2
  2*	alt -> 3, 1
  3	match
`},
	{"a+b+", `  0	fail
  1*	rune1 "a" -> 2
  2	alt -> 1, 3
  3	rune1 "b" -> 4
  4	alt -> 3, 5
  5	match
`},
	{"(a+)(b+)", `  0	fail
  1*	cap 2 -> 2
  2	rune1 "a" -> 3
  3	alt -> 2, 4
  4	cap 3 -> 5
  5	cap 4 -> 6
  6	rune1 "b" -> 7
  7	alt -> 6, 8
  8	cap 5 -> 9
  9	match
`},
	{"a+|b+", `  0	fail
  1	rune1 "a" -> 2
  2	alt -> 1, 6
  3	rune1 "b" -> 4
  4	alt -> 3, 6
  5*	alt -> 1, 3
  6	match
`},
	{"A[Aa]", `  0	fail
  1*	rune1 "A" -> 2
  2	rune "A"/i -> 3
  3	match
`},
	{"(?:(?:^).)", `  0	fail
  1*	empty 4 -> 2
  2	anynotnl -> 3
  3	match
`},
}

func TestCompile(t *testing.T) {
	for _, tt := range compileTests {
		re, _ := Parse(tt.Regexp, Perl)
		p, _ := Compile(re)
		s := p.String()
		if s != tt.Prog {
			t.Errorf("compiled %#q:\n--- have\n%s---\n--- want\n%s---", tt.Regexp, s, tt.Prog)
		}
	}
}

func BenchmarkEmptyOpContext(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var r1 rune = -1
		for _, r2 := range "foo, bar, baz\nsome input text.\n" {
			EmptyOpContext(r1, r2)
			r1 = r2
		}
		EmptyOpContext(r1, -1)
	}
}

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

var onePass = &Prog{}

var onePassTests = []struct {
	re      string
	onePass *Prog
	prog    string
}{
	{`^(?:a|(?:a*))$`, NotOnePass, noStr},
	{`^(?:(a)|(?:a*))$`, NotOnePass, noStr},
	{`^(?:(?:(?:.(?:$))?))$`, onePass, `a`},
	{`^abcd$`, onePass, `abcd`},
	{`^abcd$`, onePass, `abcde`},
	{`^(?:(?:a{0,})*?)$`, onePass, `a`},
	{`^(?:(?:a+)*)$`, onePass, ``},
	{`^(?:(?:a|(?:aa)))$`, onePass, ``},
	{`^(?:[^\s\S])$`, onePass, ``},
	{`^(?:(?:a{3,4}){0,})$`, NotOnePass, `aaaaaa`},
	{`^(?:(?:a+)*)$`, onePass, `a`},
	{`^(?:(?:(?:a*)+))$`, onePass, noStr},
	{`^(?:(?:a+)*)$`, onePass, ``},
	{`^[a-c]+$`, onePass, `abc`},
	{`^[a-c]*$`, onePass, `abcdabc`},
	{`^(?:a*)$`, onePass, `aaaaaaa`},
	{`^(?:(?:aa)|a)$`, onePass, `a`},
	{`^[a-c]*`, NotOnePass, `abcdabc`},
	{`^[a-c]*$`, onePass, `abc`},
	{`^...$`, onePass, ``},
	{`^(?:a|(?:aa))$`, onePass, `a`},
	{`^[a-c]*`, NotOnePass, `abcabc`},
	{`^a((b))c$`, onePass, noStr},
	{`^a.[l-nA-Cg-j]?e$`, onePass, noStr},
	{`^a((b))$`, onePass, noStr},
	{`^a(?:(b)|(c))c$`, onePass, noStr},
	{`^a(?:(b*)|(c))c$`, NotOnePass, noStr},
	{`^a(?:b|c)$`, onePass, noStr},
	{`^a(?:b?|c)$`, onePass, noStr},
	{`^a(?:b?|c?)$`, NotOnePass, noStr},
	{`^a(?:b?|c+)$`, onePass, noStr},
	{`^a(?:b+|(bc))d$`, NotOnePass, noStr},
	{`^a(?:bc)+$`, onePass, noStr},
	{`^a(?:[bcd])+$`, onePass, noStr},
	{`^a((?:[bcd])+)$`, onePass, noStr},
	{`^a(:?b|c)*d$`, onePass, `abbbccbbcbbd"`},
	{`^.bc(d|e)*$`, onePass, `abcddddddeeeededd`},
	{`^(?:(?:aa)|.)$`, NotOnePass, `a`},
	{`^(?:(?:a{1,2}){1,2})$`, NotOnePass, `aaaa`},
}

func TestCompileOnePass(t *testing.T) {
	var (
		p   *Prog
		re  *Regexp
		err error
	)
	for _, test := range onePassTests {
		if re, err = Parse(test.re, Perl); err != nil {
			t.Errorf("Parse(%q) got err:%s, want success", test.re, err)
			continue
		}
		// needs to be done before compile...
		re = re.Simplify()
		if p, err = Compile(re); err != nil {
			t.Errorf("Compile(%q) got err:%s, want success", test.re, err)
			continue
		}
		onePass = p.CompileOnePass()
		if (onePass == NotOnePass) != (test.onePass == NotOnePass) {
			t.Errorf("CompileOnePass(%q) got %v, expected %v", test.re, onePass, test.onePass)
		}
	}
}
