// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"strings"
	"testing"
	"unicode"
)

type parseTest struct {
	Regexp string
	Dump   string
}

var parseTests = []parseTest{
	// Base cases
	{`a`, `lit{a}`},
	{`a.`, `cat{lit{a}dot{}}`},
	{`a.b`, `cat{lit{a}dot{}lit{b}}`},
	{`ab`, `str{ab}`},
	{`a.b.c`, `cat{lit{a}dot{}lit{b}dot{}lit{c}}`},
	{`abc`, `str{abc}`},
	{`a|^`, `alt{lit{a}bol{}}`},
	{`a|b`, `cc{0x61-0x62}`},
	{`(a)`, `cap{lit{a}}`},
	{`(a)|b`, `alt{cap{lit{a}}lit{b}}`},
	{`a*`, `star{lit{a}}`},
	{`a+`, `plus{lit{a}}`},
	{`a?`, `que{lit{a}}`},
	{`a{2}`, `rep{2,2 lit{a}}`},
	{`a{2,3}`, `rep{2,3 lit{a}}`},
	{`a{2,}`, `rep{2,-1 lit{a}}`},
	{`a*?`, `nstar{lit{a}}`},
	{`a+?`, `nplus{lit{a}}`},
	{`a??`, `nque{lit{a}}`},
	{`a{2}?`, `nrep{2,2 lit{a}}`},
	{`a{2,3}?`, `nrep{2,3 lit{a}}`},
	{`a{2,}?`, `nrep{2,-1 lit{a}}`},
	// Malformed { } are treated as literals.
	{`x{1001`, `str{x{1001}`},
	{`x{9876543210`, `str{x{9876543210}`},
	{`x{9876543210,`, `str{x{9876543210,}`},
	{`x{2,1`, `str{x{2,1}`},
	{`x{1,9876543210`, `str{x{1,9876543210}`},
	{``, `emp{}`},
	{`|`, `emp{}`}, // alt{emp{}emp{}} but got factored
	{`|x|`, `alt{emp{}lit{x}emp{}}`},
	{`.`, `dot{}`},
	{`^`, `bol{}`},
	{`$`, `eol{}`},
	{`\|`, `lit{|}`},
	{`\(`, `lit{(}`},
	{`\)`, `lit{)}`},
	{`\*`, `lit{*}`},
	{`\+`, `lit{+}`},
	{`\?`, `lit{?}`},
	{`{`, `lit{{}`},
	{`}`, `lit{}}`},
	{`\.`, `lit{.}`},
	{`\^`, `lit{^}`},
	{`\$`, `lit{$}`},
	{`\\`, `lit{\}`},
	{`[ace]`, `cc{0x61 0x63 0x65}`},
	{`[abc]`, `cc{0x61-0x63}`},
	{`[a-z]`, `cc{0x61-0x7a}`},
	{`[a]`, `lit{a}`},
	{`\-`, `lit{-}`},
	{`-`, `lit{-}`},
	{`\_`, `lit{_}`},
	{`abc`, `str{abc}`},
	{`abc|def`, `alt{str{abc}str{def}}`},
	{`abc|def|ghi`, `alt{str{abc}str{def}str{ghi}}`},

	// Posix and Perl extensions
	{`[[:lower:]]`, `cc{0x61-0x7a}`},
	{`[a-z]`, `cc{0x61-0x7a}`},
	{`[^[:lower:]]`, `cc{0x0-0x60 0x7b-0x10ffff}`},
	{`[[:^lower:]]`, `cc{0x0-0x60 0x7b-0x10ffff}`},
	{`(?i)[[:lower:]]`, `cc{0x41-0x5a 0x61-0x7a 0x17f 0x212a}`},
	{`(?i)[a-z]`, `cc{0x41-0x5a 0x61-0x7a 0x17f 0x212a}`},
	{`(?i)[^[:lower:]]`, `cc{0x0-0x40 0x5b-0x60 0x7b-0x17e 0x180-0x2129 0x212b-0x10ffff}`},
	{`(?i)[[:^lower:]]`, `cc{0x0-0x40 0x5b-0x60 0x7b-0x17e 0x180-0x2129 0x212b-0x10ffff}`},
	{`\d`, `cc{0x30-0x39}`},
	{`\D`, `cc{0x0-0x2f 0x3a-0x10ffff}`},
	{`\s`, `cc{0x9-0xa 0xc-0xd 0x20}`},
	{`\S`, `cc{0x0-0x8 0xb 0xe-0x1f 0x21-0x10ffff}`},
	{`\w`, `cc{0x30-0x39 0x41-0x5a 0x5f 0x61-0x7a}`},
	{`\W`, `cc{0x0-0x2f 0x3a-0x40 0x5b-0x5e 0x60 0x7b-0x10ffff}`},
	{`(?i)\w`, `cc{0x30-0x39 0x41-0x5a 0x5f 0x61-0x7a 0x17f 0x212a}`},
	{`(?i)\W`, `cc{0x0-0x2f 0x3a-0x40 0x5b-0x5e 0x60 0x7b-0x17e 0x180-0x2129 0x212b-0x10ffff}`},
	{`[^\\]`, `cc{0x0-0x5b 0x5d-0x10ffff}`},
	//	{ `\C`, `byte{}` },  // probably never

	// Unicode, negatives, and a double negative.
	{`\p{Braille}`, `cc{0x2800-0x28ff}`},
	{`\P{Braille}`, `cc{0x0-0x27ff 0x2900-0x10ffff}`},
	{`\p{^Braille}`, `cc{0x0-0x27ff 0x2900-0x10ffff}`},
	{`\P{^Braille}`, `cc{0x2800-0x28ff}`},
	{`\pZ`, `cc{0x20 0xa0 0x1680 0x2000-0x200a 0x2028-0x2029 0x202f 0x205f 0x3000}`},
	{`[\p{Braille}]`, `cc{0x2800-0x28ff}`},
	{`[\P{Braille}]`, `cc{0x0-0x27ff 0x2900-0x10ffff}`},
	{`[\p{^Braille}]`, `cc{0x0-0x27ff 0x2900-0x10ffff}`},
	{`[\P{^Braille}]`, `cc{0x2800-0x28ff}`},
	{`[\pZ]`, `cc{0x20 0xa0 0x1680 0x2000-0x200a 0x2028-0x2029 0x202f 0x205f 0x3000}`},
	{`\p{Lu}`, mkCharClass(unicode.IsUpper)},
	{`[\p{Lu}]`, mkCharClass(unicode.IsUpper)},
	{`(?i)[\p{Lu}]`, mkCharClass(isUpperFold)},
	{`\p{Any}`, `dot{}`},
	{`\p{^Any}`, `cc{}`},

	// Hex, octal.
	{`[\012-\234]\141`, `cat{cc{0xa-0x9c}lit{a}}`},
	{`[\x{41}-\x7a]\x61`, `cat{cc{0x41-0x7a}lit{a}}`},

	// More interesting regular expressions.
	{`a{,2}`, `str{a{,2}}`},
	{`\.\^\$\\`, `str{.^$\}`},
	{`[a-zABC]`, `cc{0x41-0x43 0x61-0x7a}`},
	{`[^a]`, `cc{0x0-0x60 0x62-0x10ffff}`},
	{`[α-ε☺]`, `cc{0x3b1-0x3b5 0x263a}`}, // utf-8
	{`a*{`, `cat{star{lit{a}}lit{{}}`},

	// Test precedences
	{`(?:ab)*`, `star{str{ab}}`},
	{`(ab)*`, `star{cap{str{ab}}}`},
	{`ab|cd`, `alt{str{ab}str{cd}}`},
	{`a(b|c)d`, `cat{lit{a}cap{cc{0x62-0x63}}lit{d}}`},

	// Test flattening.
	{`(?:a)`, `lit{a}`},
	{`(?:ab)(?:cd)`, `str{abcd}`},
	{`(?:a+b+)(?:c+d+)`, `cat{plus{lit{a}}plus{lit{b}}plus{lit{c}}plus{lit{d}}}`},
	{`(?:a+|b+)|(?:c+|d+)`, `alt{plus{lit{a}}plus{lit{b}}plus{lit{c}}plus{lit{d}}}`},
	{`(?:a|b)|(?:c|d)`, `cc{0x61-0x64}`},
	{`a|.`, `dot{}`},
	{`.|a`, `dot{}`},
	{`(?:[abc]|A|Z|hello|world)`, `alt{cc{0x41 0x5a 0x61-0x63}str{hello}str{world}}`},
	{`(?:[abc]|A|Z)`, `cc{0x41 0x5a 0x61-0x63}`},

	// Test Perl quoted literals
	{`\Q+|*?{[\E`, `str{+|*?{[}`},
	{`\Q+\E+`, `plus{lit{+}}`},
	{`\Qab\E+`, `cat{lit{a}plus{lit{b}}}`},
	{`\Q\\E`, `lit{\}`},
	{`\Q\\\E`, `str{\\}`},

	// Test Perl \A and \z
	{`(?m)^`, `bol{}`},
	{`(?m)$`, `eol{}`},
	{`(?-m)^`, `bot{}`},
	{`(?-m)$`, `eot{}`},
	{`(?m)\A`, `bot{}`},
	{`(?m)\z`, `eot{\z}`},
	{`(?-m)\A`, `bot{}`},
	{`(?-m)\z`, `eot{\z}`},

	// Test named captures
	{`(?P<name>a)`, `cap{name:lit{a}}`},

	// Case-folded literals
	{`[Aa]`, `litfold{A}`},
	{`[\x{100}\x{101}]`, `litfold{Ā}`},
	{`[Δδ]`, `litfold{Δ}`},

	// Strings
	{`abcde`, `str{abcde}`},
	{`[Aa][Bb]cd`, `cat{strfold{AB}str{cd}}`},

	// Factoring.
	{`abc|abd|aef|bcx|bcy`, `alt{cat{lit{a}alt{cat{lit{b}cc{0x63-0x64}}str{ef}}}cat{str{bc}cc{0x78-0x79}}}`},
	{`ax+y|ax+z|ay+w`, `cat{lit{a}alt{cat{plus{lit{x}}lit{y}}cat{plus{lit{x}}lit{z}}cat{plus{lit{y}}lit{w}}}}`},

	// Bug fixes.
	{`(?:.)`, `dot{}`},
	{`(?:x|(?:xa))`, `cat{lit{x}alt{emp{}lit{a}}}`},
	{`(?:.|(?:.a))`, `cat{dot{}alt{emp{}lit{a}}}`},
	{`(?:A(?:A|a))`, `cat{lit{A}litfold{A}}`},
	{`(?:A|a)`, `litfold{A}`},
	{`A|(?:A|a)`, `litfold{A}`},
	{`(?s).`, `dot{}`},
	{`(?-s).`, `dnl{}`},
	{`(?:(?:^).)`, `cat{bol{}dot{}}`},
	{`(?-s)(?:(?:^).)`, `cat{bol{}dnl{}}`},

	// RE2 prefix_tests
	{`abc|abd`, `cat{str{ab}cc{0x63-0x64}}`},
	{`a(?:b)c|abd`, `cat{str{ab}cc{0x63-0x64}}`},
	{`abc|abd|aef|bcx|bcy`,
		`alt{cat{lit{a}alt{cat{lit{b}cc{0x63-0x64}}str{ef}}}` +
			`cat{str{bc}cc{0x78-0x79}}}`},
	{`abc|x|abd`, `alt{str{abc}lit{x}str{abd}}`},
	{`(?i)abc|ABD`, `cat{strfold{AB}cc{0x43-0x44 0x63-0x64}}`},
	{`[ab]c|[ab]d`, `cat{cc{0x61-0x62}cc{0x63-0x64}}`},
	{`.c|.d`, `cat{dot{}cc{0x63-0x64}}`},
	{`x{2}|x{2}[0-9]`,
		`cat{rep{2,2 lit{x}}alt{emp{}cc{0x30-0x39}}}`},
	{`x{2}y|x{2}[0-9]y`,
		`cat{rep{2,2 lit{x}}alt{lit{y}cat{cc{0x30-0x39}lit{y}}}}`},
	{`a.*?c|a.*?b`,
		`cat{lit{a}alt{cat{nstar{dot{}}lit{c}}cat{nstar{dot{}}lit{b}}}}`},

	// Valid repetitions.
	{`((((((((((x{2}){2}){2}){2}){2}){2}){2}){2}){2}))`, ``},
	{`((((((((((x{1}){2}){2}){2}){2}){2}){2}){2}){2}){2})`, ``},
}

const testFlags = MatchNL | PerlX | UnicodeGroups

func TestParseSimple(t *testing.T) {
	testParseDump(t, parseTests, testFlags)
}

var foldcaseTests = []parseTest{
	{`AbCdE`, `strfold{ABCDE}`},
	{`[Aa]`, `litfold{A}`},
	{`a`, `litfold{A}`},

	// 0x17F is an old English long s (looks like an f) and folds to s.
	// 0x212A is the Kelvin symbol and folds to k.
	{`A[F-g]`, `cat{litfold{A}cc{0x41-0x7a 0x17f 0x212a}}`}, // [Aa][A-z...]
	{`[[:upper:]]`, `cc{0x41-0x5a 0x61-0x7a 0x17f 0x212a}`},
	{`[[:lower:]]`, `cc{0x41-0x5a 0x61-0x7a 0x17f 0x212a}`},
}

func TestParseFoldCase(t *testing.T) {
	testParseDump(t, foldcaseTests, FoldCase)
}

var literalTests = []parseTest{
	{"(|)^$.[*+?]{5,10},\\", "str{(|)^$.[*+?]{5,10},\\}"},
}

func TestParseLiteral(t *testing.T) {
	testParseDump(t, literalTests, Literal)
}

var matchnlTests = []parseTest{
	{`.`, `dot{}`},
	{"\n", "lit{\n}"},
	{`[^a]`, `cc{0x0-0x60 0x62-0x10ffff}`},
	{`[a\n]`, `cc{0xa 0x61}`},
}

func TestParseMatchNL(t *testing.T) {
	testParseDump(t, matchnlTests, MatchNL)
}

var nomatchnlTests = []parseTest{
	{`.`, `dnl{}`},
	{"\n", "lit{\n}"},
	{`[^a]`, `cc{0x0-0x9 0xb-0x60 0x62-0x10ffff}`},
	{`[a\n]`, `cc{0xa 0x61}`},
}

func TestParseNoMatchNL(t *testing.T) {
	testParseDump(t, nomatchnlTests, 0)
}

// Test Parse -> Dump.
func testParseDump(t *testing.T, tests []parseTest, flags Flags) {
	for _, tt := range tests {
		re, err := Parse(tt.Regexp, flags)
		if err != nil {
			t.Errorf("Parse(%#q): %v", tt.Regexp, err)
			continue
		}
		if tt.Dump == "" {
			// It parsed. That's all we care about.
			continue
		}
		d := dump(re)
		if d != tt.Dump {
			t.Errorf("Parse(%#q).Dump() = %#q want %#q", tt.Regexp, d, tt.Dump)
		}
	}
}

// dump prints a string representation of the regexp showing
// the structure explicitly.
func dump(re *Regexp) string {
	var b strings.Builder
	dumpRegexp(&b, re)
	return b.String()
}

var opNames = []string{
	OpNoMatch:        "no",
	OpEmptyMatch:     "emp",
	OpLiteral:        "lit",
	OpCharClass:      "cc",
	OpAnyCharNotNL:   "dnl",
	OpAnyChar:        "dot",
	OpBeginLine:      "bol",
	OpEndLine:        "eol",
	OpBeginText:      "bot",
	OpEndText:        "eot",
	OpWordBoundary:   "wb",
	OpNoWordBoundary: "nwb",
	OpCapture:        "cap",
	OpStar:           "star",
	OpPlus:           "plus",
	OpQuest:          "que",
	OpRepeat:         "rep",
	OpConcat:         "cat",
	OpAlternate:      "alt",
}

// dumpRegexp writes an encoding of the syntax tree for the regexp re to b.
// It is used during testing to distinguish between parses that might print
// the same using re's String method.
func dumpRegexp(b *strings.Builder, re *Regexp) {
	if int(re.Op) >= len(opNames) || opNames[re.Op] == "" {
		fmt.Fprintf(b, "op%d", re.Op)
	} else {
		switch re.Op {
		default:
			b.WriteString(opNames[re.Op])
		case OpStar, OpPlus, OpQuest, OpRepeat:
			if re.Flags&NonGreedy != 0 {
				b.WriteByte('n')
			}
			b.WriteString(opNames[re.Op])
		case OpLiteral:
			if len(re.Rune) > 1 {
				b.WriteString("str")
			} else {
				b.WriteString("lit")
			}
			if re.Flags&FoldCase != 0 {
				for _, r := range re.Rune {
					if unicode.SimpleFold(r) != r {
						b.WriteString("fold")
						break
					}
				}
			}
		}
	}
	b.WriteByte('{')
	switch re.Op {
	case OpEndText:
		if re.Flags&WasDollar == 0 {
			b.WriteString(`\z`)
		}
	case OpLiteral:
		for _, r := range re.Rune {
			b.WriteRune(r)
		}
	case OpConcat, OpAlternate:
		for _, sub := range re.Sub {
			dumpRegexp(b, sub)
		}
	case OpStar, OpPlus, OpQuest:
		dumpRegexp(b, re.Sub[0])
	case OpRepeat:
		fmt.Fprintf(b, "%d,%d ", re.Min, re.Max)
		dumpRegexp(b, re.Sub[0])
	case OpCapture:
		if re.Name != "" {
			b.WriteString(re.Name)
			b.WriteByte(':')
		}
		dumpRegexp(b, re.Sub[0])
	case OpCharClass:
		sep := ""
		for i := 0; i < len(re.Rune); i += 2 {
			b.WriteString(sep)
			sep = " "
			lo, hi := re.Rune[i], re.Rune[i+1]
			if lo == hi {
				fmt.Fprintf(b, "%#x", lo)
			} else {
				fmt.Fprintf(b, "%#x-%#x", lo, hi)
			}
		}
	}
	b.WriteByte('}')
}

func mkCharClass(f func(rune) bool) string {
	re := &Regexp{Op: OpCharClass}
	lo := rune(-1)
	for i := rune(0); i <= unicode.MaxRune; i++ {
		if f(i) {
			if lo < 0 {
				lo = i
			}
		} else {
			if lo >= 0 {
				re.Rune = append(re.Rune, lo, i-1)
				lo = -1
			}
		}
	}
	if lo >= 0 {
		re.Rune = append(re.Rune, lo, unicode.MaxRune)
	}
	return dump(re)
}

func isUpperFold(r rune) bool {
	if unicode.IsUpper(r) {
		return true
	}
	c := unicode.SimpleFold(r)
	for c != r {
		if unicode.IsUpper(c) {
			return true
		}
		c = unicode.SimpleFold(c)
	}
	return false
}

func TestFoldConstants(t *testing.T) {
	last := rune(-1)
	for i := rune(0); i <= unicode.MaxRune; i++ {
		if unicode.SimpleFold(i) == i {
			continue
		}
		if last == -1 && minFold != i {
			t.Errorf("minFold=%#U should be %#U", minFold, i)
		}
		last = i
	}
	if maxFold != last {
		t.Errorf("maxFold=%#U should be %#U", maxFold, last)
	}
}

func TestAppendRangeCollapse(t *testing.T) {
	// AppendRange should collapse each of the new ranges
	// into the earlier ones (it looks back two ranges), so that
	// the slice never grows very large.
	// Note that we are not calling cleanClass.
	var r []rune
	for i := rune('A'); i <= 'Z'; i++ {
		r = appendRange(r, i, i)
		r = appendRange(r, i+'a'-'A', i+'a'-'A')
	}
	if string(r) != "AZaz" {
		t.Errorf("appendRange interlaced A-Z a-z = %s, want AZaz", string(r))
	}
}

var invalidRegexps = []string{
	`(`,
	`)`,
	`(a`,
	`a)`,
	`(a))`,
	`(a|b|`,
	`a|b|)`,
	`(a|b|))`,
	`(a|b`,
	`a|b)`,
	`(a|b))`,
	`[a-z`,
	`([a-z)`,
	`[a-z)`,
	`([a-z]))`,
	`x{1001}`,
	`x{9876543210}`,
	`x{2,1}`,
	`x{1,9876543210}`,
	"\xff", // Invalid UTF-8
	"[\xff]",
	"[\\\xff]",
	"\\\xff",
	`(?P<name>a`,
	`(?P<name>`,
	`(?P<name`,
	`(?P<x y>a)`,
	`(?P<>a)`,
	`[a-Z]`,
	`(?i)[a-Z]`,
	`a{100000}`,
	`a{100000,}`,
	"((((((((((x{2}){2}){2}){2}){2}){2}){2}){2}){2}){2})",
	`\Q\E*`,
}

var onlyPerl = []string{
	`[a-b-c]`,
	`\Qabc\E`,
	`\Q*+?{[\E`,
	`\Q\\E`,
	`\Q\\\E`,
	`\Q\\\\E`,
	`\Q\\\\\E`,
	`(?:a)`,
	`(?P<name>a)`,
}

var onlyPOSIX = []string{
	"a++",
	"a**",
	"a?*",
	"a+*",
	"a{1}*",
	".{1}{2}.{3}",
}

func TestParseInvalidRegexps(t *testing.T) {
	for _, regexp := range invalidRegexps {
		if re, err := Parse(regexp, Perl); err == nil {
			t.Errorf("Parse(%#q, Perl) = %s, should have failed", regexp, dump(re))
		}
		if re, err := Parse(regexp, POSIX); err == nil {
			t.Errorf("Parse(%#q, POSIX) = %s, should have failed", regexp, dump(re))
		}
	}
	for _, regexp := range onlyPerl {
		if _, err := Parse(regexp, Perl); err != nil {
			t.Errorf("Parse(%#q, Perl): %v", regexp, err)
		}
		if re, err := Parse(regexp, POSIX); err == nil {
			t.Errorf("Parse(%#q, POSIX) = %s, should have failed", regexp, dump(re))
		}
	}
	for _, regexp := range onlyPOSIX {
		if re, err := Parse(regexp, Perl); err == nil {
			t.Errorf("Parse(%#q, Perl) = %s, should have failed", regexp, dump(re))
		}
		if _, err := Parse(regexp, POSIX); err != nil {
			t.Errorf("Parse(%#q, POSIX): %v", regexp, err)
		}
	}
}

func TestToStringEquivalentParse(t *testing.T) {
	for _, tt := range parseTests {
		re, err := Parse(tt.Regexp, testFlags)
		if err != nil {
			t.Errorf("Parse(%#q): %v", tt.Regexp, err)
			continue
		}
		if tt.Dump == "" {
			// It parsed. That's all we care about.
			continue
		}
		d := dump(re)
		if d != tt.Dump {
			t.Errorf("Parse(%#q).Dump() = %#q want %#q", tt.Regexp, d, tt.Dump)
			continue
		}

		s := re.String()
		if s != tt.Regexp {
			// If ToString didn't return the original regexp,
			// it must have found one with fewer parens.
			// Unfortunately we can't check the length here, because
			// ToString produces "\\{" for a literal brace,
			// but "{" is a shorter equivalent in some contexts.
			nre, err := Parse(s, testFlags)
			if err != nil {
				t.Errorf("Parse(%#q.String() = %#q): %v", tt.Regexp, s, err)
				continue
			}
			nd := dump(nre)
			if d != nd {
				t.Errorf("Parse(%#q) -> %#q; %#q vs %#q", tt.Regexp, s, d, nd)
			}

			ns := nre.String()
			if s != ns {
				t.Errorf("Parse(%#q) -> %#q -> %#q", tt.Regexp, s, ns)
			}
		}
	}
}
