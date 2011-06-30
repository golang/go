// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"fmt"
	"testing"
	"unicode"
)

var parseTests = []struct {
	Regexp string
	Dump   string
}{
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
	{`\pZ`, `cc{0x20 0xa0 0x1680 0x180e 0x2000-0x200a 0x2028-0x2029 0x202f 0x205f 0x3000}`},
	{`[\p{Braille}]`, `cc{0x2800-0x28ff}`},
	{`[\P{Braille}]`, `cc{0x0-0x27ff 0x2900-0x10ffff}`},
	{`[\p{^Braille}]`, `cc{0x0-0x27ff 0x2900-0x10ffff}`},
	{`[\P{^Braille}]`, `cc{0x2800-0x28ff}`},
	{`[\pZ]`, `cc{0x20 0xa0 0x1680 0x180e 0x2000-0x200a 0x2028-0x2029 0x202f 0x205f 0x3000}`},
	{`\p{Lu}`, mkCharClass(unicode.IsUpper)},
	{`[\p{Lu}]`, mkCharClass(unicode.IsUpper)},
	{`(?i)[\p{Lu}]`, mkCharClass(isUpperFold)},

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
	{`ax+y|ax+z|ay+w`, `cat{lit{a}alt{cat{plus{lit{x}}cc{0x79-0x7a}}cat{plus{lit{y}}lit{w}}}}`},
}

const testFlags = MatchNL | PerlX | UnicodeGroups

// Test Parse -> Dump.
func TestParseDump(t *testing.T) {
	for _, tt := range parseTests {
		re, err := Parse(tt.Regexp, testFlags)
		if err != nil {
			t.Errorf("Parse(%#q): %v", tt.Regexp, err)
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
	var b bytes.Buffer
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
func dumpRegexp(b *bytes.Buffer, re *Regexp) {
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

func mkCharClass(f func(int) bool) string {
	re := &Regexp{Op: OpCharClass}
	lo := -1
	for i := 0; i <= unicode.MaxRune; i++ {
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

func isUpperFold(rune int) bool {
	if unicode.IsUpper(rune) {
		return true
	}
	c := unicode.SimpleFold(rune)
	for c != rune {
		if unicode.IsUpper(c) {
			return true
		}
		c = unicode.SimpleFold(c)
	}
	return false
}

func TestFoldConstants(t *testing.T) {
	last := -1
	for i := 0; i <= unicode.MaxRune; i++ {
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
	var r []int
	for i := 'A'; i <= 'Z'; i++ {
		r = appendRange(r, i, i)
		r = appendRange(r, i+'a'-'A', i+'a'-'A')
	}
	if string(r) != "AZaz" {
		t.Errorf("appendRange interlaced A-Z a-z = %s, want AZaz", string(r))
	}
}
