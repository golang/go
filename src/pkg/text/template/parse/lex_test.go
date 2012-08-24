// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

import (
	"testing"
)

type lexTest struct {
	name  string
	input string
	items []item
}

var (
	tEOF      = item{itemEOF, 0, ""}
	tLeft     = item{itemLeftDelim, 0, "{{"}
	tRight    = item{itemRightDelim, 0, "}}"}
	tRange    = item{itemRange, 0, "range"}
	tPipe     = item{itemPipe, 0, "|"}
	tFor      = item{itemIdentifier, 0, "for"}
	tQuote    = item{itemString, 0, `"abc \n\t\" "`}
	raw       = "`" + `abc\n\t\" ` + "`"
	tRawQuote = item{itemRawString, 0, raw}
)

var lexTests = []lexTest{
	{"empty", "", []item{tEOF}},
	{"spaces", " \t\n", []item{{itemText, 0, " \t\n"}, tEOF}},
	{"text", `now is the time`, []item{{itemText, 0, "now is the time"}, tEOF}},
	{"text with comment", "hello-{{/* this is a comment */}}-world", []item{
		{itemText, 0, "hello-"},
		{itemText, 0, "-world"},
		tEOF,
	}},
	{"punctuation", "{{,@%}}", []item{
		tLeft,
		{itemChar, 0, ","},
		{itemChar, 0, "@"},
		{itemChar, 0, "%"},
		tRight,
		tEOF,
	}},
	{"parens", "{{((3))}}", []item{
		tLeft,
		{itemLeftParen, 0, "("},
		{itemLeftParen, 0, "("},
		{itemNumber, 0, "3"},
		{itemRightParen, 0, ")"},
		{itemRightParen, 0, ")"},
		tRight,
		tEOF,
	}},
	{"empty action", `{{}}`, []item{tLeft, tRight, tEOF}},
	{"for", `{{for }}`, []item{tLeft, tFor, tRight, tEOF}},
	{"quote", `{{"abc \n\t\" "}}`, []item{tLeft, tQuote, tRight, tEOF}},
	{"raw quote", "{{" + raw + "}}", []item{tLeft, tRawQuote, tRight, tEOF}},
	{"numbers", "{{1 02 0x14 -7.2i 1e3 +1.2e-4 4.2i 1+2i}}", []item{
		tLeft,
		{itemNumber, 0, "1"},
		{itemNumber, 0, "02"},
		{itemNumber, 0, "0x14"},
		{itemNumber, 0, "-7.2i"},
		{itemNumber, 0, "1e3"},
		{itemNumber, 0, "+1.2e-4"},
		{itemNumber, 0, "4.2i"},
		{itemComplex, 0, "1+2i"},
		tRight,
		tEOF,
	}},
	{"characters", `{{'a' '\n' '\'' '\\' '\u00FF' '\xFF' '本'}}`, []item{
		tLeft,
		{itemCharConstant, 0, `'a'`},
		{itemCharConstant, 0, `'\n'`},
		{itemCharConstant, 0, `'\''`},
		{itemCharConstant, 0, `'\\'`},
		{itemCharConstant, 0, `'\u00FF'`},
		{itemCharConstant, 0, `'\xFF'`},
		{itemCharConstant, 0, `'本'`},
		tRight,
		tEOF,
	}},
	{"bools", "{{true false}}", []item{
		tLeft,
		{itemBool, 0, "true"},
		{itemBool, 0, "false"},
		tRight,
		tEOF,
	}},
	{"dot", "{{.}}", []item{
		tLeft,
		{itemDot, 0, "."},
		tRight,
		tEOF,
	}},
	{"nil", "{{nil}}", []item{
		tLeft,
		{itemNil, 0, "nil"},
		tRight,
		tEOF,
	}},
	{"dots", "{{.x . .2 .x.y}}", []item{
		tLeft,
		{itemField, 0, ".x"},
		{itemDot, 0, "."},
		{itemNumber, 0, ".2"},
		{itemField, 0, ".x.y"},
		tRight,
		tEOF,
	}},
	{"keywords", "{{range if else end with}}", []item{
		tLeft,
		{itemRange, 0, "range"},
		{itemIf, 0, "if"},
		{itemElse, 0, "else"},
		{itemEnd, 0, "end"},
		{itemWith, 0, "with"},
		tRight,
		tEOF,
	}},
	{"variables", "{{$c := printf $ $hello $23 $ $var.Field .Method}}", []item{
		tLeft,
		{itemVariable, 0, "$c"},
		{itemColonEquals, 0, ":="},
		{itemIdentifier, 0, "printf"},
		{itemVariable, 0, "$"},
		{itemVariable, 0, "$hello"},
		{itemVariable, 0, "$23"},
		{itemVariable, 0, "$"},
		{itemVariable, 0, "$var.Field"},
		{itemField, 0, ".Method"},
		tRight,
		tEOF,
	}},
	{"pipeline", `intro {{echo hi 1.2 |noargs|args 1 "hi"}} outro`, []item{
		{itemText, 0, "intro "},
		tLeft,
		{itemIdentifier, 0, "echo"},
		{itemIdentifier, 0, "hi"},
		{itemNumber, 0, "1.2"},
		tPipe,
		{itemIdentifier, 0, "noargs"},
		tPipe,
		{itemIdentifier, 0, "args"},
		{itemNumber, 0, "1"},
		{itemString, 0, `"hi"`},
		tRight,
		{itemText, 0, " outro"},
		tEOF,
	}},
	{"declaration", "{{$v := 3}}", []item{
		tLeft,
		{itemVariable, 0, "$v"},
		{itemColonEquals, 0, ":="},
		{itemNumber, 0, "3"},
		tRight,
		tEOF,
	}},
	{"2 declarations", "{{$v , $w := 3}}", []item{
		tLeft,
		{itemVariable, 0, "$v"},
		{itemChar, 0, ","},
		{itemVariable, 0, "$w"},
		{itemColonEquals, 0, ":="},
		{itemNumber, 0, "3"},
		tRight,
		tEOF,
	}},
	// errors
	{"badchar", "#{{\x01}}", []item{
		{itemText, 0, "#"},
		tLeft,
		{itemError, 0, "unrecognized character in action: U+0001"},
	}},
	{"unclosed action", "{{\n}}", []item{
		tLeft,
		{itemError, 0, "unclosed action"},
	}},
	{"EOF in action", "{{range", []item{
		tLeft,
		tRange,
		{itemError, 0, "unclosed action"},
	}},
	{"unclosed quote", "{{\"\n\"}}", []item{
		tLeft,
		{itemError, 0, "unterminated quoted string"},
	}},
	{"unclosed raw quote", "{{`xx\n`}}", []item{
		tLeft,
		{itemError, 0, "unterminated raw quoted string"},
	}},
	{"unclosed char constant", "{{'\n}}", []item{
		tLeft,
		{itemError, 0, "unterminated character constant"},
	}},
	{"bad number", "{{3k}}", []item{
		tLeft,
		{itemError, 0, `bad number syntax: "3k"`},
	}},
	{"unclosed paren", "{{(3}}", []item{
		tLeft,
		{itemLeftParen, 0, "("},
		{itemNumber, 0, "3"},
		{itemError, 0, `unclosed left paren`},
	}},
	{"extra right paren", "{{3)}}", []item{
		tLeft,
		{itemNumber, 0, "3"},
		{itemRightParen, 0, ")"},
		{itemError, 0, `unexpected right paren U+0029 ')'`},
	}},

	// Fixed bugs
	// Many elements in an action blew the lookahead until
	// we made lexInsideAction not loop.
	{"long pipeline deadlock", "{{|||||}}", []item{
		tLeft,
		tPipe,
		tPipe,
		tPipe,
		tPipe,
		tPipe,
		tRight,
		tEOF,
	}},
	{"text with bad comment", "hello-{{/*/}}-world", []item{
		{itemText, 0, "hello-"},
		{itemError, 0, `unclosed comment`},
	}},
}

// collect gathers the emitted items into a slice.
func collect(t *lexTest, left, right string) (items []item) {
	l := lex(t.name, t.input, left, right)
	for {
		item := l.nextItem()
		items = append(items, item)
		if item.typ == itemEOF || item.typ == itemError {
			break
		}
	}
	return
}

func equal(i1, i2 []item, checkPos bool) bool {
	if len(i1) != len(i2) {
		return false
	}
	for k := range i1 {
		if i1[k].typ != i2[k].typ {
			return false
		}
		if i1[k].val != i2[k].val {
			return false
		}
		if checkPos && i1[k].pos != i2[k].pos {
			return false
		}
	}
	return true
}

func TestLex(t *testing.T) {
	for _, test := range lexTests {
		items := collect(&test, "", "")
		if !equal(items, test.items, false) {
			t.Errorf("%s: got\n\t%v\nexpected\n\t%v", test.name, items, test.items)
		}
	}
}

// Some easy cases from above, but with delimiters $$ and @@
var lexDelimTests = []lexTest{
	{"punctuation", "$$,@%{{}}@@", []item{
		tLeftDelim,
		{itemChar, 0, ","},
		{itemChar, 0, "@"},
		{itemChar, 0, "%"},
		{itemChar, 0, "{"},
		{itemChar, 0, "{"},
		{itemChar, 0, "}"},
		{itemChar, 0, "}"},
		tRightDelim,
		tEOF,
	}},
	{"empty action", `$$@@`, []item{tLeftDelim, tRightDelim, tEOF}},
	{"for", `$$for @@`, []item{tLeftDelim, tFor, tRightDelim, tEOF}},
	{"quote", `$$"abc \n\t\" "@@`, []item{tLeftDelim, tQuote, tRightDelim, tEOF}},
	{"raw quote", "$$" + raw + "@@", []item{tLeftDelim, tRawQuote, tRightDelim, tEOF}},
}

var (
	tLeftDelim  = item{itemLeftDelim, 0, "$$"}
	tRightDelim = item{itemRightDelim, 0, "@@"}
)

func TestDelims(t *testing.T) {
	for _, test := range lexDelimTests {
		items := collect(&test, "$$", "@@")
		if !equal(items, test.items, false) {
			t.Errorf("%s: got\n\t%v\nexpected\n\t%v", test.name, items, test.items)
		}
	}
}

var lexPosTests = []lexTest{
	{"empty", "", []item{tEOF}},
	{"punctuation", "{{,@%#}}", []item{
		{itemLeftDelim, 0, "{{"},
		{itemChar, 2, ","},
		{itemChar, 3, "@"},
		{itemChar, 4, "%"},
		{itemChar, 5, "#"},
		{itemRightDelim, 6, "}}"},
		{itemEOF, 8, ""},
	}},
	{"sample", "0123{{hello}}xyz", []item{
		{itemText, 0, "0123"},
		{itemLeftDelim, 4, "{{"},
		{itemIdentifier, 6, "hello"},
		{itemRightDelim, 11, "}}"},
		{itemText, 13, "xyz"},
		{itemEOF, 16, ""},
	}},
}

// The other tests don't check position, to make the test cases easier to construct.
// This one does.
func TestPos(t *testing.T) {
	for _, test := range lexPosTests {
		items := collect(&test, "", "")
		if !equal(items, test.items, true) {
			t.Errorf("%s: got\n\t%v\nexpected\n\t%v", test.name, items, test.items)
			if len(items) == len(test.items) {
				// Detailed print; avoid item.String() to expose the position value.
				for i := range items {
					if !equal(items[i:i+1], test.items[i:i+1], true) {
						i1 := items[i]
						i2 := test.items[i]
						t.Errorf("\t#%d: got {%v %d %q} expected  {%v %d %q}", i, i1.typ, i1.pos, i1.val, i2.typ, i2.pos, i2.val)
					}
				}
			}
		}
	}
}
