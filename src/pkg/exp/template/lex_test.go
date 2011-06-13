// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"reflect"
	"testing"
)

// Make the types prettyprint.
var itemName = map[itemType]string{
	itemError:      "Error",
	itemText:       "Text",
	itemLeftMeta:   "LeftMeta",
	itemRightMeta:  "RightMeta",
	itemPipe:       "Pipe",
	itemIdentifier: "Identifier",
	itemNumber:     "Number",
	itemRawString:  "RawString",
	itemString:     "String",
	itemEOF:        "EOF",
}

func (i itemType) String() string {
	s := itemName[i]
	if s == "" {
		return fmt.Sprintf("item%d", int(i))
	}
	return s
}

type lexTest struct {
	name  string
	input string
	items []item
}

var (
	tEOF      = item{itemEOF, ""}
	tLeft     = item{itemLeftMeta, "{{"}
	tRight    = item{itemRightMeta, "}}"}
	tPipe     = item{itemPipe, "|"}
	tFor      = item{itemIdentifier, "for"}
	tQuote    = item{itemString, `"abc \n\t\" "`}
	raw       = "`" + `abc\n\t\" ` + "`"
	tRawQuote = item{itemRawString, raw}
)

var lexTests = []lexTest{
	{"empty", "", []item{tEOF}},
	{"spaces", " \t\n", []item{{itemText, " \t\n"}, tEOF}},
	{"text", `now is the time`, []item{{itemText, "now is the time"}, tEOF}},
	{"empty action", `{{}}`, []item{tLeft, tRight, tEOF}},
	{"for", `{{for }}`, []item{tLeft, tFor, tRight, tEOF}},
	{"quote", `{{"abc \n\t\" "}}`, []item{tLeft, tQuote, tRight, tEOF}},
	{"raw quote", "{{" + raw + "}}", []item{tLeft, tRawQuote, tRight, tEOF}},
	{"numbers", "{{1 02 0x14 -7.2i 1e3 +1.2e-4}}", []item{
		tLeft,
		{itemNumber, "1"},
		{itemNumber, "02"},
		{itemNumber, "0x14"},
		{itemNumber, "-7.2i"},
		{itemNumber, "1e3"},
		{itemNumber, "+1.2e-4"},
		tRight,
		tEOF,
	}},
	{"pipeline", `intro {{echo hi 1.2 |noargs|args 1 "hi"}} outro`, []item{
		{itemText, "intro "},
		tLeft,
		{itemIdentifier, "echo"},
		{itemIdentifier, "hi"},
		{itemNumber, "1.2"},
		tPipe,
		{itemIdentifier, "noargs"},
		tPipe,
		{itemIdentifier, "args"},
		{itemNumber, "1"},
		{itemString, `"hi"`},
		tRight,
		{itemText, " outro"},
		tEOF,
	}},
	// errors
	{"badchar", "#{{#}}", []item{
		{itemText, "#"},
		tLeft,
		{itemError, "badchar:1 unrecognized character in action: U+0023 '#'"},
	}},
	{"unclosed action", "{{\n}}", []item{
		tLeft,
		{itemError, "unclosed action:2 unclosed action"},
	}},
	{"unclosed quote", "{{\"\n\"}}", []item{
		tLeft,
		{itemError, "unclosed quote:2 unterminated quoted string"},
	}},
	{"unclosed raw quote", "{{`xx\n`}}", []item{
		tLeft,
		{itemError, "unclosed raw quote:2 unterminated raw quoted string"},
	}},
	{"bad number", "{{3k}}", []item{
		tLeft,
		{itemError, `bad number:1 bad number syntax: "3k"`},
	}},
}

// collect gathers the emitted items into a slice.
func collect(t *lexTest) (items []item) {
	for i := range lex(t.name, t.input) {
		items = append(items, i)
	}
	return
}

func TestLex(t *testing.T) {
	for _, test := range lexTests {
		items := collect(&test)
		if !reflect.DeepEqual(items, test.items) {
			t.Errorf("%s: got\n\t%v; expected\n\t%v", test.name, items, test.items)
		}
	}
}
