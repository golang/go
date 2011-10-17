// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csv

import (
	"reflect"
	"strings"
	"testing"
)

var readTests = []struct {
	Name               string
	Input              string
	Output             [][]string
	UseFieldsPerRecord bool // false (default) means FieldsPerRecord is -1

	// These fields are copied into the Reader
	Comma            int
	Comment          int
	FieldsPerRecord  int
	LazyQuotes       bool
	TrailingComma    bool
	TrimLeadingSpace bool

	Error  string
	Line   int // Expected error line if != 0
	Column int // Expected error column if line != 0
}{
	{
		Name:   "Simple",
		Input:  "a,b,c\n",
		Output: [][]string{{"a", "b", "c"}},
	},
	{
		Name:   "CRLF",
		Input:  "a,b\r\nc,d\r\n",
		Output: [][]string{{"a", "b"}, {"c", "d"}},
	},
	{
		Name:   "BareCR",
		Input:  "a,b\rc,d\r\n",
		Output: [][]string{{"a", "b\rc", "d"}},
	},
	{
		Name:               "RFC4180test",
		UseFieldsPerRecord: true,
		Input: `#field1,field2,field3
"aaa","bb
b","ccc"
"a,a","b""bb","ccc"
zzz,yyy,xxx
`,
		Output: [][]string{
			{"#field1", "field2", "field3"},
			{"aaa", "bb\nb", "ccc"},
			{"a,a", `b"bb`, "ccc"},
			{"zzz", "yyy", "xxx"},
		},
	},
	{
		Name:   "NoEOLTest",
		Input:  "a,b,c",
		Output: [][]string{{"a", "b", "c"}},
	},
	{
		Name:   "Semicolon",
		Comma:  ';',
		Input:  "a;b;c\n",
		Output: [][]string{{"a", "b", "c"}},
	},
	{
		Name: "MultiLine",
		Input: `"two
line","one line","three
line
field"`,
		Output: [][]string{{"two\nline", "one line", "three\nline\nfield"}},
	},
	{
		Name:  "BlankLine",
		Input: "a,b,c\n\nd,e,f\n\n",
		Output: [][]string{
			{"a", "b", "c"},
			{"d", "e", "f"},
		},
	},
	{
		Name:             "TrimSpace",
		Input:            " a,  b,   c\n",
		TrimLeadingSpace: true,
		Output:           [][]string{{"a", "b", "c"}},
	},
	{
		Name:   "LeadingSpace",
		Input:  " a,  b,   c\n",
		Output: [][]string{{" a", "  b", "   c"}},
	},
	{
		Name:    "Comment",
		Comment: '#',
		Input:   "#1,2,3\na,b,c\n#comment",
		Output:  [][]string{{"a", "b", "c"}},
	},
	{
		Name:   "NoComment",
		Input:  "#1,2,3\na,b,c",
		Output: [][]string{{"#1", "2", "3"}, {"a", "b", "c"}},
	},
	{
		Name:       "LazyQuotes",
		LazyQuotes: true,
		Input:      `a "word","1"2",a","b`,
		Output:     [][]string{{`a "word"`, `1"2`, `a"`, `b`}},
	},
	{
		Name:       "BareQuotes",
		LazyQuotes: true,
		Input:      `a "word","1"2",a"`,
		Output:     [][]string{{`a "word"`, `1"2`, `a"`}},
	},
	{
		Name:       "BareDoubleQuotes",
		LazyQuotes: true,
		Input:      `a""b,c`,
		Output:     [][]string{{`a""b`, `c`}},
	},
	{
		Name:  "BadDoubleQuotes",
		Input: `a""b,c`,
		Error: `bare " in non-quoted-field`, Line: 1, Column: 1,
	},
	{
		Name:             "TrimQuote",
		Input:            ` "a"," b",c`,
		TrimLeadingSpace: true,
		Output:           [][]string{{"a", " b", "c"}},
	},
	{
		Name:  "BadBareQuote",
		Input: `a "word","b"`,
		Error: `bare " in non-quoted-field`, Line: 1, Column: 2,
	},
	{
		Name:  "BadTrailingQuote",
		Input: `"a word",b"`,
		Error: `bare " in non-quoted-field`, Line: 1, Column: 10,
	},
	{
		Name:  "ExtraneousQuote",
		Input: `"a "word","b"`,
		Error: `extraneous " in field`, Line: 1, Column: 3,
	},
	{
		Name:               "BadFieldCount",
		UseFieldsPerRecord: true,
		Input:              "a,b,c\nd,e",
		Error:              "wrong number of fields", Line: 2,
	},
	{
		Name:               "BadFieldCount1",
		UseFieldsPerRecord: true,
		FieldsPerRecord:    2,
		Input:              `a,b,c`,
		Error:              "wrong number of fields", Line: 1,
	},
	{
		Name:   "FieldCount",
		Input:  "a,b,c\nd,e",
		Output: [][]string{{"a", "b", "c"}, {"d", "e"}},
	},
	{
		Name:  "BadTrailingCommaEOF",
		Input: "a,b,c,",
		Error: "extra delimiter at end of line", Line: 1, Column: 5,
	},
	{
		Name:  "BadTrailingCommaEOL",
		Input: "a,b,c,\n",
		Error: "extra delimiter at end of line", Line: 1, Column: 5,
	},
	{
		Name:             "BadTrailingCommaSpaceEOF",
		TrimLeadingSpace: true,
		Input:            "a,b,c, ",
		Error:            "extra delimiter at end of line", Line: 1, Column: 5,
	},
	{
		Name:             "BadTrailingCommaSpaceEOL",
		TrimLeadingSpace: true,
		Input:            "a,b,c, \n",
		Error:            "extra delimiter at end of line", Line: 1, Column: 5,
	},
	{
		Name:             "BadTrailingCommaLine3",
		TrimLeadingSpace: true,
		Input:            "a,b,c\nd,e,f\ng,hi,",
		Error:            "extra delimiter at end of line", Line: 3, Column: 4,
	},
	{
		Name:   "NotTrailingComma3",
		Input:  "a,b,c, \n",
		Output: [][]string{{"a", "b", "c", " "}},
	},
	{
		Name:          "CommaFieldTest",
		TrailingComma: true,
		Input: `x,y,z,w
x,y,z,
x,y,,
x,,,
,,,
"x","y","z","w"
"x","y","z",""
"x","y","",""
"x","","",""
"","","",""
`,
		Output: [][]string{
			{"x", "y", "z", "w"},
			{"x", "y", "z", ""},
			{"x", "y", "", ""},
			{"x", "", "", ""},
			{"", "", "", ""},
			{"x", "y", "z", "w"},
			{"x", "y", "z", ""},
			{"x", "y", "", ""},
			{"x", "", "", ""},
			{"", "", "", ""},
		},
	},
	{
		Name:             "Issue 2366",
		TrailingComma:    true,
		TrimLeadingSpace: true,
		Input:            "a,b,\nc,d,e",
		Output: [][]string{
			{"a", "b", ""},
			{"c", "d", "e"},
		},
	},
	{
		Name:             "Issue 2366a",
		TrailingComma:    false,
		TrimLeadingSpace: true,
		Input:            "a,b,\nc,d,e",
		Error:            "extra delimiter at end of line",
	},
}

func TestRead(t *testing.T) {
	for _, tt := range readTests {
		r := NewReader(strings.NewReader(tt.Input))
		r.Comment = tt.Comment
		if tt.UseFieldsPerRecord {
			r.FieldsPerRecord = tt.FieldsPerRecord
		} else {
			r.FieldsPerRecord = -1
		}
		r.LazyQuotes = tt.LazyQuotes
		r.TrailingComma = tt.TrailingComma
		r.TrimLeadingSpace = tt.TrimLeadingSpace
		if tt.Comma != 0 {
			r.Comma = tt.Comma
		}
		out, err := r.ReadAll()
		perr, _ := err.(*ParseError)
		if tt.Error != "" {
			if err == nil || !strings.Contains(err.String(), tt.Error) {
				t.Errorf("%s: error %v, want error %q", tt.Name, err, tt.Error)
			} else if tt.Line != 0 && (tt.Line != perr.Line || tt.Column != perr.Column) {
				t.Errorf("%s: error at %d:%d expected %d:%d", tt.Name, perr.Line, perr.Column, tt.Line, tt.Column)
			}
		} else if err != nil {
			t.Errorf("%s: unexpected error %v", tt.Name, err)
		} else if !reflect.DeepEqual(out, tt.Output) {
			t.Errorf("%s: out=%q want %q", tt.Name, out, tt.Output)
		}
	}
}
