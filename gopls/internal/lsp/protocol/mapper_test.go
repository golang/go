// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
)

// This file tests Mapper's logic for converting between
// span.Point and UTF-16 columns. (The strange form attests to an
// earlier abstraction.)

// 𐐀 is U+10400 = [F0 90 90 80] in UTF-8, [D801 DC00] in UTF-16.
var funnyString = []byte("𐐀23\n𐐀45")

var toUTF16Tests = []struct {
	scenario    string
	input       []byte
	line        int    // 1-indexed count
	col         int    // 1-indexed byte position in line
	offset      int    // 0-indexed byte offset into input
	resUTF16col int    // 1-indexed UTF-16 col number
	pre         string // everything before the cursor on the line
	post        string // everything from the cursor onwards
	err         string // expected error string in call to ToUTF16Column
	issue       *bool
}{
	{
		scenario: "cursor missing content",
		input:    nil,
		offset:   -1,
		err:      "point has neither offset nor line/column",
	},
	{
		scenario: "cursor missing position",
		input:    funnyString,
		line:     -1,
		col:      -1,
		offset:   -1,
		err:      "point has neither offset nor line/column",
	},
	{
		scenario:    "zero length input; cursor at first col, first line",
		input:       []byte(""),
		line:        1,
		col:         1,
		offset:      0,
		resUTF16col: 1,
	},
	{
		scenario:    "cursor before funny character; first line",
		input:       funnyString,
		line:        1,
		col:         1,
		offset:      0,
		resUTF16col: 1,
		pre:         "",
		post:        "𐐀23",
	},
	{
		scenario:    "cursor after funny character; first line",
		input:       funnyString,
		line:        1,
		col:         5, // 4 + 1 (1-indexed)
		offset:      4, // (unused since we have line+col)
		resUTF16col: 3, // 2 + 1 (1-indexed)
		pre:         "𐐀",
		post:        "23",
	},
	{
		scenario:    "cursor after last character on first line",
		input:       funnyString,
		line:        1,
		col:         7, // 4 + 1 + 1 + 1 (1-indexed)
		offset:      6, // 4 + 1 + 1 (unused since we have line+col)
		resUTF16col: 5, // 2 + 1 + 1 + 1 (1-indexed)
		pre:         "𐐀23",
		post:        "",
	},
	{
		scenario:    "cursor before funny character; second line",
		input:       funnyString,
		line:        2,
		col:         1,
		offset:      7, // length of first line (unused since we have line+col)
		resUTF16col: 1,
		pre:         "",
		post:        "𐐀45",
	},
	{
		scenario:    "cursor after funny character; second line",
		input:       funnyString,
		line:        1,
		col:         5,  // 4 + 1 (1-indexed)
		offset:      11, // 7 (length of first line) + 4 (unused since we have line+col)
		resUTF16col: 3,  // 2 + 1 (1-indexed)
		pre:         "𐐀",
		post:        "45",
	},
	{
		scenario:    "cursor after last character on second line",
		input:       funnyString,
		line:        2,
		col:         7,  // 4 + 1 + 1 + 1 (1-indexed)
		offset:      13, // 7 (length of first line) + 4 + 1 + 1 (unused since we have line+col)
		resUTF16col: 5,  // 2 + 1 + 1 + 1 (1-indexed)
		pre:         "𐐀45",
		post:        "",
	},
	{
		scenario: "cursor beyond end of file",
		input:    funnyString,
		line:     2,
		col:      8,  // 4 + 1 + 1 + 1 + 1 (1-indexed)
		offset:   14, // 4 + 1 + 1 + 1 (unused since we have line+col)
		err:      "column is beyond end of file",
	},
}

var fromUTF16Tests = []struct {
	scenario  string
	input     []byte
	line      int    // 1-indexed line number (isn't actually used)
	utf16col  int    // 1-indexed UTF-16 col number
	resCol    int    // 1-indexed byte position in line
	resOffset int    // 0-indexed byte offset into input
	pre       string // everything before the cursor on the line
	post      string // everything from the cursor onwards
	err       string // expected error string in call to ToUTF16Column
}{
	{
		scenario:  "zero length input; cursor at first col, first line",
		input:     []byte(""),
		line:      1,
		utf16col:  1,
		resCol:    1,
		resOffset: 0,
		pre:       "",
		post:      "",
	},
	{
		scenario:  "cursor before funny character",
		input:     funnyString,
		line:      1,
		utf16col:  1,
		resCol:    1,
		resOffset: 0,
		pre:       "",
		post:      "𐐀23",
	},
	{
		scenario:  "cursor after funny character",
		input:     funnyString,
		line:      1,
		utf16col:  3,
		resCol:    5,
		resOffset: 4,
		pre:       "𐐀",
		post:      "23",
	},
	{
		scenario:  "cursor after last character on line",
		input:     funnyString,
		line:      1,
		utf16col:  5,
		resCol:    7,
		resOffset: 6,
		pre:       "𐐀23",
		post:      "",
	},
	{
		scenario:  "cursor beyond last character on line",
		input:     funnyString,
		line:      1,
		utf16col:  6,
		resCol:    7,
		resOffset: 6,
		pre:       "𐐀23",
		post:      "",
		err:       "column is beyond end of line",
	},
	{
		scenario:  "cursor before funny character; second line",
		input:     funnyString,
		line:      2,
		utf16col:  1,
		resCol:    1,
		resOffset: 7,
		pre:       "",
		post:      "𐐀45",
	},
	{
		scenario:  "cursor after funny character; second line",
		input:     funnyString,
		line:      2,
		utf16col:  3,  // 2 + 1 (1-indexed)
		resCol:    5,  // 4 + 1 (1-indexed)
		resOffset: 11, // 7 (length of first line) + 4
		pre:       "𐐀",
		post:      "45",
	},
	{
		scenario:  "cursor after last character on second line",
		input:     funnyString,
		line:      2,
		utf16col:  5,  // 2 + 1 + 1 + 1 (1-indexed)
		resCol:    7,  // 4 + 1 + 1 + 1 (1-indexed)
		resOffset: 13, // 7 (length of first line) + 4 + 1 + 1
		pre:       "𐐀45",
		post:      "",
	},
	{
		scenario:  "cursor beyond end of file",
		input:     funnyString,
		line:      2,
		utf16col:  6,  // 2 + 1 + 1 + 1 + 1(1-indexed)
		resCol:    8,  // 4 + 1 + 1 + 1 + 1 (1-indexed)
		resOffset: 14, // 7 (length of first line) + 4 + 1 + 1 + 1
		err:       "column is beyond end of file",
	},
}

func TestToUTF16(t *testing.T) {
	for _, e := range toUTF16Tests {
		t.Run(e.scenario, func(t *testing.T) {
			if e.issue != nil && !*e.issue {
				t.Skip("expected to fail")
			}
			p := span.NewPoint(e.line, e.col, e.offset)
			m := protocol.NewMapper("", e.input)
			pos, err := m.PointPosition(p)
			if err != nil {
				if err.Error() != e.err {
					t.Fatalf("expected error %v; got %v", e.err, err)
				}
				return
			}
			if e.err != "" {
				t.Fatalf("unexpected success; wanted %v", e.err)
			}
			got := int(pos.Character) + 1
			if got != e.resUTF16col {
				t.Fatalf("expected result %v; got %v", e.resUTF16col, got)
			}
			pre, post := getPrePost(e.input, p.Offset())
			if string(pre) != e.pre {
				t.Fatalf("expected #%d pre %q; got %q", p.Offset(), e.pre, pre)
			}
			if string(post) != e.post {
				t.Fatalf("expected #%d, post %q; got %q", p.Offset(), e.post, post)
			}
		})
	}
}

func TestFromUTF16(t *testing.T) {
	for _, e := range fromUTF16Tests {
		t.Run(e.scenario, func(t *testing.T) {
			m := protocol.NewMapper("", []byte(e.input))
			p, err := m.PositionPoint(protocol.Position{
				Line:      uint32(e.line - 1),
				Character: uint32(e.utf16col - 1),
			})
			if err != nil {
				if err.Error() != e.err {
					t.Fatalf("expected error %v; got %v", e.err, err)
				}
				return
			}
			if e.err != "" {
				t.Fatalf("unexpected success; wanted %v", e.err)
			}
			if p.Column() != e.resCol {
				t.Fatalf("expected resulting col %v; got %v", e.resCol, p.Column())
			}
			if p.Offset() != e.resOffset {
				t.Fatalf("expected resulting offset %v; got %v", e.resOffset, p.Offset())
			}
			pre, post := getPrePost(e.input, p.Offset())
			if string(pre) != e.pre {
				t.Fatalf("expected #%d pre %q; got %q", p.Offset(), e.pre, pre)
			}
			if string(post) != e.post {
				t.Fatalf("expected #%d post %q; got %q", p.Offset(), e.post, post)
			}
		})
	}
}

func getPrePost(content []byte, offset int) (string, string) {
	pre, post := string(content)[:offset], string(content)[offset:]
	if i := strings.LastIndex(pre, "\n"); i >= 0 {
		pre = pre[i+1:]
	}
	if i := strings.IndexRune(post, '\n'); i >= 0 {
		post = post[:i]
	}
	return pre, post
}

// -- these are the historical lsppos tests --

type testCase struct {
	content            string      // input text
	substrOrOffset     interface{} // explicit integer offset, or a substring
	wantLine, wantChar int         // expected LSP position information
}

// offset returns the test case byte offset
func (c testCase) offset() int {
	switch x := c.substrOrOffset.(type) {
	case int:
		return x
	case string:
		i := strings.Index(c.content, x)
		if i < 0 {
			panic(fmt.Sprintf("%q does not contain substring %q", c.content, x))
		}
		return i
	}
	panic("substrOrIndex must be an integer or string")
}

var tests = []testCase{
	{"a𐐀b", "a", 0, 0},
	{"a𐐀b", "𐐀", 0, 1},
	{"a𐐀b", "b", 0, 3},
	{"a𐐀b\n", "\n", 0, 4},
	{"a𐐀b\r\n", "\n", 0, 4}, // \r|\n is not a valid position, so we move back to the end of the first line.
	{"a𐐀b\r\nx", "x", 1, 0},
	{"a𐐀b\r\nx\ny", "y", 2, 0},

	// Testing EOL and EOF positions
	{"", 0, 0, 0}, // 0th position of an empty buffer is (0, 0)
	{"abc", "c", 0, 2},
	{"abc", 3, 0, 3},
	{"abc\n", "\n", 0, 3},
	{"abc\n", 4, 1, 0}, // position after a newline is on the next line
}

func TestLineChar(t *testing.T) {
	for _, test := range tests {
		m := protocol.NewMapper("", []byte(test.content))
		offset := test.offset()
		posn, _ := m.OffsetPosition(offset)
		gotLine, gotChar := int(posn.Line), int(posn.Character)
		if gotLine != test.wantLine || gotChar != test.wantChar {
			t.Errorf("LineChar(%d) = (%d,%d), want (%d,%d)", offset, gotLine, gotChar, test.wantLine, test.wantChar)
		}
	}
}

func TestInvalidOffset(t *testing.T) {
	content := []byte("a𐐀b\r\nx\ny")
	m := protocol.NewMapper("", content)
	for _, offset := range []int{-1, 100} {
		posn, err := m.OffsetPosition(offset)
		if err == nil {
			t.Errorf("OffsetPosition(%d) = %s, want error", offset, posn)
		}
	}
}

func TestPosition(t *testing.T) {
	for _, test := range tests {
		m := protocol.NewMapper("", []byte(test.content))
		offset := test.offset()
		got, err := m.OffsetPosition(offset)
		if err != nil {
			t.Errorf("OffsetPosition(%d) failed: %v", offset, err)
			continue
		}
		want := protocol.Position{Line: uint32(test.wantLine), Character: uint32(test.wantChar)}
		if got != want {
			t.Errorf("Position(%d) = %v, want %v", offset, got, want)
		}
	}
}

func TestRange(t *testing.T) {
	for _, test := range tests {
		m := protocol.NewMapper("", []byte(test.content))
		offset := test.offset()
		got, err := m.OffsetRange(0, offset)
		if err != nil {
			t.Fatal(err)
		}
		want := protocol.Range{
			End: protocol.Position{Line: uint32(test.wantLine), Character: uint32(test.wantChar)},
		}
		if got != want {
			t.Errorf("Range(%d) = %v, want %v", offset, got, want)
		}
	}
}

func TestBytesOffset(t *testing.T) {
	tests := []struct {
		text string
		pos  protocol.Position
		want int
	}{
		// U+10400 encodes as [F0 90 90 80] in UTF-8 and [D801 DC00] in UTF-16.
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 0}, want: 0},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 1}, want: 1},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 2}, want: 1},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 3}, want: 5},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 4}, want: 6},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 5}, want: -1},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 0, Character: 3}, want: 3},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 0, Character: 4}, want: -1},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 0}, want: 4},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 3}, want: 7},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 4}, want: -1},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 1}, want: -1},
		{text: "aaa\nbbb\n\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
	}

	for i, test := range tests {
		fname := fmt.Sprintf("test %d", i)
		uri := span.URIFromPath(fname)
		mapper := protocol.NewMapper(uri, []byte(test.text))
		got, err := mapper.PositionPoint(test.pos)
		if err != nil && test.want != -1 {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
		if err == nil && got.Offset() != test.want {
			t.Errorf("want %d for %q(Line:%d,Character:%d), but got %d", test.want, test.text, int(test.pos.Line), int(test.pos.Character), got.Offset())
		}
	}
}
