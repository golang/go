// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span_test

import (
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/span"
)

// The funny character below is 4 bytes long in UTF-8; two UTF-16 code points
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
		err:      "ToUTF16Column: point is missing position",
	},
	{
		scenario: "cursor missing position",
		input:    funnyString,
		line:     -1,
		col:      -1,
		err:      "ToUTF16Column: point is missing position",
	},
	{
		scenario: "cursor missing offset",
		input:    funnyString,
		line:     1,
		col:      1,
		offset:   -1,
		err:      "ToUTF16Column: point is missing offset",
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
		offset:      4,
		resUTF16col: 3, // 2 + 1 (1-indexed)
		pre:         "𐐀",
		post:        "23",
	},
	{
		scenario:    "cursor after last character on first line",
		input:       funnyString,
		line:        1,
		col:         7, // 4 + 1 + 1 + 1 (1-indexed)
		offset:      6, // 4 + 1 + 1
		resUTF16col: 5, // 2 + 1 + 1 + 1 (1-indexed)
		pre:         "𐐀23",
		post:        "",
	},
	{
		scenario:    "cursor before funny character; second line",
		input:       funnyString,
		line:        2,
		col:         1,
		offset:      7, // length of first line
		resUTF16col: 1,
		pre:         "",
		post:        "𐐀45",
	},
	{
		scenario:    "cursor after funny character; second line",
		input:       funnyString,
		line:        1,
		col:         5,  // 4 + 1 (1-indexed)
		offset:      11, // 7 (length of first line) + 4
		resUTF16col: 3,  // 2 + 1 (1-indexed)
		pre:         "𐐀",
		post:        "45",
	},
	{
		scenario:    "cursor after last character on second line",
		input:       funnyString,
		line:        2,
		col:         7,  // 4 + 1 + 1 + 1 (1-indexed)
		offset:      13, // 7 (length of first line) + 4 + 1 + 1
		resUTF16col: 5,  // 2 + 1 + 1 + 1 (1-indexed)
		pre:         "𐐀45",
		post:        "",
	},
	{
		scenario: "cursor beyond end of file",
		input:    funnyString,
		line:     2,
		col:      8,  // 4 + 1 + 1 + 1 + 1 (1-indexed)
		offset:   14, // 4 + 1 + 1 + 1
		err:      "ToUTF16Column: offsets 7-14 outside file contents (13)",
	},
}

var fromUTF16Tests = []struct {
	scenario  string
	input     []byte
	line      int    // 1-indexed line number (isn't actually used)
	offset    int    // 0-indexed byte offset to beginning of line
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
		offset:    0,
		utf16col:  1,
		resCol:    1,
		resOffset: 0,
		pre:       "",
		post:      "",
	},
	{
		scenario: "missing offset",
		input:    funnyString,
		line:     1,
		offset:   -1,
		err:      "FromUTF16Column: point is missing offset",
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
		offset:    0,
		utf16col:  6,
		resCol:    7,
		resOffset: 6,
		pre:       "𐐀23",
		post:      "",
	},
	{
		scenario:  "cursor before funny character; second line",
		input:     funnyString,
		line:      2,
		offset:    7, // length of first line
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
		offset:    7,  // length of first line
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
		offset:    7,  // length of first line
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
		offset:    7,
		utf16col:  6,  // 2 + 1 + 1 + 1 + 1(1-indexed)
		resCol:    8,  // 4 + 1 + 1 + 1 + 1 (1-indexed)
		resOffset: 14, // 7 (length of first line) + 4 + 1 + 1 + 1
		err:       "FromUTF16Column: chr goes beyond the content",
	},
	{
		scenario: "offset beyond end of file",
		input:    funnyString,
		line:     2,
		offset:   14,
		utf16col: 2,
		err:      "FromUTF16Column: offset (14) greater than length of content (13)",
	},
}

func TestToUTF16(t *testing.T) {
	for _, e := range toUTF16Tests {
		t.Run(e.scenario, func(t *testing.T) {
			if e.issue != nil && !*e.issue {
				t.Skip("expected to fail")
			}
			p := span.NewPoint(e.line, e.col, e.offset)
			got, err := span.ToUTF16Column(p, e.input)
			if err != nil {
				if err.Error() != e.err {
					t.Fatalf("expected error %v; got %v", e.err, err)
				}
				return
			}
			if e.err != "" {
				t.Fatalf("unexpected success; wanted %v", e.err)
			}
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
			p := span.NewPoint(e.line, 1, e.offset)
			p, err := span.FromUTF16Column(p, e.utf16col, []byte(e.input))
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
