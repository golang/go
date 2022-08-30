// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsppos_test

import (
	"fmt"
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/lsppos"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

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
		m := NewMapper([]byte(test.content))
		offset := test.offset()
		gotLine, gotChar := m.LineColUTF16(offset)
		if gotLine != test.wantLine || gotChar != test.wantChar {
			t.Errorf("LineChar(%d) = (%d,%d), want (%d,%d)", offset, gotLine, gotChar, test.wantLine, test.wantChar)
		}
	}
}

func TestInvalidOffset(t *testing.T) {
	content := []byte("a𐐀b\r\nx\ny")
	m := NewMapper(content)
	for _, offset := range []int{-1, 100} {
		gotLine, gotChar := m.LineColUTF16(offset)
		if gotLine != -1 {
			t.Errorf("LineChar(%d) = (%d,%d), want (-1,-1)", offset, gotLine, gotChar)
		}
	}
}

func TestPosition(t *testing.T) {
	for _, test := range tests {
		m := NewMapper([]byte(test.content))
		offset := test.offset()
		got, ok := m.Position(offset)
		if !ok {
			t.Error("invalid position for", test.substrOrOffset)
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
		m := NewMapper([]byte(test.content))
		offset := test.offset()
		got, err := m.Range(0, offset)
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
