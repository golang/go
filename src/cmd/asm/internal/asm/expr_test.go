// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"cmd/asm/internal/lex"
	"strings"
	"testing"
	"text/scanner"
)

type exprTest struct {
	input  string
	output int64
	atEOF  bool
}

var exprTests = []exprTest{
	// Simple
	{"0", 0, true},
	{"3", 3, true},
	{"070", 8 * 7, true},
	{"0x0f", 15, true},
	{"0xFF", 255, true},
	{"9223372036854775807", 9223372036854775807, true}, // max int64
	// Unary
	{"-0", 0, true},
	{"~0", -1, true},
	{"~0*0", 0, true},
	{"+3", 3, true},
	{"-3", -3, true},
	{"-9223372036854775808", -9223372036854775808, true}, // min int64
	// Binary
	{"3+4", 3 + 4, true},
	{"3-4", 3 - 4, true},
	{"2|5", 2 | 5, true},
	{"3^4", 3 ^ 4, true},
	{"3*4", 3 * 4, true},
	{"14/4", 14 / 4, true},
	{"3<<4", 3 << 4, true},
	{"48>>3", 48 >> 3, true},
	{"3&9", 3 & 9, true},
	// General
	{"3*2+3", 3*2 + 3, true},
	{"3+2*3", 3 + 2*3, true},
	{"3*(2+3)", 3 * (2 + 3), true},
	{"3*-(2+3)", 3 * -(2 + 3), true},
	{"3<<2+4", 3<<2 + 4, true},
	{"3<<2+4", 3<<2 + 4, true},
	{"3<<(2+4)", 3 << (2 + 4), true},
	// Junk at EOF.
	{"3 x", 3, false},
	// Big number
	{"4611686018427387904", 4611686018427387904, true},
}

func TestExpr(t *testing.T) {
	p := NewParser(nil, nil, nil) // Expression evaluation uses none of these fields of the parser.
	for i, test := range exprTests {
		p.start(lex.Tokenize(test.input))
		result := int64(p.expr())
		if result != test.output {
			t.Errorf("%d: %q evaluated to %d; expected %d", i, test.input, result, test.output)
		}
		tok := p.next()
		if test.atEOF && tok.ScanToken != scanner.EOF {
			t.Errorf("%d: %q: at EOF got %s", i, test.input, tok)
		} else if !test.atEOF && tok.ScanToken == scanner.EOF {
			t.Errorf("%d: %q: expected not EOF but at EOF", i, test.input)
		}
	}
}

type badExprTest struct {
	input string
	error string // Empty means no error.
}

var badExprTests = []badExprTest{
	{"0/0", "division by zero"},
	{"3/0", "division by zero"},
	{"(1<<63)/0", "divide of value with high bit set"},
	{"3%0", "modulo by zero"},
	{"(1<<63)%0", "modulo of value with high bit set"},
	{"3<<-4", "negative left shift count"},
	{"3<<(1<<63)", "negative left shift count"},
	{"3>>-4", "negative right shift count"},
	{"3>>(1<<63)", "negative right shift count"},
	{"(1<<63)>>2", "right shift of value with high bit set"},
	{"(1<<62)>>2", ""},
	{`'\x80'`, "illegal UTF-8 encoding for character constant"},
	{"(23*4", "missing closing paren"},
	{")23*4", "unexpected ) evaluating expression"},
	{"18446744073709551616", "value out of range"},
}

func TestBadExpr(t *testing.T) {
	for i, test := range badExprTests {
		err := runBadTest(i, test, t)
		if err == nil {
			if test.error != "" {
				t.Errorf("#%d: %q: expected error %q; got none", i, test.input, test.error)
			}
			continue
		}
		if !strings.Contains(err.Error(), test.error) {
			t.Errorf("#%d: expected error %q; got %q", i, test.error, err)
			continue
		}
	}
}

func runBadTest(i int, test badExprTest, t *testing.T) (err error) {
	p := NewParser(nil, nil, nil) // Expression evaluation uses none of these fields of the parser.
	p.start(lex.Tokenize(test.input))
	return tryParse(t, func() {
		p.expr()
	})
}
