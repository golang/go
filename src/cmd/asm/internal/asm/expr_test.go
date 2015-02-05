// +build ignore

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"cmd/asm/internal/lex"
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
