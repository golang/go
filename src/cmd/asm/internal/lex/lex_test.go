// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lex

import (
	"bytes"
	"strings"
	"testing"
	"text/scanner"
)

type lexTest struct {
	name   string
	input  string
	output string
}

var lexTests = []lexTest{
	{
		"empty",
		"",
		"",
	},
	{
		"simple",
		"1 (a)",
		"1.(.a.)",
	},
	{
		"simple define",
		lines(
			"#define A 1234",
			"A",
		),
		"1234.\n",
	},
	{
		"define without value",
		"#define A",
		"",
	},
	{
		"macro without arguments",
		"#define A() 1234\n" + "A()\n",
		"1234.\n",
	},
	{
		"macro with just parens as body",
		"#define A () \n" + "A\n",
		"(.).\n",
	},
	{
		"macro with parens but no arguments",
		"#define A (x) \n" + "A\n",
		"(.x.).\n",
	},
	{
		"macro with arguments",
		"#define A(x, y, z) x+z+y\n" + "A(1, 2, 3)\n",
		"1.+.3.+.2.\n",
	},
	{
		"argumented macro invoked without arguments",
		lines(
			"#define X() foo ",
			"X()",
			"X",
		),
		"foo.\n.X.\n",
	},
	{
		"multiline macro without arguments",
		lines(
			"#define A 1\\",
			"\t2\\",
			"\t3",
			"before",
			"A",
			"after",
		),
		"before.\n.1.\n.2.\n.3.\n.after.\n",
	},
	{
		"multiline macro with arguments",
		lines(
			"#define A(a, b, c) a\\",
			"\tb\\",
			"\tc",
			"before",
			"A(1, 2, 3)",
			"after",
		),
		"before.\n.1.\n.2.\n.3.\n.after.\n",
	},
	{
		"LOAD macro",
		lines(
			"#define LOAD(off, reg) \\",
			"\tMOVBLZX	(off*4)(R12),	reg \\",
			"\tADDB	reg,		DX",
			"",
			"LOAD(8, AX)",
		),
		"\n.\n.MOVBLZX.(.8.*.4.).(.R12.).,.AX.\n.ADDB.AX.,.DX.\n",
	},
	{
		"nested multiline macro",
		lines(
			"#define KEYROUND(xmm, load, off, r1, r2, index) \\",
			"\tMOVBLZX	(BP)(DX*4),	R8 \\",
			"\tload((off+1), r2) \\",
			"\tMOVB	R8,		(off*4)(R12) \\",
			"\tPINSRW	$index, (BP)(R8*4), xmm",
			"#define LOAD(off, reg) \\",
			"\tMOVBLZX	(off*4)(R12),	reg \\",
			"\tADDB	reg,		DX",
			"KEYROUND(X0, LOAD, 8, AX, BX, 0)",
		),
		"\n.MOVBLZX.(.BP.).(.DX.*.4.).,.R8.\n.\n.MOVBLZX.(.(.8.+.1.).*.4.).(.R12.).,.BX.\n.ADDB.BX.,.DX.\n.MOVB.R8.,.(.8.*.4.).(.R12.).\n.PINSRW.$.0.,.(.BP.).(.R8.*.4.).,.X0.\n",
	},
}

func TestLex(t *testing.T) {
	for _, test := range lexTests {
		input := NewInput(test.name)
		input.Push(NewTokenizer(test.name, strings.NewReader(test.input), nil))
		result := drain(input)
		if result != test.output {
			t.Errorf("%s: got %q expected %q", test.name, result, test.output)
		}
	}
}

// lines joins the arguments together as complete lines.
func lines(a ...string) string {
	return strings.Join(a, "\n") + "\n"
}

// drain returns a single string representing the processed input tokens.
func drain(input *Input) string {
	var buf bytes.Buffer
	for {
		tok := input.Next()
		if tok == scanner.EOF {
			return buf.String()
		}
		if buf.Len() > 0 {
			buf.WriteByte('.')
		}
		buf.WriteString(input.Text())
	}
}
