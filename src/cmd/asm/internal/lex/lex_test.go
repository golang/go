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
	{
		"taken #ifdef",
		lines(
			"#define A",
			"#ifdef A",
			"#define B 1234",
			"#endif",
			"B",
		),
		"1234.\n",
	},
	{
		"not taken #ifdef",
		lines(
			"#ifdef A",
			"#define B 1234",
			"#endif",
			"B",
		),
		"B.\n",
	},
	{
		"taken #ifdef with else",
		lines(
			"#define A",
			"#ifdef A",
			"#define B 1234",
			"#else",
			"#define B 5678",
			"#endif",
			"B",
		),
		"1234.\n",
	},
	{
		"not taken #ifdef with else",
		lines(
			"#ifdef A",
			"#define B 1234",
			"#else",
			"#define B 5678",
			"#endif",
			"B",
		),
		"5678.\n",
	},
	{
		"nested taken/taken #ifdef",
		lines(
			"#define A",
			"#define B",
			"#ifdef A",
			"#ifdef B",
			"#define C 1234",
			"#else",
			"#define C 5678",
			"#endif",
			"#endif",
			"C",
		),
		"1234.\n",
	},
	{
		"nested taken/not-taken #ifdef",
		lines(
			"#define A",
			"#ifdef A",
			"#ifdef B",
			"#define C 1234",
			"#else",
			"#define C 5678",
			"#endif",
			"#endif",
			"C",
		),
		"5678.\n",
	},
	{
		"nested not-taken/would-be-taken #ifdef",
		lines(
			"#define B",
			"#ifdef A",
			"#ifdef B",
			"#define C 1234",
			"#else",
			"#define C 5678",
			"#endif",
			"#endif",
			"C",
		),
		"C.\n",
	},
	{
		"nested not-taken/not-taken #ifdef",
		lines(
			"#ifdef A",
			"#ifdef B",
			"#define C 1234",
			"#else",
			"#define C 5678",
			"#endif",
			"#endif",
			"C",
		),
		"C.\n",
	},
	{
		"nested #define",
		lines(
			"#define A #define B THIS",
			"A",
			"B",
		),
		"THIS.\n",
	},
	{
		"nested #define with args",
		lines(
			"#define A #define B(x) x",
			"A",
			"B(THIS)",
		),
		"THIS.\n",
	},
	/* This one fails. See comment in Slice.Col.
	{
		"nested #define with args",
		lines(
			"#define A #define B (x) x",
			"A",
			"B(THIS)",
		),
		"x.\n",
	},
	*/
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

type badLexTest struct {
	input string
	error string
}

var badLexTests = []badLexTest{
	{
		"3 #define foo bar\n",
		"'#' must be first item on line",
	},
	{
		"#ifdef foo\nhello",
		"unclosed #ifdef or #ifndef",
	},
	{
		"#ifndef foo\nhello",
		"unclosed #ifdef or #ifndef",
	},
	{
		"#ifdef foo\nhello\n#else\nbye",
		"unclosed #ifdef or #ifndef",
	},
	{
		"#define A() A()\nA()",
		"recursive macro invocation",
	},
	{
		"#define A a\n#define A a\n",
		"redefinition of macro",
	},
	{
		"#define A a",
		"no newline after macro definition",
	},
}

func TestBadLex(t *testing.T) {
	for _, test := range badLexTests {
		input := NewInput(test.error)
		input.Push(NewTokenizer(test.error, strings.NewReader(test.input), nil))
		err := firstError(input)
		if err == nil {
			t.Errorf("%s: got no error", test.error)
			continue
		}
		if !strings.Contains(err.Error(), test.error) {
			t.Errorf("got error %q expected %q", err.Error(), test.error)
		}
	}
}

// firstError returns the first error value triggered by the input.
func firstError(input *Input) (err error) {
	panicOnError = true
	defer func() {
		panicOnError = false
		switch e := recover(); e := e.(type) {
		case nil:
		case error:
			err = e
		default:
			panic(e)
		}
	}()

	for {
		tok := input.Next()
		if tok == scanner.EOF {
			return
		}
	}
}
