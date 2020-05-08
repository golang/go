// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"
)

// errh is a default error handler for basic tests.
func errh(line, col uint, msg string) {
	panic(fmt.Sprintf("%d:%d: %s", line, col, msg))
}

// Don't bother with other tests if TestSmoke doesn't pass.
func TestSmoke(t *testing.T) {
	const src = "if (+foo\t+=..123/***/0.9_0e-0i'a'`raw`\"string\"..f;//$"
	tokens := []token{_If, _Lparen, _Operator, _Name, _AssignOp, _Dot, _Literal, _Literal, _Literal, _Literal, _Literal, _Dot, _Dot, _Name, _Semi, _EOF}

	var got scanner
	got.init(strings.NewReader(src), errh, 0)
	for _, want := range tokens {
		got.next()
		if got.tok != want {
			t.Errorf("%d:%d: got %s; want %s", got.line, got.col, got.tok, want)
			continue
		}
	}
}

// Once TestSmoke passes, run TestTokens next.
func TestTokens(t *testing.T) {
	var got scanner
	for _, want := range sampleTokens {
		got.init(strings.NewReader(want.src), func(line, col uint, msg string) {
			t.Errorf("%s:%d:%d: %s", want.src, line, col, msg)
		}, 0)
		got.next()
		if got.tok != want.tok {
			t.Errorf("%s: got %s; want %s", want.src, got.tok, want.tok)
			continue
		}
		if (got.tok == _Name || got.tok == _Literal) && got.lit != want.src {
			t.Errorf("%s: got %q; want %q", want.src, got.lit, want.src)
		}
	}
}

func TestScanner(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	filename := *src_ // can be changed via -src flag
	src, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	var s scanner
	s.init(src, errh, 0)
	for {
		s.next()
		if s.tok == _EOF {
			break
		}
		if !testing.Verbose() {
			continue
		}
		switch s.tok {
		case _Name, _Literal:
			fmt.Printf("%s:%d:%d: %s => %s\n", filename, s.line, s.col, s.tok, s.lit)
		case _Operator:
			fmt.Printf("%s:%d:%d: %s => %s (prec = %d)\n", filename, s.line, s.col, s.tok, s.op, s.prec)
		default:
			fmt.Printf("%s:%d:%d: %s\n", filename, s.line, s.col, s.tok)
		}
	}
}

func TestEmbeddedTokens(t *testing.T) {
	// make source
	var buf bytes.Buffer
	for i, s := range sampleTokens {
		buf.WriteString("\t\t\t\t"[:i&3])                            // leading indentation
		buf.WriteString(s.src)                                       // token
		buf.WriteString("        "[:i&7])                            // trailing spaces
		buf.WriteString(fmt.Sprintf("/*line foo:%d */ // bar\n", i)) // comments + newline (don't crash w/o directive handler)
	}

	// scan source
	var got scanner
	var src string
	got.init(&buf, func(line, col uint, msg string) {
		t.Fatalf("%s:%d:%d: %s", src, line, col, msg)
	}, 0)
	got.next()
	for i, want := range sampleTokens {
		src = want.src
		nlsemi := false

		if got.line-linebase != uint(i) {
			t.Errorf("%s: got line %d; want %d", src, got.line-linebase, i)
		}

		if got.tok != want.tok {
			t.Errorf("%s: got tok %s; want %s", src, got.tok, want.tok)
			continue
		}

		switch want.tok {
		case _Semi:
			if got.lit != "semicolon" {
				t.Errorf("%s: got %s; want semicolon", src, got.lit)
			}

		case _Name, _Literal:
			if got.lit != want.src {
				t.Errorf("%s: got lit %q; want %q", src, got.lit, want.src)
				continue
			}
			nlsemi = true

		case _Operator, _AssignOp, _IncOp:
			if got.op != want.op {
				t.Errorf("%s: got op %s; want %s", src, got.op, want.op)
				continue
			}
			if got.prec != want.prec {
				t.Errorf("%s: got prec %d; want %d", src, got.prec, want.prec)
				continue
			}
			nlsemi = want.tok == _IncOp

		case _Rparen, _Rbrack, _Rbrace, _Break, _Continue, _Fallthrough, _Return:
			nlsemi = true
		}

		if nlsemi {
			got.next()
			if got.tok != _Semi {
				t.Errorf("%s: got tok %s; want ;", src, got.tok)
				continue
			}
			if got.lit != "newline" {
				t.Errorf("%s: got %s; want newline", src, got.lit)
			}
		}

		got.next()
	}

	if got.tok != _EOF {
		t.Errorf("got %q; want _EOF", got.tok)
	}
}

var sampleTokens = [...]struct {
	tok  token
	src  string
	op   Operator
	prec int
}{
	// name samples
	{_Name, "x", 0, 0},
	{_Name, "X123", 0, 0},
	{_Name, "foo", 0, 0},
	{_Name, "Foo123", 0, 0},
	{_Name, "foo_bar", 0, 0},
	{_Name, "_", 0, 0},
	{_Name, "_foobar", 0, 0},
	{_Name, "a۰۱۸", 0, 0},
	{_Name, "foo६४", 0, 0},
	{_Name, "bar９８７６", 0, 0},
	{_Name, "ŝ", 0, 0},
	{_Name, "ŝfoo", 0, 0},

	// literal samples
	{_Literal, "0", 0, 0},
	{_Literal, "1", 0, 0},
	{_Literal, "12345", 0, 0},
	{_Literal, "123456789012345678890123456789012345678890", 0, 0},
	{_Literal, "01234567", 0, 0},
	{_Literal, "0_1_234_567", 0, 0},
	{_Literal, "0X0", 0, 0},
	{_Literal, "0xcafebabe", 0, 0},
	{_Literal, "0x_cafe_babe", 0, 0},
	{_Literal, "0O0", 0, 0},
	{_Literal, "0o000", 0, 0},
	{_Literal, "0o_000", 0, 0},
	{_Literal, "0B1", 0, 0},
	{_Literal, "0b01100110", 0, 0},
	{_Literal, "0b_0110_0110", 0, 0},
	{_Literal, "0.", 0, 0},
	{_Literal, "0.e0", 0, 0},
	{_Literal, "0.e-1", 0, 0},
	{_Literal, "0.e+123", 0, 0},
	{_Literal, ".0", 0, 0},
	{_Literal, ".0E00", 0, 0},
	{_Literal, ".0E-0123", 0, 0},
	{_Literal, ".0E+12345678901234567890", 0, 0},
	{_Literal, ".45e1", 0, 0},
	{_Literal, "3.14159265", 0, 0},
	{_Literal, "1e0", 0, 0},
	{_Literal, "1e+100", 0, 0},
	{_Literal, "1e-100", 0, 0},
	{_Literal, "2.71828e-1000", 0, 0},
	{_Literal, "0i", 0, 0},
	{_Literal, "1i", 0, 0},
	{_Literal, "012345678901234567889i", 0, 0},
	{_Literal, "123456789012345678890i", 0, 0},
	{_Literal, "0.i", 0, 0},
	{_Literal, ".0i", 0, 0},
	{_Literal, "3.14159265i", 0, 0},
	{_Literal, "1e0i", 0, 0},
	{_Literal, "1e+100i", 0, 0},
	{_Literal, "1e-100i", 0, 0},
	{_Literal, "2.71828e-1000i", 0, 0},
	{_Literal, "'a'", 0, 0},
	{_Literal, "'\\000'", 0, 0},
	{_Literal, "'\\xFF'", 0, 0},
	{_Literal, "'\\uff16'", 0, 0},
	{_Literal, "'\\U0000ff16'", 0, 0},
	{_Literal, "`foobar`", 0, 0},
	{_Literal, "`foo\tbar`", 0, 0},
	{_Literal, "`\r`", 0, 0},

	// operators
	{_Operator, "||", OrOr, precOrOr},

	{_Operator, "&&", AndAnd, precAndAnd},

	{_Operator, "==", Eql, precCmp},
	{_Operator, "!=", Neq, precCmp},
	{_Operator, "<", Lss, precCmp},
	{_Operator, "<=", Leq, precCmp},
	{_Operator, ">", Gtr, precCmp},
	{_Operator, ">=", Geq, precCmp},

	{_Operator, "+", Add, precAdd},
	{_Operator, "-", Sub, precAdd},
	{_Operator, "|", Or, precAdd},
	{_Operator, "^", Xor, precAdd},

	{_Star, "*", Mul, precMul},
	{_Operator, "/", Div, precMul},
	{_Operator, "%", Rem, precMul},
	{_Operator, "&", And, precMul},
	{_Operator, "&^", AndNot, precMul},
	{_Operator, "<<", Shl, precMul},
	{_Operator, ">>", Shr, precMul},

	// assignment operations
	{_AssignOp, "+=", Add, precAdd},
	{_AssignOp, "-=", Sub, precAdd},
	{_AssignOp, "|=", Or, precAdd},
	{_AssignOp, "^=", Xor, precAdd},

	{_AssignOp, "*=", Mul, precMul},
	{_AssignOp, "/=", Div, precMul},
	{_AssignOp, "%=", Rem, precMul},
	{_AssignOp, "&=", And, precMul},
	{_AssignOp, "&^=", AndNot, precMul},
	{_AssignOp, "<<=", Shl, precMul},
	{_AssignOp, ">>=", Shr, precMul},

	// other operations
	{_IncOp, "++", Add, precAdd},
	{_IncOp, "--", Sub, precAdd},
	{_Assign, "=", 0, 0},
	{_Define, ":=", 0, 0},
	{_Arrow, "<-", 0, 0},

	// delimiters
	{_Lparen, "(", 0, 0},
	{_Lbrack, "[", 0, 0},
	{_Lbrace, "{", 0, 0},
	{_Rparen, ")", 0, 0},
	{_Rbrack, "]", 0, 0},
	{_Rbrace, "}", 0, 0},
	{_Comma, ",", 0, 0},
	{_Semi, ";", 0, 0},
	{_Colon, ":", 0, 0},
	{_Dot, ".", 0, 0},
	{_DotDotDot, "...", 0, 0},

	// keywords
	{_Break, "break", 0, 0},
	{_Case, "case", 0, 0},
	{_Chan, "chan", 0, 0},
	{_Const, "const", 0, 0},
	{_Continue, "continue", 0, 0},
	{_Default, "default", 0, 0},
	{_Defer, "defer", 0, 0},
	{_Else, "else", 0, 0},
	{_Fallthrough, "fallthrough", 0, 0},
	{_For, "for", 0, 0},
	{_Func, "func", 0, 0},
	{_Go, "go", 0, 0},
	{_Goto, "goto", 0, 0},
	{_If, "if", 0, 0},
	{_Import, "import", 0, 0},
	{_Interface, "interface", 0, 0},
	{_Map, "map", 0, 0},
	{_Package, "package", 0, 0},
	{_Range, "range", 0, 0},
	{_Return, "return", 0, 0},
	{_Select, "select", 0, 0},
	{_Struct, "struct", 0, 0},
	{_Switch, "switch", 0, 0},
	{_Type, "type", 0, 0},
	{_Var, "var", 0, 0},
}

func TestComments(t *testing.T) {
	type comment struct {
		line, col uint // 0-based
		text      string
	}

	for _, test := range []struct {
		src  string
		want comment
	}{
		// no comments
		{"no comment here", comment{0, 0, ""}},
		{" /", comment{0, 0, ""}},
		{"\n /*/", comment{0, 0, ""}},

		//-style comments
		{"// line comment\n", comment{0, 0, "// line comment"}},
		{"package p // line comment\n", comment{0, 10, "// line comment"}},
		{"//\n//\n\t// want this one\r\n", comment{2, 1, "// want this one\r"}},
		{"\n\n//\n", comment{2, 0, "//"}},
		{"//", comment{0, 0, "//"}},

		/*-style comments */
		{"123/* regular comment */", comment{0, 3, "/* regular comment */"}},
		{"package p /* regular comment", comment{0, 0, ""}},
		{"\n\n\n/*\n*//* want this one */", comment{4, 2, "/* want this one */"}},
		{"\n\n/**/", comment{2, 0, "/**/"}},
		{"/*", comment{0, 0, ""}},
	} {
		var s scanner
		var got comment
		s.init(strings.NewReader(test.src), func(line, col uint, msg string) {
			if msg[0] != '/' {
				// error
				if msg != "comment not terminated" {
					t.Errorf("%q: %s", test.src, msg)
				}
				return
			}
			got = comment{line - linebase, col - colbase, msg} // keep last one
		}, comments)

		for {
			s.next()
			if s.tok == _EOF {
				break
			}
		}

		want := test.want
		if got.line != want.line || got.col != want.col {
			t.Errorf("%q: got position %d:%d; want %d:%d", test.src, got.line, got.col, want.line, want.col)
		}
		if got.text != want.text {
			t.Errorf("%q: got %q; want %q", test.src, got.text, want.text)
		}
	}
}

func TestNumbers(t *testing.T) {
	for _, test := range []struct {
		kind             LitKind
		src, tokens, err string
	}{
		// binaries
		{IntLit, "0b0", "0b0", ""},
		{IntLit, "0b1010", "0b1010", ""},
		{IntLit, "0B1110", "0B1110", ""},

		{IntLit, "0b", "0b", "binary literal has no digits"},
		{IntLit, "0b0190", "0b0190", "invalid digit '9' in binary literal"},
		{IntLit, "0b01a0", "0b01 a0", ""}, // only accept 0-9

		{FloatLit, "0b.", "0b.", "invalid radix point in binary literal"},
		{FloatLit, "0b.1", "0b.1", "invalid radix point in binary literal"},
		{FloatLit, "0b1.0", "0b1.0", "invalid radix point in binary literal"},
		{FloatLit, "0b1e10", "0b1e10", "'e' exponent requires decimal mantissa"},
		{FloatLit, "0b1P-1", "0b1P-1", "'P' exponent requires hexadecimal mantissa"},

		{ImagLit, "0b10i", "0b10i", ""},
		{ImagLit, "0b10.0i", "0b10.0i", "invalid radix point in binary literal"},

		// octals
		{IntLit, "0o0", "0o0", ""},
		{IntLit, "0o1234", "0o1234", ""},
		{IntLit, "0O1234", "0O1234", ""},

		{IntLit, "0o", "0o", "octal literal has no digits"},
		{IntLit, "0o8123", "0o8123", "invalid digit '8' in octal literal"},
		{IntLit, "0o1293", "0o1293", "invalid digit '9' in octal literal"},
		{IntLit, "0o12a3", "0o12 a3", ""}, // only accept 0-9

		{FloatLit, "0o.", "0o.", "invalid radix point in octal literal"},
		{FloatLit, "0o.2", "0o.2", "invalid radix point in octal literal"},
		{FloatLit, "0o1.2", "0o1.2", "invalid radix point in octal literal"},
		{FloatLit, "0o1E+2", "0o1E+2", "'E' exponent requires decimal mantissa"},
		{FloatLit, "0o1p10", "0o1p10", "'p' exponent requires hexadecimal mantissa"},

		{ImagLit, "0o10i", "0o10i", ""},
		{ImagLit, "0o10e0i", "0o10e0i", "'e' exponent requires decimal mantissa"},

		// 0-octals
		{IntLit, "0", "0", ""},
		{IntLit, "0123", "0123", ""},

		{IntLit, "08123", "08123", "invalid digit '8' in octal literal"},
		{IntLit, "01293", "01293", "invalid digit '9' in octal literal"},
		{IntLit, "0F.", "0 F .", ""}, // only accept 0-9
		{IntLit, "0123F.", "0123 F .", ""},
		{IntLit, "0123456x", "0123456 x", ""},

		// decimals
		{IntLit, "1", "1", ""},
		{IntLit, "1234", "1234", ""},

		{IntLit, "1f", "1 f", ""}, // only accept 0-9

		{ImagLit, "0i", "0i", ""},
		{ImagLit, "0678i", "0678i", ""},

		// decimal floats
		{FloatLit, "0.", "0.", ""},
		{FloatLit, "123.", "123.", ""},
		{FloatLit, "0123.", "0123.", ""},

		{FloatLit, ".0", ".0", ""},
		{FloatLit, ".123", ".123", ""},
		{FloatLit, ".0123", ".0123", ""},

		{FloatLit, "0.0", "0.0", ""},
		{FloatLit, "123.123", "123.123", ""},
		{FloatLit, "0123.0123", "0123.0123", ""},

		{FloatLit, "0e0", "0e0", ""},
		{FloatLit, "123e+0", "123e+0", ""},
		{FloatLit, "0123E-1", "0123E-1", ""},

		{FloatLit, "0.e+1", "0.e+1", ""},
		{FloatLit, "123.E-10", "123.E-10", ""},
		{FloatLit, "0123.e123", "0123.e123", ""},

		{FloatLit, ".0e-1", ".0e-1", ""},
		{FloatLit, ".123E+10", ".123E+10", ""},
		{FloatLit, ".0123E123", ".0123E123", ""},

		{FloatLit, "0.0e1", "0.0e1", ""},
		{FloatLit, "123.123E-10", "123.123E-10", ""},
		{FloatLit, "0123.0123e+456", "0123.0123e+456", ""},

		{FloatLit, "0e", "0e", "exponent has no digits"},
		{FloatLit, "0E+", "0E+", "exponent has no digits"},
		{FloatLit, "1e+f", "1e+ f", "exponent has no digits"},
		{FloatLit, "0p0", "0p0", "'p' exponent requires hexadecimal mantissa"},
		{FloatLit, "1.0P-1", "1.0P-1", "'P' exponent requires hexadecimal mantissa"},

		{ImagLit, "0.i", "0.i", ""},
		{ImagLit, ".123i", ".123i", ""},
		{ImagLit, "123.123i", "123.123i", ""},
		{ImagLit, "123e+0i", "123e+0i", ""},
		{ImagLit, "123.E-10i", "123.E-10i", ""},
		{ImagLit, ".123E+10i", ".123E+10i", ""},

		// hexadecimals
		{IntLit, "0x0", "0x0", ""},
		{IntLit, "0x1234", "0x1234", ""},
		{IntLit, "0xcafef00d", "0xcafef00d", ""},
		{IntLit, "0XCAFEF00D", "0XCAFEF00D", ""},

		{IntLit, "0x", "0x", "hexadecimal literal has no digits"},
		{IntLit, "0x1g", "0x1 g", ""},

		{ImagLit, "0xf00i", "0xf00i", ""},

		// hexadecimal floats
		{FloatLit, "0x0p0", "0x0p0", ""},
		{FloatLit, "0x12efp-123", "0x12efp-123", ""},
		{FloatLit, "0xABCD.p+0", "0xABCD.p+0", ""},
		{FloatLit, "0x.0189P-0", "0x.0189P-0", ""},
		{FloatLit, "0x1.ffffp+1023", "0x1.ffffp+1023", ""},

		{FloatLit, "0x.", "0x.", "hexadecimal literal has no digits"},
		{FloatLit, "0x0.", "0x0.", "hexadecimal mantissa requires a 'p' exponent"},
		{FloatLit, "0x.0", "0x.0", "hexadecimal mantissa requires a 'p' exponent"},
		{FloatLit, "0x1.1", "0x1.1", "hexadecimal mantissa requires a 'p' exponent"},
		{FloatLit, "0x1.1e0", "0x1.1e0", "hexadecimal mantissa requires a 'p' exponent"},
		{FloatLit, "0x1.2gp1a", "0x1.2 gp1a", "hexadecimal mantissa requires a 'p' exponent"},
		{FloatLit, "0x0p", "0x0p", "exponent has no digits"},
		{FloatLit, "0xeP-", "0xeP-", "exponent has no digits"},
		{FloatLit, "0x1234PAB", "0x1234P AB", "exponent has no digits"},
		{FloatLit, "0x1.2p1a", "0x1.2p1 a", ""},

		{ImagLit, "0xf00.bap+12i", "0xf00.bap+12i", ""},

		// separators
		{IntLit, "0b_1000_0001", "0b_1000_0001", ""},
		{IntLit, "0o_600", "0o_600", ""},
		{IntLit, "0_466", "0_466", ""},
		{IntLit, "1_000", "1_000", ""},
		{FloatLit, "1_000.000_1", "1_000.000_1", ""},
		{ImagLit, "10e+1_2_3i", "10e+1_2_3i", ""},
		{IntLit, "0x_f00d", "0x_f00d", ""},
		{FloatLit, "0x_f00d.0p1_2", "0x_f00d.0p1_2", ""},

		{IntLit, "0b__1000", "0b__1000", "'_' must separate successive digits"},
		{IntLit, "0o60___0", "0o60___0", "'_' must separate successive digits"},
		{IntLit, "0466_", "0466_", "'_' must separate successive digits"},
		{FloatLit, "1_.", "1_.", "'_' must separate successive digits"},
		{FloatLit, "0._1", "0._1", "'_' must separate successive digits"},
		{FloatLit, "2.7_e0", "2.7_e0", "'_' must separate successive digits"},
		{ImagLit, "10e+12_i", "10e+12_i", "'_' must separate successive digits"},
		{IntLit, "0x___0", "0x___0", "'_' must separate successive digits"},
		{FloatLit, "0x1.0_p0", "0x1.0_p0", "'_' must separate successive digits"},
	} {
		var s scanner
		var err string
		s.init(strings.NewReader(test.src), func(_, _ uint, msg string) {
			if err == "" {
				err = msg
			}
		}, 0)

		for i, want := range strings.Split(test.tokens, " ") {
			err = ""
			s.next()

			if err != "" && !s.bad {
				t.Errorf("%q: got error but bad not set", test.src)
			}

			// compute lit where where s.lit is not defined
			var lit string
			switch s.tok {
			case _Name, _Literal:
				lit = s.lit
			case _Dot:
				lit = "."
			}

			if i == 0 {
				if s.tok != _Literal || s.kind != test.kind {
					t.Errorf("%q: got token %s (kind = %d); want literal (kind = %d)", test.src, s.tok, s.kind, test.kind)
				}
				if err != test.err {
					t.Errorf("%q: got error %q; want %q", test.src, err, test.err)
				}
			}

			if lit != want {
				t.Errorf("%q: got literal %q (%s); want %s", test.src, lit, s.tok, want)
			}
		}

		// make sure we read all
		s.next()
		if s.tok == _Semi {
			s.next()
		}
		if s.tok != _EOF {
			t.Errorf("%q: got %s; want EOF", test.src, s.tok)
		}
	}
}

func TestScanErrors(t *testing.T) {
	for _, test := range []struct {
		src, err  string
		line, col uint // 0-based
	}{
		// Note: Positions for lexical errors are the earliest position
		// where the error is apparent, not the beginning of the respective
		// token.

		// rune-level errors
		{"fo\x00o", "invalid NUL character", 0, 2},
		{"foo\n\ufeff bar", "invalid BOM in the middle of the file", 1, 0},
		{"foo\n\n\xff    ", "invalid UTF-8 encoding", 2, 0},

		// token-level errors
		{"\u00BD" /* ½ */, "invalid character U+00BD '½' in identifier", 0, 0},
		{"\U0001d736\U0001d737\U0001d738_½" /* 𝜶𝜷𝜸_½ */, "invalid character U+00BD '½' in identifier", 0, 13 /* byte offset */},
		{"\U0001d7d8" /* 𝟘 */, "identifier cannot begin with digit U+1D7D8 '𝟘'", 0, 0},
		{"foo\U0001d7d8_½" /* foo𝟘_½ */, "invalid character U+00BD '½' in identifier", 0, 8 /* byte offset */},

		{"x + ~y", "invalid character U+007E '~'", 0, 4},
		{"foo$bar = 0", "invalid character U+0024 '$'", 0, 3},
		{"0123456789", "invalid digit '8' in octal literal", 0, 8},
		{"0123456789. /* foobar", "comment not terminated", 0, 12},   // valid float constant
		{"0123456789e0 /*\nfoobar", "comment not terminated", 0, 13}, // valid float constant
		{"var a, b = 09, 07\n", "invalid digit '9' in octal literal", 0, 12},

		{`''`, "empty rune literal or unescaped '", 0, 1},
		{"'\n", "newline in rune literal", 0, 1},
		{`'\`, "rune literal not terminated", 0, 0},
		{`'\'`, "rune literal not terminated", 0, 0},
		{`'\x`, "rune literal not terminated", 0, 0},
		{`'\x'`, "invalid character '\\'' in hexadecimal escape", 0, 3},
		{`'\y'`, "unknown escape", 0, 2},
		{`'\x0'`, "invalid character '\\'' in hexadecimal escape", 0, 4},
		{`'\00'`, "invalid character '\\'' in octal escape", 0, 4},
		{`'\377' /*`, "comment not terminated", 0, 7}, // valid octal escape
		{`'\378`, "invalid character '8' in octal escape", 0, 4},
		{`'\400'`, "octal escape value 256 > 255", 0, 5},
		{`'xx`, "rune literal not terminated", 0, 0},
		{`'xx'`, "more than one character in rune literal", 0, 0},

		{"\n   \"foo\n", "newline in string", 1, 7},
		{`"`, "string not terminated", 0, 0},
		{`"foo`, "string not terminated", 0, 0},
		{"`", "string not terminated", 0, 0},
		{"`foo", "string not terminated", 0, 0},
		{"/*/", "comment not terminated", 0, 0},
		{"/*\n\nfoo", "comment not terminated", 0, 0},
		{`"\`, "string not terminated", 0, 0},
		{`"\"`, "string not terminated", 0, 0},
		{`"\x`, "string not terminated", 0, 0},
		{`"\x"`, "invalid character '\"' in hexadecimal escape", 0, 3},
		{`"\y"`, "unknown escape", 0, 2},
		{`"\x0"`, "invalid character '\"' in hexadecimal escape", 0, 4},
		{`"\00"`, "invalid character '\"' in octal escape", 0, 4},
		{`"\377" /*`, "comment not terminated", 0, 7}, // valid octal escape
		{`"\378"`, "invalid character '8' in octal escape", 0, 4},
		{`"\400"`, "octal escape value 256 > 255", 0, 5},

		{`s := "foo\z"`, "unknown escape", 0, 10},
		{`s := "foo\z00\nbar"`, "unknown escape", 0, 10},
		{`"\x`, "string not terminated", 0, 0},
		{`"\x"`, "invalid character '\"' in hexadecimal escape", 0, 3},
		{`var s string = "\x"`, "invalid character '\"' in hexadecimal escape", 0, 18},
		{`return "\Uffffffff"`, "escape is invalid Unicode code point U+FFFFFFFF", 0, 18},

		{"0b.0", "invalid radix point in binary literal", 0, 2},
		{"0x.p0\n", "hexadecimal literal has no digits", 0, 3},

		// former problem cases
		{"package p\n\n\xef", "invalid UTF-8 encoding", 2, 0},
	} {
		var s scanner
		var line, col uint
		var err string
		s.init(strings.NewReader(test.src), func(l, c uint, msg string) {
			if err == "" {
				line, col = l-linebase, c-colbase
				err = msg
			}
		}, 0)

		for {
			s.next()
			if s.tok == _EOF {
				break
			}
		}

		if err != "" {
			if err != test.err {
				t.Errorf("%q: got err = %q; want %q", test.src, err, test.err)
			}
			if line != test.line {
				t.Errorf("%q: got line = %d; want %d", test.src, line, test.line)
			}
			if col != test.col {
				t.Errorf("%q: got col = %d; want %d", test.src, col, test.col)
			}
		} else {
			t.Errorf("%q: got no error; want %q", test.src, test.err)
		}
	}
}

func TestDirectives(t *testing.T) {
	for _, src := range []string{
		"line",
		"// line",
		"//line",
		"//line foo",
		"//line foo%bar",

		"go",
		"// go:",
		"//go:",
		"//go :foo",
		"//go:foo",
		"//go:foo%bar",
	} {
		got := ""
		var s scanner
		s.init(strings.NewReader(src), func(_, col uint, msg string) {
			if col != colbase {
				t.Errorf("%s: got col = %d; want %d", src, col, colbase)
			}
			if msg == "" {
				t.Errorf("%s: handler called with empty msg", src)
			}
			got = msg
		}, directives)

		s.next()
		if strings.HasPrefix(src, "//line ") || strings.HasPrefix(src, "//go:") {
			// handler should have been called
			if got != src {
				t.Errorf("got %s; want %s", got, src)
			}
		} else {
			// handler should not have been called
			if got != "" {
				t.Errorf("got %s for %s", got, src)
			}
		}
	}
}

func TestIssue21938(t *testing.T) {
	s := "/*" + strings.Repeat(" ", 4089) + "*/ .5"

	var got scanner
	got.init(strings.NewReader(s), errh, 0)
	got.next()

	if got.tok != _Literal || got.lit != ".5" {
		t.Errorf("got %s %q; want %s %q", got.tok, got.lit, _Literal, ".5")
	}
}

func TestIssue33961(t *testing.T) {
	literals := `08__ 0b.p 0b_._p 0x.e 0x.p`
	for _, lit := range strings.Split(literals, " ") {
		n := 0
		var got scanner
		got.init(strings.NewReader(lit), func(_, _ uint, msg string) {
			// fmt.Printf("%s: %s\n", lit, msg) // uncomment for debugging
			n++
		}, 0)
		got.next()

		if n != 1 {
			t.Errorf("%q: got %d errors; want 1", lit, n)
			continue
		}

		if !got.bad {
			t.Errorf("%q: got error but bad not set", lit)
		}
	}
}
