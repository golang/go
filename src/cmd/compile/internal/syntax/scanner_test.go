// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"os"
	"testing"
)

func TestScanner(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	src, err := os.Open("parser.go")
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	var s scanner
	s.init(src, nil, nil)
	for {
		s.next()
		if s.tok == _EOF {
			break
		}
		switch s.tok {
		case _Name:
			fmt.Println(s.line, s.tok, "=>", s.lit)
		case _Operator:
			fmt.Println(s.line, s.tok, "=>", s.op, s.prec)
		default:
			fmt.Println(s.line, s.tok)
		}
	}
}

func TestTokens(t *testing.T) {
	// make source
	var buf []byte
	for i, s := range sampleTokens {
		buf = append(buf, "\t\t\t\t"[:i&3]...)     // leading indentation
		buf = append(buf, s.src...)                // token
		buf = append(buf, "        "[:i&7]...)     // trailing spaces
		buf = append(buf, "/* foo */ // bar\n"...) // comments
	}

	// scan source
	var got scanner
	got.init(&bytesReader{buf}, nil, nil)
	got.next()
	for i, want := range sampleTokens {
		nlsemi := false

		if got.line != uint(i+linebase) {
			t.Errorf("got line %d; want %d", got.line, i+linebase)
		}

		if got.tok != want.tok {
			t.Errorf("got tok = %s; want %s", got.tok, want.tok)
			continue
		}

		switch want.tok {
		case _Semi:
			if got.lit != "semicolon" {
				t.Errorf("got %s; want semicolon", got.lit)
			}

		case _Name, _Literal:
			if got.lit != want.src {
				t.Errorf("got lit = %q; want %q", got.lit, want.src)
				continue
			}
			nlsemi = true

		case _Operator, _AssignOp, _IncOp:
			if got.op != want.op {
				t.Errorf("got op = %s; want %s", got.op, want.op)
				continue
			}
			if got.prec != want.prec {
				t.Errorf("got prec = %d; want %d", got.prec, want.prec)
				continue
			}
			nlsemi = want.tok == _IncOp

		case _Rparen, _Rbrack, _Rbrace, _Break, _Continue, _Fallthrough, _Return:
			nlsemi = true
		}

		if nlsemi {
			got.next()
			if got.tok != _Semi {
				t.Errorf("got tok = %s; want ;", got.tok)
				continue
			}
			if got.lit != "newline" {
				t.Errorf("got %s; want newline", got.lit)
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
	{_Literal, "0x0", 0, 0},
	{_Literal, "0xcafebabe", 0, 0},
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

func TestScanErrors(t *testing.T) {
	for _, test := range []struct {
		src, msg  string
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
		{"\u00BD" /* ½ */, "invalid identifier character U+00BD '½'", 0, 0},
		{"\U0001d736\U0001d737\U0001d738_½" /* 𝜶𝜷𝜸_½ */, "invalid identifier character U+00BD '½'", 0, 13 /* byte offset */},
		{"\U0001d7d8" /* 𝟘 */, "identifier cannot begin with digit U+1D7D8 '𝟘'", 0, 0},
		{"foo\U0001d7d8_½" /* foo𝟘_½ */, "invalid identifier character U+00BD '½'", 0, 8 /* byte offset */},

		{"x + ~y", "bitwise complement operator is ^", 0, 4},
		{"foo$bar = 0", "invalid character U+0024 '$'", 0, 3},
		{"const x = 0xyz", "malformed hex constant", 0, 12},
		{"0123456789", "malformed octal constant", 0, 10},
		{"0123456789. /* foobar", "comment not terminated", 0, 12},   // valid float constant
		{"0123456789e0 /*\nfoobar", "comment not terminated", 0, 13}, // valid float constant
		{"var a, b = 08, 07\n", "malformed octal constant", 0, 13},
		{"(x + 1.0e+x)", "malformed floating-point constant exponent", 0, 10},

		{`''`, "empty character literal or unescaped ' in character literal", 0, 1},
		{"'\n", "newline in character literal", 0, 1},
		{`'\`, "invalid character literal (missing closing ')", 0, 0},
		{`'\'`, "invalid character literal (missing closing ')", 0, 0},
		{`'\x`, "invalid character literal (missing closing ')", 0, 0},
		{`'\x'`, "non-hex character in escape sequence: '", 0, 3},
		{`'\y'`, "unknown escape sequence", 0, 2},
		{`'\x0'`, "non-hex character in escape sequence: '", 0, 4},
		{`'\00'`, "non-octal character in escape sequence: '", 0, 4},
		{`'\377' /*`, "comment not terminated", 0, 7}, // valid octal escape
		{`'\378`, "non-octal character in escape sequence: 8", 0, 4},
		{`'\400'`, "octal escape value > 255: 256", 0, 5},
		{`'xx`, "invalid character literal (missing closing ')", 0, 0},
		{`'xx'`, "invalid character literal (more than one character)", 0, 0},

		{"\"\n", "newline in string", 0, 1},
		{`"`, "string not terminated", 0, 0},
		{`"foo`, "string not terminated", 0, 0},
		{"`", "string not terminated", 0, 0},
		{"`foo", "string not terminated", 0, 0},
		{"/*/", "comment not terminated", 0, 0},
		{"/*\n\nfoo", "comment not terminated", 0, 0},
		{"/*\n\nfoo", "comment not terminated", 0, 0},
		{`"\`, "string not terminated", 0, 0},
		{`"\"`, "string not terminated", 0, 0},
		{`"\x`, "string not terminated", 0, 0},
		{`"\x"`, "non-hex character in escape sequence: \"", 0, 3},
		{`"\y"`, "unknown escape sequence", 0, 2},
		{`"\x0"`, "non-hex character in escape sequence: \"", 0, 4},
		{`"\00"`, "non-octal character in escape sequence: \"", 0, 4},
		{`"\377" /*`, "comment not terminated", 0, 7}, // valid octal escape
		{`"\378"`, "non-octal character in escape sequence: 8", 0, 4},
		{`"\400"`, "octal escape value > 255: 256", 0, 5},

		{`s := "foo\z"`, "unknown escape sequence", 0, 10},
		{`s := "foo\z00\nbar"`, "unknown escape sequence", 0, 10},
		{`"\x`, "string not terminated", 0, 0},
		{`"\x"`, "non-hex character in escape sequence: \"", 0, 3},
		{`var s string = "\x"`, "non-hex character in escape sequence: \"", 0, 18},
		{`return "\Uffffffff"`, "escape sequence is invalid Unicode code point", 0, 18},

		// former problem cases
		{"package p\n\n\xef", "invalid UTF-8 encoding", 2, 0},
	} {
		var s scanner
		nerrors := 0
		s.init(&bytesReader{[]byte(test.src)}, func(line, col uint, msg string) {
			nerrors++
			// only check the first error
			if nerrors == 1 {
				if msg != test.msg {
					t.Errorf("%q: got msg = %q; want %q", test.src, msg, test.msg)
				}
				if line != test.line+linebase {
					t.Errorf("%q: got line = %d; want %d", test.src, line, test.line+linebase)
				}
				if col != test.col+colbase {
					t.Errorf("%q: got col = %d; want %d", test.src, col, test.col+colbase)
				}
			} else if nerrors > 1 {
				// TODO(gri) make this use position info
				t.Errorf("%q: got unexpected %q at line = %d", test.src, msg, line)
			}
		}, nil)

		for {
			s.next()
			if s.tok == _EOF {
				break
			}
		}

		if nerrors == 0 {
			t.Errorf("%q: got no error; want %q", test.src, test.msg)
		}
	}
}
