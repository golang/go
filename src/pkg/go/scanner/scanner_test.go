// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"go/scanner";
	"go/token";
	"strings";
	"testing";
)


const /* class */ (
	special = iota;
	literal;
	operator;
	keyword;
)


func tokenclass(tok token.Token) int {
	switch {
	case tok.IsLiteral(): return literal;
	case tok.IsOperator(): return operator;
	case tok.IsKeyword(): return keyword;
	}
	return special;
}


type elt struct {
	tok token.Token;
	lit string;
	class int;
}


var tokens = [...]elt{
	// Special tokens
	elt{ token.COMMENT, "/* a comment */", special },
	elt{ token.COMMENT, "// a comment \n", special },

	// Identifiers and basic type literals
	elt{ token.IDENT, "foobar", literal },
	elt{ token.IDENT, "a۰۱۸", literal },
	elt{ token.IDENT, "foo६४", literal },
	elt{ token.IDENT, "bar９８７６", literal },
	elt{ token.INT, "0", literal },
	elt{ token.INT, "01234567", literal },
	elt{ token.INT, "0xcafebabe", literal },
	elt{ token.FLOAT, "0.", literal },
	elt{ token.FLOAT, ".0", literal },
	elt{ token.FLOAT, "3.14159265", literal },
	elt{ token.FLOAT, "1e0", literal },
	elt{ token.FLOAT, "1e+100", literal },
	elt{ token.FLOAT, "1e-100", literal },
	elt{ token.FLOAT, "2.71828e-1000", literal },
	elt{ token.CHAR, "'a'", literal },
	elt{ token.CHAR, "'\\000'", literal },
	elt{ token.CHAR, "'\\xFF'", literal },
	elt{ token.CHAR, "'\\uff16'", literal },
	elt{ token.CHAR, "'\\U0000ff16'", literal },
	elt{ token.STRING, "`foobar`", literal },
	elt{ token.STRING, "`" `foo
	                        bar` "`", literal },

	// Operators and delimitors
	elt{ token.ADD, "+", operator },
	elt{ token.SUB, "-", operator },
	elt{ token.MUL, "*", operator },
	elt{ token.QUO, "/", operator },
	elt{ token.REM, "%", operator },

	elt{ token.AND, "&", operator },
	elt{ token.OR, "|", operator },
	elt{ token.XOR, "^", operator },
	elt{ token.SHL, "<<", operator },
	elt{ token.SHR, ">>", operator },
	elt{ token.AND_NOT, "&^", operator },

	elt{ token.ADD_ASSIGN, "+=", operator },
	elt{ token.SUB_ASSIGN, "-=", operator },
	elt{ token.MUL_ASSIGN, "*=", operator },
	elt{ token.QUO_ASSIGN, "/=", operator },
	elt{ token.REM_ASSIGN, "%=", operator },

	elt{ token.AND_ASSIGN, "&=", operator },
	elt{ token.OR_ASSIGN, "|=", operator },
	elt{ token.XOR_ASSIGN, "^=", operator },
	elt{ token.SHL_ASSIGN, "<<=", operator },
	elt{ token.SHR_ASSIGN, ">>=", operator },
	elt{ token.AND_NOT_ASSIGN, "&^=", operator },

	elt{ token.LAND, "&&", operator },
	elt{ token.LOR, "||", operator },
	elt{ token.ARROW, "<-", operator },
	elt{ token.INC, "++", operator },
	elt{ token.DEC, "--", operator },

	elt{ token.EQL, "==", operator },
	elt{ token.LSS, "<", operator },
	elt{ token.GTR, ">", operator },
	elt{ token.ASSIGN, "=", operator },
	elt{ token.NOT, "!", operator },

	elt{ token.NEQ, "!=", operator },
	elt{ token.LEQ, "<=", operator },
	elt{ token.GEQ, ">=", operator },
	elt{ token.DEFINE, ":=", operator },
	elt{ token.ELLIPSIS, "...", operator },

	elt{ token.LPAREN, "(", operator },
	elt{ token.LBRACK, "[", operator },
	elt{ token.LBRACE, "{", operator },
	elt{ token.COMMA, ",", operator },
	elt{ token.PERIOD, ".", operator },

	elt{ token.RPAREN, ")", operator },
	elt{ token.RBRACK, "]", operator },
	elt{ token.RBRACE, "}", operator },
	elt{ token.SEMICOLON, ";", operator },
	elt{ token.COLON, ":", operator },

	// Keywords
	elt{ token.BREAK, "break", keyword },
	elt{ token.CASE, "case", keyword },
	elt{ token.CHAN, "chan", keyword },
	elt{ token.CONST, "const", keyword },
	elt{ token.CONTINUE, "continue", keyword },

	elt{ token.DEFAULT, "default", keyword },
	elt{ token.DEFER, "defer", keyword },
	elt{ token.ELSE, "else", keyword },
	elt{ token.FALLTHROUGH, "fallthrough", keyword },
	elt{ token.FOR, "for", keyword },

	elt{ token.FUNC, "func", keyword },
	elt{ token.GO, "go", keyword },
	elt{ token.GOTO, "goto", keyword },
	elt{ token.IF, "if", keyword },
	elt{ token.IMPORT, "import", keyword },

	elt{ token.INTERFACE, "interface", keyword },
	elt{ token.MAP, "map", keyword },
	elt{ token.PACKAGE, "package", keyword },
	elt{ token.RANGE, "range", keyword },
	elt{ token.RETURN, "return", keyword },

	elt{ token.SELECT, "select", keyword },
	elt{ token.STRUCT, "struct", keyword },
	elt{ token.SWITCH, "switch", keyword },
	elt{ token.TYPE, "type", keyword },
	elt{ token.VAR, "var", keyword },
}


const whitespace = "  \t  \n\n\n";  // to separate tokens

type TestErrorHandler struct {
	t *testing.T
}

func (h *TestErrorHandler) Error(pos token.Position, msg string) {
	h.t.Errorf("Error() called (msg = %s)", msg);
}


func NewlineCount(s string) int {
	n := 0;
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			n++;
		}
	}
	return n;
}


// Verify that calling Scan() provides the correct results.
func TestScan(t *testing.T) {
	// make source
	var src string;
	for i, e := range tokens {
		src += e.lit + whitespace;
	}
	whitespace_linecount := NewlineCount(whitespace);

	// verify scan
	index := 0;
	eloc := token.Position{0, 1, 1};
	nerrors := scanner.Tokenize(strings.Bytes(src), &TestErrorHandler{t}, scanner.ScanComments,
		func (pos token.Position, tok token.Token, litb []byte) bool {
			e := elt{token.EOF, "", special};
			if index < len(tokens) {
				e = tokens[index];
			}
			lit := string(litb);
			if tok == token.EOF {
				lit = "<EOF>";
				eloc.Column = 0;
			}
			if pos.Offset != eloc.Offset {
				t.Errorf("bad position for %s: got %d, expected %d", lit, pos.Offset, eloc.Offset);
			}
			if pos.Line != eloc.Line {
				t.Errorf("bad line for %s: got %d, expected %d", lit, pos.Line, eloc.Line);
			}
			if pos.Column!= eloc.Column {
				t.Errorf("bad column for %s: got %d, expected %d", lit, pos.Column, eloc.Column);
			}
			if tok != e.tok {
				t.Errorf("bad token for %s: got %s, expected %s", lit, tok.String(), e.tok.String());
			}
			if e.tok.IsLiteral() && lit != e.lit {
				t.Errorf("bad literal for %s: got %s, expected %s", lit, lit, e.lit);
			}
			if tokenclass(tok) != e.class {
				t.Errorf("bad class for %s: got %d, expected %d", lit, tokenclass(tok), e.class);
			}
			eloc.Offset += len(lit) + len(whitespace);
			eloc.Line += NewlineCount(lit) + whitespace_linecount;
			if tok == token.COMMENT && litb[1] == '/' {
				// correct for unaccounted '/n' in //-style comment
				eloc.Offset++;
				eloc.Line++;
			}
			index++;
			return tok != token.EOF;
		}
	);
	if nerrors != 0 {
		t.Errorf("found %d errors", nerrors);
	}
}


// Verify that initializing the same scanner more then once works correctly.
func TestInit(t *testing.T) {
	var s scanner.Scanner;

	// 1st init
	s.Init(strings.Bytes("if true { }"), nil, 0);
	s.Scan();  // if
	s.Scan();  // true
	pos, tok, lit := s.Scan();  // {
	if tok != token.LBRACE {
		t.Errorf("bad token: got %s, expected %s", tok.String(), token.LBRACE);
	}

	// 2nd init
	s.Init(strings.Bytes("go true { ]"), nil, 0);
	pos, tok, lit = s.Scan();  // go
	if tok != token.GO {
		t.Errorf("bad token: got %s, expected %s", tok.String(), token.GO);
	}

	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount);
	}
}


func TestIllegalChars(t *testing.T) {
	var s scanner.Scanner;

	const src = "*?*$*@*";
	s.Init(strings.Bytes(src), &TestErrorHandler{t}, scanner.AllowIllegalChars);
	for offs, ch := range src {
		pos, tok, lit := s.Scan();
		if pos.Offset != offs {
			t.Errorf("bad position for %s: got %d, expected %d", string(lit), pos.Offset, offs);
		}
		if tok == token.ILLEGAL && string(lit) != string(ch) {
			t.Errorf("bad token: got %s, expected %s", string(lit), string(ch));
		}
	}

	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount);
	}
}
