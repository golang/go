// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"io";
	"token";
	"scanner";
	"testing";
)


const /* class */ (
	special = iota;
	literal;
	operator;
	keyword;
)


func tokenclass(tok int) int {
	switch {
	case token.IsLiteral(tok): return literal;
	case token.IsOperator(tok): return operator;
	case token.IsKeyword(tok): return keyword;
	}
	return special;
}


type elt struct {
	pos int;
	tok int;
	lit string;
	class int;
}


var tokens = [...]elt{
	// Special tokens
	elt{ 0, token.COMMENT, "/* a comment */", special },
	elt{ 0, token.COMMENT, "\n", special },

	// Identifiers and basic type literals
	elt{ 0, token.IDENT, "foobar", literal },
	elt{ 0, token.IDENT, "a۰۱۸", literal },
	elt{ 0, token.IDENT, "foo६४", literal },
	elt{ 0, token.IDENT, "bar９８７６", literal },
	elt{ 0, token.INT, "0", literal },
	elt{ 0, token.INT, "01234567", literal },
	elt{ 0, token.INT, "0xcafebabe", literal },
	elt{ 0, token.FLOAT, "0.", literal },
	elt{ 0, token.FLOAT, ".0", literal },
	elt{ 0, token.FLOAT, "3.14159265", literal },
	elt{ 0, token.FLOAT, "1e0", literal },
	elt{ 0, token.FLOAT, "1e+100", literal },
	elt{ 0, token.FLOAT, "1e-100", literal },
	elt{ 0, token.FLOAT, "2.71828e-1000", literal },
	elt{ 0, token.CHAR, "'a'", literal },
	elt{ 0, token.CHAR, "'\\000'", literal },
	elt{ 0, token.CHAR, "'\\xFF'", literal },
	elt{ 0, token.CHAR, "'\\uff16'", literal },
	elt{ 0, token.CHAR, "'\\U0000ff16'", literal },
	elt{ 0, token.STRING, "`foobar`", literal },

	// Operators and delimitors
	elt{ 0, token.ADD, "+", operator },
	elt{ 0, token.SUB, "-", operator },
	elt{ 0, token.MUL, "*", operator },
	elt{ 0, token.QUO, "/", operator },
	elt{ 0, token.REM, "%", operator },

	elt{ 0, token.AND, "&", operator },
	elt{ 0, token.OR, "|", operator },
	elt{ 0, token.XOR, "^", operator },
	elt{ 0, token.SHL, "<<", operator },
	elt{ 0, token.SHR, ">>", operator },

	elt{ 0, token.ADD_ASSIGN, "+=", operator },
	elt{ 0, token.SUB_ASSIGN, "-=", operator },
	elt{ 0, token.MUL_ASSIGN, "*=", operator },
	elt{ 0, token.QUO_ASSIGN, "/=", operator },
	elt{ 0, token.REM_ASSIGN, "%=", operator },

	elt{ 0, token.AND_ASSIGN, "&=", operator },
	elt{ 0, token.OR_ASSIGN, "|=", operator },
	elt{ 0, token.XOR_ASSIGN, "^=", operator },
	elt{ 0, token.SHL_ASSIGN, "<<=", operator },
	elt{ 0, token.SHR_ASSIGN, ">>=", operator },

	elt{ 0, token.LAND, "&&", operator },
	elt{ 0, token.LOR, "||", operator },
	elt{ 0, token.ARROW, "<-", operator },
	elt{ 0, token.INC, "++", operator },
	elt{ 0, token.DEC, "--", operator },

	elt{ 0, token.EQL, "==", operator },
	elt{ 0, token.LSS, "<", operator },
	elt{ 0, token.GTR, ">", operator },
	elt{ 0, token.ASSIGN, "=", operator },
	elt{ 0, token.NOT, "!", operator },

	elt{ 0, token.NEQ, "!=", operator },
	elt{ 0, token.LEQ, "<=", operator },
	elt{ 0, token.GEQ, ">=", operator },
	elt{ 0, token.DEFINE, ":=", operator },
	elt{ 0, token.ELLIPSIS, "...", operator },

	elt{ 0, token.LPAREN, "(", operator },
	elt{ 0, token.LBRACK, "[", operator },
	elt{ 0, token.LBRACE, "{", operator },
	elt{ 0, token.COMMA, ",", operator },
	elt{ 0, token.PERIOD, ".", operator },

	elt{ 0, token.RPAREN, ")", operator },
	elt{ 0, token.RBRACK, "]", operator },
	elt{ 0, token.RBRACE, "}", operator },
	elt{ 0, token.SEMICOLON, ";", operator },
	elt{ 0, token.COLON, ":", operator },

	// Keywords
	elt{ 0, token.BREAK, "break", keyword },
	elt{ 0, token.CASE, "case", keyword },
	elt{ 0, token.CHAN, "chan", keyword },
	elt{ 0, token.CONST, "const", keyword },
	elt{ 0, token.CONTINUE, "continue", keyword },

	elt{ 0, token.DEFAULT, "default", keyword },
	elt{ 0, token.DEFER, "defer", keyword },
	elt{ 0, token.ELSE, "else", keyword },
	elt{ 0, token.FALLTHROUGH, "fallthrough", keyword },
	elt{ 0, token.FOR, "for", keyword },

	elt{ 0, token.FUNC, "func", keyword },
	elt{ 0, token.GO, "go", keyword },
	elt{ 0, token.GOTO, "goto", keyword },
	elt{ 0, token.IF, "if", keyword },
	elt{ 0, token.IMPORT, "import", keyword },

	elt{ 0, token.INTERFACE, "interface", keyword },
	elt{ 0, token.MAP, "map", keyword },
	elt{ 0, token.PACKAGE, "package", keyword },
	elt{ 0, token.RANGE, "range", keyword },
	elt{ 0, token.RETURN, "return", keyword },

	elt{ 0, token.SELECT, "select", keyword },
	elt{ 0, token.STRUCT, "struct", keyword },
	elt{ 0, token.SWITCH, "switch", keyword },
	elt{ 0, token.TYPE, "type", keyword },
	elt{ 0, token.VAR, "var", keyword },
}


func init() {
	// set pos fields
	pos := 0;
	for i := 0; i < len(tokens); i++ {
		tokens[i].pos = pos;
		pos += len(tokens[i].lit) + 1;  // + 1 for space in between
	}
}


type TestErrorHandler struct {
	t *testing.T
}

func (h *TestErrorHandler) Error(pos int, msg string) {
	h.t.Errorf("Error() called (pos = %d, msg = %s)", pos, msg);
}


func Test(t *testing.T) {
	// make source
	var src string;
	for i, e := range tokens {
		src += e.lit + " ";
	}

	// set up scanner
	var s scanner.Scanner;
	s.Init(io.StringBytes(src), &TestErrorHandler{t}, true);

	// verify scan
	for i, e := range tokens {
		pos, tok, lit := s.Scan();
		if pos != e.pos {
			t.Errorf("bad position for %s: got %d, expected %d", e.lit, pos, e.pos);
		}
		if tok != e.tok {
			t.Errorf("bad token for %s: got %s, expected %s", e.lit, token.TokenString(tok), token.TokenString(e.tok));
		}
		if token.IsLiteral(e.tok) && string(lit) != e.lit {
			t.Errorf("bad literal for %s: got %s, expected %s", e.lit, string(lit), e.lit);
		}
		if tokenclass(tok) != e.class {
			t.Errorf("bad class for %s: got %d, expected %d", e.lit, tokenclass(tok), e.class);
		}
	}
	pos, tok, lit := s.Scan();
	if tok != token.EOF {
		t.Errorf("bad token at eof: got %s, expected EOF", token.TokenString(tok));
	}
	if tokenclass(tok) != special {
		t.Errorf("bad class at eof: got %d, expected %d", tokenclass(tok), special);
	}
}
