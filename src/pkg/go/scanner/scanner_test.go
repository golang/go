// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"go/token";
	"os";
	"strings";
	"testing";
)


const /* class */ (
	special	= iota;
	literal;
	operator;
	keyword;
)


func tokenclass(tok token.Token) int {
	switch {
	case tok.IsLiteral():
		return literal
	case tok.IsOperator():
		return operator
	case tok.IsKeyword():
		return keyword
	}
	return special;
}


type elt struct {
	tok	token.Token;
	lit	string;
	class	int;
}


var tokens = [...]elt{
	// Special tokens
	elt{token.COMMENT, "/* a comment */", special},
	elt{token.COMMENT, "// a comment \n", special},

	// Identifiers and basic type literals
	elt{token.IDENT, "foobar", literal},
	elt{token.IDENT, "a۰۱۸", literal},
	elt{token.IDENT, "foo६४", literal},
	elt{token.IDENT, "bar９８７６", literal},
	elt{token.INT, "0", literal},
	elt{token.INT, "01234567", literal},
	elt{token.INT, "0xcafebabe", literal},
	elt{token.FLOAT, "0.", literal},
	elt{token.FLOAT, ".0", literal},
	elt{token.FLOAT, "3.14159265", literal},
	elt{token.FLOAT, "1e0", literal},
	elt{token.FLOAT, "1e+100", literal},
	elt{token.FLOAT, "1e-100", literal},
	elt{token.FLOAT, "2.71828e-1000", literal},
	elt{token.CHAR, "'a'", literal},
	elt{token.CHAR, "'\\000'", literal},
	elt{token.CHAR, "'\\xFF'", literal},
	elt{token.CHAR, "'\\uff16'", literal},
	elt{token.CHAR, "'\\U0000ff16'", literal},
	elt{token.STRING, "`foobar`", literal},
	elt{token.STRING, "`" `foo
	                        bar`
		"`",
		literal,
	},

	// Operators and delimitors
	elt{token.ADD, "+", operator},
	elt{token.SUB, "-", operator},
	elt{token.MUL, "*", operator},
	elt{token.QUO, "/", operator},
	elt{token.REM, "%", operator},

	elt{token.AND, "&", operator},
	elt{token.OR, "|", operator},
	elt{token.XOR, "^", operator},
	elt{token.SHL, "<<", operator},
	elt{token.SHR, ">>", operator},
	elt{token.AND_NOT, "&^", operator},

	elt{token.ADD_ASSIGN, "+=", operator},
	elt{token.SUB_ASSIGN, "-=", operator},
	elt{token.MUL_ASSIGN, "*=", operator},
	elt{token.QUO_ASSIGN, "/=", operator},
	elt{token.REM_ASSIGN, "%=", operator},

	elt{token.AND_ASSIGN, "&=", operator},
	elt{token.OR_ASSIGN, "|=", operator},
	elt{token.XOR_ASSIGN, "^=", operator},
	elt{token.SHL_ASSIGN, "<<=", operator},
	elt{token.SHR_ASSIGN, ">>=", operator},
	elt{token.AND_NOT_ASSIGN, "&^=", operator},

	elt{token.LAND, "&&", operator},
	elt{token.LOR, "||", operator},
	elt{token.ARROW, "<-", operator},
	elt{token.INC, "++", operator},
	elt{token.DEC, "--", operator},

	elt{token.EQL, "==", operator},
	elt{token.LSS, "<", operator},
	elt{token.GTR, ">", operator},
	elt{token.ASSIGN, "=", operator},
	elt{token.NOT, "!", operator},

	elt{token.NEQ, "!=", operator},
	elt{token.LEQ, "<=", operator},
	elt{token.GEQ, ">=", operator},
	elt{token.DEFINE, ":=", operator},
	elt{token.ELLIPSIS, "...", operator},

	elt{token.LPAREN, "(", operator},
	elt{token.LBRACK, "[", operator},
	elt{token.LBRACE, "{", operator},
	elt{token.COMMA, ",", operator},
	elt{token.PERIOD, ".", operator},

	elt{token.RPAREN, ")", operator},
	elt{token.RBRACK, "]", operator},
	elt{token.RBRACE, "}", operator},
	elt{token.SEMICOLON, ";", operator},
	elt{token.COLON, ":", operator},

	// Keywords
	elt{token.BREAK, "break", keyword},
	elt{token.CASE, "case", keyword},
	elt{token.CHAN, "chan", keyword},
	elt{token.CONST, "const", keyword},
	elt{token.CONTINUE, "continue", keyword},

	elt{token.DEFAULT, "default", keyword},
	elt{token.DEFER, "defer", keyword},
	elt{token.ELSE, "else", keyword},
	elt{token.FALLTHROUGH, "fallthrough", keyword},
	elt{token.FOR, "for", keyword},

	elt{token.FUNC, "func", keyword},
	elt{token.GO, "go", keyword},
	elt{token.GOTO, "goto", keyword},
	elt{token.IF, "if", keyword},
	elt{token.IMPORT, "import", keyword},

	elt{token.INTERFACE, "interface", keyword},
	elt{token.MAP, "map", keyword},
	elt{token.PACKAGE, "package", keyword},
	elt{token.RANGE, "range", keyword},
	elt{token.RETURN, "return", keyword},

	elt{token.SELECT, "select", keyword},
	elt{token.STRUCT, "struct", keyword},
	elt{token.SWITCH, "switch", keyword},
	elt{token.TYPE, "type", keyword},
	elt{token.VAR, "var", keyword},
}


const whitespace = "  \t  \n\n\n"	// to separate tokens

type TestErrorHandler struct {
	t *testing.T;
}

func (h *TestErrorHandler) Error(pos token.Position, msg string) {
	h.t.Errorf("Error() called (msg = %s)", msg)
}


func NewlineCount(s string) int {
	n := 0;
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			n++
		}
	}
	return n;
}


func checkPos(t *testing.T, lit string, pos, expected token.Position) {
	if pos.Filename != expected.Filename {
		t.Errorf("bad filename for %s: got %s, expected %s", lit, pos.Filename, expected.Filename)
	}
	if pos.Offset != expected.Offset {
		t.Errorf("bad position for %s: got %d, expected %d", lit, pos.Offset, expected.Offset)
	}
	if pos.Line != expected.Line {
		t.Errorf("bad line for %s: got %d, expected %d", lit, pos.Line, expected.Line)
	}
	if pos.Column != expected.Column {
		t.Errorf("bad column for %s: got %d, expected %d", lit, pos.Column, expected.Column)
	}
}


// Verify that calling Scan() provides the correct results.
func TestScan(t *testing.T) {
	// make source
	var src string;
	for _, e := range tokens {
		src += e.lit + whitespace
	}
	whitespace_linecount := NewlineCount(whitespace);

	// verify scan
	index := 0;
	epos := token.Position{"", 0, 1, 1};
	nerrors := Tokenize("", strings.Bytes(src), &TestErrorHandler{t}, ScanComments,
		func(pos token.Position, tok token.Token, litb []byte) bool {
			e := elt{token.EOF, "", special};
			if index < len(tokens) {
				e = tokens[index]
			}
			lit := string(litb);
			if tok == token.EOF {
				lit = "<EOF>";
				epos.Column = 0;
			}
			checkPos(t, lit, pos, epos);
			if tok != e.tok {
				t.Errorf("bad token for %s: got %s, expected %s", lit, tok.String(), e.tok.String())
			}
			if e.tok.IsLiteral() && lit != e.lit {
				t.Errorf("bad literal for %s: got %s, expected %s", lit, lit, e.lit)
			}
			if tokenclass(tok) != e.class {
				t.Errorf("bad class for %s: got %d, expected %d", lit, tokenclass(tok), e.class)
			}
			epos.Offset += len(lit)+len(whitespace);
			epos.Line += NewlineCount(lit) + whitespace_linecount;
			if tok == token.COMMENT && litb[1] == '/' {
				// correct for unaccounted '/n' in //-style comment
				epos.Offset++;
				epos.Line++;
			}
			index++;
			return tok != token.EOF;
		});
	if nerrors != 0 {
		t.Errorf("found %d errors", nerrors)
	}
}


type seg struct {
	srcline		string;	// a line of source text
	filename	string;	// filename for current token
	line		int;	// line number for current token
}


var segments = []seg{
	// exactly one token per line since the test consumes one token per segment
	seg{"  line1", "TestLineComments", 1},
	seg{"\nline2", "TestLineComments", 2},
	seg{"\nline3  //line File1.go:100", "TestLineComments", 3},	// bad line comment, ignored
	seg{"\nline4", "TestLineComments", 4},
	seg{"\n//line File1.go:100\n  line100", "File1.go", 100},
	seg{"\n//line File2.go:200\n  line200", "File2.go", 200},
	seg{"\n//line :1\n  line1", "", 1},
	seg{"\n//line foo:42\n  line42", "foo", 42},
	seg{"\n //line foo:42\n  line44", "foo", 44},	// bad line comment, ignored
	seg{"\n//line foo 42\n  line46", "foo", 46},	// bad line comment, ignored
	seg{"\n//line foo:42 extra text\n  line48", "foo", 48},	// bad line comment, ignored
	seg{"\n//line foo:42\n  line42", "foo", 42},
	seg{"\n//line foo:42\n  line42", "foo", 42},
	seg{"\n//line File1.go:100\n  line100", "File1.go", 100},
}


// Verify that comments of the form "//line filename:line" are interpreted correctly.
func TestLineComments(t *testing.T) {
	// make source
	var src string;
	for _, e := range segments {
		src += e.srcline
	}

	// verify scan
	var S Scanner;
	S.Init("TestLineComments", strings.Bytes(src), nil, 0);
	for _, s := range segments {
		pos, _, lit := S.Scan();
		checkPos(t, string(lit), pos, token.Position{s.filename, pos.Offset, s.line, pos.Column});
	}

	if S.ErrorCount != 0 {
		t.Errorf("found %d errors", S.ErrorCount)
	}
}


// Verify that initializing the same scanner more then once works correctly.
func TestInit(t *testing.T) {
	var s Scanner;

	// 1st init
	s.Init("", strings.Bytes("if true { }"), nil, 0);
	s.Scan();		// if
	s.Scan();		// true
	_, tok, _ := s.Scan();	// {
	if tok != token.LBRACE {
		t.Errorf("bad token: got %s, expected %s", tok.String(), token.LBRACE)
	}

	// 2nd init
	s.Init("", strings.Bytes("go true { ]"), nil, 0);
	_, tok, _ = s.Scan();	// go
	if tok != token.GO {
		t.Errorf("bad token: got %s, expected %s", tok.String(), token.GO)
	}

	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount)
	}
}


func TestIllegalChars(t *testing.T) {
	var s Scanner;

	const src = "*?*$*@*";
	s.Init("", strings.Bytes(src), &TestErrorHandler{t}, AllowIllegalChars);
	for offs, ch := range src {
		pos, tok, lit := s.Scan();
		if pos.Offset != offs {
			t.Errorf("bad position for %s: got %d, expected %d", string(lit), pos.Offset, offs)
		}
		if tok == token.ILLEGAL && string(lit) != string(ch) {
			t.Errorf("bad token: got %s, expected %s", string(lit), string(ch))
		}
	}

	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount)
	}
}


func TestStdErrorHander(t *testing.T) {
	const src = "@\n"	// illegal character, cause an error
		"@ @\n"	// two errors on the same line
		"//line File2:20\n"
		"@\n"	// different file, but same line
		"//line File2:1\n"
		"@ @\n"	// same file, decreasing line number
		"//line File1:1\n"
		"@ @ @";	// original file, line 1 again


	v := NewErrorVector();
	nerrors := Tokenize("File1", strings.Bytes(src), v, 0,
		func(pos token.Position, tok token.Token, litb []byte) bool {
			return tok != token.EOF
		});

	list := v.GetErrorList(Raw);
	if len(list) != 9 {
		t.Errorf("found %d raw errors, expected 9", len(list));
		PrintError(os.Stderr, list);
	}

	list = v.GetErrorList(Sorted);
	if len(list) != 9 {
		t.Errorf("found %d sorted errors, expected 9", len(list));
		PrintError(os.Stderr, list);
	}

	list = v.GetErrorList(NoMultiples);
	if len(list) != 4 {
		t.Errorf("found %d one-per-line errors, expected 4", len(list));
		PrintError(os.Stderr, list);
	}

	if v.ErrorCount() != nerrors {
		t.Errorf("found %d errors, expected %d", v.ErrorCount(), nerrors)
	}
}
