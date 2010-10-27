// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"
)


// A StringReader delivers its data one string segment at a time via Read.
type StringReader struct {
	data []string
	step int
}


func (r *StringReader) Read(p []byte) (n int, err os.Error) {
	if r.step < len(r.data) {
		s := r.data[r.step]
		n = copy(p, s)
		r.step++
	} else {
		err = os.EOF
	}
	return
}


func readRuneSegments(t *testing.T, segments []string) {
	got := ""
	want := strings.Join(segments, "")
	s := new(Scanner).Init(&StringReader{data: segments})
	for {
		ch := s.Next()
		if ch == EOF {
			break
		}
		got += string(ch)
	}
	if got != want {
		t.Errorf("segments=%v got=%s want=%s", segments, got, want)
	}
}


var segmentList = [][]string{
	{},
	{""},
	{"日", "本語"},
	{"\u65e5", "\u672c", "\u8a9e"},
	{"\U000065e5", " ", "\U0000672c", "\U00008a9e"},
	{"\xe6", "\x97\xa5\xe6", "\x9c\xac\xe8\xaa\x9e"},
	{"Hello", ", ", "World", "!"},
	{"Hello", ", ", "", "World", "!"},
}


func TestNext(t *testing.T) {
	for _, s := range segmentList {
		readRuneSegments(t, s)
	}
}


type token struct {
	tok  int
	text string
}

var f100 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

var tokenList = []token{
	{Comment, "// line comments\n"},
	{Comment, "//\n"},
	{Comment, "////\n"},
	{Comment, "// comment\n"},
	{Comment, "// /* comment */\n"},
	{Comment, "// // comment //\n"},
	{Comment, "//" + f100 + "\n"},

	{Comment, "// general comments\n"},
	{Comment, "/**/"},
	{Comment, "/***/"},
	{Comment, "/* comment */"},
	{Comment, "/* // comment */"},
	{Comment, "/* /* comment */"},
	{Comment, "/*\n comment\n*/"},
	{Comment, "/*" + f100 + "*/"},

	{Comment, "// identifiers\n"},
	{Ident, "a"},
	{Ident, "a0"},
	{Ident, "foobar"},
	{Ident, "abc123"},
	{Ident, "LGTM"},
	{Ident, "_"},
	{Ident, "_abc123"},
	{Ident, "abc123_"},
	{Ident, "_abc_123_"},
	{Ident, "_äöü"},
	{Ident, "_本"},
	// TODO for unknown reasons these fail when checking the literals
	/*
		token{Ident, "äöü"},
		token{Ident, "本"},
	*/
	{Ident, "a۰۱۸"},
	{Ident, "foo६४"},
	{Ident, "bar９８７６"},
	{Ident, f100},

	{Comment, "// decimal ints\n"},
	{Int, "0"},
	{Int, "1"},
	{Int, "9"},
	{Int, "42"},
	{Int, "1234567890"},

	{Comment, "// octal ints\n"},
	{Int, "00"},
	{Int, "01"},
	{Int, "07"},
	{Int, "042"},
	{Int, "01234567"},

	{Comment, "// hexadecimal ints\n"},
	{Int, "0x0"},
	{Int, "0x1"},
	{Int, "0xf"},
	{Int, "0x42"},
	{Int, "0x123456789abcDEF"},
	{Int, "0x" + f100},
	{Int, "0X0"},
	{Int, "0X1"},
	{Int, "0XF"},
	{Int, "0X42"},
	{Int, "0X123456789abcDEF"},
	{Int, "0X" + f100},

	{Comment, "// floats\n"},
	{Float, "0."},
	{Float, "1."},
	{Float, "42."},
	{Float, "01234567890."},
	{Float, ".0"},
	{Float, ".1"},
	{Float, ".42"},
	{Float, ".0123456789"},
	{Float, "0.0"},
	{Float, "1.0"},
	{Float, "42.0"},
	{Float, "01234567890.0"},
	{Float, "0e0"},
	{Float, "1e0"},
	{Float, "42e0"},
	{Float, "01234567890e0"},
	{Float, "0E0"},
	{Float, "1E0"},
	{Float, "42E0"},
	{Float, "01234567890E0"},
	{Float, "0e+10"},
	{Float, "1e-10"},
	{Float, "42e+10"},
	{Float, "01234567890e-10"},
	{Float, "0E+10"},
	{Float, "1E-10"},
	{Float, "42E+10"},
	{Float, "01234567890E-10"},

	{Comment, "// chars\n"},
	{Char, `' '`},
	{Char, `'a'`},
	{Char, `'本'`},
	{Char, `'\a'`},
	{Char, `'\b'`},
	{Char, `'\f'`},
	{Char, `'\n'`},
	{Char, `'\r'`},
	{Char, `'\t'`},
	{Char, `'\v'`},
	{Char, `'\''`},
	{Char, `'\000'`},
	{Char, `'\777'`},
	{Char, `'\x00'`},
	{Char, `'\xff'`},
	{Char, `'\u0000'`},
	{Char, `'\ufA16'`},
	{Char, `'\U00000000'`},
	{Char, `'\U0000ffAB'`},

	{Comment, "// strings\n"},
	{String, `" "`},
	{String, `"a"`},
	{String, `"本"`},
	{String, `"\a"`},
	{String, `"\b"`},
	{String, `"\f"`},
	{String, `"\n"`},
	{String, `"\r"`},
	{String, `"\t"`},
	{String, `"\v"`},
	{String, `"\""`},
	{String, `"\000"`},
	{String, `"\777"`},
	{String, `"\x00"`},
	{String, `"\xff"`},
	{String, `"\u0000"`},
	{String, `"\ufA16"`},
	{String, `"\U00000000"`},
	{String, `"\U0000ffAB"`},
	{String, `"` + f100 + `"`},

	{Comment, "// raw strings\n"},
	{String, "``"},
	{String, "`\\`"},
	{String, "`" + "\n\n/* foobar */\n\n" + "`"},
	{String, "`" + f100 + "`"},

	{Comment, "// individual characters\n"},
	// NUL character is not allowed
	{'\x01', "\x01"},
	{' ' - 1, string(' ' - 1)},
	{'+', "+"},
	{'/', "/"},
	{'.', "."},
	{'~', "~"},
	{'(', "("},
}


func makeSource(pattern string) *bytes.Buffer {
	var buf bytes.Buffer
	for _, k := range tokenList {
		fmt.Fprintf(&buf, pattern, k.text)
	}
	return &buf
}


func checkTok(t *testing.T, s *Scanner, line, got, want int, text string) {
	if got != want {
		t.Fatalf("tok = %s, want %s for %q", TokenString(got), TokenString(want), text)
	}
	if s.Line != line {
		t.Errorf("line = %d, want %d for %q", s.Line, line, text)
	}
	stext := s.TokenText()
	if stext != text {
		t.Errorf("text = %q, want %q", stext, text)
	} else {
		// check idempotency of TokenText() call
		stext = s.TokenText()
		if stext != text {
			t.Errorf("text = %q, want %q (idempotency check)", stext, text)
		}
	}
}


func countNewlines(s string) int {
	n := 0
	for _, ch := range s {
		if ch == '\n' {
			n++
		}
	}
	return n
}


func testScan(t *testing.T, mode uint) {
	s := new(Scanner).Init(makeSource(" \t%s\t\n\r"))
	s.Mode = mode
	tok := s.Scan()
	line := 1
	for _, k := range tokenList {
		if mode&SkipComments == 0 || k.tok != Comment {
			checkTok(t, s, line, tok, k.tok, k.text)
			tok = s.Scan()
		}
		line += countNewlines(k.text) + 1 // each token is on a new line
	}
	checkTok(t, s, line, tok, -1, "")
}


func TestScan(t *testing.T) {
	testScan(t, GoTokens)
	testScan(t, GoTokens&^SkipComments)
}


func TestPosition(t *testing.T) {
	src := makeSource("\t\t\t\t%s\n")
	s := new(Scanner).Init(src)
	s.Mode = GoTokens &^ SkipComments
	s.Scan()
	pos := Position{"", 4, 1, 5}
	for _, k := range tokenList {
		if s.Offset != pos.Offset {
			t.Errorf("offset = %d, want %d for %q", s.Offset, pos.Offset, k.text)
		}
		if s.Line != pos.Line {
			t.Errorf("line = %d, want %d for %q", s.Line, pos.Line, k.text)
		}
		if s.Column != pos.Column {
			t.Errorf("column = %d, want %d for %q", s.Column, pos.Column, k.text)
		}
		pos.Offset += 4 + len(k.text) + 1     // 4 tabs + token bytes + newline
		pos.Line += countNewlines(k.text) + 1 // each token is on a new line
		s.Scan()
	}
}


func TestScanZeroMode(t *testing.T) {
	src := makeSource("%s\n")
	str := src.String()
	s := new(Scanner).Init(src)
	s.Mode = 0       // don't recognize any token classes
	s.Whitespace = 0 // don't skip any whitespace
	tok := s.Scan()
	for i, ch := range str {
		if tok != ch {
			t.Fatalf("%d. tok = %s, want %s", i, TokenString(tok), TokenString(ch))
		}
		tok = s.Scan()
	}
	if tok != EOF {
		t.Fatalf("tok = %s, want EOF", TokenString(tok))
	}
}


func testScanSelectedMode(t *testing.T, mode uint, class int) {
	src := makeSource("%s\n")
	s := new(Scanner).Init(src)
	s.Mode = mode
	tok := s.Scan()
	for tok != EOF {
		if tok < 0 && tok != class {
			t.Fatalf("tok = %s, want %s", TokenString(tok), TokenString(class))
		}
		tok = s.Scan()
	}
}


func TestScanSelectedMask(t *testing.T) {
	testScanSelectedMode(t, 0, 0)
	testScanSelectedMode(t, ScanIdents, Ident)
	// Don't test ScanInts and ScanNumbers since some parts of
	// the floats in the source look like (illegal) octal ints
	// and ScanNumbers may return either Int or Float.
	testScanSelectedMode(t, ScanChars, Char)
	testScanSelectedMode(t, ScanStrings, String)
	testScanSelectedMode(t, SkipComments, 0)
	testScanSelectedMode(t, ScanComments, Comment)
}


func TestScanNext(t *testing.T) {
	s := new(Scanner).Init(bytes.NewBufferString("if a == bcd /* comment */ {\n\ta += c\n}"))
	checkTok(t, s, 1, s.Scan(), Ident, "if")
	checkTok(t, s, 1, s.Scan(), Ident, "a")
	checkTok(t, s, 1, s.Scan(), '=', "=")
	checkTok(t, s, 1, s.Next(), '=', "")
	checkTok(t, s, 1, s.Next(), ' ', "")
	checkTok(t, s, 1, s.Next(), 'b', "")
	checkTok(t, s, 1, s.Scan(), Ident, "cd")
	checkTok(t, s, 1, s.Scan(), '{', "{")
	checkTok(t, s, 2, s.Scan(), Ident, "a")
	checkTok(t, s, 2, s.Scan(), '+', "+")
	checkTok(t, s, 2, s.Next(), '=', "")
	checkTok(t, s, 2, s.Scan(), Ident, "c")
	checkTok(t, s, 3, s.Scan(), '}', "}")
	checkTok(t, s, 3, s.Scan(), -1, "")
}


func TestScanWhitespace(t *testing.T) {
	var buf bytes.Buffer
	var ws uint64
	// start at 1, NUL character is not allowed
	for ch := byte(1); ch < ' '; ch++ {
		buf.WriteByte(ch)
		ws |= 1 << ch
	}
	const orig = 'x'
	buf.WriteByte(orig)

	s := new(Scanner).Init(&buf)
	s.Mode = 0
	s.Whitespace = ws
	tok := s.Scan()
	if tok != orig {
		t.Errorf("tok = %s, want %s", TokenString(tok), TokenString(orig))
	}
}


func testError(t *testing.T, src, msg string, tok int) {
	s := new(Scanner).Init(bytes.NewBufferString(src))
	errorCalled := false
	s.Error = func(s *Scanner, m string) {
		if !errorCalled {
			// only look at first error
			if m != msg {
				t.Errorf("msg = %q, want %q for %q", m, msg, src)
			}
			errorCalled = true
		}
	}
	tk := s.Scan()
	if tk != tok {
		t.Errorf("tok = %s, want %s for %q", TokenString(tk), TokenString(tok), src)
	}
	if !errorCalled {
		t.Errorf("error handler not called for %q", src)
	}
	if s.ErrorCount == 0 {
		t.Errorf("count = %d, want > 0 for %q", s.ErrorCount, src)
	}
}


func TestError(t *testing.T) {
	testError(t, `01238`, "illegal octal number", Int)
	testError(t, `'\"'`, "illegal char escape", Char)
	testError(t, `'aa'`, "illegal char literal", Char)
	testError(t, `'`, "literal not terminated", Char)
	testError(t, `"\'"`, "illegal char escape", String)
	testError(t, `"abc`, "literal not terminated", String)
	testError(t, "`abc", "literal not terminated", String)
	testError(t, `//`, "comment not terminated", EOF)
	testError(t, `/*/`, "comment not terminated", EOF)
	testError(t, `"abc`+"\x00"+`def"`, "illegal character NUL", String)
	testError(t, `"abc`+"\xff"+`def"`, "illegal UTF-8 encoding", String)
}


func checkPos(t *testing.T, s *Scanner, offset, line, column, char int) {
	pos := s.Pos()
	if pos.Offset != offset {
		t.Errorf("offset = %d, want %d", pos.Offset, offset)
	}
	if pos.Line != line {
		t.Errorf("line = %d, want %d", pos.Line, line)
	}
	if pos.Column != column {
		t.Errorf("column = %d, want %d", pos.Column, column)
	}
	ch := s.Scan()
	if ch != char {
		t.Errorf("ch = %s, want %s", TokenString(ch), TokenString(char))
	}
}


func TestPos(t *testing.T) {
	s := new(Scanner).Init(bytes.NewBufferString("abc\n012\n\nx"))
	s.Mode = 0
	s.Whitespace = 0
	checkPos(t, s, 0, 1, 1, 'a')
	checkPos(t, s, 1, 1, 2, 'b')
	checkPos(t, s, 2, 1, 3, 'c')
	checkPos(t, s, 3, 2, 0, '\n')
	checkPos(t, s, 4, 2, 1, '0')
	checkPos(t, s, 5, 2, 2, '1')
	checkPos(t, s, 6, 2, 3, '2')
	checkPos(t, s, 7, 3, 0, '\n')
	checkPos(t, s, 8, 4, 0, '\n')
	checkPos(t, s, 9, 4, 1, 'x')
	checkPos(t, s, 9, 4, 1, EOF)
	checkPos(t, s, 9, 4, 1, EOF) // after EOF, position doesn't change
}
