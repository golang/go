// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"
	"unicode/utf8"
)

// A StringReader delivers its data one string segment at a time via Read.
type StringReader struct {
	data []string
	step int
}

func (r *StringReader) Read(p []byte) (n int, err error) {
	if r.step < len(r.data) {
		s := r.data[r.step]
		n = copy(p, s)
		r.step++
	} else {
		err = io.EOF
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
	tok  rune
	text string
}

var f100 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

var tokenList = []token{
	{Comment, "// line comments"},
	{Comment, "//"},
	{Comment, "////"},
	{Comment, "// comment"},
	{Comment, "// /* comment */"},
	{Comment, "// // comment //"},
	{Comment, "//" + f100},

	{Comment, "// general comments"},
	{Comment, "/**/"},
	{Comment, "/***/"},
	{Comment, "/* comment */"},
	{Comment, "/* // comment */"},
	{Comment, "/* /* comment */"},
	{Comment, "/*\n comment\n*/"},
	{Comment, "/*" + f100 + "*/"},

	{Comment, "// identifiers"},
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
	{Ident, "äöü"},
	{Ident, "本"},
	{Ident, "a۰۱۸"},
	{Ident, "foo६४"},
	{Ident, "bar９８７６"},
	{Ident, f100},

	{Comment, "// decimal ints"},
	{Int, "0"},
	{Int, "1"},
	{Int, "9"},
	{Int, "42"},
	{Int, "1234567890"},

	{Comment, "// octal ints"},
	{Int, "00"},
	{Int, "01"},
	{Int, "07"},
	{Int, "042"},
	{Int, "01234567"},

	{Comment, "// hexadecimal ints"},
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

	{Comment, "// floats"},
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

	{Comment, "// chars"},
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

	{Comment, "// strings"},
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

	{Comment, "// raw strings"},
	{RawString, "``"},
	{RawString, "`\\`"},
	{RawString, "`" + "\n\n/* foobar */\n\n" + "`"},
	{RawString, "`" + f100 + "`"},

	{Comment, "// individual characters"},
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

func checkTok(t *testing.T, s *Scanner, line int, got, want rune, text string) {
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

func checkTokErr(t *testing.T, s *Scanner, line int, want rune, text string) {
	prevCount := s.ErrorCount
	checkTok(t, s, line, s.Scan(), want, text)
	if s.ErrorCount != prevCount+1 {
		t.Fatalf("want error for %q", text)
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
	s := new(Scanner).Init(makeSource(" \t%s\n"))
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
	checkTok(t, s, line, tok, EOF, "")
}

func TestScan(t *testing.T) {
	testScan(t, GoTokens)
	testScan(t, GoTokens&^SkipComments)
}

func TestInvalidExponent(t *testing.T) {
	const src = "1.5e 1.5E 1e+ 1e- 1.5z"
	s := new(Scanner).Init(strings.NewReader(src))
	s.Error = func(s *Scanner, msg string) {
		const want = "exponent has no digits"
		if msg != want {
			t.Errorf("%s: got error %q; want %q", s.TokenText(), msg, want)
		}
	}
	checkTokErr(t, s, 1, Float, "1.5e")
	checkTokErr(t, s, 1, Float, "1.5E")
	checkTokErr(t, s, 1, Float, "1e+")
	checkTokErr(t, s, 1, Float, "1e-")
	checkTok(t, s, 1, s.Scan(), Float, "1.5")
	checkTok(t, s, 1, s.Scan(), Ident, "z")
	checkTok(t, s, 1, s.Scan(), EOF, "")
	if s.ErrorCount != 4 {
		t.Errorf("%d errors, want 4", s.ErrorCount)
	}
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
	// make sure there were no token-internal errors reported by scanner
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
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
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}
}

func testScanSelectedMode(t *testing.T, mode uint, class rune) {
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
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}
}

func TestScanSelectedMask(t *testing.T) {
	testScanSelectedMode(t, 0, 0)
	testScanSelectedMode(t, ScanIdents, Ident)
	// Don't test ScanInts and ScanNumbers since some parts of
	// the floats in the source look like (invalid) octal ints
	// and ScanNumbers may return either Int or Float.
	testScanSelectedMode(t, ScanChars, Char)
	testScanSelectedMode(t, ScanStrings, String)
	testScanSelectedMode(t, SkipComments, 0)
	testScanSelectedMode(t, ScanComments, Comment)
}

func TestScanCustomIdent(t *testing.T) {
	const src = "faab12345 a12b123 a12 3b"
	s := new(Scanner).Init(strings.NewReader(src))
	// ident = ( 'a' | 'b' ) { digit } .
	// digit = '0' .. '3' .
	// with a maximum length of 4
	s.IsIdentRune = func(ch rune, i int) bool {
		return i == 0 && (ch == 'a' || ch == 'b') || 0 < i && i < 4 && '0' <= ch && ch <= '3'
	}
	checkTok(t, s, 1, s.Scan(), 'f', "f")
	checkTok(t, s, 1, s.Scan(), Ident, "a")
	checkTok(t, s, 1, s.Scan(), Ident, "a")
	checkTok(t, s, 1, s.Scan(), Ident, "b123")
	checkTok(t, s, 1, s.Scan(), Int, "45")
	checkTok(t, s, 1, s.Scan(), Ident, "a12")
	checkTok(t, s, 1, s.Scan(), Ident, "b123")
	checkTok(t, s, 1, s.Scan(), Ident, "a12")
	checkTok(t, s, 1, s.Scan(), Int, "3")
	checkTok(t, s, 1, s.Scan(), Ident, "b")
	checkTok(t, s, 1, s.Scan(), EOF, "")
}

func TestScanNext(t *testing.T) {
	const BOM = '\uFEFF'
	BOMs := string(BOM)
	s := new(Scanner).Init(strings.NewReader(BOMs + "if a == bcd /* com" + BOMs + "ment */ {\n\ta += c\n}" + BOMs + "// line comment ending in eof"))
	checkTok(t, s, 1, s.Scan(), Ident, "if") // the first BOM is ignored
	checkTok(t, s, 1, s.Scan(), Ident, "a")
	checkTok(t, s, 1, s.Scan(), '=', "=")
	checkTok(t, s, 0, s.Next(), '=', "")
	checkTok(t, s, 0, s.Next(), ' ', "")
	checkTok(t, s, 0, s.Next(), 'b', "")
	checkTok(t, s, 1, s.Scan(), Ident, "cd")
	checkTok(t, s, 1, s.Scan(), '{', "{")
	checkTok(t, s, 2, s.Scan(), Ident, "a")
	checkTok(t, s, 2, s.Scan(), '+', "+")
	checkTok(t, s, 0, s.Next(), '=', "")
	checkTok(t, s, 2, s.Scan(), Ident, "c")
	checkTok(t, s, 3, s.Scan(), '}', "}")
	checkTok(t, s, 3, s.Scan(), BOM, BOMs)
	checkTok(t, s, 3, s.Scan(), -1, "")
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}
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

func testError(t *testing.T, src, pos, msg string, tok rune) {
	s := new(Scanner).Init(strings.NewReader(src))
	errorCalled := false
	s.Error = func(s *Scanner, m string) {
		if !errorCalled {
			// only look at first error
			if p := s.Pos().String(); p != pos {
				t.Errorf("pos = %q, want %q for %q", p, pos, src)
			}
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
	testError(t, "\x00", "<input>:1:1", "invalid character NUL", 0)
	testError(t, "\x80", "<input>:1:1", "invalid UTF-8 encoding", utf8.RuneError)
	testError(t, "\xff", "<input>:1:1", "invalid UTF-8 encoding", utf8.RuneError)

	testError(t, "a\x00", "<input>:1:2", "invalid character NUL", Ident)
	testError(t, "ab\x80", "<input>:1:3", "invalid UTF-8 encoding", Ident)
	testError(t, "abc\xff", "<input>:1:4", "invalid UTF-8 encoding", Ident)

	testError(t, `"a`+"\x00", "<input>:1:3", "invalid character NUL", String)
	testError(t, `"ab`+"\x80", "<input>:1:4", "invalid UTF-8 encoding", String)
	testError(t, `"abc`+"\xff", "<input>:1:5", "invalid UTF-8 encoding", String)

	testError(t, "`a"+"\x00", "<input>:1:3", "invalid character NUL", RawString)
	testError(t, "`ab"+"\x80", "<input>:1:4", "invalid UTF-8 encoding", RawString)
	testError(t, "`abc"+"\xff", "<input>:1:5", "invalid UTF-8 encoding", RawString)

	testError(t, `'\"'`, "<input>:1:3", "invalid char escape", Char)
	testError(t, `"\'"`, "<input>:1:3", "invalid char escape", String)

	testError(t, `01238`, "<input>:1:6", "invalid digit '8' in octal literal", Int)
	testError(t, `01238123`, "<input>:1:9", "invalid digit '8' in octal literal", Int)
	testError(t, `0x`, "<input>:1:3", "hexadecimal literal has no digits", Int)
	testError(t, `0xg`, "<input>:1:3", "hexadecimal literal has no digits", Int)
	testError(t, `'aa'`, "<input>:1:4", "invalid char literal", Char)
	testError(t, `1.5e`, "<input>:1:5", "exponent has no digits", Float)
	testError(t, `1.5E`, "<input>:1:5", "exponent has no digits", Float)
	testError(t, `1.5e+`, "<input>:1:6", "exponent has no digits", Float)
	testError(t, `1.5e-`, "<input>:1:6", "exponent has no digits", Float)

	testError(t, `'`, "<input>:1:2", "literal not terminated", Char)
	testError(t, `'`+"\n", "<input>:1:2", "literal not terminated", Char)
	testError(t, `"abc`, "<input>:1:5", "literal not terminated", String)
	testError(t, `"abc`+"\n", "<input>:1:5", "literal not terminated", String)
	testError(t, "`abc\n", "<input>:2:1", "literal not terminated", RawString)
	testError(t, `/*/`, "<input>:1:4", "comment not terminated", EOF)
}

// An errReader returns (0, err) where err is not io.EOF.
type errReader struct{}

func (errReader) Read(b []byte) (int, error) {
	return 0, io.ErrNoProgress // some error that is not io.EOF
}

func TestIOError(t *testing.T) {
	s := new(Scanner).Init(errReader{})
	errorCalled := false
	s.Error = func(s *Scanner, msg string) {
		if !errorCalled {
			if want := io.ErrNoProgress.Error(); msg != want {
				t.Errorf("msg = %q, want %q", msg, want)
			}
			errorCalled = true
		}
	}
	tok := s.Scan()
	if tok != EOF {
		t.Errorf("tok = %s, want EOF", TokenString(tok))
	}
	if !errorCalled {
		t.Errorf("error handler not called")
	}
}

func checkPos(t *testing.T, got, want Position) {
	if got.Offset != want.Offset || got.Line != want.Line || got.Column != want.Column {
		t.Errorf("got offset, line, column = %d, %d, %d; want %d, %d, %d",
			got.Offset, got.Line, got.Column, want.Offset, want.Line, want.Column)
	}
}

func checkNextPos(t *testing.T, s *Scanner, offset, line, column int, char rune) {
	if ch := s.Next(); ch != char {
		t.Errorf("ch = %s, want %s", TokenString(ch), TokenString(char))
	}
	want := Position{Offset: offset, Line: line, Column: column}
	checkPos(t, s.Pos(), want)
}

func checkScanPos(t *testing.T, s *Scanner, offset, line, column int, char rune) {
	want := Position{Offset: offset, Line: line, Column: column}
	checkPos(t, s.Pos(), want)
	if ch := s.Scan(); ch != char {
		t.Errorf("ch = %s, want %s", TokenString(ch), TokenString(char))
		if string(ch) != s.TokenText() {
			t.Errorf("tok = %q, want %q", s.TokenText(), string(ch))
		}
	}
	checkPos(t, s.Position, want)
}

func TestPos(t *testing.T) {
	// corner case: empty source
	s := new(Scanner).Init(strings.NewReader(""))
	checkPos(t, s.Pos(), Position{Offset: 0, Line: 1, Column: 1})
	s.Peek() // peek doesn't affect the position
	checkPos(t, s.Pos(), Position{Offset: 0, Line: 1, Column: 1})

	// corner case: source with only a newline
	s = new(Scanner).Init(strings.NewReader("\n"))
	checkPos(t, s.Pos(), Position{Offset: 0, Line: 1, Column: 1})
	checkNextPos(t, s, 1, 2, 1, '\n')
	// after EOF position doesn't change
	for i := 10; i > 0; i-- {
		checkScanPos(t, s, 1, 2, 1, EOF)
	}
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}

	// corner case: source with only a single character
	s = new(Scanner).Init(strings.NewReader("本"))
	checkPos(t, s.Pos(), Position{Offset: 0, Line: 1, Column: 1})
	checkNextPos(t, s, 3, 1, 2, '本')
	// after EOF position doesn't change
	for i := 10; i > 0; i-- {
		checkScanPos(t, s, 3, 1, 2, EOF)
	}
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}

	// positions after calling Next
	s = new(Scanner).Init(strings.NewReader("  foo६४  \n\n本語\n"))
	checkNextPos(t, s, 1, 1, 2, ' ')
	s.Peek() // peek doesn't affect the position
	checkNextPos(t, s, 2, 1, 3, ' ')
	checkNextPos(t, s, 3, 1, 4, 'f')
	checkNextPos(t, s, 4, 1, 5, 'o')
	checkNextPos(t, s, 5, 1, 6, 'o')
	checkNextPos(t, s, 8, 1, 7, '६')
	checkNextPos(t, s, 11, 1, 8, '४')
	checkNextPos(t, s, 12, 1, 9, ' ')
	checkNextPos(t, s, 13, 1, 10, ' ')
	checkNextPos(t, s, 14, 2, 1, '\n')
	checkNextPos(t, s, 15, 3, 1, '\n')
	checkNextPos(t, s, 18, 3, 2, '本')
	checkNextPos(t, s, 21, 3, 3, '語')
	checkNextPos(t, s, 22, 4, 1, '\n')
	// after EOF position doesn't change
	for i := 10; i > 0; i-- {
		checkScanPos(t, s, 22, 4, 1, EOF)
	}
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}

	// positions after calling Scan
	s = new(Scanner).Init(strings.NewReader("abc\n本語\n\nx"))
	s.Mode = 0
	s.Whitespace = 0
	checkScanPos(t, s, 0, 1, 1, 'a')
	s.Peek() // peek doesn't affect the position
	checkScanPos(t, s, 1, 1, 2, 'b')
	checkScanPos(t, s, 2, 1, 3, 'c')
	checkScanPos(t, s, 3, 1, 4, '\n')
	checkScanPos(t, s, 4, 2, 1, '本')
	checkScanPos(t, s, 7, 2, 2, '語')
	checkScanPos(t, s, 10, 2, 3, '\n')
	checkScanPos(t, s, 11, 3, 1, '\n')
	checkScanPos(t, s, 12, 4, 1, 'x')
	// after EOF position doesn't change
	for i := 10; i > 0; i-- {
		checkScanPos(t, s, 13, 4, 2, EOF)
	}
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}
}

type countReader int

func (r *countReader) Read([]byte) (int, error) {
	*r++
	return 0, io.EOF
}

func TestNextEOFHandling(t *testing.T) {
	var r countReader

	// corner case: empty source
	s := new(Scanner).Init(&r)

	tok := s.Next()
	if tok != EOF {
		t.Error("1) EOF not reported")
	}

	tok = s.Peek()
	if tok != EOF {
		t.Error("2) EOF not reported")
	}

	if r != 1 {
		t.Errorf("scanner called Read %d times, not once", r)
	}
}

func TestScanEOFHandling(t *testing.T) {
	var r countReader

	// corner case: empty source
	s := new(Scanner).Init(&r)

	tok := s.Scan()
	if tok != EOF {
		t.Error("1) EOF not reported")
	}

	tok = s.Peek()
	if tok != EOF {
		t.Error("2) EOF not reported")
	}

	if r != 1 {
		t.Errorf("scanner called Read %d times, not once", r)
	}
}

func TestIssue29723(t *testing.T) {
	s := new(Scanner).Init(strings.NewReader(`x "`))
	s.Error = func(s *Scanner, _ string) {
		got := s.TokenText() // this call shouldn't panic
		const want = `"`
		if got != want {
			t.Errorf("got %q; want %q", got, want)
		}
	}
	for r := s.Scan(); r != EOF; r = s.Scan() {
	}
}

func TestNumbers(t *testing.T) {
	for _, test := range []struct {
		tok              rune
		src, tokens, err string
	}{
		// binaries
		{Int, "0b0", "0b0", ""},
		{Int, "0b1010", "0b1010", ""},
		{Int, "0B1110", "0B1110", ""},

		{Int, "0b", "0b", "binary literal has no digits"},
		{Int, "0b0190", "0b0190", "invalid digit '9' in binary literal"},
		{Int, "0b01a0", "0b01 a0", ""}, // only accept 0-9

		// binary floats (invalid)
		{Float, "0b.", "0b.", "invalid radix point in binary literal"},
		{Float, "0b.1", "0b.1", "invalid radix point in binary literal"},
		{Float, "0b1.0", "0b1.0", "invalid radix point in binary literal"},
		{Float, "0b1e10", "0b1e10", "'e' exponent requires decimal mantissa"},
		{Float, "0b1P-1", "0b1P-1", "'P' exponent requires hexadecimal mantissa"},

		// octals
		{Int, "0o0", "0o0", ""},
		{Int, "0o1234", "0o1234", ""},
		{Int, "0O1234", "0O1234", ""},

		{Int, "0o", "0o", "octal literal has no digits"},
		{Int, "0o8123", "0o8123", "invalid digit '8' in octal literal"},
		{Int, "0o1293", "0o1293", "invalid digit '9' in octal literal"},
		{Int, "0o12a3", "0o12 a3", ""}, // only accept 0-9

		// octal floats (invalid)
		{Float, "0o.", "0o.", "invalid radix point in octal literal"},
		{Float, "0o.2", "0o.2", "invalid radix point in octal literal"},
		{Float, "0o1.2", "0o1.2", "invalid radix point in octal literal"},
		{Float, "0o1E+2", "0o1E+2", "'E' exponent requires decimal mantissa"},
		{Float, "0o1p10", "0o1p10", "'p' exponent requires hexadecimal mantissa"},

		// 0-octals
		{Int, "0", "0", ""},
		{Int, "0123", "0123", ""},

		{Int, "08123", "08123", "invalid digit '8' in octal literal"},
		{Int, "01293", "01293", "invalid digit '9' in octal literal"},
		{Int, "0F.", "0 F .", ""}, // only accept 0-9
		{Int, "0123F.", "0123 F .", ""},
		{Int, "0123456x", "0123456 x", ""},

		// decimals
		{Int, "1", "1", ""},
		{Int, "1234", "1234", ""},

		{Int, "1f", "1 f", ""}, // only accept 0-9

		// decimal floats
		{Float, "0.", "0.", ""},
		{Float, "123.", "123.", ""},
		{Float, "0123.", "0123.", ""},

		{Float, ".0", ".0", ""},
		{Float, ".123", ".123", ""},
		{Float, ".0123", ".0123", ""},

		{Float, "0.0", "0.0", ""},
		{Float, "123.123", "123.123", ""},
		{Float, "0123.0123", "0123.0123", ""},

		{Float, "0e0", "0e0", ""},
		{Float, "123e+0", "123e+0", ""},
		{Float, "0123E-1", "0123E-1", ""},

		{Float, "0.e+1", "0.e+1", ""},
		{Float, "123.E-10", "123.E-10", ""},
		{Float, "0123.e123", "0123.e123", ""},

		{Float, ".0e-1", ".0e-1", ""},
		{Float, ".123E+10", ".123E+10", ""},
		{Float, ".0123E123", ".0123E123", ""},

		{Float, "0.0e1", "0.0e1", ""},
		{Float, "123.123E-10", "123.123E-10", ""},
		{Float, "0123.0123e+456", "0123.0123e+456", ""},

		{Float, "0e", "0e", "exponent has no digits"},
		{Float, "0E+", "0E+", "exponent has no digits"},
		{Float, "1e+f", "1e+ f", "exponent has no digits"},
		{Float, "0p0", "0p0", "'p' exponent requires hexadecimal mantissa"},
		{Float, "1.0P-1", "1.0P-1", "'P' exponent requires hexadecimal mantissa"},

		// hexadecimals
		{Int, "0x0", "0x0", ""},
		{Int, "0x1234", "0x1234", ""},
		{Int, "0xcafef00d", "0xcafef00d", ""},
		{Int, "0XCAFEF00D", "0XCAFEF00D", ""},

		{Int, "0x", "0x", "hexadecimal literal has no digits"},
		{Int, "0x1g", "0x1 g", ""},

		// hexadecimal floats
		{Float, "0x0p0", "0x0p0", ""},
		{Float, "0x12efp-123", "0x12efp-123", ""},
		{Float, "0xABCD.p+0", "0xABCD.p+0", ""},
		{Float, "0x.0189P-0", "0x.0189P-0", ""},
		{Float, "0x1.ffffp+1023", "0x1.ffffp+1023", ""},

		{Float, "0x.", "0x.", "hexadecimal literal has no digits"},
		{Float, "0x0.", "0x0.", "hexadecimal mantissa requires a 'p' exponent"},
		{Float, "0x.0", "0x.0", "hexadecimal mantissa requires a 'p' exponent"},
		{Float, "0x1.1", "0x1.1", "hexadecimal mantissa requires a 'p' exponent"},
		{Float, "0x1.1e0", "0x1.1e0", "hexadecimal mantissa requires a 'p' exponent"},
		{Float, "0x1.2gp1a", "0x1.2 gp1a", "hexadecimal mantissa requires a 'p' exponent"},
		{Float, "0x0p", "0x0p", "exponent has no digits"},
		{Float, "0xeP-", "0xeP-", "exponent has no digits"},
		{Float, "0x1234PAB", "0x1234P AB", "exponent has no digits"},
		{Float, "0x1.2p1a", "0x1.2p1 a", ""},

		// separators
		{Int, "0b_1000_0001", "0b_1000_0001", ""},
		{Int, "0o_600", "0o_600", ""},
		{Int, "0_466", "0_466", ""},
		{Int, "1_000", "1_000", ""},
		{Float, "1_000.000_1", "1_000.000_1", ""},
		{Int, "0x_f00d", "0x_f00d", ""},
		{Float, "0x_f00d.0p1_2", "0x_f00d.0p1_2", ""},

		{Int, "0b__1000", "0b__1000", "'_' must separate successive digits"},
		{Int, "0o60___0", "0o60___0", "'_' must separate successive digits"},
		{Int, "0466_", "0466_", "'_' must separate successive digits"},
		{Float, "1_.", "1_.", "'_' must separate successive digits"},
		{Float, "0._1", "0._1", "'_' must separate successive digits"},
		{Float, "2.7_e0", "2.7_e0", "'_' must separate successive digits"},
		{Int, "0x___0", "0x___0", "'_' must separate successive digits"},
		{Float, "0x1.0_p0", "0x1.0_p0", "'_' must separate successive digits"},
	} {
		s := new(Scanner).Init(strings.NewReader(test.src))
		var err string
		s.Error = func(s *Scanner, msg string) {
			if err == "" {
				err = msg
			}
		}

		for i, want := range strings.Split(test.tokens, " ") {
			err = ""
			tok := s.Scan()
			lit := s.TokenText()
			if i == 0 {
				if tok != test.tok {
					t.Errorf("%q: got token %s; want %s", test.src, TokenString(tok), TokenString(test.tok))
				}
				if err != test.err {
					t.Errorf("%q: got error %q; want %q", test.src, err, test.err)
				}
			}
			if lit != want {
				t.Errorf("%q: got literal %q (%s); want %s", test.src, lit, TokenString(tok), want)
			}
		}

		// make sure we read all
		if tok := s.Scan(); tok != EOF {
			t.Errorf("%q: got %s; want EOF", test.src, TokenString(tok))
		}
	}
}

func TestIssue30320(t *testing.T) {
	for _, test := range []struct {
		in, want string
		mode     uint
	}{
		{"foo01.bar31.xx-0-1-1-0", "01 31 0 1 1 0", ScanInts},
		{"foo0/12/0/5.67", "0 12 0 5 67", ScanInts},
		{"xxx1e0yyy", "1 0", ScanInts},
		{"1_2", "1_2", ScanInts},
		{"xxx1.0yyy2e3ee", "1 0 2 3", ScanInts},
		{"xxx1.0yyy2e3ee", "1.0 2e3", ScanFloats},
	} {
		got := extractInts(test.in, test.mode)
		if got != test.want {
			t.Errorf("%q: got %q; want %q", test.in, got, test.want)
		}
	}
}

func extractInts(t string, mode uint) (res string) {
	var s Scanner
	s.Init(strings.NewReader(t))
	s.Mode = mode
	for {
		switch tok := s.Scan(); tok {
		case Int, Float:
			if len(res) > 0 {
				res += " "
			}
			res += s.TokenText()
		case EOF:
			return
		}
	}
}
