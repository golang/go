// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

var fset = token.NewFileSet()

const /* class */ (
	special = iota
	literal
	operator
	keyword
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
	return special
}

type elt struct {
	tok   token.Token
	lit   string
	class int
}

var tokens = [...]elt{
	// Special tokens
	{token.COMMENT, "/* a comment */", special},
	{token.COMMENT, "// a comment \n", special},
	{token.COMMENT, "/*\r*/", special},
	{token.COMMENT, "/**\r/*/", special}, // issue 11151
	{token.COMMENT, "/**\r\r/*/", special},
	{token.COMMENT, "//\r\n", special},

	// Identifiers and basic type literals
	{token.IDENT, "foobar", literal},
	{token.IDENT, "a۰۱۸", literal},
	{token.IDENT, "foo६४", literal},
	{token.IDENT, "bar９８７６", literal},
	{token.IDENT, "ŝ", literal},    // was bug (issue 4000)
	{token.IDENT, "ŝfoo", literal}, // was bug (issue 4000)
	{token.INT, "0", literal},
	{token.INT, "1", literal},
	{token.INT, "123456789012345678890", literal},
	{token.INT, "01234567", literal},
	{token.INT, "0xcafebabe", literal},
	{token.FLOAT, "0.", literal},
	{token.FLOAT, ".0", literal},
	{token.FLOAT, "3.14159265", literal},
	{token.FLOAT, "1e0", literal},
	{token.FLOAT, "1e+100", literal},
	{token.FLOAT, "1e-100", literal},
	{token.FLOAT, "2.71828e-1000", literal},
	{token.IMAG, "0i", literal},
	{token.IMAG, "1i", literal},
	{token.IMAG, "012345678901234567889i", literal},
	{token.IMAG, "123456789012345678890i", literal},
	{token.IMAG, "0.i", literal},
	{token.IMAG, ".0i", literal},
	{token.IMAG, "3.14159265i", literal},
	{token.IMAG, "1e0i", literal},
	{token.IMAG, "1e+100i", literal},
	{token.IMAG, "1e-100i", literal},
	{token.IMAG, "2.71828e-1000i", literal},
	{token.CHAR, "'a'", literal},
	{token.CHAR, "'\\000'", literal},
	{token.CHAR, "'\\xFF'", literal},
	{token.CHAR, "'\\uff16'", literal},
	{token.CHAR, "'\\U0000ff16'", literal},
	{token.STRING, "`foobar`", literal},
	{token.STRING, "`" + `foo
	                        bar` +
		"`",
		literal,
	},
	{token.STRING, "`\r`", literal},
	{token.STRING, "`foo\r\nbar`", literal},

	// Operators and delimiters
	{token.ADD, "+", operator},
	{token.SUB, "-", operator},
	{token.MUL, "*", operator},
	{token.QUO, "/", operator},
	{token.REM, "%", operator},

	{token.AND, "&", operator},
	{token.OR, "|", operator},
	{token.XOR, "^", operator},
	{token.SHL, "<<", operator},
	{token.SHR, ">>", operator},
	{token.AND_NOT, "&^", operator},

	{token.ADD_ASSIGN, "+=", operator},
	{token.SUB_ASSIGN, "-=", operator},
	{token.MUL_ASSIGN, "*=", operator},
	{token.QUO_ASSIGN, "/=", operator},
	{token.REM_ASSIGN, "%=", operator},

	{token.AND_ASSIGN, "&=", operator},
	{token.OR_ASSIGN, "|=", operator},
	{token.XOR_ASSIGN, "^=", operator},
	{token.SHL_ASSIGN, "<<=", operator},
	{token.SHR_ASSIGN, ">>=", operator},
	{token.AND_NOT_ASSIGN, "&^=", operator},

	{token.LAND, "&&", operator},
	{token.LOR, "||", operator},
	{token.ARROW, "<-", operator},
	{token.INC, "++", operator},
	{token.DEC, "--", operator},

	{token.EQL, "==", operator},
	{token.LSS, "<", operator},
	{token.GTR, ">", operator},
	{token.ASSIGN, "=", operator},
	{token.NOT, "!", operator},

	{token.NEQ, "!=", operator},
	{token.LEQ, "<=", operator},
	{token.GEQ, ">=", operator},
	{token.DEFINE, ":=", operator},
	{token.ELLIPSIS, "...", operator},

	{token.LPAREN, "(", operator},
	{token.LBRACK, "[", operator},
	{token.LBRACE, "{", operator},
	{token.COMMA, ",", operator},
	{token.PERIOD, ".", operator},

	{token.RPAREN, ")", operator},
	{token.RBRACK, "]", operator},
	{token.RBRACE, "}", operator},
	{token.SEMICOLON, ";", operator},
	{token.COLON, ":", operator},

	// Keywords
	{token.BREAK, "break", keyword},
	{token.CASE, "case", keyword},
	{token.CHAN, "chan", keyword},
	{token.CONST, "const", keyword},
	{token.CONTINUE, "continue", keyword},

	{token.DEFAULT, "default", keyword},
	{token.DEFER, "defer", keyword},
	{token.ELSE, "else", keyword},
	{token.FALLTHROUGH, "fallthrough", keyword},
	{token.FOR, "for", keyword},

	{token.FUNC, "func", keyword},
	{token.GO, "go", keyword},
	{token.GOTO, "goto", keyword},
	{token.IF, "if", keyword},
	{token.IMPORT, "import", keyword},

	{token.INTERFACE, "interface", keyword},
	{token.MAP, "map", keyword},
	{token.PACKAGE, "package", keyword},
	{token.RANGE, "range", keyword},
	{token.RETURN, "return", keyword},

	{token.SELECT, "select", keyword},
	{token.STRUCT, "struct", keyword},
	{token.SWITCH, "switch", keyword},
	{token.TYPE, "type", keyword},
	{token.VAR, "var", keyword},
}

const whitespace = "  \t  \n\n\n" // to separate tokens

var source = func() []byte {
	var src []byte
	for _, t := range tokens {
		src = append(src, t.lit...)
		src = append(src, whitespace...)
	}
	return src
}()

func newlineCount(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			n++
		}
	}
	return n
}

func checkPos(t *testing.T, lit string, p token.Pos, expected token.Position) {
	pos := fset.Position(p)
	// Check cleaned filenames so that we don't have to worry about
	// different os.PathSeparator values.
	if pos.Filename != expected.Filename && filepath.Clean(pos.Filename) != filepath.Clean(expected.Filename) {
		t.Errorf("bad filename for %q: got %s, expected %s", lit, pos.Filename, expected.Filename)
	}
	if pos.Offset != expected.Offset {
		t.Errorf("bad position for %q: got %d, expected %d", lit, pos.Offset, expected.Offset)
	}
	if pos.Line != expected.Line {
		t.Errorf("bad line for %q: got %d, expected %d", lit, pos.Line, expected.Line)
	}
	if pos.Column != expected.Column {
		t.Errorf("bad column for %q: got %d, expected %d", lit, pos.Column, expected.Column)
	}
}

// Verify that calling Scan() provides the correct results.
func TestScan(t *testing.T) {
	whitespace_linecount := newlineCount(whitespace)

	// error handler
	eh := func(_ token.Position, msg string) {
		t.Errorf("error handler called (msg = %s)", msg)
	}

	// verify scan
	var s Scanner
	s.Init(fset.AddFile("", fset.Base(), len(source)), source, eh, ScanComments|dontInsertSemis)

	// set up expected position
	epos := token.Position{
		Filename: "",
		Offset:   0,
		Line:     1,
		Column:   1,
	}

	index := 0
	for {
		pos, tok, lit := s.Scan()

		// check position
		if tok == token.EOF {
			// correction for EOF
			epos.Line = newlineCount(string(source))
			epos.Column = 2
		}
		checkPos(t, lit, pos, epos)

		// check token
		e := elt{token.EOF, "", special}
		if index < len(tokens) {
			e = tokens[index]
			index++
		}
		if tok != e.tok {
			t.Errorf("bad token for %q: got %s, expected %s", lit, tok, e.tok)
		}

		// check token class
		if tokenclass(tok) != e.class {
			t.Errorf("bad class for %q: got %d, expected %d", lit, tokenclass(tok), e.class)
		}

		// check literal
		elit := ""
		switch e.tok {
		case token.COMMENT:
			// no CRs in comments
			elit = string(stripCR([]byte(e.lit), e.lit[1] == '*'))
			//-style comment literal doesn't contain newline
			if elit[1] == '/' {
				elit = elit[0 : len(elit)-1]
			}
		case token.IDENT:
			elit = e.lit
		case token.SEMICOLON:
			elit = ";"
		default:
			if e.tok.IsLiteral() {
				// no CRs in raw string literals
				elit = e.lit
				if elit[0] == '`' {
					elit = string(stripCR([]byte(elit), false))
				}
			} else if e.tok.IsKeyword() {
				elit = e.lit
			}
		}
		if lit != elit {
			t.Errorf("bad literal for %q: got %q, expected %q", lit, lit, elit)
		}

		if tok == token.EOF {
			break
		}

		// update position
		epos.Offset += len(e.lit) + len(whitespace)
		epos.Line += newlineCount(e.lit) + whitespace_linecount

	}

	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount)
	}
}

func TestStripCR(t *testing.T) {
	for _, test := range []struct{ have, want string }{
		{"//\n", "//\n"},
		{"//\r\n", "//\n"},
		{"//\r\r\r\n", "//\n"},
		{"//\r*\r/\r\n", "//*/\n"},
		{"/**/", "/**/"},
		{"/*\r/*/", "/*/*/"},
		{"/*\r*/", "/**/"},
		{"/**\r/*/", "/**\r/*/"},
		{"/*\r/\r*\r/*/", "/*/*\r/*/"},
		{"/*\r\r\r\r*/", "/**/"},
	} {
		got := string(stripCR([]byte(test.have), len(test.have) >= 2 && test.have[1] == '*'))
		if got != test.want {
			t.Errorf("stripCR(%q) = %q; want %q", test.have, got, test.want)
		}
	}
}

func checkSemi(t *testing.T, line string, mode Mode) {
	var S Scanner
	file := fset.AddFile("TestSemis", fset.Base(), len(line))
	S.Init(file, []byte(line), nil, mode)
	pos, tok, lit := S.Scan()
	for tok != token.EOF {
		if tok == token.ILLEGAL {
			// the illegal token literal indicates what
			// kind of semicolon literal to expect
			semiLit := "\n"
			if lit[0] == '#' {
				semiLit = ";"
			}
			// next token must be a semicolon
			semiPos := file.Position(pos)
			semiPos.Offset++
			semiPos.Column++
			pos, tok, lit = S.Scan()
			if tok == token.SEMICOLON {
				if lit != semiLit {
					t.Errorf(`bad literal for %q: got %q, expected %q`, line, lit, semiLit)
				}
				checkPos(t, line, pos, semiPos)
			} else {
				t.Errorf("bad token for %q: got %s, expected ;", line, tok)
			}
		} else if tok == token.SEMICOLON {
			t.Errorf("bad token for %q: got ;, expected no ;", line)
		}
		pos, tok, lit = S.Scan()
	}
}

var lines = []string{
	// # indicates a semicolon present in the source
	// $ indicates an automatically inserted semicolon
	"",
	"\ufeff#;", // first BOM is ignored
	"#;",
	"foo$\n",
	"123$\n",
	"1.2$\n",
	"'x'$\n",
	`"x"` + "$\n",
	"`x`$\n",

	"+\n",
	"-\n",
	"*\n",
	"/\n",
	"%\n",

	"&\n",
	"|\n",
	"^\n",
	"<<\n",
	">>\n",
	"&^\n",

	"+=\n",
	"-=\n",
	"*=\n",
	"/=\n",
	"%=\n",

	"&=\n",
	"|=\n",
	"^=\n",
	"<<=\n",
	">>=\n",
	"&^=\n",

	"&&\n",
	"||\n",
	"<-\n",
	"++$\n",
	"--$\n",

	"==\n",
	"<\n",
	">\n",
	"=\n",
	"!\n",

	"!=\n",
	"<=\n",
	">=\n",
	":=\n",
	"...\n",

	"(\n",
	"[\n",
	"{\n",
	",\n",
	".\n",

	")$\n",
	"]$\n",
	"}$\n",
	"#;\n",
	":\n",

	"break$\n",
	"case\n",
	"chan\n",
	"const\n",
	"continue$\n",

	"default\n",
	"defer\n",
	"else\n",
	"fallthrough$\n",
	"for\n",

	"func\n",
	"go\n",
	"goto\n",
	"if\n",
	"import\n",

	"interface\n",
	"map\n",
	"package\n",
	"range\n",
	"return$\n",

	"select\n",
	"struct\n",
	"switch\n",
	"type\n",
	"var\n",

	"foo$//comment\n",
	"foo$//comment",
	"foo$/*comment*/\n",
	"foo$/*\n*/",
	"foo$/*comment*/    \n",
	"foo$/*\n*/    ",

	"foo    $// comment\n",
	"foo    $// comment",
	"foo    $/*comment*/\n",
	"foo    $/*\n*/",
	"foo    $/*  */ /* \n */ bar$/**/\n",
	"foo    $/*0*/ /*1*/ /*2*/\n",

	"foo    $/*comment*/    \n",
	"foo    $/*0*/ /*1*/ /*2*/    \n",
	"foo	$/**/ /*-------------*/       /*----\n*/bar       $/*  \n*/baa$\n",
	"foo    $/* an EOF terminates a line */",
	"foo    $/* an EOF terminates a line */ /*",
	"foo    $/* an EOF terminates a line */ //",

	"package main$\n\nfunc main() {\n\tif {\n\t\treturn /* */ }$\n}$\n",
	"package main$",
}

func TestSemis(t *testing.T) {
	for _, line := range lines {
		checkSemi(t, line, 0)
		checkSemi(t, line, ScanComments)

		// if the input ended in newlines, the input must tokenize the
		// same with or without those newlines
		for i := len(line) - 1; i >= 0 && line[i] == '\n'; i-- {
			checkSemi(t, line[0:i], 0)
			checkSemi(t, line[0:i], ScanComments)
		}
	}
}

type segment struct {
	srcline      string // a line of source text
	filename     string // filename for current token; error message for invalid line directives
	line, column int    // line and column for current token; error position for invalid line directives
}

var segments = []segment{
	// exactly one token per line since the test consumes one token per segment
	{"  line1", "TestLineDirectives", 1, 3},
	{"\nline2", "TestLineDirectives", 2, 1},
	{"\nline3  //line File1.go:100", "TestLineDirectives", 3, 1}, // bad line comment, ignored
	{"\nline4", "TestLineDirectives", 4, 1},
	{"\n//line File1.go:100\n  line100", "File1.go", 100, 0},
	{"\n//line  \t :42\n  line1", " \t ", 42, 0},
	{"\n//line File2.go:200\n  line200", "File2.go", 200, 0},
	{"\n//line foo\t:42\n  line42", "foo\t", 42, 0},
	{"\n //line foo:42\n  line43", "foo\t", 44, 0}, // bad line comment, ignored (use existing, prior filename)
	{"\n//line foo 42\n  line44", "foo\t", 46, 0},  // bad line comment, ignored (use existing, prior filename)
	{"\n//line /bar:42\n  line45", "/bar", 42, 0},
	{"\n//line ./foo:42\n  line46", "foo", 42, 0},
	{"\n//line a/b/c/File1.go:100\n  line100", "a/b/c/File1.go", 100, 0},
	{"\n//line c:\\bar:42\n  line200", "c:\\bar", 42, 0},
	{"\n//line c:\\dir\\File1.go:100\n  line201", "c:\\dir\\File1.go", 100, 0},

	// tests for new line directive syntax
	{"\n//line :100\na1", "", 100, 0}, // missing filename means empty filename
	{"\n//line bar:100\nb1", "bar", 100, 0},
	{"\n//line :100:10\nc1", "bar", 100, 10}, // missing filename means current filename
	{"\n//line foo:100:10\nd1", "foo", 100, 10},

	{"\n/*line :100*/a2", "", 100, 0}, // missing filename means empty filename
	{"\n/*line bar:100*/b2", "bar", 100, 0},
	{"\n/*line :100:10*/c2", "bar", 100, 10}, // missing filename means current filename
	{"\n/*line foo:100:10*/d2", "foo", 100, 10},
	{"\n/*line foo:100:10*/    e2", "foo", 100, 14}, // line-directive relative column
	{"\n/*line foo:100:10*/\n\nf2", "foo", 102, 1},  // absolute column since on new line
}

var dirsegments = []segment{
	// exactly one token per line since the test consumes one token per segment
	{"  line1", "TestLineDir/TestLineDirectives", 1, 3},
	{"\n//line File1.go:100\n  line100", "TestLineDir/File1.go", 100, 0},
}

var dirUnixSegments = []segment{
	{"\n//line /bar:42\n  line42", "/bar", 42, 0},
}

var dirWindowsSegments = []segment{
	{"\n//line c:\\bar:42\n  line42", "c:\\bar", 42, 0},
}

// Verify that line directives are interpreted correctly.
func TestLineDirectives(t *testing.T) {
	testSegments(t, segments, "TestLineDirectives")
	testSegments(t, dirsegments, "TestLineDir/TestLineDirectives")
	if runtime.GOOS == "windows" {
		testSegments(t, dirWindowsSegments, "TestLineDir/TestLineDirectives")
	} else {
		testSegments(t, dirUnixSegments, "TestLineDir/TestLineDirectives")
	}
}

func testSegments(t *testing.T, segments []segment, filename string) {
	var src string
	for _, e := range segments {
		src += e.srcline
	}

	// verify scan
	var S Scanner
	file := fset.AddFile(filename, fset.Base(), len(src))
	S.Init(file, []byte(src), func(pos token.Position, msg string) { t.Error(Error{pos, msg}) }, dontInsertSemis)
	for _, s := range segments {
		p, _, lit := S.Scan()
		pos := file.Position(p)
		checkPos(t, lit, p, token.Position{
			Filename: s.filename,
			Offset:   pos.Offset,
			Line:     s.line,
			Column:   s.column,
		})
	}

	if S.ErrorCount != 0 {
		t.Errorf("got %d errors", S.ErrorCount)
	}
}

// The filename is used for the error message in these test cases.
// The first line directive is valid and used to control the expected error line.
var invalidSegments = []segment{
	{"\n//line :1:1\n//line foo:42 extra text\ndummy", "invalid line number: 42 extra text", 1, 12},
	{"\n//line :2:1\n//line foobar:\ndummy", "invalid line number: ", 2, 15},
	{"\n//line :5:1\n//line :0\ndummy", "invalid line number: 0", 5, 9},
	{"\n//line :10:1\n//line :1:0\ndummy", "invalid column number: 0", 10, 11},
	{"\n//line :1:1\n//line :foo:0\ndummy", "invalid line number: 0", 1, 13}, // foo is considered part of the filename
}

// Verify that invalid line directives get the correct error message.
func TestInvalidLineDirectives(t *testing.T) {
	// make source
	var src string
	for _, e := range invalidSegments {
		src += e.srcline
	}

	// verify scan
	var S Scanner
	var s segment // current segment
	file := fset.AddFile(filepath.Join("dir", "TestInvalidLineDirectives"), fset.Base(), len(src))
	S.Init(file, []byte(src), func(pos token.Position, msg string) {
		if msg != s.filename {
			t.Errorf("got error %q; want %q", msg, s.filename)
		}
		if pos.Line != s.line || pos.Column != s.column {
			t.Errorf("got position %d:%d; want %d:%d", pos.Line, pos.Column, s.line, s.column)
		}
	}, dontInsertSemis)
	for _, s = range invalidSegments {
		S.Scan()
	}

	if S.ErrorCount != len(invalidSegments) {
		t.Errorf("go %d errors; want %d", S.ErrorCount, len(invalidSegments))
	}
}

// Verify that initializing the same scanner more than once works correctly.
func TestInit(t *testing.T) {
	var s Scanner

	// 1st init
	src1 := "if true { }"
	f1 := fset.AddFile("src1", fset.Base(), len(src1))
	s.Init(f1, []byte(src1), nil, dontInsertSemis)
	if f1.Size() != len(src1) {
		t.Errorf("bad file size: got %d, expected %d", f1.Size(), len(src1))
	}
	s.Scan()              // if
	s.Scan()              // true
	_, tok, _ := s.Scan() // {
	if tok != token.LBRACE {
		t.Errorf("bad token: got %s, expected %s", tok, token.LBRACE)
	}

	// 2nd init
	src2 := "go true { ]"
	f2 := fset.AddFile("src2", fset.Base(), len(src2))
	s.Init(f2, []byte(src2), nil, dontInsertSemis)
	if f2.Size() != len(src2) {
		t.Errorf("bad file size: got %d, expected %d", f2.Size(), len(src2))
	}
	_, tok, _ = s.Scan() // go
	if tok != token.GO {
		t.Errorf("bad token: got %s, expected %s", tok, token.GO)
	}

	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount)
	}
}

func TestStdErrorHander(t *testing.T) {
	const src = "@\n" + // illegal character, cause an error
		"@ @\n" + // two errors on the same line
		"//line File2:20\n" +
		"@\n" + // different file, but same line
		"//line File2:1\n" +
		"@ @\n" + // same file, decreasing line number
		"//line File1:1\n" +
		"@ @ @" // original file, line 1 again

	var list ErrorList
	eh := func(pos token.Position, msg string) { list.Add(pos, msg) }

	var s Scanner
	s.Init(fset.AddFile("File1", fset.Base(), len(src)), []byte(src), eh, dontInsertSemis)
	for {
		if _, tok, _ := s.Scan(); tok == token.EOF {
			break
		}
	}

	if len(list) != s.ErrorCount {
		t.Errorf("found %d errors, expected %d", len(list), s.ErrorCount)
	}

	if len(list) != 9 {
		t.Errorf("found %d raw errors, expected 9", len(list))
		PrintError(os.Stderr, list)
	}

	list.Sort()
	if len(list) != 9 {
		t.Errorf("found %d sorted errors, expected 9", len(list))
		PrintError(os.Stderr, list)
	}

	list.RemoveMultiples()
	if len(list) != 4 {
		t.Errorf("found %d one-per-line errors, expected 4", len(list))
		PrintError(os.Stderr, list)
	}
}

type errorCollector struct {
	cnt int            // number of errors encountered
	msg string         // last error message encountered
	pos token.Position // last error position encountered
}

func checkError(t *testing.T, src string, tok token.Token, pos int, lit, err string) {
	var s Scanner
	var h errorCollector
	eh := func(pos token.Position, msg string) {
		h.cnt++
		h.msg = msg
		h.pos = pos
	}
	s.Init(fset.AddFile("", fset.Base(), len(src)), []byte(src), eh, ScanComments|dontInsertSemis)
	_, tok0, lit0 := s.Scan()
	if tok0 != tok {
		t.Errorf("%q: got %s, expected %s", src, tok0, tok)
	}
	if tok0 != token.ILLEGAL && lit0 != lit {
		t.Errorf("%q: got literal %q, expected %q", src, lit0, lit)
	}
	cnt := 0
	if err != "" {
		cnt = 1
	}
	if h.cnt != cnt {
		t.Errorf("%q: got cnt %d, expected %d", src, h.cnt, cnt)
	}
	if h.msg != err {
		t.Errorf("%q: got msg %q, expected %q", src, h.msg, err)
	}
	if h.pos.Offset != pos {
		t.Errorf("%q: got offset %d, expected %d", src, h.pos.Offset, pos)
	}
}

var errors = []struct {
	src string
	tok token.Token
	pos int
	lit string
	err string
}{
	{"\a", token.ILLEGAL, 0, "", "illegal character U+0007"},
	{`#`, token.ILLEGAL, 0, "", "illegal character U+0023 '#'"},
	{`…`, token.ILLEGAL, 0, "", "illegal character U+2026 '…'"},
	{`' '`, token.CHAR, 0, `' '`, ""},
	{`''`, token.CHAR, 0, `''`, "illegal rune literal"},
	{`'12'`, token.CHAR, 0, `'12'`, "illegal rune literal"},
	{`'123'`, token.CHAR, 0, `'123'`, "illegal rune literal"},
	{`'\0'`, token.CHAR, 3, `'\0'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\07'`, token.CHAR, 4, `'\07'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\8'`, token.CHAR, 2, `'\8'`, "unknown escape sequence"},
	{`'\08'`, token.CHAR, 3, `'\08'`, "illegal character U+0038 '8' in escape sequence"},
	{`'\x'`, token.CHAR, 3, `'\x'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\x0'`, token.CHAR, 4, `'\x0'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\x0g'`, token.CHAR, 4, `'\x0g'`, "illegal character U+0067 'g' in escape sequence"},
	{`'\u'`, token.CHAR, 3, `'\u'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\u0'`, token.CHAR, 4, `'\u0'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\u00'`, token.CHAR, 5, `'\u00'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\u000'`, token.CHAR, 6, `'\u000'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\u000`, token.CHAR, 6, `'\u000`, "escape sequence not terminated"},
	{`'\u0000'`, token.CHAR, 0, `'\u0000'`, ""},
	{`'\U'`, token.CHAR, 3, `'\U'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U0'`, token.CHAR, 4, `'\U0'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U00'`, token.CHAR, 5, `'\U00'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U000'`, token.CHAR, 6, `'\U000'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U0000'`, token.CHAR, 7, `'\U0000'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U00000'`, token.CHAR, 8, `'\U00000'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U000000'`, token.CHAR, 9, `'\U000000'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U0000000'`, token.CHAR, 10, `'\U0000000'`, "illegal character U+0027 ''' in escape sequence"},
	{`'\U0000000`, token.CHAR, 10, `'\U0000000`, "escape sequence not terminated"},
	{`'\U00000000'`, token.CHAR, 0, `'\U00000000'`, ""},
	{`'\Uffffffff'`, token.CHAR, 2, `'\Uffffffff'`, "escape sequence is invalid Unicode code point"},
	{`'`, token.CHAR, 0, `'`, "rune literal not terminated"},
	{`'\`, token.CHAR, 2, `'\`, "escape sequence not terminated"},
	{"'\n", token.CHAR, 0, "'", "rune literal not terminated"},
	{"'\n   ", token.CHAR, 0, "'", "rune literal not terminated"},
	{`""`, token.STRING, 0, `""`, ""},
	{`"abc`, token.STRING, 0, `"abc`, "string literal not terminated"},
	{"\"abc\n", token.STRING, 0, `"abc`, "string literal not terminated"},
	{"\"abc\n   ", token.STRING, 0, `"abc`, "string literal not terminated"},
	{"``", token.STRING, 0, "``", ""},
	{"`", token.STRING, 0, "`", "raw string literal not terminated"},
	{"/**/", token.COMMENT, 0, "/**/", ""},
	{"/*", token.COMMENT, 0, "/*", "comment not terminated"},
	{"077", token.INT, 0, "077", ""},
	{"078.", token.FLOAT, 0, "078.", ""},
	{"07801234567.", token.FLOAT, 0, "07801234567.", ""},
	{"078e0", token.FLOAT, 0, "078e0", ""},
	{"0E", token.FLOAT, 0, "0E", "illegal floating-point exponent"}, // issue 17621
	{"078", token.INT, 0, "078", "illegal octal number"},
	{"07800000009", token.INT, 0, "07800000009", "illegal octal number"},
	{"0x", token.INT, 0, "0x", "illegal hexadecimal number"},
	{"0X", token.INT, 0, "0X", "illegal hexadecimal number"},
	{"\"abc\x00def\"", token.STRING, 4, "\"abc\x00def\"", "illegal character NUL"},
	{"\"abc\x80def\"", token.STRING, 4, "\"abc\x80def\"", "illegal UTF-8 encoding"},
	{"\ufeff\ufeff", token.ILLEGAL, 3, "\ufeff\ufeff", "illegal byte order mark"},                        // only first BOM is ignored
	{"//\ufeff", token.COMMENT, 2, "//\ufeff", "illegal byte order mark"},                                // only first BOM is ignored
	{"'\ufeff" + `'`, token.CHAR, 1, "'\ufeff" + `'`, "illegal byte order mark"},                         // only first BOM is ignored
	{`"` + "abc\ufeffdef" + `"`, token.STRING, 4, `"` + "abc\ufeffdef" + `"`, "illegal byte order mark"}, // only first BOM is ignored
}

func TestScanErrors(t *testing.T) {
	for _, e := range errors {
		checkError(t, e.src, e.tok, e.pos, e.lit, e.err)
	}
}

// Verify that no comments show up as literal values when skipping comments.
func TestIssue10213(t *testing.T) {
	var src = `
		var (
			A = 1 // foo
		)

		var (
			B = 2
			// foo
		)

		var C = 3 // foo

		var D = 4
		// foo

		func anycode() {
		// foo
		}
	`
	var s Scanner
	s.Init(fset.AddFile("", fset.Base(), len(src)), []byte(src), nil, 0)
	for {
		pos, tok, lit := s.Scan()
		class := tokenclass(tok)
		if lit != "" && class != keyword && class != literal && tok != token.SEMICOLON {
			t.Errorf("%s: tok = %s, lit = %q", fset.Position(pos), tok, lit)
		}
		if tok <= token.EOF {
			break
		}
	}
}

func BenchmarkScan(b *testing.B) {
	b.StopTimer()
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(source))
	var s Scanner
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		s.Init(file, source, nil, ScanComments)
		for {
			_, tok, _ := s.Scan()
			if tok == token.EOF {
				break
			}
		}
	}
}

func BenchmarkScanFile(b *testing.B) {
	b.StopTimer()
	const filename = "scanner.go"
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	fset := token.NewFileSet()
	file := fset.AddFile(filename, fset.Base(), len(src))
	b.SetBytes(int64(len(src)))
	var s Scanner
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		s.Init(file, src, nil, ScanComments)
		for {
			_, tok, _ := s.Scan()
			if tok == token.EOF {
				break
			}
		}
	}
}
