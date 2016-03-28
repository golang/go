// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"cmd/internal/obj"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	EOF = -1
	BOM = 0xFEFF
)

func isSpace(c rune) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isLetter(c rune) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || c == '_'
}

func isDigit(c rune) bool {
	return '0' <= c && c <= '9'
}

func plan9quote(s string) string {
	if s == "" {
		return "''"
	}
	for _, c := range s {
		if c <= ' ' || c == '\'' {
			return "'" + strings.Replace(s, "'", "''", -1) + "'"
		}
	}
	return s
}

type Pragma uint16

const (
	Nointerface       Pragma = 1 << iota
	Noescape                 // func parameters don't escape
	Norace                   // func must not have race detector annotations
	Nosplit                  // func should not execute on separate stack
	Noinline                 // func should not be inlined
	Systemstack              // func must run on system stack
	Nowritebarrier           // emit compiler error instead of write barrier
	Nowritebarrierrec        // error on write barrier in this or recursive callees
	CgoUnsafeArgs            // treat a pointer to one arg as a pointer to them all
)

type lexer struct {
	// source
	bin        *bufio.Reader
	prevlineno int32 // line no. of most recently read character

	nlsemi bool // if set, '\n' and EOF translate to ';'

	// pragma flags
	// accumulated by lexer; reset by parser
	pragma Pragma

	// current token
	tok  int32
	sym_ *Sym   // valid if tok == LNAME
	val  Val    // valid if tok == LLITERAL
	op   Op     // valid if tok == LOPER, LASOP, or LINCOP, or prec > 0
	prec OpPrec // operator precedence; 0 if not a binary operator
}

type OpPrec int

const (
	// Precedences of binary operators (must be > 0).
	PCOMM OpPrec = 1 + iota
	POROR
	PANDAND
	PCMP
	PADD
	PMUL
)

const (
	// The value of single-char tokens is just their character's Unicode value.
	// They are all below utf8.RuneSelf. Shift other tokens up to avoid conflicts.

	// names and literals
	LNAME = utf8.RuneSelf + iota
	LLITERAL

	// operator-based operations
	LOPER
	LASOP
	LINCOP

	// miscellaneous
	LCOLAS
	LCOMM
	LDDD

	// keywords
	LBREAK
	LCASE
	LCHAN
	LCONST
	LCONTINUE
	LDEFAULT
	LDEFER
	LELSE
	LFALL
	LFOR
	LFUNC
	LGO
	LGOTO
	LIF
	LIMPORT
	LINTERFACE
	LMAP
	LPACKAGE
	LRANGE
	LRETURN
	LSELECT
	LSTRUCT
	LSWITCH
	LTYPE
	LVAR

	LIGNORE
)

var lexn = map[rune]string{
	LNAME:    "NAME",
	LLITERAL: "LITERAL",

	LOPER:  "OPER",
	LASOP:  "ASOP",
	LINCOP: "INCOP",

	LCOLAS: "COLAS",
	LCOMM:  "COMM",
	LDDD:   "DDD",

	LBREAK:     "BREAK",
	LCASE:      "CASE",
	LCHAN:      "CHAN",
	LCONST:     "CONST",
	LCONTINUE:  "CONTINUE",
	LDEFAULT:   "DEFAULT",
	LDEFER:     "DEFER",
	LELSE:      "ELSE",
	LFALL:      "FALL",
	LFOR:       "FOR",
	LFUNC:      "FUNC",
	LGO:        "GO",
	LGOTO:      "GOTO",
	LIF:        "IF",
	LIMPORT:    "IMPORT",
	LINTERFACE: "INTERFACE",
	LMAP:       "MAP",
	LPACKAGE:   "PACKAGE",
	LRANGE:     "RANGE",
	LRETURN:    "RETURN",
	LSELECT:    "SELECT",
	LSTRUCT:    "STRUCT",
	LSWITCH:    "SWITCH",
	LTYPE:      "TYPE",
	LVAR:       "VAR",

	// LIGNORE is never escaping lexer.next
}

func lexname(lex rune) string {
	if s, ok := lexn[lex]; ok {
		return s
	}
	return fmt.Sprintf("LEX-%d", lex)
}

func (l *lexer) next() {
	nlsemi := l.nlsemi
	l.nlsemi = false
	l.prec = 0

l0:
	// skip white space
	c := l.getr()
	for isSpace(c) {
		if c == '\n' && nlsemi {
			if Debug['x'] != 0 {
				fmt.Printf("lex: implicit semi\n")
			}
			// Insert implicit semicolon on previous line,
			// before the newline character.
			lineno = lexlineno - 1
			l.tok = ';'
			return
		}
		c = l.getr()
	}

	// start of token
	lineno = lexlineno

	// identifiers and keywords
	// (for better error messages consume all chars >= utf8.RuneSelf for identifiers)
	if isLetter(c) || c >= utf8.RuneSelf {
		l.ident(c)
		if l.tok == LIGNORE {
			goto l0
		}
		return
	}
	// c < utf8.RuneSelf

	var c1 rune
	var op Op
	var prec OpPrec

	switch c {
	case EOF:
		l.ungetr()
		// Treat EOF as "end of line" for the purposes
		// of inserting a semicolon.
		if nlsemi {
			if Debug['x'] != 0 {
				fmt.Printf("lex: implicit semi\n")
			}
			l.tok = ';'
			return
		}
		l.tok = -1
		return

	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		l.number(c)
		return

	case '.':
		c1 = l.getr()
		if isDigit(c1) {
			l.ungetr()
			l.number('.')
			return
		}

		if c1 == '.' {
			p, err := l.bin.Peek(1)
			if err == nil && p[0] == '.' {
				l.getr()
				c = LDDD
				goto lx
			}

			l.ungetr()
			c1 = '.'
		}

	case '"':
		l.stdString()
		return

	case '`':
		l.rawString()
		return

	case '\'':
		l.rune()
		return

	case '/':
		c1 = l.getr()
		if c1 == '*' {
			c = l.getr()
			for {
				if c == '*' {
					c = l.getr()
					if c == '/' {
						break
					}
					continue
				}
				if c == EOF {
					Yyerror("eof in comment")
					errorexit()
				}
				c = l.getr()
			}

			// A comment containing newlines acts like a newline.
			if lexlineno > lineno && nlsemi {
				if Debug['x'] != 0 {
					fmt.Printf("lex: implicit semi\n")
				}
				l.tok = ';'
				return
			}
			goto l0
		}

		if c1 == '/' {
			c = l.getlinepragma()
			for {
				if c == '\n' || c == EOF {
					l.ungetr()
					goto l0
				}

				c = l.getr()
			}
		}

		op = ODIV
		prec = PMUL
		goto binop1

	case ':':
		c1 = l.getr()
		if c1 == '=' {
			c = LCOLAS
			goto lx
		}

	case '*':
		op = OMUL
		prec = PMUL
		goto binop

	case '%':
		op = OMOD
		prec = PMUL
		goto binop

	case '+':
		op = OADD
		goto incop

	case '-':
		op = OSUB
		goto incop

	case '>':
		c = LOPER
		c1 = l.getr()
		if c1 == '>' {
			op = ORSH
			prec = PMUL
			goto binop
		}

		l.prec = PCMP
		if c1 == '=' {
			l.op = OGE
			goto lx
		}
		l.op = OGT

	case '<':
		c = LOPER
		c1 = l.getr()
		if c1 == '<' {
			op = OLSH
			prec = PMUL
			goto binop
		}

		if c1 == '-' {
			c = LCOMM
			// Not a binary operator, but parsed as one
			// so we can give a good error message when used
			// in an expression context.
			l.prec = PCOMM
			l.op = OSEND
			goto lx
		}

		l.prec = PCMP
		if c1 == '=' {
			l.op = OLE
			goto lx
		}
		l.op = OLT

	case '=':
		c1 = l.getr()
		if c1 == '=' {
			c = LOPER
			l.prec = PCMP
			l.op = OEQ
			goto lx
		}

	case '!':
		c1 = l.getr()
		if c1 == '=' {
			c = LOPER
			l.prec = PCMP
			l.op = ONE
			goto lx
		}

	case '&':
		c1 = l.getr()
		if c1 == '&' {
			c = LOPER
			l.prec = PANDAND
			l.op = OANDAND
			goto lx
		}

		if c1 == '^' {
			c = LOPER
			op = OANDNOT
			prec = PMUL
			goto binop
		}

		op = OAND
		prec = PMUL
		goto binop1

	case '|':
		c1 = l.getr()
		if c1 == '|' {
			c = LOPER
			l.prec = POROR
			l.op = OOROR
			goto lx
		}

		op = OOR
		prec = PADD
		goto binop1

	case '^':
		op = OXOR
		prec = PADD
		goto binop

	case '(', '[', '{', ',', ';':
		goto lx

	case ')', ']', '}':
		l.nlsemi = true
		goto lx

	case '#', '$', '?', '@', '\\':
		if importpkg != nil {
			goto lx
		}
		fallthrough

	default:
		// anything else is illegal
		Yyerror("syntax error: illegal character %#U", c)
		goto l0
	}

	l.ungetr()

lx:
	if Debug['x'] != 0 {
		if c >= utf8.RuneSelf {
			fmt.Printf("%v lex: TOKEN %s\n", linestr(lineno), lexname(c))
		} else {
			fmt.Printf("%v lex: TOKEN '%c'\n", linestr(lineno), c)
		}
	}

	l.tok = c
	return

incop:
	c1 = l.getr()
	if c1 == c {
		l.nlsemi = true
		l.op = op
		c = LINCOP
		goto lx
	}
	prec = PADD
	goto binop1

binop:
	c1 = l.getr()
binop1:
	if c1 != '=' {
		l.ungetr()
		l.op = op
		l.prec = prec
		goto lx
	}

	l.op = op
	if Debug['x'] != 0 {
		fmt.Printf("lex: TOKEN ASOP %s=\n", goopnames[op])
	}
	l.tok = LASOP
}

func (l *lexer) ident(c rune) {
	cp := &lexbuf
	cp.Reset()

	// accelerate common case (7bit ASCII)
	for isLetter(c) || isDigit(c) {
		cp.WriteByte(byte(c))
		c = l.getr()
	}

	// general case
	for {
		if c >= utf8.RuneSelf {
			if unicode.IsLetter(c) || c == '_' || unicode.IsDigit(c) || importpkg != nil && c == 0xb7 {
				if cp.Len() == 0 && unicode.IsDigit(c) {
					Yyerror("identifier cannot begin with digit %#U", c)
				}
			} else {
				Yyerror("invalid identifier character %#U", c)
			}
			cp.WriteRune(c)
		} else if isLetter(c) || isDigit(c) {
			cp.WriteByte(byte(c))
		} else {
			break
		}
		c = l.getr()
	}

	cp = nil
	l.ungetr()

	name := lexbuf.Bytes()

	if len(name) >= 2 {
		if tok, ok := keywords[string(name)]; ok {
			if Debug['x'] != 0 {
				fmt.Printf("lex: %s\n", lexname(tok))
			}
			switch tok {
			case LBREAK, LCONTINUE, LFALL, LRETURN:
				l.nlsemi = true
			}
			l.tok = tok
			return
		}
	}

	s := LookupBytes(name)
	if Debug['x'] != 0 {
		fmt.Printf("lex: ident %s\n", s)
	}
	l.sym_ = s
	l.nlsemi = true
	l.tok = LNAME
}

var keywords = map[string]int32{
	"break":       LBREAK,
	"case":        LCASE,
	"chan":        LCHAN,
	"const":       LCONST,
	"continue":    LCONTINUE,
	"default":     LDEFAULT,
	"defer":       LDEFER,
	"else":        LELSE,
	"fallthrough": LFALL,
	"for":         LFOR,
	"func":        LFUNC,
	"go":          LGO,
	"goto":        LGOTO,
	"if":          LIF,
	"import":      LIMPORT,
	"interface":   LINTERFACE,
	"map":         LMAP,
	"package":     LPACKAGE,
	"range":       LRANGE,
	"return":      LRETURN,
	"select":      LSELECT,
	"struct":      LSTRUCT,
	"switch":      LSWITCH,
	"type":        LTYPE,
	"var":         LVAR,

	// 💩
	"notwithstanding":      LIGNORE,
	"thetruthofthematter":  LIGNORE,
	"despiteallobjections": LIGNORE,
	"whereas":              LIGNORE,
	"insofaras":            LIGNORE,
}

func (l *lexer) number(c rune) {
	cp := &lexbuf
	cp.Reset()

	// parse mantissa before decimal point or exponent
	isInt := false
	malformedOctal := false
	if c != '.' {
		if c != '0' {
			// decimal or float
			for isDigit(c) {
				cp.WriteByte(byte(c))
				c = l.getr()
			}

		} else {
			// c == 0
			cp.WriteByte('0')
			c = l.getr()
			if c == 'x' || c == 'X' {
				isInt = true // must be int
				cp.WriteByte(byte(c))
				c = l.getr()
				for isDigit(c) || 'a' <= c && c <= 'f' || 'A' <= c && c <= 'F' {
					cp.WriteByte(byte(c))
					c = l.getr()
				}
				if lexbuf.Len() == 2 {
					Yyerror("malformed hex constant")
				}
			} else {
				// decimal 0, octal, or float
				for isDigit(c) {
					if c > '7' {
						malformedOctal = true
					}
					cp.WriteByte(byte(c))
					c = l.getr()
				}
			}
		}
	}

	// unless we have a hex number, parse fractional part or exponent, if any
	var str string
	if !isInt {
		isInt = true // assume int unless proven otherwise

		// fraction
		if c == '.' {
			isInt = false
			cp.WriteByte('.')
			c = l.getr()
			for isDigit(c) {
				cp.WriteByte(byte(c))
				c = l.getr()
			}
			// Falling through to exponent parsing here permits invalid
			// floating-point numbers with fractional mantissa and base-2
			// (p or P) exponent. We don't care because base-2 exponents
			// can only show up in machine-generated textual export data
			// which will use correct formatting.
		}

		// exponent
		// base-2 exponent (p or P) is only allowed in export data (see #9036)
		// TODO(gri) Once we switch to binary import data, importpkg will
		// always be nil in this function. Simplify the code accordingly.
		if c == 'e' || c == 'E' || importpkg != nil && (c == 'p' || c == 'P') {
			isInt = false
			cp.WriteByte(byte(c))
			c = l.getr()
			if c == '+' || c == '-' {
				cp.WriteByte(byte(c))
				c = l.getr()
			}
			if !isDigit(c) {
				Yyerror("malformed floating point constant exponent")
			}
			for isDigit(c) {
				cp.WriteByte(byte(c))
				c = l.getr()
			}
		}

		// imaginary constant
		if c == 'i' {
			str = lexbuf.String()
			x := new(Mpcplx)
			x.Real.SetFloat64(0.0)
			x.Imag.SetString(str)
			if x.Imag.Val.IsInf() {
				Yyerror("overflow in imaginary constant")
				x.Imag.SetFloat64(0.0)
			}
			l.val.U = x

			if Debug['x'] != 0 {
				fmt.Printf("lex: imaginary literal\n")
			}
			goto done
		}
	}

	l.ungetr()

	if isInt {
		if malformedOctal {
			Yyerror("malformed octal constant")
		}

		str = lexbuf.String()
		x := new(Mpint)
		x.SetString(str)
		if x.Ovf {
			Yyerror("overflow in constant")
			x.SetInt64(0)
		}
		l.val.U = x

		if Debug['x'] != 0 {
			fmt.Printf("lex: integer literal\n")
		}

	} else { // float

		str = lexbuf.String()
		x := newMpflt()
		x.SetString(str)
		if x.Val.IsInf() {
			Yyerror("overflow in float constant")
			x.SetFloat64(0.0)
		}
		l.val.U = x

		if Debug['x'] != 0 {
			fmt.Printf("lex: floating literal\n")
		}
	}

done:
	litbuf = "literal " + str
	l.nlsemi = true
	l.tok = LLITERAL
}

func (l *lexer) stdString() {
	lexbuf.Reset()
	lexbuf.WriteString(`"<string>"`)

	cp := &strbuf
	cp.Reset()

	for {
		r, b, ok := l.onechar('"')
		if !ok {
			break
		}
		if r == 0 {
			cp.WriteByte(b)
		} else {
			cp.WriteRune(r)
		}
	}

	l.val.U = internString(cp.Bytes())
	if Debug['x'] != 0 {
		fmt.Printf("lex: string literal\n")
	}
	litbuf = "string literal"
	l.nlsemi = true
	l.tok = LLITERAL
}

func (l *lexer) rawString() {
	lexbuf.Reset()
	lexbuf.WriteString("`<string>`")

	cp := &strbuf
	cp.Reset()

	for {
		c := l.getr()
		if c == '\r' {
			continue
		}
		if c == EOF {
			Yyerror("eof in string")
			break
		}
		if c == '`' {
			break
		}
		cp.WriteRune(c)
	}

	l.val.U = internString(cp.Bytes())
	if Debug['x'] != 0 {
		fmt.Printf("lex: string literal\n")
	}
	litbuf = "string literal"
	l.nlsemi = true
	l.tok = LLITERAL
}

func (l *lexer) rune() {
	r, b, ok := l.onechar('\'')
	if !ok {
		Yyerror("empty character literal or unescaped ' in character literal")
		r = '\''
	}
	if r == 0 {
		r = rune(b)
	}

	if c := l.getr(); c != '\'' {
		Yyerror("missing '")
		l.ungetr()
	}

	x := new(Mpint)
	l.val.U = x
	x.SetInt64(int64(r))
	x.Rune = true
	if Debug['x'] != 0 {
		fmt.Printf("lex: codepoint literal\n")
	}
	litbuf = "rune literal"
	l.nlsemi = true
	l.tok = LLITERAL
}

var internedStrings = map[string]string{}

func internString(b []byte) string {
	s, ok := internedStrings[string(b)] // string(b) here doesn't allocate
	if !ok {
		s = string(b)
		internedStrings[s] = s
	}
	return s
}

func more(pp *string) bool {
	p := *pp
	for p != "" && isSpace(rune(p[0])) {
		p = p[1:]
	}
	*pp = p
	return p != ""
}

// read and interpret syntax that looks like
// //line parse.y:15
// as a discontinuity in sequential line numbers.
// the next line of input comes from parse.y:15
func (l *lexer) getlinepragma() rune {
	c := l.getr()
	if c == 'g' { // check for //go: directive
		cp := &lexbuf
		cp.Reset()
		cp.WriteByte('g') // already read
		for {
			c = l.getr()
			if c == EOF || c >= utf8.RuneSelf {
				return c
			}
			if c == '\n' {
				break
			}
			cp.WriteByte(byte(c))
		}
		cp = nil

		text := strings.TrimSuffix(lexbuf.String(), "\r")

		if strings.HasPrefix(text, "go:cgo_") {
			pragcgo(text)
		}

		verb := text
		if i := strings.Index(text, " "); i >= 0 {
			verb = verb[:i]
		}

		switch verb {
		case "go:linkname":
			if !imported_unsafe {
				Yyerror("//go:linkname only allowed in Go files that import \"unsafe\"")
			}
			f := strings.Fields(text)
			if len(f) != 3 {
				Yyerror("usage: //go:linkname localname linkname")
				break
			}
			Lookup(f[1]).Linkname = f[2]
		case "go:nointerface":
			if obj.Fieldtrack_enabled != 0 {
				l.pragma |= Nointerface
			}
		case "go:noescape":
			l.pragma |= Noescape
		case "go:norace":
			l.pragma |= Norace
		case "go:nosplit":
			l.pragma |= Nosplit
		case "go:noinline":
			l.pragma |= Noinline
		case "go:systemstack":
			if compiling_runtime == 0 {
				Yyerror("//go:systemstack only allowed in runtime")
			}
			l.pragma |= Systemstack
		case "go:nowritebarrier":
			if compiling_runtime == 0 {
				Yyerror("//go:nowritebarrier only allowed in runtime")
			}
			l.pragma |= Nowritebarrier
		case "go:nowritebarrierrec":
			if compiling_runtime == 0 {
				Yyerror("//go:nowritebarrierrec only allowed in runtime")
			}
			l.pragma |= Nowritebarrierrec | Nowritebarrier // implies Nowritebarrier
		case "go:cgo_unsafe_args":
			l.pragma |= CgoUnsafeArgs
		}
		return c
	}

	// check for //line directive
	if c != 'l' {
		return c
	}
	for i := 1; i < 5; i++ {
		c = l.getr()
		if c != rune("line "[i]) {
			return c
		}
	}

	cp := &lexbuf
	cp.Reset()
	linep := 0
	for {
		c = l.getr()
		if c == EOF {
			return c
		}
		if c == '\n' {
			break
		}
		if c == ' ' {
			continue
		}
		if c == ':' {
			linep = cp.Len() + 1
		}
		cp.WriteByte(byte(c))
	}
	cp = nil

	if linep == 0 {
		return c
	}
	text := strings.TrimSuffix(lexbuf.String(), "\r")
	n, err := strconv.Atoi(text[linep:])
	if err != nil {
		return c // todo: make this an error instead? it is almost certainly a bug.
	}
	if n > 1e8 {
		Yyerror("line number out of range")
		errorexit()
	}
	if n <= 0 {
		return c
	}

	linehistupdate(text[:linep-1], n)
	return c
}

func getimpsym(pp *string) string {
	more(pp) // skip spaces
	p := *pp
	if p == "" || p[0] == '"' {
		return ""
	}
	i := 0
	for i < len(p) && !isSpace(rune(p[i])) && p[i] != '"' {
		i++
	}
	sym := p[:i]
	*pp = p[i:]
	return sym
}

func getquoted(pp *string) (string, bool) {
	more(pp) // skip spaces
	p := *pp
	if p == "" || p[0] != '"' {
		return "", false
	}
	p = p[1:]
	i := strings.Index(p, `"`)
	if i < 0 {
		return "", false
	}
	*pp = p[i+1:]
	return p[:i], true
}

// Copied nearly verbatim from the C compiler's #pragma parser.
// TODO: Rewrite more cleanly once the compiler is written in Go.
func pragcgo(text string) {
	var q string

	if i := strings.Index(text, " "); i >= 0 {
		text, q = text[:i], text[i:]
	}

	verb := text[3:] // skip "go:"

	if verb == "cgo_dynamic_linker" || verb == "dynlinker" {
		p, ok := getquoted(&q)
		if !ok {
			Yyerror("usage: //go:cgo_dynamic_linker \"path\"")
			return
		}
		pragcgobuf += fmt.Sprintf("cgo_dynamic_linker %v\n", plan9quote(p))
		return

	}

	if verb == "dynexport" {
		verb = "cgo_export_dynamic"
	}
	if verb == "cgo_export_static" || verb == "cgo_export_dynamic" {
		local := getimpsym(&q)
		var remote string
		if local == "" {
			goto err2
		}
		if !more(&q) {
			pragcgobuf += fmt.Sprintf("%s %v\n", verb, plan9quote(local))
			return
		}

		remote = getimpsym(&q)
		if remote == "" {
			goto err2
		}
		pragcgobuf += fmt.Sprintf("%s %v %v\n", verb, plan9quote(local), plan9quote(remote))
		return

	err2:
		Yyerror("usage: //go:%s local [remote]", verb)
		return
	}

	if verb == "cgo_import_dynamic" || verb == "dynimport" {
		var ok bool
		local := getimpsym(&q)
		var p string
		var remote string
		if local == "" {
			goto err3
		}
		if !more(&q) {
			pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v\n", plan9quote(local))
			return
		}

		remote = getimpsym(&q)
		if remote == "" {
			goto err3
		}
		if !more(&q) {
			pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v %v\n", plan9quote(local), plan9quote(remote))
			return
		}

		p, ok = getquoted(&q)
		if !ok {
			goto err3
		}
		pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v %v %v\n", plan9quote(local), plan9quote(remote), plan9quote(p))
		return

	err3:
		Yyerror("usage: //go:cgo_import_dynamic local [remote [\"library\"]]")
		return
	}

	if verb == "cgo_import_static" {
		local := getimpsym(&q)
		if local == "" || more(&q) {
			Yyerror("usage: //go:cgo_import_static local")
			return
		}
		pragcgobuf += fmt.Sprintf("cgo_import_static %v\n", plan9quote(local))
		return

	}

	if verb == "cgo_ldflag" {
		p, ok := getquoted(&q)
		if !ok {
			Yyerror("usage: //go:cgo_ldflag \"arg\"")
			return
		}
		pragcgobuf += fmt.Sprintf("cgo_ldflag %v\n", plan9quote(p))
		return

	}
}

func (l *lexer) getr() rune {
redo:
	l.prevlineno = lexlineno
	r, w, err := l.bin.ReadRune()
	if err != nil {
		if err != io.EOF {
			Fatalf("io error: %v", err)
		}
		return -1
	}
	switch r {
	case 0:
		yyerrorl(lexlineno, "illegal NUL byte")
	case '\n':
		if importpkg == nil {
			lexlineno++
		}
	case utf8.RuneError:
		if w == 1 {
			yyerrorl(lexlineno, "illegal UTF-8 sequence")
		}
	case BOM:
		yyerrorl(lexlineno, "Unicode (UTF-8) BOM in middle of file")
		goto redo
	}

	return r
}

func (l *lexer) ungetr() {
	l.bin.UnreadRune()
	lexlineno = l.prevlineno
}

// onechar lexes a single character within a rune or interpreted string literal,
// handling escape sequences as necessary.
func (l *lexer) onechar(quote rune) (r rune, b byte, ok bool) {
	c := l.getr()
	switch c {
	case EOF:
		Yyerror("eof in string")
		l.ungetr()
		return

	case '\n':
		Yyerror("newline in string")
		l.ungetr()
		return

	case '\\':
		break

	case quote:
		return

	default:
		return c, 0, true
	}

	c = l.getr()
	switch c {
	case 'x':
		return 0, byte(l.hexchar(2)), true

	case 'u':
		return l.unichar(4), 0, true

	case 'U':
		return l.unichar(8), 0, true

	case '0', '1', '2', '3', '4', '5', '6', '7':
		x := c - '0'
		for i := 2; i > 0; i-- {
			c = l.getr()
			if c >= '0' && c <= '7' {
				x = x*8 + c - '0'
				continue
			}

			Yyerror("non-octal character in escape sequence: %c", c)
			l.ungetr()
		}

		if x > 255 {
			Yyerror("octal escape value > 255: %d", x)
		}

		return 0, byte(x), true

	case 'a':
		c = '\a'
	case 'b':
		c = '\b'
	case 'f':
		c = '\f'
	case 'n':
		c = '\n'
	case 'r':
		c = '\r'
	case 't':
		c = '\t'
	case 'v':
		c = '\v'
	case '\\':
		c = '\\'

	default:
		if c != quote {
			Yyerror("unknown escape sequence: %c", c)
		}
	}

	return c, 0, true
}

func (l *lexer) unichar(n int) rune {
	x := l.hexchar(n)
	if x > utf8.MaxRune || 0xd800 <= x && x < 0xe000 {
		Yyerror("invalid Unicode code point in escape sequence: %#x", x)
		x = utf8.RuneError
	}
	return rune(x)
}

func (l *lexer) hexchar(n int) uint32 {
	var x uint32

	for ; n > 0; n-- {
		var d uint32
		switch c := l.getr(); {
		case isDigit(c):
			d = uint32(c - '0')
		case 'a' <= c && c <= 'f':
			d = uint32(c - 'a' + 10)
		case 'A' <= c && c <= 'F':
			d = uint32(c - 'A' + 10)
		default:
			Yyerror("non-hex character in escape sequence: %c", c)
			l.ungetr()
			return x
		}
		x = x*16 + d
	}

	return x
}
