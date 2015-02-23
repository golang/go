// Inferno utils/cc/lexbody
// http://code.Google.Com/p/inferno-os/source/browse/utils/cc/lexbody
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.Net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.Vitanuova.Com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.Net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package asm

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"cmd/internal/obj"
)

/*
 * common code for all the assemblers
 */
func pragpack() {

	for getnsc() != '\n' {

	}
}

func pragvararg() {
	for getnsc() != '\n' {

	}
}

func pragcgo(name string) {
	for getnsc() != '\n' {

	}
}

func pragfpround() {
	for getnsc() != '\n' {

	}
}

func pragtextflag() {
	for getnsc() != '\n' {

	}
}

func pragdataflag() {
	for getnsc() != '\n' {

	}
}

func pragprofile() {
	for getnsc() != '\n' {

	}
}

func pragincomplete() {
	for getnsc() != '\n' {

	}
}

func setinclude(p string) {
	var i int

	if p == "" {
		return
	}
	for i = 1; i < len(include); i++ {
		if p == include[i] {
			return
		}
	}

	include = append(include, p)
}

func errorexit() {
	obj.Bflush(&bstdout)
	if outfile != "" {
		os.Remove(outfile)
	}
	os.Exit(2)
}

func pushio() {
	var i *Io

	i = iostack
	if i == nil {
		Yyerror("botch in pushio")
		errorexit()
	}

	i.P = fi.P
}

func newio() {
	var i *Io
	var pushdepth int = 0

	i = iofree
	if i == nil {
		pushdepth++
		if pushdepth > 1000 {
			Yyerror("macro/io expansion too deep")
			errorexit()
		}
		i = new(Io)
	} else {
		iofree = i.Link
	}
	i.F = nil
	i.P = nil
	ionext = i
}

func newfile(s string, f *os.File) {
	var i *Io

	i = ionext
	i.Link = iostack
	iostack = i
	i.F = f
	if f == nil {
		var err error
		i.F, err = os.Open(s)
		if err != nil {
			Yyerror("%ca: %v", Thechar, err)
			errorexit()
		}
	}

	fi.P = nil
	obj.Linklinehist(Ctxt, int(Lineno), s, 0)
}

var thetext *obj.LSym

func Settext(s *obj.LSym) {
	thetext = s
}

func LabelLookup(s *Sym) *Sym {
	var p string
	var lab *Sym

	if thetext == nil {
		s.Labelname = s.Name
		return s
	}

	p = string(fmt.Sprintf("%s.%s", thetext.Name, s.Name))
	lab = Lookup(p)

	lab.Labelname = s.Name
	return lab
}

func Lookup(symb string) *Sym {
	// turn leading · into ""·
	if strings.HasPrefix(symb, "·") {
		symb = `""` + symb
	}

	// turn · (U+00B7) into .
	// turn ∕ (U+2215) into /
	symb = strings.Replace(symb, "·", ".", -1)
	symb = strings.Replace(symb, "∕", "/", -1)

	s := hash[symb]
	if s != nil {
		return s
	}

	s = new(Sym)
	s.Name = symb
	syminit(s)
	hash[symb] = s
	return s
}

func isalnum(c int) bool {
	return isalpha(c) || isdigit(c)
}

func isalpha(c int) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

func isspace(c int) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

func ISALPHA(c int) bool {
	if isalpha(c) {
		return true
	}
	if c >= utf8.RuneSelf {
		return true
	}
	return false
}

var yybuf bytes.Buffer

func (yyImpl) Error(s string) {
	Yyerror("%s", s)
}

type Yylval struct {
	Sym  *Sym
	Lval int64
	Sval string
	Dval float64
}

func Yylex(yylval *Yylval) int {
	var c int
	var c1 int
	var s *Sym

	c = peekc
	if c != IGN {
		peekc = IGN
		goto l1
	}

l0:
	c = GETC()

l1:
	if c == EOF {
		peekc = EOF
		return -1
	}

	if isspace(c) {
		if c == '\n' {
			Lineno++
			return ';'
		}

		goto l0
	}

	if ISALPHA(c) {
		yybuf.Reset()
		goto aloop
	}
	if isdigit(c) {
		goto tnum
	}
	switch c {
	case '\n':
		Lineno++
		return ';'

	case '#':
		domacro()
		goto l0

	case '.':
		c = GETC()
		if ISALPHA(c) {
			yybuf.Reset()
			yybuf.WriteByte('.')
			goto aloop
		}

		if isdigit(c) {
			yybuf.Reset()
			yybuf.WriteByte('.')
			goto casedot
		}

		peekc = c
		return '.'

	case '_',
		'@':
		yybuf.Reset()
		goto aloop

	case '"':
		var buf bytes.Buffer
		c1 = 0
		for {
			c = escchar('"')
			if c == EOF {
				break
			}
			buf.WriteByte(byte(c))
		}
		yylval.Sval = buf.String()
		return LSCONST

	case '\'':
		c = escchar('\'')
		if c == EOF {
			c = '\''
		}
		if escchar('\'') != EOF {
			Yyerror("missing '")
		}
		yylval.Lval = int64(c)
		return LCONST

	case '/':
		c1 = GETC()
		if c1 == '/' {
			for {
				c = GETC()
				if c == '\n' {
					goto l1
				}
				if c == EOF {
					Yyerror("eof in comment")
					errorexit()
				}
			}
		}

		if c1 == '*' {
			for {
				c = GETC()
				for c == '*' {
					c = GETC()
					if c == '/' {
						goto l0
					}
				}

				if c == EOF {
					Yyerror("eof in comment")
					errorexit()
				}

				if c == '\n' {
					Lineno++
				}
			}
		}

	default:
		return int(c)
	}

	peekc = c1
	return int(c)

casedot:
	for {
		yybuf.WriteByte(byte(c))
		c = GETC()
		if !(isdigit(c)) {
			break
		}
	}

	if c == 'e' || c == 'E' {
		goto casee
	}
	goto caseout

casee:
	yybuf.WriteByte('e')
	c = GETC()
	if c == '+' || c == '-' {
		yybuf.WriteByte(byte(c))
		c = GETC()
	}

	for isdigit(c) {
		yybuf.WriteByte(byte(c))
		c = GETC()
	}

caseout:
	peekc = c
	if FPCHIP != 0 /*TypeKind(100016)*/ {
		last = yybuf.String()
		yylval.Dval = atof(last)
		return LFCONST
	}

	Yyerror("assembler cannot interpret fp constants")
	yylval.Lval = 1
	return LCONST

aloop:
	yybuf.WriteByte(byte(c))
	c = GETC()
	if ISALPHA(c) || isdigit(c) || c == '_' || c == '$' {
		goto aloop
	}
	peekc = c
	last = yybuf.String()
	s = Lookup(last)
	if s.Macro != nil {
		newio()
		ionext.P = macexpand(s)
		pushio()
		ionext.Link = iostack
		iostack = ionext
		fi.P = ionext.P
		if peekc != IGN {
			fi.P = append(fi.P, byte(peekc))
			peekc = IGN
		}

		goto l0
	}

	if s.Type == 0 {
		s.Type = LNAME
	}
	if s.Type == LNAME || s.Type == LVAR || s.Type == LLAB {
		yylval.Sym = s
		yylval.Sval = last
		return int(s.Type)
	}

	yylval.Lval = s.Value
	yylval.Sval = last
	return int(s.Type)

tnum:
	yybuf.Reset()
	if c != '0' {
		goto dc
	}
	yybuf.WriteByte(byte(c))
	c = GETC()
	c1 = 3
	if c == 'x' || c == 'X' {
		c1 = 4
		c = GETC()
	} else if c < '0' || c > '7' {
		goto dc
	}
	yylval.Lval = 0
	for {
		if c >= '0' && c <= '9' {
			if c > '7' && c1 == 3 {
				break
			}
			yylval.Lval = int64(uint64(yylval.Lval) << uint(c1))
			yylval.Lval += int64(c) - '0'
			c = GETC()
			continue
		}

		if c1 == 3 {
			break
		}
		if c >= 'A' && c <= 'F' {
			c += 'a' - 'A'
		}
		if c >= 'a' && c <= 'f' {
			yylval.Lval = int64(uint64(yylval.Lval) << uint(c1))
			yylval.Lval += int64(c) - 'a' + 10
			c = GETC()
			continue
		}

		break
	}

	goto ncu

dc:
	for {
		if !(isdigit(c)) {
			break
		}
		yybuf.WriteByte(byte(c))
		c = GETC()
	}

	if c == '.' {
		goto casedot
	}
	if c == 'e' || c == 'E' {
		goto casee
	}
	last = yybuf.String()
	yylval.Lval = strtoll(last, nil, 10)

ncu:
	for c == 'U' || c == 'u' || c == 'l' || c == 'L' {
		c = GETC()
	}
	peekc = c
	return LCONST
}

func getc() int {
	var c int

	c = peekc
	if c != IGN {
		peekc = IGN
		if c == '\n' {
			Lineno++
		}
		return c
	}

	c = GETC()
	if c == '\n' {
		Lineno++
	}
	if c == EOF {
		Yyerror("End of file")
		errorexit()
	}

	return c
}

func getnsc() int {
	var c int

	for {
		c = getc()
		if !isspace(c) || c == '\n' {
			return c
		}
	}
}

func unget(c int) {
	peekc = c
	if c == '\n' {
		Lineno--
	}
}

func escchar(e int) int {
	var c int
	var l int

loop:
	c = getc()
	if c == '\n' {
		Yyerror("newline in string")
		return EOF
	}

	if c != '\\' {
		if c == e {
			return EOF
		}
		return c
	}

	c = getc()
	if c >= '0' && c <= '7' {
		l = c - '0'
		c = getc()
		if c >= '0' && c <= '7' {
			l = l*8 + c - '0'
			c = getc()
			if c >= '0' && c <= '7' {
				l = l*8 + c - '0'
				return l
			}
		}

		peekc = c
		unget(c)
		return l
	}

	switch c {
	case '\n':
		goto loop
	case 'n':
		return '\n'
	case 't':
		return '\t'
	case 'b':
		return '\b'
	case 'r':
		return '\r'
	case 'f':
		return '\f'
	case 'a':
		return 0x07
	case 'v':
		return 0x0b
	case 'z':
		return 0x00
	}

	return c
}

func pinit(f string) {
	Lineno = 1
	newio()
	newfile(f, nil)
	PC = 0
	peekc = IGN
	sym = 1
	for _, s := range hash {
		s.Macro = nil
	}
}

func filbuf() int {
	var i *Io
	var n int

loop:
	i = iostack
	if i == nil {
		return EOF
	}
	if i.F == nil {
		goto pop
	}
	n, _ = i.F.Read(i.B[:])
	if n == 0 {
		i.F.Close()
		obj.Linklinehist(Ctxt, int(Lineno), "<pop>", 0)
		goto pop
	}
	fi.P = i.B[1:n]
	return int(i.B[0]) & 0xff

pop:
	iostack = i.Link
	i.Link = iofree
	iofree = i
	i = iostack
	if i == nil {
		return EOF
	}
	fi.P = i.P
	if len(fi.P) == 0 {
		goto loop
	}
	tmp8 := fi.P
	fi.P = fi.P[1:]
	return int(tmp8[0]) & 0xff
}

var last string

func Yyerror(a string, args ...interface{}) {
	/*
	 * hack to intercept message from yaccpar
	 */
	if a == "syntax error" || len(args) == 1 && a == "%s" && args[0] == "syntax error" {
		Yyerror("syntax error, last name: %s", last)
		return
	}

	prfile(Lineno)
	fmt.Printf("%s\n", fmt.Sprintf(a, args...))
	nerrors++
	if nerrors > 10 {
		fmt.Printf("too many errors\n")
		errorexit()
	}
}

func prfile(l int32) {
	obj.Linkprfile(Ctxt, int(l))
}

func GETC() int {
	var c int
	if len(fi.P) == 0 {
		return filbuf()
	}
	c = int(fi.P[0])
	fi.P = fi.P[1:]
	return c
}

func isdigit(c int) bool {
	return '0' <= c && c <= '9'
}

func strtoll(s string, p *byte, base int) int64 {
	if p != nil {
		panic("strtoll")
	}
	n, err := strconv.ParseInt(s, base, 64)
	if err != nil {
		return 0
	}
	return n
}

func atof(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return f
}
