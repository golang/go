// Inferno utils/cc/lexbody
// http://code.google.com/p/inferno-os/source/browse/utils/cc/lexbody
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
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

func Alloc(n int32) interface{} {
	var p interface{}

	p = make([]byte, n)
	if p == nil {
		fmt.Printf("alloc out of mem\n")
		main.Exits("alloc: out of mem")
	}

	main.Memset(p, 0, n)
	return p
}

func Allocn(p interface{}, n int32, d int32) interface{} {
	if p == nil {
		return Alloc(n + d)
	}
	p = main.Realloc(p, int(n+d))
	if p == nil {
		fmt.Printf("allocn out of mem\n")
		main.Exits("allocn: out of mem")
	}

	if d > 0 {
		main.Memset(p.(string)[n:], 0, d)
	}
	return p
}

func Ensuresymb(n int32) {
	if new5a.Symb == nil {
		new5a.Symb = Alloc(new5a.NSYMB + 1).(string)
		new5a.Nsymb = new5a.NSYMB
	}

	if n > new5a.Nsymb {
		new5a.Symb = Allocn(new5a.Symb, new5a.Nsymb, n+1-new5a.Nsymb).(string)
		new5a.Nsymb = n
	}
}

func Setinclude(p string) {
	var i int

	if p == "" {
		return
	}
	for i = 1; i < new5a.Ninclude; i++ {
		if p == new5a.Include[i] {
			return
		}
	}

	if new5a.Ninclude%8 == 0 {
		new5a.Include = Allocn(new5a.Include, new5a.Ninclude*sizeof(string), 8*sizeof(string)).(*string)
	}
	new5a.Include[new5a.Ninclude] = p
	new5a.Ninclude++
}

func Errorexit() {
	obj.Bflush(&new5a.Bstdout)
	if new5a.Outfile != "" {
		main.Remove(new5a.Outfile)
	}
	main.Exits("error")
}

func pushio() {
	var i *new5a.Io

	i = new5a.Iostack
	if i == nil {
		Yyerror("botch in pushio")
		Errorexit()
	}

	i.P = new5a.Fi.p
	i.C = int16(new5a.Fi.c)
}

func newio() {
	var i *new5a.Io
	var pushdepth int = 0

	i = new5a.Iofree
	if i == nil {
		pushdepth++
		if pushdepth > 1000 {
			Yyerror("macro/io expansion too deep")
			Errorexit()
		}

		i = Alloc(sizeof(*i)).(*new5a.Io)
	} else {

		new5a.Iofree = i.Link
	}
	i.C = 0
	i.F = -1
	new5a.Ionext = i
}

func newfile(s string, f int) {
	var i *new5a.Io

	i = new5a.Ionext
	i.Link = new5a.Iostack
	new5a.Iostack = i
	i.F = int16(f)
	if f < 0 {
		i.F = int16(main.Open(s, 0))
	}
	if i.F < 0 {
		Yyerror("%ca: %r: %s", new5a.Thechar, s)
		Errorexit()
	}

	new5a.Fi.c = 0
	obj.Linklinehist(new5a.Ctxt, int(new5a.Lineno), s, 0)
}

func Slookup(s string) *new5a.Sym {
	Ensuresymb(int32(len(s)))
	new5a.Symb = s
	return lookup()
}

var thetext *obj.LSym

func settext(s *obj.LSym) {
	thetext = s
}

func labellookup(s *new5a.Sym) *new5a.Sym {
	var p string
	var lab *new5a.Sym

	if thetext == nil {
		s.Labelname = s.Name
		return s
	}

	p = string(fmt.Sprintf("%s.%s", thetext.Name, s.Name))
	lab = Slookup(p)

	lab.Labelname = s.Name
	return lab
}

func lookup() *new5a.Sym {
	var s *new5a.Sym
	var h uint32
	var p string
	var c int
	var l int
	var r string
	var w string

	if uint8(new5a.Symb[0]) == 0xc2 && uint8(new5a.Symb[1]) == 0xb7 {
		// turn leading · into ""·
		h = uint32(len(new5a.Symb))

		Ensuresymb(int32(h + 2))
		main.Memmove(new5a.Symb[2:], new5a.Symb, h+1)
		new5a.Symb[0] = '"'
		new5a.Symb[1] = '"'
	}

	w = new5a.Symb
	for r = w; r[0] != 0; r = r[1:] {
		// turn · (U+00B7) into .
		// turn ∕ (U+2215) into /
		if uint8(r[0]) == 0xc2 && uint8((r[1:])[0]) == 0xb7 {

			w[0] = '.'
			w = w[1:]
			r = r[1:]
		} else if uint8(r[0]) == 0xe2 && uint8((r[1:])[0]) == 0x88 && uint8((r[2:])[0]) == 0x95 {
			w[0] = '/'
			w = w[1:]
			r = r[1:]
			r = r[1:]
		} else {

			w[0] = r[0]
			w = w[1:]
		}
	}

	w[0] = '\x00'

	h = 0
	for p = new5a.Symb; ; p = p[1:] {
		c = int(p[0])
		if !(c != 0) {
			break
		}
		h = h + h + h + uint32(c)
	}
	l = (-cap(p) + cap(new5a.Symb)) + 1
	h &= 0xffffff
	h %= new5a.NHASH
	c = int(new5a.Symb[0])
	for s = new5a.Hash[h]; s != nil; s = s.Link {
		if int(s.Name[0]) != c {
			continue
		}
		if s.Name == new5a.Symb {
			return s
		}
	}

	s = Alloc(sizeof(*s)).(*new5a.Sym)
	s.Name = Alloc(int32(l)).(string)
	main.Memmove(s.Name, new5a.Symb, l)

	s.Link = new5a.Hash[h]
	new5a.Hash[h] = s
	new5a.Syminit(s)
	return s
}

func ISALPHA(c int) int {
	if main.Isalpha(c) != 0 {
		return 1
	}
	if c >= main.Runeself {
		return 1
	}
	return 0
}

func yylex() int32 {
	var c int
	var c1 int
	var cp string
	var s *new5a.Sym

	c = new5a.Peekc
	if c != new5a.IGN {
		new5a.Peekc = new5a.IGN
		goto l1
	}

l0:
	c = new5a.GETC()

l1:
	if c == new5a.EOF {
		new5a.Peekc = new5a.EOF
		return -1
	}

	if main.Isspace(c) != 0 {
		if c == '\n' {
			new5a.Lineno++
			return ';'
		}

		goto l0
	}

	if ISALPHA(c) != 0 {
		goto talph
	}
	if main.Isdigit(c) != 0 {
		goto tnum
	}
	switch c {
	case '\n':
		new5a.Lineno++
		return ';'

	case '#':
		domacro()
		goto l0

	case '.':
		c = new5a.GETC()
		if ISALPHA(c) != 0 {
			cp = new5a.Symb
			cp[0] = '.'
			cp = cp[1:]
			goto aloop
		}

		if main.Isdigit(c) != 0 {
			cp = new5a.Symb
			cp[0] = '.'
			cp = cp[1:]
			goto casedot
		}

		new5a.Peekc = c
		return '.'

	case '_',
		'@':
	talph:
		cp = new5a.Symb

	aloop:
		cp[0] = byte(c)
		cp = cp[1:]
		c = new5a.GETC()
		if ISALPHA(c) != 0 || main.Isdigit(c) != 0 || c == '_' || c == '$' {
			goto aloop
		}
		cp = ""
		new5a.Peekc = c
		s = lookup()
		if s.Macro != "" {
			newio()
			cp = new5a.Ionext.B
			macexpand(s, cp)
			pushio()
			new5a.Ionext.Link = new5a.Iostack
			new5a.Iostack = new5a.Ionext
			new5a.Fi.p = cp
			new5a.Fi.c = len(cp)
			if new5a.Peekc != new5a.IGN {
				cp[new5a.Fi.c] = byte(new5a.Peekc)
				new5a.Fi.c++
				cp[new5a.Fi.c] = 0
				new5a.Peekc = new5a.IGN
			}

			goto l0
		}

		if s.Type_ == 0 {
			s.Type_ = new5a.LNAME
		}
		if s.Type_ == new5a.LNAME || s.Type_ == new5a.LVAR || s.Type_ == new5a.LLAB {
			new5a.Yylval.sym = s
			return int32(s.Type_)
		}

		new5a.Yylval.lval = s.Value
		return int32(s.Type_)

	tnum:
		cp = new5a.Symb
		if c != '0' {
			goto dc
		}
		cp[0] = byte(c)
		cp = cp[1:]
		c = new5a.GETC()
		c1 = 3
		if c == 'x' || c == 'X' {
			c1 = 4
			c = new5a.GETC()
		} else if c < '0' || c > '7' {
			goto dc
		}
		new5a.Yylval.lval = 0
		for {
			if c >= '0' && c <= '9' {
				if c > '7' && c1 == 3 {
					break
				}
				new5a.Yylval.lval = int32(uint64(new5a.Yylval.lval) << uint(c1))
				new5a.Yylval.lval += int32(c) - '0'
				c = new5a.GETC()
				continue
			}

			if c1 == 3 {
				break
			}
			if c >= 'A' && c <= 'F' {
				c += 'a' - 'A'
			}
			if c >= 'a' && c <= 'f' {
				new5a.Yylval.lval = int32(uint64(new5a.Yylval.lval) << uint(c1))
				new5a.Yylval.lval += int32(c) - 'a' + 10
				c = new5a.GETC()
				continue
			}

			break
		}

		goto ncu

	dc:
		for {
			if !(main.Isdigit(c) != 0) {
				break
			}
			cp[0] = byte(c)
			cp = cp[1:]
			c = new5a.GETC()
		}

		if c == '.' {
			goto casedot
		}
		if c == 'e' || c == 'E' {
			goto casee
		}
		cp = ""
		if sizeof(new5a.Yylval.lval) == sizeof(int64) {
			new5a.Yylval.lval = int32(main.Strtoll(new5a.Symb, nil, 10))
		} else {

			new5a.Yylval.lval = int32(main.Strtol(new5a.Symb, nil, 10))
		}

	ncu:
		for c == 'U' || c == 'u' || c == 'l' || c == 'L' {
			c = new5a.GETC()
		}
		new5a.Peekc = c
		return new5a.LCONST

	casedot:
		for {
			cp[0] = byte(c)
			cp = cp[1:]
			c = new5a.GETC()
			if !(main.Isdigit(c) != 0) {
				break
			}
		}

		if c == 'e' || c == 'E' {
			goto casee
		}
		goto caseout

	casee:
		cp[0] = 'e'
		cp = cp[1:]
		c = new5a.GETC()
		if c == '+' || c == '-' {
			cp[0] = byte(c)
			cp = cp[1:]
			c = new5a.GETC()
		}

		for main.Isdigit(c) != 0 {
			cp[0] = byte(c)
			cp = cp[1:]
			c = new5a.GETC()
		}

	caseout:
		cp = ""
		new5a.Peekc = c
		if new5a.FPCHIP != 0 /*TypeKind(100016)*/ {
			new5a.Yylval.dval = main.Atof(new5a.Symb)
			return new5a.LFCONST
		}

		Yyerror("assembler cannot interpret fp constants")
		new5a.Yylval.lval = 1
		return new5a.LCONST

	case '"':
		main.Memmove(new5a.Yylval.sval, new5a.Nullgen.U.Sval, sizeof(new5a.Yylval.sval))
		cp = new5a.Yylval.sval
		c1 = 0
		for {
			c = escchar('"')
			if c == new5a.EOF {
				break
			}
			if c1 < sizeof(new5a.Yylval.sval) {
				cp[0] = byte(c)
				cp = cp[1:]
			}
			c1++
		}

		if c1 > sizeof(new5a.Yylval.sval) {
			Yyerror("string constant too long")
		}
		return new5a.LSCONST

	case '\'':
		c = escchar('\'')
		if c == new5a.EOF {
			c = '\''
		}
		if escchar('\'') != new5a.EOF {
			Yyerror("missing '")
		}
		new5a.Yylval.lval = int32(c)
		return new5a.LCONST

	case '/':
		c1 = new5a.GETC()
		if c1 == '/' {
			for {
				c = new5a.GETC()
				if c == '\n' {
					goto l1
				}
				if c == new5a.EOF {
					Yyerror("eof in comment")
					Errorexit()
				}
			}
		}

		if c1 == '*' {
			for {
				c = new5a.GETC()
				for c == '*' {
					c = new5a.GETC()
					if c == '/' {
						goto l0
					}
				}

				if c == new5a.EOF {
					Yyerror("eof in comment")
					Errorexit()
				}

				if c == '\n' {
					new5a.Lineno++
				}
			}
		}

	default:
		return int32(c)
	}

	new5a.Peekc = c1
	return int32(c)
}

func getc() int {
	var c int

	c = new5a.Peekc
	if c != new5a.IGN {
		new5a.Peekc = new5a.IGN
		return c
	}

	c = new5a.GETC()
	if c == '\n' {
		new5a.Lineno++
	}
	if c == new5a.EOF {
		Yyerror("End of file")
		Errorexit()
	}

	return c
}

func getnsc() int {
	var c int

	for {
		c = getc()
		if !(main.Isspace(c) != 0) || c == '\n' {
			return c
		}
	}
}

func unget(c int) {
	new5a.Peekc = c
	if c == '\n' {
		new5a.Lineno--
	}
}

func escchar(e int) int {
	var c int
	var l int

loop:
	c = getc()
	if c == '\n' {
		Yyerror("newline in string")
		return new5a.EOF
	}

	if c != '\\' {
		if c == e {
			return new5a.EOF
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

		new5a.Peekc = c
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

func Pinit(f string) {
	var i int
	var s *new5a.Sym

	new5a.Lineno = 1
	newio()
	newfile(f, -1)
	new5a.Pc = 0
	new5a.Peekc = new5a.IGN
	new5a.Sym = 1
	for i = 0; i < new5a.NHASH; i++ {
		for s = new5a.Hash[i]; s != nil; s = s.Link {
			s.Macro = ""
		}
	}
}

func filbuf() int {
	var i *new5a.Io

loop:
	i = new5a.Iostack
	if i == nil {
		return new5a.EOF
	}
	if i.F < 0 {
		goto pop
	}
	new5a.Fi.c = main.Read(int(i.F), i.B, new5a.BUFSIZ) - 1
	if new5a.Fi.c < 0 {
		main.Close(int(i.F))
		obj.Linklinehist(new5a.Ctxt, int(new5a.Lineno), "", 0)
		goto pop
	}

	new5a.Fi.p = i.B[1:]
	return int(i.B[0]) & 0xff

pop:
	new5a.Iostack = i.Link
	i.Link = new5a.Iofree
	new5a.Iofree = i
	i = new5a.Iostack
	if i == nil {
		return new5a.EOF
	}
	new5a.Fi.p = i.P
	new5a.Fi.c = int(i.C)
	new5a.Fi.c--
	if new5a.Fi.c < 0 {
		goto loop
	}
	tmp8 := new5a.Fi.p
	new5a.Fi.p = new5a.Fi.p[1:]
	return int(tmp8[0]) & 0xff
}

func Yyerror(a string, args ...interface{}) {
	var buf string
	var arg []interface{}

	/*
	 * hack to intercept message from yaccpar
	 */
	if a == "syntax error" {

		Yyerror("syntax error, last name: %s", new5a.Symb)
		return
	}

	prfile(new5a.Lineno)
	main.Va_start(arg, a)
	obj.Vseprint(buf, buf[sizeof(buf):], a, arg)
	main.Va_end(arg)
	fmt.Printf("%s\n", buf)
	new5a.Nerrors++
	if new5a.Nerrors > 10 {
		fmt.Printf("too many errors\n")
		Errorexit()
	}
}

func prfile(l int32) {
	obj.Linkprfile(new5a.Ctxt, l)
}
