// Inferno utils/cc/macbody
// http://code.google.com/p/inferno-os/source/browse/utils/cc/macbody
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

const (
	VARMAC = 0x80
)

func getnsn() int32 {
	var n int32
	var c int

	c = getnsc()
	if c < '0' || c > '9' {
		return -1
	}
	n = 0
	for c >= '0' && c <= '9' {
		n = n*10 + int32(c) - '0'
		c = getc()
	}

	unget(c)
	return n
}

func getsym() *new5a.Sym {
	var c int
	var cp string

	c = getnsc()
	if !(main.Isalpha(c) != 0) && c != '_' && c < 0x80 {
		unget(c)
		return nil
	}

	for cp = new5a.Symb; ; {
		if cp <= new5a.Symb[new5a.NSYMB-4:] {
			cp[0] = byte(c)
			cp = cp[1:]
		}
		c = getc()
		if main.Isalnum(c) != 0 || c == '_' || c >= 0x80 {
			continue
		}
		unget(c)
		break
	}

	cp = ""
	if cp > new5a.Symb[new5a.NSYMB-4:] {
		Yyerror("symbol too large: %s", new5a.Symb)
	}
	return lookup()
}

func getsymdots(dots *int) *new5a.Sym {
	var c int
	var s *new5a.Sym

	s = getsym()
	if s != nil {
		return s
	}

	c = getnsc()
	if c != '.' {
		unget(c)
		return nil
	}

	if getc() != '.' || getc() != '.' {
		Yyerror("bad dots in macro")
	}
	*dots = 1
	return Slookup("__VA_ARGS__")
}

func getcom() int {
	var c int

	for {
		c = getnsc()
		if c != '/' {
			break
		}
		c = getc()
		if c == '/' {
			for c != '\n' {
				c = getc()
			}
			break
		}

		if c != '*' {
			break
		}
		c = getc()
		for {
			if c == '*' {
				c = getc()
				if c != '/' {
					continue
				}
				c = getc()
				break
			}

			if c == '\n' {
				Yyerror("comment across newline")
				break
			}

			c = getc()
		}

		if c == '\n' {
			break
		}
	}

	return c
}

func Dodefine(cp string) {
	var s *new5a.Sym
	var p string
	var l int32

	Ensuresymb(int32(len(cp)))
	new5a.Symb = cp
	p = main.Strchr(new5a.Symb, '=')
	if p != "" {
		p = ""
		p = p[1:]
		s = lookup()
		l = int32(len(p)) + 2 /* +1 null, +1 nargs */
		s.Macro = Alloc(l).(string)
		s.Macro[1:] = p
	} else {

		s = lookup()
		s.Macro = "\0001" /* \000 is nargs */
	}

	if new5a.Debug['m'] != 0 {
		fmt.Printf("#define (-D) %s %s\n", s.Name, s.Macro[1:])
	}
}

var mactab = []struct {
	macname string
	macf    func()
}{
	{"ifdef", nil},  /* macif(0) */
	{"ifndef", nil}, /* macif(1) */
	{"else", nil},   /* macif(2) */
	{"line", maclin},
	{"define", macdef},
	{"include", macinc},
	{"undef", macund},
	{"pragma", macprag},
	{"endif", macend},
}

func domacro() {
	var i int
	var s *new5a.Sym

	s = getsym()
	if s == nil {
		s = Slookup("endif")
	}
	for i = 0; mactab[i].macname != ""; i++ {
		if s.Name == mactab[i].macname {
			if mactab[i].macf != nil {
				(*mactab[i].macf)()
			} else {

				macif(i)
			}
			return
		}
	}

	Yyerror("unknown #: %s", s.Name)
	macend()
}

func macund() {
	var s *new5a.Sym

	s = getsym()
	macend()
	if s == nil {
		Yyerror("syntax in #undef")
		return
	}

	s.Macro = ""
}

const (
	NARG = 25
)

func macdef() {
	var s *new5a.Sym
	var a *new5a.Sym
	var args [NARG]string
	var np string
	var base string
	var n int
	var i int
	var c int
	var len int
	var dots int
	var ischr int

	s = getsym()
	if s == nil {
		goto bad
	}
	if s.Macro != "" {
		Yyerror("macro redefined: %s", s.Name)
	}
	c = getc()
	n = -1
	dots = 0
	if c == '(' {
		n++
		c = getnsc()
		if c != ')' {
			unget(c)
			for {
				a = getsymdots(&dots)
				if a == nil {
					goto bad
				}
				if n >= NARG {
					Yyerror("too many arguments in #define: %s", s.Name)
					goto bad
				}

				args[n] = a.Name
				n++
				c = getnsc()
				if c == ')' {
					break
				}
				if c != ',' || dots != 0 {
					goto bad
				}
			}
		}

		c = getc()
	}

	if main.Isspace(c) != 0 {
		if c != '\n' {
			c = getnsc()
		}
	}
	base = new5a.Hunk
	len = 1
	ischr = 0
	for {
		if main.Isalpha(c) != 0 || c == '_' {
			np = new5a.Symb
			np[0] = byte(c)
			np = np[1:]
			c = getc()
			for main.Isalnum(c) != 0 || c == '_' {
				np[0] = byte(c)
				np = np[1:]
				c = getc()
			}

			np = ""
			for i = 0; i < n; i++ {
				if new5a.Symb == args[i] {
					break
				}
			}
			if i >= n {
				i = len(new5a.Symb)
				base = Allocn(base, int32(len), int32(i)).(string)
				main.Memmove(base[len:], new5a.Symb, i)
				len += i
				continue
			}

			base = Allocn(base, int32(len), 2).(string)
			base[len] = '#'
			len++
			base[len] = byte('a' + i)
			len++
			continue
		}

		if ischr != 0 {
			if c == '\\' {
				base = Allocn(base, int32(len), 1).(string)
				base[len] = byte(c)
				len++
				c = getc()
			} else if c == ischr {
				ischr = 0
			}
		} else {

			if c == '"' || c == '\'' {
				base = Allocn(base, int32(len), 1).(string)
				base[len] = byte(c)
				len++
				ischr = c
				c = getc()
				continue
			}

			if c == '/' {
				c = getc()
				if c == '/' {
					c = getc()
					for {
						if c == '\n' {
							break
						}
						c = getc()
					}

					continue
				}

				if c == '*' {
					c = getc()
					for {
						if c == '*' {
							c = getc()
							if c != '/' {
								continue
							}
							c = getc()
							break
						}

						if c == '\n' {
							Yyerror("comment and newline in define: %s", s.Name)
							break
						}

						c = getc()
					}

					continue
				}

				base = Allocn(base, int32(len), 1).(string)
				base[len] = '/'
				len++
				continue
			}
		}

		if c == '\\' {
			c = getc()
			if c == '\n' {
				c = getc()
				continue
			} else if c == '\r' {
				c = getc()
				if c == '\n' {
					c = getc()
					continue
				}
			}

			base = Allocn(base, int32(len), 1).(string)
			base[len] = '\\'
			len++
			continue
		}

		if c == '\n' {
			break
		}
		if c == '#' {
			if n > 0 {
				base = Allocn(base, int32(len), 1).(string)
				base[len] = byte(c)
				len++
			}
		}

		base = Allocn(base, int32(len), 1).(string)
		base[len] = byte(c)
		len++
		new5a.Fi.c--
		var tmp C.int
		if new5a.Fi.c < 0 {
			tmp = C.int(filbuf())
		} else {
			tmp = new5a.Fi.p[0] & 0xff
		}
		c = int(tmp)
		if c == '\n' {
			new5a.Lineno++
		}
		if c == -1 {
			Yyerror("eof in a macro: %s", s.Name)
			break
		}
	}

	for {
		base = Allocn(base, int32(len), 1).(string)
		base[len] = 0
		len++
		if !(len&3 != 0 /*untyped*/) {
			break
		}
	}

	base[0] = byte(n + 1)
	if dots != 0 {
		base[0] |= VARMAC
	}
	s.Macro = base
	if new5a.Debug['m'] != 0 {
		fmt.Printf("#define %s %s\n", s.Name, s.Macro[1:])
	}
	return

bad:
	if s == nil {
		Yyerror("syntax in #define")
	} else {

		Yyerror("syntax in #define: %s", s.Name)
	}
	macend()
}

func macexpand(s *new5a.Sym, b string) {
	var buf string
	var n int
	var l int
	var c int
	var nargs int
	var arg [NARG]string
	var cp string
	var ob string
	var ecp string
	var dots int8

	ob = b
	if s.Macro[0] == 0 {
		b = s.Macro[1:]
		if new5a.Debug['m'] != 0 {
			fmt.Printf("#expand %s %s\n", s.Name, ob)
		}
		return
	}

	nargs = int(int8(s.Macro[0]&^VARMAC)) - 1
	dots = int8(s.Macro[0] & VARMAC)

	c = getnsc()
	if c != '(' {
		goto bad
	}
	n = 0
	c = getc()
	if c != ')' {
		unget(c)
		l = 0
		cp = buf
		ecp = cp[sizeof(buf)-4:]
		arg[n] = cp
		n++
		for {
			if cp >= ecp {
				goto toobig
			}
			c = getc()
			if c == '"' {
				for {
					if cp >= ecp {
						goto toobig
					}
					cp[0] = byte(c)
					cp = cp[1:]
					c = getc()
					if c == '\\' {
						cp[0] = byte(c)
						cp = cp[1:]
						c = getc()
						continue
					}

					if c == '\n' {
						goto bad
					}
					if c == '"' {
						break
					}
				}
			}

			if c == '\'' {
				for {
					if cp >= ecp {
						goto toobig
					}
					cp[0] = byte(c)
					cp = cp[1:]
					c = getc()
					if c == '\\' {
						cp[0] = byte(c)
						cp = cp[1:]
						c = getc()
						continue
					}

					if c == '\n' {
						goto bad
					}
					if c == '\'' {
						break
					}
				}
			}

			if c == '/' {
				c = getc()
				switch c {
				case '*':
					for {
						c = getc()
						if c == '*' {
							c = getc()
							if c == '/' {
								break
							}
						}
					}

					cp[0] = ' '
					cp = cp[1:]
					continue

				case '/':
					for {
						c = getc()
						if !(c != '\n') {
							break
						}
					}

				default:
					unget(c)
					c = '/'
				}
			}

			if l == 0 {
				if c == ',' {
					if n == nargs && dots != 0 {
						cp[0] = ','
						cp = cp[1:]
						continue
					}

					cp = ""
					cp = cp[1:]
					arg[n] = cp
					n++
					if n > nargs {
						break
					}
					continue
				}

				if c == ')' {
					break
				}
			}

			if c == '\n' {
				c = ' '
			}
			cp[0] = byte(c)
			cp = cp[1:]
			if c == '(' {
				l++
			}
			if c == ')' {
				l--
			}
		}

		cp = ""
	}

	if n != nargs {
		Yyerror("argument mismatch expanding: %s", s.Name)
		b = ""
		return
	}

	cp = s.Macro[1:]
	for {
		c = int(cp[0])
		cp = cp[1:]
		if c == '\n' {
			c = ' '
		}
		if c != '#' {
			b[0] = byte(c)
			b = b[1:]
			if c == 0 {
				break
			}
			continue
		}

		c = int(cp[0])
		cp = cp[1:]
		if c == 0 {
			goto bad
		}
		if c == '#' {
			b[0] = byte(c)
			b = b[1:]
			continue
		}

		c -= 'a'
		if c < 0 || c >= n {
			continue
		}
		b = arg[c]
		b = b[len(arg[c]):]
	}

	b = ""
	if new5a.Debug['m'] != 0 {
		fmt.Printf("#expand %s %s\n", s.Name, ob)
	}
	return

bad:
	Yyerror("syntax in macro expansion: %s", s.Name)
	b = ""
	return

toobig:
	Yyerror("too much text in macro expansion: %s", s.Name)
	b = ""
}

func macinc() {
	var c0 int
	var c int
	var i int
	var f int
	var str string
	var hp string

	c0 = getnsc()
	if c0 != '"' {
		c = c0
		if c0 != '<' {
			goto bad
		}
		c0 = '>'
	}

	for hp = str; ; {
		c = getc()
		if c == c0 {
			break
		}
		if c == '\n' {
			goto bad
		}
		hp[0] = byte(c)
		hp = hp[1:]
	}

	hp = ""

	c = getcom()
	if c != '\n' {
		goto bad
	}

	f = -1
	for i = 0; i < new5a.Ninclude; i++ {
		if i == 0 && c0 == '>' {
			continue
		}
		Ensuresymb(int32(len(new5a.Include[i])) + int32(len(str)) + 2)
		new5a.Symb = new5a.Include[i]
		new5a.Symb += "/"
		if new5a.Symb == "./" {
			new5a.Symb = ""
		}
		new5a.Symb += str
		f = main.Open(new5a.Symb, main.OREAD)
		if f >= 0 {
			break
		}
	}

	if f < 0 {
		new5a.Symb = str
	}
	c = len(new5a.Symb) + 1
	hp = Alloc(int32(c)).(string)
	main.Memmove(hp, new5a.Symb, c)
	newio()
	pushio()
	newfile(hp, f)
	return

bad:
	unget(c)
	Yyerror("syntax in #include")
	macend()
}

func maclin() {
	var cp string
	var c int
	var n int32

	n = getnsn()
	c = getc()
	if n < 0 {
		goto bad
	}

	for {
		if c == ' ' || c == '\t' {
			c = getc()
			continue
		}

		if c == '"' {
			break
		}
		if c == '\n' {
			new5a.Symb = "<noname>"
			goto nn
		}

		goto bad
	}

	cp = new5a.Symb
	for {
		c = getc()
		if c == '"' {
			break
		}
		cp[0] = byte(c)
		cp = cp[1:]
	}

	cp = ""
	c = getcom()
	if c != '\n' {
		goto bad
	}

nn:
	c = len(new5a.Symb) + 1
	cp = Alloc(int32(c)).(string)
	main.Memmove(cp, new5a.Symb, c)
	obj.Linklinehist(new5a.Ctxt, int(new5a.Lineno), cp, int(n))
	return

bad:
	unget(c)
	Yyerror("syntax in #line")
	macend()
}

func macif(f int) {
	var c int
	var l int
	var bol int
	var s *new5a.Sym

	if f == 2 {
		goto skip
	}
	s = getsym()
	if s == nil {
		goto bad
	}
	if getcom() != '\n' {
		goto bad
	}
	if (s.Macro != "")^f != 0 /*untyped*/ {
		return
	}

skip:
	bol = 1
	l = 0
	for {
		c = getc()
		if c != '#' {
			if !(main.Isspace(c) != 0) {
				bol = 0
			}
			if c == '\n' {
				bol = 1
			}
			continue
		}

		if !(bol != 0) {
			continue
		}
		s = getsym()
		if s == nil {
			continue
		}
		if s.Name == "endif" {
			if l != 0 {
				l--
				continue
			}

			macend()
			return
		}

		if s.Name == "ifdef" || s.Name == "ifndef" {
			l++
			continue
		}

		if l == 0 && f != 2 && s.Name == "else" {
			macend()
			return
		}
	}

bad:
	Yyerror("syntax in #if(n)def")
	macend()
}

func macprag() {
	var s *new5a.Sym
	var c0 int
	var c int
	var hp string

	s = getsym()

	if s != nil && s.Name == "lib" {
		goto praglib
	}
	if s != nil && s.Name == "pack" {
		pragpack()
		return
	}

	if s != nil && s.Name == "fpround" {
		pragfpround()
		return
	}

	if s != nil && s.Name == "textflag" {
		pragtextflag()
		return
	}

	if s != nil && s.Name == "dataflag" {
		pragdataflag()
		return
	}

	if s != nil && s.Name == "varargck" {
		pragvararg()
		return
	}

	if s != nil && s.Name == "incomplete" {
		pragincomplete()
		return
	}

	if s != nil && (strings.HasPrefix(s.Name, "cgo_") || strings.HasPrefix(s.Name, "dyn")) {
		pragcgo(s.Name)
		return
	}

	for getnsc() != '\n' {

	}
	return

praglib:
	c0 = getnsc()
	if c0 != '"' {
		c = c0
		if c0 != '<' {
			goto bad
		}
		c0 = '>'
	}

	for hp = new5a.Symb; ; {
		c = getc()
		if c == c0 {
			break
		}
		if c == '\n' {
			goto bad
		}
		hp[0] = byte(c)
		hp = hp[1:]
	}

	hp = ""
	c = getcom()
	if c != '\n' {
		goto bad
	}

	/*
	 * put pragma-line in as a funny history
	 */
	c = len(new5a.Symb) + 1

	hp = Alloc(int32(c)).(string)
	main.Memmove(hp, new5a.Symb, c)

	obj.Linklinehist(new5a.Ctxt, int(new5a.Lineno), hp, -1)
	return

bad:
	unget(c)
	Yyerror("syntax in #pragma lib")
	macend()
}

func macend() {
	var c int

	for {
		c = getnsc()
		if c < 0 || c == '\n' {
			return
		}
	}
}
