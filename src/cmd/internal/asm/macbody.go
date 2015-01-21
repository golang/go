// Inferno utils/cc/macbody
// http://code.Google.Com/p/inferno-os/source/browse/utils/cc/macbody
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
	"cmd/internal/obj"
	"fmt"
	"os"
	"strings"
)

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

func getsym() *Sym {
	var c int

	c = getnsc()
	if !isalpha(c) && c != '_' && c < 0x80 {
		unget(c)
		return nil
	}

	var buf bytes.Buffer
	for {
		buf.WriteByte(byte(c))
		c = getc()
		if isalnum(c) || c == '_' || c >= 0x80 {
			continue
		}
		unget(c)
		break
	}
	last = buf.String()
	return Lookup(last)
}

func getsymdots(dots *int) *Sym {
	var c int
	var s *Sym

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
	return Lookup("__VA_ARGS__")
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

func dodefine(cp string) {
	var s *Sym
	var p string

	if i := strings.Index(cp, "="); i >= 0 {
		p = cp[i+1:]
		cp = cp[:i]
		s = Lookup(cp)
		s.Macro = &Macro{Text: p}
	} else {
		s = Lookup(cp)
		s.Macro = &Macro{Text: "1"}
	}

	if debug['m'] != 0 {
		fmt.Printf("#define (-D) %s %s\n", s.Name, s.Macro.Text)
	}
}

var mactab = []struct {
	Macname string
	Macf    func()
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
	var s *Sym

	s = getsym()
	if s == nil {
		s = Lookup("endif")
	}
	for i = 0; i < len(mactab); i++ {
		if s.Name == mactab[i].Macname {
			if mactab[i].Macf != nil {
				mactab[i].Macf()
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
	var s *Sym

	s = getsym()
	macend()
	if s == nil {
		Yyerror("syntax in #undef")
		return
	}

	s.Macro = nil
}

const (
	NARG = 25
)

func macdef() {
	var s *Sym
	var a *Sym
	var args [NARG]string
	var n int
	var i int
	var c int
	var dots int
	var ischr int
	var base bytes.Buffer

	s = getsym()
	if s == nil {
		goto bad
	}
	if s.Macro != nil {
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

	if isspace(c) {
		if c != '\n' {
			c = getnsc()
		}
	}
	ischr = 0
	for {
		if isalpha(c) || c == '_' {
			var buf bytes.Buffer
			buf.WriteByte(byte(c))
			c = getc()
			for isalnum(c) || c == '_' {
				buf.WriteByte(byte(c))
				c = getc()
			}

			symb := buf.String()
			for i = 0; i < n; i++ {
				if symb == args[i] {
					break
				}
			}
			if i >= n {
				base.WriteString(symb)
				continue
			}

			base.WriteByte('#')
			base.WriteByte(byte('a' + i))
			continue
		}

		if ischr != 0 {
			if c == '\\' {
				base.WriteByte(byte(c))
				c = getc()
			} else if c == ischr {
				ischr = 0
			}
		} else {

			if c == '"' || c == '\'' {
				base.WriteByte(byte(c))
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

				base.WriteByte('/')
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

			base.WriteByte('\\')
			continue
		}

		if c == '\n' {
			break
		}
		if c == '#' {
			if n > 0 {
				base.WriteByte(byte(c))
			}
		}

		base.WriteByte(byte(c))
		c = GETC()
		if c == '\n' {
			Lineno++
		}
		if c == -1 {
			Yyerror("eof in a macro: %s", s.Name)
			break
		}
	}

	s.Macro = &Macro{
		Text: base.String(),
		Narg: n + 1,
		Dots: dots != 0,
	}
	if debug['m'] != 0 {
		fmt.Printf("#define %s %s\n", s.Name, s.Macro.Text)
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

func macexpand(s *Sym) []byte {
	var l int
	var c int
	var arg []string
	var out bytes.Buffer
	var buf bytes.Buffer
	var cp string

	if s.Macro.Narg == 0 {
		if debug['m'] != 0 {
			fmt.Printf("#expand %s %s\n", s.Name, s.Macro.Text)
		}
		return []byte(s.Macro.Text)
	}

	nargs := s.Macro.Narg - 1
	dots := s.Macro.Dots

	c = getnsc()
	if c != '(' {
		goto bad
	}
	c = getc()
	if c != ')' {
		unget(c)
		l = 0
		for {
			c = getc()
			if c == '"' {
				for {
					buf.WriteByte(byte(c))
					c = getc()
					if c == '\\' {
						buf.WriteByte(byte(c))
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
					buf.WriteByte(byte(c))
					c = getc()
					if c == '\\' {
						buf.WriteByte(byte(c))
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

					buf.WriteByte(' ')
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
					if len(arg) == nargs-1 && dots {
						buf.WriteByte(',')
						continue
					}

					arg = append(arg, buf.String())
					buf.Reset()
					continue
				}

				if c == ')' {
					arg = append(arg, buf.String())
					break
				}
			}

			if c == '\n' {
				c = ' '
			}
			buf.WriteByte(byte(c))
			if c == '(' {
				l++
			}
			if c == ')' {
				l--
			}
		}
	}

	if len(arg) != nargs {
		Yyerror("argument mismatch expanding: %s", s.Name)
		return nil
	}

	cp = s.Macro.Text
	for i := 0; i < len(cp); i++ {
		c = int(cp[i])
		if c == '\n' {
			c = ' '
		}
		if c != '#' {
			out.WriteByte(byte(c))
			continue
		}

		i++
		if i >= len(cp) {
			goto bad
		}
		c = int(cp[i])
		if c == '#' {
			out.WriteByte(byte(c))
			continue
		}

		c -= 'a'
		if c < 0 || c >= len(arg) {
			continue
		}
		out.WriteString(arg[c])
	}

	if debug['m'] != 0 {
		fmt.Printf("#expand %s %s\n", s.Name, out.String())
	}
	return out.Bytes()

bad:
	Yyerror("syntax in macro expansion: %s", s.Name)
	return nil
}

func macinc() {
	var c0 int
	var c int
	var i int
	var buf bytes.Buffer
	var f *os.File
	var hp string
	var str string
	var symb string

	c0 = getnsc()
	if c0 != '"' {
		c = c0
		if c0 != '<' {
			goto bad
		}
		c0 = '>'
	}

	for {
		c = getc()
		if c == c0 {
			break
		}
		if c == '\n' {
			goto bad
		}
		buf.WriteByte(byte(c))
	}
	str = buf.String()

	c = getcom()
	if c != '\n' {
		goto bad
	}

	for i = 0; i < len(include); i++ {
		if i == 0 && c0 == '>' {
			continue
		}
		symb = include[i]
		symb += "/"
		if symb == "./" {
			symb = ""
		}
		symb += str
		var err error
		f, err = os.Open(symb)
		if err == nil {
			break
		}
	}

	if f == nil {
		symb = str
	}
	hp = symb
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
	var c int
	var n int32
	var buf bytes.Buffer
	var symb string

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
			symb = "<noname>"
			goto nn
		}

		goto bad
	}

	for {
		c = getc()
		if c == '"' {
			break
		}
		buf.WriteByte(byte(c))
	}
	symb = buf.String()

	c = getcom()
	if c != '\n' {
		goto bad
	}

nn:
	obj.Linklinehist(Ctxt, int(Lineno), symb, int(n))
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
	var s *Sym

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
	if (s.Macro != nil) != (f != 0) {
		return
	}

skip:
	bol = 1
	l = 0
	for {
		c = getc()
		if c != '#' {
			if !isspace(c) {
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
	var s *Sym
	var c0 int
	var c int
	var buf bytes.Buffer
	var symb string

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

	for {
		c = getc()
		if c == c0 {
			break
		}
		if c == '\n' {
			goto bad
		}
		buf.WriteByte(byte(c))
	}
	symb = buf.String()

	c = getcom()
	if c != '\n' {
		goto bad
	}

	/*
	 * put pragma-line in as a funny history
	 */
	obj.Linklinehist(Ctxt, int(Lineno), symb, -1)
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
