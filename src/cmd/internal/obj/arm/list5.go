// Inferno utils/5c/list.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/list.c
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

package arm

import (
	"cmd/internal/obj"
	"fmt"
)

const (
	STRINGSZ = 1000
)

var extra = []string{
	".EQ",
	".NE",
	".CS",
	".CC",
	".MI",
	".PL",
	".VS",
	".VC",
	".HI",
	".LS",
	".GE",
	".LT",
	".GT",
	".LE",
	"",
	".NV",
}

var bigP *obj.Prog

func Pconv(p *obj.Prog) string {
	var str string
	var sc string
	var fp string

	var a int
	var s int

	a = int(p.As)
	s = int(p.Scond)
	sc = extra[s&C_SCOND]
	if s&C_SBIT != 0 {
		sc += ".S"
	}
	if s&C_PBIT != 0 {
		sc += ".P"
	}
	if s&C_WBIT != 0 {
		sc += ".W"
	}
	if s&C_UBIT != 0 { /* ambiguous with FBIT */
		sc += ".U"
	}
	if a == AMOVM {
		if p.From.Type == D_CONST {
			str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, RAconv(&p.From), Dconv(p, 0, &p.To))
		} else if p.To.Type == D_CONST {
			str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), RAconv(&p.To))
		} else {

			str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))
		}
	} else if a == ADATA {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v/%d,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.Reg, Dconv(p, 0, &p.To))
	} else if p.As == ATEXT {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%d,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.Reg, Dconv(p, 0, &p.To))
	} else if p.Reg == NREG {
		str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))
	} else if p.From.Type != D_FREG {
		str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,R%d,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), p.Reg, Dconv(p, 0, &p.To))
	} else {

		str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,F%d,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), p.Reg, Dconv(p, 0, &p.To))
	}

	fp += str
	return fp
}

func Aconv(a int) string {
	var s string
	var fp string

	s = "???"
	if a >= AXXX && a < ALAST {
		s = Anames[a]
	}
	fp += s
	return fp
}

func Dconv(p *obj.Prog, flag int, a *obj.Addr) string {
	var str string
	var fp string

	var op string
	var v int

	switch a.Type {
	default:
		str = fmt.Sprintf("GOK-type(%d)", a.Type)

	case D_NONE:
		str = ""
		if a.Name != D_NONE || a.Reg != NREG || a.Sym != nil {
			str = fmt.Sprintf("%v(R%d)(NONE)", Mconv(a), a.Reg)
		}

	case D_CONST:
		if a.Reg != NREG {
			str = fmt.Sprintf("$%v(R%d)", Mconv(a), a.Reg)
		} else {

			str = fmt.Sprintf("$%v", Mconv(a))
		}

	case D_CONST2:
		str = fmt.Sprintf("$%d-%d", a.Offset, a.Offset2)

	case D_SHIFT:
		v = int(a.Offset)
		op = string("<<>>->@>"[((v>>5)&3)<<1:])
		if v&(1<<4) != 0 {
			str = fmt.Sprintf("R%d%c%cR%d", v&15, op[0], op[1], (v>>8)&15)
		} else {

			str = fmt.Sprintf("R%d%c%c%d", v&15, op[0], op[1], (v>>7)&31)
		}
		if a.Reg != NREG {
			str += fmt.Sprintf("(R%d)", a.Reg)
		}

	case D_OREG:
		if a.Reg != NREG {
			str = fmt.Sprintf("%v(R%d)", Mconv(a), a.Reg)
		} else {

			str = fmt.Sprintf("%v", Mconv(a))
		}

	case D_REG:
		str = fmt.Sprintf("R%d", a.Reg)
		if a.Name != D_NONE || a.Sym != nil {
			str = fmt.Sprintf("%v(R%d)(REG)", Mconv(a), a.Reg)
		}

	case D_FREG:
		str = fmt.Sprintf("F%d", a.Reg)
		if a.Name != D_NONE || a.Sym != nil {
			str = fmt.Sprintf("%v(R%d)(REG)", Mconv(a), a.Reg)
		}

	case D_PSR:
		str = fmt.Sprintf("PSR")
		if a.Name != D_NONE || a.Sym != nil {
			str = fmt.Sprintf("%v(PSR)(REG)", Mconv(a))
		}

	case D_BRANCH:
		if a.Sym != nil {
			str = fmt.Sprintf("%s(SB)", a.Sym.Name)
		} else if p != nil && p.Pcond != nil {
			str = fmt.Sprintf("%d", p.Pcond.Pc)
		} else if a.U.Branch != nil {
			str = fmt.Sprintf("%d", a.U.Branch.Pc)
		} else {

			str = fmt.Sprintf("%d(PC)", a.Offset) /*-pc*/
		}

	case D_FCONST:
		str = fmt.Sprintf("$%.17g", a.U.Dval)

	case D_SCONST:
		str = fmt.Sprintf("$\"%q\"", a.U.Sval)
		break
	}

	fp += str
	return fp
}

func RAconv(a *obj.Addr) string {
	var str string
	var fp string

	var i int
	var v int

	str = fmt.Sprintf("GOK-reglist")
	switch a.Type {
	case D_CONST,
		D_CONST2:
		if a.Reg != NREG {
			break
		}
		if a.Sym != nil {
			break
		}
		v = int(a.Offset)
		str = ""
		for i = 0; i < NREG; i++ {
			if v&(1<<uint(i)) != 0 {
				if str[0] == 0 {
					str += "[R"
				} else {

					str += ",R"
				}
				str += fmt.Sprintf("%d", i)
			}
		}

		str += "]"
	}

	fp += str
	return fp
}

func Rconv(r int) string {
	var fp string

	var str string

	str = fmt.Sprintf("R%d", r)
	fp += str
	return fp
}

func DRconv(a int) string {
	var s string
	var fp string

	s = "C_??"
	if a >= C_NONE && a <= C_NCLASS {
		s = cnames5[a]
	}
	fp += s
	return fp
}

func Mconv(a *obj.Addr) string {
	var str string
	var fp string

	var s *obj.LSym

	s = a.Sym
	if s == nil {
		str = fmt.Sprintf("%d", int(a.Offset))
		goto out
	}

	switch a.Name {
	default:
		str = fmt.Sprintf("GOK-name(%d)", a.Name)

	case D_NONE:
		str = fmt.Sprintf("%d", a.Offset)

	case D_EXTERN:
		str = fmt.Sprintf("%s+%d(SB)", s.Name, int(a.Offset))

	case D_STATIC:
		str = fmt.Sprintf("%s<>+%d(SB)", s.Name, int(a.Offset))

	case D_AUTO:
		str = fmt.Sprintf("%s-%d(SP)", s.Name, int(-a.Offset))

	case D_PARAM:
		str = fmt.Sprintf("%s+%d(FP)", s.Name, int(a.Offset))
		break
	}

out:
	fp += str
	return fp
}
