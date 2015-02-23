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
	sc = extra[(s&C_SCOND)^C_SCOND_XOR]
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
		if p.From.Type == obj.TYPE_CONST {
			str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, RAconv(&p.From), Dconv(p, 0, &p.To))
		} else if p.To.Type == obj.TYPE_CONST {
			str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), RAconv(&p.To))
		} else {
			str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))
		}
	} else if a == obj.ADATA {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v/%d,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.From3.Offset, Dconv(p, 0, &p.To))
	} else if p.As == obj.ATEXT {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%d,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.From3.Offset, Dconv(p, 0, &p.To))
	} else if p.Reg == 0 {
		str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))
	} else {
		str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v,%v", p.Pc, p.Line(), Aconv(a), sc, Dconv(p, 0, &p.From), Rconv(int(p.Reg)), Dconv(p, 0, &p.To))
	}

	fp += str
	return fp
}

func Aconv(a int) string {
	var s string
	var fp string

	s = "???"
	if a >= obj.AXXX && a < ALAST {
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

	case obj.TYPE_NONE:
		str = ""
		if a.Name != obj.TYPE_NONE || a.Reg != 0 || a.Sym != nil {
			str = fmt.Sprintf("%v(%v)(NONE)", Mconv(a), Rconv(int(a.Reg)))
		}

	case obj.TYPE_CONST,
		obj.TYPE_ADDR:
		if a.Reg != 0 {
			str = fmt.Sprintf("$%v(%v)", Mconv(a), Rconv(int(a.Reg)))
		} else {
			str = fmt.Sprintf("$%v", Mconv(a))
		}

	case obj.TYPE_TEXTSIZE:
		if a.U.Argsize == obj.ArgsSizeUnknown {
			str = fmt.Sprintf("$%d", a.Offset)
		} else {
			str = fmt.Sprintf("$%d-%d", a.Offset, a.U.Argsize)
		}

	case obj.TYPE_SHIFT:
		v = int(a.Offset)
		op = string("<<>>->@>"[((v>>5)&3)<<1:])
		if v&(1<<4) != 0 {
			str = fmt.Sprintf("R%d%c%cR%d", v&15, op[0], op[1], (v>>8)&15)
		} else {
			str = fmt.Sprintf("R%d%c%c%d", v&15, op[0], op[1], (v>>7)&31)
		}
		if a.Reg != 0 {
			str += fmt.Sprintf("(%v)", Rconv(int(a.Reg)))
		}

	case obj.TYPE_MEM:
		if a.Reg != 0 {
			str = fmt.Sprintf("%v(%v)", Mconv(a), Rconv(int(a.Reg)))
		} else {
			str = fmt.Sprintf("%v", Mconv(a))
		}

	case obj.TYPE_REG:
		str = fmt.Sprintf("%v", Rconv(int(a.Reg)))
		if a.Name != obj.TYPE_NONE || a.Sym != nil {
			str = fmt.Sprintf("%v(%v)(REG)", Mconv(a), Rconv(int(a.Reg)))
		}

	case obj.TYPE_BRANCH:
		if a.Sym != nil {
			str = fmt.Sprintf("%s(SB)", a.Sym.Name)
		} else if p != nil && p.Pcond != nil {
			str = fmt.Sprintf("%d", p.Pcond.Pc)
		} else if a.U.Branch != nil {
			str = fmt.Sprintf("%d", a.U.Branch.Pc)
		} else {
			str = fmt.Sprintf("%d(PC)", a.Offset) /*-pc*/
		}

	case obj.TYPE_FCONST:
		str = fmt.Sprintf("$%.17g", a.U.Dval)

	case obj.TYPE_SCONST:
		str = fmt.Sprintf("$%q", a.U.Sval)

	case obj.TYPE_REGREG:
		str = fmt.Sprintf("(%v, %v)", Rconv(int(a.Reg)), Rconv(int(a.Offset)))

	case obj.TYPE_REGREG2:
		str = fmt.Sprintf("%v, %v", Rconv(int(a.Reg)), Rconv(int(a.Offset)))
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
	case obj.TYPE_CONST:
		if a.Reg != 0 {
			break
		}
		if a.Sym != nil {
			break
		}
		v = int(a.Offset)
		str = ""
		for i = 0; i < NREG; i++ {
			if v&(1<<uint(i)) != 0 {
				if str == "" {
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

	if r == 0 {
		fp += "NONE"
		return fp
	}
	if REG_R0 <= r && r <= REG_R15 {
		fp += fmt.Sprintf("R%d", r-REG_R0)
		return fp
	}
	if REG_F0 <= r && r <= REG_F15 {
		fp += fmt.Sprintf("F%d", r-REG_F0)
		return fp
	}

	switch r {
	case REG_FPSR:
		fp += "FPSR"
		return fp

	case REG_FPCR:
		fp += "FPCR"
		return fp

	case REG_CPSR:
		fp += "CPSR"
		return fp

	case REG_SPSR:
		fp += "SPSR"
		return fp
	}

	fp += fmt.Sprintf("badreg(%d)", r)
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

	case obj.NAME_NONE:
		str = fmt.Sprintf("%d", a.Offset)

	case obj.NAME_EXTERN:
		str = fmt.Sprintf("%s+%d(SB)", s.Name, int(a.Offset))

	case obj.NAME_STATIC:
		str = fmt.Sprintf("%s<>+%d(SB)", s.Name, int(a.Offset))

	case obj.NAME_AUTO:
		str = fmt.Sprintf("%s-%d(SP)", s.Name, int(-a.Offset))

	case obj.NAME_PARAM:
		str = fmt.Sprintf("%s+%d(FP)", s.Name, int(a.Offset))
	}

out:
	fp += str
	return fp
}
