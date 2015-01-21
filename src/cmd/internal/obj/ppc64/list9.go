// cmd/9l/list.c from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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

package ppc64

import (
	"cmd/internal/obj"
	"fmt"
)

const (
	STRINGSZ = 1000
)

//
// Format conversions
//	%A int		Opcodes (instruction mnemonics)
//
//	%D Addr*	Addresses (instruction operands)
//		Flags: "%lD": seperate the high and low words of a constant by "-"
//
//	%P Prog*	Instructions
//
//	%R int		Registers
//
//	%$ char*	String constant addresses (for internal use only)
//	%^ int   	C_* classes (for liblink internal use)

var bigP *obj.Prog

func Pconv(p *obj.Prog) string {
	var str string
	var fp string

	var a int
	var ch int

	a = int(p.As)

	if a == ADATA || a == AINIT || a == ADYNT {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v/%d,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.Reg, Dconv(p, 0, &p.To))
	} else if a == ATEXT {
		if p.Reg != 0 {
			str = fmt.Sprintf("%.5d (%v)        %v      %v,%d,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.Reg, Dconv(p, fmtLong, &p.To))
		} else {

			str = fmt.Sprintf("%.5d (%v)        %v      %v,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), Dconv(p, fmtLong, &p.To))
		}
	} else if a == AGLOBL {
		if p.Reg != 0 {
			str = fmt.Sprintf("%.5d (%v)        %v      %v,%d,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.Reg, Dconv(p, 0, &p.To))
		} else {

			str = fmt.Sprintf("%.5d (%v)        %v      %v,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))
		}
	} else {

		if p.Mark&NOSCHED != 0 {
			str += fmt.Sprintf("*")
		}
		if p.Reg == NREG && p.From3.Type == D_NONE {
			str += fmt.Sprintf("%.5d (%v)\t%v\t%v,%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))
		} else if a != ATEXT && p.From.Type == D_OREG {
			str += fmt.Sprintf("%.5d (%v)\t%v\t%d(R%d+R%d),%v", p.Pc, p.Line(), Aconv(a), p.From.Offset, p.From.Reg, p.Reg, Dconv(p, 0, &p.To))
		} else if p.To.Type == D_OREG {
			str += fmt.Sprintf("%.5d (%v)\t%v\t%v,%d(R%d+R%d)", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From), p.To.Offset, p.To.Reg, p.Reg)
		} else {

			str += fmt.Sprintf("%.5d (%v)\t%v\t%v", p.Pc, p.Line(), Aconv(a), Dconv(p, 0, &p.From))
			if p.Reg != NREG {
				ch = 'R'
				if p.From.Type == D_FREG {
					ch = 'F'
				}
				str += fmt.Sprintf(",%c%d", ch, p.Reg)
			}

			if p.From3.Type != D_NONE {
				str += fmt.Sprintf(",%v", Dconv(p, 0, &p.From3))
			}
			str += fmt.Sprintf(",%v", Dconv(p, 0, &p.To))
		}

		if p.Spadj != 0 {
			fp += fmt.Sprintf("%s # spadj=%d", str, p.Spadj)
			return fp
		}
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

	var v int32

	if flag&fmtLong != 0 /*untyped*/ {
		if a.Type == D_CONST {
			str = fmt.Sprintf("$%d-%d", int32(a.Offset), int32(a.Offset>>32))
		} else {

			// ATEXT dst is not constant
			str = fmt.Sprintf("!!%v", Dconv(p, 0, a))
		}

		goto ret
	}

	switch a.Type {
	default:
		str = fmt.Sprintf("GOK-type(%d)", a.Type)

	case D_NONE:
		str = ""
		if a.Name != D_NONE || a.Reg != NREG || a.Sym != nil {
			str = fmt.Sprintf("%v(R%d)(NONE)", Mconv(a), a.Reg)
		}

	case D_CONST,
		D_DCONST:
		if a.Reg != NREG {
			str = fmt.Sprintf("$%v(R%d)", Mconv(a), a.Reg)
		} else {

			str = fmt.Sprintf("$%v", Mconv(a))
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
			str = fmt.Sprintf("%v(F%d)(REG)", Mconv(a), a.Reg)
		}

	case D_CREG:
		if a.Reg == NREG {
			str = "CR"
		} else {

			str = fmt.Sprintf("CR%d", a.Reg)
		}
		if a.Name != D_NONE || a.Sym != nil {
			str = fmt.Sprintf("%v(C%d)(REG)", Mconv(a), a.Reg)
		}

	case D_SPR:
		if a.Name == D_NONE && a.Sym == nil {
			switch uint32(a.Offset) {
			case D_XER:
				str = fmt.Sprintf("XER")
			case D_LR:
				str = fmt.Sprintf("LR")
			case D_CTR:
				str = fmt.Sprintf("CTR")
			default:
				str = fmt.Sprintf("SPR(%d)", a.Offset)
				break
			}

			break
		}

		str = fmt.Sprintf("SPR-GOK(%d)", a.Reg)
		if a.Name != D_NONE || a.Sym != nil {
			str = fmt.Sprintf("%v(SPR-GOK%d)(REG)", Mconv(a), a.Reg)
		}

	case D_DCR:
		if a.Name == D_NONE && a.Sym == nil {
			str = fmt.Sprintf("DCR(%d)", a.Offset)
			break
		}

		str = fmt.Sprintf("DCR-GOK(%d)", a.Reg)
		if a.Name != D_NONE || a.Sym != nil {
			str = fmt.Sprintf("%v(DCR-GOK%d)(REG)", Mconv(a), a.Reg)
		}

	case D_OPT:
		str = fmt.Sprintf("OPT(%d)", a.Reg)

	case D_FPSCR:
		if a.Reg == NREG {
			str = "FPSCR"
		} else {

			str = fmt.Sprintf("FPSCR(%d)", a.Reg)
		}

	case D_MSR:
		str = fmt.Sprintf("MSR")

	case D_BRANCH:
		if p.Pcond != nil {
			v = int32(p.Pcond.Pc)

			//if(v >= INITTEXT)
			//	v -= INITTEXT-HEADR;
			if a.Sym != nil {

				str = fmt.Sprintf("%s+%.5x(BRANCH)", a.Sym.Name, uint32(v))
			} else {

				str = fmt.Sprintf("%.5x(BRANCH)", uint32(v))
			}
		} else if a.U.Branch != nil {
			str = fmt.Sprintf("%d", a.U.Branch.Pc)
		} else if a.Sym != nil {
			str = fmt.Sprintf("%s+%d(APC)", a.Sym.Name, a.Offset)
		} else {

			str = fmt.Sprintf("%d(APC)", a.Offset)
		}

		//sprint(str, "$%lux-%lux", a->ieee.h, a->ieee.l);
	case D_FCONST:
		str = fmt.Sprintf("$%.17g", a.U.Dval)

	case D_SCONST:
		str = fmt.Sprintf("$\"%q\"", a.U.Sval)
		break
	}

ret:
	fp += str
	return fp
}

func Mconv(a *obj.Addr) string {
	var str string
	var fp string

	var s *obj.LSym
	var l int32

	s = a.Sym

	//if(s == nil) {
	//	l = a->offset;
	//	if((vlong)l != a->offset)
	//		sprint(str, "0x%llux", a->offset);
	//	else
	//		sprint(str, "%lld", a->offset);
	//	goto out;
	//}
	switch a.Name {

	default:
		str = fmt.Sprintf("GOK-name(%d)", a.Name)

	case D_NONE:
		l = int32(a.Offset)
		if int64(l) != a.Offset {
			str = fmt.Sprintf("0x%x", uint64(a.Offset))
		} else {

			str = fmt.Sprintf("%d", a.Offset)
		}

	case D_EXTERN:
		if a.Offset != 0 {
			str = fmt.Sprintf("%s+%d(SB)", s.Name, a.Offset)
		} else {

			str = fmt.Sprintf("%s(SB)", s.Name)
		}

	case D_STATIC:
		str = fmt.Sprintf("%s<>+%d(SB)", s.Name, a.Offset)

	case D_AUTO:
		if s == nil {
			str = fmt.Sprintf("%d(SP)", -a.Offset)
		} else {

			str = fmt.Sprintf("%s-%d(SP)", s.Name, -a.Offset)
		}

	case D_PARAM:
		if s == nil {
			str = fmt.Sprintf("%d(FP)", a.Offset)
		} else {

			str = fmt.Sprintf("%s+%d(FP)", s.Name, a.Offset)
		}
		break
	}

	//out:
	fp += str
	return fp
}

func Rconv(r int) string {
	var str string
	var fp string

	if r < NREG {
		str = fmt.Sprintf("r%d", r)
	} else {

		str = fmt.Sprintf("f%d", r-NREG)
	}
	fp += str
	return fp
}

func DRconv(a int) string {
	var s string
	var fp string

	s = "C_??"
	if a >= C_NONE && a <= C_NCLASS {
		s = cnames9[a]
	}
	fp += s
	return fp
}
