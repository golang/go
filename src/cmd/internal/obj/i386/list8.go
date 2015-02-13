// Inferno utils/8c/list.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/list.c
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

package i386

import (
	"cmd/internal/obj"
	"fmt"
)

const (
	STRINGSZ = 1000
)

var bigP *obj.Prog

func Pconv(p *obj.Prog) string {
	var str string
	var fp string

	switch p.As {
	case obj.ADATA:
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v/%d,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), p.From3.Offset, Dconv(p, 0, &p.To))

	case obj.ATEXT:
		if p.From3.Offset != 0 {
			str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%d,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), p.From3.Offset, Dconv(p, 0, &p.To))
			break
		}

		str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))

	default:
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))

		// TODO(rsc): This special case is for SHRQ $32, AX:DX, which encodes as
		//	SHRQ $32(DX*0), AX
		// Remove.
		if (p.From.Type == obj.TYPE_REG || p.From.Type == obj.TYPE_CONST) && p.From.Index != 0 {
			str += fmt.Sprintf(":%v", Rconv(int(p.From.Index)))
		}
	}

	fp += str
	return fp
}

func Aconv(i int) string {
	var fp string

	fp += Anames[i]
	return fp
}

func Dconv(p *obj.Prog, flag int, a *obj.Addr) string {
	var str string
	var s string
	var fp string

	switch a.Type {
	default:
		str = fmt.Sprintf("type=%d", a.Type)

	case obj.TYPE_NONE:
		str = ""

		// TODO(rsc): This special case is for instructions like
	//	PINSRQ	CX,$1,X6
	// where the $1 is included in the p->to Addr.
	// Move into a new field.
	case obj.TYPE_REG:
		if a.Offset != 0 {
			str = fmt.Sprintf("$%d,%v", a.Offset, Rconv(int(a.Reg)))
			break
		}

		str = fmt.Sprintf("%v", Rconv(int(a.Reg)))

	case obj.TYPE_BRANCH:
		if a.Sym != nil {
			str = fmt.Sprintf("%s(SB)", a.Sym.Name)
		} else if p != nil && p.Pcond != nil {
			str = fmt.Sprintf("%d", p.Pcond.Pc)
		} else if a.U.Branch != nil {
			str = fmt.Sprintf("%d", a.U.Branch.Pc)
		} else {
			str = fmt.Sprintf("%d(PC)", a.Offset)
		}

	case obj.TYPE_MEM:
		switch a.Name {
		default:
			str = fmt.Sprintf("name=%d", a.Name)

		case obj.NAME_NONE:
			if a.Offset != 0 {
				str = fmt.Sprintf("%d(%v)", a.Offset, Rconv(int(a.Reg)))
			} else {
				str = fmt.Sprintf("(%v)", Rconv(int(a.Reg)))
			}

		case obj.NAME_EXTERN:
			str = fmt.Sprintf("%s+%d(SB)", a.Sym.Name, a.Offset)

		case obj.NAME_STATIC:
			str = fmt.Sprintf("%s<>+%d(SB)", a.Sym.Name, a.Offset)

		case obj.NAME_AUTO:
			if a.Sym != nil {
				str = fmt.Sprintf("%s+%d(SP)", a.Sym.Name, a.Offset)
			} else {
				str = fmt.Sprintf("%d(SP)", a.Offset)
			}

		case obj.NAME_PARAM:
			if a.Sym != nil {
				str = fmt.Sprintf("%s+%d(FP)", a.Sym.Name, a.Offset)
			} else {
				str = fmt.Sprintf("%d(FP)", a.Offset)
			}
		}

		if a.Index != REG_NONE {
			s = fmt.Sprintf("(%v*%d)", Rconv(int(a.Index)), int(a.Scale))
			str += s
		}

	case obj.TYPE_CONST:
		str = fmt.Sprintf("$%d", a.Offset)

	case obj.TYPE_TEXTSIZE:
		if a.U.Argsize == obj.ArgsSizeUnknown {
			str = fmt.Sprintf("$%d", a.Offset)
		} else {
			str = fmt.Sprintf("$%d-%d", a.Offset, a.U.Argsize)
		}

	case obj.TYPE_FCONST:
		str = fmt.Sprintf("$(%.17g)", a.U.Dval)

	case obj.TYPE_SCONST:
		str = fmt.Sprintf("$%q", a.U.Sval)

	case obj.TYPE_ADDR:
		a.Type = obj.TYPE_MEM
		str = fmt.Sprintf("$%v", Dconv(p, 0, a))
		a.Type = obj.TYPE_ADDR
	}

	fp += str
	return fp
}

var Register = []string{
	"AL", /* [REG_AL] */
	"CL",
	"DL",
	"BL",
	"AH",
	"CH",
	"DH",
	"BH",
	"AX", /* [REG_AX] */
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",
	"F0", /* [REG_F0] */
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",
	"CS", /* [REG_CS] */
	"SS",
	"DS",
	"ES",
	"FS",
	"GS",
	"GDTR", /* [REG_GDTR] */
	"IDTR", /* [REG_IDTR] */
	"LDTR", /* [REG_LDTR] */
	"MSW",  /* [REG_MSW] */
	"TASK", /* [REG_TASK] */
	"CR0",  /* [REG_CR] */
	"CR1",
	"CR2",
	"CR3",
	"CR4",
	"CR5",
	"CR6",
	"CR7",
	"DR0", /* [REG_DR] */
	"DR1",
	"DR2",
	"DR3",
	"DR4",
	"DR5",
	"DR6",
	"DR7",
	"TR0", /* [REG_TR] */
	"TR1",
	"TR2",
	"TR3",
	"TR4",
	"TR5",
	"TR6",
	"TR7",
	"X0", /* [REG_X0] */
	"X1",
	"X2",
	"X3",
	"X4",
	"X5",
	"X6",
	"X7",
	"TLS",    /* [REG_TLS] */
	"MAXREG", /* [MAXREG] */
}

func Rconv(r int) string {
	var str string
	var fp string

	if r == REG_NONE {
		fp += "NONE"
		return fp
	}
	if r >= REG_AL && r-REG_AL < len(Register) {
		str = fmt.Sprintf("%s", Register[r-REG_AL])
	} else {
		str = fmt.Sprintf("gok(%d)", r)
	}

	fp += str
	return fp
}
