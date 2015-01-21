// Inferno utils/6c/list.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/list.c
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

package x86

import (
	"cmd/internal/obj"
	"fmt"
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

const (
	STRINGSZ = 1000
)

var bigP *obj.Prog

func Pconv(p *obj.Prog) string {
	var str string
	var fp string

	switch p.As {
	case ADATA:
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v/%d,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), p.From.Scale, Dconv(p, 0, &p.To))

	case ATEXT:
		if p.From.Scale != 0 {
			str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%d,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), p.From.Scale, Dconv(p, fmtLong, &p.To))
			break
		}

		str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), Dconv(p, fmtLong, &p.To))

	default:
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%v", p.Pc, p.Line(), Aconv(int(p.As)), Dconv(p, 0, &p.From), Dconv(p, 0, &p.To))
		break
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

	var i int

	i = int(a.Type)

	if flag&fmtLong != 0 /*untyped*/ {
		if i == D_CONST {
			str = fmt.Sprintf("$%d-%d", a.Offset&0xffffffff, a.Offset>>32)
		} else {

			// ATEXT dst is not constant
			str = fmt.Sprintf("!!%v", Dconv(p, 0, a))
		}

		goto brk
	}

	if i >= D_INDIR {
		if a.Offset != 0 {
			str = fmt.Sprintf("%d(%v)", a.Offset, Rconv(i-D_INDIR))
		} else {

			str = fmt.Sprintf("(%v)", Rconv(i-D_INDIR))
		}
		goto brk
	}

	switch i {
	default:
		if a.Offset != 0 {
			str = fmt.Sprintf("$%d,%v", a.Offset, Rconv(i))
		} else {

			str = fmt.Sprintf("%v", Rconv(i))
		}

	case D_NONE:
		str = ""

	case D_BRANCH:
		if a.Sym != nil {
			str = fmt.Sprintf("%s(SB)", a.Sym.Name)
		} else if p != nil && p.Pcond != nil {
			str = fmt.Sprintf("%d", p.Pcond.Pc)
		} else if a.U.Branch != nil {
			str = fmt.Sprintf("%d", a.U.Branch.Pc)
		} else {

			str = fmt.Sprintf("%d(PC)", a.Offset)
		}

	case D_EXTERN:
		str = fmt.Sprintf("%s+%d(SB)", a.Sym.Name, a.Offset)

	case D_STATIC:
		str = fmt.Sprintf("%s<>+%d(SB)", a.Sym.Name, a.Offset)

	case D_AUTO:
		if a.Sym != nil {
			str = fmt.Sprintf("%s+%d(SP)", a.Sym.Name, a.Offset)
		} else {

			str = fmt.Sprintf("%d(SP)", a.Offset)
		}

	case D_PARAM:
		if a.Sym != nil {
			str = fmt.Sprintf("%s+%d(FP)", a.Sym.Name, a.Offset)
		} else {

			str = fmt.Sprintf("%d(FP)", a.Offset)
		}

	case D_CONST:
		str = fmt.Sprintf("$%d", a.Offset)

	case D_FCONST:
		str = fmt.Sprintf("$(%.17g)", a.U.Dval)

	case D_SCONST:
		str = fmt.Sprintf("$\"%q\"", a.U.Sval)

	case D_ADDR:
		a.Type = int16(a.Index)
		a.Index = D_NONE
		str = fmt.Sprintf("$%v", Dconv(p, 0, a))
		a.Index = uint8(a.Type)
		a.Type = D_ADDR
		goto conv
	}

brk:
	if a.Index != D_NONE {
		s = fmt.Sprintf("(%v*%d)", Rconv(int(a.Index)), int(a.Scale))
		str += s
	}

conv:
	fp += str
	return fp
}

var Register = []string{
	"AL", /* [D_AL] */
	"CL",
	"DL",
	"BL",
	"SPB",
	"BPB",
	"SIB",
	"DIB",
	"R8B",
	"R9B",
	"R10B",
	"R11B",
	"R12B",
	"R13B",
	"R14B",
	"R15B",
	"AX", /* [D_AX] */
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",
	"R8",
	"R9",
	"R10",
	"R11",
	"R12",
	"R13",
	"R14",
	"R15",
	"AH",
	"CH",
	"DH",
	"BH",
	"F0", /* [D_F0] */
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",
	"M0",
	"M1",
	"M2",
	"M3",
	"M4",
	"M5",
	"M6",
	"M7",
	"X0",
	"X1",
	"X2",
	"X3",
	"X4",
	"X5",
	"X6",
	"X7",
	"X8",
	"X9",
	"X10",
	"X11",
	"X12",
	"X13",
	"X14",
	"X15",
	"CS", /* [D_CS] */
	"SS",
	"DS",
	"ES",
	"FS",
	"GS",
	"GDTR", /* [D_GDTR] */
	"IDTR", /* [D_IDTR] */
	"LDTR", /* [D_LDTR] */
	"MSW",  /* [D_MSW] */
	"TASK", /* [D_TASK] */
	"CR0",  /* [D_CR] */
	"CR1",
	"CR2",
	"CR3",
	"CR4",
	"CR5",
	"CR6",
	"CR7",
	"CR8",
	"CR9",
	"CR10",
	"CR11",
	"CR12",
	"CR13",
	"CR14",
	"CR15",
	"DR0", /* [D_DR] */
	"DR1",
	"DR2",
	"DR3",
	"DR4",
	"DR5",
	"DR6",
	"DR7",
	"TR0", /* [D_TR] */
	"TR1",
	"TR2",
	"TR3",
	"TR4",
	"TR5",
	"TR6",
	"TR7",
	"TLS",  /* [D_TLS] */
	"NONE", /* [D_NONE] */
}

func Rconv(r int) string {
	var str string
	var fp string

	if r >= D_AL && r <= D_NONE {
		str = fmt.Sprintf("%s", Register[r-D_AL])
	} else {

		str = fmt.Sprintf("gok(%d)", r)
	}

	fp += str
	return fp
}
