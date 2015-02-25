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

	a = int(p.As)

	str = ""
	if a == obj.ADATA {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v/%d,%v",
			p.Pc, p.Line(), Aconv(a), obj.Dconv(p, Rconv, &p.From), p.From3.Offset, obj.Dconv(p, Rconv, &p.To))
	} else if a == obj.ATEXT || a == obj.AGLOBL {
		if p.From3.Offset != 0 {
			str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%d,%v",
				p.Pc, p.Line(), Aconv(a), obj.Dconv(p, Rconv, &p.From), p.From3.Offset, obj.Dconv(p, Rconv, &p.To))
		} else {
			str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%v",
				p.Pc, p.Line(), Aconv(a), obj.Dconv(p, Rconv, &p.From), obj.Dconv(p, Rconv, &p.To))
		}
	} else {
		if p.Mark&NOSCHED != 0 {
			str += fmt.Sprintf("*")
		}
		if p.Reg == 0 && p.From3.Type == obj.TYPE_NONE {
			str += fmt.Sprintf("%.5d (%v)\t%v\t%v,%v",
				p.Pc, p.Line(), Aconv(a), obj.Dconv(p, Rconv, &p.From), obj.Dconv(p, Rconv, &p.To))
		} else if a != obj.ATEXT && p.From.Type == obj.TYPE_MEM {
			off := ""
			if p.From.Offset != 0 {
				off = fmt.Sprintf("%d", p.From.Offset)
			}
			str += fmt.Sprintf("%.5d (%v)\t%v\t%s(%v+%v),%v",
				p.Pc, p.Line(), Aconv(a), off, Rconv(int(p.From.Reg)), Rconv(int(p.Reg)), obj.Dconv(p, Rconv, &p.To))
		} else if p.To.Type == obj.TYPE_MEM {
			off := ""
			if p.From.Offset != 0 {
				off = fmt.Sprintf("%d", p.From.Offset)
			}
			str += fmt.Sprintf("%.5d (%v)\t%v\t%v,%s(%v+%v)",
				p.Pc, p.Line(), Aconv(a), obj.Dconv(p, Rconv, &p.From), off, Rconv(int(p.To.Reg)), Rconv(int(p.Reg)))
		} else {
			str += fmt.Sprintf("%.5d (%v)\t%v\t%v",
				p.Pc, p.Line(), Aconv(a), obj.Dconv(p, Rconv, &p.From))
			if p.Reg != 0 {
				str += fmt.Sprintf(",%v", Rconv(int(p.Reg)))
			}
			if p.From3.Type != obj.TYPE_NONE {
				str += fmt.Sprintf(",%v", obj.Dconv(p, Rconv, &p.From3))
			}
			str += fmt.Sprintf(",%v", obj.Dconv(p, Rconv, &p.To))
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
	if a >= obj.AXXX && a < ALAST {
		s = Anames[a]
	}
	fp += s
	return fp
}

func Rconv(r int) string {
	var fp string

	if r == 0 {
		fp += "NONE"
		return fp
	}
	if REG_R0 <= r && r <= REG_R31 {
		fp += fmt.Sprintf("R%d", r-REG_R0)
		return fp
	}
	if REG_F0 <= r && r <= REG_F31 {
		fp += fmt.Sprintf("F%d", r-REG_F0)
		return fp
	}
	if REG_C0 <= r && r <= REG_C7 {
		fp += fmt.Sprintf("C%d", r-REG_C0)
		return fp
	}
	if r == REG_CR {
		fp += "CR"
		return fp
	}
	if REG_SPR0 <= r && r <= REG_SPR0+1023 {
		switch r {
		case REG_XER:
			fp += "XER"
			return fp

		case REG_LR:
			fp += "LR"
			return fp

		case REG_CTR:
			fp += "CTR"
			return fp
		}

		fp += fmt.Sprintf("SPR(%d)", r-REG_SPR0)
		return fp
	}

	if REG_DCR0 <= r && r <= REG_DCR0+1023 {
		fp += fmt.Sprintf("DCR(%d)", r-REG_DCR0)
		return fp
	}
	if r == REG_FPSCR {
		fp += "FPSCR"
		return fp
	}
	if r == REG_MSR {
		fp += "MSR"
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
		s = cnames9[a]
	}
	fp += s
	return fp
}
