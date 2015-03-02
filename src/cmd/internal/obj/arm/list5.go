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
	a := int(p.As)
	s := int(p.Scond)
	sc := extra[(s&C_SCOND)^C_SCOND_XOR]
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
	var str string
	if a == obj.ADATA {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v/%d,%v",
			p.Pc, p.Line(), Aconv(a), obj.Dconv(p, &p.From), p.From3.Offset, obj.Dconv(p, &p.To))
	} else if p.As == obj.ATEXT {
		str = fmt.Sprintf("%.5d (%v)\t%v\t%v,%d,%v",
			p.Pc, p.Line(), Aconv(a), obj.Dconv(p, &p.From), p.From3.Offset, obj.Dconv(p, &p.To))
	} else if p.Reg == 0 {
		str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v",
			p.Pc, p.Line(), Aconv(a), sc, obj.Dconv(p, &p.From), obj.Dconv(p, &p.To))
	} else {
		str = fmt.Sprintf("%.5d (%v)\t%v%s\t%v,%v,%v",
			p.Pc, p.Line(), Aconv(a), sc, obj.Dconv(p, &p.From), Rconv(int(p.Reg)), obj.Dconv(p, &p.To))
	}

	var fp string
	fp += str
	return fp
}

func Aconv(a int) string {
	s := "???"
	if a >= obj.AXXX && a < ALAST {
		s = Anames[a]
	}
	var fp string
	fp += s
	return fp
}

func RAconv(a *obj.Addr) string {
	str := fmt.Sprintf("GOK-reglist")
	switch a.Type {
	case obj.TYPE_CONST:
		if a.Reg != 0 {
			break
		}
		if a.Sym != nil {
			break
		}
		v := int(a.Offset)
		str = ""
		for i := 0; i < NREG; i++ {
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

	var fp string
	fp += str
	return fp
}

func init() {
	obj.RegisterRegister(obj.RBaseARM, MAXREG, Rconv)
}

func Rconv(r int) string {
	if r == 0 {
		return "NONE"
	}
	if REG_R0 <= r && r <= REG_R15 {
		return fmt.Sprintf("R%d", r-REG_R0)
	}
	if REG_F0 <= r && r <= REG_F15 {
		return fmt.Sprintf("F%d", r-REG_F0)
	}

	switch r {
	case REG_FPSR:
		return "FPSR"

	case REG_FPCR:
		return "FPCR"

	case REG_CPSR:
		return "CPSR"

	case REG_SPSR:
		return "SPSR"
	}

	return fmt.Sprintf("Rgok(%d)", r-obj.RBaseARM)
}

func DRconv(a int) string {
	s := "C_??"
	if a >= C_NONE && a <= C_NCLASS {
		s = cnames5[a]
	}
	var fp string
	fp += s
	return fp
}
