// Inferno utils/5c/list.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/5c/list.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

func init() {
	obj.RegisterRegister(obj.RBaseARM, MAXREG, rconv)
	obj.RegisterOpcode(obj.ABaseARM, Anames)
	obj.RegisterRegisterList(obj.RegListARMLo, obj.RegListARMHi, rlconv)
	obj.RegisterOpSuffix("arm", obj.CConvARM)
}

func rconv(r int) string {
	if r == 0 {
		return "NONE"
	}
	if r == REGG {
		// Special case.
		return "g"
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

	case REG_MB_SY:
		return "MB_SY"
	case REG_MB_ST:
		return "MB_ST"
	case REG_MB_ISH:
		return "MB_ISH"
	case REG_MB_ISHST:
		return "MB_ISHST"
	case REG_MB_NSH:
		return "MB_NSH"
	case REG_MB_NSHST:
		return "MB_NSHST"
	case REG_MB_OSH:
		return "MB_OSH"
	case REG_MB_OSHST:
		return "MB_OSHST"
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

func rlconv(list int64) string {
	str := ""
	for i := 0; i < 16; i++ {
		if list&(1<<uint(i)) != 0 {
			if str == "" {
				str += "["
			} else {
				str += ","
			}
			// This is ARM-specific; R10 is g.
			if i == REGG-REG_R0 {
				str += "g"
			} else {
				str += fmt.Sprintf("R%d", i)
			}
		}
	}

	str += "]"
	return str
}
