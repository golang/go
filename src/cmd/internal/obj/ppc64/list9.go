// cmd/9l/list.c from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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

package ppc64

import (
	"cmd/internal/obj"
	"fmt"
)

func init() {
	obj.RegisterRegister(obj.RBasePPC64, REG_SPR0+1024, rconv)
	// Note, the last entry in Anames is "LASTAOUT", it is not a real opcode.
	obj.RegisterOpcode(obj.ABasePPC64, Anames[:len(Anames)-1])
	obj.RegisterOpcode(AFIRSTGEN, GenAnames)
}

func rconv(r int) string {
	if r == 0 {
		return "NONE"
	}
	if r == REGG {
		// Special case.
		return "g"
	}
	if REG_R0 <= r && r <= REG_R31 {
		return fmt.Sprintf("R%d", r-REG_R0)
	}
	if REG_F0 <= r && r <= REG_F31 {
		return fmt.Sprintf("F%d", r-REG_F0)
	}
	if REG_V0 <= r && r <= REG_V31 {
		return fmt.Sprintf("V%d", r-REG_V0)
	}
	if REG_VS0 <= r && r <= REG_VS63 {
		return fmt.Sprintf("VS%d", r-REG_VS0)
	}
	if REG_CR0 <= r && r <= REG_CR7 {
		return fmt.Sprintf("CR%d", r-REG_CR0)
	}
	if REG_CR0LT <= r && r <= REG_CR7SO {
		bits := [4]string{"LT", "GT", "EQ", "SO"}
		crf := (r - REG_CR0LT) / 4
		return fmt.Sprintf("CR%d%s", crf, bits[r%4])
	}
	if REG_A0 <= r && r <= REG_A7 {
		return fmt.Sprintf("A%d", r-REG_A0)
	}
	if r == REG_CR {
		return "CR"
	}
	if REG_SPR0 <= r && r <= REG_SPR0+1023 {
		switch r {
		case REG_XER:
			return "XER"

		case REG_LR:
			return "LR"

		case REG_CTR:
			return "CTR"
		}

		return fmt.Sprintf("SPR(%d)", r-REG_SPR0)
	}

	if r == REG_FPSCR {
		return "FPSCR"
	}
	if r == REG_MSR {
		return "MSR"
	}

	return fmt.Sprintf("Rgok(%d)", r-obj.RBasePPC64)
}

func DRconv(a int) string {
	s := "C_??"
	if a >= C_NONE && a <= C_NCLASS {
		s = cnames9[a]
	}
	var fp string
	fp += s
	return fp
}

func ConstantToCRbit(c int64) (int16, bool) {
	reg64 := REG_CRBIT0 + c
	success := reg64 >= REG_CR0LT && reg64 <= REG_CR7SO
	return int16(reg64), success
}
