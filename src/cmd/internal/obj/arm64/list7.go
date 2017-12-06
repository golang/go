// cmd/7l/list.c and cmd/7l/sub.c from Vita Nuova.
// https://code.google.com/p/ken-cc/source/browse/
//
// 	Copyright © 1994-1999 Lucent Technologies Inc. All rights reserved.
// 	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
// 	Portions Copyright © 1997-1999 Vita Nuova Limited
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
// 	Portions Copyright © 2004,2006 Bruce Ellis
// 	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
// 	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
// 	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package arm64

import (
	"cmd/internal/obj"
	"fmt"
)

var strcond = [16]string{
	"EQ",
	"NE",
	"HS",
	"LO",
	"MI",
	"PL",
	"VS",
	"VC",
	"HI",
	"LS",
	"GE",
	"LT",
	"GT",
	"LE",
	"AL",
	"NV",
}

func init() {
	obj.RegisterRegister(obj.RBaseARM64, REG_SPECIAL+1024, rconv)
	obj.RegisterOpcode(obj.ABaseARM64, Anames)
	obj.RegisterRegisterList(obj.RegListARM64Lo, obj.RegListARM64Hi, rlconv)
}

func arrange(a int) string {
	switch a {
	case ARNG_8B:
		return "B8"
	case ARNG_16B:
		return "B16"
	case ARNG_4H:
		return "H4"
	case ARNG_8H:
		return "H8"
	case ARNG_2S:
		return "S2"
	case ARNG_4S:
		return "S4"
	case ARNG_1D:
		return "D1"
	case ARNG_2D:
		return "D2"
	case ARNG_B:
		return "B"
	case ARNG_H:
		return "H"
	case ARNG_S:
		return "S"
	case ARNG_D:
		return "D"
	default:
		return ""
	}
}

func rconv(r int) string {
	if r == REGG {
		return "g"
	}
	switch {
	case REG_R0 <= r && r <= REG_R30:
		return fmt.Sprintf("R%d", r-REG_R0)
	case r == REG_R31:
		return "ZR"
	case REG_F0 <= r && r <= REG_F31:
		return fmt.Sprintf("F%d", r-REG_F0)
	case REG_V0 <= r && r <= REG_V31:
		return fmt.Sprintf("V%d", r-REG_V0)
	case COND_EQ <= r && r <= COND_NV:
		return strcond[r-COND_EQ]
	case r == REGSP:
		return "RSP"
	case r == REG_DAIF:
		return "DAIF"
	case r == REG_NZCV:
		return "NZCV"
	case r == REG_FPSR:
		return "FPSR"
	case r == REG_FPCR:
		return "FPCR"
	case r == REG_SPSR_EL1:
		return "SPSR_EL1"
	case r == REG_ELR_EL1:
		return "ELR_EL1"
	case r == REG_SPSR_EL2:
		return "SPSR_EL2"
	case r == REG_ELR_EL2:
		return "ELR_EL2"
	case r == REG_CurrentEL:
		return "CurrentEL"
	case r == REG_SP_EL0:
		return "SP_EL0"
	case r == REG_SPSel:
		return "SPSel"
	case r == REG_DAIFSet:
		return "DAIFSet"
	case r == REG_DAIFClr:
		return "DAIFClr"
	case REG_UXTB <= r && r < REG_UXTH:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.UXTB<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.UXTB", r&31)
		}
	case REG_UXTH <= r && r < REG_UXTW:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.UXTH<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.UXTH", r&31)
		}
	case REG_UXTW <= r && r < REG_UXTX:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.UXTW<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.UXTW", r&31)
		}
	case REG_UXTX <= r && r < REG_SXTB:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.UXTX<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.UXTX", r&31)
		}
	case REG_SXTB <= r && r < REG_SXTH:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.SXTB<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.SXTB", r&31)
		}
	case REG_SXTH <= r && r < REG_SXTW:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.SXTH<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.SXTH", r&31)
		}
	case REG_SXTW <= r && r < REG_SXTX:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.SXTW<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.SXTW", r&31)
		}
	case REG_SXTX <= r && r < REG_SPECIAL:
		if (r>>5)&7 != 0 {
			return fmt.Sprintf("R%d.SXTX<<%d", r&31, (r>>5)&7)
		} else {
			return fmt.Sprintf("R%d.SXTX", r&31)
		}
	case REG_ARNG <= r && r < REG_ELEM:
		return fmt.Sprintf("V%d.%s", r&31, arrange((r>>5)&15))
	case REG_ELEM <= r && r < REG_ELEM_END:
		return fmt.Sprintf("V%d.%s", r&31, arrange((r>>5)&15))
	}
	return fmt.Sprintf("badreg(%d)", r)
}

func DRconv(a int) string {
	if a >= C_NONE && a <= C_NCLASS {
		return cnames7[a]
	}
	return "C_??"
}

func rlconv(list int64) string {
	str := ""

	// ARM64 register list follows ARM64 instruction decode schema
	// | 31 | 30 | ... | 15 - 12 | 11 - 10 | ... |
	// +----+----+-----+---------+---------+-----+
	// |    | Q  | ... | opcode  |   size  | ... |

	firstReg := int(list & 31)
	opcode := (list >> 12) & 15
	var regCnt int
	var t string
	switch opcode {
	case 0x7:
		regCnt = 1
	case 0xa:
		regCnt = 2
	case 0x6:
		regCnt = 3
	case 0x2:
		regCnt = 4
	default:
		regCnt = -1
	}
	// Q:size
	arng := ((list>>30)&1)<<2 | (list>>10)&3
	switch arng {
	case 0:
		t = "B8"
	case 4:
		t = "B16"
	case 1:
		t = "H4"
	case 5:
		t = "H8"
	case 2:
		t = "S2"
	case 6:
		t = "S4"
	case 3:
		t = "D1"
	case 7:
		t = "D2"
	}
	for i := 0; i < regCnt; i++ {
		if str == "" {
			str += "["
		} else {
			str += ","
		}
		str += fmt.Sprintf("V%d.", (firstReg+i)&31)
		str += t
	}
	str += "]"
	return str
}
