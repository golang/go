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
	obj.RegisterOpSuffix("arm64", obj.CConvARM)
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
	case ARNG_1Q:
		return "Q1"
	default:
		return ""
	}
}

func rconv(r int) string {
	ext := (r >> 5) & 7
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
	case r == REG_DAIFSet:
		return "DAIFSet"
	case r == REG_DAIFClr:
		return "DAIFClr"
	case r == REG_PLDL1KEEP:
		return "PLDL1KEEP"
	case r == REG_PLDL1STRM:
		return "PLDL1STRM"
	case r == REG_PLDL2KEEP:
		return "PLDL2KEEP"
	case r == REG_PLDL2STRM:
		return "PLDL2STRM"
	case r == REG_PLDL3KEEP:
		return "PLDL3KEEP"
	case r == REG_PLDL3STRM:
		return "PLDL3STRM"
	case r == REG_PLIL1KEEP:
		return "PLIL1KEEP"
	case r == REG_PLIL1STRM:
		return "PLIL1STRM"
	case r == REG_PLIL2KEEP:
		return "PLIL2KEEP"
	case r == REG_PLIL2STRM:
		return "PLIL2STRM"
	case r == REG_PLIL3KEEP:
		return "PLIL3KEEP"
	case r == REG_PLIL3STRM:
		return "PLIL3STRM"
	case r == REG_PSTL1KEEP:
		return "PSTL1KEEP"
	case r == REG_PSTL1STRM:
		return "PSTL1STRM"
	case r == REG_PSTL2KEEP:
		return "PSTL2KEEP"
	case r == REG_PSTL2STRM:
		return "PSTL2STRM"
	case r == REG_PSTL3KEEP:
		return "PSTL3KEEP"
	case r == REG_PSTL3STRM:
		return "PSTL3STRM"
	case REG_UXTB <= r && r < REG_UXTH:
		if ext != 0 {
			return fmt.Sprintf("%s.UXTB<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.UXTB", regname(r))
		}
	case REG_UXTH <= r && r < REG_UXTW:
		if ext != 0 {
			return fmt.Sprintf("%s.UXTH<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.UXTH", regname(r))
		}
	case REG_UXTW <= r && r < REG_UXTX:
		if ext != 0 {
			return fmt.Sprintf("%s.UXTW<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.UXTW", regname(r))
		}
	case REG_UXTX <= r && r < REG_SXTB:
		if ext != 0 {
			return fmt.Sprintf("%s.UXTX<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.UXTX", regname(r))
		}
	case REG_SXTB <= r && r < REG_SXTH:
		if ext != 0 {
			return fmt.Sprintf("%s.SXTB<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.SXTB", regname(r))
		}
	case REG_SXTH <= r && r < REG_SXTW:
		if ext != 0 {
			return fmt.Sprintf("%s.SXTH<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.SXTH", regname(r))
		}
	case REG_SXTW <= r && r < REG_SXTX:
		if ext != 0 {
			return fmt.Sprintf("%s.SXTW<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.SXTW", regname(r))
		}
	case REG_SXTX <= r && r < REG_SPECIAL:
		if ext != 0 {
			return fmt.Sprintf("%s.SXTX<<%d", regname(r), ext)
		} else {
			return fmt.Sprintf("%s.SXTX", regname(r))
		}
	// bits 0-4 indicate register, bits 5-7 indicate shift amount, bit 8 equals to 0.
	case REG_LSL <= r && r < (REG_LSL+1<<8):
		return fmt.Sprintf("R%d<<%d", r&31, (r>>5)&7)
	case REG_ARNG <= r && r < REG_ELEM:
		return fmt.Sprintf("V%d.%s", r&31, arrange((r>>5)&15))
	case REG_ELEM <= r && r < REG_ELEM_END:
		return fmt.Sprintf("V%d.%s", r&31, arrange((r>>5)&15))
	}
	// Return system register name.
	name, _, _ := SysRegEnc(int16(r))
	if name != "" {
		return name
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

func regname(r int) string {
	if r&31 == 31 {
		return "ZR"
	}
	return fmt.Sprintf("R%d", r&31)
}
