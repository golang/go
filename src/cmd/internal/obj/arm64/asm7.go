// cmd/7l/asm.c, cmd/7l/asmout.c, cmd/7l/optab.c, cmd/7l/span.c, cmd/ld/sub.c, cmd/ld/mod.c, from Vita Nuova.
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
	"log"
	"math"
	"sort"
)

const (
	FuncAlign = 16
)

const (
	REGFROM = 1
)

type Optab struct {
	as    obj.As
	a1    uint8
	a2    uint8
	a3    uint8
	type_ int8
	size  int8
	param int16
	flag  int8
	scond uint16
}

var oprange [ALAST & obj.AMask][]Optab

var xcmp [C_NCLASS][C_NCLASS]bool

const (
	S32     = 0 << 31
	S64     = 1 << 31
	Sbit    = 1 << 29
	LSL0_32 = 2 << 13
	LSL0_64 = 3 << 13
)

func OPDP2(x uint32) uint32 {
	return 0<<30 | 0<<29 | 0xd6<<21 | x<<10
}

func OPDP3(sf uint32, op54 uint32, op31 uint32, o0 uint32) uint32 {
	return sf<<31 | op54<<29 | 0x1B<<24 | op31<<21 | o0<<15
}

func OPBcc(x uint32) uint32 {
	return 0x2A<<25 | 0<<24 | 0<<4 | x&15
}

func OPBLR(x uint32) uint32 {
	/* x=0, JMP; 1, CALL; 2, RET */
	return 0x6B<<25 | 0<<23 | x<<21 | 0x1F<<16 | 0<<10
}

func SYSOP(l uint32, op0 uint32, op1 uint32, crn uint32, crm uint32, op2 uint32, rt uint32) uint32 {
	return 0x354<<22 | l<<21 | op0<<19 | op1<<16 | crn&15<<12 | crm&15<<8 | op2<<5 | rt
}

func SYSHINT(x uint32) uint32 {
	return SYSOP(0, 0, 3, 2, 0, x, 0x1F)
}

func LDSTR12U(sz uint32, v uint32, opc uint32) uint32 {
	return sz<<30 | 7<<27 | v<<26 | 1<<24 | opc<<22
}

func LDSTR9S(sz uint32, v uint32, opc uint32) uint32 {
	return sz<<30 | 7<<27 | v<<26 | 0<<24 | opc<<22
}

func LD2STR(o uint32) uint32 {
	return o &^ (3 << 22)
}

func LDSTX(sz uint32, o2 uint32, l uint32, o1 uint32, o0 uint32) uint32 {
	return sz<<30 | 0x8<<24 | o2<<23 | l<<22 | o1<<21 | o0<<15
}

func FPCMP(m uint32, s uint32, type_ uint32, op uint32, op2 uint32) uint32 {
	return m<<31 | s<<29 | 0x1E<<24 | type_<<22 | 1<<21 | op<<14 | 8<<10 | op2
}

func FPCCMP(m uint32, s uint32, type_ uint32, op uint32) uint32 {
	return m<<31 | s<<29 | 0x1E<<24 | type_<<22 | 1<<21 | 1<<10 | op<<4
}

func FPOP1S(m uint32, s uint32, type_ uint32, op uint32) uint32 {
	return m<<31 | s<<29 | 0x1E<<24 | type_<<22 | 1<<21 | op<<15 | 0x10<<10
}

func FPOP2S(m uint32, s uint32, type_ uint32, op uint32) uint32 {
	return m<<31 | s<<29 | 0x1E<<24 | type_<<22 | 1<<21 | op<<12 | 2<<10
}

func FPCVTI(sf uint32, s uint32, type_ uint32, rmode uint32, op uint32) uint32 {
	return sf<<31 | s<<29 | 0x1E<<24 | type_<<22 | 1<<21 | rmode<<19 | op<<16 | 0<<10
}

func ADR(p uint32, o uint32, rt uint32) uint32 {
	return p<<31 | (o&3)<<29 | 0x10<<24 | ((o>>2)&0x7FFFF)<<5 | rt&31
}

func OPBIT(x uint32) uint32 {
	return 1<<30 | 0<<29 | 0xD6<<21 | 0<<16 | x<<10
}

const (
	LFROM = 1 << 0
	LTO   = 1 << 1
)

var optab = []Optab{
	/* struct Optab:
	OPCODE, from, prog->reg, to, type,size,param,flag,scond */
	{obj.ATEXT, C_ADDR, C_NONE, C_TEXTSIZE, 0, 0, 0, 0, 0},

	/* arithmetic operations */
	{AADD, C_REG, C_REG, C_REG, 1, 4, 0, 0, 0},
	{AADD, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{AADC, C_REG, C_REG, C_REG, 1, 4, 0, 0, 0},
	{AADC, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{ANEG, C_REG, C_NONE, C_REG, 25, 4, 0, 0, 0},
	{ANEG, C_NONE, C_NONE, C_REG, 25, 4, 0, 0, 0},
	{ANGC, C_REG, C_NONE, C_REG, 17, 4, 0, 0, 0},
	{ACMP, C_REG, C_REG, C_NONE, 1, 4, 0, 0, 0},
	{AADD, C_ADDCON, C_RSP, C_RSP, 2, 4, 0, 0, 0},
	{AADD, C_ADDCON, C_NONE, C_RSP, 2, 4, 0, 0, 0},
	{ACMP, C_ADDCON, C_RSP, C_NONE, 2, 4, 0, 0, 0},
	// TODO: these don't work properly.
	// {AADD, C_MBCON, C_RSP, C_RSP, 2, 4, 0, 0, 0},
	// {AADD, C_MBCON, C_NONE, C_RSP, 2, 4, 0, 0, 0},
	// {ACMP, C_MBCON, C_RSP, C_NONE, 2, 4, 0, 0, 0},
	{AADD, C_VCON, C_RSP, C_RSP, 13, 8, 0, LFROM, 0},
	{AADD, C_VCON, C_NONE, C_RSP, 13, 8, 0, LFROM, 0},
	{ACMP, C_VCON, C_REG, C_NONE, 13, 8, 0, LFROM, 0},
	{AADD, C_SHIFT, C_REG, C_REG, 3, 4, 0, 0, 0},
	{AADD, C_SHIFT, C_NONE, C_REG, 3, 4, 0, 0, 0},
	{AMVN, C_SHIFT, C_NONE, C_REG, 3, 4, 0, 0, 0},
	{ACMP, C_SHIFT, C_REG, C_NONE, 3, 4, 0, 0, 0},
	{ANEG, C_SHIFT, C_NONE, C_REG, 26, 4, 0, 0, 0},
	{AADD, C_REG, C_RSP, C_RSP, 27, 4, 0, 0, 0},
	{AADD, C_REG, C_NONE, C_RSP, 27, 4, 0, 0, 0},
	{ACMP, C_REG, C_RSP, C_NONE, 27, 4, 0, 0, 0},
	{AADD, C_EXTREG, C_RSP, C_RSP, 27, 4, 0, 0, 0},
	{AADD, C_EXTREG, C_NONE, C_RSP, 27, 4, 0, 0, 0},
	{AMVN, C_EXTREG, C_NONE, C_RSP, 27, 4, 0, 0, 0},
	{ACMP, C_EXTREG, C_RSP, C_NONE, 27, 4, 0, 0, 0},
	{AADD, C_REG, C_REG, C_REG, 1, 4, 0, 0, 0},
	{AADD, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},

	/* logical operations */
	{AAND, C_REG, C_REG, C_REG, 1, 4, 0, 0, 0},
	{AAND, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{ABIC, C_REG, C_REG, C_REG, 1, 4, 0, 0, 0},
	{ABIC, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	// TODO: these don't work properly.
	// {AAND, C_BITCON, C_REG, C_REG, 53, 4, 0, 0, 0},
	// {AAND, C_BITCON, C_NONE, C_REG, 53, 4, 0, 0, 0},
	// {ABIC, C_BITCON, C_REG, C_REG, 53, 4, 0, 0, 0},
	// {ABIC, C_BITCON, C_NONE, C_REG, 53, 4, 0, 0, 0},
	{AAND, C_VCON, C_REG, C_REG, 28, 8, 0, LFROM, 0},
	{AAND, C_VCON, C_NONE, C_REG, 28, 8, 0, LFROM, 0},
	{ABIC, C_VCON, C_REG, C_REG, 28, 8, 0, LFROM, 0},
	{ABIC, C_VCON, C_NONE, C_REG, 28, 8, 0, LFROM, 0},
	{AAND, C_SHIFT, C_REG, C_REG, 3, 4, 0, 0, 0},
	{AAND, C_SHIFT, C_NONE, C_REG, 3, 4, 0, 0, 0},
	{ABIC, C_SHIFT, C_REG, C_REG, 3, 4, 0, 0, 0},
	{ABIC, C_SHIFT, C_NONE, C_REG, 3, 4, 0, 0, 0},
	{AMOVD, C_RSP, C_NONE, C_RSP, 24, 4, 0, 0, 0},
	{AMVN, C_REG, C_NONE, C_REG, 24, 4, 0, 0, 0},
	{AMOVB, C_REG, C_NONE, C_REG, 45, 4, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_REG, 45, 4, 0, 0, 0},
	{AMOVH, C_REG, C_NONE, C_REG, 45, 4, 0, 0, 0}, /* also MOVHU */
	{AMOVW, C_REG, C_NONE, C_REG, 45, 4, 0, 0, 0}, /* also MOVWU */
	/* TODO: MVN C_SHIFT */

	/* MOVs that become MOVK/MOVN/MOVZ/ADD/SUB/OR */
	{AMOVW, C_MOVCON, C_NONE, C_REG, 32, 4, 0, 0, 0},
	{AMOVD, C_MOVCON, C_NONE, C_REG, 32, 4, 0, 0, 0},

	// TODO: these don't work properly.
	// { AMOVW,		C_ADDCON,	C_NONE,	C_REG,		2, 4, 0 , 0},
	// { AMOVD,		C_ADDCON,	C_NONE,	C_REG,		2, 4, 0 , 0},
	// { AMOVW,		C_BITCON,	C_NONE,	C_REG,		53, 4, 0 , 0},
	// { AMOVD,		C_BITCON,	C_NONE,	C_REG,		53, 4, 0 , 0},

	{AMOVK, C_VCON, C_NONE, C_REG, 33, 4, 0, 0, 0},
	{AMOVD, C_AACON, C_NONE, C_REG, 4, 4, REGFROM, 0, 0},
	{ASDIV, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{ASDIV, C_REG, C_REG, C_REG, 1, 4, 0, 0, 0},
	{AB, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0},
	{ABL, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0},
	{AB, C_NONE, C_NONE, C_ZOREG, 6, 4, 0, 0, 0},
	{ABL, C_NONE, C_NONE, C_REG, 6, 4, 0, 0, 0},
	{ABL, C_REG, C_NONE, C_REG, 6, 4, 0, 0, 0},
	{ABL, C_NONE, C_NONE, C_ZOREG, 6, 4, 0, 0, 0},
	{obj.ARET, C_NONE, C_NONE, C_REG, 6, 4, 0, 0, 0},
	{obj.ARET, C_NONE, C_NONE, C_ZOREG, 6, 4, 0, 0, 0},
	{AADRP, C_SBRA, C_NONE, C_REG, 60, 4, 0, 0, 0},
	{AADR, C_SBRA, C_NONE, C_REG, 61, 4, 0, 0, 0},
	{ABFM, C_VCON, C_REG, C_REG, 42, 4, 0, 0, 0},
	{ABFI, C_VCON, C_REG, C_REG, 43, 4, 0, 0, 0},
	{AEXTR, C_VCON, C_REG, C_REG, 44, 4, 0, 0, 0},
	{ASXTB, C_REG, C_NONE, C_REG, 45, 4, 0, 0, 0},
	{ACLS, C_REG, C_NONE, C_REG, 46, 4, 0, 0, 0},
	{ABEQ, C_NONE, C_NONE, C_SBRA, 7, 4, 0, 0, 0},
	{ALSL, C_VCON, C_REG, C_REG, 8, 4, 0, 0, 0},
	{ALSL, C_VCON, C_NONE, C_REG, 8, 4, 0, 0, 0},
	{ALSL, C_REG, C_NONE, C_REG, 9, 4, 0, 0, 0},
	{ALSL, C_REG, C_REG, C_REG, 9, 4, 0, 0, 0},
	{ASVC, C_NONE, C_NONE, C_VCON, 10, 4, 0, 0, 0},
	{ASVC, C_NONE, C_NONE, C_NONE, 10, 4, 0, 0, 0},
	{ADWORD, C_NONE, C_NONE, C_VCON, 11, 8, 0, 0, 0},
	{ADWORD, C_NONE, C_NONE, C_LEXT, 11, 8, 0, 0, 0},
	{ADWORD, C_NONE, C_NONE, C_ADDR, 11, 8, 0, 0, 0},
	{ADWORD, C_NONE, C_NONE, C_LACON, 11, 8, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_LCON, 14, 4, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_LEXT, 14, 4, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_ADDR, 14, 4, 0, 0, 0},
	{AMOVW, C_VCON, C_NONE, C_REG, 12, 4, 0, LFROM, 0},
	{AMOVW, C_VCONADDR, C_NONE, C_REG, 68, 8, 0, 0, 0},
	{AMOVD, C_VCON, C_NONE, C_REG, 12, 4, 0, LFROM, 0},
	{AMOVD, C_VCONADDR, C_NONE, C_REG, 68, 8, 0, 0, 0},
	{AMOVB, C_REG, C_NONE, C_ADDR, 64, 12, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_ADDR, 64, 12, 0, 0, 0},
	{AMOVH, C_REG, C_NONE, C_ADDR, 64, 12, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_ADDR, 64, 12, 0, 0, 0},
	{AMOVD, C_REG, C_NONE, C_ADDR, 64, 12, 0, 0, 0},
	{AMOVB, C_ADDR, C_NONE, C_REG, 65, 12, 0, 0, 0},
	{AMOVBU, C_ADDR, C_NONE, C_REG, 65, 12, 0, 0, 0},
	{AMOVH, C_ADDR, C_NONE, C_REG, 65, 12, 0, 0, 0},
	{AMOVW, C_ADDR, C_NONE, C_REG, 65, 12, 0, 0, 0},
	{AMOVD, C_ADDR, C_NONE, C_REG, 65, 12, 0, 0, 0},
	{AMOVD, C_GOTADDR, C_NONE, C_REG, 71, 8, 0, 0, 0},
	{AMOVD, C_TLS_LE, C_NONE, C_REG, 69, 4, 0, 0, 0},
	{AMOVD, C_TLS_IE, C_NONE, C_REG, 70, 8, 0, 0, 0},
	{AMUL, C_REG, C_REG, C_REG, 15, 4, 0, 0, 0},
	{AMUL, C_REG, C_NONE, C_REG, 15, 4, 0, 0, 0},
	{AMADD, C_REG, C_REG, C_REG, 15, 4, 0, 0, 0},
	{AREM, C_REG, C_REG, C_REG, 16, 8, 0, 0, 0},
	{AREM, C_REG, C_NONE, C_REG, 16, 8, 0, 0, 0},
	{ACSEL, C_COND, C_REG, C_REG, 18, 4, 0, 0, 0}, /* from3 optional */
	{ACSET, C_COND, C_NONE, C_REG, 18, 4, 0, 0, 0},
	{ACCMN, C_COND, C_REG, C_VCON, 19, 4, 0, 0, 0}, /* from3 either C_REG or C_VCON */

	/* scaled 12-bit unsigned displacement store */
	{AMOVB, C_REG, C_NONE, C_UAUTO4K, 20, 4, REGSP, 0, 0},
	{AMOVB, C_REG, C_NONE, C_UOREG4K, 20, 4, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_UAUTO4K, 20, 4, REGSP, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_UOREG4K, 20, 4, 0, 0, 0},

	{AMOVH, C_REG, C_NONE, C_UAUTO8K, 20, 4, REGSP, 0, 0},
	{AMOVH, C_REG, C_NONE, C_ZOREG, 20, 4, 0, 0, 0},
	{AMOVH, C_REG, C_NONE, C_UOREG8K, 20, 4, 0, 0, 0},

	{AMOVW, C_REG, C_NONE, C_UAUTO16K, 20, 4, REGSP, 0, 0},
	{AMOVW, C_REG, C_NONE, C_ZOREG, 20, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_UOREG16K, 20, 4, 0, 0, 0},

	/* unscaled 9-bit signed displacement store */
	{AMOVB, C_REG, C_NONE, C_NSAUTO, 20, 4, REGSP, 0, 0},
	{AMOVB, C_REG, C_NONE, C_NSOREG, 20, 4, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_NSAUTO, 20, 4, REGSP, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_NSOREG, 20, 4, 0, 0, 0},

	{AMOVH, C_REG, C_NONE, C_NSAUTO, 20, 4, REGSP, 0, 0},
	{AMOVH, C_REG, C_NONE, C_NSOREG, 20, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_NSAUTO, 20, 4, REGSP, 0, 0},
	{AMOVW, C_REG, C_NONE, C_NSOREG, 20, 4, 0, 0, 0},

	{AMOVD, C_REG, C_NONE, C_UAUTO32K, 20, 4, REGSP, 0, 0},
	{AMOVD, C_REG, C_NONE, C_ZOREG, 20, 4, 0, 0, 0},
	{AMOVD, C_REG, C_NONE, C_UOREG32K, 20, 4, 0, 0, 0},
	{AMOVD, C_REG, C_NONE, C_NSOREG, 20, 4, 0, 0, 0},
	{AMOVD, C_REG, C_NONE, C_NSAUTO, 20, 4, REGSP, 0, 0},

	/* short displacement load */
	{AMOVB, C_UAUTO4K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVB, C_NSAUTO, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVB, C_ZOREG, C_NONE, C_REG, 21, 4, 0, 0, 0},
	{AMOVB, C_UOREG4K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVB, C_NSOREG, C_NONE, C_REG, 21, 4, REGSP, 0, 0},

	{AMOVBU, C_UAUTO4K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVBU, C_NSAUTO, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVBU, C_ZOREG, C_NONE, C_REG, 21, 4, 0, 0, 0},
	{AMOVBU, C_UOREG4K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVBU, C_NSOREG, C_NONE, C_REG, 21, 4, REGSP, 0, 0},

	{AMOVH, C_UAUTO8K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVH, C_NSAUTO, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVH, C_ZOREG, C_NONE, C_REG, 21, 4, 0, 0, 0},
	{AMOVH, C_UOREG8K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVH, C_NSOREG, C_NONE, C_REG, 21, 4, REGSP, 0, 0},

	{AMOVW, C_UAUTO16K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVW, C_NSAUTO, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVW, C_ZOREG, C_NONE, C_REG, 21, 4, 0, 0, 0},
	{AMOVW, C_UOREG16K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVW, C_NSOREG, C_NONE, C_REG, 21, 4, REGSP, 0, 0},

	{AMOVD, C_UAUTO32K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVD, C_NSAUTO, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVD, C_ZOREG, C_NONE, C_REG, 21, 4, 0, 0, 0},
	{AMOVD, C_UOREG32K, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVD, C_NSOREG, C_NONE, C_REG, 21, 4, REGSP, 0, 0},

	/* long displacement store */
	{AMOVB, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, 0, 0},
	{AMOVB, C_REG, C_NONE, C_LOREG, 30, 8, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_LOREG, 30, 8, 0, 0, 0},
	{AMOVH, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, 0, 0},
	{AMOVH, C_REG, C_NONE, C_LOREG, 30, 8, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, 0, 0},
	{AMOVW, C_REG, C_NONE, C_LOREG, 30, 8, 0, 0, 0},
	{AMOVD, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, 0, 0},
	{AMOVD, C_REG, C_NONE, C_LOREG, 30, 8, 0, 0, 0},

	/* long displacement load */
	{AMOVB, C_LAUTO, C_NONE, C_REG, 31, 8, REGSP, 0, 0},
	{AMOVB, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVB, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVBU, C_LAUTO, C_NONE, C_REG, 31, 8, REGSP, 0, 0},
	{AMOVBU, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVBU, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVH, C_LAUTO, C_NONE, C_REG, 31, 8, REGSP, 0, 0},
	{AMOVH, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVH, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVW, C_LAUTO, C_NONE, C_REG, 31, 8, REGSP, 0, 0},
	{AMOVW, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVW, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVD, C_LAUTO, C_NONE, C_REG, 31, 8, REGSP, 0, 0},
	{AMOVD, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},
	{AMOVD, C_LOREG, C_NONE, C_REG, 31, 8, 0, 0, 0},

	/* load long effective stack address (load int32 offset and add) */
	{AMOVD, C_LACON, C_NONE, C_REG, 34, 8, REGSP, LFROM, 0},

	/* pre/post-indexed load (unscaled, signed 9-bit offset) */
	{AMOVD, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPOST},
	{AMOVW, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPOST},
	{AMOVH, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPOST},
	{AMOVB, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPOST},
	{AMOVBU, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPOST},
	{AFMOVS, C_LOREG, C_NONE, C_FREG, 22, 4, 0, 0, C_XPOST},
	{AFMOVD, C_LOREG, C_NONE, C_FREG, 22, 4, 0, 0, C_XPOST},
	{AMOVD, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPRE},
	{AMOVW, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPRE},
	{AMOVH, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPRE},
	{AMOVB, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPRE},
	{AMOVBU, C_LOREG, C_NONE, C_REG, 22, 4, 0, 0, C_XPRE},
	{AFMOVS, C_LOREG, C_NONE, C_FREG, 22, 4, 0, 0, C_XPRE},
	{AFMOVD, C_LOREG, C_NONE, C_FREG, 22, 4, 0, 0, C_XPRE},

	/* pre/post-indexed store (unscaled, signed 9-bit offset) */
	{AMOVD, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPOST},
	{AMOVW, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPOST},
	{AMOVH, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPOST},
	{AMOVB, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPOST},
	{AMOVBU, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPOST},
	{AFMOVS, C_FREG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPOST},
	{AFMOVD, C_FREG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPOST},
	{AMOVD, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPRE},
	{AMOVW, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPRE},
	{AMOVH, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPRE},
	{AMOVB, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPRE},
	{AMOVBU, C_REG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPRE},
	{AFMOVS, C_FREG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPRE},
	{AFMOVD, C_FREG, C_NONE, C_LOREG, 23, 4, 0, 0, C_XPRE},

	/* pre/post-indexed load/store register pair
	   (unscaled, signed 10-bit quad-aligned offset) */
	{ALDP, C_LOREG, C_NONE, C_PAIR, 66, 4, 0, 0, C_XPRE},
	{ALDP, C_LOREG, C_NONE, C_PAIR, 66, 4, 0, 0, C_XPOST},
	{ASTP, C_PAIR, C_NONE, C_LOREG, 67, 4, 0, 0, C_XPRE},
	{ASTP, C_PAIR, C_NONE, C_LOREG, 67, 4, 0, 0, C_XPOST},

	/* special */
	{AMOVD, C_SPR, C_NONE, C_REG, 35, 4, 0, 0, 0},
	{AMRS, C_SPR, C_NONE, C_REG, 35, 4, 0, 0, 0},
	{AMOVD, C_REG, C_NONE, C_SPR, 36, 4, 0, 0, 0},
	{AMSR, C_REG, C_NONE, C_SPR, 36, 4, 0, 0, 0},
	{AMOVD, C_VCON, C_NONE, C_SPR, 37, 4, 0, 0, 0},
	{AMSR, C_VCON, C_NONE, C_SPR, 37, 4, 0, 0, 0},
	{AERET, C_NONE, C_NONE, C_NONE, 41, 4, 0, 0, 0},
	{AFMOVS, C_FREG, C_NONE, C_UAUTO16K, 20, 4, REGSP, 0, 0},
	{AFMOVS, C_FREG, C_NONE, C_NSAUTO, 20, 4, REGSP, 0, 0},
	{AFMOVS, C_FREG, C_NONE, C_ZOREG, 20, 4, 0, 0, 0},
	{AFMOVS, C_FREG, C_NONE, C_UOREG16K, 20, 4, 0, 0, 0},
	{AFMOVS, C_FREG, C_NONE, C_NSOREG, 20, 4, 0, 0, 0},
	{AFMOVD, C_FREG, C_NONE, C_UAUTO32K, 20, 4, REGSP, 0, 0},
	{AFMOVD, C_FREG, C_NONE, C_NSAUTO, 20, 4, REGSP, 0, 0},
	{AFMOVD, C_FREG, C_NONE, C_ZOREG, 20, 4, 0, 0, 0},
	{AFMOVD, C_FREG, C_NONE, C_UOREG32K, 20, 4, 0, 0, 0},
	{AFMOVD, C_FREG, C_NONE, C_NSOREG, 20, 4, 0, 0, 0},
	{AFMOVS, C_UAUTO16K, C_NONE, C_FREG, 21, 4, REGSP, 0, 0},
	{AFMOVS, C_NSAUTO, C_NONE, C_FREG, 21, 4, REGSP, 0, 0},
	{AFMOVS, C_ZOREG, C_NONE, C_FREG, 21, 4, 0, 0, 0},
	{AFMOVS, C_UOREG16K, C_NONE, C_FREG, 21, 4, 0, 0, 0},
	{AFMOVS, C_NSOREG, C_NONE, C_FREG, 21, 4, 0, 0, 0},
	{AFMOVD, C_UAUTO32K, C_NONE, C_FREG, 21, 4, REGSP, 0, 0},
	{AFMOVD, C_NSAUTO, C_NONE, C_FREG, 21, 4, REGSP, 0, 0},
	{AFMOVD, C_ZOREG, C_NONE, C_FREG, 21, 4, 0, 0, 0},
	{AFMOVD, C_UOREG32K, C_NONE, C_FREG, 21, 4, 0, 0, 0},
	{AFMOVD, C_NSOREG, C_NONE, C_FREG, 21, 4, 0, 0, 0},
	{AFMOVS, C_FREG, C_NONE, C_LAUTO, 30, 8, REGSP, LTO, 0},
	{AFMOVS, C_FREG, C_NONE, C_LOREG, 30, 8, 0, LTO, 0},
	{AFMOVD, C_FREG, C_NONE, C_LAUTO, 30, 8, REGSP, LTO, 0},
	{AFMOVD, C_FREG, C_NONE, C_LOREG, 30, 8, 0, LTO, 0},
	{AFMOVS, C_LAUTO, C_NONE, C_FREG, 31, 8, REGSP, LFROM, 0},
	{AFMOVS, C_LOREG, C_NONE, C_FREG, 31, 8, 0, LFROM, 0},
	{AFMOVD, C_LAUTO, C_NONE, C_FREG, 31, 8, REGSP, LFROM, 0},
	{AFMOVD, C_LOREG, C_NONE, C_FREG, 31, 8, 0, LFROM, 0},
	{AFMOVS, C_FREG, C_NONE, C_ADDR, 64, 12, 0, 0, 0},
	{AFMOVS, C_ADDR, C_NONE, C_FREG, 65, 12, 0, 0, 0},
	{AFMOVD, C_FREG, C_NONE, C_ADDR, 64, 12, 0, 0, 0},
	{AFMOVD, C_ADDR, C_NONE, C_FREG, 65, 12, 0, 0, 0},
	{AFADDS, C_FREG, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AFADDS, C_FREG, C_FREG, C_FREG, 54, 4, 0, 0, 0},
	{AFADDS, C_FCON, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AFADDS, C_FCON, C_FREG, C_FREG, 54, 4, 0, 0, 0},
	{AFMOVS, C_FCON, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AFMOVS, C_FREG, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AFMOVD, C_FCON, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AFMOVD, C_FREG, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AFCVTZSD, C_FREG, C_NONE, C_REG, 29, 4, 0, 0, 0},
	{ASCVTFD, C_REG, C_NONE, C_FREG, 29, 4, 0, 0, 0},
	{AFCMPS, C_FREG, C_FREG, C_NONE, 56, 4, 0, 0, 0},
	{AFCMPS, C_FCON, C_FREG, C_NONE, 56, 4, 0, 0, 0},
	{AFCCMPS, C_COND, C_REG, C_VCON, 57, 4, 0, 0, 0},
	{AFCSELD, C_COND, C_REG, C_FREG, 18, 4, 0, 0, 0},
	{AFCVTSD, C_FREG, C_NONE, C_FREG, 29, 4, 0, 0, 0},
	{ACLREX, C_NONE, C_NONE, C_VCON, 38, 4, 0, 0, 0},
	{ACLREX, C_NONE, C_NONE, C_NONE, 38, 4, 0, 0, 0},
	{ACBZ, C_REG, C_NONE, C_SBRA, 39, 4, 0, 0, 0},
	{ATBZ, C_VCON, C_REG, C_SBRA, 40, 4, 0, 0, 0},
	{ASYS, C_VCON, C_NONE, C_NONE, 50, 4, 0, 0, 0},
	{ASYS, C_VCON, C_REG, C_NONE, 50, 4, 0, 0, 0},
	{ASYSL, C_VCON, C_NONE, C_REG, 50, 4, 0, 0, 0},
	{ADMB, C_VCON, C_NONE, C_NONE, 51, 4, 0, 0, 0},
	{AHINT, C_VCON, C_NONE, C_NONE, 52, 4, 0, 0, 0},
	{ALDAR, C_ZOREG, C_NONE, C_REG, 58, 4, 0, 0, 0},
	{ALDXR, C_ZOREG, C_NONE, C_REG, 58, 4, 0, 0, 0},
	{ALDAXR, C_ZOREG, C_NONE, C_REG, 58, 4, 0, 0, 0},
	{ALDXP, C_ZOREG, C_REG, C_REG, 58, 4, 0, 0, 0},
	{ASTLR, C_REG, C_NONE, C_ZOREG, 59, 4, 0, 0, 0},  // to3=C_NONE
	{ASTXR, C_REG, C_NONE, C_ZOREG, 59, 4, 0, 0, 0},  // to3=C_REG
	{ASTLXR, C_REG, C_NONE, C_ZOREG, 59, 4, 0, 0, 0}, // to3=C_REG

	//	{ ASTXP,		C_REG, C_NONE,	C_ZOREG,		59, 4, 0 , 0}, // TODO(aram):

	{AAESD, C_VREG, C_NONE, C_VREG, 29, 4, 0, 0, 0},
	{ASHA1C, C_VREG, C_REG, C_VREG, 1, 4, 0, 0, 0},

	{obj.AUNDEF, C_NONE, C_NONE, C_NONE, 90, 4, 0, 0, 0},
	{obj.AUSEFIELD, C_ADDR, C_NONE, C_NONE, 0, 0, 0, 0, 0},
	{obj.APCDATA, C_VCON, C_NONE, C_VCON, 0, 0, 0, 0, 0},
	{obj.AFUNCDATA, C_VCON, C_NONE, C_ADDR, 0, 0, 0, 0, 0},
	{obj.ANOP, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0, 0},
	{obj.ADUFFZERO, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0}, // same as AB/ABL
	{obj.ADUFFCOPY, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0}, // same as AB/ABL

	{obj.AXXX, C_NONE, C_NONE, C_NONE, 0, 4, 0, 0, 0},
}

/*
 * valid pstate field values, and value to use in instruction
 */
var pstatefield = []struct {
	a uint32
	b uint32
}{
	{REG_SPSel, 0<<16 | 4<<12 | 5<<5},
	{REG_DAIFSet, 3<<16 | 4<<12 | 6<<5},
	{REG_DAIFClr, 3<<16 | 4<<12 | 7<<5},
}

var pool struct {
	start uint32
	size  uint32
}

func prasm(p *obj.Prog) {
	fmt.Printf("%v\n", p)
}

func span7(ctxt *obj.Link, cursym *obj.LSym) {
	p := cursym.Text
	if p == nil || p.Link == nil { // handle external functions and ELF section symbols
		return
	}
	ctxt.Cursym = cursym
	ctxt.Autosize = int32(p.To.Offset&0xffffffff) + 8

	if oprange[AAND&obj.AMask] == nil {
		buildop(ctxt)
	}

	bflag := 1
	c := int64(0)
	p.Pc = c
	var m int
	var o *Optab
	for p = p.Link; p != nil; p = p.Link {
		ctxt.Curp = p
		if p.As == ADWORD && (c&7) != 0 {
			c += 4
		}
		p.Pc = c
		o = oplook(ctxt, p)
		m = int(o.size)
		if m == 0 {
			if p.As != obj.ANOP && p.As != obj.AFUNCDATA && p.As != obj.APCDATA && p.As != obj.AUSEFIELD {
				ctxt.Diag("zero-width instruction\n%v", p)
			}
			continue
		}

		switch o.flag & (LFROM | LTO) {
		case LFROM:
			addpool(ctxt, p, &p.From)

		case LTO:
			addpool(ctxt, p, &p.To)
			break
		}

		if p.As == AB || p.As == obj.ARET || p.As == AERET { /* TODO: other unconditional operations */
			checkpool(ctxt, p, 0)
		}
		c += int64(m)
		if ctxt.Blitrl != nil {
			checkpool(ctxt, p, 1)
		}
	}

	cursym.Size = c

	/*
	 * if any procedure is large enough to
	 * generate a large SBRA branch, then
	 * generate extra passes putting branches
	 * around jmps to fix. this is rare.
	 */
	for bflag != 0 {
		if ctxt.Debugvlog != 0 {
			fmt.Fprintf(ctxt.Bso, "%5.2f span1\n", obj.Cputime())
		}
		bflag = 0
		c = 0
		for p = cursym.Text.Link; p != nil; p = p.Link {
			if p.As == ADWORD && (c&7) != 0 {
				c += 4
			}
			p.Pc = c
			o = oplook(ctxt, p)

			/* very large branches */
			if o.type_ == 7 && p.Pcond != nil {
				otxt := p.Pcond.Pc - c
				if otxt <= -(1<<18)+10 || otxt >= (1<<18)-10 {
					q := ctxt.NewProg()
					q.Link = p.Link
					p.Link = q
					q.As = AB
					q.To.Type = obj.TYPE_BRANCH
					q.Pcond = p.Pcond
					p.Pcond = q
					q = ctxt.NewProg()
					q.Link = p.Link
					p.Link = q
					q.As = AB
					q.To.Type = obj.TYPE_BRANCH
					q.Pcond = q.Link.Link
					bflag = 1
				}
			}
			m = int(o.size)

			if m == 0 {
				if p.As != obj.ANOP && p.As != obj.AFUNCDATA && p.As != obj.APCDATA && p.As != obj.AUSEFIELD {
					ctxt.Diag("zero-width instruction\n%v", p)
				}
				continue
			}

			c += int64(m)
		}
	}

	c += -c & (FuncAlign - 1)
	cursym.Size = c

	/*
	 * lay out the code, emitting code and data relocations.
	 */
	cursym.Grow(cursym.Size)
	bp := cursym.P
	psz := int32(0)
	var i int
	var out [6]uint32
	for p := cursym.Text.Link; p != nil; p = p.Link {
		ctxt.Pc = p.Pc
		ctxt.Curp = p
		o = oplook(ctxt, p)

		// need to align DWORDs on 8-byte boundary. The ISA doesn't
		// require it, but the various 64-bit loads we generate assume it.
		if o.as == ADWORD && psz%8 != 0 {
			bp[3] = 0
			bp[2] = bp[3]
			bp[1] = bp[2]
			bp[0] = bp[1]
			bp = bp[4:]
			psz += 4
		}

		if int(o.size) > 4*len(out) {
			log.Fatalf("out array in span7 is too small, need at least %d for %v", o.size/4, p)
		}
		asmout(ctxt, p, o, out[:])
		for i = 0; i < int(o.size/4); i++ {
			ctxt.Arch.ByteOrder.PutUint32(bp, out[i])
			bp = bp[4:]
			psz += 4
		}
	}
}

/*
 * when the first reference to the literal pool threatens
 * to go out of range of a 1Mb PC-relative offset
 * drop the pool now, and branch round it.
 */
func checkpool(ctxt *obj.Link, p *obj.Prog, skip int) {
	if pool.size >= 0xffff0 || !ispcdisp(int32(p.Pc+4+int64(pool.size)-int64(pool.start)+8)) {
		flushpool(ctxt, p, skip)
	} else if p.Link == nil {
		flushpool(ctxt, p, 2)
	}
}

func flushpool(ctxt *obj.Link, p *obj.Prog, skip int) {
	if ctxt.Blitrl != nil {
		if skip != 0 {
			if ctxt.Debugvlog != 0 && skip == 1 {
				fmt.Printf("note: flush literal pool at %#x: len=%d ref=%x\n", uint64(p.Pc+4), pool.size, pool.start)
			}
			q := ctxt.NewProg()
			q.As = AB
			q.To.Type = obj.TYPE_BRANCH
			q.Pcond = p.Link
			q.Link = ctxt.Blitrl
			q.Lineno = p.Lineno
			ctxt.Blitrl = q
		} else if p.Pc+int64(pool.size)-int64(pool.start) < maxPCDisp {
			return
		}

		// The line number for constant pool entries doesn't really matter.
		// We set it to the line number of the preceding instruction so that
		// there are no deltas to encode in the pc-line tables.
		for q := ctxt.Blitrl; q != nil; q = q.Link {
			q.Lineno = p.Lineno
		}

		ctxt.Elitrl.Link = p.Link
		p.Link = ctxt.Blitrl

		ctxt.Blitrl = nil /* BUG: should refer back to values until out-of-range */
		ctxt.Elitrl = nil
		pool.size = 0
		pool.start = 0
	}
}

/*
 * TODO: hash
 */
func addpool(ctxt *obj.Link, p *obj.Prog, a *obj.Addr) {
	c := aclass(ctxt, a)
	t := *ctxt.NewProg()
	t.As = AWORD
	sz := 4

	// MOVW foo(SB), R is actually
	//	MOV addr, REGTEMP
	//	MOVW REGTEMP, R
	// where addr is the address of the DWORD containing the address of foo.
	if p.As == AMOVD || c == C_ADDR || c == C_VCON {
		t.As = ADWORD
		sz = 8
	}

	switch c {
	// TODO(aram): remove.
	default:
		if a.Name != obj.NAME_EXTERN {
			fmt.Printf("addpool: %v in %v shouldn't go to default case\n", DRconv(c), p)
		}

		t.To.Offset = a.Offset
		t.To.Sym = a.Sym
		t.To.Type = a.Type
		t.To.Name = a.Name

		/* This is here to work around a bug where we generate negative
		operands that match C_MOVCON, but we use them with
		instructions that only accept unsigned immediates. This
		will cause oplook to return a variant of the instruction
		that loads the negative constant from memory, rather than
		using the immediate form. Because of that load, we get here,
		so we need to know what to do with C_MOVCON.

		The correct fix is to use the "negation" instruction variant,
		e.g. CMN $1, R instead of CMP $-1, R, or SUB $1, R instead
		of ADD $-1, R. */
	case C_MOVCON,

		/* This is here because MOV uint12<<12, R is disabled in optab.
		Because of this, we need to load the constant from memory. */
		C_ADDCON,

		/* These are here because they are disabled in optab.
		Because of this, we need to load the constant from memory. */
		C_BITCON,
		C_ABCON,
		C_MBCON,
		C_PSAUTO,
		C_PPAUTO,
		C_UAUTO4K,
		C_UAUTO8K,
		C_UAUTO16K,
		C_UAUTO32K,
		C_UAUTO64K,
		C_NSAUTO,
		C_NPAUTO,
		C_LAUTO,
		C_PPOREG,
		C_PSOREG,
		C_UOREG4K,
		C_UOREG8K,
		C_UOREG16K,
		C_UOREG32K,
		C_UOREG64K,
		C_NSOREG,
		C_NPOREG,
		C_LOREG,
		C_LACON,
		C_LCON,
		C_VCON:
		if a.Name == obj.NAME_EXTERN {
			fmt.Printf("addpool: %v in %v needs reloc\n", DRconv(c), p)
		}

		t.To.Type = obj.TYPE_CONST
		t.To.Offset = ctxt.Instoffset
		break
	}

	for q := ctxt.Blitrl; q != nil; q = q.Link { /* could hash on t.t0.offset */
		if q.To == t.To {
			p.Pcond = q
			return
		}
	}

	q := ctxt.NewProg()
	*q = t
	q.Pc = int64(pool.size)
	if ctxt.Blitrl == nil {
		ctxt.Blitrl = q
		pool.start = uint32(p.Pc)
	} else {
		ctxt.Elitrl.Link = q
	}
	ctxt.Elitrl = q
	pool.size = -pool.size & (FuncAlign - 1)
	pool.size += uint32(sz)
	p.Pcond = q
}

func regoff(ctxt *obj.Link, a *obj.Addr) uint32 {
	ctxt.Instoffset = 0
	aclass(ctxt, a)
	return uint32(ctxt.Instoffset)
}

// Maximum PC-relative displacement.
// The actual limit is ±2²⁰, but we are conservative
// to avoid needing to recompute the literal pool flush points
// as span-dependent jumps are enlarged.
const maxPCDisp = 512 * 1024

// ispcdisp reports whether v is a valid PC-relative displacement.
func ispcdisp(v int32) bool {
	return -maxPCDisp < v && v < maxPCDisp && v&3 == 0
}

func isaddcon(v int64) bool {
	/* uimm12 or uimm24? */
	if v < 0 {
		return false
	}
	if (v & 0xFFF) == 0 {
		v >>= 12
	}
	return v <= 0xFFF
}

func isbitcon(v uint64) bool {
	/*  fancy bimm32 or bimm64? */
	// TODO(aram):
	return false
	// return findmask(v) != nil || (v>>32) == 0 && findmask(v|(v<<32)) != nil
}

func autoclass(l int64) int {
	if l < 0 {
		if l >= -256 {
			return C_NSAUTO
		}
		if l >= -512 && (l&7) == 0 {
			return C_NPAUTO
		}
		return C_LAUTO
	}

	if l <= 255 {
		return C_PSAUTO
	}
	if l <= 504 && (l&7) == 0 {
		return C_PPAUTO
	}
	if l <= 4095 {
		return C_UAUTO4K
	}
	if l <= 8190 && (l&1) == 0 {
		return C_UAUTO8K
	}
	if l <= 16380 && (l&3) == 0 {
		return C_UAUTO16K
	}
	if l <= 32760 && (l&7) == 0 {
		return C_UAUTO32K
	}
	if l <= 65520 && (l&0xF) == 0 {
		return C_UAUTO64K
	}
	return C_LAUTO
}

func oregclass(l int64) int {
	if l == 0 {
		return C_ZOREG
	}
	return autoclass(l) - C_NPAUTO + C_NPOREG
}

/*
 * given an offset v and a class c (see above)
 * return the offset value to use in the instruction,
 * scaled if necessary
 */
func offsetshift(ctxt *obj.Link, v int64, c int) int64 {
	s := 0
	if c >= C_SEXT1 && c <= C_SEXT16 {
		s = c - C_SEXT1
	} else if c >= C_UAUTO4K && c <= C_UAUTO64K {
		s = c - C_UAUTO4K
	} else if c >= C_UOREG4K && c <= C_UOREG64K {
		s = c - C_UOREG4K
	}
	vs := v >> uint(s)
	if vs<<uint(s) != v {
		ctxt.Diag("odd offset: %d\n%v", v, ctxt.Curp)
	}
	return vs
}

/*
 * if v contains a single 16-bit value aligned
 * on a 16-bit field, and thus suitable for movk/movn,
 * return the field index 0 to 3; otherwise return -1
 */
func movcon(v int64) int {
	for s := 0; s < 64; s += 16 {
		if (uint64(v) &^ (uint64(0xFFFF) << uint(s))) == 0 {
			return s / 16
		}
	}
	return -1
}

func rclass(r int16) int {
	switch {
	case REG_R0 <= r && r <= REG_R30: // not 31
		return C_REG
	case r == REGZERO:
		return C_ZCON
	case REG_F0 <= r && r <= REG_F31:
		return C_FREG
	case REG_V0 <= r && r <= REG_V31:
		return C_VREG
	case COND_EQ <= r && r <= COND_NV:
		return C_COND
	case r == REGSP:
		return C_RSP
	case r&REG_EXT != 0:
		return C_EXTREG
	case r >= REG_SPECIAL:
		return C_SPR
	}
	return C_GOK
}

func aclass(ctxt *obj.Link, a *obj.Addr) int {
	switch a.Type {
	case obj.TYPE_NONE:
		return C_NONE

	case obj.TYPE_REG:
		return rclass(a.Reg)

	case obj.TYPE_REGREG:
		return C_PAIR

	case obj.TYPE_SHIFT:
		return C_SHIFT

	case obj.TYPE_MEM:
		switch a.Name {
		case obj.NAME_EXTERN, obj.NAME_STATIC:
			if a.Sym == nil {
				break
			}
			ctxt.Instoffset = a.Offset
			if a.Sym != nil { // use relocation
				if a.Sym.Type == obj.STLSBSS {
					if ctxt.Flag_shared {
						return C_TLS_IE
					} else {
						return C_TLS_LE
					}
				}
				return C_ADDR
			}
			return C_LEXT

		case obj.NAME_GOTREF:
			return C_GOTADDR

		case obj.NAME_AUTO:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset
			return autoclass(ctxt.Instoffset)

		case obj.NAME_PARAM:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset + 8
			return autoclass(ctxt.Instoffset)

		case obj.NAME_NONE:
			ctxt.Instoffset = a.Offset
			return oregclass(ctxt.Instoffset)
		}
		return C_GOK

	case obj.TYPE_FCONST:
		return C_FCON

	case obj.TYPE_TEXTSIZE:
		return C_TEXTSIZE

	case obj.TYPE_CONST, obj.TYPE_ADDR:
		switch a.Name {
		case obj.NAME_NONE:
			ctxt.Instoffset = a.Offset
			if a.Reg != 0 && a.Reg != REGZERO {
				goto aconsize
			}
			v := ctxt.Instoffset
			if v == 0 {
				return C_ZCON
			}
			if isaddcon(v) {
				if v <= 0xFFF {
					return C_ADDCON0
				}
				if isbitcon(uint64(v)) {
					return C_ABCON
				}
				return C_ADDCON
			}

			t := movcon(v)
			if t >= 0 {
				if isbitcon(uint64(v)) {
					return C_MBCON
				}
				return C_MOVCON
			}

			t = movcon(^v)
			if t >= 0 {
				if isbitcon(uint64(v)) {
					return C_MBCON
				}
				return C_MOVCON
			}

			if isbitcon(uint64(v)) {
				return C_BITCON
			}

			if uint64(v) == uint64(uint32(v)) || v == int64(int32(v)) {
				return C_LCON
			}
			return C_VCON

		case obj.NAME_EXTERN, obj.NAME_STATIC:
			if a.Sym == nil {
				break
			}
			if a.Sym.Type == obj.STLSBSS {
				ctxt.Diag("taking address of TLS variable is not supported")
			}
			ctxt.Instoffset = a.Offset
			return C_VCONADDR

		case obj.NAME_AUTO:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset
			goto aconsize

		case obj.NAME_PARAM:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset + 8
			goto aconsize
		}
		return C_GOK

	aconsize:
		if isaddcon(ctxt.Instoffset) {
			return C_AACON
		}
		return C_LACON

	case obj.TYPE_BRANCH:
		return C_SBRA
	}

	return C_GOK
}

func oplook(ctxt *obj.Link, p *obj.Prog) *Optab {
	a1 := int(p.Optab)
	if a1 != 0 {
		return &optab[a1-1]
	}
	a1 = int(p.From.Class)
	if a1 == 0 {
		a1 = aclass(ctxt, &p.From) + 1
		p.From.Class = int8(a1)
	}

	a1--
	a3 := int(p.To.Class)
	if a3 == 0 {
		a3 = aclass(ctxt, &p.To) + 1
		p.To.Class = int8(a3)
	}

	a3--
	a2 := C_NONE
	if p.Reg != 0 {
		a2 = rclass(p.Reg)
	}

	if false {
		fmt.Printf("oplook %v %d %d %d\n", obj.Aconv(p.As), a1, a2, a3)
		fmt.Printf("\t\t%d %d\n", p.From.Type, p.To.Type)
	}

	ops := oprange[p.As&obj.AMask]
	c1 := &xcmp[a1]
	c2 := &xcmp[a2]
	c3 := &xcmp[a3]
	c4 := &xcmp[p.Scond>>5]
	for i := range ops {
		op := &ops[i]
		if (int(op.a2) == a2 || c2[op.a2]) && c4[op.scond>>5] && c1[op.a1] && c3[op.a3] {
			p.Optab = uint16(cap(optab) - cap(ops) + i + 1)
			return op
		}
	}

	ctxt.Diag("illegal combination %v %v %v %v, %d %d", p, DRconv(a1), DRconv(a2), DRconv(a3), p.From.Type, p.To.Type)
	prasm(p)
	if ops == nil {
		ops = optab
	}
	return &ops[0]
}

func cmp(a int, b int) bool {
	if a == b {
		return true
	}
	switch a {
	case C_RSP:
		if b == C_REG {
			return true
		}

	case C_REG:
		if b == C_ZCON {
			return true
		}

	case C_ADDCON0:
		if b == C_ZCON {
			return true
		}

	case C_ADDCON:
		if b == C_ZCON || b == C_ADDCON0 || b == C_ABCON {
			return true
		}

	case C_BITCON:
		if b == C_ABCON || b == C_MBCON {
			return true
		}

	case C_MOVCON:
		if b == C_MBCON || b == C_ZCON || b == C_ADDCON0 {
			return true
		}

	case C_LCON:
		if b == C_ZCON || b == C_BITCON || b == C_ADDCON || b == C_ADDCON0 || b == C_ABCON || b == C_MBCON || b == C_MOVCON {
			return true
		}

	case C_VCON:
		return cmp(C_LCON, b)

	case C_LACON:
		if b == C_AACON {
			return true
		}

	case C_SEXT2:
		if b == C_SEXT1 {
			return true
		}

	case C_SEXT4:
		if b == C_SEXT1 || b == C_SEXT2 {
			return true
		}

	case C_SEXT8:
		if b >= C_SEXT1 && b <= C_SEXT4 {
			return true
		}

	case C_SEXT16:
		if b >= C_SEXT1 && b <= C_SEXT8 {
			return true
		}

	case C_LEXT:
		if b >= C_SEXT1 && b <= C_SEXT16 {
			return true
		}

	case C_PPAUTO:
		if b == C_PSAUTO {
			return true
		}

	case C_UAUTO4K:
		if b == C_PSAUTO || b == C_PPAUTO {
			return true
		}

	case C_UAUTO8K:
		return cmp(C_UAUTO4K, b)

	case C_UAUTO16K:
		return cmp(C_UAUTO8K, b)

	case C_UAUTO32K:
		return cmp(C_UAUTO16K, b)

	case C_UAUTO64K:
		return cmp(C_UAUTO32K, b)

	case C_NPAUTO:
		return cmp(C_NSAUTO, b)

	case C_LAUTO:
		return cmp(C_NPAUTO, b) || cmp(C_UAUTO64K, b)

	case C_PSOREG:
		if b == C_ZOREG {
			return true
		}

	case C_PPOREG:
		if b == C_ZOREG || b == C_PSOREG {
			return true
		}

	case C_UOREG4K:
		if b == C_ZOREG || b == C_PSAUTO || b == C_PSOREG || b == C_PPAUTO || b == C_PPOREG {
			return true
		}

	case C_UOREG8K:
		return cmp(C_UOREG4K, b)

	case C_UOREG16K:
		return cmp(C_UOREG8K, b)

	case C_UOREG32K:
		return cmp(C_UOREG16K, b)

	case C_UOREG64K:
		return cmp(C_UOREG32K, b)

	case C_NPOREG:
		return cmp(C_NSOREG, b)

	case C_LOREG:
		return cmp(C_NPOREG, b) || cmp(C_UOREG64K, b)

	case C_LBRA:
		if b == C_SBRA {
			return true
		}
	}

	return false
}

type ocmp []Optab

func (x ocmp) Len() int {
	return len(x)
}

func (x ocmp) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

func (x ocmp) Less(i, j int) bool {
	p1 := &x[i]
	p2 := &x[j]
	if p1.as != p2.as {
		return p1.as < p2.as
	}
	if p1.a1 != p2.a1 {
		return p1.a1 < p2.a1
	}
	if p1.a2 != p2.a2 {
		return p1.a2 < p2.a2
	}
	if p1.a3 != p2.a3 {
		return p1.a3 < p2.a3
	}
	if p1.scond != p2.scond {
		return p1.scond < p2.scond
	}
	return false
}

func oprangeset(a obj.As, t []Optab) {
	oprange[a&obj.AMask] = t
}

func buildop(ctxt *obj.Link) {
	var n int
	for i := 0; i < C_GOK; i++ {
		for n = 0; n < C_GOK; n++ {
			if cmp(n, i) {
				xcmp[i][n] = true
			}
		}
	}
	for n = 0; optab[n].as != obj.AXXX; n++ {
	}
	sort.Sort(ocmp(optab[:n]))
	for i := 0; i < n; i++ {
		r := optab[i].as
		start := i
		for optab[i].as == r {
			i++
		}
		t := optab[start:i]
		i--
		oprangeset(r, t)
		switch r {
		default:
			ctxt.Diag("unknown op in build: %v", obj.Aconv(r))
			log.Fatalf("bad code")

		case AADD:
			oprangeset(AADDS, t)
			oprangeset(ASUB, t)
			oprangeset(ASUBS, t)
			oprangeset(AADDW, t)
			oprangeset(AADDSW, t)
			oprangeset(ASUBW, t)
			oprangeset(ASUBSW, t)

		case AAND: /* logical immediate, logical shifted register */
			oprangeset(AANDS, t)

			oprangeset(AANDSW, t)
			oprangeset(AANDW, t)
			oprangeset(AEOR, t)
			oprangeset(AEORW, t)
			oprangeset(AORR, t)
			oprangeset(AORRW, t)

		case ABIC: /* only logical shifted register */
			oprangeset(ABICS, t)

			oprangeset(ABICSW, t)
			oprangeset(ABICW, t)
			oprangeset(AEON, t)
			oprangeset(AEONW, t)
			oprangeset(AORN, t)
			oprangeset(AORNW, t)

		case ANEG:
			oprangeset(ANEGS, t)
			oprangeset(ANEGSW, t)
			oprangeset(ANEGW, t)

		case AADC: /* rn=Rd */
			oprangeset(AADCW, t)

			oprangeset(AADCS, t)
			oprangeset(AADCSW, t)
			oprangeset(ASBC, t)
			oprangeset(ASBCW, t)
			oprangeset(ASBCS, t)
			oprangeset(ASBCSW, t)

		case ANGC: /* rn=REGZERO */
			oprangeset(ANGCW, t)

			oprangeset(ANGCS, t)
			oprangeset(ANGCSW, t)

		case ACMP:
			oprangeset(ACMPW, t)
			oprangeset(ACMN, t)
			oprangeset(ACMNW, t)

		case ATST:
			oprangeset(ATSTW, t)

			/* register/register, and shifted */
		case AMVN:
			oprangeset(AMVNW, t)

		case AMOVK:
			oprangeset(AMOVKW, t)
			oprangeset(AMOVN, t)
			oprangeset(AMOVNW, t)
			oprangeset(AMOVZ, t)
			oprangeset(AMOVZW, t)

		case ABEQ:
			oprangeset(ABNE, t)
			oprangeset(ABCS, t)
			oprangeset(ABHS, t)
			oprangeset(ABCC, t)
			oprangeset(ABLO, t)
			oprangeset(ABMI, t)
			oprangeset(ABPL, t)
			oprangeset(ABVS, t)
			oprangeset(ABVC, t)
			oprangeset(ABHI, t)
			oprangeset(ABLS, t)
			oprangeset(ABGE, t)
			oprangeset(ABLT, t)
			oprangeset(ABGT, t)
			oprangeset(ABLE, t)

		case ALSL:
			oprangeset(ALSLW, t)
			oprangeset(ALSR, t)
			oprangeset(ALSRW, t)
			oprangeset(AASR, t)
			oprangeset(AASRW, t)
			oprangeset(AROR, t)
			oprangeset(ARORW, t)

		case ACLS:
			oprangeset(ACLSW, t)
			oprangeset(ACLZ, t)
			oprangeset(ACLZW, t)
			oprangeset(ARBIT, t)
			oprangeset(ARBITW, t)
			oprangeset(AREV, t)
			oprangeset(AREVW, t)
			oprangeset(AREV16, t)
			oprangeset(AREV16W, t)
			oprangeset(AREV32, t)

		case ASDIV:
			oprangeset(ASDIVW, t)
			oprangeset(AUDIV, t)
			oprangeset(AUDIVW, t)
			oprangeset(ACRC32B, t)
			oprangeset(ACRC32CB, t)
			oprangeset(ACRC32CH, t)
			oprangeset(ACRC32CW, t)
			oprangeset(ACRC32CX, t)
			oprangeset(ACRC32H, t)
			oprangeset(ACRC32W, t)
			oprangeset(ACRC32X, t)

		case AMADD:
			oprangeset(AMADDW, t)
			oprangeset(AMSUB, t)
			oprangeset(AMSUBW, t)
			oprangeset(ASMADDL, t)
			oprangeset(ASMSUBL, t)
			oprangeset(AUMADDL, t)
			oprangeset(AUMSUBL, t)

		case AREM:
			oprangeset(AREMW, t)
			oprangeset(AUREM, t)
			oprangeset(AUREMW, t)

		case AMUL:
			oprangeset(AMULW, t)
			oprangeset(AMNEG, t)
			oprangeset(AMNEGW, t)
			oprangeset(ASMNEGL, t)
			oprangeset(ASMULL, t)
			oprangeset(ASMULH, t)
			oprangeset(AUMNEGL, t)
			oprangeset(AUMULH, t)
			oprangeset(AUMULL, t)

		case AMOVB:
			oprangeset(AMOVBU, t)

		case AMOVH:
			oprangeset(AMOVHU, t)

		case AMOVW:
			oprangeset(AMOVWU, t)

		case ABFM:
			oprangeset(ABFMW, t)
			oprangeset(ASBFM, t)
			oprangeset(ASBFMW, t)
			oprangeset(AUBFM, t)
			oprangeset(AUBFMW, t)

		case ABFI:
			oprangeset(ABFIW, t)
			oprangeset(ABFXIL, t)
			oprangeset(ABFXILW, t)
			oprangeset(ASBFIZ, t)
			oprangeset(ASBFIZW, t)
			oprangeset(ASBFX, t)
			oprangeset(ASBFXW, t)
			oprangeset(AUBFIZ, t)
			oprangeset(AUBFIZW, t)
			oprangeset(AUBFX, t)
			oprangeset(AUBFXW, t)

		case AEXTR:
			oprangeset(AEXTRW, t)

		case ASXTB:
			oprangeset(ASXTBW, t)
			oprangeset(ASXTH, t)
			oprangeset(ASXTHW, t)
			oprangeset(ASXTW, t)
			oprangeset(AUXTB, t)
			oprangeset(AUXTH, t)
			oprangeset(AUXTW, t)
			oprangeset(AUXTBW, t)
			oprangeset(AUXTHW, t)

		case ACCMN:
			oprangeset(ACCMNW, t)
			oprangeset(ACCMP, t)
			oprangeset(ACCMPW, t)

		case ACSEL:
			oprangeset(ACSELW, t)
			oprangeset(ACSINC, t)
			oprangeset(ACSINCW, t)
			oprangeset(ACSINV, t)
			oprangeset(ACSINVW, t)
			oprangeset(ACSNEG, t)
			oprangeset(ACSNEGW, t)

			// aliases Rm=Rn, !cond
			oprangeset(ACINC, t)

			oprangeset(ACINCW, t)
			oprangeset(ACINV, t)
			oprangeset(ACINVW, t)
			oprangeset(ACNEG, t)
			oprangeset(ACNEGW, t)

			// aliases, Rm=Rn=REGZERO, !cond
		case ACSET:
			oprangeset(ACSETW, t)

			oprangeset(ACSETM, t)
			oprangeset(ACSETMW, t)

		case AMOVD,
			AMOVBU,
			AB,
			ABL,
			AWORD,
			ADWORD,
			obj.ARET,
			obj.ATEXT,
			ASTP,
			ALDP:
			break

		case AERET:
			oprangeset(AWFE, t)
			oprangeset(AWFI, t)
			oprangeset(AYIELD, t)
			oprangeset(ASEV, t)
			oprangeset(ASEVL, t)
			oprangeset(ADRPS, t)

		case ACBZ:
			oprangeset(ACBZW, t)
			oprangeset(ACBNZ, t)
			oprangeset(ACBNZW, t)

		case ATBZ:
			oprangeset(ATBNZ, t)

		case AADR, AADRP:
			break

		case ACLREX:
			break

		case ASVC:
			oprangeset(AHLT, t)
			oprangeset(AHVC, t)
			oprangeset(ASMC, t)
			oprangeset(ABRK, t)
			oprangeset(ADCPS1, t)
			oprangeset(ADCPS2, t)
			oprangeset(ADCPS3, t)

		case AFADDS:
			oprangeset(AFADDD, t)
			oprangeset(AFSUBS, t)
			oprangeset(AFSUBD, t)
			oprangeset(AFMULS, t)
			oprangeset(AFMULD, t)
			oprangeset(AFNMULS, t)
			oprangeset(AFNMULD, t)
			oprangeset(AFDIVS, t)
			oprangeset(AFMAXD, t)
			oprangeset(AFMAXS, t)
			oprangeset(AFMIND, t)
			oprangeset(AFMINS, t)
			oprangeset(AFMAXNMD, t)
			oprangeset(AFMAXNMS, t)
			oprangeset(AFMINNMD, t)
			oprangeset(AFMINNMS, t)
			oprangeset(AFDIVD, t)

		case AFCVTSD:
			oprangeset(AFCVTDS, t)
			oprangeset(AFABSD, t)
			oprangeset(AFABSS, t)
			oprangeset(AFNEGD, t)
			oprangeset(AFNEGS, t)
			oprangeset(AFSQRTD, t)
			oprangeset(AFSQRTS, t)
			oprangeset(AFRINTNS, t)
			oprangeset(AFRINTND, t)
			oprangeset(AFRINTPS, t)
			oprangeset(AFRINTPD, t)
			oprangeset(AFRINTMS, t)
			oprangeset(AFRINTMD, t)
			oprangeset(AFRINTZS, t)
			oprangeset(AFRINTZD, t)
			oprangeset(AFRINTAS, t)
			oprangeset(AFRINTAD, t)
			oprangeset(AFRINTXS, t)
			oprangeset(AFRINTXD, t)
			oprangeset(AFRINTIS, t)
			oprangeset(AFRINTID, t)
			oprangeset(AFCVTDH, t)
			oprangeset(AFCVTHS, t)
			oprangeset(AFCVTHD, t)
			oprangeset(AFCVTSH, t)

		case AFCMPS:
			oprangeset(AFCMPD, t)
			oprangeset(AFCMPES, t)
			oprangeset(AFCMPED, t)

		case AFCCMPS:
			oprangeset(AFCCMPD, t)
			oprangeset(AFCCMPES, t)
			oprangeset(AFCCMPED, t)

		case AFCSELD:
			oprangeset(AFCSELS, t)

		case AFMOVS, AFMOVD:
			break

		case AFCVTZSD:
			oprangeset(AFCVTZSDW, t)
			oprangeset(AFCVTZSS, t)
			oprangeset(AFCVTZSSW, t)
			oprangeset(AFCVTZUD, t)
			oprangeset(AFCVTZUDW, t)
			oprangeset(AFCVTZUS, t)
			oprangeset(AFCVTZUSW, t)

		case ASCVTFD:
			oprangeset(ASCVTFS, t)
			oprangeset(ASCVTFWD, t)
			oprangeset(ASCVTFWS, t)
			oprangeset(AUCVTFD, t)
			oprangeset(AUCVTFS, t)
			oprangeset(AUCVTFWD, t)
			oprangeset(AUCVTFWS, t)

		case ASYS:
			oprangeset(AAT, t)
			oprangeset(ADC, t)
			oprangeset(AIC, t)
			oprangeset(ATLBI, t)

		case ASYSL, AHINT:
			break

		case ADMB:
			oprangeset(ADSB, t)
			oprangeset(AISB, t)

		case AMRS, AMSR:
			break

		case ALDAR:
			oprangeset(ALDARW, t)
			fallthrough

		case ALDXR:
			oprangeset(ALDXRB, t)
			oprangeset(ALDXRH, t)
			oprangeset(ALDXRW, t)

		case ALDAXR:
			oprangeset(ALDAXRW, t)

		case ALDXP:
			oprangeset(ALDXPW, t)

		case ASTLR:
			oprangeset(ASTLRW, t)

		case ASTXR:
			oprangeset(ASTXRB, t)
			oprangeset(ASTXRH, t)
			oprangeset(ASTXRW, t)

		case ASTLXR:
			oprangeset(ASTLXRW, t)

		case ASTXP:
			oprangeset(ASTXPW, t)

		case AAESD:
			oprangeset(AAESE, t)
			oprangeset(AAESMC, t)
			oprangeset(AAESIMC, t)
			oprangeset(ASHA1H, t)
			oprangeset(ASHA1SU1, t)
			oprangeset(ASHA256SU0, t)

		case ASHA1C:
			oprangeset(ASHA1P, t)
			oprangeset(ASHA1M, t)
			oprangeset(ASHA1SU0, t)
			oprangeset(ASHA256H, t)
			oprangeset(ASHA256H2, t)
			oprangeset(ASHA256SU1, t)

		case obj.ANOP,
			obj.AUNDEF,
			obj.AUSEFIELD,
			obj.AFUNCDATA,
			obj.APCDATA,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			break
		}
	}
}

func chipfloat7(ctxt *obj.Link, e float64) int {
	ei := math.Float64bits(e)
	l := uint32(int32(ei))
	h := uint32(int32(ei >> 32))

	if l != 0 || h&0xffff != 0 {
		return -1
	}
	h1 := h & 0x7fc00000
	if h1 != 0x40000000 && h1 != 0x3fc00000 {
		return -1
	}
	n := 0

	// sign bit (a)
	if h&0x80000000 != 0 {
		n |= 1 << 7
	}

	// exp sign bit (b)
	if h1 == 0x3fc00000 {
		n |= 1 << 6
	}

	// rest of exp and mantissa (cd-efgh)
	n |= int((h >> 16) & 0x3f)

	//print("match %.8lux %.8lux %d\n", l, h, n);
	return n
}

/* form offset parameter to SYS; special register number */
func SYSARG5(op0 int, op1 int, Cn int, Cm int, op2 int) int {
	return op0<<19 | op1<<16 | Cn<<12 | Cm<<8 | op2<<5
}

func SYSARG4(op1 int, Cn int, Cm int, op2 int) int {
	return SYSARG5(0, op1, Cn, Cm, op2)
}

func asmout(ctxt *obj.Link, p *obj.Prog, o *Optab, out []uint32) {
	o1 := uint32(0)
	o2 := uint32(0)
	o3 := uint32(0)
	o4 := uint32(0)
	o5 := uint32(0)
	if false { /*debug['P']*/
		fmt.Printf("%x: %v\ttype %d\n", uint32(p.Pc), p, o.type_)
	}
	switch o.type_ {
	default:
		ctxt.Diag("unknown asm %d", o.type_)
		prasm(p)

	case 0: /* pseudo ops */
		break

	case 1: /* op Rm,[Rn],Rd; default Rn=Rd -> op Rm<<0,[Rn,]Rd (shifted register) */
		o1 = oprrr(ctxt, p.As)

		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = REGZERO
		}
		if r == 0 {
			r = rt
		}
		o1 |= (uint32(rf&31) << 16) | (uint32(r&31) << 5) | uint32(rt&31)

	case 2: /* add/sub $(uimm12|uimm24)[,R],R; cmp $(uimm12|uimm24),R */
		o1 = opirr(ctxt, p.As)

		rt := int(p.To.Reg)
		if p.To.Type == obj.TYPE_NONE {
			if (o1 & Sbit) == 0 {
				ctxt.Diag("ineffective ZR destination\n%v", p)
			}
			rt = REGZERO
		}

		r := int(p.Reg)
		if r == 0 {
			r = rt
		}
		v := int32(regoff(ctxt, &p.From))
		o1 = oaddi(ctxt, int32(o1), v, r, rt)

	case 3: /* op R<<n[,R],R (shifted register) */
		o1 = oprrr(ctxt, p.As)

		o1 |= uint32(p.From.Offset) /* includes reg, op, etc */
		rt := int(p.To.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = REGZERO
		}
		r := int(p.Reg)
		if p.As == AMVN || p.As == AMVNW {
			r = REGZERO
		} else if r == 0 {
			r = rt
		}
		o1 |= (uint32(r&31) << 5) | uint32(rt&31)

	case 4: /* mov $addcon, R; mov $recon, R; mov $racon, R */
		o1 = opirr(ctxt, p.As)

		rt := int(p.To.Reg)
		r := int(o.param)
		if r == 0 {
			r = REGZERO
		} else if r == REGFROM {
			r = int(p.From.Reg)
		}
		if r == 0 {
			r = REGSP
		}
		v := int32(regoff(ctxt, &p.From))
		if (v & 0xFFF000) != 0 {
			v >>= 12
			o1 |= 1 << 22 /* shift, by 12 */
		}

		o1 |= ((uint32(v) & 0xFFF) << 10) | (uint32(r&31) << 5) | uint32(rt&31)

	case 5: /* b s; bl s */
		o1 = opbra(ctxt, p.As)

		if p.To.Sym == nil {
			o1 |= uint32(brdist(ctxt, p, 0, 26, 2))
			break
		}

		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.To.Sym
		rel.Add = p.To.Offset
		rel.Type = obj.R_CALLARM64

	case 6: /* b ,O(R); bl ,O(R) */
		o1 = opbrr(ctxt, p.As)

		o1 |= uint32(p.To.Reg&31) << 5
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 0
		rel.Type = obj.R_CALLIND

	case 7: /* beq s */
		o1 = opbra(ctxt, p.As)

		o1 |= uint32(brdist(ctxt, p, 0, 19, 2) << 5)

	case 8: /* lsl $c,[R],R -> ubfm $(W-1)-c,$(-c MOD (W-1)),Rn,Rd */
		rt := int(p.To.Reg)

		rf := int(p.Reg)
		if rf == 0 {
			rf = rt
		}
		v := int32(p.From.Offset)
		switch p.As {
		case AASR:
			o1 = opbfm(ctxt, ASBFM, int(v), 63, rf, rt)

		case AASRW:
			o1 = opbfm(ctxt, ASBFMW, int(v), 31, rf, rt)

		case ALSL:
			o1 = opbfm(ctxt, AUBFM, int((64-v)&63), int(63-v), rf, rt)

		case ALSLW:
			o1 = opbfm(ctxt, AUBFMW, int((32-v)&31), int(31-v), rf, rt)

		case ALSR:
			o1 = opbfm(ctxt, AUBFM, int(v), 63, rf, rt)

		case ALSRW:
			o1 = opbfm(ctxt, AUBFMW, int(v), 31, rf, rt)

		case AROR:
			o1 = opextr(ctxt, AEXTR, v, rf, rf, rt)

		case ARORW:
			o1 = opextr(ctxt, AEXTRW, v, rf, rf, rt)

		default:
			ctxt.Diag("bad shift $con\n%v", ctxt.Curp)
			break
		}

	case 9: /* lsl Rm,[Rn],Rd -> lslv Rm, Rn, Rd */
		o1 = oprrr(ctxt, p.As)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 |= (uint32(p.From.Reg&31) << 16) | (uint32(r&31) << 5) | uint32(p.To.Reg&31)

	case 10: /* brk/hvc/.../svc [$con] */
		o1 = opimm(ctxt, p.As)

		if p.To.Type != obj.TYPE_NONE {
			o1 |= uint32((p.To.Offset & 0xffff) << 5)
		}

	case 11: /* dword */
		aclass(ctxt, &p.To)

		o1 = uint32(ctxt.Instoffset)
		o2 = uint32(ctxt.Instoffset >> 32)
		if p.To.Sym != nil {
			rel := obj.Addrel(ctxt.Cursym)
			rel.Off = int32(ctxt.Pc)
			rel.Siz = 8
			rel.Sym = p.To.Sym
			rel.Add = p.To.Offset
			rel.Type = obj.R_ADDR
			o2 = 0
			o1 = o2
		}

	case 12: /* movT $vcon, reg */
		o1 = omovlit(ctxt, p.As, p, &p.From, int(p.To.Reg))

	case 13: /* addop $vcon, [R], R (64 bit literal); cmp $lcon,R -> addop $lcon,R, ZR */
		o1 = omovlit(ctxt, AMOVD, p, &p.From, REGTMP)

		if !(o1 != 0) {
			break
		}
		rt := int(p.To.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = REGZERO
		}
		r := int(p.Reg)
		if r == 0 {
			r = rt
		}
		if p.To.Type != obj.TYPE_NONE && (p.To.Reg == REGSP || r == REGSP) {
			o2 = opxrrr(ctxt, p.As)
			o2 |= REGTMP & 31 << 16
			o2 |= LSL0_64
		} else {
			o2 = oprrr(ctxt, p.As)
			o2 |= REGTMP & 31 << 16 /* shift is 0 */
		}

		o2 |= uint32(r&31) << 5
		o2 |= uint32(rt & 31)

	case 14: /* word */
		if aclass(ctxt, &p.To) == C_ADDR {
			ctxt.Diag("address constant needs DWORD\n%v", p)
		}
		o1 = uint32(ctxt.Instoffset)
		if p.To.Sym != nil {
			// This case happens with words generated
			// in the PC stream as part of the literal pool.
			rel := obj.Addrel(ctxt.Cursym)

			rel.Off = int32(ctxt.Pc)
			rel.Siz = 4
			rel.Sym = p.To.Sym
			rel.Add = p.To.Offset
			rel.Type = obj.R_ADDR
			o1 = 0
		}

	case 15: /* mul/mneg/umulh/umull r,[r,]r; madd/msub Rm,Rn,Ra,Rd */
		o1 = oprrr(ctxt, p.As)

		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		var r int
		var ra int
		if p.From3Type() == obj.TYPE_REG {
			r = int(p.From3.Reg)
			ra = int(p.Reg)
			if ra == 0 {
				ra = REGZERO
			}
		} else {
			r = int(p.Reg)
			if r == 0 {
				r = rt
			}
			ra = REGZERO
		}

		o1 |= (uint32(rf&31) << 16) | (uint32(ra&31) << 10) | (uint32(r&31) << 5) | uint32(rt&31)

	case 16: /* XremY R[,R],R -> XdivY; XmsubY */
		o1 = oprrr(ctxt, p.As)

		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if r == 0 {
			r = rt
		}
		o1 |= (uint32(rf&31) << 16) | (uint32(r&31) << 5) | REGTMP&31
		o2 = oprrr(ctxt, AMSUBW)
		o2 |= o1 & (1 << 31) /* same size */
		o2 |= (uint32(rf&31) << 16) | (uint32(r&31) << 10) | (REGTMP & 31 << 5) | uint32(rt&31)

	case 17: /* op Rm,[Rn],Rd; default Rn=ZR */
		o1 = oprrr(ctxt, p.As)

		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = REGZERO
		}
		if r == 0 {
			r = REGZERO
		}
		o1 |= (uint32(rf&31) << 16) | (uint32(r&31) << 5) | uint32(rt&31)

	case 18: /* csel cond,Rn,Rm,Rd; cinc/cinv/cneg cond,Rn,Rd; cset cond,Rd */
		o1 = oprrr(ctxt, p.As)

		cond := int(p.From.Reg)
		r := int(p.Reg)
		var rf int
		if r != 0 {
			if p.From3Type() == obj.TYPE_NONE {
				/* CINC/CINV/CNEG */
				rf = r

				cond ^= 1
			} else {
				rf = int(p.From3.Reg) /* CSEL */
			}
		} else {
			/* CSET */
			if p.From3Type() != obj.TYPE_NONE {
				ctxt.Diag("invalid combination\n%v", p)
			}
			rf = REGZERO
			r = rf
			cond ^= 1
		}

		rt := int(p.To.Reg)
		o1 |= (uint32(rf&31) << 16) | (uint32(cond&31) << 12) | (uint32(r&31) << 5) | uint32(rt&31)

	case 19: /* CCMN cond, (Rm|uimm5),Rn, uimm4 -> ccmn Rn,Rm,uimm4,cond */
		nzcv := int(p.To.Offset)

		cond := int(p.From.Reg)
		var rf int
		if p.From3.Type == obj.TYPE_REG {
			o1 = oprrr(ctxt, p.As)
			rf = int(p.From3.Reg) /* Rm */
		} else {
			o1 = opirr(ctxt, p.As)
			rf = int(p.From3.Offset & 0x1F)
		}

		o1 |= (uint32(rf&31) << 16) | (uint32(cond) << 12) | (uint32(p.Reg&31) << 5) | uint32(nzcv)

	case 20: /* movT R,O(R) -> strT */
		v := int32(regoff(ctxt, &p.To))

		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		if v < 0 { /* unscaled 9-bit signed */
			o1 = olsr9s(ctxt, int32(opstr9(ctxt, p.As)), v, r, int(p.From.Reg))
		} else {
			v = int32(offsetshift(ctxt, int64(v), int(o.a3)))
			o1 = olsr12u(ctxt, int32(opstr12(ctxt, p.As)), v, r, int(p.From.Reg))
		}

	case 21: /* movT O(R),R -> ldrT */
		v := int32(regoff(ctxt, &p.From))

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		if v < 0 { /* unscaled 9-bit signed */
			o1 = olsr9s(ctxt, int32(opldr9(ctxt, p.As)), v, r, int(p.To.Reg))
		} else {
			v = int32(offsetshift(ctxt, int64(v), int(o.a1)))

			//print("offset=%lld v=%ld a1=%d\n", instoffset, v, o->a1);
			o1 = olsr12u(ctxt, int32(opldr12(ctxt, p.As)), v, r, int(p.To.Reg))
		}

	case 22: /* movT (R)O!,R; movT O(R)!, R -> ldrT */
		v := int32(p.From.Offset)

		if v < -256 || v > 255 {
			ctxt.Diag("offset out of range\n%v", p)
		}
		o1 = opldrpp(ctxt, p.As)
		if o.scond == C_XPOST {
			o1 |= 1 << 10
		} else {
			o1 |= 3 << 10
		}
		o1 |= ((uint32(v) & 0x1FF) << 12) | (uint32(p.From.Reg&31) << 5) | uint32(p.To.Reg&31)

	case 23: /* movT R,(R)O!; movT O(R)!, R -> strT */
		v := int32(p.To.Offset)

		if v < -256 || v > 255 {
			ctxt.Diag("offset out of range\n%v", p)
		}
		o1 = LD2STR(opldrpp(ctxt, p.As))
		if o.scond == C_XPOST {
			o1 |= 1 << 10
		} else {
			o1 |= 3 << 10
		}
		o1 |= ((uint32(v) & 0x1FF) << 12) | (uint32(p.To.Reg&31) << 5) | uint32(p.From.Reg&31)

	case 24: /* mov/mvn Rs,Rd -> add $0,Rs,Rd or orr Rs,ZR,Rd */
		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		s := rf == REGSP || rt == REGSP
		if p.As == AMVN || p.As == AMVNW {
			if s {
				ctxt.Diag("illegal SP reference\n%v", p)
			}
			o1 = oprrr(ctxt, p.As)
			o1 |= (uint32(rf&31) << 16) | (REGZERO & 31 << 5) | uint32(rt&31)
		} else if s {
			o1 = opirr(ctxt, p.As)
			o1 |= (uint32(rf&31) << 5) | uint32(rt&31)
		} else {
			o1 = oprrr(ctxt, p.As)
			o1 |= (uint32(rf&31) << 16) | (REGZERO & 31 << 5) | uint32(rt&31)
		}

	case 25: /* negX Rs, Rd -> subX Rs<<0, ZR, Rd */
		o1 = oprrr(ctxt, p.As)

		rf := int(p.From.Reg)
		if rf == C_NONE {
			rf = int(p.To.Reg)
		}
		rt := int(p.To.Reg)
		o1 |= (uint32(rf&31) << 16) | (REGZERO & 31 << 5) | uint32(rt&31)

	case 26: /* negX Rm<<s, Rd -> subX Rm<<s, ZR, Rd */
		o1 = oprrr(ctxt, p.As)

		o1 |= uint32(p.From.Offset) /* includes reg, op, etc */
		rt := int(p.To.Reg)
		o1 |= (REGZERO & 31 << 5) | uint32(rt&31)

	case 27: /* op Rm<<n[,Rn],Rd (extended register) */
		o1 = opxrrr(ctxt, p.As)

		if (p.From.Reg-obj.RBaseARM64)&REG_EXT != 0 {
			ctxt.Diag("extended register not implemented\n%v", p)
			// o1 |= uint32(p.From.Offset) /* includes reg, op, etc */
		} else {
			o1 |= uint32(p.From.Reg&31) << 16
		}
		rt := int(p.To.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = REGZERO
		}
		r := int(p.Reg)
		if r == 0 {
			r = rt
		}
		o1 |= (uint32(r&31) << 5) | uint32(rt&31)

	case 28: /* logop $vcon, [R], R (64 bit literal) */
		o1 = omovlit(ctxt, AMOVD, p, &p.From, REGTMP)

		if !(o1 != 0) {
			break
		}
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o2 = oprrr(ctxt, p.As)
		o2 |= REGTMP & 31 << 16 /* shift is 0 */
		o2 |= uint32(r&31) << 5
		o2 |= uint32(p.To.Reg & 31)

	case 29: /* op Rn, Rd */
		o1 = oprrr(ctxt, p.As)

		o1 |= uint32(p.From.Reg&31)<<5 | uint32(p.To.Reg&31)

	case 30: /* movT R,L(R) -> strT */
		s := movesize(o.as)

		if s < 0 {
			ctxt.Diag("unexpected long move, op %v tab %v\n%v", obj.Aconv(p.As), obj.Aconv(o.as), p)
		}
		v := int32(regoff(ctxt, &p.To))
		if v < 0 {
			ctxt.Diag("negative large offset\n%v", p)
		}
		if (v & ((1 << uint(s)) - 1)) != 0 {
			ctxt.Diag("misaligned offset\n%v", p)
		}
		hi := v - (v & (0xFFF << uint(s)))
		if (hi & 0xFFF) != 0 {
			ctxt.Diag("internal: miscalculated offset %d [%d]\n%v", v, s, p)
		}

		//fprint(2, "v=%ld (%#lux) s=%d hi=%ld (%#lux) v'=%ld (%#lux)\n", v, v, s, hi, hi, ((v-hi)>>s)&0xFFF, ((v-hi)>>s)&0xFFF);
		r := int(p.To.Reg)

		if r == 0 {
			r = int(o.param)
		}
		o1 = oaddi(ctxt, int32(opirr(ctxt, AADD)), hi, r, REGTMP)
		o2 = olsr12u(ctxt, int32(opstr12(ctxt, p.As)), ((v-hi)>>uint(s))&0xFFF, REGTMP, int(p.From.Reg))

	case 31: /* movT L(R), R -> ldrT */
		s := movesize(o.as)

		if s < 0 {
			ctxt.Diag("unexpected long move, op %v tab %v\n%v", obj.Aconv(p.As), obj.Aconv(o.as), p)
		}
		v := int32(regoff(ctxt, &p.From))
		if v < 0 {
			ctxt.Diag("negative large offset\n%v", p)
		}
		if (v & ((1 << uint(s)) - 1)) != 0 {
			ctxt.Diag("misaligned offset\n%v", p)
		}
		hi := v - (v & (0xFFF << uint(s)))
		if (hi & 0xFFF) != 0 {
			ctxt.Diag("internal: miscalculated offset %d [%d]\n%v", v, s, p)
		}

		//fprint(2, "v=%ld (%#lux) s=%d hi=%ld (%#lux) v'=%ld (%#lux)\n", v, v, s, hi, hi, ((v-hi)>>s)&0xFFF, ((v-hi)>>s)&0xFFF);
		r := int(p.From.Reg)

		if r == 0 {
			r = int(o.param)
		}
		o1 = oaddi(ctxt, int32(opirr(ctxt, AADD)), hi, r, REGTMP)
		o2 = olsr12u(ctxt, int32(opldr12(ctxt, p.As)), ((v-hi)>>uint(s))&0xFFF, REGTMP, int(p.To.Reg))

	case 32: /* mov $con, R -> movz/movn */
		r := 32

		if p.As == AMOVD {
			r = 64
		}
		d := p.From.Offset
		s := movcon(d)
		if s < 0 || s >= r {
			d = ^d
			s = movcon(d)
			if s < 0 || s >= r {
				ctxt.Diag("impossible move wide: %#x\n%v", uint64(p.From.Offset), p)
			}
			if p.As == AMOVD {
				o1 = opirr(ctxt, AMOVN)
			} else {
				o1 = opirr(ctxt, AMOVNW)
			}
		} else {
			if p.As == AMOVD {
				o1 = opirr(ctxt, AMOVZ)
			} else {
				o1 = opirr(ctxt, AMOVZW)
			}
		}

		rt := int(p.To.Reg)
		o1 |= uint32((((d >> uint(s*16)) & 0xFFFF) << 5) | int64((uint32(s)&3)<<21) | int64(rt&31))

	case 33: /* movk $uimm16 << pos */
		o1 = opirr(ctxt, p.As)

		d := p.From.Offset
		if (d >> 16) != 0 {
			ctxt.Diag("requires uimm16\n%v", p)
		}
		s := 0
		if p.From3Type() != obj.TYPE_NONE {
			if p.From3.Type != obj.TYPE_CONST {
				ctxt.Diag("missing bit position\n%v", p)
			}
			s = int(p.From3.Offset / 16)
			if (s*16&0xF) != 0 || s >= 4 || (o1&S64) == 0 && s >= 2 {
				ctxt.Diag("illegal bit position\n%v", p)
			}
		}

		rt := int(p.To.Reg)
		o1 |= uint32(((d & 0xFFFF) << 5) | int64((uint32(s)&3)<<21) | int64(rt&31))

	case 34: /* mov $lacon,R */
		o1 = omovlit(ctxt, AMOVD, p, &p.From, REGTMP)

		if !(o1 != 0) {
			break
		}
		o2 = opxrrr(ctxt, AADD)
		o2 |= REGTMP & 31 << 16
		o2 |= LSL0_64
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 |= uint32(r&31) << 5
		o2 |= uint32(p.To.Reg & 31)

	case 35: /* mov SPR,R -> mrs */
		o1 = oprrr(ctxt, AMRS)

		v := int32(p.From.Offset)
		if (o1 & uint32(v&^(3<<19))) != 0 {
			ctxt.Diag("MRS register value overlap\n%v", p)
		}
		o1 |= uint32(v)
		o1 |= uint32(p.To.Reg & 31)

	case 36: /* mov R,SPR */
		o1 = oprrr(ctxt, AMSR)

		v := int32(p.To.Offset)
		if (o1 & uint32(v&^(3<<19))) != 0 {
			ctxt.Diag("MSR register value overlap\n%v", p)
		}
		o1 |= uint32(v)
		o1 |= uint32(p.From.Reg & 31)

	case 37: /* mov $con,PSTATEfield -> MSR [immediate] */
		if (uint64(p.From.Offset) &^ uint64(0xF)) != 0 {
			ctxt.Diag("illegal immediate for PSTATE field\n%v", p)
		}
		o1 = opirr(ctxt, AMSR)
		o1 |= uint32((p.From.Offset & 0xF) << 8) /* Crm */
		v := int32(0)
		for i := 0; i < len(pstatefield); i++ {
			if int64(pstatefield[i].a) == p.To.Offset {
				v = int32(pstatefield[i].b)
				break
			}
		}

		if v == 0 {
			ctxt.Diag("illegal PSTATE field for immediate move\n%v", p)
		}
		o1 |= uint32(v)

	case 38: /* clrex [$imm] */
		o1 = opimm(ctxt, p.As)

		if p.To.Type == obj.TYPE_NONE {
			o1 |= 0xF << 8
		} else {
			o1 |= uint32((p.To.Offset & 0xF) << 8)
		}

	case 39: /* cbz R, rel */
		o1 = opirr(ctxt, p.As)

		o1 |= uint32(p.From.Reg & 31)
		o1 |= uint32(brdist(ctxt, p, 0, 19, 2) << 5)

	case 40: /* tbz */
		o1 = opirr(ctxt, p.As)

		v := int32(p.From.Offset)
		if v < 0 || v > 63 {
			ctxt.Diag("illegal bit number\n%v", p)
		}
		o1 |= ((uint32(v) & 0x20) << (31 - 5)) | ((uint32(v) & 0x1F) << 19)
		o1 |= uint32(brdist(ctxt, p, 0, 14, 2) << 5)
		o1 |= uint32(p.Reg)

	case 41: /* eret, nop, others with no operands */
		o1 = op0(ctxt, p.As)

	case 42: /* bfm R,r,s,R */
		o1 = opbfm(ctxt, p.As, int(p.From.Offset), int(p.From3.Offset), int(p.Reg), int(p.To.Reg))

	case 43: /* bfm aliases */
		r := int(p.From.Offset)

		s := int(p.From3.Offset)
		rf := int(p.Reg)
		rt := int(p.To.Reg)
		if rf == 0 {
			rf = rt
		}
		switch p.As {
		case ABFI:
			o1 = opbfm(ctxt, ABFM, 64-r, s-1, rf, rt)

		case ABFIW:
			o1 = opbfm(ctxt, ABFMW, 32-r, s-1, rf, rt)

		case ABFXIL:
			o1 = opbfm(ctxt, ABFM, r, r+s-1, rf, rt)

		case ABFXILW:
			o1 = opbfm(ctxt, ABFMW, r, r+s-1, rf, rt)

		case ASBFIZ:
			o1 = opbfm(ctxt, ASBFM, 64-r, s-1, rf, rt)

		case ASBFIZW:
			o1 = opbfm(ctxt, ASBFMW, 32-r, s-1, rf, rt)

		case ASBFX:
			o1 = opbfm(ctxt, ASBFM, r, r+s-1, rf, rt)

		case ASBFXW:
			o1 = opbfm(ctxt, ASBFMW, r, r+s-1, rf, rt)

		case AUBFIZ:
			o1 = opbfm(ctxt, AUBFM, 64-r, s-1, rf, rt)

		case AUBFIZW:
			o1 = opbfm(ctxt, AUBFMW, 32-r, s-1, rf, rt)

		case AUBFX:
			o1 = opbfm(ctxt, AUBFM, r, r+s-1, rf, rt)

		case AUBFXW:
			o1 = opbfm(ctxt, AUBFMW, r, r+s-1, rf, rt)

		default:
			ctxt.Diag("bad bfm alias\n%v", ctxt.Curp)
			break
		}

	case 44: /* extr $b, Rn, Rm, Rd */
		o1 = opextr(ctxt, p.As, int32(p.From.Offset), int(p.From3.Reg), int(p.Reg), int(p.To.Reg))

	case 45: /* sxt/uxt[bhw] R,R; movT R,R -> sxtT R,R */
		rf := int(p.From.Reg)

		rt := int(p.To.Reg)
		as := p.As
		if rf == REGZERO {
			as = AMOVWU /* clearer in disassembly */
		}
		switch as {
		case AMOVB, ASXTB:
			o1 = opbfm(ctxt, ASBFM, 0, 7, rf, rt)

		case AMOVH, ASXTH:
			o1 = opbfm(ctxt, ASBFM, 0, 15, rf, rt)

		case AMOVW, ASXTW:
			o1 = opbfm(ctxt, ASBFM, 0, 31, rf, rt)

		case AMOVBU, AUXTB:
			o1 = opbfm(ctxt, AUBFM, 0, 7, rf, rt)

		case AMOVHU, AUXTH:
			o1 = opbfm(ctxt, AUBFM, 0, 15, rf, rt)

		case AMOVWU:
			o1 = oprrr(ctxt, as) | (uint32(rf&31) << 16) | (REGZERO & 31 << 5) | uint32(rt&31)

		case AUXTW:
			o1 = opbfm(ctxt, AUBFM, 0, 31, rf, rt)

		case ASXTBW:
			o1 = opbfm(ctxt, ASBFMW, 0, 7, rf, rt)

		case ASXTHW:
			o1 = opbfm(ctxt, ASBFMW, 0, 15, rf, rt)

		case AUXTBW:
			o1 = opbfm(ctxt, AUBFMW, 0, 7, rf, rt)

		case AUXTHW:
			o1 = opbfm(ctxt, AUBFMW, 0, 15, rf, rt)

		default:
			ctxt.Diag("bad sxt %v", obj.Aconv(as))
			break
		}

	case 46: /* cls */
		o1 = opbit(ctxt, p.As)

		o1 |= uint32(p.From.Reg&31) << 5
		o1 |= uint32(p.To.Reg & 31)

	case 47: /* movT R,V(R) -> strT (huge offset) */
		o1 = omovlit(ctxt, AMOVW, p, &p.To, REGTMP)

		if !(o1 != 0) {
			break
		}
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = olsxrr(ctxt, p.As, REGTMP, r, int(p.From.Reg))

	case 48: /* movT V(R), R -> ldrT (huge offset) */
		o1 = omovlit(ctxt, AMOVW, p, &p.From, REGTMP)

		if !(o1 != 0) {
			break
		}
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = olsxrr(ctxt, p.As, REGTMP, r, int(p.To.Reg))

	case 50: /* sys/sysl */
		o1 = opirr(ctxt, p.As)

		if (p.From.Offset &^ int64(SYSARG4(0x7, 0xF, 0xF, 0x7))) != 0 {
			ctxt.Diag("illegal SYS argument\n%v", p)
		}
		o1 |= uint32(p.From.Offset)
		if p.To.Type == obj.TYPE_REG {
			o1 |= uint32(p.To.Reg & 31)
		} else if p.Reg != 0 {
			o1 |= uint32(p.Reg & 31)
		} else {
			o1 |= 0x1F
		}

	case 51: /* dmb */
		o1 = opirr(ctxt, p.As)

		if p.From.Type == obj.TYPE_CONST {
			o1 |= uint32((p.From.Offset & 0xF) << 8)
		}

	case 52: /* hint */
		o1 = opirr(ctxt, p.As)

		o1 |= uint32((p.From.Offset & 0x7F) << 5)

	case 53: /* and/or/eor/bic/... $bimmN, Rn, Rd -> op (N,r,s), Rn, Rd */
		ctxt.Diag("bitmask immediate not implemented\n%v", p)

	case 54: /* floating point arith */
		o1 = oprrr(ctxt, p.As)

		var rf int
		if p.From.Type == obj.TYPE_CONST {
			rf = chipfloat7(ctxt, p.From.Val.(float64))
			if rf < 0 || true {
				ctxt.Diag("invalid floating-point immediate\n%v", p)
				rf = 0
			}

			rf |= (1 << 3)
		} else {
			rf = int(p.From.Reg)
		}
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if (o1&(0x1F<<24)) == (0x1E<<24) && (o1&(1<<11)) == 0 { /* monadic */
			r = rf
			rf = 0
		} else if r == 0 {
			r = rt
		}
		o1 |= (uint32(rf&31) << 16) | (uint32(r&31) << 5) | uint32(rt&31)

	case 56: /* floating point compare */
		o1 = oprrr(ctxt, p.As)

		var rf int
		if p.From.Type == obj.TYPE_CONST {
			o1 |= 8 /* zero */
			rf = 0
		} else {
			rf = int(p.From.Reg)
		}
		rt := int(p.Reg)
		o1 |= uint32(rf&31)<<16 | uint32(rt&31)<<5

	case 57: /* floating point conditional compare */
		o1 = oprrr(ctxt, p.As)

		cond := int(p.From.Reg)
		nzcv := int(p.To.Offset)
		if nzcv&^0xF != 0 {
			ctxt.Diag("implausible condition\n%v", p)
		}
		rf := int(p.Reg)
		if p.From3 == nil || p.From3.Reg < REG_F0 || p.From3.Reg > REG_F31 {
			ctxt.Diag("illegal FCCMP\n%v", p)
			break
		}
		rt := int(p.From3.Reg)
		o1 |= uint32(rf&31)<<16 | uint32(cond)<<12 | uint32(rt&31)<<5 | uint32(nzcv)

	case 58: /* ldar/ldxr/ldaxr */
		o1 = opload(ctxt, p.As)

		o1 |= 0x1F << 16
		o1 |= uint32(p.From.Reg) << 5
		if p.Reg != 0 {
			o1 |= uint32(p.Reg) << 10
		} else {
			o1 |= 0x1F << 10
		}
		o1 |= uint32(p.To.Reg & 31)

	case 59: /* stxr/stlxr */
		o1 = opstore(ctxt, p.As)

		if p.RegTo2 != obj.REG_NONE {
			o1 |= uint32(p.RegTo2&31) << 16
		} else {
			o1 |= 0x1F << 16
		}

		// TODO(aram): add support for STXP
		o1 |= uint32(p.To.Reg&31) << 5

		o1 |= uint32(p.From.Reg & 31)

	case 60: /* adrp label,r */
		d := brdist(ctxt, p, 12, 21, 0)

		o1 = ADR(1, uint32(d), uint32(p.To.Reg))

	case 61: /* adr label, r */
		d := brdist(ctxt, p, 0, 21, 0)

		o1 = ADR(0, uint32(d), uint32(p.To.Reg))

		/* reloc ops */
	case 64: /* movT R,addr -> adrp + add + movT R, (REGTMP) */
		o1 = ADR(1, 0, REGTMP)
		o2 = opirr(ctxt, AADD) | REGTMP&31<<5 | REGTMP&31
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 8
		rel.Sym = p.To.Sym
		rel.Add = p.To.Offset
		rel.Type = obj.R_ADDRARM64
		o3 = olsr12u(ctxt, int32(opstr12(ctxt, p.As)), 0, REGTMP, int(p.From.Reg))

	case 65: /* movT addr,R -> adrp + add + movT (REGTMP), R */
		o1 = ADR(1, 0, REGTMP)
		o2 = opirr(ctxt, AADD) | REGTMP&31<<5 | REGTMP&31
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 8
		rel.Sym = p.From.Sym
		rel.Add = p.From.Offset
		rel.Type = obj.R_ADDRARM64
		o3 = olsr12u(ctxt, int32(opldr12(ctxt, p.As)), 0, REGTMP, int(p.To.Reg))

	case 66: /* ldp O(R)!, (r1, r2); ldp (R)O!, (r1, r2) */
		v := int32(p.From.Offset)

		if v < -512 || v > 504 {
			ctxt.Diag("offset out of range\n%v", p)
		}
		if o.scond == C_XPOST {
			o1 |= 1 << 23
		} else {
			o1 |= 3 << 23
		}
		o1 |= 1 << 22
		o1 |= uint32(int64(2<<30|5<<27|((uint32(v)/8)&0x7f)<<15) | p.To.Offset<<10 | int64(uint32(p.From.Reg&31)<<5) | int64(p.To.Reg&31))

	case 67: /* stp (r1, r2), O(R)!; stp (r1, r2), (R)O! */
		v := int32(p.To.Offset)

		if v < -512 || v > 504 {
			ctxt.Diag("offset out of range\n%v", p)
		}
		if o.scond == C_XPOST {
			o1 |= 1 << 23
		} else {
			o1 |= 3 << 23
		}
		o1 |= uint32(int64(2<<30|5<<27|((uint32(v)/8)&0x7f)<<15) | p.From.Offset<<10 | int64(uint32(p.To.Reg&31)<<5) | int64(p.From.Reg&31))

	case 68: /* movT $vconaddr(SB), reg -> adrp + add + reloc */
		if p.As == AMOVW {
			ctxt.Diag("invalid load of 32-bit address: %v", p)
		}
		o1 = ADR(1, 0, uint32(p.To.Reg))
		o2 = opirr(ctxt, AADD) | uint32(p.To.Reg&31)<<5 | uint32(p.To.Reg&31)
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 8
		rel.Sym = p.From.Sym
		rel.Add = p.From.Offset
		rel.Type = obj.R_ADDRARM64

	case 69: /* LE model movd $tlsvar, reg -> movz reg, 0 + reloc */
		o1 = opirr(ctxt, AMOVZ)
		o1 |= uint32(p.To.Reg & 31)
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.From.Sym
		rel.Type = obj.R_ARM64_TLS_LE
		if p.From.Offset != 0 {
			ctxt.Diag("invalid offset on MOVW $tlsvar")
		}

	case 70: /* IE model movd $tlsvar, reg -> adrp REGTMP, 0; ldr reg, [REGTMP, #0] + relocs */
		o1 = ADR(1, 0, REGTMP)
		o2 = olsr12u(ctxt, int32(opldr12(ctxt, AMOVD)), 0, REGTMP, int(p.To.Reg))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 8
		rel.Sym = p.From.Sym
		rel.Add = 0
		rel.Type = obj.R_ARM64_TLS_IE
		if p.From.Offset != 0 {
			ctxt.Diag("invalid offset on MOVW $tlsvar")
		}

	case 71: /* movd sym@GOT, reg -> adrp REGTMP, #0; ldr reg, [REGTMP, #0] + relocs */
		o1 = ADR(1, 0, REGTMP)
		o2 = olsr12u(ctxt, int32(opldr12(ctxt, AMOVD)), 0, REGTMP, int(p.To.Reg))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 8
		rel.Sym = p.From.Sym
		rel.Add = 0
		rel.Type = obj.R_ARM64_GOTPCREL

	// This is supposed to be something that stops execution.
	// It's not supposed to be reached, ever, but if it is, we'd
	// like to be able to tell how we got there. Assemble as
	// 0xbea71700 which is guaranteed to raise undefined instruction
	// exception.
	case 90:
		o1 = 0xbea71700

		break
	}

	out[0] = o1
	out[1] = o2
	out[2] = o3
	out[3] = o4
	out[4] = o5
	return
}

/*
 * basic Rm op Rn -> Rd (using shifted register with 0)
 * also op Rn -> Rt
 * also Rm*Rn op Ra -> Rd
 */
func oprrr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AADC:
		return S64 | 0<<30 | 0<<29 | 0xd0<<21 | 0<<10

	case AADCW:
		return S32 | 0<<30 | 0<<29 | 0xd0<<21 | 0<<10

	case AADCS:
		return S64 | 0<<30 | 1<<29 | 0xd0<<21 | 0<<10

	case AADCSW:
		return S32 | 0<<30 | 1<<29 | 0xd0<<21 | 0<<10

	case ANGC, ASBC:
		return S64 | 1<<30 | 0<<29 | 0xd0<<21 | 0<<10

	case ANGCS, ASBCS:
		return S64 | 1<<30 | 1<<29 | 0xd0<<21 | 0<<10

	case ANGCW, ASBCW:
		return S32 | 1<<30 | 0<<29 | 0xd0<<21 | 0<<10

	case ANGCSW, ASBCSW:
		return S32 | 1<<30 | 1<<29 | 0xd0<<21 | 0<<10

	case AADD:
		return S64 | 0<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case AADDW:
		return S32 | 0<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case ACMN, AADDS:
		return S64 | 0<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case ACMNW, AADDSW:
		return S32 | 0<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case ASUB:
		return S64 | 1<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case ASUBW:
		return S32 | 1<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case ACMP, ASUBS:
		return S64 | 1<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case ACMPW, ASUBSW:
		return S32 | 1<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 0<<21 | 0<<10

	case AAND:
		return S64 | 0<<29 | 0xA<<24

	case AANDW:
		return S32 | 0<<29 | 0xA<<24

	case AMOVD, AORR:
		return S64 | 1<<29 | 0xA<<24

		//	case AMOVW:
	case AMOVWU, AORRW:
		return S32 | 1<<29 | 0xA<<24

	case AEOR:
		return S64 | 2<<29 | 0xA<<24

	case AEORW:
		return S32 | 2<<29 | 0xA<<24

	case AANDS:
		return S64 | 3<<29 | 0xA<<24

	case AANDSW:
		return S32 | 3<<29 | 0xA<<24

	case ABIC:
		return S64 | 0<<29 | 0xA<<24 | 1<<21

	case ABICW:
		return S32 | 0<<29 | 0xA<<24 | 1<<21

	case ABICS:
		return S64 | 3<<29 | 0xA<<24 | 1<<21

	case ABICSW:
		return S32 | 3<<29 | 0xA<<24 | 1<<21

	case AEON:
		return S64 | 2<<29 | 0xA<<24 | 1<<21

	case AEONW:
		return S32 | 2<<29 | 0xA<<24 | 1<<21

	case AMVN, AORN:
		return S64 | 1<<29 | 0xA<<24 | 1<<21

	case AMVNW, AORNW:
		return S32 | 1<<29 | 0xA<<24 | 1<<21

	case AASR:
		return S64 | OPDP2(10) /* also ASRV */

	case AASRW:
		return S32 | OPDP2(10)

	case ALSL:
		return S64 | OPDP2(8)

	case ALSLW:
		return S32 | OPDP2(8)

	case ALSR:
		return S64 | OPDP2(9)

	case ALSRW:
		return S32 | OPDP2(9)

	case AROR:
		return S64 | OPDP2(11)

	case ARORW:
		return S32 | OPDP2(11)

	case ACCMN:
		return S64 | 0<<30 | 1<<29 | 0xD2<<21 | 0<<11 | 0<<10 | 0<<4 /* cond<<12 | nzcv<<0 */

	case ACCMNW:
		return S32 | 0<<30 | 1<<29 | 0xD2<<21 | 0<<11 | 0<<10 | 0<<4

	case ACCMP:
		return S64 | 1<<30 | 1<<29 | 0xD2<<21 | 0<<11 | 0<<10 | 0<<4 /* imm5<<16 | cond<<12 | nzcv<<0 */

	case ACCMPW:
		return S32 | 1<<30 | 1<<29 | 0xD2<<21 | 0<<11 | 0<<10 | 0<<4

	case ACRC32B:
		return S32 | OPDP2(16)

	case ACRC32H:
		return S32 | OPDP2(17)

	case ACRC32W:
		return S32 | OPDP2(18)

	case ACRC32X:
		return S64 | OPDP2(19)

	case ACRC32CB:
		return S32 | OPDP2(20)

	case ACRC32CH:
		return S32 | OPDP2(21)

	case ACRC32CW:
		return S32 | OPDP2(22)

	case ACRC32CX:
		return S64 | OPDP2(23)

	case ACSEL:
		return S64 | 0<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 0<<10

	case ACSELW:
		return S32 | 0<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 0<<10

	case ACSET:
		return S64 | 0<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 1<<10

	case ACSETW:
		return S32 | 0<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 1<<10

	case ACSETM:
		return S64 | 1<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 0<<10

	case ACSETMW:
		return S32 | 1<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 0<<10

	case ACINC, ACSINC:
		return S64 | 0<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 1<<10

	case ACINCW, ACSINCW:
		return S32 | 0<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 1<<10

	case ACINV, ACSINV:
		return S64 | 1<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 0<<10

	case ACINVW, ACSINVW:
		return S32 | 1<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 0<<10

	case ACNEG, ACSNEG:
		return S64 | 1<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 1<<10

	case ACNEGW, ACSNEGW:
		return S32 | 1<<30 | 0<<29 | 0xD4<<21 | 0<<11 | 1<<10

	case AMUL, AMADD:
		return S64 | 0<<29 | 0x1B<<24 | 0<<21 | 0<<15

	case AMULW, AMADDW:
		return S32 | 0<<29 | 0x1B<<24 | 0<<21 | 0<<15

	case AMNEG, AMSUB:
		return S64 | 0<<29 | 0x1B<<24 | 0<<21 | 1<<15

	case AMNEGW, AMSUBW:
		return S32 | 0<<29 | 0x1B<<24 | 0<<21 | 1<<15

	case AMRS:
		return SYSOP(1, 2, 0, 0, 0, 0, 0)

	case AMSR:
		return SYSOP(0, 2, 0, 0, 0, 0, 0)

	case ANEG:
		return S64 | 1<<30 | 0<<29 | 0xB<<24 | 0<<21

	case ANEGW:
		return S32 | 1<<30 | 0<<29 | 0xB<<24 | 0<<21

	case ANEGS:
		return S64 | 1<<30 | 1<<29 | 0xB<<24 | 0<<21

	case ANEGSW:
		return S32 | 1<<30 | 1<<29 | 0xB<<24 | 0<<21

	case AREM, ASDIV:
		return S64 | OPDP2(3)

	case AREMW, ASDIVW:
		return S32 | OPDP2(3)

	case ASMULL, ASMADDL:
		return OPDP3(1, 0, 1, 0)

	case ASMNEGL, ASMSUBL:
		return OPDP3(1, 0, 1, 1)

	case ASMULH:
		return OPDP3(1, 0, 2, 0)

	case AUMULL, AUMADDL:
		return OPDP3(1, 0, 5, 0)

	case AUMNEGL, AUMSUBL:
		return OPDP3(1, 0, 5, 1)

	case AUMULH:
		return OPDP3(1, 0, 6, 0)

	case AUREM, AUDIV:
		return S64 | OPDP2(2)

	case AUREMW, AUDIVW:
		return S32 | OPDP2(2)

	case AAESE:
		return 0x4E<<24 | 2<<20 | 8<<16 | 4<<12 | 2<<10

	case AAESD:
		return 0x4E<<24 | 2<<20 | 8<<16 | 5<<12 | 2<<10

	case AAESMC:
		return 0x4E<<24 | 2<<20 | 8<<16 | 6<<12 | 2<<10

	case AAESIMC:
		return 0x4E<<24 | 2<<20 | 8<<16 | 7<<12 | 2<<10

	case ASHA1C:
		return 0x5E<<24 | 0<<12

	case ASHA1P:
		return 0x5E<<24 | 1<<12

	case ASHA1M:
		return 0x5E<<24 | 2<<12

	case ASHA1SU0:
		return 0x5E<<24 | 3<<12

	case ASHA256H:
		return 0x5E<<24 | 4<<12

	case ASHA256H2:
		return 0x5E<<24 | 5<<12

	case ASHA256SU1:
		return 0x5E<<24 | 6<<12

	case ASHA1H:
		return 0x5E<<24 | 2<<20 | 8<<16 | 0<<12 | 2<<10

	case ASHA1SU1:
		return 0x5E<<24 | 2<<20 | 8<<16 | 1<<12 | 2<<10

	case ASHA256SU0:
		return 0x5E<<24 | 2<<20 | 8<<16 | 2<<12 | 2<<10

	case AFCVTZSD:
		return FPCVTI(1, 0, 1, 3, 0)

	case AFCVTZSDW:
		return FPCVTI(0, 0, 1, 3, 0)

	case AFCVTZSS:
		return FPCVTI(1, 0, 0, 3, 0)

	case AFCVTZSSW:
		return FPCVTI(0, 0, 0, 3, 0)

	case AFCVTZUD:
		return FPCVTI(1, 0, 1, 3, 1)

	case AFCVTZUDW:
		return FPCVTI(0, 0, 1, 3, 1)

	case AFCVTZUS:
		return FPCVTI(1, 0, 0, 3, 1)

	case AFCVTZUSW:
		return FPCVTI(0, 0, 0, 3, 1)

	case ASCVTFD:
		return FPCVTI(1, 0, 1, 0, 2)

	case ASCVTFS:
		return FPCVTI(1, 0, 0, 0, 2)

	case ASCVTFWD:
		return FPCVTI(0, 0, 1, 0, 2)

	case ASCVTFWS:
		return FPCVTI(0, 0, 0, 0, 2)

	case AUCVTFD:
		return FPCVTI(1, 0, 1, 0, 3)

	case AUCVTFS:
		return FPCVTI(1, 0, 0, 0, 3)

	case AUCVTFWD:
		return FPCVTI(0, 0, 1, 0, 3)

	case AUCVTFWS:
		return FPCVTI(0, 0, 0, 0, 3)

	case AFADDS:
		return FPOP2S(0, 0, 0, 2)

	case AFADDD:
		return FPOP2S(0, 0, 1, 2)

	case AFSUBS:
		return FPOP2S(0, 0, 0, 3)

	case AFSUBD:
		return FPOP2S(0, 0, 1, 3)

	case AFMULS:
		return FPOP2S(0, 0, 0, 0)

	case AFMULD:
		return FPOP2S(0, 0, 1, 0)

	case AFDIVS:
		return FPOP2S(0, 0, 0, 1)

	case AFDIVD:
		return FPOP2S(0, 0, 1, 1)

	case AFMAXS:
		return FPOP2S(0, 0, 0, 4)

	case AFMINS:
		return FPOP2S(0, 0, 0, 5)

	case AFMAXD:
		return FPOP2S(0, 0, 1, 4)

	case AFMIND:
		return FPOP2S(0, 0, 1, 5)

	case AFMAXNMS:
		return FPOP2S(0, 0, 0, 6)

	case AFMAXNMD:
		return FPOP2S(0, 0, 1, 6)

	case AFMINNMS:
		return FPOP2S(0, 0, 0, 7)

	case AFMINNMD:
		return FPOP2S(0, 0, 1, 7)

	case AFNMULS:
		return FPOP2S(0, 0, 0, 8)

	case AFNMULD:
		return FPOP2S(0, 0, 1, 8)

	case AFCMPS:
		return FPCMP(0, 0, 0, 0, 0)

	case AFCMPD:
		return FPCMP(0, 0, 1, 0, 0)

	case AFCMPES:
		return FPCMP(0, 0, 0, 0, 16)

	case AFCMPED:
		return FPCMP(0, 0, 1, 0, 16)

	case AFCCMPS:
		return FPCCMP(0, 0, 0, 0)

	case AFCCMPD:
		return FPCCMP(0, 0, 1, 0)

	case AFCCMPES:
		return FPCCMP(0, 0, 0, 1)

	case AFCCMPED:
		return FPCCMP(0, 0, 1, 1)

	case AFCSELS:
		return 0x1E<<24 | 0<<22 | 1<<21 | 3<<10

	case AFCSELD:
		return 0x1E<<24 | 1<<22 | 1<<21 | 3<<10

	case AFMOVS:
		return FPOP1S(0, 0, 0, 0)

	case AFABSS:
		return FPOP1S(0, 0, 0, 1)

	case AFNEGS:
		return FPOP1S(0, 0, 0, 2)

	case AFSQRTS:
		return FPOP1S(0, 0, 0, 3)

	case AFCVTSD:
		return FPOP1S(0, 0, 0, 5)

	case AFCVTSH:
		return FPOP1S(0, 0, 0, 7)

	case AFRINTNS:
		return FPOP1S(0, 0, 0, 8)

	case AFRINTPS:
		return FPOP1S(0, 0, 0, 9)

	case AFRINTMS:
		return FPOP1S(0, 0, 0, 10)

	case AFRINTZS:
		return FPOP1S(0, 0, 0, 11)

	case AFRINTAS:
		return FPOP1S(0, 0, 0, 12)

	case AFRINTXS:
		return FPOP1S(0, 0, 0, 14)

	case AFRINTIS:
		return FPOP1S(0, 0, 0, 15)

	case AFMOVD:
		return FPOP1S(0, 0, 1, 0)

	case AFABSD:
		return FPOP1S(0, 0, 1, 1)

	case AFNEGD:
		return FPOP1S(0, 0, 1, 2)

	case AFSQRTD:
		return FPOP1S(0, 0, 1, 3)

	case AFCVTDS:
		return FPOP1S(0, 0, 1, 4)

	case AFCVTDH:
		return FPOP1S(0, 0, 1, 7)

	case AFRINTND:
		return FPOP1S(0, 0, 1, 8)

	case AFRINTPD:
		return FPOP1S(0, 0, 1, 9)

	case AFRINTMD:
		return FPOP1S(0, 0, 1, 10)

	case AFRINTZD:
		return FPOP1S(0, 0, 1, 11)

	case AFRINTAD:
		return FPOP1S(0, 0, 1, 12)

	case AFRINTXD:
		return FPOP1S(0, 0, 1, 14)

	case AFRINTID:
		return FPOP1S(0, 0, 1, 15)

	case AFCVTHS:
		return FPOP1S(0, 0, 3, 4)

	case AFCVTHD:
		return FPOP1S(0, 0, 3, 5)
	}

	ctxt.Diag("bad rrr %d %v", a, obj.Aconv(a))
	prasm(ctxt.Curp)
	return 0
}

/*
 * imm -> Rd
 * imm op Rn -> Rd
 */
func opirr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	/* op $addcon, Rn, Rd */
	case AMOVD, AADD:
		return S64 | 0<<30 | 0<<29 | 0x11<<24

	case ACMN, AADDS:
		return S64 | 0<<30 | 1<<29 | 0x11<<24

	case AMOVW, AADDW:
		return S32 | 0<<30 | 0<<29 | 0x11<<24

	case ACMNW, AADDSW:
		return S32 | 0<<30 | 1<<29 | 0x11<<24

	case ASUB:
		return S64 | 1<<30 | 0<<29 | 0x11<<24

	case ACMP, ASUBS:
		return S64 | 1<<30 | 1<<29 | 0x11<<24

	case ASUBW:
		return S32 | 1<<30 | 0<<29 | 0x11<<24

	case ACMPW, ASUBSW:
		return S32 | 1<<30 | 1<<29 | 0x11<<24

		/* op $imm(SB), Rd; op label, Rd */
	case AADR:
		return 0<<31 | 0x10<<24

	case AADRP:
		return 1<<31 | 0x10<<24

		/* op $bimm, Rn, Rd */
	case AAND:
		return S64 | 0<<29 | 0x24<<23

	case AANDW:
		return S32 | 0<<29 | 0x24<<23 | 0<<22

	case AORR:
		return S64 | 1<<29 | 0x24<<23

	case AORRW:
		return S32 | 1<<29 | 0x24<<23 | 0<<22

	case AEOR:
		return S64 | 2<<29 | 0x24<<23

	case AEORW:
		return S32 | 2<<29 | 0x24<<23 | 0<<22

	case AANDS:
		return S64 | 3<<29 | 0x24<<23

	case AANDSW:
		return S32 | 3<<29 | 0x24<<23 | 0<<22

	case AASR:
		return S64 | 0<<29 | 0x26<<23 /* alias of SBFM */

	case AASRW:
		return S32 | 0<<29 | 0x26<<23 | 0<<22

		/* op $width, $lsb, Rn, Rd */
	case ABFI:
		return S64 | 2<<29 | 0x26<<23 | 1<<22
		/* alias of BFM */

	case ABFIW:
		return S32 | 2<<29 | 0x26<<23 | 0<<22

		/* op $imms, $immr, Rn, Rd */
	case ABFM:
		return S64 | 1<<29 | 0x26<<23 | 1<<22

	case ABFMW:
		return S32 | 1<<29 | 0x26<<23 | 0<<22

	case ASBFM:
		return S64 | 0<<29 | 0x26<<23 | 1<<22

	case ASBFMW:
		return S32 | 0<<29 | 0x26<<23 | 0<<22

	case AUBFM:
		return S64 | 2<<29 | 0x26<<23 | 1<<22

	case AUBFMW:
		return S32 | 2<<29 | 0x26<<23 | 0<<22

	case ABFXIL:
		return S64 | 1<<29 | 0x26<<23 | 1<<22 /* alias of BFM */

	case ABFXILW:
		return S32 | 1<<29 | 0x26<<23 | 0<<22

	case AEXTR:
		return S64 | 0<<29 | 0x27<<23 | 1<<22 | 0<<21

	case AEXTRW:
		return S32 | 0<<29 | 0x27<<23 | 0<<22 | 0<<21

	case ACBNZ:
		return S64 | 0x1A<<25 | 1<<24

	case ACBNZW:
		return S32 | 0x1A<<25 | 1<<24

	case ACBZ:
		return S64 | 0x1A<<25 | 0<<24

	case ACBZW:
		return S32 | 0x1A<<25 | 0<<24

	case ACCMN:
		return S64 | 0<<30 | 1<<29 | 0xD2<<21 | 1<<11 | 0<<10 | 0<<4 /* imm5<<16 | cond<<12 | nzcv<<0 */

	case ACCMNW:
		return S32 | 0<<30 | 1<<29 | 0xD2<<21 | 1<<11 | 0<<10 | 0<<4

	case ACCMP:
		return S64 | 1<<30 | 1<<29 | 0xD2<<21 | 1<<11 | 0<<10 | 0<<4 /* imm5<<16 | cond<<12 | nzcv<<0 */

	case ACCMPW:
		return S32 | 1<<30 | 1<<29 | 0xD2<<21 | 1<<11 | 0<<10 | 0<<4

	case AMOVK:
		return S64 | 3<<29 | 0x25<<23

	case AMOVKW:
		return S32 | 3<<29 | 0x25<<23

	case AMOVN:
		return S64 | 0<<29 | 0x25<<23

	case AMOVNW:
		return S32 | 0<<29 | 0x25<<23

	case AMOVZ:
		return S64 | 2<<29 | 0x25<<23

	case AMOVZW:
		return S32 | 2<<29 | 0x25<<23

	case AMSR:
		return SYSOP(0, 0, 0, 4, 0, 0, 0x1F) /* MSR (immediate) */

	case AAT,
		ADC,
		AIC,
		ATLBI,
		ASYS:
		return SYSOP(0, 1, 0, 0, 0, 0, 0)

	case ASYSL:
		return SYSOP(1, 1, 0, 0, 0, 0, 0)

	case ATBZ:
		return 0x36 << 24

	case ATBNZ:
		return 0x37 << 24

	case ADSB:
		return SYSOP(0, 0, 3, 3, 0, 4, 0x1F)

	case ADMB:
		return SYSOP(0, 0, 3, 3, 0, 5, 0x1F)

	case AISB:
		return SYSOP(0, 0, 3, 3, 0, 6, 0x1F)

	case AHINT:
		return SYSOP(0, 0, 3, 2, 0, 0, 0x1F)
	}

	ctxt.Diag("bad irr %v", obj.Aconv(a))
	prasm(ctxt.Curp)
	return 0
}

func opbit(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case ACLS:
		return S64 | OPBIT(5)

	case ACLSW:
		return S32 | OPBIT(5)

	case ACLZ:
		return S64 | OPBIT(4)

	case ACLZW:
		return S32 | OPBIT(4)

	case ARBIT:
		return S64 | OPBIT(0)

	case ARBITW:
		return S32 | OPBIT(0)

	case AREV:
		return S64 | OPBIT(3)

	case AREVW:
		return S32 | OPBIT(2)

	case AREV16:
		return S64 | OPBIT(1)

	case AREV16W:
		return S32 | OPBIT(1)

	case AREV32:
		return S64 | OPBIT(2)

	default:
		ctxt.Diag("bad bit op\n%v", ctxt.Curp)
		return 0
	}
}

/*
 * add/subtract extended register
 */
func opxrrr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AADD:
		return S64 | 0<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_64

	case AADDW:
		return S32 | 0<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_32

	case ACMN, AADDS:
		return S64 | 0<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_64

	case ACMNW, AADDSW:
		return S32 | 0<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_32

	case ASUB:
		return S64 | 1<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_64

	case ASUBW:
		return S32 | 1<<30 | 0<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_32

	case ACMP, ASUBS:
		return S64 | 1<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_64

	case ACMPW, ASUBSW:
		return S32 | 1<<30 | 1<<29 | 0x0b<<24 | 0<<22 | 1<<21 | LSL0_32
	}

	ctxt.Diag("bad opxrrr %v\n%v", obj.Aconv(a), ctxt.Curp)
	return 0
}

func opimm(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case ASVC:
		return 0xD4<<24 | 0<<21 | 1 /* imm16<<5 */

	case AHVC:
		return 0xD4<<24 | 0<<21 | 2

	case ASMC:
		return 0xD4<<24 | 0<<21 | 3

	case ABRK:
		return 0xD4<<24 | 1<<21 | 0

	case AHLT:
		return 0xD4<<24 | 2<<21 | 0

	case ADCPS1:
		return 0xD4<<24 | 5<<21 | 1

	case ADCPS2:
		return 0xD4<<24 | 5<<21 | 2

	case ADCPS3:
		return 0xD4<<24 | 5<<21 | 3

	case ACLREX:
		return SYSOP(0, 0, 3, 3, 0, 2, 0x1F)
	}

	ctxt.Diag("bad imm %v", obj.Aconv(a))
	prasm(ctxt.Curp)
	return 0
}

func brdist(ctxt *obj.Link, p *obj.Prog, preshift int, flen int, shift int) int64 {
	v := int64(0)
	t := int64(0)
	if p.Pcond != nil {
		v = (p.Pcond.Pc >> uint(preshift)) - (ctxt.Pc >> uint(preshift))
		if (v & ((1 << uint(shift)) - 1)) != 0 {
			ctxt.Diag("misaligned label\n%v", p)
		}
		v >>= uint(shift)
		t = int64(1) << uint(flen-1)
		if v < -t || v >= t {
			ctxt.Diag("branch too far %#x vs %#x [%p]\n%v\n%v", v, t, ctxt.Blitrl, p, p.Pcond)
			panic("branch too far")
		}
	}

	return v & ((t << 1) - 1)
}

/*
 * pc-relative branches
 */
func opbra(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case ABEQ:
		return OPBcc(0x0)

	case ABNE:
		return OPBcc(0x1)

	case ABCS:
		return OPBcc(0x2)

	case ABHS:
		return OPBcc(0x2)

	case ABCC:
		return OPBcc(0x3)

	case ABLO:
		return OPBcc(0x3)

	case ABMI:
		return OPBcc(0x4)

	case ABPL:
		return OPBcc(0x5)

	case ABVS:
		return OPBcc(0x6)

	case ABVC:
		return OPBcc(0x7)

	case ABHI:
		return OPBcc(0x8)

	case ABLS:
		return OPBcc(0x9)

	case ABGE:
		return OPBcc(0xa)

	case ABLT:
		return OPBcc(0xb)

	case ABGT:
		return OPBcc(0xc)

	case ABLE:
		return OPBcc(0xd) /* imm19<<5 | cond */

	case AB:
		return 0<<31 | 5<<26 /* imm26 */

	case obj.ADUFFZERO,
		ABL:
		return 1<<31 | 5<<26
	}

	ctxt.Diag("bad bra %v", obj.Aconv(a))
	prasm(ctxt.Curp)
	return 0
}

func opbrr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case ABL:
		return OPBLR(1) /* BLR */

	case AB:
		return OPBLR(0) /* BR */

	case obj.ARET:
		return OPBLR(2) /* RET */
	}

	ctxt.Diag("bad brr %v", obj.Aconv(a))
	prasm(ctxt.Curp)
	return 0
}

func op0(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case ADRPS:
		return 0x6B<<25 | 5<<21 | 0x1F<<16 | 0x1F<<5

	case AERET:
		return 0x6B<<25 | 4<<21 | 0x1F<<16 | 0<<10 | 0x1F<<5

	// case ANOP:
	// 	return SYSHINT(0)

	case AYIELD:
		return SYSHINT(1)

	case AWFE:
		return SYSHINT(2)

	case AWFI:
		return SYSHINT(3)

	case ASEV:
		return SYSHINT(4)

	case ASEVL:
		return SYSHINT(5)
	}

	ctxt.Diag("bad op0 %v", obj.Aconv(a))
	prasm(ctxt.Curp)
	return 0
}

/*
 * register offset
 */
func opload(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case ALDAR:
		return LDSTX(3, 1, 1, 0, 1) | 0x1F<<10

	case ALDARW:
		return LDSTX(2, 1, 1, 0, 1) | 0x1F<<10

	case ALDARB:
		return LDSTX(0, 1, 1, 0, 1) | 0x1F<<10

	case ALDARH:
		return LDSTX(1, 1, 1, 0, 1) | 0x1F<<10

	case ALDAXP:
		return LDSTX(3, 0, 1, 1, 1)

	case ALDAXPW:
		return LDSTX(2, 0, 1, 1, 1)

	case ALDAXR:
		return LDSTX(3, 0, 1, 0, 1) | 0x1F<<10

	case ALDAXRW:
		return LDSTX(2, 0, 1, 0, 1) | 0x1F<<10

	case ALDAXRB:
		return LDSTX(0, 0, 1, 0, 1) | 0x1F<<10

	case ALDAXRH:
		return LDSTX(1, 0, 1, 0, 1) | 0x1F<<10

	case ALDXR:
		return LDSTX(3, 0, 1, 0, 0) | 0x1F<<10

	case ALDXRB:
		return LDSTX(0, 0, 1, 0, 0) | 0x1F<<10

	case ALDXRH:
		return LDSTX(1, 0, 1, 0, 0) | 0x1F<<10

	case ALDXRW:
		return LDSTX(2, 0, 1, 0, 0) | 0x1F<<10

	case ALDXP:
		return LDSTX(3, 0, 1, 1, 0)

	case ALDXPW:
		return LDSTX(2, 0, 1, 1, 0)

	case AMOVNP:
		return S64 | 0<<30 | 5<<27 | 0<<26 | 0<<23 | 1<<22

	case AMOVNPW:
		return S32 | 0<<30 | 5<<27 | 0<<26 | 0<<23 | 1<<22
	}

	ctxt.Diag("bad opload %v\n%v", obj.Aconv(a), ctxt.Curp)
	return 0
}

func opstore(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case ASTLR:
		return LDSTX(3, 1, 0, 0, 1) | 0x1F<<10

	case ASTLRB:
		return LDSTX(0, 1, 0, 0, 1) | 0x1F<<10

	case ASTLRH:
		return LDSTX(1, 1, 0, 0, 1) | 0x1F<<10

	case ASTLP:
		return LDSTX(3, 0, 0, 1, 1)

	case ASTLPW:
		return LDSTX(2, 0, 0, 1, 1)

	case ASTLRW:
		return LDSTX(2, 1, 0, 0, 1) | 0x1F<<10

	case ASTLXP:
		return LDSTX(2, 0, 0, 1, 1)

	case ASTLXPW:
		return LDSTX(3, 0, 0, 1, 1)

	case ASTLXR:
		return LDSTX(3, 0, 0, 0, 1) | 0x1F<<10

	case ASTLXRB:
		return LDSTX(0, 0, 0, 0, 1) | 0x1F<<10

	case ASTLXRH:
		return LDSTX(1, 0, 0, 0, 1) | 0x1F<<10

	case ASTLXRW:
		return LDSTX(2, 0, 0, 0, 1) | 0x1F<<10

	case ASTXR:
		return LDSTX(3, 0, 0, 0, 0) | 0x1F<<10

	case ASTXRB:
		return LDSTX(0, 0, 0, 0, 0) | 0x1F<<10

	case ASTXRH:
		return LDSTX(1, 0, 0, 0, 0) | 0x1F<<10

	case ASTXP:
		return LDSTX(3, 0, 0, 1, 0)

	case ASTXPW:
		return LDSTX(2, 0, 0, 1, 0)

	case ASTXRW:
		return LDSTX(2, 0, 0, 0, 0) | 0x1F<<10

	case AMOVNP:
		return S64 | 0<<30 | 5<<27 | 0<<26 | 0<<23 | 1<<22

	case AMOVNPW:
		return S32 | 0<<30 | 5<<27 | 0<<26 | 0<<23 | 1<<22
	}

	ctxt.Diag("bad opstore %v\n%v", obj.Aconv(a), ctxt.Curp)
	return 0
}

/*
 * load/store register (unsigned immediate) C3.3.13
 *	these produce 64-bit values (when there's an option)
 */
func olsr12u(ctxt *obj.Link, o int32, v int32, b int, r int) uint32 {
	if v < 0 || v >= (1<<12) {
		ctxt.Diag("offset out of range: %d\n%v", v, ctxt.Curp)
	}
	o |= (v & 0xFFF) << 10
	o |= int32(b&31) << 5
	o |= int32(r & 31)
	return uint32(o)
}

func opldr12(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AMOVD:
		return LDSTR12U(3, 0, 1) /* imm12<<10 | Rn<<5 | Rt */

	case AMOVW:
		return LDSTR12U(2, 0, 2)

	case AMOVWU:
		return LDSTR12U(2, 0, 1)

	case AMOVH:
		return LDSTR12U(1, 0, 2)

	case AMOVHU:
		return LDSTR12U(1, 0, 1)

	case AMOVB:
		return LDSTR12U(0, 0, 2)

	case AMOVBU:
		return LDSTR12U(0, 0, 1)

	case AFMOVS:
		return LDSTR12U(2, 1, 1)

	case AFMOVD:
		return LDSTR12U(3, 1, 1)
	}

	ctxt.Diag("bad opldr12 %v\n%v", obj.Aconv(a), ctxt.Curp)
	return 0
}

func opstr12(ctxt *obj.Link, a obj.As) uint32 {
	return LD2STR(opldr12(ctxt, a))
}

/*
 * load/store register (unscaled immediate) C3.3.12
 */
func olsr9s(ctxt *obj.Link, o int32, v int32, b int, r int) uint32 {
	if v < -256 || v > 255 {
		ctxt.Diag("offset out of range: %d\n%v", v, ctxt.Curp)
	}
	o |= (v & 0x1FF) << 12
	o |= int32(b&31) << 5
	o |= int32(r & 31)
	return uint32(o)
}

func opldr9(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AMOVD:
		return LDSTR9S(3, 0, 1) /* simm9<<12 | Rn<<5 | Rt */

	case AMOVW:
		return LDSTR9S(2, 0, 2)

	case AMOVWU:
		return LDSTR9S(2, 0, 1)

	case AMOVH:
		return LDSTR9S(1, 0, 2)

	case AMOVHU:
		return LDSTR9S(1, 0, 1)

	case AMOVB:
		return LDSTR9S(0, 0, 2)

	case AMOVBU:
		return LDSTR9S(0, 0, 1)

	case AFMOVS:
		return LDSTR9S(2, 1, 1)

	case AFMOVD:
		return LDSTR9S(3, 1, 1)
	}

	ctxt.Diag("bad opldr9 %v\n%v", obj.Aconv(a), ctxt.Curp)
	return 0
}

func opstr9(ctxt *obj.Link, a obj.As) uint32 {
	return LD2STR(opldr9(ctxt, a))
}

func opldrpp(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AMOVD:
		return 3<<30 | 7<<27 | 0<<26 | 0<<24 | 1<<22 /* simm9<<12 | Rn<<5 | Rt */

	case AMOVW:
		return 2<<30 | 7<<27 | 0<<26 | 0<<24 | 2<<22

	case AMOVWU:
		return 2<<30 | 7<<27 | 0<<26 | 0<<24 | 1<<22

	case AMOVH:
		return 1<<30 | 7<<27 | 0<<26 | 0<<24 | 2<<22

	case AMOVHU:
		return 1<<30 | 7<<27 | 0<<26 | 0<<24 | 1<<22

	case AMOVB:
		return 0<<30 | 7<<27 | 0<<26 | 0<<24 | 2<<22

	case AMOVBU:
		return 0<<30 | 7<<27 | 0<<26 | 0<<24 | 1<<22
	}

	ctxt.Diag("bad opldr %v\n%v", obj.Aconv(a), ctxt.Curp)
	return 0
}

/*
 * load/store register (extended register)
 */
func olsxrr(ctxt *obj.Link, as obj.As, rt int, r1 int, r2 int) uint32 {
	ctxt.Diag("need load/store extended register\n%v", ctxt.Curp)
	return 0xffffffff
}

func oaddi(ctxt *obj.Link, o1 int32, v int32, r int, rt int) uint32 {
	if (v & 0xFFF000) != 0 {
		if v&0xFFF != 0 {
			ctxt.Diag("%v misuses oaddi", ctxt.Curp)
		}
		v >>= 12
		o1 |= 1 << 22
	}

	o1 |= ((v & 0xFFF) << 10) | (int32(r&31) << 5) | int32(rt&31)
	return uint32(o1)
}

/*
 * load a a literal value into dr
 */
func omovlit(ctxt *obj.Link, as obj.As, p *obj.Prog, a *obj.Addr, dr int) uint32 {
	var o1 int32
	if p.Pcond == nil { /* not in literal pool */
		aclass(ctxt, a)
		fmt.Fprintf(ctxt.Bso, "omovlit add %d (%#x)\n", ctxt.Instoffset, uint64(ctxt.Instoffset))

		/* TODO: could be clever, and use general constant builder */
		o1 = int32(opirr(ctxt, AADD))

		v := int32(ctxt.Instoffset)
		if v != 0 && (v&0xFFF) == 0 {
			v >>= 12
			o1 |= 1 << 22 /* shift, by 12 */
		}

		o1 |= ((v & 0xFFF) << 10) | (REGZERO & 31 << 5) | int32(dr&31)
	} else {
		fp := 0
		w := 0 /* default: 32 bit, unsigned */
		switch as {
		case AFMOVS:
			fp = 1

		case AFMOVD:
			fp = 1
			w = 1 /* 64 bit simd&fp */

		case AMOVD:
			if p.Pcond.As == ADWORD {
				w = 1 /* 64 bit */
			} else if p.Pcond.To.Offset < 0 {
				w = 2 /* sign extend */
			}

		case AMOVB, AMOVH, AMOVW:
			w = 2 /* 32 bit, sign-extended to 64 */
			break
		}

		v := int32(brdist(ctxt, p, 0, 19, 2))
		o1 = (int32(w) << 30) | (int32(fp) << 26) | (3 << 27)
		o1 |= (v & 0x7FFFF) << 5
		o1 |= int32(dr & 31)
	}

	return uint32(o1)
}

func opbfm(ctxt *obj.Link, a obj.As, r int, s int, rf int, rt int) uint32 {
	var c uint32
	o := opirr(ctxt, a)
	if (o & (1 << 31)) == 0 {
		c = 32
	} else {
		c = 64
	}
	if r < 0 || uint32(r) >= c {
		ctxt.Diag("illegal bit number\n%v", ctxt.Curp)
	}
	o |= (uint32(r) & 0x3F) << 16
	if s < 0 || uint32(s) >= c {
		ctxt.Diag("illegal bit number\n%v", ctxt.Curp)
	}
	o |= (uint32(s) & 0x3F) << 10
	o |= (uint32(rf&31) << 5) | uint32(rt&31)
	return o
}

func opextr(ctxt *obj.Link, a obj.As, v int32, rn int, rm int, rt int) uint32 {
	var c uint32
	o := opirr(ctxt, a)
	if (o & (1 << 31)) != 0 {
		c = 63
	} else {
		c = 31
	}
	if v < 0 || uint32(v) > c {
		ctxt.Diag("illegal bit number\n%v", ctxt.Curp)
	}
	o |= uint32(v) << 10
	o |= uint32(rn&31) << 5
	o |= uint32(rm&31) << 16
	o |= uint32(rt & 31)
	return o
}

/*
 * size in log2(bytes)
 */
func movesize(a obj.As) int {
	switch a {
	case AMOVD:
		return 3

	case AMOVW, AMOVWU:
		return 2

	case AMOVH, AMOVHU:
		return 1

	case AMOVB, AMOVBU:
		return 0

	case AFMOVS:
		return 2

	case AFMOVD:
		return 3

	default:
		return -1
	}
}
