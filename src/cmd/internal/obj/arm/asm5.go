// Inferno utils/5l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/span.c
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
	"log"
	"math"
	"sort"
)

type Optab struct {
	as       obj.As
	a1       uint8
	a2       int8
	a3       uint8
	type_    uint8
	size     int8
	param    int16
	flag     int8
	pcrelsiz uint8
}

type Opcross [32][2][32]uint8

const (
	LFROM  = 1 << 0
	LTO    = 1 << 1
	LPOOL  = 1 << 2
	LPCREL = 1 << 3
)

var optab = []Optab{
	/* struct Optab:
	OPCODE,	from, prog->reg, to,		 type,size,param,flag */
	{obj.ATEXT, C_ADDR, C_NONE, C_TEXTSIZE, 0, 0, 0, 0, 0},
	{AADD, C_REG, C_REG, C_REG, 1, 4, 0, 0, 0},
	{AADD, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{AMVN, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{ACMP, C_REG, C_REG, C_NONE, 1, 4, 0, 0, 0},
	{AADD, C_RCON, C_REG, C_REG, 2, 4, 0, 0, 0},
	{AADD, C_RCON, C_NONE, C_REG, 2, 4, 0, 0, 0},
	{AMOVW, C_RCON, C_NONE, C_REG, 2, 4, 0, 0, 0},
	{AMVN, C_RCON, C_NONE, C_REG, 2, 4, 0, 0, 0},
	{ACMP, C_RCON, C_REG, C_NONE, 2, 4, 0, 0, 0},
	{AADD, C_SHIFT, C_REG, C_REG, 3, 4, 0, 0, 0},
	{AADD, C_SHIFT, C_NONE, C_REG, 3, 4, 0, 0, 0},
	{AMVN, C_SHIFT, C_NONE, C_REG, 3, 4, 0, 0, 0},
	{ACMP, C_SHIFT, C_REG, C_NONE, 3, 4, 0, 0, 0},
	{AMOVW, C_RACON, C_NONE, C_REG, 4, 4, REGSP, 0, 0},
	{AB, C_NONE, C_NONE, C_SBRA, 5, 4, 0, LPOOL, 0},
	{ABL, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0},
	{ABX, C_NONE, C_NONE, C_SBRA, 74, 20, 0, 0, 0},
	{ABEQ, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0},
	{ABEQ, C_RCON, C_NONE, C_SBRA, 5, 4, 0, 0, 0}, // prediction hinted form, hint ignored

	{AB, C_NONE, C_NONE, C_ROREG, 6, 4, 0, LPOOL, 0},
	{ABL, C_NONE, C_NONE, C_ROREG, 7, 4, 0, 0, 0},
	{ABL, C_REG, C_NONE, C_ROREG, 7, 4, 0, 0, 0},
	{ABX, C_NONE, C_NONE, C_ROREG, 75, 12, 0, 0, 0},
	{ABXRET, C_NONE, C_NONE, C_ROREG, 76, 4, 0, 0, 0},
	{ASLL, C_RCON, C_REG, C_REG, 8, 4, 0, 0, 0},
	{ASLL, C_RCON, C_NONE, C_REG, 8, 4, 0, 0, 0},
	{ASLL, C_REG, C_NONE, C_REG, 9, 4, 0, 0, 0},
	{ASLL, C_REG, C_REG, C_REG, 9, 4, 0, 0, 0},
	{ASWI, C_NONE, C_NONE, C_NONE, 10, 4, 0, 0, 0},
	{ASWI, C_NONE, C_NONE, C_LOREG, 10, 4, 0, 0, 0},
	{ASWI, C_NONE, C_NONE, C_LCON, 10, 4, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_LCON, 11, 4, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_LCONADDR, 11, 4, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_ADDR, 11, 4, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_TLS_LE, 103, 4, 0, 0, 0},
	{AWORD, C_NONE, C_NONE, C_TLS_IE, 104, 4, 0, 0, 0},
	{AMOVW, C_NCON, C_NONE, C_REG, 12, 4, 0, 0, 0},
	{AMOVW, C_LCON, C_NONE, C_REG, 12, 4, 0, LFROM, 0},
	{AMOVW, C_LCONADDR, C_NONE, C_REG, 12, 4, 0, LFROM | LPCREL, 4},
	{AADD, C_NCON, C_REG, C_REG, 13, 8, 0, 0, 0},
	{AADD, C_NCON, C_NONE, C_REG, 13, 8, 0, 0, 0},
	{AMVN, C_NCON, C_NONE, C_REG, 13, 8, 0, 0, 0},
	{ACMP, C_NCON, C_REG, C_NONE, 13, 8, 0, 0, 0},
	{AADD, C_LCON, C_REG, C_REG, 13, 8, 0, LFROM, 0},
	{AADD, C_LCON, C_NONE, C_REG, 13, 8, 0, LFROM, 0},
	{AMVN, C_LCON, C_NONE, C_REG, 13, 8, 0, LFROM, 0},
	{ACMP, C_LCON, C_REG, C_NONE, 13, 8, 0, LFROM, 0},
	{AMOVB, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{AMOVBS, C_REG, C_NONE, C_REG, 14, 8, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_REG, 58, 4, 0, 0, 0},
	{AMOVH, C_REG, C_NONE, C_REG, 1, 4, 0, 0, 0},
	{AMOVHS, C_REG, C_NONE, C_REG, 14, 8, 0, 0, 0},
	{AMOVHU, C_REG, C_NONE, C_REG, 14, 8, 0, 0, 0},
	{AMUL, C_REG, C_REG, C_REG, 15, 4, 0, 0, 0},
	{AMUL, C_REG, C_NONE, C_REG, 15, 4, 0, 0, 0},
	{ADIV, C_REG, C_REG, C_REG, 16, 4, 0, 0, 0},
	{ADIV, C_REG, C_NONE, C_REG, 16, 4, 0, 0, 0},
	{AMULL, C_REG, C_REG, C_REGREG, 17, 4, 0, 0, 0},
	{AMULA, C_REG, C_REG, C_REGREG2, 17, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_SAUTO, 20, 4, REGSP, 0, 0},
	{AMOVW, C_REG, C_NONE, C_SOREG, 20, 4, 0, 0, 0},
	{AMOVB, C_REG, C_NONE, C_SAUTO, 20, 4, REGSP, 0, 0},
	{AMOVB, C_REG, C_NONE, C_SOREG, 20, 4, 0, 0, 0},
	{AMOVBS, C_REG, C_NONE, C_SAUTO, 20, 4, REGSP, 0, 0},
	{AMOVBS, C_REG, C_NONE, C_SOREG, 20, 4, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_SAUTO, 20, 4, REGSP, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_SOREG, 20, 4, 0, 0, 0},
	{AMOVW, C_SAUTO, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVW, C_SOREG, C_NONE, C_REG, 21, 4, 0, 0, 0},
	{AMOVBU, C_SAUTO, C_NONE, C_REG, 21, 4, REGSP, 0, 0},
	{AMOVBU, C_SOREG, C_NONE, C_REG, 21, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, LTO, 0},
	{AMOVW, C_REG, C_NONE, C_LOREG, 30, 8, 0, LTO, 0},
	{AMOVW, C_REG, C_NONE, C_ADDR, 64, 8, 0, LTO | LPCREL, 4},
	{AMOVB, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, LTO, 0},
	{AMOVB, C_REG, C_NONE, C_LOREG, 30, 8, 0, LTO, 0},
	{AMOVB, C_REG, C_NONE, C_ADDR, 64, 8, 0, LTO | LPCREL, 4},
	{AMOVBS, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, LTO, 0},
	{AMOVBS, C_REG, C_NONE, C_LOREG, 30, 8, 0, LTO, 0},
	{AMOVBS, C_REG, C_NONE, C_ADDR, 64, 8, 0, LTO | LPCREL, 4},
	{AMOVBU, C_REG, C_NONE, C_LAUTO, 30, 8, REGSP, LTO, 0},
	{AMOVBU, C_REG, C_NONE, C_LOREG, 30, 8, 0, LTO, 0},
	{AMOVBU, C_REG, C_NONE, C_ADDR, 64, 8, 0, LTO | LPCREL, 4},
	{AMOVW, C_TLS_LE, C_NONE, C_REG, 101, 4, 0, LFROM, 0},
	{AMOVW, C_TLS_IE, C_NONE, C_REG, 102, 8, 0, LFROM, 0},
	{AMOVW, C_LAUTO, C_NONE, C_REG, 31, 8, REGSP, LFROM, 0},
	{AMOVW, C_LOREG, C_NONE, C_REG, 31, 8, 0, LFROM, 0},
	{AMOVW, C_ADDR, C_NONE, C_REG, 65, 8, 0, LFROM | LPCREL, 4},
	{AMOVBU, C_LAUTO, C_NONE, C_REG, 31, 8, REGSP, LFROM, 0},
	{AMOVBU, C_LOREG, C_NONE, C_REG, 31, 8, 0, LFROM, 0},
	{AMOVBU, C_ADDR, C_NONE, C_REG, 65, 8, 0, LFROM | LPCREL, 4},
	{AMOVW, C_LACON, C_NONE, C_REG, 34, 8, REGSP, LFROM, 0},
	{AMOVW, C_PSR, C_NONE, C_REG, 35, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_PSR, 36, 4, 0, 0, 0},
	{AMOVW, C_RCON, C_NONE, C_PSR, 37, 4, 0, 0, 0},
	{AMOVM, C_REGLIST, C_NONE, C_SOREG, 38, 4, 0, 0, 0},
	{AMOVM, C_SOREG, C_NONE, C_REGLIST, 39, 4, 0, 0, 0},
	{ASWPW, C_SOREG, C_REG, C_REG, 40, 4, 0, 0, 0},
	{ARFE, C_NONE, C_NONE, C_NONE, 41, 4, 0, 0, 0},
	{AMOVF, C_FREG, C_NONE, C_FAUTO, 50, 4, REGSP, 0, 0},
	{AMOVF, C_FREG, C_NONE, C_FOREG, 50, 4, 0, 0, 0},
	{AMOVF, C_FAUTO, C_NONE, C_FREG, 51, 4, REGSP, 0, 0},
	{AMOVF, C_FOREG, C_NONE, C_FREG, 51, 4, 0, 0, 0},
	{AMOVF, C_FREG, C_NONE, C_LAUTO, 52, 12, REGSP, LTO, 0},
	{AMOVF, C_FREG, C_NONE, C_LOREG, 52, 12, 0, LTO, 0},
	{AMOVF, C_LAUTO, C_NONE, C_FREG, 53, 12, REGSP, LFROM, 0},
	{AMOVF, C_LOREG, C_NONE, C_FREG, 53, 12, 0, LFROM, 0},
	{AMOVF, C_FREG, C_NONE, C_ADDR, 68, 8, 0, LTO | LPCREL, 4},
	{AMOVF, C_ADDR, C_NONE, C_FREG, 69, 8, 0, LFROM | LPCREL, 4},
	{AADDF, C_FREG, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AADDF, C_FREG, C_REG, C_FREG, 54, 4, 0, 0, 0},
	{AMOVF, C_FREG, C_NONE, C_FREG, 54, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_FCR, 56, 4, 0, 0, 0},
	{AMOVW, C_FCR, C_NONE, C_REG, 57, 4, 0, 0, 0},
	{AMOVW, C_SHIFT, C_NONE, C_REG, 59, 4, 0, 0, 0},
	{AMOVBU, C_SHIFT, C_NONE, C_REG, 59, 4, 0, 0, 0},
	{AMOVB, C_SHIFT, C_NONE, C_REG, 60, 4, 0, 0, 0},
	{AMOVBS, C_SHIFT, C_NONE, C_REG, 60, 4, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_SHIFT, 61, 4, 0, 0, 0},
	{AMOVB, C_REG, C_NONE, C_SHIFT, 61, 4, 0, 0, 0},
	{AMOVBS, C_REG, C_NONE, C_SHIFT, 61, 4, 0, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_SHIFT, 61, 4, 0, 0, 0},
	{AMOVH, C_REG, C_NONE, C_HAUTO, 70, 4, REGSP, 0, 0},
	{AMOVH, C_REG, C_NONE, C_HOREG, 70, 4, 0, 0, 0},
	{AMOVHS, C_REG, C_NONE, C_HAUTO, 70, 4, REGSP, 0, 0},
	{AMOVHS, C_REG, C_NONE, C_HOREG, 70, 4, 0, 0, 0},
	{AMOVHU, C_REG, C_NONE, C_HAUTO, 70, 4, REGSP, 0, 0},
	{AMOVHU, C_REG, C_NONE, C_HOREG, 70, 4, 0, 0, 0},
	{AMOVB, C_HAUTO, C_NONE, C_REG, 71, 4, REGSP, 0, 0},
	{AMOVB, C_HOREG, C_NONE, C_REG, 71, 4, 0, 0, 0},
	{AMOVBS, C_HAUTO, C_NONE, C_REG, 71, 4, REGSP, 0, 0},
	{AMOVBS, C_HOREG, C_NONE, C_REG, 71, 4, 0, 0, 0},
	{AMOVH, C_HAUTO, C_NONE, C_REG, 71, 4, REGSP, 0, 0},
	{AMOVH, C_HOREG, C_NONE, C_REG, 71, 4, 0, 0, 0},
	{AMOVHS, C_HAUTO, C_NONE, C_REG, 71, 4, REGSP, 0, 0},
	{AMOVHS, C_HOREG, C_NONE, C_REG, 71, 4, 0, 0, 0},
	{AMOVHU, C_HAUTO, C_NONE, C_REG, 71, 4, REGSP, 0, 0},
	{AMOVHU, C_HOREG, C_NONE, C_REG, 71, 4, 0, 0, 0},
	{AMOVH, C_REG, C_NONE, C_LAUTO, 72, 8, REGSP, LTO, 0},
	{AMOVH, C_REG, C_NONE, C_LOREG, 72, 8, 0, LTO, 0},
	{AMOVH, C_REG, C_NONE, C_ADDR, 94, 8, 0, LTO | LPCREL, 4},
	{AMOVHS, C_REG, C_NONE, C_LAUTO, 72, 8, REGSP, LTO, 0},
	{AMOVHS, C_REG, C_NONE, C_LOREG, 72, 8, 0, LTO, 0},
	{AMOVHS, C_REG, C_NONE, C_ADDR, 94, 8, 0, LTO | LPCREL, 4},
	{AMOVHU, C_REG, C_NONE, C_LAUTO, 72, 8, REGSP, LTO, 0},
	{AMOVHU, C_REG, C_NONE, C_LOREG, 72, 8, 0, LTO, 0},
	{AMOVHU, C_REG, C_NONE, C_ADDR, 94, 8, 0, LTO | LPCREL, 4},
	{AMOVB, C_LAUTO, C_NONE, C_REG, 73, 8, REGSP, LFROM, 0},
	{AMOVB, C_LOREG, C_NONE, C_REG, 73, 8, 0, LFROM, 0},
	{AMOVB, C_ADDR, C_NONE, C_REG, 93, 8, 0, LFROM | LPCREL, 4},
	{AMOVBS, C_LAUTO, C_NONE, C_REG, 73, 8, REGSP, LFROM, 0},
	{AMOVBS, C_LOREG, C_NONE, C_REG, 73, 8, 0, LFROM, 0},
	{AMOVBS, C_ADDR, C_NONE, C_REG, 93, 8, 0, LFROM | LPCREL, 4},
	{AMOVH, C_LAUTO, C_NONE, C_REG, 73, 8, REGSP, LFROM, 0},
	{AMOVH, C_LOREG, C_NONE, C_REG, 73, 8, 0, LFROM, 0},
	{AMOVH, C_ADDR, C_NONE, C_REG, 93, 8, 0, LFROM | LPCREL, 4},
	{AMOVHS, C_LAUTO, C_NONE, C_REG, 73, 8, REGSP, LFROM, 0},
	{AMOVHS, C_LOREG, C_NONE, C_REG, 73, 8, 0, LFROM, 0},
	{AMOVHS, C_ADDR, C_NONE, C_REG, 93, 8, 0, LFROM | LPCREL, 4},
	{AMOVHU, C_LAUTO, C_NONE, C_REG, 73, 8, REGSP, LFROM, 0},
	{AMOVHU, C_LOREG, C_NONE, C_REG, 73, 8, 0, LFROM, 0},
	{AMOVHU, C_ADDR, C_NONE, C_REG, 93, 8, 0, LFROM | LPCREL, 4},
	{ALDREX, C_SOREG, C_NONE, C_REG, 77, 4, 0, 0, 0},
	{ASTREX, C_SOREG, C_REG, C_REG, 78, 4, 0, 0, 0},
	{AMOVF, C_ZFCON, C_NONE, C_FREG, 80, 8, 0, 0, 0},
	{AMOVF, C_SFCON, C_NONE, C_FREG, 81, 4, 0, 0, 0},
	{ACMPF, C_FREG, C_REG, C_NONE, 82, 8, 0, 0, 0},
	{ACMPF, C_FREG, C_NONE, C_NONE, 83, 8, 0, 0, 0},
	{AMOVFW, C_FREG, C_NONE, C_FREG, 84, 4, 0, 0, 0},
	{AMOVWF, C_FREG, C_NONE, C_FREG, 85, 4, 0, 0, 0},
	{AMOVFW, C_FREG, C_NONE, C_REG, 86, 8, 0, 0, 0},
	{AMOVWF, C_REG, C_NONE, C_FREG, 87, 8, 0, 0, 0},
	{AMOVW, C_REG, C_NONE, C_FREG, 88, 4, 0, 0, 0},
	{AMOVW, C_FREG, C_NONE, C_REG, 89, 4, 0, 0, 0},
	{ATST, C_REG, C_NONE, C_NONE, 90, 4, 0, 0, 0},
	{ALDREXD, C_SOREG, C_NONE, C_REG, 91, 4, 0, 0, 0},
	{ASTREXD, C_SOREG, C_REG, C_REG, 92, 4, 0, 0, 0},
	{APLD, C_SOREG, C_NONE, C_NONE, 95, 4, 0, 0, 0},
	{obj.AUNDEF, C_NONE, C_NONE, C_NONE, 96, 4, 0, 0, 0},
	{ACLZ, C_REG, C_NONE, C_REG, 97, 4, 0, 0, 0},
	{AMULWT, C_REG, C_REG, C_REG, 98, 4, 0, 0, 0},
	{AMULAWT, C_REG, C_REG, C_REGREG2, 99, 4, 0, 0, 0},
	{obj.AUSEFIELD, C_ADDR, C_NONE, C_NONE, 0, 0, 0, 0, 0},
	{obj.APCDATA, C_LCON, C_NONE, C_LCON, 0, 0, 0, 0, 0},
	{obj.AFUNCDATA, C_LCON, C_NONE, C_ADDR, 0, 0, 0, 0, 0},
	{obj.ANOP, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0, 0},
	{obj.ADUFFZERO, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0}, // same as ABL
	{obj.ADUFFCOPY, C_NONE, C_NONE, C_SBRA, 5, 4, 0, 0, 0}, // same as ABL

	{ADATABUNDLE, C_NONE, C_NONE, C_NONE, 100, 4, 0, 0, 0},
	{ADATABUNDLEEND, C_NONE, C_NONE, C_NONE, 100, 0, 0, 0, 0},
	{obj.AXXX, C_NONE, C_NONE, C_NONE, 0, 4, 0, 0, 0},
}

var pool struct {
	start uint32
	size  uint32
	extra uint32
}

var oprange [ALAST & obj.AMask][]Optab

var xcmp [C_GOK + 1][C_GOK + 1]bool

var deferreturn *obj.LSym

// Note about encoding: Prog.scond holds the condition encoding,
// but XOR'ed with C_SCOND_XOR, so that C_SCOND_NONE == 0.
// The code that shifts the value << 28 has the responsibility
// for XORing with C_SCOND_XOR too.

// asmoutnacl assembles the instruction p. It replaces asmout for NaCl.
// It returns the total number of bytes put in out, and it can change
// p->pc if extra padding is necessary.
// In rare cases, asmoutnacl might split p into two instructions.
// origPC is the PC for this Prog (no padding is taken into account).
func asmoutnacl(ctxt *obj.Link, origPC int32, p *obj.Prog, o *Optab, out []uint32) int {
	size := int(o.size)

	// instruction specific
	switch p.As {
	default:
		if out != nil {
			asmout(ctxt, p, o, out)
		}

	case ADATABUNDLE, // align to 16-byte boundary
		ADATABUNDLEEND: // zero width instruction, just to align next instruction to 16-byte boundary
		p.Pc = (p.Pc + 15) &^ 15

		if out != nil {
			asmout(ctxt, p, o, out)
		}

	case obj.AUNDEF,
		APLD:
		size = 4
		if out != nil {
			switch p.As {
			case obj.AUNDEF:
				out[0] = 0xe7fedef0 // NACL_INSTR_ARM_ABORT_NOW (UDF #0xEDE0)

			case APLD:
				out[0] = 0xe1a01001 // (MOVW R1, R1)
			}
		}

	case AB, ABL:
		if p.To.Type != obj.TYPE_MEM {
			if out != nil {
				asmout(ctxt, p, o, out)
			}
		} else {
			if p.To.Offset != 0 || size != 4 || p.To.Reg > REG_R15 || p.To.Reg < REG_R0 {
				ctxt.Diag("unsupported instruction: %v", p)
			}
			if p.Pc&15 == 12 {
				p.Pc += 4
			}
			if out != nil {
				out[0] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x03c0013f | (uint32(p.To.Reg)&15)<<12 | (uint32(p.To.Reg)&15)<<16 // BIC $0xc000000f, Rx
				if p.As == AB {
					out[1] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x012fff10 | (uint32(p.To.Reg)&15)<<0 // BX Rx
				} else { // ABL
					out[1] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x012fff30 | (uint32(p.To.Reg)&15)<<0 // BLX Rx
				}
			}

			size = 8
		}

		// align the last instruction (the actual BL) to the last instruction in a bundle
		if p.As == ABL {
			if deferreturn == nil {
				deferreturn = obj.Linklookup(ctxt, "runtime.deferreturn", 0)
			}
			if p.To.Sym == deferreturn {
				p.Pc = ((int64(origPC) + 15) &^ 15) + 16 - int64(size)
			} else {
				p.Pc += (16 - ((p.Pc + int64(size)) & 15)) & 15
			}
		}

	case ALDREX,
		ALDREXD,
		AMOVB,
		AMOVBS,
		AMOVBU,
		AMOVD,
		AMOVF,
		AMOVH,
		AMOVHS,
		AMOVHU,
		AMOVM,
		AMOVW,
		ASTREX,
		ASTREXD:
		if p.To.Type == obj.TYPE_REG && p.To.Reg == REG_R15 && p.From.Reg == REG_R13 { // MOVW.W x(R13), PC
			if out != nil {
				asmout(ctxt, p, o, out)
			}
			if size == 4 {
				if out != nil {
					// Note: 5c and 5g reg.c know that DIV/MOD smashes R12
					// so that this return instruction expansion is valid.
					out[0] = out[0] &^ 0x3000                                         // change PC to R12
					out[1] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x03ccc13f // BIC $0xc000000f, R12
					out[2] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x012fff1c // BX R12
				}

				size += 8
				if (p.Pc+int64(size))&15 == 4 {
					p.Pc += 4
				}
				break
			} else {
				// if the instruction used more than 4 bytes, then it must have used a very large
				// offset to update R13, so we need to additionally mask R13.
				if out != nil {
					out[size/4-1] &^= 0x3000                                                 // change PC to R12
					out[size/4] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x03cdd103   // BIC $0xc0000000, R13
					out[size/4+1] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x03ccc13f // BIC $0xc000000f, R12
					out[size/4+2] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x012fff1c // BX R12
				}

				// p->pc+size is only ok at 4 or 12 mod 16.
				if (p.Pc+int64(size))%8 == 0 {
					p.Pc += 4
				}
				size += 12
				break
			}
		}

		if p.To.Type == obj.TYPE_REG && p.To.Reg == REG_R15 {
			ctxt.Diag("unsupported instruction (move to another register and use indirect jump instead): %v", p)
		}

		if p.To.Type == obj.TYPE_MEM && p.To.Reg == REG_R13 && (p.Scond&C_WBIT != 0) && size > 4 {
			// function prolog with very large frame size: MOVW.W R14,-100004(R13)
			// split it into two instructions:
			// 	ADD $-100004, R13
			// 	MOVW R14, 0(R13)
			q := ctxt.NewProg()

			p.Scond &^= C_WBIT
			*q = *p
			a := &p.To
			var a2 *obj.Addr
			if p.To.Type == obj.TYPE_MEM {
				a2 = &q.To
			} else {
				a2 = &q.From
			}
			nocache(q)
			nocache(p)

			// insert q after p
			q.Link = p.Link

			p.Link = q
			q.Pcond = nil

			// make p into ADD $X, R13
			p.As = AADD

			p.From = *a
			p.From.Reg = 0
			p.From.Type = obj.TYPE_CONST
			p.To = obj.Addr{}
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_R13

			// make q into p but load/store from 0(R13)
			q.Spadj = 0

			*a2 = obj.Addr{}
			a2.Type = obj.TYPE_MEM
			a2.Reg = REG_R13
			a2.Sym = nil
			a2.Offset = 0
			size = int(oplook(ctxt, p).size)
			break
		}

		if (p.To.Type == obj.TYPE_MEM && p.To.Reg != REG_R9) || // MOVW Rx, X(Ry), y != 9
			(p.From.Type == obj.TYPE_MEM && p.From.Reg != REG_R9) { // MOVW X(Rx), Ry, x != 9
			var a *obj.Addr
			if p.To.Type == obj.TYPE_MEM {
				a = &p.To
			} else {
				a = &p.From
			}
			reg := int(a.Reg)
			if size == 4 {
				// if addr.reg == 0, then it is probably load from x(FP) with small x, no need to modify.
				if reg == 0 {
					if out != nil {
						asmout(ctxt, p, o, out)
					}
				} else {
					if out != nil {
						out[0] = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x03c00103 | (uint32(reg)&15)<<16 | (uint32(reg)&15)<<12 // BIC $0xc0000000, Rx
					}
					if p.Pc&15 == 12 {
						p.Pc += 4
					}
					size += 4
					if out != nil {
						asmout(ctxt, p, o, out[1:])
					}
				}

				break
			} else {
				// if a load/store instruction takes more than 1 word to implement, then
				// we need to separate the instruction into two:
				// 1. explicitly load the address into R11.
				// 2. load/store from R11.
				// This won't handle .W/.P, so we should reject such code.
				if p.Scond&(C_PBIT|C_WBIT) != 0 {
					ctxt.Diag("unsupported instruction (.P/.W): %v", p)
				}
				q := ctxt.NewProg()
				*q = *p
				var a2 *obj.Addr
				if p.To.Type == obj.TYPE_MEM {
					a2 = &q.To
				} else {
					a2 = &q.From
				}
				nocache(q)
				nocache(p)

				// insert q after p
				q.Link = p.Link

				p.Link = q
				q.Pcond = nil

				// make p into MOVW $X(R), R11
				p.As = AMOVW

				p.From = *a
				p.From.Type = obj.TYPE_ADDR
				p.To = obj.Addr{}
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_R11

				// make q into p but load/store from 0(R11)
				*a2 = obj.Addr{}

				a2.Type = obj.TYPE_MEM
				a2.Reg = REG_R11
				a2.Sym = nil
				a2.Offset = 0
				size = int(oplook(ctxt, p).size)
				break
			}
		} else if out != nil {
			asmout(ctxt, p, o, out)
		}
	}

	// destination register specific
	if p.To.Type == obj.TYPE_REG {
		switch p.To.Reg {
		case REG_R9:
			ctxt.Diag("invalid instruction, cannot write to R9: %v", p)

		case REG_R13:
			if out != nil {
				out[size/4] = 0xe3cdd103 // BIC $0xc0000000, R13
			}
			if (p.Pc+int64(size))&15 == 0 {
				p.Pc += 4
			}
			size += 4
		}
	}

	return size
}

func span5(ctxt *obj.Link, cursym *obj.LSym) {
	var p *obj.Prog
	var op *obj.Prog

	p = cursym.Text
	if p == nil || p.Link == nil { // handle external functions and ELF section symbols
		return
	}

	if oprange[AAND&obj.AMask] == nil {
		buildop(ctxt)
	}

	ctxt.Cursym = cursym

	ctxt.Autosize = int32(p.To.Offset + 4)
	c := int32(0)

	op = p
	p = p.Link
	var i int
	var m int
	var o *Optab
	for ; p != nil || ctxt.Blitrl != nil; op, p = p, p.Link {
		if p == nil {
			if checkpool(ctxt, op, 0) {
				p = op
				continue
			}

			// can't happen: blitrl is not nil, but checkpool didn't flushpool
			ctxt.Diag("internal inconsistency")

			break
		}

		ctxt.Curp = p
		p.Pc = int64(c)
		o = oplook(ctxt, p)
		if ctxt.Headtype != obj.Hnacl {
			m = int(o.size)
		} else {
			m = asmoutnacl(ctxt, c, p, o, nil)
			c = int32(p.Pc)     // asmoutnacl might change pc for alignment
			o = oplook(ctxt, p) // asmoutnacl might change p in rare cases
		}

		if m%4 != 0 || p.Pc%4 != 0 {
			ctxt.Diag("!pc invalid: %v size=%d", p, m)
		}

		// must check literal pool here in case p generates many instructions
		if ctxt.Blitrl != nil {
			i = m
			if checkpool(ctxt, op, i) {
				p = op
				continue
			}
		}

		if m == 0 && (p.As != obj.AFUNCDATA && p.As != obj.APCDATA && p.As != ADATABUNDLEEND && p.As != obj.ANOP && p.As != obj.AUSEFIELD) {
			ctxt.Diag("zero-width instruction\n%v", p)
			continue
		}

		switch o.flag & (LFROM | LTO | LPOOL) {
		case LFROM:
			addpool(ctxt, p, &p.From)

		case LTO:
			addpool(ctxt, p, &p.To)

		case LPOOL:
			if p.Scond&C_SCOND == C_SCOND_NONE {
				flushpool(ctxt, p, 0, 0)
			}
		}

		if p.As == AMOVW && p.To.Type == obj.TYPE_REG && p.To.Reg == REGPC && p.Scond&C_SCOND == C_SCOND_NONE {
			flushpool(ctxt, p, 0, 0)
		}
		c += int32(m)
	}

	cursym.Size = int64(c)

	/*
	 * if any procedure is large enough to
	 * generate a large SBRA branch, then
	 * generate extra passes putting branches
	 * around jmps to fix. this is rare.
	 */
	times := 0

	var bflag int
	var opc int32
	var out [6 + 3]uint32
	for {
		if ctxt.Debugvlog != 0 {
			fmt.Fprintf(ctxt.Bso, "%5.2f span1\n", obj.Cputime())
		}
		bflag = 0
		c = 0
		times++
		cursym.Text.Pc = 0 // force re-layout the code.
		for p = cursym.Text; p != nil; p = p.Link {
			ctxt.Curp = p
			o = oplook(ctxt, p)
			if int64(c) > p.Pc {
				p.Pc = int64(c)
			}

			/* very large branches
			if(o->type == 6 && p->pcond) {
				otxt = p->pcond->pc - c;
				if(otxt < 0)
					otxt = -otxt;
				if(otxt >= (1L<<17) - 10) {
					q = emallocz(sizeof(Prog));
					q->link = p->link;
					p->link = q;
					q->as = AB;
					q->to.type = TYPE_BRANCH;
					q->pcond = p->pcond;
					p->pcond = q;
					q = emallocz(sizeof(Prog));
					q->link = p->link;
					p->link = q;
					q->as = AB;
					q->to.type = TYPE_BRANCH;
					q->pcond = q->link->link;
					bflag = 1;
				}
			}
			*/
			opc = int32(p.Pc)

			if ctxt.Headtype != obj.Hnacl {
				m = int(o.size)
			} else {
				m = asmoutnacl(ctxt, c, p, o, nil)
			}
			if p.Pc != int64(opc) {
				bflag = 1
			}

			//print("%v pc changed %d to %d in iter. %d\n", p, opc, (int32)p->pc, times);
			c = int32(p.Pc + int64(m))

			if m%4 != 0 || p.Pc%4 != 0 {
				ctxt.Diag("pc invalid: %v size=%d", p, m)
			}

			if m/4 > len(out) {
				ctxt.Diag("instruction size too large: %d > %d", m/4, len(out))
			}
			if m == 0 && (p.As != obj.AFUNCDATA && p.As != obj.APCDATA && p.As != ADATABUNDLEEND && p.As != obj.ANOP && p.As != obj.AUSEFIELD) {
				if p.As == obj.ATEXT {
					ctxt.Autosize = int32(p.To.Offset + 4)
					continue
				}

				ctxt.Diag("zero-width instruction\n%v", p)
				continue
			}
		}

		cursym.Size = int64(c)
		if bflag == 0 {
			break
		}
	}

	if c%4 != 0 {
		ctxt.Diag("sym->size=%d, invalid", c)
	}

	/*
	 * lay out the code.  all the pc-relative code references,
	 * even cross-function, are resolved now;
	 * only data references need to be relocated.
	 * with more work we could leave cross-function
	 * code references to be relocated too, and then
	 * perhaps we'd be able to parallelize the span loop above.
	 */

	p = cursym.Text
	ctxt.Autosize = int32(p.To.Offset + 4)
	cursym.Grow(cursym.Size)

	bp := cursym.P
	c = int32(p.Pc) // even p->link might need extra padding
	var v int
	for p = p.Link; p != nil; p = p.Link {
		ctxt.Pc = p.Pc
		ctxt.Curp = p
		o = oplook(ctxt, p)
		opc = int32(p.Pc)
		if ctxt.Headtype != obj.Hnacl {
			asmout(ctxt, p, o, out[:])
			m = int(o.size)
		} else {
			m = asmoutnacl(ctxt, c, p, o, out[:])
			if int64(opc) != p.Pc {
				ctxt.Diag("asmoutnacl broken: pc changed (%d->%d) in last stage: %v", opc, int32(p.Pc), p)
			}
		}

		if m%4 != 0 || p.Pc%4 != 0 {
			ctxt.Diag("final stage: pc invalid: %v size=%d", p, m)
		}

		if int64(c) > p.Pc {
			ctxt.Diag("PC padding invalid: want %#d, has %#d: %v", p.Pc, c, p)
		}
		for int64(c) != p.Pc {
			// emit 0xe1a00000 (MOVW R0, R0)
			bp[0] = 0x00
			bp = bp[1:]

			bp[0] = 0x00
			bp = bp[1:]
			bp[0] = 0xa0
			bp = bp[1:]
			bp[0] = 0xe1
			bp = bp[1:]
			c += 4
		}

		for i = 0; i < m/4; i++ {
			v = int(out[i])
			bp[0] = byte(v)
			bp = bp[1:]
			bp[0] = byte(v >> 8)
			bp = bp[1:]
			bp[0] = byte(v >> 16)
			bp = bp[1:]
			bp[0] = byte(v >> 24)
			bp = bp[1:]
		}

		c += int32(m)
	}
}

/*
 * when the first reference to the literal pool threatens
 * to go out of range of a 12-bit PC-relative offset,
 * drop the pool now, and branch round it.
 * this happens only in extended basic blocks that exceed 4k.
 */
func checkpool(ctxt *obj.Link, p *obj.Prog, sz int) bool {
	if pool.size >= 0xff0 || immaddr(int32((p.Pc+int64(sz)+4)+4+int64(12+pool.size)-int64(pool.start+8))) == 0 {
		return flushpool(ctxt, p, 1, 0)
	} else if p.Link == nil {
		return flushpool(ctxt, p, 2, 0)
	}
	return false
}

func flushpool(ctxt *obj.Link, p *obj.Prog, skip int, force int) bool {
	if ctxt.Blitrl != nil {
		if skip != 0 {
			if false && skip == 1 {
				fmt.Printf("note: flush literal pool at %x: len=%d ref=%x\n", uint64(p.Pc+4), pool.size, pool.start)
			}
			q := ctxt.NewProg()
			q.As = AB
			q.To.Type = obj.TYPE_BRANCH
			q.Pcond = p.Link
			q.Link = ctxt.Blitrl
			q.Lineno = p.Lineno
			ctxt.Blitrl = q
		} else if force == 0 && (p.Pc+int64(12+pool.size)-int64(pool.start) < 2048) { // 12 take into account the maximum nacl literal pool alignment padding size
			return false
		}
		if ctxt.Headtype == obj.Hnacl && pool.size%16 != 0 {
			// if pool is not multiple of 16 bytes, add an alignment marker
			q := ctxt.NewProg()

			q.As = ADATABUNDLEEND
			ctxt.Elitrl.Link = q
			ctxt.Elitrl = q
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
		pool.extra = 0
		return true
	}

	return false
}

func addpool(ctxt *obj.Link, p *obj.Prog, a *obj.Addr) {
	var t obj.Prog

	c := aclass(ctxt, a)

	t.Ctxt = ctxt
	t.As = AWORD

	switch c {
	default:
		t.To.Offset = a.Offset
		t.To.Sym = a.Sym
		t.To.Type = a.Type
		t.To.Name = a.Name

		if ctxt.Flag_shared && t.To.Sym != nil {
			t.Rel = p
		}

	case C_SROREG,
		C_LOREG,
		C_ROREG,
		C_FOREG,
		C_SOREG,
		C_HOREG,
		C_FAUTO,
		C_SAUTO,
		C_LAUTO,
		C_LACON:
		t.To.Type = obj.TYPE_CONST
		t.To.Offset = ctxt.Instoffset
	}

	if t.Rel == nil {
		for q := ctxt.Blitrl; q != nil; q = q.Link { /* could hash on t.t0.offset */
			if q.Rel == nil && q.To == t.To {
				p.Pcond = q
				return
			}
		}
	}

	if ctxt.Headtype == obj.Hnacl && pool.size%16 == 0 {
		// start a new data bundle
		q := ctxt.NewProg()
		q.As = ADATABUNDLE
		q.Pc = int64(pool.size)
		pool.size += 4
		if ctxt.Blitrl == nil {
			ctxt.Blitrl = q
			pool.start = uint32(p.Pc)
		} else {
			ctxt.Elitrl.Link = q
		}

		ctxt.Elitrl = q
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
	pool.size += 4

	p.Pcond = q
}

func regoff(ctxt *obj.Link, a *obj.Addr) int32 {
	ctxt.Instoffset = 0
	aclass(ctxt, a)
	return int32(ctxt.Instoffset)
}

func immrot(v uint32) int32 {
	for i := 0; i < 16; i++ {
		if v&^0xff == 0 {
			return int32(uint32(int32(i)<<8) | v | 1<<25)
		}
		v = v<<2 | v>>30
	}

	return 0
}

func immaddr(v int32) int32 {
	if v >= 0 && v <= 0xfff {
		return v&0xfff | 1<<24 | 1<<23 /* pre indexing */ /* pre indexing, up */
	}
	if v >= -0xfff && v < 0 {
		return -v&0xfff | 1<<24 /* pre indexing */
	}
	return 0
}

func immfloat(v int32) bool {
	return v&0xC03 == 0 /* offset will fit in floating-point load/store */
}

func immhalf(v int32) bool {
	if v >= 0 && v <= 0xff {
		return v|1<<24|1<<23 != 0 /* pre indexing */ /* pre indexing, up */
	}
	if v >= -0xff && v < 0 {
		return -v&0xff|1<<24 != 0 /* pre indexing */
	}
	return false
}

func aclass(ctxt *obj.Link, a *obj.Addr) int {
	switch a.Type {
	case obj.TYPE_NONE:
		return C_NONE

	case obj.TYPE_REG:
		ctxt.Instoffset = 0
		if REG_R0 <= a.Reg && a.Reg <= REG_R15 {
			return C_REG
		}
		if REG_F0 <= a.Reg && a.Reg <= REG_F15 {
			return C_FREG
		}
		if a.Reg == REG_FPSR || a.Reg == REG_FPCR {
			return C_FCR
		}
		if a.Reg == REG_CPSR || a.Reg == REG_SPSR {
			return C_PSR
		}
		return C_GOK

	case obj.TYPE_REGREG:
		return C_REGREG

	case obj.TYPE_REGREG2:
		return C_REGREG2

	case obj.TYPE_REGLIST:
		return C_REGLIST

	case obj.TYPE_SHIFT:
		return C_SHIFT

	case obj.TYPE_MEM:
		switch a.Name {
		case obj.NAME_EXTERN,
			obj.NAME_GOTREF,
			obj.NAME_STATIC:
			if a.Sym == nil || a.Sym.Name == "" {
				fmt.Printf("null sym external\n")
				return C_GOK
			}

			ctxt.Instoffset = 0 // s.b. unused but just in case
			if a.Sym.Type == obj.STLSBSS {
				if ctxt.Flag_shared {
					return C_TLS_IE
				} else {
					return C_TLS_LE
				}
			}

			return C_ADDR

		case obj.NAME_AUTO:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset
			t := int(immaddr(int32(ctxt.Instoffset)))
			if t != 0 {
				if immhalf(int32(ctxt.Instoffset)) {
					if immfloat(int32(t)) {
						return C_HFAUTO
					}
					return C_HAUTO
				}

				if immfloat(int32(t)) {
					return C_FAUTO
				}
				return C_SAUTO
			}

			return C_LAUTO

		case obj.NAME_PARAM:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset + 4
			t := int(immaddr(int32(ctxt.Instoffset)))
			if t != 0 {
				if immhalf(int32(ctxt.Instoffset)) {
					if immfloat(int32(t)) {
						return C_HFAUTO
					}
					return C_HAUTO
				}

				if immfloat(int32(t)) {
					return C_FAUTO
				}
				return C_SAUTO
			}

			return C_LAUTO

		case obj.NAME_NONE:
			ctxt.Instoffset = a.Offset
			t := int(immaddr(int32(ctxt.Instoffset)))
			if t != 0 {
				if immhalf(int32(ctxt.Instoffset)) { /* n.b. that it will also satisfy immrot */
					if immfloat(int32(t)) {
						return C_HFOREG
					}
					return C_HOREG
				}

				if immfloat(int32(t)) {
					return C_FOREG /* n.b. that it will also satisfy immrot */
				}
				t := int(immrot(uint32(ctxt.Instoffset)))
				if t != 0 {
					return C_SROREG
				}
				if immhalf(int32(ctxt.Instoffset)) {
					return C_HOREG
				}
				return C_SOREG
			}

			t = int(immrot(uint32(ctxt.Instoffset)))
			if t != 0 {
				return C_ROREG
			}
			return C_LOREG
		}

		return C_GOK

	case obj.TYPE_FCONST:
		if chipzero5(ctxt, a.Val.(float64)) >= 0 {
			return C_ZFCON
		}
		if chipfloat5(ctxt, a.Val.(float64)) >= 0 {
			return C_SFCON
		}
		return C_LFCON

	case obj.TYPE_TEXTSIZE:
		return C_TEXTSIZE

	case obj.TYPE_CONST,
		obj.TYPE_ADDR:
		switch a.Name {
		case obj.NAME_NONE:
			ctxt.Instoffset = a.Offset
			if a.Reg != 0 {
				return aconsize(ctxt)
			}

			t := int(immrot(uint32(ctxt.Instoffset)))
			if t != 0 {
				return C_RCON
			}
			t = int(immrot(^uint32(ctxt.Instoffset)))
			if t != 0 {
				return C_NCON
			}
			return C_LCON

		case obj.NAME_EXTERN,
			obj.NAME_GOTREF,
			obj.NAME_STATIC:
			s := a.Sym
			if s == nil {
				break
			}
			ctxt.Instoffset = 0 // s.b. unused but just in case
			return C_LCONADDR

		case obj.NAME_AUTO:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset
			return aconsize(ctxt)

		case obj.NAME_PARAM:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset + 4
			return aconsize(ctxt)
		}

		return C_GOK

	case obj.TYPE_BRANCH:
		return C_SBRA
	}

	return C_GOK
}

func aconsize(ctxt *obj.Link) int {
	t := int(immrot(uint32(ctxt.Instoffset)))
	if t != 0 {
		return C_RACON
	}
	return C_LACON
}

func prasm(p *obj.Prog) {
	fmt.Printf("%v\n", p)
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
		a2 = C_REG
	}

	if false { /*debug['O']*/
		fmt.Printf("oplook %v %v %v %v\n", obj.Aconv(p.As), DRconv(a1), DRconv(a2), DRconv(a3))
		fmt.Printf("\t\t%d %d\n", p.From.Type, p.To.Type)
	}

	ops := oprange[p.As&obj.AMask]
	c1 := &xcmp[a1]
	c3 := &xcmp[a3]
	for i := range ops {
		op := &ops[i]
		if int(op.a2) == a2 && c1[op.a1] && c3[op.a3] {
			p.Optab = uint16(cap(optab) - cap(ops) + i + 1)
			return op
		}
	}

	ctxt.Diag("illegal combination %v; %v %v %v, %d %d", p, DRconv(a1), DRconv(a2), DRconv(a3), p.From.Type, p.To.Type)
	ctxt.Diag("from %d %d to %d %d\n", p.From.Type, p.From.Name, p.To.Type, p.To.Name)
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
	case C_LCON:
		if b == C_RCON || b == C_NCON {
			return true
		}

	case C_LACON:
		if b == C_RACON {
			return true
		}

	case C_LFCON:
		if b == C_ZFCON || b == C_SFCON {
			return true
		}

	case C_HFAUTO:
		return b == C_HAUTO || b == C_FAUTO

	case C_FAUTO, C_HAUTO:
		return b == C_HFAUTO

	case C_SAUTO:
		return cmp(C_HFAUTO, b)

	case C_LAUTO:
		return cmp(C_SAUTO, b)

	case C_HFOREG:
		return b == C_HOREG || b == C_FOREG

	case C_FOREG, C_HOREG:
		return b == C_HFOREG

	case C_SROREG:
		return cmp(C_SOREG, b) || cmp(C_ROREG, b)

	case C_SOREG, C_ROREG:
		return b == C_SROREG || cmp(C_HFOREG, b)

	case C_LOREG:
		return cmp(C_SROREG, b)

	case C_LBRA:
		if b == C_SBRA {
			return true
		}

	case C_HREG:
		return cmp(C_SP, b) || cmp(C_PC, b)
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
	n := int(p1.as) - int(p2.as)
	if n != 0 {
		return n < 0
	}
	n = int(p1.a1) - int(p2.a1)
	if n != 0 {
		return n < 0
	}
	n = int(p1.a2) - int(p2.a2)
	if n != 0 {
		return n < 0
	}
	n = int(p1.a3) - int(p2.a3)
	if n != 0 {
		return n < 0
	}
	return false
}

func opset(a, b0 obj.As) {
	oprange[a&obj.AMask] = oprange[b0]
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
		if optab[n].flag&LPCREL != 0 {
			if ctxt.Flag_shared {
				optab[n].size += int8(optab[n].pcrelsiz)
			} else {
				optab[n].flag &^= LPCREL
			}
		}
	}

	sort.Sort(ocmp(optab[:n]))
	for i := 0; i < n; i++ {
		r := optab[i].as
		r0 := r & obj.AMask
		start := i
		for optab[i].as == r {
			i++
		}
		oprange[r0] = optab[start:i]
		i--

		switch r {
		default:
			ctxt.Diag("unknown op in build: %v", obj.Aconv(r))
			log.Fatalf("bad code")

		case AADD:
			opset(AAND, r0)
			opset(AEOR, r0)
			opset(ASUB, r0)
			opset(ARSB, r0)
			opset(AADC, r0)
			opset(ASBC, r0)
			opset(ARSC, r0)
			opset(AORR, r0)
			opset(ABIC, r0)

		case ACMP:
			opset(ATEQ, r0)
			opset(ACMN, r0)

		case AMVN:
			break

		case ABEQ:
			opset(ABNE, r0)
			opset(ABCS, r0)
			opset(ABHS, r0)
			opset(ABCC, r0)
			opset(ABLO, r0)
			opset(ABMI, r0)
			opset(ABPL, r0)
			opset(ABVS, r0)
			opset(ABVC, r0)
			opset(ABHI, r0)
			opset(ABLS, r0)
			opset(ABGE, r0)
			opset(ABLT, r0)
			opset(ABGT, r0)
			opset(ABLE, r0)

		case ASLL:
			opset(ASRL, r0)
			opset(ASRA, r0)

		case AMUL:
			opset(AMULU, r0)

		case ADIV:
			opset(AMOD, r0)
			opset(AMODU, r0)
			opset(ADIVU, r0)

		case AMOVW,
			AMOVB,
			AMOVBS,
			AMOVBU,
			AMOVH,
			AMOVHS,
			AMOVHU:
			break

		case ASWPW:
			opset(ASWPBU, r0)

		case AB,
			ABL,
			ABX,
			ABXRET,
			obj.ADUFFZERO,
			obj.ADUFFCOPY,
			ASWI,
			AWORD,
			AMOVM,
			ARFE,
			obj.ATEXT,
			obj.AUSEFIELD,
			obj.ATYPE:
			break

		case AADDF:
			opset(AADDD, r0)
			opset(ASUBF, r0)
			opset(ASUBD, r0)
			opset(AMULF, r0)
			opset(AMULD, r0)
			opset(ADIVF, r0)
			opset(ADIVD, r0)
			opset(ASQRTF, r0)
			opset(ASQRTD, r0)
			opset(AMOVFD, r0)
			opset(AMOVDF, r0)
			opset(AABSF, r0)
			opset(AABSD, r0)

		case ACMPF:
			opset(ACMPD, r0)

		case AMOVF:
			opset(AMOVD, r0)

		case AMOVFW:
			opset(AMOVDW, r0)

		case AMOVWF:
			opset(AMOVWD, r0)

		case AMULL:
			opset(AMULAL, r0)
			opset(AMULLU, r0)
			opset(AMULALU, r0)

		case AMULWT:
			opset(AMULWB, r0)

		case AMULAWT:
			opset(AMULAWB, r0)

		case AMULA,
			ALDREX,
			ASTREX,
			ALDREXD,
			ASTREXD,
			ATST,
			APLD,
			obj.AUNDEF,
			ACLZ,
			obj.AFUNCDATA,
			obj.APCDATA,
			obj.ANOP,
			ADATABUNDLE,
			ADATABUNDLEEND:
			break
		}
	}
}

func asmout(ctxt *obj.Link, p *obj.Prog, o *Optab, out []uint32) {
	ctxt.Printp = p
	o1 := uint32(0)
	o2 := uint32(0)
	o3 := uint32(0)
	o4 := uint32(0)
	o5 := uint32(0)
	o6 := uint32(0)
	ctxt.Armsize += int32(o.size)
	if false { /*debug['P']*/
		fmt.Printf("%x: %v\ttype %d\n", uint32(p.Pc), p, o.type_)
	}
	switch o.type_ {
	default:
		ctxt.Diag("unknown asm %d", o.type_)
		prasm(p)

	case 0: /* pseudo ops */
		if false { /*debug['G']*/
			fmt.Printf("%x: %s: arm\n", uint32(p.Pc), p.From.Sym.Name)
		}

	case 1: /* op R,[R],R */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = 0
		}
		if p.As == AMOVB || p.As == AMOVH || p.As == AMOVW || p.As == AMVN {
			r = 0
		} else if r == 0 {
			r = rt
		}
		o1 |= (uint32(rf)&15)<<0 | (uint32(r)&15)<<16 | (uint32(rt)&15)<<12

	case 2: /* movbu $I,[R],R */
		aclass(ctxt, &p.From)

		o1 = oprrr(ctxt, p.As, int(p.Scond))
		o1 |= uint32(immrot(uint32(ctxt.Instoffset)))
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = 0
		}
		if p.As == AMOVW || p.As == AMVN {
			r = 0
		} else if r == 0 {
			r = rt
		}
		o1 |= (uint32(r)&15)<<16 | (uint32(rt)&15)<<12

	case 3: /* add R<<[IR],[R],R */
		o1 = mov(ctxt, p)

	case 4: /* add $I,[R],R */
		aclass(ctxt, &p.From)

		o1 = oprrr(ctxt, AADD, int(p.Scond))
		o1 |= uint32(immrot(uint32(ctxt.Instoffset)))
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 |= (uint32(r) & 15) << 16
		o1 |= (uint32(p.To.Reg) & 15) << 12

	case 5: /* bra s */
		o1 = opbra(ctxt, p, p.As, int(p.Scond))

		v := int32(-8)
		if p.To.Sym != nil {
			rel := obj.Addrel(ctxt.Cursym)
			rel.Off = int32(ctxt.Pc)
			rel.Siz = 4
			rel.Sym = p.To.Sym
			v += int32(p.To.Offset)
			rel.Add = int64(o1) | (int64(v)>>2)&0xffffff
			rel.Type = obj.R_CALLARM
			break
		}

		if p.Pcond != nil {
			v = int32((p.Pcond.Pc - ctxt.Pc) - 8)
		}
		o1 |= (uint32(v) >> 2) & 0xffffff

	case 6: /* b ,O(R) -> add $O,R,PC */
		aclass(ctxt, &p.To)

		o1 = oprrr(ctxt, AADD, int(p.Scond))
		o1 |= uint32(immrot(uint32(ctxt.Instoffset)))
		o1 |= (uint32(p.To.Reg) & 15) << 16
		o1 |= (REGPC & 15) << 12

	case 7: /* bl (R) -> blx R */
		aclass(ctxt, &p.To)

		if ctxt.Instoffset != 0 {
			ctxt.Diag("%v: doesn't support BL offset(REG) with non-zero offset %d", p, ctxt.Instoffset)
		}
		o1 = oprrr(ctxt, ABL, int(p.Scond))
		o1 |= (uint32(p.To.Reg) & 15) << 0
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 0
		rel.Type = obj.R_CALLIND

	case 8: /* sll $c,[R],R -> mov (R<<$c),R */
		aclass(ctxt, &p.From)

		o1 = oprrr(ctxt, p.As, int(p.Scond))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 |= (uint32(r) & 15) << 0
		o1 |= uint32((ctxt.Instoffset & 31) << 7)
		o1 |= (uint32(p.To.Reg) & 15) << 12

	case 9: /* sll R,[R],R -> mov (R<<R),R */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 |= (uint32(r) & 15) << 0
		o1 |= (uint32(p.From.Reg)&15)<<8 | 1<<4
		o1 |= (uint32(p.To.Reg) & 15) << 12

	case 10: /* swi [$con] */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		if p.To.Type != obj.TYPE_NONE {
			aclass(ctxt, &p.To)
			o1 |= uint32(ctxt.Instoffset & 0xffffff)
		}

	case 11: /* word */
		aclass(ctxt, &p.To)

		o1 = uint32(ctxt.Instoffset)
		if p.To.Sym != nil {
			// This case happens with words generated
			// in the PC stream as part of the literal pool.
			rel := obj.Addrel(ctxt.Cursym)

			rel.Off = int32(ctxt.Pc)
			rel.Siz = 4
			rel.Sym = p.To.Sym
			rel.Add = p.To.Offset

			if ctxt.Flag_shared {
				if p.To.Name == obj.NAME_GOTREF {
					rel.Type = obj.R_GOTPCREL
				} else {
					rel.Type = obj.R_PCREL
				}
				rel.Add += ctxt.Pc - p.Rel.Pc - 8
			} else {
				rel.Type = obj.R_ADDR
			}
			o1 = 0
		}

	case 12: /* movw $lcon, reg */
		o1 = omvl(ctxt, p, &p.From, int(p.To.Reg))

		if o.flag&LPCREL != 0 {
			o2 = oprrr(ctxt, AADD, int(p.Scond)) | (uint32(p.To.Reg)&15)<<0 | (REGPC&15)<<16 | (uint32(p.To.Reg)&15)<<12
		}

	case 13: /* op $lcon, [R], R */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}
		o2 = oprrr(ctxt, p.As, int(p.Scond))
		o2 |= REGTMP & 15
		r := int(p.Reg)
		if p.As == AMOVW || p.As == AMVN {
			r = 0
		} else if r == 0 {
			r = int(p.To.Reg)
		}
		o2 |= (uint32(r) & 15) << 16
		if p.To.Type != obj.TYPE_NONE {
			o2 |= (uint32(p.To.Reg) & 15) << 12
		}

	case 14: /* movb/movbu/movh/movhu R,R */
		o1 = oprrr(ctxt, ASLL, int(p.Scond))

		if p.As == AMOVBU || p.As == AMOVHU {
			o2 = oprrr(ctxt, ASRL, int(p.Scond))
		} else {
			o2 = oprrr(ctxt, ASRA, int(p.Scond))
		}

		r := int(p.To.Reg)
		o1 |= (uint32(p.From.Reg)&15)<<0 | (uint32(r)&15)<<12
		o2 |= uint32(r)&15 | (uint32(r)&15)<<12
		if p.As == AMOVB || p.As == AMOVBS || p.As == AMOVBU {
			o1 |= 24 << 7
			o2 |= 24 << 7
		} else {
			o1 |= 16 << 7
			o2 |= 16 << 7
		}

	case 15: /* mul r,[r,]r */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if r == 0 {
			r = rt
		}
		if rt == r {
			r = rf
			rf = rt
		}

		if false {
			if rt == r || rf == REGPC&15 || r == REGPC&15 || rt == REGPC&15 {
				ctxt.Diag("bad registers in MUL")
				prasm(p)
			}
		}

		o1 |= (uint32(rf)&15)<<8 | (uint32(r)&15)<<0 | (uint32(rt)&15)<<16

	case 16: /* div r,[r,]r */
		o1 = 0xf << 28

		o2 = 0

	case 17:
		o1 = oprrr(ctxt, p.As, int(p.Scond))
		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		rt2 := int(p.To.Offset)
		r := int(p.Reg)
		o1 |= (uint32(rf)&15)<<8 | (uint32(r)&15)<<0 | (uint32(rt)&15)<<16 | (uint32(rt2)&15)<<12

	case 20: /* mov/movb/movbu R,O(R) */
		aclass(ctxt, &p.To)

		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = osr(ctxt, p.As, int(p.From.Reg), int32(ctxt.Instoffset), r, int(p.Scond))

	case 21: /* mov/movbu O(R),R -> lr */
		aclass(ctxt, &p.From)

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = olr(ctxt, int32(ctxt.Instoffset), r, int(p.To.Reg), int(p.Scond))
		if p.As != AMOVW {
			o1 |= 1 << 22
		}

	case 30: /* mov/movb/movbu R,L(R) */
		o1 = omvl(ctxt, p, &p.To, REGTMP)

		if o1 == 0 {
			break
		}
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = osrr(ctxt, int(p.From.Reg), REGTMP&15, r, int(p.Scond))
		if p.As != AMOVW {
			o2 |= 1 << 22
		}

	case 31: /* mov/movbu L(R),R -> lr[b] */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = olrr(ctxt, REGTMP&15, r, int(p.To.Reg), int(p.Scond))
		if p.As == AMOVBU || p.As == AMOVBS || p.As == AMOVB {
			o2 |= 1 << 22
		}

	case 34: /* mov $lacon,R */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}

		o2 = oprrr(ctxt, AADD, int(p.Scond))
		o2 |= REGTMP & 15
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 |= (uint32(r) & 15) << 16
		if p.To.Type != obj.TYPE_NONE {
			o2 |= (uint32(p.To.Reg) & 15) << 12
		}

	case 35: /* mov PSR,R */
		o1 = 2<<23 | 0xf<<16 | 0<<0

		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28
		o1 |= (uint32(p.From.Reg) & 1) << 22
		o1 |= (uint32(p.To.Reg) & 15) << 12

	case 36: /* mov R,PSR */
		o1 = 2<<23 | 0x29f<<12 | 0<<4

		if p.Scond&C_FBIT != 0 {
			o1 ^= 0x010 << 12
		}
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28
		o1 |= (uint32(p.To.Reg) & 1) << 22
		o1 |= (uint32(p.From.Reg) & 15) << 0

	case 37: /* mov $con,PSR */
		aclass(ctxt, &p.From)

		o1 = 2<<23 | 0x29f<<12 | 0<<4
		if p.Scond&C_FBIT != 0 {
			o1 ^= 0x010 << 12
		}
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28
		o1 |= uint32(immrot(uint32(ctxt.Instoffset)))
		o1 |= (uint32(p.To.Reg) & 1) << 22
		o1 |= (uint32(p.From.Reg) & 15) << 0

	case 38, 39:
		switch o.type_ {
		case 38: /* movm $con,oreg -> stm */
			o1 = 0x4 << 25

			o1 |= uint32(p.From.Offset & 0xffff)
			o1 |= (uint32(p.To.Reg) & 15) << 16
			aclass(ctxt, &p.To)

		case 39: /* movm oreg,$con -> ldm */
			o1 = 0x4<<25 | 1<<20

			o1 |= uint32(p.To.Offset & 0xffff)
			o1 |= (uint32(p.From.Reg) & 15) << 16
			aclass(ctxt, &p.From)
		}

		if ctxt.Instoffset != 0 {
			ctxt.Diag("offset must be zero in MOVM; %v", p)
		}
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28
		if p.Scond&C_PBIT != 0 {
			o1 |= 1 << 24
		}
		if p.Scond&C_UBIT != 0 {
			o1 |= 1 << 23
		}
		if p.Scond&C_SBIT != 0 {
			o1 |= 1 << 22
		}
		if p.Scond&C_WBIT != 0 {
			o1 |= 1 << 21
		}

	case 40: /* swp oreg,reg,reg */
		aclass(ctxt, &p.From)

		if ctxt.Instoffset != 0 {
			ctxt.Diag("offset must be zero in SWP")
		}
		o1 = 0x2<<23 | 0x9<<4
		if p.As != ASWPW {
			o1 |= 1 << 22
		}
		o1 |= (uint32(p.From.Reg) & 15) << 16
		o1 |= (uint32(p.Reg) & 15) << 0
		o1 |= (uint32(p.To.Reg) & 15) << 12
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

	case 41: /* rfe -> movm.s.w.u 0(r13),[r15] */
		o1 = 0xe8fd8000

	case 50: /* floating point store */
		v := regoff(ctxt, &p.To)

		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = ofsr(ctxt, p.As, int(p.From.Reg), v, r, int(p.Scond), p)

	case 51: /* floating point load */
		v := regoff(ctxt, &p.From)

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = ofsr(ctxt, p.As, int(p.To.Reg), v, r, int(p.Scond), p) | 1<<20

	case 52: /* floating point store, int32 offset UGLY */
		o1 = omvl(ctxt, p, &p.To, REGTMP)

		if o1 == 0 {
			break
		}
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = oprrr(ctxt, AADD, int(p.Scond)) | (REGTMP&15)<<12 | (REGTMP&15)<<16 | (uint32(r)&15)<<0
		o3 = ofsr(ctxt, p.As, int(p.From.Reg), 0, REGTMP, int(p.Scond), p)

	case 53: /* floating point load, int32 offset UGLY */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = oprrr(ctxt, AADD, int(p.Scond)) | (REGTMP&15)<<12 | (REGTMP&15)<<16 | (uint32(r)&15)<<0
		o3 = ofsr(ctxt, p.As, int(p.To.Reg), 0, (REGTMP&15), int(p.Scond), p) | 1<<20

	case 54: /* floating point arith */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		rf := int(p.From.Reg)
		rt := int(p.To.Reg)
		r := int(p.Reg)
		if r == 0 {
			r = rt
			if p.As == AMOVF || p.As == AMOVD || p.As == AMOVFD || p.As == AMOVDF || p.As == ASQRTF || p.As == ASQRTD || p.As == AABSF || p.As == AABSD {
				r = 0
			}
		}

		o1 |= (uint32(rf)&15)<<0 | (uint32(r)&15)<<16 | (uint32(rt)&15)<<12

	case 56: /* move to FP[CS]R */
		o1 = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0xe<<24 | 1<<8 | 1<<4

		o1 |= ((uint32(p.To.Reg)&1)+1)<<21 | (uint32(p.From.Reg)&15)<<12

	case 57: /* move from FP[CS]R */
		o1 = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0xe<<24 | 1<<8 | 1<<4

		o1 |= ((uint32(p.From.Reg)&1)+1)<<21 | (uint32(p.To.Reg)&15)<<12 | 1<<20

	case 58: /* movbu R,R */
		o1 = oprrr(ctxt, AAND, int(p.Scond))

		o1 |= uint32(immrot(0xff))
		rt := int(p.To.Reg)
		r := int(p.From.Reg)
		if p.To.Type == obj.TYPE_NONE {
			rt = 0
		}
		if r == 0 {
			r = rt
		}
		o1 |= (uint32(r)&15)<<16 | (uint32(rt)&15)<<12

	case 59: /* movw/bu R<<I(R),R -> ldr indexed */
		if p.From.Reg == 0 {
			if p.As != AMOVW {
				ctxt.Diag("byte MOV from shifter operand")
			}
			o1 = mov(ctxt, p)
			break
		}

		if p.From.Offset&(1<<4) != 0 {
			ctxt.Diag("bad shift in LDR")
		}
		o1 = olrr(ctxt, int(p.From.Offset), int(p.From.Reg), int(p.To.Reg), int(p.Scond))
		if p.As == AMOVBU {
			o1 |= 1 << 22
		}

	case 60: /* movb R(R),R -> ldrsb indexed */
		if p.From.Reg == 0 {
			ctxt.Diag("byte MOV from shifter operand")
			o1 = mov(ctxt, p)
			break
		}

		if p.From.Offset&(^0xf) != 0 {
			ctxt.Diag("bad shift in LDRSB")
		}
		o1 = olhrr(ctxt, int(p.From.Offset), int(p.From.Reg), int(p.To.Reg), int(p.Scond))
		o1 ^= 1<<5 | 1<<6

	case 61: /* movw/b/bu R,R<<[IR](R) -> str indexed */
		if p.To.Reg == 0 {
			ctxt.Diag("MOV to shifter operand")
		}
		o1 = osrr(ctxt, int(p.From.Reg), int(p.To.Offset), int(p.To.Reg), int(p.Scond))
		if p.As == AMOVB || p.As == AMOVBS || p.As == AMOVBU {
			o1 |= 1 << 22
		}

		/* reloc ops */
	case 64: /* mov/movb/movbu R,addr */
		o1 = omvl(ctxt, p, &p.To, REGTMP)

		if o1 == 0 {
			break
		}
		o2 = osr(ctxt, p.As, int(p.From.Reg), 0, REGTMP, int(p.Scond))
		if o.flag&LPCREL != 0 {
			o3 = o2
			o2 = oprrr(ctxt, AADD, int(p.Scond)) | REGTMP&15 | (REGPC&15)<<16 | (REGTMP&15)<<12
		}

	case 65: /* mov/movbu addr,R */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}
		o2 = olr(ctxt, 0, REGTMP, int(p.To.Reg), int(p.Scond))
		if p.As == AMOVBU || p.As == AMOVBS || p.As == AMOVB {
			o2 |= 1 << 22
		}
		if o.flag&LPCREL != 0 {
			o3 = o2
			o2 = oprrr(ctxt, AADD, int(p.Scond)) | REGTMP&15 | (REGPC&15)<<16 | (REGTMP&15)<<12
		}

	case 101: /* movw tlsvar,R, local exec*/
		if p.Scond&C_SCOND != C_SCOND_NONE {
			ctxt.Diag("conditional tls")
		}
		o1 = omvl(ctxt, p, &p.From, int(p.To.Reg))

	case 102: /* movw tlsvar,R, initial exec*/
		if p.Scond&C_SCOND != C_SCOND_NONE {
			ctxt.Diag("conditional tls")
		}
		o1 = omvl(ctxt, p, &p.From, int(p.To.Reg))
		o2 = olrr(ctxt, int(p.To.Reg)&15, (REGPC & 15), int(p.To.Reg), int(p.Scond))

	case 103: /* word tlsvar, local exec */
		if p.To.Sym == nil {
			ctxt.Diag("nil sym in tls %v", p)
		}
		if p.To.Offset != 0 {
			ctxt.Diag("offset against tls var in %v", p)
		}
		// This case happens with words generated in the PC stream as part of
		// the literal pool.
		rel := obj.Addrel(ctxt.Cursym)

		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.To.Sym
		rel.Type = obj.R_TLS_LE
		o1 = 0

	case 104: /* word tlsvar, initial exec */
		if p.To.Sym == nil {
			ctxt.Diag("nil sym in tls %v", p)
		}
		if p.To.Offset != 0 {
			ctxt.Diag("offset against tls var in %v", p)
		}
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.To.Sym
		rel.Type = obj.R_TLS_IE
		rel.Add = ctxt.Pc - p.Rel.Pc - 8 - int64(rel.Siz)

	case 68: /* floating point store -> ADDR */
		o1 = omvl(ctxt, p, &p.To, REGTMP)

		if o1 == 0 {
			break
		}
		o2 = ofsr(ctxt, p.As, int(p.From.Reg), 0, REGTMP, int(p.Scond), p)
		if o.flag&LPCREL != 0 {
			o3 = o2
			o2 = oprrr(ctxt, AADD, int(p.Scond)) | REGTMP&15 | (REGPC&15)<<16 | (REGTMP&15)<<12
		}

	case 69: /* floating point load <- ADDR */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}
		o2 = ofsr(ctxt, p.As, int(p.To.Reg), 0, (REGTMP&15), int(p.Scond), p) | 1<<20
		if o.flag&LPCREL != 0 {
			o3 = o2
			o2 = oprrr(ctxt, AADD, int(p.Scond)) | REGTMP&15 | (REGPC&15)<<16 | (REGTMP&15)<<12
		}

		/* ArmV4 ops: */
	case 70: /* movh/movhu R,O(R) -> strh */
		aclass(ctxt, &p.To)

		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = oshr(ctxt, int(p.From.Reg), int32(ctxt.Instoffset), r, int(p.Scond))

	case 71: /* movb/movh/movhu O(R),R -> ldrsb/ldrsh/ldrh */
		aclass(ctxt, &p.From)

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = olhr(ctxt, int32(ctxt.Instoffset), r, int(p.To.Reg), int(p.Scond))
		if p.As == AMOVB || p.As == AMOVBS {
			o1 ^= 1<<5 | 1<<6
		} else if p.As == AMOVH || p.As == AMOVHS {
			o1 ^= (1 << 6)
		}

	case 72: /* movh/movhu R,L(R) -> strh */
		o1 = omvl(ctxt, p, &p.To, REGTMP)

		if o1 == 0 {
			break
		}
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = oshrr(ctxt, int(p.From.Reg), REGTMP&15, r, int(p.Scond))

	case 73: /* movb/movh/movhu L(R),R -> ldrsb/ldrsh/ldrh */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o2 = olhrr(ctxt, REGTMP&15, r, int(p.To.Reg), int(p.Scond))
		if p.As == AMOVB || p.As == AMOVBS {
			o2 ^= 1<<5 | 1<<6
		} else if p.As == AMOVH || p.As == AMOVHS {
			o2 ^= (1 << 6)
		}

	case 74: /* bx $I */
		ctxt.Diag("ABX $I")

	case 75: /* bx O(R) */
		aclass(ctxt, &p.To)

		if ctxt.Instoffset != 0 {
			ctxt.Diag("non-zero offset in ABX")
		}

		/*
			o1 = 	oprrr(ctxt, AADD, p->scond) | immrot(0) | ((REGPC&15)<<16) | ((REGLINK&15)<<12);	// mov PC, LR
			o2 = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | (0x12fff<<8) | (1<<4) | ((p->to.reg&15) << 0);		// BX R
		*/
		// p->to.reg may be REGLINK
		o1 = oprrr(ctxt, AADD, int(p.Scond))

		o1 |= uint32(immrot(uint32(ctxt.Instoffset)))
		o1 |= (uint32(p.To.Reg) & 15) << 16
		o1 |= (REGTMP & 15) << 12
		o2 = oprrr(ctxt, AADD, int(p.Scond)) | uint32(immrot(0)) | (REGPC&15)<<16 | (REGLINK&15)<<12 // mov PC, LR
		o3 = ((uint32(p.Scond)&C_SCOND)^C_SCOND_XOR)<<28 | 0x12fff<<8 | 1<<4 | REGTMP&15             // BX Rtmp

	case 76: /* bx O(R) when returning from fn*/
		ctxt.Diag("ABXRET")

	case 77: /* ldrex oreg,reg */
		aclass(ctxt, &p.From)

		if ctxt.Instoffset != 0 {
			ctxt.Diag("offset must be zero in LDREX")
		}
		o1 = 0x19<<20 | 0xf9f
		o1 |= (uint32(p.From.Reg) & 15) << 16
		o1 |= (uint32(p.To.Reg) & 15) << 12
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

	case 78: /* strex reg,oreg,reg */
		aclass(ctxt, &p.From)

		if ctxt.Instoffset != 0 {
			ctxt.Diag("offset must be zero in STREX")
		}
		o1 = 0x18<<20 | 0xf90
		o1 |= (uint32(p.From.Reg) & 15) << 16
		o1 |= (uint32(p.Reg) & 15) << 0
		o1 |= (uint32(p.To.Reg) & 15) << 12
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

	case 80: /* fmov zfcon,freg */
		if p.As == AMOVD {
			o1 = 0xeeb00b00 // VMOV imm 64
			o2 = oprrr(ctxt, ASUBD, int(p.Scond))
		} else {
			o1 = 0x0eb00a00 // VMOV imm 32
			o2 = oprrr(ctxt, ASUBF, int(p.Scond))
		}

		v := int32(0x70) // 1.0
		r := (int(p.To.Reg) & 15) << 0

		// movf $1.0, r
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

		o1 |= (uint32(r) & 15) << 12
		o1 |= (uint32(v) & 0xf) << 0
		o1 |= (uint32(v) & 0xf0) << 12

		// subf r,r,r
		o2 |= (uint32(r)&15)<<0 | (uint32(r)&15)<<16 | (uint32(r)&15)<<12

	case 81: /* fmov sfcon,freg */
		o1 = 0x0eb00a00 // VMOV imm 32
		if p.As == AMOVD {
			o1 = 0xeeb00b00 // VMOV imm 64
		}
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28
		o1 |= (uint32(p.To.Reg) & 15) << 12
		v := int32(chipfloat5(ctxt, p.From.Val.(float64)))
		o1 |= (uint32(v) & 0xf) << 0
		o1 |= (uint32(v) & 0xf0) << 12

	case 82: /* fcmp freg,freg, */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.Reg)&15)<<12 | (uint32(p.From.Reg)&15)<<0
		o2 = 0x0ef1fa10 // VMRS R15
		o2 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

	case 83: /* fcmp freg,, */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.From.Reg)&15)<<12 | 1<<16
		o2 = 0x0ef1fa10 // VMRS R15
		o2 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

	case 84: /* movfw freg,freg - truncate float-to-fix */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.From.Reg) & 15) << 0
		o1 |= (uint32(p.To.Reg) & 15) << 12

	case 85: /* movwf freg,freg - fix-to-float */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.From.Reg) & 15) << 0
		o1 |= (uint32(p.To.Reg) & 15) << 12

		// macro for movfw freg,FTMP; movw FTMP,reg
	case 86: /* movfw freg,reg - truncate float-to-fix */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.From.Reg) & 15) << 0
		o1 |= (FREGTMP & 15) << 12
		o2 = oprrr(ctxt, -AMOVFW, int(p.Scond))
		o2 |= (FREGTMP & 15) << 16
		o2 |= (uint32(p.To.Reg) & 15) << 12

		// macro for movw reg,FTMP; movwf FTMP,freg
	case 87: /* movwf reg,freg - fix-to-float */
		o1 = oprrr(ctxt, -AMOVWF, int(p.Scond))

		o1 |= (uint32(p.From.Reg) & 15) << 12
		o1 |= (FREGTMP & 15) << 16
		o2 = oprrr(ctxt, p.As, int(p.Scond))
		o2 |= (FREGTMP & 15) << 0
		o2 |= (uint32(p.To.Reg) & 15) << 12

	case 88: /* movw reg,freg  */
		o1 = oprrr(ctxt, -AMOVWF, int(p.Scond))

		o1 |= (uint32(p.From.Reg) & 15) << 12
		o1 |= (uint32(p.To.Reg) & 15) << 16

	case 89: /* movw freg,reg  */
		o1 = oprrr(ctxt, -AMOVFW, int(p.Scond))

		o1 |= (uint32(p.From.Reg) & 15) << 16
		o1 |= (uint32(p.To.Reg) & 15) << 12

	case 90: /* tst reg  */
		o1 = oprrr(ctxt, -ACMP, int(p.Scond))

		o1 |= (uint32(p.From.Reg) & 15) << 16

	case 91: /* ldrexd oreg,reg */
		aclass(ctxt, &p.From)

		if ctxt.Instoffset != 0 {
			ctxt.Diag("offset must be zero in LDREX")
		}
		o1 = 0x1b<<20 | 0xf9f
		o1 |= (uint32(p.From.Reg) & 15) << 16
		o1 |= (uint32(p.To.Reg) & 15) << 12
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

	case 92: /* strexd reg,oreg,reg */
		aclass(ctxt, &p.From)

		if ctxt.Instoffset != 0 {
			ctxt.Diag("offset must be zero in STREX")
		}
		o1 = 0x1a<<20 | 0xf90
		o1 |= (uint32(p.From.Reg) & 15) << 16
		o1 |= (uint32(p.Reg) & 15) << 0
		o1 |= (uint32(p.To.Reg) & 15) << 12
		o1 |= ((uint32(p.Scond) & C_SCOND) ^ C_SCOND_XOR) << 28

	case 93: /* movb/movh/movhu addr,R -> ldrsb/ldrsh/ldrh */
		o1 = omvl(ctxt, p, &p.From, REGTMP)

		if o1 == 0 {
			break
		}
		o2 = olhr(ctxt, 0, REGTMP, int(p.To.Reg), int(p.Scond))
		if p.As == AMOVB || p.As == AMOVBS {
			o2 ^= 1<<5 | 1<<6
		} else if p.As == AMOVH || p.As == AMOVHS {
			o2 ^= (1 << 6)
		}
		if o.flag&LPCREL != 0 {
			o3 = o2
			o2 = oprrr(ctxt, AADD, int(p.Scond)) | REGTMP&15 | (REGPC&15)<<16 | (REGTMP&15)<<12
		}

	case 94: /* movh/movhu R,addr -> strh */
		o1 = omvl(ctxt, p, &p.To, REGTMP)

		if o1 == 0 {
			break
		}
		o2 = oshr(ctxt, int(p.From.Reg), 0, REGTMP, int(p.Scond))
		if o.flag&LPCREL != 0 {
			o3 = o2
			o2 = oprrr(ctxt, AADD, int(p.Scond)) | REGTMP&15 | (REGPC&15)<<16 | (REGTMP&15)<<12
		}

	case 95: /* PLD off(reg) */
		o1 = 0xf5d0f000

		o1 |= (uint32(p.From.Reg) & 15) << 16
		if p.From.Offset < 0 {
			o1 &^= (1 << 23)
			o1 |= uint32((-p.From.Offset) & 0xfff)
		} else {
			o1 |= uint32(p.From.Offset & 0xfff)
		}

		// This is supposed to be something that stops execution.
	// It's not supposed to be reached, ever, but if it is, we'd
	// like to be able to tell how we got there. Assemble as
	// 0xf7fabcfd which is guaranteed to raise undefined instruction
	// exception.
	case 96: /* UNDEF */
		o1 = 0xf7fabcfd

	case 97: /* CLZ Rm, Rd */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.To.Reg) & 15) << 12
		o1 |= (uint32(p.From.Reg) & 15) << 0

	case 98: /* MULW{T,B} Rs, Rm, Rd */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.To.Reg) & 15) << 16
		o1 |= (uint32(p.From.Reg) & 15) << 8
		o1 |= (uint32(p.Reg) & 15) << 0

	case 99: /* MULAW{T,B} Rs, Rm, Rn, Rd */
		o1 = oprrr(ctxt, p.As, int(p.Scond))

		o1 |= (uint32(p.To.Reg) & 15) << 12
		o1 |= (uint32(p.From.Reg) & 15) << 8
		o1 |= (uint32(p.Reg) & 15) << 0
		o1 |= uint32((p.To.Offset & 15) << 16)

		// DATABUNDLE: BKPT $0x5be0, signify the start of NaCl data bundle;
	// DATABUNDLEEND: zero width alignment marker
	case 100:
		if p.As == ADATABUNDLE {
			o1 = 0xe125be70
		}
	}

	out[0] = o1
	out[1] = o2
	out[2] = o3
	out[3] = o4
	out[4] = o5
	out[5] = o6
	return
}

func mov(ctxt *obj.Link, p *obj.Prog) uint32 {
	aclass(ctxt, &p.From)
	o1 := oprrr(ctxt, p.As, int(p.Scond))
	o1 |= uint32(p.From.Offset)
	rt := int(p.To.Reg)
	if p.To.Type == obj.TYPE_NONE {
		rt = 0
	}
	r := int(p.Reg)
	if p.As == AMOVW || p.As == AMVN {
		r = 0
	} else if r == 0 {
		r = rt
	}
	o1 |= (uint32(r)&15)<<16 | (uint32(rt)&15)<<12
	return o1
}

func oprrr(ctxt *obj.Link, a obj.As, sc int) uint32 {
	o := ((uint32(sc) & C_SCOND) ^ C_SCOND_XOR) << 28
	if sc&C_SBIT != 0 {
		o |= 1 << 20
	}
	if sc&(C_PBIT|C_WBIT) != 0 {
		ctxt.Diag(".nil/.W on dp instruction")
	}
	switch a {
	case AMULU, AMUL:
		return o | 0x0<<21 | 0x9<<4
	case AMULA:
		return o | 0x1<<21 | 0x9<<4
	case AMULLU:
		return o | 0x4<<21 | 0x9<<4
	case AMULL:
		return o | 0x6<<21 | 0x9<<4
	case AMULALU:
		return o | 0x5<<21 | 0x9<<4
	case AMULAL:
		return o | 0x7<<21 | 0x9<<4
	case AAND:
		return o | 0x0<<21
	case AEOR:
		return o | 0x1<<21
	case ASUB:
		return o | 0x2<<21
	case ARSB:
		return o | 0x3<<21
	case AADD:
		return o | 0x4<<21
	case AADC:
		return o | 0x5<<21
	case ASBC:
		return o | 0x6<<21
	case ARSC:
		return o | 0x7<<21
	case ATST:
		return o | 0x8<<21 | 1<<20
	case ATEQ:
		return o | 0x9<<21 | 1<<20
	case ACMP:
		return o | 0xa<<21 | 1<<20
	case ACMN:
		return o | 0xb<<21 | 1<<20
	case AORR:
		return o | 0xc<<21

	case AMOVB, AMOVH, AMOVW:
		return o | 0xd<<21
	case ABIC:
		return o | 0xe<<21
	case AMVN:
		return o | 0xf<<21
	case ASLL:
		return o | 0xd<<21 | 0<<5
	case ASRL:
		return o | 0xd<<21 | 1<<5
	case ASRA:
		return o | 0xd<<21 | 2<<5
	case ASWI:
		return o | 0xf<<24

	case AADDD:
		return o | 0xe<<24 | 0x3<<20 | 0xb<<8 | 0<<4
	case AADDF:
		return o | 0xe<<24 | 0x3<<20 | 0xa<<8 | 0<<4
	case ASUBD:
		return o | 0xe<<24 | 0x3<<20 | 0xb<<8 | 4<<4
	case ASUBF:
		return o | 0xe<<24 | 0x3<<20 | 0xa<<8 | 4<<4
	case AMULD:
		return o | 0xe<<24 | 0x2<<20 | 0xb<<8 | 0<<4
	case AMULF:
		return o | 0xe<<24 | 0x2<<20 | 0xa<<8 | 0<<4
	case ADIVD:
		return o | 0xe<<24 | 0x8<<20 | 0xb<<8 | 0<<4
	case ADIVF:
		return o | 0xe<<24 | 0x8<<20 | 0xa<<8 | 0<<4
	case ASQRTD:
		return o | 0xe<<24 | 0xb<<20 | 1<<16 | 0xb<<8 | 0xc<<4
	case ASQRTF:
		return o | 0xe<<24 | 0xb<<20 | 1<<16 | 0xa<<8 | 0xc<<4
	case AABSD:
		return o | 0xe<<24 | 0xb<<20 | 0<<16 | 0xb<<8 | 0xc<<4
	case AABSF:
		return o | 0xe<<24 | 0xb<<20 | 0<<16 | 0xa<<8 | 0xc<<4
	case ACMPD:
		return o | 0xe<<24 | 0xb<<20 | 4<<16 | 0xb<<8 | 0xc<<4
	case ACMPF:
		return o | 0xe<<24 | 0xb<<20 | 4<<16 | 0xa<<8 | 0xc<<4

	case AMOVF:
		return o | 0xe<<24 | 0xb<<20 | 0<<16 | 0xa<<8 | 4<<4
	case AMOVD:
		return o | 0xe<<24 | 0xb<<20 | 0<<16 | 0xb<<8 | 4<<4

	case AMOVDF:
		return o | 0xe<<24 | 0xb<<20 | 7<<16 | 0xa<<8 | 0xc<<4 | 1<<8 // dtof
	case AMOVFD:
		return o | 0xe<<24 | 0xb<<20 | 7<<16 | 0xa<<8 | 0xc<<4 | 0<<8 // dtof

	case AMOVWF:
		if sc&C_UBIT == 0 {
			o |= 1 << 7 /* signed */
		}
		return o | 0xe<<24 | 0xb<<20 | 8<<16 | 0xa<<8 | 4<<4 | 0<<18 | 0<<8 // toint, double

	case AMOVWD:
		if sc&C_UBIT == 0 {
			o |= 1 << 7 /* signed */
		}
		return o | 0xe<<24 | 0xb<<20 | 8<<16 | 0xa<<8 | 4<<4 | 0<<18 | 1<<8 // toint, double

	case AMOVFW:
		if sc&C_UBIT == 0 {
			o |= 1 << 16 /* signed */
		}
		return o | 0xe<<24 | 0xb<<20 | 8<<16 | 0xa<<8 | 4<<4 | 1<<18 | 0<<8 | 1<<7 // toint, double, trunc

	case AMOVDW:
		if sc&C_UBIT == 0 {
			o |= 1 << 16 /* signed */
		}
		return o | 0xe<<24 | 0xb<<20 | 8<<16 | 0xa<<8 | 4<<4 | 1<<18 | 1<<8 | 1<<7 // toint, double, trunc

	case -AMOVWF: // copy WtoF
		return o | 0xe<<24 | 0x0<<20 | 0xb<<8 | 1<<4

	case -AMOVFW: // copy FtoW
		return o | 0xe<<24 | 0x1<<20 | 0xb<<8 | 1<<4

	case -ACMP: // cmp imm
		return o | 0x3<<24 | 0x5<<20

		// CLZ doesn't support .nil
	case ACLZ:
		return o&(0xf<<28) | 0x16f<<16 | 0xf1<<4

	case AMULWT:
		return o&(0xf<<28) | 0x12<<20 | 0xe<<4

	case AMULWB:
		return o&(0xf<<28) | 0x12<<20 | 0xa<<4

	case AMULAWT:
		return o&(0xf<<28) | 0x12<<20 | 0xc<<4

	case AMULAWB:
		return o&(0xf<<28) | 0x12<<20 | 0x8<<4

	case ABL: // BLX REG
		return o&(0xf<<28) | 0x12fff3<<4
	}

	ctxt.Diag("bad rrr %d", a)
	prasm(ctxt.Curp)
	return 0
}

func opbra(ctxt *obj.Link, p *obj.Prog, a obj.As, sc int) uint32 {
	if sc&(C_SBIT|C_PBIT|C_WBIT) != 0 {
		ctxt.Diag("%v: .nil/.nil/.W on bra instruction", p)
	}
	sc &= C_SCOND
	sc ^= C_SCOND_XOR
	if a == ABL || a == obj.ADUFFZERO || a == obj.ADUFFCOPY {
		return uint32(sc)<<28 | 0x5<<25 | 0x1<<24
	}
	if sc != 0xe {
		ctxt.Diag("%v: .COND on bcond instruction", p)
	}
	switch a {
	case ABEQ:
		return 0x0<<28 | 0x5<<25
	case ABNE:
		return 0x1<<28 | 0x5<<25
	case ABCS:
		return 0x2<<28 | 0x5<<25
	case ABHS:
		return 0x2<<28 | 0x5<<25
	case ABCC:
		return 0x3<<28 | 0x5<<25
	case ABLO:
		return 0x3<<28 | 0x5<<25
	case ABMI:
		return 0x4<<28 | 0x5<<25
	case ABPL:
		return 0x5<<28 | 0x5<<25
	case ABVS:
		return 0x6<<28 | 0x5<<25
	case ABVC:
		return 0x7<<28 | 0x5<<25
	case ABHI:
		return 0x8<<28 | 0x5<<25
	case ABLS:
		return 0x9<<28 | 0x5<<25
	case ABGE:
		return 0xa<<28 | 0x5<<25
	case ABLT:
		return 0xb<<28 | 0x5<<25
	case ABGT:
		return 0xc<<28 | 0x5<<25
	case ABLE:
		return 0xd<<28 | 0x5<<25
	case AB:
		return 0xe<<28 | 0x5<<25
	}

	ctxt.Diag("bad bra %v", obj.Aconv(a))
	prasm(ctxt.Curp)
	return 0
}

func olr(ctxt *obj.Link, v int32, b int, r int, sc int) uint32 {
	if sc&C_SBIT != 0 {
		ctxt.Diag(".nil on LDR/STR instruction")
	}
	o := ((uint32(sc) & C_SCOND) ^ C_SCOND_XOR) << 28
	if sc&C_PBIT == 0 {
		o |= 1 << 24
	}
	if sc&C_UBIT == 0 {
		o |= 1 << 23
	}
	if sc&C_WBIT != 0 {
		o |= 1 << 21
	}
	o |= 1<<26 | 1<<20
	if v < 0 {
		if sc&C_UBIT != 0 {
			ctxt.Diag(".U on neg offset")
		}
		v = -v
		o ^= 1 << 23
	}

	if v >= 1<<12 || v < 0 {
		ctxt.Diag("literal span too large: %d (R%d)\n%v", v, b, ctxt.Printp)
	}
	o |= uint32(v)
	o |= (uint32(b) & 15) << 16
	o |= (uint32(r) & 15) << 12
	return o
}

func olhr(ctxt *obj.Link, v int32, b int, r int, sc int) uint32 {
	if sc&C_SBIT != 0 {
		ctxt.Diag(".nil on LDRH/STRH instruction")
	}
	o := ((uint32(sc) & C_SCOND) ^ C_SCOND_XOR) << 28
	if sc&C_PBIT == 0 {
		o |= 1 << 24
	}
	if sc&C_WBIT != 0 {
		o |= 1 << 21
	}
	o |= 1<<23 | 1<<20 | 0xb<<4
	if v < 0 {
		v = -v
		o ^= 1 << 23
	}

	if v >= 1<<8 || v < 0 {
		ctxt.Diag("literal span too large: %d (R%d)\n%v", v, b, ctxt.Printp)
	}
	o |= uint32(v)&0xf | (uint32(v)>>4)<<8 | 1<<22
	o |= (uint32(b) & 15) << 16
	o |= (uint32(r) & 15) << 12
	return o
}

func osr(ctxt *obj.Link, a obj.As, r int, v int32, b int, sc int) uint32 {
	o := olr(ctxt, v, b, r, sc) ^ (1 << 20)
	if a != AMOVW {
		o |= 1 << 22
	}
	return o
}

func oshr(ctxt *obj.Link, r int, v int32, b int, sc int) uint32 {
	o := olhr(ctxt, v, b, r, sc) ^ (1 << 20)
	return o
}

func osrr(ctxt *obj.Link, r int, i int, b int, sc int) uint32 {
	return olr(ctxt, int32(i), b, r, sc) ^ (1<<25 | 1<<20)
}

func oshrr(ctxt *obj.Link, r int, i int, b int, sc int) uint32 {
	return olhr(ctxt, int32(i), b, r, sc) ^ (1<<22 | 1<<20)
}

func olrr(ctxt *obj.Link, i int, b int, r int, sc int) uint32 {
	return olr(ctxt, int32(i), b, r, sc) ^ (1 << 25)
}

func olhrr(ctxt *obj.Link, i int, b int, r int, sc int) uint32 {
	return olhr(ctxt, int32(i), b, r, sc) ^ (1 << 22)
}

func ofsr(ctxt *obj.Link, a obj.As, r int, v int32, b int, sc int, p *obj.Prog) uint32 {
	if sc&C_SBIT != 0 {
		ctxt.Diag(".nil on FLDR/FSTR instruction: %v", p)
	}
	o := ((uint32(sc) & C_SCOND) ^ C_SCOND_XOR) << 28
	if sc&C_PBIT == 0 {
		o |= 1 << 24
	}
	if sc&C_WBIT != 0 {
		o |= 1 << 21
	}
	o |= 6<<25 | 1<<24 | 1<<23 | 10<<8
	if v < 0 {
		v = -v
		o ^= 1 << 23
	}

	if v&3 != 0 {
		ctxt.Diag("odd offset for floating point op: %d\n%v", v, p)
	} else if v >= 1<<10 || v < 0 {
		ctxt.Diag("literal span too large: %d\n%v", v, p)
	}
	o |= (uint32(v) >> 2) & 0xFF
	o |= (uint32(b) & 15) << 16
	o |= (uint32(r) & 15) << 12

	switch a {
	default:
		ctxt.Diag("bad fst %v", obj.Aconv(a))
		fallthrough

	case AMOVD:
		o |= 1 << 8
		fallthrough

	case AMOVF:
		break
	}

	return o
}

func omvl(ctxt *obj.Link, p *obj.Prog, a *obj.Addr, dr int) uint32 {
	var o1 uint32
	if p.Pcond == nil {
		aclass(ctxt, a)
		v := immrot(^uint32(ctxt.Instoffset))
		if v == 0 {
			ctxt.Diag("missing literal")
			prasm(p)
			return 0
		}

		o1 = oprrr(ctxt, AMVN, int(p.Scond)&C_SCOND)
		o1 |= uint32(v)
		o1 |= (uint32(dr) & 15) << 12
	} else {
		v := int32(p.Pcond.Pc - p.Pc - 8)
		o1 = olr(ctxt, v, REGPC, dr, int(p.Scond)&C_SCOND)
	}

	return o1
}

func chipzero5(ctxt *obj.Link, e float64) int {
	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if ctxt.Goarm < 7 || e != 0 {
		return -1
	}
	return 0
}

func chipfloat5(ctxt *obj.Link, e float64) int {
	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if ctxt.Goarm < 7 {
		return -1
	}

	ei := math.Float64bits(e)
	l := uint32(ei)
	h := uint32(ei >> 32)

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

func nocache(p *obj.Prog) {
	p.Optab = 0
	p.From.Class = 0
	if p.From3 != nil {
		p.From3.Class = 0
	}
	p.To.Class = 0
}
