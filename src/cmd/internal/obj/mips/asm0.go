// cmd/9l/optab.c, cmd/9l/asmout.c from Vita Nuova.
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

package mips

import (
	"cmd/internal/obj"
	"fmt"
	"log"
	"sort"
)

// Instruction layout.

const (
	mips64FuncAlign = 8
)

const (
	r0iszero = 1
)

type Optab struct {
	as    obj.As
	a1    uint8
	a2    uint8
	a3    uint8
	type_ int8
	size  int8
	param int16
	mode  int
}

var optab = []Optab{
	{obj.ATEXT, C_LEXT, C_NONE, C_TEXTSIZE, 0, 0, 0, Mips64},
	{obj.ATEXT, C_ADDR, C_NONE, C_TEXTSIZE, 0, 0, 0, 0},

	{AMOVW, C_REG, C_NONE, C_REG, 1, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_REG, 1, 4, 0, Mips64},
	{AMOVB, C_REG, C_NONE, C_REG, 12, 8, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_REG, 13, 4, 0, 0},
	{AMOVWU, C_REG, C_NONE, C_REG, 14, 8, 0, Mips64},

	{ASUB, C_REG, C_REG, C_REG, 2, 4, 0, 0},
	{ASUBV, C_REG, C_REG, C_REG, 2, 4, 0, Mips64},
	{AADD, C_REG, C_REG, C_REG, 2, 4, 0, 0},
	{AADDV, C_REG, C_REG, C_REG, 2, 4, 0, Mips64},
	{AAND, C_REG, C_REG, C_REG, 2, 4, 0, 0},
	{ASUB, C_REG, C_NONE, C_REG, 2, 4, 0, 0},
	{ASUBV, C_REG, C_NONE, C_REG, 2, 4, 0, Mips64},
	{AADD, C_REG, C_NONE, C_REG, 2, 4, 0, 0},
	{AADDV, C_REG, C_NONE, C_REG, 2, 4, 0, Mips64},
	{AAND, C_REG, C_NONE, C_REG, 2, 4, 0, 0},
	{ACMOVN, C_REG, C_REG, C_REG, 2, 4, 0, 0},

	{ASLL, C_REG, C_NONE, C_REG, 9, 4, 0, 0},
	{ASLL, C_REG, C_REG, C_REG, 9, 4, 0, 0},
	{ASLLV, C_REG, C_NONE, C_REG, 9, 4, 0, Mips64},
	{ASLLV, C_REG, C_REG, C_REG, 9, 4, 0, Mips64},
	{ACLO, C_REG, C_NONE, C_REG, 9, 4, 0, 0},

	{AADDF, C_FREG, C_NONE, C_FREG, 32, 4, 0, 0},
	{AADDF, C_FREG, C_REG, C_FREG, 32, 4, 0, 0},
	{ACMPEQF, C_FREG, C_REG, C_NONE, 32, 4, 0, 0},
	{AABSF, C_FREG, C_NONE, C_FREG, 33, 4, 0, 0},
	{AMOVVF, C_FREG, C_NONE, C_FREG, 33, 4, 0, Mips64},
	{AMOVF, C_FREG, C_NONE, C_FREG, 33, 4, 0, 0},
	{AMOVD, C_FREG, C_NONE, C_FREG, 33, 4, 0, 0},

	{AMOVW, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64},
	{AMOVWU, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64},
	{AMOVV, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64},
	{AMOVB, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64},
	{AMOVBU, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64},
	{AMOVWL, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64},
	{AMOVVL, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64},
	{AMOVW, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, 0},
	{AMOVWU, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, Mips64},
	{AMOVV, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, Mips64},
	{AMOVB, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, 0},
	{AMOVBU, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, 0},
	{AMOVWL, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, 0},
	{AMOVVL, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, Mips64},
	{AMOVW, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, 0},
	{AMOVWU, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, Mips64},
	{AMOVV, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, Mips64},
	{AMOVB, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, 0},
	{AMOVBU, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, 0},
	{AMOVWL, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, 0},
	{AMOVVL, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, Mips64},
	{ASC, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, 0},

	{AMOVW, C_SEXT, C_NONE, C_REG, 8, 4, REGSB, Mips64},
	{AMOVWU, C_SEXT, C_NONE, C_REG, 8, 4, REGSB, Mips64},
	{AMOVV, C_SEXT, C_NONE, C_REG, 8, 4, REGSB, Mips64},
	{AMOVB, C_SEXT, C_NONE, C_REG, 8, 4, REGSB, Mips64},
	{AMOVBU, C_SEXT, C_NONE, C_REG, 8, 4, REGSB, Mips64},
	{AMOVWL, C_SEXT, C_NONE, C_REG, 8, 4, REGSB, Mips64},
	{AMOVVL, C_SEXT, C_NONE, C_REG, 8, 4, REGSB, Mips64},
	{AMOVW, C_SAUTO, C_NONE, C_REG, 8, 4, REGSP, 0},
	{AMOVWU, C_SAUTO, C_NONE, C_REG, 8, 4, REGSP, Mips64},
	{AMOVV, C_SAUTO, C_NONE, C_REG, 8, 4, REGSP, Mips64},
	{AMOVB, C_SAUTO, C_NONE, C_REG, 8, 4, REGSP, 0},
	{AMOVBU, C_SAUTO, C_NONE, C_REG, 8, 4, REGSP, 0},
	{AMOVWL, C_SAUTO, C_NONE, C_REG, 8, 4, REGSP, 0},
	{AMOVVL, C_SAUTO, C_NONE, C_REG, 8, 4, REGSP, Mips64},
	{AMOVW, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, 0},
	{AMOVWU, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, Mips64},
	{AMOVV, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, Mips64},
	{AMOVB, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, 0},
	{AMOVBU, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, 0},
	{AMOVWL, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, 0},
	{AMOVVL, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, Mips64},
	{ALL, C_SOREG, C_NONE, C_REG, 8, 4, REGZERO, 0},

	{AMOVW, C_REG, C_NONE, C_LEXT, 35, 12, REGSB, Mips64},
	{AMOVWU, C_REG, C_NONE, C_LEXT, 35, 12, REGSB, Mips64},
	{AMOVV, C_REG, C_NONE, C_LEXT, 35, 12, REGSB, Mips64},
	{AMOVB, C_REG, C_NONE, C_LEXT, 35, 12, REGSB, Mips64},
	{AMOVBU, C_REG, C_NONE, C_LEXT, 35, 12, REGSB, Mips64},
	{AMOVW, C_REG, C_NONE, C_LAUTO, 35, 12, REGSP, 0},
	{AMOVWU, C_REG, C_NONE, C_LAUTO, 35, 12, REGSP, Mips64},
	{AMOVV, C_REG, C_NONE, C_LAUTO, 35, 12, REGSP, Mips64},
	{AMOVB, C_REG, C_NONE, C_LAUTO, 35, 12, REGSP, 0},
	{AMOVBU, C_REG, C_NONE, C_LAUTO, 35, 12, REGSP, 0},
	{AMOVW, C_REG, C_NONE, C_LOREG, 35, 12, REGZERO, 0},
	{AMOVWU, C_REG, C_NONE, C_LOREG, 35, 12, REGZERO, Mips64},
	{AMOVV, C_REG, C_NONE, C_LOREG, 35, 12, REGZERO, Mips64},
	{AMOVB, C_REG, C_NONE, C_LOREG, 35, 12, REGZERO, 0},
	{AMOVBU, C_REG, C_NONE, C_LOREG, 35, 12, REGZERO, 0},
	{ASC, C_REG, C_NONE, C_LOREG, 35, 12, REGZERO, 0},
	{AMOVW, C_REG, C_NONE, C_ADDR, 50, 8, 0, Mips32},
	{AMOVW, C_REG, C_NONE, C_ADDR, 50, 12, 0, Mips64},
	{AMOVWU, C_REG, C_NONE, C_ADDR, 50, 12, 0, Mips64},
	{AMOVV, C_REG, C_NONE, C_ADDR, 50, 12, 0, Mips64},
	{AMOVB, C_REG, C_NONE, C_ADDR, 50, 8, 0, Mips32},
	{AMOVB, C_REG, C_NONE, C_ADDR, 50, 12, 0, Mips64},
	{AMOVBU, C_REG, C_NONE, C_ADDR, 50, 8, 0, Mips32},
	{AMOVBU, C_REG, C_NONE, C_ADDR, 50, 12, 0, Mips64},
	{AMOVW, C_REG, C_NONE, C_TLS, 53, 8, 0, 0},
	{AMOVWU, C_REG, C_NONE, C_TLS, 53, 8, 0, Mips64},
	{AMOVV, C_REG, C_NONE, C_TLS, 53, 8, 0, Mips64},
	{AMOVB, C_REG, C_NONE, C_TLS, 53, 8, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_TLS, 53, 8, 0, 0},

	{AMOVW, C_LEXT, C_NONE, C_REG, 36, 12, REGSB, Mips64},
	{AMOVWU, C_LEXT, C_NONE, C_REG, 36, 12, REGSB, Mips64},
	{AMOVV, C_LEXT, C_NONE, C_REG, 36, 12, REGSB, Mips64},
	{AMOVB, C_LEXT, C_NONE, C_REG, 36, 12, REGSB, Mips64},
	{AMOVBU, C_LEXT, C_NONE, C_REG, 36, 12, REGSB, Mips64},
	{AMOVW, C_LAUTO, C_NONE, C_REG, 36, 12, REGSP, 0},
	{AMOVWU, C_LAUTO, C_NONE, C_REG, 36, 12, REGSP, Mips64},
	{AMOVV, C_LAUTO, C_NONE, C_REG, 36, 12, REGSP, Mips64},
	{AMOVB, C_LAUTO, C_NONE, C_REG, 36, 12, REGSP, 0},
	{AMOVBU, C_LAUTO, C_NONE, C_REG, 36, 12, REGSP, 0},
	{AMOVW, C_LOREG, C_NONE, C_REG, 36, 12, REGZERO, 0},
	{AMOVWU, C_LOREG, C_NONE, C_REG, 36, 12, REGZERO, Mips64},
	{AMOVV, C_LOREG, C_NONE, C_REG, 36, 12, REGZERO, Mips64},
	{AMOVB, C_LOREG, C_NONE, C_REG, 36, 12, REGZERO, 0},
	{AMOVBU, C_LOREG, C_NONE, C_REG, 36, 12, REGZERO, 0},
	{AMOVW, C_ADDR, C_NONE, C_REG, 51, 8, 0, Mips32},
	{AMOVW, C_ADDR, C_NONE, C_REG, 51, 12, 0, Mips64},
	{AMOVWU, C_ADDR, C_NONE, C_REG, 51, 12, 0, Mips64},
	{AMOVV, C_ADDR, C_NONE, C_REG, 51, 12, 0, Mips64},
	{AMOVB, C_ADDR, C_NONE, C_REG, 51, 8, 0, Mips32},
	{AMOVB, C_ADDR, C_NONE, C_REG, 51, 12, 0, Mips64},
	{AMOVBU, C_ADDR, C_NONE, C_REG, 51, 8, 0, Mips32},
	{AMOVBU, C_ADDR, C_NONE, C_REG, 51, 12, 0, Mips64},
	{AMOVW, C_TLS, C_NONE, C_REG, 54, 8, 0, 0},
	{AMOVWU, C_TLS, C_NONE, C_REG, 54, 8, 0, Mips64},
	{AMOVV, C_TLS, C_NONE, C_REG, 54, 8, 0, Mips64},
	{AMOVB, C_TLS, C_NONE, C_REG, 54, 8, 0, 0},
	{AMOVBU, C_TLS, C_NONE, C_REG, 54, 8, 0, 0},

	{AMOVW, C_SECON, C_NONE, C_REG, 3, 4, REGSB, Mips64},
	{AMOVV, C_SECON, C_NONE, C_REG, 3, 4, REGSB, Mips64},
	{AMOVW, C_SACON, C_NONE, C_REG, 3, 4, REGSP, 0},
	{AMOVV, C_SACON, C_NONE, C_REG, 3, 4, REGSP, Mips64},
	{AMOVW, C_LECON, C_NONE, C_REG, 52, 8, REGSB, Mips32},
	{AMOVW, C_LECON, C_NONE, C_REG, 52, 12, REGSB, Mips64},
	{AMOVV, C_LECON, C_NONE, C_REG, 52, 12, REGSB, Mips64},

	{AMOVW, C_LACON, C_NONE, C_REG, 26, 12, REGSP, 0},
	{AMOVV, C_LACON, C_NONE, C_REG, 26, 12, REGSP, Mips64},
	{AMOVW, C_ADDCON, C_NONE, C_REG, 3, 4, REGZERO, 0},
	{AMOVV, C_ADDCON, C_NONE, C_REG, 3, 4, REGZERO, Mips64},
	{AMOVW, C_ANDCON, C_NONE, C_REG, 3, 4, REGZERO, 0},
	{AMOVV, C_ANDCON, C_NONE, C_REG, 3, 4, REGZERO, Mips64},
	{AMOVW, C_STCON, C_NONE, C_REG, 55, 8, 0, 0},
	{AMOVV, C_STCON, C_NONE, C_REG, 55, 8, 0, Mips64},

	{AMOVW, C_UCON, C_NONE, C_REG, 24, 4, 0, 0},
	{AMOVV, C_UCON, C_NONE, C_REG, 24, 4, 0, Mips64},
	{AMOVW, C_LCON, C_NONE, C_REG, 19, 8, 0, 0},
	{AMOVV, C_LCON, C_NONE, C_REG, 19, 8, 0, Mips64},

	{AMOVW, C_HI, C_NONE, C_REG, 20, 4, 0, 0},
	{AMOVV, C_HI, C_NONE, C_REG, 20, 4, 0, Mips64},
	{AMOVW, C_LO, C_NONE, C_REG, 20, 4, 0, 0},
	{AMOVV, C_LO, C_NONE, C_REG, 20, 4, 0, Mips64},
	{AMOVW, C_REG, C_NONE, C_HI, 21, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_HI, 21, 4, 0, Mips64},
	{AMOVW, C_REG, C_NONE, C_LO, 21, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_LO, 21, 4, 0, Mips64},

	{AMUL, C_REG, C_REG, C_NONE, 22, 4, 0, 0},
	{AMUL, C_REG, C_REG, C_REG, 22, 4, 0, 0},
	{AMULV, C_REG, C_REG, C_NONE, 22, 4, 0, Mips64},

	{AADD, C_ADD0CON, C_REG, C_REG, 4, 4, 0, 0},
	{AADD, C_ADD0CON, C_NONE, C_REG, 4, 4, 0, 0},
	{AADD, C_ANDCON, C_REG, C_REG, 10, 8, 0, 0},
	{AADD, C_ANDCON, C_NONE, C_REG, 10, 8, 0, 0},

	{AADDV, C_ADD0CON, C_REG, C_REG, 4, 4, 0, Mips64},
	{AADDV, C_ADD0CON, C_NONE, C_REG, 4, 4, 0, Mips64},
	{AADDV, C_ANDCON, C_REG, C_REG, 10, 8, 0, Mips64},
	{AADDV, C_ANDCON, C_NONE, C_REG, 10, 8, 0, Mips64},

	{AAND, C_AND0CON, C_REG, C_REG, 4, 4, 0, 0},
	{AAND, C_AND0CON, C_NONE, C_REG, 4, 4, 0, 0},
	{AAND, C_ADDCON, C_REG, C_REG, 10, 8, 0, 0},
	{AAND, C_ADDCON, C_NONE, C_REG, 10, 8, 0, 0},

	{AADD, C_UCON, C_REG, C_REG, 25, 8, 0, 0},
	{AADD, C_UCON, C_NONE, C_REG, 25, 8, 0, 0},
	{AADDV, C_UCON, C_REG, C_REG, 25, 8, 0, Mips64},
	{AADDV, C_UCON, C_NONE, C_REG, 25, 8, 0, Mips64},
	{AAND, C_UCON, C_REG, C_REG, 25, 8, 0, 0},
	{AAND, C_UCON, C_NONE, C_REG, 25, 8, 0, 0},

	{AADD, C_LCON, C_NONE, C_REG, 23, 12, 0, 0},
	{AADDV, C_LCON, C_NONE, C_REG, 23, 12, 0, Mips64},
	{AAND, C_LCON, C_NONE, C_REG, 23, 12, 0, 0},
	{AADD, C_LCON, C_REG, C_REG, 23, 12, 0, 0},
	{AADDV, C_LCON, C_REG, C_REG, 23, 12, 0, Mips64},
	{AAND, C_LCON, C_REG, C_REG, 23, 12, 0, 0},

	{ASLL, C_SCON, C_REG, C_REG, 16, 4, 0, 0},
	{ASLL, C_SCON, C_NONE, C_REG, 16, 4, 0, 0},

	{ASLLV, C_SCON, C_REG, C_REG, 16, 4, 0, Mips64},
	{ASLLV, C_SCON, C_NONE, C_REG, 16, 4, 0, Mips64},

	{ASYSCALL, C_NONE, C_NONE, C_NONE, 5, 4, 0, 0},

	{ABEQ, C_REG, C_REG, C_SBRA, 6, 4, 0, 0},
	{ABEQ, C_REG, C_NONE, C_SBRA, 6, 4, 0, 0},
	{ABLEZ, C_REG, C_NONE, C_SBRA, 6, 4, 0, 0},
	{ABFPT, C_NONE, C_NONE, C_SBRA, 6, 8, 0, 0},

	{AJMP, C_NONE, C_NONE, C_LBRA, 11, 4, 0, 0},
	{AJAL, C_NONE, C_NONE, C_LBRA, 11, 4, 0, 0},

	{AJMP, C_NONE, C_NONE, C_ZOREG, 18, 4, REGZERO, 0},
	{AJAL, C_NONE, C_NONE, C_ZOREG, 18, 4, REGLINK, 0},

	{AMOVW, C_SEXT, C_NONE, C_FREG, 27, 4, REGSB, Mips64},
	{AMOVF, C_SEXT, C_NONE, C_FREG, 27, 4, REGSB, Mips64},
	{AMOVD, C_SEXT, C_NONE, C_FREG, 27, 4, REGSB, Mips64},
	{AMOVW, C_SAUTO, C_NONE, C_FREG, 27, 4, REGSP, Mips64},
	{AMOVF, C_SAUTO, C_NONE, C_FREG, 27, 4, REGSP, 0},
	{AMOVD, C_SAUTO, C_NONE, C_FREG, 27, 4, REGSP, 0},
	{AMOVW, C_SOREG, C_NONE, C_FREG, 27, 4, REGZERO, Mips64},
	{AMOVF, C_SOREG, C_NONE, C_FREG, 27, 4, REGZERO, 0},
	{AMOVD, C_SOREG, C_NONE, C_FREG, 27, 4, REGZERO, 0},

	{AMOVW, C_LEXT, C_NONE, C_FREG, 27, 12, REGSB, Mips64},
	{AMOVF, C_LEXT, C_NONE, C_FREG, 27, 12, REGSB, Mips64},
	{AMOVD, C_LEXT, C_NONE, C_FREG, 27, 12, REGSB, Mips64},
	{AMOVW, C_LAUTO, C_NONE, C_FREG, 27, 12, REGSP, Mips64},
	{AMOVF, C_LAUTO, C_NONE, C_FREG, 27, 12, REGSP, 0},
	{AMOVD, C_LAUTO, C_NONE, C_FREG, 27, 12, REGSP, 0},
	{AMOVW, C_LOREG, C_NONE, C_FREG, 27, 12, REGZERO, Mips64},
	{AMOVF, C_LOREG, C_NONE, C_FREG, 27, 12, REGZERO, 0},
	{AMOVD, C_LOREG, C_NONE, C_FREG, 27, 12, REGZERO, 0},
	{AMOVF, C_ADDR, C_NONE, C_FREG, 51, 8, 0, Mips32},
	{AMOVF, C_ADDR, C_NONE, C_FREG, 51, 12, 0, Mips64},
	{AMOVD, C_ADDR, C_NONE, C_FREG, 51, 8, 0, Mips32},
	{AMOVD, C_ADDR, C_NONE, C_FREG, 51, 12, 0, Mips64},

	{AMOVW, C_FREG, C_NONE, C_SEXT, 28, 4, REGSB, Mips64},
	{AMOVF, C_FREG, C_NONE, C_SEXT, 28, 4, REGSB, Mips64},
	{AMOVD, C_FREG, C_NONE, C_SEXT, 28, 4, REGSB, Mips64},
	{AMOVW, C_FREG, C_NONE, C_SAUTO, 28, 4, REGSP, Mips64},
	{AMOVF, C_FREG, C_NONE, C_SAUTO, 28, 4, REGSP, 0},
	{AMOVD, C_FREG, C_NONE, C_SAUTO, 28, 4, REGSP, 0},
	{AMOVW, C_FREG, C_NONE, C_SOREG, 28, 4, REGZERO, Mips64},
	{AMOVF, C_FREG, C_NONE, C_SOREG, 28, 4, REGZERO, 0},
	{AMOVD, C_FREG, C_NONE, C_SOREG, 28, 4, REGZERO, 0},

	{AMOVW, C_FREG, C_NONE, C_LEXT, 28, 12, REGSB, Mips64},
	{AMOVF, C_FREG, C_NONE, C_LEXT, 28, 12, REGSB, Mips64},
	{AMOVD, C_FREG, C_NONE, C_LEXT, 28, 12, REGSB, Mips64},
	{AMOVW, C_FREG, C_NONE, C_LAUTO, 28, 12, REGSP, Mips64},
	{AMOVF, C_FREG, C_NONE, C_LAUTO, 28, 12, REGSP, 0},
	{AMOVD, C_FREG, C_NONE, C_LAUTO, 28, 12, REGSP, 0},
	{AMOVW, C_FREG, C_NONE, C_LOREG, 28, 12, REGZERO, Mips64},
	{AMOVF, C_FREG, C_NONE, C_LOREG, 28, 12, REGZERO, 0},
	{AMOVD, C_FREG, C_NONE, C_LOREG, 28, 12, REGZERO, 0},
	{AMOVF, C_FREG, C_NONE, C_ADDR, 50, 8, 0, Mips32},
	{AMOVF, C_FREG, C_NONE, C_ADDR, 50, 12, 0, Mips64},
	{AMOVD, C_FREG, C_NONE, C_ADDR, 50, 8, 0, Mips32},
	{AMOVD, C_FREG, C_NONE, C_ADDR, 50, 12, 0, Mips64},

	{AMOVW, C_REG, C_NONE, C_FREG, 30, 4, 0, 0},
	{AMOVW, C_FREG, C_NONE, C_REG, 31, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_FREG, 47, 4, 0, Mips64},
	{AMOVV, C_FREG, C_NONE, C_REG, 48, 4, 0, Mips64},

	{AMOVW, C_ADDCON, C_NONE, C_FREG, 34, 8, 0, Mips64},
	{AMOVW, C_ANDCON, C_NONE, C_FREG, 34, 8, 0, Mips64},

	{AMOVW, C_REG, C_NONE, C_MREG, 37, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_MREG, 37, 4, 0, Mips64},
	{AMOVW, C_MREG, C_NONE, C_REG, 38, 4, 0, 0},
	{AMOVV, C_MREG, C_NONE, C_REG, 38, 4, 0, Mips64},

	{AWORD, C_LCON, C_NONE, C_NONE, 40, 4, 0, 0},

	{AMOVW, C_REG, C_NONE, C_FCREG, 41, 8, 0, 0},
	{AMOVV, C_REG, C_NONE, C_FCREG, 41, 8, 0, Mips64},
	{AMOVW, C_FCREG, C_NONE, C_REG, 42, 4, 0, 0},
	{AMOVV, C_FCREG, C_NONE, C_REG, 42, 4, 0, Mips64},

	{ATEQ, C_SCON, C_REG, C_REG, 15, 4, 0, 0},
	{ATEQ, C_SCON, C_NONE, C_REG, 15, 4, 0, 0},
	{ACMOVT, C_REG, C_NONE, C_REG, 17, 4, 0, 0},

	{ABREAK, C_REG, C_NONE, C_SEXT, 7, 4, REGSB, Mips64}, /* really CACHE instruction */
	{ABREAK, C_REG, C_NONE, C_SAUTO, 7, 4, REGSP, Mips64},
	{ABREAK, C_REG, C_NONE, C_SOREG, 7, 4, REGZERO, Mips64},
	{ABREAK, C_NONE, C_NONE, C_NONE, 5, 4, 0, 0},

	{obj.AUNDEF, C_NONE, C_NONE, C_NONE, 49, 4, 0, 0},
	{obj.AUSEFIELD, C_ADDR, C_NONE, C_NONE, 0, 0, 0, 0},
	{obj.APCDATA, C_LCON, C_NONE, C_LCON, 0, 0, 0, 0},
	{obj.AFUNCDATA, C_SCON, C_NONE, C_ADDR, 0, 0, 0, 0},
	{obj.ANOP, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0},
	{obj.ADUFFZERO, C_NONE, C_NONE, C_LBRA, 11, 4, 0, 0}, // same as AJMP
	{obj.ADUFFCOPY, C_NONE, C_NONE, C_LBRA, 11, 4, 0, 0}, // same as AJMP

	{obj.AXXX, C_NONE, C_NONE, C_NONE, 0, 4, 0, 0},
}

var oprange [ALAST & obj.AMask][]Optab

var xcmp [C_NCLASS][C_NCLASS]bool

func span0(ctxt *obj.Link, cursym *obj.LSym) {
	p := cursym.Text
	if p == nil || p.Link == nil { // handle external functions and ELF section symbols
		return
	}
	ctxt.Cursym = cursym
	ctxt.Autosize = int32(p.To.Offset + ctxt.FixedFrameSize())

	if oprange[AOR&obj.AMask] == nil {
		buildop(ctxt)
	}

	c := int64(0)
	p.Pc = c

	var m int
	var o *Optab
	for p = p.Link; p != nil; p = p.Link {
		ctxt.Curp = p
		p.Pc = c
		o = oplook(ctxt, p)
		m = int(o.size)
		if m == 0 {
			if p.As != obj.ANOP && p.As != obj.AFUNCDATA && p.As != obj.APCDATA && p.As != obj.AUSEFIELD {
				ctxt.Diag("zero-width instruction\n%v", p)
			}
			continue
		}

		c += int64(m)
	}

	cursym.Size = c

	/*
	 * if any procedure is large enough to
	 * generate a large SBRA branch, then
	 * generate extra passes putting branches
	 * around jmps to fix. this is rare.
	 */
	bflag := 1

	var otxt int64
	var q *obj.Prog
	for bflag != 0 {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f span1\n", obj.Cputime())
		}
		bflag = 0
		c = 0
		for p = cursym.Text.Link; p != nil; p = p.Link {
			p.Pc = c
			o = oplook(ctxt, p)

			// very large conditional branches
			if o.type_ == 6 && p.Pcond != nil {
				otxt = p.Pcond.Pc - c
				if otxt < -(1<<17)+10 || otxt >= (1<<17)-10 {
					q = ctxt.NewProg()
					q.Link = p.Link
					p.Link = q
					q.As = AJMP
					q.Lineno = p.Lineno
					q.To.Type = obj.TYPE_BRANCH
					q.Pcond = p.Pcond
					p.Pcond = q
					q = ctxt.NewProg()
					q.Link = p.Link
					p.Link = q
					q.As = AJMP
					q.Lineno = p.Lineno
					q.To.Type = obj.TYPE_BRANCH
					q.Pcond = q.Link.Link

					addnop(ctxt, p.Link)
					addnop(ctxt, p)
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

		cursym.Size = c
	}
	if ctxt.Mode&Mips64 != 0 {
		c += -c & (mips64FuncAlign - 1)
	}
	cursym.Size = c

	/*
	 * lay out the code, emitting code and data relocations.
	 */

	cursym.Grow(cursym.Size)

	bp := cursym.P
	var i int32
	var out [4]uint32
	for p := cursym.Text.Link; p != nil; p = p.Link {
		ctxt.Pc = p.Pc
		ctxt.Curp = p
		o = oplook(ctxt, p)
		if int(o.size) > 4*len(out) {
			log.Fatalf("out array in span0 is too small, need at least %d for %v", o.size/4, p)
		}
		asmout(ctxt, p, o, out[:])
		for i = 0; i < int32(o.size/4); i++ {
			ctxt.Arch.ByteOrder.PutUint32(bp, out[i])
			bp = bp[4:]
		}
	}
}

func isint32(v int64) bool {
	return int64(int32(v)) == v
}

func isuint32(v uint64) bool {
	return uint64(uint32(v)) == v
}

func aclass(ctxt *obj.Link, a *obj.Addr) int {
	switch a.Type {
	case obj.TYPE_NONE:
		return C_NONE

	case obj.TYPE_REG:
		if REG_R0 <= a.Reg && a.Reg <= REG_R31 {
			return C_REG
		}
		if REG_F0 <= a.Reg && a.Reg <= REG_F31 {
			return C_FREG
		}
		if REG_M0 <= a.Reg && a.Reg <= REG_M31 {
			return C_MREG
		}
		if REG_FCR0 <= a.Reg && a.Reg <= REG_FCR31 {
			return C_FCREG
		}
		if a.Reg == REG_LO {
			return C_LO
		}
		if a.Reg == REG_HI {
			return C_HI
		}
		return C_GOK

	case obj.TYPE_MEM:
		switch a.Name {
		case obj.NAME_EXTERN,
			obj.NAME_STATIC:
			if a.Sym == nil {
				break
			}
			ctxt.Instoffset = a.Offset
			if a.Sym != nil { // use relocation
				if a.Sym.Type == obj.STLSBSS {
					return C_TLS
				}
				return C_ADDR
			}
			return C_LEXT

		case obj.NAME_AUTO:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset
			if ctxt.Instoffset >= -BIG && ctxt.Instoffset < BIG {
				return C_SAUTO
			}
			return C_LAUTO

		case obj.NAME_PARAM:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset + ctxt.FixedFrameSize()
			if ctxt.Instoffset >= -BIG && ctxt.Instoffset < BIG {
				return C_SAUTO
			}
			return C_LAUTO

		case obj.NAME_NONE:
			ctxt.Instoffset = a.Offset
			if ctxt.Instoffset == 0 {
				return C_ZOREG
			}
			if ctxt.Instoffset >= -BIG && ctxt.Instoffset < BIG {
				return C_SOREG
			}
			return C_LOREG
		}

		return C_GOK

	case obj.TYPE_TEXTSIZE:
		return C_TEXTSIZE

	case obj.TYPE_CONST,
		obj.TYPE_ADDR:
		switch a.Name {
		case obj.NAME_NONE:
			ctxt.Instoffset = a.Offset
			if a.Reg != 0 {
				if -BIG <= ctxt.Instoffset && ctxt.Instoffset <= BIG {
					return C_SACON
				}
				if isint32(ctxt.Instoffset) {
					return C_LACON
				}
				return C_DACON
			}

			goto consize

		case obj.NAME_EXTERN,
			obj.NAME_STATIC:
			s := a.Sym
			if s == nil {
				break
			}
			if s.Type == obj.SCONST {
				ctxt.Instoffset = a.Offset
				goto consize
			}

			ctxt.Instoffset = a.Offset
			if s.Type == obj.STLSBSS {
				return C_STCON // address of TLS variable
			}
			return C_LECON

		case obj.NAME_AUTO:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset
			if ctxt.Instoffset >= -BIG && ctxt.Instoffset < BIG {
				return C_SACON
			}
			return C_LACON

		case obj.NAME_PARAM:
			ctxt.Instoffset = int64(ctxt.Autosize) + a.Offset + ctxt.FixedFrameSize()
			if ctxt.Instoffset >= -BIG && ctxt.Instoffset < BIG {
				return C_SACON
			}
			return C_LACON
		}

		return C_GOK

	consize:
		if ctxt.Instoffset >= 0 {
			if ctxt.Instoffset == 0 {
				return C_ZCON
			}
			if ctxt.Instoffset <= 0x7fff {
				return C_SCON
			}
			if ctxt.Instoffset <= 0xffff {
				return C_ANDCON
			}
			if ctxt.Instoffset&0xffff == 0 && isuint32(uint64(ctxt.Instoffset)) { /* && (instoffset & (1<<31)) == 0) */
				return C_UCON
			}
			if isint32(ctxt.Instoffset) || isuint32(uint64(ctxt.Instoffset)) {
				return C_LCON
			}
			return C_LCON // C_DCON
		}

		if ctxt.Instoffset >= -0x8000 {
			return C_ADDCON
		}
		if ctxt.Instoffset&0xffff == 0 && isint32(ctxt.Instoffset) {
			return C_UCON
		}
		if isint32(ctxt.Instoffset) {
			return C_LCON
		}
		return C_LCON // C_DCON

	case obj.TYPE_BRANCH:
		return C_SBRA
	}

	return C_GOK
}

func prasm(p *obj.Prog) {
	fmt.Printf("%v\n", p)
}

func oplook(ctxt *obj.Link, p *obj.Prog) *Optab {
	if oprange[AOR&obj.AMask] == nil {
		buildop(ctxt)
	}

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

	//print("oplook %P %d %d %d\n", p, a1, a2, a3);

	ops := oprange[p.As&obj.AMask]
	c1 := &xcmp[a1]
	c3 := &xcmp[a3]
	for i := range ops {
		op := &ops[i]
		if int(op.a2) == a2 && c1[op.a1] && c3[op.a3] && (ctxt.Mode&op.mode == op.mode) {
			p.Optab = uint16(cap(optab) - cap(ops) + i + 1)
			return op
		}
	}

	ctxt.Diag("illegal combination %v %v %v %v", p.As, DRconv(a1), DRconv(a2), DRconv(a3))
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
		if b == C_ZCON || b == C_SCON || b == C_UCON || b == C_ADDCON || b == C_ANDCON {
			return true
		}

	case C_ADD0CON:
		if b == C_ADDCON {
			return true
		}
		fallthrough

	case C_ADDCON:
		if b == C_ZCON || b == C_SCON {
			return true
		}

	case C_AND0CON:
		if b == C_ANDCON {
			return true
		}
		fallthrough

	case C_ANDCON:
		if b == C_ZCON || b == C_SCON {
			return true
		}

	case C_UCON:
		if b == C_ZCON {
			return true
		}

	case C_SCON:
		if b == C_ZCON {
			return true
		}

	case C_LACON:
		if b == C_SACON {
			return true
		}

	case C_LBRA:
		if b == C_SBRA {
			return true
		}

	case C_LEXT:
		if b == C_SEXT {
			return true
		}

	case C_LAUTO:
		if b == C_SAUTO {
			return true
		}

	case C_REG:
		if b == C_ZCON {
			return r0iszero != 0 /*TypeKind(100016)*/
		}

	case C_LOREG:
		if b == C_ZOREG || b == C_SOREG {
			return true
		}

	case C_SOREG:
		if b == C_ZOREG {
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

	for i := 0; i < C_NCLASS; i++ {
		for n = 0; n < C_NCLASS; n++ {
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
		r0 := r & obj.AMask
		start := i
		for optab[i].as == r {
			i++
		}
		oprange[r0] = optab[start:i]
		i--

		switch r {
		default:
			ctxt.Diag("unknown op in build: %v", r)
			log.Fatalf("bad code")

		case AABSF:
			opset(AMOVFD, r0)
			opset(AMOVDF, r0)
			opset(AMOVWF, r0)
			opset(AMOVFW, r0)
			opset(AMOVWD, r0)
			opset(AMOVDW, r0)
			opset(ANEGF, r0)
			opset(ANEGD, r0)
			opset(AABSD, r0)
			opset(ATRUNCDW, r0)
			opset(ATRUNCFW, r0)
			opset(ASQRTF, r0)
			opset(ASQRTD, r0)

		case AMOVVF:
			opset(AMOVVD, r0)
			opset(AMOVFV, r0)
			opset(AMOVDV, r0)
			opset(ATRUNCDV, r0)
			opset(ATRUNCFV, r0)

		case AADD:
			opset(ASGT, r0)
			opset(ASGTU, r0)
			opset(AADDU, r0)

		case AADDV:
			opset(AADDVU, r0)

		case AADDF:
			opset(ADIVF, r0)
			opset(ADIVD, r0)
			opset(AMULF, r0)
			opset(AMULD, r0)
			opset(ASUBF, r0)
			opset(ASUBD, r0)
			opset(AADDD, r0)

		case AAND:
			opset(AOR, r0)
			opset(AXOR, r0)

		case ABEQ:
			opset(ABNE, r0)

		case ABLEZ:
			opset(ABGEZ, r0)
			opset(ABGEZAL, r0)
			opset(ABLTZ, r0)
			opset(ABLTZAL, r0)
			opset(ABGTZ, r0)

		case AMOVB:
			opset(AMOVH, r0)

		case AMOVBU:
			opset(AMOVHU, r0)

		case AMUL:
			opset(AREM, r0)
			opset(AREMU, r0)
			opset(ADIVU, r0)
			opset(AMULU, r0)
			opset(ADIV, r0)

		case AMULV:
			opset(ADIVV, r0)
			opset(ADIVVU, r0)
			opset(AMULVU, r0)
			opset(AREMV, r0)
			opset(AREMVU, r0)

		case ASLL:
			opset(ASRL, r0)
			opset(ASRA, r0)

		case ASLLV:
			opset(ASRAV, r0)
			opset(ASRLV, r0)

		case ASUB:
			opset(ASUBU, r0)
			opset(ANOR, r0)

		case ASUBV:
			opset(ASUBVU, r0)

		case ASYSCALL:
			opset(ASYNC, r0)
			opset(ATLBP, r0)
			opset(ATLBR, r0)
			opset(ATLBWI, r0)
			opset(ATLBWR, r0)

		case ACMPEQF:
			opset(ACMPGTF, r0)
			opset(ACMPGTD, r0)
			opset(ACMPGEF, r0)
			opset(ACMPGED, r0)
			opset(ACMPEQD, r0)

		case ABFPT:
			opset(ABFPF, r0)

		case AMOVWL:
			opset(AMOVWR, r0)

		case AMOVVL:
			opset(AMOVVR, r0)

		case AMOVW,
			AMOVD,
			AMOVF,
			AMOVV,
			ABREAK,
			ARFE,
			AJAL,
			AJMP,
			AMOVWU,
			ALL,
			ASC,
			AWORD,
			obj.ANOP,
			obj.ATEXT,
			obj.AUNDEF,
			obj.AUSEFIELD,
			obj.AFUNCDATA,
			obj.APCDATA,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			break

		case ACMOVN:
			opset(ACMOVZ, r0)

		case ACMOVT:
			opset(ACMOVF, r0)

		case ACLO:
			opset(ACLZ, r0)

		case ATEQ:
			opset(ATNE, r0)
		}
	}
}

func OP(x uint32, y uint32) uint32 {
	return x<<3 | y<<0
}

func SP(x uint32, y uint32) uint32 {
	return x<<29 | y<<26
}

func BCOND(x uint32, y uint32) uint32 {
	return x<<19 | y<<16
}

func MMU(x uint32, y uint32) uint32 {
	return SP(2, 0) | 16<<21 | x<<3 | y<<0
}

func FPF(x uint32, y uint32) uint32 {
	return SP(2, 1) | 16<<21 | x<<3 | y<<0
}

func FPD(x uint32, y uint32) uint32 {
	return SP(2, 1) | 17<<21 | x<<3 | y<<0
}

func FPW(x uint32, y uint32) uint32 {
	return SP(2, 1) | 20<<21 | x<<3 | y<<0
}

func FPV(x uint32, y uint32) uint32 {
	return SP(2, 1) | 21<<21 | x<<3 | y<<0
}

func OP_RRR(op uint32, r1 uint32, r2 uint32, r3 uint32) uint32 {
	return op | (r1&31)<<16 | (r2&31)<<21 | (r3&31)<<11
}

func OP_IRR(op uint32, i uint32, r2 uint32, r3 uint32) uint32 {
	return op | i&0xFFFF | (r2&31)<<21 | (r3&31)<<16
}

func OP_SRR(op uint32, s uint32, r2 uint32, r3 uint32) uint32 {
	return op | (s&31)<<6 | (r2&31)<<16 | (r3&31)<<11
}

func OP_FRRR(op uint32, r1 uint32, r2 uint32, r3 uint32) uint32 {
	return op | (r1&31)<<16 | (r2&31)<<11 | (r3&31)<<6
}

func OP_JMP(op uint32, i uint32) uint32 {
	return op | i&0x3FFFFFF
}

func asmout(ctxt *obj.Link, p *obj.Prog, o *Optab, out []uint32) {
	o1 := uint32(0)
	o2 := uint32(0)
	o3 := uint32(0)
	o4 := uint32(0)

	add := AADDU

	if ctxt.Mode&Mips64 != 0 {
		add = AADDVU
	}
	switch o.type_ {
	default:
		ctxt.Diag("unknown type %d %v", o.type_)
		prasm(p)

	case 0: /* pseudo ops */
		break

	case 1: /* mov r1,r2 ==> OR r1,r0,r2 */
		a := AOR
		if p.As == AMOVW && ctxt.Mode&Mips64 != 0 {
			a = AADDU // sign-extended to high 32 bits
		}
		o1 = OP_RRR(oprrr(ctxt, a), uint32(REGZERO), uint32(p.From.Reg), uint32(p.To.Reg))

	case 2: /* add/sub r1,[r2],r3 */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = OP_RRR(oprrr(ctxt, p.As), uint32(p.From.Reg), uint32(r), uint32(p.To.Reg))

	case 3: /* mov $soreg, r ==> or/add $i,o,r */
		v := regoff(ctxt, &p.From)

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		a := add
		if o.a1 == C_ANDCON {
			a = AOR
		}

		o1 = OP_IRR(opirr(ctxt, a), uint32(v), uint32(r), uint32(p.To.Reg))

	case 4: /* add $scon,[r1],r2 */
		v := regoff(ctxt, &p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}

		o1 = OP_IRR(opirr(ctxt, p.As), uint32(v), uint32(r), uint32(p.To.Reg))

	case 5: /* syscall */
		o1 = oprrr(ctxt, p.As)

	case 6: /* beq r1,[r2],sbra */
		v := int32(0)
		if p.Pcond == nil {
			v = int32(-4) >> 2
		} else {
			v = int32(p.Pcond.Pc-p.Pc-4) >> 2
		}
		if (v<<16)>>16 != v {
			ctxt.Diag("short branch too far\n%v", p)
		}
		o1 = OP_IRR(opirr(ctxt, p.As), uint32(v), uint32(p.From.Reg), uint32(p.Reg))
		// for ABFPT and ABFPF only: always fill delay slot with 0
		// see comments in func preprocess for details.
		o2 = 0

	case 7: /* mov r, soreg ==> sw o(r) */
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		v := regoff(ctxt, &p.To)
		o1 = OP_IRR(opirr(ctxt, p.As), uint32(v), uint32(r), uint32(p.From.Reg))

	case 8: /* mov soreg, r ==> lw o(r) */
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		v := regoff(ctxt, &p.From)
		o1 = OP_IRR(opirr(ctxt, -p.As), uint32(v), uint32(r), uint32(p.To.Reg))

	case 9: /* sll r1,[r2],r3 */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = OP_RRR(oprrr(ctxt, p.As), uint32(r), uint32(p.From.Reg), uint32(p.To.Reg))

	case 10: /* add $con,[r1],r2 ==> mov $con, t; add t,[r1],r2 */
		v := regoff(ctxt, &p.From)
		a := AOR
		if v < 0 {
			a = AADDU
		}
		o1 = OP_IRR(opirr(ctxt, a), uint32(v), uint32(0), uint32(REGTMP))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o2 = OP_RRR(oprrr(ctxt, p.As), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 11: /* jmp lbra */
		v := int32(0)
		if aclass(ctxt, &p.To) == C_SBRA && p.To.Sym == nil && p.As == AJMP {
			// use PC-relative branch for short branches
			// BEQ	R0, R0, sbra
			if p.Pcond == nil {
				v = int32(-4) >> 2
			} else {
				v = int32(p.Pcond.Pc-p.Pc-4) >> 2
			}
			if (v<<16)>>16 == v {
				o1 = OP_IRR(opirr(ctxt, ABEQ), uint32(v), uint32(REGZERO), uint32(REGZERO))
				break
			}
		}
		if p.Pcond == nil {
			v = int32(p.Pc) >> 2
		} else {
			v = int32(p.Pcond.Pc) >> 2
		}
		o1 = OP_JMP(opirr(ctxt, p.As), uint32(v))
		if p.To.Sym == nil {
			p.To.Sym = ctxt.Cursym.Text.From.Sym
			p.To.Offset = p.Pcond.Pc
		}
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.To.Sym
		rel.Add = p.To.Offset
		if p.As == AJAL {
			rel.Type = obj.R_CALLMIPS
		} else {
			rel.Type = obj.R_JMPMIPS
		}

	case 12: /* movbs r,r */
		v := 16
		if p.As == AMOVB {
			v = 24
		}
		o1 = OP_SRR(opirr(ctxt, ASLL), uint32(v), uint32(p.From.Reg), uint32(p.To.Reg))
		o2 = OP_SRR(opirr(ctxt, ASRA), uint32(v), uint32(p.To.Reg), uint32(p.To.Reg))

	case 13: /* movbu r,r */
		if p.As == AMOVBU {
			o1 = OP_IRR(opirr(ctxt, AAND), uint32(0xff), uint32(p.From.Reg), uint32(p.To.Reg))
		} else {
			o1 = OP_IRR(opirr(ctxt, AAND), uint32(0xffff), uint32(p.From.Reg), uint32(p.To.Reg))
		}

	case 14: /* movwu r,r */
		o1 = OP_SRR(opirr(ctxt, -ASLLV), uint32(0), uint32(p.From.Reg), uint32(p.To.Reg))
		o2 = OP_SRR(opirr(ctxt, -ASRLV), uint32(0), uint32(p.To.Reg), uint32(p.To.Reg))

	case 15: /* teq $c r,r */
		v := regoff(ctxt, &p.From)
		r := int(p.Reg)
		if r == 0 {
			r = REGZERO
		}
		/* only use 10 bits of trap code */
		o1 = OP_IRR(opirr(ctxt, p.As), (uint32(v)&0x3FF)<<6, uint32(p.Reg), uint32(p.To.Reg))

	case 16: /* sll $c,[r1],r2 */
		v := regoff(ctxt, &p.From)
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}

		/* OP_SRR will use only the low 5 bits of the shift value */
		if v >= 32 && vshift(p.As) {
			o1 = OP_SRR(opirr(ctxt, -p.As), uint32(v-32), uint32(r), uint32(p.To.Reg))
		} else {
			o1 = OP_SRR(opirr(ctxt, p.As), uint32(v), uint32(r), uint32(p.To.Reg))
		}

	case 17:
		o1 = OP_RRR(oprrr(ctxt, p.As), uint32(REGZERO), uint32(p.From.Reg), uint32(p.To.Reg))

	case 18: /* jmp [r1],0(r2) */
		r := int(p.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = OP_RRR(oprrr(ctxt, p.As), uint32(0), uint32(p.To.Reg), uint32(r))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 0
		rel.Type = obj.R_CALLIND

	case 19: /* mov $lcon,r ==> lu+or */
		v := regoff(ctxt, &p.From)
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(v>>16), uint32(REGZERO), uint32(p.To.Reg))
		o2 = OP_IRR(opirr(ctxt, AOR), uint32(v), uint32(p.To.Reg), uint32(p.To.Reg))

	case 20: /* mov lo/hi,r */
		a := OP(2, 0) /* mfhi */
		if p.From.Reg == REG_LO {
			a = OP(2, 2) /* mflo */
		}
		o1 = OP_RRR(a, uint32(REGZERO), uint32(REGZERO), uint32(p.To.Reg))

	case 21: /* mov r,lo/hi */
		a := OP(2, 1) /* mthi */
		if p.To.Reg == REG_LO {
			a = OP(2, 3) /* mtlo */
		}
		o1 = OP_RRR(a, uint32(REGZERO), uint32(p.From.Reg), uint32(REGZERO))

	case 22: /* mul r1,r2 [r3]*/
		if p.To.Reg != 0 {
			r := int(p.Reg)
			if r == 0 {
				r = int(p.To.Reg)
			}
			a := SP(3, 4) | 2 /* mul */
			o1 = OP_RRR(a, uint32(p.From.Reg), uint32(r), uint32(p.To.Reg))
		} else {
			o1 = OP_RRR(oprrr(ctxt, p.As), uint32(p.From.Reg), uint32(p.Reg), uint32(REGZERO))
		}

	case 23: /* add $lcon,r1,r2 ==> lu+or+add */
		v := regoff(ctxt, &p.From)
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(v>>16), uint32(REGZERO), uint32(REGTMP))
		o2 = OP_IRR(opirr(ctxt, AOR), uint32(v), uint32(REGTMP), uint32(REGTMP))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o3 = OP_RRR(oprrr(ctxt, p.As), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 24: /* mov $ucon,r ==> lu r */
		v := regoff(ctxt, &p.From)
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(v>>16), uint32(REGZERO), uint32(p.To.Reg))

	case 25: /* add/and $ucon,[r1],r2 ==> lu $con,t; add t,[r1],r2 */
		v := regoff(ctxt, &p.From)
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(v>>16), uint32(REGZERO), uint32(REGTMP))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o2 = OP_RRR(oprrr(ctxt, p.As), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 26: /* mov $lsext/auto/oreg,r ==> lu+or+add */
		v := regoff(ctxt, &p.From)
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(v>>16), uint32(REGZERO), uint32(REGTMP))
		o2 = OP_IRR(opirr(ctxt, AOR), uint32(v), uint32(REGTMP), uint32(REGTMP))
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o3 = OP_RRR(oprrr(ctxt, add), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 27: /* mov [sl]ext/auto/oreg,fr ==> lwc1 o(r) */
		v := regoff(ctxt, &p.From)
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		a := -AMOVF
		if p.As == AMOVD {
			a = -AMOVD
		}
		switch o.size {
		case 12:
			o1 = OP_IRR(opirr(ctxt, ALUI), uint32((v+1<<15)>>16), uint32(REGZERO), uint32(REGTMP))
			o2 = OP_RRR(oprrr(ctxt, add), uint32(r), uint32(REGTMP), uint32(REGTMP))
			o3 = OP_IRR(opirr(ctxt, a), uint32(v), uint32(REGTMP), uint32(p.To.Reg))

		case 4:
			o1 = OP_IRR(opirr(ctxt, a), uint32(v), uint32(r), uint32(p.To.Reg))
		}

	case 28: /* mov fr,[sl]ext/auto/oreg ==> swc1 o(r) */
		v := regoff(ctxt, &p.To)
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		a := AMOVF
		if p.As == AMOVD {
			a = AMOVD
		}
		switch o.size {
		case 12:
			o1 = OP_IRR(opirr(ctxt, ALUI), uint32((v+1<<15)>>16), uint32(REGZERO), uint32(REGTMP))
			o2 = OP_RRR(oprrr(ctxt, add), uint32(r), uint32(REGTMP), uint32(REGTMP))
			o3 = OP_IRR(opirr(ctxt, a), uint32(v), uint32(REGTMP), uint32(p.From.Reg))

		case 4:
			o1 = OP_IRR(opirr(ctxt, a), uint32(v), uint32(r), uint32(p.From.Reg))
		}

	case 30: /* movw r,fr */
		a := SP(2, 1) | (4 << 21) /* mtc1 */
		o1 = OP_RRR(a, uint32(p.From.Reg), uint32(0), uint32(p.To.Reg))

	case 31: /* movw fr,r */
		a := SP(2, 1) | (0 << 21) /* mtc1 */
		o1 = OP_RRR(a, uint32(p.To.Reg), uint32(0), uint32(p.From.Reg))

	case 32: /* fadd fr1,[fr2],fr3 */
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = OP_FRRR(oprrr(ctxt, p.As), uint32(p.From.Reg), uint32(r), uint32(p.To.Reg))

	case 33: /* fabs fr1, fr3 */
		o1 = OP_FRRR(oprrr(ctxt, p.As), uint32(0), uint32(p.From.Reg), uint32(p.To.Reg))

	case 34: /* mov $con,fr ==> or/add $i,t; mov t,fr */
		v := regoff(ctxt, &p.From)
		a := AADDU
		if o.a1 == C_ANDCON {
			a = AOR
		}
		o1 = OP_IRR(opirr(ctxt, a), uint32(v), uint32(0), uint32(REGTMP))
		o2 = OP_RRR(SP(2, 1)|(4<<21), uint32(REGTMP), uint32(0), uint32(p.To.Reg)) /* mtc1 */

	case 35: /* mov r,lext/auto/oreg ==> sw o(REGTMP) */
		v := regoff(ctxt, &p.To)
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32((v+1<<15)>>16), uint32(REGZERO), uint32(REGTMP))
		o2 = OP_RRR(oprrr(ctxt, add), uint32(r), uint32(REGTMP), uint32(REGTMP))
		o3 = OP_IRR(opirr(ctxt, p.As), uint32(v), uint32(REGTMP), uint32(p.From.Reg))

	case 36: /* mov lext/auto/oreg,r ==> lw o(REGTMP) */
		v := regoff(ctxt, &p.From)
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32((v+1<<15)>>16), uint32(REGZERO), uint32(REGTMP))
		o2 = OP_RRR(oprrr(ctxt, add), uint32(r), uint32(REGTMP), uint32(REGTMP))
		o3 = OP_IRR(opirr(ctxt, -p.As), uint32(v), uint32(REGTMP), uint32(p.To.Reg))

	case 37: /* movw r,mr */
		a := SP(2, 0) | (4 << 21) /* mtc0 */
		if p.As == AMOVV {
			a = SP(2, 0) | (5 << 21) /* dmtc0 */
		}
		o1 = OP_RRR(a, uint32(p.From.Reg), uint32(0), uint32(p.To.Reg))

	case 38: /* movw mr,r */
		a := SP(2, 0) | (0 << 21) /* mfc0 */
		if p.As == AMOVV {
			a = SP(2, 0) | (1 << 21) /* dmfc0 */
		}
		o1 = OP_RRR(a, uint32(p.To.Reg), uint32(0), uint32(p.From.Reg))

	case 40: /* word */
		o1 = uint32(regoff(ctxt, &p.From))

	case 41: /* movw f,fcr */
		o1 = OP_RRR(SP(2, 1)|(2<<21), uint32(REGZERO), uint32(0), uint32(p.To.Reg))    /* mfcc1 */
		o2 = OP_RRR(SP(2, 1)|(6<<21), uint32(p.From.Reg), uint32(0), uint32(p.To.Reg)) /* mtcc1 */

	case 42: /* movw fcr,r */
		o1 = OP_RRR(SP(2, 1)|(2<<21), uint32(p.To.Reg), uint32(0), uint32(p.From.Reg)) /* mfcc1 */

	case 47: /* movv r,fr */
		a := SP(2, 1) | (5 << 21) /* dmtc1 */
		o1 = OP_RRR(a, uint32(p.From.Reg), uint32(0), uint32(p.To.Reg))

	case 48: /* movv fr,r */
		a := SP(2, 1) | (1 << 21) /* dmtc1 */
		o1 = OP_RRR(a, uint32(p.To.Reg), uint32(0), uint32(p.From.Reg))

	case 49: /* undef */
		o1 = 52 /* trap -- teq r0, r0 */

	/* relocation operations */
	case 50: /* mov r,addr ==> lu + add REGSB, REGTMP + sw o(REGTMP) */
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(0), uint32(REGZERO), uint32(REGTMP))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.To.Sym
		rel.Add = p.To.Offset
		rel.Type = obj.R_ADDRMIPSU
		o2 = OP_IRR(opirr(ctxt, p.As), uint32(0), uint32(REGTMP), uint32(p.From.Reg))
		rel2 := obj.Addrel(ctxt.Cursym)
		rel2.Off = int32(ctxt.Pc + 4)
		rel2.Siz = 4
		rel2.Sym = p.To.Sym
		rel2.Add = p.To.Offset
		rel2.Type = obj.R_ADDRMIPS

		if o.size == 12 {
			o3 = o2
			o2 = OP_RRR(oprrr(ctxt, AADDVU), uint32(REGSB), uint32(REGTMP), uint32(REGTMP))
			rel2.Off += 4
		}

	case 51: /* mov addr,r ==> lu + add REGSB, REGTMP + lw o(REGTMP) */
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(0), uint32(REGZERO), uint32(REGTMP))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.From.Sym
		rel.Add = p.From.Offset
		rel.Type = obj.R_ADDRMIPSU
		o2 = OP_IRR(opirr(ctxt, -p.As), uint32(0), uint32(REGTMP), uint32(p.To.Reg))
		rel2 := obj.Addrel(ctxt.Cursym)
		rel2.Off = int32(ctxt.Pc + 4)
		rel2.Siz = 4
		rel2.Sym = p.From.Sym
		rel2.Add = p.From.Offset
		rel2.Type = obj.R_ADDRMIPS

		if o.size == 12 {
			o3 = o2
			o2 = OP_RRR(oprrr(ctxt, AADDVU), uint32(REGSB), uint32(REGTMP), uint32(REGTMP))
			rel2.Off += 4
		}

	case 52: /* mov $lext, r ==> lu + add REGSB, r + add */
		o1 = OP_IRR(opirr(ctxt, ALUI), uint32(0), uint32(REGZERO), uint32(p.To.Reg))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.From.Sym
		rel.Add = p.From.Offset
		rel.Type = obj.R_ADDRMIPSU
		o2 = OP_IRR(opirr(ctxt, add), uint32(0), uint32(p.To.Reg), uint32(p.To.Reg))
		rel2 := obj.Addrel(ctxt.Cursym)
		rel2.Off = int32(ctxt.Pc + 4)
		rel2.Siz = 4
		rel2.Sym = p.From.Sym
		rel2.Add = p.From.Offset
		rel2.Type = obj.R_ADDRMIPS

		if o.size == 12 {
			o3 = o2
			o2 = OP_RRR(oprrr(ctxt, AADDVU), uint32(REGSB), uint32(p.To.Reg), uint32(p.To.Reg))
			rel2.Off += 4
		}

	case 53: /* mov r, tlsvar ==> rdhwr + sw o(r3) */
		// clobbers R3 !
		// load thread pointer with RDHWR, R3 is used for fast kernel emulation on Linux
		o1 = (037<<26 + 073) | (29 << 11) | (3 << 16) // rdhwr $29, r3
		o2 = OP_IRR(opirr(ctxt, p.As), uint32(0), uint32(REG_R3), uint32(p.From.Reg))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc + 4)
		rel.Siz = 4
		rel.Sym = p.To.Sym
		rel.Add = p.To.Offset
		rel.Type = obj.R_ADDRMIPSTLS

	case 54: /* mov tlsvar, r ==> rdhwr + lw o(r3) */
		// clobbers R3 !
		o1 = (037<<26 + 073) | (29 << 11) | (3 << 16) // rdhwr $29, r3
		o2 = OP_IRR(opirr(ctxt, -p.As), uint32(0), uint32(REG_R3), uint32(p.To.Reg))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc + 4)
		rel.Siz = 4
		rel.Sym = p.From.Sym
		rel.Add = p.From.Offset
		rel.Type = obj.R_ADDRMIPSTLS

	case 55: /* mov $tlsvar, r ==> rdhwr + add */
		// clobbers R3 !
		o1 = (037<<26 + 073) | (29 << 11) | (3 << 16) // rdhwr $29, r3
		o2 = OP_IRR(opirr(ctxt, add), uint32(0), uint32(REG_R3), uint32(p.To.Reg))
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc + 4)
		rel.Siz = 4
		rel.Sym = p.From.Sym
		rel.Add = p.From.Offset
		rel.Type = obj.R_ADDRMIPSTLS
	}

	out[0] = o1
	out[1] = o2
	out[2] = o3
	out[3] = o4
	return
}

func vregoff(ctxt *obj.Link, a *obj.Addr) int64 {
	ctxt.Instoffset = 0
	aclass(ctxt, a)
	return ctxt.Instoffset
}

func regoff(ctxt *obj.Link, a *obj.Addr) int32 {
	return int32(vregoff(ctxt, a))
}

func oprrr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AADD:
		return OP(4, 0)
	case AADDU:
		return OP(4, 1)
	case ASGT:
		return OP(5, 2)
	case ASGTU:
		return OP(5, 3)
	case AAND:
		return OP(4, 4)
	case AOR:
		return OP(4, 5)
	case AXOR:
		return OP(4, 6)
	case ASUB:
		return OP(4, 2)
	case ASUBU:
		return OP(4, 3)
	case ANOR:
		return OP(4, 7)
	case ASLL:
		return OP(0, 4)
	case ASRL:
		return OP(0, 6)
	case ASRA:
		return OP(0, 7)
	case ASLLV:
		return OP(2, 4)
	case ASRLV:
		return OP(2, 6)
	case ASRAV:
		return OP(2, 7)
	case AADDV:
		return OP(5, 4)
	case AADDVU:
		return OP(5, 5)
	case ASUBV:
		return OP(5, 6)
	case ASUBVU:
		return OP(5, 7)
	case AREM,
		ADIV:
		return OP(3, 2)
	case AREMU,
		ADIVU:
		return OP(3, 3)
	case AMUL:
		return OP(3, 0)
	case AMULU:
		return OP(3, 1)
	case AREMV,
		ADIVV:
		return OP(3, 6)
	case AREMVU,
		ADIVVU:
		return OP(3, 7)
	case AMULV:
		return OP(3, 4)
	case AMULVU:
		return OP(3, 5)

	case AJMP:
		return OP(1, 0)
	case AJAL:
		return OP(1, 1)

	case ABREAK:
		return OP(1, 5)
	case ASYSCALL:
		return OP(1, 4)
	case ATLBP:
		return MMU(1, 0)
	case ATLBR:
		return MMU(0, 1)
	case ATLBWI:
		return MMU(0, 2)
	case ATLBWR:
		return MMU(0, 6)
	case ARFE:
		return MMU(2, 0)

	case ADIVF:
		return FPF(0, 3)
	case ADIVD:
		return FPD(0, 3)
	case AMULF:
		return FPF(0, 2)
	case AMULD:
		return FPD(0, 2)
	case ASUBF:
		return FPF(0, 1)
	case ASUBD:
		return FPD(0, 1)
	case AADDF:
		return FPF(0, 0)
	case AADDD:
		return FPD(0, 0)
	case ATRUNCFV:
		return FPF(1, 1)
	case ATRUNCDV:
		return FPD(1, 1)
	case ATRUNCFW:
		return FPF(1, 5)
	case ATRUNCDW:
		return FPD(1, 5)
	case AMOVFV:
		return FPF(4, 5)
	case AMOVDV:
		return FPD(4, 5)
	case AMOVVF:
		return FPV(4, 0)
	case AMOVVD:
		return FPV(4, 1)
	case AMOVFW:
		return FPF(4, 4)
	case AMOVDW:
		return FPD(4, 4)
	case AMOVWF:
		return FPW(4, 0)
	case AMOVDF:
		return FPD(4, 0)
	case AMOVWD:
		return FPW(4, 1)
	case AMOVFD:
		return FPF(4, 1)
	case AABSF:
		return FPF(0, 5)
	case AABSD:
		return FPD(0, 5)
	case AMOVF:
		return FPF(0, 6)
	case AMOVD:
		return FPD(0, 6)
	case ANEGF:
		return FPF(0, 7)
	case ANEGD:
		return FPD(0, 7)
	case ACMPEQF:
		return FPF(6, 2)
	case ACMPEQD:
		return FPD(6, 2)
	case ACMPGTF:
		return FPF(7, 4)
	case ACMPGTD:
		return FPD(7, 4)
	case ACMPGEF:
		return FPF(7, 6)
	case ACMPGED:
		return FPD(7, 6)

	case ASQRTF:
		return FPF(0, 4)
	case ASQRTD:
		return FPD(0, 4)

	case ASYNC:
		return OP(1, 7)

	case ACMOVN:
		return OP(1, 3)
	case ACMOVZ:
		return OP(1, 2)
	case ACMOVT:
		return OP(0, 1) | (1 << 16)
	case ACMOVF:
		return OP(0, 1) | (0 << 16)
	case ACLO:
		return SP(3, 4) | OP(4, 1)
	case ACLZ:
		return SP(3, 4) | OP(4, 0)
	}

	if a < 0 {
		ctxt.Diag("bad rrr opcode -%v", -a)
	} else {
		ctxt.Diag("bad rrr opcode %v", a)
	}
	return 0
}

func opirr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AADD:
		return SP(1, 0)
	case AADDU:
		return SP(1, 1)
	case ASGT:
		return SP(1, 2)
	case ASGTU:
		return SP(1, 3)
	case AAND:
		return SP(1, 4)
	case AOR:
		return SP(1, 5)
	case AXOR:
		return SP(1, 6)
	case ALUI:
		return SP(1, 7)
	case ASLL:
		return OP(0, 0)
	case ASRL:
		return OP(0, 2)
	case ASRA:
		return OP(0, 3)
	case AADDV:
		return SP(3, 0)
	case AADDVU:
		return SP(3, 1)

	case AJMP:
		return SP(0, 2)
	case AJAL,
		obj.ADUFFZERO,
		obj.ADUFFCOPY:
		return SP(0, 3)
	case ABEQ:
		return SP(0, 4)
	case -ABEQ:
		return SP(2, 4) /* likely */
	case ABNE:
		return SP(0, 5)
	case -ABNE:
		return SP(2, 5) /* likely */
	case ABGEZ:
		return SP(0, 1) | BCOND(0, 1)
	case -ABGEZ:
		return SP(0, 1) | BCOND(0, 3) /* likely */
	case ABGEZAL:
		return SP(0, 1) | BCOND(2, 1)
	case -ABGEZAL:
		return SP(0, 1) | BCOND(2, 3) /* likely */
	case ABGTZ:
		return SP(0, 7)
	case -ABGTZ:
		return SP(2, 7) /* likely */
	case ABLEZ:
		return SP(0, 6)
	case -ABLEZ:
		return SP(2, 6) /* likely */
	case ABLTZ:
		return SP(0, 1) | BCOND(0, 0)
	case -ABLTZ:
		return SP(0, 1) | BCOND(0, 2) /* likely */
	case ABLTZAL:
		return SP(0, 1) | BCOND(2, 0)
	case -ABLTZAL:
		return SP(0, 1) | BCOND(2, 2) /* likely */
	case ABFPT:
		return SP(2, 1) | (257 << 16)
	case -ABFPT:
		return SP(2, 1) | (259 << 16) /* likely */
	case ABFPF:
		return SP(2, 1) | (256 << 16)
	case -ABFPF:
		return SP(2, 1) | (258 << 16) /* likely */

	case AMOVB,
		AMOVBU:
		return SP(5, 0)
	case AMOVH,
		AMOVHU:
		return SP(5, 1)
	case AMOVW,
		AMOVWU:
		return SP(5, 3)
	case AMOVV:
		return SP(7, 7)
	case AMOVF:
		return SP(7, 1)
	case AMOVD:
		return SP(7, 5)
	case AMOVWL:
		return SP(5, 2)
	case AMOVWR:
		return SP(5, 6)
	case AMOVVL:
		return SP(5, 4)
	case AMOVVR:
		return SP(5, 5)

	case ABREAK:
		return SP(5, 7)

	case -AMOVWL:
		return SP(4, 2)
	case -AMOVWR:
		return SP(4, 6)
	case -AMOVVL:
		return SP(3, 2)
	case -AMOVVR:
		return SP(3, 3)
	case -AMOVB:
		return SP(4, 0)
	case -AMOVBU:
		return SP(4, 4)
	case -AMOVH:
		return SP(4, 1)
	case -AMOVHU:
		return SP(4, 5)
	case -AMOVW:
		return SP(4, 3)
	case -AMOVWU:
		return SP(4, 7)
	case -AMOVV:
		return SP(6, 7)
	case -AMOVF:
		return SP(6, 1)
	case -AMOVD:
		return SP(6, 5)

	case ASLLV:
		return OP(7, 0)
	case ASRLV:
		return OP(7, 2)
	case ASRAV:
		return OP(7, 3)
	case -ASLLV:
		return OP(7, 4)
	case -ASRLV:
		return OP(7, 6)
	case -ASRAV:
		return OP(7, 7)

	case ATEQ:
		return OP(6, 4)
	case ATNE:
		return OP(6, 6)
	case -ALL:
		return SP(6, 0)
	case ASC:
		return SP(7, 0)
	}

	if a < 0 {
		ctxt.Diag("bad irr opcode -%v", -a)
	} else {
		ctxt.Diag("bad irr opcode %v", a)
	}
	return 0
}

func vshift(a obj.As) bool {
	switch a {
	case ASLLV,
		ASRLV,
		ASRAV:
		return true
	}
	return false
}
