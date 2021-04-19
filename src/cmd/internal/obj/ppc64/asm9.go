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

package ppc64

import (
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"log"
	"sort"
)

// Instruction layout.

const (
	funcAlign = 8
)

const (
	r0iszero = 1
)

type Optab struct {
	as    obj.As // Opcode
	a1    uint8
	a2    uint8
	a3    uint8
	a4    uint8
	type_ int8 // cases in asmout below. E.g., 44 = st r,(ra+rb); 45 = ld (ra+rb), r
	size  int8
	param int16
}

var optab = []Optab{
	{obj.ATEXT, C_LEXT, C_NONE, C_NONE, C_TEXTSIZE, 0, 0, 0},
	{obj.ATEXT, C_LEXT, C_NONE, C_LCON, C_TEXTSIZE, 0, 0, 0},
	{obj.ATEXT, C_ADDR, C_NONE, C_NONE, C_TEXTSIZE, 0, 0, 0},
	{obj.ATEXT, C_ADDR, C_NONE, C_LCON, C_TEXTSIZE, 0, 0, 0},
	/* move register */
	{AMOVD, C_REG, C_NONE, C_NONE, C_REG, 1, 4, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_REG, 12, 4, 0},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_REG, 13, 4, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_REG, 12, 4, 0},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_REG, 13, 4, 0},
	{AADD, C_REG, C_REG, C_NONE, C_REG, 2, 4, 0},
	{AADD, C_REG, C_NONE, C_NONE, C_REG, 2, 4, 0},
	{AADD, C_ADDCON, C_REG, C_NONE, C_REG, 4, 4, 0},
	{AADD, C_ADDCON, C_NONE, C_NONE, C_REG, 4, 4, 0},
	{AADD, C_UCON, C_REG, C_NONE, C_REG, 20, 4, 0},
	{AADD, C_UCON, C_NONE, C_NONE, C_REG, 20, 4, 0},
	{AADD, C_LCON, C_REG, C_NONE, C_REG, 22, 12, 0},
	{AADD, C_LCON, C_NONE, C_NONE, C_REG, 22, 12, 0},
	{AADDC, C_REG, C_REG, C_NONE, C_REG, 2, 4, 0},
	{AADDC, C_REG, C_NONE, C_NONE, C_REG, 2, 4, 0},
	{AADDC, C_ADDCON, C_REG, C_NONE, C_REG, 4, 4, 0},
	{AADDC, C_ADDCON, C_NONE, C_NONE, C_REG, 4, 4, 0},
	{AADDC, C_LCON, C_REG, C_NONE, C_REG, 22, 12, 0},
	{AADDC, C_LCON, C_NONE, C_NONE, C_REG, 22, 12, 0},
	{AAND, C_REG, C_REG, C_NONE, C_REG, 6, 4, 0}, /* logical, no literal */
	{AAND, C_REG, C_NONE, C_NONE, C_REG, 6, 4, 0},
	{AANDCC, C_REG, C_REG, C_NONE, C_REG, 6, 4, 0},
	{AANDCC, C_REG, C_NONE, C_NONE, C_REG, 6, 4, 0},
	{AANDCC, C_ANDCON, C_NONE, C_NONE, C_REG, 58, 4, 0},
	{AANDCC, C_ANDCON, C_REG, C_NONE, C_REG, 58, 4, 0},
	{AANDCC, C_UCON, C_NONE, C_NONE, C_REG, 59, 4, 0},
	{AANDCC, C_UCON, C_REG, C_NONE, C_REG, 59, 4, 0},
	{AANDCC, C_LCON, C_NONE, C_NONE, C_REG, 23, 12, 0},
	{AANDCC, C_LCON, C_REG, C_NONE, C_REG, 23, 12, 0},
	{AMULLW, C_REG, C_REG, C_NONE, C_REG, 2, 4, 0},
	{AMULLW, C_REG, C_NONE, C_NONE, C_REG, 2, 4, 0},
	{AMULLW, C_ADDCON, C_REG, C_NONE, C_REG, 4, 4, 0},
	{AMULLW, C_ADDCON, C_NONE, C_NONE, C_REG, 4, 4, 0},
	{AMULLW, C_ANDCON, C_REG, C_NONE, C_REG, 4, 4, 0},
	{AMULLW, C_ANDCON, C_NONE, C_NONE, C_REG, 4, 4, 0},
	{AMULLW, C_LCON, C_REG, C_NONE, C_REG, 22, 12, 0},
	{AMULLW, C_LCON, C_NONE, C_NONE, C_REG, 22, 12, 0},
	{ASUBC, C_REG, C_REG, C_NONE, C_REG, 10, 4, 0},
	{ASUBC, C_REG, C_NONE, C_NONE, C_REG, 10, 4, 0},
	{ASUBC, C_REG, C_NONE, C_ADDCON, C_REG, 27, 4, 0},
	{ASUBC, C_REG, C_NONE, C_LCON, C_REG, 28, 12, 0},
	{AOR, C_REG, C_REG, C_NONE, C_REG, 6, 4, 0}, /* logical, literal not cc (or/xor) */
	{AOR, C_REG, C_NONE, C_NONE, C_REG, 6, 4, 0},
	{AOR, C_ANDCON, C_NONE, C_NONE, C_REG, 58, 4, 0},
	{AOR, C_ANDCON, C_REG, C_NONE, C_REG, 58, 4, 0},
	{AOR, C_UCON, C_NONE, C_NONE, C_REG, 59, 4, 0},
	{AOR, C_UCON, C_REG, C_NONE, C_REG, 59, 4, 0},
	{AOR, C_LCON, C_NONE, C_NONE, C_REG, 23, 12, 0},
	{AOR, C_LCON, C_REG, C_NONE, C_REG, 23, 12, 0},
	{ADIVW, C_REG, C_REG, C_NONE, C_REG, 2, 4, 0}, /* op r1[,r2],r3 */
	{ADIVW, C_REG, C_NONE, C_NONE, C_REG, 2, 4, 0},
	{ASUB, C_REG, C_REG, C_NONE, C_REG, 10, 4, 0}, /* op r2[,r1],r3 */
	{ASUB, C_REG, C_NONE, C_NONE, C_REG, 10, 4, 0},
	{ASLW, C_REG, C_NONE, C_NONE, C_REG, 6, 4, 0},
	{ASLW, C_REG, C_REG, C_NONE, C_REG, 6, 4, 0},
	{ASLD, C_REG, C_NONE, C_NONE, C_REG, 6, 4, 0},
	{ASLD, C_REG, C_REG, C_NONE, C_REG, 6, 4, 0},
	{ASLD, C_SCON, C_REG, C_NONE, C_REG, 25, 4, 0},
	{ASLD, C_SCON, C_NONE, C_NONE, C_REG, 25, 4, 0},
	{ASLW, C_SCON, C_REG, C_NONE, C_REG, 57, 4, 0},
	{ASLW, C_SCON, C_NONE, C_NONE, C_REG, 57, 4, 0},
	{ASRAW, C_REG, C_NONE, C_NONE, C_REG, 6, 4, 0},
	{ASRAW, C_REG, C_REG, C_NONE, C_REG, 6, 4, 0},
	{ASRAW, C_SCON, C_REG, C_NONE, C_REG, 56, 4, 0},
	{ASRAW, C_SCON, C_NONE, C_NONE, C_REG, 56, 4, 0},
	{ASRAD, C_REG, C_NONE, C_NONE, C_REG, 6, 4, 0},
	{ASRAD, C_REG, C_REG, C_NONE, C_REG, 6, 4, 0},
	{ASRAD, C_SCON, C_REG, C_NONE, C_REG, 56, 4, 0},
	{ASRAD, C_SCON, C_NONE, C_NONE, C_REG, 56, 4, 0},
	{ARLWMI, C_SCON, C_REG, C_LCON, C_REG, 62, 4, 0},
	{ARLWMI, C_REG, C_REG, C_LCON, C_REG, 63, 4, 0},
	{ARLDMI, C_SCON, C_REG, C_LCON, C_REG, 30, 4, 0},
	{ARLDC, C_SCON, C_REG, C_LCON, C_REG, 29, 4, 0},
	{ARLDCL, C_SCON, C_REG, C_LCON, C_REG, 29, 4, 0},
	{ARLDCL, C_REG, C_REG, C_LCON, C_REG, 14, 4, 0},
	{ARLDICL, C_REG, C_REG, C_LCON, C_REG, 14, 4, 0},
	{ARLDICL, C_SCON, C_REG, C_LCON, C_REG, 14, 4, 0},
	{ARLDCL, C_REG, C_NONE, C_LCON, C_REG, 14, 4, 0},
	{AFADD, C_FREG, C_NONE, C_NONE, C_FREG, 2, 4, 0},
	{AFADD, C_FREG, C_FREG, C_NONE, C_FREG, 2, 4, 0},
	{AFABS, C_FREG, C_NONE, C_NONE, C_FREG, 33, 4, 0},
	{AFABS, C_NONE, C_NONE, C_NONE, C_FREG, 33, 4, 0},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_FREG, 33, 4, 0},
	{AFMADD, C_FREG, C_FREG, C_FREG, C_FREG, 34, 4, 0},
	{AFMUL, C_FREG, C_NONE, C_NONE, C_FREG, 32, 4, 0},
	{AFMUL, C_FREG, C_FREG, C_NONE, C_FREG, 32, 4, 0},

	/* store, short offset */
	{AMOVD, C_REG, C_REG, C_NONE, C_ZOREG, 7, 4, REGZERO},
	{AMOVW, C_REG, C_REG, C_NONE, C_ZOREG, 7, 4, REGZERO},
	{AMOVWZ, C_REG, C_REG, C_NONE, C_ZOREG, 7, 4, REGZERO},
	{AMOVBZ, C_REG, C_REG, C_NONE, C_ZOREG, 7, 4, REGZERO},
	{AMOVBZU, C_REG, C_REG, C_NONE, C_ZOREG, 7, 4, REGZERO},
	{AMOVB, C_REG, C_REG, C_NONE, C_ZOREG, 7, 4, REGZERO},
	{AMOVBU, C_REG, C_REG, C_NONE, C_ZOREG, 7, 4, REGZERO},
	{AMOVD, C_REG, C_NONE, C_NONE, C_SEXT, 7, 4, REGSB},
	{AMOVW, C_REG, C_NONE, C_NONE, C_SEXT, 7, 4, REGSB},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_SEXT, 7, 4, REGSB},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_SEXT, 7, 4, REGSB},
	{AMOVB, C_REG, C_NONE, C_NONE, C_SEXT, 7, 4, REGSB},
	{AMOVD, C_REG, C_NONE, C_NONE, C_SAUTO, 7, 4, REGSP},
	{AMOVW, C_REG, C_NONE, C_NONE, C_SAUTO, 7, 4, REGSP},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_SAUTO, 7, 4, REGSP},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_SAUTO, 7, 4, REGSP},
	{AMOVB, C_REG, C_NONE, C_NONE, C_SAUTO, 7, 4, REGSP},
	{AMOVD, C_REG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},
	{AMOVW, C_REG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},
	{AMOVBZU, C_REG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},
	{AMOVB, C_REG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},

	/* load, short offset */
	{AMOVD, C_ZOREG, C_REG, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVW, C_ZOREG, C_REG, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVWZ, C_ZOREG, C_REG, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVBZ, C_ZOREG, C_REG, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVBZU, C_ZOREG, C_REG, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVB, C_ZOREG, C_REG, C_NONE, C_REG, 9, 8, REGZERO},
	{AMOVBU, C_ZOREG, C_REG, C_NONE, C_REG, 9, 8, REGZERO},
	{AMOVD, C_SEXT, C_NONE, C_NONE, C_REG, 8, 4, REGSB},
	{AMOVW, C_SEXT, C_NONE, C_NONE, C_REG, 8, 4, REGSB},
	{AMOVWZ, C_SEXT, C_NONE, C_NONE, C_REG, 8, 4, REGSB},
	{AMOVBZ, C_SEXT, C_NONE, C_NONE, C_REG, 8, 4, REGSB},
	{AMOVB, C_SEXT, C_NONE, C_NONE, C_REG, 9, 8, REGSB},
	{AMOVD, C_SAUTO, C_NONE, C_NONE, C_REG, 8, 4, REGSP},
	{AMOVW, C_SAUTO, C_NONE, C_NONE, C_REG, 8, 4, REGSP},
	{AMOVWZ, C_SAUTO, C_NONE, C_NONE, C_REG, 8, 4, REGSP},
	{AMOVBZ, C_SAUTO, C_NONE, C_NONE, C_REG, 8, 4, REGSP},
	{AMOVB, C_SAUTO, C_NONE, C_NONE, C_REG, 9, 8, REGSP},
	{AMOVD, C_SOREG, C_NONE, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVW, C_SOREG, C_NONE, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVWZ, C_SOREG, C_NONE, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVBZ, C_SOREG, C_NONE, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVBZU, C_SOREG, C_NONE, C_NONE, C_REG, 8, 4, REGZERO},
	{AMOVB, C_SOREG, C_NONE, C_NONE, C_REG, 9, 8, REGZERO},
	{AMOVBU, C_SOREG, C_NONE, C_NONE, C_REG, 9, 8, REGZERO},

	/* store, long offset */
	{AMOVD, C_REG, C_NONE, C_NONE, C_LEXT, 35, 8, REGSB},
	{AMOVW, C_REG, C_NONE, C_NONE, C_LEXT, 35, 8, REGSB},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_LEXT, 35, 8, REGSB},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_LEXT, 35, 8, REGSB},
	{AMOVB, C_REG, C_NONE, C_NONE, C_LEXT, 35, 8, REGSB},
	{AMOVD, C_REG, C_NONE, C_NONE, C_LAUTO, 35, 8, REGSP},
	{AMOVW, C_REG, C_NONE, C_NONE, C_LAUTO, 35, 8, REGSP},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_LAUTO, 35, 8, REGSP},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_LAUTO, 35, 8, REGSP},
	{AMOVB, C_REG, C_NONE, C_NONE, C_LAUTO, 35, 8, REGSP},
	{AMOVD, C_REG, C_NONE, C_NONE, C_LOREG, 35, 8, REGZERO},
	{AMOVW, C_REG, C_NONE, C_NONE, C_LOREG, 35, 8, REGZERO},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_LOREG, 35, 8, REGZERO},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_LOREG, 35, 8, REGZERO},
	{AMOVB, C_REG, C_NONE, C_NONE, C_LOREG, 35, 8, REGZERO},
	{AMOVD, C_REG, C_NONE, C_NONE, C_ADDR, 74, 8, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_ADDR, 74, 8, 0},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_ADDR, 74, 8, 0},
	{AMOVBZ, C_REG, C_NONE, C_NONE, C_ADDR, 74, 8, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_ADDR, 74, 8, 0},

	/* load, long offset */
	{AMOVD, C_LEXT, C_NONE, C_NONE, C_REG, 36, 8, REGSB},
	{AMOVW, C_LEXT, C_NONE, C_NONE, C_REG, 36, 8, REGSB},
	{AMOVWZ, C_LEXT, C_NONE, C_NONE, C_REG, 36, 8, REGSB},
	{AMOVBZ, C_LEXT, C_NONE, C_NONE, C_REG, 36, 8, REGSB},
	{AMOVB, C_LEXT, C_NONE, C_NONE, C_REG, 37, 12, REGSB},
	{AMOVD, C_LAUTO, C_NONE, C_NONE, C_REG, 36, 8, REGSP},
	{AMOVW, C_LAUTO, C_NONE, C_NONE, C_REG, 36, 8, REGSP},
	{AMOVWZ, C_LAUTO, C_NONE, C_NONE, C_REG, 36, 8, REGSP},
	{AMOVBZ, C_LAUTO, C_NONE, C_NONE, C_REG, 36, 8, REGSP},
	{AMOVB, C_LAUTO, C_NONE, C_NONE, C_REG, 37, 12, REGSP},
	{AMOVD, C_LOREG, C_NONE, C_NONE, C_REG, 36, 8, REGZERO},
	{AMOVW, C_LOREG, C_NONE, C_NONE, C_REG, 36, 8, REGZERO},
	{AMOVWZ, C_LOREG, C_NONE, C_NONE, C_REG, 36, 8, REGZERO},
	{AMOVBZ, C_LOREG, C_NONE, C_NONE, C_REG, 36, 8, REGZERO},
	{AMOVB, C_LOREG, C_NONE, C_NONE, C_REG, 37, 12, REGZERO},
	{AMOVD, C_ADDR, C_NONE, C_NONE, C_REG, 75, 8, 0},
	{AMOVW, C_ADDR, C_NONE, C_NONE, C_REG, 75, 8, 0},
	{AMOVWZ, C_ADDR, C_NONE, C_NONE, C_REG, 75, 8, 0},
	{AMOVBZ, C_ADDR, C_NONE, C_NONE, C_REG, 75, 8, 0},
	{AMOVB, C_ADDR, C_NONE, C_NONE, C_REG, 76, 12, 0},

	{AMOVD, C_TLS_LE, C_NONE, C_NONE, C_REG, 79, 4, 0},
	{AMOVD, C_TLS_IE, C_NONE, C_NONE, C_REG, 80, 8, 0},

	{AMOVD, C_GOTADDR, C_NONE, C_NONE, C_REG, 81, 8, 0},

	/* load constant */
	{AMOVD, C_SECON, C_NONE, C_NONE, C_REG, 3, 4, REGSB},
	{AMOVD, C_SACON, C_NONE, C_NONE, C_REG, 3, 4, REGSP},
	{AMOVD, C_LECON, C_NONE, C_NONE, C_REG, 26, 8, REGSB},
	{AMOVD, C_LACON, C_NONE, C_NONE, C_REG, 26, 8, REGSP},
	{AMOVD, C_ADDCON, C_NONE, C_NONE, C_REG, 3, 4, REGZERO},
	{AMOVW, C_SECON, C_NONE, C_NONE, C_REG, 3, 4, REGSB}, /* TO DO: check */
	{AMOVW, C_SACON, C_NONE, C_NONE, C_REG, 3, 4, REGSP},
	{AMOVW, C_LECON, C_NONE, C_NONE, C_REG, 26, 8, REGSB},
	{AMOVW, C_LACON, C_NONE, C_NONE, C_REG, 26, 8, REGSP},
	{AMOVW, C_ADDCON, C_NONE, C_NONE, C_REG, 3, 4, REGZERO},
	{AMOVWZ, C_SECON, C_NONE, C_NONE, C_REG, 3, 4, REGSB}, /* TO DO: check */
	{AMOVWZ, C_SACON, C_NONE, C_NONE, C_REG, 3, 4, REGSP},
	{AMOVWZ, C_LECON, C_NONE, C_NONE, C_REG, 26, 8, REGSB},
	{AMOVWZ, C_LACON, C_NONE, C_NONE, C_REG, 26, 8, REGSP},
	{AMOVWZ, C_ADDCON, C_NONE, C_NONE, C_REG, 3, 4, REGZERO},

	/* load unsigned/long constants (TO DO: check) */
	{AMOVD, C_UCON, C_NONE, C_NONE, C_REG, 3, 4, REGZERO},
	{AMOVD, C_LCON, C_NONE, C_NONE, C_REG, 19, 8, 0},
	{AMOVW, C_UCON, C_NONE, C_NONE, C_REG, 3, 4, REGZERO},
	{AMOVW, C_LCON, C_NONE, C_NONE, C_REG, 19, 8, 0},
	{AMOVWZ, C_UCON, C_NONE, C_NONE, C_REG, 3, 4, REGZERO},
	{AMOVWZ, C_LCON, C_NONE, C_NONE, C_REG, 19, 8, 0},
	{AMOVHBR, C_ZOREG, C_REG, C_NONE, C_REG, 45, 4, 0},
	{AMOVHBR, C_ZOREG, C_NONE, C_NONE, C_REG, 45, 4, 0},
	{AMOVHBR, C_REG, C_REG, C_NONE, C_ZOREG, 44, 4, 0},
	{AMOVHBR, C_REG, C_NONE, C_NONE, C_ZOREG, 44, 4, 0},
	{ASYSCALL, C_NONE, C_NONE, C_NONE, C_NONE, 5, 4, 0},
	{ASYSCALL, C_REG, C_NONE, C_NONE, C_NONE, 77, 12, 0},
	{ASYSCALL, C_SCON, C_NONE, C_NONE, C_NONE, 77, 12, 0},
	{ABEQ, C_NONE, C_NONE, C_NONE, C_SBRA, 16, 4, 0},
	{ABEQ, C_CREG, C_NONE, C_NONE, C_SBRA, 16, 4, 0},
	{ABR, C_NONE, C_NONE, C_NONE, C_LBRA, 11, 4, 0},
	{ABR, C_NONE, C_NONE, C_NONE, C_LBRAPIC, 11, 8, 0},
	{ABC, C_SCON, C_REG, C_NONE, C_SBRA, 16, 4, 0},
	{ABC, C_SCON, C_REG, C_NONE, C_LBRA, 17, 4, 0},
	{ABR, C_NONE, C_NONE, C_NONE, C_LR, 18, 4, 0},
	{ABR, C_NONE, C_NONE, C_NONE, C_CTR, 18, 4, 0},
	{ABR, C_REG, C_NONE, C_NONE, C_CTR, 18, 4, 0},
	{ABR, C_NONE, C_NONE, C_NONE, C_ZOREG, 15, 8, 0},
	{ABC, C_NONE, C_REG, C_NONE, C_LR, 18, 4, 0},
	{ABC, C_NONE, C_REG, C_NONE, C_CTR, 18, 4, 0},
	{ABC, C_SCON, C_REG, C_NONE, C_LR, 18, 4, 0},
	{ABC, C_SCON, C_REG, C_NONE, C_CTR, 18, 4, 0},
	{ABC, C_NONE, C_NONE, C_NONE, C_ZOREG, 15, 8, 0},
	{AFMOVD, C_SEXT, C_NONE, C_NONE, C_FREG, 8, 4, REGSB},
	{AFMOVD, C_SAUTO, C_NONE, C_NONE, C_FREG, 8, 4, REGSP},
	{AFMOVD, C_SOREG, C_NONE, C_NONE, C_FREG, 8, 4, REGZERO},
	{AFMOVD, C_LEXT, C_NONE, C_NONE, C_FREG, 36, 8, REGSB},
	{AFMOVD, C_LAUTO, C_NONE, C_NONE, C_FREG, 36, 8, REGSP},
	{AFMOVD, C_LOREG, C_NONE, C_NONE, C_FREG, 36, 8, REGZERO},
	{AFMOVD, C_ADDR, C_NONE, C_NONE, C_FREG, 75, 8, 0},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_SEXT, 7, 4, REGSB},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_SAUTO, 7, 4, REGSP},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_SOREG, 7, 4, REGZERO},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_LEXT, 35, 8, REGSB},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_LAUTO, 35, 8, REGSP},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_LOREG, 35, 8, REGZERO},
	{AFMOVD, C_FREG, C_NONE, C_NONE, C_ADDR, 74, 8, 0},
	{AFMOVSX, C_ZOREG, C_REG, C_NONE, C_FREG, 45, 4, 0},
	{AFMOVSX, C_ZOREG, C_NONE, C_NONE, C_FREG, 45, 4, 0},
	{AFMOVSX, C_FREG, C_REG, C_NONE, C_ZOREG, 44, 4, 0},
	{AFMOVSX, C_FREG, C_NONE, C_NONE, C_ZOREG, 44, 4, 0},
	{AFMOVSZ, C_ZOREG, C_REG, C_NONE, C_FREG, 45, 4, 0},
	{AFMOVSZ, C_ZOREG, C_NONE, C_NONE, C_FREG, 45, 4, 0},
	{ASYNC, C_NONE, C_NONE, C_NONE, C_NONE, 46, 4, 0},
	{AWORD, C_LCON, C_NONE, C_NONE, C_NONE, 40, 4, 0},
	{ADWORD, C_LCON, C_NONE, C_NONE, C_NONE, 31, 8, 0},
	{ADWORD, C_DCON, C_NONE, C_NONE, C_NONE, 31, 8, 0},
	{AADDME, C_REG, C_NONE, C_NONE, C_REG, 47, 4, 0},
	{AEXTSB, C_REG, C_NONE, C_NONE, C_REG, 48, 4, 0},
	{AEXTSB, C_NONE, C_NONE, C_NONE, C_REG, 48, 4, 0},
	{AISEL, C_LCON, C_REG, C_REG, C_REG, 84, 4, 0},
	{AISEL, C_ZCON, C_REG, C_REG, C_REG, 84, 4, 0},
	{ANEG, C_REG, C_NONE, C_NONE, C_REG, 47, 4, 0},
	{ANEG, C_NONE, C_NONE, C_NONE, C_REG, 47, 4, 0},
	{AREM, C_REG, C_NONE, C_NONE, C_REG, 50, 12, 0},
	{AREM, C_REG, C_REG, C_NONE, C_REG, 50, 12, 0},
	{AREMU, C_REG, C_NONE, C_NONE, C_REG, 50, 16, 0},
	{AREMU, C_REG, C_REG, C_NONE, C_REG, 50, 16, 0},
	{AREMD, C_REG, C_NONE, C_NONE, C_REG, 51, 12, 0},
	{AREMD, C_REG, C_REG, C_NONE, C_REG, 51, 12, 0},
	{AREMDU, C_REG, C_NONE, C_NONE, C_REG, 51, 12, 0},
	{AREMDU, C_REG, C_REG, C_NONE, C_REG, 51, 12, 0},
	{AMTFSB0, C_SCON, C_NONE, C_NONE, C_NONE, 52, 4, 0},
	{AMOVFL, C_FPSCR, C_NONE, C_NONE, C_FREG, 53, 4, 0},
	{AMOVFL, C_FREG, C_NONE, C_NONE, C_FPSCR, 64, 4, 0},
	{AMOVFL, C_FREG, C_NONE, C_LCON, C_FPSCR, 64, 4, 0},
	{AMOVFL, C_LCON, C_NONE, C_NONE, C_FPSCR, 65, 4, 0},
	{AMOVD, C_MSR, C_NONE, C_NONE, C_REG, 54, 4, 0},  /* mfmsr */
	{AMOVD, C_REG, C_NONE, C_NONE, C_MSR, 54, 4, 0},  /* mtmsrd */
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_MSR, 54, 4, 0}, /* mtmsr */

	/* Vector instructions */

	/* Vector load */
	{ALV, C_SOREG, C_NONE, C_NONE, C_VREG, 45, 4, 0}, /* vector load, x-form */

	/* Vector store */
	{ASTV, C_VREG, C_NONE, C_NONE, C_SOREG, 44, 4, 0}, /* vector store, x-form */

	/* Vector logical */
	{AVAND, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector and, vx-form */
	{AVOR, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0},  /* vector or, vx-form */

	/* Vector add */
	{AVADDUM, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector add unsigned modulo, vx-form */
	{AVADDCU, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector add & write carry unsigned, vx-form */
	{AVADDUS, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector add unsigned saturate, vx-form */
	{AVADDSS, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector add signed saturate, vx-form */
	{AVADDE, C_VREG, C_VREG, C_VREG, C_VREG, 83, 4, 0},  /* vector add extended, va-form */

	/* Vector subtract */
	{AVSUBUM, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector subtract unsigned modulo, vx-form */
	{AVSUBCU, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector subtract & write carry unsigned, vx-form */
	{AVSUBUS, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector subtract unsigned saturate, vx-form */
	{AVSUBSS, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector subtract signed saturate, vx-form */
	{AVSUBE, C_VREG, C_VREG, C_VREG, C_VREG, 83, 4, 0},  /* vector subtract extended, va-form */

	/* Vector rotate */
	{AVR, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector rotate, vx-form */

	/* Vector shift */
	{AVS, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0},     /* vector shift, vx-form */
	{AVSA, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0},    /* vector shift algebraic, vx-form */
	{AVSOI, C_ANDCON, C_VREG, C_VREG, C_VREG, 83, 4, 0}, /* vector shift by octet immediate, va-form */

	/* Vector count */
	{AVCLZ, C_VREG, C_NONE, C_NONE, C_VREG, 85, 4, 0},    /* vector count leading zeros, vx-form */
	{AVPOPCNT, C_VREG, C_NONE, C_NONE, C_VREG, 85, 4, 0}, /* vector population count, vx-form */

	/* Vector compare */
	{AVCMPEQ, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector compare equal, vc-form */
	{AVCMPGT, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector compare greater than, vc-form */

	/* Vector permute */
	{AVPERM, C_VREG, C_VREG, C_VREG, C_VREG, 83, 4, 0}, /* vector permute, va-form */

	/* Vector select */
	{AVSEL, C_VREG, C_VREG, C_VREG, C_VREG, 83, 4, 0}, /* vector select, va-form */

	/* Vector splat */
	{AVSPLT, C_SCON, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector splat, vx-form */
	{AVSPLT, C_ADDCON, C_VREG, C_NONE, C_VREG, 82, 4, 0},
	{AVSPLTI, C_SCON, C_NONE, C_NONE, C_VREG, 82, 4, 0}, /* vector splat immediate, vx-form */
	{AVSPLTI, C_ADDCON, C_NONE, C_NONE, C_VREG, 82, 4, 0},

	/* Vector AES */
	{AVCIPH, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0},  /* vector AES cipher, vx-form */
	{AVNCIPH, C_VREG, C_VREG, C_NONE, C_VREG, 82, 4, 0}, /* vector AES inverse cipher, vx-form */
	{AVSBOX, C_VREG, C_NONE, C_NONE, C_VREG, 82, 4, 0},  /* vector AES subbytes, vx-form */

	/* Vector SHA */
	{AVSHASIGMA, C_ANDCON, C_VREG, C_ANDCON, C_VREG, 82, 4, 0}, /* vector SHA sigma, vx-form */

	/* VSX vector load */
	{ALXV, C_SOREG, C_NONE, C_NONE, C_VSREG, 87, 4, 0}, /* vsx vector load, xx1-form */

	/* VSX vector store */
	{ASTXV, C_VSREG, C_NONE, C_NONE, C_SOREG, 86, 4, 0}, /* vsx vector store, xx1-form */

	/* VSX scalar load */
	{ALXS, C_SOREG, C_NONE, C_NONE, C_VSREG, 87, 4, 0}, /* vsx scalar load, xx1-form */

	/* VSX scalar store */
	{ASTXS, C_VSREG, C_NONE, C_NONE, C_SOREG, 86, 4, 0}, /* vsx scalar store, xx1-form */

	/* VSX scalar as integer load */
	{ALXSI, C_SOREG, C_NONE, C_NONE, C_VSREG, 87, 4, 0}, /* vsx scalar as integer load, xx1-form */

	/* VSX scalar store as integer */
	{ASTXSI, C_VSREG, C_NONE, C_NONE, C_SOREG, 86, 4, 0}, /* vsx scalar as integer store, xx1-form */

	/* VSX move from VSR */
	{AMFVSR, C_VSREG, C_NONE, C_NONE, C_REG, 88, 4, 0}, /* vsx move from vsr, xx1-form */

	/* VSX move to VSR */
	{AMTVSR, C_REG, C_NONE, C_NONE, C_VSREG, 88, 4, 0}, /* vsx move to vsr, xx1-form */

	/* VSX logical */
	{AXXLAND, C_VSREG, C_VSREG, C_NONE, C_VSREG, 90, 4, 0}, /* vsx and, xx3-form */
	{AXXLOR, C_VSREG, C_VSREG, C_NONE, C_VSREG, 90, 4, 0},  /* vsx or, xx3-form */

	/* VSX select */
	{AXXSEL, C_VSREG, C_VSREG, C_VSREG, C_VSREG, 91, 4, 0}, /* vsx select, xx4-form */

	/* VSX merge */
	{AXXMRG, C_VSREG, C_VSREG, C_NONE, C_VSREG, 90, 4, 0}, /* vsx merge, xx3-form */

	/* VSX splat */
	{AXXSPLT, C_VSREG, C_NONE, C_SCON, C_VSREG, 89, 4, 0}, /* vsx splat, xx2-form */

	/* VSX permute */
	{AXXPERM, C_VSREG, C_VSREG, C_SCON, C_VSREG, 90, 4, 0}, /* vsx permute, xx3-form */

	/* VSX shift */
	{AXXSI, C_VSREG, C_VSREG, C_SCON, C_VSREG, 90, 4, 0}, /* vsx shift immediate, xx3-form */

	/* VSX scalar FP-FP conversion */
	{AXSCV, C_VSREG, C_NONE, C_NONE, C_VSREG, 89, 4, 0}, /* vsx scalar fp-fp conversion, xx2-form */

	/* VSX vector FP-FP conversion */
	{AXVCV, C_VSREG, C_NONE, C_NONE, C_VSREG, 89, 4, 0}, /* vsx vector fp-fp conversion, xx2-form */

	/* VSX scalar FP-integer conversion */
	{AXSCVX, C_VSREG, C_NONE, C_NONE, C_VSREG, 89, 4, 0}, /* vsx scalar fp-integer conversion, xx2-form */

	/* VSX scalar integer-FP conversion */
	{AXSCVXP, C_VSREG, C_NONE, C_NONE, C_VSREG, 89, 4, 0}, /* vsx scalar integer-fp conversion, xx2-form */

	/* VSX vector FP-integer conversion */
	{AXVCVX, C_VSREG, C_NONE, C_NONE, C_VSREG, 89, 4, 0}, /* vsx vector fp-integer conversion, xx2-form */

	/* VSX vector integer-FP conversion */
	{AXVCVXP, C_VSREG, C_NONE, C_NONE, C_VSREG, 89, 4, 0}, /* vsx vector integer-fp conversion, xx2-form */

	/* 64-bit special registers */
	{AMOVD, C_REG, C_NONE, C_NONE, C_SPR, 66, 4, 0},
	{AMOVD, C_REG, C_NONE, C_NONE, C_LR, 66, 4, 0},
	{AMOVD, C_REG, C_NONE, C_NONE, C_CTR, 66, 4, 0},
	{AMOVD, C_REG, C_NONE, C_NONE, C_XER, 66, 4, 0},
	{AMOVD, C_SPR, C_NONE, C_NONE, C_REG, 66, 4, 0},
	{AMOVD, C_LR, C_NONE, C_NONE, C_REG, 66, 4, 0},
	{AMOVD, C_CTR, C_NONE, C_NONE, C_REG, 66, 4, 0},
	{AMOVD, C_XER, C_NONE, C_NONE, C_REG, 66, 4, 0},

	/* 32-bit special registers (gloss over sign-extension or not?) */
	{AMOVW, C_REG, C_NONE, C_NONE, C_SPR, 66, 4, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_CTR, 66, 4, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_XER, 66, 4, 0},
	{AMOVW, C_SPR, C_NONE, C_NONE, C_REG, 66, 4, 0},
	{AMOVW, C_XER, C_NONE, C_NONE, C_REG, 66, 4, 0},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_SPR, 66, 4, 0},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_CTR, 66, 4, 0},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_XER, 66, 4, 0},
	{AMOVWZ, C_SPR, C_NONE, C_NONE, C_REG, 66, 4, 0},
	{AMOVWZ, C_XER, C_NONE, C_NONE, C_REG, 66, 4, 0},
	{AMOVFL, C_FPSCR, C_NONE, C_NONE, C_CREG, 73, 4, 0},
	{AMOVFL, C_CREG, C_NONE, C_NONE, C_CREG, 67, 4, 0},
	{AMOVW, C_CREG, C_NONE, C_NONE, C_REG, 68, 4, 0},
	{AMOVWZ, C_CREG, C_NONE, C_NONE, C_REG, 68, 4, 0},
	{AMOVFL, C_REG, C_NONE, C_LCON, C_CREG, 69, 4, 0},
	{AMOVFL, C_REG, C_NONE, C_NONE, C_CREG, 69, 4, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_CREG, 69, 4, 0},
	{AMOVWZ, C_REG, C_NONE, C_NONE, C_CREG, 69, 4, 0},
	{ACMP, C_REG, C_NONE, C_NONE, C_REG, 70, 4, 0},
	{ACMP, C_REG, C_REG, C_NONE, C_REG, 70, 4, 0},
	{ACMP, C_REG, C_NONE, C_NONE, C_ADDCON, 71, 4, 0},
	{ACMP, C_REG, C_REG, C_NONE, C_ADDCON, 71, 4, 0},
	{ACMPU, C_REG, C_NONE, C_NONE, C_REG, 70, 4, 0},
	{ACMPU, C_REG, C_REG, C_NONE, C_REG, 70, 4, 0},
	{ACMPU, C_REG, C_NONE, C_NONE, C_ANDCON, 71, 4, 0},
	{ACMPU, C_REG, C_REG, C_NONE, C_ANDCON, 71, 4, 0},
	{AFCMPO, C_FREG, C_NONE, C_NONE, C_FREG, 70, 4, 0},
	{AFCMPO, C_FREG, C_REG, C_NONE, C_FREG, 70, 4, 0},
	{ATW, C_LCON, C_REG, C_NONE, C_REG, 60, 4, 0},
	{ATW, C_LCON, C_REG, C_NONE, C_ADDCON, 61, 4, 0},
	{ADCBF, C_ZOREG, C_NONE, C_NONE, C_NONE, 43, 4, 0},
	{ADCBF, C_ZOREG, C_REG, C_NONE, C_NONE, 43, 4, 0},
	{AECOWX, C_REG, C_REG, C_NONE, C_ZOREG, 44, 4, 0},
	{AECIWX, C_ZOREG, C_REG, C_NONE, C_REG, 45, 4, 0},
	{AECOWX, C_REG, C_NONE, C_NONE, C_ZOREG, 44, 4, 0},
	{AECIWX, C_ZOREG, C_NONE, C_NONE, C_REG, 45, 4, 0},
	{AEIEIO, C_NONE, C_NONE, C_NONE, C_NONE, 46, 4, 0},
	{ATLBIE, C_REG, C_NONE, C_NONE, C_NONE, 49, 4, 0},
	{ATLBIE, C_SCON, C_NONE, C_NONE, C_REG, 49, 4, 0},
	{ASLBMFEE, C_REG, C_NONE, C_NONE, C_REG, 55, 4, 0},
	{ASLBMTE, C_REG, C_NONE, C_NONE, C_REG, 55, 4, 0},
	{ASTSW, C_REG, C_NONE, C_NONE, C_ZOREG, 44, 4, 0},
	{ASTSW, C_REG, C_NONE, C_LCON, C_ZOREG, 41, 4, 0},
	{ALSW, C_ZOREG, C_NONE, C_NONE, C_REG, 45, 4, 0},
	{ALSW, C_ZOREG, C_NONE, C_LCON, C_REG, 42, 4, 0},
	{obj.AUNDEF, C_NONE, C_NONE, C_NONE, C_NONE, 78, 4, 0},
	{obj.AUSEFIELD, C_ADDR, C_NONE, C_NONE, C_NONE, 0, 0, 0},
	{obj.APCDATA, C_LCON, C_NONE, C_NONE, C_LCON, 0, 0, 0},
	{obj.AFUNCDATA, C_SCON, C_NONE, C_NONE, C_ADDR, 0, 0, 0},
	{obj.ANOP, C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0},
	{obj.ADUFFZERO, C_NONE, C_NONE, C_NONE, C_LBRA, 11, 4, 0}, // same as ABR/ABL
	{obj.ADUFFCOPY, C_NONE, C_NONE, C_NONE, C_LBRA, 11, 4, 0}, // same as ABR/ABL

	{obj.AXXX, C_NONE, C_NONE, C_NONE, C_NONE, 0, 4, 0},
}

var oprange [ALAST & obj.AMask][]Optab

var xcmp [C_NCLASS][C_NCLASS]bool

func span9(ctxt *obj.Link, cursym *obj.LSym) {
	p := cursym.Text
	if p == nil || p.Link == nil { // handle external functions and ELF section symbols
		return
	}
	ctxt.Cursym = cursym
	ctxt.Autosize = int32(p.To.Offset)

	if oprange[AANDN&obj.AMask] == nil {
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
			if (o.type_ == 16 || o.type_ == 17) && p.Pcond != nil {
				otxt = p.Pcond.Pc - c
				if otxt < -(1<<15)+10 || otxt >= (1<<15)-10 {
					q = ctxt.NewProg()
					q.Link = p.Link
					p.Link = q
					q.As = ABR
					q.To.Type = obj.TYPE_BRANCH
					q.Pcond = p.Pcond
					p.Pcond = q
					q = ctxt.NewProg()
					q.Link = p.Link
					p.Link = q
					q.As = ABR
					q.To.Type = obj.TYPE_BRANCH
					q.Pcond = q.Link.Link

					//addnop(p->link);
					//addnop(p);
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

	c += -c & (funcAlign - 1)
	cursym.Size = c

	/*
	 * lay out the code, emitting code and data relocations.
	 */

	cursym.Grow(cursym.Size)

	bp := cursym.P
	var i int32
	var out [6]uint32
	for p := cursym.Text.Link; p != nil; p = p.Link {
		ctxt.Pc = p.Pc
		ctxt.Curp = p
		o = oplook(ctxt, p)
		if int(o.size) > 4*len(out) {
			log.Fatalf("out array in span9 is too small, need at least %d for %v", o.size/4, p)
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
		if REG_V0 <= a.Reg && a.Reg <= REG_V31 {
			return C_VREG
		}
		if REG_VS0 <= a.Reg && a.Reg <= REG_VS63 {
			return C_VSREG
		}
		if REG_CR0 <= a.Reg && a.Reg <= REG_CR7 || a.Reg == REG_CR {
			return C_CREG
		}
		if REG_SPR0 <= a.Reg && a.Reg <= REG_SPR0+1023 {
			switch a.Reg {
			case REG_LR:
				return C_LR

			case REG_XER:
				return C_XER

			case REG_CTR:
				return C_CTR
			}

			return C_SPR
		}

		if REG_DCR0 <= a.Reg && a.Reg <= REG_DCR0+1023 {
			return C_SPR
		}
		if a.Reg == REG_FPSCR {
			return C_FPSCR
		}
		if a.Reg == REG_MSR {
			return C_MSR
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

			/* not sure why this barfs */
			return C_LCON

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
			return C_DCON
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
		return C_DCON

	case obj.TYPE_BRANCH:
		if a.Sym != nil && ctxt.Flag_dynlink {
			return C_LBRAPIC
		}
		return C_SBRA
	}

	return C_GOK
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
	a3 := C_NONE + 1
	if p.From3 != nil {
		a3 = int(p.From3.Class)
		if a3 == 0 {
			a3 = aclass(ctxt, p.From3) + 1
			p.From3.Class = int8(a3)
		}
	}

	a3--
	a4 := int(p.To.Class)
	if a4 == 0 {
		a4 = aclass(ctxt, &p.To) + 1
		p.To.Class = int8(a4)
	}

	a4--
	a2 := C_NONE
	if p.Reg != 0 {
		if REG_R0 <= p.Reg && p.Reg <= REG_R31 {
			a2 = C_REG
		} else if REG_V0 <= p.Reg && p.Reg <= REG_V31 {
			a2 = C_VREG
		} else if REG_VS0 <= p.Reg && p.Reg <= REG_VS63 {
			a2 = C_VSREG
		} else if REG_F0 <= p.Reg && p.Reg <= REG_F31 {
			a2 = C_FREG
		}
	}

	//print("oplook %v %d %d %d %d\n", p, a1, a2, a3, a4);
	ops := oprange[p.As&obj.AMask]
	c1 := &xcmp[a1]
	c3 := &xcmp[a3]
	c4 := &xcmp[a4]
	for i := range ops {
		op := &ops[i]
		if int(op.a2) == a2 && c1[op.a1] && c3[op.a3] && c4[op.a4] {
			p.Optab = uint16(cap(optab) - cap(ops) + i + 1)
			return op
		}
	}

	ctxt.Diag("illegal combination %v %v %v %v %v", p.As, DRconv(a1), DRconv(a2), DRconv(a3), DRconv(a4))
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

	case C_ADDCON:
		if b == C_ZCON || b == C_SCON {
			return true
		}

	case C_ANDCON:
		if b == C_ZCON || b == C_SCON {
			return true
		}

	case C_SPR:
		if b == C_LR || b == C_XER || b == C_CTR {
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

	case C_ANY:
		return true
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
	n = int(p1.a4) - int(p2.a4)
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
			log.Fatalf("instruction missing from switch in asm9.go:buildop: %v", r)

		case ADCBF: /* unary indexed: op (b+a); op (b) */
			opset(ADCBI, r0)

			opset(ADCBST, r0)
			opset(ADCBT, r0)
			opset(ADCBTST, r0)
			opset(ADCBZ, r0)
			opset(AICBI, r0)

		case AECOWX: /* indexed store: op s,(b+a); op s,(b) */
			opset(ASTWCCC, r0)
			opset(ASTBCCC, r0)

			opset(ASTDCCC, r0)

		case AREM: /* macro */
			opset(AREMCC, r0)

			opset(AREMV, r0)
			opset(AREMVCC, r0)

		case AREMU:
			opset(AREMU, r0)
			opset(AREMUCC, r0)
			opset(AREMUV, r0)
			opset(AREMUVCC, r0)

		case AREMD:
			opset(AREMDCC, r0)
			opset(AREMDV, r0)
			opset(AREMDVCC, r0)

		case AREMDU:
			opset(AREMDU, r0)
			opset(AREMDUCC, r0)
			opset(AREMDUV, r0)
			opset(AREMDUVCC, r0)

		case ADIVW: /* op Rb[,Ra],Rd */
			opset(AMULHW, r0)

			opset(AMULHWCC, r0)
			opset(AMULHWU, r0)
			opset(AMULHWUCC, r0)
			opset(AMULLWCC, r0)
			opset(AMULLWVCC, r0)
			opset(AMULLWV, r0)
			opset(ADIVWCC, r0)
			opset(ADIVWV, r0)
			opset(ADIVWVCC, r0)
			opset(ADIVWU, r0)
			opset(ADIVWUCC, r0)
			opset(ADIVWUV, r0)
			opset(ADIVWUVCC, r0)
			opset(AADDCC, r0)
			opset(AADDCV, r0)
			opset(AADDCVCC, r0)
			opset(AADDV, r0)
			opset(AADDVCC, r0)
			opset(AADDE, r0)
			opset(AADDECC, r0)
			opset(AADDEV, r0)
			opset(AADDEVCC, r0)
			opset(ACRAND, r0)
			opset(ACRANDN, r0)
			opset(ACREQV, r0)
			opset(ACRNAND, r0)
			opset(ACRNOR, r0)
			opset(ACROR, r0)
			opset(ACRORN, r0)
			opset(ACRXOR, r0)
			opset(AMULHD, r0)
			opset(AMULHDCC, r0)
			opset(AMULHDU, r0)
			opset(AMULHDUCC, r0)
			opset(AMULLD, r0)
			opset(AMULLDCC, r0)
			opset(AMULLDVCC, r0)
			opset(AMULLDV, r0)
			opset(ADIVD, r0)
			opset(ADIVDCC, r0)
			opset(ADIVDE, r0)
			opset(ADIVDEU, r0)
			opset(ADIVDECC, r0)
			opset(ADIVDEUCC, r0)
			opset(ADIVDVCC, r0)
			opset(ADIVDV, r0)
			opset(ADIVDU, r0)
			opset(ADIVDUCC, r0)
			opset(ADIVDUVCC, r0)
			opset(ADIVDUCC, r0)

		case AMOVBZ: /* lbz, stz, rlwm(r/r), lhz, lha, stz, and x variants */
			opset(AMOVH, r0)

			opset(AMOVHZ, r0)

		case AMOVBZU: /* lbz[x]u, stb[x]u, lhz[x]u, lha[x]u, sth[u]x, ld[x]u, std[u]x */
			opset(AMOVHU, r0)

			opset(AMOVHZU, r0)
			opset(AMOVWU, r0)
			opset(AMOVWZU, r0)
			opset(AMOVDU, r0)
			opset(AMOVMW, r0)

		case ALV: /* lvebx, lvehx, lvewx, lvx, lvxl, lvsl, lvsr */
			opset(ALVEBX, r0)
			opset(ALVEHX, r0)
			opset(ALVEWX, r0)
			opset(ALVX, r0)
			opset(ALVXL, r0)
			opset(ALVSL, r0)
			opset(ALVSR, r0)

		case ASTV: /* stvebx, stvehx, stvewx, stvx, stvxl */
			opset(ASTVEBX, r0)
			opset(ASTVEHX, r0)
			opset(ASTVEWX, r0)
			opset(ASTVX, r0)
			opset(ASTVXL, r0)

		case AVAND: /* vand, vandc, vnand */
			opset(AVANDL, r0)
			opset(AVANDC, r0)
			opset(AVNAND, r0)

		case AVOR: /* vor, vorc, vxor, vnor, veqv */
			opset(AVORL, r0)
			opset(AVORC, r0)
			opset(AVXOR, r0)
			opset(AVNOR, r0)
			opset(AVEQV, r0)

		case AVADDUM: /* vaddubm, vadduhm, vadduwm, vaddudm, vadduqm */
			opset(AVADDUBM, r0)
			opset(AVADDUHM, r0)
			opset(AVADDUWM, r0)
			opset(AVADDUDM, r0)
			opset(AVADDUQM, r0)

		case AVADDCU: /* vaddcuq, vaddcuw */
			opset(AVADDCUQ, r0)
			opset(AVADDCUW, r0)

		case AVADDUS: /* vaddubs, vadduhs, vadduws */
			opset(AVADDUBS, r0)
			opset(AVADDUHS, r0)
			opset(AVADDUWS, r0)

		case AVADDSS: /* vaddsbs, vaddshs, vaddsws */
			opset(AVADDSBS, r0)
			opset(AVADDSHS, r0)
			opset(AVADDSWS, r0)

		case AVADDE: /* vaddeuqm, vaddecuq */
			opset(AVADDEUQM, r0)
			opset(AVADDECUQ, r0)

		case AVSUBUM: /* vsububm, vsubuhm, vsubuwm, vsubudm, vsubuqm */
			opset(AVSUBUBM, r0)
			opset(AVSUBUHM, r0)
			opset(AVSUBUWM, r0)
			opset(AVSUBUDM, r0)
			opset(AVSUBUQM, r0)

		case AVSUBCU: /* vsubcuq, vsubcuw */
			opset(AVSUBCUQ, r0)
			opset(AVSUBCUW, r0)

		case AVSUBUS: /* vsububs, vsubuhs, vsubuws */
			opset(AVSUBUBS, r0)
			opset(AVSUBUHS, r0)
			opset(AVSUBUWS, r0)

		case AVSUBSS: /* vsubsbs, vsubshs, vsubsws */
			opset(AVSUBSBS, r0)
			opset(AVSUBSHS, r0)
			opset(AVSUBSWS, r0)

		case AVSUBE: /* vsubeuqm, vsubecuq */
			opset(AVSUBEUQM, r0)
			opset(AVSUBECUQ, r0)

		case AVR: /* vrlb, vrlh, vrlw, vrld */
			opset(AVRLB, r0)
			opset(AVRLH, r0)
			opset(AVRLW, r0)
			opset(AVRLD, r0)

		case AVS: /* vs[l,r], vs[l,r]o, vs[l,r]b, vs[l,r]h, vs[l,r]w, vs[l,r]d */
			opset(AVSLB, r0)
			opset(AVSLH, r0)
			opset(AVSLW, r0)
			opset(AVSL, r0)
			opset(AVSLO, r0)
			opset(AVSRB, r0)
			opset(AVSRH, r0)
			opset(AVSRW, r0)
			opset(AVSR, r0)
			opset(AVSRO, r0)
			opset(AVSLD, r0)
			opset(AVSRD, r0)

		case AVSA: /* vsrab, vsrah, vsraw, vsrad */
			opset(AVSRAB, r0)
			opset(AVSRAH, r0)
			opset(AVSRAW, r0)
			opset(AVSRAD, r0)

		case AVSOI: /* vsldoi */
			opset(AVSLDOI, r0)

		case AVCLZ: /* vclzb, vclzh, vclzw, vclzd */
			opset(AVCLZB, r0)
			opset(AVCLZH, r0)
			opset(AVCLZW, r0)
			opset(AVCLZD, r0)

		case AVPOPCNT: /* vpopcntb, vpopcnth, vpopcntw, vpopcntd */
			opset(AVPOPCNTB, r0)
			opset(AVPOPCNTH, r0)
			opset(AVPOPCNTW, r0)
			opset(AVPOPCNTD, r0)

		case AVCMPEQ: /* vcmpequb[.], vcmpequh[.], vcmpequw[.], vcmpequd[.] */
			opset(AVCMPEQUB, r0)
			opset(AVCMPEQUBCC, r0)
			opset(AVCMPEQUH, r0)
			opset(AVCMPEQUHCC, r0)
			opset(AVCMPEQUW, r0)
			opset(AVCMPEQUWCC, r0)
			opset(AVCMPEQUD, r0)
			opset(AVCMPEQUDCC, r0)

		case AVCMPGT: /* vcmpgt[u,s]b[.], vcmpgt[u,s]h[.], vcmpgt[u,s]w[.], vcmpgt[u,s]d[.] */
			opset(AVCMPGTUB, r0)
			opset(AVCMPGTUBCC, r0)
			opset(AVCMPGTUH, r0)
			opset(AVCMPGTUHCC, r0)
			opset(AVCMPGTUW, r0)
			opset(AVCMPGTUWCC, r0)
			opset(AVCMPGTUD, r0)
			opset(AVCMPGTUDCC, r0)
			opset(AVCMPGTSB, r0)
			opset(AVCMPGTSBCC, r0)
			opset(AVCMPGTSH, r0)
			opset(AVCMPGTSHCC, r0)
			opset(AVCMPGTSW, r0)
			opset(AVCMPGTSWCC, r0)
			opset(AVCMPGTSD, r0)
			opset(AVCMPGTSDCC, r0)

		case AVPERM: /* vperm */
			opset(AVPERM, r0)

		case AVSEL: /* vsel */
			opset(AVSEL, r0)

		case AVSPLT: /* vspltb, vsplth, vspltw */
			opset(AVSPLTB, r0)
			opset(AVSPLTH, r0)
			opset(AVSPLTW, r0)

		case AVSPLTI: /* vspltisb, vspltish, vspltisw */
			opset(AVSPLTISB, r0)
			opset(AVSPLTISH, r0)
			opset(AVSPLTISW, r0)

		case AVCIPH: /* vcipher, vcipherlast */
			opset(AVCIPHER, r0)
			opset(AVCIPHERLAST, r0)

		case AVNCIPH: /* vncipher, vncipherlast */
			opset(AVNCIPHER, r0)
			opset(AVNCIPHERLAST, r0)

		case AVSBOX: /* vsbox */
			opset(AVSBOX, r0)

		case AVSHASIGMA: /* vshasigmaw, vshasigmad */
			opset(AVSHASIGMAW, r0)
			opset(AVSHASIGMAD, r0)

		case ALXV: /* lxvd2x, lxvdsx, lxvw4x */
			opset(ALXVD2X, r0)
			opset(ALXVDSX, r0)
			opset(ALXVW4X, r0)

		case ASTXV: /* stxvd2x, stxvdsx, stxvw4x */
			opset(ASTXVD2X, r0)
			opset(ASTXVW4X, r0)

		case ALXS: /* lxsdx  */
			opset(ALXSDX, r0)

		case ASTXS: /* stxsdx */
			opset(ASTXSDX, r0)

		case ALXSI: /* lxsiwax, lxsiwzx  */
			opset(ALXSIWAX, r0)
			opset(ALXSIWZX, r0)

		case ASTXSI: /* stxsiwx */
			opset(ASTXSIWX, r0)

		case AMFVSR: /* mfvsrd, mfvsrwz */
			opset(AMFVSRD, r0)
			opset(AMFVSRWZ, r0)

		case AMTVSR: /* mtvsrd, mtvsrwa, mtvsrwz */
			opset(AMTVSRD, r0)
			opset(AMTVSRWA, r0)
			opset(AMTVSRWZ, r0)

		case AXXLAND: /* xxland, xxlandc, xxleqv, xxlnand */
			opset(AXXLANDQ, r0)
			opset(AXXLANDC, r0)
			opset(AXXLEQV, r0)
			opset(AXXLNAND, r0)

		case AXXLOR: /* xxlorc, xxlnor, xxlor, xxlxor */
			opset(AXXLORC, r0)
			opset(AXXLNOR, r0)
			opset(AXXLORQ, r0)
			opset(AXXLXOR, r0)

		case AXXSEL: /* xxsel */
			opset(AXXSEL, r0)

		case AXXMRG: /* xxmrghw, xxmrglw */
			opset(AXXMRGHW, r0)
			opset(AXXMRGLW, r0)

		case AXXSPLT: /* xxspltw */
			opset(AXXSPLTW, r0)

		case AXXPERM: /* xxpermdi */
			opset(AXXPERMDI, r0)

		case AXXSI: /* xxsldwi */
			opset(AXXSLDWI, r0)

		case AXSCV: /* xscvdpsp, xscvspdp, xscvdpspn, xscvspdpn */
			opset(AXSCVDPSP, r0)
			opset(AXSCVSPDP, r0)
			opset(AXSCVDPSPN, r0)
			opset(AXSCVSPDPN, r0)

		case AXVCV: /* xvcvdpsp, xvcvspdp */
			opset(AXVCVDPSP, r0)
			opset(AXVCVSPDP, r0)

		case AXSCVX: /* xscvdpsxds, xscvdpsxws, xscvdpuxds, xscvdpuxws */
			opset(AXSCVDPSXDS, r0)
			opset(AXSCVDPSXWS, r0)
			opset(AXSCVDPUXDS, r0)
			opset(AXSCVDPUXWS, r0)

		case AXSCVXP: /* xscvsxddp, xscvuxddp, xscvsxdsp, xscvuxdsp */
			opset(AXSCVSXDDP, r0)
			opset(AXSCVUXDDP, r0)
			opset(AXSCVSXDSP, r0)
			opset(AXSCVUXDSP, r0)

		case AXVCVX: /* xvcvdpsxds, xvcvdpsxws, xvcvdpuxds, xvcvdpuxws, xvcvspsxds, xvcvspsxws, xvcvspuxds, xvcvspuxws */
			opset(AXVCVDPSXDS, r0)
			opset(AXVCVDPSXWS, r0)
			opset(AXVCVDPUXDS, r0)
			opset(AXVCVDPUXWS, r0)
			opset(AXVCVSPSXDS, r0)
			opset(AXVCVSPSXWS, r0)
			opset(AXVCVSPUXDS, r0)
			opset(AXVCVSPUXWS, r0)

		case AXVCVXP: /* xvcvsxddp, xvcvsxwdp, xvcvuxddp, xvcvuxwdp, xvcvsxdsp, xvcvsxwsp, xvcvuxdsp, xvcvuxwsp */
			opset(AXVCVSXDDP, r0)
			opset(AXVCVSXWDP, r0)
			opset(AXVCVUXDDP, r0)
			opset(AXVCVUXWDP, r0)
			opset(AXVCVSXDSP, r0)
			opset(AXVCVSXWSP, r0)
			opset(AXVCVUXDSP, r0)
			opset(AXVCVUXWSP, r0)

		case AAND: /* logical op Rb,Rs,Ra; no literal */
			opset(AANDN, r0)

			opset(AANDNCC, r0)
			opset(AEQV, r0)
			opset(AEQVCC, r0)
			opset(ANAND, r0)
			opset(ANANDCC, r0)
			opset(ANOR, r0)
			opset(ANORCC, r0)
			opset(AORCC, r0)
			opset(AORN, r0)
			opset(AORNCC, r0)
			opset(AXORCC, r0)

		case AADDME: /* op Ra, Rd */
			opset(AADDMECC, r0)

			opset(AADDMEV, r0)
			opset(AADDMEVCC, r0)
			opset(AADDZE, r0)
			opset(AADDZECC, r0)
			opset(AADDZEV, r0)
			opset(AADDZEVCC, r0)
			opset(ASUBME, r0)
			opset(ASUBMECC, r0)
			opset(ASUBMEV, r0)
			opset(ASUBMEVCC, r0)
			opset(ASUBZE, r0)
			opset(ASUBZECC, r0)
			opset(ASUBZEV, r0)
			opset(ASUBZEVCC, r0)

		case AADDC:
			opset(AADDCCC, r0)

		case ABEQ:
			opset(ABGE, r0)
			opset(ABGT, r0)
			opset(ABLE, r0)
			opset(ABLT, r0)
			opset(ABNE, r0)
			opset(ABVC, r0)
			opset(ABVS, r0)

		case ABR:
			opset(ABL, r0)

		case ABC:
			opset(ABCL, r0)

		case AEXTSB: /* op Rs, Ra */
			opset(AEXTSBCC, r0)

			opset(AEXTSH, r0)
			opset(AEXTSHCC, r0)
			opset(ACNTLZW, r0)
			opset(ACNTLZWCC, r0)
			opset(ACNTLZD, r0)
			opset(AEXTSW, r0)
			opset(AEXTSWCC, r0)
			opset(ACNTLZDCC, r0)

		case AFABS: /* fop [s,]d */
			opset(AFABSCC, r0)

			opset(AFNABS, r0)
			opset(AFNABSCC, r0)
			opset(AFNEG, r0)
			opset(AFNEGCC, r0)
			opset(AFRSP, r0)
			opset(AFRSPCC, r0)
			opset(AFCTIW, r0)
			opset(AFCTIWCC, r0)
			opset(AFCTIWZ, r0)
			opset(AFCTIWZCC, r0)
			opset(AFCTID, r0)
			opset(AFCTIDCC, r0)
			opset(AFCTIDZ, r0)
			opset(AFCTIDZCC, r0)
			opset(AFCFID, r0)
			opset(AFCFIDCC, r0)
			opset(AFCFIDU, r0)
			opset(AFCFIDUCC, r0)
			opset(AFRES, r0)
			opset(AFRESCC, r0)
			opset(AFRIM, r0)
			opset(AFRIMCC, r0)
			opset(AFRIP, r0)
			opset(AFRIPCC, r0)
			opset(AFRIZ, r0)
			opset(AFRIZCC, r0)
			opset(AFRSQRTE, r0)
			opset(AFRSQRTECC, r0)
			opset(AFSQRT, r0)
			opset(AFSQRTCC, r0)
			opset(AFSQRTS, r0)
			opset(AFSQRTSCC, r0)

		case AFADD:
			opset(AFADDS, r0)
			opset(AFADDCC, r0)
			opset(AFADDSCC, r0)
			opset(AFDIV, r0)
			opset(AFDIVS, r0)
			opset(AFDIVCC, r0)
			opset(AFDIVSCC, r0)
			opset(AFSUB, r0)
			opset(AFSUBS, r0)
			opset(AFSUBCC, r0)
			opset(AFSUBSCC, r0)

		case AFMADD:
			opset(AFMADDCC, r0)
			opset(AFMADDS, r0)
			opset(AFMADDSCC, r0)
			opset(AFMSUB, r0)
			opset(AFMSUBCC, r0)
			opset(AFMSUBS, r0)
			opset(AFMSUBSCC, r0)
			opset(AFNMADD, r0)
			opset(AFNMADDCC, r0)
			opset(AFNMADDS, r0)
			opset(AFNMADDSCC, r0)
			opset(AFNMSUB, r0)
			opset(AFNMSUBCC, r0)
			opset(AFNMSUBS, r0)
			opset(AFNMSUBSCC, r0)
			opset(AFSEL, r0)
			opset(AFSELCC, r0)

		case AFMUL:
			opset(AFMULS, r0)
			opset(AFMULCC, r0)
			opset(AFMULSCC, r0)

		case AFCMPO:
			opset(AFCMPU, r0)

		case AISEL:
			opset(AISEL, r0)

		case AMTFSB0:
			opset(AMTFSB0CC, r0)
			opset(AMTFSB1, r0)
			opset(AMTFSB1CC, r0)

		case ANEG: /* op [Ra,] Rd */
			opset(ANEGCC, r0)

			opset(ANEGV, r0)
			opset(ANEGVCC, r0)

		case AOR: /* or/xor Rb,Rs,Ra; ori/xori $uimm,Rs,Ra; oris/xoris $uimm,Rs,Ra */
			opset(AXOR, r0)

		case ASLW:
			opset(ASLWCC, r0)
			opset(ASRW, r0)
			opset(ASRWCC, r0)

		case ASLD:
			opset(ASLDCC, r0)
			opset(ASRD, r0)
			opset(ASRDCC, r0)

		case ASRAW: /* sraw Rb,Rs,Ra; srawi sh,Rs,Ra */
			opset(ASRAWCC, r0)

		case ASRAD: /* sraw Rb,Rs,Ra; srawi sh,Rs,Ra */
			opset(ASRADCC, r0)

		case ASUB: /* SUB Ra,Rb,Rd => subf Rd,ra,rb */
			opset(ASUB, r0)

			opset(ASUBCC, r0)
			opset(ASUBV, r0)
			opset(ASUBVCC, r0)
			opset(ASUBCCC, r0)
			opset(ASUBCV, r0)
			opset(ASUBCVCC, r0)
			opset(ASUBE, r0)
			opset(ASUBECC, r0)
			opset(ASUBEV, r0)
			opset(ASUBEVCC, r0)

		case ASYNC:
			opset(AISYNC, r0)
			opset(ALWSYNC, r0)
			opset(APTESYNC, r0)
			opset(ATLBSYNC, r0)

		case ARLWMI:
			opset(ARLWMICC, r0)
			opset(ARLWNM, r0)
			opset(ARLWNMCC, r0)

		case ARLDMI:
			opset(ARLDMICC, r0)
			opset(ARLDIMI, r0)
			opset(ARLDIMICC, r0)

		case ARLDC:
			opset(ARLDCCC, r0)

		case ARLDCL:
			opset(ARLDCR, r0)
			opset(ARLDCLCC, r0)
			opset(ARLDCRCC, r0)

		case ARLDICL:
			opset(ARLDICLCC, r0)
			opset(ARLDICR, r0)
			opset(ARLDICRCC, r0)

		case AFMOVD:
			opset(AFMOVDCC, r0)
			opset(AFMOVDU, r0)
			opset(AFMOVS, r0)
			opset(AFMOVSU, r0)

		case AECIWX:
			opset(ALBAR, r0)
			opset(ALWAR, r0)
			opset(ALDAR, r0)

		case ASYSCALL: /* just the op; flow of control */
			opset(ARFI, r0)

			opset(ARFCI, r0)
			opset(ARFID, r0)
			opset(AHRFID, r0)

		case AMOVHBR:
			opset(AMOVWBR, r0)
			opset(AMOVDBR, r0)

		case ASLBMFEE:
			opset(ASLBMFEV, r0)

		case ATW:
			opset(ATD, r0)

		case ATLBIE:
			opset(ASLBIE, r0)
			opset(ATLBIEL, r0)

		case AEIEIO:
			opset(ASLBIA, r0)

		case ACMP:
			opset(ACMPW, r0)

		case ACMPU:
			opset(ACMPWU, r0)

		case AADD,
			AANDCC, /* and. Rb,Rs,Ra; andi. $uimm,Rs,Ra; andis. $uimm,Rs,Ra */
			AFMOVSX,
			AFMOVSZ,
			ALSW,
			AMOVW,
			/* load/store/move word with sign extension; special 32-bit move; move 32-bit literals */
			AMOVWZ, /* load/store/move word with zero extension; move 32-bit literals  */
			AMOVD,  /* load/store/move 64-bit values, including 32-bit literals with/without sign-extension */
			AMOVB,  /* macro: move byte with sign extension */
			AMOVBU, /* macro: move byte with sign extension & update */
			AMOVFL,
			AMULLW,
			/* op $s[,r2],r3; op r1[,r2],r3; no cc/v */
			ASUBC, /* op r1,$s,r3; op r1[,r2],r3 */
			ASTSW,
			ASLBMTE,
			AWORD,
			ADWORD,
			obj.ANOP,
			obj.ATEXT,
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

func OPVXX1(o uint32, xo uint32, oe uint32) uint32 {
	return o<<26 | xo<<1 | oe<<11
}

func OPVXX2(o uint32, xo uint32, oe uint32) uint32 {
	return o<<26 | xo<<2 | oe<<11
}

func OPVXX3(o uint32, xo uint32, oe uint32) uint32 {
	return o<<26 | xo<<3 | oe<<11
}

func OPVXX4(o uint32, xo uint32, oe uint32) uint32 {
	return o<<26 | xo<<4 | oe<<11
}

func OPVX(o uint32, xo uint32, oe uint32, rc uint32) uint32 {
	return o<<26 | xo | oe<<11 | rc&1
}

func OPVC(o uint32, xo uint32, oe uint32, rc uint32) uint32 {
	return o<<26 | xo | oe<<11 | (rc&1)<<10
}

func OPVCC(o uint32, xo uint32, oe uint32, rc uint32) uint32 {
	return o<<26 | xo<<1 | oe<<10 | rc&1
}

func OPCC(o uint32, xo uint32, rc uint32) uint32 {
	return OPVCC(o, xo, 0, rc)
}

func OP(o uint32, xo uint32) uint32 {
	return OPVCC(o, xo, 0, 0)
}

/* the order is dest, a/s, b/imm for both arithmetic and logical operations */
func AOP_RRR(op uint32, d uint32, a uint32, b uint32) uint32 {
	return op | (d&31)<<21 | (a&31)<<16 | (b&31)<<11
}

/* VX-form 2-register operands, r/r/none */
func AOP_RR(op uint32, d uint32, a uint32) uint32 {
	return op | (d&31)<<21 | (a&31)<<11
}

/* VA-form 4-register operands */
func AOP_RRRR(op uint32, d uint32, a uint32, b uint32, c uint32) uint32 {
	return op | (d&31)<<21 | (a&31)<<16 | (b&31)<<11 | (c&31)<<6
}

func AOP_IRR(op uint32, d uint32, a uint32, simm uint32) uint32 {
	return op | (d&31)<<21 | (a&31)<<16 | simm&0xFFFF
}

/* VX-form 2-register + UIM operands */
func AOP_VIRR(op uint32, d uint32, a uint32, simm uint32) uint32 {
	return op | (d&31)<<21 | (simm&0xFFFF)<<16 | (a&31)<<11
}

/* VX-form 2-register + ST + SIX operands */
func AOP_IIRR(op uint32, d uint32, a uint32, sbit uint32, simm uint32) uint32 {
	return op | (d&31)<<21 | (a&31)<<16 | (sbit&1)<<15 | (simm&0xF)<<11
}

/* VA-form 3-register + SHB operands */
func AOP_IRRR(op uint32, d uint32, a uint32, b uint32, simm uint32) uint32 {
	return op | (d&31)<<21 | (a&31)<<16 | (b&31)<<11 | (simm&0xF)<<6
}

/* VX-form 1-register + SIM operands */
func AOP_IR(op uint32, d uint32, simm uint32) uint32 {
	return op | (d&31)<<21 | (simm&31)<<16
}

/* XX1-form 3-register operands, 1 VSR operand */
func AOP_XX1(op uint32, d uint32, a uint32, b uint32) uint32 {
	/* For the XX-form encodings, we need the VSX register number to be exactly */
	/* between 0-63, so we can properly set the rightmost bits. */
	r := d - REG_VS0
	return op | (r&31)<<21 | (a&31)<<16 | (b&31)<<11 | (r&32)>>5
}

/* XX2-form 3-register operands, 2 VSR operands */
func AOP_XX2(op uint32, d uint32, a uint32, b uint32) uint32 {
	xt := d - REG_VS0
	xb := b - REG_VS0
	return op | (xt&31)<<21 | (a&3)<<16 | (xb&31)<<11 | (xb&32)>>4 | (xt&32)>>5
}

/* XX3-form 3 VSR operands */
func AOP_XX3(op uint32, d uint32, a uint32, b uint32) uint32 {
	xt := d - REG_VS0
	xa := a - REG_VS0
	xb := b - REG_VS0
	return op | (xt&31)<<21 | (xa&31)<<16 | (xb&31)<<11 | (xa&32)>>3 | (xb&32)>>4 | (xt&32)>>5
}

/* XX3-form 3 VSR operands + immediate */
func AOP_XX3I(op uint32, d uint32, a uint32, b uint32, c uint32) uint32 {
	xt := d - REG_VS0
	xa := a - REG_VS0
	xb := b - REG_VS0
	return op | (xt&31)<<21 | (xa&31)<<16 | (xb&31)<<11 | (c&3)<<8 | (xa&32)>>3 | (xb&32)>>4 | (xt&32)>>5
}

/* XX4-form, 4 VSR operands */
func AOP_XX4(op uint32, d uint32, a uint32, b uint32, c uint32) uint32 {
	xt := d - REG_VS0
	xa := a - REG_VS0
	xb := b - REG_VS0
	xc := c - REG_VS0
	return op | (xt&31)<<21 | (xa&31)<<16 | (xb&31)<<11 | (xc&31)<<6 | (xc&32)>>2 | (xa&32)>>3 | (xb&32)>>4 | (xt&32)>>5
}

func LOP_RRR(op uint32, a uint32, s uint32, b uint32) uint32 {
	return op | (s&31)<<21 | (a&31)<<16 | (b&31)<<11
}

func LOP_IRR(op uint32, a uint32, s uint32, uimm uint32) uint32 {
	return op | (s&31)<<21 | (a&31)<<16 | uimm&0xFFFF
}

func OP_BR(op uint32, li uint32, aa uint32) uint32 {
	return op | li&0x03FFFFFC | aa<<1
}

func OP_BC(op uint32, bo uint32, bi uint32, bd uint32, aa uint32) uint32 {
	return op | (bo&0x1F)<<21 | (bi&0x1F)<<16 | bd&0xFFFC | aa<<1
}

func OP_BCR(op uint32, bo uint32, bi uint32) uint32 {
	return op | (bo&0x1F)<<21 | (bi&0x1F)<<16
}

func OP_RLW(op uint32, a uint32, s uint32, sh uint32, mb uint32, me uint32) uint32 {
	return op | (s&31)<<21 | (a&31)<<16 | (sh&31)<<11 | (mb&31)<<6 | (me&31)<<1
}

func AOP_ISEL(op uint32, t uint32, a uint32, b uint32, bc uint32) uint32 {
	return op | (t&31)<<21 | (a&31)<<16 | (b&31)<<11 | (bc&0x1F)<<6
}

const (
	/* each rhs is OPVCC(_, _, _, _) */
	OP_ADD    = 31<<26 | 266<<1 | 0<<10 | 0
	OP_ADDI   = 14<<26 | 0<<1 | 0<<10 | 0
	OP_ADDIS  = 15<<26 | 0<<1 | 0<<10 | 0
	OP_ANDI   = 28<<26 | 0<<1 | 0<<10 | 0
	OP_EXTSB  = 31<<26 | 954<<1 | 0<<10 | 0
	OP_EXTSH  = 31<<26 | 922<<1 | 0<<10 | 0
	OP_EXTSW  = 31<<26 | 986<<1 | 0<<10 | 0
	OP_ISEL   = 31<<26 | 15<<1 | 0<<10 | 0
	OP_MCRF   = 19<<26 | 0<<1 | 0<<10 | 0
	OP_MCRFS  = 63<<26 | 64<<1 | 0<<10 | 0
	OP_MCRXR  = 31<<26 | 512<<1 | 0<<10 | 0
	OP_MFCR   = 31<<26 | 19<<1 | 0<<10 | 0
	OP_MFFS   = 63<<26 | 583<<1 | 0<<10 | 0
	OP_MFMSR  = 31<<26 | 83<<1 | 0<<10 | 0
	OP_MFSPR  = 31<<26 | 339<<1 | 0<<10 | 0
	OP_MFSR   = 31<<26 | 595<<1 | 0<<10 | 0
	OP_MFSRIN = 31<<26 | 659<<1 | 0<<10 | 0
	OP_MTCRF  = 31<<26 | 144<<1 | 0<<10 | 0
	OP_MTFSF  = 63<<26 | 711<<1 | 0<<10 | 0
	OP_MTFSFI = 63<<26 | 134<<1 | 0<<10 | 0
	OP_MTMSR  = 31<<26 | 146<<1 | 0<<10 | 0
	OP_MTMSRD = 31<<26 | 178<<1 | 0<<10 | 0
	OP_MTSPR  = 31<<26 | 467<<1 | 0<<10 | 0
	OP_MTSR   = 31<<26 | 210<<1 | 0<<10 | 0
	OP_MTSRIN = 31<<26 | 242<<1 | 0<<10 | 0
	OP_MULLW  = 31<<26 | 235<<1 | 0<<10 | 0
	OP_MULLD  = 31<<26 | 233<<1 | 0<<10 | 0
	OP_OR     = 31<<26 | 444<<1 | 0<<10 | 0
	OP_ORI    = 24<<26 | 0<<1 | 0<<10 | 0
	OP_ORIS   = 25<<26 | 0<<1 | 0<<10 | 0
	OP_RLWINM = 21<<26 | 0<<1 | 0<<10 | 0
	OP_SUBF   = 31<<26 | 40<<1 | 0<<10 | 0
	OP_RLDIC  = 30<<26 | 4<<1 | 0<<10 | 0
	OP_RLDICR = 30<<26 | 2<<1 | 0<<10 | 0
	OP_RLDICL = 30<<26 | 0<<1 | 0<<10 | 0
)

func oclass(a *obj.Addr) int {
	return int(a.Class) - 1
}

const (
	D_FORM = iota
	DS_FORM
)

// opform returns the form (D_FORM or DS_FORM) of an instruction. Used to decide on
// which relocation to use with a load or store and only supports the needed
// instructions.
func opform(ctxt *obj.Link, insn uint32) int {
	switch insn {
	default:
		ctxt.Diag("bad insn in loadform: %x", insn)
	case OPVCC(58, 0, 0, 0), // ld
		OPVCC(58, 0, 0, 0) | 1<<1, // lwa
		OPVCC(62, 0, 0, 0):        // std
		return DS_FORM
	case OP_ADDI, // add
		OPVCC(32, 0, 0, 0), // lwz
		OPVCC(42, 0, 0, 0), // lha
		OPVCC(40, 0, 0, 0), // lhz
		OPVCC(34, 0, 0, 0), // lbz
		OPVCC(50, 0, 0, 0), // lfd
		OPVCC(48, 0, 0, 0), // lfs
		OPVCC(36, 0, 0, 0), // stw
		OPVCC(44, 0, 0, 0), // sth
		OPVCC(38, 0, 0, 0), // stb
		OPVCC(54, 0, 0, 0), // stfd
		OPVCC(52, 0, 0, 0): // stfs
		return D_FORM
	}
	return 0
}

// Encode instructions and create relocation for accessing s+d according to the
// instruction op with source or destination (as appropriate) register reg.
func symbolAccess(ctxt *obj.Link, s *obj.LSym, d int64, reg int16, op uint32) (o1, o2 uint32) {
	var base uint32
	form := opform(ctxt, op)
	if ctxt.Flag_shared {
		base = REG_R2
	} else {
		base = REG_R0
	}
	o1 = AOP_IRR(OP_ADDIS, REGTMP, base, 0)
	o2 = AOP_IRR(op, uint32(reg), REGTMP, 0)
	rel := obj.Addrel(ctxt.Cursym)
	rel.Off = int32(ctxt.Pc)
	rel.Siz = 8
	rel.Sym = s
	rel.Add = d
	if ctxt.Flag_shared {
		switch form {
		case D_FORM:
			rel.Type = obj.R_ADDRPOWER_TOCREL
		case DS_FORM:
			rel.Type = obj.R_ADDRPOWER_TOCREL_DS
		}

	} else {
		switch form {
		case D_FORM:
			rel.Type = obj.R_ADDRPOWER
		case DS_FORM:
			rel.Type = obj.R_ADDRPOWER_DS
		}
	}
	return
}

/*
 * 32-bit masks
 */
func getmask(m []byte, v uint32) bool {
	m[1] = 0
	m[0] = m[1]
	if v != ^uint32(0) && v&(1<<31) != 0 && v&1 != 0 { /* MB > ME */
		if getmask(m, ^v) {
			i := int(m[0])
			m[0] = m[1] + 1
			m[1] = byte(i - 1)
			return true
		}

		return false
	}

	for i := 0; i < 32; i++ {
		if v&(1<<uint(31-i)) != 0 {
			m[0] = byte(i)
			for {
				m[1] = byte(i)
				i++
				if i >= 32 || v&(1<<uint(31-i)) == 0 {
					break
				}
			}

			for ; i < 32; i++ {
				if v&(1<<uint(31-i)) != 0 {
					return false
				}
			}
			return true
		}
	}

	return false
}

func maskgen(ctxt *obj.Link, p *obj.Prog, m []byte, v uint32) {
	if !getmask(m, v) {
		ctxt.Diag("cannot generate mask #%x\n%v", v, p)
	}
}

/*
 * 64-bit masks (rldic etc)
 */
func getmask64(m []byte, v uint64) bool {
	m[1] = 0
	m[0] = m[1]
	for i := 0; i < 64; i++ {
		if v&(uint64(1)<<uint(63-i)) != 0 {
			m[0] = byte(i)
			for {
				m[1] = byte(i)
				i++
				if i >= 64 || v&(uint64(1)<<uint(63-i)) == 0 {
					break
				}
			}

			for ; i < 64; i++ {
				if v&(uint64(1)<<uint(63-i)) != 0 {
					return false
				}
			}
			return true
		}
	}

	return false
}

func maskgen64(ctxt *obj.Link, p *obj.Prog, m []byte, v uint64) {
	if !getmask64(m, v) {
		ctxt.Diag("cannot generate mask #%x\n%v", v, p)
	}
}

func loadu32(r int, d int64) uint32 {
	v := int32(d >> 16)
	if isuint32(uint64(d)) {
		return LOP_IRR(OP_ORIS, uint32(r), REGZERO, uint32(v))
	}
	return AOP_IRR(OP_ADDIS, uint32(r), REGZERO, uint32(v))
}

func high16adjusted(d int32) uint16 {
	if d&0x8000 != 0 {
		return uint16((d >> 16) + 1)
	}
	return uint16(d >> 16)
}

func asmout(ctxt *obj.Link, p *obj.Prog, o *Optab, out []uint32) {
	o1 := uint32(0)
	o2 := uint32(0)
	o3 := uint32(0)
	o4 := uint32(0)
	o5 := uint32(0)

	//print("%v => case %d\n", p, o->type);
	switch o.type_ {
	default:
		ctxt.Diag("unknown type %d", o.type_)
		prasm(p)

	case 0: /* pseudo ops */
		break

	case 1: /* mov r1,r2 ==> OR Rs,Rs,Ra */
		if p.To.Reg == REGZERO && p.From.Type == obj.TYPE_CONST {
			v := regoff(ctxt, &p.From)
			if r0iszero != 0 /*TypeKind(100016)*/ && v != 0 {
				//nerrors--;
				ctxt.Diag("literal operation on R0\n%v", p)
			}

			o1 = LOP_IRR(OP_ADDI, REGZERO, REGZERO, uint32(v))
			break
		}

		o1 = LOP_RRR(OP_OR, uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.From.Reg))

	case 2: /* int/cr/fp op Rb,[Ra],Rd */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(p.From.Reg))

	case 3: /* mov $soreg/addcon/ucon, r ==> addis/addi $i,reg',r */
		d := vregoff(ctxt, &p.From)

		v := int32(d)
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		if r0iszero != 0 /*TypeKind(100016)*/ && p.To.Reg == 0 && (r != 0 || v != 0) {
			ctxt.Diag("literal operation on R0\n%v", p)
		}
		a := OP_ADDI
		if o.a1 == C_UCON {
			if d&0xffff != 0 {
				log.Fatalf("invalid handling of %v", p)
			}
			v >>= 16
			if r == REGZERO && isuint32(uint64(d)) {
				o1 = LOP_IRR(OP_ORIS, uint32(p.To.Reg), REGZERO, uint32(v))
				break
			}

			a = OP_ADDIS
		} else {
			if int64(int16(d)) != d {
				log.Fatalf("invalid handling of %v", p)
			}
		}

		o1 = AOP_IRR(uint32(a), uint32(p.To.Reg), uint32(r), uint32(v))

	case 4: /* add/mul $scon,[r1],r2 */
		v := regoff(ctxt, &p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		if r0iszero != 0 /*TypeKind(100016)*/ && p.To.Reg == 0 {
			ctxt.Diag("literal operation on R0\n%v", p)
		}
		if int32(int16(v)) != v {
			log.Fatalf("mishandled instruction %v", p)
		}
		o1 = AOP_IRR(opirr(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(v))

	case 5: /* syscall */
		o1 = oprrr(ctxt, p.As)

	case 6: /* logical op Rb,[Rs,]Ra; no literal */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = LOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(p.From.Reg))

	case 7: /* mov r, soreg ==> stw o(r) */
		r := int(p.To.Reg)

		if r == 0 {
			r = int(o.param)
		}
		v := regoff(ctxt, &p.To)
		if p.To.Type == obj.TYPE_MEM && p.To.Index != 0 {
			if v != 0 {
				ctxt.Diag("illegal indexed instruction\n%v", p)
			}
			if ctxt.Flag_shared && r == REG_R13 {
				rel := obj.Addrel(ctxt.Cursym)
				rel.Off = int32(ctxt.Pc)
				rel.Siz = 4
				// This (and the matching part in the load case
				// below) are the only places in the ppc64 toolchain
				// that knows the name of the tls variable. Possibly
				// we could add some assembly syntax so that the name
				// of the variable does not have to be assumed.
				rel.Sym = obj.Linklookup(ctxt, "runtime.tls_g", 0)
				rel.Type = obj.R_POWER_TLS
			}
			o1 = AOP_RRR(opstorex(ctxt, p.As), uint32(p.From.Reg), uint32(p.To.Index), uint32(r))
		} else {
			if int32(int16(v)) != v {
				log.Fatalf("mishandled instruction %v", p)
			}
			o1 = AOP_IRR(opstore(ctxt, p.As), uint32(p.From.Reg), uint32(r), uint32(v))
		}

	case 8: /* mov soreg, r ==> lbz/lhz/lwz o(r) */
		r := int(p.From.Reg)

		if r == 0 {
			r = int(o.param)
		}
		v := regoff(ctxt, &p.From)
		if p.From.Type == obj.TYPE_MEM && p.From.Index != 0 {
			if v != 0 {
				ctxt.Diag("illegal indexed instruction\n%v", p)
			}
			if ctxt.Flag_shared && r == REG_R13 {
				rel := obj.Addrel(ctxt.Cursym)
				rel.Off = int32(ctxt.Pc)
				rel.Siz = 4
				rel.Sym = obj.Linklookup(ctxt, "runtime.tls_g", 0)
				rel.Type = obj.R_POWER_TLS
			}
			o1 = AOP_RRR(oploadx(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Index), uint32(r))
		} else {
			if int32(int16(v)) != v {
				log.Fatalf("mishandled instruction %v", p)
			}
			o1 = AOP_IRR(opload(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(v))
		}

	case 9: /* movb soreg, r ==> lbz o(r),r2; extsb r2,r2 */
		r := int(p.From.Reg)

		if r == 0 {
			r = int(o.param)
		}
		v := regoff(ctxt, &p.From)
		if p.From.Type == obj.TYPE_MEM && p.From.Index != 0 {
			if v != 0 {
				ctxt.Diag("illegal indexed instruction\n%v", p)
			}
			o1 = AOP_RRR(oploadx(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Index), uint32(r))
		} else {
			o1 = AOP_IRR(opload(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(v))
		}
		o2 = LOP_RRR(OP_EXTSB, uint32(p.To.Reg), uint32(p.To.Reg), 0)

	case 10: /* sub Ra,[Rb],Rd => subf Rd,Ra,Rb */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(r))

	case 11: /* br/bl lbra */
		v := int32(0)

		if p.Pcond != nil {
			v = int32(p.Pcond.Pc - p.Pc)
			if v&03 != 0 {
				ctxt.Diag("odd branch target address\n%v", p)
				v &^= 03
			}

			if v < -(1<<25) || v >= 1<<24 {
				ctxt.Diag("branch too far\n%v", p)
			}
		}

		o1 = OP_BR(opirr(ctxt, p.As), uint32(v), 0)
		if p.To.Sym != nil {
			rel := obj.Addrel(ctxt.Cursym)
			rel.Off = int32(ctxt.Pc)
			rel.Siz = 4
			rel.Sym = p.To.Sym
			v += int32(p.To.Offset)
			if v&03 != 0 {
				ctxt.Diag("odd branch target address\n%v", p)
				v &^= 03
			}

			rel.Add = int64(v)
			rel.Type = obj.R_CALLPOWER
		}
		o2 = 0x60000000 // nop, sometimes overwritten by ld r2, 24(r1) when dynamic linking

	case 12: /* movb r,r (extsb); movw r,r (extsw) */
		if p.To.Reg == REGZERO && p.From.Type == obj.TYPE_CONST {
			v := regoff(ctxt, &p.From)
			if r0iszero != 0 /*TypeKind(100016)*/ && v != 0 {
				ctxt.Diag("literal operation on R0\n%v", p)
			}

			o1 = LOP_IRR(OP_ADDI, REGZERO, REGZERO, uint32(v))
			break
		}

		if p.As == AMOVW {
			o1 = LOP_RRR(OP_EXTSW, uint32(p.To.Reg), uint32(p.From.Reg), 0)
		} else {
			o1 = LOP_RRR(OP_EXTSB, uint32(p.To.Reg), uint32(p.From.Reg), 0)
		}

	case 13: /* mov[bhw]z r,r; uses rlwinm not andi. to avoid changing CC */
		if p.As == AMOVBZ {
			o1 = OP_RLW(OP_RLWINM, uint32(p.To.Reg), uint32(p.From.Reg), 0, 24, 31)
		} else if p.As == AMOVH {
			o1 = LOP_RRR(OP_EXTSH, uint32(p.To.Reg), uint32(p.From.Reg), 0)
		} else if p.As == AMOVHZ {
			o1 = OP_RLW(OP_RLWINM, uint32(p.To.Reg), uint32(p.From.Reg), 0, 16, 31)
		} else if p.As == AMOVWZ {
			o1 = OP_RLW(OP_RLDIC, uint32(p.To.Reg), uint32(p.From.Reg), 0, 0, 0) | 1<<5 /* MB=32 */
		} else {
			ctxt.Diag("internal: bad mov[bhw]z\n%v", p)
		}

	case 14: /* rldc[lr] Rb,Rs,$mask,Ra -- left, right give different masks */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		d := vregoff(ctxt, p.From3)
		var a int
		switch p.As {

		// These opcodes expect a mask operand that has to be converted into the
		// appropriate operand.  The way these were defined, not all valid masks are possible.
		// Left here for compatibility in case they were used or generated.
		case ARLDCL, ARLDCLCC:
			var mask [2]uint8
			maskgen64(ctxt, p, mask[:], uint64(d))

			a = int(mask[0]) /* MB */
			if mask[1] != 63 {
				ctxt.Diag("invalid mask for rotate: %x (end != bit 63)\n%v", uint64(d), p)
			}

		case ARLDCR, ARLDCRCC:
			var mask [2]uint8
			maskgen64(ctxt, p, mask[:], uint64(d))

			a = int(mask[1]) /* ME */
			if mask[0] != 0 {
				ctxt.Diag("invalid mask for rotate: %x (start != 0)\n%v", uint64(d), p)
			}

		// These opcodes use a shift count like the ppc64 asm, no mask conversion done
		case ARLDICR, ARLDICRCC, ARLDICL, ARLDICLCC:
			a = int(d)

		default:
			ctxt.Diag("unexpected op in rldc case\n%v", p)
			a = 0
		}

		o1 = LOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(p.From.Reg))
		o1 |= (uint32(a) & 31) << 6
		if a&0x20 != 0 {
			o1 |= 1 << 5 /* mb[5] is top bit */
		}

	case 17, /* bc bo,bi,lbra (same for now) */
		16: /* bc bo,bi,sbra */
		a := 0

		r := int(p.Reg)

		if p.From.Type == obj.TYPE_CONST {
			a = int(regoff(ctxt, &p.From))
		} else if p.From.Type == obj.TYPE_REG {
			if r != 0 {
				ctxt.Diag("unexpected register setting for branch with CR: %d\n", r)
			}
			// BI values for the CR
			switch p.From.Reg {
			case REG_CR0:
				r = BI_CR0
			case REG_CR1:
				r = BI_CR1
			case REG_CR2:
				r = BI_CR2
			case REG_CR3:
				r = BI_CR3
			case REG_CR4:
				r = BI_CR4
			case REG_CR5:
				r = BI_CR5
			case REG_CR6:
				r = BI_CR6
			case REG_CR7:
				r = BI_CR7
			default:
				ctxt.Diag("unrecognized register: expecting CR\n")
			}
		}
		v := int32(0)
		if p.Pcond != nil {
			v = int32(p.Pcond.Pc - p.Pc)
		}
		if v&03 != 0 {
			ctxt.Diag("odd branch target address\n%v", p)
			v &^= 03
		}

		if v < -(1<<16) || v >= 1<<15 {
			ctxt.Diag("branch too far\n%v", p)
		}
		o1 = OP_BC(opirr(ctxt, p.As), uint32(a), uint32(r), uint32(v), 0)

	case 15: /* br/bl (r) => mov r,lr; br/bl (lr) */
		var v int32
		if p.As == ABC || p.As == ABCL {
			v = regoff(ctxt, &p.To) & 31
		} else {
			v = 20 /* unconditional */
		}
		o1 = AOP_RRR(OP_MTSPR, uint32(p.To.Reg), 0, 0) | (REG_LR&0x1f)<<16 | ((REG_LR>>5)&0x1f)<<11
		o2 = OPVCC(19, 16, 0, 0)
		if p.As == ABL || p.As == ABCL {
			o2 |= 1
		}
		o2 = OP_BCR(o2, uint32(v), uint32(p.To.Index))

	case 18: /* br/bl (lr/ctr); bc/bcl bo,bi,(lr/ctr) */
		var v int32
		if p.As == ABC || p.As == ABCL {
			v = regoff(ctxt, &p.From) & 31
		} else {
			v = 20 /* unconditional */
		}
		r := int(p.Reg)
		if r == 0 {
			r = 0
		}
		switch oclass(&p.To) {
		case C_CTR:
			o1 = OPVCC(19, 528, 0, 0)

		case C_LR:
			o1 = OPVCC(19, 16, 0, 0)

		default:
			ctxt.Diag("bad optab entry (18): %d\n%v", p.To.Class, p)
			v = 0
		}

		if p.As == ABL || p.As == ABCL {
			o1 |= 1
		}
		o1 = OP_BCR(o1, uint32(v), uint32(r))

	case 19: /* mov $lcon,r ==> cau+or */
		d := vregoff(ctxt, &p.From)

		if p.From.Sym == nil {
			o1 = loadu32(int(p.To.Reg), d)
			o2 = LOP_IRR(OP_ORI, uint32(p.To.Reg), uint32(p.To.Reg), uint32(int32(d)))
		} else {
			o1, o2 = symbolAccess(ctxt, p.From.Sym, d, p.To.Reg, OP_ADDI)
		}

	//if(dlm) reloc(&p->from, p->pc, 0);

	case 20: /* add $ucon,,r */
		v := regoff(ctxt, &p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		if p.As == AADD && (r0iszero == 0 /*TypeKind(100016)*/ && p.Reg == 0 || r0iszero != 0 /*TypeKind(100016)*/ && p.To.Reg == 0) {
			ctxt.Diag("literal operation on R0\n%v", p)
		}
		o1 = AOP_IRR(opirr(ctxt, -p.As), uint32(p.To.Reg), uint32(r), uint32(v)>>16)

	case 22: /* add $lcon,r1,r2 ==> cau+or+add */ /* could do add/sub more efficiently */
		if p.To.Reg == REGTMP || p.Reg == REGTMP {
			ctxt.Diag("can't synthesize large constant\n%v", p)
		}
		d := vregoff(ctxt, &p.From)
		o1 = loadu32(REGTMP, d)
		o2 = LOP_IRR(OP_ORI, REGTMP, REGTMP, uint32(int32(d)))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o3 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), REGTMP, uint32(r))
		if p.From.Sym != nil {
			ctxt.Diag("%v is not supported", p)
		}

	//if(dlm) reloc(&p->from, p->pc, 0);

	case 23: /* and $lcon,r1,r2 ==> cau+or+and */ /* masks could be done using rlnm etc. */
		if p.To.Reg == REGTMP || p.Reg == REGTMP {
			ctxt.Diag("can't synthesize large constant\n%v", p)
		}
		d := vregoff(ctxt, &p.From)
		o1 = loadu32(REGTMP, d)
		o2 = LOP_IRR(OP_ORI, REGTMP, REGTMP, uint32(int32(d)))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o3 = LOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), REGTMP, uint32(r))
		if p.From.Sym != nil {
			ctxt.Diag("%v is not supported", p)
		}

		//if(dlm) reloc(&p->from, p->pc, 0);

		/*24*/
	case 25:
		/* sld[.] $sh,rS,rA -> rldicr[.] $sh,rS,mask(0,63-sh),rA; srd[.] -> rldicl */
		v := regoff(ctxt, &p.From)

		if v < 0 {
			v = 0
		} else if v > 63 {
			v = 63
		}
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		var a int
		switch p.As {
		case ASLD, ASLDCC:
			a = int(63 - v)
			o1 = OP_RLDICR

		case ASRD, ASRDCC:
			a = int(v)
			v = 64 - v
			o1 = OP_RLDICL

		default:
			ctxt.Diag("unexpected op in sldi case\n%v", p)
			a = 0
			o1 = 0
		}

		o1 = AOP_RRR(o1, uint32(r), uint32(p.To.Reg), (uint32(v) & 0x1F))
		o1 |= (uint32(a) & 31) << 6
		if v&0x20 != 0 {
			o1 |= 1 << 1
		}
		if a&0x20 != 0 {
			o1 |= 1 << 5 /* mb[5] is top bit */
		}
		if p.As == ASLDCC || p.As == ASRDCC {
			o1 |= 1 /* Rc */
		}

	case 26: /* mov $lsext/auto/oreg,,r2 ==> addis+addi */
		if p.To.Reg == REGTMP {
			ctxt.Diag("can't synthesize large constant\n%v", p)
		}
		v := regoff(ctxt, &p.From)
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = AOP_IRR(OP_ADDIS, REGTMP, uint32(r), uint32(high16adjusted(v)))
		o2 = AOP_IRR(OP_ADDI, uint32(p.To.Reg), REGTMP, uint32(v))

	case 27: /* subc ra,$simm,rd => subfic rd,ra,$simm */
		v := regoff(ctxt, p.From3)

		r := int(p.From.Reg)
		o1 = AOP_IRR(opirr(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(v))

	case 28: /* subc r1,$lcon,r2 ==> cau+or+subfc */
		if p.To.Reg == REGTMP || p.From.Reg == REGTMP {
			ctxt.Diag("can't synthesize large constant\n%v", p)
		}
		v := regoff(ctxt, p.From3)
		o1 = AOP_IRR(OP_ADDIS, REGTMP, REGZERO, uint32(v)>>16)
		o2 = LOP_IRR(OP_ORI, REGTMP, REGTMP, uint32(v))
		o3 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), REGTMP)
		if p.From.Sym != nil {
			ctxt.Diag("%v is not supported", p)
		}

	//if(dlm) reloc(&p->from3, p->pc, 0);

	case 29: /* rldic[lr]? $sh,s,$mask,a -- left, right, plain give different masks */
		v := regoff(ctxt, &p.From)

		d := vregoff(ctxt, p.From3)
		var mask [2]uint8
		maskgen64(ctxt, p, mask[:], uint64(d))
		var a int
		switch p.As {
		case ARLDC, ARLDCCC:
			a = int(mask[0]) /* MB */
			if int32(mask[1]) != (63 - v) {
				ctxt.Diag("invalid mask for shift: %x (shift %d)\n%v", uint64(d), v, p)
			}

		case ARLDCL, ARLDCLCC:
			a = int(mask[0]) /* MB */
			if mask[1] != 63 {
				ctxt.Diag("invalid mask for shift: %x (shift %d)\n%v", uint64(d), v, p)
			}

		case ARLDCR, ARLDCRCC:
			a = int(mask[1]) /* ME */
			if mask[0] != 0 {
				ctxt.Diag("invalid mask for shift: %x (shift %d)\n%v", uint64(d), v, p)
			}

		default:
			ctxt.Diag("unexpected op in rldic case\n%v", p)
			a = 0
		}

		o1 = AOP_RRR(opirr(ctxt, p.As), uint32(p.Reg), uint32(p.To.Reg), (uint32(v) & 0x1F))
		o1 |= (uint32(a) & 31) << 6
		if v&0x20 != 0 {
			o1 |= 1 << 1
		}
		if a&0x20 != 0 {
			o1 |= 1 << 5 /* mb[5] is top bit */
		}

	case 30: /* rldimi $sh,s,$mask,a */
		v := regoff(ctxt, &p.From)

		d := vregoff(ctxt, p.From3)

		// Original opcodes had mask operands which had to be converted to a shift count as expected by
		// the ppc64 asm.
		switch p.As {
		case ARLDMI, ARLDMICC:
			var mask [2]uint8
			maskgen64(ctxt, p, mask[:], uint64(d))
			if int32(mask[1]) != (63 - v) {
				ctxt.Diag("invalid mask for shift: %x (shift %d)\n%v", uint64(d), v, p)
			}
			o1 = AOP_RRR(opirr(ctxt, p.As), uint32(p.Reg), uint32(p.To.Reg), (uint32(v) & 0x1F))
			o1 |= (uint32(mask[0]) & 31) << 6
			if v&0x20 != 0 {
				o1 |= 1 << 1
			}
			if mask[0]&0x20 != 0 {
				o1 |= 1 << 5 /* mb[5] is top bit */
			}

		// Opcodes with shift count operands.
		case ARLDIMI, ARLDIMICC:
			o1 = AOP_RRR(opirr(ctxt, p.As), uint32(p.Reg), uint32(p.To.Reg), (uint32(v) & 0x1F))
			o1 |= (uint32(d) & 31) << 6
			if v&0x20 != 0 {
				o1 |= 1 << 1
			}
		}

	case 31: /* dword */
		d := vregoff(ctxt, &p.From)

		if ctxt.Arch.ByteOrder == binary.BigEndian {
			o1 = uint32(d >> 32)
			o2 = uint32(d)
		} else {
			o1 = uint32(d)
			o2 = uint32(d >> 32)
		}

		if p.From.Sym != nil {
			rel := obj.Addrel(ctxt.Cursym)
			rel.Off = int32(ctxt.Pc)
			rel.Siz = 8
			rel.Sym = p.From.Sym
			rel.Add = p.From.Offset
			rel.Type = obj.R_ADDR
			o2 = 0
			o1 = o2
		}

	case 32: /* fmul frc,fra,frd */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(r), 0) | (uint32(p.From.Reg)&31)<<6

	case 33: /* fabs [frb,]frd; fmr. frb,frd */
		r := int(p.From.Reg)

		if oclass(&p.From) == C_NONE {
			r = int(p.To.Reg)
		}
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), 0, uint32(r))

	case 34: /* FMADDx fra,frb,frc,frd (d=a*b+c); FSELx a<0? (d=b): (d=c) */
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg)) | (uint32(p.From3.Reg)&31)<<6

	case 35: /* mov r,lext/lauto/loreg ==> cau $(v>>16),sb,r'; store o(r') */
		v := regoff(ctxt, &p.To)

		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = AOP_IRR(OP_ADDIS, REGTMP, uint32(r), uint32(high16adjusted(v)))
		o2 = AOP_IRR(opstore(ctxt, p.As), uint32(p.From.Reg), REGTMP, uint32(v))

	case 36: /* mov bz/h/hz lext/lauto/lreg,r ==> lbz/lha/lhz etc */
		v := regoff(ctxt, &p.From)

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = AOP_IRR(OP_ADDIS, REGTMP, uint32(r), uint32(high16adjusted(v)))
		o2 = AOP_IRR(opload(ctxt, p.As), uint32(p.To.Reg), REGTMP, uint32(v))

	case 37: /* movb lext/lauto/lreg,r ==> lbz o(reg),r; extsb r */
		v := regoff(ctxt, &p.From)

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = AOP_IRR(OP_ADDIS, REGTMP, uint32(r), uint32(high16adjusted(v)))
		o2 = AOP_IRR(opload(ctxt, p.As), uint32(p.To.Reg), REGTMP, uint32(v))
		o3 = LOP_RRR(OP_EXTSB, uint32(p.To.Reg), uint32(p.To.Reg), 0)

	case 40: /* word */
		o1 = uint32(regoff(ctxt, &p.From))

	case 41: /* stswi */
		o1 = AOP_RRR(opirr(ctxt, p.As), uint32(p.From.Reg), uint32(p.To.Reg), 0) | (uint32(regoff(ctxt, p.From3))&0x7F)<<11

	case 42: /* lswi */
		o1 = AOP_RRR(opirr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), 0) | (uint32(regoff(ctxt, p.From3))&0x7F)<<11

	case 43: /* unary indexed source: dcbf (b); dcbf (a+b) */
		o1 = AOP_RRR(oprrr(ctxt, p.As), 0, uint32(p.From.Index), uint32(p.From.Reg))

	case 44: /* indexed store */
		o1 = AOP_RRR(opstorex(ctxt, p.As), uint32(p.From.Reg), uint32(p.To.Index), uint32(p.To.Reg))

	case 45: /* indexed load */
		o1 = AOP_RRR(oploadx(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Index), uint32(p.From.Reg))

	case 46: /* plain op */
		o1 = oprrr(ctxt, p.As)

	case 47: /* op Ra, Rd; also op [Ra,] Rd */
		r := int(p.From.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(r), 0)

	case 48: /* op Rs, Ra */
		r := int(p.From.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = LOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(r), 0)

	case 49: /* op Rb; op $n, Rb */
		if p.From.Type != obj.TYPE_REG { /* tlbie $L, rB */
			v := regoff(ctxt, &p.From) & 1
			o1 = AOP_RRR(oprrr(ctxt, p.As), 0, 0, uint32(p.To.Reg)) | uint32(v)<<21
		} else {
			o1 = AOP_RRR(oprrr(ctxt, p.As), 0, 0, uint32(p.From.Reg))
		}

	case 50: /* rem[u] r1[,r2],r3 */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		v := oprrr(ctxt, p.As)
		t := v & (1<<10 | 1) /* OE|Rc */
		o1 = AOP_RRR(v&^t, REGTMP, uint32(r), uint32(p.From.Reg))
		o2 = AOP_RRR(OP_MULLW, REGTMP, REGTMP, uint32(p.From.Reg))
		o3 = AOP_RRR(OP_SUBF|t, uint32(p.To.Reg), REGTMP, uint32(r))
		if p.As == AREMU {
			o4 = o3

			/* Clear top 32 bits */
			o3 = OP_RLW(OP_RLDIC, REGTMP, REGTMP, 0, 0, 0) | 1<<5
		}

	case 51: /* remd[u] r1[,r2],r3 */
		r := int(p.Reg)

		if r == 0 {
			r = int(p.To.Reg)
		}
		v := oprrr(ctxt, p.As)
		t := v & (1<<10 | 1) /* OE|Rc */
		o1 = AOP_RRR(v&^t, REGTMP, uint32(r), uint32(p.From.Reg))
		o2 = AOP_RRR(OP_MULLD, REGTMP, REGTMP, uint32(p.From.Reg))
		o3 = AOP_RRR(OP_SUBF|t, uint32(p.To.Reg), REGTMP, uint32(r))

	case 52: /* mtfsbNx cr(n) */
		v := regoff(ctxt, &p.From) & 31

		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(v), 0, 0)

	case 53: /* mffsX ,fr1 */
		o1 = AOP_RRR(OP_MFFS, uint32(p.To.Reg), 0, 0)

	case 54: /* mov msr,r1; mov r1, msr*/
		if oclass(&p.From) == C_REG {
			if p.As == AMOVD {
				o1 = AOP_RRR(OP_MTMSRD, uint32(p.From.Reg), 0, 0)
			} else {
				o1 = AOP_RRR(OP_MTMSR, uint32(p.From.Reg), 0, 0)
			}
		} else {
			o1 = AOP_RRR(OP_MFMSR, uint32(p.To.Reg), 0, 0)
		}

	case 55: /* op Rb, Rd */
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), 0, uint32(p.From.Reg))

	case 56: /* sra $sh,[s,]a; srd $sh,[s,]a */
		v := regoff(ctxt, &p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = AOP_RRR(opirr(ctxt, p.As), uint32(r), uint32(p.To.Reg), uint32(v)&31)
		if (p.As == ASRAD || p.As == ASRADCC) && (v&0x20 != 0) {
			o1 |= 1 << 1 /* mb[5] */
		}

	case 57: /* slw $sh,[s,]a -> rlwinm ... */
		v := regoff(ctxt, &p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}

		/*
			 * Let user (gs) shoot himself in the foot.
			 * qc has already complained.
			 *
			if(v < 0 || v > 31)
				ctxt->diag("illegal shift %ld\n%v", v, p);
		*/
		if v < 0 {
			v = 0
		} else if v > 32 {
			v = 32
		}
		var mask [2]uint8
		if p.As == ASRW || p.As == ASRWCC { /* shift right */
			mask[0] = uint8(v)
			mask[1] = 31
			v = 32 - v
		} else {
			mask[0] = 0
			mask[1] = uint8(31 - v)
		}

		o1 = OP_RLW(OP_RLWINM, uint32(p.To.Reg), uint32(r), uint32(v), uint32(mask[0]), uint32(mask[1]))
		if p.As == ASLWCC || p.As == ASRWCC {
			o1 |= 1 /* Rc */
		}

	case 58: /* logical $andcon,[s],a */
		v := regoff(ctxt, &p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = LOP_IRR(opirr(ctxt, p.As), uint32(p.To.Reg), uint32(r), uint32(v))

	case 59: /* or/and $ucon,,r */
		v := regoff(ctxt, &p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = LOP_IRR(opirr(ctxt, -p.As), uint32(p.To.Reg), uint32(r), uint32(v)>>16) /* oris, xoris, andis */

	case 60: /* tw to,a,b */
		r := int(regoff(ctxt, &p.From) & 31)

		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(r), uint32(p.Reg), uint32(p.To.Reg))

	case 61: /* tw to,a,$simm */
		r := int(regoff(ctxt, &p.From) & 31)

		v := regoff(ctxt, &p.To)
		o1 = AOP_IRR(opirr(ctxt, p.As), uint32(r), uint32(p.Reg), uint32(v))

	case 62: /* rlwmi $sh,s,$mask,a */
		v := regoff(ctxt, &p.From)

		var mask [2]uint8
		maskgen(ctxt, p, mask[:], uint32(regoff(ctxt, p.From3)))
		o1 = AOP_RRR(opirr(ctxt, p.As), uint32(p.Reg), uint32(p.To.Reg), uint32(v))
		o1 |= (uint32(mask[0])&31)<<6 | (uint32(mask[1])&31)<<1

	case 63: /* rlwmi b,s,$mask,a */
		var mask [2]uint8
		maskgen(ctxt, p, mask[:], uint32(regoff(ctxt, p.From3)))

		o1 = AOP_RRR(opirr(ctxt, p.As), uint32(p.Reg), uint32(p.To.Reg), uint32(p.From.Reg))
		o1 |= (uint32(mask[0])&31)<<6 | (uint32(mask[1])&31)<<1

	case 64: /* mtfsf fr[, $m] {,fpcsr} */
		var v int32
		if p.From3Type() != obj.TYPE_NONE {
			v = regoff(ctxt, p.From3) & 255
		} else {
			v = 255
		}
		o1 = OP_MTFSF | uint32(v)<<17 | uint32(p.From.Reg)<<11

	case 65: /* MOVFL $imm,FPSCR(n) => mtfsfi crfd,imm */
		if p.To.Reg == 0 {
			ctxt.Diag("must specify FPSCR(n)\n%v", p)
		}
		o1 = OP_MTFSFI | (uint32(p.To.Reg)&15)<<23 | (uint32(regoff(ctxt, &p.From))&31)<<12

	case 66: /* mov spr,r1; mov r1,spr, also dcr */
		var r int
		var v int32
		if REG_R0 <= p.From.Reg && p.From.Reg <= REG_R31 {
			r = int(p.From.Reg)
			v = int32(p.To.Reg)
			if REG_DCR0 <= v && v <= REG_DCR0+1023 {
				o1 = OPVCC(31, 451, 0, 0) /* mtdcr */
			} else {
				o1 = OPVCC(31, 467, 0, 0) /* mtspr */
			}
		} else {
			r = int(p.To.Reg)
			v = int32(p.From.Reg)
			if REG_DCR0 <= v && v <= REG_DCR0+1023 {
				o1 = OPVCC(31, 323, 0, 0) /* mfdcr */
			} else {
				o1 = OPVCC(31, 339, 0, 0) /* mfspr */
			}
		}

		o1 = AOP_RRR(o1, uint32(r), 0, 0) | (uint32(v)&0x1f)<<16 | ((uint32(v)>>5)&0x1f)<<11

	case 67: /* mcrf crfD,crfS */
		if p.From.Type != obj.TYPE_REG || p.From.Reg < REG_CR0 || REG_CR7 < p.From.Reg || p.To.Type != obj.TYPE_REG || p.To.Reg < REG_CR0 || REG_CR7 < p.To.Reg {
			ctxt.Diag("illegal CR field number\n%v", p)
		}
		o1 = AOP_RRR(OP_MCRF, ((uint32(p.To.Reg) & 7) << 2), ((uint32(p.From.Reg) & 7) << 2), 0)

	case 68: /* mfcr rD; mfocrf CRM,rD */
		if p.From.Type == obj.TYPE_REG && REG_CR0 <= p.From.Reg && p.From.Reg <= REG_CR7 {
			v := int32(1 << uint(7-(p.To.Reg&7)))                                 /* CR(n) */
			o1 = AOP_RRR(OP_MFCR, uint32(p.To.Reg), 0, 0) | 1<<20 | uint32(v)<<12 /* new form, mfocrf */
		} else {
			o1 = AOP_RRR(OP_MFCR, uint32(p.To.Reg), 0, 0) /* old form, whole register */
		}

	case 69: /* mtcrf CRM,rS */
		var v int32
		if p.From3Type() != obj.TYPE_NONE {
			if p.To.Reg != 0 {
				ctxt.Diag("can't use both mask and CR(n)\n%v", p)
			}
			v = regoff(ctxt, p.From3) & 0xff
		} else {
			if p.To.Reg == 0 {
				v = 0xff /* CR */
			} else {
				v = 1 << uint(7-(p.To.Reg&7)) /* CR(n) */
			}
		}

		o1 = AOP_RRR(OP_MTCRF, uint32(p.From.Reg), 0, 0) | uint32(v)<<12

	case 70: /* [f]cmp r,r,cr*/
		var r int
		if p.Reg == 0 {
			r = 0
		} else {
			r = (int(p.Reg) & 7) << 2
		}
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(r), uint32(p.From.Reg), uint32(p.To.Reg))

	case 71: /* cmp[l] r,i,cr*/
		var r int
		if p.Reg == 0 {
			r = 0
		} else {
			r = (int(p.Reg) & 7) << 2
		}
		o1 = AOP_RRR(opirr(ctxt, p.As), uint32(r), uint32(p.From.Reg), 0) | uint32(regoff(ctxt, &p.To))&0xffff

	case 72: /* slbmte (Rb+Rs -> slb[Rb]) -> Rs, Rb */
		o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.From.Reg), 0, uint32(p.To.Reg))

	case 73: /* mcrfs crfD,crfS */
		if p.From.Type != obj.TYPE_REG || p.From.Reg != REG_FPSCR || p.To.Type != obj.TYPE_REG || p.To.Reg < REG_CR0 || REG_CR7 < p.To.Reg {
			ctxt.Diag("illegal FPSCR/CR field number\n%v", p)
		}
		o1 = AOP_RRR(OP_MCRFS, ((uint32(p.To.Reg) & 7) << 2), ((0 & 7) << 2), 0)

	case 77: /* syscall $scon, syscall Rx */
		if p.From.Type == obj.TYPE_CONST {
			if p.From.Offset > BIG || p.From.Offset < -BIG {
				ctxt.Diag("illegal syscall, sysnum too large: %v", p)
			}
			o1 = AOP_IRR(OP_ADDI, REGZERO, REGZERO, uint32(p.From.Offset))
		} else if p.From.Type == obj.TYPE_REG {
			o1 = LOP_RRR(OP_OR, REGZERO, uint32(p.From.Reg), uint32(p.From.Reg))
		} else {
			ctxt.Diag("illegal syscall: %v", p)
			o1 = 0x7fe00008 // trap always
		}

		o2 = oprrr(ctxt, p.As)
		o3 = AOP_RRR(oprrr(ctxt, AXOR), REGZERO, REGZERO, REGZERO) // XOR R0, R0

	case 78: /* undef */
		o1 = 0 /* "An instruction consisting entirely of binary 0s is guaranteed
		   always to be an illegal instruction."  */

	/* relocation operations */
	case 74:
		v := vregoff(ctxt, &p.To)
		o1, o2 = symbolAccess(ctxt, p.To.Sym, v, p.From.Reg, opstore(ctxt, p.As))

	//if(dlm) reloc(&p->to, p->pc, 1);

	case 75:
		v := vregoff(ctxt, &p.From)
		o1, o2 = symbolAccess(ctxt, p.From.Sym, v, p.To.Reg, opload(ctxt, p.As))

	//if(dlm) reloc(&p->from, p->pc, 1);

	case 76:
		v := vregoff(ctxt, &p.From)
		o1, o2 = symbolAccess(ctxt, p.From.Sym, v, p.To.Reg, opload(ctxt, p.As))
		o3 = LOP_RRR(OP_EXTSB, uint32(p.To.Reg), uint32(p.To.Reg), 0)

		//if(dlm) reloc(&p->from, p->pc, 1);

	case 79:
		if p.From.Offset != 0 {
			ctxt.Diag("invalid offset against tls var %v", p)
		}
		o1 = AOP_IRR(OP_ADDI, uint32(p.To.Reg), REGZERO, 0)
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 4
		rel.Sym = p.From.Sym
		rel.Type = obj.R_POWER_TLS_LE

	case 80:
		if p.From.Offset != 0 {
			ctxt.Diag("invalid offset against tls var %v", p)
		}
		o1 = AOP_IRR(OP_ADDIS, uint32(p.To.Reg), REG_R2, 0)
		o2 = AOP_IRR(opload(ctxt, AMOVD), uint32(p.To.Reg), uint32(p.To.Reg), 0)
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 8
		rel.Sym = p.From.Sym
		rel.Type = obj.R_POWER_TLS_IE

	case 81:
		v := vregoff(ctxt, &p.To)
		if v != 0 {
			ctxt.Diag("invalid offset against GOT slot %v", p)
		}

		o1 = AOP_IRR(OP_ADDIS, uint32(p.To.Reg), REG_R2, 0)
		o2 = AOP_IRR(opload(ctxt, AMOVD), uint32(p.To.Reg), uint32(p.To.Reg), 0)
		rel := obj.Addrel(ctxt.Cursym)
		rel.Off = int32(ctxt.Pc)
		rel.Siz = 8
		rel.Sym = p.From.Sym
		rel.Type = obj.R_ADDRPOWER_GOT
	case 82: /* vector instructions, VX-form and VC-form */
		if p.From.Type == obj.TYPE_REG {
			/* reg reg none OR reg reg reg */
			/* 3-register operand order: VRA, VRB, VRT */
			/* 2-register operand order: VRA, VRT */
			o1 = AOP_RRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg))
		} else if p.From3Type() == obj.TYPE_CONST {
			/* imm imm reg reg */
			/* operand order: SIX, VRA, ST, VRT */
			six := int(regoff(ctxt, &p.From))
			st := int(regoff(ctxt, p.From3))
			o1 = AOP_IIRR(opiirr(ctxt, p.As), uint32(p.To.Reg), uint32(p.Reg), uint32(st), uint32(six))
		} else if p.From3Type() == obj.TYPE_NONE && p.Reg != 0 {
			/* imm reg reg */
			/* operand order: UIM, VRB, VRT */
			uim := int(regoff(ctxt, &p.From))
			o1 = AOP_VIRR(opirr(ctxt, p.As), uint32(p.To.Reg), uint32(p.Reg), uint32(uim))
		} else {
			/* imm reg */
			/* operand order: SIM, VRT */
			sim := int(regoff(ctxt, &p.From))
			o1 = AOP_IR(opirr(ctxt, p.As), uint32(p.To.Reg), uint32(sim))
		}

	case 83: /* vector instructions, VA-form */
		if p.From.Type == obj.TYPE_REG {
			/* reg reg reg reg */
			/* 4-register operand order: VRA, VRB, VRC, VRT */
			o1 = AOP_RRRR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg), uint32(p.From3.Reg))
		} else if p.From.Type == obj.TYPE_CONST {
			/* imm reg reg reg */
			/* operand order: SHB, VRA, VRB, VRT */
			shb := int(regoff(ctxt, &p.From))
			o1 = AOP_IRRR(opirrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.Reg), uint32(p.From3.Reg), uint32(shb))
		}

	case 84: // ISEL BC,RA,RB,RT -> isel rt,ra,rb,bc
		bc := vregoff(ctxt, &p.From)

		// rt = To.Reg, ra = p.Reg, rb = p.From3.Reg
		o1 = AOP_ISEL(OP_ISEL, uint32(p.To.Reg), uint32(p.Reg), uint32(p.From3.Reg), uint32(bc))

	case 85: /* vector instructions, VX-form */
		/* reg none reg */
		/* 2-register operand order: VRB, VRT */
		o1 = AOP_RR(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg))

	case 86: /* VSX indexed store, XX1-form */
		/* reg reg reg */
		/* 3-register operand order: XT, (RB)(RA*1) */
		o1 = AOP_XX1(opstorex(ctxt, p.As), uint32(p.From.Reg), uint32(p.To.Index), uint32(p.To.Reg))

	case 87: /* VSX indexed load, XX1-form */
		/* reg reg reg */
		/* 3-register operand order: (RB)(RA*1), XT */
		o1 = AOP_XX1(oploadx(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Index), uint32(p.From.Reg))

	case 88: /* VSX instructions, XX1-form */
		/* reg reg none OR reg reg reg */
		/* 3-register operand order: RA, RB, XT */
		/* 2-register operand order: XS, RA or RA, XT */
		xt := int32(p.To.Reg)
		xs := int32(p.From.Reg)
		if REG_VS0 <= xt && xt <= REG_VS63 {
			o1 = AOP_XX1(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg))
		} else if REG_VS0 <= xs && xs <= REG_VS63 {
			o1 = AOP_XX1(oprrr(ctxt, p.As), uint32(p.From.Reg), uint32(p.To.Reg), uint32(p.Reg))
		}

	case 89: /* VSX instructions, XX2-form */
		/* reg none reg OR reg imm reg */
		/* 2-register operand order: XB, XT or XB, UIM, XT*/
		uim := int(regoff(ctxt, p.From3))
		o1 = AOP_XX2(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(uim), uint32(p.From.Reg))

	case 90: /* VSX instructions, XX3-form */
		if p.From3Type() == obj.TYPE_NONE {
			/* reg reg reg */
			/* 3-register operand order: XA, XB, XT */
			o1 = AOP_XX3(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg))
		} else if p.From3Type() == obj.TYPE_CONST {
			/* reg reg reg imm */
			/* operand order: XA, XB, DM, XT */
			dm := int(regoff(ctxt, p.From3))
			o1 = AOP_XX3I(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg), uint32(dm))
		}

	case 91: /* VSX instructions, XX4-form */
		/* reg reg reg reg */
		/* 3-register operand order: XA, XB, XC, XT */
		o1 = AOP_XX4(oprrr(ctxt, p.As), uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg), uint32(p.From3.Reg))

	}

	out[0] = o1
	out[1] = o2
	out[2] = o3
	out[3] = o4
	out[4] = o5
	return
}

func vregoff(ctxt *obj.Link, a *obj.Addr) int64 {
	ctxt.Instoffset = 0
	if a != nil {
		aclass(ctxt, a)
	}
	return ctxt.Instoffset
}

func regoff(ctxt *obj.Link, a *obj.Addr) int32 {
	return int32(vregoff(ctxt, a))
}

func oprrr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AADD:
		return OPVCC(31, 266, 0, 0)
	case AADDCC:
		return OPVCC(31, 266, 0, 1)
	case AADDV:
		return OPVCC(31, 266, 1, 0)
	case AADDVCC:
		return OPVCC(31, 266, 1, 1)
	case AADDC:
		return OPVCC(31, 10, 0, 0)
	case AADDCCC:
		return OPVCC(31, 10, 0, 1)
	case AADDCV:
		return OPVCC(31, 10, 1, 0)
	case AADDCVCC:
		return OPVCC(31, 10, 1, 1)
	case AADDE:
		return OPVCC(31, 138, 0, 0)
	case AADDECC:
		return OPVCC(31, 138, 0, 1)
	case AADDEV:
		return OPVCC(31, 138, 1, 0)
	case AADDEVCC:
		return OPVCC(31, 138, 1, 1)
	case AADDME:
		return OPVCC(31, 234, 0, 0)
	case AADDMECC:
		return OPVCC(31, 234, 0, 1)
	case AADDMEV:
		return OPVCC(31, 234, 1, 0)
	case AADDMEVCC:
		return OPVCC(31, 234, 1, 1)
	case AADDZE:
		return OPVCC(31, 202, 0, 0)
	case AADDZECC:
		return OPVCC(31, 202, 0, 1)
	case AADDZEV:
		return OPVCC(31, 202, 1, 0)
	case AADDZEVCC:
		return OPVCC(31, 202, 1, 1)

	case AAND:
		return OPVCC(31, 28, 0, 0)
	case AANDCC:
		return OPVCC(31, 28, 0, 1)
	case AANDN:
		return OPVCC(31, 60, 0, 0)
	case AANDNCC:
		return OPVCC(31, 60, 0, 1)

	case ACMP:
		return OPVCC(31, 0, 0, 0) | 1<<21 /* L=1 */
	case ACMPU:
		return OPVCC(31, 32, 0, 0) | 1<<21
	case ACMPW:
		return OPVCC(31, 0, 0, 0) /* L=0 */
	case ACMPWU:
		return OPVCC(31, 32, 0, 0)

	case ACNTLZW:
		return OPVCC(31, 26, 0, 0)
	case ACNTLZWCC:
		return OPVCC(31, 26, 0, 1)
	case ACNTLZD:
		return OPVCC(31, 58, 0, 0)
	case ACNTLZDCC:
		return OPVCC(31, 58, 0, 1)

	case ACRAND:
		return OPVCC(19, 257, 0, 0)
	case ACRANDN:
		return OPVCC(19, 129, 0, 0)
	case ACREQV:
		return OPVCC(19, 289, 0, 0)
	case ACRNAND:
		return OPVCC(19, 225, 0, 0)
	case ACRNOR:
		return OPVCC(19, 33, 0, 0)
	case ACROR:
		return OPVCC(19, 449, 0, 0)
	case ACRORN:
		return OPVCC(19, 417, 0, 0)
	case ACRXOR:
		return OPVCC(19, 193, 0, 0)

	case ADCBF:
		return OPVCC(31, 86, 0, 0)
	case ADCBI:
		return OPVCC(31, 470, 0, 0)
	case ADCBST:
		return OPVCC(31, 54, 0, 0)
	case ADCBT:
		return OPVCC(31, 278, 0, 0)
	case ADCBTST:
		return OPVCC(31, 246, 0, 0)
	case ADCBZ:
		return OPVCC(31, 1014, 0, 0)

	case AREM, ADIVW:
		return OPVCC(31, 491, 0, 0)

	case AREMCC, ADIVWCC:
		return OPVCC(31, 491, 0, 1)

	case AREMV, ADIVWV:
		return OPVCC(31, 491, 1, 0)

	case AREMVCC, ADIVWVCC:
		return OPVCC(31, 491, 1, 1)

	case AREMU, ADIVWU:
		return OPVCC(31, 459, 0, 0)

	case AREMUCC, ADIVWUCC:
		return OPVCC(31, 459, 0, 1)

	case AREMUV, ADIVWUV:
		return OPVCC(31, 459, 1, 0)

	case AREMUVCC, ADIVWUVCC:
		return OPVCC(31, 459, 1, 1)

	case AREMD, ADIVD:
		return OPVCC(31, 489, 0, 0)

	case AREMDCC, ADIVDCC:
		return OPVCC(31, 489, 0, 1)

	case ADIVDE:
		return OPVCC(31, 425, 0, 0)

	case ADIVDECC:
		return OPVCC(31, 425, 0, 1)

	case ADIVDEU:
		return OPVCC(31, 393, 0, 0)

	case ADIVDEUCC:
		return OPVCC(31, 393, 0, 1)

	case AREMDV, ADIVDV:
		return OPVCC(31, 489, 1, 0)

	case AREMDVCC, ADIVDVCC:
		return OPVCC(31, 489, 1, 1)

	case AREMDU, ADIVDU:
		return OPVCC(31, 457, 0, 0)

	case AREMDUCC, ADIVDUCC:
		return OPVCC(31, 457, 0, 1)

	case AREMDUV, ADIVDUV:
		return OPVCC(31, 457, 1, 0)

	case AREMDUVCC, ADIVDUVCC:
		return OPVCC(31, 457, 1, 1)

	case AEIEIO:
		return OPVCC(31, 854, 0, 0)

	case AEQV:
		return OPVCC(31, 284, 0, 0)
	case AEQVCC:
		return OPVCC(31, 284, 0, 1)

	case AEXTSB:
		return OPVCC(31, 954, 0, 0)
	case AEXTSBCC:
		return OPVCC(31, 954, 0, 1)
	case AEXTSH:
		return OPVCC(31, 922, 0, 0)
	case AEXTSHCC:
		return OPVCC(31, 922, 0, 1)
	case AEXTSW:
		return OPVCC(31, 986, 0, 0)
	case AEXTSWCC:
		return OPVCC(31, 986, 0, 1)

	case AFABS:
		return OPVCC(63, 264, 0, 0)
	case AFABSCC:
		return OPVCC(63, 264, 0, 1)
	case AFADD:
		return OPVCC(63, 21, 0, 0)
	case AFADDCC:
		return OPVCC(63, 21, 0, 1)
	case AFADDS:
		return OPVCC(59, 21, 0, 0)
	case AFADDSCC:
		return OPVCC(59, 21, 0, 1)
	case AFCMPO:
		return OPVCC(63, 32, 0, 0)
	case AFCMPU:
		return OPVCC(63, 0, 0, 0)
	case AFCFID:
		return OPVCC(63, 846, 0, 0)
	case AFCFIDCC:
		return OPVCC(63, 846, 0, 1)
	case AFCFIDU:
		return OPVCC(63, 974, 0, 0)
	case AFCFIDUCC:
		return OPVCC(63, 974, 0, 1)
	case AFCTIW:
		return OPVCC(63, 14, 0, 0)
	case AFCTIWCC:
		return OPVCC(63, 14, 0, 1)
	case AFCTIWZ:
		return OPVCC(63, 15, 0, 0)
	case AFCTIWZCC:
		return OPVCC(63, 15, 0, 1)
	case AFCTID:
		return OPVCC(63, 814, 0, 0)
	case AFCTIDCC:
		return OPVCC(63, 814, 0, 1)
	case AFCTIDZ:
		return OPVCC(63, 815, 0, 0)
	case AFCTIDZCC:
		return OPVCC(63, 815, 0, 1)
	case AFDIV:
		return OPVCC(63, 18, 0, 0)
	case AFDIVCC:
		return OPVCC(63, 18, 0, 1)
	case AFDIVS:
		return OPVCC(59, 18, 0, 0)
	case AFDIVSCC:
		return OPVCC(59, 18, 0, 1)
	case AFMADD:
		return OPVCC(63, 29, 0, 0)
	case AFMADDCC:
		return OPVCC(63, 29, 0, 1)
	case AFMADDS:
		return OPVCC(59, 29, 0, 0)
	case AFMADDSCC:
		return OPVCC(59, 29, 0, 1)

	case AFMOVS, AFMOVD:
		return OPVCC(63, 72, 0, 0) /* load */
	case AFMOVDCC:
		return OPVCC(63, 72, 0, 1)
	case AFMSUB:
		return OPVCC(63, 28, 0, 0)
	case AFMSUBCC:
		return OPVCC(63, 28, 0, 1)
	case AFMSUBS:
		return OPVCC(59, 28, 0, 0)
	case AFMSUBSCC:
		return OPVCC(59, 28, 0, 1)
	case AFMUL:
		return OPVCC(63, 25, 0, 0)
	case AFMULCC:
		return OPVCC(63, 25, 0, 1)
	case AFMULS:
		return OPVCC(59, 25, 0, 0)
	case AFMULSCC:
		return OPVCC(59, 25, 0, 1)
	case AFNABS:
		return OPVCC(63, 136, 0, 0)
	case AFNABSCC:
		return OPVCC(63, 136, 0, 1)
	case AFNEG:
		return OPVCC(63, 40, 0, 0)
	case AFNEGCC:
		return OPVCC(63, 40, 0, 1)
	case AFNMADD:
		return OPVCC(63, 31, 0, 0)
	case AFNMADDCC:
		return OPVCC(63, 31, 0, 1)
	case AFNMADDS:
		return OPVCC(59, 31, 0, 0)
	case AFNMADDSCC:
		return OPVCC(59, 31, 0, 1)
	case AFNMSUB:
		return OPVCC(63, 30, 0, 0)
	case AFNMSUBCC:
		return OPVCC(63, 30, 0, 1)
	case AFNMSUBS:
		return OPVCC(59, 30, 0, 0)
	case AFNMSUBSCC:
		return OPVCC(59, 30, 0, 1)
	case AFRES:
		return OPVCC(59, 24, 0, 0)
	case AFRESCC:
		return OPVCC(59, 24, 0, 1)
	case AFRIM:
		return OPVCC(63, 488, 0, 0)
	case AFRIMCC:
		return OPVCC(63, 488, 0, 1)
	case AFRIP:
		return OPVCC(63, 456, 0, 0)
	case AFRIPCC:
		return OPVCC(63, 456, 0, 1)
	case AFRIZ:
		return OPVCC(63, 424, 0, 0)
	case AFRIZCC:
		return OPVCC(63, 424, 0, 1)
	case AFRSP:
		return OPVCC(63, 12, 0, 0)
	case AFRSPCC:
		return OPVCC(63, 12, 0, 1)
	case AFRSQRTE:
		return OPVCC(63, 26, 0, 0)
	case AFRSQRTECC:
		return OPVCC(63, 26, 0, 1)
	case AFSEL:
		return OPVCC(63, 23, 0, 0)
	case AFSELCC:
		return OPVCC(63, 23, 0, 1)
	case AFSQRT:
		return OPVCC(63, 22, 0, 0)
	case AFSQRTCC:
		return OPVCC(63, 22, 0, 1)
	case AFSQRTS:
		return OPVCC(59, 22, 0, 0)
	case AFSQRTSCC:
		return OPVCC(59, 22, 0, 1)
	case AFSUB:
		return OPVCC(63, 20, 0, 0)
	case AFSUBCC:
		return OPVCC(63, 20, 0, 1)
	case AFSUBS:
		return OPVCC(59, 20, 0, 0)
	case AFSUBSCC:
		return OPVCC(59, 20, 0, 1)

	case AICBI:
		return OPVCC(31, 982, 0, 0)
	case AISYNC:
		return OPVCC(19, 150, 0, 0)

	case AMTFSB0:
		return OPVCC(63, 70, 0, 0)
	case AMTFSB0CC:
		return OPVCC(63, 70, 0, 1)
	case AMTFSB1:
		return OPVCC(63, 38, 0, 0)
	case AMTFSB1CC:
		return OPVCC(63, 38, 0, 1)

	case AMULHW:
		return OPVCC(31, 75, 0, 0)
	case AMULHWCC:
		return OPVCC(31, 75, 0, 1)
	case AMULHWU:
		return OPVCC(31, 11, 0, 0)
	case AMULHWUCC:
		return OPVCC(31, 11, 0, 1)
	case AMULLW:
		return OPVCC(31, 235, 0, 0)
	case AMULLWCC:
		return OPVCC(31, 235, 0, 1)
	case AMULLWV:
		return OPVCC(31, 235, 1, 0)
	case AMULLWVCC:
		return OPVCC(31, 235, 1, 1)

	case AMULHD:
		return OPVCC(31, 73, 0, 0)
	case AMULHDCC:
		return OPVCC(31, 73, 0, 1)
	case AMULHDU:
		return OPVCC(31, 9, 0, 0)
	case AMULHDUCC:
		return OPVCC(31, 9, 0, 1)
	case AMULLD:
		return OPVCC(31, 233, 0, 0)
	case AMULLDCC:
		return OPVCC(31, 233, 0, 1)
	case AMULLDV:
		return OPVCC(31, 233, 1, 0)
	case AMULLDVCC:
		return OPVCC(31, 233, 1, 1)

	case ANAND:
		return OPVCC(31, 476, 0, 0)
	case ANANDCC:
		return OPVCC(31, 476, 0, 1)
	case ANEG:
		return OPVCC(31, 104, 0, 0)
	case ANEGCC:
		return OPVCC(31, 104, 0, 1)
	case ANEGV:
		return OPVCC(31, 104, 1, 0)
	case ANEGVCC:
		return OPVCC(31, 104, 1, 1)
	case ANOR:
		return OPVCC(31, 124, 0, 0)
	case ANORCC:
		return OPVCC(31, 124, 0, 1)
	case AOR:
		return OPVCC(31, 444, 0, 0)
	case AORCC:
		return OPVCC(31, 444, 0, 1)
	case AORN:
		return OPVCC(31, 412, 0, 0)
	case AORNCC:
		return OPVCC(31, 412, 0, 1)

	case ARFI:
		return OPVCC(19, 50, 0, 0)
	case ARFCI:
		return OPVCC(19, 51, 0, 0)
	case ARFID:
		return OPVCC(19, 18, 0, 0)
	case AHRFID:
		return OPVCC(19, 274, 0, 0)

	case ARLWMI:
		return OPVCC(20, 0, 0, 0)
	case ARLWMICC:
		return OPVCC(20, 0, 0, 1)
	case ARLWNM:
		return OPVCC(23, 0, 0, 0)
	case ARLWNMCC:
		return OPVCC(23, 0, 0, 1)

	case ARLDCL:
		return OPVCC(30, 8, 0, 0)
	case ARLDCR:
		return OPVCC(30, 9, 0, 0)

	case ARLDICL:
		return OPVCC(30, 0, 0, 0)
	case ARLDICLCC:
		return OPVCC(30, 0, 0, 1)
	case ARLDICR:
		return OPVCC(30, 0, 0, 0) | 2<<1 // rldicr
	case ARLDICRCC:
		return OPVCC(30, 0, 0, 1) | 2<<1 // rldicr.

	case ASYSCALL:
		return OPVCC(17, 1, 0, 0)

	case ASLW:
		return OPVCC(31, 24, 0, 0)
	case ASLWCC:
		return OPVCC(31, 24, 0, 1)
	case ASLD:
		return OPVCC(31, 27, 0, 0)
	case ASLDCC:
		return OPVCC(31, 27, 0, 1)

	case ASRAW:
		return OPVCC(31, 792, 0, 0)
	case ASRAWCC:
		return OPVCC(31, 792, 0, 1)
	case ASRAD:
		return OPVCC(31, 794, 0, 0)
	case ASRADCC:
		return OPVCC(31, 794, 0, 1)

	case ASRW:
		return OPVCC(31, 536, 0, 0)
	case ASRWCC:
		return OPVCC(31, 536, 0, 1)
	case ASRD:
		return OPVCC(31, 539, 0, 0)
	case ASRDCC:
		return OPVCC(31, 539, 0, 1)

	case ASUB:
		return OPVCC(31, 40, 0, 0)
	case ASUBCC:
		return OPVCC(31, 40, 0, 1)
	case ASUBV:
		return OPVCC(31, 40, 1, 0)
	case ASUBVCC:
		return OPVCC(31, 40, 1, 1)
	case ASUBC:
		return OPVCC(31, 8, 0, 0)
	case ASUBCCC:
		return OPVCC(31, 8, 0, 1)
	case ASUBCV:
		return OPVCC(31, 8, 1, 0)
	case ASUBCVCC:
		return OPVCC(31, 8, 1, 1)
	case ASUBE:
		return OPVCC(31, 136, 0, 0)
	case ASUBECC:
		return OPVCC(31, 136, 0, 1)
	case ASUBEV:
		return OPVCC(31, 136, 1, 0)
	case ASUBEVCC:
		return OPVCC(31, 136, 1, 1)
	case ASUBME:
		return OPVCC(31, 232, 0, 0)
	case ASUBMECC:
		return OPVCC(31, 232, 0, 1)
	case ASUBMEV:
		return OPVCC(31, 232, 1, 0)
	case ASUBMEVCC:
		return OPVCC(31, 232, 1, 1)
	case ASUBZE:
		return OPVCC(31, 200, 0, 0)
	case ASUBZECC:
		return OPVCC(31, 200, 0, 1)
	case ASUBZEV:
		return OPVCC(31, 200, 1, 0)
	case ASUBZEVCC:
		return OPVCC(31, 200, 1, 1)

	case ASYNC:
		return OPVCC(31, 598, 0, 0)
	case ALWSYNC:
		return OPVCC(31, 598, 0, 0) | 1<<21

	case APTESYNC:
		return OPVCC(31, 598, 0, 0) | 2<<21

	case ATLBIE:
		return OPVCC(31, 306, 0, 0)
	case ATLBIEL:
		return OPVCC(31, 274, 0, 0)
	case ATLBSYNC:
		return OPVCC(31, 566, 0, 0)
	case ASLBIA:
		return OPVCC(31, 498, 0, 0)
	case ASLBIE:
		return OPVCC(31, 434, 0, 0)
	case ASLBMFEE:
		return OPVCC(31, 915, 0, 0)
	case ASLBMFEV:
		return OPVCC(31, 851, 0, 0)
	case ASLBMTE:
		return OPVCC(31, 402, 0, 0)

	case ATW:
		return OPVCC(31, 4, 0, 0)
	case ATD:
		return OPVCC(31, 68, 0, 0)

	/* Vector (VMX/Altivec) instructions */
	/* ISA 2.03 enables these for PPC970. For POWERx processors, these */
	/* are enabled starting at POWER6 (ISA 2.05). */
	case AVANDL:
		return OPVX(4, 1028, 0, 0) /* vand - v2.03 */
	case AVANDC:
		return OPVX(4, 1092, 0, 0) /* vandc - v2.03 */
	case AVNAND:
		return OPVX(4, 1412, 0, 0) /* vnand - v2.07 */

	case AVORL:
		return OPVX(4, 1156, 0, 0) /* vor - v2.03 */
	case AVORC:
		return OPVX(4, 1348, 0, 0) /* vorc - v2.07 */
	case AVNOR:
		return OPVX(4, 1284, 0, 0) /* vnor - v2.03 */
	case AVXOR:
		return OPVX(4, 1220, 0, 0) /* vxor - v2.03 */
	case AVEQV:
		return OPVX(4, 1668, 0, 0) /* veqv - v2.07 */

	case AVADDUBM:
		return OPVX(4, 0, 0, 0) /* vaddubm - v2.03 */
	case AVADDUHM:
		return OPVX(4, 64, 0, 0) /* vadduhm - v2.03 */
	case AVADDUWM:
		return OPVX(4, 128, 0, 0) /* vadduwm - v2.03 */
	case AVADDUDM:
		return OPVX(4, 192, 0, 0) /* vaddudm - v2.07 */
	case AVADDUQM:
		return OPVX(4, 256, 0, 0) /* vadduqm - v2.07 */

	case AVADDCUQ:
		return OPVX(4, 320, 0, 0) /* vaddcuq - v2.07 */
	case AVADDCUW:
		return OPVX(4, 384, 0, 0) /* vaddcuw - v2.03 */

	case AVADDUBS:
		return OPVX(4, 512, 0, 0) /* vaddubs - v2.03 */
	case AVADDUHS:
		return OPVX(4, 576, 0, 0) /* vadduhs - v2.03 */
	case AVADDUWS:
		return OPVX(4, 640, 0, 0) /* vadduws - v2.03 */

	case AVADDSBS:
		return OPVX(4, 768, 0, 0) /* vaddsbs - v2.03 */
	case AVADDSHS:
		return OPVX(4, 832, 0, 0) /* vaddshs - v2.03 */
	case AVADDSWS:
		return OPVX(4, 896, 0, 0) /* vaddsws - v2.03 */

	case AVADDEUQM:
		return OPVX(4, 60, 0, 0) /* vaddeuqm - v2.07 */
	case AVADDECUQ:
		return OPVX(4, 61, 0, 0) /* vaddecuq - v2.07 */

	case AVSUBUBM:
		return OPVX(4, 1024, 0, 0) /* vsububm - v2.03 */
	case AVSUBUHM:
		return OPVX(4, 1088, 0, 0) /* vsubuhm - v2.03 */
	case AVSUBUWM:
		return OPVX(4, 1152, 0, 0) /* vsubuwm - v2.03 */
	case AVSUBUDM:
		return OPVX(4, 1216, 0, 0) /* vsubudm - v2.07 */
	case AVSUBUQM:
		return OPVX(4, 1280, 0, 0) /* vsubuqm - v2.07 */

	case AVSUBCUQ:
		return OPVX(4, 1344, 0, 0) /* vsubcuq - v2.07 */
	case AVSUBCUW:
		return OPVX(4, 1408, 0, 0) /* vsubcuw - v2.03 */

	case AVSUBUBS:
		return OPVX(4, 1536, 0, 0) /* vsububs - v2.03 */
	case AVSUBUHS:
		return OPVX(4, 1600, 0, 0) /* vsubuhs - v2.03 */
	case AVSUBUWS:
		return OPVX(4, 1664, 0, 0) /* vsubuws - v2.03 */

	case AVSUBSBS:
		return OPVX(4, 1792, 0, 0) /* vsubsbs - v2.03 */
	case AVSUBSHS:
		return OPVX(4, 1856, 0, 0) /* vsubshs - v2.03 */
	case AVSUBSWS:
		return OPVX(4, 1920, 0, 0) /* vsubsws - v2.03 */

	case AVSUBEUQM:
		return OPVX(4, 62, 0, 0) /* vsubeuqm - v2.07 */
	case AVSUBECUQ:
		return OPVX(4, 63, 0, 0) /* vsubecuq - v2.07 */

	case AVRLB:
		return OPVX(4, 4, 0, 0) /* vrlb - v2.03 */
	case AVRLH:
		return OPVX(4, 68, 0, 0) /* vrlh - v2.03 */
	case AVRLW:
		return OPVX(4, 132, 0, 0) /* vrlw - v2.03 */
	case AVRLD:
		return OPVX(4, 196, 0, 0) /* vrld - v2.07 */

	case AVSLB:
		return OPVX(4, 260, 0, 0) /* vslh - v2.03 */
	case AVSLH:
		return OPVX(4, 324, 0, 0) /* vslh - v2.03 */
	case AVSLW:
		return OPVX(4, 388, 0, 0) /* vslw - v2.03 */
	case AVSL:
		return OPVX(4, 452, 0, 0) /* vsl - v2.03 */
	case AVSLO:
		return OPVX(4, 1036, 0, 0) /* vsl - v2.03 */
	case AVSRB:
		return OPVX(4, 516, 0, 0) /* vsrb - v2.03 */
	case AVSRH:
		return OPVX(4, 580, 0, 0) /* vsrh - v2.03 */
	case AVSRW:
		return OPVX(4, 644, 0, 0) /* vsrw - v2.03 */
	case AVSR:
		return OPVX(4, 708, 0, 0) /* vsr - v2.03 */
	case AVSRO:
		return OPVX(4, 1100, 0, 0) /* vsro - v2.03 */
	case AVSLD:
		return OPVX(4, 1476, 0, 0) /* vsld - v2.07 */
	case AVSRD:
		return OPVX(4, 1732, 0, 0) /* vsrd - v2.07 */

	case AVSRAB:
		return OPVX(4, 772, 0, 0) /* vsrab - v2.03 */
	case AVSRAH:
		return OPVX(4, 836, 0, 0) /* vsrah - v2.03 */
	case AVSRAW:
		return OPVX(4, 900, 0, 0) /* vsraw - v2.03 */
	case AVSRAD:
		return OPVX(4, 964, 0, 0) /* vsrad - v2.07 */

	case AVCLZB:
		return OPVX(4, 1794, 0, 0) /* vclzb - v2.07 */
	case AVCLZH:
		return OPVX(4, 1858, 0, 0) /* vclzh - v2.07 */
	case AVCLZW:
		return OPVX(4, 1922, 0, 0) /* vclzw - v2.07 */
	case AVCLZD:
		return OPVX(4, 1986, 0, 0) /* vclzd - v2.07 */

	case AVPOPCNTB:
		return OPVX(4, 1795, 0, 0) /* vpopcntb - v2.07 */
	case AVPOPCNTH:
		return OPVX(4, 1859, 0, 0) /* vpopcnth - v2.07 */
	case AVPOPCNTW:
		return OPVX(4, 1923, 0, 0) /* vpopcntw - v2.07 */
	case AVPOPCNTD:
		return OPVX(4, 1987, 0, 0) /* vpopcntd - v2.07 */

	case AVCMPEQUB:
		return OPVC(4, 6, 0, 0) /* vcmpequb - v2.03 */
	case AVCMPEQUBCC:
		return OPVC(4, 6, 0, 1) /* vcmpequb. - v2.03 */
	case AVCMPEQUH:
		return OPVC(4, 70, 0, 0) /* vcmpequh - v2.03 */
	case AVCMPEQUHCC:
		return OPVC(4, 70, 0, 1) /* vcmpequh. - v2.03 */
	case AVCMPEQUW:
		return OPVC(4, 134, 0, 0) /* vcmpequw - v2.03 */
	case AVCMPEQUWCC:
		return OPVC(4, 134, 0, 1) /* vcmpequw. - v2.03 */
	case AVCMPEQUD:
		return OPVC(4, 199, 0, 0) /* vcmpequd - v2.07 */
	case AVCMPEQUDCC:
		return OPVC(4, 199, 0, 1) /* vcmpequd. - v2.07 */

	case AVCMPGTUB:
		return OPVC(4, 518, 0, 0) /* vcmpgtub - v2.03 */
	case AVCMPGTUBCC:
		return OPVC(4, 518, 0, 1) /* vcmpgtub. - v2.03 */
	case AVCMPGTUH:
		return OPVC(4, 582, 0, 0) /* vcmpgtuh - v2.03 */
	case AVCMPGTUHCC:
		return OPVC(4, 582, 0, 1) /* vcmpgtuh. - v2.03 */
	case AVCMPGTUW:
		return OPVC(4, 646, 0, 0) /* vcmpgtuw - v2.03 */
	case AVCMPGTUWCC:
		return OPVC(4, 646, 0, 1) /* vcmpgtuw. - v2.03 */
	case AVCMPGTUD:
		return OPVC(4, 711, 0, 0) /* vcmpgtud - v2.07 */
	case AVCMPGTUDCC:
		return OPVC(4, 711, 0, 1) /* vcmpgtud. v2.07 */
	case AVCMPGTSB:
		return OPVC(4, 774, 0, 0) /* vcmpgtsb - v2.03 */
	case AVCMPGTSBCC:
		return OPVC(4, 774, 0, 1) /* vcmpgtsb. - v2.03 */
	case AVCMPGTSH:
		return OPVC(4, 838, 0, 0) /* vcmpgtsh - v2.03 */
	case AVCMPGTSHCC:
		return OPVC(4, 838, 0, 1) /* vcmpgtsh. - v2.03 */
	case AVCMPGTSW:
		return OPVC(4, 902, 0, 0) /* vcmpgtsw - v2.03 */
	case AVCMPGTSWCC:
		return OPVC(4, 902, 0, 1) /* vcmpgtsw. - v2.03 */
	case AVCMPGTSD:
		return OPVC(4, 967, 0, 0) /* vcmpgtsd - v2.07 */
	case AVCMPGTSDCC:
		return OPVC(4, 967, 0, 1) /* vcmpgtsd. - v2.07 */

	case AVPERM:
		return OPVX(4, 43, 0, 0) /* vperm - v2.03 */

	case AVSEL:
		return OPVX(4, 42, 0, 0) /* vsel - v2.03 */

	case AVCIPHER:
		return OPVX(4, 1288, 0, 0) /* vcipher - v2.07 */
	case AVCIPHERLAST:
		return OPVX(4, 1289, 0, 0) /* vcipherlast - v2.07 */
	case AVNCIPHER:
		return OPVX(4, 1352, 0, 0) /* vncipher - v2.07 */
	case AVNCIPHERLAST:
		return OPVX(4, 1353, 0, 0) /* vncipherlast - v2.07 */
	case AVSBOX:
		return OPVX(4, 1480, 0, 0) /* vsbox - v2.07 */
	/* End of vector instructions */

	/* Vector scalar (VSX) instructions */
	/* ISA 2.06 enables these for POWER7. */
	case AMFVSRD:
		return OPVXX1(31, 51, 0) /* mfvsrd - v2.07 */
	case AMFVSRWZ:
		return OPVXX1(31, 115, 0) /* mfvsrwz - v2.07 */

	case AMTVSRD:
		return OPVXX1(31, 179, 0) /* mtvsrd - v2.07 */
	case AMTVSRWA:
		return OPVXX1(31, 211, 0) /* mtvsrwa - v2.07 */
	case AMTVSRWZ:
		return OPVXX1(31, 243, 0) /* mtvsrwz - v2.07 */

	case AXXLANDQ:
		return OPVXX3(60, 130, 0) /* xxland - v2.06 */
	case AXXLANDC:
		return OPVXX3(60, 138, 0) /* xxlandc - v2.06 */
	case AXXLEQV:
		return OPVXX3(60, 186, 0) /* xxleqv - v2.07 */
	case AXXLNAND:
		return OPVXX3(60, 178, 0) /* xxlnand - v2.07 */

	case AXXLORC:
		return OPVXX3(60, 170, 0) /* xxlorc - v2.07 */
	case AXXLNOR:
		return OPVXX3(60, 162, 0) /* xxlnor - v2.06 */
	case AXXLORQ:
		return OPVXX3(60, 146, 0) /* xxlor - v2.06 */
	case AXXLXOR:
		return OPVXX3(60, 154, 0) /* xxlxor - v2.06 */

	case AXXSEL:
		return OPVXX4(60, 3, 0) /* xxsel - v2.06 */

	case AXXMRGHW:
		return OPVXX3(60, 18, 0) /* xxmrghw - v2.06 */
	case AXXMRGLW:
		return OPVXX3(60, 50, 0) /* xxmrglw - v2.06 */

	case AXXSPLTW:
		return OPVXX2(60, 164, 0) /* xxspltw - v2.06 */

	case AXXPERMDI:
		return OPVXX3(60, 10, 0) /* xxpermdi - v2.06 */

	case AXXSLDWI:
		return OPVXX3(60, 2, 0) /* xxsldwi - v2.06 */

	case AXSCVDPSP:
		return OPVXX2(60, 265, 0) /* xscvdpsp - v2.06 */
	case AXSCVSPDP:
		return OPVXX2(60, 329, 0) /* xscvspdp - v2.06 */
	case AXSCVDPSPN:
		return OPVXX2(60, 267, 0) /* xscvdpspn - v2.07 */
	case AXSCVSPDPN:
		return OPVXX2(60, 331, 0) /* xscvspdpn - v2.07 */

	case AXVCVDPSP:
		return OPVXX2(60, 393, 0) /* xvcvdpsp - v2.06 */
	case AXVCVSPDP:
		return OPVXX2(60, 457, 0) /* xvcvspdp - v2.06 */

	case AXSCVDPSXDS:
		return OPVXX2(60, 344, 0) /* xscvdpsxds - v2.06 */
	case AXSCVDPSXWS:
		return OPVXX2(60, 88, 0) /* xscvdpsxws - v2.06 */
	case AXSCVDPUXDS:
		return OPVXX2(60, 328, 0) /* xscvdpuxds - v2.06 */
	case AXSCVDPUXWS:
		return OPVXX2(60, 72, 0) /* xscvdpuxws - v2.06 */

	case AXSCVSXDDP:
		return OPVXX2(60, 376, 0) /* xscvsxddp - v2.06 */
	case AXSCVUXDDP:
		return OPVXX2(60, 360, 0) /* xscvuxddp - v2.06 */
	case AXSCVSXDSP:
		return OPVXX2(60, 312, 0) /* xscvsxdsp - v2.06 */
	case AXSCVUXDSP:
		return OPVXX2(60, 296, 0) /* xscvuxdsp - v2.06 */

	case AXVCVDPSXDS:
		return OPVXX2(60, 472, 0) /* xvcvdpsxds - v2.06 */
	case AXVCVDPSXWS:
		return OPVXX2(60, 216, 0) /* xvcvdpsxws - v2.06 */
	case AXVCVDPUXDS:
		return OPVXX2(60, 456, 0) /* xvcvdpuxds - v2.06 */
	case AXVCVDPUXWS:
		return OPVXX2(60, 200, 0) /* xvcvdpuxws - v2.06 */
	case AXVCVSPSXDS:
		return OPVXX2(60, 408, 0) /* xvcvspsxds - v2.07 */
	case AXVCVSPSXWS:
		return OPVXX2(60, 152, 0) /* xvcvspsxws - v2.07 */
	case AXVCVSPUXDS:
		return OPVXX2(60, 392, 0) /* xvcvspuxds - v2.07 */
	case AXVCVSPUXWS:
		return OPVXX2(60, 136, 0) /* xvcvspuxws - v2.07 */

	case AXVCVSXDDP:
		return OPVXX2(60, 504, 0) /* xvcvsxddp - v2.06 */
	case AXVCVSXWDP:
		return OPVXX2(60, 248, 0) /* xvcvsxwdp - v2.06 */
	case AXVCVUXDDP:
		return OPVXX2(60, 488, 0) /* xvcvuxddp - v2.06 */
	case AXVCVUXWDP:
		return OPVXX2(60, 232, 0) /* xvcvuxwdp - v2.06 */
	case AXVCVSXDSP:
		return OPVXX2(60, 440, 0) /* xvcvsxdsp - v2.06 */
	case AXVCVSXWSP:
		return OPVXX2(60, 184, 0) /* xvcvsxwsp - v2.06 */
	case AXVCVUXDSP:
		return OPVXX2(60, 424, 0) /* xvcvuxdsp - v2.06 */
	case AXVCVUXWSP:
		return OPVXX2(60, 168, 0) /* xvcvuxwsp - v2.06 */
	/* End of VSX instructions */

	case AXOR:
		return OPVCC(31, 316, 0, 0)
	case AXORCC:
		return OPVCC(31, 316, 0, 1)
	}

	ctxt.Diag("bad r/r, r/r/r or r/r/r/r opcode %v", a)
	return 0
}

func opirrr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	/* Vector (VMX/Altivec) instructions */
	/* ISA 2.03 enables these for PPC970. For POWERx processors, these */
	/* are enabled starting at POWER6 (ISA 2.05). */
	case AVSLDOI:
		return OPVX(4, 44, 0, 0) /* vsldoi - v2.03 */
	}

	ctxt.Diag("bad i/r/r/r opcode %v", a)
	return 0
}

func opiirr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	/* Vector (VMX/Altivec) instructions */
	/* ISA 2.07 enables these for POWER8 and beyond. */
	case AVSHASIGMAW:
		return OPVX(4, 1666, 0, 0) /* vshasigmaw - v2.07 */
	case AVSHASIGMAD:
		return OPVX(4, 1730, 0, 0) /* vshasigmad - v2.07 */
	}

	ctxt.Diag("bad i/i/r/r opcode %v", a)
	return 0
}

func opirr(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AADD:
		return OPVCC(14, 0, 0, 0)
	case AADDC:
		return OPVCC(12, 0, 0, 0)
	case AADDCCC:
		return OPVCC(13, 0, 0, 0)
	case -AADD:
		return OPVCC(15, 0, 0, 0) /* ADDIS/CAU */

	case AANDCC:
		return OPVCC(28, 0, 0, 0)
	case -AANDCC:
		return OPVCC(29, 0, 0, 0) /* ANDIS./ANDIU. */

	case ABR:
		return OPVCC(18, 0, 0, 0)
	case ABL:
		return OPVCC(18, 0, 0, 0) | 1
	case obj.ADUFFZERO:
		return OPVCC(18, 0, 0, 0) | 1
	case obj.ADUFFCOPY:
		return OPVCC(18, 0, 0, 0) | 1
	case ABC:
		return OPVCC(16, 0, 0, 0)
	case ABCL:
		return OPVCC(16, 0, 0, 0) | 1

	case ABEQ:
		return AOP_RRR(16<<26, 12, 2, 0)
	case ABGE:
		return AOP_RRR(16<<26, 4, 0, 0)
	case ABGT:
		return AOP_RRR(16<<26, 12, 1, 0)
	case ABLE:
		return AOP_RRR(16<<26, 4, 1, 0)
	case ABLT:
		return AOP_RRR(16<<26, 12, 0, 0)
	case ABNE:
		return AOP_RRR(16<<26, 4, 2, 0)
	case ABVC:
		return AOP_RRR(16<<26, 4, 3, 0) // apparently unordered-clear
	case ABVS:
		return AOP_RRR(16<<26, 12, 3, 0) // apparently unordered-set

	case ACMP:
		return OPVCC(11, 0, 0, 0) | 1<<21 /* L=1 */
	case ACMPU:
		return OPVCC(10, 0, 0, 0) | 1<<21
	case ACMPW:
		return OPVCC(11, 0, 0, 0) /* L=0 */
	case ACMPWU:
		return OPVCC(10, 0, 0, 0)
	case ALSW:
		return OPVCC(31, 597, 0, 0)

	case AMULLW:
		return OPVCC(7, 0, 0, 0)

	case AOR:
		return OPVCC(24, 0, 0, 0)
	case -AOR:
		return OPVCC(25, 0, 0, 0) /* ORIS/ORIU */

	case ARLWMI:
		return OPVCC(20, 0, 0, 0) /* rlwimi */
	case ARLWMICC:
		return OPVCC(20, 0, 0, 1)
	case ARLDMI:
		return OPVCC(30, 0, 0, 0) | 3<<2 /* rldimi */
	case ARLDMICC:
		return OPVCC(30, 0, 0, 1) | 3<<2
	case ARLDIMI:
		return OPVCC(30, 0, 0, 0) | 3<<2 /* rldimi */
	case ARLDIMICC:
		return OPVCC(30, 0, 0, 1) | 3<<2
	case ARLWNM:
		return OPVCC(21, 0, 0, 0) /* rlwinm */
	case ARLWNMCC:
		return OPVCC(21, 0, 0, 1)

	case ARLDCL:
		return OPVCC(30, 0, 0, 0) /* rldicl */
	case ARLDCLCC:
		return OPVCC(30, 0, 0, 1)
	case ARLDCR:
		return OPVCC(30, 1, 0, 0) /* rldicr */
	case ARLDCRCC:
		return OPVCC(30, 1, 0, 1)
	case ARLDC:
		return OPVCC(30, 0, 0, 0) | 2<<2
	case ARLDCCC:
		return OPVCC(30, 0, 0, 1) | 2<<2

	case ASRAW:
		return OPVCC(31, 824, 0, 0)
	case ASRAWCC:
		return OPVCC(31, 824, 0, 1)
	case ASRAD:
		return OPVCC(31, (413 << 1), 0, 0)
	case ASRADCC:
		return OPVCC(31, (413 << 1), 0, 1)

	case ASTSW:
		return OPVCC(31, 725, 0, 0)

	case ASUBC:
		return OPVCC(8, 0, 0, 0)

	case ATW:
		return OPVCC(3, 0, 0, 0)
	case ATD:
		return OPVCC(2, 0, 0, 0)

	/* Vector (VMX/Altivec) instructions */
	/* ISA 2.03 enables these for PPC970. For POWERx processors, these */
	/* are enabled starting at POWER6 (ISA 2.05). */
	case AVSPLTB:
		return OPVX(4, 524, 0, 0) /* vspltb - v2.03 */
	case AVSPLTH:
		return OPVX(4, 588, 0, 0) /* vsplth - v2.03 */
	case AVSPLTW:
		return OPVX(4, 652, 0, 0) /* vspltw - v2.03 */

	case AVSPLTISB:
		return OPVX(4, 780, 0, 0) /* vspltisb - v2.03 */
	case AVSPLTISH:
		return OPVX(4, 844, 0, 0) /* vspltish - v2.03 */
	case AVSPLTISW:
		return OPVX(4, 908, 0, 0) /* vspltisw - v2.03 */
	/* End of vector instructions */

	case AXOR:
		return OPVCC(26, 0, 0, 0) /* XORIL */
	case -AXOR:
		return OPVCC(27, 0, 0, 0) /* XORIU */
	}

	ctxt.Diag("bad opcode i/r or i/r/r %v", a)
	return 0
}

/*
 * load o(a),d
 */
func opload(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AMOVD:
		return OPVCC(58, 0, 0, 0) /* ld */
	case AMOVDU:
		return OPVCC(58, 0, 0, 1) /* ldu */
	case AMOVWZ:
		return OPVCC(32, 0, 0, 0) /* lwz */
	case AMOVWZU:
		return OPVCC(33, 0, 0, 0) /* lwzu */
	case AMOVW:
		return OPVCC(58, 0, 0, 0) | 1<<1 /* lwa */

		/* no AMOVWU */
	case AMOVB, AMOVBZ:
		return OPVCC(34, 0, 0, 0)
		/* load */

	case AMOVBU, AMOVBZU:
		return OPVCC(35, 0, 0, 0)
	case AFMOVD:
		return OPVCC(50, 0, 0, 0)
	case AFMOVDU:
		return OPVCC(51, 0, 0, 0)
	case AFMOVS:
		return OPVCC(48, 0, 0, 0)
	case AFMOVSU:
		return OPVCC(49, 0, 0, 0)
	case AMOVH:
		return OPVCC(42, 0, 0, 0)
	case AMOVHU:
		return OPVCC(43, 0, 0, 0)
	case AMOVHZ:
		return OPVCC(40, 0, 0, 0)
	case AMOVHZU:
		return OPVCC(41, 0, 0, 0)
	case AMOVMW:
		return OPVCC(46, 0, 0, 0) /* lmw */
	}

	ctxt.Diag("bad load opcode %v", a)
	return 0
}

/*
 * indexed load a(b),d
 */
func oploadx(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AMOVWZ:
		return OPVCC(31, 23, 0, 0) /* lwzx */
	case AMOVWZU:
		return OPVCC(31, 55, 0, 0) /* lwzux */
	case AMOVW:
		return OPVCC(31, 341, 0, 0) /* lwax */
	case AMOVWU:
		return OPVCC(31, 373, 0, 0) /* lwaux */

	case AMOVB, AMOVBZ:
		return OPVCC(31, 87, 0, 0) /* lbzx */

	case AMOVBU, AMOVBZU:
		return OPVCC(31, 119, 0, 0) /* lbzux */
	case AFMOVD:
		return OPVCC(31, 599, 0, 0) /* lfdx */
	case AFMOVDU:
		return OPVCC(31, 631, 0, 0) /*  lfdux */
	case AFMOVS:
		return OPVCC(31, 535, 0, 0) /* lfsx */
	case AFMOVSU:
		return OPVCC(31, 567, 0, 0) /* lfsux */
	case AFMOVSX:
		return OPVCC(31, 855, 0, 0) /* lfiwax - power6, isa 2.05 */
	case AFMOVSZ:
		return OPVCC(31, 887, 0, 0) /* lfiwzx - power7, isa 2.06 */
	case AMOVH:
		return OPVCC(31, 343, 0, 0) /* lhax */
	case AMOVHU:
		return OPVCC(31, 375, 0, 0) /* lhaux */
	case AMOVHBR:
		return OPVCC(31, 790, 0, 0) /* lhbrx */
	case AMOVWBR:
		return OPVCC(31, 534, 0, 0) /* lwbrx */
	case AMOVDBR:
		return OPVCC(31, 532, 0, 0) /* ldbrx */
	case AMOVHZ:
		return OPVCC(31, 279, 0, 0) /* lhzx */
	case AMOVHZU:
		return OPVCC(31, 311, 0, 0) /* lhzux */
	case AECIWX:
		return OPVCC(31, 310, 0, 0) /* eciwx */
	case ALBAR:
		return OPVCC(31, 52, 0, 0) /* lbarx */
	case ALWAR:
		return OPVCC(31, 20, 0, 0) /* lwarx */
	case ALDAR:
		return OPVCC(31, 84, 0, 0)
	case ALSW:
		return OPVCC(31, 533, 0, 0) /* lswx */
	case AMOVD:
		return OPVCC(31, 21, 0, 0) /* ldx */
	case AMOVDU:
		return OPVCC(31, 53, 0, 0) /* ldux */

	/* Vector (VMX/Altivec) instructions */
	/* ISA 2.03 enables these for PPC970. For POWERx processors, these */
	/* are enabled starting at POWER6 (ISA 2.05). */
	case ALVEBX:
		return OPVCC(31, 7, 0, 0) /* lvebx - v2.03 */
	case ALVEHX:
		return OPVCC(31, 39, 0, 0) /* lvehx - v2.03 */
	case ALVEWX:
		return OPVCC(31, 71, 0, 0) /* lvewx - v2.03 */
	case ALVX:
		return OPVCC(31, 103, 0, 0) /* lvx - v2.03 */
	case ALVXL:
		return OPVCC(31, 359, 0, 0) /* lvxl - v2.03 */
	case ALVSL:
		return OPVCC(31, 6, 0, 0) /* lvsl - v2.03 */
	case ALVSR:
		return OPVCC(31, 38, 0, 0) /* lvsr - v2.03 */
		/* End of vector instructions */

	/* Vector scalar (VSX) instructions */
	/* ISA 2.06 enables these for POWER7. */
	case ALXVD2X:
		return OPVXX1(31, 844, 0) /* lxvd2x - v2.06 */
	case ALXVDSX:
		return OPVXX1(31, 332, 0) /* lxvdsx - v2.06 */
	case ALXVW4X:
		return OPVXX1(31, 780, 0) /* lxvw4x - v2.06 */

	case ALXSDX:
		return OPVXX1(31, 588, 0) /* lxsdx - v2.06 */

	case ALXSIWAX:
		return OPVXX1(31, 76, 0) /* lxsiwax - v2.07 */
	case ALXSIWZX:
		return OPVXX1(31, 12, 0) /* lxsiwzx - v2.07 */
		/* End of vector scalar instructions */

	}

	ctxt.Diag("bad loadx opcode %v", a)
	return 0
}

/*
 * store s,o(d)
 */
func opstore(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AMOVB, AMOVBZ:
		return OPVCC(38, 0, 0, 0) /* stb */

	case AMOVBU, AMOVBZU:
		return OPVCC(39, 0, 0, 0) /* stbu */
	case AFMOVD:
		return OPVCC(54, 0, 0, 0) /* stfd */
	case AFMOVDU:
		return OPVCC(55, 0, 0, 0) /* stfdu */
	case AFMOVS:
		return OPVCC(52, 0, 0, 0) /* stfs */
	case AFMOVSU:
		return OPVCC(53, 0, 0, 0) /* stfsu */

	case AMOVHZ, AMOVH:
		return OPVCC(44, 0, 0, 0) /* sth */

	case AMOVHZU, AMOVHU:
		return OPVCC(45, 0, 0, 0) /* sthu */
	case AMOVMW:
		return OPVCC(47, 0, 0, 0) /* stmw */
	case ASTSW:
		return OPVCC(31, 725, 0, 0) /* stswi */

	case AMOVWZ, AMOVW:
		return OPVCC(36, 0, 0, 0) /* stw */

	case AMOVWZU, AMOVWU:
		return OPVCC(37, 0, 0, 0) /* stwu */
	case AMOVD:
		return OPVCC(62, 0, 0, 0) /* std */
	case AMOVDU:
		return OPVCC(62, 0, 0, 1) /* stdu */
	}

	ctxt.Diag("unknown store opcode %v", a)
	return 0
}

/*
 * indexed store s,a(b)
 */
func opstorex(ctxt *obj.Link, a obj.As) uint32 {
	switch a {
	case AMOVB, AMOVBZ:
		return OPVCC(31, 215, 0, 0) /* stbx */

	case AMOVBU, AMOVBZU:
		return OPVCC(31, 247, 0, 0) /* stbux */
	case AFMOVD:
		return OPVCC(31, 727, 0, 0) /* stfdx */
	case AFMOVDU:
		return OPVCC(31, 759, 0, 0) /* stfdux */
	case AFMOVS:
		return OPVCC(31, 663, 0, 0) /* stfsx */
	case AFMOVSU:
		return OPVCC(31, 695, 0, 0) /* stfsux */
	case AFMOVSX:
		return OPVCC(31, 983, 0, 0) /* stfiwx */

	case AMOVHZ, AMOVH:
		return OPVCC(31, 407, 0, 0) /* sthx */
	case AMOVHBR:
		return OPVCC(31, 918, 0, 0) /* sthbrx */

	case AMOVHZU, AMOVHU:
		return OPVCC(31, 439, 0, 0) /* sthux */

	case AMOVWZ, AMOVW:
		return OPVCC(31, 151, 0, 0) /* stwx */

	case AMOVWZU, AMOVWU:
		return OPVCC(31, 183, 0, 0) /* stwux */
	case ASTSW:
		return OPVCC(31, 661, 0, 0) /* stswx */
	case AMOVWBR:
		return OPVCC(31, 662, 0, 0) /* stwbrx */
	case ASTBCCC:
		return OPVCC(31, 694, 0, 1) /* stbcx. */
	case ASTWCCC:
		return OPVCC(31, 150, 0, 1) /* stwcx. */
	case ASTDCCC:
		return OPVCC(31, 214, 0, 1) /* stwdx. */
	case AECOWX:
		return OPVCC(31, 438, 0, 0) /* ecowx */
	case AMOVD:
		return OPVCC(31, 149, 0, 0) /* stdx */
	case AMOVDU:
		return OPVCC(31, 181, 0, 0) /* stdux */

	/* Vector (VMX/Altivec) instructions */
	/* ISA 2.03 enables these for PPC970. For POWERx processors, these */
	/* are enabled starting at POWER6 (ISA 2.05). */
	case ASTVEBX:
		return OPVCC(31, 135, 0, 0) /* stvebx - v2.03 */
	case ASTVEHX:
		return OPVCC(31, 167, 0, 0) /* stvehx - v2.03 */
	case ASTVEWX:
		return OPVCC(31, 199, 0, 0) /* stvewx - v2.03 */
	case ASTVX:
		return OPVCC(31, 231, 0, 0) /* stvx - v2.03 */
	case ASTVXL:
		return OPVCC(31, 487, 0, 0) /* stvxl - v2.03 */
		/* End of vector instructions */

	/* Vector scalar (VSX) instructions */
	/* ISA 2.06 enables these for POWER7. */
	case ASTXVD2X:
		return OPVXX1(31, 972, 0) /* stxvd2x - v2.06 */
	case ASTXVW4X:
		return OPVXX1(31, 908, 0) /* stxvw4x - v2.06 */

	case ASTXSDX:
		return OPVXX1(31, 716, 0) /* stxsdx - v2.06 */

	case ASTXSIWX:
		return OPVXX1(31, 140, 0) /* stxsiwx - v2.07 */
		/* End of vector scalar instructions */

	}

	ctxt.Diag("unknown storex opcode %v", a)
	return 0
}
