// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"fmt"
	"log"
	"slices"
)

// ctxt0 holds state while assembling a single function.
// Each function gets a fresh ctxt0.
// This allows for multiple functions to be safely concurrently assembled.
type ctxt0 struct {
	ctxt       *obj.Link
	newprog    obj.ProgAlloc
	cursym     *obj.LSym
	autosize   int32
	instoffset int64
	pc         int64
}

// Instruction layout.

const (
	FuncAlign = 4
	loopAlign = 16
)

type Optab struct {
	as    obj.As
	from1 uint8
	reg   uint8
	from3 uint8
	to1   uint8
	to2   uint8
	type_ int8
	size  int8
	param int16
	flag  uint8
}

const (
	NOTUSETMP = 1 << iota // p expands to multiple instructions, but does NOT use REGTMP

	// branchLoopHead marks loop entry.
	// Used to insert padding for under-aligned loops.
	branchLoopHead
)

var optab = []Optab{
	{obj.ATEXT, C_ADDR, C_NONE, C_NONE, C_TEXTSIZE, C_NONE, 0, 0, 0, 0},

	{AMOVW, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 1, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 1, 4, 0, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 12, 4, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 12, 4, 0, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 12, 4, 0, 0},

	{ASUB, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ASUBV, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AADD, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AADDV, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AAND, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ASUB, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ASUBV, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AADD, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AADDV, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AAND, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ANEGW, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ANEGV, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AMASKEQZ, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ASLL, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ASLL, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ASLLV, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{ASLLV, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AMUL, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AMUL, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AMULV, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AMULV, C_REG, C_REG, C_NONE, C_REG, C_NONE, 2, 4, 0, 0},
	{AADDF, C_FREG, C_NONE, C_NONE, C_FREG, C_NONE, 2, 4, 0, 0},
	{AADDF, C_FREG, C_FREG, C_NONE, C_FREG, C_NONE, 2, 4, 0, 0},
	{ACMPEQF, C_FREG, C_FREG, C_NONE, C_FCCREG, C_NONE, 2, 4, 0, 0},
	{AVSEQB, C_VREG, C_VREG, C_NONE, C_VREG, C_NONE, 2, 4, 0, 0},
	{AXVSEQB, C_XREG, C_XREG, C_NONE, C_XREG, C_NONE, 2, 4, 0, 0},

	{ACLOW, C_REG, C_NONE, C_NONE, C_REG, C_NONE, 9, 4, 0, 0},
	{AABSF, C_FREG, C_NONE, C_NONE, C_FREG, C_NONE, 9, 4, 0, 0},
	{AMOVVF, C_FREG, C_NONE, C_NONE, C_FREG, C_NONE, 9, 4, 0, 0},
	{AMOVF, C_FREG, C_NONE, C_NONE, C_FREG, C_NONE, 9, 4, 0, 0},
	{AMOVD, C_FREG, C_NONE, C_NONE, C_FREG, C_NONE, 9, 4, 0, 0},
	{AVPCNTB, C_VREG, C_NONE, C_NONE, C_VREG, C_NONE, 9, 4, 0, 0},
	{AXVPCNTB, C_XREG, C_NONE, C_NONE, C_XREG, C_NONE, 9, 4, 0, 0},

	{AFMADDF, C_FREG, C_FREG, C_NONE, C_FREG, C_NONE, 37, 4, 0, 0},
	{AFMADDF, C_FREG, C_FREG, C_FREG, C_FREG, C_NONE, 37, 4, 0, 0},

	{AMOVW, C_REG, C_NONE, C_NONE, C_SAUTO, C_NONE, 7, 4, REGSP, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_SAUTO, C_NONE, 7, 4, REGSP, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_SAUTO, C_NONE, 7, 4, REGSP, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_SAUTO, C_NONE, 7, 4, REGSP, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_SAUTO, C_NONE, 7, 4, REGSP, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{AVMOVQ, C_VREG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{AXVMOVQ, C_XREG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{AVMOVQ, C_VREG, C_NONE, C_NONE, C_SAUTO, C_NONE, 7, 4, REGZERO, 0},
	{AXVMOVQ, C_XREG, C_NONE, C_NONE, C_SAUTO, C_NONE, 7, 4, REGZERO, 0},
	{ASC, C_REG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},
	{ASCV, C_REG, C_NONE, C_NONE, C_SOREG, C_NONE, 7, 4, REGZERO, 0},

	{AMOVW, C_SAUTO, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGSP, 0},
	{AMOVWU, C_SAUTO, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGSP, 0},
	{AMOVV, C_SAUTO, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGSP, 0},
	{AMOVB, C_SAUTO, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGSP, 0},
	{AMOVBU, C_SAUTO, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGSP, 0},
	{AMOVW, C_SOREG, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGZERO, 0},
	{AMOVWU, C_SOREG, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGZERO, 0},
	{AMOVV, C_SOREG, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGZERO, 0},
	{AMOVB, C_SOREG, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGZERO, 0},
	{AMOVBU, C_SOREG, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGZERO, 0},
	{AVMOVQ, C_SOREG, C_NONE, C_NONE, C_VREG, C_NONE, 8, 4, REGZERO, 0},
	{AXVMOVQ, C_SOREG, C_NONE, C_NONE, C_XREG, C_NONE, 8, 4, REGZERO, 0},
	{AVMOVQ, C_SAUTO, C_NONE, C_NONE, C_VREG, C_NONE, 8, 4, REGZERO, 0},
	{AXVMOVQ, C_SAUTO, C_NONE, C_NONE, C_XREG, C_NONE, 8, 4, REGZERO, 0},
	{ALL, C_SOREG, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGZERO, 0},
	{ALLV, C_SOREG, C_NONE, C_NONE, C_REG, C_NONE, 8, 4, REGZERO, 0},

	{AMOVW, C_REG, C_NONE, C_NONE, C_LAUTO, C_NONE, 35, 12, REGSP, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_LAUTO, C_NONE, 35, 12, REGSP, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_LAUTO, C_NONE, 35, 12, REGSP, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_LAUTO, C_NONE, 35, 12, REGSP, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_LAUTO, C_NONE, 35, 12, REGSP, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_LOREG, C_NONE, 35, 12, REGZERO, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_LOREG, C_NONE, 35, 12, REGZERO, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_LOREG, C_NONE, 35, 12, REGZERO, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_LOREG, C_NONE, 35, 12, REGZERO, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_LOREG, C_NONE, 35, 12, REGZERO, 0},
	{ASC, C_REG, C_NONE, C_NONE, C_LOREG, C_NONE, 35, 12, REGZERO, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_ADDR, C_NONE, 50, 8, 0, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_ADDR, C_NONE, 50, 8, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_ADDR, C_NONE, 50, 8, 0, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_ADDR, C_NONE, 50, 8, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_ADDR, C_NONE, 50, 8, 0, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_TLS_LE, C_NONE, 53, 16, 0, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_TLS_LE, C_NONE, 53, 16, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_TLS_LE, C_NONE, 53, 16, 0, 0},
	{AMOVB, C_REG, C_NONE, C_NONE, C_TLS_LE, C_NONE, 53, 16, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_TLS_LE, C_NONE, 53, 16, 0, 0},

	{AMOVW, C_LAUTO, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGSP, 0},
	{AMOVWU, C_LAUTO, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGSP, 0},
	{AMOVV, C_LAUTO, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGSP, 0},
	{AMOVB, C_LAUTO, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGSP, 0},
	{AMOVBU, C_LAUTO, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGSP, 0},
	{AMOVW, C_LOREG, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGZERO, 0},
	{AMOVWU, C_LOREG, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGZERO, 0},
	{AMOVV, C_LOREG, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGZERO, 0},
	{AMOVB, C_LOREG, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGZERO, 0},
	{AMOVBU, C_LOREG, C_NONE, C_NONE, C_REG, C_NONE, 36, 12, REGZERO, 0},
	{AMOVW, C_ADDR, C_NONE, C_NONE, C_REG, C_NONE, 51, 8, 0, 0},
	{AMOVWU, C_ADDR, C_NONE, C_NONE, C_REG, C_NONE, 51, 8, 0, 0},
	{AMOVV, C_ADDR, C_NONE, C_NONE, C_REG, C_NONE, 51, 8, 0, 0},
	{AMOVB, C_ADDR, C_NONE, C_NONE, C_REG, C_NONE, 51, 8, 0, 0},
	{AMOVBU, C_ADDR, C_NONE, C_NONE, C_REG, C_NONE, 51, 8, 0, 0},
	{AMOVW, C_TLS_LE, C_NONE, C_NONE, C_REG, C_NONE, 54, 16, 0, 0},
	{AMOVWU, C_TLS_LE, C_NONE, C_NONE, C_REG, C_NONE, 54, 16, 0, 0},
	{AMOVV, C_TLS_LE, C_NONE, C_NONE, C_REG, C_NONE, 54, 16, 0, 0},
	{AMOVB, C_TLS_LE, C_NONE, C_NONE, C_REG, C_NONE, 54, 16, 0, 0},
	{AMOVBU, C_TLS_LE, C_NONE, C_NONE, C_REG, C_NONE, 54, 16, 0, 0},

	{AMOVW, C_SACON, C_NONE, C_NONE, C_REG, C_NONE, 3, 4, REGSP, 0},
	{AMOVV, C_SACON, C_NONE, C_NONE, C_REG, C_NONE, 3, 4, REGSP, 0},
	{AMOVW, C_EXTADDR, C_NONE, C_NONE, C_REG, C_NONE, 52, 8, 0, NOTUSETMP},
	{AMOVV, C_EXTADDR, C_NONE, C_NONE, C_REG, C_NONE, 52, 8, 0, NOTUSETMP},

	{AMOVW, C_LACON, C_NONE, C_NONE, C_REG, C_NONE, 26, 12, REGSP, 0},
	{AMOVV, C_LACON, C_NONE, C_NONE, C_REG, C_NONE, 26, 12, REGSP, 0},
	{AMOVW, C_ADDCON, C_NONE, C_NONE, C_REG, C_NONE, 3, 4, REGZERO, 0},
	{AMOVV, C_ADDCON, C_NONE, C_NONE, C_REG, C_NONE, 3, 4, REGZERO, 0},
	{AMOVW, C_ANDCON, C_NONE, C_NONE, C_REG, C_NONE, 3, 4, REGZERO, 0},
	{AMOVV, C_ANDCON, C_NONE, C_NONE, C_REG, C_NONE, 3, 4, REGZERO, 0},

	{AMOVW, C_UCON, C_NONE, C_NONE, C_REG, C_NONE, 24, 4, 0, 0},
	{AMOVV, C_UCON, C_NONE, C_NONE, C_REG, C_NONE, 24, 4, 0, 0},
	{AMOVW, C_LCON, C_NONE, C_NONE, C_REG, C_NONE, 19, 8, 0, NOTUSETMP},
	{AMOVV, C_LCON, C_NONE, C_NONE, C_REG, C_NONE, 19, 8, 0, NOTUSETMP},
	{AMOVV, C_DCON, C_NONE, C_NONE, C_REG, C_NONE, 59, 16, 0, NOTUSETMP},

	{AADD, C_ADD0CON, C_REG, C_NONE, C_REG, C_NONE, 4, 4, 0, 0},
	{AADD, C_ADD0CON, C_NONE, C_NONE, C_REG, C_NONE, 4, 4, 0, 0},
	{AADD, C_ANDCON, C_REG, C_NONE, C_REG, C_NONE, 10, 8, 0, 0},
	{AADD, C_ANDCON, C_NONE, C_NONE, C_REG, C_NONE, 10, 8, 0, 0},

	{AADDV, C_ADD0CON, C_REG, C_NONE, C_REG, C_NONE, 4, 4, 0, 0},
	{AADDV, C_ADD0CON, C_NONE, C_NONE, C_REG, C_NONE, 4, 4, 0, 0},
	{AADDV, C_ANDCON, C_REG, C_NONE, C_REG, C_NONE, 10, 8, 0, 0},
	{AADDV, C_ANDCON, C_NONE, C_NONE, C_REG, C_NONE, 10, 8, 0, 0},

	{AAND, C_AND0CON, C_REG, C_NONE, C_REG, C_NONE, 4, 4, 0, 0},
	{AAND, C_AND0CON, C_NONE, C_NONE, C_REG, C_NONE, 4, 4, 0, 0},
	{AAND, C_ADDCON, C_REG, C_NONE, C_REG, C_NONE, 10, 8, 0, 0},
	{AAND, C_ADDCON, C_NONE, C_NONE, C_REG, C_NONE, 10, 8, 0, 0},

	{AADD, C_UCON, C_REG, C_NONE, C_REG, C_NONE, 25, 8, 0, 0},
	{AADD, C_UCON, C_NONE, C_NONE, C_REG, C_NONE, 25, 8, 0, 0},
	{AADDV, C_UCON, C_REG, C_NONE, C_REG, C_NONE, 25, 8, 0, 0},
	{AADDV, C_UCON, C_NONE, C_NONE, C_REG, C_NONE, 25, 8, 0, 0},
	{AAND, C_UCON, C_REG, C_NONE, C_REG, C_NONE, 25, 8, 0, 0},
	{AAND, C_UCON, C_NONE, C_NONE, C_REG, C_NONE, 25, 8, 0, 0},

	{AADD, C_LCON, C_NONE, C_NONE, C_REG, C_NONE, 23, 12, 0, 0},
	{AADDV, C_LCON, C_NONE, C_NONE, C_REG, C_NONE, 23, 12, 0, 0},
	{AAND, C_LCON, C_NONE, C_NONE, C_REG, C_NONE, 23, 12, 0, 0},
	{AADD, C_LCON, C_REG, C_NONE, C_REG, C_NONE, 23, 12, 0, 0},
	{AADDV, C_LCON, C_REG, C_NONE, C_REG, C_NONE, 23, 12, 0, 0},
	{AAND, C_LCON, C_REG, C_NONE, C_REG, C_NONE, 23, 12, 0, 0},

	{AADDV, C_DCON, C_NONE, C_NONE, C_REG, C_NONE, 60, 20, 0, 0},
	{AADDV, C_DCON, C_REG, C_NONE, C_REG, C_NONE, 60, 20, 0, 0},

	{ASLL, C_SCON, C_REG, C_NONE, C_REG, C_NONE, 16, 4, 0, 0},
	{ASLL, C_SCON, C_NONE, C_NONE, C_REG, C_NONE, 16, 4, 0, 0},

	{ASLLV, C_SCON, C_REG, C_NONE, C_REG, C_NONE, 16, 4, 0, 0},
	{ASLLV, C_SCON, C_NONE, C_NONE, C_REG, C_NONE, 16, 4, 0, 0},

	{ABSTRPICKW, C_SCON, C_REG, C_SCON, C_REG, C_NONE, 17, 4, 0, 0},
	{ABSTRPICKW, C_SCON, C_REG, C_ZCON, C_REG, C_NONE, 17, 4, 0, 0},
	{ABSTRPICKW, C_ZCON, C_REG, C_ZCON, C_REG, C_NONE, 17, 4, 0, 0},

	{ASYSCALL, C_NONE, C_NONE, C_NONE, C_NONE, C_NONE, 5, 4, 0, 0},
	{ASYSCALL, C_ANDCON, C_NONE, C_NONE, C_NONE, C_NONE, 5, 4, 0, 0},

	{ABEQ, C_REG, C_REG, C_NONE, C_BRAN, C_NONE, 6, 4, 0, 0},
	{ABEQ, C_REG, C_NONE, C_NONE, C_BRAN, C_NONE, 6, 4, 0, 0},
	{ABLEZ, C_REG, C_NONE, C_NONE, C_BRAN, C_NONE, 6, 4, 0, 0},
	{ABFPT, C_NONE, C_NONE, C_NONE, C_BRAN, C_NONE, 6, 4, 0, NOTUSETMP},

	{AJMP, C_NONE, C_NONE, C_NONE, C_BRAN, C_NONE, 11, 4, 0, 0}, // b
	{AJAL, C_NONE, C_NONE, C_NONE, C_BRAN, C_NONE, 11, 4, 0, 0}, // bl

	{AJMP, C_NONE, C_NONE, C_NONE, C_ZOREG, C_NONE, 18, 4, REGZERO, 0}, // jirl r0, rj, 0
	{AJAL, C_NONE, C_NONE, C_NONE, C_ZOREG, C_NONE, 18, 4, REGLINK, 0}, // jirl r1, rj, 0

	{AMOVF, C_SAUTO, C_NONE, C_NONE, C_FREG, C_NONE, 27, 4, REGSP, 0},
	{AMOVD, C_SAUTO, C_NONE, C_NONE, C_FREG, C_NONE, 27, 4, REGSP, 0},
	{AMOVF, C_SOREG, C_NONE, C_NONE, C_FREG, C_NONE, 27, 4, REGZERO, 0},
	{AMOVD, C_SOREG, C_NONE, C_NONE, C_FREG, C_NONE, 27, 4, REGZERO, 0},

	{AMOVF, C_LAUTO, C_NONE, C_NONE, C_FREG, C_NONE, 27, 12, REGSP, 0},
	{AMOVD, C_LAUTO, C_NONE, C_NONE, C_FREG, C_NONE, 27, 12, REGSP, 0},
	{AMOVF, C_LOREG, C_NONE, C_NONE, C_FREG, C_NONE, 27, 12, REGZERO, 0},
	{AMOVD, C_LOREG, C_NONE, C_NONE, C_FREG, C_NONE, 27, 12, REGZERO, 0},
	{AMOVF, C_ADDR, C_NONE, C_NONE, C_FREG, C_NONE, 51, 8, 0, 0},
	{AMOVD, C_ADDR, C_NONE, C_NONE, C_FREG, C_NONE, 51, 8, 0, 0},

	{AMOVF, C_FREG, C_NONE, C_NONE, C_SAUTO, C_NONE, 28, 4, REGSP, 0},
	{AMOVD, C_FREG, C_NONE, C_NONE, C_SAUTO, C_NONE, 28, 4, REGSP, 0},
	{AMOVF, C_FREG, C_NONE, C_NONE, C_SOREG, C_NONE, 28, 4, REGZERO, 0},
	{AMOVD, C_FREG, C_NONE, C_NONE, C_SOREG, C_NONE, 28, 4, REGZERO, 0},

	{AMOVF, C_FREG, C_NONE, C_NONE, C_LAUTO, C_NONE, 28, 12, REGSP, 0},
	{AMOVD, C_FREG, C_NONE, C_NONE, C_LAUTO, C_NONE, 28, 12, REGSP, 0},
	{AMOVF, C_FREG, C_NONE, C_NONE, C_LOREG, C_NONE, 28, 12, REGZERO, 0},
	{AMOVD, C_FREG, C_NONE, C_NONE, C_LOREG, C_NONE, 28, 12, REGZERO, 0},
	{AMOVF, C_FREG, C_NONE, C_NONE, C_ADDR, C_NONE, 50, 8, 0, 0},
	{AMOVD, C_FREG, C_NONE, C_NONE, C_ADDR, C_NONE, 50, 8, 0, 0},

	{AMOVW, C_REG, C_NONE, C_NONE, C_FREG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_FREG, C_NONE, 30, 4, 0, 0},
	{AMOVW, C_FREG, C_NONE, C_NONE, C_REG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_FREG, C_NONE, C_NONE, C_REG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_FCCREG, C_NONE, C_NONE, C_REG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_FCSRREG, C_NONE, C_NONE, C_REG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_FCCREG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_FCSRREG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_FREG, C_NONE, C_NONE, C_FCCREG, C_NONE, 30, 4, 0, 0},
	{AMOVV, C_FCCREG, C_NONE, C_NONE, C_FREG, C_NONE, 30, 4, 0, 0},

	{AMOVW, C_ADDCON, C_NONE, C_NONE, C_FREG, C_NONE, 34, 8, 0, 0},
	{AMOVW, C_ANDCON, C_NONE, C_NONE, C_FREG, C_NONE, 34, 8, 0, 0},

	{AMOVB, C_REG, C_NONE, C_NONE, C_TLS_IE, C_NONE, 56, 16, 0, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_TLS_IE, C_NONE, 56, 16, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_TLS_IE, C_NONE, 56, 16, 0, 0},
	{AMOVBU, C_REG, C_NONE, C_NONE, C_TLS_IE, C_NONE, 56, 16, 0, 0},
	{AMOVWU, C_REG, C_NONE, C_NONE, C_TLS_IE, C_NONE, 56, 16, 0, 0},

	{AMOVB, C_TLS_IE, C_NONE, C_NONE, C_REG, C_NONE, 57, 16, 0, 0},
	{AMOVW, C_TLS_IE, C_NONE, C_NONE, C_REG, C_NONE, 57, 16, 0, 0},
	{AMOVV, C_TLS_IE, C_NONE, C_NONE, C_REG, C_NONE, 57, 16, 0, 0},
	{AMOVBU, C_TLS_IE, C_NONE, C_NONE, C_REG, C_NONE, 57, 16, 0, 0},
	{AMOVWU, C_TLS_IE, C_NONE, C_NONE, C_REG, C_NONE, 57, 16, 0, 0},

	{AWORD, C_LCON, C_NONE, C_NONE, C_NONE, C_NONE, 38, 4, 0, 0},
	{AWORD, C_DCON, C_NONE, C_NONE, C_NONE, C_NONE, 61, 4, 0, 0},

	{AMOVV, C_GOTADDR, C_NONE, C_NONE, C_REG, C_NONE, 65, 8, 0, 0},

	{ATEQ, C_SCON, C_REG, C_NONE, C_REG, C_NONE, 15, 8, 0, 0},
	{ATEQ, C_SCON, C_NONE, C_NONE, C_REG, C_NONE, 15, 8, 0, 0},

	{ARDTIMELW, C_NONE, C_NONE, C_NONE, C_REG, C_REG, 62, 4, 0, 0},
	{AAMSWAPW, C_REG, C_NONE, C_NONE, C_ZOREG, C_REG, 66, 4, 0, 0},
	{ANOOP, C_NONE, C_NONE, C_NONE, C_NONE, C_NONE, 49, 4, 0, 0},

	/* store with extended register offset */
	{AMOVB, C_REG, C_NONE, C_NONE, C_ROFF, C_NONE, 20, 4, 0, 0},
	{AMOVW, C_REG, C_NONE, C_NONE, C_ROFF, C_NONE, 20, 4, 0, 0},
	{AMOVV, C_REG, C_NONE, C_NONE, C_ROFF, C_NONE, 20, 4, 0, 0},
	{AMOVF, C_FREG, C_NONE, C_NONE, C_ROFF, C_NONE, 20, 4, 0, 0},
	{AMOVD, C_FREG, C_NONE, C_NONE, C_ROFF, C_NONE, 20, 4, 0, 0},
	{AVMOVQ, C_VREG, C_NONE, C_NONE, C_ROFF, C_NONE, 20, 4, 0, 0},
	{AXVMOVQ, C_XREG, C_NONE, C_NONE, C_ROFF, C_NONE, 20, 4, 0, 0},

	/* load with extended register offset */
	{AMOVB, C_ROFF, C_NONE, C_NONE, C_REG, C_NONE, 21, 4, 0, 0},
	{AMOVBU, C_ROFF, C_NONE, C_NONE, C_REG, C_NONE, 21, 4, 0, 0},
	{AMOVW, C_ROFF, C_NONE, C_NONE, C_REG, C_NONE, 21, 4, 0, 0},
	{AMOVWU, C_ROFF, C_NONE, C_NONE, C_REG, C_NONE, 21, 4, 0, 0},
	{AMOVV, C_ROFF, C_NONE, C_NONE, C_REG, C_NONE, 21, 4, 0, 0},
	{AMOVF, C_ROFF, C_NONE, C_NONE, C_FREG, C_NONE, 21, 4, 0, 0},
	{AMOVD, C_ROFF, C_NONE, C_NONE, C_FREG, C_NONE, 21, 4, 0, 0},
	{AVMOVQ, C_ROFF, C_NONE, C_NONE, C_VREG, C_NONE, 21, 4, 0, 0},
	{AXVMOVQ, C_ROFF, C_NONE, C_NONE, C_XREG, C_NONE, 21, 4, 0, 0},

	{AVMOVQ, C_REG, C_NONE, C_NONE, C_ELEM, C_NONE, 39, 4, 0, 0},
	{AVMOVQ, C_ELEM, C_NONE, C_NONE, C_REG, C_NONE, 40, 4, 0, 0},
	{AXVMOVQ, C_REG, C_NONE, C_NONE, C_ELEM, C_NONE, 39, 4, 0, 0},
	{AXVMOVQ, C_ELEM, C_NONE, C_NONE, C_REG, C_NONE, 40, 4, 0, 0},

	{AXVMOVQ, C_XREG, C_NONE, C_NONE, C_ELEM, C_NONE, 43, 4, 0, 0},
	{AXVMOVQ, C_ELEM, C_NONE, C_NONE, C_XREG, C_NONE, 44, 4, 0, 0},

	{AVMOVQ, C_REG, C_NONE, C_NONE, C_ARNG, C_NONE, 41, 4, 0, 0},
	{AXVMOVQ, C_REG, C_NONE, C_NONE, C_ARNG, C_NONE, 41, 4, 0, 0},
	{AXVMOVQ, C_XREG, C_NONE, C_NONE, C_ARNG, C_NONE, 42, 4, 0, 0},

	{AVMOVQ, C_ELEM, C_NONE, C_NONE, C_ARNG, C_NONE, 45, 4, 0, 0},

	{obj.APCALIGN, C_SCON, C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0},
	{obj.APCDATA, C_LCON, C_NONE, C_NONE, C_LCON, C_NONE, 0, 0, 0, 0},
	{obj.APCDATA, C_DCON, C_NONE, C_NONE, C_DCON, C_NONE, 0, 0, 0, 0},
	{obj.AFUNCDATA, C_SCON, C_NONE, C_NONE, C_ADDR, C_NONE, 0, 0, 0, 0},
	{obj.ANOP, C_NONE, C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0},
	{obj.ANOP, C_LCON, C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0}, // nop variants, see #40689
	{obj.ANOP, C_DCON, C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0}, // nop variants, see #40689
	{obj.ANOP, C_REG, C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0},
	{obj.ANOP, C_FREG, C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0, 0},
	{obj.ADUFFZERO, C_NONE, C_NONE, C_NONE, C_BRAN, C_NONE, 11, 4, 0, 0}, // same as AJMP
	{obj.ADUFFCOPY, C_NONE, C_NONE, C_NONE, C_BRAN, C_NONE, 11, 4, 0, 0}, // same as AJMP

	{obj.AXXX, C_NONE, C_NONE, C_NONE, C_NONE, C_NONE, 0, 4, 0, 0},
}

var atomicInst = map[obj.As]uint32{
	AAMSWAPB:   0x070B8 << 15, // amswap.b
	AAMSWAPH:   0x070B9 << 15, // amswap.h
	AAMSWAPW:   0x070C0 << 15, // amswap.w
	AAMSWAPV:   0x070C1 << 15, // amswap.d
	AAMCASB:    0x070B0 << 15, // amcas.b
	AAMCASH:    0x070B1 << 15, // amcas.h
	AAMCASW:    0x070B2 << 15, // amcas.w
	AAMCASV:    0x070B3 << 15, // amcas.d
	AAMADDW:    0x070C2 << 15, // amadd.w
	AAMADDV:    0x070C3 << 15, // amadd.d
	AAMANDW:    0x070C4 << 15, // amand.w
	AAMANDV:    0x070C5 << 15, // amand.d
	AAMORW:     0x070C6 << 15, // amor.w
	AAMORV:     0x070C7 << 15, // amor.d
	AAMXORW:    0x070C8 << 15, // amxor.w
	AAMXORV:    0x070C9 << 15, // amxor.d
	AAMMAXW:    0x070CA << 15, // ammax.w
	AAMMAXV:    0x070CB << 15, // ammax.d
	AAMMINW:    0x070CC << 15, // ammin.w
	AAMMINV:    0x070CD << 15, // ammin.d
	AAMMAXWU:   0x070CE << 15, // ammax.wu
	AAMMAXVU:   0x070CF << 15, // ammax.du
	AAMMINWU:   0x070D0 << 15, // ammin.wu
	AAMMINVU:   0x070D1 << 15, // ammin.du
	AAMSWAPDBB: 0x070BC << 15, // amswap_db.b
	AAMSWAPDBH: 0x070BD << 15, // amswap_db.h
	AAMSWAPDBW: 0x070D2 << 15, // amswap_db.w
	AAMSWAPDBV: 0x070D3 << 15, // amswap_db.d
	AAMCASDBB:  0x070B4 << 15, // amcas_db.b
	AAMCASDBH:  0x070B5 << 15, // amcas_db.h
	AAMCASDBW:  0x070B6 << 15, // amcas_db.w
	AAMCASDBV:  0x070B7 << 15, // amcas_db.d
	AAMADDDBW:  0x070D4 << 15, // amadd_db.w
	AAMADDDBV:  0x070D5 << 15, // amadd_db.d
	AAMANDDBW:  0x070D6 << 15, // amand_db.w
	AAMANDDBV:  0x070D7 << 15, // amand_db.d
	AAMORDBW:   0x070D8 << 15, // amor_db.w
	AAMORDBV:   0x070D9 << 15, // amor_db.d
	AAMXORDBW:  0x070DA << 15, // amxor_db.w
	AAMXORDBV:  0x070DB << 15, // amxor_db.d
	AAMMAXDBW:  0x070DC << 15, // ammax_db.w
	AAMMAXDBV:  0x070DD << 15, // ammax_db.d
	AAMMINDBW:  0x070DE << 15, // ammin_db.w
	AAMMINDBV:  0x070DF << 15, // ammin_db.d
	AAMMAXDBWU: 0x070E0 << 15, // ammax_db.wu
	AAMMAXDBVU: 0x070E1 << 15, // ammax_db.du
	AAMMINDBWU: 0x070E2 << 15, // ammin_db.wu
	AAMMINDBVU: 0x070E3 << 15, // ammin_db.du
}

func IsAtomicInst(as obj.As) bool {
	_, ok := atomicInst[as]

	return ok
}

// pcAlignPadLength returns the number of bytes required to align pc to alignedValue,
// reporting an error if alignedValue is not a power of two or is out of range.
func pcAlignPadLength(ctxt *obj.Link, pc int64, alignedValue int64) int {
	if !((alignedValue&(alignedValue-1) == 0) && 8 <= alignedValue && alignedValue <= 2048) {
		ctxt.Diag("alignment value of an instruction must be a power of two and in the range [8, 2048], got %d\n", alignedValue)
	}
	return int(-pc & (alignedValue - 1))
}

var oprange [ALAST & obj.AMask][]Optab

var xcmp [C_NCLASS][C_NCLASS]bool

func span0(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	if ctxt.Retpoline {
		ctxt.Diag("-spectre=ret not supported on loong64")
		ctxt.Retpoline = false // don't keep printing
	}

	p := cursym.Func().Text
	if p == nil || p.Link == nil { // handle external functions and ELF section symbols
		return
	}

	c := ctxt0{ctxt: ctxt, newprog: newprog, cursym: cursym, autosize: int32(p.To.Offset + ctxt.Arch.FixedFrameSize)}

	if oprange[AOR&obj.AMask] == nil {
		c.ctxt.Diag("loong64 ops not initialized, call loong64.buildop first")
	}

	pc := int64(0)
	p.Pc = pc

	var m int
	var o *Optab
	for p = p.Link; p != nil; p = p.Link {
		p.Pc = pc
		o = c.oplook(p)
		m = int(o.size)
		if m == 0 {
			switch p.As {
			case obj.APCALIGN:
				alignedValue := p.From.Offset
				m = pcAlignPadLength(ctxt, pc, alignedValue)
				// Update the current text symbol alignment value.
				if int32(alignedValue) > cursym.Func().Align {
					cursym.Func().Align = int32(alignedValue)
				}
				break
			case obj.ANOP, obj.AFUNCDATA, obj.APCDATA:
				continue
			default:
				c.ctxt.Diag("zero-width instruction\n%v", p)
			}
		}

		pc += int64(m)
	}

	c.cursym.Size = pc

	// mark loop entry instructions for padding
	// loop entrances are defined as targets of backward branches
	for p = c.cursym.Func().Text.Link; p != nil; p = p.Link {
		if q := p.To.Target(); q != nil && q.Pc < p.Pc {
			q.Mark |= branchLoopHead
		}
	}

	// Run these passes until convergence.
	for {
		rescan := false
		pc = 0
		prev := c.cursym.Func().Text
		for p = prev.Link; p != nil; prev, p = p, p.Link {
			p.Pc = pc
			o = c.oplook(p)

			// Prepend a PCALIGN $loopAlign to each of the loop heads
			// that need padding, if not already done so (because this
			// pass may execute more than once).
			//
			// This needs to come before any pass that look at pc,
			// because pc will be adjusted if padding happens.
			if p.Mark&branchLoopHead != 0 && pc&(loopAlign-1) != 0 &&
				!(prev.As == obj.APCALIGN && prev.From.Offset >= loopAlign) {
				q := c.newprog()
				prev.Link = q
				q.Link = p
				q.Pc = pc
				q.As = obj.APCALIGN
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = loopAlign
				// Don't associate the synthesized PCALIGN with
				// the original source position, for deterministic
				// mapping between source and corresponding asm.
				// q.Pos = p.Pos

				// Manually make the PCALIGN come into effect,
				// since this loop iteration is for p.
				pc += int64(pcAlignPadLength(ctxt, pc, loopAlign))
				p.Pc = pc
				rescan = true
			}

			// very large conditional branches
			//
			// if any procedure is large enough to generate a large SBRA branch, then
			// generate extra passes putting branches around jmps to fix. this is rare.
			if o.type_ == 6 && p.To.Target() != nil {
				otxt := p.To.Target().Pc - pc

				// On loong64, the immediate value field of the conditional branch instructions
				// BFPT and BFPT is 21 bits, and the others are 16 bits. The jump target address
				// is to logically shift the immediate value in the instruction code to the left
				// by 2 bits and then sign extend.
				bound := int64(1 << (18 - 1))

				switch p.As {
				case ABFPT, ABFPF:
					bound = int64(1 << (23 - 1))
				}

				if otxt < -bound || otxt >= bound {
					q := c.newprog()
					q.Link = p.Link
					p.Link = q
					q.As = AJMP
					q.Pos = p.Pos
					q.To.Type = obj.TYPE_BRANCH
					q.To.SetTarget(p.To.Target())
					p.To.SetTarget(q)
					q = c.newprog()
					q.Link = p.Link
					p.Link = q
					q.As = AJMP
					q.Pos = p.Pos
					q.To.Type = obj.TYPE_BRANCH
					q.To.SetTarget(q.Link.Link)
					rescan = true
				}
			}

			m = int(o.size)
			if m == 0 {
				switch p.As {
				case obj.APCALIGN:
					alignedValue := p.From.Offset
					m = pcAlignPadLength(ctxt, pc, alignedValue)
					break
				case obj.ANOP, obj.AFUNCDATA, obj.APCDATA:
					continue
				default:
					c.ctxt.Diag("zero-width instruction\n%v", p)
				}
			}

			pc += int64(m)
		}

		c.cursym.Size = pc

		if !rescan {
			break
		}
	}

	pc += -pc & (FuncAlign - 1)
	c.cursym.Size = pc

	// lay out the code, emitting code and data relocations.

	c.cursym.Grow(c.cursym.Size)

	bp := c.cursym.P
	var i int32
	var out [5]uint32
	for p := c.cursym.Func().Text.Link; p != nil; p = p.Link {
		c.pc = p.Pc
		o = c.oplook(p)
		if int(o.size) > 4*len(out) {
			log.Fatalf("out array in span0 is too small, need at least %d for %v", o.size/4, p)
		}
		if p.As == obj.APCALIGN {
			alignedValue := p.From.Offset
			v := pcAlignPadLength(c.ctxt, p.Pc, alignedValue)
			for i = 0; i < int32(v/4); i++ {
				// emit ANOOP instruction by the padding size
				c.ctxt.Arch.ByteOrder.PutUint32(bp, OP_12IRR(c.opirr(AAND), 0, 0, 0))
				bp = bp[4:]
			}
			continue
		}
		c.asmout(p, o, out[:])
		for i = 0; i < int32(o.size/4); i++ {
			c.ctxt.Arch.ByteOrder.PutUint32(bp, out[i])
			bp = bp[4:]
		}
	}

	// Mark nonpreemptible instruction sequences.
	// We use REGTMP as a scratch register during call injection,
	// so instruction sequences that use REGTMP are unsafe to
	// preempt asynchronously.
	obj.MarkUnsafePoints(c.ctxt, c.cursym.Func().Text, c.newprog, c.isUnsafePoint, c.isRestartable)
}

// isUnsafePoint returns whether p is an unsafe point.
func (c *ctxt0) isUnsafePoint(p *obj.Prog) bool {
	// If p explicitly uses REGTMP, it's unsafe to preempt, because the
	// preemption sequence clobbers REGTMP.
	return p.From.Reg == REGTMP || p.To.Reg == REGTMP || p.Reg == REGTMP
}

// isRestartable returns whether p is a multi-instruction sequence that,
// if preempted, can be restarted.
func (c *ctxt0) isRestartable(p *obj.Prog) bool {
	if c.isUnsafePoint(p) {
		return false
	}
	// If p is a multi-instruction sequence with uses REGTMP inserted by
	// the assembler in order to materialize a large constant/offset, we
	// can restart p (at the start of the instruction sequence), recompute
	// the content of REGTMP, upon async preemption. Currently, all cases
	// of assembler-inserted REGTMP fall into this category.
	// If p doesn't use REGTMP, it can be simply preempted, so we don't
	// mark it.
	o := c.oplook(p)
	return o.size > 4 && o.flag&NOTUSETMP == 0
}

func isint32(v int64) bool {
	return int64(int32(v)) == v
}

func isuint32(v uint64) bool {
	return uint64(uint32(v)) == v
}

func (c *ctxt0) aclass(a *obj.Addr) int {
	switch a.Type {
	case obj.TYPE_NONE:
		return C_NONE

	case obj.TYPE_REG:
		return c.rclass(a.Reg)

	case obj.TYPE_MEM:
		switch a.Name {
		case obj.NAME_EXTERN,
			obj.NAME_STATIC:
			if a.Sym == nil {
				break
			}
			c.instoffset = a.Offset
			if a.Sym.Type == objabi.STLSBSS {
				if c.ctxt.Flag_shared {
					return C_TLS_IE
				} else {
					return C_TLS_LE
				}
			}
			return C_ADDR

		case obj.NAME_AUTO:
			if a.Reg == REGSP {
				// unset base register for better printing, since
				// a.Offset is still relative to pseudo-SP.
				a.Reg = obj.REG_NONE
			}
			c.instoffset = int64(c.autosize) + a.Offset
			if c.instoffset >= -BIG && c.instoffset < BIG {
				return C_SAUTO
			}
			return C_LAUTO

		case obj.NAME_PARAM:
			if a.Reg == REGSP {
				// unset base register for better printing, since
				// a.Offset is still relative to pseudo-FP.
				a.Reg = obj.REG_NONE
			}
			c.instoffset = int64(c.autosize) + a.Offset + c.ctxt.Arch.FixedFrameSize
			if c.instoffset >= -BIG && c.instoffset < BIG {
				return C_SAUTO
			}
			return C_LAUTO

		case obj.NAME_NONE:
			if a.Index != 0 {
				if a.Offset != 0 {
					return C_GOK
				}
				// register offset
				return C_ROFF
			}

			c.instoffset = a.Offset
			if c.instoffset == 0 {
				return C_ZOREG
			}
			if c.instoffset >= -BIG && c.instoffset < BIG {
				return C_SOREG
			}
			return C_LOREG

		case obj.NAME_GOTREF:
			return C_GOTADDR
		}

		return C_GOK

	case obj.TYPE_TEXTSIZE:
		return C_TEXTSIZE

	case obj.TYPE_CONST,
		obj.TYPE_ADDR:
		switch a.Name {
		case obj.NAME_NONE:
			c.instoffset = a.Offset
			if a.Reg != 0 {
				if -BIG <= c.instoffset && c.instoffset <= BIG {
					return C_SACON
				}
				if isint32(c.instoffset) {
					return C_LACON
				}
				return C_DACON
			}

		case obj.NAME_EXTERN,
			obj.NAME_STATIC:
			s := a.Sym
			if s == nil {
				return C_GOK
			}

			c.instoffset = a.Offset
			if s.Type == objabi.STLSBSS {
				c.ctxt.Diag("taking address of TLS variable is not supported")
			}
			return C_EXTADDR

		case obj.NAME_AUTO:
			if a.Reg == REGSP {
				// unset base register for better printing, since
				// a.Offset is still relative to pseudo-SP.
				a.Reg = obj.REG_NONE
			}
			c.instoffset = int64(c.autosize) + a.Offset
			if c.instoffset >= -BIG && c.instoffset < BIG {
				return C_SACON
			}
			return C_LACON

		case obj.NAME_PARAM:
			if a.Reg == REGSP {
				// unset base register for better printing, since
				// a.Offset is still relative to pseudo-FP.
				a.Reg = obj.REG_NONE
			}
			c.instoffset = int64(c.autosize) + a.Offset + c.ctxt.Arch.FixedFrameSize
			if c.instoffset >= -BIG && c.instoffset < BIG {
				return C_SACON
			}
			return C_LACON

		default:
			return C_GOK
		}

		if c.instoffset != int64(int32(c.instoffset)) {
			return C_DCON
		}

		if c.instoffset >= 0 {
			if c.instoffset == 0 {
				return C_ZCON
			}
			if c.instoffset <= 0x7ff {
				return C_SCON
			}
			if c.instoffset <= 0xfff {
				return C_ANDCON
			}
			if c.instoffset&0xfff == 0 && isuint32(uint64(c.instoffset)) { // && ((instoffset & (1<<31)) == 0)
				return C_UCON
			}
			if isint32(c.instoffset) || isuint32(uint64(c.instoffset)) {
				return C_LCON
			}
			return C_LCON
		}

		if c.instoffset >= -0x800 {
			return C_ADDCON
		}
		if c.instoffset&0xfff == 0 && isint32(c.instoffset) {
			return C_UCON
		}
		if isint32(c.instoffset) {
			return C_LCON
		}
		return C_LCON

	case obj.TYPE_BRANCH:
		return C_BRAN
	}

	return C_GOK
}

// In Loong64，there are 8 CFRs, denoted as fcc0-fcc7.
// There are 4 FCSRs, denoted as fcsr0-fcsr3.
func (c *ctxt0) rclass(r int16) int {
	switch {
	case REG_R0 <= r && r <= REG_R31:
		return C_REG
	case REG_F0 <= r && r <= REG_F31:
		return C_FREG
	case REG_FCC0 <= r && r <= REG_FCC7:
		return C_FCCREG
	case REG_FCSR0 <= r && r <= REG_FCSR3:
		return C_FCSRREG
	case REG_V0 <= r && r <= REG_V31:
		return C_VREG
	case REG_X0 <= r && r <= REG_X31:
		return C_XREG
	case r >= REG_ARNG && r < REG_ELEM:
		return C_ARNG
	case r >= REG_ELEM && r < REG_ELEM_END:
		return C_ELEM
	}

	return C_GOK
}

func oclass(a *obj.Addr) int {
	return int(a.Class) - 1
}

func prasm(p *obj.Prog) {
	fmt.Printf("%v\n", p)
}

func (c *ctxt0) oplook(p *obj.Prog) *Optab {
	if oprange[AOR&obj.AMask] == nil {
		c.ctxt.Diag("loong64 ops not initialized, call loong64.buildop first")
	}

	a1 := int(p.Optab)
	if a1 != 0 {
		return &optab[a1-1]
	}

	// first source operand
	a1 = int(p.From.Class)
	if a1 == 0 {
		a1 = c.aclass(&p.From) + 1
		p.From.Class = int8(a1)
	}
	a1--

	// first destination operand
	a4 := int(p.To.Class)
	if a4 == 0 {
		a4 = c.aclass(&p.To) + 1
		p.To.Class = int8(a4)
	}
	a4--

	// 2nd source operand
	a2 := C_NONE
	if p.Reg != 0 {
		a2 = c.rclass(p.Reg)
	}

	// 2nd destination operand
	a5 := C_NONE
	if p.RegTo2 != 0 {
		a5 = C_REG
	}

	// 3rd source operand
	a3 := C_NONE
	if len(p.RestArgs) > 0 {
		a3 = int(p.RestArgs[0].Class)
		if a3 == 0 {
			a3 = c.aclass(&p.RestArgs[0].Addr) + 1
			p.RestArgs[0].Class = int8(a3)
		}
		a3--
	}

	ops := oprange[p.As&obj.AMask]
	c1 := &xcmp[a1]
	c4 := &xcmp[a4]
	for i := range ops {
		op := &ops[i]
		if (int(op.reg) == a2) && int(op.from3) == a3 && c1[op.from1] && c4[op.to1] && (int(op.to2) == a5) {
			p.Optab = uint16(cap(optab) - cap(ops) + i + 1)
			return op
		}
	}

	c.ctxt.Diag("illegal combination %v %v %v %v %v %v", p.As, DRconv(a1), DRconv(a2), DRconv(a3), DRconv(a4), DRconv(a5))
	prasm(p)
	// Turn illegal instruction into an UNDEF, avoid crashing in asmout.
	return &Optab{obj.AUNDEF, C_NONE, C_NONE, C_NONE, C_NONE, C_NONE, 49, 4, 0, 0}
}

func cmp(a int, b int) bool {
	if a == b {
		return true
	}
	switch a {
	case C_DCON:
		if b == C_LCON {
			return true
		}
		fallthrough
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

	case C_LAUTO:
		if b == C_SAUTO {
			return true
		}

	case C_REG:
		if b == C_ZCON {
			return true
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

func ocmp(p1, p2 Optab) int {
	if p1.as != p2.as {
		return int(p1.as) - int(p2.as)
	}
	if p1.from1 != p2.from1 {
		return int(p1.from1) - int(p2.from1)
	}
	if p1.reg != p2.reg {
		return int(p1.reg) - int(p2.reg)
	}
	if p1.to1 != p2.to1 {
		return int(p1.to1) - int(p2.to1)
	}
	return 0
}

func opset(a, b0 obj.As) {
	oprange[a&obj.AMask] = oprange[b0]
}

func buildop(ctxt *obj.Link) {
	if ctxt.DiagFunc == nil {
		ctxt.DiagFunc = func(format string, args ...interface{}) {
			log.Printf(format, args...)
		}
	}

	if oprange[AOR&obj.AMask] != nil {
		// Already initialized; stop now.
		// This happens in the cmd/asm tests,
		// each of which re-initializes the arch.
		return
	}

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
	slices.SortFunc(optab[:n], ocmp)
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
			ctxt.DiagFlush()
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
			opset(AFCLASSF, r0)
			opset(AFCLASSD, r0)
			opset(AFLOGBF, r0)
			opset(AFLOGBD, r0)

		case AMOVVF:
			opset(AMOVVD, r0)
			opset(AMOVFV, r0)
			opset(AMOVDV, r0)
			opset(ATRUNCDV, r0)
			opset(ATRUNCFV, r0)
			opset(AFFINTFW, r0)
			opset(AFFINTFV, r0)
			opset(AFFINTDW, r0)
			opset(AFFINTDV, r0)
			opset(AFTINTWF, r0)
			opset(AFTINTWD, r0)
			opset(AFTINTVF, r0)
			opset(AFTINTVD, r0)
			opset(AFTINTRPWF, r0)
			opset(AFTINTRPWD, r0)
			opset(AFTINTRPVF, r0)
			opset(AFTINTRPVD, r0)
			opset(AFTINTRMWF, r0)
			opset(AFTINTRMWD, r0)
			opset(AFTINTRMVF, r0)
			opset(AFTINTRMVD, r0)
			opset(AFTINTRZWF, r0)
			opset(AFTINTRZWD, r0)
			opset(AFTINTRZVF, r0)
			opset(AFTINTRZVD, r0)
			opset(AFTINTRNEWF, r0)
			opset(AFTINTRNEWD, r0)
			opset(AFTINTRNEVF, r0)
			opset(AFTINTRNEVD, r0)

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
			opset(AFMINF, r0)
			opset(AFMIND, r0)
			opset(AFMAXF, r0)
			opset(AFMAXD, r0)
			opset(AFCOPYSGF, r0)
			opset(AFCOPYSGD, r0)
			opset(AFSCALEBF, r0)
			opset(AFSCALEBD, r0)

		case AFMADDF:
			opset(AFMADDD, r0)
			opset(AFMSUBF, r0)
			opset(AFMSUBD, r0)
			opset(AFNMADDF, r0)
			opset(AFNMADDD, r0)
			opset(AFNMSUBF, r0)
			opset(AFNMSUBD, r0)

		case AAND:
			opset(AOR, r0)
			opset(AXOR, r0)
			opset(AORN, r0)
			opset(AANDN, r0)

		case ABEQ:
			opset(ABNE, r0)
			opset(ABLT, r0)
			opset(ABGE, r0)
			opset(ABGEU, r0)
			opset(ABLTU, r0)

		case ABLEZ:
			opset(ABGEZ, r0)
			opset(ABLTZ, r0)
			opset(ABGTZ, r0)

		case AMOVB:
			opset(AMOVH, r0)

		case AMOVBU:
			opset(AMOVHU, r0)

		case AMUL:
			opset(AMULU, r0)
			opset(AMULH, r0)
			opset(AMULHU, r0)
			opset(AREM, r0)
			opset(AREMU, r0)
			opset(ADIV, r0)
			opset(ADIVU, r0)

		case AMULV:
			opset(AMULVU, r0)
			opset(AMULHV, r0)
			opset(AMULHVU, r0)
			opset(AREMV, r0)
			opset(AREMVU, r0)
			opset(ADIVV, r0)
			opset(ADIVVU, r0)

		case ASLL:
			opset(ASRL, r0)
			opset(ASRA, r0)
			opset(AROTR, r0)

		case ASLLV:
			opset(ASRAV, r0)
			opset(ASRLV, r0)
			opset(AROTRV, r0)

		case ABSTRPICKW:
			opset(ABSTRPICKV, r0)
			opset(ABSTRINSW, r0)
			opset(ABSTRINSV, r0)

		case ASUB:
			opset(ASUBU, r0)
			opset(ANOR, r0)

		case ASUBV:
			opset(ASUBVU, r0)

		case ASYSCALL:
			opset(ADBAR, r0)
			opset(ABREAK, r0)

		case ACMPEQF:
			opset(ACMPGTF, r0)
			opset(ACMPGTD, r0)
			opset(ACMPGEF, r0)
			opset(ACMPGED, r0)
			opset(ACMPEQD, r0)

		case ABFPT:
			opset(ABFPF, r0)

		case AMOVW,
			AMOVD,
			AMOVF,
			AMOVV,
			ARFE,
			AJAL,
			AJMP,
			AMOVWU,
			AVMOVQ,
			AXVMOVQ,
			ALL,
			ALLV,
			ASC,
			ASCV,
			ANEGW,
			ANEGV,
			AWORD,
			obj.ANOP,
			obj.ATEXT,
			obj.AFUNCDATA,
			obj.APCALIGN,
			obj.APCDATA,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			break

		case ARDTIMELW:
			opset(ARDTIMEHW, r0)
			opset(ARDTIMED, r0)

		case ACLOW:
			opset(ACLZW, r0)
			opset(ACTOW, r0)
			opset(ACTZW, r0)
			opset(ACLOV, r0)
			opset(ACLZV, r0)
			opset(ACTOV, r0)
			opset(ACTZV, r0)
			opset(AREVB2H, r0)
			opset(AREVB4H, r0)
			opset(AREVB2W, r0)
			opset(AREVBV, r0)
			opset(AREVH2W, r0)
			opset(AREVHV, r0)
			opset(ABITREV4B, r0)
			opset(ABITREV8B, r0)
			opset(ABITREVW, r0)
			opset(ABITREVV, r0)
			opset(AEXTWB, r0)
			opset(AEXTWH, r0)
			opset(ACPUCFG, r0)

		case ATEQ:
			opset(ATNE, r0)

		case AMASKEQZ:
			opset(AMASKNEZ, r0)
			opset(ACRCWBW, r0)
			opset(ACRCWHW, r0)
			opset(ACRCWWW, r0)
			opset(ACRCWVW, r0)
			opset(ACRCCWBW, r0)
			opset(ACRCCWHW, r0)
			opset(ACRCCWWW, r0)
			opset(ACRCCWVW, r0)

		case ANOOP:
			opset(obj.AUNDEF, r0)

		case AAMSWAPW:
			for i := range atomicInst {
				if i == AAMSWAPW {
					continue
				}
				opset(i, r0)
			}
		case AVSEQB:
			opset(AVSEQH, r0)
			opset(AVSEQW, r0)
			opset(AVSEQV, r0)

		case AXVSEQB:
			opset(AXVSEQH, r0)
			opset(AXVSEQW, r0)
			opset(AXVSEQV, r0)

		case AVPCNTB:
			opset(AVPCNTH, r0)
			opset(AVPCNTW, r0)
			opset(AVPCNTV, r0)

		case AXVPCNTB:
			opset(AXVPCNTH, r0)
			opset(AXVPCNTW, r0)
			opset(AXVPCNTV, r0)
		}
	}
}

func OP_RRRR(op uint32, r1 uint32, r2 uint32, r3 uint32, r4 uint32) uint32 {
	return op | (r1&0x1F)<<15 | (r2&0x1F)<<10 | (r3&0x1F)<<5 | (r4 & 0x1F)
}

// r1 -> rk
// r2 -> rj
// r3 -> rd
func OP_RRR(op uint32, r1 uint32, r2 uint32, r3 uint32) uint32 {
	return op | (r1&0x1F)<<10 | (r2&0x1F)<<5 | (r3&0x1F)<<0
}

// r2 -> rj
// r3 -> rd
func OP_RR(op uint32, r2 uint32, r3 uint32) uint32 {
	return op | (r2&0x1F)<<5 | (r3&0x1F)<<0
}

func OP_16IR_5I(op uint32, i uint32, r2 uint32) uint32 {
	return op | (i&0xFFFF)<<10 | (r2&0x1F)<<5 | ((i >> 16) & 0x1F)
}

func OP_16IRR(op uint32, i uint32, r2 uint32, r3 uint32) uint32 {
	return op | (i&0xFFFF)<<10 | (r2&0x1F)<<5 | (r3&0x1F)<<0
}

func OP_12IRR(op uint32, i uint32, r2 uint32, r3 uint32) uint32 {
	return op | (i&0xFFF)<<10 | (r2&0x1F)<<5 | (r3&0x1F)<<0
}

func OP_IR(op uint32, i uint32, r2 uint32) uint32 {
	return op | (i&0xFFFFF)<<5 | (r2&0x1F)<<0 // ui20, rd5
}

func OP_15I(op uint32, i uint32) uint32 {
	return op | (i&0x7FFF)<<0
}

// i1 -> msb
// r2 -> rj
// i3 -> lsb
// r4 -> rd
func OP_IRIR(op uint32, i1 uint32, r2 uint32, i3 uint32, r4 uint32) uint32 {
	return op | (i1 << 16) | (r2&0x1F)<<5 | (i3 << 10) | (r4&0x1F)<<0
}

// Encoding for the 'b' or 'bl' instruction.
func OP_B_BL(op uint32, i uint32) uint32 {
	return op | ((i & 0xFFFF) << 10) | ((i >> 16) & 0x3FF)
}

func (c *ctxt0) asmout(p *obj.Prog, o *Optab, out []uint32) {
	o1 := uint32(0)
	o2 := uint32(0)
	o3 := uint32(0)
	o4 := uint32(0)
	o5 := uint32(0)

	add := AADDU
	add = AADDVU

	switch o.type_ {
	default:
		c.ctxt.Diag("unknown type %d %v", o.type_)
		prasm(p)

	case 0: // pseudo ops
		break

	case 1: // mov r1,r2 ==> OR r1,r0,r2
		a := AOR
		if p.As == AMOVW {
			a = ASLL
		}
		o1 = OP_RRR(c.oprrr(a), uint32(REGZERO), uint32(p.From.Reg), uint32(p.To.Reg))

	case 2: // add/sub r1,[r2],r3
		r := int(p.Reg)
		if p.As == ANEGW || p.As == ANEGV {
			r = REGZERO
		}
		if r == 0 {
			r = int(p.To.Reg)
		}
		o1 = OP_RRR(c.oprrr(p.As), uint32(p.From.Reg), uint32(r), uint32(p.To.Reg))

	case 3: // mov $soreg, r ==> or/add $i,o,r
		v := c.regoff(&p.From)

		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		a := add
		if o.from1 == C_ANDCON {
			a = AOR
		}

		o1 = OP_12IRR(c.opirr(a), uint32(v), uint32(r), uint32(p.To.Reg))

	case 4: // add $scon,[r1],r2
		v := c.regoff(&p.From)

		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}

		o1 = OP_12IRR(c.opirr(p.As), uint32(v), uint32(r), uint32(p.To.Reg))

	case 5: // syscall
		v := c.regoff(&p.From)
		o1 = OP_15I(c.opi(p.As), uint32(v))

	case 6: // beq r1,[r2],sbra
		v := int32(0)
		if p.To.Target() != nil {
			v = int32(p.To.Target().Pc-p.Pc) >> 2
		}
		as, rd, rj, width := p.As, p.Reg, p.From.Reg, 16
		switch as {
		case ABGTZ, ABLEZ:
			rd, rj = rj, rd
		case ABFPT, ABFPF:
			width = 21
			// FCC0 is the implicit source operand, now that we
			// don't register-allocate from the FCC bank.
			rj = REG_FCC0
		case ABEQ, ABNE:
			if rd == 0 || rd == REGZERO || rj == REGZERO {
				// BEQZ/BNEZ can be encoded with 21-bit offsets.
				width = 21
				as = -as
				if rj == 0 || rj == REGZERO {
					rj = rd
				}
			}
		}
		switch width {
		case 21:
			if (v<<11)>>11 != v {
				c.ctxt.Diag("21 bit-width, short branch too far\n%v", p)
			}
			o1 = OP_16IR_5I(c.opirr(as), uint32(v), uint32(rj))
		case 16:
			if (v<<16)>>16 != v {
				c.ctxt.Diag("16 bit-width, short branch too far\n%v", p)
			}
			o1 = OP_16IRR(c.opirr(as), uint32(v), uint32(rj), uint32(rd))
		default:
			c.ctxt.Diag("unexpected branch encoding\n%v", p)
		}

	case 7: // mov r, soreg
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		v := c.regoff(&p.To)
		o1 = OP_12IRR(c.opirr(p.As), uint32(v), uint32(r), uint32(p.From.Reg))

	case 8: // mov soreg, r
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		v := c.regoff(&p.From)
		o1 = OP_12IRR(c.opirr(-p.As), uint32(v), uint32(r), uint32(p.To.Reg))

	case 9: // sll r1,[r2],r3
		o1 = OP_RR(c.oprr(p.As), uint32(p.From.Reg), uint32(p.To.Reg))

	case 10: // add $con,[r1],r2 ==> mov $con, t; add t,[r1],r2
		v := c.regoff(&p.From)
		a := AOR
		if v < 0 {
			a = AADDU
		}
		o1 = OP_12IRR(c.opirr(a), uint32(v), uint32(0), uint32(REGTMP))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o2 = OP_RRR(c.oprrr(p.As), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 11: // jmp lbra
		v := int32(0)
		if p.To.Target() != nil {
			v = int32(p.To.Target().Pc-p.Pc) >> 2
		}
		o1 = OP_B_BL(c.opirr(p.As), uint32(v))
		if p.To.Sym != nil {
			c.cursym.AddRel(c.ctxt, obj.Reloc{
				Type: objabi.R_CALLLOONG64,
				Off:  int32(c.pc),
				Siz:  4,
				Sym:  p.To.Sym,
				Add:  p.To.Offset,
			})
		}

	case 12: // movbs r,r
		switch p.As {
		case AMOVB:
			o1 = OP_RR(c.oprr(AEXTWB), uint32(p.From.Reg), uint32(p.To.Reg))
		case AMOVH:
			o1 = OP_RR(c.oprr(AEXTWH), uint32(p.From.Reg), uint32(p.To.Reg))
		case AMOVBU:
			o1 = OP_12IRR(c.opirr(AAND), uint32(0xff), uint32(p.From.Reg), uint32(p.To.Reg))
		case AMOVHU:
			o1 = OP_IRIR(c.opirir(ABSTRPICKV), 15, uint32(p.From.Reg), 0, uint32(p.To.Reg))
		case AMOVWU:
			o1 = OP_IRIR(c.opirir(ABSTRPICKV), 31, uint32(p.From.Reg), 0, uint32(p.To.Reg))
		default:
			c.ctxt.Diag("unexpected encoding\n%v", p)
		}

	case 15: // teq $c r,r
		v := c.regoff(&p.From)
		r := int(p.Reg)
		if r == 0 {
			r = REGZERO
		}
		/*
			teq c, r1, r2
			fallthrough
			==>
			bne r1, r2, 2
			break c
			fallthrough
		*/
		if p.As == ATEQ {
			o1 = OP_16IRR(c.opirr(ABNE), uint32(2), uint32(r), uint32(p.To.Reg))
		} else { // ATNE
			o1 = OP_16IRR(c.opirr(ABEQ), uint32(2), uint32(r), uint32(p.To.Reg))
		}
		o2 = OP_15I(c.opi(ABREAK), uint32(v))

	case 16: // sll $c,[r1],r2
		v := c.regoff(&p.From)
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}

		// instruction ending with V:6-digit immediate, others:5-digit immediate
		if v >= 32 && vshift(p.As) {
			o1 = OP_16IRR(c.opirr(p.As), uint32(v)&0x3f, uint32(r), uint32(p.To.Reg))
		} else {
			o1 = OP_16IRR(c.opirr(p.As), uint32(v)&0x1f, uint32(r), uint32(p.To.Reg))
		}

	case 17: // bstrpickw $msbw, r1, $lsbw, r2
		rd, rj := p.To.Reg, p.Reg
		if rj == obj.REG_NONE {
			rj = rd
		}
		msb, lsb := p.From.Offset, p.GetFrom3().Offset

		// check the range of msb and lsb
		var b uint32
		if p.As == ABSTRPICKW || p.As == ABSTRINSW {
			b = 32
		} else {
			b = 64
		}
		if lsb < 0 || uint32(lsb) >= b || msb < 0 || uint32(msb) >= b || uint32(lsb) > uint32(msb) {
			c.ctxt.Diag("illegal bit number\n%v", p)
		}

		o1 = OP_IRIR(c.opirir(p.As), uint32(msb), uint32(rj), uint32(lsb), uint32(rd))

	case 18: // jmp [r1],0(r2)
		r := int(p.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = OP_RRR(c.oprrr(p.As), uint32(0), uint32(p.To.Reg), uint32(r))
		if p.As == obj.ACALL {
			c.cursym.AddRel(c.ctxt, obj.Reloc{
				Type: objabi.R_CALLIND,
				Off:  int32(c.pc),
			})
		}

	case 19: // mov $lcon,r
		// NOTE: this case does not use REGTMP. If it ever does,
		// remove the NOTUSETMP flag in optab.
		v := c.regoff(&p.From)
		o1 = OP_IR(c.opir(ALU12IW), uint32(v>>12), uint32(p.To.Reg))
		o2 = OP_12IRR(c.opirr(AOR), uint32(v), uint32(p.To.Reg), uint32(p.To.Reg))

	case 20: // mov Rsrc, (Rbase)(Roff)
		o1 = OP_RRR(c.oprrr(p.As), uint32(p.To.Index), uint32(p.To.Reg), uint32(p.From.Reg))

	case 21: // mov (Rbase)(Roff), Rdst
		o1 = OP_RRR(c.oprrr(-p.As), uint32(p.From.Index), uint32(p.From.Reg), uint32(p.To.Reg))

	case 23: // add $lcon,r1,r2
		v := c.regoff(&p.From)
		o1 = OP_IR(c.opir(ALU12IW), uint32(v>>12), uint32(REGTMP))
		o2 = OP_12IRR(c.opirr(AOR), uint32(v), uint32(REGTMP), uint32(REGTMP))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o3 = OP_RRR(c.oprrr(p.As), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 24: // mov $ucon,r
		v := c.regoff(&p.From)
		o1 = OP_IR(c.opir(ALU12IW), uint32(v>>12), uint32(p.To.Reg))

	case 25: // add/and $ucon,[r1],r2
		v := c.regoff(&p.From)
		o1 = OP_IR(c.opir(ALU12IW), uint32(v>>12), uint32(REGTMP))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o2 = OP_RRR(c.oprrr(p.As), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 26: // mov $lsext/auto/oreg,r
		v := c.regoff(&p.From)
		o1 = OP_IR(c.opir(ALU12IW), uint32(v>>12), uint32(REGTMP))
		o2 = OP_12IRR(c.opirr(AOR), uint32(v), uint32(REGTMP), uint32(REGTMP))
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o3 = OP_RRR(c.oprrr(add), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 27: // mov [sl]ext/auto/oreg,fr
		v := c.regoff(&p.From)
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		switch o.size {
		case 12:
			o1 = OP_IR(c.opir(ALU12IW), uint32((v+1<<11)>>12), uint32(REGTMP))
			o2 = OP_RRR(c.oprrr(add), uint32(r), uint32(REGTMP), uint32(REGTMP))
			o3 = OP_12IRR(c.opirr(-p.As), uint32(v), uint32(REGTMP), uint32(p.To.Reg))

		case 4:
			o1 = OP_12IRR(c.opirr(-p.As), uint32(v), uint32(r), uint32(p.To.Reg))
		}

	case 28: // mov fr,[sl]ext/auto/oreg
		v := c.regoff(&p.To)
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		switch o.size {
		case 12:
			o1 = OP_IR(c.opir(ALU12IW), uint32((v+1<<11)>>12), uint32(REGTMP))
			o2 = OP_RRR(c.oprrr(add), uint32(r), uint32(REGTMP), uint32(REGTMP))
			o3 = OP_12IRR(c.opirr(p.As), uint32(v), uint32(REGTMP), uint32(p.From.Reg))

		case 4:
			o1 = OP_12IRR(c.opirr(p.As), uint32(v), uint32(r), uint32(p.From.Reg))
		}

	case 30: // mov gr/fr/fcc/fcsr, fr/fcc/fcsr/gr
		a := c.specialFpMovInst(p.As, oclass(&p.From), oclass(&p.To))
		o1 = OP_RR(a, uint32(p.From.Reg), uint32(p.To.Reg))

	case 34: // mov $con,fr
		v := c.regoff(&p.From)
		a := AADDU
		if o.from1 == C_ANDCON {
			a = AOR
		}
		a2 := c.specialFpMovInst(p.As, C_REG, oclass(&p.To))
		o1 = OP_12IRR(c.opirr(a), uint32(v), uint32(0), uint32(REGTMP))
		o2 = OP_RR(a2, uint32(REGTMP), uint32(p.To.Reg))

	case 35: // mov r,lext/auto/oreg
		v := c.regoff(&p.To)
		r := int(p.To.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = OP_IR(c.opir(ALU12IW), uint32((v+1<<11)>>12), uint32(REGTMP))
		o2 = OP_RRR(c.oprrr(add), uint32(r), uint32(REGTMP), uint32(REGTMP))
		o3 = OP_12IRR(c.opirr(p.As), uint32(v), uint32(REGTMP), uint32(p.From.Reg))

	case 36: // mov lext/auto/oreg,r
		v := c.regoff(&p.From)
		r := int(p.From.Reg)
		if r == 0 {
			r = int(o.param)
		}
		o1 = OP_IR(c.opir(ALU12IW), uint32((v+1<<11)>>12), uint32(REGTMP))
		o2 = OP_RRR(c.oprrr(add), uint32(r), uint32(REGTMP), uint32(REGTMP))
		o3 = OP_12IRR(c.opirr(-p.As), uint32(v), uint32(REGTMP), uint32(p.To.Reg))

	case 37: // fmadd r1, r2, [r3], r4
		r := int(p.To.Reg)
		if len(p.RestArgs) > 0 {
			r = int(p.GetFrom3().Reg)
		}
		o1 = OP_RRRR(c.oprrrr(p.As), uint32(p.From.Reg), uint32(p.Reg), uint32(r), uint32(p.To.Reg))

	case 38: // word
		o1 = uint32(c.regoff(&p.From))

	case 39: // vmov Rn, Vd.<T>[index]
		v, m := c.specialLsxMovInst(p.As, p.From.Reg, p.To.Reg)
		if v == 0 {
			c.ctxt.Diag("illegal arng type combination: %v\n", p)
		}

		Rj := uint32(p.From.Reg & EXT_REG_MASK)
		Vd := uint32(p.To.Reg & EXT_REG_MASK)
		index := uint32(p.To.Index)
		c.checkindex(p, index, m)
		o1 = v | (index << 10) | (Rj << 5) | Vd

	case 40: // vmov Vd.<T>[index], Rn
		v, m := c.specialLsxMovInst(p.As, p.From.Reg, p.To.Reg)
		if v == 0 {
			c.ctxt.Diag("illegal arng type combination: %v\n", p)
		}

		Vj := uint32(p.From.Reg & EXT_REG_MASK)
		Rd := uint32(p.To.Reg & EXT_REG_MASK)
		index := uint32(p.From.Index)
		c.checkindex(p, index, m)
		o1 = v | (index << 10) | (Vj << 5) | Rd

	case 41: // vmov Rn, Vd.<T>
		v, _ := c.specialLsxMovInst(p.As, p.From.Reg, p.To.Reg)
		if v == 0 {
			c.ctxt.Diag("illegal arng type combination: %v\n", p)
		}

		Rj := uint32(p.From.Reg & EXT_REG_MASK)
		Vd := uint32(p.To.Reg & EXT_REG_MASK)
		o1 = v | (Rj << 5) | Vd

	case 42: // vmov  xj, xd.<T>
		v, _ := c.specialLsxMovInst(p.As, p.From.Reg, p.To.Reg)
		if v == 0 {
			c.ctxt.Diag("illegal arng type combination: %v\n", p)
		}

		Xj := uint32(p.From.Reg & EXT_REG_MASK)
		Xd := uint32(p.To.Reg & EXT_REG_MASK)
		o1 = v | (Xj << 5) | Xd

	case 43: // vmov  xj, xd.<T>[index]
		v, m := c.specialLsxMovInst(p.As, p.From.Reg, p.To.Reg)
		if v == 0 {
			c.ctxt.Diag("illegal arng type combination: %v\n", p)
		}

		Xj := uint32(p.From.Reg & EXT_REG_MASK)
		Xd := uint32(p.To.Reg & EXT_REG_MASK)
		index := uint32(p.To.Index)
		c.checkindex(p, index, m)
		o1 = v | (index << 10) | (Xj << 5) | Xd

	case 44: // vmov  xj.<T>[index], xd
		v, m := c.specialLsxMovInst(p.As, p.From.Reg, p.To.Reg)
		if v == 0 {
			c.ctxt.Diag("illegal arng type combination: %v\n", p)
		}

		Xj := uint32(p.From.Reg & EXT_REG_MASK)
		Xd := uint32(p.To.Reg & EXT_REG_MASK)
		index := uint32(p.From.Index)
		c.checkindex(p, index, m)
		o1 = v | (index << 10) | (Xj << 5) | Xd

	case 45: // vmov  vj.<T>[index], vd.<T>
		v, m := c.specialLsxMovInst(p.As, p.From.Reg, p.To.Reg)
		if v == 0 {
			c.ctxt.Diag("illegal arng type combination: %v\n", p)
		}

		vj := uint32(p.From.Reg & EXT_REG_MASK)
		vd := uint32(p.To.Reg & EXT_REG_MASK)
		index := uint32(p.From.Index)
		c.checkindex(p, index, m)
		o1 = v | (index << 10) | (vj << 5) | vd

	case 49:
		if p.As == ANOOP {
			// andi r0, r0, 0
			o1 = OP_12IRR(c.opirr(AAND), 0, 0, 0)
		} else {
			// undef
			o1 = OP_15I(c.opi(ABREAK), 0)
		}

	// relocation operations
	case 50: // mov r,addr ==> pcalau12i + sw
		o1 = OP_IR(c.opir(APCALAU12I), uint32(0), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_ADDR_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.To.Sym,
			Add:  p.To.Offset,
		})
		o2 = OP_12IRR(c.opirr(p.As), uint32(0), uint32(REGTMP), uint32(p.From.Reg))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_ADDR_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.To.Sym,
			Add:  p.To.Offset,
		})

	case 51: // mov addr,r ==> pcalau12i + lw
		o1 = OP_IR(c.opir(APCALAU12I), uint32(0), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_ADDR_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.From.Sym,
			Add:  p.From.Offset,
		})
		o2 = OP_12IRR(c.opirr(-p.As), uint32(0), uint32(REGTMP), uint32(p.To.Reg))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_ADDR_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.From.Sym,
			Add:  p.From.Offset,
		})

	case 52: // mov $ext, r
		// NOTE: this case does not use REGTMP. If it ever does,
		// remove the NOTUSETMP flag in optab.
		o1 = OP_IR(c.opir(APCALAU12I), uint32(0), uint32(p.To.Reg))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_ADDR_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.From.Sym,
			Add:  p.From.Offset,
		})
		o2 = OP_12IRR(c.opirr(add), uint32(0), uint32(p.To.Reg), uint32(p.To.Reg))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_ADDR_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.From.Sym,
			Add:  p.From.Offset,
		})

	case 53: // mov r, tlsvar ==>  lu12i.w + ori + add r2, regtmp + sw o(regtmp)
		// NOTE: this case does not use REGTMP. If it ever does,
		// remove the NOTUSETMP flag in optab.
		o1 = OP_IR(c.opir(ALU12IW), uint32(0), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_LE_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.To.Sym,
			Add:  p.To.Offset,
		})
		o2 = OP_12IRR(c.opirr(AOR), uint32(0), uint32(REGTMP), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_LE_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.To.Sym,
			Add:  p.To.Offset,
		})
		o3 = OP_RRR(c.oprrr(AADDV), uint32(REG_R2), uint32(REGTMP), uint32(REGTMP))
		o4 = OP_12IRR(c.opirr(p.As), uint32(0), uint32(REGTMP), uint32(p.From.Reg))

	case 54: // lu12i.w + ori + add r2, regtmp + lw o(regtmp)
		// NOTE: this case does not use REGTMP. If it ever does,
		// remove the NOTUSETMP flag in optab.
		o1 = OP_IR(c.opir(ALU12IW), uint32(0), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_LE_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.From.Sym,
			Add:  p.From.Offset,
		})
		o2 = OP_12IRR(c.opirr(AOR), uint32(0), uint32(REGTMP), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_LE_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.From.Sym,
			Add:  p.From.Offset,
		})
		o3 = OP_RRR(c.oprrr(AADDV), uint32(REG_R2), uint32(REGTMP), uint32(REGTMP))
		o4 = OP_12IRR(c.opirr(-p.As), uint32(0), uint32(REGTMP), uint32(p.To.Reg))

	case 56: // mov r, tlsvar IE model ==> (pcalau12i + ld.d)tlsvar@got + add.d + st.d
		o1 = OP_IR(c.opir(APCALAU12I), uint32(0), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_IE_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.To.Sym,
		})
		o2 = OP_12IRR(c.opirr(-p.As), uint32(0), uint32(REGTMP), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_IE_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.To.Sym,
		})
		o3 = OP_RRR(c.oprrr(AADDVU), uint32(REGTMP), uint32(REG_R2), uint32(REGTMP))
		o4 = OP_12IRR(c.opirr(p.As), uint32(0), uint32(REGTMP), uint32(p.From.Reg))

	case 57: // mov tlsvar, r IE model ==> (pcalau12i + ld.d)tlsvar@got + add.d + ld.d
		o1 = OP_IR(c.opir(APCALAU12I), uint32(0), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_IE_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.From.Sym,
		})
		o2 = OP_12IRR(c.opirr(-p.As), uint32(0), uint32(REGTMP), uint32(REGTMP))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_TLS_IE_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.From.Sym,
		})
		o3 = OP_RRR(c.oprrr(AADDVU), uint32(REGTMP), uint32(REG_R2), uint32(REGTMP))
		o4 = OP_12IRR(c.opirr(-p.As), uint32(0), uint32(REGTMP), uint32(p.To.Reg))

	case 59: // mov $dcon,r
		// NOTE: this case does not use REGTMP. If it ever does,
		// remove the NOTUSETMP flag in optab.
		v := c.vregoff(&p.From)
		o1 = OP_IR(c.opir(ALU12IW), uint32(v>>12), uint32(p.To.Reg))
		o2 = OP_12IRR(c.opirr(AOR), uint32(v), uint32(p.To.Reg), uint32(p.To.Reg))
		o3 = OP_IR(c.opir(ALU32ID), uint32(v>>32), uint32(p.To.Reg))
		o4 = OP_12IRR(c.opirr(ALU52ID), uint32(v>>52), uint32(p.To.Reg), uint32(p.To.Reg))

	case 60: // add $dcon,r1,r2
		v := c.vregoff(&p.From)
		o1 = OP_IR(c.opir(ALU12IW), uint32(v>>12), uint32(REGTMP))
		o2 = OP_12IRR(c.opirr(AOR), uint32(v), uint32(REGTMP), uint32(REGTMP))
		o3 = OP_IR(c.opir(ALU32ID), uint32(v>>32), uint32(REGTMP))
		o4 = OP_12IRR(c.opirr(ALU52ID), uint32(v>>52), uint32(REGTMP), uint32(REGTMP))
		r := int(p.Reg)
		if r == 0 {
			r = int(p.To.Reg)
		}
		o5 = OP_RRR(c.oprrr(p.As), uint32(REGTMP), uint32(r), uint32(p.To.Reg))

	case 61: // word C_DCON
		o1 = uint32(c.vregoff(&p.From))
		o2 = uint32(c.vregoff(&p.From) >> 32)

	case 62: // rdtimex rd, rj
		o1 = OP_RR(c.oprr(p.As), uint32(p.To.Reg), uint32(p.RegTo2))

	case 65: // mov sym@GOT, r ==> pcalau12i + ld.d
		o1 = OP_IR(c.opir(APCALAU12I), uint32(0), uint32(p.To.Reg))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_GOT_HI,
			Off:  int32(c.pc),
			Siz:  4,
			Sym:  p.From.Sym,
		})
		o2 = OP_12IRR(c.opirr(-p.As), uint32(0), uint32(p.To.Reg), uint32(p.To.Reg))
		c.cursym.AddRel(c.ctxt, obj.Reloc{
			Type: objabi.R_LOONG64_GOT_LO,
			Off:  int32(c.pc + 4),
			Siz:  4,
			Sym:  p.From.Sym,
		})

	case 66: // am* From, To, RegTo2 ==> am* RegTo2, From, To
		rk := p.From.Reg
		rj := p.To.Reg
		rd := p.RegTo2

		// See section 2.2.7.1 of https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html
		// for the register usage constraints.
		if rd == rj || rd == rk {
			c.ctxt.Diag("illegal register combination: %v\n", p)
		}
		o1 = OP_RRR(atomicInst[p.As], uint32(rk), uint32(rj), uint32(rd))
	}

	out[0] = o1
	out[1] = o2
	out[2] = o3
	out[3] = o4
	out[4] = o5
}

// checkindex checks if index >= 0 && index <= maxindex
func (c *ctxt0) checkindex(p *obj.Prog, index uint32, mask uint32) {
	if (index & ^mask) != 0 {
		c.ctxt.Diag("register element index out of range 0 to %d: %v", mask, p)
	}
}

func (c *ctxt0) vregoff(a *obj.Addr) int64 {
	c.instoffset = 0
	c.aclass(a)
	return c.instoffset
}

func (c *ctxt0) regoff(a *obj.Addr) int32 {
	return int32(c.vregoff(a))
}

func (c *ctxt0) oprrrr(a obj.As) uint32 {
	switch a {
	case AFMADDF:
		return 0x81 << 20 // fmadd.s
	case AFMADDD:
		return 0x82 << 20 // fmadd.d
	case AFMSUBF:
		return 0x85 << 20 // fmsub.s
	case AFMSUBD:
		return 0x86 << 20 // fmsub.d
	case AFNMADDF:
		return 0x89 << 20 // fnmadd.f
	case AFNMADDD:
		return 0x8a << 20 // fnmadd.d
	case AFNMSUBF:
		return 0x8d << 20 // fnmsub.s
	case AFNMSUBD:
		return 0x8e << 20 // fnmsub.d
	}

	c.ctxt.Diag("bad rrrr opcode %v", a)
	return 0
}

func (c *ctxt0) oprrr(a obj.As) uint32 {
	switch a {
	case AADD:
		return 0x20 << 15
	case AADDU:
		return 0x20 << 15
	case ASGT:
		return 0x24 << 15 // SLT
	case ASGTU:
		return 0x25 << 15 // SLTU
	case AMASKEQZ:
		return 0x26 << 15
	case AMASKNEZ:
		return 0x27 << 15
	case AAND:
		return 0x29 << 15
	case AOR:
		return 0x2a << 15
	case AXOR:
		return 0x2b << 15
	case AORN:
		return 0x2c << 15 // orn
	case AANDN:
		return 0x2d << 15 // andn
	case ASUB:
		return 0x22 << 15
	case ASUBU, ANEGW:
		return 0x22 << 15
	case ANOR:
		return 0x28 << 15
	case ASLL:
		return 0x2e << 15
	case ASRL:
		return 0x2f << 15
	case ASRA:
		return 0x30 << 15
	case AROTR:
		return 0x36 << 15
	case ASLLV:
		return 0x31 << 15
	case ASRLV:
		return 0x32 << 15
	case ASRAV:
		return 0x33 << 15
	case AROTRV:
		return 0x37 << 15
	case AADDV:
		return 0x21 << 15
	case AADDVU:
		return 0x21 << 15
	case ASUBV:
		return 0x23 << 15
	case ASUBVU, ANEGV:
		return 0x23 << 15

	case AMUL:
		return 0x38 << 15 // mul.w
	case AMULU:
		return 0x38 << 15 // mul.w
	case AMULH:
		return 0x39 << 15 // mulh.w
	case AMULHU:
		return 0x3a << 15 // mulhu.w
	case AMULV:
		return 0x3b << 15 // mul.d
	case AMULVU:
		return 0x3b << 15 // mul.d
	case AMULHV:
		return 0x3c << 15 // mulh.d
	case AMULHVU:
		return 0x3d << 15 // mulhu.d
	case ADIV:
		return 0x40 << 15 // div.w
	case ADIVU:
		return 0x42 << 15 // div.wu
	case ADIVV:
		return 0x44 << 15 // div.d
	case ADIVVU:
		return 0x46 << 15 // div.du
	case AREM:
		return 0x41 << 15 // mod.w
	case AREMU:
		return 0x43 << 15 // mod.wu
	case AREMV:
		return 0x45 << 15 // mod.d
	case AREMVU:
		return 0x47 << 15 // mod.du
	case ACRCWBW:
		return 0x48 << 15 // crc.w.b.w
	case ACRCWHW:
		return 0x49 << 15 // crc.w.h.w
	case ACRCWWW:
		return 0x4a << 15 // crc.w.w.w
	case ACRCWVW:
		return 0x4b << 15 // crc.w.d.w
	case ACRCCWBW:
		return 0x4c << 15 // crcc.w.b.w
	case ACRCCWHW:
		return 0x4d << 15 // crcc.w.h.w
	case ACRCCWWW:
		return 0x4e << 15 // crcc.w.w.w
	case ACRCCWVW:
		return 0x4f << 15 // crcc.w.d.w
	case AJMP:
		return 0x13 << 26 // jirl r0, rj, 0
	case AJAL:
		return (0x13 << 26) | 1 // jirl r1, rj, 0

	case ADIVF:
		return 0x20d << 15
	case ADIVD:
		return 0x20e << 15
	case AMULF:
		return 0x209 << 15
	case AMULD:
		return 0x20a << 15
	case ASUBF:
		return 0x205 << 15
	case ASUBD:
		return 0x206 << 15
	case AADDF:
		return 0x201 << 15
	case AADDD:
		return 0x202 << 15
	case ACMPEQF:
		return 0x0c1<<20 | 0x4<<15 // FCMP.CEQ.S
	case ACMPEQD:
		return 0x0c2<<20 | 0x4<<15 // FCMP.CEQ.D
	case ACMPGED:
		return 0x0c2<<20 | 0x7<<15 // FCMP.SLE.D
	case ACMPGEF:
		return 0x0c1<<20 | 0x7<<15 // FCMP.SLE.S
	case ACMPGTD:
		return 0x0c2<<20 | 0x3<<15 // FCMP.SLT.D
	case ACMPGTF:
		return 0x0c1<<20 | 0x3<<15 // FCMP.SLT.S
	case AFMINF:
		return 0x215 << 15 // fmin.s
	case AFMIND:
		return 0x216 << 15 // fmin.d
	case AFMAXF:
		return 0x211 << 15 // fmax.s
	case AFMAXD:
		return 0x212 << 15 // fmax.d
	case AFSCALEBF:
		return 0x221 << 15 // fscaleb.s
	case AFSCALEBD:
		return 0x222 << 15 // fscaleb.d
	case AFCOPYSGF:
		return 0x225 << 15 // fcopysign.s
	case AFCOPYSGD:
		return 0x226 << 15 // fcopysign.d
	case -AMOVB:
		return 0x07000 << 15 // ldx.b
	case -AMOVH:
		return 0x07008 << 15 // ldx.h
	case -AMOVW:
		return 0x07010 << 15 // ldx.w
	case -AMOVV:
		return 0x07018 << 15 // ldx.d
	case -AMOVBU:
		return 0x07040 << 15 // ldx.bu
	case -AMOVHU:
		return 0x07048 << 15 // ldx.hu
	case -AMOVWU:
		return 0x07050 << 15 // ldx.wu
	case AMOVB:
		return 0x07020 << 15 // stx.b
	case AMOVH:
		return 0x07028 << 15 // stx.h
	case AMOVW:
		return 0x07030 << 15 // stx.w
	case AMOVV:
		return 0x07038 << 15 // stx.d
	case -AMOVF:
		return 0x07060 << 15 // fldx.s
	case -AMOVD:
		return 0x07068 << 15 // fldx.d
	case AMOVF:
		return 0x07070 << 15 // fstx.s
	case AMOVD:
		return 0x07078 << 15 // fstx.d
	case -AVMOVQ:
		return 0x07080 << 15 // vldx
	case -AXVMOVQ:
		return 0x07090 << 15 // xvldx
	case AVMOVQ:
		return 0x07088 << 15 // vstx
	case AXVMOVQ:
		return 0x07098 << 15 // xvstx
	case AVSEQB:
		return 0x0e000 << 15 // vseq.b
	case AXVSEQB:
		return 0x0e800 << 15 // xvseq.b
	case AVSEQH:
		return 0x0e001 << 15 // vseq.h
	case AXVSEQH:
		return 0x0e801 << 15 // xvseq.h
	case AVSEQW:
		return 0x0e002 << 15 // vseq.w
	case AXVSEQW:
		return 0x0e802 << 15 // xvseq.w
	case AVSEQV:
		return 0x0e003 << 15 // vseq.d
	case AXVSEQV:
		return 0x0e803 << 15 // xvseq.d
	}

	if a < 0 {
		c.ctxt.Diag("bad rrr opcode -%v", -a)
	} else {
		c.ctxt.Diag("bad rrr opcode %v", a)
	}
	return 0
}

func (c *ctxt0) oprr(a obj.As) uint32 {
	switch a {
	case ACLOW:
		return 0x4 << 10 // clo.w
	case ACLZW:
		return 0x5 << 10 // clz.w
	case ACTOW:
		return 0x6 << 10 // cto.w
	case ACTZW:
		return 0x7 << 10 // ctz.w
	case ACLOV:
		return 0x8 << 10 // clo.d
	case ACLZV:
		return 0x9 << 10 // clz.d
	case ACTOV:
		return 0xa << 10 // cto.d
	case ACTZV:
		return 0xb << 10 // ctz.d
	case AREVB2H:
		return 0xc << 10 // revb.2h
	case AREVB4H:
		return 0xd << 10 // revb.4h
	case AREVB2W:
		return 0xe << 10 // revb.2w
	case AREVBV:
		return 0xf << 10 // revb.d
	case AREVH2W:
		return 0x10 << 10 // revh.2w
	case AREVHV:
		return 0x11 << 10 // revh.d
	case ABITREV4B:
		return 0x12 << 10 // bitrev.4b
	case ABITREV8B:
		return 0x13 << 10 // bitrev.8b
	case ABITREVW:
		return 0x14 << 10 // bitrev.w
	case ABITREVV:
		return 0x15 << 10 // bitrev.d
	case AEXTWH:
		return 0x16 << 10 // ext.w.h
	case AEXTWB:
		return 0x17 << 10 // ext.w.h
	case ACPUCFG:
		return 0x1b << 10
	case ARDTIMELW:
		return 0x18 << 10
	case ARDTIMEHW:
		return 0x19 << 10
	case ARDTIMED:
		return 0x1a << 10
	case ATRUNCFV:
		return 0x46a9 << 10
	case ATRUNCDV:
		return 0x46aa << 10
	case ATRUNCFW:
		return 0x46a1 << 10
	case ATRUNCDW:
		return 0x46a2 << 10
	case AMOVFV:
		return 0x46c9 << 10
	case AMOVDV:
		return 0x46ca << 10
	case AMOVVF:
		return 0x4746 << 10
	case AMOVVD:
		return 0x474a << 10
	case AMOVFW:
		return 0x46c1 << 10
	case AMOVDW:
		return 0x46c2 << 10
	case AMOVWF:
		return 0x4744 << 10
	case AMOVDF:
		return 0x4646 << 10
	case AMOVWD:
		return 0x4748 << 10
	case AMOVFD:
		return 0x4649 << 10
	case AABSF:
		return 0x4501 << 10
	case AABSD:
		return 0x4502 << 10
	case AMOVF:
		return 0x4525 << 10
	case AMOVD:
		return 0x4526 << 10
	case ANEGF:
		return 0x4505 << 10
	case ANEGD:
		return 0x4506 << 10
	case ASQRTF:
		return 0x4511 << 10
	case ASQRTD:
		return 0x4512 << 10
	case AFLOGBF:
		return 0x4509 << 10 // flogb.s
	case AFLOGBD:
		return 0x450a << 10 // flogb.d
	case AFCLASSF:
		return 0x450d << 10 // fclass.s
	case AFCLASSD:
		return 0x450e << 10 // fclass.d
	case AFFINTFW:
		return 0x4744 << 10 // ffint.s.w
	case AFFINTFV:
		return 0x4746 << 10 // ffint.s.l
	case AFFINTDW:
		return 0x4748 << 10 // ffint.d.w
	case AFFINTDV:
		return 0x474a << 10 // ffint.d.l
	case AFTINTWF:
		return 0x46c1 << 10 // ftint.w.s
	case AFTINTWD:
		return 0x46c2 << 10 // ftint.w.d
	case AFTINTVF:
		return 0x46c9 << 10 // ftint.l.s
	case AFTINTVD:
		return 0x46ca << 10 // ftint.l.d
	case AFTINTRMWF:
		return 0x4681 << 10 // ftintrm.w.s
	case AFTINTRMWD:
		return 0x4682 << 10 // ftintrm.w.d
	case AFTINTRMVF:
		return 0x4689 << 10 // ftintrm.l.s
	case AFTINTRMVD:
		return 0x468a << 10 // ftintrm.l.d
	case AFTINTRPWF:
		return 0x4691 << 10 // ftintrp.w.s
	case AFTINTRPWD:
		return 0x4692 << 10 // ftintrp.w.d
	case AFTINTRPVF:
		return 0x4699 << 10 // ftintrp.l.s
	case AFTINTRPVD:
		return 0x469a << 10 // ftintrp.l.d
	case AFTINTRZWF:
		return 0x46a1 << 10 // ftintrz.w.s
	case AFTINTRZWD:
		return 0x46a2 << 10 // ftintrz.w.d
	case AFTINTRZVF:
		return 0x46a9 << 10 // ftintrz.l.s
	case AFTINTRZVD:
		return 0x46aa << 10 // ftintrz.l.d
	case AFTINTRNEWF:
		return 0x46b1 << 10 // ftintrne.w.s
	case AFTINTRNEWD:
		return 0x46b2 << 10 // ftintrne.w.d
	case AFTINTRNEVF:
		return 0x46b9 << 10 // ftintrne.l.s
	case AFTINTRNEVD:
		return 0x46ba << 10 // ftintrne.l.d
	case AVPCNTB:
		return 0x1ca708 << 10 // vpcnt.b
	case AVPCNTH:
		return 0x1ca709 << 10 // vpcnt.h
	case AVPCNTW:
		return 0x1ca70a << 10 // vpcnt.w
	case AVPCNTV:
		return 0x1ca70b << 10 // vpcnt.v
	case AXVPCNTB:
		return 0x1da708 << 10 // xvpcnt.b
	case AXVPCNTH:
		return 0x1da709 << 10 // xvpcnt.h
	case AXVPCNTW:
		return 0x1da70a << 10 // xvpcnt.w
	case AXVPCNTV:
		return 0x1da70b << 10 // xvpcnt.v
	}

	c.ctxt.Diag("bad rr opcode %v", a)
	return 0
}

func (c *ctxt0) opi(a obj.As) uint32 {
	switch a {
	case ASYSCALL:
		return 0x56 << 15
	case ABREAK:
		return 0x54 << 15
	case ADBAR:
		return 0x70e4 << 15
	}

	c.ctxt.Diag("bad ic opcode %v", a)

	return 0
}

func (c *ctxt0) opir(a obj.As) uint32 {
	switch a {
	case ALU12IW:
		return 0x0a << 25
	case ALU32ID:
		return 0x0b << 25
	case APCALAU12I:
		return 0x0d << 25
	case APCADDU12I:
		return 0x0e << 25
	}
	return 0
}

func (c *ctxt0) opirr(a obj.As) uint32 {
	switch a {
	case AADD, AADDU:
		return 0x00a << 22
	case ASGT:
		return 0x008 << 22
	case ASGTU:
		return 0x009 << 22
	case AAND:
		return 0x00d << 22
	case AOR:
		return 0x00e << 22
	case ALU52ID:
		return 0x00c << 22
	case AXOR:
		return 0x00f << 22
	case ASLL:
		return 0x00081 << 15
	case ASRL:
		return 0x00089 << 15
	case ASRA:
		return 0x00091 << 15
	case AROTR:
		return 0x00099 << 15
	case AADDV:
		return 0x00b << 22
	case AADDVU:
		return 0x00b << 22

	case AJMP:
		return 0x14 << 26
	case AJAL,
		obj.ADUFFZERO,
		obj.ADUFFCOPY:
		return 0x15 << 26

	case AJIRL:
		return 0x13 << 26
	case ABLTU:
		return 0x1a << 26
	case ABLT, ABLTZ, ABGTZ:
		return 0x18 << 26
	case ABGEU:
		return 0x1b << 26
	case ABGE, ABGEZ, ABLEZ:
		return 0x19 << 26
	case -ABEQ: // beqz
		return 0x10 << 26
	case -ABNE: // bnez
		return 0x11 << 26
	case ABEQ:
		return 0x16 << 26
	case ABNE:
		return 0x17 << 26
	case ABFPT:
		return 0x12<<26 | 0x1<<8
	case ABFPF:
		return 0x12<<26 | 0x0<<8

	case AMOVB,
		AMOVBU:
		return 0x0a4 << 22
	case AMOVH,
		AMOVHU:
		return 0x0a5 << 22
	case AMOVW,
		AMOVWU:
		return 0x0a6 << 22
	case AMOVV:
		return 0x0a7 << 22
	case AMOVF:
		return 0x0ad << 22
	case AMOVD:
		return 0x0af << 22
	case -AMOVB:
		return 0x0a0 << 22
	case -AMOVBU:
		return 0x0a8 << 22
	case -AMOVH:
		return 0x0a1 << 22
	case -AMOVHU:
		return 0x0a9 << 22
	case -AMOVW:
		return 0x0a2 << 22
	case -AMOVWU:
		return 0x0aa << 22
	case -AMOVV:
		return 0x0a3 << 22
	case -AMOVF:
		return 0x0ac << 22
	case -AMOVD:
		return 0x0ae << 22
	case -AVMOVQ:
		return 0x0b0 << 22 // vld
	case -AXVMOVQ:
		return 0x0b2 << 22 // xvld
	case AVMOVQ:
		return 0x0b1 << 22 // vst
	case AXVMOVQ:
		return 0x0b3 << 22 // xvst
	case ASLLV:
		return 0x0041 << 16
	case ASRLV:
		return 0x0045 << 16
	case ASRAV:
		return 0x0049 << 16
	case AROTRV:
		return 0x004d << 16
	case -ALL:
		return 0x020 << 24
	case -ALLV:
		return 0x022 << 24
	case ASC:
		return 0x021 << 24
	case ASCV:
		return 0x023 << 24
	}

	if a < 0 {
		c.ctxt.Diag("bad irr opcode -%v", -a)
	} else {
		c.ctxt.Diag("bad irr opcode %v", a)
	}
	return 0
}

func (c *ctxt0) opirir(a obj.As) uint32 {
	switch a {
	case ABSTRINSW:
		return 0x3<<21 | 0x0<<15 // bstrins.w
	case ABSTRINSV:
		return 0x2 << 22 // bstrins.d
	case ABSTRPICKW:
		return 0x3<<21 | 0x1<<15 // bstrpick.w
	case ABSTRPICKV:
		return 0x3 << 22 // bstrpick.d
	}

	return 0
}

func (c *ctxt0) specialFpMovInst(a obj.As, fclass int, tclass int) uint32 {
	switch a {
	case AMOVV:
		switch fclass {
		case C_REG:
			switch tclass {
			case C_FREG:
				return 0x452a << 10 // movgr2fr.d
			case C_FCCREG:
				return 0x4536 << 10 // movgr2cf
			case C_FCSRREG:
				return 0x4530 << 10 // movgr2fcsr
			}
		case C_FREG:
			switch tclass {
			case C_REG:
				return 0x452e << 10 // movfr2gr.d
			case C_FCCREG:
				return 0x4534 << 10 // movfr2cf
			}
		case C_FCCREG:
			switch tclass {
			case C_REG:
				return 0x4537 << 10 // movcf2gr
			case C_FREG:
				return 0x4535 << 10 // movcf2fr
			}
		case C_FCSRREG:
			switch tclass {
			case C_REG:
				return 0x4532 << 10 // movfcsr2gr
			}
		}

	case AMOVW:
		switch fclass {
		case C_REG:
			switch tclass {
			case C_FREG:
				return 0x4529 << 10 // movgr2fr.w
			}
		case C_FREG:
			switch tclass {
			case C_REG:
				return 0x452d << 10 // movfr2gr.s
			}
		}
	}

	c.ctxt.Diag("bad class combination: %s %s,%s\n", a, fclass, tclass)

	return 0
}

func (c *ctxt0) specialLsxMovInst(a obj.As, fReg, tReg int16) (op_code, index_mask uint32) {
	farng := (fReg >> EXT_TYPE_SHIFT) & EXT_TYPE_MASK
	tarng := (tReg >> EXT_TYPE_SHIFT) & EXT_TYPE_MASK
	fclass := c.rclass(fReg)
	tclass := c.rclass(tReg)

	switch fclass | (tclass << 16) {
	case C_REG | (C_ELEM << 16):
		// vmov Rn, Vd.<T>[index]
		switch a {
		case AVMOVQ:
			switch tarng {
			case ARNG_B:
				return (0x01CBAE << 14), 0xf // vinsgr2vr.b
			case ARNG_H:
				return (0x03975E << 13), 0x7 // vinsgr2vr.h
			case ARNG_W:
				return (0x072EBE << 12), 0x3 // vinsgr2vr.w
			case ARNG_V:
				return (0x0E5D7E << 11), 0x1 // vinsgr2vr.d
			}
		case AXVMOVQ:
			switch tarng {
			case ARNG_W:
				return (0x03B75E << 13), 0x7 // xvinsgr2vr.w
			case ARNG_V:
				return (0x076EBE << 12), 0x3 // xvinsgr2vr.d
			}
		}

	case C_ELEM | (C_REG << 16):
		// vmov Vd.<T>[index], Rn
		switch a {
		case AVMOVQ:
			switch farng {
			case ARNG_B:
				return (0x01CBBE << 14), 0xf // vpickve2gr.b
			case ARNG_H:
				return (0x03977E << 13), 0x7 // vpickve2gr.h
			case ARNG_W:
				return (0x072EFE << 12), 0x3 // vpickve2gr.w
			case ARNG_V:
				return (0x0E5DFE << 11), 0x1 // vpickve2gr.d
			case ARNG_BU:
				return (0x01CBCE << 14), 0xf // vpickve2gr.bu
			case ARNG_HU:
				return (0x03979E << 13), 0x7 // vpickve2gr.hu
			case ARNG_WU:
				return (0x072F3E << 12), 0x3 // vpickve2gr.wu
			case ARNG_VU:
				return (0x0E5E7E << 11), 0x1 // vpickve2gr.du
			}
		case AXVMOVQ:
			switch farng {
			case ARNG_W:
				return (0x03B77E << 13), 0x7 // xvpickve2gr.w
			case ARNG_V:
				return (0x076EFE << 12), 0x3 // xvpickve2gr.d
			case ARNG_WU:
				return (0x03B79E << 13), 0x7 // xvpickve2gr.wu
			case ARNG_VU:
				return (0x076F3E << 12), 0x3 // xvpickve2gr.du
			}
		}

	case C_REG | (C_ARNG << 16):
		// vmov Rn, Vd.<T>
		switch a {
		case AVMOVQ:
			switch tarng {
			case ARNG_16B:
				return (0x1CA7C0 << 10), 0x0 // vreplgr2vr.b
			case ARNG_8H:
				return (0x1CA7C1 << 10), 0x0 // vreplgr2vr.h
			case ARNG_4W:
				return (0x1CA7C2 << 10), 0x0 // vreplgr2vr.w
			case ARNG_2V:
				return (0x1CA7C3 << 10), 0x0 // vreplgr2vr.d
			}
		case AXVMOVQ:
			switch tarng {
			case ARNG_32B:
				return (0x1DA7C0 << 10), 0x0 // xvreplgr2vr.b
			case ARNG_16H:
				return (0x1DA7C1 << 10), 0x0 // xvreplgr2vr.h
			case ARNG_8W:
				return (0x1DA7C2 << 10), 0x0 // xvreplgr2vr.w
			case ARNG_4V:
				return (0x1DA7C3 << 10), 0x0 // xvreplgr2vr.d
			}
		}

	case C_XREG | (C_ARNG << 16):
		// vmov  xj, xd.<T>
		switch a {
		case AVMOVQ:
			return 0, 0 // unsupported op
		case AXVMOVQ:
			switch tarng {
			case ARNG_32B:
				return (0x1DC1C0 << 10), 0x0 // xvreplve0.b
			case ARNG_16H:
				return (0x1DC1E0 << 10), 0x0 // xvreplve0.h
			case ARNG_8W:
				return (0x1DC1F0 << 10), 0x0 // xvreplve0.w
			case ARNG_4V:
				return (0x1DC1F8 << 10), 0x0 // xvreplve0.d
			case ARNG_2Q:
				return (0x1DC1FC << 10), 0x0 // xvreplve0.q
			}
		}

	case C_XREG | (C_ELEM << 16):
		// vmov  xj, xd.<T>[index]
		switch a {
		case AVMOVQ:
			return 0, 0 // unsupported op
		case AXVMOVQ:
			switch tarng {
			case ARNG_W:
				return (0x03B7FE << 13), 0x7 // xvinsve0.w
			case ARNG_V:
				return (0x076FFE << 12), 0x3 // xvinsve0.d
			}
		}

	case C_ELEM | (C_XREG << 16):
		// vmov  xj.<T>[index], xd
		switch a {
		case AVMOVQ:
			return 0, 0 // unsupported op
		case AXVMOVQ:
			switch farng {
			case ARNG_W:
				return (0x03B81E << 13), 0x7 // xvpickve.w
			case ARNG_V:
				return (0x07703E << 12), 0x3 // xvpickve.d
			}
		}

	case C_ELEM | (C_ARNG << 16):
		// vmov  vj.<T>[index], vd.<T>
		switch a {
		case AVMOVQ:
			switch int32(farng) | (int32(tarng) << 16) {
			case int32(ARNG_B) | (int32(ARNG_16B) << 16):
				return (0x01CBDE << 14), 0xf // vreplvei.b
			case int32(ARNG_H) | (int32(ARNG_8H) << 16):
				return (0x0397BE << 13), 0x7 // vreplvei.h
			case int32(ARNG_W) | (int32(ARNG_4W) << 16):
				return (0x072F7E << 12), 0x3 // vreplvei.w
			case int32(ARNG_V) | (int32(ARNG_2V) << 16):
				return (0x0E5EFE << 11), 0x1 // vreplvei.d
			}
		case AXVMOVQ:
			return 0, 0 // unsupported op
		}
	}

	return 0, 0
}

func vshift(a obj.As) bool {
	switch a {
	case ASLLV,
		ASRLV,
		ASRAV,
		AROTRV:
		return true
	}
	return false
}
