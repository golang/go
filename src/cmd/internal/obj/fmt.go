/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

package obj

// (The comments in this file were copied from the manpage files rune.3,
// isalpharune.3, and runestrcat.3. Some formatting changes were also made
// to conform to Google style. /JRM 11/11/05)

type Fmt struct {
	runes     uint8
	start     interface{}
	to        interface{}
	stop      interface{}
	flush     func(*Fmt) int
	farg      interface{}
	nfmt      int
	args      []interface{}
	r         uint
	width     int
	prec      int
	flags     uint32
	decimal   string
	thousands string
	grouping  string
}

const (
	FmtWidth    = 1
	FmtLeft     = FmtWidth << 1
	FmtPrec     = FmtLeft << 1
	FmtSharp    = FmtPrec << 1
	FmtSpace    = FmtSharp << 1
	FmtSign     = FmtSpace << 1
	FmtApost    = FmtSign << 1
	FmtZero     = FmtApost << 1
	FmtUnsigned = FmtZero << 1
	FmtShort    = FmtUnsigned << 1
	FmtLong     = FmtShort << 1
	FmtVLong    = FmtLong << 1
	FmtComma    = FmtVLong << 1
	FmtByte     = FmtComma << 1
	FmtLDouble  = FmtByte << 1
	FmtFlag     = FmtLDouble << 1
)

var fmtdoquote func(int) int

/* Edit .+1,/^$/ | cfn $PLAN9/src/lib9/fmt/?*.c | grep -v static |grep -v __ */
