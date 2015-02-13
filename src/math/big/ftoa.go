// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the 'e', 'f', 'g' floating-point formats.
// It is closely following the corresponding implementation in
// strconv/ftoa.go, but modified and simplified for big.Float.

// Algorithm:
//   1) convert Float to multiprecision decimal
//   2) round to desired precision
//   3) read digits out and format

package big

import "strconv"

// TODO(gri) Consider moving sign into decimal - could make the signatures below cleaner.

// bigFtoa formats a float for the %e, %E, %f, %g, and %G formats.
func (f *Float) bigFtoa(buf []byte, fmt byte, prec int) []byte {
	// TODO(gri) handle Inf.

	// 1) convert Float to multiprecision decimal
	var d decimal
	d.init(f.mant, int(f.exp)-f.mant.bitLen())

	// 2) round to desired precision
	shortest := false
	if prec < 0 {
		shortest = true
		panic("unimplemented")
		// TODO(gri) complete this
		// roundShortest(&d, f.mant, int(f.exp))
		// Precision for shortest representation mode.
		switch fmt {
		case 'e', 'E':
			prec = len(d.mant) - 1
		case 'f':
			prec = max(len(d.mant)-d.exp, 0)
		case 'g', 'G':
			prec = len(d.mant)
		}
	} else {
		// round appropriately
		switch fmt {
		case 'e', 'E':
			// one digit before and number of digits after decimal point
			d.round(1 + prec)
		case 'f':
			// number of digits before and after decimal point
			d.round(d.exp + prec)
		case 'g', 'G':
			if prec == 0 {
				prec = 1
			}
			d.round(prec)
		}
	}

	// 3) read digits out and format
	switch fmt {
	case 'e', 'E':
		return fmtE(buf, fmt, prec, f.neg, d)
	case 'f':
		return fmtF(buf, prec, f.neg, d)
	case 'g', 'G':
		// trim trailing fractional zeros in %e format
		eprec := prec
		if eprec > len(d.mant) && len(d.mant) >= d.exp {
			eprec = len(d.mant)
		}
		// %e is used if the exponent from the conversion
		// is less than -4 or greater than or equal to the precision.
		// If precision was the shortest possible, use eprec = 6 for
		// this decision.
		if shortest {
			eprec = 6
		}
		exp := d.exp - 1
		if exp < -4 || exp >= eprec {
			if prec > len(d.mant) {
				prec = len(d.mant)
			}
			return fmtE(buf, fmt+'e'-'g', prec-1, f.neg, d)
		}
		if prec > d.exp {
			prec = len(d.mant)
		}
		return fmtF(buf, max(prec-d.exp, 0), f.neg, d)
	}

	// unknown format
	return append(buf, '%', fmt)
}

// %e: -d.ddddde±dd
func fmtE(buf []byte, fmt byte, prec int, neg bool, d decimal) []byte {
	// sign
	if neg {
		buf = append(buf, '-')
	}

	// first digit
	ch := byte('0')
	if len(d.mant) > 0 {
		ch = d.mant[0]
	}
	buf = append(buf, ch)

	// .moredigits
	if prec > 0 {
		buf = append(buf, '.')
		i := 1
		m := min(len(d.mant), prec+1)
		if i < m {
			buf = append(buf, d.mant[i:m]...)
			i = m
		}
		for ; i <= prec; i++ {
			buf = append(buf, '0')
		}
	}

	// e±
	buf = append(buf, fmt)
	var exp int64
	if len(d.mant) > 0 {
		exp = int64(d.exp) - 1 // -1 because first digit was printed before '.'
	}
	if exp < 0 {
		ch = '-'
		exp = -exp
	} else {
		ch = '+'
	}
	buf = append(buf, ch)

	// dd...d
	if exp < 10 {
		buf = append(buf, '0') // at least 2 exponent digits
	}
	return strconv.AppendInt(buf, exp, 10)
}

// %f: -ddddddd.ddddd
func fmtF(buf []byte, prec int, neg bool, d decimal) []byte {
	// sign
	if neg {
		buf = append(buf, '-')
	}

	// integer, padded with zeros as needed
	if d.exp > 0 {
		m := min(len(d.mant), d.exp)
		buf = append(buf, d.mant[:m]...)
		for ; m < d.exp; m++ {
			buf = append(buf, '0')
		}
	} else {
		buf = append(buf, '0')
	}

	// fraction
	if prec > 0 {
		buf = append(buf, '.')
		for i := 0; i < prec; i++ {
			ch := byte('0')
			if j := d.exp + i; 0 <= j && j < len(d.mant) {
				ch = d.mant[j]
			}
			buf = append(buf, ch)
		}
	}

	return buf
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
