// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Decimal to binary floating point conversion.
// Algorithm:
//   1) Store input in multiprecision decimal.
//   2) Multiply/divide decimal by powers of two until in range [0.5, 1)
//   3) Multiply by 2^precision and round to get mantissa.

package strconv

import "strconv"

// TODO(rsc): Better truncation handling, check for overflow in exponent.
func StringToDecimal(s string) (neg bool, d *Decimal, trunc bool, ok bool) {
	i := 0;

	// optional sign
	if i >= len(s) {
		return;
	}
	switch {
	case s[i] == '+':
		i++;
	case s[i] == '-':
		neg = true;
		i++;
	}

	// digits
	b := new(Decimal);
	sawdot := false;
	sawdigits := false;
	for ; i < len(s); i++ {
		switch {
		case s[i] == '.':
			if sawdot {
				return;
			}
			sawdot = true;
			b.dp = b.nd;
			continue;

		case '0' <= s[i] && s[i] <= '9':
			sawdigits = true;
			if s[i] == '0' && b.nd == 0 {	// ignore leading zeros
				b.dp--;
				continue;
			}
			b.d[b.nd] = s[i];
			b.nd++;
			continue;
		}
		break;
	}
	if !sawdigits {
		return;
	}
	if !sawdot {
		b.dp = b.nd;
	}

	// optional exponent moves decimal point
	if i < len(s) && s[i] == 'e' {
		i++;
		if i >= len(s) {
			return;
		}
		esign := 1;
		if s[i] == '+' {
			i++;
		} else if s[i] == '-' {
			i++;
			esign = -1;
		}
		if i >= len(s) || s[i] < '0' || s[i] > '9' {
			return;
		}
		e := 0;
		for ; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
			e = e*10 + int(s[i]) - '0';
		}
		b.dp += e*esign;
	}

	if i != len(s) {
		return;
	}

	d = b;
	ok = true;
	return;
}

// Decimal power of ten to binary power of two.
var powtab = []int{
	1, 3, 6, 9, 13, 16, 19, 23, 26
}

func DecimalToFloatBits(neg bool, d *Decimal, trunc bool, flt *FloatInfo) (b uint64, overflow bool) {
	// Zero is always a special case.
	if d.nd == 0 {
		return 0, false
	}

	// TODO: check for obvious overflow

	// Scale by powers of two until in range [0.5, 1.0)
	exp := 0;
	for d.dp > 0 {
		var n int;
		if d.dp >= len(powtab) {
			n = 27;
		} else {
			n = powtab[d.dp];
		}
		d.Shift(-n);
		exp += n;
	}
	for d.dp < 0 || d.dp == 0 && d.d[0] < '5' {
		var n int;
		if -d.dp >= len(powtab) {
			n = 27;
		} else {
			n = powtab[-d.dp];
		}
		d.Shift(n);
		exp -= n;
	}

	// Our range is [0.5,1) but floating point range is [1,2).
	exp--;

	// Minimum representable exponent is flt.bias+1.
	// If the exponent is smaller, move it up and
	// adjust d accordingly.
	if exp < flt.bias+1 {
		n := flt.bias+1 - exp;
		d.Shift(-n);
		exp += n;
	}

	// TODO: overflow/underflow

	// Extract 1+flt.mantbits bits.
	mant := d.Shift(int(1+flt.mantbits)).RoundedInteger();

	// Denormalized?
	if mant&(1<<flt.mantbits) == 0 {
		if exp != flt.bias+1 {
			// TODO: remove - has no business panicking
			panicln("DecimalToFloatBits", exp, flt.bias+1);
		}
		exp--;
	} else {
		if exp <= flt.bias {
			// TODO: remove - has no business panicking
			panicln("DecimalToFloatBits1", exp, flt.bias);
		}
	}

	// Assemble bits.
	bits := mant & (uint64(1)<<flt.mantbits - 1);
	bits |= uint64((exp-flt.bias)&(1<<flt.expbits - 1)) << flt.mantbits;
	if neg {
		bits |= 1<<flt.mantbits<<flt.expbits;
	}
	return bits, false;
}

// If possible to convert decimal d to 64-bit float f exactly,
// entirely in floating-point math, do so, avoiding the machinery above.
func DecimalToFloat64(neg bool, d *Decimal, trunc bool) (f float64, ok bool) {
	// TODO: Fill in.
	return 0, false;
}

// If possible to convert decimal d to 32-bit float f exactly,
// entirely in floating-point math, do so, avoiding the machinery above.
func DecimalToFloat32(neg bool, d *Decimal, trunc bool) (f float32, ok bool) {
	// TODO: Fill in.
	return 0, false;
}

export func atof64(s string) (f float64, overflow bool, ok bool) {
	neg, d, trunc, ok1 := StringToDecimal(s);
	if !ok1 {
		return 0, false, false;
	}
	if f, ok := DecimalToFloat64(neg, d, trunc); ok {
		return f, false, true;
	}
	b, overflow1 := DecimalToFloatBits(neg, d, trunc, &float64info);
	return sys.float64frombits(b), overflow1, true;
}

export func atof32(s string) (f float32, overflow bool, ok bool) {
	neg, d, trunc, ok1 := StringToDecimal(s);
	if !ok1 {
		return 0, false, false;
	}
	if f, ok := DecimalToFloat32(neg, d, trunc); ok {
		return f, false, true;
	}
	b, overflow1 := DecimalToFloatBits(neg, d, trunc, &float32info);
	return sys.float32frombits(uint32(b)), overflow1, true;
}

export func atof(s string) (f float, overflow bool, ok bool) {
	if floatsize == 32 {
		var f1 float32;
		f1, overflow, ok = atof32(s);
		return float(f1), overflow, ok;
	}
	var f1 float64;
	f1, overflow, ok = atof64(s);
	return float(f1), overflow, ok;
}

