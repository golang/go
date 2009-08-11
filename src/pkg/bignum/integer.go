// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Integer numbers
//
// Integers are normalized if the mantissa is normalized and the sign is
// false for mant == 0. Use MakeInt to create normalized Integers.

package bignum

import (
	"bignum";
	"fmt";
)

// TODO(gri) Complete the set of in-place operations.

// Integer represents a signed integer value of arbitrary precision.
//
type Integer struct {
	sign bool;
	mant Natural;
}


// MakeInt makes an integer given a sign and a mantissa.
// The number is positive (>= 0) if sign is false or the
// mantissa is zero; it is negative otherwise.
//
func MakeInt(sign bool, mant Natural) *Integer {
	if mant.IsZero() {
		sign = false;  // normalize
	}
	return &Integer{sign, mant};
}


// Int creates a small integer with value x.
//
func Int(x int64) *Integer {
	var ux uint64;
	if x < 0 {
		// For the most negative x, -x == x, and
		// the bit pattern has the correct value.
		ux = uint64(-x);
	} else {
		ux = uint64(x);
	}
	return MakeInt(x < 0, Nat(ux));
}


// Value returns the value of x, if x fits into an int64;
// otherwise the result is undefined.
//
func (x *Integer) Value() int64 {
	z := int64(x.mant.Value());
	if x.sign {
		z = -z;
	}
	return z;
}


// Abs returns the absolute value of x.
//
func (x *Integer) Abs() Natural {
	return x.mant;
}


// Predicates

// IsEven returns true iff x is divisible by 2.
//
func (x *Integer) IsEven() bool {
	return x.mant.IsEven();
}


// IsOdd returns true iff x is not divisible by 2.
//
func (x *Integer) IsOdd() bool {
	return x.mant.IsOdd();
}


// IsZero returns true iff x == 0.
//
func (x *Integer) IsZero() bool {
	return x.mant.IsZero();
}


// IsNeg returns true iff x < 0.
//
func (x *Integer) IsNeg() bool {
	return x.sign && !x.mant.IsZero()
}


// IsPos returns true iff x >= 0.
//
func (x *Integer) IsPos() bool {
	return !x.sign && !x.mant.IsZero()
}


// Operations

// Neg returns the negated value of x.
//
func (x *Integer) Neg() *Integer {
	return MakeInt(!x.sign, x.mant);
}


// Iadd sets z to the sum x + y.
// z must exist and may be x or y.
//
func Iadd(z, x, y *Integer) {
	if x.sign == y.sign {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z.sign = x.sign;
		Nadd(&z.mant, x.mant, y.mant);
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if x.mant.Cmp(y.mant) >= 0 {
			z.sign = x.sign;
			Nsub(&z.mant, x.mant, y.mant);
		} else {
			z.sign = !x.sign;
			Nsub(&z.mant, y.mant, x.mant);
		}
	}
}


// Add returns the sum x + y.
//
func (x *Integer) Add(y *Integer) *Integer {
	var z Integer;
	Iadd(&z, x, y);
	return &z;
}


func Isub(z, x, y *Integer) {
	if x.sign != y.sign {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z.sign = x.sign;
		Nadd(&z.mant, x.mant, y.mant);
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if x.mant.Cmp(y.mant) >= 0 {
			z.sign = x.sign;
			Nsub(&z.mant, x.mant, y.mant);
		} else {
			z.sign = !x.sign;
			Nsub(&z.mant, y.mant, x.mant);
		}
	}
}


// Sub returns the difference x - y.
//
func (x *Integer) Sub(y *Integer) *Integer {
	var z Integer;
	Isub(&z, x, y);
	return &z;
}


// Nscale sets *z to the scaled value (*z) * d.
//
func Iscale(z *Integer, d int64) {
	f := uint64(d);
	if d < 0 {
		f = uint64(-d);
	}
	z.sign = z.sign != (d < 0);
	Nscale(&z.mant, f);
}


// Mul1 returns the product x * d.
//
func (x *Integer) Mul1(d int64) *Integer {
	f := uint64(d);
	if d < 0 {
		f = uint64(-d);
	}
	return MakeInt(x.sign != (d < 0), x.mant.Mul1(f));
}


// Mul returns the product x * y.
//
func (x *Integer) Mul(y *Integer) *Integer {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	return MakeInt(x.sign != y.sign, x.mant.Mul(y.mant));
}


// MulNat returns the product x * y, where y is a (unsigned) natural number.
//
func (x *Integer) MulNat(y Natural) *Integer {
	// x * y == x * y
	// (-x) * y == -(x * y)
	return MakeInt(x.sign, x.mant.Mul(y));
}


// Quo returns the quotient q = x / y for y != 0.
// If y == 0, a division-by-zero run-time error occurs.
//
// Quo and Rem implement T-division and modulus (like C99):
//
//   q = x.Quo(y) = trunc(x/y)  (truncation towards zero)
//   r = x.Rem(y) = x - y*q
//
// (Daan Leijen, ``Division and Modulus for Computer Scientists''.)
//
func (x *Integer) Quo(y *Integer) *Integer {
	// x / y == x / y
	// x / (-y) == -(x / y)
	// (-x) / y == -(x / y)
	// (-x) / (-y) == x / y
	return MakeInt(x.sign != y.sign, x.mant.Div(y.mant));
}


// Rem returns the remainder r of the division x / y for y != 0,
// with r = x - y*x.Quo(y). Unless r is zero, its sign corresponds
// to the sign of x.
// If y == 0, a division-by-zero run-time error occurs.
//
func (x *Integer) Rem(y *Integer) *Integer {
	// x % y == x % y
	// x % (-y) == x % y
	// (-x) % y == -(x % y)
	// (-x) % (-y) == -(x % y)
	return MakeInt(x.sign, x.mant.Mod(y.mant));
}


// QuoRem returns the pair (x.Quo(y), x.Rem(y)) for y != 0.
// If y == 0, a division-by-zero run-time error occurs.
//
func (x *Integer) QuoRem(y *Integer) (*Integer, *Integer) {
	q, r := x.mant.DivMod(y.mant);
	return MakeInt(x.sign != y.sign, q), MakeInt(x.sign, r);
}


// Div returns the quotient q = x / y for y != 0.
// If y == 0, a division-by-zero run-time error occurs.
//
// Div and Mod implement Euclidian division and modulus:
//
//   q = x.Div(y)
//   r = x.Mod(y) with: 0 <= r < |q| and: y = x*q + r
//
// (Raymond T. Boute, ``The Euclidian definition of the functions
// div and mod''. ACM Transactions on Programming Languages and
// Systems (TOPLAS), 14(2):127-144, New York, NY, USA, 4/1992.
// ACM press.)
//
func (x *Integer) Div(y *Integer) *Integer {
	q, r := x.QuoRem(y);
	if r.IsNeg() {
		if y.IsPos() {
			q = q.Sub(Int(1));
		} else {
			q = q.Add(Int(1));
		}
	}
	return q;
}


// Mod returns the modulus r of the division x / y for y != 0,
// with r = x - y*x.Div(y). r is always positive.
// If y == 0, a division-by-zero run-time error occurs.
//
func (x *Integer) Mod(y *Integer) *Integer {
	r := x.Rem(y);
	if r.IsNeg() {
		if y.IsPos() {
			r = r.Add(y);
		} else {
			r = r.Sub(y);
		}
	}
	return r;
}


// DivMod returns the pair (x.Div(y), x.Mod(y)).
//
func (x *Integer) DivMod(y *Integer) (*Integer, *Integer) {
	q, r := x.QuoRem(y);
	if r.IsNeg() {
		if y.IsPos() {
			q = q.Sub(Int(1));
			r = r.Add(y);
		} else {
			q = q.Add(Int(1));
			r = r.Sub(y);
		}
	}
	return q, r;
}


// Shl implements ``shift left'' x << s. It returns x * 2^s.
//
func (x *Integer) Shl(s uint) *Integer {
	return MakeInt(x.sign, x.mant.Shl(s));
}


// The bitwise operations on integers are defined on the 2's-complement
// representation of integers. From
//
//   -x == ^x + 1  (1)  2's complement representation
//
// follows:
//
//   -(x) == ^(x) + 1
//   -(-x) == ^(-x) + 1
//   x-1 == ^(-x)
//   ^(x-1) == -x  (2)
//
// Using (1) and (2), operations on negative integers of the form -x are
// converted to operations on negated positive integers of the form ~(x-1).


// Shr implements ``shift right'' x >> s. It returns x / 2^s.
//
func (x *Integer) Shr(s uint) *Integer {
	if x.sign {
		// (-x) >> s == ^(x-1) >> s == ^((x-1) >> s) == -(((x-1) >> s) + 1)
		return MakeInt(true, x.mant.Sub(Nat(1)).Shr(s).Add(Nat(1)));
	}
	
	return MakeInt(false, x.mant.Shr(s));
}


// Not returns the ``bitwise not'' ^x for the 2's-complement representation of x.
func (x *Integer) Not() *Integer {
	if x.sign {
		// ^(-x) == ^(^(x-1)) == x-1
		return MakeInt(false, x.mant.Sub(Nat(1)));
	}

	// ^x == -x-1 == -(x+1)
	return MakeInt(true, x.mant.Add(Nat(1)));
}


// And returns the ``bitwise and'' x & y for the 2's-complement representation of x and y.
//
func (x *Integer) And(y *Integer) *Integer {
	if x.sign == y.sign {
		if x.sign {
			// (-x) & (-y) == ^(x-1) & ^(y-1) == ^((x-1) | (y-1)) == -(((x-1) | (y-1)) + 1)
			return MakeInt(true, x.mant.Sub(Nat(1)).Or(y.mant.Sub(Nat(1))).Add(Nat(1)));
		}

		// x & y == x & y
		return MakeInt(false, x.mant.And(y.mant));
	}

	// x.sign != y.sign
	if x.sign {
		x, y = y, x;  // & is symmetric
	}

	// x & (-y) == x & ^(y-1) == x &^ (y-1)
	return MakeInt(false, x.mant.AndNot(y.mant.Sub(Nat(1))));
}


// AndNot returns the ``bitwise clear'' x &^ y for the 2's-complement representation of x and y.
//
func (x *Integer) AndNot(y *Integer) *Integer {
	if x.sign == y.sign {
		if x.sign {
			// (-x) &^ (-y) == ^(x-1) &^ ^(y-1) == ^(x-1) & (y-1) == (y-1) &^ (x-1)
			return MakeInt(false, y.mant.Sub(Nat(1)).AndNot(x.mant.Sub(Nat(1))));
		}

		// x &^ y == x &^ y
		return MakeInt(false, x.mant.AndNot(y.mant));
	}

	if x.sign {
		// (-x) &^ y == ^(x-1) &^ y == ^(x-1) & ^y == ^((x-1) | y) == -(((x-1) | y) + 1)
		return MakeInt(true, x.mant.Sub(Nat(1)).Or(y.mant).Add(Nat(1)));
	}

	// x &^ (-y) == x &^ ^(y-1) == x & (y-1)
	return MakeInt(false, x.mant.And(y.mant.Sub(Nat(1))));
}


// Or returns the ``bitwise or'' x | y for the 2's-complement representation of x and y.
//
func (x *Integer) Or(y *Integer) *Integer {
	if x.sign == y.sign {
		if x.sign {
			// (-x) | (-y) == ^(x-1) | ^(y-1) == ^((x-1) & (y-1)) == -(((x-1) & (y-1)) + 1)
			return MakeInt(true, x.mant.Sub(Nat(1)).And(y.mant.Sub(Nat(1))).Add(Nat(1)));
		}

		// x | y == x | y
		return MakeInt(false, x.mant.Or(y.mant));
	}

	// x.sign != y.sign
	if x.sign {
		x, y = y, x;  // | or symmetric
	}

	// x | (-y) == x | ^(y-1) == ^((y-1) &^ x) == -(^((y-1) &^ x) + 1)
	return MakeInt(true, y.mant.Sub(Nat(1)).AndNot(x.mant).Add(Nat(1)));
}


// Xor returns the ``bitwise xor'' x | y for the 2's-complement representation of x and y.
//
func (x *Integer) Xor(y *Integer) *Integer {
	if x.sign == y.sign {
		if x.sign {
			// (-x) ^ (-y) == ^(x-1) ^ ^(y-1) == (x-1) ^ (y-1)
			return MakeInt(false, x.mant.Sub(Nat(1)).Xor(y.mant.Sub(Nat(1))));
		}

		// x ^ y == x ^ y
		return MakeInt(false, x.mant.Xor(y.mant));
	}

	// x.sign != y.sign
	if x.sign {
		x, y = y, x;  // ^ is symmetric
	}

	// x ^ (-y) == x ^ ^(y-1) == ^(x ^ (y-1)) == -((x ^ (y-1)) + 1)
	return MakeInt(true, x.mant.Xor(y.mant.Sub(Nat(1))).Add(Nat(1)));
}


// Cmp compares x and y. The result is an int value that is
//
//   <  0 if x <  y
//   == 0 if x == y
//   >  0 if x >  y
//
func (x *Integer) Cmp(y *Integer) int {
	// x cmp y == x cmp y
	// x cmp (-y) == x
	// (-x) cmp y == y
	// (-x) cmp (-y) == -(x cmp y)
	var r int;
	switch {
	case x.sign == y.sign:
		r = x.mant.Cmp(y.mant);
		if x.sign {
			r = -r;
		}
	case x.sign: r = -1;
	case y.sign: r = 1;
	}
	return r;
}


// ToString converts x to a string for a given base, with 2 <= base <= 16.
//
func (x *Integer) ToString(base uint) string {
	if x.mant.IsZero() {
		return "0";
	}
	var s string;
	if x.sign {
		s = "-";
	}
	return s + x.mant.ToString(base);
}


// String converts x to its decimal string representation.
// x.String() is the same as x.ToString(10).
//
func (x *Integer) String() string {
	return x.ToString(10);
}


// Format is a support routine for fmt.Formatter. It accepts
// the formats 'b' (binary), 'o' (octal), and 'x' (hexadecimal).
//
func (x *Integer) Format(h fmt.State, c int) {
	fmt.Fprintf(h, "%s", x.ToString(fmtbase(c)));
}


// IntFromString returns the integer corresponding to the
// longest possible prefix of s representing an integer in a
// given conversion base, the actual conversion base used, and
// the prefix length. The syntax of integers follows the syntax
// of signed integer literals in Go.
//
// If the base argument is 0, the string prefix determines the actual
// conversion base. A prefix of ``0x'' or ``0X'' selects base 16; the
// ``0'' prefix selects base 8. Otherwise the selected base is 10.
//
func IntFromString(s string, base uint) (*Integer, uint, int) {
	// skip sign, if any
	i0 := 0;
	if len(s) > 0 && (s[0] == '-' || s[0] == '+') {
		i0 = 1;
	}

	mant, base, slen := NatFromString(s[i0 : len(s)], base);

	return MakeInt(i0 > 0 && s[0] == '-', mant), base, i0 + slen;
}
