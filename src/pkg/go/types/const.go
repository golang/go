// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements operations on constant values.

package types

import (
	"fmt"
	"go/token"
	"math/big"
	"strconv"
)

// TODO(gri) At the moment, constants are different types
// passed around as interface{} values. Consider introducing
// a Const type and use methods instead of xConst functions.

// Representation of constant values.
//
// bool     ->  bool (true, false)
// numeric  ->  int64, *big.Int, *big.Rat, Complex (ordered by increasing data structure "size")
// string   ->  string
// nil      ->  NilType (nilConst)
//
// Numeric constants are normalized after each operation such
// that they are represented by the "smallest" data structure
// required to represent the constant, independent of actual
// type. Non-numeric constants are always normalized.

// Representation of complex numbers.
type Complex struct {
	Re, Im *big.Rat
}

func (c Complex) String() string {
	if c.Re.Sign() == 0 {
		return fmt.Sprintf("%si", c.Im)
	}
	// normalized complex values always have an imaginary part
	return fmt.Sprintf("(%s + %si)", c.Re, c.Im)
}

// Representation of nil.
type NilType struct{}

func (NilType) String() string {
	return "nil"
}

// Implementation-specific constants.
// TODO(gri) These need to go elsewhere.
const (
	intBits = 32
	ptrBits = 64
)

// Frequently used values.
var (
	nilConst  = NilType{}
	zeroConst = int64(0)
)

// int64 bounds
var (
	minInt64 = big.NewInt(-1 << 63)
	maxInt64 = big.NewInt(1<<63 - 1)
)

// normalizeIntConst returns the smallest constant representation
// for the specific value of x; either an int64 or a *big.Int value.
//
func normalizeIntConst(x *big.Int) interface{} {
	if minInt64.Cmp(x) <= 0 && x.Cmp(maxInt64) <= 0 {
		return x.Int64()
	}
	return x
}

// normalizeRatConst returns the smallest constant representation
// for the specific value of x; either an int64, *big.Int,
// or *big.Rat value.
//
func normalizeRatConst(x *big.Rat) interface{} {
	if x.IsInt() {
		return normalizeIntConst(x.Num())
	}
	return x
}

// newComplex returns the smallest constant representation
// for the specific value re + im*i; either an int64, *big.Int,
// *big.Rat, or complex value.
//
func newComplex(re, im *big.Rat) interface{} {
	if im.Sign() == 0 {
		return normalizeRatConst(re)
	}
	return Complex{re, im}
}

// makeRuneConst returns the int64 code point for the rune literal
// lit. The result is nil if lit is not a correct rune literal.
//
func makeRuneConst(lit string) interface{} {
	if n := len(lit); n >= 2 {
		if code, _, _, err := strconv.UnquoteChar(lit[1:n-1], '\''); err == nil {
			return int64(code)
		}
	}
	return nil
}

// makeRuneConst returns the smallest integer constant representation
// (int64, *big.Int) for the integer literal lit. The result is nil if
// lit is not a correct integer literal.
//
func makeIntConst(lit string) interface{} {
	if x, err := strconv.ParseInt(lit, 0, 64); err == nil {
		return x
	}
	if x, ok := new(big.Int).SetString(lit, 0); ok {
		return x
	}
	return nil
}

// makeFloatConst returns the smallest floating-point constant representation
// (int64, *big.Int, *big.Rat) for the floating-point literal lit. The result
// is nil if lit is not a correct floating-point literal.
//
func makeFloatConst(lit string) interface{} {
	if x, ok := new(big.Rat).SetString(lit); ok {
		return normalizeRatConst(x)
	}
	return nil
}

// makeComplexConst returns the complex constant representation (Complex) for
// the imaginary literal lit. The result is nil if lit is not a correct imaginary
// literal.
//
func makeComplexConst(lit string) interface{} {
	n := len(lit)
	if n > 0 && lit[n-1] == 'i' {
		if im, ok := new(big.Rat).SetString(lit[0 : n-1]); ok {
			return newComplex(big.NewRat(0, 1), im)
		}
	}
	return nil
}

// makeStringConst returns the string constant representation (string) for
// the string literal lit. The result is nil if lit is not a correct string
// literal.
//
func makeStringConst(lit string) interface{} {
	if s, err := strconv.Unquote(lit); err == nil {
		return s
	}
	return nil
}

// toImagConst returns the constant Complex(0, x) for a non-complex x.
func toImagConst(x interface{}) interface{} {
	var im *big.Rat
	switch x := x.(type) {
	case int64:
		im = big.NewRat(x, 1)
	case *big.Int:
		im = new(big.Rat).SetFrac(x, int1)
	case *big.Rat:
		im = x
	default:
		unreachable()
	}
	return Complex{rat0, im}
}

// isZeroConst reports whether the value of constant x is 0.
// x must be normalized.
//
func isZeroConst(x interface{}) bool {
	i, ok := x.(int64) // good enough since constants are normalized
	return ok && i == 0
}

// isNegConst reports whether the value of constant x is < 0.
// x must be a non-complex numeric value.
//
func isNegConst(x interface{}) bool {
	switch x := x.(type) {
	case int64:
		return x < 0
	case *big.Int:
		return x.Sign() < 0
	case *big.Rat:
		return x.Sign() < 0
	}
	unreachable()
	return false
}

// isRepresentableConst reports whether the value of constant x can
// be represented as a value of the basic type Typ[as] without loss
// of precision.
//
func isRepresentableConst(x interface{}, as BasicKind) bool {
	switch x := x.(type) {
	case bool:
		return as == Bool || as == UntypedBool

	case int64:
		switch as {
		case Int:
			return -1<<(intBits-1) <= x && x <= 1<<(intBits-1)-1
		case Int8:
			return -1<<(8-1) <= x && x <= 1<<(8-1)-1
		case Int16:
			return -1<<(16-1) <= x && x <= 1<<(16-1)-1
		case Int32, UntypedRune:
			return -1<<(32-1) <= x && x <= 1<<(32-1)-1
		case Int64:
			return true
		case Uint:
			return 0 <= x && x <= 1<<intBits-1
		case Uint8:
			return 0 <= x && x <= 1<<8-1
		case Uint16:
			return 0 <= x && x <= 1<<16-1
		case Uint32:
			return 0 <= x && x <= 1<<32-1
		case Uint64:
			return 0 <= x
		case Uintptr:
			assert(ptrBits == 64)
			return 0 <= x
		case Float32:
			return true // TODO(gri) fix this
		case Float64:
			return true // TODO(gri) fix this
		case Complex64:
			return true // TODO(gri) fix this
		case Complex128:
			return true // TODO(gri) fix this
		case UntypedInt, UntypedFloat, UntypedComplex:
			return true
		}

	case *big.Int:
		switch as {
		case Uint:
			return x.Sign() >= 0 && x.BitLen() <= intBits
		case Uint64:
			return x.Sign() >= 0 && x.BitLen() <= 64
		case Uintptr:
			return x.Sign() >= 0 && x.BitLen() <= ptrBits
		case Float32:
			return true // TODO(gri) fix this
		case Float64:
			return true // TODO(gri) fix this
		case Complex64:
			return true // TODO(gri) fix this
		case Complex128:
			return true // TODO(gri) fix this
		case UntypedInt, UntypedFloat, UntypedComplex:
			return true
		}

	case *big.Rat:
		switch as {
		case Float32:
			return true // TODO(gri) fix this
		case Float64:
			return true // TODO(gri) fix this
		case Complex64:
			return true // TODO(gri) fix this
		case Complex128:
			return true // TODO(gri) fix this
		case UntypedFloat, UntypedComplex:
			return true
		}

	case Complex:
		switch as {
		case Complex64:
			return true // TODO(gri) fix this
		case Complex128:
			return true // TODO(gri) fix this
		case UntypedComplex:
			return true
		}

	case string:
		return as == String || as == UntypedString

	case NilType:
		return as == UntypedNil || as == UnsafePointer

	default:
		unreachable()
	}

	return false
}

var (
	int1 = big.NewInt(1)
	rat0 = big.NewRat(0, 1)
)

// complexity returns a measure of representation complexity for constant x.
func complexity(x interface{}) int {
	switch x.(type) {
	case bool, string, NilType:
		return 1
	case int64:
		return 2
	case *big.Int:
		return 3
	case *big.Rat:
		return 4
	case Complex:
		return 5
	}
	unreachable()
	return 0
}

// matchConst returns the matching representation (same type) with the
// smallest complexity for two constant values x and y. They must be
// of the same "kind" (boolean, numeric, string, or NilType).
//
func matchConst(x, y interface{}) (_, _ interface{}) {
	if complexity(x) > complexity(y) {
		y, x = matchConst(y, x)
		return x, y
	}
	// complexity(x) <= complexity(y)

	switch x := x.(type) {
	case bool, Complex, string, NilType:
		return x, y

	case int64:
		switch y := y.(type) {
		case int64:
			return x, y
		case *big.Int:
			return big.NewInt(x), y
		case *big.Rat:
			return big.NewRat(x, 1), y
		case Complex:
			return Complex{big.NewRat(x, 1), rat0}, y
		}

	case *big.Int:
		switch y := y.(type) {
		case *big.Int:
			return x, y
		case *big.Rat:
			return new(big.Rat).SetFrac(x, int1), y
		case Complex:
			return Complex{new(big.Rat).SetFrac(x, int1), rat0}, y
		}

	case *big.Rat:
		switch y := y.(type) {
		case *big.Rat:
			return x, y
		case Complex:
			return Complex{x, rat0}, y
		}
	}

	unreachable()
	return nil, nil
}

// is32bit reports whether x can be represented using 32 bits.
func is32bit(x int64) bool {
	return -1<<31 <= x && x <= 1<<31-1
}

// is63bit reports whether x can be represented using 63 bits.
func is63bit(x int64) bool {
	return -1<<62 <= x && x <= 1<<62-1
}

// unaryOpConst returns the result of the constant evaluation op x where x is of the given type.
func unaryOpConst(x interface{}, op token.Token, typ *Basic) interface{} {
	switch op {
	case token.ADD:
		return x // nothing to do
	case token.SUB:
		switch x := x.(type) {
		case int64:
			if z := -x; z != x {
				return z // no overflow
			}
			// overflow - need to convert to big.Int
			return normalizeIntConst(new(big.Int).Neg(big.NewInt(x)))
		case *big.Int:
			return normalizeIntConst(new(big.Int).Neg(x))
		case *big.Rat:
			return normalizeRatConst(new(big.Rat).Neg(x))
		case Complex:
			return newComplex(new(big.Rat).Neg(x.Re), new(big.Rat).Neg(x.Im))
		}
	case token.XOR:
		var z big.Int
		switch x := x.(type) {
		case int64:
			z.Not(big.NewInt(x))
		case *big.Int:
			z.Not(x)
		default:
			unreachable()
		}
		// For unsigned types, the result will be negative and
		// thus "too large": We must limit the result size to
		// the type's size.
		if typ.Info&IsUnsigned != 0 {
			s := uint(typ.Size) * 8
			if s == 0 {
				// platform-specific type
				// TODO(gri) this needs to be factored out
				switch typ.Kind {
				case Uint:
					s = intBits
				case Uintptr:
					s = ptrBits
				default:
					unreachable()
				}
			}
			// z &^= (-1)<<s
			z.AndNot(&z, new(big.Int).Lsh(big.NewInt(-1), s))
		}
		return normalizeIntConst(&z)
	case token.NOT:
		return !x.(bool)
	}
	unreachable()
	return nil
}

// binaryOpConst returns the result of the constant evaluation x op y;
// both operands must be of the same constant "kind" (boolean, numeric, or string).
// If typ is an integer type, division (op == token.QUO) is using integer division
// (and the result is guaranteed to be integer) rather than floating-point
// division. Division by zero leads to a run-time panic.
//
func binaryOpConst(x, y interface{}, op token.Token, typ *Basic) interface{} {
	x, y = matchConst(x, y)

	switch x := x.(type) {
	case bool:
		y := y.(bool)
		switch op {
		case token.LAND:
			return x && y
		case token.LOR:
			return x || y
		}

	case int64:
		y := y.(int64)
		switch op {
		case token.ADD:
			// TODO(gri) can do better than this
			if is63bit(x) && is63bit(y) {
				return x + y
			}
			return normalizeIntConst(new(big.Int).Add(big.NewInt(x), big.NewInt(y)))
		case token.SUB:
			// TODO(gri) can do better than this
			if is63bit(x) && is63bit(y) {
				return x - y
			}
			return normalizeIntConst(new(big.Int).Sub(big.NewInt(x), big.NewInt(y)))
		case token.MUL:
			// TODO(gri) can do better than this
			if is32bit(x) && is32bit(y) {
				return x * y
			}
			return normalizeIntConst(new(big.Int).Mul(big.NewInt(x), big.NewInt(y)))
		case token.REM:
			return x % y
		case token.QUO:
			if typ.Info&IsInteger != 0 {
				return x / y
			}
			return normalizeRatConst(new(big.Rat).SetFrac(big.NewInt(x), big.NewInt(y)))
		case token.AND:
			return x & y
		case token.OR:
			return x | y
		case token.XOR:
			return x ^ y
		case token.AND_NOT:
			return x &^ y
		}

	case *big.Int:
		y := y.(*big.Int)
		var z big.Int
		switch op {
		case token.ADD:
			z.Add(x, y)
		case token.SUB:
			z.Sub(x, y)
		case token.MUL:
			z.Mul(x, y)
		case token.REM:
			z.Rem(x, y)
		case token.QUO:
			if typ.Info&IsInteger != 0 {
				z.Quo(x, y)
			} else {
				return normalizeRatConst(new(big.Rat).SetFrac(x, y))
			}
		case token.AND:
			z.And(x, y)
		case token.OR:
			z.Or(x, y)
		case token.XOR:
			z.Xor(x, y)
		case token.AND_NOT:
			z.AndNot(x, y)
		default:
			unreachable()
		}
		return normalizeIntConst(&z)

	case *big.Rat:
		y := y.(*big.Rat)
		var z big.Rat
		switch op {
		case token.ADD:
			z.Add(x, y)
		case token.SUB:
			z.Sub(x, y)
		case token.MUL:
			z.Mul(x, y)
		case token.QUO:
			z.Quo(x, y)
		default:
			unreachable()
		}
		return normalizeRatConst(&z)

	case Complex:
		y := y.(Complex)
		a, b := x.Re, x.Im
		c, d := y.Re, y.Im
		var re, im big.Rat
		switch op {
		case token.ADD:
			// (a+c) + i(b+d)
			re.Add(a, c)
			im.Add(b, d)
		case token.SUB:
			// (a-c) + i(b-d)
			re.Sub(a, c)
			im.Sub(b, d)
		case token.MUL:
			// (ac-bd) + i(bc+ad)
			var ac, bd, bc, ad big.Rat
			ac.Mul(a, c)
			bd.Mul(b, d)
			bc.Mul(b, c)
			ad.Mul(a, d)
			re.Sub(&ac, &bd)
			im.Add(&bc, &ad)
		case token.QUO:
			// (ac+bd)/s + i(bc-ad)/s, with s = cc + dd
			var ac, bd, bc, ad, s big.Rat
			ac.Mul(a, c)
			bd.Mul(b, d)
			bc.Mul(b, c)
			ad.Mul(a, d)
			s.Add(c.Mul(c, c), d.Mul(d, d))
			re.Add(&ac, &bd)
			re.Quo(&re, &s)
			im.Sub(&bc, &ad)
			im.Quo(&im, &s)
		default:
			unreachable()
		}
		return newComplex(&re, &im)

	case string:
		if op == token.ADD {
			return x + y.(string)
		}
	}

	unreachable()
	return nil
}

// shiftConst returns the result of the constant evaluation x op s
// where op is token.SHL or token.SHR (<< or >>). x must be an
// integer constant.
//
func shiftConst(x interface{}, s uint, op token.Token) interface{} {
	switch x := x.(type) {
	case int64:
		switch op {
		case token.SHL:
			z := big.NewInt(x)
			return normalizeIntConst(z.Lsh(z, s))
		case token.SHR:
			return x >> s
		}

	case *big.Int:
		var z big.Int
		switch op {
		case token.SHL:
			return normalizeIntConst(z.Lsh(x, s))
		case token.SHR:
			return normalizeIntConst(z.Rsh(x, s))
		}
	}

	unreachable()
	return nil
}

// compareConst returns the result of the constant comparison x op y;
// both operands must be of the same "kind" (boolean, numeric, string,
// or NilType).
//
func compareConst(x, y interface{}, op token.Token) (z bool) {
	x, y = matchConst(x, y)

	// x == y  =>  x == y
	// x != y  =>  x != y
	// x >  y  =>  y <  x
	// x >= y  =>  u <= x
	swap := false
	switch op {
	case token.GTR:
		swap = true
		op = token.LSS
	case token.GEQ:
		swap = true
		op = token.LEQ
	}

	// x == y  =>    x == y
	// x != y  =>  !(x == y)
	// x <  y  =>    x <  y
	// x <= y  =>  !(y <  x)
	negate := false
	switch op {
	case token.NEQ:
		negate = true
		op = token.EQL
	case token.LEQ:
		swap = !swap
		negate = true
		op = token.LSS
	}

	if negate {
		defer func() { z = !z }()
	}

	if swap {
		x, y = y, x
	}

	switch x := x.(type) {
	case bool:
		if op == token.EQL {
			return x == y.(bool)
		}

	case int64:
		y := y.(int64)
		switch op {
		case token.EQL:
			return x == y
		case token.LSS:
			return x < y
		}

	case *big.Int:
		s := x.Cmp(y.(*big.Int))
		switch op {
		case token.EQL:
			return s == 0
		case token.LSS:
			return s < 0
		}

	case *big.Rat:
		s := x.Cmp(y.(*big.Rat))
		switch op {
		case token.EQL:
			return s == 0
		case token.LSS:
			return s < 0
		}

	case Complex:
		y := y.(Complex)
		if op == token.EQL {
			return x.Re.Cmp(y.Re) == 0 && x.Im.Cmp(y.Im) == 0
		}

	case string:
		y := y.(string)
		switch op {
		case token.EQL:
			return x == y
		case token.LSS:
			return x < y
		}

	case NilType:
		if op == token.EQL {
			return x == y.(NilType)
		}
	}

	fmt.Printf("x = %s (%T), y = %s (%T)\n", x, x, y, y)
	unreachable()
	return
}
