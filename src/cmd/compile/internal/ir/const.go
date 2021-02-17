// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"go/constant"
	"math"
	"math/big"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
)

func NewBool(b bool) Node {
	return NewLiteral(constant.MakeBool(b))
}

func NewInt(v int64) Node {
	return NewLiteral(constant.MakeInt64(v))
}

func NewString(s string) Node {
	return NewLiteral(constant.MakeString(s))
}

const (
	// Maximum size in bits for big.Ints before signalling
	// overflow and also mantissa precision for big.Floats.
	ConstPrec = 512
)

func BigFloat(v constant.Value) *big.Float {
	f := new(big.Float)
	f.SetPrec(ConstPrec)
	switch u := constant.Val(v).(type) {
	case int64:
		f.SetInt64(u)
	case *big.Int:
		f.SetInt(u)
	case *big.Float:
		f.Set(u)
	case *big.Rat:
		f.SetRat(u)
	default:
		base.Fatalf("unexpected: %v", u)
	}
	return f
}

// ConstOverflow reports whether constant value v is too large
// to represent with type t.
func ConstOverflow(v constant.Value, t *types.Type) bool {
	switch {
	case t.IsInteger():
		bits := uint(8 * t.Size())
		if t.IsUnsigned() {
			x, ok := constant.Uint64Val(v)
			return !ok || x>>bits != 0
		}
		x, ok := constant.Int64Val(v)
		if x < 0 {
			x = ^x
		}
		return !ok || x>>(bits-1) != 0
	case t.IsFloat():
		switch t.Size() {
		case 4:
			f, _ := constant.Float32Val(v)
			return math.IsInf(float64(f), 0)
		case 8:
			f, _ := constant.Float64Val(v)
			return math.IsInf(f, 0)
		}
	case t.IsComplex():
		ft := types.FloatForComplex(t)
		return ConstOverflow(constant.Real(v), ft) || ConstOverflow(constant.Imag(v), ft)
	}
	base.Fatalf("ConstOverflow: %v, %v", v, t)
	panic("unreachable")
}

// IsConstNode reports whether n is a Go language constant (as opposed to a
// compile-time constant).
//
// Expressions derived from nil, like string([]byte(nil)), while they
// may be known at compile time, are not Go language constants.
func IsConstNode(n Node) bool {
	return n.Op() == OLITERAL
}

func IsSmallIntConst(n Node) bool {
	if n.Op() == OLITERAL {
		v, ok := constant.Int64Val(n.Val())
		return ok && int64(int32(v)) == v
	}
	return false
}
