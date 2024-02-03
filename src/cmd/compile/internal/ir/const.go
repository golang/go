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
	"cmd/internal/src"
)

// NewBool returns an OLITERAL representing b as an untyped boolean.
func NewBool(pos src.XPos, b bool) Node {
	return NewBasicLit(pos, types.UntypedBool, constant.MakeBool(b))
}

// NewInt returns an OLITERAL representing v as an untyped integer.
func NewInt(pos src.XPos, v int64) Node {
	return NewBasicLit(pos, types.UntypedInt, constant.MakeInt64(v))
}

// NewString returns an OLITERAL representing s as an untyped string.
func NewString(pos src.XPos, s string) Node {
	return NewBasicLit(pos, types.UntypedString, constant.MakeString(s))
}

// NewUintptr returns an OLITERAL representing v as a uintptr.
func NewUintptr(pos src.XPos, v int64) Node {
	return NewBasicLit(pos, types.Types[types.TUINTPTR], constant.MakeInt64(v))
}

// NewZero returns a zero value of the given type.
func NewZero(pos src.XPos, typ *types.Type) Node {
	switch {
	case typ.HasNil():
		return NewNilExpr(pos, typ)
	case typ.IsInteger():
		return NewBasicLit(pos, typ, intZero)
	case typ.IsFloat():
		return NewBasicLit(pos, typ, floatZero)
	case typ.IsComplex():
		return NewBasicLit(pos, typ, complexZero)
	case typ.IsBoolean():
		return NewBasicLit(pos, typ, constant.MakeBool(false))
	case typ.IsString():
		return NewBasicLit(pos, typ, constant.MakeString(""))
	case typ.IsArray() || typ.IsStruct():
		// TODO(mdempsky): Return a typechecked expression instead.
		return NewCompLitExpr(pos, OCOMPLIT, typ, nil)
	}

	base.FatalfAt(pos, "unexpected type: %v", typ)
	panic("unreachable")
}

var (
	intZero     = constant.MakeInt64(0)
	floatZero   = constant.ToFloat(intZero)
	complexZero = constant.ToComplex(intZero)
)

// NewOne returns an OLITERAL representing 1 with the given type.
func NewOne(pos src.XPos, typ *types.Type) Node {
	var val constant.Value
	switch {
	case typ.IsInteger():
		val = intOne
	case typ.IsFloat():
		val = floatOne
	case typ.IsComplex():
		val = complexOne
	default:
		base.FatalfAt(pos, "%v cannot represent 1", typ)
	}

	return NewBasicLit(pos, typ, val)
}

var (
	intOne     = constant.MakeInt64(1)
	floatOne   = constant.ToFloat(intOne)
	complexOne = constant.ToComplex(intOne)
)

const (
	// Maximum size in bits for big.Ints before signaling
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
