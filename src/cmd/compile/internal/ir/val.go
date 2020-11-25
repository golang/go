// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"go/constant"
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
)

func ConstType(n Node) constant.Kind {
	if n == nil || n.Op() != OLITERAL {
		return constant.Unknown
	}
	return n.Val().Kind()
}

// ValueInterface returns the constant value stored in n as an interface{}.
// It returns int64s for ints and runes, float64s for floats,
// and complex128s for complex values.
func ConstValue(n Node) interface{} {
	switch v := n.Val(); v.Kind() {
	default:
		base.Fatalf("unexpected constant: %v", v)
		panic("unreachable")
	case constant.Bool:
		return constant.BoolVal(v)
	case constant.String:
		return constant.StringVal(v)
	case constant.Int:
		return Int64Val(n.Type(), v)
	case constant.Float:
		return Float64Val(v)
	case constant.Complex:
		return complex(Float64Val(constant.Real(v)), Float64Val(constant.Imag(v)))
	}
}

// int64Val returns v converted to int64.
// Note: if t is uint64, very large values will be converted to negative int64.
func Int64Val(t *types.Type, v constant.Value) int64 {
	if t.IsUnsigned() {
		if x, ok := constant.Uint64Val(v); ok {
			return int64(x)
		}
	} else {
		if x, ok := constant.Int64Val(v); ok {
			return x
		}
	}
	base.Fatalf("%v out of range for %v", v, t)
	panic("unreachable")
}

func Float64Val(v constant.Value) float64 {
	if x, _ := constant.Float64Val(v); !math.IsInf(x, 0) {
		return x + 0 // avoid -0 (should not be needed, but be conservative)
	}
	base.Fatalf("bad float64 value: %v", v)
	panic("unreachable")
}

func AssertValidTypeForConst(t *types.Type, v constant.Value) {
	if !ValidTypeForConst(t, v) {
		base.Fatalf("%v does not represent %v", t, v)
	}
}

func ValidTypeForConst(t *types.Type, v constant.Value) bool {
	switch v.Kind() {
	case constant.Unknown:
		return OKForConst[t.Etype]
	case constant.Bool:
		return t.IsBoolean()
	case constant.String:
		return t.IsString()
	case constant.Int:
		return t.IsInteger()
	case constant.Float:
		return t.IsFloat()
	case constant.Complex:
		return t.IsComplex()
	}

	base.Fatalf("unexpected constant kind: %v", v)
	panic("unreachable")
}

// nodlit returns a new untyped constant with value v.
func NewLiteral(v constant.Value) Node {
	n := Nod(OLITERAL, nil, nil)
	if k := v.Kind(); k != constant.Unknown {
		n.SetType(idealType(k))
		n.SetVal(v)
	}
	return n
}

func idealType(ct constant.Kind) *types.Type {
	switch ct {
	case constant.String:
		return types.UntypedString
	case constant.Bool:
		return types.UntypedBool
	case constant.Int:
		return types.UntypedInt
	case constant.Float:
		return types.UntypedFloat
	case constant.Complex:
		return types.UntypedComplex
	}
	base.Fatalf("unexpected Ctype: %v", ct)
	return nil
}

var OKForConst [types.NTYPE]bool
