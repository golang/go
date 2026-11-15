// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const logosInterpType = "[correctness::interp_type] cannot interpolate value of type %s"

type LogosInterpKind int

const (
	LogosInterpString LogosInterpKind = iota
	LogosInterpBytes
	LogosInterpInt
	LogosInterpUint
	LogosInterpFloat32
	LogosInterpFloat64
	LogosInterpBool
	LogosInterpError
	LogosInterpStringer
)

func LogosInterpKindOf(T Type) (LogosInterpKind, bool) {
	if T == nil || !isValid(T) {
		return 0, false
	}
	if c, ok := Unalias(T).(*ConstView); ok {
		return LogosInterpKindOf(c.Elem)
	}
	if _, ok := Unalias(T).(*TypeParam); ok {
		return 0, false
	}
	if errIface, ok := universeError.Underlying().(*Interface); ok && Implements(T, errIface) {
		return LogosInterpError, true
	}
	if logosHasStringMethod(T) {
		return LogosInterpStringer, true
	}
	if b, ok := Unalias(T).Underlying().(*Basic); ok {
		if b.Kind() == UntypedNil {
			return 0, false
		}
		info := b.Info()
		switch {
		case info&IsString != 0:
			return LogosInterpString, true
		case info&IsBoolean != 0:
			return LogosInterpBool, true
		case info&IsInteger != 0:
			switch b.Kind() {
			case Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:
				return LogosInterpUint, true
			default:
				return LogosInterpInt, true
			}
		case info&IsFloat != 0:
			if b.Kind() == Float32 {
				return LogosInterpFloat32, true
			}
			return LogosInterpFloat64, true
		}
		return 0, false
	}
	if s, ok := Unalias(T).Underlying().(*Slice); ok {
		if e, ok := s.Elem().Underlying().(*Basic); ok && e.Kind() == Uint8 {
			return LogosInterpBytes, true
		}
		return 0, false
	}
	return 0, false
}

func logosHasStringMethod(T Type) bool {
	obj, _, _ := LookupFieldOrMethod(T, true, nil, "String")
	fn, ok := obj.(*Func)
	if !ok {
		return false
	}
	sig, ok := Unalias(fn.Type()).(*Signature)
	if !ok || sig.Params().Len() != 0 || sig.Results().Len() != 1 {
		return false
	}
	if b, ok := sig.Results().At(0).Type().Underlying().(*Basic); ok {
		return b.Info()&IsString != 0
	}
	return false
}

func (check *Checker) logosInterp(x *operand, e *syntax.InterpExpr) exprKind {
	for _, op := range e.Exprs {
		var o operand
		check.expr(nil, &o, op)
		if !o.isValid() {
			continue
		}
		if _, ok := LogosInterpKindOf(o.typ()); !ok {
			check.errorf(e, UnassignableOperand, logosInterpType, o.typ())
		}
	}
	x.mode_ = value
	x.typ_ = Typ[String]
	x.expr = e
	return expression
}
