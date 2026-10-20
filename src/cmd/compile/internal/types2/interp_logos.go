// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M6-S1 type rules for `$"..."` interpolated strings (SPEC Sec
// 1.4, T-104.4). Upstream files call in through one hook line only:
// expr.go. The rule applies in loGo/s packages; the parser only builds
// InterpExpr in LogosMode, so plain Go never reaches it.
//
// Each `{expr}` operand must have one of the allowed types: string, any
// integer, bool, float32 or float64, `[]byte`, `error` (bound to its
// `Error()` string by lowering), or `fmt.Stringer` (bound to its
// `String()` string). Anything else reports interp_type. Kind checks
// precede method checks, so a named integer with a String method still
// formats as an integer. The result is always string. Operands are typed
// once, left to right; single evaluation is lowering's job (M6-S1).

const logosInterpType = "[correctness::interp_type] cannot interpolate value of type %s"

// LogosInterpKind is one allowed `$"..."` operand kind (SPEC T-104.4).
// Lowering (cmd/compile/internal/logos) switches on it, so the checker
// and the generated appends agree by construction.
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

// LogosInterpKindOf classifies T for `$"..."` (SPEC T-104.4): string,
// any integer, bool, float32/float64, `[]byte`, `error`, or
// `fmt.Stringer`. Method checks precede kind checks, so a named string
// with a String method calls it, matching fmt. It reports false for
// anything else, including type parameters (whose core cannot size the
// buffer).
func LogosInterpKindOf(T Type) (LogosInterpKind, bool) {
	// A nil type is never allowed; note isValid(nil) is true, so the
	// nil check cannot fold into it.
	if T == nil || !isValid(T) {
		return 0, false
	}
	// A const view reads as its element (SPEC T-201: the qualifier is
	// erased before code generation), so a scalar read through a view
	// interpolates like the plain value.
	if c, ok := Unalias(T).(*ConstView); ok {
		return LogosInterpKindOf(c.Elem)
	}
	if _, ok := Unalias(T).(*TypeParam); ok {
		return 0, false
	}
	// universeError is the defined type error; its underlying interface
	// is the Implements target.
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

// logosHasStringMethod reports whether the method set of T (with T
// addressable, matching lowering's operand temporaries) holds
// `String() string`.
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

// logosInterp type-checks an *syntax.InterpExpr node: every operand must
// have an allowed type, and the expression has type string. A disallowed
// operand is reported at the `$`, because the operand sits inside the
// literal where no error pin can name its column.
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
