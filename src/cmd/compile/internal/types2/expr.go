// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of expressions.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
	"go/token"
	"math"
)

/*
Basic algorithm:

Expressions are checked recursively, top down. Expression checker functions
are generally of the form:

  func f(x *operand, e *syntax.Expr, ...)

where e is the expression to be checked, and x is the result of the check.
The check performed by f may fail in which case x.mode == invalid, and
related error messages will have been issued by f.

If a hint argument is present, it is the composite literal element type
of an outer composite literal; it is used to type-check composite literal
elements that have no explicit type specification in the source
(e.g.: []T{{...}, {...}}, the hint is the type T in this case).

All expressions are checked via rawExpr, which dispatches according
to expression kind. Upon returning, rawExpr is recording the types and
constant values for all expressions that have an untyped type (those types
may change on the way up in the expression tree). Usually these are constants,
but the results of comparisons or non-constant shifts of untyped constants
may also be untyped, but not constant.

Untyped expressions may eventually become fully typed (i.e., not untyped),
typically when the value is assigned to a variable, or is used otherwise.
The updateExprType method is used to record this final type and update
the recorded types: the type-checked expression tree is again traversed down,
and the new type is propagated as needed. Untyped constant expression values
that become fully typed must now be representable by the full type (constant
sub-expression trees are left alone except for their roots). This mechanism
ensures that a client sees the actual (run-time) type an untyped value would
have. It also permits type-checking of lhs shift operands "as if the shift
were not present": when updateExprType visits an untyped lhs shift operand
and assigns it it's final type, that type must be an integer type, and a
constant lhs must be representable as an integer.

When an expression gets its final type, either on the way out from rawExpr,
on the way down in updateExprType, or at the end of the type checker run,
the type (and constant value, if any) is recorded via Info.Types, if present.
*/

type opPredicates map[syntax.Operator]func(Type) bool

var unaryOpPredicates = opPredicates{
	syntax.Add: isNumeric,
	syntax.Sub: isNumeric,
	syntax.Xor: isInteger,
	syntax.Not: isBoolean,
}

func (check *Checker) op(m opPredicates, x *operand, op syntax.Operator) bool {
	if pred := m[op]; pred != nil {
		if !pred(x.typ) {
			check.invalidOpf(x, "operator %s not defined for %s", op, x)
			return false
		}
	} else {
		check.invalidASTf(x, "unknown operator %s", op)
		return false
	}
	return true
}

func op2token(op syntax.Operator) token.Token {
	switch op {
	case syntax.Def: // :
		unreachable()
	case syntax.Not: // !
		return token.NOT
	case syntax.Recv: // <-
		unreachable()

	case syntax.OrOr: // ||
		return token.LOR
	case syntax.AndAnd: // &&
		return token.LAND

	case syntax.Eql: // ==
		return token.EQL
	case syntax.Neq: // !=
		return token.NEQ
	case syntax.Lss: // <
		return token.LSS
	case syntax.Leq: // <=
		return token.LEQ
	case syntax.Gtr: // >
		return token.GTR
	case syntax.Geq: // >=
		return token.GEQ

	case syntax.Add: // +
		return token.ADD
	case syntax.Sub: // -
		return token.SUB
	case syntax.Or: // |
		return token.OR
	case syntax.Xor: // ^
		return token.XOR

	case syntax.Mul: // *
		return token.MUL
	case syntax.Div: // /
		return token.QUO
	case syntax.Rem: // %
		return token.REM
	case syntax.And: // &
		return token.AND
	case syntax.AndNot: // &^
		return token.AND_NOT
	case syntax.Shl: // <<
		return token.SHL
	case syntax.Shr: // >>
		return token.SHR
	}

	return token.ILLEGAL
}

// The unary expression e may be nil. It's passed in for better error messages only.
func (check *Checker) unary(x *operand, e *syntax.Operation, op syntax.Operator) {
	switch op {
	case syntax.And:
		// spec: "As an exception to the addressability
		// requirement x may also be a composite literal."
		if _, ok := unparen(x.expr).(*syntax.CompositeLit); !ok && x.mode != variable {
			check.invalidOpf(x, "cannot take address of %s", x)
			x.mode = invalid
			return
		}
		x.mode = value
		x.typ = &Pointer{base: x.typ}
		return

	case syntax.Recv:
		typ := x.typ.Chan()
		if typ == nil {
			check.invalidOpf(x, "cannot receive from non-channel %s", x)
			x.mode = invalid
			return
		}
		if typ.dir == SendOnly {
			check.invalidOpf(x, "cannot receive from send-only channel %s", x)
			x.mode = invalid
			return
		}
		x.mode = commaok
		x.typ = typ.elem
		check.hasCallOrRecv = true
		return
	}

	if !check.op(unaryOpPredicates, x, op) {
		x.mode = invalid
		return
	}

	if x.mode == constant_ {
		typ := x.typ.Basic()
		var prec uint
		if isUnsigned(typ) {
			prec = uint(check.conf.sizeof(typ) * 8)
		}
		x.val = constant.UnaryOp(op2token(op), x.val, prec)
		// Typed constants must be representable in
		// their type after each constant operation.
		if isTyped(typ) {
			if e != nil {
				x.expr = e // for better error message
			}
			check.representable(x, typ)
		}
		return
	}

	x.mode = value
	// x.typ remains unchanged
}

func isShift(op syntax.Operator) bool {
	return op == syntax.Shl || op == syntax.Shr
}

func isComparison(op syntax.Operator) bool {
	// Note: tokens are not ordered well to make this much easier
	switch op {
	case syntax.Eql, syntax.Neq, syntax.Lss, syntax.Leq, syntax.Gtr, syntax.Geq:
		return true
	}
	return false
}

func fitsFloat32(x constant.Value) bool {
	f32, _ := constant.Float32Val(x)
	f := float64(f32)
	return !math.IsInf(f, 0)
}

func roundFloat32(x constant.Value) constant.Value {
	f32, _ := constant.Float32Val(x)
	f := float64(f32)
	if !math.IsInf(f, 0) {
		return constant.MakeFloat64(f)
	}
	return nil
}

func fitsFloat64(x constant.Value) bool {
	f, _ := constant.Float64Val(x)
	return !math.IsInf(f, 0)
}

func roundFloat64(x constant.Value) constant.Value {
	f, _ := constant.Float64Val(x)
	if !math.IsInf(f, 0) {
		return constant.MakeFloat64(f)
	}
	return nil
}

// representableConst reports whether x can be represented as
// value of the given basic type and for the configuration
// provided (only needed for int/uint sizes).
//
// If rounded != nil, *rounded is set to the rounded value of x for
// representable floating-point and complex values, and to an Int
// value for integer values; it is left alone otherwise.
// It is ok to provide the addressof the first argument for rounded.
//
// The check parameter may be nil if representableConst is invoked
// (indirectly) through an exported API call (AssignableTo, ConvertibleTo)
// because we don't need the Checker's config for those calls.
func representableConst(x constant.Value, check *Checker, typ *Basic, rounded *constant.Value) bool {
	if x.Kind() == constant.Unknown {
		return true // avoid follow-up errors
	}

	var conf *Config
	if check != nil {
		conf = check.conf
	}

	switch {
	case isInteger(typ):
		x := constant.ToInt(x)
		if x.Kind() != constant.Int {
			return false
		}
		if rounded != nil {
			*rounded = x
		}
		if x, ok := constant.Int64Val(x); ok {
			switch typ.kind {
			case Int:
				var s = uint(conf.sizeof(typ)) * 8
				return int64(-1)<<(s-1) <= x && x <= int64(1)<<(s-1)-1
			case Int8:
				const s = 8
				return -1<<(s-1) <= x && x <= 1<<(s-1)-1
			case Int16:
				const s = 16
				return -1<<(s-1) <= x && x <= 1<<(s-1)-1
			case Int32:
				const s = 32
				return -1<<(s-1) <= x && x <= 1<<(s-1)-1
			case Int64, UntypedInt:
				return true
			case Uint, Uintptr:
				if s := uint(conf.sizeof(typ)) * 8; s < 64 {
					return 0 <= x && x <= int64(1)<<s-1
				}
				return 0 <= x
			case Uint8:
				const s = 8
				return 0 <= x && x <= 1<<s-1
			case Uint16:
				const s = 16
				return 0 <= x && x <= 1<<s-1
			case Uint32:
				const s = 32
				return 0 <= x && x <= 1<<s-1
			case Uint64:
				return 0 <= x
			default:
				unreachable()
			}
		}
		// x does not fit into int64
		switch n := constant.BitLen(x); typ.kind {
		case Uint, Uintptr:
			var s = uint(conf.sizeof(typ)) * 8
			return constant.Sign(x) >= 0 && n <= int(s)
		case Uint64:
			return constant.Sign(x) >= 0 && n <= 64
		case UntypedInt:
			return true
		}

	case isFloat(typ):
		x := constant.ToFloat(x)
		if x.Kind() != constant.Float {
			return false
		}
		switch typ.kind {
		case Float32:
			if rounded == nil {
				return fitsFloat32(x)
			}
			r := roundFloat32(x)
			if r != nil {
				*rounded = r
				return true
			}
		case Float64:
			if rounded == nil {
				return fitsFloat64(x)
			}
			r := roundFloat64(x)
			if r != nil {
				*rounded = r
				return true
			}
		case UntypedFloat:
			return true
		default:
			unreachable()
		}

	case isComplex(typ):
		x := constant.ToComplex(x)
		if x.Kind() != constant.Complex {
			return false
		}
		switch typ.kind {
		case Complex64:
			if rounded == nil {
				return fitsFloat32(constant.Real(x)) && fitsFloat32(constant.Imag(x))
			}
			re := roundFloat32(constant.Real(x))
			im := roundFloat32(constant.Imag(x))
			if re != nil && im != nil {
				*rounded = constant.BinaryOp(re, token.ADD, constant.MakeImag(im))
				return true
			}
		case Complex128:
			if rounded == nil {
				return fitsFloat64(constant.Real(x)) && fitsFloat64(constant.Imag(x))
			}
			re := roundFloat64(constant.Real(x))
			im := roundFloat64(constant.Imag(x))
			if re != nil && im != nil {
				*rounded = constant.BinaryOp(re, token.ADD, constant.MakeImag(im))
				return true
			}
		case UntypedComplex:
			return true
		default:
			unreachable()
		}

	case isString(typ):
		return x.Kind() == constant.String

	case isBoolean(typ):
		return x.Kind() == constant.Bool
	}

	return false
}

// representable checks that a constant operand is representable in the given basic type.
func (check *Checker) representable(x *operand, typ *Basic) {
	assert(x.mode == constant_)
	if !representableConst(x.val, check, typ, &x.val) {
		var msg string
		if isNumeric(x.typ) && isNumeric(typ) {
			// numeric conversion : error msg
			//
			// integer -> integer : overflows
			// integer -> float   : overflows (actually not possible)
			// float   -> integer : truncated
			// float   -> float   : overflows
			//
			if !isInteger(x.typ) && isInteger(typ) {
				msg = "%s truncated to %s"
			} else {
				msg = "%s overflows %s"
			}
		} else {
			msg = "cannot convert %s to %s"
		}
		check.errorf(x, msg, x, typ)
		x.mode = invalid
	}
}

// updateExprType updates the type of x to typ and invokes itself
// recursively for the operands of x, depending on expression kind.
// If typ is still an untyped and not the final type, updateExprType
// only updates the recorded untyped type for x and possibly its
// operands. Otherwise (i.e., typ is not an untyped type anymore,
// or it is the final type for x), the type and value are recorded.
// Also, if x is a constant, it must be representable as a value of typ,
// and if x is the (formerly untyped) lhs operand of a non-constant
// shift, it must be an integer value.
//
func (check *Checker) updateExprType(x syntax.Expr, typ Type, final bool) {
	old, found := check.untyped[x]
	if !found {
		return // nothing to do
	}

	// update operands of x if necessary
	switch x := x.(type) {
	case *syntax.BadExpr,
		*syntax.FuncLit,
		*syntax.CompositeLit,
		*syntax.IndexExpr,
		*syntax.SliceExpr,
		*syntax.AssertExpr,
		//*syntax.StarExpr,
		*syntax.KeyValueExpr,
		*syntax.ArrayType,
		*syntax.StructType,
		*syntax.FuncType,
		*syntax.InterfaceType,
		*syntax.MapType,
		*syntax.ChanType:
		// These expression are never untyped - nothing to do.
		// The respective sub-expressions got their final types
		// upon assignment or use.
		if debug {
			check.dump("%v: found old type(%s): %s (new: %s)", posFor(x), x, old.typ, typ)
			unreachable()
		}
		return

	case *syntax.CallExpr:
		// Resulting in an untyped constant (e.g., built-in complex).
		// The respective calls take care of calling updateExprType
		// for the arguments if necessary.

	case *syntax.Name, *syntax.BasicLit, *syntax.SelectorExpr:
		// An identifier denoting a constant, a constant literal,
		// or a qualified identifier (imported untyped constant).
		// No operands to take care of.

	case *syntax.ParenExpr:
		check.updateExprType(x.X, typ, final)

	// case *syntax.UnaryExpr:
	// 	// If x is a constant, the operands were constants.
	// 	// The operands don't need to be updated since they
	// 	// never get "materialized" into a typed value. If
	// 	// left in the untyped map, they will be processed
	// 	// at the end of the type check.
	// 	if old.val != nil {
	// 		break
	// 	}
	// 	check.updateExprType(x.X, typ, final)

	case *syntax.Operation:
		if x.Y == nil {
			// unary expression
			if x.Op == syntax.Mul {
				// see commented out code for StarExpr above
				// TODO(gri) needs cleanup
				if debug {
					unimplemented()
				}
				return
			}
			// If x is a constant, the operands were constants.
			// The operands don't need to be updated since they
			// never get "materialized" into a typed value. If
			// left in the untyped map, they will be processed
			// at the end of the type check.
			if old.val != nil {
				break
			}
			check.updateExprType(x.X, typ, final)
			break
		}

		// binary expression
		if old.val != nil {
			break // see comment for unary expressions
		}
		if isComparison(x.Op) {
			// The result type is independent of operand types
			// and the operand types must have final types.
		} else if isShift(x.Op) {
			// The result type depends only on lhs operand.
			// The rhs type was updated when checking the shift.
			check.updateExprType(x.X, typ, final)
		} else {
			// The operand types match the result type.
			check.updateExprType(x.X, typ, final)
			check.updateExprType(x.Y, typ, final)
		}

	default:
		unreachable()
	}

	// If the new type is not final and still untyped, just
	// update the recorded type.
	if !final && isUntyped(typ) {
		old.typ = typ.Basic()
		check.untyped[x] = old
		return
	}

	// Otherwise we have the final (typed or untyped type).
	// Remove it from the map of yet untyped expressions.
	delete(check.untyped, x)

	if old.isLhs {
		// If x is the lhs of a shift, its final type must be integer.
		// We already know from the shift check that it is representable
		// as an integer if it is a constant.
		if !isInteger(typ) {
			check.invalidOpf(x, "shifted operand %s (type %s) must be integer", x, typ)
			return
		}
		// Even if we have an integer, if the value is a constant we
		// still must check that it is representable as the specific
		// int type requested (was issue #22969). Fall through here.
	}
	if old.val != nil {
		// If x is a constant, it must be representable as a value of typ.
		c := operand{old.mode, x, old.typ, old.val, 0}
		check.convertUntyped(&c, typ)
		if c.mode == invalid {
			return
		}
	}

	// Everything's fine, record final type and value for x.
	check.recordTypeAndValue(x, old.mode, typ, old.val)
}

// updateExprVal updates the value of x to val.
func (check *Checker) updateExprVal(x syntax.Expr, val constant.Value) {
	if info, ok := check.untyped[x]; ok {
		info.val = val
		check.untyped[x] = info
	}
}

// convertUntyped attempts to set the type of an untyped value to the target type.
func (check *Checker) convertUntyped(x *operand, target Type) {
	target = expand(target)
	if x.mode == invalid || isTyped(x.typ) || target == Typ[Invalid] {
		return
	}

	// TODO(gri) Sloppy code - clean up. This function is central
	//           to assignment and expression checking.

	if isUntyped(target) {
		// both x and target are untyped
		xkind := x.typ.(*Basic).kind
		tkind := target.(*Basic).kind
		if isNumeric(x.typ) && isNumeric(target) {
			if xkind < tkind {
				x.typ = target
				check.updateExprType(x.expr, target, false)
			}
		} else if xkind != tkind {
			goto Error
		}
		return
	}

	// In case of a type parameter, conversion must succeed against
	// all types enumerated by the type parameter bound.
	// TODO(gri) We should not need this because we have the code
	// for Sum types in convertUntypedInternal. But at least one
	// test fails. Investigate.
	if t := target.TypeParam(); t != nil {
		types := t.Bound().allTypes
		if types == nil {
			goto Error
		}

		for _, t := range unpack(types) {
			check.convertUntypedInternal(x, t)
			if x.mode == invalid {
				goto Error
			}
		}

		// keep nil untyped (was bug #39755)
		if x.isNil() {
			target = Typ[UntypedNil]
		}
		x.typ = target
		check.updateExprType(x.expr, target, true) // UntypedNils are final
		return
	}

	check.convertUntypedInternal(x, target)
	return

Error:
	// TODO(gri) better error message (explain cause)
	check.errorf(x, "cannot convert %s to %s", x, target)
	x.mode = invalid
}

// convertUntypedInternal should only be called by convertUntyped.
func (check *Checker) convertUntypedInternal(x *operand, target Type) {
	assert(isTyped(target))

	// typed target
	switch t := optype(target.Under()).(type) {
	case *Basic:
		if x.mode == constant_ {
			check.representable(x, t)
			if x.mode == invalid {
				return
			}
			// expression value may have been rounded - update if needed
			check.updateExprVal(x.expr, x.val)
		} else {
			// Non-constant untyped values may appear as the
			// result of comparisons (untyped bool), intermediate
			// (delayed-checked) rhs operands of shifts, and as
			// the value nil.
			switch x.typ.(*Basic).kind {
			case UntypedBool:
				if !isBoolean(target) {
					goto Error
				}
			case UntypedInt, UntypedRune, UntypedFloat, UntypedComplex:
				if !isNumeric(target) {
					goto Error
				}
			case UntypedString:
				// Non-constant untyped string values are not
				// permitted by the spec and should not occur.
				unreachable()
			case UntypedNil:
				// Unsafe.Pointer is a basic type that includes nil.
				if !hasNil(target) {
					goto Error
				}
			default:
				goto Error
			}
		}
	case *Sum:
		t.is(func(t Type) bool {
			check.convertUntypedInternal(x, t)
			return x.mode != invalid
		})
	case *Interface:
		// Update operand types to the default type rather then
		// the target (interface) type: values must have concrete
		// dynamic types. If the value is nil, keep it untyped
		// (this is important for tools such as go vet which need
		// the dynamic type for argument checking of say, print
		// functions)
		if x.isNil() {
			target = Typ[UntypedNil]
		} else {
			// cannot assign untyped values to non-empty interfaces
			check.completeInterface(nopos, t)
			if !t.Empty() {
				goto Error
			}
			target = Default(x.typ)
		}
	case *Pointer, *Signature, *Slice, *Map, *Chan:
		if !x.isNil() {
			goto Error
		}
		// keep nil untyped - see comment for interfaces, above
		target = Typ[UntypedNil]
	default:
		goto Error
	}

	x.typ = target
	check.updateExprType(x.expr, target, true) // UntypedNils are final
	return

Error:
	check.errorf(x, "cannot convert %s to %s", x, target)
	x.mode = invalid
}

func (check *Checker) comparison(x, y *operand, op syntax.Operator) {
	// spec: "In any comparison, the first operand must be assignable
	// to the type of the second operand, or vice versa."
	err := ""
	if x.assignableTo(check, y.typ, nil) || y.assignableTo(check, x.typ, nil) {
		defined := false
		switch op {
		case syntax.Eql, syntax.Neq:
			// spec: "The equality operators == and != apply to operands that are comparable."
			defined = Comparable(x.typ) && Comparable(y.typ) || x.isNil() && hasNil(y.typ) || y.isNil() && hasNil(x.typ)
		case syntax.Lss, syntax.Leq, syntax.Gtr, syntax.Geq:
			// spec: The ordering operators <, <=, >, and >= apply to operands that are ordered."
			defined = isOrdered(x.typ) && isOrdered(y.typ)
		default:
			unreachable()
		}
		if !defined {
			typ := x.typ
			if x.isNil() {
				typ = y.typ
			}
			err = check.sprintf("operator %s not defined for %s", op, typ)
		}
	} else {
		err = check.sprintf("mismatched types %s and %s", x.typ, y.typ)
	}

	if err != "" {
		check.errorf(x, "cannot compare %s %s %s (%s)", x.expr, op, y.expr, err)
		x.mode = invalid
		return
	}

	if x.mode == constant_ && y.mode == constant_ {
		x.val = constant.MakeBool(constant.Compare(x.val, op2token(op), y.val))
		// The operands are never materialized; no need to update
		// their types.
	} else {
		x.mode = value
		// The operands have now their final types, which at run-
		// time will be materialized. Update the expression trees.
		// If the current types are untyped, the materialized type
		// is the respective default type.
		check.updateExprType(x.expr, Default(x.typ), true)
		check.updateExprType(y.expr, Default(y.typ), true)
	}

	// spec: "Comparison operators compare two operands and yield
	//        an untyped boolean value."
	x.typ = Typ[UntypedBool]
}

func (check *Checker) shift(x, y *operand, e *syntax.Operation, op syntax.Operator) {
	untypedx := isUntyped(x.typ)

	var xval constant.Value
	if x.mode == constant_ {
		xval = constant.ToInt(x.val)
	}

	if isInteger(x.typ) || untypedx && xval != nil && xval.Kind() == constant.Int {
		// The lhs is of integer type or an untyped constant representable
		// as an integer. Nothing to do.
	} else {
		// shift has no chance
		check.invalidOpf(x, "shifted operand %s must be integer", x)
		x.mode = invalid
		return
	}

	// spec: "The right operand in a shift expression must have integer type
	// or be an untyped constant representable by a value of type uint."
	switch {
	case isInteger(y.typ):
		// nothing to do
	case isUntyped(y.typ):
		check.convertUntyped(y, Typ[Uint])
		if y.mode == invalid {
			x.mode = invalid
			return
		}
	default:
		check.invalidOpf(y, "shift count %s must be integer", y)
		x.mode = invalid
		return
	}

	var yval constant.Value
	if y.mode == constant_ {
		// rhs must be an integer value
		// (Either it was of an integer type already, or it was
		// untyped and successfully converted to a uint above.)
		yval = constant.ToInt(y.val)
		assert(yval.Kind() == constant.Int)
		if constant.Sign(yval) < 0 {
			check.invalidOpf(y, "negative shift count %s", y)
			x.mode = invalid
			return
		}
	}

	if x.mode == constant_ {
		if y.mode == constant_ {
			// rhs must be within reasonable bounds in constant shifts
			const shiftBound = 1023 - 1 + 52 // so we can express smallestFloat64
			s, ok := constant.Uint64Val(yval)
			if !ok || s > shiftBound {
				check.invalidOpf(y, "invalid shift count %s", y)
				x.mode = invalid
				return
			}
			// The lhs is representable as an integer but may not be an integer
			// (e.g., 2.0, an untyped float) - this can only happen for untyped
			// non-integer numeric constants. Correct the type so that the shift
			// result is of integer type.
			if !isInteger(x.typ) {
				x.typ = Typ[UntypedInt]
			}
			// x is a constant so xval != nil and it must be of Int kind.
			x.val = constant.Shift(xval, op2token(op), uint(s))
			// Typed constants must be representable in
			// their type after each constant operation.
			if isTyped(x.typ) {
				if e != nil {
					x.expr = e // for better error message
				}
				check.representable(x, x.typ.Basic())
			}
			return
		}

		// non-constant shift with constant lhs
		if untypedx {
			// spec: "If the left operand of a non-constant shift
			// expression is an untyped constant, the type of the
			// constant is what it would be if the shift expression
			// were replaced by its left operand alone.".
			//
			// Delay operand checking until we know the final type
			// by marking the lhs expression as lhs shift operand.
			//
			// Usually (in correct programs), the lhs expression
			// is in the untyped map. However, it is possible to
			// create incorrect programs where the same expression
			// is evaluated twice (via a declaration cycle) such
			// that the lhs expression type is determined in the
			// first round and thus deleted from the map, and then
			// not found in the second round (double insertion of
			// the same expr node still just leads to one entry for
			// that node, and it can only be deleted once).
			// Be cautious and check for presence of entry.
			// Example: var e, f = int(1<<""[f]) // issue 11347
			if info, found := check.untyped[x.expr]; found {
				info.isLhs = true
				check.untyped[x.expr] = info
			}
			// keep x's type
			x.mode = value
			return
		}
	}

	// non-constant shift - lhs must be an integer
	if !isInteger(x.typ) {
		check.invalidOpf(x, "shifted operand %s must be integer", x)
		x.mode = invalid
		return
	}

	x.mode = value
}

var binaryOpPredicates = opPredicates{
	syntax.Add: func(typ Type) bool { return isNumeric(typ) || isString(typ) },
	syntax.Sub: isNumeric,
	syntax.Mul: isNumeric,
	syntax.Div: isNumeric,
	syntax.Rem: isInteger,

	syntax.And:    isInteger,
	syntax.Or:     isInteger,
	syntax.Xor:    isInteger,
	syntax.AndNot: isInteger,

	syntax.AndAnd: isBoolean,
	syntax.OrOr:   isBoolean,
}

// The binary expression e may be nil. It's passed in for better error messages only.
func (check *Checker) binary(x *operand, e *syntax.Operation, lhs, rhs syntax.Expr, op syntax.Operator) {
	var y operand

	check.expr(x, lhs)
	check.expr(&y, rhs)

	if x.mode == invalid {
		return
	}
	if y.mode == invalid {
		x.mode = invalid
		x.expr = y.expr
		return
	}

	if isShift(op) {
		check.shift(x, &y, e, op)
		return
	}

	check.convertUntyped(x, y.typ)
	if x.mode == invalid {
		return
	}
	check.convertUntyped(&y, x.typ)
	if y.mode == invalid {
		x.mode = invalid
		return
	}

	if isComparison(op) {
		check.comparison(x, &y, op)
		return
	}

	if !check.identical(x.typ, y.typ) {
		// only report an error if we have valid types
		// (otherwise we had an error reported elsewhere already)
		if x.typ != Typ[Invalid] && y.typ != Typ[Invalid] {
			check.invalidOpf(x, "mismatched types %s and %s", x.typ, y.typ)
		}
		x.mode = invalid
		return
	}

	if !check.op(binaryOpPredicates, x, op) {
		x.mode = invalid
		return
	}

	if op == syntax.Div || op == syntax.Rem {
		// check for zero divisor
		if (x.mode == constant_ || isInteger(x.typ)) && y.mode == constant_ && constant.Sign(y.val) == 0 {
			check.invalidOpf(&y, "division by zero")
			x.mode = invalid
			return
		}

		// check for divisor underflow in complex division (see issue 20227)
		if x.mode == constant_ && y.mode == constant_ && isComplex(x.typ) {
			re, im := constant.Real(y.val), constant.Imag(y.val)
			re2, im2 := constant.BinaryOp(re, token.MUL, re), constant.BinaryOp(im, token.MUL, im)
			if constant.Sign(re2) == 0 && constant.Sign(im2) == 0 {
				check.invalidOpf(&y, "division by zero")
				x.mode = invalid
				return
			}
		}
	}

	if x.mode == constant_ && y.mode == constant_ {
		xval := x.val
		yval := y.val
		typ := x.typ.Basic()
		// force integer division of integer operands
		tok := op2token(op)
		if op == syntax.Div && isInteger(typ) {
			tok = token.QUO_ASSIGN
		}
		x.val = constant.BinaryOp(xval, tok, yval)
		// Typed constants must be representable in
		// their type after each constant operation.
		if isTyped(typ) {
			if e != nil {
				x.expr = e // for better error message
			}
			check.representable(x, typ)
		}
		return
	}

	x.mode = value
	// x.typ is unchanged
}

// index checks an index expression for validity.
// If max >= 0, it is the upper bound for index.
// If the result typ is != Typ[Invalid], index is valid and typ is its (possibly named) integer type.
// If the result val >= 0, index is valid and val is its constant int value.
func (check *Checker) index(index syntax.Expr, max int64) (typ Type, val int64) {
	typ = Typ[Invalid]
	val = -1

	var x operand
	check.expr(&x, index)
	if x.mode == invalid {
		return
	}

	// an untyped constant must be representable as Int
	check.convertUntyped(&x, Typ[Int])
	if x.mode == invalid {
		return
	}

	// the index must be of integer type
	if !isInteger(x.typ) {
		check.invalidArgf(&x, "index %s must be integer", &x)
		return
	}

	if x.mode != constant_ {
		return x.typ, -1
	}

	// a constant index i must be in bounds
	if constant.Sign(x.val) < 0 {
		check.invalidArgf(&x, "index %s must not be negative", &x)
		return
	}

	v, valid := constant.Int64Val(constant.ToInt(x.val))
	if !valid || max >= 0 && v >= max {
		check.errorf(&x, "index %s is out of bounds", &x)
		return
	}

	// 0 <= v [ && v < max ]
	return Typ[Int], v
}

// indexElts checks the elements (elts) of an array or slice composite literal
// against the literal's element type (typ), and the element indices against
// the literal length if known (length >= 0). It returns the length of the
// literal (maximum index value + 1).
//
func (check *Checker) indexedElts(elts []syntax.Expr, typ Type, length int64) int64 {
	visited := make(map[int64]bool, len(elts))
	var index, max int64
	for _, e := range elts {
		// determine and check index
		validIndex := false
		eval := e
		if kv, _ := e.(*syntax.KeyValueExpr); kv != nil {
			if typ, i := check.index(kv.Key, length); typ != Typ[Invalid] {
				if i >= 0 {
					index = i
					validIndex = true
				} else {
					check.errorf(e, "index %s must be integer constant", kv.Key)
				}
			}
			eval = kv.Value
		} else if length >= 0 && index >= length {
			check.errorf(e, "index %d is out of bounds (>= %d)", index, length)
		} else {
			validIndex = true
		}

		// if we have a valid index, check for duplicate entries
		if validIndex {
			if visited[index] {
				check.errorf(e, "duplicate index %d in array or slice literal", index)
			}
			visited[index] = true
		}
		index++
		if index > max {
			max = index
		}

		// check element against composite literal element type
		var x operand
		check.exprWithHint(&x, eval, typ)
		check.assignment(&x, typ, "array or slice literal")
	}
	return max
}

// exprKind describes the kind of an expression; the kind
// determines if an expression is valid in 'statement context'.
type exprKind int

const (
	conversion exprKind = iota
	expression
	statement
)

// rawExpr typechecks expression e and initializes x with the expression
// value or type. If an error occurred, x.mode is set to invalid.
// If hint != nil, it is the type of a composite literal element.
//
func (check *Checker) rawExpr(x *operand, e syntax.Expr, hint Type) exprKind {
	if check.conf.Trace {
		check.trace(e.Pos(), "expr %s", e)
		check.indent++
		defer func() {
			check.indent--
			check.trace(e.Pos(), "=> %s", x)
		}()
	}

	kind := check.exprInternal(x, e, hint)

	// convert x into a user-friendly set of values
	// TODO(gri) this code can be simplified
	var typ Type
	var val constant.Value
	switch x.mode {
	case invalid:
		typ = Typ[Invalid]
	case novalue:
		typ = (*Tuple)(nil)
	case constant_:
		typ = x.typ
		val = x.val
	default:
		typ = x.typ
	}
	assert(x.expr != nil && typ != nil)

	if isUntyped(typ) {
		// delay type and value recording until we know the type
		// or until the end of type checking
		check.rememberUntyped(x.expr, false, x.mode, typ.(*Basic), val)
	} else {
		check.recordTypeAndValue(e, x.mode, typ, val)
	}

	return kind
}

// exprInternal contains the core of type checking of expressions.
// Must only be called by rawExpr.
//
func (check *Checker) exprInternal(x *operand, e syntax.Expr, hint Type) exprKind {
	// make sure x has a valid state in case of bailout
	// (was issue 5770)
	x.mode = invalid
	x.typ = Typ[Invalid]

	switch e := e.(type) {
	case nil:
		unreachable()

	case *syntax.BadExpr:
		goto Error // error was reported before

	case *syntax.Name:
		check.ident(x, e, nil, false)

	case *syntax.DotsType:
		// ellipses are handled explicitly where they are legal
		// (array composite literals and parameter lists)
		check.error(e, "invalid use of '...'")
		goto Error

	case *syntax.BasicLit:
		x.setConst(e.Kind, e.Value)
		if x.mode == invalid {
			check.invalidASTf(e, "invalid literal %v", e.Value)
			goto Error
		}

	case *syntax.FuncLit:
		if sig, ok := check.typ(e.Type).(*Signature); ok {
			// Anonymous functions are considered part of the
			// init expression/func declaration which contains
			// them: use existing package-level declaration info.
			decl := check.decl // capture for use in closure below
			iota := check.iota // capture for use in closure below (#22345)
			// Don't type-check right away because the function may
			// be part of a type definition to which the function
			// body refers. Instead, type-check as soon as possible,
			// but before the enclosing scope contents changes (#22992).
			check.later(func() {
				check.funcBody(decl, "<function literal>", sig, e.Body, iota)
			})
			x.mode = value
			x.typ = sig
		} else {
			check.invalidASTf(e, "invalid function literal %s", e)
			goto Error
		}

	case *syntax.CompositeLit:
		var typ, base Type

		switch {
		case e.Type != nil:
			// composite literal type present - use it
			// [...]T array types may only appear with composite literals.
			// Check for them here so we don't have to handle ... in general.
			if atyp, _ := e.Type.(*syntax.ArrayType); atyp != nil && atyp.Len == nil {
				// We have an "open" [...]T array type.
				// Create a new ArrayType with unknown length (-1)
				// and finish setting it up after analyzing the literal.
				typ = &Array{len: -1, elem: check.varType(atyp.Elem)}
				base = typ
				break
			}
			typ = check.typ(e.Type)
			base = typ

		case hint != nil:
			// no composite literal type present - use hint (element type of enclosing type)
			typ = hint
			base, _ = deref(typ.Under()) // *T implies &T{}

		default:
			// TODO(gri) provide better error messages depending on context
			check.error(e, "missing type in composite literal")
			goto Error
		}

		switch utyp := optype(base.Under()).(type) {
		case *Struct:
			if len(e.ElemList) == 0 {
				break
			}
			fields := utyp.fields
			if _, ok := e.ElemList[0].(*syntax.KeyValueExpr); ok {
				// all elements must have keys
				visited := make([]bool, len(fields))
				for _, e := range e.ElemList {
					kv, _ := e.(*syntax.KeyValueExpr)
					if kv == nil {
						check.error(e, "mixture of field:value and value elements in struct literal")
						continue
					}
					key, _ := kv.Key.(*syntax.Name)
					// do all possible checks early (before exiting due to errors)
					// so we don't drop information on the floor
					check.expr(x, kv.Value)
					if key == nil {
						check.errorf(kv, "invalid field name %s in struct literal", kv.Key)
						continue
					}
					i := fieldIndex(utyp.fields, check.pkg, key.Value)
					if i < 0 {
						check.errorf(kv, "unknown field %s in struct literal", key.Value)
						continue
					}
					fld := fields[i]
					check.recordUse(key, fld)
					etyp := fld.typ
					check.assignment(x, etyp, "struct literal")
					// 0 <= i < len(fields)
					if visited[i] {
						check.errorf(kv, "duplicate field name %s in struct literal", key.Value)
						continue
					}
					visited[i] = true
				}
			} else {
				// no element must have a key
				for i, e := range e.ElemList {
					if kv, _ := e.(*syntax.KeyValueExpr); kv != nil {
						check.error(kv, "mixture of field:value and value elements in struct literal")
						continue
					}
					check.expr(x, e)
					if i >= len(fields) {
						check.error(x, "too many values in struct literal")
						break // cannot continue
					}
					// i < len(fields)
					fld := fields[i]
					if !fld.Exported() && fld.pkg != check.pkg {
						check.errorf(x, "implicit assignment to unexported field %s in %s literal", fld.name, typ)
						continue
					}
					etyp := fld.typ
					check.assignment(x, etyp, "struct literal")
				}
				if len(e.ElemList) < len(fields) {
					check.error(e.Rbrace, "too few values in struct literal")
					// ok to continue
				}
			}

		case *Array:
			// Prevent crash if the array referred to is not yet set up. Was issue #18643.
			// This is a stop-gap solution. Should use Checker.objPath to report entire
			// path starting with earliest declaration in the source. TODO(gri) fix this.
			if utyp.elem == nil {
				check.error(e, "illegal cycle in type declaration")
				goto Error
			}
			n := check.indexedElts(e.ElemList, utyp.elem, utyp.len)
			// If we have an array of unknown length (usually [...]T arrays, but also
			// arrays [n]T where n is invalid) set the length now that we know it and
			// record the type for the array (usually done by check.typ which is not
			// called for [...]T). We handle [...]T arrays and arrays with invalid
			// length the same here because it makes sense to "guess" the length for
			// the latter if we have a composite literal; e.g. for [n]int{1, 2, 3}
			// where n is invalid for some reason, it seems fair to assume it should
			// be 3 (see also Checked.arrayLength and issue #27346).
			if utyp.len < 0 {
				utyp.len = n
				// e.Type is missing if we have a composite literal element
				// that is itself a composite literal with omitted type. In
				// that case there is nothing to record (there is no type in
				// the source at that point).
				if e.Type != nil {
					check.recordTypeAndValue(e.Type, typexpr, utyp, nil)
				}
			}

		case *Slice:
			// Prevent crash if the slice referred to is not yet set up.
			// See analogous comment for *Array.
			if utyp.elem == nil {
				check.error(e, "illegal cycle in type declaration")
				goto Error
			}
			check.indexedElts(e.ElemList, utyp.elem, -1)

		case *Map:
			// Prevent crash if the map referred to is not yet set up.
			// See analogous comment for *Array.
			if utyp.key == nil || utyp.elem == nil {
				check.error(e, "illegal cycle in type declaration")
				goto Error
			}
			visited := make(map[interface{}][]Type, len(e.ElemList))
			for _, e := range e.ElemList {
				kv, _ := e.(*syntax.KeyValueExpr)
				if kv == nil {
					check.error(e, "missing key in map literal")
					continue
				}
				check.exprWithHint(x, kv.Key, utyp.key)
				check.assignment(x, utyp.key, "map literal")
				if x.mode == invalid {
					continue
				}
				if x.mode == constant_ {
					duplicate := false
					// if the key is of interface type, the type is also significant when checking for duplicates
					xkey := keyVal(x.val)
					if utyp.key.Interface() != nil {
						for _, vtyp := range visited[xkey] {
							if check.identical(vtyp, x.typ) {
								duplicate = true
								break
							}
						}
						visited[xkey] = append(visited[xkey], x.typ)
					} else {
						_, duplicate = visited[xkey]
						visited[xkey] = nil
					}
					if duplicate {
						check.errorf(x, "duplicate key %s in map literal", x.val)
						continue
					}
				}
				check.exprWithHint(x, kv.Value, utyp.elem)
				check.assignment(x, utyp.elem, "map literal")
			}

		default:
			// when "using" all elements unpack KeyValueExpr
			// explicitly because check.use doesn't accept them
			for _, e := range e.ElemList {
				if kv, _ := e.(*syntax.KeyValueExpr); kv != nil {
					// Ideally, we should also "use" kv.Key but we can't know
					// if it's an externally defined struct key or not. Going
					// forward anyway can lead to other errors. Give up instead.
					e = kv.Value
				}
				check.use(e)
			}
			// if utyp is invalid, an error was reported before
			if utyp != Typ[Invalid] {
				check.errorf(e, "invalid composite literal type %s", typ)
				goto Error
			}
		}

		x.mode = value
		x.typ = typ

	case *syntax.ParenExpr:
		kind := check.rawExpr(x, e.X, nil)
		x.expr = e
		return kind

	case *syntax.SelectorExpr:
		check.selector(x, e)

	case *syntax.IndexExpr:
		check.exprOrType(x, e.X)
		if x.mode == invalid {
			check.use(e.Index)
			goto Error
		}

		if x.mode == typexpr {
			if isGeneric(x.typ) {
				// type instantiation
				x.mode = invalid
				x.typ = check.varType(e)
				if x.typ != Typ[Invalid] {
					x.mode = typexpr
				}
				return expression
			}
			check.errorf(x, "%s is not a generic type", x.typ)
			goto Error
		}

		if sig := x.typ.Signature(); sig != nil {
			return check.call(x, nil, e)
		}

		valid := false
		length := int64(-1) // valid if >= 0
		switch typ := optype(x.typ.Under()).(type) {
		case *Basic:
			if isString(typ) {
				valid = true
				if x.mode == constant_ {
					length = int64(len(constant.StringVal(x.val)))
				}
				// an indexed string always yields a byte value
				// (not a constant) even if the string and the
				// index are constant
				x.mode = value
				x.typ = universeByte // use 'byte' name
			}

		case *Array:
			valid = true
			length = typ.len
			if x.mode != variable {
				x.mode = value
			}
			x.typ = typ.elem

		case *Pointer:
			if typ := typ.base.Array(); typ != nil {
				valid = true
				length = typ.len
				x.mode = variable
				x.typ = typ.elem
			}

		case *Slice:
			valid = true
			x.mode = variable
			x.typ = typ.elem

		case *Map:
			var key operand
			check.expr(&key, e.Index)
			check.assignment(&key, typ.key, "map index")
			if x.mode == invalid {
				goto Error
			}
			x.mode = mapindex
			x.typ = typ.elem
			x.expr = e
			return expression

		case *Sum:
			// A sum type can be indexed if all the sum's types
			// support indexing and have the same element type.
			var elem Type
			if typ.is(func(t Type) bool {
				var e Type
				switch t := t.Under().(type) {
				case *Basic:
					if isString(t) {
						e = universeByte
					}
				case *Array:
					e = t.elem
				case *Pointer:
					if t := t.base.Array(); t != nil {
						e = t.elem
					}
				case *Slice:
					e = t.elem
				case *Map:
					e = t.elem
				case *TypeParam:
					check.errorf(x, "type of %s contains a type parameter - cannot index (implementation restriction)", x)
				case *instance:
					unimplemented()
				}
				if e != nil && (e == elem || elem == nil) {
					elem = e
					return true
				}
				return false
			}) {
				valid = true
				x.mode = variable
				x.typ = elem
			}
		}

		if !valid {
			check.invalidOpf(x, "cannot index %s", x)
			goto Error
		}

		if e.Index == nil {
			check.invalidASTf(e, "missing index for %s", x)
			goto Error
		}

		// In pathological (invalid) cases (e.g.: type T1 [][[]T1{}[0][0]]T0)
		// the element type may be accessed before it's set. Make sure we have
		// a valid type.
		if x.typ == nil {
			x.typ = Typ[Invalid]
		}

		check.index(e.Index, length)
		// ok to continue

	case *syntax.SliceExpr:
		check.expr(x, e.X)
		if x.mode == invalid {
			check.use(e.Index[:]...)
			goto Error
		}

		valid := false
		length := int64(-1) // valid if >= 0
		switch typ := optype(x.typ.Under()).(type) {
		case *Basic:
			if isString(typ) {
				if e.Full {
					check.invalidOpf(x, "3-index slice of string")
					goto Error
				}
				valid = true
				if x.mode == constant_ {
					length = int64(len(constant.StringVal(x.val)))
				}
				// spec: "For untyped string operands the result
				// is a non-constant value of type string."
				if typ.kind == UntypedString {
					x.typ = Typ[String]
				}
			}

		case *Array:
			valid = true
			length = typ.len
			if x.mode != variable {
				check.invalidOpf(x, "cannot slice %s (value not addressable)", x)
				goto Error
			}
			x.typ = &Slice{elem: typ.elem}

		case *Pointer:
			if typ := typ.base.Array(); typ != nil {
				valid = true
				length = typ.len
				x.typ = &Slice{elem: typ.elem}
			}

		case *Slice:
			valid = true
			// x.typ doesn't change

		case *Sum, *TypeParam:
			check.errorf(x, "generic slice expressions not yet implemented")
			goto Error
		}

		if !valid {
			check.invalidOpf(x, "cannot slice %s", x)
			goto Error
		}

		x.mode = value

		// spec: "Only the first index may be omitted; it defaults to 0."
		if e.Full && (e.Index[1] == nil || e.Index[2] == nil) {
			check.error(e, "2nd and 3rd index required in 3-index slice")
			goto Error
		}

		// check indices
		var ind [3]int64
		for i, expr := range e.Index {
			x := int64(-1)
			switch {
			case expr != nil:
				// The "capacity" is only known statically for strings, arrays,
				// and pointers to arrays, and it is the same as the length for
				// those types.
				max := int64(-1)
				if length >= 0 {
					max = length + 1
				}
				if _, v := check.index(expr, max); v >= 0 {
					x = v
				}
			case i == 0:
				// default is 0 for the first index
				x = 0
			case length >= 0:
				// default is length (== capacity) otherwise
				x = length
			}
			ind[i] = x
		}

		// constant indices must be in range
		// (check.index already checks that existing indices >= 0)
	L:
		for i, x := range ind[:len(ind)-1] {
			if x > 0 {
				for _, y := range ind[i+1:] {
					if y >= 0 && x > y {
						check.errorf(e, "invalid slice indices: %d > %d", x, y)
						break L // only report one error, ok to continue
					}
				}
			}
		}

	case *syntax.AssertExpr:
		check.expr(x, e.X)
		if x.mode == invalid {
			goto Error
		}
		var xtyp *Interface
		var strict bool
		switch t := optype(x.typ.Under()).(type) {
		case *Interface:
			xtyp = t
		// Disabled for now. It is not clear what the right approach is
		// here. Also, the implementation below is inconsistent because
		// the underlying type of a type parameter is either itself or
		// a sum type if the corresponding type bound contains a type list.
		// case *TypeParam:
		// 	xtyp = t.Bound()
		// 	strict = true
		default:
			check.invalidOpf(x, "%s is not an interface type", x)
			goto Error
		}
		// x.(type) expressions are encoded via TypeSwitchGuards
		if e.Type == nil {
			check.invalidASTf(e, "invalid use of AssertExpr")
			goto Error
		}
		T := check.varType(e.Type)
		if T == Typ[Invalid] {
			goto Error
		}
		check.typeAssertion(posFor(x), x, xtyp, T, strict)
		x.mode = commaok
		x.typ = T

	case *syntax.TypeSwitchGuard:
		// x.(type) expressions are handled explicitly in type switches
		check.invalidASTf(e, "use of .(type) outside type switch")
		goto Error

	case *syntax.CallExpr:
		return check.call(x, e, e)

	// case *syntax.UnaryExpr:
	// 	check.expr(x, e.X)
	// 	if x.mode == invalid {
	// 		goto Error
	// 	}
	// 	check.unary(x, e, e.Op)
	// 	if x.mode == invalid {
	// 		goto Error
	// 	}
	// 	if e.Op == token.ARROW {
	// 		x.expr = e
	// 		return statement // receive operations may appear in statement context
	// 	}

	// case *syntax.BinaryExpr:
	// 	check.binary(x, e, e.X, e.Y, e.Op)
	// 	if x.mode == invalid {
	// 		goto Error
	// 	}

	case *syntax.Operation:
		if e.Y == nil {
			// unary expression
			if e.Op == syntax.Mul {
				// pointer indirection
				check.exprOrType(x, e.X)
				switch x.mode {
				case invalid:
					goto Error
				case typexpr:
					x.typ = &Pointer{base: x.typ}
				default:
					if typ := x.typ.Pointer(); typ != nil {
						x.mode = variable
						x.typ = typ.base
					} else {
						check.invalidOpf(x, "cannot indirect %s", x)
						goto Error
					}
				}
				break
			}

			check.expr(x, e.X)
			if x.mode == invalid {
				goto Error
			}
			check.unary(x, e, e.Op)
			if x.mode == invalid {
				goto Error
			}
			if e.Op == syntax.Recv {
				x.expr = e
				return statement // receive operations may appear in statement context
			}
			break
		}

		// binary expression
		check.binary(x, e, e.X, e.Y, e.Op)
		if x.mode == invalid {
			goto Error
		}

	case *syntax.KeyValueExpr:
		// key:value expressions are handled in composite literals
		check.invalidASTf(e, "no key:value expected")
		goto Error

	case *syntax.ArrayType, *syntax.SliceType, *syntax.StructType, *syntax.FuncType,
		*syntax.InterfaceType, *syntax.MapType, *syntax.ChanType:
		x.mode = typexpr
		x.typ = check.typ(e)
		// Note: rawExpr (caller of exprInternal) will call check.recordTypeAndValue
		// even though check.typ has already called it. This is fine as both
		// times the same expression and type are recorded. It is also not a
		// performance issue because we only reach here for composite literal
		// types, which are comparatively rare.

	default:
		panic(fmt.Sprintf("%s: unknown expression type %T", posFor(e), e))
	}

	// everything went well
	x.expr = e
	return expression

Error:
	x.mode = invalid
	x.expr = e
	return statement // avoid follow-up errors
}

func keyVal(x constant.Value) interface{} {
	switch x.Kind() {
	case constant.Bool:
		return constant.BoolVal(x)
	case constant.String:
		return constant.StringVal(x)
	case constant.Int:
		if v, ok := constant.Int64Val(x); ok {
			return v
		}
		if v, ok := constant.Uint64Val(x); ok {
			return v
		}
	case constant.Float:
		v, _ := constant.Float64Val(x)
		return v
	case constant.Complex:
		r, _ := constant.Float64Val(constant.Real(x))
		i, _ := constant.Float64Val(constant.Imag(x))
		return complex(r, i)
	}
	return x
}

// typeAssertion checks that x.(T) is legal; xtyp must be the type of x.
func (check *Checker) typeAssertion(pos syntax.Pos, x *operand, xtyp *Interface, T Type, strict bool) {
	method, wrongType := check.assertableTo(xtyp, T, strict)
	if method == nil {
		return
	}
	var msg string
	if wrongType != nil {
		if check.identical(method.typ, wrongType.typ) {
			msg = fmt.Sprintf("missing method %s (%s has pointer receiver)", method.name, method.name)
		} else {
			msg = fmt.Sprintf("wrong type for method %s (have %s, want %s)", method.name, wrongType.typ, method.typ)
		}
	} else {
		msg = "missing method " + method.name
	}
	check.errorf(pos, "%s cannot have dynamic type %s (%s)", x, T, msg)
}

// expr typechecks expression e and initializes x with the expression value.
// The result must be a single value.
// If an error occurred, x.mode is set to invalid.
//
func (check *Checker) expr(x *operand, e syntax.Expr) {
	check.rawExpr(x, e, nil)
	check.exclude(x, 1<<novalue|1<<builtin|1<<typexpr)
	check.singleValue(x)
}

// multiExpr is like expr but the result may also be a multi-value.
func (check *Checker) multiExpr(x *operand, e syntax.Expr) {
	check.rawExpr(x, e, nil)
	check.exclude(x, 1<<novalue|1<<builtin|1<<typexpr)
}

// multiExprOrType is like multiExpr but the result may also be a type.
func (check *Checker) multiExprOrType(x *operand, e syntax.Expr) {
	check.rawExpr(x, e, nil)
	check.exclude(x, 1<<novalue|1<<builtin)
}

// exprWithHint typechecks expression e and initializes x with the expression value;
// hint is the type of a composite literal element.
// If an error occurred, x.mode is set to invalid.
//
func (check *Checker) exprWithHint(x *operand, e syntax.Expr, hint Type) {
	assert(hint != nil)
	check.rawExpr(x, e, hint)
	check.exclude(x, 1<<novalue|1<<builtin|1<<typexpr)
	check.singleValue(x)
}

// exprOrType typechecks expression or type e and initializes x with the expression value or type.
// If an error occurred, x.mode is set to invalid.
//
func (check *Checker) exprOrType(x *operand, e syntax.Expr) {
	check.rawExpr(x, e, nil)
	check.exclude(x, 1<<novalue)
	check.singleValue(x)
}

// exclude reports an error if x.mode is in modeset and sets x.mode to invalid.
// The modeset may contain any of 1<<novalue, 1<<builtin, 1<<typexpr.
func (check *Checker) exclude(x *operand, modeset uint) {
	if modeset&(1<<x.mode) != 0 {
		var msg string
		switch x.mode {
		case novalue:
			if modeset&(1<<typexpr) != 0 {
				msg = "%s used as value"
			} else {
				msg = "%s used as value or type"
			}
		case builtin:
			msg = "%s must be called"
		case typexpr:
			msg = "%s is not an expression"
		default:
			unreachable()
		}
		check.errorf(x, msg, x)
		x.mode = invalid
	}
}

// singleValue reports an error if x describes a tuple and sets x.mode to invalid.
func (check *Checker) singleValue(x *operand) {
	if x.mode == value {
		// tuple types are never named - no need for underlying type below
		if t, ok := x.typ.(*Tuple); ok {
			assert(t.Len() != 1)
			check.errorf(x, "%d-valued %s where single value is expected", t.Len(), x)
			x.mode = invalid
		}
	}
}
