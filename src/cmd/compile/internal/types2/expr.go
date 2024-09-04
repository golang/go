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
	. "internal/types/errors"
	"strings"
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

var unaryOpPredicates opPredicates

func init() {
	// Setting unaryOpPredicates in init avoids declaration cycles.
	unaryOpPredicates = opPredicates{
		syntax.Add: allNumeric,
		syntax.Sub: allNumeric,
		syntax.Xor: allInteger,
		syntax.Not: allBoolean,
	}
}

func (check *Checker) op(m opPredicates, x *operand, op syntax.Operator) bool {
	if pred := m[op]; pred != nil {
		if !pred(x.typ) {
			check.errorf(x, UndefinedOp, invalidOp+"operator %s not defined on %s", op, x)
			return false
		}
	} else {
		check.errorf(x, InvalidSyntaxTree, "unknown operator %s", op)
		return false
	}
	return true
}

// opPos returns the position of the operator if x is an operation;
// otherwise it returns the start position of x.
func opPos(x syntax.Expr) syntax.Pos {
	switch op := x.(type) {
	case nil:
		return nopos // don't crash
	case *syntax.Operation:
		return op.Pos()
	default:
		return syntax.StartPos(x)
	}
}

// opName returns the name of the operation if x is an operation
// that might overflow; otherwise it returns the empty string.
func opName(x syntax.Expr) string {
	if e, _ := x.(*syntax.Operation); e != nil {
		op := int(e.Op)
		if e.Y == nil {
			if op < len(op2str1) {
				return op2str1[op]
			}
		} else {
			if op < len(op2str2) {
				return op2str2[op]
			}
		}
	}
	return ""
}

var op2str1 = [...]string{
	syntax.Xor: "bitwise complement",
}

// This is only used for operations that may cause overflow.
var op2str2 = [...]string{
	syntax.Add: "addition",
	syntax.Sub: "subtraction",
	syntax.Xor: "bitwise XOR",
	syntax.Mul: "multiplication",
	syntax.Shl: "shift",
}

// If typ is a type parameter, underIs returns the result of typ.underIs(f).
// Otherwise, underIs returns the result of f(under(typ)).
func underIs(typ Type, f func(Type) bool) bool {
	typ = Unalias(typ)
	if tpar, _ := typ.(*TypeParam); tpar != nil {
		return tpar.underIs(f)
	}
	return f(under(typ))
}

func (check *Checker) unary(x *operand, e *syntax.Operation) {
	check.expr(nil, x, e.X)
	if x.mode == invalid {
		return
	}

	op := e.Op
	switch op {
	case syntax.And:
		// spec: "As an exception to the addressability
		// requirement x may also be a composite literal."
		if _, ok := syntax.Unparen(e.X).(*syntax.CompositeLit); !ok && x.mode != variable {
			check.errorf(x, UnaddressableOperand, invalidOp+"cannot take address of %s", x)
			x.mode = invalid
			return
		}
		x.mode = value
		x.typ = &Pointer{base: x.typ}
		return

	case syntax.Recv:
		u := coreType(x.typ)
		if u == nil {
			check.errorf(x, InvalidReceive, invalidOp+"cannot receive from %s (no core type)", x)
			x.mode = invalid
			return
		}
		ch, _ := u.(*Chan)
		if ch == nil {
			check.errorf(x, InvalidReceive, invalidOp+"cannot receive from non-channel %s", x)
			x.mode = invalid
			return
		}
		if ch.dir == SendOnly {
			check.errorf(x, InvalidReceive, invalidOp+"cannot receive from send-only channel %s", x)
			x.mode = invalid
			return
		}
		x.mode = commaok
		x.typ = ch.elem
		check.hasCallOrRecv = true
		return

	case syntax.Tilde:
		// Provide a better error position and message than what check.op below would do.
		if !allInteger(x.typ) {
			check.error(e, UndefinedOp, "cannot use ~ outside of interface or type constraint")
			x.mode = invalid
			return
		}
		check.error(e, UndefinedOp, "cannot use ~ outside of interface or type constraint (use ^ for bitwise complement)")
		op = syntax.Xor
	}

	if !check.op(unaryOpPredicates, x, op) {
		x.mode = invalid
		return
	}

	if x.mode == constant_ {
		if x.val.Kind() == constant.Unknown {
			// nothing to do (and don't cause an error below in the overflow check)
			return
		}
		var prec uint
		if isUnsigned(x.typ) {
			prec = uint(check.conf.sizeof(x.typ) * 8)
		}
		x.val = constant.UnaryOp(op2tok[op], x.val, prec)
		x.expr = e
		check.overflow(x, opPos(x.expr))
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

// updateExprType updates the type of x to typ and invokes itself
// recursively for the operands of x, depending on expression kind.
// If typ is still an untyped and not the final type, updateExprType
// only updates the recorded untyped type for x and possibly its
// operands. Otherwise (i.e., typ is not an untyped type anymore,
// or it is the final type for x), the type and value are recorded.
// Also, if x is a constant, it must be representable as a value of typ,
// and if x is the (formerly untyped) lhs operand of a non-constant
// shift, it must be an integer value.
func (check *Checker) updateExprType(x syntax.Expr, typ Type, final bool) {
	check.updateExprType0(nil, x, typ, final)
}

func (check *Checker) updateExprType0(parent, x syntax.Expr, typ Type, final bool) {
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
		*syntax.ListExpr,
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
			check.dump("%v: found old type(%s): %s (new: %s)", atPos(x), x, old.typ, typ)
			panic("unreachable")
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
		check.updateExprType0(x, x.X, typ, final)

	// case *syntax.UnaryExpr:
	// 	// If x is a constant, the operands were constants.
	// 	// The operands don't need to be updated since they
	// 	// never get "materialized" into a typed value. If
	// 	// left in the untyped map, they will be processed
	// 	// at the end of the type check.
	// 	if old.val != nil {
	// 		break
	// 	}
	// 	check.updateExprType0(x, x.X, typ, final)

	case *syntax.Operation:
		if x.Y == nil {
			// unary expression
			if x.Op == syntax.Mul {
				// see commented out code for StarExpr above
				// TODO(gri) needs cleanup
				if debug {
					panic("unimplemented")
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
			check.updateExprType0(x, x.X, typ, final)
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
			check.updateExprType0(x, x.X, typ, final)
		} else {
			// The operand types match the result type.
			check.updateExprType0(x, x.X, typ, final)
			check.updateExprType0(x, x.Y, typ, final)
		}

	default:
		panic("unreachable")
	}

	// If the new type is not final and still untyped, just
	// update the recorded type.
	if !final && isUntyped(typ) {
		old.typ = under(typ).(*Basic)
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
		if !allInteger(typ) {
			check.errorf(x, InvalidShiftOperand, invalidOp+"shifted operand %s (type %s) must be integer", x, typ)
			return
		}
		// Even if we have an integer, if the value is a constant we
		// still must check that it is representable as the specific
		// int type requested (was go.dev/issue/22969). Fall through here.
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

// implicitTypeAndValue returns the implicit type of x when used in a context
// where the target type is expected. If no such implicit conversion is
// possible, it returns a nil Type and non-zero error code.
//
// If x is a constant operand, the returned constant.Value will be the
// representation of x in this context.
func (check *Checker) implicitTypeAndValue(x *operand, target Type) (Type, constant.Value, Code) {
	if x.mode == invalid || isTyped(x.typ) || !isValid(target) {
		return x.typ, nil, 0
	}
	// x is untyped

	if isUntyped(target) {
		// both x and target are untyped
		if m := maxType(x.typ, target); m != nil {
			return m, nil, 0
		}
		return nil, nil, InvalidUntypedConversion
	}

	if x.isNil() {
		assert(isUntyped(x.typ))
		if hasNil(target) {
			return target, nil, 0
		}
		return nil, nil, InvalidUntypedConversion
	}

	switch u := under(target).(type) {
	case *Basic:
		if x.mode == constant_ {
			v, code := check.representation(x, u)
			if code != 0 {
				return nil, nil, code
			}
			return target, v, code
		}
		// Non-constant untyped values may appear as the
		// result of comparisons (untyped bool), intermediate
		// (delayed-checked) rhs operands of shifts, and as
		// the value nil.
		switch x.typ.(*Basic).kind {
		case UntypedBool:
			if !isBoolean(target) {
				return nil, nil, InvalidUntypedConversion
			}
		case UntypedInt, UntypedRune, UntypedFloat, UntypedComplex:
			if !isNumeric(target) {
				return nil, nil, InvalidUntypedConversion
			}
		case UntypedString:
			// Non-constant untyped string values are not permitted by the spec and
			// should not occur during normal typechecking passes, but this path is
			// reachable via the AssignableTo API.
			if !isString(target) {
				return nil, nil, InvalidUntypedConversion
			}
		default:
			return nil, nil, InvalidUntypedConversion
		}
	case *Interface:
		if isTypeParam(target) {
			if !u.typeSet().underIs(func(u Type) bool {
				if u == nil {
					return false
				}
				t, _, _ := check.implicitTypeAndValue(x, u)
				return t != nil
			}) {
				return nil, nil, InvalidUntypedConversion
			}
			break
		}
		// Update operand types to the default type rather than the target
		// (interface) type: values must have concrete dynamic types.
		// Untyped nil was handled upfront.
		if !u.Empty() {
			return nil, nil, InvalidUntypedConversion // cannot assign untyped values to non-empty interfaces
		}
		return Default(x.typ), nil, 0 // default type for nil is nil
	default:
		return nil, nil, InvalidUntypedConversion
	}
	return target, nil, 0
}

// If switchCase is true, the operator op is ignored.
func (check *Checker) comparison(x, y *operand, op syntax.Operator, switchCase bool) {
	// Avoid spurious errors if any of the operands has an invalid type (go.dev/issue/54405).
	if !isValid(x.typ) || !isValid(y.typ) {
		x.mode = invalid
		return
	}

	if switchCase {
		op = syntax.Eql
	}

	errOp := x  // operand for which error is reported, if any
	cause := "" // specific error cause, if any

	// spec: "In any comparison, the first operand must be assignable
	// to the type of the second operand, or vice versa."
	code := MismatchedTypes
	ok, _ := x.assignableTo(check, y.typ, nil)
	if !ok {
		ok, _ = y.assignableTo(check, x.typ, nil)
	}
	if !ok {
		// Report the error on the 2nd operand since we only
		// know after seeing the 2nd operand whether we have
		// a type mismatch.
		errOp = y
		cause = check.sprintf("mismatched types %s and %s", x.typ, y.typ)
		goto Error
	}

	// check if comparison is defined for operands
	code = UndefinedOp
	switch op {
	case syntax.Eql, syntax.Neq:
		// spec: "The equality operators == and != apply to operands that are comparable."
		switch {
		case x.isNil() || y.isNil():
			// Comparison against nil requires that the other operand type has nil.
			typ := x.typ
			if x.isNil() {
				typ = y.typ
			}
			if !hasNil(typ) {
				// This case should only be possible for "nil == nil".
				// Report the error on the 2nd operand since we only
				// know after seeing the 2nd operand whether we have
				// an invalid comparison.
				errOp = y
				goto Error
			}

		case !Comparable(x.typ):
			errOp = x
			cause = check.incomparableCause(x.typ)
			goto Error

		case !Comparable(y.typ):
			errOp = y
			cause = check.incomparableCause(y.typ)
			goto Error
		}

	case syntax.Lss, syntax.Leq, syntax.Gtr, syntax.Geq:
		// spec: The ordering operators <, <=, >, and >= apply to operands that are ordered."
		switch {
		case !allOrdered(x.typ):
			errOp = x
			goto Error
		case !allOrdered(y.typ):
			errOp = y
			goto Error
		}

	default:
		panic("unreachable")
	}

	// comparison is ok
	if x.mode == constant_ && y.mode == constant_ {
		x.val = constant.MakeBool(constant.Compare(x.val, op2tok[op], y.val))
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
	return

Error:
	// We have an offending operand errOp and possibly an error cause.
	if cause == "" {
		if isTypeParam(x.typ) || isTypeParam(y.typ) {
			// TODO(gri) should report the specific type causing the problem, if any
			if !isTypeParam(x.typ) {
				errOp = y
			}
			cause = check.sprintf("type parameter %s is not comparable with %s", errOp.typ, op)
		} else {
			cause = check.sprintf("operator %s not defined on %s", op, check.kindString(errOp.typ)) // catch-all
		}
	}
	if switchCase {
		check.errorf(x, code, "invalid case %s in switch on %s (%s)", x.expr, y.expr, cause) // error position always at 1st operand
	} else {
		check.errorf(errOp, code, invalidOp+"%s %s %s (%s)", x.expr, op, y.expr, cause)
	}
	x.mode = invalid
}

// incomparableCause returns a more specific cause why typ is not comparable.
// If there is no more specific cause, the result is "".
func (check *Checker) incomparableCause(typ Type) string {
	switch under(typ).(type) {
	case *Slice, *Signature, *Map:
		return check.kindString(typ) + " can only be compared to nil"
	}
	// see if we can extract a more specific error
	var cause string
	comparableType(typ, true, nil, func(format string, args ...interface{}) {
		cause = check.sprintf(format, args...)
	})
	return cause
}

// kindString returns the type kind as a string.
func (check *Checker) kindString(typ Type) string {
	switch under(typ).(type) {
	case *Array:
		return "array"
	case *Slice:
		return "slice"
	case *Struct:
		return "struct"
	case *Pointer:
		return "pointer"
	case *Signature:
		return "func"
	case *Interface:
		if isTypeParam(typ) {
			return check.sprintf("type parameter %s", typ)
		}
		return "interface"
	case *Map:
		return "map"
	case *Chan:
		return "chan"
	default:
		return check.sprintf("%s", typ) // catch-all
	}
}

// If e != nil, it must be the shift expression; it may be nil for non-constant shifts.
func (check *Checker) shift(x, y *operand, e syntax.Expr, op syntax.Operator) {
	// TODO(gri) This function seems overly complex. Revisit.

	var xval constant.Value
	if x.mode == constant_ {
		xval = constant.ToInt(x.val)
	}

	if allInteger(x.typ) || isUntyped(x.typ) && xval != nil && xval.Kind() == constant.Int {
		// The lhs is of integer type or an untyped constant representable
		// as an integer. Nothing to do.
	} else {
		// shift has no chance
		check.errorf(x, InvalidShiftOperand, invalidOp+"shifted operand %s must be integer", x)
		x.mode = invalid
		return
	}

	// spec: "The right operand in a shift expression must have integer type
	// or be an untyped constant representable by a value of type uint."

	// Check that constants are representable by uint, but do not convert them
	// (see also go.dev/issue/47243).
	var yval constant.Value
	if y.mode == constant_ {
		// Provide a good error message for negative shift counts.
		yval = constant.ToInt(y.val) // consider -1, 1.0, but not -1.1
		if yval.Kind() == constant.Int && constant.Sign(yval) < 0 {
			check.errorf(y, InvalidShiftCount, invalidOp+"negative shift count %s", y)
			x.mode = invalid
			return
		}

		if isUntyped(y.typ) {
			// Caution: Check for representability here, rather than in the switch
			// below, because isInteger includes untyped integers (was bug go.dev/issue/43697).
			check.representable(y, Typ[Uint])
			if y.mode == invalid {
				x.mode = invalid
				return
			}
		}
	} else {
		// Check that RHS is otherwise at least of integer type.
		switch {
		case allInteger(y.typ):
			if !allUnsigned(y.typ) && !check.verifyVersionf(y, go1_13, invalidOp+"signed shift count %s", y) {
				x.mode = invalid
				return
			}
		case isUntyped(y.typ):
			// This is incorrect, but preserves pre-existing behavior.
			// See also go.dev/issue/47410.
			check.convertUntyped(y, Typ[Uint])
			if y.mode == invalid {
				x.mode = invalid
				return
			}
		default:
			check.errorf(y, InvalidShiftCount, invalidOp+"shift count %s must be integer", y)
			x.mode = invalid
			return
		}
	}

	if x.mode == constant_ {
		if y.mode == constant_ {
			// if either x or y has an unknown value, the result is unknown
			if x.val.Kind() == constant.Unknown || y.val.Kind() == constant.Unknown {
				x.val = constant.MakeUnknown()
				// ensure the correct type - see comment below
				if !isInteger(x.typ) {
					x.typ = Typ[UntypedInt]
				}
				return
			}
			// rhs must be within reasonable bounds in constant shifts
			const shiftBound = 1023 - 1 + 52 // so we can express smallestFloat64 (see go.dev/issue/44057)
			s, ok := constant.Uint64Val(yval)
			if !ok || s > shiftBound {
				check.errorf(y, InvalidShiftCount, invalidOp+"invalid shift count %s", y)
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
			x.val = constant.Shift(xval, op2tok[op], uint(s))
			x.expr = e
			check.overflow(x, opPos(x.expr))
			return
		}

		// non-constant shift with constant lhs
		if isUntyped(x.typ) {
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
			// Example: var e, f = int(1<<""[f]) // go.dev/issue/11347
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
	if !allInteger(x.typ) {
		check.errorf(x, InvalidShiftOperand, invalidOp+"shifted operand %s must be integer", x)
		x.mode = invalid
		return
	}

	x.mode = value
}

var binaryOpPredicates opPredicates

func init() {
	// Setting binaryOpPredicates in init avoids declaration cycles.
	binaryOpPredicates = opPredicates{
		syntax.Add: allNumericOrString,
		syntax.Sub: allNumeric,
		syntax.Mul: allNumeric,
		syntax.Div: allNumeric,
		syntax.Rem: allInteger,

		syntax.And:    allInteger,
		syntax.Or:     allInteger,
		syntax.Xor:    allInteger,
		syntax.AndNot: allInteger,

		syntax.AndAnd: allBoolean,
		syntax.OrOr:   allBoolean,
	}
}

// If e != nil, it must be the binary expression; it may be nil for non-constant expressions
// (when invoked for an assignment operation where the binary expression is implicit).
func (check *Checker) binary(x *operand, e syntax.Expr, lhs, rhs syntax.Expr, op syntax.Operator) {
	var y operand

	check.expr(nil, x, lhs)
	check.expr(nil, &y, rhs)

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

	check.matchTypes(x, &y)
	if x.mode == invalid {
		return
	}

	if isComparison(op) {
		check.comparison(x, &y, op, false)
		return
	}

	if !Identical(x.typ, y.typ) {
		// only report an error if we have valid types
		// (otherwise we had an error reported elsewhere already)
		if isValid(x.typ) && isValid(y.typ) {
			if e != nil {
				check.errorf(x, MismatchedTypes, invalidOp+"%s (mismatched types %s and %s)", e, x.typ, y.typ)
			} else {
				check.errorf(x, MismatchedTypes, invalidOp+"%s %s= %s (mismatched types %s and %s)", lhs, op, rhs, x.typ, y.typ)
			}
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
		if (x.mode == constant_ || allInteger(x.typ)) && y.mode == constant_ && constant.Sign(y.val) == 0 {
			check.error(&y, DivByZero, invalidOp+"division by zero")
			x.mode = invalid
			return
		}

		// check for divisor underflow in complex division (see go.dev/issue/20227)
		if x.mode == constant_ && y.mode == constant_ && isComplex(x.typ) {
			re, im := constant.Real(y.val), constant.Imag(y.val)
			re2, im2 := constant.BinaryOp(re, token.MUL, re), constant.BinaryOp(im, token.MUL, im)
			if constant.Sign(re2) == 0 && constant.Sign(im2) == 0 {
				check.error(&y, DivByZero, invalidOp+"division by zero")
				x.mode = invalid
				return
			}
		}
	}

	if x.mode == constant_ && y.mode == constant_ {
		// if either x or y has an unknown value, the result is unknown
		if x.val.Kind() == constant.Unknown || y.val.Kind() == constant.Unknown {
			x.val = constant.MakeUnknown()
			// x.typ is unchanged
			return
		}
		// force integer division for integer operands
		tok := op2tok[op]
		if op == syntax.Div && isInteger(x.typ) {
			tok = token.QUO_ASSIGN
		}
		x.val = constant.BinaryOp(x.val, tok, y.val)
		x.expr = e
		check.overflow(x, opPos(x.expr))
		return
	}

	x.mode = value
	// x.typ is unchanged
}

// matchTypes attempts to convert any untyped types x and y such that they match.
// If an error occurs, x.mode is set to invalid.
func (check *Checker) matchTypes(x, y *operand) {
	// mayConvert reports whether the operands x and y may
	// possibly have matching types after converting one
	// untyped operand to the type of the other.
	// If mayConvert returns true, we try to convert the
	// operands to each other's types, and if that fails
	// we report a conversion failure.
	// If mayConvert returns false, we continue without an
	// attempt at conversion, and if the operand types are
	// not compatible, we report a type mismatch error.
	mayConvert := func(x, y *operand) bool {
		// If both operands are typed, there's no need for an implicit conversion.
		if isTyped(x.typ) && isTyped(y.typ) {
			return false
		}
		// An untyped operand may convert to its default type when paired with an empty interface
		// TODO(gri) This should only matter for comparisons (the only binary operation that is
		//           valid with interfaces), but in that case the assignability check should take
		//           care of the conversion. Verify and possibly eliminate this extra test.
		if isNonTypeParamInterface(x.typ) || isNonTypeParamInterface(y.typ) {
			return true
		}
		// A boolean type can only convert to another boolean type.
		if allBoolean(x.typ) != allBoolean(y.typ) {
			return false
		}
		// A string type can only convert to another string type.
		if allString(x.typ) != allString(y.typ) {
			return false
		}
		// Untyped nil can only convert to a type that has a nil.
		if x.isNil() {
			return hasNil(y.typ)
		}
		if y.isNil() {
			return hasNil(x.typ)
		}
		// An untyped operand cannot convert to a pointer.
		// TODO(gri) generalize to type parameters
		if isPointer(x.typ) || isPointer(y.typ) {
			return false
		}
		return true
	}

	if mayConvert(x, y) {
		check.convertUntyped(x, y.typ)
		if x.mode == invalid {
			return
		}
		check.convertUntyped(y, x.typ)
		if y.mode == invalid {
			x.mode = invalid
			return
		}
	}
}

// exprKind describes the kind of an expression; the kind
// determines if an expression is valid in 'statement context'.
type exprKind int

const (
	conversion exprKind = iota
	expression
	statement
)

// target represent the (signature) type and description of the LHS
// variable of an assignment, or of a function result variable.
type target struct {
	sig  *Signature
	desc string
}

// newTarget creates a new target for the given type and description.
// The result is nil if typ is not a signature.
func newTarget(typ Type, desc string) *target {
	if typ != nil {
		if sig, _ := under(typ).(*Signature); sig != nil {
			return &target{sig, desc}
		}
	}
	return nil
}

// rawExpr typechecks expression e and initializes x with the expression
// value or type. If an error occurred, x.mode is set to invalid.
// If a non-nil target T is given and e is a generic function,
// T is used to infer the type arguments for e.
// If hint != nil, it is the type of a composite literal element.
// If allowGeneric is set, the operand type may be an uninstantiated
// parameterized type or function value.
func (check *Checker) rawExpr(T *target, x *operand, e syntax.Expr, hint Type, allowGeneric bool) exprKind {
	if check.conf.Trace {
		check.trace(e.Pos(), "-- expr %s", e)
		check.indent++
		defer func() {
			check.indent--
			check.trace(e.Pos(), "=> %s", x)
		}()
	}

	kind := check.exprInternal(T, x, e, hint)

	if !allowGeneric {
		check.nonGeneric(T, x)
	}

	check.record(x)

	return kind
}

// If x is a generic type, or a generic function whose type arguments cannot be inferred
// from a non-nil target T, nonGeneric reports an error and invalidates x.mode and x.typ.
// Otherwise it leaves x alone.
func (check *Checker) nonGeneric(T *target, x *operand) {
	if x.mode == invalid || x.mode == novalue {
		return
	}
	var what string
	switch t := x.typ.(type) {
	case *Alias, *Named:
		if isGeneric(t) {
			what = "type"
		}
	case *Signature:
		if t.tparams != nil {
			if enableReverseTypeInference && T != nil {
				check.funcInst(T, x.Pos(), x, nil, true)
				return
			}
			what = "function"
		}
	}
	if what != "" {
		check.errorf(x.expr, WrongTypeArgCount, "cannot use generic %s %s without instantiation", what, x.expr)
		x.mode = invalid
		x.typ = Typ[Invalid]
	}
}

// langCompat reports an error if the representation of a numeric
// literal is not compatible with the current language version.
func (check *Checker) langCompat(lit *syntax.BasicLit) {
	s := lit.Value
	if len(s) <= 2 || check.allowVersion(lit, go1_13) {
		return
	}
	// len(s) > 2
	if strings.Contains(s, "_") {
		check.versionErrorf(lit, go1_13, "underscore in numeric literal")
		return
	}
	if s[0] != '0' {
		return
	}
	radix := s[1]
	if radix == 'b' || radix == 'B' {
		check.versionErrorf(lit, go1_13, "binary literal")
		return
	}
	if radix == 'o' || radix == 'O' {
		check.versionErrorf(lit, go1_13, "0o/0O-style octal literal")
		return
	}
	if lit.Kind != syntax.IntLit && (radix == 'x' || radix == 'X') {
		check.versionErrorf(lit, go1_13, "hexadecimal floating-point literal")
	}
}

// exprInternal contains the core of type checking of expressions.
// Must only be called by rawExpr.
// (See rawExpr for an explanation of the parameters.)
func (check *Checker) exprInternal(T *target, x *operand, e syntax.Expr, hint Type) exprKind {
	// make sure x has a valid state in case of bailout
	// (was go.dev/issue/5770)
	x.mode = invalid
	x.typ = Typ[Invalid]

	switch e := e.(type) {
	case nil:
		panic("unreachable")

	case *syntax.BadExpr:
		goto Error // error was reported before

	case *syntax.Name:
		check.ident(x, e, nil, false)

	case *syntax.DotsType:
		// dots are handled explicitly where they are legal
		// (array composite literals and parameter lists)
		check.error(e, BadDotDotDotSyntax, "invalid use of '...'")
		goto Error

	case *syntax.BasicLit:
		if e.Bad {
			goto Error // error reported during parsing
		}
		switch e.Kind {
		case syntax.IntLit, syntax.FloatLit, syntax.ImagLit:
			check.langCompat(e)
			// The max. mantissa precision for untyped numeric values
			// is 512 bits, or 4048 bits for each of the two integer
			// parts of a fraction for floating-point numbers that are
			// represented accurately in the go/constant package.
			// Constant literals that are longer than this many bits
			// are not meaningful; and excessively long constants may
			// consume a lot of space and time for a useless conversion.
			// Cap constant length with a generous upper limit that also
			// allows for separators between all digits.
			const limit = 10000
			if len(e.Value) > limit {
				check.errorf(e, InvalidConstVal, "excessively long constant: %s... (%d chars)", e.Value[:10], len(e.Value))
				goto Error
			}
		}
		x.setConst(e.Kind, e.Value)
		if x.mode == invalid {
			// The parser already establishes syntactic correctness.
			// If we reach here it's because of number under-/overflow.
			// TODO(gri) setConst (and in turn the go/constant package)
			// should return an error describing the issue.
			check.errorf(e, InvalidConstVal, "malformed constant: %s", e.Value)
			goto Error
		}
		// Ensure that integer values don't overflow (go.dev/issue/54280).
		x.expr = e // make sure that check.overflow below has an error position
		check.overflow(x, opPos(x.expr))

	case *syntax.FuncLit:
		if sig, ok := check.typ(e.Type).(*Signature); ok {
			// Set the Scope's extent to the complete "func (...) {...}"
			// so that Scope.Innermost works correctly.
			sig.scope.pos = e.Pos()
			sig.scope.end = syntax.EndPos(e)
			if !check.conf.IgnoreFuncBodies && e.Body != nil {
				// Anonymous functions are considered part of the
				// init expression/func declaration which contains
				// them: use existing package-level declaration info.
				decl := check.decl // capture for use in closure below
				iota := check.iota // capture for use in closure below (go.dev/issue/22345)
				// Don't type-check right away because the function may
				// be part of a type definition to which the function
				// body refers. Instead, type-check as soon as possible,
				// but before the enclosing scope contents changes (go.dev/issue/22992).
				check.later(func() {
					check.funcBody(decl, "<function literal>", sig, e.Body, iota)
				}).describef(e, "func literal")
			}
			x.mode = value
			x.typ = sig
		} else {
			check.errorf(e, InvalidSyntaxTree, "invalid function literal %v", e)
			goto Error
		}

	case *syntax.CompositeLit:
		check.compositeLit(T, x, e, hint)
		if x.mode == invalid {
			goto Error
		}

	case *syntax.ParenExpr:
		// type inference doesn't go past parentheses (target type T = nil)
		kind := check.rawExpr(nil, x, e.X, nil, false)
		x.expr = e
		return kind

	case *syntax.SelectorExpr:
		check.selector(x, e, nil, false)

	case *syntax.IndexExpr:
		if check.indexExpr(x, e) {
			if !enableReverseTypeInference {
				T = nil
			}
			check.funcInst(T, e.Pos(), x, e, true)
		}
		if x.mode == invalid {
			goto Error
		}

	case *syntax.SliceExpr:
		check.sliceExpr(x, e)
		if x.mode == invalid {
			goto Error
		}

	case *syntax.AssertExpr:
		check.expr(nil, x, e.X)
		if x.mode == invalid {
			goto Error
		}
		// x.(type) expressions are encoded via TypeSwitchGuards
		if e.Type == nil {
			check.error(e, InvalidSyntaxTree, "invalid use of AssertExpr")
			goto Error
		}
		if isTypeParam(x.typ) {
			check.errorf(x, InvalidAssert, invalidOp+"cannot use type assertion on type parameter value %s", x)
			goto Error
		}
		if _, ok := under(x.typ).(*Interface); !ok {
			check.errorf(x, InvalidAssert, invalidOp+"%s is not an interface", x)
			goto Error
		}
		T := check.varType(e.Type)
		if !isValid(T) {
			goto Error
		}
		check.typeAssertion(e, x, T, false)
		x.mode = commaok
		x.typ = T

	case *syntax.TypeSwitchGuard:
		// x.(type) expressions are handled explicitly in type switches
		check.error(e, InvalidSyntaxTree, "use of .(type) outside type switch")
		check.use(e.X)
		goto Error

	case *syntax.CallExpr:
		return check.callExpr(x, e)

	case *syntax.ListExpr:
		// catch-all for unexpected expression lists
		check.error(e, InvalidSyntaxTree, "unexpected list of expressions")
		goto Error

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
				check.exprOrType(x, e.X, false)
				switch x.mode {
				case invalid:
					goto Error
				case typexpr:
					check.validVarType(e.X, x.typ)
					x.typ = &Pointer{base: x.typ}
				default:
					var base Type
					if !underIs(x.typ, func(u Type) bool {
						p, _ := u.(*Pointer)
						if p == nil {
							check.errorf(x, InvalidIndirection, invalidOp+"cannot indirect %s", x)
							return false
						}
						if base != nil && !Identical(p.base, base) {
							check.errorf(x, InvalidIndirection, invalidOp+"pointers of %s must have identical base types", x)
							return false
						}
						base = p.base
						return true
					}) {
						goto Error
					}
					x.mode = variable
					x.typ = base
				}
				break
			}

			check.unary(x, e)
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
		check.error(e, InvalidSyntaxTree, "no key:value expected")
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
		panic(fmt.Sprintf("%s: unknown expression type %T", atPos(e), e))
	}

	// everything went well
	x.expr = e
	return expression

Error:
	x.mode = invalid
	x.expr = e
	return statement // avoid follow-up errors
}

// keyVal maps a complex, float, integer, string or boolean constant value
// to the corresponding complex128, float64, int64, uint64, string, or bool
// Go value if possible; otherwise it returns x.
// A complex constant that can be represented as a float (such as 1.2 + 0i)
// is returned as a floating point value; if a floating point value can be
// represented as an integer (such as 1.0) it is returned as an integer value.
// This ensures that constants of different kind but equal value (such as
// 1.0 + 0i, 1.0, 1) result in the same value.
func keyVal(x constant.Value) interface{} {
	switch x.Kind() {
	case constant.Complex:
		f := constant.ToFloat(x)
		if f.Kind() != constant.Float {
			r, _ := constant.Float64Val(constant.Real(x))
			i, _ := constant.Float64Val(constant.Imag(x))
			return complex(r, i)
		}
		x = f
		fallthrough
	case constant.Float:
		i := constant.ToInt(x)
		if i.Kind() != constant.Int {
			v, _ := constant.Float64Val(x)
			return v
		}
		x = i
		fallthrough
	case constant.Int:
		if v, ok := constant.Int64Val(x); ok {
			return v
		}
		if v, ok := constant.Uint64Val(x); ok {
			return v
		}
	case constant.String:
		return constant.StringVal(x)
	case constant.Bool:
		return constant.BoolVal(x)
	}
	return x
}

// typeAssertion checks x.(T). The type of x must be an interface.
func (check *Checker) typeAssertion(e syntax.Expr, x *operand, T Type, typeSwitch bool) {
	var cause string
	if check.assertableTo(x.typ, T, &cause) {
		return // success
	}

	if typeSwitch {
		check.errorf(e, ImpossibleAssert, "impossible type switch case: %s\n\t%s cannot have dynamic type %s %s", e, x, T, cause)
		return
	}

	check.errorf(e, ImpossibleAssert, "impossible type assertion: %s\n\t%s does not implement %s %s", e, T, x.typ, cause)
}

// expr typechecks expression e and initializes x with the expression value.
// If a non-nil target T is given and e is a generic function or
// a function call, T is used to infer the type arguments for e.
// The result must be a single value.
// If an error occurred, x.mode is set to invalid.
func (check *Checker) expr(T *target, x *operand, e syntax.Expr) {
	check.rawExpr(T, x, e, nil, false)
	check.exclude(x, 1<<novalue|1<<builtin|1<<typexpr)
	check.singleValue(x)
}

// genericExpr is like expr but the result may also be generic.
func (check *Checker) genericExpr(x *operand, e syntax.Expr) {
	check.rawExpr(nil, x, e, nil, true)
	check.exclude(x, 1<<novalue|1<<builtin|1<<typexpr)
	check.singleValue(x)
}

// multiExpr typechecks e and returns its value (or values) in list.
// If allowCommaOk is set and e is a map index, comma-ok, or comma-err
// expression, the result is a two-element list containing the value
// of e, and an untyped bool value or an error value, respectively.
// If an error occurred, list[0] is not valid.
func (check *Checker) multiExpr(e syntax.Expr, allowCommaOk bool) (list []*operand, commaOk bool) {
	var x operand
	check.rawExpr(nil, &x, e, nil, false)
	check.exclude(&x, 1<<novalue|1<<builtin|1<<typexpr)

	if t, ok := x.typ.(*Tuple); ok && x.mode != invalid {
		// multiple values
		list = make([]*operand, t.Len())
		for i, v := range t.vars {
			list[i] = &operand{mode: value, expr: e, typ: v.typ}
		}
		return
	}

	// exactly one (possibly invalid or comma-ok) value
	list = []*operand{&x}
	if allowCommaOk && (x.mode == mapindex || x.mode == commaok || x.mode == commaerr) {
		x2 := &operand{mode: value, expr: e, typ: Typ[UntypedBool]}
		if x.mode == commaerr {
			x2.typ = universeError
		}
		list = append(list, x2)
		commaOk = true
	}

	return
}

// exprWithHint typechecks expression e and initializes x with the expression value;
// hint is the type of a composite literal element.
// If an error occurred, x.mode is set to invalid.
func (check *Checker) exprWithHint(x *operand, e syntax.Expr, hint Type) {
	assert(hint != nil)
	check.rawExpr(nil, x, e, hint, false)
	check.exclude(x, 1<<novalue|1<<builtin|1<<typexpr)
	check.singleValue(x)
}

// exprOrType typechecks expression or type e and initializes x with the expression value or type.
// If allowGeneric is set, the operand type may be an uninstantiated parameterized type or function
// value.
// If an error occurred, x.mode is set to invalid.
func (check *Checker) exprOrType(x *operand, e syntax.Expr, allowGeneric bool) {
	check.rawExpr(nil, x, e, nil, allowGeneric)
	check.exclude(x, 1<<novalue)
	check.singleValue(x)
}

// exclude reports an error if x.mode is in modeset and sets x.mode to invalid.
// The modeset may contain any of 1<<novalue, 1<<builtin, 1<<typexpr.
func (check *Checker) exclude(x *operand, modeset uint) {
	if modeset&(1<<x.mode) != 0 {
		var msg string
		var code Code
		switch x.mode {
		case novalue:
			if modeset&(1<<typexpr) != 0 {
				msg = "%s used as value"
			} else {
				msg = "%s used as value or type"
			}
			code = TooManyValues
		case builtin:
			msg = "%s must be called"
			code = UncalledBuiltin
		case typexpr:
			msg = "%s is not an expression"
			code = NotAnExpr
		default:
			panic("unreachable")
		}
		check.errorf(x, code, msg, x)
		x.mode = invalid
	}
}

// singleValue reports an error if x describes a tuple and sets x.mode to invalid.
func (check *Checker) singleValue(x *operand) {
	if x.mode == value {
		// tuple types are never named - no need for underlying type below
		if t, ok := x.typ.(*Tuple); ok {
			assert(t.Len() != 1)
			check.errorf(x, TooManyValues, "multiple-value %s in single-value context", x)
			x.mode = invalid
		}
	}
}

// op2tok translates syntax.Operators into token.Tokens.
var op2tok = [...]token.Token{
	syntax.Def:  token.ILLEGAL,
	syntax.Not:  token.NOT,
	syntax.Recv: token.ILLEGAL,

	syntax.OrOr:   token.LOR,
	syntax.AndAnd: token.LAND,

	syntax.Eql: token.EQL,
	syntax.Neq: token.NEQ,
	syntax.Lss: token.LSS,
	syntax.Leq: token.LEQ,
	syntax.Gtr: token.GTR,
	syntax.Geq: token.GEQ,

	syntax.Add: token.ADD,
	syntax.Sub: token.SUB,
	syntax.Or:  token.OR,
	syntax.Xor: token.XOR,

	syntax.Mul:    token.MUL,
	syntax.Div:    token.QUO,
	syntax.Rem:    token.REM,
	syntax.And:    token.AND,
	syntax.AndNot: token.AND_NOT,
	syntax.Shl:    token.SHL,
	syntax.Shr:    token.SHR,
}
