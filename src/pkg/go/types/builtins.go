// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of builtin function calls.

package types

import (
	"go/ast"
	"go/token"
)

// builtin typechecks a built-in call. The built-in type is bin, and iota is the current
// value of iota or -1 if iota doesn't have a value in the current context. The result
// of the call is returned via x. If the call has type errors, the returned x is marked
// as invalid (x.mode == invalid).
//
func (check *checker) builtin(x *operand, call *ast.CallExpr, bin *builtin, iota int) {
	args := call.Args
	id := bin.id

	// declare before goto's
	var arg0 ast.Expr // first argument, if present

	// check argument count
	n := len(args)
	msg := ""
	if n < bin.nargs {
		msg = "not enough"
	} else if !bin.isVariadic && n > bin.nargs {
		msg = "too many"
	}
	if msg != "" {
		check.invalidOp(call.Pos(), msg+" arguments for %s (expected %d, found %d)", call, bin.nargs, n)
		goto Error
	}

	// common case: evaluate first argument if present;
	// if it is an expression, x has the expression value
	if n > 0 {
		arg0 = args[0]
		switch id {
		case _Make, _New, _Trace:
			// respective cases below do the work
		default:
			// argument must be an expression
			check.expr(x, arg0, nil, iota)
			if x.mode == invalid {
				goto Error
			}
		}
	}

	switch id {
	case _Append:
		if _, ok := underlying(x.typ).(*Slice); !ok {
			check.invalidArg(x.pos(), "%s is not a typed slice", x)
			goto Error
		}
		resultTyp := x.typ
		for _, arg := range args[1:] {
			check.expr(x, arg, nil, iota)
			if x.mode == invalid {
				goto Error
			}
			// TODO(gri) check assignability
		}
		x.mode = value
		x.typ = resultTyp

	case _Cap, _Len:
		mode := invalid
		var val interface{}
		switch typ := implicitDeref(underlying(x.typ)).(type) {
		case *Basic:
			if isString(typ) && id == _Len {
				if x.mode == constant {
					mode = constant
					val = int64(len(x.val.(string)))
				} else {
					mode = value
				}
			}

		case *Array:
			mode = value
			if !containsCallsOrReceives(arg0) {
				mode = constant
				val = typ.Len
			}

		case *Slice, *Chan:
			mode = value

		case *Map:
			if id == _Len {
				mode = value
			}
		}

		if mode == invalid {
			check.invalidArg(x.pos(), "%s for %s", x, bin.name)
			goto Error
		}
		x.mode = mode
		x.typ = Typ[Int]
		x.val = val

	case _Close:
		ch, ok := underlying(x.typ).(*Chan)
		if !ok {
			check.invalidArg(x.pos(), "%s is not a channel", x)
			goto Error
		}
		if ch.Dir&ast.SEND == 0 {
			check.invalidArg(x.pos(), "%s must not be a receive-only channel", x)
			goto Error
		}
		x.mode = novalue

	case _Complex:
		if !check.complexArg(x) {
			goto Error
		}

		var y operand
		check.expr(&y, args[1], nil, iota)
		if y.mode == invalid {
			goto Error
		}
		if !check.complexArg(&y) {
			goto Error
		}

		check.convertUntyped(x, y.typ)
		if x.mode == invalid {
			goto Error
		}
		check.convertUntyped(&y, x.typ)
		if y.mode == invalid {
			goto Error
		}

		if !IsIdentical(x.typ, y.typ) {
			check.invalidArg(x.pos(), "mismatched types %s and %s", x.typ, y.typ)
			goto Error
		}

		typ := underlying(x.typ).(*Basic)
		if x.mode == constant && y.mode == constant {
			x.val = binaryOpConst(x.val, toImagConst(y.val), token.ADD, typ)
		} else {
			x.mode = value
		}

		switch typ.Kind {
		case Float32:
			x.typ = Typ[Complex64]
		case Float64:
			x.typ = Typ[Complex128]
		case UntypedInt, UntypedRune, UntypedFloat:
			x.typ = Typ[UntypedComplex]
		default:
			check.invalidArg(x.pos(), "float32 or float64 arguments expected")
			goto Error
		}

	case _Copy:
		var y operand
		check.expr(&y, args[1], nil, iota)
		if y.mode == invalid {
			goto Error
		}

		var dst, src Type
		if t, ok := underlying(x.typ).(*Slice); ok {
			dst = t.Elt
		}
		switch t := underlying(y.typ).(type) {
		case *Basic:
			if isString(y.typ) {
				src = Typ[Byte]
			}
		case *Slice:
			src = t.Elt
		}

		if dst == nil || src == nil {
			check.invalidArg(x.pos(), "copy expects slice arguments; found %s and %s", x, &y)
			goto Error
		}

		if !IsIdentical(dst, src) {
			check.invalidArg(x.pos(), "arguments to copy %s and %s have different element types %s and %s", x, &y, dst, src)
			goto Error
		}

		x.mode = value
		x.typ = Typ[Int]

	case _Delete:
		m, ok := underlying(x.typ).(*Map)
		if !ok {
			check.invalidArg(x.pos(), "%s is not a map", x)
			goto Error
		}
		check.expr(x, args[1], nil, iota)
		if x.mode == invalid {
			goto Error
		}
		if !x.isAssignable(m.Key) {
			check.invalidArg(x.pos(), "%s is not assignable to %s", x, m.Key)
			goto Error
		}
		x.mode = novalue

	case _Imag, _Real:
		if !isComplex(x.typ) {
			check.invalidArg(x.pos(), "%s must be a complex number", x)
			goto Error
		}
		if x.mode == constant {
			// nothing to do for x.val == 0
			if !isZeroConst(x.val) {
				c := x.val.(Complex)
				if id == _Real {
					x.val = c.Re
				} else {
					x.val = c.Im
				}
			}
		} else {
			x.mode = value
		}
		k := Invalid
		switch underlying(x.typ).(*Basic).Kind {
		case Complex64:
			k = Float32
		case Complex128:
			k = Float64
		case UntypedComplex:
			k = UntypedFloat
		default:
			unreachable()
		}
		x.typ = Typ[k]

	case _Make:
		resultTyp := check.typ(arg0, false)
		if resultTyp == Typ[Invalid] {
			goto Error
		}
		var min int // minimum number of arguments
		switch underlying(resultTyp).(type) {
		case *Slice:
			min = 2
		case *Map, *Chan:
			min = 1
		default:
			check.invalidArg(arg0.Pos(), "cannot make %s; type must be slice, map, or channel", arg0)
			goto Error
		}
		if n := len(args); n < min || min+1 < n {
			check.errorf(call.Pos(), "%s expects %d or %d arguments; found %d", call, min, min+1, n)
			goto Error
		}
		var sizes []interface{} // constant integer arguments, if any
		for _, arg := range args[1:] {
			check.expr(x, arg, nil, iota)
			if x.isInteger() {
				if x.mode == constant {
					if isNegConst(x.val) {
						check.invalidArg(x.pos(), "%s must not be negative", x)
						// safe to continue
					} else {
						sizes = append(sizes, x.val) // x.val >= 0
					}
				}
			} else {
				check.invalidArg(x.pos(), "%s must be an integer", x)
				// safe to continue
			}
		}
		if len(sizes) == 2 && compareConst(sizes[0], sizes[1], token.GTR) {
			check.invalidArg(args[1].Pos(), "length and capacity swapped")
			// safe to continue
		}
		x.mode = variable
		x.typ = resultTyp

	case _New:
		resultTyp := check.typ(arg0, false)
		if resultTyp == Typ[Invalid] {
			goto Error
		}
		x.mode = variable
		x.typ = &Pointer{Base: resultTyp}

	case _Panic, _Print, _Println:
		for _, arg := range args[1:] {
			check.expr(x, arg, nil, -1)
		}
		x.mode = novalue

	case _Recover:
		x.mode = value
		x.typ = new(Interface)

	case _Alignof:
		x.mode = constant
		x.typ = Typ[Uintptr]
		// For now we return 1 always as it satisfies the spec's alignment guarantees.
		// TODO(gri) Extend typechecker API so that platform-specific values can be
		//           provided.
		x.val = int64(1)

	case _Offsetof:
		if _, ok := unparen(x.expr).(*ast.SelectorExpr); !ok {
			check.invalidArg(x.pos(), "%s is not a selector", x)
			goto Error
		}
		x.mode = constant
		x.typ = Typ[Uintptr]
		// because of the size guarantees for basic types (> 0 for some),
		// returning 0 is only correct if two distinct non-zero size
		// structs can have the same address (the spec permits that)
		x.val = int64(0)

	case _Sizeof:
		x.mode = constant
		x.val = sizeof(check.ctxt, x.typ)
		x.typ = Typ[Uintptr]

	case _Assert:
		// assert(pred) causes a typechecker error if pred is false.
		// The result of assert is the value of pred if there is no error.
		// Note: assert is only available in self-test mode.
		if x.mode != constant || !isBoolean(x.typ) {
			check.invalidArg(x.pos(), "%s is not a boolean constant", x)
			goto Error
		}
		pred, ok := x.val.(bool)
		if !ok {
			check.errorf(x.pos(), "internal error: value of %s should be a boolean constant", x)
			goto Error
		}
		if !pred {
			check.errorf(call.Pos(), "%s failed", call)
			// compile-time assertion failure - safe to continue
		}

	case _Trace:
		// trace(x, y, z, ...) dumps the positions, expressions, and
		// values of its arguments. The result of trace is the value
		// of the first argument.
		// Note: trace is only available in self-test mode.
		if len(args) == 0 {
			check.dump("%s: trace() without arguments", call.Pos())
			x.mode = novalue
			x.expr = call
			return
		}
		var t operand
		x1 := x
		for _, arg := range args {
			check.rawExpr(x1, arg, nil, iota, true) // permit trace for types, e.g.: new(trace(T))
			check.dump("%s: %s", x1.pos(), x1)
			x1 = &t // use incoming x only for first argument
		}

	default:
		check.invalidAST(call.Pos(), "unknown builtin id %d", id)
		goto Error
	}

	x.expr = call
	return

Error:
	x.mode = invalid
	x.expr = call
}

// implicitDeref returns A if typ is of the form *A and A is an array;
// otherwise it returns typ.
//
func implicitDeref(typ Type) Type {
	if p, ok := typ.(*Pointer); ok {
		if a, ok := underlying(p.Base).(*Array); ok {
			return a
		}
	}
	return typ
}

// containsCallsOrReceives returns true if the expression x contains
// function calls or channel receives; it returns false otherwise.
//
func containsCallsOrReceives(x ast.Expr) bool {
	res := false
	ast.Inspect(x, func(x ast.Node) bool {
		switch x := x.(type) {
		case *ast.CallExpr:
			res = true
			return false
		case *ast.UnaryExpr:
			if x.Op == token.ARROW {
				res = true
				return false
			}
		}
		return true
	})
	return res
}

// unparen removes any parentheses surrounding an expression and returns
// the naked expression.
//
func unparen(x ast.Expr) ast.Expr {
	if p, ok := x.(*ast.ParenExpr); ok {
		return unparen(p.X)
	}
	return x
}

func (check *checker) complexArg(x *operand) bool {
	t, _ := underlying(x.typ).(*Basic)
	if t != nil && (t.Info&IsFloat != 0 || t.Kind == UntypedInt || t.Kind == UntypedRune) {
		return true
	}
	check.invalidArg(x.pos(), "%s must be a float32, float64, or an untyped non-complex numeric constant", x)
	return false
}

func sizeof(ctxt *Context, typ Type) int64 {
	switch typ := underlying(typ).(type) {
	case *Basic:
		switch typ.Kind {
		case Int, Uint:
			return ctxt.IntSize
		case Uintptr:
			return ctxt.PtrSize
		}
		return typ.Size
	case *Array:
		return sizeof(ctxt, typ.Elt) * typ.Len
	case *Struct:
		var size int64
		for _, f := range typ.Fields {
			size += sizeof(ctxt, f.Type)
		}
		return size
	}
	return ctxt.PtrSize // good enough
}
