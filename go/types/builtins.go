// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of builtin function calls.

package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// TODO(gri): Several built-ins are missing assignment checks. As a result,
//            non-constant shift arguments may not be properly type-checked.

// builtin typechecks a call to a built-in and returns the result via x.
// If the call has type errors, the returned x is marked as invalid.
//
func (check *checker) builtin(x *operand, call *ast.CallExpr, id builtinId) {
	args := call.Args

	// declare before goto's
	var arg0 ast.Expr // first argument, if present

	// check argument count
	n := len(args)
	msg := ""
	bin := predeclaredFuncs[id]
	if n < bin.nargs {
		msg = "not enough"
	} else if !bin.variadic && n > bin.nargs {
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
		case _Make, _New, _Print, _Println, _Offsetof, _Trace:
			// respective cases below do the work
		default:
			// argument must be an expression
			check.expr(x, arg0)
			if x.mode == invalid {
				goto Error
			}
		}
	}

	switch id {
	case _Append:
		if _, ok := x.typ.Underlying().(*Slice); !ok {
			check.invalidArg(x.pos(), "%s is not a typed slice", x)
			goto Error
		}
		resultTyp := x.typ
		for _, arg := range args[1:] {
			check.expr(x, arg)
			if x.mode == invalid {
				goto Error
			}
			// TODO(gri) check assignability
		}
		x.mode = value
		x.typ = resultTyp

	case _Cap, _Len:
		mode := invalid
		var val exact.Value
		switch typ := implicitArrayDeref(x.typ.Underlying()).(type) {
		case *Basic:
			if isString(typ) && id == _Len {
				if x.mode == constant {
					mode = constant
					val = exact.MakeInt64(int64(len(exact.StringVal(x.val))))
				} else {
					mode = value
				}
			}

		case *Array:
			mode = value
			// spec: "The expressions len(s) and cap(s) are constants
			// if the type of s is an array or pointer to an array and
			// the expression s does not contain channel receives or
			// function calls; in this case s is not evaluated."
			if !check.containsCallsOrReceives(arg0) {
				mode = constant
				val = exact.MakeInt64(typ.len)
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
		ch, ok := x.typ.Underlying().(*Chan)
		if !ok {
			check.invalidArg(x.pos(), "%s is not a channel", x)
			goto Error
		}
		if ch.dir&ast.SEND == 0 {
			check.invalidArg(x.pos(), "%s must not be a receive-only channel", x)
			goto Error
		}
		x.mode = novalue

	case _Complex:
		if !check.complexArg(x) {
			goto Error
		}

		var y operand
		check.expr(&y, args[1])
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

		typ := x.typ.Underlying().(*Basic)
		if x.mode == constant && y.mode == constant {
			x.val = exact.BinaryOp(x.val, token.ADD, exact.MakeImag(y.val))
		} else {
			x.mode = value
		}

		switch typ.kind {
		case Float32:
			x.typ = Typ[Complex64]
		case Float64:
			x.typ = Typ[Complex128]
		case UntypedInt, UntypedRune, UntypedFloat:
			if x.mode == constant {
				typ = defaultType(typ).(*Basic)
				x.typ = Typ[UntypedComplex]
			} else {
				// untyped but not constant; probably because one
				// operand is a non-constant shift of untyped lhs
				typ = Typ[Float64]
				x.typ = Typ[Complex128]
			}
		default:
			check.invalidArg(x.pos(), "float32 or float64 arguments expected")
			goto Error
		}

		if x.mode != constant {
			// The arguments have now their final types, which at run-
			// time will be materialized. Update the expression trees.
			// If the current types are untyped, the materialized type
			// is the respective default type.
			// (If the result is constant, the arguments are never
			// materialized and there is nothing to do.)
			check.updateExprType(args[0], typ, true)
			check.updateExprType(args[1], typ, true)
		}

	case _Copy:
		var y operand
		check.expr(&y, args[1])
		if y.mode == invalid {
			goto Error
		}

		var dst, src Type
		if t, ok := x.typ.Underlying().(*Slice); ok {
			dst = t.elt
		}
		switch t := y.typ.Underlying().(type) {
		case *Basic:
			if isString(y.typ) {
				src = Typ[Byte]
			}
		case *Slice:
			src = t.elt
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
		m, ok := x.typ.Underlying().(*Map)
		if !ok {
			check.invalidArg(x.pos(), "%s is not a map", x)
			goto Error
		}
		check.expr(x, args[1])
		if x.mode == invalid {
			goto Error
		}
		if !x.isAssignableTo(check.conf, m.key) {
			check.invalidArg(x.pos(), "%s is not assignable to %s", x, m.key)
			goto Error
		}
		x.mode = novalue

	case _Imag, _Real:
		if !isComplex(x.typ) {
			check.invalidArg(x.pos(), "%s must be a complex number", x)
			goto Error
		}
		if x.mode == constant {
			if id == _Real {
				x.val = exact.Real(x.val)
			} else {
				x.val = exact.Imag(x.val)
			}
		} else {
			x.mode = value
		}
		k := Invalid
		switch x.typ.Underlying().(*Basic).kind {
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
		resultTyp := check.typ(arg0, nil, false)
		if resultTyp == Typ[Invalid] {
			goto Error
		}
		var min int // minimum number of arguments
		switch resultTyp.Underlying().(type) {
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
		var sizes []int64 // constant integer arguments, if any
		for _, arg := range args[1:] {
			if s, ok := check.index(arg, -1); ok && s >= 0 {
				sizes = append(sizes, s)
			}
		}
		if len(sizes) == 2 && sizes[0] > sizes[1] {
			check.invalidArg(args[1].Pos(), "length and capacity swapped")
			// safe to continue
		}
		x.mode = variable
		x.typ = resultTyp

	case _New:
		resultTyp := check.typ(arg0, nil, false)
		if resultTyp == Typ[Invalid] {
			goto Error
		}
		x.mode = variable
		x.typ = &Pointer{base: resultTyp}

	case _Panic:
		x.mode = novalue

	case _Print, _Println:
		for _, arg := range args {
			check.expr(x, arg)
			if x.mode == invalid {
				goto Error
			}
		}
		x.mode = novalue

	case _Recover:
		x.mode = value
		x.typ = new(Interface)

	case _Alignof:
		x.mode = constant
		x.val = exact.MakeInt64(check.conf.alignof(x.typ))
		x.typ = Typ[Uintptr]

	case _Offsetof:
		arg, ok := unparen(arg0).(*ast.SelectorExpr)
		if !ok {
			check.invalidArg(arg0.Pos(), "%s is not a selector expression", arg0)
			goto Error
		}
		check.expr(x, arg.X)
		if x.mode == invalid {
			goto Error
		}
		base := derefStructPtr(x.typ)
		sel := arg.Sel.Name
		obj, index, indirect := LookupFieldOrMethod(base, check.pkg, arg.Sel.Name)
		switch obj.(type) {
		case nil:
			check.invalidArg(x.pos(), "%s has no single field %s", base, sel)
			goto Error
		case *Func:
			check.invalidArg(arg0.Pos(), "%s is a method value", arg0)
			goto Error
		}
		if indirect {
			check.invalidArg(x.pos(), "field %s is embedded via a pointer in %s", sel, base)
			goto Error
		}

		// TODO(gri) Should we pass x.typ instead of base (and indirect report if derefStructPtr indirected)?
		check.recordSelection(arg, FieldVal, base, obj, index, false)

		offs := check.conf.offsetof(base, index)
		x.mode = constant
		x.val = exact.MakeInt64(offs)
		x.typ = Typ[Uintptr]

	case _Sizeof:
		x.mode = constant
		x.val = exact.MakeInt64(check.conf.sizeof(x.typ))
		x.typ = Typ[Uintptr]

	case _Assert:
		// assert(pred) causes a typechecker error if pred is false.
		// The result of assert is the value of pred if there is no error.
		// Note: assert is only available in self-test mode.
		if x.mode != constant || !isBoolean(x.typ) {
			check.invalidArg(x.pos(), "%s is not a boolean constant", x)
			goto Error
		}
		if x.val.Kind() != exact.Bool {
			check.errorf(x.pos(), "internal error: value of %s should be a boolean constant", x)
			goto Error
		}
		if !exact.BoolVal(x.val) {
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
			check.rawExpr(x1, arg, nil) // permit trace for types, e.g.: new(trace(T))
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

// implicitArrayDeref returns A if typ is of the form *A and A is an array;
// otherwise it returns typ.
//
func implicitArrayDeref(typ Type) Type {
	if p, ok := typ.(*Pointer); ok {
		if a, ok := p.base.Underlying().(*Array); ok {
			return a
		}
	}
	return typ
}

// containsCallsOrReceives reports if x contains function calls or channel receives.
// Expects that x was type-checked already.
//
func (check *checker) containsCallsOrReceives(x ast.Expr) (found bool) {
	ast.Inspect(x, func(x ast.Node) bool {
		switch x := x.(type) {
		case *ast.CallExpr:
			// calls and conversions look the same
			if !check.conversions[x] {
				found = true
			}
		case *ast.UnaryExpr:
			if x.Op == token.ARROW {
				found = true
			}
		}
		return !found // no need to continue if found
	})
	return
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
	t, _ := x.typ.Underlying().(*Basic)
	if t != nil && (t.info&IsFloat != 0 || t.kind == UntypedInt || t.kind == UntypedRune) {
		return true
	}
	check.invalidArg(x.pos(), "%s must be a float32, float64, or an untyped non-complex numeric constant", x)
	return false
}
