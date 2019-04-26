// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of builtin function calls.

package types

import (
	"go/ast"
	"go/constant"
	"go/token"
)

// builtin type-checks a call to the built-in specified by id and
// reports whether the call is valid, with *x holding the result;
// but x.expr is not set. If the call is invalid, the result is
// false, and *x is undefined.
//
func (check *Checker) builtin(x *operand, call *ast.CallExpr, id builtinId) (_ bool) {
	// append is the only built-in that permits the use of ... for the last argument
	bin := predeclaredFuncs[id]
	if call.Ellipsis.IsValid() && id != _Append {
		check.invalidOp(call.Ellipsis, "invalid use of ... with built-in %s", bin.name)
		check.use(call.Args...)
		return
	}

	// For len(x) and cap(x) we need to know if x contains any function calls or
	// receive operations. Save/restore current setting and set hasCallOrRecv to
	// false for the evaluation of x so that we can check it afterwards.
	// Note: We must do this _before_ calling unpack because unpack evaluates the
	//       first argument before we even call arg(x, 0)!
	if id == _Len || id == _Cap {
		defer func(b bool) {
			check.hasCallOrRecv = b
		}(check.hasCallOrRecv)
		check.hasCallOrRecv = false
	}

	// determine actual arguments
	var arg getter
	nargs := len(call.Args)
	switch id {
	default:
		// make argument getter
		arg, nargs, _ = unpack(func(x *operand, i int) { check.multiExpr(x, call.Args[i]) }, nargs, false)
		if arg == nil {
			return
		}
		// evaluate first argument, if present
		if nargs > 0 {
			arg(x, 0)
			if x.mode == invalid {
				return
			}
		}
	case _Make, _New, _Offsetof, _Trace:
		// arguments require special handling
	}

	// check argument count
	{
		msg := ""
		if nargs < bin.nargs {
			msg = "not enough"
		} else if !bin.variadic && nargs > bin.nargs {
			msg = "too many"
		}
		if msg != "" {
			check.invalidOp(call.Rparen, "%s arguments for %s (expected %d, found %d)", msg, call, bin.nargs, nargs)
			return
		}
	}

	switch id {
	case _Append:
		// append(s S, x ...T) S, where T is the element type of S
		// spec: "The variadic function append appends zero or more values x to s of type
		// S, which must be a slice type, and returns the resulting slice, also of type S.
		// The values x are passed to a parameter of type ...T where T is the element type
		// of S and the respective parameter passing rules apply."
		S := x.typ
		var T Type
		if s, _ := S.Underlying().(*Slice); s != nil {
			T = s.elem
		} else {
			check.invalidArg(x.pos(), "%s is not a slice", x)
			return
		}

		// remember arguments that have been evaluated already
		alist := []operand{*x}

		// spec: "As a special case, append also accepts a first argument assignable
		// to type []byte with a second argument of string type followed by ... .
		// This form appends the bytes of the string.
		if nargs == 2 && call.Ellipsis.IsValid() && x.assignableTo(check, NewSlice(universeByte), nil) {
			arg(x, 1)
			if x.mode == invalid {
				return
			}
			if isString(x.typ) {
				if check.Types != nil {
					sig := makeSig(S, S, x.typ)
					sig.variadic = true
					check.recordBuiltinType(call.Fun, sig)
				}
				x.mode = value
				x.typ = S
				break
			}
			alist = append(alist, *x)
			// fallthrough
		}

		// check general case by creating custom signature
		sig := makeSig(S, S, NewSlice(T)) // []T required for variadic signature
		sig.variadic = true
		check.arguments(x, call, sig, func(x *operand, i int) {
			// only evaluate arguments that have not been evaluated before
			if i < len(alist) {
				*x = alist[i]
				return
			}
			arg(x, i)
		}, nargs)
		// ok to continue even if check.arguments reported errors

		x.mode = value
		x.typ = S
		if check.Types != nil {
			check.recordBuiltinType(call.Fun, sig)
		}

	case _Cap, _Len:
		// cap(x)
		// len(x)
		mode := invalid
		var typ Type
		var val constant.Value
		switch typ = implicitArrayDeref(x.typ.Underlying()); t := typ.(type) {
		case *Basic:
			if isString(t) && id == _Len {
				if x.mode == constant_ {
					mode = constant_
					val = constant.MakeInt64(int64(len(constant.StringVal(x.val))))
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
			if !check.hasCallOrRecv {
				mode = constant_
				if t.len >= 0 {
					val = constant.MakeInt64(t.len)
				} else {
					val = constant.MakeUnknown()
				}
			}

		case *Slice, *Chan:
			mode = value

		case *Map:
			if id == _Len {
				mode = value
			}
		}

		if mode == invalid && typ != Typ[Invalid] {
			check.invalidArg(x.pos(), "%s for %s", x, bin.name)
			return
		}

		x.mode = mode
		x.typ = Typ[Int]
		x.val = val
		if check.Types != nil && mode != constant_ {
			check.recordBuiltinType(call.Fun, makeSig(x.typ, typ))
		}

	case _Close:
		// close(c)
		c, _ := x.typ.Underlying().(*Chan)
		if c == nil {
			check.invalidArg(x.pos(), "%s is not a channel", x)
			return
		}
		if c.dir == RecvOnly {
			check.invalidArg(x.pos(), "%s must not be a receive-only channel", x)
			return
		}

		x.mode = novalue
		if check.Types != nil {
			check.recordBuiltinType(call.Fun, makeSig(nil, c))
		}

	case _Complex:
		// complex(x, y floatT) complexT
		var y operand
		arg(&y, 1)
		if y.mode == invalid {
			return
		}

		// convert or check untyped arguments
		d := 0
		if isUntyped(x.typ) {
			d |= 1
		}
		if isUntyped(y.typ) {
			d |= 2
		}
		switch d {
		case 0:
			// x and y are typed => nothing to do
		case 1:
			// only x is untyped => convert to type of y
			check.convertUntyped(x, y.typ)
		case 2:
			// only y is untyped => convert to type of x
			check.convertUntyped(&y, x.typ)
		case 3:
			// x and y are untyped =>
			// 1) if both are constants, convert them to untyped
			//    floating-point numbers if possible,
			// 2) if one of them is not constant (possible because
			//    it contains a shift that is yet untyped), convert
			//    both of them to float64 since they must have the
			//    same type to succeed (this will result in an error
			//    because shifts of floats are not permitted)
			if x.mode == constant_ && y.mode == constant_ {
				toFloat := func(x *operand) {
					if isNumeric(x.typ) && constant.Sign(constant.Imag(x.val)) == 0 {
						x.typ = Typ[UntypedFloat]
					}
				}
				toFloat(x)
				toFloat(&y)
			} else {
				check.convertUntyped(x, Typ[Float64])
				check.convertUntyped(&y, Typ[Float64])
				// x and y should be invalid now, but be conservative
				// and check below
			}
		}
		if x.mode == invalid || y.mode == invalid {
			return
		}

		// both argument types must be identical
		if !Identical(x.typ, y.typ) {
			check.invalidArg(x.pos(), "mismatched types %s and %s", x.typ, y.typ)
			return
		}

		// the argument types must be of floating-point type
		if !isFloat(x.typ) {
			check.invalidArg(x.pos(), "arguments have type %s, expected floating-point", x.typ)
			return
		}

		// if both arguments are constants, the result is a constant
		if x.mode == constant_ && y.mode == constant_ {
			x.val = constant.BinaryOp(constant.ToFloat(x.val), token.ADD, constant.MakeImag(constant.ToFloat(y.val)))
		} else {
			x.mode = value
		}

		// determine result type
		var res BasicKind
		switch x.typ.Underlying().(*Basic).kind {
		case Float32:
			res = Complex64
		case Float64:
			res = Complex128
		case UntypedFloat:
			res = UntypedComplex
		default:
			unreachable()
		}
		resTyp := Typ[res]

		if check.Types != nil && x.mode != constant_ {
			check.recordBuiltinType(call.Fun, makeSig(resTyp, x.typ, x.typ))
		}

		x.typ = resTyp

	case _Copy:
		// copy(x, y []T) int
		var dst Type
		if t, _ := x.typ.Underlying().(*Slice); t != nil {
			dst = t.elem
		}

		var y operand
		arg(&y, 1)
		if y.mode == invalid {
			return
		}
		var src Type
		switch t := y.typ.Underlying().(type) {
		case *Basic:
			if isString(y.typ) {
				src = universeByte
			}
		case *Slice:
			src = t.elem
		}

		if dst == nil || src == nil {
			check.invalidArg(x.pos(), "copy expects slice arguments; found %s and %s", x, &y)
			return
		}

		if !Identical(dst, src) {
			check.invalidArg(x.pos(), "arguments to copy %s and %s have different element types %s and %s", x, &y, dst, src)
			return
		}

		if check.Types != nil {
			check.recordBuiltinType(call.Fun, makeSig(Typ[Int], x.typ, y.typ))
		}
		x.mode = value
		x.typ = Typ[Int]

	case _Delete:
		// delete(m, k)
		m, _ := x.typ.Underlying().(*Map)
		if m == nil {
			check.invalidArg(x.pos(), "%s is not a map", x)
			return
		}
		arg(x, 1) // k
		if x.mode == invalid {
			return
		}

		if !x.assignableTo(check, m.key, nil) {
			check.invalidArg(x.pos(), "%s is not assignable to %s", x, m.key)
			return
		}

		x.mode = novalue
		if check.Types != nil {
			check.recordBuiltinType(call.Fun, makeSig(nil, m, m.key))
		}

	case _Imag, _Real:
		// imag(complexT) floatT
		// real(complexT) floatT

		// convert or check untyped argument
		if isUntyped(x.typ) {
			if x.mode == constant_ {
				// an untyped constant number can alway be considered
				// as a complex constant
				if isNumeric(x.typ) {
					x.typ = Typ[UntypedComplex]
				}
			} else {
				// an untyped non-constant argument may appear if
				// it contains a (yet untyped non-constant) shift
				// expression: convert it to complex128 which will
				// result in an error (shift of complex value)
				check.convertUntyped(x, Typ[Complex128])
				// x should be invalid now, but be conservative and check
				if x.mode == invalid {
					return
				}
			}
		}

		// the argument must be of complex type
		if !isComplex(x.typ) {
			check.invalidArg(x.pos(), "argument has type %s, expected complex type", x.typ)
			return
		}

		// if the argument is a constant, the result is a constant
		if x.mode == constant_ {
			if id == _Real {
				x.val = constant.Real(x.val)
			} else {
				x.val = constant.Imag(x.val)
			}
		} else {
			x.mode = value
		}

		// determine result type
		var res BasicKind
		switch x.typ.Underlying().(*Basic).kind {
		case Complex64:
			res = Float32
		case Complex128:
			res = Float64
		case UntypedComplex:
			res = UntypedFloat
		default:
			unreachable()
		}
		resTyp := Typ[res]

		if check.Types != nil && x.mode != constant_ {
			check.recordBuiltinType(call.Fun, makeSig(resTyp, x.typ))
		}

		x.typ = resTyp

	case _Make:
		// make(T, n)
		// make(T, n, m)
		// (no argument evaluated yet)
		arg0 := call.Args[0]
		T := check.typ(arg0)
		if T == Typ[Invalid] {
			return
		}

		var min int // minimum number of arguments
		switch T.Underlying().(type) {
		case *Slice:
			min = 2
		case *Map, *Chan:
			min = 1
		default:
			check.invalidArg(arg0.Pos(), "cannot make %s; type must be slice, map, or channel", arg0)
			return
		}
		if nargs < min || min+1 < nargs {
			check.errorf(call.Pos(), "%v expects %d or %d arguments; found %d", call, min, min+1, nargs)
			return
		}
		var sizes []int64 // constant integer arguments, if any
		for _, arg := range call.Args[1:] {
			if s, ok := check.index(arg, -1); ok && s >= 0 {
				sizes = append(sizes, s)
			}
		}
		if len(sizes) == 2 && sizes[0] > sizes[1] {
			check.invalidArg(call.Args[1].Pos(), "length and capacity swapped")
			// safe to continue
		}
		x.mode = value
		x.typ = T
		if check.Types != nil {
			params := [...]Type{T, Typ[Int], Typ[Int]}
			check.recordBuiltinType(call.Fun, makeSig(x.typ, params[:1+len(sizes)]...))
		}

	case _New:
		// new(T)
		// (no argument evaluated yet)
		T := check.typ(call.Args[0])
		if T == Typ[Invalid] {
			return
		}

		x.mode = value
		x.typ = &Pointer{base: T}
		if check.Types != nil {
			check.recordBuiltinType(call.Fun, makeSig(x.typ, T))
		}

	case _Panic:
		// panic(x)
		// record panic call if inside a function with result parameters
		// (for use in Checker.isTerminating)
		if check.sig != nil && check.sig.results.Len() > 0 {
			// function has result parameters
			p := check.isPanic
			if p == nil {
				// allocate lazily
				p = make(map[*ast.CallExpr]bool)
				check.isPanic = p
			}
			p[call] = true
		}

		check.assignment(x, &emptyInterface, "argument to panic")
		if x.mode == invalid {
			return
		}

		x.mode = novalue
		if check.Types != nil {
			check.recordBuiltinType(call.Fun, makeSig(nil, &emptyInterface))
		}

	case _Print, _Println:
		// print(x, y, ...)
		// println(x, y, ...)
		var params []Type
		if nargs > 0 {
			params = make([]Type, nargs)
			for i := 0; i < nargs; i++ {
				if i > 0 {
					arg(x, i) // first argument already evaluated
				}
				check.assignment(x, nil, "argument to "+predeclaredFuncs[id].name)
				if x.mode == invalid {
					// TODO(gri) "use" all arguments?
					return
				}
				params[i] = x.typ
			}
		}

		x.mode = novalue
		if check.Types != nil {
			check.recordBuiltinType(call.Fun, makeSig(nil, params...))
		}

	case _Recover:
		// recover() interface{}
		x.mode = value
		x.typ = &emptyInterface
		if check.Types != nil {
			check.recordBuiltinType(call.Fun, makeSig(x.typ))
		}

	case _Alignof:
		// unsafe.Alignof(x T) uintptr
		check.assignment(x, nil, "argument to unsafe.Alignof")
		if x.mode == invalid {
			return
		}

		x.mode = constant_
		x.val = constant.MakeInt64(check.conf.alignof(x.typ))
		x.typ = Typ[Uintptr]
		// result is constant - no need to record signature

	case _Offsetof:
		// unsafe.Offsetof(x T) uintptr, where x must be a selector
		// (no argument evaluated yet)
		arg0 := call.Args[0]
		selx, _ := unparen(arg0).(*ast.SelectorExpr)
		if selx == nil {
			check.invalidArg(arg0.Pos(), "%s is not a selector expression", arg0)
			check.use(arg0)
			return
		}

		check.expr(x, selx.X)
		if x.mode == invalid {
			return
		}

		base := derefStructPtr(x.typ)
		sel := selx.Sel.Name
		obj, index, indirect := LookupFieldOrMethod(base, false, check.pkg, sel)
		switch obj.(type) {
		case nil:
			check.invalidArg(x.pos(), "%s has no single field %s", base, sel)
			return
		case *Func:
			// TODO(gri) Using derefStructPtr may result in methods being found
			// that don't actually exist. An error either way, but the error
			// message is confusing. See: https://play.golang.org/p/al75v23kUy ,
			// but go/types reports: "invalid argument: x.m is a method value".
			check.invalidArg(arg0.Pos(), "%s is a method value", arg0)
			return
		}
		if indirect {
			check.invalidArg(x.pos(), "field %s is embedded via a pointer in %s", sel, base)
			return
		}

		// TODO(gri) Should we pass x.typ instead of base (and indirect report if derefStructPtr indirected)?
		check.recordSelection(selx, FieldVal, base, obj, index, false)

		offs := check.conf.offsetof(base, index)
		x.mode = constant_
		x.val = constant.MakeInt64(offs)
		x.typ = Typ[Uintptr]
		// result is constant - no need to record signature

	case _Sizeof:
		// unsafe.Sizeof(x T) uintptr
		check.assignment(x, nil, "argument to unsafe.Sizeof")
		if x.mode == invalid {
			return
		}

		x.mode = constant_
		x.val = constant.MakeInt64(check.conf.sizeof(x.typ))
		x.typ = Typ[Uintptr]
		// result is constant - no need to record signature

	case _Assert:
		// assert(pred) causes a typechecker error if pred is false.
		// The result of assert is the value of pred if there is no error.
		// Note: assert is only available in self-test mode.
		if x.mode != constant_ || !isBoolean(x.typ) {
			check.invalidArg(x.pos(), "%s is not a boolean constant", x)
			return
		}
		if x.val.Kind() != constant.Bool {
			check.errorf(x.pos(), "internal error: value of %s should be a boolean constant", x)
			return
		}
		if !constant.BoolVal(x.val) {
			check.errorf(call.Pos(), "%v failed", call)
			// compile-time assertion failure - safe to continue
		}
		// result is constant - no need to record signature

	case _Trace:
		// trace(x, y, z, ...) dumps the positions, expressions, and
		// values of its arguments. The result of trace is the value
		// of the first argument.
		// Note: trace is only available in self-test mode.
		// (no argument evaluated yet)
		if nargs == 0 {
			check.dump("%v: trace() without arguments", call.Pos())
			x.mode = novalue
			break
		}
		var t operand
		x1 := x
		for _, arg := range call.Args {
			check.rawExpr(x1, arg, nil) // permit trace for types, e.g.: new(trace(T))
			check.dump("%v: %s", x1.pos(), x1)
			x1 = &t // use incoming x only for first argument
		}
		// trace is only available in test mode - no need to record signature

	default:
		unreachable()
	}

	return true
}

// makeSig makes a signature for the given argument and result types.
// Default types are used for untyped arguments, and res may be nil.
func makeSig(res Type, args ...Type) *Signature {
	list := make([]*Var, len(args))
	for i, param := range args {
		list[i] = NewVar(token.NoPos, nil, "", Default(param))
	}
	params := NewTuple(list...)
	var result *Tuple
	if res != nil {
		assert(!isUntyped(res))
		result = NewTuple(NewVar(token.NoPos, nil, "", res))
	}
	return &Signature{params: params, results: result}
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

// unparen returns e with any enclosing parentheses stripped.
func unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}
