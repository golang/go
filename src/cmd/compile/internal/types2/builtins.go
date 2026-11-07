// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of built-in function calls.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	"go/token"
	. "internal/types/errors"
)

// For each built-in there is a Checker method of the form:
//
// func (check *Checker) _builtin(z *operand, call *syntax.CallExpr)
//
// which implements (argument) type checking for the given call.
// The result is returned via z; z is invalid when the function is called.
// If type-checking is successful, the function needs to set z to a valid mode.
// If type-checking fails, z must be invalid upon return.

// append(s S, x ...E) S, where E is the element type of S
func (check *Checker) _Append(z *operand, call *syntax.CallExpr) {
	// spec: "The variadic function append appends zero or more values x to
	// a slice s of type S and returns the resulting slice, also of type S.
	// The values x are passed to a parameter of type ...E where E is the
	// element type of S and the respective parameter passing rules apply.
	// As a special case, append also accepts a first argument assignable
	// to type []byte with a second argument of string type followed by ... .
	// This form appends the bytes of the string."

	x, args, nargs := check.arg1(call, 1, true)
	if x == nil {
		return
	}

	// In either case, the first argument must be a slice; in particular it
	// cannot be the predeclared nil value. Note that nil is not excluded by
	// the assignability requirement alone for the special case (go.dev/issue/76220).
	// spec: "If S is a type parameter, all types in its type set
	// must have the same underlying slice type []E."
	E, err := sliceElem(x)
	if err != nil {
		check.errorf(x, InvalidAppend, "invalid append: %s", err.format(check))
		check.use(call.ArgList[1:]...)
		return
	}

	if args == nil { // first argument is evaluated and is a slice
		// process arguments so we can create a signature later
		args = make([]*operand, nargs)
		args[0] = x
		if nargs == 2 && hasDots(call) {
			// append(x, s...)
			args[1] = new(operand)
			check.expr(newTarget(x.typ(), "argument to append"), args[1], call.ArgList[1])
		} else {
			// append(x, e1, e2, e3)
			for i := 1; i < nargs; i++ {
				// call.ArgList[i] could be a generic function value;
				// like any argument, it is assigned to a parameter variable
				args[i] = new(operand)
				check.genericExpr(newVarTarget(E, "argument to append"), args[i], call.ArgList[i])
			}
		}
	}

	// Handle append(bytes, y...) special case, where
	// the type set of y is {string} or {string, []byte}.
	var sig *Signature
	if nargs == 2 && hasDots(call) {
		if ok, _ := x.assignableTo(check, NewSlice(universeByte), nil); ok {
			y := args[1]
			hasString := false
			for _, u := range typeset(y.typ()) {
				if s, _ := u.(*Slice); s != nil && Identical(s.elem, universeByte) {
					// typeset ⊇ {[]byte}
				} else if u != nil && isString(u) {
					// typeset ⊇ {string}
					hasString = true
				} else {
					y = nil
					break
				}
			}
			if y != nil && hasString {
				// setting the signature also signals that we're done
				sig = makeSig(x.typ(), x.typ(), y.typ())
				sig.variadic = true
			}
		}
	}

	// general case
	if sig == nil {
		// check arguments by creating custom signature
		sig = makeSig(x.typ(), x.typ(), NewSlice(E)) // []E required for variadic signature
		sig.variadic = true
		check.arguments(call, sig, nil, nil, args, nil) // discard result (we know the result type)
		// ok to continue even if check.arguments reported errors
	}

	z.mode_ = value
	z.typ_ = x.typ()
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, sig)
	}
}

// cap(x) int
// len(x) int
func (check *Checker) _CapLen(z *operand, call *syntax.CallExpr, isLen bool, name string) {
	// For len(a) and cap(a) we need to know if a contains any function calls or
	// receive operations. Save/restore current setting and set hasCallOrRecv to
	// false for the evaluation of the call so that we can check it afterwards.
	defer func(b bool) {
		check.hasCallOrRecv = b
	}(check.hasCallOrRecv)
	check.hasCallOrRecv = false

	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	switch t := arrayPtrDeref(x.typ().Underlying()).(type) {
	case *Basic:
		if isString(t) && isLen {
			if x.mode() == constant_ {
				z.mode_ = constant_
				z.val = constant.MakeInt64(constant.StringLen(x.val))
			} else {
				z.mode_ = value
			}
		}

	case *Array:
		z.mode_ = value
		// spec: "The expressions len(s) and cap(s) are constants
		// if the type of s is an array or pointer to an array and
		// the expression s does not contain channel receives or
		// function calls; in this case s is not evaluated."
		if !check.hasCallOrRecv {
			z.mode_ = constant_
			if t.len >= 0 {
				z.val = constant.MakeInt64(t.len)
			} else {
				z.val = constant.MakeUnknown()
			}
		}

	case *Slice, *Chan:
		z.mode_ = value

	case *Map:
		if isLen {
			z.mode_ = value
		}

	case *Interface:
		if !isTypeParam(x.typ()) {
			break
		}
		if underIs(x.typ(), func(u Type) bool {
			switch t := arrayPtrDeref(u).(type) {
			case *Basic:
				if isString(t) && isLen {
					return true
				}
			case *Array, *Slice, *Chan:
				return true
			case *Map:
				if isLen {
					return true
				}
			}
			return false
		}) {
			z.mode_ = value
		}
	}

	if !z.isValid() {
		// avoid error if the argument's underlying type is invalid
		if isValid(x.typ().Underlying()) {
			code := InvalidCap
			if isLen {
				code = InvalidLen
			}
			check.errorf(x, code, invalidArg+"%s for built-in %s", x, name)
		}
		return
	}

	z.typ_ = Typ[Int]
	if check.recordTypes() && z.mode_ != constant_ {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), x.typ()))
	}
}

// cap(x) int
func (check *Checker) _Cap(z *operand, call *syntax.CallExpr) {
	check._CapLen(z, call, false, "cap")
}

// clear(x)
func (check *Checker) _Clear(z *operand, call *syntax.CallExpr) {
	check.verifyVersionf(call.Fun, go1_21, "clear")

	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	if !underIs(x.typ(), func(u Type) bool {
		switch u.(type) {
		case *Map, *Slice:
			return true
		}
		check.errorf(x, InvalidClear, invalidArg+"cannot clear %s: argument must be (or constrained by) map or slice", x)
		return false
	}) {
		return
	}

	z.mode_ = novalue
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(nil, x.typ()))
	}
}

// close(x)
func (check *Checker) _Close(z *operand, call *syntax.CallExpr) {
	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	if !underIs(x.typ(), func(u Type) bool {
		uch, _ := u.(*Chan)
		if uch == nil {
			check.errorf(x, InvalidClose, invalidOp+"cannot close non-channel %s", x)
			return false
		}
		if uch.dir == RecvOnly {
			check.errorf(x, InvalidClose, invalidOp+"cannot close receive-only channel %s", x)
			return false
		}
		return true
	}) {
		return
	}

	z.mode_ = novalue
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(nil, x.typ()))
	}
}

// complex(x, y floatT) complexT
func (check *Checker) _Complex(z *operand, call *syntax.CallExpr) {
	x, y, ok := check.argn(call, 2)
	if !ok {
		return
	}

	// convert or check untyped arguments
	d := 0
	if x.isUntyped() {
		d |= 1
	}
	if y.isUntyped() {
		d |= 2
	}
	switch d {
	case 0:
		// x and y are typed => nothing to do
	case 1:
		// only x is untyped => convert to type of y
		check.convertUntyped(x, y.typ())
	case 2:
		// only y is untyped => convert to type of x
		check.convertUntyped(y, x.typ())
	case 3:
		// x and y are untyped =>
		// 1) if both are constants, convert them to untyped
		//    floating-point numbers if possible,
		// 2) if one of them is not constant (possible because
		//    it contains a shift that is yet untyped), convert
		//    both of them to float64 since they must have the
		//    same type to succeed (this will result in an error
		//    because shifts of floats are not permitted)
		if x.mode() == constant_ && y.mode() == constant_ {
			toFloat := func(x *operand) {
				if isNumeric(x.typ()) && constant.Sign(constant.Imag(x.val)) == 0 {
					x.typ_ = Typ[UntypedFloat]
				}
			}
			toFloat(x)
			toFloat(y)
		} else {
			check.convertUntyped(x, Typ[Float64])
			check.convertUntyped(y, Typ[Float64])
			// x and y should be invalid now, but be conservative
			// and check below
		}
	}
	if !x.isValid() || !y.isValid() {
		return
	}

	// both argument types must be identical
	if !Identical(x.typ(), y.typ()) {
		check.errorf(x, InvalidComplex, invalidOp+"%v (mismatched types %s and %s)", call, x.typ(), y.typ())
		return
	}

	// the argument types must be of floating-point type
	// (applyTypeFunc never calls f with a type parameter)
	f := func(typ Type) Type {
		assert(!isTypeParam(typ))
		if t, _ := typ.Underlying().(*Basic); t != nil {
			switch t.kind {
			case Float32:
				return Typ[Complex64]
			case Float64:
				return Typ[Complex128]
			case UntypedFloat:
				return Typ[UntypedComplex]
			}
		}
		return nil
	}
	z.typ_ = check.applyTypeFunc(f, x, _Complex)
	if z.typ() == nil {
		check.errorf(x, InvalidComplex, invalidArg+"arguments have type %s, expected floating-point", x.typ())
		return
	}

	// if both arguments are constants, the result is a constant
	z.mode_ = value
	if x.mode() == constant_ && y.mode() == constant_ {
		z.mode_ = constant_
		z.val = constant.BinaryOp(constant.ToFloat(x.val), token.ADD, constant.MakeImag(constant.ToFloat(y.val)))
	}

	if check.recordTypes() && z.mode() != constant_ {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), x.typ(), x.typ()))
	}
}

// copy(x, y []E) int
func (check *Checker) _Copy(z *operand, call *syntax.CallExpr) {
	// spec: "The function copy copies slice elements from a source src to a destination
	// dst and returns the number of elements copied. Both arguments must have identical
	// element type E and must be assignable to a slice of type []E.
	// The number of elements copied is the minimum of len(src) and len(dst).
	// As a special case, copy also accepts a destination argument assignable to type
	// []byte with a source argument of a string type.
	// This form copies the bytes from the string into the byte slice."

	x, args, _ := check.arg1(call, 2, false)
	if x == nil {
		return
	}

	// In either case, the first argument must be a slice; in particular it
	// cannot be the predeclared nil value. Note that nil is not excluded by
	// the assignability requirement alone for the special case (go.dev/issue/79687).
	// spec: "If the type of one or both arguments is a type parameter, all types
	// in their respective type sets must have the same underlying slice type []E."
	dstE, err := sliceElem(x)
	if err != nil {
		check.errorf(x, InvalidCopy, "invalid copy: %s", err.format(check))
		check.use(call.ArgList[1:]...)
		return
	}

	var y *operand
	if args == nil { // first argument is evaluated and is a slice
		y = new(operand)
		check.expr(newTarget(x.typ(), "argument to copy"), y, call.ArgList[1])
		if !y.isValid() {
			return
		}
	} else {
		y = args[1]
	}

	// get special case out of the way
	var special bool
	if ok, _ := x.assignableTo(check, NewSlice(universeByte), nil); ok {
		special = true
		for _, u := range typeset(y.typ()) {
			if s, _ := u.(*Slice); s != nil && Identical(s.elem, universeByte) {
				// typeset ⊇ {[]byte}
			} else if u != nil && isString(u) {
				// typeset ⊇ {string}
			} else {
				special = false
				break
			}
		}
	}

	// general case
	if !special {
		srcE, err := sliceElem(y)
		if err != nil {
			// If we have a string, for a better error message proceed with byte element type.
			if !allString(y.typ()) {
				check.errorf(y, InvalidCopy, "invalid copy: %s", err.format(check))
				return
			}
			srcE = universeByte
		}
		if !Identical(dstE, srcE) {
			check.errorf(x, InvalidCopy, "invalid copy: arguments %s and %s have different element types %s and %s", x, y, dstE, srcE)
			return
		}
	}

	z.mode_ = value
	z.typ_ = Typ[Int]
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), x.typ(), y.typ()))
	}
}

// delete(map_, key)
func (check *Checker) _Delete(z *operand, call *syntax.CallExpr) {
	x, args, _ := check.arg1(call, 2, false)
	if x == nil {
		return
	}

	// map_ must be a map type or a type parameter describing map types.
	// The key cannot be a type parameter for now.
	map_ := x.typ()
	var key Type
	if !underIs(map_, func(u Type) bool {
		map_, _ := u.(*Map)
		if map_ == nil {
			check.errorf(x, InvalidDelete, invalidArg+"%s is not a map", x)
			return false
		}
		if key != nil && !Identical(map_.key, key) {
			check.errorf(x, InvalidDelete, invalidArg+"maps of %s must have identical key types", x)
			return false
		}
		key = map_.key
		return true
	}) {
		check.use(call.ArgList[1:]...)
		return
	}
	if args == nil {
		// call.ArgList[1] could be a generic function value
		check.genericExpr(newTarget(key, "argument to delete"), x, call.ArgList[1])
	} else {
		*x = *args[1] // key
	}
	check.assignment(x, key, "argument to delete")
	if !x.isValid() {
		return
	}

	z.mode_ = novalue
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(nil, map_, key))
	}
}

// imag(complexT) floatT
// real(complexT) floatT
func (check *Checker) _ImagReal(z *operand, call *syntax.CallExpr, id builtinId) {
	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	// convert or check untyped argument
	if x.isUntyped() {
		if x.mode() == constant_ {
			// an untyped constant number can always be considered
			// as a complex constant
			if isNumeric(x.typ()) {
				x.typ_ = Typ[UntypedComplex]
			}
		} else {
			// an untyped non-constant argument may appear if
			// it contains a (yet untyped non-constant) shift
			// expression: convert it to complex128 which will
			// result in an error (shift of complex value)
			check.convertUntyped(x, Typ[Complex128])
			// x should be invalid now, but be conservative and check
			if !x.isValid() {
				return
			}
		}
	}

	// the argument must be of complex type
	// (applyTypeFunc never calls f with a type parameter)
	f := func(typ Type) Type {
		assert(!isTypeParam(typ))
		if t, _ := typ.Underlying().(*Basic); t != nil {
			switch t.kind {
			case Complex64:
				return Typ[Float32]
			case Complex128:
				return Typ[Float64]
			case UntypedComplex:
				return Typ[UntypedFloat]
			}
		}
		return nil
	}
	z.typ_ = check.applyTypeFunc(f, x, id)
	if z.typ() == nil {
		code := InvalidImag
		if id == _Real {
			code = InvalidReal
		}
		check.errorf(x, code, invalidArg+"argument has type %s, expected complex type", x.typ())
		return
	}

	// if the argument is a constant, the result is a constant
	z.mode_ = value
	if x.mode() == constant_ {
		z.mode_ = constant_
		if id == _Real {
			z.val = constant.Real(x.val)
		} else {
			z.val = constant.Imag(x.val)
		}
	}

	if check.recordTypes() && z.mode() != constant_ {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), x.typ()))
	}
}

// imag(complexT) floatT
func (check *Checker) _Imag(z *operand, call *syntax.CallExpr) {
	check._ImagReal(z, call, _Imag)
}

// len(x) int
func (check *Checker) _Len(z *operand, call *syntax.CallExpr) {
	check._CapLen(z, call, true, "len")
}

// make(T, n)
// make(T, n, m)
func (check *Checker) _Make(z *operand, call *syntax.CallExpr) {
	argList := call.ArgList
	nargs := len(argList)
	if !check.argCount(call, nargs, 1, true) {
		return
	}

	arg0 := argList[0]
	T := check.varType(arg0)
	if !isValid(T) {
		return
	}

	u, err := commonUnder(T, func(_, u Type) *typeError {
		switch u.(type) {
		case *Slice, *Map, *Chan:
			return nil // ok
		case nil:
			return typeErrorf("no specific type")
		default:
			return typeErrorf("type must be slice, map, or channel")
		}
	})
	if err != nil {
		check.errorf(arg0, InvalidMake, invalidArg+"cannot make %s: %s", arg0, err.format(check))
		return
	}

	var min int // minimum number of arguments
	switch u.(type) {
	case *Slice:
		min = 2
	case *Map, *Chan:
		min = 1
	default:
		// any other type was excluded above
		panic("unreachable")
	}
	if nargs < min || min+1 < nargs {
		check.errorf(call, WrongArgCount, invalidOp+"%v expects %d or %d arguments; found %d", call, min, min+1, nargs)
		return
	}

	types := []Type{T}
	var sizes []int64 // constant integer arguments, if any
	for _, arg := range argList[1:] {
		typ, size := check.index(arg, -1) // ok to continue with typ == Typ[Invalid]
		types = append(types, typ)
		if size >= 0 {
			sizes = append(sizes, size)
		}
	}
	if len(sizes) == 2 && sizes[0] > sizes[1] {
		check.error(argList[1], SwappedMakeArgs, invalidArg+"length and capacity swapped")
		// safe to continue
	}
	z.mode_ = value
	z.typ_ = T
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), types...))
	}
}

// max(x T, ...) T
// min(x T, ...) T
func (check *Checker) _MaxMin(z *operand, call *syntax.CallExpr, isMin bool, name string) {
	check.verifyVersionf(call.Fun, go1_21, "built-in %s", name)

	args := check.exprList(call.ArgList)
	for _, a := range args {
		if !a.isValid() {
			return
		}
	}
	if !check.argCount(call, len(args), 1, true) {
		return
	}

	op := token.GTR
	if isMin {
		op = token.LSS
	}

	var x operand
	for i, a := range args {
		if !allOrdered(a.typ()) {
			check.errorf(a, InvalidMinMaxOperand, invalidArg+"%s cannot be ordered", a)
			return
		}

		if i == 0 {
			x = *a
		} else {
			check.matchTypes(&x, a)
			if !x.isValid() {
				return
			}

			if !Identical(x.typ(), a.typ()) {
				check.errorf(a, MismatchedTypes, invalidArg+"mismatched types %s (previous argument) and %s (type of %s)", x.typ(), a.typ(), a.expr)
				return
			}

			if x.mode() == constant_ && a.mode() == constant_ {
				if constant.Compare(a.val, op, x.val) {
					x = *a
				}
			} else {
				x.mode_ = value
			}
		}
	}

	// If nargs == 1, make sure x.mode is either a value or a constant.
	if x.mode() != constant_ {
		x.mode_ = value
		// A value must not be untyped.
		check.assignment(&x, &emptyInterface, "argument to built-in "+name)
		if !x.isValid() {
			return
		}
	}

	// Use the final type computed above for all arguments.
	for _, a := range args {
		check.updateExprType(a.expr, x.typ(), true)
	}

	*z = x
	if check.recordTypes() && z.mode() != constant_ {
		types := make([]Type, len(args))
		for i := range types {
			types[i] = z.typ()
		}
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), types...))
	}
}

// max(x T, ...) T
func (check *Checker) _Max(z *operand, call *syntax.CallExpr) {
	check._MaxMin(z, call, false, "max")
}

// min(x T, ...) T
func (check *Checker) _Min(z *operand, call *syntax.CallExpr) {
	check._MaxMin(z, call, true, "min")
}

// new(T)
// new(expr)
func (check *Checker) _New(z *operand, call *syntax.CallExpr) {
	if !check.argCount(call, len(call.ArgList), 1, false) {
		return
	}

	arg := call.ArgList[0]
	x := new(operand)
	check.exprOrType(x, arg, false)
	check.exclude(x, 1<<novalue|1<<builtin)
	switch x.mode() {
	case invalid:
		return
	case typexpr:
		// new(T)
		check.validVarType(arg, x.typ())
	default:
		// new(expr)
		if x.isUntyped() {
			// check for overflow and untyped nil
			check.assignment(x, nil, "argument to new")
			if !x.isValid() {
				return
			}
			assert(isTyped(x.typ()))
		}
		// report version error only if there are no other errors
		check.verifyVersionf(call.Fun, go1_26, "new(%s)", arg)
	}

	T := x.typ()
	z.mode_ = value
	z.typ_ = NewPointer(T)
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), T))
	}
}

// panic(x)
func (check *Checker) _Panic(z *operand, call *syntax.CallExpr) {
	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	// record panic call if inside a function with result parameters
	// (for use in Checker.isTerminating)
	if check.sig != nil && check.sig.results.Len() > 0 {
		// function has result parameters
		p := check.isPanic
		if p == nil {
			// allocate lazily
			p = make(map[*syntax.CallExpr]bool)
			check.isPanic = p
		}
		p[call] = true
	}

	check.assignment(x, &emptyInterface, "argument to panic")
	if !x.isValid() {
		return
	}

	z.mode_ = novalue
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(nil, &emptyInterface))
	}
}

// print(x, y, ...)
// println(x, y, ...)
func (check *Checker) _PrintLn(z *operand, call *syntax.CallExpr, name string) {
	args := check.exprList(call.ArgList)
	for _, a := range args {
		if !a.isValid() {
			return
		}
	}

	var params []Type
	if nargs := len(args); nargs > 0 {
		params = make([]Type, nargs)
		for i, a := range args {
			check.assignment(a, nil, "argument to built-in "+name)
			if !a.isValid() {
				return
			}
			params[i] = a.typ()
		}
	}

	z.mode_ = novalue
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(nil, params...))
	}
}

// print(x, y, ...)
func (check *Checker) _Print(z *operand, call *syntax.CallExpr) {
	check._PrintLn(z, call, "print")
}

// println(x, y, ...)
func (check *Checker) _Println(z *operand, call *syntax.CallExpr) {
	check._PrintLn(z, call, "println")
}

// real(complexT) floatT
func (check *Checker) _Real(z *operand, call *syntax.CallExpr) {
	check._ImagReal(z, call, _Real)
}

// recover() interface{}
func (check *Checker) _Recover(z *operand, call *syntax.CallExpr) {
	if _, _, ok := check.argn(call, 0); !ok {
		return
	}

	z.mode_ = value
	z.typ_ = &emptyInterface
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ()))
	}
}

// unsafe.Add(ptr unsafe.Pointer, len IntegerType) unsafe.Pointer
func (check *Checker) _Add(z *operand, call *syntax.CallExpr) {
	check.verifyVersionf(call.Fun, go1_17, "unsafe.Add")

	x, y, ok := check.argn(call, 2)
	if !ok {
		return
	}

	check.assignment(x, Typ[UnsafePointer], "argument to unsafe.Add")
	if !x.isValid() {
		return
	}

	if !check.isValidIndex(y, InvalidUnsafeAdd, "length", true) {
		return
	}

	z.mode_ = value
	z.typ_ = Typ[UnsafePointer]
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), z.typ(), y.typ()))
	}
}

// unsafe.Alignof(x T) uintptr
func (check *Checker) _Alignof(z *operand, call *syntax.CallExpr) {
	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	check.assignment(x, nil, "argument to unsafe.Alignof")
	if !x.isValid() {
		return
	}

	if check.hasVarSize(x.typ()) {
		z.mode_ = value
		if check.recordTypes() {
			check.recordBuiltinType(call.Fun, makeSig(Typ[Uintptr], x.typ()))
		}
	} else {
		z.mode_ = constant_
		z.val = constant.MakeInt64(check.conf.alignof(x.typ()))
		// result is constant - no need to record signature
	}
	z.typ_ = Typ[Uintptr]
}

// unsafe.Offsetof(x T) uintptr, where x must be a selector
func (check *Checker) _Offsetof(z *operand, call *syntax.CallExpr) {
	if !check.argCount(call, len(call.ArgList), 1, false) {
		return
	}

	arg0 := call.ArgList[0]
	selx, _ := syntax.Unparen(arg0).(*syntax.SelectorExpr)
	if selx == nil {
		check.errorf(arg0, BadOffsetofSyntax, invalidArg+"%s is not a selector expression", arg0)
		check.use(arg0)
		return
	}

	x := new(operand)
	check.expr(nil, x, selx.X)
	if !x.isValid() {
		return
	}

	base := derefStructPtr(x.typ())
	sel := selx.Sel.Value
	obj, index, indirect := lookupFieldOrMethod(base, false, check.pkg, sel, false)
	switch obj.(type) {
	case nil:
		check.errorf(x, MissingFieldOrMethod, invalidArg+"%s has no single field %s", base, sel)
		return
	case *Func:
		// TODO(gri) Using derefStructPtr may result in methods being found
		// that don't actually exist. An error either way, but the error
		// message is confusing. See: https://play.golang.org/p/al75v23kUy ,
		// but go/types reports: "invalid argument: x.m is a method value".
		check.errorf(arg0, InvalidOffsetof, invalidArg+"%s is a method value", arg0)
		return
	}
	if indirect {
		check.errorf(x, InvalidOffsetof, invalidArg+"field %s is embedded via a pointer in %s", sel, base)
		return
	}

	// TODO(gri) Should we pass x.typ instead of base (and have indirect report if derefStructPtr indirected)?
	check.recordSelection(selx, FieldVal, base, obj, index, false)

	// record the selector expression (was bug - go.dev/issue/47895)
	{
		mode := value
		if x.mode() == variable || indirect {
			mode = variable
		}
		check.record(&operand{mode, selx, obj.Type(), nil, 0})
	}

	// The field offset is considered a variable even if the field is declared before
	// the part of the struct which is variable-sized. This makes both the rules
	// simpler and also permits (or at least doesn't prevent) a compiler from re-
	// arranging struct fields if it wanted to.
	if check.hasVarSize(base) {
		z.mode_ = value
		if check.recordTypes() {
			check.recordBuiltinType(call.Fun, makeSig(Typ[Uintptr], obj.Type()))
		}
	} else {
		offs := check.conf.offsetof(base, index)
		if offs < 0 {
			check.errorf(x, TypeTooLarge, "%s is too large", x)
			return
		}
		z.mode_ = constant_
		z.val = constant.MakeInt64(offs)
		// result is constant - no need to record signature
	}
	z.typ_ = Typ[Uintptr]
}

// unsafe.Sizeof(x T) uintptr
func (check *Checker) _Sizeof(z *operand, call *syntax.CallExpr) {
	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	check.assignment(x, nil, "argument to unsafe.Sizeof")
	if !x.isValid() {
		return
	}

	if check.hasVarSize(x.typ()) {
		z.mode_ = value
		if check.recordTypes() {
			check.recordBuiltinType(call.Fun, makeSig(Typ[Uintptr], x.typ()))
		}
	} else {
		size := check.conf.sizeof(x.typ())
		if size < 0 {
			check.errorf(x, TypeTooLarge, "%s is too large", x)
			return
		}
		z.mode_ = constant_
		z.val = constant.MakeInt64(size)
		// result is constant - no need to record signature
	}
	z.typ_ = Typ[Uintptr]
}

// unsafe.Slice(ptr *T, len IntegerType) []T
func (check *Checker) _Slice(z *operand, call *syntax.CallExpr) {
	check.verifyVersionf(call.Fun, go1_17, "unsafe.Slice")

	x, y, ok := check.argn(call, 2)
	if !ok {
		return
	}

	u, _ := commonUnder(x.typ(), nil)
	ptr, _ := u.(*Pointer)
	if ptr == nil {
		check.errorf(x, InvalidUnsafeSlice, invalidArg+"%s is not a pointer", x)
		return
	}

	if !check.isValidIndex(y, InvalidUnsafeSlice, "length", false) {
		return
	}

	z.mode_ = value
	z.typ_ = NewSlice(ptr.base)
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), ptr, y.typ()))
	}
}

// unsafe.SliceData(slice []T) *T
func (check *Checker) _SliceData(z *operand, call *syntax.CallExpr) {
	check.verifyVersionf(call.Fun, go1_20, "unsafe.SliceData")

	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	u, _ := commonUnder(x.typ(), nil)
	slice, _ := u.(*Slice)
	if slice == nil {
		check.errorf(x, InvalidUnsafeSliceData, invalidArg+"%s is not a slice", x)
		return
	}

	z.mode_ = value
	z.typ_ = NewPointer(slice.elem)
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), slice))
	}
}

// unsafe.String(ptr *byte, len IntegerType) string
func (check *Checker) _String(z *operand, call *syntax.CallExpr) {
	check.verifyVersionf(call.Fun, go1_20, "unsafe.String")

	x, y, ok := check.argn(call, 2)
	if !ok {
		return
	}

	check.assignment(x, NewPointer(universeByte), "argument to unsafe.String")
	if !x.isValid() {
		return
	}

	if !check.isValidIndex(y, InvalidUnsafeString, "length", false) {
		return
	}

	z.mode_ = value
	z.typ_ = Typ[String]
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), NewPointer(universeByte), y.typ()))
	}
}

// unsafe.StringData(str string) *byte
func (check *Checker) _StringData(z *operand, call *syntax.CallExpr) {
	check.verifyVersionf(call.Fun, go1_20, "unsafe.StringData")

	x, _, ok := check.argn(call, 1)
	if !ok {
		return
	}

	check.assignment(x, Typ[String], "argument to unsafe.StringData")
	if !x.isValid() {
		return
	}

	z.mode_ = value
	z.typ_ = NewPointer(universeByte)
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(z.typ(), Typ[String]))
	}
}

// assert(pred)
func (check *Checker) _Assert(z *operand, call *syntax.CallExpr) {
	// assert(pred) causes a typechecker error if pred is false.
	// The result of assert is the value of pred if there is no error.
	// Note: assert is only available in self-test mode.
	args := check.exprList(call.ArgList)
	if !check.argCount(call, len(args), 1, false) {
		return
	}
	x := args[0]
	if x.mode() != constant_ || !isBoolean(x.typ()) {
		check.errorf(x, Test, invalidArg+"%s is not a boolean constant", x)
		return
	}
	if x.val.Kind() != constant.Bool {
		check.internalErrorf(x, "value of %s should be a boolean constant", x)
		return
	}
	if !constant.BoolVal(x.val) {
		check.errorf(call, Test, "%v failed", call)
		// compile-time assertion failure - safe to continue
	}
	*z = *x
	// result is constant - no need to record signature
}

// trace(x, y, z, ...)
func (check *Checker) _Trace(z *operand, call *syntax.CallExpr) {
	// trace(x, y, z, ...) dumps the positions, expressions, and
	// values of its arguments. The result of trace is the value
	// of the first argument.
	// Note: trace is only available in self-test mode.
	argList := call.ArgList
	if len(argList) == 0 {
		check.dump("%v: trace() without arguments", atPos(call))
		z.mode_ = novalue
		return
	}
	var t operand
	x1 := z
	for _, arg := range argList {
		check.rawExpr(nil, x1, arg, false) // permit trace for types, e.g.: new(trace(T))
		check.dump("%v: %s", atPos(x1), x1)
		x1 = &t // use incoming z only for first argument
	}
	// trace is only available in test mode - no need to record signature
}

// ----------------------------------------------------------------------------
// Support functions

func (check *Checker) argCount(call *syntax.CallExpr, nargs, want int, variadic bool) bool {
	var msg string
	switch {
	case nargs < want:
		msg = "not enough"
	case !variadic && nargs > want:
		msg = "too many"
	default:
		return true
	}
	check.errorf(argErrPos(call), WrongArgCount, invalidOp+"%s arguments for %v (expected %d, found %d)", msg, call, want, nargs)
	return false
}

// arg1 evaluates the first argument of call (if any) and returns the first argument (if any),
// the evaluated arguments (if they can all be evaluated), and the total number of arguments(incl. non-evaluated arguments).
// If not all arguments can be evaluated, args will be nil.
// If there are not exactly n arguments, or if an evaluated argument is invalid,
// an error is reported and the result is (nil, nil, 0).
func (check *Checker) arg1(call *syntax.CallExpr, n int, variadic bool) (x *operand, args []*operand, nargs int) {
	argList := call.ArgList
	nargs = len(argList)
	// Careful: There is a tension between argument inference and tuple arguments to built-ins.
	if nargs == 1 {
		// The argument may be a tuple argument to a built-in and thus potentially
		// nargs != len(argList). We can evaluate argList upfront because e.g. {}
		// in delete({}) has no other argument from which to get a target type.
		args = check.exprList(argList)
		for _, a := range args {
			if !a.isValid() {
				return nil, nil, 0
			}
		}
		nargs = len(args) // correct nargs
		if nargs > 0 {
			x = args[0]
		}
	} else { // nargs != 1
		// No argument can be a tuple argument to a built-in and thus necessarily
		// nargs == len(argList). We cannot evaluate argList upfront because e.g. {}
		// in delete(m, {}) needs a target type from m.
		// Type the first argument (if available) upfront for convenience.
		if nargs > 0 {
			x = new(operand)
			check.expr(nil, x, argList[0])
			if !x.isValid() {
				check.use(argList[1:]...)
				return nil, nil, 0
			}
		}
		// args == nil in this case
	}
	if !check.argCount(call, nargs, n, variadic) {
		return nil, nil, 0
	}
	return x, args, nargs
}

// argn evaluates all arguments of call, and returns up to two in x and y.
// If there are not exactly n arguments, or if an argument is invalid,
// an error is reported and the result is (nil, nil, false).
// Otherwise, ok is set to true.
func (check *Checker) argn(call *syntax.CallExpr, n int) (x, y *operand, ok bool) {
	args := check.exprList(call.ArgList)
	for _, a := range args {
		if !a.isValid() {
			return
		}
	}
	if !check.argCount(call, len(args), n, false) {
		return
	}
	ok = true
	switch len(args) {
	case 0:
		// nothing to do
	case 1:
		x = args[0]
	case 2:
		x, y = args[0], args[1]
	default:
		panic("args does not handle more than 2 arguments")
	}
	return
}

// sliceElem returns the slice element type for a slice operand x
// or a type error if x is not a slice (or a type set of slices).
func sliceElem(x *operand) (Type, *typeError) {
	var E Type
	for _, u := range typeset(x.typ()) {
		s, _ := u.(*Slice)
		if s == nil {
			if x.isNil() {
				// Printing x in this case would just print "nil".
				// Special case this so we can emphasize "untyped".
				return nil, typeErrorf("argument must be a slice; have untyped nil")
			} else {
				return nil, typeErrorf("argument must be a slice; have %s", x)
			}
		}
		if E == nil {
			E = s.elem
		} else if !Identical(E, s.elem) {
			return nil, typeErrorf("mismatched slice element types %s and %s in %s", E, s.elem, x)
		}
	}
	return E, nil
}

// hasVarSize reports if the size of type t is variable due to type parameters
// or if the type is infinitely-sized due to a cycle for which the type has not
// yet been checked.
func (check *Checker) hasVarSize(t Type) bool {
	// Note: We could use Underlying here, but passing through the RHS may yield
	// better error messages and allows us to stash the result on each traversed
	// Named type.
	switch t := Unalias(t).(type) {
	case *Named:
		if t.stateHas(hasVarSize) {
			return t.varSize
		}

		if i, ok := check.objPathIdx[t.obj]; ok {
			cycle := check.objPath[i:]
			check.cycleError(cycle, firstInSrc(cycle))
			return true
		}

		obj := t.obj
		check.push(obj)
		defer check.pop()

		// Careful, we're inspecting t.fromRHS, so we need to unpack first.
		t.unpack()
		varSize := check.hasVarSize(t.rhs())

		// Special case for portable simd types that rewrite to unknown sizes.
		if pkg := obj.Pkg(); pkg != nil && pkg.Path() == "simd" && obj.Name() == "_simd" {
			varSize = true
		}

		t.mu.Lock()
		defer t.mu.Unlock()

		// Careful, t.varSize has lock-free readers. Since we might be racing
		// another call to hasVarSize, we have to avoid overwriting t.varSize.
		// Otherwise, the race detector will be tripped.
		if !t.stateHas(hasVarSize) {
			t.varSize = varSize
			t.setState(hasVarSize)
		}

		return varSize

	case *Array:
		// The array length is already computed. If it was a valid length, it
		// is constant; else, an error was reported in the computation.
		return check.hasVarSize(t.elem)

	case *Struct:
		for _, f := range t.fields {
			if check.hasVarSize(f.typ) {
				return true
			}
		}

	case *TypeParam:
		return true
	}

	return false
}

// applyTypeFunc applies f to x. If x is a type parameter,
// the result is a type parameter constrained by a new
// interface bound. The type bounds for that interface
// are computed by applying f to each of the type bounds
// of x. If any of these applications of f return nil,
// applyTypeFunc returns nil.
// If x is not a type parameter, the result is f(x).
func (check *Checker) applyTypeFunc(f func(Type) Type, x *operand, id builtinId) Type {
	if tp, _ := Unalias(x.typ()).(*TypeParam); tp != nil {
		// Test if t satisfies the requirements for the argument
		// type and collect possible result types at the same time.
		var terms []*Term
		if !tp.is(func(t *term) bool {
			if t == nil {
				return false
			}
			if r := f(t.typ); r != nil {
				terms = append(terms, NewTerm(t.tilde, r))
				return true
			}
			return false
		}) {
			return nil
		}

		// We can type-check this fine but we're introducing a synthetic
		// type parameter for the result. It's not clear what the API
		// implications are here. Report an error for 1.18 (see go.dev/issue/50912),
		// but continue type-checking.
		var code Code
		switch id {
		case _Real:
			code = InvalidReal
		case _Imag:
			code = InvalidImag
		case _Complex:
			code = InvalidComplex
		default:
			panic("unreachable")
		}
		check.softErrorf(x, code, "%s not supported as argument to built-in %s for go1.18 (see go.dev/issue/50937)", x, predeclaredFuncs[id].name)

		// Construct a suitable new type parameter for the result type.
		// The type parameter is placed in the current package so export/import
		// works as expected.
		tpar := NewTypeName(nopos, check.pkg, tp.obj.name, nil)
		ptyp := check.newTypeParam(tpar, NewInterfaceType(nil, []Type{NewUnion(terms)})) // assigns type to tpar as a side-effect
		ptyp.index = tp.index

		return ptyp
	}

	return f(x.typ())
}

// makeSig makes a signature for the given argument and result types.
// Default types are used for untyped arguments, and res may be nil.
func makeSig(res Type, args ...Type) *Signature {
	list := make([]*Var, len(args))
	for i, param := range args {
		list[i] = NewParam(nopos, nil, "", Default(param))
	}
	params := NewTuple(list...)
	var result *Tuple
	if res != nil {
		assert(!isUntyped(res))
		result = NewTuple(newVar(ResultVar, nopos, nil, "", res))
	}
	return &Signature{params: params, results: result}
}

// arrayPtrDeref returns A if typ is of the form *A and A is an array;
// otherwise it returns typ.
func arrayPtrDeref(typ Type) Type {
	if p, ok := Unalias(typ).(*Pointer); ok {
		if a, _ := p.base.Underlying().(*Array); a != nil {
			return a
		}
	}
	return typ
}
