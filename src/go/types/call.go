// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of call and selector expressions.

package types

import (
	"go/ast"
	"go/token"
)

func (check *Checker) call(x *operand, e *ast.CallExpr) exprKind {
	check.exprOrType(x, e.Fun)

	switch x.mode {
	case invalid:
		check.use(e.Args...)
		x.mode = invalid
		x.expr = e
		return statement

	case typexpr:
		// conversion
		T := x.typ
		x.mode = invalid
		switch n := len(e.Args); n {
		case 0:
			check.errorf(e.Rparen, "missing argument in conversion to %s", T)
		case 1:
			check.expr(x, e.Args[0])
			if x.mode != invalid {
				check.conversion(x, T)
			}
		default:
			check.errorf(e.Args[n-1].Pos(), "too many arguments in conversion to %s", T)
		}
		x.expr = e
		return conversion

	case builtin:
		id := x.id
		if !check.builtin(x, e, id) {
			x.mode = invalid
		}
		x.expr = e
		// a non-constant result implies a function call
		if x.mode != invalid && x.mode != constant {
			check.hasCallOrRecv = true
		}
		return predeclaredFuncs[id].kind

	default:
		// function/method call
		sig, _ := x.typ.Underlying().(*Signature)
		if sig == nil {
			check.invalidOp(x.pos(), "cannot call non-function %s", x)
			x.mode = invalid
			x.expr = e
			return statement
		}

		arg, n, _ := unpack(func(x *operand, i int) { check.expr(x, e.Args[i]) }, len(e.Args), false)
		if arg == nil {
			x.mode = invalid
			x.expr = e
			return statement
		}

		check.arguments(x, e, sig, arg, n)

		// determine result
		switch sig.results.Len() {
		case 0:
			x.mode = novalue
		case 1:
			x.mode = value
			x.typ = sig.results.vars[0].typ // unpack tuple
		default:
			x.mode = value
			x.typ = sig.results
		}
		x.expr = e
		check.hasCallOrRecv = true

		return statement
	}
}

// use type-checks each argument.
// Useful to make sure expressions are evaluated
// (and variables are "used") in the presence of other errors.
func (check *Checker) use(arg ...ast.Expr) {
	var x operand
	for _, e := range arg {
		check.rawExpr(&x, e, nil)
	}
}

// useGetter is like use, but takes a getter instead of a list of expressions.
// It should be called instead of use if a getter is present to avoid repeated
// evaluation of the first argument (since the getter was likely obtained via
// unpack, which may have evaluated the first argument already).
func (check *Checker) useGetter(get getter, n int) {
	var x operand
	for i := 0; i < n; i++ {
		get(&x, i)
	}
}

// A getter sets x as the i'th operand, where 0 <= i < n and n is the total
// number of operands (context-specific, and maintained elsewhere). A getter
// type-checks the i'th operand; the details of the actual check are getter-
// specific.
type getter func(x *operand, i int)

// unpack takes a getter get and a number of operands n. If n == 1, unpack
// calls the incoming getter for the first operand. If that operand is
// invalid, unpack returns (nil, 0, false). Otherwise, if that operand is a
// function call, or a comma-ok expression and allowCommaOk is set, the result
// is a new getter and operand count providing access to the function results,
// or comma-ok values, respectively. The third result value reports if it
// is indeed the comma-ok case. In all other cases, the incoming getter and
// operand count are returned unchanged, and the third result value is false.
//
// In other words, if there's exactly one operand that - after type-checking
// by calling get - stands for multiple operands, the resulting getter provides
// access to those operands instead.
//
// If the returned getter is called at most once for a given operand index i
// (including i == 0), that operand is guaranteed to cause only one call of
// the incoming getter with that i.
//
func unpack(get getter, n int, allowCommaOk bool) (getter, int, bool) {
	if n == 1 {
		// possibly result of an n-valued function call or comma,ok value
		var x0 operand
		get(&x0, 0)
		if x0.mode == invalid {
			return nil, 0, false
		}

		if t, ok := x0.typ.(*Tuple); ok {
			// result of an n-valued function call
			return func(x *operand, i int) {
				x.mode = value
				x.expr = x0.expr
				x.typ = t.At(i).typ
			}, t.Len(), false
		}

		if x0.mode == mapindex || x0.mode == commaok {
			// comma-ok value
			if allowCommaOk {
				a := [2]Type{x0.typ, Typ[UntypedBool]}
				return func(x *operand, i int) {
					x.mode = value
					x.expr = x0.expr
					x.typ = a[i]
				}, 2, true
			}
			x0.mode = value
		}

		// single value
		return func(x *operand, i int) {
			if i != 0 {
				unreachable()
			}
			*x = x0
		}, 1, false
	}

	// zero or multiple values
	return get, n, false
}

// arguments checks argument passing for the call with the given signature.
// The arg function provides the operand for the i'th argument.
func (check *Checker) arguments(x *operand, call *ast.CallExpr, sig *Signature, arg getter, n int) {
	if call.Ellipsis.IsValid() {
		// last argument is of the form x...
		if len(call.Args) == 1 && n > 1 {
			// f()... is not permitted if f() is multi-valued
			check.errorf(call.Ellipsis, "cannot use ... with %d-valued expression %s", n, call.Args[0])
			check.useGetter(arg, n)
			return
		}
		if !sig.variadic {
			check.errorf(call.Ellipsis, "cannot use ... in call to non-variadic %s", call.Fun)
			check.useGetter(arg, n)
			return
		}
	}

	// evaluate arguments
	for i := 0; i < n; i++ {
		arg(x, i)
		if x.mode != invalid {
			var ellipsis token.Pos
			if i == n-1 && call.Ellipsis.IsValid() {
				ellipsis = call.Ellipsis
			}
			check.argument(sig, i, x, ellipsis)
		}
	}

	// check argument count
	if sig.variadic {
		// a variadic function accepts an "empty"
		// last argument: count one extra
		n++
	}
	if n < sig.params.Len() {
		check.errorf(call.Rparen, "too few arguments in call to %s", call.Fun)
		// ok to continue
	}
}

// argument checks passing of argument x to the i'th parameter of the given signature.
// If ellipsis is valid, the argument is followed by ... at that position in the call.
func (check *Checker) argument(sig *Signature, i int, x *operand, ellipsis token.Pos) {
	n := sig.params.Len()

	// determine parameter type
	var typ Type
	switch {
	case i < n:
		typ = sig.params.vars[i].typ
	case sig.variadic:
		typ = sig.params.vars[n-1].typ
		if debug {
			if _, ok := typ.(*Slice); !ok {
				check.dump("%s: expected unnamed slice type, got %s", sig.params.vars[n-1].Pos(), typ)
			}
		}
	default:
		check.errorf(x.pos(), "too many arguments")
		return
	}

	if ellipsis.IsValid() {
		// argument is of the form x...
		if i != n-1 {
			check.errorf(ellipsis, "can only use ... with matching parameter")
			return
		}
		switch t := x.typ.Underlying().(type) {
		case *Slice:
			// ok
		case *Tuple:
			check.errorf(ellipsis, "cannot use ... with %d-valued expression %s", t.Len(), x)
			return
		default:
			check.errorf(x.pos(), "cannot use %s as parameter of type %s", x, typ)
			return
		}
	} else if sig.variadic && i >= n-1 {
		// use the variadic parameter slice's element type
		typ = typ.(*Slice).elem
	}

	if !check.assignment(x, typ) && x.mode != invalid {
		check.errorf(x.pos(), "cannot pass argument %s to parameter of type %s", x, typ)
	}
}

func (check *Checker) selector(x *operand, e *ast.SelectorExpr) {
	// these must be declared before the "goto Error" statements
	var (
		obj      Object
		index    []int
		indirect bool
	)

	sel := e.Sel.Name
	// If the identifier refers to a package, handle everything here
	// so we don't need a "package" mode for operands: package names
	// can only appear in qualified identifiers which are mapped to
	// selector expressions.
	if ident, ok := e.X.(*ast.Ident); ok {
		_, obj := check.scope.LookupParent(ident.Name)
		if pkg, _ := obj.(*PkgName); pkg != nil {
			assert(pkg.pkg == check.pkg)
			check.recordUse(ident, pkg)
			pkg.used = true
			exp := pkg.imported.scope.Lookup(sel)
			if exp == nil {
				if !pkg.imported.fake {
					check.errorf(e.Pos(), "%s not declared by package %s", sel, ident)
				}
				goto Error
			}
			if !exp.Exported() {
				check.errorf(e.Pos(), "%s not exported by package %s", sel, ident)
				// ok to continue
			}
			check.recordUse(e.Sel, exp)
			// Simplified version of the code for *ast.Idents:
			// - imported objects are always fully initialized
			switch exp := exp.(type) {
			case *Const:
				assert(exp.Val() != nil)
				x.mode = constant
				x.typ = exp.typ
				x.val = exp.val
			case *TypeName:
				x.mode = typexpr
				x.typ = exp.typ
			case *Var:
				x.mode = variable
				x.typ = exp.typ
			case *Func:
				x.mode = value
				x.typ = exp.typ
			case *Builtin:
				x.mode = builtin
				x.typ = exp.typ
				x.id = exp.id
			default:
				unreachable()
			}
			x.expr = e
			return
		}
	}

	check.exprOrType(x, e.X)
	if x.mode == invalid {
		goto Error
	}

	obj, index, indirect = LookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, sel)
	if obj == nil {
		switch {
		case index != nil:
			// TODO(gri) should provide actual type where the conflict happens
			check.invalidOp(e.Pos(), "ambiguous selector %s", sel)
		case indirect:
			check.invalidOp(e.Pos(), "%s is not in method set of %s", sel, x.typ)
		default:
			check.invalidOp(e.Pos(), "%s has no field or method %s", x, sel)
		}
		goto Error
	}

	if x.mode == typexpr {
		// method expression
		m, _ := obj.(*Func)
		if m == nil {
			check.invalidOp(e.Pos(), "%s has no method %s", x, sel)
			goto Error
		}

		check.recordSelection(e, MethodExpr, x.typ, m, index, indirect)

		// the receiver type becomes the type of the first function
		// argument of the method expression's function type
		var params []*Var
		sig := m.typ.(*Signature)
		if sig.params != nil {
			params = sig.params.vars
		}
		x.mode = value
		x.typ = &Signature{
			params:   NewTuple(append([]*Var{NewVar(token.NoPos, check.pkg, "", x.typ)}, params...)...),
			results:  sig.results,
			variadic: sig.variadic,
		}

		check.addDeclDep(m)

	} else {
		// regular selector
		switch obj := obj.(type) {
		case *Var:
			check.recordSelection(e, FieldVal, x.typ, obj, index, indirect)
			if x.mode == variable || indirect {
				x.mode = variable
			} else {
				x.mode = value
			}
			x.typ = obj.typ

		case *Func:
			// TODO(gri) If we needed to take into account the receiver's
			// addressability, should we report the type &(x.typ) instead?
			check.recordSelection(e, MethodVal, x.typ, obj, index, indirect)

			if debug {
				// Verify that LookupFieldOrMethod and MethodSet.Lookup agree.
				typ := x.typ
				if x.mode == variable {
					// If typ is not an (unnamed) pointer or an interface,
					// use *typ instead, because the method set of *typ
					// includes the methods of typ.
					// Variables are addressable, so we can always take their
					// address.
					if _, ok := typ.(*Pointer); !ok && !IsInterface(typ) {
						typ = &Pointer{base: typ}
					}
				}
				// If we created a synthetic pointer type above, we will throw
				// away the method set computed here after use.
				// TODO(gri) Method set computation should probably always compute
				// both, the value and the pointer receiver method set and represent
				// them in a single structure.
				// TODO(gri) Consider also using a method set cache for the lifetime
				// of checker once we rely on MethodSet lookup instead of individual
				// lookup.
				mset := NewMethodSet(typ)
				if m := mset.Lookup(check.pkg, sel); m == nil || m.obj != obj {
					check.dump("%s: (%s).%v -> %s", e.Pos(), typ, obj.name, m)
					check.dump("%s\n", mset)
					panic("method sets and lookup don't agree")
				}
			}

			x.mode = value

			// remove receiver
			sig := *obj.typ.(*Signature)
			sig.recv = nil
			x.typ = &sig

			check.addDeclDep(obj)

		default:
			unreachable()
		}
	}

	// everything went well
	x.expr = e
	return

Error:
	x.mode = invalid
	x.expr = e
}
