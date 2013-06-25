// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of call and selector expressions.

package types

import (
	"go/ast"
	"go/token"
)

func (check *checker) call(x *operand, e *ast.CallExpr, iota int) {
	check.exprOrType(x, e.Fun, iota, false)
	if x.mode == invalid {
		// We don't have a valid call or conversion but we have list of arguments.
		// Typecheck them independently for better partial type information in
		// the presence of type errors.
		for _, arg := range e.Args {
			check.expr(x, arg, nil, iota)
		}
		goto Error

	} else if x.mode == typexpr {
		check.conversion(x, e, x.typ, iota)

	} else if sig, ok := x.typ.Underlying().(*Signature); ok {
		// check parameters

		// If we have a trailing ... at the end of the parameter
		// list, the last argument must match the parameter type
		// []T of a variadic function parameter x ...T.
		passSlice := false
		if e.Ellipsis.IsValid() {
			if sig.isVariadic {
				passSlice = true
			} else {
				check.errorf(e.Ellipsis, "cannot use ... in call to %s", e.Fun)
				// ok to continue
			}
		}

		// If we have a single argument that is a function call
		// we need to handle it separately. Determine if this
		// is the case without checking the argument.
		var call *ast.CallExpr
		if len(e.Args) == 1 {
			call, _ = unparen(e.Args[0]).(*ast.CallExpr)
		}

		n := 0 // parameter count
		if call != nil {
			// We have a single argument that is a function call.
			check.expr(x, call, nil, -1)
			if x.mode == invalid {
				goto Error // TODO(gri): we can do better
			}
			if t, ok := x.typ.(*Tuple); ok {
				// multiple result values
				n = t.Len()
				for i := 0; i < n; i++ {
					obj := t.At(i)
					x.mode = value
					x.expr = nil // TODO(gri) can we do better here? (for good error messages)
					x.typ = obj.typ
					check.argument(sig, i, nil, x, passSlice && i+1 == n)
				}
			} else {
				// single result value
				n = 1
				check.argument(sig, 0, nil, x, passSlice)
			}

		} else {
			// We don't have a single argument or it is not a function call.
			n = len(e.Args)
			for i, arg := range e.Args {
				check.argument(sig, i, arg, x, passSlice && i+1 == n)
			}
		}

		// determine if we have enough arguments
		if sig.isVariadic {
			// a variadic function accepts an "empty"
			// last argument: count one extra
			n++
		}
		if n < sig.params.Len() {
			check.errorf(e.Fun.Pos(), "too few arguments in call to %s", e.Fun)
			// ok to continue
		}

		// determine result
		switch sig.results.Len() {
		case 0:
			x.mode = novalue
		case 1:
			x.mode = value
			x.typ = sig.results.vars[0].typ
		default:
			x.mode = value
			x.typ = sig.results
		}

	} else if bin, ok := x.typ.(*Builtin); ok {
		check.builtin(x, e, bin, iota)

	} else {
		check.invalidOp(x.pos(), "cannot call non-function %s", x)
		goto Error
	}

	// everything went well
	x.expr = e
	return

Error:
	x.mode = invalid
	x.expr = e
}

// argument typechecks passing an argument arg (if arg != nil) or
// x (if arg == nil) to the i'th parameter of the given signature.
// If passSlice is set, the argument is followed by ... in the call.
//
func (check *checker) argument(sig *Signature, i int, arg ast.Expr, x *operand, passSlice bool) {
	// determine parameter
	var par *Var
	n := sig.params.Len()
	if i < n {
		par = sig.params.vars[i]
	} else if sig.isVariadic {
		par = sig.params.vars[n-1]
	} else {
		var pos token.Pos
		switch {
		case arg != nil:
			pos = arg.Pos()
		case x != nil:
			pos = x.pos()
		default:
			// TODO(gri) what position to use?
		}
		check.errorf(pos, "too many arguments")
		return
	}

	// determine argument
	var z operand
	z.mode = variable
	z.expr = nil // TODO(gri) can we do better here? (for good error messages)
	z.typ = par.typ

	if arg != nil {
		check.expr(x, arg, z.typ, -1)
	}
	if x.mode == invalid {
		return // ignore this argument
	}

	// check last argument of the form x...
	if passSlice {
		if i+1 != n {
			check.errorf(x.pos(), "can only use ... with matching parameter")
			return // ignore this argument
		}
		// spec: "If the final argument is assignable to a slice type []T,
		// it may be passed unchanged as the value for a ...T parameter if
		// the argument is followed by ..."
		z.typ = &Slice{elt: z.typ} // change final parameter type to []T
	}

	if !check.assignment(x, z.typ) && x.mode != invalid {
		check.errorf(x.pos(), "cannot pass argument %s to %s", x, &z)
	}
}

func (check *checker) selector(x *operand, e *ast.SelectorExpr, iota int) {
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
		if pkg, ok := check.topScope.LookupParent(ident.Name).(*Package); ok {
			check.callIdent(ident, pkg)
			exp := pkg.scope.Lookup(nil, sel)
			if exp == nil {
				check.errorf(e.Pos(), "%s not declared by package %s", sel, ident)
				goto Error
			} else if !ast.IsExported(exp.Name()) {
				// gcimported package scopes contain non-exported
				// objects such as types used in partially exported
				// objects - do not accept them
				check.errorf(e.Pos(), "%s not exported by package %s", sel, ident)
				goto Error
			}
			check.callIdent(e.Sel, exp)
			// Simplified version of the code for *ast.Idents:
			// - imported packages use types.Scope and types.Objects
			// - imported objects are always fully initialized
			switch exp := exp.(type) {
			case *Const:
				assert(exp.Val != nil)
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
			default:
				unreachable()
			}
			x.expr = e
			return
		}
	}

	check.exprOrType(x, e.X, iota, false)
	if x.mode == invalid {
		goto Error
	}

	obj, index, indirect = LookupFieldOrMethod(x.typ, check.pkg, sel)
	if obj == nil {
		if index != nil {
			// TODO(gri) should provide actual type where the conflict happens
			check.invalidOp(e.Pos(), "ambiguous selector %s", sel)
		} else {
			check.invalidOp(e.Pos(), "%s has no field or method %s", x, sel)
		}
		goto Error
	}

	check.callIdent(e.Sel, obj)

	if x.mode == typexpr {
		// method expression
		m, _ := obj.(*Func)
		if m == nil {
			check.invalidOp(e.Pos(), "%s has no method %s", x, sel)
			goto Error
		}

		// verify that m is in the method set of x.typ
		// (the receiver is nil if f is an interface method)
		if recv := m.typ.(*Signature).recv; recv != nil {
			if _, isPtr := deref(recv.typ); isPtr && !indirect {
				check.invalidOp(e.Pos(), "%s is not in method set of %s", sel, x.typ)
				goto Error
			}
		}

		// the receiver type becomes the type of the first function
		// argument of the method expression's function type
		var params []*Var
		sig := m.typ.(*Signature)
		if sig.params != nil {
			params = sig.params.vars
		}
		x.mode = value
		x.typ = &Signature{
			params:     NewTuple(append([]*Var{NewVar(token.NoPos, check.pkg, "", x.typ)}, params...)...),
			results:    sig.results,
			isVariadic: sig.isVariadic,
		}

	} else {
		// regular selector
		switch obj := obj.(type) {
		case *Field:
			x.mode = variable
			x.typ = obj.typ
		case *Func:
			// TODO(gri) Temporary check to verify corresponding lookup via method sets.
			//           Remove eventually.
			if m := NewMethodSet(x.typ).Lookup(check.pkg, sel); m != obj {
				check.dump("%s: %v", e.Pos(), obj.name)
				panic("method sets and lookup don't agree")
			}

			// TODO(gri) This code appears elsewhere, too. Factor!
			// verify that obj is in the method set of x.typ (or &(x.typ) if x is addressable)
			// (the receiver is nil if obj is an interface method)
			//
			// spec: "A method call x.m() is valid if the method set of (the type of) x
			//        contains m and the argument list can be assigned to the parameter
			//        list of m. If x is addressable and &x's method set contains m, x.m()
			//        is shorthand for (&x).m()".
			if recv := obj.typ.(*Signature).recv; recv != nil {
				if _, isPtr := deref(recv.typ); isPtr && !indirect && x.mode != variable {
					check.invalidOp(e.Pos(), "%s is not in method set of %s", sel, x)
					goto Error
				}
			}

			x.mode = value
			x.typ = obj.typ
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
