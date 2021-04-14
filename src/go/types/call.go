// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of call and selector expressions.

package types

import (
	"go/ast"
	"go/token"
	"strings"
	"unicode"
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
			check.errorf(inNode(e, e.Rparen), _WrongArgCount, "missing argument in conversion to %s", T)
		case 1:
			check.expr(x, e.Args[0])
			if x.mode != invalid {
				if e.Ellipsis.IsValid() {
					check.errorf(e.Args[0], _BadDotDotDotSyntax, "invalid use of ... in conversion to %s", T)
					break
				}
				check.conversion(x, T)
			}
		default:
			check.use(e.Args...)
			check.errorf(e.Args[n-1], _WrongArgCount, "too many arguments in conversion to %s", T)
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
		if x.mode != invalid && x.mode != constant_ {
			check.hasCallOrRecv = true
		}
		return predeclaredFuncs[id].kind

	default:
		// function/method call
		cgocall := x.mode == cgofunc

		sig, _ := x.typ.Underlying().(*Signature)
		if sig == nil {
			check.invalidOp(x, _InvalidCall, "cannot call non-function %s", x)
			x.mode = invalid
			x.expr = e
			return statement
		}

		arg, n, _ := unpack(func(x *operand, i int) { check.multiExpr(x, e.Args[i]) }, len(e.Args), false)
		if arg != nil {
			check.arguments(x, e, sig, arg, n)
		} else {
			x.mode = invalid
		}

		// determine result
		switch sig.results.Len() {
		case 0:
			x.mode = novalue
		case 1:
			if cgocall {
				x.mode = commaerr
			} else {
				x.mode = value
			}
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
// The arguments may be nil.
func (check *Checker) use(arg ...ast.Expr) {
	var x operand
	for _, e := range arg {
		// The nil check below is necessary since certain AST fields
		// may legally be nil (e.g., the ast.SliceExpr.High field).
		if e != nil {
			check.rawExpr(&x, e, nil)
		}
	}
}

// useLHS is like use, but doesn't "use" top-level identifiers.
// It should be called instead of use if the arguments are
// expressions on the lhs of an assignment.
// The arguments must not be nil.
func (check *Checker) useLHS(arg ...ast.Expr) {
	var x operand
	for _, e := range arg {
		// If the lhs is an identifier denoting a variable v, this assignment
		// is not a 'use' of v. Remember current value of v.used and restore
		// after evaluating the lhs via check.rawExpr.
		var v *Var
		var v_used bool
		if ident, _ := unparen(e).(*ast.Ident); ident != nil {
			// never type-check the blank name on the lhs
			if ident.Name == "_" {
				continue
			}
			if _, obj := check.scope.LookupParent(ident.Name, token.NoPos); obj != nil {
				// It's ok to mark non-local variables, but ignore variables
				// from other packages to avoid potential race conditions with
				// dot-imported variables.
				if w, _ := obj.(*Var); w != nil && w.pkg == check.pkg {
					v = w
					v_used = v.used
				}
			}
		}
		check.rawExpr(&x, e, nil)
		if v != nil {
			v.used = v_used // restore v.used
		}
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
	if n != 1 {
		// zero or multiple values
		return get, n, false
	}
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

	if x0.mode == mapindex || x0.mode == commaok || x0.mode == commaerr {
		// comma-ok value
		if allowCommaOk {
			a := [2]Type{x0.typ, Typ[UntypedBool]}
			if x0.mode == commaerr {
				a[1] = universeError
			}
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

// arguments checks argument passing for the call with the given signature.
// The arg function provides the operand for the i'th argument.
func (check *Checker) arguments(x *operand, call *ast.CallExpr, sig *Signature, arg getter, n int) {
	if call.Ellipsis.IsValid() {
		// last argument is of the form x...
		if !sig.variadic {
			check.errorf(atPos(call.Ellipsis), _NonVariadicDotDotDot, "cannot use ... in call to non-variadic %s", call.Fun)
			check.useGetter(arg, n)
			return
		}
		if len(call.Args) == 1 && n > 1 {
			// f()... is not permitted if f() is multi-valued
			check.errorf(atPos(call.Ellipsis), _InvalidDotDotDotOperand, "cannot use ... with %d-valued %s", n, call.Args[0])
			check.useGetter(arg, n)
			return
		}
	}

	// evaluate arguments
	context := check.sprintf("argument to %s", call.Fun)
	for i := 0; i < n; i++ {
		arg(x, i)
		if x.mode != invalid {
			var ellipsis token.Pos
			if i == n-1 && call.Ellipsis.IsValid() {
				ellipsis = call.Ellipsis
			}
			check.argument(sig, i, x, ellipsis, context)
		}
	}

	// check argument count
	if sig.variadic {
		// a variadic function accepts an "empty"
		// last argument: count one extra
		n++
	}
	if n < sig.params.Len() {
		check.errorf(inNode(call, call.Rparen), _WrongArgCount, "too few arguments in call to %s", call.Fun)
		// ok to continue
	}
}

// argument checks passing of argument x to the i'th parameter of the given signature.
// If ellipsis is valid, the argument is followed by ... at that position in the call.
func (check *Checker) argument(sig *Signature, i int, x *operand, ellipsis token.Pos, context string) {
	check.singleValue(x)
	if x.mode == invalid {
		return
	}

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
				check.dump("%v: expected unnamed slice type, got %s", sig.params.vars[n-1].Pos(), typ)
			}
		}
	default:
		check.errorf(x, _WrongArgCount, "too many arguments")
		return
	}

	if ellipsis.IsValid() {
		if i != n-1 {
			check.errorf(atPos(ellipsis), _MisplacedDotDotDot, "can only use ... with matching parameter")
			return
		}
		// argument is of the form x... and x is single-valued
		if _, ok := x.typ.Underlying().(*Slice); !ok && x.typ != Typ[UntypedNil] { // see issue #18268
			check.errorf(x, _InvalidDotDotDotOperand, "cannot use %s as parameter of type %s", x, typ)
			return
		}
	} else if sig.variadic && i >= n-1 {
		// use the variadic parameter slice's element type
		typ = typ.(*Slice).elem
	}

	check.assignment(x, typ, context)
}

var cgoPrefixes = [...]string{
	"_Ciconst_",
	"_Cfconst_",
	"_Csconst_",
	"_Ctype_",
	"_Cvar_", // actually a pointer to the var
	"_Cfpvar_fp_",
	"_Cfunc_",
	"_Cmacro_", // function to evaluate the expanded expression
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
		obj := check.lookup(ident.Name)
		if pname, _ := obj.(*PkgName); pname != nil {
			assert(pname.pkg == check.pkg)
			check.recordUse(ident, pname)
			pname.used = true
			pkg := pname.imported

			var exp Object
			funcMode := value
			if pkg.cgo {
				// cgo special cases C.malloc: it's
				// rewritten to _CMalloc and does not
				// support two-result calls.
				if sel == "malloc" {
					sel = "_CMalloc"
				} else {
					funcMode = cgofunc
				}
				for _, prefix := range cgoPrefixes {
					// cgo objects are part of the current package (in file
					// _cgo_gotypes.go). Use regular lookup.
					_, exp = check.scope.LookupParent(prefix+sel, check.pos)
					if exp != nil {
						break
					}
				}
				if exp == nil {
					check.errorf(e.Sel, _UndeclaredImportedName, "%s not declared by package C", sel)
					goto Error
				}
				check.objDecl(exp, nil)
			} else {
				exp = pkg.scope.Lookup(sel)
				if exp == nil {
					if !pkg.fake {
						check.errorf(e.Sel, _UndeclaredImportedName, "%s not declared by package %s", sel, pkg.name)
					}
					goto Error
				}
				if !exp.Exported() {
					check.errorf(e.Sel, _UnexportedName, "%s not exported by package %s", sel, pkg.name)
					// ok to continue
				}
			}
			check.recordUse(e.Sel, exp)

			// Simplified version of the code for *ast.Idents:
			// - imported objects are always fully initialized
			switch exp := exp.(type) {
			case *Const:
				assert(exp.Val() != nil)
				x.mode = constant_
				x.typ = exp.typ
				x.val = exp.val
			case *TypeName:
				x.mode = typexpr
				x.typ = exp.typ
			case *Var:
				x.mode = variable
				x.typ = exp.typ
				if pkg.cgo && strings.HasPrefix(exp.name, "_Cvar_") {
					x.typ = x.typ.(*Pointer).base
				}
			case *Func:
				x.mode = funcMode
				x.typ = exp.typ
				if pkg.cgo && strings.HasPrefix(exp.name, "_Cmacro_") {
					x.mode = value
					x.typ = x.typ.(*Signature).results.vars[0].typ
				}
			case *Builtin:
				x.mode = builtin
				x.typ = exp.typ
				x.id = exp.id
			default:
				check.dump("unexpected object %v", exp)
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

	obj, index, indirect = check.lookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, sel)
	if obj == nil {
		switch {
		case index != nil:
			// TODO(gri) should provide actual type where the conflict happens
			check.errorf(e.Sel, _AmbiguousSelector, "ambiguous selector %s.%s", x.expr, sel)
		case indirect:
			check.errorf(e.Sel, _InvalidMethodExpr, "cannot call pointer method %s on %s", sel, x.typ)
		default:
			// Check if capitalization of sel matters and provide better error
			// message in that case.
			if len(sel) > 0 {
				var changeCase string
				if r := rune(sel[0]); unicode.IsUpper(r) {
					changeCase = string(unicode.ToLower(r)) + sel[1:]
				} else {
					changeCase = string(unicode.ToUpper(r)) + sel[1:]
				}
				if obj, _, _ = check.lookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, changeCase); obj != nil {
					check.errorf(e.Sel, _MissingFieldOrMethod, "%s.%s undefined (type %s has no field or method %s, but does have %s)", x.expr, sel, x.typ, sel, changeCase)
					break
				}
			}
			check.errorf(e.Sel, _MissingFieldOrMethod, "%s.%s undefined (type %s has no field or method %s)", x.expr, sel, x.typ, sel)
		}
		goto Error
	}

	// methods may not have a fully set up signature yet
	if m, _ := obj.(*Func); m != nil {
		check.objDecl(m, nil)
	}

	if x.mode == typexpr {
		// method expression
		m, _ := obj.(*Func)
		if m == nil {
			// TODO(gri) should check if capitalization of sel matters and provide better error message in that case
			check.errorf(e.Sel, _MissingFieldOrMethod, "%s.%s undefined (type %s has no method %s)", x.expr, sel, x.typ, sel)
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
				// TODO(gri) This only works because we call LookupFieldOrMethod
				// _before_ calling NewMethodSet: LookupFieldOrMethod completes
				// any incomplete interfaces so they are available to NewMethodSet
				// (which assumes that interfaces have been completed already).
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
					check.dump("%v: (%s).%v -> %s", e.Pos(), typ, obj.name, m)
					check.dump("%s\n", mset)
					// Caution: MethodSets are supposed to be used externally
					// only (after all interface types were completed). It's
					// now possible that we get here incorrectly. Not urgent
					// to fix since we only run this code in debug mode.
					// TODO(gri) fix this eventually.
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
