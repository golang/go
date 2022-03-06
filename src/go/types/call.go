// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of call and selector expressions.

package types

import (
	"go/ast"
	"go/internal/typeparams"
	"go/token"
	"strings"
	"unicode"
)

// funcInst type-checks a function instantiation inst and returns the result in x.
// The operand x must be the evaluation of inst.X and its type must be a signature.
func (check *Checker) funcInst(x *operand, ix *typeparams.IndexExpr) {
	if !check.allowVersion(check.pkg, 1, 18) {
		check.softErrorf(inNode(ix.Orig, ix.Lbrack), _UnsupportedFeature, "function instantiation requires go1.18 or later")
	}

	targs := check.typeList(ix.Indices)
	if targs == nil {
		x.mode = invalid
		x.expr = ix.Orig
		return
	}
	assert(len(targs) == len(ix.Indices))

	// check number of type arguments (got) vs number of type parameters (want)
	sig := x.typ.(*Signature)
	got, want := len(targs), sig.TypeParams().Len()
	if got > want {
		check.errorf(ix.Indices[got-1], _WrongTypeArgCount, "got %d type arguments but want %d", got, want)
		x.mode = invalid
		x.expr = ix.Orig
		return
	}

	if got < want {
		targs = check.infer(ix.Orig, sig.TypeParams().list(), targs, nil, nil)
		if targs == nil {
			// error was already reported
			x.mode = invalid
			x.expr = ix.Orig
			return
		}
		got = len(targs)
	}
	assert(got == want)

	// instantiate function signature
	res := check.instantiateSignature(x.Pos(), sig, targs, ix.Indices)
	assert(res.TypeParams().Len() == 0) // signature is not generic anymore
	check.recordInstance(ix.Orig, targs, res)
	x.typ = res
	x.mode = value
	x.expr = ix.Orig
}

func (check *Checker) instantiateSignature(pos token.Pos, typ *Signature, targs []Type, xlist []ast.Expr) (res *Signature) {
	assert(check != nil)
	assert(len(targs) == typ.TypeParams().Len())

	if trace {
		check.trace(pos, "-- instantiating %s with %s", typ, targs)
		check.indent++
		defer func() {
			check.indent--
			check.trace(pos, "=> %s (under = %s)", res, res.Underlying())
		}()
	}

	inst := check.instance(pos, typ, targs, check.bestContext(nil)).(*Signature)
	assert(len(xlist) <= len(targs))

	// verify instantiation lazily (was issue #50450)
	check.later(func() {
		tparams := typ.TypeParams().list()
		if i, err := check.verify(pos, tparams, targs); err != nil {
			// best position for error reporting
			pos := pos
			if i < len(xlist) {
				pos = xlist[i].Pos()
			}
			check.softErrorf(atPos(pos), _InvalidTypeArg, "%s", err)
		} else {
			check.mono.recordInstance(check.pkg, pos, tparams, targs, xlist)
		}
	})

	return inst
}

func (check *Checker) callExpr(x *operand, call *ast.CallExpr) exprKind {
	ix := typeparams.UnpackIndexExpr(call.Fun)
	if ix != nil {
		if check.indexExpr(x, ix) {
			// Delay function instantiation to argument checking,
			// where we combine type and value arguments for type
			// inference.
			assert(x.mode == value)
		} else {
			ix = nil
		}
		x.expr = call.Fun
		check.record(x)

	} else {
		check.exprOrType(x, call.Fun, true)
	}
	// x.typ may be generic

	switch x.mode {
	case invalid:
		check.use(call.Args...)
		x.expr = call
		return statement

	case typexpr:
		// conversion
		check.nonGeneric(x)
		if x.mode == invalid {
			return conversion
		}
		T := x.typ
		x.mode = invalid
		switch n := len(call.Args); n {
		case 0:
			check.errorf(inNode(call, call.Rparen), _WrongArgCount, "missing argument in conversion to %s", T)
		case 1:
			check.expr(x, call.Args[0])
			if x.mode != invalid {
				if call.Ellipsis.IsValid() {
					check.errorf(call.Args[0], _BadDotDotDotSyntax, "invalid use of ... in conversion to %s", T)
					break
				}
				if t, _ := under(T).(*Interface); t != nil && !isTypeParam(T) {
					if !t.IsMethodSet() {
						check.errorf(call, _MisplacedConstraintIface, "cannot use interface %s in conversion (contains specific type constraints or is comparable)", T)
						break
					}
				}
				check.conversion(x, T)
			}
		default:
			check.use(call.Args...)
			check.errorf(call.Args[n-1], _WrongArgCount, "too many arguments in conversion to %s", T)
		}
		x.expr = call
		return conversion

	case builtin:
		// no need to check for non-genericity here
		id := x.id
		if !check.builtin(x, call, id) {
			x.mode = invalid
		}
		x.expr = call
		// a non-constant result implies a function call
		if x.mode != invalid && x.mode != constant_ {
			check.hasCallOrRecv = true
		}
		return predeclaredFuncs[id].kind
	}

	// ordinary function/method call
	// signature may be generic
	cgocall := x.mode == cgofunc

	// a type parameter may be "called" if all types have the same signature
	sig, _ := coreType(x.typ).(*Signature)
	if sig == nil {
		check.invalidOp(x, _InvalidCall, "cannot call non-function %s", x)
		x.mode = invalid
		x.expr = call
		return statement
	}

	// evaluate type arguments, if any
	var xlist []ast.Expr
	var targs []Type
	if ix != nil {
		xlist = ix.Indices
		targs = check.typeList(xlist)
		if targs == nil {
			check.use(call.Args...)
			x.mode = invalid
			x.expr = call
			return statement
		}
		assert(len(targs) == len(xlist))

		// check number of type arguments (got) vs number of type parameters (want)
		got, want := len(targs), sig.TypeParams().Len()
		if got > want {
			check.errorf(xlist[want], _WrongTypeArgCount, "got %d type arguments but want %d", got, want)
			check.use(call.Args...)
			x.mode = invalid
			x.expr = call
			return statement
		}
	}

	// evaluate arguments
	args, _ := check.exprList(call.Args, false)
	isGeneric := sig.TypeParams().Len() > 0
	sig = check.arguments(call, sig, targs, args, xlist)

	if isGeneric && sig.TypeParams().Len() == 0 {
		// Update the recorded type of call.Fun to its instantiated type.
		check.recordTypeAndValue(call.Fun, value, sig, nil)
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
	x.expr = call
	check.hasCallOrRecv = true

	// if type inference failed, a parametrized result must be invalidated
	// (operands cannot have a parametrized type)
	if x.mode == value && sig.TypeParams().Len() > 0 && isParameterized(sig.TypeParams().list(), x.typ) {
		x.mode = invalid
	}

	return statement
}

func (check *Checker) exprList(elist []ast.Expr, allowCommaOk bool) (xlist []*operand, commaOk bool) {
	switch len(elist) {
	case 0:
		// nothing to do

	case 1:
		// single (possibly comma-ok) value, or function returning multiple values
		e := elist[0]
		var x operand
		check.multiExpr(&x, e)
		if t, ok := x.typ.(*Tuple); ok && x.mode != invalid {
			// multiple values
			xlist = make([]*operand, t.Len())
			for i, v := range t.vars {
				xlist[i] = &operand{mode: value, expr: e, typ: v.typ}
			}
			break
		}

		// exactly one (possibly invalid or comma-ok) value
		xlist = []*operand{&x}
		if allowCommaOk && (x.mode == mapindex || x.mode == commaok || x.mode == commaerr) {
			x2 := &operand{mode: value, expr: e, typ: Typ[UntypedBool]}
			if x.mode == commaerr {
				x2.typ = universeError
			}
			xlist = append(xlist, x2)
			commaOk = true
		}

	default:
		// multiple (possibly invalid) values
		xlist = make([]*operand, len(elist))
		for i, e := range elist {
			var x operand
			check.expr(&x, e)
			xlist[i] = &x
		}
	}

	return
}

// xlist is the list of type argument expressions supplied in the source code.
func (check *Checker) arguments(call *ast.CallExpr, sig *Signature, targs []Type, args []*operand, xlist []ast.Expr) (rsig *Signature) {
	rsig = sig

	// TODO(gri) try to eliminate this extra verification loop
	for _, a := range args {
		switch a.mode {
		case typexpr:
			check.errorf(a, 0, "%s used as value", a)
			return
		case invalid:
			return
		}
	}

	// Function call argument/parameter count requirements
	//
	//               | standard call    | dotdotdot call |
	// --------------+------------------+----------------+
	// standard func | nargs == npars   | invalid        |
	// --------------+------------------+----------------+
	// variadic func | nargs >= npars-1 | nargs == npars |
	// --------------+------------------+----------------+

	nargs := len(args)
	npars := sig.params.Len()
	ddd := call.Ellipsis.IsValid()

	// set up parameters
	sigParams := sig.params // adjusted for variadic functions (may be nil for empty parameter lists!)
	adjusted := false       // indicates if sigParams is different from t.params
	if sig.variadic {
		if ddd {
			// variadic_func(a, b, c...)
			if len(call.Args) == 1 && nargs > 1 {
				// f()... is not permitted if f() is multi-valued
				check.errorf(inNode(call, call.Ellipsis), _InvalidDotDotDot, "cannot use ... with %d-valued %s", nargs, call.Args[0])
				return
			}
		} else {
			// variadic_func(a, b, c)
			if nargs >= npars-1 {
				// Create custom parameters for arguments: keep
				// the first npars-1 parameters and add one for
				// each argument mapping to the ... parameter.
				vars := make([]*Var, npars-1) // npars > 0 for variadic functions
				copy(vars, sig.params.vars)
				last := sig.params.vars[npars-1]
				typ := last.typ.(*Slice).elem
				for len(vars) < nargs {
					vars = append(vars, NewParam(last.pos, last.pkg, last.name, typ))
				}
				sigParams = NewTuple(vars...) // possibly nil!
				adjusted = true
				npars = nargs
			} else {
				// nargs < npars-1
				npars-- // for correct error message below
			}
		}
	} else {
		if ddd {
			// standard_func(a, b, c...)
			check.errorf(inNode(call, call.Ellipsis), _NonVariadicDotDotDot, "cannot use ... in call to non-variadic %s", call.Fun)
			return
		}
		// standard_func(a, b, c)
	}

	// check argument count
	if nargs != npars {
		var at positioner = call
		qualifier := "not enough"
		if nargs > npars {
			at = args[npars].expr // report at first extra argument
			qualifier = "too many"
		} else {
			at = atPos(call.Rparen) // report at closing )
		}
		// take care of empty parameter lists represented by nil tuples
		var params []*Var
		if sig.params != nil {
			params = sig.params.vars
		}
		check.errorf(at, _WrongArgCount, "%s arguments in call to %s\n\thave %s\n\twant %s",
			qualifier, call.Fun,
			check.typesSummary(operandTypes(args), false),
			check.typesSummary(varTypes(params), sig.variadic),
		)
		return
	}

	// infer type arguments and instantiate signature if necessary
	if sig.TypeParams().Len() > 0 {
		if !check.allowVersion(check.pkg, 1, 18) {
			switch call.Fun.(type) {
			case *ast.IndexExpr, *ast.IndexListExpr:
				ix := typeparams.UnpackIndexExpr(call.Fun)
				check.softErrorf(inNode(call.Fun, ix.Lbrack), _UnsupportedFeature, "function instantiation requires go1.18 or later")
			default:
				check.softErrorf(inNode(call, call.Lparen), _UnsupportedFeature, "implicit function instantiation requires go1.18 or later")
			}
		}
		targs := check.infer(call, sig.TypeParams().list(), targs, sigParams, args)
		if targs == nil {
			return // error already reported
		}

		// compute result signature
		rsig = check.instantiateSignature(call.Pos(), sig, targs, xlist)
		assert(rsig.TypeParams().Len() == 0) // signature is not generic anymore
		check.recordInstance(call.Fun, targs, rsig)

		// Optimization: Only if the parameter list was adjusted do we
		// need to compute it from the adjusted list; otherwise we can
		// simply use the result signature's parameter list.
		if adjusted {
			sigParams = check.subst(call.Pos(), sigParams, makeSubstMap(sig.TypeParams().list(), targs), nil).(*Tuple)
		} else {
			sigParams = rsig.params
		}
	}

	// check arguments
	if len(args) > 0 {
		context := check.sprintf("argument to %s", call.Fun)
		for i, a := range args {
			check.assignment(a, sigParams.vars[i].typ, context)
		}
	}

	return
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

func (check *Checker) selector(x *operand, e *ast.SelectorExpr, def *Named) {
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
				check.dump("%v: unexpected object %v", e.Sel.Pos(), exp)
				unreachable()
			}
			x.expr = e
			return
		}
	}

	check.exprOrType(x, e.X, false)
	switch x.mode {
	case typexpr:
		// don't crash for "type T T.x" (was issue #51509)
		if def != nil && x.typ == def {
			check.cycleError([]Object{def.obj})
			goto Error
		}
	case builtin:
		// types2 uses the position of '.' for the error
		check.errorf(e.Sel, _UncalledBuiltin, "cannot select on %s", x)
		goto Error
	case invalid:
		goto Error
	}

	obj, index, indirect = LookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, sel)
	if obj == nil {
		// Don't report another error if the underlying type was invalid (issue #49541).
		if under(x.typ) == Typ[Invalid] {
			goto Error
		}

		if index != nil {
			// TODO(gri) should provide actual type where the conflict happens
			check.errorf(e.Sel, _AmbiguousSelector, "ambiguous selector %s.%s", x.expr, sel)
			goto Error
		}

		if indirect {
			check.errorf(e.Sel, _InvalidMethodExpr, "cannot call pointer method %s on %s", sel, x.typ)
			goto Error
		}

		var why string
		if isInterfacePtr(x.typ) {
			why = check.interfacePtrError(x.typ)
		} else {
			why = check.sprintf("type %s has no field or method %s", x.typ, sel)
			// Check if capitalization of sel matters and provide better error message in that case.
			// TODO(gri) This code only looks at the first character but LookupFieldOrMethod should
			//           have an (internal) mechanism for case-insensitive lookup that we should use
			//           instead (see types2).
			if len(sel) > 0 {
				var changeCase string
				if r := rune(sel[0]); unicode.IsUpper(r) {
					changeCase = string(unicode.ToLower(r)) + sel[1:]
				} else {
					changeCase = string(unicode.ToUpper(r)) + sel[1:]
				}
				if obj, _, _ = LookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, changeCase); obj != nil {
					why += ", but does have " + changeCase
				}
			}
		}
		check.errorf(e.Sel, _MissingFieldOrMethod, "%s.%s undefined (%s)", x.expr, sel, why)
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

		sig := m.typ.(*Signature)
		if sig.recv == nil {
			check.error(e, _InvalidDeclCycle, "illegal cycle in method declaration")
			goto Error
		}

		// the receiver type becomes the type of the first function
		// argument of the method expression's function type
		var params []*Var
		if sig.params != nil {
			params = sig.params.vars
		}
		// Be consistent about named/unnamed parameters. This is not needed
		// for type-checking, but the newly constructed signature may appear
		// in an error message and then have mixed named/unnamed parameters.
		// (An alternative would be to not print parameter names in errors,
		// but it's useful to see them; this is cheap and method expressions
		// are rare.)
		name := ""
		if len(params) > 0 && params[0].name != "" {
			// name needed
			name = sig.recv.name
			if name == "" {
				name = "_"
			}
		}
		params = append([]*Var{NewVar(sig.recv.pos, sig.recv.pkg, name, x.typ)}, params...)
		x.mode = value
		x.typ = &Signature{
			tparams:  sig.tparams,
			params:   NewTuple(params...),
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

			// TODO(gri) The verification pass below is disabled for now because
			//           method sets don't match method lookup in some cases.
			//           For instance, if we made a copy above when creating a
			//           custom method for a parameterized received type, the
			//           method set method doesn't match (no copy there). There
			///          may be other situations.
			disabled := true
			if !disabled && debug {
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
			check.rawExpr(&x, e, nil, false)
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
		check.rawExpr(&x, e, nil, false)
		if v != nil {
			v.used = v_used // restore v.used
		}
	}
}
