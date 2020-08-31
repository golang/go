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

func (check *Checker) call(x *operand, call *ast.CallExpr) exprKind {
	check.exprOrType(x, call.Fun)

	switch x.mode {
	case invalid:
		check.use(call.Args...)
		x.expr = call
		return statement

	case typexpr:
		// conversion or type instantiation
		T := x.typ
		x.mode = invalid
		if isGeneric(T) {
			// type instantiation
			x.typ = check.typ(call)
			if x.typ != Typ[Invalid] {
				x.mode = typexpr
			}
			return expression
		}

		// conversion
		switch n := len(call.Args); n {
		case 0:
			check.errorf(call.Rparen, "missing argument in conversion to %s", T)
		case 1:
			check.expr(x, call.Args[0])
			if x.mode != invalid {
				if t := T.Interface(); t != nil {
					check.completeInterface(token.NoPos, t)
					if t.IsConstraint() {
						check.errorf(call.Pos(), "cannot use interface %s in conversion (contains type list or is comparable)", T)
						break
					}
				}
				check.conversion(x, T)
			}
		default:
			check.use(call.Args...)
			check.errorf(call.Args[n-1].Pos(), "too many arguments in conversion to %s", T)
		}
		x.expr = call
		return conversion

	case builtin:
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

	default:
		// function/method call
		cgocall := x.mode == cgofunc

		sig := x.typ.Signature()
		if sig == nil {
			check.invalidOp(x.pos(), "cannot call non-function %s", x)
			x.mode = invalid
			x.expr = call
			return statement
		}

		// evaluate arguments
		args, _ := check.exprOrTypeList(call.Args)

		// instantiate function if needed
		if n := len(args); n > 0 && len(sig.tparams) > 0 && args[0].mode == typexpr {
			// If the first argument is a type, assume we have explicit type arguments.

			// check number of type arguments
			if !check.conf.InferFromConstraints && n != len(sig.tparams) || n > len(sig.tparams) {
				check.errorf(args[n-1].pos(), "got %d type arguments but want %d", n, len(sig.tparams))
				x.mode = invalid
				x.expr = call
				return expression
			}

			// collect types
			targs := make([]Type, n)
			poslist := make([]token.Pos, n)
			for i, a := range args {
				if a.mode != typexpr {
					// error was reported earlier
					x.mode = invalid
					x.expr = call
					return expression
				}
				targs[i] = a.typ
				poslist[i] = a.pos()
			}

			// if we don't have enough type arguments, use constraint type inference
			if n < len(sig.tparams) {
				var failed int
				targs, failed = check.inferB(sig.tparams, targs)
				if targs == nil {
					// error was already reported
					x.mode = invalid
					x.expr = call
					return expression
				}
				if failed >= 0 {
					// at least one type argument couldn't be inferred
					assert(targs[failed] == nil)
					tpar := sig.tparams[failed]
					ppos := check.fset.Position(tpar.pos).String()
					check.errorf(call.Rparen, "cannot infer %s (%s) (%s)", tpar.name, ppos, targs)
					x.mode = invalid
					x.expr = call
					return expression
				}
				// all type arguments were inferred sucessfully
				if debug {
					for _, targ := range targs {
						assert(targ != nil)
					}
				}
				//check.dump("### inferred targs = %s", targs)
				n = len(targs)
			}
			assert(n == len(sig.tparams))

			// instantiate function signature
			for i, typ := range targs {
				// some positions may be missing if types are inferred
				var pos token.Pos
				if i < len(poslist) {
					pos = poslist[i]
				}
				check.ordinaryType(pos, typ)
			}
			res := check.instantiate(x.pos(), sig, targs, poslist).(*Signature)
			assert(res.tparams == nil) // signature is not generic anymore
			x.typ = res
			x.mode = value
			x.expr = call
			return expression
		}

		sig = check.arguments(call, sig, args)

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
		if x.mode == value && len(sig.tparams) > 0 && isParameterized(sig.tparams, x.typ) {
			x.mode = invalid
		}

		return statement
	}
}

// exprOrTypeList returns a list of operands and reports an error if the
// list contains a mix of values and types (ignoring invalid operands).
func (check *Checker) exprOrTypeList(elist []ast.Expr) (xlist []*operand, ok bool) {
	ok = true

	switch len(elist) {
	case 0:
		// nothing to do

	case 1:
		// single (possibly comma-ok) value or type, or function returning multiple values
		e := elist[0]
		var x operand
		check.multiExprOrType(&x, e)
		if t, ok := x.typ.(*Tuple); ok && x.mode != invalid && x.mode != typexpr {
			// multiple values
			xlist = make([]*operand, t.Len())
			for i, v := range t.vars {
				xlist[i] = &operand{mode: value, expr: e, typ: v.typ}
			}
			break
		}

		check.instantiatedOperand(&x)

		// exactly one (possibly invalid or comma-ok) value or type
		xlist = []*operand{&x}

	default:
		// multiple (possibly invalid) values or types
		xlist = make([]*operand, len(elist))
		ntypes := 0
		for i, e := range elist {
			var x operand
			check.exprOrType(&x, e)
			xlist[i] = &x
			switch x.mode {
			case invalid:
				ntypes = len(xlist) // make 'if' condition fail below (no additional error in this case)
			case typexpr:
				ntypes++
				check.instantiatedOperand(&x)
			}
		}
		if 0 < ntypes && ntypes < len(xlist) {
			check.errorf(xlist[0].pos(), "mix of value and type expressions")
			ok = false
		}
	}

	return
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
			x.mode = value
			xlist = append(xlist, &operand{mode: value, expr: e, typ: Typ[UntypedBool]})
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

func (check *Checker) arguments(call *ast.CallExpr, sig *Signature, args []*operand) (rsig *Signature) {
	rsig = sig

	// TODO(gri) try to eliminate this extra verification loop
	for _, a := range args {
		switch a.mode {
		case typexpr:
			check.errorf(a.pos(), "%s used as value", a)
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
	sig_params := sig.params // adjusted for variadic functions (may be nil for empty parameter lists!)
	adjusted := false        // indicates if sig_params is different from t.params
	if sig.variadic {
		if ddd {
			// variadic_func(a, b, c...)
			if len(call.Args) == 1 && nargs > 1 {
				// f()... is not permitted if f() is multi-valued
				check.errorf(call.Ellipsis, "cannot use ... with %d-valued %s", nargs, call.Args[0])
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
				sig_params = NewTuple(vars...) // possibly nil!
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
			check.errorf(call.Ellipsis, "cannot use ... in call to non-variadic %s", call.Fun)
			return
		}
		// standard_func(a, b, c)
	}

	// check argument count
	switch {
	case nargs < npars:
		check.errorf(call.Rparen, "not enough arguments in call to %s", call.Fun)
		return
	case nargs > npars:
		check.errorf(args[npars].pos(), "too many arguments in call to %s", call.Fun) // report at first extra argument
		return
	}

	// infer type arguments and instantiate signature if necessary
	if len(sig.tparams) > 0 {
		// TODO(gri) provide position information for targs so we can feed
		//           it to the instantiate call for better error reporting
		targs, failed := check.infer(sig.tparams, sig_params, args)
		if targs == nil {
			return // error already reported
		}
		if failed >= 0 {
			// Some type arguments couldn't be inferred. Use
			// bounds type inference to try to make progress.
			if check.conf.InferFromConstraints {
				targs, failed = check.inferB(sig.tparams, targs)
				if targs == nil {
					return // error already reported
				}
			}
			if failed >= 0 {
				// at least one type argument couldn't be inferred
				assert(targs[failed] == nil)
				tpar := sig.tparams[failed]
				ppos := check.fset.Position(tpar.pos).String()
				check.errorf(call.Rparen, "cannot infer %s (%s) (%s)", tpar.name, ppos, targs)
				return
			}
		}
		// all type arguments were inferred sucessfully
		if debug {
			for _, targ := range targs {
				assert(targ != nil)
			}
		}
		//check.dump("### inferred targs = %s", targs)

		// compute result signature
		rsig = check.instantiate(call.Pos(), sig, targs, nil).(*Signature)
		assert(rsig.tparams == nil) // signature is not generic anymore
		check.recordInferred(call, targs, rsig)

		// Optimization: Only if the parameter list was adjusted do we
		// need to compute it from the adjusted list; otherwise we can
		// simply use the result signature's parameter list.
		if adjusted {
			sig_params = check.subst(call.Pos(), sig_params, makeSubstMap(sig.tparams, targs)).(*Tuple)
		} else {
			sig_params = rsig.params
		}
	}

	// check arguments
	// TODO(gri) Possible optimization (may be tricky): We could avoid
	//           checking arguments from which we inferred type arguments.
	for i, a := range args {
		check.assignment(a, sig_params.vars[i].typ, "argument")
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
					check.errorf(e.Sel.Pos(), "%s not declared by package C", sel)
					goto Error
				}
				check.objDecl(exp, nil)
			} else {
				exp = pkg.scope.Lookup(sel)
				if exp == nil {
					if !pkg.fake {
						check.errorf(e.Sel.Pos(), "%s not declared by package %s", sel, pkg.name)
					}
					goto Error
				}
				if !exp.Exported() {
					check.errorf(e.Sel.Pos(), "%s not exported by package %s", sel, pkg.name)
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

	check.exprOrType(x, e.X)
	if x.mode == invalid {
		goto Error
	}

	check.instantiatedOperand(x)

	obj, index, indirect = check.lookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, sel)
	if obj == nil {
		switch {
		case index != nil:
			// TODO(gri) should provide actual type where the conflict happens
			check.errorf(e.Sel.Pos(), "ambiguous selector %s.%s", x.expr, sel)
		case indirect:
			check.errorf(e.Sel.Pos(), "cannot call pointer method %s on %s", sel, x.typ)
		default:
			var why string
			if tpar := x.typ.TypeParam(); tpar != nil {
				// Type parameter bounds don't specify fields, so don't mention "field".
				switch obj := tpar.Bound().obj.(type) {
				case nil:
					why = check.sprintf("type bound for %s has no method %s", x.typ, sel)
				case *TypeName:
					why = check.sprintf("interface %s has no method %s", obj.name, sel)
				}
			} else {
				why = check.sprintf("type %s has no field or method %s", x.typ, sel)
			}

			// Check if capitalization of sel matters and provide better error message in that case.
			if len(sel) > 0 {
				var changeCase string
				if r := rune(sel[0]); unicode.IsUpper(r) {
					changeCase = string(unicode.ToLower(r)) + sel[1:]
				} else {
					changeCase = string(unicode.ToUpper(r)) + sel[1:]
				}
				if obj, _, _ = check.lookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, changeCase); obj != nil {
					why += ", but does have " + changeCase
				}
			}

			check.errorf(e.Sel.Pos(), "%s.%s undefined (%s)", x.expr, sel, why)

		}
		goto Error
	}

	// methods may not have a fully set up signature yet
	if m, _ := obj.(*Func); m != nil {
		// check.dump("### found method %s", m)
		check.objDecl(m, nil)
		// If m has a parameterized receiver type, infer the type parameter
		// values from the actual receiver provided and then substitute the
		// type parameters in the signature accordingly.
		// TODO(gri) factor this code out
		sig := m.typ.(*Signature)
		if len(sig.rparams) > 0 {
			//check.dump("### recv typ = %s", x.typ)
			//check.dump("### method = %s rparams = %s tparams = %s", m, sig.rparams, sig.tparams)
			// The method may have a pointer receiver, but the actually provided receiver
			// may be a (hopefully addressable) non-pointer value, or vice versa. Here we
			// only care about inferring receiver type parameters; to make the inference
			// work, match up pointer-ness of receiver and argument.
			arg := x
			if ptrRecv := isPointer(sig.recv.typ); ptrRecv != isPointer(arg.typ) {
				copy := *arg
				if ptrRecv {
					copy.typ = NewPointer(arg.typ)
				} else {
					copy.typ = arg.typ.(*Pointer).base
				}
				arg = &copy
			}
			targs, failed := check.infer(sig.rparams, NewTuple(sig.recv), []*operand{arg})
			//check.dump("### inferred targs = %s", targs)
			if failed >= 0 {
				// We may reach here if there were other errors (see issue #40056).
				// check.infer will report a follow-up error.
				// TODO(gri) avoid the follow-up error or provide better explanation.
				goto Error
			}
			// Don't modify m. Instead - for now - make a copy of m and use that instead.
			// (If we modify m, some tests will fail; possibly because the m is in use.)
			// TODO(gri) investigate and provide a correct explanation here
			copy := *m
			copy.typ = check.subst(e.Pos(), m.typ, makeSubstMap(sig.rparams, targs))
			obj = &copy
		}
		// TODO(gri) we also need to do substitution for parameterized interface methods
		//           (this breaks code in testdata/linalg.go2 at the moment)
		//           12/20/2019: Is this TODO still correct?
	}

	if x.mode == typexpr {
		// method expression
		m, _ := obj.(*Func)
		if m == nil {
			// TODO(gri) should check if capitalization of sel matters and provide better error message in that case
			check.errorf(e.Sel.Pos(), "%s.%s undefined (type %s has no method %s)", x.expr, sel, x.typ, sel)
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
			tparams:  sig.tparams,
			params:   NewTuple(append([]*Var{NewVar(token.NoPos, check.pkg, "_", x.typ)}, params...)...),
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

// instantiatedOperand reports an error of x is an uninstantiated (generic) type and sets x.typ to Typ[Invalid].
func (check *Checker) instantiatedOperand(x *operand) {
	if x.mode == typexpr && isGeneric(x.typ) {
		check.errorf(x.pos(), "cannot use generic type %s without instantiation", x.typ)
		x.typ = Typ[Invalid]
	}
}
