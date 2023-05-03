// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of call and selector expressions.

package types

import (
	"fmt"
	"go/ast"
	"go/internal/typeparams"
	"go/token"
	. "internal/types/errors"
	"strings"
	"unicode"
)

// funcInst type-checks a function instantiation and returns the result in x.
// The incoming x must be an uninstantiated generic function. If ix != nil,
// it provides (some or all of) the type arguments (ix.Indices) for the
// instantiation. If the target type tsig != nil, the signature's parameter
// types are used to infer additional missing type arguments of x, if any.
// At least one of tsig or ix must be provided.
func (check *Checker) funcInst(tsig *Signature, pos token.Pos, x *operand, ix *typeparams.IndexExpr) {
	assert(tsig != nil || ix != nil)

	var instErrPos positioner
	if ix != nil {
		instErrPos = inNode(ix.Orig, ix.Lbrack)
	} else {
		instErrPos = atPos(pos)
	}
	versionErr := !check.verifyVersionf(check.pkg, instErrPos, go1_18, "function instantiation")

	// targs and xlist are the type arguments and corresponding type expressions, or nil.
	var targs []Type
	var xlist []ast.Expr
	if ix != nil {
		xlist = ix.Indices
		targs = check.typeList(xlist)
		if targs == nil {
			x.mode = invalid
			x.expr = ix
			return
		}
		assert(len(targs) == len(xlist))
	}

	// Check the number of type arguments (got) vs number of type parameters (want).
	// Note that x is a function value, not a type expression, so we don't need to
	// call under below.
	sig := x.typ.(*Signature)
	got, want := len(targs), sig.TypeParams().Len()
	if got > want {
		// Providing too many type arguments is always an error.
		check.errorf(ix.Indices[got-1], WrongTypeArgCount, "got %d type arguments but want %d", got, want)
		x.mode = invalid
		x.expr = ix.Orig
		return
	}

	if got < want {
		// If the uninstantiated or partially instantiated function x is used in an
		// assignment (tsig != nil), use the respective function parameter and result
		// types to infer additional type arguments.
		var args []*operand
		var params []*Var
		if tsig != nil && sig.tparams != nil && tsig.params.Len() == sig.params.Len() && tsig.results.Len() == sig.results.Len() {
			// x is a generic function and the signature arity matches the target function.
			// To infer x's missing type arguments, treat the function assignment as a call
			// of a synthetic function f where f's parameters are the parameters and results
			// of x and where the arguments to the call of f are values of the parameter and
			// result types of x.
			if !versionErr && !check.allowVersion(check.pkg, instErrPos, go1_21) {
				if ix != nil {
					check.versionErrorf(instErrPos, go1_21, "partially instantiated function in assignment")
				} else {
					check.versionErrorf(instErrPos, go1_21, "implicitly instantiated function in assignment")
				}
			}
			n := tsig.params.Len()
			m := tsig.results.Len()
			args = make([]*operand, n+m)
			params = make([]*Var, n+m)
			for i := 0; i < n; i++ {
				lvar := tsig.params.At(i)
				lname := ast.NewIdent(paramName(lvar.name, i, "parameter"))
				lname.NamePos = x.Pos() // correct position
				args[i] = &operand{mode: value, expr: lname, typ: lvar.typ}
				params[i] = sig.params.At(i)
			}
			for i := 0; i < m; i++ {
				lvar := tsig.results.At(i)
				lname := ast.NewIdent(paramName(lvar.name, i, "result parameter"))
				lname.NamePos = x.Pos() // correct position
				args[n+i] = &operand{mode: value, expr: lname, typ: lvar.typ}
				params[n+i] = sig.results.At(i)
			}
		}

		// Rename type parameters to avoid problems with recursive instantiations.
		// Note that NewTuple(params...) below is nil if len(params) == 0, as desired.
		tparams, params2 := check.renameTParams(pos, sig.TypeParams().list(), NewTuple(params...))

		targs = check.infer(atPos(pos), tparams, targs, params2, args)
		if targs == nil {
			// error was already reported
			x.mode = invalid
			x.expr = ix // TODO(gri) is this correct?
			return
		}
		got = len(targs)
	}
	assert(got == want)

	// instantiate function signature
	sig = check.instantiateSignature(x.Pos(), sig, targs, xlist)
	assert(sig.TypeParams().Len() == 0) // signature is not generic anymore

	x.typ = sig
	x.mode = value
	// If we don't have an index expression, keep the existing expression of x.
	if ix != nil {
		x.expr = ix.Orig
	}
	check.recordInstance(x.expr, targs, sig)
}

func paramName(name string, i int, kind string) string {
	if name != "" {
		return name
	}
	return nth(i+1) + " " + kind
}

func nth(n int) string {
	switch n {
	case 1:
		return "1st"
	case 2:
		return "2nd"
	case 3:
		return "3rd"
	}
	return fmt.Sprintf("%dth", n)
}

func (check *Checker) instantiateSignature(pos token.Pos, typ *Signature, targs []Type, xlist []ast.Expr) (res *Signature) {
	assert(check != nil)
	assert(len(targs) == typ.TypeParams().Len())

	if check.conf._Trace {
		check.trace(pos, "-- instantiating signature %s with %s", typ, targs)
		check.indent++
		defer func() {
			check.indent--
			check.trace(pos, "=> %s (under = %s)", res, res.Underlying())
		}()
	}

	inst := check.instance(pos, typ, targs, nil, check.context()).(*Signature)
	assert(len(xlist) <= len(targs))

	// verify instantiation lazily (was go.dev/issue/50450)
	check.later(func() {
		tparams := typ.TypeParams().list()
		if i, err := check.verify(pos, tparams, targs, check.context()); err != nil {
			// best position for error reporting
			pos := pos
			if i < len(xlist) {
				pos = xlist[i].Pos()
			}
			check.softErrorf(atPos(pos), InvalidTypeArg, "%s", err)
		} else {
			check.mono.recordInstance(check.pkg, pos, tparams, targs, xlist)
		}
	}).describef(atPos(pos), "verify instantiation")

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
		check.nonGeneric(nil, x)
		if x.mode == invalid {
			return conversion
		}
		T := x.typ
		x.mode = invalid
		switch n := len(call.Args); n {
		case 0:
			check.errorf(inNode(call, call.Rparen), WrongArgCount, "missing argument in conversion to %s", T)
		case 1:
			check.expr(nil, x, call.Args[0])
			if x.mode != invalid {
				if call.Ellipsis.IsValid() {
					check.errorf(call.Args[0], BadDotDotDotSyntax, "invalid use of ... in conversion to %s", T)
					break
				}
				if t, _ := under(T).(*Interface); t != nil && !isTypeParam(T) {
					if !t.IsMethodSet() {
						check.errorf(call, MisplacedConstraintIface, "cannot use interface %s in conversion (contains specific type constraints or is comparable)", T)
						break
					}
				}
				check.conversion(x, T)
			}
		default:
			check.use(call.Args...)
			check.errorf(call.Args[n-1], WrongArgCount, "too many arguments in conversion to %s", T)
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
		check.errorf(x, InvalidCall, invalidOp+"cannot call non-function %s", x)
		x.mode = invalid
		x.expr = call
		return statement
	}

	// Capture wasGeneric before sig is potentially instantiated below.
	wasGeneric := sig.TypeParams().Len() > 0

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
			check.errorf(xlist[want], WrongTypeArgCount, "got %d type arguments but want %d", got, want)
			check.use(call.Args...)
			x.mode = invalid
			x.expr = call
			return statement
		}

		// If sig is generic and all type arguments are provided, preempt function
		// argument type inference by explicitly instantiating the signature. This
		// ensures that we record accurate type information for sig, even if there
		// is an error checking its arguments (for example, if an incorrect number
		// of arguments is supplied).
		if got == want && want > 0 {
			check.verifyVersionf(check.pkg, atPos(ix.Lbrack), go1_18, "function instantiation")

			sig = check.instantiateSignature(ix.Pos(), sig, targs, xlist)
			assert(sig.TypeParams().Len() == 0) // signature is not generic anymore
			check.recordInstance(ix.Orig, targs, sig)

			// targs have been consumed; proceed with checking arguments of the
			// non-generic signature.
			targs = nil
			xlist = nil
		}
	}

	// evaluate arguments
	args := check.genericExprList(call.Args)
	sig = check.arguments(call, sig, targs, args, xlist)

	if wasGeneric && sig.TypeParams().Len() == 0 {
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

	// if type inference failed, a parameterized result must be invalidated
	// (operands cannot have a parameterized type)
	if x.mode == value && sig.TypeParams().Len() > 0 && isParameterized(sig.TypeParams().list(), x.typ) {
		x.mode = invalid
	}

	return statement
}

// exprList evaluates a list of expressions and returns the corresponding operands.
// A single-element expression list may evaluate to multiple operands.
func (check *Checker) exprList(elist []ast.Expr) (xlist []*operand) {
	switch len(elist) {
	case 0:
		// nothing to do
	case 1:
		xlist, _ = check.multiExpr(elist[0], false)
	default:
		// multiple (possibly invalid) values
		xlist = make([]*operand, len(elist))
		for i, e := range elist {
			var x operand
			check.expr(nil, &x, e)
			xlist[i] = &x
		}
	}
	return
}

// genericExprList is like exprList but result operands may be generic (not fully instantiated).
func (check *Checker) genericExprList(elist []ast.Expr) (xlist []*operand) {
	switch len(elist) {
	case 0:
		// nothing to do
	case 1:
		xlist = check.genericMultiExpr(elist[0])
	default:
		// multiple (possibly invalid) values
		xlist = make([]*operand, len(elist))
		for i, e := range elist {
			var x operand
			check.genericExpr(&x, e)
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
			check.errorf(a, NotAnExpr, "%s used as value", a)
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
	adjusted := false       // indicates if sigParams is different from sig.params
	if sig.variadic {
		if ddd {
			// variadic_func(a, b, c...)
			if len(call.Args) == 1 && nargs > 1 {
				// f()... is not permitted if f() is multi-valued
				check.errorf(inNode(call, call.Ellipsis), InvalidDotDotDot, "cannot use ... with %d-valued %s", nargs, call.Args[0])
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
			check.errorf(inNode(call, call.Ellipsis), NonVariadicDotDotDot, "cannot use ... in call to non-variadic %s", call.Fun)
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
		err := newErrorf(at, WrongArgCount, "%s arguments in call to %s", qualifier, call.Fun)
		err.errorf(nopos, "have %s", check.typesSummary(operandTypes(args), false))
		err.errorf(nopos, "want %s", check.typesSummary(varTypes(params), sig.variadic))
		check.report(err)
		return
	}

	// collect type parameters of callee and generic function arguments
	var tparams []*TypeParam

	// collect type parameters of callee
	n := sig.TypeParams().Len()
	if n > 0 {
		if !check.allowVersion(check.pkg, call, go1_18) {
			switch call.Fun.(type) {
			case *ast.IndexExpr, *ast.IndexListExpr:
				ix := typeparams.UnpackIndexExpr(call.Fun)
				check.versionErrorf(inNode(call.Fun, ix.Lbrack), go1_18, "function instantiation")
			default:
				check.versionErrorf(inNode(call, call.Lparen), go1_18, "implicit function instantiation")
			}
		}
		// rename type parameters to avoid problems with recursive calls
		tparams, sigParams = check.renameTParams(call.Pos(), sig.TypeParams().list(), sigParams)
	}

	// collect type parameters from generic function arguments
	var genericArgs []int // indices of generic function arguments
	if check.conf._EnableReverseTypeInference {
		for i, arg := range args {
			// generic arguments cannot have a defined (*Named) type - no need for underlying type below
			if asig, _ := arg.typ.(*Signature); asig != nil && asig.TypeParams().Len() > 0 {
				// TODO(gri) need to also rename type parameters for cases like f(g, g)
				tparams = append(tparams, asig.TypeParams().list()...)
				genericArgs = append(genericArgs, i)
			}
		}
	}
	// at the moment we only support implicit instantiations of argument functions
	_ = len(genericArgs) > 0 && check.verifyVersionf(check.pkg, args[genericArgs[0]], go1_21, "implicitly instantiated function as argument")

	// tparams holds the type parameters of the callee and generic function arguments, if any:
	// the first n type parameters belong to the callee, followed by mi type parameters for each
	// of the generic function arguments, where mi = args[i].typ.(*Signature).TypeParams().Len().

	// infer missing type arguments of callee and function arguments
	if len(tparams) > 0 {
		targs = check.infer(call, tparams, targs, sigParams, args)
		if targs == nil {
			// TODO(gri) If infer inferred the first targs[:n], consider instantiating
			//           the call signature for better error messages/gopls behavior.
			//           Perhaps instantiate as much as we can, also for arguments.
			//           This will require changes to how infer returns its results.
			return // error already reported
		}

		// compute result signature: instantiate if needed
		rsig = sig
		if n > 0 {
			rsig = check.instantiateSignature(call.Pos(), sig, targs[:n], xlist)
			assert(rsig.TypeParams().Len() == 0) // signature is not generic anymore
			check.recordInstance(call.Fun, targs[:n], rsig)
		}

		// Optimization: Only if the callee's parameter list was adjusted do we need to
		// compute it from the adjusted list; otherwise we can simply use the result
		// signature's parameter list. We only need the n type parameters and arguments
		// of the callee.
		if n > 0 && adjusted {
			sigParams = check.subst(call.Pos(), sigParams, makeSubstMap(tparams[:n], targs[:n]), nil, check.context()).(*Tuple)
		} else {
			sigParams = rsig.params
		}

		// compute argument signatures: instantiate if needed
		j := n
		for _, i := range genericArgs {
			asig := args[i].typ.(*Signature)
			k := j + asig.TypeParams().Len()
			// targs[j:k] are the inferred type arguments for asig
			asig = check.instantiateSignature(call.Pos(), asig, targs[j:k], nil) // TODO(gri) provide xlist if possible (partial instantiations)
			assert(asig.TypeParams().Len() == 0)                                 // signature is not generic anymore
			args[i].typ = asig
			check.recordInstance(args[i].expr, targs[j:k], asig)
			j = k
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

func (check *Checker) selector(x *operand, e *ast.SelectorExpr, def *Named, wantType bool) {
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
					check.errorf(e.Sel, UndeclaredImportedName, "undefined: %s", ast.Expr(e)) // cast to ast.Expr to silence vet
					goto Error
				}
				check.objDecl(exp, nil)
			} else {
				exp = pkg.scope.Lookup(sel)
				if exp == nil {
					if !pkg.fake {
						check.errorf(e.Sel, UndeclaredImportedName, "undefined: %s", ast.Expr(e))
					}
					goto Error
				}
				if !exp.Exported() {
					check.errorf(e.Sel, UnexportedName, "%s not exported by package %s", sel, pkg.name)
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
		// don't crash for "type T T.x" (was go.dev/issue/51509)
		if def != nil && x.typ == def {
			check.cycleError([]Object{def.obj})
			goto Error
		}
	case builtin:
		// types2 uses the position of '.' for the error
		check.errorf(e.Sel, UncalledBuiltin, "cannot select on %s", x)
		goto Error
	case invalid:
		goto Error
	}

	// Avoid crashing when checking an invalid selector in a method declaration
	// (i.e., where def is not set):
	//
	//   type S[T any] struct{}
	//   type V = S[any]
	//   func (fs *S[T]) M(x V.M) {}
	//
	// All codepaths below return a non-type expression. If we get here while
	// expecting a type expression, it is an error.
	//
	// See go.dev/issue/57522 for more details.
	//
	// TODO(rfindley): We should do better by refusing to check selectors in all cases where
	// x.typ is incomplete.
	if wantType {
		check.errorf(e.Sel, NotAType, "%s is not a type", ast.Expr(e))
		goto Error
	}

	obj, index, indirect = LookupFieldOrMethod(x.typ, x.mode == variable, check.pkg, sel)
	if obj == nil {
		// Don't report another error if the underlying type was invalid (go.dev/issue/49541).
		if under(x.typ) == Typ[Invalid] {
			goto Error
		}

		if index != nil {
			// TODO(gri) should provide actual type where the conflict happens
			check.errorf(e.Sel, AmbiguousSelector, "ambiguous selector %s.%s", x.expr, sel)
			goto Error
		}

		if indirect {
			if x.mode == typexpr {
				check.errorf(e.Sel, InvalidMethodExpr, "invalid method expression %s.%s (needs pointer receiver (*%s).%s)", x.typ, sel, x.typ, sel)
			} else {
				check.errorf(e.Sel, InvalidMethodExpr, "cannot call pointer method %s on %s", sel, x.typ)
			}
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
		check.errorf(e.Sel, MissingFieldOrMethod, "%s.%s undefined (%s)", x.expr, sel, why)
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
			check.errorf(e.Sel, MissingFieldOrMethod, "%s.%s undefined (type %s has no method %s)", x.expr, sel, x.typ, sel)
			goto Error
		}

		check.recordSelection(e, MethodExpr, x.typ, m, index, indirect)

		sig := m.typ.(*Signature)
		if sig.recv == nil {
			check.error(e, InvalidDeclCycle, "illegal cycle in method declaration")
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
// (and variables are "used") in the presence of
// other errors. Arguments may be nil.
// Reports if all arguments evaluated without error.
func (check *Checker) use(args ...ast.Expr) bool { return check.useN(args, false) }

// useLHS is like use, but doesn't "use" top-level identifiers.
// It should be called instead of use if the arguments are
// expressions on the lhs of an assignment.
func (check *Checker) useLHS(args ...ast.Expr) bool { return check.useN(args, true) }

func (check *Checker) useN(args []ast.Expr, lhs bool) bool {
	ok := true
	for _, e := range args {
		if !check.use1(e, lhs) {
			ok = false
		}
	}
	return ok
}

func (check *Checker) use1(e ast.Expr, lhs bool) bool {
	var x operand
	x.mode = value // anything but invalid
	switch n := unparen(e).(type) {
	case nil:
		// nothing to do
	case *ast.Ident:
		// don't report an error evaluating blank
		if n.Name == "_" {
			break
		}
		// If the lhs is an identifier denoting a variable v, this assignment
		// is not a 'use' of v. Remember current value of v.used and restore
		// after evaluating the lhs via check.rawExpr.
		var v *Var
		var v_used bool
		if lhs {
			if _, obj := check.scope.LookupParent(n.Name, nopos); obj != nil {
				// It's ok to mark non-local variables, but ignore variables
				// from other packages to avoid potential race conditions with
				// dot-imported variables.
				if w, _ := obj.(*Var); w != nil && w.pkg == check.pkg {
					v = w
					v_used = v.used
				}
			}
		}
		check.exprOrType(&x, n, true)
		if v != nil {
			v.used = v_used // restore v.used
		}
	default:
		check.rawExpr(nil, &x, e, nil, true)
	}
	return x.mode != invalid
}
