// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"

	"fmt"
)

// package all the arguments that match a ... T parameter into a []T.
func MakeDotArgs(typ *types.Type, args []ir.Node) ir.Node {
	var n ir.Node
	if len(args) == 0 {
		n = NodNil()
		n.SetType(typ)
	} else {
		lit := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(typ).(ir.Ntype), nil)
		lit.List.Append(args...)
		lit.SetImplicit(true)
		n = lit
	}

	n = Expr(n)
	if n.Type() == nil {
		base.Fatalf("mkdotargslice: typecheck failed")
	}
	return n
}

// FixVariadicCall rewrites calls to variadic functions to use an
// explicit ... argument if one is not already present.
func FixVariadicCall(call *ir.CallExpr) {
	fntype := call.X.Type()
	if !fntype.IsVariadic() || call.IsDDD {
		return
	}

	vi := fntype.NumParams() - 1
	vt := fntype.Params().Field(vi).Type

	args := call.Args
	extra := args[vi:]
	slice := MakeDotArgs(vt, extra)
	for i := range extra {
		extra[i] = nil // allow GC
	}

	call.Args.Set(append(args[:vi], slice))
	call.IsDDD = true
}

// ClosureType returns the struct type used to hold all the information
// needed in the closure for clo (clo must be a OCLOSURE node).
// The address of a variable of the returned type can be cast to a func.
func ClosureType(clo *ir.ClosureExpr) *types.Type {
	// Create closure in the form of a composite literal.
	// supposing the closure captures an int i and a string s
	// and has one float64 argument and no results,
	// the generated code looks like:
	//
	//	clos = &struct{.F uintptr; i *int; s *string}{func.1, &i, &s}
	//
	// The use of the struct provides type information to the garbage
	// collector so that it can walk the closure. We could use (in this case)
	// [3]unsafe.Pointer instead, but that would leave the gc in the dark.
	// The information appears in the binary in the form of type descriptors;
	// the struct is unnamed so that closures in multiple packages with the
	// same struct type can share the descriptor.
	fields := []*ir.Field{
		ir.NewField(base.Pos, Lookup(".F"), nil, types.Types[types.TUINTPTR]),
	}
	for _, v := range clo.Func.ClosureVars {
		typ := v.Type()
		if !v.Byval() {
			typ = types.NewPtr(typ)
		}
		fields = append(fields, ir.NewField(base.Pos, v.Sym(), nil, typ))
	}
	typ := NewStructType(fields)
	typ.SetNoalg(true)
	return typ
}

// PartialCallType returns the struct type used to hold all the information
// needed in the closure for n (n must be a OCALLPART node).
// The address of a variable of the returned type can be cast to a func.
func PartialCallType(n *ir.CallPartExpr) *types.Type {
	t := NewStructType([]*ir.Field{
		ir.NewField(base.Pos, Lookup("F"), nil, types.Types[types.TUINTPTR]),
		ir.NewField(base.Pos, Lookup("R"), nil, n.X.Type()),
	})
	t.SetNoalg(true)
	return t
}

// CaptureVars is called in a separate phase after all typechecking is done.
// It decides whether each variable captured by a closure should be captured
// by value or by reference.
// We use value capturing for values <= 128 bytes that are never reassigned
// after capturing (effectively constant).
func CaptureVars(fn *ir.Func) {
	lno := base.Pos
	base.Pos = fn.Pos()
	cvars := fn.ClosureVars
	out := cvars[:0]
	for _, v := range cvars {
		if v.Type() == nil {
			// If v.Type is nil, it means v looked like it
			// was going to be used in the closure, but
			// isn't. This happens in struct literals like
			// s{f: x} where we can't distinguish whether
			// f is a field identifier or expression until
			// resolving s.
			continue
		}
		out = append(out, v)

		// type check the & of closed variables outside the closure,
		// so that the outer frame also grabs them and knows they escape.
		types.CalcSize(v.Type())

		var outer ir.Node
		outer = v.Outer
		outermost := v.Defn.(*ir.Name)

		// out parameters will be assigned to implicitly upon return.
		if outermost.Class_ != ir.PPARAMOUT && !outermost.Name().Addrtaken() && !outermost.Name().Assigned() && v.Type().Width <= 128 {
			v.SetByval(true)
		} else {
			outermost.Name().SetAddrtaken(true)
			outer = NodAddr(outer)
		}

		if base.Flag.LowerM > 1 {
			var name *types.Sym
			if v.Curfn != nil && v.Curfn.Nname != nil {
				name = v.Curfn.Sym()
			}
			how := "ref"
			if v.Byval() {
				how = "value"
			}
			base.WarnfAt(v.Pos(), "%v capturing by %s: %v (addr=%v assign=%v width=%d)", name, how, v.Sym(), outermost.Name().Addrtaken(), outermost.Name().Assigned(), int32(v.Type().Width))
		}

		outer = Expr(outer)
		fn.ClosureEnter.Append(outer)
	}

	fn.ClosureVars = out
	base.Pos = lno
}

// typecheckclosure typechecks an OCLOSURE node. It also creates the named
// function associated with the closure.
// TODO: This creation of the named function should probably really be done in a
// separate pass from type-checking.
func typecheckclosure(clo *ir.ClosureExpr, top int) {
	fn := clo.Func
	// Set current associated iota value, so iota can be used inside
	// function in ConstSpec, see issue #22344
	if x := getIotaValue(); x >= 0 {
		fn.Iota = x
	}

	fn.ClosureType = check(fn.ClosureType, ctxType)
	clo.SetType(fn.ClosureType.Type())
	fn.SetClosureCalled(top&ctxCallee != 0)

	// Do not typecheck fn twice, otherwise, we will end up pushing
	// fn to Target.Decls multiple times, causing initLSym called twice.
	// See #30709
	if fn.Typecheck() == 1 {
		return
	}

	for _, ln := range fn.ClosureVars {
		n := ln.Defn
		if !n.Name().Captured() {
			n.Name().SetCaptured(true)
			if n.Name().Decldepth == 0 {
				base.Fatalf("typecheckclosure: var %v does not have decldepth assigned", n)
			}

			// Ignore assignments to the variable in straightline code
			// preceding the first capturing by a closure.
			if n.Name().Decldepth == decldepth {
				n.Name().SetAssigned(false)
			}
		}
	}

	fn.Nname.SetSym(closurename(ir.CurFunc))
	ir.MarkFunc(fn.Nname)
	Func(fn)

	// Type check the body now, but only if we're inside a function.
	// At top level (in a variable initialization: curfn==nil) we're not
	// ready to type check code yet; we'll check it later, because the
	// underlying closure function we create is added to Target.Decls.
	if ir.CurFunc != nil && clo.Type() != nil {
		oldfn := ir.CurFunc
		ir.CurFunc = fn
		olddd := decldepth
		decldepth = 1
		Stmts(fn.Body)
		decldepth = olddd
		ir.CurFunc = oldfn
	}

	Target.Decls = append(Target.Decls, fn)
}

// Lazy typechecking of imported bodies. For local functions, caninl will set ->typecheck
// because they're a copy of an already checked body.
func ImportedBody(fn *ir.Func) {
	lno := ir.SetPos(fn.Nname)

	ImportBody(fn)

	// typecheckinl is only for imported functions;
	// their bodies may refer to unsafe as long as the package
	// was marked safe during import (which was checked then).
	// the ->inl of a local function has been typechecked before caninl copied it.
	pkg := fnpkg(fn.Nname)

	if pkg == types.LocalPkg || pkg == nil {
		return // typecheckinl on local function
	}

	if base.Flag.LowerM > 2 || base.Debug.Export != 0 {
		fmt.Printf("typecheck import [%v] %L { %v }\n", fn.Sym(), fn, ir.Nodes(fn.Inl.Body))
	}

	savefn := ir.CurFunc
	ir.CurFunc = fn
	Stmts(fn.Inl.Body)
	ir.CurFunc = savefn

	// During expandInline (which imports fn.Func.Inl.Body),
	// declarations are added to fn.Func.Dcl by funcHdr(). Move them
	// to fn.Func.Inl.Dcl for consistency with how local functions
	// behave. (Append because typecheckinl may be called multiple
	// times.)
	fn.Inl.Dcl = append(fn.Inl.Dcl, fn.Dcl...)
	fn.Dcl = nil

	base.Pos = lno
}

// Get the function's package. For ordinary functions it's on the ->sym, but for imported methods
// the ->sym can be re-used in the local package, so peel it off the receiver's type.
func fnpkg(fn *ir.Name) *types.Pkg {
	if ir.IsMethod(fn) {
		// method
		rcvr := fn.Type().Recv().Type

		if rcvr.IsPtr() {
			rcvr = rcvr.Elem()
		}
		if rcvr.Sym() == nil {
			base.Fatalf("receiver with no sym: [%v] %L  (%v)", fn.Sym(), fn, rcvr)
		}
		return rcvr.Sym().Pkg
	}

	// non-method
	return fn.Sym().Pkg
}

// CaptureVarsComplete is set to true when the capturevars phase is done.
var CaptureVarsComplete bool

// closurename generates a new unique name for a closure within
// outerfunc.
func closurename(outerfunc *ir.Func) *types.Sym {
	outer := "glob."
	prefix := "func"
	gen := &globClosgen

	if outerfunc != nil {
		if outerfunc.OClosure != nil {
			prefix = ""
		}

		outer = ir.FuncName(outerfunc)

		// There may be multiple functions named "_". In those
		// cases, we can't use their individual Closgens as it
		// would lead to name clashes.
		if !ir.IsBlank(outerfunc.Nname) {
			gen = &outerfunc.Closgen
		}
	}

	*gen++
	return Lookup(fmt.Sprintf("%s.%s%d", outer, prefix, *gen))
}

// globClosgen is like Func.Closgen, but for the global scope.
var globClosgen int32

// makepartialcall returns a DCLFUNC node representing the wrapper function (*-fm) needed
// for partial calls.
func makepartialcall(dot *ir.SelectorExpr, t0 *types.Type, meth *types.Sym) *ir.Func {
	rcvrtype := dot.X.Type()
	sym := ir.MethodSymSuffix(rcvrtype, meth, "-fm")

	if sym.Uniq() {
		return sym.Def.(*ir.Func)
	}
	sym.SetUniq(true)

	savecurfn := ir.CurFunc
	saveLineNo := base.Pos
	ir.CurFunc = nil

	// Set line number equal to the line number where the method is declared.
	var m *types.Field
	if lookdot0(meth, rcvrtype, &m, false) == 1 && m.Pos.IsKnown() {
		base.Pos = m.Pos
	}
	// Note: !m.Pos.IsKnown() happens for method expressions where
	// the method is implicitly declared. The Error method of the
	// built-in error type is one such method.  We leave the line
	// number at the use of the method expression in this
	// case. See issue 29389.

	tfn := ir.NewFuncType(base.Pos, nil,
		NewFuncParams(t0.Params(), true),
		NewFuncParams(t0.Results(), false))

	fn := DeclFunc(sym, tfn)
	fn.SetDupok(true)
	fn.SetNeedctxt(true)

	// Declare and initialize variable holding receiver.
	cr := ir.NewClosureRead(rcvrtype, types.Rnd(int64(types.PtrSize), int64(rcvrtype.Align)))
	ptr := NewName(Lookup(".this"))
	Declare(ptr, ir.PAUTO)
	ptr.SetUsed(true)
	var body []ir.Node
	if rcvrtype.IsPtr() || rcvrtype.IsInterface() {
		ptr.SetType(rcvrtype)
		body = append(body, ir.NewAssignStmt(base.Pos, ptr, cr))
	} else {
		ptr.SetType(types.NewPtr(rcvrtype))
		body = append(body, ir.NewAssignStmt(base.Pos, ptr, NodAddr(cr)))
	}

	call := ir.NewCallExpr(base.Pos, ir.OCALL, ir.NewSelectorExpr(base.Pos, ir.OXDOT, ptr, meth), nil)
	call.Args.Set(ir.ParamNames(tfn.Type()))
	call.IsDDD = tfn.Type().IsVariadic()
	if t0.NumResults() != 0 {
		ret := ir.NewReturnStmt(base.Pos, nil)
		ret.Results = []ir.Node{call}
		body = append(body, ret)
	} else {
		body = append(body, call)
	}

	fn.Body.Set(body)
	FinishFuncBody()

	Func(fn)
	// Need to typecheck the body of the just-generated wrapper.
	// typecheckslice() requires that Curfn is set when processing an ORETURN.
	ir.CurFunc = fn
	Stmts(fn.Body)
	sym.Def = fn
	Target.Decls = append(Target.Decls, fn)
	ir.CurFunc = savecurfn
	base.Pos = saveLineNo

	return fn
}

func typecheckpartialcall(n ir.Node, sym *types.Sym) *ir.CallPartExpr {
	switch n.Op() {
	case ir.ODOTINTER, ir.ODOTMETH:
		break

	default:
		base.Fatalf("invalid typecheckpartialcall")
	}
	dot := n.(*ir.SelectorExpr)

	// Create top-level function.
	fn := makepartialcall(dot, dot.Type(), sym)
	fn.SetWrapper(true)

	return ir.NewCallPartExpr(dot.Pos(), dot.X, dot.Selection, fn)
}
