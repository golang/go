// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"

	"fmt"
	"go/constant"
	"go/token"
)

// MakeDotArgs package all the arguments that match a ... T parameter into a []T.
func MakeDotArgs(pos src.XPos, typ *types.Type, args []ir.Node) ir.Node {
	var n ir.Node
	if len(args) == 0 {
		n = ir.NewNilExpr(pos)
		n.SetType(typ)
	} else {
		args = append([]ir.Node(nil), args...)
		lit := ir.NewCompLitExpr(pos, ir.OCOMPLIT, ir.TypeNode(typ), args)
		lit.SetImplicit(true)
		n = lit
	}

	n = Expr(n)
	if n.Type() == nil {
		base.FatalfAt(pos, "mkdotargslice: typecheck failed")
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
	slice := MakeDotArgs(call.Pos(), vt, extra)
	for i := range extra {
		extra[i] = nil // allow GC
	}

	call.Args = append(args[:vi], slice)
	call.IsDDD = true
}

// FixMethodCall rewrites a method call t.M(...) into a function call T.M(t, ...).
func FixMethodCall(call *ir.CallExpr) {
	if call.X.Op() != ir.ODOTMETH {
		return
	}

	dot := call.X.(*ir.SelectorExpr)

	fn := Expr(ir.NewSelectorExpr(dot.Pos(), ir.OXDOT, ir.TypeNode(dot.X.Type()), dot.Selection.Sym))

	args := make([]ir.Node, 1+len(call.Args))
	args[0] = dot.X
	copy(args[1:], call.Args)

	call.SetOp(ir.OCALLFUNC)
	call.X = fn
	call.Args = args
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

	// Make sure the .F field is in the same package as the rest of the
	// fields. This deals with closures in instantiated functions, which are
	// compiled as if from the source package of the generic function.
	var pkg *types.Pkg
	if len(clo.Func.ClosureVars) == 0 {
		pkg = types.LocalPkg
	} else {
		for _, v := range clo.Func.ClosureVars {
			if pkg == nil {
				pkg = v.Sym().Pkg
			} else if pkg != v.Sym().Pkg {
				base.Fatalf("Closure variables from multiple packages")
			}
		}
	}

	fields := []*types.Field{
		types.NewField(base.Pos, pkg.Lookup(".F"), types.Types[types.TUINTPTR]),
	}
	for _, v := range clo.Func.ClosureVars {
		typ := v.Type()
		if !v.Byval() {
			typ = types.NewPtr(typ)
		}
		fields = append(fields, types.NewField(base.Pos, v.Sym(), typ))
	}
	typ := types.NewStruct(types.NoPkg, fields)
	typ.SetNoalg(true)
	return typ
}

// MethodValueType returns the struct type used to hold all the information
// needed in the closure for a OMETHVALUE node. The address of a variable of
// the returned type can be cast to a func.
func MethodValueType(n *ir.SelectorExpr) *types.Type {
	t := types.NewStruct(types.NoPkg, []*types.Field{
		types.NewField(base.Pos, Lookup("F"), types.Types[types.TUINTPTR]),
		types.NewField(base.Pos, Lookup("R"), n.X.Type()),
	})
	t.SetNoalg(true)
	return t
}

// True if we are typechecking an inline body in ImportedBody below. We use this
// flag to not create a new closure function in tcClosure when we are just
// typechecking an inline body, as opposed to the body of a real function.
var inTypeCheckInl bool

// ImportedBody returns immediately if the inlining information for fn is
// populated. Otherwise, fn must be an imported function. If so, ImportedBody
// loads in the dcls and body for fn, and typechecks as needed.
func ImportedBody(fn *ir.Func) {
	if fn.Inl.Body != nil {
		return
	}
	lno := ir.SetPos(fn.Nname)

	// When we load an inlined body, we need to allow OADDR
	// operations on untyped expressions. We will fix the
	// addrtaken flags on all the arguments of the OADDR with the
	// computeAddrtaken call below (after we typecheck the body).
	// TODO: export/import types and addrtaken marks along with inlined bodies,
	// so this will be unnecessary.
	IncrementalAddrtaken = false
	defer func() {
		if DirtyAddrtaken {
			// We do ComputeAddrTaken on function instantiations, but not
			// generic functions (since we may not yet know if x in &x[i]
			// is an array or a slice).
			if !fn.Type().HasTParam() {
				ComputeAddrtaken(fn.Inl.Body) // compute addrtaken marks once types are available
			}
			DirtyAddrtaken = false
		}
		IncrementalAddrtaken = true
	}()

	ImportBody(fn)

	// Stmts(fn.Inl.Body) below is only for imported functions;
	// their bodies may refer to unsafe as long as the package
	// was marked safe during import (which was checked then).
	// the ->inl of a local function has been typechecked before CanInline copied it.
	pkg := fnpkg(fn.Nname)

	if pkg == types.LocalPkg || pkg == nil {
		return // ImportedBody on local function
	}

	if base.Flag.LowerM > 2 || base.Debug.Export != 0 {
		fmt.Printf("typecheck import [%v] %L { %v }\n", fn.Sym(), fn, ir.Nodes(fn.Inl.Body))
	}

	if !go117ExportTypes {
		// If we didn't export & import types, typecheck the code here.
		savefn := ir.CurFunc
		ir.CurFunc = fn
		if inTypeCheckInl {
			base.Fatalf("inTypeCheckInl should not be set recursively")
		}
		inTypeCheckInl = true
		Stmts(fn.Inl.Body)
		inTypeCheckInl = false
		ir.CurFunc = savefn
	}

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

// tcClosure typechecks an OCLOSURE node. It also creates the named
// function associated with the closure.
// TODO: This creation of the named function should probably really be done in a
// separate pass from type-checking.
func tcClosure(clo *ir.ClosureExpr, top int) ir.Node {
	fn := clo.Func

	// We used to allow IR builders to typecheck the underlying Func
	// themselves, but that led to too much variety and inconsistency
	// around who's responsible for naming the function, typechecking
	// it, or adding it to Target.Decls.
	//
	// It's now all or nothing. Callers are still allowed to do these
	// themselves, but then they assume responsibility for all of them.
	if fn.Typecheck() == 1 {
		base.FatalfAt(fn.Pos(), "underlying closure func already typechecked: %v", fn)
	}

	ir.NameClosure(clo, ir.CurFunc)
	Func(fn)

	// Type check the body now, but only if we're inside a function.
	// At top level (in a variable initialization: curfn==nil) we're not
	// ready to type check code yet; we'll check it later, because the
	// underlying closure function we create is added to Target.Decls.
	if ir.CurFunc != nil {
		oldfn := ir.CurFunc
		ir.CurFunc = fn
		Stmts(fn.Body)
		ir.CurFunc = oldfn
	}

	out := 0
	for _, v := range fn.ClosureVars {
		if v.Type() == nil {
			// If v.Type is nil, it means v looked like it was going to be
			// used in the closure, but isn't. This happens in struct
			// literals like s{f: x} where we can't distinguish whether f is
			// a field identifier or expression until resolving s.
			continue
		}

		// type check closed variables outside the closure, so that the
		// outer frame also captures them.
		Expr(v.Outer)

		fn.ClosureVars[out] = v
		out++
	}
	fn.ClosureVars = fn.ClosureVars[:out]

	clo.SetType(fn.Type())

	target := Target
	if inTypeCheckInl {
		// We're typechecking an imported function, so it's not actually
		// part of Target. Skip adding it to Target.Decls so we don't
		// compile it again.
		target = nil
	}

	return ir.UseClosure(clo, target)
}

// type check function definition
// To be called by typecheck, not directly.
// (Call typecheck.Func instead.)
func tcFunc(n *ir.Func) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("tcFunc", n)(nil)
	}

	if name := n.Nname; name.Typecheck() == 0 {
		if name.Ntype != nil {
			name.Ntype = typecheckNtype(name.Ntype)
			name.SetType(name.Ntype.Type())
		}
		name.SetTypecheck(1)
	}
}

// tcCall typechecks an OCALL node.
func tcCall(n *ir.CallExpr, top int) ir.Node {
	Stmts(n.Init()) // imported rewritten f(g()) calls (#30907)
	n.X = typecheck(n.X, ctxExpr|ctxType|ctxCallee)

	l := n.X

	if l.Op() == ir.ONAME && l.(*ir.Name).BuiltinOp != 0 {
		l := l.(*ir.Name)
		if n.IsDDD && l.BuiltinOp != ir.OAPPEND {
			base.Errorf("invalid use of ... with builtin %v", l)
		}

		// builtin: OLEN, OCAP, etc.
		switch l.BuiltinOp {
		default:
			base.Fatalf("unknown builtin %v", l)

		case ir.OAPPEND, ir.ODELETE, ir.OMAKE, ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
			n.SetOp(l.BuiltinOp)
			n.X = nil
			n.SetTypecheck(0) // re-typechecking new op is OK, not a loop
			return typecheck(n, top)

		case ir.OCAP, ir.OCLOSE, ir.OIMAG, ir.OLEN, ir.OPANIC, ir.OREAL:
			typecheckargs(n)
			fallthrough
		case ir.ONEW, ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
			arg, ok := needOneArg(n, "%v", n.Op())
			if !ok {
				n.SetType(nil)
				return n
			}
			u := ir.NewUnaryExpr(n.Pos(), l.BuiltinOp, arg)
			return typecheck(ir.InitExpr(n.Init(), u), top) // typecheckargs can add to old.Init

		case ir.OCOMPLEX, ir.OCOPY, ir.OUNSAFEADD, ir.OUNSAFESLICE:
			typecheckargs(n)
			arg1, arg2, ok := needTwoArgs(n)
			if !ok {
				n.SetType(nil)
				return n
			}
			b := ir.NewBinaryExpr(n.Pos(), l.BuiltinOp, arg1, arg2)
			return typecheck(ir.InitExpr(n.Init(), b), top) // typecheckargs can add to old.Init
		}
		panic("unreachable")
	}

	n.X = DefaultLit(n.X, nil)
	l = n.X
	if l.Op() == ir.OTYPE {
		if n.IsDDD {
			base.Fatalf("invalid use of ... in type conversion to %v", l.Type())
		}

		// pick off before type-checking arguments
		arg, ok := needOneArg(n, "conversion to %v", l.Type())
		if !ok {
			n.SetType(nil)
			return n
		}

		n := ir.NewConvExpr(n.Pos(), ir.OCONV, nil, arg)
		n.SetType(l.Type())
		return tcConv(n)
	}

	typecheckargs(n)
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	types.CheckSize(t)

	switch l.Op() {
	case ir.ODOTINTER:
		n.SetOp(ir.OCALLINTER)

	case ir.ODOTMETH:
		l := l.(*ir.SelectorExpr)
		n.SetOp(ir.OCALLMETH)

		// typecheckaste was used here but there wasn't enough
		// information further down the call chain to know if we
		// were testing a method receiver for unexported fields.
		// It isn't necessary, so just do a sanity check.
		tp := t.Recv().Type

		if l.X == nil || !types.Identical(l.X.Type(), tp) {
			base.Fatalf("method receiver")
		}

	default:
		n.SetOp(ir.OCALLFUNC)
		if t.Kind() != types.TFUNC {
			if o := ir.Orig(l); o.Name() != nil && types.BuiltinPkg.Lookup(o.Sym().Name).Def != nil {
				// be more specific when the non-function
				// name matches a predeclared function
				base.Errorf("cannot call non-function %L, declared at %s",
					l, base.FmtPos(o.Name().Pos()))
			} else {
				base.Errorf("cannot call non-function %L", l)
			}
			n.SetType(nil)
			return n
		}
	}

	typecheckaste(ir.OCALL, n.X, n.IsDDD, t.Params(), n.Args, func() string { return fmt.Sprintf("argument to %v", n.X) })
	FixMethodCall(n)
	if t.NumResults() == 0 {
		return n
	}
	if t.NumResults() == 1 {
		n.SetType(l.Type().Results().Field(0).Type)

		if n.Op() == ir.OCALLFUNC && n.X.Op() == ir.ONAME {
			if sym := n.X.(*ir.Name).Sym(); types.IsRuntimePkg(sym.Pkg) && sym.Name == "getg" {
				// Emit code for runtime.getg() directly instead of calling function.
				// Most such rewrites (for example the similar one for math.Sqrt) should be done in walk,
				// so that the ordering pass can make sure to preserve the semantics of the original code
				// (in particular, the exact time of the function call) by introducing temporaries.
				// In this case, we know getg() always returns the same result within a given function
				// and we want to avoid the temporaries, so we do the rewrite earlier than is typical.
				n.SetOp(ir.OGETG)
			}
		}
		return n
	}

	// multiple return
	if top&(ctxMultiOK|ctxStmt) == 0 {
		base.Errorf("multiple-value %v() in single-value context", l)
		return n
	}

	n.SetType(l.Type().Results())
	return n
}

// tcAppend typechecks an OAPPEND node.
func tcAppend(n *ir.CallExpr) ir.Node {
	typecheckargs(n)
	args := n.Args
	if len(args) == 0 {
		base.Errorf("missing arguments to append")
		n.SetType(nil)
		return n
	}

	t := args[0].Type()
	if t == nil {
		n.SetType(nil)
		return n
	}

	n.SetType(t)
	if !t.IsSlice() {
		if ir.IsNil(args[0]) {
			base.Errorf("first argument to append must be typed slice; have untyped nil")
			n.SetType(nil)
			return n
		}

		base.Errorf("first argument to append must be slice; have %L", t)
		n.SetType(nil)
		return n
	}

	if n.IsDDD {
		if len(args) == 1 {
			base.Errorf("cannot use ... on first argument to append")
			n.SetType(nil)
			return n
		}

		if len(args) != 2 {
			base.Errorf("too many arguments to append")
			n.SetType(nil)
			return n
		}

		if t.Elem().IsKind(types.TUINT8) && args[1].Type().IsString() {
			args[1] = DefaultLit(args[1], types.Types[types.TSTRING])
			return n
		}

		args[1] = AssignConv(args[1], t.Underlying(), "append")
		return n
	}

	as := args[1:]
	for i, n := range as {
		if n.Type() == nil {
			continue
		}
		as[i] = AssignConv(n, t.Elem(), "append")
		types.CheckSize(as[i].Type()) // ensure width is calculated for backend
	}
	return n
}

// tcClose typechecks an OCLOSE node.
func tcClose(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	n.X = DefaultLit(n.X, nil)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	if !t.IsChan() {
		base.Errorf("invalid operation: %v (non-chan type %v)", n, t)
		n.SetType(nil)
		return n
	}

	if !t.ChanDir().CanSend() {
		base.Errorf("invalid operation: %v (cannot close receive-only channel)", n)
		n.SetType(nil)
		return n
	}
	return n
}

// tcComplex typechecks an OCOMPLEX node.
func tcComplex(n *ir.BinaryExpr) ir.Node {
	l := Expr(n.X)
	r := Expr(n.Y)
	if l.Type() == nil || r.Type() == nil {
		n.SetType(nil)
		return n
	}
	l, r = defaultlit2(l, r, false)
	if l.Type() == nil || r.Type() == nil {
		n.SetType(nil)
		return n
	}
	n.X = l
	n.Y = r

	if !types.Identical(l.Type(), r.Type()) {
		base.Errorf("invalid operation: %v (mismatched types %v and %v)", n, l.Type(), r.Type())
		n.SetType(nil)
		return n
	}

	var t *types.Type
	switch l.Type().Kind() {
	default:
		base.Errorf("invalid operation: %v (arguments have type %v, expected floating-point)", n, l.Type())
		n.SetType(nil)
		return n

	case types.TIDEAL:
		t = types.UntypedComplex

	case types.TFLOAT32:
		t = types.Types[types.TCOMPLEX64]

	case types.TFLOAT64:
		t = types.Types[types.TCOMPLEX128]
	}
	n.SetType(t)
	return n
}

// tcCopy typechecks an OCOPY node.
func tcCopy(n *ir.BinaryExpr) ir.Node {
	n.SetType(types.Types[types.TINT])
	n.X = Expr(n.X)
	n.X = DefaultLit(n.X, nil)
	n.Y = Expr(n.Y)
	n.Y = DefaultLit(n.Y, nil)
	if n.X.Type() == nil || n.Y.Type() == nil {
		n.SetType(nil)
		return n
	}

	// copy([]byte, string)
	if n.X.Type().IsSlice() && n.Y.Type().IsString() {
		if types.Identical(n.X.Type().Elem(), types.ByteType) {
			return n
		}
		base.Errorf("arguments to copy have different element types: %L and string", n.X.Type())
		n.SetType(nil)
		return n
	}

	if !n.X.Type().IsSlice() || !n.Y.Type().IsSlice() {
		if !n.X.Type().IsSlice() && !n.Y.Type().IsSlice() {
			base.Errorf("arguments to copy must be slices; have %L, %L", n.X.Type(), n.Y.Type())
		} else if !n.X.Type().IsSlice() {
			base.Errorf("first argument to copy should be slice; have %L", n.X.Type())
		} else {
			base.Errorf("second argument to copy should be slice or string; have %L", n.Y.Type())
		}
		n.SetType(nil)
		return n
	}

	if !types.Identical(n.X.Type().Elem(), n.Y.Type().Elem()) {
		base.Errorf("arguments to copy have different element types: %L and %L", n.X.Type(), n.Y.Type())
		n.SetType(nil)
		return n
	}
	return n
}

// tcDelete typechecks an ODELETE node.
func tcDelete(n *ir.CallExpr) ir.Node {
	typecheckargs(n)
	args := n.Args
	if len(args) == 0 {
		base.Errorf("missing arguments to delete")
		n.SetType(nil)
		return n
	}

	if len(args) == 1 {
		base.Errorf("missing second (key) argument to delete")
		n.SetType(nil)
		return n
	}

	if len(args) != 2 {
		base.Errorf("too many arguments to delete")
		n.SetType(nil)
		return n
	}

	l := args[0]
	r := args[1]
	if l.Type() != nil && !l.Type().IsMap() {
		base.Errorf("first argument to delete must be map; have %L", l.Type())
		n.SetType(nil)
		return n
	}

	args[1] = AssignConv(r, l.Type().Key(), "delete")
	return n
}

// tcMake typechecks an OMAKE node.
func tcMake(n *ir.CallExpr) ir.Node {
	args := n.Args
	if len(args) == 0 {
		base.Errorf("missing argument to make")
		n.SetType(nil)
		return n
	}

	n.Args = nil
	l := args[0]
	l = typecheck(l, ctxType)
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}

	i := 1
	var nn ir.Node
	switch t.Kind() {
	default:
		base.Errorf("cannot make type %v", t)
		n.SetType(nil)
		return n

	case types.TSLICE:
		if i >= len(args) {
			base.Errorf("missing len argument to make(%v)", t)
			n.SetType(nil)
			return n
		}

		l = args[i]
		i++
		l = Expr(l)
		var r ir.Node
		if i < len(args) {
			r = args[i]
			i++
			r = Expr(r)
		}

		if l.Type() == nil || (r != nil && r.Type() == nil) {
			n.SetType(nil)
			return n
		}
		if !checkmake(t, "len", &l) || r != nil && !checkmake(t, "cap", &r) {
			n.SetType(nil)
			return n
		}
		if ir.IsConst(l, constant.Int) && r != nil && ir.IsConst(r, constant.Int) && constant.Compare(l.Val(), token.GTR, r.Val()) {
			base.Errorf("len larger than cap in make(%v)", t)
			n.SetType(nil)
			return n
		}
		nn = ir.NewMakeExpr(n.Pos(), ir.OMAKESLICE, l, r)

	case types.TMAP:
		if i < len(args) {
			l = args[i]
			i++
			l = Expr(l)
			l = DefaultLit(l, types.Types[types.TINT])
			if l.Type() == nil {
				n.SetType(nil)
				return n
			}
			if !checkmake(t, "size", &l) {
				n.SetType(nil)
				return n
			}
		} else {
			l = ir.NewInt(0)
		}
		nn = ir.NewMakeExpr(n.Pos(), ir.OMAKEMAP, l, nil)
		nn.SetEsc(n.Esc())

	case types.TCHAN:
		l = nil
		if i < len(args) {
			l = args[i]
			i++
			l = Expr(l)
			l = DefaultLit(l, types.Types[types.TINT])
			if l.Type() == nil {
				n.SetType(nil)
				return n
			}
			if !checkmake(t, "buffer", &l) {
				n.SetType(nil)
				return n
			}
		} else {
			l = ir.NewInt(0)
		}
		nn = ir.NewMakeExpr(n.Pos(), ir.OMAKECHAN, l, nil)
	}

	if i < len(args) {
		base.Errorf("too many arguments to make(%v)", t)
		n.SetType(nil)
		return n
	}

	nn.SetType(t)
	return nn
}

// tcMakeSliceCopy typechecks an OMAKESLICECOPY node.
func tcMakeSliceCopy(n *ir.MakeExpr) ir.Node {
	// Errors here are Fatalf instead of Errorf because only the compiler
	// can construct an OMAKESLICECOPY node.
	// Components used in OMAKESCLICECOPY that are supplied by parsed source code
	// have already been typechecked in OMAKE and OCOPY earlier.
	t := n.Type()

	if t == nil {
		base.Fatalf("no type specified for OMAKESLICECOPY")
	}

	if !t.IsSlice() {
		base.Fatalf("invalid type %v for OMAKESLICECOPY", n.Type())
	}

	if n.Len == nil {
		base.Fatalf("missing len argument for OMAKESLICECOPY")
	}

	if n.Cap == nil {
		base.Fatalf("missing slice argument to copy for OMAKESLICECOPY")
	}

	n.Len = Expr(n.Len)
	n.Cap = Expr(n.Cap)

	n.Len = DefaultLit(n.Len, types.Types[types.TINT])

	if !n.Len.Type().IsInteger() && n.Type().Kind() != types.TIDEAL {
		base.Errorf("non-integer len argument in OMAKESLICECOPY")
	}

	if ir.IsConst(n.Len, constant.Int) {
		if ir.ConstOverflow(n.Len.Val(), types.Types[types.TINT]) {
			base.Fatalf("len for OMAKESLICECOPY too large")
		}
		if constant.Sign(n.Len.Val()) < 0 {
			base.Fatalf("len for OMAKESLICECOPY must be non-negative")
		}
	}
	return n
}

// tcNew typechecks an ONEW node.
func tcNew(n *ir.UnaryExpr) ir.Node {
	if n.X == nil {
		// Fatalf because the OCALL above checked for us,
		// so this must be an internally-generated mistake.
		base.Fatalf("missing argument to new")
	}
	l := n.X
	l = typecheck(l, ctxType)
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	n.X = l
	n.SetType(types.NewPtr(t))
	return n
}

// tcPanic typechecks an OPANIC node.
func tcPanic(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	n.X = AssignConv(n.X, types.Types[types.TINTER], "argument to panic")
	if n.X.Type() == nil {
		n.SetType(nil)
		return n
	}
	return n
}

// tcPrint typechecks an OPRINT or OPRINTN node.
func tcPrint(n *ir.CallExpr) ir.Node {
	typecheckargs(n)
	ls := n.Args
	for i1, n1 := range ls {
		// Special case for print: int constant is int64, not int.
		if ir.IsConst(n1, constant.Int) {
			ls[i1] = DefaultLit(ls[i1], types.Types[types.TINT64])
		} else {
			ls[i1] = DefaultLit(ls[i1], nil)
		}
	}
	return n
}

// tcRealImag typechecks an OREAL or OIMAG node.
func tcRealImag(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}

	// Determine result type.
	switch t.Kind() {
	case types.TIDEAL:
		n.SetType(types.UntypedFloat)
	case types.TCOMPLEX64:
		n.SetType(types.Types[types.TFLOAT32])
	case types.TCOMPLEX128:
		n.SetType(types.Types[types.TFLOAT64])
	default:
		base.Errorf("invalid argument %L for %v", l, n.Op())
		n.SetType(nil)
		return n
	}
	return n
}

// tcRecover typechecks an ORECOVER node.
func tcRecover(n *ir.CallExpr) ir.Node {
	if len(n.Args) != 0 {
		base.Errorf("too many arguments to recover")
		n.SetType(nil)
		return n
	}

	n.SetType(types.Types[types.TINTER])
	return n
}

// tcRecoverFP typechecks an ORECOVERFP node.
func tcRecoverFP(n *ir.CallExpr) ir.Node {
	if len(n.Args) != 1 {
		base.FatalfAt(n.Pos(), "wrong number of arguments: %v", n)
	}

	n.Args[0] = Expr(n.Args[0])
	if !n.Args[0].Type().IsPtrShaped() {
		base.FatalfAt(n.Pos(), "%L is not pointer shaped", n.Args[0])
	}

	n.SetType(types.Types[types.TINTER])
	return n
}

// tcUnsafeAdd typechecks an OUNSAFEADD node.
func tcUnsafeAdd(n *ir.BinaryExpr) *ir.BinaryExpr {
	n.X = AssignConv(Expr(n.X), types.Types[types.TUNSAFEPTR], "argument to unsafe.Add")
	n.Y = DefaultLit(Expr(n.Y), types.Types[types.TINT])
	if n.X.Type() == nil || n.Y.Type() == nil {
		n.SetType(nil)
		return n
	}
	if !n.Y.Type().IsInteger() {
		n.SetType(nil)
		return n
	}
	n.SetType(n.X.Type())
	return n
}

// tcUnsafeSlice typechecks an OUNSAFESLICE node.
func tcUnsafeSlice(n *ir.BinaryExpr) *ir.BinaryExpr {
	n.X = Expr(n.X)
	n.Y = Expr(n.Y)
	if n.X.Type() == nil || n.Y.Type() == nil {
		n.SetType(nil)
		return n
	}
	t := n.X.Type()
	if !t.IsPtr() {
		base.Errorf("first argument to unsafe.Slice must be pointer; have %L", t)
	} else if t.Elem().NotInHeap() {
		// TODO(mdempsky): This can be relaxed, but should only affect the
		// Go runtime itself. End users should only see //go:notinheap
		// types due to incomplete C structs in cgo, and those types don't
		// have a meaningful size anyway.
		base.Errorf("unsafe.Slice of incomplete (or unallocatable) type not allowed")
	}

	if !checkunsafeslice(&n.Y) {
		n.SetType(nil)
		return n
	}
	n.SetType(types.NewSlice(t.Elem()))
	return n
}
