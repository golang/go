// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"fmt"
	"internal/abi"
	"internal/buildcfg"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/rttype"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// The constant is known to runtime.
const tmpstringbufsize = 32

func Walk(fn *ir.Func) {
	ir.CurFunc = fn
	errorsBefore := base.Errors()
	order(fn)
	if base.Errors() > errorsBefore {
		return
	}

	if base.Flag.W != 0 {
		s := fmt.Sprintf("\nbefore walk %v", ir.CurFunc.Sym())
		ir.DumpList(s, ir.CurFunc.Body)
	}

	walkStmtList(ir.CurFunc.Body)
	if base.Flag.W != 0 {
		s := fmt.Sprintf("after walk %v", ir.CurFunc.Sym())
		ir.DumpList(s, ir.CurFunc.Body)
	}

	// Eagerly compute sizes of all variables for SSA.
	for _, n := range fn.Dcl {
		types.CalcSize(n.Type())
	}
}

// walkRecv walks an ORECV node.
func walkRecv(n *ir.UnaryExpr) ir.Node {
	if n.Typecheck() == 0 {
		base.Fatalf("missing typecheck: %+v", n)
	}
	init := ir.TakeInit(n)

	n.X = walkExpr(n.X, &init)
	call := walkExpr(mkcall1(chanfn("chanrecv1", 2, n.X.Type()), nil, &init, n.X, typecheck.NodNil()), &init)
	return ir.InitExpr(init, call)
}

func convas(n *ir.AssignStmt, init *ir.Nodes) *ir.AssignStmt {
	if n.Op() != ir.OAS {
		base.Fatalf("convas: not OAS %v", n.Op())
	}
	n.SetTypecheck(1)

	if n.X == nil || n.Y == nil {
		return n
	}

	lt := n.X.Type()
	rt := n.Y.Type()
	if lt == nil || rt == nil {
		return n
	}

	if ir.IsBlank(n.X) {
		n.Y = typecheck.DefaultLit(n.Y, nil)
		return n
	}

	if !types.Identical(lt, rt) {
		n.Y = typecheck.AssignConv(n.Y, lt, "assignment")
		n.Y = walkExpr(n.Y, init)
	}
	types.CalcSize(n.Y.Type())

	return n
}

func vmkcall(fn ir.Node, t *types.Type, init *ir.Nodes, va []ir.Node) *ir.CallExpr {
	if init == nil {
		base.Fatalf("mkcall with nil init: %v", fn)
	}
	if fn.Type() == nil || fn.Type().Kind() != types.TFUNC {
		base.Fatalf("mkcall %v %v", fn, fn.Type())
	}

	n := fn.Type().NumParams()
	if n != len(va) {
		base.Fatalf("vmkcall %v needs %v args got %v", fn, n, len(va))
	}

	call := typecheck.Call(base.Pos, fn, va, false).(*ir.CallExpr)
	call.SetType(t)
	return walkExpr(call, init).(*ir.CallExpr)
}

func mkcall(name string, t *types.Type, init *ir.Nodes, args ...ir.Node) *ir.CallExpr {
	return vmkcall(typecheck.LookupRuntime(name), t, init, args)
}

func mkcallstmt(name string, args ...ir.Node) ir.Node {
	return mkcallstmt1(typecheck.LookupRuntime(name), args...)
}

func mkcall1(fn ir.Node, t *types.Type, init *ir.Nodes, args ...ir.Node) *ir.CallExpr {
	return vmkcall(fn, t, init, args)
}

func mkcallstmt1(fn ir.Node, args ...ir.Node) ir.Node {
	var init ir.Nodes
	n := vmkcall(fn, nil, &init, args)
	if len(init) == 0 {
		return n
	}
	init.Append(n)
	return ir.NewBlockStmt(n.Pos(), init)
}

func chanfn(name string, n int, t *types.Type) ir.Node {
	if !t.IsChan() {
		base.Fatalf("chanfn %v", t)
	}
	switch n {
	case 1:
		return typecheck.LookupRuntime(name, t.Elem())
	case 2:
		return typecheck.LookupRuntime(name, t.Elem(), t.Elem())
	}
	base.Fatalf("chanfn %d", n)
	return nil
}

func mapfn(name string, t *types.Type, isfat bool) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	if mapfast(t) == mapslow || isfat {
		return typecheck.LookupRuntime(name, t.Key(), t.Elem(), t.Key(), t.Elem())
	}
	return typecheck.LookupRuntime(name, t.Key(), t.Elem(), t.Elem())
}

func mapfndel(name string, t *types.Type) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	if mapfast(t) == mapslow {
		return typecheck.LookupRuntime(name, t.Key(), t.Elem(), t.Key())
	}
	return typecheck.LookupRuntime(name, t.Key(), t.Elem())
}

const (
	mapslow = iota
	mapfast32
	mapfast32ptr
	mapfast64
	mapfast64ptr
	mapfaststr
	nmapfast
)

type mapnames [nmapfast]string

func mkmapnames(base string, ptr string) mapnames {
	return mapnames{base, base + "_fast32", base + "_fast32" + ptr, base + "_fast64", base + "_fast64" + ptr, base + "_faststr"}
}

var mapaccess1 = mkmapnames("mapaccess1", "")
var mapaccess2 = mkmapnames("mapaccess2", "")
var mapassign = mkmapnames("mapassign", "ptr")
var mapdelete = mkmapnames("mapdelete", "")

func mapfast(t *types.Type) int {
	if buildcfg.Experiment.SwissMap {
		return mapfastSwiss(t)
	}
	return mapfastOld(t)
}

func mapfastSwiss(t *types.Type) int {
	if t.Elem().Size() > abi.SwissMapMaxElemBytes {
		return mapslow
	}
	switch reflectdata.AlgType(t.Key()) {
	case types.AMEM32:
		if !t.Key().HasPointers() {
			return mapfast32
		}
		if types.PtrSize == 4 {
			return mapfast32ptr
		}
		base.Fatalf("small pointer %v", t.Key())
	case types.AMEM64:
		if !t.Key().HasPointers() {
			return mapfast64
		}
		if types.PtrSize == 8 {
			return mapfast64ptr
		}
		// Two-word object, at least one of which is a pointer.
		// Use the slow path.
	case types.ASTRING:
		return mapfaststr
	}
	return mapslow
}

func mapfastOld(t *types.Type) int {
	if t.Elem().Size() > abi.OldMapMaxElemBytes {
		return mapslow
	}
	switch reflectdata.AlgType(t.Key()) {
	case types.AMEM32:
		if !t.Key().HasPointers() {
			return mapfast32
		}
		if types.PtrSize == 4 {
			return mapfast32ptr
		}
		base.Fatalf("small pointer %v", t.Key())
	case types.AMEM64:
		if !t.Key().HasPointers() {
			return mapfast64
		}
		if types.PtrSize == 8 {
			return mapfast64ptr
		}
		// Two-word object, at least one of which is a pointer.
		// Use the slow path.
	case types.ASTRING:
		return mapfaststr
	}
	return mapslow
}

func walkAppendArgs(n *ir.CallExpr, init *ir.Nodes) {
	walkExprListSafe(n.Args, init)

	// walkExprListSafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	ls := n.Args
	for i1, n1 := range ls {
		ls[i1] = cheapExpr(n1, init)
	}
}

// appendWalkStmt typechecks and walks stmt and then appends it to init.
func appendWalkStmt(init *ir.Nodes, stmt ir.Node) {
	op := stmt.Op()
	n := typecheck.Stmt(stmt)
	if op == ir.OAS || op == ir.OAS2 {
		// If the assignment has side effects, walkExpr will append them
		// directly to init for us, while walkStmt will wrap it in an OBLOCK.
		// We need to append them directly.
		// TODO(rsc): Clean this up.
		n = walkExpr(n, init)
	} else {
		n = walkStmt(n)
	}
	init.Append(n)
}

// The max number of defers in a function using open-coded defers. We enforce this
// limit because the deferBits bitmask is currently a single byte (to minimize code size)
const maxOpenDefers = 8

// backingArrayPtrLen extracts the pointer and length from a slice or string.
// This constructs two nodes referring to n, so n must be a cheapExpr.
func backingArrayPtrLen(n ir.Node) (ptr, length ir.Node) {
	var init ir.Nodes
	c := cheapExpr(n, &init)
	if c != n || len(init) != 0 {
		base.Fatalf("backingArrayPtrLen not cheap: %v", n)
	}
	ptr = ir.NewUnaryExpr(base.Pos, ir.OSPTR, n)
	if n.Type().IsString() {
		ptr.SetType(types.Types[types.TUINT8].PtrTo())
	} else {
		ptr.SetType(n.Type().Elem().PtrTo())
	}
	ptr.SetTypecheck(1)
	length = ir.NewUnaryExpr(base.Pos, ir.OLEN, n)
	length.SetType(types.Types[types.TINT])
	length.SetTypecheck(1)
	return ptr, length
}

// mayCall reports whether evaluating expression n may require
// function calls, which could clobber function call arguments/results
// currently on the stack.
func mayCall(n ir.Node) bool {
	// When instrumenting, any expression might require function calls.
	if base.Flag.Cfg.Instrumenting {
		return true
	}

	isSoftFloat := func(typ *types.Type) bool {
		return types.IsFloat[typ.Kind()] || types.IsComplex[typ.Kind()]
	}

	return ir.Any(n, func(n ir.Node) bool {
		// walk should have already moved any Init blocks off of
		// expressions.
		if len(n.Init()) != 0 {
			base.FatalfAt(n.Pos(), "mayCall %+v", n)
		}

		switch n.Op() {
		default:
			base.FatalfAt(n.Pos(), "mayCall %+v", n)

		case ir.OCALLFUNC, ir.OCALLINTER,
			ir.OUNSAFEADD, ir.OUNSAFESLICE:
			return true

		case ir.OINDEX, ir.OSLICE, ir.OSLICEARR, ir.OSLICE3, ir.OSLICE3ARR, ir.OSLICESTR,
			ir.ODEREF, ir.ODOTPTR, ir.ODOTTYPE, ir.ODYNAMICDOTTYPE, ir.ODIV, ir.OMOD,
			ir.OSLICE2ARR, ir.OSLICE2ARRPTR:
			// These ops might panic, make sure they are done
			// before we start marshaling args for a call. See issue 16760.
			return true

		case ir.OANDAND, ir.OOROR:
			n := n.(*ir.LogicalExpr)
			// The RHS expression may have init statements that
			// should only execute conditionally, and so cannot be
			// pulled out to the top-level init list. We could try
			// to be more precise here.
			return len(n.Y.Init()) != 0

		// When using soft-float, these ops might be rewritten to function calls
		// so we ensure they are evaluated first.
		case ir.OADD, ir.OSUB, ir.OMUL, ir.ONEG:
			return ssagen.Arch.SoftFloat && isSoftFloat(n.Type())
		case ir.OLT, ir.OEQ, ir.ONE, ir.OLE, ir.OGE, ir.OGT:
			n := n.(*ir.BinaryExpr)
			return ssagen.Arch.SoftFloat && isSoftFloat(n.X.Type())
		case ir.OCONV:
			n := n.(*ir.ConvExpr)
			return ssagen.Arch.SoftFloat && (isSoftFloat(n.Type()) || isSoftFloat(n.X.Type()))

		case ir.OMIN, ir.OMAX:
			// string or float requires runtime call, see (*ssagen.state).minmax method.
			return n.Type().IsString() || n.Type().IsFloat()

		case ir.OLITERAL, ir.ONIL, ir.ONAME, ir.OLINKSYMOFFSET, ir.OMETHEXPR,
			ir.OAND, ir.OANDNOT, ir.OLSH, ir.OOR, ir.ORSH, ir.OXOR, ir.OCOMPLEX, ir.OMAKEFACE,
			ir.OADDR, ir.OBITNOT, ir.ONOT, ir.OPLUS,
			ir.OCAP, ir.OIMAG, ir.OLEN, ir.OREAL,
			ir.OCONVNOP, ir.ODOT,
			ir.OCFUNC, ir.OIDATA, ir.OITAB, ir.OSPTR,
			ir.OBYTES2STRTMP, ir.OGETG, ir.OGETCALLERPC, ir.OGETCALLERSP, ir.OSLICEHEADER, ir.OSTRINGHEADER:
			// ok: operations that don't require function calls.
			// Expand as needed.
		}

		return false
	})
}

// itabType loads the _type field from a runtime.itab struct.
func itabType(itab ir.Node) ir.Node {
	if itabTypeField == nil {
		// internal/abi.ITab's Type field
		itabTypeField = runtimeField("Type", rttype.ITab.OffsetOf("Type"), types.NewPtr(types.Types[types.TUINT8]))
	}
	return boundedDotPtr(base.Pos, itab, itabTypeField)
}

var itabTypeField *types.Field

// boundedDotPtr returns a selector expression representing ptr.field
// and omits nil-pointer checks for ptr.
func boundedDotPtr(pos src.XPos, ptr ir.Node, field *types.Field) *ir.SelectorExpr {
	sel := ir.NewSelectorExpr(pos, ir.ODOTPTR, ptr, field.Sym)
	sel.Selection = field
	sel.SetType(field.Type)
	sel.SetTypecheck(1)
	sel.SetBounded(true) // guaranteed not to fault
	return sel
}

func runtimeField(name string, offset int64, typ *types.Type) *types.Field {
	f := types.NewField(src.NoXPos, ir.Pkgs.Runtime.Lookup(name), typ)
	f.Offset = offset
	return f
}

// ifaceData loads the data field from an interface.
// The concrete type must be known to have type t.
// It follows the pointer if !IsDirectIface(t).
func ifaceData(pos src.XPos, n ir.Node, t *types.Type) ir.Node {
	if t.IsInterface() {
		base.Fatalf("ifaceData interface: %v", t)
	}
	ptr := ir.NewUnaryExpr(pos, ir.OIDATA, n)
	if types.IsDirectIface(t) {
		ptr.SetType(t)
		ptr.SetTypecheck(1)
		return ptr
	}
	ptr.SetType(types.NewPtr(t))
	ptr.SetTypecheck(1)
	ind := ir.NewStarExpr(pos, ptr)
	ind.SetType(t)
	ind.SetTypecheck(1)
	ind.SetBounded(true)
	return ind
}
