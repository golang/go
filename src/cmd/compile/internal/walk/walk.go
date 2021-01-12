// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"errors"
	"fmt"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// The constant is known to runtime.
const tmpstringbufsize = 32
const zeroValSize = 1024 // must match value of runtime/map.go:maxZero

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

	lno := base.Pos

	base.Pos = lno
	if base.Errors() > errorsBefore {
		return
	}
	walkStmtList(ir.CurFunc.Body)
	if base.Flag.W != 0 {
		s := fmt.Sprintf("after walk %v", ir.CurFunc.Sym())
		ir.DumpList(s, ir.CurFunc.Body)
	}

	zeroResults()
	heapmoves()
	if base.Flag.W != 0 && len(ir.CurFunc.Enter) > 0 {
		s := fmt.Sprintf("enter %v", ir.CurFunc.Sym())
		ir.DumpList(s, ir.CurFunc.Enter)
	}

	if base.Flag.Cfg.Instrumenting {
		instrument(fn)
	}

	// Eagerly compute sizes of all variables for SSA.
	for _, n := range fn.Dcl {
		types.CalcSize(n.Type())
	}
}

func paramoutheap(fn *ir.Func) bool {
	for _, ln := range fn.Dcl {
		switch ln.Class {
		case ir.PPARAMOUT:
			if ir.IsParamStackCopy(ln) || ln.Addrtaken() {
				return true
			}

		case ir.PAUTO:
			// stop early - parameters are over
			return false
		}
	}

	return false
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
	defer updateHasCall(n)

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

var stop = errors.New("stop")

// paramstoheap returns code to allocate memory for heap-escaped parameters
// and to copy non-result parameters' values from the stack.
func paramstoheap(params *types.Type) []ir.Node {
	var nn []ir.Node
	for _, t := range params.Fields().Slice() {
		v := ir.AsNode(t.Nname)
		if v != nil && v.Sym() != nil && strings.HasPrefix(v.Sym().Name, "~r") { // unnamed result
			v = nil
		}
		if v == nil {
			continue
		}

		if stackcopy := v.Name().Stackcopy; stackcopy != nil {
			nn = append(nn, walkStmt(ir.NewDecl(base.Pos, ir.ODCL, v.(*ir.Name))))
			if stackcopy.Class == ir.PPARAM {
				nn = append(nn, walkStmt(typecheck.Stmt(ir.NewAssignStmt(base.Pos, v, stackcopy))))
			}
		}
	}

	return nn
}

// zeroResults zeros the return values at the start of the function.
// We need to do this very early in the function.  Defer might stop a
// panic and show the return values as they exist at the time of
// panic.  For precise stacks, the garbage collector assumes results
// are always live, so we need to zero them before any allocations,
// even allocations to move params/results to the heap.
// The generated code is added to Curfn's Enter list.
func zeroResults() {
	for _, f := range ir.CurFunc.Type().Results().Fields().Slice() {
		v := ir.AsNode(f.Nname)
		if v != nil && v.Name().Heapaddr != nil {
			// The local which points to the return value is the
			// thing that needs zeroing. This is already handled
			// by a Needzero annotation in plive.go:livenessepilogue.
			continue
		}
		if ir.IsParamHeapCopy(v) {
			// TODO(josharian/khr): Investigate whether we can switch to "continue" here,
			// and document more in either case.
			// In the review of CL 114797, Keith wrote (roughly):
			// I don't think the zeroing below matters.
			// The stack return value will never be marked as live anywhere in the function.
			// It is not written to until deferreturn returns.
			v = v.Name().Stackcopy
		}
		// Zero the stack location containing f.
		ir.CurFunc.Enter.Append(ir.NewAssignStmt(ir.CurFunc.Pos(), v, nil))
	}
}

// returnsfromheap returns code to copy values for heap-escaped parameters
// back to the stack.
func returnsfromheap(params *types.Type) []ir.Node {
	var nn []ir.Node
	for _, t := range params.Fields().Slice() {
		v := ir.AsNode(t.Nname)
		if v == nil {
			continue
		}
		if stackcopy := v.Name().Stackcopy; stackcopy != nil && stackcopy.Class == ir.PPARAMOUT {
			nn = append(nn, walkStmt(typecheck.Stmt(ir.NewAssignStmt(base.Pos, stackcopy, v))))
		}
	}

	return nn
}

// heapmoves generates code to handle migrating heap-escaped parameters
// between the stack and the heap. The generated code is added to Curfn's
// Enter and Exit lists.
func heapmoves() {
	lno := base.Pos
	base.Pos = ir.CurFunc.Pos()
	nn := paramstoheap(ir.CurFunc.Type().Recvs())
	nn = append(nn, paramstoheap(ir.CurFunc.Type().Params())...)
	nn = append(nn, paramstoheap(ir.CurFunc.Type().Results())...)
	ir.CurFunc.Enter.Append(nn...)
	base.Pos = ir.CurFunc.Endlineno
	ir.CurFunc.Exit.Append(returnsfromheap(ir.CurFunc.Type().Results())...)
	base.Pos = lno
}

func vmkcall(fn ir.Node, t *types.Type, init *ir.Nodes, va []ir.Node) *ir.CallExpr {
	if fn.Type() == nil || fn.Type().Kind() != types.TFUNC {
		base.Fatalf("mkcall %v %v", fn, fn.Type())
	}

	n := fn.Type().NumParams()
	if n != len(va) {
		base.Fatalf("vmkcall %v needs %v args got %v", fn, n, len(va))
	}

	call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, va)
	typecheck.Call(call)
	call.SetType(t)
	return walkExpr(call, init).(*ir.CallExpr)
}

func mkcall(name string, t *types.Type, init *ir.Nodes, args ...ir.Node) *ir.CallExpr {
	return vmkcall(typecheck.LookupRuntime(name), t, init, args)
}

func mkcall1(fn ir.Node, t *types.Type, init *ir.Nodes, args ...ir.Node) *ir.CallExpr {
	return vmkcall(fn, t, init, args)
}

func chanfn(name string, n int, t *types.Type) ir.Node {
	if !t.IsChan() {
		base.Fatalf("chanfn %v", t)
	}
	fn := typecheck.LookupRuntime(name)
	switch n {
	default:
		base.Fatalf("chanfn %d", n)
	case 1:
		fn = typecheck.SubstArgTypes(fn, t.Elem())
	case 2:
		fn = typecheck.SubstArgTypes(fn, t.Elem(), t.Elem())
	}
	return fn
}

func mapfn(name string, t *types.Type) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	fn := typecheck.LookupRuntime(name)
	fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem(), t.Key(), t.Elem())
	return fn
}

func mapfndel(name string, t *types.Type) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	fn := typecheck.LookupRuntime(name)
	fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem(), t.Key())
	return fn
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
	// Check runtime/map.go:maxElemSize before changing.
	if t.Elem().Width > 128 {
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

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	ls := n.Args
	for i1, n1 := range ls {
		ls[i1] = cheapExpr(n1, init)
	}
}

// Rewrite
//	go builtin(x, y, z)
// into
//	go func(a1, a2, a3) {
//		builtin(a1, a2, a3)
//	}(x, y, z)
// for print, println, and delete.
//
// Rewrite
//	go f(x, y, uintptr(unsafe.Pointer(z)))
// into
//	go func(a1, a2, a3) {
//		builtin(a1, a2, uintptr(a3))
//	}(x, y, unsafe.Pointer(z))
// for function contains unsafe-uintptr arguments.

var wrapCall_prgen int

// appendWalkStmt typechecks and walks stmt and then appends it to init.
func appendWalkStmt(init *ir.Nodes, stmt ir.Node) {
	op := stmt.Op()
	n := typecheck.Stmt(stmt)
	if op == ir.OAS || op == ir.OAS2 {
		// If the assignment has side effects, walkexpr will append them
		// directly to init for us, while walkstmt will wrap it in an OBLOCK.
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
// This constructs two nodes referring to n, so n must be a cheapexpr.
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
	length = ir.NewUnaryExpr(base.Pos, ir.OLEN, n)
	length.SetType(types.Types[types.TINT])
	return ptr, length
}

// updateHasCall checks whether expression n contains any function
// calls and sets the n.HasCall flag if so.
func updateHasCall(n ir.Node) {
	if n == nil {
		return
	}
	n.SetHasCall(calcHasCall(n))
}

func calcHasCall(n ir.Node) bool {
	if len(n.Init()) != 0 {
		// TODO(mdempsky): This seems overly conservative.
		return true
	}

	switch n.Op() {
	default:
		base.Fatalf("calcHasCall %+v", n)
		panic("unreachable")

	case ir.OLITERAL, ir.ONIL, ir.ONAME, ir.OTYPE, ir.ONAMEOFFSET:
		if n.HasCall() {
			base.Fatalf("OLITERAL/ONAME/OTYPE should never have calls: %+v", n)
		}
		return false
	case ir.OCALL, ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		return true
	case ir.OANDAND, ir.OOROR:
		// hard with instrumented code
		n := n.(*ir.LogicalExpr)
		if base.Flag.Cfg.Instrumenting {
			return true
		}
		return n.X.HasCall() || n.Y.HasCall()
	case ir.OINDEX, ir.OSLICE, ir.OSLICEARR, ir.OSLICE3, ir.OSLICE3ARR, ir.OSLICESTR,
		ir.ODEREF, ir.ODOTPTR, ir.ODOTTYPE, ir.ODIV, ir.OMOD:
		// These ops might panic, make sure they are done
		// before we start marshaling args for a call. See issue 16760.
		return true

	// When using soft-float, these ops might be rewritten to function calls
	// so we ensure they are evaluated first.
	case ir.OADD, ir.OSUB, ir.OMUL:
		n := n.(*ir.BinaryExpr)
		if ssagen.Arch.SoftFloat && (types.IsFloat[n.Type().Kind()] || types.IsComplex[n.Type().Kind()]) {
			return true
		}
		return n.X.HasCall() || n.Y.HasCall()
	case ir.ONEG:
		n := n.(*ir.UnaryExpr)
		if ssagen.Arch.SoftFloat && (types.IsFloat[n.Type().Kind()] || types.IsComplex[n.Type().Kind()]) {
			return true
		}
		return n.X.HasCall()
	case ir.OLT, ir.OEQ, ir.ONE, ir.OLE, ir.OGE, ir.OGT:
		n := n.(*ir.BinaryExpr)
		if ssagen.Arch.SoftFloat && (types.IsFloat[n.X.Type().Kind()] || types.IsComplex[n.X.Type().Kind()]) {
			return true
		}
		return n.X.HasCall() || n.Y.HasCall()
	case ir.OCONV:
		n := n.(*ir.ConvExpr)
		if ssagen.Arch.SoftFloat && ((types.IsFloat[n.Type().Kind()] || types.IsComplex[n.Type().Kind()]) || (types.IsFloat[n.X.Type().Kind()] || types.IsComplex[n.X.Type().Kind()])) {
			return true
		}
		return n.X.HasCall()

	case ir.OAND, ir.OANDNOT, ir.OLSH, ir.OOR, ir.ORSH, ir.OXOR, ir.OCOPY, ir.OCOMPLEX, ir.OEFACE:
		n := n.(*ir.BinaryExpr)
		return n.X.HasCall() || n.Y.HasCall()

	case ir.OAS:
		n := n.(*ir.AssignStmt)
		return n.X.HasCall() || n.Y != nil && n.Y.HasCall()

	case ir.OADDR:
		n := n.(*ir.AddrExpr)
		return n.X.HasCall()
	case ir.OPAREN:
		n := n.(*ir.ParenExpr)
		return n.X.HasCall()
	case ir.OBITNOT, ir.ONOT, ir.OPLUS, ir.ORECV,
		ir.OALIGNOF, ir.OCAP, ir.OCLOSE, ir.OIMAG, ir.OLEN, ir.ONEW,
		ir.OOFFSETOF, ir.OPANIC, ir.OREAL, ir.OSIZEOF,
		ir.OCHECKNIL, ir.OCFUNC, ir.OIDATA, ir.OITAB, ir.ONEWOBJ, ir.OSPTR, ir.OVARDEF, ir.OVARKILL, ir.OVARLIVE:
		n := n.(*ir.UnaryExpr)
		return n.X.HasCall()
	case ir.ODOT, ir.ODOTMETH, ir.ODOTINTER:
		n := n.(*ir.SelectorExpr)
		return n.X.HasCall()

	case ir.OGETG, ir.OMETHEXPR:
		return false

	// TODO(rsc): These look wrong in various ways but are what calcHasCall has always done.
	case ir.OADDSTR:
		// TODO(rsc): This used to check left and right, which are not part of OADDSTR.
		return false
	case ir.OBLOCK:
		// TODO(rsc): Surely the block's statements matter.
		return false
	case ir.OCONVIFACE, ir.OCONVNOP, ir.OBYTES2STR, ir.OBYTES2STRTMP, ir.ORUNES2STR, ir.OSTR2BYTES, ir.OSTR2BYTESTMP, ir.OSTR2RUNES, ir.ORUNESTR:
		// TODO(rsc): Some conversions are themselves calls, no?
		n := n.(*ir.ConvExpr)
		return n.X.HasCall()
	case ir.ODOTTYPE2:
		// TODO(rsc): Shouldn't this be up with ODOTTYPE above?
		n := n.(*ir.TypeAssertExpr)
		return n.X.HasCall()
	case ir.OSLICEHEADER:
		// TODO(rsc): What about len and cap?
		n := n.(*ir.SliceHeaderExpr)
		return n.Ptr.HasCall()
	case ir.OAS2DOTTYPE, ir.OAS2FUNC:
		// TODO(rsc): Surely we need to check List and Rlist.
		return false
	}
}

// itabType loads the _type field from a runtime.itab struct.
func itabType(itab ir.Node) ir.Node {
	if itabTypeField == nil {
		// runtime.itab's _type field
		itabTypeField = runtimeField("_type", int64(types.PtrSize), types.NewPtr(types.Types[types.TUINT8]))
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
// It follows the pointer if !isdirectiface(t).
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
