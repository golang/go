// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// largeStack is info about a function whose stack frame is too large (rare).
type largeStack struct {
	locals int64
	args   int64
	callee int64
	pos    src.XPos
}

var (
	largeStackFramesMu sync.Mutex // protects largeStackFrames
	largeStackFrames   []largeStack
)

// dotImports tracks all PkgNames that have been dot-imported.
var dotImports []*ir.PkgName

// find all the exported symbols in package referenced by PkgName,
// and make them available in the current package
func importDot(pack *ir.PkgName) {
	if typecheck.DotImportRefs == nil {
		typecheck.DotImportRefs = make(map[*ir.Ident]*ir.PkgName)
	}

	opkg := pack.Pkg
	for _, s := range opkg.Syms {
		if s.Def == nil {
			if _, ok := typecheck.DeclImporter[s]; !ok {
				continue
			}
		}
		if !types.IsExported(s.Name) || strings.ContainsRune(s.Name, 0xb7) { // 0xb7 = center dot
			continue
		}
		s1 := typecheck.Lookup(s.Name)
		if s1.Def != nil {
			pkgerror := fmt.Sprintf("during import %q", opkg.Path)
			typecheck.Redeclared(base.Pos, s1, pkgerror)
			continue
		}

		id := ir.NewIdent(src.NoXPos, s)
		typecheck.DotImportRefs[id] = pack
		s1.Def = id
		s1.Block = 1
	}

	dotImports = append(dotImports, pack)
}

// checkDotImports reports errors for any unused dot imports.
func checkDotImports() {
	for _, pack := range dotImports {
		if !pack.Used {
			base.ErrorfAt(pack.Pos(), "imported and not used: %q", pack.Pkg.Path)
		}
	}

	// No longer needed; release memory.
	dotImports = nil
	typecheck.DotImportRefs = nil
}

// backingArrayPtrLen extracts the pointer and length from a slice or string.
// This constructs two nodes referring to n, so n must be a cheapexpr.
func backingArrayPtrLen(n ir.Node) (ptr, length ir.Node) {
	var init ir.Nodes
	c := cheapexpr(n, &init)
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
		if thearch.SoftFloat && (types.IsFloat[n.Type().Kind()] || types.IsComplex[n.Type().Kind()]) {
			return true
		}
		return n.X.HasCall() || n.Y.HasCall()
	case ir.ONEG:
		n := n.(*ir.UnaryExpr)
		if thearch.SoftFloat && (types.IsFloat[n.Type().Kind()] || types.IsComplex[n.Type().Kind()]) {
			return true
		}
		return n.X.HasCall()
	case ir.OLT, ir.OEQ, ir.ONE, ir.OLE, ir.OGE, ir.OGT:
		n := n.(*ir.BinaryExpr)
		if thearch.SoftFloat && (types.IsFloat[n.X.Type().Kind()] || types.IsComplex[n.X.Type().Kind()]) {
			return true
		}
		return n.X.HasCall() || n.Y.HasCall()
	case ir.OCONV:
		n := n.(*ir.ConvExpr)
		if thearch.SoftFloat && ((types.IsFloat[n.Type().Kind()] || types.IsComplex[n.Type().Kind()]) || (types.IsFloat[n.X.Type().Kind()] || types.IsComplex[n.X.Type().Kind()])) {
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

	case ir.OGETG, ir.OCLOSUREREAD, ir.OMETHEXPR:
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

func badtype(op ir.Op, tl, tr *types.Type) {
	var s string
	if tl != nil {
		s += fmt.Sprintf("\n\t%v", tl)
	}
	if tr != nil {
		s += fmt.Sprintf("\n\t%v", tr)
	}

	// common mistake: *struct and *interface.
	if tl != nil && tr != nil && tl.IsPtr() && tr.IsPtr() {
		if tl.Elem().IsStruct() && tr.Elem().IsInterface() {
			s += "\n\t(*struct vs *interface)"
		} else if tl.Elem().IsInterface() && tr.Elem().IsStruct() {
			s += "\n\t(*interface vs *struct)"
		}
	}

	base.Errorf("illegal types for operand: %v%s", op, s)
}

// brcom returns !(op).
// For example, brcom(==) is !=.
func brcom(op ir.Op) ir.Op {
	switch op {
	case ir.OEQ:
		return ir.ONE
	case ir.ONE:
		return ir.OEQ
	case ir.OLT:
		return ir.OGE
	case ir.OGT:
		return ir.OLE
	case ir.OLE:
		return ir.OGT
	case ir.OGE:
		return ir.OLT
	}
	base.Fatalf("brcom: no com for %v\n", op)
	return op
}

// brrev returns reverse(op).
// For example, Brrev(<) is >.
func brrev(op ir.Op) ir.Op {
	switch op {
	case ir.OEQ:
		return ir.OEQ
	case ir.ONE:
		return ir.ONE
	case ir.OLT:
		return ir.OGT
	case ir.OGT:
		return ir.OLT
	case ir.OLE:
		return ir.OGE
	case ir.OGE:
		return ir.OLE
	}
	base.Fatalf("brrev: no rev for %v\n", op)
	return op
}

// return side effect-free n, appending side effects to init.
// result is assignable if n is.
func safeexpr(n ir.Node, init *ir.Nodes) ir.Node {
	if n == nil {
		return nil
	}

	if len(n.Init()) != 0 {
		walkstmtlist(n.Init())
		init.Append(n.PtrInit().Take()...)
	}

	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL, ir.ONAMEOFFSET:
		return n

	case ir.OLEN, ir.OCAP:
		n := n.(*ir.UnaryExpr)
		l := safeexpr(n.X, init)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.UnaryExpr)
		a.X = l
		return walkexpr(typecheck.Expr(a), init)

	case ir.ODOT, ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		l := safeexpr(n.X, init)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.SelectorExpr)
		a.X = l
		return walkexpr(typecheck.Expr(a), init)

	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		l := safeexpr(n.X, init)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.StarExpr)
		a.X = l
		return walkexpr(typecheck.Expr(a), init)

	case ir.OINDEX, ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		l := safeexpr(n.X, init)
		r := safeexpr(n.Index, init)
		if l == n.X && r == n.Index {
			return n
		}
		a := ir.Copy(n).(*ir.IndexExpr)
		a.X = l
		a.Index = r
		return walkexpr(typecheck.Expr(a), init)

	case ir.OSTRUCTLIT, ir.OARRAYLIT, ir.OSLICELIT:
		n := n.(*ir.CompLitExpr)
		if isStaticCompositeLiteral(n) {
			return n
		}
	}

	// make a copy; must not be used as an lvalue
	if ir.IsAssignable(n) {
		base.Fatalf("missing lvalue case in safeexpr: %v", n)
	}
	return cheapexpr(n, init)
}

func copyexpr(n ir.Node, t *types.Type, init *ir.Nodes) ir.Node {
	l := typecheck.Temp(t)
	appendWalkStmt(init, ir.NewAssignStmt(base.Pos, l, n))
	return l
}

// return side-effect free and cheap n, appending side effects to init.
// result may not be assignable.
func cheapexpr(n ir.Node, init *ir.Nodes) ir.Node {
	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n
	}

	return copyexpr(n, n.Type(), init)
}

// Generate a wrapper function to convert from
// a receiver of type T to a receiver of type U.
// That is,
//
//	func (t T) M() {
//		...
//	}
//
// already exists; this function generates
//
//	func (u U) M() {
//		u.M()
//	}
//
// where the types T and U are such that u.M() is valid
// and calls the T.M method.
// The resulting function is for use in method tables.
//
//	rcvr - U
//	method - M func (t T)(), a TFIELD type struct
//	newnam - the eventual mangled name of this function
func genwrapper(rcvr *types.Type, method *types.Field, newnam *types.Sym) {
	if false && base.Flag.LowerR != 0 {
		fmt.Printf("genwrapper rcvrtype=%v method=%v newnam=%v\n", rcvr, method, newnam)
	}

	// Only generate (*T).M wrappers for T.M in T's own package.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type &&
		rcvr.Elem().Sym() != nil && rcvr.Elem().Sym().Pkg != types.LocalPkg {
		return
	}

	// Only generate I.M wrappers for I in I's own package
	// but keep doing it for error.Error (was issue #29304).
	if rcvr.IsInterface() && rcvr.Sym() != nil && rcvr.Sym().Pkg != types.LocalPkg && rcvr != types.ErrorType {
		return
	}

	base.Pos = base.AutogeneratedPos
	typecheck.DeclContext = ir.PEXTERN

	tfn := ir.NewFuncType(base.Pos,
		ir.NewField(base.Pos, typecheck.Lookup(".this"), nil, rcvr),
		typecheck.NewFuncParams(method.Type.Params(), true),
		typecheck.NewFuncParams(method.Type.Results(), false))

	fn := typecheck.DeclFunc(newnam, tfn)
	fn.SetDupok(true)

	nthis := ir.AsNode(tfn.Type().Recv().Nname)

	methodrcvr := method.Type.Recv().Type

	// generate nil pointer check for better error
	if rcvr.IsPtr() && rcvr.Elem() == methodrcvr {
		// generating wrapper from *T to T.
		n := ir.NewIfStmt(base.Pos, nil, nil, nil)
		n.Cond = ir.NewBinaryExpr(base.Pos, ir.OEQ, nthis, typecheck.NodNil())
		call := ir.NewCallExpr(base.Pos, ir.OCALL, typecheck.LookupRuntime("panicwrap"), nil)
		n.Body = []ir.Node{call}
		fn.Body.Append(n)
	}

	dot := typecheck.AddImplicitDots(ir.NewSelectorExpr(base.Pos, ir.OXDOT, nthis, method.Sym))

	// generate call
	// It's not possible to use a tail call when dynamic linking on ppc64le. The
	// bad scenario is when a local call is made to the wrapper: the wrapper will
	// call the implementation, which might be in a different module and so set
	// the TOC to the appropriate value for that module. But if it returns
	// directly to the wrapper's caller, nothing will reset it to the correct
	// value for that function.
	if !base.Flag.Cfg.Instrumenting && rcvr.IsPtr() && methodrcvr.IsPtr() && method.Embedded != 0 && !types.IsInterfaceMethod(method.Type) && !(thearch.LinkArch.Name == "ppc64le" && base.Ctxt.Flag_dynlink) {
		// generate tail call: adjust pointer receiver and jump to embedded method.
		left := dot.X // skip final .M
		if !left.Type().IsPtr() {
			left = typecheck.NodAddr(left)
		}
		as := ir.NewAssignStmt(base.Pos, nthis, typecheck.ConvNop(left, rcvr))
		fn.Body.Append(as)
		fn.Body.Append(ir.NewBranchStmt(base.Pos, ir.ORETJMP, ir.MethodSym(methodrcvr, method.Sym)))
	} else {
		fn.SetWrapper(true) // ignore frame for panic+recover matching
		call := ir.NewCallExpr(base.Pos, ir.OCALL, dot, nil)
		call.Args.Set(ir.ParamNames(tfn.Type()))
		call.IsDDD = tfn.Type().IsVariadic()
		if method.Type.NumResults() > 0 {
			ret := ir.NewReturnStmt(base.Pos, nil)
			ret.Results = []ir.Node{call}
			fn.Body.Append(ret)
		} else {
			fn.Body.Append(call)
		}
	}

	if false && base.Flag.LowerR != 0 {
		ir.DumpList("genwrapper body", fn.Body)
	}

	typecheck.FinishFuncBody()
	if base.Debug.DclStack != 0 {
		types.CheckDclstack()
	}

	typecheck.Func(fn)
	ir.CurFunc = fn
	typecheck.Stmts(fn.Body)

	// Inline calls within (*T).M wrappers. This is safe because we only
	// generate those wrappers within the same compilation unit as (T).M.
	// TODO(mdempsky): Investigate why we can't enable this more generally.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type && rcvr.Elem().Sym() != nil {
		inline.InlineCalls(fn)
	}
	escape.Batch([]*ir.Func{fn}, false)

	ir.CurFunc = nil
	typecheck.Target.Decls = append(typecheck.Target.Decls, fn)
}

func hashmem(t *types.Type) ir.Node {
	sym := ir.Pkgs.Runtime.Lookup("memhash")

	n := typecheck.NewName(sym)
	ir.MarkFunc(n)
	n.SetType(typecheck.NewFuncType(nil, []*ir.Field{
		ir.NewField(base.Pos, nil, nil, types.NewPtr(t)),
		ir.NewField(base.Pos, nil, nil, types.Types[types.TUINTPTR]),
		ir.NewField(base.Pos, nil, nil, types.Types[types.TUINTPTR]),
	}, []*ir.Field{
		ir.NewField(base.Pos, nil, nil, types.Types[types.TUINTPTR]),
	}))
	return n
}

func ngotype(n ir.Node) *types.Sym {
	if n.Type() != nil {
		return typenamesym(n.Type())
	}
	return nil
}

// The linker uses the magic symbol prefixes "go." and "type."
// Avoid potential confusion between import paths and symbols
// by rejecting these reserved imports for now. Also, people
// "can do weird things in GOPATH and we'd prefer they didn't
// do _that_ weird thing" (per rsc). See also #4257.
var reservedimports = []string{
	"go",
	"type",
}

func isbadimport(path string, allowSpace bool) bool {
	if strings.Contains(path, "\x00") {
		base.Errorf("import path contains NUL")
		return true
	}

	for _, ri := range reservedimports {
		if path == ri {
			base.Errorf("import path %q is reserved and cannot be used", path)
			return true
		}
	}

	for _, r := range path {
		if r == utf8.RuneError {
			base.Errorf("import path contains invalid UTF-8 sequence: %q", path)
			return true
		}

		if r < 0x20 || r == 0x7f {
			base.Errorf("import path contains control character: %q", path)
			return true
		}

		if r == '\\' {
			base.Errorf("import path contains backslash; use slash: %q", path)
			return true
		}

		if !allowSpace && unicode.IsSpace(r) {
			base.Errorf("import path contains space character: %q", path)
			return true
		}

		if strings.ContainsRune("!\"#$%&'()*,:;<=>?[]^`{|}", r) {
			base.Errorf("import path contains invalid character '%c': %q", r, path)
			return true
		}
	}

	return false
}

// itabType loads the _type field from a runtime.itab struct.
func itabType(itab ir.Node) ir.Node {
	typ := ir.NewSelectorExpr(base.Pos, ir.ODOTPTR, itab, nil)
	typ.SetType(types.NewPtr(types.Types[types.TUINT8]))
	typ.SetTypecheck(1)
	typ.Offset = int64(types.PtrSize) // offset of _type in runtime.itab
	typ.SetBounded(true)              // guaranteed not to fault
	return typ
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
