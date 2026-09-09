// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"fmt"
	"go/constant"
	"internal/abi"
	"internal/buildcfg"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/noder"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/rttype"
	"cmd/compile/internal/staticdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
)

// The result of walkExpr MUST be assigned back to n, e.g.
//
//	n.Left = walkExpr(n.Left, init)
func walkExpr(walkstate *walkState, n ir.Node, init *ir.Nodes) ir.Node {
	if n == nil {
		return n
	}

	if n, ok := n.(ir.InitNode); ok && init == n.PtrInit() {
		// not okay to use n->ninit when walking n,
		// because we might replace n with some other node
		// and would lose the init list.
		base.Fatalf("walkExpr init == &n->ninit")
	}

	if len(n.Init()) != 0 {
		walkStmtList(walkstate, n.Init())
		init.Append(ir.TakeInit(n)...)
	}

	lno := ir.SetPos(n)

	if base.Flag.LowerW > 1 {
		ir.Dump("before walk expr", n)
	}

	if n.Typecheck() != 1 {
		base.Fatalf("missed typecheck: %+v", n)
	}

	if n.Type().IsUntyped() {
		base.Fatalf("expression has untyped type: %+v", n)
	}

	n = walkExpr1(walkstate, n, init)

	// Eagerly compute sizes of all expressions for the back end.
	if typ := n.Type(); typ != nil && typ.Kind() != types.TBLANK && !typ.IsFuncArgStruct() {
		types.CheckSize(typ)
	}
	if n, ok := n.(*ir.Name); ok && n.Heapaddr != nil {
		types.CheckSize(n.Heapaddr.Type())
	}
	if ir.IsConst(n, constant.String) {
		// Emit string symbol now to avoid emitting
		// any concurrently during the backend.
		_ = staticdata.StringSym(n.Pos(), constant.StringVal(n.Val()))
	}

	if base.Flag.LowerW != 0 && n != nil {
		ir.Dump("after walk expr", n)
	}

	base.Pos = lno
	return n
}

func walkExpr1(walkstate *walkState, n ir.Node, init *ir.Nodes) ir.Node {
	switch n.Op() {
	default:
		ir.Dump("walk", n)
		base.Fatalf("walkExpr: switch 1 unknown op %+v", n.Op())
		panic("unreachable")

	case ir.OGETG, ir.OGETCALLERSP:
		return n

	case ir.OTYPE, ir.ONAME, ir.OLITERAL, ir.ONIL, ir.OLINKSYMOFFSET:
		// TODO(mdempsky): Just return n; see discussion on CL 38655.
		// Perhaps refactor to use Node.mayBeShared for these instead.
		// If these return early, make sure to still call
		// StringSym for constant strings.
		return n

	case ir.OMETHEXPR:
		// TODO(mdempsky): Do this right after type checking.
		n := n.(*ir.SelectorExpr)
		return n.FuncName()

	case ir.OMIN, ir.OMAX:
		n := n.(*ir.CallExpr)
		return walkMinMax(walkstate, n, init)

	case ir.ONOT, ir.ONEG, ir.OPLUS, ir.OBITNOT, ir.OREAL, ir.OIMAG, ir.OSPTR, ir.OITAB, ir.OIDATA:
		n := n.(*ir.UnaryExpr)
		n.X = walkExpr(walkstate, n.X, init)
		return n

	case ir.ODOTMETH, ir.ODOTINTER:
		n := n.(*ir.SelectorExpr)
		n.X = walkExpr(walkstate, n.X, init)
		return n

	case ir.OADDR:
		n := n.(*ir.AddrExpr)
		n.X = walkExpr(walkstate, n.X, init)
		return n

	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		n.X = walkExpr(walkstate, n.X, init)
		return n

	case ir.OMAKEFACE, ir.OAND, ir.OANDNOT, ir.OSUB, ir.OMUL, ir.OADD, ir.OOR, ir.OXOR, ir.OLSH, ir.ORSH,
		ir.OUNSAFEADD:
		n := n.(*ir.BinaryExpr)
		n.X = walkExpr(walkstate, n.X, init)
		n.Y = walkExpr(walkstate, n.Y, init)
		if n.Op() == ir.OUNSAFEADD && ir.ShouldCheckPtr(walkstate.curfunc, 1) {
			// For unsafe.Add(p, n), just walk "unsafe.Pointer(uintptr(p)+uintptr(n))"
			// for the side effects of validating unsafe.Pointer rules.
			x := typecheck.ConvNop(n.X, types.Types[types.TUINTPTR])
			y := typecheck.Conv(n.Y, types.Types[types.TUINTPTR])
			conv := typecheck.ConvNop(ir.NewBinaryExpr(n.Pos(), ir.OADD, x, y), types.Types[types.TUNSAFEPTR])
			walkExpr(walkstate, conv, init)
		}
		return n

	case ir.OUNSAFESLICE:
		n := n.(*ir.BinaryExpr)
		return walkUnsafeSlice(walkstate, n, init)

	case ir.OUNSAFESTRING:
		n := n.(*ir.BinaryExpr)
		return walkUnsafeString(walkstate, n, init)

	case ir.OUNSAFESTRINGDATA, ir.OUNSAFESLICEDATA:
		n := n.(*ir.UnaryExpr)
		return walkUnsafeData(walkstate, n, init)

	case ir.ODOT, ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		return walkDot(walkstate, n, init)

	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n := n.(*ir.TypeAssertExpr)
		return walkDotType(walkstate, n, init)

	case ir.ODYNAMICDOTTYPE, ir.ODYNAMICDOTTYPE2:
		n := n.(*ir.DynamicTypeAssertExpr)
		return walkDynamicDotType(walkstate, n, init)

	case ir.OLEN, ir.OCAP:
		n := n.(*ir.UnaryExpr)
		return walkLenCap(walkstate, n, init)

	case ir.OCOMPLEX:
		n := n.(*ir.BinaryExpr)
		n.X = walkExpr(walkstate, n.X, init)
		n.Y = walkExpr(walkstate, n.Y, init)
		return n

	case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		n := n.(*ir.BinaryExpr)
		return walkCompare(walkstate, n, init)

	case ir.OANDAND, ir.OOROR:
		n := n.(*ir.LogicalExpr)
		return walkLogical(walkstate, n, init)

	case ir.OPRINT, ir.OPRINTLN:
		return walkPrint(walkstate, n.(*ir.CallExpr), init)

	case ir.OPANIC:
		n := n.(*ir.UnaryExpr)
		return mkcall(walkstate, "gopanic", nil, init, n.X)

	case ir.ORECOVER:
		return walkRecover(walkstate, n.(*ir.CallExpr), init)

	case ir.OCFUNC:
		return n

	case ir.OCALLINTER, ir.OCALLFUNC:
		n := n.(*ir.CallExpr)
		return walkCall(walkstate, n, init)

	case ir.OAS, ir.OASOP:
		return walkAssign(walkstate, init, n)

	case ir.OAS2:
		n := n.(*ir.AssignListStmt)
		return walkAssignList(walkstate, init, n)

	// a,b,... = fn()
	case ir.OAS2FUNC:
		n := n.(*ir.AssignListStmt)
		return walkAssignFunc(walkstate, init, n)

	// x, y = <-c
	// order.stmt made sure x is addressable or blank.
	case ir.OAS2RECV:
		n := n.(*ir.AssignListStmt)
		return walkAssignRecv(walkstate, init, n)

	// a,b = m[i]
	case ir.OAS2MAPR:
		n := n.(*ir.AssignListStmt)
		return walkAssignMapRead(walkstate, init, n)

	case ir.ODELETE:
		n := n.(*ir.CallExpr)
		return walkDelete(walkstate, init, n)

	case ir.OAS2DOTTYPE:
		n := n.(*ir.AssignListStmt)
		return walkAssignDotType(walkstate, n, init)

	case ir.OCONVIFACE:
		n := n.(*ir.ConvExpr)
		return walkConvInterface(walkstate, n, init)

	case ir.OCONV, ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		return walkConv(walkstate, n, init)

	case ir.OSLICE2ARR:
		n := n.(*ir.ConvExpr)
		return walkSliceToArray(walkstate, n, init)

	case ir.OSLICE2ARRPTR:
		n := n.(*ir.ConvExpr)
		n.X = walkExpr(walkstate, n.X, init)
		return n

	case ir.ODIV, ir.OMOD:
		n := n.(*ir.BinaryExpr)
		return walkDivMod(walkstate, n, init)

	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		return walkIndex(walkstate, n, init)

	case ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		return walkIndexMap(walkstate, n, init)

	case ir.ORECV:
		base.Fatalf("walkExpr ORECV") // should see inside OAS only
		panic("unreachable")

	case ir.OSLICEHEADER:
		n := n.(*ir.SliceHeaderExpr)
		return walkSliceHeader(walkstate, n, init)

	case ir.OSTRINGHEADER:
		n := n.(*ir.StringHeaderExpr)
		return walkStringHeader(walkstate, n, init)

	case ir.OSLICE, ir.OSLICEARR, ir.OSLICESTR, ir.OSLICE3, ir.OSLICE3ARR:
		n := n.(*ir.SliceExpr)
		return walkSlice(walkstate, n, init)

	case ir.ONEW:
		n := n.(*ir.UnaryExpr)
		return walkNew(walkstate, n, init)

	case ir.OADDSTR:
		return walkAddString(walkstate, n.(*ir.AddStringExpr), init, nil)

	case ir.OAPPEND:
		// order should make sure we only see OAS(node, OAPPEND), which we handle above.
		base.Fatalf("append outside assignment")
		panic("unreachable")

	case ir.OCOPY:
		return walkCopy(walkstate, n.(*ir.BinaryExpr), init, base.Flag.Cfg.Instrumenting && !base.Flag.CompilingRuntime)

	case ir.OCLEAR:
		n := n.(*ir.UnaryExpr)
		return walkClear(walkstate, n, init)

	case ir.OCLOSE:
		n := n.(*ir.UnaryExpr)
		return walkClose(walkstate, n, init)

	case ir.OMAKECHAN:
		n := n.(*ir.MakeExpr)
		return walkMakeChan(walkstate, n, init)

	case ir.OMAKEMAP:
		n := n.(*ir.MakeExpr)
		return walkMakeMap(walkstate, n, init)

	case ir.OMAKESLICE:
		n := n.(*ir.MakeExpr)
		return walkMakeSlice(walkstate, n, init)

	case ir.OMAKESLICECOPY:
		n := n.(*ir.MakeExpr)
		return walkMakeSliceCopy(walkstate, n, init)

	case ir.ORUNESTR:
		n := n.(*ir.ConvExpr)
		return walkRuneToString(walkstate, n, init)

	case ir.OBYTES2STR, ir.ORUNES2STR:
		n := n.(*ir.ConvExpr)
		return walkBytesRunesToString(walkstate, n, init)

	case ir.OBYTES2STRTMP:
		n := n.(*ir.ConvExpr)
		return walkBytesToStringTemp(walkstate, n, init)

	case ir.OSTR2BYTES:
		n := n.(*ir.ConvExpr)
		return walkStringToBytes(walkstate, n, init)

	case ir.OSTR2BYTESTMP:
		n := n.(*ir.ConvExpr)
		return walkStringToBytesTemp(walkstate, n, init)

	case ir.OSTR2RUNES:
		n := n.(*ir.ConvExpr)
		return walkStringToRunes(walkstate, n, init)

	case ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT, ir.OSTRUCTLIT, ir.OPTRLIT:
		return walkCompLit(walkstate, n, init)

	case ir.OSEND:
		n := n.(*ir.SendStmt)
		return walkSend(walkstate, n, init)

	case ir.OCLOSURE:
		return walkClosure(walkstate, n.(*ir.ClosureExpr), init)

	case ir.OMETHVALUE:
		return walkMethodValue(walkstate, n.(*ir.SelectorExpr), init)

	case ir.OMOVE2HEAP:
		n := n.(*ir.MoveToHeapExpr)
		n.Slice = walkExpr(walkstate, n.Slice, init)
		return n
	}

	// No return! Each case must return (or panic),
	// to avoid confusion about what gets returned
	// in the presence of type assertions.
}

// walk the whole tree of the body of an
// expression or simple statement.
// the types expressions are calculated.
// compile-time constants are evaluated.
// complex side effects like statements are appended to init.
func walkExprList(walkstate *walkState, s []ir.Node, init *ir.Nodes) {
	for i := range s {
		s[i] = walkExpr(walkstate, s[i], init)
	}
}

func walkExprListCheap(walkstate *walkState, s []ir.Node, init *ir.Nodes) {
	for i, n := range s {
		s[i] = cheapExpr(walkstate, n, init)
		s[i] = walkExpr(walkstate, s[i], init)
	}
}

func walkExprListSafe(walkstate *walkState, s []ir.Node, init *ir.Nodes) {
	for i, n := range s {
		s[i] = safeExpr(walkstate, n, init)
		s[i] = walkExpr(walkstate, s[i], init)
	}
}

// return side-effect free and cheap n, appending side effects to init.
// result may not be assignable.
func cheapExpr(walkstate *walkState, n ir.Node, init *ir.Nodes) ir.Node {
	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n
	}

	return copyExpr(walkstate, n, n.Type(), init)
}

// return side effect-free n, appending side effects to init.
// result is assignable if n is.
func safeExpr(walkstate *walkState, n ir.Node, init *ir.Nodes) ir.Node {
	if n == nil {
		return nil
	}

	if len(n.Init()) != 0 {
		walkStmtList(walkstate, n.Init())
		init.Append(ir.TakeInit(n)...)
	}

	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL, ir.OLINKSYMOFFSET:
		return n

	case ir.OLEN, ir.OCAP:
		n := n.(*ir.UnaryExpr)
		l := safeExpr(walkstate, n.X, init)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.UnaryExpr)
		a.X = l
		return walkExpr(walkstate, typecheck.Expr(a), init)

	case ir.ODOT, ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		l := safeExpr(walkstate, n.X, init)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.SelectorExpr)
		a.X = l
		return walkExpr(walkstate, typecheck.Expr(a), init)

	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		l := safeExpr(walkstate, n.X, init)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.StarExpr)
		a.X = l
		return walkExpr(walkstate, typecheck.Expr(a), init)

	case ir.OINDEX, ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		l := safeExpr(walkstate, n.X, init)
		r := safeExpr(walkstate, n.Index, init)
		if l == n.X && r == n.Index {
			return n
		}
		a := ir.Copy(n).(*ir.IndexExpr)
		a.X = l
		a.Index = r
		return walkExpr(walkstate, typecheck.Expr(a), init)

	case ir.OSTRUCTLIT, ir.OARRAYLIT, ir.OSLICELIT:
		n := n.(*ir.CompLitExpr)
		if isStaticCompositeLiteral(n) {
			return n
		}
	}

	// make a copy; must not be used as an lvalue
	if ir.IsAddressable(n) {
		base.Fatalf("missing lvalue case in safeExpr: %v", n)
	}
	return cheapExpr(walkstate, n, init)
}

func copyExpr(walkstate *walkState, n ir.Node, t *types.Type, init *ir.Nodes) ir.Node {
	l := typecheck.TempAt(base.Pos, walkstate.curfunc, t)
	appendWalkStmt(walkstate, init, ir.NewAssignStmt(base.Pos, l, n))
	return l
}

// walkAddString walks a string concatenation expression x.
// If conv is non nil, x is the conv.X field.
func walkAddString(walkstate *walkState, x *ir.AddStringExpr, init *ir.Nodes, conv *ir.ConvExpr) ir.Node {
	c := len(x.List)
	if c < 2 {
		base.Fatalf("walkAddString count %d too small", c)
	}

	typ := x.Type()
	if conv != nil {
		typ = conv.Type()
	}

	// list of string arguments
	var args []ir.Node

	var fn, fnsmall, fnbig string

	buf := typecheck.NodNil()
	switch {
	default:
		base.FatalfAt(x.Pos(), "unexpected type: %v", typ)
	case typ.IsString():
		if x.Esc() == ir.EscNone {
			sz := int64(0)
			for _, n1 := range x.List {
				if n1.Op() == ir.OLITERAL {
					sz += int64(len(ir.StringVal(n1)))
				}
			}

			// Don't allocate the buffer if the result won't fit.
			if sz < tmpstringbufsize {
				// Create temporary buffer for result string on stack.
				buf = stackBufAddr(walkstate, tmpstringbufsize, types.Types[types.TUINT8])
			}
		}

		args = []ir.Node{buf}
		fnsmall, fnbig = "concatstring%d", "concatstrings"
	case typ.IsSlice() && typ.Elem().IsKind(types.TUINT8): // Optimize []byte(str1+str2+...)
		if conv != nil && conv.Esc() == ir.EscNone {
			buf = stackBufAddr(walkstate, tmpstringbufsize, types.Types[types.TUINT8])
		}
		args = []ir.Node{buf}
		fnsmall, fnbig = "concatbyte%d", "concatbytes"
	}

	if c <= 5 {
		// small numbers of strings use direct runtime helpers.
		// note: order.expr knows this cutoff too.
		fn = fmt.Sprintf(fnsmall, c)

		for _, n2 := range x.List {
			args = append(args, typecheck.Conv(n2, types.Types[types.TSTRING]))
		}
	} else {
		// large numbers of strings are passed to the runtime as a slice.
		fn = fnbig
		t := types.NewSlice(types.Types[types.TSTRING])

		slargs := make([]ir.Node, len(x.List))
		for i, n2 := range x.List {
			slargs[i] = typecheck.Conv(n2, types.Types[types.TSTRING])
		}
		slice := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, t, slargs)
		slice.Prealloc = x.Prealloc
		args = append(args, slice)
		slice.SetEsc(ir.EscNone)
	}

	cat := typecheck.LookupRuntime(fn)
	r := ir.NewCallExpr(base.Pos, ir.OCALL, cat, nil)
	r.Args = args
	r1 := typecheck.Expr(r)
	r1 = walkExpr(walkstate, r1, init)
	r1.SetType(typ)

	return r1
}

type hookInfo struct {
	paramType   types.Kind
	argsNum     int
	runtimeFunc string
}

var hooks = map[string]hookInfo{
	"strings.EqualFold": {paramType: types.TSTRING, argsNum: 2, runtimeFunc: "libfuzzerHookEqualFold"},
}

// walkCall walks an OCALLFUNC or OCALLINTER node.
func walkCall(walkstate *walkState, n *ir.CallExpr, init *ir.Nodes) ir.Node {
	if n.Op() == ir.OCALLMETH {
		base.FatalfAt(n.Pos(), "OCALLMETH missed by typecheck")
	}
	if n.Op() == ir.OCALLINTER || n.Fun.Op() == ir.OMETHEXPR {
		// We expect both interface call reflect.Type.Method and concrete
		// call reflect.(*rtype).Method.
		usemethod(walkstate, n)
	}
	if n.Op() == ir.OCALLINTER {
		reflectdata.MarkUsedIfaceMethod(n)
	}

	if n.Op() == ir.OCALLFUNC && n.Fun.Op() == ir.OCLOSURE {
		directClosureCall(walkstate, n)
	}

	if ir.IsFuncPCIntrinsic(n) {
		// For internal/abi.FuncPCABIxxx(fn), if fn is a defined function, rewrite
		// it to the address of the function of the ABI fn is defined.
		name := n.Fun.(*ir.Name).Sym().Name
		arg := n.Args[0]
		var wantABI obj.ABI
		switch name {
		case "FuncPCABI0":
			wantABI = obj.ABI0
		case "FuncPCABIInternal":
			wantABI = obj.ABIInternal
		}
		if n.Type() != types.Types[types.TUINTPTR] {
			base.FatalfAt(n.Pos(), "FuncPC intrinsic should return uintptr, got %v", n.Type()) // as expected by typecheck.FuncPC.
		}
		n := ir.FuncPC(n.Pos(), arg, wantABI)
		return walkExpr(walkstate, n, init)
	}

	if n.Op() == ir.OCALLFUNC {
		fn := ir.StaticCalleeName(n.Fun)
		if fn != nil && fn.Sym().Pkg.Path == "internal/abi" && strings.HasPrefix(fn.Sym().Name, "EscapeNonString[") {
			// internal/abi.EscapeNonString[T] is a compiler intrinsic
			// for the escape analysis to escape its argument based on
			// the type. The call itself is no-op. Just walk the
			// argument.
			ps := fn.Type().Params()
			if len(ps) == 2 && ps[1].Type.IsShape() {
				return walkExpr(walkstate, n.Args[1], init)
			}
		}
	}

	if name, ok := n.Fun.(*ir.Name); ok {
		sym := name.Sym()
		if sym.Pkg.Path == "go.runtime" && sym.Name == "deferrangefunc" {
			// Call to runtime.deferrangefunc is being shared with a range-over-func
			// body that might add defers to this frame, so we cannot use open-coded defers
			// and we need to call deferreturn even if we don't see any other explicit defers.
			walkstate.curfunc.SetHasDefer(true)
			walkstate.curfunc.SetOpenCodedDeferDisallowed(true)
		}
	}

	walkCall1(walkstate, n, init)
	return n
}

func walkCall1(walkstate *walkState, n *ir.CallExpr, init *ir.Nodes) {
	if n.Walked() {
		return // already walked
	}
	n.SetWalked(true)

	if n.Op() == ir.OCALLMETH {
		base.FatalfAt(n.Pos(), "OCALLMETH missed by typecheck")
	}

	args := n.Args
	params := n.Fun.Type().Params()

	n.Fun = walkExpr(walkstate, n.Fun, init)
	walkExprList(walkstate, args, init)

	for i, arg := range args {
		// Validate argument and parameter types match.
		param := params[i]
		if !types.Identical(arg.Type(), param.Type) {
			base.FatalfAt(n.Pos(), "assigning %L to parameter %v (type %v)", arg, param.Sym, param.Type)
		}

		// For any argument whose evaluation might require a function call,
		// store that argument into a temporary variable,
		// to prevent that calls from clobbering arguments already on the stack.
		if mayCall(arg) {
			// assignment of arg to Temp
			tmp := typecheck.TempAt(base.Pos, walkstate.curfunc, param.Type)
			init.Append(convas(walkstate, typecheck.Stmt(ir.NewAssignStmt(base.Pos, tmp, arg)).(*ir.AssignStmt), init))
			// replace arg with temp
			args[i] = tmp
		}
	}

	funSym := n.Fun.Sym()
	if base.Debug.Libfuzzer != 0 && funSym != nil {
		if hook, found := hooks[funSym.Pkg.Path+"."+funSym.Name]; found {
			if len(args) != hook.argsNum {
				panic(fmt.Sprintf("%s.%s expects %d arguments, but received %d", funSym.Pkg.Path, funSym.Name, hook.argsNum, len(args)))
			}
			var hookArgs []ir.Node
			for _, arg := range args {
				hookArgs = append(hookArgs, tracecmpArg(walkstate, arg, types.Types[hook.paramType], init))
			}
			hookArgs = append(hookArgs, fakePC(n))
			init.Append(mkcall(walkstate, hook.runtimeFunc, nil, init, hookArgs...))
		}
	}
}

// walkDivMod walks an ODIV or OMOD node.
func walkDivMod(walkstate *walkState, n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(walkstate, n.X, init)
	n.Y = walkExpr(walkstate, n.Y, init)

	// rewrite complex div into function call.
	et := n.X.Type().Kind()

	if types.IsComplex[et] && n.Op() == ir.ODIV {
		t := n.Type()
		call := mkcall(walkstate, "complex128div", types.Types[types.TCOMPLEX128], init, typecheck.Conv(n.X, types.Types[types.TCOMPLEX128]), typecheck.Conv(n.Y, types.Types[types.TCOMPLEX128]))
		return typecheck.Conv(call, t)
	}

	// Nothing to do for float divisions.
	if types.IsFloat[et] {
		return n
	}

	// rewrite 64-bit div and mod on 32-bit architectures.
	// TODO: Remove this code once we can introduce
	// runtime calls late in SSA processing.
	if types.RegSize < 8 && (et == types.TINT64 || et == types.TUINT64) {
		if n.Y.Op() == ir.OLITERAL {
			// Leave div/mod by non-zero uint64 constants.
			// The SSA backend will handle those.
			// (Zero constants should have been rejected already, but we check just in case.)
			switch et {
			case types.TINT64:
				if ir.Int64Val(n.Y) != 0 {
					return n
				}
			case types.TUINT64:
				if ir.Uint64Val(n.Y) != 0 {
					return n
				}
			}
		}
		// Build call to uint64div, uint64mod, int64div, or int64mod.
		var fn string
		if et == types.TINT64 {
			fn = "int64"
		} else {
			fn = "uint64"
		}
		if n.Op() == ir.ODIV {
			fn += "div"
		} else {
			fn += "mod"
		}
		return mkcall(walkstate, fn, n.Type(), init, typecheck.Conv(n.X, types.Types[et]), typecheck.Conv(n.Y, types.Types[et]))
	}
	return n
}

// walkDot walks an ODOT or ODOTPTR node.
func walkDot(walkstate *walkState, n *ir.SelectorExpr, init *ir.Nodes) ir.Node {
	usefield(walkstate, n)
	n.X = walkExpr(walkstate, n.X, init)
	return n
}

// walkDotType walks an ODOTTYPE or ODOTTYPE2 node.
func walkDotType(walkstate *walkState, n *ir.TypeAssertExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(walkstate, n.X, init)
	// Set up interface type addresses for back end.
	if !n.Type().IsInterface() && !n.X.Type().IsEmptyInterface() {
		n.ITab = reflectdata.ITabAddrAt(base.Pos, n.Type(), n.X.Type())
	}
	if n.X.Type().IsInterface() && n.Type().IsInterface() && !n.Type().IsEmptyInterface() {
		// This kind of conversion needs a runtime call. Allocate
		// a descriptor for that call.
		n.Descriptor = makeTypeAssertDescriptor(n.Type(), n.Op() == ir.ODOTTYPE2)
	}
	return n
}

// shapeTypeAssertImpossible reports whether a type assertion from src
// to concrete type dst can never succeed because they have
// incompatible shape types.
func shapeTypeAssertImpossible(src ir.Node, dst *types.Type) bool {
	if dst.IsInterface() {
		return false
	}
	srcShape := convIfaceShapeType(src)
	if srcShape == nil {
		return false
	}
	return !types.Identical(srcShape, noder.Shapify(dst, false)) &&
		!types.Identical(srcShape, noder.Shapify(dst, true))
}

// convIfaceShapeType returns the shape type from which src was
// created via OCONVIFACE, or nil.
func convIfaceShapeType(src ir.Node) *types.Type {
	for {
		switch s := src.(type) {
		case *ir.ParenExpr:
			src = s.X
			continue
		case *ir.ConvExpr:
			if s.Op() == ir.OCONVNOP {
				src = s.X
				continue
			}
			if s.Op() == ir.OCONVIFACE {
				srcType := s.X.Type()
				if srcType != nil && !srcType.IsInterface() && srcType.IsShape() {
					return srcType
				}
				return nil
			}
		}
		break
	}

	if name, ok := src.(*ir.Name); ok && shapeConvSources != nil {
		return shapeConvSources[name.Canonical()]
	}
	return nil
}

func makeTypeAssertDescriptor(target *types.Type, canFail bool) *obj.LSym {
	// When converting from an interface to a non-empty interface. Needs a runtime call.
	// Allocate an internal/abi.TypeAssert descriptor for that call.
	lsym := types.LocalPkg.Lookup(fmt.Sprintf(".typeAssert.%d", typeAssertGen)).LinksymABI(obj.ABI0)
	typeAssertGen++
	c := rttype.NewCursor(lsym, 0, rttype.TypeAssert)
	c.Field("Cache").WritePtr(typecheck.LookupRuntimeVar("emptyTypeAssertCache"))
	c.Field("Inter").WritePtr(reflectdata.TypeLinksym(target))
	c.Field("CanFail").WriteBool(canFail)
	objw.Global(lsym, int32(rttype.TypeAssert.Size()), obj.LOCAL)
	lsym.Gotype = reflectdata.TypeLinksym(rttype.TypeAssert)
	return lsym
}

var typeAssertGen int

// walkDynamicDotType walks an ODYNAMICDOTTYPE or ODYNAMICDOTTYPE2 node.
func walkDynamicDotType(walkstate *walkState, n *ir.DynamicTypeAssertExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(walkstate, n.X, init)
	n.RType = walkExpr(walkstate, n.RType, init)
	n.ITab = walkExpr(walkstate, n.ITab, init)
	// Convert to non-dynamic if we can.
	if n.RType != nil && n.RType.Op() == ir.OADDR {
		addr := n.RType.(*ir.AddrExpr)
		if addr.X.Op() == ir.OLINKSYMOFFSET {
			r := ir.NewTypeAssertExpr(n.Pos(), n.X, n.Type())
			if n.Op() == ir.ODYNAMICDOTTYPE2 {
				r.SetOp(ir.ODOTTYPE2)
			}
			r.SetType(n.Type())
			r.SetTypecheck(1)
			return walkExpr(walkstate, r, init)
		}
	}
	return n
}

// walkIndex walks an OINDEX node.
func walkIndex(walkstate *walkState, n *ir.IndexExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(walkstate, n.X, init)

	// save the original node for bounds checking elision.
	// If it was a ODIV/OMOD walk might rewrite it.
	r := n.Index

	n.Index = walkExpr(walkstate, n.Index, init)

	// if range of type cannot exceed static array bound,
	// disable bounds check.
	if n.Bounded() {
		return n
	}
	t := n.X.Type()
	if t != nil && t.IsPtr() {
		t = t.Elem()
	}
	if t.IsArray() {
		n.SetBounded(bounded(r, t.NumElem()))
		if base.Flag.LowerM != 0 && n.Bounded() && !ir.IsConst(n.Index, constant.Int) {
			base.Warn("index bounds check elided")
		}
	} else if ir.IsConst(n.X, constant.String) {
		n.SetBounded(bounded(r, int64(len(ir.StringVal(n.X)))))
		if base.Flag.LowerM != 0 && n.Bounded() && !ir.IsConst(n.Index, constant.Int) {
			base.Warn("index bounds check elided")
		}
	}
	return n
}

// mapKeyArg returns an expression for key that is suitable to be passed
// as the key argument for runtime map* functions.
// n is the map indexing or delete Node (to provide Pos).
func mapKeyArg(fast int, n, key ir.Node, assigned bool) ir.Node {
	if fast == mapslow {
		// standard version takes key by reference.
		// orderState.expr made sure key is addressable.
		return typecheck.NodAddr(key)
	}
	if assigned {
		// mapassign does distinguish pointer vs. integer key.
		return key
	}
	// mapaccess and mapdelete don't distinguish pointer vs. integer key.
	switch fast {
	case mapfast32ptr:
		return ir.NewConvExpr(n.Pos(), ir.OCONVNOP, types.Types[types.TUINT32], key)
	case mapfast64ptr:
		return ir.NewConvExpr(n.Pos(), ir.OCONVNOP, types.Types[types.TUINT64], key)
	default:
		// fast version takes key by value.
		return key
	}
}

// walkIndexMap walks an OINDEXMAP node.
// It replaces m[k] with *map{access1,assign}(maptype, m, &k)
func walkIndexMap(walkstate *walkState, n *ir.IndexExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(walkstate, n.X, init)
	n.Index = walkExpr(walkstate, n.Index, init)
	map_ := n.X
	t := map_.Type()
	fast := mapfast(t)
	key := mapKeyArg(fast, n, n.Index, n.Assigned)
	args := []ir.Node{reflectdata.IndexMapRType(base.Pos, n), map_, key}

	var mapFn ir.Node
	switch {
	case n.Assigned:
		mapFn = mapfn(mapassign[fast], t, false)
	case t.Elem().Size() > abi.ZeroValSize:
		args = append(args, reflectdata.ZeroAddr(t.Elem().Size()))
		mapFn = mapfn("mapaccess1_fat", t, true)
	default:
		mapFn = mapfn(mapaccess1[fast], t, false)
	}
	call := mkcall1(walkstate, mapFn, nil, init, args...)
	call.SetType(types.NewPtr(t.Elem()))
	call.MarkNonNil() // mapaccess1* and mapassign always return non-nil pointers.
	star := ir.NewStarExpr(base.Pos, call)
	star.SetType(t.Elem())
	star.SetTypecheck(1)
	return star
}

// walkLogical walks an OANDAND or OOROR node.
func walkLogical(walkstate *walkState, n *ir.LogicalExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(walkstate, n.X, init)

	// cannot put side effects from n.Right on init,
	// because they cannot run before n.Left is checked.
	// save elsewhere and store on the eventual n.Right.
	var ll ir.Nodes

	n.Y = walkExpr(walkstate, n.Y, &ll)
	n.Y = ir.InitExpr(ll, n.Y)
	return n
}

// walkSend walks an OSEND node.
func walkSend(walkstate *walkState, n *ir.SendStmt, init *ir.Nodes) ir.Node {
	n1 := n.Value
	n1 = typecheck.AssignConv(n1, n.Chan.Type().Elem(), "chan send")
	n1 = walkExpr(walkstate, n1, init)
	n1 = typecheck.NodAddr(n1)
	return mkcall1(walkstate, chanfn("chansend1", 2, n.Chan.Type()), nil, init, n.Chan, n1)
}

// walkSlice walks an OSLICE, OSLICEARR, OSLICESTR, OSLICE3, or OSLICE3ARR node.
func walkSlice(walkstate *walkState, n *ir.SliceExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(walkstate, n.X, init)
	n.Low = walkExpr(walkstate, n.Low, init)
	if n.Low != nil && ir.IsZero(n.Low) {
		// Reduce x[0:j] to x[:j] and x[0:j:k] to x[:j:k].
		n.Low = nil
	}
	n.High = walkExpr(walkstate, n.High, init)
	n.Max = walkExpr(walkstate, n.Max, init)

	if (n.Op() == ir.OSLICE || n.Op() == ir.OSLICESTR) && n.Low == nil && n.High == nil {
		// Reduce x[:] to x.
		if base.Debug.Slice > 0 {
			base.Warn("slice: omit slice operation")
		}
		return n.X
	}
	return n
}

// walkSliceHeader walks an OSLICEHEADER node.
func walkSliceHeader(walkstate *walkState, n *ir.SliceHeaderExpr, init *ir.Nodes) ir.Node {
	n.Ptr = walkExpr(walkstate, n.Ptr, init)
	n.Len = walkExpr(walkstate, n.Len, init)
	n.Cap = walkExpr(walkstate, n.Cap, init)
	return n
}

// walkStringHeader walks an OSTRINGHEADER node.
func walkStringHeader(walkstate *walkState, n *ir.StringHeaderExpr, init *ir.Nodes) ir.Node {
	n.Ptr = walkExpr(walkstate, n.Ptr, init)
	n.Len = walkExpr(walkstate, n.Len, init)
	return n
}

// bounded reports whether integer n must be in range [0, max).
func bounded(n ir.Node, max int64) bool {
	if n.Type() == nil || !n.Type().IsInteger() {
		return false
	}

	sign := n.Type().IsSigned()
	bits := int32(8 * n.Type().Size())

	if ir.IsSmallIntConst(n) {
		v := ir.Int64Val(n)
		return 0 <= v && v < max
	}

	switch n.Op() {
	case ir.OAND, ir.OANDNOT:
		n := n.(*ir.BinaryExpr)
		v := int64(-1)
		switch {
		case ir.IsSmallIntConst(n.X):
			v = ir.Int64Val(n.X)
		case ir.IsSmallIntConst(n.Y):
			v = ir.Int64Val(n.Y)
			if n.Op() == ir.OANDNOT {
				v = ^v
				if !sign {
					v &= 1<<uint(bits) - 1
				}
			}
		}
		if 0 <= v && v < max {
			return true
		}

	case ir.OMOD:
		n := n.(*ir.BinaryExpr)
		if !sign && ir.IsSmallIntConst(n.Y) {
			v := ir.Int64Val(n.Y)
			if 0 <= v && v <= max {
				return true
			}
		}

	case ir.ODIV:
		n := n.(*ir.BinaryExpr)
		if !sign && ir.IsSmallIntConst(n.Y) {
			v := ir.Int64Val(n.Y)
			for bits > 0 && v >= 2 {
				bits--
				v >>= 1
			}
		}

	case ir.ORSH:
		n := n.(*ir.BinaryExpr)
		if !sign && ir.IsSmallIntConst(n.Y) {
			v := ir.Int64Val(n.Y)
			if v > int64(bits) {
				return max > 0
			}
			bits -= int32(v)
		}
	}

	if !sign && bits <= 62 && 1<<uint(bits) <= max {
		return true
	}

	return false
}

// usemethod checks calls for uses of Method and MethodByName of reflect.Value,
// reflect.Type, reflect.(*rtype), and reflect.(*interfaceType).
func usemethod(walkstate *walkState, n *ir.CallExpr) {
	// Don't mark reflect.(*rtype).Method, etc. themselves in the reflect package.
	// Those functions may be alive via the itab, which should not cause all methods
	// alive. We only want to mark their callers.
	if base.Ctxt.Pkgpath == "reflect" {
		// TODO: is there a better way than hardcoding the names?
		switch fn := walkstate.curfunc.Nname.Sym().Name; {
		case fn == "(*rtype).Method", fn == "(*rtype).MethodByName":
			return
		case fn == "(*interfaceType).Method", fn == "(*interfaceType).MethodByName":
			return
		case fn == "Value.Method", fn == "Value.MethodByName":
			return
		}
	}

	dot, ok := n.Fun.(*ir.SelectorExpr)
	if !ok {
		return
	}

	// looking for either direct method calls and interface method calls of:
	//	reflect.Type.Method        - func(int) reflect.Method
	//	reflect.Type.MethodByName  - func(string) (reflect.Method, bool)
	//
	//	reflect.Value.Method       - func(int) reflect.Value
	//	reflect.Value.MethodByName - func(string) reflect.Value
	methodName := dot.Sel.Name
	t := dot.Selection.Type

	// Check the number of arguments and return values.
	if t.NumParams() != 1 || (t.NumResults() != 1 && t.NumResults() != 2) {
		return
	}

	// Check the type of the argument.
	switch pKind := t.Param(0).Type.Kind(); {
	case methodName == "Method" && pKind == types.TINT,
		methodName == "MethodByName" && pKind == types.TSTRING:

	default:
		// not a call to Method or MethodByName of reflect.{Type,Value}.
		return
	}

	// Check that first result type is "reflect.Method" or "reflect.Value".
	// Note that we have to check sym name and sym package separately, as
	// we can't check for exact string "reflect.Method" reliably
	// (e.g., see #19028 and #38515).
	switch s := t.Result(0).Type.Sym(); {
	case s != nil && types.ReflectSymName(s) == "Method",
		s != nil && types.ReflectSymName(s) == "Value":

	default:
		// not a call to Method or MethodByName of reflect.{Type,Value}.
		return
	}

	var targetName ir.Node
	switch dot.Op() {
	case ir.ODOTINTER:
		if methodName == "MethodByName" {
			targetName = n.Args[0]
		}
	case ir.OMETHEXPR:
		if methodName == "MethodByName" {
			targetName = n.Args[1]
		}
	default:
		base.FatalfAt(dot.Pos(), "usemethod: unexpected dot.Op() %s", dot.Op())
	}

	if ir.IsConst(targetName, constant.String) {
		name := constant.StringVal(targetName.Val())
		walkstate.curfunc.LSym.AddRel(base.Ctxt, obj.Reloc{
			Type: objabi.R_USENAMEDMETHOD,
			Sym:  staticdata.StringSymNoCommon(name),
		})
	} else {
		walkstate.curfunc.LSym.Set(obj.AttrReflectMethod, true)
	}
}

func usefield(walkstate *walkState, n *ir.SelectorExpr) {
	if !buildcfg.Experiment.FieldTrack {
		return
	}

	switch n.Op() {
	default:
		base.Fatalf("usefield %v", n.Op())

	case ir.ODOT, ir.ODOTPTR:
		break
	}

	field := n.Selection
	if field == nil {
		base.Fatalf("usefield %v %v without paramfld", n.X.Type(), n.Sel)
	}
	if field.Sym != n.Sel {
		base.Fatalf("field inconsistency: %v != %v", field.Sym, n.Sel)
	}
	if !strings.Contains(field.Note, "go:\"track\"") {
		return
	}

	outer := n.X.Type()
	if outer.IsPtr() {
		outer = outer.Elem()
	}
	if outer.Sym() == nil {
		base.Errorf("tracked field must be in named struct type")
	}

	sym := reflectdata.TrackSym(outer, field)
	if walkstate.curfunc.FieldTrack == nil {
		walkstate.curfunc.FieldTrack = make(map[*obj.LSym]struct{})
	}
	walkstate.curfunc.FieldTrack[sym] = struct{}{}
}
