// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"fmt"
	"go/constant"
	"internal/buildcfg"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/staticinit"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

// Rewrite tree to use separate statements to enforce
// order of evaluation. Makes walk easier, because it
// can (after this runs) reorder at will within an expression.
//
// Rewrite m[k] op= r into m[k] = m[k] op r if op is / or %.
//
// Introduce temporaries as needed by runtime routines.
// For example, the map runtime routines take the map key
// by reference, so make sure all map keys are addressable
// by copying them to temporaries as needed.
// The same is true for channel operations.
//
// Arrange that map index expressions only appear in direct
// assignments x = m[k] or m[k] = x, never in larger expressions.
//
// Arrange that receive expressions only appear in direct assignments
// x = <-c or as standalone statements <-c, never in larger expressions.

// orderState holds state during the ordering process.
type orderState struct {
	out  []ir.Node             // list of generated statements
	temp []*ir.Name            // stack of temporary variables
	free map[string][]*ir.Name // free list of unused temporaries, by type.LinkString().
	edit func(ir.Node) ir.Node // cached closure of o.exprNoLHS
}

// order rewrites fn.Nbody to apply the ordering constraints
// described in the comment at the top of the file.
func order(fn *ir.Func) {
	if base.Flag.W > 1 {
		s := fmt.Sprintf("\nbefore order %v", fn.Sym())
		ir.DumpList(s, fn.Body)
	}
	ir.SetPos(fn) // Set reasonable position for instrumenting code. See issue 53688.
	orderBlock(&fn.Body, map[string][]*ir.Name{})
}

// append typechecks stmt and appends it to out.
func (o *orderState) append(stmt ir.Node) {
	o.out = append(o.out, typecheck.Stmt(stmt))
}

// newTemp allocates a new temporary with the given type,
// pushes it onto the temp stack, and returns it.
// If clear is true, newTemp emits code to zero the temporary.
func (o *orderState) newTemp(t *types.Type, clear bool) *ir.Name {
	var v *ir.Name
	key := t.LinkString()
	if a := o.free[key]; len(a) > 0 {
		v = a[len(a)-1]
		if !types.Identical(t, v.Type()) {
			base.Fatalf("expected %L to have type %v", v, t)
		}
		o.free[key] = a[:len(a)-1]
	} else {
		v = typecheck.TempAt(base.Pos, ir.CurFunc, t)
	}
	if clear {
		o.append(ir.NewAssignStmt(base.Pos, v, nil))
	}

	o.temp = append(o.temp, v)
	return v
}

// copyExpr behaves like newTemp but also emits
// code to initialize the temporary to the value n.
func (o *orderState) copyExpr(n ir.Node) *ir.Name {
	return o.copyExpr1(n, false)
}

// copyExprClear is like copyExpr but clears the temp before assignment.
// It is provided for use when the evaluation of tmp = n turns into
// a function call that is passed a pointer to the temporary as the output space.
// If the call blocks before tmp has been written,
// the garbage collector will still treat the temporary as live,
// so we must zero it before entering that call.
// Today, this only happens for channel receive operations.
// (The other candidate would be map access, but map access
// returns a pointer to the result data instead of taking a pointer
// to be filled in.)
func (o *orderState) copyExprClear(n ir.Node) *ir.Name {
	return o.copyExpr1(n, true)
}

func (o *orderState) copyExpr1(n ir.Node, clear bool) *ir.Name {
	t := n.Type()
	v := o.newTemp(t, clear)
	o.append(ir.NewAssignStmt(base.Pos, v, n))
	return v
}

// cheapExpr returns a cheap version of n.
// The definition of cheap is that n is a variable or constant.
// If not, cheapExpr allocates a new tmp, emits tmp = n,
// and then returns tmp.
func (o *orderState) cheapExpr(n ir.Node) ir.Node {
	if n == nil {
		return nil
	}

	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n
	case ir.OLEN, ir.OCAP:
		n := n.(*ir.UnaryExpr)
		l := o.cheapExpr(n.X)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.UnaryExpr)
		a.X = l
		return typecheck.Expr(a)
	}

	return o.copyExpr(n)
}

// safeExpr returns a safe version of n.
// The definition of safe is that n can appear multiple times
// without violating the semantics of the original program,
// and that assigning to the safe version has the same effect
// as assigning to the original n.
//
// The intended use is to apply to x when rewriting x += y into x = x + y.
func (o *orderState) safeExpr(n ir.Node) ir.Node {
	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n

	case ir.OLEN, ir.OCAP:
		n := n.(*ir.UnaryExpr)
		l := o.safeExpr(n.X)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.UnaryExpr)
		a.X = l
		return typecheck.Expr(a)

	case ir.ODOT:
		n := n.(*ir.SelectorExpr)
		l := o.safeExpr(n.X)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.SelectorExpr)
		a.X = l
		return typecheck.Expr(a)

	case ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		l := o.cheapExpr(n.X)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.SelectorExpr)
		a.X = l
		return typecheck.Expr(a)

	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		l := o.cheapExpr(n.X)
		if l == n.X {
			return n
		}
		a := ir.Copy(n).(*ir.StarExpr)
		a.X = l
		return typecheck.Expr(a)

	case ir.OINDEX, ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		var l ir.Node
		if n.X.Type().IsArray() {
			l = o.safeExpr(n.X)
		} else {
			l = o.cheapExpr(n.X)
		}
		r := o.cheapExpr(n.Index)
		if l == n.X && r == n.Index {
			return n
		}
		a := ir.Copy(n).(*ir.IndexExpr)
		a.X = l
		a.Index = r
		return typecheck.Expr(a)

	default:
		base.Fatalf("order.safeExpr %v", n.Op())
		return nil // not reached
	}
}

// addrTemp ensures that n is okay to pass by address to runtime routines.
// If the original argument n is not okay, addrTemp creates a tmp, emits
// tmp = n, and then returns tmp.
// The result of addrTemp MUST be assigned back to n, e.g.
//
//	n.Left = o.addrTemp(n.Left)
func (o *orderState) addrTemp(n ir.Node) ir.Node {
	// Note: Avoid addrTemp with static assignment for literal strings
	// when compiling FIPS packages.
	// The problem is that panic("foo") ends up creating a static RODATA temp
	// for the implicit conversion of "foo" to any, and we can't handle
	// the relocations in that temp.
	if n.Op() == ir.ONIL || (n.Op() == ir.OLITERAL && !base.Ctxt.IsFIPS()) {
		// TODO: expand this to all static composite literal nodes?
		n = typecheck.DefaultLit(n, nil)
		types.CalcSize(n.Type())
		vstat := readonlystaticname(n.Type())
		var s staticinit.Schedule
		s.StaticAssign(vstat, 0, n, n.Type())
		if s.Out != nil {
			base.Fatalf("staticassign of const generated code: %+v", n)
		}
		vstat = typecheck.Expr(vstat).(*ir.Name)
		return vstat
	}

	// Prevent taking the address of an SSA-able local variable (#63332).
	//
	// TODO(mdempsky): Note that OuterValue unwraps OCONVNOPs, but
	// IsAddressable does not. It should be possible to skip copying for
	// at least some of these OCONVNOPs (e.g., reinsert them after the
	// OADDR operation), but at least walkCompare needs to be fixed to
	// support that (see trybot failures on go.dev/cl/541715, PS1).
	if ir.IsAddressable(n) {
		if name, ok := ir.OuterValue(n).(*ir.Name); ok && name.Op() == ir.ONAME {
			if name.Class == ir.PAUTO && !name.Addrtaken() && ssa.CanSSA(name.Type()) {
				goto Copy
			}
		}

		return n
	}

Copy:
	return o.copyExpr(n)
}

// mapKeyTemp prepares n to be a key in a map runtime call and returns n.
// The first parameter is the position of n's containing node, for use in case
// that n's position is not unique (e.g., if n is an ONAME).
func (o *orderState) mapKeyTemp(outerPos src.XPos, t *types.Type, n ir.Node) ir.Node {
	pos := outerPos
	if ir.HasUniquePos(n) {
		pos = n.Pos()
	}
	// Most map calls need to take the address of the key.
	// Exception: map*_fast* calls. See golang.org/issue/19015.
	alg := mapfast(t)
	if alg == mapslow {
		return o.addrTemp(n)
	}
	var kt *types.Type
	switch alg {
	case mapfast32:
		kt = types.Types[types.TUINT32]
	case mapfast64:
		kt = types.Types[types.TUINT64]
	case mapfast32ptr, mapfast64ptr:
		kt = types.Types[types.TUNSAFEPTR]
	case mapfaststr:
		kt = types.Types[types.TSTRING]
	}
	nt := n.Type()
	switch {
	case nt == kt:
		return n
	case nt.Kind() == kt.Kind(), nt.IsPtrShaped() && kt.IsPtrShaped():
		// can directly convert (e.g. named type to underlying type, or one pointer to another)
		return typecheck.Expr(ir.NewConvExpr(pos, ir.OCONVNOP, kt, n))
	case nt.IsInteger() && kt.IsInteger():
		// can directly convert (e.g. int32 to uint32)
		if n.Op() == ir.OLITERAL && nt.IsSigned() {
			// avoid constant overflow error
			n = ir.NewConstExpr(constant.MakeUint64(uint64(ir.Int64Val(n))), n)
			n.SetType(kt)
			return n
		}
		return typecheck.Expr(ir.NewConvExpr(pos, ir.OCONV, kt, n))
	default:
		// Unsafe cast through memory.
		// We'll need to do a load with type kt. Create a temporary of type kt to
		// ensure sufficient alignment. nt may be under-aligned.
		if uint8(kt.Alignment()) < uint8(nt.Alignment()) {
			base.Fatalf("mapKeyTemp: key type is not sufficiently aligned, kt=%v nt=%v", kt, nt)
		}
		tmp := o.newTemp(kt, true)
		// *(*nt)(&tmp) = n
		var e ir.Node = typecheck.NodAddr(tmp)
		e = ir.NewConvExpr(pos, ir.OCONVNOP, nt.PtrTo(), e)
		e = ir.NewStarExpr(pos, e)
		o.append(ir.NewAssignStmt(pos, e, n))
		return tmp
	}
}

// mapKeyReplaceStrConv replaces OBYTES2STR by OBYTES2STRTMP
// in n to avoid string allocations for keys in map lookups.
// Returns a bool that signals if a modification was made.
//
// For:
//
//	x = m[string(k)]
//	x = m[T1{... Tn{..., string(k), ...}}]
//
// where k is []byte, T1 to Tn is a nesting of struct and array literals,
// the allocation of backing bytes for the string can be avoided
// by reusing the []byte backing array. These are special cases
// for avoiding allocations when converting byte slices to strings.
// It would be nice to handle these generally, but because
// []byte keys are not allowed in maps, the use of string(k)
// comes up in important cases in practice. See issue 3512.
func mapKeyReplaceStrConv(n ir.Node) bool {
	var replaced bool
	switch n.Op() {
	case ir.OBYTES2STR:
		n := n.(*ir.ConvExpr)
		n.SetOp(ir.OBYTES2STRTMP)
		replaced = true
	case ir.OSTRUCTLIT:
		n := n.(*ir.CompLitExpr)
		for _, elem := range n.List {
			elem := elem.(*ir.StructKeyExpr)
			if mapKeyReplaceStrConv(elem.Value) {
				replaced = true
			}
		}
	case ir.OARRAYLIT:
		n := n.(*ir.CompLitExpr)
		for _, elem := range n.List {
			if elem.Op() == ir.OKEY {
				elem = elem.(*ir.KeyExpr).Value
			}
			if mapKeyReplaceStrConv(elem) {
				replaced = true
			}
		}
	}
	return replaced
}

type ordermarker int

// markTemp returns the top of the temporary variable stack.
func (o *orderState) markTemp() ordermarker {
	return ordermarker(len(o.temp))
}

// popTemp pops temporaries off the stack until reaching the mark,
// which must have been returned by markTemp.
func (o *orderState) popTemp(mark ordermarker) {
	for _, n := range o.temp[mark:] {
		key := n.Type().LinkString()
		o.free[key] = append(o.free[key], n)
	}
	o.temp = o.temp[:mark]
}

// stmtList orders each of the statements in the list.
func (o *orderState) stmtList(l ir.Nodes) {
	s := l
	for i := range s {
		orderMakeSliceCopy(s[i:])
		o.stmt(s[i])
	}
}

// orderMakeSliceCopy matches the pattern:
//
//	m = OMAKESLICE([]T, x); OCOPY(m, s)
//
// and rewrites it to:
//
//	m = OMAKESLICECOPY([]T, x, s); nil
func orderMakeSliceCopy(s []ir.Node) {
	if base.Flag.N != 0 || base.Flag.Cfg.Instrumenting {
		return
	}
	if len(s) < 2 || s[0] == nil || s[0].Op() != ir.OAS || s[1] == nil || s[1].Op() != ir.OCOPY {
		return
	}

	as := s[0].(*ir.AssignStmt)
	cp := s[1].(*ir.BinaryExpr)
	if as.Y == nil || as.Y.Op() != ir.OMAKESLICE || ir.IsBlank(as.X) ||
		as.X.Op() != ir.ONAME || cp.X.Op() != ir.ONAME || cp.Y.Op() != ir.ONAME ||
		as.X.Name() != cp.X.Name() || cp.X.Name() == cp.Y.Name() {
		// The line above this one is correct with the differing equality operators:
		// we want as.X and cp.X to be the same name,
		// but we want the initial data to be coming from a different name.
		return
	}

	mk := as.Y.(*ir.MakeExpr)
	if mk.Esc() == ir.EscNone || mk.Len == nil || mk.Cap != nil {
		return
	}
	mk.SetOp(ir.OMAKESLICECOPY)
	mk.Cap = cp.Y
	// Set bounded when m = OMAKESLICE([]T, len(s)); OCOPY(m, s)
	mk.SetBounded(mk.Len.Op() == ir.OLEN && ir.SameSafeExpr(mk.Len.(*ir.UnaryExpr).X, cp.Y))
	as.Y = typecheck.Expr(mk)
	s[1] = nil // remove separate copy call
}

// edge inserts coverage instrumentation for libfuzzer.
func (o *orderState) edge() {
	if base.Debug.Libfuzzer == 0 {
		return
	}

	// Create a new uint8 counter to be allocated in section __sancov_cntrs
	counter := staticinit.StaticName(types.Types[types.TUINT8])
	counter.SetLibfuzzer8BitCounter(true)
	// As well as setting SetLibfuzzer8BitCounter, we preemptively set the
	// symbol type to SLIBFUZZER_8BIT_COUNTER so that the race detector
	// instrumentation pass (which does not have access to the flags set by
	// SetLibfuzzer8BitCounter) knows to ignore them. This information is
	// lost by the time it reaches the compile step, so SetLibfuzzer8BitCounter
	// is still necessary.
	counter.Linksym().Type = objabi.SLIBFUZZER_8BIT_COUNTER

	// We guarantee that the counter never becomes zero again once it has been
	// incremented once. This implementation follows the NeverZero optimization
	// presented by the paper:
	// "AFL++: Combining Incremental Steps of Fuzzing Research"
	// The NeverZero policy avoids the overflow to 0 by setting the counter to one
	// after it reaches 255 and so, if an edge is executed at least one time, the entry is
	// never 0.
	// Another policy presented in the paper is the Saturated Counters policy which
	// freezes the counter when it reaches the value of 255. However, a range
	// of experiments showed that doing so decreases overall performance.
	o.append(ir.NewIfStmt(base.Pos,
		ir.NewBinaryExpr(base.Pos, ir.OEQ, counter, ir.NewInt(base.Pos, 0xff)),
		[]ir.Node{ir.NewAssignStmt(base.Pos, counter, ir.NewInt(base.Pos, 1))},
		[]ir.Node{ir.NewAssignOpStmt(base.Pos, ir.OADD, counter, ir.NewInt(base.Pos, 1))}))
}

// orderBlock orders the block of statements in n into a new slice,
// and then replaces the old slice in n with the new slice.
// free is a map that can be used to obtain temporary variables by type.
func orderBlock(n *ir.Nodes, free map[string][]*ir.Name) {
	if len(*n) != 0 {
		// Set reasonable position for instrumenting code. See issue 53688.
		// It would be nice if ir.Nodes had a position (the opening {, probably),
		// but it doesn't. So we use the first statement's position instead.
		ir.SetPos((*n)[0])
	}
	var order orderState
	order.free = free
	mark := order.markTemp()
	order.edge()
	order.stmtList(*n)
	order.popTemp(mark)
	*n = order.out
}

// exprInPlace orders the side effects in *np and
// leaves them as the init list of the final *np.
// The result of exprInPlace MUST be assigned back to n, e.g.
//
//	n.Left = o.exprInPlace(n.Left)
func (o *orderState) exprInPlace(n ir.Node) ir.Node {
	var order orderState
	order.free = o.free
	n = order.expr(n, nil)
	n = ir.InitExpr(order.out, n)

	// insert new temporaries from order
	// at head of outer list.
	o.temp = append(o.temp, order.temp...)
	return n
}

// orderStmtInPlace orders the side effects of the single statement *np
// and replaces it with the resulting statement list.
// The result of orderStmtInPlace MUST be assigned back to n, e.g.
//
//	n.Left = orderStmtInPlace(n.Left)
//
// free is a map that can be used to obtain temporary variables by type.
func orderStmtInPlace(n ir.Node, free map[string][]*ir.Name) ir.Node {
	var order orderState
	order.free = free
	mark := order.markTemp()
	order.stmt(n)
	order.popTemp(mark)
	return ir.NewBlockStmt(src.NoXPos, order.out)
}

// init moves n's init list to o.out.
func (o *orderState) init(n ir.Node) {
	if ir.MayBeShared(n) {
		// For concurrency safety, don't mutate potentially shared nodes.
		// First, ensure that no work is required here.
		if len(n.Init()) > 0 {
			base.Fatalf("order.init shared node with ninit")
		}
		return
	}
	o.stmtList(ir.TakeInit(n))
}

// call orders the call expression n.
// n.Op is OCALLFUNC/OCALLINTER or a builtin like OCOPY.
func (o *orderState) call(nn ir.Node) {
	if len(nn.Init()) > 0 {
		// Caller should have already called o.init(nn).
		base.Fatalf("%v with unexpected ninit", nn.Op())
	}
	if nn.Op() == ir.OCALLMETH {
		base.FatalfAt(nn.Pos(), "OCALLMETH missed by typecheck")
	}

	// Builtin functions.
	if nn.Op() != ir.OCALLFUNC && nn.Op() != ir.OCALLINTER {
		switch n := nn.(type) {
		default:
			base.Fatalf("unexpected call: %+v", n)
		case *ir.UnaryExpr:
			n.X = o.expr(n.X, nil)
		case *ir.ConvExpr:
			n.X = o.expr(n.X, nil)
		case *ir.BinaryExpr:
			n.X = o.expr(n.X, nil)
			n.Y = o.expr(n.Y, nil)
		case *ir.MakeExpr:
			n.Len = o.expr(n.Len, nil)
			n.Cap = o.expr(n.Cap, nil)
		case *ir.CallExpr:
			o.exprList(n.Args)
		}
		return
	}

	n := nn.(*ir.CallExpr)
	typecheck.AssertFixedCall(n)

	if ir.IsFuncPCIntrinsic(n) && ir.IsIfaceOfFunc(n.Args[0]) != nil {
		// For internal/abi.FuncPCABIxxx(fn), if fn is a defined function,
		// do not introduce temporaries here, so it is easier to rewrite it
		// to symbol address reference later in walk.
		return
	}

	n.Fun = o.expr(n.Fun, nil)
	o.exprList(n.Args)
}

// mapAssign appends n to o.out.
func (o *orderState) mapAssign(n ir.Node) {
	switch n.Op() {
	default:
		base.Fatalf("order.mapAssign %v", n.Op())

	case ir.OAS:
		n := n.(*ir.AssignStmt)
		if n.X.Op() == ir.OINDEXMAP {
			n.Y = o.safeMapRHS(n.Y)
		}
		o.out = append(o.out, n)
	case ir.OASOP:
		n := n.(*ir.AssignOpStmt)
		if n.X.Op() == ir.OINDEXMAP {
			n.Y = o.safeMapRHS(n.Y)
		}
		o.out = append(o.out, n)
	}
}

func (o *orderState) safeMapRHS(r ir.Node) ir.Node {
	// Make sure we evaluate the RHS before starting the map insert.
	// We need to make sure the RHS won't panic.  See issue 22881.
	if r.Op() == ir.OAPPEND {
		r := r.(*ir.CallExpr)
		s := r.Args[1:]
		for i, n := range s {
			s[i] = o.cheapExpr(n)
		}
		return r
	}
	return o.cheapExpr(r)
}

// stmt orders the statement n, appending to o.out.
func (o *orderState) stmt(n ir.Node) {
	if n == nil {
		return
	}

	lno := ir.SetPos(n)
	o.init(n)

	switch n.Op() {
	default:
		base.Fatalf("order.stmt %v", n.Op())

	case ir.OINLMARK:
		o.out = append(o.out, n)

	case ir.OAS:
		n := n.(*ir.AssignStmt)
		t := o.markTemp()

		// There's a delicate interaction here between two OINDEXMAP
		// optimizations.
		//
		// First, we want to handle m[k] = append(m[k], ...) with a single
		// runtime call to mapassign. This requires the m[k] expressions to
		// satisfy ir.SameSafeExpr in walkAssign.
		//
		// But if k is a slow map key type that's passed by reference (e.g.,
		// byte), then we want to avoid marking user variables as addrtaken,
		// if that might prevent the compiler from keeping k in a register.
		//
		// TODO(mdempsky): It would be better if walk was responsible for
		// inserting temporaries as needed.
		mapAppend := n.X.Op() == ir.OINDEXMAP && n.Y.Op() == ir.OAPPEND &&
			ir.SameSafeExpr(n.X, n.Y.(*ir.CallExpr).Args[0])

		n.X = o.expr(n.X, nil)
		if mapAppend {
			indexLHS := n.X.(*ir.IndexExpr)
			indexLHS.X = o.cheapExpr(indexLHS.X)
			indexLHS.Index = o.cheapExpr(indexLHS.Index)

			call := n.Y.(*ir.CallExpr)
			arg0 := call.Args[0]
			// ir.SameSafeExpr skips OCONVNOPs, so we must do the same here (#66096).
			for arg0.Op() == ir.OCONVNOP {
				arg0 = arg0.(*ir.ConvExpr).X
			}
			indexRHS := arg0.(*ir.IndexExpr)
			indexRHS.X = indexLHS.X
			indexRHS.Index = indexLHS.Index

			o.exprList(call.Args[1:])
		} else {
			n.Y = o.expr(n.Y, n.X)
		}
		o.mapAssign(n)
		o.popTemp(t)

	case ir.OASOP:
		n := n.(*ir.AssignOpStmt)
		t := o.markTemp()
		n.X = o.expr(n.X, nil)
		n.Y = o.expr(n.Y, nil)

		if base.Flag.Cfg.Instrumenting || n.X.Op() == ir.OINDEXMAP && (n.AsOp == ir.ODIV || n.AsOp == ir.OMOD) {
			// Rewrite m[k] op= r into m[k] = m[k] op r so
			// that we can ensure that if op panics
			// because r is zero, the panic happens before
			// the map assignment.
			// DeepCopy is a big hammer here, but safeExpr
			// makes sure there is nothing too deep being copied.
			l1 := o.safeExpr(n.X)
			l2 := ir.DeepCopy(src.NoXPos, l1)
			if l2.Op() == ir.OINDEXMAP {
				l2 := l2.(*ir.IndexExpr)
				l2.Assigned = false
			}
			l2 = o.copyExpr(l2)
			r := o.expr(typecheck.Expr(ir.NewBinaryExpr(n.Pos(), n.AsOp, l2, n.Y)), nil)
			as := typecheck.Stmt(ir.NewAssignStmt(n.Pos(), l1, r))
			o.mapAssign(as)
			o.popTemp(t)
			return
		}

		o.mapAssign(n)
		o.popTemp(t)

	case ir.OAS2:
		n := n.(*ir.AssignListStmt)
		t := o.markTemp()
		o.exprList(n.Lhs)
		o.exprList(n.Rhs)
		o.out = append(o.out, n)
		o.popTemp(t)

	// Special: avoid copy of func call n.Right
	case ir.OAS2FUNC:
		n := n.(*ir.AssignListStmt)
		t := o.markTemp()
		o.exprList(n.Lhs)
		call := n.Rhs[0]
		o.init(call)
		if ic, ok := call.(*ir.InlinedCallExpr); ok {
			o.stmtList(ic.Body)

			n.SetOp(ir.OAS2)
			n.Rhs = ic.ReturnVars

			o.exprList(n.Rhs)
			o.out = append(o.out, n)
		} else {
			o.call(call)
			o.as2func(n)
		}
		o.popTemp(t)

	// Special: use temporary variables to hold result,
	// so that runtime can take address of temporary.
	// No temporary for blank assignment.
	//
	// OAS2MAPR: make sure key is addressable if needed,
	//           and make sure OINDEXMAP is not copied out.
	case ir.OAS2DOTTYPE, ir.OAS2RECV, ir.OAS2MAPR:
		n := n.(*ir.AssignListStmt)
		t := o.markTemp()
		o.exprList(n.Lhs)

		switch r := n.Rhs[0]; r.Op() {
		case ir.ODOTTYPE2:
			r := r.(*ir.TypeAssertExpr)
			r.X = o.expr(r.X, nil)
		case ir.ODYNAMICDOTTYPE2:
			r := r.(*ir.DynamicTypeAssertExpr)
			r.X = o.expr(r.X, nil)
			r.RType = o.expr(r.RType, nil)
			r.ITab = o.expr(r.ITab, nil)
		case ir.ORECV:
			r := r.(*ir.UnaryExpr)
			r.X = o.expr(r.X, nil)
		case ir.OINDEXMAP:
			r := r.(*ir.IndexExpr)
			r.X = o.expr(r.X, nil)
			r.Index = o.expr(r.Index, nil)
			// See similar conversion for OINDEXMAP below.
			_ = mapKeyReplaceStrConv(r.Index)
			r.Index = o.mapKeyTemp(r.Pos(), r.X.Type(), r.Index)
		default:
			base.Fatalf("order.stmt: %v", r.Op())
		}

		o.as2ok(n)
		o.popTemp(t)

	// Special: does not save n onto out.
	case ir.OBLOCK:
		n := n.(*ir.BlockStmt)
		o.stmtList(n.List)

	// Special: n->left is not an expression; save as is.
	case ir.OBREAK,
		ir.OCONTINUE,
		ir.ODCL,
		ir.OFALL,
		ir.OGOTO,
		ir.OLABEL,
		ir.OTAILCALL:
		o.out = append(o.out, n)

	// Special: handle call arguments.
	case ir.OCALLFUNC, ir.OCALLINTER:
		n := n.(*ir.CallExpr)
		t := o.markTemp()
		o.call(n)
		o.out = append(o.out, n)
		o.popTemp(t)

	case ir.OINLCALL:
		n := n.(*ir.InlinedCallExpr)
		o.stmtList(n.Body)

		// discard results; double-check for no side effects
		for _, result := range n.ReturnVars {
			if staticinit.AnySideEffects(result) {
				base.FatalfAt(result.Pos(), "inlined call result has side effects: %v", result)
			}
		}

	case ir.OCHECKNIL, ir.OCLEAR, ir.OCLOSE, ir.OPANIC, ir.ORECV:
		n := n.(*ir.UnaryExpr)
		t := o.markTemp()
		n.X = o.expr(n.X, nil)
		o.out = append(o.out, n)
		o.popTemp(t)

	case ir.OCOPY:
		n := n.(*ir.BinaryExpr)
		t := o.markTemp()
		n.X = o.expr(n.X, nil)
		n.Y = o.expr(n.Y, nil)
		o.out = append(o.out, n)
		o.popTemp(t)

	case ir.OPRINT, ir.OPRINTLN, ir.ORECOVERFP:
		n := n.(*ir.CallExpr)
		t := o.markTemp()
		o.call(n)
		o.out = append(o.out, n)
		o.popTemp(t)

	// Special: order arguments to inner call but not call itself.
	case ir.ODEFER, ir.OGO:
		n := n.(*ir.GoDeferStmt)
		t := o.markTemp()
		o.init(n.Call)
		o.call(n.Call)
		o.out = append(o.out, n)
		o.popTemp(t)

	case ir.ODELETE:
		n := n.(*ir.CallExpr)
		t := o.markTemp()
		n.Args[0] = o.expr(n.Args[0], nil)
		n.Args[1] = o.expr(n.Args[1], nil)
		n.Args[1] = o.mapKeyTemp(n.Pos(), n.Args[0].Type(), n.Args[1])
		o.out = append(o.out, n)
		o.popTemp(t)

	// Clean temporaries from condition evaluation at
	// beginning of loop body and after for statement.
	case ir.OFOR:
		n := n.(*ir.ForStmt)
		t := o.markTemp()
		n.Cond = o.exprInPlace(n.Cond)
		orderBlock(&n.Body, o.free)
		n.Post = orderStmtInPlace(n.Post, o.free)
		o.out = append(o.out, n)
		o.popTemp(t)

	// Clean temporaries from condition at
	// beginning of both branches.
	case ir.OIF:
		n := n.(*ir.IfStmt)
		t := o.markTemp()
		n.Cond = o.exprInPlace(n.Cond)
		o.popTemp(t)
		orderBlock(&n.Body, o.free)
		orderBlock(&n.Else, o.free)
		o.out = append(o.out, n)

	case ir.ORANGE:
		// n.Right is the expression being ranged over.
		// order it, and then make a copy if we need one.
		// We almost always do, to ensure that we don't
		// see any value changes made during the loop.
		// Usually the copy is cheap (e.g., array pointer,
		// chan, slice, string are all tiny).
		// The exception is ranging over an array value
		// (not a slice, not a pointer to array),
		// which must make a copy to avoid seeing updates made during
		// the range body. Ranging over an array value is uncommon though.

		// Mark []byte(str) range expression to reuse string backing storage.
		// It is safe because the storage cannot be mutated.
		n := n.(*ir.RangeStmt)
		if x, ok := n.X.(*ir.ConvExpr); ok {
			switch x.Op() {
			case ir.OSTR2BYTES:
				x.SetOp(ir.OSTR2BYTESTMP)
				fallthrough
			case ir.OSTR2BYTESTMP:
				x.MarkNonNil() // "range []byte(nil)" is fine
			}
		}

		t := o.markTemp()
		n.X = o.expr(n.X, nil)

		orderBody := true
		xt := typecheck.RangeExprType(n.X.Type())
		switch k := xt.Kind(); {
		default:
			base.Fatalf("order.stmt range %v", n.Type())

		case types.IsInt[k]:
			// Used only once, no need to copy.

		case k == types.TARRAY, k == types.TSLICE:
			if n.Value == nil || ir.IsBlank(n.Value) {
				// for i := range x will only use x once, to compute len(x).
				// No need to copy it.
				break
			}
			fallthrough

		case k == types.TCHAN, k == types.TSTRING:
			// chan, string, slice, array ranges use value multiple times.
			// make copy.
			r := n.X

			if r.Type().IsString() && r.Type() != types.Types[types.TSTRING] {
				r = ir.NewConvExpr(base.Pos, ir.OCONV, nil, r)
				r.SetType(types.Types[types.TSTRING])
				r = typecheck.Expr(r)
			}

			n.X = o.copyExpr(r)

		case k == types.TMAP:
			if isMapClear(n) {
				// Preserve the body of the map clear pattern so it can
				// be detected during walk. The loop body will not be used
				// when optimizing away the range loop to a runtime call.
				orderBody = false
				break
			}

			// copy the map value in case it is a map literal.
			// TODO(rsc): Make tmp = literal expressions reuse tmp.
			// For maps tmp is just one word so it hardly matters.
			r := n.X
			n.X = o.copyExpr(r)

			// n.Prealloc is the temp for the iterator.
			// MapIterType contains pointers and needs to be zeroed.
			if buildcfg.Experiment.SwissMap {
				n.Prealloc = o.newTemp(reflectdata.SwissMapIterType(), true)
			} else {
				n.Prealloc = o.newTemp(reflectdata.OldMapIterType(), true)
			}
		}
		n.Key = o.exprInPlace(n.Key)
		n.Value = o.exprInPlace(n.Value)
		if orderBody {
			orderBlock(&n.Body, o.free)
		}
		o.out = append(o.out, n)
		o.popTemp(t)

	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		o.exprList(n.Results)
		o.out = append(o.out, n)

	// Special: clean case temporaries in each block entry.
	// Select must enter one of its blocks, so there is no
	// need for a cleaning at the end.
	// Doubly special: evaluation order for select is stricter
	// than ordinary expressions. Even something like p.c
	// has to be hoisted into a temporary, so that it cannot be
	// reordered after the channel evaluation for a different
	// case (if p were nil, then the timing of the fault would
	// give this away).
	case ir.OSELECT:
		n := n.(*ir.SelectStmt)
		t := o.markTemp()
		for _, ncas := range n.Cases {
			r := ncas.Comm
			ir.SetPos(ncas)

			// Append any new body prologue to ninit.
			// The next loop will insert ninit into nbody.
			if len(ncas.Init()) != 0 {
				base.Fatalf("order select ninit")
			}
			if r == nil {
				continue
			}
			switch r.Op() {
			default:
				ir.Dump("select case", r)
				base.Fatalf("unknown op in select %v", r.Op())

			case ir.OSELRECV2:
				// case x, ok = <-c
				r := r.(*ir.AssignListStmt)
				recv := r.Rhs[0].(*ir.UnaryExpr)
				recv.X = o.expr(recv.X, nil)
				if !ir.IsAutoTmp(recv.X) {
					recv.X = o.copyExpr(recv.X)
				}
				init := ir.TakeInit(r)

				colas := r.Def
				do := func(i int, t *types.Type) {
					n := r.Lhs[i]
					if ir.IsBlank(n) {
						return
					}
					// If this is case x := <-ch or case x, y := <-ch, the case has
					// the ODCL nodes to declare x and y. We want to delay that
					// declaration (and possible allocation) until inside the case body.
					// Delete the ODCL nodes here and recreate them inside the body below.
					if colas {
						if len(init) > 0 && init[0].Op() == ir.ODCL && init[0].(*ir.Decl).X == n {
							init = init[1:]

							// iimport may have added a default initialization assignment,
							// due to how it handles ODCL statements.
							if len(init) > 0 && init[0].Op() == ir.OAS && init[0].(*ir.AssignStmt).X == n {
								init = init[1:]
							}
						}
						dcl := typecheck.Stmt(ir.NewDecl(base.Pos, ir.ODCL, n.(*ir.Name)))
						ncas.PtrInit().Append(dcl)
					}
					tmp := o.newTemp(t, t.HasPointers())
					as := typecheck.Stmt(ir.NewAssignStmt(base.Pos, n, typecheck.Conv(tmp, n.Type())))
					ncas.PtrInit().Append(as)
					r.Lhs[i] = tmp
				}
				do(0, recv.X.Type().Elem())
				do(1, types.Types[types.TBOOL])
				if len(init) != 0 {
					ir.DumpList("ninit", init)
					base.Fatalf("ninit on select recv")
				}
				orderBlock(ncas.PtrInit(), o.free)

			case ir.OSEND:
				r := r.(*ir.SendStmt)
				if len(r.Init()) != 0 {
					ir.DumpList("ninit", r.Init())
					base.Fatalf("ninit on select send")
				}

				// case c <- x
				// r->left is c, r->right is x, both are always evaluated.
				r.Chan = o.expr(r.Chan, nil)

				if !ir.IsAutoTmp(r.Chan) {
					r.Chan = o.copyExpr(r.Chan)
				}
				r.Value = o.expr(r.Value, nil)
				if !ir.IsAutoTmp(r.Value) {
					r.Value = o.copyExpr(r.Value)
				}
			}
		}
		// Now that we have accumulated all the temporaries, clean them.
		// Also insert any ninit queued during the previous loop.
		// (The temporary cleaning must follow that ninit work.)
		for _, cas := range n.Cases {
			orderBlock(&cas.Body, o.free)

			// TODO(mdempsky): Is this actually necessary?
			// walkSelect appears to walk Ninit.
			cas.Body.Prepend(ir.TakeInit(cas)...)
		}

		o.out = append(o.out, n)
		o.popTemp(t)

	// Special: value being sent is passed as a pointer; make it addressable.
	case ir.OSEND:
		n := n.(*ir.SendStmt)
		t := o.markTemp()
		n.Chan = o.expr(n.Chan, nil)
		n.Value = o.expr(n.Value, nil)
		if base.Flag.Cfg.Instrumenting {
			// Force copying to the stack so that (chan T)(nil) <- x
			// is still instrumented as a read of x.
			n.Value = o.copyExpr(n.Value)
		} else {
			n.Value = o.addrTemp(n.Value)
		}
		o.out = append(o.out, n)
		o.popTemp(t)

	// TODO(rsc): Clean temporaries more aggressively.
	// Note that because walkSwitch will rewrite some of the
	// switch into a binary search, this is not as easy as it looks.
	// (If we ran that code here we could invoke order.stmt on
	// the if-else chain instead.)
	// For now just clean all the temporaries at the end.
	// In practice that's fine.
	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		if base.Debug.Libfuzzer != 0 && !hasDefaultCase(n) {
			// Add empty "default:" case for instrumentation.
			n.Cases = append(n.Cases, ir.NewCaseStmt(base.Pos, nil, nil))
		}

		t := o.markTemp()
		n.Tag = o.expr(n.Tag, nil)
		for _, ncas := range n.Cases {
			o.exprListInPlace(ncas.List)
			orderBlock(&ncas.Body, o.free)
		}

		o.out = append(o.out, n)
		o.popTemp(t)
	}

	base.Pos = lno
}

func hasDefaultCase(n *ir.SwitchStmt) bool {
	for _, ncas := range n.Cases {
		if len(ncas.List) == 0 {
			return true
		}
	}
	return false
}

// exprList orders the expression list l into o.
func (o *orderState) exprList(l ir.Nodes) {
	s := l
	for i := range s {
		s[i] = o.expr(s[i], nil)
	}
}

// exprListInPlace orders the expression list l but saves
// the side effects on the individual expression ninit lists.
func (o *orderState) exprListInPlace(l ir.Nodes) {
	s := l
	for i := range s {
		s[i] = o.exprInPlace(s[i])
	}
}

func (o *orderState) exprNoLHS(n ir.Node) ir.Node {
	return o.expr(n, nil)
}

// expr orders a single expression, appending side
// effects to o.out as needed.
// If this is part of an assignment lhs = *np, lhs is given.
// Otherwise lhs == nil. (When lhs != nil it may be possible
// to avoid copying the result of the expression to a temporary.)
// The result of expr MUST be assigned back to n, e.g.
//
//	n.Left = o.expr(n.Left, lhs)
func (o *orderState) expr(n, lhs ir.Node) ir.Node {
	if n == nil {
		return n
	}
	lno := ir.SetPos(n)
	n = o.expr1(n, lhs)
	base.Pos = lno
	return n
}

func (o *orderState) expr1(n, lhs ir.Node) ir.Node {
	o.init(n)

	switch n.Op() {
	default:
		if o.edit == nil {
			o.edit = o.exprNoLHS // create closure once
		}
		ir.EditChildren(n, o.edit)
		return n

	// Addition of strings turns into a function call.
	// Allocate a temporary to hold the strings.
	// Fewer than 5 strings use direct runtime helpers.
	case ir.OADDSTR:
		n := n.(*ir.AddStringExpr)
		o.exprList(n.List)

		if len(n.List) > 5 {
			t := types.NewArray(types.Types[types.TSTRING], int64(len(n.List)))
			n.Prealloc = o.newTemp(t, false)
		}

		// Mark string(byteSlice) arguments to reuse byteSlice backing
		// buffer during conversion. String concatenation does not
		// memorize the strings for later use, so it is safe.
		// However, we can do it only if there is at least one non-empty string literal.
		// Otherwise if all other arguments are empty strings,
		// concatstrings will return the reference to the temp string
		// to the caller.
		hasbyte := false

		haslit := false
		for _, n1 := range n.List {
			hasbyte = hasbyte || n1.Op() == ir.OBYTES2STR
			haslit = haslit || n1.Op() == ir.OLITERAL && len(ir.StringVal(n1)) != 0
		}

		if haslit && hasbyte {
			for _, n2 := range n.List {
				if n2.Op() == ir.OBYTES2STR {
					n2 := n2.(*ir.ConvExpr)
					n2.SetOp(ir.OBYTES2STRTMP)
				}
			}
		}
		return n

	case ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		n.X = o.expr(n.X, nil)
		n.Index = o.expr(n.Index, nil)
		needCopy := false

		if !n.Assigned {
			// Enforce that any []byte slices we are not copying
			// can not be changed before the map index by forcing
			// the map index to happen immediately following the
			// conversions. See copyExpr a few lines below.
			needCopy = mapKeyReplaceStrConv(n.Index)

			if base.Flag.Cfg.Instrumenting {
				// Race detector needs the copy.
				needCopy = true
			}
		}

		// key may need to be addressable
		n.Index = o.mapKeyTemp(n.Pos(), n.X.Type(), n.Index)
		if needCopy {
			return o.copyExpr(n)
		}
		return n

	// concrete type (not interface) argument might need an addressable
	// temporary to pass to the runtime conversion routine.
	case ir.OCONVIFACE:
		n := n.(*ir.ConvExpr)
		n.X = o.expr(n.X, nil)
		if n.X.Type().IsInterface() {
			return n
		}
		if _, _, needsaddr := dataWordFuncName(n.X.Type()); needsaddr || isStaticCompositeLiteral(n.X) {
			// Need a temp if we need to pass the address to the conversion function.
			// We also process static composite literal node here, making a named static global
			// whose address we can put directly in an interface (see OCONVIFACE case in walk).
			n.X = o.addrTemp(n.X)
		}
		return n

	case ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		if n.X.Op() == ir.OCALLMETH {
			base.FatalfAt(n.X.Pos(), "OCALLMETH missed by typecheck")
		}
		if n.Type().IsKind(types.TUNSAFEPTR) && n.X.Type().IsKind(types.TUINTPTR) && (n.X.Op() == ir.OCALLFUNC || n.X.Op() == ir.OCALLINTER) {
			call := n.X.(*ir.CallExpr)
			// When reordering unsafe.Pointer(f()) into a separate
			// statement, the conversion and function call must stay
			// together. See golang.org/issue/15329.
			o.init(call)
			o.call(call)
			if lhs == nil || lhs.Op() != ir.ONAME || base.Flag.Cfg.Instrumenting {
				return o.copyExpr(n)
			}
		} else {
			n.X = o.expr(n.X, nil)
		}
		return n

	case ir.OANDAND, ir.OOROR:
		// ... = LHS && RHS
		//
		// var r bool
		// r = LHS
		// if r {       // or !r, for OROR
		//     r = RHS
		// }
		// ... = r

		n := n.(*ir.LogicalExpr)
		r := o.newTemp(n.Type(), false)

		// Evaluate left-hand side.
		lhs := o.expr(n.X, nil)
		o.out = append(o.out, typecheck.Stmt(ir.NewAssignStmt(base.Pos, r, lhs)))

		// Evaluate right-hand side, save generated code.
		saveout := o.out
		o.out = nil
		t := o.markTemp()
		o.edge()
		rhs := o.expr(n.Y, nil)
		o.out = append(o.out, typecheck.Stmt(ir.NewAssignStmt(base.Pos, r, rhs)))
		o.popTemp(t)
		gen := o.out
		o.out = saveout

		// If left-hand side doesn't cause a short-circuit, issue right-hand side.
		nif := ir.NewIfStmt(base.Pos, r, nil, nil)
		if n.Op() == ir.OANDAND {
			nif.Body = gen
		} else {
			nif.Else = gen
		}
		o.out = append(o.out, nif)
		return r

	case ir.OCALLMETH:
		base.FatalfAt(n.Pos(), "OCALLMETH missed by typecheck")
		panic("unreachable")

	case ir.OCALLFUNC,
		ir.OCALLINTER,
		ir.OCAP,
		ir.OCOMPLEX,
		ir.OCOPY,
		ir.OIMAG,
		ir.OLEN,
		ir.OMAKECHAN,
		ir.OMAKEMAP,
		ir.OMAKESLICE,
		ir.OMAKESLICECOPY,
		ir.OMAX,
		ir.OMIN,
		ir.ONEW,
		ir.OREAL,
		ir.ORECOVERFP,
		ir.OSTR2BYTES,
		ir.OSTR2BYTESTMP,
		ir.OSTR2RUNES:

		if isRuneCount(n) {
			// len([]rune(s)) is rewritten to runtime.countrunes(s) later.
			conv := n.(*ir.UnaryExpr).X.(*ir.ConvExpr)
			conv.X = o.expr(conv.X, nil)
		} else {
			o.call(n)
		}

		if lhs == nil || lhs.Op() != ir.ONAME || base.Flag.Cfg.Instrumenting {
			return o.copyExpr(n)
		}
		return n

	case ir.OINLCALL:
		n := n.(*ir.InlinedCallExpr)
		o.stmtList(n.Body)
		return n.SingleResult()

	case ir.OAPPEND:
		// Check for append(x, make([]T, y)...) .
		n := n.(*ir.CallExpr)
		if isAppendOfMake(n) {
			n.Args[0] = o.expr(n.Args[0], nil) // order x
			mk := n.Args[1].(*ir.MakeExpr)
			mk.Len = o.expr(mk.Len, nil) // order y
		} else {
			o.exprList(n.Args)
		}

		if lhs == nil || lhs.Op() != ir.ONAME && !ir.SameSafeExpr(lhs, n.Args[0]) {
			return o.copyExpr(n)
		}
		return n

	case ir.OSLICE, ir.OSLICEARR, ir.OSLICESTR, ir.OSLICE3, ir.OSLICE3ARR:
		n := n.(*ir.SliceExpr)
		n.X = o.expr(n.X, nil)
		n.Low = o.cheapExpr(o.expr(n.Low, nil))
		n.High = o.cheapExpr(o.expr(n.High, nil))
		n.Max = o.cheapExpr(o.expr(n.Max, nil))
		if lhs == nil || lhs.Op() != ir.ONAME && !ir.SameSafeExpr(lhs, n.X) {
			return o.copyExpr(n)
		}
		return n

	case ir.OCLOSURE:
		n := n.(*ir.ClosureExpr)
		if n.Transient() && len(n.Func.ClosureVars) > 0 {
			n.Prealloc = o.newTemp(typecheck.ClosureType(n), false)
		}
		return n

	case ir.OMETHVALUE:
		n := n.(*ir.SelectorExpr)
		n.X = o.expr(n.X, nil)
		if n.Transient() {
			t := typecheck.MethodValueType(n)
			n.Prealloc = o.newTemp(t, false)
		}
		return n

	case ir.OSLICELIT:
		n := n.(*ir.CompLitExpr)
		o.exprList(n.List)
		if n.Transient() {
			t := types.NewArray(n.Type().Elem(), n.Len)
			n.Prealloc = o.newTemp(t, false)
		}
		return n

	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n := n.(*ir.TypeAssertExpr)
		n.X = o.expr(n.X, nil)
		if !types.IsDirectIface(n.Type()) || base.Flag.Cfg.Instrumenting {
			return o.copyExprClear(n)
		}
		return n

	case ir.ORECV:
		n := n.(*ir.UnaryExpr)
		n.X = o.expr(n.X, nil)
		return o.copyExprClear(n)

	case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		n := n.(*ir.BinaryExpr)
		n.X = o.expr(n.X, nil)
		n.Y = o.expr(n.Y, nil)

		t := n.X.Type()
		switch {
		case t.IsString():
			// Mark string(byteSlice) arguments to reuse byteSlice backing
			// buffer during conversion. String comparison does not
			// memorize the strings for later use, so it is safe.
			if n.X.Op() == ir.OBYTES2STR {
				n.X.(*ir.ConvExpr).SetOp(ir.OBYTES2STRTMP)
			}
			if n.Y.Op() == ir.OBYTES2STR {
				n.Y.(*ir.ConvExpr).SetOp(ir.OBYTES2STRTMP)
			}

		case t.IsStruct() || t.IsArray():
			// for complex comparisons, we need both args to be
			// addressable so we can pass them to the runtime.
			n.X = o.addrTemp(n.X)
			n.Y = o.addrTemp(n.Y)
		}
		return n

	case ir.OMAPLIT:
		// Order map by converting:
		//   map[int]int{
		//     a(): b(),
		//     c(): d(),
		//     e(): f(),
		//   }
		// to
		//   m := map[int]int{}
		//   m[a()] = b()
		//   m[c()] = d()
		//   m[e()] = f()
		// Then order the result.
		// Without this special case, order would otherwise compute all
		// the keys and values before storing any of them to the map.
		// See issue 26552.
		n := n.(*ir.CompLitExpr)
		entries := n.List
		statics := entries[:0]
		var dynamics []*ir.KeyExpr
		for _, r := range entries {
			r := r.(*ir.KeyExpr)

			if !isStaticCompositeLiteral(r.Key) || !isStaticCompositeLiteral(r.Value) {
				dynamics = append(dynamics, r)
				continue
			}

			// Recursively ordering some static entries can change them to dynamic;
			// e.g., OCONVIFACE nodes. See #31777.
			r = o.expr(r, nil).(*ir.KeyExpr)
			if !isStaticCompositeLiteral(r.Key) || !isStaticCompositeLiteral(r.Value) {
				dynamics = append(dynamics, r)
				continue
			}

			statics = append(statics, r)
		}
		n.List = statics

		if len(dynamics) == 0 {
			return n
		}

		// Emit the creation of the map (with all its static entries).
		m := o.newTemp(n.Type(), false)
		as := ir.NewAssignStmt(base.Pos, m, n)
		typecheck.Stmt(as)
		o.stmt(as)

		// Emit eval+insert of dynamic entries, one at a time.
		for _, r := range dynamics {
			lhs := typecheck.AssignExpr(ir.NewIndexExpr(base.Pos, m, r.Key)).(*ir.IndexExpr)
			base.AssertfAt(lhs.Op() == ir.OINDEXMAP, lhs.Pos(), "want OINDEXMAP, have %+v", lhs)
			lhs.RType = n.RType

			as := ir.NewAssignStmt(base.Pos, lhs, r.Value)
			typecheck.Stmt(as)
			o.stmt(as)
		}

		// Remember that we issued these assignments so we can include that count
		// in the map alloc hint.
		// We're assuming here that all the keys in the map literal are distinct.
		// If any are equal, this will be an overcount. Probably not worth accounting
		// for that, as equal keys in map literals are rare, and at worst we waste
		// a bit of space.
		n.Len += int64(len(dynamics))

		return m
	}

	// No return - type-assertions above. Each case must return for itself.
}

// as2func orders OAS2FUNC nodes. It creates temporaries to ensure left-to-right assignment.
// The caller should order the right-hand side of the assignment before calling order.as2func.
// It rewrites,
//
//	a, b, a = ...
//
// as
//
//	tmp1, tmp2, tmp3 = ...
//	a, b, a = tmp1, tmp2, tmp3
//
// This is necessary to ensure left to right assignment order.
func (o *orderState) as2func(n *ir.AssignListStmt) {
	results := n.Rhs[0].Type()
	as := ir.NewAssignListStmt(n.Pos(), ir.OAS2, nil, nil)
	for i, nl := range n.Lhs {
		if !ir.IsBlank(nl) {
			typ := results.Field(i).Type
			tmp := o.newTemp(typ, typ.HasPointers())
			n.Lhs[i] = tmp
			as.Lhs = append(as.Lhs, nl)
			as.Rhs = append(as.Rhs, tmp)
		}
	}

	o.out = append(o.out, n)
	o.stmt(typecheck.Stmt(as))
}

// as2ok orders OAS2XXX with ok.
// Just like as2func, this also adds temporaries to ensure left-to-right assignment.
func (o *orderState) as2ok(n *ir.AssignListStmt) {
	as := ir.NewAssignListStmt(n.Pos(), ir.OAS2, nil, nil)

	do := func(i int, typ *types.Type) {
		if nl := n.Lhs[i]; !ir.IsBlank(nl) {
			var tmp ir.Node = o.newTemp(typ, typ.HasPointers())
			n.Lhs[i] = tmp
			as.Lhs = append(as.Lhs, nl)
			if i == 1 {
				// The "ok" result is an untyped boolean according to the Go
				// spec. We need to explicitly convert it to the LHS type in
				// case the latter is a defined boolean type (#8475).
				tmp = typecheck.Conv(tmp, nl.Type())
			}
			as.Rhs = append(as.Rhs, tmp)
		}
	}

	do(0, n.Rhs[0].Type())
	do(1, types.Types[types.TBOOL])

	o.out = append(o.out, n)
	o.stmt(typecheck.Stmt(as))
}
