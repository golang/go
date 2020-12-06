// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
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

// TODO(rsc): The temporary introduction during multiple assignments
// should be moved into this file, so that the temporaries can be cleaned
// and so that conversions implicit in the OAS2FUNC and OAS2RECV
// nodes can be made explicit and then have their temporaries cleaned.

// TODO(rsc): Goto and multilevel break/continue can jump over
// inserted VARKILL annotations. Work out a way to handle these.
// The current implementation is safe, in that it will execute correctly.
// But it won't reuse temporaries as aggressively as it might, and
// it can result in unnecessary zeroing of those variables in the function
// prologue.

// Order holds state during the ordering process.
type Order struct {
	out  []ir.Node             // list of generated statements
	temp []*ir.Name            // stack of temporary variables
	free map[string][]*ir.Name // free list of unused temporaries, by type.LongString().
	edit func(ir.Node) ir.Node // cached closure of o.exprNoLHS
}

// Order rewrites fn.Nbody to apply the ordering constraints
// described in the comment at the top of the file.
func order(fn *ir.Func) {
	if base.Flag.W > 1 {
		s := fmt.Sprintf("\nbefore order %v", fn.Sym())
		ir.DumpList(s, fn.Body())
	}

	orderBlock(fn.PtrBody(), map[string][]*ir.Name{})
}

// append typechecks stmt and appends it to out.
func (o *Order) append(stmt ir.Node) {
	o.out = append(o.out, typecheck(stmt, ctxStmt))
}

// newTemp allocates a new temporary with the given type,
// pushes it onto the temp stack, and returns it.
// If clear is true, newTemp emits code to zero the temporary.
func (o *Order) newTemp(t *types.Type, clear bool) *ir.Name {
	var v *ir.Name
	// Note: LongString is close to the type equality we want,
	// but not exactly. We still need to double-check with types.Identical.
	key := t.LongString()
	a := o.free[key]
	for i, n := range a {
		if types.Identical(t, n.Type()) {
			v = a[i]
			a[i] = a[len(a)-1]
			a = a[:len(a)-1]
			o.free[key] = a
			break
		}
	}
	if v == nil {
		v = temp(t)
	}
	if clear {
		o.append(ir.Nod(ir.OAS, v, nil))
	}

	o.temp = append(o.temp, v)
	return v
}

// copyExpr behaves like newTemp but also emits
// code to initialize the temporary to the value n.
func (o *Order) copyExpr(n ir.Node) ir.Node {
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
func (o *Order) copyExprClear(n ir.Node) *ir.Name {
	return o.copyExpr1(n, true)
}

func (o *Order) copyExpr1(n ir.Node, clear bool) *ir.Name {
	t := n.Type()
	v := o.newTemp(t, clear)
	o.append(ir.Nod(ir.OAS, v, n))
	return v
}

// cheapExpr returns a cheap version of n.
// The definition of cheap is that n is a variable or constant.
// If not, cheapExpr allocates a new tmp, emits tmp = n,
// and then returns tmp.
func (o *Order) cheapExpr(n ir.Node) ir.Node {
	if n == nil {
		return nil
	}

	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n
	case ir.OLEN, ir.OCAP:
		l := o.cheapExpr(n.Left())
		if l == n.Left() {
			return n
		}
		a := ir.SepCopy(n).(*ir.UnaryExpr)
		a.SetLeft(l)
		return typecheck(a, ctxExpr)
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
func (o *Order) safeExpr(n ir.Node) ir.Node {
	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n

	case ir.OLEN, ir.OCAP:
		l := o.safeExpr(n.Left())
		if l == n.Left() {
			return n
		}
		a := ir.SepCopy(n).(*ir.UnaryExpr)
		a.SetLeft(l)
		return typecheck(a, ctxExpr)

	case ir.ODOT:
		l := o.safeExpr(n.Left())
		if l == n.Left() {
			return n
		}
		a := ir.SepCopy(n).(*ir.SelectorExpr)
		a.SetLeft(l)
		return typecheck(a, ctxExpr)

	case ir.ODOTPTR:
		l := o.cheapExpr(n.Left())
		if l == n.Left() {
			return n
		}
		a := ir.SepCopy(n).(*ir.SelectorExpr)
		a.SetLeft(l)
		return typecheck(a, ctxExpr)

	case ir.ODEREF:
		l := o.cheapExpr(n.Left())
		if l == n.Left() {
			return n
		}
		a := ir.SepCopy(n).(*ir.StarExpr)
		a.SetLeft(l)
		return typecheck(a, ctxExpr)

	case ir.OINDEX, ir.OINDEXMAP:
		var l ir.Node
		if n.Left().Type().IsArray() {
			l = o.safeExpr(n.Left())
		} else {
			l = o.cheapExpr(n.Left())
		}
		r := o.cheapExpr(n.Right())
		if l == n.Left() && r == n.Right() {
			return n
		}
		a := ir.SepCopy(n).(*ir.IndexExpr)
		a.SetLeft(l)
		a.SetRight(r)
		return typecheck(a, ctxExpr)

	default:
		base.Fatalf("order.safeExpr %v", n.Op())
		return nil // not reached
	}
}

// isaddrokay reports whether it is okay to pass n's address to runtime routines.
// Taking the address of a variable makes the liveness and optimization analyses
// lose track of where the variable's lifetime ends. To avoid hurting the analyses
// of ordinary stack variables, those are not 'isaddrokay'. Temporaries are okay,
// because we emit explicit VARKILL instructions marking the end of those
// temporaries' lifetimes.
func isaddrokay(n ir.Node) bool {
	return islvalue(n) && (n.Op() != ir.ONAME || n.(*ir.Name).Class() == ir.PEXTERN || ir.IsAutoTmp(n))
}

// addrTemp ensures that n is okay to pass by address to runtime routines.
// If the original argument n is not okay, addrTemp creates a tmp, emits
// tmp = n, and then returns tmp.
// The result of addrTemp MUST be assigned back to n, e.g.
// 	n.Left = o.addrTemp(n.Left)
func (o *Order) addrTemp(n ir.Node) ir.Node {
	if n.Op() == ir.OLITERAL || n.Op() == ir.ONIL {
		// TODO: expand this to all static composite literal nodes?
		n = defaultlit(n, nil)
		dowidth(n.Type())
		vstat := readonlystaticname(n.Type())
		var s InitSchedule
		s.staticassign(vstat, n)
		if s.out != nil {
			base.Fatalf("staticassign of const generated code: %+v", n)
		}
		vstat = typecheck(vstat, ctxExpr).(*ir.Name)
		return vstat
	}
	if isaddrokay(n) {
		return n
	}
	return o.copyExpr(n)
}

// mapKeyTemp prepares n to be a key in a map runtime call and returns n.
// It should only be used for map runtime calls which have *_fast* versions.
func (o *Order) mapKeyTemp(t *types.Type, n ir.Node) ir.Node {
	// Most map calls need to take the address of the key.
	// Exception: map*_fast* calls. See golang.org/issue/19015.
	if mapfast(t) == mapslow {
		return o.addrTemp(n)
	}
	return n
}

// mapKeyReplaceStrConv replaces OBYTES2STR by OBYTES2STRTMP
// in n to avoid string allocations for keys in map lookups.
// Returns a bool that signals if a modification was made.
//
// For:
//  x = m[string(k)]
//  x = m[T1{... Tn{..., string(k), ...}]
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
		n.SetOp(ir.OBYTES2STRTMP)
		replaced = true
	case ir.OSTRUCTLIT:
		for _, elem := range n.List().Slice() {
			elem := elem.(*ir.StructKeyExpr)
			if mapKeyReplaceStrConv(elem.Left()) {
				replaced = true
			}
		}
	case ir.OARRAYLIT:
		for _, elem := range n.List().Slice() {
			if elem.Op() == ir.OKEY {
				elem = elem.(*ir.KeyExpr).Right()
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
func (o *Order) markTemp() ordermarker {
	return ordermarker(len(o.temp))
}

// popTemp pops temporaries off the stack until reaching the mark,
// which must have been returned by markTemp.
func (o *Order) popTemp(mark ordermarker) {
	for _, n := range o.temp[mark:] {
		key := n.Type().LongString()
		o.free[key] = append(o.free[key], n)
	}
	o.temp = o.temp[:mark]
}

// cleanTempNoPop emits VARKILL instructions to *out
// for each temporary above the mark on the temporary stack.
// It does not pop the temporaries from the stack.
func (o *Order) cleanTempNoPop(mark ordermarker) []ir.Node {
	var out []ir.Node
	for i := len(o.temp) - 1; i >= int(mark); i-- {
		n := o.temp[i]
		out = append(out, typecheck(ir.Nod(ir.OVARKILL, n, nil), ctxStmt))
	}
	return out
}

// cleanTemp emits VARKILL instructions for each temporary above the
// mark on the temporary stack and removes them from the stack.
func (o *Order) cleanTemp(top ordermarker) {
	o.out = append(o.out, o.cleanTempNoPop(top)...)
	o.popTemp(top)
}

// stmtList orders each of the statements in the list.
func (o *Order) stmtList(l ir.Nodes) {
	s := l.Slice()
	for i := range s {
		orderMakeSliceCopy(s[i:])
		o.stmt(s[i])
	}
}

// orderMakeSliceCopy matches the pattern:
//  m = OMAKESLICE([]T, x); OCOPY(m, s)
// and rewrites it to:
//  m = OMAKESLICECOPY([]T, x, s); nil
func orderMakeSliceCopy(s []ir.Node) {
	if base.Flag.N != 0 || instrumenting {
		return
	}
	if len(s) < 2 || s[0] == nil || s[0].Op() != ir.OAS || s[1] == nil || s[1].Op() != ir.OCOPY {
		return
	}

	as := s[0].(*ir.AssignStmt)
	cp := s[1].(*ir.BinaryExpr)
	if as.Right() == nil || as.Right().Op() != ir.OMAKESLICE || ir.IsBlank(as.Left()) ||
		as.Left().Op() != ir.ONAME || cp.Left().Op() != ir.ONAME || cp.Right().Op() != ir.ONAME ||
		as.Left().Name() != cp.Left().Name() || cp.Left().Name() == cp.Right().Name() {
		// The line above this one is correct with the differing equality operators:
		// we want as.X and cp.X to be the same name,
		// but we want the initial data to be coming from a different name.
		return
	}

	mk := as.Right().(*ir.MakeExpr)
	if mk.Esc() == EscNone || mk.Left() == nil || mk.Right() != nil {
		return
	}
	mk.SetOp(ir.OMAKESLICECOPY)
	mk.SetRight(cp.Right())
	// Set bounded when m = OMAKESLICE([]T, len(s)); OCOPY(m, s)
	mk.SetBounded(mk.Left().Op() == ir.OLEN && samesafeexpr(mk.Left().(*ir.UnaryExpr).Left(), cp.Right()))
	as.SetRight(typecheck(mk, ctxExpr))
	s[1] = nil // remove separate copy call
}

// edge inserts coverage instrumentation for libfuzzer.
func (o *Order) edge() {
	if base.Debug.Libfuzzer == 0 {
		return
	}

	// Create a new uint8 counter to be allocated in section
	// __libfuzzer_extra_counters.
	counter := staticname(types.Types[types.TUINT8])
	counter.Name().SetLibfuzzerExtraCounter(true)

	// counter += 1
	incr := ir.NewAssignOpStmt(base.Pos, ir.OADD, counter, nodintconst(1))
	o.append(incr)
}

// orderBlock orders the block of statements in n into a new slice,
// and then replaces the old slice in n with the new slice.
// free is a map that can be used to obtain temporary variables by type.
func orderBlock(n *ir.Nodes, free map[string][]*ir.Name) {
	var order Order
	order.free = free
	mark := order.markTemp()
	order.edge()
	order.stmtList(*n)
	order.cleanTemp(mark)
	n.Set(order.out)
}

// exprInPlace orders the side effects in *np and
// leaves them as the init list of the final *np.
// The result of exprInPlace MUST be assigned back to n, e.g.
// 	n.Left = o.exprInPlace(n.Left)
func (o *Order) exprInPlace(n ir.Node) ir.Node {
	var order Order
	order.free = o.free
	n = order.expr(n, nil)
	n = initExpr(order.out, n)

	// insert new temporaries from order
	// at head of outer list.
	o.temp = append(o.temp, order.temp...)
	return n
}

// orderStmtInPlace orders the side effects of the single statement *np
// and replaces it with the resulting statement list.
// The result of orderStmtInPlace MUST be assigned back to n, e.g.
// 	n.Left = orderStmtInPlace(n.Left)
// free is a map that can be used to obtain temporary variables by type.
func orderStmtInPlace(n ir.Node, free map[string][]*ir.Name) ir.Node {
	var order Order
	order.free = free
	mark := order.markTemp()
	order.stmt(n)
	order.cleanTemp(mark)
	return liststmt(order.out)
}

// init moves n's init list to o.out.
func (o *Order) init(n ir.Node) {
	if ir.MayBeShared(n) {
		// For concurrency safety, don't mutate potentially shared nodes.
		// First, ensure that no work is required here.
		if n.Init().Len() > 0 {
			base.Fatalf("order.init shared node with ninit")
		}
		return
	}
	o.stmtList(n.Init())
	n.PtrInit().Set(nil)
}

// call orders the call expression n.
// n.Op is OCALLMETH/OCALLFUNC/OCALLINTER or a builtin like OCOPY.
func (o *Order) call(nn ir.Node) {
	if nn.Init().Len() > 0 {
		// Caller should have already called o.init(nn).
		base.Fatalf("%v with unexpected ninit", nn.Op())
	}

	// Builtin functions.
	if nn.Op() != ir.OCALLFUNC && nn.Op() != ir.OCALLMETH && nn.Op() != ir.OCALLINTER {
		switch n := nn.(type) {
		default:
			base.Fatalf("unexpected call: %+v", n)
		case *ir.UnaryExpr:
			n.SetLeft(o.expr(n.Left(), nil))
		case *ir.ConvExpr:
			n.SetLeft(o.expr(n.Left(), nil))
		case *ir.BinaryExpr:
			n.SetLeft(o.expr(n.Left(), nil))
			n.SetRight(o.expr(n.Right(), nil))
		case *ir.MakeExpr:
			n.SetLeft(o.expr(n.Left(), nil))
			n.SetRight(o.expr(n.Right(), nil))
		case *ir.CallExpr:
			o.exprList(n.List())
		}
		return
	}

	n := nn.(*ir.CallExpr)
	fixVariadicCall(n)
	n.SetLeft(o.expr(n.Left(), nil))
	o.exprList(n.List())

	if n.Op() == ir.OCALLINTER {
		return
	}
	keepAlive := func(arg ir.Node) {
		// If the argument is really a pointer being converted to uintptr,
		// arrange for the pointer to be kept alive until the call returns,
		// by copying it into a temp and marking that temp
		// still alive when we pop the temp stack.
		if arg.Op() == ir.OCONVNOP {
			if arg.Left().Type().IsUnsafePtr() {
				x := o.copyExpr(arg.Left())
				arg.SetLeft(x)
				x.Name().SetAddrtaken(true) // ensure SSA keeps the x variable
				n.PtrBody().Append(typecheck(ir.Nod(ir.OVARLIVE, x, nil), ctxStmt))
			}
		}
	}

	// Check for "unsafe-uintptr" tag provided by escape analysis.
	for i, param := range n.Left().Type().Params().FieldSlice() {
		if param.Note == unsafeUintptrTag || param.Note == uintptrEscapesTag {
			if arg := n.List().Index(i); arg.Op() == ir.OSLICELIT {
				for _, elt := range arg.List().Slice() {
					keepAlive(elt)
				}
			} else {
				keepAlive(arg)
			}
		}
	}
}

// mapAssign appends n to o.out, introducing temporaries
// to make sure that all map assignments have the form m[k] = x.
// (Note: expr has already been called on n, so we know k is addressable.)
//
// If n is the multiple assignment form ..., m[k], ... = ..., x, ..., the rewrite is
//	t1 = m
//	t2 = k
//	...., t3, ... = ..., x, ...
//	t1[t2] = t3
//
// The temporaries t1, t2 are needed in case the ... being assigned
// contain m or k. They are usually unnecessary, but in the unnecessary
// cases they are also typically registerizable, so not much harm done.
// And this only applies to the multiple-assignment form.
// We could do a more precise analysis if needed, like in walk.go.
func (o *Order) mapAssign(n ir.Node) {
	switch n.Op() {
	default:
		base.Fatalf("order.mapAssign %v", n.Op())

	case ir.OAS:
		if n.Left().Op() == ir.OINDEXMAP {
			n.SetRight(o.safeMapRHS(n.Right()))
		}
		o.out = append(o.out, n)
	case ir.OASOP:
		if n.Left().Op() == ir.OINDEXMAP {
			n.SetRight(o.safeMapRHS(n.Right()))
		}
		o.out = append(o.out, n)

	case ir.OAS2, ir.OAS2DOTTYPE, ir.OAS2MAPR, ir.OAS2FUNC:
		var post []ir.Node
		for i, m := range n.List().Slice() {
			switch {
			case m.Op() == ir.OINDEXMAP:
				m := m.(*ir.IndexExpr)
				if !ir.IsAutoTmp(m.Left()) {
					m.SetLeft(o.copyExpr(m.Left()))
				}
				if !ir.IsAutoTmp(m.Right()) {
					m.SetRight(o.copyExpr(m.Right()))
				}
				fallthrough
			case instrumenting && n.Op() == ir.OAS2FUNC && !ir.IsBlank(m):
				t := o.newTemp(m.Type(), false)
				n.List().SetIndex(i, t)
				a := ir.Nod(ir.OAS, m, t)
				post = append(post, typecheck(a, ctxStmt))
			}
		}

		o.out = append(o.out, n)
		o.out = append(o.out, post...)
	}
}

func (o *Order) safeMapRHS(r ir.Node) ir.Node {
	// Make sure we evaluate the RHS before starting the map insert.
	// We need to make sure the RHS won't panic.  See issue 22881.
	if r.Op() == ir.OAPPEND {
		s := r.List().Slice()[1:]
		for i, n := range s {
			s[i] = o.cheapExpr(n)
		}
		return r
	}
	return o.cheapExpr(r)
}

// stmt orders the statement n, appending to o.out.
// Temporaries created during the statement are cleaned
// up using VARKILL instructions as possible.
func (o *Order) stmt(n ir.Node) {
	if n == nil {
		return
	}

	lno := setlineno(n)
	o.init(n)

	switch n.Op() {
	default:
		base.Fatalf("order.stmt %v", n.Op())

	case ir.OVARKILL, ir.OVARLIVE, ir.OINLMARK:
		o.out = append(o.out, n)

	case ir.OAS:
		t := o.markTemp()
		n.SetLeft(o.expr(n.Left(), nil))
		n.SetRight(o.expr(n.Right(), n.Left()))
		o.mapAssign(n)
		o.cleanTemp(t)

	case ir.OASOP:
		t := o.markTemp()
		n.SetLeft(o.expr(n.Left(), nil))
		n.SetRight(o.expr(n.Right(), nil))

		if instrumenting || n.Left().Op() == ir.OINDEXMAP && (n.SubOp() == ir.ODIV || n.SubOp() == ir.OMOD) {
			// Rewrite m[k] op= r into m[k] = m[k] op r so
			// that we can ensure that if op panics
			// because r is zero, the panic happens before
			// the map assignment.
			// DeepCopy is a big hammer here, but safeExpr
			// makes sure there is nothing too deep being copied.
			l1 := o.safeExpr(n.Left())
			l2 := ir.DeepCopy(src.NoXPos, l1)
			if l2.Op() == ir.OINDEXMAP {
				l2.SetIndexMapLValue(false)
			}
			l2 = o.copyExpr(l2)
			r := o.expr(typecheck(ir.NewBinaryExpr(n.Pos(), n.SubOp(), l2, n.Right()), ctxExpr), nil)
			as := typecheck(ir.NodAt(n.Pos(), ir.OAS, l1, r), ctxStmt)
			o.mapAssign(as)
			o.cleanTemp(t)
			return
		}

		o.mapAssign(n)
		o.cleanTemp(t)

	case ir.OAS2:
		t := o.markTemp()
		o.exprList(n.List())
		o.exprList(n.Rlist())
		o.mapAssign(n)
		o.cleanTemp(t)

	// Special: avoid copy of func call n.Right
	case ir.OAS2FUNC:
		n := n.(*ir.AssignListStmt)
		t := o.markTemp()
		o.exprList(n.List())
		o.init(n.Rlist().First())
		o.call(n.Rlist().First())
		o.as2(n)
		o.cleanTemp(t)

	// Special: use temporary variables to hold result,
	// so that runtime can take address of temporary.
	// No temporary for blank assignment.
	//
	// OAS2MAPR: make sure key is addressable if needed,
	//           and make sure OINDEXMAP is not copied out.
	case ir.OAS2DOTTYPE, ir.OAS2RECV, ir.OAS2MAPR:
		n := n.(*ir.AssignListStmt)
		t := o.markTemp()
		o.exprList(n.List())

		switch r := n.Rlist().First(); r.Op() {
		case ir.ODOTTYPE2:
			r.SetLeft(o.expr(r.Left(), nil))
		case ir.ORECV:
			r.SetLeft(o.expr(r.Left(), nil))
		case ir.OINDEXMAP:
			r.SetLeft(o.expr(r.Left(), nil))
			r.SetRight(o.expr(r.Right(), nil))
			// See similar conversion for OINDEXMAP below.
			_ = mapKeyReplaceStrConv(r.Right())
			r.SetRight(o.mapKeyTemp(r.Left().Type(), r.Right()))
		default:
			base.Fatalf("order.stmt: %v", r.Op())
		}

		o.okAs2(n)
		o.cleanTemp(t)

	// Special: does not save n onto out.
	case ir.OBLOCK:
		o.stmtList(n.List())

	// Special: n->left is not an expression; save as is.
	case ir.OBREAK,
		ir.OCONTINUE,
		ir.ODCL,
		ir.ODCLCONST,
		ir.ODCLTYPE,
		ir.OFALL,
		ir.OGOTO,
		ir.OLABEL,
		ir.ORETJMP:
		o.out = append(o.out, n)

	// Special: handle call arguments.
	case ir.OCALLFUNC, ir.OCALLINTER, ir.OCALLMETH:
		t := o.markTemp()
		o.call(n)
		o.out = append(o.out, n)
		o.cleanTemp(t)

	case ir.OCLOSE, ir.ORECV:
		t := o.markTemp()
		n.SetLeft(o.expr(n.Left(), nil))
		o.out = append(o.out, n)
		o.cleanTemp(t)

	case ir.OCOPY:
		t := o.markTemp()
		n.SetLeft(o.expr(n.Left(), nil))
		n.SetRight(o.expr(n.Right(), nil))
		o.out = append(o.out, n)
		o.cleanTemp(t)

	case ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
		t := o.markTemp()
		o.exprList(n.List())
		o.out = append(o.out, n)
		o.cleanTemp(t)

	// Special: order arguments to inner call but not call itself.
	case ir.ODEFER, ir.OGO:
		t := o.markTemp()
		o.init(n.Left())
		o.call(n.Left())
		o.out = append(o.out, n)
		o.cleanTemp(t)

	case ir.ODELETE:
		t := o.markTemp()
		n.List().SetFirst(o.expr(n.List().First(), nil))
		n.List().SetSecond(o.expr(n.List().Second(), nil))
		n.List().SetSecond(o.mapKeyTemp(n.List().First().Type(), n.List().Second()))
		o.out = append(o.out, n)
		o.cleanTemp(t)

	// Clean temporaries from condition evaluation at
	// beginning of loop body and after for statement.
	case ir.OFOR:
		t := o.markTemp()
		n.SetLeft(o.exprInPlace(n.Left()))
		n.PtrBody().Prepend(o.cleanTempNoPop(t)...)
		orderBlock(n.PtrBody(), o.free)
		n.SetRight(orderStmtInPlace(n.Right(), o.free))
		o.out = append(o.out, n)
		o.cleanTemp(t)

	// Clean temporaries from condition at
	// beginning of both branches.
	case ir.OIF:
		t := o.markTemp()
		n.SetLeft(o.exprInPlace(n.Left()))
		n.PtrBody().Prepend(o.cleanTempNoPop(t)...)
		n.PtrRlist().Prepend(o.cleanTempNoPop(t)...)
		o.popTemp(t)
		orderBlock(n.PtrBody(), o.free)
		orderBlock(n.PtrRlist(), o.free)
		o.out = append(o.out, n)

	// Special: argument will be converted to interface using convT2E
	// so make sure it is an addressable temporary.
	case ir.OPANIC:
		t := o.markTemp()
		n.SetLeft(o.expr(n.Left(), nil))
		if !n.Left().Type().IsInterface() {
			n.SetLeft(o.addrTemp(n.Left()))
		}
		o.out = append(o.out, n)
		o.cleanTemp(t)

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
		if n.Right().Op() == ir.OSTR2BYTES {
			n.Right().(*ir.ConvExpr).SetOp(ir.OSTR2BYTESTMP)
		}

		t := o.markTemp()
		n.SetRight(o.expr(n.Right(), nil))

		orderBody := true
		switch n.Type().Kind() {
		default:
			base.Fatalf("order.stmt range %v", n.Type())

		case types.TARRAY, types.TSLICE:
			if n.List().Len() < 2 || ir.IsBlank(n.List().Second()) {
				// for i := range x will only use x once, to compute len(x).
				// No need to copy it.
				break
			}
			fallthrough

		case types.TCHAN, types.TSTRING:
			// chan, string, slice, array ranges use value multiple times.
			// make copy.
			r := n.Right()

			if r.Type().IsString() && r.Type() != types.Types[types.TSTRING] {
				r = ir.Nod(ir.OCONV, r, nil)
				r.SetType(types.Types[types.TSTRING])
				r = typecheck(r, ctxExpr)
			}

			n.SetRight(o.copyExpr(r))

		case types.TMAP:
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
			r := n.Right()
			n.SetRight(o.copyExpr(r))

			// prealloc[n] is the temp for the iterator.
			// hiter contains pointers and needs to be zeroed.
			prealloc[n] = o.newTemp(hiter(n.Type()), true)
		}
		o.exprListInPlace(n.List())
		if orderBody {
			orderBlock(n.PtrBody(), o.free)
		}
		o.out = append(o.out, n)
		o.cleanTemp(t)

	case ir.ORETURN:
		o.exprList(n.List())
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
		t := o.markTemp()
		for _, ncas := range n.List().Slice() {
			ncas := ncas.(*ir.CaseStmt)
			r := ncas.Left()
			setlineno(ncas)

			// Append any new body prologue to ninit.
			// The next loop will insert ninit into nbody.
			if ncas.Init().Len() != 0 {
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
				recv := r.Rlist().First().(*ir.UnaryExpr)
				recv.SetLeft(o.expr(recv.Left(), nil))
				if recv.Left().Op() != ir.ONAME {
					recv.SetLeft(o.copyExpr(recv.Left()))
				}
				r := r.(*ir.AssignListStmt)
				init := r.PtrInit().Slice()
				r.PtrInit().Set(nil)

				colas := r.Colas()
				do := func(i int, t *types.Type) {
					n := r.List().Index(i)
					if ir.IsBlank(n) {
						return
					}
					// If this is case x := <-ch or case x, y := <-ch, the case has
					// the ODCL nodes to declare x and y. We want to delay that
					// declaration (and possible allocation) until inside the case body.
					// Delete the ODCL nodes here and recreate them inside the body below.
					if colas {
						if len(init) > 0 && init[0].Op() == ir.ODCL && init[0].(*ir.Decl).Left() == n {
							init = init[1:]
						}
						dcl := ir.Nod(ir.ODCL, n, nil)
						dcl = typecheck(dcl, ctxStmt)
						ncas.PtrInit().Append(dcl)
					}
					tmp := o.newTemp(t, t.HasPointers())
					as := ir.Nod(ir.OAS, n, conv(tmp, n.Type()))
					as = typecheck(as, ctxStmt)
					ncas.PtrInit().Append(as)
					r.PtrList().SetIndex(i, tmp)
				}
				do(0, recv.Left().Type().Elem())
				do(1, types.Types[types.TBOOL])
				if len(init) != 0 {
					ir.DumpList("ninit", r.Init())
					base.Fatalf("ninit on select recv")
				}
				orderBlock(ncas.PtrInit(), o.free)

			case ir.OSEND:
				if r.Init().Len() != 0 {
					ir.DumpList("ninit", r.Init())
					base.Fatalf("ninit on select send")
				}

				// case c <- x
				// r->left is c, r->right is x, both are always evaluated.
				r.SetLeft(o.expr(r.Left(), nil))

				if !ir.IsAutoTmp(r.Left()) {
					r.SetLeft(o.copyExpr(r.Left()))
				}
				r.SetRight(o.expr(r.Right(), nil))
				if !ir.IsAutoTmp(r.Right()) {
					r.SetRight(o.copyExpr(r.Right()))
				}
			}
		}
		// Now that we have accumulated all the temporaries, clean them.
		// Also insert any ninit queued during the previous loop.
		// (The temporary cleaning must follow that ninit work.)
		for _, cas := range n.List().Slice() {
			cas := cas.(*ir.CaseStmt)
			orderBlock(cas.PtrBody(), o.free)
			cas.PtrBody().Prepend(o.cleanTempNoPop(t)...)

			// TODO(mdempsky): Is this actually necessary?
			// walkselect appears to walk Ninit.
			cas.PtrBody().Prepend(cas.Init().Slice()...)
			cas.PtrInit().Set(nil)
		}

		o.out = append(o.out, n)
		o.popTemp(t)

	// Special: value being sent is passed as a pointer; make it addressable.
	case ir.OSEND:
		t := o.markTemp()
		n.SetLeft(o.expr(n.Left(), nil))
		n.SetRight(o.expr(n.Right(), nil))
		if instrumenting {
			// Force copying to the stack so that (chan T)(nil) <- x
			// is still instrumented as a read of x.
			n.SetRight(o.copyExpr(n.Right()))
		} else {
			n.SetRight(o.addrTemp(n.Right()))
		}
		o.out = append(o.out, n)
		o.cleanTemp(t)

	// TODO(rsc): Clean temporaries more aggressively.
	// Note that because walkswitch will rewrite some of the
	// switch into a binary search, this is not as easy as it looks.
	// (If we ran that code here we could invoke order.stmt on
	// the if-else chain instead.)
	// For now just clean all the temporaries at the end.
	// In practice that's fine.
	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		if base.Debug.Libfuzzer != 0 && !hasDefaultCase(n) {
			// Add empty "default:" case for instrumentation.
			n.PtrList().Append(ir.Nod(ir.OCASE, nil, nil))
		}

		t := o.markTemp()
		n.SetLeft(o.expr(n.Left(), nil))
		for _, ncas := range n.List().Slice() {
			ncas := ncas.(*ir.CaseStmt)
			o.exprListInPlace(ncas.List())
			orderBlock(ncas.PtrBody(), o.free)
		}

		o.out = append(o.out, n)
		o.cleanTemp(t)
	}

	base.Pos = lno
}

func hasDefaultCase(n *ir.SwitchStmt) bool {
	for _, ncas := range n.List().Slice() {
		ncas := ncas.(*ir.CaseStmt)
		if ncas.List().Len() == 0 {
			return true
		}
	}
	return false
}

// exprList orders the expression list l into o.
func (o *Order) exprList(l ir.Nodes) {
	s := l.Slice()
	for i := range s {
		s[i] = o.expr(s[i], nil)
	}
}

// exprListInPlace orders the expression list l but saves
// the side effects on the individual expression ninit lists.
func (o *Order) exprListInPlace(l ir.Nodes) {
	s := l.Slice()
	for i := range s {
		s[i] = o.exprInPlace(s[i])
	}
}

// prealloc[x] records the allocation to use for x.
var prealloc = map[ir.Node]ir.Node{}

func (o *Order) exprNoLHS(n ir.Node) ir.Node {
	return o.expr(n, nil)
}

// expr orders a single expression, appending side
// effects to o.out as needed.
// If this is part of an assignment lhs = *np, lhs is given.
// Otherwise lhs == nil. (When lhs != nil it may be possible
// to avoid copying the result of the expression to a temporary.)
// The result of expr MUST be assigned back to n, e.g.
// 	n.Left = o.expr(n.Left, lhs)
func (o *Order) expr(n, lhs ir.Node) ir.Node {
	if n == nil {
		return n
	}
	lno := setlineno(n)
	n = o.expr1(n, lhs)
	base.Pos = lno
	return n
}

func (o *Order) expr1(n, lhs ir.Node) ir.Node {
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
		o.exprList(n.List())

		if n.List().Len() > 5 {
			t := types.NewArray(types.Types[types.TSTRING], int64(n.List().Len()))
			prealloc[n] = o.newTemp(t, false)
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
		for _, n1 := range n.List().Slice() {
			hasbyte = hasbyte || n1.Op() == ir.OBYTES2STR
			haslit = haslit || n1.Op() == ir.OLITERAL && len(ir.StringVal(n1)) != 0
		}

		if haslit && hasbyte {
			for _, n2 := range n.List().Slice() {
				if n2.Op() == ir.OBYTES2STR {
					n2.SetOp(ir.OBYTES2STRTMP)
				}
			}
		}
		return n

	case ir.OINDEXMAP:
		n.SetLeft(o.expr(n.Left(), nil))
		n.SetRight(o.expr(n.Right(), nil))
		needCopy := false

		if !n.IndexMapLValue() {
			// Enforce that any []byte slices we are not copying
			// can not be changed before the map index by forcing
			// the map index to happen immediately following the
			// conversions. See copyExpr a few lines below.
			needCopy = mapKeyReplaceStrConv(n.Right())

			if instrumenting {
				// Race detector needs the copy.
				needCopy = true
			}
		}

		// key must be addressable
		n.SetRight(o.mapKeyTemp(n.Left().Type(), n.Right()))
		if needCopy {
			return o.copyExpr(n)
		}
		return n

	// concrete type (not interface) argument might need an addressable
	// temporary to pass to the runtime conversion routine.
	case ir.OCONVIFACE:
		n.SetLeft(o.expr(n.Left(), nil))
		if n.Left().Type().IsInterface() {
			return n
		}
		if _, needsaddr := convFuncName(n.Left().Type(), n.Type()); needsaddr || isStaticCompositeLiteral(n.Left()) {
			// Need a temp if we need to pass the address to the conversion function.
			// We also process static composite literal node here, making a named static global
			// whose address we can put directly in an interface (see OCONVIFACE case in walk).
			n.SetLeft(o.addrTemp(n.Left()))
		}
		return n

	case ir.OCONVNOP:
		if n.Type().IsKind(types.TUNSAFEPTR) && n.Left().Type().IsKind(types.TUINTPTR) && (n.Left().Op() == ir.OCALLFUNC || n.Left().Op() == ir.OCALLINTER || n.Left().Op() == ir.OCALLMETH) {
			call := n.Left().(*ir.CallExpr)
			// When reordering unsafe.Pointer(f()) into a separate
			// statement, the conversion and function call must stay
			// together. See golang.org/issue/15329.
			o.init(call)
			o.call(call)
			if lhs == nil || lhs.Op() != ir.ONAME || instrumenting {
				return o.copyExpr(n)
			}
		} else {
			n.SetLeft(o.expr(n.Left(), nil))
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

		r := o.newTemp(n.Type(), false)

		// Evaluate left-hand side.
		lhs := o.expr(n.Left(), nil)
		o.out = append(o.out, typecheck(ir.Nod(ir.OAS, r, lhs), ctxStmt))

		// Evaluate right-hand side, save generated code.
		saveout := o.out
		o.out = nil
		t := o.markTemp()
		o.edge()
		rhs := o.expr(n.Right(), nil)
		o.out = append(o.out, typecheck(ir.Nod(ir.OAS, r, rhs), ctxStmt))
		o.cleanTemp(t)
		gen := o.out
		o.out = saveout

		// If left-hand side doesn't cause a short-circuit, issue right-hand side.
		nif := ir.Nod(ir.OIF, r, nil)
		if n.Op() == ir.OANDAND {
			nif.PtrBody().Set(gen)
		} else {
			nif.PtrRlist().Set(gen)
		}
		o.out = append(o.out, nif)
		return r

	case ir.OCALLFUNC,
		ir.OCALLINTER,
		ir.OCALLMETH,
		ir.OCAP,
		ir.OCOMPLEX,
		ir.OCOPY,
		ir.OIMAG,
		ir.OLEN,
		ir.OMAKECHAN,
		ir.OMAKEMAP,
		ir.OMAKESLICE,
		ir.OMAKESLICECOPY,
		ir.ONEW,
		ir.OREAL,
		ir.ORECOVER,
		ir.OSTR2BYTES,
		ir.OSTR2BYTESTMP,
		ir.OSTR2RUNES:

		if isRuneCount(n) {
			// len([]rune(s)) is rewritten to runtime.countrunes(s) later.
			conv := n.(*ir.UnaryExpr).Left().(*ir.ConvExpr)
			conv.SetLeft(o.expr(conv.Left(), nil))
		} else {
			o.call(n)
		}

		if lhs == nil || lhs.Op() != ir.ONAME || instrumenting {
			return o.copyExpr(n)
		}
		return n

	case ir.OAPPEND:
		// Check for append(x, make([]T, y)...) .
		if isAppendOfMake(n) {
			n.List().SetFirst(o.expr(n.List().First(), nil)) // order x
			mk := n.List().Second().(*ir.MakeExpr)
			mk.SetLeft(o.expr(mk.Left(), nil)) // order y
		} else {
			o.exprList(n.List())
		}

		if lhs == nil || lhs.Op() != ir.ONAME && !samesafeexpr(lhs, n.List().First()) {
			return o.copyExpr(n)
		}
		return n

	case ir.OSLICE, ir.OSLICEARR, ir.OSLICESTR, ir.OSLICE3, ir.OSLICE3ARR:
		n.SetLeft(o.expr(n.Left(), nil))
		low, high, max := n.SliceBounds()
		low = o.expr(low, nil)
		low = o.cheapExpr(low)
		high = o.expr(high, nil)
		high = o.cheapExpr(high)
		max = o.expr(max, nil)
		max = o.cheapExpr(max)
		n.SetSliceBounds(low, high, max)
		if lhs == nil || lhs.Op() != ir.ONAME && !samesafeexpr(lhs, n.Left()) {
			return o.copyExpr(n)
		}
		return n

	case ir.OCLOSURE:
		n := n.(*ir.ClosureExpr)
		if n.Transient() && len(n.Func().ClosureVars) > 0 {
			prealloc[n] = o.newTemp(closureType(n), false)
		}
		return n

	case ir.OCALLPART:
		n := n.(*ir.CallPartExpr)
		n.SetLeft(o.expr(n.Left(), nil))
		if n.Transient() {
			t := partialCallType(n)
			prealloc[n] = o.newTemp(t, false)
		}
		return n

	case ir.OSLICELIT:
		o.exprList(n.List())
		if n.Transient() {
			t := types.NewArray(n.Type().Elem(), ir.Int64Val(n.Right()))
			prealloc[n] = o.newTemp(t, false)
		}
		return n

	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n.SetLeft(o.expr(n.Left(), nil))
		if !isdirectiface(n.Type()) || instrumenting {
			return o.copyExprClear(n)
		}
		return n

	case ir.ORECV:
		n.SetLeft(o.expr(n.Left(), nil))
		return o.copyExprClear(n)

	case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		n.SetLeft(o.expr(n.Left(), nil))
		n.SetRight(o.expr(n.Right(), nil))

		t := n.Left().Type()
		switch {
		case t.IsString():
			// Mark string(byteSlice) arguments to reuse byteSlice backing
			// buffer during conversion. String comparison does not
			// memorize the strings for later use, so it is safe.
			if n.Left().Op() == ir.OBYTES2STR {
				n.Left().(*ir.ConvExpr).SetOp(ir.OBYTES2STRTMP)
			}
			if n.Right().Op() == ir.OBYTES2STR {
				n.Right().(*ir.ConvExpr).SetOp(ir.OBYTES2STRTMP)
			}

		case t.IsStruct() || t.IsArray():
			// for complex comparisons, we need both args to be
			// addressable so we can pass them to the runtime.
			n.SetLeft(o.addrTemp(n.Left()))
			n.SetRight(o.addrTemp(n.Right()))
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
		entries := n.List().Slice()
		statics := entries[:0]
		var dynamics []*ir.KeyExpr
		for _, r := range entries {
			r := r.(*ir.KeyExpr)

			if !isStaticCompositeLiteral(r.Left()) || !isStaticCompositeLiteral(r.Right()) {
				dynamics = append(dynamics, r)
				continue
			}

			// Recursively ordering some static entries can change them to dynamic;
			// e.g., OCONVIFACE nodes. See #31777.
			r = o.expr(r, nil).(*ir.KeyExpr)
			if !isStaticCompositeLiteral(r.Left()) || !isStaticCompositeLiteral(r.Right()) {
				dynamics = append(dynamics, r)
				continue
			}

			statics = append(statics, r)
		}
		n.PtrList().Set(statics)

		if len(dynamics) == 0 {
			return n
		}

		// Emit the creation of the map (with all its static entries).
		m := o.newTemp(n.Type(), false)
		as := ir.Nod(ir.OAS, m, n)
		typecheck(as, ctxStmt)
		o.stmt(as)

		// Emit eval+insert of dynamic entries, one at a time.
		for _, r := range dynamics {
			as := ir.Nod(ir.OAS, ir.Nod(ir.OINDEX, m, r.Left()), r.Right())
			typecheck(as, ctxStmt) // Note: this converts the OINDEX to an OINDEXMAP
			o.stmt(as)
		}
		return m
	}

	// No return - type-assertions above. Each case must return for itself.
}

// as2 orders OAS2XXXX nodes. It creates temporaries to ensure left-to-right assignment.
// The caller should order the right-hand side of the assignment before calling order.as2.
// It rewrites,
// 	a, b, a = ...
// as
//	tmp1, tmp2, tmp3 = ...
// 	a, b, a = tmp1, tmp2, tmp3
// This is necessary to ensure left to right assignment order.
func (o *Order) as2(n *ir.AssignListStmt) {
	tmplist := []ir.Node{}
	left := []ir.Node{}
	for ni, l := range n.List().Slice() {
		if !ir.IsBlank(l) {
			tmp := o.newTemp(l.Type(), l.Type().HasPointers())
			n.List().SetIndex(ni, tmp)
			tmplist = append(tmplist, tmp)
			left = append(left, l)
		}
	}

	o.out = append(o.out, n)

	as := ir.Nod(ir.OAS2, nil, nil)
	as.PtrList().Set(left)
	as.PtrRlist().Set(tmplist)
	o.stmt(typecheck(as, ctxStmt))
}

// okAs2 orders OAS2XXX with ok.
// Just like as2, this also adds temporaries to ensure left-to-right assignment.
func (o *Order) okAs2(n *ir.AssignListStmt) {
	var tmp1, tmp2 ir.Node
	if !ir.IsBlank(n.List().First()) {
		typ := n.Rlist().First().Type()
		tmp1 = o.newTemp(typ, typ.HasPointers())
	}

	if !ir.IsBlank(n.List().Second()) {
		tmp2 = o.newTemp(types.Types[types.TBOOL], false)
	}

	o.out = append(o.out, n)

	if tmp1 != nil {
		r := ir.Nod(ir.OAS, n.List().First(), tmp1)
		o.mapAssign(typecheck(r, ctxStmt))
		n.List().SetFirst(tmp1)
	}
	if tmp2 != nil {
		r := ir.Nod(ir.OAS, n.List().Second(), conv(tmp2, n.List().Second().Type()))
		o.mapAssign(typecheck(r, ctxStmt))
		n.List().SetSecond(tmp2)
	}
}
