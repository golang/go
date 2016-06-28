// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"strings"
)

// Rewrite tree to use separate statements to enforce
// order of evaluation. Makes walk easier, because it
// can (after this runs) reorder at will within an expression.
//
// Rewrite x op= y into x = x op y.
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
	out  []*Node // list of generated statements
	temp []*Node // stack of temporary variables
}

// Order rewrites fn->nbody to apply the ordering constraints
// described in the comment at the top of the file.
func order(fn *Node) {
	if Debug['W'] > 1 {
		s := fmt.Sprintf("\nbefore order %v", fn.Func.Nname.Sym)
		dumplist(s, fn.Nbody)
	}

	orderblockNodes(&fn.Nbody)
}

// Ordertemp allocates a new temporary with the given type,
// pushes it onto the temp stack, and returns it.
// If clear is true, ordertemp emits code to zero the temporary.
func ordertemp(t *Type, order *Order, clear bool) *Node {
	var_ := temp(t)
	if clear {
		a := Nod(OAS, var_, nil)
		a = typecheck(a, Etop)
		order.out = append(order.out, a)
	}

	order.temp = append(order.temp, var_)
	return var_
}

// Ordercopyexpr behaves like ordertemp but also emits
// code to initialize the temporary to the value n.
//
// The clear argument is provided for use when the evaluation
// of tmp = n turns into a function call that is passed a pointer
// to the temporary as the output space. If the call blocks before
// tmp has been written, the garbage collector will still treat the
// temporary as live, so we must zero it before entering that call.
// Today, this only happens for channel receive operations.
// (The other candidate would be map access, but map access
// returns a pointer to the result data instead of taking a pointer
// to be filled in.)
func ordercopyexpr(n *Node, t *Type, order *Order, clear int) *Node {
	var_ := ordertemp(t, order, clear != 0)
	a := Nod(OAS, var_, n)
	a = typecheck(a, Etop)
	order.out = append(order.out, a)
	return var_
}

// Ordercheapexpr returns a cheap version of n.
// The definition of cheap is that n is a variable or constant.
// If not, ordercheapexpr allocates a new tmp, emits tmp = n,
// and then returns tmp.
func ordercheapexpr(n *Node, order *Order) *Node {
	if n == nil {
		return nil
	}
	switch n.Op {
	case ONAME, OLITERAL:
		return n
	case OLEN, OCAP:
		l := ordercheapexpr(n.Left, order)
		if l == n.Left {
			return n
		}
		a := *n
		a.Orig = &a
		a.Left = l
		return typecheck(&a, Erv)
	}

	return ordercopyexpr(n, n.Type, order, 0)
}

// Ordersafeexpr returns a safe version of n.
// The definition of safe is that n can appear multiple times
// without violating the semantics of the original program,
// and that assigning to the safe version has the same effect
// as assigning to the original n.
//
// The intended use is to apply to x when rewriting x += y into x = x + y.
func ordersafeexpr(n *Node, order *Order) *Node {
	switch n.Op {
	case ONAME, OLITERAL:
		return n

	case ODOT, OLEN, OCAP:
		l := ordersafeexpr(n.Left, order)
		if l == n.Left {
			return n
		}
		a := *n
		a.Orig = &a
		a.Left = l
		return typecheck(&a, Erv)

	case ODOTPTR, OIND:
		l := ordercheapexpr(n.Left, order)
		if l == n.Left {
			return n
		}
		a := *n
		a.Orig = &a
		a.Left = l
		return typecheck(&a, Erv)

	case OINDEX, OINDEXMAP:
		var l *Node
		if n.Left.Type.IsArray() {
			l = ordersafeexpr(n.Left, order)
		} else {
			l = ordercheapexpr(n.Left, order)
		}
		r := ordercheapexpr(n.Right, order)
		if l == n.Left && r == n.Right {
			return n
		}
		a := *n
		a.Orig = &a
		a.Left = l
		a.Right = r
		return typecheck(&a, Erv)
	default:
		Fatalf("ordersafeexpr %v", n.Op)
		return nil // not reached
	}
}

// Istemp reports whether n is a temporary variable.
func istemp(n *Node) bool {
	if n.Op != ONAME {
		return false
	}
	return strings.HasPrefix(n.Sym.Name, "autotmp_")
}

// Isaddrokay reports whether it is okay to pass n's address to runtime routines.
// Taking the address of a variable makes the liveness and optimization analyses
// lose track of where the variable's lifetime ends. To avoid hurting the analyses
// of ordinary stack variables, those are not 'isaddrokay'. Temporaries are okay,
// because we emit explicit VARKILL instructions marking the end of those
// temporaries' lifetimes.
func isaddrokay(n *Node) bool {
	return islvalue(n) && (n.Op != ONAME || n.Class == PEXTERN || istemp(n))
}

// Orderaddrtemp ensures that *np is okay to pass by address to runtime routines.
// If the original argument *np is not okay, orderaddrtemp creates a tmp, emits
// tmp = *np, and then sets *np to the tmp variable.
func orderaddrtemp(n *Node, order *Order) *Node {
	if isaddrokay(n) {
		return n
	}
	return ordercopyexpr(n, n.Type, order, 0)
}

type ordermarker int

// Marktemp returns the top of the temporary variable stack.
func marktemp(order *Order) ordermarker {
	return ordermarker(len(order.temp))
}

// Poptemp pops temporaries off the stack until reaching the mark,
// which must have been returned by marktemp.
func poptemp(mark ordermarker, order *Order) {
	order.temp = order.temp[:mark]
}

// Cleantempnopop emits to *out VARKILL instructions for each temporary
// above the mark on the temporary stack, but it does not pop them
// from the stack.
func cleantempnopop(mark ordermarker, order *Order, out *[]*Node) {
	var kill *Node

	for i := len(order.temp) - 1; i >= int(mark); i-- {
		n := order.temp[i]
		if n.Name.Keepalive {
			n.Name.Keepalive = false
			n.Addrtaken = true // ensure SSA keeps the n variable
			kill = Nod(OVARLIVE, n, nil)
			kill = typecheck(kill, Etop)
			*out = append(*out, kill)
		}
		kill = Nod(OVARKILL, n, nil)
		kill = typecheck(kill, Etop)
		*out = append(*out, kill)
	}
}

// Cleantemp emits VARKILL instructions for each temporary above the
// mark on the temporary stack and removes them from the stack.
func cleantemp(top ordermarker, order *Order) {
	cleantempnopop(top, order, &order.out)
	poptemp(top, order)
}

// Orderstmtlist orders each of the statements in the list.
func orderstmtlist(l Nodes, order *Order) {
	for _, n := range l.Slice() {
		orderstmt(n, order)
	}
}

// Orderblock orders the block of statements l onto a new list,
// and returns the ordered list.
func orderblock(l Nodes) []*Node {
	var order Order
	mark := marktemp(&order)
	orderstmtlist(l, &order)
	cleantemp(mark, &order)
	return order.out
}

// OrderblockNodes orders the block of statements in n into a new slice,
// and then replaces the old slice in n with the new slice.
func orderblockNodes(n *Nodes) {
	var order Order
	mark := marktemp(&order)
	orderstmtlist(*n, &order)
	cleantemp(mark, &order)
	n.Set(order.out)
}

// Orderexprinplace orders the side effects in *np and
// leaves them as the init list of the final *np.
// The result of orderexprinplace MUST be assigned back to n, e.g.
// 	n.Left = orderexprinplace(n.Left, outer)
func orderexprinplace(n *Node, outer *Order) *Node {
	var order Order
	n = orderexpr(n, &order, nil)
	n = addinit(n, order.out)

	// insert new temporaries from order
	// at head of outer list.
	outer.temp = append(outer.temp, order.temp...)
	return n
}

// Orderstmtinplace orders the side effects of the single statement *np
// and replaces it with the resulting statement list.
// The result of orderstmtinplace MUST be assigned back to n, e.g.
// 	n.Left = orderstmtinplace(n.Left)
func orderstmtinplace(n *Node) *Node {
	var order Order
	mark := marktemp(&order)
	orderstmt(n, &order)
	cleantemp(mark, &order)
	return liststmt(order.out)
}

// Orderinit moves n's init list to order->out.
func orderinit(n *Node, order *Order) {
	orderstmtlist(n.Ninit, order)
	n.Ninit.Set(nil)
}

// Ismulticall reports whether the list l is f() for a multi-value function.
// Such an f() could appear as the lone argument to a multi-arg function.
func ismulticall(l Nodes) bool {
	// one arg only
	if l.Len() != 1 {
		return false
	}
	n := l.First()

	// must be call
	switch n.Op {
	default:
		return false

	case OCALLFUNC, OCALLMETH, OCALLINTER:
		break
	}

	// call must return multiple values
	return n.Left.Type.Results().NumFields() > 1
}

// Copyret emits t1, t2, ... = n, where n is a function call,
// and then returns the list t1, t2, ....
func copyret(n *Node, order *Order) []*Node {
	if !n.Type.IsFuncArgStruct() {
		Fatalf("copyret %v %d", n.Type, n.Left.Type.Results().NumFields())
	}

	var l1 []*Node
	var l2 []*Node
	for _, t := range n.Type.Fields().Slice() {
		tmp := temp(t.Type)
		l1 = append(l1, tmp)
		l2 = append(l2, tmp)
	}

	as := Nod(OAS2, nil, nil)
	as.List.Set(l1)
	as.Rlist.Set1(n)
	as = typecheck(as, Etop)
	orderstmt(as, order)

	return l2
}

// Ordercallargs orders the list of call arguments *l.
func ordercallargs(l *Nodes, order *Order) {
	if ismulticall(*l) {
		// return f() where f() is multiple values.
		l.Set(copyret(l.First(), order))
	} else {
		orderexprlist(*l, order)
	}
}

// Ordercall orders the call expression n.
// n->op is OCALLMETH/OCALLFUNC/OCALLINTER or a builtin like OCOPY.
func ordercall(n *Node, order *Order) {
	n.Left = orderexpr(n.Left, order, nil)
	n.Right = orderexpr(n.Right, order, nil) // ODDDARG temp
	ordercallargs(&n.List, order)

	if n.Op == OCALLFUNC {
		t, it := IterFields(n.Left.Type.Params())
		for i := range n.List.Slice() {
			// Check for "unsafe-uintptr" tag provided by escape analysis.
			// If present and the argument is really a pointer being converted
			// to uintptr, arrange for the pointer to be kept alive until the call
			// returns, by copying it into a temp and marking that temp
			// still alive when we pop the temp stack.
			if t == nil {
				break
			}
			if t.Note == unsafeUintptrTag || t.Note == uintptrEscapesTag {
				xp := n.List.Addr(i)
				for (*xp).Op == OCONVNOP && !(*xp).Type.IsPtr() {
					xp = &(*xp).Left
				}
				x := *xp
				if x.Type.IsPtr() {
					x = ordercopyexpr(x, x.Type, order, 0)
					x.Name.Keepalive = true
					*xp = x
				}
			}
			next := it.Next()
			if next == nil && t.Isddd && t.Note == uintptrEscapesTag {
				next = t
			}
			t = next
		}
	}
}

// Ordermapassign appends n to order->out, introducing temporaries
// to make sure that all map assignments have the form m[k] = x,
// where x is addressable.
// (Orderexpr has already been called on n, so we know k is addressable.)
//
// If n is m[k] = x where x is not addressable, the rewrite is:
//	tmp = x
//	m[k] = tmp
//
// If n is the multiple assignment form ..., m[k], ... = ..., the rewrite is
//	t1 = m
//	t2 = k
//	...., t3, ... = x
//	t1[t2] = t3
//
// The temporaries t1, t2 are needed in case the ... being assigned
// contain m or k. They are usually unnecessary, but in the unnecessary
// cases they are also typically registerizable, so not much harm done.
// And this only applies to the multiple-assignment form.
// We could do a more precise analysis if needed, like in walk.go.
//
// Ordermapassign also inserts these temporaries if needed for
// calling writebarrierfat with a pointer to n->right.
func ordermapassign(n *Node, order *Order) {
	switch n.Op {
	default:
		Fatalf("ordermapassign %v", n.Op)

	case OAS:
		order.out = append(order.out, n)

		// We call writebarrierfat only for values > 4 pointers long. See walk.go.
		// TODO(mdempsky): writebarrierfat doesn't exist anymore, but removing that
		// logic causes net/http's tests to become flaky; see CL 21242.
		if (n.Left.Op == OINDEXMAP || (needwritebarrier(n.Left, n.Right) && n.Left.Type.Width > int64(4*Widthptr))) && !isaddrokay(n.Right) {
			m := n.Left
			n.Left = ordertemp(m.Type, order, false)
			a := Nod(OAS, m, n.Left)
			a = typecheck(a, Etop)
			order.out = append(order.out, a)
		}

	case OAS2, OAS2DOTTYPE, OAS2MAPR, OAS2FUNC:
		var post []*Node
		var m *Node
		var a *Node
		for i1, n1 := range n.List.Slice() {
			if n1.Op == OINDEXMAP {
				m = n1
				if !istemp(m.Left) {
					m.Left = ordercopyexpr(m.Left, m.Left.Type, order, 0)
				}
				if !istemp(m.Right) {
					m.Right = ordercopyexpr(m.Right, m.Right.Type, order, 0)
				}
				n.List.SetIndex(i1, ordertemp(m.Type, order, false))
				a = Nod(OAS, m, n.List.Index(i1))
				a = typecheck(a, Etop)
				post = append(post, a)
			} else if instrumenting && n.Op == OAS2FUNC && !isblank(n.List.Index(i1)) {
				m = n.List.Index(i1)
				t := ordertemp(m.Type, order, false)
				n.List.SetIndex(i1, t)
				a = Nod(OAS, m, t)
				a = typecheck(a, Etop)
				post = append(post, a)
			}
		}

		order.out = append(order.out, n)
		order.out = append(order.out, post...)
	}
}

// Orderstmt orders the statement n, appending to order->out.
// Temporaries created during the statement are cleaned
// up using VARKILL instructions as possible.
func orderstmt(n *Node, order *Order) {
	if n == nil {
		return
	}

	lno := setlineno(n)

	orderinit(n, order)

	switch n.Op {
	default:
		Fatalf("orderstmt %v", n.Op)

	case OVARKILL, OVARLIVE:
		order.out = append(order.out, n)

	case OAS:
		t := marktemp(order)
		n.Left = orderexpr(n.Left, order, nil)
		n.Right = orderexpr(n.Right, order, n.Left)
		ordermapassign(n, order)
		cleantemp(t, order)

	case OAS2,
		OCLOSE,
		OCOPY,
		OPRINT,
		OPRINTN,
		ORECOVER,
		ORECV:
		t := marktemp(order)
		n.Left = orderexpr(n.Left, order, nil)
		n.Right = orderexpr(n.Right, order, nil)
		orderexprlist(n.List, order)
		orderexprlist(n.Rlist, order)
		switch n.Op {
		case OAS2, OAS2DOTTYPE:
			ordermapassign(n, order)
		default:
			order.out = append(order.out, n)
		}
		cleantemp(t, order)

	case OASOP:
		// Special: rewrite l op= r into l = l op r.
		// This simplifies quite a few operations;
		// most important is that it lets us separate
		// out map read from map write when l is
		// a map index expression.
		t := marktemp(order)

		n.Left = orderexpr(n.Left, order, nil)
		n.Left = ordersafeexpr(n.Left, order)
		tmp1 := treecopy(n.Left, 0)
		if tmp1.Op == OINDEXMAP {
			tmp1.Etype = 0 // now an rvalue not an lvalue
		}
		tmp1 = ordercopyexpr(tmp1, n.Left.Type, order, 0)
		// TODO(marvin): Fix Node.EType type union.
		n.Right = Nod(Op(n.Etype), tmp1, n.Right)
		n.Right = typecheck(n.Right, Erv)
		n.Right = orderexpr(n.Right, order, nil)
		n.Etype = 0
		n.Op = OAS
		ordermapassign(n, order)
		cleantemp(t, order)

		// Special: make sure key is addressable,
	// and make sure OINDEXMAP is not copied out.
	case OAS2MAPR:
		t := marktemp(order)

		orderexprlist(n.List, order)
		r := n.Rlist.First()
		r.Left = orderexpr(r.Left, order, nil)
		r.Right = orderexpr(r.Right, order, nil)

		// See case OINDEXMAP below.
		if r.Right.Op == OARRAYBYTESTR {
			r.Right.Op = OARRAYBYTESTRTMP
		}
		r.Right = orderaddrtemp(r.Right, order)
		ordermapassign(n, order)
		cleantemp(t, order)

		// Special: avoid copy of func call n->rlist->n.
	case OAS2FUNC:
		t := marktemp(order)

		orderexprlist(n.List, order)
		ordercall(n.Rlist.First(), order)
		ordermapassign(n, order)
		cleantemp(t, order)

		// Special: use temporary variables to hold result,
	// so that assertI2Tetc can take address of temporary.
	// No temporary for blank assignment.
	case OAS2DOTTYPE:
		t := marktemp(order)

		orderexprlist(n.List, order)
		n.Rlist.First().Left = orderexpr(n.Rlist.First().Left, order, nil) // i in i.(T)
		if isblank(n.List.First()) {
			order.out = append(order.out, n)
		} else {
			typ := n.Rlist.First().Type
			tmp1 := ordertemp(typ, order, haspointers(typ))
			order.out = append(order.out, n)
			r := Nod(OAS, n.List.First(), tmp1)
			r = typecheck(r, Etop)
			ordermapassign(r, order)
			n.List.Set([]*Node{tmp1, n.List.Second()})
		}

		cleantemp(t, order)

		// Special: use temporary variables to hold result,
	// so that chanrecv can take address of temporary.
	case OAS2RECV:
		t := marktemp(order)

		orderexprlist(n.List, order)
		n.Rlist.First().Left = orderexpr(n.Rlist.First().Left, order, nil) // arg to recv
		ch := n.Rlist.First().Left.Type
		tmp1 := ordertemp(ch.Elem(), order, haspointers(ch.Elem()))
		var tmp2 *Node
		if !isblank(n.List.Second()) {
			tmp2 = ordertemp(n.List.Second().Type, order, false)
		} else {
			tmp2 = ordertemp(Types[TBOOL], order, false)
		}
		order.out = append(order.out, n)
		r := Nod(OAS, n.List.First(), tmp1)
		r = typecheck(r, Etop)
		ordermapassign(r, order)
		r = Nod(OAS, n.List.Second(), tmp2)
		r = typecheck(r, Etop)
		ordermapassign(r, order)
		n.List.Set([]*Node{tmp1, tmp2})
		cleantemp(t, order)

		// Special: does not save n onto out.
	case OBLOCK, OEMPTY:
		orderstmtlist(n.List, order)

		// Special: n->left is not an expression; save as is.
	case OBREAK,
		OCONTINUE,
		ODCL,
		ODCLCONST,
		ODCLTYPE,
		OFALL,
		OXFALL,
		OGOTO,
		OLABEL,
		ORETJMP:
		order.out = append(order.out, n)

		// Special: handle call arguments.
	case OCALLFUNC, OCALLINTER, OCALLMETH:
		t := marktemp(order)

		ordercall(n, order)
		order.out = append(order.out, n)
		cleantemp(t, order)

		// Special: order arguments to inner call but not call itself.
	case ODEFER, OPROC:
		t := marktemp(order)

		switch n.Left.Op {
		// Delete will take the address of the key.
		// Copy key into new temp and do not clean it
		// (it persists beyond the statement).
		case ODELETE:
			orderexprlist(n.Left.List, order)

			t1 := marktemp(order)
			np := n.Left.List.Addr(1) // map key
			*np = ordercopyexpr(*np, (*np).Type, order, 0)
			poptemp(t1, order)

		default:
			ordercall(n.Left, order)
		}

		order.out = append(order.out, n)
		cleantemp(t, order)

	case ODELETE:
		t := marktemp(order)
		n.List.SetIndex(0, orderexpr(n.List.Index(0), order, nil))
		n.List.SetIndex(1, orderexpr(n.List.Index(1), order, nil))
		n.List.SetIndex(1, orderaddrtemp(n.List.Index(1), order)) // map key
		order.out = append(order.out, n)
		cleantemp(t, order)

		// Clean temporaries from condition evaluation at
	// beginning of loop body and after for statement.
	case OFOR:
		t := marktemp(order)

		n.Left = orderexprinplace(n.Left, order)
		var l []*Node
		cleantempnopop(t, order, &l)
		n.Nbody.Set(append(l, n.Nbody.Slice()...))
		orderblockNodes(&n.Nbody)
		n.Right = orderstmtinplace(n.Right)
		order.out = append(order.out, n)
		cleantemp(t, order)

		// Clean temporaries from condition at
	// beginning of both branches.
	case OIF:
		t := marktemp(order)

		n.Left = orderexprinplace(n.Left, order)
		var l []*Node
		cleantempnopop(t, order, &l)
		n.Nbody.Set(append(l, n.Nbody.Slice()...))
		l = nil
		cleantempnopop(t, order, &l)
		n.Rlist.Set(append(l, n.Rlist.Slice()...))
		poptemp(t, order)
		orderblockNodes(&n.Nbody)
		n.Rlist.Set(orderblock(n.Rlist))
		order.out = append(order.out, n)

		// Special: argument will be converted to interface using convT2E
	// so make sure it is an addressable temporary.
	case OPANIC:
		t := marktemp(order)

		n.Left = orderexpr(n.Left, order, nil)
		if !n.Left.Type.IsInterface() {
			n.Left = orderaddrtemp(n.Left, order)
		}
		order.out = append(order.out, n)
		cleantemp(t, order)

	case ORANGE:
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
		if n.Right.Op == OSTRARRAYBYTE {
			n.Right.Op = OSTRARRAYBYTETMP
		}

		t := marktemp(order)
		n.Right = orderexpr(n.Right, order, nil)
		switch n.Type.Etype {
		default:
			Fatalf("orderstmt range %v", n.Type)

		case TARRAY, TSLICE:
			if n.List.Len() < 2 || isblank(n.List.Second()) {
				// for i := range x will only use x once, to compute len(x).
				// No need to copy it.
				break
			}
			fallthrough

		case TCHAN, TSTRING:
			// chan, string, slice, array ranges use value multiple times.
			// make copy.
			r := n.Right

			if r.Type.IsString() && r.Type != Types[TSTRING] {
				r = Nod(OCONV, r, nil)
				r.Type = Types[TSTRING]
				r = typecheck(r, Erv)
			}

			n.Right = ordercopyexpr(r, r.Type, order, 0)

		case TMAP:
			// copy the map value in case it is a map literal.
			// TODO(rsc): Make tmp = literal expressions reuse tmp.
			// For maps tmp is just one word so it hardly matters.
			r := n.Right
			n.Right = ordercopyexpr(r, r.Type, order, 0)

			// n->alloc is the temp for the iterator.
			prealloc[n] = ordertemp(Types[TUINT8], order, true)
		}
		for i := range n.List.Slice() {
			n.List.SetIndex(i, orderexprinplace(n.List.Index(i), order))
		}
		orderblockNodes(&n.Nbody)
		order.out = append(order.out, n)
		cleantemp(t, order)

	case ORETURN:
		ordercallargs(&n.List, order)
		order.out = append(order.out, n)

	// Special: clean case temporaries in each block entry.
	// Select must enter one of its blocks, so there is no
	// need for a cleaning at the end.
	// Doubly special: evaluation order for select is stricter
	// than ordinary expressions. Even something like p.c
	// has to be hoisted into a temporary, so that it cannot be
	// reordered after the channel evaluation for a different
	// case (if p were nil, then the timing of the fault would
	// give this away).
	case OSELECT:
		t := marktemp(order)

		var tmp1 *Node
		var tmp2 *Node
		var r *Node
		for _, n2 := range n.List.Slice() {
			if n2.Op != OXCASE {
				Fatalf("order select case %v", n2.Op)
			}
			r = n2.Left
			setlineno(n2)

			// Append any new body prologue to ninit.
			// The next loop will insert ninit into nbody.
			if n2.Ninit.Len() != 0 {
				Fatalf("order select ninit")
			}
			if r != nil {
				switch r.Op {
				default:
					Yyerror("unknown op in select %v", r.Op)
					Dump("select case", r)

				// If this is case x := <-ch or case x, y := <-ch, the case has
				// the ODCL nodes to declare x and y. We want to delay that
				// declaration (and possible allocation) until inside the case body.
				// Delete the ODCL nodes here and recreate them inside the body below.
				case OSELRECV, OSELRECV2:
					if r.Colas {
						i := 0
						if r.Ninit.Len() != 0 && r.Ninit.First().Op == ODCL && r.Ninit.First().Left == r.Left {
							i++
						}
						if i < r.Ninit.Len() && r.Ninit.Index(i).Op == ODCL && r.List.Len() != 0 && r.Ninit.Index(i).Left == r.List.First() {
							i++
						}
						if i >= r.Ninit.Len() {
							r.Ninit.Set(nil)
						}
					}

					if r.Ninit.Len() != 0 {
						Yyerror("ninit on select recv")
						dumplist("ninit", r.Ninit)
					}

					// case x = <-c
					// case x, ok = <-c
					// r->left is x, r->ntest is ok, r->right is ORECV, r->right->left is c.
					// r->left == N means 'case <-c'.
					// c is always evaluated; x and ok are only evaluated when assigned.
					r.Right.Left = orderexpr(r.Right.Left, order, nil)

					if r.Right.Left.Op != ONAME {
						r.Right.Left = ordercopyexpr(r.Right.Left, r.Right.Left.Type, order, 0)
					}

					// Introduce temporary for receive and move actual copy into case body.
					// avoids problems with target being addressed, as usual.
					// NOTE: If we wanted to be clever, we could arrange for just one
					// temporary per distinct type, sharing the temp among all receives
					// with that temp. Similarly one ok bool could be shared among all
					// the x,ok receives. Not worth doing until there's a clear need.
					if r.Left != nil && isblank(r.Left) {
						r.Left = nil
					}
					if r.Left != nil {
						// use channel element type for temporary to avoid conversions,
						// such as in case interfacevalue = <-intchan.
						// the conversion happens in the OAS instead.
						tmp1 = r.Left

						if r.Colas {
							tmp2 = Nod(ODCL, tmp1, nil)
							tmp2 = typecheck(tmp2, Etop)
							n2.Ninit.Append(tmp2)
						}

						r.Left = ordertemp(r.Right.Left.Type.Elem(), order, haspointers(r.Right.Left.Type.Elem()))
						tmp2 = Nod(OAS, tmp1, r.Left)
						tmp2 = typecheck(tmp2, Etop)
						n2.Ninit.Append(tmp2)
					}

					if r.List.Len() != 0 && isblank(r.List.First()) {
						r.List.Set(nil)
					}
					if r.List.Len() != 0 {
						tmp1 = r.List.First()
						if r.Colas {
							tmp2 = Nod(ODCL, tmp1, nil)
							tmp2 = typecheck(tmp2, Etop)
							n2.Ninit.Append(tmp2)
						}

						r.List.Set1(ordertemp(tmp1.Type, order, false))
						tmp2 = Nod(OAS, tmp1, r.List.First())
						tmp2 = typecheck(tmp2, Etop)
						n2.Ninit.Append(tmp2)
					}
					n2.Ninit.Set(orderblock(n2.Ninit))

				case OSEND:
					if r.Ninit.Len() != 0 {
						Yyerror("ninit on select send")
						dumplist("ninit", r.Ninit)
					}

					// case c <- x
					// r->left is c, r->right is x, both are always evaluated.
					r.Left = orderexpr(r.Left, order, nil)

					if !istemp(r.Left) {
						r.Left = ordercopyexpr(r.Left, r.Left.Type, order, 0)
					}
					r.Right = orderexpr(r.Right, order, nil)
					if !istemp(r.Right) {
						r.Right = ordercopyexpr(r.Right, r.Right.Type, order, 0)
					}
				}
			}

			orderblockNodes(&n2.Nbody)
		}
		// Now that we have accumulated all the temporaries, clean them.
		// Also insert any ninit queued during the previous loop.
		// (The temporary cleaning must follow that ninit work.)
		for _, n3 := range n.List.Slice() {
			s := n3.Ninit.Slice()
			cleantempnopop(t, order, &s)
			n3.Nbody.Set(append(s, n3.Nbody.Slice()...))
			n3.Ninit.Set(nil)
		}

		order.out = append(order.out, n)
		poptemp(t, order)

		// Special: value being sent is passed as a pointer; make it addressable.
	case OSEND:
		t := marktemp(order)

		n.Left = orderexpr(n.Left, order, nil)
		n.Right = orderexpr(n.Right, order, nil)
		n.Right = orderaddrtemp(n.Right, order)
		order.out = append(order.out, n)
		cleantemp(t, order)

		// TODO(rsc): Clean temporaries more aggressively.
	// Note that because walkswitch will rewrite some of the
	// switch into a binary search, this is not as easy as it looks.
	// (If we ran that code here we could invoke orderstmt on
	// the if-else chain instead.)
	// For now just clean all the temporaries at the end.
	// In practice that's fine.
	case OSWITCH:
		t := marktemp(order)

		n.Left = orderexpr(n.Left, order, nil)
		for _, n4 := range n.List.Slice() {
			if n4.Op != OXCASE {
				Fatalf("order switch case %v", n4.Op)
			}
			orderexprlistinplace(n4.List, order)
			orderblockNodes(&n4.Nbody)
		}

		order.out = append(order.out, n)
		cleantemp(t, order)
	}

	lineno = lno
}

// Orderexprlist orders the expression list l into order.
func orderexprlist(l Nodes, order *Order) {
	s := l.Slice()
	for i := range s {
		s[i] = orderexpr(s[i], order, nil)
	}
}

// Orderexprlist orders the expression list l but saves
// the side effects on the individual expression ninit lists.
func orderexprlistinplace(l Nodes, order *Order) {
	s := l.Slice()
	for i := range s {
		s[i] = orderexprinplace(s[i], order)
	}
}

// prealloc[x] records the allocation to use for x.
var prealloc = map[*Node]*Node{}

// Orderexpr orders a single expression, appending side
// effects to order->out as needed.
// If this is part of an assignment lhs = *np, lhs is given.
// Otherwise lhs == nil. (When lhs != nil it may be possible
// to avoid copying the result of the expression to a temporary.)
// The result of orderexpr MUST be assigned back to n, e.g.
// 	n.Left = orderexpr(n.Left, order, lhs)
func orderexpr(n *Node, order *Order, lhs *Node) *Node {
	if n == nil {
		return n
	}

	lno := setlineno(n)
	orderinit(n, order)

	switch n.Op {
	default:
		n.Left = orderexpr(n.Left, order, nil)
		n.Right = orderexpr(n.Right, order, nil)
		orderexprlist(n.List, order)
		orderexprlist(n.Rlist, order)

		// Addition of strings turns into a function call.
	// Allocate a temporary to hold the strings.
	// Fewer than 5 strings use direct runtime helpers.
	case OADDSTR:
		orderexprlist(n.List, order)

		if n.List.Len() > 5 {
			t := typArray(Types[TSTRING], int64(n.List.Len()))
			prealloc[n] = ordertemp(t, order, false)
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
		for _, n1 := range n.List.Slice() {
			hasbyte = hasbyte || n1.Op == OARRAYBYTESTR
			haslit = haslit || n1.Op == OLITERAL && len(n1.Val().U.(string)) != 0
		}

		if haslit && hasbyte {
			for _, n2 := range n.List.Slice() {
				if n2.Op == OARRAYBYTESTR {
					n2.Op = OARRAYBYTESTRTMP
				}
			}
		}

	case OCMPSTR:
		n.Left = orderexpr(n.Left, order, nil)
		n.Right = orderexpr(n.Right, order, nil)

		// Mark string(byteSlice) arguments to reuse byteSlice backing
		// buffer during conversion. String comparison does not
		// memorize the strings for later use, so it is safe.
		if n.Left.Op == OARRAYBYTESTR {
			n.Left.Op = OARRAYBYTESTRTMP
		}
		if n.Right.Op == OARRAYBYTESTR {
			n.Right.Op = OARRAYBYTESTRTMP
		}

		// key must be addressable
	case OINDEXMAP:
		n.Left = orderexpr(n.Left, order, nil)

		n.Right = orderexpr(n.Right, order, nil)

		// For x = m[string(k)] where k is []byte, the allocation of
		// backing bytes for the string can be avoided by reusing
		// the []byte backing array. This is a special case that it
		// would be nice to handle more generally, but because
		// there are no []byte-keyed maps, this specific case comes
		// up in important cases in practice. See issue 3512.
		// Nothing can change the []byte we are not copying before
		// the map index, because the map access is going to
		// be forced to happen immediately following this
		// conversion (by the ordercopyexpr a few lines below).
		if n.Etype == 0 && n.Right.Op == OARRAYBYTESTR {
			n.Right.Op = OARRAYBYTESTRTMP
		}

		n.Right = orderaddrtemp(n.Right, order)
		if n.Etype == 0 {
			// use of value (not being assigned);
			// make copy in temporary.
			n = ordercopyexpr(n, n.Type, order, 0)
		}

		// concrete type (not interface) argument must be addressable
	// temporary to pass to runtime.
	case OCONVIFACE:
		n.Left = orderexpr(n.Left, order, nil)

		if !n.Left.Type.IsInterface() {
			n.Left = orderaddrtemp(n.Left, order)
		}

	case OCONVNOP:
		if n.Type.IsKind(TUNSAFEPTR) && n.Left.Type.IsKind(TUINTPTR) && (n.Left.Op == OCALLFUNC || n.Left.Op == OCALLINTER || n.Left.Op == OCALLMETH) {
			// When reordering unsafe.Pointer(f()) into a separate
			// statement, the conversion and function call must stay
			// together. See golang.org/issue/15329.
			orderinit(n.Left, order)
			ordercall(n.Left, order)
			if lhs == nil || lhs.Op != ONAME || instrumenting {
				n = ordercopyexpr(n, n.Type, order, 0)
			}
		} else {
			n.Left = orderexpr(n.Left, order, nil)
		}

	case OANDAND, OOROR:
		mark := marktemp(order)
		n.Left = orderexpr(n.Left, order, nil)

		// Clean temporaries from first branch at beginning of second.
		// Leave them on the stack so that they can be killed in the outer
		// context in case the short circuit is taken.
		var s []*Node

		cleantempnopop(mark, order, &s)
		n.Right.Ninit.Set(append(s, n.Right.Ninit.Slice()...))
		n.Right = orderexprinplace(n.Right, order)

	case OCALLFUNC,
		OCALLINTER,
		OCALLMETH,
		OCAP,
		OCOMPLEX,
		OCOPY,
		OIMAG,
		OLEN,
		OMAKECHAN,
		OMAKEMAP,
		OMAKESLICE,
		ONEW,
		OREAL,
		ORECOVER,
		OSTRARRAYBYTE,
		OSTRARRAYBYTETMP,
		OSTRARRAYRUNE:
		ordercall(n, order)
		if lhs == nil || lhs.Op != ONAME || instrumenting {
			n = ordercopyexpr(n, n.Type, order, 0)
		}

	case OAPPEND:
		ordercallargs(&n.List, order)
		if lhs == nil || lhs.Op != ONAME && !samesafeexpr(lhs, n.List.First()) {
			n = ordercopyexpr(n, n.Type, order, 0)
		}

	case OSLICE, OSLICEARR, OSLICESTR, OSLICE3, OSLICE3ARR:
		n.Left = orderexpr(n.Left, order, nil)
		low, high, max := n.SliceBounds()
		low = orderexpr(low, order, nil)
		low = ordercheapexpr(low, order)
		high = orderexpr(high, order, nil)
		high = ordercheapexpr(high, order)
		max = orderexpr(max, order, nil)
		max = ordercheapexpr(max, order)
		n.SetSliceBounds(low, high, max)
		if lhs == nil || lhs.Op != ONAME && !samesafeexpr(lhs, n.Left) {
			n = ordercopyexpr(n, n.Type, order, 0)
		}

	case OCLOSURE:
		if n.Noescape && n.Func.Cvars.Len() > 0 {
			prealloc[n] = ordertemp(Types[TUINT8], order, false) // walk will fill in correct type
		}

	case OARRAYLIT, OCALLPART:
		n.Left = orderexpr(n.Left, order, nil)
		n.Right = orderexpr(n.Right, order, nil)
		orderexprlist(n.List, order)
		orderexprlist(n.Rlist, order)
		if n.Noescape {
			prealloc[n] = ordertemp(Types[TUINT8], order, false) // walk will fill in correct type
		}

	case ODDDARG:
		if n.Noescape {
			// The ddd argument does not live beyond the call it is created for.
			// Allocate a temporary that will be cleaned up when this statement
			// completes. We could be more aggressive and try to arrange for it
			// to be cleaned up when the call completes.
			prealloc[n] = ordertemp(n.Type.Elem(), order, false)
		}

	case ODOTTYPE, ODOTTYPE2:
		n.Left = orderexpr(n.Left, order, nil)
		// TODO(rsc): The Isfat is for consistency with componentgen and walkexpr.
		// It needs to be removed in all three places.
		// That would allow inlining x.(struct{*int}) the same as x.(*int).
		if !isdirectiface(n.Type) || Isfat(n.Type) || instrumenting {
			n = ordercopyexpr(n, n.Type, order, 1)
		}

	case ORECV:
		n.Left = orderexpr(n.Left, order, nil)
		n = ordercopyexpr(n, n.Type, order, 1)

	case OEQ, ONE:
		n.Left = orderexpr(n.Left, order, nil)
		n.Right = orderexpr(n.Right, order, nil)
		t := n.Left.Type
		if t.IsStruct() || t.IsArray() {
			// for complex comparisons, we need both args to be
			// addressable so we can pass them to the runtime.
			n.Left = orderaddrtemp(n.Left, order)
			n.Right = orderaddrtemp(n.Right, order)
		}
	}

	lineno = lno
	return n
}
