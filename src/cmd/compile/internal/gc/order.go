// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"strings"
)

// Rewrite tree to use separate statements to enforce
// order of evaluation.  Makes walk easier, because it
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
	out  *NodeList // list of generated statements
	temp *NodeList // head of stack of temporary variables
	free *NodeList // free list of NodeList* structs (for use in temp)
}

// Order rewrites fn->nbody to apply the ordering constraints
// described in the comment at the top of the file.
func order(fn *Node) {
	if Debug['W'] > 1 {
		s := fmt.Sprintf("\nbefore order %v", fn.Func.Nname.Sym)
		dumplist(s, fn.Nbody)
	}

	orderblock(&fn.Nbody)
}

// Ordertemp allocates a new temporary with the given type,
// pushes it onto the temp stack, and returns it.
// If clear is true, ordertemp emits code to zero the temporary.
func ordertemp(t *Type, order *Order, clear bool) *Node {
	var_ := temp(t)
	if clear {
		a := Nod(OAS, var_, nil)
		typecheck(&a, Etop)
		order.out = list(order.out, a)
	}

	l := order.free
	if l == nil {
		l = new(NodeList)
	}
	order.free = l.Next
	l.Next = order.temp
	l.N = var_
	order.temp = l
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
	typecheck(&a, Etop)
	order.out = list(order.out, a)
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
		a := Nod(OXXX, nil, nil)
		*a = *n
		a.Orig = a
		a.Left = l
		typecheck(&a, Erv)
		return a
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
		a := Nod(OXXX, nil, nil)
		*a = *n
		a.Orig = a
		a.Left = l
		typecheck(&a, Erv)
		return a

	case ODOTPTR, OIND:
		l := ordercheapexpr(n.Left, order)
		if l == n.Left {
			return n
		}
		a := Nod(OXXX, nil, nil)
		*a = *n
		a.Orig = a
		a.Left = l
		typecheck(&a, Erv)
		return a

	case OINDEX, OINDEXMAP:
		var l *Node
		if Isfixedarray(n.Left.Type) {
			l = ordersafeexpr(n.Left, order)
		} else {
			l = ordercheapexpr(n.Left, order)
		}
		r := ordercheapexpr(n.Right, order)
		if l == n.Left && r == n.Right {
			return n
		}
		a := Nod(OXXX, nil, nil)
		*a = *n
		a.Orig = a
		a.Left = l
		a.Right = r
		typecheck(&a, Erv)
		return a
	}

	Fatalf("ordersafeexpr %v", Oconv(int(n.Op), 0))
	return nil // not reached
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
func orderaddrtemp(np **Node, order *Order) {
	n := *np
	if isaddrokay(n) {
		return
	}
	*np = ordercopyexpr(n, n.Type, order, 0)
}

// Marktemp returns the top of the temporary variable stack.
func marktemp(order *Order) *NodeList {
	return order.temp
}

// Poptemp pops temporaries off the stack until reaching the mark,
// which must have been returned by marktemp.
func poptemp(mark *NodeList, order *Order) {
	var l *NodeList

	for {
		l = order.temp
		if l == mark {
			break
		}
		order.temp = l.Next
		l.Next = order.free
		order.free = l
	}
}

// Cleantempnopop emits to *out VARKILL instructions for each temporary
// above the mark on the temporary stack, but it does not pop them
// from the stack.
func cleantempnopop(mark *NodeList, order *Order, out **NodeList) {
	var kill *Node

	for l := order.temp; l != mark; l = l.Next {
		if l.N.Name.Keepalive {
			l.N.Name.Keepalive = false
			kill = Nod(OVARLIVE, l.N, nil)
			typecheck(&kill, Etop)
			*out = list(*out, kill)
		}
		kill = Nod(OVARKILL, l.N, nil)
		typecheck(&kill, Etop)
		*out = list(*out, kill)
	}
}

// Cleantemp emits VARKILL instructions for each temporary above the
// mark on the temporary stack and removes them from the stack.
func cleantemp(top *NodeList, order *Order) {
	cleantempnopop(top, order, &order.out)
	poptemp(top, order)
}

// Orderstmtlist orders each of the statements in the list.
func orderstmtlist(l *NodeList, order *Order) {
	for ; l != nil; l = l.Next {
		orderstmt(l.N, order)
	}
}

// Orderblock orders the block of statements *l onto a new list,
// and then replaces *l with that list.
func orderblock(l **NodeList) {
	var order Order
	mark := marktemp(&order)
	orderstmtlist(*l, &order)
	cleantemp(mark, &order)
	*l = order.out
}

// Orderexprinplace orders the side effects in *np and
// leaves them as the init list of the final *np.
func orderexprinplace(np **Node, outer *Order) {
	n := *np
	var order Order
	orderexpr(&n, &order, nil)
	addinit(&n, order.out)

	// insert new temporaries from order
	// at head of outer list.
	lp := &order.temp

	for *lp != nil {
		lp = &(*lp).Next
	}
	*lp = outer.temp
	outer.temp = order.temp

	*np = n
}

// Orderstmtinplace orders the side effects of the single statement *np
// and replaces it with the resulting statement list.
func orderstmtinplace(np **Node) {
	n := *np
	var order Order
	mark := marktemp(&order)
	orderstmt(n, &order)
	cleantemp(mark, &order)
	*np = liststmt(order.out)
}

// Orderinit moves n's init list to order->out.
func orderinit(n *Node, order *Order) {
	orderstmtlist(n.Ninit, order)
	n.Ninit = nil
}

// Ismulticall reports whether the list l is f() for a multi-value function.
// Such an f() could appear as the lone argument to a multi-arg function.
func ismulticall(l *NodeList) bool {
	// one arg only
	if l == nil || l.Next != nil {
		return false
	}
	n := l.N

	// must be call
	switch n.Op {
	default:
		return false

	case OCALLFUNC, OCALLMETH, OCALLINTER:
		break
	}

	// call must return multiple values
	return n.Left.Type.Outtuple > 1
}

// Copyret emits t1, t2, ... = n, where n is a function call,
// and then returns the list t1, t2, ....
func copyret(n *Node, order *Order) *NodeList {
	if n.Type.Etype != TSTRUCT || !n.Type.Funarg {
		Fatalf("copyret %v %d", n.Type, n.Left.Type.Outtuple)
	}

	var l1 *NodeList
	var l2 *NodeList
	var tl Iter
	var tmp *Node
	for t := Structfirst(&tl, &n.Type); t != nil; t = structnext(&tl) {
		tmp = temp(t.Type)
		l1 = list(l1, tmp)
		l2 = list(l2, tmp)
	}

	as := Nod(OAS2, nil, nil)
	as.List = l1
	as.Rlist = list1(n)
	typecheck(&as, Etop)
	orderstmt(as, order)

	return l2
}

// Ordercallargs orders the list of call arguments *l.
func ordercallargs(l **NodeList, order *Order) {
	if ismulticall(*l) {
		// return f() where f() is multiple values.
		*l = copyret((*l).N, order)
	} else {
		orderexprlist(*l, order)
	}
}

// Ordercall orders the call expression n.
// n->op is OCALLMETH/OCALLFUNC/OCALLINTER or a builtin like OCOPY.
func ordercall(n *Node, order *Order) {
	orderexpr(&n.Left, order, nil)
	orderexpr(&n.Right, order, nil) // ODDDARG temp
	ordercallargs(&n.List, order)

	if n.Op == OCALLFUNC {
		for l, t := n.List, getinargx(n.Left.Type).Type; l != nil && t != nil; l, t = l.Next, t.Down {
			// Check for "unsafe-uintptr" tag provided by escape analysis.
			// If present and the argument is really a pointer being converted
			// to uintptr, arrange for the pointer to be kept alive until the call
			// returns, by copying it into a temp and marking that temp
			// still alive when we pop the temp stack.
			if t.Note != nil && *t.Note == unsafeUintptrTag {
				xp := &l.N
				for (*xp).Op == OCONVNOP && !Isptr[(*xp).Type.Etype] {
					xp = &(*xp).Left
				}
				x := *xp
				if Isptr[x.Type.Etype] {
					x = ordercopyexpr(x, x.Type, order, 0)
					x.Name.Keepalive = true
					*xp = x
				}
			}
		}
	}
}

// Ordermapassign appends n to order->out, introducing temporaries
// to make sure that all map assignments have the form m[k] = x,
// where x is adressable.
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
		Fatalf("ordermapassign %v", Oconv(int(n.Op), 0))

	case OAS:
		order.out = list(order.out, n)

		// We call writebarrierfat only for values > 4 pointers long. See walk.go.
		if (n.Left.Op == OINDEXMAP || (needwritebarrier(n.Left, n.Right) && n.Left.Type.Width > int64(4*Widthptr))) && !isaddrokay(n.Right) {
			m := n.Left
			n.Left = ordertemp(m.Type, order, false)
			a := Nod(OAS, m, n.Left)
			typecheck(&a, Etop)
			order.out = list(order.out, a)
		}

	case OAS2, OAS2DOTTYPE, OAS2MAPR, OAS2FUNC:
		var post *NodeList
		var m *Node
		var a *Node
		for l := n.List; l != nil; l = l.Next {
			if l.N.Op == OINDEXMAP {
				m = l.N
				if !istemp(m.Left) {
					m.Left = ordercopyexpr(m.Left, m.Left.Type, order, 0)
				}
				if !istemp(m.Right) {
					m.Right = ordercopyexpr(m.Right, m.Right.Type, order, 0)
				}
				l.N = ordertemp(m.Type, order, false)
				a = Nod(OAS, m, l.N)
				typecheck(&a, Etop)
				post = list(post, a)
			} else if instrumenting && n.Op == OAS2FUNC && !isblank(l.N) {
				m = l.N
				l.N = ordertemp(m.Type, order, false)
				a = Nod(OAS, m, l.N)
				typecheck(&a, Etop)
				post = list(post, a)
			}
		}

		order.out = list(order.out, n)
		order.out = concat(order.out, post)
	}
}

// Orderstmt orders the statement n, appending to order->out.
// Temporaries created during the statement are cleaned
// up using VARKILL instructions as possible.
func orderstmt(n *Node, order *Order) {
	if n == nil {
		return
	}

	lno := int(setlineno(n))

	orderinit(n, order)

	switch n.Op {
	default:
		Fatalf("orderstmt %v", Oconv(int(n.Op), 0))

	case OVARKILL, OVARLIVE:
		order.out = list(order.out, n)

	case OAS:
		t := marktemp(order)
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right, order, n.Left)
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
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right, order, nil)
		orderexprlist(n.List, order)
		orderexprlist(n.Rlist, order)
		switch n.Op {
		case OAS2, OAS2DOTTYPE:
			ordermapassign(n, order)
		default:
			order.out = list(order.out, n)
		}
		cleantemp(t, order)

	case OASOP:
		// Special: rewrite l op= r into l = l op r.
		// This simplifies quite a few operations;
		// most important is that it lets us separate
		// out map read from map write when l is
		// a map index expression.
		t := marktemp(order)

		orderexpr(&n.Left, order, nil)
		n.Left = ordersafeexpr(n.Left, order)
		tmp1 := treecopy(n.Left, 0)
		if tmp1.Op == OINDEXMAP {
			tmp1.Etype = 0 // now an rvalue not an lvalue
		}
		tmp1 = ordercopyexpr(tmp1, n.Left.Type, order, 0)
		// TODO(marvin): Fix Node.EType type union.
		n.Right = Nod(Op(n.Etype), tmp1, n.Right)
		typecheck(&n.Right, Erv)
		orderexpr(&n.Right, order, nil)
		n.Etype = 0
		n.Op = OAS
		ordermapassign(n, order)
		cleantemp(t, order)

		// Special: make sure key is addressable,
	// and make sure OINDEXMAP is not copied out.
	case OAS2MAPR:
		t := marktemp(order)

		orderexprlist(n.List, order)
		r := n.Rlist.N
		orderexpr(&r.Left, order, nil)
		orderexpr(&r.Right, order, nil)

		// See case OINDEXMAP below.
		if r.Right.Op == OARRAYBYTESTR {
			r.Right.Op = OARRAYBYTESTRTMP
		}
		orderaddrtemp(&r.Right, order)
		ordermapassign(n, order)
		cleantemp(t, order)

		// Special: avoid copy of func call n->rlist->n.
	case OAS2FUNC:
		t := marktemp(order)

		orderexprlist(n.List, order)
		ordercall(n.Rlist.N, order)
		ordermapassign(n, order)
		cleantemp(t, order)

		// Special: use temporary variables to hold result,
	// so that assertI2Tetc can take address of temporary.
	// No temporary for blank assignment.
	case OAS2DOTTYPE:
		t := marktemp(order)

		orderexprlist(n.List, order)
		orderexpr(&n.Rlist.N.Left, order, nil) // i in i.(T)
		if isblank(n.List.N) {
			order.out = list(order.out, n)
		} else {
			typ := n.Rlist.N.Type
			tmp1 := ordertemp(typ, order, haspointers(typ))
			order.out = list(order.out, n)
			r := Nod(OAS, n.List.N, tmp1)
			typecheck(&r, Etop)
			ordermapassign(r, order)
			n.List = list(list1(tmp1), n.List.Next.N)
		}

		cleantemp(t, order)

		// Special: use temporary variables to hold result,
	// so that chanrecv can take address of temporary.
	case OAS2RECV:
		t := marktemp(order)

		orderexprlist(n.List, order)
		orderexpr(&n.Rlist.N.Left, order, nil) // arg to recv
		ch := n.Rlist.N.Left.Type
		tmp1 := ordertemp(ch.Type, order, haspointers(ch.Type))
		var tmp2 *Node
		if !isblank(n.List.Next.N) {
			tmp2 = ordertemp(n.List.Next.N.Type, order, false)
		} else {
			tmp2 = ordertemp(Types[TBOOL], order, false)
		}
		order.out = list(order.out, n)
		r := Nod(OAS, n.List.N, tmp1)
		typecheck(&r, Etop)
		ordermapassign(r, order)
		r = Nod(OAS, n.List.Next.N, tmp2)
		typecheck(&r, Etop)
		ordermapassign(r, order)
		n.List = list(list1(tmp1), tmp2)
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
		order.out = list(order.out, n)

		// Special: handle call arguments.
	case OCALLFUNC, OCALLINTER, OCALLMETH:
		t := marktemp(order)

		ordercall(n, order)
		order.out = list(order.out, n)
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
			np := &n.Left.List.Next.N // map key
			*np = ordercopyexpr(*np, (*np).Type, order, 0)
			poptemp(t1, order)

		default:
			ordercall(n.Left, order)
		}

		order.out = list(order.out, n)
		cleantemp(t, order)

	case ODELETE:
		t := marktemp(order)
		orderexpr(&n.List.N, order, nil)
		orderexpr(&n.List.Next.N, order, nil)
		orderaddrtemp(&n.List.Next.N, order) // map key
		order.out = list(order.out, n)
		cleantemp(t, order)

		// Clean temporaries from condition evaluation at
	// beginning of loop body and after for statement.
	case OFOR:
		t := marktemp(order)

		orderexprinplace(&n.Left, order)
		var l *NodeList
		cleantempnopop(t, order, &l)
		n.Nbody = concat(l, n.Nbody)
		orderblock(&n.Nbody)
		orderstmtinplace(&n.Right)
		order.out = list(order.out, n)
		cleantemp(t, order)

		// Clean temporaries from condition at
	// beginning of both branches.
	case OIF:
		t := marktemp(order)

		orderexprinplace(&n.Left, order)
		var l *NodeList
		cleantempnopop(t, order, &l)
		n.Nbody = concat(l, n.Nbody)
		l = nil
		cleantempnopop(t, order, &l)
		n.Rlist = concat(l, n.Rlist)
		poptemp(t, order)
		orderblock(&n.Nbody)
		orderblock(&n.Rlist)
		order.out = list(order.out, n)

		// Special: argument will be converted to interface using convT2E
	// so make sure it is an addressable temporary.
	case OPANIC:
		t := marktemp(order)

		orderexpr(&n.Left, order, nil)
		if !Isinter(n.Left.Type) {
			orderaddrtemp(&n.Left, order)
		}
		order.out = list(order.out, n)
		cleantemp(t, order)

		// n->right is the expression being ranged over.
	// order it, and then make a copy if we need one.
	// We almost always do, to ensure that we don't
	// see any value changes made during the loop.
	// Usually the copy is cheap (e.g., array pointer, chan, slice, string are all tiny).
	// The exception is ranging over an array value (not a slice, not a pointer to array),
	// which must make a copy to avoid seeing updates made during
	// the range body. Ranging over an array value is uncommon though.
	case ORANGE:
		t := marktemp(order)

		orderexpr(&n.Right, order, nil)
		switch n.Type.Etype {
		default:
			Fatalf("orderstmt range %v", n.Type)

			// Mark []byte(str) range expression to reuse string backing storage.
		// It is safe because the storage cannot be mutated.
		case TARRAY:
			if n.Right.Op == OSTRARRAYBYTE {
				n.Right.Op = OSTRARRAYBYTETMP
			}
			if count(n.List) < 2 || isblank(n.List.Next.N) {
				// for i := range x will only use x once, to compute len(x).
				// No need to copy it.
				break
			}
			fallthrough

			// chan, string, slice, array ranges use value multiple times.
		// make copy.
		// fall through
		case TCHAN, TSTRING:
			r := n.Right

			if r.Type.Etype == TSTRING && r.Type != Types[TSTRING] {
				r = Nod(OCONV, r, nil)
				r.Type = Types[TSTRING]
				typecheck(&r, Erv)
			}

			n.Right = ordercopyexpr(r, r.Type, order, 0)

			// copy the map value in case it is a map literal.
		// TODO(rsc): Make tmp = literal expressions reuse tmp.
		// For maps tmp is just one word so it hardly matters.
		case TMAP:
			r := n.Right

			n.Right = ordercopyexpr(r, r.Type, order, 0)

			// n->alloc is the temp for the iterator.
			prealloc[n] = ordertemp(Types[TUINT8], order, true)
		}

		for l := n.List; l != nil; l = l.Next {
			orderexprinplace(&l.N, order)
		}
		orderblock(&n.Nbody)
		order.out = list(order.out, n)
		cleantemp(t, order)

	case ORETURN:
		ordercallargs(&n.List, order)
		order.out = list(order.out, n)

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
		for l := n.List; l != nil; l = l.Next {
			if l.N.Op != OXCASE {
				Fatalf("order select case %v", Oconv(int(l.N.Op), 0))
			}
			r = l.N.Left
			setlineno(l.N)

			// Append any new body prologue to ninit.
			// The next loop will insert ninit into nbody.
			if l.N.Ninit != nil {
				Fatalf("order select ninit")
			}
			if r != nil {
				switch r.Op {
				default:
					Yyerror("unknown op in select %v", Oconv(int(r.Op), 0))
					Dump("select case", r)

					// If this is case x := <-ch or case x, y := <-ch, the case has
				// the ODCL nodes to declare x and y. We want to delay that
				// declaration (and possible allocation) until inside the case body.
				// Delete the ODCL nodes here and recreate them inside the body below.
				case OSELRECV, OSELRECV2:
					if r.Colas {
						init := r.Ninit
						if init != nil && init.N.Op == ODCL && init.N.Left == r.Left {
							init = init.Next
						}
						if init != nil && init.N.Op == ODCL && r.List != nil && init.N.Left == r.List.N {
							init = init.Next
						}
						if init == nil {
							r.Ninit = nil
						}
					}

					if r.Ninit != nil {
						Yyerror("ninit on select recv")
						dumplist("ninit", r.Ninit)
					}

					// case x = <-c
					// case x, ok = <-c
					// r->left is x, r->ntest is ok, r->right is ORECV, r->right->left is c.
					// r->left == N means 'case <-c'.
					// c is always evaluated; x and ok are only evaluated when assigned.
					orderexpr(&r.Right.Left, order, nil)

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
							typecheck(&tmp2, Etop)
							l.N.Ninit = list(l.N.Ninit, tmp2)
						}

						r.Left = ordertemp(r.Right.Left.Type.Type, order, haspointers(r.Right.Left.Type.Type))
						tmp2 = Nod(OAS, tmp1, r.Left)
						typecheck(&tmp2, Etop)
						l.N.Ninit = list(l.N.Ninit, tmp2)
					}

					if r.List != nil && isblank(r.List.N) {
						r.List = nil
					}
					if r.List != nil {
						tmp1 = r.List.N
						if r.Colas {
							tmp2 = Nod(ODCL, tmp1, nil)
							typecheck(&tmp2, Etop)
							l.N.Ninit = list(l.N.Ninit, tmp2)
						}

						r.List = list1(ordertemp(tmp1.Type, order, false))
						tmp2 = Nod(OAS, tmp1, r.List.N)
						typecheck(&tmp2, Etop)
						l.N.Ninit = list(l.N.Ninit, tmp2)
					}

					orderblock(&l.N.Ninit)

				case OSEND:
					if r.Ninit != nil {
						Yyerror("ninit on select send")
						dumplist("ninit", r.Ninit)
					}

					// case c <- x
					// r->left is c, r->right is x, both are always evaluated.
					orderexpr(&r.Left, order, nil)

					if !istemp(r.Left) {
						r.Left = ordercopyexpr(r.Left, r.Left.Type, order, 0)
					}
					orderexpr(&r.Right, order, nil)
					if !istemp(r.Right) {
						r.Right = ordercopyexpr(r.Right, r.Right.Type, order, 0)
					}
				}
			}

			orderblock(&l.N.Nbody)
		}

		// Now that we have accumulated all the temporaries, clean them.
		// Also insert any ninit queued during the previous loop.
		// (The temporary cleaning must follow that ninit work.)
		for l := n.List; l != nil; l = l.Next {
			cleantempnopop(t, order, &l.N.Ninit)
			l.N.Nbody = concat(l.N.Ninit, l.N.Nbody)
			l.N.Ninit = nil
		}

		order.out = list(order.out, n)
		poptemp(t, order)

		// Special: value being sent is passed as a pointer; make it addressable.
	case OSEND:
		t := marktemp(order)

		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right, order, nil)
		orderaddrtemp(&n.Right, order)
		order.out = list(order.out, n)
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

		orderexpr(&n.Left, order, nil)
		for l := n.List; l != nil; l = l.Next {
			if l.N.Op != OXCASE {
				Fatalf("order switch case %v", Oconv(int(l.N.Op), 0))
			}
			orderexprlistinplace(l.N.List, order)
			orderblock(&l.N.Nbody)
		}

		order.out = list(order.out, n)
		cleantemp(t, order)
	}

	lineno = int32(lno)
}

// Orderexprlist orders the expression list l into order.
func orderexprlist(l *NodeList, order *Order) {
	for ; l != nil; l = l.Next {
		orderexpr(&l.N, order, nil)
	}
}

// Orderexprlist orders the expression list l but saves
// the side effects on the individual expression ninit lists.
func orderexprlistinplace(l *NodeList, order *Order) {
	for ; l != nil; l = l.Next {
		orderexprinplace(&l.N, order)
	}
}

// prealloc[x] records the allocation to use for x.
var prealloc = map[*Node]*Node{}

// Orderexpr orders a single expression, appending side
// effects to order->out as needed.
// If this is part of an assignment lhs = *np, lhs is given.
// Otherwise lhs == nil. (When lhs != nil it may be possible
// to avoid copying the result of the expression to a temporary.)
func orderexpr(np **Node, order *Order, lhs *Node) {
	n := *np
	if n == nil {
		return
	}

	lno := int(setlineno(n))
	orderinit(n, order)

	switch n.Op {
	default:
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right, order, nil)
		orderexprlist(n.List, order)
		orderexprlist(n.Rlist, order)

		// Addition of strings turns into a function call.
	// Allocate a temporary to hold the strings.
	// Fewer than 5 strings use direct runtime helpers.
	case OADDSTR:
		orderexprlist(n.List, order)

		if count(n.List) > 5 {
			t := typ(TARRAY)
			t.Bound = int64(count(n.List))
			t.Type = Types[TSTRING]
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
		for l := n.List; l != nil; l = l.Next {
			hasbyte = hasbyte || l.N.Op == OARRAYBYTESTR
			haslit = haslit || l.N.Op == OLITERAL && len(l.N.Val().U.(string)) != 0
		}

		if haslit && hasbyte {
			for l := n.List; l != nil; l = l.Next {
				if l.N.Op == OARRAYBYTESTR {
					l.N.Op = OARRAYBYTESTRTMP
				}
			}
		}

	case OCMPSTR:
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right, order, nil)

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
		orderexpr(&n.Left, order, nil)

		orderexpr(&n.Right, order, nil)

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

		orderaddrtemp(&n.Right, order)
		if n.Etype == 0 {
			// use of value (not being assigned);
			// make copy in temporary.
			n = ordercopyexpr(n, n.Type, order, 0)
		}

		// concrete type (not interface) argument must be addressable
	// temporary to pass to runtime.
	case OCONVIFACE:
		orderexpr(&n.Left, order, nil)

		if !Isinter(n.Left.Type) {
			orderaddrtemp(&n.Left, order)
		}

	case OANDAND, OOROR:
		mark := marktemp(order)
		orderexpr(&n.Left, order, nil)

		// Clean temporaries from first branch at beginning of second.
		// Leave them on the stack so that they can be killed in the outer
		// context in case the short circuit is taken.
		var l *NodeList

		cleantempnopop(mark, order, &l)
		n.Right.Ninit = concat(l, n.Right.Ninit)
		orderexprinplace(&n.Right, order)

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
		if lhs == nil || lhs.Op != ONAME && !samesafeexpr(lhs, n.List.N) {
			n = ordercopyexpr(n, n.Type, order, 0)
		}

	case OSLICE, OSLICEARR, OSLICESTR:
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right.Left, order, nil)
		n.Right.Left = ordercheapexpr(n.Right.Left, order)
		orderexpr(&n.Right.Right, order, nil)
		n.Right.Right = ordercheapexpr(n.Right.Right, order)
		if lhs == nil || lhs.Op != ONAME && !samesafeexpr(lhs, n.Left) {
			n = ordercopyexpr(n, n.Type, order, 0)
		}

	case OSLICE3, OSLICE3ARR:
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right.Left, order, nil)
		n.Right.Left = ordercheapexpr(n.Right.Left, order)
		orderexpr(&n.Right.Right.Left, order, nil)
		n.Right.Right.Left = ordercheapexpr(n.Right.Right.Left, order)
		orderexpr(&n.Right.Right.Right, order, nil)
		n.Right.Right.Right = ordercheapexpr(n.Right.Right.Right, order)
		if lhs == nil || lhs.Op != ONAME && !samesafeexpr(lhs, n.Left) {
			n = ordercopyexpr(n, n.Type, order, 0)
		}

	case OCLOSURE:
		if n.Noescape && n.Func.Cvars != nil {
			prealloc[n] = ordertemp(Types[TUINT8], order, false) // walk will fill in correct type
		}

	case OARRAYLIT, OCALLPART:
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right, order, nil)
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
			prealloc[n] = ordertemp(n.Type.Type, order, false)
		}

	case ODOTTYPE, ODOTTYPE2:
		orderexpr(&n.Left, order, nil)
		// TODO(rsc): The Isfat is for consistency with componentgen and walkexpr.
		// It needs to be removed in all three places.
		// That would allow inlining x.(struct{*int}) the same as x.(*int).
		if !isdirectiface(n.Type) || Isfat(n.Type) || instrumenting {
			n = ordercopyexpr(n, n.Type, order, 1)
		}

	case ORECV:
		orderexpr(&n.Left, order, nil)
		n = ordercopyexpr(n, n.Type, order, 1)

	case OEQ, ONE:
		orderexpr(&n.Left, order, nil)
		orderexpr(&n.Right, order, nil)
		t := n.Left.Type
		if t.Etype == TSTRUCT || Isfixedarray(t) {
			// for complex comparisons, we need both args to be
			// addressable so we can pass them to the runtime.
			orderaddrtemp(&n.Left, order)
			orderaddrtemp(&n.Right, order)
		}
	}

	lineno = int32(lno)

	*np = n
}
