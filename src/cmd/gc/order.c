// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

#include	<u.h>
#include	<libc.h>
#include	"go.h"

// Order holds state during the ordering process.
typedef struct Order Order;
struct Order
{
	NodeList *out; // list of generated statements
	NodeList *temp; // head of stack of temporary variables
	NodeList *free; // free list of NodeList* structs (for use in temp)
};

static void	orderstmt(Node*, Order*);
static void	orderstmtlist(NodeList*, Order*);
static void	orderblock(NodeList **l);
static void	orderexpr(Node**, Order*);
static void orderexprinplace(Node**, Order*);
static void	orderexprlist(NodeList*, Order*);
static void	orderexprlistinplace(NodeList*, Order*);

// Order rewrites fn->nbody to apply the ordering constraints
// described in the comment at the top of the file.
void
order(Node *fn)
{
	orderblock(&fn->nbody);
}

// Ordertemp allocates a new temporary with the given type,
// pushes it onto the temp stack, and returns it.
// If clear is true, ordertemp emits code to zero the temporary.
static Node*
ordertemp(Type *t, Order *order, int clear)
{
	Node *var, *a;
	NodeList *l;

	var = temp(t);
	if(clear) {
		a = nod(OAS, var, N);
		typecheck(&a, Etop);
		order->out = list(order->out, a);
	}
	if((l = order->free) == nil)
		l = mal(sizeof *l);
	order->free = l->next;
	l->next = order->temp;
	l->n = var;
	order->temp = l;
	return var;
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
static Node*
ordercopyexpr(Node *n, Type *t, Order *order, int clear)
{
	Node *a, *var;

	var = ordertemp(t, order, clear);
	a = nod(OAS, var, n);
	typecheck(&a, Etop);
	order->out = list(order->out, a);
	return var;
}

// Ordercheapexpr returns a cheap version of n.
// The definition of cheap is that n is a variable or constant.
// If not, ordercheapexpr allocates a new tmp, emits tmp = n,
// and then returns tmp.
static Node*
ordercheapexpr(Node *n, Order *order)
{
	switch(n->op) {
	case ONAME:
	case OLITERAL:
		return n;
	}
	return ordercopyexpr(n, n->type, order, 0);
}

// Ordersafeexpr returns a safe version of n.
// The definition of safe is that n can appear multiple times
// without violating the semantics of the original program,
// and that assigning to the safe version has the same effect
// as assigning to the original n.
//
// The intended use is to apply to x when rewriting x += y into x = x + y.
static Node*
ordersafeexpr(Node *n, Order *order)
{
	Node *l, *r, *a;
	
	switch(n->op) {
	default:
		fatal("ordersafeexpr %O", n->op);

	case ONAME:
	case OLITERAL:
		return n;

	case ODOT:
		l = ordersafeexpr(n->left, order);
		if(l == n->left)
			return n;
		a = nod(OXXX, N, N);
		*a = *n;
		a->orig = a;
		a->left = l;
		typecheck(&a, Erv);
		return a;

	case ODOTPTR:
	case OIND:
		l = ordercheapexpr(n->left, order);
		if(l == n->left)
			return n;
		a = nod(OXXX, N, N);
		*a = *n;
		a->orig = a;
		a->left = l;
		typecheck(&a, Erv);
		return a;
		
	case OINDEX:
	case OINDEXMAP:
		if(isfixedarray(n->left->type))
			l = ordersafeexpr(n->left, order);
		else
			l = ordercheapexpr(n->left, order);
		r = ordercheapexpr(n->right, order);
		if(l == n->left && r == n->right)
			return n;
		a = nod(OXXX, N, N);
		*a = *n;
		a->orig = a;
		a->left = l;
		a->right = r;
		typecheck(&a, Erv);
		return a;
	}
}		

// Istemp reports whether n is a temporary variable.
static int
istemp(Node *n)
{
	if(n->op != ONAME)
		return 0;
	return strncmp(n->sym->name, "autotmp_", 8) == 0;
}

// Isaddrokay reports whether it is okay to pass n's address to runtime routines.
// Taking the address of a variable makes the liveness and optimization analyses
// lose track of where the variable's lifetime ends. To avoid hurting the analyses
// of ordinary stack variables, those are not 'isaddrokay'. Temporaries are okay,
// because we emit explicit VARKILL instructions marking the end of those
// temporaries' lifetimes.
static int
isaddrokay(Node *n)
{
	return islvalue(n) && (n->op != ONAME || n->class == PEXTERN || istemp(n));
}

// Orderaddrtemp ensures that *np is okay to pass by address to runtime routines.
// If the original argument *np is not okay, orderaddrtemp creates a tmp, emits
// tmp = *np, and then sets *np to the tmp variable.
static void
orderaddrtemp(Node **np, Order *order)
{
	Node *n;
	
	n = *np;
	if(isaddrokay(n))
		return;
	*np = ordercopyexpr(n, n->type, order, 0);
}

// Marktemp returns the top of the temporary variable stack.
static NodeList*
marktemp(Order *order)
{
	return order->temp;
}

// Poptemp pops temporaries off the stack until reaching the mark,
// which must have been returned by marktemp.
static void
poptemp(NodeList *mark, Order *order)
{
	NodeList *l;

	while((l = order->temp) != mark) {
		order->temp = l->next;
		l->next = order->free;
		order->free = l;
	}
}

// Cleantempnopop emits to *out VARKILL instructions for each temporary
// above the mark on the temporary stack, but it does not pop them
// from the stack.
static void
cleantempnopop(NodeList *mark, Order *order, NodeList **out)
{
	NodeList *l;
	Node *kill;

	for(l=order->temp; l != mark; l=l->next) {
		kill = nod(OVARKILL, l->n, N);
		typecheck(&kill, Etop);
		*out = list(*out, kill);
	}
}

// Cleantemp emits VARKILL instructions for each temporary above the
// mark on the temporary stack and removes them from the stack.
static void
cleantemp(NodeList *top, Order *order)
{
	cleantempnopop(top, order, &order->out);
	poptemp(top, order);
}

// Orderstmtlist orders each of the statements in the list.
static void
orderstmtlist(NodeList *l, Order *order)
{
	for(; l; l=l->next)
		orderstmt(l->n, order);
}

// Orderblock orders the block of statements *l onto a new list,
// and then replaces *l with that list.
static void
orderblock(NodeList **l)
{
	Order order;
	NodeList *mark;
	
	memset(&order, 0, sizeof order);
	mark = marktemp(&order);
	orderstmtlist(*l, &order);
	cleantemp(mark, &order);
	*l = order.out;
}

// Orderexprinplace orders the side effects in *np and
// leaves them as the init list of the final *np.
static void
orderexprinplace(Node **np, Order *outer)
{
	Node *n;
	NodeList **lp;
	Order order;
	
	n = *np;
	memset(&order, 0, sizeof order);
	orderexpr(&n, &order);
	addinit(&n, order.out);
	
	// insert new temporaries from order
	// at head of outer list.
	lp = &order.temp;
	while(*lp != nil)
		lp = &(*lp)->next;
	*lp = outer->temp;
	outer->temp = order.temp;

	*np = n;
}

// Orderstmtinplace orders the side effects of the single statement *np
// and replaces it with the resulting statement list.
void
orderstmtinplace(Node **np)
{
	Node *n;
	Order order;
	NodeList *mark;
	
	n = *np;
	memset(&order, 0, sizeof order);
	mark = marktemp(&order);
	orderstmt(n, &order);
	cleantemp(mark, &order);
	*np = liststmt(order.out);
}

// Orderinit moves n's init list to order->out.
static void
orderinit(Node *n, Order *order)
{
	orderstmtlist(n->ninit, order);
	n->ninit = nil;
}

// Ismulticall reports whether the list l is f() for a multi-value function.
// Such an f() could appear as the lone argument to a multi-arg function.
static int
ismulticall(NodeList *l)
{
	Node *n;
	
	// one arg only
	if(l == nil || l->next != nil)
		return 0;
	n = l->n;
	
	// must be call
	switch(n->op) {
	default:
		return 0;
	case OCALLFUNC:
	case OCALLMETH:
	case OCALLINTER:
		break;
	}
	
	// call must return multiple values
	return n->left->type->outtuple > 1;
}

// Copyret emits t1, t2, ... = n, where n is a function call,
// and then returns the list t1, t2, ....
static NodeList*
copyret(Node *n, Order *order)
{
	Type *t;
	Node *tmp, *as;
	NodeList *l1, *l2;
	Iter tl;
	
	if(n->type->etype != TSTRUCT || !n->type->funarg)
		fatal("copyret %T %d", n->type, n->left->type->outtuple);

	l1 = nil;
	l2 = nil;
	for(t=structfirst(&tl, &n->type); t; t=structnext(&tl)) {
		tmp = temp(t->type);
		l1 = list(l1, tmp);
		l2 = list(l2, tmp);
	}
	
	as = nod(OAS2, N, N);
	as->list = l1;
	as->rlist = list1(n);
	typecheck(&as, Etop);
	orderstmt(as, order);

	return l2;
}

// Ordercallargs orders the list of call arguments *l.
static void
ordercallargs(NodeList **l, Order *order)
{
	if(ismulticall(*l)) {
		// return f() where f() is multiple values.
		*l = copyret((*l)->n, order);
	} else {
		orderexprlist(*l, order);
	}
}

// Ordercall orders the call expression n.
// n->op is OCALLMETH/OCALLFUNC/OCALLINTER or a builtin like OCOPY.
static void
ordercall(Node *n, Order *order)
{
	orderexpr(&n->left, order);
	orderexpr(&n->right, order); // ODDDARG temp
	ordercallargs(&n->list, order);
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
// We could do a more precise analysis if needed, like in walk.c.
//
// Ordermapassign also inserts these temporaries if needed for
// calling writebarrierfat with a pointer to n->right.
static void
ordermapassign(Node *n, Order *order)
{
	Node *m, *a;
	NodeList *l;
	NodeList *post;

	switch(n->op) {
	default:
		fatal("ordermapassign %O", n->op);

	case OAS:
		order->out = list(order->out, n);
		// We call writebarrierfat only for values > 4 pointers long. See walk.c.
		if((n->left->op == OINDEXMAP || (needwritebarrier(n->left, n->right) && n->left->type->width > 4*widthptr)) && !isaddrokay(n->right)) {
			m = n->left;
			n->left = ordertemp(m->type, order, 0);
			a = nod(OAS, m, n->left);
			typecheck(&a, Etop);
			order->out = list(order->out, a);
		}
		break;

	case OAS2:
	case OAS2DOTTYPE:
	case OAS2MAPR:
	case OAS2FUNC:
		post = nil;
		for(l=n->list; l != nil; l=l->next) {
			if(l->n->op == OINDEXMAP) {
				m = l->n;
				if(!istemp(m->left))
					m->left = ordercopyexpr(m->left, m->left->type, order, 0);
				if(!istemp(m->right))
					m->right = ordercopyexpr(m->right, m->right->type, order, 0);
				l->n = ordertemp(m->type, order, 0);
				a = nod(OAS, m, l->n);
				typecheck(&a, Etop);
				post = list(post, a);
			}
		}
		order->out = list(order->out, n);
		order->out = concat(order->out, post);
		break;
	}
}

// Orderstmt orders the statement n, appending to order->out.
// Temporaries created during the statement are cleaned
// up using VARKILL instructions as possible.
static void
orderstmt(Node *n, Order *order)
{
	int lno;
	NodeList *l, *t, *t1;
	Node *r, *tmp1, *tmp2, **np;
	Type *ch;

	if(n == N)
		return;

	lno = setlineno(n);

	orderinit(n, order);

	switch(n->op) {
	default:
		fatal("orderstmt %O", n->op);

	case OVARKILL:
		order->out = list(order->out, n);
		break;

	case OAS:
	case OAS2:
	case OAS2DOTTYPE:
	case OCLOSE:
	case OCOPY:
	case OPRINT:
	case OPRINTN:
	case ORECOVER:
	case ORECV:
		t = marktemp(order);
		orderexpr(&n->left, order);
		orderexpr(&n->right, order);
		orderexprlist(n->list, order);
		orderexprlist(n->rlist, order);
		switch(n->op) {
		case OAS:
		case OAS2:
		case OAS2DOTTYPE:
			ordermapassign(n, order);
			break;
		default:
			order->out = list(order->out, n);
			break;
		}
		cleantemp(t, order);
		break;

	case OASOP:
		// Special: rewrite l op= r into l = l op r.
		// This simplies quite a few operations;
		// most important is that it lets us separate
		// out map read from map write when l is
		// a map index expression.
		t = marktemp(order);
		orderexpr(&n->left, order);
		n->left = ordersafeexpr(n->left, order);
		tmp1 = treecopy(n->left);
		if(tmp1->op == OINDEXMAP)
			tmp1->etype = 0; // now an rvalue not an lvalue
		tmp1 = ordercopyexpr(tmp1, n->left->type, order, 0);
		n->right = nod(n->etype, tmp1, n->right);
		typecheck(&n->right, Erv);
		orderexpr(&n->right, order);
		n->etype = 0;
		n->op = OAS;
		ordermapassign(n, order);
		cleantemp(t, order);
		break;

	case OAS2MAPR:
		// Special: make sure key is addressable,
		// and make sure OINDEXMAP is not copied out.
		t = marktemp(order);
		orderexprlist(n->list, order);
		r = n->rlist->n;
		orderexpr(&r->left, order);
		orderexpr(&r->right, order);
		// See case OINDEXMAP below.
		if(r->right->op == OARRAYBYTESTR)
			r->right->op = OARRAYBYTESTRTMP;
		orderaddrtemp(&r->right, order);
		ordermapassign(n, order);
		cleantemp(t, order);
		break;

	case OAS2FUNC:
		// Special: avoid copy of func call n->rlist->n.
		t = marktemp(order);
		orderexprlist(n->list, order);
		ordercall(n->rlist->n, order);
		ordermapassign(n, order);
		cleantemp(t, order);
		break;

	case OAS2RECV:
		// Special: avoid copy of receive.
		// Use temporary variables to hold result,
		// so that chanrecv can take address of temporary.
		t = marktemp(order);
		orderexprlist(n->list, order);
		orderexpr(&n->rlist->n->left, order);  // arg to recv
		ch = n->rlist->n->left->type;
		tmp1 = ordertemp(ch->type, order, haspointers(ch->type));
		if(!isblank(n->list->next->n))
			tmp2 = ordertemp(n->list->next->n->type, order, 0);
		else
			tmp2 = ordertemp(types[TBOOL], order, 0);
		order->out = list(order->out, n);
		r = nod(OAS, n->list->n, tmp1);
		typecheck(&r, Etop);
		ordermapassign(r, order);
		r = nod(OAS, n->list->next->n, tmp2);
		typecheck(&r, Etop);
		ordermapassign(r, order);
		n->list = list(list1(tmp1), tmp2);
		cleantemp(t, order);
		break;

	case OBLOCK:
	case OEMPTY:
		// Special: does not save n onto out.
		orderstmtlist(n->list, order);
		break;

	case OBREAK:
	case OCONTINUE:
	case ODCL:
	case ODCLCONST:
	case ODCLTYPE:
	case OFALL:
	case OXFALL:
	case OGOTO:
	case OLABEL:
	case ORETJMP:
		// Special: n->left is not an expression; save as is.
		order->out = list(order->out, n);
		break;

	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
		// Special: handle call arguments.
		t = marktemp(order);
		ordercall(n, order);
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;

	case ODEFER:
	case OPROC:
		// Special: order arguments to inner call but not call itself.
		t = marktemp(order);
		switch(n->left->op) {
		case ODELETE:
			// Delete will take the address of the key.
			// Copy key into new temp and do not clean it
			// (it persists beyond the statement).
			orderexprlist(n->left->list, order);
			t1 = marktemp(order);
			np = &n->left->list->next->n; // map key
			*np = ordercopyexpr(*np, (*np)->type, order, 0);
			poptemp(t1, order);
			break;
		default:
			ordercall(n->left, order);
			break;
		}
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;

	case ODELETE:
		t = marktemp(order);
		orderexpr(&n->list->n, order);
		orderexpr(&n->list->next->n, order);
		orderaddrtemp(&n->list->next->n, order); // map key
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;

	case OFOR:
		// Clean temporaries from condition evaluation at
		// beginning of loop body and after for statement.
		t = marktemp(order);
		orderexprinplace(&n->ntest, order);
		l = nil;
		cleantempnopop(t, order, &l);
		n->nbody = concat(l, n->nbody);
		orderblock(&n->nbody);
		orderstmtinplace(&n->nincr);
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;
		
	case OIF:
		// Clean temporaries from condition at
		// beginning of both branches.
		t = marktemp(order);
		orderexprinplace(&n->ntest, order);
		l = nil;
		cleantempnopop(t, order, &l);
		n->nbody = concat(l, n->nbody);
		l = nil;
		cleantempnopop(t, order, &l);
		n->nelse = concat(l, n->nelse);
		poptemp(t, order);
		orderblock(&n->nbody);
		orderblock(&n->nelse);
		order->out = list(order->out, n);
		break;

	case OPANIC:
		// Special: argument will be converted to interface using convT2E
		// so make sure it is an addressable temporary.
		t = marktemp(order);
		orderexpr(&n->left, order);
		if(!isinter(n->left->type))
			orderaddrtemp(&n->left, order);
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;

	case ORANGE:
		// n->right is the expression being ranged over.
		// order it, and then make a copy if we need one.
		// We almost always do, to ensure that we don't
		// see any value changes made during the loop.
		// Usually the copy is cheap (e.g., array pointer, chan, slice, string are all tiny).
		// The exception is ranging over an array value (not a slice, not a pointer to array),
		// which must make a copy to avoid seeing updates made during
		// the range body. Ranging over an array value is uncommon though.
		t = marktemp(order);
		orderexpr(&n->right, order);
		switch(n->type->etype) {
		default:
			fatal("orderstmt range %T", n->type);
		case TARRAY:
			if(count(n->list) < 2 || isblank(n->list->next->n)) {
				// for i := range x will only use x once, to compute len(x).
				// No need to copy it.
				break;
			}
			// fall through
		case TCHAN:
		case TSTRING:
			// chan, string, slice, array ranges use value multiple times.
			// make copy.
			r = n->right;
			if(r->type->etype == TSTRING && r->type != types[TSTRING]) {
				r = nod(OCONV, r, N);
				r->type = types[TSTRING];
				typecheck(&r, Erv);
			}
			n->right = ordercopyexpr(r, r->type, order, 0);
			break;
		case TMAP:
			// copy the map value in case it is a map literal.
			// TODO(rsc): Make tmp = literal expressions reuse tmp.
			// For maps tmp is just one word so it hardly matters.
			r = n->right;
			n->right = ordercopyexpr(r, r->type, order, 0);
			// n->alloc is the temp for the iterator.
			n->alloc = ordertemp(types[TUINT8], order, 1);
			break;
		}
		for(l=n->list; l; l=l->next)
			orderexprinplace(&l->n, order);
		orderblock(&n->nbody);
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;

	case ORETURN:
		ordercallargs(&n->list, order);
		order->out = list(order->out, n);
		break;
	
	case OSELECT:
		// Special: clean case temporaries in each block entry.
		// Select must enter one of its blocks, so there is no
		// need for a cleaning at the end.
		// Doubly special: evaluation order for select is stricter
		// than ordinary expressions. Even something like p.c
		// has to be hoisted into a temporary, so that it cannot be
		// reordered after the channel evaluation for a different
		// case (if p were nil, then the timing of the fault would
		// give this away).
		t = marktemp(order);
		for(l=n->list; l; l=l->next) {
			if(l->n->op != OXCASE)
				fatal("order select case %O", l->n->op);
			r = l->n->left;
			setlineno(l->n);
			// Append any new body prologue to ninit.
			// The next loop will insert ninit into nbody.
			if(l->n->ninit != nil)
				fatal("order select ninit");
			if(r != nil) {
				switch(r->op) {
				default:
					yyerror("unknown op in select %O", r->op);
					dump("select case", r);
					break;

				case OSELRECV:
				case OSELRECV2:
					// If this is case x := <-ch or case x, y := <-ch, the case has
					// the ODCL nodes to declare x and y. We want to delay that
					// declaration (and possible allocation) until inside the case body.
					// Delete the ODCL nodes here and recreate them inside the body below.
					if(r->colas) {
						t = r->ninit;
						if(t != nil && t->n->op == ODCL && t->n->left == r->left)
							t = t->next;
						if(t != nil && t->n->op == ODCL && t->n->left == r->ntest)
							t = t->next;
						if(t == nil)
							r->ninit = nil;
					}
					if(r->ninit != nil) {
						yyerror("ninit on select recv");
						dumplist("ninit", r->ninit);
					}
					// case x = <-c
					// case x, ok = <-c
					// r->left is x, r->ntest is ok, r->right is ORECV, r->right->left is c.
					// r->left == N means 'case <-c'.
					// c is always evaluated; x and ok are only evaluated when assigned.
					orderexpr(&r->right->left, order);
					if(r->right->left->op != ONAME)
						r->right->left = ordercopyexpr(r->right->left, r->right->left->type, order, 0);

					// Introduce temporary for receive and move actual copy into case body.
					// avoids problems with target being addressed, as usual.
					// NOTE: If we wanted to be clever, we could arrange for just one
					// temporary per distinct type, sharing the temp among all receives
					// with that temp. Similarly one ok bool could be shared among all
					// the x,ok receives. Not worth doing until there's a clear need.
					if(r->left != N && isblank(r->left))
						r->left = N;
					if(r->left != N) {
						// use channel element type for temporary to avoid conversions,
						// such as in case interfacevalue = <-intchan.
						// the conversion happens in the OAS instead.
						tmp1 = r->left;
						if(r->colas) {
							tmp2 = nod(ODCL, tmp1, N);
							typecheck(&tmp2, Etop);
							l->n->ninit = list(l->n->ninit, tmp2);
						}
						r->left = ordertemp(r->right->left->type->type, order, haspointers(r->right->left->type->type));
						tmp2 = nod(OAS, tmp1, r->left);
						typecheck(&tmp2, Etop);
						l->n->ninit = list(l->n->ninit, tmp2);
					}
					if(r->ntest != N && isblank(r->ntest))
						r->ntest = N;
					if(r->ntest != N) {
						tmp1 = r->ntest;
						if(r->colas) {
							tmp2 = nod(ODCL, tmp1, N);
							typecheck(&tmp2, Etop);
							l->n->ninit = list(l->n->ninit, tmp2);
						}
						r->ntest = ordertemp(tmp1->type, order, 0);
						tmp2 = nod(OAS, tmp1, r->ntest);
						typecheck(&tmp2, Etop);
						l->n->ninit = list(l->n->ninit, tmp2);
					}
					orderblock(&l->n->ninit);
					break;

				case OSEND:
					if(r->ninit != nil) {
						yyerror("ninit on select send");
						dumplist("ninit", r->ninit);
					}
					// case c <- x
					// r->left is c, r->right is x, both are always evaluated.
					orderexpr(&r->left, order);
					if(!istemp(r->left))
						r->left = ordercopyexpr(r->left, r->left->type, order, 0);
					orderexpr(&r->right, order);
					if(!istemp(r->right))
						r->right = ordercopyexpr(r->right, r->right->type, order, 0);
					break;
				}
			}
			orderblock(&l->n->nbody);
		}
		// Now that we have accumulated all the temporaries, clean them.
		// Also insert any ninit queued during the previous loop.
		// (The temporary cleaning must follow that ninit work.)
		for(l=n->list; l; l=l->next) {
			cleantempnopop(t, order, &l->n->ninit);
			l->n->nbody = concat(l->n->ninit, l->n->nbody);
			l->n->ninit = nil;
		}
		order->out = list(order->out, n);
		poptemp(t, order);
		break;

	case OSEND:
		// Special: value being sent is passed as a pointer; make it addressable.
		t = marktemp(order);
		orderexpr(&n->left, order);
		orderexpr(&n->right, order);
		orderaddrtemp(&n->right, order);
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;

	case OSWITCH:
		// TODO(rsc): Clean temporaries more aggressively.
		// Note that because walkswitch will rewrite some of the
		// switch into a binary search, this is not as easy as it looks.
		// (If we ran that code here we could invoke orderstmt on
		// the if-else chain instead.)
		// For now just clean all the temporaries at the end.
		// In practice that's fine.
		t = marktemp(order);
		orderexpr(&n->ntest, order);
		for(l=n->list; l; l=l->next) {
			if(l->n->op != OXCASE)
				fatal("order switch case %O", l->n->op);
			orderexprlistinplace(l->n->list, order);
			orderblock(&l->n->nbody);
		}
		order->out = list(order->out, n);
		cleantemp(t, order);
		break;
	}
	
	lineno = lno;
}

// Orderexprlist orders the expression list l into order.
static void
orderexprlist(NodeList *l, Order *order)
{
	for(; l; l=l->next)
		orderexpr(&l->n, order);
}

// Orderexprlist orders the expression list l but saves
// the side effects on the individual expression ninit lists.
static void
orderexprlistinplace(NodeList *l, Order *order)
{
	for(; l; l=l->next)
		orderexprinplace(&l->n, order);
}

// Orderexpr orders a single expression, appending side
// effects to order->out as needed.
static void
orderexpr(Node **np, Order *order)
{
	Node *n;
	NodeList *mark, *l;
	Type *t;
	int lno;

	n = *np;
	if(n == N)
		return;

	lno = setlineno(n);
	orderinit(n, order);

	switch(n->op) {
	default:
		orderexpr(&n->left, order);
		orderexpr(&n->right, order);
		orderexprlist(n->list, order);
		orderexprlist(n->rlist, order);
		break;
	
	case OADDSTR:
		// Addition of strings turns into a function call.
		// Allocate a temporary to hold the strings.
		// Fewer than 5 strings use direct runtime helpers.
		orderexprlist(n->list, order);
		if(count(n->list) > 5) {
			t = typ(TARRAY);
			t->bound = count(n->list);
			t->type = types[TSTRING];
			n->alloc = ordertemp(t, order, 0);
		}
		break;

	case OINDEXMAP:
		// key must be addressable
		orderexpr(&n->left, order);
		orderexpr(&n->right, order);

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
		if(n->etype == 0 && n->right->op == OARRAYBYTESTR)
			n->right->op = OARRAYBYTESTRTMP;

		orderaddrtemp(&n->right, order);
		if(n->etype == 0) {
			// use of value (not being assigned);
			// make copy in temporary.
			n = ordercopyexpr(n, n->type, order, 0);
		}
		break;
	
	case OCONVIFACE:
		// concrete type (not interface) argument must be addressable
		// temporary to pass to runtime.
		orderexpr(&n->left, order);
		if(!isinter(n->left->type))
			orderaddrtemp(&n->left, order);
		break;
	
	case OANDAND:
	case OOROR:
		mark = marktemp(order);
		orderexpr(&n->left, order);
		// Clean temporaries from first branch at beginning of second.
		// Leave them on the stack so that they can be killed in the outer
		// context in case the short circuit is taken.
		l = nil;
		cleantempnopop(mark, order, &l);
		n->right->ninit = concat(l, n->right->ninit);
		orderexprinplace(&n->right, order);
		break;
	
	case OAPPEND:
	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
	case OCAP:
	case OCOMPLEX:
	case OCOPY:
	case OIMAG:
	case OLEN:
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case ONEW:
	case OREAL:
	case ORECOVER:
		ordercall(n, order);
		n = ordercopyexpr(n, n->type, order, 0);
		break;

	case OCLOSURE:
		if(n->noescape && n->cvars != nil)
			n->alloc = ordertemp(types[TUINT8], order, 0); // walk will fill in correct type
		break;

	case OARRAYLIT:
	case OCALLPART:
		orderexpr(&n->left, order);
		orderexpr(&n->right, order);
		orderexprlist(n->list, order);
		orderexprlist(n->rlist, order);
		if(n->noescape)
			n->alloc = ordertemp(types[TUINT8], order, 0); // walk will fill in correct type
		break;

	case ODDDARG:
		if(n->noescape) {
			// The ddd argument does not live beyond the call it is created for.
			// Allocate a temporary that will be cleaned up when this statement
			// completes. We could be more aggressive and try to arrange for it
			// to be cleaned up when the call completes.
			n->alloc = ordertemp(n->type->type, order, 0);
		}
		break;

	case ORECV:
		orderexpr(&n->left, order);
		n = ordercopyexpr(n, n->type, order, 1);
		break;

	case OEQ:
	case ONE:
		orderexpr(&n->left, order);
		orderexpr(&n->right, order);
		t = n->left->type;
		if(t->etype == TSTRUCT || isfixedarray(t)) {
			// for complex comparisons, we need both args to be
			// addressable so we can pass them to the runtime.
			orderaddrtemp(&n->left, order);
			orderaddrtemp(&n->right, order);
		}
		break;
	}
	
	lineno = lno;

	*np = n;
}
