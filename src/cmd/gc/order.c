// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Rewrite tree to use separate statements to enforce
// order of evaluation.  Makes walk easier, because it
// can (after this runs) reorder at will within an expression.

#include	<u.h>
#include	<libc.h>
#include	"go.h"

static void	orderstmt(Node*, NodeList**);
static void	orderstmtlist(NodeList*, NodeList**);
static void	orderblock(NodeList **l);
static void	orderexpr(Node**, NodeList**);
static void	orderexprlist(NodeList*, NodeList**);

void
order(Node *fn)
{
	orderblock(&fn->nbody);
}

static void
orderstmtlist(NodeList *l, NodeList **out)
{
	for(; l; l=l->next)
		orderstmt(l->n, out);
}

// Order the block of statements *l onto a new list,
// and then replace *l with that list.
static void
orderblock(NodeList **l)
{
	NodeList *out;
	
	out = nil;
	orderstmtlist(*l, &out);
	*l = out;
}

// Order the side effects in *np and leave them as
// the init list of the final *np.
static void
orderexprinplace(Node **np)
{
	Node *n;
	NodeList *out;
	
	n = *np;
	out = nil;
	orderexpr(&n, &out);
	addinit(&n, out);
	*np = n;
}

// Like orderblock, but applied to a single statement.
static void
orderstmtinplace(Node **np)
{
	Node *n;
	NodeList *out;

	n = *np;
	out = nil;
	orderstmt(n, &out);
	*np = liststmt(out);
}

// Move n's init list to *out.
static void
orderinit(Node *n, NodeList **out)
{
	orderstmtlist(n->ninit, out);
	n->ninit = nil;
}

// Is the list l actually just f() for a multi-value function?
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

// n is a multi-value function call.  Add t1, t2, .. = n to out
// and return the list t1, t2, ...
static NodeList*
copyret(Node *n, NodeList **out)
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
	orderstmt(as, out);

	return l2;
}

static void
ordercallargs(NodeList **l, NodeList **out)
{
	if(ismulticall(*l)) {
		// return f() where f() is multiple values.
		*l = copyret((*l)->n, out);
	} else {
		orderexprlist(*l, out);
	}
}

static void
ordercall(Node *n, NodeList **out)
{
	orderexpr(&n->left, out);
	ordercallargs(&n->list, out);
}

static void
orderstmt(Node *n, NodeList **out)
{
	int lno;
	NodeList *l;
	Node *r;

	if(n == N)
		return;

	lno = setlineno(n);

	orderinit(n, out);

	switch(n->op) {
	default:
		fatal("orderstmt %O", n->op);

	case OAS2:
	case OAS2DOTTYPE:
	case OAS2MAPR:
	case OAS:
	case OASOP:
	case OCLOSE:
	case OCOPY:
	case ODELETE:
	case OPANIC:
	case OPRINT:
	case OPRINTN:
	case ORECOVER:
	case ORECV:
	case OSEND:
		orderexpr(&n->left, out);
		orderexpr(&n->right, out);
		orderexprlist(n->list, out);
		orderexprlist(n->rlist, out);
		*out = list(*out, n);
		break;
	
	case OAS2FUNC:
		// Special: avoid copy of func call n->rlist->n.
		orderexprlist(n->list, out);
		ordercall(n->rlist->n, out);
		*out = list(*out, n);
		break;

	case OAS2RECV:
		// Special: avoid copy of receive.
		orderexprlist(n->list, out);
		orderexpr(&n->rlist->n->left, out);  // arg to recv
		*out = list(*out, n);
		break;

	case OBLOCK:
	case OEMPTY:
		// Special: does not save n onto out.
		orderstmtlist(n->list, out);
		break;

	case OBREAK:
	case OCONTINUE:
	case ODCL:
	case ODCLCONST:
	case ODCLTYPE:
	case OFALL:
	case_OFALL:
	case OGOTO:
	case OLABEL:
		// Special: n->left is not an expression; save as is.
		*out = list(*out, n);
		break;

	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
		// Special: handle call arguments.
		ordercall(n, out);
		*out = list(*out, n);
		break;

	case ODEFER:
	case OPROC:
		// Special: order arguments to inner call but not call itself.
		ordercall(n->left, out);
		*out = list(*out, n);
		break;

	case OFOR:
		orderexprinplace(&n->ntest);
		orderstmtinplace(&n->nincr);
		orderblock(&n->nbody);
		*out = list(*out, n);
		break;
		
	case OIF:
		orderexprinplace(&n->ntest);
		orderblock(&n->nbody);
		orderblock(&n->nelse);
		*out = list(*out, n);
		break;

	case ORANGE:
		orderexpr(&n->right, out);
		for(l=n->list; l; l=l->next)
			orderexprinplace(&l->n);
		orderblock(&n->nbody);
		*out = list(*out, n);
		break;

	case ORETURN:
		ordercallargs(&n->list, out);
		*out = list(*out, n);
		break;
		
	case OSELECT:
		for(l=n->list; l; l=l->next) {
			if(l->n->op != OXCASE)
				fatal("order select case %O", l->n->op);
			r = l->n->left;
			if(r == nil)
				continue;
			switch(r->op) {
			case OSELRECV:
			case OSELRECV2:
				orderexprinplace(&r->left);
				orderexprinplace(&r->ntest);
				orderexpr(&r->right->left, &l->n->ninit);
				break;
			case OSEND:
				orderexpr(&r->left, &l->n->ninit);
				orderexpr(&r->right, &l->n->ninit);
				break;
			}
		}
		*out = list(*out, n);
		break;

	case OSWITCH:
		orderexpr(&n->ntest, out);
		for(l=n->list; l; l=l->next) {
			if(l->n->op != OXCASE)
				fatal("order switch case %O", l->n->op);
			orderexpr(&l->n->left, &l->n->ninit);
		}
		*out = list(*out, n);
		break;

	case OXFALL:
		yyerror("fallthrough statement out of place");
		n->op = OFALL;
		goto case_OFALL;
	}
	
	lineno = lno;
}

static void
orderexprlist(NodeList *l, NodeList **out)
{
	for(; l; l=l->next)
		orderexpr(&l->n, out);
}

static void
orderexpr(Node **np, NodeList **out)
{
	Node *n;
	int lno;

	n = *np;
	if(n == N)
		return;

	lno = setlineno(n);
	orderinit(n, out);

	switch(n->op) {
	default:
		orderexpr(&n->left, out);
		orderexpr(&n->right, out);
		orderexprlist(n->list, out);
		orderexprlist(n->rlist, out);
		break;
	
	case OANDAND:
	case OOROR:
		orderexpr(&n->left, out);
		orderexprinplace(&n->right);
		break;
	
	case OCALLFUNC:
	case OCALLMETH:
	case OCALLINTER:
		ordercall(n, out);
		n = copyexpr(n, n->type, out);
		break;

	case ORECV:
		n = copyexpr(n, n->type, out);
		break;
	}
	
	lineno = lno;

	*np = n;
}
