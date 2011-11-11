// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Escape analysis.
//
// First escfunc, esc and escassign recurse over the ast of each
// function to dig out flow(dst,src) edges between any
// pointer-containing nodes and store them in dst->escflowsrc.  For
// variables assigned to a variable in an outer scope or used as a
// return value, they store a flow(theSink, src) edge to a fake node
// 'the Sink'.  For variables referenced in closures, an edge
// flow(closure, &var) is recorded and the flow of a closure itself to
// an outer scope is tracked the same way as other variables.
//
// Then escflood walks the graph starting at theSink and tags all
// variables of it can reach an & node as escaping and all function
// parameters it can reach as leaking.
//
// If a value's address is taken but the address does not escape,
// then the value can stay on the stack.  If the value new(T) does
// not escape, then new(T) can be rewritten into a stack allocation.
// The same is true of slice literals.
//
// If escape analysis is disabled (-s), this code is not used.
// Instead, the compiler assumes that any value whose address
// is taken without being immediately dereferenced
// needs to be moved to the heap, and new(T) and slice
// literals are always real allocations.

#include <u.h>
#include <libc.h>
#include "go.h"

static void escfunc(Node *func);
static void esclist(NodeList *l);
static void esc(Node *n);
static void escassign(Node *dst, Node *src);
static void esccall(Node*);
static void escflows(Node *dst, Node *src);
static void escflood(Node *dst);
static void escwalk(int level, Node *dst, Node *src);
static void esctag(Node *func);

// Fake node that all
//   - return values and output variables
//   - parameters on imported functions not marked 'safe'
//   - assignments to global variables
// flow to.
static Node	theSink;

static NodeList*	dsts;		// all dst nodes
static int	loopdepth;	// for detecting nested loop scopes
static int	pdepth;		// for debug printing in recursions.
static Strlit*	safetag;	// gets slapped on safe parameters' field types for export
static int	dstcount, edgecount;	// diagnostic
static NodeList*	noesc;	// list of possible non-escaping nodes, for printing

void
escapes(void)
{
	NodeList *l;

	theSink.op = ONAME;
	theSink.class = PEXTERN;
	theSink.sym = lookup(".sink");
	theSink.escloopdepth = -1;

	safetag = strlit("noescape");

	// flow-analyze top level functions
	for(l=xtop; l; l=l->next)
		if(l->n->op == ODCLFUNC || l->n->op == OCLOSURE)
			escfunc(l->n);

	// print("escapes: %d dsts, %d edges\n", dstcount, edgecount);

	// visit the updstream of each dst, mark address nodes with
	// addrescapes, mark parameters unsafe
	for(l = dsts; l; l=l->next)
		escflood(l->n);

	// for all top level functions, tag the typenodes corresponding to the param nodes
	for(l=xtop; l; l=l->next)
		if(l->n->op == ODCLFUNC)
			esctag(l->n);

	if(debug['m']) {
		for(l=noesc; l; l=l->next)
			if(l->n->esc == EscNone)
				warnl(l->n->lineno, "%S %hN does not escape",
					(l->n->curfn && l->n->curfn->nname) ? l->n->curfn->nname->sym : S,
					l->n);
	}
}


static void
escfunc(Node *func)
{
	Node *savefn, *n;
	NodeList *ll;
	int saveld;

	saveld = loopdepth;
	loopdepth = 1;
	savefn = curfn;
	curfn = func;

	for(ll=curfn->dcl; ll; ll=ll->next) {
		if(ll->n->op != ONAME)
			continue;
		switch (ll->n->class) {
		case PPARAMOUT:
			// output parameters flow to the sink
			escflows(&theSink, ll->n);
			ll->n->escloopdepth = loopdepth;
			break;
		case PPARAM:
			if(ll->n->type && !haspointers(ll->n->type))
				break;
			ll->n->esc = EscNone;	// prime for escflood later
			noesc = list(noesc, ll->n);
			ll->n->escloopdepth = loopdepth;
			break;
		}
	}

	// walk will take the address of cvar->closure later and assign it to cvar.
	// handle that here by linking a fake oaddr node directly to the closure.
	for(ll=curfn->cvars; ll; ll=ll->next) {
		if(ll->n->op == OXXX)  // see dcl.c:398
			continue;

		n = nod(OADDR, ll->n->closure, N);
		n->lineno = ll->n->lineno;
		typecheck(&n, Erv);
		escassign(curfn, n);
	}

	esclist(curfn->nbody);
	curfn = savefn;
	loopdepth = saveld;
}

static void
esclist(NodeList *l)
{
	for(; l; l=l->next)
		esc(l->n);
}

static void
esc(Node *n)
{
	int lno;
	NodeList *ll, *lr;

	if(n == N)
		return;

	lno = setlineno(n);

	if(n->op == OFOR || n->op == ORANGE)
		loopdepth++;

	esc(n->left);
	esc(n->right);
	esc(n->ntest);
	esc(n->nincr);
	esclist(n->ninit);
	esclist(n->nbody);
	esclist(n->nelse);
	esclist(n->list);
	esclist(n->rlist);

	if(n->op == OFOR || n->op == ORANGE)
		loopdepth--;

	if(debug['m'] > 1)
		print("%L:[%d] %S esc: %N\n", lineno, loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, n);

	switch(n->op) {
	case ODCL:
		// Record loop depth at declaration.
		if(n->left)
			n->left->escloopdepth = loopdepth;
		break;

	case OLABEL:  // TODO: new loop/scope only if there are backjumps to it.
		loopdepth++;
		break;

	case ORANGE:
		// Everything but fixed array is a dereference.
		if(isfixedarray(n->type) && n->list->next)
			escassign(n->list->next->n, n->right);
		break;

	case OSWITCH:
		if(n->ntest && n->ntest->op == OTYPESW) {
			for(ll=n->list; ll; ll=ll->next) {  // cases
				// ntest->right is the argument of the .(type),
				// ll->n->nname is the variable per case
				escassign(ll->n->nname, n->ntest->right);
			}
		}
		break;

	case OAS:
	case OASOP:
		escassign(n->left, n->right);
		break;

	case OAS2:	// x,y = a,b
		if(count(n->list) == count(n->rlist))
			for(ll=n->list, lr=n->rlist; ll; ll=ll->next, lr=lr->next)
				escassign(ll->n, lr->n);
		break;

	case OAS2RECV:		// v, ok = <-ch
	case OAS2MAPR:		// v, ok = m[k]
	case OAS2DOTTYPE:	// v, ok = x.(type)
		escassign(n->list->n, n->rlist->n);
		break;

	case OSEND:		// ch <- x
		escassign(&theSink, n->right);
		break;

	case ODEFER:
		if(loopdepth == 1)  // top level
			break;
		// arguments leak out of scope
		// TODO: leak to a dummy node instead
		// fallthrough
	case OPROC:
		// go f(x) - f and x escape
		escassign(&theSink, n->left->left);
		escassign(&theSink, n->left->right);  // ODDDARG for call
		for(ll=n->left->list; ll; ll=ll->next)
			escassign(&theSink, ll->n);
		break;

	case ORETURN:
		for(ll=n->list; ll; ll=ll->next)
			escassign(&theSink, ll->n);
		break;

	case OPANIC:
		// Argument could leak through recover.
		escassign(&theSink, n->left);
		break;

	case OAPPEND:
		if(!n->isddd)
			for(ll=n->list->next; ll; ll=ll->next)
				escassign(&theSink, ll->n);  // lose track of assign to dereference
		break;

	case OCALLMETH:
	case OCALLFUNC:
	case OCALLINTER:
		esccall(n);
		break;

	case OCONV:
	case OCONVNOP:
	case OCONVIFACE:
		escassign(n, n->left);
		break;

	case OARRAYLIT:
		if(isslice(n->type)) {
			n->esc = EscNone;  // until proven otherwise
			noesc = list(noesc, n);
			n->escloopdepth = loopdepth;
			// Values make it to memory, lose track.
			for(ll=n->list; ll; ll=ll->next)
				escassign(&theSink, ll->n->right);
		} else {
			// Link values to array.
			for(ll=n->list; ll; ll=ll->next)
				escassign(n, ll->n->right);
		}
		break;

	case OSTRUCTLIT:
		// Link values to struct.
		for(ll=n->list; ll; ll=ll->next)
			escassign(n, ll->n->right);
		break;

	case OMAPLIT:
		n->esc = EscNone;  // until proven otherwise
		noesc = list(noesc, n);
		n->escloopdepth = loopdepth;
		// Keys and values make it to memory, lose track.
		for(ll=n->list; ll; ll=ll->next) {
			escassign(&theSink, ll->n->left);
			escassign(&theSink, ll->n->right);
		}
		break;
	
	case OADDR:
	case OCLOSURE:
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case ONEW:
		n->escloopdepth = loopdepth;
		n->esc = EscNone;  // until proven otherwise
		noesc = list(noesc, n);
		break;
	}

	lineno = lno;
}

// Assert that expr somehow gets assigned to dst, if non nil.  for
// dst==nil, any name node expr still must be marked as being
// evaluated in curfn.	For expr==nil, dst must still be examined for
// evaluations inside it (e.g *f(x) = y)
static void
escassign(Node *dst, Node *src)
{
	int lno;

	if(isblank(dst) || dst == N || src == N || src->op == ONONAME || src->op == OXXX)
		return;

	if(debug['m'] > 1)
		print("%L:[%d] %S escassign: %hN = %hN\n", lineno, loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, dst, src);

	setlineno(dst);
	
	// Analyze lhs of assignment.
	// Replace dst with theSink if we can't track it.
	switch(dst->op) {
	default:
		dump("dst", dst);
		fatal("escassign: unexpected dst");

	case OARRAYLIT:
	case OCLOSURE:
	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case OMAPLIT:
	case OSTRUCTLIT:
		break;

	case ONAME:
		if(dst->class == PEXTERN)
			dst = &theSink;
		break;
	case ODOT:	      // treat "dst.x  = src" as "dst = src"
		escassign(dst->left, src);
		return;
	case OINDEX:
		if(isfixedarray(dst->left->type)) {
			escassign(dst->left, src);
			return;
		}
		dst = &theSink;  // lose track of dereference
		break;
	case OIND:
	case ODOTPTR:
		dst = &theSink;  // lose track of dereference
		break;
	case OINDEXMAP:
		// lose track of key and value
		escassign(&theSink, dst->right);
		dst = &theSink;
		break;
	}

	lno = setlineno(src);
	pdepth++;

	switch(src->op) {
	case OADDR:	// dst = &x
	case OIND:	// dst = *x
	case ODOTPTR:	// dst = (*x).f
	case ONAME:
	case OPARAM:
	case ODDDARG:
	case OARRAYLIT:
	case OMAPLIT:
	case OSTRUCTLIT:
		// loopdepth was set in the defining statement or function header
		escflows(dst, src);
		break;

	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case ODOT:
	case ODOTMETH:	// treat recv.meth as a value with recv in it, only happens in ODEFER and OPROC
			// iface.method already leaks iface in esccall, no need to put in extra ODOTINTER edge here
	case ODOTTYPE:
	case ODOTTYPE2:
	case OSLICE:
	case OSLICEARR:
		// Conversions, field access, slice all preserve the input value.
		escassign(dst, src->left);
		break;

	case OAPPEND:
		// Append returns first argument.
		escassign(dst, src->list->n);
		break;
	
	case OINDEX:
		// Index of array preserves input value.
		if(isfixedarray(src->left->type))
			escassign(dst, src->left);
		break;

	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case ONEW:
		escflows(dst, src);
		break;

	case OCLOSURE:
		escflows(dst, src);
		escfunc(src);
		break;

	case OADD:
	case OSUB:
	case OOR:
	case OXOR:
	case OMUL:
	case ODIV:
	case OMOD:
	case OLSH:
	case ORSH:
	case OAND:
	case OANDNOT:
	case OPLUS:
	case OMINUS:
	case OCOM:
		// Might be pointer arithmetic, in which case
		// the operands flow into the result.
		// TODO(rsc): Decide what the story is here.  This is unsettling.
		escassign(dst, src->left);
		escassign(dst, src->right);
		break;

	}

	pdepth--;
	lineno = lno;
}


// This is a bit messier than fortunate, pulled out of escassign's big
// switch for clarity.	We either have the paramnodes, which may be
// connected to other things throug flows or we have the parameter type
// nodes, which may be marked 'n(ofloworescape)'. Navigating the ast is slightly
// different for methods vs plain functions and for imported vs
// this-package
static void
esccall(Node *n)
{
	NodeList *ll, *lr;
	Node *a, *fn, *src;
	Type *t, *fntype;

	fn = N;
	switch(n->op) {
	default:
		fatal("esccall");

	case OCALLFUNC:
		fn = n->left;
		fntype = fn->type;
		break;

	case OCALLMETH:
		fn = n->left->right->sym->def;
		if(fn)
			fntype = fn->type;
		else
			fntype = n->left->type;
		break;

	case OCALLINTER:
		fntype = n->left->type;
		break;
	}

	ll = n->list;
	if(n->list != nil && n->list->next == nil) {
		a = n->list->n;
		if(a->type->etype == TSTRUCT && a->type->funarg) {
			// f(g()).
			// Since f's arguments are g's results and
			// all function results escape, we're done.
			ll = nil;
		}
	}
			
	if(fn && fn->op == ONAME && fn->class == PFUNC && fn->defn && fn->defn->nbody && fn->ntype) {
		// Local function.  Incorporate into flow graph.

		// Receiver.
		if(n->op != OCALLFUNC)
			escassign(fn->ntype->left->left, n->left->left);

		for(lr=fn->ntype->list; ll && lr; ll=ll->next, lr=lr->next) {
			src = ll->n;
			if(lr->n->isddd && !n->isddd) {
				// Introduce ODDDARG node to represent ... allocation.
				src = nod(ODDDARG, N, N);
				src->escloopdepth = loopdepth;
				src->lineno = n->lineno;
				src->esc = EscNone;  // until we find otherwise
				noesc = list(noesc, src);
				n->right = src;
			}
			if(lr->n->left != N)
				escassign(lr->n->left, src);
			if(src != ll->n)
				break;
		}
		// "..." arguments are untracked
		for(; ll; ll=ll->next)
			escassign(&theSink, ll->n);
		return;
	}

	// Imported function.  Use the escape tags.
	if(n->op != OCALLFUNC) {
		t = getthisx(fntype)->type;
		if(!t->note || strcmp(t->note->s, safetag->s) != 0)
			escassign(&theSink, n->left->left);
	}
	for(t=getinargx(fntype)->type; ll; ll=ll->next) {
		src = ll->n;
		if(t->isddd && !n->isddd) {
			// Introduce ODDDARG node to represent ... allocation.
			src = nod(ODDDARG, N, N);
			src->escloopdepth = loopdepth;
			src->lineno = n->lineno;
			src->esc = EscNone;  // until we find otherwise
			noesc = list(noesc, src);
			n->right = src;
		}
		if(!t->note || strcmp(t->note->s, safetag->s) != 0)
			escassign(&theSink, src);
		if(src != ll->n)
			break;
		t = t->down;
	}
	// "..." arguments are untracked
	for(; ll; ll=ll->next)
		escassign(&theSink, ll->n);
}

// Store the link src->dst in dst, throwing out some quick wins.
static void
escflows(Node *dst, Node *src)
{
	if(dst == nil || src == nil || dst == src)
		return;

	// Don't bother building a graph for scalars.
	if(src->type && !haspointers(src->type))
		return;

	if(debug['m']>2)
		print("%L::flows:: %hN <- %hN\n", lineno, dst, src);

	if(dst->escflowsrc == nil) {
		dsts = list(dsts, dst);
		dstcount++;
	}
	edgecount++;

	dst->escflowsrc = list(dst->escflowsrc, src);
}

// Whenever we hit a reference node, the level goes up by one, and whenever
// we hit an OADDR, the level goes down by one. as long as we're on a level > 0
// finding an OADDR just means we're following the upstream of a dereference,
// so this address doesn't leak (yet).
// If level == 0, it means the /value/ of this node can reach the root of this flood.
// so if this node is an OADDR, it's argument should be marked as escaping iff
// it's currfn/loopdepth are different from the flood's root.
// Once an object has been moved to the heap, all of it's upstream should be considered
// escaping to the global scope.
static void
escflood(Node *dst)
{
	NodeList *l;

	switch(dst->op) {
	case ONAME:
	case OCLOSURE:
		break;
	default:
		return;
	}

	if(debug['m']>1)
		print("\nescflood:%d: dst %hN scope:%S[%d]\n", walkgen, dst,
		      (dst->curfn && dst->curfn->nname) ? dst->curfn->nname->sym : S,
		      dst->escloopdepth);

	for(l = dst->escflowsrc; l; l=l->next) {
		walkgen++;
		escwalk(0, dst, l->n);
	}
}

static void
escwalk(int level, Node *dst, Node *src)
{
	NodeList *ll;
	int leaks;

	if(src->walkgen == walkgen)
		return;
	src->walkgen = walkgen;

	if(debug['m']>1)
		print("escwalk: level:%d depth:%d %.*s %hN scope:%S[%d]\n",
		      level, pdepth, pdepth, "\t\t\t\t\t\t\t\t\t\t", src,
		      (src->curfn && src->curfn->nname) ? src->curfn->nname->sym : S, src->escloopdepth);

	pdepth++;

	leaks = (level <= 0) && (dst->escloopdepth < src->escloopdepth);

	switch(src->op) {
	case ONAME:
		if(src->class == PPARAM && leaks && src->esc == EscNone) {
			src->esc = EscScope;
			if(debug['m'])
				warnl(src->lineno, "leaking param: %hN", src);
		}
		break;

	case OADDR:
		if(leaks) {
			src->esc = EscHeap;
			addrescapes(src->left);
			if(debug['m'])
				warnl(src->lineno, "%hN escapes to heap", src);
		}
		escwalk(level-1, dst, src->left);
		break;

	case OARRAYLIT:
		if(isfixedarray(src->type))
			break;
		// fall through
	case ODDDARG:
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case OMAPLIT:
	case ONEW:
	case OCLOSURE:
		if(leaks) {
			src->esc = EscHeap;
			if(debug['m'])
				warnl(src->lineno, "%hN escapes to heap", src);
		}
		break;

	case OINDEX:
		if(isfixedarray(src->type))
			break;
		// fall through
	case OSLICE:
	case ODOTPTR:
	case OINDEXMAP:
	case OIND:
		escwalk(level+1, dst, src->left);
	}

	for(ll=src->escflowsrc; ll; ll=ll->next)
		escwalk(level, dst, ll->n);

	pdepth--;
}

static void
esctag(Node *func)
{
	Node *savefn;
	NodeList *ll;
	
	// External functions must be assumed unsafe.
	if(func->nbody == nil)
		return;

	savefn = curfn;
	curfn = func;

	for(ll=curfn->dcl; ll; ll=ll->next) {
		if(ll->n->op != ONAME || ll->n->class != PPARAM)
			continue;

		switch (ll->n->esc) {
		case EscNone:	// not touched by escflood
			if(haspointers(ll->n->type)) // don't bother tagging for scalars
				ll->n->paramfld->note = safetag;
		case EscHeap:	// touched by escflood, moved to heap
		case EscScope:	// touched by escflood, value leaves scope
			break;
		}
	}

	curfn = savefn;
}
