// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The base version before this file existed, active with debug['s']
// == 0, assumes any node that has a reference to it created at some
// point, may flow to the global scope except
//   - if its address is dereferenced immediately with only CONVNOPs in
//     between the * and the &
//   - if it is for a closure variable and the closure executed at the
//     place it's defined
//
// Flag -s disables the old codepaths and switches on the code here:
//
// First escfunc, esc and escassign recurse over the ast of each
// function to dig out flow(dst,src) edges between any
// pointer-containing nodes and store them in dst->escflowsrc.  For
// variables assigned to a variable in an outer scope or used as a
// return value, they store a flow(theSink, src) edge to a fake node
// 'the Sink'.	For variables referenced in closures, an edge
// flow(closure, &var) is recorded and the flow of a closure itself to
// an outer scope is tracked the same way as other variables.
//
// Then escflood walks the graph starting at theSink and tags all
// variables of it can reach an & node as escaping and all function
// parameters it can reach as leaking.
//
// Watch the variables moved to the heap and parameters tagged as
// unsafe with -m, more detailed analysis output with -mm
//

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
static int	floodgen;	// loop prevention in flood/walk
static Strlit*	safetag;	// gets slapped on safe parameters' field types for export
static int	dstcount, edgecount;	// diagnostic

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
			ll->n->esc = EscNone;	// prime for escflood later
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
	NodeList *ll, *lr, *l;

	if(n == N)
		return;

	lno = setlineno(n);

	if(n->op == OFOR)
		loopdepth++;

	esclist(n->ninit);
	esclist(n->list);
	esclist(n->rlist);
	esc(n->ntest);
	esc(n->nincr);
	esclist(n->nbody);
	esc(n->left);
	esc(n->right);

	if(n->op == OFOR)
		loopdepth--;

	if(debug['m'] > 1)
		print("%L:[%d] %#S esc: %#N\n", lineno, loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, n);

	switch(n->op) {
	case ODCL:
	case ODCLFIELD:
		// a declaration ties the node to the current
		// function, but we already have that edge in
		// curfn->dcl and will follow it explicitly in
		// escflood to avoid storing redundant information
		// What does have to happen here is note if the name
		// is declared inside a looping scope.
		if(n->left)
			n->left->escloopdepth = loopdepth;
		break;

	case OLABEL:  // TODO: new loop/scope only if there are backjumps to it.
		loopdepth++;
		break;

	case ORANGE:		//  for	 <list> = range <right> { <nbody> }
		switch(n->type->etype) {
		case TARRAY:	// i, v = range sliceorarray
			if(n->list->next)
				escassign(n->list->next->n, n->right);
			break;
		case TMAP:	// k [, v] = range map
			escassign(n->list->n, n->right);
			if(n->list->next)
				escassign(n->list->next->n, n->right);
			break;
		case TCHAN:	// v = range chan
			escassign(n->list->n, n->right);
			break;
		}
		loopdepth++;
		esclist(n->nbody);
		loopdepth--;
		break;

	case OSELRECV:	  // v := <-ch	 left: v  right->op = ORECV
		escassign(n->left, n->right);
		break;

	case OSWITCH:
		if(n->ntest && n->ntest->op == OTYPESW) {
			for(ll=n->list; ll; ll=ll->next) {  // cases
				// ntest->right is the argument of the .(type),
				// ll->n->nname is the variable per case
				escassign(ll->n->nname, n->ntest->right);
				esclist(ll->n->nbody);
			}
		} else {
			escassign(N, n->ntest);
			for(ll=n->list; ll; ll=ll->next) {  // cases
				for(lr=ll->n->list; lr; lr=lr->next)
					escassign(N, lr->n);
				esclist(ll->n->nbody);
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
	case OAS2MAPW:		// m[k] = x, ok
		escassign(n->list->n, n->rlist->n);
		break;

	case OSEND:		// ch <- x
		escassign(&theSink, n->right);	 // TODO: treat as *ch = x ?
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

	case OCOPY:
		// left leaks to right, but the return value is harmless
		// TODO: treat as *dst = *src, rather than as dst = src
		escassign(n->left, n->right);
		break;

	case OAPPEND:
		// See TODO for OCOPY
		for(ll=n->list->next; ll; ll=ll->next)
			escassign(n->list->n, ll->n);
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
	case OSTRUCTLIT:
		for(l=n->list; l; l=l->next)
			escassign(n, l->n->right);
		break;
	case OMAPLIT:
		for(l=n->list; l; l=l->next) {
			escassign(n, l->n->left);
			escassign(n, l->n->right);
		}
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
	NodeList *ll;

	if(isblank(dst) || dst == N || src == N || src->op == ONONAME || src->op == OXXX)
		return;

	if(debug['m'] > 1)
		print("%L:[%d] %#S escassign: %hN = %hN\n", lineno, loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, dst, src);

	// the lhs of an assignment needs recursive analysis too
	// these are the only interesting cases
	// todo:check channel case
	setlineno(dst);

	switch(dst->op) {
	case OINDEX:
	case OSLICE:
		// slice:  "dst[x] = src"  is like *(underlying array)[x] = src
		// TODO maybe this never occurs b/c of OSLICEARR and it's inserted OADDR
		if(!isfixedarray(dst->left->type))
			goto doref;
		// fallthrough;	 treat "dst[x] = src" as "dst = src"
	case ODOT:	      // treat "dst.x  = src" as "dst = src"
		escassign(dst->left, src);
		return;
	case OINDEXMAP:
		escassign(&theSink, dst->right);	// map key is put in map
		// fallthrough
	case OIND:
	case ODOTPTR:
	case OSLICEARR:	 // ->left  is the OADDR of the array
	doref:
		// assignment to dereferences: for now we lose track
		escassign(&theSink, src);
		return;
	}

	if(src->typecheck == 0 && src->op != OKEY) {
		dump("escassign missing typecheck", src);
		fatal("escassign");
	}

	lno = setlineno(src);
	pdepth++;

	switch(src->op) {
	case OADDR:	// dst = &x
	case OIND:	// dst = *x
	case ODOTPTR:	// dst = (*x).f
	case ONAME:
	case OPARAM:
		// loopdepth was set in the defining statement or function header
		escflows(dst, src);
		break;

	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case ODOT:
	case ODOTTYPE:
	case ODOTTYPE2:
		// Conversions, field access, slice all preserve the input value.
		escassign(dst, src->left);
		break;

	case OARRAYLIT:
	case OSTRUCTLIT:
	case OMAPLIT:
		src->escloopdepth = loopdepth;
		escflows(dst, src);
		for(ll=src->list; ll; ll=ll->next) {
			escassign(src, ll->n->left);
			escassign(src, ll->n->right);
		}
		break;

	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case ONEW:
		src->curfn = curfn;  // should have been done in parse, but patch it up here.
		src->escloopdepth = loopdepth;
		escflows(dst, src);
		break;

	case OCLOSURE:
		src->curfn = curfn;  // should have been done in parse, but patch it up here.
		src->escloopdepth = loopdepth;
		escflows(dst, src);
		escfunc(src);
		break;

	// end of the leaf cases. no calls to escflows() in the cases below.
	case OAPPEND:
		escassign(dst, src->list->n);
		break;

	case OSLICEARR:	 // like an implicit OIND to the underlying buffer, but typecheck has inserted an OADDR
	case OSLICESTR:
	case OSLICE:
	case OINDEX:
	case OINDEXMAP:
		// the big thing flows, the keys just need checking
		escassign(dst, src->left);
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
	Node *a, *fn;
	Type *t, *fntype;

	fn = N;
	fntype = T;
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
			
	if(fn && fn->ntype) {
		// Local function.  Incorporate into flow graph.

		// Receiver.
		if(n->op != OCALLFUNC)
			escassign(fn->ntype->left->left, n->left->left);

		for(ll=n->list, lr=fn->ntype->list; ll && lr; ll=ll->next) {
			if (lr->n->left)
				escassign(lr->n->left, ll->n);
			else 
				escassign(&theSink, ll->n);
			if(lr->n->left && !lr->n->left->isddd)
				lr=lr->next;
		}
		return;
	}

	// Imported function.  Use the escape tags.
	if(n->op != OCALLFUNC) {
		t = getthisx(fntype)->type;
		if(!t->note || strcmp(t->note->s, safetag->s) != 0)
			escassign(&theSink, n->left->left);
	}
	for(t=getinargx(fntype)->type; ll; ll=ll->next) {
		if(!t->note || strcmp(t->note->s, safetag->s) != 0)
			escassign(&theSink, ll->n);
		if(t->down)
			t = t->down;
	}
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

	// Assignments to global variables get lumped into theSink.
	if(dst->op == ONAME && dst->class == PEXTERN)
		dst = &theSink;

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
		print("\nescflood:%d: dst %hN scope:%#S[%d]\n", floodgen, dst,
		      (dst->curfn && dst->curfn->nname) ? dst->curfn->nname->sym : S,
		      dst->escloopdepth);

	for(l = dst->escflowsrc; l; l=l->next) {
		floodgen++;
		escwalk(0, dst, l->n);
	}
}

static void
escwalk(int level, Node *dst, Node *src)
{
	NodeList *ll;
	int leaks;

	if(src->escfloodgen == floodgen)
		return;
	src->escfloodgen = floodgen;

	if(debug['m']>1)
		print("escwalk: level:%d depth:%d %.*s %hN scope:%#S[%d]\n",
		      level, pdepth, pdepth, "\t\t\t\t\t\t\t\t\t\t", src,
		      (src->curfn && src->curfn->nname) ? src->curfn->nname->sym : S, src->escloopdepth);

	pdepth++;

	leaks = (level <= 0) && (dst->escloopdepth < src->escloopdepth);

	switch(src->op) {
	case ONAME:
		if(src->class == PPARAM && leaks && src->esc == EscNone) {
			src->esc = EscScope;
			if(debug['m'])
				print("%L:leaking param: %hN\n", src->lineno, src);
		}
		break;

	case OADDR:
		if(leaks)
			addrescapes(src->left);
		escwalk(level-1, dst, src->left);
		break;

	case OINDEX:
		if(isfixedarray(src->type))
			break;
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
		default:
			fatal("messed up escape tagging: %N::%N", curfn, ll->n);
		}
	}

	curfn = savefn;
}
