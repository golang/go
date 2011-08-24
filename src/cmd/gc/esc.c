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
// First escfunc, escstmt and escexpr recurse over the ast of each
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
static void escstmtlist(NodeList *stmts);
static void escstmt(Node *stmt);
static void escexpr(Node *dst, Node *expr);
static void escexprcall(Node *dst, Node *callexpr);
static void escflows(Node* dst, Node* src);
static void escflood(Node *dst);
static void escwalk(int level, Node *dst, Node *src);
static void esctag(Node *func);

// Fake node that all
//   - return values and output variables
//   - parameters on imported functions not marked 'safe'
//   - assignments to global variables
// flow to.
static Node	theSink;

static NodeList* dsts;		// all dst nodes
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
	for (l = dsts; l; l=l->next)
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
	for (ll=curfn->cvars; ll; ll=ll->next) {
		if(ll->n->op == OXXX)  // see dcl.c:398
			continue;

		n = nod(OADDR, ll->n->closure, N);
		n->lineno = ll->n->lineno;
		typecheck(&n, Erv);
		escexpr(curfn, n);
	}

	escstmtlist(curfn->nbody);
	curfn = savefn;
	loopdepth = saveld;
}

static void
escstmtlist(NodeList* stmts)
{
	for(; stmts; stmts=stmts->next)
		escstmt(stmts->n);
}

static void
escstmt(Node *stmt)
{
	int cl, cr, lno;
	NodeList *ll, *lr;
	Node *dst;

	if(stmt == N)
		return;

	lno = setlineno(stmt);

	if(stmt->typecheck == 0 && stmt->op != ODCL) {	 // TODO something with OAS2
		dump("escstmt missing typecheck", stmt);
		fatal("missing typecheck.");
	}

	// Common to almost all statements, and nil if n/a.
	escstmtlist(stmt->ninit);

	if(debug['m'] > 1)
		print("%L:[%d] %#S statement: %#N\n", lineno, loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, stmt);

	switch(stmt->op) {
	case ODCL:
	case ODCLFIELD:
		// a declaration ties the node to the current
		// function, but we already have that edge in
		// curfn->dcl and will follow it explicitly in
		// escflood to avoid storing redundant information
		// What does have to happen here is note if the name
		// is declared inside a looping scope.
		stmt->left->escloopdepth = loopdepth;
		break;

	case OLABEL:  // TODO: new loop/scope only if there are backjumps to it.
		loopdepth++;
		break;

	case OBLOCK:
		escstmtlist(stmt->list);
		break;

	case OFOR:
		if(stmt->ntest != N) {
			escstmtlist(stmt->ntest->ninit);
			escexpr(N, stmt->ntest);
		}
		escstmt(stmt->nincr);
		loopdepth++;
		escstmtlist(stmt->nbody);
		loopdepth--;
		break;

	case ORANGE:		//  for	 <list> = range <right> { <nbody> }
		switch(stmt->type->etype) {
		case TSTRING:	// never flows
			escexpr(stmt->list->n, N);
			if(stmt->list->next)
				escexpr(stmt->list->next->n, N);
			escexpr(N, stmt->right);
			break;
		case TARRAY:	// i, v = range sliceorarray
			escexpr(stmt->list->n, N);
			if(stmt->list->next)
				escexpr(stmt->list->next->n, stmt->right);
			break;
		case TMAP:	// k [, v] = range map
			escexpr(stmt->list->n, stmt->right);
			if(stmt->list->next)
				escexpr(stmt->list->next->n, stmt->right);
			break;
		case TCHAN:	// v = range chan
			escexpr(stmt->list->n, stmt->right);
			break;
		}
		loopdepth++;
		escstmtlist(stmt->nbody);
		loopdepth--;
		break;

	case OIF:
		escexpr(N, stmt->ntest);
		escstmtlist(stmt->nbody);
		escstmtlist(stmt->nelse);
		break;

	case OSELECT:
		for(ll=stmt->list; ll; ll=ll->next) {  // cases
			escstmt(ll->n->left);
			escstmtlist(ll->n->nbody);
		}
		break;

	case OSELRECV2:	  // v, ok := <-ch  ntest:ok
		escexpr(N, stmt->ntest);
		// fallthrough
	case OSELRECV:	  // v := <-ch	 left: v  right->op = ORECV
		escexpr(N, stmt->left);
		escexpr(stmt->left, stmt->right);
		break;

	case OSWITCH:
		if(stmt->ntest && stmt->ntest->op == OTYPESW) {
			for(ll=stmt->list; ll; ll=ll->next) {  // cases
				// ntest->right is the argument of the .(type),
				// ll->n->nname is the variable per case
				escexpr(ll->n->nname, stmt->ntest->right);
				escstmtlist(ll->n->nbody);
			}
		} else {
			escexpr(N, stmt->ntest);
			for(ll=stmt->list; ll; ll=ll->next) {  // cases
				for(lr=ll->n->list; lr; lr=lr->next)
					escexpr(N, lr->n);
				escstmtlist(ll->n->nbody);
			}
		}
		break;

	case OAS:
	case OASOP:
		escexpr(stmt->left, stmt->right);
		break;

		// escape analysis happens after typecheck, so the
		// OAS2xxx have already been substituted.
	case OAS2:	// x,y = a,b
		cl = count(stmt->list);
		cr = count(stmt->rlist);
		if(cl > 1 && cr == 1) {
			for(ll=stmt->list; ll; ll=ll->next)
				escexpr(ll->n, stmt->rlist->n);
		} else {
			if(cl != cr)
				fatal("escstmt: bad OAS2: %N", stmt);
			for(ll=stmt->list, lr=stmt->rlist; ll; ll=ll->next, lr=lr->next)
				escexpr(ll->n, lr->n);
		}
		break;

	case OAS2RECV:		// v, ok = <-ch
	case OAS2MAPR:		// v, ok = m[k]
	case OAS2DOTTYPE:	// v, ok = x.(type)
		escexpr(stmt->list->n, stmt->rlist->n);
		escexpr(stmt->list->next->n, N);
		break;

	case OAS2MAPW:		// m[k] = x, ok.. stmt->list->n is the INDEXMAP, k is handled in escexpr(dst...)
		escexpr(stmt->list->n, stmt->rlist->n);
		escexpr(N, stmt->rlist->next->n);
		break;

	case ORECV:		// unary <-ch as statement
		escexpr(N, stmt->left);
		break;

	case OSEND:		// ch <- x
		escexpr(&theSink, stmt->right);	 // for now. TODO escexpr(stmt->left, stmt->right);
		break;

	case OCOPY:	// todo: treat as *dst=*src instead of as dst=src
		escexpr(stmt->left, stmt->right);
		break;

	case OAS2FUNC:	// x,y,z = f()
		for(ll = stmt->list; ll; ll=ll->next)
			escexpr(ll->n, N);
		escexpr(N, stmt->rlist->n);
		break;

	case OCALLINTER:
	case OCALLFUNC:
	case OCALLMETH:
		escexpr(N, stmt);
		break;

	case OPROC:
	case ODEFER:
		// stmt->left is a (pseud)ocall, stmt->left->left is
		// the function being called.  if this defer is at
		// loopdepth >1, everything leaks.  TODO this is
		// overly conservative, it's enough if it leaks to a
		// fake node at the function's top level
		dst = &theSink;
		if (stmt->op == ODEFER && loopdepth <= 1)
			dst = nil;
		escexpr(dst, stmt->left->left);
		for(ll=stmt->left->list; ll; ll=ll->next)
			escexpr(dst, ll->n);
		break;

	case ORETURN:
		for(ll=stmt->list; ll; ll=ll->next)
			escexpr(&theSink, ll->n);
		break;

	case OCLOSE:
	case OPRINT:
	case OPRINTN:
		escexpr(N, stmt->left);
		for(ll=stmt->list; ll; ll=ll->next)
			escexpr(N, ll->n);
		break;

	case OPANIC:
		// Argument could leak through recover.
		escexpr(&theSink, stmt->left);
		break;
	}

	lineno = lno;
}

// Assert that expr somehow gets assigned to dst, if non nil.  for
// dst==nil, any name node expr still must be marked as being
// evaluated in curfn.	For expr==nil, dst must still be examined for
// evaluations inside it (e.g *f(x) = y)
static void
escexpr(Node *dst, Node *expr)
{
	int lno;
	NodeList *ll;

	if(isblank(dst)) dst = N;

	// the lhs of an assignment needs recursive analysis too
	// these are the only interesting cases
	// todo:check channel case
	if(dst) {
		setlineno(dst);

		switch(dst->op) {
		case OINDEX:
		case OSLICE:
			escexpr(N, dst->right);

			// slice:  "dst[x] = src"  is like *(underlying array)[x] = src
			// TODO maybe this never occurs b/c of OSLICEARR and it's inserted OADDR
			if(!isfixedarray(dst->left->type))
				goto doref;

			// fallthrough;	 treat "dst[x] = src" as "dst = src"
		case ODOT:	      // treat "dst.x  = src" as "dst = src"
			escexpr(dst->left, expr);
			return;

		case OINDEXMAP:
			escexpr(&theSink, dst->right);	// map key is put in map
			// fallthrough
		case OIND:
		case ODOTPTR:
		case OSLICEARR:	 // ->left  is the OADDR of the array
		doref:
			escexpr(N, dst->left);
			// assignment to dereferences: for now we lose track
			escexpr(&theSink, expr);
			return;
		}

	}

	if(expr == N || expr->op == ONONAME || expr->op == OXXX)
		return;

	if(expr->typecheck == 0 && expr->op != OKEY) {
		dump("escexpr missing typecheck", expr);
		fatal("Missing typecheck.");
	}

	lno = setlineno(expr);
	pdepth++;

	if(debug['m'] > 1)
		print("%L:[%d] %#S \t%hN %.*s<= %hN\n", lineno, loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, dst,
		      2*pdepth, ".\t.\t.\t.\t.\t", expr);


	switch(expr->op) {
	case OADDR:	// dst = &x
	case OIND:	// dst = *x
	case ODOTPTR:	// dst = (*x).f
		// restart the recursion at x to figure out where it came from
		escexpr(expr->left, expr->left);
		// fallthrough
	case ONAME:
	case OPARAM:
		// loopdepth was set in the defining statement or function header
		escflows(dst, expr);
		break;

	case OARRAYLIT:
	case OSTRUCTLIT:
	case OMAPLIT:
		expr->escloopdepth = loopdepth;
		escflows(dst, expr);
		for(ll=expr->list; ll; ll=ll->next) {
			escexpr(expr, ll->n->left);
			escexpr(expr, ll->n->right);
		}
		break;

	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case ONEW:
		expr->curfn = curfn;  // should have been done in parse, but patch it up here.
		expr->escloopdepth = loopdepth;
		escflows(dst, expr);
		// first arg is type, all others need checking
		for(ll=expr->list->next; ll; ll=ll->next)
			escexpr(N, ll->n);
		break;

	case OCLOSURE:
		expr->curfn = curfn;  // should have been done in parse, but patch it up here.
		expr->escloopdepth = loopdepth;
		escflows(dst, expr);
		escfunc(expr);
		break;

	// end of the leaf cases. no calls to escflows() in the cases below.


	case OCONV:	// unaries that pass the value through
	case OCONVIFACE:
	case OCONVNOP:
	case ODOTTYPE:
	case ODOTTYPE2:
	case ORECV:	// leaks the whole channel
	case ODOTMETH:	// expr->right is just the field or method name
	case ODOTINTER:
	case ODOT:
		escexpr(dst, expr->left);
		break;

	case OCOPY:
		// left leaks to right, but the return value is harmless
		// TODO: treat as *dst = *src, rather than as dst = src
		escexpr(expr->left, expr->right);
		break;

	case OAPPEND:
		// See TODO for OCOPY
		escexpr(dst, expr->list->n);
		for(ll=expr->list->next; ll; ll=ll->next)
			escexpr(expr->list->n, ll->n);
		break;

	case OCALLMETH:
	case OCALLFUNC:
	case OCALLINTER:
		// Moved to separate function to isolate the hair.
		escexprcall(dst, expr);
		break;

	case OSLICEARR:	 // like an implicit OIND to the underlying buffer, but typecheck has inserted an OADDR
	case OSLICESTR:
	case OSLICE:
	case OINDEX:
	case OINDEXMAP:
		// the big thing flows, the keys just need checking
		escexpr(dst, expr->left);
		escexpr(N, expr->right);  // expr->right is the OKEY
		break;

	default: // all other harmless leaf, unary or binary cases end up here
		escexpr(N, expr->left);
		escexpr(N, expr->right);
		break;
	}

	pdepth--;
	lineno = lno;
}


// This is a bit messier than fortunate, pulled out of escexpr's big
// switch for clarity.	We either have the paramnodes, which may be
// connected to other things throug flows or we have the parameter type
// nodes, which may be marked 'n(ofloworescape)'. Navigating the ast is slightly
// different for methods vs plain functions and for imported vs
// this-package
static void
escexprcall(Node *dst, Node *expr)
{
	NodeList *ll, *lr;
	Node *fn;
	Type *t, *fntype, *thisarg, *inargs;

	fn = nil;
	fntype = nil;

	switch(expr->op) {
	case OCALLFUNC:
		fn = expr->left;
		escexpr(N, fn);
		fntype = fn->type;
		break;

	case OCALLMETH:
		fn = expr->left->right;	 // ODOTxx name
		fn = fn->sym->def;	 // resolve to definition if we have it
		if(fn)
			fntype = fn->type;
		else
			fntype = expr->left->type;
		break;

	case OCALLINTER:
		break;

	default:
		fatal("escexprcall called with non-call expression");
	}

	if(fn && fn->ntype) {
		if(debug['m'] > 2)
			print("escexprcall: have param nodes: %N\n", fn->ntype);

		if(expr->op == OCALLMETH) {
			if(debug['m'] > 2)
				print("escexprcall: this: %N\n",fn->ntype->left->left);
			escexpr(fn->ntype->left->left, expr->left->left);
		}

		// lr->n is the dclfield, ->left is the ONAME param node
		for(ll=expr->list, lr=fn->ntype->list; ll && lr; ll=ll->next) {
			if(debug['m'] > 2)
				print("escexprcall: field param: %N\n", lr->n->left);
			if (lr->n->left)
				escexpr(lr->n->left, ll->n);
			else
				escexpr(&theSink, ll->n);
			if(lr->n->left && !lr->n->left->isddd)
				lr=lr->next;
		}
		return;
	}

	if(fntype) {
		if(debug['m'] > 2)
			print("escexprcall: have param types: %T\n", fntype);

		if(expr->op == OCALLMETH) {
			thisarg = getthisx(fntype);
			t = thisarg->type;
			if(debug['m'] > 2)
				print("escexprcall: this: %T\n", t);
			if(!t->note || strcmp(t->note->s, safetag->s) != 0)
				escexpr(&theSink, expr->left->left);
			else
				escexpr(N, expr->left->left);
		}

		inargs = getinargx(fntype);
		for(ll=expr->list, t=inargs->type; ll; ll=ll->next) {
			if(debug['m'] > 2)
				print("escexprcall: field type: %T\n", t);
			if(!t->note || strcmp(t->note->s, safetag->s))
				escexpr(&theSink, ll->n);
			else
				escexpr(N, ll->n);
			if(t->down)
				t=t->down;
		}

		return;
	}

	// fallthrough if we don't have enough information:
	// can only assume all parameters are unsafe
	// OCALLINTER always ends up here

	if(debug['m']>1 && expr->op != OCALLINTER) {
		// dump("escexprcall", expr);
		print("escexprcall: %O, no nodes, no types: %N\n", expr->op, fn);
	}

	escexpr(&theSink,  expr->left->left);  // the this argument
	for(ll=expr->list; ll; ll=ll->next)
		escexpr(&theSink, ll->n);
}

// Store the link src->dst in dst, throwing out some quick wins.
static void
escflows(Node* dst, Node* src)
{
	if(dst == nil || src == nil || dst == src)
		return;

	// Don't bother building a graph for scalars.
	if (src->type && !haspointers(src->type))
		return;

	if(debug['m']>2)
		print("%L::flows:: %hN <- %hN\n", lineno, dst, src);

	// Assignments to global variables get lumped into theSink.
	if (dst->op == ONAME && dst->class == PEXTERN)
		dst = &theSink;

	if (dst->escflowsrc == nil) {
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

	for (l = dst->escflowsrc; l; l=l->next) {
		floodgen++;
		escwalk(0, dst, l->n);
	}
}

static void
escwalk(int level, Node *dst, Node *src)
{
	NodeList* ll;
	int leaks;

	if (src->escfloodgen == floodgen)
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
		if (src->class == PPARAM && leaks && src->esc == EscNone) {
			src->esc = EscScope;
			if(debug['m'])
				print("%L:leaking param: %hN\n", src->lineno, src);
		}
		break;

	case OADDR:
		if (leaks)
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

	for (ll=src->escflowsrc; ll; ll=ll->next)
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
			if (haspointers(ll->n->type)) // don't bother tagging for scalars
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
