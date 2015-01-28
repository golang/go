// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Escape analysis.

#include <u.h>
#include <libc.h>
#include "go.h"

// Run analysis on minimal sets of mutually recursive functions
// or single non-recursive functions, bottom up.
//
// Finding these sets is finding strongly connected components
// in the static call graph.  The algorithm for doing that is taken
// from Sedgewick, Algorithms, Second Edition, p. 482, with two
// adaptations.
//
// First, a hidden closure function (n->curfn != N) cannot be the
// root of a connected component. Refusing to use it as a root
// forces it into the component of the function in which it appears.
// The analysis assumes that closures and the functions in which they
// appear are analyzed together, so that the aliasing between their
// variables can be modeled more precisely.
//
// Second, each function becomes two virtual nodes in the graph,
// with numbers n and n+1. We record the function's node number as n
// but search from node n+1. If the search tells us that the component
// number (min) is n+1, we know that this is a trivial component: one function
// plus its closures. If the search tells us that the component number is
// n, then there was a path from node n+1 back to node n, meaning that
// the function set is mutually recursive. The escape analysis can be
// more precise when analyzing a single non-recursive function than
// when analyzing a set of mutually recursive functions.

static NodeList *stack;
static uint32 visitgen;
static uint32 visit(Node*);
static uint32 visitcode(Node*, uint32);
static uint32 visitcodelist(NodeList*, uint32);

static void analyze(NodeList*, int);

enum
{
	EscFuncUnknown = 0,
	EscFuncPlanned,
	EscFuncStarted,
	EscFuncTagged,
};

void
escapes(NodeList *all)
{
	NodeList *l;

	for(l=all; l; l=l->next)
		l->n->walkgen = 0;

	visitgen = 0;
	for(l=all; l; l=l->next)
		if(l->n->op == ODCLFUNC && l->n->curfn == N)
			visit(l->n);

	for(l=all; l; l=l->next)
		l->n->walkgen = 0;
}

static uint32
visit(Node *n)
{
	uint32 min, recursive;
	NodeList *l, *block;

	if(n->walkgen > 0) {
		// already visited
		return n->walkgen;
	}
	
	visitgen++;
	n->walkgen = visitgen;
	visitgen++;
	min = visitgen;

	l = mal(sizeof *l);
	l->next = stack;
	l->n = n;
	stack = l;
	min = visitcodelist(n->nbody, min);
	if((min == n->walkgen || min == n->walkgen+1) && n->curfn == N) {
		// This node is the root of a strongly connected component.

		// The original min passed to visitcodelist was n->walkgen+1.
		// If visitcodelist found its way back to n->walkgen, then this
		// block is a set of mutually recursive functions.
		// Otherwise it's just a lone function that does not recurse.
		recursive = min == n->walkgen;

		// Remove connected component from stack.
		// Mark walkgen so that future visits return a large number
		// so as not to affect the caller's min.
		block = stack;
		for(l=stack; l->n != n; l=l->next)
			l->n->walkgen = (uint32)~0U;
		n->walkgen = (uint32)~0U;
		stack = l->next;
		l->next = nil;

		// Run escape analysis on this set of functions.
		analyze(block, recursive);
	}

	return min;
}

static uint32
visitcodelist(NodeList *l, uint32 min)
{
	for(; l; l=l->next)
		min = visitcode(l->n, min);
	return min;
}

static uint32
visitcode(Node *n, uint32 min)
{
	Node *fn;
	uint32 m;

	if(n == N)
		return min;

	min = visitcodelist(n->ninit, min);
	min = visitcode(n->left, min);
	min = visitcode(n->right, min);
	min = visitcodelist(n->list, min);
	min = visitcode(n->ntest, min);
	min = visitcode(n->nincr, min);
	min = visitcodelist(n->nbody, min);
	min = visitcodelist(n->nelse, min);
	min = visitcodelist(n->rlist, min);
	
	if(n->op == OCALLFUNC || n->op == OCALLMETH) {
		fn = n->left;
		if(n->op == OCALLMETH)
			fn = n->left->right->sym->def;
		if(fn && fn->op == ONAME && fn->class == PFUNC && fn->defn)
			if((m = visit(fn->defn)) < min)
				min = m;
	}
	
	if(n->op == OCLOSURE)
		if((m = visit(n->closure)) < min)
			min = m;

	return min;
}

// An escape analysis pass for a set of functions.
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
// If optimizations are disabled (-N), this code is not used.
// Instead, the compiler assumes that any value whose address
// is taken without being immediately dereferenced
// needs to be moved to the heap, and new(T) and slice
// literals are always real allocations.

typedef struct EscState EscState;

static void escfunc(EscState*, Node *func);
static void esclist(EscState*, NodeList *l, Node *up);
static void esc(EscState*, Node *n, Node *up);
static void escloopdepthlist(EscState*, NodeList *l);
static void escloopdepth(EscState*, Node *n);
static void escassign(EscState*, Node *dst, Node *src);
static void esccall(EscState*, Node*, Node *up);
static void escflows(EscState*, Node *dst, Node *src);
static void escflood(EscState*, Node *dst);
static void escwalk(EscState*, int level, Node *dst, Node *src);
static void esctag(EscState*, Node *func);

struct EscState {
	// Fake node that all
	//   - return values and output variables
	//   - parameters on imported functions not marked 'safe'
	//   - assignments to global variables
	// flow to.
	Node	theSink;
	
	// If an analyzed function is recorded to return
	// pieces obtained via indirection from a parameter,
	// and later there is a call f(x) to that function,
	// we create a link funcParam <- x to record that fact.
	// The funcParam node is handled specially in escflood.
	Node	funcParam;	
	
	NodeList*	dsts;		// all dst nodes
	int	loopdepth;	// for detecting nested loop scopes
	int	pdepth;		// for debug printing in recursions.
	int	dstcount, edgecount;	// diagnostic
	NodeList*	noesc;	// list of possible non-escaping nodes, for printing
	int	recursive;	// recursive function or group of mutually recursive functions.
};

static Strlit *tags[16];

static Strlit*
mktag(int mask)
{
	Strlit *s;
	char buf[40];

	switch(mask&EscMask) {
	case EscNone:
	case EscReturn:
		break;
	default:
		fatal("escape mktag");
	}

	mask >>= EscBits;

	if(mask < nelem(tags) && tags[mask] != nil)
		return tags[mask];

	snprint(buf, sizeof buf, "esc:0x%x", mask);
	s = newstrlit(buf);
	if(mask < nelem(tags))
		tags[mask] = s;
	return s;
}

static int
parsetag(Strlit *note)
{
	int em;

	if(note == nil)
		return EscUnknown;
	if(strncmp(note->s, "esc:", 4) != 0)
		return EscUnknown;
	em = atoi(note->s + 4);
	if (em == 0)
		return EscNone;
	return EscReturn | (em << EscBits);
}

static void
analyze(NodeList *all, int recursive)
{
	NodeList *l;
	EscState es, *e;
	
	memset(&es, 0, sizeof es);
	e = &es;
	e->theSink.op = ONAME;
	e->theSink.orig = &e->theSink;
	e->theSink.class = PEXTERN;
	e->theSink.sym = lookup(".sink");
	e->theSink.escloopdepth = -1;
	e->recursive = recursive;
	
	e->funcParam.op = ONAME;
	e->funcParam.orig = &e->funcParam;
	e->funcParam.class = PAUTO;
	e->funcParam.sym = lookup(".param");
	e->funcParam.escloopdepth = 10000000;
	
	for(l=all; l; l=l->next)
		if(l->n->op == ODCLFUNC)
			l->n->esc = EscFuncPlanned;

	// flow-analyze functions
	for(l=all; l; l=l->next)
		if(l->n->op == ODCLFUNC)
			escfunc(e, l->n);

	// print("escapes: %d e->dsts, %d edges\n", e->dstcount, e->edgecount);

	// visit the upstream of each dst, mark address nodes with
	// addrescapes, mark parameters unsafe
	for(l = e->dsts; l; l=l->next)
		escflood(e, l->n);

	// for all top level functions, tag the typenodes corresponding to the param nodes
	for(l=all; l; l=l->next)
		if(l->n->op == ODCLFUNC)
			esctag(e, l->n);

	if(debug['m']) {
		for(l=e->noesc; l; l=l->next)
			if(l->n->esc == EscNone)
				warnl(l->n->lineno, "%S %hN does not escape",
					(l->n->curfn && l->n->curfn->nname) ? l->n->curfn->nname->sym : S,
					l->n);
	}
}


static void
escfunc(EscState *e, Node *func)
{
	Node *savefn;
	NodeList *ll;
	int saveld;

//	print("escfunc %N %s\n", func->nname, e->recursive?"(recursive)":"");

	if(func->esc != 1)
		fatal("repeat escfunc %N", func->nname);
	func->esc = EscFuncStarted;

	saveld = e->loopdepth;
	e->loopdepth = 1;
	savefn = curfn;
	curfn = func;

	for(ll=curfn->dcl; ll; ll=ll->next) {
		if(ll->n->op != ONAME)
			continue;
		switch (ll->n->class) {
		case PPARAMOUT:
			// out params are in a loopdepth between the sink and all local variables
			ll->n->escloopdepth = 0;
			break;
		case PPARAM:
			ll->n->escloopdepth = 1; 
			if(ll->n->type && !haspointers(ll->n->type))
				break;
			if(curfn->nbody == nil && !curfn->noescape)
				ll->n->esc = EscHeap;
			else
				ll->n->esc = EscNone;	// prime for escflood later
			e->noesc = list(e->noesc, ll->n);
			break;
		}
	}

	// in a mutually recursive group we lose track of the return values
	if(e->recursive)
		for(ll=curfn->dcl; ll; ll=ll->next)
			if(ll->n->op == ONAME && ll->n->class == PPARAMOUT)
				escflows(e, &e->theSink, ll->n);

	escloopdepthlist(e, curfn->nbody);
	esclist(e, curfn->nbody, curfn);
	curfn = savefn;
	e->loopdepth = saveld;
}

// Mark labels that have no backjumps to them as not increasing e->loopdepth.
// Walk hasn't generated (goto|label)->left->sym->label yet, so we'll cheat
// and set it to one of the following two.  Then in esc we'll clear it again.
static Label looping;
static Label nonlooping;

static void
escloopdepthlist(EscState *e, NodeList *l)
{
	for(; l; l=l->next)
		escloopdepth(e, l->n);
}

static void
escloopdepth(EscState *e, Node *n)
{
	if(n == N)
		return;

	escloopdepthlist(e, n->ninit);

	switch(n->op) {
	case OLABEL:
		if(!n->left || !n->left->sym)
			fatal("esc:label without label: %+N", n);
		// Walk will complain about this label being already defined, but that's not until
		// after escape analysis. in the future, maybe pull label & goto analysis out of walk and put before esc
		// if(n->left->sym->label != nil)
		//	fatal("escape analysis messed up analyzing label: %+N", n);
		n->left->sym->label = &nonlooping;
		break;
	case OGOTO:
		if(!n->left || !n->left->sym)
			fatal("esc:goto without label: %+N", n);
		// If we come past one that's uninitialized, this must be a (harmless) forward jump
		// but if it's set to nonlooping the label must have preceded this goto.
		if(n->left->sym->label == &nonlooping)
			n->left->sym->label = &looping;
		break;
	}

	escloopdepth(e, n->left);
	escloopdepth(e, n->right);
	escloopdepthlist(e, n->list);
	escloopdepth(e, n->ntest);
	escloopdepth(e, n->nincr);
	escloopdepthlist(e, n->nbody);
	escloopdepthlist(e, n->nelse);
	escloopdepthlist(e, n->rlist);

}

static void
esclist(EscState *e, NodeList *l, Node *up)
{
	for(; l; l=l->next)
		esc(e, l->n, up);
}

static void
esc(EscState *e, Node *n, Node *up)
{
	int lno;
	NodeList *ll, *lr;
	Node *a;

	if(n == N)
		return;

	lno = setlineno(n);

	// ninit logically runs at a different loopdepth than the rest of the for loop.
	esclist(e, n->ninit, n);

	if(n->op == OFOR || n->op == ORANGE)
		e->loopdepth++;

	// type switch variables have no ODCL.
	// process type switch as declaration.
	// must happen before processing of switch body,
	// so before recursion.
	if(n->op == OSWITCH && n->ntest && n->ntest->op == OTYPESW) {
		for(ll=n->list; ll; ll=ll->next) {  // cases
			// ll->n->nname is the variable per case
			if(ll->n->nname)
				ll->n->nname->escloopdepth = e->loopdepth;
		}
	}

	esc(e, n->left, n);
	esc(e, n->right, n);
	esc(e, n->ntest, n);
	esc(e, n->nincr, n);
	esclist(e, n->nbody, n);
	esclist(e, n->nelse, n);
	esclist(e, n->list, n);
	esclist(e, n->rlist, n);

	if(n->op == OFOR || n->op == ORANGE)
		e->loopdepth--;

	if(debug['m'] > 1)
		print("%L:[%d] %S esc: %N\n", lineno, e->loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, n);

	switch(n->op) {
	case ODCL:
		// Record loop depth at declaration.
		if(n->left)
			n->left->escloopdepth = e->loopdepth;
		break;

	case OLABEL:
		if(n->left->sym->label == &nonlooping) {
			if(debug['m'] > 1)
				print("%L:%N non-looping label\n", lineno, n);
		} else if(n->left->sym->label == &looping) {
			if(debug['m'] > 1)
				print("%L: %N looping label\n", lineno, n);
			e->loopdepth++;
		}
		// See case OLABEL in escloopdepth above
		// else if(n->left->sym->label == nil)
		//	fatal("escape analysis missed or messed up a label: %+N", n);

		n->left->sym->label = nil;
		break;

	case ORANGE:
		// Everything but fixed array is a dereference.
		if(isfixedarray(n->type) && n->list && n->list->next)
			escassign(e, n->list->next->n, n->right);
		break;

	case OSWITCH:
		if(n->ntest && n->ntest->op == OTYPESW) {
			for(ll=n->list; ll; ll=ll->next) {  // cases
				// ntest->right is the argument of the .(type),
				// ll->n->nname is the variable per case
				escassign(e, ll->n->nname, n->ntest->right);
			}
		}
		break;

	case OAS:
	case OASOP:
		// Filter out the following special case.
		//
		//	func (b *Buffer) Foo() {
		//		n, m := ...
		//		b.buf = b.buf[n:m]
		//	}
		//
		// This assignment is a no-op for escape analysis,
		// it does not store any new pointers into b that were not already there.
		// However, without this special case b will escape, because we assign to OIND/ODOTPTR.
		if((n->left->op == OIND || n->left->op == ODOTPTR) && n->left->left->op == ONAME && // dst is ONAME dereference
			(n->right->op == OSLICE || n->right->op == OSLICE3 || n->right->op == OSLICESTR) && // src is slice operation
			(n->right->left->op == OIND || n->right->left->op == ODOTPTR) && n->right->left->left->op == ONAME && // slice is applied to ONAME dereference
			n->left->left == n->right->left->left) { // dst and src reference the same base ONAME
			// Here we also assume that the statement will not contain calls,
			// that is, that order will move any calls to init.
			// Otherwise base ONAME value could change between the moments
			// when we evaluate it for dst and for src.
			//
			// Note, this optimization does not apply to OSLICEARR,
			// because it does introduce a new pointer into b that was not already there
			// (pointer to b itself). After such assignment, if b contents escape,
			// b escapes as well. If we ignore such OSLICEARR, we will conclude
			// that b does not escape when b contents do.
			if(debug['m']) {
				warnl(n->lineno, "%S ignoring self-assignment to %hN",
					(n->curfn && n->curfn->nname) ? n->curfn->nname->sym : S, n->left);
			}
			break;
		}
		escassign(e, n->left, n->right);
		break;

	case OAS2:	// x,y = a,b
		if(count(n->list) == count(n->rlist))
			for(ll=n->list, lr=n->rlist; ll; ll=ll->next, lr=lr->next)
				escassign(e, ll->n, lr->n);
		break;

	case OAS2RECV:		// v, ok = <-ch
	case OAS2MAPR:		// v, ok = m[k]
	case OAS2DOTTYPE:	// v, ok = x.(type)
		escassign(e, n->list->n, n->rlist->n);
		break;

	case OSEND:		// ch <- x
		escassign(e, &e->theSink, n->right);
		break;

	case ODEFER:
		if(e->loopdepth == 1)  // top level
			break;
		// arguments leak out of scope
		// TODO: leak to a dummy node instead
		// fallthrough
	case OPROC:
		// go f(x) - f and x escape
		escassign(e, &e->theSink, n->left->left);
		escassign(e, &e->theSink, n->left->right);  // ODDDARG for call
		for(ll=n->left->list; ll; ll=ll->next)
			escassign(e, &e->theSink, ll->n);
		break;

	case OCALLMETH:
	case OCALLFUNC:
	case OCALLINTER:
		esccall(e, n, up);
		break;

	case OAS2FUNC:	// x,y = f()
		// esccall already done on n->rlist->n. tie it's escretval to n->list
		lr=n->rlist->n->escretval;
		for(ll=n->list; lr && ll; lr=lr->next, ll=ll->next)
			escassign(e, ll->n, lr->n);
		if(lr || ll)
			fatal("esc oas2func");
		break;

	case ORETURN:
		ll=n->list;
		if(count(n->list) == 1 && curfn->type->outtuple > 1) {
			// OAS2FUNC in disguise
			// esccall already done on n->list->n
			// tie n->list->n->escretval to curfn->dcl PPARAMOUT's
			ll = n->list->n->escretval;
		}

		for(lr = curfn->dcl; lr && ll; lr=lr->next) {
			if (lr->n->op != ONAME || lr->n->class != PPARAMOUT)
				continue;
			escassign(e, lr->n, ll->n);
			ll = ll->next;
		}
		if (ll != nil)
			fatal("esc return list");
		break;

	case OPANIC:
		// Argument could leak through recover.
		escassign(e, &e->theSink, n->left);
		break;

	case OAPPEND:
		if(!n->isddd)
			for(ll=n->list->next; ll; ll=ll->next)
				escassign(e, &e->theSink, ll->n);  // lose track of assign to dereference
		break;

	case OCONV:
	case OCONVNOP:
	case OCONVIFACE:
		escassign(e, n, n->left);
		break;

	case OARRAYLIT:
		if(isslice(n->type)) {
			n->esc = EscNone;  // until proven otherwise
			e->noesc = list(e->noesc, n);
			n->escloopdepth = e->loopdepth;
			// Values make it to memory, lose track.
			for(ll=n->list; ll; ll=ll->next)
				escassign(e, &e->theSink, ll->n->right);
		} else {
			// Link values to array.
			for(ll=n->list; ll; ll=ll->next)
				escassign(e, n, ll->n->right);
		}
		break;

	case OSTRUCTLIT:
		// Link values to struct.
		for(ll=n->list; ll; ll=ll->next)
			escassign(e, n, ll->n->right);
		break;

	case OPTRLIT:
		n->esc = EscNone;  // until proven otherwise
		e->noesc = list(e->noesc, n);
		n->escloopdepth = e->loopdepth;
		// Link OSTRUCTLIT to OPTRLIT; if OPTRLIT escapes, OSTRUCTLIT elements do too.
		escassign(e, n, n->left);
		break;

	case OCALLPART:
		n->esc = EscNone; // until proven otherwise
		e->noesc = list(e->noesc, n);
		n->escloopdepth = e->loopdepth;
		// Contents make it to memory, lose track.
		escassign(e, &e->theSink, n->left);
		break;

	case OMAPLIT:
		n->esc = EscNone;  // until proven otherwise
		e->noesc = list(e->noesc, n);
		n->escloopdepth = e->loopdepth;
		// Keys and values make it to memory, lose track.
		for(ll=n->list; ll; ll=ll->next) {
			escassign(e, &e->theSink, ll->n->left);
			escassign(e, &e->theSink, ll->n->right);
		}
		break;
	
	case OCLOSURE:
		// Link addresses of captured variables to closure.
		for(ll=n->cvars; ll; ll=ll->next) {
			if(ll->n->op == OXXX)  // unnamed out argument; see dcl.c:/^funcargs
				continue;
			a = nod(OADDR, ll->n->closure, N);
			a->lineno = ll->n->lineno;
			a->escloopdepth = e->loopdepth;
			typecheck(&a, Erv);
			escassign(e, n, a);
		}
		// fallthrough
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case ONEW:
		n->escloopdepth = e->loopdepth;
		n->esc = EscNone;  // until proven otherwise
		e->noesc = list(e->noesc, n);
		break;

	case OARRAYBYTESTR:
	case ORUNESTR:
		n->escloopdepth = e->loopdepth;
		n->esc = EscNone;  // until proven otherwise
		e->noesc = list(e->noesc, n);
		break;

	case OADDSTR:
		n->escloopdepth = e->loopdepth;
		n->esc = EscNone;  // until proven otherwise
		e->noesc = list(e->noesc, n);
		// Arguments of OADDSTR do not escape.
		break;

	case OADDR:
		n->esc = EscNone;  // until proven otherwise
		e->noesc = list(e->noesc, n);
		// current loop depth is an upper bound on actual loop depth
		// of addressed value.
		n->escloopdepth = e->loopdepth;
		// for &x, use loop depth of x if known.
		// it should always be known, but if not, be conservative
		// and keep the current loop depth.
		if(n->left->op == ONAME) {
			switch(n->left->class) {
			case PAUTO:
				if(n->left->escloopdepth != 0)
					n->escloopdepth = n->left->escloopdepth;
				break;
			case PPARAM:
			case PPARAMOUT:
				// PPARAM is loop depth 1 always.
				// PPARAMOUT is loop depth 0 for writes
				// but considered loop depth 1 for address-of,
				// so that writing the address of one result
				// to another (or the same) result makes the
				// first result move to the heap.
				n->escloopdepth = 1;
				break;
			}
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
escassign(EscState *e, Node *dst, Node *src)
{
	int lno;
	NodeList *ll;

	if(isblank(dst) || dst == N || src == N || src->op == ONONAME || src->op == OXXX)
		return;

	if(debug['m'] > 1)
		print("%L:[%d] %S escassign: %hN(%hJ) = %hN(%hJ)\n", lineno, e->loopdepth,
		      (curfn && curfn->nname) ? curfn->nname->sym : S, dst, dst, src, src);

	setlineno(dst);
	
	// Analyze lhs of assignment.
	// Replace dst with e->theSink if we can't track it.
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
	case OPTRLIT:
	case OCALLPART:
		break;

	case ONAME:
		if(dst->class == PEXTERN)
			dst = &e->theSink;
		break;
	case ODOT:	      // treat "dst.x  = src" as "dst = src"
		escassign(e, dst->left, src);
		return;
	case OINDEX:
		if(isfixedarray(dst->left->type)) {
			escassign(e, dst->left, src);
			return;
		}
		dst = &e->theSink;  // lose track of dereference
		break;
	case OIND:
	case ODOTPTR:
		dst = &e->theSink;  // lose track of dereference
		break;
	case OINDEXMAP:
		// lose track of key and value
		escassign(e, &e->theSink, dst->right);
		dst = &e->theSink;
		break;
	}

	lno = setlineno(src);
	e->pdepth++;

	switch(src->op) {
	case OADDR:	// dst = &x
	case OIND:	// dst = *x
	case ODOTPTR:	// dst = (*x).f
	case ONAME:
	case OPARAM:
	case ODDDARG:
	case OPTRLIT:
	case OARRAYLIT:
	case OMAPLIT:
	case OSTRUCTLIT:
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case OARRAYBYTESTR:
	case OADDSTR:
	case ONEW:
	case OCLOSURE:
	case OCALLPART:
	case ORUNESTR:
		escflows(e, dst, src);
		break;

	case OCALLMETH:
	case OCALLFUNC:
	case OCALLINTER:
		// Flowing multiple returns to a single dst happens when
		// analyzing "go f(g())": here g() flows to sink (issue 4529).
		for(ll=src->escretval; ll; ll=ll->next)
			escflows(e, dst, ll->n);
		break;

	case ODOT:
		// A non-pointer escaping from a struct does not concern us.
		if(src->type && !haspointers(src->type))
			break;
		// fallthrough
	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case ODOTMETH:	// treat recv.meth as a value with recv in it, only happens in ODEFER and OPROC
			// iface.method already leaks iface in esccall, no need to put in extra ODOTINTER edge here
	case ODOTTYPE:
	case ODOTTYPE2:
	case OSLICE:
	case OSLICE3:
	case OSLICEARR:
	case OSLICE3ARR:
	case OSLICESTR:
		// Conversions, field access, slice all preserve the input value.
		escassign(e, dst, src->left);
		break;

	case OAPPEND:
		// Append returns first argument.
		escassign(e, dst, src->list->n);
		break;
	
	case OINDEX:
		// Index of array preserves input value.
		if(isfixedarray(src->left->type))
			escassign(e, dst, src->left);
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
		escassign(e, dst, src->left);
		escassign(e, dst, src->right);
		break;
	}

	e->pdepth--;
	lineno = lno;
}

static int
escassignfromtag(EscState *e, Strlit *note, NodeList *dsts, Node *src)
{
	int em, em0;
	
	em = parsetag(note);

	if(em == EscUnknown) {
		escassign(e, &e->theSink, src);
		return em;
	}

	if(em == EscNone)
		return em;
	
	// If content inside parameter (reached via indirection)
	// escapes back to results, mark as such.
	if(em & EscContentEscapes)
		escassign(e, &e->funcParam, src);

	em0 = em;
	for(em >>= EscReturnBits; em && dsts; em >>= 1, dsts=dsts->next)
		if(em & 1)
			escassign(e, dsts->n, src);

	if (em != 0 && dsts == nil)
		fatal("corrupt esc tag %Z or messed up escretval list\n", note);
	return em0;
}

// This is a bit messier than fortunate, pulled out of esc's big
// switch for clarity.	We either have the paramnodes, which may be
// connected to other things through flows or we have the parameter type
// nodes, which may be marked "noescape". Navigating the ast is slightly
// different for methods vs plain functions and for imported vs
// this-package
static void
esccall(EscState *e, Node *n, Node *up)
{
	NodeList *ll, *lr;
	Node *a, *fn, *src;
	Type *t, *fntype;
	char buf[40];
	int i;

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
		if(a->type->etype == TSTRUCT && a->type->funarg) // f(g()).
			ll = a->escretval;
	}

	if(fn && fn->op == ONAME && fn->class == PFUNC && fn->defn && fn->defn->nbody && fn->ntype && fn->defn->esc < EscFuncTagged) {
		// function in same mutually recursive group.  Incorporate into flow graph.
//		print("esc local fn: %N\n", fn->ntype);
		if(fn->defn->esc == EscFuncUnknown || n->escretval != nil)
			fatal("graph inconsistency");

		// set up out list on this call node
		for(lr=fn->ntype->rlist; lr; lr=lr->next)
			n->escretval = list(n->escretval, lr->n->left);  // type.rlist ->  dclfield -> ONAME (PPARAMOUT)

		// Receiver.
		if(n->op != OCALLFUNC)
			escassign(e, fn->ntype->left->left, n->left->left);

		for(lr=fn->ntype->list; ll && lr; ll=ll->next, lr=lr->next) {
			src = ll->n;
			if(lr->n->isddd && !n->isddd) {
				// Introduce ODDDARG node to represent ... allocation.
				src = nod(ODDDARG, N, N);
				src->type = typ(TARRAY);
				src->type->type = lr->n->type->type;
				src->type->bound = count(ll);
				src->type = ptrto(src->type); // make pointer so it will be tracked
				src->escloopdepth = e->loopdepth;
				src->lineno = n->lineno;
				src->esc = EscNone;  // until we find otherwise
				e->noesc = list(e->noesc, src);
				n->right = src;
			}
			if(lr->n->left != N)
				escassign(e, lr->n->left, src);
			if(src != ll->n)
				break;
		}
		// "..." arguments are untracked
		for(; ll; ll=ll->next)
			escassign(e, &e->theSink, ll->n);

		return;
	}

	// Imported or completely analyzed function.  Use the escape tags.
	if(n->escretval != nil)
		fatal("esc already decorated call %+N\n", n);

	// set up out list on this call node with dummy auto ONAMES in the current (calling) function.
	i = 0;
	for(t=getoutargx(fntype)->type; t; t=t->down) {
		src = nod(ONAME, N, N);
		snprint(buf, sizeof buf, ".dum%d", i++);
		src->sym = lookup(buf);
		src->type = t->type;
		src->class = PAUTO;
		src->curfn = curfn;
		src->escloopdepth = e->loopdepth;
		src->used = 1;
		src->lineno = n->lineno;
		n->escretval = list(n->escretval, src); 
	}

//	print("esc analyzed fn: %#N (%+T) returning (%+H)\n", fn, fntype, n->escretval);

	// Receiver.
	if(n->op != OCALLFUNC) {
		t = getthisx(fntype)->type;
		src = n->left->left;
		if(haspointers(t->type))
			escassignfromtag(e, t->note, n->escretval, src);
	}
	
	for(t=getinargx(fntype)->type; ll; ll=ll->next) {
		src = ll->n;
		if(t->isddd && !n->isddd) {
			// Introduce ODDDARG node to represent ... allocation.
			src = nod(ODDDARG, N, N);
			src->escloopdepth = e->loopdepth;
			src->lineno = n->lineno;
			src->type = typ(TARRAY);
			src->type->type = t->type->type;
			src->type->bound = count(ll);
			src->type = ptrto(src->type); // make pointer so it will be tracked
			src->esc = EscNone;  // until we find otherwise
			e->noesc = list(e->noesc, src);
			n->right = src;
		}
		if(haspointers(t->type)) {
			if(escassignfromtag(e, t->note, n->escretval, src) == EscNone && up->op != ODEFER && up->op != OPROC) {
				a = src;
				while(a->op == OCONVNOP)
					a = a->left;
				switch(a->op) {
				case OCALLPART:
				case OCLOSURE:
				case ODDDARG:
				case OARRAYLIT:
				case OPTRLIT:
				case OSTRUCTLIT:
					// The callee has already been analyzed, so its arguments have esc tags.
					// The argument is marked as not escaping at all.
					// Record that fact so that any temporary used for
					// synthesizing this expression can be reclaimed when
					// the function returns.
					// This 'noescape' is even stronger than the usual esc == EscNone.
					// src->esc == EscNone means that src does not escape the current function.
					// src->noescape = 1 here means that src does not escape this statement
					// in the current function.
					a->noescape = 1;
					break;
				}
			}
		}
		if(src != ll->n)
			break;
		t = t->down;
	}
	// "..." arguments are untracked
	for(; ll; ll=ll->next)
		escassign(e, &e->theSink, ll->n);
}

// Store the link src->dst in dst, throwing out some quick wins.
static void
escflows(EscState *e, Node *dst, Node *src)
{
	if(dst == nil || src == nil || dst == src)
		return;

	// Don't bother building a graph for scalars.
	if(src->type && !haspointers(src->type))
		return;

	if(debug['m']>2)
		print("%L::flows:: %hN <- %hN\n", lineno, dst, src);

	if(dst->escflowsrc == nil) {
		e->dsts = list(e->dsts, dst);
		e->dstcount++;
	}
	e->edgecount++;

	dst->escflowsrc = list(dst->escflowsrc, src);
}

// Whenever we hit a reference node, the level goes up by one, and whenever
// we hit an OADDR, the level goes down by one. as long as we're on a level > 0
// finding an OADDR just means we're following the upstream of a dereference,
// so this address doesn't leak (yet).
// If level == 0, it means the /value/ of this node can reach the root of this flood.
// so if this node is an OADDR, it's argument should be marked as escaping iff
// it's currfn/e->loopdepth are different from the flood's root.
// Once an object has been moved to the heap, all of it's upstream should be considered
// escaping to the global scope.
static void
escflood(EscState *e, Node *dst)
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
		escwalk(e, 0, dst, l->n);
	}
}

// There appear to be some loops in the escape graph, causing
// arbitrary recursion into deeper and deeper levels.
// Cut this off safely by making minLevel sticky: once you
// get that deep, you cannot go down any further but you also
// cannot go up any further. This is a conservative fix.
// Making minLevel smaller (more negative) would handle more
// complex chains of indirections followed by address-of operations,
// at the cost of repeating the traversal once for each additional
// allowed level when a loop is encountered. Using -2 suffices to
// pass all the tests we have written so far, which we assume matches
// the level of complexity we want the escape analysis code to handle.
#define MinLevel (-2)
/*c2go enum { MinLevel = -2 };*/

static void
escwalk(EscState *e, int level, Node *dst, Node *src)
{
	NodeList *ll;
	int leaks, newlevel;

	if(src->walkgen == walkgen && src->esclevel <= level)
		return;
	src->walkgen = walkgen;
	src->esclevel = level;

	if(debug['m']>1)
		print("escwalk: level:%d depth:%d %.*s %hN(%hJ) scope:%S[%d]\n",
		      level, e->pdepth, e->pdepth, "\t\t\t\t\t\t\t\t\t\t", src, src,
		      (src->curfn && src->curfn->nname) ? src->curfn->nname->sym : S, src->escloopdepth);

	e->pdepth++;

	// Input parameter flowing to output parameter?
	if(dst->op == ONAME && dst->class == PPARAMOUT && dst->vargen <= 20) {
		if(src->op == ONAME && src->class == PPARAM && src->curfn == dst->curfn && src->esc != EscScope && src->esc != EscHeap) {
			if(level == 0) {
				if(debug['m'])
					warnl(src->lineno, "leaking param: %hN to result %S", src, dst->sym);
				if((src->esc&EscMask) != EscReturn)
					src->esc = EscReturn;
				src->esc |= 1<<((dst->vargen-1) + EscReturnBits);
				goto recurse;
			} else if(level > 0) {
				if(debug['m'])
					warnl(src->lineno, "%N leaking param %hN content to result %S", src->curfn->nname, src, dst->sym);
				if((src->esc&EscMask) != EscReturn)
					src->esc = EscReturn;
				src->esc |= EscContentEscapes;
				goto recurse;
			}
		}
	}

	// The second clause is for values pointed at by an object passed to a call
	// that returns something reached via indirect from the object.
	// We don't know which result it is or how many indirects, so we treat it as leaking.
	leaks = level <= 0 && dst->escloopdepth < src->escloopdepth ||
		level < 0 && dst == &e->funcParam && haspointers(src->type);

	switch(src->op) {
	case ONAME:
		if(src->class == PPARAM && (leaks || dst->escloopdepth < 0) && src->esc != EscHeap) {
			src->esc = EscScope;
			if(debug['m'])
				warnl(src->lineno, "leaking param: %hN", src);
		}

		// Treat a PPARAMREF closure variable as equivalent to the
		// original variable.
		if(src->class == PPARAMREF) {
			if(leaks && debug['m'])
				warnl(src->lineno, "leaking closure reference %hN", src);
			escwalk(e, level, dst, src->closure);
		}
		break;

	case OPTRLIT:
	case OADDR:
		if(leaks) {
			src->esc = EscHeap;
			addrescapes(src->left);
			if(debug['m'])
				warnl(src->lineno, "%hN escapes to heap", src);
		}
		newlevel = level;
		if(level > MinLevel)
			newlevel--;
		escwalk(e, newlevel, dst, src->left);
		break;

	case OARRAYLIT:
		if(isfixedarray(src->type))
			break;
		// fall through
	case ODDDARG:
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case OARRAYBYTESTR:
	case OADDSTR:
	case OMAPLIT:
	case ONEW:
	case OCLOSURE:
	case OCALLPART:
	case ORUNESTR:
		if(leaks) {
			src->esc = EscHeap;
			if(debug['m'])
				warnl(src->lineno, "%hN escapes to heap", src);
		}
		break;

	case ODOT:
	case OSLICE:
	case OSLICEARR:
	case OSLICE3:
	case OSLICE3ARR:
	case OSLICESTR:
		escwalk(e, level, dst, src->left);
		break;

	case OINDEX:
		if(isfixedarray(src->left->type)) {
			escwalk(e, level, dst, src->left);
			break;
		}
		// fall through
	case ODOTPTR:
	case OINDEXMAP:
	case OIND:
		newlevel = level;
		if(level > MinLevel)
			newlevel++;
		escwalk(e, newlevel, dst, src->left);
	}

recurse:
	for(ll=src->escflowsrc; ll; ll=ll->next)
		escwalk(e, level, dst, ll->n);

	e->pdepth--;
}

static void
esctag(EscState *e, Node *func)
{
	Node *savefn;
	NodeList *ll;
	Type *t;

	USED(e);
	func->esc = EscFuncTagged;
	
	// External functions are assumed unsafe,
	// unless //go:noescape is given before the declaration.
	if(func->nbody == nil) {
		if(func->noescape) {
			for(t=getinargx(func->type)->type; t; t=t->down)
				if(haspointers(t->type))
					t->note = mktag(EscNone);
		}
		return;
	}

	savefn = curfn;
	curfn = func;

	for(ll=curfn->dcl; ll; ll=ll->next) {
		if(ll->n->op != ONAME || ll->n->class != PPARAM)
			continue;

		switch (ll->n->esc&EscMask) {
		case EscNone:	// not touched by escflood
		case EscReturn:	
			if(haspointers(ll->n->type)) // don't bother tagging for scalars
				ll->n->paramfld->note = mktag(ll->n->esc);
			break;
		case EscHeap:	// touched by escflood, moved to heap
		case EscScope:	// touched by escflood, value leaves scope
			break;
		}
	}

	curfn = savefn;
}
