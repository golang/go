// Derived from Inferno utils/6c/reg.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/reg.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// "Portable" optimizations.
// Compiled separately for 5g, 6g, and 8g, so allowed to use gg.h, opt.h.
// Must code to the intersection of the three back ends.

#include	<u.h>
#include	<libc.h>
#include	"go.h"

// p is a call instruction. Does the call fail to return?
int
noreturn(Prog *p)
{
	Sym *s;
	int i;
	static Sym*	symlist[10];

	if(symlist[0] == S) {
		symlist[0] = pkglookup("panicindex", runtimepkg);
		symlist[1] = pkglookup("panicslice", runtimepkg);
		symlist[2] = pkglookup("throwinit", runtimepkg);
		symlist[3] = pkglookup("gopanic", runtimepkg);
		symlist[4] = pkglookup("panicwrap", runtimepkg);
		symlist[5] = pkglookup("throwreturn", runtimepkg);
		symlist[6] = pkglookup("selectgo", runtimepkg);
		symlist[7] = pkglookup("block", runtimepkg);
	}

	if(p->to.node == nil)
		return 0;
	s = ((Node*)(p->to.node))->sym;
	if(s == S)
		return 0;
	for(i=0; symlist[i]!=S; i++)
		if(s == symlist[i])
			return 1;
	return 0;
}

// JMP chasing and removal.
//
// The code generator depends on being able to write out jump
// instructions that it can jump to now but fill in later.
// the linker will resolve them nicely, but they make the code
// longer and more difficult to follow during debugging.
// Remove them.

/* what instruction does a JMP to p eventually land on? */
static Prog*
chasejmp(Prog *p, int *jmploop)
{
	int n;

	n = 0;
	while(p != P && p->as == AJMP && p->to.type == TYPE_BRANCH) {
		if(++n > 10) {
			*jmploop = 1;
			break;
		}
		p = p->to.u.branch;
	}
	return p;
}

/*
 * reuse reg pointer for mark/sweep state.
 * leave reg==nil at end because alive==nil.
 */
#define alive ((void*)0)
#define dead ((void*)1)
/*c2go
extern void *alive;
extern void *dead;
*/

/* mark all code reachable from firstp as alive */
static void
mark(Prog *firstp)
{
	Prog *p;
	
	for(p=firstp; p; p=p->link) {
		if(p->opt != dead)
			break;
		p->opt = alive;
		if(p->as != ACALL && p->to.type == TYPE_BRANCH && p->to.u.branch)
			mark(p->to.u.branch);
		if(p->as == AJMP || p->as == ARET || p->as == AUNDEF)
			break;
	}
}

void
fixjmp(Prog *firstp)
{
	int jmploop;
	Prog *p, *last;
	
	if(debug['R'] && debug['v'])
		print("\nfixjmp\n");

	// pass 1: resolve jump to jump, mark all code as dead.
	jmploop = 0;
	for(p=firstp; p; p=p->link) {
		if(debug['R'] && debug['v'])
			print("%P\n", p);
		if(p->as != ACALL && p->to.type == TYPE_BRANCH && p->to.u.branch && p->to.u.branch->as == AJMP) {
			p->to.u.branch = chasejmp(p->to.u.branch, &jmploop);
			if(debug['R'] && debug['v'])
				print("->%P\n", p);
		}
		p->opt = dead;
	}
	if(debug['R'] && debug['v'])
		print("\n");

	// pass 2: mark all reachable code alive
	mark(firstp);
	
	// pass 3: delete dead code (mostly JMPs).
	last = nil;
	for(p=firstp; p; p=p->link) {
		if(p->opt == dead) {
			if(p->link == P && p->as == ARET && last && last->as != ARET) {
				// This is the final ARET, and the code so far doesn't have one.
				// Let it stay. The register allocator assumes that all live code in
				// the function can be traversed by starting at all the RET instructions
				// and following predecessor links. If we remove the final RET,
				// this assumption will not hold in the case of an infinite loop
				// at the end of a function.
				// Keep the RET but mark it dead for the liveness analysis.
				p->mode = 1;
			} else {
				if(debug['R'] && debug['v'])
					print("del %P\n", p);
				continue;
			}
		}
		if(last)
			last->link = p;
		last = p;
	}
	last->link = P;
	
	// pass 4: elide JMP to next instruction.
	// only safe if there are no jumps to JMPs anymore.
	if(!jmploop) {
		last = nil;
		for(p=firstp; p; p=p->link) {
			if(p->as == AJMP && p->to.type == TYPE_BRANCH && p->to.u.branch == p->link) {
				if(debug['R'] && debug['v'])
					print("del %P\n", p);
				continue;
			}
			if(last)
				last->link = p;
			last = p;
		}
		last->link = P;
	}
	
	if(debug['R'] && debug['v']) {
		print("\n");
		for(p=firstp; p; p=p->link)
			print("%P\n", p);
		print("\n");
	}
}

#undef alive
#undef dead

// Control flow analysis. The Flow structures hold predecessor and successor
// information as well as basic loop analysis.
//
//	graph = flowstart(firstp, sizeof(Flow));
//	... use flow graph ...
//	flowend(graph); // free graph
//
// Typical uses of the flow graph are to iterate over all the flow-relevant instructions:
//
//	for(f = graph->start; f != nil; f = f->link)
//
// or, given an instruction f, to iterate over all the predecessors, which is
// f->p1 and this list:
//
//	for(f2 = f->p2; f2 != nil; f2 = f2->p2link)
//	
// Often the Flow struct is embedded as the first field inside a larger struct S.
// In that case casts are needed to convert Flow* to S* in many places but the
// idea is the same. Pass sizeof(S) instead of sizeof(Flow) to flowstart.

Graph*
flowstart(Prog *firstp, int size)
{
	int nf;
	Flow *f, *f1, *start, *last;
	Graph *graph;
	Prog *p;
	ProgInfo info;

	// Count and mark instructions to annotate.
	nf = 0;
	for(p = firstp; p != P; p = p->link) {
		p->opt = nil; // should be already, but just in case
		arch.proginfo(&info, p);
		if(info.flags & Skip)
			continue;
		p->opt = (void*)1;
		nf++;
	}
	
	if(nf == 0)
		return nil;

	if(nf >= 20000) {
		// fatal("%S is too big (%d instructions)", curfn->nname->sym, nf);
		return nil;
	}

	// Allocate annotations and assign to instructions.
	graph = calloc(sizeof *graph + size*nf, 1);
	if(graph == nil)
		fatal("out of memory");
	start = (Flow*)(graph+1);
	last = nil;
	f = start;
	for(p = firstp; p != P; p = p->link) {
		if(p->opt == nil)
			continue;
		p->opt = f;
		f->prog = p;
		if(last)
			last->link = f;
		last = f;
		
		f = (Flow*)((uchar*)f + size);
	}

	// Fill in pred/succ information.
	for(f = start; f != nil; f = f->link) {
		p = f->prog;
		arch.proginfo(&info, p);
		if(!(info.flags & Break)) {
			f1 = f->link;
			f->s1 = f1;
			f1->p1 = f;
		}
		if(p->to.type == TYPE_BRANCH) {
			if(p->to.u.branch == P)
				fatal("pnil %P", p);
			f1 = p->to.u.branch->opt;
			if(f1 == nil)
				fatal("fnil %P / %P", p, p->to.u.branch);
			if(f1 == f) {
				//fatal("self loop %P", p);
				continue;
			}
			f->s2 = f1;
			f->p2link = f1->p2;
			f1->p2 = f;
		}
	}
	
	graph->start = start;
	graph->num = nf;
	return graph;
}

void
flowend(Graph *graph)
{
	Flow *f;
	
	for(f = graph->start; f != nil; f = f->link)
		f->prog->opt = nil;
	free(graph);
}

/*
 * find looping structure
 *
 * 1) find reverse postordering
 * 2) find approximate dominators,
 *	the actual dominators if the flow graph is reducible
 *	otherwise, dominators plus some other non-dominators.
 *	See Matthew S. Hecht and Jeffrey D. Ullman,
 *	"Analysis of a Simple Algorithm for Global Data Flow Problems",
 *	Conf.  Record of ACM Symp. on Principles of Prog. Langs, Boston, Massachusetts,
 *	Oct. 1-3, 1973, pp.  207-217.
 * 3) find all nodes with a predecessor dominated by the current node.
 *	such a node is a loop head.
 *	recursively, all preds with a greater rpo number are in the loop
 */
static int32
postorder(Flow *r, Flow **rpo2r, int32 n)
{
	Flow *r1;

	r->rpo = 1;
	r1 = r->s1;
	if(r1 && !r1->rpo)
		n = postorder(r1, rpo2r, n);
	r1 = r->s2;
	if(r1 && !r1->rpo)
		n = postorder(r1, rpo2r, n);
	rpo2r[n] = r;
	n++;
	return n;
}

static int32
rpolca(int32 *idom, int32 rpo1, int32 rpo2)
{
	int32 t;

	if(rpo1 == -1)
		return rpo2;
	while(rpo1 != rpo2){
		if(rpo1 > rpo2){
			t = rpo2;
			rpo2 = rpo1;
			rpo1 = t;
		}
		while(rpo1 < rpo2){
			t = idom[rpo2];
			if(t >= rpo2)
				fatal("bad idom");
			rpo2 = t;
		}
	}
	return rpo1;
}

static int
doms(int32 *idom, int32 r, int32 s)
{
	while(s > r)
		s = idom[s];
	return s == r;
}

static int
loophead(int32 *idom, Flow *r)
{
	int32 src;

	src = r->rpo;
	if(r->p1 != nil && doms(idom, src, r->p1->rpo))
		return 1;
	for(r = r->p2; r != nil; r = r->p2link)
		if(doms(idom, src, r->rpo))
			return 1;
	return 0;
}

enum {
	LOOP = 3,
};

static void
loopmark(Flow **rpo2r, int32 head, Flow *r)
{
	if(r->rpo < head || r->active == head)
		return;
	r->active = head;
	r->loop += LOOP;
	if(r->p1 != nil)
		loopmark(rpo2r, head, r->p1);
	for(r = r->p2; r != nil; r = r->p2link)
		loopmark(rpo2r, head, r);
}

void
flowrpo(Graph *g)
{
	Flow *r1;
	int32 i, d, me, nr, *idom;
	Flow **rpo2r;

	free(g->rpo);
	g->rpo = calloc(g->num*sizeof g->rpo[0], 1);
	idom = calloc(g->num*sizeof idom[0], 1);
	if(g->rpo == nil || idom == nil)
		fatal("out of memory");

	for(r1 = g->start; r1 != nil; r1 = r1->link)
		r1->active = 0;

	rpo2r = g->rpo;
	d = postorder(g->start, rpo2r, 0);
	nr = g->num;
	if(d > nr)
		fatal("too many reg nodes %d %d", d, nr);
	nr = d;
	for(i = 0; i < nr / 2; i++) {
		r1 = rpo2r[i];
		rpo2r[i] = rpo2r[nr - 1 - i];
		rpo2r[nr - 1 - i] = r1;
	}
	for(i = 0; i < nr; i++)
		rpo2r[i]->rpo = i;

	idom[0] = 0;
	for(i = 0; i < nr; i++) {
		r1 = rpo2r[i];
		me = r1->rpo;
		d = -1;
		// rpo2r[r->rpo] == r protects against considering dead code,
		// which has r->rpo == 0.
		if(r1->p1 != nil && rpo2r[r1->p1->rpo] == r1->p1 && r1->p1->rpo < me)
			d = r1->p1->rpo;
		for(r1 = r1->p2; r1 != nil; r1 = r1->p2link)
			if(rpo2r[r1->rpo] == r1 && r1->rpo < me)
				d = rpolca(idom, d, r1->rpo);
		idom[i] = d;
	}

	for(i = 0; i < nr; i++) {
		r1 = rpo2r[i];
		r1->loop++;
		if(r1->p2 != nil && loophead(idom, r1))
			loopmark(rpo2r, i, r1);
	}
	free(idom);

	for(r1 = g->start; r1 != nil; r1 = r1->link)
		r1->active = 0;
}

Flow*
uniqp(Flow *r)
{
	Flow *r1;

	r1 = r->p1;
	if(r1 == nil) {
		r1 = r->p2;
		if(r1 == nil || r1->p2link != nil)
			return nil;
	} else
		if(r->p2 != nil)
			return nil;
	return r1;
}

Flow*
uniqs(Flow *r)
{
	Flow *r1;

	r1 = r->s1;
	if(r1 == nil) {
		r1 = r->s2;
		if(r1 == nil)
			return nil;
	} else
		if(r->s2 != nil)
			return nil;
	return r1;
}

// The compilers assume they can generate temporary variables
// as needed to preserve the right semantics or simplify code
// generation and the back end will still generate good code.
// This results in a large number of ephemeral temporary variables.
// Merge temps with non-overlapping lifetimes and equal types using the
// greedy algorithm in Poletto and Sarkar, "Linear Scan Register Allocation",
// ACM TOPLAS 1999.

typedef struct TempVar TempVar;
typedef struct TempFlow TempFlow;

struct TempVar
{
	Node *node;
	TempFlow *def; // definition of temp var
	TempFlow *use; // use list, chained through TempFlow.uselink
	TempVar *freelink; // next free temp in Type.opt list
	TempVar *merge; // merge var with this one
	vlong start; // smallest Prog.pc in live range
	vlong end; // largest Prog.pc in live range
	uchar addr; // address taken - no accurate end
	uchar removed; // removed from program
};

struct TempFlow
{
	Flow	f;
	TempFlow *uselink;
};

static int
startcmp(const void *va, const void *vb)
{
	TempVar *a, *b;
	
	a = *(TempVar**)va;
	b = *(TempVar**)vb;

	if(a->start < b->start)
		return -1;
	if(a->start > b->start)
		return +1;
	return 0;
}

// Is n available for merging?
static int
canmerge(Node *n)
{
	return n->class == PAUTO && strncmp(n->sym->name, "autotmp", 7) == 0;
}

static void mergewalk(TempVar*, TempFlow*, uint32);
static void varkillwalk(TempVar*, TempFlow*, uint32);

void
mergetemp(Prog *firstp)
{
	int i, j, nvar, ninuse, nfree, nkill;
	TempVar *var, *v, *v1, **bystart, **inuse;
	TempFlow *r;
	NodeList *l, **lp;
	Node *n;
	Prog *p, *p1;
	Type *t;
	ProgInfo info, info1;
	int32 gen;
	Graph *g;

	enum { Debug = 0 };

	g = flowstart(firstp, sizeof(TempFlow));
	if(g == nil)
		return;
	
	// Build list of all mergeable variables.
	nvar = 0;
	for(l = curfn->dcl; l != nil; l = l->next)
		if(canmerge(l->n))
			nvar++;
	
	var = calloc(nvar*sizeof var[0], 1);
	nvar = 0;
	for(l = curfn->dcl; l != nil; l = l->next) {
		n = l->n;
		if(canmerge(n)) {
			v = &var[nvar++];
			n->opt = v;
			v->node = n;
		}
	}
	
	// Build list of uses.
	// We assume that the earliest reference to a temporary is its definition.
	// This is not true of variables in general but our temporaries are all
	// single-use (that's why we have so many!).
	for(r = (TempFlow*)g->start; r != nil; r = (TempFlow*)r->f.link) {
		p = r->f.prog;
		arch.proginfo(&info, p);

		if(p->from.node != N && ((Node*)(p->from.node))->opt && p->to.node != N && ((Node*)(p->to.node))->opt)
			fatal("double node %P", p);
		v = nil;
		if((n = p->from.node) != N)
			v = n->opt;
		if(v == nil && (n = p->to.node) != N)
			v = n->opt;
		if(v != nil) {
		   	if(v->def == nil)
		   		v->def = r;
			r->uselink = v->use;
			v->use = r;
			if(n == p->from.node && (info.flags & LeftAddr))
				v->addr = 1;
		}
	}
	
	if(Debug > 1)
		arch.dumpit("before", g->start, 0);
	
	nkill = 0;

	// Special case.
	for(v = var; v < var+nvar; v++) {
		if(v->addr)
			continue;
		// Used in only one instruction, which had better be a write.
		if((r = v->use) != nil && r->uselink == nil) {
			p = r->f.prog;
			arch.proginfo(&info, p);
			if(p->to.node == v->node && (info.flags & RightWrite) && !(info.flags & RightRead)) {
				p->as = ANOP;
				p->to = zprog.to;
				v->removed = 1;
				if(Debug)
					print("drop write-only %S\n", v->node->sym);
			} else
				fatal("temp used and not set: %P", p);
			nkill++;
			continue;
		}
		
		// Written in one instruction, read in the next, otherwise unused,
		// no jumps to the next instruction. Happens mainly in 386 compiler.
		if((r = v->use) != nil && r->f.link == &r->uselink->f && r->uselink->uselink == nil && uniqp(r->f.link) == &r->f) {
			p = r->f.prog;
			arch.proginfo(&info, p);
			p1 = r->f.link->prog;
			arch.proginfo(&info1, p1);
			enum {
				SizeAny = SizeB | SizeW | SizeL | SizeQ | SizeF | SizeD,
			};
			if(p->from.node == v->node && p1->to.node == v->node && (info.flags & Move) &&
			   !((info.flags|info1.flags) & (LeftAddr|RightAddr)) &&
			   (info.flags & SizeAny) == (info1.flags & SizeAny)) {
				p1->from = p->from;
				arch.excise(&r->f);
				v->removed = 1;
				if(Debug)
					print("drop immediate-use %S\n", v->node->sym);
			}
			nkill++;
			continue;
		}			   
	}

	// Traverse live range of each variable to set start, end.
	// Each flood uses a new value of gen so that we don't have
	// to clear all the r->f.active words after each variable.
	gen = 0;
	for(v = var; v < var+nvar; v++) {
		gen++;
		for(r = v->use; r != nil; r = r->uselink)
			mergewalk(v, r, gen);
		if(v->addr) {
			gen++;
			for(r = v->use; r != nil; r = r->uselink)
				varkillwalk(v, r, gen);
		}
	}

	// Sort variables by start.
	bystart = malloc(nvar*sizeof bystart[0]);
	for(i=0; i<nvar; i++)
		bystart[i] = &var[i];
	qsort(bystart, nvar, sizeof bystart[0], startcmp);

	// List of in-use variables, sorted by end, so that the ones that
	// will last the longest are the earliest ones in the array.
	// The tail inuse[nfree:] holds no-longer-used variables.
	// In theory we should use a sorted tree so that insertions are
	// guaranteed O(log n) and then the loop is guaranteed O(n log n).
	// In practice, it doesn't really matter.
	inuse = malloc(nvar*sizeof inuse[0]);
	ninuse = 0;
	nfree = nvar;
	for(i=0; i<nvar; i++) {
		v = bystart[i];
		if(v->removed)
			continue;

		// Expire no longer in use.
		while(ninuse > 0 && inuse[ninuse-1]->end < v->start) {
			v1 = inuse[--ninuse];
			inuse[--nfree] = v1;
		}

		// Find old temp to reuse if possible.
		t = v->node->type;
		for(j=nfree; j<nvar; j++) {
			v1 = inuse[j];
			// Require the types to match but also require the addrtaken bits to match.
			// If a variable's address is taken, that disables registerization for the individual
			// words of the variable (for example, the base,len,cap of a slice).
			// We don't want to merge a non-addressed var with an addressed one and
			// inhibit registerization of the former.
			if(eqtype(t, v1->node->type) && v->node->addrtaken == v1->node->addrtaken) {
				inuse[j] = inuse[nfree++];
				if(v1->merge)
					v->merge = v1->merge;
				else
					v->merge = v1;
				nkill++;
				break;
			}
		}

		// Sort v into inuse.
		j = ninuse++;
		while(j > 0 && inuse[j-1]->end < v->end) {
			inuse[j] = inuse[j-1];
			j--;
		}
		inuse[j] = v;
	}

	if(Debug) {
		print("%S [%d - %d]\n", curfn->nname->sym, nvar, nkill);
		for(v=var; v<var+nvar; v++) {
			print("var %#N %T %lld-%lld", v->node, v->node->type, v->start, v->end);
			if(v->addr)
				print(" addr=1");
			if(v->removed)
				print(" dead=1");
			if(v->merge)
				print(" merge %#N", v->merge->node);
			if(v->start == v->end)
				print(" %P", v->def->f.prog);
			print("\n");
		}
	
		if(Debug > 1)
			arch.dumpit("after", g->start, 0);
	}

	// Update node references to use merged temporaries.
	for(r = (TempFlow*)g->start; r != nil; r = (TempFlow*)r->f.link) {
		p = r->f.prog;
		if((n = p->from.node) != N && (v = n->opt) != nil && v->merge != nil)
			p->from.node = v->merge->node;
		if((n = p->to.node) != N && (v = n->opt) != nil && v->merge != nil)
			p->to.node = v->merge->node;
	}

	// Delete merged nodes from declaration list.
	for(lp = &curfn->dcl; (l = *lp); ) {
		curfn->dcl->end = l;
		n = l->n;
		v = n->opt;
		if(v && (v->merge || v->removed)) {
			*lp = l->next;
			continue;
		}
		lp = &l->next;
	}

	// Clear aux structures.
	for(v=var; v<var+nvar; v++)
		v->node->opt = nil;
	free(var);
	free(bystart);
	free(inuse);
	flowend(g);
}

static void
mergewalk(TempVar *v, TempFlow *r0, uint32 gen)
{
	Prog *p;
	TempFlow *r1, *r, *r2;
	
	for(r1 = r0; r1 != nil; r1 = (TempFlow*)r1->f.p1) {
		if(r1->f.active == gen)
			break;
		r1->f.active = gen;
		p = r1->f.prog;
		if(v->end < p->pc)
			v->end = p->pc;
		if(r1 == v->def) {
			v->start = p->pc;
			break;
		}
	}
	
	for(r = r0; r != r1; r = (TempFlow*)r->f.p1)
		for(r2 = (TempFlow*)r->f.p2; r2 != nil; r2 = (TempFlow*)r2->f.p2link)
			mergewalk(v, r2, gen);
}

static void
varkillwalk(TempVar *v, TempFlow *r0, uint32 gen)
{
	Prog *p;
	TempFlow *r1, *r;
	
	for(r1 = r0; r1 != nil; r1 = (TempFlow*)r1->f.s1) {
		if(r1->f.active == gen)
			break;
		r1->f.active = gen;
		p = r1->f.prog;
		if(v->end < p->pc)
			v->end = p->pc;
		if(v->start > p->pc)
			v->start = p->pc;
		if(p->as == ARET || (p->as == AVARKILL && p->to.node == v->node))
			break;
	}
	
	for(r = r0; r != r1; r = (TempFlow*)r->f.s1)
		varkillwalk(v, (TempFlow*)r->f.s2, gen);
}

// Eliminate redundant nil pointer checks.
//
// The code generation pass emits a CHECKNIL for every possibly nil pointer.
// This pass removes a CHECKNIL if every predecessor path has already
// checked this value for nil.
//
// Simple backwards flood from check to definition.
// Run prog loop backward from end of program to beginning to avoid quadratic
// behavior removing a run of checks.
//
// Assume that stack variables with address not taken can be loaded multiple times
// from memory without being rechecked. Other variables need to be checked on
// each load.
	
typedef struct NilVar NilVar;
typedef struct NilFlow NilFlow;

struct NilFlow {
	Flow f;
	int kill;
};

static void nilwalkback(NilFlow *rcheck);
static void nilwalkfwd(NilFlow *rcheck);

void
nilopt(Prog *firstp)
{
	NilFlow *r;
	Prog *p;
	Graph *g;
	int ncheck, nkill;

	g = flowstart(firstp, sizeof(NilFlow));
	if(g == nil)
		return;

	if(debug_checknil > 1 /* || strcmp(curfn->nname->sym->name, "f1") == 0 */)
		arch.dumpit("nilopt", g->start, 0);

	ncheck = 0;
	nkill = 0;
	for(r = (NilFlow*)g->start; r != nil; r = (NilFlow*)r->f.link) {
		p = r->f.prog;
		if(p->as != ACHECKNIL || !arch.regtyp(&p->from))
			continue;
		ncheck++;
		if(arch.stackaddr(&p->from)) {
			if(debug_checknil && p->lineno > 1)
				warnl(p->lineno, "removed nil check of SP address");
			r->kill = 1;
			continue;
		}
		nilwalkfwd(r);
		if(r->kill) {
			if(debug_checknil && p->lineno > 1)
				warnl(p->lineno, "removed nil check before indirect");
			continue;
		}
		nilwalkback(r);
		if(r->kill) {
			if(debug_checknil && p->lineno > 1)
				warnl(p->lineno, "removed repeated nil check");
			continue;
		}
	}
	
	for(r = (NilFlow*)g->start; r != nil; r = (NilFlow*)r->f.link) {
		if(r->kill) {
			nkill++;
			arch.excise(&r->f);
		}
	}

	flowend(g);
	
	if(debug_checknil > 1)
		print("%S: removed %d of %d nil checks\n", curfn->nname->sym, nkill, ncheck);
}

static void
nilwalkback(NilFlow *rcheck)
{
	Prog *p;
	ProgInfo info;
	NilFlow *r;
	
	for(r = rcheck; r != nil; r = (NilFlow*)uniqp(&r->f)) {
		p = r->f.prog;
		arch.proginfo(&info, p);
		if((info.flags & RightWrite) && arch.sameaddr(&p->to, &rcheck->f.prog->from)) {
			// Found initialization of value we're checking for nil.
			// without first finding the check, so this one is unchecked.
			return;
		}
		if(r != rcheck && p->as == ACHECKNIL && arch.sameaddr(&p->from, &rcheck->f.prog->from)) {
			rcheck->kill = 1;
			return;
		}
	}

	// Here is a more complex version that scans backward across branches.
	// It assumes rcheck->kill = 1 has been set on entry, and its job is to find a reason
	// to keep the check (setting rcheck->kill = 0).
	// It doesn't handle copying of aggregates as well as I would like,
	// nor variables with their address taken,
	// and it's too subtle to turn on this late in Go 1.2. Perhaps for Go 1.3.
	/*
	for(r1 = r0; r1 != nil; r1 = (NilFlow*)r1->f.p1) {
		if(r1->f.active == gen)
			break;
		r1->f.active = gen;
		p = r1->f.prog;
		
		// If same check, stop this loop but still check
		// alternate predecessors up to this point.
		if(r1 != rcheck && p->as == ACHECKNIL && arch.sameaddr(&p->from, &rcheck->f.prog->from))
			break;

		arch.proginfo(&info, p);
		if((info.flags & RightWrite) && arch.sameaddr(&p->to, &rcheck->f.prog->from)) {
			// Found initialization of value we're checking for nil.
			// without first finding the check, so this one is unchecked.
			rcheck->kill = 0;
			return;
		}
		
		if(r1->f.p1 == nil && r1->f.p2 == nil) {
			print("lost pred for %P\n", rcheck->f.prog);
			for(r1=r0; r1!=nil; r1=(NilFlow*)r1->f.p1) {
				arch.proginfo(&info, r1->f.prog);
				print("\t%P %d %d %D %D\n", r1->f.prog, info.flags&RightWrite, arch.sameaddr(&r1->f.prog->to, &rcheck->f.prog->from), &r1->f.prog->to, &rcheck->f.prog->from);
			}
			fatal("lost pred trail");
		}
	}

	for(r = r0; r != r1; r = (NilFlow*)r->f.p1)
		for(r2 = (NilFlow*)r->f.p2; r2 != nil; r2 = (NilFlow*)r2->f.p2link)
			nilwalkback(rcheck, r2, gen);
	*/
}

static void
nilwalkfwd(NilFlow *rcheck)
{
	NilFlow *r, *last;
	Prog *p;
	ProgInfo info;
	
	// If the path down from rcheck dereferences the address
	// (possibly with a small offset) before writing to memory
	// and before any subsequent checks, it's okay to wait for
	// that implicit check. Only consider this basic block to
	// avoid problems like:
	//	_ = *x // should panic
	//	for {} // no writes but infinite loop may be considered visible
	last = nil;
	for(r = (NilFlow*)uniqs(&rcheck->f); r != nil; r = (NilFlow*)uniqs(&r->f)) {
		p = r->f.prog;
		arch.proginfo(&info, p);
		
		if((info.flags & LeftRead) && arch.smallindir(&p->from, &rcheck->f.prog->from)) {
			rcheck->kill = 1;
			return;
		}
		if((info.flags & (RightRead|RightWrite)) && arch.smallindir(&p->to, &rcheck->f.prog->from)) {
			rcheck->kill = 1;
			return;
		}
		
		// Stop if another nil check happens.
		if(p->as == ACHECKNIL)
			return;
		// Stop if value is lost.
		if((info.flags & RightWrite) && arch.sameaddr(&p->to, &rcheck->f.prog->from))
			return;
		// Stop if memory write.
		if((info.flags & RightWrite) && !arch.regtyp(&p->to))
			return;
		// Stop if we jump backward.
		// This test is valid because all the NilFlow* are pointers into
		// a single contiguous array. We will need to add an explicit
		// numbering when the code is converted to Go.
		if(last != nil && r <= last)
			return;
		last = r;
	}
}
