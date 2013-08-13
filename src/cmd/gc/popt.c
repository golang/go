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
#include	"gg.h"
#include	"opt.h"

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
		symlist[3] = pkglookup("panic", runtimepkg);
		symlist[4] = pkglookup("panicwrap", runtimepkg);
	}

	s = p->to.sym;
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
	while(p != P && p->as == AJMP && p->to.type == D_BRANCH) {
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

/* mark all code reachable from firstp as alive */
static void
mark(Prog *firstp)
{
	Prog *p;
	
	for(p=firstp; p; p=p->link) {
		if(p->opt != dead)
			break;
		p->opt = alive;
		if(p->as != ACALL && p->to.type == D_BRANCH && p->to.u.branch)
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
		if(p->as != ACALL && p->to.type == D_BRANCH && p->to.u.branch && p->to.u.branch->as == AJMP) {
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
				// Let it stay.
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
			if(p->as == AJMP && p->to.type == D_BRANCH && p->to.u.branch == p->link) {
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
		proginfo(&info, p);
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
		proginfo(&info, p);
		if(!(info.flags & Break)) {
			f1 = f->link;
			f->s1 = f1;
			f1->p1 = f;
		}
		if(p->to.type == D_BRANCH) {
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

