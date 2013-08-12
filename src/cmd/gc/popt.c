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
