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

#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"

void
regopt(Prog *p)
{
	USED(p);
	// TODO(minux)
	return;
}

/*
 * track register variables including external registers:
 *	bit	reg
 *	0	R0
 *	1	R1
 *	...	...
 *	31	R31
 *	32+0	F0
 *	32+1	F1
 *	...	...
 *	32+31	F31
 */
uint64
RtoB(int r)
{
	if(r >= D_R0 && r <= D_R0+31)
		return 1ULL << (r - D_R0);
	return 0;
}

int
BtoR(uint64 b)
{
	b &= 0xffffffff;
	if(b == 0)
		return 0;
	return bitno(b) + D_R0;
}

uint64
FtoB(int r)
{
	if(r >= D_F0 && r <= D_F0+31)
		return 1ULL << (32 + r - D_F0);
	return 0;
}

int
BtoF(uint64 b)
{
	b >>= 32;
	if(b == 0)
		return 0;
	return bitno(b) + D_F0;
}

void
dumpone(Flow *f, int isreg)
{
	int z;
	Bits bit;
	Reg *r;

	print("%d:%P", f->loop, f->prog);
	if(isreg) {	
		r = (Reg*)f;
		for(z=0; z<BITS; z++)
			bit.b[z] =
				r->set.b[z] |
				r->use1.b[z] |
				r->use2.b[z] |
				r->refbehind.b[z] |
				r->refahead.b[z] |
				r->calbehind.b[z] |
				r->calahead.b[z] |
				r->regdiff.b[z] |
				r->act.b[z] |
					0;
		if(bany(&bit)) {
			print("\t");
			if(bany(&r->set))
				print(" s:%Q", r->set);
			if(bany(&r->use1))
				print(" u1:%Q", r->use1);
			if(bany(&r->use2))
				print(" u2:%Q", r->use2);
			if(bany(&r->refbehind))
				print(" rb:%Q ", r->refbehind);
			if(bany(&r->refahead))
				print(" ra:%Q ", r->refahead);
			if(bany(&r->calbehind))
				print(" cb:%Q ", r->calbehind);
			if(bany(&r->calahead))
				print(" ca:%Q ", r->calahead);
			if(bany(&r->regdiff))
				print(" d:%Q ", r->regdiff);
			if(bany(&r->act))
				print(" a:%Q ", r->act);
		}
	}
	print("\n");
}


void
dumpit(char *str, Flow *r0, int isreg)
{
	Flow *r, *r1;

	print("\n%s\n", str);
	for(r = r0; r != nil; r = r->link) {
		dumpone(r, isreg);
		r1 = r->p2;
		if(r1 != nil) {
			print("	pred:");
			for(; r1 != nil; r1 = r1->p2link)
				print(" %.4ud", (int)r1->prog->pc);
			print("\n");
		}
//		r1 = r->s1;
//		if(r1 != R) {
//			print("	succ:");
//			for(; r1 != R; r1 = r1->s1)
//				print(" %.4ud", (int)r1->prog->pc);
//			print("\n");
//		}
	}
}
