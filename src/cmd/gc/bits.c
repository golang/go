// Inferno utils/cc/bits.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/bits.c
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
#include "go.h"

/*
Bits
bor(Bits a, Bits b)
{
	Bits c;
	int i;

	for(i=0; i<BITS; i++)
		c.b[i] = a.b[i] | b.b[i];
	return c;
}

Bits
band(Bits a, Bits b)
{
	Bits c;
	int i;

	for(i=0; i<BITS; i++)
		c.b[i] = a.b[i] & b.b[i];
	return c;
}

Bits
bnot(Bits a)
{
	Bits c;
	int i;

	for(i=0; i<BITS; i++)
		c.b[i] = ~a.b[i];
	return c;
}
*/

int
bany(Bits *a)
{
	int i;

	for(i=0; i<BITS; i++)
		if(a->b[i])
			return 1;
	return 0;
}

/*
int
beq(Bits a, Bits b)
{
	int i;

	for(i=0; i<BITS; i++)
		if(a.b[i] != b.b[i])
			return 0;
	return 1;
}
*/

int
bnum(Bits a)
{
	int i;
	uint64 b;

	for(i=0; i<BITS; i++){
		b = a.b[i];
		if(b)
			return 64*i + bitno(b);
	}
	fatal("bad in bnum");
	return 0;
}

Bits
blsh(uint n)
{
	Bits c;

	c = zbits;
	c.b[n/64] = 1LL << (n%64);
	return c;
}

int
btest(Bits *a, uint n)
{
	return (a->b[n/64] & (1LL << (n%64))) != 0;
}

void
biset(Bits *a, uint n)
{
	a->b[n/64] |= 1LL << (n%64);
}

void
biclr(Bits *a, uint n)
{
	a->b[n/64] &= ~(1LL << (n%64));
}

int
bitno(uint64 b)
{
	int i;

	for(i=0; i<64; i++)
		if(b & (1LL<<i))
			return i;
	fatal("bad in bitno");
	return 0;
}

int
Qconv(Fmt *fp)
{
	Bits bits;
	int i, first;

	first = 1;
	bits = va_arg(fp->args, Bits);
	while(bany(&bits)) {
		i = bnum(bits);
		if(first)
			first = 0;
		else
			fmtprint(fp, " ");
		if(var[i].node == N || var[i].node->sym == S)
			fmtprint(fp, "$%d", i);
		else {
			fmtprint(fp, "%s(%d)", var[i].node->sym->name, i);
			if(var[i].offset != 0)
				fmtprint(fp, "%+lld", (vlong)var[i].offset);
		}
		biclr(&bits, i);
	}
	return 0;
}
