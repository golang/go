// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>


// funcpctab writes to dst a pc-value table mapping the code in func to the values
// returned by valfunc parameterized by arg. The invocation of valfunc to update the
// current value is, for each p,
//
//	val = valfunc(func, val, p, 0, arg);
//	record val as value at p->pc;
//	val = valfunc(func, val, p, 1, arg);
//
// where func is the function, val is the current value, p is the instruction being
// considered, and arg can be used to further parameterize valfunc.

// pctofileline computes either the file number (arg == 0)
// or the line number (arg == 1) to use at p.
// Because p->lineno applies to p, phase == 0 (before p)
// takes care of the update.

// pctospadj computes the sp adjustment in effect.
// It is oldval plus any adjustment made by p itself.
// The adjustment by p takes effect only after p, so we
// apply the change during phase == 1.

// pctopcdata computes the pcdata value in effect at p.
// A PCDATA instruction sets the value in effect at future
// non-PCDATA instructions.
// Since PCDATA instructions have no width in the final code,
// it does not matter which phase we use for the update.


// iteration over encoded pcdata tables.

static uint32
getvarint(uchar **pp)
{
	uchar *p;
	int shift;
	uint32 v;

	v = 0;
	p = *pp;
	for(shift = 0;; shift += 7) {
		v |= (uint32)(*p & 0x7F) << shift;
		if(!(*p++ & 0x80))
			break;
	}
	*pp = p;
	return v;
}

void
pciternext(Pciter *it)
{
	uint32 v;
	int32 dv;

	it->pc = it->nextpc;
	if(it->done)
		return;
	if(it->p >= it->d.p + it->d.n) {
		it->done = 1;
		return;
	}

	// value delta
	v = getvarint(&it->p);
	if(v == 0 && !it->start) {
		it->done = 1;
		return;
	}
	it->start = 0;
	dv = (int32)(v>>1) ^ ((int32)(v<<31)>>31);
	it->value += dv;
	
	// pc delta
	v = getvarint(&it->p);
	it->nextpc = it->pc + v*it->pcscale;
}

void
pciterinit(Link *ctxt, Pciter *it, Pcdata *d)
{
	it->d = *d;
	it->p = it->d.p;
	it->pc = 0;
	it->nextpc = 0;
	it->value = -1;
	it->start = 1;
	it->done = 0;
	it->pcscale = ctxt->arch->minlc;
	pciternext(it);
}
