// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 1;

typedef	struct	Hchan	Hchan;

struct	Hchan
{
	uint32	elemsize;
	uint32	hint;
	uint32	eo;
	Alg*	elemalg;
};

// newchan(elemsize uint32, elemalg uint32, hint uint32) (hchan *chan any);
void
sys·newchan(uint32 elemsize, uint32 elemalg, uint32 hint,
	Hchan* ret)
{
	Hchan *c;

	if(elemalg >= nelem(algarray)) {
		prints("0<=");
		sys·printint(elemalg);
		prints("<");
		sys·printint(nelem(algarray));
		prints("\n");

		throw("sys·newchan: elem algorithm out of range");
	}

	c = mal(sizeof(*c));

	c->elemsize = elemsize;
	c->elemalg = &algarray[elemalg];
	c->hint = hint;

	// these calculations are compiler dependent
	c->eo = rnd(sizeof(c), elemsize);

	ret = c;
	FLUSH(&ret);

	if(debug) {
		prints("newchan: chan=");
		sys·printpointer(c);
		prints("; elemsize=");
		sys·printint(elemsize);
		prints("; elemalg=");
		sys·printint(elemalg);
		prints("; hint=");
		sys·printint(hint);
		prints("\n");
	}
}

// chansend(hchan *chan any, elem any);
void
sys·chansend(Hchan* c, ...)
{
	byte *ae;

	ae = (byte*)&c + c->eo;
	if(debug) {
		prints("chansend: chan=");
		sys·printpointer(c);
		prints("; elem=");
		c->elemalg->print(c->elemsize, ae);
		prints("\n");
	}
}
