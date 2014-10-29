// Derived from Inferno utils/6c/peep.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/peep.c
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
peep(Prog *p)
{
	USED(p);
	// TODO(minux)
	return;
}

void
excise(Flow *r)
{
	Prog *p;

	p = r->prog;
	if(debug['P'] && debug['v'])
		print("%P ===delete===\n", p);
	*p = zprog;
	p->as = ANOP;
	ostats.ndelmov++;
}

int
regtyp(Adr *a)
{
	switch(a->type) {
	default:
		return 0;
	case D_REG:
	case D_FREG:
		return 1;
	}
}

int
sameaddr(Addr *a, Addr *v)
{
	if(a->type != v->type)
		return 0;
	if(regtyp(v) && a->reg == v->reg)
		return 1;
	if(v->type == D_AUTO || v->type == D_PARAM)
		if(v->offset == a->offset)
			return 1;
	return 0;
}

int
smallindir(Addr *a, Addr *reg)
{
	return reg->type == D_REG && a->type == D_OREG &&
		a->reg == reg->reg &&
		0 <= a->offset && a->offset < 4096;
}

int
stackaddr(Addr *a)
{
	return a->type == D_REG && a->reg == REGSP;
}
