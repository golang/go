// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"l.h"
#include	"../ld/lib.h"

// Software floating point.

void
softfloat(void)
{
	Prog *p, *next, *psfloat;
	Sym *symsfloat;
	int wasfloat;

	if(!debug['F'])
		return;

	symsfloat = lookup("_sfloat", 0);
	psfloat = P;
	if(symsfloat->type == STEXT)
		psfloat = symsfloat->text;

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		wasfloat = 0;
		for(p = cursym->text; p != P; p = p->link)
			if(p->cond != P)
				p->cond->mark |= LABEL;
		for(p = cursym->text; p != P; p = p->link) {
			switch(p->as) {
			case AMOVW:
				if(p->to.type == D_FREG || p->from.type == D_FREG)
					goto soft;
				goto notsoft;

			case AMOVWD:
			case AMOVWF:
			case AMOVDW:
			case AMOVFW:
			case AMOVFD:
			case AMOVDF:
			case AMOVF:
			case AMOVD:

			case ACMPF:
			case ACMPD:
			case AADDF:
			case AADDD:
			case ASUBF:
			case ASUBD:
			case AMULF:
			case AMULD:
			case ADIVF:
			case ADIVD:
			case ASQRTF:
			case ASQRTD:
			case AABSF:
			case AABSD:
				goto soft;

			default:
				goto notsoft;

			soft:
				if (psfloat == P)
					diag("floats used with _sfloat not defined");
				if (!wasfloat || (p->mark&LABEL)) {
					next = prg();
					*next = *p;
	
					// BL _sfloat(SB)
					*p = zprg;
					p->link = next;
					p->as = ABL;
	 				p->to.type = D_BRANCH;
					p->to.sym = symsfloat;
					p->cond = psfloat;
					p->line = next->line;
	
					p = next;
					wasfloat = 1;
				}
				break;

			notsoft:
				wasfloat = 0;
			}
		}
	}
}
