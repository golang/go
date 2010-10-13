// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define	EXTERN
#include	"l.h"

// Software floating point.

void
softfloat(void)
{
	Prog *p, *next, *psfloat;
	Sym *symsfloat;
	int wasfloat;
	
	symsfloat = lookup("_sfloat", 0);
	psfloat = P;
	if(symsfloat->type == STEXT)
		psfloat = symsfloat->text;

	wasfloat = 0;
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
			switch(p->as) {
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
				if (psfloat == P)
					diag("floats used with _sfloat not defined");
				if (!wasfloat) {
					next = prg();
					*next = *p;
	
					// BL		_sfloat(SB)
					*p = zprg;
					p->link = next;
					p->as = ABL;
	 				p->to.type = D_BRANCH;
					p->to.sym = symsfloat;
					p->cond = psfloat;
	
					p = next;
					wasfloat = 1;
				}
				break;
			default:
				wasfloat = 0;
			}
		}
	}
}
