// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define	EXTERN
#include	"l.h"

void
softfloat()
{
	Prog *p, *prev, *psfloat;
	Sym *symsfloat;
	int wasfloat;
	
	symsfloat = lookup("_sfloat", 0);
	psfloat = P;
	if(symsfloat->type == STEXT)
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT) {
			if(p->from.sym == symsfloat) {
				psfloat = p;
				break;
			}
		}
	}

	wasfloat = 0;
	p = firstp;
	prev = P;
	for(p = firstp; p != P; p = p->link) {
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
				if (prev == P)
					diag("float instruction without predecessor TEXT");
				// BL		_sfloat(SB)
				prev = appendp(prev);
				prev->as = ABL;
 				prev->to.type = D_BRANCH;
				prev->to.sym = symsfloat;
				prev->cond = psfloat;
				
				wasfloat = 1;
			}
			break;
		default:
			wasfloat = 0;
		}
		prev = p;
	}
}
