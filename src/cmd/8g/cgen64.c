// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"

/*
 * attempt to generate 64-bit
 *	res = n
 * return 1 on success, 0 if op not handled.
 */
void
cgen64(Node *n, Node *res)
{
	Node t1, t2, ax, dx, cx, ex, fx, *l, *r;
	Node lo1, lo2, hi1, hi2;
	Prog *p1, *p2;
	uint64 v;
	uint32 lv, hv;

	if(res->op != OINDREG && res->op != ONAME) {
		dump("n", n);
		dump("res", res);
		fatal("cgen64 %O of %O", n->op, res->op);
	}
	switch(n->op) {
	default:
		fatal("cgen64 %O", n->op);

	case OMINUS:
		cgen(n->left, res);
		split64(res, &lo1, &hi1);
		gins(ANEGL, N, &lo1);
		gins(AADCL, ncon(0), &hi1);
		gins(ANEGL, N, &hi1);
		splitclean();
		return;

	case OCOM:
		cgen(n->left, res);
		split64(res, &lo1, &hi1);
		gins(ANOTL, N, &lo1);
		gins(ANOTL, N, &hi1);
		splitclean();
		return;

	case OADD:
	case OSUB:
	case OMUL:
	case OLROT:
	case OLSH:
	case ORSH:
	case OAND:
	case OOR:
	case OXOR:
		// binary operators.
		// common setup below.
		break;
	}

	l = n->left;
	r = n->right;
	if(!l->addable) {
		tempname(&t1, l->type);
		cgen(l, &t1);
		l = &t1;
	}
	if(r != N && !r->addable) {
		tempname(&t2, r->type);
		cgen(r, &t2);
		r = &t2;
	}

	nodreg(&ax, types[TINT32], D_AX);
	nodreg(&cx, types[TINT32], D_CX);
	nodreg(&dx, types[TINT32], D_DX);

	// Setup for binary operation.
	split64(l, &lo1, &hi1);
	if(is64(r->type))
		split64(r, &lo2, &hi2);

	// Do op.  Leave result in DX:AX.
	switch(n->op) {
	case OADD:
		// TODO: Constants
		gins(AMOVL, &lo1, &ax);
		gins(AMOVL, &hi1, &dx);
		gins(AADDL, &lo2, &ax);
		gins(AADCL, &hi2, &dx);
		break;

	case OSUB:
		// TODO: Constants.
		gins(AMOVL, &lo1, &ax);
		gins(AMOVL, &hi1, &dx);
		gins(ASUBL, &lo2, &ax);
		gins(ASBBL, &hi2, &dx);
		break;

	case OMUL:
		// let's call the next two EX and FX.
		regalloc(&ex, types[TPTR32], N);
		regalloc(&fx, types[TPTR32], N);

		// load args into DX:AX and EX:CX.
		gins(AMOVL, &lo1, &ax);
		gins(AMOVL, &hi1, &dx);
		gins(AMOVL, &lo2, &cx);
		gins(AMOVL, &hi2, &ex);

		// if DX and EX are zero, use 32 x 32 -> 64 unsigned multiply.
		gins(AMOVL, &dx, &fx);
		gins(AORL, &ex, &fx);
		p1 = gbranch(AJNE, T, 0);
		gins(AMULL, &cx, N);	// implicit &ax
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);

		// full 64x64 -> 64, from 32x32 -> 64.
		gins(AIMULL, &cx, &dx);
		gins(AMOVL, &ax, &fx);
		gins(AIMULL, &ex, &fx);
		gins(AADDL, &dx, &fx);
		gins(AMOVL, &cx, &dx);
		gins(AMULL, &dx, N);	// implicit &ax
		gins(AADDL, &fx, &dx);
		patch(p2, pc);

		regfree(&ex);
		regfree(&fx);
		break;
	
	case OLROT:
		// We only rotate by a constant c in [0,64).
		// if c >= 32:
		//	lo, hi = hi, lo
		//	c -= 32
		// if c == 0:
		//	no-op
		// else:
		//	t = hi
		//	shld hi:lo, c
		//	shld lo:t, c
		v = mpgetfix(r->val.u.xval);
		if(v >= 32) {
			// reverse during load to do the first 32 bits of rotate
			v -= 32;
			gins(AMOVL, &lo1, &dx);
			gins(AMOVL, &hi1, &ax);
		} else {
			gins(AMOVL, &lo1, &ax);
			gins(AMOVL, &hi1, &dx);
		}
		if(v == 0) {
			// done
		} else {
			gins(AMOVL, &dx, &cx);
			p1 = gins(ASHLL, ncon(v), &dx);
			p1->from.index = D_AX;	// double-width shift
			p1->from.scale = 0;
			p1 = gins(ASHLL, ncon(v), &ax);
			p1->from.index = D_CX;	// double-width shift
			p1->from.scale = 0;
		}
		break;

	case OLSH:
		if(r->op == OLITERAL) {
			v = mpgetfix(r->val.u.xval);
			if(v >= 64) {
				if(is64(r->type))
					splitclean();
				splitclean();
				split64(res, &lo2, &hi2);
				gins(AMOVL, ncon(0), &lo2);
				gins(AMOVL, ncon(0), &hi2);
				splitclean();
				goto out;
			}
			if(v >= 32) {
				if(is64(r->type))
					splitclean();
				split64(res, &lo2, &hi2);
				gmove(&lo1, &hi2);
				if(v > 32) {
					gins(ASHLL, ncon(v - 32), &hi2);
				}
				gins(AMOVL, ncon(0), &lo2);
				splitclean();
				splitclean();
				goto out;
			}

			// general shift
			gins(AMOVL, &lo1, &ax);
			gins(AMOVL, &hi1, &dx);
			p1 = gins(ASHLL, ncon(v), &dx);
			p1->from.index = D_AX;	// double-width shift
			p1->from.scale = 0;
			gins(ASHLL, ncon(v), &ax);
			break;
		}

		// load value into DX:AX.
		gins(AMOVL, &lo1, &ax);
		gins(AMOVL, &hi1, &dx);

		// load shift value into register.
		// if high bits are set, zero value.
		p1 = P;
		if(is64(r->type)) {
			gins(ACMPL, &hi2, ncon(0));
			p1 = gbranch(AJNE, T, +1);
			gins(AMOVL, &lo2, &cx);
		} else {
			cx.type = types[TUINT32];
			gmove(r, &cx);
		}

		// if shift count is >=64, zero value
		gins(ACMPL, &cx, ncon(64));
		p2 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
		if(p1 != P)
			patch(p1, pc);
		gins(AXORL, &dx, &dx);
		gins(AXORL, &ax, &ax);
		patch(p2, pc);

		// if shift count is >= 32, zero low.
		gins(ACMPL, &cx, ncon(32));
		p1 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
		gins(AMOVL, &ax, &dx);
		gins(ASHLL, &cx, &dx);	// SHLL only uses bottom 5 bits of count
		gins(AXORL, &ax, &ax);
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);

		// general shift
		p1 = gins(ASHLL, &cx, &dx);
		p1->from.index = D_AX;	// double-width shift
		p1->from.scale = 0;
		gins(ASHLL, &cx, &ax);
		patch(p2, pc);
		break;

	case ORSH:
		if(r->op == OLITERAL) {
			v = mpgetfix(r->val.u.xval);
			if(v >= 64) {
				if(is64(r->type))
					splitclean();
				splitclean();
				split64(res, &lo2, &hi2);
				if(hi1.type->etype == TINT32) {
					gmove(&hi1, &lo2);
					gins(ASARL, ncon(31), &lo2);
					gmove(&hi1, &hi2);
					gins(ASARL, ncon(31), &hi2);
				} else {
					gins(AMOVL, ncon(0), &lo2);
					gins(AMOVL, ncon(0), &hi2);
				}
				splitclean();
				goto out;
			}
			if(v >= 32) {
				if(is64(r->type))
					splitclean();
				split64(res, &lo2, &hi2);
				gmove(&hi1, &lo2);
				if(v > 32)
					gins(optoas(ORSH, hi1.type), ncon(v-32), &lo2);
				if(hi1.type->etype == TINT32) {
					gmove(&hi1, &hi2);
					gins(ASARL, ncon(31), &hi2);
				} else
					gins(AMOVL, ncon(0), &hi2);
				splitclean();
				splitclean();
				goto out;
			}

			// general shift
			gins(AMOVL, &lo1, &ax);
			gins(AMOVL, &hi1, &dx);
			p1 = gins(ASHRL, ncon(v), &ax);
			p1->from.index = D_DX;	// double-width shift
			p1->from.scale = 0;
			gins(optoas(ORSH, hi1.type), ncon(v), &dx);
			break;
		}

		// load value into DX:AX.
		gins(AMOVL, &lo1, &ax);
		gins(AMOVL, &hi1, &dx);

		// load shift value into register.
		// if high bits are set, zero value.
		p1 = P;
		if(is64(r->type)) {
			gins(ACMPL, &hi2, ncon(0));
			p1 = gbranch(AJNE, T, +1);
			gins(AMOVL, &lo2, &cx);
		} else {
			cx.type = types[TUINT32];
			gmove(r, &cx);
		}

		// if shift count is >=64, zero or sign-extend value
		gins(ACMPL, &cx, ncon(64));
		p2 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
		if(p1 != P)
			patch(p1, pc);
		if(hi1.type->etype == TINT32) {
			gins(ASARL, ncon(31), &dx);
			gins(AMOVL, &dx, &ax);
		} else {
			gins(AXORL, &dx, &dx);
			gins(AXORL, &ax, &ax);
		}
		patch(p2, pc);

		// if shift count is >= 32, sign-extend hi.
		gins(ACMPL, &cx, ncon(32));
		p1 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
		gins(AMOVL, &dx, &ax);
		if(hi1.type->etype == TINT32) {
			gins(ASARL, &cx, &ax);	// SARL only uses bottom 5 bits of count
			gins(ASARL, ncon(31), &dx);
		} else {
			gins(ASHRL, &cx, &ax);
			gins(AXORL, &dx, &dx);
		}
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);

		// general shift
		p1 = gins(ASHRL, &cx, &ax);
		p1->from.index = D_DX;	// double-width shift
		p1->from.scale = 0;
		gins(optoas(ORSH, hi1.type), &cx, &dx);
		patch(p2, pc);
		break;

	case OXOR:
	case OAND:
	case OOR:
		// make constant the right side (it usually is anyway).
		if(lo1.op == OLITERAL) {
			nswap(&lo1, &lo2);
			nswap(&hi1, &hi2);
		}
		if(lo2.op == OLITERAL) {
			// special cases for constants.
			lv = mpgetfix(lo2.val.u.xval);
			hv = mpgetfix(hi2.val.u.xval);
			splitclean();	// right side
			split64(res, &lo2, &hi2);
			switch(n->op) {
			case OXOR:
				gmove(&lo1, &lo2);
				gmove(&hi1, &hi2);
				switch(lv) {
				case 0:
					break;
				case 0xffffffffu:
					gins(ANOTL, N, &lo2);
					break;
				default:
					gins(AXORL, ncon(lv), &lo2);
					break;
				}
				switch(hv) {
				case 0:
					break;
				case 0xffffffffu:
					gins(ANOTL, N, &hi2);
					break;
				default:
					gins(AXORL, ncon(hv), &hi2);
					break;
				}
				break;

			case OAND:
				switch(lv) {
				case 0:
					gins(AMOVL, ncon(0), &lo2);
					break;
				default:
					gmove(&lo1, &lo2);
					if(lv != 0xffffffffu)
						gins(AANDL, ncon(lv), &lo2);
					break;
				}
				switch(hv) {
				case 0:
					gins(AMOVL, ncon(0), &hi2);
					break;
				default:
					gmove(&hi1, &hi2);
					if(hv != 0xffffffffu)
						gins(AANDL, ncon(hv), &hi2);
					break;
				}
				break;

			case OOR:
				switch(lv) {
				case 0:
					gmove(&lo1, &lo2);
					break;
				case 0xffffffffu:
					gins(AMOVL, ncon(0xffffffffu), &lo2);
					break;
				default:
					gmove(&lo1, &lo2);
					gins(AORL, ncon(lv), &lo2);
					break;
				}
				switch(hv) {
				case 0:
					gmove(&hi1, &hi2);
					break;
				case 0xffffffffu:
					gins(AMOVL, ncon(0xffffffffu), &hi2);
					break;
				default:
					gmove(&hi1, &hi2);
					gins(AORL, ncon(hv), &hi2);
					break;
				}
				break;
			}
			splitclean();
			splitclean();
			goto out;
		}
		gins(AMOVL, &lo1, &ax);
		gins(AMOVL, &hi1, &dx);
		gins(optoas(n->op, lo1.type), &lo2, &ax);
		gins(optoas(n->op, lo1.type), &hi2, &dx);
		break;
	}
	if(is64(r->type))
		splitclean();
	splitclean();

	split64(res, &lo1, &hi1);
	gins(AMOVL, &ax, &lo1);
	gins(AMOVL, &dx, &hi1);
	splitclean();

out:;
}

/*
 * generate comparison of nl, nr, both 64-bit.
 * nl is memory; nr is constant or memory.
 */
void
cmp64(Node *nl, Node *nr, int op, int likely, Prog *to)
{
	Node lo1, hi1, lo2, hi2, rr;
	Prog *br;
	Type *t;

	split64(nl, &lo1, &hi1);
	split64(nr, &lo2, &hi2);

	// compare most significant word;
	// if they differ, we're done.
	t = hi1.type;
	if(nl->op == OLITERAL || nr->op == OLITERAL)
		gins(ACMPL, &hi1, &hi2);
	else {
		regalloc(&rr, types[TINT32], N);
		gins(AMOVL, &hi1, &rr);
		gins(ACMPL, &rr, &hi2);
		regfree(&rr);
	}
	br = P;
	switch(op) {
	default:
		fatal("cmp64 %O %T", op, t);
	case OEQ:
		// cmp hi
		// jne L
		// cmp lo
		// jeq to
		// L:
		br = gbranch(AJNE, T, -likely);
		break;
	case ONE:
		// cmp hi
		// jne to
		// cmp lo
		// jne to
		patch(gbranch(AJNE, T, likely), to);
		break;
	case OGE:
	case OGT:
		// cmp hi
		// jgt to
		// jlt L
		// cmp lo
		// jge to (or jgt to)
		// L:
		patch(gbranch(optoas(OGT, t), T, likely), to);
		br = gbranch(optoas(OLT, t), T, -likely);
		break;
	case OLE:
	case OLT:
		// cmp hi
		// jlt to
		// jgt L
		// cmp lo
		// jle to (or jlt to)
		// L:
		patch(gbranch(optoas(OLT, t), T, likely), to);
		br = gbranch(optoas(OGT, t), T, -likely);
		break;
	}

	// compare least significant word
	t = lo1.type;
	if(nl->op == OLITERAL || nr->op == OLITERAL)
		gins(ACMPL, &lo1, &lo2);
	else {
		regalloc(&rr, types[TINT32], N);
		gins(AMOVL, &lo1, &rr);
		gins(ACMPL, &rr, &lo2);
		regfree(&rr);
	}

	// jump again
	patch(gbranch(optoas(op, t), T, likely), to);

	// point first branch down here if appropriate
	if(br != P)
		patch(br, pc);

	splitclean();
	splitclean();
}

