// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

/*
 * attempt to generate 64-bit
 *	res = n
 * return 1 on success, 0 if op not handled.
 */
void
cgen64(Node *n, Node *res)
{
	Node t1, t2, *l, *r;
	Node lo1, lo2, hi1, hi2;
	Node al, ah, bl, bh, cl, ch; //, s1, s2;
	Prog *p1;
 //, *p2;
//	uint64 v;
//	uint32 lv, hv;

	if(res->op != OINDREG && res->op != ONAME) {
		dump("n", n);
		dump("res", res);
		fatal("cgen64 %O of %O", n->op, res->op);
	}
	switch(n->op) {
	default:
		fatal("cgen64 %O", n->op);

//	case OMINUS:
//		cgen(n->left, res);
//		split64(res, &lo1, &hi1);
//		gins(ANEGL, N, &lo1);
//		gins(AADCL, ncon(0), &hi1);
//		gins(ANEGL, N, &hi1);
//		splitclean();
//		return;

//	case OCOM:
//		cgen(n->left, res);
//		split64(res, &lo1, &hi1);
//		gins(ANOTL, N, &lo1);
//		gins(ANOTL, N, &hi1);
//		splitclean();
//		return;

	case OADD:
	case OSUB:
	case OMUL:
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
		tempalloc(&t1, l->type);
		cgen(l, &t1);
		l = &t1;
	}
	if(r != N && !r->addable) {
		tempalloc(&t2, r->type);
		cgen(r, &t2);
		r = &t2;
	}

	// Setup for binary operation.
	split64(l, &lo1, &hi1);
	if(is64(r->type))
		split64(r, &lo2, &hi2);

	regalloc(&al, lo1.type, N);
	regalloc(&ah, hi1.type, N);
	// Do op.  Leave result in ah:al.
	switch(n->op) {
	default:
		fatal("cgen64: not implemented: %N\n", n);

	case OADD:
		// TODO: Constants
		regalloc(&bl, types[TPTR32], N);
		regalloc(&bh, types[TPTR32], N);
		gins(AMOVW, &hi1, &ah);
		gins(AMOVW, &lo1, &al);
		gins(AMOVW, &hi2, &bh);
		gins(AMOVW, &lo2, &bl);
		gins(AADD, &bl, &al);
		gins(AADC, &bh, &ah);
		regfree(&bl);
		regfree(&bh);
		break;

//	case OSUB:
//		// TODO: Constants.
//		gins(AMOVL, &lo1, &ax);
//		gins(AMOVL, &hi1, &dx);
//		gins(ASUBL, &lo2, &ax);
//		gins(ASBBL, &hi2, &dx);
//		break;

	case OMUL:
		// TODO(kaib): this can be done with 4 regs and does not need 6
		regalloc(&bh, types[TPTR32], N);
		regalloc(&bl, types[TPTR32], N);
		regalloc(&ch, types[TPTR32], N);
		regalloc(&cl, types[TPTR32], N);

		// load args into bh:bl and bh:bl.
		gins(AMOVW, &hi1, &bh);
		gins(AMOVW, &lo1, &bl);
		gins(AMOVW, &hi2, &ch);
		gins(AMOVW, &lo2, &cl);

		// bl * cl
		p1 = gins(AMULLU, N, N);
		p1->from.type = D_REG;
		p1->from.reg = bl.val.u.reg;
		p1->reg = cl.val.u.reg;
		p1->to.type = D_REGREG;
		p1->to.reg = al.val.u.reg;
		p1->to.offset = ah.val.u.reg;
//print("%P\n", p1);

		// bl * ch
		p1 = gins(AMULALU, N, N);
		p1->from.type = D_REG;
		p1->from.reg = ah.val.u.reg;
		p1->reg = bl.val.u.reg;
		p1->to.type = D_REGREG;
		p1->to.reg = ch.val.u.reg;
		p1->to.offset = ah.val.u.reg;
//print("%P\n", p1);

		// bh * cl
		p1 = gins(AMULALU, N, N);
		p1->from.type = D_REG;
		p1->from.reg = ah.val.u.reg;
		p1->reg = bh.val.u.reg;
		p1->to.type = D_REGREG;
		p1->to.reg = cl.val.u.reg;
		p1->to.offset = ah.val.u.reg;
//print("%P\n", p1);

		regfree(&bh);
		regfree(&bl);
		regfree(&ch);
		regfree(&cl);

		break;

//	case OLSH:
		// TODO(kaib): optimize for OLITERAL
//		regalloc(&s1, types[TPTR32], N);
//		regalloc(&s2, types[TPTR32], N);

//		gins(AMOVW, &lo1, &al);
//		gins(AMOVW, &hi1, &ah);
//		if(is64(r->type)) {
//			gins(AMOVW, &lo2, &s1);
//			gins(AMOVW, &hi2, &s2);
//			p1 = gins(AOR, &s2, &s1);
//			p1->from.type = D_SHIFT;
//			p1->from.offset = 5 << 7 | s2.val.u.reg; // s2<<7
//			p1->from.reg = NREG;
//		} else
//			gins(AMOVW, r, &s1
//		p1 = gins(AMOVW, &s1, &s2);
//		p1->from.offset = -32;

//		//	MOVW	ah<<s1, ah
//		p1 = gins(AMOVW, &ah, &ah);
//		p1->from.offset = ah.val.u.reg | 1<<4 | s1.val.u.reg <<8;

		//	OR		al<<s2, ah
//		p1 = gins(AOR, &al, &ah);
//		p1->from.offset = al.val.u.reg | 1<<4 | s2.val.u.reg << 8;

		//	MOVW	al<<s1, al
//		p1 = gins(AMOVW, &al, &al);
//		p1->from.offset = al.val.u.reg | 1<<4 | s1.val.u.reg <<8;

//		regfree(&s1);
//		regfree(&s2);
//		break;

//	case ORSH:
//		if(r->op == OLITERAL) {
//			fatal("cgen64 ORSH, OLITERAL not implemented");
//			v = mpgetfix(r->val.u.xval);
//			if(v >= 64) {
//				if(is64(r->type))
//					splitclean();
//				splitclean();
//				split64(res, &lo2, &hi2);
//				if(hi1.type->etype == TINT32) {
//					gmove(&hi1, &lo2);
//					gins(ASARL, ncon(31), &lo2);
//					gmove(&hi1, &hi2);
//					gins(ASARL, ncon(31), &hi2);
//				} else {
//					gins(AMOVL, ncon(0), &lo2);
//					gins(AMOVL, ncon(0), &hi2);
//				}
//				splitclean();
//				goto out;
//			}
//			if(v >= 32) {
//				if(is64(r->type))
//					splitclean();
//				split64(res, &lo2, &hi2);
//				gmove(&hi1, &lo2);
//				if(v > 32)
//					gins(optoas(ORSH, hi1.type), ncon(v-32), &lo2);
//				if(hi1.type->etype == TINT32) {
//					gmove(&hi1, &hi2);
//					gins(ASARL, ncon(31), &hi2);
//				} else
//					gins(AMOVL, ncon(0), &hi2);
//				splitclean();
//				splitclean();
//				goto out;
//			}

//			// general shift
//			gins(AMOVL, &lo1, &ax);
//			gins(AMOVL, &hi1, &dx);
//			p1 = gins(ASHRL, ncon(v), &ax);
//			p1->from.index = D_DX;	// double-width shift
//			p1->from.scale = 0;
//			gins(optoas(ORSH, hi1.type), ncon(v), &dx);
//			break;
//		}
//		fatal("cgen64 ORSH, !OLITERAL not implemented");

//		// load value into DX:AX.
//		gins(AMOVL, &lo1, &ax);
//		gins(AMOVL, &hi1, &dx);

//		// load shift value into register.
//		// if high bits are set, zero value.
//		p1 = P;
//		if(is64(r->type)) {
//			gins(ACMPL, &hi2, ncon(0));
//			p1 = gbranch(AJNE, T);
//			gins(AMOVL, &lo2, &cx);
//		} else
//			gins(AMOVL, r, &cx);

//		// if shift count is >=64, zero or sign-extend value
//		gins(ACMPL, &cx, ncon(64));
//		p2 = gbranch(optoas(OLT, types[TUINT32]), T);
//		if(p1 != P)
//			patch(p1, pc);
//		if(hi1.type->etype == TINT32) {
//			gins(ASARL, ncon(31), &dx);
//			gins(AMOVL, &dx, &ax);
//		} else {
//			gins(AXORL, &dx, &dx);
//			gins(AXORL, &ax, &ax);
//		}
//		patch(p2, pc);

//		// if shift count is >= 32, sign-extend hi.
//		gins(ACMPL, &cx, ncon(32));
//		p1 = gbranch(optoas(OLT, types[TUINT32]), T);
//		gins(AMOVL, &dx, &ax);
//		if(hi1.type->etype == TINT32) {
//			gins(ASARL, &cx, &ax);	// SARL only uses bottom 5 bits of count
//			gins(ASARL, ncon(31), &dx);
//		} else {
//			gins(ASHRL, &cx, &ax);
//			gins(AXORL, &dx, &dx);
//		}
//		p2 = gbranch(AJMP, T);
//		patch(p1, pc);

//		// general shift
//		p1 = gins(ASHRL, &cx, &ax);
//		p1->from.index = D_DX;	// double-width shift
//		p1->from.scale = 0;
//		gins(optoas(ORSH, hi1.type), &cx, &dx);
//		patch(p2, pc);
//		break;

//	case OXOR:
//	case OAND:
//	case OOR:
//		// make constant the right side (it usually is anyway).
//		if(lo1.op == OLITERAL) {
//			nswap(&lo1, &lo2);
//			nswap(&hi1, &hi2);
//		}
//		if(lo2.op == OLITERAL) {
//			// special cases for constants.
//			lv = mpgetfix(lo2.val.u.xval);
//			hv = mpgetfix(hi2.val.u.xval);
//			splitclean();	// right side
//			split64(res, &lo2, &hi2);
//			switch(n->op) {
//			case OXOR:
//				gmove(&lo1, &lo2);
//				gmove(&hi1, &hi2);
//				switch(lv) {
//				case 0:
//					break;
//				case 0xffffffffu:
//					gins(ANOTL, N, &lo2);
//					break;
//				default:
//					gins(AXORL, ncon(lv), &lo2);
//					break;
//				}
//				switch(hv) {
//				case 0:
//					break;
//				case 0xffffffffu:
//					gins(ANOTL, N, &hi2);
//					break;
//				default:
//					gins(AXORL, ncon(hv), &hi2);
//					break;
//				}
//				break;

//			case OAND:
//				switch(lv) {
//				case 0:
//					gins(AMOVL, ncon(0), &lo2);
//					break;
//				default:
//					gmove(&lo1, &lo2);
//					if(lv != 0xffffffffu)
//						gins(AANDL, ncon(lv), &lo2);
//					break;
//				}
//				switch(hv) {
//				case 0:
//					gins(AMOVL, ncon(0), &hi2);
//					break;
//				default:
//					gmove(&hi1, &hi2);
//					if(hv != 0xffffffffu)
//						gins(AANDL, ncon(hv), &hi2);
//					break;
//				}
//				break;

//			case OOR:
//				switch(lv) {
//				case 0:
//					gmove(&lo1, &lo2);
//					break;
//				case 0xffffffffu:
//					gins(AMOVL, ncon(0xffffffffu), &lo2);
//					break;
//				default:
//					gmove(&lo1, &lo2);
//					gins(AORL, ncon(lv), &lo2);
//					break;
//				}
//				switch(hv) {
//				case 0:
//					gmove(&hi1, &hi2);
//					break;
//				case 0xffffffffu:
//					gins(AMOVL, ncon(0xffffffffu), &hi2);
//					break;
//				default:
//					gmove(&hi1, &hi2);
//					gins(AORL, ncon(hv), &hi2);
//					break;
//				}
//				break;
//			}
//			splitclean();
//			splitclean();
//			goto out;
//		}
//		gins(AMOVL, &lo1, &ax);
//		gins(AMOVL, &hi1, &dx);
//		gins(optoas(n->op, lo1.type), &lo2, &ax);
//		gins(optoas(n->op, lo1.type), &hi2, &dx);
//		break;
	}
	if(is64(r->type))
		splitclean();
	splitclean();

	split64(res, &lo1, &hi1);
	gins(AMOVW, &al, &lo1);
	gins(AMOVW, &ah, &hi1);
	splitclean();

//out:
	if(r == &t2)
		tempfree(&t2);
	if(l == &t1)
		tempfree(&t1);
	regfree(&al);
	regfree(&ah);
}

/*
 * generate comparison of nl, nr, both 64-bit.
 * nl is memory; nr is constant or memory.
 */
void
cmp64(Node *nl, Node *nr, int op, Prog *to)
{
	fatal("cmp64 not implemented");
//	Node lo1, hi1, lo2, hi2, rr;
//	Prog *br;
//	Type *t;

//	split64(nl, &lo1, &hi1);
//	split64(nr, &lo2, &hi2);

//	// compare most significant word;
//	// if they differ, we're done.
//	t = hi1.type;
//	if(nl->op == OLITERAL || nr->op == OLITERAL)
//		gins(ACMPL, &hi1, &hi2);
//	else {
//		regalloc(&rr, types[TINT32], N);
//		gins(AMOVL, &hi1, &rr);
//		gins(ACMPL, &rr, &hi2);
//		regfree(&rr);
//	}
//	br = P;
//	switch(op) {
//	default:
//		fatal("cmp64 %O %T", op, t);
//	case OEQ:
//		// cmp hi
//		// jne L
//		// cmp lo
//		// jeq to
//		// L:
//		br = gbranch(AJNE, T);
//		break;
//	case ONE:
//		// cmp hi
//		// jne to
//		// cmp lo
//		// jne to
//		patch(gbranch(AJNE, T), to);
//		break;
//	case OGE:
//	case OGT:
//		// cmp hi
//		// jgt to
//		// jlt L
//		// cmp lo
//		// jge to (or jgt to)
//		// L:
//		patch(gbranch(optoas(OGT, t), T), to);
//		br = gbranch(optoas(OLT, t), T);
//		break;
//	case OLE:
//	case OLT:
//		// cmp hi
//		// jlt to
//		// jgt L
//		// cmp lo
//		// jle to (or jlt to)
//		// L:
//		patch(gbranch(optoas(OLT, t), T), to);
//		br = gbranch(optoas(OGT, t), T);
//		break;
//	}

//	// compare least significant word
//	t = lo1.type;
//	if(nl->op == OLITERAL || nr->op == OLITERAL)
//		gins(ACMPL, &lo1, &lo2);
//	else {
//		regalloc(&rr, types[TINT32], N);
//		gins(AMOVL, &lo1, &rr);
//		gins(ACMPL, &rr, &lo2);
//		regfree(&rr);
//	}

//	// jump again
//	patch(gbranch(optoas(op, t), T), to);

//	// point first branch down here if appropriate
//	if(br != P)
//		patch(br, pc);

//	splitclean();
//	splitclean();
}
