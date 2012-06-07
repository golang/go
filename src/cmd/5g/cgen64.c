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
	Node t1, t2, *l, *r;
	Node lo1, lo2, hi1, hi2;
	Node al, ah, bl, bh, cl, ch, s, n1, creg;
	Prog *p1, *p2, *p3, *p4, *p5, *p6;

	uint64 v;

	if(res->op != OINDREG && res->op != ONAME) {
		dump("n", n);
		dump("res", res);
		fatal("cgen64 %O of %O", n->op, res->op);
	}

	l = n->left;
	if(!l->addable) {
		tempname(&t1, l->type);
		cgen(l, &t1);
		l = &t1;
	}

	split64(l, &lo1, &hi1);
	switch(n->op) {
	default:
		fatal("cgen64 %O", n->op);

	case OMINUS:
		split64(res, &lo2, &hi2);

		regalloc(&t1, lo1.type, N);
		regalloc(&al, lo1.type, N);
		regalloc(&ah, hi1.type, N);

		gins(AMOVW, &lo1, &al);
		gins(AMOVW, &hi1, &ah);

		gmove(ncon(0), &t1);
		p1 = gins(ASUB, &al, &t1);
		p1->scond |= C_SBIT;
		gins(AMOVW, &t1, &lo2);

		gmove(ncon(0), &t1);
		gins(ASBC, &ah, &t1);
		gins(AMOVW, &t1, &hi2);

		regfree(&t1);
		regfree(&al);
		regfree(&ah);
		splitclean();
		splitclean();
		return;

	case OCOM:
		regalloc(&t1, lo1.type, N);
		gmove(ncon(-1), &t1);

		split64(res, &lo2, &hi2);
		regalloc(&n1, lo1.type, N);

		gins(AMOVW, &lo1, &n1);
		gins(AEOR, &t1, &n1);
		gins(AMOVW, &n1, &lo2);

		gins(AMOVW, &hi1, &n1);
		gins(AEOR, &t1, &n1);
		gins(AMOVW, &n1, &hi2);

		regfree(&t1);
		regfree(&n1);
		splitclean();
		splitclean();
		return;

	case OADD:
	case OSUB:
	case OMUL:
	case OLSH:
	case ORSH:
	case OAND:
	case OOR:
	case OXOR:
	case OLROT:
		// binary operators.
		// common setup below.
		break;
	}

	// setup for binary operators
	r = n->right;
	if(r != N && !r->addable) {
		tempname(&t2, r->type);
		cgen(r, &t2);
		r = &t2;
	}
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
		p1 = gins(AADD, &bl, &al);
		p1->scond |= C_SBIT;
		gins(AADC, &bh, &ah);
		regfree(&bl);
		regfree(&bh);
		break;

	case OSUB:
		// TODO: Constants.
		regalloc(&bl, types[TPTR32], N);
		regalloc(&bh, types[TPTR32], N);
		gins(AMOVW, &lo1, &al);
		gins(AMOVW, &hi1, &ah);
		gins(AMOVW, &lo2, &bl);
		gins(AMOVW, &hi2, &bh);
		p1 = gins(ASUB, &bl, &al);
		p1->scond |= C_SBIT;
		gins(ASBC, &bh, &ah);
		regfree(&bl);
		regfree(&bh);
		break;

	case OMUL:
		// TODO(kaib): this can be done with 4 regs and does not need 6
		regalloc(&bl, types[TPTR32], N);
		regalloc(&bh, types[TPTR32], N);
		regalloc(&cl, types[TPTR32], N);
		regalloc(&ch, types[TPTR32], N);

		// load args into bh:bl and bh:bl.
		gins(AMOVW, &hi1, &bh);
		gins(AMOVW, &lo1, &bl);
		gins(AMOVW, &hi2, &ch);
		gins(AMOVW, &lo2, &cl);

		// bl * cl -> ah al
		p1 = gins(AMULLU, N, N);
		p1->from.type = D_REG;
		p1->from.reg = bl.val.u.reg;
		p1->reg = cl.val.u.reg;
		p1->to.type = D_REGREG;
		p1->to.reg = ah.val.u.reg;
		p1->to.offset = al.val.u.reg;
//print("%P\n", p1);

		// bl * ch + ah -> ah
		p1 = gins(AMULA, N, N);
		p1->from.type = D_REG;
		p1->from.reg = bl.val.u.reg;
		p1->reg = ch.val.u.reg;
		p1->to.type = D_REGREG2;
		p1->to.reg = ah.val.u.reg;
		p1->to.offset = ah.val.u.reg;
//print("%P\n", p1);

		// bh * cl + ah -> ah
		p1 = gins(AMULA, N, N);
		p1->from.type = D_REG;
		p1->from.reg = bh.val.u.reg;
		p1->reg = cl.val.u.reg;
		p1->to.type = D_REGREG2;
		p1->to.reg = ah.val.u.reg;
		p1->to.offset = ah.val.u.reg;
//print("%P\n", p1);

		regfree(&bh);
		regfree(&bl);
		regfree(&ch);
		regfree(&cl);

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
		regalloc(&bl, lo1.type, N);
		regalloc(&bh, hi1.type, N);
		if(v >= 32) {
			// reverse during load to do the first 32 bits of rotate
			v -= 32;
			gins(AMOVW, &hi1, &bl);
			gins(AMOVW, &lo1, &bh);
		} else {
			gins(AMOVW, &hi1, &bh);
			gins(AMOVW, &lo1, &bl);
		}
		if(v == 0) {
			gins(AMOVW, &bh, &ah);
			gins(AMOVW, &bl, &al);
		} else {
			// rotate by 1 <= v <= 31
			//	MOVW	bl<<v, al
			//	MOVW	bh<<v, ah
			//	OR		bl>>(32-v), ah
			//	OR		bh>>(32-v), al
			gshift(AMOVW, &bl, SHIFT_LL, v, &al);
			gshift(AMOVW, &bh, SHIFT_LL, v, &ah);
			gshift(AORR, &bl, SHIFT_LR, 32-v, &ah);
			gshift(AORR, &bh, SHIFT_LR, 32-v, &al);
		}
		regfree(&bl);
		regfree(&bh);
		break;

	case OLSH:
		regalloc(&bl, lo1.type, N);
		regalloc(&bh, hi1.type, N);
		gins(AMOVW, &hi1, &bh);
		gins(AMOVW, &lo1, &bl);

		if(r->op == OLITERAL) {
			v = mpgetfix(r->val.u.xval);
			if(v >= 64) {
				// TODO(kaib): replace with gins(AMOVW, nodintconst(0), &al)
				// here and below (verify it optimizes to EOR)
				gins(AEOR, &al, &al);
				gins(AEOR, &ah, &ah);
			} else
			if(v > 32) {
				gins(AEOR, &al, &al);
				//	MOVW	bl<<(v-32), ah
				gshift(AMOVW, &bl, SHIFT_LL, (v-32), &ah);
			} else
			if(v == 32) {
				gins(AEOR, &al, &al);
				gins(AMOVW, &bl, &ah);
			} else
			if(v > 0) {
				//	MOVW	bl<<v, al
				gshift(AMOVW, &bl, SHIFT_LL, v, &al);

				//	MOVW	bh<<v, ah
				gshift(AMOVW, &bh, SHIFT_LL, v, &ah);

				//	OR		bl>>(32-v), ah
				gshift(AORR, &bl, SHIFT_LR, 32-v, &ah);
			} else {
				gins(AMOVW, &bl, &al);
				gins(AMOVW, &bh, &ah);
			}
			goto olsh_break;
		}

		regalloc(&s, types[TUINT32], N);
		regalloc(&creg, types[TUINT32], N);
		if (is64(r->type)) {
			// shift is >= 1<<32
			split64(r, &cl, &ch);
			gmove(&ch, &s);
			gins(ATST, &s, N);
			p6 = gbranch(ABNE, T, 0);
			gmove(&cl, &s);
			splitclean();
		} else {
			gmove(r, &s);
			p6 = P;
		}
		gins(ATST, &s, N);

		// shift == 0
		p1 = gins(AMOVW, &bl, &al);
		p1->scond = C_SCOND_EQ;
		p1 = gins(AMOVW, &bh, &ah);
		p1->scond = C_SCOND_EQ;
		p2 = gbranch(ABEQ, T, 0);

		// shift is < 32
		nodconst(&n1, types[TUINT32], 32);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	MOVW.LO		bl<<s, al
		p1 = gregshift(AMOVW, &bl, SHIFT_LL, &s, &al);
		p1->scond = C_SCOND_LO;

		//	MOVW.LO		bh<<s, ah
		p1 = gregshift(AMOVW, &bh, SHIFT_LL, &s, &ah);
		p1->scond = C_SCOND_LO;

		//	SUB.LO		s, creg
		p1 = gins(ASUB, &s, &creg);
		p1->scond = C_SCOND_LO;

		//	OR.LO		bl>>creg, ah
		p1 = gregshift(AORR, &bl, SHIFT_LR, &creg, &ah);
		p1->scond = C_SCOND_LO;

		//	BLO	end
		p3 = gbranch(ABLO, T, 0);

		// shift == 32
		p1 = gins(AEOR, &al, &al);
		p1->scond = C_SCOND_EQ;
		p1 = gins(AMOVW, &bl, &ah);
		p1->scond = C_SCOND_EQ;
		p4 = gbranch(ABEQ, T, 0);

		// shift is < 64
		nodconst(&n1, types[TUINT32], 64);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	EOR.LO	al, al
		p1 = gins(AEOR, &al, &al);
		p1->scond = C_SCOND_LO;

		//	MOVW.LO		creg>>1, creg
		p1 = gshift(AMOVW, &creg, SHIFT_LR, 1, &creg);
		p1->scond = C_SCOND_LO;

		//	SUB.LO		creg, s
		p1 = gins(ASUB, &creg, &s);
		p1->scond = C_SCOND_LO;

		//	MOVW	bl<<s, ah
		p1 = gregshift(AMOVW, &bl, SHIFT_LL, &s, &ah);
		p1->scond = C_SCOND_LO;

		p5 = gbranch(ABLO, T, 0);

		// shift >= 64
		if (p6 != P) patch(p6, pc);
		gins(AEOR, &al, &al);
		gins(AEOR, &ah, &ah);

		patch(p2, pc);
		patch(p3, pc);
		patch(p4, pc);
		patch(p5, pc);
		regfree(&s);
		regfree(&creg);

olsh_break:
		regfree(&bl);
		regfree(&bh);
		break;


	case ORSH:
		regalloc(&bl, lo1.type, N);
		regalloc(&bh, hi1.type, N);
		gins(AMOVW, &hi1, &bh);
		gins(AMOVW, &lo1, &bl);

		if(r->op == OLITERAL) {
			v = mpgetfix(r->val.u.xval);
			if(v >= 64) {
				if(bh.type->etype == TINT32) {
					//	MOVW	bh->31, al
					gshift(AMOVW, &bh, SHIFT_AR, 31, &al);

					//	MOVW	bh->31, ah
					gshift(AMOVW, &bh, SHIFT_AR, 31, &ah);
				} else {
					gins(AEOR, &al, &al);
					gins(AEOR, &ah, &ah);
				}
			} else
			if(v > 32) {
				if(bh.type->etype == TINT32) {
					//	MOVW	bh->(v-32), al
					gshift(AMOVW, &bh, SHIFT_AR, v-32, &al);

					//	MOVW	bh->31, ah
					gshift(AMOVW, &bh, SHIFT_AR, 31, &ah);
				} else {
					//	MOVW	bh>>(v-32), al
					gshift(AMOVW, &bh, SHIFT_LR, v-32, &al);
					gins(AEOR, &ah, &ah);
				}
			} else
			if(v == 32) {
				gins(AMOVW, &bh, &al);
				if(bh.type->etype == TINT32) {
					//	MOVW	bh->31, ah
					gshift(AMOVW, &bh, SHIFT_AR, 31, &ah);
				} else {
					gins(AEOR, &ah, &ah);
				}
			} else
			if( v > 0) {
				//	MOVW	bl>>v, al
				gshift(AMOVW, &bl, SHIFT_LR, v, &al);
	
				//	OR		bh<<(32-v), al
				gshift(AORR, &bh, SHIFT_LL, 32-v, &al);

				if(bh.type->etype == TINT32) {
					//	MOVW	bh->v, ah
					gshift(AMOVW, &bh, SHIFT_AR, v, &ah);
				} else {
					//	MOVW	bh>>v, ah
					gshift(AMOVW, &bh, SHIFT_LR, v, &ah);
				}
			} else {
				gins(AMOVW, &bl, &al);
				gins(AMOVW, &bh, &ah);
			}
			goto orsh_break;
		}

		regalloc(&s, types[TUINT32], N);
		regalloc(&creg, types[TUINT32], N);
		if(is64(r->type)) {
			// shift is >= 1<<32
			split64(r, &cl, &ch);
			gmove(&ch, &s);
			gins(ATST, &s, N);
			if(bh.type->etype == TINT32)
				p1 = gshift(AMOVW, &bh, SHIFT_AR, 31, &ah);
			else
				p1 = gins(AEOR, &ah, &ah);
			p1->scond = C_SCOND_NE;
			p6 = gbranch(ABNE, T, 0);
			gmove(&cl, &s);
			splitclean();
		} else {
			gmove(r, &s);
			p6 = P;
		}
		gins(ATST, &s, N);

		// shift == 0
		p1 = gins(AMOVW, &bl, &al);
		p1->scond = C_SCOND_EQ;
		p1 = gins(AMOVW, &bh, &ah);
		p1->scond = C_SCOND_EQ;
		p2 = gbranch(ABEQ, T, 0);

		// check if shift is < 32
		nodconst(&n1, types[TUINT32], 32);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	MOVW.LO		bl>>s, al
		p1 = gregshift(AMOVW, &bl, SHIFT_LR, &s, &al);
		p1->scond = C_SCOND_LO;

		//	SUB.LO		s,creg
		p1 = gins(ASUB, &s, &creg);
		p1->scond = C_SCOND_LO;

		//	OR.LO		bh<<(32-s), al
		p1 = gregshift(AORR, &bh, SHIFT_LL, &creg, &al);
		p1->scond = C_SCOND_LO;

		if(bh.type->etype == TINT32) {
			//	MOVW	bh->s, ah
			p1 = gregshift(AMOVW, &bh, SHIFT_AR, &s, &ah);
		} else {
			//	MOVW	bh>>s, ah
			p1 = gregshift(AMOVW, &bh, SHIFT_LR, &s, &ah);
		}
		p1->scond = C_SCOND_LO;

		//	BLO	end
		p3 = gbranch(ABLO, T, 0);

		// shift == 32
		p1 = gins(AMOVW, &bh, &al);
		p1->scond = C_SCOND_EQ;
		if(bh.type->etype == TINT32)
			gshift(AMOVW, &bh, SHIFT_AR, 31, &ah);
		else
			gins(AEOR, &ah, &ah);
		p4 = gbranch(ABEQ, T, 0);

		// check if shift is < 64
		nodconst(&n1, types[TUINT32], 64);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	MOVW.LO		creg>>1, creg
		p1 = gshift(AMOVW, &creg, SHIFT_LR, 1, &creg);
		p1->scond = C_SCOND_LO;

		//	SUB.LO		creg, s
		p1 = gins(ASUB, &creg, &s);
		p1->scond = C_SCOND_LO;

		if(bh.type->etype == TINT32) {
			//	MOVW	bh->(s-32), al
			p1 = gregshift(AMOVW, &bh, SHIFT_AR, &s, &al);
			p1->scond = C_SCOND_LO;
		} else {
			//	MOVW	bh>>(v-32), al
			p1 = gregshift(AMOVW, &bh, SHIFT_LR, &s, &al);
			p1->scond = C_SCOND_LO;
		}

		//	BLO	end
		p5 = gbranch(ABLO, T, 0);

		// s >= 64
		if(p6 != P)
			patch(p6, pc);
		if(bh.type->etype == TINT32) {
			//	MOVW	bh->31, al
			gshift(AMOVW, &bh, SHIFT_AR, 31, &al);
		} else {
			gins(AEOR, &al, &al);
		}

		patch(p2, pc);
		patch(p3, pc);
		patch(p4, pc);
		patch(p5, pc);
		regfree(&s);
		regfree(&creg);


orsh_break:
		regfree(&bl);
		regfree(&bh);
		break;

	case OXOR:
	case OAND:
	case OOR:
		// TODO(kaib): literal optimizations
		// make constant the right side (it usually is anyway).
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
		regalloc(&n1, lo1.type, N);
		gins(AMOVW, &lo1, &al);
		gins(AMOVW, &hi1, &ah);
		gins(AMOVW, &lo2, &n1);
		gins(optoas(n->op, lo1.type), &n1, &al);
		gins(AMOVW, &hi2, &n1);
		gins(optoas(n->op, lo1.type), &n1, &ah);
		regfree(&n1);
		break;
	}
	if(is64(r->type))
		splitclean();
	splitclean();

	split64(res, &lo1, &hi1);
	gins(AMOVW, &al, &lo1);
	gins(AMOVW, &ah, &hi1);
	splitclean();

//out:
	regfree(&al);
	regfree(&ah);
}

/*
 * generate comparison of nl, nr, both 64-bit.
 * nl is memory; nr is constant or memory.
 */
void
cmp64(Node *nl, Node *nr, int op, int likely, Prog *to)
{
	Node lo1, hi1, lo2, hi2, r1, r2;
	Prog *br;
	Type *t;

	split64(nl, &lo1, &hi1);
	split64(nr, &lo2, &hi2);

	// compare most significant word;
	// if they differ, we're done.
	t = hi1.type;
	regalloc(&r1, types[TINT32], N);
	regalloc(&r2, types[TINT32], N);
	gins(AMOVW, &hi1, &r1);
	gins(AMOVW, &hi2, &r2);
	gcmp(ACMP, &r1, &r2);
	regfree(&r1);
	regfree(&r2);

	br = P;
	switch(op) {
	default:
		fatal("cmp64 %O %T", op, t);
	case OEQ:
		// cmp hi
		// bne L
		// cmp lo
		// beq to
		// L:
		br = gbranch(ABNE, T, -likely);
		break;
	case ONE:
		// cmp hi
		// bne to
		// cmp lo
		// bne to
		patch(gbranch(ABNE, T, likely), to);
		break;
	case OGE:
	case OGT:
		// cmp hi
		// bgt to
		// blt L
		// cmp lo
		// bge to (or bgt to)
		// L:
		patch(gbranch(optoas(OGT, t), T, likely), to);
		br = gbranch(optoas(OLT, t), T, -likely);
		break;
	case OLE:
	case OLT:
		// cmp hi
		// blt to
		// bgt L
		// cmp lo
		// ble to (or jlt to)
		// L:
		patch(gbranch(optoas(OLT, t), T, likely), to);
		br = gbranch(optoas(OGT, t), T, -likely);
		break;
	}

	// compare least significant word
	t = lo1.type;
	regalloc(&r1, types[TINT32], N);
	regalloc(&r2, types[TINT32], N);
	gins(AMOVW, &lo1, &r1);
	gins(AMOVW, &lo2, &r2);
	gcmp(ACMP, &r1, &r2);
	regfree(&r1);
	regfree(&r2);

	// jump again
	patch(gbranch(optoas(op, t), T, likely), to);

	// point first branch down here if appropriate
	if(br != P)
		patch(br, pc);

	splitclean();
	splitclean();
}
