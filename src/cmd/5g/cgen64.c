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
	Node al, ah, bl, bh, cl, ch, s, n1, creg;
	Prog *p1, *p2, *p3;

	uint64 v;

	if(res->op != OINDREG && res->op != ONAME) {
		dump("n", n);
		dump("res", res);
		fatal("cgen64 %O of %O", n->op, res->op);
	}

	l = n->left;
	if(!l->addable) {
		tempalloc(&t1, l->type);
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

		gins(ASUB, &t1, &al);
		gins(ASBC, &t1, &ah);

		gins(AMOVW, &al, &lo2);
		gins(AMOVW, &ah, &hi2);

		regfree(&t1);
		regfree(&al);
		regfree(&ah);
		splitclean();
		splitclean();
		return;

	case OCOM:
		split64(res, &lo2, &hi2);
		regalloc(&n1, lo1.type, N);

		gins(AMOVW, &lo1, &n1);
		gins(AMVN, &n1, &n1);
		gins(AMOVW, &n1, &lo2);

		gins(AMOVW, &hi1, &n1);
		gins(AMVN, &n1, &n1);
		gins(AMOVW, &n1, &hi2);

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
		// binary operators.
		// common setup below.
		break;
	}

	// setup for binary operators
	r = n->right;
	if(r != N && !r->addable) {
		tempalloc(&t2, r->type);
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
		gins(AADD, &bl, &al);
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
		gins(ASUB, &bl, &al);
		gins(ASBC, &bh, &ah);
		regfree(&bl);
		regfree(&bh);
		break;

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

	case OLSH:
		regalloc(&bh, hi1.type, N);
		regalloc(&bl, lo1.type, N);
		gins(AMOVW, &hi1, &bh);
		gins(AMOVW, &lo1, &bl);

		if(r->op == OLITERAL) {
			v = mpgetfix(r->val.u.xval);
			if(v >= 64) {
				// TODO(kaib): replace with gins(AMOVW, nodintconst(0), &al)
				// here and below (verify it optimizes to EOR)
				gins(AEOR, &al, &al);
				gins(AEOR, &ah, &ah);
				goto olsh_break;
			}
			if(v >= 32) {
				gins(AEOR, &al, &al);
				//	MOVW	bl<<(v-32), ah
				p1 = gins(AMOVW, &bl, &ah);
				p1->from.type = D_SHIFT;
				p1->from.offset = SHIFT_LL | (v-32)<<7 | bl.val.u.reg;
				p1->from.reg = NREG;
				goto olsh_break;
			}

			// general literal left shift

			//	MOVW	bl<<v, al
			p1 = gins(AMOVW, &bl, &al);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_LL | v<<7 | bl.val.u.reg;
			p1->from.reg = NREG;

			//	MOVW	bh<<v, ah
			p1 = gins(AMOVW, &bh, &ah);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_LL | v<<7 | bh.val.u.reg;
			p1->from.reg = NREG;

			//	OR		bl>>(32-v), ah
			p1 = gins(AORR, &bl, &ah);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_LR | (32-v)<<7 | bl.val.u.reg;
			p1->from.reg = NREG;
			goto olsh_break;
		}

		regalloc(&s, types[TUINT32], N);
		regalloc(&creg, types[TUINT32], N);
		gmove(r, &s);

		// check if shift is < 32
		nodconst(&n1, types[TUINT32], 32);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	MOVW.LT		bl<<s, al
		p1 = gins(AMOVW, N, &al);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LL | s.val.u.reg << 8 | 1<<4 | bl.val.u.reg;
		p1->scond = C_SCOND_LT;

		//	MOVW.LT		bh<<s, al
		p1 = gins(AMOVW, N, &al);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LL | s.val.u.reg << 8 | 1<<4 | bh.val.u.reg;
		p1->scond = C_SCOND_LT;

		//	SUB.LT		creg, s
		p1 = gins(ASUB, &creg, &s);
		p1->scond = C_SCOND_LT;

		//	OR.LT		bl>>(32-s), ah
		p1 = gins(AMOVW, N, &ah);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LR | t1.val.u.reg<<8| 1<<4 | bl.val.u.reg;
		p1->scond = C_SCOND_LT;

		//	BLT	end
		p2 = gbranch(ABLT, T);

		// check if shift is < 64
		nodconst(&n1, types[TUINT32], 64);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	EOR.LT	al, al
		p1 = gins(AEOR, &al, &al);
		p1->scond = C_SCOND_LT;

		//	MOVW.LT		creg>>1, creg
		p1 = gins(AMOVW, N, &creg);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LR | 1<<7 | creg.val.u.reg;
		p1->scond = C_SCOND_LT;

		//	SUB.LT		creg, s
		p1 = gins(ASUB, &s, &creg);
		p1->scond = C_SCOND_LT;

		//	MOVW	bl<<(s-32), ah
		p1 = gins(AMOVW, N, &ah);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LL | s.val.u.reg<<8 | 1<<4 | bl.val.u.reg;
		p1->scond = C_SCOND_LT;

		p3 = gbranch(ABLT, T);

		gins(AEOR, &al, &al);
		gins(AEOR, &ah, &ah);

		patch(p2, pc);
		patch(p3, pc);
		regfree(&s);
		regfree(&creg);

olsh_break:
		regfree(&bl);
		regfree(&bh);
		break;


	case ORSH:
		regalloc(&bh, hi1.type, N);
		regalloc(&bl, lo1.type, N);
		gins(AMOVW, &hi1, &bh);
		gins(AMOVW, &lo1, &bl);

		if(r->op == OLITERAL) {
			v = mpgetfix(r->val.u.xval);
			if(v >= 64) {
				if(bh.type->etype == TINT32) {
					//	MOVW	bh->31, al
					p1 = gins(AMOVW, N, &al);
					p1->from.type = D_SHIFT;
					p1->from.offset = SHIFT_AR | 31 << 7 | bh.val.u.reg;

					//	MOVW	bh->31, ah
					p1 = gins(AMOVW, N, &ah);
					p1->from.type = D_SHIFT;
					p1->from.offset = SHIFT_AR | 31 << 7 | bh.val.u.reg;
				} else {
					gins(AEOR, &al, &al);
					gins(AEOR, &ah, &ah);
				}
				goto orsh_break;
			}
			if(v >= 32) {
				if(bh.type->etype == TINT32) {
					//	MOVW	bh->(v-32), al
					p1 = gins(AMOVW, N, &al);
					p1->from.type = D_SHIFT;
					p1->from.offset = SHIFT_AR | (v-32)<<7 | bh.val.u.reg;

					//	MOVW	bh->31, ah
					p1 = gins(AMOVW, N, &ah);
					p1->from.type = D_SHIFT;
					p1->from.offset = SHIFT_AR | 31<<7 | bh.val.u.reg;
				} else {
					//	MOVW	bh>>(v-32), al
					p1 = gins(AMOVW, N, &al);
					p1->from.type = D_SHIFT;
					p1->from.offset = SHIFT_LR | (v-32)<<7 | bh.val.u.reg;
					gins(AEOR, &ah, &ah);
				}
				goto orsh_break;
			}

			// general literal right shift

			//	MOVW	bl>>v, al
			p1 = gins(AMOVW, N, &al);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_LR | v<<7 | bl.val.u.reg;

			//	OR		bh<<(32-v), al, al
			p1 = gins(AORR, N, &al);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_LL | (32-v)<<7 | bh.val.u.reg;
			p1->reg = al.val.u.reg;

			if(bh.type->etype == TINT32) {
				//	MOVW	bh->v, ah
				p1 = gins(AMOVW, N, &ah);
				p1->from.type = D_SHIFT;
				p1->from.offset = SHIFT_AR | v<<7 | bh.val.u.reg;
			} else {
				//	MOVW	bh>>v, ah
				p1 = gins(AMOVW, N, &ah);
				p1->from.type = D_SHIFT;
				p1->from.offset = SHIFT_LR | v<<7 | bh.val.u.reg;
			}
			goto orsh_break;
		}

		regalloc(&s, types[TUINT32], N);
		regalloc(&creg, types[TUINT32], N);
		gmove(r, &s);

		// check if shift is < 32
		nodconst(&n1, types[TUINT32], 32);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	MOVW.LT		bl>>s, al
		p1 = gins(AMOVW, N, &al);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LR | s.val.u.reg << 8 | 1<<4 | bl.val.u.reg;
		p1->scond = C_SCOND_LT;

		//	SUB.LT		creg, s
		p1 = gins(ASUB, &creg, &s);
		p1->scond = C_SCOND_LT;

		//	OR.LT		bh<<(32-s), al, al
		p1 = gins(AORR, N, &al);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LL | creg.val.u.reg << 8 | 1<<4 | bh.val.u.reg;
		p1->reg = al.val.u.reg;
		p1->scond = C_SCOND_LT;

		if(bh.type->etype == TINT32) {
			//	MOVW	bh->s, ah
			p1 = gins(AMOVW, N, &ah);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_AR | s.val.u.reg << 8 | 1<<4 | bh.val.u.reg;
		} else {
			//	MOVW	bh>>s, ah
			p1 = gins(AMOVW, N, &ah);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_LR | s.val.u.reg << 8 | 1<<4 | bh.val.u.reg;
		}
		p1->scond = C_SCOND_LT;

		//	BLT	end
		p2 = gbranch(ABLT, T);

		// check if shift is < 64
		nodconst(&n1, types[TUINT32], 64);
		gmove(&n1, &creg);
		gcmp(ACMP, &s, &creg);

		//	MOVW.LT		creg>>1, creg
		p1 = gins(AMOVW, N, &creg);
		p1->from.type = D_SHIFT;
		p1->from.offset = SHIFT_LR | 1<<7 | creg.val.u.reg;
		p1->scond = C_SCOND_LT;

		//	SUB.LT		s, creg
		p1 = gins(ASUB, &s, &creg);
		p1->scond = C_SCOND_LT;

		if(bh.type->etype == TINT32) {
			//	MOVW	bh->(s-32), al
			p1 = gins(AMOVW, N, &al);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_AR | s.val.u.reg <<8 | 1<<4 | bh.val.u.reg;
			p1->scond = C_SCOND_LT;

			//	MOVW	bh->31, ah
			p1 = gins(AMOVW, N, &ah);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_AR | 31<<7 | bh.val.u.reg;
			p1->scond = C_SCOND_LT;
		} else {
			//	MOVW	bh>>(v-32), al
			p1 = gins(AMOVW, N, &al);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_LR | s.val.u.reg<<8 | 1<<4 | bh.val.u.reg;
			p1->scond = C_SCOND_LT;

			p1 = gins(AEOR, &ah, &ah);
			p1->scond = C_SCOND_LT;
		}

		//	BLT	end
		p3 = gbranch(ABLT, T);

		// s >= 64
		if(bh.type->etype == TINT32) {
			//	MOVW	bh->31, al
			p1 = gins(AMOVW, N, &al);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_AR | 31 << 7 | bh.val.u.reg;

			//	MOVW	bh->31, ah
			p1 = gins(AMOVW, N, &ah);
			p1->from.type = D_SHIFT;
			p1->from.offset = SHIFT_AR | 31 << 7 | bh.val.u.reg;
		} else {
			gins(AEOR, &al, &al);
			gins(AEOR, &ah, &ah);
		}

		patch(p2, pc);
		patch(p3, pc);
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
		br = gbranch(ABNE, T);
		break;
	case ONE:
		// cmp hi
		// bne to
		// cmp lo
		// bne to
		patch(gbranch(ABNE, T), to);
		break;
	case OGE:
	case OGT:
		// cmp hi
		// bgt to
		// blt L
		// cmp lo
		// bge to (or bgt to)
		// L:
		patch(gbranch(optoas(OGT, t), T), to);
		br = gbranch(optoas(OLT, t), T);
		break;
	case OLE:
	case OLT:
		// cmp hi
		// blt to
		// bgt L
		// cmp lo
		// ble to (or jlt to)
		// L:
		patch(gbranch(optoas(OLT, t), T), to);
		br = gbranch(optoas(OGT, t), T);
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
	patch(gbranch(optoas(op, t), T), to);

	// point first branch down here if appropriate
	if(br != P)
		patch(br, pc);

	splitclean();
	splitclean();
}
