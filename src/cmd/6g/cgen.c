// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

/*
 * generate:
 *	res = n;
 * simplifies and calls gmove.
 */
void
cgen(Node *n, Node *res)
{
	Node *nl, *nr, *r;
	Node n1, n2;
	int a, f;
	Prog *p1, *p2, *p3;
	Addr addr;

	if(debug['g']) {
		dump("\ncgen-n", n);
		dump("cgen-res", res);
	}
	if(n == N || n->type == T)
		return;

	if(res == N || res->type == T)
		fatal("cgen: res nil");

	if(n->ullman >= UINF) {
		if(n->op == OINDREG)
			fatal("cgen: this is going to misscompile");
		if(res->ullman >= UINF) {
			tempname(&n1, n->type);
			cgen(n, &n1);
			cgen(&n1, res);
			goto ret;
		}
	}

	if(isfat(n->type)) {
		sgen(n, res, n->type->width);
		goto ret;
	}

	if(!res->addable) {
		if(n->ullman > res->ullman) {
			regalloc(&n1, n->type, res);
			cgen(n, &n1);
			if(n1.ullman > res->ullman) {
				dump("n1", &n1);
				dump("res", res);
				fatal("loop in cgen");
			}
			cgen(&n1, res);
			regfree(&n1);
			goto ret;
		}

		if(res->ullman >= UINF)
			goto gen;

		f = 1;	// gen thru register
		switch(n->op) {
		case OLITERAL:
			if(smallintconst(n))
				f = 0;
			break;
		case OREGISTER:
			f = 0;
			break;
		}

		if(sudoaddable(res, &addr)) {
			a = optoas(OAS, res->type);
			if(f) {
				regalloc(&n2, res->type, N);
				cgen(n, &n2);
				p1 = gins(a, &n2, N);
				regfree(&n2);
			} else
				p1 = gins(a, n, N);
			p1->to = addr;
			if(debug['g'])
				print("%P [ignore previous line]\n", p1);
			sudoclean();
			goto ret;
		}

	gen:
		igen(res, &n1, N);
		cgen(n, &n1);
		regfree(&n1);
		goto ret;
	}

	if(n->addable) {
		gmove(n, res);
		goto ret;
	}

	nl = n->left;
	nr = n->right;
	if(nl != N && nl->ullman >= UINF)
	if(nr != N && nr->ullman >= UINF) {
		tempname(&n1, nr->type);
		cgen(nr, &n1);
		n2 = *n;
		n2.right = &n1;
		cgen(&n2, res);
		goto ret;
	}

	if(sudoaddable(n, &addr)) {
		a = optoas(OAS, n->type);
		if(res->op == OREGISTER) {
			p1 = gins(a, N, res);
			p1->from = addr;
		} else {
			regalloc(&n2, n->type, N);
			p1 = gins(a, N, &n2);
			p1->from = addr;
			gins(a, &n2, res);
			regfree(&n2);
		}
		sudoclean();
		goto ret;
	}

	switch(n->op) {
	default:
		dump("cgen", n);
		fatal("cgen: unknown op %N", n);
		break;

	// these call bgen to get a bool value
	case OOROR:
	case OANDAND:
	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
	case ONOT:
		p1 = gbranch(AJMP, T);
		p2 = pc;
		gmove(nodbool(1), res);
		p3 = gbranch(AJMP, T);
		patch(p1, pc);
		bgen(n, 1, p2);
		gmove(nodbool(0), res);
		patch(p3, pc);
		goto ret;

	case OPLUS:
		cgen(nl, res);
		goto ret;

	// unary
	case OCOM:
		a = optoas(OXOR, nl->type);
		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);
		nodconst(&n2, nl->type, -1);
		gins(a, &n2, &n1);
		gmove(&n1, res);
		regfree(&n1);
		goto ret;

	case OMINUS:
		a = optoas(n->op, nl->type);
		goto uop;

	// symmetric binary
	case OAND:
	case OOR:
	case OXOR:
	case OADD:
	case OMUL:
		a = optoas(n->op, nl->type);
		if(a != AIMULB)
			goto sbop;
		cgen_bmul(n->op, nl, nr, res);
		break;

	// asymmetric binary
	case OSUB:
		a = optoas(n->op, nl->type);
		goto abop;

	case OCONV:
		if(eqtype(n->type, nl->type, 0)) {
			cgen(nl, res);
			break;
		}
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		gmove(&n1, res);
		regfree(&n1);
		break;

	case ODOT:
	case ODOTPTR:
	case OINDEX:
	case OIND:
	case ONAME:	// PHEAP or PPARAMREF var
		igen(n, &n1, res);
		gmove(&n1, res);
		regfree(&n1);
		break;

	case OLEN:
		if(istype(nl->type, TSTRING) || istype(nl->type, TMAP)) {
			// both string and map have len in the first 32-bit word.
			// a zero pointer means zero length
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);

			nodconst(&n2, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n1, &n2);
			p1 = gbranch(optoas(OEQ, types[tptr]), T);

			n2 = n1;
			n2.op = OINDREG;
			n2.type = types[TINT32];
			gmove(&n2, &n1);

			patch(p1, pc);

			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		if(isslice(nl->type)) {
			regalloc(&n1, types[tptr], res);
			agen(nl, &n1);
			n1.op = OINDREG;
			n1.type = types[TUINT32];
			n1.xoffset = Array_nel;
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		fatal("cgen: OLEN: unknown type %lT", nl->type);
		break;

	case OCAP:
		if(isslice(nl->type)) {
			regalloc(&n1, types[tptr], res);
			agen(nl, &n1);
			n1.op = OINDREG;
			n1.type = types[TUINT32];
			n1.xoffset = Array_cap;
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		fatal("cgen: OCAP: unknown type %lT", nl->type);
		break;

	case OADDR:
		agen(nl, res);
		break;

	case OCALLMETH:
		cgen_callmeth(n, 0);
		cgen_callret(n, res);
		break;

	case OCALLINTER:
		cgen_callinter(n, res, 0);
		cgen_callret(n, res);
		break;

	case OCALL:
		cgen_call(n, 0);
		cgen_callret(n, res);
		break;

	case OMOD:
	case ODIV:
		if(isfloat[n->type->etype]) {
			a = optoas(n->op, nl->type);
			goto abop;
		}
		cgen_div(n->op, nl, nr, res);
		break;

	case OLSH:
	case ORSH:
		cgen_shift(n->op, nl, nr, res);
		break;
	}
	goto ret;

sbop:	// symmetric binary
	if(nl->ullman < nr->ullman) {
		r = nl;
		nl = nr;
		nr = r;
	}

abop:	// asymmetric binary
	if(nl->ullman >= nr->ullman) {
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);

		if(sudoaddable(nr, &addr)) {
			p1 = gins(a, N, &n1);
			p1->from = addr;
			gmove(&n1, res);
			sudoclean();
			regfree(&n1);
			goto ret;
		}

		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);
	} else {
		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
	}
	gins(a, &n2, &n1);
	gmove(&n1, res);
	regfree(&n1);
	regfree(&n2);
	goto ret;

uop:	// unary
	regalloc(&n1, nl->type, res);
	cgen(nl, &n1);
	gins(a, N, &n1);
	gmove(&n1, res);
	regfree(&n1);
	goto ret;

ret:
	;
}

/*
 * generate:
 *	res = &n;
 */
void
agen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, tmp;
	Prog *p1;
	uint32 w;
	uint64 v;
	Type *t;

	if(debug['g']) {
		dump("\nagen-res", res);
		dump("agen-r", n);
	}
	if(n == N || n->type == T)
		return;

	if(!isptr[res->type->etype])
		fatal("agen: not tptr: %T", res->type);

	if(n->addable) {
		regalloc(&n1, types[tptr], res);
		gins(ALEAQ, n, &n1);
		gmove(&n1, res);
		regfree(&n1);
		goto ret;
	}

	nl = n->left;
	nr = n->right;

	switch(n->op) {
	default:
		fatal("agen: unknown op %N", n);
		break;

	case OCONV:
		if(!eqtype(n->type, nl->type, 0))
			fatal("agen: non-trivial OCONV");
		agen(nl, res);
		return;

	case OCALLMETH:
		cgen_callmeth(n, 0);
		cgen_aret(n, res);
		break;

	case OCALLINTER:
		cgen_callinter(n, res, 0);
		cgen_aret(n, res);
		break;

	case OCALL:
		cgen_call(n, 0);
		cgen_aret(n, res);
		break;

	case OINDEX:
		w = n->type->width;
		if(nr->addable)
			goto irad;
		if(nl->addable) {
			if(!isconst(nr, CTINT)) {
				regalloc(&n1, nr->type, N);
				cgen(nr, &n1);
			}
			regalloc(&n3, types[tptr], res);
			agen(nl, &n3);
			goto index;
		}
		cgen(nr, res);
		tempname(&tmp, nr->type);
		gmove(res, &tmp);

	irad:
		regalloc(&n3, types[tptr], res);
		agen(nl, &n3);
		if(!isconst(nr, CTINT)) {
			regalloc(&n1, nr->type, N);
			cgen(nr, &n1);
		}
		goto index;

	index:
		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// w is width

		if(w == 0)
			fatal("index is zero width");

		// constant index
		if(isconst(nr, CTINT)) {
			v = mpgetfix(nr->val.u.xval);
			if(isslice(nl->type)) {

				if(!debug['B']) {
					n1 = n3;
					n1.op = OINDREG;
					n1.type = types[tptr];
					n1.xoffset = Array_nel;
					nodconst(&n2, types[TUINT64], v);
					gins(optoas(OCMP, types[TUINT32]), &n1, &n2);
					p1 = gbranch(optoas(OGT, types[TUINT32]), T);
					ginscall(throwindex, 0);
					patch(p1, pc);
				}

				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_array;
				gmove(&n1, &n3);
			} else
			if(!debug['B']) {
				if(v < 0)
					yyerror("out of bounds on array");
				else
				if(v >= nl->type->bound)
					yyerror("out of bounds on array");
			}

			nodconst(&n2, types[tptr], v*w);
			gins(optoas(OADD, types[tptr]), &n2, &n3);

			gmove(&n3, res);
			regfree(&n3);
			break;
		}

		// type of the index
		t = types[TUINT64];
		if(issigned[n1.type->etype])
			t = types[TINT64];

		regalloc(&n2, t, &n1);			// i
		gmove(&n1, &n2);
		regfree(&n1);

		if(!debug['B']) {
			// check bounds
			if(isslice(nl->type)) {
				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_nel;
			} else
				nodconst(&n1, types[TUINT64], nl->type->bound);
			gins(optoas(OCMP, types[TUINT32]), &n2, &n1);
			p1 = gbranch(optoas(OLT, types[TUINT32]), T);
			ginscall(throwindex, 0);
			patch(p1, pc);
		}

		if(isslice(nl->type)) {
			n1 = n3;
			n1.op = OINDREG;
			n1.type = types[tptr];
			n1.xoffset = Array_array;
			gmove(&n1, &n3);
		}

		if(w == 1 || w == 2 || w == 4 || w == 8) {
			p1 = gins(ALEAQ, &n2, &n3);
			p1->from.scale = w;
			p1->from.index = p1->from.type;
			p1->from.type = p1->to.type + D_INDIR;
		} else {
			nodconst(&n1, t, w);
			gins(optoas(OMUL, t), &n1, &n2);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
			gmove(&n3, res);
		}

		gmove(&n3, res);
		regfree(&n2);
		regfree(&n3);
		break;

	case ONAME:
		// should only get here with names in this func.
		if(n->funcdepth > 0 && n->funcdepth != funcdepth) {
			dump("bad agen", n);
			fatal("agen: bad ONAME funcdepth %d != %d",
				n->funcdepth, funcdepth);
		}

		// should only get here for heap vars or paramref
		if(!(n->class & PHEAP) && n->class != PPARAMREF) {
			dump("bad agen", n);
			fatal("agen: bad ONAME class %#x", n->class);
		}
		cgen(n->heapaddr, res);
		if(n->xoffset != 0) {
			nodconst(&n1, types[TINT64], n->xoffset);
			gins(optoas(OADD, types[tptr]), &n1, res);
		}
		break;

	case OIND:
		cgen(nl, res);
		break;

	case ODOT:
		t = nl->type;
		agen(nl, res);
		if(n->xoffset != 0) {
			nodconst(&n1, types[TINT64], n->xoffset);
			gins(optoas(OADD, types[tptr]), &n1, res);
		}
		break;

	case ODOTPTR:
		t = nl->type;
		if(!isptr[t->etype])
			fatal("agen: not ptr %N", n);
		cgen(nl, res);
		if(n->xoffset != 0) {
			nodconst(&n1, types[TINT64], n->xoffset);
			gins(optoas(OADD, types[tptr]), &n1, res);
		}
		break;
	}

ret:
	;
}

/*
 * generate:
 *	newreg = &n;
 *	res = newreg
 *
 * on exit, a has been changed to be *newreg.
 * caller must regfree(a).
 */
void
igen(Node *n, Node *a, Node *res)
{
	regalloc(a, types[tptr], res);
	agen(n, a);
	a->op = OINDREG;
	a->type = n->type;
}

/*
 * generate:
 *	if(n == true) goto to;
 */
void
bgen(Node *n, int true, Prog *to)
{
	int et, a;
	Node *nl, *nr, *r;
	Node n1, n2, tmp;
	Prog *p1, *p2;

	if(debug['g']) {
		dump("\nbgen", n);
	}

	if(n == N)
		n = nodbool(1);

	nl = n->left;
	nr = n->right;

	if(n->type == T) {
		convlit(n, types[TBOOL]);
		if(n->type == T)
			goto ret;
	}

	et = n->type->etype;
	if(et != TBOOL) {
		yyerror("cgen: bad type %T for %O", n->type, n->op);
		patch(gins(AEND, N, N), to);
		goto ret;
	}
	nl = N;
	nr = N;

	switch(n->op) {
	default:
		regalloc(&n1, n->type, N);
		cgen(n, &n1);
		nodconst(&n2, n->type, 0);
		gins(optoas(OCMP, n->type), &n1, &n2);
		a = AJNE;
		if(!true)
			a = AJEQ;
		patch(gbranch(a, n->type), to);
		regfree(&n1);
		goto ret;

	case OLITERAL:
// need to ask if it is bool?
		if(!true == !n->val.u.bval)
			patch(gbranch(AJMP, T), to);
		goto ret;

	case ONAME:
		nodconst(&n1, n->type, 0);
		gins(optoas(OCMP, n->type), n, &n1);
		a = AJNE;
		if(!true)
			a = AJEQ;
		patch(gbranch(a, n->type), to);
		goto ret;

	case OANDAND:
		if(!true)
			goto caseor;

	caseand:
		p1 = gbranch(AJMP, T);
		p2 = gbranch(AJMP, T);
		patch(p1, pc);
		bgen(n->left, !true, p2);
		bgen(n->right, !true, p2);
		p1 = gbranch(AJMP, T);
		patch(p1, to);
		patch(p2, pc);
		goto ret;

	case OOROR:
		if(!true)
			goto caseand;

	caseor:
		bgen(n->left, true, to);
		bgen(n->right, true, to);
		goto ret;

	case OEQ:
	case ONE:
	case OLT:
	case OGT:
	case OLE:
	case OGE:
		nr = n->right;
		if(nr == N || nr->type == T)
			goto ret;

	case ONOT:	// unary
		nl = n->left;
		if(nl == N || nl->type == T)
			goto ret;
	}

	switch(n->op) {

	case ONOT:
		bgen(nl, !true, to);
		goto ret;

	case OEQ:
	case ONE:
	case OLT:
	case OGT:
	case OLE:
	case OGE:
		a = n->op;
		if(!true)
			a = brcom(a);

		// make simplest on right
		if(nl->op == OLITERAL || nl->ullman < nr->ullman) {
			a = brrev(a);
			r = nl;
			nl = nr;
			nr = r;
		}

		if(isslice(nl->type)) {
			// only valid to cmp darray to literal nil
			if((a != OEQ && a != ONE) || nr->op != OLITERAL) {
				yyerror("illegal array comparison");
				break;
			}
			a = optoas(a, types[tptr]);
			regalloc(&n1, types[tptr], N);
			agen(nl, &n1);
			n2 = n1;
			n2.op = OINDREG;
			n2.xoffset = Array_array;
			nodconst(&tmp, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n2, &tmp);
			patch(gbranch(a, types[tptr]), to);
			regfree(&n1);
			break;
		}

		a = optoas(a, nr->type);

		if(nr->ullman >= UINF) {
			regalloc(&n1, nr->type, N);
			cgen(nr, &n1);

			tempname(&tmp, nr->type);
			gmove(&n1, &tmp);
			regfree(&n1);

			regalloc(&n1, nl->type, N);
			cgen(nl, &n1);

			regalloc(&n2, nr->type, &n2);
			cgen(&tmp, &n2);

			gins(optoas(OCMP, nr->type), &n1, &n2);
			patch(gbranch(a, nr->type), to);

			regfree(&n1);
			regfree(&n2);
			break;
		}

		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);

		if(smallintconst(nr)) {
			gins(optoas(OCMP, nr->type), &n1, nr);
			patch(gbranch(a, nr->type), to);
			regfree(&n1);
			break;
		}

		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);

		gins(optoas(OCMP, nr->type), &n1, &n2);
		patch(gbranch(a, nr->type), to);

		regfree(&n1);
		regfree(&n2);
		break;
	}
	goto ret;

ret:
	;
}

/*
 * n is on stack, either local variable
 * or return value from function call.
 * return n's offset from SP.
 */
int32
stkof(Node *n)
{
	Type *t;
	Iter flist;

	switch(n->op) {
	case OINDREG:
		return n->xoffset;

	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
		t = n->left->type;
		if(isptr[t->etype])
			t = t->type;

		t = structfirst(&flist, getoutarg(t));
		if(t != T)
			return t->width;
		break;
	}

	// botch - probably failing to recognize address
	// arithmetic on the above. eg INDEX and DOT
	return -1000;
}

/*
 * block copy:
 *	memmove(&n, &ns, w);
 */
void
sgen(Node *n, Node *ns, int32 w)
{
	Node nodl, nodr;
	int32 c, q, odst, osrc;

	if(debug['g']) {
		print("\nsgen w=%d\n", w);
		dump("r", n);
		dump("res", ns);
	}
	if(w == 0)
		return;
	if(n->ullman >= UINF && ns->ullman >= UINF) {
		fatal("sgen UINF");
	}

	if(w < 0)
		fatal("sgen copy %d", w);

	// offset on the stack
	osrc = stkof(n);
	odst = stkof(ns);

	nodreg(&nodl, types[tptr], D_DI);
	nodreg(&nodr, types[tptr], D_SI);

	if(n->ullman >= ns->ullman) {
		agen(n, &nodr);
		agen(ns, &nodl);
	} else {
		agen(ns, &nodl);
		agen(n, &nodr);
	}

	c = w % 8;	// bytes
	q = w / 8;	// quads

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if(osrc < odst && odst < osrc+w) {
		// reverse direction
		gins(ASTD, N, N);		// set direction flag
		if(c > 0) {
			gconreg(AADDQ, w-1, D_SI);
			gconreg(AADDQ, w-1, D_DI);

			gconreg(AMOVQ, c, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSB, N, N);	// MOVB *(SI)-,*(DI)-
		}

		if(q > 0) {
			if(c > 0) {
				gconreg(AADDQ, -7, D_SI);
				gconreg(AADDQ, -7, D_DI);
			} else {
				gconreg(AADDQ, w-8, D_SI);
				gconreg(AADDQ, w-8, D_DI);
			}
			gconreg(AMOVQ, q, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSQ, N, N);	// MOVQ *(SI)-,*(DI)-
		}
		// we leave with the flag clear
		gins(ACLD, N, N);
	} else {
		// normal direction
		if(q >= 4) {
			gconreg(AMOVQ, q, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSQ, N, N);	// MOVQ *(SI)+,*(DI)+
		} else
		while(q > 0) {
			gins(AMOVSQ, N, N);	// MOVQ *(SI)+,*(DI)+
			q--;
		}

		if(c >= 4) {
			gins(AMOVSL, N, N);	// MOVL *(SI)+,*(DI)+
			c -= 4;
		}
		while(c > 0) {
			gins(AMOVSB, N, N);	// MOVB *(SI)+,*(DI)+
			c--;
		}
	}
}
