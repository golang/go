// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

void
cgen(Node *n, Node *res)
{
	Node *nl, *nr, *r;
	Node n1, n2;
	int a;
	Prog *p1, *p2, *p3;

	if(debug['g']) {
		dump("\ncgen-res", res);
		dump("cgen-r", n);
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
			cgen(&n1, res);
			regfree(&n1);
			goto ret;
		}

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
		gmove(booltrue, res);
		p3 = gbranch(AJMP, T);
		patch(p1, pc);
		bgen(n, 1, p2);
		gmove(boolfalse, res);
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
		goto sbop;

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
		if(isptrarray(n->type) && isptrdarray(nl->type)) {
			// convert dynamic array to static array
			n2 = n1;
			n2.op = OINDREG;
			n2.xoffset = offsetof(Array,array);
			n2.type = types[tptr];
			gins(AMOVQ, &n2, &n1);
		}
		if(isptrdarray(n->type) && isptrarray(nl->type)) {
			// conver static array to dynamic array
			// it is assumed that the dope is just before the array
			nodconst(&n2, types[tptr], offsetof(Array,b));
			gins(ASUBQ, &n2, &n1);
		}
		gmove(&n1, res);
		regfree(&n1);
		break;

	case OS2I:
	case OI2I:
	case OI2S:

	case OINDEXPTR:
	case OINDEX:
	case ODOT:
	case ODOTPTR:
	case OIND:
		igen(n, &n1, res);
		gmove(&n1, res);
		regfree(&n1);
		break;

	case OLEN:
		if(isptrto(nl->type, TSTRING)) {
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);
			n1.op = OINDREG;
			n1.type = types[TINT32];
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		if(isptrto(nl->type, TMAP)) {
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);
			n1.op = OINDREG;
			n1.type = types[TINT32];
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		if(isptrdarray(nl->type)) {
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);
			n1.op = OINDREG;
			n1.type = types[TUINT32];
			n1.xoffset = offsetof(Array,nel);
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		fatal("cgen: OLEN: unknown type %lT", nl->type);
		break;

	case OCAP:
		if(isptrdarray(nl->type)) {
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);
			n1.op = OINDREG;
			n1.type = types[TUINT32];
			n1.xoffset = offsetof(Array,cap);
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

void
agen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, tmp;
	Prog *p1;
	uint32 w;
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

	case OINDEXPTR:
		w = n->type->width;
		if(nr->addable)
			goto iprad;
		if(nl->addable) {
			regalloc(&n1, nr->type, N);
			cgen(nr, &n1);
			cgen(nl, res);
			goto index;
		}
		cgen(nr, res);
		tempname(&tmp, nr->type);
		gmove(res, &tmp);

	iprad:
		cgen(nl, res);
		regalloc(&n1, nr->type, N);
		cgen(nr, &n1);
		goto index;

	case OS2I:
	case OI2I:
	case OI2S:
		agen_inter(n, res);
		break;

	case OINDEX:
		w = n->type->width;
		if(nr->addable)
			goto irad;
		if(nl->addable) {
			regalloc(&n1, nr->type, N);
			cgen(nr, &n1);
			agen(nl, res);
			goto index;
		}
		cgen(nr, res);
		tempname(&tmp, nr->type);
		gmove(res, &tmp);

	irad:
		agen(nl, res);
		regalloc(&n1, nr->type, N);
		cgen(nr, &n1);
		goto index;

	index:
		// &a is in res
		// i is in &n1
		// w is width

		if(isptrdarray(nl->type)) {
			regalloc(&n2, types[tptr], res);
			gmove(res, &n2);

			if(!debug['B']) {
				// check bounds
				n3 = n2;
				n3.op = OINDREG;
				n3.type = types[tptr];
				n3.xoffset = offsetof(Array, nel);
				gins(optoas(OCMP, types[TUINT32]), &n1, &n3);

				p1 = gbranch(optoas(OLT, types[TUINT32]), T);

				nodconst(&n3, types[TUINT8], 5); // 5 is range trap
				gins(AINT, &n3, N);
				patch(p1, pc);
			}

			// fetch array base from dope
			n3 = n2;
			n3.op = OINDREG;
			n3.type = types[tptr];
			n3.xoffset = offsetof(Array, array);
			gins(AMOVQ, &n3, &n2);
			gmove(&n2, res);
			regfree(&n2);
		} else
			if(!debug['B']) {
				// check bounds
				nodconst(&n3, types[TUINT32], nl->type->bound);
				if(isptrarray(nl->type))
					nodconst(&n3, types[TUINT32], nl->type->type->bound);
				gins(optoas(OCMP, types[TUINT32]), &n1, &n3);

				p1 = gbranch(optoas(OLT, types[TUINT32]), T);

				nodconst(&n3, types[TUINT8], 5); // 5 is range trap
				gins(AINT, &n3, N);
				patch(p1, pc);
			}

		t = types[TUINT64];
		if(issigned[n1.type->etype])
			t = types[TINT64];

		regalloc(&n2, t, &n1);			// i
		gmove(&n1, &n2);
		regfree(&n1);

		nodconst(&n3, t, w);			// w
		gins(optoas(OMUL, t), &n3, &n2);
		gins(optoas(OADD, types[tptr]), &n2, res);
		regfree(&n2);
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

vlong
fieldoffset(Type *t, Node *n)
{
	if(t->etype != TSTRUCT)
		fatal("fieldoffset: not struct %lT", t);
	if(n->op != ONAME)
		fatal("fieldoffset: not field name %N", n);
	return 0;
}

void
igen(Node *n, Node *a, Node *res)
{
	regalloc(a, types[tptr], res);
	agen(n, a);
	a->op = OINDREG;
	a->type = n->type;
}

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
		n = booltrue;

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
		if(nl->ullman < nr->ullman) {
			a = brrev(a);
			r = nl;
			nl = nr;
			nr = r;
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

int32
stkof(Node *n)
{
	switch(n->op) {
	case OS2I:
		return 2*widthptr;
	case OI2I:
		return 1*widthptr;
	case OINDREG:
		return n->xoffset;
	}
	// botch - probably failing to recognize address
	// arithmetic on the above. eg INDEX and DOT
	return -1;
}

void
sgen(Node *n, Node *ns, uint32 w)
{
	Node nodl, nodr;
	int32 c, q, odst, osrc;

	if(debug['g']) {
		dump("\nsgen-res", ns);
		dump("sgen-r", n);
	}
	if(w == 0)
		return;
	if(n->ullman >= UINF && ns->ullman >= UINF) {
		fatal("sgen UINF");
	}

	// offset on the stack
	odst = stkof(ns);
	osrc = stkof(n);
	if(osrc < 0)
		odst = odst;
	if(odst < 0)
		osrc = odst;

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
	if(odst > osrc && odst-osrc < w) {
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
		// for future optimization
		// we leave with the flag clear
		gins(ACLD, N, N);
	} else {
		// normal direction
		gins(ACLD, N, N);		// clear direction flag
		if(q > 0) {
			gconreg(AMOVQ, q, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSQ, N, N);	// MOVQ *(SI)+,*(DI)+
		}

		if(c > 0) {
			gconreg(AMOVQ, c, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSB, N, N);	// MOVB *(SI)+,*(DI)+
		}
	}
}
