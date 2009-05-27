// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc):
//
//	better management of 64-bit values,
//	especially constants.  generated code is pretty awful.
//
//	assume CLD?

#include "gg.h"

static int cancgen64(Node *n, Node *res);

int
is64(Type *t)
{
	if(t == T)
		return 0;
	switch(simtype[t->etype]) {
	case TINT64:
	case TUINT64:
	case TPTR64:
		return 1;
	}
	return 0;
}

/*
 * generate:
 *	res = n;
 * simplifies and calls gmove.
 *
 * TODO:
 *	sudoaddable
 */
void
cgen(Node *n, Node *res)
{
	Node *nl, *nr, *r, n1, n2, rr;
	Prog *p1, *p2, *p3;
	int a;

	if(debug['g']) {
		dump("\ncgen-n", n);
		dump("cgen-res", res);
	}

	if(n == N || n->type == T)
		fatal("cgen: n nil");
	if(res == N || res->type == T)
		fatal("cgen: res nil");

	// function calls on both sides?  introduce temporary
	if(n->ullman >= UINF && res->ullman >= UINF) {
		tempname(&n1, n->type);
		cgen(n, &n1);
		cgen(&n1, res);
		return;
	}

	// structs etc get handled specially
	if(isfat(n->type)) {
		sgen(n, res, n->type->width);
		return;
	}
	
	// if both are addressable, move
	if(n->addable && res->addable) {
		gmove(n, res);
		return;
	}
	
	// if both are not addressable, use a temporary.
	if(!n->addable && !res->addable) {
		tempalloc(&n1, n->type);
		cgen(n, &n1);
		cgen(&n1, res);
		tempfree(&n1);
		return;
	}

	// if result is not addressable directly but n is,
	// compute its address and then store via the address.
	if(!res->addable) {
		igen(res, &n1, N);
		cgen(n, &n1);
		regfree(&n1);
		return;
	}

	// otherwise, the result is addressable but n is not.
	// let's do some computation.

	// 64-bit ops are hard on 32-bit machine.
	if(is64(n->type) && cancgen64(n, res))
		return;
	
	// use ullman to pick operand to eval first.
	nl = n->left;
	nr = n->right;
	if(nl != N && nl->ullman >= UINF)
	if(nr != N && nr->ullman >= UINF) {
		// both are hard
		tempalloc(&n1, nr->type);
		cgen(nr, &n1);
		n2 = *n;
		n2.right = &n1;
		cgen(&n2, res);
		tempfree(&n1);
		return;
	}

	switch(n->op) {
	default:
		dump("cgen", n);
		fatal("cgen %O", n->op);
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
		return;

	case OPLUS:
		cgen(nl, res);
		return;

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
		// TODO: cgen_bmul ?
		goto sbop;

	// asymmetric binary
	case OSUB:
		a = optoas(n->op, nl->type);
		goto abop;

	case OCONV:
		if(eqtype(n->type, nl->type)) {
			cgen(nl, res);
			break;
		}
		tempalloc(&n1, nl->type);
		cgen(nl, &n1);
		gmove(&n1, res);
		tempfree(&n1);
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
			tempalloc(&n1, types[tptr]);
			cgen(nl, &n1);
			regalloc(&n2, types[tptr], N);
			gmove(&n1, &n2);
			tempfree(&n1);
			n1 = n2;

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
			igen(nl, &n1, res);
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
			igen(nl, &n1, res);
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
	return;

sbop:	// symmetric binary
	if(nl->ullman < nr->ullman) {
		r = nl;
		nl = nr;
		nr = r;
	}

abop:	// asymmetric binary
	if(nl->ullman >= nr->ullman) {
		tempalloc(&n1, nl->type);
		cgen(nl, &n1);
		tempalloc(&n2, nr->type);
		cgen(nr, &n2);
	} else {
		tempalloc(&n1, nl->type);
		tempalloc(&n2, nr->type);
		cgen(nr, &n2);
		cgen(nl, &n1);
	}
	regalloc(&rr, res->type, N);
	gmove(&n1, &rr);
	gins(a, &n2, &rr);
	gmove(&rr, res);
	regfree(&rr);
	tempfree(&n2);
	tempfree(&n1);
	return;

uop:	// unary
	tempalloc(&n1, nl->type);
	cgen(nl, &n1);
	gins(a, N, &n1);
	gmove(&n1, res);
	tempfree(&n1);
	return;
}

/*
 * address gen
 *	res = &n;
 */
void
agen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, tmp;
	Type *t;
	uint32 w;
	uint64 v;
	Prog *p1;

	if(debug['g']) {
		dump("\nagen-res", res);
		dump("agen-r", n);
	}
	if(n == N || n->type == T || res == N || res->type == T)
		fatal("agen");

	// addressable var is easy
	if(n->addable) {
		regalloc(&n1, types[tptr], res);
		gins(ALEAL, n, &n1);
		gmove(&n1, res);
		regfree(&n1);
		return;
	}
	
	// let's compute
	nl = n->left;
	nr = n->right;
	
	switch(n->op) {
	default:
		fatal("agen %O", n->op);
	
	case OCONV:
		if(!eqtype(n->type, nl->type))
			fatal("agen: non-trivial OCONV");
		agen(nl, res);
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

	case OINDEX:
		w = n->type->width;
		if(nr->addable) {
			agenr(nl, &n3, res);
			if(!isconst(nr, CTINT)) {
				regalloc(&n1, nr->type, N);
				cgen(nr, &n1);
			}
		} else if(nl->addable) {
			if(!isconst(nr, CTINT)) {
				tempalloc(&tmp, nr->type);
				cgen(nr, &tmp);
				regalloc(&n1, nr->type, N);
				gmove(&tmp, &n1);
				tempfree(&tmp);
			}
			regalloc(&n3, types[tptr], res);
			agen(nl, &n3);
		} else {
			tempalloc(&tmp, nr->type);
			cgen(nr, &tmp);
			nr = &tmp;
			agenr(nl, &n3, res);
			regalloc(&n1, nr->type, N);
			gins(optoas(OAS, nr->type), &tmp, &n1);
			tempfree(&tmp);
		}

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
					nodconst(&n2, types[TUINT32], v);
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
		t = types[TUINT32];
		if(issigned[n1.type->etype])
			t = types[TINT32];

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
				nodconst(&n1, types[TUINT32], nl->type->bound);
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
			p1 = gins(ALEAL, &n2, &n3);
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
			nodconst(&n1, types[tptr], n->xoffset);
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
			nodconst(&n1, types[tptr], n->xoffset);
			gins(optoas(OADD, types[tptr]), &n1, res);
		}
		break;

	case ODOTPTR:
		t = nl->type;
		if(!isptr[t->etype])
			fatal("agen: not ptr %N", n);
		cgen(nl, res);
		if(n->xoffset != 0) {
			nodconst(&n1, types[tptr], n->xoffset);
			gins(optoas(OADD, types[tptr]), &n1, res);
		}
		break;
	}
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
	Node n1;

	tempalloc(&n1, types[tptr]);
	agen(n, &n1);
	regalloc(a, types[tptr], res);
	gins(optoas(OAS, types[tptr]), &n1, a);
	tempfree(&n1);
	a->op = OINDREG;
	a->type = n->type;
}

/*
 * generate:
 *	newreg = &n;
 *
 * caller must regfree(a).
 */
void
agenr(Node *n, Node *a, Node *res)
{
	Node n1;

	tempalloc(&n1, types[tptr]);
	agen(n, &n1);
	regalloc(a, types[tptr], res);
	gmove(&n1, a);
	tempfree(&n1);
}

/*
 * branch gen
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
			return;
	}

	et = n->type->etype;
	if(et != TBOOL) {
		yyerror("cgen: bad type %T for %O", n->type, n->op);
		patch(gins(AEND, N, N), to);
		return;
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
		return;

	case OLITERAL:
// need to ask if it is bool?
		if(!true == !n->val.u.bval)
			patch(gbranch(AJMP, T), to);
		return;

	case ONAME:
		nodconst(&n1, n->type, 0);
		gins(optoas(OCMP, n->type), n, &n1);
		a = AJNE;
		if(!true)
			a = AJEQ;
		patch(gbranch(a, n->type), to);
		return;

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
		return;

	case OOROR:
		if(!true)
			goto caseand;

	caseor:
		bgen(n->left, true, to);
		bgen(n->right, true, to);
		return;

	case OEQ:
	case ONE:
	case OLT:
	case OGT:
	case OLE:
	case OGE:
		nr = n->right;
		if(nr == N || nr->type == T)
			return;

	case ONOT:	// unary
		nl = n->left;
		if(nl == N || nl->type == T)
			return;
	}

	switch(n->op) {
	case ONOT:
		bgen(nl, !true, to);
		break;

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
		
		if(is64(nr->type)) {
			if(!nl->addable) {
				tempalloc(&n1, nl->type);
				cgen(nl, &n1);
				nl = &n1;
			}
			if(!nr->addable) {
				tempalloc(&n2, nr->type);
				cgen(nr, &n2);
				nr = &n2;
			}
			cmp64(nl, nr, a, to);
			if(nr == &n2)
				tempfree(&n2);
			if(nl == &n1)
				tempfree(&n1);
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
 * struct gen
 *	memmove(&res, &n, w);
 */
void
sgen(Node *n, Node *res, int w)
{
	Node nodl, nodr;
	int32 c, q, odst, osrc;

	if(debug['g']) {
		print("\nsgen w=%d\n", w);
		dump("r", n);
		dump("res", res);
	}
	if(w == 0)
		return;
	if(n->ullman >= UINF && res->ullman >= UINF) {
		fatal("sgen UINF");
	}

	if(w < 0)
		fatal("sgen copy %d", w);

	// offset on the stack
	osrc = stkof(n);
	odst = stkof(res);

	// TODO(rsc): Should these be tempalloc instead?
	nodreg(&nodl, types[tptr], D_DI);
	nodreg(&nodr, types[tptr], D_SI);

	if(n->ullman >= res->ullman) {
		agen(n, &nodr);
		agen(res, &nodl);
	} else {
		agen(res, &nodl);
		agen(n, &nodr);
	}

	c = w % 4;	// bytes
	q = w / 4;	// doublewords

	gins(ACLD, N, N);
	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if(osrc < odst && odst < osrc+w) {
		// reverse direction
		gins(ASTD, N, N);		// set direction flag
		if(c > 0) {
			gconreg(AADDL, w-1, D_SI);
			gconreg(AADDL, w-1, D_DI);

			gconreg(AMOVL, c, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSB, N, N);	// MOVB *(SI)-,*(DI)-
		}

		if(q > 0) {
			if(c > 0) {
				gconreg(AADDL, -7, D_SI);
				gconreg(AADDL, -7, D_DI);
			} else {
				gconreg(AADDL, w-8, D_SI);
				gconreg(AADDL, w-8, D_DI);
			}
			gconreg(AMOVL, q, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSL, N, N);	// MOVL *(SI)-,*(DI)-
		}
		// we leave with the flag clear
		gins(ACLD, N, N);
	} else {
		// normal direction
		if(q >= 4) {
			gconreg(AMOVL, q, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSL, N, N);	// MOVL *(SI)+,*(DI)+
		} else
		while(q > 0) {
			gins(AMOVSL, N, N);	// MOVL *(SI)+,*(DI)+
			q--;
		}
		while(c > 0) {
			gins(AMOVSB, N, N);	// MOVB *(SI)+,*(DI)+
			c--;
		}
	}
}

/*
 * attempt to generate 64-bit
 *	res = n
 * return 1 on success, 0 if op not handled.
 */
static int
cancgen64(Node *n, Node *res)
{
	Node adr1, adr2, t1, t2, r1, r2, r3, r4, r5, nod, *l, *r;
	Prog *p1, *p2;

	if(n->op == OCALL)
		return 0;
	if(res->op != OINDREG && res->op != ONAME) {
		dump("n", n);
		dump("res", res);
		fatal("cgen64 %O of %O", n->op, res->op);
	}
	switch(n->op) {
	default:
		return 0;

	case ONAME:
	case ODOT:
		gmove(n, res);
		return 1;

	case OMINUS:
		cgen(n->left, res);
		gins(ANEGL, N, res);
		res->xoffset += 4;
		regalloc(&nod, types[TINT32], N);
		gins(AXORL, &nod, &nod);
		gins(ASBBL, res, &nod);
		gins(AMOVL, &nod, res);
		regfree(&nod);
		return 1;

	case OADD:
	case OSUB:
	case OMUL:
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
	tempalloc(&adr1, types[TPTR32]);
	agen(l, &adr1);		
	tempalloc(&adr2, types[TPTR32]);
	agen(r, &adr2);

	nodreg(&r1, types[TPTR32], D_AX);
	nodreg(&r2, types[TPTR32], D_DX);
	nodreg(&r3, types[TPTR32], D_CX);

	switch(n->op) {
	case OADD:
	case OSUB:
		gmove(&adr1, &r3);
		r3.op = OINDREG;
		r3.xoffset = 0;
		gins(AMOVL, &r3, &r1);
		r3.xoffset = 4;
		gins(AMOVL, &r3, &r2);
		
		r3.xoffset = 0;
		r3.op = OREGISTER;
		gmove(&adr2, &r3);
		r3.op = OINDREG;
		if(n->op == OADD)
			gins(AADDL, &r3, &r1);
		else
			gins(ASUBL, &r3, &r1);
		r3.xoffset = 4;
		if(n->op == OADD)
			gins(AADCL, &r3, &r2);
		else
			gins(ASBBL, &r3, &r2);
		break;

	case OMUL:	
		regalloc(&r4, types[TPTR32], N);
		regalloc(&r5, types[TPTR32], N);
		
		// load args into r2:r1 and r4:r3.
		// leave result in r2:r1 (DX:AX)
		gmove(&adr1, &r5);
		r5.op = OINDREG;
		r5.xoffset = 0;
		gmove(&r5, &r1);
		r5.xoffset = 4;
		gmove(&r5, &r2);
		r5.xoffset = 0;
		r5.op = OREGISTER;
		gmove(&adr2, &r5);
		r5.op = OINDREG;
		gmove(&r5, &r3);
		r5.xoffset = 4;
		gmove(&r5, &r4);
		r5.xoffset = 0;
		r5.op = OREGISTER;

		// if r2|r4 == 0, use one 32 x 32 -> 64 unsigned multiply
		gmove(&r2, &r5);
		gins(AORL, &r4, &r5);
		p1 = gbranch(AJNE, T);
		gins(AMULL, &r3, N);	// AX (=r1) is implied
		p2 = gbranch(AJMP, T);
		patch(p1, pc);
	
		// full 64x64 -> 64, from 32 x 32 -> 64.
		gins(AIMULL, &r3, &r2);
		gins(AMOVL, &r1, &r5);
		gins(AIMULL, &r4, &r5);
		gins(AADDL, &r2, &r5);
		gins(AMOVL, &r3, &r2);
		gins(AMULL, &r2, N);	// AX (=r1) is implied
		gins(AADDL, &r5, &r2);
		patch(p2, pc);
		regfree(&r4);
		regfree(&r5);
		break;
	
	}
	
	tempfree(&adr2);
	tempfree(&adr1);

	// Store result.
	gins(AMOVL, &r1, res);
	res->xoffset += 4;
	gins(AMOVL, &r2, res);
	res->xoffset -= 4;
	
	if(r == &t2)
		tempfree(&t2);
	if(l == &t1)
		tempfree(&t1);
	return 1;
}

/*
 * generate comparison of nl, nr, both 64-bit.
 * nl is memory; nr is constant or memory.
 */
void
cmp64(Node *nl, Node *nr, int op, Prog *to)
{
	int64 x;
	Node adr1, adr2, rr;
	Prog *br, *p;
	Type *t;
	
	t = nr->type;
	
	memset(&adr1, 0, sizeof adr1);
	memset(&adr2, 0, sizeof adr2);

	regalloc(&adr1, types[TPTR32], N);
	agen(nl, &adr1);
	adr1.op = OINDREG;
	nl = &adr1;
	
	x = 0;
	if(nr->op == OLITERAL) {
		if(!isconst(nr, CTINT))
			fatal("bad const in cmp64");
		x = mpgetfix(nr->val.u.xval);
	} else {
		regalloc(&adr2, types[TPTR32], N);
		agen(nr, &adr2);
		adr2.op = OINDREG;
		nr = &adr2;
	}
	
	// compare most significant word
	nl->xoffset += 4;
	if(nr->op == OLITERAL) {
		p = gins(ACMPL, nl, nodintconst((uint32)(x>>32)));
	} else {
		regalloc(&rr, types[TUINT32], N);
		nr->xoffset += 4;
		gins(AMOVL, nr, &rr);
		gins(ACMPL, nl, &rr);
		nr->xoffset -= 4;
		regfree(&rr);
	}
	nl->xoffset -= 4;

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
		br = gbranch(AJNE, T);
		break;
	case ONE:
		// cmp hi
		// jne to
		// cmp lo
		// jne to
		patch(gbranch(AJNE, T), to);
		break;
	case OGE:
	case OGT:
		// cmp hi
		// jgt to
		// jlt L
		// cmp lo
		// jge to (or jgt to)
		// L:
		patch(gbranch(optoas(OGT, t), T), to);
		br = gbranch(optoas(OLT, t), T);
		break;
	case OLE:
	case OLT:
		// cmp hi
		// jlt to
		// jgt L
		// cmp lo
		// jle to (or jlt to)
		// L:
		patch(gbranch(optoas(OLT, t), T), to);
		br = gbranch(optoas(OGT, t), T);
		break;	
	}

	// compare least significant word
	if(nr->op == OLITERAL) {
		p = gins(ACMPL, nl, nodintconst((uint32)x));
	} else {
		regalloc(&rr, types[TUINT32], N);
		gins(AMOVL, nr, &rr);
		gins(ACMPL, nl, &rr);
		regfree(&rr);
	}

	// jump again
	switch(op) {
	default:
		fatal("cmp64 %O %T", op, nr->type);
	case OEQ:
	case ONE:
	case OGE:
	case OGT:
	case OLE:
	case OLT:
		patch(gbranch(optoas(op, t), T), to);
		break;	
	}

	// point first branch down here if appropriate
	if(br != P)
		patch(br, pc);

	regfree(&adr1);
	if(nr == &adr2)
		regfree(&adr2);	
}

