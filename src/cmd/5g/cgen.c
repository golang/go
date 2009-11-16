// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

void
mgen(Node *n, Node *n1, Node *rg)
{
	n1->ostk = 0;
	n1->op = OEMPTY;

	if(n->addable) {
		*n1 = *n;
		n1->ostk = 0;
		if(n1->op == OREGISTER || n1->op == OINDREG)
			reg[n->val.u.reg]++;
		return;
	}
	if(n->type->width > widthptr)
		tempname(n1, n->type);
	else
		regalloc(n1, n->type, rg);
	cgen(n, n1);
}

void
mfree(Node *n)
{
	if(n->op == OREGISTER)
		regfree(n);
}

/*
 * generate:
 *	res = n;
 * simplifies and calls gmove.
 */
void
cgen(Node *n, Node *res)
{
	Node *nl, *nr, *r;
	Node n1, n2, n3, f0, f1;
	int a, w;
	Prog *p1, *p2, *p3;
	Addr addr;

	if(debug['g']) {
		dump("\ncgen-n", n);
		dump("cgen-res", res);
	}
	if(n == N || n->type == T)
		goto ret;

	if(res == N || res->type == T)
		fatal("cgen: res nil");

	while(n->op == OCONVNOP)
		n = n->left;

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

	// update addressability for string, slice
	// can't do in walk because n->left->addable
	// changes if n->left is an escaping local variable.
	switch(n->op) {
	case OLEN:
		if(isslice(n->left->type) || istype(n->left->type, TSTRING))
			n->addable = n->left->addable;
		break;
	case OCAP:
		if(isslice(n->left->type))
			n->addable = n->left->addable;
		break;
	}

	// if both are addressable, move
	if(n->addable && res->addable) {
		if (is64(n->type) || is64(res->type) || n->op == OREGISTER || res->op == OREGISTER) {
			gmove(n, res);
		} else {
			regalloc(&n1, n->type, N);
			gmove(n, &n1);
			cgen(&n1, res);
			regfree(&n1);
		}
		goto ret;
	}

	// if both are not addressable, use a temporary.
	if(!n->addable && !res->addable) {
		// could use regalloc here sometimes,
		// but have to check for ullman >= UINF.
		tempname(&n1, n->type);
		cgen(n, &n1);
		cgen(&n1, res);
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

	// if n is sudoaddable generate addr and move
	if (!is64(n->type) && !is64(res->type)) {
		a = optoas(OAS, n->type);
		if(sudoaddable(a, n, &addr, &w)) {
			if (res->op != OREGISTER) {
				regalloc(&n2, res->type, N);
				p1 = gins(a, N, &n2);
				p1->from = addr;
				if(debug['g'])
					print("%P [ignore previous line]\n", p1);
				gmove(&n2, res);
				regfree(&n2);
			} else {
				p1 = gins(a, N, res);
				p1->from = addr;
				if(debug['g'])
					print("%P [ignore previous line]\n", p1);
			}
			sudoclean();
			goto ret;
		}
	}

	// otherwise, the result is addressable but n is not.
	// let's do some computation.

	nl = n->left;
	nr = n->right;

	if(nl != N && nl->ullman >= UINF)
	if(nr != N && nr->ullman >= UINF) {
		tempname(&n1, nl->type);
		cgen(nl, &n1);
		n2 = *n;
		n2.left = &n1;
		cgen(&n2, res);
		goto ret;
	}

	// 64-bit ops are hard on 32-bit machine.
	if(is64(n->type) || is64(res->type) || n->left != N && is64(n->left->type)) {
		switch(n->op) {
		// math goes to cgen64.
		case OMINUS:
		case OCOM:
		case OADD:
		case OSUB:
		case OMUL:
		case OLSH:
		case ORSH:
		case OAND:
		case OOR:
		case OXOR:
			cgen64(n, res);
			return;
		}
	}

	if(nl != N && isfloat[n->type->etype] && isfloat[nl->type->etype])
		goto flt;
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
		p1 = gbranch(AB, T);
		p2 = pc;
		gmove(nodbool(1), res);
		p3 = gbranch(AB, T);
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
		nodconst(&n3, nl->type, 0);
		regalloc(&n2, nl->type, res);
		regalloc(&n1, nl->type, N);
		gmove(&n3, &n2);
		cgen(nl, &n1);
		gins(optoas(OSUB, nl->type), &n1, &n2);
		gmove(&n2, res);
		regfree(&n1);
		regfree(&n2);
		goto ret;

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

	case OLSH:
	case ORSH:
		cgen_shift(n->op, nl, nr, res);
		break;

	case OCONV:
		if(eqtype(n->type, nl->type) || noconv(n->type, nl->type)) {
			cgen(nl, res);
			break;
		}

		mgen(nl, &n1, res);
		gmove(&n1, res);
		mfree(&n1);
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
		if(istype(nl->type, TMAP) || istype(nl->type, TCHAN)) {
			// map has len in the first 32-bit word.
			// a zero pointer means zero length
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);

			nodconst(&n2, types[tptr], 0);
			regalloc(&n3, n2.type, N);
			gmove(&n2, &n3);
			gcmp(optoas(OCMP, types[tptr]), &n1, &n3);
			regfree(&n3);
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
		if(istype(nl->type, TSTRING) || isslice(nl->type)) {
			// both slice and string have len one pointer into the struct.
			igen(nl, &n1, res);
			n1.op = OREGISTER;	// was OINDREG
			regalloc(&n2, types[TUINT32], &n1);
			n1.op = OINDREG;
			n1.type = types[TUINT32];
			n1.xoffset = Array_nel;
			gmove(&n1, &n2);
			gmove(&n2, res);
			regfree(&n1);
			regfree(&n2);
			break;
		}
		fatal("cgen: OLEN: unknown type %lT", nl->type);
		break;

	case OCAP:
		if(istype(nl->type, TCHAN)) {
			// chan has cap in the second 32-bit word.
			// a zero pointer means zero length
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);

			nodconst(&n2, types[tptr], 0);
			regalloc(&n3, n2.type, N);
			gmove(&n2, &n3);
			gcmp(optoas(OCMP, types[tptr]), &n1, &n3);
			regfree(&n3);
			p1 = gbranch(optoas(OEQ, types[tptr]), T);

			n2 = n1;
			n2.op = OINDREG;
			n2.xoffset = 4;
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

	case OCALLFUNC:
		cgen_call(n, 0);
		cgen_callret(n, res);
		break;

	case OMOD:
	case ODIV:
		a = optoas(n->op, nl->type);
		goto abop;
	}
	goto ret;

sbop:	// symmetric binary
	if(nl->ullman < nr->ullman) {
		r = nl;
		nl = nr;
		nr = r;
	}

abop:	// asymmetric binary
	// TODO(kaib): use fewer registers here.
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

flt:	// floating-point.
	regalloc(&f0, nl->type, res);
	if(nr != N)
		goto flt2;

	if(n->op == OMINUS) {
		nr = nodintconst(-1);
		convlit(&nr, n->type);
		n->op = OMUL;
		goto flt2;
	}

	// unary
	cgen(nl, &f0);
	if(n->op != OCONV && n->op != OPLUS)
		gins(optoas(n->op, n->type), &f0, &f0);
	gmove(&f0, res);
	regfree(&f0);
	goto ret;

flt2:	// binary
	if(nl->ullman >= nr->ullman) {
		cgen(nl, &f0);
		regalloc(&f1, n->type, N);
		gmove(&f0, &f1);
		cgen(nr, &f0);
		gins(optoas(n->op, n->type), &f0, &f1);
	} else {
		cgen(nr, &f0);
		regalloc(&f1, n->type, N);
		cgen(nl, &f1);
		gins(optoas(n->op, n->type), &f0, &f1);
	}
	gmove(&f1, res);
	regfree(&f0);
	regfree(&f1);
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
	Node n1, n2, n3, n4, n5, tmp;
	Prog *p1;
	uint32 w;
	uint64 v;
	Type *t;

	if(debug['g']) {
		dump("\nagen-res", res);
		dump("agen-r", n);
	}
	if(n == N || n->type == T || res == N || res->type == T)
		fatal("agen");

	while(n->op == OCONVNOP)
		n = n->left;

	if(n->addable) {
		memset(&n1, 0, sizeof n1);
		n1.op = OADDR;
		n1.left = n;
		regalloc(&n2, types[tptr], res);
		gins(AMOVW, &n1, &n2);
		gmove(&n2, res);
		regfree(&n2);
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

	case OCALLFUNC:
		cgen_call(n, 0);
		cgen_aret(n, res);
		break;

	case OINDEX:
		// TODO(rsc): uint64 indices
		w = n->type->width;
		if(nr->addable) {
			agenr(nl, &n3, res);
			if(!isconst(nr, CTINT)) {
				tempname(&tmp, types[TINT32]);
				cgen(nr, &tmp);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
		} else if(nl->addable) {
			if(!isconst(nr, CTINT)) {
				tempname(&tmp, types[TINT32]);
				cgen(nr, &tmp);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
			regalloc(&n3, types[tptr], res);
			agen(nl, &n3);
		} else {
			tempname(&tmp, types[TINT32]);
			cgen(nr, &tmp);
			nr = &tmp;
			agenr(nl, &n3, res);
			regalloc(&n1, tmp.type, N);
			gins(optoas(OAS, tmp.type), &tmp, &n1);
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
					regalloc(&n4, n1.type, N);
					cgen(&n1, &n4);
					nodconst(&n2, types[TUINT32], v);
					regalloc(&n5, n2.type, N);
					gmove(&n2, &n5);
					gcmp(optoas(OCMP, types[TUINT32]), &n4, &n5);
					regfree(&n4);
					regfree(&n5);
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
			regalloc(&n4, n2.type, N);
			gmove(&n2, &n4);
			gins(optoas(OADD, types[tptr]), &n4, &n3);
			regfree(&n4);

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
			regalloc(&n4, types[TUINT32], N);
			if(isslice(nl->type)) {
				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_nel;
				cgen(&n1, &n4);
			} else {
				nodconst(&n1, types[TUINT32], nl->type->bound);
				gmove(&n1, &n4);
			}
			gcmp(optoas(OCMP, types[TUINT32]), &n2, &n4);
			regfree(&n4);
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
			memset(&n4, 0, sizeof n4);
			n4.op = OADDR;
			n4.left = &n2;
			cgen(&n4, &n3);
			if (w == 1)
				gins(AADD, &n2, &n3);
			else if(w == 2)
				gshift(AADD, &n2, SHIFT_LL, 1, &n3);
			else if(w == 4)
				gshift(AADD, &n2, SHIFT_LL, 2, &n3);
			else if(w == 8)
				gshift(AADD, &n2, SHIFT_LL, 3, &n3);	
		} else {
			regalloc(&n4, t, N);
			nodconst(&n1, t, w);
			gmove(&n1, &n4);
			gins(optoas(OMUL, t), &n4, &n2);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
			regfree(&n4);
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
			nodconst(&n1, types[TINT32], n->xoffset);
			regalloc(&n2, n1.type, N);
			regalloc(&n3, types[TINT32], N);
			gmove(&n1, &n2);
			gmove(res, &n3);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
			gmove(&n3, res);
			regfree(&n2);
			regfree(&n3);
		}
		break;

	case OIND:
		cgen(nl, res);
		break;

	case ODOT:
		agen(nl, res);
		if(n->xoffset != 0) {
			nodconst(&n1, types[TINT32], n->xoffset);
			regalloc(&n2, n1.type, N);
			regalloc(&n3, types[TINT32], N);
			gmove(&n1, &n2);
			gmove(res, &n3);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
			gmove(&n3, res);
			regfree(&n2);
			regfree(&n3);
		}
		break;

	case ODOTPTR:
		cgen(nl, res);
		if(n->xoffset != 0) {
			// explicit check for nil if struct is large enough
			// that we might derive too big a pointer.
			if(nl->type->type->width >= unmappedzero) {
				regalloc(&n1, types[tptr], N);
				gmove(res, &n1);
				p1 = gins(AMOVW, &n1, &n1);
				p1->from.type = D_OREG;
				p1->from.offset = 0;
				regfree(&n1);
			}
			nodconst(&n1, types[TINT32], n->xoffset);
			regalloc(&n2, n1.type, N);
			regalloc(&n3, types[tptr], N);
			gmove(&n1, &n2);
			gmove(res, &n3);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
			gmove(&n3, res);
			regfree(&n2);
			regfree(&n3);
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
 *	newreg = &n;
 *
 * caller must regfree(a).
 */
void
agenr(Node *n, Node *a, Node *res)
{
	Node n1;

	tempname(&n1, types[tptr]);
	agen(n, &n1);
	regalloc(a, types[tptr], res);
	gmove(&n1, a);
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
	Node n1, n2, n3, n4, tmp;
	Prog *p1, *p2;

	if(debug['g']) {
		dump("\nbgen", n);
	}

	if(n == N)
		n = nodbool(1);

	nl = n->left;
	nr = n->right;

	if(n->type == T) {
		convlit(&n, types[TBOOL]);
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
	def:
		regalloc(&n1, n->type, N);
		cgen(n, &n1);
		nodconst(&n2, n->type, 0);
		regalloc(&n3, n->type, N);
		gmove(&n2, &n3);
		gcmp(optoas(OCMP, n->type), &n1, &n3);
		a = ABNE;
		if(!true)
			a = ABEQ;
		patch(gbranch(a, n->type), to);
		regfree(&n1);
		regfree(&n3);
		goto ret;

	case OLITERAL:
		// need to ask if it is bool?
		if(!true == !n->val.u.bval)
			patch(gbranch(AB, T), to);
		goto ret;

	case ONAME:
		if(n->addable == 0)
			goto def;
		nodconst(&n1, n->type, 0);
		regalloc(&n2, n->type, N);
		regalloc(&n3, n->type, N);
		gmove(&n1, &n2);
		cgen(n, &n3);
		gcmp(optoas(OCMP, n->type), &n2, &n3);
		a = ABNE;
		if(!true)
			a = ABEQ;
		patch(gbranch(a, n->type), to);
		regfree(&n2);
		regfree(&n3);
		goto ret;

	case OANDAND:
		if(!true)
			goto caseor;

	caseand:
		p1 = gbranch(AB, T);
		p2 = gbranch(AB, T);
		patch(p1, pc);
		bgen(n->left, !true, p2);
		bgen(n->right, !true, p2);
		p1 = gbranch(AB, T);
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
		if(!true) {
			if(isfloat[nl->type->etype]) {
				// brcom is not valid on floats when NaN is involved.
				p1 = gbranch(AJMP, T);
				p2 = gbranch(AJMP, T);
				patch(p1, pc);
				bgen(n, 1, p2);
				patch(gbranch(AJMP, T), to);
				patch(p2, pc);
				goto ret;
			}				
			a = brcom(a);
		}

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
			regalloc(&n3, types[tptr], N);
			regalloc(&n4, types[tptr], N);
			agen(nl, &n1);
			n2 = n1;
			n2.op = OINDREG;
			n2.xoffset = Array_array;
			gmove(&n2, &n4);
			nodconst(&tmp, types[tptr], 0);
			gmove(&tmp, &n3);
			gcmp(optoas(OCMP, types[tptr]), &n4, &n3);
			patch(gbranch(a, types[tptr]), to);
			regfree(&n4);
			regfree(&n3);
			regfree(&n1);
			break;
		}

		if(isinter(nl->type)) {
			// front end shold only leave cmp to literal nil
			if((a != OEQ && a != ONE) || nr->op != OLITERAL) {
				yyerror("illegal interface comparison");
				break;
			}
			a = optoas(a, types[tptr]);
			regalloc(&n1, types[tptr], N);
			regalloc(&n3, types[tptr], N);
			regalloc(&n4, types[tptr], N);
			agen(nl, &n1);
			n2 = n1;
			n2.op = OINDREG;
			n2.xoffset = 0;
			gmove(&n2, &n4);
			nodconst(&tmp, types[tptr], 0);
			gmove(&tmp, &n3);
			gcmp(optoas(OCMP, types[tptr]), &n4, &n3);
			patch(gbranch(a, types[tptr]), to);
			regfree(&n1);
			regfree(&n3);
			regfree(&n4);
			break;
		}

		if(is64(nr->type)) {
			if(!nl->addable) {
				tempname(&n1, nl->type);
				cgen(nl, &n1);
				nl = &n1;
			}
			if(!nr->addable) {
				tempname(&n2, nr->type);
				cgen(nr, &n2);
				nr = &n2;
			}
			cmp64(nl, nr, a, to);
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

			regalloc(&n2, nr->type, N);
			cgen(&tmp, &n2);

			gcmp(optoas(OCMP, nr->type), &n1, &n2);
			patch(gbranch(a, nr->type), to);

			regfree(&n1);
			regfree(&n2);
			break;
		}

		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);

		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);

		gcmp(optoas(OCMP, nr->type), &n1, &n2);
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
	case OCALLFUNC:
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
 *	memmove(&res, &n, w);
 * NB: character copy assumed little endian architecture
 */
void
sgen(Node *n, Node *res, int32 w)
{
	Node dst, src, tmp, nend;
	int32 c, q, odst, osrc;
	Prog *p, *ploop;

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

	if(osrc % 4 != 0 || odst %4 != 0)
		fatal("sgen: non word(4) aligned offset src %d or dst %d", osrc, odst);

	regalloc(&dst, types[tptr], res);

	if(n->ullman >= res->ullman) {
		agen(n, &dst);	// temporarily use dst
		regalloc(&src, types[tptr], N);
		gins(AMOVW, &dst, &src);
		agen(res, &dst);
	} else {
		agen(res, &dst);
		regalloc(&src, types[tptr], N);
		agen(n, &src);
	}

	regalloc(&tmp, types[TUINT32], N);

	c = w % 4;	// bytes
	q = w / 4;	// quads

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if(osrc < odst && odst < osrc+w) {
		if(c != 0)
			fatal("sgen: reverse character copy not implemented");
		if(q >= 4) {
			regalloc(&nend, types[TUINT32], N);
			// set up end marker to 4 bytes before source
			p = gins(AMOVW, &src, &nend);
			p->from.type = D_CONST;
			p->from.offset = -4;

			// move src and dest to the end of block
			p = gins(AMOVW, &src, &src);
			p->from.type = D_CONST;
			p->from.offset = (q-1)*4;

			p = gins(AMOVW, &dst, &dst);
			p->from.type = D_CONST;
			p->from.offset = (q-1)*4;

			p = gins(AMOVW, &src, &tmp);
			p->from.type = D_OREG;
			p->from.offset = -4;
			p->scond |= C_PBIT;
			ploop = p;

			p = gins(AMOVW, &tmp, &dst);
			p->to.type = D_OREG;
			p->to.offset = -4;
			p->scond |= C_PBIT;

			p = gins(ACMP, &src, N);
			raddr(&nend, p);

			patch(gbranch(ABNE, T), ploop);

 			regfree(&nend);
		} else {
			// move src and dest to the end of block
			p = gins(AMOVW, &src, &src);
			p->from.type = D_CONST;
			p->from.offset = (q-1)*4;

			p = gins(AMOVW, &dst, &dst);
			p->from.type = D_CONST;
			p->from.offset = (q-1)*4;

			while(q > 0) {
				p = gins(AMOVW, &src, &tmp);
				p->from.type = D_OREG;
				p->from.offset = -4;
 				p->scond |= C_PBIT;

				p = gins(AMOVW, &tmp, &dst);
				p->to.type = D_OREG;
				p->to.offset = -4;
 				p->scond |= C_PBIT;

				q--;
			}
		}
	} else {
		// normal direction
		if(q >= 4) {
			regalloc(&nend, types[TUINT32], N);
			p = gins(AMOVW, &src, &nend);
			p->from.type = D_CONST;
			p->from.offset = q*4;

			p = gins(AMOVW, &src, &tmp);
			p->from.type = D_OREG;
			p->from.offset = 4;
			p->scond |= C_PBIT;
			ploop = p;

			p = gins(AMOVW, &tmp, &dst);
			p->to.type = D_OREG;
			p->to.offset = 4;
			p->scond |= C_PBIT;

			p = gins(ACMP, &src, N);
			raddr(&nend, p);

			patch(gbranch(ABNE, T), ploop);

 			regfree(&nend);
		} else
		while(q > 0) {
			p = gins(AMOVW, &src, &tmp);
			p->from.type = D_OREG;
			p->from.offset = 4;
 			p->scond |= C_PBIT;

			p = gins(AMOVW, &tmp, &dst);
			p->to.type = D_OREG;
			p->to.offset = 4;
 			p->scond |= C_PBIT;

			q--;
		}

		if (c != 0) {
			//	MOVW	(src), tmp
			p = gins(AMOVW, &src, &tmp);
			p->from.type = D_OREG;

			//	MOVW	tmp<<((4-c)*8),src
			gshift(AMOVW, &tmp, SHIFT_LL, ((4-c)*8), &src);

			//	MOVW	src>>((4-c)*8),src
			gshift(AMOVW, &src, SHIFT_LR, ((4-c)*8), &src);

			//	MOVW	(dst), tmp
			p = gins(AMOVW, &dst, &tmp);
			p->from.type = D_OREG;

			//	MOVW	tmp>>(c*8),tmp
			gshift(AMOVW, &tmp, SHIFT_LR, (c*8), &tmp);

			//	MOVW	tmp<<(c*8),tmp
			gshift(AMOVW, &tmp, SHIFT_LL, c*8, &tmp);

			//	ORR		src, tmp
			gins(AORR, &src, &tmp);

			//	MOVW	tmp, (dst)
			p = gins(AMOVW, &tmp, &dst);
			p->to.type = D_OREG;
		}
	}
 	regfree(&dst);
	regfree(&src);
	regfree(&tmp);
}
