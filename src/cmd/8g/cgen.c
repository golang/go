// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc):
//	assume CLD?

#include "gg.h"

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
	Node *nl, *nr, *r, n1, n2, rr, f0, f1;
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

	// static initializations
	if(initflag && gen_as_init(n, res))
		return;

	// function calls on both sides?  introduce temporary
	if(n->ullman >= UINF && res->ullman >= UINF) {
		tempalloc(&n1, n->type);
		cgen(n, &n1);
		cgen(&n1, res);
		tempfree(&n1);
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

	if(isfloat[n->type->etype] && isfloat[nl->type->etype])
		goto flt;

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
	case OCOM:
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
		if(eqtype(n->type, nl->type) || noconv(n->type, nl->type)) {
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
		if(istype(nl->type, TMAP)) {
			// map has len in the first 32-bit word.
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
		if(istype(nl->type, TSTRING) || isslice(nl->type)) {
			// both slice and string have len one pointer into the struct.
			// a zero pointer means zero length
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

flt:	// floating-point.  387 (not SSE2) to interoperate with 6c
	nodreg(&f0, nl->type, D_F0);
	nodreg(&f1, n->type, D_F0+1);
	if(nr != N)
		goto flt2;

	// unary
	cgen(nl, &f0);
	if(n->op != OCONV)
		gins(foptoas(n->op, n->type, 0), &f0, &f0);
	gmove(&f0, res);
	return;

flt2:	// binary
	if(nl->ullman >= nr->ullman) {
		cgen(nl, &f0);
		if(nr->addable)
			gins(foptoas(n->op, n->type, 0), nr, &f0);
		else {
			cgen(nr, &f0);
			gins(foptoas(n->op, n->type, Fpop), &f0, &f1);
		}
	} else {
		cgen(nr, &f0);
		if(nl->addable)
			gins(foptoas(n->op, n->type, Frev), nl, &f0);
		else {
			cgen(nl, &f0);
			gins(foptoas(n->op, n->type, Frev|Fpop), &f0, &f1);
		}
	}
	gmove(&f0, res);
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
		if(!cvttype(n->type, nl->type))
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
				tempalloc(&tmp, nr->type);
				cgen(nr, &tmp);
				regalloc(&n1, nr->type, N);
				gmove(&tmp, &n1);
				tempfree(&tmp);
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
	Node n1, n2, tmp, t1, t2, ax;
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

		if(isfloat[nr->type->etype]) {
			nodreg(&tmp, nr->type, D_F0);
			nodreg(&n2, nr->type, D_F0 + 1);
			nodreg(&ax, types[TUINT16], D_AX);
			et = simsimtype(nr->type);
			if(et == TFLOAT64) {
				// easy - do in FPU
				cgen(nr, &tmp);
				cgen(nl, &tmp);
				gins(AFUCOMPP, &tmp, &n2);
			} else {
				// NOTE(rsc): This is wrong.
				// It's right for comparison but presumably all the
				// other ops have the same problem.  We need to
				// figure out what the right solution is, besides
				// tell people to use float64.
				tempalloc(&t1, types[TFLOAT32]);
				tempalloc(&t2, types[TFLOAT32]);
				cgen(nr, &t1);
				cgen(nl, &t2);
				gmove(&t1, &tmp);
				gins(AFCOMFP, &t1, &tmp);
				tempfree(&t2);
				tempfree(&t1);
			}
			gins(AFSTSW, N, &ax);
			gins(ASAHF, N, N);
			patch(gbranch(optoas(brrev(a), nr->type), T), to);
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
			tempalloc(&tmp, nr->type);
			cgen(nr, &tmp);

			tempalloc(&n1, nl->type);
			cgen(nl, &n1);

			regalloc(&n2, nr->type, N);
			cgen(&tmp, &n2);

			gins(optoas(OCMP, nr->type), &n1, &n2);
			patch(gbranch(a, nr->type), to);
			tempfree(&n1);
			tempfree(&tmp);
			regfree(&n2);
			break;
		}

		tempalloc(&n1, nl->type);
		cgen(nl, &n1);

		if(smallintconst(nr)) {
			gins(optoas(OCMP, nr->type), &n1, nr);
			patch(gbranch(a, nr->type), to);
			tempfree(&n1);
			break;
		}

		tempalloc(&tmp, nr->type);
		cgen(nr, &tmp);
		regalloc(&n2, nr->type, N);
		gmove(&tmp, &n2);
		tempfree(&tmp);

		gins(optoas(OCMP, nr->type), &n1, &n2);
		patch(gbranch(a, nr->type), to);
		regfree(&n2);
		tempfree(&n1);
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
	Node dst, src, tdst, tsrc;
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

	nodreg(&dst, types[tptr], D_DI);
	nodreg(&src, types[tptr], D_SI);

	tempalloc(&tsrc, types[tptr]);
	tempalloc(&tdst, types[tptr]);
	if(!n->addable)
		agen(n, &tsrc);
	if(!res->addable)
		agen(res, &tdst);
	if(n->addable)
		agen(n, &src);
	else
		gmove(&tsrc, &src);
	if(res->addable)
		agen(res, &dst);
	else
		gmove(&tdst, &dst);
	tempfree(&tdst);
	tempfree(&tsrc);

	c = w % 4;	// bytes
	q = w / 4;	// doublewords

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
				gconreg(AADDL, -3, D_SI);
				gconreg(AADDL, -3, D_DI);
			} else {
				gconreg(AADDL, w-4, D_SI);
				gconreg(AADDL, w-4, D_DI);
			}
			gconreg(AMOVL, q, D_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSL, N, N);	// MOVL *(SI)-,*(DI)-
		}
		// we leave with the flag clear
		gins(ACLD, N, N);
	} else {
		gins(ACLD, N, N);	// paranoia.  TODO(rsc): remove?
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
void
cgen64(Node *n, Node *res)
{
	Node t1, t2, ax, dx, cx, ex, fx, *l, *r;
	Node lo1, lo2, lo3, hi1, hi2, hi3;
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
		p1 = gbranch(AJNE, T);
		gins(AMULL, &cx, N);	// implicit &ax
		p2 = gbranch(AJMP, T);
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
			p1 = gbranch(AJNE, T);
			gins(AMOVL, &lo2, &cx);
		} else
			gins(AMOVL, r, &cx);

		// if shift count is >=64, zero value
		gins(ACMPL, &cx, ncon(64));
		p2 = gbranch(optoas(OLT, types[TUINT32]), T);
		if(p1 != P)
			patch(p1, pc);
		gins(AXORL, &dx, &dx);
		gins(AXORL, &ax, &ax);
		patch(p2, pc);

		// if shift count is >= 32, zero low.
		gins(ACMPL, &cx, ncon(32));
		p1 = gbranch(optoas(OLT, types[TUINT32]), T);
		gins(AMOVL, &ax, &dx);
		gins(ASHLL, &cx, &dx);	// SHLL only uses bottom 5 bits of count
		gins(AXORL, &ax, &ax);
		p2 = gbranch(AJMP, T);
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
			p1 = gbranch(AJNE, T);
			gins(AMOVL, &lo2, &cx);
		} else
			gins(AMOVL, r, &cx);

		// if shift count is >=64, zero or sign-extend value
		gins(ACMPL, &cx, ncon(64));
		p2 = gbranch(optoas(OLT, types[TUINT32]), T);
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
		p1 = gbranch(optoas(OLT, types[TUINT32]), T);
		gins(AMOVL, &dx, &ax);
		if(hi1.type->etype == TINT32) {
			gins(ASARL, &cx, &ax);	// SARL only uses bottom 5 bits of count
			gins(ASARL, ncon(31), &dx);
		} else {
			gins(ASHRL, &cx, &ax);
			gins(AXORL, &dx, &dx);
		}
		p2 = gbranch(AJMP, T);
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

out:
	if(r == &t2)
		tempfree(&t2);
	if(l == &t1)
		tempfree(&t1);
}

/*
 * generate comparison of nl, nr, both 64-bit.
 * nl is memory; nr is constant or memory.
 */
void
cmp64(Node *nl, Node *nr, int op, Prog *to)
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
	patch(gbranch(optoas(op, t), T), to);

	// point first branch down here if appropriate
	if(br != P)
		patch(br, pc);

	splitclean();
	splitclean();
}

