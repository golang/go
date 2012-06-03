// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc):
//	assume CLD?

#include <u.h>
#include <libc.h>
#include "gg.h"

void
mgen(Node *n, Node *n1, Node *rg)
{
	Node n2;

	n1->op = OEMPTY;

	if(n->addable) {
		*n1 = *n;
		if(n1->op == OREGISTER || n1->op == OINDREG)
			reg[n->val.u.reg]++;
		return;
	}
	tempname(n1, n->type);
	cgen(n, n1);
	if(n->type->width <= widthptr || isfloat[n->type->etype]) {
		n2 = *n1;
		regalloc(n1, n->type, rg);
		gmove(&n2, n1);
	}
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
 *
 * TODO:
 *	sudoaddable
 */
void
cgen(Node *n, Node *res)
{
	Node *nl, *nr, *r, n1, n2, nt, f0, f1;
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

	switch(n->op) {
	case OSLICE:
	case OSLICEARR:
	case OSLICESTR:
		if (res->op != ONAME || !res->addable) {
			tempname(&n1, n->type);
			cgen_slice(n, &n1);
			cgen(&n1, res);
		} else
			cgen_slice(n, res);
		return;
	}

	while(n->op == OCONVNOP)
		n = n->left;

	// function calls on both sides?  introduce temporary
	if(n->ullman >= UINF && res->ullman >= UINF) {
		tempname(&n1, n->type);
		cgen(n, &n1);
		cgen(&n1, res);
		return;
	}

	// structs etc get handled specially
	if(isfat(n->type)) {
		if(n->type->width < 0)
			fatal("forgot to compute width for %T", n->type);
		sgen(n, res, n->type->width);
		return;
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
	case OITAB:
		n->addable = n->left->addable;
		break;
	}

	// if both are addressable, move
	if(n->addable && res->addable) {
		gmove(n, res);
		return;
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

	// complex types
	if(complexop(n, res)) {
		complexgen(n, res);
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
		tempname(&n1, nl->type);
		cgen(nl, &n1);
		n2 = *n;
		n2.left = &n1;
		cgen(&n2, res);
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
		case OLROT:
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
		fatal("cgen %O", n->op);
		break;

	case OREAL:
	case OIMAG:
	case OCOMPLEX:
		fatal("unexpected complex");
		return;

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
		p1 = gbranch(AJMP, T, 0);
		p2 = pc;
		gmove(nodbool(1), res);
		p3 = gbranch(AJMP, T, 0);
		patch(p1, pc);
		bgen(n, 1, 0, p2);
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
		if(a == AIMULB) {
			cgen_bmul(n->op, nl, nr, res);
			break;
		}
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

		tempname(&n2, n->type);
		mgen(nl, &n1, res);
		gmove(&n1, &n2);
		gmove(&n2, res);
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

	case OITAB:
		igen(nl, &n1, res);
		n1.type = ptrto(types[TUINTPTR]);
		gmove(&n1, res);
		regfree(&n1);
		break;

	case OLEN:
		if(istype(nl->type, TMAP) || istype(nl->type, TCHAN)) {
			// map has len in the first 32-bit word.
			// a zero pointer means zero length
			tempname(&n1, types[tptr]);
			cgen(nl, &n1);
			regalloc(&n2, types[tptr], N);
			gmove(&n1, &n2);
			n1 = n2;

			nodconst(&n2, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n1, &n2);
			p1 = gbranch(optoas(OEQ, types[tptr]), T, -1);

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
			n1.type = types[TUINT32];
			n1.xoffset += Array_nel;
			gmove(&n1, res);
			regfree(&n1);
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
			gins(optoas(OCMP, types[tptr]), &n1, &n2);
			p1 = gbranch(optoas(OEQ, types[tptr]), T, -1);

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

	case OCALLFUNC:
		cgen_call(n, 0);
		cgen_callret(n, res);
		break;

	case OMOD:
	case ODIV:
		cgen_div(n->op, nl, nr, res);
		break;

	case OLSH:
	case ORSH:
	case OLROT:
		cgen_shift(n->op, n->bounded, nl, nr, res);
		break;
	}
	return;

sbop:	// symmetric binary
	if(nl->ullman < nr->ullman || nl->op == OLITERAL) {
		r = nl;
		nl = nr;
		nr = r;
	}

abop:	// asymmetric binary
	if(smallintconst(nr)) {
		regalloc(&n1, nr->type, res);
		cgen(nl, &n1);
		gins(a, nr, &n1);
		gmove(&n1, res);
		regfree(&n1);
	} else if(nl->ullman >= nr->ullman) {
		tempname(&nt, nl->type);
		cgen(nl, &nt);
		mgen(nr, &n2, N);
		regalloc(&n1, nl->type, res);
		gmove(&nt, &n1);
		gins(a, &n2, &n1);
		gmove(&n1, res);
		regfree(&n1);
		mfree(&n2);
	} else {
		regalloc(&n2, nr->type, res);
		cgen(nr, &n2);
		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);
		gins(a, &n2, &n1);
		regfree(&n2);
		gmove(&n1, res);
		regfree(&n1);
	}
	return;

uop:	// unary
	tempname(&n1, nl->type);
	cgen(nl, &n1);
	gins(a, N, &n1);
	gmove(&n1, res);
	return;

flt:	// floating-point.  387 (not SSE2) to interoperate with 8c
	nodreg(&f0, nl->type, D_F0);
	nodreg(&f1, n->type, D_F0+1);
	if(nr != N)
		goto flt2;

	// unary
	cgen(nl, &f0);
	if(n->op != OCONV && n->op != OPLUS)
		gins(foptoas(n->op, n->type, 0), N, N);
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
 * generate array index into res.
 * n might be any size; res is 32-bit.
 * returns Prog* to patch to panic call.
 */
Prog*
cgenindex(Node *n, Node *res)
{
	Node tmp, lo, hi, zero;

	if(!is64(n->type)) {
		cgen(n, res);
		return nil;
	}

	tempname(&tmp, types[TINT64]);
	cgen(n, &tmp);
	split64(&tmp, &lo, &hi);
	gmove(&lo, res);
	if(debug['B']) {
		splitclean();
		return nil;
	}
	nodconst(&zero, types[TINT32], 0);
	gins(ACMPL, &hi, &zero);
	splitclean();
	return gbranch(AJNE, T, +1);
}
		
/*
 * address gen
 *	res = &n;
 */
void
agen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, n4, tmp;
	Type *t;
	uint32 w;
	uint64 v;
	Prog *p1, *p2;

	if(debug['g']) {
		dump("\nagen-res", res);
		dump("agen-r", n);
	}
	if(n == N || n->type == T || res == N || res->type == T)
		fatal("agen");

	while(n->op == OCONVNOP)
		n = n->left;

	// addressable var is easy
	if(n->addable) {
		if(n->op == OREGISTER)
			fatal("agen OREGISTER");
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

	case OSLICE:
	case OSLICEARR:
	case OSLICESTR:
		tempname(&n1, n->type);
		cgen_slice(n, &n1);
		agen(&n1, res);
		break;

	case OINDEX:
		p2 = nil;  // to be patched to panicindex.
		w = n->type->width;
		if(nr->addable) {
			if(!isconst(nr, CTINT))
				tempname(&tmp, types[TINT32]);
			if(!isconst(nl, CTSTR))
				agenr(nl, &n3, res);
			if(!isconst(nr, CTINT)) {
				p2 = cgenindex(nr, &tmp);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
		} else if(nl->addable) {
			if(!isconst(nr, CTINT)) {
				tempname(&tmp, types[TINT32]);
				p2 = cgenindex(nr, &tmp);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
			if(!isconst(nl, CTSTR)) {
				regalloc(&n3, types[tptr], res);
				agen(nl, &n3);
			}
		} else {
			tempname(&tmp, types[TINT32]);
			p2 = cgenindex(nr, &tmp);
			nr = &tmp;
			if(!isconst(nl, CTSTR))
				agenr(nl, &n3, res);
			regalloc(&n1, tmp.type, N);
			gins(optoas(OAS, tmp.type), &tmp, &n1);
		}

		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// w is width

		// explicit check for nil if array is large enough
		// that we might derive too big a pointer.
		if(isfixedarray(nl->type) && nl->type->width >= unmappedzero) {
			regalloc(&n4, types[tptr], &n3);
			gmove(&n3, &n4);
			n4.op = OINDREG;
			n4.type = types[TUINT8];
			n4.xoffset = 0;
			gins(ATESTB, nodintconst(0), &n4);
			regfree(&n4);
		}

		// constant index
		if(isconst(nr, CTINT)) {
			if(isconst(nl, CTSTR))
				fatal("constant string constant index");
			v = mpgetfix(nr->val.u.xval);
			if(isslice(nl->type) || nl->type->etype == TSTRING) {
				if(!debug['B'] && !n->bounded) {
					n1 = n3;
					n1.op = OINDREG;
					n1.type = types[tptr];
					n1.xoffset = Array_nel;
					nodconst(&n2, types[TUINT32], v);
					gins(optoas(OCMP, types[TUINT32]), &n1, &n2);
					p1 = gbranch(optoas(OGT, types[TUINT32]), T, +1);
					ginscall(panicindex, -1);
					patch(p1, pc);
				}

				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_array;
				gmove(&n1, &n3);
			}

			if (v*w != 0) {
				nodconst(&n2, types[tptr], v*w);
				gins(optoas(OADD, types[tptr]), &n2, &n3);
			}
			gmove(&n3, res);
			regfree(&n3);
			break;
		}

		regalloc(&n2, types[TINT32], &n1);			// i
		gmove(&n1, &n2);
		regfree(&n1);

		if(!debug['B'] && !n->bounded) {
			// check bounds
			if(isconst(nl, CTSTR))
				nodconst(&n1, types[TUINT32], nl->val.u.sval->len);
			else if(isslice(nl->type) || nl->type->etype == TSTRING) {
				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_nel;
			} else
				nodconst(&n1, types[TUINT32], nl->type->bound);
			gins(optoas(OCMP, types[TUINT32]), &n2, &n1);
			p1 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
			if(p2)
				patch(p2, pc);
			ginscall(panicindex, -1);
			patch(p1, pc);
		}
		
		if(isconst(nl, CTSTR)) {
			regalloc(&n3, types[tptr], res);
			p1 = gins(ALEAL, N, &n3);
			datastring(nl->val.u.sval->s, nl->val.u.sval->len, &p1->from);
			p1->from.scale = 1;
			p1->from.index = n2.val.u.reg;
			goto indexdone;
		}

		if(isslice(nl->type) || nl->type->etype == TSTRING) {
			n1 = n3;
			n1.op = OINDREG;
			n1.type = types[tptr];
			n1.xoffset = Array_array;
			gmove(&n1, &n3);
		}

		if(w == 0) {
			// nothing to do
		} else if(w == 1 || w == 2 || w == 4 || w == 8) {
			p1 = gins(ALEAL, &n2, &n3);
			p1->from.scale = w;
			p1->from.index = p1->from.type;
			p1->from.type = p1->to.type + D_INDIR;
		} else {
			nodconst(&n1, types[TUINT32], w);
			gins(optoas(OMUL, types[TUINT32]), &n1, &n2);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
		}

	indexdone:
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
			// explicit check for nil if struct is large enough
			// that we might derive too big a pointer.
			if(nl->type->type->width >= unmappedzero) {
				regalloc(&n1, types[tptr], res);
				gmove(res, &n1);
				n1.op = OINDREG;
				n1.type = types[TUINT8];
				n1.xoffset = 0;
				gins(ATESTB, nodintconst(0), &n1);
				regfree(&n1);
			}
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
	Type *fp;
	Iter flist;
  
	switch(n->op) {
	case ONAME:
		if((n->class&PHEAP) || n->class == PPARAMREF)
			break;
		*a = *n;
		return;
 
	case OCALLFUNC:
		fp = structfirst(&flist, getoutarg(n->left->type));
		cgen_call(n, 0);
		memset(a, 0, sizeof *a);
		a->op = OINDREG;
		a->val.u.reg = D_SP;
		a->addable = 1;
		a->xoffset = fp->width;
		a->type = n->type;
		return;
	}
	// release register for now, to avoid
	// confusing tempname.
	if(res != N && res->op == OREGISTER)
		reg[res->val.u.reg]--;
	tempname(&n1, types[tptr]);
	agen(n, &n1);
	if(res != N && res->op == OREGISTER)
		reg[res->val.u.reg]++;
	regalloc(a, types[tptr], res);
	gmove(&n1, a);
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
 * branch gen
 *	if(n == true) goto to;
 */
void
bgen(Node *n, int true, int likely, Prog *to)
{
	int et, a;
	Node *nl, *nr, *r;
	Node n1, n2, tmp, t1, t2, ax;
	NodeList *ll;
	Prog *p1, *p2;

	if(debug['g']) {
		dump("\nbgen", n);
	}

	if(n == N)
		n = nodbool(1);

	if(n->ninit != nil)
		genlist(n->ninit);

	if(n->type == T) {
		convlit(&n, types[TBOOL]);
		if(n->type == T)
			return;
	}

	et = n->type->etype;
	if(et != TBOOL) {
		yyerror("cgen: bad type %T for %O", n->type, n->op);
		patch(gins(AEND, N, N), to);
		return;
	}
	nr = N;

	switch(n->op) {
	default:
	def:
		regalloc(&n1, n->type, N);
		cgen(n, &n1);
		nodconst(&n2, n->type, 0);
		gins(optoas(OCMP, n->type), &n1, &n2);
		a = AJNE;
		if(!true)
			a = AJEQ;
		patch(gbranch(a, n->type, likely), to);
		regfree(&n1);
		return;

	case OLITERAL:
		// need to ask if it is bool?
		if(!true == !n->val.u.bval)
			patch(gbranch(AJMP, T, 0), to);
		return;

	case ONAME:
		if(!n->addable)
			goto def;
		nodconst(&n1, n->type, 0);
		gins(optoas(OCMP, n->type), n, &n1);
		a = AJNE;
		if(!true)
			a = AJEQ;
		patch(gbranch(a, n->type, likely), to);
		return;

	case OANDAND:
		if(!true)
			goto caseor;

	caseand:
		p1 = gbranch(AJMP, T, 0);
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);
		bgen(n->left, !true, -likely, p2);
		bgen(n->right, !true, -likely, p2);
		p1 = gbranch(AJMP, T, 0);
		patch(p1, to);
		patch(p2, pc);
		return;

	case OOROR:
		if(!true)
			goto caseand;

	caseor:
		bgen(n->left, true, likely, to);
		bgen(n->right, true, likely, to);
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
		bgen(nl, !true, likely, to);
		break;

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
				p1 = gbranch(AJMP, T, 0);
				p2 = gbranch(AJMP, T, 0);
				patch(p1, pc);
				ll = n->ninit;  // avoid re-genning ninit
				n->ninit = nil;
				bgen(n, 1, -likely, p2);
				n->ninit = ll;
				patch(gbranch(AJMP, T, 0), to);
				patch(p2, pc);
				break;
			}				
			a = brcom(a);
			true = !true;
		}

		// make simplest on right
		if(nl->op == OLITERAL || (nl->ullman < nr->ullman && nl->ullman < UINF)) {
			a = brrev(a);
			r = nl;
			nl = nr;
			nr = r;
		}

		if(isslice(nl->type)) {
			// front end should only leave cmp to literal nil
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
			n2.type = types[tptr];
			nodconst(&tmp, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n2, &tmp);
			patch(gbranch(a, types[tptr], likely), to);
			regfree(&n1);
			break;
		}

		if(isinter(nl->type)) {
			// front end should only leave cmp to literal nil
			if((a != OEQ && a != ONE) || nr->op != OLITERAL) {
				yyerror("illegal interface comparison");
				break;
			}
			a = optoas(a, types[tptr]);
			regalloc(&n1, types[tptr], N);
			agen(nl, &n1);
			n2 = n1;
			n2.op = OINDREG;
			n2.xoffset = 0;
			nodconst(&tmp, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n2, &tmp);
			patch(gbranch(a, types[tptr], likely), to);
			regfree(&n1);
			break;
		}

		if(isfloat[nr->type->etype]) {
			a = brrev(a);	// because the args are stacked
			if(a == OGE || a == OGT) {
				// only < and <= work right with NaN; reverse if needed
				r = nr;
				nr = nl;
				nl = r;
				a = brrev(a);
			}
			nodreg(&tmp, nr->type, D_F0);
			nodreg(&n2, nr->type, D_F0 + 1);
			nodreg(&ax, types[TUINT16], D_AX);
			et = simsimtype(nr->type);
			if(et == TFLOAT64) {
				if(nl->ullman > nr->ullman) {
					cgen(nl, &tmp);
					cgen(nr, &tmp);
					gins(AFXCHD, &tmp, &n2);
				} else {
					cgen(nr, &tmp);
					cgen(nl, &tmp);
				}
				gins(AFUCOMIP, &tmp, &n2);
				gins(AFMOVDP, &tmp, &tmp);	// annoying pop but still better than STSW+SAHF
			} else {
				// TODO(rsc): The moves back and forth to memory
				// here are for truncating the value to 32 bits.
				// This handles 32-bit comparison but presumably
				// all the other ops have the same problem.
				// We need to figure out what the right general
				// solution is, besides telling people to use float64.
				tempname(&t1, types[TFLOAT32]);
				tempname(&t2, types[TFLOAT32]);
				cgen(nr, &t1);
				cgen(nl, &t2);
				gmove(&t2, &tmp);
				gins(AFCOMFP, &t1, &tmp);
				gins(AFSTSW, N, &ax);
				gins(ASAHF, N, N);
			}
			if(a == OEQ) {
				// neither NE nor P
				p1 = gbranch(AJNE, T, -likely);
				p2 = gbranch(AJPS, T, -likely);
				patch(gbranch(AJMP, T, 0), to);
				patch(p1, pc);
				patch(p2, pc);
			} else if(a == ONE) {
				// either NE or P
				patch(gbranch(AJNE, T, likely), to);
				patch(gbranch(AJPS, T, likely), to);
			} else
				patch(gbranch(optoas(a, nr->type), T, likely), to);
			break;
		}
		if(iscomplex[nl->type->etype]) {
			complexbool(a, nl, nr, true, likely, to);
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
			cmp64(nl, nr, a, likely, to);
			break;
		}

		a = optoas(a, nr->type);

		if(nr->ullman >= UINF) {
			tempname(&n1, nl->type);
			tempname(&tmp, nr->type);
			cgen(nl, &n1);
			cgen(nr, &tmp);
			regalloc(&n2, nr->type, N);
			cgen(&tmp, &n2);
			goto cmp;
		}

		tempname(&n1, nl->type);
		cgen(nl, &n1);

		if(smallintconst(nr)) {
			gins(optoas(OCMP, nr->type), &n1, nr);
			patch(gbranch(a, nr->type, likely), to);
			break;
		}

		tempname(&tmp, nr->type);
		cgen(nr, &tmp);
		regalloc(&n2, nr->type, N);
		gmove(&tmp, &n2);

cmp:
		gins(optoas(OCMP, nr->type), &n1, &n2);
		patch(gbranch(a, nr->type, likely), to);
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
	int32 off;

	switch(n->op) {
	case OINDREG:
		return n->xoffset;

	case ODOT:
		t = n->left->type;
		if(isptr[t->etype])
			break;
		off = stkof(n->left);
		if(off == -1000 || off == 1000)
			return off;
		return off + n->xoffset;

	case OINDEX:
		t = n->left->type;
		if(!isfixedarray(t))
			break;
		off = stkof(n->left);
		if(off == -1000 || off == 1000)
			return off;
		if(isconst(n->right, CTINT))
			return off + t->type->width * mpgetfix(n->right->val.u.xval);
		return 1000;
		
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
 * struct gen
 *	memmove(&res, &n, w);
 */
void
sgen(Node *n, Node *res, int64 w)
{
	Node dst, src, tdst, tsrc;
	int32 c, q, odst, osrc;

	if(debug['g']) {
		print("\nsgen w=%lld\n", w);
		dump("r", n);
		dump("res", res);
	}
	if(n->ullman >= UINF && res->ullman >= UINF)
		fatal("sgen UINF");

	if(w < 0 || (int32)w != w)
		fatal("sgen copy %lld", w);

	if(w == 0) {
		// evaluate side effects only.
		tempname(&tdst, types[tptr]);
		agen(res, &tdst);
		agen(n, &tdst);
		return;
	}

	// offset on the stack
	osrc = stkof(n);
	odst = stkof(res);
	
	if(osrc != -1000 && odst != -1000 && (osrc == 1000 || odst == 1000)) {
		// osrc and odst both on stack, and at least one is in
		// an unknown position.  Could generate code to test
		// for forward/backward copy, but instead just copy
		// to a temporary location first.
		tempname(&tsrc, n->type);
		sgen(n, &tsrc, w);
		sgen(&tsrc, res, w);
		return;
	}

	nodreg(&dst, types[tptr], D_DI);
	nodreg(&src, types[tptr], D_SI);

	tempname(&tsrc, types[tptr]);
	tempname(&tdst, types[tptr]);
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

