// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
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
	Node n1, n2, f0, f1;
	int a, w, rg;
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

	switch(n->op) {
	case OSLICE:
	case OSLICEARR:
	case OSLICESTR:
	case OSLICE3:
	case OSLICE3ARR:
		if (res->op != ONAME || !res->addable) {
			tempname(&n1, n->type);
			cgen_slice(n, &n1);
			cgen(&n1, res);
		} else
			cgen_slice(n, res);
		return;
	case OEFACE:
		if (res->op != ONAME || !res->addable) {
			tempname(&n1, n->type);
			cgen_eface(n, &n1);
			cgen(&n1, res);
		} else
			cgen_eface(n, res);
		return;
	}

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
		if(n->type->width < 0)
			fatal("forgot to compute width for %T", n->type);
		sgen(n, res, n->type->width);
		goto ret;
	}


	// update addressability for string, slice
	// can't do in walk because n->left->addable
	// changes if n->left is an escaping local variable.
	switch(n->op) {
	case OSPTR:
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
		if(is64(n->type) || is64(res->type) ||
		   n->op == OREGISTER || res->op == OREGISTER ||
		   iscomplex[n->type->etype] || iscomplex[res->type->etype]) {
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

	if(complexop(n, res)) {
		complexgen(n, res);
		return;
	}

	// if n is sudoaddable generate addr and move
	if (!is64(n->type) && !is64(res->type) && !iscomplex[n->type->etype] && !iscomplex[res->type->etype]) {
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
		fatal("cgen: unknown op %+hN", n);
		break;

	case OREAL:
	case OIMAG:
	case OCOMPLEX:
		fatal("unexpected complex");
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
		p1 = gbranch(AB, T, 0);
		p2 = pc;
		gmove(nodbool(1), res);
		p3 = gbranch(AB, T, 0);
		patch(p1, pc);
		bgen(n, 1, 0, p2);
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
		goto norm;

	case OMINUS:
		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);
		nodconst(&n2, nl->type, 0);
		gins(optoas(OMINUS, nl->type), &n2, &n1);
		goto norm;

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

	case OHMUL:
		cgen_hmul(nl, nr, res);
		break;

	case OLROT:
	case OLSH:
	case ORSH:
		cgen_shift(n->op, n->bounded, nl, nr, res);
		break;

	case OCONV:
		if(eqtype(n->type, nl->type) || noconv(n->type, nl->type)) {
			cgen(nl, res);
			break;
		}
		if(nl->addable && !is64(nl->type)) {
			regalloc(&n1, nl->type, res);
			gmove(nl, &n1);
		} else {
			if(n->type->width > widthptr || is64(nl->type) || isfloat[nl->type->etype])
				tempname(&n1, nl->type);
			else
				regalloc(&n1, nl->type, res);
			cgen(nl, &n1);
		}
		if(n->type->width > widthptr || is64(n->type) || isfloat[n->type->etype])
			tempname(&n2, n->type);
		else
			regalloc(&n2, n->type, N);
		gmove(&n1, &n2);
		gmove(&n2, res);
		if(n1.op == OREGISTER)
			regfree(&n1);
		if(n2.op == OREGISTER)
			regfree(&n2);
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
		// interface table is first word of interface value
		igen(nl, &n1, res);
		n1.type = n->type;
		gmove(&n1, res);
		regfree(&n1);
		break;

	case OSPTR:
		// pointer is the first word of string or slice.
		if(isconst(nl, CTSTR)) {
			regalloc(&n1, types[tptr], res);
			p1 = gins(AMOVW, N, &n1);
			datastring(nl->val.u.sval->s, nl->val.u.sval->len, &p1->from);
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		igen(nl, &n1, res);
		n1.type = n->type;
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
			gcmp(optoas(OCMP, types[tptr]), &n1, &n2);
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
			gcmp(optoas(OCMP, types[tptr]), &n1, &n2);
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
			n1.type = types[TUINT32];
			n1.xoffset += Array_cap;
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
	case OCALLFUNC:
		// Release res so that it is available for cgen_call.
		// Pick it up again after the call.
		rg = -1;
		if(n->ullman >= UINF) {
			if(res != N && (res->op == OREGISTER || res->op == OINDREG)) {
				rg = res->val.u.reg;
				reg[rg]--;
			}
		}
		if(n->op == OCALLMETH)
			cgen_callmeth(n, 0);
		else
			cgen_call(n, 0);
		if(rg >= 0)
			reg[rg]++;
		cgen_callret(n, res);
		break;

	case OCALLINTER:
		cgen_callinter(n, res, 0);
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
		switch(n->op) {
		case OADD:
		case OSUB:
		case OAND:
		case OOR:
		case OXOR:
			if(smallintconst(nr)) {
				n2 = *nr;
				break;
			}
		default:
			regalloc(&n2, nr->type, N);
			cgen(nr, &n2);
		}
	} else {
		switch(n->op) {
		case OADD:
		case OSUB:
		case OAND:
		case OOR:
		case OXOR:
			if(smallintconst(nr)) {
				n2 = *nr;
				break;
			}
		default:
			regalloc(&n2, nr->type, res);
			cgen(nr, &n2);
		}
		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);
	}
	gins(a, &n2, &n1);
norm:
	// Normalize result for types smaller than word.
	if(n->type->width < widthptr) {
		switch(n->op) {
		case OADD:
		case OSUB:
		case OMUL:
		case OCOM:
		case OMINUS:
			gins(optoas(OAS, n->type), &n1, &n1);
			break;
		}
	}
	gmove(&n1, res);
	regfree(&n1);
	if(n2.op != OLITERAL)
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
 * generate array index into res.
 * n might be any size; res is 32-bit.
 * returns Prog* to patch to panic call.
 */
Prog*
cgenindex(Node *n, Node *res, int bounded)
{
	Node tmp, lo, hi, zero, n1, n2;

	if(!is64(n->type)) {
		cgen(n, res);
		return nil;
	}

	tempname(&tmp, types[TINT64]);
	cgen(n, &tmp);
	split64(&tmp, &lo, &hi);
	gmove(&lo, res);
	if(bounded) {
		splitclean();
		return nil;
	}
	regalloc(&n1, types[TINT32], N);
	regalloc(&n2, types[TINT32], N);
	nodconst(&zero, types[TINT32], 0);
	gmove(&hi, &n1);
	gmove(&zero, &n2);
	gcmp(ACMP, &n1, &n2);
	regfree(&n2);
	regfree(&n1);
	splitclean();
	return gbranch(ABNE, T, -1);
}

/*
 * generate:
 *	res = &n;
 * The generated code checks that the result is not nil.
 */
void
agen(Node *n, Node *res)
{
	Node *nl;
	Node n1, n2, n3;
	int r;

	if(debug['g']) {
		dump("\nagen-res", res);
		dump("agen-r", n);
	}
	if(n == N || n->type == T || res == N || res->type == T)
		fatal("agen");

	while(n->op == OCONVNOP)
		n = n->left;

	if(isconst(n, CTNIL) && n->type->width > widthptr) {
		// Use of a nil interface or nil slice.
		// Create a temporary we can take the address of and read.
		// The generated code is just going to panic, so it need not
		// be terribly efficient. See issue 3670.
		tempname(&n1, n->type);
		gvardef(&n1);
		clearfat(&n1);
		regalloc(&n2, types[tptr], res);
		gins(AMOVW, &n1, &n2);
		gmove(&n2, res);
		regfree(&n2);
		goto ret;
	}
		

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

	switch(n->op) {
	default:
		fatal("agen: unknown op %+hN", n);
		break;

	case OCALLMETH:
	case OCALLFUNC:
		// Release res so that it is available for cgen_call.
		// Pick it up again after the call.
		r = -1;
		if(n->ullman >= UINF) {
			if(res->op == OREGISTER || res->op == OINDREG) {
				r = res->val.u.reg;
				reg[r]--;
			}
		}
		if(n->op == OCALLMETH)
			cgen_callmeth(n, 0);
		else
			cgen_call(n, 0);
		if(r >= 0)
			reg[r]++;
		cgen_aret(n, res);
		break;

	case OCALLINTER:
		cgen_callinter(n, res, 0);
		cgen_aret(n, res);
		break;

	case OSLICE:
	case OSLICEARR:
	case OSLICESTR:
	case OSLICE3:
	case OSLICE3ARR:
		tempname(&n1, n->type);
		cgen_slice(n, &n1);
		agen(&n1, res);
		break;

	case OEFACE:
		tempname(&n1, n->type);
		cgen_eface(n, &n1);
		agen(&n1, res);
		break;

	case OINDEX:
		agenr(n, &n1, res);
		gmove(&n1, res);
		regfree(&n1);
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
		cgen_checknil(res);
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
		cgen_checknil(res);
		if(n->xoffset != 0) {
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
 * The generated code checks that the result is not *nil.
 */
void
igen(Node *n, Node *a, Node *res)
{
	Node n1;
	int r;

	if(debug['g']) {
		dump("\nigen-n", n);
	}
	switch(n->op) {
	case ONAME:
		if((n->class&PHEAP) || n->class == PPARAMREF)
			break;
		*a = *n;
		return;

	case OINDREG:
		// Increase the refcount of the register so that igen's caller
		// has to call regfree.
		if(n->val.u.reg != REGSP)
			reg[n->val.u.reg]++;
		*a = *n;
		return;

	case ODOT:
		igen(n->left, a, res);
		a->xoffset += n->xoffset;
		a->type = n->type;
		return;

	case ODOTPTR:
		if(n->left->addable
			|| n->left->op == OCALLFUNC
			|| n->left->op == OCALLMETH
			|| n->left->op == OCALLINTER) {
			// igen-able nodes.
			igen(n->left, &n1, res);
			regalloc(a, types[tptr], &n1);
			gmove(&n1, a);
			regfree(&n1);
		} else {
			regalloc(a, types[tptr], res);
			cgen(n->left, a);
		}
		cgen_checknil(a);
		a->op = OINDREG;
		a->xoffset = n->xoffset;
		a->type = n->type;
		return;

	case OCALLMETH:
	case OCALLFUNC:
	case OCALLINTER:
		// Release res so that it is available for cgen_call.
		// Pick it up again after the call.
		r = -1;
		if(n->ullman >= UINF) {
			if(res != N && (res->op == OREGISTER || res->op == OINDREG)) {
				r = res->val.u.reg;
				reg[r]--;
			}
		}
		switch(n->op) {
		case OCALLMETH:
			cgen_callmeth(n, 0);
			break;
		case OCALLFUNC:
			cgen_call(n, 0);
			break;
		case OCALLINTER:
			cgen_callinter(n, N, 0);
			break;
		}
		if(r >= 0)
			reg[r]++;
		regalloc(a, types[tptr], res);
		cgen_aret(n, a);
		a->op = OINDREG;
		a->type = n->type;
		return;
	}

	agenr(n, a, res);
	a->op = OINDREG;
	a->type = n->type;
}

/*
 * allocate a register in res and generate
 *  newreg = &n
 * The caller must call regfree(a).
 */
void
cgenr(Node *n, Node *a, Node *res)
{
	Node n1;

	if(debug['g'])
		dump("cgenr-n", n);

	if(isfat(n->type))
		fatal("cgenr on fat node");

	if(n->addable) {
		regalloc(a, types[tptr], res);
		gmove(n, a);
		return;
	}

	switch(n->op) {
	case ONAME:
	case ODOT:
	case ODOTPTR:
	case OINDEX:
	case OCALLFUNC:
	case OCALLMETH:
	case OCALLINTER:
		igen(n, &n1, res);
		regalloc(a, types[tptr], &n1);
		gmove(&n1, a);
		regfree(&n1);
		break;
	default:
		regalloc(a, n->type, res);
		cgen(n, a);
		break;
	}
}

/*
 * generate:
 *	newreg = &n;
 *
 * caller must regfree(a).
 * The generated code checks that the result is not nil.
 */
void
agenr(Node *n, Node *a, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, n4, tmp;
	Prog *p1, *p2;
	uint32 w;
	uint64 v;
	int bounded;

	if(debug['g'])
		dump("agenr-n", n);

	nl = n->left;
	nr = n->right;

	switch(n->op) {
	case ODOT:
	case ODOTPTR:
	case OCALLFUNC:
	case OCALLMETH:
	case OCALLINTER:
		igen(n, &n1, res);
		regalloc(a, types[tptr], &n1);
		agen(&n1, a);
		regfree(&n1);
		break;

	case OIND:
		cgenr(n->left, a, res);
		cgen_checknil(a);
		break;

	case OINDEX:
		p2 = nil;  // to be patched to panicindex.
		w = n->type->width;
		bounded = debug['B'] || n->bounded;
		if(nr->addable) {
			if(!isconst(nr, CTINT))
				tempname(&tmp, types[TINT32]);
			if(!isconst(nl, CTSTR))
				agenr(nl, &n3, res);
			if(!isconst(nr, CTINT)) {
				p2 = cgenindex(nr, &tmp, bounded);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
		} else
		if(nl->addable) {
			if(!isconst(nr, CTINT)) {
				tempname(&tmp, types[TINT32]);
				p2 = cgenindex(nr, &tmp, bounded);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
			if(!isconst(nl, CTSTR)) {
				agenr(nl, &n3, res);
			}
		} else {
			tempname(&tmp, types[TINT32]);
			p2 = cgenindex(nr, &tmp, bounded);
			nr = &tmp;
			if(!isconst(nl, CTSTR))
				agenr(nl, &n3, res);
			regalloc(&n1, tmp.type, N);
			gins(optoas(OAS, tmp.type), &tmp, &n1);
		}

		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// w is width

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
					regalloc(&n4, n1.type, N);
					gmove(&n1, &n4);
					nodconst(&n2, types[TUINT32], v);
					gcmp(optoas(OCMP, types[TUINT32]), &n4, &n2);
					regfree(&n4);
					p1 = gbranch(optoas(OGT, types[TUINT32]), T, +1);
					ginscall(panicindex, 0);
					patch(p1, pc);
				}

				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_array;
				gmove(&n1, &n3);
			}

			nodconst(&n2, types[tptr], v*w);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
			*a = n3;
			break;
		}

		regalloc(&n2, types[TINT32], &n1);			// i
		gmove(&n1, &n2);
		regfree(&n1);

		if(!debug['B'] && !n->bounded) {
			// check bounds
			if(isconst(nl, CTSTR)) {
				nodconst(&n4, types[TUINT32], nl->val.u.sval->len);
			} else if(isslice(nl->type) || nl->type->etype == TSTRING) {
				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_nel;
				regalloc(&n4, types[TUINT32], N);
				gmove(&n1, &n4);
			} else {
				nodconst(&n4, types[TUINT32], nl->type->bound);
			}
			gcmp(optoas(OCMP, types[TUINT32]), &n2, &n4);
			if(n4.op == OREGISTER)
				regfree(&n4);
			p1 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
			if(p2)
				patch(p2, pc);
			ginscall(panicindex, 0);
			patch(p1, pc);
		}
		
		if(isconst(nl, CTSTR)) {
			regalloc(&n3, types[tptr], res);
			p1 = gins(AMOVW, N, &n3);
			datastring(nl->val.u.sval->s, nl->val.u.sval->len, &p1->from);
			p1->from.type = TYPE_ADDR;
		} else
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
			regalloc(&n4, types[TUINT32], N);
			nodconst(&n1, types[TUINT32], w);
			gmove(&n1, &n4);
			gins(optoas(OMUL, types[TUINT32]), &n4, &n2);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
			regfree(&n4);
		}

		*a = n3;
		regfree(&n2);
		break;

	default:
		regalloc(a, types[tptr], res);
		agen(n, a);
		break;
	}
}

void
gencmp0(Node *n, Type *t, int o, int likely, Prog *to)
{
	Node n1, n2, n3;
	int a;

	regalloc(&n1, t, N);
	cgen(n, &n1);
	a = optoas(OCMP, t);
	if(a != ACMP) {
		nodconst(&n2, t, 0);
		regalloc(&n3, t, N);
		gmove(&n2, &n3);
		gcmp(a, &n1, &n3);
		regfree(&n3);
	} else
		gins(ATST, &n1, N);
	a = optoas(o, t);
	patch(gbranch(a, t, likely), to);
	regfree(&n1);
}

/*
 * generate:
 *	if(n == true) goto to;
 */
void
bgen(Node *n, int true, int likely, Prog *to)
{
	int et, a;
	Node *nl, *nr, *r;
	Node n1, n2, n3, tmp;
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
			goto ret;
	}

	et = n->type->etype;
	if(et != TBOOL) {
		yyerror("cgen: bad type %T for %O", n->type, n->op);
		patch(gins(AEND, N, N), to);
		goto ret;
	}
	nr = N;

	switch(n->op) {
	default:
		a = ONE;
		if(!true)
			a = OEQ;
		gencmp0(n, n->type, a, likely, to);
		goto ret;

	case OLITERAL:
		// need to ask if it is bool?
		if(!true == !n->val.u.bval)
			patch(gbranch(AB, T, 0), to);
		goto ret;

	case OANDAND:
	case OOROR:
		if((n->op == OANDAND) == true) {
			p1 = gbranch(AJMP, T, 0);
			p2 = gbranch(AJMP, T, 0);
			patch(p1, pc);
			bgen(n->left, !true, -likely, p2);
			bgen(n->right, !true, -likely, p2);
			p1 = gbranch(AJMP, T, 0);
			patch(p1, to);
			patch(p2, pc);
		} else {
			bgen(n->left, true, likely, to);
			bgen(n->right, true, likely, to);
		}
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
		bgen(nl, !true, likely, to);
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
				p1 = gbranch(AB, T, 0);
				p2 = gbranch(AB, T, 0);
				patch(p1, pc);
				ll = n->ninit;
				n->ninit = nil;
				bgen(n, 1, -likely, p2);
				n->ninit = ll;
				patch(gbranch(AB, T, 0), to);
				patch(p2, pc);
				goto ret;
			}				
			a = brcom(a);
			true = !true;
		}

		// make simplest on right
		if(nl->op == OLITERAL || (nl->ullman < UINF && nl->ullman < nr->ullman)) {
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

			igen(nl, &n1, N);
			n1.xoffset += Array_array;
			n1.type = types[tptr];
			gencmp0(&n1, types[tptr], a, likely, to);
			regfree(&n1);
			break;
		}

		if(isinter(nl->type)) {
			// front end shold only leave cmp to literal nil
			if((a != OEQ && a != ONE) || nr->op != OLITERAL) {
				yyerror("illegal interface comparison");
				break;
			}

			igen(nl, &n1, N);
			n1.type = types[tptr];
			n1.xoffset += 0;
			gencmp0(&n1, types[tptr], a, likely, to);
			regfree(&n1);
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

		if(nr->op == OLITERAL) {
			if(isconst(nr, CTINT) &&  mpgetfix(nr->val.u.xval) == 0) {
				gencmp0(nl, nl->type, a, likely, to);
				break;
			}
			if(nr->val.ctype == CTNIL) {
				gencmp0(nl, nl->type, a, likely, to);
				break;
			}
		}

		a = optoas(a, nr->type);

		if(nr->ullman >= UINF) {
			regalloc(&n1, nl->type, N);
			cgen(nl, &n1);

			tempname(&tmp, nl->type);
			gmove(&n1, &tmp);
			regfree(&n1);

			regalloc(&n2, nr->type, N);
			cgen(nr, &n2);

			regalloc(&n1, nl->type, N);
			cgen(&tmp, &n1);

			gcmp(optoas(OCMP, nr->type), &n1, &n2);
			patch(gbranch(a, nr->type, likely), to);

			regfree(&n1);
			regfree(&n2);
			break;
		}

		tempname(&n3, nl->type);
		cgen(nl, &n3);

		tempname(&tmp, nr->type);
		cgen(nr, &tmp);

		regalloc(&n1, nl->type, N);
		gmove(&n3, &n1);

		regalloc(&n2, nr->type, N);
		gmove(&tmp, &n2);

		gcmp(optoas(OCMP, nr->type), &n1, &n2);
		if(isfloat[nl->type->etype]) {
			if(n->op == ONE) {
				p1 = gbranch(ABVS, nr->type, likely);
				patch(gbranch(a, nr->type, likely), to);
				patch(p1, to);
			} else {
				p1 = gbranch(ABVS, nr->type, -likely);
				patch(gbranch(a, nr->type, likely), to);
				patch(p1, pc);
			}
		} else {
			patch(gbranch(a, nr->type, likely), to);
		}
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
			return t->width + 4;	// correct for LR
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
sgen(Node *n, Node *res, int64 w)
{
	Node dst, src, tmp, nend, r0, r1, r2, *f;
	int32 c, odst, osrc;
	int dir, align, op;
	Prog *p, *ploop;
	NodeList *l;

	if(debug['g']) {
		print("\nsgen w=%lld\n", w);
		dump("r", n);
		dump("res", res);
	}

	if(n->ullman >= UINF && res->ullman >= UINF)
		fatal("sgen UINF");

	if(w < 0 || (int32)w != w)
		fatal("sgen copy %lld", w);

	if(n->type == T)
		fatal("sgen: missing type");

	if(w == 0) {
		// evaluate side effects only.
		regalloc(&dst, types[tptr], N);
		agen(res, &dst);
		agen(n, &dst);
		regfree(&dst);
		return;
	}

	// If copying .args, that's all the results, so record definition sites
	// for them for the liveness analysis.
	if(res->op == ONAME && strcmp(res->sym->name, ".args") == 0)
		for(l = curfn->dcl; l != nil; l = l->next)
			if(l->n->class == PPARAMOUT)
				gvardef(l->n);

	// Avoid taking the address for simple enough types.
	if(componentgen(n, res))
		return;
	
	// determine alignment.
	// want to avoid unaligned access, so have to use
	// smaller operations for less aligned types.
	// for example moving [4]byte must use 4 MOVB not 1 MOVW.
	align = n->type->align;
	switch(align) {
	default:
		fatal("sgen: invalid alignment %d for %T", align, n->type);
	case 1:
		op = AMOVB;
		break;
	case 2:
		op = AMOVH;
		break;
	case 4:
		op = AMOVW;
		break;
	}
	if(w%align)
		fatal("sgen: unaligned size %lld (align=%d) for %T", w, align, n->type);
	c = w / align;

	// offset on the stack
	osrc = stkof(n);
	odst = stkof(res);
	if(osrc != -1000 && odst != -1000 && (osrc == 1000 || odst == 1000)) {
		// osrc and odst both on stack, and at least one is in
		// an unknown position.  Could generate code to test
		// for forward/backward copy, but instead just copy
		// to a temporary location first.
		tempname(&tmp, n->type);
		sgen(n, &tmp, w);
		sgen(&tmp, res, w);
		return;
	}
	if(osrc%align != 0 || odst%align != 0)
		fatal("sgen: unaligned offset src %d or dst %d (align %d)", osrc, odst, align);

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	dir = align;
	if(osrc < odst && odst < osrc+w)
		dir = -dir;

	if(op == AMOVW && !nacl && dir > 0 && c >= 4 && c <= 128) {
		r0.op = OREGISTER;
		r0.val.u.reg = REGALLOC_R0;
		r1.op = OREGISTER;
		r1.val.u.reg = REGALLOC_R0 + 1;
		r2.op = OREGISTER;
		r2.val.u.reg = REGALLOC_R0 + 2;

		regalloc(&src, types[tptr], &r1);
		regalloc(&dst, types[tptr], &r2);
		if(n->ullman >= res->ullman) {
			// eval n first
			agen(n, &src);
			if(res->op == ONAME)
				gvardef(res);
			agen(res, &dst);
		} else {
			// eval res first
			if(res->op == ONAME)
				gvardef(res);
			agen(res, &dst);
			agen(n, &src);
		}
		regalloc(&tmp, types[tptr], &r0);
		f = sysfunc("duffcopy");
		p = gins(ADUFFCOPY, N, f);
		afunclit(&p->to, f);
		// 8 and 128 = magic constants: see ../../runtime/asm_arm.s
		p->to.offset = 8*(128-c);

		regfree(&tmp);
		regfree(&src);
		regfree(&dst);
		return;
	}
	
	if(n->ullman >= res->ullman) {
		agenr(n, &dst, res);	// temporarily use dst
		regalloc(&src, types[tptr], N);
		gins(AMOVW, &dst, &src);
		if(res->op == ONAME)
			gvardef(res);
		agen(res, &dst);
	} else {
		if(res->op == ONAME)
			gvardef(res);
		agenr(res, &dst, res);
		agenr(n, &src, N);
	}

	regalloc(&tmp, types[TUINT32], N);

	// set up end marker
	memset(&nend, 0, sizeof nend);
	if(c >= 4) {
		regalloc(&nend, types[TUINT32], N);

		p = gins(AMOVW, &src, &nend);
		p->from.type = TYPE_ADDR;
		if(dir < 0)
			p->from.offset = dir;
		else
			p->from.offset = w;
	}

	// move src and dest to the end of block if necessary
	if(dir < 0) {
		p = gins(AMOVW, &src, &src);
		p->from.type = TYPE_ADDR;
		p->from.offset = w + dir;

		p = gins(AMOVW, &dst, &dst);
		p->from.type = TYPE_ADDR;
		p->from.offset = w + dir;
	}
	
	// move
	if(c >= 4) {
		p = gins(op, &src, &tmp);
		p->from.type = TYPE_MEM;
		p->from.offset = dir;
		p->scond |= C_PBIT;
		ploop = p;

		p = gins(op, &tmp, &dst);
		p->to.type = TYPE_MEM;
		p->to.offset = dir;
		p->scond |= C_PBIT;

		p = gins(ACMP, &src, N);
		raddr(&nend, p);

		patch(gbranch(ABNE, T, 0), ploop);
 		regfree(&nend);
	} else {
		while(c-- > 0) {
			p = gins(op, &src, &tmp);
			p->from.type = TYPE_MEM;
			p->from.offset = dir;
			p->scond |= C_PBIT;
	
			p = gins(op, &tmp, &dst);
			p->to.type = TYPE_MEM;
			p->to.offset = dir;
			p->scond |= C_PBIT;
		}
	}

	regfree(&dst);
	regfree(&src);
	regfree(&tmp);
}

static int
cadable(Node *n)
{
	if(!n->addable) {
		// dont know how it happens,
		// but it does
		return 0;
	}

	switch(n->op) {
	case ONAME:
		return 1;
	}
	return 0;
}

/*
 * copy a composite value by moving its individual components.
 * Slices, strings and interfaces are supported.
 * Small structs or arrays with elements of basic type are
 * also supported.
 * nr is N when assigning a zero value.
 * return 1 if can do, 0 if cant.
 */
int
componentgen(Node *nr, Node *nl)
{
	Node nodl, nodr, tmp;
	Type *t;
	int freel, freer;
	vlong fldcount;
	vlong loffset, roffset;

	freel = 0;
	freer = 0;

	switch(nl->type->etype) {
	default:
		goto no;

	case TARRAY:
		t = nl->type;

		// Slices are ok.
		if(isslice(t))
			break;
		// Small arrays are ok.
		if(t->bound > 0 && t->bound <= 3 && !isfat(t->type))
			break;

		goto no;

	case TSTRUCT:
		// Small structs with non-fat types are ok.
		// Zero-sized structs are treated separately elsewhere.
		fldcount = 0;
		for(t=nl->type->type; t; t=t->down) {
			if(isfat(t->type))
				goto no;
			if(t->etype != TFIELD)
				fatal("componentgen: not a TFIELD: %lT", t);
			fldcount++;
		}
		if(fldcount == 0 || fldcount > 4)
			goto no;

		break;

	case TSTRING:
	case TINTER:
		break;
	}

	nodl = *nl;
	if(!cadable(nl)) {
		if(nr != N && !cadable(nr))
			goto no;
		igen(nl, &nodl, N);
		freel = 1;
	}

	if(nr != N) {
		nodr = *nr;
		if(!cadable(nr)) {
			igen(nr, &nodr, N);
			freer = 1;
		}
	} else {
		// When zeroing, prepare a register containing zero.
		nodconst(&tmp, nl->type, 0);
		regalloc(&nodr, types[TUINT], N);
		gmove(&tmp, &nodr);
		freer = 1;
	}

	// nl and nr are 'cadable' which basically means they are names (variables) now.
	// If they are the same variable, don't generate any code, because the
	// VARDEF we generate will mark the old value as dead incorrectly.
	// (And also the assignments are useless.)
	if(nr != N && nl->op == ONAME && nr->op == ONAME && nl == nr)
		goto yes;

	switch(nl->type->etype) {
	case TARRAY:
		// componentgen for arrays.
		if(nl->op == ONAME)
			gvardef(nl);
		t = nl->type;
		if(!isslice(t)) {
			nodl.type = t->type;
			nodr.type = nodl.type;
			for(fldcount=0; fldcount < t->bound; fldcount++) {
				if(nr == N)
					clearslim(&nodl);
				else
					gmove(&nodr, &nodl);
				nodl.xoffset += t->type->width;
				nodr.xoffset += t->type->width;
			}
			goto yes;
		}

		// componentgen for slices.
		nodl.xoffset += Array_array;
		nodl.type = ptrto(nl->type->type);

		if(nr != N) {
			nodr.xoffset += Array_array;
			nodr.type = nodl.type;
		}
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = types[simtype[TUINT]];

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		}
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_cap-Array_nel;
		nodl.type = types[simtype[TUINT]];

		if(nr != N) {
			nodr.xoffset += Array_cap-Array_nel;
			nodr.type = nodl.type;
		}
		gmove(&nodr, &nodl);

		goto yes;

	case TSTRING:
		if(nl->op == ONAME)
			gvardef(nl);
		nodl.xoffset += Array_array;
		nodl.type = ptrto(types[TUINT8]);

		if(nr != N) {
			nodr.xoffset += Array_array;
			nodr.type = nodl.type;
		}
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = types[simtype[TUINT]];

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		}
		gmove(&nodr, &nodl);

		goto yes;

	case TINTER:
		if(nl->op == ONAME)
			gvardef(nl);
		nodl.xoffset += Array_array;
		nodl.type = ptrto(types[TUINT8]);

		if(nr != N) {
			nodr.xoffset += Array_array;
			nodr.type = nodl.type;
		}
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = ptrto(types[TUINT8]);

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		}
		gmove(&nodr, &nodl);

		goto yes;

	case TSTRUCT:
		if(nl->op == ONAME)
			gvardef(nl);
		loffset = nodl.xoffset;
		roffset = nodr.xoffset;
		// funarg structs may not begin at offset zero.
		if(nl->type->etype == TSTRUCT && nl->type->funarg && nl->type->type)
			loffset -= nl->type->type->width;
		if(nr != N && nr->type->etype == TSTRUCT && nr->type->funarg && nr->type->type)
			roffset -= nr->type->type->width;

		for(t=nl->type->type; t; t=t->down) {
			nodl.xoffset = loffset + t->width;
			nodl.type = t->type;

			if(nr == N)
				clearslim(&nodl);
			else {
				nodr.xoffset = roffset + t->width;
				nodr.type = nodl.type;
				gmove(&nodr, &nodl);
			}
		}
		goto yes;
	}

no:
	if(freer)
		regfree(&nodr);
	if(freel)
		regfree(&nodl);
	return 0;

yes:
	if(freer)
		regfree(&nodr);
	if(freel)
		regfree(&nodl);
	return 1;
}
