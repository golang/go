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
	Node *nl, *nr, *r, n1, n2, nt;
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

	if(nl != N && isfloat[n->type->etype] && isfloat[nl->type->etype]) {
		cgen_float(n, res);
		return;
	}

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

	case OHMUL:
		cgen_hmul(nl, nr, res);
		break;

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

	case OSPTR:
		// pointer is the first word of string or slice.
		if(isconst(nl, CTSTR)) {
			regalloc(&n1, types[tptr], res);
			p1 = gins(ALEAL, N, &n1);
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
		mgen(nl, &n1, res);
		regalloc(&n2, nl->type, &n1);
		gmove(&n1, &n2);
		gins(a, nr, &n2);
		gmove(&n2, res);
		regfree(&n2);
		mfree(&n1);
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
}

/*
 * generate an addressable node in res, containing the value of n.
 * n is an array index, and might be any size; res width is <= 32-bit.
 * returns Prog* to patch to panic call.
 */
static Prog*
igenindex(Node *n, Node *res, int bounded)
{
	Node tmp, lo, hi, zero;

	if(!is64(n->type)) {
		if(n->addable) {
			// nothing to do.
			*res = *n;
		} else {
			tempname(res, types[TUINT32]);
			cgen(n, res);
		}
		return nil;
	}

	tempname(&tmp, types[TINT64]);
	cgen(n, &tmp);
	split64(&tmp, &lo, &hi);
	tempname(res, types[TUINT32]);
	gmove(&lo, res);
	if(bounded) {
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
 * The generated code checks that the result is not nil.
 */
void
agen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, tmp, nlen;
	Type *t;
	uint32 w;
	uint64 v;
	Prog *p1, *p2;
	int bounded;

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
		gins(ALEAL, &n1, &n2);
		gmove(&n2, res);
		regfree(&n2);
		return;
	}
		
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
		p2 = nil;  // to be patched to panicindex.
		w = n->type->width;
		bounded = debug['B'] || n->bounded;
		if(nr->addable) {
			// Generate &nl first, and move nr into register.
			if(!isconst(nl, CTSTR))
				igen(nl, &n3, res);
			if(!isconst(nr, CTINT)) {
				p2 = igenindex(nr, &tmp, bounded);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
		} else if(nl->addable) {
			// Generate nr first, and move &nl into register.
			if(!isconst(nr, CTINT)) {
				p2 = igenindex(nr, &tmp, bounded);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
			if(!isconst(nl, CTSTR))
				igen(nl, &n3, res);
		} else {
			p2 = igenindex(nr, &tmp, bounded);
			nr = &tmp;
			if(!isconst(nl, CTSTR))
				igen(nl, &n3, res);
			regalloc(&n1, tmp.type, N);
			gins(optoas(OAS, tmp.type), &tmp, &n1);
		}

		// For fixed array we really want the pointer in n3.
		if(isfixedarray(nl->type)) {
			regalloc(&n2, types[tptr], &n3);
			agen(&n3, &n2);
			regfree(&n3);
			n3 = n2;
		}

		// &a[0] is in n3 (allocated in res)
		// i is in n1 (if not constant)
		// len(a) is in nlen (if needed)
		// w is width

		// constant index
		if(isconst(nr, CTINT)) {
			if(isconst(nl, CTSTR))
				fatal("constant string constant index");  // front end should handle
			v = mpgetfix(nr->val.u.xval);
			if(isslice(nl->type) || nl->type->etype == TSTRING) {
				if(!debug['B'] && !n->bounded) {
					nlen = n3;
					nlen.type = types[TUINT32];
					nlen.xoffset += Array_nel;
					nodconst(&n2, types[TUINT32], v);
					gins(optoas(OCMP, types[TUINT32]), &nlen, &n2);
					p1 = gbranch(optoas(OGT, types[TUINT32]), T, +1);
					ginscall(panicindex, -1);
					patch(p1, pc);
				}
			}

			// Load base pointer in n2 = n3.
			regalloc(&n2, types[tptr], &n3);
			n3.type = types[tptr];
			n3.xoffset += Array_array;
			gmove(&n3, &n2);
			regfree(&n3);
			if (v*w != 0) {
				nodconst(&n1, types[tptr], v*w);
				gins(optoas(OADD, types[tptr]), &n1, &n2);
			}
			gmove(&n2, res);
			regfree(&n2);
			break;
		}

		// i is in register n1, extend to 32 bits.
		t = types[TUINT32];
		if(issigned[n1.type->etype])
			t = types[TINT32];

		regalloc(&n2, t, &n1);			// i
		gmove(&n1, &n2);
		regfree(&n1);

		if(!debug['B'] && !n->bounded) {
			// check bounds
			t = types[TUINT32];
			if(isconst(nl, CTSTR)) {
				nodconst(&nlen, t, nl->val.u.sval->len);
			} else if(isslice(nl->type) || nl->type->etype == TSTRING) {
				nlen = n3;
				nlen.type = t;
				nlen.xoffset += Array_nel;
			} else {
				nodconst(&nlen, t, nl->type->bound);
			}
			gins(optoas(OCMP, t), &n2, &nlen);
			p1 = gbranch(optoas(OLT, t), T, +1);
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

		// Load base pointer in n3.
		regalloc(&tmp, types[tptr], &n3);
		if(isslice(nl->type) || nl->type->etype == TSTRING) {
			n3.type = types[tptr];
			n3.xoffset += Array_array;
			gmove(&n3, &tmp);
		}
		regfree(&n3);
		n3 = tmp;

		if(w == 0) {
			// nothing to do
		} else if(w == 1 || w == 2 || w == 4 || w == 8) {
			// LEAL (n3)(n2*w), n3
			p1 = gins(ALEAL, &n2, &n3);
			p1->from.scale = w;
			p1->from.type = TYPE_MEM;
			p1->from.index = p1->from.reg;
			p1->from.reg = p1->to.reg;
		} else {
			nodconst(&tmp, types[TUINT32], w);
			gins(optoas(OMUL, types[TUINT32]), &tmp, &n2);
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
		cgen_checknil(res);
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
		cgen_checknil(res);
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
 * The generated code checks that the result is not *nil.
 */
void
igen(Node *n, Node *a, Node *res)
{
	Type *fp;
	Iter flist;
	Node n1;

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
		if(n->val.u.reg != REG_SP)
			reg[n->val.u.reg]++;
		*a = *n;
		return;

	case ODOT:
		igen(n->left, a, res);
		a->xoffset += n->xoffset;
		a->type = n->type;
		return;

	case ODOTPTR:
		switch(n->left->op) {
		case ODOT:
		case ODOTPTR:
		case OCALLFUNC:
		case OCALLMETH:
		case OCALLINTER:
			// igen-able nodes.
			igen(n->left, &n1, res);
			regalloc(a, types[tptr], &n1);
			gmove(&n1, a);
			regfree(&n1);
			break;
		default:
			regalloc(a, types[tptr], res);
			cgen(n->left, a);
		}
		cgen_checknil(a);
		a->op = OINDREG;
		a->xoffset += n->xoffset;
		a->type = n->type;
		return;

	case OCALLFUNC:
	case OCALLMETH:
	case OCALLINTER:
		switch(n->op) {
		case OCALLFUNC:
			cgen_call(n, 0);
			break;
		case OCALLMETH:
			cgen_callmeth(n, 0);
			break;
		case OCALLINTER:
			cgen_callinter(n, N, 0);
			break;
		}
		fp = structfirst(&flist, getoutarg(n->left->type));
		memset(a, 0, sizeof *a);
		a->op = OINDREG;
		a->val.u.reg = REG_SP;
		a->addable = 1;
		a->xoffset = fp->width;
		a->type = n->type;
		return;

	case OINDEX:
		// Index of fixed-size array by constant can
		// put the offset in the addressing.
		// Could do the same for slice except that we need
		// to use the real index for the bounds checking.
		if(isfixedarray(n->left->type) ||
		   (isptr[n->left->type->etype] && isfixedarray(n->left->left->type)))
		if(isconst(n->right, CTINT)) {
			// Compute &a.
			if(!isptr[n->left->type->etype])
				igen(n->left, a, res);
			else {
				igen(n->left, &n1, res);
				cgen_checknil(&n1);
				regalloc(a, types[tptr], res);
				gmove(&n1, a);
				regfree(&n1);
				a->op = OINDREG;
			}

			// Compute &a[i] as &a + i*width.
			a->type = n->type;
			a->xoffset += mpgetfix(n->right->val.u.xval)*n->type->width;
			return;
		}
		break;
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
 * branch gen
 *	if(n == true) goto to;
 */
void
bgen(Node *n, int true, int likely, Prog *to)
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

	while(n->op == OCONVNOP) {
		n = n->left;
		if(n->ninit != nil)
			genlist(n->ninit);
	}

	nl = n->left;
	nr = N;

	if(nl != N && isfloat[nl->type->etype]) {
		bgen_float(n, true, likely, to);
		return;
	}

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
				yyerror("illegal slice comparison");
				break;
			}
			a = optoas(a, types[tptr]);
			igen(nl, &n1, N);
			n1.xoffset += Array_array;
			n1.type = types[tptr];
			nodconst(&tmp, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n1, &tmp);
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
			igen(nl, &n1, N);
			n1.type = types[tptr];
			nodconst(&tmp, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n1, &tmp);
			patch(gbranch(a, types[tptr], likely), to);
			regfree(&n1);
			break;
		}

		if(iscomplex[nl->type->etype]) {
			complexbool(a, nl, nr, true, likely, to);
			break;
		}

		if(is64(nr->type)) {
			if(!nl->addable || isconst(nl, CTINT)) {
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

		if(nr->ullman >= UINF) {
			if(!nl->addable) {
				tempname(&n1, nl->type);
				cgen(nl, &n1);
				nl = &n1;
			}
			if(!nr->addable) {
				tempname(&tmp, nr->type);
				cgen(nr, &tmp);
				nr = &tmp;
			}
			regalloc(&n2, nr->type, N);
			cgen(nr, &n2);
			nr = &n2;
			goto cmp;
		}

		if(!nl->addable) {
			tempname(&n1, nl->type);
			cgen(nl, &n1);
			nl = &n1;
		}

		if(smallintconst(nr)) {
			gins(optoas(OCMP, nr->type), nl, nr);
			patch(gbranch(optoas(a, nr->type), nr->type, likely), to);
			break;
		}

		if(!nr->addable) {
			tempname(&tmp, nr->type);
			cgen(nr, &tmp);
			nr = &tmp;
		}
		regalloc(&n2, nr->type, N);
		gmove(nr, &n2);
		nr = &n2;

cmp:
		gins(optoas(OCMP, nr->type), nl, nr);
		patch(gbranch(optoas(a, nr->type), nr->type, likely), to);

		if(nl->op == OREGISTER)
			regfree(nl);
		regfree(nr);
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
	NodeList *l;
	Prog *p;

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

	// If copying .args, that's all the results, so record definition sites
	// for them for the liveness analysis.
	if(res->op == ONAME && strcmp(res->sym->name, ".args") == 0)
		for(l = curfn->dcl; l != nil; l = l->next)
			if(l->n->class == PPARAMOUT)
				gvardef(l->n);

	// Avoid taking the address for simple enough types.
	if(componentgen(n, res))
		return;

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

	nodreg(&dst, types[tptr], REG_DI);
	nodreg(&src, types[tptr], REG_SI);

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

	if(res->op == ONAME)
		gvardef(res);

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
			gconreg(AADDL, w-1, REG_SI);
			gconreg(AADDL, w-1, REG_DI);

			gconreg(AMOVL, c, REG_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSB, N, N);	// MOVB *(SI)-,*(DI)-
		}

		if(q > 0) {
			if(c > 0) {
				gconreg(AADDL, -3, REG_SI);
				gconreg(AADDL, -3, REG_DI);
			} else {
				gconreg(AADDL, w-4, REG_SI);
				gconreg(AADDL, w-4, REG_DI);
			}
			gconreg(AMOVL, q, REG_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSL, N, N);	// MOVL *(SI)-,*(DI)-
		}
		// we leave with the flag clear
		gins(ACLD, N, N);
	} else {
		gins(ACLD, N, N);	// paranoia.  TODO(rsc): remove?
		// normal direction
		if(q > 128 || (q >= 4 && nacl)) {
			gconreg(AMOVL, q, REG_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSL, N, N);	// MOVL *(SI)+,*(DI)+
		} else if(q >= 4) {
			p = gins(ADUFFCOPY, N, N);
			p->to.type = TYPE_ADDR;
			p->to.sym = linksym(pkglookup("duffcopy", runtimepkg));
			// 10 and 128 = magic constants: see ../../runtime/asm_386.s
			p->to.offset = 10*(128-q);
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
 * nr is N when assigning a zero value.
 * return 1 if can do, 0 if can't.
 */
int
componentgen(Node *nr, Node *nl)
{
	Node nodl, nodr;
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
		if(nr == N || !cadable(nr))
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
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = types[simtype[TUINT]];

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_cap-Array_nel;
		nodl.type = types[simtype[TUINT]];

		if(nr != N) {
			nodr.xoffset += Array_cap-Array_nel;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
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
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = types[simtype[TUINT]];

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
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
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = ptrto(types[TUINT8]);

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
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
