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
	Node n1, n2;
	int a, f;
	Prog *p1, *p2, *p3;
	Addr addr;

//print("cgen %N(%d) -> %N(%d)\n", n, n->addable, res, res->addable);
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
		goto ret;
	case OEFACE:
		if (res->op != ONAME || !res->addable) {
			tempname(&n1, n->type);
			cgen_eface(n, &n1);
			cgen(&n1, res);
		} else
			cgen_eface(n, res);
		goto ret;
	}

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

		if(complexop(n, res)) {
			complexgen(n, res);
			goto ret;
		}

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

		if(!iscomplex[n->type->etype]) {
			a = optoas(OAS, res->type);
			if(sudoaddable(a, res, &addr)) {
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
		}

	gen:
		igen(res, &n1, N);
		cgen(n, &n1);
		regfree(&n1);
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

	if(complexop(n, res)) {
		complexgen(n, res);
		goto ret;
	}

	// if both are addressable, move
	if(n->addable) {
		if(n->op == OREGISTER || res->op == OREGISTER) {
			gmove(n, res);
		} else {
			regalloc(&n1, n->type, N);
			gmove(n, &n1);
			cgen(&n1, res);
			regfree(&n1);
		}
		goto ret;
	}

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

	if(!iscomplex[n->type->etype]) {
		a = optoas(OAS, n->type);
		if(sudoaddable(a, n, &addr)) {
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
	}

	// TODO(minux): we shouldn't reverse FP comparisons, but then we need to synthesize
	// OGE, OLE, and ONE ourselves.
	// if(nl != N && isfloat[n->type->etype] && isfloat[nl->type->etype]) goto flt;

	switch(n->op) {
	default:
		dump("cgen", n);
		fatal("cgen: unknown op %+hN", n);
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
		p1 = gbranch(ABR, T, 0);
		p2 = pc;
		gmove(nodbool(1), res);
		p3 = gbranch(ABR, T, 0);
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
		gmove(&n1, res);
		regfree(&n1);
		goto ret;

	case OMINUS:
		if(isfloat[nl->type->etype]) {
			nr = nodintconst(-1);
			convlit(&nr, n->type);
			a = optoas(OMUL, nl->type);
			goto sbop;
		}
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

	case OHMUL:
		cgen_hmul(nl, nr, res);
		break;

	case OCONV:
		if(n->type->width > nl->type->width) {
			// If loading from memory, do conversion during load,
			// so as to avoid use of 8-bit register in, say, int(*byteptr).
			switch(nl->op) {
			case ODOT:
			case ODOTPTR:
			case OINDEX:
			case OIND:
			case ONAME:
				igen(nl, &n1, res);
				regalloc(&n2, n->type, res);
				gmove(&n1, &n2);
				gmove(&n2, res);
				regfree(&n2);
				regfree(&n1);
				goto ret;
			}
		}

		regalloc(&n1, nl->type, res);
		regalloc(&n2, n->type, &n1);
		cgen(nl, &n1);

		// if we do the conversion n1 -> n2 here
		// reusing the register, then gmove won't
		// have to allocate its own register.
		gmove(&n1, &n2);
		gmove(&n2, res);
		regfree(&n2);
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
			p1 = gins(AMOVD, N, &n1);
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
			// map and chan have len in the first int-sized word.
			// a zero pointer means zero length
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);

			nodconst(&n2, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n1, &n2);
			p1 = gbranch(optoas(OEQ, types[tptr]), T, 0);

			n2 = n1;
			n2.op = OINDREG;
			n2.type = types[simtype[TINT]];
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
			n1.type = types[simtype[TUINT]];
			n1.xoffset += Array_nel;
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		fatal("cgen: OLEN: unknown type %lT", nl->type);
		break;

	case OCAP:
		if(istype(nl->type, TCHAN)) {
			// chan has cap in the second int-sized word.
			// a zero pointer means zero length
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);

			nodconst(&n2, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n1, &n2);
			p1 = gbranch(optoas(OEQ, types[tptr]), T, 0);

			n2 = n1;
			n2.op = OINDREG;
			n2.xoffset = widthint;
			n2.type = types[simtype[TINT]];
			gmove(&n2, &n1);

			patch(p1, pc);

			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		if(isslice(nl->type)) {
			igen(nl, &n1, res);
			n1.type = types[simtype[TUINT]];
			n1.xoffset += Array_cap;
			gmove(&n1, res);
			regfree(&n1);
			break;
		}
		fatal("cgen: OCAP: unknown type %lT", nl->type);
		break;

	case OADDR:
		if(n->bounded) // let race detector avoid nil checks
			disable_checknil++;
		agen(nl, res);
		if(n->bounded)
			disable_checknil--;
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
		if(isfloat[n->type->etype]) {
			a = optoas(n->op, nl->type);
			goto abop;
		}

		if(nl->ullman >= nr->ullman) {
			regalloc(&n1, nl->type, res);
			cgen(nl, &n1);
			cgen_div(n->op, &n1, nr, res);
			regfree(&n1);
		} else {
			if(!smallintconst(nr)) {
				regalloc(&n2, nr->type, res);
				cgen(nr, &n2);
			} else {
				n2 = *nr;
			}
			cgen_div(n->op, nl, &n2, res);
			if(n2.op != OLITERAL)
				regfree(&n2);
		}
		break;

	case OLSH:
	case ORSH:
	case OLROT:
		cgen_shift(n->op, n->bounded, nl, nr, res);
		break;
	}
	goto ret;

sbop:	// symmetric binary
	/*
	 * put simplest on right - we'll generate into left
	 * and then adjust it using the computation of right.
	 * constants and variables have the same ullman
	 * count, so look for constants specially.
	 *
	 * an integer constant we can use as an immediate
	 * is simpler than a variable - we can use the immediate
	 * in the adjustment instruction directly - so it goes
	 * on the right.
	 *
	 * other constants, like big integers or floating point
	 * constants, require a mov into a register, so those
	 * might as well go on the left, so we can reuse that
	 * register for the computation.
	 */
	if(nl->ullman < nr->ullman ||
	   (nl->ullman == nr->ullman &&
	    (smallintconst(nl) || (nr->op == OLITERAL && !smallintconst(nr))))) {
		r = nl;
		nl = nr;
		nr = r;
	}

abop:	// asymmetric binary
	if(nl->ullman >= nr->ullman) {
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
	/*
	 * This generates smaller code - it avoids a MOV - but it's
	 * easily 10% slower due to not being able to
	 * optimize/manipulate the move.
	 * To see, run: go test -bench . crypto/md5
	 * with and without.
	 *
		if(sudoaddable(a, nr, &addr)) {
			p1 = gins(a, N, &n1);
			p1->from = addr;
			gmove(&n1, res);
			sudoclean();
			regfree(&n1);
			goto ret;
		}
	 *
	 */
		// TODO(minux): enable using constants directly in certain instructions.
		//if(smallintconst(nr))
		//	n2 = *nr;
		//else {
			regalloc(&n2, nr->type, N);
			cgen(nr, &n2);
		//}
	} else {
		//if(smallintconst(nr))
		//	n2 = *nr;
		//else {
			regalloc(&n2, nr->type, res);
			cgen(nr, &n2);
		//}
		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);
	}
	gins(a, &n2, &n1);
	// Normalize result for types smaller than word.
	if(n->type->width < widthreg) {
		switch(n->op) {
		case OADD:
		case OSUB:
		case OMUL:
		case OLSH:
			gins(optoas(OAS, n->type), &n1, &n1);
			break;
		}
	}
	gmove(&n1, res);
	regfree(&n1);
	if(n2.op != OLITERAL)
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
 * allocate a register (reusing res if possible) and generate
 *  a = n
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
		regalloc(a, n->type, res);
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
 * allocate a register (reusing res if possible) and generate
 * a = &n
 * The caller must call regfree(a).
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
		//bounded = debug['B'] || n->bounded;
		if(nr->addable) {
			if(!isconst(nr, CTINT))
				tempname(&tmp, types[TINT64]);
			if(!isconst(nl, CTSTR))
				agenr(nl, &n3, res);
			if(!isconst(nr, CTINT)) {
				cgen(nr, &tmp);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
		} else if(nl->addable) {
			if(!isconst(nr, CTINT)) {
				tempname(&tmp, types[TINT64]);
				cgen(nr, &tmp);
				regalloc(&n1, tmp.type, N);
				gmove(&tmp, &n1);
			}
			if(!isconst(nl, CTSTR)) {
				agenr(nl, &n3, res);
			}
		} else {
			tempname(&tmp, types[TINT64]);
			cgen(nr, &tmp);
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
					ginscon2(optoas(OCMP, types[TUINT64]), &n4, v);
					regfree(&n4);
					p1 = gbranch(optoas(OGT, types[TUINT64]), T, +1);
					ginscall(panicindex, 0);
					patch(p1, pc);
				}

				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_array;
				gmove(&n1, &n3);
			}

			if (v*w != 0) {
				ginscon(optoas(OADD, types[tptr]), v*w, &n3);
			}
			*a = n3;
			break;
		}

		regalloc(&n2, types[TINT64], &n1);			// i
		gmove(&n1, &n2);
		regfree(&n1);

		if(!debug['B'] && !n->bounded) {
			// check bounds
			if(isconst(nl, CTSTR)) {
				nodconst(&n4, types[TUINT64], nl->val.u.sval->len);
			} else if(isslice(nl->type) || nl->type->etype == TSTRING) {
				n1 = n3;
				n1.op = OINDREG;
				n1.type = types[tptr];
				n1.xoffset = Array_nel;
				regalloc(&n4, types[TUINT64], N);
				gmove(&n1, &n4);
			} else {
				if(nl->type->bound < (1<<15)-1)
					nodconst(&n4, types[TUINT64], nl->type->bound);
				else {
					regalloc(&n4, types[TUINT64], N);
					p1 = gins(AMOVD, N, &n4);
					p1->from.type = TYPE_CONST;
					p1->from.offset = nl->type->bound;
				}
			}
			gins(optoas(OCMP, types[TUINT64]), &n2, &n4);
			if(n4.op == OREGISTER)
				regfree(&n4);
			p1 = gbranch(optoas(OLT, types[TUINT64]), T, +1);
			if(p2)
				patch(p2, pc);
			ginscall(panicindex, 0);
			patch(p1, pc);
		}
		
		if(isconst(nl, CTSTR)) {
			regalloc(&n3, types[tptr], res);
			p1 = gins(AMOVD, N, &n3);
			datastring(nl->val.u.sval->s, nl->val.u.sval->len, &p1->from);
			p1->from.type = TYPE_ADDR;
		} else if(isslice(nl->type) || nl->type->etype == TSTRING) {
			n1 = n3;
			n1.op = OINDREG;
			n1.type = types[tptr];
			n1.xoffset = Array_array;
			gmove(&n1, &n3);
		}

		if(w == 0) {
			// nothing to do
		} else if(w == 1) {
			/* w already scaled */
			gins(optoas(OADD, types[tptr]), &n2, &n3);
		} /* else if(w == 2 || w == 4 || w == 8) {
			// TODO(minux): scale using shift
		} */ else {
			regalloc(&n4, types[TUINT64], N);
			nodconst(&n1, types[TUINT64], w);
			gmove(&n1, &n4);
			gins(optoas(OMUL, types[TUINT64]), &n4, &n2);
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

static void
ginsadd(int as, vlong off, Node *dst)
{
	Node n1;

	regalloc(&n1, types[tptr], dst);
	gmove(dst, &n1);
	ginscon(as, off, &n1);
	gmove(&n1, dst);
	regfree(&n1);
}

/*
 * generate:
 *	res = &n;
 * The generated code checks that the result is not nil.
 */
void
agen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3;

	if(debug['g']) {
		dump("\nagen-res", res);
		dump("agen-r", n);
	}
	if(n == N || n->type == T)
		return;

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
		memset(&n3, 0, sizeof n3);
		n3.op = OADDR;
		n3.left = &n1;
		gins(AMOVD, &n3, &n2);
		gmove(&n2, res);
		regfree(&n2);
		goto ret;
	}
		
	if(n->addable) {
		memset(&n1, 0, sizeof n1);
		n1.op = OADDR;
		n1.left = n;
		regalloc(&n2, types[tptr], res);
		gins(AMOVD, &n1, &n2);
		gmove(&n2, res);
		regfree(&n2);
		goto ret;
	}

	nl = n->left;
	nr = n->right;
	USED(nr);

	switch(n->op) {
	default:
		fatal("agen: unknown op %+hN", n);
		break;

	case OCALLMETH:
		// TODO(minux): 5g has this: Release res so that it is available for cgen_call.
		// Pick it up again after the call for OCALLMETH and OCALLFUNC.
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
			ginsadd(optoas(OADD, types[tptr]), n->xoffset, res);
		}
		break;

	case OIND:
		cgen(nl, res);
		cgen_checknil(res);
		break;

	case ODOT:
		agen(nl, res);
		if(n->xoffset != 0) {
			ginsadd(optoas(OADD, types[tptr]), n->xoffset, res);
		}
		break;

	case ODOTPTR:
		cgen(nl, res);
		cgen_checknil(res);
		if(n->xoffset != 0) {
			ginsadd(optoas(OADD, types[tptr]), n->xoffset, res);
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
		if(n->val.u.reg != REGSP)
			reg[n->val.u.reg]++;
		*a = *n;
		return;

	case ODOT:
		igen(n->left, a, res);
		a->xoffset += n->xoffset;
		a->type = n->type;
		fixlargeoffset(a);
		return;

	case ODOTPTR:
		cgenr(n->left, a, res);
		cgen_checknil(a);
		a->op = OINDREG;
		a->xoffset += n->xoffset;
		a->type = n->type;
		fixlargeoffset(a);
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
		a->val.u.reg = REGSP;
		a->addable = 1;
		a->xoffset = fp->width + widthptr; // +widthptr: saved lr at 0(SP)
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
			fixlargeoffset(a);
			return;
		}
		break;
	}

	agenr(n, a, res);
	a->op = OINDREG;
	a->type = n->type;
}

/*
 * generate:
 *	if(n == true) goto to;
 */
void
bgen(Node *n, int true, int likely, Prog *to)
{
	int et, a;
	Node *nl, *nr, *l, *r;
	Node n1, n2, tmp;
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

	while(n->op == OCONVNOP) {
		n = n->left;
		if(n->ninit != nil)
			genlist(n->ninit);
	}

	switch(n->op) {
	default:
		regalloc(&n1, n->type, N);
		cgen(n, &n1);
		nodconst(&n2, n->type, 0);
		gins(optoas(OCMP, n->type), &n1, &n2);
		a = ABNE;
		if(!true)
			a = ABEQ;
		patch(gbranch(a, n->type, likely), to);
		regfree(&n1);
		goto ret;

	case OLITERAL:
		// need to ask if it is bool?
		if(!true == !n->val.u.bval)
			patch(gbranch(ABR, T, likely), to);
		goto ret;

	case OANDAND:
		if(!true)
			goto caseor;

	caseand:
		p1 = gbranch(ABR, T, 0);
		p2 = gbranch(ABR, T, 0);
		patch(p1, pc);
		bgen(n->left, !true, -likely, p2);
		bgen(n->right, !true, -likely, p2);
		p1 = gbranch(ABR, T, 0);
		patch(p1, to);
		patch(p2, pc);
		goto ret;

	case OOROR:
		if(!true)
			goto caseand;

	caseor:
		bgen(n->left, true, likely, to);
		bgen(n->right, true, likely, to);
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
		break;
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
			if(isfloat[nr->type->etype]) {
				// brcom is not valid on floats when NaN is involved.
				p1 = gbranch(ABR, T, 0);
				p2 = gbranch(ABR, T, 0);
				patch(p1, pc);
				ll = n->ninit;   // avoid re-genning ninit
				n->ninit = nil;
				bgen(n, 1, -likely, p2);
				n->ninit = ll;
				patch(gbranch(ABR, T, 0), to);
				patch(p2, pc);
				goto ret;
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
				yyerror("illegal slice comparison");
				break;
			}
			a = optoas(a, types[tptr]);
			igen(nl, &n1, N);
			n1.xoffset += Array_array;
			n1.type = types[tptr];
			nodconst(&tmp, types[tptr], 0);
			regalloc(&n2, types[tptr], &n1);
			gmove(&n1, &n2);
			gins(optoas(OCMP, types[tptr]), &n2, &tmp);
			regfree(&n2);
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
			regalloc(&n2, types[tptr], &n1);
			gmove(&n1, &n2);
			gins(optoas(OCMP, types[tptr]), &n2, &tmp);
			regfree(&n2);
			patch(gbranch(a, types[tptr], likely), to);
			regfree(&n1);
			break;
		}
		if(iscomplex[nl->type->etype]) {
			complexbool(a, nl, nr, true, likely, to);
			break;
		}

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

			goto cmp;
		}

		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);

		// TODO(minux): cmpi does accept 16-bit signed immediate as p->to.
		// and cmpli accepts 16-bit unsigned immediate.
		//if(smallintconst(nr)) {
		//	gins(optoas(OCMP, nr->type), &n1, nr);
		//	patch(gbranch(optoas(a, nr->type), nr->type, likely), to);
		//	regfree(&n1);
		//	break;
		//}

		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);
	cmp:
		l = &n1;
		r = &n2;
		gins(optoas(OCMP, nr->type), l, r);
		if(isfloat[nr->type->etype] && (a == OLE || a == OGE)) {
			// To get NaN right, must rewrite x <= y into separate x < y or x = y.
			switch(a) {
			case OLE:
				a = OLT;
				break;
			case OGE:
				a = OGT;
				break;
			}
			patch(gbranch(optoas(a, nr->type), nr->type, likely), to);
			patch(gbranch(optoas(OEQ, nr->type), nr->type, likely), to);			
		} else {
			patch(gbranch(optoas(a, nr->type), nr->type, likely), to);
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
int64
stkof(Node *n)
{
	Type *t;
	Iter flist;
	int64 off;

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
			return t->width + widthptr;	// +widthptr: correct for saved LR
		break;
	}

	// botch - probably failing to recognize address
	// arithmetic on the above. eg INDEX and DOT
	return -1000;
}

/*
 * block copy:
 *	memmove(&ns, &n, w);
 */
void
sgen(Node *n, Node *ns, int64 w)
{
	Node dst, src, tmp, nend;
	int32 c, odst, osrc;
	int dir, align, op;
	Prog *p, *ploop;
	NodeList *l;
	Node *res = ns;

	if(debug['g']) {
		print("\nsgen w=%lld\n", w);
		dump("r", n);
		dump("res", ns);
	}

	if(n->ullman >= UINF && ns->ullman >= UINF)
		fatal("sgen UINF");

	if(w < 0)
		fatal("sgen copy %lld", w);
	
	// If copying .args, that's all the results, so record definition sites
	// for them for the liveness analysis.
	if(ns->op == ONAME && strcmp(ns->sym->name, ".args") == 0)
		for(l = curfn->dcl; l != nil; l = l->next)
			if(l->n->class == PPARAMOUT)
				gvardef(l->n);

	// Avoid taking the address for simple enough types.
	//if(componentgen(n, ns))
	//	return;
	
	if(w == 0) {
		// evaluate side effects only.
		regalloc(&dst, types[tptr], N);
		agen(res, &dst);
		agen(n, &dst);
		regfree(&dst);
		return;
	}

	// determine alignment.
	// want to avoid unaligned access, so have to use
	// smaller operations for less aligned types.
	// for example moving [4]byte must use 4 MOVB not 1 MOVW.
	align = n->type->align;
	switch(align) {
	default:
		fatal("sgen: invalid alignment %d for %T", align, n->type);
	case 1:
		op = AMOVBU;
		break;
	case 2:
		op = AMOVHU;
		break;
	case 4:
		op = AMOVWZU; // there is no lwau, only lwaux
		break;
	case 8:
		op = AMOVDU;
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

	if(n->ullman >= res->ullman) {
		agenr(n, &dst, res);	// temporarily use dst
		regalloc(&src, types[tptr], N);
		gins(AMOVD, &dst, &src);
		if(res->op == ONAME)
			gvardef(res);
		agen(res, &dst);
	} else {
		if(res->op == ONAME)
			gvardef(res);
		agenr(res, &dst, res);
		agenr(n, &src, N);
	}

	regalloc(&tmp, types[tptr], N);

	// set up end marker
	memset(&nend, 0, sizeof nend);

	// move src and dest to the end of block if necessary
	if(dir < 0) {
		if(c >= 4) {
			regalloc(&nend, types[tptr], N);
			p = gins(AMOVD, &src, &nend);
		}

		p = gins(AADD, N, &src);
		p->from.type = TYPE_CONST;
		p->from.offset = w;

		p = gins(AADD, N, &dst);
		p->from.type = TYPE_CONST;
		p->from.offset = w;
	} else {
		p = gins(AADD, N, &src);
		p->from.type = TYPE_CONST;
		p->from.offset = -dir;

		p = gins(AADD, N, &dst);
		p->from.type = TYPE_CONST;
		p->from.offset = -dir;

		if(c >= 4) {
			regalloc(&nend, types[tptr], N);
			p = gins(AMOVD, &src, &nend);
			p->from.type = TYPE_ADDR;
			p->from.offset = w;
		}
	}


	// move
	// TODO: enable duffcopy for larger copies.
	if(c >= 4) {
		p = gins(op, &src, &tmp);
		p->from.type = TYPE_MEM;
		p->from.offset = dir;
		ploop = p;

		p = gins(op, &tmp, &dst);
		p->to.type = TYPE_MEM;
		p->to.offset = dir;

		p = gins(ACMP, &src, &nend);

		patch(gbranch(ABNE, T, 0), ploop);
 		regfree(&nend);
	} else {
		// TODO(austin): Instead of generating ADD $-8,R8; ADD
		// $-8,R7; n*(MOVDU 8(R8),R9; MOVDU R9,8(R7);) just
		// generate the offsets directly and eliminate the
		// ADDs.  That will produce shorter, more
		// pipeline-able code.
		while(c-- > 0) {
			p = gins(op, &src, &tmp);
			p->from.type = TYPE_MEM;
			p->from.offset = dir;
	
			p = gins(op, &tmp, &dst);
			p->to.type = TYPE_MEM;
			p->to.offset = dir;
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
