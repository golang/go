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

	if(n->addable) {
		gmove(n, res);
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
		p1 = gbranch(AJMP, T, 0);
		p2 = pc;
		gmove(nodbool(1), res);
		p3 = gbranch(AJMP, T, 0);
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
			p1 = gins(ALEAQ, N, &n1);
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

		if(smallintconst(nr))
			n2 = *nr;
		else {
			regalloc(&n2, nr->type, N);
			cgen(nr, &n2);
		}
	} else {
		if(smallintconst(nr))
			n2 = *nr;
		else {
			regalloc(&n2, nr->type, res);
			cgen(nr, &n2);
		}
		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);
	}
	gins(a, &n2, &n1);
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
	Node n1, n2, n3, n5, tmp, tmp2, nlen;
	Prog *p1;
	Type *t;
	uint64 w;
	uint64 v;
	int freelen;

	if(debug['g']) {
		dump("\nagenr-n", n);
	}

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
		freelen = 0;
		w = n->type->width;
		// Generate the non-addressable child first.
		if(nr->addable)
			goto irad;
		if(nl->addable) {
			cgenr(nr, &n1, N);
			if(!isconst(nl, CTSTR)) {
				if(isfixedarray(nl->type)) {
					agenr(nl, &n3, res);
				} else {
					igen(nl, &nlen, res);
					freelen = 1;
					nlen.type = types[tptr];
					nlen.xoffset += Array_array;
					regalloc(&n3, types[tptr], res);
					gmove(&nlen, &n3);
					nlen.type = types[simtype[TUINT]];
					nlen.xoffset += Array_nel-Array_array;
				}
			}
			goto index;
		}
		tempname(&tmp, nr->type);
		cgen(nr, &tmp);
		nr = &tmp;
	irad:
		if(!isconst(nl, CTSTR)) {
			if(isfixedarray(nl->type)) {
				agenr(nl, &n3, res);
			} else {
				if(!nl->addable) {
					// igen will need an addressable node.
					tempname(&tmp2, nl->type);
					cgen(nl, &tmp2);
					nl = &tmp2;
				}
				igen(nl, &nlen, res);
				freelen = 1;
				nlen.type = types[tptr];
				nlen.xoffset += Array_array;
				regalloc(&n3, types[tptr], res);
				gmove(&nlen, &n3);
				nlen.type = types[simtype[TUINT]];
				nlen.xoffset += Array_nel-Array_array;
			}
		}
		if(!isconst(nr, CTINT)) {
			cgenr(nr, &n1, N);
		}
		goto index;

	index:
		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// len(a) is in nlen (if needed)
		// w is width

		// constant index
		if(isconst(nr, CTINT)) {
			if(isconst(nl, CTSTR))
				fatal("constant string constant index");	// front end should handle
			v = mpgetfix(nr->val.u.xval);
			if(isslice(nl->type) || nl->type->etype == TSTRING) {
				if(!debug['B'] && !n->bounded) {
					nodconst(&n2, types[simtype[TUINT]], v);
					if(smallintconst(nr)) {
						gins(optoas(OCMP, types[simtype[TUINT]]), &nlen, &n2);
					} else {
						regalloc(&tmp, types[simtype[TUINT]], N);
						gmove(&n2, &tmp);
						gins(optoas(OCMP, types[simtype[TUINT]]), &nlen, &tmp);
						regfree(&tmp);
					}
					p1 = gbranch(optoas(OGT, types[simtype[TUINT]]), T, +1);
					ginscall(panicindex, -1);
					patch(p1, pc);
				}
				regfree(&nlen);
			}

			if (v*w != 0)
				ginscon(optoas(OADD, types[tptr]), v*w, &n3);
			*a = n3;
			break;
		}

		// type of the index
		t = types[TUINT64];
		if(issigned[n1.type->etype])
			t = types[TINT64];

		regalloc(&n2, t, &n1);			// i
		gmove(&n1, &n2);
		regfree(&n1);

		if(!debug['B'] && !n->bounded) {
			// check bounds
			t = types[simtype[TUINT]];
			if(is64(nr->type))
				t = types[TUINT64];
			if(isconst(nl, CTSTR)) {
				nodconst(&nlen, t, nl->val.u.sval->len);
			} else if(isslice(nl->type) || nl->type->etype == TSTRING) {
				if(is64(nr->type)) {
					regalloc(&n5, t, N);
					gmove(&nlen, &n5);
					regfree(&nlen);
					nlen = n5;
				}
			} else {
				nodconst(&nlen, t, nl->type->bound);
				if(!smallintconst(&nlen)) {
					regalloc(&n5, t, N);
					gmove(&nlen, &n5);
					nlen = n5;
					freelen = 1;
				}
			}
			gins(optoas(OCMP, t), &n2, &nlen);
			p1 = gbranch(optoas(OLT, t), T, +1);
			ginscall(panicindex, -1);
			patch(p1, pc);
		}

		if(isconst(nl, CTSTR)) {
			regalloc(&n3, types[tptr], res);
			p1 = gins(ALEAQ, N, &n3);
			datastring(nl->val.u.sval->s, nl->val.u.sval->len, &p1->from);
			gins(AADDQ, &n2, &n3);
			goto indexdone;
		}

		if(w == 0) {
			// nothing to do
		} else if(w == 1 || w == 2 || w == 4 || w == 8) {
			p1 = gins(ALEAQ, &n2, &n3);
			p1->from.type = TYPE_MEM;
			p1->from.scale = w;
			p1->from.index = p1->from.reg;
			p1->from.reg = p1->to.reg;
		} else {
			ginscon(optoas(OMUL, t), w, &n2);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
		}

	indexdone:
		*a = n3;
		regfree(&n2);
		if(freelen)
			regfree(&nlen);
		break;

	default:
		regalloc(a, types[tptr], res);
		agen(n, a);
		break;
	}
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
	Node n1, n2;

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
		gins(ALEAQ, &n1, &n2);
		gmove(&n2, res);
		regfree(&n2);
		goto ret;
	}
		
	if(n->addable) {
		regalloc(&n1, types[tptr], res);
		gins(ALEAQ, n, &n1);
		gmove(&n1, res);
		regfree(&n1);
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
		if(n->xoffset != 0)
			ginscon(optoas(OADD, types[tptr]), n->xoffset, res);
		break;

	case OIND:
		cgen(nl, res);
		cgen_checknil(res);
		break;

	case ODOT:
		agen(nl, res);
		if(n->xoffset != 0)
			ginscon(optoas(OADD, types[tptr]), n->xoffset, res);
		break;

	case ODOTPTR:
		cgen(nl, res);
		cgen_checknil(res);
		if(n->xoffset != 0)
			ginscon(optoas(OADD, types[tptr]), n->xoffset, res);
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
		if(n->val.u.reg != REG_SP)
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
		goto ret;

	case OLITERAL:
		// need to ask if it is bool?
		if(!true == !n->val.u.bval)
			patch(gbranch(AJMP, T, likely), to);
		goto ret;

	case ONAME:
		if(n->addable == 0)
			goto def;
		nodconst(&n1, n->type, 0);
		gins(optoas(OCMP, n->type), n, &n1);
		a = AJNE;
		if(!true)
			a = AJEQ;
		patch(gbranch(a, n->type, likely), to);
		goto ret;

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
				p1 = gbranch(AJMP, T, 0);
				p2 = gbranch(AJMP, T, 0);
				patch(p1, pc);
				ll = n->ninit;   // avoid re-genning ninit
				n->ninit = nil;
				bgen(n, 1, -likely, p2);
				n->ninit = ll;
				patch(gbranch(AJMP, T, 0), to);
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

		if(smallintconst(nr)) {
			gins(optoas(OCMP, nr->type), &n1, nr);
			patch(gbranch(optoas(a, nr->type), nr->type, likely), to);
			regfree(&n1);
			break;
		}

		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);
	cmp:
		// only < and <= work right with NaN; reverse if needed
		l = &n1;
		r = &n2;
		if(isfloat[nl->type->etype] && (a == OGT || a == OGE)) {
			l = &n2;
			r = &n1;
			a = brrev(a);
		}

		gins(optoas(OCMP, nr->type), l, r);

		if(isfloat[nr->type->etype] && (n->op == OEQ || n->op == ONE)) {
			if(n->op == OEQ) {
				// neither NE nor P
				p1 = gbranch(AJNE, T, -likely);
				p2 = gbranch(AJPS, T, -likely);
				patch(gbranch(AJMP, T, 0), to);
				patch(p1, pc);
				patch(p2, pc);
			} else {
				// either NE or P
				patch(gbranch(AJNE, T, likely), to);
				patch(gbranch(AJPS, T, likely), to);
			}
		} else
			patch(gbranch(optoas(a, nr->type), nr->type, likely), to);
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
			return t->width;
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
	Node nodl, nodr, nodsi, noddi, cx, oldcx, tmp;
	vlong c, q, odst, osrc;
	NodeList *l;
	Prog *p;

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
	if(componentgen(n, ns))
		return;
	
	if(w == 0) {
		// evaluate side effects only
		regalloc(&nodr, types[tptr], N);
		agen(ns, &nodr);
		agen(n, &nodr);
		regfree(&nodr);
		return;
	}

	// offset on the stack
	osrc = stkof(n);
	odst = stkof(ns);

	if(osrc != -1000 && odst != -1000 && (osrc == 1000 || odst == 1000)) {
		// osrc and odst both on stack, and at least one is in
		// an unknown position.  Could generate code to test
		// for forward/backward copy, but instead just copy
		// to a temporary location first.
		tempname(&tmp, n->type);
		sgen(n, &tmp, w);
		sgen(&tmp, ns, w);
		return;
	}

	if(n->ullman >= ns->ullman) {
		agenr(n, &nodr, N);
		if(ns->op == ONAME)
			gvardef(ns);
		agenr(ns, &nodl, N);
	} else {
		if(ns->op == ONAME)
			gvardef(ns);
		agenr(ns, &nodl, N);
		agenr(n, &nodr, N);
	}
	
	nodreg(&noddi, types[tptr], REG_DI);
	nodreg(&nodsi, types[tptr], REG_SI);
	gmove(&nodl, &noddi);
	gmove(&nodr, &nodsi);
	regfree(&nodl);
	regfree(&nodr);

	c = w % 8;	// bytes
	q = w / 8;	// quads

	savex(REG_CX, &cx, &oldcx, N, types[TINT64]);

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if(osrc < odst && odst < osrc+w) {
		// reverse direction
		gins(ASTD, N, N);		// set direction flag
		if(c > 0) {
			gconreg(addptr, w-1, REG_SI);
			gconreg(addptr, w-1, REG_DI);

			gconreg(movptr, c, REG_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSB, N, N);	// MOVB *(SI)-,*(DI)-
		}

		if(q > 0) {
			if(c > 0) {
				gconreg(addptr, -7, REG_SI);
				gconreg(addptr, -7, REG_DI);
			} else {
				gconreg(addptr, w-8, REG_SI);
				gconreg(addptr, w-8, REG_DI);
			}
			gconreg(movptr, q, REG_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSQ, N, N);	// MOVQ *(SI)-,*(DI)-
		}
		// we leave with the flag clear
		gins(ACLD, N, N);
	} else {
		// normal direction
		if(q > 128 || (nacl && q >= 4)) {
			gconreg(movptr, q, REG_CX);
			gins(AREP, N, N);	// repeat
			gins(AMOVSQ, N, N);	// MOVQ *(SI)+,*(DI)+
		} else if (q >= 4) {
			p = gins(ADUFFCOPY, N, N);
			p->to.type = TYPE_ADDR;
			p->to.sym = linksym(pkglookup("duffcopy", runtimepkg));
			// 14 and 128 = magic constants: see ../../runtime/asm_amd64.s
			p->to.offset = 14*(128-q);
		} else
		while(q > 0) {
			gins(AMOVSQ, N, N);	// MOVQ *(SI)+,*(DI)+
			q--;
		}
		// copy the remaining c bytes
		if(w < 4 || c <= 1 || (odst < osrc && osrc < odst+w)) {
			while(c > 0) {
				gins(AMOVSB, N, N);	// MOVB *(SI)+,*(DI)+
				c--;
			}
		} else if(w < 8 || c <= 4) {
			nodsi.op = OINDREG;
			noddi.op = OINDREG;
			nodsi.type = types[TINT32];
			noddi.type = types[TINT32];
			if(c > 4) {
				nodsi.xoffset = 0;
				noddi.xoffset = 0;
				gmove(&nodsi, &noddi);
			}
			nodsi.xoffset = c-4;
			noddi.xoffset = c-4;
			gmove(&nodsi, &noddi);
		} else {
			nodsi.op = OINDREG;
			noddi.op = OINDREG;
			nodsi.type = types[TINT64];
			noddi.type = types[TINT64];
			nodsi.xoffset = c-8;
			noddi.xoffset = c-8;
			gmove(&nodsi, &noddi);
		}
	}

	restx(&cx, &oldcx);
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
