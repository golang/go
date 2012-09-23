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
		if(a != AIMULB)
			goto sbop;
		cgen_bmul(n->op, nl, nr, res);
		break;

	// asymmetric binary
	case OSUB:
		a = optoas(n->op, nl->type);
		goto abop;

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

	case OLEN:
		if(istype(nl->type, TMAP) || istype(nl->type, TCHAN)) {
			// map and chan have len in the first 32-bit word.
			// a zero pointer means zero length
			regalloc(&n1, types[tptr], res);
			cgen(nl, &n1);

			nodconst(&n2, types[tptr], 0);
			gins(optoas(OCMP, types[tptr]), &n1, &n2);
			p1 = gbranch(optoas(OEQ, types[tptr]), T, 0);

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
			p1 = gbranch(optoas(OEQ, types[tptr]), T, 0);

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
		if(isfloat[n->type->etype]) {
			a = optoas(n->op, nl->type);
			goto abop;
		}
		cgen_div(n->op, nl, nr, res);
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
 * generate:
 *	res = &n;
 */
void
agen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, tmp, tmp2, n4, n5, nlen;
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

	while(n->op == OCONVNOP)
		n = n->left;

	if(isconst(n, CTNIL) && n->type->width > widthptr) {
		// Use of a nil interface or nil slice.
		// Create a temporary we can take the address of and read.
		// The generated code is just going to panic, so it need not
		// be terribly efficient. See issue 3670.
		tempname(&n1, n->type);
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

	case OSLICE:
	case OSLICEARR:
	case OSLICESTR:
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
		w = n->type->width;
		// Generate the non-addressable child first.
		if(nr->addable)
			goto irad;
		if(nl->addable) {
			if(!isconst(nr, CTINT)) {
				regalloc(&n1, nr->type, N);
				cgen(nr, &n1);
			}
			if(!isconst(nl, CTSTR)) {
				regalloc(&n3, types[tptr], res);
				if(isfixedarray(nl->type))
					agen(nl, &n3);
				else {
					igen(nl, &nlen, res);
					nlen.type = types[tptr];
					nlen.xoffset += Array_array;
					gmove(&nlen, &n3);
					nlen.type = types[TUINT32];
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
			regalloc(&n3, types[tptr], res);
			if(isfixedarray(nl->type))
				agen(nl, &n3);
			else {
				if(!nl->addable) {
					// igen will need an addressable node.
					tempname(&tmp2, nl->type);
					cgen(nl, &tmp2);
					nl = &tmp2;
				}
				igen(nl, &nlen, res);
				nlen.type = types[tptr];
				nlen.xoffset += Array_array;
				gmove(&nlen, &n3);
				nlen.type = types[TUINT32];
				nlen.xoffset += Array_nel-Array_array;
			}
		}
		if(!isconst(nr, CTINT)) {
			regalloc(&n1, nr->type, N);
			cgen(nr, &n1);
		}
		goto index;

	index:
		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// len(a) is in nlen (if needed)
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
				fatal("constant string constant index");	// front end should handle
			v = mpgetfix(nr->val.u.xval);
			if(isslice(nl->type) || nl->type->etype == TSTRING) {
				if(!debug['B'] && !n->bounded) {
					nodconst(&n2, types[TUINT32], v);
					gins(optoas(OCMP, types[TUINT32]), &nlen, &n2);
					p1 = gbranch(optoas(OGT, types[TUINT32]), T, +1);
					ginscall(panicindex, -1);
					patch(p1, pc);
				}
				regfree(&nlen);
			}

			if (v*w != 0)
				ginscon(optoas(OADD, types[tptr]), v*w, &n3);
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

		if(!debug['B'] && !n->bounded) {
			// check bounds
			t = types[TUINT32];
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
			p1->from.scale = 1;
			p1->from.index = n2.val.u.reg;
			goto indexdone;
		}

		if(w == 0) {
			// nothing to do
		} else if(w == 1 || w == 2 || w == 4 || w == 8) {
			p1 = gins(ALEAQ, &n2, &n3);
			p1->from.scale = w;
			p1->from.index = p1->from.type;
			p1->from.type = p1->to.type + D_INDIR;
		} else {
			ginscon(optoas(OMUL, t), w, &n2);
			gins(optoas(OADD, types[tptr]), &n2, &n3);
		}

	indexdone:
		gmove(&n3, res);
		regfree(&n2);
		regfree(&n3);
		if(!isconst(nl, CTSTR) && !isfixedarray(nl->type))
			regfree(&nlen);
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
		break;

	case ODOT:
		agen(nl, res);
		if(n->xoffset != 0)
			ginscon(optoas(OADD, types[tptr]), n->xoffset, res);
		break;

	case ODOTPTR:
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
			ginscon(optoas(OADD, types[tptr]), n->xoffset, res);
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
	Type *fp;
	Iter flist;
	Node n1, n2;
 
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
	
	case OINDEX:
		// Index of fixed-size array by constant can
		// put the offset in the addressing.
		// Could do the same for slice except that we need
		// to use the real index for the bounds checking.
		if(isfixedarray(n->left->type) ||
		   (isptr[n->left->type->etype] && isfixedarray(n->left->left->type)))
		if(isconst(n->right, CTINT)) {
			nodconst(&n1, types[TINT64], 0);
			n2 = *n;
			n2.right = &n1;

			regalloc(a, types[tptr], res);
			agen(&n2, a);
			a->op = OINDREG;
			a->xoffset = mpgetfix(n->right->val.u.xval)*n->type->width;
			a->type = n->type;
			return;
		}
			
	}

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
 * block copy:
 *	memmove(&ns, &n, w);
 */
void
sgen(Node *n, Node *ns, int64 w)
{
	Node nodl, nodr, oldl, oldr, cx, oldcx, tmp;
	int32 c, q, odst, osrc;

	if(debug['g']) {
		print("\nsgen w=%lld\n", w);
		dump("r", n);
		dump("res", ns);
	}

	if(n->ullman >= UINF && ns->ullman >= UINF)
		fatal("sgen UINF");

	if(w < 0)
		fatal("sgen copy %lld", w);

	if(w == 16)
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
		savex(D_SI, &nodr, &oldr, N, types[tptr]);
		agen(n, &nodr);

		regalloc(&nodr, types[tptr], &nodr);	// mark nodr as live
		savex(D_DI, &nodl, &oldl, N, types[tptr]);
		agen(ns, &nodl);
		regfree(&nodr);
	} else {
		savex(D_DI, &nodl, &oldl, N, types[tptr]);
		agen(ns, &nodl);

		regalloc(&nodl, types[tptr], &nodl);	// mark nodl as live
		savex(D_SI, &nodr, &oldr, N, types[tptr]);
		agen(n, &nodr);
		regfree(&nodl);
	}

	c = w % 8;	// bytes
	q = w / 8;	// quads

	savex(D_CX, &cx, &oldcx, N, types[TINT64]);

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


	restx(&nodl, &oldl);
	restx(&nodr, &oldr);
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
 * copy a structure component by component
 * return 1 if can do, 0 if cant.
 * nr is N for copy zero
 */
int
componentgen(Node *nr, Node *nl)
{
	Node nodl, nodr;
	int freel, freer;

	freel = 0;
	freer = 0;

	switch(nl->type->etype) {
	default:
		goto no;

	case TARRAY:
		if(!isslice(nl->type))
			goto no;
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

	switch(nl->type->etype) {
	case TARRAY:
		nodl.xoffset += Array_array;
		nodl.type = ptrto(nl->type->type);

		if(nr != N) {
			nodr.xoffset += Array_array;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = types[TUINT32];

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_cap-Array_nel;
		nodl.type = types[TUINT32];

		if(nr != N) {
			nodr.xoffset += Array_cap-Array_nel;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		goto yes;

	case TSTRING:
		nodl.xoffset += Array_array;
		nodl.type = ptrto(types[TUINT8]);

		if(nr != N) {
			nodr.xoffset += Array_array;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		nodl.xoffset += Array_nel-Array_array;
		nodl.type = types[TUINT32];

		if(nr != N) {
			nodr.xoffset += Array_nel-Array_array;
			nodr.type = nodl.type;
		} else
			nodconst(&nodr, nodl.type, 0);
		gmove(&nodr, &nodl);

		goto yes;

	case TINTER:
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
