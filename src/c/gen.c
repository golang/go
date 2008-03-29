// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

#undef	EXTERN
#define	EXTERN
#include "gen.h"

static	Node*	curfn;

void
compile(Node *fn)
{
	Plist *pl;

	if(fn->nbody == N)
		return;
	if(nerrors != 0) {
		walk(fn);
		return;
	}

	if(debug['w'])
		dump("--- pre walk ---", fn->nbody);
	walk(fn);
	if(nerrors != 0)
		return;
	if(debug['w'])
		dump("--- post walk ---", fn->nbody);

	curfn = fn;

	continpc = P;
	breakpc = P;

	pc = mal(sizeof(*pc));
	firstpc = pc;
	pc->op = PEND;
	pc->addr.type = ANONE;
	pc->loc = 1;
	inarggen();
	gen(curfn->nbody);

	if(curfn->type->outtuple != 0)
		gopcodet(PPANIC, N, N);

	if(debug['p'])
		proglist();

	pl = mal(sizeof(*pl));
	pl->name = curfn->nname;
	pl->locals = autodcl;
	pl->firstpc = firstpc;

	if(plist == nil)
		plist = pl;
	else
		plast->link = pl;
	plast = pl;

	if(debug['f'])
		frame(0);
}

/*
 * compile statements
 */
void
gen(Node *n)
{
	long lno;
	Prog *scontin, *sbreak;
	Prog *p1, *p2, *p3;
	Sym *s;

	lno = dynlineno;

loop:
	if(n == N)
		goto ret;
	dynlineno = n->lineno;	// for diagnostics

	switch(n->op) {
	default:
		dump("gen: unknown op", n);
		break;

	case ODCLTYPE:
		break;

	case OLIST:
		gen(n->left);
		n = n->right;
		goto loop;

	case OPANIC:
	case OPRINT:
		genprint(n->left);
		if(n->op == OPANIC)
			gopcodet(PPANIC, N, N);
		break;

	case OCASE:
	case OFALL:
	case OXCASE:
	case OXFALL:
	case OEMPTY:
		break;

	case OLABEL:
		// before declaration, s->label points at
		// a link list of PXGOTO instructions.
		// after declaration, s->label points
		// at a PGOTO to .+1

		s = n->left->sym;
		p1 = (Prog*)s->label;

		if(p1 != P) {
			if(p1->op == PGOTO) {
				yyerror("label redeclared: %S", s);
				break;
			}
			while(p1 != P) {
				if(p1->op != PGOTOX)
					fatal("bad label pointer: %S", s);
				p2 = p1->addr.branch;
				p1->addr.branch = pc;
				p1->op = PGOTO;
				p1 = p2;
			}
		}

		s->label = pc;
		p1 = gbranch(PGOTO, N);
		patch(p1, pc);
		break;

	case OGOTO:
		s = n->left->sym;
		p1 = (Prog*)s->label;
		if(p1 != P && p1->op == PGOTO) {
			// already declared
			p2 = gbranch(PGOTO, N);
			patch(p2, p1->addr.branch);
			break;
		}

		// not declaraed yet
		p2 = gbranch(PGOTOX, N);
		p2->addr.node = n;	// info for diagnostic if never declared
		patch(p2, p1);
		s->label = p2;
		break;

	case OBREAK:
		if(breakpc == P) {
			yyerror("gen: break is not in a loop");
			break;
		}
		patch(gbranch(PGOTO, N), breakpc);
		break;

	case OCONTINUE:
		if(continpc == P) {
			yyerror("gen: continue is not in a loop");
			break;
		}
		patch(gbranch(PGOTO, N), continpc);
		break;

	case OFOR:
		gen(n->ninit);				// 		init
		p1 = gbranch(PGOTO, N);			// 		goto test
		sbreak = breakpc;
		breakpc = gbranch(PGOTO, N);		// break:	goto done
		scontin = continpc;
		continpc = pc;
		gen(n->nincr);				// contin:	incr
		patch(p1, pc);				// test:
		bgen(n->ntest, 0, breakpc);		//		if(!test) goto break
		gen(n->nbody);				//		body
		patch(gbranch(PGOTO, N), continpc);	//		goto contin
		patch(breakpc, pc);			// done:
		continpc = scontin;
		breakpc = sbreak;
		break;

	case OIF:
		gen(n->ninit);				//		init
		p1 = gbranch(PGOTO, N);			//		goto test
		p2 = gbranch(PGOTO, N);			// p2:		goto else
		patch(p1, pc);				// test:
		bgen(n->ntest, 0, p2);			// 		if(!test) goto p2
		gen(n->nbody);				//		then
		p3 = gbranch(PGOTO, N);			//		goto done
		patch(p2, pc);				// else:
		gen(n->nelse);				//		else
		patch(p3, pc);				// done:
		break;

	case OSWITCH:
		gen(n->ninit);				// 		init
		p1 = gbranch(PGOTO, N);			// 		goto test
		sbreak = breakpc;
		breakpc = gbranch(PGOTO, N);		// break:	goto done
		patch(p1, pc);				// test:
		swgen(n);				//		switch(test) body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		break;

	case OASOP:
		cgen_asop(n->left, n->right, n->kaka);
		break;

	case ODCLVAR:
	case OCOLAS:
	case OAS:
		cgen_as(n->left, n->right, n->op, n->kaka);
		break;

	case OCALL:
	case OCALLPTR:
	case OCALLMETH:
	case OCALLINTER:
		cgen_call(n, 1);
		break;

	case ORETURN:
		cgen_ret(n);
		break;
	}

ret:
	dynlineno = lno;
}

/*
 * compile expression to (unnamed) reg
 */
void
cgen(Node *n)
{
	long lno;
	Node *nl, *nr, *r;
	int a;
	Prog *p1, *p2, *p3;

	if(n == N)
		return;

	lno = dynlineno;
	if(n->op != ONAME)
		dynlineno = n->lineno;	// for diagnostics

	nl = n->left;
	nr = n->right;

	if(nr != N && nr->ullman >= UINF && nl != N && nl->ullman >= UINF) {
		cgen(nr);
		r = tempname(n->type);
		gopcodet(PSTORE, n->type, r);
		nr = r;
	}

	switch(n->op) {
	default:
		yyerror("cgen: unknown op %O", n->op);
		break;

	case ONAME:
	case OLITERAL:
		gopcodet(PLOAD, n->type, n);
		break;

	case ONEW:
		gopcodet(PNEW, n->type, n);
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
		p1 = gbranch(PGOTO, N);
		p2 = gopcodet(PLOAD, n->type, booltrue);
		p3 = gbranch(PGOTO, N);
		patch(p1, pc);
		bgen(n, 1, p2);
		p2 = gopcodet(PLOAD, n->type, boolfalse);
		patch(p3, pc);
		goto ret;

	case OPLUS:
		cgen(nl);
		goto ret;

	// unary
	case OMINUS:
	case OCOM:
		a = optopop(n->op);
		goto uop;

	// symmetric binary
	case OAND:
	case OOR:
	case OXOR:
	case OADD:
	case OMUL:
		a = optopop(n->op);
		goto sbop;

	// asymmetric binary
	case OMOD:
	case OSUB:
	case ODIV:
	case OLSH:
	case ORSH:
	case OCAT:
		a = optopop(n->op);
		goto abop;

	case OCONV:
		if(isbytearray(nl->type)) {
			if(nl->type->etype == TPTR)
				cgen(nl);
			else
				agen(nl);
			gopcode(PCONV, PTNIL, nod(OCONV, n->type, nl->type));
			break;
		}

		cgen(nl);
		gopcode(PCONV, PTNIL, nod(OCONV, n->type, nl->type));
		break;

	case OINDEXSTR:
		nl = n->left;
		nr = n->right;
		if(nl->addable) {
			cgen(nr);
			gopcodet(PINDEXZ, nr->type, nl);
			break;
		}
		cgen(nl);
		r = tempname(nl->type);
		gopcodet(PSTORE, nl->type, r);
		cgen(nr);
		gopcodet(PINDEXZ, nr->type, r);
		break;

	case OSLICE:
		nl = n->left;	// name
		nr = n->right;

		r = nr->right;	// index2
		if(!r->addable) {
			cgen(r);
			r = tempname(r->type);
			gopcodet(PSTORE, r->type, r);
		}

		// string into PTADDR
		if(!nl->addable) {
			cgen(nl);
			gconv(PTADDR, nl->type->etype);
		} else
			gopcode(PLOAD, PTADDR, nl);

		// offset in int reg
		cgen(nr->left);

		// index 2 addressed
		gopcodet(PSLICE, r->type, r);
		break;

	case OINDEXPTR:
	case OINDEX:
	case ODOT:
	case ODOTPTR:
	case OIND:
		agen(n);
		gopcodet(PLOADI, n->type, N);
		break;

	case OLEN:
		cgen(nl);
		gopcodet(PLEN, nl->type, nl);
		break;

	case ODOTMETH:
	case ODOTINTER:
		cgen(n->left);
		break;

	case OADDR:
		agen(nl);
		gconv(PTPTR, PTADDR);
		break;

	case OCALL:
	case OCALLPTR:
	case OCALLMETH:
	case OCALLINTER:
		cgen_call(n, 0);
		cgen_callret(n, N);
		break;
	}
	goto ret;

sbop:	// symmetric
	if(nl->ullman < nr->ullman) {
		r = nl;
		nl = nr;
		nr = r;
	}

abop:	// asymmetric
	if(nr->addable) {
		cgen(nl);
		gopcodet(a, n->type, nr);
		goto ret;
	}

	cgen(nr);
	r = tempname(n->type);
	gopcodet(PSTORE, n->type, r);
	cgen(nl);
	gopcodet(a, n->type, r);
	goto ret;

uop:	// unary
	cgen(nl);
	gopcodet(a, n->type, N);
	goto ret;

ret:
	dynlineno = lno;
}

/*
 * compile the address of a value
 */
void
agen(Node *n)
{
	Node *nl, *nr;
	Node *t, *r;

	if(n == N || n->type == N)
		return;
	switch(n->op) {
	default:
		dump("agen: unknown op", n);
		break;

	case ONAME:
		gopcode(PADDR, PTADDR, n);
		break;

	case OINDEXPTR:
		nl = n->left;
		nr = n->right;
		if(nl->addable) {
			cgen(nr);
			gopcode(PLOAD, PTADDR, nl);
			genindex(n);
			break;
		}
		if(nr->addable) {
			cgen(nl);
			gconv(PTADDR, nl->type->etype);
			cgen(nr);
			genindex(n);
			break;
		}
		cgen(nr);
		r = tempname(n->type);
		gopcodet(PSTORE, n->type, r);
		cgen(nl);
		gconv(PTADDR, nl->type->etype);
		cgen(r);
		genindex(n);
		break;

	case OINDEX:
		nl = n->left;
		nr = n->right;
		if(nl->addable) {
			cgen(nr);
			agen(nl);
			genindex(n);
			break;
		}
		if(nr->addable) {
			agen(nl);
			cgen(nr);
			genindex(n);
			break;
		}
		cgen(nr);
		r = tempname(n->type);
		gopcodet(PSTORE, n->type, r);
		agen(nl);
		cgen(r);
		genindex(n);
		break;

	case OIND:
		nl = n->left;
		if(nl->addable) {
			gopcode(PLOAD, PTADDR, nl);
			break;
		}
		cgen(nl);
		gconv(PTADDR, nl->type->etype);
		break;
		
	case ODOT:
	case ODOTPTR:
		nl = n->left;
		nr = n->right;
		t = nl->type;
		switch(t->etype) {
		default:
			badtype(n->op, n->left->type, n->right->type);
			break;

		case TPTR:
			if(nl->op != ONAME) {
				cgen(nl);
				gconv(PTADDR, nl->type->etype);
			} else
				gopcode(PLOAD, PTADDR, nl);
			gaddoffset(nr);
			break;

		case TSTRUCT:
			agen(nl);
			gaddoffset(nr);
			break;
		}
		break;
	}
}

/*
 * compile boolean expression
 * true is branch-true or branch-false
 * to is where to branch
 */
void
bgen(Node *n, int true, Prog *to)
{
	long lno;
	int et, a;
	Node *nl, *nr, *r;
	Prog *p1, *p2;

	if(n == N)
		n = booltrue;

	lno = dynlineno;
	if(n->op != ONAME)
		dynlineno = n->lineno;	// for diagnostics

	if(n == N)
		goto ret;
	if(n->type == N) {
		convlit(n, types[TBOOL]);
		if(n->type == N)
			goto ret;
	}

	et = n->type->etype;
	if(et != TBOOL) {
		yyerror("cgen: bad type %T for %O", n->type, n->op);
		patch(gbranch(PERROR, N), to);
		goto ret;
	}
	nl = N;
	nr = N;

	switch(n->op) {
	default:
		cgen(n);
		gopcodet(PTEST, n->type, N);
		a = PBTRUE;
		if(!true)
			a = PBFALSE;
		patch(gbranch(a, n->type), to);
		goto ret;

	case OLITERAL:
		if(!true == !n->val.vval)
			patch(gbranch(PGOTO, N), to);
		goto ret;

	case ONAME:
		gopcodet(PTEST, n->type, n);
		a = PBTRUE;
		if(!true)
			a = PBFALSE;
		patch(gbranch(a, n->type), to);
		goto ret;

	case OANDAND:
		if(!true)
			goto caseor;

	caseand:
		p1 = gbranch(PGOTO, N);
		p2 = gbranch(PGOTO, N);
		patch(p1, pc);
		bgen(n->left, !true, p2);
		bgen(n->right, !true, p2);
		p1 = gbranch(PGOTO, N);
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
		if(nr == N || nr->type == N)
			goto ret;

	case ONOT:	// unary
		nl = n->left;
		if(nl == N || nl->type == N)
			goto ret;
	}

	switch(n->op) {

	case ONOT:
		bgen(nl, !true, to);
		goto ret;

	case OEQ: a = PBEQ; goto br;
	case ONE: a = PBNE; goto br;
	case OLT: a = PBLT; goto br;
	case OGT: a = PBGT; goto br;
	case OLE: a = PBLE; goto br;
	case OGE: a = PBGE; goto br;
	br:
		if(!true)
			a = brcom(a);

		// make simplest on right
		if(nl->ullman < nr->ullman) {
			a = brrev(a);
			r = nl;
			nl = nr;
			nr = r;
		}

		if(nr->addable) {
			cgen(nl);
			gopcodet(PCMP, nr->type, nr);
			patch(gbranch(a, nr->type), to);
			break;
		}
		cgen(nr);
		r = tempname(nr->type);
		gopcodet(PSTORE, nr->type, r);
		cgen(nl);
		gopcodet(PCMP, nr->type, r);
		patch(gbranch(a, nr->type), to);
		break;
	}
	goto ret;

ret:
	dynlineno = lno;
}

void
swgen(Node *n)
{
	Node *c1, *c2;
	Case *s0, *se, *s;
	Prog *p1, *dflt;
	long lno;
	int any;
	Iter save1, save2;

	lno = dynlineno;

	p1 = gbranch(PGOTO, N);
	s0 = C;
	se = C;

	// walk thru the body placing breaks
	// and labels into the case statements

	any = 0;
	dflt = P;
	c1 = listfirst(&save1, &n->nbody);
	while(c1 != N) {
		dynlineno = c1->lineno;	// for diagnostics
		if(c1->op != OCASE) {
			if(s0 == C)
				yyerror("unreachable statements in a switch");
			gen(c1);

			any = 1;
			if(c1->op == OFALL)
				any = 0;
			c1 = listnext(&save1);
			continue;
		}

		// put in the break between cases
		if(any) {
			patch(gbranch(PGOTO, N), breakpc);
			any = 0;
		}

		// over case expressions
		c2 = listfirst(&save2, &c1->left);
		if(c2 == N)
			dflt = pc;

		while(c2 != N) {

			s = mal(sizeof(*s));
			if(s0 == C)
				s0 = s;
			else
				se->slink = s;
			se = s;

			s->scase = c2;		// case expression
			s->sprog = pc;		// where to go

			c2 = listnext(&save2);
		}

		c1 = listnext(&save1);
	}

	if(any)
		patch(gbranch(PGOTO, N), breakpc);

	patch(p1, pc);
	c1 = tempname(n->ntest->type);
	cgen(n->ntest);
	gopcodet(PSTORE, n->ntest->type, c1);

	for(s=s0; s!=C; s=s->slink) {
		cgen(s->scase);
		gopcodet(PCMP, n->ntest->type, c1);
		patch(gbranch(PBEQ, n->ntest->type), s->sprog);
	}
	if(dflt != P) {
		patch(gbranch(PGOTO, N), dflt);
		goto ret;
	}
	patch(gbranch(PGOTO, N), breakpc);

ret:
	dynlineno = lno;
}

/*
 * does this tree use
 * the pointer register
 */
int
usesptr(Node *n)
{
//	if(n->addable)
//		return 0;
	return 1;
}

void
cgen_as(Node *nl, Node *nr, int op, int kaka)
{
	Node *r;

loop:
	switch(op) {
	default:
		fatal("cgen_as: unknown op %O", op);

	case ODCLVAR:
		if(nr == N && nl->op == OLIST) {
			kaka = PAS_SINGLE;
			cgen_as(nl->left, nr, op, kaka);
			nl = nl->right;
			goto loop;
		}

	case OCOLAS:
	case OAS:
		switch(kaka) {
		default:
			yyerror("cgen_as: unknown param %d %d", kaka, PAS_CALLM);
			break;

		case PAS_CALLM: // function returning multi values
			cgen_call(nr, 0);
			cgen_callret(nr, nl);
			break;

		case PAS_SINGLE: // single return val used in expr
			if(nr == N) {
				if(nl->addable) {
					gopcodet(PSTOREZ, nl->type, nl);
					break;
				}
				agen(nl);
				gopcodet(PSTOREZIP, nl->type, N);
				break;
			}

			if(nl->addable) {
				cgen(nr);
				genconv(nl->type, nr->type);
				gopcodet(PSTORE, nl->type, nl);
				break;
			}

			if(nr->addable && !needconvert(nl->type, nr->type)) {
				agen(nl);
				gopcodet(PSTOREI, nr->type, nr);
				break;
			}
			if(!usesptr(nr)) {
				cgen(nr);
				genconv(nl->type, nr->type);
				agen(nl);
				gopcodet(PSTOREI, nr->type, N);
				break;
			}
			agen(nl);
			r = tempname(ptrto(nl->type));
			gopcode(PSTORE, PTADDR, r);
			cgen(nr);
			genconv(nl->type, nr->type);
			gopcode(PLOAD, PTADDR, r);
			gopcodet(PSTOREI, nl->type, N);
			break;

		case PAS_STRUCT: // structure assignment
			r = ptrto(nr->type);
			if(!usesptr(nr)) {
				agen(nr);
				agen(nl);
				gopcodet(PLOAD, N, r);
				gopcodet(PERROR, nr->type, N);
				break;
			}
			r = tempname(r);
			agen(nr);
			gopcode(PSTORE, PTADDR, r);

			agen(nl);
			gopcodet(PERROR, nr->type, r);
			break;
		}
		break;
	}
}

void
cgen_asop(Node *nl, Node *nr, int op)
{
	Node *r;
	int a;

	a = optopop(op);
	if(nr->addable) {
		if(nl->addable) {
			gopcodet(PLOAD, nl->type, nl);
			gopcodet(a, nr->type, nr);
			gopcodet(PSTORE, nl->type, nl);
			return;
		}

		agen(nl);
		gopcodet(PLOADI, nl->type, N);
		gopcodet(a, nr->type, nr);
		gopcodet(PSTOREI, nl->type, N);
		return;
	}

	r = tempname(nr->type);
	cgen(nr);
	gopcodet(PSTORE, nr->type, r);

	agen(nl);
	gopcodet(PLOADI, nl->type, N);
	gopcodet(a, nr->type, r);
	gopcodet(PSTOREI, nl->type, N);
}

void
inarggen(void)
{
	Iter save;
	Node *arg, *t;
	int i;

	t = curfn->type;

	arg = structfirst(&save, getthis(t));
	if(arg != N) {
		fnparam(t, 0, 0);
		gopcodet(PSTORE, arg->type, arg->nname);
	}

	i = 0;
	arg = structfirst(&save, getinarg(t));
	while(arg != N) {
		fnparam(t, 2, i);
		gopcodet(PLOADI, arg->type, arg->nname);

		arg = structnext(&save);
		i++;
	}
}

void
cgen_ret(Node *n)
{
	Node *arg, *a, *f;
	Iter save;

	arg = listfirst(&save, &n->left);	// expr list
	a = getoutargx(curfn->type);
	f = a->type;
	for(;;) {
		if(arg == N)
			break;
		if(f->etype != TFIELD)
			fatal("cgen_ret: not field");
		if(arg->addable && !needconvert(f->type, arg->type)) {
			gopcode(PLOAD, PTADDR, a->nname);
			gopcode(PADDO, PTADDR, f->nname);
			gopcodet(PSTOREI, arg->type, arg);
		} else {
			cgen(arg);
			genconv(f->type, arg->type);
			gopcode(PLOAD, PTADDR, a->nname);
			gopcode(PADDO, PTADDR, f->nname);
			gopcodet(PSTOREI, arg->type, N);
		}
		arg = listnext(&save);
		f = f->down;
	}
	gopcodet(PRETURN, N, N);
}

void
cgen_call(Node *n, int toss)
{
	Node *t, *at, *ae, *sn;
	Iter save;
	int i;

	/*
	 * open a block
	 */
	gopcodet(PCALL1, N, n->left);

	/*
	 * prepare the input args
	 */
	t = n->left->type;
	if(t->etype == TPTR)
		t = t->type;

	at = *getinarg(t);			// parameter struct
	sn = at->nname;				// in arg structure name

	at = at->type;				// parameter fields
	ae = listfirst(&save, &n->right);	// expr list

	for(i=0; i<t->intuple; i++) {
		if(ae == N)
			fatal("cgen_call: tupleness");

		if(ae->addable && !needconvert(at->type, ae->type)) {
			gopcode(PADDR, PTADDR, sn);
			gopcode(PADDO, PTADDR, at->nname);
			gopcodet(PSTOREI, at->type, ae);
		} else {
			cgen(ae);
			genconv(at->type, ae->type);
			gopcode(PADDR, PTADDR, sn);
			gopcode(PADDO, PTADDR, at->nname);
			gopcodet(PSTOREI, at->type, N);
		}
		ae = listnext(&save);
		at = at->down;
	}

	/*
	 * call the function
	 */
	switch(n->op) {
	default:
		fatal("cgen_call: %O", n->op);

	case OCALL:
		gopcodet(PCALL2, N, n->left);
		break;

	case OCALLPTR:
		cgen(n->left);
		gopcodet(PCALLI2, N, n->left);
		break;

	case OCALLMETH:
		cgen(n->left);
		gopcodet(PCALLM2, N, n->left);
		break;

	case OCALLINTER:
		cgen(n->left);
		gopcodet(PCALLF2, N, n->left);
		break;
	}

	/*
	 * toss the output args
	 */
	if(toss) {
		gopcodet(PCALL3, N, n->left);
		return;
	}
}

void
cgen_callret(Node *n, Node *mas)
{
	Node *t, *at, *ae, *sn;
	Iter save;
	int i;

	t = n->left->type;
	if(t->etype == TPTR)
		t = t->type;

	at = *getoutarg(t);			// parameter struct
	sn = at->nname;				// out arg structure name
	at = at->type;				// parameter fields

	// call w single return val to a register
	if(mas == N) {
		gopcode(PADDR, PTADDR, sn);
		gopcode(PADDO, PTADDR, at->nname);
		gopcodet(PLOADI, at->type, N);
		gopcodet(PCALL3, N, N);
		return;
	}

	// call w multiple values to lval list
	ae = listfirst(&save, &mas);	// expr list
	for(i=0; i<t->outtuple; i++) {
		if(ae == N)
			fatal("cgen_callret: output arguments do not match");

		if(ae->addable) {
			gopcode(PADDR, PTADDR, sn);
			gopcode(PADDO, PTADDR, at->nname);
			gopcodet(PLOADI, at->type, ae);
		} else {
			agen(ae);
			gopcode(PADDR, PTADDR, sn);
			gopcode(PADDO, PTADDR, at->nname);
			gopcodet(PLOADI, at->type, N);
		}

		ae = listnext(&save);
		at = at->down;
	}

	gopcodet(PCALL3, N, N);
}

void
genprint(Node *n)
{
	Node *arg;
	Iter save;

	arg = listfirst(&save, &n);
	while(arg != N) {
		cgen(arg);
		gopcodet(PPRINT, arg->type, N);
		arg = listnext(&save);
	}
}

int
needconvert(Node *tl, Node *tr)
{
	if(isinter(tl))
		if(isptrto(tr, TSTRUCT) || isinter(tr))
			return 1;
	if(isptrto(tl, TSTRUCT))
		if(isinter(tr))
			return 1;
	return 0;
}

void
genconv(Node *tl, Node *tr)
{
	if(needconvert(tl, tr))
		gopcode(PCONV, PTNIL, nod(OCONV, tl, tr));
}

void
genindex(Node *n)
{
	gopcode(PINDEX, n->right->type->etype, n);
}

int
optopop(int op)
{
	int a;

	switch(op) {
	default:
		fatal("optopop: unknown op %O\n", op);

	case OMINUS:	a = PMINUS;	break;
	case OCOM:	a = PCOM;	break;
	case OAND:	a = PAND;	break;
	case OOR:	a = POR;	break;
	case OXOR:	a = PXOR;	break;
	case OADD:	a = PADD;	break;
	case OMUL:	a = PMUL;	break;
	case OMOD:	a = PMOD;	break;
	case OSUB:	a = PSUB;	break;
	case ODIV:	a = PDIV;	break;
	case OLSH:	a = PLSH;	break;
	case ORSH:	a = PRSH;	break;
	case OCAT:	a = PCAT;	break;
	}
	return a;
}
