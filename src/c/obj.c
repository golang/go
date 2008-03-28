// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"gen.h"

static	Prog*	firstp;
static	Prog*	lastp;
static	int	typeexpand;

void
dumpobj(void)
{
	Plist *pl;
	Prog *p;
	long lno;

	Bprint(bout, "\n\n/*\n");
	Bprint(bout, " * automatic code generated from\n");
	Bprint(bout, " * %s in package \"%s\"\n", curio.infile, package);
	dumpexport();
	Bprint(bout, " */\n", curio.infile, package);
	Bprint(bout, "#include \"gort.h\"\n");

	// put out external variables and types
	doframe(externdcl, "external");
	dumpmethods();

	// put out signatures
	dumpsignatures();

	// put out functions
	for(pl=plist; pl!=nil; pl=pl->link) {
		/* print out the function header */
		dumpfunct(pl);

		/* clear the marks */
		for(p=pl->firstpc; p!=nil; p=p->link)
			p->mark = 0;

		/* relinearize the object code */
		firstp = mal(sizeof(*firstp));
		lastp = firstp;
		follow(pl->firstpc);
		lastp->link = P;
		pl->firstpc = firstp->link;

		/* clear the marks - relabel the locations */
		for(p=pl->firstpc; p!=nil; p=p->link)
			p->mark = 0;

		/* mark the labels */
		for(p=pl->firstpc; p!=nil; p=p->link) {
			if(p->addr.branch != P)
				p->addr.branch->mark = 1;
		}

		/* interpret the instructions */
		lno = dynlineno;
		for(p=pl->firstpc; p!=nil; p=p->link) {
			dynlineno = p->lineno;
			dynloc = p->loc;
			obj(p);
		}
		dynlineno = lno;
		Bprint(bout, "}\n");
	}
}

void
obj1(Prog *p)
{
	Node *n;
	static long uloc, olino;

	Bprint(bout, "\n\t// %P\n", p);
	if(p->mark)
		Bprint(bout, "_L%ld:\n", p->loc);

	uloc++;
	if(p->lineno != 0)
		olino = p->lineno;
	Bprint(bout, "\tgotrace(%ld, %ld);\n", uloc, olino);

	switch(p->op) {
	default:
		warn("obj: unknown opcode %A", p);
		Bprint(bout, "\tprintf(\"unknown line %ld-%ld: %A\\n\");\n",
			dynloc, dynlineno, p);

	case PPANIC:
		Bprint(bout, "\tprintf(\"panic line %ld\\n\");\n", dynlineno);
		Bprint(bout, "\tgoexit(1);\n");
		break;

	case PPRINT:
		Bprint(bout, "\tprint%s(%R);\n", getfmt(p->pt), p->pt);
		break;

	case PGOTO:
		Bprint(bout, "\tgoto %D;\n", p);
		break;

	case PGOTOX:
		yyerror("label not declared: %S", p->addr.node->left->sym);
		break;

	case PCMP:
		if(p->pt == PTSTRING)
			goto pcmpz;

		switch(p->link->op) {
		case PBEQ:
			Bprint(bout, "\tif(%R == %D) {\n", p->pt, p);
			break;
		case PBNE:
			Bprint(bout, "\tif(%R != %D) {\n", p->pt, p);
			break;
		case PBLT:
			Bprint(bout, "\tif(%R < %D) {\n", p->pt, p);
			break;
		case PBLE:
			Bprint(bout, "\tif(%R <= %D) {\n", p->pt, p);
			break;
		case PBGE:
			Bprint(bout, "\tif(%R >= %D) {\n", p->pt, p);
			break;
		case PBGT:
			Bprint(bout, "\tif(%R > %D) {\n", p->pt, p);
			break;
		}
		break;

	pcmpz:
		Bprint(bout, "\tif(cmpZ(%D) ", p);
		switch(p->link->op) {
		case PBEQ:
			Bprint(bout, "== 0) {\n");
			break;
		case PBNE:
			Bprint(bout, "!= 0) {\n");
			break;
		case PBLT:
			Bprint(bout, "< 0) {\n");
			break;
		case PBLE:
			Bprint(bout, "<= 0) {\n");
			break;
		case PBGE:
			Bprint(bout, ">= 0) {\n");
			break;
		case PBGT:
			Bprint(bout, "> 0) {\n");
			break;
		}
		break;

	case PTEST:
		switch(p->link->op) {
		case PBTRUE:
			Bprint(bout, "\tif(%D != 0) {\n", p);
			break;
		case PBFALSE:
			Bprint(bout, "\tif(%D == 0) {\n", p);
			break;
		}
		break;

	case PBEQ:
	case PBNE:
	case PBLT:
	case PBLE:
	case PBGE:
	case PBGT:
	case PBTRUE:
	case PBFALSE:
		Bprint(bout, "\t\tgoto %D; }\n", p);
		break;

	case PLEN:
		Bprint(bout, "\t%R = %D->len;\n", PTINT32, p);
		break;

	case PNEW:
		if(p->addr.type != ANODE)
			goto bad;
		n = p->addr.node;
		n = n->type;
		n = n->type;
		if(n == N || n->op != OTYPE)
			goto bad;
		Bprint(bout, "\t%R = gomal(sizeof(%C%lC));\n", p->pt, n, n);
		break;

	case PLOAD:
		if(p->pt == PTPTR || p->pt == PTADDR) {
			Bprint(bout, "\t%R = (%Q)%D;\n", p->pt, PTPTR, p);
			break;
		}
		Bprint(bout, "\t%R = %D;\n", p->pt, p);
		break;

	case PLOADI:	// R/D = *(A)
		Bprint(bout, "\t%D = *(%Q*)%R;\n", p, p->pt, PTADDR);
		break;

	case PSTORE:
		if(p->pt == PTPTR || p->pt == PTADDR) {
			if(p->addr.type != ANODE)
				goto bad;
			n = p->addr.node;
			if(n == N || n->type == N)
				goto bad;
			Bprint(bout, "\t%D = (%C)%R;\n", p, n->type, p->pt);
			break;
		}
		Bprint(bout, "\t%D = %R;\n", p, p->pt);
		break;

	case PSTOREI:	// *(A) = R/D
		Bprint(bout, "\t*(%Q*)%R = %D;\n", p->pt, PTADDR, p);
		break;

	case PSTOREZ:
		switch(p->pt) {
		default:
			Bprint(bout, "\t%D = 0;\n", p);
			break;

		case PTARRAY:
		case PTSTRUCT:
			Bprint(bout, "\tmemset(&%D, 0, sizeof(%D));\n", p, p);
			break;

		case PTINTER:
			Bprint(bout, "\t%D.s = 0; %D.m = 0;\n", p, p);
			break;

		case PTSTRING:
			Bprint(bout, "\t%D = &nilstring;\n", p);
			break;
		}
		break;

	case PCONV:
		doconv(p);
		break;

	case PADDR:
		Bprint(bout, "\t%R = (%Q)&%D;\n", p->pt, p->pt, p);
		break;

	case PADDO:
		if(p->addr.type != ANODE)
			goto bad;
		n = p->addr.node;
		if(n == N || n->op != ONAME || n->sym == S)
			goto bad;
		if(n->uberstruct == N || n->uberstruct->etype != TSTRUCT)
			goto bad;

		Bprint(bout, "\t%R = (%Q)((char*)%R + offsetof(_T_%ld, %s));\n",
			p->pt, PTADDR, p->pt,
//			n->uberstruct->nname->sym->package,
			n->uberstruct->vargen, n->sym->name);
		break;

	case PINDEXZ:
		Bprint(bout, "\t%R = %D->string[%R];\n",
			PTUINT8, p, p->pt);
		break;

	case PINDEX:
		if(p->addr.type != ANODE)
			goto bad;
		n = p->addr.node;
		Bprint(bout, "\t%R += (%R)*sizeof(%C);\n",
			PTADDR, p->pt, n->type);
		break;

	case PSLICE:
		if(p->addr.type != ANODE)
			goto bad;
		n = p->addr.node;
		Bprint(bout, "\tsliceZ(%R, %D);\n", p->pt, p);
		break;

	case PCAT:
		Bprint(bout, "\tcatZ(%D);\n", p);
		break;

	case PADD:
		Bprint(bout, "\t%R += %D;\n", p->pt, p);
		break;

	case PSUB:
		Bprint(bout, "\t%R -= %D;\n", p->pt, p);
		break;

	case PMUL:
		Bprint(bout, "\t%R *= %D;\n", p->pt, p);
		break;

	case PDIV:
		Bprint(bout, "\t%R /= %D;\n", p->pt, p);
		break;

	case PLSH:
		Bprint(bout, "\t%R <<= %D;\n", p->pt, p);
		break;

	case PRSH:
		Bprint(bout, "\t%R >>= %D;\n", p->pt, p);
		break;

	case PMOD:
		Bprint(bout, "\t%R %%= %D;\n", p->pt, p);
		break;

	case PAND:
		Bprint(bout, "\t%R &= %D;\n", p->pt, p);
		break;

	case POR:
		Bprint(bout, "\t%R |= %D;\n", p->pt, p);
		break;

	case PXOR:
		Bprint(bout, "\t%R ^= %D;\n", p->pt, p);
		break;

	case PMINUS:
		Bprint(bout, "\t%R = -%R;\n", p->pt, p->pt);
		break;

	case PCOM:
		Bprint(bout, "\t%R = ~%R;\n", p->pt, p->pt);
		break;

	case PRETURN:
		Bprint(bout, "\treturn;\n");
		break;

	case PCALL1:	// process the arguments
		docall1(p);
		break;

	case PCALL2:	// call the normal function
		docall2(p);
		break;

	case PCALLI2:	// call the indirect function
		docalli2(p);
		break;

	case PCALLM2:	// call the method function
		docallm2(p);
		break;

	case PCALLF2:	// call the interface method function
		docallf2(p);
		break;

	case PCALL3:	// process the return
		docall3(p);
		break;

	case PEND:
		Bprint(bout, "\treturn;\n");
		break;
	}
	return;

bad:
	print("bad code generation on\n\t// %P\n", p);
}


void
follow(Prog *p)
{
	Prog *q;
	int i, op;

loop:
	if(p == P)
		return;

	if(p->op == PGOTO) {
		q = p->addr.branch;
		if(q != P) {
			p->mark = 1;
			p = q;
			if(p->mark == 0)
				goto loop;
		}
	}

	if(p->mark) {
		/* copy up to 4 instructions to avoid branch */
		for(i=0, q=p; i<4; i++, q=q->link) {
			if(q == P)
				break;
			if(q == lastp)
				break;
			if(q->op == PGOTO)
				break;
			if(q->addr.branch == P)
				continue;
			if(q->addr.branch->mark)
				continue;
			if(q->op == PCALL1)
				continue;

			// we found an invertable now copy
//			for(;;) {
//				q = copyp(p);
//				p = p->link;
//				q->mark = 1;
//				lastp->link = q;
//				lastp = q;
//				if(q->op != a || q->addr.branch == P || q->addr.branch->mark)
//					continue;
//
//				q->op = relinv(q->op);
//				p = q->addr.branch;
//				q->addr.branch = q->link;
//				q->link = p;
//				follow(q->link);
//				p = q->link;
//				if(p->mark)
//					return;
//				goto loop;
//			}
		}

		q = mal(sizeof(*q));
		q->op = PGOTO;
		q->lineno = p->lineno;
		q->addr.type = ABRANCH;
		q->addr.branch = gotochain(p);
		p = q;
	}

	p->mark = 1;
	p->loc = lastp->loc+1;
	lastp->link = p;
	lastp = p;

	op = p->op;
	if(op == PGOTO || op == PRETURN || op == OEND)
		return;

	if(op == PCALL1 || p->addr.branch == P) {
		p = p->link;
		goto loop;
	}

	q = gotochain(p->link);
	if(q != P && q->mark) {
		p->op = brcom(op);
		p->link = p->addr.branch;
		p->addr.branch = q;
	}
	follow(p->link);
	q = gotochain(p->addr.branch);
	p->addr.branch = q;
	if(q != P && q->mark)
		return;

	p = q;
	goto loop;
}

void
obj(Prog *p)
{
	Node *n;
	String *s;
	long i;

	if(p->addr.type != ANODE)
		goto out;
	n = p->addr.node;
	if(n == N || n->op != OLITERAL)
		goto out;
	if(p->pt != PTSTRING)
		goto out;

	s = n->val.sval;
	Bprint(bout, "\t{ static struct {_T_U32	l;_T_U8	s[%d]; } slit = { %d", s->len, s->len);
	for(i=0; i<s->len; i++) {
		if(i%16 == 0)
			Bprint(bout, "\n\t\t");
		Bprint(bout, ",%d", s->s[i]);
	}
	Bprint(bout, " };\n");

	obj1(p);
	Bprint(bout, "\t}\n");
	return;

out:
	obj1(p);
}

Prog*
gotochain(Prog *p)
{
	int i;

	for(i=0; i<20; i++) {
		if(p == P || p->op != PGOTO)
			return p;
		p = p->addr.branch;
	}
	return P;
}

/*
 * print a C type
 */
int
Cconv(Fmt *fp)
{
	char buf[1000], buf1[100];
	Node *t, *f, *n;
	Iter it;
	int pt;
	long v1, v2;

	t = va_arg(fp->args, Node*);
	if(t == N)
		return fmtstrcpy(fp, "<C>");

	t->recur++;
	if(t->op != OTYPE) {
		snprint(buf, sizeof(buf), "C-%O", t->op);
		goto out;
	}
	if(t->recur > 5) {
		snprint(buf, sizeof(buf), "C-%E ...", t->etype);
		goto out;
	}

	// post-name format
	if(fp->flags & FmtLong) {
		strcpy(buf, "");
		switch(t->etype) {
		default:
			break;
		case TARRAY:
			snprint(buf, sizeof(buf), "[%ld]", t->bound);
			break;
		case TFUNC:
			if(t->thistuple > 0) {
				f = *getthis(t);
				v1 = 9999;
				v2 = 9999;
				if(f != N) {
					v1 = f->vargen;
					if(f->nname != N)
						v2 = f->nname->vargen;
				}
				snprint(buf1, sizeof(buf1), "(_T_%ld* _V_%ld",
					v1, v2);
				strncat(buf, buf1, sizeof(buf));
			} else
				strncat(buf, "(void* _dummythis", sizeof(buf));

			if(t->outtuple > 0) {
				f = *getoutarg(t);
				v1 = 9999;
				v2 = 9999;
				if(f != N) {
					v1 = f->vargen;
					if(f->nname != N)
						v2 = f->nname->vargen;
				}
				snprint(buf1, sizeof(buf1), ", _T_%ld* _V_%ld",
					v1, v2);
				strncat(buf, buf1, sizeof(buf));
			} else
				strncat(buf, ", void* _dummyout", sizeof(buf));

			if(t->intuple > 0) {
				f = *getinarg(t);
				v1 = 9999;
				v2 = 9999;
				if(f != N) {
					v1 = f->vargen;
					if(f->nname != N)
						v2 = f->nname->vargen;
				}
				snprint(buf1, sizeof(buf1), ", _T_%ld* _V_%ld)",
					v1, v2);
				strncat(buf, buf1, sizeof(buf));
			} else
				strncat(buf, ", void* _dummyin)", sizeof(buf));
			break;
		}
		goto out;
	}

	if(t->vargen != 0 && !typeexpand) {
		if(t->etype == TFUNC) {
			strcpy(buf, "void");
			goto out;
		}
		snprint(buf, sizeof(buf), "_T_%ld", t->vargen);
		goto out;
	}

	switch(t->etype) {
	default:
		pt = conv2pt(t);
		snprint(buf, sizeof(buf), "%Q", pt);
		break;

	case TSTRUCT:
		if(fp->flags & FmtShort) {
			strcpy(buf, "{");
		} else {
			if(t->vargen != 0) {
				snprint(buf, sizeof(buf), "_T_%ld", t->vargen);
				goto out;
			}
			strcpy(buf, "struct{");
		}

		f = structfirst(&it, &t);
		while(f != N) {
			n = f->type;
			if(n->etype == TFUNC)
				goto next;
			if(f->sym == S)
				snprint(buf1, sizeof(buf1), "%C;", n);
			else
				snprint(buf1, sizeof(buf1), "%C %s;", n, f->sym->name);
			strncat(buf, buf1, sizeof(buf));
		next:
			f = structnext(&it);
		}
		strncat(buf, "}", sizeof(buf));
		break;

	case TPTR:
		if(isptrto(t, TSTRING)) {
			snprint(buf, sizeof(buf), "%C", t->type);
			break;
		}
		snprint(buf, sizeof(buf), "%C*", t->type);
		break;

	case TARRAY:
		snprint(buf, sizeof(buf), "%C", t->type);
		break;

	case TFUNC:
		strcpy(buf, "void");
		break;
	}

out:
	t->recur--;
	return fmtstrcpy(fp, buf);
}

/*
 * print Prog operand
 */
int
Dconv(Fmt *fp)
{
	char buf[500];
	Prog *p;
	Node *n;

	if(fp->flags & FmtLong) {
		p = nil;
		n = va_arg(fp->args, Node*);
		goto prnode;
	}
	p = va_arg(fp->args, Prog*);

	switch(p->addr.type) {
	default:
		snprint(buf, sizeof(buf), "addr.type=%d", p->addr.type);
		break;

	case ANONE:
		snprint(buf, sizeof(buf), "%R", p->pt);
		break;

	case ANODE:
		n = p->addr.node;
		goto prnode;

	case ABRANCH:
		p = p->addr.branch;
		if(p == P) {
			snprint(buf, sizeof(buf), "addr.branch=nil");
			break;
		}
		snprint(buf, sizeof(buf), "_L%ld", p->loc);
		break;
	}
	goto out;

prnode:
	if(n == N) {
		snprint(buf, sizeof(buf), "addr.node=nil");
		goto out;
	}
	switch(n->op) {
	default:
		snprint(buf, sizeof(buf), "%N", p->addr.node);
		break;

	case ONAME:
		if(n->vargen != 0) {
			snprint(buf, sizeof(buf), "_V_%ld", n->vargen);
			break;
		}
		snprint(buf, sizeof(buf), "%s_%s", n->sym->opackage, n->sym->name);
		break;

	case OLITERAL:
		switch(p->pt) {
		badlit:
		default:
			snprint(buf, sizeof(buf), "BADLIT-%d pt-%d", p->pt, n->val.ctype);
			break;
		case PTINT8:
		case PTINT16:
		case PTINT32:
		case PTUINT8:
		case PTUINT16:
		case PTUINT32:
			switch(n->val.ctype) {
			default:
				goto badlit;
			case CTINT:
			case CTSINT:
			case CTUINT:
				if(n->val.vval < 0)
					snprint(buf, sizeof(buf), "-0x%llux", -n->val.vval);
				else
					snprint(buf, sizeof(buf), "0x%llux", n->val.vval);
				break;
			}
			break;
		case PTINT64:
		case PTUINT64:
			switch(n->val.ctype) {
			default:
				goto badlit;
			case CTINT:
			case CTSINT:
			case CTUINT:
				snprint(buf, sizeof(buf), "0x%lluxll", n->val.vval);
				break;
			}
			break;
		case PTFLOAT32:
		case PTFLOAT64:
		case PTFLOAT80:
			switch(n->val.ctype) {
			default:
				goto badlit;
			case CTFLT:
				snprint(buf, sizeof(buf), "%.17e", n->val.dval);
				break;
			}
			break;
		case PTBOOL:
			switch(n->val.ctype) {
			default:
				goto badlit;
			case CTBOOL:
				snprint(buf, sizeof(buf), "%lld", n->val.vval);
				break;
			}
			break;
		case PTPTR:
			switch(n->val.ctype) {
			default:
				goto badlit;
			case CTSTR:
				snprint(buf, sizeof(buf), "\"%Z\"", n->val.sval);
				break;
			case CTNIL:
				snprint(buf, sizeof(buf), "(void*)0", n->val.sval);
				break;
			}
			break;

		case PTSTRING:
			snprint(buf, sizeof(buf), "(_T_Z)&slit");
			break;

		}
		break;
	}

out:
	return fmtstrcpy(fp, buf);
}

char*
thistypenam(Node *t)
{
	char *typ;
	Node *n;

	typ = "???";
	if(t == N)
		return typ;
	n = getthisx(t);	// struct{field a *T}
	if(n != N)
		n = n->type;	// field a *T
	if(n != N)
		n = n->type;	// *T
	if(n != N)
		n = n->type;	// T
	if(n != N && n->sym != S)
		typ = n->sym->name;
	return typ;
}

void
dumpfunct(Plist *pl)
{
	Node *t;
	char *pkg, *typ, *fun;

	t = pl->name->type;
	pkg = pl->name->sym->opackage;
	fun = pl->name->sym->name;

	if(t->thistuple > 0) {
		typ = thistypenam(t);	// struct{field a *T}
		Bprint(bout, "\n%C %s_%s_%s%lC", t, pkg, typ, fun, t);
	} else {
		Bprint(bout, "\n%C %s_%s%lC", t, pkg, fun, t);
	}

	Bprint(bout, "\n{\n");
	doframe(pl->locals, "local");
}

void
dumpmethods()
{
	Node *t;
	char *pkg, *typ, *fun;
	Plist *pl;

	for(pl=plist; pl!=nil; pl=pl->link) {
		t = pl->name->type;
		if(t->thistuple > 0) {
			pkg = pl->name->sym->opackage;
			fun = pl->name->sym->name;
			typ = thistypenam(t);
			Bprint(bout, "\n%C %s_%s_%s%lC;\n", t, pkg, typ, fun, t);
		}
	}
}

static int
sigcmp(Sig *a, Sig *b)
{
	return strcmp(a->fun, b->fun);
}

void
dumpsignatures(void)
{
	Dcl *d;
	Node *t, *f;
	Sym *s1, *s;
	char *pkg, *typ, *fun;
	int et, o, any;
	Sig *a, *b;

	/* put all the names into a linked
	 * list so that it may be generated in sorted order.
	 * the runtime will be linear rather than quadradic
	 */

	any = 1;
	for(d=externdcl; d!=D; d=d->forw) {
		if(d->op != OTYPE)
			continue;

		t = d->dnode;
		et = t->etype;
		if(et != TSTRUCT && et != TINTER)
			continue;

		s = d->dsym;
		if(s == S)
			continue;

		typ = s->name;
		if(typ[0] == '_')
			continue;

		pkg = s->opackage;
		if(pkg != package) {
			if(et == TINTER)
				Bprint(bout, "extern	_Sigi sig_%s_%s[];\n", pkg, typ);
			else
				Bprint(bout, "extern	_Sigs sig_%s_%s[];\n", pkg, typ);
			continue;
		}

		a = nil;
		o = 0;
		for(f=t->type; f!=N; f=f->down) {
			if(f->type->etype != TFUNC)
				continue;

			if(f->etype != TFIELD)
				fatal("dumpsignatures: not field");

			s1 = f->sym;
			if(s1 == nil)
				continue;
			fun = s1->name;
			if(fun[0] == '_')
				continue;

			b = mal(sizeof(*b));
			b->link = a;
			a = b;

			a->fun = fun;
			a->hash = PRIME8*stringhash(fun) + PRIME9*typehash(f->type, 0);
			a->offset = o;
			o++;
		}

		if(1 || et == TINTER || a != nil) {
			if(any) {
				Bprint(bout, "\n");
				any = 0;
			}

			a = lsort(a, sigcmp);

			if(et == TINTER) {
				o = 0;
				for(b=a; b!=nil; b=b->link)
					o++;
				Bprint(bout, "_Sigi sig_%s_%s[] =\n", pkg, typ);
				Bprint(bout, "{\n");
				Bprint(bout, "\t{ \"\", 0, %d}, // count\n", o);
				for(b=a; b!=nil; b=b->link) {
					Bprint(bout, "\t{ \"%s\", 0x%.8lux, %d},\n",
						b->fun, b->hash, b->offset);
				}
			} else {
				Bprint(bout, "_Sigs sig_%s_%s[] =\n", pkg, typ);
				Bprint(bout, "{\n");
				for(b=a; b!=nil; b=b->link) {
					Bprint(bout, "\t{ \"%s\", 0x%.8lux, &%s_%s_%s },\n",
						b->fun, b->hash, pkg, typ, b->fun);
				}
			}
			Bprint(bout, "\t{ 0,0,0 }\n");
			Bprint(bout, "};\n");
		}
	}
}

int
istypstr(Node *t)
{
	if(t == N)
		fatal("istypstr: t nil");
	if(t->etype == TSTRUCT)
		return 1;
	return 0;
}

static int XXX = 0;
static int YYY = 0;

int
alldefined(Node *t, int first)
{
	Node *t1;

	if(t == N)
		return 1;

	if(t->op != OTYPE)
		fatal("alldefined: not OTYPE: %O", t->op);

	if(t->recur)
		return 1;

	if(!first && t->sym!=S && t->sym->undef != 0)
		return 1;

	t->recur++;

	switch(t->etype) {
	default:
		// should be basic types
		return 1;

	case TPTR:
	case TARRAY:
	case TFIELD:
		if(!alldefined(t->type, 0))
			goto no;
		break;

	case TSTRUCT:
	case TFUNC:
		for(t1=t->type; t1!=N; t1=t1->down) {
			if(!alldefined(t1, 0))
				goto no;
		}
		break;
	}

	t->recur--;
	return 1;

no:
	t->recur--;
	return 0;
}

void
doframe(Dcl *r, char *msg)
{
	Sym *s;
	Dcl *d;
	Node *n, *t;
	int flag, pass, any;
	char *tab, *nam, *pkg, *typ;

	tab = "\t";
	if(msg[0] != 'l')
		tab = "";

	// put out types
	flag = 1;
	typeexpand = 1;
	for(pass=0;; pass++) {
if(XXX)print("\npass %d\n\n", pass);
		any = 0;
		for(d=r; d!=D; d=d->forw) {
			if(d->op != OTYPE)
				continue;

			if(flag) {
				Bprint(bout, "\n%s// %s types\n", tab, msg);
				flag = 0;
			}

			n = d->dnode;
			nam = "???";
			s = d->dsym;
			if(s != S)
				nam = s->name;

			if(pass == 0) {
				if(s != S)
					s->undef = 0;
				if(istypstr(n)) {
					Bprint(bout, "%stypedef struct _T_%ld _T_%ld; // %s\n",
						tab, n->vargen, n->vargen, nam);
if(XXX)print("\t1 pass-%d ", pass);
if(XXX)print("typedef struct _T_%ld _T_%ld; // %s\n", n->vargen, n->vargen, nam);
				}
				any = 1;
				continue;
			}

if(XXX)if(s != S) print("looking at %s undef=%d: %lT\n", s->name, s->undef, n);

			if(s != S && s->undef == 0 && alldefined(n, 1)) {
if(XXX)print("\t2 pass-%d ", pass);
				if(istypstr(n)) {
					Bprint(bout, "%sstruct _T_%ld %hC; // %s\n",
						tab, n->vargen, n, nam);
if(XXX)print("struct _T_%ld %hC; // %s\n", n->vargen, n, nam);
				} else {
					if(n->etype != TFUNC)
					Bprint(bout, "%stypedef %C _T_%ld%lC; // %s\n",
						tab, n, n->vargen, n, nam);
if(XXX)print("typedef %C _T_%ld%lC; // %s\n", n, n->vargen, n, nam);
				}
				s->undef = 1;
				any = 1;
			}
		}
		if(any)
			continue;

		for(d=r; d!=D; d=d->forw) {
			if(d->op != OTYPE)
				continue;
			n = d->dnode;
			s = d->dsym;
			if(s != S) {
				if(s->undef == 0)
					fatal("doframe: couldnt resolve type %s %lT\n",
						s->name, n);
				continue;
			}
if(XXX)print("\t-3 pass-%d ", pass);
			if(istypstr(n)) {
				Bprint(bout, "%sstruct _T_%ld %hC;\n",
					tab, n->vargen, n);
if(XXX)print("struct _T_%ld %hC;\n", n->vargen, n);
			} else {
				Bprint(bout, "%stypedef %C _T_%ld%lC;\n",
					tab, n, n->vargen, n);
if(XXX)print("typedef %C _T_%ld%lC;\n", n, n->vargen, n);
			}
		}
		break;
	}
	typeexpand = 0;

	flag = 1;
	for(d=r; d!=D; d=d->forw) {
		if(d->op != ONAME)
			continue;

		if(flag) {
			Bprint(bout, "\n%s// %s variables\n", tab, msg);
			flag = 0;
		}

		nam = "???";
		pkg = nam;
		s = d->dsym;
		if(s != S) {
			nam = s->name;
			pkg = s->opackage;
		}

		n = d->dnode;
		t = n->type;
		if(n->vargen != 0) {
if(YYY) print("nam-1 %s\n", nam);
			Bprint(bout, "%s%C _V_%ld%lC; // %s\n",
				tab, t, n->vargen, t, nam);
			continue;
		}

		if(t->etype == TFUNC && t->thistuple > 0) {
if(YYY) print("nam-2 %s\n", nam);
			typ = thistypenam(t);
			Bprint(bout, "%s%C %s_%s_%s%lC;\n",
				tab, t, pkg, typ, nam, t);
			continue;
		}

if(YYY) print("nam-3 %E %s %lT\n", t->etype, nam, t);
		Bprint(bout, "%s%C %s_%s%lC;\n",
			tab, t, pkg, nam, t);
	}
}

/*
 * open the frame
 * declare dummy this/in/out args
 */
void
docall1(Prog *p)
{
	Node *f, *t, *n;

	if(p->addr.type != ANODE)
		goto bad;

	f = p->addr.node;
	if(f == N)
		goto bad;
	t = f->type;
	if(t == N)
		goto bad;
	if(t->etype == TPTR)
		t = t->type;
	if(t->etype != TFUNC)
		goto bad;

	Bprint(bout, "\t{\n");		// open a block - closed in CALL2/CALL3

	if(t->thistuple > 0) {
		n = *getthis(t);
		if(n->nname == N)
			goto bad;
		Bprint(bout, "\t\t_T_%ld _V_%ld; // %S\n", n->vargen, n->nname->vargen, n->sym);
	}
	if(t->outtuple > 0) {
		n  = *getoutarg(t);
		if(n->nname == N)
			goto bad;
		Bprint(bout, "\t\t_T_%ld _V_%ld; // %S\n", n->vargen, n->nname->vargen, n->sym);
	}
	if(t->intuple > 0) {
		n = *getinarg(t);
		if(n->nname == N)
			goto bad;
		Bprint(bout, "\t\t_T_%ld _V_%ld; // %S\n", n->vargen, n->nname->vargen, n->sym);
	}

	return;

bad:
	fatal("docall1: bad %P", p);
}

/*
 * call the function
 */
void
docall2(Prog *p)
{
	Node *f, *t, *n;

	if(p->addr.type != ANODE)
		goto bad;
	f = p->addr.node;
	if(f == N)
		goto bad;
	t = f->type;
	if(t == N || t->etype != TFUNC)
		goto bad;

	Bprint(bout, "\t%D(", p);

	if(t->thistuple > 0) {
		n  = *getthis(t);
		Bprint(bout, "&_V_%ld", n->nname->vargen);
	} else
		Bprint(bout, "0");

	if(t->outtuple > 0) {
		n  = *getoutarg(t);
		Bprint(bout, ", &_V_%ld", n->nname->vargen);
	} else
		Bprint(bout, ", 0");

	if(t->intuple > 0) {
		n  = *getinarg(t);
		Bprint(bout, ", &_V_%ld);\n", n->nname->vargen);
	} else
		Bprint(bout, ", 0);\n");

	return;

bad:
	fatal("docall2: bad");
}

/*
 * call the function indirect
 */
void
docalli2(Prog *p)
{
	Node *f, *t, *n;

	if(p->addr.type != ANODE)
		goto bad;
	f = p->addr.node;
	if(f == N)
		goto bad;
	t = f->type;
	if(t == N || t->etype != TPTR)
		goto bad;
	t = t->type;
	if(t->etype != TFUNC)
		goto bad;

	// pass one -- declare the prototype
	if(t->outtuple > 0) {
		n  = *getoutarg(t);
		Bprint(bout, "\t(*(void(*)(void*, _T_%ld*", n->vargen);
	} else
		Bprint(bout, "\t(*(void(*)(void*, void*");

	if(t->intuple > 0) {
		n  = *getinarg(t);
		Bprint(bout, ", _T_%ld*)", n->vargen);
	} else
		Bprint(bout, ", void*)");

	// pass two -- pass the arguments
	if(t->outtuple > 0) {
		n  = *getoutarg(t);
		Bprint(bout, ")%R)(0, &_V_%ld", PTPTR, n->nname->vargen);
	} else
		Bprint(bout, ")%R)(0, 0", PTPTR);

	if(t->intuple > 0) {
		n  = *getinarg(t);
		Bprint(bout, ", &_V_%ld);\n", n->nname->vargen);
	} else
		Bprint(bout, ", 0);\n");

	return;

bad:
	fatal("docalli2: bad");
}

/*
 * call the method
 */
void
docallm2(Prog *p)
{
	Node *f, *t, *n;
	char *pkg, *typ, *nam;

	if(p->addr.type != ANODE)
		goto bad;
	f = p->addr.node;
	if(f == N || f->op != ODOTMETH)
		goto bad;
	t = f->type;
	if(t == N || t->etype != TFUNC)
		goto bad;

	nam = "???";
	pkg = nam;
	typ = nam;

	// get the structure name
	n = f->left;
	if(n != N)
		n = n->type;
	if(n->op == OTYPE && n->etype == TPTR)
		n = n->type;
	if(n->sym != S) {
		typ = n->sym->name;
		pkg = n->sym->opackage;
	}

	// get the function name
	n = f->right;
	if(n != N && n->op == ONAME && n->sym != S)
		nam = n->sym->name;

	Bprint(bout, "\t%s_%s_%s(%R", pkg, typ, nam, PTPTR);

	if(t->outtuple > 0) {
		n  = *getoutarg(t);
		Bprint(bout, ", (void*)&_V_%ld", n->nname->vargen);
	} else
		Bprint(bout, ", 0");

	if(t->intuple > 0) {
		n  = *getinarg(t);
		Bprint(bout, ", (void*)&_V_%ld);\n", n->nname->vargen);
	} else
		Bprint(bout, ", 0);\n");

	return;

bad:
	fatal("docallm2: bad");
}

/*
 * call the interface method
 */
void
docallf2(Prog *p)
{
	Node *f, *t, *n;
	int offset;

	if(p->addr.type != ANODE)
		goto bad;
	f = p->addr.node;
	if(f == N || f->op != ODOTINTER)
		goto bad;
	t = f->type;
	if(t == N || t->etype != TFUNC)
		goto bad;

	offset = 0;

	Bprint(bout, "\t(_U._R_I.m->fun[%d])(_U._R_I.s", f->kaka);

	if(t->outtuple > 0) {
		n  = *getoutarg(t);
		Bprint(bout, ", (void*)&_V_%ld", n->nname->vargen);
	} else
		Bprint(bout, ", 0");

	if(t->intuple > 0) {
		n  = *getinarg(t);
		Bprint(bout, ", (void*)&_V_%ld);\n", n->nname->vargen);
	} else
		Bprint(bout, ", 0);\n");

	return;

bad:
	fatal("docallf2: bad");
}

/*
 * close the frame
 */
void
docall3(Prog *p)
{
	Bprint(bout, "\t}\n");
}

char*
signame(Node *t)
{
// this code sb merged with thistypename
	static char name[100];
	char *typ, *pkg;

	typ = "???";
	pkg = typ;

	if(t == N || t->op != OTYPE)
		goto out;

	if(t->etype == TPTR) {
		t = t->type;
		if(t == N)
			goto out;
	}
	if(t->sym == S)
		goto out;
	typ = t->sym->name;
	pkg = t->sym->opackage;	// this may not be correct

out:
	snprint(name, sizeof(name), "sig_%s_%s", pkg, typ);
	return name;
}

void
doconv(Prog *p)
{
	Node *n, *tl, *tr;
	int l, pt;

	if(p->pt != PTNIL) {
		Bprint(bout, "\t%R = %R;\n", p->pt, p->pt1);
		return;
	}

	n = p->addr.node;
	if(p->addr.type != ANODE || n == N || n->op != OCONV)
		fatal("doconv: PCONV-N not OCONV");

	tl = n->left;
	tr = n->right;

	if(isinter(tl)) {
		if(isptrto(tr, TSTRUCT)) {
			Bprint(bout, "\tconvertStoI(%s, ", signame(tl));
			Bprint(bout, "%s); // _U._R_I = _U._R_P\n",
				signame(tr));
			return;
		}
		if(isinter(tr)) {
			Bprint(bout, "\tconvertItoI(%s); // _U._R_I = _U._R_I\n",
				signame(tl));
			return;
		}
	}
	if(isptrto(tl, TSTRUCT) && isinter(tr)) {
		Bprint(bout, "\t%R = %R.s;\n", TPTR, PTINTER);
		return;
	}
	if(isint[tl->etype] || isfloat[tl->etype]) {
		if(isint[tr->etype] || isfloat[tr->etype]) {
			Bprint(bout, "\t%R = %R;\n", conv2pt(tl), conv2pt(tr));
			return;
		}
	}

	if(isptrto(tl, TSTRING)) {
		if(isint[tr->etype]) {
			Bprint(bout, "\tconvertItoZ(%R);\n", conv2pt(tr));
			return;
		}
		l = isbytearray(tr);
		if(l > 0) {
			pt = PTADDR;
			if(tr->etype == TPTR)
				pt = TPTR;
			Bprint(bout, "\tconvertBtoZ(%R, %d);\n", pt, l-1);
			return;
		}
	}

	fatal("doconv: %T = %T", tl, tr);
}

char*
getfmt(int pt)
{
	switch(pt) {
	default:
		return "D";

	case PTUINT8:
	case PTUINT16:
	case PTUINT32:
	case PTUINT64:
		return "UD";

	case PTFLOAT32:
	case PTFLOAT64:
	case PTFLOAT80:
		return "F";

	case PTSTRING:
		return "Z";
	}
}
