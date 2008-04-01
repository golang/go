// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"
#include "gen.h"

Prog*
gbranch(int op, Node *t)
{
	Prog *p;

	p = prog(op);
	p->addr.type = ABRANCH;
	p->pt = conv2pt(t);
	return p;
}

Prog*
gopcode(int op, int pt, Node *n)
{
	Prog *p;

	p = prog(op);
	p->pt = pt;
	p->addr.node = n;
	if(n == N) {
		p->addr.type = ANONE;
		return p;
	}
	if(n->op == OTYPE) {
		p->pt1 = conv2pt(n);
		p->addr.type = ANONE;
		return p;
	}
	p->addr.type = ANODE;
//	p->param = n->param;
	return p;
}

Prog*
gopcodet(int op, Node *t, Node *n)
{
	return gopcode(op, conv2pt(t), n);
}

void
gaddoffset(Node *n)
{
	Prog *p;

	if(n == N || n->op != ONAME || n->sym == S)
		goto bad;
	p = gopcode(PADDO, PTADDR, n);
	return;

bad:
	fatal("gaddoffset: %N", n);

}

void
gconv(int t1, int t2)
{
	Prog *p;

	p = gopcode(PCONV, t1, N);
	p->pt1 = t2;
}

int
conv2pt(Node *t)
{
	if(t == N)
		return PTxxx;
	switch(t->etype) {
	case TPTR:
		t = t->type;
		if(t == N)
			return PTERROR;
		switch(t->etype) {
		case PTSTRING:
		case PTCHAN:
		case PTMAP:
			return t->etype;
		}
		return TPTR;
	}
	return t->etype;
}

void
patch(Prog *p, Prog *to)
{
	if(p->addr.type != ABRANCH)
		yyerror("patch: not a branch");
	p->addr.branch = to;
}

Prog*
prog(int as)
{
	Prog *p;

	p = pc;
	pc = mal(sizeof(*pc));

	pc->op = PEND;
	pc->addr.type = ANONE;
	pc->loc = p->loc+1;

	p->op = as;
	p->lineno = dynlineno;
	p->link = pc;
	return p;
}

void
proglist(void)
{
	Prog *p;

	print("--- prog list ---\n");
	for(p=firstpc; p!=P; p=p->link)
		print("%P\n", p);
}

char*	ptnames[] =
{
	[PTxxx]		= "",
	[PTINT8]	= "I8",
	[PTUINT8]	= "U8",
	[PTINT16]	= "I16",
	[PTUINT16]	= "U16",
	[PTINT32]	= "I32",
	[PTUINT32]	= "U32",
	[PTINT64]	= "I64",
	[PTUINT64]	= "U64",
	[PTFLOAT32]	= "F32",
	[PTFLOAT64]	= "F64",
	[PTFLOAT80]	= "F80",
	[PTBOOL]	= "B",
	[PTPTR]		= "P",
	[PTADDR]	= "A",
	[PTINTER]	= "I",
	[PTNIL]		= "N",
	[PTSTRUCT]	= "S",
	[PTSTRING]	= "Z",
	[PTCHAN]	= "C",
	[PTMAP]		= "M",
	[PTERROR]	= "?",
};

int
Xconv(Fmt *fp)
{
	char buf[100];
	int pt;

	pt = va_arg(fp->args, int);
	if(pt < 0 || pt >= nelem(ptnames) || ptnames[pt] == nil) {
		snprint(buf, sizeof(buf), "PT(%d)", pt);
		return fmtstrcpy(fp, buf);
	}
	return fmtstrcpy(fp, ptnames[pt]);
}

int
Qconv(Fmt *fp)
{
	char buf[100];
	int pt;

	pt = va_arg(fp->args, int);
	if(pt == PTADDR)
		pt = PTPTR;
	snprint(buf, sizeof(buf), "_T_%X", pt);
	return fmtstrcpy(fp, buf);
}

int
Rconv(Fmt *fp)
{
	char buf[100];
	int pt;

	pt = va_arg(fp->args, int);
	if(pt == PTADDR)
		snprint(buf, sizeof(buf), "_R_%X", pt);
	else
		snprint(buf, sizeof(buf), "_U._R_%X", pt);
	return fmtstrcpy(fp, buf);
}

/*
s%[ 	]*%%g
s%(\/\*.*)*%%g
s%,%\n%g
s%\n+%\n%g
s%(=0)*%%g
s%^P(.+)%	[P\1]		= "\1",%g
s%^	........*\]		=%&~%g
s%	=~%=%g
*/

static char*
pnames[] =
{
	[PXXX]		= "XXX",
	[PERROR]	= "ERROR",
	[PPANIC]	= "PANIC",
	[PPRINT]	= "PRINT",
	[PGOTO]		= "GOTO",
	[PGOTOX]	= "GOTOX",
	[PCMP]		= "CMP",
	[PNEW]		= "NEW",
	[PLEN]		= "LEN",
	[PTEST]		= "TEST",
	[PCALL1]	= "CALL1",
	[PCALL2]	= "CALL2",
	[PCALLI2]	= "CALLI2",
	[PCALLM2]	= "CALLM2",
	[PCALLF2]	= "CALLF2",
	[PCALL3]	= "CALL3",
	[PRETURN]	= "RETURN",
	[PBEQ]		= "BEQ",
	[PBNE]		= "BNE",
	[PBLT]		= "BLT",
	[PBLE]		= "BLE",
	[PBGE]		= "BGE",
	[PBGT]		= "BGT",
	[PBTRUE]	= "BTRUE",
	[PBFALSE]	= "BFALSE",
	[PLOAD]		= "LOAD",
	[PLOADI]	= "LOADI",
	[PSTORE]	= "STORE",
	[PSTOREI]	= "STOREI",
	[PSTOREZ]	= "STOREZ",
	[PSTOREZI]	= "STOREZI",
	[PCONV]		= "CONV",
	[PADDR]		= "ADDR",
	[PADDO]		= "ADDO",
	[PINDEX]	= "INDEX",
	[PINDEXZ]	= "INDEXZ",
	[PCAT]		= "CAT",
	[PADD]		= "ADD",
	[PSUB]		= "SUB",
	[PSLICE]	= "SLICE",
	[PMUL]		= "MUL",
	[PDIV]		= "DIV",
	[PLSH]		= "LSH",
	[PRSH]		= "RSH",
	[PMOD]		= "MOD",
	[PMINUS]	= "MINUS",
	[PCOM]		= "COM",
	[PAND]		= "AND",
	[POR]		= "OR",
	[PXOR]		= "XOR",
	[PEND]		= "END",
};

int
Aconv(Fmt *fp)
{
	char buf[100], buf1[100];
	Prog *p;
	int o;

	p = va_arg(fp->args, Prog*);
	if(p == P) {
		snprint(buf, sizeof(buf), "<P>");
		goto ret;
	}

	o = p->op;
	if(o < 0 || o >= nelem(pnames) || pnames[o] == nil)
		snprint(buf, sizeof(buf), "(A%d)", o);
	else
		snprint(buf, sizeof(buf), "%s", pnames[o]);

	o = p->pt;
	if(o != PTxxx) {
		snprint(buf1, sizeof(buf1), "-%X", o);
		strncat(buf, buf1, sizeof(buf));
	}

	o = p->pt1;
	if(o != PTxxx) {
		snprint(buf1, sizeof(buf1), "-%X", o);
		strncat(buf, buf1, sizeof(buf));
	}

ret:
	return fmtstrcpy(fp, buf);
}

int
Pconv(Fmt *fp)
{
	char buf[500], buf1[500];
	Prog *p;

	p = va_arg(fp->args, Prog*);
	snprint(buf1, sizeof(buf1), "%4ld %4ld %-9A", p->loc, p->lineno, p);

	switch(p->addr.type) {
	default:
		snprint(buf, sizeof(buf), "?%d", p->addr.type);
		break;

	case ANONE:
		goto out;

	case ANODE:
		snprint(buf, sizeof(buf), "%N", p->addr.node);
		break;

	case ABRANCH:
		if(p->addr.branch == P) {
			snprint(buf, sizeof(buf), "<nil>");
			break;
		}
		snprint(buf, sizeof(buf), "%ld", p->addr.branch->loc);
		break;
	}

	strncat(buf1, " ", sizeof(buf1));
	strncat(buf1, buf, sizeof(buf1));

out:
	return fmtstrcpy(fp, buf1);
}

static char*
typedefs[] =
{
	"int",		"int32",
	"uint",		"uint32",
	"rune",		"uint32",
	"short",	"int16",
	"ushort",	"uint16",
	"long",		"int32",
	"ulong",	"uint32",
	"vlong",	"int64",
	"uvlong",	"uint64",
	"float",	"float32",
	"double",	"float64",

};

void
belexinit(int lextype)
{
	int i;
	Sym *s0, *s1;

	for(i=0; i<nelem(typedefs); i+=2) {
		s1 = lookup(typedefs[i+1]);
		if(s1->lexical != lextype)
			yyerror("need %s to define %s",
				typedefs[i+1], typedefs[i+0]);
		s0 = lookup(typedefs[i+0]);
		s0->lexical = s1->lexical;
		s0->otype = s1->otype;
	}

	fmtinstall('A', Aconv);		// asm opcodes
	fmtinstall('P', Pconv);		// asm instruction
	fmtinstall('R', Rconv);		// interpreted register
	fmtinstall('Q', Qconv);		// interpreted etype
	fmtinstall('X', Xconv);		// interpreted etype

	fmtinstall('D', Dconv);		// addressed operand
	fmtinstall('C', Cconv);		// C type
}

vlong
convvtox(vlong v, int et)
{
	/* botch - do truncation conversion when energetic */
	return v;
}

/*
 * return !(op)
 * eg == <=> !=
 */
int
brcom(int a)
{
	switch(a) {
	case PBEQ:	return PBNE;
	case PBNE:	return PBEQ;
	case PBLT:	return PBGE;
	case PBGT:	return PBLE;
	case PBLE:	return PBGT;
	case PBGE:	return PBLT;
	case PBTRUE:	return PBFALSE;
	case PBFALSE:	return PBTRUE;
	}
	fatal("brcom: no com for %A\n", a);
	return PERROR;
}

/*
 * return reverse(op)
 * eg a op b <=> b r(op) a
 */
int
brrev(int a)
{
	switch(a) {
	case PBEQ:	return PBEQ;
	case PBNE:	return PBNE;
	case PBLT:	return PBGT;
	case PBGT:	return PBLT;
	case PBLE:	return PBGE;
	case PBGE:	return PBLE;
	}
	fatal("brcom: no rev for %A\n", a);
	return PERROR;
}

/*
 * codegen the address of the ith
 * element in the jth argument.
 */
void
fnparam(Node *t, int j, int i)
{
	Node *a, *f;

	switch(j) {
	default:
		fatal("fnparam: bad j");
	case 0:
		a = getthisx(t);
		break;
	case 1:
		a = getoutargx(t);
		break;
	case 2:
		a = getinargx(t);
		break;
	}

	f = a->type;
	while(i > 0) {
		f = f->down;
		i--;
	}
	if(f->etype != TFIELD)
		fatal("fnparam: not field");

	gopcode(PLOAD, PTADDR, a->nname);
	gopcode(PADDO, PTADDR, f->nname);
}

Sig*
lsort(Sig *l, int(*f)(Sig*, Sig*))
{
	Sig *l1, *l2, *le;

	if(l == 0 || l->link == 0)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = l2->link;
		if(l2 == 0)
			break;
		l2 = l2->link;
		if(l2 == 0)
			break;
		l1 = l1->link;
	}

	l2 = l1->link;
	l1->link = 0;
	l1 = lsort(l, f);
	l2 = lsort(l2, f);

	/* set up lead element */
	if((*f)(l1, l2) < 0) {
		l = l1;
		l1 = l1->link;
	} else {
		l = l2;
		l2 = l2->link;
	}
	le = l;

	for(;;) {
		if(l1 == 0) {
			while(l2) {
				le->link = l2;
				le = l2;
				l2 = l2->link;
			}
			le->link = 0;
			break;
		}
		if(l2 == 0) {
			while(l1) {
				le->link = l1;
				le = l1;
				l1 = l1->link;
			}
			break;
		}
		if((*f)(l1, l2) < 0) {
			le->link = l1;
			le = l1;
			l1 = l1->link;
		} else {
			le->link = l2;
			le = l2;
			l2 = l2->link;
		}
	}
	le->link = 0;
	return l;
}
