// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"y.tab.h"

void
errorexit(void)
{
	if(outfile)
		remove(outfile);
	myexit(1);
}

void
yyerror(char *fmt, ...)
{
	va_list arg;

	print("%L: ");
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	print("\n");
	if(debug['h'])
		*(int*)0 = 0;

	nerrors++;
	if(nerrors >= 10)
		fatal("too many errors");
}

void
warn(char *fmt, ...)
{
	va_list arg;

	print("%L: ");
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	print("\n");
	if(debug['h'])
		*(int*)0 = 0;
}

void
fatal(char *fmt, ...)
{
	va_list arg;

	print("%L: fatal error: ");
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	print("\n");
	if(debug['h'])
		*(int*)0 = 0;
	myexit(1);
}

void
linehist(char *file, long off)
{
	Hist *h;

	if(debug['i'])
	if(file != nil)
		print("%L: import %s\n", file);
	else
		print("%L: <eof>\n");

	h = alloc(sizeof(Hist));
	h->name = file;
	h->line = lineno;
	h->offset = off;
	h->link = H;
	if(ehist == H) {
		hist = h;
		ehist = h;
		return;
	}
	ehist->link = h;
	ehist = h;
}

ulong
stringhash(char *p)
{
	long h;
	int c;

	h = 0;
	for(;;) {
		c = *p++;
		if(c == 0)
			break;
		h = h*PRIME1 + c;
	}

	if(h < 0) {
		h = -h;
		if(h < 0)
			h = 0;
	}
	return h;
}

Sym*
lookup(char *p)
{
	Sym *s;
	ulong h;
	int c;

	h = stringhash(p) % NHASH;
	c = p[0];

	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != c)
			continue;
		if(strcmp(s->name, p) == 0)
			if(strcmp(s->package, package) == 0)
				return s;
	}

	s = mal(sizeof(*s));
	s->lexical = LNAME;
	s->name = mal(strlen(p)+1);
	s->opackage = package;
	s->package = package;

	strcpy(s->name, p);

	s->link = hash[h];
	hash[h] = s;

	return s;
}

Sym*
pkglookup(char *p, char *k)
{
	Sym *s;
	ulong h;
	int c;

	h = stringhash(p) % NHASH;
	c = p[0];
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != c)
			continue;
		if(strcmp(s->name, p) == 0)
			if(strcmp(s->package, k) == 0)
				return s;
	}

	s = mal(sizeof(*s));
	s->lexical = LNAME;
	s->name = mal(strlen(p)+1);
	strcpy(s->name, p);

	// botch - should probably try to reuse the pkg string
	s->package = mal(strlen(k)+1);
	s->opackage = s->package;
	strcpy(s->package, k);

	s->link = hash[h];
	hash[h] = s;

	return s;
}

void
gethunk(void)
{
	char *h;
	long nh;

	nh = NHUNK;
	if(thunk >= 10L*NHUNK)
		nh = 10L*NHUNK;
	h = (char*)malloc(nh);
	if(h == (char*)-1) {
		yyerror("out of memory");
		errorexit();
	}
	hunk = h;
	nhunk = nh;
	thunk += nh;
}

void*
mal(long n)
{
	void *p;

	while((ulong)hunk & MAXALIGN) {
		hunk++;
		nhunk--;
	}
	while(nhunk < n)
		gethunk();

	p = hunk;
	nhunk -= n;
	hunk += n;
	memset(p, 0, n);
	return p;
}

void*
remal(void *p, long on, long n)
{
	void *q;

	q = (uchar*)p + on;
	if(q != hunk || nhunk < n) {
		while(nhunk < on+n)
			gethunk();
		memmove(hunk, p, on);
		p = hunk;
		hunk += on;
		nhunk -= on;
	}
	hunk += n;
	nhunk -= n;
	return p;
}

Dcl*
dcl(void)
{
	Dcl *d;

	d = mal(sizeof(*d));
	d->lineno = dynlineno;
	return d;
}

Node*
nod(int op, Node *nleft, Node *nright)
{
	Node *n;

	n = mal(sizeof(*n));
	n->op = op;
	n->left = nleft;
	n->right = nright;
	n->lineno = dynlineno;
	if(dynlineno == 0)
		n->lineno = lineno;
	return n;
}

Node*
list(Node *a, Node *b)
{
	if(a == N)
		return b;
	if(b == N)
		return a;
	return nod(OLIST, a, b);
}

Type*
typ(int et)
{
	Type *t;

	t = mal(sizeof(*t));
	t->etype = et;
	return t;
}

Node*
dobad(void)
{
	return nod(OBAD, N, N);
}

Node*
nodintconst(long v)
{
	Node *c;

	c = nod(OLITERAL, N, N);
	c->addable = 1;
	c->val.vval = v;
	c->val.ctype = CTINT;
	c->type = types[TINT32];
	ullmancalc(c);
	return c;
}

Node*
rev(Node *na)
{
	Node *i, *n;

	/*
	 * since yacc wants to build lists
	 * stacked down on the left -
	 * this routine converts them to
	 * stack down on the right -
	 * in memory without recursion
	 */

	if(na == N || na->op != OLIST)
		return na;
	i = na;
	for(n = na->left; n != N; n = n->left) {
		if(n->op != OLIST)
			break;
		i->left = n->right;
		n->right = i;
		i = n;
	}
	i->left = n;
	return i;
}

Node*
unrev(Node *na)
{
	Node *i, *n;

	/*
	 * this restores a reverse list
	 */
	if(na == N || na->op != OLIST)
		return na;
	i = na;
	for(n = na->right; n != N; n = n->right) {
		if(n->op != OLIST)
			break;
		i->right = n->left;
		n->left = i;
		i = n;
	}
	i->right = n;
	return i;
}

Type*
aindex(Node *b, Type *t)
{
	Type *r;

	r = typ(TARRAY);
	r->type = t;

	if(t->etype == TDARRAY)
		yyerror("dynamic array type cannot be a dynamic array");

	walktype(b, Erv);
	switch(whatis(b)) {
	default:
		yyerror("array bound must be a constant integer expression");
		break;

	case Wnil:	// default zero lb
		r->bound = 0;
		break;

	case Wlitint:	// fixed lb
		r->bound = b->val.vval;
		break;
	}
	return r;
}

void
indent(int dep)
{
	int i;

	for(i=0; i<dep; i++)
		print(".   ");
}

void
dodump(Node *n, int dep)
{

loop:
	if(n == N)
		return;

	switch(n->op) {
	case OLIST:
		if(n->left != N && n->left->op == OLIST)
			dodump(n->left, dep+1);
		else
			dodump(n->left, dep);
		n = n->right;
		goto loop;

//	case ODCLFUNC:
//		dodump(n->nname, dep);
//		if(n->this) {
//			indent(dep);
//			print("%O-this\n", n->op);
//			dodump(n->this, dep+1);
//		}
//		if(n->argout) {
//			indent(dep);
//			print("%O-outarg\n", n->op);
//			dodump(n->argout, dep+1);
//		}
//		if(n->argin) {
//			indent(dep);
//			print("%O-inarg\n", n->op);
//			dodump(n->argin, dep+1);
//		}
//		n = n->nbody;
//		goto loop;

	case OIF:
	case OSWITCH:
	case OFOR:
		dodump(n->ninit, dep);
		break;
	}

	indent(dep);
	if(dep > 10) {
		print("...\n");
		return;
	}

	switch(n->op) {
	default:
		print("%N\n", n);
		break;

	case OTYPE:
		print("%O-%E %lT\n", n->op, n->etype, n);
		break;

	case OIF:
		print("%O%J\n", n->op, n);
		dodump(n->ntest, dep+1);
		if(n->nbody != N) {
			indent(dep);
			print("%O-then\n", n->op);
			dodump(n->nbody, dep+1);
		}
		if(n->nelse != N) {
			indent(dep);
			print("%O-else\n", n->op);
			dodump(n->nelse, dep+1);
		}
		return;

	case OSWITCH:
	case OFOR:
		print("%O%J\n", n->op, n);
		dodump(n->ntest, dep+1);

		if(n->nbody != N) {
			indent(dep);
			print("%O-body\n", n->op);
			dodump(n->nbody, dep+1);
		}

		if(n->nincr != N) {
			indent(dep);
			print("%O-incr\n", n->op);
			dodump(n->nincr, dep+1);
		}
		return;

	case OCASE:
		// the right side points to the next case
		print("%O%J\n", n->op, n);
		dodump(n->left, dep+1);
		return;
	}

	dodump(n->left, dep+1);
	n = n->right;
	dep++;
	goto loop;
}

void
dump(char *s, Node *n)
{
	print("%s\n", s);
	dodump(n, 1);
}

int
whatis(Node *n)
{
	Type *t;

	if(n == N)
		return Wnil;

	if(n->op == OLITERAL) {
		switch(n->val.ctype) {
		default:
			break;
		case CTINT:
		case CTSINT:
		case CTUINT:
			return Wlitint;
		case CTFLT:
			return Wlitfloat;
		case CTBOOL:
			return Wlitbool;
		case CTSTR:
			return Wlitstr;
		case CTNIL:
			return Wlitnil;	// not used
		}
		return Wtunkn;
	}

	t = n->type;
	if(t == T)
		return Wtnil;

	switch(t->etype) {
	case TINT8:
	case TINT16:
	case TINT32:
	case TINT64:
	case TUINT8:
	case TUINT16:
	case TUINT32:
	case TUINT64:
		return Wtint;
	case TFLOAT32:
	case TFLOAT64:
	case TFLOAT80:
		return Wtfloat;
	case TBOOL:
		return Wtbool;

	case TPTR32:
	case TPTR64:
		if(isptrto(t, TSTRING))
			return Wtstr;
		break;
	}
	return Wtunkn;
}

/*
s%,%,\n%g
s%\n+%\n%g
s%^[ 	]*O%%g
s%,.*%%g
s%.+%	[O&]		= "&",%g
s%^	........*\]%&~%g
s%~	%%g
*/

static char*
opnames[] =
{
	[OADDR]		= "ADDR",
	[OADD]		= "ADD",
	[OANDAND]	= "ANDAND",
	[OAND]		= "AND",
	[OARRAY]	= "ARRAY",
	[OASOP]		= "ASOP",
	[OAS]		= "AS",
	[OBAD]		= "BAD",
	[OBREAK]	= "BREAK",
	[OCALL]		= "CALL",
	[OCALLMETH]	= "CALLMETH",
	[OCALLINTER]	= "CALLINTER",
	[OCASE]		= "CASE",
	[OXCASE]	= "XCASE",
	[OCMP]		= "CMP",
	[OFALL]		= "FALL",
	[OCONV]		= "CONV",
	[OCOM]		= "COM",
	[OCONST]	= "CONST",
	[OCONTINUE]	= "CONTINUE",
	[ODCLARG]	= "DCLARG",
	[ODCLFIELD]	= "DCLFIELD",
	[ODCLFUNC]	= "DCLFUNC",
	[ODIV]		= "DIV",
	[ODOT]		= "DOT",
	[ODOTPTR]	= "DOTPTR",
	[ODOTMETH]	= "DOTMETH",
	[ODOTINTER]	= "DOTINTER",
	[OEMPTY]	= "EMPTY",
	[OEND]		= "END",
	[OEQ]		= "EQ",
	[OFOR]		= "FOR",
	[OFUNC]		= "FUNC",
	[OGE]		= "GE",
	[OPROC]		= "PROC",
	[OGOTO]		= "GOTO",
	[OGT]		= "GT",
	[OIF]		= "IF",
	[OINDEX]	= "INDEX",
	[OINDEXPTR]	= "INDEXPTR",
	[OIND]		= "IND",
	[OLABEL]	= "LABEL",
	[OLE]		= "LE",
	[OLEN]		= "LEN",
	[OLIST]		= "LIST",
	[OLITERAL]	= "LITERAL",
	[OLSH]		= "LSH",
	[OLT]		= "LT",
	[OMINUS]	= "MINUS",
	[OMOD]		= "MOD",
	[OMUL]		= "MUL",
	[ONAME]		= "NAME",
	[ONONAME]	= "NONAME",
	[ONE]		= "NE",
	[ONOT]		= "NOT",
	[OOROR]		= "OROR",
	[OOR]		= "OR",
	[OPLUS]		= "PLUS",
	[OREGISTER]	= "REGISTER",
	[OINDREG]	= "INDREG",
	[OSEND]		= "SEND",
	[ORECV]		= "RECV",
	[OPTR]		= "PTR",
	[ORETURN]	= "RETURN",
	[ORSH]		= "RSH",
	[OI2S]		= "I2S",
	[OS2I]		= "S2I",
	[OI2I]		= "I2I",
	[OSLICE]	= "SLICE",
	[OSUB]		= "SUB",
	[OSWITCH]	= "SWITCH",
	[OTYPE]		= "TYPE",
	[OVAR]		= "VAR",
	[OEXPORT]	= "EXPORT",
	[OIMPORT]	= "IMPORT",
	[OXOR]		= "XOR",
	[ONEW]		= "NEW",
	[OFALL]		= "FALL",
	[OXFALL]	= "XFALL",
	[OPANIC]	= "PANIC",
	[OPRINT]	= "PRINT",
	[OXXX]		= "XXX",
};

int
Oconv(Fmt *fp)
{
	char buf[500];
	int o;

	o = va_arg(fp->args, int);
	if(o < 0 || o >= nelem(opnames) || opnames[o] == nil) {
		snprint(buf, sizeof(buf), "O-%d", o);
		return fmtstrcpy(fp, buf);
	}
	return fmtstrcpy(fp, opnames[o]);
}

int
Lconv(Fmt *fp)
{
	char str[STRINGSZ], s[STRINGSZ];
	struct
	{
		Hist*	incl;	/* start of this include file */
		long	idel;	/* delta line number to apply to include */
		Hist*	line;	/* start of this #line directive */
		long	ldel;	/* delta line number to apply to #line */
	} a[HISTSZ];
	long lno, d;
	int i, n;
	Hist *h;

	lno = dynlineno;
	if(lno == 0)
		lno = lineno;

	n = 0;
	for(h=hist; h!=H; h=h->link) {
		if(lno < h->line)
			break;
		if(h->name) {
			if(n < HISTSZ) {	/* beginning of file */
				a[n].incl = h;
				a[n].idel = h->line;
				a[n].line = 0;
			}
			n++;
			continue;
		}
		n--;
		if(n > 0 && n < HISTSZ) {
			d = h->line - a[n].incl->line;
			a[n-1].ldel += d;
			a[n-1].idel += d;
		}
	}

	if(n > HISTSZ)
		n = HISTSZ;

	str[0] = 0;
	for(i=n-1; i>=0; i--) {
		if(i != n-1) {
			if(fp->flags & ~(FmtWidth|FmtPrec))
				break;
			strcat(str, " ");
		}
		if(a[i].line)
			snprint(s, STRINGSZ, "%s:%ld[%s:%ld]",
				a[i].line->name, lno-a[i].ldel+1,
				a[i].incl->name, lno-a[i].idel+1);
		else
			snprint(s, STRINGSZ, "%s:%ld",
				a[i].incl->name, lno-a[i].idel+1);
		if(strlen(s)+strlen(str) >= STRINGSZ-10)
			break;
		strcat(str, s);
		lno = a[i].incl->line - 1;	/* now print out start of this file */
	}
	if(n == 0)
		strcat(str, "<eof>");

	return fmtstrcpy(fp, str);
}

/*
s%,%,\n%g
s%\n+%\n%g
s%^[ 	]*T%%g
s%,.*%%g
s%.+%	[T&]		= "&",%g
s%^	........*\]%&~%g
s%~	%%g
*/

static char*
etnames[] =
{
	[TINT8]		= "INT8",
	[TUINT8]	= "UINT8",
	[TINT16]	= "INT16",
	[TUINT16]	= "UINT16",
	[TINT32]	= "INT32",
	[TUINT32]	= "UINT32",
	[TINT64]	= "INT64",
	[TUINT64]	= "UINT64",
	[TFLOAT32]	= "FLOAT32",
	[TFLOAT64]	= "FLOAT64",
	[TFLOAT80]	= "FLOAT80",
	[TBOOL]		= "BOOL",
	[TPTR32]	= "PTR32",
	[TPTR64]	= "PTR64",
	[TFUNC]		= "FUNC",
	[TARRAY]	= "ARRAY",
	[TDARRAY]	= "DARRAY",
	[TSTRUCT]	= "STRUCT",
	[TCHAN]		= "CHAN",
	[TMAP]		= "MAP",
	[TINTER]	= "INTER",
	[TFORW]		= "FORW",
	[TFIELD]	= "FIELD",
	[TSTRING]	= "STRING",
	[TCHAN]		= "CHAN",
	[TANY]		= "ANY",
};

int
Econv(Fmt *fp)
{
	char buf[500];
	int et;

	et = va_arg(fp->args, int);
	if(et < 0 || et >= nelem(etnames) || etnames[et] == nil) {
		snprint(buf, sizeof(buf), "E-%d", et);
		return fmtstrcpy(fp, buf);
	}
	return fmtstrcpy(fp, etnames[et]);
}

int
Jconv(Fmt *fp)
{
	char buf[500], buf1[100];
	Node *n;

	n = va_arg(fp->args, Node*);
	strcpy(buf, "");

	if(n->ullman != 0) {
		snprint(buf1, sizeof(buf1), " u(%d)", n->ullman);
		strncat(buf, buf1, sizeof(buf));
	}

	if(n->addable != 0) {
		snprint(buf1, sizeof(buf1), " a(%d)", n->addable);
		strncat(buf, buf1, sizeof(buf));
	}

	if(n->vargen != 0) {
		snprint(buf1, sizeof(buf1), " g(%ld)", n->vargen);
		strncat(buf, buf1, sizeof(buf));
	}

	if(n->lineno != 0) {
		snprint(buf1, sizeof(buf1), " l(%ld)", n->lineno);
		strncat(buf, buf1, sizeof(buf));
	}

	return fmtstrcpy(fp, buf);
}

int
Gconv(Fmt *fp)
{
	char buf[100];
	Type *t;

	t = va_arg(fp->args, Type*);

	if(t->etype == TFUNC) {
		if(t->vargen != 0) {
			snprint(buf, sizeof(buf), "-%d%d%d g(%ld)",
				t->thistuple, t->outtuple, t->intuple, t->vargen);
			goto out;
		}
		snprint(buf, sizeof(buf), "-%d%d%d",
			t->thistuple, t->outtuple, t->intuple);
		goto out;
	}
	if(t->vargen != 0) {
		snprint(buf, sizeof(buf), " g(%ld)", t->vargen);
		goto out;
	}
	strcpy(buf, "");

out:
	return fmtstrcpy(fp, buf);
}

int
Sconv(Fmt *fp)
{
	char buf[500];
	Sym *s;
	char *opk, *pkg, *nam;

	s = va_arg(fp->args, Sym*);
	if(s == S) {
		snprint(buf, sizeof(buf), "<S>");
		goto out;
	}

	pkg = "<nil>";
	nam = pkg;
	opk = pkg;

	if(s->opackage != nil)
		opk = s->opackage;
	if(s->package != nil)
		pkg = s->package;
	if(s->name != nil)
		nam = s->name;

	if(strcmp(pkg, package) || strcmp(opk, package) || (fp->flags & FmtLong)) {
		if(strcmp(opk, pkg) == 0) {
			snprint(buf, sizeof(buf), "%s.%s", pkg, nam);
			goto out;
		}
		snprint(buf, sizeof(buf), "(%s)%s.%s", opk, pkg, nam);
		goto out;
	}
	snprint(buf, sizeof(buf), "%s", nam);

out:
	return fmtstrcpy(fp, buf);
}

int
Tconv(Fmt *fp)
{
	char buf[500], buf1[500];
	Type *t, *t1;
	int et;

	t = va_arg(fp->args, Type*);
	if(t == T)
		return fmtstrcpy(fp, "<T>");

	t->trecur++;
	et = t->etype;

	strcpy(buf, "");
	if(t->sym != S) {
		if(t->sym->name[0] != '_')
		snprint(buf, sizeof(buf), "<%S>", t->sym);
	}
	if(t->trecur > 5) {
		strncat(buf, "...", sizeof(buf));
		goto out;
	}

	switch(et) {
	default:
		snprint(buf1, sizeof(buf1), "%E", et);
		strncat(buf, buf1, sizeof(buf));
		if(t->type != T) {
			snprint(buf1, sizeof(buf1), " %T", t->type);
			strncat(buf, buf1, sizeof(buf));
		}
		break;

	case TFIELD:
		snprint(buf1, sizeof(buf1), "%T", t->type);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TFUNC:
		if(fp->flags & FmtLong)
			snprint(buf1, sizeof(buf1), "%d%d%d(%lT,%lT)%lT",
				t->thistuple, t->intuple, t->outtuple,
				t->type, t->type->down->down, t->type->down);
		else
			snprint(buf1, sizeof(buf1), "%d%d%d(%T,%T)%T",
				t->thistuple, t->intuple, t->outtuple,
				t->type, t->type->down->down, t->type->down);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TINTER:
		strncat(buf, "I{", sizeof(buf));
		if(fp->flags & FmtLong) {
			for(t1=t->type; t1!=T; t1=t1->down) {
				snprint(buf1, sizeof(buf1), "%lT;", t1);
				strncat(buf, buf1, sizeof(buf));
			}
		}
		strncat(buf, "}", sizeof(buf));
		break;

	case TSTRUCT:
		strncat(buf, "{", sizeof(buf));
		if(fp->flags & FmtLong) {
			for(t1=t->type; t1!=T; t1=t1->down) {
				snprint(buf1, sizeof(buf1), "%lT;", t1);
				strncat(buf, buf1, sizeof(buf));
			}
		}
		strncat(buf, "}", sizeof(buf));
		break;

	case TMAP:
		snprint(buf, sizeof(buf), "MAP[%T]%T", t->down, t->type);
		break;

	case TARRAY:
		snprint(buf1, sizeof(buf1), "[%ld]%T", t->bound, t->type);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TDARRAY:
		snprint(buf1, sizeof(buf1), "[]%T", t->type);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TPTR32:
	case TPTR64:
		snprint(buf1, sizeof(buf1), "*%T", t->type);
		strncat(buf, buf1, sizeof(buf));
		break;
	}

out:
	t->trecur--;
	return fmtstrcpy(fp, buf);
}

int
Nconv(Fmt *fp)
{
	char buf[500], buf1[500];
	Node *n;

	n = va_arg(fp->args, Node*);
	if(n == N) {
		snprint(buf, sizeof(buf), "<N>");
		goto out;
	}

	switch(n->op) {
	default:
		snprint(buf, sizeof(buf), "%O%J", n->op, n);
		break;

	case ONAME:
	case ONONAME:
		if(n->sym == S) {
			snprint(buf, sizeof(buf), "%O%J", n->op, n);
			break;
		}
		snprint(buf, sizeof(buf), "%O-%S G%ld%J", n->op,
			n->sym, n->sym->vargen, n);
		goto ptyp;

	case OREGISTER:
		snprint(buf, sizeof(buf), "%O-%R%J", n->op, (int)n->val.vval, n);
		break;

	case OLITERAL:
		switch(n->val.ctype) {
		default:
			snprint(buf1, sizeof(buf1), "LITERAL-ctype=%d%lld", n->val.ctype, n->val.vval);
			break;
		case CTINT:
			snprint(buf1, sizeof(buf1), "I%lld", n->val.vval);
			break;
		case CTSINT:
			snprint(buf1, sizeof(buf1), "S%lld", n->val.vval);
			break;
		case CTUINT:
			snprint(buf1, sizeof(buf1), "U%lld", n->val.vval);
			break;
		case CTFLT:
			snprint(buf1, sizeof(buf1), "F%g", n->val.dval);
			break;
		case CTSTR:
			snprint(buf1, sizeof(buf1), "S\"%Z\"", n->val.sval);
			break;
		case CTBOOL:
			snprint(buf1, sizeof(buf1), "B%lld", n->val.vval);
			break;
		case CTNIL:
			snprint(buf1, sizeof(buf1), "N");
			break;
		}
		snprint(buf, sizeof(buf), "%O-%s%J", n->op, buf1, n);
		break;
		
	case OASOP:
		snprint(buf, sizeof(buf), "%O-%O%J", n->op, n->etype, n);
		break;

	case OTYPE:
		snprint(buf, sizeof(buf), "%O-%E%J", n->op, n->etype, n);
		break;
	}
	if(n->sym != S) {
		snprint(buf1, sizeof(buf1), " %S G%ld", n->sym, n->sym->vargen);
		strncat(buf, buf1, sizeof(buf));
	}

ptyp:
	if(n->type != T) {
		snprint(buf1, sizeof(buf1), " %T", n->type);
		strncat(buf, buf1, sizeof(buf));
	}

out:
	return fmtstrcpy(fp, buf);
}

int
Zconv(Fmt *fp)
{
	char *s, *se;
	char *p;
	char buf[500];
	int c;
	String *sp;

	sp = va_arg(fp->args, String*);
	if(sp == nil) {
		snprint(buf, sizeof(buf), "<nil>");
		goto out;
	}
	s = sp->s;
	se = s + sp->len;

	p = buf;

loop:
	c = *s++;
	if(s > se)
		c = 0;
	switch(c) {
	default:
		*p++ = c;
		break;
	case 0:
		*p = 0;
		goto out;
	case '\t':
		*p++ = '\\';
		*p++ = 't';
		break;
	case '\n':
		*p++ = '\\';
		*p++ = 'n';
		break;
	}
	goto loop;	

out:
	return fmtstrcpy(fp, buf);
}

int
isnil(Node *n)
{
	if(n == N)
		return 0;
	if(n->op != OLITERAL)
		return 0;
	if(n->val.ctype != CTNIL)
		return 0;
	return 1;
}

int
isptrto(Type *t, int et)
{
	if(t == T)
		return 0;
	if(!isptr[t->etype])
		return 0;
	t = t->type;
	if(t == T)
		return 0;
	if(t->etype != et)
		return 0;
	return 1;
}

int
isinter(Type *t)
{
	if(t != T && t->etype == TINTER)
		return 1;
	return 0;
}

int
isbytearray(Type *t)
{
	if(t == T)
		return 0;
	if(isptr[t->etype]) {
		t = t->type;
		if(t == T)
			return 0;
	}
	if(t->etype != TARRAY)
		return 0;
	return t->bound+1;
}

int
eqtype(Type *t1, Type *t2, int d)
{
	if(d >= 10)
		return 1;

	if(t1 == t2)
		return 1;
	if(t1 == T || t2 == T)
		return 0;

	if(t1->etype != t2->etype)
		return 0;

	switch(t1->etype) {
	case TINTER:
	case TSTRUCT:
		t1 = t1->type;
		t2 = t2->type;
		for(;;) {
			if(!eqtype(t1, t2, 0))
				return 0;
			if(t1 == T)
				return 1;
			if(t1->nname != N && t1->nname->sym != S) {
				if(t2->nname == N || t2->nname->sym == S)
					return 0;
				if(strcmp(t1->nname->sym->name, t2->nname->sym->name) != 0) {
					// assigned names dont count
					if(t1->nname->sym->name[0] != '_' ||
				   	   t2->nname->sym->name[0] != '_')
						return 0;
				}
			}
			t1 = t1->down;
			t2 = t2->down;
		}
		return 1;

	case TFUNC:
		t1 = t1->type;
		t2 = t2->type;
		for(;;) {
			if(t1 == t2)
				break;
			if(t1 == T || t2 == T)
				return 0;
			if(t1->etype != TSTRUCT || t2->etype != TSTRUCT)
				return 0;

			if(!eqtype(t1->type, t2->type, 0))
				return 0;

			t1 = t1->down;
			t2 = t2->down;
		}
		return 1;
	}
	return eqtype(t1->type, t2->type, d+1);
}

static int
subtype(Type **stp, Type *t)
{
	Type *st;

loop:
	st = *stp;
	if(st == T)
		return 0;
	switch(st->etype) {
	default:
		return 0;

	case TPTR32:
	case TPTR64:
		stp = &st->type;
		goto loop;

	case TANY:
		*stp = t;
		break;

	case TMAP:
		if(subtype(&st->down, t))
			break;
		stp = &st->type;
		goto loop;

	case TFUNC:
		for(;;) {
			if(subtype(&st->type, t))
				break;
			if(subtype(&st->type->down->down, t))
				break;
			if(subtype(&st->type->down, t))
				break;
			return 0;
		}
		break;

	case TSTRUCT:
		for(st=st->type; st!=T; st=st->down)
			if(subtype(&st->type, t))
				return 1;
		return 0;
	}
	return 1;
}

void
argtype(Node *on, Type *t)
{
	if(!subtype(&on->type, t))
		fatal("argtype: failed %N %T\n", on, t);
}

Type*
shallow(Type *t)
{
	Type *nt;

	if(t == T)
		return T;
	nt = typ(0);
	*nt = *t;
	return nt;
}

Type*
deep(Type *t)
{
	Type *nt, *xt;

	if(t == T)
		return T;

	switch(t->etype) {
	default:
		nt = t;	// share from here down
		break;

	case TPTR32:
	case TPTR64:
		nt = shallow(t);
		nt->type = deep(t->type);
		break;

	case TMAP:
		nt = shallow(t);
		nt->down = deep(t->down);
		nt->type = deep(t->type);
		break;

	case TFUNC:
		nt = shallow(t);
		nt->type = deep(t->type);
		nt->type->down = deep(t->type->down);
		nt->type->down->down = deep(t->type->down->down);
		break;

	case TSTRUCT:
		nt = shallow(t);
		nt->type = shallow(t->type);
		xt = nt->type;

		for(t=t->type; t!=T; t=t->down) {
			xt->type = deep(t->type);
			xt->down = shallow(t->down);
			xt = xt->down;
		}
		break;
	}
	return nt;
}

Node*
syslook(char *name, int copy)
{
	Sym *s;
	Node *n;

	s = pkglookup(name, "sys");
	if(s == S || s->oname == N)
		fatal("looksys: cant find sys.%s", name);

	if(!copy)
		return s->oname;

	n = nod(0, N, N);
	*n = *s->oname;
	n->type = deep(s->oname->type);

	return n;
}

/*
 * are the arg names of two
 * functions the same. we know
 * that eqtype has been called
 * and has returned true.
 */
int
eqargs(Type *t1, Type *t2)
{
	if(t1 == t2)
		return 1;
	if(t1 == T || t2 == T)
		return 0;

	if(t1->etype != t2->etype)
		return 0;

	if(t1->etype != TFUNC)
		fatal("eqargs: oops %E", t1->etype);

	t1 = t1->type;
	t2 = t2->type;
	for(;;) {
		if(t1 == t2)
			break;
		if(!eqtype(t1, t2, 0))
			return 0;
		t1 = t1->down;
		t2 = t2->down;
	}
	return 1;
}

ulong
typehash(Type *at, int d)
{
	ulong h;
	Type *t;

	if(at == T)
		return PRIME2;
	if(d >= 5)
		return PRIME3;

	if(at->recur)
		return 0;
	at->recur = 1;

	h = at->etype*PRIME4;

	switch(at->etype) {
	default:
		h += PRIME5 * typehash(at->type, d+1);
		break;

	case TINTER:
		// botch -- should be sorted?
		for(t=at->type; t!=T; t=t->down)
			h += PRIME6 * typehash(t, d+1);
		break;

	case TSTRUCT:
		for(t=at->type; t!=T; t=t->down)
			h += PRIME7 * typehash(t, d+1);
		break;

	case TFUNC:
		t = at->type;
		// skip this argument
		if(t != T)
			t = t->down;
		for(; t!=T; t=t->down)
			h += PRIME7 * typehash(t, d+1);
		break;
	}

	at->recur = 0;
	return h;
}

Type*
ptrto(Type *t)
{
	Type *t1;

	if(tptr == 0)
		fatal("ptrto: nil");
	t1 = typ(tptr);
	t1->type = t;
	return t1;
}

Node*
literal(long v)
{
	Node *n;

	n = nod(OLITERAL, N, N);
	n->val.ctype = CTINT;
	n->val.vval = v;
	return n;
}

void
frame(int context)
{
	char *p;
	Dcl *d;
	int flag;

	p = "stack";
	d = autodcl;
	if(context) {
		p = "external";
		d = externdcl;
	}

	flag = 1;
	for(; d!=D; d=d->forw) {
		switch(d->op) {
		case ONAME:
			if(flag)
				print("--- %s frame ---\n", p);
			print("%O %S G%ld T\n", d->op, d->dsym, d->dnode->vargen, d->dnode->type);
			flag = 0;
			break;

		case OTYPE:
			if(flag)
				print("--- %s frame ---\n", p);
			print("%O %lT\n", d->op, d->dnode);
			flag = 0;
			break;
		}
	}
}

/*
 * calculate sethi/ullman number
 * roughly how many registers needed to
 * compile a node. used to compile the
 * hardest side first to minimize registers.
 */
void
ullmancalc(Node *n)
{
	int ul, ur;

	if(n == N)
		return;

	switch(n->op) {
	case OLITERAL:
	case ONAME:
		ul = 0;
		goto out;
	case OS2I:
	case OI2S:
	case OI2I:
	case OCALL:
	case OCALLMETH:
	case OCALLINTER:
		ul = UINF;
		goto out;
	}
	ul = 0;
	if(n->left != N)
		ul = n->left->ullman;
	ur = 0;
	if(n->right != N)
		ur = n->right->ullman;
	if(ul == ur)
		ul += 1;
	if(ur > ul)
		ul = ur;

out:
	n->ullman = ul;
}

void
badtype(int o, Type *tl, Type *tr)
{

loop:
	switch(o) {
	case OCALL:
		if(tl == T || tr == T)
			break;
		if(isptr[tl->etype] && isptr[tr->etype]) {
			tl = tl->type;
			tr = tr->type;
			goto loop;
		}
		if(tl->etype != TFUNC || tr->etype != TFUNC)
			break;
//		if(eqtype(t1, t2, 0))
	}

	yyerror("illegal types for operand: %O", o);
	if(tl != T)
		print("	(%lT)\n", tl);
	if(tr != T)
		print("	(%lT)\n", tr);
}

/*
 * this routine gets the parsing of
 * a parameter list that can have
 * name, type and name-type.
 * it must distribute lone names
 * with trailing types to give every
 * name a type. (a,b,c int) comes out
 * (a int, b int, c int).
 */
Node*
cleanidlist(Node *r)
{
	Node *t, *n, *nn, *l;
	Type *dt;

	t = N;		// untyped name
	nn = r;		// next node to take

loop:
	n = nn;
	if(n == N) {
		if(t != N) {
			yyerror("syntax error in parameter list");
			dt = types[TINT32];
			goto distrib;
		}
		return r;
	}

	l = n;
	nn = N;
	if(l->op == OLIST) {
		nn = l->right;
		l = l->left;
	}

	if(l->op != ODCLFIELD)
		fatal("cleanformal: %O", n->op);

	if(l->type == T) {
		if(t == N)
			t = n;
		goto loop;
	}

	if(t == N)
		goto loop;

	dt = l->type;	// type to be distributed

distrib:
	while(t != n) {
		if(t->op != OLIST) {
			if(t->type == T)
				t->type = dt;
			break;
		}
		if(t->left->type == T)
			t->left->type = dt;
		t = t->right;
	}

	t = N;
	goto loop;
}

/*
 * iterator to walk a structure declaration
 */
Type*
structfirst(Iter *s, Type **nn)
{
	Type *n, *t;

	n = *nn;
	if(n == T)
		goto bad;

	switch(n->etype) {
	default:
		goto bad;

	case TSTRUCT:
	case TINTER:
	case TFUNC:
		break;
	}

	t = n->type;
	if(t == T)
		goto rnil;

	if(t->etype != TFIELD)
		fatal("structfirst: not field %T", t);

	s->t = t;
	return t;

bad:
	fatal("structfirst: not struct %T", n);

rnil:
	return T;
}

Type*
structnext(Iter *s)
{
	Type *n, *t;

	n = s->t;
	t = n->down;
	if(t == T)
		goto rnil;

	if(t->etype != TFIELD)
		goto bad;

	s->t = t;
	return t;

bad:
	fatal("structnext: not struct %T", n);

rnil:
	return T;
}

/*
 * iterator to this and inargs in a function
 */
Type*
funcfirst(Iter *s, Type *t)
{
	Type *fp;

	if(t == T)
		goto bad;

	if(t->etype != TFUNC)
		goto bad;

	s->tfunc = t;
	s->done = 0;
	fp = structfirst(s, getthis(t));
	if(fp == T) {
		s->done = 1;
		fp = structfirst(s, getinarg(t));
	}
	return fp;

bad:
	fatal("funcfirst: not func %T", t);
	return T;
}

Type*
funcnext(Iter *s)
{
	Type *fp;

	fp = structnext(s);
	if(fp == T && !s->done) {
		s->done = 1;
		fp = structfirst(s, getinarg(s->tfunc));
	}
	return fp;
}

/*
 * iterator to walk a list
 */
Node*
listfirst(Iter *s, Node **nn)
{
	Node *n;

	n = *nn;
	if(n == N) {
		s->done = 1;
		s->an = &s->n;
		s->n = N;
		return N;
	}

	if(n->op == OLIST) {
		s->done = 0;
		s->n = n;
		s->an = &n->left;
		return n->left;
	}

	s->done = 1;
	s->an = nn;
	return n;
}

Node*
listnext(Iter *s)
{
	Node *n, *r;

	if(s->done) {
		s->an = &s->n;
		s->n = N;
		return N;
	}

	n = s->n;
	r = n->right;
	if(r == N) {
		s->an = &s->n;
		s->n = N;
		return N;
	}
	if(r->op == OLIST) {
		s->n = r;
		s->an = &r->left;
		return r->left;
	}

	s->done = 1;
	s->an = &n->right;
	return n->right;
}

Type**
getthis(Type *t)
{
	if(t->etype != TFUNC)
		fatal("getthis: not a func %N", t);
	return &t->type;
}

Type**
getoutarg(Type *t)
{
	if(t->etype != TFUNC)
		fatal("getoutarg: not a func %N", t);
	return &t->type->down;
}

Type**
getinarg(Type *t)
{
	if(t->etype != TFUNC)
		fatal("getinarg: not a func %N", t);
	return &t->type->down->down;
}

Type*
getthisx(Type *t)
{
	return *getthis(t);
}

Type*
getoutargx(Type *t)
{
	return *getoutarg(t);
}

Type*
getinargx(Type *t)
{
	return *getinarg(t);
}
