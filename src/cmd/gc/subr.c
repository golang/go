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

	print("%L: ", lineno);
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	if(strcmp(fmt, "syntax error") == 0)
		print(" near %s", namebuf);
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

	print("%L: ", lineno);
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

	print("%L: fatal error: ", lineno);
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	print("\n");
	if(debug['h'])
		*(int*)0 = 0;
	myexit(1);
}

void
linehist(char *file, int32 off)
{
	Hist *h;
	char *cp;

	if(debug['i']) {
		if(file != nil) {
			if(off < 0)
				print("pragma %s at line %L\n", file, lineno);
			else
				print("import %s at line %L\n", file, lineno);
		} else
			print("end of import at line %L\n", lineno);
	}

	if(off < 0 && file[0] != '/') {
		cp = mal(strlen(file) + strlen(pathname) + 2);
		sprint(cp, "%s/%s", pathname, file);
		file = cp;
	}

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

int32
setlineno(Node *n)
{
	int32 lno;

	lno = lineno;
	if(n != N && n->op != ONAME) {
		lineno = n->lineno;
		if(lineno == 0) {
			if(debug['K'])
				warn("setlineno: line 0");
			lineno = lno;
		}
	}
	return lno;
}

uint32
stringhash(char *p)
{
	int32 h;
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
	uint32 h;
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
	uint32 h;
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
	int32 nh;

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
mal(int32 n)
{
	void *p;

	while((uintptr)hunk & MAXALIGN) {
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
remal(void *p, int32 on, int32 n)
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
	d->lineno = lineno;
	return d;
}

extern int yychar;
Node*
nod(int op, Node *nleft, Node *nright)
{
	Node *n;

	n = mal(sizeof(*n));
	n->op = op;
	n->left = nleft;
	n->right = nright;
	if(yychar <= 0)	// no lookahead
		n->lineno = lineno;
	else
		n->lineno = prevlineno;
	return n;
}

int
algtype(Type *t)
{
	int a;

	a = AUNK;
	if(issimple[t->etype])
		a = ASIMP;	// simple mem
	else
	if(isptrto(t, TSTRING))
		a = ASTRING;	// string
	else
	if(isptr[t->etype])
		a = APTR;	// pointer
	else
	if(t->etype == TARRAY && t->bound < 0)
		a = ASLICE;
	else
	if(t->etype == TSTRUCT)
		a = ASTRUCT;
	else
	if(isinter(t))
		a = AINTER;	// interface
//	else
//		fatal("algtype: cant find type %T", t);
	return a;
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
nodintconst(int32 v)
{
	Node *c;

	c = nod(OLITERAL, N, N);
	c->addable = 1;
	c->val.u.xval = mal(sizeof(*c->val.u.xval));
	mpmovecfix(c->val.u.xval, v);
	c->val.ctype = CTINT;
	c->type = types[TINT];
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

/*
 * na and nb are reversed lists.
 * append them into one big reversed list.
 */
Node*
appendr(Node *na, Node *nb)
{
	Node **l, *n;

	for(l=&nb; (n=*l)->op == OLIST; l=&n->left)
		;
	*l = nod(OLIST, na, *l);
	return nb;
}

Type*
aindex(Node *b, Type *t)
{
	Type *r;
	int bound;

	bound = -1;	// open bound
	walktype(b, Erv);
	switch(whatis(b)) {
	default:	// variable bound
		yyerror("array bound must be an integer expression");
		break;

	case Wnil:	// open bound
		break;

	case Wlitint:	// fixed bound
		bound = mpgetfix(b->val.u.xval);
		if(bound < 0)
			yyerror("array bound must be non negative");
		break;
	}

	// fixed array
	r = typ(TARRAY);
	r->type = t;
	r->bound = bound;
	checkwidth(r);
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
	}

	indent(dep);
	if(dep > 10) {
		print("...\n");
		return;
	}

	if(n->ninit != N) {
		print("%O-init\n", n->op);
		dodump(n->ninit, dep+1);
		indent(dep);
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

	case OSELECT:
		print("%O%J\n", n->op, n);
		dodump(n->nbody, dep+1);
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
	case TINT:
	case TINT8:
	case TINT16:
	case TINT32:
	case TINT64:
	case TUINT:
	case TUINT8:
	case TUINT16:
	case TUINT32:
	case TUINT64:
	case TUINTPTR:
		return Wtint;
	case TFLOAT:
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
	[OCOMP]		= "COMP",
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
	[OKEY]		= "KEY",
	[OLABEL]	= "LABEL",
	[OLE]		= "LE",
	[OLEN]		= "LEN",
	[OCAP]		= "CAP",
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
	[ORANGE]	= "RANGE",
	[ORECV]		= "RECV",
	[OPTR]		= "PTR",
	[ORETURN]	= "RETURN",
	[ORSH]		= "RSH",
	[OSLICE]	= "SLICE",
	[OSUB]		= "SUB",
	[OSELECT]	= "SELECT",
	[OSWITCH]	= "SWITCH",
	[OTYPE]		= "TYPE",
	[OVAR]		= "VAR",
	[OIMPORT]	= "IMPORT",
	[OXOR]		= "XOR",
	[OMAKE]		= "MAKE",
	[ONEW]		= "NEW",
	[OFALL]		= "FALL",
	[OXFALL]	= "XFALL",
	[OPANIC]	= "PANIC",
	[OPANICN]	= "PANICN",
	[OPRINT]	= "PRINT",
	[OPRINTN]	= "PRINTN",
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
		int32	idel;	/* delta line number to apply to include */
		Hist*	line;	/* start of this #line directive */
		int32	ldel;	/* delta line number to apply to #line */
	} a[HISTSZ];
	int32 lno, d;
	int i, n;
	Hist *h;

	lno = va_arg(fp->args, int32);

	n = 0;
	for(h=hist; h!=H; h=h->link) {
		if(h->offset < 0)
			continue;
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
		strcat(str, "<epoch>");

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
	[TINT]		= "INT",
	[TUINT]		= "UINT",
	[TINT8]		= "INT8",
	[TUINT8]	= "UINT8",
	[TINT16]	= "INT16",
	[TUINT16]	= "UINT16",
	[TINT32]	= "INT32",
	[TUINT32]	= "UINT32",
	[TINT64]	= "INT64",
	[TUINT64]	= "UINT64",
	[TUINTPTR]	= "UINTPTR",
	[TFLOAT]	= "FLOAT",
	[TFLOAT32]	= "FLOAT32",
	[TFLOAT64]	= "FLOAT64",
	[TFLOAT80]	= "FLOAT80",
	[TBOOL]		= "BOOL",
	[TPTR32]	= "PTR32",
	[TPTR64]	= "PTR64",
	[TDDD]		= "DDD",
	[TFUNC]		= "FUNC",
	[TARRAY]	= "ARRAY",
//	[TDARRAY]	= "DARRAY",
	[TSTRUCT]	= "STRUCT",
	[TCHAN]		= "CHAN",
	[TMAP]		= "MAP",
	[TINTER]	= "INTER",
	[TFORW]		= "FORW",
	[TFIELD]	= "FIELD",
	[TSTRING]	= "STRING",
	[TCHAN]		= "CHAN",
	[TANY]		= "ANY",
	[TFORWINTER]	= "FORWINTER",
	[TFORWSTRUCT]	= "FORWSTRUCT",
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
	Sym *s;
	char *opk, *pkg, *nam;

	s = va_arg(fp->args, Sym*);
	if(s == S) {
		fmtstrcpy(fp, "<S>");
		return 0;
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

	if(!(fp->flags & FmtShort))
	if(strcmp(opk, package) != 0 || (fp->flags & FmtLong)) {
		fmtprint(fp, "%s.%s", opk, nam);
		return 0;
	}
	fmtstrcpy(fp, nam);
	return 0;
}

static char*
basicnames[] =
{
	[TINT]		= "int",
	[TUINT]		= "uint",
	[TINT8]		= "int8",
	[TUINT8]	= "uint8",
	[TINT16]	= "int16",
	[TUINT16]	= "uint16",
	[TINT32]	= "int32",
	[TUINT32]	= "uint32",
	[TINT64]	= "int64",
	[TUINT64]	= "uint64",
	[TUINTPTR]	= "uintptr",
	[TFLOAT]	= "float",
	[TFLOAT32]	= "float32",
	[TFLOAT64]	= "float64",
	[TFLOAT80]	= "float80",
	[TBOOL]		= "bool",
	[TANY]		= "any",
	[TDDD]		= "...",
};

int
Tpretty(Fmt *fp, Type *t)
{
	Type *t1;
	Sym *s;

	if(t->etype != TFIELD
	&& t->sym != S
	&& !(fp->flags&FmtLong)) {
		s = t->sym;
		if(t == types[t->etype] || t == types[TSTRING])
			return fmtprint(fp, "%s", s->name);
		if(exporting) {
			if(fp->flags & FmtShort)
				fmtprint(fp, "%hS", s);
			else
				fmtprint(fp, "%lS", s);
			if(strcmp(s->opackage, package) == 0)
			if(s->otype != t || (!s->export && !s->imported)) {
				fmtprint(fp, "·%s", filename);
				if(t->vargen)
					fmtprint(fp, "·%d", t->vargen);
			}
			return 0;
		}
		return fmtprint(fp, "%S", s);
	}

	if(t->etype < nelem(basicnames) && basicnames[t->etype] != nil)
		return fmtprint(fp, "%s", basicnames[t->etype]);

	switch(t->etype) {
	case TPTR32:
	case TPTR64:
		t1 = t->type;
		if(t1 != T) {
			switch(t1->etype) {
			case TSTRING:
				return fmtprint(fp, "string");
			case TMAP:
				return fmtprint(fp, "map[%T] %T", t1->down, t1->type);
			case TCHAN:
				return fmtprint(fp, "chan %T", t1->type);
			}
		}
		if(fp->flags&FmtShort)	// pass flag thru for methodsym
			return fmtprint(fp, "*%hT", t1);
		return fmtprint(fp, "*%T", t1);

	// Should not see these: should see ptr instead, handled above.
	case TSTRING:
		return fmtprint(fp, "STRING", t->type);
	case TCHAN:
		return fmtprint(fp, "CHAN %T", t->type);
	case TMAP:
		return fmtprint(fp, "MAP[%T] %T", t->down, t->type);

	case TFUNC:
		// t->type is method struct
		// t->type->down is result struct
		// t->type->down->down is arg struct
		if(t->thistuple && !(fp->flags&FmtSharp) && !(fp->flags&FmtShort)) {
			fmtprint(fp, "method(");
			for(t1=getthisx(t)->type; t1; t1=t1->down) {
				fmtprint(fp, "%T", t1);
				if(t1->down)
					fmtprint(fp, ", ");
			}
			fmtprint(fp, ")");
		}

		fmtprint(fp, "(");
		for(t1=getinargx(t)->type; t1; t1=t1->down) {
			fmtprint(fp, "%T", t1);
			if(t1->down)
				fmtprint(fp, ", ");
		}
		fmtprint(fp, ")");
		switch(t->outtuple) {
		case 0:
			break;
		case 1:
			t1 = getoutargx(t)->type;
			if(t1->etype != TFIELD) {
				fmtprint(fp, " %T", t1);
				break;
			}
		default:
			t1 = getoutargx(t)->type;
			fmtprint(fp, " (");
			for(; t1; t1=t1->down) {
				fmtprint(fp, "%T", t1);
				if(t1->down)
					fmtprint(fp, ", ");
			}
			fmtprint(fp, ")");
			break;
		}
		return 0;

	case TARRAY:
		if(t->bound >= 0)
			return fmtprint(fp, "[%d]%T", (int)t->bound, t->type);
		return fmtprint(fp, "[]%T", t->type);

	case TINTER:
		fmtprint(fp, "interface {");
		for(t1=t->type; t1!=T; t1=t1->down) {
			fmtprint(fp, " %hS %hT", t1->sym, t1->type);
			if(t1->down)
				fmtprint(fp, ";");
		}
		return fmtprint(fp, " }");

	case TSTRUCT:
		fmtprint(fp, "struct {");
		for(t1=t->type; t1!=T; t1=t1->down) {
			fmtprint(fp, " %T", t1);
			if(t1->down)
				fmtprint(fp, ";");
		}
		return fmtprint(fp, " }");

	case TFIELD:
		if(t->sym == S || t->embedded) {
			if(exporting)
				fmtprint(fp, "? ");
			fmtprint(fp, "%T", t->type);
		} else
			fmtprint(fp, "%hS %T", t->sym, t->type);
		if(t->note)
			fmtprint(fp, " \"%Z\"", t->note);
		return 0;

	case TFORW:
		if(exporting)
			yyerror("undefined type %S", t->sym);
		if(t->sym)
			return fmtprint(fp, "undefined %S", t->sym);
		return fmtprint(fp, "undefined");
	}

	// Don't know how to handle - fall back to detailed prints.
	return -1;
}



int
Tconv(Fmt *fp)
{
	char buf[500], buf1[500];
	Type *t, *t1;
	int et, exp;

	t = va_arg(fp->args, Type*);
	if(t == T)
		return fmtstrcpy(fp, "<T>");

	t->trecur++;
	if(t->trecur > 5) {
		strncat(buf, "...", sizeof(buf));
		goto out;
	}

	if(!debug['t']) {
		exp = (fp->flags & FmtSharp);
		if(exp)
			exporting++;
		if(Tpretty(fp, t) >= 0) {
			t->trecur--;
			if(exp)
				exporting--;
			return 0;
		}
		if(exp)
			exporting--;
	}

	et = t->etype;
	snprint(buf, sizeof buf, "%E.", et);
	if(t->sym != S) {
		snprint(buf1, sizeof(buf1), "<%S>", t->sym);
		strncat(buf, buf1, sizeof(buf));
	}

	switch(et) {
	default:
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
		if(t->bound >= 0)
			snprint(buf1, sizeof(buf1), "[%ld]%T", t->bound, t->type);
		else
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
		snprint(buf, sizeof(buf), "%O-%R%J", n->op, n->val.u.reg, n);
		break;

	case OLITERAL:
		switch(n->val.ctype) {
		default:
			snprint(buf1, sizeof(buf1), "LITERAL-ctype=%d", n->val.ctype);
			break;
		case CTINT:
		case CTSINT:
		case CTUINT:
			snprint(buf1, sizeof(buf1), "I%B", n->val.u.xval);
			break;
		case CTFLT:
			snprint(buf1, sizeof(buf1), "F%g", mpgetflt(n->val.u.fval));
			break;
		case CTSTR:
			snprint(buf1, sizeof(buf1), "S\"%Z\"", n->val.u.sval);
			break;
		case CTBOOL:
			snprint(buf1, sizeof(buf1), "B%d", n->val.u.bval);
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

Node*
treecopy(Node *n)
{
	Node *m;

	if(n == N)
		return N;

	switch(n->op) {
	default:
		m = nod(OXXX, N, N);
		*m = *n;
		m->left = treecopy(n->left);
		m->right = treecopy(n->right);
		break;

	case OLITERAL:
		if(n->iota) {
			m = literal(iota);
			break;
		}
		m = nod(OXXX, N, N);
		*m = *n;
		break;

	case ONAME:
		m = nod(OXXX, N, N);
		*m = *n;
		break;
	}
	return m;
}

int
Zconv(Fmt *fp)
{
	Rune r;
	String *sp;
	char *s, *se;

	sp = va_arg(fp->args, String*);
	if(sp == nil)
		return fmtstrcpy(fp, "<nil>");

	s = sp->s;
	se = s + sp->len;
	while(s < se) {
		s += chartorune(&r, s);
		switch(r) {
		default:
			fmtrune(fp, r);
			break;
		case '\0':
			fmtstrcpy(fp, "\\x00");
			break;
		case '\t':
			fmtstrcpy(fp, "\\t");
			break;
		case '\n':
			fmtstrcpy(fp, "\\n");
			break;
		case '\"':
		case '\\':
			fmtrune(fp, '\\');
			fmtrune(fp, r);
			break;
		}
	}
	return 0;
}

static char*
wnames[] =
{
	[Wnil] =	"Wnil",
	[Wtnil] =	"Wtnil",

	[Wtfloat] =	"Wtfloat",
	[Wtint] =	"Wtint",
	[Wtbool] =	"Wtbool",
	[Wtstr] =	"Wtstr",

	[Wlitfloat] =	"float constant",
	[Wlitint] =	"int constant",
	[Wlitbool] =	"bool",
	[Wlitstr] =	"string",
	[Wlitnil] =	"nil",
};

int
Wconv(Fmt *fp)
{
	int w;

	w = va_arg(fp->args, int);
	if(w < 0 || w >= nelem(wnames) || wnames[w] == nil)
		return fmtprint(fp, "W-%d", w);
	return fmtstrcpy(fp, wnames[w]);
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
isptrsarray(Type *t)
{
	if(isptrto(t, TARRAY))
		if(t->type->bound >= 0)
			return 1;
	return 0;
}

int
isptrdarray(Type *t)
{
	if(isptrto(t, TARRAY))
		if(t->type->bound < 0)
			return 1;
	return 0;
}

int
issarray(Type *t)
{
	if(t != T && t->etype == TARRAY && t->bound >= 0)
		return 1;
	return 0;
}

int
isdarray(Type *t)
{
	if(t != T && t->etype == TARRAY && t->bound < 0)
		return 1;
	return 0;
}

int
isselect(Node *n)
{
	Sym *s;

	if(n == N)
		return 0;
	n = n->left;
	s = pkglookup("selectsend", "sys");
	if(s == n->sym)
		return 1;
	s = pkglookup("selectrecv", "sys");
	if(s == n->sym)
		return 1;
	s = pkglookup("selectdefault", "sys");
	if(s == n->sym)
		return 1;
	return 0;
}

int
isinter(Type *t)
{
	if(t != T) {
		if(t->etype == TINTER)
			return 1;
		if(t->etype == TDDD)
			return 1;
	}
	return 0;
}

int
isnilinter(Type *t)
{
	if(!isinter(t))
		return 0;
	if(t->type != T)
		return 0;
	return 1;
}

int
isddd(Type *t)
{
	if(t != T && t->etype == TDDD)
		return 1;
	return 0;
}

/*
 * given receiver of type t (t == r or t == *r)
 * return type to hang methods off (r).
 */
Type*
dclmethod(Type *t)
{
	int ptr;

	if(t == T)
		return T;

	// strip away pointer if it's there
	ptr = 0;
	if(isptr[t->etype]) {
		if(t->sym != S)
			return T;
		ptr = 1;
		t = t->type;
		if(t == T)
			return T;
	}

	// need a type name
	if(t->sym == S)
		return T;

	// check that all method receivers are consistent
	if(t->methptr != 0 && t->methptr != (1<<ptr)) {
		if(t->methptr != 3) {
			t->methptr = 3;
			yyerror("methods on both %T and *%T", t, t);
		}
	}
	t->methptr |= 1<<ptr;

	// check types
	// TODO(rsc): map, chan etc are not quite right
	if(!issimple[t->etype])
	switch(t->etype) {
	default:
		return T;
	case TSTRUCT:
	case TARRAY:
		break;
	}

	return t;
}

/*
 * this is dclmethod() without side effects.
 */
Type*
methtype(Type *t)
{
	if(t == T)
		return T;
	if(isptr[t->etype]) {
		if(t->sym != S)
			return T;
		t = t->type;
	}
	if(t == T || t->etype == TINTER || t->sym == S)
		return T;
	return t;
}

/*
 * given type t in a method call, returns op
 * to convert t into appropriate receiver.
 * returns OADDR if t==x and method takes *x
 * returns OIND if t==*x and method takes x
 */
int
methconv(Type *t)
{
	Type *m;

	m = methtype(t);
	if(m == T)
		return 0;
	if(m->methptr&2) {
		// want pointer
		if(t == m)
			return OADDR;
		return 0;
	}
	// want non-pointer
	if(t != m)
		return OIND;
	return 0;
}

int
iscomposite(Type *t)
{
	if(t == T)
		return 0;
	switch(t->etype) {
	case TARRAY:
	case TSTRUCT:
		return 1;
	}
	if(isptr[t->etype] && t->type != T && t->type->etype == TMAP)
		return 1;
	return 0;
}

Sym*
signame(Type *t)
{
	Sym *ss;
	char *e;
	Dcl *x;
	char buf[NSYMB];

//print("signame %T\n", t);
	if(t == T)
		goto bad;

	e = "sigt";
	if(t->etype == TINTER)
		e = "sigi";

	// name is exported name, like *[]byte or *Struct or Interface
	// (special symbols don't bother the linker).
	snprint(buf, sizeof(buf), "%#T", t);

	// special case: empty interface is named sigi.empty
	// so that it can be referred to by the runtime.
	if(strcmp(buf, "interface { }") == 0)
		strcpy(buf, "empty");

	ss = pkglookup(buf, e);
	if(ss->oname == N) {
		ss->oname = newname(ss);
		ss->oname->type = types[TUINT8];
		ss->oname->class = PEXTERN;
	}

	if(!t->siggen) {
//print("siggen %T\n", t);
		// special case: don't generate the empty interface
		if(strcmp(buf, "empty") == 0)
			goto out;

		// record internal type for signature generation
		x = mal(sizeof(*x));
		x->op = OTYPE;
		x->dtype = t;
		x->forw = signatlist;
		t->siggen = 1;
		signatlist = x;
	}

out:
	return ss;

bad:
	return S;
}

int
bytearraysz(Type *t)
{
	if(t == T)
		return -2;
	if(isptr[t->etype]) {
		t = t->type;
		if(t == T)
			return -2;
	}
	if(t->etype != TARRAY)
		return -2;
	if(!eqtype(t->type, types[TUINT8], 0))
		return -2;
	return t->bound;	// -1 is dyn, >=0 is fixed
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
			if(!eqtype(t1, t2, d+1))
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
		// Loop over structs: receiver, in, out.
		t1 = t1->type;
		t2 = t2->type;
		for(;;) {
			Type *ta, *tb;
			if(t1 == t2)
				break;
			if(t1 == T || t2 == T)
				return 0;
			if(t1->etype != TSTRUCT || t2->etype != TSTRUCT)
				return 0;

			// Loop over fields in structs, checking type only.
			ta = t1->type;
			tb = t2->type;
			while(ta != tb) {
				if(ta == T || tb == T)
					return 0;
				if(ta->etype != TFIELD || tb->etype != TFIELD)
					return 0;
				if(!eqtype(ta->type, tb->type, 0))
					return 0;
				ta = ta->down;
				tb = tb->down;
			}

			t1 = t1->down;
			t2 = t2->down;
		}
		return 1;

	case TARRAY:
		if(t1->bound == t2->bound)
			break;
		return 0;
	}
	return eqtype(t1->type, t2->type, d+1);
}

static int
subtype(Type **stp, Type *t, int d)
{
	Type *st;

loop:
	st = *stp;
	if(st == T)
		return 0;

	d++;
	if(d >= 10)
		return 0;

	switch(st->etype) {
	default:
		return 0;

	case TPTR32:
	case TPTR64:
	case TCHAN:
	case TARRAY:
		stp = &st->type;
		goto loop;

	case TANY:
		if(!st->copyany)
			return 0;
		*stp = t;
		break;

	case TMAP:
		if(subtype(&st->down, t, d))
			break;
		stp = &st->type;
		goto loop;

	case TFUNC:
		for(;;) {
			if(subtype(&st->type, t, d))
				break;
			if(subtype(&st->type->down->down, t, d))
				break;
			if(subtype(&st->type->down, t, d))
				break;
			return 0;
		}
		break;

	case TSTRUCT:
		for(st=st->type; st!=T; st=st->down)
			if(subtype(&st->type, t, d))
				return 1;
		return 0;
	}
	return 1;
}

void
argtype(Node *on, Type *t)
{
	if(!subtype(&on->type, t, 0))
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

	case TANY:
		nt = shallow(t);
		nt->copyany = 1;
		break;

	case TPTR32:
	case TPTR64:
	case TCHAN:
	case TARRAY:
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

uint32
typehash(Type *at, int d)
{
	uint32 h;
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
	t1->width = types[tptr]->width;
	return t1;
}

Node*
literal(int32 v)
{
	Node *n;

	n = nod(OLITERAL, N, N);
	n->val.u.xval = mal(sizeof(*n->val.u.xval));
	n->val.ctype = CTINT;
	mpmovecfix(n->val.u.xval, v);
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
		ul = 1;
		goto out;
	case OCALL:
	case OCALLMETH:
	case OCALLINTER:
		ul = UINF;
		goto out;
	}
	ul = 1;
	if(n->left != N)
		ul = n->left->ullman;
	ur = 1;
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
		print("	%T\n", tl);
	if(tr != T)
		print("	%T\n", tr);

	// common mistake: *struct and *interface.
	if(tl && tr && isptr[tl->etype] && isptr[tr->etype]) {
		if(tl->type->etype == TSTRUCT && tr->type->etype == TINTER)
			print("	(*struct vs *interface)\n");
		else if(tl->type->etype == TINTER && tr->type->etype == TSTRUCT)
			print("	(*interface vs *struct)\n");
	}
}

/*
 * this routine gets called to propagate the type
 * of the last decl up to the arguments before it.
 * (a,b,c int) comes out (a int, b int, c int).
 */
Node*
cleanidlist(Node *na)
{
	Node *last, *n;

	if(na->op != OLIST)
		return na;

	for(last=na; last->op == OLIST; last=last->right)
		;
	if(last->op != ODCLFIELD)
		fatal("cleanidlist: %O", last->op);
	if(last->type == T)
		fatal("cleanidlist: no type");

	for(n=na; n->op == OLIST; n=n->right) {
		n->left->type = last->type;
		n->left->val = last->val;
	}
	return na;
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

/*
 * code to resolve elided DOTs
 * in embedded types
 */

// search depth 0 --
// return count of fields+methods
// found with a given name
int
lookdot0(Sym *s, Type *t, Type **save)
{
	Type *f, *u;
	int c;

	u = t;
	if(isptr[u->etype])
		u = u->type;

	c = 0;
	if(u->etype == TSTRUCT || u->etype == TINTER) {
		for(f=u->type; f!=T; f=f->down)
			if(f->sym == s) {
				if(save)
					*save = f;
				c++;
			}
	}
	u = methtype(t);
	if(u != T) {
		for(f=u->method; f!=T; f=f->down)
			if(f->sym == s && f->embedded == 0) {
				if(save)
					*save = f;
				c++;
			}
	}
	return c;
}

// search depth d --
// return count of fields+methods
// found at search depth.
// answer is in dotlist array and
// count of number of ways is returned.
int
adddot1(Sym *s, Type *t, int d, Type **save)
{
	Type *f, *u;
	int c, a;

	if(t->trecur)
		return 0;
	t->trecur = 1;

	if(d == 0) {
		c = lookdot0(s, t, save);
		goto out;
	}

	c = 0;
	u = t;
	if(isptr[u->etype])
		u = u->type;
	if(u->etype != TSTRUCT && u->etype != TINTER)
		goto out;

	d--;
	for(f=u->type; f!=T; f=f->down) {
		if(!f->embedded)
			continue;
		if(f->sym == S)
			continue;
		a = adddot1(s, f->type, d, save);
		if(a != 0 && c == 0)
			dotlist[d].field = f;
		c += a;
	}

out:
	t->trecur = 0;
	return c;
}

// in T.field
// find missing fields that
// will give shortest unique addressing.
// modify the tree with missing type names.
Node*
adddot(Node *n)
{
	Type *t;
	Sym *s;
	int c, d;

	walktype(n->left, Erv);
	t = n->left->type;
	if(t == T)
		goto ret;

	if(n->right->op != ONAME)
		goto ret;
	s = n->right->sym;
	if(s == S)
		goto ret;

	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(s, t, d, nil);
		if(c > 0)
			goto out;
	}
	goto ret;

out:
	if(c > 1)
		yyerror("ambiguous DOT reference %T.%S", t, s);

	// rebuild elided dots
	for(c=d-1; c>=0; c--) {
		n = nod(ODOT, n, n->right);
		n->left->right = newname(dotlist[c].field->sym);
	}
ret:
	n->ninit = list(addtop, n->ninit);
	addtop = N;
	return n;
}


/*
 * code to help generate trampoline
 * functions for methods on embedded
 * subtypes.
 * these are approx the same as
 * the corresponding adddot routines
 * except that they expect to be called
 * with unique tasks and they return
 * the actual methods.
 */

typedef	struct	Symlink	Symlink;
struct	Symlink
{
	Type*		field;
	uchar		good;
	Symlink*	link;
};
static	Symlink*	slist;

void
expand0(Type *t)
{
	Type *f, *u;
	Symlink *sl;

	u = t;
	if(isptr[u->etype])
		u = u->type;

	u = methtype(t);
	if(u != T) {
		for(f=u->method; f!=T; f=f->down) {
			if(f->sym->uniq)
				continue;
			f->sym->uniq = 1;
			sl = mal(sizeof(*sl));
			sl->field = f;
			sl->link = slist;
			slist = sl;
		}
	}
}

void
expand1(Type *t, int d)
{
	Type *f, *u;

	if(t->trecur)
		return;
	if(d == 0)
		return;
	t->trecur = 1;

	if(d != nelem(dotlist)-1)
		expand0(t);

	u = t;
	if(isptr[u->etype])
		u = u->type;
	if(u->etype != TSTRUCT && u->etype != TINTER)
		goto out;

	for(f=u->type; f!=T; f=f->down) {
		if(!f->embedded)
			continue;
		if(f->sym == S)
			continue;
		expand1(f->type, d-1);
	}

out:
	t->trecur = 0;
}

void
expandmeth(Sym *s, Type *t)
{
	Symlink *sl;
	Type *f;
	int c, d;

	if(s == S)
		return;
	if(t == T)
		return;

//print("s=%S t=%lT\n", s, t);

	// generate all reachable methods
	slist = nil;
	expand1(t, nelem(dotlist)-1);

	// check each method to be uniquely reachable
	for(sl=slist; sl!=nil; sl=sl->link) {
		sl->field->sym->uniq = 0;
		for(d=0; d<nelem(dotlist); d++) {
			c = adddot1(sl->field->sym, t, d, &f);
			if(c == 0)
				continue;
			if(c == 1 && f == sl->field)
				sl->good = 1;
			break;
		}
	}

	for(sl=slist; sl!=nil; sl=sl->link) {
		if(sl->good) {
//print("	%lT\n", sl->field);
			// add it to the base type method list
			f = typ(TFIELD);
			*f = *sl->field;
			f->embedded = 1;	// needs a trampoline

			f->down = t->method;
			t->method = f;

		}
	}
}

/*
 * Given funarg struct list, return list of ODCLFIELD Node fn args.
 */
Node*
structargs(Type **tl, int mustname)
{
	Iter savet;
	Node *args, *a;
	Type *t;
	char nam[100];
	int n;

	args = N;
	n = 0;
	for(t = structfirst(&savet, tl); t != T; t = structnext(&savet)) {
		if(t->sym)
			a = nametodcl(newname(t->sym), t->type);
		else if(mustname) {
			// have to give it a name so we can refer to it in trampoline
			snprint(nam, sizeof nam, ".anon%d", n++);
			a = nametodcl(newname(lookup(nam)), t->type);
		} else
			a = anondcl(t->type);
		args = list(args, a);
	}
	args = rev(args);
	return args;
}

/*
 * Generate a trampoline to convert
 * from an indirect receiver to a direct receiver
 * or vice versa.
 *
 *	method - short name of method (Len)
 *	oldname - old mangled method name (x·y·Len)
 *	oldthis - old this type (y)
 *	oldtype - type of method being called;
 *		only in and out params are known okay,
 *		receiver might be != oldthis.
 *	newnam [sic] - new mangled method name (x·*y·Len)
 *	newthis - new this type (*y)
 */
void
genptrtramp(Sym *method, Sym *oldname, Type *oldthis, Type *oldtype, Sym *newnam, Type *newthis)
{
	Node *fn, *args, *l, *in, *call, *out, *this, *rcvr, *meth;
	Iter savel;

	if(debug['r']) {
		print("\ngenptrtramp method=%S oldname=%S oldthis=%T\n",
			method, oldname, oldthis);
		print("\toldtype=%T newnam=%S newthis=%T\n",
			oldtype, newnam, newthis);
	}

	dclcontext = PEXTERN;
	markdcl();

	this = nametodcl(newname(lookup(".this")), newthis);
	in = structargs(getinarg(oldtype), 1);
	out = structargs(getoutarg(oldtype), 0);

	// fix up oldtype
	markdcl();
	oldtype = functype(nametodcl(newname(lookup(".this")), oldthis), in, out);
	popdcl();

	fn = nod(ODCLFUNC, N, N);
	fn->nname = newname(newnam);
	fn->type = functype(this, in, out);
	funchdr(fn);

	// arg list
	args = N;
	for(l = listfirst(&savel, &in); l; l = listnext(&savel))
		args = list(args, l->left);
	args = rev(args);

	// method to call
	if(isptr[oldthis->etype])
		rcvr = nod(OADDR, this->left, N);
	else
		rcvr = nod(OIND, this->left, N);
	gettype(rcvr, N);
	meth = nod(ODOTMETH, rcvr, newname(oldname));
	meth->xoffset = BADWIDTH;	// TODO(rsc): necessary?
	meth->type = oldtype;

	call = nod(OCALL, meth, args);
	fn->nbody = call;
	if(oldtype->outtuple > 0)
		fn->nbody = nod(ORETURN, call, N);

	if(debug['r'])
		dump("genptrtramp body", fn->nbody);

	funcbody(fn);
}

/*
 * delayed interface type check.
 * remember that there is an interface conversion
 * on the given line.  once the file is completely read
 * and all methods are known, we can check that
 * the conversions are valid.
 */

typedef struct Icheck Icheck;
struct Icheck
{
	Icheck *next;
	Type *dst;
	Type *src;
	int lineno;
};
Icheck *icheck;
Icheck *ichecktail;

void
ifacecheck(Type *dst, Type *src, int lineno)
{
	Icheck *p;

	p = mal(sizeof *p);
	if(ichecktail)
		ichecktail->next = p;
	else
		icheck = p;
	p->dst = dst;
	p->src = src;
	p->lineno = lineno;
	ichecktail = p;
}

Type*
ifacelookdot(Sym *s, Type *t)
{
	int c, d;
	Type *m;

	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(s, t, d, &m);
		if(c > 1) {
			yyerror("%T.%S is ambiguous", t, s);
			return T;
		}
		if(c == 1)
			return m;
	}
	return T;
}

int
hasiface(Type *t, Type *iface, Type **m)
{
	Type *im, *tm;
	int imhash;

	t = methtype(t);
	if(t == T)
		return 0;

	// if this is too slow,
	// could sort these first
	// and then do one loop.

	// could also do full type compare
	// instead of using hash, but have to
	// avoid checking receivers, and
	// typehash already does that for us.
	// also, it's what the runtime will do,
	// so we can both be wrong together.

	for(im=iface->type; im; im=im->down) {
		imhash = typehash(im, 0);
		tm = ifacelookdot(im->sym, t);
		if(tm == T || typehash(tm, 0) != imhash) {
			*m = im;
			return 0;
		}
	}
	return 1;
}

void
runifacechecks(void)
{
	Icheck *p;
	int lno;
	Type *m, *l, *r;

	lno = lineno;
	for(p=icheck; p; p=p->next) {
		lineno = p->lineno;
		if(isinter(p->dst)) {
			l = p->src;
			r = p->dst;
		} else {
			l = p->dst;
			r = p->src;
		}
		if(!hasiface(l, r, &m))
			yyerror("%T is not %T - missing %S%hT",
				l, r, m->sym, m->type);
	}
	lineno = lno;
}
