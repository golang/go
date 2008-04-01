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
myexit(int x)
{
	if(x)
		exits("error");
	exits(nil);
}

void
yyerror(char *fmt, ...)
{
	va_list arg;
	long lno;

	lno = dynlineno;
	if(lno == 0)
		lno = curio.lineno;

	print("%s:%ld: ", curio.infile, lno);
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
	long lno;

	lno = dynlineno;
	if(lno == 0)
		lno = curio.lineno;

	print("%s:%ld: ", curio.infile, lno);
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
	long lno;

	lno = dynlineno;
	if(lno == 0)
		lno = curio.lineno;

	print("%s:%ld: fatal error: ", curio.infile, lno);
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	print("\n");
	if(debug['h'])
		*(int*)0 = 0;
	myexit(1);
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
		n->lineno = curio.lineno;
	return n;
}

Node*
dobad(void)
{
	return nod(OBAD, N, N);
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

Node*
aindex(Node *b, Node *t)
{
	Node *r;

	r = nod(OTYPE, N, N);
	r->type = t;
	r->etype = TARRAY;

	if(t->etype == TDARRAY)
		yyerror("dynamic array type cannot be a dynamic array");

	walktype(b, 0);
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

	case ODCLFUNC:
		dodump(n->nname, dep);
		if(n->this) {
			indent(dep);
			print("%O-this\n", n->op);
			dodump(n->this, dep+1);
		}
		if(n->argout) {
			indent(dep);
			print("%O-outarg\n", n->op);
			dodump(n->argout, dep+1);
		}
		if(n->argin) {
			indent(dep);
			print("%O-inarg\n", n->op);
			dodump(n->argin, dep+1);
		}
		n = n->nbody;
		goto loop;

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
	Node *t;

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
		}
		return Wtunkn;
	}

	t = n->type;
	if(t == N)
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

	case TPTR:
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
	[OCALLPTR]	= "CALLPTR",
	[OCALLMETH]	= "CALLMETH",
	[OCALLINTER]	= "CALLINTER",
	[OCAT]		= "CAT",
	[OCASE]		= "CASE",
	[OXCASE]	= "XCASE",
	[OFALL]		= "FALL",
	[OCONV]		= "CONV",
	[OCOLAS]	= "COLAS",
	[OCOM]		= "COM",
	[OCONST]	= "CONST",
	[OCONTINUE]	= "CONTINUE",
	[ODCLARG]	= "DCLARG",
	[ODCLCONST]	= "DCLCONST",
	[ODCLFIELD]	= "DCLFIELD",
	[ODCLFUNC]	= "DCLFUNC",
	[ODCLTYPE]	= "DCLTYPE",
	[ODCLVAR]	= "DCLVAR",
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
	[OINDEXSTR]	= "INDEXSTR",
	[OINDEXMAP]	= "INDEXMAP",
	[OINDEXPTRMAP]	= "INDEXPTRMAP",
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
	[ONE]		= "NE",
	[ONOT]		= "NOT",
	[OOROR]		= "OROR",
	[OOR]		= "OR",
	[OPLUS]		= "PLUS",
	[ODEC]		= "DEC",
	[OINC]		= "INC",
	[OSEND]		= "SEND",
	[ORECV]		= "RECV",
	[OPTR]		= "PTR",
	[ORETURN]	= "RETURN",
	[ORSH]		= "RSH",
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
	[TPTR]		= "PTR",
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
	Node *t;

	t = va_arg(fp->args, Node*);

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
	Node *t, *t1;
	int et;

	t = va_arg(fp->args, Node*);
	if(t == N)
		return fmtstrcpy(fp, "<T>");

	t->trecur++;
	if(t->op != OTYPE) {
		snprint(buf, sizeof(buf), "T-%O", t->op);
		goto out;
	}
	et = t->etype;

	strcpy(buf, "");
	if(t->sym != S) {
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
		if(t->type != N) {
			snprint(buf1, sizeof(buf1), " %T", t->type);
			strncat(buf, buf1, sizeof(buf));
		}
		break;

	case TFIELD:
		snprint(buf1, sizeof(buf1), "%T", t->type);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TFUNC:
		snprint(buf1, sizeof(buf1), "%d%d%d(%lT,%lT,%lT)",
			t->thistuple, t->outtuple, t->intuple,
			t->type, t->type->down, t->type->down->down);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TINTER:
		strncat(buf, "I{", sizeof(buf));
		if(fp->flags & FmtLong) {
			for(t1=t->type; t1!=N; t1=t1->down) {
				snprint(buf1, sizeof(buf1), "%T;", t1);
				strncat(buf, buf1, sizeof(buf));
			}
		}
		strncat(buf, "}", sizeof(buf));
		break;

	case TSTRUCT:
		strncat(buf, "{", sizeof(buf));
		if(fp->flags & FmtLong) {
			for(t1=t->type; t1!=N; t1=t1->down) {
				snprint(buf1, sizeof(buf1), "%T;", t1);
				strncat(buf, buf1, sizeof(buf));
			}
		}
		strncat(buf, "}", sizeof(buf));
		break;

	case TMAP:
		snprint(buf, sizeof(buf), "[%T]%T", t->down, t->type);
		break;

	case TARRAY:
		snprint(buf1, sizeof(buf1), "[%ld]%T", t->bound, t->type);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TDARRAY:
		snprint(buf1, sizeof(buf1), "[]%T", t->type);
		strncat(buf, buf1, sizeof(buf));
		break;

	case TPTR:
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
		if(n->sym == S) {
			snprint(buf, sizeof(buf), "%O%J", n->op, n);
			break;
		}
		snprint(buf, sizeof(buf), "%O-%S G%ld%J", n->op,
			n->sym, n->sym->vargen, n);
		goto ptyp;

	case OLITERAL:
		switch(n->val.ctype) {
		default:
			snprint(buf1, sizeof(buf1), "LITERAL-%d", n->val.ctype);
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
		snprint(buf, sizeof(buf1), "%O-%s%J", n->op, buf1, n);
		break;
		
	case OASOP:
		snprint(buf, sizeof(buf), "%O-%O%J", n->op, n->kaka, n);
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
	if(n->type != N) {
		snprint(buf1, sizeof(buf1), " %T", n->type);
		strncat(buf, buf1, sizeof(buf));
	}

out:
	return fmtstrcpy(fp, buf);
}

int
Zconv(Fmt *fp)
{
	uchar *s, *se;
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
isptrto(Node *t, int et)
{
	if(t == N)
		return 0;
	if(t->etype != TPTR)
		return 0;
	t = t->type;
	if(t == N)
		return 0;
	if(t->etype != et)
		return 0;
	return 1;
}

int
isinter(Node *t)
{
	if(t != N && t->etype == TINTER)
		return 1;
	return 0;
}

int
isbytearray(Node *t)
{
	if(t == N)
		return 0;
	if(t->etype == TPTR) {
		t = t->type;
		if(t == N)
			return 0;
	}
	if(t->etype != TARRAY)
		return 0;
	return t->bound+1;
}

int
eqtype(Node *t1, Node *t2, int d)
{
	if(d >= 10)
		return 1;

	if(t1 == t2)
		return 1;
	if(t1 == N || t2 == N)
		return 0;
	if(t1->op != OTYPE || t2->op != OTYPE)
		fatal("eqtype: oops %O %O", t1->op, t2->op);

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
			if(t1 == N)
				return 1;
			if(t1->nname != N && t1->nname->sym != S) {
				if(t2->nname == N || t2->nname->sym == S)
					return 0;
				if(strcmp(t1->nname->sym->name, t2->nname->sym->name) != 0)
					return 0;
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
			if(t1 == N || t2 == N)
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

ulong
typehash(Node *at, int d)
{
	ulong h;
	Node *t;

	if(at == N)
		return PRIME2;
	if(d >= 5)
		return PRIME3;

	if(at->op != OTYPE)
		fatal("typehash: oops %O", at->op);

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
		for(t=at->type; t!=N; t=t->down)
			h += PRIME6 * typehash(t, d+1);
		break;

	case TSTRUCT:
		for(t=at->type; t!=N; t=t->down)
			h += PRIME7 * typehash(t, d+1);
		break;

	case TFUNC:
		t = at->type;
		// skip this argument
		if(t != N)
			t = t->down;
		for(; t!=N; t=t->down)
			h += PRIME7 * typehash(t, d+1);
		break;
	}

	at->recur = 0;
	return h;
}

Node*
ptrto(Node *t)
{
	Node *p;

	p = nod(OTYPE, N, N);
	p->etype = TPTR;
	p->type = t;
	return p;
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
	case OCALL:
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
badtype(int o, Node *tl, Node *tr)
{
	yyerror("illegal types for operand");
	if(tl != N)
		print("	(%T)", tl);
	print(" %O ", o);
	if(tr != N)
		print("(%T)", tr);
	print("\n");
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
	Node *t, *l, *n, *nn;

	t = N;		// untyped name
	nn = r;		// next node to take

loop:
	n = nn;
	if(n == N) {
		if(t != N) {
			yyerror("syntax error in parameter list");
			l = types[TINT32];
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

	if(l->type == N) {
		if(t == N)
			t = n;
		goto loop;
	}

	if(t == N)
		goto loop;

	l = l->type;	// type to be distributed

distrib:
	while(t != n) {
		if(t->op != OLIST) {
			if(t->type == N)
				t->type = l;
			break;
		}
		if(t->left->type == N)
			t->left->type = l;
		t = t->right;
	}

	t = N;
	goto loop;
}

/*
 * iterator to walk a structure declaration
 */
Node*
structfirst(Iter *s, Node **nn)
{
	Node *r, *n;

	n = *nn;
	if(n == N || n->op != OTYPE)
		goto bad;

	switch(n->etype) {
	default:
		goto bad;

	case TSTRUCT:
	case TINTER:
	case TFUNC:
		break;
	}

	r = n->type;
	if(r == N)
		goto rnil;

	if(r->op != OTYPE || r->etype != TFIELD)
		fatal("structfirst: not field %N", r);

	s->n = r;
	return r;

bad:
	fatal("structfirst: not struct %N", n);

rnil:
	return N;
}

Node*
structnext(Iter *s)
{
	Node *n, *r;

	n = s->n;
	r = n->down;
	if(r == N)
		goto rnil;

	if(r->op != OTYPE || r->etype != TFIELD)
		goto bad;

	s->n = r;
	return r;

bad:
	fatal("structnext: not struct %N", n);

rnil:
	return N;
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
	if(r->op == OLIST) {
		s->n = r;
		s->an = &r->left;
		return r->left;
	}

	s->done = 1;
	s->an = &n->right;
	return n->right;
}

Node**
getthis(Node *t)
{
	if(t->etype != TFUNC)
		fatal("getthis: not a func %N", t);
	return &t->type;
}

Node**
getoutarg(Node *t)
{
	if(t->etype != TFUNC)
		fatal("getoutarg: not a func %N", t);
	return &t->type->down;
}

Node**
getinarg(Node *t)
{
	if(t->etype != TFUNC)
		fatal("getinarg: not a func %N", t);
	return &t->type->down->down;
}

Node*
getthisx(Node *t)
{
	return *getthis(t);
}

Node*
getoutargx(Node *t)
{
	return *getoutarg(t);
}

Node*
getinargx(Node *t)
{
	return *getinarg(t);
}
