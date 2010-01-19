// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"md5.h"
#include	"y.tab.h"
#include	"opnames.h"

typedef struct Error Error;
struct Error
{
	int lineno;
	int seq;
	char *msg;
};
static Error *err;
static int nerr;
static int merr;

void
errorexit(void)
{
	flusherrors();
	if(outfile)
		remove(outfile);
	exit(1);
}

extern int yychar;
int
parserline(void)
{
	if(yychar != 0 && yychar != -2)	// parser has one symbol lookahead
		return prevlineno;
	return lineno;
}

static void
adderr(int line, char *fmt, va_list arg)
{
	Fmt f;
	Error *p;

	fmtstrinit(&f);
	fmtprint(&f, "%L: ", line);
	fmtvprint(&f, fmt, arg);
	fmtprint(&f, "\n");

	if(nerr >= merr) {
		if(merr == 0)
			merr = 16;
		else
			merr *= 2;
		p = realloc(err, merr*sizeof err[0]);
		if(p == nil) {
			merr = nerr;
			flusherrors();
			print("out of memory\n");
			errorexit();
		}
		err = p;
	}
	err[nerr].seq = nerr;
	err[nerr].lineno = line;
	err[nerr].msg = fmtstrflush(&f);
	nerr++;
}

static int
errcmp(const void *va, const void *vb)
{
	Error *a, *b;

	a = (Error*)va;
	b = (Error*)vb;
	if(a->lineno != b->lineno)
		return a->lineno - b->lineno;
	if(a->seq != b->seq)
		return a->seq - b->seq;
	return 0;
}

void
flusherrors(void)
{
	int i;

	if(nerr == 0)
		return;
	qsort(err, nerr, sizeof err[0], errcmp);
	for(i=0; i<nerr; i++)
		print("%s", err[i].msg);
	nerr = 0;
}

static void
hcrash(void)
{
	if(debug['h']) {
		flusherrors();
		if(outfile)
			unlink(outfile);
		*(int*)0 = 0;
	}
}

void
yyerrorl(int line, char *fmt, ...)
{
	va_list arg;

	va_start(arg, fmt);
	adderr(line, fmt, arg);
	va_end(arg);

	hcrash();
	nerrors++;
	if(nerrors >= 10 && !debug['e'])
		fatal("too many errors");
}

void
yyerror(char *fmt, ...)
{
	va_list arg;

	if(strcmp(fmt, "syntax error") == 0) {
		yyerrorl(lexlineno, "syntax error near %s", lexbuf);
		nsyntaxerrors++;
		return;
	}

	va_start(arg, fmt);
	adderr(parserline(), fmt, arg);
	va_end(arg);

	hcrash();
	nerrors++;
	if(nerrors >= 10 && !debug['e'])
		fatal("too many errors");
}

void
warn(char *fmt, ...)
{
	va_list arg;

	va_start(arg, fmt);
	adderr(parserline(), fmt, arg);
	va_end(arg);

	hcrash();
}

void
fatal(char *fmt, ...)
{
	va_list arg;

	flusherrors();

	print("%L: fatal error: ", lineno);
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	print("\n");

	hcrash();
	errorexit();
}

void
linehist(char *file, int32 off, int relative)
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

	if(off < 0 && file[0] != '/' && !relative) {
		cp = mal(strlen(file) + strlen(pathname) + 2);
		sprint(cp, "%s/%s", pathname, file);
		file = cp;
	}

	h = mal(sizeof(Hist));
	h->name = file;
	h->line = lexlineno;
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
	if(n != N)
	switch(n->op) {
	case ONAME:
	case OTYPE:
	case OPACK:
	case OLITERAL:
	case ONONAME:
		break;
	default:
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
			if(s->package && strcmp(s->package, package) == 0)
				return s;
	}

	s = mal(sizeof(*s));
	s->name = mal(strlen(p)+1);
	s->package = package;
	s->lexical = LNAME;

	strcpy(s->name, p);

	s->link = hash[h];
	hash[h] = s;

	return s;
}

Sym*
pkglookup(char *name, char *pkg)
{
	Sym *s;
	uint32 h;
	int c;

	h = stringhash(name) % NHASH;
	c = name[0];
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != c)
			continue;
		if(strcmp(s->name, name) == 0)
			if(s->package && strcmp(s->package, pkg) == 0)
				return s;
	}

	s = mal(sizeof(*s));
	s->name = mal(strlen(name)+1);
	strcpy(s->name, name);

	// botch - should probably try to reuse the pkg string
	s->package = mal(strlen(pkg)+1);
	strcpy(s->package, pkg);

	s->link = hash[h];
	hash[h] = s;
	s->lexical = LNAME;

	return s;
}

Sym*
restrictlookup(char *name, char *pkg)
{
	if(!exportname(name) && strcmp(pkg, package) != 0)
		yyerror("cannot refer to unexported name %s.%s", pkg, name);
	return pkglookup(name, pkg);
}


// find all the exported symbols in package opkg
// and make them available in the current package
void
importdot(Sym *opkg, Node *pack)
{
	Sym *s, *s1;
	uint32 h;
	int c, n;

	if(strcmp(opkg->name, package) == 0)
		return;

	n = 0;
	c = opkg->name[0];
	for(h=0; h<NHASH; h++) {
		for(s = hash[h]; s != S; s = s->link) {
			if(s->package[0] != c)
				continue;
			if(!exportname(s->name) || utfrune(s->name, 0xb7))	// 0xb7 = center dot
				continue;
			if(strcmp(s->package, opkg->name) != 0)
				continue;
			s1 = lookup(s->name);
			if(s1->def != N) {
				redeclare(s1, "during import");
				continue;
			}
			s1->def = s->def;
			s1->block = s->block;
			s1->def->pack = pack;
			n++;
		}
	}
	if(n == 0) {
		// can't possibly be used - there were no symbols
		yyerrorl(pack->lineno, "imported and not used: %s", pack->sym->name);
	}
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
		flusherrors();
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

Node*
nod(int op, Node *nleft, Node *nright)
{
	Node *n;

	n = mal(sizeof(*n));
	n->op = op;
	n->left = nleft;
	n->right = nright;
	n->lineno = parserline();
	n->xoffset = BADWIDTH;
	return n;
}

int
algtype(Type *t)
{
	int a;

	if(issimple[t->etype] || isptr[t->etype] || t->etype == TCHAN || t->etype == TFUNC || t->etype == TMAP)
		a = AMEM;	// just bytes (int, ptr, etc)
	else if(t->etype == TSTRING)
		a = ASTRING;	// string
	else if(isnilinter(t))
		a = ANILINTER;	// nil interface
	else if(t->etype == TINTER)
		a = AINTER;	// interface
	else
		a = ANOEQ;	// just bytes, but no hash/eq
	return a;
}

Type*
maptype(Type *key, Type *val)
{
	Type *t;

	if(key != nil && key->etype != TANY && algtype(key) == ANOEQ) {
		if(key->etype == TFORW) {
			// map[key] used during definition of key.
			// postpone check until key is fully defined.
			// if there are multiple uses of map[key]
			// before key is fully defined, the error
			// will only be printed for the first one.
			// good enough.
			if(key->maplineno == 0)
				key->maplineno = lineno;
		} else
			yyerror("invalid map key type %T", key);
	}
	t = typ(TMAP);
	t->down = key;
	t->type = val;
	return t;
}

int
iskeytype(Type *t)
{
	return algtype(t) != ANOEQ;
}

Type*
typ(int et)
{
	Type *t;

	t = mal(sizeof(*t));
	t->etype = et;
	t->width = BADWIDTH;
	t->lineno = lineno;
	return t;
}


Type*
sortinter(Type *t)
{
	return t;
}

Node*
nodintconst(int64 v)
{
	Node *c;

	c = nod(OLITERAL, N, N);
	c->addable = 1;
	c->val.u.xval = mal(sizeof(*c->val.u.xval));
	mpmovecfix(c->val.u.xval, v);
	c->val.ctype = CTINT;
	c->type = types[TIDEAL];
	ullmancalc(c);
	return c;
}

void
nodconst(Node *n, Type *t, int64 v)
{
	memset(n, 0, sizeof(*n));
	n->op = OLITERAL;
	n->addable = 1;
	ullmancalc(n);
	n->val.u.xval = mal(sizeof(*n->val.u.xval));
	mpmovecfix(n->val.u.xval, v);
	n->val.ctype = CTINT;
	n->type = t;

	if(isfloat[t->etype])
		fatal("nodconst: bad type %T", t);
}

Node*
nodnil(void)
{
	Node *c;

	c = nodintconst(0);
	c->val.ctype = CTNIL;
	c->type = types[TNIL];
	return c;
}

Node*
nodbool(int b)
{
	Node *c;

	c = nodintconst(0);
	c->val.ctype = CTBOOL;
	c->val.u.bval = b;
	c->type = idealbool;
	return c;
}

Type*
aindex(Node *b, Type *t)
{
	NodeList *init;
	Type *r;
	int bound;

	bound = -1;	// open bound
	init = nil;
	typecheck(&b, Erv);
	if(b != nil) {
		switch(consttype(b)) {
		default:
			yyerror("array bound must be an integer expression");
			break;
		case CTINT:
			bound = mpgetfix(b->val.u.xval);
			if(bound < 0)
				yyerror("array bound must be non negative");
			break;
		}
	}

	// fixed array
	r = typ(TARRAY);
	r->type = t;
	r->bound = bound;
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
dodumplist(NodeList *l, int dep)
{
	for(; l; l=l->next)
		dodump(l->n, dep);
}

void
dodump(Node *n, int dep)
{
	if(n == N)
		return;

	indent(dep);
	if(dep > 10) {
		print("...\n");
		return;
	}

	if(n->ninit != nil) {
		print("%O-init\n", n->op);
		dodumplist(n->ninit, dep+1);
		indent(dep);
	}

	switch(n->op) {
	default:
		print("%N\n", n);
		dodump(n->left, dep+1);
		dodump(n->right, dep+1);
		break;

	case OTYPE:
		print("%O %S type=%T\n", n->op, n->sym, n->type);
		if(n->type == T && n->ntype) {
			indent(dep);
			print("%O-ntype\n", n->op);
			dodump(n->ntype, dep+1);
		}
		break;

	case OIF:
		print("%O%J\n", n->op, n);
		dodump(n->ntest, dep+1);
		if(n->nbody != nil) {
			indent(dep);
			print("%O-then\n", n->op);
			dodumplist(n->nbody, dep+1);
		}
		if(n->nelse != nil) {
			indent(dep);
			print("%O-else\n", n->op);
			dodumplist(n->nelse, dep+1);
		}
		break;

	case OSELECT:
		print("%O%J\n", n->op, n);
		dodumplist(n->nbody, dep+1);
		break;

	case OSWITCH:
	case OFOR:
		print("%O%J\n", n->op, n);
		dodump(n->ntest, dep+1);

		if(n->nbody != nil) {
			indent(dep);
			print("%O-body\n", n->op);
			dodumplist(n->nbody, dep+1);
		}

		if(n->nincr != N) {
			indent(dep);
			print("%O-incr\n", n->op);
			dodump(n->nincr, dep+1);
		}
		break;

	case OCASE:
		// the right side points to label of the body
		if(n->right != N && n->right->op == OGOTO && n->right->left->op == ONAME)
			print("%O%J GOTO %N\n", n->op, n, n->right->left);
		else
			print("%O%J\n", n->op, n);
		dodump(n->left, dep+1);
		break;

	case OXCASE:
		print("%N\n", n);
		dodump(n->left, dep+1);
		dodump(n->right, dep+1);
		indent(dep);
		print("%O-nbody\n", n->op);
		dodumplist(n->nbody, dep+1);
		break;
	}

	if(0 && n->ntype != nil) {
		indent(dep);
		print("%O-ntype\n", n->op);
		dodump(n->ntype, dep+1);
	}
	if(n->list != nil) {
		indent(dep);
		print("%O-list\n", n->op);
		dodumplist(n->list, dep+1);
	}
	if(n->rlist != nil) {
		indent(dep);
		print("%O-rlist\n", n->op);
		dodumplist(n->rlist, dep+1);
	}
	if(n->op != OIF && n->nbody != nil) {
		indent(dep);
		print("%O-nbody\n", n->op);
		dodumplist(n->nbody, dep+1);
	}
}

void
dumplist(char *s, NodeList *l)
{
	print("%s\n", s);
	dodumplist(l, 1);
}

void
dump(char *s, Node *n)
{
	print("%s [%p]\n", s, n);
	dodump(n, 1);
}

static char*
goopnames[] =
{
	[OADDR]		= "&",
	[OADD]		= "+",
	[OANDAND]	= "&&",
	[OANDNOT]	= "&^",
	[OAND]		= "&",
	[OAS]		= "=",
	[OAS2]		= "=",
	[OBREAK]	= "break",
	[OCAP]		= "cap",
	[OCASE]		= "case",
	[OCLOSED]	= "closed",
	[OCLOSE]	= "close",
	[OCOM]		= "^",
	[OCONTINUE]	= "continue",
	[OCOPY]		= "copy",
	[ODEC]		= "--",
	[ODEFER]	= "defer",
	[ODIV]		= "/",
	[OEQ]		= "==",
	[OFALL]		= "fallthrough",
	[OFOR]		= "for",
	[OGE]		= ">=",
	[OGOTO]		= "goto",
	[OGT]		= ">",
	[OIF]		= "if",
	[OINC]		= "++",
	[OIND]		= "*",
	[OLEN]		= "len",
	[OLE]		= "<=",
	[OLSH]		= "<<",
	[OLT]		= "<",
	[OMAKE]		= "make",
	[OMINUS]	= "-",
	[OMOD]		= "%",
	[OMUL]		= "*",
	[ONEW]		= "new",
	[ONE]		= "!=",
	[ONOT]		= "!",
	[OOROR]		= "||",
	[OOR]		= "|",
	[OPANICN]	= "panicln",
	[OPANIC]	= "panic",
	[OPLUS]		= "+",
	[OPRINTN]	= "println",
	[OPRINT]	= "print",
	[ORANGE]	= "range",
	[ORECV]		= "<-",
	[ORETURN]	= "return",
	[ORSH]		= ">>",
	[OSELECT]	= "select",
	[OSEND]		= "<-",
	[OSUB]		= "-",
	[OSWITCH]	= "switch",
	[OXOR]		= "^",
};

int
Oconv(Fmt *fp)
{
	char buf[500];
	int o;

	o = va_arg(fp->args, int);
	if((fp->flags & FmtSharp) && o >= 0 && o < nelem(goopnames) && goopnames[o] != nil)
		return fmtstrcpy(fp, goopnames[o]);
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
	[TBOOL]		= "BOOL",
	[TPTR32]	= "PTR32",
	[TPTR64]	= "PTR64",
	[TDDD]		= "DDD",
	[TFUNC]		= "FUNC",
	[TARRAY]	= "ARRAY",
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
	Node *n;

	n = va_arg(fp->args, Node*);
	if(n->ullman != 0)
		fmtprint(fp, " u(%d)", n->ullman);

	if(n->addable != 0)
		fmtprint(fp, " a(%d)", n->addable);

	if(n->vargen != 0)
		fmtprint(fp, " g(%ld)", n->vargen);

	if(n->lineno != 0)
		fmtprint(fp, " l(%ld)", n->lineno);

	if(n->xoffset != 0)
		fmtprint(fp, " x(%lld)", n->xoffset);

	if(n->class != 0)
		fmtprint(fp, " class(%d)", n->class);

	if(n->colas != 0)
		fmtprint(fp, " colas(%d)", n->colas);

	if(n->funcdepth != 0)
		fmtprint(fp, " f(%d)", n->funcdepth);

	if(n->typecheck != 0)
		fmtprint(fp, " tc(%d)", n->typecheck);

	if(n->dodata != 0)
		fmtprint(fp, " dd(%d)", n->dodata);

	return 0;
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
	char *pkg, *nam;

	s = va_arg(fp->args, Sym*);
	if(s == S) {
		fmtstrcpy(fp, "<S>");
		return 0;
	}

	pkg = "<nil>";
	nam = pkg;

	if(s->package != nil)
		pkg = s->package;
	if(s->name != nil)
		nam = s->name;

	if(!(fp->flags & FmtShort))
	if(strcmp(pkg, package) != 0 || (fp->flags & FmtLong)) {
		fmtprint(fp, "%s.%s", pkg, nam);
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
	[TBOOL]		= "bool",
	[TANY]		= "any",
	[TDDD]		= "...",
	[TSTRING]		= "string",
	[TNIL]		= "nil",
	[TIDEAL]		= "ideal",
	[TBLANK]		= "blank",
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
		if(t == types[t->etype])
			return fmtprint(fp, "%s", s->name);
		if(exporting) {
			if(fp->flags & FmtShort)
				fmtprint(fp, "%hS", s);
			else
				fmtprint(fp, "%lS", s);
			if(strcmp(s->package, package) != 0)
				return 0;
			if(s->flags & SymImported)
				return 0;
			if(t->vargen)
				fmtprint(fp, "·%d", t->vargen);
			return 0;
		}
		return fmtprint(fp, "%S", s);
	}

	if(t->etype < nelem(basicnames) && basicnames[t->etype] != nil) {
		if(isideal(t) && t->etype != TIDEAL && t->etype != TNIL)
			fmtprint(fp, "ideal ");
		return fmtprint(fp, "%s", basicnames[t->etype]);
	}

	switch(t->etype) {
	case TPTR32:
	case TPTR64:
		if(fp->flags&FmtShort)	// pass flag thru for methodsym
			return fmtprint(fp, "*%hT", t->type);
		return fmtprint(fp, "*%T", t->type);

	case TCHAN:
		switch(t->chan) {
		case Crecv:
			return fmtprint(fp, "<-chan %T", t->type);
		case Csend:
			if(t->type != T && t->type->etype == TCHAN)
				return fmtprint(fp, "chan<- (%T)", t->type);
			return fmtprint(fp, "chan<- %T", t->type);
		}
		return fmtprint(fp, "chan %T", t->type);

	case TMAP:
		return fmtprint(fp, "map[%T] %T", t->down, t->type);

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

		if(!(fp->flags&FmtByte))
			fmtprint(fp, "func");
		fmtprint(fp, "(");
		for(t1=getinargx(t)->type; t1; t1=t1->down) {
			if(noargnames && t1->etype == TFIELD)
				fmtprint(fp, "%T", t1->type);
			else
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
			if(t1 == T) {
				// failure to typecheck earlier; don't know the type
				fmtprint(fp, " ?unknown-type?");
				break;
			}
			if(t1->etype != TFIELD && t1->etype != TFUNC) {
				fmtprint(fp, " %T", t1);
				break;
			}
		default:
			t1 = getoutargx(t)->type;
			fmtprint(fp, " (");
			for(; t1; t1=t1->down) {
				if(noargnames && t1->etype == TFIELD)
					fmtprint(fp, "%T", t1->type);
				else
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
		if(t->bound == -100)
			return fmtprint(fp, "[...]%T", t->type);
		return fmtprint(fp, "[]%T", t->type);

	case TINTER:
		fmtprint(fp, "interface {");
		for(t1=t->type; t1!=T; t1=t1->down) {
			fmtprint(fp, " %hS %hhT", t1->sym, t1->type);
			if(t1->down)
				fmtprint(fp, ";");
		}
		return fmtprint(fp, " }");

	case TSTRUCT:
		if(t->funarg) {
			fmtprint(fp, "(");
			for(t1=t->type; t1!=T; t1=t1->down) {
				fmtprint(fp, "%T", t1);
				if(t1->down)
					fmtprint(fp, ", ");
			}
			return fmtprint(fp, ")");
		}
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
	Type *t, *t1;
	int r, et, sharp, minus;

	sharp = (fp->flags & FmtSharp);
	minus = (fp->flags & FmtLeft);
	fp->flags &= ~(FmtSharp|FmtLeft);

	t = va_arg(fp->args, Type*);
	if(t == T)
		return fmtstrcpy(fp, "<T>");

	t->trecur++;
	if(t->trecur > 5) {
		fmtprint(fp, "...");
		goto out;
	}

	if(!debug['t']) {
		if(sharp)
			exporting++;
		if(minus)
			noargnames++;
		r = Tpretty(fp, t);
		if(sharp)
			exporting--;
		if(minus)
			noargnames--;
		if(r >= 0) {
			t->trecur--;
			return 0;
		}
	}

	et = t->etype;
	fmtprint(fp, "%E ", et);
	if(t->sym != S)
		fmtprint(fp, "<%S>", t->sym);

	switch(et) {
	default:
		if(t->type != T)
			fmtprint(fp, " %T", t->type);
		break;

	case TFIELD:
		fmtprint(fp, "%T", t->type);
		break;

	case TFUNC:
		if(fp->flags & FmtLong)
			fmtprint(fp, "%d%d%d(%lT,%lT)%lT",
				t->thistuple, t->intuple, t->outtuple,
				t->type, t->type->down->down, t->type->down);
		else
			fmtprint(fp, "%d%d%d(%T,%T)%T",
				t->thistuple, t->intuple, t->outtuple,
				t->type, t->type->down->down, t->type->down);
		break;

	case TINTER:
		fmtprint(fp, "{");
		if(fp->flags & FmtLong)
			for(t1=t->type; t1!=T; t1=t1->down)
				fmtprint(fp, "%lT;", t1);
		fmtprint(fp, "}");
		break;

	case TSTRUCT:
		fmtprint(fp, "{");
		if(fp->flags & FmtLong)
			for(t1=t->type; t1!=T; t1=t1->down)
				fmtprint(fp, "%lT;", t1);
		fmtprint(fp, "}");
		break;

	case TMAP:
		fmtprint(fp, "[%T]%T", t->down, t->type);
		break;

	case TARRAY:
		if(t->bound >= 0)
			fmtprint(fp, "[%ld]%T", t->bound, t->type);
		else
			fmtprint(fp, "[]%T", t->type);
		break;

	case TPTR32:
	case TPTR64:
		fmtprint(fp, "%T", t->type);
		break;
	}

out:
	t->trecur--;
	return 0;
}

int
Nconv(Fmt *fp)
{
	char buf1[500];
	Node *n;

	n = va_arg(fp->args, Node*);
	if(n == N) {
		fmtprint(fp, "<N>");
		goto out;
	}

	if(fp->flags & FmtSign) {
		if(n->type == T)
			fmtprint(fp, "%#N", n);
		else if(n->type->etype == TNIL)
			fmtprint(fp, "nil");
		else
			fmtprint(fp, "%#N (type %T)", n, n->type);
		goto out;
	}

	if(fp->flags & FmtSharp) {
		exprfmt(fp, n, 0);
		goto out;
	}

	switch(n->op) {
	default:
		fmtprint(fp, "%O%J", n->op, n);
		break;

	case ONAME:
	case ONONAME:
		if(n->sym == S) {
			fmtprint(fp, "%O%J", n->op, n);
			break;
		}
		fmtprint(fp, "%O-%S G%ld%J", n->op,
			n->sym, n->vargen, n);
		goto ptyp;

	case OREGISTER:
		fmtprint(fp, "%O-%R%J", n->op, n->val.u.reg, n);
		break;

	case OLITERAL:
		switch(n->val.ctype) {
		default:
			snprint(buf1, sizeof(buf1), "LITERAL-ctype=%d", n->val.ctype);
			break;
		case CTINT:
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
		fmtprint(fp, "%O-%s%J", n->op, buf1, n);
		break;

	case OASOP:
		fmtprint(fp, "%O-%O%J", n->op, n->etype, n);
		break;

	case OTYPE:
		fmtprint(fp, "%O %T", n->op, n->type);
		break;
	}
	if(n->sym != S)
		fmtprint(fp, " %S G%ld", n->sym, n->vargen);

ptyp:
	if(n->type != T)
		fmtprint(fp, " %T", n->type);

out:
	return 0;
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
		m->list = listtreecopy(n->list);
		if(m->defn)
			abort();
		break;

	case ONONAME:
		if(n->iota) {
			m = nod(OIOTA, n, nodintconst(iota));
			break;
		}
		// fall through
	case OLITERAL:
	case ONAME:
	case OTYPE:
		m = n;
		break;
	}
	return m;
}

int
Zconv(Fmt *fp)
{
	Rune r;
	Strlit *sp;
	char *s, *se;

	sp = va_arg(fp->args, Strlit*);
	if(sp == nil)
		return fmtstrcpy(fp, "<nil>");

	s = sp->s;
	se = s + sp->len;
	while(s < se) {
		s += chartorune(&r, s);
		switch(r) {
		default:
			if(r < ' ') {
				fmtprint(fp, "\\x%02x", r);
				break;
			}
			fmtrune(fp, r);
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
istype(Type *t, int et)
{
	return t != T && t->etype == et;
}

int
isfixedarray(Type *t)
{
	return t != T && t->etype == TARRAY && t->bound >= 0;
}

int
isslice(Type *t)
{
	return t != T && t->etype == TARRAY && t->bound < 0;
}

int
isblank(Node *n)
{
	char *p;

	if(n == N || n->sym == S)
		return 0;
	p = n->sym->name;
	if(p == nil)
		return 0;
	return p[0] == '_' && p[1] == '\0';
}

int
isselect(Node *n)
{
	Sym *s;

	if(n == N)
		return 0;
	n = n->left;
	s = pkglookup("selectsend", "runtime");
	if(s == n->sym)
		return 1;
	s = pkglookup("selectrecv", "runtime");
	if(s == n->sym)
		return 1;
	s = pkglookup("selectdefault", "runtime");
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

int
isideal(Type *t)
{
	if(t == T)
		return 0;
	return t == idealstring || t == idealbool || t->etype == TNIL || t->etype == TIDEAL;
}

/*
 * given receiver of type t (t == r or t == *r)
 * return type to hang methods off (r).
 */
Type*
methtype(Type *t)
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

	// check types
	if(!issimple[t->etype])
	switch(t->etype) {
	default:
		return T;
	case TSTRUCT:
	case TARRAY:
	case TMAP:
	case TCHAN:
	case TSTRING:
	case TFUNC:
		break;
	}

	return t;
}

int
iscomposite(Type *t)
{
	if(t == T)
		return 0;
	switch(t->etype) {
	case TARRAY:
	case TSTRUCT:
	case TMAP:
		return 1;
	}
	return 0;
}

int
eqtype1(Type *t1, Type *t2, int d, int names)
{
	if(d >= 20)
		return 1;
	if(t1 == t2)
		return 1;
	if(t1 == T || t2 == T)
		return 0;
	if(t1->etype != t2->etype)
		return 0;
	if(names && t1->etype != TFIELD && t1->sym && t2->sym && t1 != t2)
		return 0;
	switch(t1->etype) {
	case TINTER:
	case TSTRUCT:
		t1 = t1->type;
		t2 = t2->type;
		for(;;) {
			if(!eqtype1(t1, t2, d+1, names))
				return 0;
			if(t1 == T)
				return 1;
			if(t1->embedded != t2->embedded)
				return 0;
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
				if(!eqtype1(ta->type, tb->type, d+1, names))
					return 0;
				ta = ta->down;
				tb = tb->down;
			}

			t1 = t1->down;
			t2 = t2->down;
		}
		return 1;

	case TARRAY:
		if(t1->bound != t2->bound)
			return 0;
		break;

	case TCHAN:
		if(t1->chan != t2->chan)
			return 0;
		break;

	case TMAP:
		if(!eqtype1(t1->down, t2->down, d+1, names))
			return 0;
		break;
	}
	return eqtype1(t1->type, t2->type, d+1, names);
}

int
eqtype(Type *t1, Type *t2)
{
	return eqtype1(t1, t2, 0, 1);
}

/*
 * can we convert from type src to dst with
 * a trivial conversion (no bits changing)?
 */
int
cvttype(Type *dst, Type *src)
{
	return eqtype1(dst, src, 0, 0);
}

int
eqtypenoname(Type *t1, Type *t2)
{
	if(t1 == T || t2 == T || t1->etype != TSTRUCT || t2->etype != TSTRUCT)
		return eqtype(t1, t2);

	t1 = t1->type;
	t2 = t2->type;
	for(;;) {
		if(!eqtype(t1, t2))
			return 0;
		if(t1 == T)
			return 1;
		t1 = t1->down;
		t2 = t2->down;
	}
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

/*
 * Is this a 64-bit type?
 */
int
is64(Type *t)
{
	if(t == T)
		return 0;
	switch(simtype[t->etype]) {
	case TINT64:
	case TUINT64:
	case TPTR64:
		return 1;
	}
	return 0;
}

/*
 * Is a conversion between t1 and t2 a no-op?
 */
int
noconv(Type *t1, Type *t2)
{
	int e1, e2;

	e1 = simtype[t1->etype];
	e2 = simtype[t2->etype];

	switch(e1) {
	case TINT8:
	case TUINT8:
		return e2 == TINT8 || e2 == TUINT8;

	case TINT16:
	case TUINT16:
		return e2 == TINT16 || e2 == TUINT16;

	case TINT32:
	case TUINT32:
	case TPTR32:
		return e2 == TINT32 || e2 == TUINT32 || e2 == TPTR32;

	case TINT64:
	case TUINT64:
	case TPTR64:
		return e2 == TINT64 || e2 == TUINT64 || e2 == TPTR64;

	case TFLOAT32:
		return e2 == TFLOAT32;

	case TFLOAT64:
		return e2 == TFLOAT64;
	}
	return 0;
}

void
argtype(Node *on, Type *t)
{
	dowidth(t);
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

	s = pkglookup(name, "runtime");
	if(s == S || s->def == N)
		fatal("looksys: cant find runtime.%s", name);

	if(!copy)
		return s->def;

	n = nod(0, N, N);
	*n = *s->def;
	n->type = deep(s->def->type);

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
		if(!eqtype(t1, t2))
			return 0;
		t1 = t1->down;
		t2 = t2->down;
	}
	return 1;
}

/*
 * compute a hash value for type t.
 * if t is a method type, ignore the receiver
 * so that the hash can be used in interface checks.
 * %#-T (which calls Tpretty, above) already contains
 * all the necessary logic to generate a representation
 * of the type that completely describes it.
 * using smprint here avoids duplicating that code.
 * using md5 here is overkill, but i got tired of
 * accidental collisions making the runtime think
 * two types are equal when they really aren't.
 */
uint32
typehash(Type *t)
{
	char *p;
	MD5 d;

	if(t->thistuple) {
		// hide method receiver from Tpretty
		t->thistuple = 0;
		p = smprint("%#-T", t);
		t->thistuple = 1;
	}else
		p = smprint("%#-T", t);
	md5reset(&d);
	md5write(&d, (uchar*)p, strlen(p));
	free(p);
	return md5sum(&d);
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

void
frame(int context)
{
	char *p;
	NodeList *l;
	Node *n;
	int flag;

	p = "stack";
	l = nil;
	if(curfn)
		l = curfn->dcl;
	if(context) {
		p = "external";
		l = externdcl;
	}

	flag = 1;
	for(; l; l=l->next) {
		n = l->n;
		switch(n->op) {
		case ONAME:
			if(flag)
				print("--- %s frame ---\n", p);
			print("%O %S G%ld %T\n", n->op, n->sym, n->vargen, n->type);
			flag = 0;
			break;

		case OTYPE:
			if(flag)
				print("--- %s frame ---\n", p);
			print("%O %T\n", n->op, n->type);
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
	case OREGISTER:
	case OLITERAL:
	case ONAME:
		ul = 1;
		if(n->class == PPARAMREF || (n->class & PHEAP))
			ul++;
		goto out;
	case OCALL:
	case OCALLFUNC:
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

Type**
getthis(Type *t)
{
	if(t->etype != TFUNC)
		fatal("getthis: not a func %T", t);
	return &t->type;
}

Type**
getoutarg(Type *t)
{
	if(t->etype != TFUNC)
		fatal("getoutarg: not a func %T", t);
	return &t->type->down;
}

Type**
getinarg(Type *t)
{
	if(t->etype != TFUNC)
		fatal("getinarg: not a func %T", t);
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
 * return !(op)
 * eg == <=> !=
 */
int
brcom(int a)
{
	switch(a) {
	case OEQ:	return ONE;
	case ONE:	return OEQ;
	case OLT:	return OGE;
	case OGT:	return OLE;
	case OLE:	return OGT;
	case OGE:	return OLT;
	}
	fatal("brcom: no com for %A\n", a);
	return a;
}

/*
 * return reverse(op)
 * eg a op b <=> b r(op) a
 */
int
brrev(int a)
{
	switch(a) {
	case OEQ:	return OEQ;
	case ONE:	return ONE;
	case OLT:	return OGT;
	case OGT:	return OLT;
	case OLE:	return OGE;
	case OGE:	return OLE;
	}
	fatal("brcom: no rev for %A\n", a);
	return a;
}

Node*
staticname(Type *t)
{
	Node *n;

	snprint(namebuf, sizeof(namebuf), "statictmp_%.4d", statuniqgen);
	statuniqgen++;
	n = newname(lookup(namebuf));
	addvar(n, t, PEXTERN);
	return n;
}

/*
 * return side effect-free, assignable n, appending side effects to init.
 */
Node*
saferef(Node *n, NodeList **init)
{
	Node *l;
	Node *r;
	Node *a;

	switch(n->op) {
	case ONAME:
		return n;
	case ODOT:
		l = saferef(n->left, init);
		if(l == n->left)
			return n;
		r = nod(OXXX, N, N);
		*r = *n;
		r->left = l;
		typecheck(&r, Erv);
		walkexpr(&r, init);
		return r;

	case OINDEX:
	case ODOTPTR:
	case OIND:
		l = nod(OXXX, N, N);
		tempname(l, ptrto(n->type));
		a = nod(OAS, l, nod(OADDR, n, N));
		typecheck(&a, Etop);
		walkexpr(&a, init);
		*init = list(*init, a);
		r = nod(OIND, l, N);
		typecheck(&r, Erv);
		walkexpr(&r, init);
		return r;
	}
	fatal("saferef %N", n);
	return N;
}

/*
 * return side effect-free n, appending side effects to init.
 */
Node*
safeval(Node *n, NodeList **init)
{
	Node *l;
	Node *a;

	// is this a local variable or a dot of a local variable?
	for(l=n; l->op == ODOT; l=l->left)
		if(l->left->type != T && isptr[l->left->type->etype])
			goto copy;
	if(l->op == ONAME && (l->class == PAUTO || l->class == PPARAM))
		return n;

copy:
	l = nod(OXXX, N, N);
	tempname(l, n->type);
	a = nod(OAS, l, n);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);
	return l;
}

void
setmaxarg(Type *t)
{
	int32 w;

	dowidth(t);
	w = t->argwid;
	if(t->argwid >= MAXWIDTH)
		fatal("bad argwid %T", t);
	if(w > maxarg)
		maxarg = w;
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

	typecheck(&n->left, Etype|Erv);
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
	for(c=d-1; c>=0; c--)
		n->left = nod(ODOT, n->left, newname(dotlist[c].field->sym));
ret:
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
	uchar		followptr;
	Symlink*	link;
};
static	Symlink*	slist;

static void
expand0(Type *t, int followptr)
{
	Type *f, *u;
	Symlink *sl;

	u = t;
	if(isptr[u->etype]) {
		followptr = 1;
		u = u->type;
	}

	if(u->etype == TINTER) {
		for(f=u->type; f!=T; f=f->down) {
			if(!exportname(f->sym->name) && strcmp(f->sym->package, package) != 0)
				continue;
			if(f->sym->flags & SymUniq)
				continue;
			f->sym->flags |= SymUniq;
			sl = mal(sizeof(*sl));
			sl->field = f;
			sl->link = slist;
			sl->followptr = followptr;
			slist = sl;
		}
		return;
	}

	u = methtype(t);
	if(u != T) {
		for(f=u->method; f!=T; f=f->down) {
			if(!exportname(f->sym->name) && strcmp(f->sym->package, package) != 0)
				continue;
			if(f->sym->flags & SymUniq)
				continue;
			f->sym->flags |= SymUniq;
			sl = mal(sizeof(*sl));
			sl->field = f;
			sl->link = slist;
			sl->followptr = followptr;
			slist = sl;
		}
	}
}

static void
expand1(Type *t, int d, int followptr)
{
	Type *f, *u;

	if(t->trecur)
		return;
	if(d == 0)
		return;
	t->trecur = 1;

	if(d != nelem(dotlist)-1)
		expand0(t, followptr);

	u = t;
	if(isptr[u->etype]) {
		followptr = 1;
		u = u->type;
	}
	if(u->etype != TSTRUCT && u->etype != TINTER)
		goto out;

	for(f=u->type; f!=T; f=f->down) {
		if(!f->embedded)
			continue;
		if(f->sym == S)
			continue;
		expand1(f->type, d-1, followptr);
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
	if(t == T || t->xmethod != nil)
		return;

	// generate all reachable methods
	slist = nil;
	expand1(t, nelem(dotlist)-1, 0);

	// check each method to be uniquely reachable
	for(sl=slist; sl!=nil; sl=sl->link) {
		sl->field->sym->flags &= ~SymUniq;
		for(d=0; d<nelem(dotlist); d++) {
			c = adddot1(sl->field->sym, t, d, &f);
			if(c == 0)
				continue;
			if(c == 1) {
				sl->good = 1;
				sl->field = f;
			}
			break;
		}
	}

	t->xmethod = t->method;
	for(sl=slist; sl!=nil; sl=sl->link) {
		if(sl->good) {
			// add it to the base type method list
			f = typ(TFIELD);
			*f = *sl->field;
			f->embedded = 1;	// needs a trampoline
			if(sl->followptr)
				f->embedded = 2;
			f->down = t->xmethod;
			t->xmethod = f;

		}
	}
}

/*
 * Given funarg struct list, return list of ODCLFIELD Node fn args.
 */
NodeList*
structargs(Type **tl, int mustname)
{
	Iter savet;
	Node *a, *n;
	NodeList *args;
	Type *t;
	char buf[100];
	int gen;

	args = nil;
	gen = 0;
	for(t = structfirst(&savet, tl); t != T; t = structnext(&savet)) {
		n = N;
		if(t->sym)
			n = newname(t->sym);
		else if(mustname) {
			// have to give it a name so we can refer to it in trampoline
			snprint(buf, sizeof buf, ".anon%d", gen++);
			n = newname(lookup(buf));
		}
		a = nod(ODCLFIELD, n, typenod(t->type));
		args = list(args, a);
	}
	return args;
}

/*
 * Generate a wrapper function to convert from
 * a receiver of type T to a receiver of type U.
 * That is,
 *
 *	func (t T) M() {
 *		...
 *	}
 *
 * already exists; this function generates
 *
 *	func (u U) M() {
 *		u.M()
 *	}
 *
 * where the types T and U are such that u.M() is valid
 * and calls the T.M method.
 * The resulting function is for use in method tables.
 *
 *	rcvr - U
 *	method - M func (t T)(), a TFIELD type struct
 *	newnam - the eventual mangled name of this function
 */
void
genwrapper(Type *rcvr, Type *method, Sym *newnam)
{
	Node *this, *fn, *call, *n, *t;
	NodeList *l, *args, *in, *out;

	if(debug['r'])
		print("genwrapper rcvrtype=%T method=%T newnam=%S\n",
			rcvr, method, newnam);

	dclcontext = PEXTERN;
	markdcl();

	this = nod(ODCLFIELD, newname(lookup(".this")), typenod(rcvr));
	this->left->ntype = this->right;
	in = structargs(getinarg(method->type), 1);
	out = structargs(getoutarg(method->type), 0);

	fn = nod(ODCLFUNC, N, N);
	fn->nname = newname(newnam);
	t = nod(OTFUNC, this, N);
	t->list = in;
	t->rlist = out;
	fn->nname->ntype = t;
	funchdr(fn);

	// arg list
	args = nil;
	for(l=in; l; l=l->next)
		args = list(args, l->n->left);

	// generate call
	call = nod(OCALL, adddot(nod(OXDOT, this->left, newname(method->sym))), N);
	call->list = args;
	fn->nbody = list1(call);
	if(method->type->outtuple > 0) {
		n = nod(ORETURN, N, N);
		n->list = fn->nbody;
		fn->nbody = list1(n);
	}

	if(debug['r'])
		dumplist("genwrapper body", fn->nbody);

	funcbody(fn);
	typecheck(&fn, Etop);
	funccompile(fn);
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
	int explicit;
};
Icheck *icheck;
Icheck *ichecktail;

void
ifacecheck(Type *dst, Type *src, int lineno, int explicit)
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
	p->explicit = explicit;
	ichecktail = p;
}

Type*
ifacelookdot(Sym *s, Type *t, int *followptr)
{
	int i, c, d;
	Type *m;

	*followptr = 0;

	if(t == T)
		return T;

	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(s, t, d, &m);
		if(c > 1) {
			yyerror("%T.%S is ambiguous", t, s);
			return T;
		}
		if(c == 1) {
			for(i=0; i<d; i++) {
				if(isptr[dotlist[i].field->type->etype]) {
					*followptr = 1;
					break;
				}
			}
			if(m->type->etype != TFUNC || m->type->thistuple == 0) {
				yyerror("%T.%S is a field, not a method", t, s);
				return T;
			}
			return m;
		}
	}
	return T;
}

// check whether non-interface type t
// satisifes inteface type iface.
int
ifaceokT2I(Type *t0, Type *iface, Type **m, Type **samename)
{
	Type *t, *im, *tm, *rcvr;
	int imhash, followptr;

	t = methtype(t0);

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
		imhash = typehash(im->type);
		tm = ifacelookdot(im->sym, t, &followptr);
		if(tm == T || typehash(tm->type) != imhash) {
			*m = im;
			*samename = tm;
			return 0;
		}
		// if pointer receiver in method,
		// the method does not exist for value types.
		rcvr = getthisx(tm->type)->type->type;
		if(isptr[rcvr->etype] && !isptr[t0->etype] && !followptr && !isifacemethod(tm->type)) {
			if(debug['r'])
				yyerror("interface pointer mismatch");
			*m = im;
			*samename = nil;
			return 0;
		}
	}
	return 1;
}

// check whether interface type i1 satisifes interface type i2.
int
ifaceokI2I(Type *i1, Type *i2, Type **m)
{
	Type *m1, *m2;

	// if this is too slow,
	// could sort these first
	// and then do one loop.

	for(m2=i2->type; m2; m2=m2->down) {
		for(m1=i1->type; m1; m1=m1->down)
			if(m1->sym == m2->sym && typehash(m1) == typehash(m2))
				goto found;
		*m = m2;
		return 0;
	found:;
	}
	return 1;
}

void
runifacechecks(void)
{
	Icheck *p;
	int lno, wrong, needexplicit;
	Type *m, *t, *iface, *samename;

	lno = lineno;
	for(p=icheck; p; p=p->next) {
		lineno = p->lineno;
		wrong = 0;
		needexplicit = 0;
		m = nil;
		samename = nil;
		if(isinter(p->dst) && isinter(p->src)) {
			iface = p->dst;
			t = p->src;
			needexplicit = !ifaceokI2I(t, iface, &m);
		}
		else if(isinter(p->dst)) {
			t = p->src;
			iface = p->dst;
			wrong = !ifaceokT2I(t, iface, &m, &samename);
		} else {
			t = p->dst;
			iface = p->src;
			wrong = !ifaceokT2I(t, iface, &m, &samename);
			needexplicit = 1;
		}
		if(wrong) {
			if(samename)
				yyerror("%T is not %T\n\tmissing %S%hhT\n\tdo have %S%hhT",
					t, iface, m->sym, m->type, samename->sym, samename->type);
			else
				yyerror("%T is not %T\n\tmissing %S%hhT", t, iface, m->sym, m->type);
		}
		else if(!p->explicit && needexplicit) {
			if(m) {
				if(samename)
					yyerror("need type assertion to use %T as %T\n\tmissing %S %hhT\n\tdo have %S%hhT",
						p->src, p->dst, m->sym, m->type, samename->sym, samename->type);
				else
					yyerror("need type assertion to use %T as %T\n\tmissing %S%hhT",
						p->src, p->dst, m->sym, m->type);
			} else
				yyerror("need type assertion to use %T as %T",
					p->src, p->dst);
		}
	}
	lineno = lno;
}

/*
 * even simpler simtype; get rid of ptr, bool.
 * assuming that the front end has rejected
 * all the invalid conversions (like ptr -> bool)
 */
int
simsimtype(Type *t)
{
	int et;

	if(t == 0)
		return 0;

	et = simtype[t->etype];
	switch(et) {
	case TPTR32:
		et = TUINT32;
		break;
	case TPTR64:
		et = TUINT64;
		break;
	case TBOOL:
		et = TUINT8;
		break;
	}
	return et;
}

NodeList*
concat(NodeList *a, NodeList *b)
{
	if(a == nil)
		return b;
	if(b == nil)
		return a;

	a->end->next = b;
	a->end = b->end;
	b->end = nil;
	return a;
}

NodeList*
list1(Node *n)
{
	NodeList *l;

	if(n == nil)
		return nil;
	if(n->op == OBLOCK && n->ninit == nil)
		return n->list;
	l = mal(sizeof *l);
	l->n = n;
	l->end = l;
	return l;
}

NodeList*
list(NodeList *l, Node *n)
{
	return concat(l, list1(n));
}

NodeList*
listtreecopy(NodeList *l)
{
	NodeList *out;

	out = nil;
	for(; l; l=l->next)
		out = list(out, treecopy(l->n));
	return out;
}

Node*
liststmt(NodeList *l)
{
	Node *n;

	n = nod(OBLOCK, N, N);
	n->list = l;
	if(l)
		n->lineno = l->n->lineno;
	return n;
}

/*
 * return nelem of list
 */
int
count(NodeList *l)
{
	int n;

	n = 0;
	for(; l; l=l->next)
		n++;
	return n;
}

/*
 * return nelem of list
 */
int
structcount(Type *t)
{
	int v;
	Iter s;

	v = 0;
	for(t = structfirst(&s, &t); t != T; t = structnext(&s))
		v++;
	return v;
}

/*
 * return power of 2 of the constant
 * operand. -1 if it is not a power of 2.
 * 1000+ if it is a -(power of 2)
 */
int
powtwo(Node *n)
{
	uvlong v, b;
	int i;

	if(n == N || n->op != OLITERAL || n->type == T)
		goto no;
	if(!isint[n->type->etype])
		goto no;

	v = mpgetfix(n->val.u.xval);
	b = 1ULL;
	for(i=0; i<64; i++) {
		if(b == v)
			return i;
		b = b<<1;
	}

	if(!issigned[n->type->etype])
		goto no;

	v = -v;
	b = 1ULL;
	for(i=0; i<64; i++) {
		if(b == v)
			return i+1000;
		b = b<<1;
	}

no:
	return -1;
}

/*
 * return the unsigned type for
 * a signed integer type.
 * returns T if input is not a
 * signed integer type.
 */
Type*
tounsigned(Type *t)
{

	// this is types[et+1], but not sure
	// that this relation is immutable
	switch(t->etype) {
	default:
		print("tounsigned: unknown type %T\n", t);
		t = T;
		break;
	case TINT:
		t = types[TUINT];
		break;
	case TINT8:
		t = types[TUINT8];
		break;
	case TINT16:
		t = types[TUINT16];
		break;
	case TINT32:
		t = types[TUINT32];
		break;
	case TINT64:
		t = types[TUINT64];
		break;
	}
	return t;
}

/*
 * magic number for signed division
 * see hacker's delight chapter 10
 */
void
smagic(Magic *m)
{
	int p;
	uint64 ad, anc, delta, q1, r1, q2, r2, t;
	uint64 mask, two31;

	m->bad = 0;
	switch(m->w) {
	default:
		m->bad = 1;
		return;
	case 8:
		mask = 0xffLL;
		break;
	case 16:
		mask = 0xffffLL;
		break;
	case 32:
		mask = 0xffffffffLL;
		break;
	case 64:
		mask = 0xffffffffffffffffLL;
		break;
	}
	two31 = mask ^ (mask>>1);

	p = m->w-1;
	ad = m->sd;
	if(m->sd < 0)
		ad = -m->sd;

	// bad denominators
	if(ad == 0 || ad == 1 || ad == two31) {
		m->bad = 1;
		return;
	}

	t = two31;
	ad &= mask;

	anc = t - 1 - t%ad;
	anc &= mask;

	q1 = two31/anc;
	r1 = two31 - q1*anc;
	q1 &= mask;
	r1 &= mask;

	q2 = two31/ad;
	r2 = two31 - q2*ad;
	q2 &= mask;
	r2 &= mask;

	for(;;) {
		p++;
		q1 <<= 1;
		r1 <<= 1;
		q1 &= mask;
		r1 &= mask;
		if(r1 >= anc) {
			q1++;
			r1 -= anc;
			q1 &= mask;
			r1 &= mask;
		}

		q2 <<= 1;
		r2 <<= 1;
		q2 &= mask;
		r2 &= mask;
		if(r2 >= ad) {
			q2++;
			r2 -= ad;
			q2 &= mask;
			r2 &= mask;
		}

		delta = ad - r2;
		delta &= mask;
		if(q1 < delta || (q1 == delta && r1 == 0)) {
			continue;
		}
		break;
	}

	m->sm = q2+1;
	if(m->sm & two31)
		m->sm |= ~mask;
	m->s = p-m->w;
}

/*
 * magic number for unsigned division
 * see hacker's delight chapter 10
 */
void
umagic(Magic *m)
{
	int p;
	uint64 nc, delta, q1, r1, q2, r2;
	uint64 mask, two31;

	m->bad = 0;
	m->ua = 0;

	switch(m->w) {
	default:
		m->bad = 1;
		return;
	case 8:
		mask = 0xffLL;
		break;
	case 16:
		mask = 0xffffLL;
		break;
	case 32:
		mask = 0xffffffffLL;
		break;
	case 64:
		mask = 0xffffffffffffffffLL;
		break;
	}
	two31 = mask ^ (mask>>1);

	m->ud &= mask;
	if(m->ud == 0 || m->ud == two31) {
		m->bad = 1;
		return;
	}
	nc = mask - (-m->ud&mask)%m->ud;
	p = m->w-1;

	q1 = two31/nc;
	r1 = two31 - q1*nc;
	q1 &= mask;
	r1 &= mask;

	q2 = (two31-1) / m->ud;
	r2 = (two31-1) - q2*m->ud;
	q2 &= mask;
	r2 &= mask;

	for(;;) {
		p++;
		if(r1 >= nc-r1) {
			q1 <<= 1;
			q1++;
			r1 <<= 1;
			r1 -= nc;
		} else {
			q1 <<= 1;
			r1 <<= 1;
		}
		q1 &= mask;
		r1 &= mask;
		if(r2+1 >= m->ud-r2) {
			if(q2 >= two31-1) {
				m->ua = 1;
			}
			q2 <<= 1;
			q2++;
			r2 <<= 1;
			r2++;
			r2 -= m->ud;
		} else {
			if(q2 >= two31) {
				m->ua = 1;
			}
			q2 <<= 1;
			r2 <<= 1;
			r2++;
		}
		q2 &= mask;
		r2 &= mask;

		delta = m->ud - 1 - r2;
		delta &= mask;

		if(p < m->w+m->w)
		if(q1 < delta || (q1 == delta && r1 == 0)) {
			continue;
		}
		break;
	}
	m->um = q2+1;
	m->s = p-m->w;
}

Sym*
ngotype(Node *n)
{
	if(n->sym != S && strncmp(n->sym->name, "autotmp_", 8) != 0)
	if(n->type->etype != TFUNC || n->type->thistuple == 0)
	if(n->type->etype != TSTRUCT || n->type->funarg == 0)
		return typename(n->type)->left->sym;
	return S;
}
