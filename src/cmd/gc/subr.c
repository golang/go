// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#include	"md5.h"
#include	"y.tab.h"
#include	"yerr.h"

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
	exits("error");
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
	return strcmp(a->msg, b->msg);
}

void
flusherrors(void)
{
	int i;

	if(nerr == 0)
		return;
	qsort(err, nerr, sizeof err[0], errcmp);
	for(i=0; i<nerr; i++)
		if(i==0 || strcmp(err[i].msg, err[i-1].msg) != 0)
			print("%s", err[i].msg);
	nerr = 0;
}

static void
hcrash(void)
{
	if(debug['h']) {
		flusherrors();
		if(outfile)
			remove(outfile);
		*(volatile int*)0 = 0;
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
	if(nsavederrors+nerrors >= 10 && !debug['e']) {
		flusherrors();
		print("%L: too many errors\n", line);
		errorexit();
	}
}

extern int yystate, yychar;

void
yyerror(char *fmt, ...)
{
	int i;
	static int lastsyntax;
	va_list arg;
	char buf[512], *p;

	if(strncmp(fmt, "syntax error", 12) == 0) {
		nsyntaxerrors++;
		
		if(debug['x'])	
			print("yyerror: yystate=%d yychar=%d\n", yystate, yychar);

		// An unexpected EOF caused a syntax error. Use the previous
		// line number since getc generated a fake newline character.
		if(curio.eofnl)
			lexlineno = prevlineno;

		// only one syntax error per line
		if(lastsyntax == lexlineno)
			return;
		lastsyntax = lexlineno;
			
		if(strstr(fmt, "{ or {") || strstr(fmt, " or ?") || strstr(fmt, " or @")) {
			// The grammar has { and LBRACE but both show up as {.
			// Rewrite syntax error referring to "{ or {" to say just "{".
			strecpy(buf, buf+sizeof buf, fmt);
			p = strstr(buf, "{ or {");
			if(p)
				memmove(p+1, p+6, strlen(p+6)+1);
			
			// The grammar has ? and @ but only for reading imports.
			// Silence them in ordinary errors.
			p = strstr(buf, " or ?");
			if(p)
				memmove(p, p+5, strlen(p+5)+1);
			p = strstr(buf, " or @");
			if(p)
				memmove(p, p+5, strlen(p+5)+1);
			fmt = buf;
		}
		
		// look for parse state-specific errors in list (see go.errors).
		for(i=0; i<nelem(yymsg); i++) {
			if(yymsg[i].yystate == yystate && yymsg[i].yychar == yychar) {
				yyerrorl(lexlineno, "syntax error: %s", yymsg[i].msg);
				return;
			}
		}
		
		// plain "syntax error" gets "near foo" added
		if(strcmp(fmt, "syntax error") == 0) {
			yyerrorl(lexlineno, "syntax error near %s", lexbuf);
			return;
		}
		
		// if bison says "syntax error, more info"; print "syntax error: more info".
		if(fmt[12] == ',') {
			yyerrorl(lexlineno, "syntax error:%s", fmt+13);
			return;
		}

		yyerrorl(lexlineno, "%s", fmt);
		return;
	}

	va_start(arg, fmt);
	adderr(parserline(), fmt, arg);
	va_end(arg);

	hcrash();
	nerrors++;
	if(nsavederrors+nerrors >= 10 && !debug['e']) {
		flusherrors();
		print("%L: too many errors\n", parserline());
		errorexit();
	}
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
warnl(int line, char *fmt, ...)
{
	va_list arg;

	va_start(arg, fmt);
	adderr(line, fmt, arg);
	va_end(arg);
	if(debug['m'])
		flusherrors();
}

void
fatal(char *fmt, ...)
{
	va_list arg;

	flusherrors();

	print("%L: internal compiler error: ", lineno);
	va_start(arg, fmt);
	vfprint(1, fmt, arg);
	va_end(arg);
	print("\n");
	
	// If this is a released compiler version, ask for a bug report.
	if(strncmp(getgoversion(), "release", 7) == 0) {
		print("\n");
		print("Please file a bug report including a short program that triggers the error.\n");
		print("http://code.google.com/p/go/issues/entry?template=compilerbug\n");
	}
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
				print("pragma %s", file);
			else
			if(off > 0)
				print("line %s", file);
			else
				print("import %s", file);
		} else
			print("end of import");
		print(" at line %L\n", lexlineno);
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
lookup(char *name)
{
	return pkglookup(name, localpkg);
}

Sym*
pkglookup(char *name, Pkg *pkg)
{
	Sym *s;
	uint32 h;
	int c;

	h = stringhash(name) % NHASH;
	c = name[0];
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != c || s->pkg != pkg)
			continue;
		if(strcmp(s->name, name) == 0)
			return s;
	}

	s = mal(sizeof(*s));
	s->name = mal(strlen(name)+1);
	strcpy(s->name, name);

	s->pkg = pkg;

	s->link = hash[h];
	hash[h] = s;
	s->lexical = LNAME;

	return s;
}

Sym*
restrictlookup(char *name, Pkg *pkg)
{
	if(!exportname(name) && pkg != localpkg)
		yyerror("cannot refer to unexported name %s.%s", pkg->name, name);
	return pkglookup(name, pkg);
}


// find all the exported symbols in package opkg
// and make them available in the current package
void
importdot(Pkg *opkg, Node *pack)
{
	Sym *s, *s1;
	uint32 h;
	int n;
	char *pkgerror;

	n = 0;
	for(h=0; h<NHASH; h++) {
		for(s = hash[h]; s != S; s = s->link) {
			if(s->pkg != opkg)
				continue;
			if(s->def == N)
				continue;
			if(!exportname(s->name) || utfrune(s->name, 0xb7))	// 0xb7 = center dot
				continue;
			s1 = lookup(s->name);
			if(s1->def != N) {
				pkgerror = smprint("during import \"%Z\"", opkg->path);
				redeclare(s1, pkgerror);
				continue;
			}
			s1->def = s->def;
			s1->block = s->block;
			s1->def->pack = pack;
			s1->origpkg = opkg;
			n++;
		}
	}
	if(n == 0) {
		// can't possibly be used - there were no symbols
		yyerrorl(pack->lineno, "imported and not used: \"%Z\"", opkg->path);
	}
}

static void
gethunk(void)
{
	char *h;
	int32 nh;

	nh = NHUNK;
	if(thunk >= 10L*NHUNK)
		nh = 10L*NHUNK;
	h = (char*)malloc(nh);
	if(h == nil) {
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

	if(n >= NHUNK) {
		p = malloc(n);
		if(p == nil) {
			flusherrors();
			yyerror("out of memory");
			errorexit();
		}
		memset(p, 0, n);
		return p;
	}

	while((uintptr)hunk & MAXALIGN) {
		hunk++;
		nhunk--;
	}
	if(nhunk < n)
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
		if(on+n >= NHUNK) {
			q = mal(on+n);
			memmove(q, p, on);
			return q;
		}
		if(nhunk < on+n)
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
	n->orig = n;
	n->curfn = curfn;
	return n;
}

void
saveorignode(Node *n)
{
	Node *norig;

	if(n->orig != N)
		return;
	norig = nod(n->op, N, N);
	*norig = *n;
	n->orig = norig;
}

// ispaddedfield returns whether the given field
// is followed by padding. For the case where t is
// the last field, total gives the size of the enclosing struct.
static int
ispaddedfield(Type *t, vlong total)
{
	if(t->etype != TFIELD)
		fatal("ispaddedfield called non-field %T", t);
	if(t->down == T)
		return t->width + t->type->width != total;
	return t->width + t->type->width != t->down->width;
}

int
algtype1(Type *t, Type **bad)
{
	int a, ret;
	Type *t1;
	
	if(bad)
		*bad = T;

	switch(t->etype) {
	case TANY:
	case TFORW:
		// will be defined later.
		*bad = t;
		return -1;

	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TINT:
	case TUINT:
	case TUINTPTR:
	case TBOOL:
	case TPTR32:
	case TPTR64:
	case TCHAN:
	case TUNSAFEPTR:
		return AMEM;

	case TFUNC:
	case TMAP:
		if(bad)
			*bad = t;
		return ANOEQ;

	case TFLOAT32:
		return AFLOAT32;

	case TFLOAT64:
		return AFLOAT64;

	case TCOMPLEX64:
		return ACPLX64;

	case TCOMPLEX128:
		return ACPLX128;

	case TSTRING:
		return ASTRING;
	
	case TINTER:
		if(isnilinter(t))
			return ANILINTER;
		return AINTER;
	
	case TARRAY:
		if(isslice(t)) {
			if(bad)
				*bad = t;
			return ANOEQ;
		}
		if(t->bound == 0)
			return AMEM;
		a = algtype1(t->type, bad);
		if(a == ANOEQ || a == AMEM) {
			if(a == ANOEQ && bad)
				*bad = t;
			return a;
		}
		return -1;  // needs special compare

	case TSTRUCT:
		if(t->type != T && t->type->down == T) {
			// One-field struct is same as that one field alone.
			return algtype1(t->type->type, bad);
		}
		ret = AMEM;
		for(t1=t->type; t1!=T; t1=t1->down) {
			// Blank fields and padding must be ignored,
			// so need special compare.
			if(isblanksym(t1->sym) || ispaddedfield(t1, t->width)) {
				ret = -1;
				continue;
			}
			a = algtype1(t1->type, bad);
			if(a == ANOEQ)
				return ANOEQ;  // not comparable
			if(a != AMEM)
				ret = -1;  // needs special compare
		}
		return ret;
	}

	fatal("algtype1: unexpected type %T", t);
	return 0;
}

int
algtype(Type *t)
{
	int a;
	
	a = algtype1(t, nil);
	if(a == AMEM || a == ANOEQ) {
		if(isslice(t))
			return ASLICE;
		switch(t->width) {
		case 0:
			return a + AMEM0 - AMEM;
		case 1:
			return a + AMEM8 - AMEM;
		case 2:
			return a + AMEM16 - AMEM;
		case 4:
			return a + AMEM32 - AMEM;
		case 8:
			return a + AMEM64 - AMEM;
		case 16:
			return a + AMEM128 - AMEM;
		}
	}
	return a;
}

Type*
maptype(Type *key, Type *val)
{
	Type *t;
	Type *bad;
	int atype;

	if(key != nil) {
		atype = algtype1(key, &bad);
		switch(bad == T ? key->etype : bad->etype) {
		default:
			if(atype == ANOEQ)
				yyerror("invalid map key type %T", key);
			break;
		case TANY:
			// will be resolved later.
			break;
		case TFORW:
			// map[key] used during definition of key.
			// postpone check until key is fully defined.
			// if there are multiple uses of map[key]
			// before key is fully defined, the error
			// will only be printed for the first one.
			// good enough.
			if(key->maplineno == 0)
				key->maplineno = lineno;
			break;
		}
	}
	t = typ(TMAP);
	t->down = key;
	t->type = val;
	return t;
}

Type*
typ(int et)
{
	Type *t;

	t = mal(sizeof(*t));
	t->etype = et;
	t->width = BADWIDTH;
	t->lineno = lineno;
	t->orig = t;
	return t;
}

static int
methcmp(const void *va, const void *vb)
{
	Type *a, *b;
	int i;
	
	a = *(Type**)va;
	b = *(Type**)vb;
	if(a->sym == S && b->sym == S)
		return 0;
	if(a->sym == S)
		return -1;
	if(b->sym == S)
		return 1;
	i = strcmp(a->sym->name, b->sym->name);
	if(i != 0)
		return i;
	if(!exportname(a->sym->name)) {
		i = strcmp(a->sym->pkg->path->s, b->sym->pkg->path->s);
		if(i != 0)
			return i;
	}
	return 0;
}

Type*
sortinter(Type *t)
{
	Type *f;
	int i;
	Type **a;
	
	if(t->type == nil || t->type->down == nil)
		return t;

	i=0;
	for(f=t->type; f; f=f->down)
		i++;
	a = mal(i*sizeof f);
	i = 0;
	for(f=t->type; f; f=f->down)
		a[i++] = f;
	qsort(a, i, sizeof a[0], methcmp);
	while(i-- > 0) {
		a[i]->down = f;
		f = a[i];
	}
	t->type = f;
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

Node*
nodfltconst(Mpflt* v)
{
	Node *c;

	c = nod(OLITERAL, N, N);
	c->addable = 1;
	c->val.u.fval = mal(sizeof(*c->val.u.fval));
	mpmovefltflt(c->val.u.fval, v);
	c->val.ctype = CTFLT;
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
	Type *r;
	int64 bound;

	bound = -1;	// open bound
	typecheck(&b, Erv);
	if(b != nil) {
		switch(consttype(b)) {
		default:
			yyerror("array bound must be an integer expression");
			break;
		case CTINT:
		case CTRUNE:
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
		m->orig = m;
		m->left = treecopy(n->left);
		m->right = treecopy(n->right);
		m->list = listtreecopy(n->list);
		if(m->defn)
			abort();
		break;

	case ONONAME:
		if(n->sym == lookup("iota")) {
			// Not sure yet whether this is the real iota,
			// but make a copy of the Node* just in case,
			// so that all the copies of this const definition
			// don't have the same iota value.
			m = nod(OXXX, N, N);
			*m = *n;
			m->iota = iota;
			break;
		}
		// fall through
	case ONAME:
	case OLITERAL:
	case OTYPE:
		m = n;
		break;
	}
	return m;
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
	if(n == N)
		return 0;
	return isblanksym(n->sym);
}

int
isblanksym(Sym *s)
{
	char *p;

	if(s == S)
		return 0;
	p = s->name;
	if(p == nil)
		return 0;
	return p[0] == '_' && p[1] == '\0';
}

int
isinter(Type *t)
{
	return t != T && t->etype == TINTER;
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
isideal(Type *t)
{
	if(t == T)
		return 0;
	if(t == idealstring || t == idealbool)
		return 1;
	switch(t->etype) {
	case TNIL:
	case TIDEAL:
		return 1;
	}
	return 0;
}

/*
 * given receiver of type t (t == r or t == *r)
 * return type to hang methods off (r).
 */
Type*
methtype(Type *t, int mustname)
{
	if(t == T)
		return T;

	// strip away pointer if it's there
	if(isptr[t->etype]) {
		if(t->sym != S)
			return T;
		t = t->type;
		if(t == T)
			return T;
	}

	// need a type name
	if(t->sym == S && (mustname || t->etype != TSTRUCT))
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
cplxsubtype(int et)
{
	switch(et) {
	case TCOMPLEX64:
		return TFLOAT32;
	case TCOMPLEX128:
		return TFLOAT64;
	}
	fatal("cplxsubtype: %E\n", et);
	return 0;
}

static int
eqnote(Strlit *a, Strlit *b)
{
	if(a == b)
		return 1;
	if(a == nil || b == nil)
		return 0;
	if(a->len != b->len)
		return 0;
	return memcmp(a->s, b->s, a->len) == 0;
}

typedef struct TypePairList TypePairList;
struct TypePairList
{
	Type *t1;
	Type *t2;
	TypePairList *next;
};

static int
onlist(TypePairList *l, Type *t1, Type *t2) 
{
	for(; l; l=l->next)
		if((l->t1 == t1 && l->t2 == t2) || (l->t1 == t2 && l->t2 == t1))
			return 1;
	return 0;
}

static int eqtype1(Type*, Type*, TypePairList*);

// Return 1 if t1 and t2 are identical, following the spec rules.
//
// Any cyclic type must go through a named type, and if one is
// named, it is only identical to the other if they are the same
// pointer (t1 == t2), so there's no chance of chasing cycles
// ad infinitum, so no need for a depth counter.
int
eqtype(Type *t1, Type *t2)
{
	return eqtype1(t1, t2, nil);
}

static int
eqtype1(Type *t1, Type *t2, TypePairList *assumed_equal)
{
	TypePairList l;

	if(t1 == t2)
		return 1;
	if(t1 == T || t2 == T || t1->etype != t2->etype)
		return 0;
	if(t1->sym || t2->sym) {
		// Special case: we keep byte and uint8 separate
		// for error messages.  Treat them as equal.
		switch(t1->etype) {
		case TUINT8:
			if((t1 == types[TUINT8] || t1 == bytetype) && (t2 == types[TUINT8] || t2 == bytetype))
				return 1;
			break;
		case TINT:
		case TINT32:
			if((t1 == types[runetype->etype] || t1 == runetype) && (t2 == types[runetype->etype] || t2 == runetype))
				return 1;
			break;
		}
		return 0;
	}

	if(onlist(assumed_equal, t1, t2))
		return 1;
	l.next = assumed_equal;
	l.t1 = t1;
	l.t2 = t2;

	switch(t1->etype) {
	case TINTER:
	case TSTRUCT:
		for(t1=t1->type, t2=t2->type; t1 && t2; t1=t1->down, t2=t2->down) {
			if(t1->etype != TFIELD || t2->etype != TFIELD)
				fatal("struct/interface missing field: %T %T", t1, t2);
			if(t1->sym != t2->sym || t1->embedded != t2->embedded || !eqtype1(t1->type, t2->type, &l) || !eqnote(t1->note, t2->note))
				goto no;
		}
		if(t1 == T && t2 == T)
			goto yes;
		goto no;

	case TFUNC:
		// Loop over structs: receiver, in, out.
		for(t1=t1->type, t2=t2->type; t1 && t2; t1=t1->down, t2=t2->down) {
			Type *ta, *tb;

			if(t1->etype != TSTRUCT || t2->etype != TSTRUCT)
				fatal("func missing struct: %T %T", t1, t2);

			// Loop over fields in structs, ignoring argument names.
			for(ta=t1->type, tb=t2->type; ta && tb; ta=ta->down, tb=tb->down) {
				if(ta->etype != TFIELD || tb->etype != TFIELD)
					fatal("func struct missing field: %T %T", ta, tb);
				if(ta->isddd != tb->isddd || !eqtype1(ta->type, tb->type, &l))
					goto no;
			}
			if(ta != T || tb != T)
				goto no;
		}
		if(t1 == T && t2 == T)
			goto yes;
		goto no;
	
	case TARRAY:
		if(t1->bound != t2->bound)
			goto no;
		break;
	
	case TCHAN:
		if(t1->chan != t2->chan)
			goto no;
		break;
	}

	if(eqtype1(t1->down, t2->down, &l) && eqtype1(t1->type, t2->type, &l))
		goto yes;
	goto no;

yes:
	return 1;

no:
	return 0;
}

// Are t1 and t2 equal struct types when field names are ignored?
// For deciding whether the result struct from g can be copied
// directly when compiling f(g()).
int
eqtypenoname(Type *t1, Type *t2)
{
	if(t1 == T || t2 == T || t1->etype != TSTRUCT || t2->etype != TSTRUCT)
		return 0;

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

// Is type src assignment compatible to type dst?
// If so, return op code to use in conversion.
// If not, return 0.
int
assignop(Type *src, Type *dst, char **why)
{
	Type *missing, *have;
	int ptr;

	if(why != nil)
		*why = "";

	// TODO(rsc,lvd): This behaves poorly in the presence of inlining.
	// https://code.google.com/p/go/issues/detail?id=2795
	if(safemode && importpkg == nil && src != T && src->etype == TUNSAFEPTR) {
		yyerror("cannot use unsafe.Pointer");
		errorexit();
	}

	if(src == dst)
		return OCONVNOP;
	if(src == T || dst == T || src->etype == TFORW || dst->etype == TFORW || src->orig == T || dst->orig == T)
		return 0;

	// 1. src type is identical to dst.
	if(eqtype(src, dst))
		return OCONVNOP;
	
	// 2. src and dst have identical underlying types
	// and either src or dst is not a named type or
	// both are interface types.
	if(eqtype(src->orig, dst->orig) && (src->sym == S || dst->sym == S || src->etype == TINTER))
		return OCONVNOP;

	// 3. dst is an interface type and src implements dst.
	if(dst->etype == TINTER && src->etype != TNIL) {
		if(implements(src, dst, &missing, &have, &ptr))
			return OCONVIFACE;

		// we'll have complained about this method anyway, supress spurious messages.
		if(have && have->sym == missing->sym && (have->type->broke || missing->type->broke))
			return OCONVIFACE;

		if(why != nil) {
			if(isptrto(src, TINTER))
				*why = smprint(":\n\t%T is pointer to interface, not interface", src);
			else if(have && have->sym == missing->sym)
				*why = smprint(":\n\t%T does not implement %T (wrong type for %S method)\n"
					"\t\thave %S%hhT\n\t\twant %S%hhT", src, dst, missing->sym,
					have->sym, have->type, missing->sym, missing->type);
			else if(ptr)
				*why = smprint(":\n\t%T does not implement %T (%S method requires pointer receiver)",
					src, dst, missing->sym);
			else if(have)
				*why = smprint(":\n\t%T does not implement %T (missing %S method)\n"
					"\t\thave %S%hhT\n\t\twant %S%hhT", src, dst, missing->sym,
					have->sym, have->type, missing->sym, missing->type);
			else
				*why = smprint(":\n\t%T does not implement %T (missing %S method)",
					src, dst, missing->sym);
		}
		return 0;
	}
	if(isptrto(dst, TINTER)) {
		if(why != nil)
			*why = smprint(":\n\t%T is pointer to interface, not interface", dst);
		return 0;
	}
	if(src->etype == TINTER && dst->etype != TBLANK) {
		if(why != nil && implements(dst, src, &missing, &have, &ptr))
			*why = ": need type assertion";
		return 0;
	}

	// 4. src is a bidirectional channel value, dst is a channel type,
	// src and dst have identical element types, and
	// either src or dst is not a named type.
	if(src->etype == TCHAN && src->chan == Cboth && dst->etype == TCHAN)
	if(eqtype(src->type, dst->type) && (src->sym == S || dst->sym == S))
		return OCONVNOP;

	// 5. src is the predeclared identifier nil and dst is a nillable type.
	if(src->etype == TNIL) {
		switch(dst->etype) {
		case TARRAY:
			if(dst->bound != -100)	// not slice
				break;
		case TPTR32:
		case TPTR64:
		case TFUNC:
		case TMAP:
		case TCHAN:
		case TINTER:
			return OCONVNOP;
		}
	}

	// 6. rule about untyped constants - already converted by defaultlit.
	
	// 7. Any typed value can be assigned to the blank identifier.
	if(dst->etype == TBLANK)
		return OCONVNOP;

	return 0;
}

// Can we convert a value of type src to a value of type dst?
// If so, return op code to use in conversion (maybe OCONVNOP).
// If not, return 0.
int
convertop(Type *src, Type *dst, char **why)
{
	int op;
	
	if(why != nil)
		*why = "";

	if(src == dst)
		return OCONVNOP;
	if(src == T || dst == T)
		return 0;
	
	// 1. src can be assigned to dst.
	if((op = assignop(src, dst, why)) != 0)
		return op;

	// The rules for interfaces are no different in conversions
	// than assignments.  If interfaces are involved, stop now
	// with the good message from assignop.
	// Otherwise clear the error.
	if(src->etype == TINTER || dst->etype == TINTER)
		return 0;
	if(why != nil)
		*why = "";

	// 2. src and dst have identical underlying types.
	if(eqtype(src->orig, dst->orig))
		return OCONVNOP;
	
	// 3. src and dst are unnamed pointer types 
	// and their base types have identical underlying types.
	if(isptr[src->etype] && isptr[dst->etype] && src->sym == S && dst->sym == S)
	if(eqtype(src->type->orig, dst->type->orig))
		return OCONVNOP;

	// 4. src and dst are both integer or floating point types.
	if((isint[src->etype] || isfloat[src->etype]) && (isint[dst->etype] || isfloat[dst->etype])) {
		if(simtype[src->etype] == simtype[dst->etype])
			return OCONVNOP;
		return OCONV;
	}

	// 5. src and dst are both complex types.
	if(iscomplex[src->etype] && iscomplex[dst->etype]) {
		if(simtype[src->etype] == simtype[dst->etype])
			return OCONVNOP;
		return OCONV;
	}

	// 6. src is an integer or has type []byte or []rune
	// and dst is a string type.
	if(isint[src->etype] && dst->etype == TSTRING)
		return ORUNESTR;

	if(isslice(src) && dst->etype == TSTRING) {
		if(src->type->etype == bytetype->etype)
			return OARRAYBYTESTR;
		if(src->type->etype == runetype->etype)
			return OARRAYRUNESTR;
	}
	
	// 7. src is a string and dst is []byte or []rune.
	// String to slice.
	if(src->etype == TSTRING && isslice(dst)) {
		if(dst->type->etype == bytetype->etype)
			return OSTRARRAYBYTE;
		if(dst->type->etype == runetype->etype)
			return OSTRARRAYRUNE;
	}
	
	// 8. src is a pointer or uintptr and dst is unsafe.Pointer.
	if((isptr[src->etype] || src->etype == TUINTPTR) && dst->etype == TUNSAFEPTR)
		return OCONVNOP;

	// 9. src is unsafe.Pointer and dst is a pointer or uintptr.
	if(src->etype == TUNSAFEPTR && (isptr[dst->etype] || dst->etype == TUINTPTR))
		return OCONVNOP;

	return 0;
}

// Convert node n for assignment to type t.
Node*
assignconv(Node *n, Type *t, char *context)
{
	int op;
	Node *r, *old;
	char *why;
	
	if(n == N || n->type == T || n->type->broke)
		return n;

	old = n;
	old->diag++;  // silence errors about n; we'll issue one below
	defaultlit(&n, t);
	old->diag--;
	if(t->etype == TBLANK)
		return n;

	// Convert ideal bool from comparison to plain bool
	// if the next step is non-bool (like interface{}).
	if(n->type == idealbool && t->etype != TBOOL) {
		if(n->op == ONAME || n->op == OLITERAL) {
			r = nod(OCONVNOP, n, N);
			r->type = types[TBOOL];
			r->typecheck = 1;
			r->implicit = 1;
			n = r;
		}
	}

	if(eqtype(n->type, t))
		return n;

	op = assignop(n->type, t, &why);
	if(op == 0) {
		yyerror("cannot use %lN as type %T in %s%s", n, t, context, why);
		op = OCONV;
	}

	r = nod(op, n, N);
	r->type = t;
	r->typecheck = 1;
	r->implicit = 1;
	r->orig = n->orig;
	return r;
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
	if(t->orig == t)
		nt->orig = nt;
	return nt;
}

static Type*
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

	s = pkglookup(name, runtimepkg);
	if(s == S || s->def == N)
		fatal("syslook: can't find runtime.%s", name);

	if(!copy)
		return s->def;

	n = nod(0, N, N);
	*n = *s->def;
	n->type = deep(s->def->type);

	return n;
}

/*
 * compute a hash value for type t.
 * if t is a method type, ignore the receiver
 * so that the hash can be used in interface checks.
 * %T already contains
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
		p = smprint("%-uT", t);
		t->thistuple = 1;
	} else
		p = smprint("%-uT", t);
	//print("typehash: %s\n", p);
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
		fatal("ptrto: no tptr");
	t1 = typ(tptr);
	t1->type = t;
	t1->width = widthptr;
	t1->align = widthptr;
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
			print("%O %S G%d %T\n", n->op, n->sym, n->vargen, n->type);
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

	if(n->ninit != nil) {
		ul = UINF;
		goto out;
	}

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
	case OANDAND:
	case OOROR:
		// hard with race detector
		if(flag_race) {
			ul = UINF;
			goto out;
		}
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
	if(ul > 200)
		ul = 200; // clamp to uchar with room to grow
	n->ullman = ul;
}

void
badtype(int o, Type *tl, Type *tr)
{
	Fmt fmt;
	char *s;
	
	fmtstrinit(&fmt);
	if(tl != T)
		fmtprint(&fmt, "\n	%T", tl);
	if(tr != T)
		fmtprint(&fmt, "\n	%T", tr);

	// common mistake: *struct and *interface.
	if(tl && tr && isptr[tl->etype] && isptr[tr->etype]) {
		if(tl->type->etype == TSTRUCT && tr->type->etype == TINTER)
			fmtprint(&fmt, "\n	(*struct vs *interface)");
		else if(tl->type->etype == TINTER && tr->type->etype == TSTRUCT)
			fmtprint(&fmt, "\n	(*interface vs *struct)");
	}
	s = fmtstrflush(&fmt);
	yyerror("illegal types for operand: %O%s", o, s);
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

/*
 * return side effect-free n, appending side effects to init.
 * result is assignable if n is.
 */
Node*
safeexpr(Node *n, NodeList **init)
{
	Node *l;
	Node *r;
	Node *a;

	if(n == N)
		return N;

	if(n->ninit) {
		walkstmtlist(n->ninit);
		*init = concat(*init, n->ninit);
		n->ninit = nil;
	}

	switch(n->op) {
	case ONAME:
	case OLITERAL:
		return n;

	case ODOT:
		l = safeexpr(n->left, init);
		if(l == n->left)
			return n;
		r = nod(OXXX, N, N);
		*r = *n;
		r->left = l;
		typecheck(&r, Erv);
		walkexpr(&r, init);
		return r;

	case ODOTPTR:
	case OIND:
		l = safeexpr(n->left, init);
		if(l == n->left)
			return n;
		a = nod(OXXX, N, N);
		*a = *n;
		a->left = l;
		walkexpr(&a, init);
		return a;

	case OINDEX:
	case OINDEXMAP:
		l = safeexpr(n->left, init);
		r = safeexpr(n->right, init);
		if(l == n->left && r == n->right)
			return n;
		a = nod(OXXX, N, N);
		*a = *n;
		a->left = l;
		a->right = r;
		walkexpr(&a, init);
		return a;
	}

	// make a copy; must not be used as an lvalue
	if(islvalue(n))
		fatal("missing lvalue case in safeexpr: %N", n);
	return cheapexpr(n, init);
}

Node*
copyexpr(Node *n, Type *t, NodeList **init)
{
	Node *a, *l;
	
	l = temp(t);
	a = nod(OAS, l, n);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);
	return l;
}

/*
 * return side-effect free and cheap n, appending side effects to init.
 * result may not be assignable.
 */
Node*
cheapexpr(Node *n, NodeList **init)
{
	switch(n->op) {
	case ONAME:
	case OLITERAL:
		return n;
	}

	return copyexpr(n, n->type, init);
}

/*
 * return n in a local variable of type t if it is not already.
 * the value is guaranteed not to change except by direct
 * assignment to it.
 */
Node*
localexpr(Node *n, Type *t, NodeList **init)
{
	if(n->op == ONAME && !n->addrtaken &&
		(n->class == PAUTO || n->class == PPARAM || n->class == PPARAMOUT) &&
		convertop(n->type, t, nil) == OCONVNOP)
		return n;
	
	return copyexpr(n, t, init);
}

void
setmaxarg(Type *t)
{
	int64 w;

	dowidth(t);
	w = t->argwid;
	if(t->argwid >= MAXWIDTH)
		fatal("bad argwid %T", t);
	if(w > maxarg)
		maxarg = w;
}

/*
 * unicode-aware case-insensitive strcmp
 */

static int
ucistrcmp(char *p, char *q)
{
	Rune rp, rq;

	while(*p || *q) {
		if(*p == 0)
			return +1;
		if(*q == 0)
			return -1;
		p += chartorune(&rp, p);
		q += chartorune(&rq, q);
		rp = tolowerrune(rp);
		rq = tolowerrune(rq);
		if(rp < rq)
			return -1;
		if(rp > rq)
			return +1;
	}
	return 0;
}

/*
 * code to resolve elided DOTs
 * in embedded types
 */

// search depth 0 --
// return count of fields+methods
// found with a given name
static int
lookdot0(Sym *s, Type *t, Type **save, int ignorecase)
{
	Type *f, *u;
	int c;

	u = t;
	if(isptr[u->etype])
		u = u->type;

	c = 0;
	if(u->etype == TSTRUCT || u->etype == TINTER) {
		for(f=u->type; f!=T; f=f->down)
			if(f->sym == s || (ignorecase && ucistrcmp(f->sym->name, s->name) == 0)) {
				if(save)
					*save = f;
				c++;
			}
	}
	u = methtype(t, 0);
	if(u != T) {
		for(f=u->method; f!=T; f=f->down)
			if(f->embedded == 0 && (f->sym == s || (ignorecase && ucistrcmp(f->sym->name, s->name) == 0))) {
				if(save)
					*save = f;
				c++;
			}
	}
	return c;
}

// search depth d for field/method s --
// return count of fields+methods
// found at search depth.
// answer is in dotlist array and
// count of number of ways is returned.
int
adddot1(Sym *s, Type *t, int d, Type **save, int ignorecase)
{
	Type *f, *u;
	int c, a;

	if(t->trecur)
		return 0;
	t->trecur = 1;

	if(d == 0) {
		c = lookdot0(s, t, save, ignorecase);
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
		a = adddot1(s, f->type, d, save, ignorecase);
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
	
	if(n->left->op == OTYPE)
		goto ret;

	if(n->right->op != ONAME)
		goto ret;
	s = n->right->sym;
	if(s == S)
		goto ret;

	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(s, t, d, nil, 0);
		if(c > 0)
			goto out;
	}
	goto ret;

out:
	if(c > 1) {
		yyerror("ambiguous selector %N", n);
		n->left = N;
		return n;
	}

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

	u = methtype(t, 0);
	if(u != T) {
		for(f=u->method; f!=T; f=f->down) {
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
expandmeth(Type *t)
{
	Symlink *sl;
	Type *f;
	int c, d;

	if(t == T || t->xmethod != nil)
		return;

	// mark top-level method symbols
	// so that expand1 doesn't consider them.
	for(f=t->method; f != nil; f=f->down)
		f->sym->flags |= SymUniq;

	// generate all reachable methods
	slist = nil;
	expand1(t, nelem(dotlist)-1, 0);

	// check each method to be uniquely reachable
	for(sl=slist; sl!=nil; sl=sl->link) {
		sl->field->sym->flags &= ~SymUniq;
		for(d=0; d<nelem(dotlist); d++) {
			c = adddot1(sl->field->sym, t, d, &f, 0);
			if(c == 0)
				continue;
			if(c == 1) {
				// addot1 may have dug out arbitrary fields, we only want methods.
				if(f->type->etype == TFUNC && f->type->thistuple > 0) {
					sl->good = 1;
					sl->field = f;
				}
			}
			break;
		}
	}

	for(f=t->method; f != nil; f=f->down)
		f->sym->flags &= ~SymUniq;

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
static NodeList*
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
		if(mustname && (t->sym == nil || strcmp(t->sym->name, "_") == 0)) {
			// invent a name so that we can refer to it in the trampoline
			snprint(buf, sizeof buf, ".anon%d", gen++);
			n = newname(lookup(buf));
		} else if(t->sym)
			n = newname(t->sym);
		a = nod(ODCLFIELD, n, typenod(t->type));
		a->isddd = t->isddd;
		if(n != N)
			n->isddd = t->isddd;
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
genwrapper(Type *rcvr, Type *method, Sym *newnam, int iface)
{
	Node *this, *fn, *call, *n, *t, *pad;
	NodeList *l, *args, *in, *out;
	Type *tpad;
	int isddd;
	Val v;

	if(debug['r'])
		print("genwrapper rcvrtype=%T method=%T newnam=%S\n",
			rcvr, method, newnam);

	lineno = 1;	// less confusing than end of input

	dclcontext = PEXTERN;
	markdcl();

	this = nod(ODCLFIELD, newname(lookup(".this")), typenod(rcvr));
	this->left->ntype = this->right;
	in = structargs(getinarg(method->type), 1);
	out = structargs(getoutarg(method->type), 0);

	t = nod(OTFUNC, N, N);
	l = list1(this);
	if(iface && rcvr->width < types[tptr]->width) {
		// Building method for interface table and receiver
		// is smaller than the single pointer-sized word
		// that the interface call will pass in.
		// Add a dummy padding argument after the
		// receiver to make up the difference.
		tpad = typ(TARRAY);
		tpad->type = types[TUINT8];
		tpad->bound = types[tptr]->width - rcvr->width;
		pad = nod(ODCLFIELD, newname(lookup(".pad")), typenod(tpad));
		l = list(l, pad);
	}
	t->list = concat(l, in);
	t->rlist = out;

	fn = nod(ODCLFUNC, N, N);
	fn->nname = newname(newnam);
	fn->nname->defn = fn;
	fn->nname->ntype = t;
	declare(fn->nname, PFUNC);
	funchdr(fn);

	// arg list
	args = nil;
	isddd = 0;
	for(l=in; l; l=l->next) {
		args = list(args, l->n->left);
		isddd = l->n->left->isddd;
	}
	
	// generate nil pointer check for better error
	if(isptr[rcvr->etype] && rcvr->type == getthisx(method->type)->type->type) {
		// generating wrapper from *T to T.
		n = nod(OIF, N, N);
		n->ntest = nod(OEQ, this->left, nodnil());
		// these strings are already in the reflect tables,
		// so no space cost to use them here.
		l = nil;
		v.ctype = CTSTR;
		v.u.sval = strlit(rcvr->type->sym->pkg->name);  // package name
		l = list(l, nodlit(v));
		v.u.sval = strlit(rcvr->type->sym->name);  // type name
		l = list(l, nodlit(v));
		v.u.sval = strlit(method->sym->name);
		l = list(l, nodlit(v));  // method name
		call = nod(OCALL, syslook("panicwrap", 0), N);
		call->list = l;
		n->nbody = list1(call);
		fn->nbody = list(fn->nbody, n);
	}

	// generate call
	call = nod(OCALL, adddot(nod(OXDOT, this->left, newname(method->sym))), N);
	call->list = args;
	call->isddd = isddd;
	if(method->type->outtuple > 0) {
		n = nod(ORETURN, N, N);
		n->list = list1(call);
		call = n;
	}
	fn->nbody = list(fn->nbody, call);

	if(0 && debug['r'])
		dumplist("genwrapper body", fn->nbody);

	funcbody(fn);
	curfn = fn;
	// wrappers where T is anonymous (struct{ NamedType }) can be duplicated.
	if(rcvr->etype == TSTRUCT || isptr[rcvr->etype] && rcvr->type->etype == TSTRUCT)
		fn->dupok = 1;
	typecheck(&fn, Etop);
	typechecklist(fn->nbody, Etop);
	inlcalls(fn);
	curfn = nil;
	funccompile(fn, 0);
}

static Node*
hashmem(Type *t)
{
	Node *tfn, *n;
	Sym *sym;
	
	sym = pkglookup("memhash", runtimepkg);

	n = newname(sym);
	n->class = PFUNC;
	tfn = nod(OTFUNC, N, N);
	tfn->list = list(tfn->list, nod(ODCLFIELD, N, typenod(ptrto(types[TUINTPTR]))));
	tfn->list = list(tfn->list, nod(ODCLFIELD, N, typenod(types[TUINTPTR])));
	tfn->list = list(tfn->list, nod(ODCLFIELD, N, typenod(ptrto(t))));
	typecheck(&tfn, Etype);
	n->type = tfn->type;
	return n;
}

static Node*
hashfor(Type *t)
{
	int a;
	Sym *sym;
	Node *tfn, *n;

	a = algtype1(t, nil);
	switch(a) {
	case AMEM:
		return hashmem(t);
	case AINTER:
		sym = pkglookup("interhash", runtimepkg);
		break;
	case ANILINTER:
		sym = pkglookup("nilinterhash", runtimepkg);
		break;
	case ASTRING:
		sym = pkglookup("strhash", runtimepkg);
		break;
	case AFLOAT32:
		sym = pkglookup("f32hash", runtimepkg);
		break;
	case AFLOAT64:
		sym = pkglookup("f64hash", runtimepkg);
		break;
	case ACPLX64:
		sym = pkglookup("c64hash", runtimepkg);
		break;
	case ACPLX128:
		sym = pkglookup("c128hash", runtimepkg);
		break;
	default:
		sym = typesymprefix(".hash", t);
		break;
	}

	n = newname(sym);
	n->class = PFUNC;
	tfn = nod(OTFUNC, N, N);
	tfn->list = list(tfn->list, nod(ODCLFIELD, N, typenod(ptrto(types[TUINTPTR]))));
	tfn->list = list(tfn->list, nod(ODCLFIELD, N, typenod(types[TUINTPTR])));
	tfn->list = list(tfn->list, nod(ODCLFIELD, N, typenod(ptrto(t))));
	typecheck(&tfn, Etype);
	n->type = tfn->type;
	return n;
}

/*
 * Generate a helper function to compute the hash of a value of type t.
 */
void
genhash(Sym *sym, Type *t)
{
	Node *n, *fn, *np, *nh, *ni, *call, *nx, *na, *tfn;
	Node *hashel;
	Type *first, *t1;
	int old_safemode;
	int64 size, mul, offend;

	if(debug['r'])
		print("genhash %S %T\n", sym, t);

	lineno = 1;  // less confusing than end of input
	dclcontext = PEXTERN;
	markdcl();

	// func sym(h *uintptr, s uintptr, p *T)
	fn = nod(ODCLFUNC, N, N);
	fn->nname = newname(sym);
	fn->nname->class = PFUNC;
	tfn = nod(OTFUNC, N, N);
	fn->nname->ntype = tfn;

	n = nod(ODCLFIELD, newname(lookup("h")), typenod(ptrto(types[TUINTPTR])));
	tfn->list = list(tfn->list, n);
	nh = n->left;
	n = nod(ODCLFIELD, newname(lookup("s")), typenod(types[TUINTPTR]));
	tfn->list = list(tfn->list, n);
	n = nod(ODCLFIELD, newname(lookup("p")), typenod(ptrto(t)));
	tfn->list = list(tfn->list, n);
	np = n->left;

	funchdr(fn);
	typecheck(&fn->nname->ntype, Etype);

	// genhash is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch(t->etype) {
	default:
		fatal("genhash %T", t);
	case TARRAY:
		if(isslice(t))
			fatal("genhash %T", t);
		// An array of pure memory would be handled by the
		// standard algorithm, so the element type must not be
		// pure memory.
		hashel = hashfor(t->type);
		n = nod(ORANGE, N, nod(OIND, np, N));
		ni = newname(lookup("i"));
		ni->type = types[TINT];
		n->list = list1(ni);
		n->colas = 1;
		colasdefn(n->list, n);
		ni = n->list->n;

		// *h = *h<<3 | *h>>61
		n->nbody = list(n->nbody,
			nod(OAS,
				nod(OIND, nh, N),
				nod(OOR,
					nod(OLSH, nod(OIND, nh, N), nodintconst(3)),
					nod(ORSH, nod(OIND, nh, N), nodintconst(widthptr*8-3)))));

		// *h *= mul
		// Same multipliers as in runtime.memhash.
		if(widthptr == 4)
			mul = 3267000013LL;
		else
			mul = 23344194077549503LL;
		n->nbody = list(n->nbody,
			nod(OAS,
				nod(OIND, nh, N),
				nod(OMUL, nod(OIND, nh, N), nodintconst(mul))));

		// hashel(h, sizeof(p[i]), &p[i])
		call = nod(OCALL, hashel, N);
		call->list = list(call->list, nh);
		call->list = list(call->list, nodintconst(t->type->width));
		nx = nod(OINDEX, np, ni);
		nx->bounded = 1;
		na = nod(OADDR, nx, N);
		na->etype = 1;  // no escape to heap
		call->list = list(call->list, na);
		n->nbody = list(n->nbody, call);

		fn->nbody = list(fn->nbody, n);
		break;

	case TSTRUCT:
		// Walk the struct using memhash for runs of AMEM
		// and calling specific hash functions for the others.
		first = T;
		offend = 0;
		for(t1=t->type;; t1=t1->down) {
			if(t1 != T && algtype1(t1->type, nil) == AMEM && !isblanksym(t1->sym)) {
				offend = t1->width + t1->type->width;
				if(first == T)
					first = t1;
				// If it's a memory field but it's padded, stop here.
				if(ispaddedfield(t1, t->width))
					t1 = t1->down;
				else
					continue;
			}
			// Run memhash for fields up to this one.
			if(first != T) {
				size = offend - first->width; // first->width is offset
				hashel = hashmem(first->type);
				// hashel(h, size, &p.first)
				call = nod(OCALL, hashel, N);
				call->list = list(call->list, nh);
				call->list = list(call->list, nodintconst(size));
				nx = nod(OXDOT, np, newname(first->sym));  // TODO: fields from other packages?
				na = nod(OADDR, nx, N);
				na->etype = 1;  // no escape to heap
				call->list = list(call->list, na);
				fn->nbody = list(fn->nbody, call);

				first = T;
			}
			if(t1 == T)
				break;
			if(isblanksym(t1->sym))
				continue;

			// Run hash for this field.
			hashel = hashfor(t1->type);
			// hashel(h, size, &p.t1)
			call = nod(OCALL, hashel, N);
			call->list = list(call->list, nh);
			call->list = list(call->list, nodintconst(t1->type->width));
			nx = nod(OXDOT, np, newname(t1->sym));  // TODO: fields from other packages?
			na = nod(OADDR, nx, N);
			na->etype = 1;  // no escape to heap
			call->list = list(call->list, na);
			fn->nbody = list(fn->nbody, call);
		}
		// make sure body is not empty.
		fn->nbody = list(fn->nbody, nod(ORETURN, N, N));
		break;
	}

	if(debug['r'])
		dumplist("genhash body", fn->nbody);

	funcbody(fn);
	curfn = fn;
	fn->dupok = 1;
	typecheck(&fn, Etop);
	typechecklist(fn->nbody, Etop);
	curfn = nil;

	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode = safemode;
	safemode = 0;
	funccompile(fn, 0);
	safemode = old_safemode;
}

// Return node for
//	if p.field != q.field { *eq = false; return }
static Node*
eqfield(Node *p, Node *q, Node *field, Node *eq)
{
	Node *nif, *nx, *ny;

	nx = nod(OXDOT, p, field);
	ny = nod(OXDOT, q, field);
	nif = nod(OIF, N, N);
	nif->ntest = nod(ONE, nx, ny);
	nif->nbody = list(nif->nbody, nod(OAS, nod(OIND, eq, N), nodbool(0)));
	nif->nbody = list(nif->nbody, nod(ORETURN, N, N));
	return nif;
}

static Node*
eqmemfunc(vlong size, Type *type)
{
	char buf[30];
	Node *fn;

	switch(size) {
	default:
		fn = syslook("memequal", 1);
		break;
	case 1:
	case 2:
	case 4:
	case 8:
	case 16:
		snprint(buf, sizeof buf, "memequal%d", (int)size*8);
		fn = syslook(buf, 1);
		break;
	}
	argtype(fn, type);
	argtype(fn, type);
	return fn;
}

// Return node for
//	if memequal(size, &p.field, &q.field, eq); !*eq { return }
static Node*
eqmem(Node *p, Node *q, Node *field, vlong size, Node *eq)
{
	Node *nif, *nx, *ny, *call;

	nx = nod(OADDR, nod(OXDOT, p, field), N);
	nx->etype = 1;  // does not escape
	ny = nod(OADDR, nod(OXDOT, q, field), N);
	ny->etype = 1;  // does not escape
	typecheck(&nx, Erv);
	typecheck(&ny, Erv);

	call = nod(OCALL, eqmemfunc(size, nx->type->type), N);
	call->list = list(call->list, eq);
	call->list = list(call->list, nodintconst(size));
	call->list = list(call->list, nx);
	call->list = list(call->list, ny);

	nif = nod(OIF, N, N);
	nif->ninit = list(nif->ninit, call);
	nif->ntest = nod(ONOT, nod(OIND, eq, N), N);
	nif->nbody = list(nif->nbody, nod(ORETURN, N, N));
	return nif;
}

/*
 * Generate a helper function to check equality of two values of type t.
 */
void
geneq(Sym *sym, Type *t)
{
	Node *n, *fn, *np, *neq, *nq, *tfn, *nif, *ni, *nx, *ny, *nrange;
	Type *t1, *first;
	int old_safemode;
	int64 size;
	int64 offend;

	if(debug['r'])
		print("geneq %S %T\n", sym, t);

	lineno = 1;  // less confusing than end of input
	dclcontext = PEXTERN;
	markdcl();

	// func sym(eq *bool, s uintptr, p, q *T)
	fn = nod(ODCLFUNC, N, N);
	fn->nname = newname(sym);
	fn->nname->class = PFUNC;
	tfn = nod(OTFUNC, N, N);
	fn->nname->ntype = tfn;

	n = nod(ODCLFIELD, newname(lookup("eq")), typenod(ptrto(types[TBOOL])));
	tfn->list = list(tfn->list, n);
	neq = n->left;
	n = nod(ODCLFIELD, newname(lookup("s")), typenod(types[TUINTPTR]));
	tfn->list = list(tfn->list, n);
	n = nod(ODCLFIELD, newname(lookup("p")), typenod(ptrto(t)));
	tfn->list = list(tfn->list, n);
	np = n->left;
	n = nod(ODCLFIELD, newname(lookup("q")), typenod(ptrto(t)));
	tfn->list = list(tfn->list, n);
	nq = n->left;

	funchdr(fn);

	// geneq is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch(t->etype) {
	default:
		fatal("geneq %T", t);
	case TARRAY:
		if(isslice(t))
			fatal("geneq %T", t);
		// An array of pure memory would be handled by the
		// standard memequal, so the element type must not be
		// pure memory.  Even if we unrolled the range loop,
		// each iteration would be a function call, so don't bother
		// unrolling.
		nrange = nod(ORANGE, N, nod(OIND, np, N));
		ni = newname(lookup("i"));
		ni->type = types[TINT];
		nrange->list = list1(ni);
		nrange->colas = 1;
		colasdefn(nrange->list, nrange);
		ni = nrange->list->n;
		
		// if p[i] != q[i] { *eq = false; return }
		nx = nod(OINDEX, np, ni);
		nx->bounded = 1;
		ny = nod(OINDEX, nq, ni);
		ny->bounded = 1;

		nif = nod(OIF, N, N);
		nif->ntest = nod(ONE, nx, ny);
		nif->nbody = list(nif->nbody, nod(OAS, nod(OIND, neq, N), nodbool(0)));
		nif->nbody = list(nif->nbody, nod(ORETURN, N, N));
		nrange->nbody = list(nrange->nbody, nif);
		fn->nbody = list(fn->nbody, nrange);

		// *eq = true;
		fn->nbody = list(fn->nbody, nod(OAS, nod(OIND, neq, N), nodbool(1)));
		break;

	case TSTRUCT:
		// Walk the struct using memequal for runs of AMEM
		// and calling specific equality tests for the others.
		// Skip blank-named fields.
		first = T;
		offend = 0;
		for(t1=t->type;; t1=t1->down) {
			if(t1 != T && algtype1(t1->type, nil) == AMEM && !isblanksym(t1->sym)) {
				offend = t1->width + t1->type->width;
				if(first == T)
					first = t1;
				// If it's a memory field but it's padded, stop here.
				if(ispaddedfield(t1, t->width))
					t1 = t1->down;
				else
					continue;
			}
			// Run memequal for fields up to this one.
			// TODO(rsc): All the calls to newname are wrong for
			// cross-package unexported fields.
			if(first != T) {
				if(first->down == t1) {
					fn->nbody = list(fn->nbody, eqfield(np, nq, newname(first->sym), neq));
				} else if(first->down->down == t1) {
					fn->nbody = list(fn->nbody, eqfield(np, nq, newname(first->sym), neq));
					first = first->down;
					if(!isblanksym(first->sym))
						fn->nbody = list(fn->nbody, eqfield(np, nq, newname(first->sym), neq));
				} else {
					// More than two fields: use memequal.
					size = offend - first->width; // first->width is offset
					fn->nbody = list(fn->nbody, eqmem(np, nq, newname(first->sym), size, neq));
				}
				first = T;
			}
			if(t1 == T)
				break;
			if(isblanksym(t1->sym))
				continue;

			// Check this field, which is not just memory.
			fn->nbody = list(fn->nbody, eqfield(np, nq, newname(t1->sym), neq));
		}

		// *eq = true;
		fn->nbody = list(fn->nbody, nod(OAS, nod(OIND, neq, N), nodbool(1)));
		break;
	}

	if(debug['r'])
		dumplist("geneq body", fn->nbody);

	funcbody(fn);
	curfn = fn;
	fn->dupok = 1;
	typecheck(&fn, Etop);
	typechecklist(fn->nbody, Etop);
	curfn = nil;
	
	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode = safemode;
	safemode = 0;
	funccompile(fn, 0);
	safemode = old_safemode;
}

static Type*
ifacelookdot(Sym *s, Type *t, int *followptr, int ignorecase)
{
	int i, c, d;
	Type *m;

	*followptr = 0;

	if(t == T)
		return T;

	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(s, t, d, &m, ignorecase);
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

int
implements(Type *t, Type *iface, Type **m, Type **samename, int *ptr)
{
	Type *t0, *im, *tm, *rcvr, *imtype;
	int followptr;

	t0 = t;
	if(t == T)
		return 0;

	// if this is too slow,
	// could sort these first
	// and then do one loop.

	if(t->etype == TINTER) {
		for(im=iface->type; im; im=im->down) {
			for(tm=t->type; tm; tm=tm->down) {
				if(tm->sym == im->sym) {
					if(eqtype(tm->type, im->type))
						goto found;
					*m = im;
					*samename = tm;
					*ptr = 0;
					return 0;
				}
			}
			*m = im;
			*samename = nil;
			*ptr = 0;
			return 0;
		found:;
		}
		return 1;
	}

	t = methtype(t, 0);
	if(t != T)
		expandmeth(t);
	for(im=iface->type; im; im=im->down) {
		imtype = methodfunc(im->type, 0);
		tm = ifacelookdot(im->sym, t, &followptr, 0);
		if(tm == T || tm->nointerface || !eqtype(methodfunc(tm->type, 0), imtype)) {
			if(tm == T)
				tm = ifacelookdot(im->sym, t, &followptr, 1);
			*m = im;
			*samename = tm;
			*ptr = 0;
			return 0;
		}
		// if pointer receiver in method,
		// the method does not exist for value types.
		rcvr = getthisx(tm->type)->type->type;
		if(isptr[rcvr->etype] && !isptr[t0->etype] && !followptr && !isifacemethod(tm->type)) {
			if(0 && debug['r'])
				yyerror("interface pointer mismatch");

			*m = im;
			*samename = nil;
			*ptr = 1;
			return 0;
		}
	}
	return 1;
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
	if(n->op == OBLOCK && n->ninit == nil) {
		// Flatten list and steal storage.
		// Poison pointer to catch errant uses.
		l = n->list;
		n->list = (NodeList*)1;
		return l;
	}
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

void
listsort(NodeList** l, int(*f)(Node*, Node*))
{
	NodeList *l1, *l2, *le;

	if(*l == nil || (*l)->next == nil)
		return;

	l1 = *l;
	l2 = *l;
	for(;;) {
		l2 = l2->next;
		if(l2 == nil)
			break;
		l2 = l2->next;
		if(l2 == nil)
			break;
		l1 = l1->next;
	}

	l2 = l1->next;
	l1->next = nil;
	l2->end = (*l)->end;
	(*l)->end = l1;

	l1 = *l;
	listsort(&l1, f);
	listsort(&l2, f);

	if((*f)(l1->n, l2->n) < 0) {
		*l = l1;
	} else {
		*l = l2;
		l2 = l1;
		l1 = *l;
	}

	// now l1 == *l; and l1 < l2

	while ((l1 != nil) && (l2 != nil)) {
		while ((l1->next != nil) && (*f)(l1->next->n, l2->n) < 0)
			l1 = l1->next;
		
		// l1 is last one from l1 that is < l2
		le = l1->next;		// le is the rest of l1, first one that is >= l2
		if(le != nil)
			le->end = (*l)->end;

		(*l)->end = l1;		// cut *l at l1
		*l = concat(*l, l2);	// glue l2 to *l's tail

		l1 = l2;		// l1 is the first element of *l that is < the new l2
		l2 = le;		// ... because l2 now is the old tail of l1
	}

	*l = concat(*l, l2);		// any remainder 
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
	vlong n;

	n = 0;
	for(; l; l=l->next)
		n++;
	if((int)n != n) { // Overflow.
		yyerror("too many elements in list");
	}
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
	if(n->type != T)
		return typenamesym(n->type);
	return S;
}

/*
 * Convert raw string to the prefix that will be used in the symbol
 * table.  All control characters, space, '%' and '"', as well as
 * non-7-bit clean bytes turn into %xx.  The period needs escaping
 * only in the last segment of the path, and it makes for happier
 * users if we escape that as little as possible.
 *
 * If you edit this, edit ../ld/lib.c:/^pathtoprefix copy too.
 */
static char*
pathtoprefix(char *s)
{
	static char hex[] = "0123456789abcdef";
	char *p, *r, *w, *l;
	int n;

	// find first character past the last slash, if any.
	l = s;
	for(r=s; *r; r++)
		if(*r == '/')
			l = r+1;

	// check for chars that need escaping
	n = 0;
	for(r=s; *r; r++)
		if(*r <= ' ' || (*r == '.' && r >= l) || *r == '%' || *r == '"' || *r >= 0x7f)
			n++;

	// quick exit
	if(n == 0)
		return s;

	// escape
	p = mal((r-s)+1+2*n);
	for(r=s, w=p; *r; r++) {
		if(*r <= ' ' || (*r == '.' && r >= l) || *r == '%' || *r == '"' || *r >= 0x7f) {
			*w++ = '%';
			*w++ = hex[(*r>>4)&0xF];
			*w++ = hex[*r&0xF];
		} else
			*w++ = *r;
	}
	*w = '\0';
	return p;
}

Pkg*
mkpkg(Strlit *path)
{
	Pkg *p;
	int h;

	h = stringhash(path->s) & (nelem(phash)-1);
	for(p=phash[h]; p; p=p->link)
		if(p->path->len == path->len && memcmp(path->s, p->path->s, path->len) == 0)
			return p;

	p = mal(sizeof *p);
	p->path = path;
	p->prefix = pathtoprefix(path->s);
	p->link = phash[h];
	phash[h] = p;
	return p;
}

Strlit*
strlit(char *s)
{
	Strlit *t;
	
	t = mal(sizeof *t + strlen(s));
	strcpy(t->s, s);
	t->len = strlen(s);
	return t;
}

void
addinit(Node **np, NodeList *init)
{
	Node *n;
	
	if(init == nil)
		return;

	n = *np;
	switch(n->op) {
	case ONAME:
	case OLITERAL:
		// There may be multiple refs to this node;
		// introduce OCONVNOP to hold init list.
		n = nod(OCONVNOP, n, N);
		n->type = n->left->type;
		n->typecheck = 1;
		*np = n;
		break;
	}
	n->ninit = concat(init, n->ninit);
	n->ullman = UINF;
}

static char* reservedimports[] = {
	"go",
	"type",
};

int
isbadimport(Strlit *path)
{
	int i;
	char *s;
	Rune r;

	if(strlen(path->s) != path->len) {
		yyerror("import path contains NUL");
		return 1;
	}
	
	for(i=0; i<nelem(reservedimports); i++) {
		if(strcmp(path->s, reservedimports[i]) == 0) {
			yyerror("import path \"%s\" is reserved and cannot be used", path->s);
			return 1;
		}
	}

	s = path->s;
	while(*s) {
		s += chartorune(&r, s);
		if(r == Runeerror) {
			yyerror("import path contains invalid UTF-8 sequence: \"%Z\"", path);
			return 1;
		}
		if(r < 0x20 || r == 0x7f) {
			yyerror("import path contains control character: \"%Z\"", path);
			return 1;
		}
		if(r == '\\') {
			yyerror("import path contains backslash; use slash: \"%Z\"", path);
			return 1;
		}
		if(isspacerune(r)) {
			yyerror("import path contains space character: \"%Z\"", path);
			return 1;
		}
		if(utfrune("!\"#$%&'()*,:;<=>?[]^`{|}", r)) {
			yyerror("import path contains invalid character '%C': \"%Z\"", r, path);
			return 1;
		}
	}
	return 0;
}

void
checknotnil(Node *x, NodeList **init)
{
	Node *n;
	
	if(isinter(x->type)) {
		x = nod(OITAB, x, N);
		typecheck(&x, Erv);
	}
	n = nod(OCHECKNOTNIL, x, N);
	n->typecheck = 1;
	*init = list(*init, n);
}
