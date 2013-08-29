// Inferno utils/cc/dpchk.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/dpchk.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include	<u.h>
#include	"cc.h"
#include	"y.tab.h"

enum
{
	Fnone	= 0,
	Fl,
	Fvl,
	Fignor,
	Fstar,
	Fadj,

	Fverb	= 10,
};

typedef	struct	Tprot	Tprot;
struct	Tprot
{
	Type*	type;
	Bits	flag;
	Tprot*	link;
};

typedef	struct	Tname	Tname;
struct	Tname
{
	char*	name;
	int	param;
	int	count;
	Tname*	link;
	Tprot*	prot;
};

static	Type*	indchar;
static	uchar	flagbits[512];
static	char*	lastfmt;
static	int	lastadj;
static	int	lastverb;
static	int	nstar;
static	Tprot*	tprot;
static	Tname*	tname;

void
argflag(int c, int v)
{

	switch(v) {
	case Fignor:
	case Fstar:
	case Fl:
	case Fvl:
		flagbits[c] = v;
		break;
	case Fverb:
		flagbits[c] = lastverb;
/*print("flag-v %c %d\n", c, lastadj);*/
		lastverb++;
		break;
	case Fadj:
		flagbits[c] = lastadj;
/*print("flag-l %c %d\n", c, lastadj);*/
		lastadj++;
		break;
	}
}

Bits
getflag(char *s)
{
	Bits flag;
	int f;
	Fmt fmt;
	Rune c;

	flag = zbits;
	nstar = 0;
	fmtstrinit(&fmt);
	for(;;) {
		s += chartorune(&c, s);
		if(c == 0 || c >= nelem(flagbits))
			break;
		fmtrune(&fmt, c);
		f = flagbits[c];
		switch(f) {
		case Fnone:
			argflag(c, Fverb);
			f = flagbits[c];
			break;
		case Fstar:
			nstar++;
		case Fignor:
			continue;
		case Fl:
			if(bset(flag, Fl))
				flag = bor(flag, blsh(Fvl));
		}
		flag = bor(flag, blsh(f));
		if(f >= Fverb)
			break;
	}
	free(lastfmt);
	lastfmt = fmtstrflush(&fmt);
	return flag;
}

static void
newprot(Sym *m, Type *t, char *s, Tprot **prot)
{
	Bits flag;
	Tprot *l;

	if(t == T) {
		warn(Z, "%s: newprot: type not defined", m->name);
		return;
	}
	flag = getflag(s);
	for(l=*prot; l; l=l->link)
		if(beq(flag, l->flag) && sametype(t, l->type))
			return;
	l = alloc(sizeof(*l));
	l->type = t;
	l->flag = flag;
	l->link = *prot;
	*prot = l;
}

static Tname*
newname(char *s, int p, int count)
{
	Tname *l;

	for(l=tname; l; l=l->link)
		if(strcmp(l->name, s) == 0) {
			if(p >= 0 && l->param != p)
				yyerror("vargck %s already defined\n", s);
			return l;
		}
	if(p < 0)
		return nil;

	l = alloc(sizeof(*l));
	l->name = s;
	l->param = p;
	l->link = tname;
	l->count = count;
	tname = l;
	return l;
}

void
arginit(void)
{
	int i;

/* debug['F'] = 1;*/
/* debug['w'] = 1;*/

	lastadj = Fadj;
	lastverb = Fverb;
	indchar = typ(TIND, types[TCHAR]);

	memset(flagbits, Fnone, sizeof(flagbits));

	for(i='0'; i<='9'; i++)
		argflag(i, Fignor);
	argflag('.', Fignor);
	argflag('#', Fignor);
	argflag('u', Fignor);
	argflag('h', Fignor);
	argflag('+', Fignor);
	argflag('-', Fignor);

	argflag('*', Fstar);
	argflag('l', Fl);

	argflag('o', Fverb);
	flagbits['x'] = flagbits['o'];
	flagbits['X'] = flagbits['o'];
}

static char*
getquoted(void)
{
	int c;
	Rune r;
	Fmt fmt;

	c = getnsc();
	if(c != '"')
		return nil;
	fmtstrinit(&fmt);
	for(;;) {
		r = getr();
		if(r == '\n') {
			free(fmtstrflush(&fmt));
			return nil;
		}
		if(r == '"')
			break;
		fmtrune(&fmt, r);
	}
	free(lastfmt);
	lastfmt = fmtstrflush(&fmt);
	return strdup(lastfmt);
}

void
pragvararg(void)
{
	Sym *s;
	int n, c;
	char *t;
	Type *ty;
	Tname *l;

	if(!debug['F'])
		goto out;
	s = getsym();
	if(s && strcmp(s->name, "argpos") == 0)
		goto ckpos;
	if(s && strcmp(s->name, "type") == 0)
		goto cktype;
	if(s && strcmp(s->name, "flag") == 0)
		goto ckflag;
	if(s && strcmp(s->name, "countpos") == 0)
		goto ckcount;
	yyerror("syntax in #pragma varargck");
	goto out;

ckpos:
/*#pragma	varargck	argpos	warn	2*/
	s = getsym();
	if(s == S)
		goto bad;
	n = getnsn();
	if(n < 0)
		goto bad;
	newname(s->name, n, 0);
	goto out;

ckcount:
/*#pragma	varargck	countpos	name 2*/
	s = getsym();
	if(s == S)
		goto bad;
	n = getnsn();
	if(n < 0)
		goto bad;
	newname(s->name, 0, n);
	goto out;

ckflag:
/*#pragma	varargck	flag	'c'*/
	c = getnsc();
	if(c != '\'')
		goto bad;
	c = getr();
	if(c == '\\')
		c = getr();
	else if(c == '\'')
		goto bad;
	if(c == '\n')
		goto bad;
	if(getc() != '\'')
		goto bad;
	argflag(c, Fignor);
	goto out;

cktype:
	c = getnsc();
	unget(c);
	if(c != '"') {
/*#pragma	varargck	type	name	int*/
		s = getsym();
		if(s == S)
			goto bad;
		l = newname(s->name, -1, -1);
		s = getsym();
		if(s == S)
			goto bad;
		ty = s->type;
		while((c = getnsc()) == '*')
			ty = typ(TIND, ty);
		unget(c);
		newprot(s, ty, "a", &l->prot);
		goto out;
	}

/*#pragma	varargck	type	O	int*/
	t = getquoted();
	if(t == nil)
		goto bad;
	s = getsym();
	if(s == S)
		goto bad;
	ty = s->type;
	while((c = getnsc()) == '*')
		ty = typ(TIND, ty);
	unget(c);
	newprot(s, ty, t, &tprot);
	goto out;

bad:
	yyerror("syntax in #pragma varargck");

out:
	while(getnsc() != '\n')
		;
}

Node*
nextarg(Node *n, Node **a)
{
	if(n == Z) {
		*a = Z;
		return Z;
	}
	if(n->op == OLIST) {
		*a = n->left;
		return n->right;
	}
	*a = n;
	return Z;
}

void
checkargs(Node *nn, char *s, int pos)
{
	Node *a, *n;
	Bits flag;
	Tprot *l;

	if(!debug['F'])
		return;
	n = nn;
	for(;;) {
		s = strchr(s, '%');
		if(s == 0) {
			nextarg(n, &a);
			if(a != Z)
				warn(nn, "more arguments than format %T",
					a->type);
			return;
		}
		s++;
		flag = getflag(s);
		while(nstar > 0) {
			n = nextarg(n, &a);
			pos++;
			nstar--;
			if(a == Z) {
				warn(nn, "more format than arguments %s",
					lastfmt);
				return;
			}
			if(a->type == T)
				continue;
			if(!sametype(types[TINT], a->type) &&
			   !sametype(types[TUINT], a->type))
				warn(nn, "format mismatch '*' in %s %T, arg %d",
					lastfmt, a->type, pos);
		}
		for(l=tprot; l; l=l->link)
			if(sametype(types[TVOID], l->type)) {
				if(beq(flag, l->flag)) {
					s++;
					goto loop;
				}
			}

		n = nextarg(n, &a);
		pos++;
		if(a == Z) {
			warn(nn, "more format than arguments %s",
				lastfmt);
			return;
		}
		if(a->type == 0)
			continue;
		for(l=tprot; l; l=l->link)
			if(sametype(a->type, l->type)) {
/*print("checking %T/%ux %T/%ux\n", a->type, flag.b[0], l->type, l->flag.b[0]);*/
				if(beq(flag, l->flag))
					goto loop;
			}
		warn(nn, "format mismatch %s %T, arg %d", lastfmt, a->type, pos);
	loop:;
	}
}

void
dpcheck(Node *n)
{
	char *s;
	Node *a, *b;
	Tname *l;
	Tprot *tl;
	int i, j;

	if(n == Z)
		return;
	b = n->left;
	if(b == Z || b->op != ONAME)
		return;
	s = b->sym->name;
	for(l=tname; l; l=l->link)
		if(strcmp(s, l->name) == 0)
			break;
	if(l == 0)
		return;

	if(l->count > 0) {
		// fetch count, then check remaining length
		i = l->count;
		a = nil;
		b = n->right;
		while(i > 0) {
			b = nextarg(b, &a);
			i--;
		}
		if(a == Z) {
			diag(n, "can't find count arg");
			return;
		}
		if(a->op != OCONST || !typechl[a->type->etype]) {
			diag(n, "count is invalid constant");
			return;
		}
		j = a->vconst;
		i = 0;
		while(b != Z) {
			b = nextarg(b, &a);
			i++;
		}
		if(i != j)
			diag(n, "found %d argument%s after count %d", i, i == 1 ? "" : "s", j);
	}

	if(l->prot != nil) {
		// check that all arguments after param or count
		// are listed in type list.
		i = l->count;
		if(i == 0)
			i = l->param;
		if(i == 0)
			return;
		a = nil;
		b = n->right;
		while(i > 0) {
			b = nextarg(b, &a);
			i--;
		}
		if(a == Z) {
			diag(n, "can't find count/param arg");
			return;
		}
		while(b != Z) {
			b = nextarg(b, &a);
			for(tl=l->prot; tl; tl=tl->link)
				if(sametype(a->type, tl->type))
					break;
			if(tl == nil)
				diag(a, "invalid type %T in call to %s", a->type, s);
		}
	}

	if(l->param <= 0)
		return;
	i = l->param;
	a = nil;
	b = n->right;
	while(i > 0) {
		b = nextarg(b, &a);
		i--;
	}
	if(a == Z) {
		diag(n, "can't find format arg");
		return;
	}
	if(!sametype(indchar, a->type)) {
		diag(n, "format arg type %T", a->type);
		return;
	}
	if(a->op != OADDR || a->left->op != ONAME || a->left->sym != symstring) {
/*		warn(n, "format arg not constant string");*/
		return;
	}
	s = a->left->cstring;
	checkargs(b, s, l->param);
}

void
pragpack(void)
{
	Sym *s;

	packflg = 0;
	s = getsym();
	if(s) {
		packflg = atoi(s->name+1);
		if(strcmp(s->name, "on") == 0 ||
		   strcmp(s->name, "yes") == 0)
			packflg = 1;
	}
	while(getnsc() != '\n')
		;
	if(debug['f'])
		if(packflg)
			print("%4d: pack %d\n", lineno, packflg);
		else
			print("%4d: pack off\n", lineno);
}

void
pragfpround(void)
{
	Sym *s;

	fproundflg = 0;
	s = getsym();
	if(s) {
		fproundflg = atoi(s->name+1);
		if(strcmp(s->name, "on") == 0 ||
		   strcmp(s->name, "yes") == 0)
			fproundflg = 1;
	}
	while(getnsc() != '\n')
		;
	if(debug['f'])
		if(fproundflg)
			print("%4d: fproundflg %d\n", lineno, fproundflg);
		else
			print("%4d: fproundflg off\n", lineno);
}

void
pragtextflag(void)
{
	Sym *s;

	s = getsym();
	if(s == S) {
		textflag = getnsn();
	} else {
		if(s->macro) {
			macexpand(s, symb);
		}
		if(symb[0] < '0' || symb[0] > '9')
			yyerror("pragma textflag not an integer");
		textflag = atoi(symb);
	}
	while(getnsc() != '\n')
		;
	if(debug['f'])
		print("%4d: textflag %d\n", lineno, textflag);
}

void
pragdataflag(void)
{
	Sym *s;

	s = getsym();
	if(s == S) {
		dataflag = getnsn();
	} else {
		if(s->macro) {
			macexpand(s, symb);
		}
		if(symb[0] < '0' || symb[0] > '9')
			yyerror("pragma dataflag not an integer");
		dataflag = atoi(symb);
	}
	while(getnsc() != '\n')
		;
	if(debug['f'])
		print("%4d: dataflag %d\n", lineno, dataflag);
}

void
pragincomplete(void)
{
	Sym *s;
	Type *t;
	int istag, w, et;

	istag = 0;
	s = getsym();
	if(s == nil)
		goto out;
	et = 0;
	w = s->lexical;
	if(w == LSTRUCT)
		et = TSTRUCT;
	else if(w == LUNION)
		et = TUNION;
	if(et != 0){
		s = getsym();
		if(s == nil){
			yyerror("missing struct/union tag in pragma incomplete");
			goto out;
		}
		if(s->lexical != LNAME && s->lexical != LTYPE){
			yyerror("invalid struct/union tag: %s", s->name);
			goto out;
		}
		dotag(s, et, 0);
		istag = 1;
	}else if(strcmp(s->name, "_off_") == 0){
		debug['T'] = 0;
		goto out;
	}else if(strcmp(s->name, "_on_") == 0){
		debug['T'] = 1;
		goto out;
	}
	t = s->type;
	if(istag)
		t = s->suetag;
	if(t == T)
		yyerror("unknown type %s in pragma incomplete", s->name);
	else if(!typesu[t->etype])
		yyerror("not struct/union type in pragma incomplete: %s", s->name);
	else
		t->garb |= GINCOMPLETE;
out:
	while(getnsc() != '\n')
		;
	if(debug['f'])
		print("%s incomplete\n", s->name);
}

Sym*
getimpsym(void)
{
	int c;
	char *cp;

	c = getnsc();
	if(isspace(c) || c == '"') {
		unget(c);
		return S;
	}
	for(cp = symb;;) {
		if(cp <= symb+NSYMB-4)
			*cp++ = c;
		c = getc();
		if(c > 0 && !isspace(c) && c != '"')
			continue;
		unget(c);
		break;
	}
	*cp = 0;
	if(cp > symb+NSYMB-4)
		yyerror("symbol too large: %s", symb);
	return lookup();
}

static int
more(void)
{
	int c;
	
	do
		c = getnsc();
	while(c == ' ' || c == '\t');
	unget(c);
	return c != '\n';
}

void
pragcgo(char *verb)
{
	Sym *local, *remote;
	char *p;

	if(strcmp(verb, "cgo_dynamic_linker") == 0 || strcmp(verb, "dynlinker") == 0) {
		p = getquoted();
		if(p == nil)
			goto err1;
		fmtprint(&pragcgobuf, "cgo_dynamic_linker %q\n", p);
		goto out;
	
	err1:
		yyerror("usage: #pragma cgo_dynamic_linker \"path\"");
		goto out;
	}	
	
	if(strcmp(verb, "dynexport") == 0)
		verb = "cgo_export_dynamic";
	if(strcmp(verb, "cgo_export_static") == 0 || strcmp(verb, "cgo_export_dynamic") == 0) {
		local = getimpsym();
		if(local == nil)
			goto err2;
		if(!more()) {
			fmtprint(&pragcgobuf, "%s %q\n", verb, local->name);
			goto out;
		}
		remote = getimpsym();
		if(remote == nil)
			goto err2;
		fmtprint(&pragcgobuf, "%s %q %q\n", verb, local->name, remote->name);
		goto out;
	
	err2:
		yyerror("usage: #pragma %s local [remote]", verb);
		goto out;
	}
	
	if(strcmp(verb, "cgo_import_dynamic") == 0 || strcmp(verb, "dynimport") == 0) {
		local = getimpsym();
		if(local == nil)
			goto err3;
		if(!more()) {
			fmtprint(&pragcgobuf, "cgo_import_dynamic %q\n", local->name);
			goto out;
		}
		remote = getimpsym();
		if(remote == nil)
			goto err3;
		if(!more()) {
			fmtprint(&pragcgobuf, "cgo_import_dynamic %q %q\n", local->name, remote->name);
			goto out;
		}
		p = getquoted();
		if(p == nil)	
			goto err3;
		fmtprint(&pragcgobuf, "cgo_import_dynamic %q %q %q\n", local->name, remote->name, p);
		goto out;
	
	err3:
		yyerror("usage: #pragma cgo_import_dynamic local [remote [\"library\"]]");
		goto out;
	}
	
	if(strcmp(verb, "cgo_import_static") == 0) {
		local = getimpsym();
		if(local == nil)
			goto err4;
		fmtprint(&pragcgobuf, "cgo_import_static %q\n", local->name);
		goto out;

	err4:
		yyerror("usage: #pragma cgo_import_static local [remote]");
		goto out;
	}
	
	if(strcmp(verb, "cgo_ldflag") == 0) {
		p = getquoted();
		if(p == nil)
			goto err5;
		fmtprint(&pragcgobuf, "cgo_ldflag %q\n", p);
		goto out;

	err5:
		yyerror("usage: #pragma cgo_ldflag \"arg\"");
		goto out;
	}
	
out:
	while(getnsc() != '\n')
		;
}
