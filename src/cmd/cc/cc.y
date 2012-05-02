// Inferno utils/cc/cc.y
// http://code.google.com/p/inferno-os/source/browse/utils/cc/cc.y
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

%{
#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and cc.h re-#defines getc */
#include "cc.h"
%}
%union	{
	Node*	node;
	Sym*	sym;
	Type*	type;
	struct
	{
		Type*	t;
		uchar	c;
	} tycl;
	struct
	{
		Type*	t1;
		Type*	t2;
		Type*	t3;
		uchar	c;
	} tyty;
	struct
	{
		char*	s;
		int32	l;
	} sval;
	int32	lval;
	double	dval;
	vlong	vval;
}
%type	<sym>	ltag
%type	<lval>	gctname gcname cname gname tname
%type	<lval>	gctnlist gcnlist zgnlist
%type	<type>	tlist sbody complex
%type	<tycl>	types
%type	<node>	zarglist arglist zcexpr
%type	<node>	name block stmnt cexpr expr xuexpr pexpr
%type	<node>	zelist elist adecl slist uexpr string lstring
%type	<node>	xdecor xdecor2 labels label ulstmnt
%type	<node>	adlist edecor tag qual qlist
%type	<node>	abdecor abdecor1 abdecor2 abdecor3
%type	<node>	zexpr lexpr init ilist forexpr

%left	';'
%left	','
%right	'=' LPE LME LMLE LDVE LMDE LRSHE LLSHE LANDE LXORE LORE
%right	'?' ':'
%left	LOROR
%left	LANDAND
%left	'|'
%left	'^'
%left	'&'
%left	LEQ LNE
%left	'<' '>' LLE LGE
%left	LLSH LRSH
%left	'+' '-'
%left	'*' '/' '%'
%right	LMM LPP LMG '.' '[' '('

%token	<sym>	LNAME LTYPE
%token	<dval>	LFCONST LDCONST
%token	<vval>	LCONST LLCONST LUCONST LULCONST LVLCONST LUVLCONST
%token	<sval>	LSTRING LLSTRING
%token		LAUTO LBREAK LCASE LCHAR LCONTINUE LDEFAULT LDO
%token		LDOUBLE LELSE LEXTERN LFLOAT LFOR LGOTO
%token	LIF LINT LLONG LPREFETCH LREGISTER LRETURN LSHORT LSIZEOF LUSED
%token	LSTATIC LSTRUCT LSWITCH LTYPEDEF LTYPESTR LUNION LUNSIGNED
%token	LWHILE LVOID LENUM LSIGNED LCONSTNT LVOLATILE LSET LSIGNOF
%token	LRESTRICT LINLINE
%%
prog:
|	prog xdecl

/*
 * external declarator
 */
xdecl:
	zctlist ';'
	{
		dodecl(xdecl, lastclass, lasttype, Z);
	}
|	zctlist xdlist ';'
|	zctlist xdecor
	{
		lastdcl = T;
		firstarg = S;
		dodecl(xdecl, lastclass, lasttype, $2);
		if(lastdcl == T || lastdcl->etype != TFUNC) {
			diag($2, "not a function");
			lastdcl = types[TFUNC];
		}
		thisfn = lastdcl;
		markdcl();
		firstdcl = dclstack;
		argmark($2, 0);
	}
	pdecl
	{
		argmark($2, 1);
	}
	block
	{
		Node *n;

		n = revertdcl();
		if(n)
			$6 = new(OLIST, n, $6);
		if(!debug['a'] && !debug['Z'])
			codgen($6, $2);
	}

xdlist:
	xdecor
	{
		dodecl(xdecl, lastclass, lasttype, $1);
	}
|	xdecor
	{
		$1 = dodecl(xdecl, lastclass, lasttype, $1);
	}
	'=' init
	{
		doinit($1->sym, $1->type, 0L, $4);
	}
|	xdlist ',' xdlist

xdecor:
	xdecor2
|	'*' zgnlist xdecor
	{
		$$ = new(OIND, $3, Z);
		$$->garb = simpleg($2);
	}

xdecor2:
	tag
|	'(' xdecor ')'
	{
		$$ = $2;
	}
|	xdecor2 '(' zarglist ')'
	{
		$$ = new(OFUNC, $1, $3);
	}
|	xdecor2 '[' zexpr ']'
	{
		$$ = new(OARRAY, $1, $3);
	}

/*
 * automatic declarator
 */
adecl:
	ctlist ';'
	{
		$$ = dodecl(adecl, lastclass, lasttype, Z);
	}
|	ctlist adlist ';'
	{
		$$ = $2;
	}

adlist:
	xdecor
	{
		dodecl(adecl, lastclass, lasttype, $1);
		$$ = Z;
	}
|	xdecor
	{
		$1 = dodecl(adecl, lastclass, lasttype, $1);
	}
	'=' init
	{
		int32 w;

		w = $1->sym->type->width;
		$$ = doinit($1->sym, $1->type, 0L, $4);
		$$ = contig($1->sym, $$, w);
	}
|	adlist ',' adlist
	{
		$$ = $1;
		if($3 != Z) {
			$$ = $3;
			if($1 != Z)
				$$ = new(OLIST, $1, $3);
		}
	}

/*
 * parameter declarator
 */
pdecl:
|	pdecl ctlist pdlist ';'

pdlist:
	xdecor
	{
		dodecl(pdecl, lastclass, lasttype, $1);
	}
|	pdlist ',' pdlist

/*
 * structure element declarator
 */
edecl:
	tlist
	{
		lasttype = $1;
	}
	zedlist ';'
|	edecl tlist
	{
		lasttype = $2;
	}
	zedlist ';'

zedlist:					/* extension */
	{
		lastfield = 0;
		edecl(CXXX, lasttype, S);
	}
|	edlist

edlist:
	edecor
	{
		dodecl(edecl, CXXX, lasttype, $1);
	}
|	edlist ',' edlist

edecor:
	xdecor
	{
		lastbit = 0;
		firstbit = 1;
	}
|	tag ':' lexpr
	{
		$$ = new(OBIT, $1, $3);
	}
|	':' lexpr
	{
		$$ = new(OBIT, Z, $2);
	}

/*
 * abstract declarator
 */
abdecor:
	{
		$$ = (Z);
	}
|	abdecor1

abdecor1:
	'*' zgnlist
	{
		$$ = new(OIND, (Z), Z);
		$$->garb = simpleg($2);
	}
|	'*' zgnlist abdecor1
	{
		$$ = new(OIND, $3, Z);
		$$->garb = simpleg($2);
	}
|	abdecor2

abdecor2:
	abdecor3
|	abdecor2 '(' zarglist ')'
	{
		$$ = new(OFUNC, $1, $3);
	}
|	abdecor2 '[' zexpr ']'
	{
		$$ = new(OARRAY, $1, $3);
	}

abdecor3:
	'(' ')'
	{
		$$ = new(OFUNC, (Z), Z);
	}
|	'[' zexpr ']'
	{
		$$ = new(OARRAY, (Z), $2);
	}
|	'(' abdecor1 ')'
	{
		$$ = $2;
	}

init:
	expr
|	'{' ilist '}'
	{
		$$ = new(OINIT, invert($2), Z);
	}

qual:
	'[' lexpr ']'
	{
		$$ = new(OARRAY, $2, Z);
	}
|	'.' ltag
	{
		$$ = new(OELEM, Z, Z);
		$$->sym = $2;
	}
|	qual '='

qlist:
	init ','
|	qlist init ','
	{
		$$ = new(OLIST, $1, $2);
	}
|	qual
|	qlist qual
	{
		$$ = new(OLIST, $1, $2);
	}

ilist:
	qlist
|	init
|	qlist init
	{
		$$ = new(OLIST, $1, $2);
	}

zarglist:
	{
		$$ = Z;
	}
|	arglist
	{
		$$ = invert($1);
	}


arglist:
	name
|	tlist abdecor
	{
		$$ = new(OPROTO, $2, Z);
		$$->type = $1;
	}
|	tlist xdecor
	{
		$$ = new(OPROTO, $2, Z);
		$$->type = $1;
	}
|	'.' '.' '.'
	{
		$$ = new(ODOTDOT, Z, Z);
	}
|	arglist ',' arglist
	{
		$$ = new(OLIST, $1, $3);
	}

block:
	'{' slist '}'
	{
		$$ = invert($2);
	//	if($2 != Z)
	//		$$ = new(OLIST, $2, $$);
		if($$ == Z)
			$$ = new(OLIST, Z, Z);
	}

slist:
	{
		$$ = Z;
	}
|	slist adecl
	{
		$$ = new(OLIST, $1, $2);
	}
|	slist stmnt
	{
		$$ = new(OLIST, $1, $2);
	}

labels:
	label
|	labels label
	{
		$$ = new(OLIST, $1, $2);
	}

label:
	LCASE expr ':'
	{
		$$ = new(OCASE, $2, Z);
	}
|	LDEFAULT ':'
	{
		$$ = new(OCASE, Z, Z);
	}
|	LNAME ':'
	{
		$$ = new(OLABEL, dcllabel($1, 1), Z);
	}

stmnt:
	error ';'
	{
		$$ = Z;
	}
|	ulstmnt
|	labels ulstmnt
	{
		$$ = new(OLIST, $1, $2);
	}

forexpr:
	zcexpr
|	ctlist adlist
	{
		$$ = $2;
	}

ulstmnt:
	zcexpr ';'
|	{
		markdcl();
	}
	block
	{
		$$ = revertdcl();
		if($$)
			$$ = new(OLIST, $$, $2);
		else
			$$ = $2;
	}
|	LIF '(' cexpr ')' stmnt
	{
		$$ = new(OIF, $3, new(OLIST, $5, Z));
		if($5 == Z)
			warn($3, "empty if body");
	}
|	LIF '(' cexpr ')' stmnt LELSE stmnt
	{
		$$ = new(OIF, $3, new(OLIST, $5, $7));
		if($5 == Z)
			warn($3, "empty if body");
		if($7 == Z)
			warn($3, "empty else body");
	}
|	{ markdcl(); } LFOR '(' forexpr ';' zcexpr ';' zcexpr ')' stmnt
	{
		$$ = revertdcl();
		if($$){
			if($4)
				$4 = new(OLIST, $$, $4);
			else
				$4 = $$;
		}
		$$ = new(OFOR, new(OLIST, $6, new(OLIST, $4, $8)), $10);
	}
|	LWHILE '(' cexpr ')' stmnt
	{
		$$ = new(OWHILE, $3, $5);
	}
|	LDO stmnt LWHILE '(' cexpr ')' ';'
	{
		$$ = new(ODWHILE, $5, $2);
	}
|	LRETURN zcexpr ';'
	{
		$$ = new(ORETURN, $2, Z);
		$$->type = thisfn->link;
	}
|	LSWITCH '(' cexpr ')' stmnt
	{
		$$ = new(OCONST, Z, Z);
		$$->vconst = 0;
		$$->type = types[TINT];
		$3 = new(OSUB, $$, $3);

		$$ = new(OCONST, Z, Z);
		$$->vconst = 0;
		$$->type = types[TINT];
		$3 = new(OSUB, $$, $3);

		$$ = new(OSWITCH, $3, $5);
	}
|	LBREAK ';'
	{
		$$ = new(OBREAK, Z, Z);
	}
|	LCONTINUE ';'
	{
		$$ = new(OCONTINUE, Z, Z);
	}
|	LGOTO ltag ';'
	{
		$$ = new(OGOTO, dcllabel($2, 0), Z);
	}
|	LUSED '(' zelist ')' ';'
	{
		$$ = new(OUSED, $3, Z);
	}
|	LPREFETCH '(' zelist ')' ';'
	{
		$$ = new(OPREFETCH, $3, Z);
	}
|	LSET '(' zelist ')' ';'
	{
		$$ = new(OSET, $3, Z);
	}

zcexpr:
	{
		$$ = Z;
	}
|	cexpr

zexpr:
	{
		$$ = Z;
	}
|	lexpr

lexpr:
	expr
	{
		$$ = new(OCAST, $1, Z);
		$$->type = types[TLONG];
	}

cexpr:
	expr
|	cexpr ',' cexpr
	{
		$$ = new(OCOMMA, $1, $3);
	}

expr:
	xuexpr
|	expr '*' expr
	{
		$$ = new(OMUL, $1, $3);
	}
|	expr '/' expr
	{
		$$ = new(ODIV, $1, $3);
	}
|	expr '%' expr
	{
		$$ = new(OMOD, $1, $3);
	}
|	expr '+' expr
	{
		$$ = new(OADD, $1, $3);
	}
|	expr '-' expr
	{
		$$ = new(OSUB, $1, $3);
	}
|	expr LRSH expr
	{
		$$ = new(OASHR, $1, $3);
	}
|	expr LLSH expr
	{
		$$ = new(OASHL, $1, $3);
	}
|	expr '<' expr
	{
		$$ = new(OLT, $1, $3);
	}
|	expr '>' expr
	{
		$$ = new(OGT, $1, $3);
	}
|	expr LLE expr
	{
		$$ = new(OLE, $1, $3);
	}
|	expr LGE expr
	{
		$$ = new(OGE, $1, $3);
	}
|	expr LEQ expr
	{
		$$ = new(OEQ, $1, $3);
	}
|	expr LNE expr
	{
		$$ = new(ONE, $1, $3);
	}
|	expr '&' expr
	{
		$$ = new(OAND, $1, $3);
	}
|	expr '^' expr
	{
		$$ = new(OXOR, $1, $3);
	}
|	expr '|' expr
	{
		$$ = new(OOR, $1, $3);
	}
|	expr LANDAND expr
	{
		$$ = new(OANDAND, $1, $3);
	}
|	expr LOROR expr
	{
		$$ = new(OOROR, $1, $3);
	}
|	expr '?' cexpr ':' expr
	{
		$$ = new(OCOND, $1, new(OLIST, $3, $5));
	}
|	expr '=' expr
	{
		$$ = new(OAS, $1, $3);
	}
|	expr LPE expr
	{
		$$ = new(OASADD, $1, $3);
	}
|	expr LME expr
	{
		$$ = new(OASSUB, $1, $3);
	}
|	expr LMLE expr
	{
		$$ = new(OASMUL, $1, $3);
	}
|	expr LDVE expr
	{
		$$ = new(OASDIV, $1, $3);
	}
|	expr LMDE expr
	{
		$$ = new(OASMOD, $1, $3);
	}
|	expr LLSHE expr
	{
		$$ = new(OASASHL, $1, $3);
	}
|	expr LRSHE expr
	{
		$$ = new(OASASHR, $1, $3);
	}
|	expr LANDE expr
	{
		$$ = new(OASAND, $1, $3);
	}
|	expr LXORE expr
	{
		$$ = new(OASXOR, $1, $3);
	}
|	expr LORE expr
	{
		$$ = new(OASOR, $1, $3);
	}

xuexpr:
	uexpr
|	'(' tlist abdecor ')' xuexpr
	{
		$$ = new(OCAST, $5, Z);
		dodecl(NODECL, CXXX, $2, $3);
		$$->type = lastdcl;
		$$->xcast = 1;
	}
|	'(' tlist abdecor ')' '{' ilist '}'	/* extension */
	{
		$$ = new(OSTRUCT, $6, Z);
		dodecl(NODECL, CXXX, $2, $3);
		$$->type = lastdcl;
	}

uexpr:
	pexpr
|	'*' xuexpr
	{
		$$ = new(OIND, $2, Z);
	}
|	'&' xuexpr
	{
		$$ = new(OADDR, $2, Z);
	}
|	'+' xuexpr
	{
		$$ = new(OPOS, $2, Z);
	}
|	'-' xuexpr
	{
		$$ = new(ONEG, $2, Z);
	}
|	'!' xuexpr
	{
		$$ = new(ONOT, $2, Z);
	}
|	'~' xuexpr
	{
		$$ = new(OCOM, $2, Z);
	}
|	LPP xuexpr
	{
		$$ = new(OPREINC, $2, Z);
	}
|	LMM xuexpr
	{
		$$ = new(OPREDEC, $2, Z);
	}
|	LSIZEOF uexpr
	{
		$$ = new(OSIZE, $2, Z);
	}
|	LSIGNOF uexpr
	{
		$$ = new(OSIGN, $2, Z);
	}

pexpr:
	'(' cexpr ')'
	{
		$$ = $2;
	}
|	LSIZEOF '(' tlist abdecor ')'
	{
		$$ = new(OSIZE, Z, Z);
		dodecl(NODECL, CXXX, $3, $4);
		$$->type = lastdcl;
	}
|	LSIGNOF '(' tlist abdecor ')'
	{
		$$ = new(OSIGN, Z, Z);
		dodecl(NODECL, CXXX, $3, $4);
		$$->type = lastdcl;
	}
|	pexpr '(' zelist ')'
	{
		$$ = new(OFUNC, $1, Z);
		if($1->op == ONAME)
		if($1->type == T)
			dodecl(xdecl, CXXX, types[TINT], $$);
		$$->right = invert($3);
	}
|	pexpr '[' cexpr ']'
	{
		$$ = new(OIND, new(OADD, $1, $3), Z);
	}
|	pexpr LMG ltag
	{
		$$ = new(ODOT, new(OIND, $1, Z), Z);
		$$->sym = $3;
	}
|	pexpr '.' ltag
	{
		$$ = new(ODOT, $1, Z);
		$$->sym = $3;
	}
|	pexpr LPP
	{
		$$ = new(OPOSTINC, $1, Z);
	}
|	pexpr LMM
	{
		$$ = new(OPOSTDEC, $1, Z);
	}
|	name
|	LCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TINT];
		$$->vconst = $1;
		$$->cstring = strdup(symb);
	}
|	LLCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TLONG];
		$$->vconst = $1;
		$$->cstring = strdup(symb);
	}
|	LUCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TUINT];
		$$->vconst = $1;
		$$->cstring = strdup(symb);
	}
|	LULCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TULONG];
		$$->vconst = $1;
		$$->cstring = strdup(symb);
	}
|	LDCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TDOUBLE];
		$$->fconst = $1;
		$$->cstring = strdup(symb);
	}
|	LFCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TFLOAT];
		$$->fconst = $1;
		$$->cstring = strdup(symb);
	}
|	LVLCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TVLONG];
		$$->vconst = $1;
		$$->cstring = strdup(symb);
	}
|	LUVLCONST
	{
		$$ = new(OCONST, Z, Z);
		$$->type = types[TUVLONG];
		$$->vconst = $1;
		$$->cstring = strdup(symb);
	}
|	string
|	lstring

string:
	LSTRING
	{
		$$ = new(OSTRING, Z, Z);
		$$->type = typ(TARRAY, types[TCHAR]);
		$$->type->width = $1.l + 1;
		$$->cstring = $1.s;
		$$->sym = symstring;
		$$->etype = TARRAY;
		$$->class = CSTATIC;
	}
|	string LSTRING
	{
		char *s;
		int n;

		n = $1->type->width - 1;
		s = alloc(n+$2.l+MAXALIGN);

		memcpy(s, $1->cstring, n);
		memcpy(s+n, $2.s, $2.l);
		s[n+$2.l] = 0;

		$$ = $1;
		$$->type->width += $2.l;
		$$->cstring = s;
	}

lstring:
	LLSTRING
	{
		$$ = new(OLSTRING, Z, Z);
		$$->type = typ(TARRAY, types[TUSHORT]);
		$$->type->width = $1.l + sizeof(ushort);
		$$->rstring = (ushort*)$1.s;
		$$->sym = symstring;
		$$->etype = TARRAY;
		$$->class = CSTATIC;
	}
|	lstring LLSTRING
	{
		char *s;
		int n;

		n = $1->type->width - sizeof(ushort);
		s = alloc(n+$2.l+MAXALIGN);

		memcpy(s, $1->rstring, n);
		memcpy(s+n, $2.s, $2.l);
		*(ushort*)(s+n+$2.l) = 0;

		$$ = $1;
		$$->type->width += $2.l;
		$$->rstring = (ushort*)s;
	}

zelist:
	{
		$$ = Z;
	}
|	elist

elist:
	expr
|	elist ',' elist
	{
		$$ = new(OLIST, $1, $3);
	}

sbody:
	'{'
	{
		$<tyty>$.t1 = strf;
		$<tyty>$.t2 = strl;
		$<tyty>$.t3 = lasttype;
		$<tyty>$.c = lastclass;
		strf = T;
		strl = T;
		lastbit = 0;
		firstbit = 1;
		lastclass = CXXX;
		lasttype = T;
	}
	edecl '}'
	{
		$$ = strf;
		strf = $<tyty>2.t1;
		strl = $<tyty>2.t2;
		lasttype = $<tyty>2.t3;
		lastclass = $<tyty>2.c;
	}

zctlist:
	{
		lastclass = CXXX;
		lasttype = types[TINT];
	}
|	ctlist

types:
	complex
	{
		$$.t = $1;
		$$.c = CXXX;
	}
|	tname
	{
		$$.t = simplet($1);
		$$.c = CXXX;
	}
|	gcnlist
	{
		$$.t = simplet($1);
		$$.c = simplec($1);
		$$.t = garbt($$.t, $1);
	}
|	complex gctnlist
	{
		$$.t = $1;
		$$.c = simplec($2);
		$$.t = garbt($$.t, $2);
		if($2 & ~BCLASS & ~BGARB)
			diag(Z, "duplicate types given: %T and %Q", $1, $2);
	}
|	tname gctnlist
	{
		$$.t = simplet(typebitor($1, $2));
		$$.c = simplec($2);
		$$.t = garbt($$.t, $2);
	}
|	gcnlist complex zgnlist
	{
		$$.t = $2;
		$$.c = simplec($1);
		$$.t = garbt($$.t, $1|$3);
	}
|	gcnlist tname
	{
		$$.t = simplet($2);
		$$.c = simplec($1);
		$$.t = garbt($$.t, $1);
	}
|	gcnlist tname gctnlist
	{
		$$.t = simplet(typebitor($2, $3));
		$$.c = simplec($1|$3);
		$$.t = garbt($$.t, $1|$3);
	}

tlist:
	types
	{
		$$ = $1.t;
		if($1.c != CXXX)
			diag(Z, "illegal combination of class 4: %s", cnames[$1.c]);
	}

ctlist:
	types
	{
		lasttype = $1.t;
		lastclass = $1.c;
	}

complex:
	LSTRUCT ltag
	{
		dotag($2, TSTRUCT, 0);
		$$ = $2->suetag;
	}
|	LSTRUCT ltag
	{
		dotag($2, TSTRUCT, autobn);
	}
	sbody
	{
		$$ = $2->suetag;
		if($$->link != T)
			diag(Z, "redeclare tag: %s", $2->name);
		$$->link = $4;
		sualign($$);
	}
|	LSTRUCT sbody
	{
		taggen++;
		sprint(symb, "_%d_", taggen);
		$$ = dotag(lookup(), TSTRUCT, autobn);
		$$->link = $2;
		sualign($$);
	}
|	LUNION ltag
	{
		dotag($2, TUNION, 0);
		$$ = $2->suetag;
	}
|	LUNION ltag
	{
		dotag($2, TUNION, autobn);
	}
	sbody
	{
		$$ = $2->suetag;
		if($$->link != T)
			diag(Z, "redeclare tag: %s", $2->name);
		$$->link = $4;
		sualign($$);
	}
|	LUNION sbody
	{
		taggen++;
		sprint(symb, "_%d_", taggen);
		$$ = dotag(lookup(), TUNION, autobn);
		$$->link = $2;
		sualign($$);
	}
|	LENUM ltag
	{
		dotag($2, TENUM, 0);
		$$ = $2->suetag;
		if($$->link == T)
			$$->link = types[TINT];
		$$ = $$->link;
	}
|	LENUM ltag
	{
		dotag($2, TENUM, autobn);
	}
	'{'
	{
		en.tenum = T;
		en.cenum = T;
	}
	enum '}'
	{
		$$ = $2->suetag;
		if($$->link != T)
			diag(Z, "redeclare tag: %s", $2->name);
		if(en.tenum == T) {
			diag(Z, "enum type ambiguous: %s", $2->name);
			en.tenum = types[TINT];
		}
		$$->link = en.tenum;
		$$ = en.tenum;
	}
|	LENUM '{'
	{
		en.tenum = T;
		en.cenum = T;
	}
	enum '}'
	{
		$$ = en.tenum;
	}
|	LTYPE
	{
		$$ = tcopy($1->type);
	}

gctnlist:
	gctname
|	gctnlist gctname
	{
		$$ = typebitor($1, $2);
	}

zgnlist:
	{
		$$ = 0;
	}
|	zgnlist gname
	{
		$$ = typebitor($1, $2);
	}

gctname:
	tname
|	gname
|	cname

gcnlist:
	gcname
|	gcnlist gcname
	{
		$$ = typebitor($1, $2);
	}

gcname:
	gname
|	cname

enum:
	LNAME
	{
		doenum($1, Z);
	}
|	LNAME '=' expr
	{
		doenum($1, $3);
	}
|	enum ','
|	enum ',' enum

tname:	/* type words */
	LCHAR { $$ = BCHAR; }
|	LSHORT { $$ = BSHORT; }
|	LINT { $$ = BINT; }
|	LLONG { $$ = BLONG; }
|	LSIGNED { $$ = BSIGNED; }
|	LUNSIGNED { $$ = BUNSIGNED; }
|	LFLOAT { $$ = BFLOAT; }
|	LDOUBLE { $$ = BDOUBLE; }
|	LVOID { $$ = BVOID; }

cname:	/* class words */
	LAUTO { $$ = BAUTO; }
|	LSTATIC { $$ = BSTATIC; }
|	LEXTERN { $$ = BEXTERN; }
|	LTYPEDEF { $$ = BTYPEDEF; }
|	LTYPESTR { $$ = BTYPESTR; }
|	LREGISTER { $$ = BREGISTER; }
|	LINLINE { $$ = 0; }

gname:	/* garbage words */
	LCONSTNT { $$ = BCONSTNT; }
|	LVOLATILE { $$ = BVOLATILE; }
|	LRESTRICT { $$ = 0; }

name:
	LNAME
	{
		$$ = new(ONAME, Z, Z);
		if($1->class == CLOCAL)
			$1 = mkstatic($1);
		$$->sym = $1;
		$$->type = $1->type;
		$$->etype = TVOID;
		if($$->type != T)
			$$->etype = $$->type->etype;
		$$->xoffset = $1->offset;
		$$->class = $1->class;
		$1->aused = 1;
	}
tag:
	ltag
	{
		$$ = new(ONAME, Z, Z);
		$$->sym = $1;
		$$->type = $1->type;
		$$->etype = TVOID;
		if($$->type != T)
			$$->etype = $$->type->etype;
		$$->xoffset = $1->offset;
		$$->class = $1->class;
	}
ltag:
	LNAME
|	LTYPE
%%
