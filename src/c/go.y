// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

%{
#include "go.h"
%}
%union	{
	Node*		node;
	Sym*		sym;
	struct	Val	val;
	int		lint;
}
%token	<sym>		LNAME LBASETYPE LATYPE LANY LPACK LACONST
%token	<val>		LLITERAL LASOP
%token			LPACKAGE LIMPORT LEXPORT
%token			LMAP LCHAN LINTERFACE LFUNC LSTRUCT
%token			LCOLAS LFALL LRETURN
%token			LNEW LLEN
%token			LVAR LTYPE LCONST LCONVERT
%token			LFOR LIF LELSE LSWITCH LCASE LDEFAULT
%token			LBREAK LCONTINUE LGO LGOTO LRANGE
%token			LOROR LANDAND LEQ LNE LLE LLT LGE LGT
%token			LLSH LRSH LINC LDEC
%token			LNIL LTRUE LFALSE LIOTA
%token			LPANIC LPRINT LIGNORE

%type	<sym>		sym laconst lname latype
%type	<lint>		chantype
%type	<node>		xdcl xdcl_list_r oxdcl_list common_dcl
%type	<node>		oarg_type_list arg_type_list_r arg_type
%type	<node>		stmt empty_stmt else_stmt
%type	<node>		complex_stmt compound_stmt stmt_list_r ostmt_list
%type	<node>		for_stmt for_body for_header
%type	<node>		if_stmt if_body if_header
%type	<node>		range_header range_body range_stmt
%type	<node>		simple_stmt osimple_stmt
%type	<node>		expr uexpr pexpr expr_list oexpr oexpr_list expr_list_r
%type	<node>		name name_name new_name new_name_list_r
%type	<node>		type polytype
%type	<node>		new_type
%type	<node>		vardcl_list_r vardcl
%type	<node>		constdcl_list_r constdcl
%type	<node>		typedcl_list_r typedcl
%type	<node>		interfacedcl_list_r interfacedcl
%type	<node>		structdcl_list_r structdcl
%type	<node>		export_list_r export
%type	<node>		hidden_importsym_list_r ohidden_importsym_list hidden_importsym isym
%type	<node>		hidden_importfield_list_r ohidden_importfield_list hidden_importfield
%type	<node>		fntype fnbody fntypeh fnlitdcl intype
%type	<node>		fnres fnliteral xfndcl fndcl
%type	<node>		keyval_list_r keyval

%left			LOROR
%left			LANDAND
%left			LEQ LNE LLE LGE LLT LGT
%left			'+' '-' '|' '^'
%left			'*' '/' '%' '&' LLSH LRSH
%%
file:
	package imports oxdcl_list
	{
		if(debug['f'])
			frame(1);
		testdclstack();
	}

package:
	{
		yyerror("package statement must be first");
		mkpackage("main");
	}
|	LPACKAGE sym
	{
		mkpackage($2->name);
	}

imports:
|	imports import

import:
	LIMPORT import_stmt
|	LIMPORT '(' import_stmt_list_r osemi ')'

import_stmt:
	import_here import_there

import_here:
	LLITERAL
	{
		// import with original name
		pkgmyname = S;
		importfile(&$1);
	}
|	sym LLITERAL
	{
		// import with given name
		pkgmyname = $1;
		pkgmyname->lexical = LPACK;
		importfile(&$2);
	}
|	'.' LLITERAL
	{
		// import with my name
		pkgmyname = lookup(package);
		importfile(&$2);
	}

import_there:
	hidden_import_list_r ')' ')'
	{
		unimportfile();
	}
|	LIMPORT '(' '(' hidden_import_list_r ')' ')'

/*
 * declarations
 */
xdcl:
	common_dcl
|	LEXPORT export_list_r
	{
		markexport(rev($2));
	}
|	LEXPORT '(' export_list_r ')'
	{
		markexport(rev($3));
	}
|	xfndcl
|	';'
	{
		$$ = N;
	}

common_dcl:
	LVAR vardcl
	{
		$$ = $2;
	}
|	LVAR '(' vardcl_list_r osemi ')'
	{
		$$ = rev($3);
	}
|	LCONST constdcl
	{
		$$ = $2;
		iota = 0;
	}
|	LCONST '(' constdcl_list_r osemi ')'
	{
		$$ = rev($3);
		iota = 0;
	}
|	LTYPE typedcl
	{
		$$ = $2;
	}
|	LTYPE '(' typedcl_list_r osemi ')'
	{
		$$ = rev($3);
	}

vardcl:
	new_name_list_r type
	{
		$$ = rev($1);
		dodclvar($$, $2);

		$$ = nod(ODCLVAR, $$, N);
		$$->type = $2;
	}
|	new_name_list_r type '=' oexpr_list
	{
		$$ = rev($1);
		dodclvar($$, $2);

		$$ = nod(ODCLVAR, $$, $4);
		$$->type = $2;
	}
|	new_name '=' expr
	{
		walktype($3, 0);	// this is a little harry
		defaultlit($3);
		dodclvar($1, $3->type);

		$$ = nod(ODCLVAR, $1, $3);
		$$->type = $3->type;
	}

constdcl:
	new_name '=' expr
	{
		walktype($3, 0);
		dodclconst($1, $3);

		$$ = nod(ODCLCONST, $1, $3);
		iota += 1;
	}
|	new_name type '=' expr
	{
		walktype($4, 0);
		convlit($4, $2);
		dodclconst($1, $4);

		$$ = nod(ODCLCONST, $1, $4);
		iota += 1;
	}

typedcl:
	new_type type
	{
		dodcltype($1, $2);

		$$ = nod(ODCLTYPE, $1, N);
		$$->type = $2;
	}

/*
 * statements
 */
stmt:
	error ';'
	{
		$$ = N;
		context = nil;
	}
|	common_dcl ';'
	{
		$$ = $1;
	}
|	simple_stmt ';'
|	complex_stmt
|	compound_stmt
|	empty_stmt

empty_stmt:
	';'
	{
		$$ = nod(OEMPTY, N, N);
	}

else_stmt:
	stmt
	{
		$$ = $1;
		switch($$->op) {
		case OLABEL:
		case OXCASE:
		case OXFALL:
			yyerror("statement cannot be labeled");
		}
	}

simple_stmt:
	expr
	{
		$$ = $1;
	}
|	expr LINC
	{
		$$ = nod(OASOP, $1, literal(1));
		$$->kaka = OADD;
	}
|	expr LDEC
	{
		$$ = nod(OASOP, $1, literal(1));
		$$->kaka = OSUB;
	}
|	expr LASOP expr
	{
		$$ = nod(OASOP, $1, $3);
		$$->kaka = $2.vval;	// rathole to pass opcode
	}
|	expr_list '=' expr_list
	{
		$$ = nod(OAS, $1, $3);
	}
|	new_name LCOLAS expr
	{
		walktype($3, 0);	// this is a little harry
		defaultlit($3);
		dodclvar($1, $3->type);
		$$ = nod(OCOLAS, $1, $3);
	}

complex_stmt:
	LFOR for_stmt
	{
		/* FOR and WHILE are the same keyword */
		popdcl("for/while");
		$$ = $2;
	}
|	LSWITCH if_stmt
	{
		popdcl("if/switch");
		if(!casebody($2->nbody))
			yyerror("switch statement must have case labels");
		$$ = $2;
		$$->op = OSWITCH;
	}
|	LIF if_stmt
	{
		popdcl("if/switch");
		$$ = $2;
	}
|	LIF if_stmt LELSE else_stmt
	{
		popdcl("if/switch");
		$$ = $2;
		$$->nelse = $4;
	}
|	LRANGE range_stmt
	{
		popdcl("range");
		$$ = $2;
	}
|	LRETURN oexpr_list ';'
	{
		$$ = nod(ORETURN, $2, N);
	}
|	LCASE expr_list ':'
	{
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		poptodcl();
		$$ = nod(OXCASE, $2, N);
	}
|	LDEFAULT ':'
	{
		poptodcl();
		$$ = nod(OXCASE, N, N);
	}
|	LFALL ';'
	{
		// will be converted to OFALL
		$$ = nod(OXFALL, N, N);
	}
|	LBREAK oexpr ';'
	{
		$$ = nod(OBREAK, $2, N);
	}
|	LCONTINUE oexpr ';'
	{
		$$ = nod(OCONTINUE, $2, N);
	}
|	LGO pexpr '(' oexpr_list ')' ';'
	{
		$$ = nod(OPROC, $2, $4);
	}
|	LPRINT expr_list ';'
	{
		$$ = nod(OPRINT, $2, N);
	}
|	LPANIC oexpr_list ';'
	{
		$$ = nod(OPANIC, $2, N);
	}
|	LGOTO new_name ';'
	{
		$$ = nod(OGOTO, $2, N);
	}
|	new_name ':'
	{
		$$ = nod(OLABEL, $1, N);
	}

compound_stmt:
	'{'
	{
		markdcl("compound");
	} ostmt_list '}'
	{
		$$ = $3;
		if($$ == N)
			$$ = nod(OEMPTY, N, N);
		popdcl("compound");
	}

for_header:
	osimple_stmt ';' osimple_stmt ';' osimple_stmt
	{
		// init ; test ; incr
		$$ = nod(OFOR, N, N);
		$$->ninit = $1;
		$$->ntest = $3;
		$$->nincr = $5;
	}
|	osimple_stmt
	{
		// test
		$$ = nod(OFOR, N, N);
		$$->ninit = N;
		$$->ntest = $1;
		$$->nincr = N;
	}

for_body:
	for_header compound_stmt
	{
		$$ = $1;
		$$->nbody = $2;
	}

for_stmt:
	{
		markdcl("for/while");
	} for_body
	{
		$$ = $2;
	}

if_header:
	osimple_stmt
	{
		// test
		$$ = nod(OIF, N, N);
		$$->ninit = N;
		$$->ntest = $1;
	}
|	osimple_stmt ';' osimple_stmt
	{
		// init ; test
		$$ = nod(OIF, N, N);
		$$->ninit = $1;
		$$->ntest = $3;
	}

if_body:
	if_header compound_stmt
	{
		$$ = $1;
		$$->nbody = $2;
	}

if_stmt:
	{
		markdcl("if/switch");
	} if_body
	{
		$$ = $2;
	}

range_header:
	new_name LCOLAS expr
	{
		$$ = N;
	}
|	new_name ',' new_name LCOLAS expr
	{
		$$ = N;
	}
|	new_name ',' new_name '=' expr
	{
		yyerror("range statement only allows := assignment");
		$$ = N;
	}

range_body:
	range_header compound_stmt
	{
		$$ = $1;
		$$->nbody = $2;
	}

range_stmt:
	{
		markdcl("range");
	} range_body
	{
		$$ = $2;
	}

/*
 * expressions
 */
expr:
	uexpr
|	expr LOROR expr
	{
		$$ = nod(OOROR, $1, $3);
	}
|	expr LANDAND expr
	{
		$$ = nod(OANDAND, $1, $3);
	}
|	expr LEQ expr
	{
		$$ = nod(OEQ, $1, $3);
	}
|	expr LNE expr
	{
		$$ = nod(ONE, $1, $3);
	}
|	expr LLT expr
	{
		$$ = nod(OLT, $1, $3);
	}
|	expr LLE expr
	{
		$$ = nod(OLE, $1, $3);
	}
|	expr LGE expr
	{
		$$ = nod(OGE, $1, $3);
	}
|	expr LGT expr
	{
		$$ = nod(OGT, $1, $3);
	}
|	expr '+' expr
	{
		$$ = nod(OADD, $1, $3);
	}
|	expr '-' expr
	{
		$$ = nod(OSUB, $1, $3);
	}
|	expr '|' expr
	{
		$$ = nod(OOR, $1, $3);
	}
|	expr '^' expr
	{
		$$ = nod(OXOR, $1, $3);
	}
|	expr '*' expr
	{
		$$ = nod(OMUL, $1, $3);
	}
|	expr '/' expr
	{
		$$ = nod(ODIV, $1, $3);
	}
|	expr '%' expr
	{
		$$ = nod(OMOD, $1, $3);
	}
|	expr '&' expr
	{
		$$ = nod(OAND, $1, $3);
	}
|	expr LLSH expr
	{
		$$ = nod(OLSH, $1, $3);
	}
|	expr LRSH expr
	{
		$$ = nod(ORSH, $1, $3);
	}

uexpr:
	pexpr
|	LCONVERT '(' type ',' expr ')'
	{
		$$ = nod(OCONV, $5, N);
		$$->type = $3;
	}
|	'*' uexpr
	{
		$$ = nod(OIND, $2, N);
	}
|	'&' uexpr
	{
		$$ = nod(OADDR, $2, N);
	}
|	'+' uexpr
	{
		$$ = nod(OPLUS, $2, N);
	}
|	'-' uexpr
	{
		$$ = nod(OMINUS, $2, N);
	}
|	'!' uexpr
	{
		$$ = nod(ONOT, $2, N);
	}
|	'~' uexpr
	{
		yyerror("the OCOM operator is ^");
		$$ = nod(OCOM, $2, N);
	}
|	'^' uexpr
	{
		$$ = nod(OCOM, $2, N);
	}
|	LLT uexpr
	{
		$$ = nod(ORECV, $2, N);
	}
|	LGT uexpr
	{
		$$ = nod(OSEND, $2, N);
	}

pexpr:
	LLITERAL
	{
		$$ = nod(OLITERAL, N, N);
		$$->val = $1;
	}
|	laconst
	{
		$$ = nod(OLITERAL, N, N);
		$$->val = $1->oconst->val;
		$$->type = $1->oconst->type;
	}
|	LNIL
	{
		$$ = nod(OLITERAL, N, N);
		$$->val.ctype = CTNIL;
		$$->val.vval = 0;
	}
|	LTRUE
	{
		$$ = booltrue;
	}
|	LFALSE
	{
		$$ = boolfalse;
	}
|	LIOTA
	{
		$$ = literal(iota);
	}
|	name
|	'(' expr ')'
	{
		$$ = $2;
	}
|	pexpr '.' sym
	{
		$$ = nod(ODOT, $1, newname($3));
	}
|	pexpr '[' expr ']'
	{
		$$ = nod(OINDEX, $1, $3);
	}
|	pexpr '[' keyval ']'
	{
		$$ = nod(OSLICE, $1, $3);
	}
|	pexpr '(' oexpr_list ')'
	{
		$$ = nod(OCALL, $1, $3);
	}
|	LLEN '(' name ')'
	{
		$$ = nod(OLEN, $3, N);
	}
|	LNEW '(' type ')'
	{
		$$ = nod(ONEW, N, N);
		$$->type = ptrto($3);
	}
|	fnliteral
|	'[' expr_list ']'
	{
		// array literal
		$$ = N;
	}
|	'[' keyval_list_r ']'
	{
		// map literal
		$$ = N;
	}
|	latype '(' oexpr_list ')'
	{
		// struct literal and conversions
		$$ = nod(OCONV, $3, N);
		$$->type = $1->otype;
	}

/*
 * lexical symbols that can be
 * from other packages
 */
lpack:
	LPACK 
	{
		context = $1->name;
	}

laconst:
	LACONST
|	lpack '.' LACONST
	{
		$$ = $3;
		context = nil;
	}

lname:
	LNAME
|	lpack '.' LNAME
	{
		$$ = $3;
		context = nil;
	}

latype:
	LATYPE
|	lpack '.' LATYPE
	{
		$$ = $3;
		context = nil;
	}

/*
 * names and types
 *	newname is used before declared
 *	oldname is used after declared
 */
name_name:
	LNAME
	{
		$$ = newname($1);
	}

new_name:
	sym
	{
		$$ = newname($1);
	}

new_type:
	sym
	{
		$$ = newtype($1);
	}

sym:
	LATYPE
|	LNAME
|	LACONST
|	LPACK

name:
	lname
	{
		$$ = oldname($1);
	}

type:
	latype
	{
		$$ = oldtype($1);
	}
|	'[' oexpr ']' type
	{
		$$ = aindex($2, $4);
	}
|	LCHAN chantype polytype
	{
		$$ = nod(OTYPE, N, N);
		$$->etype = TCHAN;
		$$->type = $3;
		$$->chan = $2;
	}
|	LMAP '[' type ']' polytype
	{
		$$ = nod(OTYPE, N, N);
		$$->etype = TMAP;
		$$->down = $3;
		$$->type = $5;
	}
|	LSTRUCT '{' structdcl_list_r osemi '}'
	{
		$$ = dostruct(rev($3), TSTRUCT);
	}
|	LSTRUCT '{' '}'
	{
		$$ = dostruct(N, TSTRUCT);
	}
|	LINTERFACE '{' interfacedcl_list_r osemi '}'
	{
		$$ = dostruct(rev($3), TINTER);
		$$ = sortinter($$);
	}
|	LINTERFACE '{' '}'
	{
		$$ = dostruct(N, TINTER);
	}
|	fntypeh
|	'*' type
	{
		$$ = ptrto($2);
	}
|	'*' lname
	{
		// dont know if this is an error or not
		if(dclcontext != PEXTERN)
			yyerror("foreward type in function body %s", $2->name);
		$$ = forwdcl($2);
	}

polytype:
	type
|	LANY
	{
		$$ = nod(OTYPE, N, N);
		$$->etype = TPOLY;
	}

chantype:
	{
		$$ = Cboth;
	}
|	LLT
	{
		$$ = Crecv;
	}
|	LGT
	{
		$$ = Csend;
	}

keyval:
	expr ':' expr
	{
		$$ = nod(OLIST, $1, $3);
	}

/*
 * function stuff
 * all in one place to show how crappy it all is
 */
xfndcl:
	LFUNC fndcl fnbody
	{
		$$ = $2;
		$$->nbody = $3;
		funcbody($$);
	}

fndcl:
	new_name '(' oarg_type_list ')' fnres
	{
		b0stack = dclstack;	// mark base for fn literals
		$$ = nod(ODCLFUNC, N, N);
		$$->nname = $1;
		$$->type = functype(N, $3, $5);
		funchdr($$);
	}
|	'(' oarg_type_list ')' new_name '(' oarg_type_list ')' fnres
	{
		b0stack = dclstack;	// mark base for fn literals
		if($2 == N || $2->op == OLIST)
			yyerror("syntax error in method receiver");
		$$ = nod(ODCLFUNC, N, N);
		$$->nname = $4;
		$$->type = functype($2, $6, $8);
		funchdr($$);
	}

fntypeh:
	LFUNC '(' oarg_type_list ')' fnres
	{
		$$ = functype(N, $3, $5);
		funcnam($$, nil);
	}
/* i dont believe that this form is useful for nothing */
|	LFUNC '(' oarg_type_list ')' '.' '(' oarg_type_list ')' fnres
	{
		if($3 == N || $3->op == OLIST)
			yyerror("syntax error in method receiver");
		$$ = functype($3, $7, $9);
		funcnam($$, nil);
	}

fntype:
	fntypeh
|	latype
	{
		$$ = oldtype($1);
		if($$ == N || $$->etype != TFUNC)
			yyerror("illegal type for function literal");
	}

fnlitdcl:
	fntype
	{
		markdclstack();	// save dcl stack and revert to block0
		$$ = $1;
		funcargs($$);
	}

fnliteral:
	fnlitdcl '{' ostmt_list '}'
	{
		popdcl("fnlit");

		vargen++;
		snprint(namebuf, sizeof(namebuf), "_f%.3ld", vargen);

		$$ = newname(lookup(namebuf));
		addvar($$, $1, PEXTERN);

		{
			Node *n;

			n = nod(ODCLFUNC, N, N);
			n->nname = $$;
			n->type = $1;
			n->nbody = $3;
			if(n->nbody == N)
				n->nbody = nod(ORETURN, N, N);
			compile(n);
		}

		$$ = nod(OADDR, $$, N);
	}

fnbody:
	compound_stmt
	{
		$$ = $1;
		if($$->op == OEMPTY)
			$$ = nod(ORETURN, N, N);
	}
|	';'
	{
		$$ = N;
	}

fnres:
	{
		$$ = N;
	}
|	type
	{
		$$ = nod(ODCLFIELD, N, N);
		$$->type = $1;
		$$ = cleanidlist($$);
	}
|	'(' oarg_type_list ')'
	{
		$$ = $2;
	}

/*
 * lists of things
 * note that they are left recursive
 * to conserve yacc stack. they need to
 * be reversed to interpret correctly
 */
xdcl_list_r:
	xdcl
|	xdcl_list_r xdcl
	{
		$$ = nod(OLIST, $1, $2);
	}

vardcl_list_r:
	vardcl
|	vardcl_list_r ';' vardcl
	{
		$$ = nod(OLIST, $1, $3);
	}

constdcl_list_r:
	constdcl
|	constdcl_list_r ';' constdcl
	{
		$$ = nod(OLIST, $1, $3);
	}

typedcl_list_r:
	typedcl
|	typedcl_list_r ';' typedcl
	{
		$$ = nod(OLIST, $1, $3);
	}

structdcl_list_r:
	structdcl
	{
		$$ = cleanidlist($1);
	}
|	structdcl_list_r ';' structdcl
	{
		$$ = cleanidlist($3);
		$$ = nod(OLIST, $1, $$);
	}

interfacedcl_list_r:
	interfacedcl
	{
		$$ = cleanidlist($1);
	}
|	interfacedcl_list_r ';' interfacedcl
	{
		$$ = cleanidlist($3);
		$$ = nod(OLIST, $1, $$);
	}

structdcl:
	new_name ',' structdcl
	{
		$$ = nod(ODCLFIELD, $1, N);
		$$ = nod(OLIST, $$, $3);
	}
|	new_name type
	{
		$$ = nod(ODCLFIELD, $1, N);
		$$->type = $2;
	}

interfacedcl:
	new_name ',' interfacedcl
	{
		$$ = nod(ODCLFIELD, $1, N);
		$$ = nod(OLIST, $$, $3);
	}
|	new_name intype
	{
		$$ = nod(ODCLFIELD, $1, N);
		$$->type = $2;
	}

intype:
	'(' oarg_type_list ')' fnres
	{
		// without func keyword
		$$ = functype(N, $2, $4);
		funcnam($$, nil);
	}
|	LFUNC '(' oarg_type_list ')' fnres
	{
		// with func keyword
		$$ = functype(N, $3, $5);
		funcnam($$, nil);
	}
|	latype
	{
		$$ = oldtype($1);
		if($$ == N || $$->etype != TFUNC)
			yyerror("illegal type for function literal");
	}

arg_type:
	name_name
	{
		$$ = nod(ODCLFIELD, $1, N);
	}
|	type
	{
		$$ = nod(ODCLFIELD, N, N);
		$$->type = $1;
	}
|	new_name type
	{
		$$ = nod(ODCLFIELD, $1, N);
		$$->type = $2;
	}

arg_type_list_r:
	arg_type
|	arg_type_list_r ',' arg_type
	{
		$$ = nod(OLIST, $1, $3);
	}

stmt_list_r:
	stmt
	{
		$$ = $1;
	}
|	stmt_list_r stmt
	{
		$$ = nod(OLIST, $1, $2);
	}

expr_list_r:
	expr
|	expr_list_r ',' expr
	{
		$$ = nod(OLIST, $1, $3);
	}

new_name_list_r:
	new_name
|	new_name_list_r ',' new_name
	{
		$$ = nod(OLIST, $1, $3);
	}

export_list_r:
	export
|	export_list_r ocomma export
	{
		$$ = nod(OLIST, $1, $3);
	}

export:
	sym
	{
		$$ = nod(OEXPORT, N, N);
		$$->sym = $1;
	}
|	sym '.' sym
	{
		$$ = nod(OEXPORT, N, N);
		$$->psym = $1;
		$$->sym = $3;
	}

import_stmt_list_r:
	import_stmt
|	import_stmt_list_r osemi import_stmt

hidden_import_list_r:
	hidden_import
|	hidden_import_list_r hidden_import

hidden_importsym_list_r:
	hidden_importsym
|	hidden_importsym_list_r hidden_importsym
	{
		$$ = nod(OLIST, $1, $2);
	}

hidden_importfield_list_r:
	hidden_importfield
|	hidden_importfield_list_r hidden_importfield
	{
		$$ = nod(OLIST, $1, $2);
	}

keyval_list_r:
	keyval
|	keyval_list_r ',' keyval
	{
		$$ = nod(OLIST, $1, $3);
	}

/*
 * the one compromise of a
 * non-reversed list
 */
expr_list:
	expr_list_r
	{
		$$ = rev($1);
	}

/*
 * optional things
 */
osemi:
|	';'

ocomma:
|	','

oexpr:
	{
		$$ = N;
	}
|	expr

oexpr_list:
	{
		$$ = N;
	}
|	expr_list

osimple_stmt:
	{
		$$ = N;
	}
|	simple_stmt

ostmt_list:
	{
		$$ = N;
	}
|	stmt_list_r
	{
		$$ = rev($1);
	}

oxdcl_list:
	{
		$$ = N;
	}
|	xdcl_list_r
	{
		$$ = rev($1);
	}

ohidden_importsym_list:
	{
		$$ = N;
	}
|	hidden_importsym_list_r
	{
		$$ = rev($1);
	}

ohidden_importfield_list:
	{
		$$ = N;
	}
|	hidden_importfield_list_r
	{
		$$ = rev($1);
	}

oarg_type_list:
	{
		$$ = N;
	}
|	arg_type_list_r
	{
		$$ = cleanidlist(rev($1));
	}

/*
 * import syntax from header of
 * an output package
 */
hidden_import:
	/* variables */
	LVAR hidden_importsym hidden_importsym
	{
		// var
		doimportv1($2, $3);
	}

	/* constants */
|	LCONST hidden_importsym LLITERAL
	{
		doimportc1($2, &$3);
	}
|	LCONST hidden_importsym hidden_importsym LLITERAL
	{
		doimportc2($2, $3, &$4);
	}

	/* types */
|	LTYPE hidden_importsym '[' hidden_importsym ']' hidden_importsym
	{
		// type map
		doimport1($2, $4, $6);
	}
|	LTYPE hidden_importsym '[' LLITERAL ']' hidden_importsym
	{
		// type array
		doimport2($2, &$4, $6);
	}
|	LTYPE hidden_importsym '(' ohidden_importsym_list ')'
	{
		// type function
		doimport3($2, $4);
	}
|	LTYPE hidden_importsym '{' ohidden_importfield_list '}'
	{
		// type structure
		doimport4($2, $4);
	}
|	LTYPE hidden_importsym LLITERAL
	{
		// type basic
		doimport5($2, &$3);
	}
|	LTYPE hidden_importsym '*' hidden_importsym
	{
		// type pointer
		doimport6($2, $4);
	}
|	LTYPE hidden_importsym LLT ohidden_importfield_list LGT
	{
		// type interface
		doimport7($2, $4);
	}

isym:
	sym '.' sym
	{
		$$ = nod(OIMPORT, N, N);
		$$->osym = $1;
		$$->psym = $1;
		$$->sym = $3;
	}
|	'(' sym ')' sym '.' sym
	{
		$$ = nod(OIMPORT, N, N);
		$$->osym = $2;
		$$->psym = $4;
		$$->sym = $6;
	}

hidden_importsym:
	isym
|	'!' isym
	{
		$$ = $2;
		$$->kaka = 1;
	}

hidden_importfield:
	sym isym
	{
		$$ = $2;
		$$->fsym = $1;
	}
