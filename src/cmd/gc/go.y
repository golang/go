// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

%{
#include "go.h"
%}
%union	{
	Node*		node;
	Type*		type;
	Sym*		sym;
	struct	Val	val;
	int		lint;
}
%token	<val>		LLITERAL
%token	<lint>		LASOP
%token	<sym>		LNAME LBASETYPE LATYPE LPACK LACONST
%token	<sym>		LPACKAGE LIMPORT LEXPORT
%token	<sym>		LMAP LCHAN LINTERFACE LFUNC LSTRUCT
%token	<sym>		LCOLAS LFALL LRETURN
%token	<sym>		LNEW LLEN LCAP LTYPEOF LPANIC LPRINT
%token	<sym>		LVAR LTYPE LCONST LCONVERT LSELECT
%token	<sym>		LFOR LIF LELSE LSWITCH LCASE LDEFAULT
%token	<sym>		LBREAK LCONTINUE LGO LGOTO LRANGE
%token	<sym>		LNIL LTRUE LFALSE LIOTA

%token			LOROR LANDAND LEQ LNE LLE LLT LGE LGT
%token			LLSH LRSH LINC LDEC LCOMM
%token			LIGNORE

%type	<sym>		sym sym1 sym2 keyword laconst lname latype non_type_sym
%type	<node>		xdcl xdcl_list_r oxdcl_list
%type	<node>		common_dcl Acommon_dcl Bcommon_dcl
%type	<node>		oarg_type_list arg_type_list_r arg_type
%type	<node>		else_stmt1 else_stmt2 inc_stmt noninc_stmt
%type	<node>		complex_stmt compound_stmt ostmt_list
%type	<node>		stmt_list_r Astmt_list_r Bstmt_list_r
%type	<node>		Astmt Bstmt Cstmt Dstmt
%type	<node>		for_stmt for_body for_header
%type	<node>		if_stmt if_body if_header
%type	<node>		range_header range_body range_stmt select_stmt
%type	<node>		simple_stmt osimple_stmt semi_stmt
%type	<node>		expr uexpr pexpr expr_list oexpr oexpr_list expr_list_r
%type	<node>		name name_name onew_name new_name new_name_list_r non_type_new_name
%type	<node>		vardcl_list_r vardcl Avardcl Bvardcl
%type	<node>		interfacedcl_list_r interfacedcl
%type	<node>		structdcl_list_r structdcl
%type	<node>		hidden_importsym_list_r ohidden_importsym_list hidden_importsym isym
%type	<node>		hidden_importfield_list_r ohidden_importfield_list hidden_importfield
%type	<node>		fnres Afnres Bfnres fnliteral xfndcl fndcl fnbody
%type	<node>		keyexpr_list keyval_list_r keyval
%type	<node>		typedcl Atypedcl Btypedcl

%type	<type>		fntype fnlitdcl Afntype Bfntype fullAtype
%type	<type>		type Atype Btype indcl new_type fullBtype
%type	<type>		structtype interfacetype convtype
%type	<type>		Achantype Bchantype

%left			LOROR
%left			LANDAND
%left			LCOMM
%left			LEQ LNE LLE LGE LLT LGT
%left			'+' '-' '|' '^'
%left			'*' '/' '%' '&' LLSH LRSH
%%
file:
	package import_there imports oxdcl_list
	{
		if(debug['f'])
			frame(1);
		fninit($4);
		testdclstack();
	}

package:
	{
		yyerror("package statement must be first");
		mkpackage("main");
		cannedimports();
	}
|	LPACKAGE sym
	{
		mkpackage($2->name);
		cannedimports();
	}

imports:
|	imports import

import:
	LIMPORT import_stmt
|	LIMPORT '(' import_stmt_list_r osemi ')'

import_stmt:
	import_here import_package import_there

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

import_package:
	LPACKAGE sym
	{
		pkgimportname = $2;

		// if we are not remapping the package name
		// then the imported package name is LPACK
		if(pkgmyname == S)
			pkgimportname->lexical = LPACK;
	}

import_there:
	hidden_import_list_r ')' ')'
	{
		checkimports();
		unimportfile();
	}
|	LIMPORT '(' '(' hidden_import_list_r ')' ')'
	{
		checkimports();
	}

/*
 * declarations
 */
xdcl:
	common_dcl
|	xfndcl
	{
		$$ = N;
	}
|	LEXPORT export_list_r
	{
		$$ = N;
	}
|	LEXPORT { exportadj = 1; } common_dcl
	{
		$$ = $3;
		exportadj = 0;
	}
|	LEXPORT '(' export_list_r ')'
	{
		$$ = N;
	}
|	LEXPORT xfndcl
	{
		if($2 != N && $2->nname != N)
			exportsym($2->nname->sym);
		$$ = N;
	}
|	';'
	{
		$$ = N;
	}
|	error xdcl
	{
		$$ = $2;
	}

common_dcl:
	Acommon_dcl
|	Bcommon_dcl

Acommon_dcl:
	LVAR Avardcl
	{
		$$ = $2;
	}
|	LVAR '(' vardcl_list_r osemi ')'
	{
		$$ = rev($3);
	}
|	LCONST '(' constdcl osemi ')'
	{
		iota = 0;
		lastconst = N;
		$$ = N;
	}
|	LCONST '(' constdcl ';' constdcl_list_r osemi ')'
	{
		iota = 0;
		lastconst = N;
		$$ = N;
	}
|	LTYPE Atypedcl
	{
		$$ = N;
	}
|	LTYPE '(' typedcl_list_r osemi ')'
	{
		$$ = N;
	}

Bcommon_dcl:
	LVAR Bvardcl
	{
		$$ = $2;
	}
|	LCONST constdcl
	{
		$$ = N;
		iota = 0;
		lastconst = N;
	}
|	LTYPE Btypedcl
	{
		$$ = N;
	}

vardcl:
	Avardcl
|	Bvardcl

Avardcl:
	new_name_list_r fullAtype
	{
		$$ = rev($1);
		dodclvar($$, $2);

		$$ = nod(OAS, $$, N);
	}

Bvardcl:
	new_name_list_r fullBtype
	{
		$$ = rev($1);
		dodclvar($$, $2);

		$$ = nod(OAS, $$, N);
	}
|	new_name_list_r type '=' expr_list
	{
		$$ = rev($1);
		dodclvar($$, $2);

		$$ = nod(OAS, $$, $4);
		addtotop($$);
	}
|	new_name '=' expr
	{
		$$ = nod(OAS, $1, N);
		gettype($3, $$);
		defaultlit($3);
		dodclvar($1, $3->type);
		$$->right = $3;
	}

constdcl:
	new_name type '=' expr
	{
		Node *c = treecopy($4);
		gettype(c, N);
		convlit(c, $2);
		dodclconst($1, c);

		lastconst = $4;
		iota += 1;
	}
|	new_name '=' expr
	{
		Node *c = treecopy($3);
		gettype(c, N);
		dodclconst($1, c);

		lastconst = $3;
		iota += 1;
	}

constdcl1:
	constdcl
|	new_name type
	{
		Node *c = treecopy(lastconst);
		gettype(c, N);
		convlit(c, $2);
		dodclconst($1, c);

		iota += 1;
	}
|	new_name
	{
		Node *c = treecopy(lastconst);
		gettype(c, N);
		dodclconst($1, c);

		iota += 1;
	}

typedcl:
	Atypedcl
|	Btypedcl

Atypedcl:
	new_type fullAtype
	{
		dodcltype($1, $2);
	}

Btypedcl:
	new_type fullBtype
	{
		dodcltype($1, $2);
	}

else_stmt1:
	complex_stmt
|	compound_stmt

else_stmt2:
	simple_stmt
|	semi_stmt
|	';'
	{
		$$ = N;
	}

simple_stmt:
	inc_stmt
|	noninc_stmt

noninc_stmt:
	expr
	{
		$$ = $1;
	}
|	expr LASOP expr
	{
		$$ = nod(OASOP, $1, $3);
		$$->etype = $2;			// rathole to pass opcode
	}
|	expr_list '=' expr_list
	{
		$$ = nod(OAS, $1, $3);
	}
|	expr_list LCOLAS expr_list
	{
		$$ = nod(OAS, colas($1, $3), $3);
		addtotop($$);
	}
|	LPRINT '(' oexpr_list ')'
	{
		$$ = nod(OPRINT, $3, N);
	}
|	LPANIC '(' oexpr_list ')'
	{
		$$ = nod(OPANIC, $3, N);
	}

inc_stmt:
	expr LINC
	{
		$$ = nod(OASOP, $1, literal(1));
		$$->etype = OADD;
	}
|	expr LDEC
	{
		$$ = nod(OASOP, $1, literal(1));
		$$->etype = OSUB;
	}

complex_stmt:
	LFOR for_stmt
	{
		popdcl();
		$$ = $2;
	}
|	LSWITCH if_stmt
	{
		popdcl();
		$$ = $2;
		$$->op = OSWITCH;
	}
|	LIF if_stmt
	{
		popdcl();
		$$ = $2;
	}
|	LIF if_stmt LELSE else_stmt1
	{
		popdcl();
		$$ = $2;
		$$->nelse = $4;
	}
|	LSELECT select_stmt
	{
		popdcl();
		$$ = $2;
	}
|	LRANGE range_stmt
	{
		popdcl();
		$$ = $2;
	}
|	LCASE expr_list ':'
	{
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		poptodcl();
		$$ = nod(OXCASE, $2, N);
	}
|	LCASE name '=' expr ':'
	{
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		poptodcl();
		$$ = nod(OAS, $2, $4);
		$$ = nod(OXCASE, $$, N);
	}
|	LCASE name LCOLAS expr ':'
	{
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		poptodcl();
		$$ = nod(OAS, selectas($2,$4), $4);
		$$ = nod(OXCASE, $$, N);
	}
|	LDEFAULT ':'
	{
		poptodcl();
		$$ = nod(OXCASE, N, N);
	}

semi_stmt:
	LFALL
	{
		// will be converted to OFALL
		$$ = nod(OXFALL, N, N);
	}
|	LBREAK onew_name
	{
		$$ = nod(OBREAK, $2, N);
	}
|	LCONTINUE onew_name
	{
		$$ = nod(OCONTINUE, $2, N);
	}
|	LGO pexpr '(' oexpr_list ')'
	{
		$$ = nod(OCALL, $2, $4);
		$$ = nod(OPROC, $$, N);
	}
|	LGOTO new_name
	{
		$$ = nod(OGOTO, $2, N);
	}
|	LRETURN oexpr_list
	{
		$$ = nod(ORETURN, $2, N);
	}
|	LIF if_stmt LELSE else_stmt2
	{
		popdcl();
		$$ = $2;
		$$->nelse = $4;
	}

compound_stmt:
	'{'
	{
		markdcl();
	} ostmt_list '}'
	{
		$$ = $3;
		if($$ == N)
			$$ = nod(OEMPTY, N, N);
		popdcl();
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
		markdcl();
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
		markdcl();
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
		markdcl();
	} range_body
	{
		$$ = $2;
	}

select_stmt:
	{
		markdcl();
	}
	compound_stmt
	{
		$$ = nod(OSELECT, $2, N);
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
|	expr LCOMM expr
	{
		$$ = nod(OSEND, $1, $3);
	}

uexpr:
	pexpr
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
|	LCOMM uexpr
	{
		$$ = nod(ORECV, $2, N);
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
		$$->iota = 1;	// flag to reevaluate on copy
	}
|	name
|	'(' expr ')'
	{
		$$ = $2;
	}
|	pexpr '.' sym2
	{
		$$ = nod(ODOT, $1, newname($3));
	}
|	pexpr '.' '(' type ')'
	{
		$$ = nod(OCONV, $1, N);
		$$->type = $4;
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
|	LLEN '(' expr ')'
	{
		$$ = nod(OLEN, $3, N);
	}
|	LCAP '(' expr ')'
	{
		$$ = nod(OCAP, $3, N);
	}
|	LTYPEOF '(' type ')'
	{
		$$ = nod(OTYPEOF, N, N);
		$$->type = $3;
	}
|	LNEW '(' type ')'
	{
		$$ = nod(ONEW, N, N);
		$$->type = ptrto($3);
	}
|	LNEW '(' type ',' expr_list ')'
	{
		$$ = nod(ONEW, $5, N);
		$$->type = ptrto($3);
	}
|	LCONVERT '(' type ',' keyexpr_list ')'
	{
		$$ = nod(OCONV, $5, N);
		$$->type = $3;
	}
|	latype '(' expr ')'
	{
		$$ = nod(OCONV, $3, N);
		$$->type = oldtype($1);
	}
|	convtype '{' keyexpr_list '}'
	{
		// struct literal and conversions
		$$ = nod(OCONV, rev($3), N);
		$$->type = $1;
	}
|	fnliteral

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
	sym1
	{
		$$ = newname($1);
	}

non_type_new_name:
	non_type_sym
	{
		$$ = newname($1);
	}

new_type:
	sym1
	{
		$$ = newtype($1);
	}

onew_name:
	{
		$$ = N;
	}
|	new_name

sym:
	LATYPE
|	LNAME
|	LACONST
|	LPACK

sym1:
	sym
|	keyword

non_type_sym:
	LNAME
|	LACONST
|	LPACK
|	keyword

sym2:
	sym
|	keyword

/*
 * keywords that we can
 * use as variable/type names
 */
keyword:
	LNIL
|	LTRUE
|	LFALSE
|	LIOTA
|	LLEN
|	LCAP
|	LPANIC
|	LPRINT
|	LNEW
|	LBASETYPE
|	LTYPEOF
|	LCONVERT

name:
	lname
	{
		$$ = oldname($1);
	}

convtype:
	latype
	{
		$$ = oldtype($1);
	}
|	'[' oexpr ']' type
	{
		// array literal
		$$ = aindex($2, $4);
	}
|	LMAP '[' type ']' type
	{
		// map literal
		$$ = typ(TMAP);
		$$->down = $3;
		$$->type = $5;
	}
|	structtype

type:
	fullAtype
|	fullBtype

Atype:
	latype
	{
		$$ = oldtype($1);
	}
|	'[' oexpr ']' fullAtype
	{
		$$ = aindex($2, $4);
	}
|	LCOMM LCHAN fullAtype
	{
		$$ = typ(TCHAN);
		$$->type = $3;
		$$->chan = Crecv;
	}
|	LCHAN LCOMM Atype  /* not full Atype */
	{
		$$ = typ(TCHAN);
		$$->type = $3;
		$$->chan = Csend;
	}
|	LMAP '[' type ']' fullAtype
	{
		$$ = typ(TMAP);
		$$->down = $3;
		$$->type = $5;
	}
|	structtype
|	interfacetype
|	'*' fullAtype
	{
		dowidth($2);
		$$ = ptrto($2);
	}

Achantype:
	LCHAN fullAtype
	{
		$$ = typ(TCHAN);
		$$->type = $2;
		$$->chan = Cboth;
	}

fullAtype:
	Atype
|	Afntype
|	Achantype

Btype:
	'[' oexpr ']' fullBtype
	{
		$$ = aindex($2, $4);
	}
|	LCOMM LCHAN fullBtype
	{
		$$ = typ(TCHAN);
		$$->type = $3;
		$$->chan = Crecv;
	}
|	LCHAN LCOMM Btype  // not full Btype
	{
		$$ = typ(TCHAN);
		$$->type = $3;
		$$->chan = Csend;
	}
|	LMAP '[' type ']' fullBtype
	{
		$$ = typ(TMAP);
		$$->down = $3;
		$$->type = $5;
	}
|	'*' fullBtype
	{
		dowidth($2);
		$$ = ptrto($2);
	}
|	'*' lname
	{
		// dont know if this is an error or not
		if(dclcontext != PEXTERN)
			yyerror("forward type in function body %s", $2->name);
		$$ = forwdcl($2);
	}

Bchantype:
	LCHAN fullBtype
	{
		$$ = typ(TCHAN);
		$$->type = $2;
		$$->chan = Cboth;
	}

fullBtype:
	Btype
|	Bfntype
|	Bchantype

structtype:
	LSTRUCT '{' structdcl_list_r osemi '}'
	{
		$$ = dostruct(rev($3), TSTRUCT);
	}
|	LSTRUCT '{' '}'
	{
		$$ = dostruct(N, TSTRUCT);
	}

interfacetype:
	LINTERFACE '{' interfacedcl_list_r osemi '}'
	{
		$$ = dostruct(rev($3), TINTER);
		$$ = sortinter($$);
	}
|	LINTERFACE '{' '}'
	{
		$$ = dostruct(N, TINTER);
	}

keyval:
	expr ':' expr
	{
		$$ = nod(OKEY, $1, $3);
	}

/*
 * function stuff
 * all in one place to show how crappy it all is
 */
xfndcl:
	LFUNC
	{
		maxarg = 0;
		stksize = 0;
	} fndcl fnbody
	{
		$$ = $3;
		$$->nbody = $4;
		funcbody($$);
	}

fndcl:
	new_name '(' oarg_type_list ')' fnres
	{
		b0stack = dclstack;	// mark base for fn literals
		$$ = nod(ODCLFUNC, N, N);
		$$->nname = $1;
		if($3 == N && $5 == N)
			$$->nname = renameinit($1);
		$$->type = functype(N, $3, $5);
		funchdr($$);
	}
|	'(' oarg_type_list ')' new_name '(' oarg_type_list ')' fnres
	{
		b0stack = dclstack;	// mark base for fn literals
		$$ = nod(ODCLFUNC, N, N);
		$$->nname = methodname($4, $2->type);
		$$->type = functype($2, $6, $8);
		funchdr($$);

		addmethod($4, $$->type, 1);
	}

fntype:
	Afntype
|	Bfntype

Afntype:
	'(' oarg_type_list ')' Afnres
	{
		$$ = functype(N, $2, $4);
		funcnam($$, nil);
	}

Bfntype:
	'(' oarg_type_list ')' Bfnres
	{
		$$ = functype(N, $2, $4);
		funcnam($$, nil);
	}

fnlitdcl:
	fntype
	{
		markdclstack();	// save dcl stack and revert to block0
		$$ = $1;
		funcargs($$);
	}

fnliteral:
	LFUNC fnlitdcl '{' ostmt_list '}'
	{
		popdcl();

		vargen++;
		snprint(namebuf, sizeof(namebuf), "_f%.3ld", vargen);

		$$ = newname(lookup(namebuf));
		addvar($$, $2, PEXTERN);

		{
			Node *n;

			n = nod(ODCLFUNC, N, N);
			n->nname = $$;
			n->type = $2;
			n->nbody = $4;
			if(n->nbody == N)
				n->nbody = nod(ORETURN, N, N);
			compile(n);
		}

		$$ = nod(OADDR, $$, N);
	}

fnbody:
	'{' ostmt_list '}'
	{
		$$ = $2;
		if($$ == N)
			$$ = nod(ORETURN, N, N);
	}
|	';'
	{
		$$ = N;
	}

fnres:
	Afnres
|	Bfnres

Afnres:
	Atype
	{
		$$ = nod(ODCLFIELD, N, N);
		$$->type = $1;
		$$ = cleanidlist($$);
	}
|	'(' oarg_type_list ')'
	{
		$$ = $2;
	}

Bfnres:
	{
		$$ = N;
	}
|	Btype
	{
		$$ = nod(ODCLFIELD, N, N);
		$$->type = $1;
		$$ = cleanidlist($$);
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
		$$ = list($1, $2);
	}

vardcl_list_r:
	vardcl
|	vardcl_list_r ';' vardcl
	{
		$$ = nod(OLIST, $1, $3);
	}

constdcl_list_r:
	constdcl1
|	constdcl_list_r ';' constdcl1

typedcl_list_r:
	typedcl
|	typedcl_list_r ';' typedcl

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
|	new_name indcl
	{
		$$ = nod(ODCLFIELD, $1, N);
		$$->type = $2;
	}

indcl:
	'(' oarg_type_list ')' fnres
	{
		// without func keyword
		$$ = functype(fakethis(), $2, $4);
		funcnam($$, nil);
	}
|	latype
	{
		$$ = oldtype($1);
		if($$ == T || $$->etype != TFUNC)
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
|	non_type_new_name type
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

/*
 * need semi in front NO
 * need semi in back  NO
 */
Astmt:
	complex_stmt
|	compound_stmt
|	Acommon_dcl
|	';'
	{
		$$ = N;
	}
|	error Astmt
	{
		$$ = N;
	}

/*
 * need semi in front NO
 * need semi in back  YES
 */
Bstmt:
	semi_stmt
|	Bcommon_dcl
|	error Bstmt
	{
		$$ = N;
	}

/*
 * need semi in front YES
 * need semi in back  YES
 */
Cstmt:
	noninc_stmt

/*
 * need semi in front YES
 * need semi in back  NO
 */
Dstmt:
	inc_stmt
|	new_name ':'
	{
		$$ = nod(OLABEL, $1, N);
	}

/*
 * statement list that ends AorD
 */
Astmt_list_r:
	Astmt
|	Dstmt
|	Astmt_list_r Astmt
	{
		$$ = list($1, $2);
	}
|	Astmt_list_r Dstmt
	{
		$$ = list($1, $2);
	}
|	Bstmt_list_r Astmt
	{
		$$ = list($1, $2);
	}

/*
 * statement list that ends BorC
 */
Bstmt_list_r:
	Bstmt
|	Cstmt
|	Astmt_list_r Bstmt
	{
		$$ = list($1, $2);
	}
|	Astmt_list_r Cstmt
	{
		$$ = list($1, $2);
	}
|	Bstmt_list_r Bstmt
	{
		$$ = list($1, $2);
	}

stmt_list_r:
	Astmt_list_r
|	Bstmt_list_r

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

export:
	sym
	{
		exportsym($1);
	}
|	sym '.' sym2
	{
		exportsym(pkglookup($3->name, $1->name));
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

keyexpr_list:
	keyval_list_r
	{
		$$ = rev($1);
	}
|	oexpr_list

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
	/* leftover import ignored */
	LPACKAGE sym
	/* variables */
|	LVAR hidden_importsym hidden_importsym
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
|	LTYPE hidden_importsym '[' ']' hidden_importsym
	{
		// type array
		doimport2($2, nil, $5);
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
|	LTYPE hidden_importsym LLITERAL hidden_importsym
	{
		// type interface
		doimport8($2, &$3, $4);
	}
|	LFUNC sym1 hidden_importsym
	{
		// method
		doimport9($2, $3);
	}

isym:
	sym1 '.' sym2
	{
		$$ = nod(OIMPORT, N, N);
		$$->osym = $1;
		$$->psym = $1;
		$$->sym = $3;
		renamepkg($$);
	}
|	'(' sym1 ')' sym1 '.' sym2
	{
		$$ = nod(OIMPORT, N, N);
		$$->osym = $2;
		$$->psym = $4;
		$$->sym = $6;
		renamepkg($$);
	}

hidden_importsym:
	isym
|	'!' isym
	{
		$$ = $2;
		$$->etype = 1;
	}

hidden_importfield:
	sym1 isym
	{
		$$ = $2;
		$$->fsym = $1;
	}

/*
 * helpful error messages.
 * THIS SECTION MUST BE AT THE END OF THE FILE.
 *
 * these rules trigger reduce/reduce conflicts in the grammar.
 * they are safe because reduce/reduce conflicts are resolved
 * in favor of rules appearing earlier in the grammar, and these
 * are at the end of the file.
 *
 * to check whether the rest of the grammar is free of
 * reduce/reduce conflicts, comment this section out by
 * removing the slash on the next line.
 */
lpack:
	LATYPE
	{
		yyerror("%s is type, not package", $1->name);
		YYERROR;
	}

laconst:
	LPACK
	{
		// for LALR(1) reasons, using laconst works here
		// but lname does not.  even so, the messages make
		// more sense saying "var" instead of "const".
		yyerror("%s is package, not var", $1->name);
		YYERROR;
	}
|	LATYPE
	{
		yyerror("%s is type, not var", $1->name);
		YYERROR;
	}

latype:
	LACONST
	{
		yyerror("%s is const, not type", $1->name);
		YYERROR;
	}
|	LPACK
	{
		yyerror("%s is package, not type", $1->name);
		YYERROR;
	}
|	LNAME
	{
		yyerror("no type %s", $1->name);
		YYERROR;
	}
|	lpack '.' LNAME
	{
		yyerror("no type %s.%s", context, $3->name);
		YYERROR;
	}

/**/

