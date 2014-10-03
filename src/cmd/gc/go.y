// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Go language grammar.
 *
 * The Go semicolon rules are:
 *
 *  1. all statements and declarations are terminated by semicolons.
 *  2. semicolons can be omitted before a closing ) or }.
 *  3. semicolons are inserted by the lexer before a newline
 *      following a specific list of tokens.
 *
 * Rules #1 and #2 are accomplished by writing the lists as
 * semicolon-separated lists with an optional trailing semicolon.
 * Rule #3 is implemented in yylex.
 */

%{
#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and go.h re-#defines getc */
#include <libc.h>
#include "go.h"

static void fixlbrace(int);
%}
%union	{
	Node*		node;
	NodeList*		list;
	Type*		type;
	Sym*		sym;
	struct	Val	val;
	int		i;
}

// |sed 's/.*	//' |9 fmt -l1 |sort |9 fmt -l50 | sed 's/^/%xxx		/'

%token	<val>	LLITERAL
%token	<i>	LASOP LCOLAS
%token	<sym>	LBREAK LCASE LCHAN LCONST LCONTINUE LDDD
%token	<sym>	LDEFAULT LDEFER LELSE LFALL LFOR LFUNC LGO LGOTO
%token	<sym>	LIF LIMPORT LINTERFACE LMAP LNAME
%token	<sym>	LPACKAGE LRANGE LRETURN LSELECT LSTRUCT LSWITCH
%token	<sym>	LTYPE LVAR

%token		LANDAND LANDNOT LBODY LCOMM LDEC LEQ LGE LGT
%token		LIGNORE LINC LLE LLSH LLT LNE LOROR LRSH

%type	<i>	lbrace import_here
%type	<sym>	sym packname
%type	<val>	oliteral

%type	<node>	stmt ntype
%type	<node>	arg_type
%type	<node>	case caseblock
%type	<node>	compound_stmt dotname embed expr complitexpr bare_complitexpr
%type	<node>	expr_or_type
%type	<node>	fndcl hidden_fndcl fnliteral
%type	<node>	for_body for_header for_stmt if_header if_stmt non_dcl_stmt
%type	<node>	interfacedcl keyval labelname name
%type	<node>	name_or_type non_expr_type
%type	<node>	new_name dcl_name oexpr typedclname
%type	<node>	onew_name
%type	<node>	osimple_stmt pexpr pexpr_no_paren
%type	<node>	pseudocall range_stmt select_stmt
%type	<node>	simple_stmt
%type	<node>	switch_stmt uexpr
%type	<node>	xfndcl typedcl start_complit

%type	<list>	xdcl fnbody fnres loop_body dcl_name_list
%type	<list>	new_name_list expr_list keyval_list braced_keyval_list expr_or_type_list xdcl_list
%type	<list>	oexpr_list caseblock_list elseif elseif_list else stmt_list oarg_type_list_ocomma arg_type_list
%type	<list>	interfacedcl_list vardcl vardcl_list structdcl structdcl_list
%type	<list>	common_dcl constdcl constdcl1 constdcl_list typedcl_list

%type	<node>	convtype comptype dotdotdot
%type	<node>	indcl interfacetype structtype ptrtype
%type	<node>	recvchantype non_recvchantype othertype fnret_type fntype

%type	<sym>	hidden_importsym hidden_pkg_importsym

%type	<node>	hidden_constant hidden_literal hidden_funarg
%type	<node>	hidden_interfacedcl hidden_structdcl

%type	<list>	hidden_funres
%type	<list>	ohidden_funres
%type	<list>	hidden_funarg_list ohidden_funarg_list
%type	<list>	hidden_interfacedcl_list ohidden_interfacedcl_list
%type	<list>	hidden_structdcl_list ohidden_structdcl_list

%type	<type>	hidden_type hidden_type_misc hidden_pkgtype
%type	<type>	hidden_type_func
%type	<type>	hidden_type_recv_chan hidden_type_non_recv_chan

%left		LCOMM	/* outside the usual hierarchy; here for good error messages */

%left		LOROR
%left		LANDAND
%left		LEQ LNE LLE LGE LLT LGT
%left		'+' '-' '|' '^'
%left		'*' '/' '%' '&' LLSH LRSH LANDNOT

/*
 * manual override of shift/reduce conflicts.
 * the general form is that we assign a precedence
 * to the token being shifted and then introduce
 * NotToken with lower precedence or PreferToToken with higher
 * and annotate the reducing rule accordingly.
 */
%left		NotPackage
%left		LPACKAGE

%left		NotParen
%left		'('

%left		')'
%left		PreferToRightParen

%error-verbose

%%
file:
	loadsys
	package
	imports
	xdcl_list
	{
		xtop = concat(xtop, $4);
	}

package:
	%prec NotPackage
	{
		prevlineno = lineno;
		yyerror("package statement must be first");
		errorexit();
	}
|	LPACKAGE sym ';'
	{
		mkpackage($2->name);
	}

/*
 * this loads the definitions for the low-level runtime functions,
 * so that the compiler can generate calls to them,
 * but does not make the name "runtime" visible as a package.
 */
loadsys:
	{
		importpkg = runtimepkg;

		if(debug['A'])
			cannedimports("runtime.builtin", "package runtime\n\n$$\n\n");
		else
			cannedimports("runtime.builtin", runtimeimport);
		curio.importsafe = 1;
	}
	import_package
	import_there
	{
		importpkg = nil;
	}

imports:
|	imports import ';'

import:
	LIMPORT import_stmt
|	LIMPORT '(' import_stmt_list osemi ')'
|	LIMPORT '(' ')'

import_stmt:
	import_here import_package import_there
	{
		Pkg *ipkg;
		Sym *my;
		Node *pack;
		
		ipkg = importpkg;
		my = importmyname;
		importpkg = nil;
		importmyname = S;

		if(my == nil)
			my = lookup(ipkg->name);

		pack = nod(OPACK, N, N);
		pack->sym = my;
		pack->pkg = ipkg;
		pack->lineno = $1;

		if(my->name[0] == '.') {
			importdot(ipkg, pack);
			break;
		}
		if(strcmp(my->name, "init") == 0) {
			yyerror("cannot import package as init - init must be a func");
			break;
		}
		if(my->name[0] == '_' && my->name[1] == '\0')
			break;
		if(my->def) {
			lineno = $1;
			redeclare(my, "as imported package name");
		}
		my->def = pack;
		my->lastlineno = $1;
		my->block = 1;	// at top level
	}
|	import_here import_there
	{
		// When an invalid import path is passed to importfile,
		// it calls yyerror and then sets up a fake import with
		// no package statement. This allows us to test more
		// than one invalid import statement in a single file.
		if(nerrors == 0)
			fatal("phase error in import");
	}

import_stmt_list:
	import_stmt
|	import_stmt_list ';' import_stmt

import_here:
	LLITERAL
	{
		// import with original name
		$$ = parserline();
		importmyname = S;
		importfile(&$1, $$);
	}
|	sym LLITERAL
	{
		// import with given name
		$$ = parserline();
		importmyname = $1;
		importfile(&$2, $$);
	}
|	'.' LLITERAL
	{
		// import into my name space
		$$ = parserline();
		importmyname = lookup(".");
		importfile(&$2, $$);
	}

import_package:
	LPACKAGE LNAME import_safety ';'
	{
		if(importpkg->name == nil) {
			importpkg->name = $2->name;
			pkglookup($2->name, nil)->npkg++;
		} else if(strcmp(importpkg->name, $2->name) != 0)
			yyerror("conflicting names %s and %s for package \"%Z\"", importpkg->name, $2->name, importpkg->path);
		importpkg->direct = 1;
		importpkg->safe = curio.importsafe;

		if(safemode && !curio.importsafe)
			yyerror("cannot import unsafe package \"%Z\"", importpkg->path);
	}

import_safety:
|	LNAME
	{
		if(strcmp($1->name, "safe") == 0)
			curio.importsafe = 1;
	}

import_there:
	{
		defercheckwidth();
	}
	hidden_import_list '$' '$'
	{
		resumecheckwidth();
		unimportfile();
	}

/*
 * declarations
 */
xdcl:
	{
		yyerror("empty top-level declaration");
		$$ = nil;
	}
|	common_dcl
|	xfndcl
	{
		$$ = list1($1);
	}
|	non_dcl_stmt
	{
		yyerror("non-declaration statement outside function body");
		$$ = nil;
	}
|	error
	{
		$$ = nil;
	}

common_dcl:
	LVAR vardcl
	{
		$$ = $2;
	}
|	LVAR '(' vardcl_list osemi ')'
	{
		$$ = $3;
	}
|	LVAR '(' ')'
	{
		$$ = nil;
	}
|	lconst constdcl
	{
		$$ = $2;
		iota = -100000;
		lastconst = nil;
	}
|	lconst '(' constdcl osemi ')'
	{
		$$ = $3;
		iota = -100000;
		lastconst = nil;
	}
|	lconst '(' constdcl ';' constdcl_list osemi ')'
	{
		$$ = concat($3, $5);
		iota = -100000;
		lastconst = nil;
	}
|	lconst '(' ')'
	{
		$$ = nil;
		iota = -100000;
	}
|	LTYPE typedcl
	{
		$$ = list1($2);
	}
|	LTYPE '(' typedcl_list osemi ')'
	{
		$$ = $3;
	}
|	LTYPE '(' ')'
	{
		$$ = nil;
	}

lconst:
	LCONST
	{
		iota = 0;
	}

vardcl:
	dcl_name_list ntype
	{
		$$ = variter($1, $2, nil);
	}
|	dcl_name_list ntype '=' expr_list
	{
		$$ = variter($1, $2, $4);
	}
|	dcl_name_list '=' expr_list
	{
		$$ = variter($1, nil, $3);
	}

constdcl:
	dcl_name_list ntype '=' expr_list
	{
		$$ = constiter($1, $2, $4);
	}
|	dcl_name_list '=' expr_list
	{
		$$ = constiter($1, N, $3);
	}

constdcl1:
	constdcl
|	dcl_name_list ntype
	{
		$$ = constiter($1, $2, nil);
	}
|	dcl_name_list
	{
		$$ = constiter($1, N, nil);
	}

typedclname:
	sym
	{
		// different from dclname because the name
		// becomes visible right here, not at the end
		// of the declaration.
		$$ = typedcl0($1);
	}

typedcl:
	typedclname ntype
	{
		$$ = typedcl1($1, $2, 1);
	}

simple_stmt:
	expr
	{
		$$ = $1;

		// These nodes do not carry line numbers.
		// Since a bare name used as an expression is an error,
		// introduce a wrapper node to give the correct line.
		switch($$->op) {
		case ONAME:
		case ONONAME:
		case OTYPE:
		case OPACK:
		case OLITERAL:
			$$ = nod(OPAREN, $$, N);
			$$->implicit = 1;
			break;
		}
	}
|	expr LASOP expr
	{
		$$ = nod(OASOP, $1, $3);
		$$->etype = $2;			// rathole to pass opcode
	}
|	expr_list '=' expr_list
	{
		if($1->next == nil && $3->next == nil) {
			// simple
			$$ = nod(OAS, $1->n, $3->n);
			break;
		}
		// multiple
		$$ = nod(OAS2, N, N);
		$$->list = $1;
		$$->rlist = $3;
	}
|	expr_list LCOLAS expr_list
	{
		if($3->n->op == OTYPESW) {
			$$ = nod(OTYPESW, N, $3->n->right);
			if($3->next != nil)
				yyerror("expr.(type) must be alone in list");
			if($1->next != nil)
				yyerror("argument count mismatch: %d = %d", count($1), 1);
			else if(($1->n->op != ONAME && $1->n->op != OTYPE && $1->n->op != ONONAME) || isblank($1->n))
				yyerror("invalid variable name %N in type switch", $1->n);
			else
				$$->left = dclname($1->n->sym);  // it's a colas, so must not re-use an oldname.
			break;
		}
		$$ = colas($1, $3, $2);
	}
|	expr LINC
	{
		$$ = nod(OASOP, $1, nodintconst(1));
		$$->implicit = 1;
		$$->etype = OADD;
	}
|	expr LDEC
	{
		$$ = nod(OASOP, $1, nodintconst(1));
		$$->implicit = 1;
		$$->etype = OSUB;
	}

case:
	LCASE expr_or_type_list ':'
	{
		Node *n, *nn;

		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		$$ = nod(OXCASE, N, N);
		$$->list = $2;
		if(typesw != N && typesw->right != N && (n=typesw->right->left) != N) {
			// type switch - declare variable
			nn = newname(n->sym);
			declare(nn, dclcontext);
			$$->nname = nn;

			// keep track of the instances for reporting unused
			nn->defn = typesw->right;
		}
	}
|	LCASE expr_or_type_list '=' expr ':'
	{
		Node *n;

		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		$$ = nod(OXCASE, N, N);
		if($2->next == nil)
			n = nod(OAS, $2->n, $4);
		else {
			n = nod(OAS2, N, N);
			n->list = $2;
			n->rlist = list1($4);
		}
		$$->list = list1(n);
	}
|	LCASE expr_or_type_list LCOLAS expr ':'
	{
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		$$ = nod(OXCASE, N, N);
		$$->list = list1(colas($2, list1($4), $3));
	}
|	LDEFAULT ':'
	{
		Node *n, *nn;

		markdcl();
		$$ = nod(OXCASE, N, N);
		if(typesw != N && typesw->right != N && (n=typesw->right->left) != N) {
			// type switch - declare variable
			nn = newname(n->sym);
			declare(nn, dclcontext);
			$$->nname = nn;

			// keep track of the instances for reporting unused
			nn->defn = typesw->right;
		}
	}

compound_stmt:
	'{'
	{
		markdcl();
	}
	stmt_list '}'
	{
		if($3 == nil)
			$$ = nod(OEMPTY, N, N);
		else
			$$ = liststmt($3);
		popdcl();
	}

caseblock:
	case
	{
		// If the last token read by the lexer was consumed
		// as part of the case, clear it (parser has cleared yychar).
		// If the last token read by the lexer was the lookahead
		// leave it alone (parser has it cached in yychar).
		// This is so that the stmt_list action doesn't look at
		// the case tokens if the stmt_list is empty.
		yylast = yychar;
		$1->xoffset = block;
	}
	stmt_list
	{
		int last;

		// This is the only place in the language where a statement
		// list is not allowed to drop the final semicolon, because
		// it's the only place where a statement list is not followed 
		// by a closing brace.  Handle the error for pedantry.

		// Find the final token of the statement list.
		// yylast is lookahead; yyprev is last of stmt_list
		last = yyprev;

		if(last > 0 && last != ';' && yychar != '}')
			yyerror("missing statement after label");
		$$ = $1;
		$$->nbody = $3;
		popdcl();
	}

caseblock_list:
	{
		$$ = nil;
	}
|	caseblock_list caseblock
	{
		$$ = list($1, $2);
	}

loop_body:
	LBODY
	{
		markdcl();
	}
	stmt_list '}'
	{
		$$ = $3;
		popdcl();
	}

range_stmt:
	expr_list '=' LRANGE expr
	{
		$$ = nod(ORANGE, N, $4);
		$$->list = $1;
		$$->etype = 0;	// := flag
	}
|	expr_list LCOLAS LRANGE expr
	{
		$$ = nod(ORANGE, N, $4);
		$$->list = $1;
		$$->colas = 1;
		colasdefn($1, $$);
	}
|	LRANGE expr
	{
		$$ = nod(ORANGE, N, $2);
		$$->etype = 0; // := flag
	}

for_header:
	osimple_stmt ';' osimple_stmt ';' osimple_stmt
	{
		// init ; test ; incr
		if($5 != N && $5->colas != 0)
			yyerror("cannot declare in the for-increment");
		$$ = nod(OFOR, N, N);
		if($1 != N)
			$$->ninit = list1($1);
		$$->ntest = $3;
		$$->nincr = $5;
	}
|	osimple_stmt
	{
		// normal test
		$$ = nod(OFOR, N, N);
		$$->ntest = $1;
	}
|	range_stmt

for_body:
	for_header loop_body
	{
		$$ = $1;
		$$->nbody = concat($$->nbody, $2);
	}

for_stmt:
	LFOR
	{
		markdcl();
	}
	for_body
	{
		$$ = $3;
		popdcl();
	}

if_header:
	osimple_stmt
	{
		// test
		$$ = nod(OIF, N, N);
		$$->ntest = $1;
	}
|	osimple_stmt ';' osimple_stmt
	{
		// init ; test
		$$ = nod(OIF, N, N);
		if($1 != N)
			$$->ninit = list1($1);
		$$->ntest = $3;
	}

/* IF cond body (ELSE IF cond body)* (ELSE block)? */
if_stmt:
	LIF
	{
		markdcl();
	}
	if_header
	{
		if($3->ntest == N)
			yyerror("missing condition in if statement");
	}
	loop_body
	{
		$3->nbody = $5;
	}
	elseif_list else
	{
		Node *n;
		NodeList *nn;

		$$ = $3;
		n = $3;
		popdcl();
		for(nn = concat($7, $8); nn; nn = nn->next) {
			if(nn->n->op == OIF)
				popdcl();
			n->nelse = list1(nn->n);
			n = nn->n;
		}
	}

elseif:
	LELSE LIF 
	{
		markdcl();
	}
	if_header loop_body
	{
		if($4->ntest == N)
			yyerror("missing condition in if statement");
		$4->nbody = $5;
		$$ = list1($4);
	}

elseif_list:
	{
		$$ = nil;
	}
|	elseif_list elseif
	{
		$$ = concat($1, $2);
	}

else:
	{
		$$ = nil;
	}
|	LELSE compound_stmt
	{
		NodeList *node;
		
		node = mal(sizeof *node);
		node->n = $2;
		node->end = node;
		$$ = node;
	}

switch_stmt:
	LSWITCH
	{
		markdcl();
	}
	if_header
	{
		Node *n;
		n = $3->ntest;
		if(n != N && n->op != OTYPESW)
			n = N;
		typesw = nod(OXXX, typesw, n);
	}
	LBODY caseblock_list '}'
	{
		$$ = $3;
		$$->op = OSWITCH;
		$$->list = $6;
		typesw = typesw->left;
		popdcl();
	}

select_stmt:
	LSELECT
	{
		typesw = nod(OXXX, typesw, N);
	}
	LBODY caseblock_list '}'
	{
		$$ = nod(OSELECT, N, N);
		$$->lineno = typesw->lineno;
		$$->list = $4;
		typesw = typesw->left;
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
|	expr LANDNOT expr
	{
		$$ = nod(OANDNOT, $1, $3);
	}
|	expr LLSH expr
	{
		$$ = nod(OLSH, $1, $3);
	}
|	expr LRSH expr
	{
		$$ = nod(ORSH, $1, $3);
	}
	/* not an expression anymore, but left in so we can give a good error */
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
		if($2->op == OCOMPLIT) {
			// Special case for &T{...}: turn into (*T){...}.
			$$ = $2;
			$$->right = nod(OIND, $$->right, N);
			$$->right->implicit = 1;
		} else {
			$$ = nod(OADDR, $2, N);
		}
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
		yyerror("the bitwise complement operator is ^");
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

/*
 * call-like statements that
 * can be preceded by 'defer' and 'go'
 */
pseudocall:
	pexpr '(' ')'
	{
		$$ = nod(OCALL, $1, N);
	}
|	pexpr '(' expr_or_type_list ocomma ')'
	{
		$$ = nod(OCALL, $1, N);
		$$->list = $3;
	}
|	pexpr '(' expr_or_type_list LDDD ocomma ')'
	{
		$$ = nod(OCALL, $1, N);
		$$->list = $3;
		$$->isddd = 1;
	}

pexpr_no_paren:
	LLITERAL
	{
		$$ = nodlit($1);
	}
|	name
|	pexpr '.' sym
	{
		if($1->op == OPACK) {
			Sym *s;
			s = restrictlookup($3->name, $1->pkg);
			$1->used = 1;
			$$ = oldname(s);
			break;
		}
		$$ = nod(OXDOT, $1, newname($3));
	}
|	pexpr '.' '(' expr_or_type ')'
	{
		$$ = nod(ODOTTYPE, $1, $4);
	}
|	pexpr '.' '(' LTYPE ')'
	{
		$$ = nod(OTYPESW, N, $1);
	}
|	pexpr '[' expr ']'
	{
		$$ = nod(OINDEX, $1, $3);
	}
|	pexpr '[' oexpr ':' oexpr ']'
	{
		$$ = nod(OSLICE, $1, nod(OKEY, $3, $5));
	}
|	pexpr '[' oexpr ':' oexpr ':' oexpr ']'
	{
		if($5 == N)
			yyerror("middle index required in 3-index slice");
		if($7 == N)
			yyerror("final index required in 3-index slice");
		$$ = nod(OSLICE3, $1, nod(OKEY, $3, nod(OKEY, $5, $7)));
	}
|	pseudocall
|	convtype '(' expr ocomma ')'
	{
		// conversion
		$$ = nod(OCALL, $1, N);
		$$->list = list1($3);
	}
|	comptype lbrace start_complit braced_keyval_list '}'
	{
		$$ = $3;
		$$->right = $1;
		$$->list = $4;
		fixlbrace($2);
	}
|	pexpr_no_paren '{' start_complit braced_keyval_list '}'
	{
		$$ = $3;
		$$->right = $1;
		$$->list = $4;
	}
|	'(' expr_or_type ')' '{' start_complit braced_keyval_list '}'
	{
		yyerror("cannot parenthesize type in composite literal");
		$$ = $5;
		$$->right = $2;
		$$->list = $6;
	}
|	fnliteral

start_complit:
	{
		// composite expression.
		// make node early so we get the right line number.
		$$ = nod(OCOMPLIT, N, N);
	}

keyval:
	expr ':' complitexpr
	{
		$$ = nod(OKEY, $1, $3);
	}

bare_complitexpr:
	expr
	{
		// These nodes do not carry line numbers.
		// Since a composite literal commonly spans several lines,
		// the line number on errors may be misleading.
		// Introduce a wrapper node to give the correct line.
		$$ = $1;
		switch($$->op) {
		case ONAME:
		case ONONAME:
		case OTYPE:
		case OPACK:
		case OLITERAL:
			$$ = nod(OPAREN, $$, N);
			$$->implicit = 1;
		}
	}
|	'{' start_complit braced_keyval_list '}'
	{
		$$ = $2;
		$$->list = $3;
	}

complitexpr:
	expr
|	'{' start_complit braced_keyval_list '}'
	{
		$$ = $2;
		$$->list = $3;
	}

pexpr:
	pexpr_no_paren
|	'(' expr_or_type ')'
	{
		$$ = $2;
		
		// Need to know on lhs of := whether there are ( ).
		// Don't bother with the OPAREN in other cases:
		// it's just a waste of memory and time.
		switch($$->op) {
		case ONAME:
		case ONONAME:
		case OPACK:
		case OTYPE:
		case OLITERAL:
		case OTYPESW:
			$$ = nod(OPAREN, $$, N);
		}
	}

expr_or_type:
	expr
|	non_expr_type	%prec PreferToRightParen

name_or_type:
	ntype

lbrace:
	LBODY
	{
		$$ = LBODY;
	}
|	'{'
	{
		$$ = '{';
	}

/*
 * names and types
 *	newname is used before declared
 *	oldname is used after declared
 */
new_name:
	sym
	{
		if($1 == S)
			$$ = N;
		else
			$$ = newname($1);
	}

dcl_name:
	sym
	{
		$$ = dclname($1);
	}

onew_name:
	{
		$$ = N;
	}
|	new_name

sym:
	LNAME
	{
		$$ = $1;
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if(importpkg != nil && !exportname($1->name))
			$$ = pkglookup($1->name, builtinpkg);
	}
|	hidden_importsym
|	'?'
	{
		$$ = S;
	}

hidden_importsym:
	'@' LLITERAL '.' LNAME
	{
		Pkg *p;

		if($2.u.sval->len == 0)
			p = importpkg;
		else {
			if(isbadimport($2.u.sval))
				errorexit();
			p = mkpkg($2.u.sval);
		}
		$$ = pkglookup($4->name, p);
	}
|	'@' LLITERAL '.' '?'
	{
		Pkg *p;

		if($2.u.sval->len == 0)
			p = importpkg;
		else {
			if(isbadimport($2.u.sval))
				errorexit();
			p = mkpkg($2.u.sval);
		}
		$$ = pkglookup("?", p);
	}

name:
	sym	%prec NotParen
	{
		$$ = oldname($1);
		if($$->pack != N)
			$$->pack->used = 1;
	}

labelname:
	new_name

/*
 * to avoid parsing conflicts, type is split into
 *	channel types
 *	function types
 *	parenthesized types
 *	any other type
 * the type system makes additional restrictions,
 * but those are not implemented in the grammar.
 */
dotdotdot:
	LDDD
	{
		yyerror("final argument in variadic function missing type");
		$$ = nod(ODDD, typenod(typ(TINTER)), N);
	}
|	LDDD ntype
	{
		$$ = nod(ODDD, $2, N);
	}

ntype:
	recvchantype
|	fntype
|	othertype
|	ptrtype
|	dotname
|	'(' ntype ')'
	{
		$$ = $2;
	}

non_expr_type:
	recvchantype
|	fntype
|	othertype
|	'*' non_expr_type
	{
		$$ = nod(OIND, $2, N);
	}

non_recvchantype:
	fntype
|	othertype
|	ptrtype
|	dotname
|	'(' ntype ')'
	{
		$$ = $2;
	}

convtype:
	fntype
|	othertype

comptype:
	othertype

fnret_type:
	recvchantype
|	fntype
|	othertype
|	ptrtype
|	dotname

dotname:
	name
|	name '.' sym
	{
		if($1->op == OPACK) {
			Sym *s;
			s = restrictlookup($3->name, $1->pkg);
			$1->used = 1;
			$$ = oldname(s);
			break;
		}
		$$ = nod(OXDOT, $1, newname($3));
	}

othertype:
	'[' oexpr ']' ntype
	{
		$$ = nod(OTARRAY, $2, $4);
	}
|	'[' LDDD ']' ntype
	{
		// array literal of nelem
		$$ = nod(OTARRAY, nod(ODDD, N, N), $4);
	}
|	LCHAN non_recvchantype
	{
		$$ = nod(OTCHAN, $2, N);
		$$->etype = Cboth;
	}
|	LCHAN LCOMM ntype
	{
		$$ = nod(OTCHAN, $3, N);
		$$->etype = Csend;
	}
|	LMAP '[' ntype ']' ntype
	{
		$$ = nod(OTMAP, $3, $5);
	}
|	structtype
|	interfacetype

ptrtype:
	'*' ntype
	{
		$$ = nod(OIND, $2, N);
	}

recvchantype:
	LCOMM LCHAN ntype
	{
		$$ = nod(OTCHAN, $3, N);
		$$->etype = Crecv;
	}

structtype:
	LSTRUCT lbrace structdcl_list osemi '}'
	{
		$$ = nod(OTSTRUCT, N, N);
		$$->list = $3;
		fixlbrace($2);
	}
|	LSTRUCT lbrace '}'
	{
		$$ = nod(OTSTRUCT, N, N);
		fixlbrace($2);
	}

interfacetype:
	LINTERFACE lbrace interfacedcl_list osemi '}'
	{
		$$ = nod(OTINTER, N, N);
		$$->list = $3;
		fixlbrace($2);
	}
|	LINTERFACE lbrace '}'
	{
		$$ = nod(OTINTER, N, N);
		fixlbrace($2);
	}

/*
 * function stuff
 * all in one place to show how crappy it all is
 */
xfndcl:
	LFUNC fndcl fnbody
	{
		$$ = $2;
		if($$ == N)
			break;
		if(noescape && $3 != nil)
			yyerror("can only use //go:noescape with external func implementations");
		$$->nbody = $3;
		$$->endlineno = lineno;
		$$->noescape = noescape;
		$$->nosplit = nosplit;
		funcbody($$);
	}

fndcl:
	sym '(' oarg_type_list_ocomma ')' fnres
	{
		Node *t;

		$$ = N;
		$3 = checkarglist($3, 1);

		if(strcmp($1->name, "init") == 0) {
			$1 = renameinit();
			if($3 != nil || $5 != nil)
				yyerror("func init must have no arguments and no return values");
		}
		if(strcmp(localpkg->name, "main") == 0 && strcmp($1->name, "main") == 0) {
			if($3 != nil || $5 != nil)
				yyerror("func main must have no arguments and no return values");
		}

		t = nod(OTFUNC, N, N);
		t->list = $3;
		t->rlist = $5;

		$$ = nod(ODCLFUNC, N, N);
		$$->nname = newname($1);
		$$->nname->defn = $$;
		$$->nname->ntype = t;		// TODO: check if nname already has an ntype
		declare($$->nname, PFUNC);

		funchdr($$);
	}
|	'(' oarg_type_list_ocomma ')' sym '(' oarg_type_list_ocomma ')' fnres
	{
		Node *rcvr, *t;

		$$ = N;
		$2 = checkarglist($2, 0);
		$6 = checkarglist($6, 1);

		if($2 == nil) {
			yyerror("method has no receiver");
			break;
		}
		if($2->next != nil) {
			yyerror("method has multiple receivers");
			break;
		}
		rcvr = $2->n;
		if(rcvr->op != ODCLFIELD) {
			yyerror("bad receiver in method");
			break;
		}

		t = nod(OTFUNC, rcvr, N);
		t->list = $6;
		t->rlist = $8;

		$$ = nod(ODCLFUNC, N, N);
		$$->shortname = newname($4);
		$$->nname = methodname1($$->shortname, rcvr->right);
		$$->nname->defn = $$;
		$$->nname->ntype = t;
		$$->nname->nointerface = nointerface;
		declare($$->nname, PFUNC);

		funchdr($$);
	}

hidden_fndcl:
	hidden_pkg_importsym '(' ohidden_funarg_list ')' ohidden_funres
	{
		Sym *s;
		Type *t;

		$$ = N;

		s = $1;
		t = functype(N, $3, $5);

		importsym(s, ONAME);
		if(s->def != N && s->def->op == ONAME) {
			if(eqtype(t, s->def->type)) {
				dclcontext = PDISCARD;  // since we skip funchdr below
				break;
			}
			yyerror("inconsistent definition for func %S during import\n\t%T\n\t%T", s, s->def->type, t);
		}

		$$ = newname(s);
		$$->type = t;
		declare($$, PFUNC);

		funchdr($$);
	}
|	'(' hidden_funarg_list ')' sym '(' ohidden_funarg_list ')' ohidden_funres
	{
		$$ = methodname1(newname($4), $2->n->right); 
		$$->type = functype($2->n, $6, $8);

		checkwidth($$->type);
		addmethod($4, $$->type, 0, nointerface);
		nointerface = 0;
		funchdr($$);
		
		// inl.c's inlnode in on a dotmeth node expects to find the inlineable body as
		// (dotmeth's type)->nname->inl, and dotmeth's type has been pulled
		// out by typecheck's lookdot as this $$->ttype.  So by providing
		// this back link here we avoid special casing there.
		$$->type->nname = $$;
	}

fntype:
	LFUNC '(' oarg_type_list_ocomma ')' fnres
	{
		$3 = checkarglist($3, 1);
		$$ = nod(OTFUNC, N, N);
		$$->list = $3;
		$$->rlist = $5;
	}

fnbody:
	{
		$$ = nil;
	}
|	'{' stmt_list '}'
	{
		$$ = $2;
		if($$ == nil)
			$$ = list1(nod(OEMPTY, N, N));
	}

fnres:
	%prec NotParen
	{
		$$ = nil;
	}
|	fnret_type
	{
		$$ = list1(nod(ODCLFIELD, N, $1));
	}
|	'(' oarg_type_list_ocomma ')'
	{
		$2 = checkarglist($2, 0);
		$$ = $2;
	}

fnlitdcl:
	fntype
	{
		closurehdr($1);
	}

fnliteral:
	fnlitdcl lbrace stmt_list '}'
	{
		$$ = closurebody($3);
		fixlbrace($2);
	}
|	fnlitdcl error
	{
		$$ = closurebody(nil);
	}

/*
 * lists of things
 * note that they are left recursive
 * to conserve yacc stack. they need to
 * be reversed to interpret correctly
 */
xdcl_list:
	{
		$$ = nil;
	}
|	xdcl_list xdcl ';'
	{
		$$ = concat($1, $2);
		if(nsyntaxerrors == 0)
			testdclstack();
		nointerface = 0;
		noescape = 0;
		nosplit = 0;
	}

vardcl_list:
	vardcl
|	vardcl_list ';' vardcl
	{
		$$ = concat($1, $3);
	}

constdcl_list:
	constdcl1
|	constdcl_list ';' constdcl1
	{
		$$ = concat($1, $3);
	}

typedcl_list:
	typedcl
	{
		$$ = list1($1);
	}
|	typedcl_list ';' typedcl
	{
		$$ = list($1, $3);
	}

structdcl_list:
	structdcl
|	structdcl_list ';' structdcl
	{
		$$ = concat($1, $3);
	}

interfacedcl_list:
	interfacedcl
	{
		$$ = list1($1);
	}
|	interfacedcl_list ';' interfacedcl
	{
		$$ = list($1, $3);
	}

structdcl:
	new_name_list ntype oliteral
	{
		NodeList *l;

		Node *n;
		l = $1;
		if(l == nil) {
			// ? symbol, during import (list1(N) == nil)
			n = $2;
			if(n->op == OIND)
				n = n->left;
			n = embedded(n->sym, importpkg);
			n->right = $2;
			n->val = $3;
			$$ = list1(n);
			break;
		}

		for(l=$1; l; l=l->next) {
			l->n = nod(ODCLFIELD, l->n, $2);
			l->n->val = $3;
		}
	}
|	embed oliteral
	{
		$1->val = $2;
		$$ = list1($1);
	}
|	'(' embed ')' oliteral
	{
		$2->val = $4;
		$$ = list1($2);
		yyerror("cannot parenthesize embedded type");
	}
|	'*' embed oliteral
	{
		$2->right = nod(OIND, $2->right, N);
		$2->val = $3;
		$$ = list1($2);
	}
|	'(' '*' embed ')' oliteral
	{
		$3->right = nod(OIND, $3->right, N);
		$3->val = $5;
		$$ = list1($3);
		yyerror("cannot parenthesize embedded type");
	}
|	'*' '(' embed ')' oliteral
	{
		$3->right = nod(OIND, $3->right, N);
		$3->val = $5;
		$$ = list1($3);
		yyerror("cannot parenthesize embedded type");
	}

packname:
	LNAME
	{
		Node *n;

		$$ = $1;
		n = oldname($1);
		if(n->pack != N)
			n->pack->used = 1;
	}
|	LNAME '.' sym
	{
		Pkg *pkg;

		if($1->def == N || $1->def->op != OPACK) {
			yyerror("%S is not a package", $1);
			pkg = localpkg;
		} else {
			$1->def->used = 1;
			pkg = $1->def->pkg;
		}
		$$ = restrictlookup($3->name, pkg);
	}

embed:
	packname
	{
		$$ = embedded($1, localpkg);
	}

interfacedcl:
	new_name indcl
	{
		$$ = nod(ODCLFIELD, $1, $2);
		ifacedcl($$);
	}
|	packname
	{
		$$ = nod(ODCLFIELD, N, oldname($1));
	}
|	'(' packname ')'
	{
		$$ = nod(ODCLFIELD, N, oldname($2));
		yyerror("cannot parenthesize embedded type");
	}

indcl:
	'(' oarg_type_list_ocomma ')' fnres
	{
		// without func keyword
		$2 = checkarglist($2, 1);
		$$ = nod(OTFUNC, fakethis(), N);
		$$->list = $2;
		$$->rlist = $4;
	}

/*
 * function arguments.
 */
arg_type:
	name_or_type
|	sym name_or_type
	{
		$$ = nod(ONONAME, N, N);
		$$->sym = $1;
		$$ = nod(OKEY, $$, $2);
	}
|	sym dotdotdot
	{
		$$ = nod(ONONAME, N, N);
		$$->sym = $1;
		$$ = nod(OKEY, $$, $2);
	}
|	dotdotdot

arg_type_list:
	arg_type
	{
		$$ = list1($1);
	}
|	arg_type_list ',' arg_type
	{
		$$ = list($1, $3);
	}

oarg_type_list_ocomma:
	{
		$$ = nil;
	}
|	arg_type_list ocomma
	{
		$$ = $1;
	}

/*
 * statement
 */
stmt:
	{
		$$ = N;
	}
|	compound_stmt
|	common_dcl
	{
		$$ = liststmt($1);
	}
|	non_dcl_stmt
|	error
	{
		$$ = N;
	}

non_dcl_stmt:
	simple_stmt
|	for_stmt
|	switch_stmt
|	select_stmt
|	if_stmt
|	labelname ':'
	{
		$1 = nod(OLABEL, $1, N);
		$1->sym = dclstack;  // context, for goto restrictions
	}
	stmt
	{
		NodeList *l;

		$1->defn = $4;
		l = list1($1);
		if($4)
			l = list(l, $4);
		$$ = liststmt(l);
	}
|	LFALL
	{
		// will be converted to OFALL
		$$ = nod(OXFALL, N, N);
		$$->xoffset = block;
	}
|	LBREAK onew_name
	{
		$$ = nod(OBREAK, $2, N);
	}
|	LCONTINUE onew_name
	{
		$$ = nod(OCONTINUE, $2, N);
	}
|	LGO pseudocall
	{
		$$ = nod(OPROC, $2, N);
	}
|	LDEFER pseudocall
	{
		$$ = nod(ODEFER, $2, N);
	}
|	LGOTO new_name
	{
		$$ = nod(OGOTO, $2, N);
		$$->sym = dclstack;  // context, for goto restrictions
	}
|	LRETURN oexpr_list
	{
		$$ = nod(ORETURN, N, N);
		$$->list = $2;
		if($$->list == nil && curfn != N) {
			NodeList *l;

			for(l=curfn->dcl; l; l=l->next) {
				if(l->n->class == PPARAM)
					continue;
				if(l->n->class != PPARAMOUT)
					break;
				if(l->n->sym->def != l->n)
					yyerror("%s is shadowed during return", l->n->sym->name);
			}
		}
	}

stmt_list:
	stmt
	{
		$$ = nil;
		if($1 != N)
			$$ = list1($1);
	}
|	stmt_list ';' stmt
	{
		$$ = $1;
		if($3 != N)
			$$ = list($$, $3);
	}

new_name_list:
	new_name
	{
		$$ = list1($1);
	}
|	new_name_list ',' new_name
	{
		$$ = list($1, $3);
	}

dcl_name_list:
	dcl_name
	{
		$$ = list1($1);
	}
|	dcl_name_list ',' dcl_name
	{
		$$ = list($1, $3);
	}

expr_list:
	expr
	{
		$$ = list1($1);
	}
|	expr_list ',' expr
	{
		$$ = list($1, $3);
	}

expr_or_type_list:
	expr_or_type
	{
		$$ = list1($1);
	}
|	expr_or_type_list ',' expr_or_type
	{
		$$ = list($1, $3);
	}

/*
 * list of combo of keyval and val
 */
keyval_list:
	keyval
	{
		$$ = list1($1);
	}
|	bare_complitexpr
	{
		$$ = list1($1);
	}
|	keyval_list ',' keyval
	{
		$$ = list($1, $3);
	}
|	keyval_list ',' bare_complitexpr
	{
		$$ = list($1, $3);
	}

braced_keyval_list:
	{
		$$ = nil;
	}
|	keyval_list ocomma
	{
		$$ = $1;
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
		$$ = nil;
	}
|	expr_list

osimple_stmt:
	{
		$$ = N;
	}
|	simple_stmt

ohidden_funarg_list:
	{
		$$ = nil;
	}
|	hidden_funarg_list

ohidden_structdcl_list:
	{
		$$ = nil;
	}
|	hidden_structdcl_list

ohidden_interfacedcl_list:
	{
		$$ = nil;
	}
|	hidden_interfacedcl_list

oliteral:
	{
		$$.ctype = CTxxx;
	}
|	LLITERAL

/*
 * import syntax from package header
 */
hidden_import:
	LIMPORT LNAME LLITERAL ';'
	{
		importimport($2, $3.u.sval);
	}
|	LVAR hidden_pkg_importsym hidden_type ';'
	{
		importvar($2, $3);
	}
|	LCONST hidden_pkg_importsym '=' hidden_constant ';'
	{
		importconst($2, types[TIDEAL], $4);
	}
|	LCONST hidden_pkg_importsym hidden_type '=' hidden_constant ';'
	{
		importconst($2, $3, $5);
	}
|	LTYPE hidden_pkgtype hidden_type ';'
	{
		importtype($2, $3);
	}
|	LFUNC hidden_fndcl fnbody ';'
	{
		if($2 == N) {
			dclcontext = PEXTERN;  // since we skip the funcbody below
			break;
		}

		$2->inl = $3;

		funcbody($2);
		importlist = list(importlist, $2);

		if(debug['E']) {
			print("import [%Z] func %lN \n", importpkg->path, $2);
			if(debug['m'] > 2 && $2->inl)
				print("inl body:%+H\n", $2->inl);
		}
	}

hidden_pkg_importsym:
	hidden_importsym
	{
		$$ = $1;
		structpkg = $$->pkg;
	}

hidden_pkgtype:
	hidden_pkg_importsym
	{
		$$ = pkgtype($1);
		importsym($1, OTYPE);
	}

/*
 *  importing types
 */

hidden_type:
	hidden_type_misc
|	hidden_type_recv_chan
|	hidden_type_func

hidden_type_non_recv_chan:
	hidden_type_misc
|	hidden_type_func

hidden_type_misc:
	hidden_importsym
	{
		$$ = pkgtype($1);
	}
|	LNAME
	{
		// predefined name like uint8
		$1 = pkglookup($1->name, builtinpkg);
		if($1->def == N || $1->def->op != OTYPE) {
			yyerror("%s is not a type", $1->name);
			$$ = T;
		} else
			$$ = $1->def->type;
	}
|	'[' ']' hidden_type
	{
		$$ = aindex(N, $3);
	}
|	'[' LLITERAL ']' hidden_type
	{
		$$ = aindex(nodlit($2), $4);
	}
|	LMAP '[' hidden_type ']' hidden_type
	{
		$$ = maptype($3, $5);
	}
|	LSTRUCT '{' ohidden_structdcl_list '}'
	{
		$$ = tostruct($3);
	}
|	LINTERFACE '{' ohidden_interfacedcl_list '}'
	{
		$$ = tointerface($3);
	}
|	'*' hidden_type
	{
		$$ = ptrto($2);
	}
|	LCHAN hidden_type_non_recv_chan
	{
		$$ = typ(TCHAN);
		$$->type = $2;
		$$->chan = Cboth;
	}
|	LCHAN '(' hidden_type_recv_chan ')'
	{
		$$ = typ(TCHAN);
		$$->type = $3;
		$$->chan = Cboth;
	}
|	LCHAN LCOMM hidden_type
	{
		$$ = typ(TCHAN);
		$$->type = $3;
		$$->chan = Csend;
	}

hidden_type_recv_chan:
	LCOMM LCHAN hidden_type
	{
		$$ = typ(TCHAN);
		$$->type = $3;
		$$->chan = Crecv;
	}

hidden_type_func:
	LFUNC '(' ohidden_funarg_list ')' ohidden_funres
	{
		$$ = functype(nil, $3, $5);
	}

hidden_funarg:
	sym hidden_type oliteral
	{
		$$ = nod(ODCLFIELD, N, typenod($2));
		if($1)
			$$->left = newname($1);
		$$->val = $3;
	}
|	sym LDDD hidden_type oliteral
	{
		Type *t;
	
		t = typ(TARRAY);
		t->bound = -1;
		t->type = $3;

		$$ = nod(ODCLFIELD, N, typenod(t));
		if($1)
			$$->left = newname($1);
		$$->isddd = 1;
		$$->val = $4;
	}

hidden_structdcl:
	sym hidden_type oliteral
	{
		Sym *s;
		Pkg *p;

		if($1 != S && strcmp($1->name, "?") != 0) {
			$$ = nod(ODCLFIELD, newname($1), typenod($2));
			$$->val = $3;
		} else {
			s = $2->sym;
			if(s == S && isptr[$2->etype])
				s = $2->type->sym;
			p = importpkg;
			if($1 != S)
				p = $1->pkg;
			$$ = embedded(s, p);
			$$->right = typenod($2);
			$$->val = $3;
		}
	}

hidden_interfacedcl:
	sym '(' ohidden_funarg_list ')' ohidden_funres
	{
		$$ = nod(ODCLFIELD, newname($1), typenod(functype(fakethis(), $3, $5)));
	}
|	hidden_type
	{
		$$ = nod(ODCLFIELD, N, typenod($1));
	}

ohidden_funres:
	{
		$$ = nil;
	}
|	hidden_funres

hidden_funres:
	'(' ohidden_funarg_list ')'
	{
		$$ = $2;
	}
|	hidden_type
	{
		$$ = list1(nod(ODCLFIELD, N, typenod($1)));
	}

/*
 *  importing constants
 */

hidden_literal:
	LLITERAL
	{
		$$ = nodlit($1);
	}
|	'-' LLITERAL
	{
		$$ = nodlit($2);
		switch($$->val.ctype){
		case CTINT:
		case CTRUNE:
			mpnegfix($$->val.u.xval);
			break;
		case CTFLT:
			mpnegflt($$->val.u.fval);
			break;
		case CTCPLX:
			mpnegflt(&$$->val.u.cval->real);
			mpnegflt(&$$->val.u.cval->imag);
			break;
		default:
			yyerror("bad negated constant");
		}
	}
|	sym
	{
		$$ = oldname(pkglookup($1->name, builtinpkg));
		if($$->op != OLITERAL)
			yyerror("bad constant %S", $$->sym);
	}

hidden_constant:
	hidden_literal
|	'(' hidden_literal '+' hidden_literal ')'
	{
		if($2->val.ctype == CTRUNE && $4->val.ctype == CTINT) {
			$$ = $2;
			mpaddfixfix($2->val.u.xval, $4->val.u.xval, 0);
			break;
		}
		$4->val.u.cval->real = $4->val.u.cval->imag;
		mpmovecflt(&$4->val.u.cval->imag, 0.0);
		$$ = nodcplxlit($2->val, $4->val);
	}

hidden_import_list:
|	hidden_import_list hidden_import

hidden_funarg_list:
	hidden_funarg
	{
		$$ = list1($1);
	}
|	hidden_funarg_list ',' hidden_funarg
	{
		$$ = list($1, $3);
	}

hidden_structdcl_list:
	hidden_structdcl
	{
		$$ = list1($1);
	}
|	hidden_structdcl_list ';' hidden_structdcl
	{
		$$ = list($1, $3);
	}

hidden_interfacedcl_list:
	hidden_interfacedcl
	{
		$$ = list1($1);
	}
|	hidden_interfacedcl_list ';' hidden_interfacedcl
	{
		$$ = list($1, $3);
	}

%%

static void
fixlbrace(int lbr)
{
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.c about loophack.
	if(lbr == LBODY)
		loophack = 1;
}

