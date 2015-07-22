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
package gc

import (
	"fmt"
	"strings"
)
%}
%union	{
	node *Node
	list *NodeList
	typ *Type
	sym *Sym
	val Val
	i int
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

%type	<typ>	hidden_type hidden_type_misc hidden_pkgtype
%type	<typ>	hidden_type_func
%type	<typ>	hidden_type_recv_chan hidden_type_non_recv_chan

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

%error loadsys package LIMPORT '(' LLITERAL import_package import_there ',':
	"unexpected comma during import block"

%error loadsys package LIMPORT LNAME ';':
	"missing import path; require quoted string"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LIF if_header ';':
	"missing { after if clause"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LSWITCH if_header ';':
	"missing { after switch clause"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LFOR for_header ';':
	"missing { after for clause"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LFOR ';' LBODY:
	"missing { after for clause"

%error loadsys package imports LFUNC LNAME '(' ')' ';' '{':
	"unexpected semicolon or newline before {"

%error loadsys package imports LTYPE LNAME ';':
	"unexpected semicolon or newline in type declaration"

%error loadsys package imports LCHAN '}':
	"unexpected } in channel type"

%error loadsys package imports LCHAN ')':
	"unexpected ) in channel type"

%error loadsys package imports LCHAN ',':
	"unexpected comma in channel type"

%error loadsys package imports LFUNC LNAME '(' ')' '{' if_stmt ';' LELSE:
	"unexpected semicolon or newline before else"

%error loadsys package imports LTYPE LNAME LINTERFACE '{' LNAME ',' LNAME:
	"name list not allowed in interface type"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LFOR LVAR LNAME '=' LNAME:
	"var declaration not allowed in for initializer"

%error loadsys package imports LVAR LNAME '[' ']' LNAME '{':
	"unexpected { at end of statement"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LVAR LNAME '[' ']' LNAME '{':
	"unexpected { at end of statement"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LDEFER LNAME ';':
	"argument to go/defer must be function call"

%error loadsys package imports LVAR LNAME '=' LNAME '{' LNAME ';':
	"need trailing comma before newline in composite literal"

%error loadsys package imports LVAR LNAME '=' comptype '{' LNAME ';':
	"need trailing comma before newline in composite literal"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LFUNC LNAME:
	"nested func not allowed"

%error loadsys package imports LFUNC LNAME '(' ')' '{' LIF if_header loop_body LELSE ';':
	"else must be followed by if or statement block"

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
		Yyerror("package statement must be first");
		errorexit();
	}
|	LPACKAGE sym ';'
	{
		mkpackage($2.Name);
	}

/*
 * this loads the definitions for the low-level runtime functions,
 * so that the compiler can generate calls to them,
 * but does not make the name "runtime" visible as a package.
 */
loadsys:
	{
		importpkg = Runtimepkg;

		if Debug['A'] != 0 {
			cannedimports("runtime.Builtin", "package runtime\n\n$$\n\n");
		} else {
			cannedimports("runtime.Builtin", runtimeimport);
		}
		curio.importsafe = true
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
		ipkg := importpkg;
		my := importmyname;
		importpkg = nil;
		importmyname = nil;

		if my == nil {
			my = Lookup(ipkg.Name);
		}

		pack := Nod(OPACK, nil, nil);
		pack.Sym = my;
		pack.Name.Pkg = ipkg;
		pack.Lineno = int32($1);

		if strings.HasPrefix(my.Name, ".") {
			importdot(ipkg, pack);
			break;
		}
		if my.Name == "init" {
			Yyerror("cannot import package as init - init must be a func");
			break;
		}
		if my.Name == "_" {
			break;
		}
		if my.Def != nil {
			lineno = int32($1);
			redeclare(my, "as imported package name");
		}
		my.Def = pack;
		my.Lastlineno = int32($1);
		my.Block = 1;	// at top level
	}
|	import_here import_there
	{
		// When an invalid import path is passed to importfile,
		// it calls Yyerror and then sets up a fake import with
		// no package statement. This allows us to test more
		// than one invalid import statement in a single file.
		if nerrors == 0 {
			Fatal("phase error in import");
		}
	}

import_stmt_list:
	import_stmt
|	import_stmt_list ';' import_stmt

import_here:
	LLITERAL
	{
		// import with original name
		$$ = parserline();
		importmyname = nil;
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
		importmyname = Lookup(".");
		importfile(&$2, $$);
	}

import_package:
	LPACKAGE LNAME import_safety ';'
	{
		if importpkg.Name == "" {
			importpkg.Name = $2.Name;
			numImport[$2.Name]++
		} else if importpkg.Name != $2.Name {
			Yyerror("conflicting names %s and %s for package %q", importpkg.Name, $2.Name, importpkg.Path);
		}
		importpkg.Direct = 1;
		importpkg.Safe = curio.importsafe

		if safemode != 0 && !curio.importsafe {
			Yyerror("cannot import unsafe package %q", importpkg.Path);
		}
	}

import_safety:
|	LNAME
	{
		if $1.Name == "safe" {
			curio.importsafe = true
		}
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
		Yyerror("empty top-level declaration");
		$$ = nil;
	}
|	common_dcl
|	xfndcl
	{
		$$ = list1($1);
	}
|	non_dcl_stmt
	{
		Yyerror("non-declaration statement outside function body");
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
		iota_ = -100000;
		lastconst = nil;
	}
|	lconst '(' constdcl osemi ')'
	{
		$$ = $3;
		iota_ = -100000;
		lastconst = nil;
	}
|	lconst '(' constdcl ';' constdcl_list osemi ')'
	{
		$$ = concat($3, $5);
		iota_ = -100000;
		lastconst = nil;
	}
|	lconst '(' ')'
	{
		$$ = nil;
		iota_ = -100000;
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
		iota_ = 0;
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
		$$ = constiter($1, nil, $3);
	}

constdcl1:
	constdcl
|	dcl_name_list ntype
	{
		$$ = constiter($1, $2, nil);
	}
|	dcl_name_list
	{
		$$ = constiter($1, nil, nil);
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
		$$ = typedcl1($1, $2, true);
	}

simple_stmt:
	expr
	{
		$$ = $1;

		// These nodes do not carry line numbers.
		// Since a bare name used as an expression is an error,
		// introduce a wrapper node to give the correct line.
		switch($$.Op) {
		case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
			$$ = Nod(OPAREN, $$, nil);
			$$.Implicit = true;
			break;
		}
	}
|	expr LASOP expr
	{
		$$ = Nod(OASOP, $1, $3);
		$$.Etype = uint8($2);			// rathole to pass opcode
	}
|	expr_list '=' expr_list
	{
		if $1.Next == nil && $3.Next == nil {
			// simple
			$$ = Nod(OAS, $1.N, $3.N);
			break;
		}
		// multiple
		$$ = Nod(OAS2, nil, nil);
		$$.List = $1;
		$$.Rlist = $3;
	}
|	expr_list LCOLAS expr_list
	{
		if $3.N.Op == OTYPESW {
			$$ = Nod(OTYPESW, nil, $3.N.Right);
			if $3.Next != nil {
				Yyerror("expr.(type) must be alone in list");
			}
			if $1.Next != nil {
				Yyerror("argument count mismatch: %d = %d", count($1), 1);
			} else if ($1.N.Op != ONAME && $1.N.Op != OTYPE && $1.N.Op != ONONAME) || isblank($1.N) {
				Yyerror("invalid variable name %s in type switch", $1.N);
			} else {
				$$.Left = dclname($1.N.Sym);
			}  // it's a colas, so must not re-use an oldname.
			break;
		}
		$$ = colas($1, $3, int32($2));
	}
|	expr LINC
	{
		$$ = Nod(OASOP, $1, Nodintconst(1));
		$$.Implicit = true;
		$$.Etype = OADD;
	}
|	expr LDEC
	{
		$$ = Nod(OASOP, $1, Nodintconst(1));
		$$.Implicit = true;
		$$.Etype = OSUB;
	}

case:
	LCASE expr_or_type_list ':'
	{
		var n, nn *Node

		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		$$ = Nod(OXCASE, nil, nil);
		$$.List = $2;
		if typesw != nil && typesw.Right != nil {
			n = typesw.Right.Left
			if n != nil {
				// type switch - declare variable
				nn = newname(n.Sym);
				declare(nn, dclcontext);
				$$.Rlist = list1(nn);
	
				// keep track of the instances for reporting unused
				nn.Name.Defn = typesw.Right;
			}
		}
	}
|	LCASE expr_or_type_list '=' expr ':'
	{
		var n *Node

		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		$$ = Nod(OXCASE, nil, nil);
		if $2.Next == nil {
			n = Nod(OAS, $2.N, $4);
		} else {
			n = Nod(OAS2, nil, nil);
			n.List = $2;
			n.Rlist = list1($4);
		}
		$$.List = list1(n);
	}
|	LCASE expr_or_type_list LCOLAS expr ':'
	{
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		$$ = Nod(OXCASE, nil, nil);
		$$.List = list1(colas($2, list1($4), int32($3)));
	}
|	LDEFAULT ':'
	{
		var n, nn *Node

		markdcl();
		$$ = Nod(OXCASE, nil, nil);
		if typesw != nil && typesw.Right != nil {
			n = typesw.Right.Left
			if n != nil {
				// type switch - declare variable
				nn = newname(n.Sym);
				declare(nn, dclcontext);
				$$.Rlist = list1(nn);
	
				// keep track of the instances for reporting unused
				nn.Name.Defn = typesw.Right;
			}
		}
	}

compound_stmt:
	'{'
	{
		markdcl();
	}
	stmt_list '}'
	{
		if $3 == nil {
			$$ = Nod(OEMPTY, nil, nil);
		} else {
			$$ = liststmt($3);
		}
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
		$1.Xoffset = int64(block);
	}
	stmt_list
	{
		// This is the only place in the language where a statement
		// list is not allowed to drop the final semicolon, because
		// it's the only place where a statement list is not followed 
		// by a closing brace.  Handle the error for pedantry.

		// Find the final token of the statement list.
		// yylast is lookahead; yyprev is last of stmt_list
		last := yyprev;

		if last > 0 && last != ';' && yychar != '}' {
			Yyerror("missing statement after label");
		}
		$$ = $1;
		$$.Nbody = $3;
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
		$$ = Nod(ORANGE, nil, $4);
		$$.List = $1;
		$$.Etype = 0;	// := flag
	}
|	expr_list LCOLAS LRANGE expr
	{
		$$ = Nod(ORANGE, nil, $4);
		$$.List = $1;
		$$.Colas = true;
		colasdefn($1, $$);
	}
|	LRANGE expr
	{
		$$ = Nod(ORANGE, nil, $2);
		$$.Etype = 0; // := flag
	}

for_header:
	osimple_stmt ';' osimple_stmt ';' osimple_stmt
	{
		// init ; test ; incr
		if $5 != nil && $5.Colas {
			Yyerror("cannot declare in the for-increment");
		}
		$$ = Nod(OFOR, nil, nil);
		if $1 != nil {
			$$.Ninit = list1($1);
		}
		$$.Left = $3;
		$$.Right = $5;
	}
|	osimple_stmt
	{
		// normal test
		$$ = Nod(OFOR, nil, nil);
		$$.Left = $1;
	}
|	range_stmt

for_body:
	for_header loop_body
	{
		$$ = $1;
		$$.Nbody = concat($$.Nbody, $2);
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
		$$ = Nod(OIF, nil, nil);
		$$.Left = $1;
	}
|	osimple_stmt ';' osimple_stmt
	{
		// init ; test
		$$ = Nod(OIF, nil, nil);
		if $1 != nil {
			$$.Ninit = list1($1);
		}
		$$.Left = $3;
	}

/* IF cond body (ELSE IF cond body)* (ELSE block)? */
if_stmt:
	LIF
	{
		markdcl();
	}
	if_header
	{
		if $3.Left == nil {
			Yyerror("missing condition in if statement");
		}
	}
	loop_body
	{
		$3.Nbody = $5;
	}
	elseif_list else
	{
		var n *Node
		var nn *NodeList

		$$ = $3;
		n = $3;
		popdcl();
		for nn = concat($7, $8); nn != nil; nn = nn.Next {
			if nn.N.Op == OIF {
				popdcl();
			}
			n.Rlist = list1(nn.N);
			n = nn.N;
		}
	}

elseif:
	LELSE LIF 
	{
		markdcl();
	}
	if_header loop_body
	{
		if $4.Left == nil {
			Yyerror("missing condition in if statement");
		}
		$4.Nbody = $5;
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
		l := &NodeList{N: $2}
		l.End = l
		$$ = l;
	}

switch_stmt:
	LSWITCH
	{
		markdcl();
	}
	if_header
	{
		var n *Node
		n = $3.Left;
		if n != nil && n.Op != OTYPESW {
			n = nil;
		}
		typesw = Nod(OXXX, typesw, n);
	}
	LBODY caseblock_list '}'
	{
		$$ = $3;
		$$.Op = OSWITCH;
		$$.List = $6;
		typesw = typesw.Left;
		popdcl();
	}

select_stmt:
	LSELECT
	{
		typesw = Nod(OXXX, typesw, nil);
	}
	LBODY caseblock_list '}'
	{
		$$ = Nod(OSELECT, nil, nil);
		$$.Lineno = typesw.Lineno;
		$$.List = $4;
		typesw = typesw.Left;
	}

/*
 * expressions
 */
expr:
	uexpr
|	expr LOROR expr
	{
		$$ = Nod(OOROR, $1, $3);
	}
|	expr LANDAND expr
	{
		$$ = Nod(OANDAND, $1, $3);
	}
|	expr LEQ expr
	{
		$$ = Nod(OEQ, $1, $3);
	}
|	expr LNE expr
	{
		$$ = Nod(ONE, $1, $3);
	}
|	expr LLT expr
	{
		$$ = Nod(OLT, $1, $3);
	}
|	expr LLE expr
	{
		$$ = Nod(OLE, $1, $3);
	}
|	expr LGE expr
	{
		$$ = Nod(OGE, $1, $3);
	}
|	expr LGT expr
	{
		$$ = Nod(OGT, $1, $3);
	}
|	expr '+' expr
	{
		$$ = Nod(OADD, $1, $3);
	}
|	expr '-' expr
	{
		$$ = Nod(OSUB, $1, $3);
	}
|	expr '|' expr
	{
		$$ = Nod(OOR, $1, $3);
	}
|	expr '^' expr
	{
		$$ = Nod(OXOR, $1, $3);
	}
|	expr '*' expr
	{
		$$ = Nod(OMUL, $1, $3);
	}
|	expr '/' expr
	{
		$$ = Nod(ODIV, $1, $3);
	}
|	expr '%' expr
	{
		$$ = Nod(OMOD, $1, $3);
	}
|	expr '&' expr
	{
		$$ = Nod(OAND, $1, $3);
	}
|	expr LANDNOT expr
	{
		$$ = Nod(OANDNOT, $1, $3);
	}
|	expr LLSH expr
	{
		$$ = Nod(OLSH, $1, $3);
	}
|	expr LRSH expr
	{
		$$ = Nod(ORSH, $1, $3);
	}
	/* not an expression anymore, but left in so we can give a good error */
|	expr LCOMM expr
	{
		$$ = Nod(OSEND, $1, $3);
	}

uexpr:
	pexpr
|	'*' uexpr
	{
		$$ = Nod(OIND, $2, nil);
	}
|	'&' uexpr
	{
		if $2.Op == OCOMPLIT {
			// Special case for &T{...}: turn into (*T){...}.
			$$ = $2;
			$$.Right = Nod(OIND, $$.Right, nil);
			$$.Right.Implicit = true;
		} else {
			$$ = Nod(OADDR, $2, nil);
		}
	}
|	'+' uexpr
	{
		$$ = Nod(OPLUS, $2, nil);
	}
|	'-' uexpr
	{
		$$ = Nod(OMINUS, $2, nil);
	}
|	'!' uexpr
	{
		$$ = Nod(ONOT, $2, nil);
	}
|	'~' uexpr
	{
		Yyerror("the bitwise complement operator is ^");
		$$ = Nod(OCOM, $2, nil);
	}
|	'^' uexpr
	{
		$$ = Nod(OCOM, $2, nil);
	}
|	LCOMM uexpr
	{
		$$ = Nod(ORECV, $2, nil);
	}

/*
 * call-like statements that
 * can be preceded by 'defer' and 'go'
 */
pseudocall:
	pexpr '(' ')'
	{
		$$ = Nod(OCALL, $1, nil);
	}
|	pexpr '(' expr_or_type_list ocomma ')'
	{
		$$ = Nod(OCALL, $1, nil);
		$$.List = $3;
	}
|	pexpr '(' expr_or_type_list LDDD ocomma ')'
	{
		$$ = Nod(OCALL, $1, nil);
		$$.List = $3;
		$$.Isddd = true;
	}

pexpr_no_paren:
	LLITERAL
	{
		$$ = nodlit($1);
	}
|	name
|	pexpr '.' sym
	{
		if $1.Op == OPACK {
			var s *Sym
			s = restrictlookup($3.Name, $1.Name.Pkg);
			$1.Used = true;
			$$ = oldname(s);
			break;
		}
		$$ = Nod(OXDOT, $1, newname($3));
	}
|	pexpr '.' '(' expr_or_type ')'
	{
		$$ = Nod(ODOTTYPE, $1, $4);
	}
|	pexpr '.' '(' LTYPE ')'
	{
		$$ = Nod(OTYPESW, nil, $1);
	}
|	pexpr '[' expr ']'
	{
		$$ = Nod(OINDEX, $1, $3);
	}
|	pexpr '[' oexpr ':' oexpr ']'
	{
		$$ = Nod(OSLICE, $1, Nod(OKEY, $3, $5));
	}
|	pexpr '[' oexpr ':' oexpr ':' oexpr ']'
	{
		if $5 == nil {
			Yyerror("middle index required in 3-index slice");
		}
		if $7 == nil {
			Yyerror("final index required in 3-index slice");
		}
		$$ = Nod(OSLICE3, $1, Nod(OKEY, $3, Nod(OKEY, $5, $7)));
	}
|	pseudocall
|	convtype '(' expr ocomma ')'
	{
		// conversion
		$$ = Nod(OCALL, $1, nil);
		$$.List = list1($3);
	}
|	comptype lbrace start_complit braced_keyval_list '}'
	{
		$$ = $3;
		$$.Right = $1;
		$$.List = $4;
		fixlbrace($2);
	}
|	pexpr_no_paren '{' start_complit braced_keyval_list '}'
	{
		$$ = $3;
		$$.Right = $1;
		$$.List = $4;
	}
|	'(' expr_or_type ')' '{' start_complit braced_keyval_list '}'
	{
		Yyerror("cannot parenthesize type in composite literal");
		$$ = $5;
		$$.Right = $2;
		$$.List = $6;
	}
|	fnliteral

start_complit:
	{
		// composite expression.
		// make node early so we get the right line number.
		$$ = Nod(OCOMPLIT, nil, nil);
	}

keyval:
	complitexpr ':' complitexpr
	{
		$$ = Nod(OKEY, $1, $3);
	}

bare_complitexpr:
	expr
	{
		// These nodes do not carry line numbers.
		// Since a composite literal commonly spans several lines,
		// the line number on errors may be misleading.
		// Introduce a wrapper node to give the correct line.
		$$ = $1;
		switch($$.Op) {
		case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
			$$ = Nod(OPAREN, $$, nil);
			$$.Implicit = true;
		}
	}
|	'{' start_complit braced_keyval_list '}'
	{
		$$ = $2;
		$$.List = $3;
	}

complitexpr:
	expr
|	'{' start_complit braced_keyval_list '}'
	{
		$$ = $2;
		$$.List = $3;
	}

pexpr:
	pexpr_no_paren
|	'(' expr_or_type ')'
	{
		$$ = $2;
		
		// Need to know on lhs of := whether there are ( ).
		// Don't bother with the OPAREN in other cases:
		// it's just a waste of memory and time.
		switch($$.Op) {
		case ONAME, ONONAME, OPACK, OTYPE, OLITERAL, OTYPESW:
			$$ = Nod(OPAREN, $$, nil);
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
		if $1 == nil {
			$$ = nil;
		} else {
			$$ = newname($1);
		}
	}

dcl_name:
	sym
	{
		$$ = dclname($1);
	}

onew_name:
	{
		$$ = nil;
	}
|	new_name

sym:
	LNAME
	{
		$$ = $1;
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if importpkg != nil && !exportname($1.Name) {
			$$ = Pkglookup($1.Name, builtinpkg);
		}
	}
|	hidden_importsym
|	'?'
	{
		$$ = nil;
	}

hidden_importsym:
	'@' LLITERAL '.' LNAME
	{
		var p *Pkg

		if $2.U.(string) == "" {
			p = importpkg;
		} else {
			if isbadimport($2.U.(string)) {
				errorexit();
			}
			p = mkpkg($2.U.(string));
		}
		$$ = Pkglookup($4.Name, p);
	}
|	'@' LLITERAL '.' '?'
	{
		var p *Pkg

		if $2.U.(string) == "" {
			p = importpkg;
		} else {
			if isbadimport($2.U.(string)) {
				errorexit();
			}
			p = mkpkg($2.U.(string));
		}
		$$ = Pkglookup("?", p);
	}

name:
	sym	%prec NotParen
	{
		$$ = oldname($1);
		if $$.Name != nil && $$.Name.Pack != nil {
			$$.Name.Pack.Used = true;
		}
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
		Yyerror("final argument in variadic function missing type");
		$$ = Nod(ODDD, typenod(typ(TINTER)), nil);
	}
|	LDDD ntype
	{
		$$ = Nod(ODDD, $2, nil);
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
		$$ = Nod(OIND, $2, nil);
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
		if $1.Op == OPACK {
			var s *Sym
			s = restrictlookup($3.Name, $1.Name.Pkg);
			$1.Used = true;
			$$ = oldname(s);
			break;
		}
		$$ = Nod(OXDOT, $1, newname($3));
	}

othertype:
	'[' oexpr ']' ntype
	{
		$$ = Nod(OTARRAY, $2, $4);
	}
|	'[' LDDD ']' ntype
	{
		// array literal of nelem
		$$ = Nod(OTARRAY, Nod(ODDD, nil, nil), $4);
	}
|	LCHAN non_recvchantype
	{
		$$ = Nod(OTCHAN, $2, nil);
		$$.Etype = Cboth;
	}
|	LCHAN LCOMM ntype
	{
		$$ = Nod(OTCHAN, $3, nil);
		$$.Etype = Csend;
	}
|	LMAP '[' ntype ']' ntype
	{
		$$ = Nod(OTMAP, $3, $5);
	}
|	structtype
|	interfacetype

ptrtype:
	'*' ntype
	{
		$$ = Nod(OIND, $2, nil);
	}

recvchantype:
	LCOMM LCHAN ntype
	{
		$$ = Nod(OTCHAN, $3, nil);
		$$.Etype = Crecv;
	}

structtype:
	LSTRUCT lbrace structdcl_list osemi '}'
	{
		$$ = Nod(OTSTRUCT, nil, nil);
		$$.List = $3;
		fixlbrace($2);
	}
|	LSTRUCT lbrace '}'
	{
		$$ = Nod(OTSTRUCT, nil, nil);
		fixlbrace($2);
	}

interfacetype:
	LINTERFACE lbrace interfacedcl_list osemi '}'
	{
		$$ = Nod(OTINTER, nil, nil);
		$$.List = $3;
		fixlbrace($2);
	}
|	LINTERFACE lbrace '}'
	{
		$$ = Nod(OTINTER, nil, nil);
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
		if $$ == nil {
			break;
		}
		if noescape && $3 != nil {
			Yyerror("can only use //go:noescape with external func implementations");
		}
		$$.Nbody = $3;
		$$.Func.Endlineno = lineno;
		$$.Noescape = noescape;
		$$.Func.Norace = norace;
		$$.Func.Nosplit = nosplit;
		$$.Func.Nowritebarrier = nowritebarrier;
		$$.Func.Systemstack = systemstack;
		funcbody($$);
	}

fndcl:
	sym '(' oarg_type_list_ocomma ')' fnres
	{
		var t *Node

		$$ = nil;
		$3 = checkarglist($3, 1);

		if $1.Name == "init" {
			$1 = renameinit();
			if $3 != nil || $5 != nil {
				Yyerror("func init must have no arguments and no return values");
			}
		}
		if localpkg.Name == "main" && $1.Name == "main" {
			if $3 != nil || $5 != nil {
				Yyerror("func main must have no arguments and no return values");
			}
		}

		t = Nod(OTFUNC, nil, nil);
		t.List = $3;
		t.Rlist = $5;

		$$ = Nod(ODCLFUNC, nil, nil);
		$$.Func.Nname = newfuncname($1);
		$$.Func.Nname.Name.Defn = $$;
		$$.Func.Nname.Name.Param.Ntype = t;		// TODO: check if nname already has an ntype
		declare($$.Func.Nname, PFUNC);

		funchdr($$);
	}
|	'(' oarg_type_list_ocomma ')' sym '(' oarg_type_list_ocomma ')' fnres
	{
		var rcvr, t *Node

		$$ = nil;
		$2 = checkarglist($2, 0);
		$6 = checkarglist($6, 1);

		if $2 == nil {
			Yyerror("method has no receiver");
			break;
		}
		if $2.Next != nil {
			Yyerror("method has multiple receivers");
			break;
		}
		rcvr = $2.N;
		if rcvr.Op != ODCLFIELD {
			Yyerror("bad receiver in method");
			break;
		}

		t = Nod(OTFUNC, rcvr, nil);
		t.List = $6;
		t.Rlist = $8;

		$$ = Nod(ODCLFUNC, nil, nil);
		$$.Func.Shortname = newfuncname($4);
		$$.Func.Nname = methodname1($$.Func.Shortname, rcvr.Right);
		$$.Func.Nname.Name.Defn = $$;
		$$.Func.Nname.Name.Param.Ntype = t;
		$$.Func.Nname.Nointerface = nointerface;
		declare($$.Func.Nname, PFUNC);

		funchdr($$);
	}

hidden_fndcl:
	hidden_pkg_importsym '(' ohidden_funarg_list ')' ohidden_funres
	{
		var s *Sym
		var t *Type

		$$ = nil;

		s = $1;
		t = functype(nil, $3, $5);

		importsym(s, ONAME);
		if s.Def != nil && s.Def.Op == ONAME {
			if Eqtype(t, s.Def.Type) {
				dclcontext = PDISCARD;  // since we skip funchdr below
				break;
			}
			Yyerror("inconsistent definition for func %v during import\n\t%v\n\t%v", s, s.Def.Type, t);
		}

		$$ = newfuncname(s);
		$$.Type = t;
		declare($$, PFUNC);

		funchdr($$);
	}
|	'(' hidden_funarg_list ')' sym '(' ohidden_funarg_list ')' ohidden_funres
	{
		$$ = methodname1(newname($4), $2.N.Right); 
		$$.Type = functype($2.N, $6, $8);

		checkwidth($$.Type);
		addmethod($4, $$.Type, false, nointerface);
		nointerface = false
		funchdr($$);
		
		// inl.C's inlnode in on a dotmeth node expects to find the inlineable body as
		// (dotmeth's type).Nname.Inl, and dotmeth's type has been pulled
		// out by typecheck's lookdot as this $$.ttype.  So by providing
		// this back link here we avoid special casing there.
		$$.Type.Nname = $$;
	}

fntype:
	LFUNC '(' oarg_type_list_ocomma ')' fnres
	{
		$3 = checkarglist($3, 1);
		$$ = Nod(OTFUNC, nil, nil);
		$$.List = $3;
		$$.Rlist = $5;
	}

fnbody:
	{
		$$ = nil;
	}
|	'{' stmt_list '}'
	{
		$$ = $2;
		if $$ == nil {
			$$ = list1(Nod(OEMPTY, nil, nil));
		}
	}

fnres:
	%prec NotParen
	{
		$$ = nil;
	}
|	fnret_type
	{
		$$ = list1(Nod(ODCLFIELD, nil, $1));
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
		if nsyntaxerrors == 0 {
			testdclstack();
		}
		nointerface = false
		noescape = false
		norace = false
		nosplit = false
		nowritebarrier = false
		systemstack = false
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
		var l *NodeList

		var n *Node
		l = $1;
		if l == nil || l.N.Sym.Name == "?" {
			// ? symbol, during import (list1(nil) == nil)
			n = $2;
			if n.Op == OIND {
				n = n.Left;
			}
			n = embedded(n.Sym, importpkg);
			n.Right = $2;
			n.SetVal($3)
			$$ = list1(n);
			break;
		}

		for l=$1; l != nil; l=l.Next {
			l.N = Nod(ODCLFIELD, l.N, $2);
			l.N.SetVal($3)
		}
	}
|	embed oliteral
	{
		$1.SetVal($2)
		$$ = list1($1);
	}
|	'(' embed ')' oliteral
	{
		$2.SetVal($4)
		$$ = list1($2);
		Yyerror("cannot parenthesize embedded type");
	}
|	'*' embed oliteral
	{
		$2.Right = Nod(OIND, $2.Right, nil);
		$2.SetVal($3)
		$$ = list1($2);
	}
|	'(' '*' embed ')' oliteral
	{
		$3.Right = Nod(OIND, $3.Right, nil);
		$3.SetVal($5)
		$$ = list1($3);
		Yyerror("cannot parenthesize embedded type");
	}
|	'*' '(' embed ')' oliteral
	{
		$3.Right = Nod(OIND, $3.Right, nil);
		$3.SetVal($5)
		$$ = list1($3);
		Yyerror("cannot parenthesize embedded type");
	}

packname:
	LNAME
	{
		var n *Node

		$$ = $1;
		n = oldname($1);
		if n.Name != nil && n.Name.Pack != nil {
			n.Name.Pack.Used = true;
		}
	}
|	LNAME '.' sym
	{
		var pkg *Pkg

		if $1.Def == nil || $1.Def.Op != OPACK {
			Yyerror("%v is not a package", $1);
			pkg = localpkg;
		} else {
			$1.Def.Used = true;
			pkg = $1.Def.Name.Pkg;
		}
		$$ = restrictlookup($3.Name, pkg);
	}

embed:
	packname
	{
		$$ = embedded($1, localpkg);
	}

interfacedcl:
	new_name indcl
	{
		$$ = Nod(ODCLFIELD, $1, $2);
		ifacedcl($$);
	}
|	packname
	{
		$$ = Nod(ODCLFIELD, nil, oldname($1));
	}
|	'(' packname ')'
	{
		$$ = Nod(ODCLFIELD, nil, oldname($2));
		Yyerror("cannot parenthesize embedded type");
	}

indcl:
	'(' oarg_type_list_ocomma ')' fnres
	{
		// without func keyword
		$2 = checkarglist($2, 1);
		$$ = Nod(OTFUNC, fakethis(), nil);
		$$.List = $2;
		$$.Rlist = $4;
	}

/*
 * function arguments.
 */
arg_type:
	name_or_type
|	sym name_or_type
	{
		$$ = Nod(ONONAME, nil, nil);
		$$.Sym = $1;
		$$ = Nod(OKEY, $$, $2);
	}
|	sym dotdotdot
	{
		$$ = Nod(ONONAME, nil, nil);
		$$.Sym = $1;
		$$ = Nod(OKEY, $$, $2);
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
		$$ = nil;
	}
|	compound_stmt
|	common_dcl
	{
		$$ = liststmt($1);
	}
|	non_dcl_stmt
|	error
	{
		$$ = nil;
	}

non_dcl_stmt:
	simple_stmt
|	for_stmt
|	switch_stmt
|	select_stmt
|	if_stmt
|	labelname ':'
	{
		$1 = Nod(OLABEL, $1, nil);
		$1.Sym = dclstack;  // context, for goto restrictions
	}
	stmt
	{
		var l *NodeList

		$1.Name.Defn = $4;
		l = list1($1);
		if $4 != nil {
			l = list(l, $4);
		}
		$$ = liststmt(l);
	}
|	LFALL
	{
		// will be converted to OFALL
		$$ = Nod(OXFALL, nil, nil);
		$$.Xoffset = int64(block);
	}
|	LBREAK onew_name
	{
		$$ = Nod(OBREAK, $2, nil);
	}
|	LCONTINUE onew_name
	{
		$$ = Nod(OCONTINUE, $2, nil);
	}
|	LGO pseudocall
	{
		$$ = Nod(OPROC, $2, nil);
	}
|	LDEFER pseudocall
	{
		$$ = Nod(ODEFER, $2, nil);
	}
|	LGOTO new_name
	{
		$$ = Nod(OGOTO, $2, nil);
		$$.Sym = dclstack;  // context, for goto restrictions
	}
|	LRETURN oexpr_list
	{
		$$ = Nod(ORETURN, nil, nil);
		$$.List = $2;
		if $$.List == nil && Curfn != nil {
			var l *NodeList

			for l=Curfn.Func.Dcl; l != nil; l=l.Next {
				if l.N.Class == PPARAM {
					continue;
				}
				if l.N.Class != PPARAMOUT {
					break;
				}
				if l.N.Sym.Def != l.N {
					Yyerror("%s is shadowed during return", l.N.Sym.Name);
				}
			}
		}
	}

stmt_list:
	stmt
	{
		$$ = nil;
		if $1 != nil {
			$$ = list1($1);
		}
	}
|	stmt_list ';' stmt
	{
		$$ = $1;
		if $3 != nil {
			$$ = list($$, $3);
		}
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
		$$ = nil;
	}
|	expr

oexpr_list:
	{
		$$ = nil;
	}
|	expr_list

osimple_stmt:
	{
		$$ = nil;
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
		$$.U = nil
	}
|	LLITERAL

/*
 * import syntax from package header
 */
hidden_import:
	LIMPORT LNAME LLITERAL ';'
	{
		importimport($2, $3.U.(string));
	}
|	LVAR hidden_pkg_importsym hidden_type ';'
	{
		importvar($2, $3);
	}
|	LCONST hidden_pkg_importsym '=' hidden_constant ';'
	{
		importconst($2, Types[TIDEAL], $4);
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
		if $2 == nil {
			dclcontext = PEXTERN;  // since we skip the funcbody below
			break;
		}

		$2.Func.Inl = $3;

		funcbody($2);
		importlist = list(importlist, $2);

		if Debug['E'] > 0 {
			fmt.Printf("import [%q] func %v \n", importpkg.Path, $2)
			if Debug['m'] > 2 && $2.Func.Inl != nil {
				fmt.Printf("inl body:%v\n", $2.Func.Inl)
			}
		}
	}

hidden_pkg_importsym:
	hidden_importsym
	{
		$$ = $1;
		structpkg = $$.Pkg;
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
		$1 = Pkglookup($1.Name, builtinpkg);
		if $1.Def == nil || $1.Def.Op != OTYPE {
			Yyerror("%s is not a type", $1.Name);
			$$ = nil;
		} else {
			$$ = $1.Def.Type;
		}
	}
|	'[' ']' hidden_type
	{
		$$ = aindex(nil, $3);
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
		$$ = Ptrto($2);
	}
|	LCHAN hidden_type_non_recv_chan
	{
		$$ = typ(TCHAN);
		$$.Type = $2;
		$$.Chan = Cboth;
	}
|	LCHAN '(' hidden_type_recv_chan ')'
	{
		$$ = typ(TCHAN);
		$$.Type = $3;
		$$.Chan = Cboth;
	}
|	LCHAN LCOMM hidden_type
	{
		$$ = typ(TCHAN);
		$$.Type = $3;
		$$.Chan = Csend;
	}

hidden_type_recv_chan:
	LCOMM LCHAN hidden_type
	{
		$$ = typ(TCHAN);
		$$.Type = $3;
		$$.Chan = Crecv;
	}

hidden_type_func:
	LFUNC '(' ohidden_funarg_list ')' ohidden_funres
	{
		$$ = functype(nil, $3, $5);
	}

hidden_funarg:
	sym hidden_type oliteral
	{
		$$ = Nod(ODCLFIELD, nil, typenod($2));
		if $1 != nil {
			$$.Left = newname($1);
		}
		$$.SetVal($3)
	}
|	sym LDDD hidden_type oliteral
	{
		var t *Type
	
		t = typ(TARRAY);
		t.Bound = -1;
		t.Type = $3;

		$$ = Nod(ODCLFIELD, nil, typenod(t));
		if $1 != nil {
			$$.Left = newname($1);
		}
		$$.Isddd = true;
		$$.SetVal($4)
	}

hidden_structdcl:
	sym hidden_type oliteral
	{
		var s *Sym
		var p *Pkg

		if $1 != nil && $1.Name != "?" {
			$$ = Nod(ODCLFIELD, newname($1), typenod($2));
			$$.SetVal($3)
		} else {
			s = $2.Sym;
			if s == nil && Isptr[$2.Etype] {
				s = $2.Type.Sym;
			}
			p = importpkg;
			if $1 != nil {
				p = $1.Pkg;
			}
			$$ = embedded(s, p);
			$$.Right = typenod($2);
			$$.SetVal($3)
		}
	}

hidden_interfacedcl:
	sym '(' ohidden_funarg_list ')' ohidden_funres
	{
		$$ = Nod(ODCLFIELD, newname($1), typenod(functype(fakethis(), $3, $5)));
	}
|	hidden_type
	{
		$$ = Nod(ODCLFIELD, nil, typenod($1));
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
		$$ = list1(Nod(ODCLFIELD, nil, typenod($1)));
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
		switch($$.Val().Ctype()){
		case CTINT, CTRUNE:
			mpnegfix($$.Val().U.(*Mpint));
			break;
		case CTFLT:
			mpnegflt($$.Val().U.(*Mpflt));
			break;
		case CTCPLX:
			mpnegflt(&$$.Val().U.(*Mpcplx).Real);
			mpnegflt(&$$.Val().U.(*Mpcplx).Imag);
			break;
		default:
			Yyerror("bad negated constant");
		}
	}
|	sym
	{
		$$ = oldname(Pkglookup($1.Name, builtinpkg));
		if $$.Op != OLITERAL {
			Yyerror("bad constant %v", $$.Sym);
		}
	}

hidden_constant:
	hidden_literal
|	'(' hidden_literal '+' hidden_literal ')'
	{
		if $2.Val().Ctype() == CTRUNE && $4.Val().Ctype() == CTINT {
			$$ = $2;
			mpaddfixfix($2.Val().U.(*Mpint), $4.Val().U.(*Mpint), 0);
			break;
		}
		$4.Val().U.(*Mpcplx).Real = $4.Val().U.(*Mpcplx).Imag;
		Mpmovecflt(&$4.Val().U.(*Mpcplx).Imag, 0.0);
		$$ = nodcplxlit($2.Val(), $4.Val());
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
func fixlbrace(lbr int) {
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.C about loophack.
	if lbr == LBODY {
		loophack = 1
	}
}
