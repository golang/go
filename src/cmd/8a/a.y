// Inferno utils/8a/a.y
// http://code.google.com/p/inferno-os/source/browse/utils/8a/a.y
//
//	Copyright © 1994-1999 Lucent Technologies Inc.	All rights reserved.
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
#include <stdio.h>	/* if we don't, bison will, and a.h re-#defines getc */
#include <libc.h>
#include "a.h"
#include "../../runtime/funcdata.h"
%}
%union	{
	Sym	*sym;
	int32	lval;
	struct {
		int32 v1;
		int32 v2;
	} con2;
	double	dval;
	char	sval[8];
	Addr	addr;
	Addr2	addr2;
}
%left	'|'
%left	'^'
%left	'&'
%left	'<' '>'
%left	'+' '-'
%left	'*' '/' '%'
%token	<lval>	LTYPE0 LTYPE1 LTYPE2 LTYPE3 LTYPE4
%token	<lval>	LTYPEC LTYPED LTYPEN LTYPER LTYPET LTYPES LTYPEM LTYPEI LTYPEG LTYPEXC
%token	<lval>	LTYPEX LTYPEPC LTYPEF LCONST LFP LPC LSB
%token	<lval>	LBREG LLREG LSREG LFREG LXREG
%token	<dval>	LFCONST
%token	<sval>	LSCONST LSP
%token	<sym>	LNAME LLAB LVAR
%type	<lval>	con expr pointer offset
%type	<con2>	con2
%type	<addr>	mem imm imm2 reg nam rel rem rim rom omem nmem
%type	<addr2>	nonnon nonrel nonrem rimnon rimrem remrim
%type	<addr2>	spec1 spec2 spec3 spec4 spec5 spec6 spec7 spec8 spec9 spec10 spec11 spec12
%%
prog:
|	prog
	{
		stmtline = lineno;
	}
	line

line:
	LLAB ':'
	{
		if($1->value != pc)
			yyerror("redeclaration of %s", $1->name);
		$1->value = pc;
	}
	line
|	LNAME ':'
	{
		$1->type = LLAB;
		$1->value = pc;
	}
	line
|	';'
|	inst ';'
|	error ';'

inst:
	LNAME '=' expr
	{
		$1->type = LVAR;
		$1->value = $3;
	}
|	LVAR '=' expr
	{
		if($1->value != $3)
			yyerror("redeclaration of %s", $1->name);
		$1->value = $3;
	}
|	LTYPE0 nonnon	{ outcode($1, &$2); }
|	LTYPE1 nonrem	{ outcode($1, &$2); }
|	LTYPE2 rimnon	{ outcode($1, &$2); }
|	LTYPE3 rimrem	{ outcode($1, &$2); }
|	LTYPE4 remrim	{ outcode($1, &$2); }
|	LTYPER nonrel	{ outcode($1, &$2); }
|	LTYPED spec1	{ outcode($1, &$2); }
|	LTYPET spec2	{ outcode($1, &$2); }
|	LTYPEC spec3	{ outcode($1, &$2); }
|	LTYPEN spec4	{ outcode($1, &$2); }
|	LTYPES spec5	{ outcode($1, &$2); }
|	LTYPEM spec6	{ outcode($1, &$2); }
|	LTYPEI spec7	{ outcode($1, &$2); }
|	LTYPEG spec8	{ outcode($1, &$2); }
|	LTYPEXC spec9	{ outcode($1, &$2); }
|	LTYPEX spec10	{ outcode($1, &$2); }
|	LTYPEPC spec11	{ outcode($1, &$2); }
|	LTYPEF spec12	{ outcode($1, &$2); }

nonnon:
	{
		$$.from = nullgen;
		$$.to = nullgen;
	}
|	','
	{
		$$.from = nullgen;
		$$.to = nullgen;
	}

rimrem:
	rim ',' rem
	{
		$$.from = $1;
		$$.to = $3;
	}

remrim:
	rem ',' rim
	{
		$$.from = $1;
		$$.to = $3;
	}

rimnon:
	rim ','
	{
		$$.from = $1;
		$$.to = nullgen;
	}
|	rim
	{
		$$.from = $1;
		$$.to = nullgen;
	}

nonrem:
	',' rem
	{
		$$.from = nullgen;
		$$.to = $2;
	}
|	rem
	{
		$$.from = nullgen;
		$$.to = $1;
	}

nonrel:
	',' rel
	{
		$$.from = nullgen;
		$$.to = $2;
	}
|	rel
	{
		$$.from = nullgen;
		$$.to = $1;
	}
|	imm ',' rel
	{
		$$.from = $1;
		$$.to = $3;
	}

spec1:	/* DATA */
	nam '/' con ',' imm
	{
		$$.from = $1;
		$$.from.scale = $3;
		$$.to = $5;
	}

spec2:	/* TEXT */
	mem ',' imm2
	{
		$$.from = $1;
		$$.to = $3;
	}
|	mem ',' con ',' imm2
	{
		$$.from = $1;
		$$.from.scale = $3;
		$$.to = $5;
	}

spec3:	/* JMP/CALL */
	',' rom
	{
		$$.from = nullgen;
		$$.to = $2;
	}
|	rom
	{
		$$.from = nullgen;
		$$.to = $1;
	}
|	'*' nam
	{
		$$.from = nullgen;
		$$.to = $2;
		$$.to.index = $2.type;
		$$.to.type = D_INDIR+D_ADDR;
	}

spec4:	/* NOP */
	nonnon
|	nonrem

spec5:	/* SHL/SHR */
	rim ',' rem
	{
		$$.from = $1;
		$$.to = $3;
	}
|	rim ',' rem ':' LLREG
	{
		$$.from = $1;
		$$.to = $3;
		if($$.from.index != D_NONE)
			yyerror("dp shift with lhs index");
		$$.from.index = $5;
	}

spec6:	/* MOVW/MOVL */
	rim ',' rem
	{
		$$.from = $1;
		$$.to = $3;
	}
|	rim ',' rem ':' LSREG
	{
		$$.from = $1;
		$$.to = $3;
		if($$.to.index != D_NONE)
			yyerror("dp move with lhs index");
		$$.to.index = $5;
	}

spec7:
	rim ','
	{
		$$.from = $1;
		$$.to = nullgen;
	}
|	rim
	{
		$$.from = $1;
		$$.to = nullgen;
	}
|	rim ',' rem
	{
		$$.from = $1;
		$$.to = $3;
	}

spec8:	/* GLOBL */
	mem ',' imm
	{
		$$.from = $1;
		$$.to = $3;
	}
|	mem ',' con ',' imm
	{
		$$.from = $1;
		$$.from.scale = $3;
		$$.to = $5;
	}

spec9:	/* CMPPS/CMPPD */
	reg ',' rem ',' con
	{
		$$.from = $1;
		$$.to = $3;
		$$.to.offset = $5;
	}

spec10:	/* PINSRD */
	imm ',' rem ',' reg
	{
		$$.from = $3;
		$$.to = $5;
		if($1.type != D_CONST)
			yyerror("illegal constant");
		$$.to.offset = $1.offset;
	}

spec11:	/* PCDATA */
	rim ',' rim
	{
		if($1.type != D_CONST || $3.type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		$$.from = $1;
		$$.to = $3;
	}

spec12:	/* FUNCDATA */
	rim ',' rim
	{
		if($1.type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if($3.type != D_EXTERN && $3.type != D_STATIC)
			yyerror("value for FUNCDATA must be symbol reference");
 		$$.from = $1;
 		$$.to = $3;
 	}

rem:
	reg
|	mem

rom:
	rel
|	nmem
|	'*' reg
	{
		$$ = $2;
	}
|	'*' omem
	{
		$$ = $2;
	}
|	reg
|	omem
|	imm

rim:
	rem
|	imm

rel:
	con '(' LPC ')'
	{
		$$ = nullgen;
		$$.type = D_BRANCH;
		$$.offset = $1 + pc;
	}
|	LNAME offset
	{
		$$ = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", $1->name);
		$$.type = D_BRANCH;
		$$.offset = $2;
	}
|	LLAB offset
	{
		$$ = nullgen;
		$$.type = D_BRANCH;
		$$.offset = $1->value + $2;
	}

reg:
	LBREG
	{
		$$ = nullgen;
		$$.type = $1;
	}
|	LFREG
	{
		$$ = nullgen;
		$$.type = $1;
	}
|	LLREG
	{
		$$ = nullgen;
		$$.type = $1;
	}
|	LXREG
	{
		$$ = nullgen;
		$$.type = $1;
	}
|	LSP
	{
		$$ = nullgen;
		$$.type = D_SP;
	}
|	LSREG
	{
		$$ = nullgen;
		$$.type = $1;
	}

imm:
	'$' con
	{
		$$ = nullgen;
		$$.type = D_CONST;
		$$.offset = $2;
	}
|	'$' nam
	{
		$$ = $2;
		$$.index = $2.type;
		$$.type = D_ADDR;
		/*
		if($2.type == D_AUTO || $2.type == D_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.sym->name);
		 */
	}
|	'$' LSCONST
	{
		$$ = nullgen;
		$$.type = D_SCONST;
		memcpy($$.u.sval, $2, sizeof($$.u.sval));
	}
|	'$' LFCONST
	{
		$$ = nullgen;
		$$.type = D_FCONST;
		$$.u.dval = $2;
	}
|	'$' '(' LFCONST ')'
	{
		$$ = nullgen;
		$$.type = D_FCONST;
		$$.u.dval = $3;
	}
|	'$' '(' '-' LFCONST ')'
	{
		$$ = nullgen;
		$$.type = D_FCONST;
		$$.u.dval = -$4;
	}
|	'$' '-' LFCONST
	{
		$$ = nullgen;
		$$.type = D_FCONST;
		$$.u.dval = -$3;
	}

imm2:
	'$' con2
	{
		$$ = nullgen;
		$$.type = D_CONST2;
		$$.offset = $2.v1;
		$$.offset2 = $2.v2;
	}

con2:
	LCONST
	{
		$$.v1 = $1;
		$$.v2 = ArgsSizeUnknown;
	}
|	'-' LCONST
	{
		$$.v1 = -$2;
		$$.v2 = ArgsSizeUnknown;
	}
|	LCONST '-' LCONST
	{
		$$.v1 = $1;
		$$.v2 = $3;
	}
|	'-' LCONST '-' LCONST
	{
		$$.v1 = -$2;
		$$.v2 = $4;
	}

mem:
	omem
|	nmem

omem:
	con
	{
		$$ = nullgen;
		$$.type = D_INDIR+D_NONE;
		$$.offset = $1;
	}
|	con '(' LLREG ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+$3;
		$$.offset = $1;
	}
|	con '(' LSP ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+D_SP;
		$$.offset = $1;
	}
|	con '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+D_NONE;
		$$.offset = $1;
		$$.index = $3;
		$$.scale = $5;
		checkscale($$.scale);
	}
|	con '(' LLREG ')' '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+$3;
		$$.offset = $1;
		$$.index = $6;
		$$.scale = $8;
		checkscale($$.scale);
	}
|	con '(' LLREG ')' '(' LSREG '*' con ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+$3;
		$$.offset = $1;
		$$.index = $6;
		$$.scale = $8;
		checkscale($$.scale);
	}
|	'(' LLREG ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+$2;
	}
|	'(' LSP ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+D_SP;
	}
|	con '(' LSREG ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+$3;
		$$.offset = $1;
	}
|	'(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+D_NONE;
		$$.index = $2;
		$$.scale = $4;
		checkscale($$.scale);
	}
|	'(' LLREG ')' '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.type = D_INDIR+$2;
		$$.index = $5;
		$$.scale = $7;
		checkscale($$.scale);
	}

nmem:
	nam
	{
		$$ = $1;
	}
|	nam '(' LLREG '*' con ')'
	{
		$$ = $1;
		$$.index = $3;
		$$.scale = $5;
		checkscale($$.scale);
	}

nam:
	LNAME offset '(' pointer ')'
	{
		$$ = nullgen;
		$$.type = $4;
		$$.sym = linklookup(ctxt, $1->name, 0);
		$$.offset = $2;
	}
|	LNAME '<' '>' offset '(' LSB ')'
	{
		$$ = nullgen;
		$$.type = D_STATIC;
		$$.sym = linklookup(ctxt, $1->name, 1);
		$$.offset = $4;
	}

offset:
	{
		$$ = 0;
	}
|	'+' con
	{
		$$ = $2;
	}
|	'-' con
	{
		$$ = -$2;
	}

pointer:
	LSB
|	LSP
	{
		$$ = D_AUTO;
	}
|	LFP

con:
	LCONST
|	LVAR
	{
		$$ = $1->value;
	}
|	'-' con
	{
		$$ = -$2;
	}
|	'+' con
	{
		$$ = $2;
	}
|	'~' con
	{
		$$ = ~$2;
	}
|	'(' expr ')'
	{
		$$ = $2;
	}

expr:
	con
|	expr '+' expr
	{
		$$ = $1 + $3;
	}
|	expr '-' expr
	{
		$$ = $1 - $3;
	}
|	expr '*' expr
	{
		$$ = $1 * $3;
	}
|	expr '/' expr
	{
		$$ = $1 / $3;
	}
|	expr '%' expr
	{
		$$ = $1 % $3;
	}
|	expr '<' '<' expr
	{
		$$ = $1 << $4;
	}
|	expr '>' '>' expr
	{
		$$ = $1 >> $4;
	}
|	expr '&' expr
	{
		$$ = $1 & $3;
	}
|	expr '^' expr
	{
		$$ = $1 ^ $3;
	}
|	expr '|' expr
	{
		$$ = $1 | $3;
	}
