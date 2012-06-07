// Inferno utils/5a/a.y
// http://code.google.com/p/inferno-os/source/browse/utils/5a/a.y
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
%}
%union
{
	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Gen	gen;
}
%left	'|'
%left	'^'
%left	'&'
%left	'<' '>'
%left	'+' '-'
%left	'*' '/' '%'
%token	<lval>	LTYPE1 LTYPE2 LTYPE3 LTYPE4 LTYPE5
%token	<lval>	LTYPE6 LTYPE7 LTYPE8 LTYPE9 LTYPEA
%token	<lval>	LTYPEB LTYPEC LTYPED LTYPEE LTYPEF
%token	<lval>	LTYPEG LTYPEH LTYPEI LTYPEJ LTYPEK
%token	<lval>	LTYPEL LTYPEM LTYPEN LTYPEBX LTYPEPLD
%token	<lval>	LCONST LSP LSB LFP LPC
%token	<lval>	LTYPEX LR LREG LF LFREG LC LCREG LPSR LFCR
%token	<lval>	LCOND LS LAT
%token	<dval>	LFCONST
%token	<sval>	LSCONST
%token	<sym>	LNAME LLAB LVAR
%type	<lval>	con expr oexpr pointer offset sreg spreg creg
%type	<lval>	rcon cond reglist
%type	<gen>	gen rel reg regreg freg shift fcon frcon
%type	<gen>	imm ximm name oreg ireg nireg ioreg imsr
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
|	LNAME '=' expr ';'
	{
		$1->type = LVAR;
		$1->value = $3;
	}
|	LVAR '=' expr ';'
	{
		if($1->value != $3)
			yyerror("redeclaration of %s", $1->name);
		$1->value = $3;
	}
|	';'
|	inst ';'
|	error ';'

inst:
/*
 * ADD
 */
	LTYPE1 cond imsr ',' spreg ',' reg
	{
		outcode($1, $2, &$3, $5, &$7);
	}
|	LTYPE1 cond imsr ',' spreg ','
	{
		outcode($1, $2, &$3, $5, &nullgen);
	}
|	LTYPE1 cond imsr ',' reg
	{
		outcode($1, $2, &$3, NREG, &$5);
	}
/*
 * MVN
 */
|	LTYPE2 cond imsr ',' reg
	{
		outcode($1, $2, &$3, NREG, &$5);
	}
/*
 * MOVW
 */
|	LTYPE3 cond gen ',' gen
	{
		outcode($1, $2, &$3, NREG, &$5);
	}
/*
 * B/BL
 */
|	LTYPE4 cond comma rel
	{
		outcode($1, $2, &nullgen, NREG, &$4);
	}
|	LTYPE4 cond comma nireg
	{
		outcode($1, $2, &nullgen, NREG, &$4);
	}
/*
 * BX
 */
|	LTYPEBX comma ireg
	{
		outcode($1, Always, &nullgen, NREG, &$3);
	}
/*
 * BEQ
 */
|	LTYPE5 comma rel
	{
		outcode($1, Always, &nullgen, NREG, &$3);
	}
/*
 * SWI
 */
|	LTYPE6 cond comma gen
	{
		outcode($1, $2, &nullgen, NREG, &$4);
	}
/*
 * CMP
 */
|	LTYPE7 cond imsr ',' spreg comma
	{
		outcode($1, $2, &$3, $5, &nullgen);
	}
/*
 * MOVM
 */
|	LTYPE8 cond ioreg ',' '[' reglist ']'
	{
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = $6;
		outcode($1, $2, &$3, NREG, &g);
	}
|	LTYPE8 cond '[' reglist ']' ',' ioreg
	{
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = $4;
		outcode($1, $2, &g, NREG, &$7);
	}
/*
 * SWAP
 */
|	LTYPE9 cond reg ',' ireg ',' reg
	{
		outcode($1, $2, &$5, $3.reg, &$7);
	}
|	LTYPE9 cond reg ',' ireg comma
	{
		outcode($1, $2, &$5, $3.reg, &$3);
	}
|	LTYPE9 cond comma ireg ',' reg
	{
		outcode($1, $2, &$4, $6.reg, &$6);
	}
/*
 * RET
 */
|	LTYPEA cond comma
	{
		outcode($1, $2, &nullgen, NREG, &nullgen);
	}
/*
 * TEXT/GLOBL
 */
|	LTYPEB name ',' imm
	{
		outcode($1, Always, &$2, 0, &$4);
	}
|	LTYPEB name ',' con ',' imm
	{
		outcode($1, Always, &$2, $4, &$6);
	}
/*
 * DATA
 */
|	LTYPEC name '/' con ',' ximm
	{
		outcode($1, Always, &$2, $4, &$6);
	}
/*
 * CASE
 */
|	LTYPED cond reg comma
	{
		outcode($1, $2, &$3, NREG, &nullgen);
	}
/*
 * word
 */
|	LTYPEH comma ximm
	{
		outcode($1, Always, &nullgen, NREG, &$3);
	}
/*
 * floating-point coprocessor
 */
|	LTYPEI cond freg ',' freg
	{
		outcode($1, $2, &$3, NREG, &$5);
	}
|	LTYPEK cond frcon ',' freg
	{
		outcode($1, $2, &$3, NREG, &$5);
	}
|	LTYPEK cond frcon ',' LFREG ',' freg
	{
		outcode($1, $2, &$3, $5, &$7);
	}
|	LTYPEL cond freg ',' freg comma
	{
		outcode($1, $2, &$3, $5.reg, &nullgen);
	}
/*
 * MCR MRC
 */
|	LTYPEJ cond con ',' expr ',' spreg ',' creg ',' creg oexpr
	{
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset =
			(0xe << 24) |		/* opcode */
			($1 << 20) |		/* MCR/MRC */
			($2 << 28) |		/* scond */
			(($3 & 15) << 8) |	/* coprocessor number */
			(($5 & 7) << 21) |	/* coprocessor operation */
			(($7 & 15) << 12) |	/* arm register */
			(($9 & 15) << 16) |	/* Crn */
			(($11 & 15) << 0) |	/* Crm */
			(($12 & 7) << 5) |	/* coprocessor information */
			(1<<4);			/* must be set */
		outcode(AWORD, Always, &nullgen, NREG, &g);
	}
/*
 * MULL r1,r2,(hi,lo)
 */
|	LTYPEM cond reg ',' reg ',' regreg
	{
		outcode($1, $2, &$3, $5.reg, &$7);
	}
/*
 * MULA r1,r2,r3,r4: (r1*r2+r3) & 0xffffffff -> r4
 * MULAW{T,B} r1,r2,r3,r4
 */
|	LTYPEN cond reg ',' reg ',' reg ',' spreg
	{
		$7.type = D_REGREG2;
		$7.offset = $9;
		outcode($1, $2, &$3, $5.reg, &$7);
	}
/*
 * PLD
 */
|	LTYPEPLD oreg
	{
		outcode($1, Always, &$2, NREG, &nullgen);
	}
/*
 * END
 */
|	LTYPEE comma
	{
		outcode($1, Always, &nullgen, NREG, &nullgen);
	}

cond:
	{
		$$ = Always;
	}
|	cond LCOND
	{
		$$ = ($1 & ~C_SCOND) | $2;
	}
|	cond LS
	{
		$$ = $1 | $2;
	}

comma:
|	',' comma

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
		$$.sym = $1;
		$$.offset = $2;
	}
|	LLAB offset
	{
		$$ = nullgen;
		$$.type = D_BRANCH;
		$$.sym = $1;
		$$.offset = $1->value + $2;
	}

ximm:	'$' con
	{
		$$ = nullgen;
		$$.type = D_CONST;
		$$.offset = $2;
	}
|	'$' oreg
	{
		$$ = $2;
		$$.type = D_CONST;
	}
|	'$' '*' '$' oreg
	{
		$$ = $4;
		$$.type = D_OCONST;
	}
|	'$' LSCONST
	{
		$$ = nullgen;
		$$.type = D_SCONST;
		memcpy($$.sval, $2, sizeof($$.sval));
	}
|	fcon

fcon:
	'$' LFCONST
	{
		$$ = nullgen;
		$$.type = D_FCONST;
		$$.dval = $2;
	}
|	'$' '-' LFCONST
	{
		$$ = nullgen;
		$$.type = D_FCONST;
		$$.dval = -$3;
	}

reglist:
	spreg
	{
		$$ = 1 << $1;
	}
|	spreg '-' spreg
	{
		int i;
		$$=0;
		for(i=$1; i<=$3; i++)
			$$ |= 1<<i;
		for(i=$3; i<=$1; i++)
			$$ |= 1<<i;
	}
|	spreg comma reglist
	{
		$$ = (1<<$1) | $3;
	}

gen:
	reg
|	ximm
|	shift
|	shift '(' spreg ')'
	{
		$$ = $1;
		$$.reg = $3;
	}
|	LPSR
	{
		$$ = nullgen;
		$$.type = D_PSR;
		$$.reg = $1;
	}
|	LFCR
	{
		$$ = nullgen;
		$$.type = D_FPCR;
		$$.reg = $1;
	}
|	con
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.offset = $1;
	}
|	oreg
|	freg

nireg:
	ireg
|	name
	{
		$$ = $1;
		if($1.name != D_EXTERN && $1.name != D_STATIC) {
		}
	}

ireg:
	'(' spreg ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.reg = $2;
		$$.offset = 0;
	}

ioreg:
	ireg
|	con '(' sreg ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.reg = $3;
		$$.offset = $1;
	}

oreg:
	name
|	name '(' sreg ')'
	{
		$$ = $1;
		$$.type = D_OREG;
		$$.reg = $3;
	}
|	ioreg

imsr:
	reg
|	imm
|	shift

imm:	'$' con
	{
		$$ = nullgen;
		$$.type = D_CONST;
		$$.offset = $2;
	}

reg:
	spreg
	{
		$$ = nullgen;
		$$.type = D_REG;
		$$.reg = $1;
	}

regreg:
	'(' spreg ',' spreg ')'
	{
		$$ = nullgen;
		$$.type = D_REGREG;
		$$.reg = $2;
		$$.offset = $4;
	}

shift:
	spreg '<' '<' rcon
	{
		$$ = nullgen;
		$$.type = D_SHIFT;
		$$.offset = $1 | $4 | (0 << 5);
	}
|	spreg '>' '>' rcon
	{
		$$ = nullgen;
		$$.type = D_SHIFT;
		$$.offset = $1 | $4 | (1 << 5);
	}
|	spreg '-' '>' rcon
	{
		$$ = nullgen;
		$$.type = D_SHIFT;
		$$.offset = $1 | $4 | (2 << 5);
	}
|	spreg LAT '>' rcon
	{
		$$ = nullgen;
		$$.type = D_SHIFT;
		$$.offset = $1 | $4 | (3 << 5);
	}

rcon:
	spreg
	{
		if($$ < 0 || $$ >= 16)
			print("register value out of range\n");
		$$ = (($1&15) << 8) | (1 << 4);
	}
|	con
	{
		if($$ < 0 || $$ >= 32)
			print("shift value out of range\n");
		$$ = ($1&31) << 7;
	}

sreg:
	LREG
|	LPC
	{
		$$ = REGPC;
	}
|	LR '(' expr ')'
	{
		if($3 < 0 || $3 >= NREG)
			print("register value out of range\n");
		$$ = $3;
	}

spreg:
	sreg
|	LSP
	{
		$$ = REGSP;
	}

creg:
	LCREG
|	LC '(' expr ')'
	{
		if($3 < 0 || $3 >= NREG)
			print("register value out of range\n");
		$$ = $3;
	}

frcon:
	freg
|	fcon

freg:
	LFREG
	{
		$$ = nullgen;
		$$.type = D_FREG;
		$$.reg = $1;
	}
|	LF '(' con ')'
	{
		$$ = nullgen;
		$$.type = D_FREG;
		$$.reg = $3;
	}

name:
	con '(' pointer ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.name = $3;
		$$.sym = S;
		$$.offset = $1;
	}
|	LNAME offset '(' pointer ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.name = $4;
		$$.sym = $1;
		$$.offset = $2;
	}
|	LNAME '<' '>' offset '(' LSB ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.name = D_STATIC;
		$$.sym = $1;
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

oexpr:
	{
		$$ = 0;
	}
|	',' expr
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
