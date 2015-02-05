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
#include "../../runtime/funcdata.h"
%}
%union
{
	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Addr	addr;
}
%left	'|'
%left	'^'
%left	'&'
%left	'<' '>'
%left	'+' '-'
%left	'*' '/' '%'
%token	<lval>	LTYPE1 LTYPE2 LTYPE3 LTYPE4 LTYPE5
%token	<lval>	LTYPE6 LTYPE7 LTYPE8 LTYPE9 LTYPEA
%token	<lval>	LTYPEB LGLOBL LTYPEC LTYPED LTYPEE
%token	<lval>	LTYPEG LTYPEH LTYPEI LTYPEJ LTYPEK
%token	<lval>	LTYPEL LTYPEM LTYPEN LTYPEBX LTYPEPLD
%token	<lval>	LCONST LSP LSB LFP LPC
%token	<lval>	LTYPEX LTYPEPC LTYPEF LR LREG LF LFREG LC LCREG LPSR LFCR
%token	<lval>	LCOND LS LAT
%token	<dval>	LFCONST
%token	<sval>	LSCONST
%token	<sym>	LNAME LLAB LVAR
%type	<lval>	con expr oexpr pointer offset sreg spreg creg
%type	<lval>	rcon cond reglist
%type	<addr>	gen rel reg regreg freg shift fcon frcon textsize
%type	<addr>	imm ximm name oreg ireg nireg ioreg imsr
%%
prog:
|	prog
	{
		stmtline = lineno;
	}
	line

line:
	LNAME ':'
	{
		$1 = labellookup($1);
		if($1->type == LLAB && $1->value != pc)
			yyerror("redeclaration of %s", $1->labelname);
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
		outcode($1, $2, &$3, 0, &$5);
	}
/*
 * MVN
 */
|	LTYPE2 cond imsr ',' reg
	{
		outcode($1, $2, &$3, 0, &$5);
	}
/*
 * MOVW
 */
|	LTYPE3 cond gen ',' gen
	{
		outcode($1, $2, &$3, 0, &$5);
	}
/*
 * B/BL
 */
|	LTYPE4 cond comma rel
	{
		outcode($1, $2, &nullgen, 0, &$4);
	}
|	LTYPE4 cond comma nireg
	{
		outcode($1, $2, &nullgen, 0, &$4);
	}
/*
 * BX
 */
|	LTYPEBX comma ireg
	{
		outcode($1, Always, &nullgen, 0, &$3);
	}
/*
 * BEQ
 */
|	LTYPE5 comma rel
	{
		outcode($1, Always, &nullgen, 0, &$3);
	}
/*
 * SWI
 */
|	LTYPE6 cond comma gen
	{
		outcode($1, $2, &nullgen, 0, &$4);
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
		Addr g;

		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = $6;
		outcode($1, $2, &$3, 0, &g);
	}
|	LTYPE8 cond '[' reglist ']' ',' ioreg
	{
		Addr g;

		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = $4;
		outcode($1, $2, &g, 0, &$7);
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
		outcode($1, $2, &nullgen, 0, &nullgen);
	}
/*
 * TEXT
 */
|	LTYPEB name ',' '$' textsize
	{
		settext($2.sym);
		outcode($1, Always, &$2, 0, &$5);
	}
|	LTYPEB name ',' con ',' '$' textsize
	{
		settext($2.sym);
		outcode($1, Always, &$2, 0, &$7);
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = $4;
		}
	}
/*
 * GLOBL
 */
|	LGLOBL name ',' imm
	{
		settext($2.sym);
		outcode($1, Always, &$2, 0, &$4);
	}
|	LGLOBL name ',' con ',' imm
	{
		settext($2.sym);
		outcode($1, Always, &$2, 0, &$6);
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = $4;
		}
	}
/*
 * DATA
 */
|	LTYPEC name '/' con ',' ximm
	{
		outcode($1, Always, &$2, 0, &$6);
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = $4;
		}
	}
/*
 * CASE
 */
|	LTYPED cond reg comma
	{
		outcode($1, $2, &$3, 0, &nullgen);
	}
/*
 * word
 */
|	LTYPEH comma ximm
	{
		outcode($1, Always, &nullgen, 0, &$3);
	}
/*
 * floating-point coprocessor
 */
|	LTYPEI cond freg ',' freg
	{
		outcode($1, $2, &$3, 0, &$5);
	}
|	LTYPEK cond frcon ',' freg
	{
		outcode($1, $2, &$3, 0, &$5);
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
		Addr g;

		g = nullgen;
		g.type = TYPE_CONST;
		g.offset =
			(0xe << 24) |		/* opcode */
			($1 << 20) |		/* MCR/MRC */
			(($2^C_SCOND_XOR) << 28) |		/* scond */
			(($3 & 15) << 8) |	/* coprocessor number */
			(($5 & 7) << 21) |	/* coprocessor operation */
			(($7 & 15) << 12) |	/* arm register */
			(($9 & 15) << 16) |	/* Crn */
			(($11 & 15) << 0) |	/* Crm */
			(($12 & 7) << 5) |	/* coprocessor information */
			(1<<4);			/* must be set */
		outcode(AMRC, Always, &nullgen, 0, &g);
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
		$7.type = TYPE_REGREG2;
		$7.offset = $9;
		outcode($1, $2, &$3, $5.reg, &$7);
	}
/*
 * PLD
 */
|	LTYPEPLD oreg
	{
		outcode($1, Always, &$2, 0, &nullgen);
	}
/*
 * PCDATA
 */
|	LTYPEPC gen ',' gen
	{
		if($2.type != TYPE_CONST || $4.type != TYPE_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode($1, Always, &$2, 0, &$4);
	}
/*
 * FUNCDATA
 */
|	LTYPEF gen ',' gen
	{
		if($2.type != TYPE_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if($4.type != NAME_EXTERN && $4.type != NAME_STATIC && $4.type != TYPE_MEM)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode($1, Always, &$2, 0, &$4);
	}
/*
 * END
 */
|	LTYPEE comma
	{
		outcode($1, Always, &nullgen, 0, &nullgen);
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
		$$.type = TYPE_BRANCH;
		$$.offset = $1 + pc;
	}
|	LNAME offset
	{
		$1 = labellookup($1);
		$$ = nullgen;
		if(pass == 2 && $1->type != LLAB)
			yyerror("undefined label: %s", $1->labelname);
		$$.type = TYPE_BRANCH;
		$$.offset = $1->value + $2;
	}

textsize:
	LCONST
	{
		$$ = nullgen;
		$$.type = TYPE_TEXTSIZE;
		$$.offset = $1;
		$$.u.argsize = ArgsSizeUnknown;
	}
|	'-' LCONST
	{
		$$ = nullgen;
		$$.type = TYPE_TEXTSIZE;
		$$.offset = -$2;
		$$.u.argsize = ArgsSizeUnknown;
	}
|	LCONST '-' LCONST
	{
		$$ = nullgen;
		$$.type = TYPE_TEXTSIZE;
		$$.offset = $1;
		$$.u.argsize = $3;
	}
|	'-' LCONST '-' LCONST
	{
		$$ = nullgen;
		$$.type = TYPE_TEXTSIZE;
		$$.offset = -$2;
		$$.u.argsize = $4;
	}

ximm:	'$' con
	{
		$$ = nullgen;
		$$.type = TYPE_CONST;
		$$.offset = $2;
	}
|	'$' oreg
	{
		$$ = $2;
		$$.type = TYPE_ADDR;
	}
|	'$' LSCONST
	{
		$$ = nullgen;
		$$.type = TYPE_SCONST;
		memcpy($$.u.sval, $2, sizeof($$.u.sval));
	}
|	fcon

fcon:
	'$' LFCONST
	{
		$$ = nullgen;
		$$.type = TYPE_FCONST;
		$$.u.dval = $2;
	}
|	'$' '-' LFCONST
	{
		$$ = nullgen;
		$$.type = TYPE_FCONST;
		$$.u.dval = -$3;
	}

reglist:
	spreg
	{
		if($1 < REG_R0 || $1 > REG_R15)
			yyerror("invalid register in reglist");

		$$ = 1 << ($1&15);
	}
|	spreg '-' spreg
	{
		int i;

		if($1 < REG_R0 || $1 > REG_R15)
			yyerror("invalid register in reglist");
		if($3 < REG_R0 || $3 > REG_R15)
			yyerror("invalid register in reglist");

		$$=0;
		for(i=$1; i<=$3; i++)
			$$ |= 1<<(i&15);
		for(i=$3; i<=$1; i++)
			$$ |= 1<<(i&15);
	}
|	spreg comma reglist
	{
		if($1 < REG_R0 || $1 > REG_R15)
			yyerror("invalid register in reglist");

		$$ = (1<<($1&15)) | $3;
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
		$$.type = TYPE_REG;
		$$.reg = $1;
	}
|	LFCR
	{
		$$ = nullgen;
		$$.type = TYPE_REG;
		$$.reg = $1;
	}
|	con
	{
		$$ = nullgen;
		$$.type = TYPE_MEM;
		$$.offset = $1;
	}
|	oreg
|	freg

nireg:
	ireg
|	name
	{
		$$ = $1;
		if($1.name != NAME_EXTERN && $1.name != NAME_STATIC) {
		}
	}

ireg:
	'(' spreg ')'
	{
		$$ = nullgen;
		$$.type = TYPE_MEM;
		$$.reg = $2;
		$$.offset = 0;
	}

ioreg:
	ireg
|	con '(' sreg ')'
	{
		$$ = nullgen;
		$$.type = TYPE_MEM;
		$$.reg = $3;
		$$.offset = $1;
	}

oreg:
	name
|	name '(' sreg ')'
	{
		$$ = $1;
		$$.type = TYPE_MEM;
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
		$$.type = TYPE_CONST;
		$$.offset = $2;
	}

reg:
	spreg
	{
		$$ = nullgen;
		$$.type = TYPE_REG;
		$$.reg = $1;
	}

regreg:
	'(' spreg ',' spreg ')'
	{
		$$ = nullgen;
		$$.type = TYPE_REGREG;
		$$.reg = $2;
		$$.offset = $4;
	}

shift:
	spreg '<' '<' rcon
	{
		$$ = nullgen;
		$$.type = TYPE_SHIFT;
		$$.offset = $1&15 | $4 | (0 << 5);
	}
|	spreg '>' '>' rcon
	{
		$$ = nullgen;
		$$.type = TYPE_SHIFT;
		$$.offset = $1&15 | $4 | (1 << 5);
	}
|	spreg '-' '>' rcon
	{
		$$ = nullgen;
		$$.type = TYPE_SHIFT;
		$$.offset = $1&15 | $4 | (2 << 5);
	}
|	spreg LAT '>' rcon
	{
		$$ = nullgen;
		$$.type = TYPE_SHIFT;
		$$.offset = $1&15 | $4 | (3 << 5);
	}

rcon:
	spreg
	{
		if($$ < REG_R0 || $$ > REG_R15)
			print("register value out of range in shift\n");
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
			print("register value out of range in R(...)\n");
		$$ = REG_R0 + $3;
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
			print("register value out of range in C(...)\n");
		$$ = $3; // TODO(rsc): REG_C0+$3
	}

frcon:
	freg
|	fcon

freg:
	LFREG
	{
		$$ = nullgen;
		$$.type = TYPE_REG;
		$$.reg = $1;
	}
|	LF '(' con ')'
	{
		$$ = nullgen;
		$$.type = TYPE_REG;
		$$.reg = REG_F0 + $3;
	}

name:
	con '(' pointer ')'
	{
		$$ = nullgen;
		$$.type = TYPE_MEM;
		$$.name = $3;
		$$.sym = nil;
		$$.offset = $1;
	}
|	LNAME offset '(' pointer ')'
	{
		$$ = nullgen;
		$$.type = TYPE_MEM;
		$$.name = $4;
		$$.sym = linklookup(ctxt, $1->name, 0);
		$$.offset = $2;
	}
|	LNAME '<' '>' offset '(' LSB ')'
	{
		$$ = nullgen;
		$$.type = TYPE_MEM;
		$$.name = NAME_STATIC;
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
