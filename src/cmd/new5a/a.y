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
package main

import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	. "cmd/internal/obj/arm"
)
%}

%union {
	sym *asm.Sym
	lval int32
	dval float64
	sval string
	addr obj.Addr
}

%left	'|'
%left	'^'
%left	'&'
%left	'<' '>'
%left	'+' '-'
%left	'*' '/' '%'
%token	<lval>	LTYPE1 LTYPE2 LTYPE3 LTYPE4 LTYPE5
%token	<lval>	LTYPE6 LTYPE7 LTYPE8 LTYPE9 LTYPEA
%token	<lval>	LTYPEB LTYPEC LTYPED LTYPEE
%token	<lval>	LTYPEG LTYPEH LTYPEI LTYPEJ LTYPEK
%token	<lval>	LTYPEL LTYPEM LTYPEN LTYPEBX LTYPEPLD
%token	<lval>	LCONST LSP LSB LFP LPC
%token	<lval>	LTYPEX LTYPEPC LTYPEF LR LREG LF LFREG LC LCREG LPSR LFCR
%token	<lval>	LCOND LS LAT LGLOBL
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
		stmtline = asm.Lineno;
	}
	line

line:
	LNAME ':'
	{
		$1 = asm.LabelLookup($1);
		if $1.Type == LLAB && $1.Value != int64(asm.PC) {
			yyerror("redeclaration of %s", $1.Labelname)
		}
		$1.Type = LLAB;
		$1.Value = int64(asm.PC)
	}
	line
|	LNAME '=' expr ';'
	{
		$1.Type = LVAR;
		$1.Value = int64($3);
	}
|	LVAR '=' expr ';'
	{
		if $1.Value != int64($3) {
			yyerror("redeclaration of %s", $1.Name)
		}
		$1.Value = int64($3);
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
		var g obj.Addr

		g = nullgen;
		g.Type = obj.TYPE_CONST;
		g.Offset = int64($6);
		outcode($1, $2, &$3, 0, &g);
	}
|	LTYPE8 cond '[' reglist ']' ',' ioreg
	{
		var g obj.Addr

		g = nullgen;
		g.Type = obj.TYPE_CONST;
		g.Offset = int64($4);
		outcode($1, $2, &g, 0, &$7);
	}
/*
 * SWAP
 */
|	LTYPE9 cond reg ',' ireg ',' reg
	{
		outcode($1, $2, &$5, int32($3.Reg), &$7);
	}
|	LTYPE9 cond reg ',' ireg comma
	{
		outcode($1, $2, &$5, int32($3.Reg), &$3);
	}
|	LTYPE9 cond comma ireg ',' reg
	{
		outcode($1, $2, &$4, int32($6.Reg), &$6);
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
		asm.Settext($2.Sym);
		outcode($1, Always, &$2, 0, &$5);
	}
|	LTYPEB name ',' con ',' '$' textsize
	{
		asm.Settext($2.Sym);
		outcode($1, Always, &$2, 0, &$7);
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST;
			lastpc.From3.Offset = int64($4)
		}
	}
/*
 * GLOBL
 */
|	LGLOBL name ',' imm
	{
		asm.Settext($2.Sym)
		outcode($1, Always, &$2, 0, &$4)
	}
|	LGLOBL name ',' con ',' imm
	{
		asm.Settext($2.Sym)
		outcode($1, Always, &$2, 0, &$6)
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = int64($4)
		}
	}

/*
 * DATA
 */
|	LTYPEC name '/' con ',' ximm
	{
		outcode($1, Always, &$2, 0, &$6)
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = int64($4)
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
		outcode($1, $2, &$3, int32($5.Reg), &nullgen);
	}
/*
 * MCR MRC
 */
|	LTYPEJ cond con ',' expr ',' spreg ',' creg ',' creg oexpr
	{
		var g obj.Addr

		g = nullgen;
		g.Type = obj.TYPE_CONST;
		g.Offset = int64(
			(0xe << 24) |		/* opcode */
			($1 << 20) |		/* MCR/MRC */
			(($2^C_SCOND_XOR) << 28) |		/* scond */
			(($3 & 15) << 8) |	/* coprocessor number */
			(($5 & 7) << 21) |	/* coprocessor operation */
			(($7 & 15) << 12) |	/* arm register */
			(($9 & 15) << 16) |	/* Crn */
			(($11 & 15) << 0) |	/* Crm */
			(($12 & 7) << 5) |	/* coprocessor information */
			(1<<4));			/* must be set */
		outcode(AMRC, Always, &nullgen, 0, &g);
	}
/*
 * MULL r1,r2,(hi,lo)
 */
|	LTYPEM cond reg ',' reg ',' regreg
	{
		outcode($1, $2, &$3, int32($5.Reg), &$7);
	}
/*
 * MULA r1,r2,r3,r4: (r1*r2+r3) & 0xffffffff . r4
 * MULAW{T,B} r1,r2,r3,r4
 */
|	LTYPEN cond reg ',' reg ',' reg ',' spreg
	{
		$7.Type = obj.TYPE_REGREG2;
		$7.Offset = int64($9);
		outcode($1, $2, &$3, int32($5.Reg), &$7);
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
		if $2.Type != obj.TYPE_CONST || $4.Type != obj.TYPE_CONST {
			yyerror("arguments to PCDATA must be integer constants")
		}
		outcode($1, Always, &$2, 0, &$4);
	}
/*
 * FUNCDATA
 */
|	LTYPEF gen ',' gen
	{
		if $2.Type != obj.TYPE_CONST {
			yyerror("index for FUNCDATA must be integer constant")
		}
		if $4.Type != obj.NAME_EXTERN && $4.Type != obj.NAME_STATIC && $4.Type != obj.TYPE_MEM {
			yyerror("value for FUNCDATA must be symbol reference")
		}
 		outcode($1, Always, &$2, 0, &$4);
	}
/*
 * END
 */
|	LTYPEE comma
	{
		outcode($1, Always, &nullgen, 0, &nullgen);
	}

textsize:
	LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = int64($1)
		$$.U.Argsize = obj.ArgsSizeUnknown;
	}
|	'-' LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = -int64($2)
		$$.U.Argsize = obj.ArgsSizeUnknown;
	}
|	LCONST '-' LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = int64($1)
		$$.U.Argsize = int32($3);
	}
|	'-' LCONST '-' LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = -int64($2)
		$$.U.Argsize = int32($4);
	}

cond:
	{
		$$ = Always;
	}
|	cond LCOND
	{
		$$ = ($1 & ^ C_SCOND) | $2;
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
		$$.Type = obj.TYPE_BRANCH;
		$$.Offset = int64($1) + int64(asm.PC);
	}
|	LNAME offset
	{
		$1 = asm.LabelLookup($1);
		$$ = nullgen;
		if asm.Pass == 2 && $1.Type != LLAB {
			yyerror("undefined label: %s", $1.Labelname)
		}
		$$.Type = obj.TYPE_BRANCH;
		$$.Offset = $1.Value + int64($2);
	}

ximm:	'$' con
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_CONST;
		$$.Offset = int64($2);
	}
|	'$' oreg
	{
		$$ = $2;
		$$.Type = obj.TYPE_ADDR;
	}
|	'$' LSCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_SCONST;
		$$.U.Sval = $2
	}
|	fcon

fcon:
	'$' LFCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_FCONST;
		$$.U.Dval = $2;
	}
|	'$' '-' LFCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_FCONST;
		$$.U.Dval = -$3;
	}

reglist:
	spreg
	{
		$$ = 1 << uint($1&15);
	}
|	spreg '-' spreg
	{
		$$=0;
		for i:=$1; i<=$3; i++ {
			$$ |= 1<<uint(i&15)
		}
		for i:=$3; i<=$1; i++ {
			$$ |= 1<<uint(i&15)
		}
	}
|	spreg comma reglist
	{
		$$ = (1<<uint($1&15)) | $3;
	}

gen:
	reg
|	ximm
|	shift
|	shift '(' spreg ')'
	{
		$$ = $1;
		$$.Reg = int16($3);
	}
|	LPSR
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1);
	}
|	LFCR
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1);
	}
|	con
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Offset = int64($1);
	}
|	oreg
|	freg

nireg:
	ireg
|	name
	{
		$$ = $1;
		if($1.Name != obj.NAME_EXTERN && $1.Name != obj.NAME_STATIC) {
		}
	}

ireg:
	'(' spreg ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Reg = int16($2);
		$$.Offset = 0;
	}

ioreg:
	ireg
|	con '(' sreg ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Reg = int16($3);
		$$.Offset = int64($1);
	}

oreg:
	name
|	name '(' sreg ')'
	{
		$$ = $1;
		$$.Type = obj.TYPE_MEM;
		$$.Reg = int16($3);
	}
|	ioreg

imsr:
	reg
|	imm
|	shift

imm:	'$' con
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_CONST;
		$$.Offset = int64($2);
	}

reg:
	spreg
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}

regreg:
	'(' spreg ',' spreg ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REGREG;
		$$.Reg = int16($2);
		$$.Offset = int64($4);
	}

shift:
	spreg '<' '<' rcon
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_SHIFT;
		$$.Offset = int64($1&15) | int64($4) | (0 << 5);
	}
|	spreg '>' '>' rcon
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_SHIFT;
		$$.Offset = int64($1&15) | int64($4) | (1 << 5);
	}
|	spreg '-' '>' rcon
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_SHIFT;
		$$.Offset = int64($1&15) | int64($4) | (2 << 5);
	}
|	spreg LAT '>' rcon
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_SHIFT;
		$$.Offset = int64($1&15) | int64($4) | (3 << 5);
	}

rcon:
	spreg
	{
		if $$ < REG_R0 || $$ > REG_R15 {
			print("register value out of range\n")
		}
		$$ = (($1&15) << 8) | (1 << 4);
	}
|	con
	{
		if $$ < 0 || $$ >= 32 {
			print("shift value out of range\n")
		}
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
		if $3 < 0 || $3 >= NREG {
			print("register value out of range\n")
		}
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
		if $3 < 0 || $3 >= NREG {
			print("register value out of range\n")
		}
		$$ = $3; // TODO(rsc): REG_C0+$3
	}

frcon:
	freg
|	fcon

freg:
	LFREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}
|	LF '(' con ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16(REG_F0 + $3);
	}

name:
	con '(' pointer ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Name = int8($3);
		$$.Sym = nil;
		$$.Offset = int64($1);
	}
|	LNAME offset '(' pointer ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Name = int8($4);
		$$.Sym = obj.Linklookup(asm.Ctxt, $1.Name, 0);
		$$.Offset = int64($2);
	}
|	LNAME '<' '>' offset '(' LSB ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Name = obj.NAME_STATIC;
		$$.Sym = obj.Linklookup(asm.Ctxt, $1.Name, 1);
		$$.Offset = int64($4);
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
		$$ = int32($1.Value);
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
		$$ = ^$2;
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
		$$ = $1 << uint($4);
	}
|	expr '>' '>' expr
	{
		$$ = $1 >> uint($4);
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
