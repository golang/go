// Inferno utils/6a/a.y
// http://code.google.com/p/inferno-os/source/browse/utils/6a/a.y
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
	"cmd/internal/obj/x86"
)
%}

%union {
	sym *asm.Sym
	lval int64
	dval float64
	sval string
	addr obj.Addr
	addr2 Addr2
}

%left	'|'
%left	'^'
%left	'&'
%left	'<' '>'
%left	'+' '-'
%left	'*' '/' '%'
%token	<lval>	LTYPE0 LTYPE1 LTYPE2 LTYPE3 LTYPE4
%token	<lval>	LTYPEC LTYPED LTYPEN LTYPER LTYPET LTYPEG LTYPEPC
%token	<lval>	LTYPES LTYPEM LTYPEI LTYPEXC LTYPEX LTYPERT LTYPEF
%token	<lval>	LCONST LFP LPC LSB
%token	<lval>	LBREG LLREG LSREG LFREG LMREG LXREG
%token	<dval>	LFCONST
%token	<sval>	LSCONST LSP
%token	<sym>	LNAME LLAB LVAR
%type	<lval>	con con2 expr pointer offset
%type	<addr>	mem imm imm2 reg nam rel rem rim rom omem nmem
%type	<addr2>	nonnon nonrel nonrem rimnon rimrem remrim
%type	<addr2>	spec1 spec2 spec3 spec4 spec5 spec6 spec7 spec8 spec9
%type	<addr2>	spec10 spec11 spec12 spec13
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
			yyerror("redeclaration of %s (%s)", $1.Labelname, $1.Name);
		}
		$1.Type = LLAB;
		$1.Value = int64(asm.PC)
	}
	line
|	';'
|	inst ';'
|	error ';'

inst:
	LNAME '=' expr
	{
		$1.Type = LVAR;
		$1.Value = $3;
	}
|	LVAR '=' expr
	{
		if $1.Value != $3 {
			yyerror("redeclaration of %s", $1.Name);
		}
		$1.Value = $3;
	}
|	LTYPE0 nonnon	{ outcode(int($1), &$2); }
|	LTYPE1 nonrem	{ outcode(int($1), &$2); }
|	LTYPE2 rimnon	{ outcode(int($1), &$2); }
|	LTYPE3 rimrem	{ outcode(int($1), &$2); }
|	LTYPE4 remrim	{ outcode(int($1), &$2); }
|	LTYPER nonrel	{ outcode(int($1), &$2); }
|	LTYPED spec1	{ outcode(int($1), &$2); }
|	LTYPET spec2	{ outcode(int($1), &$2); }
|	LTYPEC spec3	{ outcode(int($1), &$2); }
|	LTYPEN spec4	{ outcode(int($1), &$2); }
|	LTYPES spec5	{ outcode(int($1), &$2); }
|	LTYPEM spec6	{ outcode(int($1), &$2); }
|	LTYPEI spec7	{ outcode(int($1), &$2); }
|	LTYPEXC spec8	{ outcode(int($1), &$2); }
|	LTYPEX spec9	{ outcode(int($1), &$2); }
|	LTYPERT spec10	{ outcode(int($1), &$2); }
|	LTYPEG spec11	{ outcode(int($1), &$2); }
|	LTYPEPC spec12	{ outcode(int($1), &$2); }
|	LTYPEF spec13	{ outcode(int($1), &$2); }

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
		$$.from.Scale = int8($3);
		$$.to = $5;
	}

spec2:	/* TEXT */
	mem ',' imm2
	{
		asm.Settext($1.Sym);
		$$.from = $1;
		$$.to = $3;
	}
|	mem ',' con ',' imm2
	{
		asm.Settext($1.Sym);
		$$.from = $1;
		$$.from.Scale = int8($3);
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
		if $$.from.Index != x86.D_NONE {
			yyerror("dp shift with lhs index");
		}
		$$.from.Index = uint8($5);
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
		if $$.to.Index != x86.D_NONE {
			yyerror("dp move with lhs index");
		}
		$$.to.Index = uint8($5);
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

spec8:	/* CMPPS/CMPPD */
	reg ',' rem ',' con
	{
		$$.from = $1;
		$$.to = $3;
		$$.to.Offset = $5;
	}

spec9:	/* shufl */
	imm ',' rem ',' reg
	{
		$$.from = $3;
		$$.to = $5;
		if $1.Type != x86.D_CONST {
			yyerror("illegal constant");
		}
		$$.to.Offset = $1.Offset;
	}

spec10:	/* RET/RETF */
	{
		$$.from = nullgen;
		$$.to = nullgen;
	}
|	imm
	{
		$$.from = $1;
		$$.to = nullgen;
	}

spec11:	/* GLOBL */
	mem ',' imm
	{
		$$.from = $1;
		$$.to = $3;
	}
|	mem ',' con ',' imm
	{
		$$.from = $1;
		$$.from.Scale = int8($3);
		$$.to = $5;
	}

spec12:	/* asm.PCDATA */
	rim ',' rim
	{
		if $1.Type != x86.D_CONST || $3.Type != x86.D_CONST {
			yyerror("arguments to asm.PCDATA must be integer constants");
		}
		$$.from = $1;
		$$.to = $3;
	}

spec13:	/* FUNCDATA */
	rim ',' rim
	{
		if $1.Type != x86.D_CONST {
			yyerror("index for FUNCDATA must be integer constant");
		}
		if $3.Type != x86.D_EXTERN && $3.Type != x86.D_STATIC {
			yyerror("value for FUNCDATA must be symbol reference");
		}
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

rim:
	rem
|	imm

rel:
	con '(' LPC ')'
	{
		$$ = nullgen;
		$$.Type = x86.D_BRANCH;
		$$.Offset = $1 + int64(asm.PC);
	}
|	LNAME offset
	{
		$1 = asm.LabelLookup($1);
		$$ = nullgen;
		if asm.Pass == 2 && $1.Type != LLAB {
			yyerror("undefined label: %s", $1.Labelname);
		}
		$$.Type = x86.D_BRANCH;
		$$.Offset = $1.Value + $2;
	}

reg:
	LBREG
	{
		$$ = nullgen;
		$$.Type = int16($1);
	}
|	LFREG
	{
		$$ = nullgen;
		$$.Type = int16($1);
	}
|	LLREG
	{
		$$ = nullgen;
		$$.Type = int16($1);
	}
|	LMREG
	{
		$$ = nullgen;
		$$.Type = int16($1);
	}
|	LSP
	{
		$$ = nullgen;
		$$.Type = x86.D_SP;
	}
|	LSREG
	{
		$$ = nullgen;
		$$.Type = int16($1);
	}
|	LXREG
	{
		$$ = nullgen;
		$$.Type = int16($1);
	}
imm2:
	'$' con2
	{
		$$ = nullgen;
		$$.Type = x86.D_CONST;
		$$.Offset = $2;
	}

imm:
	'$' con
	{
		$$ = nullgen;
		$$.Type = x86.D_CONST;
		$$.Offset = $2;
	}
|	'$' nam
	{
		$$ = $2;
		$$.Index = uint8($2.Type);
		$$.Type = x86.D_ADDR;
		/*
		if($2.Type == x86.D_AUTO || $2.Type == x86.D_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.sym.Name);
		 */
	}
|	'$' LSCONST
	{
		$$ = nullgen;
		$$.Type = x86.D_SCONST;
		$$.U.Sval = ($2+"\x00\x00\x00\x00\x00\x00\x00\x00")[:8]
	}
|	'$' LFCONST
	{
		$$ = nullgen;
		$$.Type = x86.D_FCONST;
		$$.U.Dval = $2;
	}
|	'$' '(' LFCONST ')'
	{
		$$ = nullgen;
		$$.Type = x86.D_FCONST;
		$$.U.Dval = $3;
	}
|	'$' '(' '-' LFCONST ')'
	{
		$$ = nullgen;
		$$.Type = x86.D_FCONST;
		$$.U.Dval = -$4;
	}
|	'$' '-' LFCONST
	{
		$$ = nullgen;
		$$.Type = x86.D_FCONST;
		$$.U.Dval = -$3;
	}

mem:
	omem
|	nmem

omem:
	con
	{
		$$ = nullgen;
		$$.Type = x86.D_INDIR+x86.D_NONE;
		$$.Offset = $1;
	}
|	con '(' LLREG ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+$3);
		$$.Offset = $1;
	}
|	con '(' LSP ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+x86.D_SP);
		$$.Offset = $1;
	}
|	con '(' LSREG ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+$3);
		$$.Offset = $1;
	}
|	con '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+x86.D_NONE);
		$$.Offset = $1;
		$$.Index = uint8($3);
		$$.Scale = int8($5);
		checkscale($$.Scale);
	}
|	con '(' LLREG ')' '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+$3);
		$$.Offset = $1;
		$$.Index = uint8($6);
		$$.Scale = int8($8);
		checkscale($$.Scale);
	}
|	con '(' LLREG ')' '(' LSREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+$3);
		$$.Offset = $1;
		$$.Index = uint8($6);
		$$.Scale = int8($8);
		checkscale($$.Scale);
	}
|	'(' LLREG ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+$2);
	}
|	'(' LSP ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+x86.D_SP);
	}
|	'(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+x86.D_NONE);
		$$.Index = uint8($2);
		$$.Scale = int8($4);
		checkscale($$.Scale);
	}
|	'(' LLREG ')' '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = int16(x86.D_INDIR+$2);
		$$.Index = uint8($5);
		$$.Scale = int8($7);
		checkscale($$.Scale);
	}

nmem:
	nam
	{
		$$ = $1;
	}
|	nam '(' LLREG '*' con ')'
	{
		$$ = $1;
		$$.Index = uint8($3);
		$$.Scale = int8($5);
		checkscale($$.Scale);
	}

nam:
	LNAME offset '(' pointer ')'
	{
		$$ = nullgen;
		$$.Type = int16($4);
		$$.Sym = obj.Linklookup(asm.Ctxt, $1.Name, 0);
		$$.Offset = $2;
	}
|	LNAME '<' '>' offset '(' LSB ')'
	{
		$$ = nullgen;
		$$.Type = x86.D_STATIC;
		$$.Sym = obj.Linklookup(asm.Ctxt, $1.Name, 1);
		$$.Offset = $4;
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
		$$ = x86.D_AUTO;
	}
|	LFP

con:
	LCONST
|	LVAR
	{
		$$ = $1.Value;
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

con2:
	LCONST
	{
		$$ = int64(uint64($1 & 0xffffffff) + (obj.ArgsSizeUnknown<<32))
	}
|	'-' LCONST
	{
		$$ = int64(uint64(-$2 & 0xffffffff) + (obj.ArgsSizeUnknown<<32))
	}
|	LCONST '-' LCONST
	{
		$$ = ($1 & 0xffffffff) + (($3 & 0xffff) << 32);
	}
|	'-' LCONST '-' LCONST
	{
		$$ = (-$2 & 0xffffffff) + (($4 & 0xffff) << 32);
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
