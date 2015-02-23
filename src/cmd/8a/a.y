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
package main

import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	. "cmd/internal/obj/i386"
)
%}

%union {
	sym *asm.Sym
	lval int64
	con2 struct {
		v1 int32
		v2 int32
	}
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
%token	<lval>	LTYPEC LTYPED LTYPEN LTYPER LTYPET LTYPES LTYPEM LTYPEI LTYPEG LTYPEXC
%token	<lval>	LTYPEX LTYPEPC LTYPEF LCONST LFP LPC LSB
%token	<lval>	LBREG LLREG LSREG LFREG LXREG
%token	<dval>	LFCONST
%token	<sval>	LSCONST LSP
%token	<sym>	LNAME LLAB LVAR
%type	<lval>	con expr pointer offset
%type	<addr>	mem imm reg nam rel rem rim rom omem nmem textsize
%type	<addr2>	nonnon nonrel nonrem rimnon rimrem remrim
%type	<addr2>	spec3 spec4 spec5 spec6 spec7 spec9 spec10 spec11 spec12
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
		if $1.Value != int64($3) {
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
|	spec1
|	spec2
|	LTYPEC spec3	{ outcode(int($1), &$2); }
|	LTYPEN spec4	{ outcode(int($1), &$2); }
|	LTYPES spec5	{ outcode(int($1), &$2); }
|	LTYPEM spec6	{ outcode(int($1), &$2); }
|	LTYPEI spec7	{ outcode(int($1), &$2); }
|	spec8
|	LTYPEXC spec9	{ outcode(int($1), &$2); }
|	LTYPEX spec10	{ outcode(int($1), &$2); }
|	LTYPEPC spec11	{ outcode(int($1), &$2); }
|	LTYPEF spec12	{ outcode(int($1), &$2); }

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
	LTYPED nam '/' con ',' imm
	{
		outcode(obj.ADATA, &Addr2{$2, $6})
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
	}

spec2:	/* TEXT */
	LTYPET mem ',' '$' textsize
	{
		asm.Settext($2.Sym);
		outcode(obj.ATEXT, &Addr2{$2, $5})
	}
|	LTYPET mem ',' con ',' '$' textsize
	{
		asm.Settext($2.Sym);
		outcode(obj.ATEXT, &Addr2{$2, $7})
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
	}

spec8:	/* GLOBL */
	LTYPEG mem ',' imm
	{
		asm.Settext($2.Sym);
		outcode(obj.AGLOBL, &Addr2{$2, $4})
	}
|	LTYPEG mem ',' con ',' imm
	{
		asm.Settext($2.Sym);
		outcode(obj.AGLOBL, &Addr2{$2, $6})
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
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
		$$.to.Type = obj.TYPE_INDIR
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
		if $$.from.Index != obj.TYPE_NONE {
			yyerror("dp shift with lhs index");
		}
		$$.from.Index = int16($5);
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
		if $$.to.Index != obj.TYPE_NONE {
			yyerror("dp move with lhs index");
		}
		$$.to.Index = int16($5);
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

spec9:	/* CMPPS/CMPPD */
	reg ',' rem ',' con
	{
		$$.from = $1;
		$$.to = $3;
		$$.to.Offset = $5;
	}

spec10:	/* PINSRD */
	imm ',' rem ',' reg
	{
		$$.from = $3;
		$$.to = $5;
		if $1.Type != obj.TYPE_CONST {
			yyerror("illegal constant")
		}
		$$.to.Offset = $1.Offset;
	}

spec11:	/* PCDATA */
	rim ',' rim
	{
		if $1.Type != obj.TYPE_CONST || $3.Type != obj.TYPE_CONST {
			yyerror("arguments to PCDATA must be integer constants");
		}
		$$.from = $1;
		$$.to = $3;
	}

spec12:	/* FUNCDATA */
	rim ',' rim
	{
		if $1.Type != obj.TYPE_CONST {
			yyerror("index for FUNCDATA must be integer constant");
		}
		if $3.Type != obj.TYPE_MEM || ($3.Name != obj.NAME_EXTERN && $3.Name != obj.NAME_STATIC) {
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
|	imm

rim:
	rem
|	imm

rel:
	con '(' LPC ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_BRANCH;
		$$.Offset = $1 + int64(asm.PC);
	}
|	LNAME offset
	{
		$1 = asm.LabelLookup($1);
		$$ = nullgen;
		if asm.Pass == 2 && $1.Type != LLAB {
			yyerror("undefined label: %s", $1.Labelname);
		}
		$$.Type = obj.TYPE_BRANCH;
		$$.Offset = $1.Value + $2;
	}

reg:
	LBREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1);
	}
|	LFREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1);
	}
|	LLREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1);
	}
|	LXREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1);
	}
|	LSP
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = REG_SP;
	}
|	LSREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1);
	}

imm:
	'$' con
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_CONST;
		$$.Offset = $2;
	}
|	'$' nam
	{
		$$ = $2;
		$$.Type = obj.TYPE_ADDR
		/*
		if($2.Type == D_AUTO || $2.Type == D_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.Sym.name);
		 */
	}
|	'$' LSCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_SCONST;
		$$.U.Sval = $2
	}
|	'$' LFCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_FCONST;
		$$.U.Dval = $2;
	}
|	'$' '(' LFCONST ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_FCONST;
		$$.U.Dval = $3;
	}
|	'$' '(' '-' LFCONST ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_FCONST;
		$$.U.Dval = -$4;
	}
|	'$' '-' LFCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_FCONST;
		$$.U.Dval = -$3;
	}

textsize:
	LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = $1;
		$$.U.Argsize = obj.ArgsSizeUnknown;
	}
|	'-' LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = -$2;
		$$.U.Argsize = obj.ArgsSizeUnknown;
	}
|	LCONST '-' LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = $1;
		$$.U.Argsize = int32($3);
	}
|	'-' LCONST '-' LCONST
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_TEXTSIZE;
		$$.Offset = -$2;
		$$.U.Argsize = int32($4);
	}


mem:
	omem
|	nmem

omem:
	con
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Offset = $1;
	}
|	con '(' LLREG ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = int16($3)
		$$.Offset = $1;
	}
|	con '(' LSP ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = REG_SP
		$$.Offset = $1;
	}
|	con '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Offset = $1;
		$$.Index = int16($3);
		$$.Scale = int8($5);
		checkscale($$.Scale);
	}
|	con '(' LLREG ')' '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = int16($3)
		$$.Offset = $1;
		$$.Index = int16($6);
		$$.Scale = int8($8);
		checkscale($$.Scale);
	}
|	con '(' LLREG ')' '(' LSREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = int16($3)
		$$.Offset = $1;
		$$.Index = int16($6);
		$$.Scale = int8($8);
		checkscale($$.Scale);
	}
|	'(' LLREG ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = int16($2);
	}
|	'(' LSP ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = REG_SP
	}
|	con '(' LSREG ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = int16($3)
		$$.Offset = $1;
	}
|	'(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Index = int16($2);
		$$.Scale = int8($4);
		checkscale($$.Scale);
	}
|	'(' LLREG ')' '(' LLREG '*' con ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Reg = int16($2)
		$$.Index = int16($5);
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
		$$.Index = int16($3);
		$$.Scale = int8($5);
		checkscale($$.Scale);
	}

nam:
	LNAME offset '(' pointer ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Name = int8($4);
		$$.Sym = obj.Linklookup(asm.Ctxt, $1.Name, 0);
		$$.Offset = $2;
	}
|	LNAME '<' '>' offset '(' LSB ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM
		$$.Name = obj.NAME_STATIC
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
		$$ = obj.NAME_AUTO;
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
