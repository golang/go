// cmd/9a/a.y from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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
#include "a.h"
%}
%union
{
	Sym	*sym;
	vlong	lval;
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
%token	<lval>	LMOVW LMOVB LABS LLOGW LSHW LADDW LCMP LCROP
%token	<lval>	LBRA LFMOV LFCONV LFCMP LFADD LFMA LTRAP LXORW
%token	<lval>	LNOP LEND LRETT LWORD LTEXT LDATA LRETRN
%token	<lval>	LCONST LSP LSB LFP LPC LCREG LFLUSH
%token	<lval>	LREG LFREG LR LCR LF LFPSCR
%token	<lval>	LLR LCTR LSPR LSPREG LSEG LMSR
%token	<lval>	LSCHED LXLD LXST LXOP LXMV
%token	<lval>	LRLWM LMOVMW LMOVEM LMOVFL LMTFSB LMA
%token	<dval>	LFCONST
%token	<sval>	LSCONST
%token	<sym>	LNAME LLAB LVAR
%type	<lval>	con expr pointer offset sreg
%type	<gen>	addr rreg regaddr name creg freg xlreg lr ctr
%type	<gen>	imm ximm fimm rel psr lcr cbit fpscr fpscrf msr mask
%%
prog:
|	prog line

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
|	LSCHED ';'
	{
		nosched = $1;
	}
|	';'
|	inst ';'
|	error ';'

inst:
/*
 * load ints and bytes
 */
	LMOVW rreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW addr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW regaddr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVB rreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVB addr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVB regaddr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * load floats
 */
|	LFMOV addr ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFMOV regaddr ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFMOV fimm ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFMOV freg ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFMOV freg ',' addr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFMOV freg ',' regaddr
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * store ints and bytes
 */
|	LMOVW rreg ',' addr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW rreg ',' regaddr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVB rreg ',' addr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVB rreg ',' regaddr
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * store floats
 */
|	LMOVW freg ',' addr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW freg ',' regaddr
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * floating point status
 */
|	LMOVW fpscr ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW freg ','  fpscr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW freg ',' imm ',' fpscr
	{
		outgcode($1, &$2, NREG, &$4, &$6);
	}
|	LMOVW fpscr ',' creg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW imm ',' fpscrf
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMTFSB imm ',' con
	{
		outcode($1, &$2, $4, &nullgen);
	}
/*
 * field moves (mtcrf)
 */
|	LMOVW rreg ',' imm ',' lcr
	{
		outgcode($1, &$2, NREG, &$4, &$6);
	}
|	LMOVW rreg ',' creg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW rreg ',' lcr
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * integer operations
 * logical instructions
 * shift instructions
 * unary instructions
 */
|	LADDW rreg ',' sreg ',' rreg
	{
		outcode($1, &$2, $4, &$6);
	}
|	LADDW imm ',' sreg ',' rreg
	{
		outcode($1, &$2, $4, &$6);
	}
|	LADDW rreg ',' imm ',' rreg
	{
		outgcode($1, &$2, NREG, &$4, &$6);
	}
|	LADDW rreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LADDW imm ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LLOGW rreg ',' sreg ',' rreg
	{
		outcode($1, &$2, $4, &$6);
	}
|	LLOGW rreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LSHW rreg ',' sreg ',' rreg
	{
		outcode($1, &$2, $4, &$6);
	}
|	LSHW rreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LSHW imm ',' sreg ',' rreg
	{
		outcode($1, &$2, $4, &$6);
	}
|	LSHW imm ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LABS rreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LABS rreg
	{
		outcode($1, &$2, NREG, &$2);
	}
/*
 * multiply-accumulate
 */
|	LMA rreg ',' sreg ',' rreg
	{
		outcode($1, &$2, $4, &$6);
	}
/*
 * move immediate: macro for cau+or, addi, addis, and other combinations
 */
|	LMOVW imm ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW ximm ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * condition register operations
 */
|	LCROP cbit ',' cbit
	{
		outcode($1, &$2, $4.reg, &$4);
	}
|	LCROP cbit ',' con ',' cbit
	{
		outcode($1, &$2, $4, &$6);
	}
/*
 * condition register moves
 * move from machine state register
 */
|	LMOVW creg ',' creg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW psr ',' creg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW lcr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW psr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW xlreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW rreg ',' xlreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW creg ',' psr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVW rreg ',' psr
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * branch, branch conditional
 * branch conditional register
 * branch conditional to count register
 */
|	LBRA rel
	{
		outcode($1, &nullgen, NREG, &$2);
	}
|	LBRA addr
	{
		outcode($1, &nullgen, NREG, &$2);
	}
|	LBRA '(' xlreg ')'
	{
		outcode($1, &nullgen, NREG, &$3);
	}
|	LBRA ',' rel
	{
		outcode($1, &nullgen, NREG, &$3);
	}
|	LBRA ',' addr
	{
		outcode($1, &nullgen, NREG, &$3);
	}
|	LBRA ',' '(' xlreg ')'
	{
		outcode($1, &nullgen, NREG, &$4);
	}
|	LBRA creg ',' rel
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LBRA creg ',' addr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LBRA creg ',' '(' xlreg ')'
	{
		outcode($1, &$2, NREG, &$5);
	}
|	LBRA con ',' rel
	{
		outcode($1, &nullgen, $2, &$4);
	}
|	LBRA con ',' addr
	{
		outcode($1, &nullgen, $2, &$4);
	}
|	LBRA con ',' '(' xlreg ')'
	{
		outcode($1, &nullgen, $2, &$5);
	}
|	LBRA con ',' con ',' rel
	{
		Gen g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = $2;
		outcode($1, &g, $4, &$6);
	}
|	LBRA con ',' con ',' addr
	{
		Gen g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = $2;
		outcode($1, &g, $4, &$6);
	}
|	LBRA con ',' con ',' '(' xlreg ')'
	{
		Gen g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = $2;
		outcode($1, &g, $4, &$7);
	}
/*
 * conditional trap
 */
|	LTRAP rreg ',' sreg
	{
		outcode($1, &$2, $4, &nullgen);
	}
|	LTRAP imm ',' sreg
	{
		outcode($1, &$2, $4, &nullgen);
	}
|	LTRAP rreg comma
	{
		outcode($1, &$2, NREG, &nullgen);
	}
|	LTRAP comma
	{
		outcode($1, &nullgen, NREG, &nullgen);
	}
/*
 * floating point operate
 */
|	LFCONV freg ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFADD freg ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFADD freg ',' freg ',' freg
	{
		outcode($1, &$2, $4.reg, &$6);
	}
|	LFMA freg ',' freg ',' freg ',' freg
	{
		outgcode($1, &$2, $4.reg, &$6, &$8);
	}
|	LFCMP freg ',' freg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LFCMP freg ',' freg ',' creg
	{
		outcode($1, &$2, $6.reg, &$4);
	}
/*
 * CMP
 */
|	LCMP rreg ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LCMP rreg ',' imm
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LCMP rreg ',' rreg ',' creg
	{
		outcode($1, &$2, $6.reg, &$4);
	}
|	LCMP rreg ',' imm ',' creg
	{
		outcode($1, &$2, $6.reg, &$4);
	}
/*
 * rotate and mask
 */
|	LRLWM  imm ',' rreg ',' imm ',' rreg
	{
		outgcode($1, &$2, $4.reg, &$6, &$8);
	}
|	LRLWM  imm ',' rreg ',' mask ',' rreg
	{
		outgcode($1, &$2, $4.reg, &$6, &$8);
	}
|	LRLWM  rreg ',' rreg ',' imm ',' rreg
	{
		outgcode($1, &$2, $4.reg, &$6, &$8);
	}
|	LRLWM  rreg ',' rreg ',' mask ',' rreg
	{
		outgcode($1, &$2, $4.reg, &$6, &$8);
	}
/*
 * load/store multiple
 */
|	LMOVMW addr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LMOVMW rreg ',' addr
	{
		outcode($1, &$2, NREG, &$4);
	}
/*
 * various indexed load/store
 * indexed unary (eg, cache clear)
 */
|	LXLD regaddr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LXLD regaddr ',' imm ',' rreg
	{
		outgcode($1, &$2, NREG, &$4, &$6);
	}
|	LXST rreg ',' regaddr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LXST rreg ',' imm ',' regaddr
	{
		outgcode($1, &$2, NREG, &$4, &$6);
	}
|	LXMV regaddr ',' rreg
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LXMV rreg ',' regaddr
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LXOP regaddr
	{
		outcode($1, &$2, NREG, &nullgen);
	}
/*
 * NOP
 */
|	LNOP comma
	{
		outcode($1, &nullgen, NREG, &nullgen);
	}
|	LNOP rreg comma
	{
		outcode($1, &$2, NREG, &nullgen);
	}
|	LNOP freg comma
	{
		outcode($1, &$2, NREG, &nullgen);
	}
|	LNOP ',' rreg
	{
		outcode($1, &nullgen, NREG, &$3);
	}
|	LNOP ',' freg
	{
		outcode($1, &nullgen, NREG, &$3);
	}
/*
 * word
 */
|	LWORD imm comma
	{
		if($1 == ADWORD && $2.type == D_CONST)
			$2.type = D_DCONST;
		outcode($1, &$2, NREG, &nullgen);
	}
|	LWORD ximm comma
	{
		if($1 == ADWORD && $2.type == D_CONST)
			$2.type = D_DCONST;
		outcode($1, &$2, NREG, &nullgen);
	}
/*
 * END
 */
|	LEND comma
	{
		outcode($1, &nullgen, NREG, &nullgen);
	}
/*
 * TEXT/GLOBL
 */
|	LTEXT name ',' imm
	{
		outcode($1, &$2, NREG, &$4);
	}
|	LTEXT name ',' con ',' imm
	{
		outcode($1, &$2, $4, &$6);
	}
|	LTEXT name ',' imm ':' imm
	{
		outgcode($1, &$2, NREG, &$6, &$4);
	}
|	LTEXT name ',' con ',' imm ':' imm
	{
		outgcode($1, &$2, $4, &$8, &$6);
	}
/*
 * DATA
 */
|	LDATA name '/' con ',' imm
	{
		outcode($1, &$2, $4, &$6);
	}
|	LDATA name '/' con ',' ximm
	{
		outcode($1, &$2, $4, &$6);
	}
|	LDATA name '/' con ',' fimm
	{
		outcode($1, &$2, $4, &$6);
	}
/*
 * RETURN
 */
|	LRETRN	comma
	{
		outcode($1, &nullgen, NREG, &nullgen);
	}

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

rreg:
	sreg
	{
		$$ = nullgen;
		$$.type = D_REG;
		$$.reg = $1;
	}

xlreg:
	lr
|	ctr

lr:
	LLR
	{
		$$ = nullgen;
		$$.type = D_SPR;
		$$.offset = $1;
	}

lcr:
	LCR
	{
		$$ = nullgen;
		$$.type = D_CREG;
		$$.reg = NREG;	/* whole register */
	}

ctr:
	LCTR
	{
		$$ = nullgen;
		$$.type = D_SPR;
		$$.offset = $1;
	}

msr:
	LMSR
	{
		$$ = nullgen;
		$$.type = D_MSR;
	}

psr:
	LSPREG
	{
		$$ = nullgen;
		$$.type = D_SPR;
		$$.offset = $1;
	}
|	LSPR '(' con ')'
	{
		$$ = nullgen;
		$$.type = $1;
		$$.offset = $3;
	}
|	msr

fpscr:
	LFPSCR
	{
		$$ = nullgen;
		$$.type = D_FPSCR;
		$$.reg = NREG;
	}

fpscrf:
	LFPSCR '(' con ')'
	{
		$$ = nullgen;
		$$.type = D_FPSCR;
		$$.reg = $3;
	}

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

creg:
	LCREG
	{
		$$ = nullgen;
		$$.type = D_CREG;
		$$.reg = $1;
	}
|	LCR '(' con ')'
	{
		$$ = nullgen;
		$$.type = D_CREG;
		$$.reg = $3;
	}


cbit:	con
	{
		$$ = nullgen;
		$$.type = D_REG;
		$$.reg = $1;
	}

mask:
	con ',' con
	{
		int mb, me;
		ulong v;

		$$ = nullgen;
		$$.type = D_CONST;
		mb = $1;
		me = $3;
		if(mb < 0 || mb > 31 || me < 0 || me > 31){
			yyerror("illegal mask start/end value(s)");
			mb = me = 0;
		}
		if(mb <= me)
			v = ((ulong)~0L>>mb) & (~0L<<(31-me));
		else
			v = ~(((ulong)~0L>>(me+1)) & (~0L<<(31-(mb-1))));
		$$.offset = v;
	}

ximm:
	'$' addr
	{
		$$ = $2;
		$$.type = D_CONST;
	}
|	'$' LSCONST
	{
		$$ = nullgen;
		$$.type = D_SCONST;
		memcpy($$.sval, $2, sizeof($$.sval));
	}

fimm:
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

imm:	'$' con
	{
		$$ = nullgen;
		$$.type = D_CONST;
		$$.offset = $2;
	}

sreg:
	LREG
|	LR '(' con ')'
	{
		if($$ < 0 || $$ >= NREG)
			print("register value out of range\n");
		$$ = $3;
	}

regaddr:
	'(' sreg ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.reg = $2;
		$$.offset = 0;
	}
|	'(' sreg '+' sreg ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.reg = $2;
		$$.xreg = $4;
		$$.offset = 0;
	}

addr:
	name
|	con '(' sreg ')'
	{
		$$ = nullgen;
		$$.type = D_OREG;
		$$.reg = $3;
		$$.offset = $1;
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

comma:
|	','

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
