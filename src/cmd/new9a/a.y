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
package main

import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	. "cmd/internal/obj/ppc64"
)
%}

%union
{
	sym *asm.Sym
	lval int64
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
%token	<lval>	LMOVW LMOVB LABS LLOGW LSHW LADDW LCMP LCROP
%token	<lval>	LBRA LFMOV LFCONV LFCMP LFADD LFMA LTRAP LXORW
%token	<lval>	LNOP LEND LRETT LWORD LTEXT LDATA LGLOBL LRETRN
%token	<lval>	LCONST LSP LSB LFP LPC LCREG LFLUSH
%token	<lval>	LREG LFREG LR LCR LF LFPSCR
%token	<lval>	LLR LCTR LSPR LSPREG LSEG LMSR
%token	<lval>	LPCDAT LFUNCDAT LSCHED LXLD LXST LXOP LXMV
%token	<lval>	LRLWM LMOVMW LMOVEM LMOVFL LMTFSB LMA
%token	<dval>	LFCONST
%token	<sval>	LSCONST
%token	<sym>	LNAME LLAB LVAR
%type	<lval>	con expr pointer offset sreg
%type	<addr>	addr rreg regaddr name creg freg xlreg lr ctr textsize
%type	<addr>	imm ximm fimm rel psr lcr cbit fpscr msr mask
%%
prog:
|	prog
	{
		stmtline = asm.Lineno
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
		$1.Value = int64(asm.PC);
	}
	line
|	LNAME '=' expr ';'
	{
		$1.Type = LVAR;
		$1.Value = $3;
	}
|	LVAR '=' expr ';'
	{
		if $1.Value != $3 {
			yyerror("redeclaration of %s", $1.Name)
		}
		$1.Value = $3;
	}
|	LSCHED ';'
	{
		nosched = int($1);
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
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW addr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW regaddr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVB rreg ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVB addr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVB regaddr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * load floats
 */
|	LFMOV addr ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFMOV regaddr ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFMOV fimm ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFMOV freg ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFMOV freg ',' addr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFMOV freg ',' regaddr
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * store ints and bytes
 */
|	LMOVW rreg ',' addr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW rreg ',' regaddr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVB rreg ',' addr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVB rreg ',' regaddr
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * store floats
 */
|	LMOVW freg ',' addr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW freg ',' regaddr
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * floating point status
 */
|	LMOVW fpscr ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW freg ','  fpscr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW freg ',' imm ',' fpscr
	{
		outgcode(int($1), &$2, 0, &$4, &$6);
	}
|	LMOVW fpscr ',' creg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMTFSB imm ',' con
	{
		outcode(int($1), &$2, int($4), &nullgen);
	}
/*
 * field moves (mtcrf)
 */
|	LMOVW rreg ',' imm ',' lcr
	{
		outgcode(int($1), &$2, 0, &$4, &$6);
	}
|	LMOVW rreg ',' creg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW rreg ',' lcr
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * integer operations
 * logical instructions
 * shift instructions
 * unary instructions
 */
|	LADDW rreg ',' sreg ',' rreg
	{
		outcode(int($1), &$2, int($4), &$6);
	}
|	LADDW imm ',' sreg ',' rreg
	{
		outcode(int($1), &$2, int($4), &$6);
	}
|	LADDW rreg ',' imm ',' rreg
	{
		outgcode(int($1), &$2, 0, &$4, &$6);
	}
|	LADDW rreg ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LADDW imm ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LLOGW rreg ',' sreg ',' rreg
	{
		outcode(int($1), &$2, int($4), &$6);
	}
|	LLOGW rreg ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LSHW rreg ',' sreg ',' rreg
	{
		outcode(int($1), &$2, int($4), &$6);
	}
|	LSHW rreg ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LSHW imm ',' sreg ',' rreg
	{
		outcode(int($1), &$2, int($4), &$6);
	}
|	LSHW imm ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LABS rreg ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LABS rreg
	{
		outcode(int($1), &$2, 0, &$2);
	}
/*
 * multiply-accumulate
 */
|	LMA rreg ',' sreg ',' rreg
	{
		outcode(int($1), &$2, int($4), &$6);
	}
/*
 * move immediate: macro for cau+or, addi, addis, and other combinations
 */
|	LMOVW imm ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW ximm ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * condition register operations
 */
|	LCROP cbit ',' cbit
	{
		outcode(int($1), &$2, int($4.Reg), &$4);
	}
|	LCROP cbit ',' con ',' cbit
	{
		outcode(int($1), &$2, int($4), &$6);
	}
/*
 * condition register moves
 * move from machine state register
 */
|	LMOVW creg ',' creg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW psr ',' creg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW lcr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW psr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW xlreg ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW rreg ',' xlreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW creg ',' psr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVW rreg ',' psr
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * branch, branch conditional
 * branch conditional register
 * branch conditional to count register
 */
|	LBRA rel
	{
		outcode(int($1), &nullgen, 0, &$2);
	}
|	LBRA addr
	{
		outcode(int($1), &nullgen, 0, &$2);
	}
|	LBRA '(' xlreg ')'
	{
		outcode(int($1), &nullgen, 0, &$3);
	}
|	LBRA ',' rel
	{
		outcode(int($1), &nullgen, 0, &$3);
	}
|	LBRA ',' addr
	{
		outcode(int($1), &nullgen, 0, &$3);
	}
|	LBRA ',' '(' xlreg ')'
	{
		outcode(int($1), &nullgen, 0, &$4);
	}
|	LBRA creg ',' rel
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LBRA creg ',' addr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LBRA creg ',' '(' xlreg ')'
	{
		outcode(int($1), &$2, 0, &$5);
	}
|	LBRA con ',' rel
	{
		outcode(int($1), &nullgen, int($2), &$4);
	}
|	LBRA con ',' addr
	{
		outcode(int($1), &nullgen, int($2), &$4);
	}
|	LBRA con ',' '(' xlreg ')'
	{
		outcode(int($1), &nullgen, int($2), &$5);
	}
|	LBRA con ',' con ',' rel
	{
		var g obj.Addr
		g = nullgen;
		g.Type = obj.TYPE_CONST;
		g.Offset = $2;
		outcode(int($1), &g, int(REG_R0+$4), &$6);
	}
|	LBRA con ',' con ',' addr
	{
		var g obj.Addr
		g = nullgen;
		g.Type = obj.TYPE_CONST;
		g.Offset = $2;
		outcode(int($1), &g, int(REG_R0+$4), &$6);
	}
|	LBRA con ',' con ',' '(' xlreg ')'
	{
		var g obj.Addr
		g = nullgen;
		g.Type = obj.TYPE_CONST;
		g.Offset = $2;
		outcode(int($1), &g, int(REG_R0+$4), &$7);
	}
/*
 * conditional trap
 */
|	LTRAP rreg ',' sreg
	{
		outcode(int($1), &$2, int($4), &nullgen);
	}
|	LTRAP imm ',' sreg
	{
		outcode(int($1), &$2, int($4), &nullgen);
	}
|	LTRAP rreg comma
	{
		outcode(int($1), &$2, 0, &nullgen);
	}
|	LTRAP comma
	{
		outcode(int($1), &nullgen, 0, &nullgen);
	}
/*
 * floating point operate
 */
|	LFCONV freg ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFADD freg ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFADD freg ',' freg ',' freg
	{
		outcode(int($1), &$2, int($4.Reg), &$6);
	}
|	LFMA freg ',' freg ',' freg ',' freg
	{
		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
	}
|	LFCMP freg ',' freg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LFCMP freg ',' freg ',' creg
	{
		outcode(int($1), &$2, int($6.Reg), &$4);
	}
/*
 * CMP
 */
|	LCMP rreg ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LCMP rreg ',' imm
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LCMP rreg ',' rreg ',' creg
	{
		outcode(int($1), &$2, int($6.Reg), &$4);
	}
|	LCMP rreg ',' imm ',' creg
	{
		outcode(int($1), &$2, int($6.Reg), &$4);
	}
/*
 * rotate and mask
 */
|	LRLWM  imm ',' rreg ',' imm ',' rreg
	{
		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
	}
|	LRLWM  imm ',' rreg ',' mask ',' rreg
	{
		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
	}
|	LRLWM  rreg ',' rreg ',' imm ',' rreg
	{
		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
	}
|	LRLWM  rreg ',' rreg ',' mask ',' rreg
	{
		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
	}
/*
 * load/store multiple
 */
|	LMOVMW addr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LMOVMW rreg ',' addr
	{
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * various indexed load/store
 * indexed unary (eg, cache clear)
 */
|	LXLD regaddr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LXLD regaddr ',' imm ',' rreg
	{
		outgcode(int($1), &$2, 0, &$4, &$6);
	}
|	LXST rreg ',' regaddr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LXST rreg ',' imm ',' regaddr
	{
		outgcode(int($1), &$2, 0, &$4, &$6);
	}
|	LXMV regaddr ',' rreg
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LXMV rreg ',' regaddr
	{
		outcode(int($1), &$2, 0, &$4);
	}
|	LXOP regaddr
	{
		outcode(int($1), &$2, 0, &nullgen);
	}
/*
 * NOP
 */
|	LNOP comma
	{
		outcode(int($1), &nullgen, 0, &nullgen);
	}
|	LNOP rreg comma
	{
		outcode(int($1), &$2, 0, &nullgen);
	}
|	LNOP freg comma
	{
		outcode(int($1), &$2, 0, &nullgen);
	}
|	LNOP ',' rreg
	{
		outcode(int($1), &nullgen, 0, &$3);
	}
|	LNOP ',' freg
	{
		outcode(int($1), &nullgen, 0, &$3);
	}
|	LNOP imm /* SYSCALL $num: load $num to R0 before syscall and restore R0 to 0 afterwards. */
	{
		outcode(int($1), &$2, 0, &nullgen);
	}
/*
 * word
 */
|	LWORD imm comma
	{
		outcode(int($1), &$2, 0, &nullgen);
	}
|	LWORD ximm comma
	{
		outcode(int($1), &$2, 0, &nullgen);
	}
/*
 * PCDATA
 */
|	LPCDAT imm ',' imm
	{
		if $2.Type != obj.TYPE_CONST || $4.Type != obj.TYPE_CONST {
			yyerror("arguments to PCDATA must be integer constants")
		}
		outcode(int($1), &$2, 0, &$4);
	}
/*
 * FUNCDATA
 */
|	LFUNCDAT imm ',' addr
	{
		if $2.Type != obj.TYPE_CONST {
			yyerror("index for FUNCDATA must be integer constant")
		}
		if $4.Type != obj.TYPE_MEM || ($4.Name != obj.NAME_EXTERN && $4.Name != obj.NAME_STATIC) {
			yyerror("value for FUNCDATA must be symbol reference")
		}
 		outcode(int($1), &$2, 0, &$4);
	}
/*
 * END
 */
|	LEND comma
	{
		outcode(int($1), &nullgen, 0, &nullgen);
	}
/*
 * TEXT
 */
|	LTEXT name ',' '$' textsize
	{
		asm.Settext($2.Sym);
		outcode(int($1), &$2, 0, &$5);
	}
|	LTEXT name ',' con ',' '$' textsize
	{
		asm.Settext($2.Sym);
		outcode(int($1), &$2, int($4), &$7);
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
	}
/*
 * GLOBL
 */
|	LGLOBL name ',' imm
	{
		asm.Settext($2.Sym)
		outcode(int($1), &$2, 0, &$4)
	}
|	LGLOBL name ',' con ',' imm
	{
		asm.Settext($2.Sym)
		outcode(int($1), &$2, 0, &$6)
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
	}

/*
 * DATA
 */
|	LDATA name '/' con ',' imm
	{
		outcode(int($1), &$2, 0, &$6);
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
	}
|	LDATA name '/' con ',' ximm
	{
		outcode(int($1), &$2, 0, &$6);
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
	}
|	LDATA name '/' con ',' fimm
	{
		outcode(int($1), &$2, 0, &$6);
		if asm.Pass > 1 {
			lastpc.From3.Type = obj.TYPE_CONST
			lastpc.From3.Offset = $4
		}
	}
/*
 * RETURN
 */
|	LRETRN	comma
	{
		outcode(int($1), &nullgen, 0, &nullgen);
	}

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
			yyerror("undefined label: %s", $1.Labelname)
		}
		$$.Type = obj.TYPE_BRANCH;
		$$.Offset = $1.Value + $2;
	}

rreg:
	sreg
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}

xlreg:
	lr
|	ctr

lr:
	LLR
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}

lcr:
	LCR
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);	/* whole register */
	}

ctr:
	LCTR
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}

msr:
	LMSR
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1)
	}

psr:
	LSPREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}
|	LSPR '(' con ')'
	{
		if $3 < 0 || $3 >= 1024 {
			yyerror("SPR/DCR out of range")
		}
		$$ = nullgen;
		$$.Type = obj.TYPE_REG
		$$.Reg = int16($1 + $3);
	}
|	msr

fpscr:
	LFPSCR
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}

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

creg:
	LCREG
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}
|	LCR '(' con ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16(REG_C0 + $3);
	}


cbit:	con
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_REG;
		$$.Reg = int16($1);
	}

mask:
	con ',' con
	{
		var mb, me int
		var v uint32

		$$ = nullgen;
		$$.Type = obj.TYPE_CONST;
		mb = int($1);
		me = int($3);
		if(mb < 0 || mb > 31 || me < 0 || me > 31){
			yyerror("illegal mask start/end value(s)");
			mb = 0
			me = 0;
		}
		if mb <= me {
			v = (^uint32(0)>>uint(mb)) & (^uint32(0)<<uint(31-me))
		} else {
			v = (^uint32(0)>>uint(me+1)) & (^uint32(0)<<uint(31-(mb-1)))
		}
		$$.Offset = int64(v);
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

ximm:
	'$' addr
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

fimm:
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

imm:	'$' con
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_CONST;
		$$.Offset = $2;
	}

sreg:
	LREG
|	LR '(' con ')'
	{
		if $$ < 0 || $$ >= NREG {
			print("register value out of range\n")
		}
		$$ = REG_R0 + $3;
	}

regaddr:
	'(' sreg ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Reg = int16($2);
		$$.Offset = 0;
	}
|	'(' sreg '+' sreg ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Reg = int16($2);
		$$.Scale = int8($4);
		$$.Offset = 0;
	}

addr:
	name
|	con '(' sreg ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Reg = int16($3);
		$$.Offset = $1;
	}

name:
	con '(' pointer ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Name = int8($3);
		$$.Sym = nil;
		$$.Offset = $1;
	}
|	LNAME offset '(' pointer ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Name = int8($4);
		$$.Sym = obj.Linklookup(asm.Ctxt, $1.Name, 0);
		$$.Offset = $2;
	}
|	LNAME '<' '>' offset '(' LSB ')'
	{
		$$ = nullgen;
		$$.Type = obj.TYPE_MEM;
		$$.Name = obj.NAME_STATIC;
		$$.Sym = obj.Linklookup(asm.Ctxt, $1.Name, 1);
		$$.Offset = $4;
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
