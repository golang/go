// Inferno utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
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

// Instruction layout.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>
#include "../cmd/6l/6.out.h"
#include "../runtime/stack.h"

enum
{
	MaxAlign = 32,	// max data alignment
	
	// Loop alignment constants:
	// want to align loop entry to LoopAlign-byte boundary,
	// and willing to insert at most MaxLoopPad bytes of NOP to do so.
	// We define a loop entry as the target of a backward jump.
	//
	// gcc uses MaxLoopPad = 10 for its 'generic x86-64' config,
	// and it aligns all jump targets, not just backward jump targets.
	//
	// As of 6/1/2012, the effect of setting MaxLoopPad = 10 here
	// is very slight but negative, so the alignment is disabled by
	// setting MaxLoopPad = 0. The code is here for reference and
	// for future experiments.
	// 
	LoopAlign = 16,
	MaxLoopPad = 0,

	FuncAlign = 16
};

typedef	struct	Optab	Optab;
typedef	struct	Movtab	Movtab;

struct	Optab
{
	short	as;
	uchar*	ytab;
	uchar	prefix;
	uchar	op[23];
};
struct	Movtab
{
	short	as;
	uchar	ft;
	uchar	tt;
	uchar	code;
	uchar	op[4];
};

enum
{
	Yxxx		= 0,
	Ynone,
	Yi0,
	Yi1,
	Yi8,
	Ys32,
	Yi32,
	Yi64,
	Yiauto,
	Yal,
	Ycl,
	Yax,
	Ycx,
	Yrb,
	Yrl,
	Yrf,
	Yf0,
	Yrx,
	Ymb,
	Yml,
	Ym,
	Ybr,
	Ycol,

	Ycs,	Yss,	Yds,	Yes,	Yfs,	Ygs,
	Ygdtr,	Yidtr,	Yldtr,	Ymsw,	Ytask,
	Ycr0,	Ycr1,	Ycr2,	Ycr3,	Ycr4,	Ycr5,	Ycr6,	Ycr7,	Ycr8,
	Ydr0,	Ydr1,	Ydr2,	Ydr3,	Ydr4,	Ydr5,	Ydr6,	Ydr7,
	Ytr0,	Ytr1,	Ytr2,	Ytr3,	Ytr4,	Ytr5,	Ytr6,	Ytr7,	Yrl32,	Yrl64,
	Ymr, Ymm,
	Yxr, Yxm,
	Ytls,
	Ytextsize,
	Ymax,

	Zxxx		= 0,

	Zlit,
	Zlitm_r,
	Z_rp,
	Zbr,
	Zcall,
	Zcallindreg,
	Zib_,
	Zib_rp,
	Zibo_m,
	Zibo_m_xm,
	Zil_,
	Zil_rp,
	Ziq_rp,
	Zilo_m,
	Ziqo_m,
	Zjmp,
	Zloop,
	Zo_iw,
	Zm_o,
	Zm_r,
	Zm2_r,
	Zm_r_xm,
	Zm_r_i_xm,
	Zm_r_3d,
	Zm_r_xm_nr,
	Zr_m_xm_nr,
	Zibm_r,	/* mmx1,mmx2/mem64,imm8 */
	Zmb_r,
	Zaut_r,
	Zo_m,
	Zo_m64,
	Zpseudo,
	Zr_m,
	Zr_m_xm,
	Zr_m_i_xm,
	Zrp_,
	Z_ib,
	Z_il,
	Zm_ibo,
	Zm_ilo,
	Zib_rr,
	Zil_rr,
	Zclr,
	Zbyte,
	Zmax,

	Px		= 0,
	P32		= 0x32,	/* 32-bit only */
	Pe		= 0x66,	/* operand escape */
	Pm		= 0x0f,	/* 2byte opcode escape */
	Pq		= 0xff,	/* both escapes: 66 0f */
	Pb		= 0xfe,	/* byte operands */
	Pf2		= 0xf2,	/* xmm escape 1: f2 0f */
	Pf3		= 0xf3,	/* xmm escape 2: f3 0f */
	Pq3		= 0x67, /* xmm escape 3: 66 48 0f */
	Pw		= 0x48,	/* Rex.w */
	Py		= 0x80,	/* defaults to 64-bit mode */

	Rxf		= 1<<9,	/* internal flag for Rxr on from */
	Rxt		= 1<<8,	/* internal flag for Rxr on to */
	Rxw		= 1<<3,	/* =1, 64-bit operand size */
	Rxr		= 1<<2,	/* extend modrm reg */
	Rxx		= 1<<1,	/* extend sib index */
	Rxb		= 1<<0,	/* extend modrm r/m, sib base, or opcode reg */

	Maxand	= 10,		/* in -a output width of the byte codes */
};

static uchar ycover[Ymax*Ymax];
static	int	reg[MAXREG];
static	int	regrex[MAXREG+1];
static	void	asmins(Link *ctxt, Prog *p);

static uchar	ynone[] =
{
	Ynone,	Ynone,	Zlit,	1,
	0
};
static uchar	ytext[] =
{
	Ymb,	Ytextsize,	Zpseudo,1,
	0
};
static uchar	ynop[] =
{
	Ynone,	Ynone,	Zpseudo,0,
	Ynone,	Yiauto,	Zpseudo,0,
	Ynone,	Yml,	Zpseudo,0,
	Ynone,	Yrf,	Zpseudo,0,
	Ynone,	Yxr,	Zpseudo,0,
	Yiauto,	Ynone,	Zpseudo,0,
	Yml,	Ynone,	Zpseudo,0,
	Yrf,	Ynone,	Zpseudo,0,
	Yxr,	Ynone,	Zpseudo,1,
	0
};
static uchar	yfuncdata[] =
{
	Yi32,	Ym,	Zpseudo,	0,
	0
};
static uchar	ypcdata[] = 
{
	Yi32,	Yi32,	Zpseudo,	0,
	0
};
static uchar	yxorb[] =
{
	Yi32,	Yal,	Zib_,	1,
	Yi32,	Ymb,	Zibo_m,	2,
	Yrb,	Ymb,	Zr_m,	1,
	Ymb,	Yrb,	Zm_r,	1,
	0
};
static uchar	yxorl[] =
{
	Yi8,	Yml,	Zibo_m,	2,
	Yi32,	Yax,	Zil_,	1,
	Yi32,	Yml,	Zilo_m,	2,
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	0
};
static uchar	yaddl[] =
{
	Yi8,	Yml,	Zibo_m,	2,
	Yi32,	Yax,	Zil_,	1,
	Yi32,	Yml,	Zilo_m,	2,
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	0
};
static uchar	yincb[] =
{
	Ynone,	Ymb,	Zo_m,	2,
	0
};
static uchar	yincw[] =
{
	Ynone,	Yml,	Zo_m,	2,
	0
};
static uchar	yincl[] =
{
	Ynone,	Yml,	Zo_m,	2,
	0
};
static uchar	ycmpb[] =
{
	Yal,	Yi32,	Z_ib,	1,
	Ymb,	Yi32,	Zm_ibo,	2,
	Ymb,	Yrb,	Zm_r,	1,
	Yrb,	Ymb,	Zr_m,	1,
	0
};
static uchar	ycmpl[] =
{
	Yml,	Yi8,	Zm_ibo,	2,
	Yax,	Yi32,	Z_il,	1,
	Yml,	Yi32,	Zm_ilo,	2,
	Yml,	Yrl,	Zm_r,	1,
	Yrl,	Yml,	Zr_m,	1,
	0
};
static uchar	yshb[] =
{
	Yi1,	Ymb,	Zo_m,	2,
	Yi32,	Ymb,	Zibo_m,	2,
	Ycx,	Ymb,	Zo_m,	2,
	0
};
static uchar	yshl[] =
{
	Yi1,	Yml,	Zo_m,	2,
	Yi32,	Yml,	Zibo_m,	2,
	Ycl,	Yml,	Zo_m,	2,
	Ycx,	Yml,	Zo_m,	2,
	0
};
static uchar	ytestb[] =
{
	Yi32,	Yal,	Zib_,	1,
	Yi32,	Ymb,	Zibo_m,	2,
	Yrb,	Ymb,	Zr_m,	1,
	Ymb,	Yrb,	Zm_r,	1,
	0
};
static uchar	ytestl[] =
{
	Yi32,	Yax,	Zil_,	1,
	Yi32,	Yml,	Zilo_m,	2,
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	0
};
static uchar	ymovb[] =
{
	Yrb,	Ymb,	Zr_m,	1,
	Ymb,	Yrb,	Zm_r,	1,
	Yi32,	Yrb,	Zib_rp,	1,
	Yi32,	Ymb,	Zibo_m,	2,
	0
};
static uchar	ymbs[] =
{
	Ymb,	Ynone,	Zm_o,	2,
	0
};
static uchar	ybtl[] =
{
	Yi8,	Yml,	Zibo_m,	2,
	Yrl,	Yml,	Zr_m,	1,
	0
};
static uchar	ymovw[] =
{
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	Yi0,	Yrl,	Zclr,	1,
	Yi32,	Yrl,	Zil_rp,	1,
	Yi32,	Yml,	Zilo_m,	2,
	Yiauto,	Yrl,	Zaut_r,	2,
	0
};
static uchar	ymovl[] =
{
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	Yi0,	Yrl,	Zclr,	1,
	Yi32,	Yrl,	Zil_rp,	1,
	Yi32,	Yml,	Zilo_m,	2,
	Yml,	Ymr,	Zm_r_xm,	1,	// MMX MOVD
	Ymr,	Yml,	Zr_m_xm,	1,	// MMX MOVD
	Yml,	Yxr,	Zm_r_xm,	2,	// XMM MOVD (32 bit)
	Yxr,	Yml,	Zr_m_xm,	2,	// XMM MOVD (32 bit)
	Yiauto,	Yrl,	Zaut_r,	2,
	0
};
static uchar	yret[] =
{
	Ynone,	Ynone,	Zo_iw,	1,
	Yi32,	Ynone,	Zo_iw,	1,
	0
};
static uchar	ymovq[] =
{
	Yrl,	Yml,	Zr_m,	1,	// 0x89
	Yml,	Yrl,	Zm_r,	1,	// 0x8b
	Yi0,	Yrl,	Zclr,	1,	// 0x31
	Ys32,	Yrl,	Zilo_m,	2,	// 32 bit signed 0xc7,(0)
	Yi64,	Yrl,	Ziq_rp,	1,	// 0xb8 -- 32/64 bit immediate
	Yi32,	Yml,	Zilo_m,	2,	// 0xc7,(0)
	Ym,	Ymr,	Zm_r_xm_nr,	1,	// MMX MOVQ (shorter encoding)
	Ymr,	Ym,	Zr_m_xm_nr,	1,	// MMX MOVQ
	Ymm,	Ymr,	Zm_r_xm,	1,	// MMX MOVD
	Ymr,	Ymm,	Zr_m_xm,	1,	// MMX MOVD
	Yxr,	Ymr,	Zm_r_xm_nr,	2,	// MOVDQ2Q
	Yxm,	Yxr,	Zm_r_xm_nr,	2, // MOVQ xmm1/m64 -> xmm2
	Yxr,	Yxm,	Zr_m_xm_nr,	2, // MOVQ xmm1 -> xmm2/m64
	Yml,	Yxr,	Zm_r_xm,	2,	// MOVD xmm load
	Yxr,	Yml,	Zr_m_xm,	2,	// MOVD xmm store
	Yiauto,	Yrl,	Zaut_r,	2,	// built-in LEAQ
	0
};
static uchar	ym_rl[] =
{
	Ym,	Yrl,	Zm_r,	1,
	0
};
static uchar	yrl_m[] =
{
	Yrl,	Ym,	Zr_m,	1,
	0
};
static uchar	ymb_rl[] =
{
	Ymb,	Yrl,	Zmb_r,	1,
	0
};
static uchar	yml_rl[] =
{
	Yml,	Yrl,	Zm_r,	1,
	0
};
static uchar	yrl_ml[] =
{
	Yrl,	Yml,	Zr_m,	1,
	0
};
static uchar	yml_mb[] =
{
	Yrb,	Ymb,	Zr_m,	1,
	Ymb,	Yrb,	Zm_r,	1,
	0
};
static uchar	yrb_mb[] =
{
	Yrb,	Ymb,	Zr_m,	1,
	0
};
static uchar	yxchg[] =
{
	Yax,	Yrl,	Z_rp,	1,
	Yrl,	Yax,	Zrp_,	1,
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	0
};
static uchar	ydivl[] =
{
	Yml,	Ynone,	Zm_o,	2,
	0
};
static uchar	ydivb[] =
{
	Ymb,	Ynone,	Zm_o,	2,
	0
};
static uchar	yimul[] =
{
	Yml,	Ynone,	Zm_o,	2,
	Yi8,	Yrl,	Zib_rr,	1,
	Yi32,	Yrl,	Zil_rr,	1,
	Yml,	Yrl,	Zm_r,	2,
	0
};
static uchar	yimul3[] =
{
	Yml,	Yrl,	Zibm_r,	2,
	0
};
static uchar	ybyte[] =
{
	Yi64,	Ynone,	Zbyte,	1,
	0
};
static uchar	yin[] =
{
	Yi32,	Ynone,	Zib_,	1,
	Ynone,	Ynone,	Zlit,	1,
	0
};
static uchar	yint[] =
{
	Yi32,	Ynone,	Zib_,	1,
	0
};
static uchar	ypushl[] =
{
	Yrl,	Ynone,	Zrp_,	1,
	Ym,	Ynone,	Zm_o,	2,
	Yi8,	Ynone,	Zib_,	1,
	Yi32,	Ynone,	Zil_,	1,
	0
};
static uchar	ypopl[] =
{
	Ynone,	Yrl,	Z_rp,	1,
	Ynone,	Ym,	Zo_m,	2,
	0
};
static uchar	ybswap[] =
{
	Ynone,	Yrl,	Z_rp,	2,
	0,
};
static uchar	yscond[] =
{
	Ynone,	Ymb,	Zo_m,	2,
	0
};
static uchar	yjcond[] =
{
	Ynone,	Ybr,	Zbr,	0,
	Yi0,	Ybr,	Zbr,	0,
	Yi1,	Ybr,	Zbr,	1,
	0
};
static uchar	yloop[] =
{
	Ynone,	Ybr,	Zloop,	1,
	0
};
static uchar	ycall[] =
{
	Ynone,	Yml,	Zcallindreg,	0,
	Yrx,	Yrx,	Zcallindreg,	2,
	Ynone,	Ybr,	Zcall,	1,
	0
};
static uchar	yduff[] =
{
	Ynone,	Yi32,	Zcall,	1,
	0
};
static uchar	yjmp[] =
{
	Ynone,	Yml,	Zo_m64,	2,
	Ynone,	Ybr,	Zjmp,	1,
	0
};

static uchar	yfmvd[] =
{
	Ym,	Yf0,	Zm_o,	2,
	Yf0,	Ym,	Zo_m,	2,
	Yrf,	Yf0,	Zm_o,	2,
	Yf0,	Yrf,	Zo_m,	2,
	0
};
static uchar	yfmvdp[] =
{
	Yf0,	Ym,	Zo_m,	2,
	Yf0,	Yrf,	Zo_m,	2,
	0
};
static uchar	yfmvf[] =
{
	Ym,	Yf0,	Zm_o,	2,
	Yf0,	Ym,	Zo_m,	2,
	0
};
static uchar	yfmvx[] =
{
	Ym,	Yf0,	Zm_o,	2,
	0
};
static uchar	yfmvp[] =
{
	Yf0,	Ym,	Zo_m,	2,
	0
};
static uchar	yfadd[] =
{
	Ym,	Yf0,	Zm_o,	2,
	Yrf,	Yf0,	Zm_o,	2,
	Yf0,	Yrf,	Zo_m,	2,
	0
};
static uchar	yfaddp[] =
{
	Yf0,	Yrf,	Zo_m,	2,
	0
};
static uchar	yfxch[] =
{
	Yf0,	Yrf,	Zo_m,	2,
	Yrf,	Yf0,	Zm_o,	2,
	0
};
static uchar	ycompp[] =
{
	Yf0,	Yrf,	Zo_m,	2,	/* botch is really f0,f1 */
	0
};
static uchar	ystsw[] =
{
	Ynone,	Ym,	Zo_m,	2,
	Ynone,	Yax,	Zlit,	1,
	0
};
static uchar	ystcw[] =
{
	Ynone,	Ym,	Zo_m,	2,
	Ym,	Ynone,	Zm_o,	2,
	0
};
static uchar	ysvrs[] =
{
	Ynone,	Ym,	Zo_m,	2,
	Ym,	Ynone,	Zm_o,	2,
	0
};
static uchar	ymm[] = 
{
	Ymm,	Ymr,	Zm_r_xm,	1,
	Yxm,	Yxr,	Zm_r_xm,	2,
	0
};
static uchar	yxm[] = 
{
	Yxm,	Yxr,	Zm_r_xm,	1,
	0
};
static uchar	yxcvm1[] = 
{
	Yxm,	Yxr,	Zm_r_xm,	2,
	Yxm,	Ymr,	Zm_r_xm,	2,
	0
};
static uchar	yxcvm2[] =
{
	Yxm,	Yxr,	Zm_r_xm,	2,
	Ymm,	Yxr,	Zm_r_xm,	2,
	0
};
/*
static uchar	yxmq[] = 
{
	Yxm,	Yxr,	Zm_r_xm,	2,
	0
};
*/
static uchar	yxr[] = 
{
	Yxr,	Yxr,	Zm_r_xm,	1,
	0
};
static uchar	yxr_ml[] =
{
	Yxr,	Yml,	Zr_m_xm,	1,
	0
};
static uchar	ymr[] =
{
	Ymr,	Ymr,	Zm_r,	1,
	0
};
static uchar	ymr_ml[] =
{
	Ymr,	Yml,	Zr_m_xm,	1,
	0
};
static uchar	yxcmp[] =
{
	Yxm,	Yxr, Zm_r_xm,	1,
	0
};
static uchar	yxcmpi[] =
{
	Yxm,	Yxr, Zm_r_i_xm,	2,
	0
};
static uchar	yxmov[] =
{
	Yxm,	Yxr,	Zm_r_xm,	1,
	Yxr,	Yxm,	Zr_m_xm,	1,
	0
};
static uchar	yxcvfl[] = 
{
	Yxm,	Yrl,	Zm_r_xm,	1,
	0
};
static uchar	yxcvlf[] =
{
	Yml,	Yxr,	Zm_r_xm,	1,
	0
};
static uchar	yxcvfq[] = 
{
	Yxm,	Yrl,	Zm_r_xm,	2,
	0
};
static uchar	yxcvqf[] =
{
	Yml,	Yxr,	Zm_r_xm,	2,
	0
};
static uchar	yps[] = 
{
	Ymm,	Ymr,	Zm_r_xm,	1,
	Yi8,	Ymr,	Zibo_m_xm,	2,
	Yxm,	Yxr,	Zm_r_xm,	2,
	Yi8,	Yxr,	Zibo_m_xm,	3,
	0
};
static uchar	yxrrl[] =
{
	Yxr,	Yrl,	Zm_r,	1,
	0
};
static uchar	ymfp[] =
{
	Ymm,	Ymr,	Zm_r_3d,	1,
	0,
};
static uchar	ymrxr[] =
{
	Ymr,	Yxr,	Zm_r,	1,
	Yxm,	Yxr,	Zm_r_xm,	1,
	0
};
static uchar	ymshuf[] =
{
	Ymm,	Ymr,	Zibm_r,	2,
	0
};
static uchar	ymshufb[] =
{
	Yxm,	Yxr,	Zm2_r,	2,
	0
};
static uchar	yxshuf[] =
{
	Yxm,	Yxr,	Zibm_r,	2,
	0
};
static uchar	yextrw[] =
{
	Yxr,	Yrl,	Zibm_r,	2,
	0
};
static uchar	yinsrw[] =
{
	Yml,	Yxr,	Zibm_r,	2,
	0
};
static uchar	yinsr[] =
{
	Ymm,	Yxr,	Zibm_r,	3,
	0
};
static uchar	ypsdq[] =
{
	Yi8,	Yxr,	Zibo_m,	2,
	0
};
static uchar	ymskb[] =
{
	Yxr,	Yrl,	Zm_r_xm,	2,
	Ymr,	Yrl,	Zm_r_xm,	1,
	0
};
static uchar	ycrc32l[] =
{
	Yml,	Yrl,	Zlitm_r,	0,
};
static uchar	yprefetch[] =
{
	Ym,	Ynone,	Zm_o,	2,
	0,
};
static uchar	yaes[] =
{
	Yxm,	Yxr,	Zlitm_r,	2,
	0
};
static uchar	yaes2[] =
{
	Yxm,	Yxr,	Zibm_r,	2,
	0
};

/*
 * You are doasm, holding in your hand a Prog* with p->as set to, say, ACRC32,
 * and p->from and p->to as operands (Addr*).  The linker scans optab to find
 * the entry with the given p->as and then looks through the ytable for that
 * instruction (the second field in the optab struct) for a line whose first
 * two values match the Ytypes of the p->from and p->to operands.  The function
 * oclass in span.c computes the specific Ytype of an operand and then the set
 * of more general Ytypes that it satisfies is implied by the ycover table, set
 * up in instinit.  For example, oclass distinguishes the constants 0 and 1
 * from the more general 8-bit constants, but instinit says
 *
 *        ycover[Yi0*Ymax + Ys32] = 1;
 *        ycover[Yi1*Ymax + Ys32] = 1;
 *        ycover[Yi8*Ymax + Ys32] = 1;
 *
 * which means that Yi0, Yi1, and Yi8 all count as Ys32 (signed 32)
 * if that's what an instruction can handle.
 *
 * In parallel with the scan through the ytable for the appropriate line, there
 * is a z pointer that starts out pointing at the strange magic byte list in
 * the Optab struct.  With each step past a non-matching ytable line, z
 * advances by the 4th entry in the line.  When a matching line is found, that
 * z pointer has the extra data to use in laying down the instruction bytes.
 * The actual bytes laid down are a function of the 3rd entry in the line (that
 * is, the Ztype) and the z bytes.
 *
 * For example, let's look at AADDL.  The optab line says:
 *        { AADDL,        yaddl,  Px, 0x83,(00),0x05,0x81,(00),0x01,0x03 },
 *
 * and yaddl says
 *        uchar   yaddl[] =
 *        {
 *                Yi8,    Yml,    Zibo_m, 2,
 *                Yi32,   Yax,    Zil_,   1,
 *                Yi32,   Yml,    Zilo_m, 2,
 *                Yrl,    Yml,    Zr_m,   1,
 *                Yml,    Yrl,    Zm_r,   1,
 *                0
 *        };
 *
 * so there are 5 possible types of ADDL instruction that can be laid down, and
 * possible states used to lay them down (Ztype and z pointer, assuming z
 * points at {0x83,(00),0x05,0x81,(00),0x01,0x03}) are:
 *
 *        Yi8, Yml -> Zibo_m, z (0x83, 00)
 *        Yi32, Yax -> Zil_, z+2 (0x05)
 *        Yi32, Yml -> Zilo_m, z+2+1 (0x81, 0x00)
 *        Yrl, Yml -> Zr_m, z+2+1+2 (0x01)
 *        Yml, Yrl -> Zm_r, z+2+1+2+1 (0x03)
 *
 * The Pconstant in the optab line controls the prefix bytes to emit.  That's
 * relatively straightforward as this program goes.
 *
 * The switch on t[2] in doasm implements the various Z cases.  Zibo_m, for
 * example, is an opcode byte (z[0]) then an asmando (which is some kind of
 * encoded addressing mode for the Yml arg), and then a single immediate byte.
 * Zilo_m is the same but a long (32-bit) immediate.
 */
static Optab optab[] =
/*	as, ytab, andproto, opcode */
{
	{ AXXX },
	{ AAAA,		ynone,	P32, {0x37} },
	{ AAAD,		ynone,	P32, {0xd5,0x0a} },
	{ AAAM,		ynone,	P32, {0xd4,0x0a} },
	{ AAAS,		ynone,	P32, {0x3f} },
	{ AADCB,	yxorb,	Pb, {0x14,0x80,(02),0x10,0x10} },
	{ AADCL,	yxorl,	Px, {0x83,(02),0x15,0x81,(02),0x11,0x13} },
	{ AADCQ,	yxorl,	Pw, {0x83,(02),0x15,0x81,(02),0x11,0x13} },
	{ AADCW,	yxorl,	Pe, {0x83,(02),0x15,0x81,(02),0x11,0x13} },
	{ AADDB,	yxorb,	Pb, {0x04,0x80,(00),0x00,0x02} },
	{ AADDL,	yaddl,	Px, {0x83,(00),0x05,0x81,(00),0x01,0x03} },
	{ AADDPD,	yxm,	Pq, {0x58} },
	{ AADDPS,	yxm,	Pm, {0x58} },
	{ AADDQ,	yaddl,	Pw, {0x83,(00),0x05,0x81,(00),0x01,0x03} },
	{ AADDSD,	yxm,	Pf2, {0x58} },
	{ AADDSS,	yxm,	Pf3, {0x58} },
	{ AADDW,	yaddl,	Pe, {0x83,(00),0x05,0x81,(00),0x01,0x03} },
	{ AADJSP },
	{ AANDB,	yxorb,	Pb, {0x24,0x80,(04),0x20,0x22} },
	{ AANDL,	yxorl,	Px, {0x83,(04),0x25,0x81,(04),0x21,0x23} },
	{ AANDNPD,	yxm,	Pq, {0x55} },
	{ AANDNPS,	yxm,	Pm, {0x55} },
	{ AANDPD,	yxm,	Pq, {0x54} },
	{ AANDPS,	yxm,	Pq, {0x54} },
	{ AANDQ,	yxorl,	Pw, {0x83,(04),0x25,0x81,(04),0x21,0x23} },
	{ AANDW,	yxorl,	Pe, {0x83,(04),0x25,0x81,(04),0x21,0x23} },
	{ AARPL,	yrl_ml,	P32, {0x63} },
	{ ABOUNDL,	yrl_m,	P32, {0x62} },
	{ ABOUNDW,	yrl_m,	Pe, {0x62} },
	{ ABSFL,	yml_rl,	Pm, {0xbc} },
	{ ABSFQ,	yml_rl,	Pw, {0x0f,0xbc} },
	{ ABSFW,	yml_rl,	Pq, {0xbc} },
	{ ABSRL,	yml_rl,	Pm, {0xbd} },
	{ ABSRQ,	yml_rl,	Pw, {0x0f,0xbd} },
	{ ABSRW,	yml_rl,	Pq, {0xbd} },
	{ ABSWAPL,	ybswap,	Px, {0x0f,0xc8} },
	{ ABSWAPQ,	ybswap,	Pw, {0x0f,0xc8} },
	{ ABTCL,	ybtl,	Pm, {0xba,(07),0xbb} },
	{ ABTCQ,	ybtl,	Pw, {0x0f,0xba,(07),0x0f,0xbb} },
	{ ABTCW,	ybtl,	Pq, {0xba,(07),0xbb} },
	{ ABTL,		ybtl,	Pm, {0xba,(04),0xa3} },
	{ ABTQ,		ybtl,	Pw, {0x0f,0xba,(04),0x0f,0xa3}},
	{ ABTRL,	ybtl,	Pm, {0xba,(06),0xb3} },
	{ ABTRQ,	ybtl,	Pw, {0x0f,0xba,(06),0x0f,0xb3} },
	{ ABTRW,	ybtl,	Pq, {0xba,(06),0xb3} },
	{ ABTSL,	ybtl,	Pm, {0xba,(05),0xab } },
	{ ABTSQ,	ybtl,	Pw, {0x0f,0xba,(05),0x0f,0xab} },
	{ ABTSW,	ybtl,	Pq, {0xba,(05),0xab } },
	{ ABTW,		ybtl,	Pq, {0xba,(04),0xa3} },
	{ ABYTE,	ybyte,	Px, {1} },
	{ ACALL,	ycall,	Px, {0xff,(02),0xe8} },
	{ ACDQ,		ynone,	Px, {0x99} },
	{ ACLC,		ynone,	Px, {0xf8} },
	{ ACLD,		ynone,	Px, {0xfc} },
	{ ACLI,		ynone,	Px, {0xfa} },
	{ ACLTS,	ynone,	Pm, {0x06} },
	{ ACMC,		ynone,	Px, {0xf5} },
	{ ACMOVLCC,	yml_rl,	Pm, {0x43} },
	{ ACMOVLCS,	yml_rl,	Pm, {0x42} },
	{ ACMOVLEQ,	yml_rl,	Pm, {0x44} },
	{ ACMOVLGE,	yml_rl,	Pm, {0x4d} },
	{ ACMOVLGT,	yml_rl,	Pm, {0x4f} },
	{ ACMOVLHI,	yml_rl,	Pm, {0x47} },
	{ ACMOVLLE,	yml_rl,	Pm, {0x4e} },
	{ ACMOVLLS,	yml_rl,	Pm, {0x46} },
	{ ACMOVLLT,	yml_rl,	Pm, {0x4c} },
	{ ACMOVLMI,	yml_rl,	Pm, {0x48} },
	{ ACMOVLNE,	yml_rl,	Pm, {0x45} },
	{ ACMOVLOC,	yml_rl,	Pm, {0x41} },
	{ ACMOVLOS,	yml_rl,	Pm, {0x40} },
	{ ACMOVLPC,	yml_rl,	Pm, {0x4b} },
	{ ACMOVLPL,	yml_rl,	Pm, {0x49} },
	{ ACMOVLPS,	yml_rl,	Pm, {0x4a} },
	{ ACMOVQCC,	yml_rl,	Pw, {0x0f,0x43} },
	{ ACMOVQCS,	yml_rl,	Pw, {0x0f,0x42} },
	{ ACMOVQEQ,	yml_rl,	Pw, {0x0f,0x44} },
	{ ACMOVQGE,	yml_rl,	Pw, {0x0f,0x4d} },
	{ ACMOVQGT,	yml_rl,	Pw, {0x0f,0x4f} },
	{ ACMOVQHI,	yml_rl,	Pw, {0x0f,0x47} },
	{ ACMOVQLE,	yml_rl,	Pw, {0x0f,0x4e} },
	{ ACMOVQLS,	yml_rl,	Pw, {0x0f,0x46} },
	{ ACMOVQLT,	yml_rl,	Pw, {0x0f,0x4c} },
	{ ACMOVQMI,	yml_rl,	Pw, {0x0f,0x48} },
	{ ACMOVQNE,	yml_rl,	Pw, {0x0f,0x45} },
	{ ACMOVQOC,	yml_rl,	Pw, {0x0f,0x41} },
	{ ACMOVQOS,	yml_rl,	Pw, {0x0f,0x40} },
	{ ACMOVQPC,	yml_rl,	Pw, {0x0f,0x4b} },
	{ ACMOVQPL,	yml_rl,	Pw, {0x0f,0x49} },
	{ ACMOVQPS,	yml_rl,	Pw, {0x0f,0x4a} },
	{ ACMOVWCC,	yml_rl,	Pq, {0x43} },
	{ ACMOVWCS,	yml_rl,	Pq, {0x42} },
	{ ACMOVWEQ,	yml_rl,	Pq, {0x44} },
	{ ACMOVWGE,	yml_rl,	Pq, {0x4d} },
	{ ACMOVWGT,	yml_rl,	Pq, {0x4f} },
	{ ACMOVWHI,	yml_rl,	Pq, {0x47} },
	{ ACMOVWLE,	yml_rl,	Pq, {0x4e} },
	{ ACMOVWLS,	yml_rl,	Pq, {0x46} },
	{ ACMOVWLT,	yml_rl,	Pq, {0x4c} },
	{ ACMOVWMI,	yml_rl,	Pq, {0x48} },
	{ ACMOVWNE,	yml_rl,	Pq, {0x45} },
	{ ACMOVWOC,	yml_rl,	Pq, {0x41} },
	{ ACMOVWOS,	yml_rl,	Pq, {0x40} },
	{ ACMOVWPC,	yml_rl,	Pq, {0x4b} },
	{ ACMOVWPL,	yml_rl,	Pq, {0x49} },
	{ ACMOVWPS,	yml_rl,	Pq, {0x4a} },
	{ ACMPB,	ycmpb,	Pb, {0x3c,0x80,(07),0x38,0x3a} },
	{ ACMPL,	ycmpl,	Px, {0x83,(07),0x3d,0x81,(07),0x39,0x3b} },
	{ ACMPPD,	yxcmpi,	Px, {Pe,0xc2} },
	{ ACMPPS,	yxcmpi,	Pm, {0xc2,0} },
	{ ACMPQ,	ycmpl,	Pw, {0x83,(07),0x3d,0x81,(07),0x39,0x3b} },
	{ ACMPSB,	ynone,	Pb, {0xa6} },
	{ ACMPSD,	yxcmpi,	Px, {Pf2,0xc2} },
	{ ACMPSL,	ynone,	Px, {0xa7} },
	{ ACMPSQ,	ynone,	Pw, {0xa7} },
	{ ACMPSS,	yxcmpi,	Px, {Pf3,0xc2} },
	{ ACMPSW,	ynone,	Pe, {0xa7} },
	{ ACMPW,	ycmpl,	Pe, {0x83,(07),0x3d,0x81,(07),0x39,0x3b} },
	{ ACOMISD,	yxcmp,	Pe, {0x2f} },
	{ ACOMISS,	yxcmp,	Pm, {0x2f} },
	{ ACPUID,	ynone,	Pm, {0xa2} },
	{ ACVTPL2PD,	yxcvm2,	Px, {Pf3,0xe6,Pe,0x2a} },
	{ ACVTPL2PS,	yxcvm2,	Pm, {0x5b,0,0x2a,0,} },
	{ ACVTPD2PL,	yxcvm1,	Px, {Pf2,0xe6,Pe,0x2d} },
	{ ACVTPD2PS,	yxm,	Pe, {0x5a} },
	{ ACVTPS2PL,	yxcvm1, Px, {Pe,0x5b,Pm,0x2d} },
	{ ACVTPS2PD,	yxm,	Pm, {0x5a} },
	{ API2FW,	ymfp,	Px, {0x0c} },
	{ ACVTSD2SL,	yxcvfl, Pf2, {0x2d} },
	{ ACVTSD2SQ,	yxcvfq, Pw, {Pf2,0x2d} },
	{ ACVTSD2SS,	yxm,	Pf2, {0x5a} },
	{ ACVTSL2SD,	yxcvlf, Pf2, {0x2a} },
	{ ACVTSQ2SD,	yxcvqf, Pw, {Pf2,0x2a} },
	{ ACVTSL2SS,	yxcvlf, Pf3, {0x2a} },
	{ ACVTSQ2SS,	yxcvqf, Pw, {Pf3,0x2a} },
	{ ACVTSS2SD,	yxm,	Pf3, {0x5a} },
	{ ACVTSS2SL,	yxcvfl, Pf3, {0x2d} },
	{ ACVTSS2SQ,	yxcvfq, Pw, {Pf3,0x2d} },
	{ ACVTTPD2PL,	yxcvm1,	Px, {Pe,0xe6,Pe,0x2c} },
	{ ACVTTPS2PL,	yxcvm1,	Px, {Pf3,0x5b,Pm,0x2c} },
	{ ACVTTSD2SL,	yxcvfl, Pf2, {0x2c} },
	{ ACVTTSD2SQ,	yxcvfq, Pw, {Pf2,0x2c} },
	{ ACVTTSS2SL,	yxcvfl,	Pf3, {0x2c} },
	{ ACVTTSS2SQ,	yxcvfq, Pw, {Pf3,0x2c} },
	{ ACWD,		ynone,	Pe, {0x99} },
	{ ACQO,		ynone,	Pw, {0x99} },
	{ ADAA,		ynone,	P32, {0x27} },
	{ ADAS,		ynone,	P32, {0x2f} },
	{ ADATA },
	{ ADECB,	yincb,	Pb, {0xfe,(01)} },
	{ ADECL,	yincl,	Px, {0xff,(01)} },
	{ ADECQ,	yincl,	Pw, {0xff,(01)} },
	{ ADECW,	yincw,	Pe, {0xff,(01)} },
	{ ADIVB,	ydivb,	Pb, {0xf6,(06)} },
	{ ADIVL,	ydivl,	Px, {0xf7,(06)} },
	{ ADIVPD,	yxm,	Pe, {0x5e} },
	{ ADIVPS,	yxm,	Pm, {0x5e} },
	{ ADIVQ,	ydivl,	Pw, {0xf7,(06)} },
	{ ADIVSD,	yxm,	Pf2, {0x5e} },
	{ ADIVSS,	yxm,	Pf3, {0x5e} },
	{ ADIVW,	ydivl,	Pe, {0xf7,(06)} },
	{ AEMMS,	ynone,	Pm, {0x77} },
	{ AENTER },				/* botch */
	{ AFXRSTOR,	ysvrs,	Pm, {0xae,(01),0xae,(01)} },
	{ AFXSAVE,	ysvrs,	Pm, {0xae,(00),0xae,(00)} },
	{ AFXRSTOR64,	ysvrs,	Pw, {0x0f,0xae,(01),0x0f,0xae,(01)} },
	{ AFXSAVE64,	ysvrs,	Pw, {0x0f,0xae,(00),0x0f,0xae,(00)} },
	{ AGLOBL },
	{ AHLT,		ynone,	Px, {0xf4} },
	{ AIDIVB,	ydivb,	Pb, {0xf6,(07)} },
	{ AIDIVL,	ydivl,	Px, {0xf7,(07)} },
	{ AIDIVQ,	ydivl,	Pw, {0xf7,(07)} },
	{ AIDIVW,	ydivl,	Pe, {0xf7,(07)} },
	{ AIMULB,	ydivb,	Pb, {0xf6,(05)} },
	{ AIMULL,	yimul,	Px, {0xf7,(05),0x6b,0x69,Pm,0xaf} },
	{ AIMULQ,	yimul,	Pw, {0xf7,(05),0x6b,0x69,Pm,0xaf} },
	{ AIMULW,	yimul,	Pe, {0xf7,(05),0x6b,0x69,Pm,0xaf} },
	{ AIMUL3Q,	yimul3,	Pw, {0x6b,(00)} },
	{ AINB,		yin,	Pb, {0xe4,0xec} },
	{ AINCB,	yincb,	Pb, {0xfe,(00)} },
	{ AINCL,	yincl,	Px, {0xff,(00)} },
	{ AINCQ,	yincl,	Pw, {0xff,(00)} },
	{ AINCW,	yincw,	Pe, {0xff,(00)} },
	{ AINL,		yin,	Px, {0xe5,0xed} },
	{ AINSB,	ynone,	Pb, {0x6c} },
	{ AINSL,	ynone,	Px, {0x6d} },
	{ AINSW,	ynone,	Pe, {0x6d} },
	{ AINT,		yint,	Px, {0xcd} },
	{ AINTO,	ynone,	P32, {0xce} },
	{ AINW,		yin,	Pe, {0xe5,0xed} },
	{ AIRETL,	ynone,	Px, {0xcf} },
	{ AIRETQ,	ynone,	Pw, {0xcf} },
	{ AIRETW,	ynone,	Pe, {0xcf} },
	{ AJCC,		yjcond,	Px, {0x73,0x83,(00)} },
	{ AJCS,		yjcond,	Px, {0x72,0x82} },
	{ AJCXZL,	yloop,	Px, {0xe3} },
	{ AJCXZQ,	yloop,	Px, {0xe3} },
	{ AJEQ,		yjcond,	Px, {0x74,0x84} },
	{ AJGE,		yjcond,	Px, {0x7d,0x8d} },
	{ AJGT,		yjcond,	Px, {0x7f,0x8f} },
	{ AJHI,		yjcond,	Px, {0x77,0x87} },
	{ AJLE,		yjcond,	Px, {0x7e,0x8e} },
	{ AJLS,		yjcond,	Px, {0x76,0x86} },
	{ AJLT,		yjcond,	Px, {0x7c,0x8c} },
	{ AJMI,		yjcond,	Px, {0x78,0x88} },
	{ AJMP,		yjmp,	Px, {0xff,(04),0xeb,0xe9} },
	{ AJNE,		yjcond,	Px, {0x75,0x85} },
	{ AJOC,		yjcond,	Px, {0x71,0x81,(00)} },
	{ AJOS,		yjcond,	Px, {0x70,0x80,(00)} },
	{ AJPC,		yjcond,	Px, {0x7b,0x8b} },
	{ AJPL,		yjcond,	Px, {0x79,0x89} },
	{ AJPS,		yjcond,	Px, {0x7a,0x8a} },
	{ ALAHF,	ynone,	Px, {0x9f} },
	{ ALARL,	yml_rl,	Pm, {0x02} },
	{ ALARW,	yml_rl,	Pq, {0x02} },
	{ ALDMXCSR,	ysvrs,	Pm, {0xae,(02),0xae,(02)} },
	{ ALEAL,	ym_rl,	Px, {0x8d} },
	{ ALEAQ,	ym_rl,	Pw, {0x8d} },
	{ ALEAVEL,	ynone,	P32, {0xc9} },
	{ ALEAVEQ,	ynone,	Py, {0xc9} },
	{ ALEAVEW,	ynone,	Pe, {0xc9} },
	{ ALEAW,	ym_rl,	Pe, {0x8d} },
	{ ALOCK,	ynone,	Px, {0xf0} },
	{ ALODSB,	ynone,	Pb, {0xac} },
	{ ALODSL,	ynone,	Px, {0xad} },
	{ ALODSQ,	ynone,	Pw, {0xad} },
	{ ALODSW,	ynone,	Pe, {0xad} },
	{ ALONG,	ybyte,	Px, {4} },
	{ ALOOP,	yloop,	Px, {0xe2} },
	{ ALOOPEQ,	yloop,	Px, {0xe1} },
	{ ALOOPNE,	yloop,	Px, {0xe0} },
	{ ALSLL,	yml_rl,	Pm, {0x03 } },
	{ ALSLW,	yml_rl,	Pq, {0x03 } },
	{ AMASKMOVOU,	yxr,	Pe, {0xf7} },
	{ AMASKMOVQ,	ymr,	Pm, {0xf7} },
	{ AMAXPD,	yxm,	Pe, {0x5f} },
	{ AMAXPS,	yxm,	Pm, {0x5f} },
	{ AMAXSD,	yxm,	Pf2, {0x5f} },
	{ AMAXSS,	yxm,	Pf3, {0x5f} },
	{ AMINPD,	yxm,	Pe, {0x5d} },
	{ AMINPS,	yxm,	Pm, {0x5d} },
	{ AMINSD,	yxm,	Pf2, {0x5d} },
	{ AMINSS,	yxm,	Pf3, {0x5d} },
	{ AMOVAPD,	yxmov,	Pe, {0x28,0x29} },
	{ AMOVAPS,	yxmov,	Pm, {0x28,0x29} },
	{ AMOVB,	ymovb,	Pb, {0x88,0x8a,0xb0,0xc6,(00)} },
	{ AMOVBLSX,	ymb_rl,	Pm, {0xbe} },
	{ AMOVBLZX,	ymb_rl,	Pm, {0xb6} },
	{ AMOVBQSX,	ymb_rl,	Pw, {0x0f,0xbe} },
	{ AMOVBQZX,	ymb_rl,	Pm, {0xb6} },
	{ AMOVBWSX,	ymb_rl,	Pq, {0xbe} },
	{ AMOVBWZX,	ymb_rl,	Pq, {0xb6} },
	{ AMOVO,	yxmov,	Pe, {0x6f,0x7f} },
	{ AMOVOU,	yxmov,	Pf3, {0x6f,0x7f} },
	{ AMOVHLPS,	yxr,	Pm, {0x12} },
	{ AMOVHPD,	yxmov,	Pe, {0x16,0x17} },
	{ AMOVHPS,	yxmov,	Pm, {0x16,0x17} },
	{ AMOVL,	ymovl,	Px, {0x89,0x8b,0x31,0xb8,0xc7,(00),0x6e,0x7e,Pe,0x6e,Pe,0x7e,0} },
	{ AMOVLHPS,	yxr,	Pm, {0x16} },
	{ AMOVLPD,	yxmov,	Pe, {0x12,0x13} },
	{ AMOVLPS,	yxmov,	Pm, {0x12,0x13} },
	{ AMOVLQSX,	yml_rl,	Pw, {0x63} },
	{ AMOVLQZX,	yml_rl,	Px, {0x8b} },
	{ AMOVMSKPD,	yxrrl,	Pq, {0x50} },
	{ AMOVMSKPS,	yxrrl,	Pm, {0x50} },
	{ AMOVNTO,	yxr_ml,	Pe, {0xe7} },
	{ AMOVNTPD,	yxr_ml,	Pe, {0x2b} },
	{ AMOVNTPS,	yxr_ml,	Pm, {0x2b} },
	{ AMOVNTQ,	ymr_ml,	Pm, {0xe7} },
	{ AMOVQ,	ymovq,	Pw, {0x89, 0x8b, 0x31, 0xc7,(00), 0xb8, 0xc7,(00), 0x6f, 0x7f, 0x6e, 0x7e, Pf2,0xd6, Pf3,0x7e, Pe,0xd6, Pe,0x6e, Pe,0x7e,0} },
	{ AMOVQOZX,	ymrxr,	Pf3, {0xd6,0x7e} },
	{ AMOVSB,	ynone,	Pb, {0xa4} },
	{ AMOVSD,	yxmov,	Pf2, {0x10,0x11} },
	{ AMOVSL,	ynone,	Px, {0xa5} },
	{ AMOVSQ,	ynone,	Pw, {0xa5} },
	{ AMOVSS,	yxmov,	Pf3, {0x10,0x11} },
	{ AMOVSW,	ynone,	Pe, {0xa5} },
	{ AMOVUPD,	yxmov,	Pe, {0x10,0x11} },
	{ AMOVUPS,	yxmov,	Pm, {0x10,0x11} },
	{ AMOVW,	ymovw,	Pe, {0x89,0x8b,0x31,0xb8,0xc7,(00),0} },
	{ AMOVWLSX,	yml_rl,	Pm, {0xbf} },
	{ AMOVWLZX,	yml_rl,	Pm, {0xb7} },
	{ AMOVWQSX,	yml_rl,	Pw, {0x0f,0xbf} },
	{ AMOVWQZX,	yml_rl,	Pw, {0x0f,0xb7} },
	{ AMULB,	ydivb,	Pb, {0xf6,(04)} },
	{ AMULL,	ydivl,	Px, {0xf7,(04)} },
	{ AMULPD,	yxm,	Pe, {0x59} },
	{ AMULPS,	yxm,	Ym, {0x59} },
	{ AMULQ,	ydivl,	Pw, {0xf7,(04)} },
	{ AMULSD,	yxm,	Pf2, {0x59} },
	{ AMULSS,	yxm,	Pf3, {0x59} },
	{ AMULW,	ydivl,	Pe, {0xf7,(04)} },
	{ ANEGB,	yscond,	Pb, {0xf6,(03)} },
	{ ANEGL,	yscond,	Px, {0xf7,(03)} },
	{ ANEGQ,	yscond,	Pw, {0xf7,(03)} },
	{ ANEGW,	yscond,	Pe, {0xf7,(03)} },
	{ ANOP,		ynop,	Px, {0,0} },
	{ ANOTB,	yscond,	Pb, {0xf6,(02)} },
	{ ANOTL,	yscond,	Px, {0xf7,(02)} },
	{ ANOTQ,	yscond,	Pw, {0xf7,(02)} },
	{ ANOTW,	yscond,	Pe, {0xf7,(02)} },
	{ AORB,		yxorb,	Pb, {0x0c,0x80,(01),0x08,0x0a} },
	{ AORL,		yxorl,	Px, {0x83,(01),0x0d,0x81,(01),0x09,0x0b} },
	{ AORPD,	yxm,	Pq, {0x56} },
	{ AORPS,	yxm,	Pm, {0x56} },
	{ AORQ,		yxorl,	Pw, {0x83,(01),0x0d,0x81,(01),0x09,0x0b} },
	{ AORW,		yxorl,	Pe, {0x83,(01),0x0d,0x81,(01),0x09,0x0b} },
	{ AOUTB,	yin,	Pb, {0xe6,0xee} },
	{ AOUTL,	yin,	Px, {0xe7,0xef} },
	{ AOUTSB,	ynone,	Pb, {0x6e} },
	{ AOUTSL,	ynone,	Px, {0x6f} },
	{ AOUTSW,	ynone,	Pe, {0x6f} },
	{ AOUTW,	yin,	Pe, {0xe7,0xef} },
	{ APACKSSLW,	ymm,	Py, {0x6b,Pe,0x6b} },
	{ APACKSSWB,	ymm,	Py, {0x63,Pe,0x63} },
	{ APACKUSWB,	ymm,	Py, {0x67,Pe,0x67} },
	{ APADDB,	ymm,	Py, {0xfc,Pe,0xfc} },
	{ APADDL,	ymm,	Py, {0xfe,Pe,0xfe} },
	{ APADDQ,	yxm,	Pe, {0xd4} },
	{ APADDSB,	ymm,	Py, {0xec,Pe,0xec} },
	{ APADDSW,	ymm,	Py, {0xed,Pe,0xed} },
	{ APADDUSB,	ymm,	Py, {0xdc,Pe,0xdc} },
	{ APADDUSW,	ymm,	Py, {0xdd,Pe,0xdd} },
	{ APADDW,	ymm,	Py, {0xfd,Pe,0xfd} },
	{ APAND,	ymm,	Py, {0xdb,Pe,0xdb} },
	{ APANDN,	ymm,	Py, {0xdf,Pe,0xdf} },
	{ APAUSE,	ynone,	Px, {0xf3,0x90} },
	{ APAVGB,	ymm,	Py, {0xe0,Pe,0xe0} },
	{ APAVGW,	ymm,	Py, {0xe3,Pe,0xe3} },
	{ APCMPEQB,	ymm,	Py, {0x74,Pe,0x74} },
	{ APCMPEQL,	ymm,	Py, {0x76,Pe,0x76} },
	{ APCMPEQW,	ymm,	Py, {0x75,Pe,0x75} },
	{ APCMPGTB,	ymm,	Py, {0x64,Pe,0x64} },
	{ APCMPGTL,	ymm,	Py, {0x66,Pe,0x66} },
	{ APCMPGTW,	ymm,	Py, {0x65,Pe,0x65} },
	{ APEXTRW,	yextrw,	Pq, {0xc5,(00)} },
	{ APF2IL,	ymfp,	Px, {0x1d} },
	{ APF2IW,	ymfp,	Px, {0x1c} },
	{ API2FL,	ymfp,	Px, {0x0d} },
	{ APFACC,	ymfp,	Px, {0xae} },
	{ APFADD,	ymfp,	Px, {0x9e} },
	{ APFCMPEQ,	ymfp,	Px, {0xb0} },
	{ APFCMPGE,	ymfp,	Px, {0x90} },
	{ APFCMPGT,	ymfp,	Px, {0xa0} },
	{ APFMAX,	ymfp,	Px, {0xa4} },
	{ APFMIN,	ymfp,	Px, {0x94} },
	{ APFMUL,	ymfp,	Px, {0xb4} },
	{ APFNACC,	ymfp,	Px, {0x8a} },
	{ APFPNACC,	ymfp,	Px, {0x8e} },
	{ APFRCP,	ymfp,	Px, {0x96} },
	{ APFRCPIT1,	ymfp,	Px, {0xa6} },
	{ APFRCPI2T,	ymfp,	Px, {0xb6} },
	{ APFRSQIT1,	ymfp,	Px, {0xa7} },
	{ APFRSQRT,	ymfp,	Px, {0x97} },
	{ APFSUB,	ymfp,	Px, {0x9a} },
	{ APFSUBR,	ymfp,	Px, {0xaa} },
	{ APINSRW,	yinsrw,	Pq, {0xc4,(00)} },
	{ APINSRD,	yinsr,	Pq, {0x3a, 0x22, (00)} },
	{ APINSRQ,	yinsr,	Pq3, {0x3a, 0x22, (00)} },
	{ APMADDWL,	ymm,	Py, {0xf5,Pe,0xf5} },
	{ APMAXSW,	yxm,	Pe, {0xee} },
	{ APMAXUB,	yxm,	Pe, {0xde} },
	{ APMINSW,	yxm,	Pe, {0xea} },
	{ APMINUB,	yxm,	Pe, {0xda} },
	{ APMOVMSKB,	ymskb,	Px, {Pe,0xd7,0xd7} },
	{ APMULHRW,	ymfp,	Px, {0xb7} },
	{ APMULHUW,	ymm,	Py, {0xe4,Pe,0xe4} },
	{ APMULHW,	ymm,	Py, {0xe5,Pe,0xe5} },
	{ APMULLW,	ymm,	Py, {0xd5,Pe,0xd5} },
	{ APMULULQ,	ymm,	Py, {0xf4,Pe,0xf4} },
	{ APOPAL,	ynone,	P32, {0x61} },
	{ APOPAW,	ynone,	Pe, {0x61} },
	{ APOPFL,	ynone,	P32, {0x9d} },
	{ APOPFQ,	ynone,	Py, {0x9d} },
	{ APOPFW,	ynone,	Pe, {0x9d} },
	{ APOPL,	ypopl,	P32, {0x58,0x8f,(00)} },
	{ APOPQ,	ypopl,	Py, {0x58,0x8f,(00)} },
	{ APOPW,	ypopl,	Pe, {0x58,0x8f,(00)} },
	{ APOR,		ymm,	Py, {0xeb,Pe,0xeb} },
	{ APSADBW,	yxm,	Pq, {0xf6} },
	{ APSHUFHW,	yxshuf,	Pf3, {0x70,(00)} },
	{ APSHUFL,	yxshuf,	Pq, {0x70,(00)} },
	{ APSHUFLW,	yxshuf,	Pf2, {0x70,(00)} },
	{ APSHUFW,	ymshuf,	Pm, {0x70,(00)} },
	{ APSHUFB,	ymshufb,Pq, {0x38, 0x00} },
	{ APSLLO,	ypsdq,	Pq, {0x73,(07)} },
	{ APSLLL,	yps,	Py, {0xf2, 0x72,(06), Pe,0xf2, Pe,0x72,(06)} },
	{ APSLLQ,	yps,	Py, {0xf3, 0x73,(06), Pe,0xf3, Pe,0x73,(06)} },
	{ APSLLW,	yps,	Py, {0xf1, 0x71,(06), Pe,0xf1, Pe,0x71,(06)} },
	{ APSRAL,	yps,	Py, {0xe2, 0x72,(04), Pe,0xe2, Pe,0x72,(04)} },
	{ APSRAW,	yps,	Py, {0xe1, 0x71,(04), Pe,0xe1, Pe,0x71,(04)} },
	{ APSRLO,	ypsdq,	Pq, {0x73,(03)} },
	{ APSRLL,	yps,	Py, {0xd2, 0x72,(02), Pe,0xd2, Pe,0x72,(02)} },
	{ APSRLQ,	yps,	Py, {0xd3, 0x73,(02), Pe,0xd3, Pe,0x73,(02)} },
	{ APSRLW,	yps,	Py, {0xd1, 0x71,(02), Pe,0xe1, Pe,0x71,(02)} },
	{ APSUBB,	yxm,	Pe, {0xf8} },
	{ APSUBL,	yxm,	Pe, {0xfa} },
	{ APSUBQ,	yxm,	Pe, {0xfb} },
	{ APSUBSB,	yxm,	Pe, {0xe8} },
	{ APSUBSW,	yxm,	Pe, {0xe9} },
	{ APSUBUSB,	yxm,	Pe, {0xd8} },
	{ APSUBUSW,	yxm,	Pe, {0xd9} },
	{ APSUBW,	yxm,	Pe, {0xf9} },
	{ APSWAPL,	ymfp,	Px, {0xbb} },
	{ APUNPCKHBW,	ymm,	Py, {0x68,Pe,0x68} },
	{ APUNPCKHLQ,	ymm,	Py, {0x6a,Pe,0x6a} },
	{ APUNPCKHQDQ,	yxm,	Pe, {0x6d} },
	{ APUNPCKHWL,	ymm,	Py, {0x69,Pe,0x69} },
	{ APUNPCKLBW,	ymm,	Py, {0x60,Pe,0x60} },
	{ APUNPCKLLQ,	ymm,	Py, {0x62,Pe,0x62} },
	{ APUNPCKLQDQ,	yxm,	Pe, {0x6c} },
	{ APUNPCKLWL,	ymm,	Py, {0x61,Pe,0x61} },
	{ APUSHAL,	ynone,	P32, {0x60} },
	{ APUSHAW,	ynone,	Pe, {0x60} },
	{ APUSHFL,	ynone,	P32, {0x9c} },
	{ APUSHFQ,	ynone,	Py, {0x9c} },
	{ APUSHFW,	ynone,	Pe, {0x9c} },
	{ APUSHL,	ypushl,	P32, {0x50,0xff,(06),0x6a,0x68} },
	{ APUSHQ,	ypushl,	Py, {0x50,0xff,(06),0x6a,0x68} },
	{ APUSHW,	ypushl,	Pe, {0x50,0xff,(06),0x6a,0x68} },
	{ APXOR,	ymm,	Py, {0xef,Pe,0xef} },
	{ AQUAD,	ybyte,	Px, {8} },
	{ ARCLB,	yshb,	Pb, {0xd0,(02),0xc0,(02),0xd2,(02)} },
	{ ARCLL,	yshl,	Px, {0xd1,(02),0xc1,(02),0xd3,(02),0xd3,(02)} },
	{ ARCLQ,	yshl,	Pw, {0xd1,(02),0xc1,(02),0xd3,(02),0xd3,(02)} },
	{ ARCLW,	yshl,	Pe, {0xd1,(02),0xc1,(02),0xd3,(02),0xd3,(02)} },
	{ ARCPPS,	yxm,	Pm, {0x53} },
	{ ARCPSS,	yxm,	Pf3, {0x53} },
	{ ARCRB,	yshb,	Pb, {0xd0,(03),0xc0,(03),0xd2,(03)} },
	{ ARCRL,	yshl,	Px, {0xd1,(03),0xc1,(03),0xd3,(03),0xd3,(03)} },
	{ ARCRQ,	yshl,	Pw, {0xd1,(03),0xc1,(03),0xd3,(03),0xd3,(03)} },
	{ ARCRW,	yshl,	Pe, {0xd1,(03),0xc1,(03),0xd3,(03),0xd3,(03)} },
	{ AREP,		ynone,	Px, {0xf3} },
	{ AREPN,	ynone,	Px, {0xf2} },
	{ ARET,		ynone,	Px, {0xc3} },
	{ ARETFW,	yret,	Pe, {0xcb,0xca} },
	{ ARETFL,	yret,	Px, {0xcb,0xca} },
	{ ARETFQ,	yret,	Pw, {0xcb,0xca} },
	{ AROLB,	yshb,	Pb, {0xd0,(00),0xc0,(00),0xd2,(00)} },
	{ AROLL,	yshl,	Px, {0xd1,(00),0xc1,(00),0xd3,(00),0xd3,(00)} },
	{ AROLQ,	yshl,	Pw, {0xd1,(00),0xc1,(00),0xd3,(00),0xd3,(00)} },
	{ AROLW,	yshl,	Pe, {0xd1,(00),0xc1,(00),0xd3,(00),0xd3,(00)} },
	{ ARORB,	yshb,	Pb, {0xd0,(01),0xc0,(01),0xd2,(01)} },
	{ ARORL,	yshl,	Px, {0xd1,(01),0xc1,(01),0xd3,(01),0xd3,(01)} },
	{ ARORQ,	yshl,	Pw, {0xd1,(01),0xc1,(01),0xd3,(01),0xd3,(01)} },
	{ ARORW,	yshl,	Pe, {0xd1,(01),0xc1,(01),0xd3,(01),0xd3,(01)} },
	{ ARSQRTPS,	yxm,	Pm, {0x52} },
	{ ARSQRTSS,	yxm,	Pf3, {0x52} },
	{ ASAHF,	ynone,	Px, {0x86,0xe0,0x50,0x9d} },	/* XCHGB AH,AL; PUSH AX; POPFL */
	{ ASALB,	yshb,	Pb, {0xd0,(04),0xc0,(04),0xd2,(04)} },
	{ ASALL,	yshl,	Px, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASALQ,	yshl,	Pw, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASALW,	yshl,	Pe, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASARB,	yshb,	Pb, {0xd0,(07),0xc0,(07),0xd2,(07)} },
	{ ASARL,	yshl,	Px, {0xd1,(07),0xc1,(07),0xd3,(07),0xd3,(07)} },
	{ ASARQ,	yshl,	Pw, {0xd1,(07),0xc1,(07),0xd3,(07),0xd3,(07)} },
	{ ASARW,	yshl,	Pe, {0xd1,(07),0xc1,(07),0xd3,(07),0xd3,(07)} },
	{ ASBBB,	yxorb,	Pb, {0x1c,0x80,(03),0x18,0x1a} },
	{ ASBBL,	yxorl,	Px, {0x83,(03),0x1d,0x81,(03),0x19,0x1b} },
	{ ASBBQ,	yxorl,	Pw, {0x83,(03),0x1d,0x81,(03),0x19,0x1b} },
	{ ASBBW,	yxorl,	Pe, {0x83,(03),0x1d,0x81,(03),0x19,0x1b} },
	{ ASCASB,	ynone,	Pb, {0xae} },
	{ ASCASL,	ynone,	Px, {0xaf} },
	{ ASCASQ,	ynone,	Pw, {0xaf} },
	{ ASCASW,	ynone,	Pe, {0xaf} },
	{ ASETCC,	yscond,	Pb, {0x0f,0x93,(00)} },
	{ ASETCS,	yscond,	Pb, {0x0f,0x92,(00)} },
	{ ASETEQ,	yscond,	Pb, {0x0f,0x94,(00)} },
	{ ASETGE,	yscond,	Pb, {0x0f,0x9d,(00)} },
	{ ASETGT,	yscond,	Pb, {0x0f,0x9f,(00)} },
	{ ASETHI,	yscond,	Pb, {0x0f,0x97,(00)} },
	{ ASETLE,	yscond,	Pb, {0x0f,0x9e,(00)} },
	{ ASETLS,	yscond,	Pb, {0x0f,0x96,(00)} },
	{ ASETLT,	yscond,	Pb, {0x0f,0x9c,(00)} },
	{ ASETMI,	yscond,	Pb, {0x0f,0x98,(00)} },
	{ ASETNE,	yscond,	Pb, {0x0f,0x95,(00)} },
	{ ASETOC,	yscond,	Pb, {0x0f,0x91,(00)} },
	{ ASETOS,	yscond,	Pb, {0x0f,0x90,(00)} },
	{ ASETPC,	yscond,	Pb, {0x0f,0x9b,(00)} },
	{ ASETPL,	yscond,	Pb, {0x0f,0x99,(00)} },
	{ ASETPS,	yscond,	Pb, {0x0f,0x9a,(00)} },
	{ ASHLB,	yshb,	Pb, {0xd0,(04),0xc0,(04),0xd2,(04)} },
	{ ASHLL,	yshl,	Px, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASHLQ,	yshl,	Pw, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASHLW,	yshl,	Pe, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASHRB,	yshb,	Pb, {0xd0,(05),0xc0,(05),0xd2,(05)} },
	{ ASHRL,	yshl,	Px, {0xd1,(05),0xc1,(05),0xd3,(05),0xd3,(05)} },
	{ ASHRQ,	yshl,	Pw, {0xd1,(05),0xc1,(05),0xd3,(05),0xd3,(05)} },
	{ ASHRW,	yshl,	Pe, {0xd1,(05),0xc1,(05),0xd3,(05),0xd3,(05)} },
	{ ASHUFPD,	yxshuf,	Pq, {0xc6,(00)} },
	{ ASHUFPS,	yxshuf,	Pm, {0xc6,(00)} },
	{ ASQRTPD,	yxm,	Pe, {0x51} },
	{ ASQRTPS,	yxm,	Pm, {0x51} },
	{ ASQRTSD,	yxm,	Pf2, {0x51} },
	{ ASQRTSS,	yxm,	Pf3, {0x51} },
	{ ASTC,		ynone,	Px, {0xf9} },
	{ ASTD,		ynone,	Px, {0xfd} },
	{ ASTI,		ynone,	Px, {0xfb} },
	{ ASTMXCSR,	ysvrs,	Pm, {0xae,(03),0xae,(03)} },
	{ ASTOSB,	ynone,	Pb, {0xaa} },
	{ ASTOSL,	ynone,	Px, {0xab} },
	{ ASTOSQ,	ynone,	Pw, {0xab} },
	{ ASTOSW,	ynone,	Pe, {0xab} },
	{ ASUBB,	yxorb,	Pb, {0x2c,0x80,(05),0x28,0x2a} },
	{ ASUBL,	yaddl,	Px, {0x83,(05),0x2d,0x81,(05),0x29,0x2b} },
	{ ASUBPD,	yxm,	Pe, {0x5c} },
	{ ASUBPS,	yxm,	Pm, {0x5c} },
	{ ASUBQ,	yaddl,	Pw, {0x83,(05),0x2d,0x81,(05),0x29,0x2b} },
	{ ASUBSD,	yxm,	Pf2, {0x5c} },
	{ ASUBSS,	yxm,	Pf3, {0x5c} },
	{ ASUBW,	yaddl,	Pe, {0x83,(05),0x2d,0x81,(05),0x29,0x2b} },
	{ ASWAPGS,	ynone,	Pm, {0x01,0xf8} },
	{ ASYSCALL,	ynone,	Px, {0x0f,0x05} },	/* fast syscall */
	{ ATESTB,	ytestb,	Pb, {0xa8,0xf6,(00),0x84,0x84} },
	{ ATESTL,	ytestl,	Px, {0xa9,0xf7,(00),0x85,0x85} },
	{ ATESTQ,	ytestl,	Pw, {0xa9,0xf7,(00),0x85,0x85} },
	{ ATESTW,	ytestl,	Pe, {0xa9,0xf7,(00),0x85,0x85} },
	{ ATEXT,	ytext,	Px },
	{ AUCOMISD,	yxcmp,	Pe, {0x2e} },
	{ AUCOMISS,	yxcmp,	Pm, {0x2e} },
	{ AUNPCKHPD,	yxm,	Pe, {0x15} },
	{ AUNPCKHPS,	yxm,	Pm, {0x15} },
	{ AUNPCKLPD,	yxm,	Pe, {0x14} },
	{ AUNPCKLPS,	yxm,	Pm, {0x14} },
	{ AVERR,	ydivl,	Pm, {0x00,(04)} },
	{ AVERW,	ydivl,	Pm, {0x00,(05)} },
	{ AWAIT,	ynone,	Px, {0x9b} },
	{ AWORD,	ybyte,	Px, {2} },
	{ AXCHGB,	yml_mb,	Pb, {0x86,0x86} },
	{ AXCHGL,	yxchg,	Px, {0x90,0x90,0x87,0x87} },
	{ AXCHGQ,	yxchg,	Pw, {0x90,0x90,0x87,0x87} },
	{ AXCHGW,	yxchg,	Pe, {0x90,0x90,0x87,0x87} },
	{ AXLAT,	ynone,	Px, {0xd7} },
	{ AXORB,	yxorb,	Pb, {0x34,0x80,(06),0x30,0x32} },
	{ AXORL,	yxorl,	Px, {0x83,(06),0x35,0x81,(06),0x31,0x33} },
	{ AXORPD,	yxm,	Pe, {0x57} },
	{ AXORPS,	yxm,	Pm, {0x57} },
	{ AXORQ,	yxorl,	Pw, {0x83,(06),0x35,0x81,(06),0x31,0x33} },
	{ AXORW,	yxorl,	Pe, {0x83,(06),0x35,0x81,(06),0x31,0x33} },

	{ AFMOVB,	yfmvx,	Px, {0xdf,(04)} },
	{ AFMOVBP,	yfmvp,	Px, {0xdf,(06)} },
	{ AFMOVD,	yfmvd,	Px, {0xdd,(00),0xdd,(02),0xd9,(00),0xdd,(02)} },
	{ AFMOVDP,	yfmvdp,	Px, {0xdd,(03),0xdd,(03)} },
	{ AFMOVF,	yfmvf,	Px, {0xd9,(00),0xd9,(02)} },
	{ AFMOVFP,	yfmvp,	Px, {0xd9,(03)} },
	{ AFMOVL,	yfmvf,	Px, {0xdb,(00),0xdb,(02)} },
	{ AFMOVLP,	yfmvp,	Px, {0xdb,(03)} },
	{ AFMOVV,	yfmvx,	Px, {0xdf,(05)} },
	{ AFMOVVP,	yfmvp,	Px, {0xdf,(07)} },
	{ AFMOVW,	yfmvf,	Px, {0xdf,(00),0xdf,(02)} },
	{ AFMOVWP,	yfmvp,	Px, {0xdf,(03)} },
	{ AFMOVX,	yfmvx,	Px, {0xdb,(05)} },
	{ AFMOVXP,	yfmvp,	Px, {0xdb,(07)} },

	{ AFCOMB },
	{ AFCOMBP },
	{ AFCOMD,	yfadd,	Px, {0xdc,(02),0xd8,(02),0xdc,(02)} },	/* botch */
	{ AFCOMDP,	yfadd,	Px, {0xdc,(03),0xd8,(03),0xdc,(03)} },	/* botch */
	{ AFCOMDPP,	ycompp,	Px, {0xde,(03)} },
	{ AFCOMF,	yfmvx,	Px, {0xd8,(02)} },
	{ AFCOMFP,	yfmvx,	Px, {0xd8,(03)} },
	{ AFCOML,	yfmvx,	Px, {0xda,(02)} },
	{ AFCOMLP,	yfmvx,	Px, {0xda,(03)} },
	{ AFCOMW,	yfmvx,	Px, {0xde,(02)} },
	{ AFCOMWP,	yfmvx,	Px, {0xde,(03)} },

	{ AFUCOM,	ycompp,	Px, {0xdd,(04)} },
	{ AFUCOMP,	ycompp, Px, {0xdd,(05)} },
	{ AFUCOMPP,	ycompp,	Px, {0xda,(13)} },

	{ AFADDDP,	yfaddp,	Px, {0xde,(00)} },
	{ AFADDW,	yfmvx,	Px, {0xde,(00)} },
	{ AFADDL,	yfmvx,	Px, {0xda,(00)} },
	{ AFADDF,	yfmvx,	Px, {0xd8,(00)} },
	{ AFADDD,	yfadd,	Px, {0xdc,(00),0xd8,(00),0xdc,(00)} },

	{ AFMULDP,	yfaddp,	Px, {0xde,(01)} },
	{ AFMULW,	yfmvx,	Px, {0xde,(01)} },
	{ AFMULL,	yfmvx,	Px, {0xda,(01)} },
	{ AFMULF,	yfmvx,	Px, {0xd8,(01)} },
	{ AFMULD,	yfadd,	Px, {0xdc,(01),0xd8,(01),0xdc,(01)} },

	{ AFSUBDP,	yfaddp,	Px, {0xde,(05)} },
	{ AFSUBW,	yfmvx,	Px, {0xde,(04)} },
	{ AFSUBL,	yfmvx,	Px, {0xda,(04)} },
	{ AFSUBF,	yfmvx,	Px, {0xd8,(04)} },
	{ AFSUBD,	yfadd,	Px, {0xdc,(04),0xd8,(04),0xdc,(05)} },

	{ AFSUBRDP,	yfaddp,	Px, {0xde,(04)} },
	{ AFSUBRW,	yfmvx,	Px, {0xde,(05)} },
	{ AFSUBRL,	yfmvx,	Px, {0xda,(05)} },
	{ AFSUBRF,	yfmvx,	Px, {0xd8,(05)} },
	{ AFSUBRD,	yfadd,	Px, {0xdc,(05),0xd8,(05),0xdc,(04)} },

	{ AFDIVDP,	yfaddp,	Px, {0xde,(07)} },
	{ AFDIVW,	yfmvx,	Px, {0xde,(06)} },
	{ AFDIVL,	yfmvx,	Px, {0xda,(06)} },
	{ AFDIVF,	yfmvx,	Px, {0xd8,(06)} },
	{ AFDIVD,	yfadd,	Px, {0xdc,(06),0xd8,(06),0xdc,(07)} },

	{ AFDIVRDP,	yfaddp,	Px, {0xde,(06)} },
	{ AFDIVRW,	yfmvx,	Px, {0xde,(07)} },
	{ AFDIVRL,	yfmvx,	Px, {0xda,(07)} },
	{ AFDIVRF,	yfmvx,	Px, {0xd8,(07)} },
	{ AFDIVRD,	yfadd,	Px, {0xdc,(07),0xd8,(07),0xdc,(06)} },

	{ AFXCHD,	yfxch,	Px, {0xd9,(01),0xd9,(01)} },
	{ AFFREE },
	{ AFLDCW,	ystcw,	Px, {0xd9,(05),0xd9,(05)} },
	{ AFLDENV,	ystcw,	Px, {0xd9,(04),0xd9,(04)} },
	{ AFRSTOR,	ysvrs,	Px, {0xdd,(04),0xdd,(04)} },
	{ AFSAVE,	ysvrs,	Px, {0xdd,(06),0xdd,(06)} },
	{ AFSTCW,	ystcw,	Px, {0xd9,(07),0xd9,(07)} },
	{ AFSTENV,	ystcw,	Px, {0xd9,(06),0xd9,(06)} },
	{ AFSTSW,	ystsw,	Px, {0xdd,(07),0xdf,0xe0} },
	{ AF2XM1,	ynone,	Px, {0xd9, 0xf0} },
	{ AFABS,	ynone,	Px, {0xd9, 0xe1} },
	{ AFCHS,	ynone,	Px, {0xd9, 0xe0} },
	{ AFCLEX,	ynone,	Px, {0xdb, 0xe2} },
	{ AFCOS,	ynone,	Px, {0xd9, 0xff} },
	{ AFDECSTP,	ynone,	Px, {0xd9, 0xf6} },
	{ AFINCSTP,	ynone,	Px, {0xd9, 0xf7} },
	{ AFINIT,	ynone,	Px, {0xdb, 0xe3} },
	{ AFLD1,	ynone,	Px, {0xd9, 0xe8} },
	{ AFLDL2E,	ynone,	Px, {0xd9, 0xea} },
	{ AFLDL2T,	ynone,	Px, {0xd9, 0xe9} },
	{ AFLDLG2,	ynone,	Px, {0xd9, 0xec} },
	{ AFLDLN2,	ynone,	Px, {0xd9, 0xed} },
	{ AFLDPI,	ynone,	Px, {0xd9, 0xeb} },
	{ AFLDZ,	ynone,	Px, {0xd9, 0xee} },
	{ AFNOP,	ynone,	Px, {0xd9, 0xd0} },
	{ AFPATAN,	ynone,	Px, {0xd9, 0xf3} },
	{ AFPREM,	ynone,	Px, {0xd9, 0xf8} },
	{ AFPREM1,	ynone,	Px, {0xd9, 0xf5} },
	{ AFPTAN,	ynone,	Px, {0xd9, 0xf2} },
	{ AFRNDINT,	ynone,	Px, {0xd9, 0xfc} },
	{ AFSCALE,	ynone,	Px, {0xd9, 0xfd} },
	{ AFSIN,	ynone,	Px, {0xd9, 0xfe} },
	{ AFSINCOS,	ynone,	Px, {0xd9, 0xfb} },
	{ AFSQRT,	ynone,	Px, {0xd9, 0xfa} },
	{ AFTST,	ynone,	Px, {0xd9, 0xe4} },
	{ AFXAM,	ynone,	Px, {0xd9, 0xe5} },
	{ AFXTRACT,	ynone,	Px, {0xd9, 0xf4} },
	{ AFYL2X,	ynone,	Px, {0xd9, 0xf1} },
	{ AFYL2XP1,	ynone,	Px, {0xd9, 0xf9} },

	{ ACMPXCHGB,	yrb_mb,	Pb, {0x0f,0xb0} },
	{ ACMPXCHGL,	yrl_ml,	Px, {0x0f,0xb1} },
	{ ACMPXCHGW,	yrl_ml,	Pe, {0x0f,0xb1} },
	{ ACMPXCHGQ,	yrl_ml,	Pw, {0x0f,0xb1} },
	{ ACMPXCHG8B,	yscond,	Pm, {0xc7,(01)} },
	{ AINVD,	ynone,	Pm, {0x08} },
	{ AINVLPG,	ymbs,	Pm, {0x01,(07)} },
	{ ALFENCE,	ynone,	Pm, {0xae,0xe8} },
	{ AMFENCE,	ynone,	Pm, {0xae,0xf0} },
	{ AMOVNTIL,	yrl_ml,	Pm, {0xc3} },
	{ AMOVNTIQ,	yrl_ml, Pw, {0x0f,0xc3} },
	{ ARDMSR,	ynone,	Pm, {0x32} },
	{ ARDPMC,	ynone,	Pm, {0x33} },
	{ ARDTSC,	ynone,	Pm, {0x31} },
	{ ARSM,		ynone,	Pm, {0xaa} },
	{ ASFENCE,	ynone,	Pm, {0xae,0xf8} },
	{ ASYSRET,	ynone,	Pm, {0x07} },
	{ AWBINVD,	ynone,	Pm, {0x09} },
	{ AWRMSR,	ynone,	Pm, {0x30} },

	{ AXADDB,	yrb_mb,	Pb, {0x0f,0xc0} },
	{ AXADDL,	yrl_ml,	Px, {0x0f,0xc1} },
	{ AXADDQ,	yrl_ml,	Pw, {0x0f,0xc1} },
	{ AXADDW,	yrl_ml,	Pe, {0x0f,0xc1} },

	{ ACRC32B,       ycrc32l,Px, {0xf2,0x0f,0x38,0xf0,0} },
	{ ACRC32Q,       ycrc32l,Pw, {0xf2,0x0f,0x38,0xf1,0} },
	
	{ APREFETCHT0,	yprefetch,	Pm,	{0x18,(01)} },
	{ APREFETCHT1,	yprefetch,	Pm,	{0x18,(02)} },
	{ APREFETCHT2,	yprefetch,	Pm,	{0x18,(03)} },
	{ APREFETCHNTA,	yprefetch,	Pm,	{0x18,(00)} },
	
	{ AMOVQL,	yrl_ml,	Px, {0x89} },

	{ AUNDEF,		ynone,	Px, {0x0f, 0x0b} },

	{ AAESENC,	yaes,	Pq, {0x38,0xdc,(0)} },
	{ AAESENCLAST,	yaes,	Pq, {0x38,0xdd,(0)} },
	{ AAESDEC,	yaes,	Pq, {0x38,0xde,(0)} },
	{ AAESDECLAST,	yaes,	Pq, {0x38,0xdf,(0)} },
	{ AAESIMC,	yaes,	Pq, {0x38,0xdb,(0)} },
	{ AAESKEYGENASSIST,	yaes2,	Pq, {0x3a,0xdf,(0)} },

	{ APSHUFD,	yaes2,	Pq,	{0x70,(0)} },
	{ APCLMULQDQ,	yxshuf,	Pq, {0x3a,0x44,0} },

	{ AUSEFIELD,	ynop,	Px, {0,0} },
	{ ATYPE },
	{ AFUNCDATA,	yfuncdata,	Px, {0,0} },
	{ APCDATA,	ypcdata,	Px, {0,0} },
	{ ACHECKNIL },
	{ AVARDEF },
	{ AVARKILL },
	{ ADUFFCOPY,	yduff,	Px, {0xe8} },
	{ ADUFFZERO,	yduff,	Px, {0xe8} },

	{ AEND },
	{0}
};

static Optab*	opindex[ALAST+1];
static vlong	vaddr(Link*, Prog*, Addr*, Reloc*);

// isextern reports whether s describes an external symbol that must avoid pc-relative addressing.
// This happens on systems like Solaris that call .so functions instead of system calls.
// It does not seem to be necessary for any other systems. This is probably working
// around a Solaris-specific bug that should be fixed differently, but we don't know
// what that bug is. And this does fix it.
static int
isextern(LSym *s)
{
	// All the Solaris dynamic imports from libc.so begin with "libc_".
	return strncmp(s->name, "libc_", 5) == 0;
}

// single-instruction no-ops of various lengths.
// constructed by hand and disassembled with gdb to verify.
// see http://www.agner.org/optimize/optimizing_assembly.pdf for discussion.
static uchar nop[][16] = {
	{0x90},
	{0x66, 0x90},
	{0x0F, 0x1F, 0x00},
	{0x0F, 0x1F, 0x40, 0x00},
	{0x0F, 0x1F, 0x44, 0x00, 0x00},
	{0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00},
	{0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00},
	{0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
	{0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
	// Native Client rejects the repeated 0x66 prefix.
	// {0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
};

static void
fillnop(uchar *p, int n)
{
	int m;

	while(n > 0) {
		m = n;
		if(m > nelem(nop))
			m = nelem(nop);
		memmove(p, nop[m-1], m);
		p += m;
		n -= m;
	}
}

static void instinit(void);

static int32
naclpad(Link *ctxt, LSym *s, int32 c, int32 pad)
{
	symgrow(ctxt, s, c+pad);
	fillnop(s->p+c, pad);
	return c+pad;
}

static int
spadjop(Link *ctxt, Prog *p, int l, int q)
{
	if(p->mode != 64 || ctxt->arch->ptrsize == 4)
		return l;
	return q;
}

void
span6(Link *ctxt, LSym *s)
{
	Prog *p, *q;
	int32 c, v, loop;
	uchar *bp;
	int n, m, i;

	ctxt->cursym = s;
	
	if(s->p != nil)
		return;
	
	if(ycover[0] == 0)
		instinit();
	
	for(p = ctxt->cursym->text; p != nil; p = p->link) {
		if(p->to.type == TYPE_BRANCH)
			if(p->pcond == nil)
				p->pcond = p;
		if(p->as == AADJSP) {
			p->to.type = TYPE_REG;
			p->to.reg = REG_SP;
			v = -p->from.offset;
			p->from.offset = v;
			p->as = spadjop(ctxt, p, AADDL, AADDQ);
			if(v < 0) {
				p->as = spadjop(ctxt, p, ASUBL, ASUBQ);
				v = -v;
				p->from.offset = v;
			}
			if(v == 0)
				p->as = ANOP;
		}
	}

	for(p = s->text; p != nil; p = p->link) {
		p->back = 2;	// use short branches first time through
		if((q = p->pcond) != nil && (q->back & 2)) {
			p->back |= 1;	// backward jump
			q->back |= 4;   // loop head
		}

		if(p->as == AADJSP) {
			p->to.type = TYPE_REG;
			p->to.reg = REG_SP;
			v = -p->from.offset;
			p->from.offset = v;
			p->as = spadjop(ctxt, p, AADDL, AADDQ);
			if(v < 0) {
				p->as = spadjop(ctxt, p, ASUBL, ASUBQ);
				v = -v;
				p->from.offset = v;
			}
			if(v == 0)
				p->as = ANOP;
		}
	}
	
	n = 0;
	do {
		loop = 0;
		memset(s->r, 0, s->nr*sizeof s->r[0]);
		s->nr = 0;
		s->np = 0;
		c = 0;
		for(p = s->text; p != nil; p = p->link) {
			if(ctxt->headtype == Hnacl && p->isize > 0) {
				static LSym *deferreturn;
				
				if(deferreturn == nil)
					deferreturn = linklookup(ctxt, "runtime.deferreturn", 0);

				// pad everything to avoid crossing 32-byte boundary
				if((c>>5) != ((c+p->isize-1)>>5))
					c = naclpad(ctxt, s, c, -c&31);
				// pad call deferreturn to start at 32-byte boundary
				// so that subtracting 5 in jmpdefer will jump back
				// to that boundary and rerun the call.
				if(p->as == ACALL && p->to.sym == deferreturn)
					c = naclpad(ctxt, s, c, -c&31);
				// pad call to end at 32-byte boundary
				if(p->as == ACALL)
					c = naclpad(ctxt, s, c, -(c+p->isize)&31);
				
				// the linker treats REP and STOSQ as different instructions
				// but in fact the REP is a prefix on the STOSQ.
				// make sure REP has room for 2 more bytes, so that
				// padding will not be inserted before the next instruction.
				if((p->as == AREP || p->as == AREPN) && (c>>5) != ((c+3-1)>>5))
					c = naclpad(ctxt, s, c, -c&31);
				
				// same for LOCK.
				// various instructions follow; the longest is 4 bytes.
				// give ourselves 8 bytes so as to avoid surprises.
				if(p->as == ALOCK && (c>>5) != ((c+8-1)>>5))
					c = naclpad(ctxt, s, c, -c&31);
			}

			if((p->back & 4) && (c&(LoopAlign-1)) != 0) {
				// pad with NOPs
				v = -c&(LoopAlign-1);
				if(v <= MaxLoopPad) {
					symgrow(ctxt, s, c+v);
					fillnop(s->p+c, v);
					c += v;
				}
			}

			p->pc = c;

			// process forward jumps to p
			for(q = p->comefrom; q != nil; q = q->forwd) {
				v = p->pc - (q->pc + q->mark);
				if(q->back & 2)	{	// short
					if(v > 127) {
						loop++;
						q->back ^= 2;
					}
					if(q->as == AJCXZL)
						s->p[q->pc+2] = v;
					else
						s->p[q->pc+1] = v;
				} else {
					bp = s->p + q->pc + q->mark - 4;
					*bp++ = v;
					*bp++ = v>>8;
					*bp++ = v>>16;
					*bp = v>>24;
				}	
			}
			p->comefrom = nil;

			p->pc = c;
			asmins(ctxt, p);
			m = ctxt->andptr-ctxt->and;
			if(p->isize != m) {
				p->isize = m;
				loop++;
			}
			symgrow(ctxt, s, p->pc+m);
			memmove(s->p+p->pc, ctxt->and, m);
			p->mark = m;
			c += m;
		}
		if(++n > 20) {
			ctxt->diag("span must be looping");
			sysfatal("loop");
		}
	} while(loop);
	
	if(ctxt->headtype == Hnacl)
		c = naclpad(ctxt, s, c, -c&31);
	
	c += -c&(FuncAlign-1);
	s->size = c;

	if(0 /* debug['a'] > 1 */) {
		print("span1 %s %lld (%d tries)\n %.6ux", s->name, s->size, n, 0);
		for(i=0; i<s->np; i++) {
			print(" %.2ux", s->p[i]);
			if(i%16 == 15)
				print("\n  %.6ux", i+1);
		}
		if(i%16)
			print("\n");
	
		for(i=0; i<s->nr; i++) {
			Reloc *r;
			
			r = &s->r[i];
			print(" rel %#.4ux/%d %s%+lld\n", r->off, r->siz, r->sym->name, r->add);
		}
	}
}

static void
instinit(void)
{
	int c, i;

	for(i=1; optab[i].as; i++) {
		c = optab[i].as;
		if(opindex[c] != nil)
			sysfatal("phase error in optab: %d (%A)", i, c);
		opindex[c] = &optab[i];
	}

	for(i=0; i<Ymax; i++)
		ycover[i*Ymax + i] = 1;

	ycover[Yi0*Ymax + Yi8] = 1;
	ycover[Yi1*Ymax + Yi8] = 1;

	ycover[Yi0*Ymax + Ys32] = 1;
	ycover[Yi1*Ymax + Ys32] = 1;
	ycover[Yi8*Ymax + Ys32] = 1;

	ycover[Yi0*Ymax + Yi32] = 1;
	ycover[Yi1*Ymax + Yi32] = 1;
	ycover[Yi8*Ymax + Yi32] = 1;
	ycover[Ys32*Ymax + Yi32] = 1;

	ycover[Yi0*Ymax + Yi64] = 1;
	ycover[Yi1*Ymax + Yi64] = 1;
	ycover[Yi8*Ymax + Yi64] = 1;
	ycover[Ys32*Ymax + Yi64] = 1;
	ycover[Yi32*Ymax + Yi64] = 1;

	ycover[Yal*Ymax + Yrb] = 1;
	ycover[Ycl*Ymax + Yrb] = 1;
	ycover[Yax*Ymax + Yrb] = 1;
	ycover[Ycx*Ymax + Yrb] = 1;
	ycover[Yrx*Ymax + Yrb] = 1;
	ycover[Yrl*Ymax + Yrb] = 1;

	ycover[Ycl*Ymax + Ycx] = 1;

	ycover[Yax*Ymax + Yrx] = 1;
	ycover[Ycx*Ymax + Yrx] = 1;

	ycover[Yax*Ymax + Yrl] = 1;
	ycover[Ycx*Ymax + Yrl] = 1;
	ycover[Yrx*Ymax + Yrl] = 1;

	ycover[Yf0*Ymax + Yrf] = 1;

	ycover[Yal*Ymax + Ymb] = 1;
	ycover[Ycl*Ymax + Ymb] = 1;
	ycover[Yax*Ymax + Ymb] = 1;
	ycover[Ycx*Ymax + Ymb] = 1;
	ycover[Yrx*Ymax + Ymb] = 1;
	ycover[Yrb*Ymax + Ymb] = 1;
	ycover[Yrl*Ymax + Ymb] = 1;
	ycover[Ym*Ymax + Ymb] = 1;

	ycover[Yax*Ymax + Yml] = 1;
	ycover[Ycx*Ymax + Yml] = 1;
	ycover[Yrx*Ymax + Yml] = 1;
	ycover[Yrl*Ymax + Yml] = 1;
	ycover[Ym*Ymax + Yml] = 1;

	ycover[Yax*Ymax + Ymm] = 1;
	ycover[Ycx*Ymax + Ymm] = 1;
	ycover[Yrx*Ymax + Ymm] = 1;
	ycover[Yrl*Ymax + Ymm] = 1;
	ycover[Ym*Ymax + Ymm] = 1;
	ycover[Ymr*Ymax + Ymm] = 1;

	ycover[Ym*Ymax + Yxm] = 1;
	ycover[Yxr*Ymax + Yxm] = 1;

	for(i=0; i<MAXREG; i++) {
		reg[i] = -1;
		if(i >= REG_AL && i <= REG_R15B) {
			reg[i] = (i-REG_AL) & 7;
			if(i >= REG_SPB && i <= REG_DIB)
				regrex[i] = 0x40;
			if(i >= REG_R8B && i <= REG_R15B)
				regrex[i] = Rxr | Rxx | Rxb;
		}
		if(i >= REG_AH && i<= REG_BH)
			reg[i] = 4 + ((i-REG_AH) & 7);
		if(i >= REG_AX && i <= REG_R15) {
			reg[i] = (i-REG_AX) & 7;
			if(i >= REG_R8)
				regrex[i] = Rxr | Rxx | Rxb;
		}
		if(i >= REG_F0 && i <= REG_F0+7)
			reg[i] = (i-REG_F0) & 7;
		if(i >= REG_M0 && i <= REG_M0+7)
			reg[i] = (i-REG_M0) & 7;
		if(i >= REG_X0 && i <= REG_X0+15) {
			reg[i] = (i-REG_X0) & 7;
			if(i >= REG_X0+8)
				regrex[i] = Rxr | Rxx | Rxb;
		}
		if(i >= REG_CR+8 && i <= REG_CR+15)
			regrex[i] = Rxr;
	}
}

static int
prefixof(Link *ctxt, Addr *a)
{
	if(a->type == TYPE_MEM && a->name == NAME_NONE) {
		switch(a->reg) {
		case REG_CS:
			return 0x2e;
		case REG_DS:
			return 0x3e;
		case REG_ES:
			return 0x26;
		case REG_FS:
			return 0x64;
		case REG_GS:
			return 0x65;
		case REG_TLS:
			// NOTE: Systems listed here should be only systems that
			// support direct TLS references like 8(TLS) implemented as
			// direct references from FS or GS. Systems that require
			// the initial-exec model, where you load the TLS base into
			// a register and then index from that register, do not reach
			// this code and should not be listed.
			switch(ctxt->headtype) {
			default:
				sysfatal("unknown TLS base register for %s", headstr(ctxt->headtype));
			case Hdragonfly:
			case Hfreebsd:
			case Hlinux:
			case Hnetbsd:
			case Hopenbsd:
			case Hsolaris:
				return 0x64; // FS
			case Hdarwin:
				return 0x65; // GS
			}
		}
	}
	switch(a->index) {
	case REG_CS:
		return 0x2e;
	case REG_DS:
		return 0x3e;
	case REG_ES:
		return 0x26;
	case REG_FS:
		return 0x64;
	case REG_GS:
		return 0x65;
	}
	return 0;
}

static int
oclass(Link *ctxt, Addr *a)
{
	vlong v;
	int32 l;
	
	// TODO(rsc): This special case is for SHRQ $3, AX:DX,
	// which encodes as SHRQ $32(DX*0), AX.
	// Similarly SHRQ CX, AX:DX is really SHRQ CX(DX*0), AX.
	// Change encoding and remove.
	if((a->type == TYPE_CONST || a->type == TYPE_REG) && a->index != REG_NONE && a->scale == 0)
		return Ycol;

	switch(a->type) {
	case TYPE_NONE:
		return Ynone;

	case TYPE_BRANCH:
		return Ybr;

	case TYPE_MEM:
		return Ym;

	case TYPE_ADDR:
		switch(a->name) {
		case NAME_EXTERN:
		case NAME_STATIC:
			if(a->sym != nil && isextern(a->sym))
				return Yi32;
			return Yiauto; // use pc-relative addressing
		case NAME_AUTO:
		case NAME_PARAM:
			return Yiauto;
		}

		// TODO(rsc): DUFFZERO/DUFFCOPY encoding forgot to set a->index
		// and got Yi32 in an earlier version of this code.
		// Keep doing that until we fix yduff etc.
		if(a->sym != nil && strncmp(a->sym->name, "runtime.duff", 12) == 0)
			return Yi32;
		
		if(a->sym != nil || a->name != NAME_NONE)
			ctxt->diag("unexpected addr: %D", a);
		// fall through

	case TYPE_CONST:
		if(a->sym != nil)
			ctxt->diag("TYPE_CONST with symbol: %D", a);

		v = a->offset;
		if(v == 0)
			return Yi0;
		if(v == 1)
			return Yi1;
		if(v >= -128 && v <= 127)
			return Yi8;
		l = v;
		if((vlong)l == v)
			return Ys32;	/* can sign extend */
		if((v>>32) == 0)
			return Yi32;	/* unsigned */
		return Yi64;

	case TYPE_TEXTSIZE:
		return Ytextsize;
	}
	
	if(a->type != TYPE_REG) {
		ctxt->diag("unexpected addr1: type=%d %D", a->type, a);
		return Yxxx;
	}

	switch(a->reg) {
	case REG_AL:
		return Yal;

	case REG_AX:
		return Yax;

/*
	case REG_SPB:
*/
	case REG_BPB:
	case REG_SIB:
	case REG_DIB:
	case REG_R8B:
	case REG_R9B:
	case REG_R10B:
	case REG_R11B:
	case REG_R12B:
	case REG_R13B:
	case REG_R14B:
	case REG_R15B:
		if(ctxt->asmode != 64)
			return Yxxx;
	case REG_DL:
	case REG_BL:
	case REG_AH:
	case REG_CH:
	case REG_DH:
	case REG_BH:
		return Yrb;

	case REG_CL:
		return Ycl;

	case REG_CX:
		return Ycx;

	case REG_DX:
	case REG_BX:
		return Yrx;

	case REG_R8:	/* not really Yrl */
	case REG_R9:
	case REG_R10:
	case REG_R11:
	case REG_R12:
	case REG_R13:
	case REG_R14:
	case REG_R15:
		if(ctxt->asmode != 64)
			return Yxxx;
	case REG_SP:
	case REG_BP:
	case REG_SI:
	case REG_DI:
		return Yrl;

	case REG_F0+0:
		return	Yf0;

	case REG_F0+1:
	case REG_F0+2:
	case REG_F0+3:
	case REG_F0+4:
	case REG_F0+5:
	case REG_F0+6:
	case REG_F0+7:
		return	Yrf;

	case REG_M0+0:
	case REG_M0+1:
	case REG_M0+2:
	case REG_M0+3:
	case REG_M0+4:
	case REG_M0+5:
	case REG_M0+6:
	case REG_M0+7:
		return	Ymr;

	case REG_X0+0:
	case REG_X0+1:
	case REG_X0+2:
	case REG_X0+3:
	case REG_X0+4:
	case REG_X0+5:
	case REG_X0+6:
	case REG_X0+7:
	case REG_X0+8:
	case REG_X0+9:
	case REG_X0+10:
	case REG_X0+11:
	case REG_X0+12:
	case REG_X0+13:
	case REG_X0+14:
	case REG_X0+15:
		return	Yxr;

	case REG_CS:	return	Ycs;
	case REG_SS:	return	Yss;
	case REG_DS:	return	Yds;
	case REG_ES:	return	Yes;
	case REG_FS:	return	Yfs;
	case REG_GS:	return	Ygs;
	case REG_TLS:	return	Ytls;

	case REG_GDTR:	return	Ygdtr;
	case REG_IDTR:	return	Yidtr;
	case REG_LDTR:	return	Yldtr;
	case REG_MSW:	return	Ymsw;
	case REG_TASK:	return	Ytask;

	case REG_CR+0:	return	Ycr0;
	case REG_CR+1:	return	Ycr1;
	case REG_CR+2:	return	Ycr2;
	case REG_CR+3:	return	Ycr3;
	case REG_CR+4:	return	Ycr4;
	case REG_CR+5:	return	Ycr5;
	case REG_CR+6:	return	Ycr6;
	case REG_CR+7:	return	Ycr7;
	case REG_CR+8:	return	Ycr8;

	case REG_DR+0:	return	Ydr0;
	case REG_DR+1:	return	Ydr1;
	case REG_DR+2:	return	Ydr2;
	case REG_DR+3:	return	Ydr3;
	case REG_DR+4:	return	Ydr4;
	case REG_DR+5:	return	Ydr5;
	case REG_DR+6:	return	Ydr6;
	case REG_DR+7:	return	Ydr7;

	case REG_TR+0:	return	Ytr0;
	case REG_TR+1:	return	Ytr1;
	case REG_TR+2:	return	Ytr2;
	case REG_TR+3:	return	Ytr3;
	case REG_TR+4:	return	Ytr4;
	case REG_TR+5:	return	Ytr5;
	case REG_TR+6:	return	Ytr6;
	case REG_TR+7:	return	Ytr7;

	}
	return Yxxx;
}

static void
asmidx(Link *ctxt, int scale, int index, int base)
{
	int i;

	switch(index) {
	default:
		goto bad;

	case REG_NONE:
		i = 4 << 3;
		goto bas;

	case REG_R8:
	case REG_R9:
	case REG_R10:
	case REG_R11:
	case REG_R12:
	case REG_R13:
	case REG_R14:
	case REG_R15:
		if(ctxt->asmode != 64)
			goto bad;
	case REG_AX:
	case REG_CX:
	case REG_DX:
	case REG_BX:
	case REG_BP:
	case REG_SI:
	case REG_DI:
		i = reg[index] << 3;
		break;
	}
	switch(scale) {
	default:
		goto bad;
	case 1:
		break;
	case 2:
		i |= (1<<6);
		break;
	case 4:
		i |= (2<<6);
		break;
	case 8:
		i |= (3<<6);
		break;
	}
bas:
	switch(base) {
	default:
		goto bad;
	case REG_NONE:	/* must be mod=00 */
		i |= 5;
		break;
	case REG_R8:
	case REG_R9:
	case REG_R10:
	case REG_R11:
	case REG_R12:
	case REG_R13:
	case REG_R14:
	case REG_R15:
		if(ctxt->asmode != 64)
			goto bad;
	case REG_AX:
	case REG_CX:
	case REG_DX:
	case REG_BX:
	case REG_SP:
	case REG_BP:
	case REG_SI:
	case REG_DI:
		i |= reg[base];
		break;
	}
	*ctxt->andptr++ = i;
	return;
bad:
	ctxt->diag("asmidx: bad address %d/%d/%d", scale, index, base);
	*ctxt->andptr++ = 0;
	return;
}

static void
put4(Link *ctxt, int32 v)
{
	ctxt->andptr[0] = v;
	ctxt->andptr[1] = v>>8;
	ctxt->andptr[2] = v>>16;
	ctxt->andptr[3] = v>>24;
	ctxt->andptr += 4;
}

static void
relput4(Link *ctxt, Prog *p, Addr *a)
{
	vlong v;
	Reloc rel, *r;
	
	v = vaddr(ctxt, p, a, &rel);
	if(rel.siz != 0) {
		if(rel.siz != 4)
			ctxt->diag("bad reloc");
		r = addrel(ctxt->cursym);
		*r = rel;
		r->off = p->pc + ctxt->andptr - ctxt->and;
	}
	put4(ctxt, v);
}

static void
put8(Link *ctxt, vlong v)
{
	ctxt->andptr[0] = v;
	ctxt->andptr[1] = v>>8;
	ctxt->andptr[2] = v>>16;
	ctxt->andptr[3] = v>>24;
	ctxt->andptr[4] = v>>32;
	ctxt->andptr[5] = v>>40;
	ctxt->andptr[6] = v>>48;
	ctxt->andptr[7] = v>>56;
	ctxt->andptr += 8;
}

/*
static void
relput8(Prog *p, Addr *a)
{
	vlong v;
	Reloc rel, *r;
	
	v = vaddr(ctxt, p, a, &rel);
	if(rel.siz != 0) {
		r = addrel(ctxt->cursym);
		*r = rel;
		r->siz = 8;
		r->off = p->pc + ctxt->andptr - ctxt->and;
	}
	put8(ctxt, v);
}
*/

static vlong
vaddr(Link *ctxt, Prog *p, Addr *a, Reloc *r)
{
	LSym *s;
	
	USED(p);

	if(r != nil)
		memset(r, 0, sizeof *r);

	switch(a->name) {
	case NAME_STATIC:
	case NAME_EXTERN:
		s = a->sym;
		if(r == nil) {
			ctxt->diag("need reloc for %D", a);
			sysfatal("reloc");
		}
		if(isextern(s)) {
			r->siz = 4;
			r->type = R_ADDR;
		} else {
			r->siz = 4;
			r->type = R_PCREL;
		}
		r->off = -1;	// caller must fill in
		r->sym = s;
		r->add = a->offset;
		if(s->type == STLSBSS) {
			r->xadd = r->add - r->siz;
			r->type = R_TLS;
			r->xsym = s;
		}
		return 0;
	}
	
	if((a->type == TYPE_MEM || a->type == TYPE_ADDR) && a->reg == REG_TLS) {
		if(r == nil) {
			ctxt->diag("need reloc for %D", a);
			sysfatal("reloc");
		}
		r->type = R_TLS_LE;
		r->siz = 4;
		r->off = -1;	// caller must fill in
		r->add = a->offset;
		return 0;
	}

	return a->offset;
}

static void
asmandsz(Link *ctxt, Prog *p, Addr *a, int r, int rex, int m64)
{
	int32 v;
	int base;
	Reloc rel;

	USED(m64);
	USED(p);

	rex &= (0x40 | Rxr);
	v = a->offset;
	rel.siz = 0;

	switch(a->type) {
	case TYPE_ADDR:
		if(a->name == NAME_NONE)
			ctxt->diag("unexpected TYPE_ADDR with NAME_NONE");
		if(a->index == REG_TLS)
			ctxt->diag("unexpected TYPE_ADDR with index==REG_TLS");
		goto bad;
	
	case TYPE_REG:
		if(a->reg < REG_AL || REG_X0+15 < a->reg)
			goto bad;
		if(v)
			goto bad;
		*ctxt->andptr++ = (3 << 6) | (reg[a->reg] << 0) | (r << 3);
		ctxt->rexflag |= (regrex[a->reg] & (0x40 | Rxb)) | rex;
		return;
	}

	if(a->type != TYPE_MEM)
		goto bad;

	if(a->index != REG_NONE && a->index != REG_TLS) {
		base = a->reg;
		switch(a->name) {
		case NAME_EXTERN:
		case NAME_STATIC:
			if(!isextern(a->sym))
				goto bad;
			base = REG_NONE;
			v = vaddr(ctxt, p, a, &rel);
			break;
		case NAME_AUTO:
		case NAME_PARAM:
			base = REG_SP;
			break;
		}
		
		ctxt->rexflag |= (regrex[(int)a->index] & Rxx) | (regrex[base] & Rxb) | rex;
		if(base == REG_NONE) {
			*ctxt->andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, a->scale, a->index, base);
			goto putrelv;
		}
		if(v == 0 && rel.siz == 0 && base != REG_BP && base != REG_R13) {
			*ctxt->andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, a->scale, a->index, base);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0) {
			*ctxt->andptr++ = (1 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, a->scale, a->index, base);
			*ctxt->andptr++ = v;
			return;
		}
		*ctxt->andptr++ = (2 << 6) | (4 << 0) | (r << 3);
		asmidx(ctxt, a->scale, a->index, base);
		goto putrelv;
	}

	base = a->reg;
	switch(a->name) {
	case NAME_STATIC:
	case NAME_EXTERN:
		if(a->sym == nil)
			ctxt->diag("bad addr: %P", p);
		base = REG_NONE;
		v = vaddr(ctxt, p, a, &rel);
		break;
	case NAME_AUTO:
	case NAME_PARAM:
		base = REG_SP;
		break;
	}

	if(base == REG_TLS)
		v = vaddr(ctxt, p, a, &rel);
	
	ctxt->rexflag |= (regrex[base] & Rxb) | rex;
	if(base == REG_NONE || (REG_CS <= base && base <= REG_GS) || base == REG_TLS) {
		if((a->sym == nil || !isextern(a->sym)) && base == REG_NONE && (a->name == NAME_STATIC || a->name == NAME_EXTERN) || ctxt->asmode != 64) {
			*ctxt->andptr++ = (0 << 6) | (5 << 0) | (r << 3);
			goto putrelv;
		}
		/* temporary */
		*ctxt->andptr++ = (0 <<  6) | (4 << 0) | (r << 3);	/* sib present */
		*ctxt->andptr++ = (0 << 6) | (4 << 3) | (5 << 0);	/* DS:d32 */
		goto putrelv;
	}

	if(base == REG_SP || base == REG_R12) {
		if(v == 0) {
			*ctxt->andptr++ = (0 << 6) | (reg[base] << 0) | (r << 3);
			asmidx(ctxt, a->scale, REG_NONE, base);
			return;
		}
		if(v >= -128 && v < 128) {
			*ctxt->andptr++ = (1 << 6) | (reg[base] << 0) | (r << 3);
			asmidx(ctxt, a->scale, REG_NONE, base);
			*ctxt->andptr++ = v;
			return;
		}
		*ctxt->andptr++ = (2 << 6) | (reg[base] << 0) | (r << 3);
		asmidx(ctxt, a->scale, REG_NONE, base);
		goto putrelv;
	}

	if(REG_AX <= base && base <= REG_R15) {
		if(a->index == REG_TLS) {
			memset(&rel, 0, sizeof rel);
			rel.type = R_TLS_IE;
			rel.siz = 4;
			rel.sym = nil;
			rel.add = v;
			v = 0;
		}
		if(v == 0 && rel.siz == 0 && base != REG_BP && base != REG_R13) {
			*ctxt->andptr++ = (0 << 6) | (reg[base] << 0) | (r << 3);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0) {
			ctxt->andptr[0] = (1 << 6) | (reg[base] << 0) | (r << 3);
			ctxt->andptr[1] = v;
			ctxt->andptr += 2;
			return;
		}
		*ctxt->andptr++ = (2 << 6) | (reg[base] << 0) | (r << 3);
		goto putrelv;
	}

	goto bad;
	
putrelv:
	if(rel.siz != 0) {
		Reloc *r;

		if(rel.siz != 4) {
			ctxt->diag("bad rel");
			goto bad;
		}
		r = addrel(ctxt->cursym);
		*r = rel;
		r->off = ctxt->curp->pc + ctxt->andptr - ctxt->and;
	}
		
	put4(ctxt, v);
	return;

bad:
	ctxt->diag("asmand: bad address %D", a);
	return;
}

static void
asmand(Link *ctxt, Prog *p, Addr *a, Addr *ra)
{
	asmandsz(ctxt, p, a, reg[ra->reg], regrex[ra->reg], 0);
}

static void
asmando(Link *ctxt, Prog *p, Addr *a, int o)
{
	asmandsz(ctxt, p, a, o, 0, 0);
}

static void
bytereg(Addr *a, uint8 *t)
{
	if(a->type == TYPE_REG && a->index == REG_NONE && (REG_AX <= a->reg && a->reg <= REG_R15)) {
		a->reg += REG_AL - REG_AX;
		*t = 0;
	}
}

enum {
	E = 0xff,
};
static Movtab	ymovtab[] =
{
/* push */
	{APUSHL,	Ycs,	Ynone,	0,	{0x0e,E,0,0}},
	{APUSHL,	Yss,	Ynone,	0,	{0x16,E,0,0}},
	{APUSHL,	Yds,	Ynone,	0,	{0x1e,E,0,0}},
	{APUSHL,	Yes,	Ynone,	0,	{0x06,E,0,0}},
	{APUSHL,	Yfs,	Ynone,	0,	{0x0f,0xa0,E,0}},
	{APUSHL,	Ygs,	Ynone,	0,	{0x0f,0xa8,E,0}},
	{APUSHQ,	Yfs,	Ynone,	0,	{0x0f,0xa0,E,0}},
	{APUSHQ,	Ygs,	Ynone,	0,	{0x0f,0xa8,E,0}},

	{APUSHW,	Ycs,	Ynone,	0,	{Pe,0x0e,E,0}},
	{APUSHW,	Yss,	Ynone,	0,	{Pe,0x16,E,0}},
	{APUSHW,	Yds,	Ynone,	0,	{Pe,0x1e,E,0}},
	{APUSHW,	Yes,	Ynone,	0,	{Pe,0x06,E,0}},
	{APUSHW,	Yfs,	Ynone,	0,	{Pe,0x0f,0xa0,E}},
	{APUSHW,	Ygs,	Ynone,	0,	{Pe,0x0f,0xa8,E}},

/* pop */
	{APOPL,	Ynone,	Yds,	0,	{0x1f,E,0,0}},
	{APOPL,	Ynone,	Yes,	0,	{0x07,E,0,0}},
	{APOPL,	Ynone,	Yss,	0,	{0x17,E,0,0}},
	{APOPL,	Ynone,	Yfs,	0,	{0x0f,0xa1,E,0}},
	{APOPL,	Ynone,	Ygs,	0,	{0x0f,0xa9,E,0}},
	{APOPQ,	Ynone,	Yfs,	0,	{0x0f,0xa1,E,0}},
	{APOPQ,	Ynone,	Ygs,	0,	{0x0f,0xa9,E,0}},

	{APOPW,	Ynone,	Yds,	0,	{Pe,0x1f,E,0}},
	{APOPW,	Ynone,	Yes,	0,	{Pe,0x07,E,0}},
	{APOPW,	Ynone,	Yss,	0,	{Pe,0x17,E,0}},
	{APOPW,	Ynone,	Yfs,	0,	{Pe,0x0f,0xa1,E}},
	{APOPW,	Ynone,	Ygs,	0,	{Pe,0x0f,0xa9,E}},

/* mov seg */
	{AMOVW,	Yes,	Yml,	1,	{0x8c,0,0,0}},
	{AMOVW,	Ycs,	Yml,	1,	{0x8c,1,0,0}},
	{AMOVW,	Yss,	Yml,	1,	{0x8c,2,0,0}},
	{AMOVW,	Yds,	Yml,	1,	{0x8c,3,0,0}},
	{AMOVW,	Yfs,	Yml,	1,	{0x8c,4,0,0}},
	{AMOVW,	Ygs,	Yml,	1,	{0x8c,5,0,0}},

	{AMOVW,	Yml,	Yes,	2,	{0x8e,0,0,0}},
	{AMOVW,	Yml,	Ycs,	2,	{0x8e,1,0,0}},
	{AMOVW,	Yml,	Yss,	2,	{0x8e,2,0,0}},
	{AMOVW,	Yml,	Yds,	2,	{0x8e,3,0,0}},
	{AMOVW,	Yml,	Yfs,	2,	{0x8e,4,0,0}},
	{AMOVW,	Yml,	Ygs,	2,	{0x8e,5,0,0}},

/* mov cr */
	{AMOVL,	Ycr0,	Yml,	3,	{0x0f,0x20,0,0}},
	{AMOVL,	Ycr2,	Yml,	3,	{0x0f,0x20,2,0}},
	{AMOVL,	Ycr3,	Yml,	3,	{0x0f,0x20,3,0}},
	{AMOVL,	Ycr4,	Yml,	3,	{0x0f,0x20,4,0}},
	{AMOVL,	Ycr8,	Yml,	3,	{0x0f,0x20,8,0}},
	{AMOVQ,	Ycr0,	Yml,	3,	{0x0f,0x20,0,0}},
	{AMOVQ,	Ycr2,	Yml,	3,	{0x0f,0x20,2,0}},
	{AMOVQ,	Ycr3,	Yml,	3,	{0x0f,0x20,3,0}},
	{AMOVQ,	Ycr4,	Yml,	3,	{0x0f,0x20,4,0}},
	{AMOVQ,	Ycr8,	Yml,	3,	{0x0f,0x20,8,0}},

	{AMOVL,	Yml,	Ycr0,	4,	{0x0f,0x22,0,0}},
	{AMOVL,	Yml,	Ycr2,	4,	{0x0f,0x22,2,0}},
	{AMOVL,	Yml,	Ycr3,	4,	{0x0f,0x22,3,0}},
	{AMOVL,	Yml,	Ycr4,	4,	{0x0f,0x22,4,0}},
	{AMOVL,	Yml,	Ycr8,	4,	{0x0f,0x22,8,0}},
	{AMOVQ,	Yml,	Ycr0,	4,	{0x0f,0x22,0,0}},
	{AMOVQ,	Yml,	Ycr2,	4,	{0x0f,0x22,2,0}},
	{AMOVQ,	Yml,	Ycr3,	4,	{0x0f,0x22,3,0}},
	{AMOVQ,	Yml,	Ycr4,	4,	{0x0f,0x22,4,0}},
	{AMOVQ,	Yml,	Ycr8,	4,	{0x0f,0x22,8,0}},

/* mov dr */
	{AMOVL,	Ydr0,	Yml,	3,	{0x0f,0x21,0,0}},
	{AMOVL,	Ydr6,	Yml,	3,	{0x0f,0x21,6,0}},
	{AMOVL,	Ydr7,	Yml,	3,	{0x0f,0x21,7,0}},
	{AMOVQ,	Ydr0,	Yml,	3,	{0x0f,0x21,0,0}},
	{AMOVQ,	Ydr6,	Yml,	3,	{0x0f,0x21,6,0}},
	{AMOVQ,	Ydr7,	Yml,	3,	{0x0f,0x21,7,0}},

	{AMOVL,	Yml,	Ydr0,	4,	{0x0f,0x23,0,0}},
	{AMOVL,	Yml,	Ydr6,	4,	{0x0f,0x23,6,0}},
	{AMOVL,	Yml,	Ydr7,	4,	{0x0f,0x23,7,0}},
	{AMOVQ,	Yml,	Ydr0,	4,	{0x0f,0x23,0,0}},
	{AMOVQ,	Yml,	Ydr6,	4,	{0x0f,0x23,6,0}},
	{AMOVQ,	Yml,	Ydr7,	4,	{0x0f,0x23,7,0}},

/* mov tr */
	{AMOVL,	Ytr6,	Yml,	3,	{0x0f,0x24,6,0}},
	{AMOVL,	Ytr7,	Yml,	3,	{0x0f,0x24,7,0}},

	{AMOVL,	Yml,	Ytr6,	4,	{0x0f,0x26,6,E}},
	{AMOVL,	Yml,	Ytr7,	4,	{0x0f,0x26,7,E}},

/* lgdt, sgdt, lidt, sidt */
	{AMOVL,	Ym,	Ygdtr,	4,	{0x0f,0x01,2,0}},
	{AMOVL,	Ygdtr,	Ym,	3,	{0x0f,0x01,0,0}},
	{AMOVL,	Ym,	Yidtr,	4,	{0x0f,0x01,3,0}},
	{AMOVL,	Yidtr,	Ym,	3,	{0x0f,0x01,1,0}},
	{AMOVQ,	Ym,	Ygdtr,	4,	{0x0f,0x01,2,0}},
	{AMOVQ,	Ygdtr,	Ym,	3,	{0x0f,0x01,0,0}},
	{AMOVQ,	Ym,	Yidtr,	4,	{0x0f,0x01,3,0}},
	{AMOVQ,	Yidtr,	Ym,	3,	{0x0f,0x01,1,0}},

/* lldt, sldt */
	{AMOVW,	Yml,	Yldtr,	4,	{0x0f,0x00,2,0}},
	{AMOVW,	Yldtr,	Yml,	3,	{0x0f,0x00,0,0}},

/* lmsw, smsw */
	{AMOVW,	Yml,	Ymsw,	4,	{0x0f,0x01,6,0}},
	{AMOVW,	Ymsw,	Yml,	3,	{0x0f,0x01,4,0}},

/* ltr, str */
	{AMOVW,	Yml,	Ytask,	4,	{0x0f,0x00,3,0}},
	{AMOVW,	Ytask,	Yml,	3,	{0x0f,0x00,1,0}},

/* load full pointer */
	{AMOVL,	Yml,	Ycol,	5,	{0,0,0,0}},
	{AMOVW,	Yml,	Ycol,	5,	{Pe,0,0,0}},

/* double shift */
	{ASHLL,	Ycol,	Yml,	6,	{0xa4,0xa5,0,0}},
	{ASHRL,	Ycol,	Yml,	6,	{0xac,0xad,0,0}},
	{ASHLQ,	Ycol,	Yml,	6,	{Pw,0xa4,0xa5,0}},
	{ASHRQ,	Ycol,	Yml,	6,	{Pw,0xac,0xad,0}},
	{ASHLW,	Ycol,	Yml,	6,	{Pe,0xa4,0xa5,0}},
	{ASHRW,	Ycol,	Yml,	6,	{Pe,0xac,0xad,0}},

/* load TLS base */
	{AMOVQ,	Ytls,	Yrl,	7,	{0,0,0,0}},

	{0}
};

static int
isax(Addr *a)
{
	switch(a->reg) {
	case REG_AX:
	case REG_AL:
	case REG_AH:
		return 1;
	}
	if(a->index == REG_AX)
		return 1;
	return 0;
}

static void
subreg(Prog *p, int from, int to)
{
	if(0 /* debug['Q'] */)
		print("\n%P	s/%R/%R/\n", p, from, to);

	if(p->from.reg == from) {
		p->from.reg = to;
		p->ft = 0;
	}
	if(p->to.reg == from) {
		p->to.reg = to;
		p->tt = 0;
	}

	if(p->from.index == from) {
		p->from.index = to;
		p->ft = 0;
	}
	if(p->to.index == from) {
		p->to.index = to;
		p->tt = 0;
	}

	if(0 /* debug['Q'] */)
		print("%P\n", p);
}

static int
mediaop(Link *ctxt, Optab *o, int op, int osize, int z)
{
	switch(op){
	case Pm:
	case Pe:
	case Pf2:
	case Pf3:
		if(osize != 1){
			if(op != Pm)
				*ctxt->andptr++ = op;
			*ctxt->andptr++ = Pm;
			op = o->op[++z];
			break;
		}
	default:
		if(ctxt->andptr == ctxt->and || ctxt->and[ctxt->andptr - ctxt->and - 1] != Pm)
			*ctxt->andptr++ = Pm;
		break;
	}
	*ctxt->andptr++ = op;
	return z;
}

static void
doasm(Link *ctxt, Prog *p)
{
	Optab *o;
	Prog *q, pp;
	uchar *t;
	Movtab *mo;
	int z, op, ft, tt, xo, l, pre;
	vlong v;
	Reloc rel, *r;
	Addr *a;
	
	ctxt->curp = p;	// TODO

	o = opindex[p->as];
	if(o == nil) {
		ctxt->diag("asmins: missing op %P", p);
		return;
	}

	pre = prefixof(ctxt, &p->from);
	if(pre)
		*ctxt->andptr++ = pre;
	pre = prefixof(ctxt, &p->to);
	if(pre)
		*ctxt->andptr++ = pre;

	if(p->ft == 0)
		p->ft = oclass(ctxt, &p->from);
	if(p->tt == 0)
		p->tt = oclass(ctxt, &p->to);

	ft = p->ft * Ymax;
	tt = p->tt * Ymax;

	t = o->ytab;
	if(t == 0) {
		ctxt->diag("asmins: noproto %P", p);
		return;
	}
	xo = o->op[0] == 0x0f;
	for(z=0; *t; z+=t[3]+xo,t+=4)
		if(ycover[ft+t[0]])
		if(ycover[tt+t[1]])
			goto found;
	goto domov;

found:
	switch(o->prefix) {
	case Pq:	/* 16 bit escape and opcode escape */
		*ctxt->andptr++ = Pe;
		*ctxt->andptr++ = Pm;
		break;
	case Pq3:	/* 16 bit escape, Rex.w, and opcode escape */
		*ctxt->andptr++ = Pe;
		*ctxt->andptr++ = Pw;
		*ctxt->andptr++ = Pm;
		break;

	case Pf2:	/* xmm opcode escape */
	case Pf3:
		*ctxt->andptr++ = o->prefix;
		*ctxt->andptr++ = Pm;
		break;

	case Pm:	/* opcode escape */
		*ctxt->andptr++ = Pm;
		break;

	case Pe:	/* 16 bit escape */
		*ctxt->andptr++ = Pe;
		break;

	case Pw:	/* 64-bit escape */
		if(p->mode != 64)
			ctxt->diag("asmins: illegal 64: %P", p);
		ctxt->rexflag |= Pw;
		break;

	case Pb:	/* botch */
		bytereg(&p->from, &p->ft);
		bytereg(&p->to, &p->tt);
		break;

	case P32:	/* 32 bit but illegal if 64-bit mode */
		if(p->mode == 64)
			ctxt->diag("asmins: illegal in 64-bit mode: %P", p);
		break;

	case Py:	/* 64-bit only, no prefix */
		if(p->mode != 64)
			ctxt->diag("asmins: illegal in %d-bit mode: %P", p->mode, p);
		break;
	}

	if(z >= nelem(o->op))
		sysfatal("asmins bad table %P", p);
	op = o->op[z];
	if(op == 0x0f) {
		*ctxt->andptr++ = op;
		op = o->op[++z];
	}
	switch(t[2]) {
	default:
		ctxt->diag("asmins: unknown z %d %P", t[2], p);
		return;

	case Zpseudo:
		break;

	case Zlit:
		for(; op = o->op[z]; z++)
			*ctxt->andptr++ = op;
		break;

	case Zlitm_r:
		for(; op = o->op[z]; z++)
			*ctxt->andptr++ = op;
		asmand(ctxt, p, &p->from, &p->to);
		break;

	case Zmb_r:
		bytereg(&p->from, &p->ft);
		/* fall through */
	case Zm_r:
		*ctxt->andptr++ = op;
		asmand(ctxt, p, &p->from, &p->to);
		break;
	case Zm2_r:
		*ctxt->andptr++ = op;
		*ctxt->andptr++ = o->op[z+1];
		asmand(ctxt, p, &p->from, &p->to);
		break;

	case Zm_r_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, p, &p->from, &p->to);
		break;

	case Zm_r_xm_nr:
		ctxt->rexflag = 0;
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, p, &p->from, &p->to);
		break;

	case Zm_r_i_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, p, &p->from, &p->to);
		*ctxt->andptr++ = p->to.offset;
		break;

	case Zm_r_3d:
		*ctxt->andptr++ = 0x0f;
		*ctxt->andptr++ = 0x0f;
		asmand(ctxt, p, &p->from, &p->to);
		*ctxt->andptr++ = op;
		break;

	case Zibm_r:
		while ((op = o->op[z++]) != 0)
			*ctxt->andptr++ = op;
		asmand(ctxt, p, &p->from, &p->to);
		*ctxt->andptr++ = p->to.offset;
		break;

	case Zaut_r:
		*ctxt->andptr++ = 0x8d;	/* leal */
		if(p->from.type != TYPE_ADDR)
			ctxt->diag("asmins: Zaut sb type ADDR");
		p->from.type = TYPE_MEM;
		asmand(ctxt, p, &p->from, &p->to);
		p->from.type = TYPE_ADDR;
		break;

	case Zm_o:
		*ctxt->andptr++ = op;
		asmando(ctxt, p, &p->from, o->op[z+1]);
		break;

	case Zr_m:
		*ctxt->andptr++ = op;
		asmand(ctxt, p, &p->to, &p->from);
		break;

	case Zr_m_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, p, &p->to, &p->from);
		break;

	case Zr_m_xm_nr:
		ctxt->rexflag = 0;
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, p, &p->to, &p->from);
		break;

	case Zr_m_i_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, p, &p->to, &p->from);
		*ctxt->andptr++ = p->from.offset;
		break;

	case Zo_m:
		*ctxt->andptr++ = op;
		asmando(ctxt, p, &p->to, o->op[z+1]);
		break;

	case Zcallindreg:
		r = addrel(ctxt->cursym);
		r->off = p->pc;
		r->type = R_CALLIND;
		r->siz = 0;
		// fallthrough
	case Zo_m64:
		*ctxt->andptr++ = op;
		asmandsz(ctxt, p, &p->to, o->op[z+1], 0, 1);
		break;

	case Zm_ibo:
		*ctxt->andptr++ = op;
		asmando(ctxt, p, &p->from, o->op[z+1]);
		*ctxt->andptr++ = vaddr(ctxt, p, &p->to, nil);
		break;

	case Zibo_m:
		*ctxt->andptr++ = op;
		asmando(ctxt, p, &p->to, o->op[z+1]);
		*ctxt->andptr++ = vaddr(ctxt, p, &p->from, nil);
		break;

	case Zibo_m_xm:
		z = mediaop(ctxt, o, op, t[3], z);
		asmando(ctxt, p, &p->to, o->op[z+1]);
		*ctxt->andptr++ = vaddr(ctxt, p, &p->from, nil);
		break;

	case Z_ib:
	case Zib_:
		if(t[2] == Zib_)
			a = &p->from;
		else
			a = &p->to;
		*ctxt->andptr++ = op;
		*ctxt->andptr++ = vaddr(ctxt, p, a, nil);
		break;

	case Zib_rp:
		ctxt->rexflag |= regrex[p->to.reg] & (Rxb|0x40);
		*ctxt->andptr++ = op + reg[p->to.reg];
		*ctxt->andptr++ = vaddr(ctxt, p, &p->from, nil);
		break;

	case Zil_rp:
		ctxt->rexflag |= regrex[p->to.reg] & Rxb;
		*ctxt->andptr++ = op + reg[p->to.reg];
		if(o->prefix == Pe) {
			v = vaddr(ctxt, p, &p->from, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		else
			relput4(ctxt, p, &p->from);
		break;

	case Zo_iw:
		*ctxt->andptr++ = op;
		if(p->from.type != TYPE_NONE){
			v = vaddr(ctxt, p, &p->from, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		break;

	case Ziq_rp:
		v = vaddr(ctxt, p, &p->from, &rel);
		l = v>>32;
		if(l == 0 && rel.siz != 8){
			//p->mark |= 0100;
			//print("zero: %llux %P\n", v, p);
			ctxt->rexflag &= ~(0x40|Rxw);
			ctxt->rexflag |= regrex[p->to.reg] & Rxb;
			*ctxt->andptr++ = 0xb8 + reg[p->to.reg];
			if(rel.type != 0) {
				r = addrel(ctxt->cursym);
				*r = rel;
				r->off = p->pc + ctxt->andptr - ctxt->and;
			}
			put4(ctxt, v);
		}else if(l == -1 && (v&((uvlong)1<<31))!=0){	/* sign extend */
			//p->mark |= 0100;
			//print("sign: %llux %P\n", v, p);
			*ctxt->andptr ++ = 0xc7;
			asmando(ctxt, p, &p->to, 0);
			put4(ctxt, v);
		}else{	/* need all 8 */
			//print("all: %llux %P\n", v, p);
			ctxt->rexflag |= regrex[p->to.reg] & Rxb;
			*ctxt->andptr++ = op + reg[p->to.reg];
			if(rel.type != 0) {
				r = addrel(ctxt->cursym);
				*r = rel;
				r->off = p->pc + ctxt->andptr - ctxt->and;
			}
			put8(ctxt, v);
		}
		break;

	case Zib_rr:
		*ctxt->andptr++ = op;
		asmand(ctxt, p, &p->to, &p->to);
		*ctxt->andptr++ = vaddr(ctxt, p, &p->from, nil);
		break;

	case Z_il:
	case Zil_:
		if(t[2] == Zil_)
			a = &p->from;
		else
			a = &p->to;
		*ctxt->andptr++ = op;
		if(o->prefix == Pe) {
			v = vaddr(ctxt, p, a, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		else
			relput4(ctxt, p, a);
		break;

	case Zm_ilo:
	case Zilo_m:
		*ctxt->andptr++ = op;
		if(t[2] == Zilo_m) {
			a = &p->from;
			asmando(ctxt, p, &p->to, o->op[z+1]);
		} else {
			a = &p->to;
			asmando(ctxt, p, &p->from, o->op[z+1]);
		}
		if(o->prefix == Pe) {
			v = vaddr(ctxt, p, a, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		else
			relput4(ctxt, p, a);
		break;

	case Zil_rr:
		*ctxt->andptr++ = op;
		asmand(ctxt, p, &p->to, &p->to);
		if(o->prefix == Pe) {
			v = vaddr(ctxt, p, &p->from, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		else
			relput4(ctxt, p, &p->from);
		break;

	case Z_rp:
		ctxt->rexflag |= regrex[p->to.reg] & (Rxb|0x40);
		*ctxt->andptr++ = op + reg[p->to.reg];
		break;

	case Zrp_:
		ctxt->rexflag |= regrex[p->from.reg] & (Rxb|0x40);
		*ctxt->andptr++ = op + reg[p->from.reg];
		break;

	case Zclr:
		ctxt->rexflag &= ~Pw;
		*ctxt->andptr++ = op;
		asmand(ctxt, p, &p->to, &p->to);
		break;

	case Zcall:
		if(p->to.sym == nil) {
			ctxt->diag("call without target");
			sysfatal("bad code");
		}
		*ctxt->andptr++ = op;
		r = addrel(ctxt->cursym);
		r->off = p->pc + ctxt->andptr - ctxt->and;
		r->sym = p->to.sym;
		r->add = p->to.offset;
		r->type = R_CALL;
		r->siz = 4;
		put4(ctxt, 0);
		break;

	case Zbr:
	case Zjmp:
	case Zloop:
		// TODO: jump across functions needs reloc
		if(p->to.sym != nil) {
			if(t[2] != Zjmp) {
				ctxt->diag("branch to ATEXT");
				sysfatal("bad code");
			}
			*ctxt->andptr++ = o->op[z+1];
			r = addrel(ctxt->cursym);
			r->off = p->pc + ctxt->andptr - ctxt->and;
			r->sym = p->to.sym;
			r->type = R_PCREL;
			r->siz = 4;
			put4(ctxt, 0);
			break;
		}
		// Assumes q is in this function.
		// TODO: Check in input, preserve in brchain.

		// Fill in backward jump now.
		q = p->pcond;
		if(q == nil) {
			ctxt->diag("jmp/branch/loop without target");
			sysfatal("bad code");
		}
		if(p->back & 1) {
			v = q->pc - (p->pc + 2);
			if(v >= -128) {
				if(p->as == AJCXZL)
					*ctxt->andptr++ = 0x67;
				*ctxt->andptr++ = op;
				*ctxt->andptr++ = v;
			} else if(t[2] == Zloop) {
				ctxt->diag("loop too far: %P", p);
			} else {
				v -= 5-2;
				if(t[2] == Zbr) {
					*ctxt->andptr++ = 0x0f;
					v--;
				}
				*ctxt->andptr++ = o->op[z+1];
				*ctxt->andptr++ = v;
				*ctxt->andptr++ = v>>8;
				*ctxt->andptr++ = v>>16;
				*ctxt->andptr++ = v>>24;
			}
			break;
		}
		
		// Annotate target; will fill in later.
		p->forwd = q->comefrom;
		q->comefrom = p;
		if(p->back & 2)	{ // short
			if(p->as == AJCXZL)
				*ctxt->andptr++ = 0x67;
			*ctxt->andptr++ = op;
			*ctxt->andptr++ = 0;
		} else if(t[2] == Zloop) {
			ctxt->diag("loop too far: %P", p);
		} else {
			if(t[2] == Zbr)
				*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = o->op[z+1];
			*ctxt->andptr++ = 0;
			*ctxt->andptr++ = 0;
			*ctxt->andptr++ = 0;
			*ctxt->andptr++ = 0;
		}
		break;
				
/*
		v = q->pc - p->pc - 2;
		if((v >= -128 && v <= 127) || p->pc == -1 || q->pc == -1) {
			*ctxt->andptr++ = op;
			*ctxt->andptr++ = v;
		} else {
			v -= 5-2;
			if(t[2] == Zbr) {
				*ctxt->andptr++ = 0x0f;
				v--;
			}
			*ctxt->andptr++ = o->op[z+1];
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
			*ctxt->andptr++ = v>>16;
			*ctxt->andptr++ = v>>24;
		}
*/
		break;

	case Zbyte:
		v = vaddr(ctxt, p, &p->from, &rel);
		if(rel.siz != 0) {
			rel.siz = op;
			r = addrel(ctxt->cursym);
			*r = rel;
			r->off = p->pc + ctxt->andptr - ctxt->and;
		}
		*ctxt->andptr++ = v;
		if(op > 1) {
			*ctxt->andptr++ = v>>8;
			if(op > 2) {
				*ctxt->andptr++ = v>>16;
				*ctxt->andptr++ = v>>24;
				if(op > 4) {
					*ctxt->andptr++ = v>>32;
					*ctxt->andptr++ = v>>40;
					*ctxt->andptr++ = v>>48;
					*ctxt->andptr++ = v>>56;
				}
			}
		}
		break;
	}
	return;

domov:
	for(mo=ymovtab; mo->as; mo++)
		if(p->as == mo->as)
		if(ycover[ft+mo->ft])
		if(ycover[tt+mo->tt]){
			t = mo->op;
			goto mfound;
		}
bad:
	if(p->mode != 64){
		/*
		 * here, the assembly has failed.
		 * if its a byte instruction that has
		 * unaddressable registers, try to
		 * exchange registers and reissue the
		 * instruction with the operands renamed.
		 */
		pp = *p;
		z = p->from.reg;
		if(p->from.type == TYPE_REG && z >= REG_BP && z <= REG_DI) {
			if(isax(&p->to) || p->to.type == TYPE_NONE) {
				// We certainly don't want to exchange
				// with AX if the op is MUL or DIV.
				*ctxt->andptr++ = 0x87;			/* xchg lhs,bx */
				asmando(ctxt, p, &p->from, reg[REG_BX]);
				subreg(&pp, z, REG_BX);
				doasm(ctxt, &pp);
				*ctxt->andptr++ = 0x87;			/* xchg lhs,bx */
				asmando(ctxt, p, &p->from, reg[REG_BX]);
			} else {
				*ctxt->andptr++ = 0x90 + reg[z];		/* xchg lsh,ax */
				subreg(&pp, z, REG_AX);
				doasm(ctxt, &pp);
				*ctxt->andptr++ = 0x90 + reg[z];		/* xchg lsh,ax */
			}
			return;
		}
		z = p->to.reg;
		if(p->to.type == TYPE_REG && z >= REG_BP && z <= REG_DI) {
			if(isax(&p->from)) {
				*ctxt->andptr++ = 0x87;			/* xchg rhs,bx */
				asmando(ctxt, p, &p->to, reg[REG_BX]);
				subreg(&pp, z, REG_BX);
				doasm(ctxt, &pp);
				*ctxt->andptr++ = 0x87;			/* xchg rhs,bx */
				asmando(ctxt, p, &p->to, reg[REG_BX]);
			} else {
				*ctxt->andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
				subreg(&pp, z, REG_AX);
				doasm(ctxt, &pp);
				*ctxt->andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
			}
			return;
		}
	}
	ctxt->diag("doasm: notfound ft=%d tt=%d %P %d %d", p->ft, p->tt, p, oclass(ctxt, &p->from), oclass(ctxt, &p->to));
	return;

mfound:
	switch(mo->code) {
	default:
		ctxt->diag("asmins: unknown mov %d %P", mo->code, p);
		break;

	case 0:	/* lit */
		for(z=0; t[z]!=E; z++)
			*ctxt->andptr++ = t[z];
		break;

	case 1:	/* r,m */
		*ctxt->andptr++ = t[0];
		asmando(ctxt, p, &p->to, t[1]);
		break;

	case 2:	/* m,r */
		*ctxt->andptr++ = t[0];
		asmando(ctxt, p, &p->from, t[1]);
		break;

	case 3:	/* r,m - 2op */
		*ctxt->andptr++ = t[0];
		*ctxt->andptr++ = t[1];
		asmando(ctxt, p, &p->to, t[2]);
		ctxt->rexflag |= regrex[p->from.reg] & (Rxr|0x40);
		break;

	case 4:	/* m,r - 2op */
		*ctxt->andptr++ = t[0];
		*ctxt->andptr++ = t[1];
		asmando(ctxt, p, &p->from, t[2]);
		ctxt->rexflag |= regrex[p->to.reg] & (Rxr|0x40);
		break;

	case 5:	/* load full pointer, trash heap */
		if(t[0])
			*ctxt->andptr++ = t[0];
		switch(p->to.index) {
		default:
			goto bad;
		case REG_DS:
			*ctxt->andptr++ = 0xc5;
			break;
		case REG_SS:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = 0xb2;
			break;
		case REG_ES:
			*ctxt->andptr++ = 0xc4;
			break;
		case REG_FS:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = 0xb4;
			break;
		case REG_GS:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = 0xb5;
			break;
		}
		asmand(ctxt, p, &p->from, &p->to);
		break;

	case 6:	/* double shift */
		if(t[0] == Pw){
			if(p->mode != 64)
				ctxt->diag("asmins: illegal 64: %P", p);
			ctxt->rexflag |= Pw;
			t++;
		}else if(t[0] == Pe){
			*ctxt->andptr++ = Pe;
			t++;
		}
		switch(p->from.type) {
		default:
			goto bad;
		case TYPE_CONST:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = t[0];
			asmandsz(ctxt, p, &p->to, reg[(int)p->from.index], regrex[(int)p->from.index], 0);
			*ctxt->andptr++ = p->from.offset;
			break;
		case TYPE_REG:
			switch(p->from.reg) {
			default:
				goto bad;
			case REG_CL:
			case REG_CX:
				*ctxt->andptr++ = 0x0f;
				*ctxt->andptr++ = t[1];
				asmandsz(ctxt, p, &p->to, reg[(int)p->from.index], regrex[(int)p->from.index], 0);
				break;
			}
		}
		break;
	
	case 7:	/* mov tls, r */
		// NOTE: The systems listed here are the ones that use the "TLS initial exec" model,
		// where you load the TLS base register into a register and then index off that
		// register to access the actual TLS variables. Systems that allow direct TLS access
		// are handled in prefixof above and should not be listed here.
		switch(ctxt->headtype) {
		default:
			sysfatal("unknown TLS base location for %s", headstr(ctxt->headtype));

		case Hplan9:
			if(ctxt->plan9privates == nil)
				ctxt->plan9privates = linklookup(ctxt, "_privates", 0);
			memset(&pp.from, 0, sizeof pp.from);
			pp.from.type = TYPE_MEM;
			pp.from.name = NAME_EXTERN;
			pp.from.sym = ctxt->plan9privates;
			pp.from.offset = 0;
			pp.from.index = REG_NONE;
			ctxt->rexflag |= Pw;
			*ctxt->andptr++ = 0x8B;
			asmand(ctxt, p, &pp.from, &p->to);
			break;

		case Hsolaris: // TODO(rsc): Delete Hsolaris from list. Should not use this code. See progedit in obj6.c.
			// TLS base is 0(FS).
			pp.from = p->from;
			pp.from.type = TYPE_MEM;
			pp.from.name = NAME_NONE;
			pp.from.reg = REG_NONE;
			pp.from.offset = 0;
			pp.from.index = REG_NONE;
			pp.from.scale = 0;
			ctxt->rexflag |= Pw;
			*ctxt->andptr++ = 0x64; // FS
			*ctxt->andptr++ = 0x8B;
			asmand(ctxt, p, &pp.from, &p->to);
			break;
		
		case Hwindows:
			// Windows TLS base is always 0x28(GS).
			pp.from = p->from;
			pp.from.type = TYPE_MEM;
			pp.from.name = NAME_NONE;
			pp.from.reg = REG_GS;
			pp.from.offset = 0x28;
			pp.from.index = REG_NONE;
			pp.from.scale = 0;
			ctxt->rexflag |= Pw;
			*ctxt->andptr++ = 0x65; // GS
			*ctxt->andptr++ = 0x8B;
			asmand(ctxt, p, &pp.from, &p->to);
			break;
		}
		break;
	}
}

static uchar naclret[] = {
	0x5e, // POPL SI
	// 0x8b, 0x7d, 0x00, // MOVL (BP), DI - catch return to invalid address, for debugging
	0x83, 0xe6, 0xe0,	// ANDL $~31, SI
	0x4c, 0x01, 0xfe,	// ADDQ R15, SI
	0xff, 0xe6, // JMP SI
};

static uchar naclspfix[] = {
	0x4c, 0x01, 0xfc, // ADDQ R15, SP
};

static uchar naclbpfix[] = {
	0x4c, 0x01, 0xfd, // ADDQ R15, BP
};

static uchar naclmovs[] = {
	0x89, 0xf6,	// MOVL SI, SI
	0x49, 0x8d, 0x34, 0x37,	// LEAQ (R15)(SI*1), SI
	0x89, 0xff,	// MOVL DI, DI
	0x49, 0x8d, 0x3c, 0x3f,	// LEAQ (R15)(DI*1), DI
};

static uchar naclstos[] = {
	0x89, 0xff,	// MOVL DI, DI
	0x49, 0x8d, 0x3c, 0x3f,	// LEAQ (R15)(DI*1), DI
};

static void
nacltrunc(Link *ctxt, int reg)
{	
	if(reg >= REG_R8)
		*ctxt->andptr++ = 0x45;
	reg = (reg - REG_AX) & 7;
	*ctxt->andptr++ = 0x89;
	*ctxt->andptr++ = (3<<6) | (reg<<3) | reg;
}

static void
asmins(Link *ctxt, Prog *p)
{
	int i, n, np, c;
	uchar *and0;
	Reloc *r;
	
	ctxt->andptr = ctxt->and;
	ctxt->asmode = p->mode;
	
	if(p->as == AUSEFIELD) {
		r = addrel(ctxt->cursym);
		r->off = 0;
		r->siz = 0;
		r->sym = p->from.sym;
		r->type = R_USEFIELD;
		return;
	}
	
	if(ctxt->headtype == Hnacl) {
		if(p->as == AREP) {
			ctxt->rep++;
			return;
		}
		if(p->as == AREPN) {
			ctxt->repn++;
			return;
		}
		if(p->as == ALOCK) {
			ctxt->lock++;
			return;
		}
		if(p->as != ALEAQ && p->as != ALEAL) {
			if(p->from.index != TYPE_NONE && p->from.scale > 0)
				nacltrunc(ctxt, p->from.index);
			if(p->to.index != TYPE_NONE && p->to.scale > 0)
				nacltrunc(ctxt, p->to.index);
		}
		switch(p->as) {
		case ARET:
			memmove(ctxt->andptr, naclret, sizeof naclret);
			ctxt->andptr += sizeof naclret;
			return;
		case ACALL:
		case AJMP:
			if(p->to.type == TYPE_REG && REG_AX <= p->to.reg && p->to.reg <= REG_DI) {
				// ANDL $~31, reg
				*ctxt->andptr++ = 0x83;
				*ctxt->andptr++ = 0xe0 | (p->to.reg - REG_AX);
				*ctxt->andptr++ = 0xe0;
				// ADDQ R15, reg
				*ctxt->andptr++ = 0x4c;
				*ctxt->andptr++ = 0x01;
				*ctxt->andptr++ = 0xf8 | (p->to.reg - REG_AX);
			}
			if(p->to.type == TYPE_REG && REG_R8 <= p->to.reg && p->to.reg <= REG_R15) {
				// ANDL $~31, reg
				*ctxt->andptr++ = 0x41;
				*ctxt->andptr++ = 0x83;
				*ctxt->andptr++ = 0xe0 | (p->to.reg - REG_R8);
				*ctxt->andptr++ = 0xe0;
				// ADDQ R15, reg
				*ctxt->andptr++ = 0x4d;
				*ctxt->andptr++ = 0x01;
				*ctxt->andptr++ = 0xf8 | (p->to.reg - REG_R8);
			}
			break;
		case AINT:
			*ctxt->andptr++ = 0xf4;
			return;
		case ASCASB:
		case ASCASW:
		case ASCASL:
		case ASCASQ:
		case ASTOSB:
		case ASTOSW:
		case ASTOSL:
		case ASTOSQ:
			memmove(ctxt->andptr, naclstos, sizeof naclstos);
			ctxt->andptr += sizeof naclstos;
			break;
		case AMOVSB:
		case AMOVSW:
		case AMOVSL:
		case AMOVSQ:
			memmove(ctxt->andptr, naclmovs, sizeof naclmovs);
			ctxt->andptr += sizeof naclmovs;
			break;
		}
		if(ctxt->rep) {
			*ctxt->andptr++ = 0xf3;
			ctxt->rep = 0;
		}
		if(ctxt->repn) {
			*ctxt->andptr++ = 0xf2;
			ctxt->repn = 0;
		}
		if(ctxt->lock) {
			*ctxt->andptr++ = 0xf0;
			ctxt->lock = 0;
		}
	}		

	ctxt->rexflag = 0;
	and0 = ctxt->andptr;
	ctxt->asmode = p->mode;
	doasm(ctxt, p);
	if(ctxt->rexflag){
		/*
		 * as befits the whole approach of the architecture,
		 * the rex prefix must appear before the first opcode byte
		 * (and thus after any 66/67/f2/f3/26/2e/3e prefix bytes, but
		 * before the 0f opcode escape!), or it might be ignored.
		 * note that the handbook often misleadingly shows 66/f2/f3 in `opcode'.
		 */
		if(p->mode != 64)
			ctxt->diag("asmins: illegal in mode %d: %P", p->mode, p);
		n = ctxt->andptr - and0;
		for(np = 0; np < n; np++) {
			c = and0[np];
			if(c != 0xf2 && c != 0xf3 && (c < 0x64 || c > 0x67) && c != 0x2e && c != 0x3e && c != 0x26)
				break;
		}
		memmove(and0+np+1, and0+np, n-np);
		and0[np] = 0x40 | ctxt->rexflag;
		ctxt->andptr++;
	}
	n = ctxt->andptr - ctxt->and;
	for(i=ctxt->cursym->nr-1; i>=0; i--) {
		r = ctxt->cursym->r+i;
		if(r->off < p->pc)
			break;
		if(ctxt->rexflag)
			r->off++;
		if(r->type == R_PCREL || r->type == R_CALL)
			r->add -= p->pc + n - (r->off + r->siz);
	}

	if(ctxt->headtype == Hnacl && p->as != ACMPL && p->as != ACMPQ && p->to.type == TYPE_REG) {
		switch(p->to.reg) {
		case REG_SP:
			memmove(ctxt->andptr, naclspfix, sizeof naclspfix);
			ctxt->andptr += sizeof naclspfix;
			break;
		case REG_BP:
			memmove(ctxt->andptr, naclbpfix, sizeof naclbpfix);
			ctxt->andptr += sizeof naclbpfix;
			break;
		}
	}
}
