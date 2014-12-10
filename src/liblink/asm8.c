// Inferno utils/8l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/span.c
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
#include "../cmd/8l/8.out.h"
#include "../runtime/stack.h"

enum
{
	MaxAlign = 32,	// max data alignment
	FuncAlign = 16
};

typedef	struct	Optab	Optab;

struct	Optab
{
	short	as;
	uchar*	ytab;
	uchar	prefix;
	uchar	op[13];
};

enum
{
	Yxxx		= 0,
	Ynone,
	Yi0,
	Yi1,
	Yi8,
	Yi32,
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
	Ytls,

	Ycs,	Yss,	Yds,	Yes,	Yfs,	Ygs,
	Ygdtr,	Yidtr,	Yldtr,	Ymsw,	Ytask,
	Ycr0,	Ycr1,	Ycr2,	Ycr3,	Ycr4,	Ycr5,	Ycr6,	Ycr7,
	Ydr0,	Ydr1,	Ydr2,	Ydr3,	Ydr4,	Ydr5,	Ydr6,	Ydr7,
	Ytr0,	Ytr1,	Ytr2,	Ytr3,	Ytr4,	Ytr5,	Ytr6,	Ytr7,
	Ymr, Ymm,
	Yxr, Yxm,
	Ymax,

	Zxxx		= 0,

	Zlit,
	Zlitm_r,
	Z_rp,
	Zbr,
	Zcall,
	Zcallcon,
	Zcallind,
	Zcallindreg,
	Zib_,
	Zib_rp,
	Zibo_m,
	Zil_,
	Zil_rp,
	Zilo_m,
	Zjmp,
	Zjmpcon,
	Zloop,
	Zm_o,
	Zm_r,
	Zm2_r,
	Zm_r_xm,
	Zm_r_i_xm,
	Zaut_r,
	Zo_m,
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
	Zibm_r,	/* mmx1,mmx2/mem64,imm8 */
	Zbyte,
	Zmov,
	Zmax,

	Px		= 0,
	Pe		= 0x66,	/* operand escape */
	Pm		= 0x0f,	/* 2byte opcode escape */
	Pq		= 0xff,	/* both escape */
	Pb		= 0xfe,	/* byte operands */
	Pf2		= 0xf2,	/* xmm escape 1 */
	Pf3		= 0xf3,	/* xmm escape 2 */
};

static	uchar	ycover[Ymax*Ymax];
static	int	reg[D_NONE];
static	void	asmins(Link *ctxt, Prog *p);

static uchar	ynone[] =
{
	Ynone,	Ynone,	Zlit,	1,
	0
};
static uchar	ytext[] =
{
	Ymb,	Yi32,	Zpseudo,1,
	0
};
static uchar	ynop[] =
{
	Ynone,	Ynone,	Zpseudo,0,
	Ynone,	Yiauto,	Zpseudo,0,
	Ynone,	Yml,	Zpseudo,0,
	Ynone,	Yrf,	Zpseudo,0,
	Yiauto,	Ynone,	Zpseudo,0,
	Ynone,	Yxr,	Zpseudo,0,
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
	0,
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
static uchar	yincl[] =
{
	Ynone,	Yrl,	Z_rp,	1,
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
static uchar	ymovw[] =
{
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	Yi0,	Yrl,	Zclr,	1+2,
//	Yi0,	Yml,	Zibo_m,	2,	// shorter but slower AND $0,dst
	Yi32,	Yrl,	Zil_rp,	1,
	Yi32,	Yml,	Zilo_m,	2,
	Yiauto,	Yrl,	Zaut_r,	1,
	0
};
static uchar	ymovl[] =
{
	Yrl,	Yml,	Zr_m,	1,
	Yml,	Yrl,	Zm_r,	1,
	Yi0,	Yrl,	Zclr,	1+2,
//	Yi0,	Yml,	Zibo_m,	2,	// shorter but slower AND $0,dst
	Yi32,	Yrl,	Zil_rp,	1,
	Yi32,	Yml,	Zilo_m,	2,
	Yml,	Yxr,	Zm_r_xm,	2,	// XMM MOVD (32 bit)
	Yxr,	Yml,	Zr_m_xm,	2,	// XMM MOVD (32 bit)
	Yiauto,	Yrl,	Zaut_r,	1,
	0
};
static uchar	ymovq[] =
{
	Yml,	Yxr,	Zm_r_xm,	2,
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
	Ymb,	Yrl,	Zm_r,	1,
	0
};
static uchar	yml_rl[] =
{
	Yml,	Yrl,	Zm_r,	1,
	0
};
static uchar	yrb_mb[] =
{
	Yrb,	Ymb,	Zr_m,	1,
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
	0
};
static uchar	ybyte[] =
{
	Yi32,	Ynone,	Zbyte,	1,
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
	Ynone,	Yrl,	Z_rp,	1,
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
	Ynone,	Ycol,	Zcallind,	2,
	Ynone,	Ybr,	Zcall,	0,
	Ynone,	Yi32,	Zcallcon,	1,
	0
};
static uchar	yduff[] =
{
	Ynone,	Yi32,	Zcall,	1,
	0
};
static uchar	yjmp[] =
{
	Ynone,	Yml,	Zo_m,	2,
	Ynone,	Ybr,	Zjmp,	0,
	Ynone,	Yi32,	Zjmpcon,	1,
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
static uchar	yfcmv[] =
{
	Yrf,	Yf0,	Zm_o,	2,
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
static uchar	ymskb[] =
{
	Yxr,	Yrl,	Zm_r_xm,	2,
	Ymr,	Yrl,	Zm_r_xm,	1,
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
static uchar	yxmq[] = 
{
	Yxm,	Yxr,	Zm_r_xm,	2,
	0
};
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
/*
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
*/
static uchar	yxrrl[] =
{
	Yxr,	Yrl,	Zm_r,	1,
	0
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
static uchar	yinsrd[] =
{
	Yml,	Yxr,	Zibm_r,	2,
	0
};
static uchar	ymshufb[] =
{
	Yxm,	Yxr,	Zm2_r,	2,
	0
};

static uchar   yxshuf[] =
{
	Yxm,    Yxr,    Zibm_r, 2,
	0
};

static Optab optab[] =
/*	as, ytab, andproto, opcode */
{
	{ AXXX },
	{ AAAA,		ynone,	Px, {0x37} },
	{ AAAD,		ynone,	Px, {0xd5,0x0a} },
	{ AAAM,		ynone,	Px, {0xd4,0x0a} },
	{ AAAS,		ynone,	Px, {0x3f} },
	{ AADCB,	yxorb,	Pb, {0x14,0x80,(02),0x10,0x10} },
	{ AADCL,	yxorl,	Px, {0x83,(02),0x15,0x81,(02),0x11,0x13} },
	{ AADCW,	yxorl,	Pe, {0x83,(02),0x15,0x81,(02),0x11,0x13} },
	{ AADDB,	yxorb,	Px, {0x04,0x80,(00),0x00,0x02} },
	{ AADDL,	yaddl,	Px, {0x83,(00),0x05,0x81,(00),0x01,0x03} },
	{ AADDW,	yaddl,	Pe, {0x83,(00),0x05,0x81,(00),0x01,0x03} },
	{ AADJSP },
	{ AANDB,	yxorb,	Pb, {0x24,0x80,(04),0x20,0x22} },
	{ AANDL,	yxorl,	Px, {0x83,(04),0x25,0x81,(04),0x21,0x23} },
	{ AANDW,	yxorl,	Pe, {0x83,(04),0x25,0x81,(04),0x21,0x23} },
	{ AARPL,	yrl_ml,	Px, {0x63} },
	{ ABOUNDL,	yrl_m,	Px, {0x62} },
	{ ABOUNDW,	yrl_m,	Pe, {0x62} },
	{ ABSFL,	yml_rl,	Pm, {0xbc} },
	{ ABSFW,	yml_rl,	Pq, {0xbc} },
	{ ABSRL,	yml_rl,	Pm, {0xbd} },
	{ ABSRW,	yml_rl,	Pq, {0xbd} },
	{ ABTL,		yml_rl,	Pm, {0xa3} },
	{ ABTW,		yml_rl,	Pq, {0xa3} },
	{ ABTCL,	yml_rl,	Pm, {0xbb} },
	{ ABTCW,	yml_rl,	Pq, {0xbb} },
	{ ABTRL,	yml_rl,	Pm, {0xb3} },
	{ ABTRW,	yml_rl,	Pq, {0xb3} },
	{ ABTSL,	yml_rl,	Pm, {0xab} },
	{ ABTSW,	yml_rl,	Pq, {0xab} },
	{ ABYTE,	ybyte,	Px, {1} },
	{ ACALL,	ycall,	Px, {0xff,(02),0xff,(0x15),0xe8} },
	{ ACLC,		ynone,	Px, {0xf8} },
	{ ACLD,		ynone,	Px, {0xfc} },
	{ ACLI,		ynone,	Px, {0xfa} },
	{ ACLTS,	ynone,	Pm, {0x06} },
	{ ACMC,		ynone,	Px, {0xf5} },
	{ ACMPB,	ycmpb,	Pb, {0x3c,0x80,(07),0x38,0x3a} },
	{ ACMPL,	ycmpl,	Px, {0x83,(07),0x3d,0x81,(07),0x39,0x3b} },
	{ ACMPW,	ycmpl,	Pe, {0x83,(07),0x3d,0x81,(07),0x39,0x3b} },
	{ ACMPSB,	ynone,	Pb, {0xa6} },
	{ ACMPSL,	ynone,	Px, {0xa7} },
	{ ACMPSW,	ynone,	Pe, {0xa7} },
	{ ADAA,		ynone,	Px, {0x27} },
	{ ADAS,		ynone,	Px, {0x2f} },
	{ ADATA },
	{ ADECB,	yincb,	Pb, {0xfe,(01)} },
	{ ADECL,	yincl,	Px, {0x48,0xff,(01)} },
	{ ADECW,	yincl,	Pe, {0x48,0xff,(01)} },
	{ ADIVB,	ydivb,	Pb, {0xf6,(06)} },
	{ ADIVL,	ydivl,	Px, {0xf7,(06)} },
	{ ADIVW,	ydivl,	Pe, {0xf7,(06)} },
	{ AENTER },				/* botch */
	{ AGLOBL },
	{ AGOK },
	{ AHISTORY },
	{ AHLT,		ynone,	Px, {0xf4} },
	{ AIDIVB,	ydivb,	Pb, {0xf6,(07)} },
	{ AIDIVL,	ydivl,	Px, {0xf7,(07)} },
	{ AIDIVW,	ydivl,	Pe, {0xf7,(07)} },
	{ AIMULB,	ydivb,	Pb, {0xf6,(05)} },
	{ AIMULL,	yimul,	Px, {0xf7,(05),0x6b,0x69} },
	{ AIMULW,	yimul,	Pe, {0xf7,(05),0x6b,0x69} },
	{ AINB,		yin,	Pb, {0xe4,0xec} },
	{ AINL,		yin,	Px, {0xe5,0xed} },
	{ AINW,		yin,	Pe, {0xe5,0xed} },
	{ AINCB,	yincb,	Pb, {0xfe,(00)} },
	{ AINCL,	yincl,	Px, {0x40,0xff,(00)} },
	{ AINCW,	yincl,	Pe, {0x40,0xff,(00)} },
	{ AINSB,	ynone,	Pb, {0x6c} },
	{ AINSL,	ynone,	Px, {0x6d} },
	{ AINSW,	ynone,	Pe, {0x6d} },
	{ AINT,		yint,	Px, {0xcd} },
	{ AINTO,	ynone,	Px, {0xce} },
	{ AIRETL,	ynone,	Px, {0xcf} },
	{ AIRETW,	ynone,	Pe, {0xcf} },
	{ AJCC,		yjcond,	Px, {0x73,0x83,(00)} },
	{ AJCS,		yjcond,	Px, {0x72,0x82} },
	{ AJCXZL,	yloop,	Px, {0xe3} },
	{ AJCXZW,	yloop,	Px, {0xe3} },
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
	{ ALEAL,	ym_rl,	Px, {0x8d} },
	{ ALEAW,	ym_rl,	Pe, {0x8d} },
	{ ALEAVEL,	ynone,	Px, {0xc9} },
	{ ALEAVEW,	ynone,	Pe, {0xc9} },
	{ ALOCK,	ynone,	Px, {0xf0} },
	{ ALODSB,	ynone,	Pb, {0xac} },
	{ ALODSL,	ynone,	Px, {0xad} },
	{ ALODSW,	ynone,	Pe, {0xad} },
	{ ALONG,	ybyte,	Px, {4} },
	{ ALOOP,	yloop,	Px, {0xe2} },
	{ ALOOPEQ,	yloop,	Px, {0xe1} },
	{ ALOOPNE,	yloop,	Px, {0xe0} },
	{ ALSLL,	yml_rl,	Pm, {0x03 } },
	{ ALSLW,	yml_rl,	Pq, {0x03 } },
	{ AMOVB,	ymovb,	Pb, {0x88,0x8a,0xb0,0xc6,(00)} },
	{ AMOVL,	ymovl,	Px, {0x89,0x8b,0x31,0x83,(04),0xb8,0xc7,(00),Pe,0x6e,Pe,0x7e,0} },
	{ AMOVW,	ymovw,	Pe, {0x89,0x8b,0x31,0x83,(04),0xb8,0xc7,(00),0} },
	{ AMOVQ,	ymovq,	Pf3, {0x7e} },
	{ AMOVBLSX,	ymb_rl,	Pm, {0xbe} },
	{ AMOVBLZX,	ymb_rl,	Pm, {0xb6} },
	{ AMOVBWSX,	ymb_rl,	Pq, {0xbe} },
	{ AMOVBWZX,	ymb_rl,	Pq, {0xb6} },
	{ AMOVWLSX,	yml_rl,	Pm, {0xbf} },
	{ AMOVWLZX,	yml_rl,	Pm, {0xb7} },
	{ AMOVSB,	ynone,	Pb, {0xa4} },
	{ AMOVSL,	ynone,	Px, {0xa5} },
	{ AMOVSW,	ynone,	Pe, {0xa5} },
	{ AMULB,	ydivb,	Pb, {0xf6,(04)} },
	{ AMULL,	ydivl,	Px, {0xf7,(04)} },
	{ AMULW,	ydivl,	Pe, {0xf7,(04)} },
	{ ANAME },
	{ ANEGB,	yscond,	Px, {0xf6,(03)} },
	{ ANEGL,	yscond,	Px, {0xf7,(03)} },
	{ ANEGW,	yscond,	Pe, {0xf7,(03)} },
	{ ANOP,		ynop,	Px, {0,0} },
	{ ANOTB,	yscond,	Px, {0xf6,(02)} },
	{ ANOTL,	yscond,	Px, {0xf7,(02)} },
	{ ANOTW,	yscond,	Pe, {0xf7,(02)} },
	{ AORB,		yxorb,	Pb, {0x0c,0x80,(01),0x08,0x0a} },
	{ AORL,		yxorl,	Px, {0x83,(01),0x0d,0x81,(01),0x09,0x0b} },
	{ AORW,		yxorl,	Pe, {0x83,(01),0x0d,0x81,(01),0x09,0x0b} },
	{ AOUTB,	yin,	Pb, {0xe6,0xee} },
	{ AOUTL,	yin,	Px, {0xe7,0xef} },
	{ AOUTW,	yin,	Pe, {0xe7,0xef} },
	{ AOUTSB,	ynone,	Pb, {0x6e} },
	{ AOUTSL,	ynone,	Px, {0x6f} },
	{ AOUTSW,	ynone,	Pe, {0x6f} },
	{ APAUSE,	ynone,	Px, {0xf3,0x90} },
	{ APOPAL,	ynone,	Px, {0x61} },
	{ APOPAW,	ynone,	Pe, {0x61} },
	{ APOPFL,	ynone,	Px, {0x9d} },
	{ APOPFW,	ynone,	Pe, {0x9d} },
	{ APOPL,	ypopl,	Px, {0x58,0x8f,(00)} },
	{ APOPW,	ypopl,	Pe, {0x58,0x8f,(00)} },
	{ APUSHAL,	ynone,	Px, {0x60} },
	{ APUSHAW,	ynone,	Pe, {0x60} },
	{ APUSHFL,	ynone,	Px, {0x9c} },
	{ APUSHFW,	ynone,	Pe, {0x9c} },
	{ APUSHL,	ypushl,	Px, {0x50,0xff,(06),0x6a,0x68} },
	{ APUSHW,	ypushl,	Pe, {0x50,0xff,(06),0x6a,0x68} },
	{ ARCLB,	yshb,	Pb, {0xd0,(02),0xc0,(02),0xd2,(02)} },
	{ ARCLL,	yshl,	Px, {0xd1,(02),0xc1,(02),0xd3,(02),0xd3,(02)} },
	{ ARCLW,	yshl,	Pe, {0xd1,(02),0xc1,(02),0xd3,(02),0xd3,(02)} },
	{ ARCRB,	yshb,	Pb, {0xd0,(03),0xc0,(03),0xd2,(03)} },
	{ ARCRL,	yshl,	Px, {0xd1,(03),0xc1,(03),0xd3,(03),0xd3,(03)} },
	{ ARCRW,	yshl,	Pe, {0xd1,(03),0xc1,(03),0xd3,(03),0xd3,(03)} },
	{ AREP,		ynone,	Px, {0xf3} },
	{ AREPN,	ynone,	Px, {0xf2} },
	{ ARET,		ynone,	Px, {0xc3} },
	{ AROLB,	yshb,	Pb, {0xd0,(00),0xc0,(00),0xd2,(00)} },
	{ AROLL,	yshl,	Px, {0xd1,(00),0xc1,(00),0xd3,(00),0xd3,(00)} },
	{ AROLW,	yshl,	Pe, {0xd1,(00),0xc1,(00),0xd3,(00),0xd3,(00)} },
	{ ARORB,	yshb,	Pb, {0xd0,(01),0xc0,(01),0xd2,(01)} },
	{ ARORL,	yshl,	Px, {0xd1,(01),0xc1,(01),0xd3,(01),0xd3,(01)} },
	{ ARORW,	yshl,	Pe, {0xd1,(01),0xc1,(01),0xd3,(01),0xd3,(01)} },
	{ ASAHF,	ynone,	Px, {0x9e} },
	{ ASALB,	yshb,	Pb, {0xd0,(04),0xc0,(04),0xd2,(04)} },
	{ ASALL,	yshl,	Px, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASALW,	yshl,	Pe, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASARB,	yshb,	Pb, {0xd0,(07),0xc0,(07),0xd2,(07)} },
	{ ASARL,	yshl,	Px, {0xd1,(07),0xc1,(07),0xd3,(07),0xd3,(07)} },
	{ ASARW,	yshl,	Pe, {0xd1,(07),0xc1,(07),0xd3,(07),0xd3,(07)} },
	{ ASBBB,	yxorb,	Pb, {0x1c,0x80,(03),0x18,0x1a} },
	{ ASBBL,	yxorl,	Px, {0x83,(03),0x1d,0x81,(03),0x19,0x1b} },
	{ ASBBW,	yxorl,	Pe, {0x83,(03),0x1d,0x81,(03),0x19,0x1b} },
	{ ASCASB,	ynone,	Pb, {0xae} },
	{ ASCASL,	ynone,	Px, {0xaf} },
	{ ASCASW,	ynone,	Pe, {0xaf} },
	{ ASETCC,	yscond,	Pm, {0x93,(00)} },
	{ ASETCS,	yscond,	Pm, {0x92,(00)} },
	{ ASETEQ,	yscond,	Pm, {0x94,(00)} },
	{ ASETGE,	yscond,	Pm, {0x9d,(00)} },
	{ ASETGT,	yscond,	Pm, {0x9f,(00)} },
	{ ASETHI,	yscond,	Pm, {0x97,(00)} },
	{ ASETLE,	yscond,	Pm, {0x9e,(00)} },
	{ ASETLS,	yscond,	Pm, {0x96,(00)} },
	{ ASETLT,	yscond,	Pm, {0x9c,(00)} },
	{ ASETMI,	yscond,	Pm, {0x98,(00)} },
	{ ASETNE,	yscond,	Pm, {0x95,(00)} },
	{ ASETOC,	yscond,	Pm, {0x91,(00)} },
	{ ASETOS,	yscond,	Pm, {0x90,(00)} },
	{ ASETPC,	yscond,	Pm, {0x9b,(00)} },
	{ ASETPL,	yscond,	Pm, {0x99,(00)} },
	{ ASETPS,	yscond,	Pm, {0x9a,(00)} },
	{ ACDQ,		ynone,	Px, {0x99} },
	{ ACWD,		ynone,	Pe, {0x99} },
	{ ASHLB,	yshb,	Pb, {0xd0,(04),0xc0,(04),0xd2,(04)} },
	{ ASHLL,	yshl,	Px, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASHLW,	yshl,	Pe, {0xd1,(04),0xc1,(04),0xd3,(04),0xd3,(04)} },
	{ ASHRB,	yshb,	Pb, {0xd0,(05),0xc0,(05),0xd2,(05)} },
	{ ASHRL,	yshl,	Px, {0xd1,(05),0xc1,(05),0xd3,(05),0xd3,(05)} },
	{ ASHRW,	yshl,	Pe, {0xd1,(05),0xc1,(05),0xd3,(05),0xd3,(05)} },
	{ ASTC,		ynone,	Px, {0xf9} },
	{ ASTD,		ynone,	Px, {0xfd} },
	{ ASTI,		ynone,	Px, {0xfb} },
	{ ASTOSB,	ynone,	Pb, {0xaa} },
	{ ASTOSL,	ynone,	Px, {0xab} },
	{ ASTOSW,	ynone,	Pe, {0xab} },
	{ ASUBB,	yxorb,	Pb, {0x2c,0x80,(05),0x28,0x2a} },
	{ ASUBL,	yaddl,	Px, {0x83,(05),0x2d,0x81,(05),0x29,0x2b} },
	{ ASUBW,	yaddl,	Pe, {0x83,(05),0x2d,0x81,(05),0x29,0x2b} },
	{ ASYSCALL,	ynone,	Px, {0xcd,100} },
	{ ATESTB,	ytestb,	Pb, {0xa8,0xf6,(00),0x84,0x84} },
	{ ATESTL,	ytestl,	Px, {0xa9,0xf7,(00),0x85,0x85} },
	{ ATESTW,	ytestl,	Pe, {0xa9,0xf7,(00),0x85,0x85} },
	{ ATEXT,	ytext,	Px },
	{ AVERR,	ydivl,	Pm, {0x00,(04)} },
	{ AVERW,	ydivl,	Pm, {0x00,(05)} },
	{ AWAIT,	ynone,	Px, {0x9b} },
	{ AWORD,	ybyte,	Px, {2} },
	{ AXCHGB,	yml_mb,	Pb, {0x86,0x86} },
	{ AXCHGL,	yxchg,	Px, {0x90,0x90,0x87,0x87} },
	{ AXCHGW,	yxchg,	Pe, {0x90,0x90,0x87,0x87} },
	{ AXLAT,	ynone,	Px, {0xd7} },
	{ AXORB,	yxorb,	Pb, {0x34,0x80,(06),0x30,0x32} },
	{ AXORL,	yxorl,	Px, {0x83,(06),0x35,0x81,(06),0x31,0x33} },
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
	{ AFCOMI,	yfmvx,	Px, {0xdb,(06)} },
	{ AFCOMIP,	yfmvx,	Px, {0xdf,(06)} },
	{ AFCOML,	yfmvx,	Px, {0xda,(02)} },
	{ AFCOMLP,	yfmvx,	Px, {0xda,(03)} },
	{ AFCOMW,	yfmvx,	Px, {0xde,(02)} },
	{ AFCOMWP,	yfmvx,	Px, {0xde,(03)} },

	{ AFUCOM,	ycompp,	Px, {0xdd,(04)} },
	{ AFUCOMI,	ycompp,	Px, {0xdb,(05)} },
	{ AFUCOMIP,	ycompp,	Px, {0xdf,(05)} },
	{ AFUCOMP,	ycompp,	Px, {0xdd,(05)} },
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
	{ AEND },
	{ ADYNT_ },
	{ AINIT_ },
	{ ASIGNAME },
	{ ACMPXCHGB,	yrb_mb,	Pm, {0xb0} },
	{ ACMPXCHGL,	yrl_ml,	Pm, {0xb1} },
	{ ACMPXCHGW,	yrl_ml,	Pm, {0xb1} },
	{ ACMPXCHG8B,	yscond,	Pm, {0xc7,(01)} },

	{ ACPUID,	ynone,	Pm, {0xa2} },
	{ ARDTSC,	ynone,	Pm, {0x31} },

	{ AXADDB,	yrb_mb,	Pb, {0x0f,0xc0} },
	{ AXADDL,	yrl_ml,	Pm, {0xc1} },
	{ AXADDW,	yrl_ml,	Pe, {0x0f,0xc1} },

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

	{ AFCMOVCC,	yfcmv,	Px, {0xdb,(00)} },
	{ AFCMOVCS,	yfcmv,	Px, {0xda,(00)} },
	{ AFCMOVEQ,	yfcmv,	Px, {0xda,(01)} },
	{ AFCMOVHI,	yfcmv,	Px, {0xdb,(02)} },
	{ AFCMOVLS,	yfcmv,	Px, {0xda,(02)} },
	{ AFCMOVNE,	yfcmv,	Px, {0xdb,(01)} },
	{ AFCMOVNU,	yfcmv,	Px, {0xdb,(03)} },
	{ AFCMOVUN,	yfcmv,	Px, {0xda,(03)} },

	{ ALFENCE, ynone, Pm, {0xae,0xe8} },
	{ AMFENCE, ynone, Pm, {0xae,0xf0} },
	{ ASFENCE, ynone, Pm, {0xae,0xf8} },

	{ AEMMS, ynone, Pm, {0x77} },

	{ APREFETCHT0,	yprefetch,	Pm,	{0x18,(01)} },
	{ APREFETCHT1,	yprefetch,	Pm,	{0x18,(02)} },
	{ APREFETCHT2,	yprefetch,	Pm,	{0x18,(03)} },
	{ APREFETCHNTA,	yprefetch,	Pm,	{0x18,(00)} },

	{ ABSWAPL,	ybswap,	Pm,	{0xc8} },
	
	{ AUNDEF,		ynone,	Px,	{0x0f, 0x0b} },

	{ AADDPD,	yxm,	Pq, {0x58} },
	{ AADDPS,	yxm,	Pm, {0x58} },
	{ AADDSD,	yxm,	Pf2, {0x58} },
	{ AADDSS,	yxm,	Pf3, {0x58} },
	{ AANDNPD,	yxm,	Pq, {0x55} },
	{ AANDNPS,	yxm,	Pm, {0x55} },
	{ AANDPD,	yxm,	Pq, {0x54} },
	{ AANDPS,	yxm,	Pq, {0x54} },
	{ ACMPPD,	yxcmpi,	Px, {Pe,0xc2} },
	{ ACMPPS,	yxcmpi,	Pm, {0xc2,0} },
	{ ACMPSD,	yxcmpi,	Px, {Pf2,0xc2} },
	{ ACMPSS,	yxcmpi,	Px, {Pf3,0xc2} },
	{ ACOMISD,	yxcmp,	Pe, {0x2f} },
	{ ACOMISS,	yxcmp,	Pm, {0x2f} },
	{ ACVTPL2PD,	yxcvm2,	Px, {Pf3,0xe6,Pe,0x2a} },
	{ ACVTPL2PS,	yxcvm2,	Pm, {0x5b,0,0x2a,0,} },
	{ ACVTPD2PL,	yxcvm1,	Px, {Pf2,0xe6,Pe,0x2d} },
	{ ACVTPD2PS,	yxm,	Pe, {0x5a} },
	{ ACVTPS2PL,	yxcvm1, Px, {Pe,0x5b,Pm,0x2d} },
	{ ACVTPS2PD,	yxm,	Pm, {0x5a} },
	{ ACVTSD2SL,	yxcvfl, Pf2, {0x2d} },
 	{ ACVTSD2SS,	yxm,	Pf2, {0x5a} },
	{ ACVTSL2SD,	yxcvlf, Pf2, {0x2a} },
	{ ACVTSL2SS,	yxcvlf, Pf3, {0x2a} },
	{ ACVTSS2SD,	yxm,	Pf3, {0x5a} },
	{ ACVTSS2SL,	yxcvfl, Pf3, {0x2d} },
	{ ACVTTPD2PL,	yxcvm1,	Px, {Pe,0xe6,Pe,0x2c} },
	{ ACVTTPS2PL,	yxcvm1,	Px, {Pf3,0x5b,Pm,0x2c} },
	{ ACVTTSD2SL,	yxcvfl, Pf2, {0x2c} },
	{ ACVTTSS2SL,	yxcvfl,	Pf3, {0x2c} },
	{ ADIVPD,	yxm,	Pe, {0x5e} },
	{ ADIVPS,	yxm,	Pm, {0x5e} },
	{ ADIVSD,	yxm,	Pf2, {0x5e} },
	{ ADIVSS,	yxm,	Pf3, {0x5e} },
	{ AMASKMOVOU,	yxr,	Pe, {0xf7} },
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
	{ AMOVO,	yxmov,	Pe, {0x6f,0x7f} },
	{ AMOVOU,	yxmov,	Pf3, {0x6f,0x7f} },
	{ AMOVHLPS,	yxr,	Pm, {0x12} },
	{ AMOVHPD,	yxmov,	Pe, {0x16,0x17} },
	{ AMOVHPS,	yxmov,	Pm, {0x16,0x17} },
	{ AMOVLHPS,	yxr,	Pm, {0x16} },
	{ AMOVLPD,	yxmov,	Pe, {0x12,0x13} },
	{ AMOVLPS,	yxmov,	Pm, {0x12,0x13} },
	{ AMOVMSKPD,	yxrrl,	Pq, {0x50} },
	{ AMOVMSKPS,	yxrrl,	Pm, {0x50} },
	{ AMOVNTO,	yxr_ml,	Pe, {0xe7} },
	{ AMOVNTPD,	yxr_ml,	Pe, {0x2b} },
	{ AMOVNTPS,	yxr_ml,	Pm, {0x2b} },
	{ AMOVSD,	yxmov,	Pf2, {0x10,0x11} },
	{ AMOVSS,	yxmov,	Pf3, {0x10,0x11} },
	{ AMOVUPD,	yxmov,	Pe, {0x10,0x11} },
	{ AMOVUPS,	yxmov,	Pm, {0x10,0x11} },
	{ AMULPD,	yxm,	Pe, {0x59} },
	{ AMULPS,	yxm,	Ym, {0x59} },
	{ AMULSD,	yxm,	Pf2, {0x59} },
	{ AMULSS,	yxm,	Pf3, {0x59} },
	{ AORPD,	yxm,	Pq, {0x56} },
	{ AORPS,	yxm,	Pm, {0x56} },
	{ APADDQ,	yxm,	Pe, {0xd4} },
	{ APAND,	yxm,	Pe, {0xdb} },
	{ APCMPEQB,	yxmq,	Pe, {0x74} },
	{ APMAXSW,	yxm,	Pe, {0xee} },
	{ APMAXUB,	yxm,	Pe, {0xde} },
	{ APMINSW,	yxm,	Pe, {0xea} },
	{ APMINUB,	yxm,	Pe, {0xda} },
	{ APMOVMSKB,	ymskb,	Px, {Pe,0xd7,0xd7} },
	{ APSADBW,	yxm,	Pq, {0xf6} },
	{ APSUBB,	yxm,	Pe, {0xf8} },
	{ APSUBL,	yxm,	Pe, {0xfa} },
	{ APSUBQ,	yxm,	Pe, {0xfb} },
	{ APSUBSB,	yxm,	Pe, {0xe8} },
	{ APSUBSW,	yxm,	Pe, {0xe9} },
	{ APSUBUSB,	yxm,	Pe, {0xd8} },
	{ APSUBUSW,	yxm,	Pe, {0xd9} },
	{ APSUBW,	yxm,	Pe, {0xf9} },
	{ APUNPCKHQDQ,	yxm,	Pe, {0x6d} },
	{ APUNPCKLQDQ,	yxm,	Pe, {0x6c} },
	{ APXOR,	yxm,	Pe, {0xef} },
	{ ARCPPS,	yxm,	Pm, {0x53} },
	{ ARCPSS,	yxm,	Pf3, {0x53} },
	{ ARSQRTPS,	yxm,	Pm, {0x52} },
	{ ARSQRTSS,	yxm,	Pf3, {0x52} },
	{ ASQRTPD,	yxm,	Pe, {0x51} },
	{ ASQRTPS,	yxm,	Pm, {0x51} },
	{ ASQRTSD,	yxm,	Pf2, {0x51} },
	{ ASQRTSS,	yxm,	Pf3, {0x51} },
	{ ASUBPD,	yxm,	Pe, {0x5c} },
	{ ASUBPS,	yxm,	Pm, {0x5c} },
	{ ASUBSD,	yxm,	Pf2, {0x5c} },
	{ ASUBSS,	yxm,	Pf3, {0x5c} },
	{ AUCOMISD,	yxcmp,	Pe, {0x2e} },
	{ AUCOMISS,	yxcmp,	Pm, {0x2e} },
	{ AUNPCKHPD,	yxm,	Pe, {0x15} },
	{ AUNPCKHPS,	yxm,	Pm, {0x15} },
	{ AUNPCKLPD,	yxm,	Pe, {0x14} },
	{ AUNPCKLPS,	yxm,	Pm, {0x14} },
	{ AXORPD,	yxm,	Pe, {0x57} },
	{ AXORPS,	yxm,	Pm, {0x57} },
	{ APSHUFHW,	yxshuf,	Pf3, {0x70,(00)} },
	{ APSHUFL,	yxshuf,	Pq, {0x70,(00)} },
	{ APSHUFLW,	yxshuf,	Pf2, {0x70,(00)} },


	{ AAESENC,	yaes,	Pq, {0x38,0xdc,(0)} },
	{ APINSRD,	yinsrd,	Pq, {0x3a, 0x22, (00)} },
	{ APSHUFB,	ymshufb,Pq, {0x38, 0x00} },

	{ AUSEFIELD,	ynop,	Px, {0,0} },
	{ ATYPE },
	{ AFUNCDATA,	yfuncdata,	Px, {0,0} },
	{ APCDATA,	ypcdata,	Px, {0,0} },
	{ ACHECKNIL },
	{ AVARDEF },
	{ AVARKILL },
	{ ADUFFCOPY,	yduff,	Px, {0xe8} },
	{ ADUFFZERO,	yduff,	Px, {0xe8} },

	{0}
};

static int32	vaddr(Link*, Addr*, Reloc*);

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

static int32
naclpad(Link *ctxt, LSym *s, int32 c, int32 pad)
{
	symgrow(ctxt, s, c+pad);
	fillnop(s->p+c, pad);
	return c+pad;
}

static void instinit(void);

void
span8(Link *ctxt, LSym *s)
{
	Prog *p, *q;
	int32 c, v, loop;
	uchar *bp;
	int n, m, i;

	ctxt->cursym = s;

	if(s->text == nil || s->text->link == nil)
		return;

	if(ycover[0] == 0)
		instinit();

	for(p = s->text; p != nil; p = p->link) {
		n = 0;
		if(p->to.type == D_BRANCH)
			if(p->pcond == nil)
				p->pcond = p;
		if((q = p->pcond) != nil)
			if(q->back != 2)
				n = 1;
		p->back = n;
		if(p->as == AADJSP) {
			p->to.type = D_SP;
			v = -p->from.offset;
			p->from.offset = v;
			p->as = AADDL;
			if(v < 0) {
				p->as = ASUBL;
				v = -v;
				p->from.offset = v;
			}
			if(v == 0)
				p->as = ANOP;
		}
	}

	for(p = s->text; p != nil; p = p->link) {
		p->back = 2;	// use short branches first time through
		if((q = p->pcond) != nil && (q->back & 2))
			p->back |= 1;	// backward jump

		if(p->as == AADJSP) {
			p->to.type = D_SP;
			v = -p->from.offset;
			p->from.offset = v;
			p->as = AADDL;
			if(v < 0) {
				p->as = ASUBL;
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
				if(p->as == AREP && (c>>5) != ((c+3-1)>>5))
					c = naclpad(ctxt, s, c, -c&31);
				
				// same for LOCK.
				// various instructions follow; the longest is 4 bytes.
				// give ourselves 8 bytes so as to avoid surprises.
				if(p->as == ALOCK && (c>>5) != ((c+8-1)>>5))
					c = naclpad(ctxt, s, c, -c&31);
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
					if(q->as == AJCXZW)
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
			sysfatal("bad code");
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
	int i;

	for(i=1; optab[i].as; i++)
		if(i != optab[i].as)
			sysfatal("phase error in optab: at %A found %A", i, optab[i].as);

	for(i=0; i<Ymax; i++)
		ycover[i*Ymax + i] = 1;

	ycover[Yi0*Ymax + Yi8] = 1;
	ycover[Yi1*Ymax + Yi8] = 1;

	ycover[Yi0*Ymax + Yi32] = 1;
	ycover[Yi1*Ymax + Yi32] = 1;
	ycover[Yi8*Ymax + Yi32] = 1;

	ycover[Yal*Ymax + Yrb] = 1;
	ycover[Ycl*Ymax + Yrb] = 1;
	ycover[Yax*Ymax + Yrb] = 1;
	ycover[Ycx*Ymax + Yrb] = 1;
	ycover[Yrx*Ymax + Yrb] = 1;

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

	for(i=0; i<D_NONE; i++) {
		reg[i] = -1;
		if(i >= D_AL && i <= D_BH)
			reg[i] = (i-D_AL) & 7;
		if(i >= D_AX && i <= D_DI)
			reg[i] = (i-D_AX) & 7;
		if(i >= D_F0 && i <= D_F0+7)
			reg[i] = (i-D_F0) & 7;
		if(i >= D_X0 && i <= D_X0+7)
			reg[i] = (i-D_X0) & 7;
	}
}

static int
prefixof(Link *ctxt, Addr *a)
{
	switch(a->type) {
	case D_INDIR+D_CS:
		return 0x2e;
	case D_INDIR+D_DS:
		return 0x3e;
	case D_INDIR+D_ES:
		return 0x26;
	case D_INDIR+D_FS:
		return 0x64;
	case D_INDIR+D_GS:
		return 0x65;
	case D_INDIR+D_TLS:
		// NOTE: Systems listed here should be only systems that
		// support direct TLS references like 8(TLS) implemented as
		// direct references from FS or GS. Systems that require
		// the initial-exec model, where you load the TLS base into
		// a register and then index from that register, do not reach
		// this code and should not be listed.
		switch(ctxt->headtype) {
		default:
			sysfatal("unknown TLS base register for %s", headstr(ctxt->headtype));
		case Hdarwin:
		case Hdragonfly:
		case Hfreebsd:
		case Hnetbsd:
		case Hopenbsd:
			return 0x65; // GS
		}
	}
	return 0;
}

static int
oclass(Addr *a)
{
	int32 v;

	if((a->type >= D_INDIR && a->type < 2*D_INDIR) || a->index != D_NONE) {
		if(a->index != D_NONE && a->scale == 0) {
			if(a->type == D_ADDR) {
				switch(a->index) {
				case D_EXTERN:
				case D_STATIC:
					return Yi32;
				case D_AUTO:
				case D_PARAM:
					return Yiauto;
				}
				return Yxxx;
			}
			//if(a->type == D_INDIR+D_ADDR)
			//	print("*Ycol\n");
			return Ycol;
		}
		return Ym;
	}
	switch(a->type)
	{
	case D_AL:
		return Yal;

	case D_AX:
		return Yax;

	case D_CL:
	case D_DL:
	case D_BL:
	case D_AH:
	case D_CH:
	case D_DH:
	case D_BH:
		return Yrb;

	case D_CX:
		return Ycx;

	case D_DX:
	case D_BX:
		return Yrx;

	case D_SP:
	case D_BP:
	case D_SI:
	case D_DI:
		return Yrl;

	case D_F0+0:
		return	Yf0;

	case D_F0+1:
	case D_F0+2:
	case D_F0+3:
	case D_F0+4:
	case D_F0+5:
	case D_F0+6:
	case D_F0+7:
		return	Yrf;

	case D_X0+0:
	case D_X0+1:
	case D_X0+2:
	case D_X0+3:
	case D_X0+4:
	case D_X0+5:
	case D_X0+6:
	case D_X0+7:
		return	Yxr;

	case D_NONE:
		return Ynone;

	case D_CS:	return	Ycs;
	case D_SS:	return	Yss;
	case D_DS:	return	Yds;
	case D_ES:	return	Yes;
	case D_FS:	return	Yfs;
	case D_GS:	return	Ygs;
	case D_TLS:	return	Ytls;

	case D_GDTR:	return	Ygdtr;
	case D_IDTR:	return	Yidtr;
	case D_LDTR:	return	Yldtr;
	case D_MSW:	return	Ymsw;
	case D_TASK:	return	Ytask;

	case D_CR+0:	return	Ycr0;
	case D_CR+1:	return	Ycr1;
	case D_CR+2:	return	Ycr2;
	case D_CR+3:	return	Ycr3;
	case D_CR+4:	return	Ycr4;
	case D_CR+5:	return	Ycr5;
	case D_CR+6:	return	Ycr6;
	case D_CR+7:	return	Ycr7;

	case D_DR+0:	return	Ydr0;
	case D_DR+1:	return	Ydr1;
	case D_DR+2:	return	Ydr2;
	case D_DR+3:	return	Ydr3;
	case D_DR+4:	return	Ydr4;
	case D_DR+5:	return	Ydr5;
	case D_DR+6:	return	Ydr6;
	case D_DR+7:	return	Ydr7;

	case D_TR+0:	return	Ytr0;
	case D_TR+1:	return	Ytr1;
	case D_TR+2:	return	Ytr2;
	case D_TR+3:	return	Ytr3;
	case D_TR+4:	return	Ytr4;
	case D_TR+5:	return	Ytr5;
	case D_TR+6:	return	Ytr6;
	case D_TR+7:	return	Ytr7;

	case D_EXTERN:
	case D_STATIC:
	case D_AUTO:
	case D_PARAM:
		return Ym;

	case D_CONST:
	case D_CONST2:
	case D_ADDR:
		if(a->sym == nil) {
			v = a->offset;
			if(v == 0)
				return Yi0;
			if(v == 1)
				return Yi1;
			if(v >= -128 && v <= 127)
				return Yi8;
		}
		return Yi32;

	case D_BRANCH:
		return Ybr;
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

	case D_NONE:
		i = 4 << 3;
		goto bas;

	case D_AX:
	case D_CX:
	case D_DX:
	case D_BX:
	case D_BP:
	case D_SI:
	case D_DI:
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
	case D_NONE:	/* must be mod=00 */
		i |= 5;
		break;
	case D_AX:
	case D_CX:
	case D_DX:
	case D_BX:
	case D_SP:
	case D_BP:
	case D_SI:
	case D_DI:
		i |= reg[base];
		break;
	}
	*ctxt->andptr++ = i;
	return;
bad:
	ctxt->diag("asmidx: bad address %d,%d,%d", scale, index, base);
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
	
	v = vaddr(ctxt, a, &rel);
	if(rel.siz != 0) {
		if(rel.siz != 4)
			ctxt->diag("bad reloc");
		r = addrel(ctxt->cursym);
		*r = rel;
		r->off = p->pc + ctxt->andptr - ctxt->and;
	}
	put4(ctxt, v);
}

static int32
vaddr(Link *ctxt, Addr *a, Reloc *r)
{
	int t;
	int32 v;
	LSym *s;
	
	if(r != nil)
		memset(r, 0, sizeof *r);

	t = a->type;
	v = a->offset;
	if(t == D_ADDR)
		t = a->index;
	switch(t) {
	case D_STATIC:
	case D_EXTERN:
		s = a->sym;
		if(s != nil) {
			if(r == nil) {
				ctxt->diag("need reloc for %D", a);
				sysfatal("bad code");
			}
			r->type = R_ADDR;
			r->siz = 4;
			r->off = -1;
			r->sym = s;
			r->add = v;
			v = 0;
		}
		break;
	
	case D_INDIR+D_TLS:
		if(r == nil) {
			ctxt->diag("need reloc for %D", a);
			sysfatal("bad code");
		}
		r->type = R_TLS_LE;
		r->siz = 4;
		r->off = -1; // caller must fill in
		r->add = v;
		v = 0;
		break;
	}
	return v;
}

static void
asmand(Link *ctxt, Addr *a, int r)
{
	int32 v;
	int t, scale;
	Reloc rel;

	v = a->offset;
	t = a->type;
	rel.siz = 0;
	if(a->index != D_NONE && a->index != D_TLS) {
		if(t < D_INDIR || t >= 2*D_INDIR) {
			switch(t) {
			default:
				goto bad;
			case D_STATIC:
			case D_EXTERN:
				t = D_NONE;
				v = vaddr(ctxt, a, &rel);
				break;
			case D_AUTO:
			case D_PARAM:
				t = D_SP;
				break;
			}
		} else
			t -= D_INDIR;

		if(t == D_NONE) {
			*ctxt->andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, a->scale, a->index, t);
			goto putrelv;
		}
		if(v == 0 && rel.siz == 0 && t != D_BP) {
			*ctxt->andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, a->scale, a->index, t);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0) {
			*ctxt->andptr++ = (1 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, a->scale, a->index, t);
			*ctxt->andptr++ = v;
			return;
		}
		*ctxt->andptr++ = (2 << 6) | (4 << 0) | (r << 3);
		asmidx(ctxt, a->scale, a->index, t);
		goto putrelv;
	}
	if(t >= D_AL && t <= D_F7 || t >= D_X0 && t <= D_X7) {
		if(v)
			goto bad;
		*ctxt->andptr++ = (3 << 6) | (reg[t] << 0) | (r << 3);
		return;
	}
	
	scale = a->scale;
	if(t < D_INDIR || t >= 2*D_INDIR) {
		switch(a->type) {
		default:
			goto bad;
		case D_STATIC:
		case D_EXTERN:
			t = D_NONE;
			v = vaddr(ctxt, a, &rel);
			break;
		case D_AUTO:
		case D_PARAM:
			t = D_SP;
			break;
		}
		scale = 1;
	} else
		t -= D_INDIR;
	if(t == D_TLS)
		v = vaddr(ctxt, a, &rel);

	if(t == D_NONE || (D_CS <= t && t <= D_GS) || t == D_TLS) {
		*ctxt->andptr++ = (0 << 6) | (5 << 0) | (r << 3);
		goto putrelv;
	}
	if(t == D_SP) {
		if(v == 0 && rel.siz == 0) {
			*ctxt->andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, scale, D_NONE, t);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0) {
			*ctxt->andptr++ = (1 << 6) | (4 << 0) | (r << 3);
			asmidx(ctxt, scale, D_NONE, t);
			*ctxt->andptr++ = v;
			return;
		}
		*ctxt->andptr++ = (2 << 6) | (4 << 0) | (r << 3);
		asmidx(ctxt, scale, D_NONE, t);
		goto putrelv;
	}
	if(t >= D_AX && t <= D_DI) {
		if(a->index == D_TLS) {
			memset(&rel, 0, sizeof rel);
			rel.type = R_TLS_IE;
			rel.siz = 4;
			rel.sym = nil;
			rel.add = v;
			v = 0;
		}
		if(v == 0 && rel.siz == 0 && t != D_BP) {
			*ctxt->andptr++ = (0 << 6) | (reg[t] << 0) | (r << 3);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0)  {
			ctxt->andptr[0] = (1 << 6) | (reg[t] << 0) | (r << 3);
			ctxt->andptr[1] = v;
			ctxt->andptr += 2;
			return;
		}
		*ctxt->andptr++ = (2 << 6) | (reg[t] << 0) | (r << 3);
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

enum
{
	E = 0xff,
};

static uchar	ymovtab[] =
{
/* push */
	APUSHL,	Ycs,	Ynone,	0,	0x0e,E,0,0,
	APUSHL,	Yss,	Ynone,	0,	0x16,E,0,0,
	APUSHL,	Yds,	Ynone,	0,	0x1e,E,0,0,
	APUSHL,	Yes,	Ynone,	0,	0x06,E,0,0,
	APUSHL,	Yfs,	Ynone,	0,	0x0f,0xa0,E,0,
	APUSHL,	Ygs,	Ynone,	0,	0x0f,0xa8,E,0,

	APUSHW,	Ycs,	Ynone,	0,	Pe,0x0e,E,0,
	APUSHW,	Yss,	Ynone,	0,	Pe,0x16,E,0,
	APUSHW,	Yds,	Ynone,	0,	Pe,0x1e,E,0,
	APUSHW,	Yes,	Ynone,	0,	Pe,0x06,E,0,
	APUSHW,	Yfs,	Ynone,	0,	Pe,0x0f,0xa0,E,
	APUSHW,	Ygs,	Ynone,	0,	Pe,0x0f,0xa8,E,

/* pop */
	APOPL,	Ynone,	Yds,	0,	0x1f,E,0,0,
	APOPL,	Ynone,	Yes,	0,	0x07,E,0,0,
	APOPL,	Ynone,	Yss,	0,	0x17,E,0,0,
	APOPL,	Ynone,	Yfs,	0,	0x0f,0xa1,E,0,
	APOPL,	Ynone,	Ygs,	0,	0x0f,0xa9,E,0,

	APOPW,	Ynone,	Yds,	0,	Pe,0x1f,E,0,
	APOPW,	Ynone,	Yes,	0,	Pe,0x07,E,0,
	APOPW,	Ynone,	Yss,	0,	Pe,0x17,E,0,
	APOPW,	Ynone,	Yfs,	0,	Pe,0x0f,0xa1,E,
	APOPW,	Ynone,	Ygs,	0,	Pe,0x0f,0xa9,E,

/* mov seg */
	AMOVW,	Yes,	Yml,	1,	0x8c,0,0,0,
	AMOVW,	Ycs,	Yml,	1,	0x8c,1,0,0,
	AMOVW,	Yss,	Yml,	1,	0x8c,2,0,0,
	AMOVW,	Yds,	Yml,	1,	0x8c,3,0,0,
	AMOVW,	Yfs,	Yml,	1,	0x8c,4,0,0,
	AMOVW,	Ygs,	Yml,	1,	0x8c,5,0,0,

	AMOVW,	Yml,	Yes,	2,	0x8e,0,0,0,
	AMOVW,	Yml,	Ycs,	2,	0x8e,1,0,0,
	AMOVW,	Yml,	Yss,	2,	0x8e,2,0,0,
	AMOVW,	Yml,	Yds,	2,	0x8e,3,0,0,
	AMOVW,	Yml,	Yfs,	2,	0x8e,4,0,0,
	AMOVW,	Yml,	Ygs,	2,	0x8e,5,0,0,

/* mov cr */
	AMOVL,	Ycr0,	Yml,	3,	0x0f,0x20,0,0,
	AMOVL,	Ycr2,	Yml,	3,	0x0f,0x20,2,0,
	AMOVL,	Ycr3,	Yml,	3,	0x0f,0x20,3,0,
	AMOVL,	Ycr4,	Yml,	3,	0x0f,0x20,4,0,

	AMOVL,	Yml,	Ycr0,	4,	0x0f,0x22,0,0,
	AMOVL,	Yml,	Ycr2,	4,	0x0f,0x22,2,0,
	AMOVL,	Yml,	Ycr3,	4,	0x0f,0x22,3,0,
	AMOVL,	Yml,	Ycr4,	4,	0x0f,0x22,4,0,

/* mov dr */
	AMOVL,	Ydr0,	Yml,	3,	0x0f,0x21,0,0,
	AMOVL,	Ydr6,	Yml,	3,	0x0f,0x21,6,0,
	AMOVL,	Ydr7,	Yml,	3,	0x0f,0x21,7,0,

	AMOVL,	Yml,	Ydr0,	4,	0x0f,0x23,0,0,
	AMOVL,	Yml,	Ydr6,	4,	0x0f,0x23,6,0,
	AMOVL,	Yml,	Ydr7,	4,	0x0f,0x23,7,0,

/* mov tr */
	AMOVL,	Ytr6,	Yml,	3,	0x0f,0x24,6,0,
	AMOVL,	Ytr7,	Yml,	3,	0x0f,0x24,7,0,

	AMOVL,	Yml,	Ytr6,	4,	0x0f,0x26,6,E,
	AMOVL,	Yml,	Ytr7,	4,	0x0f,0x26,7,E,

/* lgdt, sgdt, lidt, sidt */
	AMOVL,	Ym,	Ygdtr,	4,	0x0f,0x01,2,0,
	AMOVL,	Ygdtr,	Ym,	3,	0x0f,0x01,0,0,
	AMOVL,	Ym,	Yidtr,	4,	0x0f,0x01,3,0,
	AMOVL,	Yidtr,	Ym,	3,	0x0f,0x01,1,0,

/* lldt, sldt */
	AMOVW,	Yml,	Yldtr,	4,	0x0f,0x00,2,0,
	AMOVW,	Yldtr,	Yml,	3,	0x0f,0x00,0,0,

/* lmsw, smsw */
	AMOVW,	Yml,	Ymsw,	4,	0x0f,0x01,6,0,
	AMOVW,	Ymsw,	Yml,	3,	0x0f,0x01,4,0,

/* ltr, str */
	AMOVW,	Yml,	Ytask,	4,	0x0f,0x00,3,0,
	AMOVW,	Ytask,	Yml,	3,	0x0f,0x00,1,0,

/* load full pointer */
	AMOVL,	Yml,	Ycol,	5,	0,0,0,0,
	AMOVW,	Yml,	Ycol,	5,	Pe,0,0,0,

/* double shift */
	ASHLL,	Ycol,	Yml,	6,	0xa4,0xa5,0,0,
	ASHRL,	Ycol,	Yml,	6,	0xac,0xad,0,0,

/* extra imul */
	AIMULW,	Yml,	Yrl,	7,	Pq,0xaf,0,0,
	AIMULL,	Yml,	Yrl,	7,	Pm,0xaf,0,0,

/* load TLS base pointer */
	AMOVL,	Ytls,	Yrl,	8,	0,0,0,0,

	0
};

// byteswapreg returns a byte-addressable register (AX, BX, CX, DX)
// which is not referenced in a->type.
// If a is empty, it returns BX to account for MULB-like instructions
// that might use DX and AX.
static int
byteswapreg(Link *ctxt, Addr *a)
{
	int cana, canb, canc, cand;

	cana = canb = canc = cand = 1;

	switch(a->type) {
	case D_NONE:
		cana = cand = 0;
		break;
	case D_AX:
	case D_AL:
	case D_AH:
	case D_INDIR+D_AX:
		cana = 0;
		break;
	case D_BX:
	case D_BL:
	case D_BH:
	case D_INDIR+D_BX:
		canb = 0;
		break;
	case D_CX:
	case D_CL:
	case D_CH:
	case D_INDIR+D_CX:
		canc = 0;
		break;
	case D_DX:
	case D_DL:
	case D_DH:
	case D_INDIR+D_DX:
		cand = 0;
		break;
	}
	switch(a->index) {
	case D_AX:
		cana = 0;
		break;
	case D_BX:
		canb = 0;
		break;
	case D_CX:
		canc = 0;
		break;
	case D_DX:
		cand = 0;
		break;
	}
	if(cana)
		return D_AX;
	if(canb)
		return D_BX;
	if(canc)
		return D_CX;
	if(cand)
		return D_DX;

	ctxt->diag("impossible byte register");
	sysfatal("bad code");
	return 0;
}

static void
subreg(Prog *p, int from, int to)
{

	if(0 /* debug['Q'] */)
		print("\n%P	s/%R/%R/\n", p, from, to);

	if(p->from.type == from) {
		p->from.type = to;
		p->ft = 0;
	}
	if(p->to.type == from) {
		p->to.type = to;
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

	from += D_INDIR;
	if(p->from.type == from) {
		p->from.type = to+D_INDIR;
		p->ft = 0;
	}
	if(p->to.type == from) {
		p->to.type = to+D_INDIR;
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
	int z, op, ft, tt, breg;
	int32 v, pre;
	Reloc rel, *r;
	Addr *a;
	
	ctxt->curp = p;	// TODO

	pre = prefixof(ctxt, &p->from);
	if(pre)
		*ctxt->andptr++ = pre;
	pre = prefixof(ctxt, &p->to);
	if(pre)
		*ctxt->andptr++ = pre;

	if(p->ft == 0)
		p->ft = oclass(&p->from);
	if(p->tt == 0)
		p->tt = oclass(&p->to);

	ft = p->ft * Ymax;
	tt = p->tt * Ymax;
	o = &optab[p->as];
	t = o->ytab;
	if(t == 0) {
		ctxt->diag("asmins: noproto %P", p);
		return;
	}
	for(z=0; *t; z+=t[3],t+=4)
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

	case Pb:	/* botch */
		break;
	}

	op = o->op[z];
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
		asmand(ctxt, &p->from, reg[p->to.type]);
		break;

	case Zm_r:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->from, reg[p->to.type]);
		break;

	case Zm2_r:
		*ctxt->andptr++ = op;
		*ctxt->andptr++ = o->op[z+1];
		asmand(ctxt, &p->from, reg[p->to.type]);
		break;

	case Zm_r_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, &p->from, reg[p->to.type]);
		break;

	case Zm_r_i_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, &p->from, reg[p->to.type]);
		*ctxt->andptr++ = p->to.offset;
		break;

	case Zibm_r:
		while ((op = o->op[z++]) != 0)
			*ctxt->andptr++ = op;
		asmand(ctxt, &p->from, reg[p->to.type]);
		*ctxt->andptr++ = p->to.offset;
		break;

	case Zaut_r:
		*ctxt->andptr++ = 0x8d;	/* leal */
		if(p->from.type != D_ADDR)
			ctxt->diag("asmins: Zaut sb type ADDR");
		p->from.type = p->from.index;
		p->from.index = D_NONE;
		p->ft = 0;
		asmand(ctxt, &p->from, reg[p->to.type]);
		p->from.index = p->from.type;
		p->from.type = D_ADDR;
		p->ft = 0;
		break;

	case Zm_o:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->from, o->op[z+1]);
		break;

	case Zr_m:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->to, reg[p->from.type]);
		break;

	case Zr_m_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, &p->to, reg[p->from.type]);
		break;

	case Zr_m_i_xm:
		mediaop(ctxt, o, op, t[3], z);
		asmand(ctxt, &p->to, reg[p->from.type]);
		*ctxt->andptr++ = p->from.offset;
		break;

	case Zcallindreg:
		r = addrel(ctxt->cursym);
		r->off = p->pc;
		r->type = R_CALLIND;
		r->siz = 0;
		// fallthrough
	case Zo_m:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->to, o->op[z+1]);
		break;

	case Zm_ibo:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->from, o->op[z+1]);
		*ctxt->andptr++ = vaddr(ctxt, &p->to, nil);
		break;

	case Zibo_m:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->to, o->op[z+1]);
		*ctxt->andptr++ = vaddr(ctxt, &p->from, nil);
		break;

	case Z_ib:
	case Zib_:
		if(t[2] == Zib_)
			a = &p->from;
		else
			a = &p->to;
		v = vaddr(ctxt, a, nil);
		*ctxt->andptr++ = op;
		*ctxt->andptr++ = v;
		break;

	case Zib_rp:
		*ctxt->andptr++ = op + reg[p->to.type];
		*ctxt->andptr++ = vaddr(ctxt, &p->from, nil);
		break;

	case Zil_rp:
		*ctxt->andptr++ = op + reg[p->to.type];
		if(o->prefix == Pe) {
			v = vaddr(ctxt, &p->from, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		else
			relput4(ctxt, p, &p->from);
		break;

	case Zib_rr:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->to, reg[p->to.type]);
		*ctxt->andptr++ = vaddr(ctxt, &p->from, nil);
		break;

	case Z_il:
	case Zil_:
		if(t[2] == Zil_)
			a = &p->from;
		else
			a = &p->to;
		*ctxt->andptr++ = op;
		if(o->prefix == Pe) {
			v = vaddr(ctxt, a, nil);
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
			asmand(ctxt, &p->to, o->op[z+1]);
		} else {
			a = &p->to;
			asmand(ctxt, &p->from, o->op[z+1]);
		}
		if(o->prefix == Pe) {
			v = vaddr(ctxt, a, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		else
			relput4(ctxt, p, a);
		break;

	case Zil_rr:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->to, reg[p->to.type]);
		if(o->prefix == Pe) {
			v = vaddr(ctxt, &p->from, nil);
			*ctxt->andptr++ = v;
			*ctxt->andptr++ = v>>8;
		}
		else
			relput4(ctxt, p, &p->from);
		break;

	case Z_rp:
		*ctxt->andptr++ = op + reg[p->to.type];
		break;

	case Zrp_:
		*ctxt->andptr++ = op + reg[p->from.type];
		break;

	case Zclr:
		*ctxt->andptr++ = op;
		asmand(ctxt, &p->to, reg[p->to.type]);
		break;
	
	case Zcall:
		if(p->to.sym == nil) {
			ctxt->diag("call without target");
			sysfatal("bad code");
		}
		*ctxt->andptr++ = op;
		r = addrel(ctxt->cursym);
		r->off = p->pc + ctxt->andptr - ctxt->and;
		r->type = R_CALL;
		r->siz = 4;
		r->sym = p->to.sym;
		r->add = p->to.offset;
		put4(ctxt, 0);
		break;

	case Zbr:
	case Zjmp:
	case Zloop:
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
		// Fill in backward jump now.
		q = p->pcond;
		if(q == nil) {
			ctxt->diag("jmp/branch/loop without target");
			sysfatal("bad code");
		}
		if(p->back & 1) {
			v = q->pc - (p->pc + 2);
			if(v >= -128) {
				if(p->as == AJCXZW)
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
			if(p->as == AJCXZW)
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

	case Zcallcon:
	case Zjmpcon:
		if(t[2] == Zcallcon)
			*ctxt->andptr++ = op;
		else
			*ctxt->andptr++ = o->op[z+1];
		r = addrel(ctxt->cursym);
		r->off = p->pc + ctxt->andptr - ctxt->and;
		r->type = R_PCREL;
		r->siz = 4;
		r->add = p->to.offset;
		put4(ctxt, 0);
		break;
	
	case Zcallind:
		*ctxt->andptr++ = op;
		*ctxt->andptr++ = o->op[z+1];
		r = addrel(ctxt->cursym);
		r->off = p->pc + ctxt->andptr - ctxt->and;
		r->type = R_ADDR;
		r->siz = 4;
		r->add = p->to.offset;
		r->sym = p->to.sym;
		put4(ctxt, 0);
		break;

	case Zbyte:
		v = vaddr(ctxt, &p->from, &rel);
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
			}
		}
		break;

	case Zmov:
		goto domov;
	}
	return;

domov:
	for(t=ymovtab; *t; t+=8)
		if(p->as == t[0])
		if(ycover[ft+t[1]])
		if(ycover[tt+t[2]])
			goto mfound;
bad:
	/*
	 * here, the assembly has failed.
	 * if its a byte instruction that has
	 * unaddressable registers, try to
	 * exchange registers and reissue the
	 * instruction with the operands renamed.
	 */
	pp = *p;
	z = p->from.type;
	if(z >= D_BP && z <= D_DI) {
		if((breg = byteswapreg(ctxt, &p->to)) != D_AX) {
			*ctxt->andptr++ = 0x87;			/* xchg lhs,bx */
			asmand(ctxt, &p->from, reg[breg]);
			subreg(&pp, z, breg);
			doasm(ctxt, &pp);
			*ctxt->andptr++ = 0x87;			/* xchg lhs,bx */
			asmand(ctxt, &p->from, reg[breg]);
		} else {
			*ctxt->andptr++ = 0x90 + reg[z];		/* xchg lsh,ax */
			subreg(&pp, z, D_AX);
			doasm(ctxt, &pp);
			*ctxt->andptr++ = 0x90 + reg[z];		/* xchg lsh,ax */
		}
		return;
	}
	z = p->to.type;
	if(z >= D_BP && z <= D_DI) {
		if((breg = byteswapreg(ctxt, &p->from)) != D_AX) {
			*ctxt->andptr++ = 0x87;			/* xchg rhs,bx */
			asmand(ctxt, &p->to, reg[breg]);
			subreg(&pp, z, breg);
			doasm(ctxt, &pp);
			*ctxt->andptr++ = 0x87;			/* xchg rhs,bx */
			asmand(ctxt, &p->to, reg[breg]);
		} else {
			*ctxt->andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
			subreg(&pp, z, D_AX);
			doasm(ctxt, &pp);
			*ctxt->andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
		}
		return;
	}
	ctxt->diag("doasm: notfound t2=%ux from=%ux to=%ux %P", t[2], p->from.type, p->to.type, p);
	return;

mfound:
	switch(t[3]) {
	default:
		ctxt->diag("asmins: unknown mov %d %P", t[3], p);
		break;

	case 0:	/* lit */
		for(z=4; t[z]!=E; z++)
			*ctxt->andptr++ = t[z];
		break;

	case 1:	/* r,m */
		*ctxt->andptr++ = t[4];
		asmand(ctxt, &p->to, t[5]);
		break;

	case 2:	/* m,r */
		*ctxt->andptr++ = t[4];
		asmand(ctxt, &p->from, t[5]);
		break;

	case 3:	/* r,m - 2op */
		*ctxt->andptr++ = t[4];
		*ctxt->andptr++ = t[5];
		asmand(ctxt, &p->to, t[6]);
		break;

	case 4:	/* m,r - 2op */
		*ctxt->andptr++ = t[4];
		*ctxt->andptr++ = t[5];
		asmand(ctxt, &p->from, t[6]);
		break;

	case 5:	/* load full pointer, trash heap */
		if(t[4])
			*ctxt->andptr++ = t[4];
		switch(p->to.index) {
		default:
			goto bad;
		case D_DS:
			*ctxt->andptr++ = 0xc5;
			break;
		case D_SS:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = 0xb2;
			break;
		case D_ES:
			*ctxt->andptr++ = 0xc4;
			break;
		case D_FS:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = 0xb4;
			break;
		case D_GS:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = 0xb5;
			break;
		}
		asmand(ctxt, &p->from, reg[p->to.type]);
		break;

	case 6:	/* double shift */
		z = p->from.type;
		switch(z) {
		default:
			goto bad;
		case D_CONST:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = t[4];
			asmand(ctxt, &p->to, reg[p->from.index]);
			*ctxt->andptr++ = p->from.offset;
			break;
		case D_CL:
		case D_CX:
			*ctxt->andptr++ = 0x0f;
			*ctxt->andptr++ = t[5];
			asmand(ctxt, &p->to, reg[p->from.index]);
			break;
		}
		break;

	case 7: /* imul rm,r */
		if(t[4] == Pq) {
			*ctxt->andptr++ = Pe;
			*ctxt->andptr++ = Pm;
		} else
			*ctxt->andptr++ = t[4];
		*ctxt->andptr++ = t[5];
		asmand(ctxt, &p->from, reg[p->to.type]);
		break;
	
	case 8: /* mov tls, r */
		// NOTE: The systems listed here are the ones that use the "TLS initial exec" model,
		// where you load the TLS base register into a register and then index off that
		// register to access the actual TLS variables. Systems that allow direct TLS access
		// are handled in prefixof above and should not be listed here.
		switch(ctxt->headtype) {
		default:
			sysfatal("unknown TLS base location for %s", headstr(ctxt->headtype));

		case Hlinux:
		case Hnacl:
			// ELF TLS base is 0(GS).
			pp.from = p->from;
			pp.from.type = D_INDIR+D_GS;
			pp.from.offset = 0;
			pp.from.index = D_NONE;
			pp.from.scale = 0;
			*ctxt->andptr++ = 0x65; // GS
			*ctxt->andptr++ = 0x8B;
			asmand(ctxt, &pp.from, reg[p->to.type]);
			break;
		
		case Hplan9:
			if(ctxt->plan9privates == nil)
				ctxt->plan9privates = linklookup(ctxt, "_privates", 0);
			memset(&pp.from, 0, sizeof pp.from);
			pp.from.type = D_EXTERN;
			pp.from.sym = ctxt->plan9privates;
			pp.from.offset = 0;
			pp.from.index = D_NONE;
			*ctxt->andptr++ = 0x8B;
			asmand(ctxt, &pp.from, reg[p->to.type]);
			break;

		case Hwindows:
			// Windows TLS base is always 0x14(FS).
			pp.from = p->from;
			pp.from.type = D_INDIR+D_FS;
			pp.from.offset = 0x14;
			pp.from.index = D_NONE;
			pp.from.scale = 0;
			*ctxt->andptr++ = 0x64; // FS
			*ctxt->andptr++ = 0x8B;
			asmand(ctxt, &pp.from, reg[p->to.type]);
			break;
		}
		break;
	}
}

static uchar naclret[] = {
	0x5d, // POPL BP
	// 0x8b, 0x7d, 0x00, // MOVL (BP), DI - catch return to invalid address, for debugging
	0x83, 0xe5, 0xe0,	// ANDL $~31, BP
	0xff, 0xe5, // JMP BP
};

static void
asmins(Link *ctxt, Prog *p)
{
	Reloc *r;

	ctxt->andptr = ctxt->and;
	
	if(p->as == AUSEFIELD) {
		r = addrel(ctxt->cursym);
		r->off = 0;
		r->sym = p->from.sym;
		r->type = R_USEFIELD;
		r->siz = 0;
		return;
	}

	if(ctxt->headtype == Hnacl) {
		switch(p->as) {
		case ARET:
			memmove(ctxt->andptr, naclret, sizeof naclret);
			ctxt->andptr += sizeof naclret;
			return;
		case ACALL:
		case AJMP:
			if(D_AX <= p->to.type && p->to.type <= D_DI) {
				*ctxt->andptr++ = 0x83;
				*ctxt->andptr++ = 0xe0 | (p->to.type - D_AX);
				*ctxt->andptr++ = 0xe0;
			}
			break;
		case AINT:
			*ctxt->andptr++ = 0xf4;
			return;
		}
	}

	doasm(ctxt, p);
	if(ctxt->andptr > ctxt->and+sizeof ctxt->and) {
		print("and[] is too short - %ld byte instruction\n", ctxt->andptr - ctxt->and);
		sysfatal("bad code");
	}
}
