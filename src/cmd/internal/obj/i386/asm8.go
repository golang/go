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

package i386

import (
	"cmd/internal/obj"
	"fmt"
	"log"
	"strings"
)

// Instruction layout.

const (
	MaxAlign  = 32
	FuncAlign = 16
)

type Optab struct {
	as     int16
	ytab   []byte
	prefix uint8
	op     [13]uint8
}

var opindex [ALAST + 1]*Optab

const (
	Yxxx = 0 + iota
	Ynone
	Yi0
	Yi1
	Yi8
	Yi32
	Yiauto
	Yal
	Ycl
	Yax
	Ycx
	Yrb
	Yrl
	Yrf
	Yf0
	Yrx
	Ymb
	Yml
	Ym
	Ybr
	Ycol
	Ytextsize
	Ytls
	Ycs
	Yss
	Yds
	Yes
	Yfs
	Ygs
	Ygdtr
	Yidtr
	Yldtr
	Ymsw
	Ytask
	Ycr0
	Ycr1
	Ycr2
	Ycr3
	Ycr4
	Ycr5
	Ycr6
	Ycr7
	Ydr0
	Ydr1
	Ydr2
	Ydr3
	Ydr4
	Ydr5
	Ydr6
	Ydr7
	Ytr0
	Ytr1
	Ytr2
	Ytr3
	Ytr4
	Ytr5
	Ytr6
	Ytr7
	Ymr
	Ymm
	Yxr
	Yxm
	Ymax
	Zxxx = 0 + iota - 63
	Zlit
	Zlitm_r
	Z_rp
	Zbr
	Zcall
	Zcallcon
	Zcallind
	Zcallindreg
	Zib_
	Zib_rp
	Zibo_m
	Zil_
	Zil_rp
	Zilo_m
	Zjmp
	Zjmpcon
	Zloop
	Zm_o
	Zm_r
	Zm2_r
	Zm_r_xm
	Zm_r_i_xm
	Zaut_r
	Zo_m
	Zpseudo
	Zr_m
	Zr_m_xm
	Zr_m_i_xm
	Zrp_
	Z_ib
	Z_il
	Zm_ibo
	Zm_ilo
	Zib_rr
	Zil_rr
	Zclr
	Zibm_r
	Zbyte
	Zmov
	Zmax
	Px  = 0
	Pe  = 0x66
	Pm  = 0x0f
	Pq  = 0xff
	Pb  = 0xfe
	Pf2 = 0xf2
	Pf3 = 0xf3
)

var ycover [Ymax * Ymax]uint8

var reg [MAXREG]int

var ynone = []uint8{
	Ynone,
	Ynone,
	Zlit,
	1,
	0,
}

var ytext = []uint8{
	Ymb,
	Ytextsize,
	Zpseudo,
	1,
	0,
}

var ynop = []uint8{
	Ynone,
	Ynone,
	Zpseudo,
	0,
	Ynone,
	Yiauto,
	Zpseudo,
	0,
	Ynone,
	Yml,
	Zpseudo,
	0,
	Ynone,
	Yrf,
	Zpseudo,
	0,
	Yiauto,
	Ynone,
	Zpseudo,
	0,
	Ynone,
	Yxr,
	Zpseudo,
	0,
	Yml,
	Ynone,
	Zpseudo,
	0,
	Yrf,
	Ynone,
	Zpseudo,
	0,
	Yxr,
	Ynone,
	Zpseudo,
	1,
	0,
}

var yfuncdata = []uint8{
	Yi32,
	Ym,
	Zpseudo,
	0,
	0,
}

var ypcdata = []uint8{
	Yi32,
	Yi32,
	Zpseudo,
	0,
	0,
}

var yxorb = []uint8{
	Yi32,
	Yal,
	Zib_,
	1,
	Yi32,
	Ymb,
	Zibo_m,
	2,
	Yrb,
	Ymb,
	Zr_m,
	1,
	Ymb,
	Yrb,
	Zm_r,
	1,
	0,
}

var yxorl = []uint8{
	Yi8,
	Yml,
	Zibo_m,
	2,
	Yi32,
	Yax,
	Zil_,
	1,
	Yi32,
	Yml,
	Zilo_m,
	2,
	Yrl,
	Yml,
	Zr_m,
	1,
	Yml,
	Yrl,
	Zm_r,
	1,
	0,
}

var yaddl = []uint8{
	Yi8,
	Yml,
	Zibo_m,
	2,
	Yi32,
	Yax,
	Zil_,
	1,
	Yi32,
	Yml,
	Zilo_m,
	2,
	Yrl,
	Yml,
	Zr_m,
	1,
	Yml,
	Yrl,
	Zm_r,
	1,
	0,
}

var yincb = []uint8{
	Ynone,
	Ymb,
	Zo_m,
	2,
	0,
}

var yincl = []uint8{
	Ynone,
	Yrl,
	Z_rp,
	1,
	Ynone,
	Yml,
	Zo_m,
	2,
	0,
}

var ycmpb = []uint8{
	Yal,
	Yi32,
	Z_ib,
	1,
	Ymb,
	Yi32,
	Zm_ibo,
	2,
	Ymb,
	Yrb,
	Zm_r,
	1,
	Yrb,
	Ymb,
	Zr_m,
	1,
	0,
}

var ycmpl = []uint8{
	Yml,
	Yi8,
	Zm_ibo,
	2,
	Yax,
	Yi32,
	Z_il,
	1,
	Yml,
	Yi32,
	Zm_ilo,
	2,
	Yml,
	Yrl,
	Zm_r,
	1,
	Yrl,
	Yml,
	Zr_m,
	1,
	0,
}

var yshb = []uint8{
	Yi1,
	Ymb,
	Zo_m,
	2,
	Yi32,
	Ymb,
	Zibo_m,
	2,
	Ycx,
	Ymb,
	Zo_m,
	2,
	0,
}

var yshl = []uint8{
	Yi1,
	Yml,
	Zo_m,
	2,
	Yi32,
	Yml,
	Zibo_m,
	2,
	Ycl,
	Yml,
	Zo_m,
	2,
	Ycx,
	Yml,
	Zo_m,
	2,
	0,
}

var ytestb = []uint8{
	Yi32,
	Yal,
	Zib_,
	1,
	Yi32,
	Ymb,
	Zibo_m,
	2,
	Yrb,
	Ymb,
	Zr_m,
	1,
	Ymb,
	Yrb,
	Zm_r,
	1,
	0,
}

var ytestl = []uint8{
	Yi32,
	Yax,
	Zil_,
	1,
	Yi32,
	Yml,
	Zilo_m,
	2,
	Yrl,
	Yml,
	Zr_m,
	1,
	Yml,
	Yrl,
	Zm_r,
	1,
	0,
}

var ymovb = []uint8{
	Yrb,
	Ymb,
	Zr_m,
	1,
	Ymb,
	Yrb,
	Zm_r,
	1,
	Yi32,
	Yrb,
	Zib_rp,
	1,
	Yi32,
	Ymb,
	Zibo_m,
	2,
	0,
}

var ymovw = []uint8{
	Yrl,
	Yml,
	Zr_m,
	1,
	Yml,
	Yrl,
	Zm_r,
	1,
	Yi0,
	Yrl,
	Zclr,
	1 + 2,
	//	Yi0,	Yml,	Zibo_m,	2,	// shorter but slower AND $0,dst
	Yi32,
	Yrl,
	Zil_rp,
	1,
	Yi32,
	Yml,
	Zilo_m,
	2,
	Yiauto,
	Yrl,
	Zaut_r,
	1,
	0,
}

var ymovl = []uint8{
	Yrl,
	Yml,
	Zr_m,
	1,
	Yml,
	Yrl,
	Zm_r,
	1,
	Yi0,
	Yrl,
	Zclr,
	1 + 2,
	//	Yi0,	Yml,	Zibo_m,	2,	// shorter but slower AND $0,dst
	Yi32,
	Yrl,
	Zil_rp,
	1,
	Yi32,
	Yml,
	Zilo_m,
	2,
	Yml,
	Yxr,
	Zm_r_xm,
	2, // XMM MOVD (32 bit)
	Yxr,
	Yml,
	Zr_m_xm,
	2, // XMM MOVD (32 bit)
	Yiauto,
	Yrl,
	Zaut_r,
	1,
	0,
}

var ymovq = []uint8{
	Yml,
	Yxr,
	Zm_r_xm,
	2,
	0,
}

var ym_rl = []uint8{
	Ym,
	Yrl,
	Zm_r,
	1,
	0,
}

var yrl_m = []uint8{
	Yrl,
	Ym,
	Zr_m,
	1,
	0,
}

var ymb_rl = []uint8{
	Ymb,
	Yrl,
	Zm_r,
	1,
	0,
}

var yml_rl = []uint8{
	Yml,
	Yrl,
	Zm_r,
	1,
	0,
}

var yrb_mb = []uint8{
	Yrb,
	Ymb,
	Zr_m,
	1,
	0,
}

var yrl_ml = []uint8{
	Yrl,
	Yml,
	Zr_m,
	1,
	0,
}

var yml_mb = []uint8{
	Yrb,
	Ymb,
	Zr_m,
	1,
	Ymb,
	Yrb,
	Zm_r,
	1,
	0,
}

var yxchg = []uint8{
	Yax,
	Yrl,
	Z_rp,
	1,
	Yrl,
	Yax,
	Zrp_,
	1,
	Yrl,
	Yml,
	Zr_m,
	1,
	Yml,
	Yrl,
	Zm_r,
	1,
	0,
}

var ydivl = []uint8{
	Yml,
	Ynone,
	Zm_o,
	2,
	0,
}

var ydivb = []uint8{
	Ymb,
	Ynone,
	Zm_o,
	2,
	0,
}

var yimul = []uint8{
	Yml,
	Ynone,
	Zm_o,
	2,
	Yi8,
	Yrl,
	Zib_rr,
	1,
	Yi32,
	Yrl,
	Zil_rr,
	1,
	0,
}

var ybyte = []uint8{
	Yi32,
	Ynone,
	Zbyte,
	1,
	0,
}

var yin = []uint8{
	Yi32,
	Ynone,
	Zib_,
	1,
	Ynone,
	Ynone,
	Zlit,
	1,
	0,
}

var yint = []uint8{
	Yi32,
	Ynone,
	Zib_,
	1,
	0,
}

var ypushl = []uint8{
	Yrl,
	Ynone,
	Zrp_,
	1,
	Ym,
	Ynone,
	Zm_o,
	2,
	Yi8,
	Ynone,
	Zib_,
	1,
	Yi32,
	Ynone,
	Zil_,
	1,
	0,
}

var ypopl = []uint8{
	Ynone,
	Yrl,
	Z_rp,
	1,
	Ynone,
	Ym,
	Zo_m,
	2,
	0,
}

var ybswap = []uint8{
	Ynone,
	Yrl,
	Z_rp,
	1,
	0,
}

var yscond = []uint8{
	Ynone,
	Ymb,
	Zo_m,
	2,
	0,
}

var yjcond = []uint8{
	Ynone,
	Ybr,
	Zbr,
	0,
	Yi0,
	Ybr,
	Zbr,
	0,
	Yi1,
	Ybr,
	Zbr,
	1,
	0,
}

var yloop = []uint8{
	Ynone,
	Ybr,
	Zloop,
	1,
	0,
}

var ycall = []uint8{
	Ynone,
	Yml,
	Zcallindreg,
	0,
	Yrx,
	Yrx,
	Zcallindreg,
	2,
	Ynone,
	Ycol,
	Zcallind,
	2,
	Ynone,
	Ybr,
	Zcall,
	0,
	Ynone,
	Yi32,
	Zcallcon,
	1,
	0,
}

var yduff = []uint8{
	Ynone,
	Yi32,
	Zcall,
	1,
	0,
}

var yjmp = []uint8{
	Ynone,
	Yml,
	Zo_m,
	2,
	Ynone,
	Ybr,
	Zjmp,
	0,
	Ynone,
	Yi32,
	Zjmpcon,
	1,
	0,
}

var yfmvd = []uint8{
	Ym,
	Yf0,
	Zm_o,
	2,
	Yf0,
	Ym,
	Zo_m,
	2,
	Yrf,
	Yf0,
	Zm_o,
	2,
	Yf0,
	Yrf,
	Zo_m,
	2,
	0,
}

var yfmvdp = []uint8{
	Yf0,
	Ym,
	Zo_m,
	2,
	Yf0,
	Yrf,
	Zo_m,
	2,
	0,
}

var yfmvf = []uint8{
	Ym,
	Yf0,
	Zm_o,
	2,
	Yf0,
	Ym,
	Zo_m,
	2,
	0,
}

var yfmvx = []uint8{
	Ym,
	Yf0,
	Zm_o,
	2,
	0,
}

var yfmvp = []uint8{
	Yf0,
	Ym,
	Zo_m,
	2,
	0,
}

var yfcmv = []uint8{
	Yrf,
	Yf0,
	Zm_o,
	2,
	0,
}

var yfadd = []uint8{
	Ym,
	Yf0,
	Zm_o,
	2,
	Yrf,
	Yf0,
	Zm_o,
	2,
	Yf0,
	Yrf,
	Zo_m,
	2,
	0,
}

var yfaddp = []uint8{
	Yf0,
	Yrf,
	Zo_m,
	2,
	0,
}

var yfxch = []uint8{
	Yf0,
	Yrf,
	Zo_m,
	2,
	Yrf,
	Yf0,
	Zm_o,
	2,
	0,
}

var ycompp = []uint8{
	Yf0,
	Yrf,
	Zo_m,
	2, /* botch is really f0,f1 */
	0,
}

var ystsw = []uint8{
	Ynone,
	Ym,
	Zo_m,
	2,
	Ynone,
	Yax,
	Zlit,
	1,
	0,
}

var ystcw = []uint8{
	Ynone,
	Ym,
	Zo_m,
	2,
	Ym,
	Ynone,
	Zm_o,
	2,
	0,
}

var ysvrs = []uint8{
	Ynone,
	Ym,
	Zo_m,
	2,
	Ym,
	Ynone,
	Zm_o,
	2,
	0,
}

var ymskb = []uint8{
	Yxr,
	Yrl,
	Zm_r_xm,
	2,
	Ymr,
	Yrl,
	Zm_r_xm,
	1,
	0,
}

var yxm = []uint8{
	Yxm,
	Yxr,
	Zm_r_xm,
	1,
	0,
}

var yxcvm1 = []uint8{
	Yxm,
	Yxr,
	Zm_r_xm,
	2,
	Yxm,
	Ymr,
	Zm_r_xm,
	2,
	0,
}

var yxcvm2 = []uint8{
	Yxm,
	Yxr,
	Zm_r_xm,
	2,
	Ymm,
	Yxr,
	Zm_r_xm,
	2,
	0,
}

var yxmq = []uint8{
	Yxm,
	Yxr,
	Zm_r_xm,
	2,
	0,
}

var yxr = []uint8{
	Yxr,
	Yxr,
	Zm_r_xm,
	1,
	0,
}

var yxr_ml = []uint8{
	Yxr,
	Yml,
	Zr_m_xm,
	1,
	0,
}

var yxcmp = []uint8{
	Yxm,
	Yxr,
	Zm_r_xm,
	1,
	0,
}

var yxcmpi = []uint8{
	Yxm,
	Yxr,
	Zm_r_i_xm,
	2,
	0,
}

var yxmov = []uint8{
	Yxm,
	Yxr,
	Zm_r_xm,
	1,
	Yxr,
	Yxm,
	Zr_m_xm,
	1,
	0,
}

var yxcvfl = []uint8{
	Yxm,
	Yrl,
	Zm_r_xm,
	1,
	0,
}

var yxcvlf = []uint8{
	Yml,
	Yxr,
	Zm_r_xm,
	1,
	0,
}

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
var yxrrl = []uint8{
	Yxr,
	Yrl,
	Zm_r,
	1,
	0,
}

var yprefetch = []uint8{
	Ym,
	Ynone,
	Zm_o,
	2,
	0,
}

var yaes = []uint8{
	Yxm,
	Yxr,
	Zlitm_r,
	2,
	0,
}

var yinsrd = []uint8{
	Yml,
	Yxr,
	Zibm_r,
	2,
	0,
}

var ymshufb = []uint8{
	Yxm,
	Yxr,
	Zm2_r,
	2,
	0,
}

var yxshuf = []uint8{
	Yxm,
	Yxr,
	Zibm_r,
	2,
	0,
}

var optab = /*	as, ytab, andproto, opcode */
[]Optab{
	Optab{obj.AXXX, nil, 0, [13]uint8{}},
	Optab{AAAA, ynone, Px, [13]uint8{0x37}},
	Optab{AAAD, ynone, Px, [13]uint8{0xd5, 0x0a}},
	Optab{AAAM, ynone, Px, [13]uint8{0xd4, 0x0a}},
	Optab{AAAS, ynone, Px, [13]uint8{0x3f}},
	Optab{AADCB, yxorb, Pb, [13]uint8{0x14, 0x80, 02, 0x10, 0x10}},
	Optab{AADCL, yxorl, Px, [13]uint8{0x83, 02, 0x15, 0x81, 02, 0x11, 0x13}},
	Optab{AADCW, yxorl, Pe, [13]uint8{0x83, 02, 0x15, 0x81, 02, 0x11, 0x13}},
	Optab{AADDB, yxorb, Px, [13]uint8{0x04, 0x80, 00, 0x00, 0x02}},
	Optab{AADDL, yaddl, Px, [13]uint8{0x83, 00, 0x05, 0x81, 00, 0x01, 0x03}},
	Optab{AADDW, yaddl, Pe, [13]uint8{0x83, 00, 0x05, 0x81, 00, 0x01, 0x03}},
	Optab{AADJSP, nil, 0, [13]uint8{}},
	Optab{AANDB, yxorb, Pb, [13]uint8{0x24, 0x80, 04, 0x20, 0x22}},
	Optab{AANDL, yxorl, Px, [13]uint8{0x83, 04, 0x25, 0x81, 04, 0x21, 0x23}},
	Optab{AANDW, yxorl, Pe, [13]uint8{0x83, 04, 0x25, 0x81, 04, 0x21, 0x23}},
	Optab{AARPL, yrl_ml, Px, [13]uint8{0x63}},
	Optab{ABOUNDL, yrl_m, Px, [13]uint8{0x62}},
	Optab{ABOUNDW, yrl_m, Pe, [13]uint8{0x62}},
	Optab{ABSFL, yml_rl, Pm, [13]uint8{0xbc}},
	Optab{ABSFW, yml_rl, Pq, [13]uint8{0xbc}},
	Optab{ABSRL, yml_rl, Pm, [13]uint8{0xbd}},
	Optab{ABSRW, yml_rl, Pq, [13]uint8{0xbd}},
	Optab{ABTL, yml_rl, Pm, [13]uint8{0xa3}},
	Optab{ABTW, yml_rl, Pq, [13]uint8{0xa3}},
	Optab{ABTCL, yml_rl, Pm, [13]uint8{0xbb}},
	Optab{ABTCW, yml_rl, Pq, [13]uint8{0xbb}},
	Optab{ABTRL, yml_rl, Pm, [13]uint8{0xb3}},
	Optab{ABTRW, yml_rl, Pq, [13]uint8{0xb3}},
	Optab{ABTSL, yml_rl, Pm, [13]uint8{0xab}},
	Optab{ABTSW, yml_rl, Pq, [13]uint8{0xab}},
	Optab{ABYTE, ybyte, Px, [13]uint8{1}},
	Optab{obj.ACALL, ycall, Px, [13]uint8{0xff, 02, 0xff, 0x15, 0xe8}},
	Optab{ACLC, ynone, Px, [13]uint8{0xf8}},
	Optab{ACLD, ynone, Px, [13]uint8{0xfc}},
	Optab{ACLI, ynone, Px, [13]uint8{0xfa}},
	Optab{ACLTS, ynone, Pm, [13]uint8{0x06}},
	Optab{ACMC, ynone, Px, [13]uint8{0xf5}},
	Optab{ACMPB, ycmpb, Pb, [13]uint8{0x3c, 0x80, 07, 0x38, 0x3a}},
	Optab{ACMPL, ycmpl, Px, [13]uint8{0x83, 07, 0x3d, 0x81, 07, 0x39, 0x3b}},
	Optab{ACMPW, ycmpl, Pe, [13]uint8{0x83, 07, 0x3d, 0x81, 07, 0x39, 0x3b}},
	Optab{ACMPSB, ynone, Pb, [13]uint8{0xa6}},
	Optab{ACMPSL, ynone, Px, [13]uint8{0xa7}},
	Optab{ACMPSW, ynone, Pe, [13]uint8{0xa7}},
	Optab{ADAA, ynone, Px, [13]uint8{0x27}},
	Optab{ADAS, ynone, Px, [13]uint8{0x2f}},
	Optab{obj.ADATA, nil, 0, [13]uint8{}},
	Optab{ADECB, yincb, Pb, [13]uint8{0xfe, 01}},
	Optab{ADECL, yincl, Px, [13]uint8{0x48, 0xff, 01}},
	Optab{ADECW, yincl, Pe, [13]uint8{0x48, 0xff, 01}},
	Optab{ADIVB, ydivb, Pb, [13]uint8{0xf6, 06}},
	Optab{ADIVL, ydivl, Px, [13]uint8{0xf7, 06}},
	Optab{ADIVW, ydivl, Pe, [13]uint8{0xf7, 06}},
	Optab{AENTER, nil, 0, [13]uint8{}}, /* botch */
	Optab{obj.AGLOBL, nil, 0, [13]uint8{}},
	Optab{AHLT, ynone, Px, [13]uint8{0xf4}},
	Optab{AIDIVB, ydivb, Pb, [13]uint8{0xf6, 07}},
	Optab{AIDIVL, ydivl, Px, [13]uint8{0xf7, 07}},
	Optab{AIDIVW, ydivl, Pe, [13]uint8{0xf7, 07}},
	Optab{AIMULB, ydivb, Pb, [13]uint8{0xf6, 05}},
	Optab{AIMULL, yimul, Px, [13]uint8{0xf7, 05, 0x6b, 0x69}},
	Optab{AIMULW, yimul, Pe, [13]uint8{0xf7, 05, 0x6b, 0x69}},
	Optab{AINB, yin, Pb, [13]uint8{0xe4, 0xec}},
	Optab{AINL, yin, Px, [13]uint8{0xe5, 0xed}},
	Optab{AINW, yin, Pe, [13]uint8{0xe5, 0xed}},
	Optab{AINCB, yincb, Pb, [13]uint8{0xfe, 00}},
	Optab{AINCL, yincl, Px, [13]uint8{0x40, 0xff, 00}},
	Optab{AINCW, yincl, Pe, [13]uint8{0x40, 0xff, 00}},
	Optab{AINSB, ynone, Pb, [13]uint8{0x6c}},
	Optab{AINSL, ynone, Px, [13]uint8{0x6d}},
	Optab{AINSW, ynone, Pe, [13]uint8{0x6d}},
	Optab{AINT, yint, Px, [13]uint8{0xcd}},
	Optab{AINTO, ynone, Px, [13]uint8{0xce}},
	Optab{AIRETL, ynone, Px, [13]uint8{0xcf}},
	Optab{AIRETW, ynone, Pe, [13]uint8{0xcf}},
	Optab{AJCC, yjcond, Px, [13]uint8{0x73, 0x83, 00}},
	Optab{AJCS, yjcond, Px, [13]uint8{0x72, 0x82}},
	Optab{AJCXZL, yloop, Px, [13]uint8{0xe3}},
	Optab{AJCXZW, yloop, Px, [13]uint8{0xe3}},
	Optab{AJEQ, yjcond, Px, [13]uint8{0x74, 0x84}},
	Optab{AJGE, yjcond, Px, [13]uint8{0x7d, 0x8d}},
	Optab{AJGT, yjcond, Px, [13]uint8{0x7f, 0x8f}},
	Optab{AJHI, yjcond, Px, [13]uint8{0x77, 0x87}},
	Optab{AJLE, yjcond, Px, [13]uint8{0x7e, 0x8e}},
	Optab{AJLS, yjcond, Px, [13]uint8{0x76, 0x86}},
	Optab{AJLT, yjcond, Px, [13]uint8{0x7c, 0x8c}},
	Optab{AJMI, yjcond, Px, [13]uint8{0x78, 0x88}},
	Optab{obj.AJMP, yjmp, Px, [13]uint8{0xff, 04, 0xeb, 0xe9}},
	Optab{AJNE, yjcond, Px, [13]uint8{0x75, 0x85}},
	Optab{AJOC, yjcond, Px, [13]uint8{0x71, 0x81, 00}},
	Optab{AJOS, yjcond, Px, [13]uint8{0x70, 0x80, 00}},
	Optab{AJPC, yjcond, Px, [13]uint8{0x7b, 0x8b}},
	Optab{AJPL, yjcond, Px, [13]uint8{0x79, 0x89}},
	Optab{AJPS, yjcond, Px, [13]uint8{0x7a, 0x8a}},
	Optab{ALAHF, ynone, Px, [13]uint8{0x9f}},
	Optab{ALARL, yml_rl, Pm, [13]uint8{0x02}},
	Optab{ALARW, yml_rl, Pq, [13]uint8{0x02}},
	Optab{ALEAL, ym_rl, Px, [13]uint8{0x8d}},
	Optab{ALEAW, ym_rl, Pe, [13]uint8{0x8d}},
	Optab{ALEAVEL, ynone, Px, [13]uint8{0xc9}},
	Optab{ALEAVEW, ynone, Pe, [13]uint8{0xc9}},
	Optab{ALOCK, ynone, Px, [13]uint8{0xf0}},
	Optab{ALODSB, ynone, Pb, [13]uint8{0xac}},
	Optab{ALODSL, ynone, Px, [13]uint8{0xad}},
	Optab{ALODSW, ynone, Pe, [13]uint8{0xad}},
	Optab{ALONG, ybyte, Px, [13]uint8{4}},
	Optab{ALOOP, yloop, Px, [13]uint8{0xe2}},
	Optab{ALOOPEQ, yloop, Px, [13]uint8{0xe1}},
	Optab{ALOOPNE, yloop, Px, [13]uint8{0xe0}},
	Optab{ALSLL, yml_rl, Pm, [13]uint8{0x03}},
	Optab{ALSLW, yml_rl, Pq, [13]uint8{0x03}},
	Optab{AMOVB, ymovb, Pb, [13]uint8{0x88, 0x8a, 0xb0, 0xc6, 00}},
	Optab{AMOVL, ymovl, Px, [13]uint8{0x89, 0x8b, 0x31, 0x83, 04, 0xb8, 0xc7, 00, Pe, 0x6e, Pe, 0x7e, 0}},
	Optab{AMOVW, ymovw, Pe, [13]uint8{0x89, 0x8b, 0x31, 0x83, 04, 0xb8, 0xc7, 00, 0}},
	Optab{AMOVQ, ymovq, Pf3, [13]uint8{0x7e}},
	Optab{AMOVBLSX, ymb_rl, Pm, [13]uint8{0xbe}},
	Optab{AMOVBLZX, ymb_rl, Pm, [13]uint8{0xb6}},
	Optab{AMOVBWSX, ymb_rl, Pq, [13]uint8{0xbe}},
	Optab{AMOVBWZX, ymb_rl, Pq, [13]uint8{0xb6}},
	Optab{AMOVWLSX, yml_rl, Pm, [13]uint8{0xbf}},
	Optab{AMOVWLZX, yml_rl, Pm, [13]uint8{0xb7}},
	Optab{AMOVSB, ynone, Pb, [13]uint8{0xa4}},
	Optab{AMOVSL, ynone, Px, [13]uint8{0xa5}},
	Optab{AMOVSW, ynone, Pe, [13]uint8{0xa5}},
	Optab{AMULB, ydivb, Pb, [13]uint8{0xf6, 04}},
	Optab{AMULL, ydivl, Px, [13]uint8{0xf7, 04}},
	Optab{AMULW, ydivl, Pe, [13]uint8{0xf7, 04}},
	Optab{ANEGB, yscond, Px, [13]uint8{0xf6, 03}},
	Optab{ANEGL, yscond, Px, [13]uint8{0xf7, 03}}, // TODO(rsc): yscond is wrong here.
	Optab{ANEGW, yscond, Pe, [13]uint8{0xf7, 03}}, // TODO(rsc): yscond is wrong here.
	Optab{obj.ANOP, ynop, Px, [13]uint8{0, 0}},
	Optab{ANOTB, yscond, Px, [13]uint8{0xf6, 02}},
	Optab{ANOTL, yscond, Px, [13]uint8{0xf7, 02}}, // TODO(rsc): yscond is wrong here.
	Optab{ANOTW, yscond, Pe, [13]uint8{0xf7, 02}}, // TODO(rsc): yscond is wrong here.
	Optab{AORB, yxorb, Pb, [13]uint8{0x0c, 0x80, 01, 0x08, 0x0a}},
	Optab{AORL, yxorl, Px, [13]uint8{0x83, 01, 0x0d, 0x81, 01, 0x09, 0x0b}},
	Optab{AORW, yxorl, Pe, [13]uint8{0x83, 01, 0x0d, 0x81, 01, 0x09, 0x0b}},
	Optab{AOUTB, yin, Pb, [13]uint8{0xe6, 0xee}},
	Optab{AOUTL, yin, Px, [13]uint8{0xe7, 0xef}},
	Optab{AOUTW, yin, Pe, [13]uint8{0xe7, 0xef}},
	Optab{AOUTSB, ynone, Pb, [13]uint8{0x6e}},
	Optab{AOUTSL, ynone, Px, [13]uint8{0x6f}},
	Optab{AOUTSW, ynone, Pe, [13]uint8{0x6f}},
	Optab{APAUSE, ynone, Px, [13]uint8{0xf3, 0x90}},
	Optab{APOPAL, ynone, Px, [13]uint8{0x61}},
	Optab{APOPAW, ynone, Pe, [13]uint8{0x61}},
	Optab{APOPFL, ynone, Px, [13]uint8{0x9d}},
	Optab{APOPFW, ynone, Pe, [13]uint8{0x9d}},
	Optab{APOPL, ypopl, Px, [13]uint8{0x58, 0x8f, 00}},
	Optab{APOPW, ypopl, Pe, [13]uint8{0x58, 0x8f, 00}},
	Optab{APUSHAL, ynone, Px, [13]uint8{0x60}},
	Optab{APUSHAW, ynone, Pe, [13]uint8{0x60}},
	Optab{APUSHFL, ynone, Px, [13]uint8{0x9c}},
	Optab{APUSHFW, ynone, Pe, [13]uint8{0x9c}},
	Optab{APUSHL, ypushl, Px, [13]uint8{0x50, 0xff, 06, 0x6a, 0x68}},
	Optab{APUSHW, ypushl, Pe, [13]uint8{0x50, 0xff, 06, 0x6a, 0x68}},
	Optab{ARCLB, yshb, Pb, [13]uint8{0xd0, 02, 0xc0, 02, 0xd2, 02}},
	Optab{ARCLL, yshl, Px, [13]uint8{0xd1, 02, 0xc1, 02, 0xd3, 02, 0xd3, 02}},
	Optab{ARCLW, yshl, Pe, [13]uint8{0xd1, 02, 0xc1, 02, 0xd3, 02, 0xd3, 02}},
	Optab{ARCRB, yshb, Pb, [13]uint8{0xd0, 03, 0xc0, 03, 0xd2, 03}},
	Optab{ARCRL, yshl, Px, [13]uint8{0xd1, 03, 0xc1, 03, 0xd3, 03, 0xd3, 03}},
	Optab{ARCRW, yshl, Pe, [13]uint8{0xd1, 03, 0xc1, 03, 0xd3, 03, 0xd3, 03}},
	Optab{AREP, ynone, Px, [13]uint8{0xf3}},
	Optab{AREPN, ynone, Px, [13]uint8{0xf2}},
	Optab{obj.ARET, ynone, Px, [13]uint8{0xc3}},
	Optab{AROLB, yshb, Pb, [13]uint8{0xd0, 00, 0xc0, 00, 0xd2, 00}},
	Optab{AROLL, yshl, Px, [13]uint8{0xd1, 00, 0xc1, 00, 0xd3, 00, 0xd3, 00}},
	Optab{AROLW, yshl, Pe, [13]uint8{0xd1, 00, 0xc1, 00, 0xd3, 00, 0xd3, 00}},
	Optab{ARORB, yshb, Pb, [13]uint8{0xd0, 01, 0xc0, 01, 0xd2, 01}},
	Optab{ARORL, yshl, Px, [13]uint8{0xd1, 01, 0xc1, 01, 0xd3, 01, 0xd3, 01}},
	Optab{ARORW, yshl, Pe, [13]uint8{0xd1, 01, 0xc1, 01, 0xd3, 01, 0xd3, 01}},
	Optab{ASAHF, ynone, Px, [13]uint8{0x9e}},
	Optab{ASALB, yshb, Pb, [13]uint8{0xd0, 04, 0xc0, 04, 0xd2, 04}},
	Optab{ASALL, yshl, Px, [13]uint8{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	Optab{ASALW, yshl, Pe, [13]uint8{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	Optab{ASARB, yshb, Pb, [13]uint8{0xd0, 07, 0xc0, 07, 0xd2, 07}},
	Optab{ASARL, yshl, Px, [13]uint8{0xd1, 07, 0xc1, 07, 0xd3, 07, 0xd3, 07}},
	Optab{ASARW, yshl, Pe, [13]uint8{0xd1, 07, 0xc1, 07, 0xd3, 07, 0xd3, 07}},
	Optab{ASBBB, yxorb, Pb, [13]uint8{0x1c, 0x80, 03, 0x18, 0x1a}},
	Optab{ASBBL, yxorl, Px, [13]uint8{0x83, 03, 0x1d, 0x81, 03, 0x19, 0x1b}},
	Optab{ASBBW, yxorl, Pe, [13]uint8{0x83, 03, 0x1d, 0x81, 03, 0x19, 0x1b}},
	Optab{ASCASB, ynone, Pb, [13]uint8{0xae}},
	Optab{ASCASL, ynone, Px, [13]uint8{0xaf}},
	Optab{ASCASW, ynone, Pe, [13]uint8{0xaf}},
	Optab{ASETCC, yscond, Pm, [13]uint8{0x93, 00}},
	Optab{ASETCS, yscond, Pm, [13]uint8{0x92, 00}},
	Optab{ASETEQ, yscond, Pm, [13]uint8{0x94, 00}},
	Optab{ASETGE, yscond, Pm, [13]uint8{0x9d, 00}},
	Optab{ASETGT, yscond, Pm, [13]uint8{0x9f, 00}},
	Optab{ASETHI, yscond, Pm, [13]uint8{0x97, 00}},
	Optab{ASETLE, yscond, Pm, [13]uint8{0x9e, 00}},
	Optab{ASETLS, yscond, Pm, [13]uint8{0x96, 00}},
	Optab{ASETLT, yscond, Pm, [13]uint8{0x9c, 00}},
	Optab{ASETMI, yscond, Pm, [13]uint8{0x98, 00}},
	Optab{ASETNE, yscond, Pm, [13]uint8{0x95, 00}},
	Optab{ASETOC, yscond, Pm, [13]uint8{0x91, 00}},
	Optab{ASETOS, yscond, Pm, [13]uint8{0x90, 00}},
	Optab{ASETPC, yscond, Pm, [13]uint8{0x9b, 00}},
	Optab{ASETPL, yscond, Pm, [13]uint8{0x99, 00}},
	Optab{ASETPS, yscond, Pm, [13]uint8{0x9a, 00}},
	Optab{ACDQ, ynone, Px, [13]uint8{0x99}},
	Optab{ACWD, ynone, Pe, [13]uint8{0x99}},
	Optab{ASHLB, yshb, Pb, [13]uint8{0xd0, 04, 0xc0, 04, 0xd2, 04}},
	Optab{ASHLL, yshl, Px, [13]uint8{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	Optab{ASHLW, yshl, Pe, [13]uint8{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	Optab{ASHRB, yshb, Pb, [13]uint8{0xd0, 05, 0xc0, 05, 0xd2, 05}},
	Optab{ASHRL, yshl, Px, [13]uint8{0xd1, 05, 0xc1, 05, 0xd3, 05, 0xd3, 05}},
	Optab{ASHRW, yshl, Pe, [13]uint8{0xd1, 05, 0xc1, 05, 0xd3, 05, 0xd3, 05}},
	Optab{ASTC, ynone, Px, [13]uint8{0xf9}},
	Optab{ASTD, ynone, Px, [13]uint8{0xfd}},
	Optab{ASTI, ynone, Px, [13]uint8{0xfb}},
	Optab{ASTOSB, ynone, Pb, [13]uint8{0xaa}},
	Optab{ASTOSL, ynone, Px, [13]uint8{0xab}},
	Optab{ASTOSW, ynone, Pe, [13]uint8{0xab}},
	Optab{ASUBB, yxorb, Pb, [13]uint8{0x2c, 0x80, 05, 0x28, 0x2a}},
	Optab{ASUBL, yaddl, Px, [13]uint8{0x83, 05, 0x2d, 0x81, 05, 0x29, 0x2b}},
	Optab{ASUBW, yaddl, Pe, [13]uint8{0x83, 05, 0x2d, 0x81, 05, 0x29, 0x2b}},
	Optab{ASYSCALL, ynone, Px, [13]uint8{0xcd, 100}},
	Optab{ATESTB, ytestb, Pb, [13]uint8{0xa8, 0xf6, 00, 0x84, 0x84}},
	Optab{ATESTL, ytestl, Px, [13]uint8{0xa9, 0xf7, 00, 0x85, 0x85}},
	Optab{ATESTW, ytestl, Pe, [13]uint8{0xa9, 0xf7, 00, 0x85, 0x85}},
	Optab{obj.ATEXT, ytext, Px, [13]uint8{}},
	Optab{AVERR, ydivl, Pm, [13]uint8{0x00, 04}},
	Optab{AVERW, ydivl, Pm, [13]uint8{0x00, 05}},
	Optab{AWAIT, ynone, Px, [13]uint8{0x9b}},
	Optab{AWORD, ybyte, Px, [13]uint8{2}},
	Optab{AXCHGB, yml_mb, Pb, [13]uint8{0x86, 0x86}},
	Optab{AXCHGL, yxchg, Px, [13]uint8{0x90, 0x90, 0x87, 0x87}},
	Optab{AXCHGW, yxchg, Pe, [13]uint8{0x90, 0x90, 0x87, 0x87}},
	Optab{AXLAT, ynone, Px, [13]uint8{0xd7}},
	Optab{AXORB, yxorb, Pb, [13]uint8{0x34, 0x80, 06, 0x30, 0x32}},
	Optab{AXORL, yxorl, Px, [13]uint8{0x83, 06, 0x35, 0x81, 06, 0x31, 0x33}},
	Optab{AXORW, yxorl, Pe, [13]uint8{0x83, 06, 0x35, 0x81, 06, 0x31, 0x33}},
	Optab{AFMOVB, yfmvx, Px, [13]uint8{0xdf, 04}},
	Optab{AFMOVBP, yfmvp, Px, [13]uint8{0xdf, 06}},
	Optab{AFMOVD, yfmvd, Px, [13]uint8{0xdd, 00, 0xdd, 02, 0xd9, 00, 0xdd, 02}},
	Optab{AFMOVDP, yfmvdp, Px, [13]uint8{0xdd, 03, 0xdd, 03}},
	Optab{AFMOVF, yfmvf, Px, [13]uint8{0xd9, 00, 0xd9, 02}},
	Optab{AFMOVFP, yfmvp, Px, [13]uint8{0xd9, 03}},
	Optab{AFMOVL, yfmvf, Px, [13]uint8{0xdb, 00, 0xdb, 02}},
	Optab{AFMOVLP, yfmvp, Px, [13]uint8{0xdb, 03}},
	Optab{AFMOVV, yfmvx, Px, [13]uint8{0xdf, 05}},
	Optab{AFMOVVP, yfmvp, Px, [13]uint8{0xdf, 07}},
	Optab{AFMOVW, yfmvf, Px, [13]uint8{0xdf, 00, 0xdf, 02}},
	Optab{AFMOVWP, yfmvp, Px, [13]uint8{0xdf, 03}},
	Optab{AFMOVX, yfmvx, Px, [13]uint8{0xdb, 05}},
	Optab{AFMOVXP, yfmvp, Px, [13]uint8{0xdb, 07}},
	Optab{AFCOMB, nil, 0, [13]uint8{}},
	Optab{AFCOMBP, nil, 0, [13]uint8{}},
	Optab{AFCOMD, yfadd, Px, [13]uint8{0xdc, 02, 0xd8, 02, 0xdc, 02}},  /* botch */
	Optab{AFCOMDP, yfadd, Px, [13]uint8{0xdc, 03, 0xd8, 03, 0xdc, 03}}, /* botch */
	Optab{AFCOMDPP, ycompp, Px, [13]uint8{0xde, 03}},
	Optab{AFCOMF, yfmvx, Px, [13]uint8{0xd8, 02}},
	Optab{AFCOMFP, yfmvx, Px, [13]uint8{0xd8, 03}},
	Optab{AFCOMI, yfmvx, Px, [13]uint8{0xdb, 06}},
	Optab{AFCOMIP, yfmvx, Px, [13]uint8{0xdf, 06}},
	Optab{AFCOML, yfmvx, Px, [13]uint8{0xda, 02}},
	Optab{AFCOMLP, yfmvx, Px, [13]uint8{0xda, 03}},
	Optab{AFCOMW, yfmvx, Px, [13]uint8{0xde, 02}},
	Optab{AFCOMWP, yfmvx, Px, [13]uint8{0xde, 03}},
	Optab{AFUCOM, ycompp, Px, [13]uint8{0xdd, 04}},
	Optab{AFUCOMI, ycompp, Px, [13]uint8{0xdb, 05}},
	Optab{AFUCOMIP, ycompp, Px, [13]uint8{0xdf, 05}},
	Optab{AFUCOMP, ycompp, Px, [13]uint8{0xdd, 05}},
	Optab{AFUCOMPP, ycompp, Px, [13]uint8{0xda, 13}},
	Optab{AFADDDP, yfaddp, Px, [13]uint8{0xde, 00}},
	Optab{AFADDW, yfmvx, Px, [13]uint8{0xde, 00}},
	Optab{AFADDL, yfmvx, Px, [13]uint8{0xda, 00}},
	Optab{AFADDF, yfmvx, Px, [13]uint8{0xd8, 00}},
	Optab{AFADDD, yfadd, Px, [13]uint8{0xdc, 00, 0xd8, 00, 0xdc, 00}},
	Optab{AFMULDP, yfaddp, Px, [13]uint8{0xde, 01}},
	Optab{AFMULW, yfmvx, Px, [13]uint8{0xde, 01}},
	Optab{AFMULL, yfmvx, Px, [13]uint8{0xda, 01}},
	Optab{AFMULF, yfmvx, Px, [13]uint8{0xd8, 01}},
	Optab{AFMULD, yfadd, Px, [13]uint8{0xdc, 01, 0xd8, 01, 0xdc, 01}},
	Optab{AFSUBDP, yfaddp, Px, [13]uint8{0xde, 05}},
	Optab{AFSUBW, yfmvx, Px, [13]uint8{0xde, 04}},
	Optab{AFSUBL, yfmvx, Px, [13]uint8{0xda, 04}},
	Optab{AFSUBF, yfmvx, Px, [13]uint8{0xd8, 04}},
	Optab{AFSUBD, yfadd, Px, [13]uint8{0xdc, 04, 0xd8, 04, 0xdc, 05}},
	Optab{AFSUBRDP, yfaddp, Px, [13]uint8{0xde, 04}},
	Optab{AFSUBRW, yfmvx, Px, [13]uint8{0xde, 05}},
	Optab{AFSUBRL, yfmvx, Px, [13]uint8{0xda, 05}},
	Optab{AFSUBRF, yfmvx, Px, [13]uint8{0xd8, 05}},
	Optab{AFSUBRD, yfadd, Px, [13]uint8{0xdc, 05, 0xd8, 05, 0xdc, 04}},
	Optab{AFDIVDP, yfaddp, Px, [13]uint8{0xde, 07}},
	Optab{AFDIVW, yfmvx, Px, [13]uint8{0xde, 06}},
	Optab{AFDIVL, yfmvx, Px, [13]uint8{0xda, 06}},
	Optab{AFDIVF, yfmvx, Px, [13]uint8{0xd8, 06}},
	Optab{AFDIVD, yfadd, Px, [13]uint8{0xdc, 06, 0xd8, 06, 0xdc, 07}},
	Optab{AFDIVRDP, yfaddp, Px, [13]uint8{0xde, 06}},
	Optab{AFDIVRW, yfmvx, Px, [13]uint8{0xde, 07}},
	Optab{AFDIVRL, yfmvx, Px, [13]uint8{0xda, 07}},
	Optab{AFDIVRF, yfmvx, Px, [13]uint8{0xd8, 07}},
	Optab{AFDIVRD, yfadd, Px, [13]uint8{0xdc, 07, 0xd8, 07, 0xdc, 06}},
	Optab{AFXCHD, yfxch, Px, [13]uint8{0xd9, 01, 0xd9, 01}},
	Optab{AFFREE, nil, 0, [13]uint8{}},
	Optab{AFLDCW, ystcw, Px, [13]uint8{0xd9, 05, 0xd9, 05}},
	Optab{AFLDENV, ystcw, Px, [13]uint8{0xd9, 04, 0xd9, 04}},
	Optab{AFRSTOR, ysvrs, Px, [13]uint8{0xdd, 04, 0xdd, 04}},
	Optab{AFSAVE, ysvrs, Px, [13]uint8{0xdd, 06, 0xdd, 06}},
	Optab{AFSTCW, ystcw, Px, [13]uint8{0xd9, 07, 0xd9, 07}},
	Optab{AFSTENV, ystcw, Px, [13]uint8{0xd9, 06, 0xd9, 06}},
	Optab{AFSTSW, ystsw, Px, [13]uint8{0xdd, 07, 0xdf, 0xe0}},
	Optab{AF2XM1, ynone, Px, [13]uint8{0xd9, 0xf0}},
	Optab{AFABS, ynone, Px, [13]uint8{0xd9, 0xe1}},
	Optab{AFCHS, ynone, Px, [13]uint8{0xd9, 0xe0}},
	Optab{AFCLEX, ynone, Px, [13]uint8{0xdb, 0xe2}},
	Optab{AFCOS, ynone, Px, [13]uint8{0xd9, 0xff}},
	Optab{AFDECSTP, ynone, Px, [13]uint8{0xd9, 0xf6}},
	Optab{AFINCSTP, ynone, Px, [13]uint8{0xd9, 0xf7}},
	Optab{AFINIT, ynone, Px, [13]uint8{0xdb, 0xe3}},
	Optab{AFLD1, ynone, Px, [13]uint8{0xd9, 0xe8}},
	Optab{AFLDL2E, ynone, Px, [13]uint8{0xd9, 0xea}},
	Optab{AFLDL2T, ynone, Px, [13]uint8{0xd9, 0xe9}},
	Optab{AFLDLG2, ynone, Px, [13]uint8{0xd9, 0xec}},
	Optab{AFLDLN2, ynone, Px, [13]uint8{0xd9, 0xed}},
	Optab{AFLDPI, ynone, Px, [13]uint8{0xd9, 0xeb}},
	Optab{AFLDZ, ynone, Px, [13]uint8{0xd9, 0xee}},
	Optab{AFNOP, ynone, Px, [13]uint8{0xd9, 0xd0}},
	Optab{AFPATAN, ynone, Px, [13]uint8{0xd9, 0xf3}},
	Optab{AFPREM, ynone, Px, [13]uint8{0xd9, 0xf8}},
	Optab{AFPREM1, ynone, Px, [13]uint8{0xd9, 0xf5}},
	Optab{AFPTAN, ynone, Px, [13]uint8{0xd9, 0xf2}},
	Optab{AFRNDINT, ynone, Px, [13]uint8{0xd9, 0xfc}},
	Optab{AFSCALE, ynone, Px, [13]uint8{0xd9, 0xfd}},
	Optab{AFSIN, ynone, Px, [13]uint8{0xd9, 0xfe}},
	Optab{AFSINCOS, ynone, Px, [13]uint8{0xd9, 0xfb}},
	Optab{AFSQRT, ynone, Px, [13]uint8{0xd9, 0xfa}},
	Optab{AFTST, ynone, Px, [13]uint8{0xd9, 0xe4}},
	Optab{AFXAM, ynone, Px, [13]uint8{0xd9, 0xe5}},
	Optab{AFXTRACT, ynone, Px, [13]uint8{0xd9, 0xf4}},
	Optab{AFYL2X, ynone, Px, [13]uint8{0xd9, 0xf1}},
	Optab{AFYL2XP1, ynone, Px, [13]uint8{0xd9, 0xf9}},
	Optab{obj.AEND, nil, 0, [13]uint8{}},
	Optab{ACMPXCHGB, yrb_mb, Pm, [13]uint8{0xb0}},
	Optab{ACMPXCHGL, yrl_ml, Pm, [13]uint8{0xb1}},
	Optab{ACMPXCHGW, yrl_ml, Pm, [13]uint8{0xb1}},
	Optab{ACMPXCHG8B, yscond, Pm, [13]uint8{0xc7, 01}}, // TODO(rsc): yscond is wrong here.

	Optab{ACPUID, ynone, Pm, [13]uint8{0xa2}},
	Optab{ARDTSC, ynone, Pm, [13]uint8{0x31}},
	Optab{AXADDB, yrb_mb, Pb, [13]uint8{0x0f, 0xc0}},
	Optab{AXADDL, yrl_ml, Pm, [13]uint8{0xc1}},
	Optab{AXADDW, yrl_ml, Pe, [13]uint8{0x0f, 0xc1}},
	Optab{ACMOVLCC, yml_rl, Pm, [13]uint8{0x43}},
	Optab{ACMOVLCS, yml_rl, Pm, [13]uint8{0x42}},
	Optab{ACMOVLEQ, yml_rl, Pm, [13]uint8{0x44}},
	Optab{ACMOVLGE, yml_rl, Pm, [13]uint8{0x4d}},
	Optab{ACMOVLGT, yml_rl, Pm, [13]uint8{0x4f}},
	Optab{ACMOVLHI, yml_rl, Pm, [13]uint8{0x47}},
	Optab{ACMOVLLE, yml_rl, Pm, [13]uint8{0x4e}},
	Optab{ACMOVLLS, yml_rl, Pm, [13]uint8{0x46}},
	Optab{ACMOVLLT, yml_rl, Pm, [13]uint8{0x4c}},
	Optab{ACMOVLMI, yml_rl, Pm, [13]uint8{0x48}},
	Optab{ACMOVLNE, yml_rl, Pm, [13]uint8{0x45}},
	Optab{ACMOVLOC, yml_rl, Pm, [13]uint8{0x41}},
	Optab{ACMOVLOS, yml_rl, Pm, [13]uint8{0x40}},
	Optab{ACMOVLPC, yml_rl, Pm, [13]uint8{0x4b}},
	Optab{ACMOVLPL, yml_rl, Pm, [13]uint8{0x49}},
	Optab{ACMOVLPS, yml_rl, Pm, [13]uint8{0x4a}},
	Optab{ACMOVWCC, yml_rl, Pq, [13]uint8{0x43}},
	Optab{ACMOVWCS, yml_rl, Pq, [13]uint8{0x42}},
	Optab{ACMOVWEQ, yml_rl, Pq, [13]uint8{0x44}},
	Optab{ACMOVWGE, yml_rl, Pq, [13]uint8{0x4d}},
	Optab{ACMOVWGT, yml_rl, Pq, [13]uint8{0x4f}},
	Optab{ACMOVWHI, yml_rl, Pq, [13]uint8{0x47}},
	Optab{ACMOVWLE, yml_rl, Pq, [13]uint8{0x4e}},
	Optab{ACMOVWLS, yml_rl, Pq, [13]uint8{0x46}},
	Optab{ACMOVWLT, yml_rl, Pq, [13]uint8{0x4c}},
	Optab{ACMOVWMI, yml_rl, Pq, [13]uint8{0x48}},
	Optab{ACMOVWNE, yml_rl, Pq, [13]uint8{0x45}},
	Optab{ACMOVWOC, yml_rl, Pq, [13]uint8{0x41}},
	Optab{ACMOVWOS, yml_rl, Pq, [13]uint8{0x40}},
	Optab{ACMOVWPC, yml_rl, Pq, [13]uint8{0x4b}},
	Optab{ACMOVWPL, yml_rl, Pq, [13]uint8{0x49}},
	Optab{ACMOVWPS, yml_rl, Pq, [13]uint8{0x4a}},
	Optab{AFCMOVCC, yfcmv, Px, [13]uint8{0xdb, 00}},
	Optab{AFCMOVCS, yfcmv, Px, [13]uint8{0xda, 00}},
	Optab{AFCMOVEQ, yfcmv, Px, [13]uint8{0xda, 01}},
	Optab{AFCMOVHI, yfcmv, Px, [13]uint8{0xdb, 02}},
	Optab{AFCMOVLS, yfcmv, Px, [13]uint8{0xda, 02}},
	Optab{AFCMOVNE, yfcmv, Px, [13]uint8{0xdb, 01}},
	Optab{AFCMOVNU, yfcmv, Px, [13]uint8{0xdb, 03}},
	Optab{AFCMOVUN, yfcmv, Px, [13]uint8{0xda, 03}},
	Optab{ALFENCE, ynone, Pm, [13]uint8{0xae, 0xe8}},
	Optab{AMFENCE, ynone, Pm, [13]uint8{0xae, 0xf0}},
	Optab{ASFENCE, ynone, Pm, [13]uint8{0xae, 0xf8}},
	Optab{AEMMS, ynone, Pm, [13]uint8{0x77}},
	Optab{APREFETCHT0, yprefetch, Pm, [13]uint8{0x18, 01}},
	Optab{APREFETCHT1, yprefetch, Pm, [13]uint8{0x18, 02}},
	Optab{APREFETCHT2, yprefetch, Pm, [13]uint8{0x18, 03}},
	Optab{APREFETCHNTA, yprefetch, Pm, [13]uint8{0x18, 00}},
	Optab{ABSWAPL, ybswap, Pm, [13]uint8{0xc8}},
	Optab{obj.AUNDEF, ynone, Px, [13]uint8{0x0f, 0x0b}},
	Optab{AADDPD, yxm, Pq, [13]uint8{0x58}},
	Optab{AADDPS, yxm, Pm, [13]uint8{0x58}},
	Optab{AADDSD, yxm, Pf2, [13]uint8{0x58}},
	Optab{AADDSS, yxm, Pf3, [13]uint8{0x58}},
	Optab{AANDNPD, yxm, Pq, [13]uint8{0x55}},
	Optab{AANDNPS, yxm, Pm, [13]uint8{0x55}},
	Optab{AANDPD, yxm, Pq, [13]uint8{0x54}},
	Optab{AANDPS, yxm, Pq, [13]uint8{0x54}},
	Optab{ACMPPD, yxcmpi, Px, [13]uint8{Pe, 0xc2}},
	Optab{ACMPPS, yxcmpi, Pm, [13]uint8{0xc2, 0}},
	Optab{ACMPSD, yxcmpi, Px, [13]uint8{Pf2, 0xc2}},
	Optab{ACMPSS, yxcmpi, Px, [13]uint8{Pf3, 0xc2}},
	Optab{ACOMISD, yxcmp, Pe, [13]uint8{0x2f}},
	Optab{ACOMISS, yxcmp, Pm, [13]uint8{0x2f}},
	Optab{ACVTPL2PD, yxcvm2, Px, [13]uint8{Pf3, 0xe6, Pe, 0x2a}},
	Optab{ACVTPL2PS, yxcvm2, Pm, [13]uint8{0x5b, 0, 0x2a, 0}},
	Optab{ACVTPD2PL, yxcvm1, Px, [13]uint8{Pf2, 0xe6, Pe, 0x2d}},
	Optab{ACVTPD2PS, yxm, Pe, [13]uint8{0x5a}},
	Optab{ACVTPS2PL, yxcvm1, Px, [13]uint8{Pe, 0x5b, Pm, 0x2d}},
	Optab{ACVTPS2PD, yxm, Pm, [13]uint8{0x5a}},
	Optab{ACVTSD2SL, yxcvfl, Pf2, [13]uint8{0x2d}},
	Optab{ACVTSD2SS, yxm, Pf2, [13]uint8{0x5a}},
	Optab{ACVTSL2SD, yxcvlf, Pf2, [13]uint8{0x2a}},
	Optab{ACVTSL2SS, yxcvlf, Pf3, [13]uint8{0x2a}},
	Optab{ACVTSS2SD, yxm, Pf3, [13]uint8{0x5a}},
	Optab{ACVTSS2SL, yxcvfl, Pf3, [13]uint8{0x2d}},
	Optab{ACVTTPD2PL, yxcvm1, Px, [13]uint8{Pe, 0xe6, Pe, 0x2c}},
	Optab{ACVTTPS2PL, yxcvm1, Px, [13]uint8{Pf3, 0x5b, Pm, 0x2c}},
	Optab{ACVTTSD2SL, yxcvfl, Pf2, [13]uint8{0x2c}},
	Optab{ACVTTSS2SL, yxcvfl, Pf3, [13]uint8{0x2c}},
	Optab{ADIVPD, yxm, Pe, [13]uint8{0x5e}},
	Optab{ADIVPS, yxm, Pm, [13]uint8{0x5e}},
	Optab{ADIVSD, yxm, Pf2, [13]uint8{0x5e}},
	Optab{ADIVSS, yxm, Pf3, [13]uint8{0x5e}},
	Optab{AMASKMOVOU, yxr, Pe, [13]uint8{0xf7}},
	Optab{AMAXPD, yxm, Pe, [13]uint8{0x5f}},
	Optab{AMAXPS, yxm, Pm, [13]uint8{0x5f}},
	Optab{AMAXSD, yxm, Pf2, [13]uint8{0x5f}},
	Optab{AMAXSS, yxm, Pf3, [13]uint8{0x5f}},
	Optab{AMINPD, yxm, Pe, [13]uint8{0x5d}},
	Optab{AMINPS, yxm, Pm, [13]uint8{0x5d}},
	Optab{AMINSD, yxm, Pf2, [13]uint8{0x5d}},
	Optab{AMINSS, yxm, Pf3, [13]uint8{0x5d}},
	Optab{AMOVAPD, yxmov, Pe, [13]uint8{0x28, 0x29}},
	Optab{AMOVAPS, yxmov, Pm, [13]uint8{0x28, 0x29}},
	Optab{AMOVO, yxmov, Pe, [13]uint8{0x6f, 0x7f}},
	Optab{AMOVOU, yxmov, Pf3, [13]uint8{0x6f, 0x7f}},
	Optab{AMOVHLPS, yxr, Pm, [13]uint8{0x12}},
	Optab{AMOVHPD, yxmov, Pe, [13]uint8{0x16, 0x17}},
	Optab{AMOVHPS, yxmov, Pm, [13]uint8{0x16, 0x17}},
	Optab{AMOVLHPS, yxr, Pm, [13]uint8{0x16}},
	Optab{AMOVLPD, yxmov, Pe, [13]uint8{0x12, 0x13}},
	Optab{AMOVLPS, yxmov, Pm, [13]uint8{0x12, 0x13}},
	Optab{AMOVMSKPD, yxrrl, Pq, [13]uint8{0x50}},
	Optab{AMOVMSKPS, yxrrl, Pm, [13]uint8{0x50}},
	Optab{AMOVNTO, yxr_ml, Pe, [13]uint8{0xe7}},
	Optab{AMOVNTPD, yxr_ml, Pe, [13]uint8{0x2b}},
	Optab{AMOVNTPS, yxr_ml, Pm, [13]uint8{0x2b}},
	Optab{AMOVSD, yxmov, Pf2, [13]uint8{0x10, 0x11}},
	Optab{AMOVSS, yxmov, Pf3, [13]uint8{0x10, 0x11}},
	Optab{AMOVUPD, yxmov, Pe, [13]uint8{0x10, 0x11}},
	Optab{AMOVUPS, yxmov, Pm, [13]uint8{0x10, 0x11}},
	Optab{AMULPD, yxm, Pe, [13]uint8{0x59}},
	Optab{AMULPS, yxm, Ym, [13]uint8{0x59}},
	Optab{AMULSD, yxm, Pf2, [13]uint8{0x59}},
	Optab{AMULSS, yxm, Pf3, [13]uint8{0x59}},
	Optab{AORPD, yxm, Pq, [13]uint8{0x56}},
	Optab{AORPS, yxm, Pm, [13]uint8{0x56}},
	Optab{APADDQ, yxm, Pe, [13]uint8{0xd4}},
	Optab{APAND, yxm, Pe, [13]uint8{0xdb}},
	Optab{APCMPEQB, yxmq, Pe, [13]uint8{0x74}},
	Optab{APMAXSW, yxm, Pe, [13]uint8{0xee}},
	Optab{APMAXUB, yxm, Pe, [13]uint8{0xde}},
	Optab{APMINSW, yxm, Pe, [13]uint8{0xea}},
	Optab{APMINUB, yxm, Pe, [13]uint8{0xda}},
	Optab{APMOVMSKB, ymskb, Px, [13]uint8{Pe, 0xd7, 0xd7}},
	Optab{APSADBW, yxm, Pq, [13]uint8{0xf6}},
	Optab{APSUBB, yxm, Pe, [13]uint8{0xf8}},
	Optab{APSUBL, yxm, Pe, [13]uint8{0xfa}},
	Optab{APSUBQ, yxm, Pe, [13]uint8{0xfb}},
	Optab{APSUBSB, yxm, Pe, [13]uint8{0xe8}},
	Optab{APSUBSW, yxm, Pe, [13]uint8{0xe9}},
	Optab{APSUBUSB, yxm, Pe, [13]uint8{0xd8}},
	Optab{APSUBUSW, yxm, Pe, [13]uint8{0xd9}},
	Optab{APSUBW, yxm, Pe, [13]uint8{0xf9}},
	Optab{APUNPCKHQDQ, yxm, Pe, [13]uint8{0x6d}},
	Optab{APUNPCKLQDQ, yxm, Pe, [13]uint8{0x6c}},
	Optab{APXOR, yxm, Pe, [13]uint8{0xef}},
	Optab{ARCPPS, yxm, Pm, [13]uint8{0x53}},
	Optab{ARCPSS, yxm, Pf3, [13]uint8{0x53}},
	Optab{ARSQRTPS, yxm, Pm, [13]uint8{0x52}},
	Optab{ARSQRTSS, yxm, Pf3, [13]uint8{0x52}},
	Optab{ASQRTPD, yxm, Pe, [13]uint8{0x51}},
	Optab{ASQRTPS, yxm, Pm, [13]uint8{0x51}},
	Optab{ASQRTSD, yxm, Pf2, [13]uint8{0x51}},
	Optab{ASQRTSS, yxm, Pf3, [13]uint8{0x51}},
	Optab{ASUBPD, yxm, Pe, [13]uint8{0x5c}},
	Optab{ASUBPS, yxm, Pm, [13]uint8{0x5c}},
	Optab{ASUBSD, yxm, Pf2, [13]uint8{0x5c}},
	Optab{ASUBSS, yxm, Pf3, [13]uint8{0x5c}},
	Optab{AUCOMISD, yxcmp, Pe, [13]uint8{0x2e}},
	Optab{AUCOMISS, yxcmp, Pm, [13]uint8{0x2e}},
	Optab{AUNPCKHPD, yxm, Pe, [13]uint8{0x15}},
	Optab{AUNPCKHPS, yxm, Pm, [13]uint8{0x15}},
	Optab{AUNPCKLPD, yxm, Pe, [13]uint8{0x14}},
	Optab{AUNPCKLPS, yxm, Pm, [13]uint8{0x14}},
	Optab{AXORPD, yxm, Pe, [13]uint8{0x57}},
	Optab{AXORPS, yxm, Pm, [13]uint8{0x57}},
	Optab{APSHUFHW, yxshuf, Pf3, [13]uint8{0x70, 00}},
	Optab{APSHUFL, yxshuf, Pq, [13]uint8{0x70, 00}},
	Optab{APSHUFLW, yxshuf, Pf2, [13]uint8{0x70, 00}},
	Optab{AAESENC, yaes, Pq, [13]uint8{0x38, 0xdc, 0}},
	Optab{APINSRD, yinsrd, Pq, [13]uint8{0x3a, 0x22, 00}},
	Optab{APSHUFB, ymshufb, Pq, [13]uint8{0x38, 0x00}},
	Optab{obj.AUSEFIELD, ynop, Px, [13]uint8{0, 0}},
	Optab{obj.ATYPE, nil, 0, [13]uint8{}},
	Optab{obj.AFUNCDATA, yfuncdata, Px, [13]uint8{0, 0}},
	Optab{obj.APCDATA, ypcdata, Px, [13]uint8{0, 0}},
	Optab{obj.ACHECKNIL, nil, 0, [13]uint8{}},
	Optab{obj.AVARDEF, nil, 0, [13]uint8{}},
	Optab{obj.AVARKILL, nil, 0, [13]uint8{}},
	Optab{obj.ADUFFCOPY, yduff, Px, [13]uint8{0xe8}},
	Optab{obj.ADUFFZERO, yduff, Px, [13]uint8{0xe8}},
	Optab{0, nil, 0, [13]uint8{}},
}

// single-instruction no-ops of various lengths.
// constructed by hand and disassembled with gdb to verify.
// see http://www.agner.org/optimize/optimizing_assembly.pdf for discussion.
var nop = [][16]uint8{
	[16]uint8{0x90},
	[16]uint8{0x66, 0x90},
	[16]uint8{0x0F, 0x1F, 0x00},
	[16]uint8{0x0F, 0x1F, 0x40, 0x00},
	[16]uint8{0x0F, 0x1F, 0x44, 0x00, 0x00},
	[16]uint8{0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00},
	[16]uint8{0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00},
	[16]uint8{0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
	[16]uint8{0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
}

// Native Client rejects the repeated 0x66 prefix.
// {0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
func fillnop(p []byte, n int) {
	var m int

	for n > 0 {
		m = n
		if m > len(nop) {
			m = len(nop)
		}
		copy(p[:m], nop[m-1][:m])
		p = p[m:]
		n -= m
	}
}

func naclpad(ctxt *obj.Link, s *obj.LSym, c int32, pad int32) int32 {
	obj.Symgrow(ctxt, s, int64(c)+int64(pad))
	fillnop(s.P[c:], int(pad))
	return c + pad
}

func span8(ctxt *obj.Link, s *obj.LSym) {
	var p *obj.Prog
	var q *obj.Prog
	var c int32
	var v int32
	var loop int32
	var bp []byte
	var n int
	var m int
	var i int

	ctxt.Cursym = s

	if s.Text == nil || s.Text.Link == nil {
		return
	}

	if ycover[0] == 0 {
		instinit()
	}

	for p = s.Text; p != nil; p = p.Link {
		if p.To.Type == obj.TYPE_BRANCH {
			if p.Pcond == nil {
				p.Pcond = p
			}
		}
		if p.As == AADJSP {
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_SP
			v = int32(-p.From.Offset)
			p.From.Offset = int64(v)
			p.As = AADDL
			if v < 0 {
				p.As = ASUBL
				v = -v
				p.From.Offset = int64(v)
			}

			if v == 0 {
				p.As = obj.ANOP
			}
		}
	}

	for p = s.Text; p != nil; p = p.Link {
		p.Back = 2 // use short branches first time through
		q = p.Pcond
		if q != nil && (q.Back&2 != 0) {
			p.Back |= 1 // backward jump
		}

		if p.As == AADJSP {
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_SP
			v = int32(-p.From.Offset)
			p.From.Offset = int64(v)
			p.As = AADDL
			if v < 0 {
				p.As = ASUBL
				v = -v
				p.From.Offset = int64(v)
			}

			if v == 0 {
				p.As = obj.ANOP
			}
		}
	}

	n = 0
	for {
		loop = 0
		for i = 0; i < len(s.R); i++ {
			s.R[i] = obj.Reloc{}
		}
		s.R = s.R[:0]
		s.P = s.P[:0]
		c = 0
		for p = s.Text; p != nil; p = p.Link {
			if ctxt.Headtype == obj.Hnacl && p.Isize > 0 {
				var deferreturn *obj.LSym

				if deferreturn == nil {
					deferreturn = obj.Linklookup(ctxt, "runtime.deferreturn", 0)
				}

				// pad everything to avoid crossing 32-byte boundary
				if c>>5 != (c+int32(p.Isize)-1)>>5 {
					c = naclpad(ctxt, s, c, -c&31)
				}

				// pad call deferreturn to start at 32-byte boundary
				// so that subtracting 5 in jmpdefer will jump back
				// to that boundary and rerun the call.
				if p.As == obj.ACALL && p.To.Sym == deferreturn {
					c = naclpad(ctxt, s, c, -c&31)
				}

				// pad call to end at 32-byte boundary
				if p.As == obj.ACALL {
					c = naclpad(ctxt, s, c, -(c+int32(p.Isize))&31)
				}

				// the linker treats REP and STOSQ as different instructions
				// but in fact the REP is a prefix on the STOSQ.
				// make sure REP has room for 2 more bytes, so that
				// padding will not be inserted before the next instruction.
				if p.As == AREP && c>>5 != (c+3-1)>>5 {
					c = naclpad(ctxt, s, c, -c&31)
				}

				// same for LOCK.
				// various instructions follow; the longest is 4 bytes.
				// give ourselves 8 bytes so as to avoid surprises.
				if p.As == ALOCK && c>>5 != (c+8-1)>>5 {
					c = naclpad(ctxt, s, c, -c&31)
				}
			}

			p.Pc = int64(c)

			// process forward jumps to p
			for q = p.Comefrom; q != nil; q = q.Forwd {
				v = int32(p.Pc - (q.Pc + int64(q.Mark)))
				if q.Back&2 != 0 { // short
					if v > 127 {
						loop++
						q.Back ^= 2
					}

					if q.As == AJCXZW {
						s.P[q.Pc+2] = byte(v)
					} else {
						s.P[q.Pc+1] = byte(v)
					}
				} else {
					bp = s.P[q.Pc+int64(q.Mark)-4:]
					bp[0] = byte(v)
					bp = bp[1:]
					bp[0] = byte(v >> 8)
					bp = bp[1:]
					bp[0] = byte(v >> 16)
					bp = bp[1:]
					bp[0] = byte(v >> 24)
				}
			}

			p.Comefrom = nil

			p.Pc = int64(c)
			asmins(ctxt, p)
			m = -cap(ctxt.Andptr) + cap(ctxt.And[:])
			if int(p.Isize) != m {
				p.Isize = uint8(m)
				loop++
			}

			obj.Symgrow(ctxt, s, p.Pc+int64(m))
			copy(s.P[p.Pc:][:m], ctxt.And[:m])
			p.Mark = uint16(m)
			c += int32(m)
		}

		n++
		if n > 20 {
			ctxt.Diag("span must be looping")
			log.Fatalf("bad code")
		}
		if loop == 0 {
			break
		}
	}

	if ctxt.Headtype == obj.Hnacl {
		c = naclpad(ctxt, s, c, -c&31)
	}
	c += -c & (FuncAlign - 1)
	s.Size = int64(c)

	if false { /* debug['a'] > 1 */
		fmt.Printf("span1 %s %d (%d tries)\n %.6x", s.Name, s.Size, n, 0)
		for i = 0; i < len(s.P); i++ {
			fmt.Printf(" %.2x", s.P[i])
			if i%16 == 15 {
				fmt.Printf("\n  %.6x", uint(i+1))
			}
		}

		if i%16 != 0 {
			fmt.Printf("\n")
		}

		for i = 0; i < len(s.R); i++ {
			var r *obj.Reloc

			r = &s.R[i]
			fmt.Printf(" rel %#.4x/%d %s%+d\n", uint32(r.Off), r.Siz, r.Sym.Name, r.Add)
		}
	}
}

func instinit() {
	var i int
	var c int

	for i = 1; optab[i].as != 0; i++ {
		c = int(optab[i].as)
		if opindex[c] != nil {
			log.Fatalf("phase error in optab: %d (%v)", i, Aconv(c))
		}
		opindex[c] = &optab[i]
	}

	for i = 0; i < Ymax; i++ {
		ycover[i*Ymax+i] = 1
	}

	ycover[Yi0*Ymax+Yi8] = 1
	ycover[Yi1*Ymax+Yi8] = 1

	ycover[Yi0*Ymax+Yi32] = 1
	ycover[Yi1*Ymax+Yi32] = 1
	ycover[Yi8*Ymax+Yi32] = 1

	ycover[Yal*Ymax+Yrb] = 1
	ycover[Ycl*Ymax+Yrb] = 1
	ycover[Yax*Ymax+Yrb] = 1
	ycover[Ycx*Ymax+Yrb] = 1
	ycover[Yrx*Ymax+Yrb] = 1

	ycover[Yax*Ymax+Yrx] = 1
	ycover[Ycx*Ymax+Yrx] = 1

	ycover[Yax*Ymax+Yrl] = 1
	ycover[Ycx*Ymax+Yrl] = 1
	ycover[Yrx*Ymax+Yrl] = 1

	ycover[Yf0*Ymax+Yrf] = 1

	ycover[Yal*Ymax+Ymb] = 1
	ycover[Ycl*Ymax+Ymb] = 1
	ycover[Yax*Ymax+Ymb] = 1
	ycover[Ycx*Ymax+Ymb] = 1
	ycover[Yrx*Ymax+Ymb] = 1
	ycover[Yrb*Ymax+Ymb] = 1
	ycover[Ym*Ymax+Ymb] = 1

	ycover[Yax*Ymax+Yml] = 1
	ycover[Ycx*Ymax+Yml] = 1
	ycover[Yrx*Ymax+Yml] = 1
	ycover[Yrl*Ymax+Yml] = 1
	ycover[Ym*Ymax+Yml] = 1

	ycover[Yax*Ymax+Ymm] = 1
	ycover[Ycx*Ymax+Ymm] = 1
	ycover[Yrx*Ymax+Ymm] = 1
	ycover[Yrl*Ymax+Ymm] = 1
	ycover[Ym*Ymax+Ymm] = 1
	ycover[Ymr*Ymax+Ymm] = 1

	ycover[Ym*Ymax+Yxm] = 1
	ycover[Yxr*Ymax+Yxm] = 1

	for i = 0; i < MAXREG; i++ {
		reg[i] = -1
		if i >= REG_AL && i <= REG_BH {
			reg[i] = (i - REG_AL) & 7
		}
		if i >= REG_AX && i <= REG_DI {
			reg[i] = (i - REG_AX) & 7
		}
		if i >= REG_F0 && i <= REG_F0+7 {
			reg[i] = (i - REG_F0) & 7
		}
		if i >= REG_X0 && i <= REG_X0+7 {
			reg[i] = (i - REG_X0) & 7
		}
	}
}

func prefixof(ctxt *obj.Link, a *obj.Addr) int {
	if a.Type == obj.TYPE_MEM && a.Name == obj.NAME_NONE {
		switch a.Reg {
		case REG_CS:
			return 0x2e

		case REG_DS:
			return 0x3e

		case REG_ES:
			return 0x26

		case REG_FS:
			return 0x64

		case REG_GS:
			return 0x65

			// NOTE: Systems listed here should be only systems that
		// support direct TLS references like 8(TLS) implemented as
		// direct references from FS or GS. Systems that require
		// the initial-exec model, where you load the TLS base into
		// a register and then index from that register, do not reach
		// this code and should not be listed.
		case REG_TLS:
			switch ctxt.Headtype {
			default:
				log.Fatalf("unknown TLS base register for %s", obj.Headstr(ctxt.Headtype))

			case obj.Hdarwin,
				obj.Hdragonfly,
				obj.Hfreebsd,
				obj.Hnetbsd,
				obj.Hopenbsd:
				return 0x65 // GS
			}
		}
	}

	return 0
}

func oclass(ctxt *obj.Link, p *obj.Prog, a *obj.Addr) int {
	var v int32

	// TODO(rsc): This special case is for SHRQ $3, AX:DX,
	// which encodes as SHRQ $32(DX*0), AX.
	// Similarly SHRQ CX, AX:DX is really SHRQ CX(DX*0), AX.
	// Change encoding and remove.
	if (a.Type == obj.TYPE_CONST || a.Type == obj.TYPE_REG) && a.Index != REG_NONE && a.Scale == 0 {
		return Ycol
	}

	switch a.Type {
	case obj.TYPE_NONE:
		return Ynone

	case obj.TYPE_BRANCH:
		return Ybr

		// TODO(rsc): Why this is also Ycol is a mystery. Should split the two meanings.
	case obj.TYPE_INDIR:
		if a.Name != obj.NAME_NONE && a.Reg == REG_NONE && a.Index == REG_NONE && a.Scale == 0 {
			return Ycol
		}
		return Yxxx

	case obj.TYPE_MEM:
		return Ym

	case obj.TYPE_ADDR:
		switch a.Name {
		case obj.NAME_EXTERN,
			obj.NAME_STATIC:
			return Yi32

		case obj.NAME_AUTO,
			obj.NAME_PARAM:
			return Yiauto
		}

		// DUFFZERO/DUFFCOPY encoding forgot to set a->index
		// and got Yi32 in an earlier version of this code.
		// Keep doing that until we fix yduff etc.
		if a.Sym != nil && strings.HasPrefix(a.Sym.Name, "runtime.duff") {
			return Yi32
		}

		if a.Sym != nil || a.Name != obj.NAME_NONE {
			ctxt.Diag("unexpected addr: %v", Dconv(p, 0, a))
		}
		fallthrough

		// fall through

	case obj.TYPE_CONST:
		if a.Sym != nil {
			ctxt.Diag("TYPE_CONST with symbol: %v", Dconv(p, 0, a))
		}

		v = int32(a.Offset)
		if v == 0 {
			return Yi0
		}
		if v == 1 {
			return Yi1
		}
		if v >= -128 && v <= 127 {
			return Yi8
		}
		return Yi32

	case obj.TYPE_TEXTSIZE:
		return Ytextsize
	}

	if a.Type != obj.TYPE_REG {
		ctxt.Diag("unexpected addr1: type=%d %v", a.Type, Dconv(p, 0, a))
		return Yxxx
	}

	switch a.Reg {
	case REG_AL:
		return Yal

	case REG_AX:
		return Yax

	case REG_CL,
		REG_DL,
		REG_BL,
		REG_AH,
		REG_CH,
		REG_DH,
		REG_BH:
		return Yrb

	case REG_CX:
		return Ycx

	case REG_DX,
		REG_BX:
		return Yrx

	case REG_SP,
		REG_BP,
		REG_SI,
		REG_DI:
		return Yrl

	case REG_F0 + 0:
		return Yf0

	case REG_F0 + 1,
		REG_F0 + 2,
		REG_F0 + 3,
		REG_F0 + 4,
		REG_F0 + 5,
		REG_F0 + 6,
		REG_F0 + 7:
		return Yrf

	case REG_X0 + 0,
		REG_X0 + 1,
		REG_X0 + 2,
		REG_X0 + 3,
		REG_X0 + 4,
		REG_X0 + 5,
		REG_X0 + 6,
		REG_X0 + 7:
		return Yxr

	case REG_CS:
		return Ycs
	case REG_SS:
		return Yss
	case REG_DS:
		return Yds
	case REG_ES:
		return Yes
	case REG_FS:
		return Yfs
	case REG_GS:
		return Ygs
	case REG_TLS:
		return Ytls

	case REG_GDTR:
		return Ygdtr
	case REG_IDTR:
		return Yidtr
	case REG_LDTR:
		return Yldtr
	case REG_MSW:
		return Ymsw
	case REG_TASK:
		return Ytask

	case REG_CR + 0:
		return Ycr0
	case REG_CR + 1:
		return Ycr1
	case REG_CR + 2:
		return Ycr2
	case REG_CR + 3:
		return Ycr3
	case REG_CR + 4:
		return Ycr4
	case REG_CR + 5:
		return Ycr5
	case REG_CR + 6:
		return Ycr6
	case REG_CR + 7:
		return Ycr7

	case REG_DR + 0:
		return Ydr0
	case REG_DR + 1:
		return Ydr1
	case REG_DR + 2:
		return Ydr2
	case REG_DR + 3:
		return Ydr3
	case REG_DR + 4:
		return Ydr4
	case REG_DR + 5:
		return Ydr5
	case REG_DR + 6:
		return Ydr6
	case REG_DR + 7:
		return Ydr7

	case REG_TR + 0:
		return Ytr0
	case REG_TR + 1:
		return Ytr1
	case REG_TR + 2:
		return Ytr2
	case REG_TR + 3:
		return Ytr3
	case REG_TR + 4:
		return Ytr4
	case REG_TR + 5:
		return Ytr5
	case REG_TR + 6:
		return Ytr6
	case REG_TR + 7:
		return Ytr7
	}

	return Yxxx
}

func asmidx(ctxt *obj.Link, scale int, index int, base int) {
	var i int

	switch index {
	default:
		goto bad

	case obj.TYPE_NONE:
		i = 4 << 3
		goto bas

	case REG_AX,
		REG_CX,
		REG_DX,
		REG_BX,
		REG_BP,
		REG_SI,
		REG_DI:
		i = reg[index] << 3
	}

	switch scale {
	default:
		goto bad

	case 1:
		break

	case 2:
		i |= 1 << 6

	case 4:
		i |= 2 << 6

	case 8:
		i |= 3 << 6
	}

bas:
	switch base {
	default:
		goto bad

	case REG_NONE: /* must be mod=00 */
		i |= 5

	case REG_AX,
		REG_CX,
		REG_DX,
		REG_BX,
		REG_SP,
		REG_BP,
		REG_SI,
		REG_DI:
		i |= reg[base]
	}

	ctxt.Andptr[0] = byte(i)
	ctxt.Andptr = ctxt.Andptr[1:]
	return

bad:
	ctxt.Diag("asmidx: bad address %d,%d,%d", scale, index, base)
	ctxt.Andptr[0] = 0
	ctxt.Andptr = ctxt.Andptr[1:]
	return
}

func put4(ctxt *obj.Link, v int32) {
	ctxt.Andptr[0] = byte(v)
	ctxt.Andptr[1] = byte(v >> 8)
	ctxt.Andptr[2] = byte(v >> 16)
	ctxt.Andptr[3] = byte(v >> 24)
	ctxt.Andptr = ctxt.Andptr[4:]
}

func relput4(ctxt *obj.Link, p *obj.Prog, a *obj.Addr) {
	var v int64
	var rel obj.Reloc
	var r *obj.Reloc

	v = int64(vaddr(ctxt, p, a, &rel))
	if rel.Siz != 0 {
		if rel.Siz != 4 {
			ctxt.Diag("bad reloc")
		}
		r = obj.Addrel(ctxt.Cursym)
		*r = rel
		r.Off = int32(p.Pc + int64(-cap(ctxt.Andptr)+cap(ctxt.And[:])))
	}

	put4(ctxt, int32(v))
}

func vaddr(ctxt *obj.Link, p *obj.Prog, a *obj.Addr, r *obj.Reloc) int32 {
	var s *obj.LSym

	if r != nil {
		*r = obj.Reloc{}
	}

	switch a.Name {
	case obj.NAME_STATIC,
		obj.NAME_EXTERN:
		s = a.Sym
		if s != nil {
			if r == nil {
				ctxt.Diag("need reloc for %v", Dconv(p, 0, a))
				log.Fatalf("bad code")
			}

			r.Type = obj.R_ADDR
			r.Siz = 4
			r.Off = -1
			r.Sym = s
			r.Add = a.Offset
			return 0
		}

		return int32(a.Offset)
	}

	if (a.Type == obj.TYPE_MEM || a.Type == obj.TYPE_ADDR) && a.Reg == REG_TLS {
		if r == nil {
			ctxt.Diag("need reloc for %v", Dconv(p, 0, a))
			log.Fatalf("bad code")
		}

		r.Type = obj.R_TLS_LE
		r.Siz = 4
		r.Off = -1 // caller must fill in
		r.Add = a.Offset
		return 0
	}

	return int32(a.Offset)
}

func asmand(ctxt *obj.Link, p *obj.Prog, a *obj.Addr, r int) {
	var v int32
	var base int
	var rel obj.Reloc

	v = int32(a.Offset)
	rel.Siz = 0

	switch a.Type {
	case obj.TYPE_ADDR:
		if a.Name == obj.NAME_NONE {
			ctxt.Diag("unexpected TYPE_ADDR with NAME_NONE")
		}
		if a.Index == REG_TLS {
			ctxt.Diag("unexpected TYPE_ADDR with index==REG_TLS")
		}
		goto bad

	case obj.TYPE_REG:
		if (a.Reg < REG_AL || REG_F7 < a.Reg) && (a.Reg < REG_X0 || REG_X0+7 < a.Reg) {
			goto bad
		}
		if v != 0 {
			goto bad
		}
		ctxt.Andptr[0] = byte(3<<6 | reg[a.Reg]<<0 | r<<3)
		ctxt.Andptr = ctxt.Andptr[1:]
		return
	}

	if a.Type != obj.TYPE_MEM {
		goto bad
	}

	if a.Index != REG_NONE && a.Index != REG_TLS {
		base = int(a.Reg)
		switch a.Name {
		case obj.NAME_EXTERN,
			obj.NAME_STATIC:
			base = REG_NONE
			v = vaddr(ctxt, p, a, &rel)

		case obj.NAME_AUTO,
			obj.NAME_PARAM:
			base = REG_SP
		}

		if base == REG_NONE {
			ctxt.Andptr[0] = byte(0<<6 | 4<<0 | r<<3)
			ctxt.Andptr = ctxt.Andptr[1:]
			asmidx(ctxt, int(a.Scale), int(a.Index), base)
			goto putrelv
		}

		if v == 0 && rel.Siz == 0 && base != REG_BP {
			ctxt.Andptr[0] = byte(0<<6 | 4<<0 | r<<3)
			ctxt.Andptr = ctxt.Andptr[1:]
			asmidx(ctxt, int(a.Scale), int(a.Index), base)
			return
		}

		if v >= -128 && v < 128 && rel.Siz == 0 {
			ctxt.Andptr[0] = byte(1<<6 | 4<<0 | r<<3)
			ctxt.Andptr = ctxt.Andptr[1:]
			asmidx(ctxt, int(a.Scale), int(a.Index), base)
			ctxt.Andptr[0] = byte(v)
			ctxt.Andptr = ctxt.Andptr[1:]
			return
		}

		ctxt.Andptr[0] = byte(2<<6 | 4<<0 | r<<3)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmidx(ctxt, int(a.Scale), int(a.Index), base)
		goto putrelv
	}

	base = int(a.Reg)
	switch a.Name {
	case obj.NAME_STATIC,
		obj.NAME_EXTERN:
		base = REG_NONE
		v = vaddr(ctxt, p, a, &rel)

	case obj.NAME_AUTO,
		obj.NAME_PARAM:
		base = REG_SP
	}

	if base == REG_TLS {
		v = vaddr(ctxt, p, a, &rel)
	}

	if base == REG_NONE || (REG_CS <= base && base <= REG_GS) || base == REG_TLS {
		ctxt.Andptr[0] = byte(0<<6 | 5<<0 | r<<3)
		ctxt.Andptr = ctxt.Andptr[1:]
		goto putrelv
	}

	if base == REG_SP {
		if v == 0 && rel.Siz == 0 {
			ctxt.Andptr[0] = byte(0<<6 | 4<<0 | r<<3)
			ctxt.Andptr = ctxt.Andptr[1:]
			asmidx(ctxt, int(a.Scale), REG_NONE, base)
			return
		}

		if v >= -128 && v < 128 && rel.Siz == 0 {
			ctxt.Andptr[0] = byte(1<<6 | 4<<0 | r<<3)
			ctxt.Andptr = ctxt.Andptr[1:]
			asmidx(ctxt, int(a.Scale), REG_NONE, base)
			ctxt.Andptr[0] = byte(v)
			ctxt.Andptr = ctxt.Andptr[1:]
			return
		}

		ctxt.Andptr[0] = byte(2<<6 | 4<<0 | r<<3)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmidx(ctxt, int(a.Scale), REG_NONE, base)
		goto putrelv
	}

	if REG_AX <= base && base <= REG_DI {
		if a.Index == REG_TLS {
			rel = obj.Reloc{}
			rel.Type = obj.R_TLS_IE
			rel.Siz = 4
			rel.Sym = nil
			rel.Add = int64(v)
			v = 0
		}

		if v == 0 && rel.Siz == 0 && base != REG_BP {
			ctxt.Andptr[0] = byte(0<<6 | reg[base]<<0 | r<<3)
			ctxt.Andptr = ctxt.Andptr[1:]
			return
		}

		if v >= -128 && v < 128 && rel.Siz == 0 {
			ctxt.Andptr[0] = byte(1<<6 | reg[base]<<0 | r<<3)
			ctxt.Andptr[1] = byte(v)
			ctxt.Andptr = ctxt.Andptr[2:]
			return
		}

		ctxt.Andptr[0] = byte(2<<6 | reg[base]<<0 | r<<3)
		ctxt.Andptr = ctxt.Andptr[1:]
		goto putrelv
	}

	goto bad

putrelv:
	if rel.Siz != 0 {
		var r *obj.Reloc

		if rel.Siz != 4 {
			ctxt.Diag("bad rel")
			goto bad
		}

		r = obj.Addrel(ctxt.Cursym)
		*r = rel
		r.Off = int32(ctxt.Curp.Pc + int64(-cap(ctxt.Andptr)+cap(ctxt.And[:])))
	}

	put4(ctxt, v)
	return

bad:
	ctxt.Diag("asmand: bad address %v", Dconv(p, 0, a))
	return
}

const (
	E = 0xff
)

var ymovtab = []uint8{
	/* push */
	APUSHL,
	Ycs,
	Ynone,
	0,
	0x0e,
	E,
	0,
	0,
	APUSHL,
	Yss,
	Ynone,
	0,
	0x16,
	E,
	0,
	0,
	APUSHL,
	Yds,
	Ynone,
	0,
	0x1e,
	E,
	0,
	0,
	APUSHL,
	Yes,
	Ynone,
	0,
	0x06,
	E,
	0,
	0,
	APUSHL,
	Yfs,
	Ynone,
	0,
	0x0f,
	0xa0,
	E,
	0,
	APUSHL,
	Ygs,
	Ynone,
	0,
	0x0f,
	0xa8,
	E,
	0,
	APUSHW,
	Ycs,
	Ynone,
	0,
	Pe,
	0x0e,
	E,
	0,
	APUSHW,
	Yss,
	Ynone,
	0,
	Pe,
	0x16,
	E,
	0,
	APUSHW,
	Yds,
	Ynone,
	0,
	Pe,
	0x1e,
	E,
	0,
	APUSHW,
	Yes,
	Ynone,
	0,
	Pe,
	0x06,
	E,
	0,
	APUSHW,
	Yfs,
	Ynone,
	0,
	Pe,
	0x0f,
	0xa0,
	E,
	APUSHW,
	Ygs,
	Ynone,
	0,
	Pe,
	0x0f,
	0xa8,
	E,

	/* pop */
	APOPL,
	Ynone,
	Yds,
	0,
	0x1f,
	E,
	0,
	0,
	APOPL,
	Ynone,
	Yes,
	0,
	0x07,
	E,
	0,
	0,
	APOPL,
	Ynone,
	Yss,
	0,
	0x17,
	E,
	0,
	0,
	APOPL,
	Ynone,
	Yfs,
	0,
	0x0f,
	0xa1,
	E,
	0,
	APOPL,
	Ynone,
	Ygs,
	0,
	0x0f,
	0xa9,
	E,
	0,
	APOPW,
	Ynone,
	Yds,
	0,
	Pe,
	0x1f,
	E,
	0,
	APOPW,
	Ynone,
	Yes,
	0,
	Pe,
	0x07,
	E,
	0,
	APOPW,
	Ynone,
	Yss,
	0,
	Pe,
	0x17,
	E,
	0,
	APOPW,
	Ynone,
	Yfs,
	0,
	Pe,
	0x0f,
	0xa1,
	E,
	APOPW,
	Ynone,
	Ygs,
	0,
	Pe,
	0x0f,
	0xa9,
	E,

	/* mov seg */
	AMOVW,
	Yes,
	Yml,
	1,
	0x8c,
	0,
	0,
	0,
	AMOVW,
	Ycs,
	Yml,
	1,
	0x8c,
	1,
	0,
	0,
	AMOVW,
	Yss,
	Yml,
	1,
	0x8c,
	2,
	0,
	0,
	AMOVW,
	Yds,
	Yml,
	1,
	0x8c,
	3,
	0,
	0,
	AMOVW,
	Yfs,
	Yml,
	1,
	0x8c,
	4,
	0,
	0,
	AMOVW,
	Ygs,
	Yml,
	1,
	0x8c,
	5,
	0,
	0,
	AMOVW,
	Yml,
	Yes,
	2,
	0x8e,
	0,
	0,
	0,
	AMOVW,
	Yml,
	Ycs,
	2,
	0x8e,
	1,
	0,
	0,
	AMOVW,
	Yml,
	Yss,
	2,
	0x8e,
	2,
	0,
	0,
	AMOVW,
	Yml,
	Yds,
	2,
	0x8e,
	3,
	0,
	0,
	AMOVW,
	Yml,
	Yfs,
	2,
	0x8e,
	4,
	0,
	0,
	AMOVW,
	Yml,
	Ygs,
	2,
	0x8e,
	5,
	0,
	0,

	/* mov cr */
	AMOVL,
	Ycr0,
	Yml,
	3,
	0x0f,
	0x20,
	0,
	0,
	AMOVL,
	Ycr2,
	Yml,
	3,
	0x0f,
	0x20,
	2,
	0,
	AMOVL,
	Ycr3,
	Yml,
	3,
	0x0f,
	0x20,
	3,
	0,
	AMOVL,
	Ycr4,
	Yml,
	3,
	0x0f,
	0x20,
	4,
	0,
	AMOVL,
	Yml,
	Ycr0,
	4,
	0x0f,
	0x22,
	0,
	0,
	AMOVL,
	Yml,
	Ycr2,
	4,
	0x0f,
	0x22,
	2,
	0,
	AMOVL,
	Yml,
	Ycr3,
	4,
	0x0f,
	0x22,
	3,
	0,
	AMOVL,
	Yml,
	Ycr4,
	4,
	0x0f,
	0x22,
	4,
	0,

	/* mov dr */
	AMOVL,
	Ydr0,
	Yml,
	3,
	0x0f,
	0x21,
	0,
	0,
	AMOVL,
	Ydr6,
	Yml,
	3,
	0x0f,
	0x21,
	6,
	0,
	AMOVL,
	Ydr7,
	Yml,
	3,
	0x0f,
	0x21,
	7,
	0,
	AMOVL,
	Yml,
	Ydr0,
	4,
	0x0f,
	0x23,
	0,
	0,
	AMOVL,
	Yml,
	Ydr6,
	4,
	0x0f,
	0x23,
	6,
	0,
	AMOVL,
	Yml,
	Ydr7,
	4,
	0x0f,
	0x23,
	7,
	0,

	/* mov tr */
	AMOVL,
	Ytr6,
	Yml,
	3,
	0x0f,
	0x24,
	6,
	0,
	AMOVL,
	Ytr7,
	Yml,
	3,
	0x0f,
	0x24,
	7,
	0,
	AMOVL,
	Yml,
	Ytr6,
	4,
	0x0f,
	0x26,
	6,
	E,
	AMOVL,
	Yml,
	Ytr7,
	4,
	0x0f,
	0x26,
	7,
	E,

	/* lgdt, sgdt, lidt, sidt */
	AMOVL,
	Ym,
	Ygdtr,
	4,
	0x0f,
	0x01,
	2,
	0,
	AMOVL,
	Ygdtr,
	Ym,
	3,
	0x0f,
	0x01,
	0,
	0,
	AMOVL,
	Ym,
	Yidtr,
	4,
	0x0f,
	0x01,
	3,
	0,
	AMOVL,
	Yidtr,
	Ym,
	3,
	0x0f,
	0x01,
	1,
	0,

	/* lldt, sldt */
	AMOVW,
	Yml,
	Yldtr,
	4,
	0x0f,
	0x00,
	2,
	0,
	AMOVW,
	Yldtr,
	Yml,
	3,
	0x0f,
	0x00,
	0,
	0,

	/* lmsw, smsw */
	AMOVW,
	Yml,
	Ymsw,
	4,
	0x0f,
	0x01,
	6,
	0,
	AMOVW,
	Ymsw,
	Yml,
	3,
	0x0f,
	0x01,
	4,
	0,

	/* ltr, str */
	AMOVW,
	Yml,
	Ytask,
	4,
	0x0f,
	0x00,
	3,
	0,
	AMOVW,
	Ytask,
	Yml,
	3,
	0x0f,
	0x00,
	1,
	0,

	/* load full pointer */
	AMOVL,
	Yml,
	Ycol,
	5,
	0,
	0,
	0,
	0,
	AMOVW,
	Yml,
	Ycol,
	5,
	Pe,
	0,
	0,
	0,

	/* double shift */
	ASHLL,
	Ycol,
	Yml,
	6,
	0xa4,
	0xa5,
	0,
	0,
	ASHRL,
	Ycol,
	Yml,
	6,
	0xac,
	0xad,
	0,
	0,

	/* extra imul */
	AIMULW,
	Yml,
	Yrl,
	7,
	Pq,
	0xaf,
	0,
	0,
	AIMULL,
	Yml,
	Yrl,
	7,
	Pm,
	0xaf,
	0,
	0,

	/* load TLS base pointer */
	AMOVL,
	Ytls,
	Yrl,
	8,
	0,
	0,
	0,
	0,
	0,
}

// byteswapreg returns a byte-addressable register (AX, BX, CX, DX)
// which is not referenced in a.
// If a is empty, it returns BX to account for MULB-like instructions
// that might use DX and AX.
func byteswapreg(ctxt *obj.Link, a *obj.Addr) int {
	var cana int
	var canb int
	var canc int
	var cand int

	cand = 1
	canc = cand
	canb = canc
	cana = canb

	if a.Type == obj.TYPE_NONE {
		cand = 0
		cana = cand
	}

	if a.Type == obj.TYPE_REG || ((a.Type == obj.TYPE_MEM || a.Type == obj.TYPE_ADDR) && a.Name == obj.NAME_NONE) {
		switch a.Reg {
		case REG_NONE:
			cand = 0
			cana = cand

		case REG_AX,
			REG_AL,
			REG_AH:
			cana = 0

		case REG_BX,
			REG_BL,
			REG_BH:
			canb = 0

		case REG_CX,
			REG_CL,
			REG_CH:
			canc = 0

		case REG_DX,
			REG_DL,
			REG_DH:
			cand = 0
		}
	}

	if a.Type == obj.TYPE_MEM || a.Type == obj.TYPE_ADDR {
		switch a.Index {
		case REG_AX:
			cana = 0

		case REG_BX:
			canb = 0

		case REG_CX:
			canc = 0

		case REG_DX:
			cand = 0
		}
	}

	if cana != 0 {
		return REG_AX
	}
	if canb != 0 {
		return REG_BX
	}
	if canc != 0 {
		return REG_CX
	}
	if cand != 0 {
		return REG_DX
	}

	ctxt.Diag("impossible byte register")
	log.Fatalf("bad code")
	return 0
}

func subreg(p *obj.Prog, from int, to int) {
	if false { /* debug['Q'] */
		fmt.Printf("\n%v\ts/%v/%v/\n", p, Rconv(from), Rconv(to))
	}

	if int(p.From.Reg) == from {
		p.From.Reg = int16(to)
		p.Ft = 0
	}

	if int(p.To.Reg) == from {
		p.To.Reg = int16(to)
		p.Tt = 0
	}

	if int(p.From.Index) == from {
		p.From.Index = int16(to)
		p.Ft = 0
	}

	if int(p.To.Index) == from {
		p.To.Index = int16(to)
		p.Tt = 0
	}

	if false { /* debug['Q'] */
		fmt.Printf("%v\n", p)
	}
}

func mediaop(ctxt *obj.Link, o *Optab, op int, osize int, z int) int {
	switch op {
	case Pm,
		Pe,
		Pf2,
		Pf3:
		if osize != 1 {
			if op != Pm {
				ctxt.Andptr[0] = byte(op)
				ctxt.Andptr = ctxt.Andptr[1:]
			}
			ctxt.Andptr[0] = Pm
			ctxt.Andptr = ctxt.Andptr[1:]
			z++
			op = int(o.op[z])
			break
		}
		fallthrough

	default:
		if -cap(ctxt.Andptr) == -cap(ctxt.And) || ctxt.And[-cap(ctxt.Andptr)+cap(ctxt.And[:])-1] != Pm {
			ctxt.Andptr[0] = Pm
			ctxt.Andptr = ctxt.Andptr[1:]
		}
	}

	ctxt.Andptr[0] = byte(op)
	ctxt.Andptr = ctxt.Andptr[1:]
	return z
}

func doasm(ctxt *obj.Link, p *obj.Prog) {
	var o *Optab
	var q *obj.Prog
	var pp obj.Prog
	var t []byte
	var z int
	var op int
	var ft int
	var tt int
	var breg int
	var v int32
	var pre int32
	var rel obj.Reloc
	var r *obj.Reloc
	var a *obj.Addr

	ctxt.Curp = p // TODO

	pre = int32(prefixof(ctxt, &p.From))

	if pre != 0 {
		ctxt.Andptr[0] = byte(pre)
		ctxt.Andptr = ctxt.Andptr[1:]
	}
	pre = int32(prefixof(ctxt, &p.To))
	if pre != 0 {
		ctxt.Andptr[0] = byte(pre)
		ctxt.Andptr = ctxt.Andptr[1:]
	}

	if p.Ft == 0 {
		p.Ft = uint8(oclass(ctxt, p, &p.From))
	}
	if p.Tt == 0 {
		p.Tt = uint8(oclass(ctxt, p, &p.To))
	}

	ft = int(p.Ft) * Ymax
	tt = int(p.Tt) * Ymax
	o = opindex[p.As]
	t = o.ytab
	if t == nil {
		ctxt.Diag("asmins: noproto %v", p)
		return
	}

	for z = 0; t[0] != 0; (func() { z += int(t[3]); t = t[4:] })() {
		if ycover[ft+int(t[0])] != 0 {
			if ycover[tt+int(t[1])] != 0 {
				goto found
			}
		}
	}
	goto domov

found:
	switch o.prefix {
	case Pq: /* 16 bit escape and opcode escape */
		ctxt.Andptr[0] = Pe
		ctxt.Andptr = ctxt.Andptr[1:]

		ctxt.Andptr[0] = Pm
		ctxt.Andptr = ctxt.Andptr[1:]

	case Pf2, /* xmm opcode escape */
		Pf3:
		ctxt.Andptr[0] = byte(o.prefix)
		ctxt.Andptr = ctxt.Andptr[1:]

		ctxt.Andptr[0] = Pm
		ctxt.Andptr = ctxt.Andptr[1:]

	case Pm: /* opcode escape */
		ctxt.Andptr[0] = Pm
		ctxt.Andptr = ctxt.Andptr[1:]

	case Pe: /* 16 bit escape */
		ctxt.Andptr[0] = Pe
		ctxt.Andptr = ctxt.Andptr[1:]

	case Pb: /* botch */
		break
	}

	op = int(o.op[z])
	switch t[2] {
	default:
		ctxt.Diag("asmins: unknown z %d %v", t[2], p)
		return

	case Zpseudo:
		break

	case Zlit:
		for ; ; z++ {
			op = int(o.op[z])
			if op == 0 {
				break
			}
			ctxt.Andptr[0] = byte(op)
			ctxt.Andptr = ctxt.Andptr[1:]
		}

	case Zlitm_r:
		for ; ; z++ {
			op = int(o.op[z])
			if op == 0 {
				break
			}
			ctxt.Andptr[0] = byte(op)
			ctxt.Andptr = ctxt.Andptr[1:]
		}
		asmand(ctxt, p, &p.From, reg[p.To.Reg])

	case Zm_r:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.From, reg[p.To.Reg])

	case Zm2_r:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		ctxt.Andptr[0] = byte(o.op[z+1])
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.From, reg[p.To.Reg])

	case Zm_r_xm:
		mediaop(ctxt, o, op, int(t[3]), z)
		asmand(ctxt, p, &p.From, reg[p.To.Reg])

	case Zm_r_i_xm:
		mediaop(ctxt, o, op, int(t[3]), z)
		asmand(ctxt, p, &p.From, reg[p.To.Reg])
		ctxt.Andptr[0] = byte(p.To.Offset)
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zibm_r:
		for {
			tmp2 := z
			z++
			op = int(o.op[tmp2])
			if op == 0 {
				break
			}
			ctxt.Andptr[0] = byte(op)
			ctxt.Andptr = ctxt.Andptr[1:]
		}
		asmand(ctxt, p, &p.From, reg[p.To.Reg])
		ctxt.Andptr[0] = byte(p.To.Offset)
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zaut_r:
		ctxt.Andptr[0] = 0x8d
		ctxt.Andptr = ctxt.Andptr[1:] /* leal */
		if p.From.Type != obj.TYPE_ADDR {
			ctxt.Diag("asmins: Zaut sb type ADDR")
		}
		p.From.Type = obj.TYPE_MEM
		p.Ft = 0
		asmand(ctxt, p, &p.From, reg[p.To.Reg])
		p.From.Type = obj.TYPE_ADDR
		p.Ft = 0

	case Zm_o:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.From, int(o.op[z+1]))

	case Zr_m:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.To, reg[p.From.Reg])

	case Zr_m_xm:
		mediaop(ctxt, o, op, int(t[3]), z)
		asmand(ctxt, p, &p.To, reg[p.From.Reg])

	case Zr_m_i_xm:
		mediaop(ctxt, o, op, int(t[3]), z)
		asmand(ctxt, p, &p.To, reg[p.From.Reg])
		ctxt.Andptr[0] = byte(p.From.Offset)
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zcallindreg:
		r = obj.Addrel(ctxt.Cursym)
		r.Off = int32(p.Pc)
		r.Type = obj.R_CALLIND
		r.Siz = 0
		fallthrough

		// fallthrough
	case Zo_m:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]

		asmand(ctxt, p, &p.To, int(o.op[z+1]))

	case Zm_ibo:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.From, int(o.op[z+1]))
		ctxt.Andptr[0] = byte(vaddr(ctxt, p, &p.To, nil))
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zibo_m:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.To, int(o.op[z+1]))
		ctxt.Andptr[0] = byte(vaddr(ctxt, p, &p.From, nil))
		ctxt.Andptr = ctxt.Andptr[1:]

	case Z_ib,
		Zib_:
		if t[2] == Zib_ {
			a = &p.From
		} else {
			a = &p.To
		}
		v = vaddr(ctxt, p, a, nil)
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		ctxt.Andptr[0] = byte(v)
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zib_rp:
		ctxt.Andptr[0] = byte(op + reg[p.To.Reg])
		ctxt.Andptr = ctxt.Andptr[1:]
		ctxt.Andptr[0] = byte(vaddr(ctxt, p, &p.From, nil))
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zil_rp:
		ctxt.Andptr[0] = byte(op + reg[p.To.Reg])
		ctxt.Andptr = ctxt.Andptr[1:]
		if o.prefix == Pe {
			v = vaddr(ctxt, p, &p.From, nil)
			ctxt.Andptr[0] = byte(v)
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = byte(v >> 8)
			ctxt.Andptr = ctxt.Andptr[1:]
		} else {
			relput4(ctxt, p, &p.From)
		}

	case Zib_rr:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.To, reg[p.To.Reg])
		ctxt.Andptr[0] = byte(vaddr(ctxt, p, &p.From, nil))
		ctxt.Andptr = ctxt.Andptr[1:]

	case Z_il,
		Zil_:
		if t[2] == Zil_ {
			a = &p.From
		} else {
			a = &p.To
		}
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		if o.prefix == Pe {
			v = vaddr(ctxt, p, a, nil)
			ctxt.Andptr[0] = byte(v)
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = byte(v >> 8)
			ctxt.Andptr = ctxt.Andptr[1:]
		} else {
			relput4(ctxt, p, a)
		}

	case Zm_ilo,
		Zilo_m:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		if t[2] == Zilo_m {
			a = &p.From
			asmand(ctxt, p, &p.To, int(o.op[z+1]))
		} else {
			a = &p.To
			asmand(ctxt, p, &p.From, int(o.op[z+1]))
		}

		if o.prefix == Pe {
			v = vaddr(ctxt, p, a, nil)
			ctxt.Andptr[0] = byte(v)
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = byte(v >> 8)
			ctxt.Andptr = ctxt.Andptr[1:]
		} else {
			relput4(ctxt, p, a)
		}

	case Zil_rr:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.To, reg[p.To.Reg])
		if o.prefix == Pe {
			v = vaddr(ctxt, p, &p.From, nil)
			ctxt.Andptr[0] = byte(v)
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = byte(v >> 8)
			ctxt.Andptr = ctxt.Andptr[1:]
		} else {
			relput4(ctxt, p, &p.From)
		}

	case Z_rp:
		ctxt.Andptr[0] = byte(op + reg[p.To.Reg])
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zrp_:
		ctxt.Andptr[0] = byte(op + reg[p.From.Reg])
		ctxt.Andptr = ctxt.Andptr[1:]

	case Zclr:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.To, reg[p.To.Reg])

	case Zcall:
		if p.To.Sym == nil {
			ctxt.Diag("call without target")
			log.Fatalf("bad code")
		}

		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		r = obj.Addrel(ctxt.Cursym)
		r.Off = int32(p.Pc + int64(-cap(ctxt.Andptr)+cap(ctxt.And[:])))
		r.Type = obj.R_CALL
		r.Siz = 4
		r.Sym = p.To.Sym
		r.Add = p.To.Offset
		put4(ctxt, 0)

	case Zbr,
		Zjmp,
		Zloop:
		if p.To.Sym != nil {
			if t[2] != Zjmp {
				ctxt.Diag("branch to ATEXT")
				log.Fatalf("bad code")
			}

			ctxt.Andptr[0] = byte(o.op[z+1])
			ctxt.Andptr = ctxt.Andptr[1:]
			r = obj.Addrel(ctxt.Cursym)
			r.Off = int32(p.Pc + int64(-cap(ctxt.Andptr)+cap(ctxt.And[:])))
			r.Sym = p.To.Sym
			r.Type = obj.R_PCREL
			r.Siz = 4
			put4(ctxt, 0)
			break
		}

		// Assumes q is in this function.
		// Fill in backward jump now.
		q = p.Pcond

		if q == nil {
			ctxt.Diag("jmp/branch/loop without target")
			log.Fatalf("bad code")
		}

		if p.Back&1 != 0 {
			v = int32(q.Pc - (p.Pc + 2))
			if v >= -128 {
				if p.As == AJCXZW {
					ctxt.Andptr[0] = 0x67
					ctxt.Andptr = ctxt.Andptr[1:]
				}
				ctxt.Andptr[0] = byte(op)
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = byte(v)
				ctxt.Andptr = ctxt.Andptr[1:]
			} else if t[2] == Zloop {
				ctxt.Diag("loop too far: %v", p)
			} else {
				v -= 5 - 2
				if t[2] == Zbr {
					ctxt.Andptr[0] = 0x0f
					ctxt.Andptr = ctxt.Andptr[1:]
					v--
				}

				ctxt.Andptr[0] = byte(o.op[z+1])
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = byte(v)
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = byte(v >> 8)
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = byte(v >> 16)
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = byte(v >> 24)
				ctxt.Andptr = ctxt.Andptr[1:]
			}

			break
		}

		// Annotate target; will fill in later.
		p.Forwd = q.Comefrom

		q.Comefrom = p
		if p.Back&2 != 0 { // short
			if p.As == AJCXZW {
				ctxt.Andptr[0] = 0x67
				ctxt.Andptr = ctxt.Andptr[1:]
			}
			ctxt.Andptr[0] = byte(op)
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0
			ctxt.Andptr = ctxt.Andptr[1:]
		} else if t[2] == Zloop {
			ctxt.Diag("loop too far: %v", p)
		} else {
			if t[2] == Zbr {
				ctxt.Andptr[0] = 0x0f
				ctxt.Andptr = ctxt.Andptr[1:]
			}
			ctxt.Andptr[0] = byte(o.op[z+1])
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0
			ctxt.Andptr = ctxt.Andptr[1:]
		}

	case Zcallcon,
		Zjmpcon:
		if t[2] == Zcallcon {
			ctxt.Andptr[0] = byte(op)
			ctxt.Andptr = ctxt.Andptr[1:]
		} else {
			ctxt.Andptr[0] = byte(o.op[z+1])
			ctxt.Andptr = ctxt.Andptr[1:]
		}
		r = obj.Addrel(ctxt.Cursym)
		r.Off = int32(p.Pc + int64(-cap(ctxt.Andptr)+cap(ctxt.And[:])))
		r.Type = obj.R_PCREL
		r.Siz = 4
		r.Add = p.To.Offset
		put4(ctxt, 0)

	case Zcallind:
		ctxt.Andptr[0] = byte(op)
		ctxt.Andptr = ctxt.Andptr[1:]
		ctxt.Andptr[0] = byte(o.op[z+1])
		ctxt.Andptr = ctxt.Andptr[1:]
		r = obj.Addrel(ctxt.Cursym)
		r.Off = int32(p.Pc + int64(-cap(ctxt.Andptr)+cap(ctxt.And[:])))
		r.Type = obj.R_ADDR
		r.Siz = 4
		r.Add = p.To.Offset
		r.Sym = p.To.Sym
		put4(ctxt, 0)

	case Zbyte:
		v = vaddr(ctxt, p, &p.From, &rel)
		if rel.Siz != 0 {
			rel.Siz = uint8(op)
			r = obj.Addrel(ctxt.Cursym)
			*r = rel
			r.Off = int32(p.Pc + int64(-cap(ctxt.Andptr)+cap(ctxt.And[:])))
		}

		ctxt.Andptr[0] = byte(v)
		ctxt.Andptr = ctxt.Andptr[1:]
		if op > 1 {
			ctxt.Andptr[0] = byte(v >> 8)
			ctxt.Andptr = ctxt.Andptr[1:]
			if op > 2 {
				ctxt.Andptr[0] = byte(v >> 16)
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = byte(v >> 24)
				ctxt.Andptr = ctxt.Andptr[1:]
			}
		}

	case Zmov:
		goto domov
	}

	return

domov:
	for t = []byte(ymovtab); t[0] != 0; t = t[8:] {
		if p.As == int16(t[0]) {
			if ycover[ft+int(t[1])] != 0 {
				if ycover[tt+int(t[2])] != 0 {
					goto mfound
				}
			}
		}
	}

	/*
	 * here, the assembly has failed.
	 * if its a byte instruction that has
	 * unaddressable registers, try to
	 * exchange registers and reissue the
	 * instruction with the operands renamed.
	 */
bad:
	pp = *p

	z = int(p.From.Reg)
	if p.From.Type == obj.TYPE_REG && z >= REG_BP && z <= REG_DI {
		breg = byteswapreg(ctxt, &p.To)
		if breg != REG_AX {
			ctxt.Andptr[0] = 0x87
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg lhs,bx */
			asmand(ctxt, p, &p.From, reg[breg])
			subreg(&pp, z, breg)
			doasm(ctxt, &pp)
			ctxt.Andptr[0] = 0x87
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg lhs,bx */
			asmand(ctxt, p, &p.From, reg[breg])
		} else {
			ctxt.Andptr[0] = byte(0x90 + reg[z])
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg lsh,ax */
			subreg(&pp, z, REG_AX)
			doasm(ctxt, &pp)
			ctxt.Andptr[0] = byte(0x90 + reg[z])
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg lsh,ax */
		}

		return
	}

	z = int(p.To.Reg)
	if p.To.Type == obj.TYPE_REG && z >= REG_BP && z <= REG_DI {
		breg = byteswapreg(ctxt, &p.From)
		if breg != REG_AX {
			ctxt.Andptr[0] = 0x87
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg rhs,bx */
			asmand(ctxt, p, &p.To, reg[breg])
			subreg(&pp, z, breg)
			doasm(ctxt, &pp)
			ctxt.Andptr[0] = 0x87
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg rhs,bx */
			asmand(ctxt, p, &p.To, reg[breg])
		} else {
			ctxt.Andptr[0] = byte(0x90 + reg[z])
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg rsh,ax */
			subreg(&pp, z, REG_AX)
			doasm(ctxt, &pp)
			ctxt.Andptr[0] = byte(0x90 + reg[z])
			ctxt.Andptr = ctxt.Andptr[1:] /* xchg rsh,ax */
		}

		return
	}

	ctxt.Diag("doasm: notfound t2=%d from=%d to=%d %v", t[2], p.Ft, p.Tt, p)
	return

mfound:
	switch t[3] {
	default:
		ctxt.Diag("asmins: unknown mov %d %v", t[3], p)

	case 0: /* lit */
		for z = 4; t[z] != E; z++ {
			ctxt.Andptr[0] = t[z]
			ctxt.Andptr = ctxt.Andptr[1:]
		}

	case 1: /* r,m */
		ctxt.Andptr[0] = t[4]
		ctxt.Andptr = ctxt.Andptr[1:]

		asmand(ctxt, p, &p.To, int(t[5]))

	case 2: /* m,r */
		ctxt.Andptr[0] = t[4]
		ctxt.Andptr = ctxt.Andptr[1:]

		asmand(ctxt, p, &p.From, int(t[5]))

	case 3: /* r,m - 2op */
		ctxt.Andptr[0] = t[4]
		ctxt.Andptr = ctxt.Andptr[1:]

		ctxt.Andptr[0] = t[5]
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.To, int(t[6]))

	case 4: /* m,r - 2op */
		ctxt.Andptr[0] = t[4]
		ctxt.Andptr = ctxt.Andptr[1:]

		ctxt.Andptr[0] = t[5]
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.From, int(t[6]))

	case 5: /* load full pointer, trash heap */
		if t[4] != 0 {
			ctxt.Andptr[0] = t[4]
			ctxt.Andptr = ctxt.Andptr[1:]
		}
		switch p.To.Index {
		default:
			goto bad

		case REG_DS:
			ctxt.Andptr[0] = 0xc5
			ctxt.Andptr = ctxt.Andptr[1:]

		case REG_SS:
			ctxt.Andptr[0] = 0x0f
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0xb2
			ctxt.Andptr = ctxt.Andptr[1:]

		case REG_ES:
			ctxt.Andptr[0] = 0xc4
			ctxt.Andptr = ctxt.Andptr[1:]

		case REG_FS:
			ctxt.Andptr[0] = 0x0f
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0xb4
			ctxt.Andptr = ctxt.Andptr[1:]

		case REG_GS:
			ctxt.Andptr[0] = 0x0f
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = 0xb5
			ctxt.Andptr = ctxt.Andptr[1:]
		}

		asmand(ctxt, p, &p.From, reg[p.To.Reg])

	case 6: /* double shift */
		switch p.From.Type {
		default:
			goto bad

		case obj.TYPE_CONST:
			ctxt.Andptr[0] = 0x0f
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = t[4]
			ctxt.Andptr = ctxt.Andptr[1:]
			asmand(ctxt, p, &p.To, reg[p.From.Index])
			ctxt.Andptr[0] = byte(p.From.Offset)
			ctxt.Andptr = ctxt.Andptr[1:]

		case obj.TYPE_REG:
			switch p.From.Reg {
			default:
				goto bad

			case REG_CL,
				REG_CX:
				ctxt.Andptr[0] = 0x0f
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = t[5]
				ctxt.Andptr = ctxt.Andptr[1:]
				asmand(ctxt, p, &p.To, reg[p.From.Index])
			}
		}

	case 7: /* imul rm,r */
		if t[4] == Pq {
			ctxt.Andptr[0] = Pe
			ctxt.Andptr = ctxt.Andptr[1:]
			ctxt.Andptr[0] = Pm
			ctxt.Andptr = ctxt.Andptr[1:]
		} else {
			ctxt.Andptr[0] = t[4]
			ctxt.Andptr = ctxt.Andptr[1:]
		}
		ctxt.Andptr[0] = t[5]
		ctxt.Andptr = ctxt.Andptr[1:]
		asmand(ctxt, p, &p.From, reg[p.To.Reg])

		// NOTE: The systems listed here are the ones that use the "TLS initial exec" model,
	// where you load the TLS base register into a register and then index off that
	// register to access the actual TLS variables. Systems that allow direct TLS access
	// are handled in prefixof above and should not be listed here.
	case 8: /* mov tls, r */
		switch ctxt.Headtype {
		default:
			log.Fatalf("unknown TLS base location for %s", obj.Headstr(ctxt.Headtype))

			// ELF TLS base is 0(GS).
		case obj.Hlinux,
			obj.Hnacl:
			pp.From = p.From

			pp.From.Type = obj.TYPE_MEM
			pp.From.Reg = REG_GS
			pp.From.Offset = 0
			pp.From.Index = REG_NONE
			pp.From.Scale = 0
			ctxt.Andptr[0] = 0x65
			ctxt.Andptr = ctxt.Andptr[1:] // GS
			ctxt.Andptr[0] = 0x8B
			ctxt.Andptr = ctxt.Andptr[1:]
			asmand(ctxt, p, &pp.From, reg[p.To.Reg])

		case obj.Hplan9:
			if ctxt.Plan9privates == nil {
				ctxt.Plan9privates = obj.Linklookup(ctxt, "_privates", 0)
			}
			pp.From = obj.Addr{}
			pp.From.Type = obj.TYPE_MEM
			pp.From.Name = obj.NAME_EXTERN
			pp.From.Sym = ctxt.Plan9privates
			pp.From.Offset = 0
			pp.From.Index = REG_NONE
			ctxt.Andptr[0] = 0x8B
			ctxt.Andptr = ctxt.Andptr[1:]
			asmand(ctxt, p, &pp.From, reg[p.To.Reg])

			// Windows TLS base is always 0x14(FS).
		case obj.Hwindows:
			pp.From = p.From

			pp.From.Type = obj.TYPE_MEM
			pp.From.Reg = REG_FS
			pp.From.Offset = 0x14
			pp.From.Index = REG_NONE
			pp.From.Scale = 0
			ctxt.Andptr[0] = 0x64
			ctxt.Andptr = ctxt.Andptr[1:] // FS
			ctxt.Andptr[0] = 0x8B
			ctxt.Andptr = ctxt.Andptr[1:]
			asmand(ctxt, p, &pp.From, reg[p.To.Reg])
		}
	}
}

var naclret = []uint8{
	0x5d, // POPL BP
	// 0x8b, 0x7d, 0x00, // MOVL (BP), DI - catch return to invalid address, for debugging
	0x83,
	0xe5,
	0xe0, // ANDL $~31, BP
	0xff,
	0xe5, // JMP BP
}

func asmins(ctxt *obj.Link, p *obj.Prog) {
	var r *obj.Reloc

	ctxt.Andptr = ctxt.And[:]

	if p.As == obj.AUSEFIELD {
		r = obj.Addrel(ctxt.Cursym)
		r.Off = 0
		r.Sym = p.From.Sym
		r.Type = obj.R_USEFIELD
		r.Siz = 0
		return
	}

	if ctxt.Headtype == obj.Hnacl {
		switch p.As {
		case obj.ARET:
			copy(ctxt.Andptr, naclret)
			ctxt.Andptr = ctxt.Andptr[len(naclret):]
			return

		case obj.ACALL,
			obj.AJMP:
			if p.To.Type == obj.TYPE_REG && REG_AX <= p.To.Reg && p.To.Reg <= REG_DI {
				ctxt.Andptr[0] = 0x83
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = byte(0xe0 | (p.To.Reg - REG_AX))
				ctxt.Andptr = ctxt.Andptr[1:]
				ctxt.Andptr[0] = 0xe0
				ctxt.Andptr = ctxt.Andptr[1:]
			}

		case AINT:
			ctxt.Andptr[0] = 0xf4
			ctxt.Andptr = ctxt.Andptr[1:]
			return
		}
	}

	doasm(ctxt, p)
	if -cap(ctxt.Andptr) > -cap(ctxt.And[len(ctxt.And):]) {
		fmt.Printf("and[] is too short - %d byte instruction\n", -cap(ctxt.Andptr)+cap(ctxt.And[:]))
		log.Fatalf("bad code")
	}
}
