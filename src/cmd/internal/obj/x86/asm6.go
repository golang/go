// Inferno utils/6l/span.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/span.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package x86

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
	"internal/binary"
	"internal/buildcfg"
	"log"
	"strings"
)

var (
	plan9privates *obj.LSym
)

// Instruction layout.

// Loop alignment constants:
// want to align loop entry to loopAlign-byte boundary,
// and willing to insert at most maxLoopPad bytes of NOP to do so.
// We define a loop entry as the target of a backward jump.
//
// gcc uses maxLoopPad = 10 for its 'generic x86-64' config,
// and it aligns all jump targets, not just backward jump targets.
//
// As of 6/1/2012, the effect of setting maxLoopPad = 10 here
// is very slight but negative, so the alignment is disabled by
// setting MaxLoopPad = 0. The code is here for reference and
// for future experiments.
const (
	loopAlign  = 16
	maxLoopPad = 0
)

// Bit flags that are used to express jump target properties.
const (
	// branchBackwards marks targets that are located behind.
	// Used to express jumps to loop headers.
	branchBackwards = (1 << iota)
	// branchShort marks branches those target is close,
	// with offset is in -128..127 range.
	branchShort
	// branchLoopHead marks loop entry.
	// Used to insert padding for misaligned loops.
	branchLoopHead
)

// opBytes holds optab encoding bytes.
// Each ytab reserves fixed amount of bytes in this array.
//
// The size should be the minimal number of bytes that
// are enough to hold biggest optab op lines.
type opBytes [31]uint8

type Optab struct {
	as     obj.As
	ytab   []ytab
	prefix uint8
	op     opBytes
}

type movtab struct {
	as   obj.As
	ft   uint8
	f3t  uint8
	tt   uint8
	code uint8
	op   [4]uint8
}

const (
	Yxxx = iota
	Ynone
	Yi0 // $0
	Yi1 // $1
	Yu2 // $x, x fits in uint2
	Yi8 // $x, x fits in int8
	Yu8 // $x, x fits in uint8
	Yu7 // $x, x in 0..127 (fits in both int8 and uint8)
	Ys32
	Yi32
	Yi64
	Yiauto
	Yal
	Ycl
	Yax
	Ycx
	Yrb
	Yrl
	Yrl32 // Yrl on 32-bit system
	Yrf
	Yf0
	Yrx
	Ymb
	Yml
	Ym
	Ybr
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
	Ycr8
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
	Yxr0          // X0 only. "<XMM0>" notation in Intel manual.
	YxrEvexMulti4 // [ X<n> - X<n+3> ]; multisource YxrEvex
	Yxr           // X0..X15
	YxrEvex       // X0..X31
	Yxm
	YxmEvex       // YxrEvex+Ym
	Yxvm          // VSIB vector array; vm32x/vm64x
	YxvmEvex      // Yxvm which permits High-16 X register as index.
	YyrEvexMulti4 // [ Y<n> - Y<n+3> ]; multisource YyrEvex
	Yyr           // Y0..Y15
	YyrEvex       // Y0..Y31
	Yym
	YymEvex   // YyrEvex+Ym
	Yyvm      // VSIB vector array; vm32y/vm64y
	YyvmEvex  // Yyvm which permits High-16 Y register as index.
	YzrMulti4 // [ Z<n> - Z<n+3> ]; multisource YzrEvex
	Yzr       // Z0..Z31
	Yzm       // Yzr+Ym
	Yzvm      // VSIB vector array; vm32z/vm64z
	Yk0       // K0
	Yknot0    // K1..K7; write mask
	Yk        // K0..K7; used for KOP
	Ykm       // Yk+Ym; used for KOP
	Ytls
	Ytextsize
	Yindir
	Ymax
)

const (
	Zxxx = iota
	Zlit
	Zlitm_r
	Zlitr_m
	Zlit_m_r
	Z_rp
	Zbr
	Zcall
	Zcallcon
	Zcallduff
	Zcallind
	Zcallindreg
	Zib_
	Zib_rp
	Zibo_m
	Zibo_m_xm
	Zil_
	Zil_rp
	Ziq_rp
	Zilo_m
	Zjmp
	Zjmpcon
	Zloop
	Zo_iw
	Zm_o
	Zm_r
	Z_m_r
	Zm2_r
	Zm_r_xm
	Zm_r_i_xm
	Zm_r_xm_nr
	Zr_m_xm_nr
	Zibm_r // mmx1,mmx2/mem64,imm8
	Zibr_m
	Zmb_r
	Zaut_r
	Zo_m
	Zo_m64
	Zpseudo
	Zr_m
	Zr_m_xm
	Zrp_
	Z_ib
	Z_il
	Zm_ibo
	Zm_ilo
	Zib_rr
	Zil_rr
	Zbyte

	Zvex_rm_v_r
	Zvex_rm_v_ro
	Zvex_r_v_rm
	Zvex_i_rm_vo
	Zvex_v_rm_r
	Zvex_i_rm_r
	Zvex_i_r_v
	Zvex_i_rm_v_r
	Zvex
	Zvex_rm_r_vo
	Zvex_i_r_rm
	Zvex_hr_rm_v_r

	Zevex_first
	Zevex_i_r_k_rm
	Zevex_i_r_rm
	Zevex_i_rm_k_r
	Zevex_i_rm_k_vo
	Zevex_i_rm_r
	Zevex_i_rm_v_k_r
	Zevex_i_rm_v_r
	Zevex_i_rm_vo
	Zevex_k_rmo
	Zevex_r_k_rm
	Zevex_r_v_k_rm
	Zevex_r_v_rm
	Zevex_rm_k_r
	Zevex_rm_v_k_r
	Zevex_rm_v_r
	Zevex_last

	Zmax
)

const (
	Px   = 0
	Px1  = 1    // symbolic; exact value doesn't matter
	P32  = 0x32 // 32-bit only
	Pe   = 0x66 // operand escape
	Pm   = 0x0f // 2byte opcode escape
	Pq   = 0xff // both escapes: 66 0f
	Pb   = 0xfe // byte operands
	Pf2  = 0xf2 // xmm escape 1: f2 0f
	Pf3  = 0xf3 // xmm escape 2: f3 0f
	Pef3 = 0xf5 // xmm escape 2 with 16-bit prefix: 66 f3 0f
	Pq3  = 0x67 // xmm escape 3: 66 48 0f
	Pq4  = 0x68 // xmm escape 4: 66 0F 38
	Pq4w = 0x69 // Pq4 with Rex.w 66 0F 38
	Pq5  = 0x6a // xmm escape 5: F3 0F 38
	Pq5w = 0x6b // Pq5 with Rex.w F3 0F 38
	Pfw  = 0xf4 // Pf3 with Rex.w: f3 48 0f
	Pw   = 0x48 // Rex.w
	Pw8  = 0x90 // symbolic; exact value doesn't matter
	Py   = 0x80 // defaults to 64-bit mode
	Py1  = 0x81 // symbolic; exact value doesn't matter
	Py3  = 0x83 // symbolic; exact value doesn't matter
	Pavx = 0x84 // symbolic; exact value doesn't matter

	RxrEvex = 1 << 4 // AVX512 extension to REX.R/VEX.R
	Rxw     = 1 << 3 // =1, 64-bit operand size
	Rxr     = 1 << 2 // extend modrm reg
	Rxx     = 1 << 1 // extend sib index
	Rxb     = 1 << 0 // extend modrm r/m, sib base, or opcode reg
)

const (
	// Encoding for VEX prefix in tables.
	// The P, L, and W fields are chosen to match
	// their eventual locations in the VEX prefix bytes.

	// Encoding for VEX prefix in tables.
	// The P, L, and W fields are chosen to match
	// their eventual locations in the VEX prefix bytes.

	// Using spare bit to make leading [E]VEX encoding byte different from
	// 0x0f even if all other VEX fields are 0.
	avxEscape = 1 << 6

	// P field - 2 bits
	vex66 = 1 << 0
	vexF3 = 2 << 0
	vexF2 = 3 << 0
	// L field - 1 bit
	vexLZ  = 0 << 2
	vexLIG = 0 << 2
	vex128 = 0 << 2
	vex256 = 1 << 2
	// W field - 1 bit
	vexWIG = 0 << 7
	vexW0  = 0 << 7
	vexW1  = 1 << 7
	// M field - 5 bits, but mostly reserved; we can store up to 3
	vex0F   = 1 << 3
	vex0F38 = 2 << 3
	vex0F3A = 3 << 3
)

var ycover [Ymax * Ymax]uint8

var reg [MAXREG]int

var regrex [MAXREG + 1]int

var ynone = []ytab{
	{Zlit, 1, argList{}},
}

var ytext = []ytab{
	{Zpseudo, 0, argList{Ymb, Ytextsize}},
	{Zpseudo, 1, argList{Ymb, Yi32, Ytextsize}},
}

var ynop = []ytab{
	{Zpseudo, 0, argList{}},
	{Zpseudo, 0, argList{Yiauto}},
	{Zpseudo, 0, argList{Yml}},
	{Zpseudo, 0, argList{Yrf}},
	{Zpseudo, 0, argList{Yxr}},
	{Zpseudo, 0, argList{Yiauto}},
	{Zpseudo, 0, argList{Yml}},
	{Zpseudo, 0, argList{Yrf}},
	{Zpseudo, 1, argList{Yxr}},
}

var yfuncdata = []ytab{
	{Zpseudo, 0, argList{Yi32, Ym}},
}

var ypcdata = []ytab{
	{Zpseudo, 0, argList{Yi32, Yi32}},
}

var yxorb = []ytab{
	{Zib_, 1, argList{Yi32, Yal}},
	{Zibo_m, 2, argList{Yi32, Ymb}},
	{Zr_m, 1, argList{Yrb, Ymb}},
	{Zm_r, 1, argList{Ymb, Yrb}},
}

var yaddl = []ytab{
	{Zibo_m, 2, argList{Yi8, Yml}},
	{Zil_, 1, argList{Yi32, Yax}},
	{Zilo_m, 2, argList{Yi32, Yml}},
	{Zr_m, 1, argList{Yrl, Yml}},
	{Zm_r, 1, argList{Yml, Yrl}},
}

var yincl = []ytab{
	{Z_rp, 1, argList{Yrl}},
	{Zo_m, 2, argList{Yml}},
}

var yincq = []ytab{
	{Zo_m, 2, argList{Yml}},
}

var ycmpb = []ytab{
	{Z_ib, 1, argList{Yal, Yi32}},
	{Zm_ibo, 2, argList{Ymb, Yi32}},
	{Zm_r, 1, argList{Ymb, Yrb}},
	{Zr_m, 1, argList{Yrb, Ymb}},
}

var ycmpl = []ytab{
	{Zm_ibo, 2, argList{Yml, Yi8}},
	{Z_il, 1, argList{Yax, Yi32}},
	{Zm_ilo, 2, argList{Yml, Yi32}},
	{Zm_r, 1, argList{Yml, Yrl}},
	{Zr_m, 1, argList{Yrl, Yml}},
}

var yshb = []ytab{
	{Zo_m, 2, argList{Yi1, Ymb}},
	{Zibo_m, 2, argList{Yu8, Ymb}},
	{Zo_m, 2, argList{Ycx, Ymb}},
}

var yshl = []ytab{
	{Zo_m, 2, argList{Yi1, Yml}},
	{Zibo_m, 2, argList{Yu8, Yml}},
	{Zo_m, 2, argList{Ycl, Yml}},
	{Zo_m, 2, argList{Ycx, Yml}},
}

var ytestl = []ytab{
	{Zil_, 1, argList{Yi32, Yax}},
	{Zilo_m, 2, argList{Yi32, Yml}},
	{Zr_m, 1, argList{Yrl, Yml}},
	{Zm_r, 1, argList{Yml, Yrl}},
}

var ymovb = []ytab{
	{Zr_m, 1, argList{Yrb, Ymb}},
	{Zm_r, 1, argList{Ymb, Yrb}},
	{Zib_rp, 1, argList{Yi32, Yrb}},
	{Zibo_m, 2, argList{Yi32, Ymb}},
}

var ybtl = []ytab{
	{Zibo_m, 2, argList{Yi8, Yml}},
	{Zr_m, 1, argList{Yrl, Yml}},
}

var ymovw = []ytab{
	{Zr_m, 1, argList{Yrl, Yml}},
	{Zm_r, 1, argList{Yml, Yrl}},
	{Zil_rp, 1, argList{Yi32, Yrl}},
	{Zilo_m, 2, argList{Yi32, Yml}},
	{Zaut_r, 2, argList{Yiauto, Yrl}},
}

var ymovl = []ytab{
	{Zr_m, 1, argList{Yrl, Yml}},
	{Zm_r, 1, argList{Yml, Yrl}},
	{Zil_rp, 1, argList{Yi32, Yrl}},
	{Zilo_m, 2, argList{Yi32, Yml}},
	{Zm_r_xm, 1, argList{Yml, Ymr}}, // MMX MOVD
	{Zr_m_xm, 1, argList{Ymr, Yml}}, // MMX MOVD
	{Zm_r_xm, 2, argList{Yml, Yxr}}, // XMM MOVD (32 bit)
	{Zr_m_xm, 2, argList{Yxr, Yml}}, // XMM MOVD (32 bit)
	{Zaut_r, 2, argList{Yiauto, Yrl}},
}

var yret = []ytab{
	{Zo_iw, 1, argList{}},
	{Zo_iw, 1, argList{Yi32}},
}

var ymovq = []ytab{
	// valid in 32-bit mode
	{Zm_r_xm_nr, 1, argList{Ym, Ymr}},  // 0x6f MMX MOVQ (shorter encoding)
	{Zr_m_xm_nr, 1, argList{Ymr, Ym}},  // 0x7f MMX MOVQ
	{Zm_r_xm_nr, 2, argList{Yxr, Ymr}}, // Pf2, 0xd6 MOVDQ2Q
	{Zm_r_xm_nr, 2, argList{Yxm, Yxr}}, // Pf3, 0x7e MOVQ xmm1/m64 -> xmm2
	{Zr_m_xm_nr, 2, argList{Yxr, Yxm}}, // Pe, 0xd6 MOVQ xmm1 -> xmm2/m64

	// valid only in 64-bit mode, usually with 64-bit prefix
	{Zr_m, 1, argList{Yrl, Yml}},      // 0x89
	{Zm_r, 1, argList{Yml, Yrl}},      // 0x8b
	{Zilo_m, 2, argList{Ys32, Yrl}},   // 32 bit signed 0xc7,(0)
	{Ziq_rp, 1, argList{Yi64, Yrl}},   // 0xb8 -- 32/64 bit immediate
	{Zilo_m, 2, argList{Yi32, Yml}},   // 0xc7,(0)
	{Zm_r_xm, 1, argList{Ymm, Ymr}},   // 0x6e MMX MOVD
	{Zr_m_xm, 1, argList{Ymr, Ymm}},   // 0x7e MMX MOVD
	{Zm_r_xm, 2, argList{Yml, Yxr}},   // Pe, 0x6e MOVD xmm load
	{Zr_m_xm, 2, argList{Yxr, Yml}},   // Pe, 0x7e MOVD xmm store
	{Zaut_r, 1, argList{Yiauto, Yrl}}, // 0 built-in LEAQ
}

var ymovbe = []ytab{
	{Zlitm_r, 3, argList{Ym, Yrl}},
	{Zlitr_m, 3, argList{Yrl, Ym}},
}

var ym_rl = []ytab{
	{Zm_r, 1, argList{Ym, Yrl}},
}

var yrl_m = []ytab{
	{Zr_m, 1, argList{Yrl, Ym}},
}

var ymb_rl = []ytab{
	{Zmb_r, 1, argList{Ymb, Yrl}},
}

var yml_rl = []ytab{
	{Zm_r, 1, argList{Yml, Yrl}},
}

var yrl_ml = []ytab{
	{Zr_m, 1, argList{Yrl, Yml}},
}

var yml_mb = []ytab{
	{Zr_m, 1, argList{Yrb, Ymb}},
	{Zm_r, 1, argList{Ymb, Yrb}},
}

var yrb_mb = []ytab{
	{Zr_m, 1, argList{Yrb, Ymb}},
}

var yxchg = []ytab{
	{Z_rp, 1, argList{Yax, Yrl}},
	{Zrp_, 1, argList{Yrl, Yax}},
	{Zr_m, 1, argList{Yrl, Yml}},
	{Zm_r, 1, argList{Yml, Yrl}},
}

var ydivl = []ytab{
	{Zm_o, 2, argList{Yml}},
}

var ydivb = []ytab{
	{Zm_o, 2, argList{Ymb}},
}

var yimul = []ytab{
	{Zm_o, 2, argList{Yml}},
	{Zib_rr, 1, argList{Yi8, Yrl}},
	{Zil_rr, 1, argList{Yi32, Yrl}},
	{Zm_r, 2, argList{Yml, Yrl}},
}

var yimul3 = []ytab{
	{Zibm_r, 2, argList{Yi8, Yml, Yrl}},
	{Zibm_r, 2, argList{Yi32, Yml, Yrl}},
}

var ybyte = []ytab{
	{Zbyte, 1, argList{Yi64}},
}

var yin = []ytab{
	{Zib_, 1, argList{Yi32}},
	{Zlit, 1, argList{}},
}

var yint = []ytab{
	{Zib_, 1, argList{Yi32}},
}

var ypushl = []ytab{
	{Zrp_, 1, argList{Yrl}},
	{Zm_o, 2, argList{Ym}},
	{Zib_, 1, argList{Yi8}},
	{Zil_, 1, argList{Yi32}},
}

var ypopl = []ytab{
	{Z_rp, 1, argList{Yrl}},
	{Zo_m, 2, argList{Ym}},
}

var ywrfsbase = []ytab{
	{Zm_o, 2, argList{Yrl}},
}

var yrdrand = []ytab{
	{Zo_m, 2, argList{Yrl}},
}

var yclflush = []ytab{
	{Zo_m, 2, argList{Ym}},
}

var ybswap = []ytab{
	{Z_rp, 2, argList{Yrl}},
}

var yscond = []ytab{
	{Zo_m, 2, argList{Ymb}},
}

var yjcond = []ytab{
	{Zbr, 0, argList{Ybr}},
	{Zbr, 0, argList{Yi0, Ybr}},
	{Zbr, 1, argList{Yi1, Ybr}},
}

var yloop = []ytab{
	{Zloop, 1, argList{Ybr}},
}

var ycall = []ytab{
	{Zcallindreg, 0, argList{Yml}},
	{Zcallindreg, 2, argList{Yrx, Yrx}},
	{Zcallind, 2, argList{Yindir}},
	{Zcall, 0, argList{Ybr}},
	{Zcallcon, 1, argList{Yi32}},
}

var yduff = []ytab{
	{Zcallduff, 1, argList{Yi32}},
}

var yjmp = []ytab{
	{Zo_m64, 2, argList{Yml}},
	{Zjmp, 0, argList{Ybr}},
	{Zjmpcon, 1, argList{Yi32}},
}

var yfmvd = []ytab{
	{Zm_o, 2, argList{Ym, Yf0}},
	{Zo_m, 2, argList{Yf0, Ym}},
	{Zm_o, 2, argList{Yrf, Yf0}},
	{Zo_m, 2, argList{Yf0, Yrf}},
}

var yfmvdp = []ytab{
	{Zo_m, 2, argList{Yf0, Ym}},
	{Zo_m, 2, argList{Yf0, Yrf}},
}

var yfmvf = []ytab{
	{Zm_o, 2, argList{Ym, Yf0}},
	{Zo_m, 2, argList{Yf0, Ym}},
}

var yfmvx = []ytab{
	{Zm_o, 2, argList{Ym, Yf0}},
}

var yfmvp = []ytab{
	{Zo_m, 2, argList{Yf0, Ym}},
}

var yfcmv = []ytab{
	{Zm_o, 2, argList{Yrf, Yf0}},
}

var yfadd = []ytab{
	{Zm_o, 2, argList{Ym, Yf0}},
	{Zm_o, 2, argList{Yrf, Yf0}},
	{Zo_m, 2, argList{Yf0, Yrf}},
}

var yfxch = []ytab{
	{Zo_m, 2, argList{Yf0, Yrf}},
	{Zm_o, 2, argList{Yrf, Yf0}},
}

var ycompp = []ytab{
	{Zo_m, 2, argList{Yf0, Yrf}}, // botch is really f0,f1
}

var ystsw = []ytab{
	{Zo_m, 2, argList{Ym}},
	{Zlit, 1, argList{Yax}},
}

var ysvrs_mo = []ytab{
	{Zm_o, 2, argList{Ym}},
}

// unaryDst version of "ysvrs_mo".
var ysvrs_om = []ytab{
	{Zo_m, 2, argList{Ym}},
}

var ymm = []ytab{
	{Zm_r_xm, 1, argList{Ymm, Ymr}},
	{Zm_r_xm, 2, argList{Yxm, Yxr}},
}

var yxm = []ytab{
	{Zm_r_xm, 1, argList{Yxm, Yxr}},
}

var yxm_q4 = []ytab{
	{Zm_r, 1, argList{Yxm, Yxr}},
}

var yxcvm1 = []ytab{
	{Zm_r_xm, 2, argList{Yxm, Yxr}},
	{Zm_r_xm, 2, argList{Yxm, Ymr}},
}

var yxcvm2 = []ytab{
	{Zm_r_xm, 2, argList{Yxm, Yxr}},
	{Zm_r_xm, 2, argList{Ymm, Yxr}},
}

var yxr = []ytab{
	{Zm_r_xm, 1, argList{Yxr, Yxr}},
}

var yxr_ml = []ytab{
	{Zr_m_xm, 1, argList{Yxr, Yml}},
}

var ymr = []ytab{
	{Zm_r, 1, argList{Ymr, Ymr}},
}

var ymr_ml = []ytab{
	{Zr_m_xm, 1, argList{Ymr, Yml}},
}

var yxcmpi = []ytab{
	{Zm_r_i_xm, 2, argList{Yxm, Yxr, Yi8}},
}

var yxmov = []ytab{
	{Zm_r_xm, 1, argList{Yxm, Yxr}},
	{Zr_m_xm, 1, argList{Yxr, Yxm}},
}

var yxcvfl = []ytab{
	{Zm_r_xm, 1, argList{Yxm, Yrl}},
}

var yxcvlf = []ytab{
	{Zm_r_xm, 1, argList{Yml, Yxr}},
}

var yxcvfq = []ytab{
	{Zm_r_xm, 2, argList{Yxm, Yrl}},
}

var yxcvqf = []ytab{
	{Zm_r_xm, 2, argList{Yml, Yxr}},
}

var yps = []ytab{
	{Zm_r_xm, 1, argList{Ymm, Ymr}},
	{Zibo_m_xm, 2, argList{Yi8, Ymr}},
	{Zm_r_xm, 2, argList{Yxm, Yxr}},
	{Zibo_m_xm, 3, argList{Yi8, Yxr}},
}

var yxrrl = []ytab{
	{Zm_r, 1, argList{Yxr, Yrl}},
}

var ymrxr = []ytab{
	{Zm_r, 1, argList{Ymr, Yxr}},
	{Zm_r_xm, 1, argList{Yxm, Yxr}},
}

var ymshuf = []ytab{
	{Zibm_r, 2, argList{Yi8, Ymm, Ymr}},
}

var ymshufb = []ytab{
	{Zm2_r, 2, argList{Yxm, Yxr}},
}

// It should never have more than 1 entry,
// because some optab entries have opcode sequences that
// are longer than 2 bytes (zoffset=2 here),
// ROUNDPD and ROUNDPS and recently added BLENDPD,
// to name a few.
var yxshuf = []ytab{
	{Zibm_r, 2, argList{Yu8, Yxm, Yxr}},
}

var yextrw = []ytab{
	{Zibm_r, 2, argList{Yu8, Yxr, Yrl}},
	{Zibr_m, 2, argList{Yu8, Yxr, Yml}},
}

var yextr = []ytab{
	{Zibr_m, 3, argList{Yu8, Yxr, Ymm}},
}

var yinsrw = []ytab{
	{Zibm_r, 2, argList{Yu8, Yml, Yxr}},
}

var yinsr = []ytab{
	{Zibm_r, 3, argList{Yu8, Ymm, Yxr}},
}

var ypsdq = []ytab{
	{Zibo_m, 2, argList{Yi8, Yxr}},
}

var ymskb = []ytab{
	{Zm_r_xm, 2, argList{Yxr, Yrl}},
	{Zm_r_xm, 1, argList{Ymr, Yrl}},
}

var ycrc32l = []ytab{
	{Zlitm_r, 0, argList{Yml, Yrl}},
}

var ycrc32b = []ytab{
	{Zlitm_r, 0, argList{Ymb, Yrl}},
}

var yprefetch = []ytab{
	{Zm_o, 2, argList{Ym}},
}

var yaes = []ytab{
	{Zlitm_r, 2, argList{Yxm, Yxr}},
}

var yxbegin = []ytab{
	{Zjmp, 1, argList{Ybr}},
}

var yxabort = []ytab{
	{Zib_, 1, argList{Yu8}},
}

var ylddqu = []ytab{
	{Zm_r, 1, argList{Ym, Yxr}},
}

var ypalignr = []ytab{
	{Zibm_r, 2, argList{Yu8, Yxm, Yxr}},
}

var ysha256rnds2 = []ytab{
	{Zlit_m_r, 0, argList{Yxr0, Yxm, Yxr}},
}

var yblendvpd = []ytab{
	{Z_m_r, 1, argList{Yxr0, Yxm, Yxr}},
}

var ymmxmm0f38 = []ytab{
	{Zlitm_r, 3, argList{Ymm, Ymr}},
	{Zlitm_r, 5, argList{Yxm, Yxr}},
}

var yextractps = []ytab{
	{Zibr_m, 2, argList{Yu2, Yxr, Yml}},
}

var ysha1rnds4 = []ytab{
	{Zibm_r, 2, argList{Yu2, Yxm, Yxr}},
}

// You are doasm, holding in your hand a *obj.Prog with p.As set to, say,
// ACRC32, and p.From and p.To as operands (obj.Addr).  The linker scans optab
// to find the entry with the given p.As and then looks through the ytable for
// that instruction (the second field in the optab struct) for a line whose
// first two values match the Ytypes of the p.From and p.To operands.  The
// function oclass computes the specific Ytype of an operand and then the set
// of more general Ytypes that it satisfies is implied by the ycover table, set
// up in instinit.  For example, oclass distinguishes the constants 0 and 1
// from the more general 8-bit constants, but instinit says
//
//	ycover[Yi0*Ymax+Ys32] = 1
//	ycover[Yi1*Ymax+Ys32] = 1
//	ycover[Yi8*Ymax+Ys32] = 1
//
// which means that Yi0, Yi1, and Yi8 all count as Ys32 (signed 32)
// if that's what an instruction can handle.
//
// In parallel with the scan through the ytable for the appropriate line, there
// is a z pointer that starts out pointing at the strange magic byte list in
// the Optab struct.  With each step past a non-matching ytable line, z
// advances by the 4th entry in the line.  When a matching line is found, that
// z pointer has the extra data to use in laying down the instruction bytes.
// The actual bytes laid down are a function of the 3rd entry in the line (that
// is, the Ztype) and the z bytes.
//
// For example, let's look at AADDL.  The optab line says:
//
//	{AADDL, yaddl, Px, opBytes{0x83, 00, 0x05, 0x81, 00, 0x01, 0x03}},
//
// and yaddl says
//
//	var yaddl = []ytab{
//	        {Yi8, Ynone, Yml, Zibo_m, 2},
//	        {Yi32, Ynone, Yax, Zil_, 1},
//	        {Yi32, Ynone, Yml, Zilo_m, 2},
//	        {Yrl, Ynone, Yml, Zr_m, 1},
//	        {Yml, Ynone, Yrl, Zm_r, 1},
//	}
//
// so there are 5 possible types of ADDL instruction that can be laid down, and
// possible states used to lay them down (Ztype and z pointer, assuming z
// points at opBytes{0x83, 00, 0x05,0x81, 00, 0x01, 0x03}) are:
//
//	Yi8, Yml -> Zibo_m, z (0x83, 00)
//	Yi32, Yax -> Zil_, z+2 (0x05)
//	Yi32, Yml -> Zilo_m, z+2+1 (0x81, 0x00)
//	Yrl, Yml -> Zr_m, z+2+1+2 (0x01)
//	Yml, Yrl -> Zm_r, z+2+1+2+1 (0x03)
//
// The Pconstant in the optab line controls the prefix bytes to emit.  That's
// relatively straightforward as this program goes.
//
// The switch on yt.zcase in doasm implements the various Z cases.  Zibo_m, for
// example, is an opcode byte (z[0]) then an asmando (which is some kind of
// encoded addressing mode for the Yml arg), and then a single immediate byte.
// Zilo_m is the same but a long (32-bit) immediate.
var optab =
// as, ytab, andproto, opcode
[...]Optab{
	{obj.AXXX, nil, 0, opBytes{}},
	{AAAA, ynone, P32, opBytes{0x37}},
	{AAAD, ynone, P32, opBytes{0xd5, 0x0a}},
	{AAAM, ynone, P32, opBytes{0xd4, 0x0a}},
	{AAAS, ynone, P32, opBytes{0x3f}},
	{AADCB, yxorb, Pb, opBytes{0x14, 0x80, 02, 0x10, 0x12}},
	{AADCL, yaddl, Px, opBytes{0x83, 02, 0x15, 0x81, 02, 0x11, 0x13}},
	{AADCQ, yaddl, Pw, opBytes{0x83, 02, 0x15, 0x81, 02, 0x11, 0x13}},
	{AADCW, yaddl, Pe, opBytes{0x83, 02, 0x15, 0x81, 02, 0x11, 0x13}},
	{AADCXL, yml_rl, Pq4, opBytes{0xf6}},
	{AADCXQ, yml_rl, Pq4w, opBytes{0xf6}},
	{AADDB, yxorb, Pb, opBytes{0x04, 0x80, 00, 0x00, 0x02}},
	{AADDL, yaddl, Px, opBytes{0x83, 00, 0x05, 0x81, 00, 0x01, 0x03}},
	{AADDPD, yxm, Pq, opBytes{0x58}},
	{AADDPS, yxm, Pm, opBytes{0x58}},
	{AADDQ, yaddl, Pw, opBytes{0x83, 00, 0x05, 0x81, 00, 0x01, 0x03}},
	{AADDSD, yxm, Pf2, opBytes{0x58}},
	{AADDSS, yxm, Pf3, opBytes{0x58}},
	{AADDSUBPD, yxm, Pq, opBytes{0xd0}},
	{AADDSUBPS, yxm, Pf2, opBytes{0xd0}},
	{AADDW, yaddl, Pe, opBytes{0x83, 00, 0x05, 0x81, 00, 0x01, 0x03}},
	{AADOXL, yml_rl, Pq5, opBytes{0xf6}},
	{AADOXQ, yml_rl, Pq5w, opBytes{0xf6}},
	{AADJSP, nil, 0, opBytes{}},
	{AANDB, yxorb, Pb, opBytes{0x24, 0x80, 04, 0x20, 0x22}},
	{AANDL, yaddl, Px, opBytes{0x83, 04, 0x25, 0x81, 04, 0x21, 0x23}},
	{AANDNPD, yxm, Pq, opBytes{0x55}},
	{AANDNPS, yxm, Pm, opBytes{0x55}},
	{AANDPD, yxm, Pq, opBytes{0x54}},
	{AANDPS, yxm, Pm, opBytes{0x54}},
	{AANDQ, yaddl, Pw, opBytes{0x83, 04, 0x25, 0x81, 04, 0x21, 0x23}},
	{AANDW, yaddl, Pe, opBytes{0x83, 04, 0x25, 0x81, 04, 0x21, 0x23}},
	{AARPL, yrl_ml, P32, opBytes{0x63}},
	{ABOUNDL, yrl_m, P32, opBytes{0x62}},
	{ABOUNDW, yrl_m, Pe, opBytes{0x62}},
	{ABSFL, yml_rl, Pm, opBytes{0xbc}},
	{ABSFQ, yml_rl, Pw, opBytes{0x0f, 0xbc}},
	{ABSFW, yml_rl, Pq, opBytes{0xbc}},
	{ABSRL, yml_rl, Pm, opBytes{0xbd}},
	{ABSRQ, yml_rl, Pw, opBytes{0x0f, 0xbd}},
	{ABSRW, yml_rl, Pq, opBytes{0xbd}},
	{ABSWAPL, ybswap, Px, opBytes{0x0f, 0xc8}},
	{ABSWAPQ, ybswap, Pw, opBytes{0x0f, 0xc8}},
	{ABTCL, ybtl, Pm, opBytes{0xba, 07, 0xbb}},
	{ABTCQ, ybtl, Pw, opBytes{0x0f, 0xba, 07, 0x0f, 0xbb}},
	{ABTCW, ybtl, Pq, opBytes{0xba, 07, 0xbb}},
	{ABTL, ybtl, Pm, opBytes{0xba, 04, 0xa3}},
	{ABTQ, ybtl, Pw, opBytes{0x0f, 0xba, 04, 0x0f, 0xa3}},
	{ABTRL, ybtl, Pm, opBytes{0xba, 06, 0xb3}},
	{ABTRQ, ybtl, Pw, opBytes{0x0f, 0xba, 06, 0x0f, 0xb3}},
	{ABTRW, ybtl, Pq, opBytes{0xba, 06, 0xb3}},
	{ABTSL, ybtl, Pm, opBytes{0xba, 05, 0xab}},
	{ABTSQ, ybtl, Pw, opBytes{0x0f, 0xba, 05, 0x0f, 0xab}},
	{ABTSW, ybtl, Pq, opBytes{0xba, 05, 0xab}},
	{ABTW, ybtl, Pq, opBytes{0xba, 04, 0xa3}},
	{ABYTE, ybyte, Px, opBytes{1}},
	{obj.ACALL, ycall, Px, opBytes{0xff, 02, 0xff, 0x15, 0xe8}},
	{ACBW, ynone, Pe, opBytes{0x98}},
	{ACDQ, ynone, Px, opBytes{0x99}},
	{ACDQE, ynone, Pw, opBytes{0x98}},
	{ACLAC, ynone, Pm, opBytes{01, 0xca}},
	{ACLC, ynone, Px, opBytes{0xf8}},
	{ACLD, ynone, Px, opBytes{0xfc}},
	{ACLDEMOTE, yclflush, Pm, opBytes{0x1c, 00}},
	{ACLFLUSH, yclflush, Pm, opBytes{0xae, 07}},
	{ACLFLUSHOPT, yclflush, Pq, opBytes{0xae, 07}},
	{ACLI, ynone, Px, opBytes{0xfa}},
	{ACLTS, ynone, Pm, opBytes{0x06}},
	{ACLWB, yclflush, Pq, opBytes{0xae, 06}},
	{ACMC, ynone, Px, opBytes{0xf5}},
	{ACMOVLCC, yml_rl, Pm, opBytes{0x43}},
	{ACMOVLCS, yml_rl, Pm, opBytes{0x42}},
	{ACMOVLEQ, yml_rl, Pm, opBytes{0x44}},
	{ACMOVLGE, yml_rl, Pm, opBytes{0x4d}},
	{ACMOVLGT, yml_rl, Pm, opBytes{0x4f}},
	{ACMOVLHI, yml_rl, Pm, opBytes{0x47}},
	{ACMOVLLE, yml_rl, Pm, opBytes{0x4e}},
	{ACMOVLLS, yml_rl, Pm, opBytes{0x46}},
	{ACMOVLLT, yml_rl, Pm, opBytes{0x4c}},
	{ACMOVLMI, yml_rl, Pm, opBytes{0x48}},
	{ACMOVLNE, yml_rl, Pm, opBytes{0x45}},
	{ACMOVLOC, yml_rl, Pm, opBytes{0x41}},
	{ACMOVLOS, yml_rl, Pm, opBytes{0x40}},
	{ACMOVLPC, yml_rl, Pm, opBytes{0x4b}},
	{ACMOVLPL, yml_rl, Pm, opBytes{0x49}},
	{ACMOVLPS, yml_rl, Pm, opBytes{0x4a}},
	{ACMOVQCC, yml_rl, Pw, opBytes{0x0f, 0x43}},
	{ACMOVQCS, yml_rl, Pw, opBytes{0x0f, 0x42}},
	{ACMOVQEQ, yml_rl, Pw, opBytes{0x0f, 0x44}},
	{ACMOVQGE, yml_rl, Pw, opBytes{0x0f, 0x4d}},
	{ACMOVQGT, yml_rl, Pw, opBytes{0x0f, 0x4f}},
	{ACMOVQHI, yml_rl, Pw, opBytes{0x0f, 0x47}},
	{ACMOVQLE, yml_rl, Pw, opBytes{0x0f, 0x4e}},
	{ACMOVQLS, yml_rl, Pw, opBytes{0x0f, 0x46}},
	{ACMOVQLT, yml_rl, Pw, opBytes{0x0f, 0x4c}},
	{ACMOVQMI, yml_rl, Pw, opBytes{0x0f, 0x48}},
	{ACMOVQNE, yml_rl, Pw, opBytes{0x0f, 0x45}},
	{ACMOVQOC, yml_rl, Pw, opBytes{0x0f, 0x41}},
	{ACMOVQOS, yml_rl, Pw, opBytes{0x0f, 0x40}},
	{ACMOVQPC, yml_rl, Pw, opBytes{0x0f, 0x4b}},
	{ACMOVQPL, yml_rl, Pw, opBytes{0x0f, 0x49}},
	{ACMOVQPS, yml_rl, Pw, opBytes{0x0f, 0x4a}},
	{ACMOVWCC, yml_rl, Pq, opBytes{0x43}},
	{ACMOVWCS, yml_rl, Pq, opBytes{0x42}},
	{ACMOVWEQ, yml_rl, Pq, opBytes{0x44}},
	{ACMOVWGE, yml_rl, Pq, opBytes{0x4d}},
	{ACMOVWGT, yml_rl, Pq, opBytes{0x4f}},
	{ACMOVWHI, yml_rl, Pq, opBytes{0x47}},
	{ACMOVWLE, yml_rl, Pq, opBytes{0x4e}},
	{ACMOVWLS, yml_rl, Pq, opBytes{0x46}},
	{ACMOVWLT, yml_rl, Pq, opBytes{0x4c}},
	{ACMOVWMI, yml_rl, Pq, opBytes{0x48}},
	{ACMOVWNE, yml_rl, Pq, opBytes{0x45}},
	{ACMOVWOC, yml_rl, Pq, opBytes{0x41}},
	{ACMOVWOS, yml_rl, Pq, opBytes{0x40}},
	{ACMOVWPC, yml_rl, Pq, opBytes{0x4b}},
	{ACMOVWPL, yml_rl, Pq, opBytes{0x49}},
	{ACMOVWPS, yml_rl, Pq, opBytes{0x4a}},
	{ACMPB, ycmpb, Pb, opBytes{0x3c, 0x80, 07, 0x38, 0x3a}},
	{ACMPL, ycmpl, Px, opBytes{0x83, 07, 0x3d, 0x81, 07, 0x39, 0x3b}},
	{ACMPPD, yxcmpi, Px, opBytes{Pe, 0xc2}},
	{ACMPPS, yxcmpi, Pm, opBytes{0xc2, 0}},
	{ACMPQ, ycmpl, Pw, opBytes{0x83, 07, 0x3d, 0x81, 07, 0x39, 0x3b}},
	{ACMPSB, ynone, Pb, opBytes{0xa6}},
	{ACMPSD, yxcmpi, Px, opBytes{Pf2, 0xc2}},
	{ACMPSL, ynone, Px, opBytes{0xa7}},
	{ACMPSQ, ynone, Pw, opBytes{0xa7}},
	{ACMPSS, yxcmpi, Px, opBytes{Pf3, 0xc2}},
	{ACMPSW, ynone, Pe, opBytes{0xa7}},
	{ACMPW, ycmpl, Pe, opBytes{0x83, 07, 0x3d, 0x81, 07, 0x39, 0x3b}},
	{ACOMISD, yxm, Pe, opBytes{0x2f}},
	{ACOMISS, yxm, Pm, opBytes{0x2f}},
	{ACPUID, ynone, Pm, opBytes{0xa2}},
	{ACVTPL2PD, yxcvm2, Px, opBytes{Pf3, 0xe6, Pe, 0x2a}},
	{ACVTPL2PS, yxcvm2, Pm, opBytes{0x5b, 0, 0x2a, 0}},
	{ACVTPD2PL, yxcvm1, Px, opBytes{Pf2, 0xe6, Pe, 0x2d}},
	{ACVTPD2PS, yxm, Pe, opBytes{0x5a}},
	{ACVTPS2PL, yxcvm1, Px, opBytes{Pe, 0x5b, Pm, 0x2d}},
	{ACVTPS2PD, yxm, Pm, opBytes{0x5a}},
	{ACVTSD2SL, yxcvfl, Pf2, opBytes{0x2d}},
	{ACVTSD2SQ, yxcvfq, Pw, opBytes{Pf2, 0x2d}},
	{ACVTSD2SS, yxm, Pf2, opBytes{0x5a}},
	{ACVTSL2SD, yxcvlf, Pf2, opBytes{0x2a}},
	{ACVTSQ2SD, yxcvqf, Pw, opBytes{Pf2, 0x2a}},
	{ACVTSL2SS, yxcvlf, Pf3, opBytes{0x2a}},
	{ACVTSQ2SS, yxcvqf, Pw, opBytes{Pf3, 0x2a}},
	{ACVTSS2SD, yxm, Pf3, opBytes{0x5a}},
	{ACVTSS2SL, yxcvfl, Pf3, opBytes{0x2d}},
	{ACVTSS2SQ, yxcvfq, Pw, opBytes{Pf3, 0x2d}},
	{ACVTTPD2PL, yxcvm1, Px, opBytes{Pe, 0xe6, Pe, 0x2c}},
	{ACVTTPS2PL, yxcvm1, Px, opBytes{Pf3, 0x5b, Pm, 0x2c}},
	{ACVTTSD2SL, yxcvfl, Pf2, opBytes{0x2c}},
	{ACVTTSD2SQ, yxcvfq, Pw, opBytes{Pf2, 0x2c}},
	{ACVTTSS2SL, yxcvfl, Pf3, opBytes{0x2c}},
	{ACVTTSS2SQ, yxcvfq, Pw, opBytes{Pf3, 0x2c}},
	{ACWD, ynone, Pe, opBytes{0x99}},
	{ACWDE, ynone, Px, opBytes{0x98}},
	{ACQO, ynone, Pw, opBytes{0x99}},
	{ADAA, ynone, P32, opBytes{0x27}},
	{ADAS, ynone, P32, opBytes{0x2f}},
	{ADECB, yscond, Pb, opBytes{0xfe, 01}},
	{ADECL, yincl, Px1, opBytes{0x48, 0xff, 01}},
	{ADECQ, yincq, Pw, opBytes{0xff, 01}},
	{ADECW, yincq, Pe, opBytes{0xff, 01}},
	{ADIVB, ydivb, Pb, opBytes{0xf6, 06}},
	{ADIVL, ydivl, Px, opBytes{0xf7, 06}},
	{ADIVPD, yxm, Pe, opBytes{0x5e}},
	{ADIVPS, yxm, Pm, opBytes{0x5e}},
	{ADIVQ, ydivl, Pw, opBytes{0xf7, 06}},
	{ADIVSD, yxm, Pf2, opBytes{0x5e}},
	{ADIVSS, yxm, Pf3, opBytes{0x5e}},
	{ADIVW, ydivl, Pe, opBytes{0xf7, 06}},
	{ADPPD, yxshuf, Pq, opBytes{0x3a, 0x41, 0}},
	{ADPPS, yxshuf, Pq, opBytes{0x3a, 0x40, 0}},
	{AEMMS, ynone, Pm, opBytes{0x77}},
	{AEXTRACTPS, yextractps, Pq, opBytes{0x3a, 0x17, 0}},
	{AENTER, nil, 0, opBytes{}}, // botch
	{AFXRSTOR, ysvrs_mo, Pm, opBytes{0xae, 01, 0xae, 01}},
	{AFXSAVE, ysvrs_om, Pm, opBytes{0xae, 00, 0xae, 00}},
	{AFXRSTOR64, ysvrs_mo, Pw, opBytes{0x0f, 0xae, 01, 0x0f, 0xae, 01}},
	{AFXSAVE64, ysvrs_om, Pw, opBytes{0x0f, 0xae, 00, 0x0f, 0xae, 00}},
	{AHLT, ynone, Px, opBytes{0xf4}},
	{AIDIVB, ydivb, Pb, opBytes{0xf6, 07}},
	{AIDIVL, ydivl, Px, opBytes{0xf7, 07}},
	{AIDIVQ, ydivl, Pw, opBytes{0xf7, 07}},
	{AIDIVW, ydivl, Pe, opBytes{0xf7, 07}},
	{AIMULB, ydivb, Pb, opBytes{0xf6, 05}},
	{AIMULL, yimul, Px, opBytes{0xf7, 05, 0x6b, 0x69, Pm, 0xaf}},
	{AIMULQ, yimul, Pw, opBytes{0xf7, 05, 0x6b, 0x69, Pm, 0xaf}},
	{AIMULW, yimul, Pe, opBytes{0xf7, 05, 0x6b, 0x69, Pm, 0xaf}},
	{AIMUL3W, yimul3, Pe, opBytes{0x6b, 00, 0x69, 00}},
	{AIMUL3L, yimul3, Px, opBytes{0x6b, 00, 0x69, 00}},
	{AIMUL3Q, yimul3, Pw, opBytes{0x6b, 00, 0x69, 00}},
	{AINB, yin, Pb, opBytes{0xe4, 0xec}},
	{AINW, yin, Pe, opBytes{0xe5, 0xed}},
	{AINL, yin, Px, opBytes{0xe5, 0xed}},
	{AINCB, yscond, Pb, opBytes{0xfe, 00}},
	{AINCL, yincl, Px1, opBytes{0x40, 0xff, 00}},
	{AINCQ, yincq, Pw, opBytes{0xff, 00}},
	{AINCW, yincq, Pe, opBytes{0xff, 00}},
	{AINSB, ynone, Pb, opBytes{0x6c}},
	{AINSL, ynone, Px, opBytes{0x6d}},
	{AINSERTPS, yxshuf, Pq, opBytes{0x3a, 0x21, 0}},
	{AINSW, ynone, Pe, opBytes{0x6d}},
	{AICEBP, ynone, Px, opBytes{0xf1}},
	{AINT, yint, Px, opBytes{0xcd}},
	{AINTO, ynone, P32, opBytes{0xce}},
	{AIRETL, ynone, Px, opBytes{0xcf}},
	{AIRETQ, ynone, Pw, opBytes{0xcf}},
	{AIRETW, ynone, Pe, opBytes{0xcf}},
	{AJCC, yjcond, Px, opBytes{0x73, 0x83, 00}},
	{AJCS, yjcond, Px, opBytes{0x72, 0x82}},
	{AJCXZL, yloop, Px, opBytes{0xe3}},
	{AJCXZW, yloop, Px, opBytes{0xe3}},
	{AJCXZQ, yloop, Px, opBytes{0xe3}},
	{AJEQ, yjcond, Px, opBytes{0x74, 0x84}},
	{AJGE, yjcond, Px, opBytes{0x7d, 0x8d}},
	{AJGT, yjcond, Px, opBytes{0x7f, 0x8f}},
	{AJHI, yjcond, Px, opBytes{0x77, 0x87}},
	{AJLE, yjcond, Px, opBytes{0x7e, 0x8e}},
	{AJLS, yjcond, Px, opBytes{0x76, 0x86}},
	{AJLT, yjcond, Px, opBytes{0x7c, 0x8c}},
	{AJMI, yjcond, Px, opBytes{0x78, 0x88}},
	{obj.AJMP, yjmp, Px, opBytes{0xff, 04, 0xeb, 0xe9}},
	{AJNE, yjcond, Px, opBytes{0x75, 0x85}},
	{AJOC, yjcond, Px, opBytes{0x71, 0x81, 00}},
	{AJOS, yjcond, Px, opBytes{0x70, 0x80, 00}},
	{AJPC, yjcond, Px, opBytes{0x7b, 0x8b}},
	{AJPL, yjcond, Px, opBytes{0x79, 0x89}},
	{AJPS, yjcond, Px, opBytes{0x7a, 0x8a}},
	{AHADDPD, yxm, Pq, opBytes{0x7c}},
	{AHADDPS, yxm, Pf2, opBytes{0x7c}},
	{AHSUBPD, yxm, Pq, opBytes{0x7d}},
	{AHSUBPS, yxm, Pf2, opBytes{0x7d}},
	{ALAHF, ynone, Px, opBytes{0x9f}},
	{ALARL, yml_rl, Pm, opBytes{0x02}},
	{ALARQ, yml_rl, Pw, opBytes{0x0f, 0x02}},
	{ALARW, yml_rl, Pq, opBytes{0x02}},
	{ALDDQU, ylddqu, Pf2, opBytes{0xf0}},
	{ALDMXCSR, ysvrs_mo, Pm, opBytes{0xae, 02, 0xae, 02}},
	{ALEAL, ym_rl, Px, opBytes{0x8d}},
	{ALEAQ, ym_rl, Pw, opBytes{0x8d}},
	{ALEAVEL, ynone, P32, opBytes{0xc9}},
	{ALEAVEQ, ynone, Py, opBytes{0xc9}},
	{ALEAVEW, ynone, Pe, opBytes{0xc9}},
	{ALEAW, ym_rl, Pe, opBytes{0x8d}},
	{ALOCK, ynone, Px, opBytes{0xf0}},
	{ALODSB, ynone, Pb, opBytes{0xac}},
	{ALODSL, ynone, Px, opBytes{0xad}},
	{ALODSQ, ynone, Pw, opBytes{0xad}},
	{ALODSW, ynone, Pe, opBytes{0xad}},
	{ALONG, ybyte, Px, opBytes{4}},
	{ALOOP, yloop, Px, opBytes{0xe2}},
	{ALOOPEQ, yloop, Px, opBytes{0xe1}},
	{ALOOPNE, yloop, Px, opBytes{0xe0}},
	{ALTR, ydivl, Pm, opBytes{0x00, 03}},
	{ALZCNTL, yml_rl, Pf3, opBytes{0xbd}},
	{ALZCNTQ, yml_rl, Pfw, opBytes{0xbd}},
	{ALZCNTW, yml_rl, Pef3, opBytes{0xbd}},
	{ALSLL, yml_rl, Pm, opBytes{0x03}},
	{ALSLW, yml_rl, Pq, opBytes{0x03}},
	{ALSLQ, yml_rl, Pw, opBytes{0x0f, 0x03}},
	{AMASKMOVOU, yxr, Pe, opBytes{0xf7}},
	{AMASKMOVQ, ymr, Pm, opBytes{0xf7}},
	{AMAXPD, yxm, Pe, opBytes{0x5f}},
	{AMAXPS, yxm, Pm, opBytes{0x5f}},
	{AMAXSD, yxm, Pf2, opBytes{0x5f}},
	{AMAXSS, yxm, Pf3, opBytes{0x5f}},
	{AMINPD, yxm, Pe, opBytes{0x5d}},
	{AMINPS, yxm, Pm, opBytes{0x5d}},
	{AMINSD, yxm, Pf2, opBytes{0x5d}},
	{AMINSS, yxm, Pf3, opBytes{0x5d}},
	{AMONITOR, ynone, Px, opBytes{0x0f, 0x01, 0xc8, 0}},
	{AMWAIT, ynone, Px, opBytes{0x0f, 0x01, 0xc9, 0}},
	{AMOVAPD, yxmov, Pe, opBytes{0x28, 0x29}},
	{AMOVAPS, yxmov, Pm, opBytes{0x28, 0x29}},
	{AMOVB, ymovb, Pb, opBytes{0x88, 0x8a, 0xb0, 0xc6, 00}},
	{AMOVBLSX, ymb_rl, Pm, opBytes{0xbe}},
	{AMOVBLZX, ymb_rl, Pm, opBytes{0xb6}},
	{AMOVBQSX, ymb_rl, Pw, opBytes{0x0f, 0xbe}},
	{AMOVBQZX, ymb_rl, Pw, opBytes{0x0f, 0xb6}},
	{AMOVBWSX, ymb_rl, Pq, opBytes{0xbe}},
	{AMOVSWW, ymb_rl, Pe, opBytes{0x0f, 0xbf}},
	{AMOVBWZX, ymb_rl, Pq, opBytes{0xb6}},
	{AMOVZWW, ymb_rl, Pe, opBytes{0x0f, 0xb7}},
	{AMOVO, yxmov, Pe, opBytes{0x6f, 0x7f}},
	{AMOVOU, yxmov, Pf3, opBytes{0x6f, 0x7f}},
	{AMOVHLPS, yxr, Pm, opBytes{0x12}},
	{AMOVHPD, yxmov, Pe, opBytes{0x16, 0x17}},
	{AMOVHPS, yxmov, Pm, opBytes{0x16, 0x17}},
	{AMOVL, ymovl, Px, opBytes{0x89, 0x8b, 0xb8, 0xc7, 00, 0x6e, 0x7e, Pe, 0x6e, Pe, 0x7e, 0}},
	{AMOVLHPS, yxr, Pm, opBytes{0x16}},
	{AMOVLPD, yxmov, Pe, opBytes{0x12, 0x13}},
	{AMOVLPS, yxmov, Pm, opBytes{0x12, 0x13}},
	{AMOVLQSX, yml_rl, Pw, opBytes{0x63}},
	{AMOVLQZX, yml_rl, Px, opBytes{0x8b}},
	{AMOVMSKPD, yxrrl, Pq, opBytes{0x50}},
	{AMOVMSKPS, yxrrl, Pm, opBytes{0x50}},
	{AMOVNTO, yxr_ml, Pe, opBytes{0xe7}},
	{AMOVNTDQA, ylddqu, Pq4, opBytes{0x2a}},
	{AMOVNTPD, yxr_ml, Pe, opBytes{0x2b}},
	{AMOVNTPS, yxr_ml, Pm, opBytes{0x2b}},
	{AMOVNTQ, ymr_ml, Pm, opBytes{0xe7}},
	{AMOVQ, ymovq, Pw8, opBytes{0x6f, 0x7f, Pf2, 0xd6, Pf3, 0x7e, Pe, 0xd6, 0x89, 0x8b, 0xc7, 00, 0xb8, 0xc7, 00, 0x6e, 0x7e, Pe, 0x6e, Pe, 0x7e, 0}},
	{AMOVQOZX, ymrxr, Pf3, opBytes{0xd6, 0x7e}},
	{AMOVSB, ynone, Pb, opBytes{0xa4}},
	{AMOVSD, yxmov, Pf2, opBytes{0x10, 0x11}},
	{AMOVSL, ynone, Px, opBytes{0xa5}},
	{AMOVSQ, ynone, Pw, opBytes{0xa5}},
	{AMOVSS, yxmov, Pf3, opBytes{0x10, 0x11}},
	{AMOVSW, ynone, Pe, opBytes{0xa5}},
	{AMOVUPD, yxmov, Pe, opBytes{0x10, 0x11}},
	{AMOVUPS, yxmov, Pm, opBytes{0x10, 0x11}},
	{AMOVW, ymovw, Pe, opBytes{0x89, 0x8b, 0xb8, 0xc7, 00, 0}},
	{AMOVWLSX, yml_rl, Pm, opBytes{0xbf}},
	{AMOVWLZX, yml_rl, Pm, opBytes{0xb7}},
	{AMOVWQSX, yml_rl, Pw, opBytes{0x0f, 0xbf}},
	{AMOVWQZX, yml_rl, Pw, opBytes{0x0f, 0xb7}},
	{AMPSADBW, yxshuf, Pq, opBytes{0x3a, 0x42, 0}},
	{AMULB, ydivb, Pb, opBytes{0xf6, 04}},
	{AMULL, ydivl, Px, opBytes{0xf7, 04}},
	{AMULPD, yxm, Pe, opBytes{0x59}},
	{AMULPS, yxm, Ym, opBytes{0x59}},
	{AMULQ, ydivl, Pw, opBytes{0xf7, 04}},
	{AMULSD, yxm, Pf2, opBytes{0x59}},
	{AMULSS, yxm, Pf3, opBytes{0x59}},
	{AMULW, ydivl, Pe, opBytes{0xf7, 04}},
	{ANEGB, yscond, Pb, opBytes{0xf6, 03}},
	{ANEGL, yscond, Px, opBytes{0xf7, 03}},
	{ANEGQ, yscond, Pw, opBytes{0xf7, 03}},
	{ANEGW, yscond, Pe, opBytes{0xf7, 03}},
	{obj.ANOP, ynop, Px, opBytes{0, 0}},
	{ANOTB, yscond, Pb, opBytes{0xf6, 02}},
	{ANOTL, yscond, Px, opBytes{0xf7, 02}}, // TODO(rsc): yscond is wrong here.
	{ANOTQ, yscond, Pw, opBytes{0xf7, 02}},
	{ANOTW, yscond, Pe, opBytes{0xf7, 02}},
	{AORB, yxorb, Pb, opBytes{0x0c, 0x80, 01, 0x08, 0x0a}},
	{AORL, yaddl, Px, opBytes{0x83, 01, 0x0d, 0x81, 01, 0x09, 0x0b}},
	{AORPD, yxm, Pq, opBytes{0x56}},
	{AORPS, yxm, Pm, opBytes{0x56}},
	{AORQ, yaddl, Pw, opBytes{0x83, 01, 0x0d, 0x81, 01, 0x09, 0x0b}},
	{AORW, yaddl, Pe, opBytes{0x83, 01, 0x0d, 0x81, 01, 0x09, 0x0b}},
	{AOUTB, yin, Pb, opBytes{0xe6, 0xee}},
	{AOUTL, yin, Px, opBytes{0xe7, 0xef}},
	{AOUTW, yin, Pe, opBytes{0xe7, 0xef}},
	{AOUTSB, ynone, Pb, opBytes{0x6e}},
	{AOUTSL, ynone, Px, opBytes{0x6f}},
	{AOUTSW, ynone, Pe, opBytes{0x6f}},
	{APABSB, yxm_q4, Pq4, opBytes{0x1c}},
	{APABSD, yxm_q4, Pq4, opBytes{0x1e}},
	{APABSW, yxm_q4, Pq4, opBytes{0x1d}},
	{APACKSSLW, ymm, Py1, opBytes{0x6b, Pe, 0x6b}},
	{APACKSSWB, ymm, Py1, opBytes{0x63, Pe, 0x63}},
	{APACKUSDW, yxm_q4, Pq4, opBytes{0x2b}},
	{APACKUSWB, ymm, Py1, opBytes{0x67, Pe, 0x67}},
	{APADDB, ymm, Py1, opBytes{0xfc, Pe, 0xfc}},
	{APADDL, ymm, Py1, opBytes{0xfe, Pe, 0xfe}},
	{APADDQ, yxm, Pe, opBytes{0xd4}},
	{APADDSB, ymm, Py1, opBytes{0xec, Pe, 0xec}},
	{APADDSW, ymm, Py1, opBytes{0xed, Pe, 0xed}},
	{APADDUSB, ymm, Py1, opBytes{0xdc, Pe, 0xdc}},
	{APADDUSW, ymm, Py1, opBytes{0xdd, Pe, 0xdd}},
	{APADDW, ymm, Py1, opBytes{0xfd, Pe, 0xfd}},
	{APALIGNR, ypalignr, Pq, opBytes{0x3a, 0x0f}},
	{APAND, ymm, Py1, opBytes{0xdb, Pe, 0xdb}},
	{APANDN, ymm, Py1, opBytes{0xdf, Pe, 0xdf}},
	{APAUSE, ynone, Px, opBytes{0xf3, 0x90}},
	{APAVGB, ymm, Py1, opBytes{0xe0, Pe, 0xe0}},
	{APAVGW, ymm, Py1, opBytes{0xe3, Pe, 0xe3}},
	{APBLENDW, yxshuf, Pq, opBytes{0x3a, 0x0e, 0}},
	{APCMPEQB, ymm, Py1, opBytes{0x74, Pe, 0x74}},
	{APCMPEQL, ymm, Py1, opBytes{0x76, Pe, 0x76}},
	{APCMPEQQ, yxm_q4, Pq4, opBytes{0x29}},
	{APCMPEQW, ymm, Py1, opBytes{0x75, Pe, 0x75}},
	{APCMPGTB, ymm, Py1, opBytes{0x64, Pe, 0x64}},
	{APCMPGTL, ymm, Py1, opBytes{0x66, Pe, 0x66}},
	{APCMPGTQ, yxm_q4, Pq4, opBytes{0x37}},
	{APCMPGTW, ymm, Py1, opBytes{0x65, Pe, 0x65}},
	{APCMPISTRI, yxshuf, Pq, opBytes{0x3a, 0x63, 0}},
	{APCMPISTRM, yxshuf, Pq, opBytes{0x3a, 0x62, 0}},
	{APEXTRW, yextrw, Pq, opBytes{0xc5, 0, 0x3a, 0x15, 0}},
	{APEXTRB, yextr, Pq, opBytes{0x3a, 0x14, 00}},
	{APEXTRD, yextr, Pq, opBytes{0x3a, 0x16, 00}},
	{APEXTRQ, yextr, Pq3, opBytes{0x3a, 0x16, 00}},
	{APHADDD, ymmxmm0f38, Px, opBytes{0x0F, 0x38, 0x02, 0, 0x66, 0x0F, 0x38, 0x02, 0}},
	{APHADDSW, yxm_q4, Pq4, opBytes{0x03}},
	{APHADDW, yxm_q4, Pq4, opBytes{0x01}},
	{APHMINPOSUW, yxm_q4, Pq4, opBytes{0x41}},
	{APHSUBD, yxm_q4, Pq4, opBytes{0x06}},
	{APHSUBSW, yxm_q4, Pq4, opBytes{0x07}},
	{APHSUBW, yxm_q4, Pq4, opBytes{0x05}},
	{APINSRW, yinsrw, Pq, opBytes{0xc4, 00}},
	{APINSRB, yinsr, Pq, opBytes{0x3a, 0x20, 00}},
	{APINSRD, yinsr, Pq, opBytes{0x3a, 0x22, 00}},
	{APINSRQ, yinsr, Pq3, opBytes{0x3a, 0x22, 00}},
	{APMADDUBSW, yxm_q4, Pq4, opBytes{0x04}},
	{APMADDWL, ymm, Py1, opBytes{0xf5, Pe, 0xf5}},
	{APMAXSB, yxm_q4, Pq4, opBytes{0x3c}},
	{APMAXSD, yxm_q4, Pq4, opBytes{0x3d}},
	{APMAXSW, yxm, Pe, opBytes{0xee}},
	{APMAXUB, yxm, Pe, opBytes{0xde}},
	{APMAXUD, yxm_q4, Pq4, opBytes{0x3f}},
	{APMAXUW, yxm_q4, Pq4, opBytes{0x3e}},
	{APMINSB, yxm_q4, Pq4, opBytes{0x38}},
	{APMINSD, yxm_q4, Pq4, opBytes{0x39}},
	{APMINSW, yxm, Pe, opBytes{0xea}},
	{APMINUB, yxm, Pe, opBytes{0xda}},
	{APMINUD, yxm_q4, Pq4, opBytes{0x3b}},
	{APMINUW, yxm_q4, Pq4, opBytes{0x3a}},
	{APMOVMSKB, ymskb, Px, opBytes{Pe, 0xd7, 0xd7}},
	{APMOVSXBD, yxm_q4, Pq4, opBytes{0x21}},
	{APMOVSXBQ, yxm_q4, Pq4, opBytes{0x22}},
	{APMOVSXBW, yxm_q4, Pq4, opBytes{0x20}},
	{APMOVSXDQ, yxm_q4, Pq4, opBytes{0x25}},
	{APMOVSXWD, yxm_q4, Pq4, opBytes{0x23}},
	{APMOVSXWQ, yxm_q4, Pq4, opBytes{0x24}},
	{APMOVZXBD, yxm_q4, Pq4, opBytes{0x31}},
	{APMOVZXBQ, yxm_q4, Pq4, opBytes{0x32}},
	{APMOVZXBW, yxm_q4, Pq4, opBytes{0x30}},
	{APMOVZXDQ, yxm_q4, Pq4, opBytes{0x35}},
	{APMOVZXWD, yxm_q4, Pq4, opBytes{0x33}},
	{APMOVZXWQ, yxm_q4, Pq4, opBytes{0x34}},
	{APMULDQ, yxm_q4, Pq4, opBytes{0x28}},
	{APMULHRSW, yxm_q4, Pq4, opBytes{0x0b}},
	{APMULHUW, ymm, Py1, opBytes{0xe4, Pe, 0xe4}},
	{APMULHW, ymm, Py1, opBytes{0xe5, Pe, 0xe5}},
	{APMULLD, yxm_q4, Pq4, opBytes{0x40}},
	{APMULLW, ymm, Py1, opBytes{0xd5, Pe, 0xd5}},
	{APMULULQ, ymm, Py1, opBytes{0xf4, Pe, 0xf4}},
	{APOPAL, ynone, P32, opBytes{0x61}},
	{APOPAW, ynone, Pe, opBytes{0x61}},
	{APOPCNTW, yml_rl, Pef3, opBytes{0xb8}},
	{APOPCNTL, yml_rl, Pf3, opBytes{0xb8}},
	{APOPCNTQ, yml_rl, Pfw, opBytes{0xb8}},
	{APOPFL, ynone, P32, opBytes{0x9d}},
	{APOPFQ, ynone, Py, opBytes{0x9d}},
	{APOPFW, ynone, Pe, opBytes{0x9d}},
	{APOPL, ypopl, P32, opBytes{0x58, 0x8f, 00}},
	{APOPQ, ypopl, Py, opBytes{0x58, 0x8f, 00}},
	{APOPW, ypopl, Pe, opBytes{0x58, 0x8f, 00}},
	{APOR, ymm, Py1, opBytes{0xeb, Pe, 0xeb}},
	{APSADBW, yxm, Pq, opBytes{0xf6}},
	{APSHUFHW, yxshuf, Pf3, opBytes{0x70, 00}},
	{APSHUFL, yxshuf, Pq, opBytes{0x70, 00}},
	{APSHUFLW, yxshuf, Pf2, opBytes{0x70, 00}},
	{APSHUFW, ymshuf, Pm, opBytes{0x70, 00}},
	{APSHUFB, ymshufb, Pq, opBytes{0x38, 0x00}},
	{APSIGNB, yxm_q4, Pq4, opBytes{0x08}},
	{APSIGND, yxm_q4, Pq4, opBytes{0x0a}},
	{APSIGNW, yxm_q4, Pq4, opBytes{0x09}},
	{APSLLO, ypsdq, Pq, opBytes{0x73, 07}},
	{APSLLL, yps, Py3, opBytes{0xf2, 0x72, 06, Pe, 0xf2, Pe, 0x72, 06}},
	{APSLLQ, yps, Py3, opBytes{0xf3, 0x73, 06, Pe, 0xf3, Pe, 0x73, 06}},
	{APSLLW, yps, Py3, opBytes{0xf1, 0x71, 06, Pe, 0xf1, Pe, 0x71, 06}},
	{APSRAL, yps, Py3, opBytes{0xe2, 0x72, 04, Pe, 0xe2, Pe, 0x72, 04}},
	{APSRAW, yps, Py3, opBytes{0xe1, 0x71, 04, Pe, 0xe1, Pe, 0x71, 04}},
	{APSRLO, ypsdq, Pq, opBytes{0x73, 03}},
	{APSRLL, yps, Py3, opBytes{0xd2, 0x72, 02, Pe, 0xd2, Pe, 0x72, 02}},
	{APSRLQ, yps, Py3, opBytes{0xd3, 0x73, 02, Pe, 0xd3, Pe, 0x73, 02}},
	{APSRLW, yps, Py3, opBytes{0xd1, 0x71, 02, Pe, 0xd1, Pe, 0x71, 02}},
	{APSUBB, yxm, Pe, opBytes{0xf8}},
	{APSUBL, yxm, Pe, opBytes{0xfa}},
	{APSUBQ, yxm, Pe, opBytes{0xfb}},
	{APSUBSB, yxm, Pe, opBytes{0xe8}},
	{APSUBSW, yxm, Pe, opBytes{0xe9}},
	{APSUBUSB, yxm, Pe, opBytes{0xd8}},
	{APSUBUSW, yxm, Pe, opBytes{0xd9}},
	{APSUBW, yxm, Pe, opBytes{0xf9}},
	{APTEST, yxm_q4, Pq4, opBytes{0x17}},
	{APUNPCKHBW, ymm, Py1, opBytes{0x68, Pe, 0x68}},
	{APUNPCKHLQ, ymm, Py1, opBytes{0x6a, Pe, 0x6a}},
	{APUNPCKHQDQ, yxm, Pe, opBytes{0x6d}},
	{APUNPCKHWL, ymm, Py1, opBytes{0x69, Pe, 0x69}},
	{APUNPCKLBW, ymm, Py1, opBytes{0x60, Pe, 0x60}},
	{APUNPCKLLQ, ymm, Py1, opBytes{0x62, Pe, 0x62}},
	{APUNPCKLQDQ, yxm, Pe, opBytes{0x6c}},
	{APUNPCKLWL, ymm, Py1, opBytes{0x61, Pe, 0x61}},
	{APUSHAL, ynone, P32, opBytes{0x60}},
	{APUSHAW, ynone, Pe, opBytes{0x60}},
	{APUSHFL, ynone, P32, opBytes{0x9c}},
	{APUSHFQ, ynone, Py, opBytes{0x9c}},
	{APUSHFW, ynone, Pe, opBytes{0x9c}},
	{APUSHL, ypushl, P32, opBytes{0x50, 0xff, 06, 0x6a, 0x68}},
	{APUSHQ, ypushl, Py, opBytes{0x50, 0xff, 06, 0x6a, 0x68}},
	{APUSHW, ypushl, Pe, opBytes{0x50, 0xff, 06, 0x6a, 0x68}},
	{APXOR, ymm, Py1, opBytes{0xef, Pe, 0xef}},
	{AQUAD, ybyte, Px, opBytes{8}},
	{ARCLB, yshb, Pb, opBytes{0xd0, 02, 0xc0, 02, 0xd2, 02}},
	{ARCLL, yshl, Px, opBytes{0xd1, 02, 0xc1, 02, 0xd3, 02, 0xd3, 02}},
	{ARCLQ, yshl, Pw, opBytes{0xd1, 02, 0xc1, 02, 0xd3, 02, 0xd3, 02}},
	{ARCLW, yshl, Pe, opBytes{0xd1, 02, 0xc1, 02, 0xd3, 02, 0xd3, 02}},
	{ARCPPS, yxm, Pm, opBytes{0x53}},
	{ARCPSS, yxm, Pf3, opBytes{0x53}},
	{ARCRB, yshb, Pb, opBytes{0xd0, 03, 0xc0, 03, 0xd2, 03}},
	{ARCRL, yshl, Px, opBytes{0xd1, 03, 0xc1, 03, 0xd3, 03, 0xd3, 03}},
	{ARCRQ, yshl, Pw, opBytes{0xd1, 03, 0xc1, 03, 0xd3, 03, 0xd3, 03}},
	{ARCRW, yshl, Pe, opBytes{0xd1, 03, 0xc1, 03, 0xd3, 03, 0xd3, 03}},
	{AREP, ynone, Px, opBytes{0xf3}},
	{AREPN, ynone, Px, opBytes{0xf2}},
	{obj.ARET, ynone, Px, opBytes{0xc3}},
	{ARETFW, yret, Pe, opBytes{0xcb, 0xca}},
	{ARETFL, yret, Px, opBytes{0xcb, 0xca}},
	{ARETFQ, yret, Pw, opBytes{0xcb, 0xca}},
	{AROLB, yshb, Pb, opBytes{0xd0, 00, 0xc0, 00, 0xd2, 00}},
	{AROLL, yshl, Px, opBytes{0xd1, 00, 0xc1, 00, 0xd3, 00, 0xd3, 00}},
	{AROLQ, yshl, Pw, opBytes{0xd1, 00, 0xc1, 00, 0xd3, 00, 0xd3, 00}},
	{AROLW, yshl, Pe, opBytes{0xd1, 00, 0xc1, 00, 0xd3, 00, 0xd3, 00}},
	{ARORB, yshb, Pb, opBytes{0xd0, 01, 0xc0, 01, 0xd2, 01}},
	{ARORL, yshl, Px, opBytes{0xd1, 01, 0xc1, 01, 0xd3, 01, 0xd3, 01}},
	{ARORQ, yshl, Pw, opBytes{0xd1, 01, 0xc1, 01, 0xd3, 01, 0xd3, 01}},
	{ARORW, yshl, Pe, opBytes{0xd1, 01, 0xc1, 01, 0xd3, 01, 0xd3, 01}},
	{ARSQRTPS, yxm, Pm, opBytes{0x52}},
	{ARSQRTSS, yxm, Pf3, opBytes{0x52}},
	{ASAHF, ynone, Px, opBytes{0x9e, 00, 0x86, 0xe0, 0x50, 0x9d}}, // XCHGB AH,AL; PUSH AX; POPFL
	{ASALB, yshb, Pb, opBytes{0xd0, 04, 0xc0, 04, 0xd2, 04}},
	{ASALL, yshl, Px, opBytes{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	{ASALQ, yshl, Pw, opBytes{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	{ASALW, yshl, Pe, opBytes{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	{ASARB, yshb, Pb, opBytes{0xd0, 07, 0xc0, 07, 0xd2, 07}},
	{ASARL, yshl, Px, opBytes{0xd1, 07, 0xc1, 07, 0xd3, 07, 0xd3, 07}},
	{ASARQ, yshl, Pw, opBytes{0xd1, 07, 0xc1, 07, 0xd3, 07, 0xd3, 07}},
	{ASARW, yshl, Pe, opBytes{0xd1, 07, 0xc1, 07, 0xd3, 07, 0xd3, 07}},
	{ASBBB, yxorb, Pb, opBytes{0x1c, 0x80, 03, 0x18, 0x1a}},
	{ASBBL, yaddl, Px, opBytes{0x83, 03, 0x1d, 0x81, 03, 0x19, 0x1b}},
	{ASBBQ, yaddl, Pw, opBytes{0x83, 03, 0x1d, 0x81, 03, 0x19, 0x1b}},
	{ASBBW, yaddl, Pe, opBytes{0x83, 03, 0x1d, 0x81, 03, 0x19, 0x1b}},
	{ASCASB, ynone, Pb, opBytes{0xae}},
	{ASCASL, ynone, Px, opBytes{0xaf}},
	{ASCASQ, ynone, Pw, opBytes{0xaf}},
	{ASCASW, ynone, Pe, opBytes{0xaf}},
	{ASETCC, yscond, Pb, opBytes{0x0f, 0x93, 00}},
	{ASETCS, yscond, Pb, opBytes{0x0f, 0x92, 00}},
	{ASETEQ, yscond, Pb, opBytes{0x0f, 0x94, 00}},
	{ASETGE, yscond, Pb, opBytes{0x0f, 0x9d, 00}},
	{ASETGT, yscond, Pb, opBytes{0x0f, 0x9f, 00}},
	{ASETHI, yscond, Pb, opBytes{0x0f, 0x97, 00}},
	{ASETLE, yscond, Pb, opBytes{0x0f, 0x9e, 00}},
	{ASETLS, yscond, Pb, opBytes{0x0f, 0x96, 00}},
	{ASETLT, yscond, Pb, opBytes{0x0f, 0x9c, 00}},
	{ASETMI, yscond, Pb, opBytes{0x0f, 0x98, 00}},
	{ASETNE, yscond, Pb, opBytes{0x0f, 0x95, 00}},
	{ASETOC, yscond, Pb, opBytes{0x0f, 0x91, 00}},
	{ASETOS, yscond, Pb, opBytes{0x0f, 0x90, 00}},
	{ASETPC, yscond, Pb, opBytes{0x0f, 0x9b, 00}},
	{ASETPL, yscond, Pb, opBytes{0x0f, 0x99, 00}},
	{ASETPS, yscond, Pb, opBytes{0x0f, 0x9a, 00}},
	{ASHLB, yshb, Pb, opBytes{0xd0, 04, 0xc0, 04, 0xd2, 04}},
	{ASHLL, yshl, Px, opBytes{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	{ASHLQ, yshl, Pw, opBytes{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	{ASHLW, yshl, Pe, opBytes{0xd1, 04, 0xc1, 04, 0xd3, 04, 0xd3, 04}},
	{ASHRB, yshb, Pb, opBytes{0xd0, 05, 0xc0, 05, 0xd2, 05}},
	{ASHRL, yshl, Px, opBytes{0xd1, 05, 0xc1, 05, 0xd3, 05, 0xd3, 05}},
	{ASHRQ, yshl, Pw, opBytes{0xd1, 05, 0xc1, 05, 0xd3, 05, 0xd3, 05}},
	{ASHRW, yshl, Pe, opBytes{0xd1, 05, 0xc1, 05, 0xd3, 05, 0xd3, 05}},
	{ASHUFPD, yxshuf, Pq, opBytes{0xc6, 00}},
	{ASHUFPS, yxshuf, Pm, opBytes{0xc6, 00}},
	{ASQRTPD, yxm, Pe, opBytes{0x51}},
	{ASQRTPS, yxm, Pm, opBytes{0x51}},
	{ASQRTSD, yxm, Pf2, opBytes{0x51}},
	{ASQRTSS, yxm, Pf3, opBytes{0x51}},
	{ASTC, ynone, Px, opBytes{0xf9}},
	{ASTD, ynone, Px, opBytes{0xfd}},
	{ASTI, ynone, Px, opBytes{0xfb}},
	{ASTMXCSR, ysvrs_om, Pm, opBytes{0xae, 03, 0xae, 03}},
	{ASTOSB, ynone, Pb, opBytes{0xaa}},
	{ASTOSL, ynone, Px, opBytes{0xab}},
	{ASTOSQ, ynone, Pw, opBytes{0xab}},
	{ASTOSW, ynone, Pe, opBytes{0xab}},
	{ASUBB, yxorb, Pb, opBytes{0x2c, 0x80, 05, 0x28, 0x2a}},
	{ASUBL, yaddl, Px, opBytes{0x83, 05, 0x2d, 0x81, 05, 0x29, 0x2b}},
	{ASUBPD, yxm, Pe, opBytes{0x5c}},
	{ASUBPS, yxm, Pm, opBytes{0x5c}},
	{ASUBQ, yaddl, Pw, opBytes{0x83, 05, 0x2d, 0x81, 05, 0x29, 0x2b}},
	{ASUBSD, yxm, Pf2, opBytes{0x5c}},
	{ASUBSS, yxm, Pf3, opBytes{0x5c}},
	{ASUBW, yaddl, Pe, opBytes{0x83, 05, 0x2d, 0x81, 05, 0x29, 0x2b}},
	{ASWAPGS, ynone, Pm, opBytes{0x01, 0xf8}},
	{ASYSCALL, ynone, Px, opBytes{0x0f, 0x05}}, // fast syscall
	{ATESTB, yxorb, Pb, opBytes{0xa8, 0xf6, 00, 0x84, 0x84}},
	{ATESTL, ytestl, Px, opBytes{0xa9, 0xf7, 00, 0x85, 0x85}},
	{ATESTQ, ytestl, Pw, opBytes{0xa9, 0xf7, 00, 0x85, 0x85}},
	{ATESTW, ytestl, Pe, opBytes{0xa9, 0xf7, 00, 0x85, 0x85}},
	{ATPAUSE, ywrfsbase, Pq, opBytes{0xae, 06}},
	{obj.ATEXT, ytext, Px, opBytes{}},
	{AUCOMISD, yxm, Pe, opBytes{0x2e}},
	{AUCOMISS, yxm, Pm, opBytes{0x2e}},
	{AUNPCKHPD, yxm, Pe, opBytes{0x15}},
	{AUNPCKHPS, yxm, Pm, opBytes{0x15}},
	{AUNPCKLPD, yxm, Pe, opBytes{0x14}},
	{AUNPCKLPS, yxm, Pm, opBytes{0x14}},
	{AUMONITOR, ywrfsbase, Pf3, opBytes{0xae, 06}},
	{AVERR, ydivl, Pm, opBytes{0x00, 04}},
	{AVERW, ydivl, Pm, opBytes{0x00, 05}},
	{AWAIT, ynone, Px, opBytes{0x9b}},
	{AWORD, ybyte, Px, opBytes{2}},
	{AXCHGB, yml_mb, Pb, opBytes{0x86, 0x86}},
	{AXCHGL, yxchg, Px, opBytes{0x90, 0x90, 0x87, 0x87}},
	{AXCHGQ, yxchg, Pw, opBytes{0x90, 0x90, 0x87, 0x87}},
	{AXCHGW, yxchg, Pe, opBytes{0x90, 0x90, 0x87, 0x87}},
	{AXLAT, ynone, Px, opBytes{0xd7}},
	{AXORB, yxorb, Pb, opBytes{0x34, 0x80, 06, 0x30, 0x32}},
	{AXORL, yaddl, Px, opBytes{0x83, 06, 0x35, 0x81, 06, 0x31, 0x33}},
	{AXORPD, yxm, Pe, opBytes{0x57}},
	{AXORPS, yxm, Pm, opBytes{0x57}},
	{AXORQ, yaddl, Pw, opBytes{0x83, 06, 0x35, 0x81, 06, 0x31, 0x33}},
	{AXORW, yaddl, Pe, opBytes{0x83, 06, 0x35, 0x81, 06, 0x31, 0x33}},
	{AFMOVB, yfmvx, Px, opBytes{0xdf, 04}},
	{AFMOVBP, yfmvp, Px, opBytes{0xdf, 06}},
	{AFMOVD, yfmvd, Px, opBytes{0xdd, 00, 0xdd, 02, 0xd9, 00, 0xdd, 02}},
	{AFMOVDP, yfmvdp, Px, opBytes{0xdd, 03, 0xdd, 03}},
	{AFMOVF, yfmvf, Px, opBytes{0xd9, 00, 0xd9, 02}},
	{AFMOVFP, yfmvp, Px, opBytes{0xd9, 03}},
	{AFMOVL, yfmvf, Px, opBytes{0xdb, 00, 0xdb, 02}},
	{AFMOVLP, yfmvp, Px, opBytes{0xdb, 03}},
	{AFMOVV, yfmvx, Px, opBytes{0xdf, 05}},
	{AFMOVVP, yfmvp, Px, opBytes{0xdf, 07}},
	{AFMOVW, yfmvf, Px, opBytes{0xdf, 00, 0xdf, 02}},
	{AFMOVWP, yfmvp, Px, opBytes{0xdf, 03}},
	{AFMOVX, yfmvx, Px, opBytes{0xdb, 05}},
	{AFMOVXP, yfmvp, Px, opBytes{0xdb, 07}},
	{AFCMOVCC, yfcmv, Px, opBytes{0xdb, 00}},
	{AFCMOVCS, yfcmv, Px, opBytes{0xda, 00}},
	{AFCMOVEQ, yfcmv, Px, opBytes{0xda, 01}},
	{AFCMOVHI, yfcmv, Px, opBytes{0xdb, 02}},
	{AFCMOVLS, yfcmv, Px, opBytes{0xda, 02}},
	{AFCMOVB, yfcmv, Px, opBytes{0xda, 00}},
	{AFCMOVBE, yfcmv, Px, opBytes{0xda, 02}},
	{AFCMOVNB, yfcmv, Px, opBytes{0xdb, 00}},
	{AFCMOVNBE, yfcmv, Px, opBytes{0xdb, 02}},
	{AFCMOVE, yfcmv, Px, opBytes{0xda, 01}},
	{AFCMOVNE, yfcmv, Px, opBytes{0xdb, 01}},
	{AFCMOVNU, yfcmv, Px, opBytes{0xdb, 03}},
	{AFCMOVU, yfcmv, Px, opBytes{0xda, 03}},
	{AFCMOVUN, yfcmv, Px, opBytes{0xda, 03}},
	{AFCOMD, yfadd, Px, opBytes{0xdc, 02, 0xd8, 02, 0xdc, 02}},  // botch
	{AFCOMDP, yfadd, Px, opBytes{0xdc, 03, 0xd8, 03, 0xdc, 03}}, // botch
	{AFCOMDPP, ycompp, Px, opBytes{0xde, 03}},
	{AFCOMF, yfmvx, Px, opBytes{0xd8, 02}},
	{AFCOMFP, yfmvx, Px, opBytes{0xd8, 03}},
	{AFCOMI, yfcmv, Px, opBytes{0xdb, 06}},
	{AFCOMIP, yfcmv, Px, opBytes{0xdf, 06}},
	{AFCOML, yfmvx, Px, opBytes{0xda, 02}},
	{AFCOMLP, yfmvx, Px, opBytes{0xda, 03}},
	{AFCOMW, yfmvx, Px, opBytes{0xde, 02}},
	{AFCOMWP, yfmvx, Px, opBytes{0xde, 03}},
	{AFUCOM, ycompp, Px, opBytes{0xdd, 04}},
	{AFUCOMI, ycompp, Px, opBytes{0xdb, 05}},
	{AFUCOMIP, ycompp, Px, opBytes{0xdf, 05}},
	{AFUCOMP, ycompp, Px, opBytes{0xdd, 05}},
	{AFUCOMPP, ycompp, Px, opBytes{0xda, 13}},
	{AFADDDP, ycompp, Px, opBytes{0xde, 00}},
	{AFADDW, yfmvx, Px, opBytes{0xde, 00}},
	{AFADDL, yfmvx, Px, opBytes{0xda, 00}},
	{AFADDF, yfmvx, Px, opBytes{0xd8, 00}},
	{AFADDD, yfadd, Px, opBytes{0xdc, 00, 0xd8, 00, 0xdc, 00}},
	{AFMULDP, ycompp, Px, opBytes{0xde, 01}},
	{AFMULW, yfmvx, Px, opBytes{0xde, 01}},
	{AFMULL, yfmvx, Px, opBytes{0xda, 01}},
	{AFMULF, yfmvx, Px, opBytes{0xd8, 01}},
	{AFMULD, yfadd, Px, opBytes{0xdc, 01, 0xd8, 01, 0xdc, 01}},
	{AFSUBDP, ycompp, Px, opBytes{0xde, 05}},
	{AFSUBW, yfmvx, Px, opBytes{0xde, 04}},
	{AFSUBL, yfmvx, Px, opBytes{0xda, 04}},
	{AFSUBF, yfmvx, Px, opBytes{0xd8, 04}},
	{AFSUBD, yfadd, Px, opBytes{0xdc, 04, 0xd8, 04, 0xdc, 05}},
	{AFSUBRDP, ycompp, Px, opBytes{0xde, 04}},
	{AFSUBRW, yfmvx, Px, opBytes{0xde, 05}},
	{AFSUBRL, yfmvx, Px, opBytes{0xda, 05}},
	{AFSUBRF, yfmvx, Px, opBytes{0xd8, 05}},
	{AFSUBRD, yfadd, Px, opBytes{0xdc, 05, 0xd8, 05, 0xdc, 04}},
	{AFDIVDP, ycompp, Px, opBytes{0xde, 07}},
	{AFDIVW, yfmvx, Px, opBytes{0xde, 06}},
	{AFDIVL, yfmvx, Px, opBytes{0xda, 06}},
	{AFDIVF, yfmvx, Px, opBytes{0xd8, 06}},
	{AFDIVD, yfadd, Px, opBytes{0xdc, 06, 0xd8, 06, 0xdc, 07}},
	{AFDIVRDP, ycompp, Px, opBytes{0xde, 06}},
	{AFDIVRW, yfmvx, Px, opBytes{0xde, 07}},
	{AFDIVRL, yfmvx, Px, opBytes{0xda, 07}},
	{AFDIVRF, yfmvx, Px, opBytes{0xd8, 07}},
	{AFDIVRD, yfadd, Px, opBytes{0xdc, 07, 0xd8, 07, 0xdc, 06}},
	{AFXCHD, yfxch, Px, opBytes{0xd9, 01, 0xd9, 01}},
	{AFFREE, nil, 0, opBytes{}},
	{AFLDCW, ysvrs_mo, Px, opBytes{0xd9, 05, 0xd9, 05}},
	{AFLDENV, ysvrs_mo, Px, opBytes{0xd9, 04, 0xd9, 04}},
	{AFRSTOR, ysvrs_mo, Px, opBytes{0xdd, 04, 0xdd, 04}},
	{AFSAVE, ysvrs_om, Px, opBytes{0xdd, 06, 0xdd, 06}},
	{AFSTCW, ysvrs_om, Px, opBytes{0xd9, 07, 0xd9, 07}},
	{AFSTENV, ysvrs_om, Px, opBytes{0xd9, 06, 0xd9, 06}},
	{AFSTSW, ystsw, Px, opBytes{0xdd, 07, 0xdf, 0xe0}},
	{AF2XM1, ynone, Px, opBytes{0xd9, 0xf0}},
	{AFABS, ynone, Px, opBytes{0xd9, 0xe1}},
	{AFBLD, ysvrs_mo, Px, opBytes{0xdf, 04}},
	{AFBSTP, yclflush, Px, opBytes{0xdf, 06}},
	{AFCHS, ynone, Px, opBytes{0xd9, 0xe0}},
	{AFCLEX, ynone, Px, opBytes{0xdb, 0xe2}},
	{AFCOS, ynone, Px, opBytes{0xd9, 0xff}},
	{AFDECSTP, ynone, Px, opBytes{0xd9, 0xf6}},
	{AFINCSTP, ynone, Px, opBytes{0xd9, 0xf7}},
	{AFINIT, ynone, Px, opBytes{0xdb, 0xe3}},
	{AFLD1, ynone, Px, opBytes{0xd9, 0xe8}},
	{AFLDL2E, ynone, Px, opBytes{0xd9, 0xea}},
	{AFLDL2T, ynone, Px, opBytes{0xd9, 0xe9}},
	{AFLDLG2, ynone, Px, opBytes{0xd9, 0xec}},
	{AFLDLN2, ynone, Px, opBytes{0xd9, 0xed}},
	{AFLDPI, ynone, Px, opBytes{0xd9, 0xeb}},
	{AFLDZ, ynone, Px, opBytes{0xd9, 0xee}},
	{AFNOP, ynone, Px, opBytes{0xd9, 0xd0}},
	{AFPATAN, ynone, Px, opBytes{0xd9, 0xf3}},
	{AFPREM, ynone, Px, opBytes{0xd9, 0xf8}},
	{AFPREM1, ynone, Px, opBytes{0xd9, 0xf5}},
	{AFPTAN, ynone, Px, opBytes{0xd9, 0xf2}},
	{AFRNDINT, ynone, Px, opBytes{0xd9, 0xfc}},
	{AFSCALE, ynone, Px, opBytes{0xd9, 0xfd}},
	{AFSIN, ynone, Px, opBytes{0xd9, 0xfe}},
	{AFSINCOS, ynone, Px, opBytes{0xd9, 0xfb}},
	{AFSQRT, ynone, Px, opBytes{0xd9, 0xfa}},
	{AFTST, ynone, Px, opBytes{0xd9, 0xe4}},
	{AFXAM, ynone, Px, opBytes{0xd9, 0xe5}},
	{AFXTRACT, ynone, Px, opBytes{0xd9, 0xf4}},
	{AFYL2X, ynone, Px, opBytes{0xd9, 0xf1}},
	{AFYL2XP1, ynone, Px, opBytes{0xd9, 0xf9}},
	{ACMPXCHGB, yrb_mb, Pb, opBytes{0x0f, 0xb0}},
	{ACMPXCHGL, yrl_ml, Px, opBytes{0x0f, 0xb1}},
	{ACMPXCHGW, yrl_ml, Pe, opBytes{0x0f, 0xb1}},
	{ACMPXCHGQ, yrl_ml, Pw, opBytes{0x0f, 0xb1}},
	{ACMPXCHG8B, yscond, Pm, opBytes{0xc7, 01}},
	{ACMPXCHG16B, yscond, Pw, opBytes{0x0f, 0xc7, 01}},
	{AINVD, ynone, Pm, opBytes{0x08}},
	{AINVLPG, ydivb, Pm, opBytes{0x01, 07}},
	{AINVPCID, ycrc32l, Pe, opBytes{0x0f, 0x38, 0x82, 0}},
	{ALFENCE, ynone, Pm, opBytes{0xae, 0xe8}},
	{AMFENCE, ynone, Pm, opBytes{0xae, 0xf0}},
	{AMOVNTIL, yrl_ml, Pm, opBytes{0xc3}},
	{AMOVNTIQ, yrl_ml, Pw, opBytes{0x0f, 0xc3}},
	{ARDPKRU, ynone, Pm, opBytes{0x01, 0xee, 0}},
	{ARDMSR, ynone, Pm, opBytes{0x32}},
	{ARDPMC, ynone, Pm, opBytes{0x33}},
	{ARDTSC, ynone, Pm, opBytes{0x31}},
	{ARSM, ynone, Pm, opBytes{0xaa}},
	{ASFENCE, ynone, Pm, opBytes{0xae, 0xf8}},
	{ASYSRET, ynone, Pm, opBytes{0x07}},
	{AWBINVD, ynone, Pm, opBytes{0x09}},
	{AWRMSR, ynone, Pm, opBytes{0x30}},
	{AWRPKRU, ynone, Pm, opBytes{0x01, 0xef, 0}},
	{AXADDB, yrb_mb, Pb, opBytes{0x0f, 0xc0}},
	{AXADDL, yrl_ml, Px, opBytes{0x0f, 0xc1}},
	{AXADDQ, yrl_ml, Pw, opBytes{0x0f, 0xc1}},
	{AXADDW, yrl_ml, Pe, opBytes{0x0f, 0xc1}},
	{ACRC32B, ycrc32b, Px, opBytes{0xf2, 0x0f, 0x38, 0xf0, 0}},
	{ACRC32L, ycrc32l, Px, opBytes{0xf2, 0x0f, 0x38, 0xf1, 0}},
	{ACRC32Q, ycrc32l, Pw, opBytes{0xf2, 0x0f, 0x38, 0xf1, 0}},
	{ACRC32W, ycrc32l, Pe, opBytes{0xf2, 0x0f, 0x38, 0xf1, 0}},
	{APREFETCHT0, yprefetch, Pm, opBytes{0x18, 01}},
	{APREFETCHT1, yprefetch, Pm, opBytes{0x18, 02}},
	{APREFETCHT2, yprefetch, Pm, opBytes{0x18, 03}},
	{APREFETCHNTA, yprefetch, Pm, opBytes{0x18, 00}},
	{AMOVQL, yrl_ml, Px, opBytes{0x89}},
	{obj.AUNDEF, ynone, Px, opBytes{0x0f, 0x0b}},
	{AAESENC, yaes, Pq, opBytes{0x38, 0xdc, 0}},
	{AAESENCLAST, yaes, Pq, opBytes{0x38, 0xdd, 0}},
	{AAESDEC, yaes, Pq, opBytes{0x38, 0xde, 0}},
	{AAESDECLAST, yaes, Pq, opBytes{0x38, 0xdf, 0}},
	{AAESIMC, yaes, Pq, opBytes{0x38, 0xdb, 0}},
	{AAESKEYGENASSIST, yxshuf, Pq, opBytes{0x3a, 0xdf, 0}},
	{AROUNDPD, yxshuf, Pq, opBytes{0x3a, 0x09, 0}},
	{AROUNDPS, yxshuf, Pq, opBytes{0x3a, 0x08, 0}},
	{AROUNDSD, yxshuf, Pq, opBytes{0x3a, 0x0b, 0}},
	{AROUNDSS, yxshuf, Pq, opBytes{0x3a, 0x0a, 0}},
	{APSHUFD, yxshuf, Pq, opBytes{0x70, 0}},
	{APCLMULQDQ, yxshuf, Pq, opBytes{0x3a, 0x44, 0}},
	{APCMPESTRI, yxshuf, Pq, opBytes{0x3a, 0x61, 0}},
	{APCMPESTRM, yxshuf, Pq, opBytes{0x3a, 0x60, 0}},
	{AMOVDDUP, yxm, Pf2, opBytes{0x12}},
	{AMOVSHDUP, yxm, Pf3, opBytes{0x16}},
	{AMOVSLDUP, yxm, Pf3, opBytes{0x12}},
	{ARDTSCP, ynone, Pm, opBytes{0x01, 0xf9, 0}},
	{ASTAC, ynone, Pm, opBytes{0x01, 0xcb, 0}},
	{AUD1, ynone, Pm, opBytes{0xb9, 0}},
	{AUD2, ynone, Pm, opBytes{0x0b, 0}},
	{AUMWAIT, ywrfsbase, Pf2, opBytes{0xae, 06}},
	{ASYSENTER, ynone, Px, opBytes{0x0f, 0x34, 0}},
	{ASYSENTER64, ynone, Pw, opBytes{0x0f, 0x34, 0}},
	{ASYSEXIT, ynone, Px, opBytes{0x0f, 0x35, 0}},
	{ASYSEXIT64, ynone, Pw, opBytes{0x0f, 0x35, 0}},
	{ALMSW, ydivl, Pm, opBytes{0x01, 06}},
	{ALLDT, ydivl, Pm, opBytes{0x00, 02}},
	{ALIDT, ysvrs_mo, Pm, opBytes{0x01, 03}},
	{ALGDT, ysvrs_mo, Pm, opBytes{0x01, 02}},
	{ATZCNTW, ycrc32l, Pe, opBytes{0xf3, 0x0f, 0xbc, 0}},
	{ATZCNTL, ycrc32l, Px, opBytes{0xf3, 0x0f, 0xbc, 0}},
	{ATZCNTQ, ycrc32l, Pw, opBytes{0xf3, 0x0f, 0xbc, 0}},
	{AXRSTOR, ydivl, Px, opBytes{0x0f, 0xae, 05}},
	{AXRSTOR64, ydivl, Pw, opBytes{0x0f, 0xae, 05}},
	{AXRSTORS, ydivl, Px, opBytes{0x0f, 0xc7, 03}},
	{AXRSTORS64, ydivl, Pw, opBytes{0x0f, 0xc7, 03}},
	{AXSAVE, yclflush, Px, opBytes{0x0f, 0xae, 04}},
	{AXSAVE64, yclflush, Pw, opBytes{0x0f, 0xae, 04}},
	{AXSAVEOPT, yclflush, Px, opBytes{0x0f, 0xae, 06}},
	{AXSAVEOPT64, yclflush, Pw, opBytes{0x0f, 0xae, 06}},
	{AXSAVEC, yclflush, Px, opBytes{0x0f, 0xc7, 04}},
	{AXSAVEC64, yclflush, Pw, opBytes{0x0f, 0xc7, 04}},
	{AXSAVES, yclflush, Px, opBytes{0x0f, 0xc7, 05}},
	{AXSAVES64, yclflush, Pw, opBytes{0x0f, 0xc7, 05}},
	{ASGDT, yclflush, Pm, opBytes{0x01, 00}},
	{ASIDT, yclflush, Pm, opBytes{0x01, 01}},
	{ARDRANDW, yrdrand, Pe, opBytes{0x0f, 0xc7, 06}},
	{ARDRANDL, yrdrand, Px, opBytes{0x0f, 0xc7, 06}},
	{ARDRANDQ, yrdrand, Pw, opBytes{0x0f, 0xc7, 06}},
	{ARDSEEDW, yrdrand, Pe, opBytes{0x0f, 0xc7, 07}},
	{ARDSEEDL, yrdrand, Px, opBytes{0x0f, 0xc7, 07}},
	{ARDSEEDQ, yrdrand, Pw, opBytes{0x0f, 0xc7, 07}},
	{ASTRW, yincq, Pe, opBytes{0x0f, 0x00, 01}},
	{ASTRL, yincq, Px, opBytes{0x0f, 0x00, 01}},
	{ASTRQ, yincq, Pw, opBytes{0x0f, 0x00, 01}},
	{AXSETBV, ynone, Pm, opBytes{0x01, 0xd1, 0}},
	{AMOVBEW, ymovbe, Pq, opBytes{0x38, 0xf0, 0, 0x38, 0xf1, 0}},
	{AMOVBEL, ymovbe, Pm, opBytes{0x38, 0xf0, 0, 0x38, 0xf1, 0}},
	{AMOVBEQ, ymovbe, Pw, opBytes{0x0f, 0x38, 0xf0, 0, 0x0f, 0x38, 0xf1, 0}},
	{ANOPW, ydivl, Pe, opBytes{0x0f, 0x1f, 00}},
	{ANOPL, ydivl, Px, opBytes{0x0f, 0x1f, 00}},
	{ASLDTW, yincq, Pe, opBytes{0x0f, 0x00, 00}},
	{ASLDTL, yincq, Px, opBytes{0x0f, 0x00, 00}},
	{ASLDTQ, yincq, Pw, opBytes{0x0f, 0x00, 00}},
	{ASMSWW, yincq, Pe, opBytes{0x0f, 0x01, 04}},
	{ASMSWL, yincq, Px, opBytes{0x0f, 0x01, 04}},
	{ASMSWQ, yincq, Pw, opBytes{0x0f, 0x01, 04}},
	{ABLENDVPS, yblendvpd, Pq4, opBytes{0x14}},
	{ABLENDVPD, yblendvpd, Pq4, opBytes{0x15}},
	{APBLENDVB, yblendvpd, Pq4, opBytes{0x10}},
	{ASHA1MSG1, yaes, Px, opBytes{0x0f, 0x38, 0xc9, 0}},
	{ASHA1MSG2, yaes, Px, opBytes{0x0f, 0x38, 0xca, 0}},
	{ASHA1NEXTE, yaes, Px, opBytes{0x0f, 0x38, 0xc8, 0}},
	{ASHA256MSG1, yaes, Px, opBytes{0x0f, 0x38, 0xcc, 0}},
	{ASHA256MSG2, yaes, Px, opBytes{0x0f, 0x38, 0xcd, 0}},
	{ASHA1RNDS4, ysha1rnds4, Pm, opBytes{0x3a, 0xcc, 0}},
	{ASHA256RNDS2, ysha256rnds2, Px, opBytes{0x0f, 0x38, 0xcb, 0}},
	{ARDFSBASEL, yrdrand, Pf3, opBytes{0xae, 00}},
	{ARDFSBASEQ, yrdrand, Pfw, opBytes{0xae, 00}},
	{ARDGSBASEL, yrdrand, Pf3, opBytes{0xae, 01}},
	{ARDGSBASEQ, yrdrand, Pfw, opBytes{0xae, 01}},
	{AWRFSBASEL, ywrfsbase, Pf3, opBytes{0xae, 02}},
	{AWRFSBASEQ, ywrfsbase, Pfw, opBytes{0xae, 02}},
	{AWRGSBASEL, ywrfsbase, Pf3, opBytes{0xae, 03}},
	{AWRGSBASEQ, ywrfsbase, Pfw, opBytes{0xae, 03}},
	{ALFSW, ym_rl, Pe, opBytes{0x0f, 0xb4}},
	{ALFSL, ym_rl, Px, opBytes{0x0f, 0xb4}},
	{ALFSQ, ym_rl, Pw, opBytes{0x0f, 0xb4}},
	{ALGSW, ym_rl, Pe, opBytes{0x0f, 0xb5}},
	{ALGSL, ym_rl, Px, opBytes{0x0f, 0xb5}},
	{ALGSQ, ym_rl, Pw, opBytes{0x0f, 0xb5}},
	{ALSSW, ym_rl, Pe, opBytes{0x0f, 0xb2}},
	{ALSSL, ym_rl, Px, opBytes{0x0f, 0xb2}},
	{ALSSQ, ym_rl, Pw, opBytes{0x0f, 0xb2}},
	{ARDPID, yrdrand, Pf3, opBytes{0xc7, 07}},

	{ABLENDPD, yxshuf, Pq, opBytes{0x3a, 0x0d, 0}},
	{ABLENDPS, yxshuf, Pq, opBytes{0x3a, 0x0c, 0}},
	{AXACQUIRE, ynone, Px, opBytes{0xf2}},
	{AXRELEASE, ynone, Px, opBytes{0xf3}},
	{AXBEGIN, yxbegin, Px, opBytes{0xc7, 0xf8}},
	{AXABORT, yxabort, Px, opBytes{0xc6, 0xf8}},
	{AXEND, ynone, Px, opBytes{0x0f, 01, 0xd5}},
	{AXTEST, ynone, Px, opBytes{0x0f, 01, 0xd6}},
	{AXGETBV, ynone, Pm, opBytes{01, 0xd0}},
	{obj.AFUNCDATA, yfuncdata, Px, opBytes{0, 0}},
	{obj.APCDATA, ypcdata, Px, opBytes{0, 0}},
	{obj.ADUFFCOPY, yduff, Px, opBytes{0xe8}},
	{obj.ADUFFZERO, yduff, Px, opBytes{0xe8}},

	{obj.AEND, nil, 0, opBytes{}},
	{0, nil, 0, opBytes{}},
}

var opindex [(ALAST + 1) & obj.AMask]*Optab

// useAbs reports whether s describes a symbol that must avoid pc-relative addressing.
// This happens on systems like Solaris that call .so functions instead of system calls.
// It does not seem to be necessary for any other systems. This is probably working
// around a Solaris-specific bug that should be fixed differently, but we don't know
// what that bug is. And this does fix it.
func useAbs(ctxt *obj.Link, s *obj.LSym) bool {
	if ctxt.Headtype == objabi.Hsolaris {
		// All the Solaris dynamic imports from libc.so begin with "libc_".
		return strings.HasPrefix(s.Name, "libc_")
	}
	return ctxt.Arch.Family == sys.I386 && !ctxt.Flag_shared
}

// single-instruction no-ops of various lengths.
// constructed by hand and disassembled with gdb to verify.
// see http://www.agner.org/optimize/optimizing_assembly.pdf for discussion.
var nop = [][16]uint8{
	{0x90},
	{0x66, 0x90},
	{0x0F, 0x1F, 0x00},
	{0x0F, 0x1F, 0x40, 0x00},
	{0x0F, 0x1F, 0x44, 0x00, 0x00},
	{0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00},
	{0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00},
	{0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
	{0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
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

func noppad(ctxt *obj.Link, s *obj.LSym, c int32, pad int32) int32 {
	s.Grow(int64(c) + int64(pad))
	fillnop(s.P[c:], int(pad))
	return c + pad
}

func spadjop(ctxt *obj.Link, l, q obj.As) obj.As {
	if ctxt.Arch.Family != sys.AMD64 || ctxt.Arch.PtrSize == 4 {
		return l
	}
	return q
}

// isJump returns whether p is a jump instruction.
// It is used to ensure that no standalone or macro-fused jump will straddle
// or end on a 32 byte boundary by inserting NOPs before the jumps.
func isJump(p *obj.Prog) bool {
	return p.To.Target() != nil || p.As == obj.AJMP || p.As == obj.ACALL ||
		p.As == obj.ARET || p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO
}

// lookForJCC returns the first real instruction starting from p, if that instruction is a conditional
// jump. Otherwise, nil is returned.
func lookForJCC(p *obj.Prog) *obj.Prog {
	// Skip any PCDATA, FUNCDATA or NOP instructions
	var q *obj.Prog
	for q = p.Link; q != nil && (q.As == obj.APCDATA || q.As == obj.AFUNCDATA || q.As == obj.ANOP); q = q.Link {
	}

	if q == nil || q.To.Target() == nil || p.As == obj.AJMP || p.As == obj.ACALL {
		return nil
	}

	switch q.As {
	case AJOS, AJOC, AJCS, AJCC, AJEQ, AJNE, AJLS, AJHI,
		AJMI, AJPL, AJPS, AJPC, AJLT, AJGE, AJLE, AJGT:
	default:
		return nil
	}

	return q
}

// fusedJump determines whether p can be fused with a subsequent conditional jump instruction.
// If it can, we return true followed by the total size of the fused jump. If it can't, we return false.
// Macro fusion rules are derived from the Intel Optimization Manual (April 2019) section 3.4.2.2.
func fusedJump(p *obj.Prog) (bool, uint8) {
	var fusedSize uint8

	// The first instruction in a macro fused pair may be preceded by the LOCK prefix,
	// or possibly an XACQUIRE/XRELEASE prefix followed by a LOCK prefix. If it is, we
	// need to be careful to insert any padding before the locks rather than directly after them.

	if p.As == AXRELEASE || p.As == AXACQUIRE {
		fusedSize += p.Isize
		for p = p.Link; p != nil && (p.As == obj.APCDATA || p.As == obj.AFUNCDATA); p = p.Link {
		}
		if p == nil {
			return false, 0
		}
	}
	if p.As == ALOCK {
		fusedSize += p.Isize
		for p = p.Link; p != nil && (p.As == obj.APCDATA || p.As == obj.AFUNCDATA); p = p.Link {
		}
		if p == nil {
			return false, 0
		}
	}
	cmp := p.As == ACMPB || p.As == ACMPL || p.As == ACMPQ || p.As == ACMPW

	cmpAddSub := p.As == AADDB || p.As == AADDL || p.As == AADDW || p.As == AADDQ ||
		p.As == ASUBB || p.As == ASUBL || p.As == ASUBW || p.As == ASUBQ || cmp

	testAnd := p.As == ATESTB || p.As == ATESTL || p.As == ATESTQ || p.As == ATESTW ||
		p.As == AANDB || p.As == AANDL || p.As == AANDQ || p.As == AANDW

	incDec := p.As == AINCB || p.As == AINCL || p.As == AINCQ || p.As == AINCW ||
		p.As == ADECB || p.As == ADECL || p.As == ADECQ || p.As == ADECW

	if !cmpAddSub && !testAnd && !incDec {
		return false, 0
	}

	if !incDec {
		var argOne obj.AddrType
		var argTwo obj.AddrType
		if cmp {
			argOne = p.From.Type
			argTwo = p.To.Type
		} else {
			argOne = p.To.Type
			argTwo = p.From.Type
		}
		if argOne == obj.TYPE_REG {
			if argTwo != obj.TYPE_REG && argTwo != obj.TYPE_CONST && argTwo != obj.TYPE_MEM {
				return false, 0
			}
		} else if argOne == obj.TYPE_MEM {
			if argTwo != obj.TYPE_REG {
				return false, 0
			}
		} else {
			return false, 0
		}
	}

	fusedSize += p.Isize
	jmp := lookForJCC(p)
	if jmp == nil {
		return false, 0
	}

	fusedSize += jmp.Isize

	if testAnd {
		return true, fusedSize
	}

	if jmp.As == AJOC || jmp.As == AJOS || jmp.As == AJMI ||
		jmp.As == AJPL || jmp.As == AJPS || jmp.As == AJPC {
		return false, 0
	}

	if cmpAddSub {
		return true, fusedSize
	}

	if jmp.As == AJCS || jmp.As == AJCC || jmp.As == AJHI || jmp.As == AJLS {
		return false, 0
	}

	return true, fusedSize
}

type padJumpsCtx int32

func makePjcCtx(ctxt *obj.Link) padJumpsCtx {
	// Disable jump padding on 32 bit builds by setting
	// padJumps to 0.
	if ctxt.Arch.Family == sys.I386 {
		return padJumpsCtx(0)
	}

	// Disable jump padding for hand written assembly code.
	if ctxt.IsAsm {
		return padJumpsCtx(0)
	}

	return padJumpsCtx(32)
}

// padJump detects whether the instruction being assembled is a standalone or a macro-fused
// jump that needs to be padded. If it is, NOPs are inserted to ensure that the jump does
// not cross or end on a 32 byte boundary.
func (pjc padJumpsCtx) padJump(ctxt *obj.Link, s *obj.LSym, p *obj.Prog, c int32) int32 {
	if pjc == 0 {
		return c
	}

	var toPad int32
	fj, fjSize := fusedJump(p)
	mask := int32(pjc - 1)
	if fj {
		if (c&mask)+int32(fjSize) >= int32(pjc) {
			toPad = int32(pjc) - (c & mask)
		}
	} else if isJump(p) {
		if (c&mask)+int32(p.Isize) >= int32(pjc) {
			toPad = int32(pjc) - (c & mask)
		}
	}
	if toPad <= 0 {
		return c
	}

	return noppad(ctxt, s, c, toPad)
}

// reAssemble is called if an instruction's size changes during assembly. If
// it does and the instruction is a standalone or a macro-fused jump we need to
// reassemble.
func (pjc padJumpsCtx) reAssemble(p *obj.Prog) bool {
	if pjc == 0 {
		return false
	}

	fj, _ := fusedJump(p)
	return fj || isJump(p)
}

type nopPad struct {
	p *obj.Prog // Instruction before the pad
	n int32     // Size of the pad
}

// Padding bytes to add to align code as requested.
// Alignment is restricted to powers of 2 between 8 and 2048 inclusive.
//
// pc: current offset in function, in bytes
// a: requested alignment, in bytes
// cursym: current function being assembled
// returns number of bytes of padding needed
func addpad(pc, a int64, ctxt *obj.Link, cursym *obj.LSym) int {
	if !((a&(a-1) == 0) && 8 <= a && a <= 2048) {
		ctxt.Diag("alignment value of an instruction must be a power of two and in the range [8, 2048], got %d\n", a)
		return 0
	}

	// By default function alignment is 32 bytes for amd64
	if cursym.Func().Align < int32(a) {
		cursym.Func().Align = int32(a)
	}

	if pc&(a-1) != 0 {
		return int(a - (pc & (a - 1)))
	}

	return 0
}

func span6(ctxt *obj.Link, s *obj.LSym, newprog obj.ProgAlloc) {
	if ctxt.Retpoline && ctxt.Arch.Family == sys.I386 {
		ctxt.Diag("-spectre=ret not supported on 386")
		ctxt.Retpoline = false // don't keep printing
	}

	pjc := makePjcCtx(ctxt)

	if s.P != nil {
		return
	}

	if ycover[0] == 0 {
		ctxt.Diag("x86 tables not initialized, call x86.instinit first")
	}

	for p := s.Func().Text; p != nil; p = p.Link {
		if p.To.Type == obj.TYPE_BRANCH && p.To.Target() == nil {
			p.To.SetTarget(p)
		}
		if p.As == AADJSP {
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_SP
			// Generate 'ADDQ $x, SP' or 'SUBQ $x, SP', with x positive.
			// One exception: It is smaller to encode $-0x80 than $0x80.
			// For that case, flip the sign and the op:
			// Instead of 'ADDQ $0x80, SP', generate 'SUBQ $-0x80, SP'.
			switch v := p.From.Offset; {
			case v == 0:
				p.As = obj.ANOP
			case v == 0x80 || (v < 0 && v != -0x80):
				p.As = spadjop(ctxt, AADDL, AADDQ)
				p.From.Offset *= -1
			default:
				p.As = spadjop(ctxt, ASUBL, ASUBQ)
			}
		}
		if ctxt.Retpoline && (p.As == obj.ACALL || p.As == obj.AJMP) && (p.To.Type == obj.TYPE_REG || p.To.Type == obj.TYPE_MEM) {
			if p.To.Type != obj.TYPE_REG {
				ctxt.Diag("non-retpoline-compatible: %v", p)
				continue
			}
			p.To.Type = obj.TYPE_BRANCH
			p.To.Name = obj.NAME_EXTERN
			p.To.Sym = ctxt.Lookup("runtime.retpoline" + obj.Rconv(int(p.To.Reg)))
			p.To.Reg = 0
			p.To.Offset = 0
		}
	}

	var count int64 // rough count of number of instructions
	for p := s.Func().Text; p != nil; p = p.Link {
		count++
		p.Back = branchShort // use short branches first time through
		if q := p.To.Target(); q != nil && (q.Back&branchShort != 0) {
			p.Back |= branchBackwards
			q.Back |= branchLoopHead
		}
	}
	s.GrowCap(count * 5) // preallocate roughly 5 bytes per instruction

	var ab AsmBuf
	var n int
	var c int32
	errors := ctxt.Errors
	var nops []nopPad // Padding for a particular assembly (reuse slice storage if multiple assemblies)
	nrelocs0 := len(s.R)
	for {
		// This loop continues while there are reasons to re-assemble
		// whole block, like the presence of long forward jumps.
		reAssemble := false
		for i := range s.R[nrelocs0:] {
			s.R[nrelocs0+i] = obj.Reloc{}
		}
		s.R = s.R[:nrelocs0] // preserve marker relocations generated by the compiler
		s.P = s.P[:0]
		c = 0
		var pPrev *obj.Prog
		nops = nops[:0]
		for p := s.Func().Text; p != nil; p = p.Link {
			c0 := c
			c = pjc.padJump(ctxt, s, p, c)

			if p.As == obj.APCALIGN {
				aln := p.From.Offset
				v := addpad(int64(c), aln, ctxt, s)
				if v > 0 {
					s.Grow(int64(c) + int64(v))
					fillnop(s.P[c:], int(v))
				}

				c += int32(v)
				pPrev = p
				continue
			}

			if maxLoopPad > 0 && p.Back&branchLoopHead != 0 && c&(loopAlign-1) != 0 {
				// pad with NOPs
				v := -c & (loopAlign - 1)

				if v <= maxLoopPad {
					s.Grow(int64(c) + int64(v))
					fillnop(s.P[c:], int(v))
					c += v
				}
			}

			p.Pc = int64(c)

			// process forward jumps to p
			for q := p.Rel; q != nil; q = q.Forwd {
				v := int32(p.Pc - (q.Pc + int64(q.Isize)))
				if q.Back&branchShort != 0 {
					if v > 127 {
						reAssemble = true
						q.Back ^= branchShort
					}

					if q.As == AJCXZL || q.As == AXBEGIN {
						s.P[q.Pc+2] = byte(v)
					} else {
						s.P[q.Pc+1] = byte(v)
					}
				} else {
					binary.LittleEndian.PutUint32(s.P[q.Pc+int64(q.Isize)-4:], uint32(v))
				}
			}

			p.Rel = nil

			p.Pc = int64(c)
			ab.asmins(ctxt, s, p)
			m := ab.Len()
			if int(p.Isize) != m {
				p.Isize = uint8(m)
				if pjc.reAssemble(p) {
					// We need to re-assemble here to check for jumps and fused jumps
					// that span or end on 32 byte boundaries.
					reAssemble = true
				}
			}

			s.Grow(p.Pc + int64(m))
			copy(s.P[p.Pc:], ab.Bytes())
			// If there was padding, remember it.
			if pPrev != nil && !ctxt.IsAsm && c > c0 {
				nops = append(nops, nopPad{p: pPrev, n: c - c0})
			}
			c += int32(m)
			pPrev = p
		}

		n++
		if n > 1000 {
			ctxt.Diag("span must be looping")
			log.Fatalf("loop")
		}
		if !reAssemble {
			break
		}
		if ctxt.Errors > errors {
			return
		}
	}
	// splice padding nops into Progs
	for _, n := range nops {
		pp := n.p
		np := &obj.Prog{Link: pp.Link, Ctxt: pp.Ctxt, As: obj.ANOP, Pos: pp.Pos.WithNotStmt(), Pc: pp.Pc + int64(pp.Isize), Isize: uint8(n.n)}
		pp.Link = np
	}

	s.Size = int64(c)

	if false { /* debug['a'] > 1 */
		fmt.Printf("span1 %s %d (%d tries)\n %.6x", s.Name, s.Size, n, 0)
		var i int
		for i = 0; i < len(s.P); i++ {
			fmt.Printf(" %.2x", s.P[i])
			if i%16 == 15 {
				fmt.Printf("\n  %.6x", uint(i+1))
			}
		}

		if i%16 != 0 {
			fmt.Printf("\n")
		}

		for i := 0; i < len(s.R); i++ {
			r := &s.R[i]
			fmt.Printf(" rel %#.4x/%d %s%+d\n", uint32(r.Off), r.Siz, r.Sym.Name, r.Add)
		}
	}

	// Mark nonpreemptible instruction sequences.
	// The 2-instruction TLS access sequence
	//	MOVQ TLS, BX
	//	MOVQ 0(BX)(TLS*1), BX
	// is not async preemptible, as if it is preempted and resumed on
	// a different thread, the TLS address may become invalid.
	if !CanUse1InsnTLS(ctxt) {
		useTLS := func(p *obj.Prog) bool {
			// Only need to mark the second instruction, which has
			// REG_TLS as Index. (It is okay to interrupt and restart
			// the first instruction.)
			return p.From.Index == REG_TLS
		}
		obj.MarkUnsafePoints(ctxt, s.Func().Text, newprog, useTLS, nil)
	}

	// Now that we know byte offsets, we can generate jump table entries.
	// TODO: could this live in obj instead of obj/$ARCH?
	for _, jt := range s.Func().JumpTables {
		for i, p := range jt.Targets {
			// The ith jumptable entry points to the p.Pc'th
			// byte in the function symbol s.
			jt.Sym.WriteAddr(ctxt, int64(i)*8, 8, s, p.Pc)
		}
	}
}

func instinit(ctxt *obj.Link) {
	if ycover[0] != 0 {
		// Already initialized; stop now.
		// This happens in the cmd/asm tests,
		// each of which re-initializes the arch.
		return
	}

	switch ctxt.Headtype {
	case objabi.Hplan9:
		plan9privates = ctxt.Lookup("_privates")
	}

	for i := range avxOptab {
		c := avxOptab[i].as
		if opindex[c&obj.AMask] != nil {
			ctxt.Diag("phase error in avxOptab: %d (%v)", i, c)
		}
		opindex[c&obj.AMask] = &avxOptab[i]
	}
	for i := 1; optab[i].as != 0; i++ {
		c := optab[i].as
		if opindex[c&obj.AMask] != nil {
			ctxt.Diag("phase error in optab: %d (%v)", i, c)
		}
		opindex[c&obj.AMask] = &optab[i]
	}

	for i := 0; i < Ymax; i++ {
		ycover[i*Ymax+i] = 1
	}

	ycover[Yi0*Ymax+Yu2] = 1
	ycover[Yi1*Ymax+Yu2] = 1

	ycover[Yi0*Ymax+Yi8] = 1
	ycover[Yi1*Ymax+Yi8] = 1
	ycover[Yu2*Ymax+Yi8] = 1
	ycover[Yu7*Ymax+Yi8] = 1

	ycover[Yi0*Ymax+Yu7] = 1
	ycover[Yi1*Ymax+Yu7] = 1
	ycover[Yu2*Ymax+Yu7] = 1

	ycover[Yi0*Ymax+Yu8] = 1
	ycover[Yi1*Ymax+Yu8] = 1
	ycover[Yu2*Ymax+Yu8] = 1
	ycover[Yu7*Ymax+Yu8] = 1

	ycover[Yi0*Ymax+Ys32] = 1
	ycover[Yi1*Ymax+Ys32] = 1
	ycover[Yu2*Ymax+Ys32] = 1
	ycover[Yu7*Ymax+Ys32] = 1
	ycover[Yu8*Ymax+Ys32] = 1
	ycover[Yi8*Ymax+Ys32] = 1

	ycover[Yi0*Ymax+Yi32] = 1
	ycover[Yi1*Ymax+Yi32] = 1
	ycover[Yu2*Ymax+Yi32] = 1
	ycover[Yu7*Ymax+Yi32] = 1
	ycover[Yu8*Ymax+Yi32] = 1
	ycover[Yi8*Ymax+Yi32] = 1
	ycover[Ys32*Ymax+Yi32] = 1

	ycover[Yi0*Ymax+Yi64] = 1
	ycover[Yi1*Ymax+Yi64] = 1
	ycover[Yu7*Ymax+Yi64] = 1
	ycover[Yu2*Ymax+Yi64] = 1
	ycover[Yu8*Ymax+Yi64] = 1
	ycover[Yi8*Ymax+Yi64] = 1
	ycover[Ys32*Ymax+Yi64] = 1
	ycover[Yi32*Ymax+Yi64] = 1

	ycover[Yal*Ymax+Yrb] = 1
	ycover[Ycl*Ymax+Yrb] = 1
	ycover[Yax*Ymax+Yrb] = 1
	ycover[Ycx*Ymax+Yrb] = 1
	ycover[Yrx*Ymax+Yrb] = 1
	ycover[Yrl*Ymax+Yrb] = 1 // but not Yrl32

	ycover[Ycl*Ymax+Ycx] = 1

	ycover[Yax*Ymax+Yrx] = 1
	ycover[Ycx*Ymax+Yrx] = 1

	ycover[Yax*Ymax+Yrl] = 1
	ycover[Ycx*Ymax+Yrl] = 1
	ycover[Yrx*Ymax+Yrl] = 1
	ycover[Yrl32*Ymax+Yrl] = 1

	ycover[Yf0*Ymax+Yrf] = 1

	ycover[Yal*Ymax+Ymb] = 1
	ycover[Ycl*Ymax+Ymb] = 1
	ycover[Yax*Ymax+Ymb] = 1
	ycover[Ycx*Ymax+Ymb] = 1
	ycover[Yrx*Ymax+Ymb] = 1
	ycover[Yrb*Ymax+Ymb] = 1
	ycover[Yrl*Ymax+Ymb] = 1 // but not Yrl32
	ycover[Ym*Ymax+Ymb] = 1

	ycover[Yax*Ymax+Yml] = 1
	ycover[Ycx*Ymax+Yml] = 1
	ycover[Yrx*Ymax+Yml] = 1
	ycover[Yrl*Ymax+Yml] = 1
	ycover[Yrl32*Ymax+Yml] = 1
	ycover[Ym*Ymax+Yml] = 1

	ycover[Yax*Ymax+Ymm] = 1
	ycover[Ycx*Ymax+Ymm] = 1
	ycover[Yrx*Ymax+Ymm] = 1
	ycover[Yrl*Ymax+Ymm] = 1
	ycover[Yrl32*Ymax+Ymm] = 1
	ycover[Ym*Ymax+Ymm] = 1
	ycover[Ymr*Ymax+Ymm] = 1

	ycover[Yxr0*Ymax+Yxr] = 1

	ycover[Ym*Ymax+Yxm] = 1
	ycover[Yxr0*Ymax+Yxm] = 1
	ycover[Yxr*Ymax+Yxm] = 1

	ycover[Ym*Ymax+Yym] = 1
	ycover[Yyr*Ymax+Yym] = 1

	ycover[Yxr0*Ymax+YxrEvex] = 1
	ycover[Yxr*Ymax+YxrEvex] = 1

	ycover[Ym*Ymax+YxmEvex] = 1
	ycover[Yxr0*Ymax+YxmEvex] = 1
	ycover[Yxr*Ymax+YxmEvex] = 1
	ycover[YxrEvex*Ymax+YxmEvex] = 1

	ycover[Yyr*Ymax+YyrEvex] = 1

	ycover[Ym*Ymax+YymEvex] = 1
	ycover[Yyr*Ymax+YymEvex] = 1
	ycover[YyrEvex*Ymax+YymEvex] = 1

	ycover[Ym*Ymax+Yzm] = 1
	ycover[Yzr*Ymax+Yzm] = 1

	ycover[Yk0*Ymax+Yk] = 1
	ycover[Yknot0*Ymax+Yk] = 1

	ycover[Yk0*Ymax+Ykm] = 1
	ycover[Yknot0*Ymax+Ykm] = 1
	ycover[Yk*Ymax+Ykm] = 1
	ycover[Ym*Ymax+Ykm] = 1

	ycover[Yxvm*Ymax+YxvmEvex] = 1

	ycover[Yyvm*Ymax+YyvmEvex] = 1

	for i := 0; i < MAXREG; i++ {
		reg[i] = -1
		if i >= REG_AL && i <= REG_R15B {
			reg[i] = (i - REG_AL) & 7
			if i >= REG_SPB && i <= REG_DIB {
				regrex[i] = 0x40
			}
			if i >= REG_R8B && i <= REG_R15B {
				regrex[i] = Rxr | Rxx | Rxb
			}
		}

		if i >= REG_AH && i <= REG_BH {
			reg[i] = 4 + ((i - REG_AH) & 7)
		}
		if i >= REG_AX && i <= REG_R15 {
			reg[i] = (i - REG_AX) & 7
			if i >= REG_R8 {
				regrex[i] = Rxr | Rxx | Rxb
			}
		}

		if i >= REG_F0 && i <= REG_F0+7 {
			reg[i] = (i - REG_F0) & 7
		}
		if i >= REG_M0 && i <= REG_M0+7 {
			reg[i] = (i - REG_M0) & 7
		}
		if i >= REG_K0 && i <= REG_K0+7 {
			reg[i] = (i - REG_K0) & 7
		}
		if i >= REG_X0 && i <= REG_X0+15 {
			reg[i] = (i - REG_X0) & 7
			if i >= REG_X0+8 {
				regrex[i] = Rxr | Rxx | Rxb
			}
		}
		if i >= REG_X16 && i <= REG_X16+15 {
			reg[i] = (i - REG_X16) & 7
			if i >= REG_X16+8 {
				regrex[i] = Rxr | Rxx | Rxb | RxrEvex
			} else {
				regrex[i] = RxrEvex
			}
		}
		if i >= REG_Y0 && i <= REG_Y0+15 {
			reg[i] = (i - REG_Y0) & 7
			if i >= REG_Y0+8 {
				regrex[i] = Rxr | Rxx | Rxb
			}
		}
		if i >= REG_Y16 && i <= REG_Y16+15 {
			reg[i] = (i - REG_Y16) & 7
			if i >= REG_Y16+8 {
				regrex[i] = Rxr | Rxx | Rxb | RxrEvex
			} else {
				regrex[i] = RxrEvex
			}
		}
		if i >= REG_Z0 && i <= REG_Z0+15 {
			reg[i] = (i - REG_Z0) & 7
			if i > REG_Z0+7 {
				regrex[i] = Rxr | Rxx | Rxb
			}
		}
		if i >= REG_Z16 && i <= REG_Z16+15 {
			reg[i] = (i - REG_Z16) & 7
			if i >= REG_Z16+8 {
				regrex[i] = Rxr | Rxx | Rxb | RxrEvex
			} else {
				regrex[i] = RxrEvex
			}
		}

		if i >= REG_CR+8 && i <= REG_CR+15 {
			regrex[i] = Rxr
		}
	}
}

var isAndroid = buildcfg.GOOS == "android"

func prefixof(ctxt *obj.Link, a *obj.Addr) int {
	if a.Reg < REG_CS && a.Index < REG_CS { // fast path
		return 0
	}
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

		case REG_TLS:
			// NOTE: Systems listed here should be only systems that
			// support direct TLS references like 8(TLS) implemented as
			// direct references from FS or GS. Systems that require
			// the initial-exec model, where you load the TLS base into
			// a register and then index from that register, do not reach
			// this code and should not be listed.
			if ctxt.Arch.Family == sys.I386 {
				switch ctxt.Headtype {
				default:
					if isAndroid {
						return 0x65 // GS
					}
					log.Fatalf("unknown TLS base register for %v", ctxt.Headtype)

				case objabi.Hdarwin,
					objabi.Hdragonfly,
					objabi.Hfreebsd,
					objabi.Hnetbsd,
					objabi.Hopenbsd:
					return 0x65 // GS
				}
			}

			switch ctxt.Headtype {
			default:
				log.Fatalf("unknown TLS base register for %v", ctxt.Headtype)

			case objabi.Hlinux:
				if isAndroid {
					return 0x64 // FS
				}

				if ctxt.Flag_shared {
					log.Fatalf("unknown TLS base register for linux with -shared")
				} else {
					return 0x64 // FS
				}

			case objabi.Hdragonfly,
				objabi.Hfreebsd,
				objabi.Hnetbsd,
				objabi.Hopenbsd,
				objabi.Hsolaris:
				return 0x64 // FS

			case objabi.Hdarwin:
				return 0x65 // GS
			}
		}
	}

	switch a.Index {
	case REG_CS:
		return 0x2e

	case REG_DS:
		return 0x3e

	case REG_ES:
		return 0x26

	case REG_TLS:
		if ctxt.Flag_shared && ctxt.Headtype != objabi.Hwindows {
			// When building for inclusion into a shared library, an instruction of the form
			//     MOV off(CX)(TLS*1), AX
			// becomes
			//     mov %gs:off(%ecx), %eax // on i386
			//     mov %fs:off(%rcx), %rax // on amd64
			// which assumes that the correct TLS offset has been loaded into CX (today
			// there is only one TLS variable -- g -- so this is OK). When not building for
			// a shared library the instruction it becomes
			//     mov 0x0(%ecx), %eax // on i386
			//     mov 0x0(%rcx), %rax // on amd64
			// and a R_TLS_LE relocation, and so does not require a prefix.
			if ctxt.Arch.Family == sys.I386 {
				return 0x65 // GS
			}
			return 0x64 // FS
		}

	case REG_FS:
		return 0x64

	case REG_GS:
		return 0x65
	}

	return 0
}

// oclassRegList returns multisource operand class for addr.
func oclassRegList(ctxt *obj.Link, addr *obj.Addr) int {
	// TODO(quasilyte): when oclass register case is refactored into
	// lookup table, use it here to get register kind more easily.
	// Helper functions like regIsXmm should go away too (they will become redundant).

	regIsXmm := func(r int) bool { return r >= REG_X0 && r <= REG_X31 }
	regIsYmm := func(r int) bool { return r >= REG_Y0 && r <= REG_Y31 }
	regIsZmm := func(r int) bool { return r >= REG_Z0 && r <= REG_Z31 }

	reg0, reg1 := decodeRegisterRange(addr.Offset)
	low := regIndex(int16(reg0))
	high := regIndex(int16(reg1))

	if ctxt.Arch.Family == sys.I386 {
		if low >= 8 || high >= 8 {
			return Yxxx
		}
	}

	switch high - low {
	case 3:
		switch {
		case regIsXmm(reg0) && regIsXmm(reg1):
			return YxrEvexMulti4
		case regIsYmm(reg0) && regIsYmm(reg1):
			return YyrEvexMulti4
		case regIsZmm(reg0) && regIsZmm(reg1):
			return YzrMulti4
		default:
			return Yxxx
		}
	default:
		return Yxxx
	}
}

// oclassVMem returns V-mem (vector memory with VSIB) operand class.
// For addr that is not V-mem returns (Yxxx, false).
func oclassVMem(ctxt *obj.Link, addr *obj.Addr) (int, bool) {
	switch addr.Index {
	case REG_X0 + 0,
		REG_X0 + 1,
		REG_X0 + 2,
		REG_X0 + 3,
		REG_X0 + 4,
		REG_X0 + 5,
		REG_X0 + 6,
		REG_X0 + 7:
		return Yxvm, true
	case REG_X8 + 0,
		REG_X8 + 1,
		REG_X8 + 2,
		REG_X8 + 3,
		REG_X8 + 4,
		REG_X8 + 5,
		REG_X8 + 6,
		REG_X8 + 7:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx, true
		}
		return Yxvm, true
	case REG_X16 + 0,
		REG_X16 + 1,
		REG_X16 + 2,
		REG_X16 + 3,
		REG_X16 + 4,
		REG_X16 + 5,
		REG_X16 + 6,
		REG_X16 + 7,
		REG_X16 + 8,
		REG_X16 + 9,
		REG_X16 + 10,
		REG_X16 + 11,
		REG_X16 + 12,
		REG_X16 + 13,
		REG_X16 + 14,
		REG_X16 + 15:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx, true
		}
		return YxvmEvex, true

	case REG_Y0 + 0,
		REG_Y0 + 1,
		REG_Y0 + 2,
		REG_Y0 + 3,
		REG_Y0 + 4,
		REG_Y0 + 5,
		REG_Y0 + 6,
		REG_Y0 + 7:
		return Yyvm, true
	case REG_Y8 + 0,
		REG_Y8 + 1,
		REG_Y8 + 2,
		REG_Y8 + 3,
		REG_Y8 + 4,
		REG_Y8 + 5,
		REG_Y8 + 6,
		REG_Y8 + 7:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx, true
		}
		return Yyvm, true
	case REG_Y16 + 0,
		REG_Y16 + 1,
		REG_Y16 + 2,
		REG_Y16 + 3,
		REG_Y16 + 4,
		REG_Y16 + 5,
		REG_Y16 + 6,
		REG_Y16 + 7,
		REG_Y16 + 8,
		REG_Y16 + 9,
		REG_Y16 + 10,
		REG_Y16 + 11,
		REG_Y16 + 12,
		REG_Y16 + 13,
		REG_Y16 + 14,
		REG_Y16 + 15:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx, true
		}
		return YyvmEvex, true

	case REG_Z0 + 0,
		REG_Z0 + 1,
		REG_Z0 + 2,
		REG_Z0 + 3,
		REG_Z0 + 4,
		REG_Z0 + 5,
		REG_Z0 + 6,
		REG_Z0 + 7:
		return Yzvm, true
	case REG_Z8 + 0,
		REG_Z8 + 1,
		REG_Z8 + 2,
		REG_Z8 + 3,
		REG_Z8 + 4,
		REG_Z8 + 5,
		REG_Z8 + 6,
		REG_Z8 + 7,
		REG_Z8 + 8,
		REG_Z8 + 9,
		REG_Z8 + 10,
		REG_Z8 + 11,
		REG_Z8 + 12,
		REG_Z8 + 13,
		REG_Z8 + 14,
		REG_Z8 + 15,
		REG_Z8 + 16,
		REG_Z8 + 17,
		REG_Z8 + 18,
		REG_Z8 + 19,
		REG_Z8 + 20,
		REG_Z8 + 21,
		REG_Z8 + 22,
		REG_Z8 + 23:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx, true
		}
		return Yzvm, true
	}

	return Yxxx, false
}

func oclass(ctxt *obj.Link, p *obj.Prog, a *obj.Addr) int {
	switch a.Type {
	case obj.TYPE_REGLIST:
		return oclassRegList(ctxt, a)

	case obj.TYPE_NONE:
		return Ynone

	case obj.TYPE_BRANCH:
		return Ybr

	case obj.TYPE_INDIR:
		if a.Name != obj.NAME_NONE && a.Reg == REG_NONE && a.Index == REG_NONE && a.Scale == 0 {
			return Yindir
		}
		return Yxxx

	case obj.TYPE_MEM:
		// Pseudo registers have negative index, but SP is
		// not pseudo on x86, hence REG_SP check is not redundant.
		if a.Index == REG_SP || a.Index < 0 {
			// Can't use FP/SB/PC/SP as the index register.
			return Yxxx
		}

		if vmem, ok := oclassVMem(ctxt, a); ok {
			return vmem
		}

		if ctxt.Arch.Family == sys.AMD64 {
			switch a.Name {
			case obj.NAME_EXTERN, obj.NAME_STATIC, obj.NAME_GOTREF:
				// Global variables can't use index registers and their
				// base register is %rip (%rip is encoded as REG_NONE).
				if a.Reg != REG_NONE || a.Index != REG_NONE || a.Scale != 0 {
					return Yxxx
				}
			case obj.NAME_AUTO, obj.NAME_PARAM:
				// These names must have a base of SP.  The old compiler
				// uses 0 for the base register. SSA uses REG_SP.
				if a.Reg != REG_SP && a.Reg != 0 {
					return Yxxx
				}
			case obj.NAME_NONE:
				// everything is ok
			default:
				// unknown name
				return Yxxx
			}
		}
		return Ym

	case obj.TYPE_ADDR:
		switch a.Name {
		case obj.NAME_GOTREF:
			ctxt.Diag("unexpected TYPE_ADDR with NAME_GOTREF")
			return Yxxx

		case obj.NAME_EXTERN,
			obj.NAME_STATIC:
			if a.Sym != nil && useAbs(ctxt, a.Sym) {
				return Yi32
			}
			return Yiauto // use pc-relative addressing

		case obj.NAME_AUTO,
			obj.NAME_PARAM:
			return Yiauto
		}

		// TODO(rsc): DUFFZERO/DUFFCOPY encoding forgot to set a->index
		// and got Yi32 in an earlier version of this code.
		// Keep doing that until we fix yduff etc.
		if a.Sym != nil && strings.HasPrefix(a.Sym.Name, "runtime.duff") {
			return Yi32
		}

		if a.Sym != nil || a.Name != obj.NAME_NONE {
			ctxt.Diag("unexpected addr: %v", obj.Dconv(p, a))
		}
		fallthrough

	case obj.TYPE_CONST:
		if a.Sym != nil {
			ctxt.Diag("TYPE_CONST with symbol: %v", obj.Dconv(p, a))
		}

		v := a.Offset
		if ctxt.Arch.Family == sys.I386 {
			v = int64(int32(v))
		}
		switch {
		case v == 0:
			return Yi0
		case v == 1:
			return Yi1
		case v >= 0 && v <= 3:
			return Yu2
		case v >= 0 && v <= 127:
			return Yu7
		case v >= 0 && v <= 255:
			return Yu8
		case v >= -128 && v <= 127:
			return Yi8
		}
		if ctxt.Arch.Family == sys.I386 {
			return Yi32
		}
		l := int32(v)
		if int64(l) == v {
			return Ys32 // can sign extend
		}
		if v>>32 == 0 {
			return Yi32 // unsigned
		}
		return Yi64

	case obj.TYPE_TEXTSIZE:
		return Ytextsize
	}

	if a.Type != obj.TYPE_REG {
		ctxt.Diag("unexpected addr1: type=%d %v", a.Type, obj.Dconv(p, a))
		return Yxxx
	}

	switch a.Reg {
	case REG_AL:
		return Yal

	case REG_AX:
		return Yax

		/*
			case REG_SPB:
		*/
	case REG_BPB,
		REG_SIB,
		REG_DIB,
		REG_R8B,
		REG_R9B,
		REG_R10B,
		REG_R11B,
		REG_R12B,
		REG_R13B,
		REG_R14B,
		REG_R15B:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx
		}
		fallthrough

	case REG_DL,
		REG_BL,
		REG_AH,
		REG_CH,
		REG_DH,
		REG_BH:
		return Yrb

	case REG_CL:
		return Ycl

	case REG_CX:
		return Ycx

	case REG_DX, REG_BX:
		return Yrx

	case REG_R8, // not really Yrl
		REG_R9,
		REG_R10,
		REG_R11,
		REG_R12,
		REG_R13,
		REG_R14,
		REG_R15:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx
		}
		fallthrough

	case REG_SP, REG_BP, REG_SI, REG_DI:
		if ctxt.Arch.Family == sys.I386 {
			return Yrl32
		}
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

	case REG_M0 + 0,
		REG_M0 + 1,
		REG_M0 + 2,
		REG_M0 + 3,
		REG_M0 + 4,
		REG_M0 + 5,
		REG_M0 + 6,
		REG_M0 + 7:
		return Ymr

	case REG_X0:
		return Yxr0

	case REG_X0 + 1,
		REG_X0 + 2,
		REG_X0 + 3,
		REG_X0 + 4,
		REG_X0 + 5,
		REG_X0 + 6,
		REG_X0 + 7,
		REG_X0 + 8,
		REG_X0 + 9,
		REG_X0 + 10,
		REG_X0 + 11,
		REG_X0 + 12,
		REG_X0 + 13,
		REG_X0 + 14,
		REG_X0 + 15:
		return Yxr

	case REG_X0 + 16,
		REG_X0 + 17,
		REG_X0 + 18,
		REG_X0 + 19,
		REG_X0 + 20,
		REG_X0 + 21,
		REG_X0 + 22,
		REG_X0 + 23,
		REG_X0 + 24,
		REG_X0 + 25,
		REG_X0 + 26,
		REG_X0 + 27,
		REG_X0 + 28,
		REG_X0 + 29,
		REG_X0 + 30,
		REG_X0 + 31:
		return YxrEvex

	case REG_Y0 + 0,
		REG_Y0 + 1,
		REG_Y0 + 2,
		REG_Y0 + 3,
		REG_Y0 + 4,
		REG_Y0 + 5,
		REG_Y0 + 6,
		REG_Y0 + 7,
		REG_Y0 + 8,
		REG_Y0 + 9,
		REG_Y0 + 10,
		REG_Y0 + 11,
		REG_Y0 + 12,
		REG_Y0 + 13,
		REG_Y0 + 14,
		REG_Y0 + 15:
		return Yyr

	case REG_Y0 + 16,
		REG_Y0 + 17,
		REG_Y0 + 18,
		REG_Y0 + 19,
		REG_Y0 + 20,
		REG_Y0 + 21,
		REG_Y0 + 22,
		REG_Y0 + 23,
		REG_Y0 + 24,
		REG_Y0 + 25,
		REG_Y0 + 26,
		REG_Y0 + 27,
		REG_Y0 + 28,
		REG_Y0 + 29,
		REG_Y0 + 30,
		REG_Y0 + 31:
		return YyrEvex

	case REG_Z0 + 0,
		REG_Z0 + 1,
		REG_Z0 + 2,
		REG_Z0 + 3,
		REG_Z0 + 4,
		REG_Z0 + 5,
		REG_Z0 + 6,
		REG_Z0 + 7:
		return Yzr

	case REG_Z0 + 8,
		REG_Z0 + 9,
		REG_Z0 + 10,
		REG_Z0 + 11,
		REG_Z0 + 12,
		REG_Z0 + 13,
		REG_Z0 + 14,
		REG_Z0 + 15,
		REG_Z0 + 16,
		REG_Z0 + 17,
		REG_Z0 + 18,
		REG_Z0 + 19,
		REG_Z0 + 20,
		REG_Z0 + 21,
		REG_Z0 + 22,
		REG_Z0 + 23,
		REG_Z0 + 24,
		REG_Z0 + 25,
		REG_Z0 + 26,
		REG_Z0 + 27,
		REG_Z0 + 28,
		REG_Z0 + 29,
		REG_Z0 + 30,
		REG_Z0 + 31:
		if ctxt.Arch.Family == sys.I386 {
			return Yxxx
		}
		return Yzr

	case REG_K0:
		return Yk0

	case REG_K0 + 1,
		REG_K0 + 2,
		REG_K0 + 3,
		REG_K0 + 4,
		REG_K0 + 5,
		REG_K0 + 6,
		REG_K0 + 7:
		return Yknot0

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
	case REG_CR + 8:
		return Ycr8

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

// AsmBuf is a simple buffer to assemble variable-length x86 instructions into
// and hold assembly state.
type AsmBuf struct {
	buf      [100]byte
	off      int
	rexflag  int
	vexflag  bool // Per inst: true for VEX-encoded
	evexflag bool // Per inst: true for EVEX-encoded
	rep      bool
	repn     bool
	lock     bool

	evex evexBits // Initialized when evexflag is true
}

// Put1 appends one byte to the end of the buffer.
func (ab *AsmBuf) Put1(x byte) {
	ab.buf[ab.off] = x
	ab.off++
}

// Put2 appends two bytes to the end of the buffer.
func (ab *AsmBuf) Put2(x, y byte) {
	ab.buf[ab.off+0] = x
	ab.buf[ab.off+1] = y
	ab.off += 2
}

// Put3 appends three bytes to the end of the buffer.
func (ab *AsmBuf) Put3(x, y, z byte) {
	ab.buf[ab.off+0] = x
	ab.buf[ab.off+1] = y
	ab.buf[ab.off+2] = z
	ab.off += 3
}

// Put4 appends four bytes to the end of the buffer.
func (ab *AsmBuf) Put4(x, y, z, w byte) {
	ab.buf[ab.off+0] = x
	ab.buf[ab.off+1] = y
	ab.buf[ab.off+2] = z
	ab.buf[ab.off+3] = w
	ab.off += 4
}

// PutInt16 writes v into the buffer using little-endian encoding.
func (ab *AsmBuf) PutInt16(v int16) {
	ab.buf[ab.off+0] = byte(v)
	ab.buf[ab.off+1] = byte(v >> 8)
	ab.off += 2
}

// PutInt32 writes v into the buffer using little-endian encoding.
func (ab *AsmBuf) PutInt32(v int32) {
	ab.buf[ab.off+0] = byte(v)
	ab.buf[ab.off+1] = byte(v >> 8)
	ab.buf[ab.off+2] = byte(v >> 16)
	ab.buf[ab.off+3] = byte(v >> 24)
	ab.off += 4
}

// PutInt64 writes v into the buffer using little-endian encoding.
func (ab *AsmBuf) PutInt64(v int64) {
	ab.buf[ab.off+0] = byte(v)
	ab.buf[ab.off+1] = byte(v >> 8)
	ab.buf[ab.off+2] = byte(v >> 16)
	ab.buf[ab.off+3] = byte(v >> 24)
	ab.buf[ab.off+4] = byte(v >> 32)
	ab.buf[ab.off+5] = byte(v >> 40)
	ab.buf[ab.off+6] = byte(v >> 48)
	ab.buf[ab.off+7] = byte(v >> 56)
	ab.off += 8
}

// Put copies b into the buffer.
func (ab *AsmBuf) Put(b []byte) {
	copy(ab.buf[ab.off:], b)
	ab.off += len(b)
}

// PutOpBytesLit writes zero terminated sequence of bytes from op,
// starting at specified offset (e.g. z counter value).
// Trailing 0 is not written.
//
// Intended to be used for literal Z cases.
// Literal Z cases usually have "Zlit" in their name (Zlit, Zlitr_m, Zlitm_r).
func (ab *AsmBuf) PutOpBytesLit(offset int, op *opBytes) {
	for int(op[offset]) != 0 {
		ab.Put1(byte(op[offset]))
		offset++
	}
}

// Insert inserts b at offset i.
func (ab *AsmBuf) Insert(i int, b byte) {
	ab.off++
	copy(ab.buf[i+1:ab.off], ab.buf[i:ab.off-1])
	ab.buf[i] = b
}

// Last returns the byte at the end of the buffer.
func (ab *AsmBuf) Last() byte { return ab.buf[ab.off-1] }

// Len returns the length of the buffer.
func (ab *AsmBuf) Len() int { return ab.off }

// Bytes returns the contents of the buffer.
func (ab *AsmBuf) Bytes() []byte { return ab.buf[:ab.off] }

// Reset empties the buffer.
func (ab *AsmBuf) Reset() { ab.off = 0 }

// At returns the byte at offset i.
func (ab *AsmBuf) At(i int) byte { return ab.buf[i] }

// asmidx emits SIB byte.
func (ab *AsmBuf) asmidx(ctxt *obj.Link, scale int, index int, base int) {
	var i int

	// X/Y index register is used in VSIB.
	switch index {
	default:
		goto bad

	case REG_NONE:
		i = 4 << 3
		goto bas

	case REG_R8,
		REG_R9,
		REG_R10,
		REG_R11,
		REG_R12,
		REG_R13,
		REG_R14,
		REG_R15,
		REG_X8,
		REG_X9,
		REG_X10,
		REG_X11,
		REG_X12,
		REG_X13,
		REG_X14,
		REG_X15,
		REG_X16,
		REG_X17,
		REG_X18,
		REG_X19,
		REG_X20,
		REG_X21,
		REG_X22,
		REG_X23,
		REG_X24,
		REG_X25,
		REG_X26,
		REG_X27,
		REG_X28,
		REG_X29,
		REG_X30,
		REG_X31,
		REG_Y8,
		REG_Y9,
		REG_Y10,
		REG_Y11,
		REG_Y12,
		REG_Y13,
		REG_Y14,
		REG_Y15,
		REG_Y16,
		REG_Y17,
		REG_Y18,
		REG_Y19,
		REG_Y20,
		REG_Y21,
		REG_Y22,
		REG_Y23,
		REG_Y24,
		REG_Y25,
		REG_Y26,
		REG_Y27,
		REG_Y28,
		REG_Y29,
		REG_Y30,
		REG_Y31,
		REG_Z8,
		REG_Z9,
		REG_Z10,
		REG_Z11,
		REG_Z12,
		REG_Z13,
		REG_Z14,
		REG_Z15,
		REG_Z16,
		REG_Z17,
		REG_Z18,
		REG_Z19,
		REG_Z20,
		REG_Z21,
		REG_Z22,
		REG_Z23,
		REG_Z24,
		REG_Z25,
		REG_Z26,
		REG_Z27,
		REG_Z28,
		REG_Z29,
		REG_Z30,
		REG_Z31:
		if ctxt.Arch.Family == sys.I386 {
			goto bad
		}
		fallthrough

	case REG_AX,
		REG_CX,
		REG_DX,
		REG_BX,
		REG_BP,
		REG_SI,
		REG_DI,
		REG_X0,
		REG_X1,
		REG_X2,
		REG_X3,
		REG_X4,
		REG_X5,
		REG_X6,
		REG_X7,
		REG_Y0,
		REG_Y1,
		REG_Y2,
		REG_Y3,
		REG_Y4,
		REG_Y5,
		REG_Y6,
		REG_Y7,
		REG_Z0,
		REG_Z1,
		REG_Z2,
		REG_Z3,
		REG_Z4,
		REG_Z5,
		REG_Z6,
		REG_Z7:
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

	case REG_NONE: // must be mod=00
		i |= 5

	case REG_R8,
		REG_R9,
		REG_R10,
		REG_R11,
		REG_R12,
		REG_R13,
		REG_R14,
		REG_R15:
		if ctxt.Arch.Family == sys.I386 {
			goto bad
		}
		fallthrough

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

	ab.Put1(byte(i))
	return

bad:
	ctxt.Diag("asmidx: bad address %d/%d/%d", scale, index, base)
	ab.Put1(0)
}

func (ab *AsmBuf) relput4(ctxt *obj.Link, cursym *obj.LSym, p *obj.Prog, a *obj.Addr) {
	var rel obj.Reloc

	v := vaddr(ctxt, p, a, &rel)
	if rel.Siz != 0 {
		if rel.Siz != 4 {
			ctxt.Diag("bad reloc")
		}
		r := obj.Addrel(cursym)
		*r = rel
		r.Off = int32(p.Pc + int64(ab.Len()))
	}

	ab.PutInt32(int32(v))
}

func vaddr(ctxt *obj.Link, p *obj.Prog, a *obj.Addr, r *obj.Reloc) int64 {
	if r != nil {
		*r = obj.Reloc{}
	}

	switch a.Name {
	case obj.NAME_STATIC,
		obj.NAME_GOTREF,
		obj.NAME_EXTERN:
		s := a.Sym
		if r == nil {
			ctxt.Diag("need reloc for %v", obj.Dconv(p, a))
			log.Fatalf("reloc")
		}

		if a.Name == obj.NAME_GOTREF {
			r.Siz = 4
			r.Type = objabi.R_GOTPCREL
		} else if useAbs(ctxt, s) {
			r.Siz = 4
			r.Type = objabi.R_ADDR
		} else {
			r.Siz = 4
			r.Type = objabi.R_PCREL
		}

		r.Off = -1 // caller must fill in
		r.Sym = s
		r.Add = a.Offset

		return 0
	}

	if (a.Type == obj.TYPE_MEM || a.Type == obj.TYPE_ADDR) && a.Reg == REG_TLS {
		if r == nil {
			ctxt.Diag("need reloc for %v", obj.Dconv(p, a))
			log.Fatalf("reloc")
		}

		if !ctxt.Flag_shared || isAndroid || ctxt.Headtype == objabi.Hdarwin {
			r.Type = objabi.R_TLS_LE
			r.Siz = 4
			r.Off = -1 // caller must fill in
			r.Add = a.Offset
		}
		return 0
	}

	return a.Offset
}

func (ab *AsmBuf) asmandsz(ctxt *obj.Link, cursym *obj.LSym, p *obj.Prog, a *obj.Addr, r int, rex int, m64 int) {
	var base int
	var rel obj.Reloc

	rex &= 0x40 | Rxr
	if a.Offset != int64(int32(a.Offset)) {
		// The rules are slightly different for 386 and AMD64,
		// mostly for historical reasons. We may unify them later,
		// but it must be discussed beforehand.
		//
		// For 64bit mode only LEAL is allowed to overflow.
		// It's how https://golang.org/cl/59630 made it.
		// crypto/sha1/sha1block_amd64.s depends on this feature.
		//
		// For 32bit mode rules are more permissive.
		// If offset fits uint32, it's permitted.
		// This is allowed for assembly that wants to use 32-bit hex
		// constants, e.g. LEAL 0x99999999(AX), AX.
		overflowOK := (ctxt.Arch.Family == sys.AMD64 && p.As == ALEAL) ||
			(ctxt.Arch.Family != sys.AMD64 &&
				int64(uint32(a.Offset)) == a.Offset &&
				ab.rexflag&Rxw == 0)
		if !overflowOK {
			ctxt.Diag("offset too large in %s", p)
		}
	}
	v := int32(a.Offset)
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
		const regFirst = REG_AL
		const regLast = REG_Z31
		if a.Reg < regFirst || regLast < a.Reg {
			goto bad
		}
		if v != 0 {
			goto bad
		}
		ab.Put1(byte(3<<6 | reg[a.Reg]<<0 | r<<3))
		ab.rexflag |= regrex[a.Reg]&(0x40|Rxb) | rex
		return
	}

	if a.Type != obj.TYPE_MEM {
		goto bad
	}

	if a.Index != REG_NONE && a.Index != REG_TLS && !(REG_CS <= a.Index && a.Index <= REG_GS) {
		base := int(a.Reg)
		switch a.Name {
		case obj.NAME_EXTERN,
			obj.NAME_GOTREF,
			obj.NAME_STATIC:
			if !useAbs(ctxt, a.Sym) && ctxt.Arch.Family == sys.AMD64 {
				goto bad
			}
			if ctxt.Arch.Family == sys.I386 && ctxt.Flag_shared {
				// The base register has already been set. It holds the PC
				// of this instruction returned by a PC-reading thunk.
				// See obj6.go:rewriteToPcrel.
			} else {
				base = REG_NONE
			}
			v = int32(vaddr(ctxt, p, a, &rel))

		case obj.NAME_AUTO,
			obj.NAME_PARAM:
			base = REG_SP
		}

		ab.rexflag |= regrex[int(a.Index)]&Rxx | regrex[base]&Rxb | rex
		if base == REG_NONE {
			ab.Put1(byte(0<<6 | 4<<0 | r<<3))
			ab.asmidx(ctxt, int(a.Scale), int(a.Index), base)
			goto putrelv
		}

		if v == 0 && rel.Siz == 0 && base != REG_BP && base != REG_R13 {
			ab.Put1(byte(0<<6 | 4<<0 | r<<3))
			ab.asmidx(ctxt, int(a.Scale), int(a.Index), base)
			return
		}

		if disp8, ok := toDisp8(v, p, ab); ok && rel.Siz == 0 {
			ab.Put1(byte(1<<6 | 4<<0 | r<<3))
			ab.asmidx(ctxt, int(a.Scale), int(a.Index), base)
			ab.Put1(disp8)
			return
		}

		ab.Put1(byte(2<<6 | 4<<0 | r<<3))
		ab.asmidx(ctxt, int(a.Scale), int(a.Index), base)
		goto putrelv
	}

	base = int(a.Reg)
	switch a.Name {
	case obj.NAME_STATIC,
		obj.NAME_GOTREF,
		obj.NAME_EXTERN:
		if a.Sym == nil {
			ctxt.Diag("bad addr: %v", p)
		}
		if ctxt.Arch.Family == sys.I386 && ctxt.Flag_shared {
			// The base register has already been set. It holds the PC
			// of this instruction returned by a PC-reading thunk.
			// See obj6.go:rewriteToPcrel.
		} else {
			base = REG_NONE
		}
		v = int32(vaddr(ctxt, p, a, &rel))

	case obj.NAME_AUTO,
		obj.NAME_PARAM:
		base = REG_SP
	}

	if base == REG_TLS {
		v = int32(vaddr(ctxt, p, a, &rel))
	}

	ab.rexflag |= regrex[base]&Rxb | rex
	if base == REG_NONE || (REG_CS <= base && base <= REG_GS) || base == REG_TLS {
		if (a.Sym == nil || !useAbs(ctxt, a.Sym)) && base == REG_NONE && (a.Name == obj.NAME_STATIC || a.Name == obj.NAME_EXTERN || a.Name == obj.NAME_GOTREF) || ctxt.Arch.Family != sys.AMD64 {
			if a.Name == obj.NAME_GOTREF && (a.Offset != 0 || a.Index != 0 || a.Scale != 0) {
				ctxt.Diag("%v has offset against gotref", p)
			}
			ab.Put1(byte(0<<6 | 5<<0 | r<<3))
			goto putrelv
		}

		// temporary
		ab.Put2(
			byte(0<<6|4<<0|r<<3), // sib present
			0<<6|4<<3|5<<0,       // DS:d32
		)
		goto putrelv
	}

	if base == REG_SP || base == REG_R12 {
		if v == 0 {
			ab.Put1(byte(0<<6 | reg[base]<<0 | r<<3))
			ab.asmidx(ctxt, int(a.Scale), REG_NONE, base)
			return
		}

		if disp8, ok := toDisp8(v, p, ab); ok {
			ab.Put1(byte(1<<6 | reg[base]<<0 | r<<3))
			ab.asmidx(ctxt, int(a.Scale), REG_NONE, base)
			ab.Put1(disp8)
			return
		}

		ab.Put1(byte(2<<6 | reg[base]<<0 | r<<3))
		ab.asmidx(ctxt, int(a.Scale), REG_NONE, base)
		goto putrelv
	}

	if REG_AX <= base && base <= REG_R15 {
		if a.Index == REG_TLS && !ctxt.Flag_shared && !isAndroid &&
			ctxt.Headtype != objabi.Hwindows {
			rel = obj.Reloc{}
			rel.Type = objabi.R_TLS_LE
			rel.Siz = 4
			rel.Sym = nil
			rel.Add = int64(v)
			v = 0
		}

		if v == 0 && rel.Siz == 0 && base != REG_BP && base != REG_R13 {
			ab.Put1(byte(0<<6 | reg[base]<<0 | r<<3))
			return
		}

		if disp8, ok := toDisp8(v, p, ab); ok && rel.Siz == 0 {
			ab.Put2(byte(1<<6|reg[base]<<0|r<<3), disp8)
			return
		}

		ab.Put1(byte(2<<6 | reg[base]<<0 | r<<3))
		goto putrelv
	}

	goto bad

putrelv:
	if rel.Siz != 0 {
		if rel.Siz != 4 {
			ctxt.Diag("bad rel")
			goto bad
		}

		r := obj.Addrel(cursym)
		*r = rel
		r.Off = int32(p.Pc + int64(ab.Len()))
	}

	ab.PutInt32(v)
	return

bad:
	ctxt.Diag("asmand: bad address %v", obj.Dconv(p, a))
}

func (ab *AsmBuf) asmand(ctxt *obj.Link, cursym *obj.LSym, p *obj.Prog, a *obj.Addr, ra *obj.Addr) {
	ab.asmandsz(ctxt, cursym, p, a, reg[ra.Reg], regrex[ra.Reg], 0)
}

func (ab *AsmBuf) asmando(ctxt *obj.Link, cursym *obj.LSym, p *obj.Prog, a *obj.Addr, o int) {
	ab.asmandsz(ctxt, cursym, p, a, o, 0, 0)
}

func bytereg(a *obj.Addr, t *uint8) {
	if a.Type == obj.TYPE_REG && a.Index == REG_NONE && (REG_AX <= a.Reg && a.Reg <= REG_R15) {
		a.Reg += REG_AL - REG_AX
		*t = 0
	}
}

func unbytereg(a *obj.Addr, t *uint8) {
	if a.Type == obj.TYPE_REG && a.Index == REG_NONE && (REG_AL <= a.Reg && a.Reg <= REG_R15B) {
		a.Reg += REG_AX - REG_AL
		*t = 0
	}
}

const (
	movLit uint8 = iota // Like Zlit
	movRegMem
	movMemReg
	movRegMem2op
	movMemReg2op
	movFullPtr // Load full pointer, trash heap (unsupported)
	movDoubleShift
	movTLSReg
)

var ymovtab = []movtab{
	// push
	{APUSHL, Ycs, Ynone, Ynone, movLit, [4]uint8{0x0e, 0}},
	{APUSHL, Yss, Ynone, Ynone, movLit, [4]uint8{0x16, 0}},
	{APUSHL, Yds, Ynone, Ynone, movLit, [4]uint8{0x1e, 0}},
	{APUSHL, Yes, Ynone, Ynone, movLit, [4]uint8{0x06, 0}},
	{APUSHL, Yfs, Ynone, Ynone, movLit, [4]uint8{0x0f, 0xa0, 0}},
	{APUSHL, Ygs, Ynone, Ynone, movLit, [4]uint8{0x0f, 0xa8, 0}},
	{APUSHQ, Yfs, Ynone, Ynone, movLit, [4]uint8{0x0f, 0xa0, 0}},
	{APUSHQ, Ygs, Ynone, Ynone, movLit, [4]uint8{0x0f, 0xa8, 0}},
	{APUSHW, Ycs, Ynone, Ynone, movLit, [4]uint8{Pe, 0x0e, 0}},
	{APUSHW, Yss, Ynone, Ynone, movLit, [4]uint8{Pe, 0x16, 0}},
	{APUSHW, Yds, Ynone, Ynone, movLit, [4]uint8{Pe, 0x1e, 0}},
	{APUSHW, Yes, Ynone, Ynone, movLit, [4]uint8{Pe, 0x06, 0}},
	{APUSHW, Yfs, Ynone, Ynone, movLit, [4]uint8{Pe, 0x0f, 0xa0, 0}},
	{APUSHW, Ygs, Ynone, Ynone, movLit, [4]uint8{Pe, 0x0f, 0xa8, 0}},

	// pop
	{APOPL, Ynone, Ynone, Yds, movLit, [4]uint8{0x1f, 0}},
	{APOPL, Ynone, Ynone, Yes, movLit, [4]uint8{0x07, 0}},
	{APOPL, Ynone, Ynone, Yss, movLit, [4]uint8{0x17, 0}},
	{APOPL, Ynone, Ynone, Yfs, movLit, [4]uint8{0x0f, 0xa1, 0}},
	{APOPL, Ynone, Ynone, Ygs, movLit, [4]uint8{0x0f, 0xa9, 0}},
	{APOPQ, Ynone, Ynone, Yfs, movLit, [4]uint8{0x0f, 0xa1, 0}},
	{APOPQ, Ynone, Ynone, Ygs, movLit, [4]uint8{0x0f, 0xa9, 0}},
	{APOPW, Ynone, Ynone, Yds, movLit, [4]uint8{Pe, 0x1f, 0}},
	{APOPW, Ynone, Ynone, Yes, movLit, [4]uint8{Pe, 0x07, 0}},
	{APOPW, Ynone, Ynone, Yss, movLit, [4]uint8{Pe, 0x17, 0}},
	{APOPW, Ynone, Ynone, Yfs, movLit, [4]uint8{Pe, 0x0f, 0xa1, 0}},
	{APOPW, Ynone, Ynone, Ygs, movLit, [4]uint8{Pe, 0x0f, 0xa9, 0}},

	// mov seg
	{AMOVW, Yes, Ynone, Yml, movRegMem, [4]uint8{0x8c, 0, 0, 0}},
	{AMOVW, Ycs, Ynone, Yml, movRegMem, [4]uint8{0x8c, 1, 0, 0}},
	{AMOVW, Yss, Ynone, Yml, movRegMem, [4]uint8{0x8c, 2, 0, 0}},
	{AMOVW, Yds, Ynone, Yml, movRegMem, [4]uint8{0x8c, 3, 0, 0}},
	{AMOVW, Yfs, Ynone, Yml, movRegMem, [4]uint8{0x8c, 4, 0, 0}},
	{AMOVW, Ygs, Ynone, Yml, movRegMem, [4]uint8{0x8c, 5, 0, 0}},
	{AMOVW, Yml, Ynone, Yes, movMemReg, [4]uint8{0x8e, 0, 0, 0}},
	{AMOVW, Yml, Ynone, Ycs, movMemReg, [4]uint8{0x8e, 1, 0, 0}},
	{AMOVW, Yml, Ynone, Yss, movMemReg, [4]uint8{0x8e, 2, 0, 0}},
	{AMOVW, Yml, Ynone, Yds, movMemReg, [4]uint8{0x8e, 3, 0, 0}},
	{AMOVW, Yml, Ynone, Yfs, movMemReg, [4]uint8{0x8e, 4, 0, 0}},
	{AMOVW, Yml, Ynone, Ygs, movMemReg, [4]uint8{0x8e, 5, 0, 0}},

	// mov cr
	{AMOVL, Ycr0, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 0, 0}},
	{AMOVL, Ycr2, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 2, 0}},
	{AMOVL, Ycr3, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 3, 0}},
	{AMOVL, Ycr4, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 4, 0}},
	{AMOVL, Ycr8, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 8, 0}},
	{AMOVQ, Ycr0, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 0, 0}},
	{AMOVQ, Ycr2, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 2, 0}},
	{AMOVQ, Ycr3, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 3, 0}},
	{AMOVQ, Ycr4, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 4, 0}},
	{AMOVQ, Ycr8, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x20, 8, 0}},
	{AMOVL, Yrl, Ynone, Ycr0, movMemReg2op, [4]uint8{0x0f, 0x22, 0, 0}},
	{AMOVL, Yrl, Ynone, Ycr2, movMemReg2op, [4]uint8{0x0f, 0x22, 2, 0}},
	{AMOVL, Yrl, Ynone, Ycr3, movMemReg2op, [4]uint8{0x0f, 0x22, 3, 0}},
	{AMOVL, Yrl, Ynone, Ycr4, movMemReg2op, [4]uint8{0x0f, 0x22, 4, 0}},
	{AMOVL, Yrl, Ynone, Ycr8, movMemReg2op, [4]uint8{0x0f, 0x22, 8, 0}},
	{AMOVQ, Yrl, Ynone, Ycr0, movMemReg2op, [4]uint8{0x0f, 0x22, 0, 0}},
	{AMOVQ, Yrl, Ynone, Ycr2, movMemReg2op, [4]uint8{0x0f, 0x22, 2, 0}},
	{AMOVQ, Yrl, Ynone, Ycr3, movMemReg2op, [4]uint8{0x0f, 0x22, 3, 0}},
	{AMOVQ, Yrl, Ynone, Ycr4, movMemReg2op, [4]uint8{0x0f, 0x22, 4, 0}},
	{AMOVQ, Yrl, Ynone, Ycr8, movMemReg2op, [4]uint8{0x0f, 0x22, 8, 0}},

	// mov dr
	{AMOVL, Ydr0, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 0, 0}},
	{AMOVL, Ydr6, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 6, 0}},
	{AMOVL, Ydr7, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 7, 0}},
	{AMOVQ, Ydr0, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 0, 0}},
	{AMOVQ, Ydr2, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 2, 0}},
	{AMOVQ, Ydr3, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 3, 0}},
	{AMOVQ, Ydr6, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 6, 0}},
	{AMOVQ, Ydr7, Ynone, Yrl, movRegMem2op, [4]uint8{0x0f, 0x21, 7, 0}},
	{AMOVL, Yrl, Ynone, Ydr0, movMemReg2op, [4]uint8{0x0f, 0x23, 0, 0}},
	{AMOVL, Yrl, Ynone, Ydr6, movMemReg2op, [4]uint8{0x0f, 0x23, 6, 0}},
	{AMOVL, Yrl, Ynone, Ydr7, movMemReg2op, [4]uint8{0x0f, 0x23, 7, 0}},
	{AMOVQ, Yrl, Ynone, Ydr0, movMemReg2op, [4]uint8{0x0f, 0x23, 0, 0}},
	{AMOVQ, Yrl, Ynone, Ydr2, movMemReg2op, [4]uint8{0x0f, 0x23, 2, 0}},
	{AMOVQ, Yrl, Ynone, Ydr3, movMemReg2op, [4]uint8{0x0f, 0x23, 3, 0}},
	{AMOVQ, Yrl, Ynone, Ydr6, movMemReg2op, [4]uint8{0x0f, 0x23, 6, 0}},
	{AMOVQ, Yrl, Ynone, Ydr7, movMemReg2op, [4]uint8{0x0f, 0x23, 7, 0}},

	// mov tr
	{AMOVL, Ytr6, Ynone, Yml, movRegMem2op, [4]uint8{0x0f, 0x24, 6, 0}},
	{AMOVL, Ytr7, Ynone, Yml, movRegMem2op, [4]uint8{0x0f, 0x24, 7, 0}},
	{AMOVL, Yml, Ynone, Ytr6, movMemReg2op, [4]uint8{0x0f, 0x26, 6, 0xff}},
	{AMOVL, Yml, Ynone, Ytr7, movMemReg2op, [4]uint8{0x0f, 0x26, 7, 0xff}},

	// lgdt, sgdt, lidt, sidt
	{AMOVL, Ym, Ynone, Ygdtr, movMemReg2op, [4]uint8{0x0f, 0x01, 2, 0}},
	{AMOVL, Ygdtr, Ynone, Ym, movRegMem2op, [4]uint8{0x0f, 0x01, 0, 0}},
	{AMOVL, Ym, Ynone, Yidtr, movMemReg2op, [4]uint8{0x0f, 0x01, 3, 0}},
	{AMOVL, Yidtr, Ynone, Ym, movRegMem2op, [4]uint8{0x0f, 0x01, 1, 0}},
	{AMOVQ, Ym, Ynone, Ygdtr, movMemReg2op, [4]uint8{0x0f, 0x01, 2, 0}},
	{AMOVQ, Ygdtr, Ynone, Ym, movRegMem2op, [4]uint8{0x0f, 0x01, 0, 0}},
	{AMOVQ, Ym, Ynone, Yidtr, movMemReg2op, [4]uint8{0x0f, 0x01, 3, 0}},
	{AMOVQ, Yidtr, Ynone, Ym, movRegMem2op, [4]uint8{0x0f, 0x01, 1, 0}},

	// lldt, sldt
	{AMOVW, Yml, Ynone, Yldtr, movMemReg2op, [4]uint8{0x0f, 0x00, 2, 0}},
	{AMOVW, Yldtr, Ynone, Yml, movRegMem2op, [4]uint8{0x0f, 0x00, 0, 0}},

	// lmsw, smsw
	{AMOVW, Yml, Ynone, Ymsw, movMemReg2op, [4]uint8{0x0f, 0x01, 6, 0}},
	{AMOVW, Ymsw, Ynone, Yml, movRegMem2op, [4]uint8{0x0f, 0x01, 4, 0}},

	// ltr, str
	{AMOVW, Yml, Ynone, Ytask, movMemReg2op, [4]uint8{0x0f, 0x00, 3, 0}},
	{AMOVW, Ytask, Ynone, Yml, movRegMem2op, [4]uint8{0x0f, 0x00, 1, 0}},

	/* load full pointer - unsupported
	{AMOVL, Yml, Ycol, movFullPtr, [4]uint8{0, 0, 0, 0}},
	{AMOVW, Yml, Ycol, movFullPtr, [4]uint8{Pe, 0, 0, 0}},
	*/

	// double shift
	{ASHLL, Yi8, Yrl, Yml, movDoubleShift, [4]uint8{0xa4, 0xa5, 0, 0}},
	{ASHLL, Ycl, Yrl, Yml, movDoubleShift, [4]uint8{0xa4, 0xa5, 0, 0}},
	{ASHLL, Ycx, Yrl, Yml, movDoubleShift, [4]uint8{0xa4, 0xa5, 0, 0}},
	{ASHRL, Yi8, Yrl, Yml, movDoubleShift, [4]uint8{0xac, 0xad, 0, 0}},
	{ASHRL, Ycl, Yrl, Yml, movDoubleShift, [4]uint8{0xac, 0xad, 0, 0}},
	{ASHRL, Ycx, Yrl, Yml, movDoubleShift, [4]uint8{0xac, 0xad, 0, 0}},
	{ASHLQ, Yi8, Yrl, Yml, movDoubleShift, [4]uint8{Pw, 0xa4, 0xa5, 0}},
	{ASHLQ, Ycl, Yrl, Yml, movDoubleShift, [4]uint8{Pw, 0xa4, 0xa5, 0}},
	{ASHLQ, Ycx, Yrl, Yml, movDoubleShift, [4]uint8{Pw, 0xa4, 0xa5, 0}},
	{ASHRQ, Yi8, Yrl, Yml, movDoubleShift, [4]uint8{Pw, 0xac, 0xad, 0}},
	{ASHRQ, Ycl, Yrl, Yml, movDoubleShift, [4]uint8{Pw, 0xac, 0xad, 0}},
	{ASHRQ, Ycx, Yrl, Yml, movDoubleShift, [4]uint8{Pw, 0xac, 0xad, 0}},
	{ASHLW, Yi8, Yrl, Yml, movDoubleShift, [4]uint8{Pe, 0xa4, 0xa5, 0}},
	{ASHLW, Ycl, Yrl, Yml, movDoubleShift, [4]uint8{Pe, 0xa4, 0xa5, 0}},
	{ASHLW, Ycx, Yrl, Yml, movDoubleShift, [4]uint8{Pe, 0xa4, 0xa5, 0}},
	{ASHRW, Yi8, Yrl, Yml, movDoubleShift, [4]uint8{Pe, 0xac, 0xad, 0}},
	{ASHRW, Ycl, Yrl, Yml, movDoubleShift, [4]uint8{Pe, 0xac, 0xad, 0}},
	{ASHRW, Ycx, Yrl, Yml, movDoubleShift, [4]uint8{Pe, 0xac, 0xad, 0}},

	// load TLS base
	{AMOVL, Ytls, Ynone, Yrl, movTLSReg, [4]uint8{0, 0, 0, 0}},
	{AMOVQ, Ytls, Ynone, Yrl, movTLSReg, [4]uint8{0, 0, 0, 0}},
	{0, 0, 0, 0, 0, [4]uint8{}},
}

func isax(a *obj.Addr) bool {
	switch a.Reg {
	case REG_AX, REG_AL, REG_AH:
		return true
	}

	return a.Index == REG_AX
}

func subreg(p *obj.Prog, from int, to int) {
	if false { /* debug['Q'] */
		fmt.Printf("\n%v\ts/%v/%v/\n", p, rconv(from), rconv(to))
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

func (ab *AsmBuf) mediaop(ctxt *obj.Link, o *Optab, op int, osize int, z int) int {
	switch op {
	case Pm, Pe, Pf2, Pf3:
		if osize != 1 {
			if op != Pm {
				ab.Put1(byte(op))
			}
			ab.Put1(Pm)
			z++
			op = int(o.op[z])
			break
		}
		fallthrough

	default:
		if ab.Len() == 0 || ab.Last() != Pm {
			ab.Put1(Pm)
		}
	}

	ab.Put1(byte(op))
	return z
}

var bpduff1 = []byte{
	0x48, 0x89, 0x6c, 0x24, 0xf0, // MOVQ BP, -16(SP)
	0x48, 0x8d, 0x6c, 0x24, 0xf0, // LEAQ -16(SP), BP
}

var bpduff2 = []byte{
	0x48, 0x8b, 0x6d, 0x00, // MOVQ 0(BP), BP
}

// asmevex emits EVEX pregis and opcode byte.
// In addition to asmvex r/m, vvvv and reg fields also requires optional
// K-masking register.
//
// Expects asmbuf.evex to be properly initialized.
func (ab *AsmBuf) asmevex(ctxt *obj.Link, p *obj.Prog, rm, v, r, k *obj.Addr) {
	ab.evexflag = true
	evex := ab.evex

	rexR := byte(1)
	evexR := byte(1)
	rexX := byte(1)
	rexB := byte(1)
	if r != nil {
		if regrex[r.Reg]&Rxr != 0 {
			rexR = 0 // "ModR/M.reg" selector 4th bit.
		}
		if regrex[r.Reg]&RxrEvex != 0 {
			evexR = 0 // "ModR/M.reg" selector 5th bit.
		}
	}
	if rm != nil {
		if rm.Index == REG_NONE && regrex[rm.Reg]&RxrEvex != 0 {
			rexX = 0
		} else if regrex[rm.Index]&Rxx != 0 {
			rexX = 0
		}
		if regrex[rm.Reg]&Rxb != 0 {
			rexB = 0
		}
	}
	// P0 = [R][X][B][R'][00][mm]
	p0 := (rexR << 7) |
		(rexX << 6) |
		(rexB << 5) |
		(evexR << 4) |
		(0 << 2) |
		(evex.M() << 0)

	vexV := byte(0)
	if v != nil {
		// 4bit-wide reg index.
		vexV = byte(reg[v.Reg]|(regrex[v.Reg]&Rxr)<<1) & 0xF
	}
	vexV ^= 0x0F
	// P1 = [W][vvvv][1][pp]
	p1 := (evex.W() << 7) |
		(vexV << 3) |
		(1 << 2) |
		(evex.P() << 0)

	suffix := evexSuffixMap[p.Scond]
	evexZ := byte(0)
	evexLL := evex.L()
	evexB := byte(0)
	evexV := byte(1)
	evexA := byte(0)
	if suffix.zeroing {
		if !evex.ZeroingEnabled() {
			ctxt.Diag("unsupported zeroing: %v", p)
		}
		if k == nil {
			// When you request zeroing you must specify a mask register.
			// See issue 57952.
			ctxt.Diag("mask register must be specified for .Z instructions: %v", p)
		} else if k.Reg == REG_K0 {
			// The mask register must not be K0. That restriction is already
			// handled by the Yknot0 restriction in the opcode tables, so we
			// won't ever reach here. But put something sensible here just in case.
			ctxt.Diag("mask register must not be K0 for .Z instructions: %v", p)
		}
		evexZ = 1
	}
	switch {
	case suffix.rounding != rcUnset:
		if rm != nil && rm.Type == obj.TYPE_MEM {
			ctxt.Diag("illegal rounding with memory argument: %v", p)
		} else if !evex.RoundingEnabled() {
			ctxt.Diag("unsupported rounding: %v", p)
		}
		evexB = 1
		evexLL = suffix.rounding
	case suffix.broadcast:
		if rm == nil || rm.Type != obj.TYPE_MEM {
			ctxt.Diag("illegal broadcast without memory argument: %v", p)
		} else if !evex.BroadcastEnabled() {
			ctxt.Diag("unsupported broadcast: %v", p)
		}
		evexB = 1
	case suffix.sae:
		if rm != nil && rm.Type == obj.TYPE_MEM {
			ctxt.Diag("illegal SAE with memory argument: %v", p)
		} else if !evex.SaeEnabled() {
			ctxt.Diag("unsupported SAE: %v", p)
		}
		evexB = 1
	}
	if rm != nil && regrex[rm.Index]&RxrEvex != 0 {
		evexV = 0
	} else if v != nil && regrex[v.Reg]&RxrEvex != 0 {
		evexV = 0 // VSR selector 5th bit.
	}
	if k != nil {
		evexA = byte(reg[k.Reg])
	}
	// P2 = [z][L'L][b][V'][aaa]
	p2 := (evexZ << 7) |
		(evexLL << 5) |
		(evexB << 4) |
		(evexV << 3) |
		(evexA << 0)

	const evexEscapeByte = 0x62
	ab.Put4(evexEscapeByte, p0, p1, p2)
	ab.Put1(evex.opcode)
}

// Emit VEX prefix and opcode byte.
// The three addresses are the r/m, vvvv, and reg fields.
// The reg and rm arguments appear in the same order as the
// arguments to asmand, which typically follows the call to asmvex.
// The final two arguments are the VEX prefix (see encoding above)
// and the opcode byte.
// For details about vex prefix see:
// https://en.wikipedia.org/wiki/VEX_prefix#Technical_description
func (ab *AsmBuf) asmvex(ctxt *obj.Link, rm, v, r *obj.Addr, vex, opcode uint8) {
	ab.vexflag = true
	rexR := 0
	if r != nil {
		rexR = regrex[r.Reg] & Rxr
	}
	rexB := 0
	rexX := 0
	if rm != nil {
		rexB = regrex[rm.Reg] & Rxb
		rexX = regrex[rm.Index] & Rxx
	}
	vexM := (vex >> 3) & 0x7
	vexWLP := vex & 0x87
	vexV := byte(0)
	if v != nil {
		vexV = byte(reg[v.Reg]|(regrex[v.Reg]&Rxr)<<1) & 0xF
	}
	vexV ^= 0xF
	if vexM == 1 && (rexX|rexB) == 0 && vex&vexW1 == 0 {
		// Can use 2-byte encoding.
		ab.Put2(0xc5, byte(rexR<<5)^0x80|vexV<<3|vexWLP)
	} else {
		// Must use 3-byte encoding.
		ab.Put3(0xc4,
			(byte(rexR|rexX|rexB)<<5)^0xE0|vexM,
			vexV<<3|vexWLP,
		)
	}
	ab.Put1(opcode)
}

// regIndex returns register index that fits in 5 bits.
//
//	R         : 3 bit | legacy instructions     | N/A
//	[R/V]EX.R : 1 bit | REX / VEX extension bit | Rxr
//	EVEX.R    : 1 bit | EVEX extension bit      | RxrEvex
//
// Examples:
//
//	REG_Z30 => 30
//	REG_X15 => 15
//	REG_R9  => 9
//	REG_AX  => 0
func regIndex(r int16) int {
	lower3bits := reg[r]
	high4bit := regrex[r] & Rxr << 1
	high5bit := regrex[r] & RxrEvex << 0
	return lower3bits | high4bit | high5bit
}

// avx2gatherValid reports whether p satisfies AVX2 gather constraints.
// Reports errors via ctxt.
func avx2gatherValid(ctxt *obj.Link, p *obj.Prog) bool {
	// If any pair of the index, mask, or destination registers
	// are the same, illegal instruction trap (#UD) is triggered.
	index := regIndex(p.GetFrom3().Index)
	mask := regIndex(p.From.Reg)
	dest := regIndex(p.To.Reg)
	if dest == mask || dest == index || mask == index {
		ctxt.Diag("mask, index, and destination registers should be distinct: %v", p)
		return false
	}

	return true
}

// avx512gatherValid reports whether p satisfies AVX512 gather constraints.
// Reports errors via ctxt.
func avx512gatherValid(ctxt *obj.Link, p *obj.Prog) bool {
	// Illegal instruction trap (#UD) is triggered if the destination vector
	// register is the same as index vector in VSIB.
	index := regIndex(p.From.Index)
	dest := regIndex(p.To.Reg)
	if dest == index {
		ctxt.Diag("index and destination registers should be distinct: %v", p)
		return false
	}

	return true
}

func (ab *AsmBuf) doasm(ctxt *obj.Link, cursym *obj.LSym, p *obj.Prog) {
	o := opindex[p.As&obj.AMask]

	if o == nil {
		ctxt.Diag("asmins: missing op %v", p)
		return
	}

	if pre := prefixof(ctxt, &p.From); pre != 0 {
		ab.Put1(byte(pre))
	}
	if pre := prefixof(ctxt, &p.To); pre != 0 {
		ab.Put1(byte(pre))
	}

	// Checks to warn about instruction/arguments combinations that
	// will unconditionally trigger illegal instruction trap (#UD).
	switch p.As {
	case AVGATHERDPD,
		AVGATHERQPD,
		AVGATHERDPS,
		AVGATHERQPS,
		AVPGATHERDD,
		AVPGATHERQD,
		AVPGATHERDQ,
		AVPGATHERQQ:
		if p.GetFrom3() == nil {
			// gathers need a 3rd arg. See issue 58822.
			ctxt.Diag("need a third arg for gather instruction: %v", p)
			return
		}
		// AVX512 gather requires explicit K mask.
		if p.GetFrom3().Reg >= REG_K0 && p.GetFrom3().Reg <= REG_K7 {
			if !avx512gatherValid(ctxt, p) {
				return
			}
		} else {
			if !avx2gatherValid(ctxt, p) {
				return
			}
		}
	}

	if p.Ft == 0 {
		p.Ft = uint8(oclass(ctxt, p, &p.From))
	}
	if p.Tt == 0 {
		p.Tt = uint8(oclass(ctxt, p, &p.To))
	}

	ft := int(p.Ft) * Ymax
	var f3t int
	tt := int(p.Tt) * Ymax

	xo := obj.Bool2int(o.op[0] == 0x0f)
	z := 0
	var a *obj.Addr
	var l int
	var op int
	var q *obj.Prog
	var r *obj.Reloc
	var rel obj.Reloc
	var v int64

	args := make([]int, 0, argListMax)
	if ft != Ynone*Ymax {
		args = append(args, ft)
	}
	for i := range p.RestArgs {
		args = append(args, oclass(ctxt, p, &p.RestArgs[i].Addr)*Ymax)
	}
	if tt != Ynone*Ymax {
		args = append(args, tt)
	}

	for _, yt := range o.ytab {
		// ytab matching is purely args-based,
		// but AVX512 suffixes like "Z" or "RU_SAE" will
		// add EVEX-only filter that will reject non-EVEX matches.
		//
		// Consider "VADDPD.BCST 2032(DX), X0, X0".
		// Without this rule, operands will lead to VEX-encoded form
		// and produce "c5b15813" encoding.
		if !yt.match(args) {
			// "xo" is always zero for VEX/EVEX encoded insts.
			z += int(yt.zoffset) + xo
		} else {
			if p.Scond != 0 && !evexZcase(yt.zcase) {
				// Do not signal error and continue to search
				// for matching EVEX-encoded form.
				z += int(yt.zoffset)
				continue
			}

			switch o.prefix {
			case Px1: // first option valid only in 32-bit mode
				if ctxt.Arch.Family == sys.AMD64 && z == 0 {
					z += int(yt.zoffset) + xo
					continue
				}
			case Pq: // 16 bit escape and opcode escape
				ab.Put2(Pe, Pm)

			case Pq3: // 16 bit escape and opcode escape + REX.W
				ab.rexflag |= Pw
				ab.Put2(Pe, Pm)

			case Pq4: // 66 0F 38
				ab.Put3(0x66, 0x0F, 0x38)

			case Pq4w: // 66 0F 38 + REX.W
				ab.rexflag |= Pw
				ab.Put3(0x66, 0x0F, 0x38)

			case Pq5: // F3 0F 38
				ab.Put3(0xF3, 0x0F, 0x38)

			case Pq5w: //  F3 0F 38 + REX.W
				ab.rexflag |= Pw
				ab.Put3(0xF3, 0x0F, 0x38)

			case Pf2, // xmm opcode escape
				Pf3:
				ab.Put2(o.prefix, Pm)

			case Pef3:
				ab.Put3(Pe, Pf3, Pm)

			case Pfw: // xmm opcode escape + REX.W
				ab.rexflag |= Pw
				ab.Put2(Pf3, Pm)

			case Pm: // opcode escape
				ab.Put1(Pm)

			case Pe: // 16 bit escape
				ab.Put1(Pe)

			case Pw: // 64-bit escape
				if ctxt.Arch.Family != sys.AMD64 {
					ctxt.Diag("asmins: illegal 64: %v", p)
				}
				ab.rexflag |= Pw

			case Pw8: // 64-bit escape if z >= 8
				if z >= 8 {
					if ctxt.Arch.Family != sys.AMD64 {
						ctxt.Diag("asmins: illegal 64: %v", p)
					}
					ab.rexflag |= Pw
				}

			case Pb: // botch
				if ctxt.Arch.Family != sys.AMD64 && (isbadbyte(&p.From) || isbadbyte(&p.To)) {
					goto bad
				}
				// NOTE(rsc): This is probably safe to do always,
				// but when enabled it chooses different encodings
				// than the old cmd/internal/obj/i386 code did,
				// which breaks our "same bits out" checks.
				// In particular, CMPB AX, $0 encodes as 80 f8 00
				// in the original obj/i386, and it would encode
				// (using a valid, shorter form) as 3c 00 if we enabled
				// the call to bytereg here.
				if ctxt.Arch.Family == sys.AMD64 {
					bytereg(&p.From, &p.Ft)
					bytereg(&p.To, &p.Tt)
				}

			case P32: // 32 bit but illegal if 64-bit mode
				if ctxt.Arch.Family == sys.AMD64 {
					ctxt.Diag("asmins: illegal in 64-bit mode: %v", p)
				}

			case Py: // 64-bit only, no prefix
				if ctxt.Arch.Family != sys.AMD64 {
					ctxt.Diag("asmins: illegal in %d-bit mode: %v", ctxt.Arch.RegSize*8, p)
				}

			case Py1: // 64-bit only if z < 1, no prefix
				if z < 1 && ctxt.Arch.Family != sys.AMD64 {
					ctxt.Diag("asmins: illegal in %d-bit mode: %v", ctxt.Arch.RegSize*8, p)
				}

			case Py3: // 64-bit only if z < 3, no prefix
				if z < 3 && ctxt.Arch.Family != sys.AMD64 {
					ctxt.Diag("asmins: illegal in %d-bit mode: %v", ctxt.Arch.RegSize*8, p)
				}
			}

			if z >= len(o.op) {
				log.Fatalf("asmins bad table %v", p)
			}
			op = int(o.op[z])
			if op == 0x0f {
				ab.Put1(byte(op))
				z++
				op = int(o.op[z])
			}

			switch yt.zcase {
			default:
				ctxt.Diag("asmins: unknown z %d %v", yt.zcase, p)
				return

			case Zpseudo:
				break

			case Zlit:
				ab.PutOpBytesLit(z, &o.op)

			case Zlitr_m:
				ab.PutOpBytesLit(z, &o.op)
				ab.asmand(ctxt, cursym, p, &p.To, &p.From)

			case Zlitm_r:
				ab.PutOpBytesLit(z, &o.op)
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Zlit_m_r:
				ab.PutOpBytesLit(z, &o.op)
				ab.asmand(ctxt, cursym, p, p.GetFrom3(), &p.To)

			case Zmb_r:
				bytereg(&p.From, &p.Ft)
				fallthrough

			case Zm_r:
				ab.Put1(byte(op))
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Z_m_r:
				ab.Put1(byte(op))
				ab.asmand(ctxt, cursym, p, p.GetFrom3(), &p.To)

			case Zm2_r:
				ab.Put2(byte(op), o.op[z+1])
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Zm_r_xm:
				ab.mediaop(ctxt, o, op, int(yt.zoffset), z)
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Zm_r_xm_nr:
				ab.rexflag = 0
				ab.mediaop(ctxt, o, op, int(yt.zoffset), z)
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Zm_r_i_xm:
				ab.mediaop(ctxt, o, op, int(yt.zoffset), z)
				ab.asmand(ctxt, cursym, p, &p.From, p.GetFrom3())
				ab.Put1(byte(p.To.Offset))

			case Zibm_r, Zibr_m:
				ab.PutOpBytesLit(z, &o.op)
				if yt.zcase == Zibr_m {
					ab.asmand(ctxt, cursym, p, &p.To, p.GetFrom3())
				} else {
					ab.asmand(ctxt, cursym, p, p.GetFrom3(), &p.To)
				}
				switch {
				default:
					ab.Put1(byte(p.From.Offset))
				case yt.args[0] == Yi32 && o.prefix == Pe:
					ab.PutInt16(int16(p.From.Offset))
				case yt.args[0] == Yi32:
					ab.PutInt32(int32(p.From.Offset))
				}

			case Zaut_r:
				ab.Put1(0x8d) // leal
				if p.From.Type != obj.TYPE_ADDR {
					ctxt.Diag("asmins: Zaut sb type ADDR")
				}
				p.From.Type = obj.TYPE_MEM
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)
				p.From.Type = obj.TYPE_ADDR

			case Zm_o:
				ab.Put1(byte(op))
				ab.asmando(ctxt, cursym, p, &p.From, int(o.op[z+1]))

			case Zr_m:
				ab.Put1(byte(op))
				ab.asmand(ctxt, cursym, p, &p.To, &p.From)

			case Zvex:
				ab.asmvex(ctxt, &p.From, p.GetFrom3(), &p.To, o.op[z], o.op[z+1])

			case Zvex_rm_v_r:
				ab.asmvex(ctxt, &p.From, p.GetFrom3(), &p.To, o.op[z], o.op[z+1])
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Zvex_rm_v_ro:
				ab.asmvex(ctxt, &p.From, p.GetFrom3(), &p.To, o.op[z], o.op[z+1])
				ab.asmando(ctxt, cursym, p, &p.From, int(o.op[z+2]))

			case Zvex_i_rm_vo:
				ab.asmvex(ctxt, p.GetFrom3(), &p.To, nil, o.op[z], o.op[z+1])
				ab.asmando(ctxt, cursym, p, p.GetFrom3(), int(o.op[z+2]))
				ab.Put1(byte(p.From.Offset))

			case Zvex_i_r_v:
				ab.asmvex(ctxt, p.GetFrom3(), &p.To, nil, o.op[z], o.op[z+1])
				regnum := byte(0x7)
				if p.GetFrom3().Reg >= REG_X0 && p.GetFrom3().Reg <= REG_X15 {
					regnum &= byte(p.GetFrom3().Reg - REG_X0)
				} else {
					regnum &= byte(p.GetFrom3().Reg - REG_Y0)
				}
				ab.Put1(o.op[z+2] | regnum)
				ab.Put1(byte(p.From.Offset))

			case Zvex_i_rm_v_r:
				imm, from, from3, to := unpackOps4(p)
				ab.asmvex(ctxt, from, from3, to, o.op[z], o.op[z+1])
				ab.asmand(ctxt, cursym, p, from, to)
				ab.Put1(byte(imm.Offset))

			case Zvex_i_rm_r:
				ab.asmvex(ctxt, p.GetFrom3(), nil, &p.To, o.op[z], o.op[z+1])
				ab.asmand(ctxt, cursym, p, p.GetFrom3(), &p.To)
				ab.Put1(byte(p.From.Offset))

			case Zvex_v_rm_r:
				ab.asmvex(ctxt, p.GetFrom3(), &p.From, &p.To, o.op[z], o.op[z+1])
				ab.asmand(ctxt, cursym, p, p.GetFrom3(), &p.To)

			case Zvex_r_v_rm:
				ab.asmvex(ctxt, &p.To, p.GetFrom3(), &p.From, o.op[z], o.op[z+1])
				ab.asmand(ctxt, cursym, p, &p.To, &p.From)

			case Zvex_rm_r_vo:
				ab.asmvex(ctxt, &p.From, &p.To, p.GetFrom3(), o.op[z], o.op[z+1])
				ab.asmando(ctxt, cursym, p, &p.From, int(o.op[z+2]))

			case Zvex_i_r_rm:
				ab.asmvex(ctxt, &p.To, nil, p.GetFrom3(), o.op[z], o.op[z+1])
				ab.asmand(ctxt, cursym, p, &p.To, p.GetFrom3())
				ab.Put1(byte(p.From.Offset))

			case Zvex_hr_rm_v_r:
				hr, from, from3, to := unpackOps4(p)
				ab.asmvex(ctxt, from, from3, to, o.op[z], o.op[z+1])
				ab.asmand(ctxt, cursym, p, from, to)
				ab.Put1(byte(regIndex(hr.Reg) << 4))

			case Zevex_k_rmo:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, &p.To, nil, nil, &p.From)
				ab.asmando(ctxt, cursym, p, &p.To, int(o.op[z+3]))

			case Zevex_i_rm_vo:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, p.GetFrom3(), &p.To, nil, nil)
				ab.asmando(ctxt, cursym, p, p.GetFrom3(), int(o.op[z+3]))
				ab.Put1(byte(p.From.Offset))

			case Zevex_i_rm_k_vo:
				imm, from, kmask, to := unpackOps4(p)
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, from, to, nil, kmask)
				ab.asmando(ctxt, cursym, p, from, int(o.op[z+3]))
				ab.Put1(byte(imm.Offset))

			case Zevex_i_r_rm:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, &p.To, nil, p.GetFrom3(), nil)
				ab.asmand(ctxt, cursym, p, &p.To, p.GetFrom3())
				ab.Put1(byte(p.From.Offset))

			case Zevex_i_r_k_rm:
				imm, from, kmask, to := unpackOps4(p)
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, to, nil, from, kmask)
				ab.asmand(ctxt, cursym, p, to, from)
				ab.Put1(byte(imm.Offset))

			case Zevex_i_rm_r:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, p.GetFrom3(), nil, &p.To, nil)
				ab.asmand(ctxt, cursym, p, p.GetFrom3(), &p.To)
				ab.Put1(byte(p.From.Offset))

			case Zevex_i_rm_k_r:
				imm, from, kmask, to := unpackOps4(p)
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, from, nil, to, kmask)
				ab.asmand(ctxt, cursym, p, from, to)
				ab.Put1(byte(imm.Offset))

			case Zevex_i_rm_v_r:
				imm, from, from3, to := unpackOps4(p)
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, from, from3, to, nil)
				ab.asmand(ctxt, cursym, p, from, to)
				ab.Put1(byte(imm.Offset))

			case Zevex_i_rm_v_k_r:
				imm, from, from3, kmask, to := unpackOps5(p)
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, from, from3, to, kmask)
				ab.asmand(ctxt, cursym, p, from, to)
				ab.Put1(byte(imm.Offset))

			case Zevex_r_v_rm:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, &p.To, p.GetFrom3(), &p.From, nil)
				ab.asmand(ctxt, cursym, p, &p.To, &p.From)

			case Zevex_rm_v_r:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, &p.From, p.GetFrom3(), &p.To, nil)
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Zevex_rm_k_r:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, &p.From, nil, &p.To, p.GetFrom3())
				ab.asmand(ctxt, cursym, p, &p.From, &p.To)

			case Zevex_r_k_rm:
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, &p.To, nil, &p.From, p.GetFrom3())
				ab.asmand(ctxt, cursym, p, &p.To, &p.From)

			case Zevex_rm_v_k_r:
				from, from3, kmask, to := unpackOps4(p)
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, from, from3, to, kmask)
				ab.asmand(ctxt, cursym, p, from, to)

			case Zevex_r_v_k_rm:
				from, from3, kmask, to := unpackOps4(p)
				ab.evex = newEVEXBits(z, &o.op)
				ab.asmevex(ctxt, p, to, from3, from, kmask)
				ab.asmand(ctxt, cursym, p, to, from)

			case Zr_m_xm:
				ab.mediaop(ctxt, o, op, int(yt.zoffset), z)
				ab.asmand(ctxt, cursym, p, &p.To, &p.From)

			case Zr_m_xm_nr:
				ab.rexflag = 0
				ab.mediaop(ctxt, o, op, int(yt.zoffset), z)
				ab.asmand(ctxt, cursym, p, &p.To, &p.From)

			case Zo_m:
				ab.Put1(byte(op))
				ab.asmando(ctxt, cursym, p, &p.To, int(o.op[z+1]))

			case Zcallindreg:
				r = obj.Addrel(cursym)
				r.Off = int32(p.Pc)
				r.Type = objabi.R_CALLIND
				r.Siz = 0
				fallthrough

			case Zo_m64:
				ab.Put1(byte(op))
				ab.asmandsz(ctxt, cursym, p, &p.To, int(o.op[z+1]), 0, 1)

			case Zm_ibo:
				ab.Put1(byte(op))
				ab.asmando(ctxt, cursym, p, &p.From, int(o.op[z+1]))
				ab.Put1(byte(vaddr(ctxt, p, &p.To, nil)))

			case Zibo_m:
				ab.Put1(byte(op))
				ab.asmando(ctxt, cursym, p, &p.To, int(o.op[z+1]))
				ab.Put1(byte(vaddr(ctxt, p, &p.From, nil)))

			case Zibo_m_xm:
				z = ab.mediaop(ctxt, o, op, int(yt.zoffset), z)
				ab.asmando(ctxt, cursym, p, &p.To, int(o.op[z+1]))
				ab.Put1(byte(vaddr(ctxt, p, &p.From, nil)))

			case Z_ib, Zib_:
				if yt.zcase == Zib_ {
					a = &p.From
				} else {
					a = &p.To
				}
				ab.Put1(byte(op))
				if p.As == AXABORT {
					ab.Put1(o.op[z+1])
				}
				ab.Put1(byte(vaddr(ctxt, p, a, nil)))

			case Zib_rp:
				ab.rexflag |= regrex[p.To.Reg] & (Rxb | 0x40)
				ab.Put2(byte(op+reg[p.To.Reg]), byte(vaddr(ctxt, p, &p.From, nil)))

			case Zil_rp:
				ab.rexflag |= regrex[p.To.Reg] & Rxb
				ab.Put1(byte(op + reg[p.To.Reg]))
				if o.prefix == Pe {
					v = vaddr(ctxt, p, &p.From, nil)
					ab.PutInt16(int16(v))
				} else {
					ab.relput4(ctxt, cursym, p, &p.From)
				}

			case Zo_iw:
				ab.Put1(byte(op))
				if p.From.Type != obj.TYPE_NONE {
					v = vaddr(ctxt, p, &p.From, nil)
					ab.PutInt16(int16(v))
				}

			case Ziq_rp:
				v = vaddr(ctxt, p, &p.From, &rel)
				l = int(v >> 32)
				if l == 0 && rel.Siz != 8 {
					ab.rexflag &^= (0x40 | Rxw)

					ab.rexflag |= regrex[p.To.Reg] & Rxb
					ab.Put1(byte(0xb8 + reg[p.To.Reg]))
					if rel.Type != 0 {
						r = obj.Addrel(cursym)
						*r = rel
						r.Off = int32(p.Pc + int64(ab.Len()))
					}

					ab.PutInt32(int32(v))
				} else if l == -1 && uint64(v)&(uint64(1)<<31) != 0 { // sign extend
					ab.Put1(0xc7)
					ab.asmando(ctxt, cursym, p, &p.To, 0)

					ab.PutInt32(int32(v)) // need all 8
				} else {
					ab.rexflag |= regrex[p.To.Reg] & Rxb
					ab.Put1(byte(op + reg[p.To.Reg]))
					if rel.Type != 0 {
						r = obj.Addrel(cursym)
						*r = rel
						r.Off = int32(p.Pc + int64(ab.Len()))
					}

					ab.PutInt64(v)
				}

			case Zib_rr:
				ab.Put1(byte(op))
				ab.asmand(ctxt, cursym, p, &p.To, &p.To)
				ab.Put1(byte(vaddr(ctxt, p, &p.From, nil)))

			case Z_il, Zil_:
				if yt.zcase == Zil_ {
					a = &p.From
				} else {
					a = &p.To
				}
				ab.Put1(byte(op))
				if o.prefix == Pe {
					v = vaddr(ctxt, p, a, nil)
					ab.PutInt16(int16(v))
				} else {
					ab.relput4(ctxt, cursym, p, a)
				}

			case Zm_ilo, Zilo_m:
				ab.Put1(byte(op))
				if yt.zcase == Zilo_m {
					a = &p.From
					ab.asmando(ctxt, cursym, p, &p.To, int(o.op[z+1]))
				} else {
					a = &p.To
					ab.asmando(ctxt, cursym, p, &p.From, int(o.op[z+1]))
				}

				if o.prefix == Pe {
					v = vaddr(ctxt, p, a, nil)
					ab.PutInt16(int16(v))
				} else {
					ab.relput4(ctxt, cursym, p, a)
				}

			case Zil_rr:
				ab.Put1(byte(op))
				ab.asmand(ctxt, cursym, p, &p.To, &p.To)
				if o.prefix == Pe {
					v = vaddr(ctxt, p, &p.From, nil)
					ab.PutInt16(int16(v))
				} else {
					ab.relput4(ctxt, cursym, p, &p.From)
				}

			case Z_rp:
				ab.rexflag |= regrex[p.To.Reg] & (Rxb | 0x40)
				ab.Put1(byte(op + reg[p.To.Reg]))

			case Zrp_:
				ab.rexflag |= regrex[p.From.Reg] & (Rxb | 0x40)
				ab.Put1(byte(op + reg[p.From.Reg]))

			case Zcallcon, Zjmpcon:
				if yt.zcase == Zcallcon {
					ab.Put1(byte(op))
				} else {
					ab.Put1(o.op[z+1])
				}
				r = obj.Addrel(cursym)
				r.Off = int32(p.Pc + int64(ab.Len()))
				r.Type = objabi.R_PCREL
				r.Siz = 4
				r.Add = p.To.Offset
				ab.PutInt32(0)

			case Zcallind:
				ab.Put2(byte(op), o.op[z+1])
				r = obj.Addrel(cursym)
				r.Off = int32(p.Pc + int64(ab.Len()))
				if ctxt.Arch.Family == sys.AMD64 {
					r.Type = objabi.R_PCREL
				} else {
					r.Type = objabi.R_ADDR
				}
				r.Siz = 4
				r.Add = p.To.Offset
				r.Sym = p.To.Sym
				ab.PutInt32(0)

			case Zcall, Zcallduff:
				if p.To.Sym == nil {
					ctxt.Diag("call without target")
					ctxt.DiagFlush()
					log.Fatalf("bad code")
				}

				if yt.zcase == Zcallduff && ctxt.Flag_dynlink {
					ctxt.Diag("directly calling duff when dynamically linking Go")
				}

				if yt.zcase == Zcallduff && ctxt.Arch.Family == sys.AMD64 {
					// Maintain BP around call, since duffcopy/duffzero can't do it
					// (the call jumps into the middle of the function).
					// This makes it possible to see call sites for duffcopy/duffzero in
					// BP-based profiling tools like Linux perf (which is the
					// whole point of maintaining frame pointers in Go).
					// MOVQ BP, -16(SP)
					// LEAQ -16(SP), BP
					ab.Put(bpduff1)
				}
				ab.Put1(byte(op))
				r = obj.Addrel(cursym)
				r.Off = int32(p.Pc + int64(ab.Len()))
				r.Sym = p.To.Sym
				r.Add = p.To.Offset
				r.Type = objabi.R_CALL
				r.Siz = 4
				ab.PutInt32(0)

				if yt.zcase == Zcallduff && ctxt.Arch.Family == sys.AMD64 {
					// Pop BP pushed above.
					// MOVQ 0(BP), BP
					ab.Put(bpduff2)
				}

			// TODO: jump across functions needs reloc
			case Zbr, Zjmp, Zloop:
				if p.As == AXBEGIN {
					ab.Put1(byte(op))
				}
				if p.To.Sym != nil {
					if yt.zcase != Zjmp {
						ctxt.Diag("branch to ATEXT")
						ctxt.DiagFlush()
						log.Fatalf("bad code")
					}

					ab.Put1(o.op[z+1])
					r = obj.Addrel(cursym)
					r.Off = int32(p.Pc + int64(ab.Len()))
					r.Sym = p.To.Sym
					// Note: R_CALL instead of R_PCREL. R_CALL is more permissive in that
					// it can point to a trampoline instead of the destination itself.
					r.Type = objabi.R_CALL
					r.Siz = 4
					ab.PutInt32(0)
					break
				}

				// Assumes q is in this function.
				// TODO: Check in input, preserve in brchain.

				// Fill in backward jump now.
				q = p.To.Target()

				if q == nil {
					ctxt.Diag("jmp/branch/loop without target")
					ctxt.DiagFlush()
					log.Fatalf("bad code")
				}

				if p.Back&branchBackwards != 0 {
					v = q.Pc - (p.Pc + 2)
					if v >= -128 && p.As != AXBEGIN {
						if p.As == AJCXZL {
							ab.Put1(0x67)
						}
						ab.Put2(byte(op), byte(v))
					} else if yt.zcase == Zloop {
						ctxt.Diag("loop too far: %v", p)
					} else {
						v -= 5 - 2
						if p.As == AXBEGIN {
							v--
						}
						if yt.zcase == Zbr {
							ab.Put1(0x0f)
							v--
						}

						ab.Put1(o.op[z+1])
						ab.PutInt32(int32(v))
					}

					break
				}

				// Annotate target; will fill in later.
				p.Forwd = q.Rel

				q.Rel = p
				if p.Back&branchShort != 0 && p.As != AXBEGIN {
					if p.As == AJCXZL {
						ab.Put1(0x67)
					}
					ab.Put2(byte(op), 0)
				} else if yt.zcase == Zloop {
					ctxt.Diag("loop too far: %v", p)
				} else {
					if yt.zcase == Zbr {
						ab.Put1(0x0f)
					}
					ab.Put1(o.op[z+1])
					ab.PutInt32(0)
				}

			case Zbyte:
				v = vaddr(ctxt, p, &p.From, &rel)
				if rel.Siz != 0 {
					rel.Siz = uint8(op)
					r = obj.Addrel(cursym)
					*r = rel
					r.Off = int32(p.Pc + int64(ab.Len()))
				}

				ab.Put1(byte(v))
				if op > 1 {
					ab.Put1(byte(v >> 8))
					if op > 2 {
						ab.PutInt16(int16(v >> 16))
						if op > 4 {
							ab.PutInt32(int32(v >> 32))
						}
					}
				}
			}

			return
		}
	}
	f3t = Ynone * Ymax
	if p.GetFrom3() != nil {
		f3t = oclass(ctxt, p, p.GetFrom3()) * Ymax
	}
	for mo := ymovtab; mo[0].as != 0; mo = mo[1:] {
		var pp obj.Prog
		var t []byte
		if p.As == mo[0].as {
			if ycover[ft+int(mo[0].ft)] != 0 && ycover[f3t+int(mo[0].f3t)] != 0 && ycover[tt+int(mo[0].tt)] != 0 {
				t = mo[0].op[:]
				switch mo[0].code {
				default:
					ctxt.Diag("asmins: unknown mov %d %v", mo[0].code, p)

				case movLit:
					for z = 0; t[z] != 0; z++ {
						ab.Put1(t[z])
					}

				case movRegMem:
					ab.Put1(t[0])
					ab.asmando(ctxt, cursym, p, &p.To, int(t[1]))

				case movMemReg:
					ab.Put1(t[0])
					ab.asmando(ctxt, cursym, p, &p.From, int(t[1]))

				case movRegMem2op: // r,m - 2op
					ab.Put2(t[0], t[1])
					ab.asmando(ctxt, cursym, p, &p.To, int(t[2]))
					ab.rexflag |= regrex[p.From.Reg] & (Rxr | 0x40)

				case movMemReg2op:
					ab.Put2(t[0], t[1])
					ab.asmando(ctxt, cursym, p, &p.From, int(t[2]))
					ab.rexflag |= regrex[p.To.Reg] & (Rxr | 0x40)

				case movFullPtr:
					if t[0] != 0 {
						ab.Put1(t[0])
					}
					switch p.To.Index {
					default:
						goto bad

					case REG_DS:
						ab.Put1(0xc5)

					case REG_SS:
						ab.Put2(0x0f, 0xb2)

					case REG_ES:
						ab.Put1(0xc4)

					case REG_FS:
						ab.Put2(0x0f, 0xb4)

					case REG_GS:
						ab.Put2(0x0f, 0xb5)
					}

					ab.asmand(ctxt, cursym, p, &p.From, &p.To)

				case movDoubleShift:
					if t[0] == Pw {
						if ctxt.Arch.Family != sys.AMD64 {
							ctxt.Diag("asmins: illegal 64: %v", p)
						}
						ab.rexflag |= Pw
						t = t[1:]
					} else if t[0] == Pe {
						ab.Put1(Pe)
						t = t[1:]
					}

					switch p.From.Type {
					default:
						goto bad

					case obj.TYPE_CONST:
						ab.Put2(0x0f, t[0])
						ab.asmandsz(ctxt, cursym, p, &p.To, reg[p.GetFrom3().Reg], regrex[p.GetFrom3().Reg], 0)
						ab.Put1(byte(p.From.Offset))

					case obj.TYPE_REG:
						switch p.From.Reg {
						default:
							goto bad

						case REG_CL, REG_CX:
							ab.Put2(0x0f, t[1])
							ab.asmandsz(ctxt, cursym, p, &p.To, reg[p.GetFrom3().Reg], regrex[p.GetFrom3().Reg], 0)
						}
					}

				// NOTE: The systems listed here are the ones that use the "TLS initial exec" model,
				// where you load the TLS base register into a register and then index off that
				// register to access the actual TLS variables. Systems that allow direct TLS access
				// are handled in prefixof above and should not be listed here.
				case movTLSReg:
					if ctxt.Arch.Family == sys.AMD64 && p.As != AMOVQ || ctxt.Arch.Family == sys.I386 && p.As != AMOVL {
						ctxt.Diag("invalid load of TLS: %v", p)
					}

					if ctxt.Arch.Family == sys.I386 {
						// NOTE: The systems listed here are the ones that use the "TLS initial exec" model,
						// where you load the TLS base register into a register and then index off that
						// register to access the actual TLS variables. Systems that allow direct TLS access
						// are handled in prefixof above and should not be listed here.
						switch ctxt.Headtype {
						default:
							log.Fatalf("unknown TLS base location for %v", ctxt.Headtype)

						case objabi.Hlinux, objabi.Hfreebsd:
							if ctxt.Flag_shared {
								// Note that this is not generating the same insns as the other cases.
								//     MOV TLS, dst
								// becomes
								//     call __x86.get_pc_thunk.dst
								//     movl (gotpc + g@gotntpoff)(dst), dst
								// which is encoded as
								//     call __x86.get_pc_thunk.dst
								//     movq 0(dst), dst
								// and R_CALL & R_TLS_IE relocs. This all assumes the only tls variable we access
								// is g, which we can't check here, but will when we assemble the second
								// instruction.
								dst := p.To.Reg
								ab.Put1(0xe8)
								r = obj.Addrel(cursym)
								r.Off = int32(p.Pc + int64(ab.Len()))
								r.Type = objabi.R_CALL
								r.Siz = 4
								r.Sym = ctxt.Lookup("__x86.get_pc_thunk." + strings.ToLower(rconv(int(dst))))
								ab.PutInt32(0)

								ab.Put2(0x8B, byte(2<<6|reg[dst]|(reg[dst]<<3)))
								r = obj.Addrel(cursym)
								r.Off = int32(p.Pc + int64(ab.Len()))
								r.Type = objabi.R_TLS_IE
								r.Siz = 4
								r.Add = 2
								ab.PutInt32(0)
							} else {
								// ELF TLS base is 0(GS).
								pp.From = p.From

								pp.From.Type = obj.TYPE_MEM
								pp.From.Reg = REG_GS
								pp.From.Offset = 0
								pp.From.Index = REG_NONE
								pp.From.Scale = 0
								ab.Put2(0x65, // GS
									0x8B)
								ab.asmand(ctxt, cursym, p, &pp.From, &p.To)
							}
						case objabi.Hplan9:
							pp.From = obj.Addr{}
							pp.From.Type = obj.TYPE_MEM
							pp.From.Name = obj.NAME_EXTERN
							pp.From.Sym = plan9privates
							pp.From.Offset = 0
							pp.From.Index = REG_NONE
							ab.Put1(0x8B)
							ab.asmand(ctxt, cursym, p, &pp.From, &p.To)
						}
						break
					}

					switch ctxt.Headtype {
					default:
						log.Fatalf("unknown TLS base location for %v", ctxt.Headtype)

					case objabi.Hlinux, objabi.Hfreebsd:
						if !ctxt.Flag_shared {
							log.Fatalf("unknown TLS base location for linux/freebsd without -shared")
						}
						// Note that this is not generating the same insn as the other cases.
						//     MOV TLS, R_to
						// becomes
						//     movq g@gottpoff(%rip), R_to
						// which is encoded as
						//     movq 0(%rip), R_to
						// and a R_TLS_IE reloc. This all assumes the only tls variable we access
						// is g, which we can't check here, but will when we assemble the second
						// instruction.
						ab.rexflag = Pw | (regrex[p.To.Reg] & Rxr)

						ab.Put2(0x8B, byte(0x05|(reg[p.To.Reg]<<3)))
						r = obj.Addrel(cursym)
						r.Off = int32(p.Pc + int64(ab.Len()))
						r.Type = objabi.R_TLS_IE
						r.Siz = 4
						r.Add = -4
						ab.PutInt32(0)

					case objabi.Hplan9:
						pp.From = obj.Addr{}
						pp.From.Type = obj.TYPE_MEM
						pp.From.Name = obj.NAME_EXTERN
						pp.From.Sym = plan9privates
						pp.From.Offset = 0
						pp.From.Index = REG_NONE
						ab.rexflag |= Pw
						ab.Put1(0x8B)
						ab.asmand(ctxt, cursym, p, &pp.From, &p.To)

					case objabi.Hsolaris: // TODO(rsc): Delete Hsolaris from list. Should not use this code. See progedit in obj6.c.
						// TLS base is 0(FS).
						pp.From = p.From

						pp.From.Type = obj.TYPE_MEM
						pp.From.Name = obj.NAME_NONE
						pp.From.Reg = REG_NONE
						pp.From.Offset = 0
						pp.From.Index = REG_NONE
						pp.From.Scale = 0
						ab.rexflag |= Pw
						ab.Put2(0x64, // FS
							0x8B)
						ab.asmand(ctxt, cursym, p, &pp.From, &p.To)
					}
				}
				return
			}
		}
	}
	goto bad

bad:
	if ctxt.Arch.Family != sys.AMD64 {
		// here, the assembly has failed.
		// if it's a byte instruction that has
		// unaddressable registers, try to
		// exchange registers and reissue the
		// instruction with the operands renamed.
		pp := *p

		unbytereg(&pp.From, &pp.Ft)
		unbytereg(&pp.To, &pp.Tt)

		z := int(p.From.Reg)
		if p.From.Type == obj.TYPE_REG && z >= REG_BP && z <= REG_DI {
			// TODO(rsc): Use this code for x86-64 too. It has bug fixes not present in the amd64 code base.
			// For now, different to keep bit-for-bit compatibility.
			if ctxt.Arch.Family == sys.I386 {
				breg := byteswapreg(ctxt, &p.To)
				if breg != REG_AX {
					ab.Put1(0x87) // xchg lhs,bx
					ab.asmando(ctxt, cursym, p, &p.From, reg[breg])
					subreg(&pp, z, breg)
					ab.doasm(ctxt, cursym, &pp)
					ab.Put1(0x87) // xchg lhs,bx
					ab.asmando(ctxt, cursym, p, &p.From, reg[breg])
				} else {
					ab.Put1(byte(0x90 + reg[z])) // xchg lsh,ax
					subreg(&pp, z, REG_AX)
					ab.doasm(ctxt, cursym, &pp)
					ab.Put1(byte(0x90 + reg[z])) // xchg lsh,ax
				}
				return
			}

			if isax(&p.To) || p.To.Type == obj.TYPE_NONE {
				// We certainly don't want to exchange
				// with AX if the op is MUL or DIV.
				ab.Put1(0x87) // xchg lhs,bx
				ab.asmando(ctxt, cursym, p, &p.From, reg[REG_BX])
				subreg(&pp, z, REG_BX)
				ab.doasm(ctxt, cursym, &pp)
				ab.Put1(0x87) // xchg lhs,bx
				ab.asmando(ctxt, cursym, p, &p.From, reg[REG_BX])
			} else {
				ab.Put1(byte(0x90 + reg[z])) // xchg lsh,ax
				subreg(&pp, z, REG_AX)
				ab.doasm(ctxt, cursym, &pp)
				ab.Put1(byte(0x90 + reg[z])) // xchg lsh,ax
			}
			return
		}

		z = int(p.To.Reg)
		if p.To.Type == obj.TYPE_REG && z >= REG_BP && z <= REG_DI {
			// TODO(rsc): Use this code for x86-64 too. It has bug fixes not present in the amd64 code base.
			// For now, different to keep bit-for-bit compatibility.
			if ctxt.Arch.Family == sys.I386 {
				breg := byteswapreg(ctxt, &p.From)
				if breg != REG_AX {
					ab.Put1(0x87) //xchg rhs,bx
					ab.asmando(ctxt, cursym, p, &p.To, reg[breg])
					subreg(&pp, z, breg)
					ab.doasm(ctxt, cursym, &pp)
					ab.Put1(0x87) // xchg rhs,bx
					ab.asmando(ctxt, cursym, p, &p.To, reg[breg])
				} else {
					ab.Put1(byte(0x90 + reg[z])) // xchg rsh,ax
					subreg(&pp, z, REG_AX)
					ab.doasm(ctxt, cursym, &pp)
					ab.Put1(byte(0x90 + reg[z])) // xchg rsh,ax
				}
				return
			}

			if isax(&p.From) {
				ab.Put1(0x87) // xchg rhs,bx
				ab.asmando(ctxt, cursym, p, &p.To, reg[REG_BX])
				subreg(&pp, z, REG_BX)
				ab.doasm(ctxt, cursym, &pp)
				ab.Put1(0x87) // xchg rhs,bx
				ab.asmando(ctxt, cursym, p, &p.To, reg[REG_BX])
			} else {
				ab.Put1(byte(0x90 + reg[z])) // xchg rsh,ax
				subreg(&pp, z, REG_AX)
				ab.doasm(ctxt, cursym, &pp)
				ab.Put1(byte(0x90 + reg[z])) // xchg rsh,ax
			}
			return
		}
	}

	ctxt.Diag("%s: invalid instruction: %v", cursym.Name, p)
}

// byteswapreg returns a byte-addressable register (AX, BX, CX, DX)
// which is not referenced in a.
// If a is empty, it returns BX to account for MULB-like instructions
// that might use DX and AX.
func byteswapreg(ctxt *obj.Link, a *obj.Addr) int {
	cana, canb, canc, cand := true, true, true, true
	if a.Type == obj.TYPE_NONE {
		cana, cand = false, false
	}

	if a.Type == obj.TYPE_REG || ((a.Type == obj.TYPE_MEM || a.Type == obj.TYPE_ADDR) && a.Name == obj.NAME_NONE) {
		switch a.Reg {
		case REG_NONE:
			cana, cand = false, false
		case REG_AX, REG_AL, REG_AH:
			cana = false
		case REG_BX, REG_BL, REG_BH:
			canb = false
		case REG_CX, REG_CL, REG_CH:
			canc = false
		case REG_DX, REG_DL, REG_DH:
			cand = false
		}
	}

	if a.Type == obj.TYPE_MEM || a.Type == obj.TYPE_ADDR {
		switch a.Index {
		case REG_AX:
			cana = false
		case REG_BX:
			canb = false
		case REG_CX:
			canc = false
		case REG_DX:
			cand = false
		}
	}

	switch {
	case cana:
		return REG_AX
	case canb:
		return REG_BX
	case canc:
		return REG_CX
	case cand:
		return REG_DX
	default:
		ctxt.Diag("impossible byte register")
		ctxt.DiagFlush()
		log.Fatalf("bad code")
		return 0
	}
}

func isbadbyte(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG && (REG_BP <= a.Reg && a.Reg <= REG_DI || REG_BPB <= a.Reg && a.Reg <= REG_DIB)
}

func (ab *AsmBuf) asmins(ctxt *obj.Link, cursym *obj.LSym, p *obj.Prog) {
	ab.Reset()

	ab.rexflag = 0
	ab.vexflag = false
	ab.evexflag = false
	mark := ab.Len()
	ab.doasm(ctxt, cursym, p)
	if ab.rexflag != 0 && !ab.vexflag && !ab.evexflag {
		// as befits the whole approach of the architecture,
		// the rex prefix must appear before the first opcode byte
		// (and thus after any 66/67/f2/f3/26/2e/3e prefix bytes, but
		// before the 0f opcode escape!), or it might be ignored.
		// note that the handbook often misleadingly shows 66/f2/f3 in `opcode'.
		if ctxt.Arch.Family != sys.AMD64 {
			ctxt.Diag("asmins: illegal in mode %d: %v (%d %d)", ctxt.Arch.RegSize*8, p, p.Ft, p.Tt)
		}
		n := ab.Len()
		var np int
		for np = mark; np < n; np++ {
			c := ab.At(np)
			if c != 0xf2 && c != 0xf3 && (c < 0x64 || c > 0x67) && c != 0x2e && c != 0x3e && c != 0x26 {
				break
			}
		}
		ab.Insert(np, byte(0x40|ab.rexflag))
	}

	n := ab.Len()
	for i := len(cursym.R) - 1; i >= 0; i-- {
		r := &cursym.R[i]
		if int64(r.Off) < p.Pc {
			break
		}
		if ab.rexflag != 0 && !ab.vexflag && !ab.evexflag {
			r.Off++
		}
		if r.Type == objabi.R_PCREL {
			if ctxt.Arch.Family == sys.AMD64 || p.As == obj.AJMP || p.As == obj.ACALL {
				// PC-relative addressing is relative to the end of the instruction,
				// but the relocations applied by the linker are relative to the end
				// of the relocation. Because immediate instruction
				// arguments can follow the PC-relative memory reference in the
				// instruction encoding, the two may not coincide. In this case,
				// adjust addend so that linker can keep relocating relative to the
				// end of the relocation.
				r.Add -= p.Pc + int64(n) - (int64(r.Off) + int64(r.Siz))
			} else if ctxt.Arch.Family == sys.I386 {
				// On 386 PC-relative addressing (for non-call/jmp instructions)
				// assumes that the previous instruction loaded the PC of the end
				// of that instruction into CX, so the adjustment is relative to
				// that.
				r.Add += int64(r.Off) - p.Pc + int64(r.Siz)
			}
		}
		if r.Type == objabi.R_GOTPCREL && ctxt.Arch.Family == sys.I386 {
			// On 386, R_GOTPCREL makes the same assumptions as R_PCREL.
			r.Add += int64(r.Off) - p.Pc + int64(r.Siz)
		}

	}
}

// unpackOps4 extracts 4 operands from p.
func unpackOps4(p *obj.Prog) (arg0, arg1, arg2, dst *obj.Addr) {
	return &p.From, &p.RestArgs[0].Addr, &p.RestArgs[1].Addr, &p.To
}

// unpackOps5 extracts 5 operands from p.
func unpackOps5(p *obj.Prog) (arg0, arg1, arg2, arg3, dst *obj.Addr) {
	return &p.From, &p.RestArgs[0].Addr, &p.RestArgs[1].Addr, &p.RestArgs[2].Addr, &p.To
}
