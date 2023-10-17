// Inferno utils/5c/5.out.h
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/5c/5.out.h
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

package arm

import "cmd/internal/obj"

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p arm

const (
	NSNAME = 8
	NSYM   = 50
	NREG   = 16
)

/* -1 disables use of REGARG */
const (
	REGARG = -1
)

const (
	REG_R0 = obj.RBaseARM + iota // must be 16-aligned
	REG_R1
	REG_R2
	REG_R3
	REG_R4
	REG_R5
	REG_R6
	REG_R7
	REG_R8
	REG_R9
	REG_R10
	REG_R11
	REG_R12
	REG_R13
	REG_R14
	REG_R15

	REG_F0 // must be 16-aligned
	REG_F1
	REG_F2
	REG_F3
	REG_F4
	REG_F5
	REG_F6
	REG_F7
	REG_F8
	REG_F9
	REG_F10
	REG_F11
	REG_F12
	REG_F13
	REG_F14
	REG_F15

	REG_FPSR // must be 2-aligned
	REG_FPCR

	REG_CPSR // must be 2-aligned
	REG_SPSR

	REGRET = REG_R0
	/* compiler allocates R1 up as temps */
	/* compiler allocates register variables R3 up */
	/* compiler allocates external registers R10 down */
	REGEXT = REG_R10
	/* these two registers are declared in runtime.h */
	REGG = REGEXT - 0
	REGM = REGEXT - 1

	REGCTXT = REG_R7
	REGTMP  = REG_R11
	REGSP   = REG_R13
	REGLINK = REG_R14
	REGPC   = REG_R15

	NFREG = 16
	/* compiler allocates register variables F0 up */
	/* compiler allocates external registers F7 down */
	FREGRET = REG_F0
	FREGEXT = REG_F7
	FREGTMP = REG_F15
)

// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0040b/IHI0040B_aadwarf.pdf
var ARMDWARFRegisters = map[int16]int16{}

func init() {
	// f assigns dwarfregisters[from:to] = (base):(step*(to-from)+base)
	f := func(from, to, base, step int16) {
		for r := int16(from); r <= to; r++ {
			ARMDWARFRegisters[r] = step*(r-from) + base
		}
	}
	f(REG_R0, REG_R15, 0, 1)
	f(REG_F0, REG_F15, 64, 2) // Use d0 through D15, aka S0, S2, ..., S30
}

// Special registers, after subtracting obj.RBaseARM, bit 9 indicates
// a special register and the low bits select the register.
const (
	REG_SPECIAL = obj.RBaseARM + 1<<9 + iota
	REG_MB_SY
	REG_MB_ST
	REG_MB_ISH
	REG_MB_ISHST
	REG_MB_NSH
	REG_MB_NSHST
	REG_MB_OSH
	REG_MB_OSHST

	MAXREG
)

const (
	C_NONE = iota
	C_REG
	C_REGREG
	C_REGREG2
	C_REGLIST
	C_SHIFT     /* register shift R>>x */
	C_SHIFTADDR /* memory address with shifted offset R>>x(R) */
	C_FREG
	C_PSR
	C_FCR
	C_SPR /* REG_MB_SY */

	C_RCON   /* 0xff rotated */
	C_NCON   /* ~RCON */
	C_RCON2A /* OR of two disjoint C_RCON constants */
	C_RCON2S /* subtraction of two disjoint C_RCON constants */
	C_SCON   /* 0xffff */
	C_LCON
	C_LCONADDR
	C_ZFCON
	C_SFCON
	C_LFCON

	C_RACON /* <=0xff rotated constant offset from auto */
	C_LACON /* Large Auto CONstant, i.e. large offset from SP */

	C_SBRA
	C_LBRA

	C_HAUTO  /* halfword insn offset (-0xff to 0xff) */
	C_FAUTO  /* float insn offset (0 to 0x3fc, word aligned) */
	C_HFAUTO /* both H and F */
	C_SAUTO  /* -0xfff to 0xfff */
	C_LAUTO

	C_HOREG
	C_FOREG
	C_HFOREG
	C_SOREG
	C_ROREG
	C_SROREG /* both nil and R */
	C_LOREG

	C_PC
	C_SP
	C_HREG

	C_ADDR /* reference to relocatable address */

	// TLS "var" in local exec mode: will become a constant offset from
	// thread local base that is ultimately chosen by the program linker.
	C_TLS_LE

	// TLS "var" in initial exec mode: will become a memory address (chosen
	// by the program linker) that the dynamic linker will fill with the
	// offset from the thread local base.
	C_TLS_IE

	C_TEXTSIZE

	C_GOK

	C_NCLASS /* must be the last */
)

const (
	AAND = obj.ABaseARM + obj.A_ARCHSPECIFIC + iota
	AEOR
	ASUB
	ARSB
	AADD
	AADC
	ASBC
	ARSC
	ATST
	ATEQ
	ACMP
	ACMN
	AORR
	ABIC

	AMVN

	/*
	 * Do not reorder or fragment the conditional branch
	 * opcodes, or the predication code will break
	 */
	ABEQ
	ABNE
	ABCS
	ABHS
	ABCC
	ABLO
	ABMI
	ABPL
	ABVS
	ABVC
	ABHI
	ABLS
	ABGE
	ABLT
	ABGT
	ABLE

	AMOVWD
	AMOVWF
	AMOVDW
	AMOVFW
	AMOVFD
	AMOVDF
	AMOVF
	AMOVD

	ACMPF
	ACMPD
	AADDF
	AADDD
	ASUBF
	ASUBD
	AMULF
	AMULD
	ANMULF
	ANMULD
	AMULAF
	AMULAD
	ANMULAF
	ANMULAD
	AMULSF
	AMULSD
	ANMULSF
	ANMULSD
	AFMULAF
	AFMULAD
	AFNMULAF
	AFNMULAD
	AFMULSF
	AFMULSD
	AFNMULSF
	AFNMULSD
	ADIVF
	ADIVD
	ASQRTF
	ASQRTD
	AABSF
	AABSD
	ANEGF
	ANEGD

	ASRL
	ASRA
	ASLL
	AMULU
	ADIVU
	AMUL
	AMMUL
	ADIV
	AMOD
	AMODU
	ADIVHW
	ADIVUHW

	AMOVB
	AMOVBS
	AMOVBU
	AMOVH
	AMOVHS
	AMOVHU
	AMOVW
	AMOVM
	ASWPBU
	ASWPW

	ARFE
	ASWI
	AMULA
	AMULS
	AMMULA
	AMMULS

	AWORD

	AMULL
	AMULAL
	AMULLU
	AMULALU

	ABX
	ABXRET
	ADWORD

	ALDREX
	ASTREX
	ALDREXD
	ASTREXD

	ADMB

	APLD

	ACLZ
	AREV
	AREV16
	AREVSH
	ARBIT

	AXTAB
	AXTAH
	AXTABU
	AXTAHU

	ABFX
	ABFXU
	ABFC
	ABFI

	AMULWT
	AMULWB
	AMULBB
	AMULAWT
	AMULAWB
	AMULABB

	AMRC // MRC/MCR

	ALAST

	// aliases
	AB  = obj.AJMP
	ABL = obj.ACALL
)

/* scond byte */
const (
	C_SCOND = (1 << 4) - 1
	C_SBIT  = 1 << 4
	C_PBIT  = 1 << 5
	C_WBIT  = 1 << 6
	C_FBIT  = 1 << 7 /* psr flags-only */
	C_UBIT  = 1 << 7 /* up bit, unsigned bit */

	// These constants are the ARM condition codes encodings,
	// XORed with 14 so that C_SCOND_NONE has value 0,
	// so that a zeroed Prog.scond means "always execute".
	C_SCOND_XOR = 14

	C_SCOND_EQ   = 0 ^ C_SCOND_XOR
	C_SCOND_NE   = 1 ^ C_SCOND_XOR
	C_SCOND_HS   = 2 ^ C_SCOND_XOR
	C_SCOND_LO   = 3 ^ C_SCOND_XOR
	C_SCOND_MI   = 4 ^ C_SCOND_XOR
	C_SCOND_PL   = 5 ^ C_SCOND_XOR
	C_SCOND_VS   = 6 ^ C_SCOND_XOR
	C_SCOND_VC   = 7 ^ C_SCOND_XOR
	C_SCOND_HI   = 8 ^ C_SCOND_XOR
	C_SCOND_LS   = 9 ^ C_SCOND_XOR
	C_SCOND_GE   = 10 ^ C_SCOND_XOR
	C_SCOND_LT   = 11 ^ C_SCOND_XOR
	C_SCOND_GT   = 12 ^ C_SCOND_XOR
	C_SCOND_LE   = 13 ^ C_SCOND_XOR
	C_SCOND_NONE = 14 ^ C_SCOND_XOR
	C_SCOND_NV   = 15 ^ C_SCOND_XOR

	/* D_SHIFT type */
	SHIFT_LL = 0 << 5
	SHIFT_LR = 1 << 5
	SHIFT_AR = 2 << 5
	SHIFT_RR = 3 << 5
)
