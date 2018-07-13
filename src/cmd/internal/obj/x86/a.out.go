// Inferno utils/6c/6.out.h
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6c/6.out.h
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

import "cmd/internal/obj"

const (
	REG_NONE = 0
)

const (
	REG_AL = obj.RBaseAMD64 + iota
	REG_CL
	REG_DL
	REG_BL
	REG_SPB
	REG_BPB
	REG_SIB
	REG_DIB
	REG_R8B
	REG_R9B
	REG_R10B
	REG_R11B
	REG_R12B
	REG_R13B
	REG_R14B
	REG_R15B

	REG_AX
	REG_CX
	REG_DX
	REG_BX
	REG_SP
	REG_BP
	REG_SI
	REG_DI
	REG_R8
	REG_R9
	REG_R10
	REG_R11
	REG_R12
	REG_R13
	REG_R14
	REG_R15

	REG_AH
	REG_CH
	REG_DH
	REG_BH

	REG_F0
	REG_F1
	REG_F2
	REG_F3
	REG_F4
	REG_F5
	REG_F6
	REG_F7

	REG_M0
	REG_M1
	REG_M2
	REG_M3
	REG_M4
	REG_M5
	REG_M6
	REG_M7

	REG_K0
	REG_K1
	REG_K2
	REG_K3
	REG_K4
	REG_K5
	REG_K6
	REG_K7

	REG_X0
	REG_X1
	REG_X2
	REG_X3
	REG_X4
	REG_X5
	REG_X6
	REG_X7
	REG_X8
	REG_X9
	REG_X10
	REG_X11
	REG_X12
	REG_X13
	REG_X14
	REG_X15
	REG_X16
	REG_X17
	REG_X18
	REG_X19
	REG_X20
	REG_X21
	REG_X22
	REG_X23
	REG_X24
	REG_X25
	REG_X26
	REG_X27
	REG_X28
	REG_X29
	REG_X30
	REG_X31

	REG_Y0
	REG_Y1
	REG_Y2
	REG_Y3
	REG_Y4
	REG_Y5
	REG_Y6
	REG_Y7
	REG_Y8
	REG_Y9
	REG_Y10
	REG_Y11
	REG_Y12
	REG_Y13
	REG_Y14
	REG_Y15
	REG_Y16
	REG_Y17
	REG_Y18
	REG_Y19
	REG_Y20
	REG_Y21
	REG_Y22
	REG_Y23
	REG_Y24
	REG_Y25
	REG_Y26
	REG_Y27
	REG_Y28
	REG_Y29
	REG_Y30
	REG_Y31

	REG_Z0
	REG_Z1
	REG_Z2
	REG_Z3
	REG_Z4
	REG_Z5
	REG_Z6
	REG_Z7
	REG_Z8
	REG_Z9
	REG_Z10
	REG_Z11
	REG_Z12
	REG_Z13
	REG_Z14
	REG_Z15
	REG_Z16
	REG_Z17
	REG_Z18
	REG_Z19
	REG_Z20
	REG_Z21
	REG_Z22
	REG_Z23
	REG_Z24
	REG_Z25
	REG_Z26
	REG_Z27
	REG_Z28
	REG_Z29
	REG_Z30
	REG_Z31

	REG_CS
	REG_SS
	REG_DS
	REG_ES
	REG_FS
	REG_GS

	REG_GDTR // global descriptor table register
	REG_IDTR // interrupt descriptor table register
	REG_LDTR // local descriptor table register
	REG_MSW  // machine status word
	REG_TASK // task register

	REG_CR0
	REG_CR1
	REG_CR2
	REG_CR3
	REG_CR4
	REG_CR5
	REG_CR6
	REG_CR7
	REG_CR8
	REG_CR9
	REG_CR10
	REG_CR11
	REG_CR12
	REG_CR13
	REG_CR14
	REG_CR15

	REG_DR0
	REG_DR1
	REG_DR2
	REG_DR3
	REG_DR4
	REG_DR5
	REG_DR6
	REG_DR7

	REG_TR0
	REG_TR1
	REG_TR2
	REG_TR3
	REG_TR4
	REG_TR5
	REG_TR6
	REG_TR7

	REG_TLS

	MAXREG

	REG_CR = REG_CR0
	REG_DR = REG_DR0
	REG_TR = REG_TR0

	REGARG   = -1
	REGRET   = REG_AX
	FREGRET  = REG_X0
	REGSP    = REG_SP
	REGCTXT  = REG_DX
	REGEXT   = REG_R15     // compiler allocates external registers R15 down
	FREGMIN  = REG_X0 + 5  // first register variable
	FREGEXT  = REG_X0 + 15 // first external register
	T_TYPE   = 1 << 0
	T_INDEX  = 1 << 1
	T_OFFSET = 1 << 2
	T_FCONST = 1 << 3
	T_SYM    = 1 << 4
	T_SCONST = 1 << 5
	T_64     = 1 << 6
	T_GOTYPE = 1 << 7
)

// https://www.uclibc.org/docs/psABI-x86_64.pdf, figure 3.36
var AMD64DWARFRegisters = map[int16]int16{
	REG_AX:  0,
	REG_DX:  1,
	REG_CX:  2,
	REG_BX:  3,
	REG_SI:  4,
	REG_DI:  5,
	REG_BP:  6,
	REG_SP:  7,
	REG_R8:  8,
	REG_R9:  9,
	REG_R10: 10,
	REG_R11: 11,
	REG_R12: 12,
	REG_R13: 13,
	REG_R14: 14,
	REG_R15: 15,
	// 16 is "Return Address RA", whatever that is.
	// 17-24 vector registers (X/Y/Z).
	REG_X0: 17,
	REG_X1: 18,
	REG_X2: 19,
	REG_X3: 20,
	REG_X4: 21,
	REG_X5: 22,
	REG_X6: 23,
	REG_X7: 24,
	// 25-32 extended vector registers (X/Y/Z).
	REG_X8:  25,
	REG_X9:  26,
	REG_X10: 27,
	REG_X11: 28,
	REG_X12: 29,
	REG_X13: 30,
	REG_X14: 31,
	REG_X15: 32,
	// ST registers. %stN => FN.
	REG_F0: 33,
	REG_F1: 34,
	REG_F2: 35,
	REG_F3: 36,
	REG_F4: 37,
	REG_F5: 38,
	REG_F6: 39,
	REG_F7: 40,
	// MMX registers. %mmN => MN.
	REG_M0: 41,
	REG_M1: 42,
	REG_M2: 43,
	REG_M3: 44,
	REG_M4: 45,
	REG_M5: 46,
	REG_M6: 47,
	REG_M7: 48,
	// 48 is flags, which doesn't have a name.
	REG_ES: 50,
	REG_CS: 51,
	REG_SS: 52,
	REG_DS: 53,
	REG_FS: 54,
	REG_GS: 55,
	// 58 and 59 are {fs,gs}base, which don't have names.
	REG_TR:   62,
	REG_LDTR: 63,
	// 64-66 are mxcsr, fcw, fsw, which don't have names.

	// 67-82 upper vector registers (X/Y/Z).
	REG_X16: 67,
	REG_X17: 68,
	REG_X18: 69,
	REG_X19: 70,
	REG_X20: 71,
	REG_X21: 72,
	REG_X22: 73,
	REG_X23: 74,
	REG_X24: 75,
	REG_X25: 76,
	REG_X26: 77,
	REG_X27: 78,
	REG_X28: 79,
	REG_X29: 80,
	REG_X30: 81,
	REG_X31: 82,

	// 118-125 vector mask registers. %kN => KN.
	REG_K0: 118,
	REG_K1: 119,
	REG_K2: 120,
	REG_K3: 121,
	REG_K4: 122,
	REG_K5: 123,
	REG_K6: 124,
	REG_K7: 125,
}

// https://www.uclibc.org/docs/psABI-i386.pdf, table 2.14
var X86DWARFRegisters = map[int16]int16{
	REG_AX: 0,
	REG_CX: 1,
	REG_DX: 2,
	REG_BX: 3,
	REG_SP: 4,
	REG_BP: 5,
	REG_SI: 6,
	REG_DI: 7,
	// 8 is "Return Address RA", whatever that is.
	// 9 is flags, which doesn't have a name.
	// ST registers. %stN => FN.
	REG_F0: 11,
	REG_F1: 12,
	REG_F2: 13,
	REG_F3: 14,
	REG_F4: 15,
	REG_F5: 16,
	REG_F6: 17,
	REG_F7: 18,
	// XMM registers. %xmmN => XN.
	REG_X0: 21,
	REG_X1: 22,
	REG_X2: 23,
	REG_X3: 24,
	REG_X4: 25,
	REG_X5: 26,
	REG_X6: 27,
	REG_X7: 28,
	// MMX registers. %mmN => MN.
	REG_M0: 29,
	REG_M1: 30,
	REG_M2: 31,
	REG_M3: 32,
	REG_M4: 33,
	REG_M5: 34,
	REG_M6: 35,
	REG_M7: 36,
	// 39 is mxcsr, which doesn't have a name.
	REG_ES:   40,
	REG_CS:   41,
	REG_SS:   42,
	REG_DS:   43,
	REG_FS:   44,
	REG_GS:   45,
	REG_TR:   48,
	REG_LDTR: 49,
}
