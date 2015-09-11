// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// copied from ../../amd64/reg.go
var regNamesAMD64 = []string{
	".AX",
	".CX",
	".DX",
	".BX",
	".SP",
	".BP",
	".SI",
	".DI",
	".R8",
	".R9",
	".R10",
	".R11",
	".R12",
	".R13",
	".R14",
	".R15",
	".X0",
	".X1",
	".X2",
	".X3",
	".X4",
	".X5",
	".X6",
	".X7",
	".X8",
	".X9",
	".X10",
	".X11",
	".X12",
	".X13",
	".X14",
	".X15",

	// pseudo-registers
	".SB",
	".FLAGS",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesAMD64) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesAMD64 {
		if name[0] != '.' {
			panic("register name " + name + " does not start with '.'")
		}
		num[name[1:]] = i
	}
	buildReg := func(s string) regMask {
		m := regMask(0)
		for _, r := range strings.Split(s, " ") {
			if n, ok := num[r]; ok {
				m |= regMask(1) << uint(n)
				continue
			}
			panic("register " + r + " not found")
		}
		return m
	}

	// Common individual register masks
	var (
		ax         = buildReg("AX")
		cx         = buildReg("CX")
		dx         = buildReg("DX")
		x15        = buildReg("X15")
		gp         = buildReg("AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15")
		fp         = buildReg("X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15")
		gpsp       = gp | buildReg("SP")
		gpspsb     = gpsp | buildReg("SB")
		flags      = buildReg("FLAGS")
		callerSave = gp | fp | flags
	)
	// Common slices of register masks
	var (
		gponly    = []regMask{gp}
		fponly    = []regMask{fp}
		flagsonly = []regMask{flags}
	)

	// Common regInfo
	var (
		gp01      = regInfo{inputs: []regMask{}, outputs: gponly}
		gp01flags = regInfo{inputs: []regMask{}, outputs: gponly, clobbers: flags}
		gp11      = regInfo{inputs: []regMask{gpsp}, outputs: gponly, clobbers: flags}
		gp11nf    = regInfo{inputs: []regMask{gpsp}, outputs: gponly} // nf: no flags clobbered
		gp11sb    = regInfo{inputs: []regMask{gpspsb}, outputs: gponly}
		gp21      = regInfo{inputs: []regMask{gpsp, gpsp}, outputs: gponly, clobbers: flags}
		gp21sb    = regInfo{inputs: []regMask{gpspsb, gpsp}, outputs: gponly}
		gp21shift = regInfo{inputs: []regMask{gpsp, cx}, outputs: []regMask{gp &^ cx}, clobbers: flags}
		gp11div   = regInfo{inputs: []regMask{ax, gpsp &^ dx}, outputs: []regMask{ax},
			clobbers: dx | flags}
		gp11hmul = regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{dx},
			clobbers: ax | flags}
		gp11mod = regInfo{inputs: []regMask{ax, gpsp &^ dx}, outputs: []regMask{dx},
			clobbers: ax | flags}
		gp10 = regInfo{inputs: []regMask{gp}}

		gp2flags  = regInfo{inputs: []regMask{gpsp, gpsp}, outputs: flagsonly}
		gp1flags  = regInfo{inputs: []regMask{gpsp}, outputs: flagsonly}
		flagsgp   = regInfo{inputs: flagsonly, outputs: gponly}
		readflags = regInfo{inputs: flagsonly, outputs: gponly}
		flagsgpax = regInfo{inputs: flagsonly, clobbers: ax | flags, outputs: []regMask{gp &^ ax}}

		gpload    = regInfo{inputs: []regMask{gpspsb, 0}, outputs: gponly}
		gploadidx = regInfo{inputs: []regMask{gpspsb, gpsp, 0}, outputs: gponly}

		gpstore    = regInfo{inputs: []regMask{gpspsb, gpsp, 0}}
		gpstoreidx = regInfo{inputs: []regMask{gpspsb, gpsp, gpsp, 0}}

		fp01    = regInfo{inputs: []regMask{}, outputs: fponly}
		fp21    = regInfo{inputs: []regMask{fp, fp}, outputs: fponly}
		fp21x15 = regInfo{inputs: []regMask{fp &^ x15, fp &^ x15},
			clobbers: x15, outputs: []regMask{fp &^ x15}}
		fpgp     = regInfo{inputs: fponly, outputs: gponly}
		gpfp     = regInfo{inputs: gponly, outputs: fponly}
		fp11     = regInfo{inputs: fponly, outputs: fponly}
		fp2flags = regInfo{inputs: []regMask{fp, fp}, outputs: flagsonly}
		// fp1flags = regInfo{inputs: fponly, outputs: flagsonly}

		fpload    = regInfo{inputs: []regMask{gpspsb, 0}, outputs: fponly}
		fploadidx = regInfo{inputs: []regMask{gpspsb, gpsp, 0}, outputs: fponly}

		fpstore    = regInfo{inputs: []regMask{gpspsb, fp, 0}}
		fpstoreidx = regInfo{inputs: []regMask{gpspsb, gpsp, fp, 0}}
	)
	// TODO: most ops clobber flags

	// Suffixes encode the bit width of various instructions.
	// Q = 64 bit, L = 32 bit, W = 16 bit, B = 8 bit

	// TODO: 2-address instructions.  Mark ops as needing matching input/output regs.
	var AMD64ops = []opData{
		// fp ops
		{name: "ADDSS", reg: fp21, asm: "ADDSS"},    // fp32 add
		{name: "ADDSD", reg: fp21, asm: "ADDSD"},    // fp64 add
		{name: "SUBSS", reg: fp21x15, asm: "SUBSS"}, // fp32 sub
		{name: "SUBSD", reg: fp21x15, asm: "SUBSD"}, // fp64 sub
		{name: "MULSS", reg: fp21, asm: "MULSS"},    // fp32 mul
		{name: "MULSD", reg: fp21, asm: "MULSD"},    // fp64 mul
		{name: "DIVSS", reg: fp21x15, asm: "DIVSS"}, // fp32 div
		{name: "DIVSD", reg: fp21x15, asm: "DIVSD"}, // fp64 div

		{name: "MOVSSload", reg: fpload, asm: "MOVSS"},        // fp32 load
		{name: "MOVSDload", reg: fpload, asm: "MOVSD"},        // fp64 load
		{name: "MOVSSconst", reg: fp01, asm: "MOVSS"},         // fp32 constant
		{name: "MOVSDconst", reg: fp01, asm: "MOVSD"},         // fp64 constant
		{name: "MOVSSloadidx4", reg: fploadidx, asm: "MOVSS"}, // fp32 load
		{name: "MOVSDloadidx8", reg: fploadidx, asm: "MOVSD"}, // fp64 load

		{name: "MOVSSstore", reg: fpstore, asm: "MOVSS"},        // fp32 store
		{name: "MOVSDstore", reg: fpstore, asm: "MOVSD"},        // fp64 store
		{name: "MOVSSstoreidx4", reg: fpstoreidx, asm: "MOVSS"}, // fp32 indexed by 4i store
		{name: "MOVSDstoreidx8", reg: fpstoreidx, asm: "MOVSD"}, // fp64 indexed by 8i store

		// binary ops
		{name: "ADDQ", reg: gp21, asm: "ADDQ"},                     // arg0 + arg1
		{name: "ADDL", reg: gp21, asm: "ADDL"},                     // arg0 + arg1
		{name: "ADDW", reg: gp21, asm: "ADDW"},                     // arg0 + arg1
		{name: "ADDB", reg: gp21, asm: "ADDB"},                     // arg0 + arg1
		{name: "ADDQconst", reg: gp11, asm: "ADDQ", typ: "UInt64"}, // arg0 + auxint
		{name: "ADDLconst", reg: gp11, asm: "ADDL"},                // arg0 + auxint
		{name: "ADDWconst", reg: gp11, asm: "ADDW"},                // arg0 + auxint
		{name: "ADDBconst", reg: gp11, asm: "ADDB"},                // arg0 + auxint

		{name: "SUBQ", reg: gp21, asm: "SUBQ"},      // arg0 - arg1
		{name: "SUBL", reg: gp21, asm: "SUBL"},      // arg0 - arg1
		{name: "SUBW", reg: gp21, asm: "SUBW"},      // arg0 - arg1
		{name: "SUBB", reg: gp21, asm: "SUBB"},      // arg0 - arg1
		{name: "SUBQconst", reg: gp11, asm: "SUBQ"}, // arg0 - auxint
		{name: "SUBLconst", reg: gp11, asm: "SUBL"}, // arg0 - auxint
		{name: "SUBWconst", reg: gp11, asm: "SUBW"}, // arg0 - auxint
		{name: "SUBBconst", reg: gp11, asm: "SUBB"}, // arg0 - auxint

		{name: "MULQ", reg: gp21, asm: "IMULQ"},      // arg0 * arg1
		{name: "MULL", reg: gp21, asm: "IMULL"},      // arg0 * arg1
		{name: "MULW", reg: gp21, asm: "IMULW"},      // arg0 * arg1
		{name: "MULB", reg: gp21, asm: "IMULW"},      // arg0 * arg1
		{name: "MULQconst", reg: gp11, asm: "IMULQ"}, // arg0 * auxint
		{name: "MULLconst", reg: gp11, asm: "IMULL"}, // arg0 * auxint
		{name: "MULWconst", reg: gp11, asm: "IMULW"}, // arg0 * auxint
		{name: "MULBconst", reg: gp11, asm: "IMULW"}, // arg0 * auxint

		{name: "HMULL", reg: gp11hmul, asm: "IMULL"}, // (arg0 * arg1) >> width
		{name: "HMULW", reg: gp11hmul, asm: "IMULW"}, // (arg0 * arg1) >> width
		{name: "HMULB", reg: gp11hmul, asm: "IMULB"}, // (arg0 * arg1) >> width
		{name: "HMULLU", reg: gp11hmul, asm: "MULL"}, // (arg0 * arg1) >> width
		{name: "HMULWU", reg: gp11hmul, asm: "MULW"}, // (arg0 * arg1) >> width
		{name: "HMULBU", reg: gp11hmul, asm: "MULB"}, // (arg0 * arg1) >> width

		{name: "DIVQ", reg: gp11div, asm: "IDIVQ"}, // arg0 / arg1
		{name: "DIVL", reg: gp11div, asm: "IDIVL"}, // arg0 / arg1
		{name: "DIVW", reg: gp11div, asm: "IDIVW"}, // arg0 / arg1
		{name: "DIVQU", reg: gp11div, asm: "DIVQ"}, // arg0 / arg1
		{name: "DIVLU", reg: gp11div, asm: "DIVL"}, // arg0 / arg1
		{name: "DIVWU", reg: gp11div, asm: "DIVW"}, // arg0 / arg1

		{name: "MODQ", reg: gp11mod, asm: "IDIVQ"}, // arg0 % arg1
		{name: "MODL", reg: gp11mod, asm: "IDIVL"}, // arg0 % arg1
		{name: "MODW", reg: gp11mod, asm: "IDIVW"}, // arg0 % arg1
		{name: "MODQU", reg: gp11mod, asm: "DIVQ"}, // arg0 % arg1
		{name: "MODLU", reg: gp11mod, asm: "DIVL"}, // arg0 % arg1
		{name: "MODWU", reg: gp11mod, asm: "DIVW"}, // arg0 % arg1

		{name: "ANDQ", reg: gp21, asm: "ANDQ"},      // arg0 & arg1
		{name: "ANDL", reg: gp21, asm: "ANDL"},      // arg0 & arg1
		{name: "ANDW", reg: gp21, asm: "ANDW"},      // arg0 & arg1
		{name: "ANDB", reg: gp21, asm: "ANDB"},      // arg0 & arg1
		{name: "ANDQconst", reg: gp11, asm: "ANDQ"}, // arg0 & auxint
		{name: "ANDLconst", reg: gp11, asm: "ANDL"}, // arg0 & auxint
		{name: "ANDWconst", reg: gp11, asm: "ANDW"}, // arg0 & auxint
		{name: "ANDBconst", reg: gp11, asm: "ANDB"}, // arg0 & auxint

		{name: "ORQ", reg: gp21, asm: "ORQ"},      // arg0 | arg1
		{name: "ORL", reg: gp21, asm: "ORL"},      // arg0 | arg1
		{name: "ORW", reg: gp21, asm: "ORW"},      // arg0 | arg1
		{name: "ORB", reg: gp21, asm: "ORB"},      // arg0 | arg1
		{name: "ORQconst", reg: gp11, asm: "ORQ"}, // arg0 | auxint
		{name: "ORLconst", reg: gp11, asm: "ORL"}, // arg0 | auxint
		{name: "ORWconst", reg: gp11, asm: "ORW"}, // arg0 | auxint
		{name: "ORBconst", reg: gp11, asm: "ORB"}, // arg0 | auxint

		{name: "XORQ", reg: gp21, asm: "XORQ"},      // arg0 ^ arg1
		{name: "XORL", reg: gp21, asm: "XORL"},      // arg0 ^ arg1
		{name: "XORW", reg: gp21, asm: "XORW"},      // arg0 ^ arg1
		{name: "XORB", reg: gp21, asm: "XORB"},      // arg0 ^ arg1
		{name: "XORQconst", reg: gp11, asm: "XORQ"}, // arg0 ^ auxint
		{name: "XORLconst", reg: gp11, asm: "XORL"}, // arg0 ^ auxint
		{name: "XORWconst", reg: gp11, asm: "XORW"}, // arg0 ^ auxint
		{name: "XORBconst", reg: gp11, asm: "XORB"}, // arg0 ^ auxint

		{name: "CMPQ", reg: gp2flags, asm: "CMPQ", typ: "Flags"},      // arg0 compare to arg1
		{name: "CMPL", reg: gp2flags, asm: "CMPL", typ: "Flags"},      // arg0 compare to arg1
		{name: "CMPW", reg: gp2flags, asm: "CMPW", typ: "Flags"},      // arg0 compare to arg1
		{name: "CMPB", reg: gp2flags, asm: "CMPB", typ: "Flags"},      // arg0 compare to arg1
		{name: "CMPQconst", reg: gp1flags, asm: "CMPQ", typ: "Flags"}, // arg0 compare to auxint
		{name: "CMPLconst", reg: gp1flags, asm: "CMPL", typ: "Flags"}, // arg0 compare to auxint
		{name: "CMPWconst", reg: gp1flags, asm: "CMPW", typ: "Flags"}, // arg0 compare to auxint
		{name: "CMPBconst", reg: gp1flags, asm: "CMPB", typ: "Flags"}, // arg0 compare to auxint

		{name: "UCOMISS", reg: fp2flags, asm: "UCOMISS", typ: "Flags"}, // arg0 compare to arg1, f32
		{name: "UCOMISD", reg: fp2flags, asm: "UCOMISD", typ: "Flags"}, // arg0 compare to arg1, f64

		{name: "TESTQ", reg: gp2flags, asm: "TESTQ", typ: "Flags"},      // (arg0 & arg1) compare to 0
		{name: "TESTL", reg: gp2flags, asm: "TESTL", typ: "Flags"},      // (arg0 & arg1) compare to 0
		{name: "TESTW", reg: gp2flags, asm: "TESTW", typ: "Flags"},      // (arg0 & arg1) compare to 0
		{name: "TESTB", reg: gp2flags, asm: "TESTB", typ: "Flags"},      // (arg0 & arg1) compare to 0
		{name: "TESTQconst", reg: gp1flags, asm: "TESTQ", typ: "Flags"}, // (arg0 & auxint) compare to 0
		{name: "TESTLconst", reg: gp1flags, asm: "TESTL", typ: "Flags"}, // (arg0 & auxint) compare to 0
		{name: "TESTWconst", reg: gp1flags, asm: "TESTW", typ: "Flags"}, // (arg0 & auxint) compare to 0
		{name: "TESTBconst", reg: gp1flags, asm: "TESTB", typ: "Flags"}, // (arg0 & auxint) compare to 0

		{name: "SHLQ", reg: gp21shift, asm: "SHLQ"}, // arg0 << arg1, shift amount is mod 64
		{name: "SHLL", reg: gp21shift, asm: "SHLL"}, // arg0 << arg1, shift amount is mod 32
		{name: "SHLW", reg: gp21shift, asm: "SHLW"}, // arg0 << arg1, shift amount is mod 32
		{name: "SHLB", reg: gp21shift, asm: "SHLB"}, // arg0 << arg1, shift amount is mod 32
		{name: "SHLQconst", reg: gp11, asm: "SHLQ"}, // arg0 << auxint, shift amount 0-63
		{name: "SHLLconst", reg: gp11, asm: "SHLL"}, // arg0 << auxint, shift amount 0-31
		{name: "SHLWconst", reg: gp11, asm: "SHLW"}, // arg0 << auxint, shift amount 0-31
		{name: "SHLBconst", reg: gp11, asm: "SHLB"}, // arg0 << auxint, shift amount 0-31
		// Note: x86 is weird, the 16 and 8 byte shifts still use all 5 bits of shift amount!

		{name: "SHRQ", reg: gp21shift, asm: "SHRQ"}, // unsigned arg0 >> arg1, shift amount is mod 64
		{name: "SHRL", reg: gp21shift, asm: "SHRL"}, // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRW", reg: gp21shift, asm: "SHRW"}, // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRB", reg: gp21shift, asm: "SHRB"}, // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRQconst", reg: gp11, asm: "SHRQ"}, // unsigned arg0 >> auxint, shift amount 0-63
		{name: "SHRLconst", reg: gp11, asm: "SHRL"}, // unsigned arg0 >> auxint, shift amount 0-31
		{name: "SHRWconst", reg: gp11, asm: "SHRW"}, // unsigned arg0 >> auxint, shift amount 0-31
		{name: "SHRBconst", reg: gp11, asm: "SHRB"}, // unsigned arg0 >> auxint, shift amount 0-31

		{name: "SARQ", reg: gp21shift, asm: "SARQ"}, // signed arg0 >> arg1, shift amount is mod 64
		{name: "SARL", reg: gp21shift, asm: "SARL"}, // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARW", reg: gp21shift, asm: "SARW"}, // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARB", reg: gp21shift, asm: "SARB"}, // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARQconst", reg: gp11, asm: "SARQ"}, // signed arg0 >> auxint, shift amount 0-63
		{name: "SARLconst", reg: gp11, asm: "SARL"}, // signed arg0 >> auxint, shift amount 0-31
		{name: "SARWconst", reg: gp11, asm: "SARW"}, // signed arg0 >> auxint, shift amount 0-31
		{name: "SARBconst", reg: gp11, asm: "SARB"}, // signed arg0 >> auxint, shift amount 0-31

		{name: "ROLQconst", reg: gp11, asm: "ROLQ"}, // arg0 rotate left auxint, rotate amount 0-63
		{name: "ROLLconst", reg: gp11, asm: "ROLL"}, // arg0 rotate left auxint, rotate amount 0-31
		{name: "ROLWconst", reg: gp11, asm: "ROLW"}, // arg0 rotate left auxint, rotate amount 0-15
		{name: "ROLBconst", reg: gp11, asm: "ROLB"}, // arg0 rotate left auxint, rotate amount 0-7

		// unary ops
		{name: "NEGQ", reg: gp11, asm: "NEGQ"}, // -arg0
		{name: "NEGL", reg: gp11, asm: "NEGL"}, // -arg0
		{name: "NEGW", reg: gp11, asm: "NEGW"}, // -arg0
		{name: "NEGB", reg: gp11, asm: "NEGB"}, // -arg0

		{name: "NOTQ", reg: gp11, asm: "NOTQ"}, // ^arg0
		{name: "NOTL", reg: gp11, asm: "NOTL"}, // ^arg0
		{name: "NOTW", reg: gp11, asm: "NOTW"}, // ^arg0
		{name: "NOTB", reg: gp11, asm: "NOTB"}, // ^arg0

		{name: "SQRTSD", reg: fp11, asm: "SQRTSD"}, // sqrt(arg0)

		{name: "SBBQcarrymask", reg: flagsgp, asm: "SBBQ"}, // (int64)(-1) if carry is set, 0 if carry is clear.
		{name: "SBBLcarrymask", reg: flagsgp, asm: "SBBL"}, // (int32)(-1) if carry is set, 0 if carry is clear.
		// Note: SBBW and SBBB are subsumed by SBBL

		{name: "SETEQ", reg: readflags, asm: "SETEQ"}, // extract == condition from arg0
		{name: "SETNE", reg: readflags, asm: "SETNE"}, // extract != condition from arg0
		{name: "SETL", reg: readflags, asm: "SETLT"},  // extract signed < condition from arg0
		{name: "SETLE", reg: readflags, asm: "SETLE"}, // extract signed <= condition from arg0
		{name: "SETG", reg: readflags, asm: "SETGT"},  // extract signed > condition from arg0
		{name: "SETGE", reg: readflags, asm: "SETGE"}, // extract signed >= condition from arg0
		{name: "SETB", reg: readflags, asm: "SETCS"},  // extract unsigned < condition from arg0
		{name: "SETBE", reg: readflags, asm: "SETLS"}, // extract unsigned <= condition from arg0
		{name: "SETA", reg: readflags, asm: "SETHI"},  // extract unsigned > condition from arg0
		{name: "SETAE", reg: readflags, asm: "SETCC"}, // extract unsigned >= condition from arg0
		// Need different opcodes for floating point conditions because
		// any comparison involving a NaN is always FALSE and thus
		// the patterns for inverting conditions cannot be used.
		{name: "SETEQF", reg: flagsgpax, asm: "SETEQ"}, // extract == condition from arg0
		{name: "SETNEF", reg: flagsgpax, asm: "SETNE"}, // extract != condition from arg0
		{name: "SETORD", reg: flagsgp, asm: "SETPC"},   // extract "ordered" (No Nan present) condition from arg0
		{name: "SETNAN", reg: flagsgp, asm: "SETPS"},   // extract "unordered" (Nan present) condition from arg0

		{name: "SETGF", reg: flagsgp, asm: "SETHI"},  // extract floating > condition from arg0
		{name: "SETGEF", reg: flagsgp, asm: "SETCC"}, // extract floating >= condition from arg0

		{name: "MOVBQSX", reg: gp11nf, asm: "MOVBQSX"}, // sign extend arg0 from int8 to int64
		{name: "MOVBQZX", reg: gp11nf, asm: "MOVBQZX"}, // zero extend arg0 from int8 to int64
		{name: "MOVWQSX", reg: gp11nf, asm: "MOVWQSX"}, // sign extend arg0 from int16 to int64
		{name: "MOVWQZX", reg: gp11nf, asm: "MOVWQZX"}, // zero extend arg0 from int16 to int64
		{name: "MOVLQSX", reg: gp11nf, asm: "MOVLQSX"}, // sign extend arg0 from int32 to int64
		{name: "MOVLQZX", reg: gp11nf, asm: "MOVLQZX"}, // zero extend arg0 from int32 to int64

		// clobbers flags as liblink will rewrite these to XOR reg, reg if the constant is zero
		// TODO: revisit when issue 12405 is fixed
		{name: "MOVBconst", reg: gp01flags, asm: "MOVB", typ: "UInt8"},  // 8 low bits of auxint
		{name: "MOVWconst", reg: gp01flags, asm: "MOVW", typ: "UInt16"}, // 16 low bits of auxint
		{name: "MOVLconst", reg: gp01flags, asm: "MOVL", typ: "UInt32"}, // 32 low bits of auxint
		{name: "MOVQconst", reg: gp01flags, asm: "MOVQ", typ: "UInt64"}, // auxint

		{name: "CVTTSD2SL", reg: fpgp, asm: "CVTTSD2SL"}, // convert float64 to int32
		{name: "CVTTSD2SQ", reg: fpgp, asm: "CVTTSD2SQ"}, // convert float64 to int64
		{name: "CVTTSS2SL", reg: fpgp, asm: "CVTTSS2SL"}, // convert float32 to int32
		{name: "CVTTSS2SQ", reg: fpgp, asm: "CVTTSS2SQ"}, // convert float32 to int64
		{name: "CVTSL2SS", reg: gpfp, asm: "CVTSL2SS"},   // convert int32 to float32
		{name: "CVTSL2SD", reg: gpfp, asm: "CVTSL2SD"},   // convert int32 to float64
		{name: "CVTSQ2SS", reg: gpfp, asm: "CVTSQ2SS"},   // convert int64 to float32
		{name: "CVTSQ2SD", reg: gpfp, asm: "CVTSQ2SD"},   // convert int64 to float64
		{name: "CVTSD2SS", reg: fp11, asm: "CVTSD2SS"},   // convert float64 to float32
		{name: "CVTSS2SD", reg: fp11, asm: "CVTSS2SD"},   // convert float32 to float64

		{name: "PXOR", reg: fp21, asm: "PXOR"}, // exclusive or, applied to X regs for float negation.

		{name: "LEAQ", reg: gp11sb},  // arg0 + auxint + offset encoded in aux
		{name: "LEAQ1", reg: gp21sb}, // arg0 + arg1 + auxint
		{name: "LEAQ2", reg: gp21sb}, // arg0 + 2*arg1 + auxint
		{name: "LEAQ4", reg: gp21sb}, // arg0 + 4*arg1 + auxint
		{name: "LEAQ8", reg: gp21sb}, // arg0 + 8*arg1 + auxint

		// auxint+aux == add auxint and the offset of the symbol in aux (if any) to the effective address
		{name: "MOVBload", reg: gpload, asm: "MOVB"},               // load byte from arg0+auxint+aux. arg1=mem
		{name: "MOVBQSXload", reg: gpload, asm: "MOVBQSX"},         // ditto, extend to int64
		{name: "MOVBQZXload", reg: gpload, asm: "MOVBQZX"},         // ditto, extend to uint64
		{name: "MOVWload", reg: gpload, asm: "MOVW"},               // load 2 bytes from arg0+auxint+aux. arg1=mem
		{name: "MOVLload", reg: gpload, asm: "MOVL"},               // load 4 bytes from arg0+auxint+aux. arg1=mem
		{name: "MOVQload", reg: gpload, asm: "MOVQ"},               // load 8 bytes from arg0+auxint+aux. arg1=mem
		{name: "MOVQloadidx8", reg: gploadidx, asm: "MOVQ"},        // load 8 bytes from arg0+8*arg1+auxint+aux. arg2=mem
		{name: "MOVBstore", reg: gpstore, asm: "MOVB", typ: "Mem"}, // store byte in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVWstore", reg: gpstore, asm: "MOVW", typ: "Mem"}, // store 2 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVLstore", reg: gpstore, asm: "MOVL", typ: "Mem"}, // store 4 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVQstore", reg: gpstore, asm: "MOVQ", typ: "Mem"}, // store 8 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVQstoreidx8", reg: gpstoreidx, asm: "MOVQ"},      // store 8 bytes in arg2 to arg0+8*arg1+auxint+aux. arg3=mem

		// arg0 = (duff-adjusted) pointer to start of memory to zero
		// arg1 = value to store (will always be zero)
		// arg2 = mem
		// auxint = offset into duffzero code to start executing
		// returns mem
		{
			name: "DUFFZERO",
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("AX")},
				clobbers: buildReg("DI FLAGS"),
			},
		},

		// arg0 = address of memory to zero
		// arg1 = # of 8-byte words to zero
		// arg2 = value to store (will always be zero)
		// arg3 = mem
		// returns mem
		{
			name: "REPSTOSQ",
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("CX"), buildReg("AX")},
				clobbers: buildReg("DI CX FLAGS"),
			},
		},

		{name: "CALLstatic", reg: regInfo{clobbers: callerSave}},                                 // call static function aux.(*gc.Sym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", reg: regInfo{[]regMask{gpsp, buildReg("DX"), 0}, callerSave, nil}}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLdefer", reg: regInfo{clobbers: callerSave}},                                  // call deferproc.  arg0=mem, auxint=argsize, returns mem
		{name: "CALLgo", reg: regInfo{clobbers: callerSave}},                                     // call newproc.  arg0=mem, auxint=argsize, returns mem
		{name: "CALLinter", reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}},           // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		{name: "REPMOVSB", reg: regInfo{[]regMask{buildReg("DI"), buildReg("SI"), buildReg("CX")}, buildReg("DI SI CX"), nil}}, // move arg2 bytes from arg1 to arg0.  arg3=mem, returns memory

		// (InvertFlags (CMPQ a b)) == (CMPQ b a)
		// So if we want (SETL (CMPQ a b)) but we can't do that because a is a constant,
		// then we do (SETL (InvertFlags (CMPQ b a))) instead.
		// Rewrites will convert this to (SETG (CMPQ b a)).
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags"}, // reverse direction of arg0

		// Pseudo-ops
		{name: "LoweredPanicNilCheck", reg: gp10},
		{name: "LoweredGetG", reg: gp01},
		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of DX (the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("DX")}}},
	}

	var AMD64blocks = []blockData{
		{name: "EQ"},
		{name: "NE"},
		{name: "LT"},
		{name: "LE"},
		{name: "GT"},
		{name: "GE"},
		{name: "ULT"},
		{name: "ULE"},
		{name: "UGT"},
		{name: "UGE"},
		{name: "EQF"},
		{name: "NEF"},
		{name: "ORD"}, // FP, ordered comparison (parity zero)
		{name: "NAN"}, // FP, unordered comparison (parity one)
	}

	archs = append(archs, arch{"AMD64", AMD64ops, AMD64blocks, regNamesAMD64})
}
