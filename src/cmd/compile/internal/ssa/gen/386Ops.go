// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import "strings"

// Notes:
//  - Integer types live in the low portion of registers. Upper portions are junk.
//  - Boolean types use the low-order byte of a register. 0=false, 1=true.
//    Upper bytes are junk.
//  - Floating-point types live in the low natural slot of an sse2 register.
//    Unused portions are junk.
//  - We do not use AH,BH,CH,DH registers.
//  - When doing sub-register operations, we try to write the whole
//    destination register to avoid a partial-register write.
//  - Unused portions of AuxInt (or the Val portion of ValAndOff) are
//    filled by sign-extending the used portion.  Users of AuxInt which interpret
//    AuxInt as unsigned (e.g. shifts) must be careful.

// Suffixes encode the bit width of various instructions.
// L (long word) = 32 bit
// W (word)      = 16 bit
// B (byte)      = 8 bit

// copied from ../../x86/reg.go
var regNames386 = []string{
	"AX",
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",
	"X0",
	"X1",
	"X2",
	"X3",
	"X4",
	"X5",
	"X6",
	"X7",

	// If you add registers, update asyncPreempt in runtime

	// pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNames386) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNames386 {
		num[name] = i
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
		bx         = buildReg("BX")
		si         = buildReg("SI")
		gp         = buildReg("AX CX DX BX BP SI DI")
		fp         = buildReg("X0 X1 X2 X3 X4 X5 X6 X7")
		gpsp       = gp | buildReg("SP")
		gpspsb     = gpsp | buildReg("SB")
		callerSave = gp | fp
	)
	// Common slices of register masks
	var (
		gponly = []regMask{gp}
		fponly = []regMask{fp}
	)

	// Common regInfo
	var (
		gp01      = regInfo{inputs: nil, outputs: gponly}
		gp11      = regInfo{inputs: []regMask{gp}, outputs: gponly}
		gp11sp    = regInfo{inputs: []regMask{gpsp}, outputs: gponly}
		gp11sb    = regInfo{inputs: []regMask{gpspsb}, outputs: gponly}
		gp21      = regInfo{inputs: []regMask{gp, gp}, outputs: gponly}
		gp11carry = regInfo{inputs: []regMask{gp}, outputs: []regMask{gp, 0}}
		gp21carry = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp, 0}}
		gp1carry1 = regInfo{inputs: []regMask{gp}, outputs: gponly}
		gp2carry1 = regInfo{inputs: []regMask{gp, gp}, outputs: gponly}
		gp21sp    = regInfo{inputs: []regMask{gpsp, gp}, outputs: gponly}
		gp21sb    = regInfo{inputs: []regMask{gpspsb, gpsp}, outputs: gponly}
		gp21shift = regInfo{inputs: []regMask{gp, cx}, outputs: []regMask{gp}}
		gp11div   = regInfo{inputs: []regMask{ax, gpsp &^ dx}, outputs: []regMask{ax}, clobbers: dx}
		gp21hmul  = regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{dx}, clobbers: ax}
		gp11mod   = regInfo{inputs: []regMask{ax, gpsp &^ dx}, outputs: []regMask{dx}, clobbers: ax}
		gp21mul   = regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{dx, ax}}

		gp2flags     = regInfo{inputs: []regMask{gpsp, gpsp}}
		gp1flags     = regInfo{inputs: []regMask{gpsp}}
		gp0flagsLoad = regInfo{inputs: []regMask{gpspsb, 0}}
		gp1flagsLoad = regInfo{inputs: []regMask{gpspsb, gpsp, 0}}
		flagsgp      = regInfo{inputs: nil, outputs: gponly}

		readflags = regInfo{inputs: nil, outputs: gponly}
		flagsgpax = regInfo{inputs: nil, clobbers: ax, outputs: []regMask{gp &^ ax}}

		gpload      = regInfo{inputs: []regMask{gpspsb, 0}, outputs: gponly}
		gp21load    = regInfo{inputs: []regMask{gp, gpspsb, 0}, outputs: gponly}
		gploadidx   = regInfo{inputs: []regMask{gpspsb, gpsp, 0}, outputs: gponly}
		gp21loadidx = regInfo{inputs: []regMask{gp, gpspsb, gpsp, 0}, outputs: gponly}

		gpstore         = regInfo{inputs: []regMask{gpspsb, gpsp, 0}}
		gpstoreconst    = regInfo{inputs: []regMask{gpspsb, 0}}
		gpstoreidx      = regInfo{inputs: []regMask{gpspsb, gpsp, gpsp, 0}}
		gpstoreconstidx = regInfo{inputs: []regMask{gpspsb, gpsp, 0}}

		fp01     = regInfo{inputs: nil, outputs: fponly}
		fp21     = regInfo{inputs: []regMask{fp, fp}, outputs: fponly}
		fp21load = regInfo{inputs: []regMask{fp, gpspsb, 0}, outputs: fponly}
		fpgp     = regInfo{inputs: fponly, outputs: gponly}
		gpfp     = regInfo{inputs: gponly, outputs: fponly}
		fp11     = regInfo{inputs: fponly, outputs: fponly}
		fp2flags = regInfo{inputs: []regMask{fp, fp}}

		fpload    = regInfo{inputs: []regMask{gpspsb, 0}, outputs: fponly}
		fploadidx = regInfo{inputs: []regMask{gpspsb, gpsp, 0}, outputs: fponly}

		fpstore    = regInfo{inputs: []regMask{gpspsb, fp, 0}}
		fpstoreidx = regInfo{inputs: []regMask{gpspsb, gpsp, fp, 0}}
	)

	var _386ops = []opData{
		// fp ops
		{name: "ADDSS", argLength: 2, reg: fp21, asm: "ADDSS", commutative: true, resultInArg0: true}, // fp32 add
		{name: "ADDSD", argLength: 2, reg: fp21, asm: "ADDSD", commutative: true, resultInArg0: true}, // fp64 add
		{name: "SUBSS", argLength: 2, reg: fp21, asm: "SUBSS", resultInArg0: true},                    // fp32 sub
		{name: "SUBSD", argLength: 2, reg: fp21, asm: "SUBSD", resultInArg0: true},                    // fp64 sub
		{name: "MULSS", argLength: 2, reg: fp21, asm: "MULSS", commutative: true, resultInArg0: true}, // fp32 mul
		{name: "MULSD", argLength: 2, reg: fp21, asm: "MULSD", commutative: true, resultInArg0: true}, // fp64 mul
		{name: "DIVSS", argLength: 2, reg: fp21, asm: "DIVSS", resultInArg0: true},                    // fp32 div
		{name: "DIVSD", argLength: 2, reg: fp21, asm: "DIVSD", resultInArg0: true},                    // fp64 div

		{name: "MOVSSload", argLength: 2, reg: fpload, asm: "MOVSS", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"}, // fp32 load
		{name: "MOVSDload", argLength: 2, reg: fpload, asm: "MOVSD", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"}, // fp64 load
		{name: "MOVSSconst", reg: fp01, asm: "MOVSS", aux: "Float32", rematerializeable: true},                               // fp32 constant
		{name: "MOVSDconst", reg: fp01, asm: "MOVSD", aux: "Float64", rematerializeable: true},                               // fp64 constant
		{name: "MOVSSloadidx1", argLength: 3, reg: fploadidx, asm: "MOVSS", aux: "SymOff", symEffect: "Read"},                // fp32 load indexed by i
		{name: "MOVSSloadidx4", argLength: 3, reg: fploadidx, asm: "MOVSS", aux: "SymOff", symEffect: "Read"},                // fp32 load indexed by 4*i
		{name: "MOVSDloadidx1", argLength: 3, reg: fploadidx, asm: "MOVSD", aux: "SymOff", symEffect: "Read"},                // fp64 load indexed by i
		{name: "MOVSDloadidx8", argLength: 3, reg: fploadidx, asm: "MOVSD", aux: "SymOff", symEffect: "Read"},                // fp64 load indexed by 8*i

		{name: "MOVSSstore", argLength: 3, reg: fpstore, asm: "MOVSS", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"}, // fp32 store
		{name: "MOVSDstore", argLength: 3, reg: fpstore, asm: "MOVSD", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"}, // fp64 store
		{name: "MOVSSstoreidx1", argLength: 4, reg: fpstoreidx, asm: "MOVSS", aux: "SymOff", symEffect: "Write"},                // fp32 indexed by i store
		{name: "MOVSSstoreidx4", argLength: 4, reg: fpstoreidx, asm: "MOVSS", aux: "SymOff", symEffect: "Write"},                // fp32 indexed by 4i store
		{name: "MOVSDstoreidx1", argLength: 4, reg: fpstoreidx, asm: "MOVSD", aux: "SymOff", symEffect: "Write"},                // fp64 indexed by i store
		{name: "MOVSDstoreidx8", argLength: 4, reg: fpstoreidx, asm: "MOVSD", aux: "SymOff", symEffect: "Write"},                // fp64 indexed by 8i store

		{name: "ADDSSload", argLength: 3, reg: fp21load, asm: "ADDSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp32 arg0 + tmp, tmp loaded from arg1+auxint+aux, arg2 = mem
		{name: "ADDSDload", argLength: 3, reg: fp21load, asm: "ADDSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp64 arg0 + tmp, tmp loaded from arg1+auxint+aux, arg2 = mem
		{name: "SUBSSload", argLength: 3, reg: fp21load, asm: "SUBSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp32 arg0 - tmp, tmp loaded from arg1+auxint+aux, arg2 = mem
		{name: "SUBSDload", argLength: 3, reg: fp21load, asm: "SUBSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp64 arg0 - tmp, tmp loaded from arg1+auxint+aux, arg2 = mem
		{name: "MULSSload", argLength: 3, reg: fp21load, asm: "MULSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp32 arg0 * tmp, tmp loaded from arg1+auxint+aux, arg2 = mem
		{name: "MULSDload", argLength: 3, reg: fp21load, asm: "MULSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp64 arg0 * tmp, tmp loaded from arg1+auxint+aux, arg2 = mem
		{name: "DIVSSload", argLength: 3, reg: fp21load, asm: "DIVSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp32 arg0 / tmp, tmp loaded from arg1+auxint+aux, arg2 = mem
		{name: "DIVSDload", argLength: 3, reg: fp21load, asm: "DIVSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"}, // fp64 arg0 / tmp, tmp loaded from arg1+auxint+aux, arg2 = mem

		// binary ops
		{name: "ADDL", argLength: 2, reg: gp21sp, asm: "ADDL", commutative: true, clobberFlags: true},                // arg0 + arg1
		{name: "ADDLconst", argLength: 1, reg: gp11sp, asm: "ADDL", aux: "Int32", typ: "UInt32", clobberFlags: true}, // arg0 + auxint

		{name: "ADDLcarry", argLength: 2, reg: gp21carry, asm: "ADDL", commutative: true, resultInArg0: true},                // arg0 + arg1, generates <carry,result> pair
		{name: "ADDLconstcarry", argLength: 1, reg: gp11carry, asm: "ADDL", aux: "Int32", resultInArg0: true},                // arg0 + auxint, generates <carry,result> pair
		{name: "ADCL", argLength: 3, reg: gp2carry1, asm: "ADCL", commutative: true, resultInArg0: true, clobberFlags: true}, // arg0+arg1+carry(arg2), where arg2 is flags
		{name: "ADCLconst", argLength: 2, reg: gp1carry1, asm: "ADCL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0+auxint+carry(arg1), where arg1 is flags

		{name: "SUBL", argLength: 2, reg: gp21, asm: "SUBL", resultInArg0: true, clobberFlags: true},                    // arg0 - arg1
		{name: "SUBLconst", argLength: 1, reg: gp11, asm: "SUBL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0 - auxint

		{name: "SUBLcarry", argLength: 2, reg: gp21carry, asm: "SUBL", resultInArg0: true},                                   // arg0-arg1, generates <borrow,result> pair
		{name: "SUBLconstcarry", argLength: 1, reg: gp11carry, asm: "SUBL", aux: "Int32", resultInArg0: true},                // arg0-auxint, generates <borrow,result> pair
		{name: "SBBL", argLength: 3, reg: gp2carry1, asm: "SBBL", resultInArg0: true, clobberFlags: true},                    // arg0-arg1-borrow(arg2), where arg2 is flags
		{name: "SBBLconst", argLength: 2, reg: gp1carry1, asm: "SBBL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0-auxint-borrow(arg1), where arg1 is flags

		{name: "MULL", argLength: 2, reg: gp21, asm: "IMULL", commutative: true, resultInArg0: true, clobberFlags: true}, // arg0 * arg1
		{name: "MULLconst", argLength: 1, reg: gp11, asm: "IMUL3L", aux: "Int32", clobberFlags: true},                    // arg0 * auxint

		{name: "MULLU", argLength: 2, reg: regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{ax, 0}, clobbers: dx}, typ: "(UInt32,Flags)", asm: "MULL", commutative: true, clobberFlags: true}, // Let x = arg0*arg1 (full 32x32->64  unsigned multiply). Returns uint32(x), and flags set to overflow if uint32(x) != x.

		{name: "HMULL", argLength: 2, reg: gp21hmul, commutative: true, asm: "IMULL", clobberFlags: true}, // (arg0 * arg1) >> width
		{name: "HMULLU", argLength: 2, reg: gp21hmul, commutative: true, asm: "MULL", clobberFlags: true}, // (arg0 * arg1) >> width

		{name: "MULLQU", argLength: 2, reg: gp21mul, commutative: true, asm: "MULL", clobberFlags: true}, // arg0 * arg1, high 32 in result[0], low 32 in result[1]

		{name: "AVGLU", argLength: 2, reg: gp21, commutative: true, resultInArg0: true, clobberFlags: true}, // (arg0 + arg1) / 2 as unsigned, all 32 result bits

		// For DIVL, DIVW, MODL and MODW, AuxInt non-zero means that the divisor has been proved to be not -1.
		{name: "DIVL", argLength: 2, reg: gp11div, asm: "IDIVL", aux: "Bool", clobberFlags: true}, // arg0 / arg1
		{name: "DIVW", argLength: 2, reg: gp11div, asm: "IDIVW", aux: "Bool", clobberFlags: true}, // arg0 / arg1
		{name: "DIVLU", argLength: 2, reg: gp11div, asm: "DIVL", clobberFlags: true},              // arg0 / arg1
		{name: "DIVWU", argLength: 2, reg: gp11div, asm: "DIVW", clobberFlags: true},              // arg0 / arg1

		{name: "MODL", argLength: 2, reg: gp11mod, asm: "IDIVL", aux: "Bool", clobberFlags: true}, // arg0 % arg1
		{name: "MODW", argLength: 2, reg: gp11mod, asm: "IDIVW", aux: "Bool", clobberFlags: true}, // arg0 % arg1
		{name: "MODLU", argLength: 2, reg: gp11mod, asm: "DIVL", clobberFlags: true},              // arg0 % arg1
		{name: "MODWU", argLength: 2, reg: gp11mod, asm: "DIVW", clobberFlags: true},              // arg0 % arg1

		{name: "ANDL", argLength: 2, reg: gp21, asm: "ANDL", commutative: true, resultInArg0: true, clobberFlags: true}, // arg0 & arg1
		{name: "ANDLconst", argLength: 1, reg: gp11, asm: "ANDL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0 & auxint

		{name: "ORL", argLength: 2, reg: gp21, asm: "ORL", commutative: true, resultInArg0: true, clobberFlags: true}, // arg0 | arg1
		{name: "ORLconst", argLength: 1, reg: gp11, asm: "ORL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0 | auxint

		{name: "XORL", argLength: 2, reg: gp21, asm: "XORL", commutative: true, resultInArg0: true, clobberFlags: true}, // arg0 ^ arg1
		{name: "XORLconst", argLength: 1, reg: gp11, asm: "XORL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0 ^ auxint

		{name: "CMPL", argLength: 2, reg: gp2flags, asm: "CMPL", typ: "Flags"},                    // arg0 compare to arg1
		{name: "CMPW", argLength: 2, reg: gp2flags, asm: "CMPW", typ: "Flags"},                    // arg0 compare to arg1
		{name: "CMPB", argLength: 2, reg: gp2flags, asm: "CMPB", typ: "Flags"},                    // arg0 compare to arg1
		{name: "CMPLconst", argLength: 1, reg: gp1flags, asm: "CMPL", typ: "Flags", aux: "Int32"}, // arg0 compare to auxint
		{name: "CMPWconst", argLength: 1, reg: gp1flags, asm: "CMPW", typ: "Flags", aux: "Int16"}, // arg0 compare to auxint
		{name: "CMPBconst", argLength: 1, reg: gp1flags, asm: "CMPB", typ: "Flags", aux: "Int8"},  // arg0 compare to auxint

		// compare *(arg0+auxint+aux) to arg1 (in that order). arg2=mem.
		{name: "CMPLload", argLength: 3, reg: gp1flagsLoad, asm: "CMPL", aux: "SymOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPWload", argLength: 3, reg: gp1flagsLoad, asm: "CMPW", aux: "SymOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPBload", argLength: 3, reg: gp1flagsLoad, asm: "CMPB", aux: "SymOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},

		// compare *(arg0+ValAndOff(AuxInt).Off()+aux) to ValAndOff(AuxInt).Val() (in that order). arg1=mem.
		{name: "CMPLconstload", argLength: 2, reg: gp0flagsLoad, asm: "CMPL", aux: "SymValAndOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPWconstload", argLength: 2, reg: gp0flagsLoad, asm: "CMPW", aux: "SymValAndOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPBconstload", argLength: 2, reg: gp0flagsLoad, asm: "CMPB", aux: "SymValAndOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},

		{name: "UCOMISS", argLength: 2, reg: fp2flags, asm: "UCOMISS", typ: "Flags"}, // arg0 compare to arg1, f32
		{name: "UCOMISD", argLength: 2, reg: fp2flags, asm: "UCOMISD", typ: "Flags"}, // arg0 compare to arg1, f64

		{name: "TESTL", argLength: 2, reg: gp2flags, commutative: true, asm: "TESTL", typ: "Flags"}, // (arg0 & arg1) compare to 0
		{name: "TESTW", argLength: 2, reg: gp2flags, commutative: true, asm: "TESTW", typ: "Flags"}, // (arg0 & arg1) compare to 0
		{name: "TESTB", argLength: 2, reg: gp2flags, commutative: true, asm: "TESTB", typ: "Flags"}, // (arg0 & arg1) compare to 0
		{name: "TESTLconst", argLength: 1, reg: gp1flags, asm: "TESTL", typ: "Flags", aux: "Int32"}, // (arg0 & auxint) compare to 0
		{name: "TESTWconst", argLength: 1, reg: gp1flags, asm: "TESTW", typ: "Flags", aux: "Int16"}, // (arg0 & auxint) compare to 0
		{name: "TESTBconst", argLength: 1, reg: gp1flags, asm: "TESTB", typ: "Flags", aux: "Int8"},  // (arg0 & auxint) compare to 0

		{name: "SHLL", argLength: 2, reg: gp21shift, asm: "SHLL", resultInArg0: true, clobberFlags: true},               // arg0 << arg1, shift amount is mod 32
		{name: "SHLLconst", argLength: 1, reg: gp11, asm: "SHLL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0 << auxint, shift amount 0-31
		// Note: x86 is weird, the 16 and 8 byte shifts still use all 5 bits of shift amount!

		{name: "SHRL", argLength: 2, reg: gp21shift, asm: "SHRL", resultInArg0: true, clobberFlags: true},               // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRW", argLength: 2, reg: gp21shift, asm: "SHRW", resultInArg0: true, clobberFlags: true},               // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRB", argLength: 2, reg: gp21shift, asm: "SHRB", resultInArg0: true, clobberFlags: true},               // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRLconst", argLength: 1, reg: gp11, asm: "SHRL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // unsigned arg0 >> auxint, shift amount 0-31
		{name: "SHRWconst", argLength: 1, reg: gp11, asm: "SHRW", aux: "Int16", resultInArg0: true, clobberFlags: true}, // unsigned arg0 >> auxint, shift amount 0-15
		{name: "SHRBconst", argLength: 1, reg: gp11, asm: "SHRB", aux: "Int8", resultInArg0: true, clobberFlags: true},  // unsigned arg0 >> auxint, shift amount 0-7

		{name: "SARL", argLength: 2, reg: gp21shift, asm: "SARL", resultInArg0: true, clobberFlags: true},               // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARW", argLength: 2, reg: gp21shift, asm: "SARW", resultInArg0: true, clobberFlags: true},               // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARB", argLength: 2, reg: gp21shift, asm: "SARB", resultInArg0: true, clobberFlags: true},               // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARLconst", argLength: 1, reg: gp11, asm: "SARL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // signed arg0 >> auxint, shift amount 0-31
		{name: "SARWconst", argLength: 1, reg: gp11, asm: "SARW", aux: "Int16", resultInArg0: true, clobberFlags: true}, // signed arg0 >> auxint, shift amount 0-15
		{name: "SARBconst", argLength: 1, reg: gp11, asm: "SARB", aux: "Int8", resultInArg0: true, clobberFlags: true},  // signed arg0 >> auxint, shift amount 0-7

		{name: "ROLLconst", argLength: 1, reg: gp11, asm: "ROLL", aux: "Int32", resultInArg0: true, clobberFlags: true}, // arg0 rotate left auxint, rotate amount 0-31
		{name: "ROLWconst", argLength: 1, reg: gp11, asm: "ROLW", aux: "Int16", resultInArg0: true, clobberFlags: true}, // arg0 rotate left auxint, rotate amount 0-15
		{name: "ROLBconst", argLength: 1, reg: gp11, asm: "ROLB", aux: "Int8", resultInArg0: true, clobberFlags: true},  // arg0 rotate left auxint, rotate amount 0-7

		// binary-op with a memory source operand
		{name: "ADDLload", argLength: 3, reg: gp21load, asm: "ADDL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},  // arg0 + tmp, tmp loaded from  arg1+auxint+aux, arg2 = mem
		{name: "SUBLload", argLength: 3, reg: gp21load, asm: "SUBL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},  // arg0 - tmp, tmp loaded from  arg1+auxint+aux, arg2 = mem
		{name: "MULLload", argLength: 3, reg: gp21load, asm: "IMULL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 * tmp, tmp loaded from  arg1+auxint+aux, arg2 = mem
		{name: "ANDLload", argLength: 3, reg: gp21load, asm: "ANDL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},  // arg0 & tmp, tmp loaded from  arg1+auxint+aux, arg2 = mem
		{name: "ORLload", argLength: 3, reg: gp21load, asm: "ORL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},    // arg0 | tmp, tmp loaded from  arg1+auxint+aux, arg2 = mem
		{name: "XORLload", argLength: 3, reg: gp21load, asm: "XORL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},  // arg0 ^ tmp, tmp loaded from  arg1+auxint+aux, arg2 = mem

		// binary-op with an indexed memory source operand
		{name: "ADDLloadidx4", argLength: 4, reg: gp21loadidx, asm: "ADDL", aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},  // arg0 + tmp, tmp loaded from  arg1+arg2*4+auxint+aux, arg3 = mem
		{name: "SUBLloadidx4", argLength: 4, reg: gp21loadidx, asm: "SUBL", aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},  // arg0 - tmp, tmp loaded from  arg1+arg2*4+auxint+aux, arg3 = mem
		{name: "MULLloadidx4", argLength: 4, reg: gp21loadidx, asm: "IMULL", aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"}, // arg0 * tmp, tmp loaded from  arg1+arg2*4+auxint+aux, arg3 = mem
		{name: "ANDLloadidx4", argLength: 4, reg: gp21loadidx, asm: "ANDL", aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},  // arg0 & tmp, tmp loaded from  arg1+arg2*4+auxint+aux, arg3 = mem
		{name: "ORLloadidx4", argLength: 4, reg: gp21loadidx, asm: "ORL", aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},    // arg0 | tmp, tmp loaded from  arg1+arg2*4+auxint+aux, arg3 = mem
		{name: "XORLloadidx4", argLength: 4, reg: gp21loadidx, asm: "XORL", aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},  // arg0 ^ tmp, tmp loaded from  arg1+arg2*4+auxint+aux, arg3 = mem

		// unary ops
		{name: "NEGL", argLength: 1, reg: gp11, asm: "NEGL", resultInArg0: true, clobberFlags: true}, // -arg0

		{name: "NOTL", argLength: 1, reg: gp11, asm: "NOTL", resultInArg0: true}, // ^arg0

		{name: "BSFL", argLength: 1, reg: gp11, asm: "BSFL", clobberFlags: true}, // arg0 # of low-order zeroes ; undef if zero
		{name: "BSFW", argLength: 1, reg: gp11, asm: "BSFW", clobberFlags: true}, // arg0 # of low-order zeroes ; undef if zero

		{name: "BSRL", argLength: 1, reg: gp11, asm: "BSRL", clobberFlags: true}, // arg0 # of high-order zeroes ; undef if zero
		{name: "BSRW", argLength: 1, reg: gp11, asm: "BSRW", clobberFlags: true}, // arg0 # of high-order zeroes ; undef if zero

		{name: "BSWAPL", argLength: 1, reg: gp11, asm: "BSWAPL", resultInArg0: true, clobberFlags: true}, // arg0 swap bytes

		{name: "SQRTSD", argLength: 1, reg: fp11, asm: "SQRTSD"}, // sqrt(arg0)
		{name: "SQRTSS", argLength: 1, reg: fp11, asm: "SQRTSS"}, // sqrt(arg0), float32

		{name: "SBBLcarrymask", argLength: 1, reg: flagsgp, asm: "SBBL"}, // (int32)(-1) if carry is set, 0 if carry is clear.
		// Note: SBBW and SBBB are subsumed by SBBL

		{name: "SETEQ", argLength: 1, reg: readflags, asm: "SETEQ"}, // extract == condition from arg0
		{name: "SETNE", argLength: 1, reg: readflags, asm: "SETNE"}, // extract != condition from arg0
		{name: "SETL", argLength: 1, reg: readflags, asm: "SETLT"},  // extract signed < condition from arg0
		{name: "SETLE", argLength: 1, reg: readflags, asm: "SETLE"}, // extract signed <= condition from arg0
		{name: "SETG", argLength: 1, reg: readflags, asm: "SETGT"},  // extract signed > condition from arg0
		{name: "SETGE", argLength: 1, reg: readflags, asm: "SETGE"}, // extract signed >= condition from arg0
		{name: "SETB", argLength: 1, reg: readflags, asm: "SETCS"},  // extract unsigned < condition from arg0
		{name: "SETBE", argLength: 1, reg: readflags, asm: "SETLS"}, // extract unsigned <= condition from arg0
		{name: "SETA", argLength: 1, reg: readflags, asm: "SETHI"},  // extract unsigned > condition from arg0
		{name: "SETAE", argLength: 1, reg: readflags, asm: "SETCC"}, // extract unsigned >= condition from arg0
		{name: "SETO", argLength: 1, reg: readflags, asm: "SETOS"},  // extract if overflow flag is set from arg0
		// Need different opcodes for floating point conditions because
		// any comparison involving a NaN is always FALSE and thus
		// the patterns for inverting conditions cannot be used.
		{name: "SETEQF", argLength: 1, reg: flagsgpax, asm: "SETEQ", clobberFlags: true}, // extract == condition from arg0
		{name: "SETNEF", argLength: 1, reg: flagsgpax, asm: "SETNE", clobberFlags: true}, // extract != condition from arg0
		{name: "SETORD", argLength: 1, reg: flagsgp, asm: "SETPC"},                       // extract "ordered" (No Nan present) condition from arg0
		{name: "SETNAN", argLength: 1, reg: flagsgp, asm: "SETPS"},                       // extract "unordered" (Nan present) condition from arg0

		{name: "SETGF", argLength: 1, reg: flagsgp, asm: "SETHI"},  // extract floating > condition from arg0
		{name: "SETGEF", argLength: 1, reg: flagsgp, asm: "SETCC"}, // extract floating >= condition from arg0

		{name: "MOVBLSX", argLength: 1, reg: gp11, asm: "MOVBLSX"}, // sign extend arg0 from int8 to int32
		{name: "MOVBLZX", argLength: 1, reg: gp11, asm: "MOVBLZX"}, // zero extend arg0 from int8 to int32
		{name: "MOVWLSX", argLength: 1, reg: gp11, asm: "MOVWLSX"}, // sign extend arg0 from int16 to int32
		{name: "MOVWLZX", argLength: 1, reg: gp11, asm: "MOVWLZX"}, // zero extend arg0 from int16 to int32

		{name: "MOVLconst", reg: gp01, asm: "MOVL", typ: "UInt32", aux: "Int32", rematerializeable: true}, // 32 low bits of auxint

		{name: "CVTTSD2SL", argLength: 1, reg: fpgp, asm: "CVTTSD2SL"}, // convert float64 to int32
		{name: "CVTTSS2SL", argLength: 1, reg: fpgp, asm: "CVTTSS2SL"}, // convert float32 to int32
		{name: "CVTSL2SS", argLength: 1, reg: gpfp, asm: "CVTSL2SS"},   // convert int32 to float32
		{name: "CVTSL2SD", argLength: 1, reg: gpfp, asm: "CVTSL2SD"},   // convert int32 to float64
		{name: "CVTSD2SS", argLength: 1, reg: fp11, asm: "CVTSD2SS"},   // convert float64 to float32
		{name: "CVTSS2SD", argLength: 1, reg: fp11, asm: "CVTSS2SD"},   // convert float32 to float64

		{name: "PXOR", argLength: 2, reg: fp21, asm: "PXOR", commutative: true, resultInArg0: true}, // exclusive or, applied to X regs for float negation.

		{name: "LEAL", argLength: 1, reg: gp11sb, aux: "SymOff", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxint + offset encoded in aux
		{name: "LEAL1", argLength: 2, reg: gp21sb, commutative: true, aux: "SymOff", symEffect: "Addr"},      // arg0 + arg1 + auxint + aux
		{name: "LEAL2", argLength: 2, reg: gp21sb, aux: "SymOff", symEffect: "Addr"},                         // arg0 + 2*arg1 + auxint + aux
		{name: "LEAL4", argLength: 2, reg: gp21sb, aux: "SymOff", symEffect: "Addr"},                         // arg0 + 4*arg1 + auxint + aux
		{name: "LEAL8", argLength: 2, reg: gp21sb, aux: "SymOff", symEffect: "Addr"},                         // arg0 + 8*arg1 + auxint + aux
		// Note: LEAL{1,2,4,8} must not have OpSB as either argument.

		// auxint+aux == add auxint and the offset of the symbol in aux (if any) to the effective address
		{name: "MOVBload", argLength: 2, reg: gpload, asm: "MOVBLZX", aux: "SymOff", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},  // load byte from arg0+auxint+aux. arg1=mem.  Zero extend.
		{name: "MOVBLSXload", argLength: 2, reg: gpload, asm: "MOVBLSX", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},             // ditto, sign extend to int32
		{name: "MOVWload", argLength: 2, reg: gpload, asm: "MOVWLZX", aux: "SymOff", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"}, // load 2 bytes from arg0+auxint+aux. arg1=mem.  Zero extend.
		{name: "MOVWLSXload", argLength: 2, reg: gpload, asm: "MOVWLSX", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},             // ditto, sign extend to int32
		{name: "MOVLload", argLength: 2, reg: gpload, asm: "MOVL", aux: "SymOff", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"},    // load 4 bytes from arg0+auxint+aux. arg1=mem.  Zero extend.
		{name: "MOVBstore", argLength: 3, reg: gpstore, asm: "MOVB", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},    // store byte in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVWstore", argLength: 3, reg: gpstore, asm: "MOVW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},    // store 2 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVLstore", argLength: 3, reg: gpstore, asm: "MOVL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},    // store 4 bytes in arg1 to arg0+auxint+aux. arg2=mem

		// direct binary-op on memory (read-modify-write)
		{name: "ADDLmodify", argLength: 3, reg: gpstore, asm: "ADDL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+auxint+aux) += arg1, arg2=mem
		{name: "SUBLmodify", argLength: 3, reg: gpstore, asm: "SUBL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+auxint+aux) -= arg1, arg2=mem
		{name: "ANDLmodify", argLength: 3, reg: gpstore, asm: "ANDL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+auxint+aux) &= arg1, arg2=mem
		{name: "ORLmodify", argLength: 3, reg: gpstore, asm: "ORL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, clobberFlags: true, symEffect: "Read,Write"},   // *(arg0+auxint+aux) |= arg1, arg2=mem
		{name: "XORLmodify", argLength: 3, reg: gpstore, asm: "XORL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+auxint+aux) ^= arg1, arg2=mem

		// direct binary-op on indexed memory (read-modify-write)
		{name: "ADDLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "ADDL", aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+arg1*4+auxint+aux) += arg2, arg3=mem
		{name: "SUBLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "SUBL", aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+arg1*4+auxint+aux) -= arg2, arg3=mem
		{name: "ANDLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "ANDL", aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+arg1*4+auxint+aux) &= arg2, arg3=mem
		{name: "ORLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "ORL", aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},   // *(arg0+arg1*4+auxint+aux) |= arg2, arg3=mem
		{name: "XORLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "XORL", aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"}, // *(arg0+arg1*4+auxint+aux) ^= arg2, arg3=mem

		// direct binary-op on memory with a constant (read-modify-write)
		{name: "ADDLconstmodify", argLength: 2, reg: gpstoreconst, asm: "ADDL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // add ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem
		{name: "ANDLconstmodify", argLength: 2, reg: gpstoreconst, asm: "ANDL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // and ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem
		{name: "ORLconstmodify", argLength: 2, reg: gpstoreconst, asm: "ORL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},   // or  ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem
		{name: "XORLconstmodify", argLength: 2, reg: gpstoreconst, asm: "XORL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // xor ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem

		// direct binary-op on indexed memory with a constant (read-modify-write)
		{name: "ADDLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "ADDL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"}, // add ValAndOff(AuxInt).Val() to arg0+arg1*4+ValAndOff(AuxInt).Off()+aux, arg2=mem
		{name: "ANDLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "ANDL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"}, // and ValAndOff(AuxInt).Val() to arg0+arg1*4+ValAndOff(AuxInt).Off()+aux, arg2=mem
		{name: "ORLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "ORL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},   // or  ValAndOff(AuxInt).Val() to arg0+arg1*4+ValAndOff(AuxInt).Off()+aux, arg2=mem
		{name: "XORLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "XORL", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"}, // xor ValAndOff(AuxInt).Val() to arg0+arg1*4+ValAndOff(AuxInt).Off()+aux, arg2=mem

		// indexed loads/stores
		{name: "MOVBloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVBLZX", aux: "SymOff", symEffect: "Read"}, // load a byte from arg0+arg1+auxint+aux. arg2=mem
		{name: "MOVWloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVWLZX", aux: "SymOff", symEffect: "Read"}, // load 2 bytes from arg0+arg1+auxint+aux. arg2=mem
		{name: "MOVWloadidx2", argLength: 3, reg: gploadidx, asm: "MOVWLZX", aux: "SymOff", symEffect: "Read"},                    // load 2 bytes from arg0+2*arg1+auxint+aux. arg2=mem
		{name: "MOVLloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVL", aux: "SymOff", symEffect: "Read"},    // load 4 bytes from arg0+arg1+auxint+aux. arg2=mem
		{name: "MOVLloadidx4", argLength: 3, reg: gploadidx, asm: "MOVL", aux: "SymOff", symEffect: "Read"},                       // load 4 bytes from arg0+4*arg1+auxint+aux. arg2=mem
		// TODO: sign-extending indexed loads
		{name: "MOVBstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVB", aux: "SymOff", symEffect: "Write"}, // store byte in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVWstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVW", aux: "SymOff", symEffect: "Write"}, // store 2 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVWstoreidx2", argLength: 4, reg: gpstoreidx, asm: "MOVW", aux: "SymOff", symEffect: "Write"},                    // store 2 bytes in arg2 to arg0+2*arg1+auxint+aux. arg3=mem
		{name: "MOVLstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVL", aux: "SymOff", symEffect: "Write"}, // store 4 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVLstoreidx4", argLength: 4, reg: gpstoreidx, asm: "MOVL", aux: "SymOff", symEffect: "Write"},                    // store 4 bytes in arg2 to arg0+4*arg1+auxint+aux. arg3=mem
		// TODO: add size-mismatched indexed loads, like MOVBstoreidx4.

		// For storeconst ops, the AuxInt field encodes both
		// the value to store and an address offset of the store.
		// Cast AuxInt to a ValAndOff to extract Val and Off fields.
		{name: "MOVBstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVB", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store low byte of ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux.  arg1=mem
		{name: "MOVWstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVW", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store low 2 bytes of ...
		{name: "MOVLstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVL", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store low 4 bytes of ...

		{name: "MOVBstoreconstidx1", argLength: 3, reg: gpstoreconstidx, asm: "MOVB", aux: "SymValAndOff", typ: "Mem", symEffect: "Write"}, // store low byte of ValAndOff(AuxInt).Val() to arg0+1*arg1+ValAndOff(AuxInt).Off()+aux.  arg2=mem
		{name: "MOVWstoreconstidx1", argLength: 3, reg: gpstoreconstidx, asm: "MOVW", aux: "SymValAndOff", typ: "Mem", symEffect: "Write"}, // store low 2 bytes of ... arg1 ...
		{name: "MOVWstoreconstidx2", argLength: 3, reg: gpstoreconstidx, asm: "MOVW", aux: "SymValAndOff", typ: "Mem", symEffect: "Write"}, // store low 2 bytes of ... 2*arg1 ...
		{name: "MOVLstoreconstidx1", argLength: 3, reg: gpstoreconstidx, asm: "MOVL", aux: "SymValAndOff", typ: "Mem", symEffect: "Write"}, // store low 4 bytes of ... arg1 ...
		{name: "MOVLstoreconstidx4", argLength: 3, reg: gpstoreconstidx, asm: "MOVL", aux: "SymValAndOff", typ: "Mem", symEffect: "Write"}, // store low 4 bytes of ... 4*arg1 ...

		// arg0 = pointer to start of memory to zero
		// arg1 = value to store (will always be zero)
		// arg2 = mem
		// auxint = offset into duffzero code to start executing
		// returns mem
		{
			name:      "DUFFZERO",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("AX")},
				clobbers: buildReg("DI CX"),
				// Note: CX is only clobbered when dynamic linking.
			},
			faultOnNilArg0: true,
		},

		// arg0 = address of memory to zero
		// arg1 = # of 4-byte words to zero
		// arg2 = value to store (will always be zero)
		// arg3 = mem
		// returns mem
		{
			name:      "REPSTOSL",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("CX"), buildReg("AX")},
				clobbers: buildReg("DI CX"),
			},
			faultOnNilArg0: true,
		},

		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                              // call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                // tail call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{inputs: []regMask{gpsp, buildReg("DX"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                        // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		// arg0 = destination pointer
		// arg1 = source pointer
		// arg2 = mem
		// auxint = offset from duffcopy symbol to call
		// returns memory
		{
			name:      "DUFFCOPY",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("SI")},
				clobbers: buildReg("DI SI CX"), // uses CX as a temporary
			},
			clobberFlags:   true,
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// arg0 = destination pointer
		// arg1 = source pointer
		// arg2 = # of 8-byte words to copy
		// arg3 = mem
		// returns memory
		{
			name:      "REPMOVSL",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("SI"), buildReg("CX")},
				clobbers: buildReg("DI SI CX"),
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// (InvertFlags (CMPL a b)) == (CMPL b a)
		// So if we want (SETL (CMPL a b)) but we can't do that because a is a constant,
		// then we do (SETL (InvertFlags (CMPL b a))) instead.
		// Rewrites will convert this to (SETG (CMPL b a)).
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags", argLength: 1}, // reverse direction of arg0

		// Pseudo-ops
		{name: "LoweredGetG", argLength: 1, reg: gp01}, // arg0=mem
		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of DX (the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("DX")}}, zeroWidth: true},
		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" getcallerpc,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},
		// LoweredGetCallerSP returns the SP of the caller of the current function.
		{name: "LoweredGetCallerSP", reg: gp01, rematerializeable: true},
		//arg0=ptr,arg1=mem, returns void.  Faults if ptr is nil.
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gpsp}}, clobberFlags: true, nilCheck: true, faultOnNilArg0: true},

		// LoweredWB invokes runtime.gcWriteBarrier. arg0=destptr, arg1=srcptr, arg2=mem, aux=runtime.gcWriteBarrier
		// It saves all GP registers if necessary, but may clobber others.
		{name: "LoweredWB", argLength: 3, reg: regInfo{inputs: []regMask{buildReg("DI"), ax}, clobbers: callerSave &^ gp}, clobberFlags: true, aux: "Sym", symEffect: "None"},

		// There are three of these functions so that they can have three different register inputs.
		// When we check 0 <= c <= cap (A), then 0 <= b <= c (B), then 0 <= a <= b (C), we want the
		// default registers to match so we don't need to copy registers around unnecessarily.
		{name: "LoweredPanicBoundsA", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{dx, bx}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsB", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{cx, dx}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsC", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{ax, cx}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		// Extend ops are the same as Bounds ops except the indexes are 64-bit.
		{name: "LoweredPanicExtendA", argLength: 4, aux: "Int64", reg: regInfo{inputs: []regMask{si, dx, bx}}, typ: "Mem", call: true}, // arg0=idxHi, arg1=idxLo, arg2=len, arg3=mem, returns memory. AuxInt contains report code (see PanicExtend in genericOps.go).
		{name: "LoweredPanicExtendB", argLength: 4, aux: "Int64", reg: regInfo{inputs: []regMask{si, cx, dx}}, typ: "Mem", call: true}, // arg0=idxHi, arg1=idxLo, arg2=len, arg3=mem, returns memory. AuxInt contains report code (see PanicExtend in genericOps.go).
		{name: "LoweredPanicExtendC", argLength: 4, aux: "Int64", reg: regInfo{inputs: []regMask{si, ax, cx}}, typ: "Mem", call: true}, // arg0=idxHi, arg1=idxLo, arg2=len, arg3=mem, returns memory. AuxInt contains report code (see PanicExtend in genericOps.go).

		// Constant flag values. For any comparison, there are 5 possible
		// outcomes: the three from the signed total order (<,==,>) and the
		// three from the unsigned total order. The == cases overlap.
		// Note: there's a sixth "unordered" outcome for floating-point
		// comparisons, but we don't use such a beast yet.
		// These ops are for temporary use by rewrite rules. They
		// cannot appear in the generated assembly.
		{name: "FlagEQ"},     // equal
		{name: "FlagLT_ULT"}, // signed < and unsigned <
		{name: "FlagLT_UGT"}, // signed < and unsigned >
		{name: "FlagGT_UGT"}, // signed > and unsigned <
		{name: "FlagGT_ULT"}, // signed > and unsigned >

		// Special ops for PIC floating-point constants.
		// MOVSXconst1 loads the address of the constant-pool entry into a register.
		// MOVSXconst2 loads the constant from that address.
		// MOVSXconst1 returns a pointer, but we type it as uint32 because it can never point to the Go heap.
		{name: "MOVSSconst1", reg: gp01, typ: "UInt32", aux: "Float32"},
		{name: "MOVSDconst1", reg: gp01, typ: "UInt32", aux: "Float64"},
		{name: "MOVSSconst2", argLength: 1, reg: gpfp, asm: "MOVSS"},
		{name: "MOVSDconst2", argLength: 1, reg: gpfp, asm: "MOVSD"},
	}

	var _386blocks = []blockData{
		{name: "EQ", controls: 1},
		{name: "NE", controls: 1},
		{name: "LT", controls: 1},
		{name: "LE", controls: 1},
		{name: "GT", controls: 1},
		{name: "GE", controls: 1},
		{name: "OS", controls: 1},
		{name: "OC", controls: 1},
		{name: "ULT", controls: 1},
		{name: "ULE", controls: 1},
		{name: "UGT", controls: 1},
		{name: "UGE", controls: 1},
		{name: "EQF", controls: 1},
		{name: "NEF", controls: 1},
		{name: "ORD", controls: 1}, // FP, ordered comparison (parity zero)
		{name: "NAN", controls: 1}, // FP, unordered comparison (parity one)
	}

	archs = append(archs, arch{
		name:            "386",
		pkg:             "cmd/internal/obj/x86",
		genfile:         "../../x86/ssa.go",
		ops:             _386ops,
		blocks:          _386blocks,
		regnames:        regNames386,
		gpregmask:       gp,
		fpregmask:       fp,
		framepointerreg: int8(num["BP"]),
		linkreg:         -1, // not used
	})
}
