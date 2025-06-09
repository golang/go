// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
//  - All SymOff opcodes require their offset to fit in an int32.

// Suffixes encode the bit width of various instructions.
// Q (quad word) = 64 bit
// L (long word) = 32 bit
// W (word)      = 16 bit
// B (byte)      = 8 bit
// D (double)    = 64 bit float
// S (single)    = 32 bit float

// copied from ../../amd64/reg.go
var regNamesAMD64 = []string{
	"AX",
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",
	"R8",
	"R9",
	"R10",
	"R11",
	"R12",
	"R13",
	"g", // a.k.a. R14
	"R15",
	"X0",
	"X1",
	"X2",
	"X3",
	"X4",
	"X5",
	"X6",
	"X7",
	"X8",
	"X9",
	"X10",
	"X11",
	"X12",
	"X13",
	"X14",
	"X15", // constant 0 in ABIInternal

	// TODO: update asyncPreempt for K registers.
	// asyncPreempt also needs to store Z0-Z15 properly.
	"K0",
	"K1",
	"K2",
	"K3",
	"K4",
	"K5",
	"K6",
	"K7",
	// If you add registers, update asyncPreempt in runtime

	// pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesAMD64) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesAMD64 {
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
		gp         = buildReg("AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R15")
		g          = buildReg("g")
		fp         = buildReg("X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14")
		x15        = buildReg("X15")
		mask       = buildReg("K1 K2 K3 K4 K5 K6 K7")
		gpsp       = gp | buildReg("SP")
		gpspsb     = gpsp | buildReg("SB")
		gpspsbg    = gpspsb | g
		callerSave = gp | fp | g // runtime.setg (and anything calling it) may clobber g
	)
	// Common slices of register masks
	var (
		gponly   = []regMask{gp}
		fponly   = []regMask{fp}
		maskonly = []regMask{mask}
	)

	// Common regInfo
	var (
		gp01           = regInfo{inputs: nil, outputs: gponly}
		gp11           = regInfo{inputs: []regMask{gp}, outputs: gponly}
		gp11sp         = regInfo{inputs: []regMask{gpsp}, outputs: gponly}
		gp11sb         = regInfo{inputs: []regMask{gpspsbg}, outputs: gponly}
		gp21           = regInfo{inputs: []regMask{gp, gp}, outputs: gponly}
		gp21sp         = regInfo{inputs: []regMask{gpsp, gp}, outputs: gponly}
		gp21sb         = regInfo{inputs: []regMask{gpspsbg, gpsp}, outputs: gponly}
		gp21shift      = regInfo{inputs: []regMask{gp, cx}, outputs: []regMask{gp}}
		gp31shift      = regInfo{inputs: []regMask{gp, gp, cx}, outputs: []regMask{gp}}
		gp11div        = regInfo{inputs: []regMask{ax, gpsp &^ dx}, outputs: []regMask{ax, dx}}
		gp21hmul       = regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{dx}, clobbers: ax}
		gp21flags      = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp, 0}}
		gp2flags1flags = regInfo{inputs: []regMask{gp, gp, 0}, outputs: []regMask{gp, 0}}

		gp2flags     = regInfo{inputs: []regMask{gpsp, gpsp}}
		gp1flags     = regInfo{inputs: []regMask{gpsp}}
		gp0flagsLoad = regInfo{inputs: []regMask{gpspsbg, 0}}
		gp1flagsLoad = regInfo{inputs: []regMask{gpspsbg, gpsp, 0}}
		gp2flagsLoad = regInfo{inputs: []regMask{gpspsbg, gpsp, gpsp, 0}}
		flagsgp      = regInfo{inputs: nil, outputs: gponly}

		gp11flags      = regInfo{inputs: []regMask{gp}, outputs: []regMask{gp, 0}}
		gp1flags1flags = regInfo{inputs: []regMask{gp, 0}, outputs: []regMask{gp, 0}}

		readflags = regInfo{inputs: nil, outputs: gponly}

		gpload         = regInfo{inputs: []regMask{gpspsbg, 0}, outputs: gponly}
		gp21load       = regInfo{inputs: []regMask{gp, gpspsbg, 0}, outputs: gponly}
		gploadidx      = regInfo{inputs: []regMask{gpspsbg, gpsp, 0}, outputs: gponly}
		gp21loadidx    = regInfo{inputs: []regMask{gp, gpspsbg, gpsp, 0}, outputs: gponly}
		gp21shxload    = regInfo{inputs: []regMask{gpspsbg, gp, 0}, outputs: gponly}
		gp21shxloadidx = regInfo{inputs: []regMask{gpspsbg, gpsp, gp, 0}, outputs: gponly}

		gpstore         = regInfo{inputs: []regMask{gpspsbg, gpsp, 0}}
		gpstoreconst    = regInfo{inputs: []regMask{gpspsbg, 0}}
		gpstoreidx      = regInfo{inputs: []regMask{gpspsbg, gpsp, gpsp, 0}}
		gpstoreconstidx = regInfo{inputs: []regMask{gpspsbg, gpsp, 0}}
		gpstorexchg     = regInfo{inputs: []regMask{gp, gpspsbg, 0}, outputs: []regMask{gp}}
		cmpxchg         = regInfo{inputs: []regMask{gp, ax, gp, 0}, outputs: []regMask{gp, 0}, clobbers: ax}
		atomicLogic     = regInfo{inputs: []regMask{gp &^ ax, gp &^ ax, 0}, outputs: []regMask{ax, 0}}

		fp01        = regInfo{inputs: nil, outputs: fponly}
		fp21        = regInfo{inputs: []regMask{fp, fp}, outputs: fponly}
		fp31        = regInfo{inputs: []regMask{fp, fp, fp}, outputs: fponly}
		fp21load    = regInfo{inputs: []regMask{fp, gpspsbg, 0}, outputs: fponly}
		fp21loadidx = regInfo{inputs: []regMask{fp, gpspsbg, gpspsb, 0}, outputs: fponly}
		fpgp        = regInfo{inputs: fponly, outputs: gponly}
		gpfp        = regInfo{inputs: gponly, outputs: fponly}
		fp11        = regInfo{inputs: fponly, outputs: fponly}
		fp2flags    = regInfo{inputs: []regMask{fp, fp}}

		fpload    = regInfo{inputs: []regMask{gpspsb, 0}, outputs: fponly}
		fploadidx = regInfo{inputs: []regMask{gpspsb, gpsp, 0}, outputs: fponly}

		fpstore    = regInfo{inputs: []regMask{gpspsb, fp, 0}}
		fpstoreidx = regInfo{inputs: []regMask{gpspsb, gpsp, fp, 0}}

		fp1m1    = regInfo{inputs: fponly, outputs: maskonly}
		m1fp1    = regInfo{inputs: maskonly, outputs: fponly}
		fp2m1    = regInfo{inputs: []regMask{fp, fp}, outputs: maskonly}
		fp1m1fp1 = regInfo{inputs: []regMask{fp, mask}, outputs: fponly}
		fp2m1fp1 = regInfo{inputs: []regMask{fp, fp, mask}, outputs: fponly}
		fp2m1m1  = regInfo{inputs: []regMask{fp, fp, mask}, outputs: maskonly}
		fp3fp1   = regInfo{inputs: []regMask{fp, fp, fp}, outputs: fponly}
		fp3m1fp1 = regInfo{inputs: []regMask{fp, fp, fp, mask}, outputs: fponly}

		prefreg = regInfo{inputs: []regMask{gpspsbg}}
	)

	var AMD64ops = []opData{
		// {ADD,SUB,MUL,DIV}Sx: floating-point arithmetic
		// x==S for float32, x==D for float64
		// computes arg0 OP arg1
		{name: "ADDSS", argLength: 2, reg: fp21, asm: "ADDSS", commutative: true, resultInArg0: true},
		{name: "ADDSD", argLength: 2, reg: fp21, asm: "ADDSD", commutative: true, resultInArg0: true},
		{name: "SUBSS", argLength: 2, reg: fp21, asm: "SUBSS", resultInArg0: true},
		{name: "SUBSD", argLength: 2, reg: fp21, asm: "SUBSD", resultInArg0: true},
		{name: "MULSS", argLength: 2, reg: fp21, asm: "MULSS", commutative: true, resultInArg0: true},
		{name: "MULSD", argLength: 2, reg: fp21, asm: "MULSD", commutative: true, resultInArg0: true},
		{name: "DIVSS", argLength: 2, reg: fp21, asm: "DIVSS", resultInArg0: true},
		{name: "DIVSD", argLength: 2, reg: fp21, asm: "DIVSD", resultInArg0: true},

		// MOVSxload: floating-point loads
		// x==S for float32, x==D for float64
		// load from arg0+auxint+aux, arg1 = mem
		{name: "MOVSSload", argLength: 2, reg: fpload, asm: "MOVSS", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVSDload", argLength: 2, reg: fpload, asm: "MOVSD", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},

		// MOVSxconst: floatint-point constants
		// x==S for float32, x==D for float64
		{name: "MOVSSconst", reg: fp01, asm: "MOVSS", aux: "Float32", rematerializeable: true},
		{name: "MOVSDconst", reg: fp01, asm: "MOVSD", aux: "Float64", rematerializeable: true},

		// MOVSxloadidx: floating-point indexed loads
		// x==S for float32, x==D for float64
		// load from arg0 + scale*arg1+auxint+aux, arg2 = mem
		{name: "MOVSSloadidx1", argLength: 3, reg: fploadidx, asm: "MOVSS", scale: 1, aux: "SymOff", symEffect: "Read"},
		{name: "MOVSSloadidx4", argLength: 3, reg: fploadidx, asm: "MOVSS", scale: 4, aux: "SymOff", symEffect: "Read"},
		{name: "MOVSDloadidx1", argLength: 3, reg: fploadidx, asm: "MOVSD", scale: 1, aux: "SymOff", symEffect: "Read"},
		{name: "MOVSDloadidx8", argLength: 3, reg: fploadidx, asm: "MOVSD", scale: 8, aux: "SymOff", symEffect: "Read"},

		// MOVSxstore: floating-point stores
		// x==S for float32, x==D for float64
		// does *(arg0+auxint+aux) = arg1, arg2 = mem
		{name: "MOVSSstore", argLength: 3, reg: fpstore, asm: "MOVSS", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVSDstore", argLength: 3, reg: fpstore, asm: "MOVSD", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"},

		// MOVSxstoreidx: floating-point indexed stores
		// x==S for float32, x==D for float64
		// does *(arg0+scale*arg1+auxint+aux) = arg2, arg3 = mem
		{name: "MOVSSstoreidx1", argLength: 4, reg: fpstoreidx, asm: "MOVSS", scale: 1, aux: "SymOff", symEffect: "Write"},
		{name: "MOVSSstoreidx4", argLength: 4, reg: fpstoreidx, asm: "MOVSS", scale: 4, aux: "SymOff", symEffect: "Write"},
		{name: "MOVSDstoreidx1", argLength: 4, reg: fpstoreidx, asm: "MOVSD", scale: 1, aux: "SymOff", symEffect: "Write"},
		{name: "MOVSDstoreidx8", argLength: 4, reg: fpstoreidx, asm: "MOVSD", scale: 8, aux: "SymOff", symEffect: "Write"},

		// {ADD,SUB,MUL,DIV}Sxload: floating-point load / op combo
		// x==S for float32, x==D for float64
		// computes arg0 OP *(arg1+auxint+aux), arg2=mem
		{name: "ADDSSload", argLength: 3, reg: fp21load, asm: "ADDSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "ADDSDload", argLength: 3, reg: fp21load, asm: "ADDSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "SUBSSload", argLength: 3, reg: fp21load, asm: "SUBSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "SUBSDload", argLength: 3, reg: fp21load, asm: "SUBSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "MULSSload", argLength: 3, reg: fp21load, asm: "MULSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "MULSDload", argLength: 3, reg: fp21load, asm: "MULSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "DIVSSload", argLength: 3, reg: fp21load, asm: "DIVSS", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "DIVSDload", argLength: 3, reg: fp21load, asm: "DIVSD", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, symEffect: "Read"},

		// {ADD,SUB,MUL,DIV}Sxloadidx: floating-point indexed load / op combo
		// x==S for float32, x==D for float64
		// computes arg0 OP *(arg1+scale*arg2+auxint+aux), arg3=mem
		{name: "ADDSSloadidx1", argLength: 4, reg: fp21loadidx, asm: "ADDSS", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "ADDSSloadidx4", argLength: 4, reg: fp21loadidx, asm: "ADDSS", scale: 4, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "ADDSDloadidx1", argLength: 4, reg: fp21loadidx, asm: "ADDSD", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "ADDSDloadidx8", argLength: 4, reg: fp21loadidx, asm: "ADDSD", scale: 8, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "SUBSSloadidx1", argLength: 4, reg: fp21loadidx, asm: "SUBSS", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "SUBSSloadidx4", argLength: 4, reg: fp21loadidx, asm: "SUBSS", scale: 4, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "SUBSDloadidx1", argLength: 4, reg: fp21loadidx, asm: "SUBSD", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "SUBSDloadidx8", argLength: 4, reg: fp21loadidx, asm: "SUBSD", scale: 8, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "MULSSloadidx1", argLength: 4, reg: fp21loadidx, asm: "MULSS", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "MULSSloadidx4", argLength: 4, reg: fp21loadidx, asm: "MULSS", scale: 4, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "MULSDloadidx1", argLength: 4, reg: fp21loadidx, asm: "MULSD", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "MULSDloadidx8", argLength: 4, reg: fp21loadidx, asm: "MULSD", scale: 8, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "DIVSSloadidx1", argLength: 4, reg: fp21loadidx, asm: "DIVSS", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "DIVSSloadidx4", argLength: 4, reg: fp21loadidx, asm: "DIVSS", scale: 4, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "DIVSDloadidx1", argLength: 4, reg: fp21loadidx, asm: "DIVSD", scale: 1, aux: "SymOff", resultInArg0: true, symEffect: "Read"},
		{name: "DIVSDloadidx8", argLength: 4, reg: fp21loadidx, asm: "DIVSD", scale: 8, aux: "SymOff", resultInArg0: true, symEffect: "Read"},

		// {ADD,SUB,MUL,DIV,AND,OR,XOR}x: binary integer ops
		//   unadorned versions compute arg0 OP arg1
		//       const versions compute arg0 OP auxint (auxint is a sign-extended 32-bit value)
		// constmodify versions compute *(arg0+ValAndOff(AuxInt).Off().aux) OP= ValAndOff(AuxInt).Val(), arg1 = mem
		// x==L operations zero the upper 4 bytes of the destination register (not meaningful for constmodify versions).
		{name: "ADDQ", argLength: 2, reg: gp21sp, asm: "ADDQ", commutative: true, clobberFlags: true},
		{name: "ADDL", argLength: 2, reg: gp21sp, asm: "ADDL", commutative: true, clobberFlags: true},
		{name: "ADDQconst", argLength: 1, reg: gp11sp, asm: "ADDQ", aux: "Int32", typ: "UInt64", clobberFlags: true},
		{name: "ADDLconst", argLength: 1, reg: gp11sp, asm: "ADDL", aux: "Int32", clobberFlags: true},
		{name: "ADDQconstmodify", argLength: 2, reg: gpstoreconst, asm: "ADDQ", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "ADDLconstmodify", argLength: 2, reg: gpstoreconst, asm: "ADDL", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},

		{name: "SUBQ", argLength: 2, reg: gp21, asm: "SUBQ", resultInArg0: true, clobberFlags: true},
		{name: "SUBL", argLength: 2, reg: gp21, asm: "SUBL", resultInArg0: true, clobberFlags: true},
		{name: "SUBQconst", argLength: 1, reg: gp11, asm: "SUBQ", aux: "Int32", resultInArg0: true, clobberFlags: true},
		{name: "SUBLconst", argLength: 1, reg: gp11, asm: "SUBL", aux: "Int32", resultInArg0: true, clobberFlags: true},

		{name: "MULQ", argLength: 2, reg: gp21, asm: "IMULQ", commutative: true, resultInArg0: true, clobberFlags: true},
		{name: "MULL", argLength: 2, reg: gp21, asm: "IMULL", commutative: true, resultInArg0: true, clobberFlags: true},
		{name: "MULQconst", argLength: 1, reg: gp11, asm: "IMUL3Q", aux: "Int32", clobberFlags: true},
		{name: "MULLconst", argLength: 1, reg: gp11, asm: "IMUL3L", aux: "Int32", clobberFlags: true},

		// Let x = arg0*arg1 (full 32x32->64  unsigned multiply). Returns uint32(x), and flags set to overflow if uint32(x) != x.
		{name: "MULLU", argLength: 2, reg: regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{ax, 0}, clobbers: dx}, typ: "(UInt32,Flags)", asm: "MULL", commutative: true, clobberFlags: true},
		// Let x = arg0*arg1 (full 64x64->128 unsigned multiply). Returns uint64(x), and flags set to overflow if uint64(x) != x.
		{name: "MULQU", argLength: 2, reg: regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{ax, 0}, clobbers: dx}, typ: "(UInt64,Flags)", asm: "MULQ", commutative: true, clobberFlags: true},

		// HMULx[U]: computes the high bits of an integer multiply.
		// computes arg0 * arg1 >> (x==L?32:64)
		// The multiply is unsigned for the U versions, signed for the non-U versions.
		// HMULx[U] are intentionally not marked as commutative, even though they are.
		// This is because they have asymmetric register requirements.
		// There are rewrite rules to try to place arguments in preferable slots.
		{name: "HMULQ", argLength: 2, reg: gp21hmul, asm: "IMULQ", clobberFlags: true},
		{name: "HMULL", argLength: 2, reg: gp21hmul, asm: "IMULL", clobberFlags: true},
		{name: "HMULQU", argLength: 2, reg: gp21hmul, asm: "MULQ", clobberFlags: true},
		{name: "HMULLU", argLength: 2, reg: gp21hmul, asm: "MULL", clobberFlags: true},

		// (arg0 + arg1) / 2 as unsigned, all 64 result bits
		{name: "AVGQU", argLength: 2, reg: gp21, commutative: true, resultInArg0: true, clobberFlags: true},

		// DIVx[U] computes [arg0 / arg1, arg0 % arg1]
		// For signed versions, AuxInt non-zero means that the divisor has been proved to be not -1.
		{name: "DIVQ", argLength: 2, reg: gp11div, typ: "(Int64,Int64)", asm: "IDIVQ", aux: "Bool", clobberFlags: true},
		{name: "DIVL", argLength: 2, reg: gp11div, typ: "(Int32,Int32)", asm: "IDIVL", aux: "Bool", clobberFlags: true},
		{name: "DIVW", argLength: 2, reg: gp11div, typ: "(Int16,Int16)", asm: "IDIVW", aux: "Bool", clobberFlags: true},
		{name: "DIVQU", argLength: 2, reg: gp11div, typ: "(UInt64,UInt64)", asm: "DIVQ", clobberFlags: true},
		{name: "DIVLU", argLength: 2, reg: gp11div, typ: "(UInt32,UInt32)", asm: "DIVL", clobberFlags: true},
		{name: "DIVWU", argLength: 2, reg: gp11div, typ: "(UInt16,UInt16)", asm: "DIVW", clobberFlags: true},

		// computes -arg0, flags set for 0-arg0.
		{name: "NEGLflags", argLength: 1, reg: gp11flags, typ: "(UInt32,Flags)", asm: "NEGL", resultInArg0: true},
		// compute arg0+auxint. flags set for arg0+auxint.
		// NOTE: we pretend the CF/OF flags are undefined for these instructions,
		// so we can use INC/DEC instead of ADDQconst if auxint is +/-1. (INC/DEC don't modify CF.)
		{name: "ADDQconstflags", argLength: 1, reg: gp11flags, aux: "Int32", asm: "ADDQ", resultInArg0: true},
		{name: "ADDLconstflags", argLength: 1, reg: gp11flags, aux: "Int32", asm: "ADDL", resultInArg0: true},

		// The following 4 add opcodes return the low 64 bits of the sum in the first result and
		// the carry (the 65th bit) in the carry flag.
		{name: "ADDQcarry", argLength: 2, reg: gp21flags, typ: "(UInt64,Flags)", asm: "ADDQ", commutative: true, resultInArg0: true}, // r = arg0+arg1
		{name: "ADCQ", argLength: 3, reg: gp2flags1flags, typ: "(UInt64,Flags)", asm: "ADCQ", commutative: true, resultInArg0: true}, // r = arg0+arg1+carry(arg2)
		{name: "ADDQconstcarry", argLength: 1, reg: gp11flags, typ: "(UInt64,Flags)", asm: "ADDQ", aux: "Int32", resultInArg0: true}, // r = arg0+auxint
		{name: "ADCQconst", argLength: 2, reg: gp1flags1flags, typ: "(UInt64,Flags)", asm: "ADCQ", aux: "Int32", resultInArg0: true}, // r = arg0+auxint+carry(arg1)

		// The following 4 add opcodes return the low 64 bits of the difference in the first result and
		// the borrow (if the result is negative) in the carry flag.
		{name: "SUBQborrow", argLength: 2, reg: gp21flags, typ: "(UInt64,Flags)", asm: "SUBQ", resultInArg0: true},                    // r = arg0-arg1
		{name: "SBBQ", argLength: 3, reg: gp2flags1flags, typ: "(UInt64,Flags)", asm: "SBBQ", resultInArg0: true},                     // r = arg0-(arg1+carry(arg2))
		{name: "SUBQconstborrow", argLength: 1, reg: gp11flags, typ: "(UInt64,Flags)", asm: "SUBQ", aux: "Int32", resultInArg0: true}, // r = arg0-auxint
		{name: "SBBQconst", argLength: 2, reg: gp1flags1flags, typ: "(UInt64,Flags)", asm: "SBBQ", aux: "Int32", resultInArg0: true},  // r = arg0-(auxint+carry(arg1))

		{name: "MULQU2", argLength: 2, reg: regInfo{inputs: []regMask{ax, gpsp}, outputs: []regMask{dx, ax}}, commutative: true, asm: "MULQ", clobberFlags: true}, // arg0 * arg1, returns (hi, lo)
		{name: "DIVQU2", argLength: 3, reg: regInfo{inputs: []regMask{dx, ax, gpsp}, outputs: []regMask{ax, dx}}, asm: "DIVQ", clobberFlags: true},                // arg0:arg1 / arg2 (128-bit divided by 64-bit), returns (q, r)

		{name: "ANDQ", argLength: 2, reg: gp21, asm: "ANDQ", commutative: true, resultInArg0: true, clobberFlags: true},                                                 // arg0 & arg1
		{name: "ANDL", argLength: 2, reg: gp21, asm: "ANDL", commutative: true, resultInArg0: true, clobberFlags: true},                                                 // arg0 & arg1
		{name: "ANDQconst", argLength: 1, reg: gp11, asm: "ANDQ", aux: "Int32", resultInArg0: true, clobberFlags: true},                                                 // arg0 & auxint
		{name: "ANDLconst", argLength: 1, reg: gp11, asm: "ANDL", aux: "Int32", resultInArg0: true, clobberFlags: true},                                                 // arg0 & auxint
		{name: "ANDQconstmodify", argLength: 2, reg: gpstoreconst, asm: "ANDQ", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // and ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem
		{name: "ANDLconstmodify", argLength: 2, reg: gpstoreconst, asm: "ANDL", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // and ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem

		{name: "ORQ", argLength: 2, reg: gp21, asm: "ORQ", commutative: true, resultInArg0: true, clobberFlags: true},                                                 // arg0 | arg1
		{name: "ORL", argLength: 2, reg: gp21, asm: "ORL", commutative: true, resultInArg0: true, clobberFlags: true},                                                 // arg0 | arg1
		{name: "ORQconst", argLength: 1, reg: gp11, asm: "ORQ", aux: "Int32", resultInArg0: true, clobberFlags: true},                                                 // arg0 | auxint
		{name: "ORLconst", argLength: 1, reg: gp11, asm: "ORL", aux: "Int32", resultInArg0: true, clobberFlags: true},                                                 // arg0 | auxint
		{name: "ORQconstmodify", argLength: 2, reg: gpstoreconst, asm: "ORQ", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // or ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem
		{name: "ORLconstmodify", argLength: 2, reg: gpstoreconst, asm: "ORL", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // or ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem

		{name: "XORQ", argLength: 2, reg: gp21, asm: "XORQ", commutative: true, resultInArg0: true, clobberFlags: true},                                                 // arg0 ^ arg1
		{name: "XORL", argLength: 2, reg: gp21, asm: "XORL", commutative: true, resultInArg0: true, clobberFlags: true},                                                 // arg0 ^ arg1
		{name: "XORQconst", argLength: 1, reg: gp11, asm: "XORQ", aux: "Int32", resultInArg0: true, clobberFlags: true},                                                 // arg0 ^ auxint
		{name: "XORLconst", argLength: 1, reg: gp11, asm: "XORL", aux: "Int32", resultInArg0: true, clobberFlags: true},                                                 // arg0 ^ auxint
		{name: "XORQconstmodify", argLength: 2, reg: gpstoreconst, asm: "XORQ", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // xor ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem
		{name: "XORLconstmodify", argLength: 2, reg: gpstoreconst, asm: "XORL", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"}, // xor ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux, arg1=mem

		// CMPx: compare arg0 to arg1.
		{name: "CMPQ", argLength: 2, reg: gp2flags, asm: "CMPQ", typ: "Flags"},
		{name: "CMPL", argLength: 2, reg: gp2flags, asm: "CMPL", typ: "Flags"},
		{name: "CMPW", argLength: 2, reg: gp2flags, asm: "CMPW", typ: "Flags"},
		{name: "CMPB", argLength: 2, reg: gp2flags, asm: "CMPB", typ: "Flags"},

		// CMPxconst: compare arg0 to auxint.
		{name: "CMPQconst", argLength: 1, reg: gp1flags, asm: "CMPQ", typ: "Flags", aux: "Int32"},
		{name: "CMPLconst", argLength: 1, reg: gp1flags, asm: "CMPL", typ: "Flags", aux: "Int32"},
		{name: "CMPWconst", argLength: 1, reg: gp1flags, asm: "CMPW", typ: "Flags", aux: "Int16"},
		{name: "CMPBconst", argLength: 1, reg: gp1flags, asm: "CMPB", typ: "Flags", aux: "Int8"},

		// CMPxload: compare *(arg0+auxint+aux) to arg1 (in that order). arg2=mem.
		{name: "CMPQload", argLength: 3, reg: gp1flagsLoad, asm: "CMPQ", aux: "SymOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPLload", argLength: 3, reg: gp1flagsLoad, asm: "CMPL", aux: "SymOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPWload", argLength: 3, reg: gp1flagsLoad, asm: "CMPW", aux: "SymOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPBload", argLength: 3, reg: gp1flagsLoad, asm: "CMPB", aux: "SymOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},

		// CMPxconstload: compare *(arg0+ValAndOff(AuxInt).Off()+aux) to ValAndOff(AuxInt).Val() (in that order). arg1=mem.
		{name: "CMPQconstload", argLength: 2, reg: gp0flagsLoad, asm: "CMPQ", aux: "SymValAndOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPLconstload", argLength: 2, reg: gp0flagsLoad, asm: "CMPL", aux: "SymValAndOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPWconstload", argLength: 2, reg: gp0flagsLoad, asm: "CMPW", aux: "SymValAndOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},
		{name: "CMPBconstload", argLength: 2, reg: gp0flagsLoad, asm: "CMPB", aux: "SymValAndOff", typ: "Flags", symEffect: "Read", faultOnNilArg0: true},

		// CMPxloadidx: compare *(arg0+N*arg1+auxint+aux) to arg2 (in that order). arg3=mem.
		{name: "CMPQloadidx8", argLength: 4, reg: gp2flagsLoad, asm: "CMPQ", scale: 8, aux: "SymOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPQloadidx1", argLength: 4, reg: gp2flagsLoad, asm: "CMPQ", scale: 1, commutative: true, aux: "SymOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPLloadidx4", argLength: 4, reg: gp2flagsLoad, asm: "CMPL", scale: 4, aux: "SymOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPLloadidx1", argLength: 4, reg: gp2flagsLoad, asm: "CMPL", scale: 1, commutative: true, aux: "SymOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPWloadidx2", argLength: 4, reg: gp2flagsLoad, asm: "CMPW", scale: 2, aux: "SymOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPWloadidx1", argLength: 4, reg: gp2flagsLoad, asm: "CMPW", scale: 1, commutative: true, aux: "SymOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPBloadidx1", argLength: 4, reg: gp2flagsLoad, asm: "CMPB", scale: 1, commutative: true, aux: "SymOff", typ: "Flags", symEffect: "Read"},

		// CMPxconstloadidx: compare *(arg0+N*arg1+ValAndOff(AuxInt).Off()+aux) to ValAndOff(AuxInt).Val() (in that order). arg2=mem.
		{name: "CMPQconstloadidx8", argLength: 3, reg: gp1flagsLoad, asm: "CMPQ", scale: 8, aux: "SymValAndOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPQconstloadidx1", argLength: 3, reg: gp1flagsLoad, asm: "CMPQ", scale: 1, commutative: true, aux: "SymValAndOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPLconstloadidx4", argLength: 3, reg: gp1flagsLoad, asm: "CMPL", scale: 4, aux: "SymValAndOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPLconstloadidx1", argLength: 3, reg: gp1flagsLoad, asm: "CMPL", scale: 1, commutative: true, aux: "SymValAndOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPWconstloadidx2", argLength: 3, reg: gp1flagsLoad, asm: "CMPW", scale: 2, aux: "SymValAndOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPWconstloadidx1", argLength: 3, reg: gp1flagsLoad, asm: "CMPW", scale: 1, commutative: true, aux: "SymValAndOff", typ: "Flags", symEffect: "Read"},
		{name: "CMPBconstloadidx1", argLength: 3, reg: gp1flagsLoad, asm: "CMPB", scale: 1, commutative: true, aux: "SymValAndOff", typ: "Flags", symEffect: "Read"},

		// UCOMISx: floating-point compare arg0 to arg1
		// x==S for float32, x==D for float64
		{name: "UCOMISS", argLength: 2, reg: fp2flags, asm: "UCOMISS", typ: "Flags"},
		{name: "UCOMISD", argLength: 2, reg: fp2flags, asm: "UCOMISD", typ: "Flags"},

		// bit test/set/clear operations
		{name: "BTL", argLength: 2, reg: gp2flags, asm: "BTL", typ: "Flags"},                                           // test whether bit arg0%32 in arg1 is set
		{name: "BTQ", argLength: 2, reg: gp2flags, asm: "BTQ", typ: "Flags"},                                           // test whether bit arg0%64 in arg1 is set
		{name: "BTCL", argLength: 2, reg: gp21, asm: "BTCL", resultInArg0: true, clobberFlags: true},                   // complement bit arg1%32 in arg0
		{name: "BTCQ", argLength: 2, reg: gp21, asm: "BTCQ", resultInArg0: true, clobberFlags: true},                   // complement bit arg1%64 in arg0
		{name: "BTRL", argLength: 2, reg: gp21, asm: "BTRL", resultInArg0: true, clobberFlags: true},                   // reset bit arg1%32 in arg0
		{name: "BTRQ", argLength: 2, reg: gp21, asm: "BTRQ", resultInArg0: true, clobberFlags: true},                   // reset bit arg1%64 in arg0
		{name: "BTSL", argLength: 2, reg: gp21, asm: "BTSL", resultInArg0: true, clobberFlags: true},                   // set bit arg1%32 in arg0
		{name: "BTSQ", argLength: 2, reg: gp21, asm: "BTSQ", resultInArg0: true, clobberFlags: true},                   // set bit arg1%64 in arg0
		{name: "BTLconst", argLength: 1, reg: gp1flags, asm: "BTL", typ: "Flags", aux: "Int8"},                         // test whether bit auxint in arg0 is set, 0 <= auxint < 32
		{name: "BTQconst", argLength: 1, reg: gp1flags, asm: "BTQ", typ: "Flags", aux: "Int8"},                         // test whether bit auxint in arg0 is set, 0 <= auxint < 64
		{name: "BTCQconst", argLength: 1, reg: gp11, asm: "BTCQ", resultInArg0: true, clobberFlags: true, aux: "Int8"}, // complement bit auxint in arg0, 31 <= auxint < 64
		{name: "BTRQconst", argLength: 1, reg: gp11, asm: "BTRQ", resultInArg0: true, clobberFlags: true, aux: "Int8"}, // reset bit auxint in arg0, 31 <= auxint < 64
		{name: "BTSQconst", argLength: 1, reg: gp11, asm: "BTSQ", resultInArg0: true, clobberFlags: true, aux: "Int8"}, // set bit auxint in arg0, 31 <= auxint < 64

		// BT[SRC]Qconstmodify
		//
		//  S: set bit
		//  R: reset (clear) bit
		//  C: complement bit
		//
		// Apply operation to bit ValAndOff(AuxInt).Val() in the 64 bits at
		// memory address arg0+ValAndOff(AuxInt).Off()+aux
		// Bit index must be in range (31-63).
		// (We use OR/AND/XOR for thinner targets and lower bit indexes.)
		// arg1=mem, returns mem
		//
		// Note that there aren't non-const versions of these instructions.
		// Well, there are such instructions, but they are slow and weird so we don't use them.
		{name: "BTSQconstmodify", argLength: 2, reg: gpstoreconst, asm: "BTSQ", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "BTRQconstmodify", argLength: 2, reg: gpstoreconst, asm: "BTRQ", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "BTCQconstmodify", argLength: 2, reg: gpstoreconst, asm: "BTCQ", aux: "SymValAndOff", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},

		// TESTx: compare (arg0 & arg1) to 0
		{name: "TESTQ", argLength: 2, reg: gp2flags, commutative: true, asm: "TESTQ", typ: "Flags"},
		{name: "TESTL", argLength: 2, reg: gp2flags, commutative: true, asm: "TESTL", typ: "Flags"},
		{name: "TESTW", argLength: 2, reg: gp2flags, commutative: true, asm: "TESTW", typ: "Flags"},
		{name: "TESTB", argLength: 2, reg: gp2flags, commutative: true, asm: "TESTB", typ: "Flags"},

		// TESTxconst: compare (arg0 & auxint) to 0
		{name: "TESTQconst", argLength: 1, reg: gp1flags, asm: "TESTQ", typ: "Flags", aux: "Int32"},
		{name: "TESTLconst", argLength: 1, reg: gp1flags, asm: "TESTL", typ: "Flags", aux: "Int32"},
		{name: "TESTWconst", argLength: 1, reg: gp1flags, asm: "TESTW", typ: "Flags", aux: "Int16"},
		{name: "TESTBconst", argLength: 1, reg: gp1flags, asm: "TESTB", typ: "Flags", aux: "Int8"},

		// S{HL, HR, AR}x: shift operations
		// SHL: shift left
		// SHR: shift right logical (0s are shifted in from beyond the word size)
		// SAR: shift right arithmetic (sign bit is shifted in from beyond the word size)
		// arg0 is the value being shifted
		// arg1 is the amount to shift, interpreted mod (Q=64,L=32,W=32,B=32)
		// (Note: x86 is weird, the 16 and 8 byte shifts still use all 5 bits of shift amount!)
		// For *const versions, use auxint instead of arg1 as the shift amount. auxint must be in the range 0 to (Q=63,L=31,W=15,B=7) inclusive.
		{name: "SHLQ", argLength: 2, reg: gp21shift, asm: "SHLQ", resultInArg0: true, clobberFlags: true},
		{name: "SHLL", argLength: 2, reg: gp21shift, asm: "SHLL", resultInArg0: true, clobberFlags: true},
		{name: "SHLQconst", argLength: 1, reg: gp11, asm: "SHLQ", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "SHLLconst", argLength: 1, reg: gp11, asm: "SHLL", aux: "Int8", resultInArg0: true, clobberFlags: true},

		{name: "SHRQ", argLength: 2, reg: gp21shift, asm: "SHRQ", resultInArg0: true, clobberFlags: true},
		{name: "SHRL", argLength: 2, reg: gp21shift, asm: "SHRL", resultInArg0: true, clobberFlags: true},
		{name: "SHRW", argLength: 2, reg: gp21shift, asm: "SHRW", resultInArg0: true, clobberFlags: true},
		{name: "SHRB", argLength: 2, reg: gp21shift, asm: "SHRB", resultInArg0: true, clobberFlags: true},
		{name: "SHRQconst", argLength: 1, reg: gp11, asm: "SHRQ", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "SHRLconst", argLength: 1, reg: gp11, asm: "SHRL", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "SHRWconst", argLength: 1, reg: gp11, asm: "SHRW", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "SHRBconst", argLength: 1, reg: gp11, asm: "SHRB", aux: "Int8", resultInArg0: true, clobberFlags: true},

		{name: "SARQ", argLength: 2, reg: gp21shift, asm: "SARQ", resultInArg0: true, clobberFlags: true},
		{name: "SARL", argLength: 2, reg: gp21shift, asm: "SARL", resultInArg0: true, clobberFlags: true},
		{name: "SARW", argLength: 2, reg: gp21shift, asm: "SARW", resultInArg0: true, clobberFlags: true},
		{name: "SARB", argLength: 2, reg: gp21shift, asm: "SARB", resultInArg0: true, clobberFlags: true},
		{name: "SARQconst", argLength: 1, reg: gp11, asm: "SARQ", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "SARLconst", argLength: 1, reg: gp11, asm: "SARL", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "SARWconst", argLength: 1, reg: gp11, asm: "SARW", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "SARBconst", argLength: 1, reg: gp11, asm: "SARB", aux: "Int8", resultInArg0: true, clobberFlags: true},

		// unsigned arg0 >> arg2, shifting in bits from arg1 (==(arg1<<64+arg0)>>arg2, keeping low 64 bits), shift amount is mod 64
		{name: "SHRDQ", argLength: 3, reg: gp31shift, asm: "SHRQ", resultInArg0: true, clobberFlags: true},
		// unsigned arg0 << arg2, shifting in bits from arg1 (==(arg0<<64+arg1)<<arg2, keeping high 64 bits), shift amount is mod 64
		{name: "SHLDQ", argLength: 3, reg: gp31shift, asm: "SHLQ", resultInArg0: true, clobberFlags: true},

		// RO{L,R}x: rotate instructions
		// computes arg0 rotate (L=left,R=right) arg1 bits.
		// Bits are rotated within the low (Q=64,L=32,W=16,B=8) bits of the register.
		// For *const versions use auxint instead of arg1 as the rotate amount. auxint must be in the range 0 to (Q=63,L=31,W=15,B=7) inclusive.
		// x==L versions zero the upper 32 bits of the destination register.
		// x==W and x==B versions leave the upper bits unspecified.
		{name: "ROLQ", argLength: 2, reg: gp21shift, asm: "ROLQ", resultInArg0: true, clobberFlags: true},
		{name: "ROLL", argLength: 2, reg: gp21shift, asm: "ROLL", resultInArg0: true, clobberFlags: true},
		{name: "ROLW", argLength: 2, reg: gp21shift, asm: "ROLW", resultInArg0: true, clobberFlags: true},
		{name: "ROLB", argLength: 2, reg: gp21shift, asm: "ROLB", resultInArg0: true, clobberFlags: true},
		{name: "RORQ", argLength: 2, reg: gp21shift, asm: "RORQ", resultInArg0: true, clobberFlags: true},
		{name: "RORL", argLength: 2, reg: gp21shift, asm: "RORL", resultInArg0: true, clobberFlags: true},
		{name: "RORW", argLength: 2, reg: gp21shift, asm: "RORW", resultInArg0: true, clobberFlags: true},
		{name: "RORB", argLength: 2, reg: gp21shift, asm: "RORB", resultInArg0: true, clobberFlags: true},
		{name: "ROLQconst", argLength: 1, reg: gp11, asm: "ROLQ", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "ROLLconst", argLength: 1, reg: gp11, asm: "ROLL", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "ROLWconst", argLength: 1, reg: gp11, asm: "ROLW", aux: "Int8", resultInArg0: true, clobberFlags: true},
		{name: "ROLBconst", argLength: 1, reg: gp11, asm: "ROLB", aux: "Int8", resultInArg0: true, clobberFlags: true},

		// [ADD,SUB,AND,OR]xload: integer load/op combo
		// L = int32, Q = int64
		// x==L operations zero the upper 4 bytes of the destination register.
		// computes arg0 op *(arg1+auxint+aux), arg2=mem
		{name: "ADDLload", argLength: 3, reg: gp21load, asm: "ADDL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "ADDQload", argLength: 3, reg: gp21load, asm: "ADDQ", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "SUBQload", argLength: 3, reg: gp21load, asm: "SUBQ", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "SUBLload", argLength: 3, reg: gp21load, asm: "SUBL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "ANDLload", argLength: 3, reg: gp21load, asm: "ANDL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "ANDQload", argLength: 3, reg: gp21load, asm: "ANDQ", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "ORQload", argLength: 3, reg: gp21load, asm: "ORQ", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "ORLload", argLength: 3, reg: gp21load, asm: "ORL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "XORQload", argLength: 3, reg: gp21load, asm: "XORQ", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},
		{name: "XORLload", argLength: 3, reg: gp21load, asm: "XORL", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},

		// integer indexed load/op combo
		// L = int32, Q = int64
		// L operations zero the upper 4 bytes of the destination register.
		// computes arg0 op *(arg1+scale*arg2+auxint+aux), arg3=mem
		{name: "ADDLloadidx1", argLength: 4, reg: gp21loadidx, asm: "ADDL", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ADDLloadidx4", argLength: 4, reg: gp21loadidx, asm: "ADDL", scale: 4, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ADDLloadidx8", argLength: 4, reg: gp21loadidx, asm: "ADDL", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ADDQloadidx1", argLength: 4, reg: gp21loadidx, asm: "ADDQ", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ADDQloadidx8", argLength: 4, reg: gp21loadidx, asm: "ADDQ", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "SUBLloadidx1", argLength: 4, reg: gp21loadidx, asm: "SUBL", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "SUBLloadidx4", argLength: 4, reg: gp21loadidx, asm: "SUBL", scale: 4, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "SUBLloadidx8", argLength: 4, reg: gp21loadidx, asm: "SUBL", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "SUBQloadidx1", argLength: 4, reg: gp21loadidx, asm: "SUBQ", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "SUBQloadidx8", argLength: 4, reg: gp21loadidx, asm: "SUBQ", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ANDLloadidx1", argLength: 4, reg: gp21loadidx, asm: "ANDL", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ANDLloadidx4", argLength: 4, reg: gp21loadidx, asm: "ANDL", scale: 4, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ANDLloadidx8", argLength: 4, reg: gp21loadidx, asm: "ANDL", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ANDQloadidx1", argLength: 4, reg: gp21loadidx, asm: "ANDQ", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ANDQloadidx8", argLength: 4, reg: gp21loadidx, asm: "ANDQ", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ORLloadidx1", argLength: 4, reg: gp21loadidx, asm: "ORL", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ORLloadidx4", argLength: 4, reg: gp21loadidx, asm: "ORL", scale: 4, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ORLloadidx8", argLength: 4, reg: gp21loadidx, asm: "ORL", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ORQloadidx1", argLength: 4, reg: gp21loadidx, asm: "ORQ", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "ORQloadidx8", argLength: 4, reg: gp21loadidx, asm: "ORQ", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "XORLloadidx1", argLength: 4, reg: gp21loadidx, asm: "XORL", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "XORLloadidx4", argLength: 4, reg: gp21loadidx, asm: "XORL", scale: 4, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "XORLloadidx8", argLength: 4, reg: gp21loadidx, asm: "XORL", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "XORQloadidx1", argLength: 4, reg: gp21loadidx, asm: "XORQ", scale: 1, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},
		{name: "XORQloadidx8", argLength: 4, reg: gp21loadidx, asm: "XORQ", scale: 8, aux: "SymOff", resultInArg0: true, clobberFlags: true, symEffect: "Read"},

		// direct binary op on memory (read-modify-write)
		// L = int32, Q = int64
		// does *(arg0+auxint+aux) op= arg1, arg2=mem
		{name: "ADDQmodify", argLength: 3, reg: gpstore, asm: "ADDQ", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "SUBQmodify", argLength: 3, reg: gpstore, asm: "SUBQ", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "ANDQmodify", argLength: 3, reg: gpstore, asm: "ANDQ", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "ORQmodify", argLength: 3, reg: gpstore, asm: "ORQ", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "XORQmodify", argLength: 3, reg: gpstore, asm: "XORQ", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "ADDLmodify", argLength: 3, reg: gpstore, asm: "ADDL", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "SUBLmodify", argLength: 3, reg: gpstore, asm: "SUBL", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "ANDLmodify", argLength: 3, reg: gpstore, asm: "ANDL", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "ORLmodify", argLength: 3, reg: gpstore, asm: "ORL", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},
		{name: "XORLmodify", argLength: 3, reg: gpstore, asm: "XORL", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Read,Write"},

		// indexed direct binary op on memory.
		// does *(arg0+scale*arg1+auxint+aux) op= arg2, arg3=mem
		{name: "ADDQmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "ADDQ", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDQmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "ADDQ", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "SUBQmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "SUBQ", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "SUBQmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "SUBQ", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDQmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "ANDQ", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDQmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "ANDQ", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORQmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "ORQ", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORQmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "ORQ", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORQmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "XORQ", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORQmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "XORQ", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDLmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "ADDL", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "ADDL", scale: 4, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDLmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "ADDL", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "SUBLmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "SUBL", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "SUBLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "SUBL", scale: 4, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "SUBLmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "SUBL", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDLmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "ANDL", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "ANDL", scale: 4, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDLmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "ANDL", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORLmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "ORL", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "ORL", scale: 4, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORLmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "ORL", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORLmodifyidx1", argLength: 4, reg: gpstoreidx, asm: "XORL", scale: 1, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORLmodifyidx4", argLength: 4, reg: gpstoreidx, asm: "XORL", scale: 4, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORLmodifyidx8", argLength: 4, reg: gpstoreidx, asm: "XORL", scale: 8, aux: "SymOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},

		// indexed direct binary op on memory with constant argument.
		// does *(arg0+scale*arg1+ValAndOff(AuxInt).Off()+aux) op= ValAndOff(AuxInt).Val(), arg2=mem
		{name: "ADDQconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "ADDQ", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDQconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "ADDQ", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDQconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "ANDQ", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDQconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "ANDQ", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORQconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "ORQ", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORQconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "ORQ", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORQconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "XORQ", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORQconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "XORQ", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDLconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "ADDL", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "ADDL", scale: 4, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ADDLconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "ADDL", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDLconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "ANDL", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "ANDL", scale: 4, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ANDLconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "ANDL", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORLconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "ORL", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "ORL", scale: 4, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "ORLconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "ORL", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORLconstmodifyidx1", argLength: 3, reg: gpstoreconstidx, asm: "XORL", scale: 1, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORLconstmodifyidx4", argLength: 3, reg: gpstoreconstidx, asm: "XORL", scale: 4, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},
		{name: "XORLconstmodifyidx8", argLength: 3, reg: gpstoreconstidx, asm: "XORL", scale: 8, aux: "SymValAndOff", typ: "Mem", clobberFlags: true, symEffect: "Read,Write"},

		// {NEG,NOT}x: unary ops
		// computes [NEG:-,NOT:^]arg0
		// L = int32, Q = int64
		// L operations zero the upper 4 bytes of the destination register.
		{name: "NEGQ", argLength: 1, reg: gp11, asm: "NEGQ", resultInArg0: true, clobberFlags: true},
		{name: "NEGL", argLength: 1, reg: gp11, asm: "NEGL", resultInArg0: true, clobberFlags: true},
		{name: "NOTQ", argLength: 1, reg: gp11, asm: "NOTQ", resultInArg0: true},
		{name: "NOTL", argLength: 1, reg: gp11, asm: "NOTL", resultInArg0: true},

		// BS{F,R}Q returns a tuple [result, flags]
		// result is undefined if the input is zero.
		// flags are set to "equal" if the input is zero, "not equal" otherwise.
		// BS{F,R}L returns only the result.
		{name: "BSFQ", argLength: 1, reg: gp11flags, asm: "BSFQ", typ: "(UInt64,Flags)"},        // # of low-order zeroes in 64-bit arg
		{name: "BSFL", argLength: 1, reg: gp11, asm: "BSFL", typ: "UInt32", clobberFlags: true}, // # of low-order zeroes in 32-bit arg
		{name: "BSRQ", argLength: 1, reg: gp11flags, asm: "BSRQ", typ: "(UInt64,Flags)"},        // # of high-order zeroes in 64-bit arg
		{name: "BSRL", argLength: 1, reg: gp11, asm: "BSRL", typ: "UInt32", clobberFlags: true}, // # of high-order zeroes in 32-bit arg

		// CMOV instructions: 64, 32 and 16-bit sizes.
		// if arg2 encodes a true result, return arg1, else arg0
		{name: "CMOVQEQ", argLength: 3, reg: gp21, asm: "CMOVQEQ", resultInArg0: true},
		{name: "CMOVQNE", argLength: 3, reg: gp21, asm: "CMOVQNE", resultInArg0: true},
		{name: "CMOVQLT", argLength: 3, reg: gp21, asm: "CMOVQLT", resultInArg0: true},
		{name: "CMOVQGT", argLength: 3, reg: gp21, asm: "CMOVQGT", resultInArg0: true},
		{name: "CMOVQLE", argLength: 3, reg: gp21, asm: "CMOVQLE", resultInArg0: true},
		{name: "CMOVQGE", argLength: 3, reg: gp21, asm: "CMOVQGE", resultInArg0: true},
		{name: "CMOVQLS", argLength: 3, reg: gp21, asm: "CMOVQLS", resultInArg0: true},
		{name: "CMOVQHI", argLength: 3, reg: gp21, asm: "CMOVQHI", resultInArg0: true},
		{name: "CMOVQCC", argLength: 3, reg: gp21, asm: "CMOVQCC", resultInArg0: true},
		{name: "CMOVQCS", argLength: 3, reg: gp21, asm: "CMOVQCS", resultInArg0: true},

		{name: "CMOVLEQ", argLength: 3, reg: gp21, asm: "CMOVLEQ", resultInArg0: true},
		{name: "CMOVLNE", argLength: 3, reg: gp21, asm: "CMOVLNE", resultInArg0: true},
		{name: "CMOVLLT", argLength: 3, reg: gp21, asm: "CMOVLLT", resultInArg0: true},
		{name: "CMOVLGT", argLength: 3, reg: gp21, asm: "CMOVLGT", resultInArg0: true},
		{name: "CMOVLLE", argLength: 3, reg: gp21, asm: "CMOVLLE", resultInArg0: true},
		{name: "CMOVLGE", argLength: 3, reg: gp21, asm: "CMOVLGE", resultInArg0: true},
		{name: "CMOVLLS", argLength: 3, reg: gp21, asm: "CMOVLLS", resultInArg0: true},
		{name: "CMOVLHI", argLength: 3, reg: gp21, asm: "CMOVLHI", resultInArg0: true},
		{name: "CMOVLCC", argLength: 3, reg: gp21, asm: "CMOVLCC", resultInArg0: true},
		{name: "CMOVLCS", argLength: 3, reg: gp21, asm: "CMOVLCS", resultInArg0: true},

		{name: "CMOVWEQ", argLength: 3, reg: gp21, asm: "CMOVWEQ", resultInArg0: true},
		{name: "CMOVWNE", argLength: 3, reg: gp21, asm: "CMOVWNE", resultInArg0: true},
		{name: "CMOVWLT", argLength: 3, reg: gp21, asm: "CMOVWLT", resultInArg0: true},
		{name: "CMOVWGT", argLength: 3, reg: gp21, asm: "CMOVWGT", resultInArg0: true},
		{name: "CMOVWLE", argLength: 3, reg: gp21, asm: "CMOVWLE", resultInArg0: true},
		{name: "CMOVWGE", argLength: 3, reg: gp21, asm: "CMOVWGE", resultInArg0: true},
		{name: "CMOVWLS", argLength: 3, reg: gp21, asm: "CMOVWLS", resultInArg0: true},
		{name: "CMOVWHI", argLength: 3, reg: gp21, asm: "CMOVWHI", resultInArg0: true},
		{name: "CMOVWCC", argLength: 3, reg: gp21, asm: "CMOVWCC", resultInArg0: true},
		{name: "CMOVWCS", argLength: 3, reg: gp21, asm: "CMOVWCS", resultInArg0: true},

		// CMOV with floating point instructions. We need separate pseudo-op to handle
		// InvertFlags correctly, and to generate special code that handles NaN (unordered flag).
		// NOTE: the fact that CMOV*EQF here is marked to generate CMOV*NE is not a bug. See
		// code generation in amd64/ssa.go.
		{name: "CMOVQEQF", argLength: 3, reg: gp21, asm: "CMOVQNE", resultInArg0: true, needIntTemp: true},
		{name: "CMOVQNEF", argLength: 3, reg: gp21, asm: "CMOVQNE", resultInArg0: true},
		{name: "CMOVQGTF", argLength: 3, reg: gp21, asm: "CMOVQHI", resultInArg0: true},
		{name: "CMOVQGEF", argLength: 3, reg: gp21, asm: "CMOVQCC", resultInArg0: true},
		{name: "CMOVLEQF", argLength: 3, reg: gp21, asm: "CMOVLNE", resultInArg0: true, needIntTemp: true},
		{name: "CMOVLNEF", argLength: 3, reg: gp21, asm: "CMOVLNE", resultInArg0: true},
		{name: "CMOVLGTF", argLength: 3, reg: gp21, asm: "CMOVLHI", resultInArg0: true},
		{name: "CMOVLGEF", argLength: 3, reg: gp21, asm: "CMOVLCC", resultInArg0: true},
		{name: "CMOVWEQF", argLength: 3, reg: gp21, asm: "CMOVWNE", resultInArg0: true, needIntTemp: true},
		{name: "CMOVWNEF", argLength: 3, reg: gp21, asm: "CMOVWNE", resultInArg0: true},
		{name: "CMOVWGTF", argLength: 3, reg: gp21, asm: "CMOVWHI", resultInArg0: true},
		{name: "CMOVWGEF", argLength: 3, reg: gp21, asm: "CMOVWCC", resultInArg0: true},

		// BSWAPx swaps the low-order (L=4,Q=8) bytes of arg0.
		// Q: abcdefgh -> hgfedcba
		// L: abcdefgh -> 0000hgfe (L zeros the upper 4 bytes)
		{name: "BSWAPQ", argLength: 1, reg: gp11, asm: "BSWAPQ", resultInArg0: true},
		{name: "BSWAPL", argLength: 1, reg: gp11, asm: "BSWAPL", resultInArg0: true},

		// POPCNTx counts the number of set bits in the low-order (L=32,Q=64) bits of arg0.
		// POPCNTx instructions are only guaranteed to be available if GOAMD64>=v2.
		// For GOAMD64<v2, any use must be preceded by a successful runtime check of runtime.x86HasPOPCNT.
		{name: "POPCNTQ", argLength: 1, reg: gp11, asm: "POPCNTQ", clobberFlags: true},
		{name: "POPCNTL", argLength: 1, reg: gp11, asm: "POPCNTL", clobberFlags: true},

		// SQRTSx computes sqrt(arg0)
		// S = float32, D = float64
		{name: "SQRTSD", argLength: 1, reg: fp11, asm: "SQRTSD"},
		{name: "SQRTSS", argLength: 1, reg: fp11, asm: "SQRTSS"},

		// ROUNDSD rounds arg0 to an integer depending on auxint
		// 0 means math.RoundToEven, 1 means math.Floor, 2 math.Ceil, 3 math.Trunc
		// (The result is still a float64.)
		// ROUNDSD instruction is only guaraneteed to be available if GOAMD64>=v2.
		// For GOAMD64<v2, any use must be preceded by a successful check of runtime.x86HasSSE41.
		{name: "ROUNDSD", argLength: 1, reg: fp11, aux: "Int8", asm: "ROUNDSD"},
		// See why we need those in issue #71204
		{name: "LoweredRound32F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},
		{name: "LoweredRound64F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},

		// VFMADD231Sx only exist on platforms with the FMA3 instruction set.
		// Any use must be preceded by a successful check of runtime.x86HasFMA or a check of GOAMD64>=v3.
		// x==S for float32, x==D for float64
		// arg0 + arg1*arg2, with no intermediate rounding.
		{name: "VFMADD231SS", argLength: 3, reg: fp31, resultInArg0: true, asm: "VFMADD231SS"},
		{name: "VFMADD231SD", argLength: 3, reg: fp31, resultInArg0: true, asm: "VFMADD231SD"},

		// Note that these operations don't exactly match the semantics of Go's
		// builtin min. In particular, these aren't commutative, because on various
		// special cases the 2nd argument is preferred.
		{name: "MINSD", argLength: 2, reg: fp21, resultInArg0: true, asm: "MINSD"}, // min(arg0,arg1)
		{name: "MINSS", argLength: 2, reg: fp21, resultInArg0: true, asm: "MINSS"}, // min(arg0,arg1)

		{name: "SBBQcarrymask", argLength: 1, reg: flagsgp, asm: "SBBQ"}, // (int64)(-1) if carry is set, 0 if carry is clear.
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
		// Variants that store result to memory
		{name: "SETEQstore", argLength: 3, reg: gpstoreconst, asm: "SETEQ", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},               // extract == condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETNEstore", argLength: 3, reg: gpstoreconst, asm: "SETNE", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},               // extract != condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETLstore", argLength: 3, reg: gpstoreconst, asm: "SETLT", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},                // extract signed < condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETLEstore", argLength: 3, reg: gpstoreconst, asm: "SETLE", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},               // extract signed <= condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETGstore", argLength: 3, reg: gpstoreconst, asm: "SETGT", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},                // extract signed > condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETGEstore", argLength: 3, reg: gpstoreconst, asm: "SETGE", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},               // extract signed >= condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETBstore", argLength: 3, reg: gpstoreconst, asm: "SETCS", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},                // extract unsigned < condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETBEstore", argLength: 3, reg: gpstoreconst, asm: "SETLS", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},               // extract unsigned <= condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETAstore", argLength: 3, reg: gpstoreconst, asm: "SETHI", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},                // extract unsigned > condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETAEstore", argLength: 3, reg: gpstoreconst, asm: "SETCC", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},               // extract unsigned >= condition from arg1 to arg0+auxint+aux, arg2=mem
		{name: "SETEQstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETEQ", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"}, // extract == condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETNEstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETNE", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"}, // extract != condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETLstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETLT", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"},  // extract signed < condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETLEstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETLE", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"}, // extract signed <= condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETGstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETGT", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"},  // extract signed > condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETGEstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETGE", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"}, // extract signed >= condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETBstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETCS", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"},  // extract unsigned < condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETBEstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETLS", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"}, // extract unsigned <= condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETAstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETHI", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"},  // extract unsigned > condition from arg2 to arg0+arg1+auxint+aux, arg3=mem
		{name: "SETAEstoreidx1", argLength: 4, reg: gpstoreconstidx, asm: "SETCC", aux: "SymOff", typ: "Mem", scale: 1, commutative: true, symEffect: "Write"}, // extract unsigned >= condition from arg2 to arg0+arg1+auxint+aux, arg3=mem

		// Need different opcodes for floating point conditions because
		// any comparison involving a NaN is always FALSE and thus
		// the patterns for inverting conditions cannot be used.
		{name: "SETEQF", argLength: 1, reg: flagsgp, asm: "SETEQ", clobberFlags: true, needIntTemp: true}, // extract == condition from arg0
		{name: "SETNEF", argLength: 1, reg: flagsgp, asm: "SETNE", clobberFlags: true, needIntTemp: true}, // extract != condition from arg0
		{name: "SETORD", argLength: 1, reg: flagsgp, asm: "SETPC"},                                        // extract "ordered" (No Nan present) condition from arg0
		{name: "SETNAN", argLength: 1, reg: flagsgp, asm: "SETPS"},                                        // extract "unordered" (Nan present) condition from arg0

		{name: "SETGF", argLength: 1, reg: flagsgp, asm: "SETHI"},  // extract floating > condition from arg0
		{name: "SETGEF", argLength: 1, reg: flagsgp, asm: "SETCC"}, // extract floating >= condition from arg0

		{name: "MOVBQSX", argLength: 1, reg: gp11, asm: "MOVBQSX"}, // sign extend arg0 from int8 to int64
		{name: "MOVBQZX", argLength: 1, reg: gp11, asm: "MOVBLZX"}, // zero extend arg0 from int8 to int64
		{name: "MOVWQSX", argLength: 1, reg: gp11, asm: "MOVWQSX"}, // sign extend arg0 from int16 to int64
		{name: "MOVWQZX", argLength: 1, reg: gp11, asm: "MOVWLZX"}, // zero extend arg0 from int16 to int64
		{name: "MOVLQSX", argLength: 1, reg: gp11, asm: "MOVLQSX"}, // sign extend arg0 from int32 to int64
		{name: "MOVLQZX", argLength: 1, reg: gp11, asm: "MOVL"},    // zero extend arg0 from int32 to int64

		{name: "MOVLconst", reg: gp01, asm: "MOVL", typ: "UInt32", aux: "Int32", rematerializeable: true}, // 32 low bits of auxint (upper 32 are zeroed)
		{name: "MOVQconst", reg: gp01, asm: "MOVQ", typ: "UInt64", aux: "Int64", rematerializeable: true}, // auxint

		{name: "CVTTSD2SL", argLength: 1, reg: fpgp, asm: "CVTTSD2SL"}, // convert float64 to int32
		{name: "CVTTSD2SQ", argLength: 1, reg: fpgp, asm: "CVTTSD2SQ"}, // convert float64 to int64
		{name: "CVTTSS2SL", argLength: 1, reg: fpgp, asm: "CVTTSS2SL"}, // convert float32 to int32
		{name: "CVTTSS2SQ", argLength: 1, reg: fpgp, asm: "CVTTSS2SQ"}, // convert float32 to int64
		{name: "CVTSL2SS", argLength: 1, reg: gpfp, asm: "CVTSL2SS"},   // convert int32 to float32
		{name: "CVTSL2SD", argLength: 1, reg: gpfp, asm: "CVTSL2SD"},   // convert int32 to float64
		{name: "CVTSQ2SS", argLength: 1, reg: gpfp, asm: "CVTSQ2SS"},   // convert int64 to float32
		{name: "CVTSQ2SD", argLength: 1, reg: gpfp, asm: "CVTSQ2SD"},   // convert int64 to float64
		{name: "CVTSD2SS", argLength: 1, reg: fp11, asm: "CVTSD2SS"},   // convert float64 to float32
		{name: "CVTSS2SD", argLength: 1, reg: fp11, asm: "CVTSS2SD"},   // convert float32 to float64

		// Move values between int and float registers, with no conversion.
		// TODO: should we have generic versions of these?
		{name: "MOVQi2f", argLength: 1, reg: gpfp, typ: "Float64"}, // move 64 bits from int to float reg
		{name: "MOVQf2i", argLength: 1, reg: fpgp, typ: "UInt64"},  // move 64 bits from float to int reg
		{name: "MOVLi2f", argLength: 1, reg: gpfp, typ: "Float32"}, // move 32 bits from int to float reg
		{name: "MOVLf2i", argLength: 1, reg: fpgp, typ: "UInt32"},  // move 32 bits from float to int reg, zero extend

		{name: "PXOR", argLength: 2, reg: fp21, asm: "PXOR", commutative: true, resultInArg0: true}, // exclusive or, applied to X regs (for float negation).
		{name: "POR", argLength: 2, reg: fp21, asm: "POR", commutative: true, resultInArg0: true},   // inclusive or, applied to X regs (for float min/max).

		{name: "LEAQ", argLength: 1, reg: gp11sb, asm: "LEAQ", aux: "SymOff", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxint + offset encoded in aux
		{name: "LEAL", argLength: 1, reg: gp11sb, asm: "LEAL", aux: "SymOff", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxint + offset encoded in aux
		{name: "LEAW", argLength: 1, reg: gp11sb, asm: "LEAW", aux: "SymOff", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxint + offset encoded in aux

		// LEAxn computes arg0 + n*arg1 + auxint + aux
		// x==L zeroes the upper 4 bytes.
		{name: "LEAQ1", argLength: 2, reg: gp21sb, asm: "LEAQ", scale: 1, commutative: true, aux: "SymOff", symEffect: "Addr"}, // arg0 + arg1 + auxint + aux
		{name: "LEAL1", argLength: 2, reg: gp21sb, asm: "LEAL", scale: 1, commutative: true, aux: "SymOff", symEffect: "Addr"}, // arg0 + arg1 + auxint + aux
		{name: "LEAW1", argLength: 2, reg: gp21sb, asm: "LEAW", scale: 1, commutative: true, aux: "SymOff", symEffect: "Addr"}, // arg0 + arg1 + auxint + aux
		{name: "LEAQ2", argLength: 2, reg: gp21sb, asm: "LEAQ", scale: 2, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 2*arg1 + auxint + aux
		{name: "LEAL2", argLength: 2, reg: gp21sb, asm: "LEAL", scale: 2, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 2*arg1 + auxint + aux
		{name: "LEAW2", argLength: 2, reg: gp21sb, asm: "LEAW", scale: 2, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 2*arg1 + auxint + aux
		{name: "LEAQ4", argLength: 2, reg: gp21sb, asm: "LEAQ", scale: 4, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 4*arg1 + auxint + aux
		{name: "LEAL4", argLength: 2, reg: gp21sb, asm: "LEAL", scale: 4, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 4*arg1 + auxint + aux
		{name: "LEAW4", argLength: 2, reg: gp21sb, asm: "LEAW", scale: 4, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 4*arg1 + auxint + aux
		{name: "LEAQ8", argLength: 2, reg: gp21sb, asm: "LEAQ", scale: 8, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 8*arg1 + auxint + aux
		{name: "LEAL8", argLength: 2, reg: gp21sb, asm: "LEAL", scale: 8, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 8*arg1 + auxint + aux
		{name: "LEAW8", argLength: 2, reg: gp21sb, asm: "LEAW", scale: 8, aux: "SymOff", symEffect: "Addr"},                    // arg0 + 8*arg1 + auxint + aux
		// Note: LEAx{1,2,4,8} must not have OpSB as either argument.

		// MOVxload: loads
		// Load (Q=8,L=4,W=2,B=1) bytes from (arg0+auxint+aux), arg1=mem.
		// "+auxint+aux" == add auxint and the offset of the symbol in aux (if any) to the effective address
		// Standard versions zero extend the result. SX versions sign extend the result.
		{name: "MOVBload", argLength: 2, reg: gpload, asm: "MOVBLZX", aux: "SymOff", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVBQSXload", argLength: 2, reg: gpload, asm: "MOVBQSX", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVWload", argLength: 2, reg: gpload, asm: "MOVWLZX", aux: "SymOff", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVWQSXload", argLength: 2, reg: gpload, asm: "MOVWQSX", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVLload", argLength: 2, reg: gpload, asm: "MOVL", aux: "SymOff", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVLQSXload", argLength: 2, reg: gpload, asm: "MOVLQSX", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVQload", argLength: 2, reg: gpload, asm: "MOVQ", aux: "SymOff", typ: "UInt64", faultOnNilArg0: true, symEffect: "Read"},

		// MOVxstore: stores
		// Store (Q=8,L=4,W=2,B=1) low bytes of arg1.
		// Does *(arg0+auxint+aux) = arg1, arg2=mem.
		{name: "MOVBstore", argLength: 3, reg: gpstore, asm: "MOVB", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVWstore", argLength: 3, reg: gpstore, asm: "MOVW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVLstore", argLength: 3, reg: gpstore, asm: "MOVL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVQstore", argLength: 3, reg: gpstore, asm: "MOVQ", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},

		// MOVOload/store: 16 byte load/store
		// These operations are only used to move data around: there is no *O arithmetic, for example.
		{name: "MOVOload", argLength: 2, reg: fpload, asm: "MOVUPS", aux: "SymOff", typ: "Int128", faultOnNilArg0: true, symEffect: "Read"}, // load 16 bytes from arg0+auxint+aux. arg1=mem
		{name: "MOVOstore", argLength: 3, reg: fpstore, asm: "MOVUPS", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 16 bytes in arg1 to arg0+auxint+aux. arg2=mem

		// MOVxloadidx: indexed loads
		// load (Q=8,L=4,W=2,B=1) bytes from (arg0+scale*arg1+auxint+aux), arg2=mem.
		// Results are zero-extended. (TODO: sign-extending indexed loads)
		{name: "MOVBloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVBLZX", scale: 1, aux: "SymOff", typ: "UInt8", symEffect: "Read"},
		{name: "MOVWloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVWLZX", scale: 1, aux: "SymOff", typ: "UInt16", symEffect: "Read"},
		{name: "MOVWloadidx2", argLength: 3, reg: gploadidx, asm: "MOVWLZX", scale: 2, aux: "SymOff", typ: "UInt16", symEffect: "Read"},
		{name: "MOVLloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVL", scale: 1, aux: "SymOff", typ: "UInt32", symEffect: "Read"},
		{name: "MOVLloadidx4", argLength: 3, reg: gploadidx, asm: "MOVL", scale: 4, aux: "SymOff", typ: "UInt32", symEffect: "Read"},
		{name: "MOVLloadidx8", argLength: 3, reg: gploadidx, asm: "MOVL", scale: 8, aux: "SymOff", typ: "UInt32", symEffect: "Read"},
		{name: "MOVQloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVQ", scale: 1, aux: "SymOff", typ: "UInt64", symEffect: "Read"},
		{name: "MOVQloadidx8", argLength: 3, reg: gploadidx, asm: "MOVQ", scale: 8, aux: "SymOff", typ: "UInt64", symEffect: "Read"},

		// MOVxstoreidx: indexed stores
		// Store (Q=8,L=4,W=2,B=1) low bytes of arg2.
		// Does *(arg0+scale*arg1+auxint+aux) = arg2, arg3=mem.
		{name: "MOVBstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVB", scale: 1, aux: "SymOff", symEffect: "Write"},
		{name: "MOVWstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVW", scale: 1, aux: "SymOff", symEffect: "Write"},
		{name: "MOVWstoreidx2", argLength: 4, reg: gpstoreidx, asm: "MOVW", scale: 2, aux: "SymOff", symEffect: "Write"},
		{name: "MOVLstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVL", scale: 1, aux: "SymOff", symEffect: "Write"},
		{name: "MOVLstoreidx4", argLength: 4, reg: gpstoreidx, asm: "MOVL", scale: 4, aux: "SymOff", symEffect: "Write"},
		{name: "MOVLstoreidx8", argLength: 4, reg: gpstoreidx, asm: "MOVL", scale: 8, aux: "SymOff", symEffect: "Write"},
		{name: "MOVQstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVQ", scale: 1, aux: "SymOff", symEffect: "Write"},
		{name: "MOVQstoreidx8", argLength: 4, reg: gpstoreidx, asm: "MOVQ", scale: 8, aux: "SymOff", symEffect: "Write"},

		// TODO: add size-mismatched indexed loads/stores, like MOVBstoreidx4?

		// MOVxstoreconst: constant stores
		// Store (O=16,Q=8,L=4,W=2,B=1) constant bytes.
		// Does *(arg0+ValAndOff(AuxInt).Off()+aux) = ValAndOff(AuxInt).Val(), arg1=mem.
		// O version can only store the constant 0.
		{name: "MOVBstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVB", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVWstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVW", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVLstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVL", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVQstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVQ", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVOstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVUPS", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},

		// MOVxstoreconstidx: constant indexed stores
		// Store (Q=8,L=4,W=2,B=1) constant bytes.
		// Does *(arg0+scale*arg1+ValAndOff(AuxInt).Off()+aux) = ValAndOff(AuxInt).Val(), arg2=mem.
		{name: "MOVBstoreconstidx1", argLength: 3, reg: gpstoreconstidx, commutative: true, asm: "MOVB", scale: 1, aux: "SymValAndOff", typ: "Mem", symEffect: "Write"},
		{name: "MOVWstoreconstidx1", argLength: 3, reg: gpstoreconstidx, commutative: true, asm: "MOVW", scale: 1, aux: "SymValAndOff", typ: "Mem", symEffect: "Write"},
		{name: "MOVWstoreconstidx2", argLength: 3, reg: gpstoreconstidx, asm: "MOVW", scale: 2, aux: "SymValAndOff", typ: "Mem", symEffect: "Write"},
		{name: "MOVLstoreconstidx1", argLength: 3, reg: gpstoreconstidx, commutative: true, asm: "MOVL", scale: 1, aux: "SymValAndOff", typ: "Mem", symEffect: "Write"},
		{name: "MOVLstoreconstidx4", argLength: 3, reg: gpstoreconstidx, asm: "MOVL", scale: 4, aux: "SymValAndOff", typ: "Mem", symEffect: "Write"},
		{name: "MOVQstoreconstidx1", argLength: 3, reg: gpstoreconstidx, commutative: true, asm: "MOVQ", scale: 1, aux: "SymValAndOff", typ: "Mem", symEffect: "Write"},
		{name: "MOVQstoreconstidx8", argLength: 3, reg: gpstoreconstidx, asm: "MOVQ", scale: 8, aux: "SymValAndOff", typ: "Mem", symEffect: "Write"},

		// arg0 = pointer to start of memory to zero
		// arg1 = mem
		// auxint = # of bytes to zero
		// returns mem
		{
			name:      "DUFFZERO",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI")},
				clobbers: buildReg("DI"),
			},
			//faultOnNilArg0: true, // Note: removed for 73748. TODO: reenable at some point
			unsafePoint: true, // FP maintenance around DUFFCOPY can be clobbered by interrupts
		},

		// arg0 = address of memory to zero
		// arg1 = # of 8-byte words to zero
		// arg2 = value to store (will always be zero)
		// arg3 = mem
		// returns mem
		{
			name:      "REPSTOSQ",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("CX"), buildReg("AX")},
				clobbers: buildReg("DI CX"),
			},
			faultOnNilArg0: true,
		},

		// With a register ABI, the actual register info for these instructions (i.e., what is used in regalloc) is augmented with per-call-site bindings of additional arguments to specific in and out registers.
		{name: "CALLstatic", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                              // call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                // tail call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: -1, reg: regInfo{inputs: []regMask{gpsp, buildReg("DX"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, last arg=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: -1, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                        // call fn by pointer.  arg0=codeptr, last arg=mem, auxint=argsize, returns mem

		// arg0 = destination pointer
		// arg1 = source pointer
		// arg2 = mem
		// auxint = # of bytes to copy, must be multiple of 16
		// returns memory
		{
			name:      "DUFFCOPY",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("SI")},
				clobbers: buildReg("DI SI X0"), // uses X0 as a temporary
			},
			clobberFlags: true,
			//faultOnNilArg0: true, // Note: removed for 73748. TODO: reenable at some point
			//faultOnNilArg1: true,
			unsafePoint: true, // FP maintenance around DUFFCOPY can be clobbered by interrupts
		},

		// arg0 = destination pointer
		// arg1 = source pointer
		// arg2 = # of 8-byte words to copy
		// arg3 = mem
		// returns memory
		{
			name:      "REPMOVSQ",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("DI"), buildReg("SI"), buildReg("CX")},
				clobbers: buildReg("DI SI CX"),
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// (InvertFlags (CMPQ a b)) == (CMPQ b a)
		// So if we want (SETL (CMPQ a b)) but we can't do that because a is a constant,
		// then we do (SETL (InvertFlags (CMPQ b a))) instead.
		// Rewrites will convert this to (SETG (CMPQ b a)).
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags", argLength: 1}, // reverse direction of arg0

		// Pseudo-ops
		{name: "LoweredGetG", argLength: 1, reg: gp01}, // arg0=mem
		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of DX (the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("DX")}}, zeroWidth: true},
		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" sys.GetCallerPC,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},
		// LoweredGetCallerSP returns the SP of the caller of the current function. arg0=mem
		{name: "LoweredGetCallerSP", argLength: 1, reg: gp01, rematerializeable: true},
		//arg0=ptr,arg1=mem, returns void.  Faults if ptr is nil.
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gpsp}}, clobberFlags: true, nilCheck: true, faultOnNilArg0: true},
		// LoweredWB invokes runtime.gcWriteBarrier{auxint}. arg0=mem, auxint=# of buffer entries needed.
		// It saves all GP registers if necessary, but may clobber others.
		// Returns a pointer to a write barrier buffer in R11.
		{name: "LoweredWB", argLength: 1, reg: regInfo{clobbers: callerSave &^ (gp | g), outputs: []regMask{buildReg("R11")}}, clobberFlags: true, aux: "Int64"},

		{name: "LoweredHasCPUFeature", argLength: 0, reg: gp01, rematerializeable: true, typ: "UInt64", aux: "Sym", symEffect: "None"},

		// There are three of these functions so that they can have three different register inputs.
		// When we check 0 <= c <= cap (A), then 0 <= b <= c (B), then 0 <= a <= b (C), we want the
		// default registers to match so we don't need to copy registers around unnecessarily.
		{name: "LoweredPanicBoundsA", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{dx, bx}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in generic.go).
		{name: "LoweredPanicBoundsB", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{cx, dx}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in generic.go).
		{name: "LoweredPanicBoundsC", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{ax, cx}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in generic.go).

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
		{name: "FlagGT_UGT"}, // signed > and unsigned >
		{name: "FlagGT_ULT"}, // signed > and unsigned <

		// Atomic loads.  These are just normal loads but return <value,memory> tuples
		// so they can be properly ordered with other loads.
		// load from arg0+auxint+aux.  arg1=mem.
		{name: "MOVBatomicload", argLength: 2, reg: gpload, asm: "MOVB", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVLatomicload", argLength: 2, reg: gpload, asm: "MOVL", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVQatomicload", argLength: 2, reg: gpload, asm: "MOVQ", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},

		// Atomic stores and exchanges.  Stores use XCHG to get the right memory ordering semantics.
		// store arg0 to arg1+auxint+aux, arg2=mem.
		// These ops return a tuple of <old contents of *(arg1+auxint+aux), memory>.
		// Note: arg0 and arg1 are backwards compared to MOVLstore (to facilitate resultInArg0)!
		{name: "XCHGB", argLength: 3, reg: gpstorexchg, asm: "XCHGB", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "XCHGL", argLength: 3, reg: gpstorexchg, asm: "XCHGL", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "XCHGQ", argLength: 3, reg: gpstorexchg, asm: "XCHGQ", aux: "SymOff", resultInArg0: true, faultOnNilArg1: true, hasSideEffects: true, symEffect: "RdWr"},

		// Atomic adds.
		// *(arg1+auxint+aux) += arg0.  arg2=mem.
		// Returns a tuple of <old contents of *(arg1+auxint+aux), memory>.
		// Note: arg0 and arg1 are backwards compared to MOVLstore (to facilitate resultInArg0)!
		{name: "XADDLlock", argLength: 3, reg: gpstorexchg, asm: "XADDL", typ: "(UInt32,Mem)", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "XADDQlock", argLength: 3, reg: gpstorexchg, asm: "XADDQ", typ: "(UInt64,Mem)", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "AddTupleFirst32", argLength: 2}, // arg1=tuple <x,y>.  Returns <x+arg0,y>.
		{name: "AddTupleFirst64", argLength: 2}, // arg1=tuple <x,y>.  Returns <x+arg0,y>.

		// Compare and swap.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory.
		// if *(arg0+auxint+aux) == arg1 {
		//   *(arg0+auxint+aux) = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// Note that these instructions also return the old value in AX, but we ignore it.
		// TODO: have these return flags instead of bool.  The current system generates:
		//    CMPXCHGQ ...
		//    SETEQ AX
		//    CMPB  AX, $0
		//    JNE ...
		// instead of just
		//    CMPXCHGQ ...
		//    JEQ ...
		// but we can't do that because memory-using ops can't generate flags yet
		// (flagalloc wants to move flag-generating instructions around).
		{name: "CMPXCHGLlock", argLength: 4, reg: cmpxchg, asm: "CMPXCHGL", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "CMPXCHGQlock", argLength: 4, reg: cmpxchg, asm: "CMPXCHGQ", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},

		// Atomic memory updates using logical operations.
		// Old style that just returns the memory state.
		{name: "ANDBlock", argLength: 3, reg: gpstore, asm: "ANDB", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"}, // *(arg0+auxint+aux) &= arg1
		{name: "ANDLlock", argLength: 3, reg: gpstore, asm: "ANDL", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"}, // *(arg0+auxint+aux) &= arg1
		{name: "ANDQlock", argLength: 3, reg: gpstore, asm: "ANDQ", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"}, // *(arg0+auxint+aux) &= arg1
		{name: "ORBlock", argLength: 3, reg: gpstore, asm: "ORB", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},   // *(arg0+auxint+aux) |= arg1
		{name: "ORLlock", argLength: 3, reg: gpstore, asm: "ORL", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},   // *(arg0+auxint+aux) |= arg1
		{name: "ORQlock", argLength: 3, reg: gpstore, asm: "ORQ", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},   // *(arg0+auxint+aux) |= arg1

		// Atomic memory updates using logical operations.
		// *(arg0+auxint+aux) op= arg1. arg2=mem.
		// New style that returns a tuple of <old contents of *(arg0+auxint+aux), memory>.
		{name: "LoweredAtomicAnd64", argLength: 3, reg: atomicLogic, resultNotInArgs: true, asm: "ANDQ", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr", unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicAnd32", argLength: 3, reg: atomicLogic, resultNotInArgs: true, asm: "ANDL", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr", unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicOr64", argLength: 3, reg: atomicLogic, resultNotInArgs: true, asm: "ORQ", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr", unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicOr32", argLength: 3, reg: atomicLogic, resultNotInArgs: true, asm: "ORL", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr", unsafePoint: true, needIntTemp: true},

		// Prefetch instructions
		// Do prefetch arg0 address. arg0=addr, arg1=memory. Instruction variant selects locality hint
		{name: "PrefetchT0", argLength: 2, reg: prefreg, asm: "PREFETCHT0", hasSideEffects: true},
		{name: "PrefetchNTA", argLength: 2, reg: prefreg, asm: "PREFETCHNTA", hasSideEffects: true},

		// CPUID feature: BMI1.
		{name: "ANDNQ", argLength: 2, reg: gp21, asm: "ANDNQ", clobberFlags: true},         // arg0 &^ arg1
		{name: "ANDNL", argLength: 2, reg: gp21, asm: "ANDNL", clobberFlags: true},         // arg0 &^ arg1
		{name: "BLSIQ", argLength: 1, reg: gp11, asm: "BLSIQ", clobberFlags: true},         // arg0 & -arg0
		{name: "BLSIL", argLength: 1, reg: gp11, asm: "BLSIL", clobberFlags: true},         // arg0 & -arg0
		{name: "BLSMSKQ", argLength: 1, reg: gp11, asm: "BLSMSKQ", clobberFlags: true},     // arg0 ^ (arg0 - 1)
		{name: "BLSMSKL", argLength: 1, reg: gp11, asm: "BLSMSKL", clobberFlags: true},     // arg0 ^ (arg0 - 1)
		{name: "BLSRQ", argLength: 1, reg: gp11flags, asm: "BLSRQ", typ: "(UInt64,Flags)"}, // arg0 & (arg0 - 1)
		{name: "BLSRL", argLength: 1, reg: gp11flags, asm: "BLSRL", typ: "(UInt32,Flags)"}, // arg0 & (arg0 - 1)
		// count the number of trailing zero bits, prefer TZCNTQ over BSFQ, as TZCNTQ(0)==64
		// and BSFQ(0) is undefined. Same for TZCNTL(0)==32
		{name: "TZCNTQ", argLength: 1, reg: gp11, asm: "TZCNTQ", clobberFlags: true},
		{name: "TZCNTL", argLength: 1, reg: gp11, asm: "TZCNTL", clobberFlags: true},

		// CPUID feature: LZCNT.
		// count the number of leading zero bits.
		{name: "LZCNTQ", argLength: 1, reg: gp11, asm: "LZCNTQ", typ: "UInt64", clobberFlags: true},
		{name: "LZCNTL", argLength: 1, reg: gp11, asm: "LZCNTL", typ: "UInt32", clobberFlags: true},

		// CPUID feature: MOVBE
		// MOVBEWload does not satisfy zero extended, so only use MOVBEWstore
		{name: "MOVBEWstore", argLength: 3, reg: gpstore, asm: "MOVBEW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // swap and store 2 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVBELload", argLength: 2, reg: gpload, asm: "MOVBEL", aux: "SymOff", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"}, // load and swap 4 bytes from arg0+auxint+aux. arg1=mem.  Zero extend.
		{name: "MOVBELstore", argLength: 3, reg: gpstore, asm: "MOVBEL", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // swap and store 4 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVBEQload", argLength: 2, reg: gpload, asm: "MOVBEQ", aux: "SymOff", typ: "UInt64", faultOnNilArg0: true, symEffect: "Read"}, // load and swap 8 bytes from arg0+auxint+aux. arg1=mem
		{name: "MOVBEQstore", argLength: 3, reg: gpstore, asm: "MOVBEQ", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // swap and store 8 bytes in arg1 to arg0+auxint+aux. arg2=mem
		// indexed MOVBE loads
		{name: "MOVBELloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVBEL", scale: 1, aux: "SymOff", typ: "UInt32", symEffect: "Read"}, // load and swap 4 bytes from arg0+arg1+auxint+aux. arg2=mem. Zero extend.
		{name: "MOVBELloadidx4", argLength: 3, reg: gploadidx, asm: "MOVBEL", scale: 4, aux: "SymOff", typ: "UInt32", symEffect: "Read"},                    // load and swap 4 bytes from arg0+4*arg1+auxint+aux. arg2=mem. Zero extend.
		{name: "MOVBELloadidx8", argLength: 3, reg: gploadidx, asm: "MOVBEL", scale: 8, aux: "SymOff", typ: "UInt32", symEffect: "Read"},                    // load and swap 4 bytes from arg0+8*arg1+auxint+aux. arg2=mem. Zero extend.
		{name: "MOVBEQloadidx1", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVBEQ", scale: 1, aux: "SymOff", typ: "UInt64", symEffect: "Read"}, // load and swap 8 bytes from arg0+arg1+auxint+aux. arg2=mem
		{name: "MOVBEQloadidx8", argLength: 3, reg: gploadidx, asm: "MOVBEQ", scale: 8, aux: "SymOff", typ: "UInt64", symEffect: "Read"},                    // load and swap 8 bytes from arg0+8*arg1+auxint+aux. arg2=mem
		// indexed MOVBE stores
		{name: "MOVBEWstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVBEW", scale: 1, aux: "SymOff", symEffect: "Write"}, // swap and store 2 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVBEWstoreidx2", argLength: 4, reg: gpstoreidx, asm: "MOVBEW", scale: 2, aux: "SymOff", symEffect: "Write"},                    // swap and store 2 bytes in arg2 to arg0+2*arg1+auxint+aux. arg3=mem
		{name: "MOVBELstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVBEL", scale: 1, aux: "SymOff", symEffect: "Write"}, // swap and store 4 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVBELstoreidx4", argLength: 4, reg: gpstoreidx, asm: "MOVBEL", scale: 4, aux: "SymOff", symEffect: "Write"},                    // swap and store 4 bytes in arg2 to arg0+4*arg1+auxint+aux. arg3=mem
		{name: "MOVBELstoreidx8", argLength: 4, reg: gpstoreidx, asm: "MOVBEL", scale: 8, aux: "SymOff", symEffect: "Write"},                    // swap and store 4 bytes in arg2 to arg0+8*arg1+auxint+aux. arg3=mem
		{name: "MOVBEQstoreidx1", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVBEQ", scale: 1, aux: "SymOff", symEffect: "Write"}, // swap and store 8 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVBEQstoreidx8", argLength: 4, reg: gpstoreidx, asm: "MOVBEQ", scale: 8, aux: "SymOff", symEffect: "Write"},                    // swap and store 8 bytes in arg2 to arg0+8*arg1+auxint+aux. arg3=mem

		// CPUID feature: BMI2.
		{name: "SARXQ", argLength: 2, reg: gp21, asm: "SARXQ"}, // signed arg0 >> arg1, shift amount is mod 64
		{name: "SARXL", argLength: 2, reg: gp21, asm: "SARXL"}, // signed int32(arg0) >> arg1, shift amount is mod 32
		{name: "SHLXQ", argLength: 2, reg: gp21, asm: "SHLXQ"}, // arg0 << arg1, shift amount is mod 64
		{name: "SHLXL", argLength: 2, reg: gp21, asm: "SHLXL"}, // arg0 << arg1, shift amount is mod 32
		{name: "SHRXQ", argLength: 2, reg: gp21, asm: "SHRXQ"}, // unsigned arg0 >> arg1, shift amount is mod 64
		{name: "SHRXL", argLength: 2, reg: gp21, asm: "SHRXL"}, // unsigned uint32(arg0) >> arg1, shift amount is mod 32

		{name: "SARXLload", argLength: 3, reg: gp21shxload, asm: "SARXL", aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // signed *(arg0+auxint+aux) >> arg1, arg2=mem, shift amount is mod 32
		{name: "SARXQload", argLength: 3, reg: gp21shxload, asm: "SARXQ", aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // signed *(arg0+auxint+aux) >> arg1, arg2=mem, shift amount is mod 64
		{name: "SHLXLload", argLength: 3, reg: gp21shxload, asm: "SHLXL", aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // *(arg0+auxint+aux) << arg1, arg2=mem, shift amount is mod 32
		{name: "SHLXQload", argLength: 3, reg: gp21shxload, asm: "SHLXQ", aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // *(arg0+auxint+aux) << arg1, arg2=mem, shift amount is mod 64
		{name: "SHRXLload", argLength: 3, reg: gp21shxload, asm: "SHRXL", aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // unsigned *(arg0+auxint+aux) >> arg1, arg2=mem, shift amount is mod 32
		{name: "SHRXQload", argLength: 3, reg: gp21shxload, asm: "SHRXQ", aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // unsigned *(arg0+auxint+aux) >> arg1, arg2=mem, shift amount is mod 64

		{name: "SARXLloadidx1", argLength: 4, reg: gp21shxloadidx, asm: "SARXL", scale: 1, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // signed *(arg0+1*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 32
		{name: "SARXLloadidx4", argLength: 4, reg: gp21shxloadidx, asm: "SARXL", scale: 4, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // signed *(arg0+4*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 32
		{name: "SARXLloadidx8", argLength: 4, reg: gp21shxloadidx, asm: "SARXL", scale: 8, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // signed *(arg0+8*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 32
		{name: "SARXQloadidx1", argLength: 4, reg: gp21shxloadidx, asm: "SARXQ", scale: 1, aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // signed *(arg0+1*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 64
		{name: "SARXQloadidx8", argLength: 4, reg: gp21shxloadidx, asm: "SARXQ", scale: 8, aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // signed *(arg0+8*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 64
		{name: "SHLXLloadidx1", argLength: 4, reg: gp21shxloadidx, asm: "SHLXL", scale: 1, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // *(arg0+1*arg1+auxint+aux) << arg2, arg3=mem, shift amount is mod 32
		{name: "SHLXLloadidx4", argLength: 4, reg: gp21shxloadidx, asm: "SHLXL", scale: 4, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // *(arg0+4*arg1+auxint+aux) << arg2, arg3=mem, shift amount is mod 32
		{name: "SHLXLloadidx8", argLength: 4, reg: gp21shxloadidx, asm: "SHLXL", scale: 8, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // *(arg0+8*arg1+auxint+aux) << arg2, arg3=mem, shift amount is mod 32
		{name: "SHLXQloadidx1", argLength: 4, reg: gp21shxloadidx, asm: "SHLXQ", scale: 1, aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // *(arg0+1*arg1+auxint+aux) << arg2, arg3=mem, shift amount is mod 64
		{name: "SHLXQloadidx8", argLength: 4, reg: gp21shxloadidx, asm: "SHLXQ", scale: 8, aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // *(arg0+8*arg1+auxint+aux) << arg2, arg3=mem, shift amount is mod 64
		{name: "SHRXLloadidx1", argLength: 4, reg: gp21shxloadidx, asm: "SHRXL", scale: 1, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // unsigned *(arg0+1*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 32
		{name: "SHRXLloadidx4", argLength: 4, reg: gp21shxloadidx, asm: "SHRXL", scale: 4, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // unsigned *(arg0+4*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 32
		{name: "SHRXLloadidx8", argLength: 4, reg: gp21shxloadidx, asm: "SHRXL", scale: 8, aux: "SymOff", typ: "Uint32", faultOnNilArg0: true, symEffect: "Read"}, // unsigned *(arg0+8*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 32
		{name: "SHRXQloadidx1", argLength: 4, reg: gp21shxloadidx, asm: "SHRXQ", scale: 1, aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // unsigned *(arg0+1*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 64
		{name: "SHRXQloadidx8", argLength: 4, reg: gp21shxloadidx, asm: "SHRXQ", scale: 8, aux: "SymOff", typ: "Uint64", faultOnNilArg0: true, symEffect: "Read"}, // unsigned *(arg0+8*arg1+auxint+aux) >> arg2, arg3=mem, shift amount is mod 64

		// Unpack bytes, low 64-bits.
		//
		// Input/output registers treated as [8]uint8.
		//
		// output = {in1[0], in2[0], in1[1], in2[1], in1[2], in2[2], in1[3], in2[3]}
		{name: "PUNPCKLBW", argLength: 2, reg: fp21, resultInArg0: true, asm: "PUNPCKLBW"},

		// Shuffle 16-bit words, low 64-bits.
		//
		// Input/output registers treated as [4]uint16.
		// aux=source word index for each destination word, 2 bits per index.
		//
		// output[i] = input[(aux>>2*i)&3].
		{name: "PSHUFLW", argLength: 1, reg: fp11, aux: "Int8", asm: "PSHUFLW"},

		// Broadcast input byte.
		//
		// Input treated as uint8, output treated as [16]uint8.
		//
		// output[i] = input.
		{name: "PSHUFBbroadcast", argLength: 1, reg: fp11, resultInArg0: true, asm: "PSHUFB"}, // PSHUFB with mask zero, (GOAMD64=v1)
		{name: "VPBROADCASTB", argLength: 1, reg: gpfp, asm: "VPBROADCASTB"},                  // Broadcast input byte from gp (GOAMD64=v3)

		// Byte negate/zero/preserve (GOAMD64=v2).
		//
		// Input/output registers treated as [16]uint8.
		//
		// if in2[i] > 0 {
		//   output[i] = in1[i]
		// } else if in2[i] == 0 {
		//   output[i] = 0
		// } else {
		//   output[i] = -1 * in1[i]
		// }
		{name: "PSIGNB", argLength: 2, reg: fp21, resultInArg0: true, asm: "PSIGNB"},

		// Byte compare.
		//
		// Input/output registers treated as [16]uint8.
		//
		// if in1[i] == in2[i] {
		//   output[i] = 0xff
		// } else {
		//   output[i] = 0
		// }
		{name: "PCMPEQB", argLength: 2, reg: fp21, resultInArg0: true, asm: "PCMPEQB", commutative: true},

		// Byte sign mask. Output is a bitmap of sign bits from each input byte.
		//
		// Input treated as [16]uint8. Output is [16]bit (uint16 bitmap).
		//
		// output[i] = (input[i] >> 7) & 1
		{name: "PMOVMSKB", argLength: 1, reg: fpgp, asm: "PMOVMSKB"},

		// XXX SIMD
		{name: "VPADDD4", argLength: 2, reg: fp21, asm: "VPADDD", commutative: true}, // arg0 + arg1

		{name: "VMOVDQUload128", argLength: 2, reg: fpload, asm: "VMOVDQU", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},    // load from arg0+auxint+aux, arg1 = mem
		{name: "VMOVDQUstore128", argLength: 3, reg: fpstore, asm: "VMOVDQU", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"}, // store, *(arg0+auxint+aux) = arg1, arg2 = mem

		{name: "VMOVDQUload256", argLength: 2, reg: fpload, asm: "VMOVDQU", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},    // load from arg0+auxint+aux, arg1 = mem
		{name: "VMOVDQUstore256", argLength: 3, reg: fpstore, asm: "VMOVDQU", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"}, // store, *(arg0+auxint+aux) = arg1, arg2 = mem

		{name: "VMOVDQUload512", argLength: 2, reg: fpload, asm: "VMOVDQU64", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},    // load from arg0+auxint+aux, arg1 = mem
		{name: "VMOVDQUstore512", argLength: 3, reg: fpstore, asm: "VMOVDQU64", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"}, // store, *(arg0+auxint+aux) = arg1, arg2 = mem

		{name: "VPMOVMToVec8x16", argLength: 1, reg: m1fp1, asm: "VPMOVM2B"},
		{name: "VPMOVMToVec8x32", argLength: 1, reg: m1fp1, asm: "VPMOVM2B"},
		{name: "VPMOVMToVec8x64", argLength: 1, reg: m1fp1, asm: "VPMOVM2B"},

		{name: "VPMOVMToVec16x8", argLength: 1, reg: m1fp1, asm: "VPMOVM2W"},
		{name: "VPMOVMToVec16x16", argLength: 1, reg: m1fp1, asm: "VPMOVM2W"},
		{name: "VPMOVMToVec16x32", argLength: 1, reg: m1fp1, asm: "VPMOVM2W"},

		{name: "VPMOVMToVec32x4", argLength: 1, reg: m1fp1, asm: "VPMOVM2D"},
		{name: "VPMOVMToVec32x8", argLength: 1, reg: m1fp1, asm: "VPMOVM2D"},
		{name: "VPMOVMToVec32x16", argLength: 1, reg: m1fp1, asm: "VPMOVM2D"},

		{name: "VPMOVMToVec64x2", argLength: 1, reg: m1fp1, asm: "VPMOVM2Q"},
		{name: "VPMOVMToVec64x4", argLength: 1, reg: m1fp1, asm: "VPMOVM2Q"},
		{name: "VPMOVMToVec64x8", argLength: 1, reg: m1fp1, asm: "VPMOVM2Q"},

		{name: "VPMOVVec8x16ToM", argLength: 1, reg: fp1m1, asm: "VPMOVB2M"},
		{name: "VPMOVVec8x32ToM", argLength: 1, reg: fp1m1, asm: "VPMOVB2M"},
		{name: "VPMOVVec8x64ToM", argLength: 1, reg: fp1m1, asm: "VPMOVB2M"},

		{name: "VPMOVVec16x8ToM", argLength: 1, reg: fp1m1, asm: "VPMOVW2M"},
		{name: "VPMOVVec16x16ToM", argLength: 1, reg: fp1m1, asm: "VPMOVW2M"},
		{name: "VPMOVVec16x32ToM", argLength: 1, reg: fp1m1, asm: "VPMOVW2M"},

		{name: "VPMOVVec32x4ToM", argLength: 1, reg: fp1m1, asm: "VPMOVD2M"},
		{name: "VPMOVVec32x8ToM", argLength: 1, reg: fp1m1, asm: "VPMOVD2M"},
		{name: "VPMOVVec32x16ToM", argLength: 1, reg: fp1m1, asm: "VPMOVD2M"},

		{name: "VPMOVVec64x2ToM", argLength: 1, reg: fp1m1, asm: "VPMOVQ2M"},
		{name: "VPMOVVec64x4ToM", argLength: 1, reg: fp1m1, asm: "VPMOVQ2M"},
		{name: "VPMOVVec64x8ToM", argLength: 1, reg: fp1m1, asm: "VPMOVQ2M"},

		{name: "Zero128", argLength: 0, reg: fp01, asm: "VPXOR"},
		{name: "Zero256", argLength: 0, reg: fp01, asm: "VPXOR"},
		{name: "Zero512", argLength: 0, reg: fp01, asm: "VPXORQ"},
	}

	var AMD64blocks = []blockData{
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

		// JUMPTABLE implements jump tables.
		// Aux is the symbol (an *obj.LSym) for the jump table.
		// control[0] is the index into the jump table.
		// control[1] is the address of the jump table (the address of the symbol stored in Aux).
		{name: "JUMPTABLE", controls: 2, aux: "Sym"},
	}

	archs = append(archs, arch{
		name:               "AMD64",
		pkg:                "cmd/internal/obj/x86",
		genfile:            "../../amd64/ssa.go",
		genSIMDfile:        "../../amd64/simdssa.go",
		ops:                append(AMD64ops, simdAMD64Ops(fp11, fp21, fp2m1, fp1m1fp1, fp2m1fp1, fp2m1m1, fp3fp1, fp3m1fp1)...), // AMD64ops,
		blocks:             AMD64blocks,
		regnames:           regNamesAMD64,
		ParamIntRegNames:   "AX BX CX DI SI R8 R9 R10 R11",
		ParamFloatRegNames: "X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14",
		gpregmask:          gp,
		fpregmask:          fp,
		specialregmask:     x15 | mask,
		framepointerreg:    int8(num["BP"]),
		linkreg:            -1, // not used
	})
}
