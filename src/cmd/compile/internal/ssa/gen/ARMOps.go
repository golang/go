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
//  - *const instructions may use a constant larger than the instruction can encode.
//    In this case the assembler expands to multiple instructions and uses tmp
//    register (R11).

// Suffixes encode the bit width of various instructions.
// W (word)      = 32 bit
// H (half word) = 16 bit
// HU            = 16 bit unsigned
// B (byte)      = 8 bit
// BU            = 8 bit unsigned
// F (float)     = 32 bit float
// D (double)    = 64 bit float

var regNamesARM = []string{
	"R0",
	"R1",
	"R2",
	"R3",
	"R4",
	"R5",
	"R6",
	"R7",
	"R8",
	"R9",
	"g",   // aka R10
	"R11", // tmp
	"R12",
	"SP",  // aka R13
	"R14", // link
	"R15", // pc

	"F0",
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",
	"F8",
	"F9",
	"F10",
	"F11",
	"F12",
	"F13",
	"F14",
	"F15", // tmp

	// If you add registers, update asyncPreempt in runtime.

	// pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesARM) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesARM {
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
		gp         = buildReg("R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14")
		gpg        = gp | buildReg("g")
		gpsp       = gp | buildReg("SP")
		gpspg      = gpg | buildReg("SP")
		gpspsbg    = gpspg | buildReg("SB")
		fp         = buildReg("F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15")
		callerSave = gp | fp | buildReg("g") // runtime.setg (and anything calling it) may clobber g
		r0         = buildReg("R0")
		r1         = buildReg("R1")
		r2         = buildReg("R2")
		r3         = buildReg("R3")
		r4         = buildReg("R4")
	)
	// Common regInfo
	var (
		gp01      = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11      = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp11carry = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp, 0}}
		gp11sp    = regInfo{inputs: []regMask{gpspg}, outputs: []regMask{gp}}
		gp1flags  = regInfo{inputs: []regMask{gpg}}
		gp1flags1 = regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}
		gp21      = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp}}
		gp21carry = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp, 0}}
		gp2flags  = regInfo{inputs: []regMask{gpg, gpg}}
		gp2flags1 = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp}}
		gp22      = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp, gp}}
		gp31      = regInfo{inputs: []regMask{gp, gp, gp}, outputs: []regMask{gp}}
		gp31carry = regInfo{inputs: []regMask{gp, gp, gp}, outputs: []regMask{gp, 0}}
		gp3flags  = regInfo{inputs: []regMask{gp, gp, gp}}
		gp3flags1 = regInfo{inputs: []regMask{gp, gp, gp}, outputs: []regMask{gp}}
		gpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}
		gpstore   = regInfo{inputs: []regMask{gpspsbg, gpg}}
		gp2load   = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{gp}}
		gp2store  = regInfo{inputs: []regMask{gpspsbg, gpg, gpg}}
		fp01      = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11      = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fp1flags  = regInfo{inputs: []regMask{fp}}
		fpgp      = regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}, clobbers: buildReg("F15")} // int-float conversion uses F15 as tmp
		gpfp      = regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}, clobbers: buildReg("F15")}
		fp21      = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp31      = regInfo{inputs: []regMask{fp, fp, fp}, outputs: []regMask{fp}}
		fp2flags  = regInfo{inputs: []regMask{fp, fp}}
		fpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{fp}}
		fpstore   = regInfo{inputs: []regMask{gpspsbg, fp}}
		readflags = regInfo{inputs: nil, outputs: []regMask{gp}}
	)
	ops := []opData{
		// binary ops
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},     // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11sp, asm: "ADD", aux: "Int32"},   // arg0 + auxInt
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB"},                        // arg0 - arg1
		{name: "SUBconst", argLength: 1, reg: gp11, asm: "SUB", aux: "Int32"},     // arg0 - auxInt
		{name: "RSB", argLength: 2, reg: gp21, asm: "RSB"},                        // arg1 - arg0
		{name: "RSBconst", argLength: 1, reg: gp11, asm: "RSB", aux: "Int32"},     // auxInt - arg0
		{name: "MUL", argLength: 2, reg: gp21, asm: "MUL", commutative: true},     // arg0 * arg1
		{name: "HMUL", argLength: 2, reg: gp21, asm: "MULL", commutative: true},   // (arg0 * arg1) >> 32, signed
		{name: "HMULU", argLength: 2, reg: gp21, asm: "MULLU", commutative: true}, // (arg0 * arg1) >> 32, unsigned

		// udiv runtime call for soft division
		// output0 = arg0/arg1, output1 = arg0%arg1
		// see ../../../../../runtime/vlop_arm.s
		{
			name:      "CALLudiv",
			argLength: 2,
			reg: regInfo{
				inputs:   []regMask{buildReg("R1"), buildReg("R0")},
				outputs:  []regMask{buildReg("R0"), buildReg("R1")},
				clobbers: buildReg("R2 R3 R12 R14"), // R14 is LR, R12 is linker trampoline scratch register
			},
			clobberFlags: true,
			typ:          "(UInt32,UInt32)",
			call:         false, // TODO(mdempsky): Should this be true?
		},

		{name: "ADDS", argLength: 2, reg: gp21carry, asm: "ADD", commutative: true}, // arg0 + arg1, set carry flag
		{name: "ADDSconst", argLength: 1, reg: gp11carry, asm: "ADD", aux: "Int32"}, // arg0 + auxInt, set carry flag
		{name: "ADC", argLength: 3, reg: gp2flags1, asm: "ADC", commutative: true},  // arg0 + arg1 + carry, arg2=flags
		{name: "ADCconst", argLength: 2, reg: gp1flags1, asm: "ADC", aux: "Int32"},  // arg0 + auxInt + carry, arg1=flags
		{name: "SUBS", argLength: 2, reg: gp21carry, asm: "SUB"},                    // arg0 - arg1, set carry flag
		{name: "SUBSconst", argLength: 1, reg: gp11carry, asm: "SUB", aux: "Int32"}, // arg0 - auxInt, set carry flag
		{name: "RSBSconst", argLength: 1, reg: gp11carry, asm: "RSB", aux: "Int32"}, // auxInt - arg0, set carry flag
		{name: "SBC", argLength: 3, reg: gp2flags1, asm: "SBC"},                     // arg0 - arg1 - carry, arg2=flags
		{name: "SBCconst", argLength: 2, reg: gp1flags1, asm: "SBC", aux: "Int32"},  // arg0 - auxInt - carry, arg1=flags
		{name: "RSCconst", argLength: 2, reg: gp1flags1, asm: "RSC", aux: "Int32"},  // auxInt - arg0 - carry, arg1=flags

		{name: "MULLU", argLength: 2, reg: gp22, asm: "MULLU", commutative: true}, // arg0 * arg1, high 32 bits in out0, low 32 bits in out1
		{name: "MULA", argLength: 3, reg: gp31, asm: "MULA"},                      // arg0 * arg1 + arg2
		{name: "MULS", argLength: 3, reg: gp31, asm: "MULS"},                      // arg2 - arg0 * arg1

		{name: "ADDF", argLength: 2, reg: fp21, asm: "ADDF", commutative: true},   // arg0 + arg1
		{name: "ADDD", argLength: 2, reg: fp21, asm: "ADDD", commutative: true},   // arg0 + arg1
		{name: "SUBF", argLength: 2, reg: fp21, asm: "SUBF"},                      // arg0 - arg1
		{name: "SUBD", argLength: 2, reg: fp21, asm: "SUBD"},                      // arg0 - arg1
		{name: "MULF", argLength: 2, reg: fp21, asm: "MULF", commutative: true},   // arg0 * arg1
		{name: "MULD", argLength: 2, reg: fp21, asm: "MULD", commutative: true},   // arg0 * arg1
		{name: "NMULF", argLength: 2, reg: fp21, asm: "NMULF", commutative: true}, // -(arg0 * arg1)
		{name: "NMULD", argLength: 2, reg: fp21, asm: "NMULD", commutative: true}, // -(arg0 * arg1)
		{name: "DIVF", argLength: 2, reg: fp21, asm: "DIVF"},                      // arg0 / arg1
		{name: "DIVD", argLength: 2, reg: fp21, asm: "DIVD"},                      // arg0 / arg1

		{name: "MULAF", argLength: 3, reg: fp31, asm: "MULAF", resultInArg0: true}, // arg0 + (arg1 * arg2)
		{name: "MULAD", argLength: 3, reg: fp31, asm: "MULAD", resultInArg0: true}, // arg0 + (arg1 * arg2)
		{name: "MULSF", argLength: 3, reg: fp31, asm: "MULSF", resultInArg0: true}, // arg0 - (arg1 * arg2)
		{name: "MULSD", argLength: 3, reg: fp31, asm: "MULSD", resultInArg0: true}, // arg0 - (arg1 * arg2)

		// FMULAD only exists on platforms with the VFPv4 instruction set.
		// Any use must be preceded by a successful check of runtime.arm_support_vfpv4.
		{name: "FMULAD", argLength: 3, reg: fp31, asm: "FMULAD", resultInArg0: true}, // arg0 + (arg1 * arg2)

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true}, // arg0 & arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int32"}, // arg0 & auxInt
		{name: "OR", argLength: 2, reg: gp21, asm: "ORR", commutative: true},  // arg0 | arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "ORR", aux: "Int32"},  // arg0 | auxInt
		{name: "XOR", argLength: 2, reg: gp21, asm: "EOR", commutative: true}, // arg0 ^ arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "EOR", aux: "Int32"}, // arg0 ^ auxInt
		{name: "BIC", argLength: 2, reg: gp21, asm: "BIC"},                    // arg0 &^ arg1
		{name: "BICconst", argLength: 1, reg: gp11, asm: "BIC", aux: "Int32"}, // arg0 &^ auxInt

		// bit extraction, AuxInt = Width<<8 | LSB
		{name: "BFX", argLength: 1, reg: gp11, asm: "BFX", aux: "Int32"},   // extract W bits from bit L in arg0, then signed extend
		{name: "BFXU", argLength: 1, reg: gp11, asm: "BFXU", aux: "Int32"}, // extract W bits from bit L in arg0, then unsigned extend

		// unary ops
		{name: "MVN", argLength: 1, reg: gp11, asm: "MVN"}, // ^arg0

		{name: "NEGF", argLength: 1, reg: fp11, asm: "NEGF"},   // -arg0, float32
		{name: "NEGD", argLength: 1, reg: fp11, asm: "NEGD"},   // -arg0, float64
		{name: "SQRTD", argLength: 1, reg: fp11, asm: "SQRTD"}, // sqrt(arg0), float64
		{name: "SQRTF", argLength: 1, reg: fp11, asm: "SQRTF"}, // sqrt(arg0), float32
		{name: "ABSD", argLength: 1, reg: fp11, asm: "ABSD"},   // abs(arg0), float64

		{name: "CLZ", argLength: 1, reg: gp11, asm: "CLZ"},     // count leading zero
		{name: "REV", argLength: 1, reg: gp11, asm: "REV"},     // reverse byte order
		{name: "REV16", argLength: 1, reg: gp11, asm: "REV16"}, // reverse byte order in 16-bit halfwords
		{name: "RBIT", argLength: 1, reg: gp11, asm: "RBIT"},   // reverse bit order

		// shifts
		{name: "SLL", argLength: 2, reg: gp21, asm: "SLL"},                    // arg0 << arg1, shift amount is mod 256
		{name: "SLLconst", argLength: 1, reg: gp11, asm: "SLL", aux: "Int32"}, // arg0 << auxInt, 0 <= auxInt < 32
		{name: "SRL", argLength: 2, reg: gp21, asm: "SRL"},                    // arg0 >> arg1, unsigned, shift amount is mod 256
		{name: "SRLconst", argLength: 1, reg: gp11, asm: "SRL", aux: "Int32"}, // arg0 >> auxInt, unsigned, 0 <= auxInt < 32
		{name: "SRA", argLength: 2, reg: gp21, asm: "SRA"},                    // arg0 >> arg1, signed, shift amount is mod 256
		{name: "SRAconst", argLength: 1, reg: gp11, asm: "SRA", aux: "Int32"}, // arg0 >> auxInt, signed, 0 <= auxInt < 32
		{name: "SRR", argLength: 2, reg: gp21},                                // arg0 right rotate by arg1 bits
		{name: "SRRconst", argLength: 1, reg: gp11, aux: "Int32"},             // arg0 right rotate by auxInt bits, 0 <= auxInt < 32

		// auxInt for all of these satisfy 0 <= auxInt < 32
		{name: "ADDshiftLL", argLength: 2, reg: gp21, asm: "ADD", aux: "Int32"}, // arg0 + arg1<<auxInt
		{name: "ADDshiftRL", argLength: 2, reg: gp21, asm: "ADD", aux: "Int32"}, // arg0 + arg1>>auxInt, unsigned shift
		{name: "ADDshiftRA", argLength: 2, reg: gp21, asm: "ADD", aux: "Int32"}, // arg0 + arg1>>auxInt, signed shift
		{name: "SUBshiftLL", argLength: 2, reg: gp21, asm: "SUB", aux: "Int32"}, // arg0 - arg1<<auxInt
		{name: "SUBshiftRL", argLength: 2, reg: gp21, asm: "SUB", aux: "Int32"}, // arg0 - arg1>>auxInt, unsigned shift
		{name: "SUBshiftRA", argLength: 2, reg: gp21, asm: "SUB", aux: "Int32"}, // arg0 - arg1>>auxInt, signed shift
		{name: "RSBshiftLL", argLength: 2, reg: gp21, asm: "RSB", aux: "Int32"}, // arg1<<auxInt - arg0
		{name: "RSBshiftRL", argLength: 2, reg: gp21, asm: "RSB", aux: "Int32"}, // arg1>>auxInt - arg0, unsigned shift
		{name: "RSBshiftRA", argLength: 2, reg: gp21, asm: "RSB", aux: "Int32"}, // arg1>>auxInt - arg0, signed shift
		{name: "ANDshiftLL", argLength: 2, reg: gp21, asm: "AND", aux: "Int32"}, // arg0 & (arg1<<auxInt)
		{name: "ANDshiftRL", argLength: 2, reg: gp21, asm: "AND", aux: "Int32"}, // arg0 & (arg1>>auxInt), unsigned shift
		{name: "ANDshiftRA", argLength: 2, reg: gp21, asm: "AND", aux: "Int32"}, // arg0 & (arg1>>auxInt), signed shift
		{name: "ORshiftLL", argLength: 2, reg: gp21, asm: "ORR", aux: "Int32"},  // arg0 | arg1<<auxInt
		{name: "ORshiftRL", argLength: 2, reg: gp21, asm: "ORR", aux: "Int32"},  // arg0 | arg1>>auxInt, unsigned shift
		{name: "ORshiftRA", argLength: 2, reg: gp21, asm: "ORR", aux: "Int32"},  // arg0 | arg1>>auxInt, signed shift
		{name: "XORshiftLL", argLength: 2, reg: gp21, asm: "EOR", aux: "Int32"}, // arg0 ^ arg1<<auxInt
		{name: "XORshiftRL", argLength: 2, reg: gp21, asm: "EOR", aux: "Int32"}, // arg0 ^ arg1>>auxInt, unsigned shift
		{name: "XORshiftRA", argLength: 2, reg: gp21, asm: "EOR", aux: "Int32"}, // arg0 ^ arg1>>auxInt, signed shift
		{name: "XORshiftRR", argLength: 2, reg: gp21, asm: "EOR", aux: "Int32"}, // arg0 ^ (arg1 right rotate by auxInt)
		{name: "BICshiftLL", argLength: 2, reg: gp21, asm: "BIC", aux: "Int32"}, // arg0 &^ (arg1<<auxInt)
		{name: "BICshiftRL", argLength: 2, reg: gp21, asm: "BIC", aux: "Int32"}, // arg0 &^ (arg1>>auxInt), unsigned shift
		{name: "BICshiftRA", argLength: 2, reg: gp21, asm: "BIC", aux: "Int32"}, // arg0 &^ (arg1>>auxInt), signed shift
		{name: "MVNshiftLL", argLength: 1, reg: gp11, asm: "MVN", aux: "Int32"}, // ^(arg0<<auxInt)
		{name: "MVNshiftRL", argLength: 1, reg: gp11, asm: "MVN", aux: "Int32"}, // ^(arg0>>auxInt), unsigned shift
		{name: "MVNshiftRA", argLength: 1, reg: gp11, asm: "MVN", aux: "Int32"}, // ^(arg0>>auxInt), signed shift

		{name: "ADCshiftLL", argLength: 3, reg: gp2flags1, asm: "ADC", aux: "Int32"}, // arg0 + arg1<<auxInt + carry, arg2=flags
		{name: "ADCshiftRL", argLength: 3, reg: gp2flags1, asm: "ADC", aux: "Int32"}, // arg0 + arg1>>auxInt + carry, unsigned shift, arg2=flags
		{name: "ADCshiftRA", argLength: 3, reg: gp2flags1, asm: "ADC", aux: "Int32"}, // arg0 + arg1>>auxInt + carry, signed shift, arg2=flags
		{name: "SBCshiftLL", argLength: 3, reg: gp2flags1, asm: "SBC", aux: "Int32"}, // arg0 - arg1<<auxInt - carry, arg2=flags
		{name: "SBCshiftRL", argLength: 3, reg: gp2flags1, asm: "SBC", aux: "Int32"}, // arg0 - arg1>>auxInt - carry, unsigned shift, arg2=flags
		{name: "SBCshiftRA", argLength: 3, reg: gp2flags1, asm: "SBC", aux: "Int32"}, // arg0 - arg1>>auxInt - carry, signed shift, arg2=flags
		{name: "RSCshiftLL", argLength: 3, reg: gp2flags1, asm: "RSC", aux: "Int32"}, // arg1<<auxInt - arg0 - carry, arg2=flags
		{name: "RSCshiftRL", argLength: 3, reg: gp2flags1, asm: "RSC", aux: "Int32"}, // arg1>>auxInt - arg0 - carry, unsigned shift, arg2=flags
		{name: "RSCshiftRA", argLength: 3, reg: gp2flags1, asm: "RSC", aux: "Int32"}, // arg1>>auxInt - arg0 - carry, signed shift, arg2=flags

		{name: "ADDSshiftLL", argLength: 2, reg: gp21carry, asm: "ADD", aux: "Int32"}, // arg0 + arg1<<auxInt, set carry flag
		{name: "ADDSshiftRL", argLength: 2, reg: gp21carry, asm: "ADD", aux: "Int32"}, // arg0 + arg1>>auxInt, unsigned shift, set carry flag
		{name: "ADDSshiftRA", argLength: 2, reg: gp21carry, asm: "ADD", aux: "Int32"}, // arg0 + arg1>>auxInt, signed shift, set carry flag
		{name: "SUBSshiftLL", argLength: 2, reg: gp21carry, asm: "SUB", aux: "Int32"}, // arg0 - arg1<<auxInt, set carry flag
		{name: "SUBSshiftRL", argLength: 2, reg: gp21carry, asm: "SUB", aux: "Int32"}, // arg0 - arg1>>auxInt, unsigned shift, set carry flag
		{name: "SUBSshiftRA", argLength: 2, reg: gp21carry, asm: "SUB", aux: "Int32"}, // arg0 - arg1>>auxInt, signed shift, set carry flag
		{name: "RSBSshiftLL", argLength: 2, reg: gp21carry, asm: "RSB", aux: "Int32"}, // arg1<<auxInt - arg0, set carry flag
		{name: "RSBSshiftRL", argLength: 2, reg: gp21carry, asm: "RSB", aux: "Int32"}, // arg1>>auxInt - arg0, unsigned shift, set carry flag
		{name: "RSBSshiftRA", argLength: 2, reg: gp21carry, asm: "RSB", aux: "Int32"}, // arg1>>auxInt - arg0, signed shift, set carry flag

		{name: "ADDshiftLLreg", argLength: 3, reg: gp31, asm: "ADD"}, // arg0 + arg1<<arg2
		{name: "ADDshiftRLreg", argLength: 3, reg: gp31, asm: "ADD"}, // arg0 + arg1>>arg2, unsigned shift
		{name: "ADDshiftRAreg", argLength: 3, reg: gp31, asm: "ADD"}, // arg0 + arg1>>arg2, signed shift
		{name: "SUBshiftLLreg", argLength: 3, reg: gp31, asm: "SUB"}, // arg0 - arg1<<arg2
		{name: "SUBshiftRLreg", argLength: 3, reg: gp31, asm: "SUB"}, // arg0 - arg1>>arg2, unsigned shift
		{name: "SUBshiftRAreg", argLength: 3, reg: gp31, asm: "SUB"}, // arg0 - arg1>>arg2, signed shift
		{name: "RSBshiftLLreg", argLength: 3, reg: gp31, asm: "RSB"}, // arg1<<arg2 - arg0
		{name: "RSBshiftRLreg", argLength: 3, reg: gp31, asm: "RSB"}, // arg1>>arg2 - arg0, unsigned shift
		{name: "RSBshiftRAreg", argLength: 3, reg: gp31, asm: "RSB"}, // arg1>>arg2 - arg0, signed shift
		{name: "ANDshiftLLreg", argLength: 3, reg: gp31, asm: "AND"}, // arg0 & (arg1<<arg2)
		{name: "ANDshiftRLreg", argLength: 3, reg: gp31, asm: "AND"}, // arg0 & (arg1>>arg2), unsigned shift
		{name: "ANDshiftRAreg", argLength: 3, reg: gp31, asm: "AND"}, // arg0 & (arg1>>arg2), signed shift
		{name: "ORshiftLLreg", argLength: 3, reg: gp31, asm: "ORR"},  // arg0 | arg1<<arg2
		{name: "ORshiftRLreg", argLength: 3, reg: gp31, asm: "ORR"},  // arg0 | arg1>>arg2, unsigned shift
		{name: "ORshiftRAreg", argLength: 3, reg: gp31, asm: "ORR"},  // arg0 | arg1>>arg2, signed shift
		{name: "XORshiftLLreg", argLength: 3, reg: gp31, asm: "EOR"}, // arg0 ^ arg1<<arg2
		{name: "XORshiftRLreg", argLength: 3, reg: gp31, asm: "EOR"}, // arg0 ^ arg1>>arg2, unsigned shift
		{name: "XORshiftRAreg", argLength: 3, reg: gp31, asm: "EOR"}, // arg0 ^ arg1>>arg2, signed shift
		{name: "BICshiftLLreg", argLength: 3, reg: gp31, asm: "BIC"}, // arg0 &^ (arg1<<arg2)
		{name: "BICshiftRLreg", argLength: 3, reg: gp31, asm: "BIC"}, // arg0 &^ (arg1>>arg2), unsigned shift
		{name: "BICshiftRAreg", argLength: 3, reg: gp31, asm: "BIC"}, // arg0 &^ (arg1>>arg2), signed shift
		{name: "MVNshiftLLreg", argLength: 2, reg: gp21, asm: "MVN"}, // ^(arg0<<arg1)
		{name: "MVNshiftRLreg", argLength: 2, reg: gp21, asm: "MVN"}, // ^(arg0>>arg1), unsigned shift
		{name: "MVNshiftRAreg", argLength: 2, reg: gp21, asm: "MVN"}, // ^(arg0>>arg1), signed shift

		{name: "ADCshiftLLreg", argLength: 4, reg: gp3flags1, asm: "ADC"}, // arg0 + arg1<<arg2 + carry, arg3=flags
		{name: "ADCshiftRLreg", argLength: 4, reg: gp3flags1, asm: "ADC"}, // arg0 + arg1>>arg2 + carry, unsigned shift, arg3=flags
		{name: "ADCshiftRAreg", argLength: 4, reg: gp3flags1, asm: "ADC"}, // arg0 + arg1>>arg2 + carry, signed shift, arg3=flags
		{name: "SBCshiftLLreg", argLength: 4, reg: gp3flags1, asm: "SBC"}, // arg0 - arg1<<arg2 - carry, arg3=flags
		{name: "SBCshiftRLreg", argLength: 4, reg: gp3flags1, asm: "SBC"}, // arg0 - arg1>>arg2 - carry, unsigned shift, arg3=flags
		{name: "SBCshiftRAreg", argLength: 4, reg: gp3flags1, asm: "SBC"}, // arg0 - arg1>>arg2 - carry, signed shift, arg3=flags
		{name: "RSCshiftLLreg", argLength: 4, reg: gp3flags1, asm: "RSC"}, // arg1<<arg2 - arg0 - carry, arg3=flags
		{name: "RSCshiftRLreg", argLength: 4, reg: gp3flags1, asm: "RSC"}, // arg1>>arg2 - arg0 - carry, unsigned shift, arg3=flags
		{name: "RSCshiftRAreg", argLength: 4, reg: gp3flags1, asm: "RSC"}, // arg1>>arg2 - arg0 - carry, signed shift, arg3=flags

		{name: "ADDSshiftLLreg", argLength: 3, reg: gp31carry, asm: "ADD"}, // arg0 + arg1<<arg2, set carry flag
		{name: "ADDSshiftRLreg", argLength: 3, reg: gp31carry, asm: "ADD"}, // arg0 + arg1>>arg2, unsigned shift, set carry flag
		{name: "ADDSshiftRAreg", argLength: 3, reg: gp31carry, asm: "ADD"}, // arg0 + arg1>>arg2, signed shift, set carry flag
		{name: "SUBSshiftLLreg", argLength: 3, reg: gp31carry, asm: "SUB"}, // arg0 - arg1<<arg2, set carry flag
		{name: "SUBSshiftRLreg", argLength: 3, reg: gp31carry, asm: "SUB"}, // arg0 - arg1>>arg2, unsigned shift, set carry flag
		{name: "SUBSshiftRAreg", argLength: 3, reg: gp31carry, asm: "SUB"}, // arg0 - arg1>>arg2, signed shift, set carry flag
		{name: "RSBSshiftLLreg", argLength: 3, reg: gp31carry, asm: "RSB"}, // arg1<<arg2 - arg0, set carry flag
		{name: "RSBSshiftRLreg", argLength: 3, reg: gp31carry, asm: "RSB"}, // arg1>>arg2 - arg0, unsigned shift, set carry flag
		{name: "RSBSshiftRAreg", argLength: 3, reg: gp31carry, asm: "RSB"}, // arg1>>arg2 - arg0, signed shift, set carry flag

		// comparisons
		{name: "CMP", argLength: 2, reg: gp2flags, asm: "CMP", typ: "Flags"},                    // arg0 compare to arg1
		{name: "CMPconst", argLength: 1, reg: gp1flags, asm: "CMP", aux: "Int32", typ: "Flags"}, // arg0 compare to auxInt
		{name: "CMN", argLength: 2, reg: gp2flags, asm: "CMN", typ: "Flags", commutative: true}, // arg0 compare to -arg1, provided arg1 is not 1<<63
		{name: "CMNconst", argLength: 1, reg: gp1flags, asm: "CMN", aux: "Int32", typ: "Flags"}, // arg0 compare to -auxInt
		{name: "TST", argLength: 2, reg: gp2flags, asm: "TST", typ: "Flags", commutative: true}, // arg0 & arg1 compare to 0
		{name: "TSTconst", argLength: 1, reg: gp1flags, asm: "TST", aux: "Int32", typ: "Flags"}, // arg0 & auxInt compare to 0
		{name: "TEQ", argLength: 2, reg: gp2flags, asm: "TEQ", typ: "Flags", commutative: true}, // arg0 ^ arg1 compare to 0
		{name: "TEQconst", argLength: 1, reg: gp1flags, asm: "TEQ", aux: "Int32", typ: "Flags"}, // arg0 ^ auxInt compare to 0
		{name: "CMPF", argLength: 2, reg: fp2flags, asm: "CMPF", typ: "Flags"},                  // arg0 compare to arg1, float32
		{name: "CMPD", argLength: 2, reg: fp2flags, asm: "CMPD", typ: "Flags"},                  // arg0 compare to arg1, float64

		{name: "CMPshiftLL", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int32", typ: "Flags"}, // arg0 compare to arg1<<auxInt
		{name: "CMPshiftRL", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int32", typ: "Flags"}, // arg0 compare to arg1>>auxInt, unsigned shift
		{name: "CMPshiftRA", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int32", typ: "Flags"}, // arg0 compare to arg1>>auxInt, signed shift
		{name: "CMNshiftLL", argLength: 2, reg: gp2flags, asm: "CMN", aux: "Int32", typ: "Flags"}, // arg0 compare to -(arg1<<auxInt)
		{name: "CMNshiftRL", argLength: 2, reg: gp2flags, asm: "CMN", aux: "Int32", typ: "Flags"}, // arg0 compare to -(arg1>>auxInt), unsigned shift
		{name: "CMNshiftRA", argLength: 2, reg: gp2flags, asm: "CMN", aux: "Int32", typ: "Flags"}, // arg0 compare to -(arg1>>auxInt), signed shift
		{name: "TSTshiftLL", argLength: 2, reg: gp2flags, asm: "TST", aux: "Int32", typ: "Flags"}, // arg0 & (arg1<<auxInt) compare to 0
		{name: "TSTshiftRL", argLength: 2, reg: gp2flags, asm: "TST", aux: "Int32", typ: "Flags"}, // arg0 & (arg1>>auxInt) compare to 0, unsigned shift
		{name: "TSTshiftRA", argLength: 2, reg: gp2flags, asm: "TST", aux: "Int32", typ: "Flags"}, // arg0 & (arg1>>auxInt) compare to 0, signed shift
		{name: "TEQshiftLL", argLength: 2, reg: gp2flags, asm: "TEQ", aux: "Int32", typ: "Flags"}, // arg0 ^ (arg1<<auxInt) compare to 0
		{name: "TEQshiftRL", argLength: 2, reg: gp2flags, asm: "TEQ", aux: "Int32", typ: "Flags"}, // arg0 ^ (arg1>>auxInt) compare to 0, unsigned shift
		{name: "TEQshiftRA", argLength: 2, reg: gp2flags, asm: "TEQ", aux: "Int32", typ: "Flags"}, // arg0 ^ (arg1>>auxInt) compare to 0, signed shift

		{name: "CMPshiftLLreg", argLength: 3, reg: gp3flags, asm: "CMP", typ: "Flags"}, // arg0 compare to arg1<<arg2
		{name: "CMPshiftRLreg", argLength: 3, reg: gp3flags, asm: "CMP", typ: "Flags"}, // arg0 compare to arg1>>arg2, unsigned shift
		{name: "CMPshiftRAreg", argLength: 3, reg: gp3flags, asm: "CMP", typ: "Flags"}, // arg0 compare to arg1>>arg2, signed shift
		{name: "CMNshiftLLreg", argLength: 3, reg: gp3flags, asm: "CMN", typ: "Flags"}, // arg0 + (arg1<<arg2) compare to 0
		{name: "CMNshiftRLreg", argLength: 3, reg: gp3flags, asm: "CMN", typ: "Flags"}, // arg0 + (arg1>>arg2) compare to 0, unsigned shift
		{name: "CMNshiftRAreg", argLength: 3, reg: gp3flags, asm: "CMN", typ: "Flags"}, // arg0 + (arg1>>arg2) compare to 0, signed shift
		{name: "TSTshiftLLreg", argLength: 3, reg: gp3flags, asm: "TST", typ: "Flags"}, // arg0 & (arg1<<arg2) compare to 0
		{name: "TSTshiftRLreg", argLength: 3, reg: gp3flags, asm: "TST", typ: "Flags"}, // arg0 & (arg1>>arg2) compare to 0, unsigned shift
		{name: "TSTshiftRAreg", argLength: 3, reg: gp3flags, asm: "TST", typ: "Flags"}, // arg0 & (arg1>>arg2) compare to 0, signed shift
		{name: "TEQshiftLLreg", argLength: 3, reg: gp3flags, asm: "TEQ", typ: "Flags"}, // arg0 ^ (arg1<<arg2) compare to 0
		{name: "TEQshiftRLreg", argLength: 3, reg: gp3flags, asm: "TEQ", typ: "Flags"}, // arg0 ^ (arg1>>arg2) compare to 0, unsigned shift
		{name: "TEQshiftRAreg", argLength: 3, reg: gp3flags, asm: "TEQ", typ: "Flags"}, // arg0 ^ (arg1>>arg2) compare to 0, signed shift

		{name: "CMPF0", argLength: 1, reg: fp1flags, asm: "CMPF", typ: "Flags"}, // arg0 compare to 0, float32
		{name: "CMPD0", argLength: 1, reg: fp1flags, asm: "CMPD", typ: "Flags"}, // arg0 compare to 0, float64

		// moves
		{name: "MOVWconst", argLength: 0, reg: gp01, aux: "Int32", asm: "MOVW", typ: "UInt32", rematerializeable: true},    // 32 low bits of auxint
		{name: "MOVFconst", argLength: 0, reg: fp01, aux: "Float64", asm: "MOVF", typ: "Float32", rematerializeable: true}, // auxint as 64-bit float, convert to 32-bit float
		{name: "MOVDconst", argLength: 0, reg: fp01, aux: "Float64", asm: "MOVD", typ: "Float64", rematerializeable: true}, // auxint as 64-bit float

		{name: "MOVWaddr", argLength: 1, reg: regInfo{inputs: []regMask{buildReg("SP") | buildReg("SB")}, outputs: []regMask{gp}}, aux: "SymOff", asm: "MOVW", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxInt + aux.(*gc.Sym), arg0=SP/SB

		{name: "MOVBload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVB", typ: "Int8", faultOnNilArg0: true, symEffect: "Read"},     // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVBUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVBU", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVH", typ: "Int16", faultOnNilArg0: true, symEffect: "Read"},    // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVHU", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"}, // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVW", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"},   // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVFload", argLength: 2, reg: fpload, aux: "SymOff", asm: "MOVF", typ: "Float32", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVDload", argLength: 2, reg: fpload, aux: "SymOff", asm: "MOVD", typ: "Float64", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.

		{name: "MOVBstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 1 byte of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVHstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVWstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVFstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "MOVF", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVDstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "MOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.

		{name: "MOVWloadidx", argLength: 3, reg: gp2load, asm: "MOVW", typ: "UInt32"},                   // load from arg0 + arg1. arg2=mem
		{name: "MOVWloadshiftLL", argLength: 3, reg: gp2load, asm: "MOVW", aux: "Int32", typ: "UInt32"}, // load from arg0 + arg1<<auxInt. arg2=mem
		{name: "MOVWloadshiftRL", argLength: 3, reg: gp2load, asm: "MOVW", aux: "Int32", typ: "UInt32"}, // load from arg0 + arg1>>auxInt, unsigned shift. arg2=mem
		{name: "MOVWloadshiftRA", argLength: 3, reg: gp2load, asm: "MOVW", aux: "Int32", typ: "UInt32"}, // load from arg0 + arg1>>auxInt, signed shift. arg2=mem
		{name: "MOVBUloadidx", argLength: 3, reg: gp2load, asm: "MOVBU", typ: "UInt8"},                  // load from arg0 + arg1. arg2=mem
		{name: "MOVBloadidx", argLength: 3, reg: gp2load, asm: "MOVB", typ: "Int8"},                     // load from arg0 + arg1. arg2=mem
		{name: "MOVHUloadidx", argLength: 3, reg: gp2load, asm: "MOVHU", typ: "UInt16"},                 // load from arg0 + arg1. arg2=mem
		{name: "MOVHloadidx", argLength: 3, reg: gp2load, asm: "MOVH", typ: "Int16"},                    // load from arg0 + arg1. arg2=mem

		{name: "MOVWstoreidx", argLength: 4, reg: gp2store, asm: "MOVW", typ: "Mem"},                   // store arg2 to arg0 + arg1. arg3=mem
		{name: "MOVWstoreshiftLL", argLength: 4, reg: gp2store, asm: "MOVW", aux: "Int32", typ: "Mem"}, // store arg2 to arg0 + arg1<<auxInt. arg3=mem
		{name: "MOVWstoreshiftRL", argLength: 4, reg: gp2store, asm: "MOVW", aux: "Int32", typ: "Mem"}, // store arg2 to arg0 + arg1>>auxInt, unsigned shift. arg3=mem
		{name: "MOVWstoreshiftRA", argLength: 4, reg: gp2store, asm: "MOVW", aux: "Int32", typ: "Mem"}, // store arg2 to arg0 + arg1>>auxInt, signed shift. arg3=mem
		{name: "MOVBstoreidx", argLength: 4, reg: gp2store, asm: "MOVB", typ: "Mem"},                   // store arg2 to arg0 + arg1. arg3=mem
		{name: "MOVHstoreidx", argLength: 4, reg: gp2store, asm: "MOVH", typ: "Mem"},                   // store arg2 to arg0 + arg1. arg3=mem

		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVBS"},  // move from arg0, sign-extended from byte
		{name: "MOVBUreg", argLength: 1, reg: gp11, asm: "MOVBU"}, // move from arg0, unsign-extended from byte
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVHS"},  // move from arg0, sign-extended from half
		{name: "MOVHUreg", argLength: 1, reg: gp11, asm: "MOVHU"}, // move from arg0, unsign-extended from half
		{name: "MOVWreg", argLength: 1, reg: gp11, asm: "MOVW"},   // move from arg0

		{name: "MOVWnop", argLength: 1, reg: regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}, resultInArg0: true}, // nop, return arg0 in same register

		{name: "MOVWF", argLength: 1, reg: gpfp, asm: "MOVWF"},  // int32 -> float32
		{name: "MOVWD", argLength: 1, reg: gpfp, asm: "MOVWD"},  // int32 -> float64
		{name: "MOVWUF", argLength: 1, reg: gpfp, asm: "MOVWF"}, // uint32 -> float32, set U bit in the instruction
		{name: "MOVWUD", argLength: 1, reg: gpfp, asm: "MOVWD"}, // uint32 -> float64, set U bit in the instruction
		{name: "MOVFW", argLength: 1, reg: fpgp, asm: "MOVFW"},  // float32 -> int32
		{name: "MOVDW", argLength: 1, reg: fpgp, asm: "MOVDW"},  // float64 -> int32
		{name: "MOVFWU", argLength: 1, reg: fpgp, asm: "MOVFW"}, // float32 -> uint32, set U bit in the instruction
		{name: "MOVDWU", argLength: 1, reg: fpgp, asm: "MOVDW"}, // float64 -> uint32, set U bit in the instruction
		{name: "MOVFD", argLength: 1, reg: fp11, asm: "MOVFD"},  // float32 -> float64
		{name: "MOVDF", argLength: 1, reg: fp11, asm: "MOVDF"},  // float64 -> float32

		// conditional instructions, for lowering shifts
		{name: "CMOVWHSconst", argLength: 2, reg: gp1flags1, asm: "MOVW", aux: "Int32", resultInArg0: true}, // replace arg0 w/ const if flags indicates HS, arg1=flags
		{name: "CMOVWLSconst", argLength: 2, reg: gp1flags1, asm: "MOVW", aux: "Int32", resultInArg0: true}, // replace arg0 w/ const if flags indicates LS, arg1=flags
		{name: "SRAcond", argLength: 3, reg: gp2flags1, asm: "SRA"},                                         // arg0 >> 31 if flags indicates HS, arg0 >> arg1 otherwise, signed shift, arg2=flags

		// function calls
		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                              // call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                // tail call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{inputs: []regMask{gpsp, buildReg("R7"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                        // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		// pseudo-ops
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gpg}}, nilCheck: true, faultOnNilArg0: true}, // panic if arg0 is nil.  arg1=mem.

		{name: "Equal", argLength: 1, reg: readflags},         // bool, true flags encode x==y false otherwise.
		{name: "NotEqual", argLength: 1, reg: readflags},      // bool, true flags encode x!=y false otherwise.
		{name: "LessThan", argLength: 1, reg: readflags},      // bool, true flags encode signed x<y false otherwise.
		{name: "LessEqual", argLength: 1, reg: readflags},     // bool, true flags encode signed x<=y false otherwise.
		{name: "GreaterThan", argLength: 1, reg: readflags},   // bool, true flags encode signed x>y false otherwise.
		{name: "GreaterEqual", argLength: 1, reg: readflags},  // bool, true flags encode signed x>=y false otherwise.
		{name: "LessThanU", argLength: 1, reg: readflags},     // bool, true flags encode unsigned x<y false otherwise.
		{name: "LessEqualU", argLength: 1, reg: readflags},    // bool, true flags encode unsigned x<=y false otherwise.
		{name: "GreaterThanU", argLength: 1, reg: readflags},  // bool, true flags encode unsigned x>y false otherwise.
		{name: "GreaterEqualU", argLength: 1, reg: readflags}, // bool, true flags encode unsigned x>=y false otherwise.

		// duffzero (must be 4-byte aligned)
		// arg0 = address of memory to zero (in R1, changed as side effect)
		// arg1 = value to store (always zero)
		// arg2 = mem
		// auxint = offset into duffzero code to start executing
		// returns mem
		{
			name:      "DUFFZERO",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R1"), buildReg("R0")},
				clobbers: buildReg("R1 R12 R14"), // R14 is LR, R12 is linker trampoline scratch register
			},
			faultOnNilArg0: true,
		},

		// duffcopy (must be 4-byte aligned)
		// arg0 = address of dst memory (in R2, changed as side effect)
		// arg1 = address of src memory (in R1, changed as side effect)
		// arg2 = mem
		// auxint = offset into duffcopy code to start executing
		// returns mem
		{
			name:      "DUFFCOPY",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R2"), buildReg("R1")},
				clobbers: buildReg("R0 R1 R2 R12 R14"), // R14 is LR, R12 is linker trampoline scratch register
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// large or unaligned zeroing
		// arg0 = address of memory to zero (in R1, changed as side effect)
		// arg1 = address of the last element to zero
		// arg2 = value to store (always zero)
		// arg3 = mem
		// returns mem
		//	MOVW.P	Rarg2, 4(R1)
		//	CMP	R1, Rarg1
		//	BLE	-2(PC)
		{
			name:      "LoweredZero",
			aux:       "Int64",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("R1"), gp, gp},
				clobbers: buildReg("R1"),
			},
			clobberFlags:   true,
			faultOnNilArg0: true,
		},

		// large or unaligned move
		// arg0 = address of dst memory (in R2, changed as side effect)
		// arg1 = address of src memory (in R1, changed as side effect)
		// arg2 = address of the last element of src
		// arg3 = mem
		// returns mem
		//	MOVW.P	4(R1), Rtmp
		//	MOVW.P	Rtmp, 4(R2)
		//	CMP	R1, Rarg2
		//	BLE	-3(PC)
		{
			name:      "LoweredMove",
			aux:       "Int64",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("R2"), buildReg("R1"), gp},
				clobbers: buildReg("R1 R2"),
			},
			clobberFlags:   true,
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of R7 (arm.REGCTXT, the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("R7")}}, zeroWidth: true},

		// LoweredGetCallerSP returns the SP of the caller of the current function.
		{name: "LoweredGetCallerSP", reg: gp01, rematerializeable: true},

		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" getcallerpc,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},

		// There are three of these functions so that they can have three different register inputs.
		// When we check 0 <= c <= cap (A), then 0 <= b <= c (B), then 0 <= a <= b (C), we want the
		// default registers to match so we don't need to copy registers around unnecessarily.
		{name: "LoweredPanicBoundsA", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r2, r3}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsB", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r1, r2}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsC", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r0, r1}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		// Extend ops are the same as Bounds ops except the indexes are 64-bit.
		{name: "LoweredPanicExtendA", argLength: 4, aux: "Int64", reg: regInfo{inputs: []regMask{r4, r2, r3}}, typ: "Mem", call: true}, // arg0=idxHi, arg1=idxLo, arg2=len, arg3=mem, returns memory. AuxInt contains report code (see PanicExtend in genericOps.go).
		{name: "LoweredPanicExtendB", argLength: 4, aux: "Int64", reg: regInfo{inputs: []regMask{r4, r1, r2}}, typ: "Mem", call: true}, // arg0=idxHi, arg1=idxLo, arg2=len, arg3=mem, returns memory. AuxInt contains report code (see PanicExtend in genericOps.go).
		{name: "LoweredPanicExtendC", argLength: 4, aux: "Int64", reg: regInfo{inputs: []regMask{r4, r0, r1}}, typ: "Mem", call: true}, // arg0=idxHi, arg1=idxLo, arg2=len, arg3=mem, returns memory. AuxInt contains report code (see PanicExtend in genericOps.go).

		// Constant flag value.
		// Note: there's an "unordered" outcome for floating-point
		// comparisons, but we don't use such a beast yet.
		// This op is for temporary use by rewrite rules. It
		// cannot appear in the generated assembly.
		{name: "FlagConstant", aux: "FlagConstant"},

		// (InvertFlags (CMP a b)) == (CMP b a)
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags", argLength: 1}, // reverse direction of arg0

		// LoweredWB invokes runtime.gcWriteBarrier. arg0=destptr, arg1=srcptr, arg2=mem, aux=runtime.gcWriteBarrier
		// It saves all GP registers if necessary,
		// but clobbers R14 (LR) because it's a call, and R12 which is linker trampoline scratch register.
		{name: "LoweredWB", argLength: 3, reg: regInfo{inputs: []regMask{buildReg("R2"), buildReg("R3")}, clobbers: (callerSave &^ gpg) | buildReg("R12 R14")}, clobberFlags: true, aux: "Sym", symEffect: "None"},
	}

	blocks := []blockData{
		{name: "EQ", controls: 1},
		{name: "NE", controls: 1},
		{name: "LT", controls: 1},
		{name: "LE", controls: 1},
		{name: "GT", controls: 1},
		{name: "GE", controls: 1},
		{name: "ULT", controls: 1},
		{name: "ULE", controls: 1},
		{name: "UGT", controls: 1},
		{name: "UGE", controls: 1},
		{name: "LTnoov", controls: 1}, // 'LT' but without honoring overflow
		{name: "LEnoov", controls: 1}, // 'LE' but without honoring overflow
		{name: "GTnoov", controls: 1}, // 'GT' but without honoring overflow
		{name: "GEnoov", controls: 1}, // 'GE' but without honoring overflow
	}

	archs = append(archs, arch{
		name:            "ARM",
		pkg:             "cmd/internal/obj/arm",
		genfile:         "../../arm/ssa.go",
		ops:             ops,
		blocks:          blocks,
		regnames:        regNamesARM,
		gpregmask:       gp,
		fpregmask:       fp,
		framepointerreg: -1, // not used
		linkreg:         int8(num["R14"]),
	})
}
