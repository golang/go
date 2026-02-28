// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// Notes:
//  - Integer types live in the low portion of registers. Upper portions are junk.
//  - Boolean types use the low-order byte of a register. 0=false, 1=true.
//    Upper bytes are junk.
//  - When doing sub-register operations, we try to write the whole
//    destination register to avoid a partial-register write.
//  - Unused portions of AuxInt (or the Val portion of ValAndOff) are
//    filled by sign-extending the used portion. Users of AuxInt which interpret
//    AuxInt as unsigned (e.g. shifts) must be careful.
//  - The SB 'register' is implemented using instruction-relative addressing. This
//    places some limitations on when and how memory operands that are addressed
//    relative to SB can be used:
//
//     1. Pseudo-instructions do not always map to a single machine instruction when
//        using the SB 'register' to address data. This is because many machine
//        instructions do not have relative long (RL suffix) equivalents. For example,
//        ADDload, which is assembled as AG.
//
//     2. Loads and stores using relative addressing require the data be aligned
//        according to its size (8-bytes for double words, 4-bytes for words
//        and so on).
//
//    We can always work around these by inserting LARL instructions (load address
//    relative long) in the assembler, but typically this results in worse code
//    generation because the address can't be re-used. Inserting instructions in the
//    assembler also means clobbering the temp register and it is a long-term goal
//    to prevent the compiler doing this so that it can be allocated as a normal
//    register.
//
// For more information about the z/Architecture, the instruction set and the
// addressing modes it supports take a look at the z/Architecture Principles of
// Operation: http://publibfp.boulder.ibm.com/epubs/pdf/dz9zr010.pdf
//
// Suffixes encode the bit width of pseudo-instructions.
// D (double word)  = 64 bit (frequently omitted)
// W (word)         = 32 bit
// H (half word)    = 16 bit
// B (byte)         = 8 bit
// S (single prec.) = 32 bit (double precision is omitted)

// copied from ../../s390x/reg.go
var regNamesS390X = []string{
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
	"R10",
	"R11",
	"R12",
	"g", // R13
	"R14",
	"SP", // R15
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
	"F15",

	// If you add registers, update asyncPreempt in runtime.

	//pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesS390X) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesS390X {
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
		sp  = buildReg("SP")
		sb  = buildReg("SB")
		r0  = buildReg("R0")
		tmp = buildReg("R11") // R11 is used as a temporary in a small number of instructions.
		lr  = buildReg("R14")

		// R10 is reserved by the assembler.
		gp   = buildReg("R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R11 R12 R14")
		gpg  = gp | buildReg("g")
		gpsp = gp | sp

		// R0 is considered to contain the value 0 in address calculations.
		ptr     = gp &^ r0
		ptrsp   = ptr | sp
		ptrspsb = ptrsp | sb

		fp         = buildReg("F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15")
		callerSave = gp | fp | buildReg("g") // runtime.setg (and anything calling it) may clobber g
		r1         = buildReg("R1")
		r2         = buildReg("R2")
		r3         = buildReg("R3")
		r9         = buildReg("R9")
	)
	// Common slices of register masks
	var (
		gponly = []regMask{gp}
		fponly = []regMask{fp}
	)

	// Common regInfo
	var (
		gp01    = regInfo{inputs: []regMask{}, outputs: gponly}
		gp11    = regInfo{inputs: []regMask{gp}, outputs: gponly}
		gp11sp  = regInfo{inputs: []regMask{gpsp}, outputs: gponly}
		gp21    = regInfo{inputs: []regMask{gp, gp}, outputs: gponly}
		gp21sp  = regInfo{inputs: []regMask{gpsp, gp}, outputs: gponly}
		gp21tmp = regInfo{inputs: []regMask{gp &^ tmp, gp &^ tmp}, outputs: []regMask{gp &^ tmp}, clobbers: tmp}

		// R0 evaluates to 0 when used as the number of bits to shift
		// so we need to exclude it from that operand.
		sh21 = regInfo{inputs: []regMask{gp, ptr}, outputs: gponly}

		addr    = regInfo{inputs: []regMask{sp | sb}, outputs: gponly}
		addridx = regInfo{inputs: []regMask{sp | sb, ptrsp}, outputs: gponly}

		gp2flags       = regInfo{inputs: []regMask{gpsp, gpsp}}
		gp1flags       = regInfo{inputs: []regMask{gpsp}}
		gp2flags1      = regInfo{inputs: []regMask{gp, gp}, outputs: gponly}
		gp11flags      = regInfo{inputs: []regMask{gp}, outputs: gponly}
		gp21flags      = regInfo{inputs: []regMask{gp, gp}, outputs: gponly}
		gp2flags1flags = regInfo{inputs: []regMask{gp, gp}, outputs: gponly}

		gpload       = regInfo{inputs: []regMask{ptrspsb, 0}, outputs: gponly}
		gploadidx    = regInfo{inputs: []regMask{ptrspsb, ptrsp, 0}, outputs: gponly}
		gpopload     = regInfo{inputs: []regMask{gp, ptrsp, 0}, outputs: gponly}
		gpstore      = regInfo{inputs: []regMask{ptrspsb, gpsp, 0}}
		gpstoreconst = regInfo{inputs: []regMask{ptrspsb, 0}}
		gpstoreidx   = regInfo{inputs: []regMask{ptrsp, ptrsp, gpsp, 0}}
		gpstorebr    = regInfo{inputs: []regMask{ptrsp, gpsp, 0}}
		gpstorelaa   = regInfo{inputs: []regMask{ptrspsb, gpsp, 0}, outputs: gponly}
		gpstorelab   = regInfo{inputs: []regMask{r1, gpsp, 0}, clobbers: r1}

		gpmvc = regInfo{inputs: []regMask{ptrsp, ptrsp, 0}}

		fp01        = regInfo{inputs: []regMask{}, outputs: fponly}
		fp21        = regInfo{inputs: []regMask{fp, fp}, outputs: fponly}
		fp31        = regInfo{inputs: []regMask{fp, fp, fp}, outputs: fponly}
		fp21clobber = regInfo{inputs: []regMask{fp, fp}, outputs: fponly}
		fpgp        = regInfo{inputs: fponly, outputs: gponly}
		gpfp        = regInfo{inputs: gponly, outputs: fponly}
		fp11        = regInfo{inputs: fponly, outputs: fponly}
		fp1flags    = regInfo{inputs: []regMask{fp}}
		fp11clobber = regInfo{inputs: fponly, outputs: fponly}
		fp2flags    = regInfo{inputs: []regMask{fp, fp}}

		fpload    = regInfo{inputs: []regMask{ptrspsb, 0}, outputs: fponly}
		fploadidx = regInfo{inputs: []regMask{ptrsp, ptrsp, 0}, outputs: fponly}

		fpstore    = regInfo{inputs: []regMask{ptrspsb, fp, 0}}
		fpstoreidx = regInfo{inputs: []regMask{ptrsp, ptrsp, fp, 0}}

		sync = regInfo{inputs: []regMask{0}}

		// LoweredAtomicCas may overwrite arg1, so force it to R0 for now.
		cas = regInfo{inputs: []regMask{ptrsp, r0, gpsp, 0}, outputs: []regMask{gp, 0}, clobbers: r0}

		// LoweredAtomicExchange overwrites the output before executing
		// CS{,G}, so the output register must not be the same as the
		// input register. For now we just force the output register to
		// R0.
		exchange = regInfo{inputs: []regMask{ptrsp, gpsp &^ r0, 0}, outputs: []regMask{r0, 0}}
	)

	var S390Xops = []opData{
		// fp ops
		{name: "FADDS", argLength: 2, reg: fp21clobber, typ: "(Float32,Flags)", asm: "FADDS", commutative: true, resultInArg0: true}, // fp32 arg0 + arg1
		{name: "FADD", argLength: 2, reg: fp21clobber, typ: "(Float64,Flags)", asm: "FADD", commutative: true, resultInArg0: true},   // fp64 arg0 + arg1
		{name: "FSUBS", argLength: 2, reg: fp21clobber, typ: "(Float32,Flags)", asm: "FSUBS", resultInArg0: true},                    // fp32 arg0 - arg1
		{name: "FSUB", argLength: 2, reg: fp21clobber, typ: "(Float64,Flags)", asm: "FSUB", resultInArg0: true},                      // fp64 arg0 - arg1
		{name: "FMULS", argLength: 2, reg: fp21, asm: "FMULS", commutative: true, resultInArg0: true},                                // fp32 arg0 * arg1
		{name: "FMUL", argLength: 2, reg: fp21, asm: "FMUL", commutative: true, resultInArg0: true},                                  // fp64 arg0 * arg1
		{name: "FDIVS", argLength: 2, reg: fp21, asm: "FDIVS", resultInArg0: true},                                                   // fp32 arg0 / arg1
		{name: "FDIV", argLength: 2, reg: fp21, asm: "FDIV", resultInArg0: true},                                                     // fp64 arg0 / arg1
		{name: "FNEGS", argLength: 1, reg: fp11clobber, asm: "FNEGS", clobberFlags: true},                                            // fp32 -arg0
		{name: "FNEG", argLength: 1, reg: fp11clobber, asm: "FNEG", clobberFlags: true},                                              // fp64 -arg0
		{name: "FMADDS", argLength: 3, reg: fp31, asm: "FMADDS", resultInArg0: true},                                                 // fp32 arg1 * arg2 + arg0
		{name: "FMADD", argLength: 3, reg: fp31, asm: "FMADD", resultInArg0: true},                                                   // fp64 arg1 * arg2 + arg0
		{name: "FMSUBS", argLength: 3, reg: fp31, asm: "FMSUBS", resultInArg0: true},                                                 // fp32 arg1 * arg2 - arg0
		{name: "FMSUB", argLength: 3, reg: fp31, asm: "FMSUB", resultInArg0: true},                                                   // fp64 arg1 * arg2 - arg0
		{name: "LPDFR", argLength: 1, reg: fp11, asm: "LPDFR"},                                                                       // fp64/fp32 set sign bit
		{name: "LNDFR", argLength: 1, reg: fp11, asm: "LNDFR"},                                                                       // fp64/fp32 clear sign bit
		{name: "CPSDR", argLength: 2, reg: fp21, asm: "CPSDR"},                                                                       // fp64/fp32 copy arg1 sign bit to arg0

		// Single element vector floating point min / max instructions
		{name: "WFMAXDB", argLength: 2, reg: fp21, asm: "WFMAXDB", typ: "Float64"}, // max[float64](arg0, arg1)
		{name: "WFMAXSB", argLength: 2, reg: fp21, asm: "WFMAXSB", typ: "Float32"}, // max[float32](arg0, arg1)
		{name: "WFMINDB", argLength: 2, reg: fp21, asm: "WFMINDB", typ: "Float64"}, // min[float64](arg0, arg1)
		{name: "WFMINSB", argLength: 2, reg: fp21, asm: "WFMINSB", typ: "Float32"}, // min[float32](arg0, arg1)

		// Round to integer, float64 only.
		//
		// aux | rounding mode
		// ----+-----------------------------------
		//   1 | round to nearest, ties away from 0
		//   4 | round to nearest, ties to even
		//   5 | round toward 0
		//   6 | round toward +∞
		//   7 | round toward -∞
		{name: "FIDBR", argLength: 1, reg: fp11, asm: "FIDBR", aux: "Int8"},

		{name: "FMOVSload", argLength: 2, reg: fpload, asm: "FMOVS", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"}, // fp32 load
		{name: "FMOVDload", argLength: 2, reg: fpload, asm: "FMOVD", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"}, // fp64 load
		{name: "FMOVSconst", reg: fp01, asm: "FMOVS", aux: "Float32", rematerializeable: true},                               // fp32 constant
		{name: "FMOVDconst", reg: fp01, asm: "FMOVD", aux: "Float64", rematerializeable: true},                               // fp64 constant
		{name: "FMOVSloadidx", argLength: 3, reg: fploadidx, asm: "FMOVS", aux: "SymOff", symEffect: "Read"},                 // fp32 load indexed by i
		{name: "FMOVDloadidx", argLength: 3, reg: fploadidx, asm: "FMOVD", aux: "SymOff", symEffect: "Read"},                 // fp64 load indexed by i

		{name: "FMOVSstore", argLength: 3, reg: fpstore, asm: "FMOVS", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"}, // fp32 store
		{name: "FMOVDstore", argLength: 3, reg: fpstore, asm: "FMOVD", aux: "SymOff", faultOnNilArg0: true, symEffect: "Write"}, // fp64 store
		{name: "FMOVSstoreidx", argLength: 4, reg: fpstoreidx, asm: "FMOVS", aux: "SymOff", symEffect: "Write"},                 // fp32 indexed by i store
		{name: "FMOVDstoreidx", argLength: 4, reg: fpstoreidx, asm: "FMOVD", aux: "SymOff", symEffect: "Write"},                 // fp64 indexed by i store

		// binary ops
		{name: "ADD", argLength: 2, reg: gp21sp, asm: "ADD", commutative: true, clobberFlags: true},                                                                  // arg0 + arg1
		{name: "ADDW", argLength: 2, reg: gp21sp, asm: "ADDW", commutative: true, clobberFlags: true},                                                                // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11sp, asm: "ADD", aux: "Int32", typ: "UInt64", clobberFlags: true},                                                   // arg0 + auxint
		{name: "ADDWconst", argLength: 1, reg: gp11sp, asm: "ADDW", aux: "Int32", clobberFlags: true},                                                                // arg0 + auxint
		{name: "ADDload", argLength: 3, reg: gpopload, asm: "ADD", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},   // arg0 + *arg1. arg2=mem
		{name: "ADDWload", argLength: 3, reg: gpopload, asm: "ADDW", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 + *arg1. arg2=mem

		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB", clobberFlags: true},                                                                                       // arg0 - arg1
		{name: "SUBW", argLength: 2, reg: gp21, asm: "SUBW", clobberFlags: true},                                                                                     // arg0 - arg1
		{name: "SUBconst", argLength: 1, reg: gp11, asm: "SUB", aux: "Int32", resultInArg0: true, clobberFlags: true},                                                // arg0 - auxint
		{name: "SUBWconst", argLength: 1, reg: gp11, asm: "SUBW", aux: "Int32", resultInArg0: true, clobberFlags: true},                                              // arg0 - auxint
		{name: "SUBload", argLength: 3, reg: gpopload, asm: "SUB", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},   // arg0 - *arg1. arg2=mem
		{name: "SUBWload", argLength: 3, reg: gpopload, asm: "SUBW", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 - *arg1. arg2=mem

		{name: "MULLD", argLength: 2, reg: gp21, asm: "MULLD", typ: "Int64", commutative: true, resultInArg0: true, clobberFlags: true},                                // arg0 * arg1
		{name: "MULLW", argLength: 2, reg: gp21, asm: "MULLW", typ: "Int32", commutative: true, resultInArg0: true, clobberFlags: true},                                // arg0 * arg1
		{name: "MULLDconst", argLength: 1, reg: gp11, asm: "MULLD", aux: "Int32", typ: "Int64", resultInArg0: true, clobberFlags: true},                                // arg0 * auxint
		{name: "MULLWconst", argLength: 1, reg: gp11, asm: "MULLW", aux: "Int32", typ: "Int32", resultInArg0: true, clobberFlags: true},                                // arg0 * auxint
		{name: "MULLDload", argLength: 3, reg: gpopload, asm: "MULLD", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 * *arg1. arg2=mem
		{name: "MULLWload", argLength: 3, reg: gpopload, asm: "MULLW", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 * *arg1. arg2=mem

		{name: "MULHD", argLength: 2, reg: gp21tmp, asm: "MULHD", typ: "Int64", commutative: true, resultInArg0: true, clobberFlags: true},   // (arg0 * arg1) >> width
		{name: "MULHDU", argLength: 2, reg: gp21tmp, asm: "MULHDU", typ: "Int64", commutative: true, resultInArg0: true, clobberFlags: true}, // (arg0 * arg1) >> width

		{name: "DIVD", argLength: 2, reg: gp21tmp, asm: "DIVD", resultInArg0: true, clobberFlags: true},   // arg0 / arg1
		{name: "DIVW", argLength: 2, reg: gp21tmp, asm: "DIVW", resultInArg0: true, clobberFlags: true},   // arg0 / arg1
		{name: "DIVDU", argLength: 2, reg: gp21tmp, asm: "DIVDU", resultInArg0: true, clobberFlags: true}, // arg0 / arg1
		{name: "DIVWU", argLength: 2, reg: gp21tmp, asm: "DIVWU", resultInArg0: true, clobberFlags: true}, // arg0 / arg1

		{name: "MODD", argLength: 2, reg: gp21tmp, asm: "MODD", resultInArg0: true, clobberFlags: true}, // arg0 % arg1
		{name: "MODW", argLength: 2, reg: gp21tmp, asm: "MODW", resultInArg0: true, clobberFlags: true}, // arg0 % arg1

		{name: "MODDU", argLength: 2, reg: gp21tmp, asm: "MODDU", resultInArg0: true, clobberFlags: true}, // arg0 % arg1
		{name: "MODWU", argLength: 2, reg: gp21tmp, asm: "MODWU", resultInArg0: true, clobberFlags: true}, // arg0 % arg1

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true, clobberFlags: true},                                                                    // arg0 & arg1
		{name: "ANDW", argLength: 2, reg: gp21, asm: "ANDW", commutative: true, clobberFlags: true},                                                                  // arg0 & arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int64", resultInArg0: true, clobberFlags: true},                                                // arg0 & auxint
		{name: "ANDWconst", argLength: 1, reg: gp11, asm: "ANDW", aux: "Int32", resultInArg0: true, clobberFlags: true},                                              // arg0 & auxint
		{name: "ANDload", argLength: 3, reg: gpopload, asm: "AND", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},   // arg0 & *arg1. arg2=mem
		{name: "ANDWload", argLength: 3, reg: gpopload, asm: "ANDW", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 & *arg1. arg2=mem

		{name: "OR", argLength: 2, reg: gp21, asm: "OR", commutative: true, clobberFlags: true},                                                                    // arg0 | arg1
		{name: "ORW", argLength: 2, reg: gp21, asm: "ORW", commutative: true, clobberFlags: true},                                                                  // arg0 | arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "OR", aux: "Int64", resultInArg0: true, clobberFlags: true},                                                // arg0 | auxint
		{name: "ORWconst", argLength: 1, reg: gp11, asm: "ORW", aux: "Int32", resultInArg0: true, clobberFlags: true},                                              // arg0 | auxint
		{name: "ORload", argLength: 3, reg: gpopload, asm: "OR", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},   // arg0 | *arg1. arg2=mem
		{name: "ORWload", argLength: 3, reg: gpopload, asm: "ORW", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 | *arg1. arg2=mem

		{name: "XOR", argLength: 2, reg: gp21, asm: "XOR", commutative: true, clobberFlags: true},                                                                    // arg0 ^ arg1
		{name: "XORW", argLength: 2, reg: gp21, asm: "XORW", commutative: true, clobberFlags: true},                                                                  // arg0 ^ arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "XOR", aux: "Int64", resultInArg0: true, clobberFlags: true},                                                // arg0 ^ auxint
		{name: "XORWconst", argLength: 1, reg: gp11, asm: "XORW", aux: "Int32", resultInArg0: true, clobberFlags: true},                                              // arg0 ^ auxint
		{name: "XORload", argLength: 3, reg: gpopload, asm: "XOR", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"},   // arg0 ^ *arg1. arg2=mem
		{name: "XORWload", argLength: 3, reg: gpopload, asm: "XORW", aux: "SymOff", resultInArg0: true, clobberFlags: true, faultOnNilArg1: true, symEffect: "Read"}, // arg0 ^ *arg1. arg2=mem

		// Arithmetic ops with carry/borrow chain.
		//
		// A carry is represented by a condition code of 2 or 3 (GT or OV).
		// A borrow is represented by a condition code of 0 or 1 (EQ or LT).
		{name: "ADDC", argLength: 2, reg: gp21flags, asm: "ADDC", typ: "(UInt64,Flags)", commutative: true},                          // (arg0 + arg1, carry out)
		{name: "ADDCconst", argLength: 1, reg: gp11flags, asm: "ADDC", typ: "(UInt64,Flags)", aux: "Int16"},                          // (arg0 + auxint, carry out)
		{name: "ADDE", argLength: 3, reg: gp2flags1flags, asm: "ADDE", typ: "(UInt64,Flags)", commutative: true, resultInArg0: true}, // (arg0 + arg1 + arg2 (carry in), carry out)
		{name: "SUBC", argLength: 2, reg: gp21flags, asm: "SUBC", typ: "(UInt64,Flags)"},                                             // (arg0 - arg1, borrow out)
		{name: "SUBE", argLength: 3, reg: gp2flags1flags, asm: "SUBE", typ: "(UInt64,Flags)", resultInArg0: true},                    // (arg0 - arg1 - arg2 (borrow in), borrow out)

		// Comparisons.
		{name: "CMP", argLength: 2, reg: gp2flags, asm: "CMP", typ: "Flags"},   // arg0 compare to arg1
		{name: "CMPW", argLength: 2, reg: gp2flags, asm: "CMPW", typ: "Flags"}, // arg0 compare to arg1

		{name: "CMPU", argLength: 2, reg: gp2flags, asm: "CMPU", typ: "Flags"},   // arg0 compare to arg1
		{name: "CMPWU", argLength: 2, reg: gp2flags, asm: "CMPWU", typ: "Flags"}, // arg0 compare to arg1

		{name: "CMPconst", argLength: 1, reg: gp1flags, asm: "CMP", typ: "Flags", aux: "Int32"},     // arg0 compare to auxint
		{name: "CMPWconst", argLength: 1, reg: gp1flags, asm: "CMPW", typ: "Flags", aux: "Int32"},   // arg0 compare to auxint
		{name: "CMPUconst", argLength: 1, reg: gp1flags, asm: "CMPU", typ: "Flags", aux: "Int32"},   // arg0 compare to auxint
		{name: "CMPWUconst", argLength: 1, reg: gp1flags, asm: "CMPWU", typ: "Flags", aux: "Int32"}, // arg0 compare to auxint

		{name: "FCMPS", argLength: 2, reg: fp2flags, asm: "CEBR", typ: "Flags"},  // arg0 compare to arg1, f32
		{name: "FCMP", argLength: 2, reg: fp2flags, asm: "FCMPU", typ: "Flags"},  // arg0 compare to arg1, f64
		{name: "LTDBR", argLength: 1, reg: fp1flags, asm: "LTDBR", typ: "Flags"}, // arg0 compare to 0, f64
		{name: "LTEBR", argLength: 1, reg: fp1flags, asm: "LTEBR", typ: "Flags"}, // arg0 compare to 0, f32

		{name: "SLD", argLength: 2, reg: sh21, asm: "SLD"},                    // arg0 << arg1, shift amount is mod 64
		{name: "SLW", argLength: 2, reg: sh21, asm: "SLW"},                    // arg0 << arg1, shift amount is mod 64
		{name: "SLDconst", argLength: 1, reg: gp11, asm: "SLD", aux: "UInt8"}, // arg0 << auxint, shift amount 0-63
		{name: "SLWconst", argLength: 1, reg: gp11, asm: "SLW", aux: "UInt8"}, // arg0 << auxint, shift amount 0-31

		{name: "SRD", argLength: 2, reg: sh21, asm: "SRD"},                    // unsigned arg0 >> arg1, shift amount is mod 64
		{name: "SRW", argLength: 2, reg: sh21, asm: "SRW"},                    // unsigned uint32(arg0) >> arg1, shift amount is mod 64
		{name: "SRDconst", argLength: 1, reg: gp11, asm: "SRD", aux: "UInt8"}, // unsigned arg0 >> auxint, shift amount 0-63
		{name: "SRWconst", argLength: 1, reg: gp11, asm: "SRW", aux: "UInt8"}, // unsigned uint32(arg0) >> auxint, shift amount 0-31

		// Arithmetic shifts clobber flags.
		{name: "SRAD", argLength: 2, reg: sh21, asm: "SRAD", clobberFlags: true},                    // signed arg0 >> arg1, shift amount is mod 64
		{name: "SRAW", argLength: 2, reg: sh21, asm: "SRAW", clobberFlags: true},                    // signed int32(arg0) >> arg1, shift amount is mod 64
		{name: "SRADconst", argLength: 1, reg: gp11, asm: "SRAD", aux: "UInt8", clobberFlags: true}, // signed arg0 >> auxint, shift amount 0-63
		{name: "SRAWconst", argLength: 1, reg: gp11, asm: "SRAW", aux: "UInt8", clobberFlags: true}, // signed int32(arg0) >> auxint, shift amount 0-31

		// Rotate instructions.
		// Note: no RLLGconst - use RISBGZ instead.
		{name: "RLLG", argLength: 2, reg: sh21, asm: "RLLG"},                  // arg0 rotate left arg1, rotate amount 0-63
		{name: "RLL", argLength: 2, reg: sh21, asm: "RLL"},                    // arg0 rotate left arg1, rotate amount 0-31
		{name: "RLLconst", argLength: 1, reg: gp11, asm: "RLL", aux: "UInt8"}, // arg0 rotate left auxint, rotate amount 0-31

		// Rotate then (and|or|xor|insert) selected bits instructions.
		//
		// Aux is an s390x.RotateParams struct containing Start, End and rotation
		// Amount fields.
		//
		// arg1 is rotated left by the rotation amount then the bits from the start
		// bit to the end bit (inclusive) are combined with arg0 using the logical
		// operation specified. Bit indices are specified from left to right - the
		// MSB is 0 and the LSB is 63.
		//
		// Examples:
		//               |          aux         |
		// | instruction | start | end | amount |          arg0         |          arg1         |         result        |
		// +-------------+-------+-----+--------+-----------------------+-----------------------+-----------------------+
		// | RXSBG (XOR) |     0 |   1 |      0 | 0xffff_ffff_ffff_ffff | 0xffff_ffff_ffff_ffff | 0x3fff_ffff_ffff_ffff |
		// | RXSBG (XOR) |    62 |  63 |      0 | 0xffff_ffff_ffff_ffff | 0xffff_ffff_ffff_ffff | 0xffff_ffff_ffff_fffc |
		// | RXSBG (XOR) |     0 |  47 |     16 | 0xffff_ffff_ffff_ffff | 0x0000_0000_0000_ffff | 0xffff_ffff_0000_ffff |
		// +-------------+-------+-----+--------+-----------------------+-----------------------+-----------------------+
		//
		{name: "RXSBG", argLength: 2, reg: gp21, asm: "RXSBG", resultInArg0: true, aux: "S390XRotateParams", clobberFlags: true}, // rotate then xor selected bits
		{name: "RISBGZ", argLength: 1, reg: gp11, asm: "RISBGZ", aux: "S390XRotateParams", clobberFlags: true},                   // rotate then insert selected bits [into zero]

		// unary ops
		{name: "NEG", argLength: 1, reg: gp11, asm: "NEG", clobberFlags: true},   // -arg0
		{name: "NEGW", argLength: 1, reg: gp11, asm: "NEGW", clobberFlags: true}, // -arg0

		{name: "NOT", argLength: 1, reg: gp11, resultInArg0: true, clobberFlags: true},  // ^arg0
		{name: "NOTW", argLength: 1, reg: gp11, resultInArg0: true, clobberFlags: true}, // ^arg0

		{name: "FSQRT", argLength: 1, reg: fp11, asm: "FSQRT"},   // sqrt(arg0)
		{name: "FSQRTS", argLength: 1, reg: fp11, asm: "FSQRTS"}, // sqrt(arg0), float32

		// Conditional register-register moves.
		// The aux for these values is an s390x.CCMask value representing the condition code mask.
		{name: "LOCGR", argLength: 3, reg: gp2flags1, resultInArg0: true, asm: "LOCGR", aux: "S390XCCMask"}, // load arg1 into arg0 if the condition code in arg2 matches a masked bit in aux.

		{name: "MOVBreg", argLength: 1, reg: gp11sp, asm: "MOVB", typ: "Int64"},    // sign extend arg0 from int8 to int64
		{name: "MOVBZreg", argLength: 1, reg: gp11sp, asm: "MOVBZ", typ: "UInt64"}, // zero extend arg0 from int8 to int64
		{name: "MOVHreg", argLength: 1, reg: gp11sp, asm: "MOVH", typ: "Int64"},    // sign extend arg0 from int16 to int64
		{name: "MOVHZreg", argLength: 1, reg: gp11sp, asm: "MOVHZ", typ: "UInt64"}, // zero extend arg0 from int16 to int64
		{name: "MOVWreg", argLength: 1, reg: gp11sp, asm: "MOVW", typ: "Int64"},    // sign extend arg0 from int32 to int64
		{name: "MOVWZreg", argLength: 1, reg: gp11sp, asm: "MOVWZ", typ: "UInt64"}, // zero extend arg0 from int32 to int64

		{name: "MOVDconst", reg: gp01, asm: "MOVD", typ: "UInt64", aux: "Int64", rematerializeable: true}, // auxint

		{name: "LDGR", argLength: 1, reg: gpfp, asm: "LDGR"}, // move int64 to float64 (no conversion)
		{name: "LGDR", argLength: 1, reg: fpgp, asm: "LGDR"}, // move float64 to int64 (no conversion)

		{name: "CFDBRA", argLength: 1, reg: fpgp, asm: "CFDBRA", clobberFlags: true}, // convert float64 to int32
		{name: "CGDBRA", argLength: 1, reg: fpgp, asm: "CGDBRA", clobberFlags: true}, // convert float64 to int64
		{name: "CFEBRA", argLength: 1, reg: fpgp, asm: "CFEBRA", clobberFlags: true}, // convert float32 to int32
		{name: "CGEBRA", argLength: 1, reg: fpgp, asm: "CGEBRA", clobberFlags: true}, // convert float32 to int64
		{name: "CEFBRA", argLength: 1, reg: gpfp, asm: "CEFBRA", clobberFlags: true}, // convert int32 to float32
		{name: "CDFBRA", argLength: 1, reg: gpfp, asm: "CDFBRA", clobberFlags: true}, // convert int32 to float64
		{name: "CEGBRA", argLength: 1, reg: gpfp, asm: "CEGBRA", clobberFlags: true}, // convert int64 to float32
		{name: "CDGBRA", argLength: 1, reg: gpfp, asm: "CDGBRA", clobberFlags: true}, // convert int64 to float64
		{name: "CLFEBR", argLength: 1, reg: fpgp, asm: "CLFEBR", clobberFlags: true}, // convert float32 to uint32
		{name: "CLFDBR", argLength: 1, reg: fpgp, asm: "CLFDBR", clobberFlags: true}, // convert float64 to uint32
		{name: "CLGEBR", argLength: 1, reg: fpgp, asm: "CLGEBR", clobberFlags: true}, // convert float32 to uint64
		{name: "CLGDBR", argLength: 1, reg: fpgp, asm: "CLGDBR", clobberFlags: true}, // convert float64 to uint64
		{name: "CELFBR", argLength: 1, reg: gpfp, asm: "CELFBR", clobberFlags: true}, // convert uint32 to float32
		{name: "CDLFBR", argLength: 1, reg: gpfp, asm: "CDLFBR", clobberFlags: true}, // convert uint32 to float64
		{name: "CELGBR", argLength: 1, reg: gpfp, asm: "CELGBR", clobberFlags: true}, // convert uint64 to float32
		{name: "CDLGBR", argLength: 1, reg: gpfp, asm: "CDLGBR", clobberFlags: true}, // convert uint64 to float64

		{name: "LEDBR", argLength: 1, reg: fp11, asm: "LEDBR"}, // convert float64 to float32
		{name: "LDEBR", argLength: 1, reg: fp11, asm: "LDEBR"}, // convert float32 to float64

		{name: "MOVDaddr", argLength: 1, reg: addr, aux: "SymOff", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxint + offset encoded in aux
		{name: "MOVDaddridx", argLength: 2, reg: addridx, aux: "SymOff", symEffect: "Addr"},                    // arg0 + arg1 + auxint + aux

		// auxint+aux == add auxint and the offset of the symbol in aux (if any) to the effective address
		{name: "MOVBZload", argLength: 2, reg: gpload, asm: "MOVBZ", aux: "SymOff", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},  // load byte from arg0+auxint+aux. arg1=mem.  Zero extend.
		{name: "MOVBload", argLength: 2, reg: gpload, asm: "MOVB", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},                  // ditto, sign extend to int64
		{name: "MOVHZload", argLength: 2, reg: gpload, asm: "MOVHZ", aux: "SymOff", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"}, // load 2 bytes from arg0+auxint+aux. arg1=mem.  Zero extend.
		{name: "MOVHload", argLength: 2, reg: gpload, asm: "MOVH", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},                  // ditto, sign extend to int64
		{name: "MOVWZload", argLength: 2, reg: gpload, asm: "MOVWZ", aux: "SymOff", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"}, // load 4 bytes from arg0+auxint+aux. arg1=mem.  Zero extend.
		{name: "MOVWload", argLength: 2, reg: gpload, asm: "MOVW", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},                  // ditto, sign extend to int64
		{name: "MOVDload", argLength: 2, reg: gpload, asm: "MOVD", aux: "SymOff", typ: "UInt64", faultOnNilArg0: true, symEffect: "Read"},   // load 8 bytes from arg0+auxint+aux. arg1=mem

		{name: "MOVWBR", argLength: 1, reg: gp11, asm: "MOVWBR"}, // arg0 swap bytes
		{name: "MOVDBR", argLength: 1, reg: gp11, asm: "MOVDBR"}, // arg0 swap bytes

		{name: "MOVHBRload", argLength: 2, reg: gpload, asm: "MOVHBR", aux: "SymOff", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"}, // load 2 bytes from arg0+auxint+aux. arg1=mem. Reverse bytes.
		{name: "MOVWBRload", argLength: 2, reg: gpload, asm: "MOVWBR", aux: "SymOff", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"}, // load 4 bytes from arg0+auxint+aux. arg1=mem. Reverse bytes.
		{name: "MOVDBRload", argLength: 2, reg: gpload, asm: "MOVDBR", aux: "SymOff", typ: "UInt64", faultOnNilArg0: true, symEffect: "Read"}, // load 8 bytes from arg0+auxint+aux. arg1=mem. Reverse bytes.

		{name: "MOVBstore", argLength: 3, reg: gpstore, asm: "MOVB", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},       // store byte in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVHstore", argLength: 3, reg: gpstore, asm: "MOVH", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},       // store 2 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVWstore", argLength: 3, reg: gpstore, asm: "MOVW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},       // store 4 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVDstore", argLength: 3, reg: gpstore, asm: "MOVD", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},       // store 8 bytes in arg1 to arg0+auxint+aux. arg2=mem
		{name: "MOVHBRstore", argLength: 3, reg: gpstorebr, asm: "MOVHBR", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes in arg1 to arg0+auxint+aux. arg2=mem. Reverse bytes.
		{name: "MOVWBRstore", argLength: 3, reg: gpstorebr, asm: "MOVWBR", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes in arg1 to arg0+auxint+aux. arg2=mem. Reverse bytes.
		{name: "MOVDBRstore", argLength: 3, reg: gpstorebr, asm: "MOVDBR", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes in arg1 to arg0+auxint+aux. arg2=mem. Reverse bytes.

		{name: "MVC", argLength: 3, reg: gpmvc, asm: "MVC", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, faultOnNilArg1: true, symEffect: "None"}, // arg0=destptr, arg1=srcptr, arg2=mem, auxint=size,off

		// indexed loads/stores
		{name: "MOVBZloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVBZ", aux: "SymOff", typ: "UInt8", symEffect: "Read"},   // load a byte from arg0+arg1+auxint+aux. arg2=mem. Zero extend.
		{name: "MOVBloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVB", aux: "SymOff", typ: "Int8", symEffect: "Read"},      // load a byte from arg0+arg1+auxint+aux. arg2=mem. Sign extend.
		{name: "MOVHZloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVHZ", aux: "SymOff", typ: "UInt16", symEffect: "Read"},  // load 2 bytes from arg0+arg1+auxint+aux. arg2=mem. Zero extend.
		{name: "MOVHloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVH", aux: "SymOff", typ: "Int16", symEffect: "Read"},     // load 2 bytes from arg0+arg1+auxint+aux. arg2=mem. Sign extend.
		{name: "MOVWZloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVWZ", aux: "SymOff", typ: "UInt32", symEffect: "Read"},  // load 4 bytes from arg0+arg1+auxint+aux. arg2=mem. Zero extend.
		{name: "MOVWloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVW", aux: "SymOff", typ: "Int32", symEffect: "Read"},     // load 4 bytes from arg0+arg1+auxint+aux. arg2=mem. Sign extend.
		{name: "MOVDloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVD", aux: "SymOff", typ: "UInt64", symEffect: "Read"},    // load 8 bytes from arg0+arg1+auxint+aux. arg2=mem
		{name: "MOVHBRloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVHBR", aux: "SymOff", typ: "Int16", symEffect: "Read"}, // load 2 bytes from arg0+arg1+auxint+aux. arg2=mem. Reverse bytes.
		{name: "MOVWBRloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVWBR", aux: "SymOff", typ: "Int32", symEffect: "Read"}, // load 4 bytes from arg0+arg1+auxint+aux. arg2=mem. Reverse bytes.
		{name: "MOVDBRloadidx", argLength: 3, reg: gploadidx, commutative: true, asm: "MOVDBR", aux: "SymOff", typ: "Int64", symEffect: "Read"}, // load 8 bytes from arg0+arg1+auxint+aux. arg2=mem. Reverse bytes.
		{name: "MOVBstoreidx", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVB", aux: "SymOff", symEffect: "Write"},                // store byte in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVHstoreidx", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVH", aux: "SymOff", symEffect: "Write"},                // store 2 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVWstoreidx", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVW", aux: "SymOff", symEffect: "Write"},                // store 4 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVDstoreidx", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVD", aux: "SymOff", symEffect: "Write"},                // store 8 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem
		{name: "MOVHBRstoreidx", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVHBR", aux: "SymOff", symEffect: "Write"},            // store 2 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem. Reverse bytes.
		{name: "MOVWBRstoreidx", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVWBR", aux: "SymOff", symEffect: "Write"},            // store 4 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem. Reverse bytes.
		{name: "MOVDBRstoreidx", argLength: 4, reg: gpstoreidx, commutative: true, asm: "MOVDBR", aux: "SymOff", symEffect: "Write"},            // store 8 bytes in arg2 to arg0+arg1+auxint+aux. arg3=mem. Reverse bytes.

		// For storeconst ops, the AuxInt field encodes both
		// the value to store and an address offset of the store.
		// Cast AuxInt to a ValAndOff to extract Val and Off fields.
		{name: "MOVBstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVB", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store low byte of ValAndOff(AuxInt).Val() to arg0+ValAndOff(AuxInt).Off()+aux.  arg1=mem
		{name: "MOVHstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVH", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store low 2 bytes of ...
		{name: "MOVWstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVW", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store low 4 bytes of ...
		{name: "MOVDstoreconst", argLength: 2, reg: gpstoreconst, asm: "MOVD", aux: "SymValAndOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of ...

		{name: "CLEAR", argLength: 2, reg: regInfo{inputs: []regMask{ptr, 0}}, asm: "CLEAR", aux: "SymValAndOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, symEffect: "Write"},

		{name: "CALLstatic", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                                // call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                  // tail call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: -1, reg: regInfo{inputs: []regMask{ptrsp, buildReg("R12"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, last arg=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: -1, reg: regInfo{inputs: []regMask{ptr}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                         // call fn by pointer.  arg0=codeptr, last arg=mem, auxint=argsize, returns mem

		// (InvertFlags (CMP a b)) == (CMP b a)
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags", argLength: 1}, // reverse direction of arg0

		// Pseudo-ops
		{name: "LoweredGetG", argLength: 1, reg: gp01}, // arg0=mem
		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of R12 (the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("R12")}}, zeroWidth: true},
		// arg0=ptr,arg1=mem, returns void.  Faults if ptr is nil.
		// LoweredGetCallerSP returns the SP of the caller of the current function. arg0=mem.
		{name: "LoweredGetCallerSP", argLength: 1, reg: gp01, rematerializeable: true},
		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" sys.GetCallerPC,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{ptrsp}}, clobberFlags: true, nilCheck: true, faultOnNilArg0: true},
		// Round ops to block fused-multiply-add extraction.
		{name: "LoweredRound32F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},
		{name: "LoweredRound64F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},

		// LoweredWB invokes runtime.gcWriteBarrier. arg0=mem, aux=# of buffer entries needed
		// It saves all GP registers if necessary,
		// but clobbers R14 (LR) because it's a call,
		// and also clobbers R1 as the PLT stub does.
		// Returns a pointer to a write barrier buffer in R9.
		{name: "LoweredWB", argLength: 1, reg: regInfo{clobbers: (callerSave &^ gpg) | buildReg("R14") | r1, outputs: []regMask{r9}}, clobberFlags: true, aux: "Int64"},

		// LoweredPanicBoundsRR takes x and y, two values that caused a bounds check to fail.
		// the RC and CR versions are used when one of the arguments is a constant. CC is used
		// when both are constant (normally both 0, as prove derives the fact that a [0] bounds
		// failure means the length must have also been 0).
		// AuxInt contains a report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsRR", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{gp &^ lr, gp &^ lr}}, typ: "Mem", call: true}, // arg0=x, arg1=y, arg2=mem, returns memory.
		{name: "LoweredPanicBoundsRC", argLength: 2, aux: "PanicBoundsC", reg: regInfo{inputs: []regMask{gp &^ lr}}, typ: "Mem", call: true},    // arg0=x, arg1=mem, returns memory.
		{name: "LoweredPanicBoundsCR", argLength: 2, aux: "PanicBoundsC", reg: regInfo{inputs: []regMask{gp &^ lr}}, typ: "Mem", call: true},    // arg0=y, arg1=mem, returns memory.
		{name: "LoweredPanicBoundsCC", argLength: 1, aux: "PanicBoundsCC", reg: regInfo{}, typ: "Mem", call: true},                              // arg0=mem, returns memory.

		// Constant condition code values. The condition code can be 0, 1, 2 or 3.
		{name: "FlagEQ"}, // CC=0 (equal)
		{name: "FlagLT"}, // CC=1 (less than)
		{name: "FlagGT"}, // CC=2 (greater than)
		{name: "FlagOV"}, // CC=3 (overflow)

		// Fast-BCR-serialization to ensure store-load ordering.
		{name: "SYNC", argLength: 1, reg: sync, asm: "SYNC", typ: "Mem"},

		// Atomic loads. These are just normal loads but return <value,memory> tuples
		// so they can be properly ordered with other loads.
		// load from arg0+auxint+aux.  arg1=mem.
		{name: "MOVBZatomicload", argLength: 2, reg: gpload, asm: "MOVBZ", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVWZatomicload", argLength: 2, reg: gpload, asm: "MOVWZ", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVDatomicload", argLength: 2, reg: gpload, asm: "MOVD", aux: "SymOff", faultOnNilArg0: true, symEffect: "Read"},

		// Atomic stores. These are just normal stores.
		// store arg1 to arg0+auxint+aux. arg2=mem.
		{name: "MOVBatomicstore", argLength: 3, reg: gpstore, asm: "MOVB", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "Write"},
		{name: "MOVWatomicstore", argLength: 3, reg: gpstore, asm: "MOVW", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "Write"},
		{name: "MOVDatomicstore", argLength: 3, reg: gpstore, asm: "MOVD", aux: "SymOff", typ: "Mem", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "Write"},

		// Atomic adds.
		// *(arg0+auxint+aux) += arg1.  arg2=mem.
		// Returns a tuple of <old contents of *(arg0+auxint+aux), memory>.
		{name: "LAA", argLength: 3, reg: gpstorelaa, asm: "LAA", typ: "(UInt32,Mem)", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "LAAG", argLength: 3, reg: gpstorelaa, asm: "LAAG", typ: "(UInt64,Mem)", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "AddTupleFirst32", argLength: 2}, // arg1=tuple <x,y>.  Returns <x+arg0,y>.
		{name: "AddTupleFirst64", argLength: 2}, // arg1=tuple <x,y>.  Returns <x+arg0,y>.

		// Atomic bitwise operations.
		// Note: 'floor' operations round the pointer down to the nearest word boundary
		// which reflects how they are used in the runtime.
		{name: "LAN", argLength: 3, reg: gpstore, asm: "LAN", typ: "Mem", clobberFlags: true, hasSideEffects: true},         // *arg0 &= arg1. arg2 = mem.
		{name: "LANfloor", argLength: 3, reg: gpstorelab, asm: "LAN", typ: "Mem", clobberFlags: true, hasSideEffects: true}, // *(floor(arg0, 4)) &= arg1. arg2 = mem.
		{name: "LAO", argLength: 3, reg: gpstore, asm: "LAO", typ: "Mem", clobberFlags: true, hasSideEffects: true},         // *arg0 |= arg1. arg2 = mem.
		{name: "LAOfloor", argLength: 3, reg: gpstorelab, asm: "LAO", typ: "Mem", clobberFlags: true, hasSideEffects: true}, // *(floor(arg0, 4)) |= arg1. arg2 = mem.

		// Compare and swap.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory.
		// if *(arg0+auxint+aux) == arg1 {
		//   *(arg0+auxint+aux) = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// Note that these instructions also return the old value in arg1, but we ignore it.
		// TODO: have these return flags instead of bool.  The current system generates:
		//    CS ...
		//    MOVD  $0, ret
		//    BNE   2(PC)
		//    MOVD  $1, ret
		//    CMPW  ret, $0
		//    BNE ...
		// instead of just
		//    CS ...
		//    BEQ ...
		// but we can't do that because memory-using ops can't generate flags yet
		// (flagalloc wants to move flag-generating instructions around).
		{name: "LoweredAtomicCas32", argLength: 4, reg: cas, asm: "CS", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "LoweredAtomicCas64", argLength: 4, reg: cas, asm: "CSG", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},

		// Lowered atomic swaps, emulated using compare-and-swap.
		// store arg1 to arg0+auxint+aux, arg2=mem.
		{name: "LoweredAtomicExchange32", argLength: 3, reg: exchange, asm: "CS", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},
		{name: "LoweredAtomicExchange64", argLength: 3, reg: exchange, asm: "CSG", aux: "SymOff", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, symEffect: "RdWr"},

		// find leftmost one
		{
			name:         "FLOGR",
			argLength:    1,
			reg:          regInfo{inputs: gponly, outputs: []regMask{buildReg("R0")}, clobbers: buildReg("R1")},
			asm:          "FLOGR",
			typ:          "UInt64",
			clobberFlags: true,
		},

		// population count
		//
		// Counts the number of ones in each byte of arg0
		// and places the result into the corresponding byte
		// of the result.
		{
			name:         "POPCNT",
			argLength:    1,
			reg:          gp11,
			asm:          "POPCNT",
			typ:          "UInt64",
			clobberFlags: true,
		},

		// unsigned multiplication (64x64 → 128)
		//
		// Multiply the two 64-bit input operands together and place the 128-bit result into
		// an even-odd register pair. The second register in the target pair also contains
		// one of the input operands. Since we don't currently have a way to specify an
		// even-odd register pair we hardcode this register pair as R2:R3.
		{
			name:      "MLGR",
			argLength: 2,
			reg:       regInfo{inputs: []regMask{gp, r3}, outputs: []regMask{r2, r3}},
			asm:       "MLGR",
		},

		// pseudo operations to sum the output of the POPCNT instruction
		{name: "SumBytes2", argLength: 1, typ: "UInt8"}, // sum the rightmost 2 bytes in arg0 ignoring overflow
		{name: "SumBytes4", argLength: 1, typ: "UInt8"}, // sum the rightmost 4 bytes in arg0 ignoring overflow
		{name: "SumBytes8", argLength: 1, typ: "UInt8"}, // sum all the bytes in arg0 ignoring overflow

		// store multiple
		{
			name:           "STMG2",
			argLength:      4,
			reg:            regInfo{inputs: []regMask{ptrsp, buildReg("R1"), buildReg("R2"), 0}},
			aux:            "SymOff",
			typ:            "Mem",
			asm:            "STMG",
			faultOnNilArg0: true,
			symEffect:      "Write",
			clobberFlags:   true, // TODO(mundaym): currently uses AGFI to handle large offsets
		},
		{
			name:           "STMG3",
			argLength:      5,
			reg:            regInfo{inputs: []regMask{ptrsp, buildReg("R1"), buildReg("R2"), buildReg("R3"), 0}},
			aux:            "SymOff",
			typ:            "Mem",
			asm:            "STMG",
			faultOnNilArg0: true,
			symEffect:      "Write",
			clobberFlags:   true, // TODO(mundaym): currently uses AGFI to handle large offsets
		},
		{
			name:      "STMG4",
			argLength: 6,
			reg: regInfo{inputs: []regMask{
				ptrsp,
				buildReg("R1"),
				buildReg("R2"),
				buildReg("R3"),
				buildReg("R4"),
				0,
			}},
			aux:            "SymOff",
			typ:            "Mem",
			asm:            "STMG",
			faultOnNilArg0: true,
			symEffect:      "Write",
			clobberFlags:   true, // TODO(mundaym): currently uses AGFI to handle large offsets
		},
		{
			name:           "STM2",
			argLength:      4,
			reg:            regInfo{inputs: []regMask{ptrsp, buildReg("R1"), buildReg("R2"), 0}},
			aux:            "SymOff",
			typ:            "Mem",
			asm:            "STMY",
			faultOnNilArg0: true,
			symEffect:      "Write",
			clobberFlags:   true, // TODO(mundaym): currently uses AGFI to handle large offsets
		},
		{
			name:           "STM3",
			argLength:      5,
			reg:            regInfo{inputs: []regMask{ptrsp, buildReg("R1"), buildReg("R2"), buildReg("R3"), 0}},
			aux:            "SymOff",
			typ:            "Mem",
			asm:            "STMY",
			faultOnNilArg0: true,
			symEffect:      "Write",
			clobberFlags:   true, // TODO(mundaym): currently uses AGFI to handle large offsets
		},
		{
			name:      "STM4",
			argLength: 6,
			reg: regInfo{inputs: []regMask{
				ptrsp,
				buildReg("R1"),
				buildReg("R2"),
				buildReg("R3"),
				buildReg("R4"),
				0,
			}},
			aux:            "SymOff",
			typ:            "Mem",
			asm:            "STMY",
			faultOnNilArg0: true,
			symEffect:      "Write",
			clobberFlags:   true, // TODO(mundaym): currently uses AGFI to handle large offsets
		},

		// large move
		// auxint = remaining bytes after loop (rem)
		// arg0 = address of dst memory (in R1, changed as a side effect)
		// arg1 = address of src memory (in R2, changed as a side effect)
		// arg2 = pointer to last address to move in loop + 256
		// arg3 = mem
		// returns mem
		//
		// mvc: MVC  $256, 0(R2), 0(R1)
		//      MOVD $256(R1), R1
		//      MOVD $256(R2), R2
		//      CMP  R2, Rarg2
		//      BNE  mvc
		//	MVC  $rem, 0(R2), 0(R1) // if rem > 0
		{
			name:      "LoweredMove",
			aux:       "Int64",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("R1"), buildReg("R2"), gpsp},
				clobbers: buildReg("R1 R2"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// large clear
		// auxint = remaining bytes after loop (rem)
		// arg0 = address of dst memory (in R1, changed as a side effect)
		// arg1 = pointer to last address to zero in loop + 256
		// arg2 = mem
		// returns mem
		//
		// clear: CLEAR $256, 0(R1)
		//        MOVD  $256(R1), R1
		//        CMP   R1, Rarg2
		//        BNE   clear
		//	  CLEAR $rem, 0(R1) // if rem > 0
		{
			name:      "LoweredZero",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R1"), gpsp},
				clobbers: buildReg("R1"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
		},
	}

	// All blocks on s390x have their condition code mask (s390x.CCMask) as the Aux value.
	// The condition code mask is a 4-bit mask where each bit corresponds to a condition
	// code value. If the value of the condition code matches a bit set in the condition
	// code mask then the first successor is executed. Otherwise the second successor is
	// executed.
	//
	// | condition code value |  mask bit  |
	// +----------------------+------------+
	// | 0 (equal)            | 0b1000 (8) |
	// | 1 (less than)        | 0b0100 (4) |
	// | 2 (greater than)     | 0b0010 (2) |
	// | 3 (unordered)        | 0b0001 (1) |
	//
	// Note: that compare-and-branch instructions must not have bit 3 (0b0001) set.
	var S390Xblocks = []blockData{
		// branch on condition
		{name: "BRC", controls: 1, aux: "S390XCCMask"}, // condition code value (flags) is Controls[0]

		// compare-and-branch (register-register)
		//  - integrates comparison of Controls[0] with Controls[1]
		//  - both control values must be in general purpose registers
		{name: "CRJ", controls: 2, aux: "S390XCCMask"},   // signed 32-bit integer comparison
		{name: "CGRJ", controls: 2, aux: "S390XCCMask"},  // signed 64-bit integer comparison
		{name: "CLRJ", controls: 2, aux: "S390XCCMask"},  // unsigned 32-bit integer comparison
		{name: "CLGRJ", controls: 2, aux: "S390XCCMask"}, // unsigned 64-bit integer comparison

		// compare-and-branch (register-immediate)
		//  - integrates comparison of Controls[0] with AuxInt
		//  - control value must be in a general purpose register
		//  - the AuxInt value is sign-extended for signed comparisons
		//    and zero-extended for unsigned comparisons
		{name: "CIJ", controls: 1, aux: "S390XCCMaskInt8"},    // signed 32-bit integer comparison
		{name: "CGIJ", controls: 1, aux: "S390XCCMaskInt8"},   // signed 64-bit integer comparison
		{name: "CLIJ", controls: 1, aux: "S390XCCMaskUint8"},  // unsigned 32-bit integer comparison
		{name: "CLGIJ", controls: 1, aux: "S390XCCMaskUint8"}, // unsigned 64-bit integer comparison
	}

	archs = append(archs, arch{
		name:               "S390X",
		pkg:                "cmd/internal/obj/s390x",
		genfile:            "../../s390x/ssa.go",
		ops:                S390Xops,
		blocks:             S390Xblocks,
		regnames:           regNamesS390X,
		ParamIntRegNames:   "R2 R3 R4 R5 R6 R7 R8 R9",
		ParamFloatRegNames: "F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15",
		gpregmask:          gp,
		fpregmask:          fp,
		framepointerreg:    -1, // not used
		linkreg:            int8(num["R14"]),
		imports: []string{
			"cmd/internal/obj/s390x",
		},
	})
}
