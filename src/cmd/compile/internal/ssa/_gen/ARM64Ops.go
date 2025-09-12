// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// Notes:
//  - Integer types live in the low portion of registers. Upper portions are junk.
//  - Boolean types use the low-order byte of a register. 0=false, 1=true.
//    Upper bytes are junk.
//  - *const instructions may use a constant larger than the instruction can encode.
//    In this case the assembler expands to multiple instructions and uses tmp
//    register (R27).
//  - All 32-bit Ops will zero the upper 32 bits of the destination register.

// Suffixes encode the bit width of various instructions.
// D (double word) = 64 bit
// W (word)        = 32 bit
// H (half word)   = 16 bit
// HU              = 16 bit unsigned
// B (byte)        = 8 bit
// BU              = 8 bit unsigned
// S (single)      = 32 bit float
// D (double)      = 64 bit float

// Note: registers not used in regalloc are not included in this list,
// so that regmask stays within int64
// Be careful when hand coding regmasks.
var regNamesARM64 = []string{
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
	"R13",
	"R14",
	"R15",
	"R16",
	"R17",
	// R18 = platform register, not used
	"R19",
	"R20",
	"R21",
	"R22",
	"R23",
	"R24",
	"R25",
	"R26",
	// R27 = REGTMP not used in regalloc
	"g",    // aka R28
	"R29",  // frame pointer, not used
	"R30",  // aka REGLINK
	"ZERO", // zero register (aka R31)
	"SP",   // stack pointer (aka R31)

	// Note: both ZERO and SP are register number 31!
	// What r31 means in a particular instruction depends on
	// the instruction.  Generally, for arguments of instructions
	// which are addresses to load or store from, r31 means SP.
	// In other instructions, r31 means ZERO. But there are
	// exceptions.
	// See https://stackoverflow.com/questions/61532867
	// This does not have much of an effect here, as the
	// cmd/internal/obj/arm64 interface treats them as two
	// different registers and picks the right instruction
	// that encodes what r31 means. But see issue 71651.

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
	"F16",
	"F17",
	"F18",
	"F19",
	"F20",
	"F21",
	"F22",
	"F23",
	"F24",
	"F25",
	"F26",
	"F27",
	"F28",
	"F29",
	"F30",
	"F31",

	// If you add registers, update asyncPreempt in runtime.

	// pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesARM64) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesARM64 {
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
		gp         = buildReg("R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30")
		gpg        = gp | buildReg("g")
		gpsp       = gp | buildReg("SP")
		gpspg      = gpg | buildReg("SP")
		gpspsbg    = gpspg | buildReg("SB")
		fp         = buildReg("F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31")
		callerSave = gp | fp | buildReg("g") // runtime.setg (and anything calling it) may clobber g
		r24to25    = buildReg("R24 R25")
		r23to25    = buildReg("R23 R24 R25")
		rz         = buildReg("ZERO")
		first16    = buildReg("R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15")
	)
	// Common regInfo
	var (
		gp01           = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp0flags1      = regInfo{inputs: []regMask{0}, outputs: []regMask{gp}}
		gp11           = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp11sp         = regInfo{inputs: []regMask{gpspg}, outputs: []regMask{gp}}
		gp1flags       = regInfo{inputs: []regMask{gpg}}
		gp1flagsflags  = regInfo{inputs: []regMask{gpg}}
		gp1flags1      = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp11flags      = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp, 0}}
		gp21           = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp}}
		gp21nog        = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp}}
		gp21flags      = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp, 0}}
		gp2flags       = regInfo{inputs: []regMask{gpg, gpg}}
		gp2flagsflags  = regInfo{inputs: []regMask{gpg, gpg}}
		gp2flags1      = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp}}
		gp2flags1flags = regInfo{inputs: []regMask{gp, gp, 0}, outputs: []regMask{gp, 0}}
		gp2load        = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{gp}}
		gp31           = regInfo{inputs: []regMask{gpg, gpg, gpg}, outputs: []regMask{gp}}
		gpload         = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}
		gpload2        = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gpg, gpg}}
		gpstore        = regInfo{inputs: []regMask{gpspsbg, gpg | rz}}
		gpstore2       = regInfo{inputs: []regMask{gpspsbg, gpg | rz, gpg | rz}}
		gpxchg         = regInfo{inputs: []regMask{gpspsbg, gpg | rz}, outputs: []regMask{gp}}
		gpcas          = regInfo{inputs: []regMask{gpspsbg, gpg | rz, gpg | rz}, outputs: []regMask{gp}}
		fp01           = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11           = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fpgp           = regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}}
		gpfp           = regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}}
		fp21           = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp31           = regInfo{inputs: []regMask{fp, fp, fp}, outputs: []regMask{fp}}
		fp2flags       = regInfo{inputs: []regMask{fp, fp}}
		fp1flags       = regInfo{inputs: []regMask{fp}}
		fpload         = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{fp}}
		fpload2        = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{fp, fp}}
		fp2load        = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{fp}}
		fpstore        = regInfo{inputs: []regMask{gpspsbg, fp}}
		fpstoreidx     = regInfo{inputs: []regMask{gpspsbg, gpg, fp}}
		fpstore2       = regInfo{inputs: []regMask{gpspsbg, fp, fp}}
		readflags      = regInfo{inputs: nil, outputs: []regMask{gp}}
		prefreg        = regInfo{inputs: []regMask{gpspsbg}}
	)
	ops := []opData{
		// binary ops
		{name: "ADCSflags", argLength: 3, reg: gp2flags1flags, typ: "(UInt64,Flags)", asm: "ADCS", commutative: true}, // arg0+arg1+carry, set flags.
		{name: "ADCzerocarry", argLength: 1, reg: gp0flags1, typ: "UInt64", asm: "ADC"},                               // ZR+ZR+carry
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},                                         // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11sp, asm: "ADD", aux: "Int64"},                                       // arg0 + auxInt
		{name: "ADDSconstflags", argLength: 1, reg: gp11flags, typ: "(UInt64,Flags)", asm: "ADDS", aux: "Int64"},      // arg0+auxint, set flags.
		{name: "ADDSflags", argLength: 2, reg: gp21flags, typ: "(UInt64,Flags)", asm: "ADDS", commutative: true},      // arg0+arg1, set flags.
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB"},                                                            // arg0 - arg1
		{name: "SUBconst", argLength: 1, reg: gp11, asm: "SUB", aux: "Int64"},                                         // arg0 - auxInt
		{name: "SBCSflags", argLength: 3, reg: gp2flags1flags, typ: "(UInt64,Flags)", asm: "SBCS"},                    // arg0-(arg1+borrowing), set flags.
		{name: "SUBSflags", argLength: 2, reg: gp21flags, typ: "(UInt64,Flags)", asm: "SUBS"},                         // arg0 - arg1, set flags.
		{name: "MUL", argLength: 2, reg: gp21, asm: "MUL", commutative: true},                                         // arg0 * arg1
		{name: "MULW", argLength: 2, reg: gp21, asm: "MULW", commutative: true},                                       // arg0 * arg1, 32-bit
		{name: "MNEG", argLength: 2, reg: gp21, asm: "MNEG", commutative: true},                                       // -arg0 * arg1
		{name: "MNEGW", argLength: 2, reg: gp21, asm: "MNEGW", commutative: true},                                     // -arg0 * arg1, 32-bit
		{name: "MULH", argLength: 2, reg: gp21, asm: "SMULH", commutative: true},                                      // (arg0 * arg1) >> 64, signed
		{name: "UMULH", argLength: 2, reg: gp21, asm: "UMULH", commutative: true},                                     // (arg0 * arg1) >> 64, unsigned
		{name: "MULL", argLength: 2, reg: gp21, asm: "SMULL", commutative: true},                                      // arg0 * arg1, signed, 32-bit mult results in 64-bit
		{name: "UMULL", argLength: 2, reg: gp21, asm: "UMULL", commutative: true},                                     // arg0 * arg1, unsigned, 32-bit mult results in 64-bit
		{name: "DIV", argLength: 2, reg: gp21, asm: "SDIV"},                                                           // arg0 / arg1, signed
		{name: "UDIV", argLength: 2, reg: gp21, asm: "UDIV"},                                                          // arg0 / arg1, unsigned
		{name: "DIVW", argLength: 2, reg: gp21, asm: "SDIVW"},                                                         // arg0 / arg1, signed, 32 bit
		{name: "UDIVW", argLength: 2, reg: gp21, asm: "UDIVW"},                                                        // arg0 / arg1, unsigned, 32 bit
		{name: "MOD", argLength: 2, reg: gp21, asm: "REM"},                                                            // arg0 % arg1, signed
		{name: "UMOD", argLength: 2, reg: gp21, asm: "UREM"},                                                          // arg0 % arg1, unsigned
		{name: "MODW", argLength: 2, reg: gp21, asm: "REMW"},                                                          // arg0 % arg1, signed, 32 bit
		{name: "UMODW", argLength: 2, reg: gp21, asm: "UREMW"},                                                        // arg0 % arg1, unsigned, 32 bit

		{name: "FADDS", argLength: 2, reg: fp21, asm: "FADDS", commutative: true},   // arg0 + arg1
		{name: "FADDD", argLength: 2, reg: fp21, asm: "FADDD", commutative: true},   // arg0 + arg1
		{name: "FSUBS", argLength: 2, reg: fp21, asm: "FSUBS"},                      // arg0 - arg1
		{name: "FSUBD", argLength: 2, reg: fp21, asm: "FSUBD"},                      // arg0 - arg1
		{name: "FMULS", argLength: 2, reg: fp21, asm: "FMULS", commutative: true},   // arg0 * arg1
		{name: "FMULD", argLength: 2, reg: fp21, asm: "FMULD", commutative: true},   // arg0 * arg1
		{name: "FNMULS", argLength: 2, reg: fp21, asm: "FNMULS", commutative: true}, // -(arg0 * arg1)
		{name: "FNMULD", argLength: 2, reg: fp21, asm: "FNMULD", commutative: true}, // -(arg0 * arg1)
		{name: "FDIVS", argLength: 2, reg: fp21, asm: "FDIVS"},                      // arg0 / arg1
		{name: "FDIVD", argLength: 2, reg: fp21, asm: "FDIVD"},                      // arg0 / arg1

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true}, // arg0 & arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int64"}, // arg0 & auxInt
		{name: "OR", argLength: 2, reg: gp21, asm: "ORR", commutative: true},  // arg0 | arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "ORR", aux: "Int64"},  // arg0 | auxInt
		{name: "XOR", argLength: 2, reg: gp21, asm: "EOR", commutative: true}, // arg0 ^ arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "EOR", aux: "Int64"}, // arg0 ^ auxInt
		{name: "BIC", argLength: 2, reg: gp21, asm: "BIC"},                    // arg0 &^ arg1
		{name: "EON", argLength: 2, reg: gp21, asm: "EON"},                    // arg0 ^ ^arg1
		{name: "ORN", argLength: 2, reg: gp21, asm: "ORN"},                    // arg0 | ^arg1

		// unary ops
		{name: "MVN", argLength: 1, reg: gp11, asm: "MVN"},                                    // ^arg0
		{name: "NEG", argLength: 1, reg: gp11, asm: "NEG"},                                    // -arg0
		{name: "NEGSflags", argLength: 1, reg: gp11flags, typ: "(UInt64,Flags)", asm: "NEGS"}, // -arg0, set flags.
		{name: "NGCzerocarry", argLength: 1, reg: gp0flags1, typ: "UInt64", asm: "NGC"},       // -1 if borrowing, 0 otherwise.
		{name: "FABSD", argLength: 1, reg: fp11, asm: "FABSD"},                                // abs(arg0), float64
		{name: "FNEGS", argLength: 1, reg: fp11, asm: "FNEGS"},                                // -arg0, float32
		{name: "FNEGD", argLength: 1, reg: fp11, asm: "FNEGD"},                                // -arg0, float64
		{name: "FSQRTD", argLength: 1, reg: fp11, asm: "FSQRTD"},                              // sqrt(arg0), float64
		{name: "FSQRTS", argLength: 1, reg: fp11, asm: "FSQRTS"},                              // sqrt(arg0), float32
		{name: "FMIND", argLength: 2, reg: fp21, asm: "FMIND"},                                // min(arg0, arg1)
		{name: "FMINS", argLength: 2, reg: fp21, asm: "FMINS"},                                // min(arg0, arg1)
		{name: "FMAXD", argLength: 2, reg: fp21, asm: "FMAXD"},                                // max(arg0, arg1)
		{name: "FMAXS", argLength: 2, reg: fp21, asm: "FMAXS"},                                // max(arg0, arg1)
		{name: "REV", argLength: 1, reg: gp11, asm: "REV"},                                    // byte reverse, 64-bit
		{name: "REVW", argLength: 1, reg: gp11, asm: "REVW"},                                  // byte reverse, 32-bit
		{name: "REV16", argLength: 1, reg: gp11, asm: "REV16"},                                // byte reverse in each 16-bit halfword, 64-bit
		{name: "REV16W", argLength: 1, reg: gp11, asm: "REV16W"},                              // byte reverse in each 16-bit halfword, 32-bit
		{name: "RBIT", argLength: 1, reg: gp11, asm: "RBIT"},                                  // bit reverse, 64-bit
		{name: "RBITW", argLength: 1, reg: gp11, asm: "RBITW"},                                // bit reverse, 32-bit
		{name: "CLZ", argLength: 1, reg: gp11, asm: "CLZ"},                                    // count leading zero, 64-bit
		{name: "CLZW", argLength: 1, reg: gp11, asm: "CLZW"},                                  // count leading zero, 32-bit
		{name: "VCNT", argLength: 1, reg: fp11, asm: "VCNT"},                                  // count set bits for each 8-bit unit and store the result in each 8-bit unit
		{name: "VUADDLV", argLength: 1, reg: fp11, asm: "VUADDLV"},                            // unsigned sum of eight bytes in a 64-bit value, zero extended to 64-bit.
		{name: "LoweredRound32F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},
		{name: "LoweredRound64F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},

		// 3-operand, the addend comes first
		{name: "FMADDS", argLength: 3, reg: fp31, asm: "FMADDS"},   // +arg0 + (arg1 * arg2)
		{name: "FMADDD", argLength: 3, reg: fp31, asm: "FMADDD"},   // +arg0 + (arg1 * arg2)
		{name: "FNMADDS", argLength: 3, reg: fp31, asm: "FNMADDS"}, // -arg0 - (arg1 * arg2)
		{name: "FNMADDD", argLength: 3, reg: fp31, asm: "FNMADDD"}, // -arg0 - (arg1 * arg2)
		{name: "FMSUBS", argLength: 3, reg: fp31, asm: "FMSUBS"},   // +arg0 - (arg1 * arg2)
		{name: "FMSUBD", argLength: 3, reg: fp31, asm: "FMSUBD"},   // +arg0 - (arg1 * arg2)
		{name: "FNMSUBS", argLength: 3, reg: fp31, asm: "FNMSUBS"}, // -arg0 + (arg1 * arg2)
		{name: "FNMSUBD", argLength: 3, reg: fp31, asm: "FNMSUBD"}, // -arg0 + (arg1 * arg2)
		{name: "MADD", argLength: 3, reg: gp31, asm: "MADD"},       // +arg0 + (arg1 * arg2)
		{name: "MADDW", argLength: 3, reg: gp31, asm: "MADDW"},     // +arg0 + (arg1 * arg2), 32-bit
		{name: "MSUB", argLength: 3, reg: gp31, asm: "MSUB"},       // +arg0 - (arg1 * arg2)
		{name: "MSUBW", argLength: 3, reg: gp31, asm: "MSUBW"},     // +arg0 - (arg1 * arg2), 32-bit

		// shifts
		{name: "SLL", argLength: 2, reg: gp21, asm: "LSL"},                        // arg0 << arg1, shift amount is mod 64
		{name: "SLLconst", argLength: 1, reg: gp11, asm: "LSL", aux: "Int64"},     // arg0 << auxInt, auxInt should be in the range 0 to 63.
		{name: "SRL", argLength: 2, reg: gp21, asm: "LSR"},                        // arg0 >> arg1, unsigned, shift amount is mod 64
		{name: "SRLconst", argLength: 1, reg: gp11, asm: "LSR", aux: "Int64"},     // arg0 >> auxInt, unsigned, auxInt should be in the range 0 to 63.
		{name: "SRA", argLength: 2, reg: gp21, asm: "ASR"},                        // arg0 >> arg1, signed, shift amount is mod 64
		{name: "SRAconst", argLength: 1, reg: gp11, asm: "ASR", aux: "Int64"},     // arg0 >> auxInt, signed, auxInt should be in the range 0 to 63.
		{name: "ROR", argLength: 2, reg: gp21, asm: "ROR"},                        // arg0 right rotate by (arg1 mod 64) bits
		{name: "RORW", argLength: 2, reg: gp21, asm: "RORW"},                      // arg0 right rotate by (arg1 mod 32) bits
		{name: "RORconst", argLength: 1, reg: gp11, asm: "ROR", aux: "Int64"},     // arg0 right rotate by auxInt bits, auxInt should be in the range 0 to 63.
		{name: "RORWconst", argLength: 1, reg: gp11, asm: "RORW", aux: "Int64"},   // uint32(arg0) right rotate by auxInt bits, auxInt should be in the range 0 to 31.
		{name: "EXTRconst", argLength: 2, reg: gp21, asm: "EXTR", aux: "Int64"},   // extract 64 bits from arg0:arg1 starting at lsb auxInt, auxInt should be in the range 0 to 63.
		{name: "EXTRWconst", argLength: 2, reg: gp21, asm: "EXTRW", aux: "Int64"}, // extract 32 bits from arg0[31:0]:arg1[31:0] starting at lsb auxInt and zero top 32 bits, auxInt should be in the range 0 to 31.

		// comparisons
		{name: "CMP", argLength: 2, reg: gp2flags, asm: "CMP", typ: "Flags"},                      // arg0 compare to arg1
		{name: "CMPconst", argLength: 1, reg: gp1flags, asm: "CMP", aux: "Int64", typ: "Flags"},   // arg0 compare to auxInt
		{name: "CMPW", argLength: 2, reg: gp2flags, asm: "CMPW", typ: "Flags"},                    // arg0 compare to arg1, 32 bit
		{name: "CMPWconst", argLength: 1, reg: gp1flags, asm: "CMPW", aux: "Int32", typ: "Flags"}, // arg0 compare to auxInt, 32 bit
		{name: "CMN", argLength: 2, reg: gp2flags, asm: "CMN", typ: "Flags", commutative: true},   // arg0 compare to -arg1, provided arg1 is not 1<<63
		{name: "CMNconst", argLength: 1, reg: gp1flags, asm: "CMN", aux: "Int64", typ: "Flags"},   // arg0 compare to -auxInt
		{name: "CMNW", argLength: 2, reg: gp2flags, asm: "CMNW", typ: "Flags", commutative: true}, // arg0 compare to -arg1, 32 bit, provided arg1 is not 1<<31
		{name: "CMNWconst", argLength: 1, reg: gp1flags, asm: "CMNW", aux: "Int32", typ: "Flags"}, // arg0 compare to -auxInt, 32 bit
		{name: "TST", argLength: 2, reg: gp2flags, asm: "TST", typ: "Flags", commutative: true},   // arg0 & arg1 compare to 0
		{name: "TSTconst", argLength: 1, reg: gp1flags, asm: "TST", aux: "Int64", typ: "Flags"},   // arg0 & auxInt compare to 0
		{name: "TSTW", argLength: 2, reg: gp2flags, asm: "TSTW", typ: "Flags", commutative: true}, // arg0 & arg1 compare to 0, 32 bit
		{name: "TSTWconst", argLength: 1, reg: gp1flags, asm: "TSTW", aux: "Int32", typ: "Flags"}, // arg0 & auxInt compare to 0, 32 bit
		{name: "FCMPS", argLength: 2, reg: fp2flags, asm: "FCMPS", typ: "Flags"},                  // arg0 compare to arg1, float32
		{name: "FCMPD", argLength: 2, reg: fp2flags, asm: "FCMPD", typ: "Flags"},                  // arg0 compare to arg1, float64
		{name: "FCMPS0", argLength: 1, reg: fp1flags, asm: "FCMPS", typ: "Flags"},                 // arg0 compare to 0, float32
		{name: "FCMPD0", argLength: 1, reg: fp1flags, asm: "FCMPD", typ: "Flags"},                 // arg0 compare to 0, float64

		// shifted ops
		{name: "MVNshiftLL", argLength: 1, reg: gp11, asm: "MVN", aux: "Int64"},                   // ^(arg0<<auxInt), auxInt should be in the range 0 to 63.
		{name: "MVNshiftRL", argLength: 1, reg: gp11, asm: "MVN", aux: "Int64"},                   // ^(arg0>>auxInt), unsigned shift, auxInt should be in the range 0 to 63.
		{name: "MVNshiftRA", argLength: 1, reg: gp11, asm: "MVN", aux: "Int64"},                   // ^(arg0>>auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "MVNshiftRO", argLength: 1, reg: gp11, asm: "MVN", aux: "Int64"},                   // ^(arg0 ROR auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "NEGshiftLL", argLength: 1, reg: gp11, asm: "NEG", aux: "Int64"},                   // -(arg0<<auxInt), auxInt should be in the range 0 to 63.
		{name: "NEGshiftRL", argLength: 1, reg: gp11, asm: "NEG", aux: "Int64"},                   // -(arg0>>auxInt), unsigned shift, auxInt should be in the range 0 to 63.
		{name: "NEGshiftRA", argLength: 1, reg: gp11, asm: "NEG", aux: "Int64"},                   // -(arg0>>auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "ADDshiftLL", argLength: 2, reg: gp21, asm: "ADD", aux: "Int64"},                   // arg0 + arg1<<auxInt, auxInt should be in the range 0 to 63.
		{name: "ADDshiftRL", argLength: 2, reg: gp21, asm: "ADD", aux: "Int64"},                   // arg0 + arg1>>auxInt, unsigned shift, auxInt should be in the range 0 to 63.
		{name: "ADDshiftRA", argLength: 2, reg: gp21, asm: "ADD", aux: "Int64"},                   // arg0 + arg1>>auxInt, signed shift, auxInt should be in the range 0 to 63.
		{name: "SUBshiftLL", argLength: 2, reg: gp21, asm: "SUB", aux: "Int64"},                   // arg0 - arg1<<auxInt, auxInt should be in the range 0 to 63.
		{name: "SUBshiftRL", argLength: 2, reg: gp21, asm: "SUB", aux: "Int64"},                   // arg0 - arg1>>auxInt, unsigned shift, auxInt should be in the range 0 to 63.
		{name: "SUBshiftRA", argLength: 2, reg: gp21, asm: "SUB", aux: "Int64"},                   // arg0 - arg1>>auxInt, signed shift, auxInt should be in the range 0 to 63.
		{name: "ANDshiftLL", argLength: 2, reg: gp21, asm: "AND", aux: "Int64"},                   // arg0 & (arg1<<auxInt), auxInt should be in the range 0 to 63.
		{name: "ANDshiftRL", argLength: 2, reg: gp21, asm: "AND", aux: "Int64"},                   // arg0 & (arg1>>auxInt), unsigned shift, auxInt should be in the range 0 to 63.
		{name: "ANDshiftRA", argLength: 2, reg: gp21, asm: "AND", aux: "Int64"},                   // arg0 & (arg1>>auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "ANDshiftRO", argLength: 2, reg: gp21, asm: "AND", aux: "Int64"},                   // arg0 & (arg1 ROR auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "ORshiftLL", argLength: 2, reg: gp21, asm: "ORR", aux: "Int64"},                    // arg0 | arg1<<auxInt, auxInt should be in the range 0 to 63.
		{name: "ORshiftRL", argLength: 2, reg: gp21, asm: "ORR", aux: "Int64"},                    // arg0 | arg1>>auxInt, unsigned shift, auxInt should be in the range 0 to 63.
		{name: "ORshiftRA", argLength: 2, reg: gp21, asm: "ORR", aux: "Int64"},                    // arg0 | arg1>>auxInt, signed shift, auxInt should be in the range 0 to 63.
		{name: "ORshiftRO", argLength: 2, reg: gp21, asm: "ORR", aux: "Int64"},                    // arg0 | arg1 ROR auxInt, signed shift, auxInt should be in the range 0 to 63.
		{name: "XORshiftLL", argLength: 2, reg: gp21, asm: "EOR", aux: "Int64"},                   // arg0 ^ arg1<<auxInt, auxInt should be in the range 0 to 63.
		{name: "XORshiftRL", argLength: 2, reg: gp21, asm: "EOR", aux: "Int64"},                   // arg0 ^ arg1>>auxInt, unsigned shift, auxInt should be in the range 0 to 63.
		{name: "XORshiftRA", argLength: 2, reg: gp21, asm: "EOR", aux: "Int64"},                   // arg0 ^ arg1>>auxInt, signed shift, auxInt should be in the range 0 to 63.
		{name: "XORshiftRO", argLength: 2, reg: gp21, asm: "EOR", aux: "Int64"},                   // arg0 ^ arg1 ROR auxInt, signed shift, auxInt should be in the range 0 to 63.
		{name: "BICshiftLL", argLength: 2, reg: gp21, asm: "BIC", aux: "Int64"},                   // arg0 &^ (arg1<<auxInt), auxInt should be in the range 0 to 63.
		{name: "BICshiftRL", argLength: 2, reg: gp21, asm: "BIC", aux: "Int64"},                   // arg0 &^ (arg1>>auxInt), unsigned shift, auxInt should be in the range 0 to 63.
		{name: "BICshiftRA", argLength: 2, reg: gp21, asm: "BIC", aux: "Int64"},                   // arg0 &^ (arg1>>auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "BICshiftRO", argLength: 2, reg: gp21, asm: "BIC", aux: "Int64"},                   // arg0 &^ (arg1 ROR auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "EONshiftLL", argLength: 2, reg: gp21, asm: "EON", aux: "Int64"},                   // arg0 ^ ^(arg1<<auxInt), auxInt should be in the range 0 to 63.
		{name: "EONshiftRL", argLength: 2, reg: gp21, asm: "EON", aux: "Int64"},                   // arg0 ^ ^(arg1>>auxInt), unsigned shift, auxInt should be in the range 0 to 63.
		{name: "EONshiftRA", argLength: 2, reg: gp21, asm: "EON", aux: "Int64"},                   // arg0 ^ ^(arg1>>auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "EONshiftRO", argLength: 2, reg: gp21, asm: "EON", aux: "Int64"},                   // arg0 ^ ^(arg1 ROR auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "ORNshiftLL", argLength: 2, reg: gp21, asm: "ORN", aux: "Int64"},                   // arg0 | ^(arg1<<auxInt), auxInt should be in the range 0 to 63.
		{name: "ORNshiftRL", argLength: 2, reg: gp21, asm: "ORN", aux: "Int64"},                   // arg0 | ^(arg1>>auxInt), unsigned shift, auxInt should be in the range 0 to 63.
		{name: "ORNshiftRA", argLength: 2, reg: gp21, asm: "ORN", aux: "Int64"},                   // arg0 | ^(arg1>>auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "ORNshiftRO", argLength: 2, reg: gp21, asm: "ORN", aux: "Int64"},                   // arg0 | ^(arg1 ROR auxInt), signed shift, auxInt should be in the range 0 to 63.
		{name: "CMPshiftLL", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int64", typ: "Flags"}, // arg0 compare to arg1<<auxInt, auxInt should be in the range 0 to 63.
		{name: "CMPshiftRL", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int64", typ: "Flags"}, // arg0 compare to arg1>>auxInt, unsigned shift, auxInt should be in the range 0 to 63.
		{name: "CMPshiftRA", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int64", typ: "Flags"}, // arg0 compare to arg1>>auxInt, signed shift, auxInt should be in the range 0 to 63.
		{name: "CMNshiftLL", argLength: 2, reg: gp2flags, asm: "CMN", aux: "Int64", typ: "Flags"}, // (arg0 + arg1<<auxInt) compare to 0, auxInt should be in the range 0 to 63.
		{name: "CMNshiftRL", argLength: 2, reg: gp2flags, asm: "CMN", aux: "Int64", typ: "Flags"}, // (arg0 + arg1>>auxInt) compare to 0, unsigned shift, auxInt should be in the range 0 to 63.
		{name: "CMNshiftRA", argLength: 2, reg: gp2flags, asm: "CMN", aux: "Int64", typ: "Flags"}, // (arg0 + arg1>>auxInt) compare to 0, signed shift, auxInt should be in the range 0 to 63.
		{name: "TSTshiftLL", argLength: 2, reg: gp2flags, asm: "TST", aux: "Int64", typ: "Flags"}, // (arg0 & arg1<<auxInt) compare to 0, auxInt should be in the range 0 to 63.
		{name: "TSTshiftRL", argLength: 2, reg: gp2flags, asm: "TST", aux: "Int64", typ: "Flags"}, // (arg0 & arg1>>auxInt) compare to 0, unsigned shift, auxInt should be in the range 0 to 63.
		{name: "TSTshiftRA", argLength: 2, reg: gp2flags, asm: "TST", aux: "Int64", typ: "Flags"}, // (arg0 & arg1>>auxInt) compare to 0, signed shift, auxInt should be in the range 0 to 63.
		{name: "TSTshiftRO", argLength: 2, reg: gp2flags, asm: "TST", aux: "Int64", typ: "Flags"}, // (arg0 & arg1 ROR auxInt) compare to 0, signed shift, auxInt should be in the range 0 to 63.

		// bitfield ops
		// for all bitfield ops lsb is auxInt>>8, width is auxInt&0xff
		// insert low width bits of arg1 into the result starting at bit lsb, copy other bits from arg0
		{name: "BFI", argLength: 2, reg: gp21nog, asm: "BFI", aux: "ARM64BitField", resultInArg0: true},
		// extract width bits of arg1 starting at bit lsb and insert at low end of result, copy other bits from arg0
		{name: "BFXIL", argLength: 2, reg: gp21nog, asm: "BFXIL", aux: "ARM64BitField", resultInArg0: true},
		// insert low width bits of arg0 into the result starting at bit lsb, bits to the left of the inserted bit field are set to the high/sign bit of the inserted bit field, bits to the right are zeroed
		{name: "SBFIZ", argLength: 1, reg: gp11, asm: "SBFIZ", aux: "ARM64BitField"},
		// extract width bits of arg0 starting at bit lsb and insert at low end of result, remaining high bits are set to the high/sign bit of the extracted bitfield
		{name: "SBFX", argLength: 1, reg: gp11, asm: "SBFX", aux: "ARM64BitField"},
		// insert low width bits of arg0 into the result starting at bit lsb, bits to the left and right of the inserted bit field are zeroed
		{name: "UBFIZ", argLength: 1, reg: gp11, asm: "UBFIZ", aux: "ARM64BitField"},
		// extract width bits of arg0 starting at bit lsb and insert at low end of result, remaining high bits are zeroed
		{name: "UBFX", argLength: 1, reg: gp11, asm: "UBFX", aux: "ARM64BitField"},

		// moves
		{name: "MOVDconst", argLength: 0, reg: gp01, aux: "Int64", asm: "MOVD", typ: "UInt64", rematerializeable: true},      // 64 bits from auxint
		{name: "FMOVSconst", argLength: 0, reg: fp01, aux: "Float64", asm: "FMOVS", typ: "Float32", rematerializeable: true}, // auxint as 64-bit float, convert to 32-bit float
		{name: "FMOVDconst", argLength: 0, reg: fp01, aux: "Float64", asm: "FMOVD", typ: "Float64", rematerializeable: true}, // auxint as 64-bit float

		{name: "MOVDaddr", argLength: 1, reg: regInfo{inputs: []regMask{buildReg("SP") | buildReg("SB")}, outputs: []regMask{gp}}, aux: "SymOff", asm: "MOVD", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxInt + aux.(*gc.Sym), arg0=SP/SB

		{name: "MOVBload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVB", typ: "Int8", faultOnNilArg0: true, symEffect: "Read"},      // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVBUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVBU", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},   // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVH", typ: "Int16", faultOnNilArg0: true, symEffect: "Read"},     // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVHU", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVW", typ: "Int32", faultOnNilArg0: true, symEffect: "Read"},     // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVWU", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVDload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVD", typ: "UInt64", faultOnNilArg0: true, symEffect: "Read"},    // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "FMOVSload", argLength: 2, reg: fpload, aux: "SymOff", asm: "FMOVS", typ: "Float32", faultOnNilArg0: true, symEffect: "Read"}, // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "FMOVDload", argLength: 2, reg: fpload, aux: "SymOff", asm: "FMOVD", typ: "Float64", faultOnNilArg0: true, symEffect: "Read"}, // load from arg0 + auxInt + aux.  arg1=mem.

		// LDP instructions load the contents of two adjacent locations in memory into registers.
		// Address to start loading is addr = arg0 + auxInt + aux.
		// x := *(*T)(addr)
		// y := *(*T)(addr+sizeof(T))
		// arg1=mem
		// Returns the tuple <x,y>.
		{name: "LDP", argLength: 2, reg: gpload2, aux: "SymOff", asm: "LDP", typ: "(UInt64,UInt64)", faultOnNilArg0: true, symEffect: "Read"},       // T=int64 (gp reg destination)
		{name: "LDPW", argLength: 2, reg: gpload2, aux: "SymOff", asm: "LDPW", typ: "(UInt32,UInt32)", faultOnNilArg0: true, symEffect: "Read"},     // T=int32 (gp reg destination) unsigned extension
		{name: "LDPSW", argLength: 2, reg: gpload2, aux: "SymOff", asm: "LDPSW", typ: "(Int32,Int32)", faultOnNilArg0: true, symEffect: "Read"},     // T=int32 (gp reg destination) signed extension
		{name: "FLDPD", argLength: 2, reg: fpload2, aux: "SymOff", asm: "FLDPD", typ: "(Float64,Float64)", faultOnNilArg0: true, symEffect: "Read"}, // T=float64 (fp reg destination)
		{name: "FLDPS", argLength: 2, reg: fpload2, aux: "SymOff", asm: "FLDPS", typ: "(Float32,Float32)", faultOnNilArg0: true, symEffect: "Read"}, // T=float32 (fp reg destination)

		// register indexed load
		{name: "MOVDloadidx", argLength: 3, reg: gp2load, asm: "MOVD", typ: "UInt64"},    // load 64-bit dword from arg0 + arg1, arg2 = mem.
		{name: "MOVWloadidx", argLength: 3, reg: gp2load, asm: "MOVW", typ: "Int32"},     // load 32-bit word from arg0 + arg1, sign-extended to 64-bit, arg2=mem.
		{name: "MOVWUloadidx", argLength: 3, reg: gp2load, asm: "MOVWU", typ: "UInt32"},  // load 32-bit word from arg0 + arg1, zero-extended to 64-bit, arg2=mem.
		{name: "MOVHloadidx", argLength: 3, reg: gp2load, asm: "MOVH", typ: "Int16"},     // load 16-bit word from arg0 + arg1, sign-extended to 64-bit, arg2=mem.
		{name: "MOVHUloadidx", argLength: 3, reg: gp2load, asm: "MOVHU", typ: "UInt16"},  // load 16-bit word from arg0 + arg1, zero-extended to 64-bit, arg2=mem.
		{name: "MOVBloadidx", argLength: 3, reg: gp2load, asm: "MOVB", typ: "Int8"},      // load 8-bit word from arg0 + arg1, sign-extended to 64-bit, arg2=mem.
		{name: "MOVBUloadidx", argLength: 3, reg: gp2load, asm: "MOVBU", typ: "UInt8"},   // load 8-bit word from arg0 + arg1, zero-extended to 64-bit, arg2=mem.
		{name: "FMOVSloadidx", argLength: 3, reg: fp2load, asm: "FMOVS", typ: "Float32"}, // load 32-bit float from arg0 + arg1, arg2=mem.
		{name: "FMOVDloadidx", argLength: 3, reg: fp2load, asm: "FMOVD", typ: "Float64"}, // load 64-bit float from arg0 + arg1, arg2=mem.

		// shifted register indexed load
		{name: "MOVHloadidx2", argLength: 3, reg: gp2load, asm: "MOVH", typ: "Int16"},     // load 16-bit half-word from arg0 + arg1*2, sign-extended to 64-bit, arg2=mem.
		{name: "MOVHUloadidx2", argLength: 3, reg: gp2load, asm: "MOVHU", typ: "UInt16"},  // load 16-bit half-word from arg0 + arg1*2, zero-extended to 64-bit, arg2=mem.
		{name: "MOVWloadidx4", argLength: 3, reg: gp2load, asm: "MOVW", typ: "Int32"},     // load 32-bit word from arg0 + arg1*4, sign-extended to 64-bit, arg2=mem.
		{name: "MOVWUloadidx4", argLength: 3, reg: gp2load, asm: "MOVWU", typ: "UInt32"},  // load 32-bit word from arg0 + arg1*4, zero-extended to 64-bit, arg2=mem.
		{name: "MOVDloadidx8", argLength: 3, reg: gp2load, asm: "MOVD", typ: "UInt64"},    // load 64-bit double-word from arg0 + arg1*8, arg2 = mem.
		{name: "FMOVSloadidx4", argLength: 3, reg: fp2load, asm: "FMOVS", typ: "Float32"}, // load 32-bit float from arg0 + arg1*4, arg2 = mem.
		{name: "FMOVDloadidx8", argLength: 3, reg: fp2load, asm: "FMOVD", typ: "Float64"}, // load 64-bit float from arg0 + arg1*8, arg2 = mem.

		{name: "MOVBstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 1 byte of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVHstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 2 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVWstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVDstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "FMOVSstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "FMOVS", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "FMOVDstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "FMOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.

		// STP instructions store the contents of two registers to adjacent locations in memory.
		// Address to start storing is addr = arg0 + auxInt + aux.
		// *(*T)(addr) = arg1
		// *(*T)(addr+sizeof(T)) = arg2
		// arg3=mem. Returns mem.
		{name: "STP", argLength: 4, reg: gpstore2, aux: "SymOff", asm: "STP", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},     // T=int64 (gp reg source)
		{name: "STPW", argLength: 4, reg: gpstore2, aux: "SymOff", asm: "STPW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // T=int32 (gp reg source)
		{name: "FSTPD", argLength: 4, reg: fpstore2, aux: "SymOff", asm: "FSTPD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // T=float64 (fp reg source)
		{name: "FSTPS", argLength: 4, reg: fpstore2, aux: "SymOff", asm: "FSTPS", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // T=float32 (fp reg source)

		// register indexed store
		{name: "MOVBstoreidx", argLength: 4, reg: gpstore2, asm: "MOVB", typ: "Mem"},     // store 1 byte of arg2 to arg0 + arg1, arg3 = mem.
		{name: "MOVHstoreidx", argLength: 4, reg: gpstore2, asm: "MOVH", typ: "Mem"},     // store 2 bytes of arg2 to arg0 + arg1, arg3 = mem.
		{name: "MOVWstoreidx", argLength: 4, reg: gpstore2, asm: "MOVW", typ: "Mem"},     // store 4 bytes of arg2 to arg0 + arg1, arg3 = mem.
		{name: "MOVDstoreidx", argLength: 4, reg: gpstore2, asm: "MOVD", typ: "Mem"},     // store 8 bytes of arg2 to arg0 + arg1, arg3 = mem.
		{name: "FMOVSstoreidx", argLength: 4, reg: fpstoreidx, asm: "FMOVS", typ: "Mem"}, // store 32-bit float of arg2 to arg0 + arg1, arg3=mem.
		{name: "FMOVDstoreidx", argLength: 4, reg: fpstoreidx, asm: "FMOVD", typ: "Mem"}, // store 64-bit float of arg2 to arg0 + arg1, arg3=mem.

		// shifted register indexed store
		{name: "MOVHstoreidx2", argLength: 4, reg: gpstore2, asm: "MOVH", typ: "Mem"},     // store 2 bytes of arg2 to arg0 + arg1*2, arg3 = mem.
		{name: "MOVWstoreidx4", argLength: 4, reg: gpstore2, asm: "MOVW", typ: "Mem"},     // store 4 bytes of arg2 to arg0 + arg1*4, arg3 = mem.
		{name: "MOVDstoreidx8", argLength: 4, reg: gpstore2, asm: "MOVD", typ: "Mem"},     // store 8 bytes of arg2 to arg0 + arg1*8, arg3 = mem.
		{name: "FMOVSstoreidx4", argLength: 4, reg: fpstoreidx, asm: "FMOVS", typ: "Mem"}, // store 32-bit float of arg2 to arg0 + arg1*4, arg3=mem.
		{name: "FMOVDstoreidx8", argLength: 4, reg: fpstoreidx, asm: "FMOVD", typ: "Mem"}, // store 64-bit float of arg2 to arg0 + arg1*8, arg3=mem.

		{name: "FMOVDgpfp", argLength: 1, reg: gpfp, asm: "FMOVD"}, // move int64 to float64 (no conversion)
		{name: "FMOVDfpgp", argLength: 1, reg: fpgp, asm: "FMOVD"}, // move float64 to int64 (no conversion)
		{name: "FMOVSgpfp", argLength: 1, reg: gpfp, asm: "FMOVS"}, // move 32bits from int to float reg (no conversion)
		{name: "FMOVSfpgp", argLength: 1, reg: fpgp, asm: "FMOVS"}, // move 32bits from float to int reg, zero extend (no conversion)

		// conversions
		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVB"},   // move from arg0, sign-extended from byte
		{name: "MOVBUreg", argLength: 1, reg: gp11, asm: "MOVBU"}, // move from arg0, unsign-extended from byte
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVH"},   // move from arg0, sign-extended from half
		{name: "MOVHUreg", argLength: 1, reg: gp11, asm: "MOVHU"}, // move from arg0, unsign-extended from half
		{name: "MOVWreg", argLength: 1, reg: gp11, asm: "MOVW"},   // move from arg0, sign-extended from word
		{name: "MOVWUreg", argLength: 1, reg: gp11, asm: "MOVWU"}, // move from arg0, unsign-extended from word
		{name: "MOVDreg", argLength: 1, reg: gp11, asm: "MOVD"},   // move from arg0

		{name: "MOVDnop", argLength: 1, reg: regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}, resultInArg0: true}, // nop, return arg0 in same register

		{name: "SCVTFWS", argLength: 1, reg: gpfp, asm: "SCVTFWS"},   // int32 -> float32
		{name: "SCVTFWD", argLength: 1, reg: gpfp, asm: "SCVTFWD"},   // int32 -> float64
		{name: "UCVTFWS", argLength: 1, reg: gpfp, asm: "UCVTFWS"},   // uint32 -> float32
		{name: "UCVTFWD", argLength: 1, reg: gpfp, asm: "UCVTFWD"},   // uint32 -> float64
		{name: "SCVTFS", argLength: 1, reg: gpfp, asm: "SCVTFS"},     // int64 -> float32
		{name: "SCVTFD", argLength: 1, reg: gpfp, asm: "SCVTFD"},     // int64 -> float64
		{name: "UCVTFS", argLength: 1, reg: gpfp, asm: "UCVTFS"},     // uint64 -> float32
		{name: "UCVTFD", argLength: 1, reg: gpfp, asm: "UCVTFD"},     // uint64 -> float64
		{name: "FCVTZSSW", argLength: 1, reg: fpgp, asm: "FCVTZSSW"}, // float32 -> int32
		{name: "FCVTZSDW", argLength: 1, reg: fpgp, asm: "FCVTZSDW"}, // float64 -> int32
		{name: "FCVTZUSW", argLength: 1, reg: fpgp, asm: "FCVTZUSW"}, // float32 -> uint32
		{name: "FCVTZUDW", argLength: 1, reg: fpgp, asm: "FCVTZUDW"}, // float64 -> uint32
		{name: "FCVTZSS", argLength: 1, reg: fpgp, asm: "FCVTZSS"},   // float32 -> int64
		{name: "FCVTZSD", argLength: 1, reg: fpgp, asm: "FCVTZSD"},   // float64 -> int64
		{name: "FCVTZUS", argLength: 1, reg: fpgp, asm: "FCVTZUS"},   // float32 -> uint64
		{name: "FCVTZUD", argLength: 1, reg: fpgp, asm: "FCVTZUD"},   // float64 -> uint64
		{name: "FCVTSD", argLength: 1, reg: fp11, asm: "FCVTSD"},     // float32 -> float64
		{name: "FCVTDS", argLength: 1, reg: fp11, asm: "FCVTDS"},     // float64 -> float32

		// floating-point round to integral
		{name: "FRINTAD", argLength: 1, reg: fp11, asm: "FRINTAD"},
		{name: "FRINTMD", argLength: 1, reg: fp11, asm: "FRINTMD"},
		{name: "FRINTND", argLength: 1, reg: fp11, asm: "FRINTND"},
		{name: "FRINTPD", argLength: 1, reg: fp11, asm: "FRINTPD"},
		{name: "FRINTZD", argLength: 1, reg: fp11, asm: "FRINTZD"},

		// conditional instructions; auxint is
		// one of the arm64 comparison pseudo-ops (LessThan, LessThanU, etc.)
		{name: "CSEL", argLength: 3, reg: gp2flags1, asm: "CSEL", aux: "CCop"},   // auxint(flags) ? arg0 : arg1
		{name: "CSEL0", argLength: 2, reg: gp1flags1, asm: "CSEL", aux: "CCop"},  // auxint(flags) ? arg0 : 0
		{name: "CSINC", argLength: 3, reg: gp2flags1, asm: "CSINC", aux: "CCop"}, // auxint(flags) ? arg0 : arg1 + 1
		{name: "CSINV", argLength: 3, reg: gp2flags1, asm: "CSINV", aux: "CCop"}, // auxint(flags) ? arg0 : ^arg1
		{name: "CSNEG", argLength: 3, reg: gp2flags1, asm: "CSNEG", aux: "CCop"}, // auxint(flags) ? arg0 : -arg1
		{name: "CSETM", argLength: 1, reg: readflags, asm: "CSETM", aux: "CCop"}, // auxint(flags) ? -1 : 0

		// conditional comparison instructions; auxint is
		// combination of Cond, Nzcv and optional ConstValue
		// Behavior:
		//   If the condition 'Cond' evaluates to true against current flags,
		//   flags are set to the result of the comparison operation.
		//   Otherwise, flags are set to the fallback value 'Nzcv'.
		{name: "CCMP", argLength: 3, reg: gp2flagsflags, asm: "CCMP", aux: "ARM64ConditionalParams", typ: "Flag"},      // If Cond then flags = CMP arg0 arg1 else flags = Nzcv
		{name: "CCMN", argLength: 3, reg: gp2flagsflags, asm: "CCMN", aux: "ARM64ConditionalParams", typ: "Flag"},      // If Cond then flags = CMN arg0 arg1 else flags = Nzcv
		{name: "CCMPconst", argLength: 2, reg: gp1flagsflags, asm: "CCMP", aux: "ARM64ConditionalParams", typ: "Flag"}, // If Cond then flags = CMPconst [ConstValue] arg0 else flags = Nzcv
		{name: "CCMNconst", argLength: 2, reg: gp1flagsflags, asm: "CCMN", aux: "ARM64ConditionalParams", typ: "Flag"}, // If Cond then flags = CMNconst [ConstValue] arg0 else flags = Nzcv

		{name: "CCMPW", argLength: 3, reg: gp2flagsflags, asm: "CCMPW", aux: "ARM64ConditionalParams", typ: "Flag"},      // If Cond then flags = CMPW arg0 arg1 else flags = Nzcv
		{name: "CCMNW", argLength: 3, reg: gp2flagsflags, asm: "CCMNW", aux: "ARM64ConditionalParams", typ: "Flag"},      // If Cond then flags = CMNW arg0 arg1 else flags = Nzcv
		{name: "CCMPWconst", argLength: 2, reg: gp1flagsflags, asm: "CCMPW", aux: "ARM64ConditionalParams", typ: "Flag"}, // If Cond then flags = CCMPWconst [ConstValue] arg0 else flags = Nzcv
		{name: "CCMNWconst", argLength: 2, reg: gp1flagsflags, asm: "CCMNW", aux: "ARM64ConditionalParams", typ: "Flag"}, // If Cond then flags = CCMNWconst [ConstValue] arg0 else flags = Nzcv

		// function calls
		{name: "CALLstatic", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                               // call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                 // tail call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: -1, reg: regInfo{inputs: []regMask{gpsp, buildReg("R26"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, last arg=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: -1, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                         // call fn by pointer.  arg0=codeptr, last arg=mem, auxint=argsize, returns mem

		// pseudo-ops
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gpg}}, nilCheck: true, faultOnNilArg0: true}, // panic if arg0 is nil.  arg1=mem.

		{name: "Equal", argLength: 1, reg: readflags},            // bool, true flags encode x==y false otherwise.
		{name: "NotEqual", argLength: 1, reg: readflags},         // bool, true flags encode x!=y false otherwise.
		{name: "LessThan", argLength: 1, reg: readflags},         // bool, true flags encode signed x<y false otherwise.
		{name: "LessEqual", argLength: 1, reg: readflags},        // bool, true flags encode signed x<=y false otherwise.
		{name: "GreaterThan", argLength: 1, reg: readflags},      // bool, true flags encode signed x>y false otherwise.
		{name: "GreaterEqual", argLength: 1, reg: readflags},     // bool, true flags encode signed x>=y false otherwise.
		{name: "LessThanU", argLength: 1, reg: readflags},        // bool, true flags encode unsigned x<y false otherwise.
		{name: "LessEqualU", argLength: 1, reg: readflags},       // bool, true flags encode unsigned x<=y false otherwise.
		{name: "GreaterThanU", argLength: 1, reg: readflags},     // bool, true flags encode unsigned x>y false otherwise.
		{name: "GreaterEqualU", argLength: 1, reg: readflags},    // bool, true flags encode unsigned x>=y false otherwise.
		{name: "LessThanF", argLength: 1, reg: readflags},        // bool, true flags encode floating-point x<y false otherwise.
		{name: "LessEqualF", argLength: 1, reg: readflags},       // bool, true flags encode floating-point x<=y false otherwise.
		{name: "GreaterThanF", argLength: 1, reg: readflags},     // bool, true flags encode floating-point x>y false otherwise.
		{name: "GreaterEqualF", argLength: 1, reg: readflags},    // bool, true flags encode floating-point x>=y false otherwise.
		{name: "NotLessThanF", argLength: 1, reg: readflags},     // bool, true flags encode floating-point x>=y || x is unordered with y, false otherwise.
		{name: "NotLessEqualF", argLength: 1, reg: readflags},    // bool, true flags encode floating-point x>y || x is unordered with y, false otherwise.
		{name: "NotGreaterThanF", argLength: 1, reg: readflags},  // bool, true flags encode floating-point x<=y || x is unordered with y, false otherwise.
		{name: "NotGreaterEqualF", argLength: 1, reg: readflags}, // bool, true flags encode floating-point x<y || x is unordered with y, false otherwise.
		{name: "LessThanNoov", argLength: 1, reg: readflags},     // bool, true flags encode signed x<y but without honoring overflow, false otherwise.
		{name: "GreaterEqualNoov", argLength: 1, reg: readflags}, // bool, true flags encode signed x>=y but without honoring overflow, false otherwise.

		// medium zeroing
		// arg0 = address of memory to zero
		// arg1 = mem
		// auxint = # of bytes to zero
		// returns mem
		{
			name:      "LoweredZero",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs: []regMask{gp},
			},
			faultOnNilArg0: true,
		},

		// large zeroing
		// arg0 = address of memory to zero
		// arg1 = mem
		// auxint = # of bytes to zero
		// returns mem
		{
			name:      "LoweredZeroLoop",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs:       []regMask{gp},
				clobbersArg0: true,
			},
			faultOnNilArg0: true,
			needIntTemp:    true,
		},

		// medium copying
		// arg0 = address of dst memory
		// arg1 = address of src memory
		// arg2 = mem
		// auxint = # of bytes to copy
		// returns mem
		{
			name:      "LoweredMove",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{gp &^ r24to25, gp &^ r24to25},
				clobbers: r24to25, // TODO: figure out needIntTemp x2
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// large copying
		// arg0 = address of dst memory
		// arg1 = address of src memory
		// arg2 = mem
		// auxint = # of bytes to copy
		// returns mem
		{
			name:      "LoweredMoveLoop",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:       []regMask{gp &^ r23to25, gp &^ r23to25},
				clobbers:     r23to25, // TODO: figure out needIntTemp x3
				clobbersArg0: true,
				clobbersArg1: true,
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of R26 (arm64.REGCTXT, the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("R26")}}, zeroWidth: true},

		// LoweredGetCallerSP returns the SP of the caller of the current function. arg0=mem
		{name: "LoweredGetCallerSP", argLength: 1, reg: gp01, rematerializeable: true},

		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" sys.GetCallerPC,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},

		// Constant flag value.
		// Note: there's an "unordered" outcome for floating-point
		// comparisons, but we don't use such a beast yet.
		// This op is for temporary use by rewrite rules. It
		// cannot appear in the generated assembly.
		{name: "FlagConstant", aux: "FlagConstant"},

		// (InvertFlags (CMP a b)) == (CMP b a)
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags", argLength: 1}, // reverse direction of arg0

		// atomic loads.
		// load from arg0. arg1=mem. auxint must be zero.
		// returns <value,memory> so they can be properly ordered with other loads.
		{name: "LDAR", argLength: 2, reg: gpload, asm: "LDAR", faultOnNilArg0: true},
		{name: "LDARB", argLength: 2, reg: gpload, asm: "LDARB", faultOnNilArg0: true},
		{name: "LDARW", argLength: 2, reg: gpload, asm: "LDARW", faultOnNilArg0: true},

		// atomic stores.
		// store arg1 to arg0. arg2=mem. returns memory. auxint must be zero.
		{name: "STLRB", argLength: 3, reg: gpstore, asm: "STLRB", faultOnNilArg0: true, hasSideEffects: true},
		{name: "STLR", argLength: 3, reg: gpstore, asm: "STLR", faultOnNilArg0: true, hasSideEffects: true},
		{name: "STLRW", argLength: 3, reg: gpstore, asm: "STLRW", faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange.
		// store arg1 to arg0. arg2=mem. returns <old content of *arg0, memory>. auxint must be zero.
		// LDAXR	(Rarg0), Rout
		// STLXR	Rarg1, (Rarg0), Rtmp
		// CBNZ		Rtmp, -2(PC)
		{name: "LoweredAtomicExchange64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicExchange32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicExchange8", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic exchange variant.
		// store arg1 to arg0. arg2=mem. returns <old content of *arg0, memory>. auxint must be zero.
		// SWPALD	Rarg1, (Rarg0), Rout
		{name: "LoweredAtomicExchange64Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicExchange32Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicExchange8Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic add.
		// *arg0 += arg1. arg2=mem. returns <new content of *arg0, memory>. auxint must be zero.
		// LDAXR	(Rarg0), Rout
		// ADD		Rarg1, Rout
		// STLXR	Rout, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		{name: "LoweredAtomicAdd64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicAdd32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic add variant.
		// *arg0 += arg1. arg2=mem. returns <new content of *arg0, memory>. auxint must be zero.
		// LDADDAL	(Rarg0), Rarg1, Rout
		// ADD		Rarg1, Rout
		{name: "LoweredAtomicAdd64Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAdd32Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic compare and swap.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory. auxint must be zero.
		// if *arg0 == arg1 {
		//   *arg0 = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// LDAXR	(Rarg0), Rtmp
		// CMP		Rarg1, Rtmp
		// BNE		3(PC)
		// STLXR	Rarg2, (Rarg0), Rtmp
		// CBNZ		Rtmp, -4(PC)
		// CSET		EQ, Rout
		{name: "LoweredAtomicCas64", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicCas32", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic compare and swap variant.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory. auxint must be zero.
		// if *arg0 == arg1 {
		//   *arg0 = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// MOV  	Rarg1, Rtmp
		// CASAL	Rtmp, (Rarg0), Rarg2
		// CMP  	Rarg1, Rtmp
		// CSET 	EQ, Rout
		{name: "LoweredAtomicCas64Variant", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicCas32Variant", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic and/or.
		// *arg0 &= (|=) arg1. arg2=mem. returns <old content of *arg0, memory>. auxint must be zero.
		// LDAXR	(Rarg0), Rout
		// AND/OR	Rarg1, Rout, tempReg
		// STLXR	tempReg, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		{name: "LoweredAtomicAnd8", argLength: 3, reg: gpxchg, resultNotInArgs: true, asm: "AND", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicOr8", argLength: 3, reg: gpxchg, resultNotInArgs: true, asm: "ORR", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicAnd64", argLength: 3, reg: gpxchg, resultNotInArgs: true, asm: "AND", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicOr64", argLength: 3, reg: gpxchg, resultNotInArgs: true, asm: "ORR", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicAnd32", argLength: 3, reg: gpxchg, resultNotInArgs: true, asm: "AND", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true, needIntTemp: true},
		{name: "LoweredAtomicOr32", argLength: 3, reg: gpxchg, resultNotInArgs: true, asm: "ORR", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true, needIntTemp: true},

		// atomic and/or variant.
		// *arg0 &= (|=) arg1. arg2=mem. returns <old content of *arg0, memory>. auxint must be zero.
		//   AND:
		// MNV       Rarg1, Rtemp
		// LDANDALB  Rtemp, (Rarg0), Rout
		//   OR:
		// LDORALB  Rarg1, (Rarg0), Rout
		{name: "LoweredAtomicAnd8Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicOr8Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAnd64Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicOr64Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAnd32Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicOr32Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// LoweredWB invokes runtime.gcWriteBarrier. arg0=mem, auxint=# of buffer entries needed
		// It saves all GP registers if necessary,
		// but clobbers R30 (LR) because it's a call.
		// R16 and R17 may be clobbered by linker trampoline.
		// Returns a pointer to a write barrier buffer in R25.
		{name: "LoweredWB", argLength: 1, reg: regInfo{clobbers: (callerSave &^ gpg) | buildReg("R16 R17 R30"), outputs: []regMask{buildReg("R25")}}, clobberFlags: true, aux: "Int64"},

		// LoweredPanicBoundsRR takes x and y, two values that caused a bounds check to fail.
		// the RC and CR versions are used when one of the arguments is a constant. CC is used
		// when both are constant (normally both 0, as prove derives the fact that a [0] bounds
		// failure means the length must have also been 0).
		// AuxInt contains a report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsRR", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{first16, first16}}, typ: "Mem", call: true}, // arg0=x, arg1=y, arg2=mem, returns memory.
		{name: "LoweredPanicBoundsRC", argLength: 2, aux: "PanicBoundsC", reg: regInfo{inputs: []regMask{first16}}, typ: "Mem", call: true},   // arg0=x, arg1=mem, returns memory.
		{name: "LoweredPanicBoundsCR", argLength: 2, aux: "PanicBoundsC", reg: regInfo{inputs: []regMask{first16}}, typ: "Mem", call: true},   // arg0=y, arg1=mem, returns memory.
		{name: "LoweredPanicBoundsCC", argLength: 1, aux: "PanicBoundsCC", reg: regInfo{}, typ: "Mem", call: true},                            // arg0=mem, returns memory.

		// Prefetch instruction
		// Do prefetch arg0 address with option aux. arg0=addr, arg1=memory, aux=option.
		{name: "PRFM", argLength: 2, aux: "Int64", reg: prefreg, asm: "PRFM", hasSideEffects: true},

		// Publication barrier
		{name: "DMB", argLength: 1, aux: "Int64", asm: "DMB", hasSideEffects: true}, // Do data barrier. arg0=memory, aux=option.
		{name: "ZERO", zeroWidth: true, fixedReg: true},                             // reads-as-zero register
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
		{name: "Z", controls: 1},                  // Control == 0 (take a register instead of flags)
		{name: "NZ", controls: 1},                 // Control != 0
		{name: "ZW", controls: 1},                 // Control == 0, 32-bit
		{name: "NZW", controls: 1},                // Control != 0, 32-bit
		{name: "TBZ", controls: 1, aux: "Int64"},  // Control & (1 << AuxInt) == 0
		{name: "TBNZ", controls: 1, aux: "Int64"}, // Control & (1 << AuxInt) != 0
		{name: "FLT", controls: 1},
		{name: "FLE", controls: 1},
		{name: "FGT", controls: 1},
		{name: "FGE", controls: 1},
		{name: "LTnoov", controls: 1}, // 'LT' but without honoring overflow
		{name: "LEnoov", controls: 1}, // 'LE' but without honoring overflow
		{name: "GTnoov", controls: 1}, // 'GT' but without honoring overflow
		{name: "GEnoov", controls: 1}, // 'GE' but without honoring overflow

		// JUMPTABLE implements jump tables.
		// Aux is the symbol (an *obj.LSym) for the jump table.
		// control[0] is the index into the jump table.
		// control[1] is the address of the jump table (the address of the symbol stored in Aux).
		{name: "JUMPTABLE", controls: 2, aux: "Sym"},
	}

	archs = append(archs, arch{
		name:               "ARM64",
		pkg:                "cmd/internal/obj/arm64",
		genfile:            "../../arm64/ssa.go",
		ops:                ops,
		blocks:             blocks,
		regnames:           regNamesARM64,
		ParamIntRegNames:   "R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15",
		ParamFloatRegNames: "F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15",
		gpregmask:          gp,
		fpregmask:          fp,
		framepointerreg:    -1, // not used
		linkreg:            int8(num["R30"]),
	})
}
