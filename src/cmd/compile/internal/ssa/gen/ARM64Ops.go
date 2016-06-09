// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import "strings"

// Notes:
//  - Integer types live in the low portion of registers. Upper portions are junk.
//  - Boolean types use the low-order byte of a register. 0=false, 1=true.
//    Upper bytes are junk.
//  - *const instructions may use a constant larger than the instruction can encode.
//    In this case the assembler expands to multiple instructions and uses tmp
//    register (R27).

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
	"R18", // platform register, not used
	"R19",
	"R20",
	"R21",
	"R22",
	"R23",
	"R24",
	"R25",
	"R26",
	// R27 = REGTMP not used in regalloc
	"g",   // aka R28
	"R29", // frame pointer, not used
	"R30", // aka REGLINK
	"SP",  // aka R31

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
	)
	// Common regInfo
	var (
		gp01      = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11      = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp11sp    = regInfo{inputs: []regMask{gpspg}, outputs: []regMask{gp}}
		gp1flags  = regInfo{inputs: []regMask{gpg}}
		gp1flags1 = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp21      = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp}}
		gp2flags  = regInfo{inputs: []regMask{gpg, gpg}}
		gp2flags1 = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp}}
		gpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}
		gpstore   = regInfo{inputs: []regMask{gpspsbg, gpg}}
		gpstore0  = regInfo{inputs: []regMask{gpspsbg}}
		gpxchg    = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{gp}}
		gpcas     = regInfo{inputs: []regMask{gpspsbg, gpg, gpg}, outputs: []regMask{gp}}
		fp01      = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11      = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fpgp      = regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}}
		gpfp      = regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}}
		fp21      = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp2flags  = regInfo{inputs: []regMask{fp, fp}}
		fpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{fp}}
		fpstore   = regInfo{inputs: []regMask{gpspsbg, fp}}
		readflags = regInfo{inputs: nil, outputs: []regMask{gp}}
	)
	ops := []opData{
		// binary ops
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},     // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11sp, asm: "ADD", aux: "Int64"},   // arg0 + auxInt
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB"},                        // arg0 - arg1
		{name: "SUBconst", argLength: 1, reg: gp11, asm: "SUB", aux: "Int64"},     // arg0 - auxInt
		{name: "MUL", argLength: 2, reg: gp21, asm: "MUL", commutative: true},     // arg0 * arg1
		{name: "MULW", argLength: 2, reg: gp21, asm: "MULW", commutative: true},   // arg0 * arg1, 32-bit
		{name: "MULH", argLength: 2, reg: gp21, asm: "SMULH", commutative: true},  // (arg0 * arg1) >> 64, signed
		{name: "UMULH", argLength: 2, reg: gp21, asm: "UMULH", commutative: true}, // (arg0 * arg1) >> 64, unsigned
		{name: "MULL", argLength: 2, reg: gp21, asm: "SMULL", commutative: true},  // arg0 * arg1, signed, 32-bit mult results in 64-bit
		{name: "UMULL", argLength: 2, reg: gp21, asm: "UMULL", commutative: true}, // arg0 * arg1, unsigned, 32-bit mult results in 64-bit
		{name: "DIV", argLength: 2, reg: gp21, asm: "SDIV"},                       // arg0 / arg1, signed
		{name: "UDIV", argLength: 2, reg: gp21, asm: "UDIV"},                      // arg0 / arg1, unsighed
		{name: "DIVW", argLength: 2, reg: gp21, asm: "SDIVW"},                     // arg0 / arg1, signed, 32 bit
		{name: "UDIVW", argLength: 2, reg: gp21, asm: "UDIVW"},                    // arg0 / arg1, unsighed, 32 bit
		{name: "MOD", argLength: 2, reg: gp21, asm: "REM"},                        // arg0 % arg1, signed
		{name: "UMOD", argLength: 2, reg: gp21, asm: "UREM"},                      // arg0 % arg1, unsigned
		{name: "MODW", argLength: 2, reg: gp21, asm: "REMW"},                      // arg0 % arg1, signed, 32 bit
		{name: "UMODW", argLength: 2, reg: gp21, asm: "UREMW"},                    // arg0 % arg1, unsigned, 32 bit

		{name: "FADDS", argLength: 2, reg: fp21, asm: "FADDS", commutative: true}, // arg0 + arg1
		{name: "FADDD", argLength: 2, reg: fp21, asm: "FADDD", commutative: true}, // arg0 + arg1
		{name: "FSUBS", argLength: 2, reg: fp21, asm: "FSUBS"},                    // arg0 - arg1
		{name: "FSUBD", argLength: 2, reg: fp21, asm: "FSUBD"},                    // arg0 - arg1
		{name: "FMULS", argLength: 2, reg: fp21, asm: "FMULS", commutative: true}, // arg0 * arg1
		{name: "FMULD", argLength: 2, reg: fp21, asm: "FMULD", commutative: true}, // arg0 * arg1
		{name: "FDIVS", argLength: 2, reg: fp21, asm: "FDIVS"},                    // arg0 / arg1
		{name: "FDIVD", argLength: 2, reg: fp21, asm: "FDIVD"},                    // arg0 / arg1

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true}, // arg0 & arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int64"}, // arg0 & auxInt
		{name: "OR", argLength: 2, reg: gp21, asm: "ORR", commutative: true},  // arg0 | arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "ORR", aux: "Int64"},  // arg0 | auxInt
		{name: "XOR", argLength: 2, reg: gp21, asm: "EOR", commutative: true}, // arg0 ^ arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "EOR", aux: "Int64"}, // arg0 ^ auxInt
		{name: "BIC", argLength: 2, reg: gp21, asm: "BIC"},                    // arg0 &^ arg1
		{name: "BICconst", argLength: 1, reg: gp11, asm: "BIC", aux: "Int64"}, // arg0 &^ auxInt

		// unary ops
		{name: "MVN", argLength: 1, reg: gp11, asm: "MVN"},       // ^arg0
		{name: "NEG", argLength: 1, reg: gp11, asm: "NEG"},       // -arg0
		{name: "FNEGS", argLength: 1, reg: fp11, asm: "FNEGS"},   // -arg0, float32
		{name: "FNEGD", argLength: 1, reg: fp11, asm: "FNEGD"},   // -arg0, float64
		{name: "FSQRTD", argLength: 1, reg: fp11, asm: "FSQRTD"}, // sqrt(arg0), float64
		{name: "REV", argLength: 1, reg: gp11, asm: "REV"},       // byte reverse, 64-bit
		{name: "REVW", argLength: 1, reg: gp11, asm: "REVW"},     // byte reverse, 32-bit
		{name: "REV16W", argLength: 1, reg: gp11, asm: "REV16W"}, // byte reverse in each 16-bit halfword, 32-bit
		{name: "RBIT", argLength: 1, reg: gp11, asm: "RBIT"},     // bit reverse, 64-bit
		{name: "RBITW", argLength: 1, reg: gp11, asm: "RBITW"},   // bit reverse, 32-bit
		{name: "CLZ", argLength: 1, reg: gp11, asm: "CLZ"},       // count leading zero, 64-bit
		{name: "CLZW", argLength: 1, reg: gp11, asm: "CLZW"},     // count leading zero, 32-bit

		// shifts
		{name: "SLL", argLength: 2, reg: gp21, asm: "LSL"},                      // arg0 << arg1, shift amount is mod 64
		{name: "SLLconst", argLength: 1, reg: gp11, asm: "LSL", aux: "Int64"},   // arg0 << auxInt
		{name: "SRL", argLength: 2, reg: gp21, asm: "LSR"},                      // arg0 >> arg1, unsigned, shift amount is mod 64
		{name: "SRLconst", argLength: 1, reg: gp11, asm: "LSR", aux: "Int64"},   // arg0 >> auxInt, unsigned
		{name: "SRA", argLength: 2, reg: gp21, asm: "ASR"},                      // arg0 >> arg1, signed, shift amount is mod 64
		{name: "SRAconst", argLength: 1, reg: gp11, asm: "ASR", aux: "Int64"},   // arg0 >> auxInt, signed
		{name: "RORconst", argLength: 1, reg: gp11, asm: "ROR", aux: "Int64"},   // arg0 right rotate by auxInt bits
		{name: "RORWconst", argLength: 1, reg: gp11, asm: "RORW", aux: "Int64"}, // uint32(arg0) right rotate by auxInt bits

		// comparisons
		{name: "CMP", argLength: 2, reg: gp2flags, asm: "CMP", typ: "Flags"},                      // arg0 compare to arg1
		{name: "CMPconst", argLength: 1, reg: gp1flags, asm: "CMP", aux: "Int64", typ: "Flags"},   // arg0 compare to auxInt
		{name: "CMPW", argLength: 2, reg: gp2flags, asm: "CMPW", typ: "Flags"},                    // arg0 compare to arg1, 32 bit
		{name: "CMPWconst", argLength: 1, reg: gp1flags, asm: "CMPW", aux: "Int32", typ: "Flags"}, // arg0 compare to auxInt, 32 bit
		{name: "CMN", argLength: 2, reg: gp2flags, asm: "CMN", typ: "Flags"},                      // arg0 compare to -arg1
		{name: "CMNconst", argLength: 1, reg: gp1flags, asm: "CMN", aux: "Int64", typ: "Flags"},   // arg0 compare to -auxInt
		{name: "CMNW", argLength: 2, reg: gp2flags, asm: "CMNW", typ: "Flags"},                    // arg0 compare to -arg1, 32 bit
		{name: "CMNWconst", argLength: 1, reg: gp1flags, asm: "CMNW", aux: "Int32", typ: "Flags"}, // arg0 compare to -auxInt, 32 bit
		{name: "FCMPS", argLength: 2, reg: fp2flags, asm: "FCMPS", typ: "Flags"},                  // arg0 compare to arg1, float32
		{name: "FCMPD", argLength: 2, reg: fp2flags, asm: "FCMPD", typ: "Flags"},                  // arg0 compare to arg1, float64

		// shifted ops
		{name: "ADDshiftLL", argLength: 2, reg: gp21, asm: "ADD", aux: "Int64"},                   // arg0 + arg1<<auxInt
		{name: "ADDshiftRL", argLength: 2, reg: gp21, asm: "ADD", aux: "Int64"},                   // arg0 + arg1>>auxInt, unsigned shift
		{name: "ADDshiftRA", argLength: 2, reg: gp21, asm: "ADD", aux: "Int64"},                   // arg0 + arg1>>auxInt, signed shift
		{name: "SUBshiftLL", argLength: 2, reg: gp21, asm: "SUB", aux: "Int64"},                   // arg0 - arg1<<auxInt
		{name: "SUBshiftRL", argLength: 2, reg: gp21, asm: "SUB", aux: "Int64"},                   // arg0 - arg1>>auxInt, unsigned shift
		{name: "SUBshiftRA", argLength: 2, reg: gp21, asm: "SUB", aux: "Int64"},                   // arg0 - arg1>>auxInt, signed shift
		{name: "ANDshiftLL", argLength: 2, reg: gp21, asm: "AND", aux: "Int64"},                   // arg0 & (arg1<<auxInt)
		{name: "ANDshiftRL", argLength: 2, reg: gp21, asm: "AND", aux: "Int64"},                   // arg0 & (arg1>>auxInt), unsigned shift
		{name: "ANDshiftRA", argLength: 2, reg: gp21, asm: "AND", aux: "Int64"},                   // arg0 & (arg1>>auxInt), signed shift
		{name: "ORshiftLL", argLength: 2, reg: gp21, asm: "ORR", aux: "Int64"},                    // arg0 | arg1<<auxInt
		{name: "ORshiftRL", argLength: 2, reg: gp21, asm: "ORR", aux: "Int64"},                    // arg0 | arg1>>auxInt, unsigned shift
		{name: "ORshiftRA", argLength: 2, reg: gp21, asm: "ORR", aux: "Int64"},                    // arg0 | arg1>>auxInt, signed shift
		{name: "XORshiftLL", argLength: 2, reg: gp21, asm: "EOR", aux: "Int64"},                   // arg0 ^ arg1<<auxInt
		{name: "XORshiftRL", argLength: 2, reg: gp21, asm: "EOR", aux: "Int64"},                   // arg0 ^ arg1>>auxInt, unsigned shift
		{name: "XORshiftRA", argLength: 2, reg: gp21, asm: "EOR", aux: "Int64"},                   // arg0 ^ arg1>>auxInt, signed shift
		{name: "BICshiftLL", argLength: 2, reg: gp21, asm: "BIC", aux: "Int64"},                   // arg0 &^ (arg1<<auxInt)
		{name: "BICshiftRL", argLength: 2, reg: gp21, asm: "BIC", aux: "Int64"},                   // arg0 &^ (arg1>>auxInt), unsigned shift
		{name: "BICshiftRA", argLength: 2, reg: gp21, asm: "BIC", aux: "Int64"},                   // arg0 &^ (arg1>>auxInt), signed shift
		{name: "CMPshiftLL", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int64", typ: "Flags"}, // arg0 compare to arg1<<auxInt
		{name: "CMPshiftRL", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int64", typ: "Flags"}, // arg0 compare to arg1>>auxInt, unsigned shift
		{name: "CMPshiftRA", argLength: 2, reg: gp2flags, asm: "CMP", aux: "Int64", typ: "Flags"}, // arg0 compare to arg1>>auxInt, signed shift

		// moves
		{name: "MOVDconst", argLength: 0, reg: gp01, aux: "Int64", asm: "MOVD", typ: "UInt64", rematerializeable: true},      // 32 low bits of auxint
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

		{name: "MOVBstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 1 byte of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVHstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 2 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVWstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVDstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},   // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "FMOVSstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "FMOVS", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "FMOVDstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "FMOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.

		{name: "MOVBstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 1 byte of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVDstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of zero to arg0 + auxInt + aux.  ar12=mem.

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

		// conditional instructions
		{name: "CSELULT", argLength: 3, reg: gp2flags1, asm: "CSEL"},  // returns arg0 if flags indicates unsigned LT, arg1 otherwise, arg2=flags
		{name: "CSELULT0", argLength: 2, reg: gp1flags1, asm: "CSEL"}, // returns arg0 if flags indicates unsigned LT, 0 otherwise, arg1=flags

		// function calls
		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "SymOff", clobberFlags: true, call: true, symEffect: "None"},                           // call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{inputs: []regMask{gpsp, buildReg("R26"), 0}, clobbers: callerSave}, aux: "Int64", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "Int64", clobberFlags: true, call: true},                         // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

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

		// duffzero
		// arg0 = address of memory to zero
		// arg1 = mem
		// auxint = offset into duffzero code to start executing
		// returns mem
		// R16 aka arm64.REGRT1 changed as side effect
		{
			name:      "DUFFZERO",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs:   []regMask{gp},
				clobbers: buildReg("R16 R30"),
			},
			faultOnNilArg0: true,
		},

		// large zeroing
		// arg0 = address of memory to zero (in R16 aka arm64.REGRT1, changed as side effect)
		// arg1 = address of the last element to zero
		// arg2 = mem
		// returns mem
		//	MOVD.P	ZR, 8(R16)
		//	CMP	Rarg1, R16
		//	BLE	-2(PC)
		// Note: the-end-of-the-memory may be not a valid pointer. it's a problem if it is spilled.
		// the-end-of-the-memory - 8 is with the area to zero, ok to spill.
		{
			name:      "LoweredZero",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R16"), gp},
				clobbers: buildReg("R16"),
			},
			clobberFlags:   true,
			faultOnNilArg0: true,
		},

		// duffcopy
		// arg0 = address of dst memory (in R17 aka arm64.REGRT2, changed as side effect)
		// arg1 = address of src memory (in R16 aka arm64.REGRT1, changed as side effect)
		// arg2 = mem
		// auxint = offset into duffcopy code to start executing
		// returns mem
		// R16, R17 changed as side effect
		{
			name:      "DUFFCOPY",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R17"), buildReg("R16")},
				clobbers: buildReg("R16 R17 R30"),
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// large move
		// arg0 = address of dst memory (in R17 aka arm64.REGRT2, changed as side effect)
		// arg1 = address of src memory (in R16 aka arm64.REGRT1, changed as side effect)
		// arg2 = address of the last element of src
		// arg3 = mem
		// returns mem
		//	MOVD.P	8(R16), Rtmp
		//	MOVD.P	Rtmp, 8(R17)
		//	CMP	Rarg2, R16
		//	BLE	-3(PC)
		// Note: the-end-of-src may be not a valid pointer. it's a problem if it is spilled.
		// the-end-of-src - 8 is within the area to copy, ok to spill.
		{
			name:      "LoweredMove",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("R17"), buildReg("R16"), gp},
				clobbers: buildReg("R16 R17"),
			},
			clobberFlags:   true,
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of R26 (arm64.REGCTXT, the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("R26")}}},

		// MOVDconvert converts between pointers and integers.
		// We have a special op for this so as to not confuse GC
		// (particularly stack maps).  It takes a memory arg so it
		// gets correctly ordered with respect to GC safepoints.
		// arg0=ptr/int arg1=mem, output=int/ptr
		{name: "MOVDconvert", argLength: 2, reg: gp11, asm: "MOVD"},

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

		// (InvertFlags (CMP a b)) == (CMP b a)
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags", argLength: 1}, // reverse direction of arg0

		// atomic loads.
		// load from arg0. arg1=mem. auxint must be zero.
		// returns <value,memory> so they can be properly ordered with other loads.
		{name: "LDAR", argLength: 2, reg: gpload, asm: "LDAR", faultOnNilArg0: true},
		{name: "LDARW", argLength: 2, reg: gpload, asm: "LDARW", faultOnNilArg0: true},

		// atomic stores.
		// store arg1 to arg0. arg2=mem. returns memory. auxint must be zero.
		{name: "STLR", argLength: 3, reg: gpstore, asm: "STLR", faultOnNilArg0: true, hasSideEffects: true},
		{name: "STLRW", argLength: 3, reg: gpstore, asm: "STLRW", faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange.
		// store arg1 to arg0. arg2=mem. returns <old content of *arg0, memory>. auxint must be zero.
		// LDAXR	(Rarg0), Rout
		// STLXR	Rarg1, (Rarg0), Rtmp
		// CBNZ		Rtmp, -2(PC)
		{name: "LoweredAtomicExchange64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicExchange32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic add.
		// *arg0 += arg1. arg2=mem. returns <new content of *arg0, memory>. auxint must be zero.
		// LDAXR	(Rarg0), Rout
		// ADD		Rarg1, Rout
		// STLXR	Rout, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		{name: "LoweredAtomicAdd64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAdd32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

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
		{name: "LoweredAtomicCas64", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicCas32", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic and/or.
		// *arg0 &= (|=) arg1. arg2=mem. returns memory. auxint must be zero.
		// LDAXRB	(Rarg0), Rtmp
		// AND/OR	Rarg1, Rtmp
		// STLXRB	Rtmp, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		{name: "LoweredAtomicAnd8", argLength: 3, reg: gpstore, asm: "AND", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicOr8", argLength: 3, reg: gpstore, asm: "ORR", faultOnNilArg0: true, hasSideEffects: true},
	}

	blocks := []blockData{
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
		{name: "Z"},   // Control == 0 (take a register instead of flags)
		{name: "NZ"},  // Control != 0
		{name: "ZW"},  // Control == 0, 32-bit
		{name: "NZW"}, // Control != 0, 32-bit
	}

	archs = append(archs, arch{
		name:            "ARM64",
		pkg:             "cmd/internal/obj/arm64",
		genfile:         "../../arm64/ssa.go",
		ops:             ops,
		blocks:          blocks,
		regnames:        regNamesARM64,
		gpregmask:       gp,
		fpregmask:       fp,
		framepointerreg: -1, // not used
		linkreg:         int8(num["R30"]),
	})
}
