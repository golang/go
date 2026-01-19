// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// Notes:
//  - Integer types live in the low portion of registers. Upper portions are junk.
//  - Boolean types use the low-order byte of a register. 0=false, 1=true.
//    Upper bytes are junk.
//  - Unused portions of AuxInt are filled by sign-extending the used portion.
//  - *const instructions may use a constant larger than the instruction can encode.
//    In this case the assembler expands to multiple instructions and uses tmp
//    register (R23).

// Suffixes encode the bit width of various instructions.
// W (word)      = 32 bit
// H (half word) = 16 bit
// HU            = 16 bit unsigned
// B (byte)      = 8 bit
// BU            = 8 bit unsigned
// F (float)     = 32 bit float
// D (double)    = 64 bit float

// Note: registers not used in regalloc are not included in this list,
// so that regmask stays within int64
// Be careful when hand coding regmasks.
var regNamesMIPS = []string{
	"R0", // constant 0
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
	"R18",
	"R19",
	"R20",
	"R21",
	"R22",
	//REGTMP
	"R24",
	"R25",
	// R26 reserved by kernel
	// R27 reserved by kernel
	"R28",
	"SP",  // aka R29
	"g",   // aka R30
	"R31", // REGLINK

	// odd FP registers contain high parts of 64-bit FP values
	"F0",
	"F2",
	"F4",
	"F6",
	"F8",
	"F10",
	"F12",
	"F14",
	"F16",
	"F18",
	"F20",
	"F22",
	"F24",
	"F26",
	"F28",
	"F30",

	"HI", // high bits of multiplication
	"LO", // low bits of multiplication

	// If you add registers, update asyncPreempt in runtime.

	// pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesMIPS) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesMIPS {
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
		gp         = buildReg("R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31")
		gpg        = gp | buildReg("g")
		gpsp       = gp | buildReg("SP")
		gpspg      = gpg | buildReg("SP")
		gpspsbg    = gpspg | buildReg("SB")
		fp         = buildReg("F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30")
		lo         = buildReg("LO")
		hi         = buildReg("HI")
		callerSave = gp | fp | lo | hi | buildReg("g") // runtime.setg (and anything calling it) may clobber g
		first16    = buildReg("R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16")
		first4     = buildReg("R1 R2 R3 R4")
	)
	// Common regInfo
	var (
		gp01      = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11      = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp11sp    = regInfo{inputs: []regMask{gpspg}, outputs: []regMask{gp}}
		gp21      = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp}}
		gp31      = regInfo{inputs: []regMask{gp, gp, gp}, outputs: []regMask{gp}}
		gp2hilo   = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{hi, lo}}
		gpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}
		gpstore   = regInfo{inputs: []regMask{gpspsbg, gpg}}
		gpxchg    = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{gp}}
		gpcas     = regInfo{inputs: []regMask{gpspsbg, gpg, gpg}, outputs: []regMask{gp}}
		gpstore0  = regInfo{inputs: []regMask{gpspsbg}}
		fpgp      = regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}}
		gpfp      = regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}}
		fp01      = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11      = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fp21      = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp2flags  = regInfo{inputs: []regMask{fp, fp}}
		fpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{fp}}
		fpstore   = regInfo{inputs: []regMask{gpspsbg, fp}}
		readflags = regInfo{inputs: nil, outputs: []regMask{gp}}
	)
	ops := []opData{
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADDU", commutative: true},                                                                           // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11sp, asm: "ADDU", aux: "Int32"},                                                                         // arg0 + auxInt
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUBU"},                                                                                              // arg0 - arg1
		{name: "SUBconst", argLength: 1, reg: gp11, asm: "SUBU", aux: "Int32"},                                                                           // arg0 - auxInt
		{name: "MUL", argLength: 2, reg: regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp}, clobbers: hi | lo}, asm: "MUL", commutative: true}, // arg0 * arg1
		{name: "MULT", argLength: 2, reg: gp2hilo, asm: "MUL", commutative: true, typ: "(Int32,Int32)"},                                                  // arg0 * arg1, signed, results hi,lo
		{name: "MULTU", argLength: 2, reg: gp2hilo, asm: "MULU", commutative: true, typ: "(UInt32,UInt32)"},                                              // arg0 * arg1, unsigned, results hi,lo
		{name: "DIV", argLength: 2, reg: gp2hilo, asm: "DIV", typ: "(Int32,Int32)"},                                                                      // arg0 / arg1, signed, results hi=arg0%arg1,lo=arg0/arg1
		{name: "DIVU", argLength: 2, reg: gp2hilo, asm: "DIVU", typ: "(UInt32,UInt32)"},                                                                  // arg0 / arg1, signed, results hi=arg0%arg1,lo=arg0/arg1

		{name: "ADDF", argLength: 2, reg: fp21, asm: "ADDF", commutative: true}, // arg0 + arg1
		{name: "ADDD", argLength: 2, reg: fp21, asm: "ADDD", commutative: true}, // arg0 + arg1
		{name: "SUBF", argLength: 2, reg: fp21, asm: "SUBF"},                    // arg0 - arg1
		{name: "SUBD", argLength: 2, reg: fp21, asm: "SUBD"},                    // arg0 - arg1
		{name: "MULF", argLength: 2, reg: fp21, asm: "MULF", commutative: true}, // arg0 * arg1
		{name: "MULD", argLength: 2, reg: fp21, asm: "MULD", commutative: true}, // arg0 * arg1
		{name: "DIVF", argLength: 2, reg: fp21, asm: "DIVF"},                    // arg0 / arg1
		{name: "DIVD", argLength: 2, reg: fp21, asm: "DIVD"},                    // arg0 / arg1

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true},                // arg0 & arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int32"},                // arg0 & auxInt
		{name: "OR", argLength: 2, reg: gp21, asm: "OR", commutative: true},                  // arg0 | arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "OR", aux: "Int32"},                  // arg0 | auxInt
		{name: "XOR", argLength: 2, reg: gp21, asm: "XOR", commutative: true, typ: "UInt32"}, // arg0 ^ arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "XOR", aux: "Int32", typ: "UInt32"}, // arg0 ^ auxInt
		{name: "NOR", argLength: 2, reg: gp21, asm: "NOR", commutative: true},                // ^(arg0 | arg1)
		{name: "NORconst", argLength: 1, reg: gp11, asm: "NOR", aux: "Int32"},                // ^(arg0 | auxInt)

		{name: "NEG", argLength: 1, reg: gp11},                 // -arg0
		{name: "NEGF", argLength: 1, reg: fp11, asm: "NEGF"},   // -arg0, float32
		{name: "NEGD", argLength: 1, reg: fp11, asm: "NEGD"},   // -arg0, float64
		{name: "ABSD", argLength: 1, reg: fp11, asm: "ABSD"},   // abs(arg0), float64
		{name: "SQRTD", argLength: 1, reg: fp11, asm: "SQRTD"}, // sqrt(arg0), float64
		{name: "SQRTF", argLength: 1, reg: fp11, asm: "SQRTF"}, // sqrt(arg0), float32

		// shifts
		{name: "SLL", argLength: 2, reg: gp21, asm: "SLL"},                    // arg0 << arg1, shift amount is mod 32
		{name: "SLLconst", argLength: 1, reg: gp11, asm: "SLL", aux: "Int32"}, // arg0 << auxInt, shift amount must be 0 through 31 inclusive
		{name: "SRL", argLength: 2, reg: gp21, asm: "SRL"},                    // arg0 >> arg1, unsigned, shift amount is mod 32
		{name: "SRLconst", argLength: 1, reg: gp11, asm: "SRL", aux: "Int32"}, // arg0 >> auxInt, shift amount must be 0 through 31 inclusive
		{name: "SRA", argLength: 2, reg: gp21, asm: "SRA"},                    // arg0 >> arg1, signed, shift amount is mod 32
		{name: "SRAconst", argLength: 1, reg: gp11, asm: "SRA", aux: "Int32"}, // arg0 >> auxInt, signed, shift amount must be 0 through 31 inclusive

		{name: "CLZ", argLength: 1, reg: gp11, asm: "CLZ"},

		// comparisons
		{name: "SGT", argLength: 2, reg: gp21, asm: "SGT", typ: "Bool"},                      // 1 if arg0 > arg1 (signed), 0 otherwise
		{name: "SGTconst", argLength: 1, reg: gp11, asm: "SGT", aux: "Int32", typ: "Bool"},   // 1 if auxInt > arg0 (signed), 0 otherwise
		{name: "SGTzero", argLength: 1, reg: gp11, asm: "SGT", typ: "Bool"},                  // 1 if arg0 > 0 (signed), 0 otherwise
		{name: "SGTU", argLength: 2, reg: gp21, asm: "SGTU", typ: "Bool"},                    // 1 if arg0 > arg1 (unsigned), 0 otherwise
		{name: "SGTUconst", argLength: 1, reg: gp11, asm: "SGTU", aux: "Int32", typ: "Bool"}, // 1 if auxInt > arg0 (unsigned), 0 otherwise
		{name: "SGTUzero", argLength: 1, reg: gp11, asm: "SGTU", typ: "Bool"},                // 1 if arg0 > 0 (unsigned), 0 otherwise

		{name: "CMPEQF", argLength: 2, reg: fp2flags, asm: "CMPEQF", typ: "Flags"}, // flags=true if arg0 = arg1, float32
		{name: "CMPEQD", argLength: 2, reg: fp2flags, asm: "CMPEQD", typ: "Flags"}, // flags=true if arg0 = arg1, float64
		{name: "CMPGEF", argLength: 2, reg: fp2flags, asm: "CMPGEF", typ: "Flags"}, // flags=true if arg0 >= arg1, float32
		{name: "CMPGED", argLength: 2, reg: fp2flags, asm: "CMPGED", typ: "Flags"}, // flags=true if arg0 >= arg1, float64
		{name: "CMPGTF", argLength: 2, reg: fp2flags, asm: "CMPGTF", typ: "Flags"}, // flags=true if arg0 > arg1, float32
		{name: "CMPGTD", argLength: 2, reg: fp2flags, asm: "CMPGTD", typ: "Flags"}, // flags=true if arg0 > arg1, float64

		// moves
		{name: "MOVWconst", argLength: 0, reg: gp01, aux: "Int32", asm: "MOVW", typ: "UInt32", rematerializeable: true},    // auxint
		{name: "MOVFconst", argLength: 0, reg: fp01, aux: "Float32", asm: "MOVF", typ: "Float32", rematerializeable: true}, // auxint as 64-bit float, convert to 32-bit float
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

		{name: "MOVBstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 1 byte of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of zero to arg0 + auxInt + aux.  arg1=mem.

		// moves (no conversion)
		{name: "MOVWfpgp", argLength: 1, reg: fpgp, asm: "MOVW"}, // move float32 to int32 (no conversion)
		{name: "MOVWgpfp", argLength: 1, reg: gpfp, asm: "MOVW"}, // move int32 to float32 (no conversion)

		// conversions
		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVB"},   // move from arg0, sign-extended from byte
		{name: "MOVBUreg", argLength: 1, reg: gp11, asm: "MOVBU"}, // move from arg0, unsign-extended from byte
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVH"},   // move from arg0, sign-extended from half
		{name: "MOVHUreg", argLength: 1, reg: gp11, asm: "MOVHU"}, // move from arg0, unsign-extended from half
		{name: "MOVWreg", argLength: 1, reg: gp11, asm: "MOVW"},   // move from arg0

		{name: "MOVWnop", argLength: 1, reg: regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}, resultInArg0: true}, // nop, return arg0 in same register

		// conditional move on zero (returns arg1 if arg2 is 0, otherwise arg0)
		// order of parameters is reversed so we can use resultInArg0 (OpCMOVZ result arg1 arg2-> CMOVZ arg2reg, arg1reg, resultReg)
		{name: "CMOVZ", argLength: 3, reg: gp31, asm: "CMOVZ", resultInArg0: true},
		{name: "CMOVZzero", argLength: 2, reg: regInfo{inputs: []regMask{gp, gpg}, outputs: []regMask{gp}}, asm: "CMOVZ", resultInArg0: true},

		{name: "MOVWF", argLength: 1, reg: fp11, asm: "MOVWF"},     // int32 -> float32
		{name: "MOVWD", argLength: 1, reg: fp11, asm: "MOVWD"},     // int32 -> float64
		{name: "TRUNCFW", argLength: 1, reg: fp11, asm: "TRUNCFW"}, // float32 -> int32
		{name: "TRUNCDW", argLength: 1, reg: fp11, asm: "TRUNCDW"}, // float64 -> int32
		{name: "MOVFD", argLength: 1, reg: fp11, asm: "MOVFD"},     // float32 -> float64
		{name: "MOVDF", argLength: 1, reg: fp11, asm: "MOVDF"},     // float64 -> float32

		// function calls
		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                               // call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                 //  tail call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{inputs: []regMask{gpsp, buildReg("R22"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                         // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		// atomic ops

		// load from arg0. arg1=mem.
		// returns <value,memory> so they can be properly ordered with other loads.
		// SYNC
		// MOV(B|W)	(Rarg0), Rout
		// SYNC
		{name: "LoweredAtomicLoad8", argLength: 2, reg: gpload, faultOnNilArg0: true},
		{name: "LoweredAtomicLoad32", argLength: 2, reg: gpload, faultOnNilArg0: true},

		// store arg1 to arg0. arg2=mem. returns memory.
		// SYNC
		// MOV(B|W)	Rarg1, (Rarg0)
		// SYNC
		{name: "LoweredAtomicStore8", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore32", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStorezero", argLength: 2, reg: gpstore0, faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange.
		// store arg1 to arg0. arg2=mem. returns <old content of *arg0, memory>.
		// SYNC
		// LL	(Rarg0), Rout
		// MOVW Rarg1, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		{name: "LoweredAtomicExchange", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic add.
		// *arg0 += arg1. arg2=mem. returns <new content of *arg0, memory>.
		// SYNC
		// LL	(Rarg0), Rout
		// ADDU Rarg1, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		// ADDU Rarg1, Rout
		{name: "LoweredAtomicAdd", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicAddconst", argLength: 2, reg: regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}, aux: "Int32", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic compare and swap.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory.
		// if *arg0 == arg1 {
		//   *arg0 = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// SYNC
		// MOVW $0, Rout
		// LL	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 4(PC)
		// MOVW Rarg2, Rout
		// SC	Rout, (Rarg0)
		// BEQ	Rout, -4(PC)
		// SYNC
		{name: "LoweredAtomicCas", argLength: 4, reg: gpcas, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic and/or.
		// *arg0 &= (|=) arg1. arg2=mem. returns memory.
		// SYNC
		// LL	(Rarg0), Rtmp
		// AND	Rarg1, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		{name: "LoweredAtomicAnd", argLength: 3, reg: gpstore, asm: "AND", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicOr", argLength: 3, reg: gpstore, asm: "OR", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// large or unaligned zeroing
		// arg0 = address of memory to zero (in R1, changed as side effect)
		// arg1 = address of the last element to zero
		// arg2 = mem
		// auxint = alignment
		// returns mem
		//	SUBU	$4, R1
		//	MOVW	R0, 4(R1)
		//	ADDU	$4, R1
		//	BNE	Rarg1, R1, -2(PC)
		{
			name:      "LoweredZero",
			aux:       "Int32",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R1"), gp},
				clobbers: buildReg("R1"),
			},
			faultOnNilArg0: true,
		},

		// large or unaligned move
		// arg0 = address of dst memory (in R2, changed as side effect)
		// arg1 = address of src memory (in R1, changed as side effect)
		// arg2 = address of the last element of src
		// arg3 = mem
		// auxint = alignment
		// returns mem
		//	SUBU	$4, R1
		//	MOVW	4(R1), Rtmp
		//	MOVW	Rtmp, (R2)
		//	ADDU	$4, R1
		//	ADDU	$4, R2
		//	BNE	Rarg2, R1, -4(PC)
		{
			name:      "LoweredMove",
			aux:       "Int32",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("R2"), buildReg("R1"), gp},
				clobbers: buildReg("R1 R2"),
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// pseudo-ops
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gpg}}, nilCheck: true, faultOnNilArg0: true}, // panic if arg0 is nil.  arg1=mem.

		{name: "FPFlagTrue", argLength: 1, reg: readflags},  // bool, true if FP flag is true
		{name: "FPFlagFalse", argLength: 1, reg: readflags}, // bool, true if FP flag is false

		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of R22 (mips.REGCTXT, the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("R22")}}, zeroWidth: true},

		// LoweredGetCallerSP returns the SP of the caller of the current function. arg0=mem.
		{name: "LoweredGetCallerSP", argLength: 1, reg: gp01, rematerializeable: true},

		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" sys.GetCallerPC,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},

		// LoweredWB invokes runtime.gcWriteBarrier. arg0=mem, auxint=# of buffer entries needed
		// It saves all GP registers if necessary,
		// but clobbers R31 (LR) because it's a call
		// and R23 (REGTMP).
		// Returns a pointer to a write barrier buffer in R25.
		{name: "LoweredWB", argLength: 1, reg: regInfo{clobbers: (callerSave &^ gpg) | buildReg("R31"), outputs: []regMask{buildReg("R25")}}, clobberFlags: true, aux: "Int64"},

		// Do data barrier. arg0=memorys
		{name: "LoweredPubBarrier", argLength: 1, asm: "SYNC", hasSideEffects: true},

		// LoweredPanicBoundsRR takes x and y, two values that caused a bounds check to fail.
		// the RC and CR versions are used when one of the arguments is a constant. CC is used
		// when both are constant (normally both 0, as prove derives the fact that a [0] bounds
		// failure means the length must have also been 0).
		// AuxInt contains a report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsRR", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{first16, first16}}, typ: "Mem", call: true}, // arg0=x, arg1=y, arg2=mem, returns memory.
		{name: "LoweredPanicBoundsRC", argLength: 2, aux: "PanicBoundsC", reg: regInfo{inputs: []regMask{first16}}, typ: "Mem", call: true},   // arg0=x, arg1=mem, returns memory.
		{name: "LoweredPanicBoundsCR", argLength: 2, aux: "PanicBoundsC", reg: regInfo{inputs: []regMask{first16}}, typ: "Mem", call: true},   // arg0=y, arg1=mem, returns memory.
		{name: "LoweredPanicBoundsCC", argLength: 1, aux: "PanicBoundsCC", reg: regInfo{}, typ: "Mem", call: true},                            // arg0=mem, returns memory.

		// Same as above, but the x value is 64 bits.
		{name: "LoweredPanicExtendRR", argLength: 4, aux: "Int64", reg: regInfo{inputs: []regMask{first4, first4, first16}}, typ: "Mem", call: true}, // arg0=x_hi, arg1=x_lo, arg2=y, arg3=mem, returns memory.
		{name: "LoweredPanicExtendRC", argLength: 3, aux: "PanicBoundsC", reg: regInfo{inputs: []regMask{first4, first4}}, typ: "Mem", call: true},   // arg0=x_hi, arg1=x_lo, arg2=mem, returns memory.
	}

	blocks := []blockData{
		{name: "EQ", controls: 1},
		{name: "NE", controls: 1},
		{name: "LTZ", controls: 1}, // < 0
		{name: "LEZ", controls: 1}, // <= 0
		{name: "GTZ", controls: 1}, // > 0
		{name: "GEZ", controls: 1}, // >= 0
		{name: "FPT", controls: 1}, // FP flag is true
		{name: "FPF", controls: 1}, // FP flag is false
	}

	archs = append(archs, arch{
		name:            "MIPS",
		pkg:             "cmd/internal/obj/mips",
		genfile:         "../../mips/ssa.go",
		ops:             ops,
		blocks:          blocks,
		regnames:        regNamesMIPS,
		gpregmask:       gp,
		fpregmask:       fp,
		specialregmask:  hi | lo,
		framepointerreg: -1, // not used
		linkreg:         int8(num["R31"]),
	})
}
