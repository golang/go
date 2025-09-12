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
//    register (R23).

// Suffixes encode the bit width of various instructions.
// V (vlong)     = 64 bit
// WU (word)     = 32 bit unsigned
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
var regNamesMIPS64 = []string{
	"ZERO", // constant 0
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
	// R23 = REGTMP not used in regalloc
	"R24",
	"R25",
	// R26 reserved by kernel
	// R27 reserved by kernel
	// R28 = REGSB not used in regalloc
	"SP",  // aka R29
	"g",   // aka R30
	"R31", // aka REGLINK

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

	"HI", // high bits of multiplication
	"LO", // low bits of multiplication

	// If you add registers, update asyncPreempt in runtime.

	// pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesMIPS64) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesMIPS64 {
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
		gp         = buildReg("R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31")
		gpg        = gp | buildReg("g")
		gpsp       = gp | buildReg("SP")
		gpspg      = gpg | buildReg("SP")
		gpspsbg    = gpspg | buildReg("SB")
		fp         = buildReg("F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31")
		lo         = buildReg("LO")
		hi         = buildReg("HI")
		callerSave = gp | fp | lo | hi | buildReg("g") // runtime.setg (and anything calling it) may clobber g
		first16    = buildReg("R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16")
		rz         = buildReg("ZERO")
	)
	// Common regInfo
	var (
		gp01     = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11     = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp11sp   = regInfo{inputs: []regMask{gpspg}, outputs: []regMask{gp}}
		gp21     = regInfo{inputs: []regMask{gpg, gpg | rz}, outputs: []regMask{gp}}
		gp2hilo  = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{hi, lo}}
		gpload   = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}
		gpstore  = regInfo{inputs: []regMask{gpspsbg, gpg | rz}}
		gpstore0 = regInfo{inputs: []regMask{gpspsbg}}
		gpxchg   = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{gp}}
		gpcas    = regInfo{inputs: []regMask{gpspsbg, gpg, gpg}, outputs: []regMask{gp}}
		fp01     = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11     = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		//fp1flags  = regInfo{inputs: []regMask{fp}}
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
		{name: "ADDV", argLength: 2, reg: gp21, asm: "ADDVU", commutative: true},                             // arg0 + arg1
		{name: "ADDVconst", argLength: 1, reg: gp11sp, asm: "ADDVU", aux: "Int64"},                           // arg0 + auxInt. auxInt is 32-bit, also in other *const ops.
		{name: "SUBV", argLength: 2, reg: gp21, asm: "SUBVU"},                                                // arg0 - arg1
		{name: "SUBVconst", argLength: 1, reg: gp11, asm: "SUBVU", aux: "Int64"},                             // arg0 - auxInt
		{name: "MULV", argLength: 2, reg: gp2hilo, asm: "MULV", commutative: true, typ: "(Int64,Int64)"},     // arg0 * arg1, signed, results hi,lo
		{name: "MULVU", argLength: 2, reg: gp2hilo, asm: "MULVU", commutative: true, typ: "(UInt64,UInt64)"}, // arg0 * arg1, unsigned, results hi,lo
		{name: "DIVV", argLength: 2, reg: gp2hilo, asm: "DIVV", typ: "(Int64,Int64)"},                        // arg0 / arg1, signed, results hi=arg0%arg1,lo=arg0/arg1
		{name: "DIVVU", argLength: 2, reg: gp2hilo, asm: "DIVVU", typ: "(UInt64,UInt64)"},                    // arg0 / arg1, signed, results hi=arg0%arg1,lo=arg0/arg1

		{name: "ADDF", argLength: 2, reg: fp21, asm: "ADDF", commutative: true}, // arg0 + arg1
		{name: "ADDD", argLength: 2, reg: fp21, asm: "ADDD", commutative: true}, // arg0 + arg1
		{name: "SUBF", argLength: 2, reg: fp21, asm: "SUBF"},                    // arg0 - arg1
		{name: "SUBD", argLength: 2, reg: fp21, asm: "SUBD"},                    // arg0 - arg1
		{name: "MULF", argLength: 2, reg: fp21, asm: "MULF", commutative: true}, // arg0 * arg1
		{name: "MULD", argLength: 2, reg: fp21, asm: "MULD", commutative: true}, // arg0 * arg1
		{name: "DIVF", argLength: 2, reg: fp21, asm: "DIVF"},                    // arg0 / arg1
		{name: "DIVD", argLength: 2, reg: fp21, asm: "DIVD"},                    // arg0 / arg1

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true},                // arg0 & arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int64"},                // arg0 & auxInt
		{name: "OR", argLength: 2, reg: gp21, asm: "OR", commutative: true},                  // arg0 | arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "OR", aux: "Int64"},                  // arg0 | auxInt
		{name: "XOR", argLength: 2, reg: gp21, asm: "XOR", commutative: true, typ: "UInt64"}, // arg0 ^ arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "XOR", aux: "Int64", typ: "UInt64"}, // arg0 ^ auxInt
		{name: "NOR", argLength: 2, reg: gp21, asm: "NOR", commutative: true},                // ^(arg0 | arg1)
		{name: "NORconst", argLength: 1, reg: gp11, asm: "NOR", aux: "Int64"},                // ^(arg0 | auxInt)

		{name: "NEGV", argLength: 1, reg: gp11},                // -arg0
		{name: "NEGF", argLength: 1, reg: fp11, asm: "NEGF"},   // -arg0, float32
		{name: "NEGD", argLength: 1, reg: fp11, asm: "NEGD"},   // -arg0, float64
		{name: "ABSD", argLength: 1, reg: fp11, asm: "ABSD"},   // abs(arg0), float64
		{name: "SQRTD", argLength: 1, reg: fp11, asm: "SQRTD"}, // sqrt(arg0), float64
		{name: "SQRTF", argLength: 1, reg: fp11, asm: "SQRTF"}, // sqrt(arg0), float32

		// shifts
		{name: "SLLV", argLength: 2, reg: gp21, asm: "SLLV"},                    // arg0 << arg1, shift amount is mod 64
		{name: "SLLVconst", argLength: 1, reg: gp11, asm: "SLLV", aux: "Int64"}, // arg0 << auxInt
		{name: "SRLV", argLength: 2, reg: gp21, asm: "SRLV"},                    // arg0 >> arg1, unsigned, shift amount is mod 64
		{name: "SRLVconst", argLength: 1, reg: gp11, asm: "SRLV", aux: "Int64"}, // arg0 >> auxInt, unsigned
		{name: "SRAV", argLength: 2, reg: gp21, asm: "SRAV"},                    // arg0 >> arg1, signed, shift amount is mod 64
		{name: "SRAVconst", argLength: 1, reg: gp11, asm: "SRAV", aux: "Int64"}, // arg0 >> auxInt, signed

		// comparisons
		{name: "SGT", argLength: 2, reg: gp21, asm: "SGT", typ: "Bool"},                      // 1 if arg0 > arg1 (signed), 0 otherwise
		{name: "SGTconst", argLength: 1, reg: gp11, asm: "SGT", aux: "Int64", typ: "Bool"},   // 1 if auxInt > arg0 (signed), 0 otherwise
		{name: "SGTU", argLength: 2, reg: gp21, asm: "SGTU", typ: "Bool"},                    // 1 if arg0 > arg1 (unsigned), 0 otherwise
		{name: "SGTUconst", argLength: 1, reg: gp11, asm: "SGTU", aux: "Int64", typ: "Bool"}, // 1 if auxInt > arg0 (unsigned), 0 otherwise

		{name: "CMPEQF", argLength: 2, reg: fp2flags, asm: "CMPEQF", typ: "Flags"}, // flags=true if arg0 = arg1, float32
		{name: "CMPEQD", argLength: 2, reg: fp2flags, asm: "CMPEQD", typ: "Flags"}, // flags=true if arg0 = arg1, float64
		{name: "CMPGEF", argLength: 2, reg: fp2flags, asm: "CMPGEF", typ: "Flags"}, // flags=true if arg0 >= arg1, float32
		{name: "CMPGED", argLength: 2, reg: fp2flags, asm: "CMPGED", typ: "Flags"}, // flags=true if arg0 >= arg1, float64
		{name: "CMPGTF", argLength: 2, reg: fp2flags, asm: "CMPGTF", typ: "Flags"}, // flags=true if arg0 > arg1, float32
		{name: "CMPGTD", argLength: 2, reg: fp2flags, asm: "CMPGTD", typ: "Flags"}, // flags=true if arg0 > arg1, float64

		// moves
		{name: "MOVVconst", argLength: 0, reg: gp01, aux: "Int64", asm: "MOVV", typ: "UInt64", rematerializeable: true},    // auxint
		{name: "MOVFconst", argLength: 0, reg: fp01, aux: "Float64", asm: "MOVF", typ: "Float32", rematerializeable: true}, // auxint as 64-bit float, convert to 32-bit float
		{name: "MOVDconst", argLength: 0, reg: fp01, aux: "Float64", asm: "MOVD", typ: "Float64", rematerializeable: true}, // auxint as 64-bit float

		{name: "MOVVaddr", argLength: 1, reg: regInfo{inputs: []regMask{buildReg("SP") | buildReg("SB")}, outputs: []regMask{gp}}, aux: "SymOff", asm: "MOVV", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxInt + aux.(*gc.Sym), arg0=SP/SB

		{name: "MOVBload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVB", typ: "Int8", faultOnNilArg0: true, symEffect: "Read"},     // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVBUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVBU", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVH", typ: "Int16", faultOnNilArg0: true, symEffect: "Read"},    // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVHU", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"}, // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVW", typ: "Int32", faultOnNilArg0: true, symEffect: "Read"},    // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVWU", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"}, // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVVload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVV", typ: "UInt64", faultOnNilArg0: true, symEffect: "Read"},   // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVFload", argLength: 2, reg: fpload, aux: "SymOff", asm: "MOVF", typ: "Float32", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVDload", argLength: 2, reg: fpload, aux: "SymOff", asm: "MOVD", typ: "Float64", faultOnNilArg0: true, symEffect: "Read"},  // load from arg0 + auxInt + aux.  arg1=mem.

		{name: "MOVBstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 1 byte of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVHstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVWstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVVstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVV", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVFstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "MOVF", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVDstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "MOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.

		{name: "ZERO", zeroWidth: true, fixedReg: true},

		// moves (no conversion)
		{name: "MOVWfpgp", argLength: 1, reg: fpgp, asm: "MOVW"}, // move float32 to int32 (no conversion). MIPS64 will perform sign-extend to 64-bit by default
		{name: "MOVWgpfp", argLength: 1, reg: gpfp, asm: "MOVW"}, // move int32 to float32 (no conversion). MIPS64 will perform sign-extend to 64-bit by default
		{name: "MOVVfpgp", argLength: 1, reg: fpgp, asm: "MOVV"}, // move float64 to int64 (no conversion).
		{name: "MOVVgpfp", argLength: 1, reg: gpfp, asm: "MOVV"}, // move int64 to float64 (no conversion).

		// conversions
		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVB"},   // move from arg0, sign-extended from byte
		{name: "MOVBUreg", argLength: 1, reg: gp11, asm: "MOVBU"}, // move from arg0, unsign-extended from byte
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVH"},   // move from arg0, sign-extended from half
		{name: "MOVHUreg", argLength: 1, reg: gp11, asm: "MOVHU"}, // move from arg0, unsign-extended from half
		{name: "MOVWreg", argLength: 1, reg: gp11, asm: "MOVW"},   // move from arg0, sign-extended from word
		{name: "MOVWUreg", argLength: 1, reg: gp11, asm: "MOVWU"}, // move from arg0, unsign-extended from word
		{name: "MOVVreg", argLength: 1, reg: gp11, asm: "MOVV"},   // move from arg0

		{name: "MOVVnop", argLength: 1, reg: regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}, resultInArg0: true}, // nop, return arg0 in same register

		{name: "MOVWF", argLength: 1, reg: fp11, asm: "MOVWF"},     // int32 -> float32
		{name: "MOVWD", argLength: 1, reg: fp11, asm: "MOVWD"},     // int32 -> float64
		{name: "MOVVF", argLength: 1, reg: fp11, asm: "MOVVF"},     // int64 -> float32
		{name: "MOVVD", argLength: 1, reg: fp11, asm: "MOVVD"},     // int64 -> float64
		{name: "TRUNCFW", argLength: 1, reg: fp11, asm: "TRUNCFW"}, // float32 -> int32
		{name: "TRUNCDW", argLength: 1, reg: fp11, asm: "TRUNCDW"}, // float64 -> int32
		{name: "TRUNCFV", argLength: 1, reg: fp11, asm: "TRUNCFV"}, // float32 -> int64
		{name: "TRUNCDV", argLength: 1, reg: fp11, asm: "TRUNCDV"}, // float64 -> int64
		{name: "MOVFD", argLength: 1, reg: fp11, asm: "MOVFD"},     // float32 -> float64
		{name: "MOVDF", argLength: 1, reg: fp11, asm: "MOVDF"},     // float64 -> float32

		// function calls
		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                               // call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                 // tail call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{inputs: []regMask{gpsp, buildReg("R22"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                         // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		// duffzero
		// arg0 = address of memory to zero
		// arg1 = mem
		// auxint = offset into duffzero code to start executing
		// returns mem
		// R1 aka mips.REGRT1 changed as side effect
		{
			name:      "DUFFZERO",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs:   []regMask{gp},
				clobbers: buildReg("R1 R31"),
			},
			faultOnNilArg0: true,
		},

		// duffcopy
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
				clobbers: buildReg("R1 R2 R31"),
			},
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// large or unaligned zeroing
		// arg0 = address of memory to zero (in R1, changed as side effect)
		// arg1 = address of the last element to zero
		// arg2 = mem
		// auxint = alignment
		// returns mem
		//	SUBV	$8, R1
		//	MOVV	R0, 8(R1)
		//	ADDV	$8, R1
		//	BNE	Rarg1, R1, -2(PC)
		{
			name:      "LoweredZero",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R1"), gp},
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
		// auxint = alignment
		// returns mem
		//	SUBV	$8, R1
		//	MOVV	8(R1), Rtmp
		//	MOVV	Rtmp, (R2)
		//	ADDV	$8, R1
		//	ADDV	$8, R2
		//	BNE	Rarg2, R1, -4(PC)
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

		// atomic and/or.
		// *arg0 &= (|=) arg1. arg2=mem. returns memory.
		// SYNC
		// LL	(Rarg0), Rtmp
		// AND	Rarg1, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		{name: "LoweredAtomicAnd32", argLength: 3, reg: gpstore, asm: "AND", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicOr32", argLength: 3, reg: gpstore, asm: "OR", faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic loads.
		// load from arg0. arg1=mem.
		// returns <value,memory> so they can be properly ordered with other loads.
		{name: "LoweredAtomicLoad8", argLength: 2, reg: gpload, faultOnNilArg0: true},
		{name: "LoweredAtomicLoad32", argLength: 2, reg: gpload, faultOnNilArg0: true},
		{name: "LoweredAtomicLoad64", argLength: 2, reg: gpload, faultOnNilArg0: true},

		// atomic stores.
		// store arg1 to arg0. arg2=mem. returns memory.
		{name: "LoweredAtomicStore8", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore32", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore64", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},
		// store zero to arg0. arg1=mem. returns memory.
		{name: "LoweredAtomicStorezero32", argLength: 2, reg: gpstore0, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStorezero64", argLength: 2, reg: gpstore0, faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange.
		// store arg1 to arg0. arg2=mem. returns <old content of *arg0, memory>.
		// SYNC
		// LL	(Rarg0), Rout
		// MOVV Rarg1, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		{name: "LoweredAtomicExchange32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicExchange64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic add.
		// *arg0 += arg1. arg2=mem. returns <new content of *arg0, memory>.
		// SYNC
		// LL	(Rarg0), Rout
		// ADDV Rarg1, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		// ADDV Rarg1, Rout
		{name: "LoweredAtomicAdd32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicAdd64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		// *arg0 += auxint. arg1=mem. returns <new content of *arg0, memory>. auxint is 32-bit.
		{name: "LoweredAtomicAddconst32", argLength: 2, reg: regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}, aux: "Int32", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicAddconst64", argLength: 2, reg: regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}, aux: "Int64", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic compare and swap.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory.
		// if *arg0 == arg1 {
		//   *arg0 = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// SYNC
		// MOVV $0, Rout
		// LL	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 4(PC)
		// MOVV Rarg2, Rout
		// SC	Rout, (Rarg0)
		// BEQ	Rout, -4(PC)
		// SYNC
		{name: "LoweredAtomicCas32", argLength: 4, reg: gpcas, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicCas64", argLength: 4, reg: gpcas, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

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
		name:            "MIPS64",
		pkg:             "cmd/internal/obj/mips",
		genfile:         "../../mips64/ssa.go",
		ops:             ops,
		blocks:          blocks,
		regnames:        regNamesMIPS64,
		gpregmask:       gp,
		fpregmask:       fp,
		specialregmask:  hi | lo,
		framepointerreg: -1, // not used
		linkreg:         int8(num["R31"]),
	})
}
