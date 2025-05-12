// Copyright 2022 The Go Authors. All rights reserved.
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
var regNamesLOONG64 = []string{
	"R0", // constant 0
	"R1",
	"SP", // aka R3
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
	"g", // aka R22
	"R23",
	"R24",
	"R25",
	"R26",
	"R27",
	"R28",
	"R29",
	// R30 is REGTMP not used in regalloc
	"R31",

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
	if len(regNamesLOONG64) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesLOONG64 {
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
		gp         = buildReg("R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R23 R24 R25 R26 R27 R28 R29 R31") // R1 is LR, R2 is thread pointer, R3 is stack pointer, R22 is g, R30 is REGTMP
		gpg        = gp | buildReg("g")
		gpsp       = gp | buildReg("SP")
		gpspg      = gpg | buildReg("SP")
		gpspsbg    = gpspg | buildReg("SB")
		fp         = buildReg("F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31")
		callerSave = gp | fp | buildReg("g") // runtime.setg (and anything calling it) may clobber g
		r1         = buildReg("R20")
		r2         = buildReg("R21")
		r3         = buildReg("R23")
		r4         = buildReg("R24")
	)
	// Common regInfo
	var (
		gp01      = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11      = regInfo{inputs: []regMask{gpg}, outputs: []regMask{gp}}
		gp11sp    = regInfo{inputs: []regMask{gpspg}, outputs: []regMask{gp}}
		gp21      = regInfo{inputs: []regMask{gpg, gpg}, outputs: []regMask{gp}}
		gpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{gp}}
		gp2load   = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{gp}}
		gpstore   = regInfo{inputs: []regMask{gpspsbg, gpg}}
		gpstore0  = regInfo{inputs: []regMask{gpspsbg}}
		gpstore2  = regInfo{inputs: []regMask{gpspsbg, gpg, gpg}}
		gpxchg    = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{gp}}
		gpcas     = regInfo{inputs: []regMask{gpspsbg, gpg, gpg}, outputs: []regMask{gp}}
		preldreg  = regInfo{inputs: []regMask{gpspg}}
		fp01      = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11      = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fp21      = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp31      = regInfo{inputs: []regMask{fp, fp, fp}, outputs: []regMask{fp}}
		fp2flags  = regInfo{inputs: []regMask{fp, fp}}
		fpload    = regInfo{inputs: []regMask{gpspsbg}, outputs: []regMask{fp}}
		fp2load   = regInfo{inputs: []regMask{gpspsbg, gpg}, outputs: []regMask{fp}}
		fpstore   = regInfo{inputs: []regMask{gpspsbg, fp}}
		fpstore2  = regInfo{inputs: []regMask{gpspsbg, gpg, fp}}
		fpgp      = regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}}
		gpfp      = regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}}
		readflags = regInfo{inputs: nil, outputs: []regMask{gp}}
	)
	ops := []opData{
		// unary ops
		{name: "NEGV", argLength: 1, reg: gp11},              // -arg0
		{name: "NEGF", argLength: 1, reg: fp11, asm: "NEGF"}, // -arg0, float32
		{name: "NEGD", argLength: 1, reg: fp11, asm: "NEGD"}, // -arg0, float64

		{name: "SQRTD", argLength: 1, reg: fp11, asm: "SQRTD"}, // sqrt(arg0), float64
		{name: "SQRTF", argLength: 1, reg: fp11, asm: "SQRTF"}, // sqrt(arg0), float32

		{name: "ABSD", argLength: 1, reg: fp11, asm: "ABSD"}, // abs(arg0), float64

		{name: "CLZW", argLength: 1, reg: gp11, asm: "CLZW"}, // Count leading (high order) zeroes (returns 0-32)
		{name: "CLZV", argLength: 1, reg: gp11, asm: "CLZV"}, // Count leading (high order) zeroes (returns 0-64)
		{name: "CTZW", argLength: 1, reg: gp11, asm: "CTZW"}, // Count trailing (low order) zeroes (returns 0-32)
		{name: "CTZV", argLength: 1, reg: gp11, asm: "CTZV"}, // Count trailing (low order) zeroes (returns 0-64)

		{name: "REVB2H", argLength: 1, reg: gp11, asm: "REVB2H"}, // Swap bytes: 0x11223344 -> 0x22114433 (sign extends to 64 bits)
		{name: "REVB2W", argLength: 1, reg: gp11, asm: "REVB2W"}, // Swap bytes: 0x1122334455667788 -> 0x4433221188776655
		{name: "REVBV", argLength: 1, reg: gp11, asm: "REVBV"},   // Swap bytes: 0x1122334455667788 -> 0x8877665544332211

		{name: "BITREV4B", argLength: 1, reg: gp11, asm: "BITREV4B"}, // Reverse the bits of each byte inside a 32-bit arg[0]
		{name: "BITREVW", argLength: 1, reg: gp11, asm: "BITREVW"},   // Reverse the bits in a 32-bit arg[0]
		{name: "BITREVV", argLength: 1, reg: gp11, asm: "BITREVV"},   // Reverse the bits in a 64-bit arg[0]

		{name: "VPCNT64", argLength: 1, reg: fp11, asm: "VPCNTV"}, // count set bits for each 64-bit unit and store the result in each 64-bit unit
		{name: "VPCNT32", argLength: 1, reg: fp11, asm: "VPCNTW"}, // count set bits for each 32-bit unit and store the result in each 32-bit unit
		{name: "VPCNT16", argLength: 1, reg: fp11, asm: "VPCNTH"}, // count set bits for each 16-bit unit and store the result in each 16-bit unit

		// binary ops
		{name: "ADDV", argLength: 2, reg: gp21, asm: "ADDVU", commutative: true},   // arg0 + arg1
		{name: "ADDVconst", argLength: 1, reg: gp11sp, asm: "ADDVU", aux: "Int64"}, // arg0 + auxInt. auxInt is 32-bit, also in other *const ops.
		{name: "SUBV", argLength: 2, reg: gp21, asm: "SUBVU"},                      // arg0 - arg1
		{name: "SUBVconst", argLength: 1, reg: gp11, asm: "SUBVU", aux: "Int64"},   // arg0 - auxInt

		{name: "MULV", argLength: 2, reg: gp21, asm: "MULV", commutative: true, typ: "Int64"},      // arg0 * arg1
		{name: "MULHV", argLength: 2, reg: gp21, asm: "MULHV", commutative: true, typ: "Int64"},    // (arg0 * arg1) >> 64, signed
		{name: "MULHVU", argLength: 2, reg: gp21, asm: "MULHVU", commutative: true, typ: "UInt64"}, // (arg0 * arg1) >> 64, unsigned
		{name: "DIVV", argLength: 2, reg: gp21, asm: "DIVV", typ: "Int64"},                         // arg0 / arg1, signed
		{name: "DIVVU", argLength: 2, reg: gp21, asm: "DIVVU", typ: "UInt64"},                      // arg0 / arg1, unsigned
		{name: "REMV", argLength: 2, reg: gp21, asm: "REMV", typ: "Int64"},                         // arg0 / arg1, signed
		{name: "REMVU", argLength: 2, reg: gp21, asm: "REMVU", typ: "UInt64"},                      // arg0 / arg1, unsigned

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

		{name: "FMADDF", argLength: 3, reg: fp31, asm: "FMADDF", commutative: true, typ: "Float32"},   // (arg0 * arg1) + arg2
		{name: "FMADDD", argLength: 3, reg: fp31, asm: "FMADDD", commutative: true, typ: "Float64"},   // (arg0 * arg1) + arg2
		{name: "FMSUBF", argLength: 3, reg: fp31, asm: "FMSUBF", commutative: true, typ: "Float32"},   // (arg0 * arg1) - arg2
		{name: "FMSUBD", argLength: 3, reg: fp31, asm: "FMSUBD", commutative: true, typ: "Float64"},   // (arg0 * arg1) - arg2
		{name: "FNMADDF", argLength: 3, reg: fp31, asm: "FNMADDF", commutative: true, typ: "Float32"}, // -((arg0 * arg1) + arg2)
		{name: "FNMADDD", argLength: 3, reg: fp31, asm: "FNMADDD", commutative: true, typ: "Float64"}, // -((arg0 * arg1) + arg2)
		{name: "FNMSUBF", argLength: 3, reg: fp31, asm: "FNMSUBF", commutative: true, typ: "Float32"}, // -((arg0 * arg1) - arg2)
		{name: "FNMSUBD", argLength: 3, reg: fp31, asm: "FNMSUBD", commutative: true, typ: "Float64"}, // -((arg0 * arg1) - arg2)

		{name: "FMINF", argLength: 2, reg: fp21, resultNotInArgs: true, asm: "FMINF", commutative: true, typ: "Float32"}, // min(arg0, arg1), float32
		{name: "FMIND", argLength: 2, reg: fp21, resultNotInArgs: true, asm: "FMIND", commutative: true, typ: "Float64"}, // min(arg0, arg1), float64
		{name: "FMAXF", argLength: 2, reg: fp21, resultNotInArgs: true, asm: "FMAXF", commutative: true, typ: "Float32"}, // max(arg0, arg1), float32
		{name: "FMAXD", argLength: 2, reg: fp21, resultNotInArgs: true, asm: "FMAXD", commutative: true, typ: "Float64"}, // max(arg0, arg1), float64

		{name: "MASKEQZ", argLength: 2, reg: gp21, asm: "MASKEQZ"},   // returns 0 if arg1 == 0, otherwise returns arg0
		{name: "MASKNEZ", argLength: 2, reg: gp21, asm: "MASKNEZ"},   // returns 0 if arg1 != 0, otherwise returns arg0
		{name: "FCOPYSGD", argLength: 2, reg: fp21, asm: "FCOPYSGD"}, // float64

		// shifts
		{name: "SLL", argLength: 2, reg: gp21, asm: "SLL"},                        // arg0 << arg1, shift amount is mod 32
		{name: "SLLV", argLength: 2, reg: gp21, asm: "SLLV"},                      // arg0 << arg1, shift amount is mod 64
		{name: "SLLconst", argLength: 1, reg: gp11, asm: "SLL", aux: "Int64"},     // arg0 << auxInt, auxInt should be in the range 0 to 31.
		{name: "SLLVconst", argLength: 1, reg: gp11, asm: "SLLV", aux: "Int64"},   // arg0 << auxInt
		{name: "SRL", argLength: 2, reg: gp21, asm: "SRL"},                        // arg0 >> arg1, shift amount is mod 32
		{name: "SRLV", argLength: 2, reg: gp21, asm: "SRLV"},                      // arg0 >> arg1, unsigned, shift amount is mod 64
		{name: "SRLconst", argLength: 1, reg: gp11, asm: "SRL", aux: "Int64"},     // arg0 >> auxInt, auxInt should be in the range 0 to 31.
		{name: "SRLVconst", argLength: 1, reg: gp11, asm: "SRLV", aux: "Int64"},   // arg0 >> auxInt, unsigned
		{name: "SRA", argLength: 2, reg: gp21, asm: "SRA"},                        // arg0 >> arg1, shift amount is mod 32
		{name: "SRAV", argLength: 2, reg: gp21, asm: "SRAV"},                      // arg0 >> arg1, signed, shift amount is mod 64
		{name: "SRAconst", argLength: 1, reg: gp11, asm: "SRA", aux: "Int64"},     // arg0 >> auxInt, signed, auxInt should be in the range 0 to 31.
		{name: "SRAVconst", argLength: 1, reg: gp11, asm: "SRAV", aux: "Int64"},   // arg0 >> auxInt, signed
		{name: "ROTR", argLength: 2, reg: gp21, asm: "ROTR"},                      // arg0 right rotate by (arg1 mod 32) bits
		{name: "ROTRV", argLength: 2, reg: gp21, asm: "ROTRV"},                    // arg0 right rotate by (arg1 mod 64) bits
		{name: "ROTRconst", argLength: 1, reg: gp11, asm: "ROTR", aux: "Int64"},   // uint32(arg0) right rotate by auxInt bits, auxInt should be in the range 0 to 31.
		{name: "ROTRVconst", argLength: 1, reg: gp11, asm: "ROTRV", aux: "Int64"}, // arg0 right rotate by auxInt bits, auxInt should be in the range 0 to 63.

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

		// bitfield ops
		// for bstrpick.w msbw is auxInt>>5, lsbw is auxInt&0x1f
		// for bstrpick.d msbd is auxInt>>6, lsbd is auxInt&0x3f
		{name: "BSTRPICKW", argLength: 1, reg: gp11, asm: "BSTRPICKW", aux: "Int64"},
		{name: "BSTRPICKV", argLength: 1, reg: gp11, asm: "BSTRPICKV", aux: "Int64"},

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

		// register indexed load
		{name: "MOVVloadidx", argLength: 3, reg: gp2load, asm: "MOVV", typ: "UInt64"},   // load 64-bit dword from arg0 + arg1, arg2 = mem.
		{name: "MOVWloadidx", argLength: 3, reg: gp2load, asm: "MOVW", typ: "Int32"},    // load 32-bit word from arg0 + arg1, sign-extended to 64-bit, arg2=mem.
		{name: "MOVWUloadidx", argLength: 3, reg: gp2load, asm: "MOVWU", typ: "UInt32"}, // load 32-bit word from arg0 + arg1, zero-extended to 64-bit, arg2=mem.
		{name: "MOVHloadidx", argLength: 3, reg: gp2load, asm: "MOVH", typ: "Int16"},    // load 16-bit word from arg0 + arg1, sign-extended to 64-bit, arg2=mem.
		{name: "MOVHUloadidx", argLength: 3, reg: gp2load, asm: "MOVHU", typ: "UInt16"}, // load 16-bit word from arg0 + arg1, zero-extended to 64-bit, arg2=mem.
		{name: "MOVBloadidx", argLength: 3, reg: gp2load, asm: "MOVB", typ: "Int8"},     // load 8-bit word from arg0 + arg1, sign-extended to 64-bit, arg2=mem.
		{name: "MOVBUloadidx", argLength: 3, reg: gp2load, asm: "MOVBU", typ: "UInt8"},  // load 8-bit word from arg0 + arg1, zero-extended to 64-bit, arg2=mem.
		{name: "MOVFloadidx", argLength: 3, reg: fp2load, asm: "MOVF", typ: "Float32"},  // load 32-bit float from arg0 + arg1, arg2=mem.
		{name: "MOVDloadidx", argLength: 3, reg: fp2load, asm: "MOVD", typ: "Float64"},  // load 64-bit float from arg0 + arg1, arg2=mem.

		{name: "MOVBstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 1 byte of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVHstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVWstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVVstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVV", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVFstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "MOVF", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVDstore", argLength: 3, reg: fpstore, aux: "SymOff", asm: "MOVD", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.

		// register indexed store
		{name: "MOVBstoreidx", argLength: 4, reg: gpstore2, asm: "MOVB", typ: "Mem"}, // store 1 byte of arg2 to arg0 + arg1, arg3 = mem.
		{name: "MOVHstoreidx", argLength: 4, reg: gpstore2, asm: "MOVH", typ: "Mem"}, // store 2 bytes of arg2 to arg0 + arg1, arg3 = mem.
		{name: "MOVWstoreidx", argLength: 4, reg: gpstore2, asm: "MOVW", typ: "Mem"}, // store 4 bytes of arg2 to arg0 + arg1, arg3 = mem.
		{name: "MOVVstoreidx", argLength: 4, reg: gpstore2, asm: "MOVV", typ: "Mem"}, // store 8 bytes of arg2 to arg0 + arg1, arg3 = mem.
		{name: "MOVFstoreidx", argLength: 4, reg: fpstore2, asm: "MOVF", typ: "Mem"}, // store 32-bit float of arg2 to arg0 + arg1, arg3=mem.
		{name: "MOVDstoreidx", argLength: 4, reg: fpstore2, asm: "MOVD", typ: "Mem"}, // store 64-bit float of arg2 to arg0 + arg1, arg3=mem.

		{name: "MOVBstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVB", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 1 byte of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVH", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVW", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes of zero to arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVVstorezero", argLength: 2, reg: gpstore0, aux: "SymOff", asm: "MOVV", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes of zero to arg0 + auxInt + aux.  ar12=mem.

		// register indexed store zero
		{name: "MOVBstorezeroidx", argLength: 3, reg: gpstore, asm: "MOVB", typ: "Mem"}, // store 1 byte of zero to arg0 + arg1, arg2 = mem.
		{name: "MOVHstorezeroidx", argLength: 3, reg: gpstore, asm: "MOVH", typ: "Mem"}, // store 2 bytes of zero to arg0 + arg1, arg2 = mem.
		{name: "MOVWstorezeroidx", argLength: 3, reg: gpstore, asm: "MOVW", typ: "Mem"}, // store 4 bytes of zero to arg0 + arg1, arg2 = mem.
		{name: "MOVVstorezeroidx", argLength: 3, reg: gpstore, asm: "MOVV", typ: "Mem"}, // store 8 bytes of zero to arg0 + arg1, arg2 = mem.

		// moves (no conversion)
		{name: "MOVWfpgp", argLength: 1, reg: fpgp, asm: "MOVW"}, // move float32 to int32 (no conversion).
		{name: "MOVWgpfp", argLength: 1, reg: gpfp, asm: "MOVW"}, // move int32 to float32 (no conversion).
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

		// Round ops to block fused-multiply-add extraction.
		{name: "LoweredRound32F", argLength: 1, reg: fp11, resultInArg0: true},
		{name: "LoweredRound64F", argLength: 1, reg: fp11, resultInArg0: true},

		// function calls
		{name: "CALLstatic", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                               // call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                                 // tail call static function aux.(*obj.LSym).  last arg=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: -1, reg: regInfo{inputs: []regMask{gpsp, buildReg("R29"), 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, last arg=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: -1, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                         // call fn by pointer.  arg0=codeptr, last arg=mem, auxint=argsize, returns mem

		// duffzero
		// arg0 = address of memory to zero
		// arg1 = mem
		// auxint = offset into duffzero code to start executing
		// returns mem
		// R20 aka loong64.REGRT1 changed as side effect
		{
			name:      "DUFFZERO",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs:   []regMask{buildReg("R20")},
				clobbers: buildReg("R20 R1"),
			},
			typ:            "Mem",
			faultOnNilArg0: true,
		},

		// duffcopy
		// arg0 = address of dst memory (in R21, changed as side effect)
		// arg1 = address of src memory (in R20, changed as side effect)
		// arg2 = mem
		// auxint = offset into duffcopy code to start executing
		// returns mem
		{
			name:      "DUFFCOPY",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R21"), buildReg("R20")},
				clobbers: buildReg("R20 R21 R1"),
			},
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		// large or unaligned zeroing
		// arg0 = address of memory to zero (in R20, changed as side effect)
		// arg1 = address of the last element to zero
		// arg2 = mem
		// auxint = alignment
		// returns mem
		//	MOVx	R0, (R20)
		//	ADDV	$sz, R20
		//	BGEU	Rarg1, R20, -2(PC)
		{
			name:      "LoweredZero",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R20"), gp},
				clobbers: buildReg("R20"),
			},
			typ:            "Mem",
			faultOnNilArg0: true,
		},

		// large or unaligned move
		// arg0 = address of dst memory (in R21, changed as side effect)
		// arg1 = address of src memory (in R20, changed as side effect)
		// arg2 = address of the last element of src
		// arg3 = mem
		// auxint = alignment
		// returns mem
		//	MOVx	(R20), Rtmp
		//	MOVx	Rtmp, (R21)
		//	ADDV	$sz, R20
		//	ADDV	$sz, R21
		//	BGEU	Rarg2, R20, -4(PC)
		{
			name:      "LoweredMove",
			aux:       "Int64",
			argLength: 4,
			reg: regInfo{
				inputs:   []regMask{buildReg("R21"), buildReg("R20"), gp},
				clobbers: buildReg("R20 R21"),
			},
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

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
		{name: "LoweredAtomicStore8Variant", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore32Variant", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore64Variant", argLength: 3, reg: gpstore, faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange.
		// store arg1 to arg0. arg2=mem. returns <old content of *arg0, memory>.
		{name: "LoweredAtomicExchange32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicExchange64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange variant.
		// store arg1 to arg0. arg2=mem. returns <old content of *arg0, memory>. auxint must be zero.
		// AMSWAPDBB   Rarg1, (Rarg0), Rout
		{name: "LoweredAtomicExchange8Variant", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic add.
		// *arg0 += arg1. arg2=mem. returns <new content of *arg0, memory>.
		{name: "LoweredAtomicAdd32", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAdd64", argLength: 3, reg: gpxchg, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic compare and swap.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory.
		// if *arg0 == arg1 {
		//   *arg0 = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// MOVV $0, Rout
		// DBAR 0x14
		// LL	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 4(PC)
		// MOVV Rarg2, Rout
		// SC	Rout, (Rarg0)
		// BEQ	Rout, -4(PC)
		// DBAR 0x12
		{name: "LoweredAtomicCas32", argLength: 4, reg: gpcas, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicCas64", argLength: 4, reg: gpcas, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// atomic compare and swap variant.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory. auxint must be zero.
		// if *arg0 == arg1 {
		//   *arg0 = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// MOVV         $0, Rout
		// MOVV         Rarg1, Rtmp
		// AMCASDBx     Rarg2, (Rarg0), Rtmp
		// BNE          Rarg1, Rtmp, 2(PC)
		// MOVV         $1, Rout
		// NOP
		{name: "LoweredAtomicCas64Variant", argLength: 4, reg: gpcas, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},
		{name: "LoweredAtomicCas32Variant", argLength: 4, reg: gpcas, resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true, unsafePoint: true},

		// Atomic 32 bit AND/OR.
		// *arg0 &= (|=) arg1. arg2=mem. returns nil.
		{name: "LoweredAtomicAnd32", argLength: 3, reg: gpxchg, asm: "AMANDDBW", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicOr32", argLength: 3, reg: gpxchg, asm: "AMORDBW", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// Atomic 32,64 bit AND/OR.
		// *arg0 &= (|=) arg1. arg2=mem. returns <old content of *arg0, memory>. auxint must be zero.
		{name: "LoweredAtomicAnd32value", argLength: 3, reg: gpxchg, asm: "AMANDDBW", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAnd64value", argLength: 3, reg: gpxchg, asm: "AMANDDBV", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicOr32value", argLength: 3, reg: gpxchg, asm: "AMORDBW", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicOr64value", argLength: 3, reg: gpxchg, asm: "AMORDBV", resultNotInArgs: true, faultOnNilArg0: true, hasSideEffects: true},

		// pseudo-ops
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gpg}}, nilCheck: true, faultOnNilArg0: true}, // panic if arg0 is nil.  arg1=mem.

		{name: "FPFlagTrue", argLength: 1, reg: readflags},  // bool, true if FP flag is true
		{name: "FPFlagFalse", argLength: 1, reg: readflags}, // bool, true if FP flag is false

		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of R22 (loong64.REGCTXT, the closure pointer)
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{buildReg("R29")}}, zeroWidth: true},

		// LoweredGetCallerSP returns the SP of the caller of the current function. arg0=mem.
		{name: "LoweredGetCallerSP", argLength: 1, reg: gp01, rematerializeable: true},

		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" sys.GetCallerPC,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},

		// LoweredWB invokes runtime.gcWriteBarrier. arg0=mem, auxint=# of buffer entries needed
		// It saves all GP registers if necessary,
		// but clobbers R1 (LR) because it's a call
		// and R30 (REGTMP).
		// Returns a pointer to a write barrier buffer in R29.
		{name: "LoweredWB", argLength: 1, reg: regInfo{clobbers: (callerSave &^ gpg) | buildReg("R1"), outputs: []regMask{buildReg("R29")}}, clobberFlags: true, aux: "Int64"},

		// Do data barrier. arg0=memorys
		{name: "LoweredPubBarrier", argLength: 1, asm: "DBAR", hasSideEffects: true},

		// There are three of these functions so that they can have three different register inputs.
		// When we check 0 <= c <= cap (A), then 0 <= b <= c (B), then 0 <= a <= b (C), we want the
		// default registers to match so we don't need to copy registers around unnecessarily.
		{name: "LoweredPanicBoundsA", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r3, r4}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsB", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r2, r3}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsC", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r1, r2}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).

		// Prefetch instruction
		// Do prefetch arg0 address with option aux. arg0=addr, arg1=memory, aux=option.
		// Note:
		//   The aux of PRELDX is actually composed of two values: $hint and $n. bit[4:0]
		//   is $hint and bit[41:5] is $n.
		{name: "PRELD", argLength: 2, aux: "Int64", reg: preldreg, asm: "PRELD", hasSideEffects: true},
		{name: "PRELDX", argLength: 2, aux: "Int64", reg: preldreg, asm: "PRELDX", hasSideEffects: true},
	}

	blocks := []blockData{
		{name: "EQ", controls: 1},
		{name: "NE", controls: 1},
		{name: "LTZ", controls: 1},  // < 0
		{name: "LEZ", controls: 1},  // <= 0
		{name: "GTZ", controls: 1},  // > 0
		{name: "GEZ", controls: 1},  // >= 0
		{name: "FPT", controls: 1},  // FP flag is true
		{name: "FPF", controls: 1},  // FP flag is false
		{name: "BEQ", controls: 2},  // controls[0] == controls[1]
		{name: "BNE", controls: 2},  // controls[0] == controls[1]
		{name: "BGE", controls: 2},  // controls[0] >= controls[1]
		{name: "BLT", controls: 2},  // controls[0] < controls[1]
		{name: "BGEU", controls: 2}, // controls[0] >= controls[1], unsigned
		{name: "BLTU", controls: 2}, // controls[0] < controls[1], unsigned
	}

	archs = append(archs, arch{
		name:     "LOONG64",
		pkg:      "cmd/internal/obj/loong64",
		genfile:  "../../loong64/ssa.go",
		ops:      ops,
		blocks:   blocks,
		regnames: regNamesLOONG64,
		// TODO: support register ABI on loong64
		ParamIntRegNames:   "R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19",
		ParamFloatRegNames: "F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15",
		gpregmask:          gp,
		fpregmask:          fp,
		framepointerreg:    -1, // not used
		linkreg:            int8(num["R1"]),
	})
}
