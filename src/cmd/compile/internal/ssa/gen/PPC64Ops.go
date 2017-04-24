// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import "strings"

// Notes:
//  - Less-than-64-bit integer types live in the low portion of registers.
//    For now, the upper portion is junk; sign/zero-extension might be optimized in the future, but not yet.
//  - Boolean types are zero or 1; stored in a byte, but loaded with AMOVBZ so the upper bytes of a register are zero.
//  - *const instructions may use a constant larger than the instruction can encode.
//    In this case the assembler expands to multiple instructions and uses tmp
//    register (R31).

var regNamesPPC64 = []string{
	"R0", // REGZERO, not used, but simplifies counting in regalloc
	"SP", // REGSP
	"SB", // REGSB
	"R3",
	"R4",
	"R5",
	"R6",
	"R7",
	"R8",
	"R9",
	"R10",
	"R11", // REGCTXT for closures
	"R12",
	"R13", // REGTLS
	"R14",
	"R15",
	"R16",
	"R17",
	"R18",
	"R19",
	"R20",
	"R21",
	"R22",
	"R23",
	"R24",
	"R25",
	"R26",
	"R27",
	"R28",
	"R29",
	"g",   // REGG.  Using name "g" and setting Config.hasGReg makes it "just happen".
	"R31", // REGTMP

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

	// "CR0",
	// "CR1",
	// "CR2",
	// "CR3",
	// "CR4",
	// "CR5",
	// "CR6",
	// "CR7",

	// "CR",
	// "XER",
	// "LR",
	// "CTR",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesPPC64) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesPPC64 {
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

	var (
		gp = buildReg("R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29")
		fp = buildReg("F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26")
		sp = buildReg("SP")
		sb = buildReg("SB")
		gr = buildReg("g")
		// cr  = buildReg("CR")
		// ctr = buildReg("CTR")
		// lr  = buildReg("LR")
		tmp  = buildReg("R31")
		ctxt = buildReg("R11")
		// tls = buildReg("R13")
		gp01        = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11        = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}
		gp21        = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{gp}}
		gp1cr       = regInfo{inputs: []regMask{gp | sp | sb}}
		gp2cr       = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}}
		crgp        = regInfo{inputs: nil, outputs: []regMask{gp}}
		gpload      = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}
		gpstore     = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}}
		gpstorezero = regInfo{inputs: []regMask{gp | sp | sb}} // ppc64.REGZERO is reserved zero value
		gpxchg      = regInfo{inputs: []regMask{gp | sp | sb, gp}, outputs: []regMask{gp}}
		gpcas       = regInfo{inputs: []regMask{gp | sp | sb, gp, gp}, outputs: []regMask{gp}}
		fp01        = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11        = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fpgp        = regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}}
		gpfp        = regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}}
		fp21        = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp31        = regInfo{inputs: []regMask{fp, fp, fp}, outputs: []regMask{fp}}
		fp2cr       = regInfo{inputs: []regMask{fp, fp}}
		fpload      = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{fp}}
		fpstore     = regInfo{inputs: []regMask{gp | sp | sb, fp}}
		callerSave  = regMask(gp | fp | gr)
	)
	ops := []opData{
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},                     // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11, asm: "ADD", aux: "SymOff", symEffect: "Addr"}, // arg0 + auxInt + aux.(*gc.Sym)
		{name: "FADD", argLength: 2, reg: fp21, asm: "FADD", commutative: true},                   // arg0+arg1
		{name: "FADDS", argLength: 2, reg: fp21, asm: "FADDS", commutative: true},                 // arg0+arg1
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB"},                                        // arg0-arg1
		{name: "FSUB", argLength: 2, reg: fp21, asm: "FSUB"},                                      // arg0-arg1
		{name: "FSUBS", argLength: 2, reg: fp21, asm: "FSUBS"},                                    // arg0-arg1

		{name: "MULLD", argLength: 2, reg: gp21, asm: "MULLD", typ: "Int64", commutative: true}, // arg0*arg1 (signed 64-bit)
		{name: "MULLW", argLength: 2, reg: gp21, asm: "MULLW", typ: "Int32", commutative: true}, // arg0*arg1 (signed 32-bit)

		{name: "MULHD", argLength: 2, reg: gp21, asm: "MULHD", commutative: true},   // (arg0 * arg1) >> 64, signed
		{name: "MULHW", argLength: 2, reg: gp21, asm: "MULHW", commutative: true},   // (arg0 * arg1) >> 32, signed
		{name: "MULHDU", argLength: 2, reg: gp21, asm: "MULHDU", commutative: true}, // (arg0 * arg1) >> 64, unsigned
		{name: "MULHWU", argLength: 2, reg: gp21, asm: "MULHWU", commutative: true}, // (arg0 * arg1) >> 32, unsigned

		{name: "FMUL", argLength: 2, reg: fp21, asm: "FMUL", commutative: true},   // arg0*arg1
		{name: "FMULS", argLength: 2, reg: fp21, asm: "FMULS", commutative: true}, // arg0*arg1

		{name: "FMADD", argLength: 3, reg: fp31, asm: "FMADD"},   // arg0*arg1 + arg2
		{name: "FMADDS", argLength: 3, reg: fp31, asm: "FMADDS"}, // arg0*arg1 + arg2
		{name: "FMSUB", argLength: 3, reg: fp31, asm: "FMSUB"},   // arg0*arg1 - arg2
		{name: "FMSUBS", argLength: 3, reg: fp31, asm: "FMSUBS"}, // arg0*arg1 - arg2

		{name: "SRAD", argLength: 2, reg: gp21, asm: "SRAD"}, // arg0 >>a arg1, 64 bits (all sign if arg1 & 64 != 0)
		{name: "SRAW", argLength: 2, reg: gp21, asm: "SRAW"}, // arg0 >>a arg1, 32 bits (all sign if arg1 & 32 != 0)
		{name: "SRD", argLength: 2, reg: gp21, asm: "SRD"},   // arg0 >> arg1, 64 bits  (0 if arg1 & 64 != 0)
		{name: "SRW", argLength: 2, reg: gp21, asm: "SRW"},   // arg0 >> arg1, 32 bits  (0 if arg1 & 32 != 0)
		{name: "SLD", argLength: 2, reg: gp21, asm: "SLD"},   // arg0 << arg1, 64 bits  (0 if arg1 & 64 != 0)
		{name: "SLW", argLength: 2, reg: gp21, asm: "SLW"},   // arg0 << arg1, 32 bits  (0 if arg1 & 32 != 0)

		{name: "ADDconstForCarry", argLength: 1, reg: regInfo{inputs: []regMask{gp | sp | sb}, clobbers: tmp}, aux: "Int16", asm: "ADDC", typ: "Flags"}, // _, carry := arg0 + aux
		{name: "MaskIfNotCarry", argLength: 1, reg: crgp, asm: "ADDME", typ: "Int64"},                                                                   // carry - 1 (if carry then 0 else -1)

		{name: "SRADconst", argLength: 1, reg: gp11, asm: "SRAD", aux: "Int64"}, // arg0 >>a aux, 64 bits
		{name: "SRAWconst", argLength: 1, reg: gp11, asm: "SRAW", aux: "Int64"}, // arg0 >>a aux, 32 bits
		{name: "SRDconst", argLength: 1, reg: gp11, asm: "SRD", aux: "Int64"},   // arg0 >> aux, 64 bits
		{name: "SRWconst", argLength: 1, reg: gp11, asm: "SRW", aux: "Int64"},   // arg0 >> aux, 32 bits
		{name: "SLDconst", argLength: 1, reg: gp11, asm: "SLD", aux: "Int64"},   // arg0 << aux, 64 bits
		{name: "SLWconst", argLength: 1, reg: gp11, asm: "SLW", aux: "Int64"},   // arg0 << aux, 32 bits

		{name: "ROTLconst", argLength: 1, reg: gp11, asm: "ROTL", aux: "Int64"},   // arg0 rotate left by auxInt bits
		{name: "ROTLWconst", argLength: 1, reg: gp11, asm: "ROTLW", aux: "Int64"}, // uint32(arg0) rotate left by auxInt bits

		{name: "CNTLZD", argLength: 1, reg: gp11, asm: "CNTLZD", clobberFlags: true}, // count leading zeros
		{name: "CNTLZW", argLength: 1, reg: gp11, asm: "CNTLZW", clobberFlags: true}, // count leading zeros (32 bit)

		{name: "POPCNTD", argLength: 1, reg: gp11, asm: "POPCNTD"}, // number of set bits in arg0
		{name: "POPCNTW", argLength: 1, reg: gp11, asm: "POPCNTW"}, // number of set bits in each word of arg0 placed in corresponding word
		{name: "POPCNTB", argLength: 1, reg: gp11, asm: "POPCNTB"}, // number of set bits in each byte of arg0 placed in corresonding byte

		{name: "FDIV", argLength: 2, reg: fp21, asm: "FDIV"},   // arg0/arg1
		{name: "FDIVS", argLength: 2, reg: fp21, asm: "FDIVS"}, // arg0/arg1

		{name: "DIVD", argLength: 2, reg: gp21, asm: "DIVD", typ: "Int64"},   // arg0/arg1 (signed 64-bit)
		{name: "DIVW", argLength: 2, reg: gp21, asm: "DIVW", typ: "Int32"},   // arg0/arg1 (signed 32-bit)
		{name: "DIVDU", argLength: 2, reg: gp21, asm: "DIVDU", typ: "Int64"}, // arg0/arg1 (unsigned 64-bit)
		{name: "DIVWU", argLength: 2, reg: gp21, asm: "DIVWU", typ: "Int32"}, // arg0/arg1 (unsigned 32-bit)

		// MOD is implemented as rem := arg0 - (arg0/arg1) * arg1

		// Conversions are all float-to-float register operations.  "Integer" refers to encoding in the FP register.
		{name: "FCTIDZ", argLength: 1, reg: fp11, asm: "FCTIDZ", typ: "Float64"}, // convert float to 64-bit int round towards zero
		{name: "FCTIWZ", argLength: 1, reg: fp11, asm: "FCTIWZ", typ: "Float64"}, // convert float to 32-bit int round towards zero
		{name: "FCFID", argLength: 1, reg: fp11, asm: "FCFID", typ: "Float64"},   // convert 64-bit integer to float
		{name: "FRSP", argLength: 1, reg: fp11, asm: "FRSP", typ: "Float64"},     // round float to 32-bit value

		// Movement between float and integer registers with no change in bits; accomplished with stores+loads on PPC.
		// Because the 32-bit load-literal-bits instructions have impoverished addressability, always widen the
		// data instead and use FMOVDload and FMOVDstore instead (this will also dodge endianess issues).
		// There are optimizations that should apply -- (Xi2f64 (MOVWload (not-ADD-ptr+offset) ) ) could use
		// the word-load instructions.  (Xi2f64 (MOVDload ptr )) can be (FMOVDload ptr)

		{name: "Xf2i64", argLength: 1, reg: fpgp, typ: "Int64"},   // move 64 bits of F register into G register
		{name: "Xi2f64", argLength: 1, reg: gpfp, typ: "Float64"}, // move 64 bits of G register into F register

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true},               // arg0&arg1
		{name: "ANDN", argLength: 2, reg: gp21, asm: "ANDN"},                                // arg0&^arg1
		{name: "OR", argLength: 2, reg: gp21, asm: "OR", commutative: true},                 // arg0|arg1
		{name: "ORN", argLength: 2, reg: gp21, asm: "ORN"},                                  // arg0|^arg1
		{name: "NOR", argLength: 2, reg: gp21, asm: "NOR", commutative: true},               // ^(arg0|arg1)
		{name: "XOR", argLength: 2, reg: gp21, asm: "XOR", typ: "Int64", commutative: true}, // arg0^arg1
		{name: "EQV", argLength: 2, reg: gp21, asm: "EQV", typ: "Int64", commutative: true}, // arg0^^arg1
		{name: "NEG", argLength: 1, reg: gp11, asm: "NEG"},                                  // -arg0 (integer)
		{name: "FNEG", argLength: 1, reg: fp11, asm: "FNEG"},                                // -arg0 (floating point)
		{name: "FSQRT", argLength: 1, reg: fp11, asm: "FSQRT"},                              // sqrt(arg0) (floating point)
		{name: "FSQRTS", argLength: 1, reg: fp11, asm: "FSQRTS"},                            // sqrt(arg0) (floating point, single precision)

		{name: "ORconst", argLength: 1, reg: gp11, asm: "OR", aux: "Int64"},                                                                                     // arg0|aux
		{name: "XORconst", argLength: 1, reg: gp11, asm: "XOR", aux: "Int64"},                                                                                   // arg0^aux
		{name: "ANDconst", argLength: 1, reg: regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}, asm: "ANDCC", aux: "Int64", clobberFlags: true}, // arg0&aux // and-immediate sets CC on PPC, always.
		{name: "ANDCCconst", argLength: 1, reg: regInfo{inputs: []regMask{gp | sp | sb}}, asm: "ANDCC", aux: "Int64", typ: "Flags"},                             // arg0&aux == 0 // and-immediate sets CC on PPC, always.

		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVB", typ: "Int64"},                                                               // sign extend int8 to int64
		{name: "MOVBZreg", argLength: 1, reg: gp11, asm: "MOVBZ", typ: "Int64"},                                                             // zero extend uint8 to uint64
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVH", typ: "Int64"},                                                               // sign extend int16 to int64
		{name: "MOVHZreg", argLength: 1, reg: gp11, asm: "MOVHZ", typ: "Int64"},                                                             // zero extend uint16 to uint64
		{name: "MOVWreg", argLength: 1, reg: gp11, asm: "MOVW", typ: "Int64"},                                                               // sign extend int32 to int64
		{name: "MOVWZreg", argLength: 1, reg: gp11, asm: "MOVWZ", typ: "Int64"},                                                             // zero extend uint32 to uint64
		{name: "MOVBZload", argLength: 2, reg: gpload, asm: "MOVBZ", aux: "SymOff", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},  // zero extend uint8 to uint64
		{name: "MOVHload", argLength: 2, reg: gpload, asm: "MOVH", aux: "SymOff", typ: "Int16", faultOnNilArg0: true, symEffect: "Read"},    // sign extend int16 to int64
		{name: "MOVHZload", argLength: 2, reg: gpload, asm: "MOVHZ", aux: "SymOff", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"}, // zero extend uint16 to uint64
		{name: "MOVWload", argLength: 2, reg: gpload, asm: "MOVW", aux: "SymOff", typ: "Int32", faultOnNilArg0: true, symEffect: "Read"},    // sign extend int32 to int64
		{name: "MOVWZload", argLength: 2, reg: gpload, asm: "MOVWZ", aux: "SymOff", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"}, // zero extend uint32 to uint64
		{name: "MOVDload", argLength: 2, reg: gpload, asm: "MOVD", aux: "SymOff", typ: "Int64", faultOnNilArg0: true, symEffect: "Read"},

		{name: "FMOVDload", argLength: 2, reg: fpload, asm: "FMOVD", aux: "SymOff", typ: "Float64", faultOnNilArg0: true, symEffect: "Read"},
		{name: "FMOVSload", argLength: 2, reg: fpload, asm: "FMOVS", aux: "SymOff", typ: "Float32", faultOnNilArg0: true, symEffect: "Read"},
		{name: "MOVBstore", argLength: 3, reg: gpstore, asm: "MOVB", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVHstore", argLength: 3, reg: gpstore, asm: "MOVH", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVWstore", argLength: 3, reg: gpstore, asm: "MOVW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "MOVDstore", argLength: 3, reg: gpstore, asm: "MOVD", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "FMOVDstore", argLength: 3, reg: fpstore, asm: "FMOVD", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},
		{name: "FMOVSstore", argLength: 3, reg: fpstore, asm: "FMOVS", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"},

		{name: "MOVBstorezero", argLength: 2, reg: gpstorezero, asm: "MOVB", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero byte to arg0+aux.  arg1=mem
		{name: "MOVHstorezero", argLength: 2, reg: gpstorezero, asm: "MOVH", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero 2 bytes to ...
		{name: "MOVWstorezero", argLength: 2, reg: gpstorezero, asm: "MOVW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero 4 bytes to ...
		{name: "MOVDstorezero", argLength: 2, reg: gpstorezero, asm: "MOVD", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero 8 bytes to ...

		{name: "MOVDaddr", argLength: 1, reg: regInfo{inputs: []regMask{sp | sb}, outputs: []regMask{gp}}, aux: "SymOff", asm: "MOVD", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxInt + aux.(*gc.Sym), arg0=SP/SB

		{name: "MOVDconst", argLength: 0, reg: gp01, aux: "Int64", asm: "MOVD", typ: "Int64", rematerializeable: true}, //
		{name: "FMOVDconst", argLength: 0, reg: fp01, aux: "Float64", asm: "FMOVD", rematerializeable: true},           //
		{name: "FMOVSconst", argLength: 0, reg: fp01, aux: "Float32", asm: "FMOVS", rematerializeable: true},           //
		{name: "FCMPU", argLength: 2, reg: fp2cr, asm: "FCMPU", typ: "Flags"},

		{name: "CMP", argLength: 2, reg: gp2cr, asm: "CMP", typ: "Flags"},     // arg0 compare to arg1
		{name: "CMPU", argLength: 2, reg: gp2cr, asm: "CMPU", typ: "Flags"},   // arg0 compare to arg1
		{name: "CMPW", argLength: 2, reg: gp2cr, asm: "CMPW", typ: "Flags"},   // arg0 compare to arg1
		{name: "CMPWU", argLength: 2, reg: gp2cr, asm: "CMPWU", typ: "Flags"}, // arg0 compare to arg1
		{name: "CMPconst", argLength: 1, reg: gp1cr, asm: "CMP", aux: "Int64", typ: "Flags"},
		{name: "CMPUconst", argLength: 1, reg: gp1cr, asm: "CMPU", aux: "Int64", typ: "Flags"},
		{name: "CMPWconst", argLength: 1, reg: gp1cr, asm: "CMPW", aux: "Int32", typ: "Flags"},
		{name: "CMPWUconst", argLength: 1, reg: gp1cr, asm: "CMPWU", aux: "Int32", typ: "Flags"},

		// pseudo-ops
		{name: "Equal", argLength: 1, reg: crgp},         // bool, true flags encode x==y false otherwise.
		{name: "NotEqual", argLength: 1, reg: crgp},      // bool, true flags encode x!=y false otherwise.
		{name: "LessThan", argLength: 1, reg: crgp},      // bool, true flags encode  x<y false otherwise.
		{name: "FLessThan", argLength: 1, reg: crgp},     // bool, true flags encode  x<y false otherwise.
		{name: "LessEqual", argLength: 1, reg: crgp},     // bool, true flags encode  x<=y false otherwise.
		{name: "FLessEqual", argLength: 1, reg: crgp},    // bool, true flags encode  x<=y false otherwise; PPC <= === !> which is wrong for NaN
		{name: "GreaterThan", argLength: 1, reg: crgp},   // bool, true flags encode  x>y false otherwise.
		{name: "FGreaterThan", argLength: 1, reg: crgp},  // bool, true flags encode  x>y false otherwise.
		{name: "GreaterEqual", argLength: 1, reg: crgp},  // bool, true flags encode  x>=y false otherwise.
		{name: "FGreaterEqual", argLength: 1, reg: crgp}, // bool, true flags encode  x>=y false otherwise.; PPC >= === !< which is wrong for NaN

		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of the closure pointer.
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{ctxt}}},

		//arg0=ptr,arg1=mem, returns void.  Faults if ptr is nil.
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gp | sp | sb}, clobbers: tmp}, clobberFlags: true, nilCheck: true, faultOnNilArg0: true},
		// Round ops to block fused-multiply-add extraction.
		{name: "LoweredRound32F", argLength: 1, reg: fp11, resultInArg0: true},
		{name: "LoweredRound64F", argLength: 1, reg: fp11, resultInArg0: true},

		// Convert pointer to integer, takes a memory operand for ordering.
		{name: "MOVDconvert", argLength: 2, reg: gp11, asm: "MOVD"},

		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "SymOff", clobberFlags: true, call: true, symEffect: "None"},                   // call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{inputs: []regMask{gp | sp, ctxt, 0}, clobbers: callerSave}, aux: "Int64", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "Int64", clobberFlags: true, call: true},                 // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		// large or unaligned zeroing
		// arg0 = address of memory to zero (in R3, changed as side effect)
		// returns mem
		//
		// a loop is generated when there is more than one iteration
		// needed to clear 4 doublewords
		//
		// 	MOVD	$len/32,R31
		//	MOVD	R31,CTR
		//	loop:
		//	MOVD	R0,(R3)
		//	MOVD	R0,8(R3)
		//	MOVD	R0,16(R3)
		//	MOVD	R0,24(R3)
		//	ADD	R3,32
		//	BC	loop

		// remaining doubleword clears generated as needed
		//	MOVD	R0,(R3)
		//	MOVD	R0,8(R3)
		//	MOVD	R0,16(R3)
		//	MOVD	R0,24(R3)

		// one or more of these to clear remainder < 8 bytes
		//	MOVW	R0,n1(R3)
		//	MOVH	R0,n2(R3)
		//	MOVB	R0,n3(R3)
		{
			name:      "LoweredZero",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs:   []regMask{buildReg("R3")},
				clobbers: buildReg("R3"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
		},
		// Loop code:
		//	MOVD len/32,REG_TMP  only for loop
		//	MOVD REG_TMP,CTR     only for loop
		// loop:
		//	MOVD (R4),R7
		//	MOVD 8(R4),R8
		//	MOVD 16(R4),R9
		//	MOVD 24(R4),R10
		//	ADD  R4,$32          only with loop
		//	MOVD R7,(R3)
		//	MOVD R8,8(R3)
		//	MOVD R9,16(R3)
		//	MOVD R10,24(R3)
		//	ADD  R3,$32          only with loop
		//	BC 16,0,loop         only with loop
		// Bytes not moved by this loop are moved
		// with a combination of the following instructions,
		// starting with the largest sizes and generating as
		// many as needed, using the appropriate offset value.
		//	MOVD  n(R4),R7
		//	MOVD  R7,n(R3)
		//	MOVW  n1(R4),R7
		//	MOVW  R7,n1(R3)
		//	MOVH  n2(R4),R7
		//	MOVH  R7,n2(R3)
		//	MOVB  n3(R4),R7
		//	MOVB  R7,n3(R3)

		{
			name:      "LoweredMove",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R3"), buildReg("R4")},
				clobbers: buildReg("R3 R4 R7 R8 R9 R10"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
		},

		{name: "LoweredAtomicStore32", argLength: 3, reg: gpstore, typ: "Mem", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore64", argLength: 3, reg: gpstore, typ: "Mem", faultOnNilArg0: true, hasSideEffects: true},

		{name: "LoweredAtomicLoad32", argLength: 2, reg: gpload, typ: "UInt32", clobberFlags: true, faultOnNilArg0: true},
		{name: "LoweredAtomicLoad64", argLength: 2, reg: gpload, typ: "Int64", clobberFlags: true, faultOnNilArg0: true},
		{name: "LoweredAtomicLoadPtr", argLength: 2, reg: gpload, typ: "Int64", clobberFlags: true, faultOnNilArg0: true},

		// atomic add32, 64
		// SYNC
		// LDAR         (Rarg0), Rout
		// ADD		Rarg1, Rout
		// STDCCC       Rout, (Rarg0)
		// BNE          -3(PC)
		// ISYNC
		// return new sum

		{name: "LoweredAtomicAdd32", argLength: 3, reg: gpxchg, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAdd64", argLength: 3, reg: gpxchg, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange32, 64
		// SYNC
		// LDAR         (Rarg0), Rout
		// STDCCC       Rarg1, (Rarg0)
		// BNE          -2(PC)
		// ISYNC
		// return old val

		{name: "LoweredAtomicExchange32", argLength: 3, reg: gpxchg, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicExchange64", argLength: 3, reg: gpxchg, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic compare and swap.
		// arg0 = pointer, arg1 = old value, arg2 = new value, arg3 = memory. auxint must be zero.
		// if *arg0 == arg1 {
		//   *arg0 = arg2
		//   return (true, memory)
		// } else {
		//   return (false, memory)
		// }
		// SYNC
		// LDAR		(Rarg0), Rtmp
		// CMP		Rarg1, Rtmp
		// BNE		3(PC)
		// STDCCC	Rarg2, (Rarg0)
		// BNE		-4(PC)
		// CBNZ         Rtmp, -4(PC)
		// CSET         EQ, Rout
		{name: "LoweredAtomicCas64", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicCas32", argLength: 4, reg: gpcas, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic 8 and/or.
		// *arg0 &= (|=) arg1. arg2=mem. returns memory. auxint must be zero.
		// LBAR		(Rarg0), Rtmp
		// AND/OR	Rarg1, Rtmp
		// STBCCC	Rtmp, (Rarg0), Rtmp
		// BNE		Rtmp, -3(PC)

		{name: "LoweredAtomicAnd8", argLength: 3, reg: gpstore, asm: "AND", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicOr8", argLength: 3, reg: gpstore, asm: "OR", faultOnNilArg0: true, hasSideEffects: true},

		// (InvertFlags (CMP a b)) == (CMP b a)
		// So if we want (LessThan (CMP a b)) but we can't do that because a is a constant,
		// then we do (LessThan (InvertFlags (CMP b a))) instead.
		// Rewrites will convert this to (GreaterThan (CMP b a)).
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags", argLength: 1}, // reverse direction of arg0

		// Constant flag values. For any comparison, there are 3 possible
		// outcomes: either the three from the signed total order (<,==,>)
		// or the three from the unsigned total order, depending on which
		// comparison operation was used (CMP or CMPU -- PPC is different from
		// the other architectures, which have a single comparison producing
		// both signed and unsigned comparison results.)

		// These ops are for temporary use by rewrite rules. They
		// cannot appear in the generated assembly.
		{name: "FlagEQ"}, // equal
		{name: "FlagLT"}, // signed < or unsigned <
		{name: "FlagGT"}, // signed > or unsigned >

	}

	blocks := []blockData{
		{name: "EQ"},
		{name: "NE"},
		{name: "LT"},
		{name: "LE"},
		{name: "GT"},
		{name: "GE"},
		{name: "FLT"},
		{name: "FLE"},
		{name: "FGT"},
		{name: "FGE"},
	}

	archs = append(archs, arch{
		name:            "PPC64",
		pkg:             "cmd/internal/obj/ppc64",
		genfile:         "../../ppc64/ssa.go",
		ops:             ops,
		blocks:          blocks,
		regnames:        regNamesPPC64,
		gpregmask:       gp,
		fpregmask:       fp,
		framepointerreg: int8(num["SP"]),
		linkreg:         -1, // not used
	})
}
