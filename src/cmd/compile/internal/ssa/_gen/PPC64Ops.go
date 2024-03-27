// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// Notes:
//  - Less-than-64-bit integer types live in the low portion of registers.
//    The upper portion is junk.
//  - Boolean types are zero or 1; stored in a byte, with upper bytes of the register containing junk.
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
	// "F31", the allocator is limited to 64 entries. We sacrifice this FPR to support XER.

	"XER",

	// If you add registers, update asyncPreempt in runtime.

	// "CR0",
	// "CR1",
	// "CR2",
	// "CR3",
	// "CR4",
	// "CR5",
	// "CR6",
	// "CR7",

	// "CR",
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
		gp  = buildReg("R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29")
		fp  = buildReg("F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30")
		sp  = buildReg("SP")
		sb  = buildReg("SB")
		gr  = buildReg("g")
		xer = buildReg("XER")
		// cr  = buildReg("CR")
		// ctr = buildReg("CTR")
		// lr  = buildReg("LR")
		tmp     = buildReg("R31")
		ctxt    = buildReg("R11")
		callptr = buildReg("R12")
		// tls = buildReg("R13")
		gp01        = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11        = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}
		xergp       = regInfo{inputs: []regMask{xer}, outputs: []regMask{gp}, clobbers: xer}
		gp11cxer    = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}, clobbers: xer}
		gp11xer     = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp, xer}}
		gp1xer1xer  = regInfo{inputs: []regMask{gp | sp | sb, xer}, outputs: []regMask{gp, xer}, clobbers: xer}
		gp21        = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{gp}}
		gp21a0      = regInfo{inputs: []regMask{gp, gp | sp | sb}, outputs: []regMask{gp}}
		gp21cxer    = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{gp}, clobbers: xer}
		gp21xer     = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{gp, xer}, clobbers: xer}
		gp2xer1xer  = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb, xer}, outputs: []regMask{gp, xer}, clobbers: xer}
		gp31        = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb, gp | sp | sb}, outputs: []regMask{gp}}
		gp1cr       = regInfo{inputs: []regMask{gp | sp | sb}}
		gp2cr       = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}}
		crgp        = regInfo{inputs: nil, outputs: []regMask{gp}}
		crgp11      = regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}
		crgp21      = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp}}
		gpload      = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}
		gploadidx   = regInfo{inputs: []regMask{gp | sp | sb, gp}, outputs: []regMask{gp}}
		prefreg     = regInfo{inputs: []regMask{gp | sp | sb}}
		gpstore     = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}}
		gpstoreidx  = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb, gp | sp | sb}}
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
		fploadidx   = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{fp}}
		fpstore     = regInfo{inputs: []regMask{gp | sp | sb, fp}}
		fpstoreidx  = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb, fp}}
		callerSave  = regMask(gp | fp | gr | xer)
		r3          = buildReg("R3")
		r4          = buildReg("R4")
		r5          = buildReg("R5")
		r6          = buildReg("R6")
	)
	ops := []opData{
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},                              // arg0 + arg1
		{name: "ADDCC", argLength: 2, reg: gp21, asm: "ADDCC", commutative: true, typ: "(Int,Flags)"},      // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11, asm: "ADD", aux: "Int64"},                              // arg0 + auxInt
		{name: "ADDCCconst", argLength: 1, reg: gp11cxer, asm: "ADDCCC", aux: "Int64", typ: "(Int,Flags)"}, // arg0 + auxInt sets CC, clobbers XER
		{name: "FADD", argLength: 2, reg: fp21, asm: "FADD", commutative: true},                            // arg0+arg1
		{name: "FADDS", argLength: 2, reg: fp21, asm: "FADDS", commutative: true},                          // arg0+arg1
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB"},                                                 // arg0-arg1
		{name: "SUBCC", argLength: 2, reg: gp21, asm: "SUBCC", typ: "(Int,Flags)"},                         // arg0-arg1 sets CC
		{name: "SUBFCconst", argLength: 1, reg: gp11cxer, asm: "SUBC", aux: "Int64"},                       // auxInt - arg0 (carry is ignored)
		{name: "FSUB", argLength: 2, reg: fp21, asm: "FSUB"},                                               // arg0-arg1
		{name: "FSUBS", argLength: 2, reg: fp21, asm: "FSUBS"},                                             // arg0-arg1

		// Note, the FPU works with float64 in register.
		{name: "XSMINJDP", argLength: 2, reg: fp21, asm: "XSMINJDP"}, // fmin(arg0,arg1)
		{name: "XSMAXJDP", argLength: 2, reg: fp21, asm: "XSMAXJDP"}, // fmax(arg0,arg1)

		{name: "MULLD", argLength: 2, reg: gp21, asm: "MULLD", typ: "Int64", commutative: true}, // arg0*arg1 (signed 64-bit)
		{name: "MULLW", argLength: 2, reg: gp21, asm: "MULLW", typ: "Int32", commutative: true}, // arg0*arg1 (signed 32-bit)
		{name: "MULLDconst", argLength: 1, reg: gp11, asm: "MULLD", aux: "Int32", typ: "Int64"}, // arg0*auxInt (signed 64-bit)
		{name: "MULLWconst", argLength: 1, reg: gp11, asm: "MULLW", aux: "Int32", typ: "Int64"}, // arg0*auxInt (signed 64-bit)
		{name: "MADDLD", argLength: 3, reg: gp31, asm: "MADDLD", typ: "Int64"},                  // (arg0*arg1)+arg2 (signed 64-bit)

		{name: "MULHD", argLength: 2, reg: gp21, asm: "MULHD", commutative: true},                             // (arg0 * arg1) >> 64, signed
		{name: "MULHW", argLength: 2, reg: gp21, asm: "MULHW", commutative: true},                             // (arg0 * arg1) >> 32, signed
		{name: "MULHDU", argLength: 2, reg: gp21, asm: "MULHDU", commutative: true},                           // (arg0 * arg1) >> 64, unsigned
		{name: "MULHDUCC", argLength: 2, reg: gp21, asm: "MULHDUCC", commutative: true, typ: "(Int64,Flags)"}, // (arg0 * arg1) >> 64, unsigned, sets CC
		{name: "MULHWU", argLength: 2, reg: gp21, asm: "MULHWU", commutative: true},                           // (arg0 * arg1) >> 32, unsigned

		{name: "FMUL", argLength: 2, reg: fp21, asm: "FMUL", commutative: true},   // arg0*arg1
		{name: "FMULS", argLength: 2, reg: fp21, asm: "FMULS", commutative: true}, // arg0*arg1

		{name: "FMADD", argLength: 3, reg: fp31, asm: "FMADD"},   // arg0*arg1 + arg2
		{name: "FMADDS", argLength: 3, reg: fp31, asm: "FMADDS"}, // arg0*arg1 + arg2
		{name: "FMSUB", argLength: 3, reg: fp31, asm: "FMSUB"},   // arg0*arg1 - arg2
		{name: "FMSUBS", argLength: 3, reg: fp31, asm: "FMSUBS"}, // arg0*arg1 - arg2

		{name: "SRAD", argLength: 2, reg: gp21cxer, asm: "SRAD"}, // signed arg0 >> (arg1&127), 64 bit width (note: 127, not 63!)
		{name: "SRAW", argLength: 2, reg: gp21cxer, asm: "SRAW"}, // signed arg0 >> (arg1&63), 32 bit width
		{name: "SRD", argLength: 2, reg: gp21, asm: "SRD"},       // unsigned arg0 >> (arg1&127), 64 bit width
		{name: "SRW", argLength: 2, reg: gp21, asm: "SRW"},       // unsigned arg0 >> (arg1&63), 32 bit width
		{name: "SLD", argLength: 2, reg: gp21, asm: "SLD"},       // arg0 << (arg1&127), 64 bit width
		{name: "SLW", argLength: 2, reg: gp21, asm: "SLW"},       // arg0 << (arg1&63), 32 bit width

		{name: "ROTL", argLength: 2, reg: gp21, asm: "ROTL"},   // arg0 rotate left by arg1 mod 64
		{name: "ROTLW", argLength: 2, reg: gp21, asm: "ROTLW"}, // uint32(arg0) rotate left by arg1 mod 32
		// The following are ops to implement the extended mnemonics for shifts as described in section C.8 of the ISA.
		// The constant shift values are packed into the aux int32.
		{name: "CLRLSLWI", argLength: 1, reg: gp11, asm: "CLRLSLWI", aux: "Int32"}, //
		{name: "CLRLSLDI", argLength: 1, reg: gp11, asm: "CLRLSLDI", aux: "Int32"}, //

		// Operations which consume or generate the CA (xer)
		{name: "ADDC", argLength: 2, reg: gp21xer, asm: "ADDC", commutative: true, typ: "(UInt64, UInt64)"},    // arg0 + arg1 -> out, CA
		{name: "SUBC", argLength: 2, reg: gp21xer, asm: "SUBC", typ: "(UInt64, UInt64)"},                       // arg0 - arg1 -> out, CA
		{name: "ADDCconst", argLength: 1, reg: gp11xer, asm: "ADDC", typ: "(UInt64, UInt64)", aux: "Int64"},    // arg0 + imm16 -> out, CA
		{name: "SUBCconst", argLength: 1, reg: gp11xer, asm: "SUBC", typ: "(UInt64, UInt64)", aux: "Int64"},    // imm16 - arg0 -> out, CA
		{name: "ADDE", argLength: 3, reg: gp2xer1xer, asm: "ADDE", typ: "(UInt64, UInt64)", commutative: true}, // arg0 + arg1 + CA (arg2) -> out, CA
		{name: "ADDZE", argLength: 2, reg: gp1xer1xer, asm: "ADDZE", typ: "(UInt64, UInt64)"},                  // arg0 + CA (arg1) -> out, CA
		{name: "SUBE", argLength: 3, reg: gp2xer1xer, asm: "SUBE", typ: "(UInt64, UInt64)"},                    // arg0 - arg1 - CA (arg2) -> out, CA
		{name: "ADDZEzero", argLength: 1, reg: xergp, asm: "ADDZE", typ: "UInt64"},                             // CA (arg0) + $0 -> out
		{name: "SUBZEzero", argLength: 1, reg: xergp, asm: "SUBZE", typ: "UInt64"},                             // $0 - CA (arg0) -> out

		{name: "SRADconst", argLength: 1, reg: gp11cxer, asm: "SRAD", aux: "Int64"}, // signed arg0 >> auxInt, 0 <= auxInt < 64, 64 bit width
		{name: "SRAWconst", argLength: 1, reg: gp11cxer, asm: "SRAW", aux: "Int64"}, // signed arg0 >> auxInt, 0 <= auxInt < 32, 32 bit width
		{name: "SRDconst", argLength: 1, reg: gp11, asm: "SRD", aux: "Int64"},       // unsigned arg0 >> auxInt, 0 <= auxInt < 64, 64 bit width
		{name: "SRWconst", argLength: 1, reg: gp11, asm: "SRW", aux: "Int64"},       // unsigned arg0 >> auxInt, 0 <= auxInt < 32, 32 bit width
		{name: "SLDconst", argLength: 1, reg: gp11, asm: "SLD", aux: "Int64"},       // arg0 << auxInt, 0 <= auxInt < 64, 64 bit width
		{name: "SLWconst", argLength: 1, reg: gp11, asm: "SLW", aux: "Int64"},       // arg0 << auxInt, 0 <= auxInt < 32, 32 bit width

		{name: "ROTLconst", argLength: 1, reg: gp11, asm: "ROTL", aux: "Int64"},   // arg0 rotate left by auxInt bits
		{name: "ROTLWconst", argLength: 1, reg: gp11, asm: "ROTLW", aux: "Int64"}, // uint32(arg0) rotate left by auxInt bits
		{name: "EXTSWSLconst", argLength: 1, reg: gp11, asm: "EXTSWSLI", aux: "Int64"},

		{name: "RLWINM", argLength: 1, reg: gp11, asm: "RLWNM", aux: "Int64"},                           // Rotate and mask by immediate "rlwinm". encodePPC64RotateMask describes aux
		{name: "RLWNM", argLength: 2, reg: gp21, asm: "RLWNM", aux: "Int64"},                            // Rotate and mask by "rlwnm". encodePPC64RotateMask describes aux
		{name: "RLWMI", argLength: 2, reg: gp21a0, asm: "RLWMI", aux: "Int64", resultInArg0: true},      // "rlwimi" similar aux encoding as above
		{name: "RLDICL", argLength: 1, reg: gp11, asm: "RLDICL", aux: "Int64"},                          // Auxint is encoded similarly to RLWINM, but only MB and SH are valid. ME is always 63.
		{name: "RLDICLCC", argLength: 1, reg: gp11, asm: "RLDICLCC", aux: "Int64", typ: "(Int, Flags)"}, // Auxint is encoded similarly to RLWINM, but only MB and SH are valid. ME is always 63. Sets CC.
		{name: "RLDICR", argLength: 1, reg: gp11, asm: "RLDICR", aux: "Int64"},                          // Likewise, but only ME and SH are valid. MB is always 0.

		{name: "CNTLZD", argLength: 1, reg: gp11, asm: "CNTLZD"},                          // count leading zeros
		{name: "CNTLZDCC", argLength: 1, reg: gp11, asm: "CNTLZDCC", typ: "(Int, Flags)"}, // count leading zeros, sets CC
		{name: "CNTLZW", argLength: 1, reg: gp11, asm: "CNTLZW"},                          // count leading zeros (32 bit)

		{name: "CNTTZD", argLength: 1, reg: gp11, asm: "CNTTZD"}, // count trailing zeros
		{name: "CNTTZW", argLength: 1, reg: gp11, asm: "CNTTZW"}, // count trailing zeros (32 bit)

		{name: "POPCNTD", argLength: 1, reg: gp11, asm: "POPCNTD"}, // number of set bits in arg0
		{name: "POPCNTW", argLength: 1, reg: gp11, asm: "POPCNTW"}, // number of set bits in each word of arg0 placed in corresponding word
		{name: "POPCNTB", argLength: 1, reg: gp11, asm: "POPCNTB"}, // number of set bits in each byte of arg0 placed in corresponding byte

		{name: "FDIV", argLength: 2, reg: fp21, asm: "FDIV"},   // arg0/arg1
		{name: "FDIVS", argLength: 2, reg: fp21, asm: "FDIVS"}, // arg0/arg1

		{name: "DIVD", argLength: 2, reg: gp21, asm: "DIVD", typ: "Int64"},   // arg0/arg1 (signed 64-bit)
		{name: "DIVW", argLength: 2, reg: gp21, asm: "DIVW", typ: "Int32"},   // arg0/arg1 (signed 32-bit)
		{name: "DIVDU", argLength: 2, reg: gp21, asm: "DIVDU", typ: "Int64"}, // arg0/arg1 (unsigned 64-bit)
		{name: "DIVWU", argLength: 2, reg: gp21, asm: "DIVWU", typ: "Int32"}, // arg0/arg1 (unsigned 32-bit)

		{name: "MODUD", argLength: 2, reg: gp21, asm: "MODUD", typ: "UInt64"}, // arg0 % arg1 (unsigned 64-bit)
		{name: "MODSD", argLength: 2, reg: gp21, asm: "MODSD", typ: "Int64"},  // arg0 % arg1 (signed 64-bit)
		{name: "MODUW", argLength: 2, reg: gp21, asm: "MODUW", typ: "UInt32"}, // arg0 % arg1 (unsigned 32-bit)
		{name: "MODSW", argLength: 2, reg: gp21, asm: "MODSW", typ: "Int32"},  // arg0 % arg1 (signed 32-bit)
		// MOD is implemented as rem := arg0 - (arg0/arg1) * arg1

		// Conversions are all float-to-float register operations.  "Integer" refers to encoding in the FP register.
		{name: "FCTIDZ", argLength: 1, reg: fp11, asm: "FCTIDZ", typ: "Float64"}, // convert float to 64-bit int round towards zero
		{name: "FCTIWZ", argLength: 1, reg: fp11, asm: "FCTIWZ", typ: "Float64"}, // convert float to 32-bit int round towards zero
		{name: "FCFID", argLength: 1, reg: fp11, asm: "FCFID", typ: "Float64"},   // convert 64-bit integer to float
		{name: "FCFIDS", argLength: 1, reg: fp11, asm: "FCFIDS", typ: "Float32"}, // convert 32-bit integer to float
		{name: "FRSP", argLength: 1, reg: fp11, asm: "FRSP", typ: "Float64"},     // round float to 32-bit value

		// Movement between float and integer registers with no change in bits; accomplished with stores+loads on PPC.
		// Because the 32-bit load-literal-bits instructions have impoverished addressability, always widen the
		// data instead and use FMOVDload and FMOVDstore instead (this will also dodge endianess issues).
		// There are optimizations that should apply -- (Xi2f64 (MOVWload (not-ADD-ptr+offset) ) ) could use
		// the word-load instructions.  (Xi2f64 (MOVDload ptr )) can be (FMOVDload ptr)

		{name: "MFVSRD", argLength: 1, reg: fpgp, asm: "MFVSRD", typ: "Int64"},   // move 64 bits of F register into G register
		{name: "MTVSRD", argLength: 1, reg: gpfp, asm: "MTVSRD", typ: "Float64"}, // move 64 bits of G register into F register

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true},                           // arg0&arg1
		{name: "ANDN", argLength: 2, reg: gp21, asm: "ANDN"},                                            // arg0&^arg1
		{name: "ANDNCC", argLength: 2, reg: gp21, asm: "ANDNCC", typ: "(Int64,Flags)"},                  // arg0&^arg1 sets CC
		{name: "ANDCC", argLength: 2, reg: gp21, asm: "ANDCC", commutative: true, typ: "(Int64,Flags)"}, // arg0&arg1 sets CC
		{name: "OR", argLength: 2, reg: gp21, asm: "OR", commutative: true},                             // arg0|arg1
		{name: "ORN", argLength: 2, reg: gp21, asm: "ORN"},                                              // arg0|^arg1
		{name: "ORCC", argLength: 2, reg: gp21, asm: "ORCC", commutative: true, typ: "(Int,Flags)"},     // arg0|arg1 sets CC
		{name: "NOR", argLength: 2, reg: gp21, asm: "NOR", commutative: true},                           // ^(arg0|arg1)
		{name: "NORCC", argLength: 2, reg: gp21, asm: "NORCC", commutative: true, typ: "(Int,Flags)"},   // ^(arg0|arg1) sets CC
		{name: "XOR", argLength: 2, reg: gp21, asm: "XOR", typ: "Int64", commutative: true},             // arg0^arg1
		{name: "XORCC", argLength: 2, reg: gp21, asm: "XORCC", commutative: true, typ: "(Int,Flags)"},   // arg0^arg1 sets CC
		{name: "EQV", argLength: 2, reg: gp21, asm: "EQV", typ: "Int64", commutative: true},             // arg0^^arg1
		{name: "NEG", argLength: 1, reg: gp11, asm: "NEG"},                                              // -arg0 (integer)
		{name: "NEGCC", argLength: 1, reg: gp11, asm: "NEGCC", typ: "(Int,Flags)"},                      // -arg0 (integer) sets CC
		{name: "BRD", argLength: 1, reg: gp11, asm: "BRD"},                                              // reversebytes64(arg0)
		{name: "BRW", argLength: 1, reg: gp11, asm: "BRW"},                                              // reversebytes32(arg0)
		{name: "BRH", argLength: 1, reg: gp11, asm: "BRH"},                                              // reversebytes16(arg0)
		{name: "FNEG", argLength: 1, reg: fp11, asm: "FNEG"},                                            // -arg0 (floating point)
		{name: "FSQRT", argLength: 1, reg: fp11, asm: "FSQRT"},                                          // sqrt(arg0) (floating point)
		{name: "FSQRTS", argLength: 1, reg: fp11, asm: "FSQRTS"},                                        // sqrt(arg0) (floating point, single precision)
		{name: "FFLOOR", argLength: 1, reg: fp11, asm: "FRIM"},                                          // floor(arg0), float64
		{name: "FCEIL", argLength: 1, reg: fp11, asm: "FRIP"},                                           // ceil(arg0), float64
		{name: "FTRUNC", argLength: 1, reg: fp11, asm: "FRIZ"},                                          // trunc(arg0), float64
		{name: "FROUND", argLength: 1, reg: fp11, asm: "FRIN"},                                          // round(arg0), float64
		{name: "FABS", argLength: 1, reg: fp11, asm: "FABS"},                                            // abs(arg0), float64
		{name: "FNABS", argLength: 1, reg: fp11, asm: "FNABS"},                                          // -abs(arg0), float64
		{name: "FCPSGN", argLength: 2, reg: fp21, asm: "FCPSGN"},                                        // copysign arg0 -> arg1, float64

		{name: "ORconst", argLength: 1, reg: gp11, asm: "OR", aux: "Int64"},                                                                                                 // arg0|aux
		{name: "XORconst", argLength: 1, reg: gp11, asm: "XOR", aux: "Int64"},                                                                                               // arg0^aux
		{name: "ANDCCconst", argLength: 1, reg: regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}, asm: "ANDCC", aux: "Int64", typ: "(Int,Flags)"},           // arg0&aux == 0 // and-immediate sets CC on PPC, always.
		{name: "ANDconst", argLength: 1, reg: regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}, clobberFlags: true, asm: "ANDCC", aux: "Int64", typ: "Int"}, // arg0&aux == 0 // and-immediate sets CC on PPC, always.

		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVB", typ: "Int64"},   // sign extend int8 to int64
		{name: "MOVBZreg", argLength: 1, reg: gp11, asm: "MOVBZ", typ: "Int64"}, // zero extend uint8 to uint64
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVH", typ: "Int64"},   // sign extend int16 to int64
		{name: "MOVHZreg", argLength: 1, reg: gp11, asm: "MOVHZ", typ: "Int64"}, // zero extend uint16 to uint64
		{name: "MOVWreg", argLength: 1, reg: gp11, asm: "MOVW", typ: "Int64"},   // sign extend int32 to int64
		{name: "MOVWZreg", argLength: 1, reg: gp11, asm: "MOVWZ", typ: "Int64"}, // zero extend uint32 to uint64

		// Load bytes in the endian order of the arch from arg0+aux+auxint into a 64 bit register.
		{name: "MOVBZload", argLength: 2, reg: gpload, asm: "MOVBZ", aux: "SymOff", typ: "UInt8", faultOnNilArg0: true, symEffect: "Read"},  // load byte zero extend
		{name: "MOVHload", argLength: 2, reg: gpload, asm: "MOVH", aux: "SymOff", typ: "Int16", faultOnNilArg0: true, symEffect: "Read"},    // load 2 bytes sign extend
		{name: "MOVHZload", argLength: 2, reg: gpload, asm: "MOVHZ", aux: "SymOff", typ: "UInt16", faultOnNilArg0: true, symEffect: "Read"}, // load 2 bytes zero extend
		{name: "MOVWload", argLength: 2, reg: gpload, asm: "MOVW", aux: "SymOff", typ: "Int32", faultOnNilArg0: true, symEffect: "Read"},    // load 4 bytes sign extend
		{name: "MOVWZload", argLength: 2, reg: gpload, asm: "MOVWZ", aux: "SymOff", typ: "UInt32", faultOnNilArg0: true, symEffect: "Read"}, // load 4 bytes zero extend
		{name: "MOVDload", argLength: 2, reg: gpload, asm: "MOVD", aux: "SymOff", typ: "Int64", faultOnNilArg0: true, symEffect: "Read"},    // load 8 bytes

		// Load bytes in reverse endian order of the arch from arg0 into a 64 bit register, all zero extend.
		// The generated instructions are indexed loads with no offset field in the instruction so the aux fields are not used.
		// In these cases the index register field is set to 0 and the full address is in the base register.
		{name: "MOVDBRload", argLength: 2, reg: gpload, asm: "MOVDBR", typ: "UInt64", faultOnNilArg0: true}, // load 8 bytes reverse order
		{name: "MOVWBRload", argLength: 2, reg: gpload, asm: "MOVWBR", typ: "UInt32", faultOnNilArg0: true}, // load 4 bytes zero extend reverse order
		{name: "MOVHBRload", argLength: 2, reg: gpload, asm: "MOVHBR", typ: "UInt16", faultOnNilArg0: true}, // load 2 bytes zero extend reverse order

		// In these cases an index register is used in addition to a base register
		// Loads from memory location arg[0] + arg[1].
		{name: "MOVBZloadidx", argLength: 3, reg: gploadidx, asm: "MOVBZ", typ: "UInt8"},  // zero extend uint8 to uint64
		{name: "MOVHloadidx", argLength: 3, reg: gploadidx, asm: "MOVH", typ: "Int16"},    // sign extend int16 to int64
		{name: "MOVHZloadidx", argLength: 3, reg: gploadidx, asm: "MOVHZ", typ: "UInt16"}, // zero extend uint16 to uint64
		{name: "MOVWloadidx", argLength: 3, reg: gploadidx, asm: "MOVW", typ: "Int32"},    // sign extend int32 to int64
		{name: "MOVWZloadidx", argLength: 3, reg: gploadidx, asm: "MOVWZ", typ: "UInt32"}, // zero extend uint32 to uint64
		{name: "MOVDloadidx", argLength: 3, reg: gploadidx, asm: "MOVD", typ: "Int64"},
		{name: "MOVHBRloadidx", argLength: 3, reg: gploadidx, asm: "MOVHBR", typ: "Int16"}, // sign extend int16 to int64
		{name: "MOVWBRloadidx", argLength: 3, reg: gploadidx, asm: "MOVWBR", typ: "Int32"}, // sign extend int32 to int64
		{name: "MOVDBRloadidx", argLength: 3, reg: gploadidx, asm: "MOVDBR", typ: "Int64"},
		{name: "FMOVDloadidx", argLength: 3, reg: fploadidx, asm: "FMOVD", typ: "Float64"},
		{name: "FMOVSloadidx", argLength: 3, reg: fploadidx, asm: "FMOVS", typ: "Float32"},

		// Prefetch instruction
		// Do prefetch of address generated with arg0 and arg1 with option aux. arg0=addr,arg1=memory, aux=option.
		{name: "DCBT", argLength: 2, aux: "Int64", reg: prefreg, asm: "DCBT", hasSideEffects: true},

		// Store bytes in the reverse endian order of the arch into arg0.
		// These are indexed stores with no offset field in the instruction so the auxint fields are not used.
		{name: "MOVDBRstore", argLength: 3, reg: gpstore, asm: "MOVDBR", typ: "Mem", faultOnNilArg0: true}, // store 8 bytes reverse order
		{name: "MOVWBRstore", argLength: 3, reg: gpstore, asm: "MOVWBR", typ: "Mem", faultOnNilArg0: true}, // store 4 bytes reverse order
		{name: "MOVHBRstore", argLength: 3, reg: gpstore, asm: "MOVHBR", typ: "Mem", faultOnNilArg0: true}, // store 2 bytes reverse order

		// Floating point loads from arg0+aux+auxint
		{name: "FMOVDload", argLength: 2, reg: fpload, asm: "FMOVD", aux: "SymOff", typ: "Float64", faultOnNilArg0: true, symEffect: "Read"}, // load double float
		{name: "FMOVSload", argLength: 2, reg: fpload, asm: "FMOVS", aux: "SymOff", typ: "Float32", faultOnNilArg0: true, symEffect: "Read"}, // load single float

		// Store bytes in the endian order of the arch into arg0+aux+auxint
		{name: "MOVBstore", argLength: 3, reg: gpstore, asm: "MOVB", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store byte
		{name: "MOVHstore", argLength: 3, reg: gpstore, asm: "MOVH", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 2 bytes
		{name: "MOVWstore", argLength: 3, reg: gpstore, asm: "MOVW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 4 bytes
		{name: "MOVDstore", argLength: 3, reg: gpstore, asm: "MOVD", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store 8 bytes

		// Store floating point value into arg0+aux+auxint
		{name: "FMOVDstore", argLength: 3, reg: fpstore, asm: "FMOVD", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store double flot
		{name: "FMOVSstore", argLength: 3, reg: fpstore, asm: "FMOVS", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store single float

		// Stores using index and base registers
		// Stores to arg[0] + arg[1]
		{name: "MOVBstoreidx", argLength: 4, reg: gpstoreidx, asm: "MOVB", typ: "Mem"},     // store bye
		{name: "MOVHstoreidx", argLength: 4, reg: gpstoreidx, asm: "MOVH", typ: "Mem"},     // store half word
		{name: "MOVWstoreidx", argLength: 4, reg: gpstoreidx, asm: "MOVW", typ: "Mem"},     // store word
		{name: "MOVDstoreidx", argLength: 4, reg: gpstoreidx, asm: "MOVD", typ: "Mem"},     // store double word
		{name: "FMOVDstoreidx", argLength: 4, reg: fpstoreidx, asm: "FMOVD", typ: "Mem"},   // store double float
		{name: "FMOVSstoreidx", argLength: 4, reg: fpstoreidx, asm: "FMOVS", typ: "Mem"},   // store single float
		{name: "MOVHBRstoreidx", argLength: 4, reg: gpstoreidx, asm: "MOVHBR", typ: "Mem"}, // store half word reversed byte using index reg
		{name: "MOVWBRstoreidx", argLength: 4, reg: gpstoreidx, asm: "MOVWBR", typ: "Mem"}, // store word reversed byte using index reg
		{name: "MOVDBRstoreidx", argLength: 4, reg: gpstoreidx, asm: "MOVDBR", typ: "Mem"}, // store double word reversed byte using index reg

		// The following ops store 0 into arg0+aux+auxint arg1=mem
		{name: "MOVBstorezero", argLength: 2, reg: gpstorezero, asm: "MOVB", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero 1 byte
		{name: "MOVHstorezero", argLength: 2, reg: gpstorezero, asm: "MOVH", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero 2 bytes
		{name: "MOVWstorezero", argLength: 2, reg: gpstorezero, asm: "MOVW", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero 4 bytes
		{name: "MOVDstorezero", argLength: 2, reg: gpstorezero, asm: "MOVD", aux: "SymOff", typ: "Mem", faultOnNilArg0: true, symEffect: "Write"}, // store zero 8 bytes

		{name: "MOVDaddr", argLength: 1, reg: regInfo{inputs: []regMask{sp | sb | gp}, outputs: []regMask{gp}}, aux: "SymOff", asm: "MOVD", rematerializeable: true, symEffect: "Addr"}, // arg0 + auxInt + aux.(*gc.Sym), arg0=SP/SB/GP

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

		// ISEL  arg2 ? arg0 : arg1
		// ISELZ arg1 ? arg0 : $0
		// auxInt values 0=LT 1=GT 2=EQ 3=SO (summary overflow/unordered) 4=GE 5=LE 6=NE 7=NSO (not summary overflow/not unordered)
		// Note, auxInt^4 inverts the comparison condition. For example, LT^4 becomes GE, and "ISEL [a] x y z" is equivalent to ISEL [a^4] y x z".
		{name: "ISEL", argLength: 3, reg: crgp21, asm: "ISEL", aux: "Int32", typ: "Int32"},
		{name: "ISELZ", argLength: 2, reg: crgp11, asm: "ISEL", aux: "Int32"},

		// SETBC auxInt values 0=LT 1=GT 2=EQ     (CRbit=1)? 1 : 0
		{name: "SETBC", argLength: 1, reg: crgp, asm: "SETBC", aux: "Int32", typ: "Int32"},
		// SETBCR auxInt values 0=LT 1=GT 2=EQ     (CRbit=1)? 0 : 1
		{name: "SETBCR", argLength: 1, reg: crgp, asm: "SETBCR", aux: "Int32", typ: "Int32"},

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
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{ctxt}}, zeroWidth: true},

		// LoweredGetCallerSP returns the SP of the caller of the current function. arg0=mem.
		{name: "LoweredGetCallerSP", argLength: 1, reg: gp01, rematerializeable: true},

		// LoweredGetCallerPC evaluates to the PC to which its "caller" will return.
		// I.e., if f calls g "calls" getcallerpc,
		// the result should be the PC within f that g will return to.
		// See runtime/stubs.go for a more detailed discussion.
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},

		//arg0=ptr,arg1=mem, returns void.  Faults if ptr is nil.
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gp | sp | sb}, clobbers: tmp}, clobberFlags: true, nilCheck: true, faultOnNilArg0: true},
		// Round ops to block fused-multiply-add extraction.
		{name: "LoweredRound32F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},
		{name: "LoweredRound64F", argLength: 1, reg: fp11, resultInArg0: true, zeroWidth: true},

		{name: "CALLstatic", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},                                       // call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLtail", argLength: -1, reg: regInfo{clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true, tailCall: true},                         // tail call static function aux.(*obj.LSym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: -1, reg: regInfo{inputs: []regMask{callptr, ctxt, 0}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: -1, reg: regInfo{inputs: []regMask{callptr}, clobbers: callerSave}, aux: "CallOff", clobberFlags: true, call: true},            // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		// large or unaligned zeroing
		// arg0 = address of memory to zero (in R3, changed as side effect)
		// returns mem
		//
		// a loop is generated when there is more than one iteration
		// needed to clear 4 doublewords
		//
		//	XXLXOR	VS32,VS32,VS32
		// 	MOVD	$len/32,R31
		//	MOVD	R31,CTR
		//	MOVD	$16,R31
		//	loop:
		//	STXVD2X VS32,(R0)(R3)
		//	STXVD2X	VS32,(R31)(R3)
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
				inputs:   []regMask{buildReg("R20")},
				clobbers: buildReg("R20"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
			unsafePoint:    true,
		},
		{
			name:      "LoweredZeroShort",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs: []regMask{gp}},
			typ:            "Mem",
			faultOnNilArg0: true,
			unsafePoint:    true,
		},
		{
			name:      "LoweredQuadZeroShort",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs: []regMask{gp},
			},
			typ:            "Mem",
			faultOnNilArg0: true,
			unsafePoint:    true,
		},
		{
			name:      "LoweredQuadZero",
			aux:       "Int64",
			argLength: 2,
			reg: regInfo{
				inputs:   []regMask{buildReg("R20")},
				clobbers: buildReg("R20"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
			unsafePoint:    true,
		},

		// R31 is temp register
		// Loop code:
		//	MOVD len/32,R31		set up loop ctr
		//	MOVD R31,CTR
		//	MOVD $16,R31		index register
		// loop:
		//	LXVD2X (R0)(R4),VS32
		//	LXVD2X (R31)(R4),VS33
		//	ADD  R4,$32          increment src
		//	STXVD2X VS32,(R0)(R3)
		//	STXVD2X VS33,(R31)(R3)
		//	ADD  R3,$32          increment dst
		//	BC 16,0,loop         branch ctr
		// For this purpose, VS32 and VS33 are treated as
		// scratch registers. Since regalloc does not
		// track vector registers, even if it could be marked
		// as clobbered it would have no effect.
		// TODO: If vector registers are managed by regalloc
		// mark these as clobbered.
		//
		// Bytes not moved by this loop are moved
		// with a combination of the following instructions,
		// starting with the largest sizes and generating as
		// many as needed, using the appropriate offset value.
		//	MOVD  n(R4),R14
		//	MOVD  R14,n(R3)
		//	MOVW  n1(R4),R14
		//	MOVW  R14,n1(R3)
		//	MOVH  n2(R4),R14
		//	MOVH  R14,n2(R3)
		//	MOVB  n3(R4),R14
		//	MOVB  R14,n3(R3)

		{
			name:      "LoweredMove",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R20"), buildReg("R21")},
				clobbers: buildReg("R20 R21"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
			unsafePoint:    true,
		},
		{
			name:      "LoweredMoveShort",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs: []regMask{gp, gp},
			},
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
			unsafePoint:    true,
		},

		// The following is similar to the LoweredMove, but uses
		// LXV instead of LXVD2X, which does not require an index
		// register and will do 4 in a loop instead of only.
		{
			name:      "LoweredQuadMove",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs:   []regMask{buildReg("R20"), buildReg("R21")},
				clobbers: buildReg("R20 R21"),
			},
			clobberFlags:   true,
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
			unsafePoint:    true,
		},

		{
			name:      "LoweredQuadMoveShort",
			aux:       "Int64",
			argLength: 3,
			reg: regInfo{
				inputs: []regMask{gp, gp},
			},
			typ:            "Mem",
			faultOnNilArg0: true,
			faultOnNilArg1: true,
			unsafePoint:    true,
		},

		{name: "LoweredAtomicStore8", argLength: 3, reg: gpstore, typ: "Mem", aux: "Int64", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore32", argLength: 3, reg: gpstore, typ: "Mem", aux: "Int64", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicStore64", argLength: 3, reg: gpstore, typ: "Mem", aux: "Int64", faultOnNilArg0: true, hasSideEffects: true},

		{name: "LoweredAtomicLoad8", argLength: 2, reg: gpload, typ: "UInt8", aux: "Int64", clobberFlags: true, faultOnNilArg0: true},
		{name: "LoweredAtomicLoad32", argLength: 2, reg: gpload, typ: "UInt32", aux: "Int64", clobberFlags: true, faultOnNilArg0: true},
		{name: "LoweredAtomicLoad64", argLength: 2, reg: gpload, typ: "Int64", aux: "Int64", clobberFlags: true, faultOnNilArg0: true},
		{name: "LoweredAtomicLoadPtr", argLength: 2, reg: gpload, typ: "Int64", aux: "Int64", clobberFlags: true, faultOnNilArg0: true},

		// atomic add32, 64
		// LWSYNC
		// LDAR         (Rarg0), Rout
		// ADD		Rarg1, Rout
		// STDCCC       Rout, (Rarg0)
		// BNE          -3(PC)
		// return new sum
		{name: "LoweredAtomicAdd32", argLength: 3, reg: gpxchg, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAdd64", argLength: 3, reg: gpxchg, resultNotInArgs: true, clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic exchange32, 64
		// LWSYNC
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
		{name: "LoweredAtomicCas64", argLength: 4, reg: gpcas, resultNotInArgs: true, aux: "Int64", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicCas32", argLength: 4, reg: gpcas, resultNotInArgs: true, aux: "Int64", clobberFlags: true, faultOnNilArg0: true, hasSideEffects: true},

		// atomic 8/32 and/or.
		// *arg0 &= (|=) arg1. arg2=mem. returns memory. auxint must be zero.
		// LBAR/LWAT	(Rarg0), Rtmp
		// AND/OR	Rarg1, Rtmp
		// STBCCC/STWCCC Rtmp, (Rarg0), Rtmp
		// BNE		Rtmp, -3(PC)
		{name: "LoweredAtomicAnd8", argLength: 3, reg: gpstore, asm: "AND", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicAnd32", argLength: 3, reg: gpstore, asm: "AND", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicOr8", argLength: 3, reg: gpstore, asm: "OR", faultOnNilArg0: true, hasSideEffects: true},
		{name: "LoweredAtomicOr32", argLength: 3, reg: gpstore, asm: "OR", faultOnNilArg0: true, hasSideEffects: true},

		// LoweredWB invokes runtime.gcWriteBarrier. arg0=mem, auxint=# of buffer entries needed
		// It preserves R0 through R17 (except special registers R1, R2, R11, R12, R13), g, and R20 and R21,
		// but may clobber anything else, including R31 (REGTMP).
		// Returns a pointer to a write barrier buffer in R29.
		{name: "LoweredWB", argLength: 1, reg: regInfo{clobbers: (callerSave &^ buildReg("R0 R3 R4 R5 R6 R7 R8 R9 R10 R14 R15 R16 R17 R20 R21 g")) | buildReg("R31"), outputs: []regMask{buildReg("R29")}}, clobberFlags: true, aux: "Int64"},

		{name: "LoweredPubBarrier", argLength: 1, asm: "LWSYNC", hasSideEffects: true}, // Do data barrier. arg0=memory
		// There are three of these functions so that they can have three different register inputs.
		// When we check 0 <= c <= cap (A), then 0 <= b <= c (B), then 0 <= a <= b (C), we want the
		// default registers to match so we don't need to copy registers around unnecessarily.
		{name: "LoweredPanicBoundsA", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r5, r6}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsB", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r4, r5}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).
		{name: "LoweredPanicBoundsC", argLength: 3, aux: "Int64", reg: regInfo{inputs: []regMask{r3, r4}}, typ: "Mem", call: true}, // arg0=idx, arg1=len, arg2=mem, returns memory. AuxInt contains report code (see PanicBounds in genericOps.go).

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
		{name: "EQ", controls: 1},
		{name: "NE", controls: 1},
		{name: "LT", controls: 1},
		{name: "LE", controls: 1},
		{name: "GT", controls: 1},
		{name: "GE", controls: 1},
		{name: "FLT", controls: 1},
		{name: "FLE", controls: 1},
		{name: "FGT", controls: 1},
		{name: "FGE", controls: 1},
	}

	archs = append(archs, arch{
		name:               "PPC64",
		pkg:                "cmd/internal/obj/ppc64",
		genfile:            "../../ppc64/ssa.go",
		ops:                ops,
		blocks:             blocks,
		regnames:           regNamesPPC64,
		ParamIntRegNames:   "R3 R4 R5 R6 R7 R8 R9 R10 R14 R15 R16 R17",
		ParamFloatRegNames: "F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12",
		gpregmask:          gp,
		fpregmask:          fp,
		specialregmask:     xer,
		framepointerreg:    -1,
		linkreg:            -1, // not used
	})
}
