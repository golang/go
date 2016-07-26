// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import "strings"

var regNamesPPC64 = []string{
	// "R0", // REGZERO
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

	"CR",
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
		fp = buildReg("F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31")
		sp = buildReg("SP")
		sb = buildReg("SB")
		// gr = buildReg("g")
		cr = buildReg("CR")
		//ctr  = buildReg("CTR")
		//lr   = buildReg("LR")
		tmp  = buildReg("R31")
		ctxt = buildReg("R11")
		//		tls	= buildReg("R13")
		gp01        = regInfo{inputs: []regMask{}, outputs: []regMask{gp}}
		gp11        = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}
		gp21        = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{gp}}
		gp1cr       = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{cr}}
		gp2cr       = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{cr}}
		crgp        = regInfo{inputs: []regMask{cr}, outputs: []regMask{gp}}
		gpload      = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{gp}}
		gpstore     = regInfo{inputs: []regMask{gp | sp | sb, gp | sp | sb}, outputs: []regMask{}}
		gpstorezero = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{}} // ppc64.REGZERO is reserved zero value
		fp01        = regInfo{inputs: []regMask{}, outputs: []regMask{fp}}
		//		fp11	   = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fp21       = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp2cr      = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{cr}}
		fpload     = regInfo{inputs: []regMask{gp | sp | sb}, outputs: []regMask{fp}}
		fpstore    = regInfo{inputs: []regMask{gp | sp | sb, fp}, outputs: []regMask{}}
		callerSave = regMask(gp | fp | cr)
	)
	ops := []opData{
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},     // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11, asm: "ADD", aux: "SymOff"},    // arg0 + auxInt + aux.(*gc.Sym)
		{name: "FADD", argLength: 2, reg: fp21, asm: "FADD", commutative: true},   // arg0+arg1
		{name: "FADDS", argLength: 2, reg: fp21, asm: "FADDS", commutative: true}, // arg0+arg1
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB"},                        // arg0-arg1
		{name: "FSUB", argLength: 2, reg: fp21, asm: "FSUB"},                      // arg0-arg1
		{name: "FSUBS", argLength: 2, reg: fp21, asm: "FSUBS"},                    // arg0-arg1
		{name: "MULLD", argLength: 2, reg: gp21, asm: "MULLD", commutative: true}, // arg0*arg1
		{name: "MULLW", argLength: 2, reg: gp21, asm: "MULLW", commutative: true}, // arg0*arg1
		{name: "FMUL", argLength: 2, reg: fp21, asm: "FMUL", commutative: true},   // arg0*arg1
		{name: "FMULS", argLength: 2, reg: fp21, asm: "FMULS", commutative: true}, // arg0*arg1
		{name: "FDIV", argLength: 2, reg: fp21, asm: "FDIV"},                      // arg0/arg1
		{name: "FDIVS", argLength: 2, reg: fp21, asm: "FDIVS"},                    // arg0/arg1
		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true},     // arg0&arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int32"},     // arg0&arg1 ??
		{name: "OR", argLength: 2, reg: gp21, asm: "OR", commutative: true},       // arg0|arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "OR", aux: "Int32"},       // arg0|arg1 ??
		{name: "XOR", argLength: 2, reg: gp21, asm: "XOR", commutative: true},     // arg0^arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "XOR", aux: "Int32"},     // arg0|arg1 ??
		{name: "NEG", argLength: 1, reg: gp11, asm: "NEG"},                        // ^arg0

		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVB"},                     // sign extend int8 to int64
		{name: "MOVBZreg", argLength: 1, reg: gp11, asm: "MOVBZ"},                   // zero extend uint8 to uint64
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVH"},                     // sign extend int16 to int64
		{name: "MOVHZreg", argLength: 1, reg: gp11, asm: "MOVHZ"},                   // zero extend uint16 to uint64
		{name: "MOVWreg", argLength: 1, reg: gp11, asm: "MOVW"},                     // sign extend int32 to int64
		{name: "MOVWZreg", argLength: 1, reg: gp11, asm: "MOVWZ"},                   // zero extend uint32 to uint64
		{name: "MOVBload", argLength: 2, reg: gpload, asm: "MOVB", typ: "Int8"},     // sign extend int8 to int64
		{name: "MOVBZload", argLength: 2, reg: gpload, asm: "MOVBZ", typ: "UInt8"},  // zero extend uint8 to uint64
		{name: "MOVHload", argLength: 2, reg: gpload, asm: "MOVH", typ: "Int16"},    // sign extend int16 to int64
		{name: "MOVHZload", argLength: 2, reg: gpload, asm: "MOVHZ", typ: "UInt16"}, // zero extend uint16 to uint64
		{name: "MOVWload", argLength: 2, reg: gpload, asm: "MOVW", typ: "Int32"},    // sign extend int32 to int64
		{name: "MOVWZload", argLength: 2, reg: gpload, asm: "MOVWZ", typ: "UInt32"}, // zero extend uint32 to uint64

		{name: "MOVDload", argLength: 2, reg: gpload, asm: "MOVD", typ: "UInt64"},
		{name: "FMOVDload", argLength: 2, reg: fpload, asm: "FMOVD", typ: "Fload64"},
		{name: "FMOVSload", argLength: 2, reg: fpload, asm: "FMOVS", typ: "Float32"},
		{name: "MOVBstore", argLength: 3, reg: gpstore, asm: "MOVB", aux: "SymOff", typ: "Mem"},
		{name: "MOVHstore", argLength: 3, reg: gpstore, asm: "MOVH", aux: "SymOff", typ: "Mem"},
		{name: "MOVWstore", argLength: 3, reg: gpstore, asm: "MOVW", aux: "SymOff", typ: "Mem"},
		{name: "MOVDstore", argLength: 3, reg: gpstore, asm: "MOVD", aux: "SymOff", typ: "Mem"},
		{name: "FMOVDstore", argLength: 3, reg: fpstore, asm: "FMOVD", aux: "SymOff", typ: "Mem"},
		{name: "FMOVSstore", argLength: 3, reg: fpstore, asm: "FMOVS", aux: "SymOff", typ: "Mem"},

		{name: "MOVBstorezero", argLength: 2, reg: gpstorezero, asm: "MOVB", aux: "SymOff", typ: "Mem"}, // store zero byte to arg0+aux.  arg1=mem
		{name: "MOVHstorezero", argLength: 2, reg: gpstorezero, asm: "MOVH", aux: "SymOff", typ: "Mem"}, // store zero 2 bytes to ...
		{name: "MOVWstorezero", argLength: 2, reg: gpstorezero, asm: "MOVW", aux: "SymOff", typ: "Mem"}, // store zero 4 bytes to ...
		{name: "MOVDstorezero", argLength: 2, reg: gpstorezero, asm: "MOVD", aux: "SymOff", typ: "Mem"}, // store zero 8 bytes to ...

		{name: "MOVDaddr", argLength: 1, reg: regInfo{inputs: []regMask{sp | sb}, outputs: []regMask{gp}}, aux: "SymOff", asm: "MOVD", rematerializeable: true}, // arg0 + auxInt + aux.(*gc.Sym), arg0=SP/SB

		{name: "MOVDconst", argLength: 0, reg: gp01, aux: "Int64", asm: "MOVD", rematerializeable: true},     //
		{name: "MOVWconst", argLength: 0, reg: gp01, aux: "Int32", asm: "MOVW", rematerializeable: true},     // 32 low bits of auxint
		{name: "FMOVDconst", argLength: 0, reg: fp01, aux: "Float64", asm: "FMOVD", rematerializeable: true}, //
		{name: "FMOVSconst", argLength: 0, reg: fp01, aux: "Float32", asm: "FMOVS", rematerializeable: true}, //
		{name: "FCMPU", argLength: 2, reg: fp2cr, asm: "FCMPU", typ: "Flags"},

		{name: "CMP", argLength: 2, reg: gp2cr, asm: "CMP", typ: "Flags"},     // arg0 compare to arg1
		{name: "CMPU", argLength: 2, reg: gp2cr, asm: "CMPU", typ: "Flags"},   // arg0 compare to arg1
		{name: "CMPW", argLength: 2, reg: gp2cr, asm: "CMPW", typ: "Flags"},   // arg0 compare to arg1
		{name: "CMPWU", argLength: 2, reg: gp2cr, asm: "CMPWU", typ: "Flags"}, // arg0 compare to arg1
		{name: "CMPconst", argLength: 1, reg: gp1cr, asm: "CMP", aux: "Int32", typ: "Flags"},

		// pseudo-ops
		{name: "Equal", argLength: 1, reg: crgp},        // bool, true flags encode x==y false otherwise.
		{name: "NotEqual", argLength: 1, reg: crgp},     // bool, true flags encode x!=y false otherwise.
		{name: "LessThan", argLength: 1, reg: crgp},     // bool, true flags encode signed x<y false otherwise.
		{name: "LessEqual", argLength: 1, reg: crgp},    // bool, true flags encode signed x<=y false otherwise.
		{name: "GreaterThan", argLength: 1, reg: crgp},  // bool, true flags encode signed x>y false otherwise.
		{name: "GreaterEqual", argLength: 1, reg: crgp}, // bool, true flags encode signed x>=y false otherwise.

		// Scheduler ensures LoweredGetClosurePtr occurs only in entry block,
		// and sorts it to the very beginning of the block to prevent other
		// use of the closure pointer.
		{name: "LoweredGetClosurePtr", reg: regInfo{outputs: []regMask{ctxt}}},

		//arg0=ptr,arg1=mem, returns void.  Faults if ptr is nil.
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gp | sp | sb}, clobbers: cr | tmp}},

		// Convert pointer to integer, takes a memory operand for ordering.
		{name: "MOVDconvert", argLength: 2, reg: gp11, asm: "MOVD"},

		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "SymOff"},                                      // call static function aux.(*gc.Sym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{inputs: []regMask{gp | sp, ctxt, 0}, clobbers: callerSave}, aux: "Int64"}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLdefer", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "Int64"},                                        // call deferproc.  arg0=mem, auxint=argsize, returns mem
		{name: "CALLgo", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "Int64"},                                           // call newproc.  arg0=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "Int64"},                 // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

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
	})
}
