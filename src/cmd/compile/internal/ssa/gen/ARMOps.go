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
//  - *const instructions may use a constant larger than the instuction can encode.
//    In this case the assembler expands to multiple instructions and uses tmp
//    register (R11).

// Suffixes encode the bit width of various instructions.
// W (word)      = 32 bit
// H (half word) = 16 bit
// HU            = 16 bit unsigned
// B (byte)      = 8 bit
// BU            = 8 bit unsigned

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
	"R10", // g
	"R11", // tmp
	"R12",
	"SP",  // aka R13
	"R14", // link
	"R15", // pc

	// pseudo-registers
	"FLAGS",
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
		gp         = buildReg("R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12")
		gpsp       = gp | buildReg("SP")
		gpspsb     = gpsp | buildReg("SB")
		flags      = buildReg("FLAGS")
		callerSave = gp
	)
	// Common regInfo
	var (
		gp01      = regInfo{inputs: []regMask{}, outputs: []regMask{gp}}
		gp11      = regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}
		gp11sb    = regInfo{inputs: []regMask{gpspsb}, outputs: []regMask{gp}}
		gp1flags  = regInfo{inputs: []regMask{gp}, outputs: []regMask{flags}}
		gp21      = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{gp}}
		gp2flags  = regInfo{inputs: []regMask{gp, gp}, outputs: []regMask{flags}}
		gpload    = regInfo{inputs: []regMask{gpspsb}, outputs: []regMask{gp}}
		gpstore   = regInfo{inputs: []regMask{gpspsb, gp}, outputs: []regMask{}}
		readflags = regInfo{inputs: []regMask{flags}, outputs: []regMask{gp}}
	)
	ops := []opData{
		// binary ops
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},    // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11sb, asm: "ADD", aux: "SymOff"}, // arg0 + auxInt + aux.(*gc.Sym)
		{name: "SUB", argLength: 2, reg: gp21, asm: "SUB"},                       // arg0 - arg1
		{name: "SUBconst", argLength: 1, reg: gp11, asm: "SUB", aux: "Int32"},    // arg0 - auxInt
		{name: "RSB", argLength: 2, reg: gp21, asm: "RSB"},                       // arg1 - arg0
		{name: "RSBconst", argLength: 1, reg: gp11, asm: "RSB", aux: "Int32"},    // auxInt - arg0

		{name: "AND", argLength: 2, reg: gp21, asm: "AND", commutative: true}, // arg0 & arg1
		{name: "ANDconst", argLength: 1, reg: gp11, asm: "AND", aux: "Int32"}, // arg0 & auxInt
		{name: "OR", argLength: 2, reg: gp21, asm: "ORR", commutative: true},  // arg0 | arg1
		{name: "ORconst", argLength: 1, reg: gp11, asm: "ORR", aux: "Int32"},  // arg0 | auxInt
		{name: "XOR", argLength: 2, reg: gp21, asm: "EOR", commutative: true}, // arg0 ^ arg1
		{name: "XORconst", argLength: 1, reg: gp11, asm: "EOR", aux: "Int32"}, // arg0 ^ auxInt
		{name: "BIC", argLength: 2, reg: gp21, asm: "BIC"},                    // arg0 &^ arg1
		{name: "BICconst", argLength: 1, reg: gp11, asm: "BIC", aux: "Int32"}, // arg0 &^ auxInt

		// unary ops
		{name: "MVN", argLength: 1, reg: gp11, asm: "MVN"}, // ^arg0

		{name: "CMP", argLength: 2, reg: gp2flags, asm: "CMP", typ: "Flags"},                    // arg0 compare to arg1
		{name: "CMPconst", argLength: 1, reg: gp1flags, asm: "CMP", aux: "Int32", typ: "Flags"}, // arg0 compare to auxInt
		{name: "CMN", argLength: 2, reg: gp2flags, asm: "CMN", typ: "Flags"},                    // arg0 compare to -arg1
		{name: "CMNconst", argLength: 1, reg: gp1flags, asm: "CMN", aux: "Int32", typ: "Flags"}, // arg0 compare to -auxInt
		{name: "TST", argLength: 2, reg: gp2flags, asm: "TST", typ: "Flags", commutative: true}, // arg0 & arg1 compare to 0
		{name: "TSTconst", argLength: 1, reg: gp1flags, asm: "TST", aux: "Int32", typ: "Flags"}, // arg0 & auxInt compare to 0
		{name: "TEQ", argLength: 2, reg: gp2flags, asm: "TEQ", typ: "Flags", commutative: true}, // arg0 ^ arg1 compare to 0
		{name: "TEQconst", argLength: 1, reg: gp1flags, asm: "TEQ", aux: "Int32", typ: "Flags"}, // arg0 ^ auxInt compare to 0

		{name: "MOVWconst", argLength: 0, reg: gp01, aux: "Int32", asm: "MOVW", rematerializeable: true}, // 32 low bits of auxint

		{name: "MOVBload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVB"},   // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVBUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVBU"}, // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVH"},   // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVHUload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVHU"}, // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVW"},   // load from arg0 + auxInt + aux.  arg1=mem.

		{name: "MOVBstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVB"}, // store 1 byte of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVHstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVH"}, // store 2 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.
		{name: "MOVWstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVW"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.

		{name: "MOVBreg", argLength: 1, reg: gp11, asm: "MOVBS"},  // move from arg0, sign-extended from byte
		{name: "MOVBUreg", argLength: 1, reg: gp11, asm: "MOVBU"}, // move from arg0, unsign-extended from byte
		{name: "MOVHreg", argLength: 1, reg: gp11, asm: "MOVHS"},  // move from arg0, sign-extended from half
		{name: "MOVHUreg", argLength: 1, reg: gp11, asm: "MOVHU"}, // move from arg0, unsign-extended from half

		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "SymOff"},                                // call static function aux.(*gc.Sym).  arg0=mem, auxint=argsize, returns mem
		{name: "CALLclosure", argLength: 3, reg: regInfo{[]regMask{gpsp, buildReg("R7"), 0}, callerSave, nil}, aux: "Int64"}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "CALLdefer", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "Int64"},                                  // call deferproc.  arg0=mem, auxint=argsize, returns mem
		{name: "CALLgo", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "Int64"},                                     // call newproc.  arg0=mem, auxint=argsize, returns mem
		{name: "CALLinter", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "Int64"},           // call fn by pointer.  arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		// pseudo-ops
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gpsp}, clobbers: flags}}, // panic if arg0 is nil.  arg1=mem.

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
		name:     "ARM",
		pkg:      "cmd/internal/obj/arm",
		genfile:  "../../arm/ssa.go",
		ops:      ops,
		blocks:   blocks,
		regnames: regNamesARM,
		flagmask: flags,
	})
}
