// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

func init() {
	var (
		gp01       = regInfo{inputs: []regMask{}, outputs: []regMask{31}}
		gp11       = regInfo{inputs: []regMask{31}, outputs: []regMask{31}}
		gp21       = regInfo{inputs: []regMask{31, 31}, outputs: []regMask{31}}
		gp2flags   = regInfo{inputs: []regMask{31, 31}, outputs: []regMask{32}}
		gpload     = regInfo{inputs: []regMask{31}, outputs: []regMask{31}}
		gpstore    = regInfo{inputs: []regMask{31, 31}, outputs: []regMask{}}
		flagsgp    = regInfo{inputs: []regMask{32}, outputs: []regMask{31}}
		callerSave = regMask(15)
	)
	ops := []opData{
		{name: "ADD", argLength: 2, reg: gp21, asm: "ADD", commutative: true},  // arg0 + arg1
		{name: "ADDconst", argLength: 1, reg: gp11, asm: "ADD", aux: "SymOff"}, // arg0 + auxInt + aux.(*gc.Sym)

		{name: "MOVWconst", argLength: 0, reg: gp01, aux: "Int32", asm: "MOVW", rematerializeable: true}, // 32 low bits of auxint

		{name: "CMP", argLength: 2, reg: gp2flags, asm: "CMP", typ: "Flags"}, // arg0 compare to arg1

		{name: "MOVWload", argLength: 2, reg: gpload, aux: "SymOff", asm: "MOVW"},   // load from arg0 + auxInt + aux.  arg1=mem.
		{name: "MOVWstore", argLength: 3, reg: gpstore, aux: "SymOff", asm: "MOVW"}, // store 4 bytes of arg1 to arg0 + auxInt + aux.  arg2=mem.

		{name: "CALLstatic", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "SymOff"}, // call static function aux.(*gc.Sym).  arg0=mem, auxint=argsize, returns mem

		// pseudo-ops
		{name: "LessThan", argLength: 1, reg: flagsgp}, // bool, 1 flags encode x<y 0 otherwise.
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

	regNames := []string{
		"R0",
		"R1",
		"R2",
		"R3",
		"SP",
		"FLAGS",
		"SB",
	}

	archs = append(archs, arch{
		name:     "ARM",
		pkg:      "cmd/internal/obj/arm",
		genfile:  "../../arm/ssa.go",
		ops:      ops,
		blocks:   blocks,
		regnames: regNames,
	})
}
