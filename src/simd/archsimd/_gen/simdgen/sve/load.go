// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"sort"

	"simd/archsimd/_gen/unify"

	"golang.org/x/arch/arm64/instgen/xmlspec"
)

// parseInstructions parses the ARM64 ISA XML files at path and returns the
// SVE / SVE2 instructions.
func parseInstructions(path string) ([]*Instruction, error) {
	xmlInsts := xmlspec.ParseXMLFiles(path)

	// One XML section can hold several iclasses with distinct mnemonics
	// (e.g. SUNPKHI + SUNPKLO), so expand to one logical instruction per iclass.
	var insts []*Instruction
	for _, xmlInst := range xmlInsts {
		if xmlInst == nil {
			continue
		}
		for i := range xmlInst.Instruction.Classes.Iclass {
			inst := &Instruction{
				Instruction: xmlInst.Instruction,
				iclass:      &xmlInst.Instruction.Classes.Iclass[i],
			}
			if inst.mnemonic() == "" || !inst.isSVE() {
				// TODO: handle more extensions?
				continue
			}
			insts = append(insts, inst)
		}
	}

	sort.Slice(insts, func(i, j int) bool {
		return insts[i].mnemonic() < insts[j].mnemonic()
	})
	return insts, nil
}

// Load parses the ARM64 ISA XML files at path and returns the SVE / SVE2
// instruction definitions as simdgen unify values.
func Load(path string) ([]*unify.Value, error) {
	insts, err := parseInstructions(path)
	if err != nil {
		return nil, err
	}
	var defs []*unify.Value
	for _, inst := range insts {
		defs = append(defs, inst.emitAll()...)
	}
	return defs, nil
}
