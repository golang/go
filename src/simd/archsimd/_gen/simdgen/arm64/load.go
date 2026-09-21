// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE: This currently only supports the advsimd (NEON) instruction class.

package arm64

import (
	"sort"

	"simd/archsimd/_gen/unify"

	"golang.org/x/arch/arm64/instgen/xmlspec"
)

// ParseInstructions loads and parses ARM64 instruction definitions from XML files at given path.
func ParseInstructions(path string) ([]*Instruction, error) {
	xmlInsts := xmlspec.ParseXMLFiles(path)

	var instructions []*Instruction
	for _, xmlInst := range xmlInsts {
		if xmlInst == nil {
			continue
		}
		inst := &Instruction{Instruction: xmlInst.Instruction}
		if inst.Mnemonic() == "" {
			continue
		}
		if inst.InstrClass() != "advsimd" {
			continue
		}
		instructions = append(instructions, inst)
	}

	sort.Slice(instructions, func(i, j int) bool {
		return instructions[i].Mnemonic() < instructions[j].Mnemonic()
	})

	return instructions, nil
}

// Load loads ARM64 instruction definitions from XML files at given path and returns them as unify values.
func Load(path string) ([]*unify.Value, error) {
	instructions, err := ParseInstructions(path)
	if err != nil {
		return nil, err
	}
	var defs []*unify.Value
	for _, instruction := range instructions {
		defs = append(defs, instruction.EmitAll()...)
	}
	return defs, nil
}
