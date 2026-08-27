// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"slices"
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
	covered := groupPredicationForms(insts)
	var defs []*unify.Value
	for _, inst := range insts {
		if covered[inst] {
			// The predicated half of a pair; it is emitted as an inVariant of its
			// unpredicated sibling so the operation has a single unifier value.
			continue
		}
		defs = append(defs, inst.emitAll()...)
	}
	return defs, nil
}

// groupPredicationForms pairs the predicated and unpredicated encodings of the
// same operation and folds them into one definition, mirroring how the AMD64
// loader treats an AVX-512 instruction's optional K-mask: the unpredicated form
// supplies the operation (and therefore the single front-end API), and the
// governing predicate becomes an inVariant that simdgen turns into predicated
// machine ops plus peepholes.
// However, different from AVX-512, where the predication mode is orthogonal to
// the operation as an instruction suffix, SVE's predication modes are separate
// instruction encodings, so the loader has to pair them up.
//
// A pair is only formed when both forms actually exist and their operand shapes
// correspond; the returned set names the predicated instructions that the pair
// covers, which the caller then skips. Everything else — an operation with only
// a predicated form (whose predicate stays implicit-all-true), or only an
// unpredicated one — is emitted unchanged.
func groupPredicationForms(insts []*Instruction) map[*Instruction]bool {
	type group struct{ unpred, pred []*Instruction }
	groups := map[string]*group{}
	for _, inst := range insts {
		key := inst.predicationGroupKey()
		if key == "" {
			continue
		}
		g := groups[key]
		if g == nil {
			g = &group{}
			groups[key] = g
		}
		if inst.predicationForm() == "unpredicated" {
			g.unpred = append(g.unpred, inst)
		} else {
			g.pred = append(g.pred, inst)
		}
	}

	covered := map[*Instruction]bool{}
	for _, g := range groups {
		if len(g.unpred) == 0 && len(g.pred) > 1 {
			groupPredicatedOnly(g.pred, covered)
			continue
		}
		if len(g.unpred) != 1 || len(g.pred) == 0 {
			// Not a clean pair (a form is missing, or the title is ambiguous);
			// leave both halves to be emitted as they are.
			continue
		}
		un := g.unpred[0]
		unOps := un.operands()
		var variants []predVariant
		for _, pr := range g.pred {
			prOps := pr.operands()
			if !sameOperandShape(unOps, prOps) {
				continue
			}
			var quals string
			for _, q := range predicationVariants(prOps) {
				quals += q
			}
			if quals == "" {
				continue
			}
			// Each encoding carries its own register symbols, so a machine op is
			// always generated from the shape of the encoding it comes from.
			outs, ins := splitRegNames(prOps)
			variants = append(variants, predVariant{quals: quals, outRegNames: outs, inRegNames: ins, predAsmPos: governingAsmPos(prOps), cpuFeature: pr.cpuFeature()})
			covered[pr] = true
		}
		if len(variants) == 0 {
			continue
		}
		un.predVariants = variants
	}
	return covered
}

// groupPredicatedOnly folds the encodings of an operation that has no
// unpredicated form at all — SVE writes ABS as "ABS <Zd>.<T>, <Pg>/M, <Zn>.<T>"
// and "ABS <Zd>.<T>, <Pg>/Z, <Zn>.<T>", and nothing else.
//
// There is no unpredicated encoding to carry the operation, so one of the
// predicated encodings does. Its governing predicate stays implicit-all-true, so
// the front-end API is still unpredicated, and every qualifier in the group —
// its own included — becomes an inVariant qualifier, which simdgen turns into
// one predicated machine op each for the peepholes to fold into.
//
// Only the merging encoding is used. Zeroing predication on these instructions
// is an Armv9.6-A extension -- ABS assembles to a different opcode under /Z, and
// baseline SVE hardware traps it -- while merging is available wherever SVE is.
// Nothing in what the XML parser exposes tells the two apart: both carry
// instr-class "sve", and the arch_variant element that does record the
// difference is not surfaced. So the zeroing encodings are dropped here rather
// than gated, to be folded in with the rest of SVE2.2 once simdgen can gate on
// the SVE sub-level.
//
// The group is folded only when the encodings are variations on one predication
// mode and nothing else: same operand shape, and one encoding per qualifier. Two
// encodings sharing a qualifier are two different instructions that happen to
// share a title (addressing modes of a load, say), and are left alone.
func groupPredicatedOnly(pred []*Instruction, covered map[*Instruction]bool) {
	byQual := map[string]*Instruction{}
	shape := pred[0].operands()
	for _, inst := range pred {
		ops := inst.operands()
		if !sameOperandShape(shape, ops) {
			return
		}
		quals := predicationVariants(ops)
		if len(quals) != 1 || quals[0] == "" {
			return
		}
		if _, dup := byQual[quals[0]]; dup {
			return
		}
		byQual[quals[0]] = inst
	}
	base, ok := byQual["M"]
	if !ok {
		return
	}
	baseOps := base.operands()
	outs, ins := splitRegNames(baseOps)
	base.predVariants = []predVariant{{quals: "M", outRegNames: outs, inRegNames: ins, predAsmPos: governingAsmPos(baseOps), cpuFeature: base.cpuFeature()}}
	for _, inst := range byQual {
		if inst != base {
			covered[inst] = true
		}
	}
}

// governingAsmPos returns the assembly position of the governing predicate in
// ops. Both callers work on encodings already classified as predicated, so a
// missing governing predicate is a broken invariant, not a case.
func governingAsmPos(ops []Operand) int {
	for i := range ops {
		if ops[i].governing {
			return ops[i].AsmPos
		}
	}
	panic("sve: predicated encoding has no governing predicate")
}

// splitRegNames returns an operand template's register symbols, results first
// and then the non-predicate inputs, in the order sameOperandShape compares
// them, so the two halves of a pair line up element by element.
func splitRegNames(ops []Operand) (outs, ins []string) {
	for i := range ops {
		if ops[i].governing {
			continue
		}
		if ops[i].role == "destination" {
			outs = append(outs, ops[i].regName)
		} else {
			ins = append(ins, ops[i].regName)
		}
	}
	return outs, ins
}

// sameOperandShape reports whether two operand templates describe the same
// operation apart from a governing predicate: same result and same sequence of
// non-predicate input classes. The predicated form of a destructive operation
// names its destination twice (once as the in-place source), which
// buildOperandList already turns into a regular input, so the two shapes line up.
func sameOperandShape(a, b []Operand) bool {
	split := func(ops []Operand) (outs, ins []string) {
		for i := range ops {
			if ops[i].governing {
				continue // the governing predicate is what differs
			}
			if ops[i].role == "destination" {
				outs = append(outs, ops[i].Class)
			} else {
				ins = append(ins, ops[i].Class)
			}
		}
		return outs, ins
	}
	ao, ai := split(a)
	bo, bi := split(b)
	return slices.Equal(ao, bo) && slices.Equal(ai, bi)
}
