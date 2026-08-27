// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"cmp"
	"fmt"
	"log"
	"slices"
	"strings"

	"simd/archsimd/_gen/unify"
)

// asComment wraps text into // comment lines of at most width columns.
func asComment(text string, width int) string {
	text = strings.TrimSpace(text)
	text = strings.ReplaceAll(text, "&amp;", "&")
	text = strings.ReplaceAll(text, "\n", " ")
	words := strings.Fields(text)
	var lines []string
	line := ""
	for _, w := range words {
		if line != "" {
			line += " "
		}
		line += w
		if len(line) >= width {
			lines = append(lines, "// "+line)
			line = ""
		}
	}
	if line != "" {
		lines = append(lines, "// "+line)
	}
	return strings.Join(lines, "\n")
}

// mixedWidthLogged dedupes the mixed-element-width warning by mnemonic, so a
// conversion family with many encodings logs once per generate run.
var mixedWidthLogged = map[string]bool{}

// emit renders an operand as a unify value. Z-vectors and predicates are
// scalable (a base type and per-operand element width, no fixed bits/lanes);
// mem, immediate and special operands are opaque (class and position only).
func (op *Operand) emit() *unify.Value {
	var db unify.DefBuilder
	db.Add("class", unify.NewValue(unify.NewStringExact(op.Class)))
	if op.BaseType != "" {
		db.Add("base", unify.NewValue(unify.NewStringExact(op.BaseType)))
	}
	switch {
	case op.Bits > 0:
		// A fixed-width SIMD&FP scalar (OperandVFP): a real bit width and lanes.
		db.Add("bits", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.Bits))))
		if op.Lanes > 0 {
			db.Add("lanes", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.Lanes))))
		}
	case op.Class == "vreg" || op.Class == "mask":
		// SVE vectors and predicates are scalable: no fixed total bit width.
		// The literal "scalable" both marks that and, because it conflicts with
		// any numeric bits, keeps these operands from unifying with the
		// fixed-width (NEON/AVX) types that share types.yaml.
		db.Add("bits", unify.NewValue(unify.NewStringExact("scalable")))
	}
	if op.ElemBits > 0 {
		db.Add("elemBits", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.ElemBits))))
	}
	if op.Predication != "" {
		// "M" (merging) or "Z" (zeroing) for a governing predicate. Some SVE
		// instructions support only one; this records which.
		db.Add("predication", unify.NewValue(unify.NewStringExact(op.Predication)))
	}
	if op.governing {
		// This operand is a governing predicate.
		db.Add("governing", unify.NewValue(unify.NewStringExact("true")))
	}
	if op.isList {
		// This register came from a single-register list ("{ <Zt>.<T> }"), a
		// distinct assembler encoding from a bare register.
		db.Add("listNumber", unify.NewValue(unify.NewStringExact("0")))
	}
	if op.regName != "" {
		// The assembly template's register symbol, e.g. "Zdn", "Zn", "Pg".
		db.Add("regName", unify.NewValue(unify.NewStringExact(op.regName)))
	}
	// The symbol this operand has in each predicated encoding, indexed to
	// match the def's inVariant. The symbols can differ from the unpredicated
	// ones to predicated ones:
	// ADD <Zd>, <Zn>, <Zm> unpredicated
	// ADD <Zdn>, <Pg>/M, <Zdn>, <Zm> predicated
	//
	// [groupPredicationForms] folds the two into one def.
	// simdgen needs these symbols to recognize resultInArg0.
	names := make([]*unify.Value, len(op.predRegName))
	for i, n := range op.predRegName {
		names[i] = unify.NewValue(unify.NewStringExact(n))
	}
	db.Add("predRegName", unify.NewValue(unify.NewTuple(names...)))
	db.Add("asmPos", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.AsmPos))))
	return unify.NewValue(db.Build())
}

// pickRegNames returns operand idx's symbol in each predicated encoding, in
// variant order. The encodings passed [sameOperandShape], so idx addresses the
// matching operand in every one of them.
func pickRegNames(variants []predVariant, idx int, sel func(predVariant) []string) []string {
	if len(variants) == 0 {
		return nil
	}
	out := make([]string, len(variants))
	for i, pv := range variants {
		names := sel(pv)
		if idx >= len(names) {
			panic(fmt.Sprintf("operand %d has no counterpart in predicated encoding %d", idx, i))
		}
		out[i] = names[idx]
	}
	return out
}

// emitOne emits a single instruction def from a fully-instantiated operand list:
// the destination is the output, every other operand (including a governing
// predicate) is a literal input.
//
// An SVE predicate is a mandatory input, not an optional AVX-512-style K-mask, so
// it goes in `in`; inVariant is emitted empty just to satisfy the types.yaml schema.
func (inst *Instruction) emitOne(asm string, ops []Operand, widthAgnostic bool) *unify.Value {
	var db unify.DefBuilder
	db.Add("asm", unify.NewValue(unify.NewStringExact(asm)))
	db.Add("goarch", unify.NewValue(unify.NewStringExact("arm64")))
	// The operation's feature level is the floor across its encodings: an
	// operation whose predicated sibling is baseline SVE is available on SVE
	// even when its unpredicated carrier needs SVE2 — the carrier is then a
	// feature-gated upgrade, recorded as unpredCpuFeature for the rules.
	feature := inst.cpuFeature()
	unpred := ""
	for _, pv := range inst.predVariants {
		if pv.cpuFeature == "SVE" && feature == "SVE2" {
			unpred = feature
			feature = pv.cpuFeature
		}
	}
	db.Add("cpuFeature", unify.NewValue(unify.NewStringExact(feature)))
	if unpred != "" {
		db.Add("unpredCPUFeature", unify.NewValue(unify.NewStringExact(unpred)))
	}
	if doc := inst.documentation(); doc != "" {
		db.Add("details", unify.NewValue(unify.NewStringExact(asComment(doc, 80))))
	}
	if widthAgnostic {
		db.Add("widthAgnostic", unify.NewValue(unify.NewStringExact("true")))
	}

	// One def can describe several encodings of one operation, grouped by
	// [groupPredicationForms] or [groupPredicatedOnly], so each operand also
	// carries the symbol it has in each predicated encoding. The symbols are
	// matched up in template order, so they must be attached before the sort
	// below reorders the inputs.
	var inOps, outOps []Operand
	var outIdx, inIdx int
	for _, op := range ops {
		switch {
		case op.governing:
			// The governing predicate is the operand the paired encodings differ in, so
			// it is not one of the symbols they are matched up by.
			inOps = append(inOps, op)
		case op.role == "destination":
			op.predRegName = pickRegNames(inst.predVariants, outIdx, func(pv predVariant) []string { return pv.outRegNames })
			outIdx++
			outOps = append(outOps, op)
		default:
			op.predRegName = pickRegNames(inst.predVariants, inIdx, func(pv predVariant) []string { return pv.inRegNames })
			inIdx++
			inOps = append(inOps, op)
		}
	}
	priority := map[string]int{"immediate": 0, "vreg": 1, "greg": 1, "memory": 1, "mask": 2}
	slices.SortStableFunc(inOps, func(a, b Operand) int {
		pa := priority[a.Class]
		pb := priority[b.Class]
		if pa != pb {
			return cmp.Compare(pa, pb)
		}
		return cmp.Compare(a.AsmPos, b.AsmPos)
	})

	var ins, outs []*unify.Value
	for i := range inOps {
		ins = append(ins, inOps[i].emit())
	}
	for i := range outOps {
		outs = append(outs, outOps[i].emit())
	}
	db.Add("in", unify.NewValue(unify.NewTuple(ins...)))
	var inVar []*unify.Value
	for _, pv := range inst.predVariants {
		// The governing predicate of the paired predicated encoding.
		var pdb unify.DefBuilder
		pdb.Add("class", unify.NewValue(unify.NewStringExact("mask")))
		pdb.Add("bits", unify.NewValue(unify.NewStringExact("scalable")))
		pdb.Add("predication", unify.NewValue(unify.NewStringExact(pv.quals)))
		pdb.Add("asmPos", unify.NewValue(unify.NewStringExact(fmt.Sprint(pv.predAsmPos))))
		inVar = append(inVar, unify.NewValue(pdb.Build()))
	}
	db.Add("inVariant", unify.NewValue(unify.NewTuple(inVar...)))
	db.Add("out", unify.NewValue(unify.NewTuple(outs...)))
	return unify.NewValue(db.Build())
}

// emitAll emits the unify defs for this instruction — the concrete variants of
// the source template. See classify (used by both emitAll and analyze) for the
// full disposition.
func (inst *Instruction) emitAll() []*unify.Value {
	// emitAll doesn't check the anomalies, that would be done by
	// a full-corpus test in analyze_test.go.
	defs, _, _ := inst.classify()
	return defs
}

// lookup returns the element width for the given size key in a table.
func lookup(rows []arngRow, size string) (int, bool) {
	for _, r := range rows {
		if r.size == size {
			return r.bits, true
		}
	}
	return 0, false
}

// emitVariants emits one def per (integer signedness × arrangement row ×
// predication). Each operand's element width comes from its own arrangement
// symbol's table, keyed by the shared size field, so uniform and non-uniform
// (widening/narrowing) forms are handled the same way; operands with no
// arrangement stay unsized. Each operand's base type is resolved per operand
// (laneIsFloat) — floating-point lanes are always "float", integer lanes take
// the signedness of the current variant — so this naturally extends to
// conversions, whose lanes will differ.
func (inst *Instruction) emitVariants(template []Operand) []*unify.Value {
	asm := inst.goOpPrefix() + inst.mnemonic()

	links := arngLinks(template)
	tables := map[string][]arngRow{}
	for _, l := range links {
		tables[l] = inst.resolveArrangementTable(l)
	}

	// Rows to iterate: the primary (destination-first) symbol's size keys, or a
	// single pass when there is no variable arrangement.
	var sizes []string
	if len(links) > 0 {
		for _, r := range tables[links[0]] {
			sizes = append(sizes, r.size)
		}
	} else {
		sizes = []string{""}
	}

	signs := inst.integerSignedness(template)

	// Governing-predicate qualifier(s) for this template: /M, /Z, both (a /<ZM>
	// encoding), or a single no-op pass when there is no governing predicate.
	preds := predicationVariants(template)

	// A bitwise operation with no variable arrangement is width-agnostic: the
	// encoding is written .D, but any element view of it computes the same
	// bits, and its predicated sibling is a per-<T> encoding. Emit a def per
	// element width so every Go type gets the API, marked so that simdgen
	// collapses the unpredicated machine op back to the single .D instruction.
	widths := []int{0}
	widthAgnostic := len(links) == 0 && inst.bitwise()
	if widthAgnostic {
		widths = []int{8, 16, 32, 64}
	}

	var defs []*unify.Value
	for _, sign := range signs {
		for _, size := range sizes {
			ops := make([]Operand, len(template))
			copy(ops, template)
			skip := false
			for i := range ops {
				eb := ops[i].fixedElem
				if ops[i].fixedBits > 0 {
					// SIMD&FP scalar with a fixed width letter (<Dd> = 64), the
					// same for every arrangement row.
					eb = ops[i].fixedBits
				} else if l := ops[i].arngLink; l != "" {
					b, ok := lookup(tables[l], size)
					if !ok {
						// This operand's symbol has no element for this size
						// (e.g. a RESERVED row on one side of a widening op).
						skip = true
						break
					}
					eb = b
				}
				base := sign
				if inst.laneIsFloat(&ops[i]) {
					base = "float"
					if eb > 0 && eb < 16 {
						// No half/quarter-word floating-point Go types.
						skip = true
						break
					}
				}
				ops[i].instantiate(base, eb)
			}
			if skip {
				continue
			}
			for _, pred := range preds {
				variant := make([]Operand, len(ops))
				copy(variant, ops)
				elem := 0
				mixedWidths := false
				for i := range variant {
					if variant[i].Class == "vreg" && variant[i].ElemBits > 0 {
						if elem == 0 {
							elem = variant[i].ElemBits
						} else if variant[i].ElemBits != elem {
							mixedWidths = true
						}
					}
				}
				for i := range variant {
					if variant[i].Class != "mask" {
						continue
					}
					if variant[i].governing {
						variant[i].Predication = pred
					}
					if variant[i].ElemBits == 0 {
						// This predicate doesn't come with an arrangement (which is usual).
						// Get it from its peer data operand.
						if mixedWidths && !mixedWidthLogged[inst.mnemonic()] {
							mixedWidthLogged[inst.mnemonic()] = true
							log.Printf("sve: %s: operands have mixed element widths; predicate width provisionally %d — derive esize from the pseudocode before generating an API from this def",
								inst.mnemonic(), elem)
						}
						variant[i].ElemBits = elem
					}
				}
				for _, w := range widths {
					v := variant
					if w > 0 {
						v = make([]Operand, len(variant))
						copy(v, variant)
						for i := range v {
							if v[i].Class == "vreg" || v[i].Class == "mask" {
								v[i].ElemBits = w
							}
						}
					}
					defs = append(defs, inst.emitOne(asm, v, widthAgnostic))
				}
			}
		}
	}
	return defs
}
