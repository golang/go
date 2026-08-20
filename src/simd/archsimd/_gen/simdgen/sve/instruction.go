// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sve loads ARM64 SVE / SVE2 instruction definitions from the ARM A64
// ISA XML files and emits them as simdgen unify values.
// TODO: merge with the arm64 package, the approach taken here should take over
// the NEON loader.
// TODO: merge with x/arch/arm64/instgen?
//
// SVE registers are "scalable": their total bit width is the hardware
// implementation-defined vector length rather than a fixed 128/256/512 bits. So
// emitted vector operands carry only a base type and an element width, without a
// fixed bits/lanes count.
//
// Arrangement is per-operand. An SVE instruction template such as
//
//	ADD  <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>
//
// stands for a family of concrete instructions, one per value of the <T>
// arrangement symbol. simdgen enumerates them by resolving each operand's
// arrangement symbol from the section's explanations. Different symbols can be
// encoded in the same instruction field but interpreted differently, the
// loader also takes care of this.
//
// It emits register, mask, immediate, memory and special operands.
// Memory and special operands are opaque at this moment.
// Register-list operands are not modeled yet, except for single-register lists,
// so instructions carrying one are skipped (TODO); see classify.
//
// TODO: Peepholes might need the structure of memory operands, implement it?
// TODO: special operands are like registers with indexing, prefetch ops, etc.
// They seem too specialized that we might want to manually implment them instead
// of via simdgen, but we can revisit this.
package sve

import (
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/arch/arm64/instgen/xmlspec"
)

// signedImmRe matches an [Instruction.brief] that describes a signed/unsigned *immediate*
// (e.g. DUP/CPY "Move signed integer immediate ..."). There the signedness is a
// property of the immediate encoding, not of the vector lane, so such ops are
// signedness-agnostic.
var signedImmRe = regexp.MustCompile(`(un)?signed(\s+\w+)?\s+immediate`)

// reZReg and rePReg detect a Z (scalable vector) or P (predicate) register in an
// assembly template, used to choose the Go opcode prefix (see goOpPrefix). The
// [^/] guard excludes the /<ZM> predication qualifier, which is not a Z register.
// Copied from x/arch/arm64/instgen/xmlspec.
var (
	reZReg = regexp.MustCompile(`(^|[^/])<Z[A-Za-z1-9]+>`)
	rePReg = regexp.MustCompile(`<P[A-Za-z1-9]+>`)
)

// Instruction is a *logical* SVE instruction, one per iclass.
type Instruction struct {
	xmlspec.Instruction
	// iclass is the specific class this logical instruction represents.
	// A raw xmlspec.Instruction can hold several iclasses with distinct mnemonics.
	// If nil, the first iclass is used.
	iclass        *xmlspec.Iclass
	mnemonicCache string
	// predVariants is set on the unpredicated instruction of a
	// predicated/unpredicated pair (see [groupPredicationForms]), one entry per
	// predicated machine op the pair implies. It is nil for an instruction that
	// comes in one form only.
	predVariants []predVariant
}

// predVariant is one predicated encoding of an operation, as seen from its
// unpredicated sibling: the governing-predicate qualifiers it offers ("M", "Z",
// or "MZ" for an encoding written <Pg>/<ZM>, which supports either) and its
// register symbols, in the same order as the sibling's own results and
// non-predicate inputs.
//
// One encoding can imply several machine ops — one per qualifier — but they
// share these symbols, because they are the same encoding. A second entry would
// mean a genuinely separate predicated encoding, which no paired operation in
// the ISA has today; the list exists so that such an encoding could be
// described with its own symbols rather than collapsed onto the first one's.
type predVariant struct {
	quals       string
	outRegNames []string
	inRegNames  []string
	// predAsmPos is the assembly position of the encoding's governing
	// predicate: 1 on every encoding grouped today, but recorded rather than
	// assumed — PTEST, with no destination, governs from position 0.
	predAsmPos int
}

// ic returns the iclass this logical instruction represents, defaulting to the
// first iclass of the section.
func (inst *Instruction) ic() *xmlspec.Iclass {
	if inst.iclass != nil {
		return inst.iclass
	}
	if len(inst.Classes.Iclass) > 0 {
		return &inst.Classes.Iclass[0]
	}
	return nil
}

// extractDocVar returns the value of the named docvar, searching from most to
// least specific: this iclass, its encodings, then the section top level.
func (inst *Instruction) extractDocVar(key string) string {
	if ic := inst.ic(); ic != nil {
		for _, dv := range ic.DocVars {
			if dv.Key == key {
				return dv.Value
			}
		}
		for _, enc := range ic.Encodings {
			for _, dv := range enc.DocVars {
				if dv.Key == key {
					return dv.Value
				}
			}
		}
	}
	for _, dv := range inst.DocVars {
		if dv.Key == key {
			return dv.Value
		}
	}
	return ""
}

// mnemonic returns the instruction mnemonic, e.g. "ADD", "FADD", "SQADD".
func (inst *Instruction) mnemonic() string {
	if inst.mnemonicCache != "" {
		return inst.mnemonicCache
	}
	m := inst.extractDocVar("mnemonic")
	if inst.isAlias() {
		m = inst.extractDocVar("alias_mnemonic")
	}
	inst.mnemonicCache = m
	return m
}

// isAlias reports whether this XML entry describes an alias of another
// instruction.
func (inst *Instruction) isAlias() bool {
	return inst.Type == "alias"
}

// instrClass returns the instruction class docvar, e.g. "sve" or "sve2".
func (inst *Instruction) instrClass() string {
	return inst.extractDocVar("instr-class")
}

// isSVE reports whether this is an SVE or SVE2 instruction.
func (inst *Instruction) isSVE() bool {
	switch inst.instrClass() {
	case "sve", "sve2":
		return true
	}
	return false
}

// cpuFeature returns the simdgen cpuFeature string for this instruction.
func (inst *Instruction) cpuFeature() string {
	switch inst.instrClass() {
	case "sve2":
		return "SVE2"
	default:
		return "SVE"
	}
}

// goOpPrefix returns the Go opcode prefix: "Z" if the instruction uses a
// scalable vector register, else "P" if it uses a predicate register, else "".
// So the Go opcode is goOpPrefix()+mnemonic, e.g. ZADD but PPTRUE. Matches
// x/arch/arm64/instgen/xmlspec.goOpcodePrefix.
func (inst *Instruction) goOpPrefix() string {
	ic := inst.ic()
	if ic == nil {
		return ""
	}
	hasZ, hasP := false, false
	for _, enc := range ic.Encodings {
		s := asmTemplateToString(enc.AsmTemplate)
		hasZ = hasZ || reZReg.MatchString(s)
		hasP = hasP || rePReg.MatchString(s)
	}
	switch {
	case hasZ:
		return "Z"
	case hasP:
		return "P"
	default:
		return ""
	}
}

// laneIsFloat reports whether the given operand's vector lane holds
// floating-point values.
//
// The int<->float conversions have different lane types on input and output, and the
// operand's role selects which side this is:
//
//   - int->float (SCVTF/SCVTFLT, UCVTF/UCVTFLT): destination float, source int.
//   - float->int (FCVTZS/FCVTZU and narrowing, FLOGB): destination int, source
//     float.
//
// Every other instruction is uniform, i.e. all lanes the same type.
func (inst *Instruction) laneIsFloat(op *Operand) bool {
	switch op.Class {
	case "vreg", "greg":
		// has a lane
	default:
		// mask lanes are always integer; mem/immediate/special have no lane.
		return false
	}
	dst := op.role == "destination"
	switch inst.mnemonic() {
	case "SCVTF", "SCVTFLT", "UCVTF", "UCVTFLT": // integer -> floating point
		return dst
	case "FCVTZS", "FCVTZSN", "FCVTZU", "FCVTZUN", "FLOGB": // floating point -> integer
		return !dst
	}
	return isFloatBrief(inst.brief())
}

// isFloatBrief reports whether a brief description names a floating-point type.
// SVE spells these as "floating-point", "bfloat", or an "X-precision" (half /
// single / double / 8-bit) qualifier.
func isFloatBrief(brief string) bool {
	b := strings.ToLower(brief)
	return strings.Contains(b, "floating-point") ||
		strings.Contains(b, "bfloat") ||
		strings.Contains(b, "precision")
}

// signedness reports whether an integer instruction interprets its lanes as
// signed, unsigned, or agnostic, so the loader emits only the signedness the
// hardware actually implements, not spurious values. Many low-half/bitwise
// ops, e.g. ADD, SUB, MUL, EOR, etc., are genuinely agnostic.
// others are signedness-specific, e.g. SMAX vs UMAX, SDIV vs UDIV,
// the int<->float converts, etc.
//
// The signal is the instruction's brief description, which names the signedness
// for the specific ops ("Signed maximum", "Unsigned divide", "Signed integer
// convert ...") and omits it for the agnostic ones.
//
// Two adjustments: a brief describing a signed/unsigned *immediate*
// (DUP/CPY) is about the immediate, not the lane, so it stays agnostic; and the
// shift-right family and FLOGB name their signedness differently (arithmetic vs
// logical shift; "logarithm as integer") and are handled explicitly.
func (inst *Instruction) signedness() string {
	switch inst.mnemonic() {
	case "ASR", "ASRD", "ASRR", "FLOGB": // arithmetic (sign-propagating) / signed exponent
		return "int"
	case "LSR", "LSRR": // logical (zero-filling) shift right
		return "uint"
	}
	b := strings.ToLower(inst.brief())
	if signedImmRe.MatchString(b) {
		return ""
	}
	switch {
	case strings.Contains(b, "unsigned"):
		return "uint"
	case strings.Contains(b, "signed"): // "unsigned" already handled, so this is the word "signed"
		return "int"
	}
	return ""
}

// integerSignedness returns the signed/unsigned base variants to enumerate for
// the instruction's integer lanes: the single value fixed by signedness for a
// signedness-specific op, both {"int","uint"} for an agnostic op with an integer
// lane (simdgen narrows later via the Go op definitions), or a single no-op pass
// when there are no integer lanes.
func (inst *Instruction) integerSignedness(ops []Operand) []string {
	switch inst.signedness() {
	case "int":
		return []string{"int"}
	case "uint":
		return []string{"uint"}
	}
	for i := range ops {
		if c := ops[i].Class; (c == "vreg" || c == "greg") && !inst.laneIsFloat(&ops[i]) {
			return []string{"int", "uint"}
		}
	}
	return []string{""}
}

// brief returns the instruction's short human-readable description, e.g. "Signed
// maximum (predicated)".
func (inst *Instruction) brief() string {
	if len(inst.Desc.Brief.Para) > 0 {
		return strings.TrimSpace(inst.Desc.Brief.Para[0].Text)
	}
	return ""
}

// findExplanation returns the explanation whose symbol is encoded with the
// given link, or nil.
func (inst *Instruction) findExplanation(link string) *xmlspec.Explanation {
	for i := range inst.Explanations.Explanations {
		if inst.Explanations.Explanations[i].Symbol.Link == link {
			return &inst.Explanations.Explanations[i]
		}
	}
	return nil
}

// symbolIsGoverning reports whether this instruction's explanation for
// register symbol name (e.g. "Pg") describes it as the governing predicate —
// the spec writes "the governing scalable predicate register" for exactly the
// symbols with that role. found reports whether any explanation names the
// symbol at all. This is the authoritative classification; [buildOperandList]
// cross-checks it against the syntactic <Pg>/qualifier signal.
func (inst *Instruction) symbolIsGoverning(name string) (governing, found bool) {
	want := "<" + name + ">"
	for i := range inst.Explanations.Explanations {
		e := &inst.Explanations.Explanations[i]
		if strings.TrimSpace(e.Symbol.Value) != want {
			continue
		}
		found = true
		if strings.Contains(strings.ToLower(e.Account.Intro), "governing") {
			return true, true
		}
	}
	return false, found
}

// arngRow is one row of an arrangement size table: the encoding value of the
// size field and the resulting element width in bits.
type arngRow struct {
	size string // the size bitfield value, e.g. "01"; the shared key across symbols
	bits int    // element width for this size (8/16/32/64)
}

// resolveArrangementTable returns the (size -> element width) rows for the
// arrangement symbol encoded with the given link, read from its definition
// table in encoding order. RESERVED and header rows (no valid element letter)
// are dropped.
//
// Crucially, the size key is the shared encoding field, so different symbols
// (<T> and <Tb>) that select on the same field line up by size. That is what
// lets non-uniform (widening/narrowing) instructions like SUNPKHI give each
// operand its own element width for the same encoded instruction.
func (inst *Instruction) resolveArrangementTable(link string) []arngRow {
	exp := inst.findExplanation(link)
	if exp == nil {
		return nil
	}
	var rows []arngRow
	for i, row := range exp.Definition.Table.TGroup.TBody.Row {
		var size string
		bits := 0
		for _, entry := range row.Entries {
			switch entry.Class {
			case "bitfield":
				size = strings.TrimSpace(entry.Value)
			case "symbol":
				bits = elemLetterBits(strings.TrimSpace(entry.Value))
			}
		}
		if bits == 0 {
			continue // header or RESERVED row
		}
		if size == "" {
			size = fmt.Sprintf("#%d", i) // single-column table: key by position
		}
		rows = append(rows, arngRow{size: size, bits: bits})
	}
	return rows
}

// arngLinks returns the distinct arrangement-symbol links used by ops, with the
// destination's link first (it is the primary size driver), preserving order.
func arngLinks(ops []Operand) []string {
	seen := map[string]bool{}
	var links []string
	add := func(l string) {
		if l != "" && !seen[l] {
			seen[l] = true
			links = append(links, l)
		}
	}
	for _, op := range ops {
		if op.role == "destination" {
			add(op.arngLink)
		}
	}
	for _, op := range ops {
		add(op.arngLink)
	}
	return links
}

// elemLetterBits maps an SVE element specifier letter to its bit width.
func elemLetterBits(letter string) int {
	switch letter {
	case "B":
		return 8
	case "H":
		return 16
	case "S":
		return 32
	case "D":
		return 64
	default:
		return 0
	}
}

// elemLetter is the inverse of elemLetterBits: it maps a bit width to its SVE
// element specifier letter (used as the arrangement in emitted defs).
func elemLetter(bits int) string {
	switch bits {
	case 8:
		return "B"
	case 16:
		return "H"
	case 32:
		return "S"
	case 64:
		return "D"
	default:
		return ""
	}
}

// allEncodingOperands returns the operand list of every distinct encoding of this iclass.
func (inst *Instruction) allEncodingOperands() [][]Operand {
	ic := inst.ic()
	if ic == nil {
		return nil
	}
	seen := map[string]bool{}
	var out [][]Operand
	for _, enc := range ic.Encodings {
		s := asmTemplateToString(enc.AsmTemplate)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		ops := func() []Operand {
			// A classification panic names only the operand; add which
			// instruction and template it came from.
			defer func() {
				if r := recover(); r != nil {
					panic(fmt.Sprintf("%v\n  in %q template %q", r, inst.Title, s))
				}
			}()
			return operandsFromTextA(enc.AsmTemplate.TextA, inst.symbolIsGoverning)
		}()
		if len(ops) > 0 {
			inst.fixMemoryDirection(ops)
			out = append(out, ops)
		}
	}
	return out
}

// fixMemoryDirection re-roles a load/store's data direction, which the operand
// order does not reveal on its own. A store's destination is its memory operand
// (unusually, at the end of the syntax, e.g. ST1B {<Zt>.<T>}, <Pg>, [<Xn|SP>]);
// a load's destination is the transferred vector register (the memory is then a
// source). Load/store is read from the brief description.
func (inst *Instruction) fixMemoryDirection(ops []Operand) {
	b := strings.ToLower(inst.brief())
	store := strings.Contains(b, "store")
	load := strings.Contains(b, "load")
	if !store && !load {
		return
	}
	for i := range ops {
		switch {
		case store && ops[i].Class == "mem":
			ops[i].role = "destination"
		case load && ops[i].Class == "vreg":
			ops[i].role = "destination"
		}
	}
}

// operands parses the operands of this instruction's first encoding form. Most
// instructions have exactly one; use templates for the complete set.
func (inst *Instruction) operands() []Operand {
	if ops := inst.allEncodingOperands(); len(ops) > 0 {
		return ops[0]
	}
	return nil
}

// hasClass reports whether any operand has the given class.
func hasClass(ops []Operand, class string) bool {
	for _, op := range ops {
		if op.Class == class {
			return true
		}
	}
	return false
}

// predicationVariants returns the governing-predicate qualifiers to emit for a
// template: the predicate operand's own qualifier ("M" or "Z"), both when a
// single encoding written "<Pg>/<ZM>" (MOVPRFX) selects merging or zeroing via a
// bit, or a single no-op pass when the template has no governing predicate.
func predicationVariants(ops []Operand) []string {
	for i := range ops {
		if ops[i].governing {
			if ops[i].Predication == "MZ" {
				return []string{"M", "Z"}
			}
			return []string{ops[i].Predication}
		}
	}
	return []string{""}
}

// predicationForm reports whether this encoding is the predicated or the
// unpredicated form of an operation, as "predicated" / "unpredicated".
//
// It reads the encoding rather than the title: an encoding that takes a
// governing predicate is the predicated one. SVE does also spell this out in
// the title of an operation that has both forms ("ADD (vectors, predicated)"
// and "ADD (vectors, unpredicated)"), and [predicationGroupKey] uses that to pair
// them, but an operation that only comes predicated says nothing in its title —
// both of ABS's encodings are titled plain "ABS".
func (inst *Instruction) predicationForm() string {
	for _, ops := range inst.allEncodingOperands() {
		for i := range ops {
			if ops[i].governing {
				return "predicated"
			}
		}
	}
	return "unpredicated"
}

// predicationGroupKey returns the key that groups the encodings of one
// operation: the title with any predicated/unpredicated qualifier removed, e.g.
// both "ADD (vectors, predicated)" and "ADD (vectors, unpredicated)" yield "add
// (vectors)", and both of ABS's encodings yield "abs".
//
// Encodings that are not variations on one another keep distinct titles — "ADD
// (immediate)", "ADD (extended register)" — so they land in groups of their own,
// which groupPredicationForms then leaves alone.
func (inst *Instruction) predicationGroupKey() string {
	t := strings.ToLower(inst.Title)
	t = strings.ReplaceAll(t, "unpredicated", "")
	t = strings.ReplaceAll(t, "predicated", "")
	// Tidy the separator the qualifier left behind: "(vectors, )" -> "(vectors)".
	t = strings.ReplaceAll(t, ", )", ")")
	t = strings.ReplaceAll(t, "( ", "(")
	return strings.Join(strings.Fields(t), " ")
}

// documentation returns a one-line description of the instruction.
func (inst *Instruction) documentation() string {
	if len(inst.Desc.Authored.Paragraphs) > 0 {
		return inst.Desc.Authored.Paragraphs[0].Text
	}
	return inst.Title
}

// asmTemplateToString flattens an AsmTemplate to its text.
func asmTemplateToString(t xmlspec.AsmTemplate) string {
	var b strings.Builder
	for _, ta := range t.TextA {
		b.WriteString(ta.Value)
	}
	return b.String()
}
