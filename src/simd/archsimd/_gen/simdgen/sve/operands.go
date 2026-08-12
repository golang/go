// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/arch/arm64/instgen/xmlspec"
)

// arngValueRe matches an arrangement symbol's displayed value: the vector forms
// <T>, <Ta>, <Tb>, and the <V> size specifier of a SIMD&FP scalar (<V><d>). Its
// <a> link identifies the size table that gives this operand's element widths
// (see Instruction.resolveArrangementTable).
var arngValueRe = regexp.MustCompile(`^<(T[a-z]*|V)>$`)

// fixedArngRe matches a hardcoded element specifier, e.g. the ".D" in <Zm>.D.
var fixedArngRe = regexp.MustCompile(`\.([BHSD])\b`)

// simdFPRe matches a SIMD&FP scalar register: a fixed-width form (<Dd>, <Sn>,
// <Hd>, <Bd>, <Qd>) or an element-sized form (<V><d>, <V><n>). These hold a
// single value (a reduction result, or a DUP source), not a scalable vector.
var simdFPRe = regexp.MustCompile(`^(<[BHSDQ][a-z]>|<V><[a-z]>)$`)

// OperandType classifies an SVE instruction operand.
type OperandType int

const (
	// OperandZReg is a scalable vector register (Z), e.g. <Zd>.<T>, <Zn>.<T>.
	// It has no fixed total bit width: the width is the implementation-defined
	// vector length. Only its element type and element width are known.
	OperandZReg OperandType = iota
	// OperandPReg is a scalable predicate register (P), e.g. <Pg>/M, <Pd>.<T>.
	// A predicate is modeled as a Go mask value.
	OperandPReg
	// OperandGReg is a general-purpose scalar register (W/X/R).
	OperandGReg
	// OperandVFP is a SIMD&FP scalar register (<Dd>, <V><d>, ...): a single
	// fixed-width value, such as a horizontal reduction's result (SADDV <Dd>) or
	// a DUP scalar source. Unlike a Z register it is not scalable.
	OperandVFP
	// OperandImm is an immediate.
	OperandImm
	// OperandMem is a memory operand, e.g. [<Xn|SP>{, #<imm>, MUL VL}] or a
	// gather/scatter address like [<Xn|SP>, <Zm>.D, SXTW]. simdgen does not yet
	// distinguish the memory addressing modes; they are all one "mem" class.
	OperandMem
	// OperandList is a register list, e.g. { <Zt>.B } or { <Zt1>.D-<Zt2>.D }.
	// TODO: register lists are not modeled yet; instructions carrying one are
	// skipped (see classify).
	OperandList
	// OperandSpecial is a recognized but not-yet-detailed operand: an indexed
	// register (<Zm>.<T>[<index>]), a register with an optional modifier
	// ({, <pattern>}), or a special token (<prfop>, <vl>, <pattern>, <const>,
	// <mod>, and NEON-style <Vd>/<Dd> reduction results).
	OperandSpecial
	// OperandUnknown is a token the classifier could not place at all; an anomaly.
	OperandUnknown
)

func (t OperandType) String() string {
	switch t {
	case OperandZReg:
		return "ZReg"
	case OperandPReg:
		return "PReg"
	case OperandGReg:
		return "GReg"
	case OperandVFP:
		return "VFP"
	case OperandImm:
		return "Imm"
	case OperandMem:
		return "Mem"
	case OperandList:
		return "List"
	case OperandSpecial:
		return "Special"
	default:
		return "Unknown"
	}
}

// Operand is an SVE instruction operand instantiated for a concrete element size.
type Operand struct {
	Type     OperandType
	Class    string // "vreg", "mask", "greg", "immediate", "mem", "reglist", "special"
	BaseType string // "int", "uint", "float" (for vreg/mask/greg)
	ElemBits int    // element width in bits (8/16/32/64); 0 if unsized
	// Bits and Lanes are set for a fixed-width scalar register — a general-purpose
	// greg (<Xd>) or a SIMD&FP vreg (<Dd>): the total register width and lane
	// count (always 1). A scalable Z-vector leaves them 0 and is marked
	// "scalable" in the emitted def instead.
	Bits  int
	Lanes int

	// Predication is "M" (merging) or "Z" (zeroing) for governing predicates,
	// otherwise "".
	Predication string
	// AsmPos is the position in the assembly syntax (0 for the destination
	// register, 1+ for inputs). It mirrors the source template order and is the
	// field simdgen uses to order operands.
	AsmPos int
	// Raw is the source operand token, retained for deferred (mem/list/special)
	// and unknown operands so diagnostics can name what was skipped.
	Raw string

	// role is the operand's internal role: "destination", "op0"/"op1"/..., or
	// "mask" (a governing predicate). It drives out/in/inVariant partitioning at
	// emit time but is NOT emitted (simdgen orders operands by AsmPos, so a role
	// field in the YAML would be redundant).
	role string
	// arngLink is the <a> link of this operand's arrangement symbol (<T>/<Ta>/
	// <Tb>), used to resolve its per-operand element widths. Empty if the
	// operand has a fixed or no arrangement.
	arngLink string
	// fixedElem is a hardcoded element width (from e.g. ".D"), or 0.
	fixedElem int
	// fixedBits is the fixed total width of a SIMD&FP scalar named by a size
	// letter (<Dd> -> 64, <Sd> -> 32, ...), or 0 for an element-sized <V><d>.
	fixedBits int
	// isList reports that this register came from a single-register list
	// ("{ <Zt>.<T> }"). It is a distinct assembler encoding from a bare register,
	// so it is preserved (emitted as listNumber) even though the register is
	// otherwise handled like any vreg.
	isList bool
	// regName is the inner register symbol, e.g. "Zdn", "Zm", "Pg".
	regName string
}

// resultInArg0 reports whether this destination register is also read, i.e. it
// is written in place (an ARM <Zdn>/<Zda>-style operand).
func (op *Operand) resultInArg0() bool {
	return op.role == "destination" && isInPlaceReg(op.regName)
}

// aElem is a single <a> symbol from an assembly template: its displayed value
// and its link. The link, not the value, is the stable key used to resolve a
// symbol's definition (see Instruction.findExplanation).
//
// For example, in the template "ADD <Zdn>.<T>, ..." the operand "<Zdn>.<T>"
// contributes two <a> elements:
//
//	{value: "<Zdn>", link: "Zdn"}   // the register symbol
//	{value: "<T>",   link: "T__3"}  // the arrangement symbol
type aElem struct {
	value string
	link  string
}

// rawTok is one operand's raw text plus the <a> symbols it contains, before
// classification. The <a> links let us resolve each operand's arrangement.
//
// For "ADD <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>", the third operand tokenizes
// to:
//
//	rawTok{
//	    text:   "<Zdn>.<T>",
//	    asmPos: 2,                 // 0 = destination, 1+ = following operands
//	    aElems: [{"<Zdn>","Zdn"}, {"<T>","T__3"}],
//	}
type rawTok struct {
	text   string
	asmPos int
	aElems []aElem
}

// tok is a rawTok after classification, before it is instantiated for
// a concrete element size. Examples of the interesting fields:
//
//	"<Zdn>.<T>"          -> {operandType: OperandZReg,  isDestination: true,
//	                         regName: "Zdn", arngLink: "T__3"}
//	"<Zm>.<T>"           -> {operandType: OperandZReg,  isDestination: false,
//	                         regName: "Zm",  arngLink: "T__3"}
//	"<Pg>/M"             -> {operandType: OperandPReg,  predication: "M",
//	                         regName: "Pg"}   // governing predicate ("Z"/"MZ" too)
//	"<Zt>.D"             -> {operandType: OperandZReg,  fixedElem: 64}
//	                         // hardcoded arrangement, so no arngLink
//	"#<imm>"             -> {operandType: OperandImm}
//	"[<Xn|SP>{, #<imm>}]"-> {operandType: OperandMem}
//	"<Zm>.<T>[<index>]"  -> {operandType: OperandSpecial} // indexed, not modeled
type tok struct {
	// text is the raw operand token, e.g. "<Zdn>.<T>".
	text string
	// asmPos is the position in the assembly syntax (0 = destination, 1+ = the
	// following operands), mirroring the template order.
	asmPos int
	// operandType is the classification (OperandZReg, OperandPReg, OperandMem,
	// OperandSpecial, ...).
	operandType OperandType
	// isDestination is true when the register is written (an ARM 'd'-role symbol
	// such as <Zd>, <Zdn>, <Pd>).
	isDestination bool
	// predication is "M" (merging), "Z" (zeroing), or "MZ" (a <Pg>/<ZM> encoding
	// selecting either) for a governing predicate; "" otherwise.
	predication string
	// regName is the inner register symbol, e.g. "Zdn", "Zm", "Pg".
	regName string
	// arngLink is the <a> link of this operand's variable arrangement symbol
	// (<T>/<Ta>/<Tb>, or <V> for a SIMD&FP scalar), used to resolve its element
	// widths; "" if the arrangement is fixed or absent.
	arngLink string
	// fixedElem is a hardcoded element width in bits from a literal ".B"/".H"/
	// ".S"/".D" (8/16/32/64), or 0.
	fixedElem int
	// fixedBits is the fixed total width of a SIMD&FP scalar named by a size
	// letter (<Dd> -> 64, <Sd> -> 32, ...), or 0 for an element-sized <V><d>.
	fixedBits int
	// isList reports that this register came from a single-register list
	isList bool
}

// operandsFromTextA parses operands from an assembly template's <text>/<a>
// sequence, preserving each operand's arrangement-symbol link.
func operandsFromTextA(textA []xmlspec.TextA) []Operand {
	return buildOperandList(classifyToks(tokenizeTextA(textA)))
}

// operands parses operands from a flattened template string. It cannot recover
// <a> links, so arrangement symbols resolve to empty links; it is used for
// classification-only paths and tests. The real loader path uses
// operandsFromTextA.
func operands(asmTemplate string) []Operand {
	return buildOperandList(classifyToks(tokenizeString(asmTemplate)))
}

// tokenizeTextA splits a <text>/<a> sequence into operand tokens on top-level
// commas, stripping the leading mnemonic and recording each <a> symbol.
func tokenizeTextA(textA []xmlspec.TextA) []rawTok {
	var toks []rawTok
	cur := rawTok{}
	depth := 0
	started := false // have we passed the mnemonic word?
	flush := func() {
		cur.text = strings.TrimSpace(cur.text)
		if cur.text != "" || len(cur.aElems) > 0 {
			cur.asmPos = len(toks)
			toks = append(toks, cur)
		}
		cur = rawTok{}
	}
	for _, ta := range textA {
		if ta.Link != "" {
			cur.text += ta.Value
			cur.aElems = append(cur.aElems, aElem{strings.TrimSpace(ta.Value), ta.Link})
			started = true
			continue
		}
		s := ta.Value
		if !started {
			// Strip the mnemonic: keep everything after the first space.
			if i := strings.IndexByte(s, ' '); i >= 0 {
				s = s[i:]
			} else {
				s = ""
			}
			started = true
		}
		for _, r := range s {
			switch r {
			case '[', '{':
				depth++
			case ']', '}':
				depth--
			case ',':
				if depth == 0 {
					flush()
					continue
				}
			}
			cur.text += string(r)
		}
	}
	flush()
	return toks
}

// tokenizeString splits a flattened template string into operand tokens. It has
// no <a> link information.
func tokenizeString(template string) []rawTok {
	template = stripMnemonic(template)
	var toks []rawTok
	depth := 0
	var cur strings.Builder
	flush := func() {
		if s := strings.TrimSpace(cur.String()); s != "" {
			toks = append(toks, rawTok{text: s, asmPos: len(toks)})
		}
		cur.Reset()
	}
	for _, r := range template {
		switch r {
		case '[', '{':
			depth++
		case ']', '}':
			depth--
		case ',':
			if depth == 0 {
				flush()
				continue
			}
		}
		cur.WriteRune(r)
	}
	flush()
	return toks
}

// stripMnemonic removes the leading mnemonic from an assembly template. A
// template with no space is a mnemonic-only (nullary) instruction.
func stripMnemonic(template string) string {
	if _, after, ok := strings.Cut(strings.TrimSpace(template), " "); ok {
		return strings.TrimSpace(after)
	}
	return ""
}

// classifyToks classifies each raw token and attaches its arrangement source.
func classifyToks(toks []rawTok) []tok {
	parsed := make([]tok, 0, len(toks))
	for _, t := range toks {
		p := classifyText(t.text, t.asmPos)
		// Per-operand arrangement: could be a variable arrangement symbol (<T>/<Ta>/<Tb>)
		// or a fixed element, or none, e.g. for a greg.
		for _, a := range t.aElems {
			if arngValueRe.MatchString(a.value) {
				p.arngLink = a.link
			}
		}
		if p.arngLink == "" {
			if m := fixedArngRe.FindStringSubmatch(t.text); m != nil {
				p.fixedElem = elemLetterBits(m[1])
			}
		}
		parsed = append(parsed, p)
	}
	return parsed
}

// classifyText determines an operand's type, destination-ness, predication and
// register symbol from its text.
//
// A register token counts as "clean" only if it has no index or optional
// modifier ('[' or '{'). Indexed/modified registers and other angle-bracket
// tokens (<prfop>, <vl>, <mod>, <Vd>, ...) are OperandSpecial; anything else is
// OperandUnknown.
func classifyText(text string, asmPos int) tok {
	p := tok{text: text, asmPos: asmPos}
	// A single-register list ("{ <Zt>.<T> }") is treated as its inner register
	// (but flagged, as it is a distinct assembler encoding); multi-register lists
	// remain OperandList (deferred).
	reg := text
	if inner, ok := singleRegList(text); ok {
		reg = inner
		p.isList = true
	}
	clean := !strings.ContainsAny(reg, "[{")
	switch {
	case strings.HasPrefix(reg, "["):
		p.operandType = OperandMem
	case strings.HasPrefix(reg, "{"):
		p.operandType = OperandList
	case strings.HasPrefix(reg, "#"), strings.HasPrefix(reg, "<const>"):
		p.operandType = OperandImm
	case simdFPRe.MatchString(reg):
		// A SIMD&FP scalar register: a reduction result <Dd>/<V><d> or a DUP
		// source <V><n>. Its width is fixed by the size letter, or element-sized
		// for the <V> form (resolved via its <a> link like <T>).
		p.operandType = OperandVFP
		p.regName = regSymbol(reg)
		p.isDestination = isDestinationReg(p.regName) || strings.Contains(reg, "<d>")
		p.fixedBits = simdFPLetterBits(reg)
	case clean && strings.HasPrefix(reg, "<Z"):
		p.operandType = OperandZReg
		p.regName = regSymbol(reg)
		p.isDestination = isDestinationReg(p.regName)
	case clean && strings.HasPrefix(reg, "<P"):
		p.operandType = OperandPReg
		p.regName = regSymbol(reg)
		p.isDestination = isDestinationReg(p.regName)
		switch {
		case strings.Contains(reg, "/<ZM>"):
			// A single encoding (MOVPRFX) whose bit selects merging or zeroing.
			p.predication = "MZ"
		case strings.HasSuffix(reg, "/M"):
			p.predication = "M"
		case strings.HasSuffix(reg, "/Z"):
			p.predication = "Z"
		}
	case clean && (strings.HasPrefix(reg, "<W") || strings.HasPrefix(reg, "<X") || strings.HasPrefix(reg, "<R")):
		p.operandType = OperandGReg
		p.regName = regSymbol(reg)
		p.isDestination = isDestinationReg(p.regName)
		p.fixedBits = gregLetterBits(reg)
	case strings.HasPrefix(reg, "<"):
		p.operandType = OperandSpecial
		// A special operand can still be a destination, e.g. an indexed
		// destination <Zd>.<T>[<index>].
		p.regName = regSymbol(reg)
		p.isDestination = isDestinationReg(p.regName) || strings.Contains(reg, "<d>")
	default:
		p.operandType = OperandUnknown
	}
	return p
}

// singleRegList reports whether text is a single-register list like
// "{ <Zt>.<T> }" and, if so, returns its inner register token. Multi-register
// lists (a comma-separated set or a "-" range) return false and stay opaque.
func singleRegList(text string) (string, bool) {
	if !strings.HasPrefix(text, "{") || !strings.HasSuffix(text, "}") {
		return "", false
	}
	inner := strings.TrimSpace(text[1 : len(text)-1])
	if strings.ContainsAny(inner, ",-") { // multiple registers or a range
		return "", false
	}
	return inner, true
}

// simdFPLetterBits returns the fixed width of a size-lettered SIMD&FP scalar
// register (<Bd>=8, <Hd>=16, <Sd>=32, <Dd>=64, <Qd>=128), or 0 for the
// element-sized <V><d> form (whose width comes from its <V> arrangement link).
func simdFPLetterBits(text string) int {
	if len(text) < 2 {
		return 0
	}
	switch text[1] {
	case 'B':
		return 8
	case 'H':
		return 16
	case 'S':
		return 32
	case 'D':
		return 64
	case 'Q':
		return 128
	}
	return 0
}

// gregLetterBits returns the width of a general-purpose scalar register from its
// size letter (<Wd>=32, <Xd>=64), or 0 when the width is not fixed by the name
// (e.g. the width-variable <R> form).
func gregLetterBits(text string) int {
	if len(text) < 2 {
		return 0
	}
	switch text[1] {
	case 'W':
		return 32
	case 'X':
		return 64
	}
	return 0
}

// regSymbol extracts the inner register symbol from a token, e.g. "<Zdn>.<T>" ->
// "Zdn", "<Pg>/M" -> "Pg".
func regSymbol(text string) string {
	if i := strings.IndexByte(text, '<'); i >= 0 {
		text = text[i+1:]
	}
	if i := strings.IndexByte(text, '>'); i >= 0 {
		text = text[:i]
	}
	return text
}

// isDestinationReg reports whether a register symbol names a destination
// register. The destination role letter 'd' appears either right after the
// class letter (Zd, Zda, Zdn), or as the trailing role letter (Pd, Wd, Xd, PNd).
func isDestinationReg(name string) bool {
	if len(name) < 2 {
		return false
	}
	return name[1] == 'd' || name[len(name)-1] == 'd'
}

// isInPlaceReg reports whether a destination register symbol is also a source
// (read-modify-write), such as <Zdn> or <Zda>. A bare <Zd> is a pure output.
func isInPlaceReg(name string) bool {
	return len(name) >= 3 && name[1] == 'd'
}

// buildOperandList lowers tokens into Operands ordered as outputs then
// inputs, assigning roles and handling read-modify-write destinations.
//
// Unlike an AMD64 AVX-512 K-mask, an SVE governing predicate is NOT optional:
// there is no K0-style "no predicate" encoding, so it is a mandatory literal
// input (class "mask", role "mask"), not an inVariant. See the discussion in
// emitOne.
func buildOperandList(parsed []tok) []Operand {
	var outs, ins []Operand
	inputCount := 0
	destAssigned := false

	// place assigns op's role — the (single) destination if isDestination,
	// otherwise the next numbered input "opN" (a repeated destination symbol is
	// the in-place source) — and files it under outs or ins.
	place := func(op Operand, isDestination bool) {
		if isDestination && !destAssigned {
			op.role = "destination"
			destAssigned = true
			outs = append(outs, op)
			return
		}
		op.role = inputRole(inputCount)
		inputCount++
		ins = append(ins, op)
	}

	deferredClass := map[OperandType]string{
		OperandMem:     "mem",
		OperandList:    "reglist",
		OperandSpecial: "special",
		OperandUnknown: "unknown",
	}

	for _, p := range parsed {
		// We don't model the details of these types yet, so just naively record them and continue.
		// TODO: we might need at least the details of OperandMem soon.
		if class, ok := deferredClass[p.operandType]; ok {
			place(Operand{
				Type: p.operandType, Class: class, Raw: p.text,
				AsmPos: p.asmPos, regName: p.regName,
			}, p.isDestination)
			continue
		}
		switch p.operandType {
		case OperandPReg:
			if p.regName == "Pg" || p.predication != "" {
				// Governing predicate: the operand named <Pg> ("g" for governing), a
				// mandatory mask input (role "mask", not a numbered opN). Most carry a
				// /Z or /M qualifier (predicated data-processing ops), but some do not
				// — e.g. the store ST1B {<Zt>.B}, <Pg>, [...] governs with a plain
				// <Pg> — so key on the register name, not the qualifier. Source
				// predicates <Pn>/<Pm> and the destination <Pd> are ordinary operands,
				// filed by place() below.
				ins = append(ins, Operand{
					Type: OperandPReg, Class: "mask", role: "mask",
					Predication: p.predication, AsmPos: p.asmPos,
					arngLink: p.arngLink, fixedElem: p.fixedElem, regName: p.regName,
				})
				continue
			}
			place(Operand{
				Type: OperandPReg, Class: "mask", AsmPos: p.asmPos,
				arngLink: p.arngLink, fixedElem: p.fixedElem, isList: p.isList, regName: p.regName,
			}, p.isDestination)
		case OperandImm:
			place(Operand{Type: OperandImm, Class: "immediate", AsmPos: p.asmPos}, false)
		default: // OperandZReg, OperandGReg, OperandVFP
			class := "vreg"
			if p.operandType == OperandGReg {
				// A general-purpose scalar register.
				class = "greg"
			}
			// A SIMD&FP scalar (OperandVFP) stays "vreg": it lives in the FP/SIMD
			// register bank, not the GP bank — just with a fixed width and lanes:1
			// rather than a scalable length.
			place(Operand{
				Type: p.operandType, Class: class, AsmPos: p.asmPos,
				arngLink: p.arngLink, fixedElem: p.fixedElem, fixedBits: p.fixedBits,
				isList: p.isList, regName: p.regName,
			}, p.isDestination)
		}
	}
	return append(outs, ins...)
}

// inputRole names an input operand: "op0", "op1", ...
func inputRole(index int) string {
	return fmt.Sprintf("op%d", index)
}

// instantiate stamps a base type and element width into a typed operand. mem,
// immediate, reglist and special operands are opaque and left unchanged.
func (op *Operand) instantiate(baseType string, elemBits int) {
	switch op.Type {
	case OperandZReg:
		// A scalable Z vector: only base type and element width; the total width
		// is the (unknown) vector length.
		op.BaseType = baseType
		op.ElemBits = elemBits
	case OperandGReg, OperandVFP:
		// A scalar register — general-purpose (<Xd>) or SIMD&FP (<Dd>) — holds a
		// single fixed-width value, so it has a concrete total width and lanes=1.
		op.BaseType = baseType
		op.ElemBits = elemBits
		op.Bits = elemBits
		op.Lanes = 1
	case OperandPReg:
		// Predicates are integer masks; their element width tracks the governed
		// vector's element width.
		op.BaseType = "int"
		op.ElemBits = elemBits
	}
}
