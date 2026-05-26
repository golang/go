// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/arch/arm64/instgen/xmlspec"
)

var (
	resultInArg0Re     = regexp.MustCompile(`= V\{[^}]*\}\(d\)`)        // pseudocode reading dest register
	floatPatternRe     = regexp.MustCompile(`-?(?:half|single|double)`) // docvar float type detection
	fixedArrangementRe = regexp.MustCompile(`\.(\d+[BHSD])`)            // hardcoded arrangements like .16B, .4S
	arngSymbolRe       = regexp.MustCompile(`\.<(T[a-z]*)>`)            // arrangement symbols like <T>, <Ta>, <Tb>
)

// Instruction represents a parsed ARM64 instruction with domain logic
type Instruction struct {
	xmlspec.Instruction               // Embedded XML data from xmlspec
	arrangementsCache   []Arrangement // Cache for arrangements
	mnemonicCache       string        // Cache for mnemonic
	arngShape           ArngShape     // Arrangement shape
}

// BaseTypeSet allows to specify the type set of values independent of arrangement's size, e.g.:
// - Float (instruction used for floating point values in lanes),
// - Uint (instruction used only for unsigned integer values in lanes with any arrangement),
// - Float|Int|Uint (e.g. VMOV V1.S[i], V0.S[j]: copy i-th lane from src vreg to j-th lane of dst vreg: basically don't care about base type).
type BaseTypeSet int

const (
	BaseTypeInt = 1 << iota
	BaseTypeUint
	BaseTypeFloat
)

func (t BaseTypeSet) String() string {
	switch t {
	case BaseTypeInt:
		return "int"
	case BaseTypeUint:
		return "uint"
	case BaseTypeFloat:
		return "float"
	default:
		return ""
	}
}

// template defines operand templates for instruction which get instantiated for each arrangement.
type template struct {
	operands    []Operand
	instruction *Instruction
}

// Arrangement defines the properties of a vector arrangement.
type Arrangement struct {
	arrangement string
	baseType    string
	elemBits    int
	bits        int
	lanes       int
}

// ArngShape makes certain vreg operands half or double bits wide.
type ArngShape int

const (
	DefaultArngs     = ArngShape(iota) // DefaultArngs indicates that vector register arguments have the same bit width.
	NarrowArngs                        // NarrowArngs signifies that the destination vector register is half the bit width of the source, used in instructions like XTN/XTN2.
	LongArngs                          // LongArngs indicates that the destination vector register is double the bit width of the source, seen in instructions like UXTL/UXTL2.
	WideArngs                          // WideArngs applies when the second argument vector register is half the bit width of the first argument or the result, as in UADDW.
	UnsupportedArngs                   // UnsupportedArngs indicates instructions whose arrangement shape is not yet supported by simdgen.
)

// extractDocVar searches for a docvar by key in the instruction's docvars
func (instruction *Instruction) extractDocVar(key string) string {
	for _, docVar := range instruction.DocVars {
		if docVar.Key == key {
			return docVar.Value
		}
	}
	return ""
}

// Mnemonic extracts the mnemonic from docvars
func (instruction *Instruction) Mnemonic() string {
	if instruction.mnemonicCache != "" {
		return instruction.mnemonicCache
	}

	var mnemonic string
	if instruction.IsAlias() {
		mnemonic = instruction.extractDocVar("alias_mnemonic")
	} else {
		mnemonic = instruction.extractDocVar("mnemonic")
	}

	instruction.mnemonicCache = mnemonic
	return mnemonic
}

// Bitwise returns true if the instruction is a bitwise operation
// by detecting "Bitwise " prefix in the brief description
func (instruction *Instruction) Bitwise() bool {
	brief := instruction.Brief()
	return strings.HasPrefix(brief, "Bitwise ")
}

// InstrClass returns the instruction Class from docvars
func (instruction *Instruction) InstrClass() string {
	return instruction.extractDocVar("instr-class")
}

// ResultInArg0 determines if result shares register with first argument.
// This occurs when the destination register is also read as an input operand.
func (instruction *Instruction) ResultInArg0() bool {
	mnemonic := instruction.Mnemonic()

	// For TBL/TBX the pattern looks like "= if is_tbl ... else V[d]".
	if mnemonic == "TBX" {
		return true
	}

	// Check pseudocode for reading destination register
	for _, PsSection := range instruction.PsSections {
		for _, Ps := range PsSection.Ps {
			for _, pstext := range Ps.PSText {
				if resultInArg0Re.MatchString(pstext) {
					return true
				}
			}
		}
	}
	return false
}

// IsAlias returns true if this instruction is an alias of another instruction
func (instruction *Instruction) IsAlias() bool {
	return instruction.Type == "alias"
}

// BaseTypeSet determines if an instruction operates on integers or floats
func (instruction *Instruction) BaseTypeSet() BaseTypeSet {
	mnemonic := instruction.Mnemonic()
	// DUP and INS are special as they can operate on any base type (int, uint, float)
	// depending on the arrangement and the source register. They are not easily
	// categorized by the float patterns in docvars.
	if mnemonic == "DUP" || mnemonic == "INS" {
		return BaseTypeInt | BaseTypeUint | BaseTypeFloat
	}

	for _, docVar := range instruction.DocVars {
		if floatPatternRe.MatchString(docVar.Value) {
			return BaseTypeFloat
		}
	}

	for _, Class := range instruction.Classes.Iclass {
		for _, docVar := range Class.DocVars {
			if floatPatternRe.MatchString(docVar.Value) {
				return BaseTypeFloat
			}
		}
	}

	return BaseTypeInt | BaseTypeUint
}

// extractUMOVArrangements handles special case for UMOV instruction
// UMOV has specific arrangements based on element size and vector register size
func (instruction *Instruction) extractUMOVArrangements() []string {
	var arrangements []string

	// Check if this is the 32-bit or 64-bit variant by looking at the assembly template
	var has32Bit, has64Bit bool
	if len(instruction.Classes.Iclass) > 0 && len(instruction.Classes.Iclass[0].Encodings) > 0 {
		for _, Encoding := range instruction.Classes.Iclass[0].Encodings {
			AsmTemplate := asmTemplateToString(Encoding.AsmTemplate)
			if strings.Contains(AsmTemplate, "<Wd>") {
				has32Bit = true
			}
			if strings.Contains(AsmTemplate, "<Xd>") {
				has64Bit = true
			}
		}
	}

	// Generate arrangements based on available variants
	if has32Bit {
		// 32-bit UMOV variants (Wd destination)
		arrangements = append(arrangements, "8B", "4H", "2S")
	}
	if has64Bit {
		// 64-bit UMOV variants (Xd destination)
		arrangements = append(arrangements, "16B", "8H", "4S", "2D")
	}

	return arrangements
}

// Arrangements collects valid arrangement/type specifiers for the instruction
func (instruction *Instruction) Arrangements() ([]Arrangement, ArngShape) {
	if instruction.arrangementsCache != nil {
		return instruction.arrangementsCache, instruction.arngShape
	}
	baseTypeSet := instruction.BaseTypeSet()
	stringArrangements, ashape := instruction.arrangementStrings()
	bitwise := instruction.Bitwise()
	var arrangements []Arrangement
	for ty := BaseTypeSet(BaseTypeInt); ty <= BaseTypeFloat; ty <<= 1 {
		if ty&baseTypeSet == 0 {
			continue
		}
		for _, arrStr := range stringArrangements {
			elemBits, bits, lanes := parseArrangement(arrStr)
			if elemBits == 0 {
				continue
			}
			if ty == BaseTypeFloat && elemBits != 32 && elemBits != 64 {
				continue
			}
			maxElemBits := elemBits
			if bitwise {
				maxElemBits = bits >> 1
			}
			l := lanes
			for e := elemBits; e <= maxElemBits; e = e * 2 {
				arrangements = append(arrangements, Arrangement{
					arrangement: arrStr,
					baseType:    ty.String(),
					elemBits:    e,
					bits:        bits,
					lanes:       l,
				})
				l = l >> 1
			}
		}
	}
	instruction.arrangementsCache = arrangements
	instruction.arngShape = ashape
	return arrangements, ashape
}

// extractFixedArrangements extracts hardcoded arrangements from asmtemplate
// (e.g., AESE with ".16B"). For instructions with variable arrangements
// (e.g., ADD with "<T>"), returns empty slice.
func (instruction *Instruction) extractFixedArrangements() []string {
	var arrangements []string

	for _, class := range instruction.Classes.Iclass {
		for _, encoding := range class.Encodings {
			var templateStr string
			for _, content := range encoding.AsmTemplate.TextA {
				if content.Link == "sa_t" || content.Link == "sa_ta" {
					return []string{}
				}
				templateStr += content.Value
			}

			// Extract hardcoded arrangements like ".16B", ".4S", ".8H", ".2D"
			matches := fixedArrangementRe.FindAllStringSubmatch(templateStr, -1)
			for _, match := range matches {
				if len(match) > 1 {
					arrangement := match[1]
					if arrangement != "" {
						// Avoid non-existing fixed arrangements like 4B (e.g. see USDOT's asmtemplate).
						_, bits, _ := parseArrangement(arrangement)
						if bits >= 64 {
							arrangements = append(arrangements, arrangement)
						}
					}
				}
			}
		}
	}
	return removeDuplicates(arrangements)
}

// arrangementSymbols extracts arrangement symbols from the first encoding
// that has arrangements in its assembly template.
//
// Variable symbols (e.g., T, Ta, Tb) appear as <T>, <Ta>, <Tb> in templates.
// Hardcoded arrangements (e.g., 16B, 2D) appear as .16B, .2D in templates.
//
// Returns nil if no arrangements are found at all.
// Returns single-element (e.g., ["T"] or ["16B"]) for uniform/fixed instructions.
// Returns multiple elements (e.g., ["Ta","Tb"] or ["Ta","2D"]) for non-uniform
// instructions where operand widths differ.
func (instruction *Instruction) arrangementSymbols() []string {
	for _, class := range instruction.Classes.Iclass {
		for _, encoding := range class.Encodings {
			templateStr := asmTemplateToString(encoding.AsmTemplate)
			if !strings.Contains(templateStr, ">.") {
				continue
			}

			seen := make(map[string]bool)
			var symbols []string

			// Extract variable arrangement symbols (e.g., T, Ta, Tb, Ts).
			for _, m := range arngSymbolRe.FindAllStringSubmatch(templateStr, -1) {
				sym := m[1]
				if !seen[sym] {
					seen[sym] = true
					symbols = append(symbols, sym)
				}
			}

			// Extract hardcoded arrangements (e.g., 16B, 4S, 2D).
			for _, m := range fixedArrangementRe.FindAllStringSubmatch(templateStr, -1) {
				arr := m[1]
				if !seen[arr] {
					seen[arr] = true
					symbols = append(symbols, arr)
				}
			}

			if len(symbols) > 0 {
				return symbols
			}
		}
	}
	return nil
}

// arrangementStrings extracts arrangement specifiers from XML explanations as strings
func (instruction *Instruction) arrangementStrings() ([]string, ArngShape) {
	var arrangements []string

	mnemonic := instruction.Mnemonic()
	ashape := DefaultArngs

	switch mnemonic {
	case "UMOV":
		return instruction.extractUMOVArrangements(), DefaultArngs

	case "INS":
		// INS instruction inserts general register values into vector elements
		// It supports all arrangements: B, H, S, D with 128-bit vector registers
		arrangements = append(arrangements, "16B", "8H", "4S", "2D")
		return arrangements, DefaultArngs
	}

	for _, Explanation := range instruction.Explanations.Explanations {
		Definition := Explanation.Definition
		if Definition.Table.TGroup.TBody.Row != nil {
			for _, Row := range Definition.Table.TGroup.TBody.Row {
				for _, Entry := range Row.Entries {
					if Entry.Class == "symbol" {
						arrangements = append(arrangements, strings.TrimSpace(Entry.Value))
					}
				}
			}
		}
	}

	fixedArrangements := instruction.extractFixedArrangements()
	arrangements = append(arrangements, fixedArrangements...)
	arrangements = removeDuplicates(arrangements)
	ashape = instruction.ArngShape()
	return arrangements, ashape
}

// regDiagramArngShape returns the expected arrangement shape based on RegDiagram for NEON.
// Used for cross-check verification by ArngShape() only.
func (instruction *Instruction) regDiagramArngShape() ArngShape {
	for _, class := range instruction.Classes.Iclass {
		psName := class.RegDiagram.PsName
		if strings.Contains(psName, "_L.") || strings.HasSuffix(psName, "_L") ||
			strings.HasSuffix(psName, "_P") { // _P = pairwise long (e.g., SADALP, SADDLP)
			return LongArngs
		}
		if strings.Contains(psName, "_W.") || strings.HasSuffix(psName, "_W") {
			return WideArngs
		}
		if strings.Contains(psName, "_N.") || strings.HasSuffix(psName, "_N") {
			return NarrowArngs
		}
	}
	switch instruction.Mnemonic() {
	case "FCVTN":
		return NarrowArngs
	case "SHLL":
		return LongArngs
	}
	return DefaultArngs
}

// ArngShape returns the arrangement shape.
// Returns UnsupportedArngs for instructions we don't support yet - those will not be emitted.
// If we were not able to classify the instruction, panic to prevent wrong yaml generation.
func (instruction *Instruction) ArngShape() ArngShape {
	symbols := instruction.arrangementSymbols()
	if symbols == nil {
		return UnsupportedArngs
	}

	for _, s := range symbols {
		switch s {
		case "T", "Ta", "Tb":
			// Known variable arrangement symbols.
		case "Ts":
			// Element-wise, not yet supported.
			return UnsupportedArngs
		default:
			// Fixed arrangements (e.g., "16B", "4S", "2D").
			if !fixedArrangementRe.MatchString("." + s) {
				panic(fmt.Sprintf("ArngShape: unknown arrangement symbol %q in %q (symbols: %v)", s, instruction.Mnemonic(), symbols))
			}
		}
	}

	var shape ArngShape
	if len(symbols) == 1 {
		if symbols[0] == "Ta" || symbols[0] == "Tb" {
			panic(fmt.Sprintf("ArngShape: unexpected lone %q in %q (symbols: %v)", symbols[0], instruction.Mnemonic(), symbols))
		}
		shape = DefaultArngs
	} else {
		switch mnemonic := instruction.Mnemonic(); mnemonic {
		// NarrowArngs: destination register is half the width of source(s).
		case "XTN", "SQXTN", "UQXTN", "SQXTUN",
			"ADDHN", "SUBHN", "RADDHN", "RSUBHN",
			"SHRN", "RSHRN",
			"SQSHRN", "SQRSHRN", "SQSHRUN", "SQRSHRUN",
			"UQSHRN", "UQRSHRN",
			"FCVTN", "FCVTXN":
			shape = NarrowArngs

		// LongArngs: destination register is double the width of source.
		case "SXTL", "UXTL",
			"SADDLP", "UADDLP", "SADALP", "UADALP",
			"SMULL", "UMULL", "PMULL",
			"SADDL", "UADDL", "SSUBL", "USUBL",
			"SSHLL", "USHLL", "SHLL",
			"SABAL", "UABAL", "SABDL", "UABDL",
			"SMLAL", "UMLAL", "SMLSL", "UMLSL",
			"SQDMLAL", "SQDMLSL", "SQDMULL",
			"FCVTL":
			shape = LongArngs

		// WideArngs: second input operand is half the width of first input/result.
		case "SADDW", "UADDW", "SSUBW", "USUBW":
			shape = WideArngs

		// These are not yet supported by simdgen.
		case "USDOT", "SDOT", "UDOT", "SUDOT",
			"BFDOT", "FDOT",
			"MLA", "MLS", "MUL", "PMUL",
			"FCMLA", "FCADD",
			"BFCVTN",
			"BFMLAL", "BFMMLA",
			"FMMLA", "SMMLA", "UMMLA", "USMMLA":
			return UnsupportedArngs

		// TBL/TBX have a fixed .16B on table and variable <Ta> on destination/index.
		case "TBL", "TBX":
			shape = DefaultArngs

		default:
			panic(fmt.Sprintf(
				"ArngShape: unhandled non-uniform arrangement instruction %q (symbols: %v)",
				mnemonic, symbols))
		}
	}

	if shape != instruction.regDiagramArngShape() {
		panic(fmt.Sprintf("ArngShape: cross-check failed for %q (symbols: %v): %v but regDiagram says %v",
			instruction.Mnemonic(), symbols, shape, instruction.regDiagramArngShape()))
	}

	return shape
}

// removeDuplicates removes duplicate strings from a slice
func removeDuplicates(slice []string) []string {
	keys := make(map[string]bool)
	result := []string{}

	for _, item := range slice {
		if _, value := keys[item]; !value {
			keys[item] = true
			result = append(result, item)
		}
	}

	return result
}

// parseArrangement gets element bits and lanes number from arrangement string like "4S", "2D", "16B"
func parseArrangement(arrangement string) (elemBits, bits, lanes int) {
	if len(arrangement) < 2 {
		return 0, 0, 0
	}

	lanesStr := arrangement[:len(arrangement)-1]
	elemType := arrangement[len(arrangement)-1:]

	lanes, err := strconv.Atoi(lanesStr)
	if err != nil {
		return 0, 0, 0
	}

	switch elemType {
	case "B": // Byte
		elemBits = 8
	case "H": // Halfword
		elemBits = 16
	case "S": // Single word
		elemBits = 32
	case "D": // Double word
		elemBits = 64
	default:
		return 0, 0, 0
	}

	return elemBits, elemBits * lanes, lanes
}

// asmTemplateToString converts an xmlspec.AsmTemplate structure to a string
func asmTemplateToString(template xmlspec.AsmTemplate) string {
	var result strings.Builder
	for _, content := range template.TextA {
		result.WriteString(content.Value)
	}
	return result.String()
}

// templates returns operand templates
func (instruction *Instruction) templates() []template {
	var operands []Operand
	AsmTemplate := ""
searchTemplate:
	for _, Class := range instruction.Classes.Iclass {
		for _, Encoding := range Class.Encodings {
			curAsmTemplate := asmTemplateToString(Encoding.AsmTemplate)
			if strings.Contains(curAsmTemplate, ">.") && curAsmTemplate != AsmTemplate {
				AsmTemplate = curAsmTemplate
				break searchTemplate
			}
		}
	}
	operands = instruction.operands(AsmTemplate)
	return []template{{operands: operands, instruction: instruction}}
}

// Documentation extracts detailed instruction documentation from the XML
func (instruction *Instruction) Documentation() string {
	documentation := instruction.Title
	if len(instruction.Desc.Authored.Paragraphs) > 0 {
		documentation = instruction.Desc.Authored.Paragraphs[0].Text
	}
	return documentation
}

// Brief returns the brief description from XML
func (instruction *Instruction) Brief() string {
	if len(instruction.Desc.Brief.Para) > 0 {
		return strings.TrimSpace(instruction.Desc.Brief.Para[0].Text)
	}
	return ""
}
