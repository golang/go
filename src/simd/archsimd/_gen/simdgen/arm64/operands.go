// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"fmt"
	"strings"
)

// OperandType defines the type of an operand for ARM64 instruction generation.
type OperandType int

const (
	OperandVReg  OperandType = iota // Vector register
	OperandGReg                     // General register
	OperandImm                      // Immediate
	OperandVElem                    // Vector element (e.g., <Vm>.H[<index>]): early-lowered into immediate + vreg with same AsmPos
	OperandList                     // List operand (e.g., { <Vn>.16B, <Vn+1>.16B }): early-lowered into vreg with ListNumber
)

func (t OperandType) String() string {
	switch t {
	case OperandVReg:
		return "VReg"
	case OperandGReg:
		return "GReg"
	case OperandImm:
		return "Imm"
	case OperandVElem:
		return "VElem"
	case OperandList:
		return "List"
	default:
		return "Unknown"
	}
}

// Operand represents an arm64 instruction operand instantiated for concrete arrangement.
type Operand struct {
	Type     OperandType
	Class    string // "vreg", "greg", "immediate"
	BaseType string // Base type ("int", "uint", "float")
	ElemBits int    // Element bits (for vectors)
	Bits     int    // Total bits
	Lanes    int    // Number of lanes (for vectors)
	ImmMax   int    // Immediate max value (for immediates)
	// The operand's role. Possible values:
	//   - "destination":      the output register
	//   - "original":         the original SSA value of "destination" (for resultInArg0 instructions)
	//   - ends with "_i":     vector element index: should get ImmMax = lanes-1
	//   - "op0", "op1", ...:  input registers
	//   - other strings:      immediate names (e.g. "immshift", "amount", "immzero")
	Role       string
	ListNumber int // List number for register list (e.g., useful for TBL/TBX instructions)
	AsmPos     int // Assembly position (usually 0 for the destination register, 1+ for inputs).
	// Input with AsmPos == 0 represents original value of the destination register for ssa form.
	// Immediate with AsmPos == subsequent register operand's AsmPos represents a vector element (the immediate specifies the lane number).
}

// token represents a raw operand string from the assembly template.
type token struct {
	text   string // Raw operand text (e.g., "<Vd>.16B", "{ <Vn>.16B, <Vn+1>.16B }")
	asmPos int    // Position in assembly syntax (0 for destination, 1+ for inputs)
}

// parsedOperand represents a classified but not lowered operand.
type parsedOperand struct {
	token                     // Embedded token with text and position
	operandType   OperandType // Operand type (VReg, GReg, Imm, VElem, List)
	isDestination bool        // True if this is a destination operand
	immName       string      // Immediate role (for imm operands)
	immMax        int         // Maximum immediate value (for imm operands)
}

// instantiate updates the operand's type information based on the given arrangement and instruction mnemonic.
// This is used when generating instruction definitions for specific vector arrangements.
func (op *Operand) instantiate(arrangement Arrangement, ashape ArngShape, vregPos int, mnemonic string) {
	switch op.Type {
	case OperandVReg:
		switch {
		case ashape == NarrowArngs && vregPos == 0:
			op.ElemBits = arrangement.elemBits / 2
			op.Bits = arrangement.bits
			op.Lanes = arrangement.bits / (arrangement.elemBits / 2)
		case ashape == LongArngs && vregPos == 0:
			op.ElemBits = arrangement.elemBits * 2
			op.Bits = arrangement.bits * 2
			if op.Bits > 128 {
				op.Bits = 128
			}
			op.Lanes = arrangement.bits / op.ElemBits
		case ashape == WideArngs && vregPos == 2:
			op.ElemBits = arrangement.elemBits / 2
			op.Bits = arrangement.bits
			op.Lanes = arrangement.bits / op.ElemBits
		default:
			op.ElemBits = arrangement.elemBits
			op.Bits = arrangement.bits
			op.Lanes = arrangement.lanes
		}
		op.BaseType = arrangement.baseType
	case OperandImm:
		// Update immediate operands based on arrangement
		// For shift operations, set immediate max to element_bits - 1 (sometimes need element_bits?)
		// For vector element indices, immediate max should be lanes - 1
		if op.ImmMax == -1 {
			// Check if this is a vector element immediate (role ends with "_i")
			if strings.HasSuffix(op.Role, "_i") {
				// Vector element index: max = lanes - 1
				op.ImmMax = arrangement.lanes - 1
			} else if mnemonic == "EXT" {
				// EXT byte index: max = register size in bytes - 1 = 15 for 128-bit
				op.ImmMax = arrangement.bits/8 - 1
			} else if ashape == NarrowArngs {
				// Narrow shift: max = destination element bits - 1 (half of source)
				op.ImmMax = arrangement.elemBits/2 - 1
			} else {
				// Shift operation: max = element_bits - 1
				op.ImmMax = arrangement.elemBits - 1
			}
		}
	case OperandGReg:
		op.BaseType = arrangement.baseType
		if mnemonic == "UMOV" {
			// Update general register width based on arrangement
			// - 2D arrangement needs 64-bit (X register)
			// - All other arrangements need 32-bit (W register)
			if arrangement.arrangement == "2D" {
				op.Bits = 64
			} else {
				op.Bits = 32
			}
		}
		if mnemonic == "INS" {
			op.Bits = arrangement.elemBits
		}
	case OperandVElem, OperandList:
		panic("expected this operand type to be early-lowered")
	}
}

// operands extracts operand information from the assembly template.
func (instruction *Instruction) operands(asmTemplate string) []Operand {
	tokens := tokenizeTemplate(asmTemplate)
	parsed := classifyTokens(tokens)
	return buildOperandList(parsed, instruction.ResultInArg0())
}

// tokenizeTemplate parses an assembly template string into a list of operand tokens.
// It handles:
// - Stripping the mnemonic from the first operand
// - Register list operands like "{ <Vn>.16B, <Vn+1>.16B }" (preserves internal commas)
// - Optional modifiers like "{, LSL #<amount>}" (merged with previous operand)
func tokenizeTemplate(template string) []token {
	template = stripMnemonic(template)
	parts := strings.Split(template, ",")

	var tokens []token
	var listBuf strings.Builder
	var inList bool

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Register list: { <Vn>.16B, <Vn+1>.16B }
		if isListStart(part) {
			inList = true
			listBuf.WriteString(part)
			if strings.Contains(part, "}") {
				tokens = append(tokens, token{text: listBuf.String(), asmPos: len(tokens)})
				listBuf.Reset()
				inList = false
			}
			continue
		}
		if inList {
			listBuf.WriteString(", ")
			listBuf.WriteString(part)
			if strings.HasSuffix(part, "}") {
				tokens = append(tokens, token{text: listBuf.String(), asmPos: len(tokens)})
				listBuf.Reset()
				inList = false
			}
			continue
		}

		// Optional modifier: previous ends with "{" or current starts with "{"
		if shouldMergeWithPrevious(part, tokens) {
			tokens[len(tokens)-1].text += ", " + part
			continue
		}

		tokens = append(tokens, token{text: part, asmPos: len(tokens)})
	}
	return tokens
}

// stripMnemonic removes the instruction mnemonic from the template string.
// For example, "ADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>" becomes "<Vd>.<T>, <Vn>.<T>, <Vm>.<T>".
func stripMnemonic(template string) string {
	if idx := strings.Index(template, " "); idx >= 0 {
		return strings.TrimSpace(template[idx+1:])
	}
	return template
}

// isListStart checks if a part is the start of a register list operand.
// Register lists start with "{" and contain a vector register like "<V".
func isListStart(part string) bool {
	return strings.HasPrefix(part, "{") && strings.Contains(part, "<V")
}

// shouldMergeWithPrevious determines if the current part should be merged with the previous token.
// This happens for optional modifiers like "{, LSL #<amount>}" that follow an operand.
func shouldMergeWithPrevious(part string, tokens []token) bool {
	if len(tokens) == 0 {
		return false
	}
	return strings.HasPrefix(part, "{") || strings.HasSuffix(tokens[len(tokens)-1].text, "{")
}

// classifyTokens analyzes each token and determines its operand type and role.
// It returns a slice of parsedOperand with type information and metadata.
func classifyTokens(tokens []token) []parsedOperand {
	parsed := make([]parsedOperand, len(tokens))
	for i, tok := range tokens {
		opType, isDest, immName, immMax := analyzeOperand(tok.text)
		parsed[i] = parsedOperand{
			token:         tok,
			operandType:   opType,
			isDestination: isDest,
			immName:       immName,
			immMax:        immMax,
		}
	}
	return parsed
}

// buildOperandList constructs the final ordered list of operands from parsed operands.
// It lowers compound operands (VElem, List) and orders them as: outputs + immediates + [original] + inputs.
func buildOperandList(parsed []parsedOperand, resultInArg0 bool) []Operand {
	var outs, ins, imms []Operand
	inputCount := 0

	for _, p := range parsed {
		switch p.operandType {
		case OperandVElem:
			idx, reg := lowerVElem(p, &inputCount)
			imms = append(imms, idx)
			if p.isDestination {
				outs = append(outs, reg)
				resultInArg0 = true // element dest implies read-modify-write
			} else {
				ins = append(ins, reg)
			}

		case OperandList:
			ins = append(ins, lowerList(p, inputCount))
			inputCount++

		case OperandImm:
			imms = append(imms, makeImm(p, inputCount))
			inputCount++

		case OperandVReg, OperandGReg:
			op := makeReg(p, p.operandType, inputCount)
			if p.isDestination {
				outs = append(outs, op)
			} else {
				ins = append(ins, op)
				inputCount++
			}
		}
	}

	// Assemble final order: outs + imms + [original] + ins
	result := append(outs, imms...)
	if resultInArg0 && len(outs) > 0 {
		original := outs[0]
		original.Role = "original"
		result = append(result, original)
	}
	return append(result, ins...)
}

// lowerVElem expands a vector element operand into an index immediate and a vector register.
// For destination elements, both get AsmPos=0. For source elements, they share the original AsmPos.
func lowerVElem(p parsedOperand, inputCount *int) (idx Operand, reg Operand) {
	if p.isDestination {
		idx = Operand{
			Type: OperandImm, Class: "immediate",
			Role: "destination_i", AsmPos: 0, ListNumber: -1, ImmMax: -1,
		}
		reg = Operand{
			Type: OperandVReg, Class: "vreg",
			Role: "destination", AsmPos: 0, ListNumber: -1,
		}
	} else {
		role := inputRole(*inputCount)
		idx = Operand{
			Type: OperandImm, Class: "immediate",
			Role: role + "_i", AsmPos: p.token.asmPos, ListNumber: -1, ImmMax: -1,
		}
		reg = Operand{
			Type: OperandVReg, Class: "vreg",
			Role: role, AsmPos: p.token.asmPos, ListNumber: -1,
		}
		*inputCount++
	}
	return
}

// lowerList expands a list operand into a vector register with ListNumber=0.
func lowerList(p parsedOperand, inputCount int) Operand {
	return Operand{
		Type: OperandVReg, Class: "vreg",
		Role: inputRole(inputCount), AsmPos: p.token.asmPos, ListNumber: 0,
	}
}

// makeImm creates an immediate operand from a parsed operand.
func makeImm(p parsedOperand, inputCount int) Operand {
	return Operand{
		Type: OperandImm, Class: "immediate",
		Role: p.immName, AsmPos: p.token.asmPos, ListNumber: -1, ImmMax: p.immMax,
	}
}

// makeReg creates a register operand (vector or general) from a parsed operand.
func makeReg(p parsedOperand, opType OperandType, inputCount int) Operand {
	class := "vreg"
	if opType == OperandGReg {
		class = "greg"
	}
	role := "destination"
	if !p.isDestination {
		role = inputRole(inputCount)
	}
	return Operand{
		Type: opType, Class: class,
		Role: role, AsmPos: p.token.asmPos, ListNumber: -1,
	}
}

// inputRole generates a role name for an input operand at the given index: "op0", "op1", "op2", etc.
// TODO: consider extracting these names from ARM64 register suffixes in templates.
func inputRole(index int) string {
	return fmt.Sprintf("op%d", index)
}

// extractImmediateInfo extracts immediate name and immMax value from immediate operand strings.
func extractImmediateInfo(operandStr string) (string, int) {
	if strings.Contains(operandStr, "#<") {
		// Immediate operand: #<immediate_name>
		// Extract the immediate name from between #< and >
		start := strings.Index(operandStr, "#<") + 2
		end := strings.Index(operandStr[start:], ">")
		if end >= 0 {
			immediateName := operandStr[start : start+end]
			return immediateName, -1
		}
		return "immediate", -1
	}
	if strings.Contains(operandStr, "#0") {
		return "immzero", 0
	}
	return "", 0
}

// detectOperandType determines the basic operand type (VReg, GReg, Imm) based on its string pattern.
func detectOperandType(operandStr string) OperandType {
	switch {
	case strings.HasPrefix(operandStr, "<V"):
		return OperandVReg
	case strings.HasPrefix(operandStr, "<W") || strings.HasPrefix(operandStr, "<X") || strings.HasPrefix(operandStr, "<R"):
		return OperandGReg
	case strings.Contains(operandStr, "#<") || strings.Contains(operandStr, "#0"):
		return OperandImm
	default:
		return OperandVReg
	}
}

// analyzeOperand analyzes an operand string and returns the operand type, whether it's a destination, and the immediate name and immMax (if any).
func analyzeOperand(operandStr string) (OperandType, bool, string, int) {
	opType := detectOperandType(operandStr)
	isDestination := strings.Contains(operandStr, "d>")
	switch opType {
	case OperandVReg:
		if strings.HasPrefix(operandStr, "{") && strings.HasSuffix(operandStr, "}") {
			return OperandList, false, "", 0
		}
		if strings.Contains(operandStr, "[<index") {
			return OperandVElem, isDestination, "", 0
		}
	case OperandImm:
		immediateName, immMax := extractImmediateInfo(operandStr)
		return opType, false, immediateName, immMax
	}
	return opType, isDestination, "", 0
}
