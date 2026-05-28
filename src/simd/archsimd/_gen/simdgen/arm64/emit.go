// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"fmt"
	"strings"

	"_gen/unify"
)

// asComment formats text as a comment
func asComment(text string, width int) string {
	text = strings.TrimSpace(text)
	text = strings.ReplaceAll(text, "&amp;", "&")
	text = strings.ReplaceAll(text, "\n", " ")
	words := strings.Fields(text)
	var lines []string
	line := ""
	for _, w := range words {
		if line != "" {
			line = line + " "
		}
		line = line + w
		if len(line) >= width {
			lines = append(lines, "// "+line)
			line = ""
		}
	}
	if len(line) > 0 {
		lines = append(lines, "// "+line)
	}
	return strings.Join(lines, "\n")
}

// Emit generates the unify.Value representation of this operand
func (op *Operand) Emit() *unify.Value {
	var opDb unify.DefBuilder
	opDb.Add("class", unify.NewValue(unify.NewStringExact(op.Class)))

	if op.BaseType != "" {
		opDb.Add("base", unify.NewValue(unify.NewStringExact(op.BaseType)))
	}

	if op.Bits > 0 {
		opDb.Add("bits", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.Bits))))
	}

	if op.ElemBits > 0 {
		opDb.Add("elemBits", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.ElemBits))))
	}

	if op.Lanes > 0 {
		opDb.Add("lanes", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.Lanes))))
	}

	if op.Type == OperandImm {
		opDb.Add("bits", unify.NewValue(unify.NewStringExact("8")))
		if op.ImmMax == 0 {
			opDb.Add("const", unify.NewValue(unify.NewStringExact("0")))
		} else {
			opDb.Add("immOffset", unify.NewValue(unify.NewStringExact("0")))
		}
		if op.ImmMax > 0 {
			opDb.Add("immMax", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.ImmMax))))
		}
	}

	if op.Role != "" {
		opDb.Add("role", unify.NewValue(unify.NewStringExact(op.Role)))
	}

	if op.ListNumber >= 0 {
		opDb.Add("listNumber", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.ListNumber))))
	}

	opDb.Add("asmPos", unify.NewValue(unify.NewStringExact(fmt.Sprint(op.AsmPos))))

	return unify.NewValue(opDb.Build())
}

// Emit generates a single instruction Definition for the given arrangement.
func (template *template) Emit(arrangement string) *unify.Value {
	var db unify.DefBuilder

	// Map mnemonic to Go assembly
	mnemonic := template.instruction.Mnemonic()
	switch mnemonic {
	case "INS", "UMOV":
		arrangement = arrangement[len(arrangement)-1:]
		mnemonic = "VMOV"
	case "DUP":
		arrangement = arrangement[len(arrangement)-1:]
		mnemonic = "V" + mnemonic
	default:
		// AES and SHA instructions do not use "V" prefix (assembler compatibility)
		// Match SHA followed by digit (SHA1, SHA256, SHA512) but not SHADD
		isAESOrSHA := strings.HasPrefix(mnemonic, "AES") ||
			(len(mnemonic) > 3 && mnemonic[:3] == "SHA" && mnemonic[3] >= '0' && mnemonic[3] <= '9')
		if !isAESOrSHA {
			mnemonic = "V" + mnemonic
		}
	}

	db.Add("asm", unify.NewValue(unify.NewStringExact(mnemonic)))
	db.Add("arrangement", unify.NewValue(unify.NewStringExact(arrangement)))
	db.Add("goarch", unify.NewValue(unify.NewStringExact("arm64")))
	db.Add("cpuFeature", unify.NewValue(unify.NewStringExact("NEON"))) // TODO: features
	db.Add("inVariant", unify.NewValue(unify.NewTuple()))

	if doc := template.instruction.Documentation(); doc != "" {
		db.Add("details", unify.NewValue(unify.NewStringExact(asComment(doc, 80))))
	}

	var inVals, outVals []*unify.Value
	for _, op := range template.operands {
		if op.Role == "destination" {
			outVals = append(outVals, op.Emit())
		} else {
			inVals = append(inVals, op.Emit())
		}
	}

	db.Add("in", unify.NewValue(unify.NewTuple(inVals...)))
	db.Add("out", unify.NewValue(unify.NewTuple(outVals...)))

	return unify.NewValue(db.Build())
}

// EmitAll generates instruction definitions for all arrangements of this instruction.
// Returns nil for instructions with UnsupportedArngs.
func (instruction *Instruction) EmitAll() []*unify.Value {
	var defs []*unify.Value

	mnemonic := instruction.Mnemonic()
	templates := instruction.templates()
	arrangements, ashape := instruction.Arrangements()
	if ashape == UnsupportedArngs {
		return nil
	}
	for _, template := range templates {
		for _, arr := range arrangements {
			// Clone template so each arrangement gets its own operand slice.
			updatedTemplate := template
			updatedTemplate.operands = make([]Operand, len(template.operands))
			copy(updatedTemplate.operands, template.operands)

			// Instantiate operands for this arrangement.
			// Most instructions (DefaultArngs) stamp the same arrangement into all vreg operands.
			// Special shapes (NarrowArngs, LongArngs, WideArngs) adjust only certain operands
			// (e.g. only the result, or only the second input) to half or double width.
			vregPos := 0
			for i := range updatedTemplate.operands {
				updatedTemplate.operands[i].instantiate(arr, ashape, vregPos, mnemonic)
				if updatedTemplate.operands[i].Type == OperandVReg {
					vregPos++
				}
			}
			defs = append(defs, updatedTemplate.Emit(arr.arrangement))
		}
	}

	return defs
}
