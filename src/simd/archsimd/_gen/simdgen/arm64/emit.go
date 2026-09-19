// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"regexp"
	"slices"
	"strings"

	"simd/archsimd/_gen/simdgen/types"
	"simd/archsimd/_gen/unify"
)

var baseTypeRegexps = map[string]*regexp.Regexp{
	"int":   regexp.MustCompile("int"),
	"uint":  regexp.MustCompile("uint"),
	"float": regexp.MustCompile("float"),
}

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

// encode generates the types.Operand representation of this operand
func (op *Operand) encode() types.Operand {
	out := types.Operand{
		Class:  op.Class,
		AsmPos: op.AsmPos,
	}
	if op.BaseType != "" {
		if re, ok := baseTypeRegexps[op.BaseType]; ok {
			out.EncodeBase = re
		} else {
			out.EncodeBase = regexp.MustCompile(op.BaseType)
		}
	}
	if op.Bits > 0 {
		out.EncodeBits = &types.VectorSize{NRaw: op.Bits}
	}
	if op.ElemBits > 0 {
		out.ElemBits = new(op.ElemBits)
	}
	if op.Lanes > 0 {
		out.Lanes = new(op.Lanes)
	}
	if op.Type == OperandImm {
		out.EncodeBits = &types.VectorSize{NRaw: 8}
		if op.ImmMax == 0 {
			out.Const = new("0")
		} else {
			out.ImmOffset = new("0")
		}
		if op.ImmMax > 0 {
			out.ImmMax = new(op.ImmMax)
		}
	}
	if op.ListNumber >= 0 {
		out.ListNumber = new(op.ListNumber)
	}
	return out
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

	var in, out []types.Operand
	for _, op := range template.operands {
		opGen := op.encode()
		if op.Role == "destination" {
			out = append(out, opGen)
		} else {
			in = append(in, opGen)
		}
	}

	slices.SortStableFunc(in, types.Operand.Compare)

	db.Add("in", unify.Encode(in))
	db.Add("out", unify.Encode(out))

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
