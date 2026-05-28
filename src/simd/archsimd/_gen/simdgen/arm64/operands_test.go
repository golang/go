// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"testing"
)

// TestOperandParsing tests operand parsing phases: tokenizeTemplate, classifyTokens, and buildOperandList.
func TestOperandParsing(t *testing.T) {
	tests := []struct {
		name         string // test case name
		template     string // assembly template string to parse
		resultInArg0 bool   // whether destination register is also an input

		// Expected after tokenizeTemplate
		wantTokenText []string // raw token strings from template

		// Expected after classifyTokens
		wantTypes  []OperandType // operand type per token
		wantIsDest []bool        // is-destination flag per token

		// Expected after buildOperandList
		wantCount    int      // number of final operands
		wantClasses  []string // operand class (vreg, greg, immediate)
		wantRoles    []string // operand role
		wantListNums []int    // ListNumber (-1 for non-list)
	}{
		{
			name:          "ADD simple binary - from add_advsimd.xml",
			template:      "ADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>",
			resultInArg0:  false,
			wantTokenText: []string{"<Vd>.<T>", "<Vn>.<T>", "<Vm>.<T>"},
			wantTypes:     []OperandType{OperandVReg, OperandVReg, OperandVReg},
			wantIsDest:    []bool{true, false, false},
			wantCount:     3,
			wantClasses:   []string{"vreg", "vreg", "vreg"},
			wantRoles:     []string{"destination", "op0", "op1"},
			wantListNums:  []int{-1, -1, -1},
		},
		{
			name:          "TBL with list operand - from tbl_advsimd.xml",
			template:      "TBL  <Vd>.8B, { <Vn>.16B, <Vn+1>.16B }, <Vm>.8B",
			resultInArg0:  false,
			wantTokenText: []string{"<Vd>.8B", "{ <Vn>.16B, <Vn+1>.16B }", "<Vm>.8B"},
			wantTypes:     []OperandType{OperandVReg, OperandList, OperandVReg},
			wantIsDest:    []bool{true, false, false},
			wantCount:     3,
			wantClasses:   []string{"vreg", "vreg", "vreg"},
			wantRoles:     []string{"destination", "op0", "op1"},
			wantListNums:  []int{-1, 0, -1},
		},
		{
			name:          "MOVI with optional LSL modifier - from movi_advsimd.xml",
			template:      "MOVI  <Vd>.4S, #<imm8>{, LSL #<amount>}",
			resultInArg0:  false,
			wantTokenText: []string{"<Vd>.4S", "#<imm8>{, LSL #<amount>}"},
			wantTypes:     []OperandType{OperandVReg, OperandImm},
			wantIsDest:    []bool{true, false},
			wantCount:     2,
			wantClasses:   []string{"vreg", "immediate"},
			wantRoles:     []string{"destination", "imm8"},
			wantListNums:  []int{-1, -1},
		},
		{
			name:          "BIC with optional LSL modifier - from bic_advsimd_imm.xml",
			template:      "BIC  <Vd>.2S, #<imm8>{, LSL #<amount>}",
			resultInArg0:  false,
			wantTokenText: []string{"<Vd>.2S", "#<imm8>{, LSL #<amount>}"},
			wantTypes:     []OperandType{OperandVReg, OperandImm},
			wantIsDest:    []bool{true, false},
			wantCount:     2,
			wantClasses:   []string{"vreg", "immediate"},
			wantRoles:     []string{"destination", "imm8"},
			wantListNums:  []int{-1, -1},
		},
		{
			name:          "INS element to element - from ins_advsimd_elt.xml",
			template:      "INS  <Vd>.4S[<index1>], <Vn>.4S[<index2>]",
			resultInArg0:  true,
			wantTokenText: []string{"<Vd>.4S[<index1>]", "<Vn>.4S[<index2>]"},
			wantTypes:     []OperandType{OperandVElem, OperandVElem},
			wantIsDest:    []bool{true, false},
			wantCount:     5,
			wantClasses:   []string{"vreg", "immediate", "immediate", "vreg", "vreg"},
			wantRoles:     []string{"destination", "destination_i", "op0_i", "original", "op0"},
			wantListNums:  []int{-1, -1, -1, -1, -1},
		},
		{
			name:          "INS general register to element - from ins_advsimd_gen.xml",
			template:      "INS  <Vd>.4S[<index1>], <Wn>",
			resultInArg0:  false, // VElem dest automatically sets resultInArg0
			wantTokenText: []string{"<Vd>.4S[<index1>]", "<Wn>"},
			wantTypes:     []OperandType{OperandVElem, OperandGReg},
			wantIsDest:    []bool{true, false},
			wantCount:     4,
			wantClasses:   []string{"vreg", "immediate", "vreg", "greg"},
			wantRoles:     []string{"destination", "destination_i", "original", "op0"},
			wantListNums:  []int{-1, -1, -1, -1},
		},
		{
			name:          "UMOV to general register - from umov_advsimd.xml",
			template:      "UMOV  <Wd>, <Vn>.4S[<index>]",
			resultInArg0:  false,
			wantTokenText: []string{"<Wd>", "<Vn>.4S[<index>]"},
			wantTypes:     []OperandType{OperandGReg, OperandVElem},
			wantIsDest:    []bool{true, false},
			wantCount:     3,
			wantClasses:   []string{"greg", "immediate", "vreg"},
			wantRoles:     []string{"destination", "op0_i", "op0"},
			wantListNums:  []int{-1, -1, -1},
		},
		{
			name:          "SHL immediate - from shl_advsimd.xml",
			template:      "SHL  <Vd>.<T>, <Vn>.<T>, #<shift>",
			resultInArg0:  false,
			wantTokenText: []string{"<Vd>.<T>", "<Vn>.<T>", "#<shift>"},
			wantTypes:     []OperandType{OperandVReg, OperandVReg, OperandImm},
			wantIsDest:    []bool{true, false, false},
			wantCount:     3,
			wantClasses:   []string{"vreg", "immediate", "vreg"},
			wantRoles:     []string{"destination", "shift", "op0"},
			wantListNums:  []int{-1, -1, -1},
		},
		{
			name:          "TBX with list (with resultInArg0) - from tbx_advsimd.xml",
			template:      "TBX  <Vd>.8B, { <Vn>.16B, <Vn+1>.16B }, <Vm>.8B",
			resultInArg0:  true,
			wantTokenText: []string{"<Vd>.8B", "{ <Vn>.16B, <Vn+1>.16B }", "<Vm>.8B"},
			wantTypes:     []OperandType{OperandVReg, OperandList, OperandVReg},
			wantIsDest:    []bool{true, false, false},
			wantCount:     4,
			wantClasses:   []string{"vreg", "vreg", "vreg", "vreg"},
			wantRoles:     []string{"destination", "original", "op0", "op1"},
			wantListNums:  []int{-1, -1, 0, -1}, // List operand is at position 2 (after "original" is inserted)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Split template string into individual operand tokens.
			tokens := tokenizeTemplate(tt.template)

			if !requireEqual(t, len(tt.wantTokenText), len(tokens)) {
				return
			}

			for i := range tokens {
				requireEqual(t, tt.wantTokenText[i], tokens[i].text)
				requireEqual(t, i, tokens[i].asmPos)
			}

			// Determine type and direction of each token.
			parsed := classifyTokens(tokens)

			if !requireEqual(t, len(tt.wantTypes), len(parsed)) {
				return
			}

			for i := range parsed {
				requireEqual(t, tt.wantTypes[i], parsed[i].operandType)
				requireEqual(t, tt.wantIsDest[i], parsed[i].isDestination)
			}

			// Build final operand list, inserting "original" if result is in arg0.
			operands := buildOperandList(parsed, tt.resultInArg0)

			if !requireEqual(t, tt.wantCount, len(operands)) {
				return
			}

			for i := range operands {
				requireEqual(t, tt.wantClasses[i], operands[i].Class)
				requireEqual(t, tt.wantRoles[i], operands[i].Role)
				requireEqual(t, tt.wantListNums[i], operands[i].ListNumber)
			}
		})
	}
}

// TestOperandInstantiateBits tests that Instantiate sets Bits correctly for INS greg operands.
func TestOperandInstantiateBits(t *testing.T) {
	tests := []struct {
		name        string
		mnemonic    string
		arrangement Arrangement
		wantBits    int
	}{
		{
			name:     "INS 16B element - 8-bit greg",
			mnemonic: "INS",
			arrangement: Arrangement{
				arrangement: "16B",
				bits:        128,
				elemBits:    8,
				lanes:       16,
				baseType:    "uint",
			},
			wantBits: 8,
		},
		{
			name:     "INS 8H element - 16-bit greg",
			mnemonic: "INS",
			arrangement: Arrangement{
				arrangement: "8H",
				bits:        128,
				elemBits:    16,
				lanes:       8,
				baseType:    "uint",
			},
			wantBits: 16,
		},
		{
			name:     "INS 4S element - 32-bit greg",
			mnemonic: "INS",
			arrangement: Arrangement{
				arrangement: "4S",
				bits:        128,
				elemBits:    32,
				lanes:       4,
				baseType:    "uint",
			},
			wantBits: 32,
		},
		{
			name:     "INS 2D element - 64-bit greg",
			mnemonic: "INS",
			arrangement: Arrangement{
				arrangement: "2D",
				bits:        128,
				elemBits:    64,
				lanes:       2,
				baseType:    "uint",
			},
			wantBits: 64,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Operand{
				Type:       OperandGReg,
				Class:      "greg",
				Role:       "op0",
				ListNumber: -1,
			}
			op.instantiate(tt.arrangement, DefaultArngs, 1, tt.mnemonic)
			requireEqual(t, tt.wantBits, op.Bits)
		})
	}
}
