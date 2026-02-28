// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

// ARM64Lt specifies BlockARM64LT
func ARM64Lt(cond, sub, alt string) ctrl {
	return ctrl{BlockARM64LT, cond, []string{sub, alt}}
}

// ARM64Gt specifies BlockARM64GT
func ARM64Gt(cond, sub, alt string) ctrl {
	return ctrl{BlockARM64GT, cond, []string{sub, alt}}
}

// ARM64Ne specifies BlockARM64NE
func ARM64Ne(cond, sub, alt string) ctrl {
	return ctrl{BlockARM64NE, cond, []string{sub, alt}}
}

// ARM64Eq specifies BlockARM64EQ
func ARM64Eq(cond, sub, alt string) ctrl {
	return ctrl{BlockARM64EQ, cond, []string{sub, alt}}
}

// isNewConditionCorrect verifies that a block has been correctly transformed
// to use conditional comparison (CCMP) with the expected parameters.
// It checks:
// - The block kind is BlockARM64LT (less than condition)
// - The control operation is OpARM64CCMPconst (conditional comparison with constant)
// - The condition code is OpARM64GreaterThan
// - The NZCV flags are set to 1
// - The constant value being compared is 4
// Returns true if all conditions match the expected transformation pattern
func isNewConditionCorrect(b *Block) bool {
	if b.Kind != BlockARM64LT {
		return false
	}

	v := b.Controls[0]
	if v.Op != OpARM64CCMPconst {
		return false
	}

	params := v.AuxArm64ConditionalParams()
	if params.Cond() != OpARM64GreaterThan {
		return false
	}
	if params.Nzcv() != 1 {
		// NZCV flags should be set to 1 for this specific transformation
		return false
	}
	if imm, ok := params.ConstValue(); !ok || imm != 4 {
		return false
	}

	return true
}

// containsOpARM64CCMP checks if a block contains any ARM64 conditional comparison
// operations (CCMP or CCMPconst). This is used in tests to verify that the
// if-conversion optimization successfully generated conditional comparison
// instructions or to ensure they were not generated when inappropriate.
func containsOpARM64CCMP(b *Block) bool {
	for _, v := range b.Values {
		if v.Op == OpARM64CCMP || v.Op == OpARM64CCMPconst {
			return true
		}
	}
	return false
}

// TestMergeConditionalBranchesWithoutPointers tests the if-conversion optimization
// on a simple case of logical AND (cond1 && cond2) without pointer operations.
// The test verifies that:
// - The optimization correctly transforms nested conditionals into CCMP instructions
// - The block structure is properly simplified (inner block becomes plain and empty)
// - The resulting control flow uses conditional comparison with correct parameters
// - No important blocks are accidentally deleted during transformation
// This represents the ideal case where the optimization should apply successfully.
func TestMergeConditionalBranchesWithoutPointers(t *testing.T) {
	t.Run("arm64", func(t *testing.T) {
		c := testConfigArch(t, "arm64")
		intType := c.config.Types.Int64
		fun := c.Fun("entry",
			Bloc("entry",
				Valu("mem",
					OpInitMem,
					types.TypeMem,
					0, nil,
				),
				Valu("a",
					OpArg,
					intType,
					0, c.Temp(intType),
				),
				Valu("b",
					OpArg,
					intType,
					1, c.Temp(intType),
				),
				Valu("cond1",
					OpARM64CMPconst,
					types.TypeFlags,
					1, nil,
					"a",
				),
				ARM64Gt("cond1", "second_comparison", "ret_false"),
			),
			Bloc("second_comparison",
				Valu("cond2",
					OpARM64CMPconst,
					types.TypeFlags,
					4, nil,
					"b",
				),
				ARM64Lt("cond2", "ret_false", "ret_true"),
			),
			Bloc("ret_true",
				Valu("const1",
					OpARM64MOVDconst,
					intType,
					1, nil,
				),
				Valu("true_result",
					OpMakeResult,
					types.TypeMem,
					0, nil,
					"const1", "mem",
				),
				Ret("true_result"),
			),
			Bloc("ret_false",
				Valu("const0",
					OpARM64MOVDconst,
					intType,
					0, nil,
				),
				Valu("false_result",
					OpMakeResult,
					types.TypeMem,
					0, nil,
					"const0", "mem",
				),
				Ret("false_result"),
			),
		)

		CheckFunc(fun.f)
		mergeConditionalBranches(fun.f)
		CheckFunc(fun.f)

		if len(fun.blocks) != 4 {
			t.Errorf("Important block was deleted")
		}

		entryBlock := fun.blocks["entry"]
		secondBlock := fun.blocks["second_comparison"]

		if secondBlock.Kind != BlockPlain || len(secondBlock.Values) != 0 {
			t.Errorf("Block with second condition wasn't cleaned")
		}

		if !isNewConditionCorrect(entryBlock) {
			t.Errorf("Entry block doesn't contain CCMP opertation")
		}
	})
}

// Test that pointer comparison with memory load doesn't generate CCMP
func TestNoCCMPWithPointerAndMemoryLoad(t *testing.T) {
	t.Run("arm64", func(t *testing.T) {
		c := testConfigArch(t, "arm64")
		intType := c.config.Types.Int64
		ptrType := c.config.Types.BytePtr

		fun := c.Fun("entry",
			Bloc("entry",
				Valu("mem",
					OpInitMem,
					types.TypeMem,
					0, nil,
				),
				Valu("ptr",
					OpArg,
					ptrType,
					0, c.Temp(ptrType),
				),
				Valu("cond1",
					OpARM64CMPconst,
					types.TypeFlags,
					0, nil, // Compare with nil (0)
					"ptr",
				),
				ARM64Ne("cond1", "second_comparison", "ret_false"), // ptr != nil
			),
			Bloc("second_comparison",
				Valu("load",
					OpLoad,
					intType,
					0, nil,
					"ptr", "mem",
				),
				Valu("cond2",
					OpARM64CMPconst,
					types.TypeFlags,
					3, nil, // Compare with 3
					"load",
				),
				ARM64Eq("cond2", "ret_true", "ret_false"), // *ptr == 3
			),
			Bloc("ret_true",
				Valu("const1",
					OpARM64MOVDconst,
					intType,
					1, nil,
				),
				Valu("true_result",
					OpMakeResult,
					types.TypeMem,
					0, nil,
					"const1", "mem",
				),
				Ret("true_result"),
			),
			Bloc("ret_false",
				Valu("const0",
					OpARM64MOVDconst,
					intType,
					0, nil,
				),
				Valu("false_result",
					OpMakeResult,
					types.TypeMem,
					0, nil,
					"const0", "mem",
				),
				Ret("false_result"),
			),
		)

		CheckFunc(fun.f)
		mergeConditionalBranches(fun.f)
		CheckFunc(fun.f)

		// Verify that the second_comparison block still exists (not optimized away)
		if fun.blocks["second_comparison"] == nil {
			t.Errorf("Second comparison block was incorrectly removed")
		}

		entryBlock := fun.blocks["entry"]
		secondBlock := fun.blocks["second_comparison"]

		// Verify that entry block doesn't contain CCMP operation
		if containsOpARM64CCMP(entryBlock) {
			t.Errorf("Entry block contains CCMP operation, but shouldn't due to memory load")
		}

		// Verify that second block contains the load operation
		hasLoad := false
		for _, v := range secondBlock.Values {
			if v.Op == OpLoad {
				hasLoad = true
				break
			}
		}
		if !hasLoad {
			t.Errorf("Second comparison block should contain load operation")
		}

		// The optimization shouldn't merge these blocks because of the memory operation
		if secondBlock.Kind == BlockPlain {
			t.Errorf("Block with memory load was incorrectly cleaned")
		}
	})
}
