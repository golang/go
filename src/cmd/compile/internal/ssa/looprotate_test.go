// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestLoopRotateNested(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("constTrue", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("outerHeader")),
		Bloc("outerHeader",
			If("constTrue", "outerBody", "outerExit")),
		Bloc("outerBody",
			Goto("innerHeader")),
		Bloc("innerHeader",
			If("constTrue", "innerBody", "innerExit")),
		Bloc("innerBody",
			Goto("innerTop")),
		Bloc("innerTop",
			Goto("innerHeader")),
		Bloc("innerExit",
			Goto("outerTop")),
		Bloc("outerTop",
			Goto("outerHeader")),
		Bloc("outerExit",
			Exit("mem")))

	blockName := make([]string, len(fun.f.Blocks)+1)
	for name, block := range fun.blocks {
		blockName[block.ID] = name
	}

	CheckFunc(fun.f)
	loopRotate(fun.f)
	CheckFunc(fun.f)

	// Verify the resulting block order
	expected := []string{
		"entry",
		"outerTop",
		"outerHeader",
		"outerBody",
		"innerTop",
		"innerHeader",
		"innerBody",
		"innerExit",
		"outerExit",
	}
	if len(expected) != len(fun.f.Blocks) {
		t.Fatalf("expected %d blocks, found %d", len(expected), len(fun.f.Blocks))
	}
	for i, b := range fun.f.Blocks {
		if expected[i] != blockName[b.ID] {
			t.Errorf("position %d: expected %s, found %s", i, expected[i], blockName[b.ID])
		}
	}
}
