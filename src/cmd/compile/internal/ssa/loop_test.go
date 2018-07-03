// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"testing"
)

func TestLoopConditionS390X(t *testing.T) {
	// Test that a simple loop condition does not generate a conditional
	// move (issue #19227).
	//
	// MOVDLT is generated when Less64 is lowered but should be
	// optimized into an LT branch.
	//
	// For example, compiling the following loop:
	//
	//   for i := 0; i < N; i++ {
	//     sum += 3
	//   }
	//
	// should generate assembly similar to:
	//   loop:
	//     CMP    R0, R1
	//     BGE    done
	//     ADD    $3, R4
	//     ADD    $1, R1
	//     BR     loop
	//   done:
	//
	// rather than:
	// loop:
	//     MOVD   $0, R2
	//     MOVD   $1, R3
	//     CMP    R0, R1
	//     MOVDLT R2, R3
	//     CMPW   R2, $0
	//     BNE    done
	//     ADD    $3, R4
	//     ADD    $1, R1
	//     BR     loop
	//   done:
	//
	c := testConfigS390X(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("SP", OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("ret", OpLocalAddr, c.config.Types.Int64.PtrTo(), 0, nil, "SP", "mem"),
			Valu("N", OpArg, c.config.Types.Int64, 0, c.Frontend().Auto(src.NoXPos, c.config.Types.Int64)),
			Valu("starti", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("startsum", OpConst64, c.config.Types.Int64, 0, nil),
			Goto("b1")),
		Bloc("b1",
			Valu("phii", OpPhi, c.config.Types.Int64, 0, nil, "starti", "i"),
			Valu("phisum", OpPhi, c.config.Types.Int64, 0, nil, "startsum", "sum"),
			Valu("cmp1", OpLess64, c.config.Types.Bool, 0, nil, "phii", "N"),
			If("cmp1", "b2", "b3")),
		Bloc("b2",
			Valu("c1", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("i", OpAdd64, c.config.Types.Int64, 0, nil, "phii", "c1"),
			Valu("c3", OpConst64, c.config.Types.Int64, 3, nil),
			Valu("sum", OpAdd64, c.config.Types.Int64, 0, nil, "phisum", "c3"),
			Goto("b1")),
		Bloc("b3",
			Valu("retdef", OpVarDef, types.TypeMem, 0, nil, "mem"),
			Valu("store", OpStore, types.TypeMem, 0, c.config.Types.Int64, "ret", "phisum", "retdef"),
			Exit("store")))
	CheckFunc(fun.f)
	Compile(fun.f)
	CheckFunc(fun.f)

	checkOpcodeCounts(t, fun.f, map[Op]int{
		OpS390XMOVDLT:    0,
		OpS390XMOVDGT:    0,
		OpS390XMOVDLE:    0,
		OpS390XMOVDGE:    0,
		OpS390XMOVDEQ:    0,
		OpS390XMOVDNE:    0,
		OpS390XCMP:       1,
		OpS390XCMPWconst: 0,
	})
}
