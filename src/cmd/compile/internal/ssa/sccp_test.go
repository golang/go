// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"strings"
	"testing"
)

func TestSCCPBasic(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("b1",
		Bloc("b1",
			Valu("mem", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("v1", ssaop.OpConst64, c.config.Types.Int64, 20, nil),
			Valu("v2", ssaop.OpConst64, c.config.Types.Int64, 21, nil),
			Valu("v3", ssaop.OpConst64F, c.config.Types.Float64, 21.0, nil),
			Valu("v4", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("t1", ssaop.OpAdd64, c.config.Types.Int64, 0, nil, "v1", "v2"),
			Valu("t2", ssaop.OpDiv64, c.config.Types.Int64, 0, nil, "t1", "v1"),
			Valu("t3", ssaop.OpAdd64, c.config.Types.Int64, 0, nil, "t1", "t2"),
			Valu("t4", ssaop.OpSub64, c.config.Types.Int64, 0, nil, "t3", "v2"),
			Valu("t5", ssaop.OpMul64, c.config.Types.Int64, 0, nil, "t4", "v2"),
			Valu("t6", ssaop.OpMod64, c.config.Types.Int64, 0, nil, "t5", "v2"),
			Valu("t7", ssaop.OpAnd64, c.config.Types.Int64, 0, nil, "t6", "v2"),
			Valu("t8", ssaop.OpOr64, c.config.Types.Int64, 0, nil, "t7", "v2"),
			Valu("t9", ssaop.OpXor64, c.config.Types.Int64, 0, nil, "t8", "v2"),
			Valu("t10", ssaop.OpNeg64, c.config.Types.Int64, 0, nil, "t9"),
			Valu("t11", ssaop.OpCom64, c.config.Types.Int64, 0, nil, "t10"),
			Valu("t12", ssaop.OpNeg64, c.config.Types.Int64, 0, nil, "t11"),
			Valu("t13", ssaop.OpFloor, c.config.Types.Float64, 0, nil, "v3"),
			Valu("t14", ssaop.OpSqrt, c.config.Types.Float64, 0, nil, "t13"),
			Valu("t15", ssaop.OpCeil, c.config.Types.Float64, 0, nil, "t14"),
			Valu("t16", ssaop.OpTrunc, c.config.Types.Float64, 0, nil, "t15"),
			Valu("t17", ssaop.OpRoundToEven, c.config.Types.Float64, 0, nil, "t16"),
			Valu("t18", ssaop.OpTrunc64to32, c.config.Types.Int64, 0, nil, "t12"),
			Valu("t19", ssaop.OpCvt64Fto64, c.config.Types.Float64, 0, nil, "t17"),
			Valu("t20", ssaop.OpCtz64, c.config.Types.Int64, 0, nil, "v2"),
			Valu("t21", ssaop.OpSlicemask, c.config.Types.Int64, 0, nil, "t20"),
			Valu("t22", ssaop.OpIsNonNil, c.config.Types.Int64, 0, nil, "v2"),
			Valu("t23", ssaop.OpNot, c.config.Types.Bool, 0, nil, "v4"),
			Valu("t24", ssaop.OpEq64, c.config.Types.Bool, 0, nil, "v1", "v2"),
			Valu("t25", ssaop.OpLess64, c.config.Types.Bool, 0, nil, "v1", "v2"),
			Valu("t26", ssaop.OpLeq64, c.config.Types.Bool, 0, nil, "v1", "v2"),
			Valu("t27", ssaop.OpEqB, c.config.Types.Bool, 0, nil, "v4", "v4"),
			Valu("t28", ssaop.OpLsh64x64, c.config.Types.Int64, 0, nil, "v2", "v1"),
			Valu("t29", ssaop.OpIsInBounds, c.config.Types.Int64, 0, nil, "v2", "v1"),
			Valu("t30", ssaop.OpIsSliceInBounds, c.config.Types.Int64, 0, nil, "v2", "v1"),
			Goto("b2")),
		Bloc("b2",
			Exit("mem")))
	sccp(fun.f)
	CheckFunc(fun.f)
	for name, value := range fun.values {
		if strings.HasPrefix(name, "t") {
			if !isConst(value) {
				t.Errorf("Must be constant: %v", value.LongString())
			}
		}
	}
}

func TestSCCPIf(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("b1",
		Bloc("b1",
			Valu("mem", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("v1", ssaop.OpConst64, c.config.Types.Int64, 0, nil),
			Valu("v2", ssaop.OpConst64, c.config.Types.Int64, 1, nil),
			Valu("cmp", ssaop.OpLess64, c.config.Types.Bool, 0, nil, "v1", "v2"),
			If("cmp", "b2", "b3")),
		Bloc("b2",
			Valu("v3", ssaop.OpConst64, c.config.Types.Int64, 3, nil),
			Goto("b4")),
		Bloc("b3",
			Valu("v4", ssaop.OpConst64, c.config.Types.Int64, 4, nil),
			Goto("b4")),
		Bloc("b4",
			Valu("merge", ssaop.OpPhi, c.config.Types.Int64, 0, nil, "v3", "v4"),
			Exit("mem")))
	sccp(fun.f)
	CheckFunc(fun.f)
	for _, b := range fun.blocks {
		for _, v := range b.Values {
			if v == fun.values["merge"] {
				if !isConst(v) {
					t.Errorf("Must be constant: %v", v.LongString())
				}
			}
		}
	}
}
