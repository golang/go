// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"fmt"
	"math"
	"testing"
)

func logosCorrelatedMode(t *testing.T, enabled int) {
	t.Helper()
	old := base.Debug.LogosOpt
	base.Debug.LogosOpt = enabled
	t.Cleanup(func() { base.Debug.LogosOpt = old })
}

func logosCorrelatedDiamond(t *testing.T, op, constOp Op, argType, resultType *types.Type, xs, ys [2]int64, constantRight bool) fun {
	t.Helper()
	c := testConfig(t)
	right := "y"
	if constantRight {
		right = "y0"
	}
	return c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
			Valu("address", OpArg, resultType.PtrTo(), 0, nil),
			Valu("x0", constOp, argType, xs[0], nil),
			Valu("x1", constOp, argType, xs[1], nil),
			Valu("y0", constOp, argType, ys[0], nil),
			Valu("y1", constOp, argType, ys[1], nil),
			If("condition", "left", "right")),
		Bloc("left", Goto("join")),
		Bloc("right", Goto("join")),
		Bloc("join",
			Valu("x", OpPhi, argType, 0, nil, "x0", "x1"),
			Valu("y", OpPhi, argType, 0, nil, "y0", "y1"),
			Valu("result", op, resultType, 0, nil, "x", right),
			Valu("store", OpStore, types.TypeMem, 0, resultType, "address", "result", "mem"),
			Exit("store")))
}

func TestLogosCorrelatedConstants(t *testing.T) {
	logosCorrelatedMode(t, 1)
	c := testConfig(t)
	tests := []struct {
		name                string
		op, constOp         Op
		argType, resultType *types.Type
		xs, ys              [2]int64
		constantRight       bool
		want                int64
	}{
		{"add64", OpAdd64, OpConst64, c.config.Types.Int64, c.config.Types.Int64, [2]int64{1, 2}, [2]int64{3, 2}, false, 4},
		{"add8_wrap", OpAdd8, OpConst8, c.config.Types.Int8, c.config.Types.Int8, [2]int64{127, 126}, [2]int64{1, 2}, false, -128},
		{"add16_wrap", OpAdd16, OpConst16, c.config.Types.Int16, c.config.Types.Int16, [2]int64{32767, 32766}, [2]int64{1, 2}, false, -32768},
		{"add32_wrap", OpAdd32, OpConst32, c.config.Types.Int32, c.config.Types.Int32, [2]int64{1<<31 - 1, 1<<31 - 2}, [2]int64{1, 2}, false, -1 << 31},
		{"add64_wrap", OpAdd64, OpConst64, c.config.Types.Int64, c.config.Types.Int64, [2]int64{1<<63 - 1, 1<<63 - 2}, [2]int64{1, 2}, false, -1 << 63},
		{"sub64", OpSub64, OpConst64, c.config.Types.Int64, c.config.Types.Int64, [2]int64{1, 2}, [2]int64{4, 5}, false, -3},
		{"sub8_wrap", OpSub8, OpConst8, c.config.Types.Int8, c.config.Types.Int8, [2]int64{-128, -127}, [2]int64{1, 2}, false, 127},
		{"mul64", OpMul64, OpConst64, c.config.Types.Int64, c.config.Types.Int64, [2]int64{2, 3}, [2]int64{6, 4}, false, 12},
		{"mul8_wrap", OpMul8, OpConst8, c.config.Types.Int8, c.config.Types.Int8, [2]int64{64, -128}, [2]int64{4, 2}, false, 0},
		{"and_constant", OpAnd64, OpConst64, c.config.Types.Int64, c.config.Types.Int64, [2]int64{0x17, 0x27}, [2]int64{15, 15}, true, 7},
		{"or", OpOr64, OpConst64, c.config.Types.Int64, c.config.Types.Int64, [2]int64{0x10, 0x11}, [2]int64{3, 2}, false, 0x13},
		{"xor", OpXor64, OpConst64, c.config.Types.Int64, c.config.Types.Int64, [2]int64{1, 6}, [2]int64{3, 4}, false, 2},
		{"eq", OpEq64, OpConst64, c.config.Types.Int64, c.config.Types.Bool, [2]int64{1, 2}, [2]int64{2, 1}, false, 0},
		{"neq", OpNeq64, OpConst64, c.config.Types.Int64, c.config.Types.Bool, [2]int64{1, 2}, [2]int64{1, 2}, false, 0},
		{"less_signed", OpLess64, OpConst64, c.config.Types.Int64, c.config.Types.Bool, [2]int64{-1, 0}, [2]int64{0, 1}, false, 1},
		{"less_unsigned", OpLess64U, OpConst64, c.config.Types.UInt64, c.config.Types.Bool, [2]int64{-1, -2}, [2]int64{0, 1}, false, 0},
		{"leq_signed", OpLeq64, OpConst64, c.config.Types.Int64, c.config.Types.Bool, [2]int64{1, 2}, [2]int64{1, 3}, false, 1},
		{"leq_unsigned", OpLeq64U, OpConst64, c.config.Types.UInt64, c.config.Types.Bool, [2]int64{0, 1}, [2]int64{-1, -2}, false, 1},
		{"eq_bool", OpEqB, OpConstBool, c.config.Types.Bool, c.config.Types.Bool, [2]int64{1, 0}, [2]int64{0, 1}, false, 0},
		{"neq_bool", OpNeqB, OpConstBool, c.config.Types.Bool, c.config.Types.Bool, [2]int64{1, 0}, [2]int64{0, 1}, false, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fn := logosCorrelatedDiamond(t, tc.op, tc.constOp, tc.argType, tc.resultType, tc.xs, tc.ys, tc.constantRight)
			CheckFunc(fn.f)
			logosCorrelatedConstants(fn.f)
			CheckFunc(fn.f)
			v := fn.values["result"]
			if !isConst(v) || v.AuxInt != tc.want {
				t.Fatalf("correlated result = %s; want constant %d", v.LongString(), tc.want)
			}
			if fn.values["store"].Op != OpStore || fn.values["store"].Args[2] != fn.values["mem"] {
				t.Fatal("correlated folding changed the store or its memory order")
			}
		})
	}
}

func TestLogosCorrelatedConstantsConservative(t *testing.T) {
	c := testConfig(t)
	tests := []struct {
		name        string
		op, constOp Op
		typ         *types.Type
		xs, ys      [2]int64
		enabled     int
	}{
		{"different_results", OpAdd64, OpConst64, c.config.Types.Int64, [2]int64{1, 2}, [2]int64{3, 5}, 1},
		{"predecessor_alignment", OpSub64, OpConst64, c.config.Types.Int64, [2]int64{1, 2}, [2]int64{2, 1}, 1},
		{"zero_divisor", OpDiv64, OpConst64, c.config.Types.Int64, [2]int64{1, 2}, [2]int64{0, 0}, 1},
		{"division_excluded", OpDiv64, OpConst64, c.config.Types.Int64, [2]int64{2, 4}, [2]int64{2, 4}, 1},
		{"floating_point_excluded", OpAdd64F, OpConst64F, c.config.Types.Float64,
			[2]int64{int64(math.Float64bits(1)), int64(math.Float64bits(2))},
			[2]int64{int64(math.Float64bits(3)), int64(math.Float64bits(2))}, 1},
		{"disabled", OpAdd64, OpConst64, c.config.Types.Int64, [2]int64{1, 2}, [2]int64{3, 2}, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logosCorrelatedMode(t, tc.enabled)
			fn := logosCorrelatedDiamond(t, tc.op, tc.constOp, tc.typ, tc.typ, tc.xs, tc.ys, false)
			CheckFunc(fn.f)
			logosCorrelatedConstants(fn.f)
			CheckFunc(fn.f)
			if v := fn.values["result"]; v.Op != tc.op {
				t.Fatalf("unproved result changed: %s", v.LongString())
			}
		})
	}
}

func TestLogosCorrelatedConstantsNonlocal(t *testing.T) {
	logosCorrelatedMode(t, 1)
	c := testConfig(t)
	i64 := c.config.Types.Int64
	fn := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
			Valu("other", OpArg, c.config.Types.Bool, 0, nil),
			Valu("address", OpArg, i64.PtrTo(), 0, nil),
			Valu("one", OpConst64, i64, 1, nil),
			Valu("two", OpConst64, i64, 2, nil),
			Valu("three", OpConst64, i64, 3, nil),
			If("condition", "left", "right")),
		Bloc("left", Goto("join")),
		Bloc("right", Goto("join")),
		Bloc("join",
			Valu("x", OpPhi, i64, 0, nil, "one", "two"),
			If("other", "secondleft", "secondright")),
		Bloc("secondleft", Goto("after")),
		Bloc("secondright", Goto("after")),
		Bloc("after",
			Valu("y", OpPhi, i64, 0, nil, "three", "two"),
			Valu("result", OpAdd64, i64, 0, nil, "x", "y"),
			Valu("store", OpStore, types.TypeMem, 0, i64, "address", "result", "mem"),
			Exit("store")))
	CheckFunc(fn.f)
	logosCorrelatedConstants(fn.f)
	CheckFunc(fn.f)
	if v := fn.values["result"]; v.Op != OpAdd64 {
		t.Fatalf("nonlocal phis must stay outside this pass: %s", v.LongString())
	}
}

func TestLogosCorrelatedConstantsLoop(t *testing.T) {
	logosCorrelatedMode(t, 1)
	c := testConfig(t)
	i64 := c.config.Types.Int64
	fn := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
			Valu("address", OpArg, i64.PtrTo(), 0, nil),
			Valu("one", OpConst64, i64, 1, nil),
			Valu("three", OpConst64, i64, 3, nil),
			Goto("loop")),
		Bloc("loop",
			Valu("x", OpPhi, i64, 0, nil, "one", "nextx"),
			Valu("y", OpPhi, i64, 0, nil, "three", "nexty"),
			Valu("result", OpAdd64, i64, 0, nil, "x", "y"),
			If("condition", "back", "exit")),
		Bloc("back",
			Valu("nextx", OpAdd64, i64, 0, nil, "x", "one"),
			Valu("nexty", OpSub64, i64, 0, nil, "y", "one"),
			Goto("loop")),
		Bloc("exit",
			Valu("store", OpStore, types.TypeMem, 0, i64, "address", "result", "mem"),
			Exit("store")))
	CheckFunc(fn.f)
	logosCorrelatedConstants(fn.f)
	CheckFunc(fn.f)
	if v := fn.values["result"]; v.Op != OpAdd64 {
		t.Fatalf("loop-carried values must not be recursively unfolded: %s", v.LongString())
	}
}

func TestLogosCorrelatedConstantsPredecessorLimit(t *testing.T) {
	logosCorrelatedMode(t, 1)
	for _, n := range []int{7, 8, 9} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			c := testConfig(t)
			i64 := c.config.Types.Int64
			var blocks []bloc
			var names, xs, ys []string
			for i := 0; i < n; i++ {
				name := fmt.Sprintf("pred%d", i)
				x := fmt.Sprintf("x%d", i)
				y := fmt.Sprintf("y%d", i)
				names = append(names, name)
				xs = append(xs, x)
				ys = append(ys, y)
				blocks = append(blocks, Bloc(name,
					Valu(x, OpConst64, i64, int64(i), nil),
					Valu(y, OpConst64, i64, int64(100-i), nil),
					Goto("join")))
			}
			blocks = append([]bloc{Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("index", OpArg, i64, 0, nil),
				Valu("address", OpArg, i64.PtrTo(), 0, nil),
				ctrl{BlockJumpTable, "index", names})}, blocks...)
			blocks = append(blocks, Bloc("join",
				Valu("x", OpPhi, i64, 0, nil, xs...),
				Valu("y", OpPhi, i64, 0, nil, ys...),
				Valu("result", OpAdd64, i64, 0, nil, "x", "y"),
				Valu("store", OpStore, types.TypeMem, 0, i64, "address", "result", "mem"),
				Exit("store")))
			fn := c.Fun("entry", blocks...)
			CheckFunc(fn.f)
			logosCorrelatedConstants(fn.f)
			CheckFunc(fn.f)
			v := fn.values["result"]
			if n <= 8 {
				if v.Op != OpConst64 || v.AuxInt != 100 {
					t.Fatalf("%d predecessors: result = %s; want constant 100", n, v.LongString())
				}
			} else if v.Op != OpAdd64 {
				t.Fatalf("%d predecessors exceed the work bound: %s", n, v.LongString())
			}
		})
	}
}
