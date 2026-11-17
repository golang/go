// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"fmt"
	"math/bits"
	"testing"
)

func logosSCCPConfig(t *testing.T) *Conf {
	t.Helper()
	old := base.Debug.LogosOpt
	base.Debug.LogosOpt = 1
	t.Cleanup(func() { base.Debug.LogosOpt = old })
	return testConfig(t)
}

func TestLogosSCCPIntrinsics(t *testing.T) {
	c := logosSCCPConfig(t)
	for _, tt := range []struct {
		op       Op
		constOp  Op
		typ      *types.Type
		width    uint
		popcount bool
	}{
		{OpBitLen8, OpConst8, c.config.Types.UInt8, 8, false},
		{OpBitLen16, OpConst16, c.config.Types.UInt16, 16, false},
		{OpBitLen32, OpConst32, c.config.Types.UInt32, 32, false},
		{OpBitLen64, OpConst64, c.config.Types.UInt64, 64, false},
		{OpPopCount8, OpConst8, c.config.Types.UInt8, 8, true},
		{OpPopCount16, OpConst16, c.config.Types.UInt16, 16, true},
		{OpPopCount32, OpConst32, c.config.Types.UInt32, 32, true},
		{OpPopCount64, OpConst64, c.config.Types.UInt64, 64, true},
	} {
		for _, input := range []int64{0, 1, -1, -1 << (tt.width - 1)} {
			t.Run(fmt.Sprintf("%s/%d", tt.op, input), func(t *testing.T) {
				c := testConfig(t)
				masked := uint64(input) & (^uint64(0) >> (64 - tt.width))
				want := int64(bits.Len64(masked))
				if tt.popcount {
					want = int64(bits.OnesCount64(masked))
				}
				fun := c.Fun("entry",
					Bloc("entry",
						Valu("mem", OpInitMem, types.TypeMem, 0, nil),
						Valu("condition", OpConstBool, c.config.Types.Bool, 1, nil),
						Valu("unknown", OpArg, tt.typ, 0, nil),
						If("condition", "live", "dead")),
					Bloc("live",
						Valu("input", tt.constOp, tt.typ, input, nil),
						Goto("merge")),
					Bloc("dead", Goto("merge")),
					Bloc("merge",
						Valu("phi", OpPhi, tt.typ, 0, nil, "input", "unknown"),
						Valu("result", tt.op, c.config.Types.Int, 0, nil, "phi"),
						Valu("want", OpConst64, c.config.Types.Int, want, nil),
						Valu("equal", OpEq64, c.config.Types.Bool, 0, nil, "result", "want"),
						If("equal", "yes", "no")),
					Bloc("yes", Ret("mem")),
					Bloc("no", Ret("mem")))
				CheckFunc(fun.f)
				sccp(fun.f)
				CheckFunc(fun.f)
				result := fun.values["result"]
				if result.Op != OpConst64 || result.AuxInt != want || result.Type != c.config.Types.Int {
					t.Errorf("result = %s, want int constant %d", result.LongString(), want)
				}
				merge := fun.blocks["merge"]
				if merge.Kind != BlockPlain || len(merge.Succs) != 1 || merge.Succs[0].b != fun.blocks["yes"] {
					t.Errorf("constant intrinsic did not remove the false branch: %s", merge.LongString())
				}
			})
		}
	}
}

func TestLogosSCCPSelections(t *testing.T) {
	c := logosSCCPConfig(t)
	for _, tt := range []struct {
		name      string
		condition string
		x, y      string
		want      int64
		constant  bool
	}{
		{"true", "true", "a", "b", 7, true},
		{"false", "false", "a", "b", 11, true},
		{"true_ignores_unknown", "true", "a", "unknown", 7, true},
		{"false_ignores_unknown", "false", "unknown", "b", 11, true},
		{"unknown_equal_arms", "condition", "a", "same", 7, true},
		{"unknown_different_arms", "condition", "a", "b", 0, false},
		{"true_unknown_arm", "true", "unknown", "b", 0, false},
		{"false_unknown_arm", "false", "a", "unknown", 0, false},
		{"unknown_unknown_arm", "condition", "a", "unknown", 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := testConfig(t)
			fun := c.Fun("entry", Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("true", OpConstBool, c.config.Types.Bool, 1, nil),
				Valu("false", OpConstBool, c.config.Types.Bool, 0, nil),
				Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
				Valu("a", OpConst64, c.config.Types.Int64, 7, nil),
				Valu("same", OpConst64, c.config.Types.Int64, 7, nil),
				Valu("b", OpConst64, c.config.Types.Int64, 11, nil),
				Valu("unknown", OpArg, c.config.Types.Int64, 0, nil),
				Valu("result", OpCondSelect, c.config.Types.Int64, 0, nil, tt.x, tt.y, tt.condition),
				Ret("mem")))
			// Visit the select before its constant operands as well. SSA value
			// order is not a data dependency order.
			values := fun.f.Entry.Values
			values[0], values[len(values)-1] = values[len(values)-1], values[0]
			CheckFunc(fun.f)
			sccp(fun.f)
			CheckFunc(fun.f)
			result := fun.values["result"]
			if tt.constant {
				if result.Op != OpConst64 || result.AuxInt != tt.want {
					t.Errorf("result = %s, want %d", result.LongString(), tt.want)
				}
			} else if result.Op != OpCondSelect {
				t.Errorf("unknown result became %s", result.LongString())
			}
		})
	}
	for _, tt := range []struct {
		op  Op
		typ *types.Type
	}{
		{OpConst8, c.config.Types.Int8},
		{OpConst16, c.config.Types.Int16},
		{OpConst32, c.config.Types.Int32},
		{OpConst64, c.config.Types.Int64},
		{OpConstBool, c.config.Types.Bool},
	} {
		t.Run(tt.op.String(), func(t *testing.T) {
			c := testConfig(t)
			fun := c.Fun("entry", Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("condition", OpConstBool, c.config.Types.Bool, 1, nil),
				Valu("one", tt.op, tt.typ, 1, nil),
				Valu("zero", tt.op, tt.typ, 0, nil),
				Valu("result", OpCondSelect, tt.typ, 0, nil, "one", "zero", "condition"),
				Ret("mem")))
			sccp(fun.f)
			CheckFunc(fun.f)
			if result := fun.values["result"]; result.Op != tt.op || result.AuxInt != 1 || result.Type != tt.typ {
				t.Errorf("result = %s, want %s < %s > [1]", result.LongString(), tt.op, tt.typ)
			}
		})
	}
}

func TestLogosSCCPBooleans(t *testing.T) {
	logosSCCPConfig(t)
	for _, op := range []Op{OpAndB, OpOrB} {
		for x := -1; x <= 1; x++ {
			for y := -1; y <= 1; y++ {
				t.Run(fmt.Sprintf("%s/%d/%d", op, x, y), func(t *testing.T) {
					c := testConfig(t)
					xOp, yOp := OpConstBool, OpConstBool
					xAux, yAux := int64(x), int64(y)
					if x < 0 {
						xOp, xAux = OpArg, 0
					}
					if y < 0 {
						yOp, yAux = OpArg, 0
					}
					fun := c.Fun("entry", Bloc("entry",
						Valu("mem", OpInitMem, types.TypeMem, 0, nil),
						Valu("x", xOp, c.config.Types.Bool, xAux, nil),
						Valu("y", yOp, c.config.Types.Bool, yAux, nil),
						Valu("result", op, c.config.Types.Bool, 0, nil, "x", "y"),
						Ret("mem")))
					sccp(fun.f)
					CheckFunc(fun.f)
					constant := x >= 0 && y >= 0 || op == OpAndB && (x == 0 || y == 0) || op == OpOrB && (x == 1 || y == 1)
					result := fun.values["result"]
					if !constant {
						if result.Op != op {
							t.Errorf("unknown result became %s", result.LongString())
						}
						return
					}
					want := int64(0)
					if op == OpAndB && x == 1 && y == 1 || op == OpOrB && (x == 1 || y == 1) {
						want = 1
					}
					if result.Op != OpConstBool || result.AuxInt != want {
						t.Errorf("result = %s, want bool constant %d", result.LongString(), want)
					}
				})
			}
		}
	}
}

func TestLogosSCCPLoopLattice(t *testing.T) {
	logosSCCPConfig(t)
	for _, op := range []Op{OpCondSelect, OpAndB, OpOrB, OpBitLen64, OpPopCount64} {
		t.Run(op.String(), func(t *testing.T) {
			c := testConfig(t)
			phiType, phiOp := c.config.Types.Bool, OpConstBool
			resultType := c.config.Types.Bool
			args := []string{"phi", "true"}
			control := "result"
			if op == OpOrB {
				args[1] = "false"
			} else if op == OpCondSelect {
				args = []string{"true", "false", "phi"}
			} else if op == OpBitLen64 || op == OpPopCount64 {
				phiType, phiOp = c.config.Types.UInt64, OpConst64
				resultType = c.config.Types.Int
				args = []string{"phi"}
				control = "equal"
			}
			loop := Bloc("loop",
				Valu("phi", OpPhi, phiType, 0, nil, "initial", "next"),
				Valu("result", op, resultType, 0, nil, args...),
				If(control, "back", "exit"))
			if control == "equal" {
				loop.valus = append(loop.valus, Valu("equal", OpEq64, c.config.Types.Bool, 0, nil, "result", "one"))
			}
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("true", OpConstBool, c.config.Types.Bool, 1, nil),
					Valu("false", OpConstBool, c.config.Types.Bool, 0, nil),
					Valu("initial", phiOp, phiType, 1, nil),
					Valu("one", OpConst64, c.config.Types.Int, 1, nil),
					Goto("loop")),
				loop,
				Bloc("back",
					Valu("next", phiOp, phiType, 0, nil),
					Goto("loop")),
				Bloc("exit", Ret("mem")))
			CheckFunc(fun.f)
			sccp(fun.f)
			CheckFunc(fun.f)
			if fun.values["result"].Op != op || fun.blocks["loop"].Kind != BlockIf || len(fun.blocks["loop"].Succs) != 2 {
				t.Errorf("loop's later value was frozen as a constant: %s", fun.values["result"].LongString())
			}
		})
	}
}

func TestLogosSCCPEffectsAndUnsupportedValues(t *testing.T) {
	c := logosSCCPConfig(t)
	fun := c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("ptr", OpArg, c.config.Types.Bool.PtrTo(), 0, nil),
		Valu("false", OpConstBool, c.config.Types.Bool, 0, nil),
		Valu("call", OpStaticCall, types.TypeMem, 0, AuxCallLSym("effect"), "mem"),
		Valu("load", OpLoad, c.config.Types.Bool, 0, nil, "ptr", "call"),
		Valu("result", OpAndB, c.config.Types.Bool, 0, nil, "load", "false"),
		Valu("store", OpStore, types.TypeMem, 0, c.config.Types.Bool, "ptr", "result", "call"),
		Ret("store")))
	CheckFunc(fun.f)
	sccp(fun.f)
	CheckFunc(fun.f)
	if result := fun.values["result"]; result.Op != OpConstBool || result.AuxInt != 0 {
		t.Errorf("result = %s, want false", result.LongString())
	}
	for _, name := range []string{"call", "load", "store", "mem"} {
		if possibleConst(fun.values[name]) {
			t.Errorf("%s was admitted to the constant lattice", name)
		}
	}
	if fun.values["store"].Args[2] != fun.values["call"] || fun.values["call"].Args[0] != fun.values["mem"] {
		t.Fatal("constant folding changed the side-effect chain")
	}
	for _, op := range []Op{OpAdd64carry, OpMul64uhilo, OpMakeTuple} {
		if possibleConst(&Value{Op: op}) {
			t.Errorf("tuple operation %s was admitted to the constant lattice", op)
		}
	}
}

func TestLogosSCCPGate(t *testing.T) {
	logosSCCPConfig(t)
	for _, setting := range []int{-1, 0, 1} {
		t.Run(fmt.Sprint(setting), func(t *testing.T) {
			base.Debug.LogosOpt = setting
			c := testConfig(t)
			fun := c.Fun("entry", Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("input", OpConst64, c.config.Types.UInt64, 255, nil),
				Valu("result", OpPopCount64, c.config.Types.Int, 0, nil, "input"),
				Ret("mem")))
			sccp(fun.f)
			CheckFunc(fun.f)
			if got := isConst(fun.values["result"]); got != (setting > 0) {
				t.Errorf("setting %d: result = %s", setting, fun.values["result"].LongString())
			}
		})
	}
}
