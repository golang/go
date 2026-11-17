// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"testing"
)

func autovecExprTestLoop(t *testing.T, arch string, size int64, op Op, count int64) fun {
	t.Helper()
	fun := autovecTestLoop(t, arch)
	typ := &fun.f.Config.Types
	var element *types.Type
	var shift int64
	switch size {
	case 1:
		element, shift = typ.UInt8, 0
	case 2:
		element, shift = typ.UInt16, 1
	case 4:
		element, shift = typ.UInt32, 2
	case 8:
		element, shift = typ.UInt64, 3
	default:
		t.Fatalf("invalid expression element width %d", size)
	}
	switch op {
	case OpRsh16x64:
		element = typ.Int16
	case OpRsh32x64:
		element = typ.Int32
	case OpRsh64x64:
		element = typ.Int64
	}
	pointer := types.NewPtr(element)
	for _, name := range []string{"dst", "src", "dstaddr", "srcaddr"} {
		fun.values[name].Type = pointer
	}
	for _, name := range []string{"delta", "load", "sum"} {
		fun.values[name].Type = element
	}
	fun.values["store"].Aux = element
	fun.values["shift"].AuxInt = shift
	if size == 1 {
		for _, name := range []string{"dstaddr", "srcaddr"} {
			fun.values[name].SetArg(1, fun.values["index"])
		}
		fun.values["offset"].reset(OpConst64)
	}
	expression := fun.values["sum"]
	expression.reset(op)
	expression.AddArg(fun.values["load"])
	switch op {
	case OpNeg8, OpNeg16, OpNeg32, OpNeg64, OpCom8, OpCom16, OpCom32, OpCom64:
	default:
		// A body-local constant requires the rewrite to place a scalar count
		// before the vector loop, rather than broadcast or illegally reuse it.
		expression.AddArg(fun.blocks["body"].NewValue0I(src.NoXPos, OpConst64, typ.UInt64, count))
	}
	return fun
}

func TestLogosAutovecExpressions(t *testing.T) {
	autovecTestSettings(t)
	tests := []struct {
		name           string
		size           int64
		scalar, vector Op
	}{
		{"neg8", 1, OpNeg8, OpSubInt8x16},
		{"neg16", 2, OpNeg16, OpSubInt16x8},
		{"neg32", 4, OpNeg32, OpSubInt32x4},
		{"neg64", 8, OpNeg64, OpSubInt64x2},
		{"com8", 1, OpCom8, OpXorInt8x16},
		{"com16", 2, OpCom16, OpXorInt16x8},
		{"com32", 4, OpCom32, OpXorInt32x4},
		{"com64", 8, OpCom64, OpXorInt64x2},
		{"left8", 1, OpLsh8x64, OpAddInt8x16},
		{"left16", 2, OpLsh16x64, OpShiftAllLeftInt16x8},
		{"left32", 4, OpLsh32x64, OpShiftAllLeftInt32x4},
		{"left64", 8, OpLsh64x64, OpShiftAllLeftInt64x2},
		{"unsignedRight16", 2, OpRsh16Ux64, OpShiftAllRightUint16x8},
		{"unsignedRight32", 4, OpRsh32Ux64, OpShiftAllRightUint32x4},
		{"unsignedRight64", 8, OpRsh64Ux64, OpShiftAllRightUint64x2},
		{"signedRight16", 2, OpRsh16x64, OpShiftAllRightInt16x8},
		{"signedRight32", 4, OpRsh32x64, OpShiftAllRightInt32x4},
	}
	for _, arch := range []string{"amd64", "arm64"} {
		for _, test := range tests {
			counts := []int64{1}
			if test.scalar != OpLsh8x64 {
				counts = append(counts, test.size*8-1)
			}
			for _, count := range counts {
				t.Run(fmt.Sprintf("%s/%s/count%d", arch, test.name, count), func(t *testing.T) {
					fun := autovecExprTestLoop(t, arch, test.size, test.scalar, count)
					CheckFunc(fun.f)
					autovec(fun.f)
					CheckFunc(fun.f)
					if got := autovecVectorStores(fun.f); got != 1 {
						t.Fatalf("vector stores = %d, want 1", got)
					}
					var vector *Value
					for _, block := range fun.f.Blocks {
						for _, value := range block.Values {
							if value.Op == test.vector && value.Type.IsSIMD() {
								vector = value
							}
						}
					}
					if vector == nil {
						t.Fatalf("missing vector operation %s", test.vector)
					}
					if test.scalar == OpLsh8x64 {
						if vector.Args[0] != vector.Args[1] {
							t.Fatal("byte shift by one does not add the operand to itself")
						}
					} else if len(fun.values["sum"].Args) == 2 {
						scalarCount := vector.Args[1]
						if scalarCount.Op != OpConst64 || scalarCount.AuxInt != count ||
							scalarCount.Type != fun.f.Config.Types.UInt64 || scalarCount.Block != fun.blocks["header"].Preds[0].b.Preds[0].b {
							t.Fatalf("vector shift count was not preserved as a predecessor scalar: %s", scalarCount.LongString())
						}
					}
				})
			}
		}
	}
}

func TestLogosAutovecRejectsUnsupportedExpressions(t *testing.T) {
	autovecTestSettings(t)
	tests := []struct {
		name     string
		size     int64
		op       Op
		count    int64
		variable bool
	}{
		{"byteLeft2", 1, OpLsh8x64, 2, false},
		{"byteLeft3", 1, OpLsh8x64, 3, false},
		{"byteRight", 1, OpRsh8Ux64, 1, false},
		{"signedRight64", 8, OpRsh64x64, 1, false},
		{"leftAtWidth", 4, OpLsh32x64, 32, false},
		{"leftPastWidth", 4, OpLsh32x64, 33, false},
		{"rightAtWidth", 2, OpRsh16x64, 16, false},
		{"rightPastWidth", 2, OpRsh16x64, 17, false},
		{"hugeUnsignedCount", 8, OpLsh64x64, -1, false},
		{"variableCount", 4, OpLsh32x64, 1, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fun := autovecExprTestLoop(t, "amd64", test.size, test.op, test.count)
			if test.variable {
				count := fun.blocks["entry"].NewValue0(src.NoXPos, OpArg, fun.f.Config.Types.UInt64)
				fun.values["sum"].SetArg(1, count)
			}
			CheckFunc(fun.f)
			before := fun.f.String()
			autovec(fun.f)
			CheckFunc(fun.f)
			if fun.f.String() != before {
				t.Fatal("unsupported expression changed the scalar loop")
			}
		})
	}
}
