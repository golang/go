// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/src"
	"fmt"
	"math"
	"math/big"
	"testing"
)

// startGuardEvaluator executes the generated integer guard on synthetic values.
// It never turns the synthetic addresses into Go pointers.
type startGuardEvaluator struct {
	t            *testing.T
	inputs       map[*Value]uint64
	paniconfault bool
}

func (e startGuardEvaluator) integer(v *Value) uint64 {
	if n, ok := e.inputs[v]; ok {
		return n
	}
	switch v.Op {
	case OpConst64:
		return uint64(v.AuxInt)
	case OpConvert:
		return e.integer(v.Args[0])
	case OpAnd64:
		return e.integer(v.Args[0]) & e.integer(v.Args[1])
	case OpAdd64:
		return e.integer(v.Args[0]) + e.integer(v.Args[1])
	case OpSub64:
		return e.integer(v.Args[0]) - e.integer(v.Args[1])
	case OpRsh64Ux64:
		return e.integer(v.Args[0]) >> e.integer(v.Args[1])
	case OpCondSelect:
		if e.boolean(v.Args[2]) {
			return e.integer(v.Args[0])
		}
		return e.integer(v.Args[1])
	default:
		e.t.Fatalf("unexpected integer operation: %s", v.LongString())
		return 0
	}
}

func (e startGuardEvaluator) boolean(v *Value) bool {
	switch v.Op {
	case OpNot:
		return !e.boolean(v.Args[0])
	case OpSelectN, OpLoad:
		return (autovecGuardEvaluator{t: e.t, inputs: e.inputs, paniconfault: e.paniconfault}).faultState(v)
	case OpLess64:
		return int64(e.integer(v.Args[0])) < int64(e.integer(v.Args[1]))
	case OpLess64U:
		return e.integer(v.Args[0]) < e.integer(v.Args[1])
	case OpLeq64U:
		return e.integer(v.Args[0]) <= e.integer(v.Args[1])
	case OpEqPtr:
		return e.integer(v.Args[0]) == e.integer(v.Args[1])
	case OpOrB:
		return e.boolean(v.Args[0]) || e.boolean(v.Args[1])
	case OpAndB:
		return e.boolean(v.Args[0]) && e.boolean(v.Args[1])
	default:
		e.t.Fatalf("unexpected Boolean operation: %s", v.LongString())
		return false
	}
}

func TestLogosAutovecConstantNonzeroStart(t *testing.T) {
	autovecTestSettings(t)
	fun := autovecTestLoop(t, "amd64")
	fun.values["zero"].AuxInt = 1
	CheckFunc(fun.f)
	autovec(fun.f)
	CheckFunc(fun.f)
	if got := autovecVectorStores(fun.f); got != 1 {
		t.Fatalf("vector stores = %d, want 1 for constant nonzero start", got)
	}
}

func TestLogosAutovecInvariantStartAndAliasGuard(t *testing.T) {
	autovecTestSettings(t)
	bounds := []struct {
		start, limit int64
	}{
		{1, 1}, {9, 3}, {1, 4}, {1, 5}, {1, 6}, {8, 13},
		{-9, -3}, {-5, 3}, {-1, 1}, {0, 0}, {0, 5},
		{math.MinInt64, math.MinInt64}, {math.MinInt64, math.MinInt64 + 4},
		{math.MinInt64, 0}, {math.MinInt64, math.MaxInt64},
		{math.MaxInt64 - 7, math.MaxInt64}, {math.MaxInt64, math.MinInt64},
		{math.MaxInt64, math.MaxInt64},
	}
	addresses := [][2]uint64{
		{0, 0}, {0, 12}, {0, 15}, {0, 16}, {0, 17}, {0, 20},
		{0, 32}, {0, ^uint64(0)}, {^uint64(0) - 16, ^uint64(0)},
	}
	for _, arch := range []string{"amd64", "arm64"} {
		for _, reversed := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/reversed=%t", arch, reversed), func(t *testing.T) {
				fun := autovecTestLoop(t, arch)
				f, h := fun.f, fun.blocks["header"]
				start := fun.blocks["entry"].NewValue0(src.NoXPos, OpArg, f.Config.Types.Int64)
				fun.values["index"].SetArg(0, start)
				entryIndex, backIndex := 0, 1
				if reversed {
					h.Preds[0], h.Preds[1] = h.Preds[1], h.Preds[0]
					for i, edge := range h.Preds {
						edge.b.Succs[edge.i].i = i
					}
					for _, name := range []string{"index", "loopmem"} {
						v := fun.values[name]
						v.Args[0], v.Args[1] = v.Args[1], v.Args[0]
					}
					entryIndex, backIndex = 1, 0
				}
				CheckFunc(f)
				autovec(f)
				CheckFunc(f)
				if got := autovecVectorStores(f); got != 1 {
					t.Fatalf("vector stores = %d, want 1 for invariant start", got)
				}
				vectorHeader := h.Preds[entryIndex].b
				vectorIndex := fun.values["index"].Args[entryIndex]
				if vectorIndex.Op != OpPhi || vectorIndex.Block != vectorHeader || vectorIndex.Args[0] != start {
					t.Fatal("vector index does not begin at the original start")
				}
				if fun.values["index"].Args[backIndex] != fun.values["next"] ||
					fun.values["loopmem"].Args[backIndex] != fun.values["store"] {
					t.Fatal("original scalar backedge changed")
				}
				bound := vectorHeader.Controls[0].Args[1]
				for _, b := range bounds {
					for _, pair := range addresses {
						for _, reverse := range []bool{false, true} {
							dst, source := pair[0], pair[1]
							if reverse {
								dst, source = source, dst
							}
							count := new(big.Int)
							if b.start < b.limit {
								count.Sub(big.NewInt(b.limit), big.NewInt(b.start))
								count.Div(count, big.NewInt(4))
								count.Mul(count, big.NewInt(4))
							}
							span := new(big.Int).Lsh(new(big.Int).Set(count), 2)
							distance := new(big.Int).Sub(new(big.Int).SetUint64(dst), new(big.Int).SetUint64(source))
							distance.Abs(distance)
							safe := dst == source || span.Cmp(distance) <= 0
							want := big.NewInt(b.start)
							if safe {
								want.Add(want, count)
							}
							eval := startGuardEvaluator{t: t, inputs: map[*Value]uint64{
								start:               uint64(b.start),
								fun.values["limit"]: uint64(b.limit),
								fun.values["dst"]:   dst,
								fun.values["src"]:   source,
							}}
							if got := int64(eval.integer(bound)); got != want.Int64() {
								t.Fatalf("start=%d limit=%d dst=%#x src=%#x: bound=%d, want %s",
									b.start, b.limit, dst, source, got, want)
							}
							eval.paniconfault = true
							if got := int64(eval.integer(bound)); got != b.start {
								t.Fatalf("fault recovery with start=%d limit=%d: bound=%d, want start", b.start, b.limit, got)
							}
						}
					}
				}
			})
		}
	}
}
