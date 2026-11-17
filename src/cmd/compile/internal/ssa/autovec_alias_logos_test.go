// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"fmt"
	"math/big"
	"testing"
)

// autovecGuardEvaluator evaluates only the integer and Boolean operations in
// the emitted alias guard. Pointer arguments are synthetic addresses; they are
// never converted to Go pointers or dereferenced.
type autovecGuardEvaluator struct {
	t            *testing.T
	inputs       map[*Value]uint64
	paniconfault bool
}

func (e autovecGuardEvaluator) integer(v *Value) uint64 {
	if input, ok := e.inputs[v]; ok {
		return input
	}
	switch v.Op {
	case OpConst64:
		return uint64(v.AuxInt)
	case OpConvert:
		return e.integer(v.Args[0])
	case OpAnd64:
		return e.integer(v.Args[0]) & e.integer(v.Args[1])
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
		e.t.Fatalf("unexpected integer operation in alias guard: %s", v.LongString())
		return 0
	}
}

func (e autovecGuardEvaluator) boolean(v *Value) bool {
	switch v.Op {
	case OpNot:
		return !e.boolean(v.Args[0])
	case OpSelectN, OpLoad:
		return e.faultState(v)
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
		e.t.Fatalf("unexpected Boolean operation in alias guard: %s", v.LongString())
		return false
	}
}

func (e autovecGuardEvaluator) faultState(v *Value) bool {
	var call *Value
	switch v.Op {
	case OpSelectN:
		call = v.Args[0]
		if v.AuxInt != 0 || !v.Type.IsBoolean() {
			e.t.Fatalf("invalid register fault-state result: %s", v.LongString())
		}
	case OpLoad:
		mem := v.Args[1]
		if mem.Op != OpSelectN || mem.AuxInt != 0 || !v.Type.IsBoolean() {
			e.t.Fatalf("invalid stack fault-state result: %s", v.LongString())
		}
		call = mem.Args[0]
	}
	if call.Op != OpStaticCall || call.Aux.(*AuxCall).Fn.Name != "logosAutovecPanicOnFault" {
		e.t.Fatalf("unexpected Boolean source in vector guard: %s", v.LongString())
	}
	aux := call.Aux.(*AuxCall)
	if v.Op == OpSelectN {
		if len(aux.RegsOfResult(0)) != 1 {
			e.t.Fatal("fault-state selector does not name a register result")
		}
	} else {
		addr := v.Args[0]
		if len(aux.RegsOfResult(0)) != 0 || addr.Op != OpOffPtr || addr.Args[0].Op != OpSP || addr.AuxInt != aux.OffsetOfResult(0) {
			e.t.Fatalf("fault-state load does not use its ABI result slot: %s", addr.LongString())
		}
	}
	return e.paniconfault
}

func TestLogosAutovecAliasGuardAddresses(t *testing.T) {
	autovecTestSettings(t)
	const maxAddress = ^uint64(0)
	const addressSignBit = uint64(1) << 63
	for _, arch := range []string{"amd64", "arm64"} {
		for laneIndex, size := range []int64{1, 2, 4, 8} {
			t.Run(fmt.Sprintf("%s/width%d", arch, size), func(t *testing.T) {
				fun := autovecTestLoop(t, arch)
				typ := &fun.f.Config.Types
				element := []*types.Type{typ.UInt8, typ.UInt16, typ.UInt32, typ.UInt64}[laneIndex]
				pointer := types.NewPtr(element)
				for _, name := range []string{"dst", "src", "dstaddr", "srcaddr"} {
					fun.values[name].Type = pointer
				}
				for _, name := range []string{"delta", "load", "sum"} {
					fun.values[name].Type = element
				}
				fun.values["sum"].Op = [...]Op{OpAdd8, OpAdd16, OpAdd32, OpAdd64}[laneIndex]
				fun.values["store"].Aux = element
				fun.values["shift"].AuxInt = int64(laneIndex)
				if size == 1 {
					for _, name := range []string{"dstaddr", "srcaddr"} {
						fun.values[name].SetArg(1, fun.values["index"])
					}
					fun.values["offset"].reset(OpConst64)
				}
				CheckFunc(fun.f)
				autovec(fun.f)
				CheckFunc(fun.f)
				if got := autovecVectorStores(fun.f); got != 1 {
					t.Fatalf("vector stores = %d, want 1", got)
				}
				vectorHeader := fun.blocks["header"].Preds[0].b
				bound := vectorHeader.Controls[0].Args[1]
				if bound.Op != OpCondSelect {
					t.Fatalf("vector bound has no overlap guard: %s", bound.LongString())
				}
				guard := bound.Args[2]
				lanes := 16 / size
				for _, count := range []int64{0, 1, lanes - 1, lanes, lanes + 1, 3*lanes + 1, 1<<63 - 1} {
					prefix := count - count%lanes
					// Exact interval endpoints provide an independent oracle, even
					// when the byte span or an endpoint exceeds the address width.
					span := new(big.Int).Mul(big.NewInt(prefix), big.NewInt(size))
					cases := []struct {
						name     string
						dst, src uint64
					}{
						{"equal-zero", 0, 0},
						{"equal-max", maxAddress, maxAddress},
						{"byte-separated", 0, 1},
						{"width-minus-one", 0, uint64(size - 1)},
						{"one-vector-minus-one", 0, 15},
						{"one-vector", 0, 16},
						{"one-vector-plus-one", 0, 17},
						{"three-vectors-minus-one", 0, 47},
						{"three-vectors", 0, 48},
						{"three-vectors-plus-one", 0, 49},
						{"max-minus-one", maxAddress - 1, maxAddress},
						{"max-minus-vector", maxAddress - 16, maxAddress},
						{"max-minus-unaligned-vector", maxAddress - 17, maxAddress},
						{"sign-bit-crossing", addressSignBit - 1, addressSignBit},
						{"sign-bit-vector", addressSignBit - 16, addressSignBit},
						{"sign-bit-unaligned-vector", addressSignBit - 17, addressSignBit},
						{"lower-half", 0, addressSignBit},
						{"upper-half", addressSignBit, maxAddress},
						{"whole-address-space", 0, maxAddress},
					}
					for _, delta := range []int64{-1, 0, 1} {
						distance := new(big.Int).Add(span, big.NewInt(delta))
						if distance.IsUint64() {
							cases = append(cases, struct {
								name     string
								dst, src uint64
							}{fmt.Sprintf("prefix-span%+d", delta), 0, distance.Uint64()})
						}
					}
					for _, test := range cases {
						for _, reverse := range []bool{false, true} {
							t.Run(fmt.Sprintf("count%d/%s/reverse%t", count, test.name, reverse), func(t *testing.T) {
								dst, src := test.dst, test.src
								if reverse {
									dst, src = src, dst
								}
								dstStart := new(big.Int).SetUint64(dst)
								srcStart := new(big.Int).SetUint64(src)
								dstEnd := new(big.Int).Add(dstStart, span)
								srcEnd := new(big.Int).Add(srcStart, span)
								wantSafe := dst == src || dstEnd.Cmp(srcStart) <= 0 || srcEnd.Cmp(dstStart) <= 0
								eval := autovecGuardEvaluator{t: t, inputs: map[*Value]uint64{
									fun.values["dst"]:   dst,
									fun.values["src"]:   src,
									fun.values["limit"]: uint64(count),
								}}
								if got := eval.boolean(guard); got != wantSafe {
									t.Fatalf("guard(%#x, %#x), prefix %d, byte span %s = %t, want %t", dst, src, prefix, span, got, wantSafe)
								}
								var wantBound uint64
								if wantSafe {
									wantBound = uint64(prefix)
								}
								if got := eval.integer(bound); got != wantBound {
									t.Fatalf("guarded vector bound = %d, want %d", got, wantBound)
								}
							})
						}
					}
				}
			})
		}
	}
}
