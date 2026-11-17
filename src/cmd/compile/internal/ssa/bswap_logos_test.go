// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"fmt"
	"testing"
)

var logosBswapCases = [...]struct {
	name                        string
	op, inputConst, outputConst Op
	add, mul, equal             Op
	inputKind, outputKind       types.Kind
	width, outputWidth          uint
}{
	{"16", OpBswap16, OpConst16, OpConst16, OpAdd16, OpMul16, OpEq16, types.TUINT16, types.TUINT16, 16, 16},
	{"32", OpBswap32, OpConst32, OpConst32, OpAdd32, OpMul32, OpEq32, types.TUINT32, types.TUINT32, 32, 32},
	{"64", OpBswap64, OpConst64, OpConst64, OpAdd64, OpMul64, OpEq64, types.TUINT64, types.TUINT64, 64, 64},
	// Power10's ReverseBytes16 intrinsic uses uint as the result type.
	{"16_uint", OpBswap16, OpConst16, OpConst64, OpAdd16, OpMul16, OpEq64, types.TUINT16, types.TUINT, 16, 64},
}

// Use a byte-at-a-time oracle independent of the compiler's bit intrinsic.
func logosBswapReference(x uint64, width uint) uint64 {
	var result uint64
	for shift := uint(0); shift < width; shift += 8 {
		result = result<<8 | (x>>shift)&255
	}
	return result
}

func TestLogosBswapKnownBits(t *testing.T) {
	enableLogosKnownBits(t)
	for _, tc := range logosBswapCases {
		t.Run(tc.name, func(t *testing.T) {
			c := testConfig(t)
			fun := c.Fun("entry", Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("input", OpArg, types.Types[tc.inputKind], 0, nil),
				Valu("result", tc.op, types.Types[tc.outputKind], 0, nil, "input"),
				Ret("mem")))
			CheckFunc(fun.f)
			kb := knownBitsState{
				entries:         make([]knownBitsEntry, fun.f.NumValues()),
				seenValues:      newBitset(fun.f.NumValues()),
				reachableBlocks: newBitset(fun.f.NumBlocks()),
			}
			mask := ^uint64(0) >> (64 - tc.width)
			check := func(value, known uint64) {
				t.Helper()
				clear(kb.entries)
				clear(kb.seenValues)
				id := fun.values["input"].ID
				kb.entries[id] = logosKnownBitsFact(value, known, tc.width)
				kb.seenValues.Set(uint32(id))
				v, k := kb.fold(fun.values["result"])
				got := knownBitsEntry{value: v, known: k}
				want := logosKnownBitsFact(logosBswapReference(value, tc.width), logosBswapReference(known, tc.width)|^mask, tc.outputWidth)
				if got != want {
					t.Fatalf("%s value=%#x known=%#x: got %v, want %v", tc.op, value, known, got, want)
				}
			}
			// Every input bit independently carries a known zero or one;
			// every other bit remains unknown, including the sign bit.
			check(0, 0)
			for bit := uint(0); bit < tc.width; bit++ {
				check(0, uint64(1)<<bit)
				check(uint64(1)<<bit, uint64(1)<<bit)
			}
			for _, x := range []uint64{0, 1, mask, mask >> 1, uint64(1) << (tc.width - 1), 0x0123456789abcdef & mask} {
				check(x, mask)
			}
		})
	}
}

func TestLogosBswapSCCPLateConstant(t *testing.T) {
	logosSCCPConfig(t)
	for _, tc := range logosBswapCases {
		mask := ^uint64(0) >> (64 - tc.width)
		for _, input := range []uint64{0, 1, mask, mask >> 1, uint64(1) << (tc.width - 1), 0x0123456789abcdef & mask} {
			t.Run(fmt.Sprintf("%s/%x", tc.name, input), func(t *testing.T) {
				c := testConfig(t)
				inputType, outputType := types.Types[tc.inputKind], types.Types[tc.outputKind]
				inputAux := logosKnownBitsFact(input, mask, tc.width).value
				want := logosKnownBitsFact(logosBswapReference(input, tc.width), ^uint64(0), tc.outputWidth).value
				fun := c.Fun("entry",
					Bloc("entry",
						Valu("mem", OpInitMem, types.TypeMem, 0, nil),
						Valu("condition", OpConstBool, c.config.Types.Bool, 1, nil),
						Valu("unknown", OpArg, inputType, 0, nil),
						If("condition", "live", "dead")),
					Bloc("live", Valu("input", tc.inputConst, inputType, inputAux, nil), Goto("merge")),
					Bloc("dead", Goto("merge")),
					Bloc("merge",
						Valu("phi", OpPhi, inputType, 0, nil, "input", "unknown"),
						Valu("result", tc.op, outputType, 0, nil, "phi"),
						Valu("want", tc.outputConst, outputType, want, nil),
						Valu("equal", tc.equal, c.config.Types.Bool, 0, nil, "result", "want"),
						If("equal", "yes", "no")),
					Bloc("yes", Ret("mem")),
					Bloc("no", Ret("mem")))
				values := fun.blocks["merge"].Values
				values[0], values[1] = values[1], values[0]
				CheckFunc(fun.f)
				sccp(fun.f)
				CheckFunc(fun.f)
				result := fun.values["result"]
				if result.Op != tc.outputConst || result.AuxInt != want || result.Type != outputType {
					t.Fatalf("result = %s, want %s <%v> [%d]", result.LongString(), tc.outputConst, outputType, want)
				}
				merge := fun.blocks["merge"]
				if merge.Kind != BlockPlain || len(merge.Succs) != 1 || merge.Succs[0].b != fun.blocks["yes"] {
					t.Fatal("late byte-swap constant did not expose the live branch")
				}
			})
		}
	}
}

func TestLogosBswapSCCPLoops(t *testing.T) {
	logosSCCPConfig(t)
	for _, tc := range logosBswapCases {
		for _, changing := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/changing=%t", tc.name, changing), func(t *testing.T) {
				c := testConfig(t)
				inputType, outputType := types.Types[tc.inputKind], types.Types[tc.outputKind]
				want := int64(logosBswapReference(1, tc.width))
				next := Valu("next", tc.mul, inputType, 0, nil, "phi", "phi")
				if changing {
					next = Valu("next", tc.inputConst, inputType, 0, nil)
				}
				fun := c.Fun("entry",
					Bloc("entry",
						Valu("mem", OpInitMem, types.TypeMem, 0, nil),
						Valu("initial", tc.inputConst, inputType, 1, nil),
						Valu("want", tc.outputConst, outputType, want, nil),
						Goto("loop")),
					Bloc("loop",
						Valu("phi", OpPhi, inputType, 0, nil, "initial", "next"),
						Valu("result", tc.op, outputType, 0, nil, "phi"),
						Valu("equal", tc.equal, c.config.Types.Bool, 0, nil, "result", "want"),
						If("equal", "back", "exit")),
					Bloc("back", next, Goto("loop")),
					Bloc("exit", Ret("mem")))
				// Visit the swap while its phi is still Top, then revisit it
				// after the first constant and any conflicting backedge.
				values := fun.blocks["loop"].Values
				values[0], values[1] = values[1], values[0]
				CheckFunc(fun.f)
				sccp(fun.f)
				CheckFunc(fun.f)
				result, loop := fun.values["result"], fun.blocks["loop"]
				if changing {
					if result.Op != tc.op || loop.Kind != BlockIf || len(loop.Succs) != 2 {
						t.Fatalf("new backedge froze a changing value: %s", result.LongString())
					}
				} else if result.Op != tc.outputConst || result.AuxInt != want || result.Type != outputType || loop.Kind != BlockPlain {
					t.Fatalf("multiplicative fixed point did not fold: %s", result.LongString())
				}
			})
		}
	}
}

func TestLogosBswapKnownBitsLoop(t *testing.T) {
	enableLogosKnownBits(t)
	for _, tc := range logosBswapCases {
		t.Run(tc.name, func(t *testing.T) {
			c := testConfig(t)
			inputType, outputType := types.Types[tc.inputKind], types.Types[tc.outputKind]
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("zero", tc.inputConst, inputType, 0, nil),
					Valu("one", tc.inputConst, inputType, 1, nil),
					Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
					Goto("loop")),
				Bloc("loop",
					Valu("phi", OpPhi, inputType, 0, nil, "zero", "next"),
					Valu("next", tc.add, inputType, 0, nil, "phi", "one"),
					Valu("result", tc.op, outputType, 0, nil, "phi"),
					If("condition", "loop", "exit")),
				Bloc("exit", Ret("mem")))
			CheckFunc(fun.f)
			kb := knownBitsState{
				entries:         make([]knownBitsEntry, fun.f.NumValues()),
				seenValues:      newBitset(fun.f.NumValues()),
				reachableBlocks: newBitset(fun.f.NumBlocks()),
			}
			for _, b := range fun.f.Blocks {
				kb.reachableBlocks.Set(uint32(b.ID))
			}
			value, known := kb.fold(fun.values["result"])
			mask := ^uint64(0) >> (64 - tc.width)
			want := logosKnownBitsFact(0, ^mask, tc.outputWidth)
			if got := (knownBitsEntry{value: value, known: known}); got != want {
				t.Fatalf("wrapping induction invented byte facts: %v, want %v", got, want)
			}
		})
	}
}

func TestLogosBswapGate(t *testing.T) {
	logosSCCPConfig(t)
	for _, setting := range []int{-1, 0, 1} {
		for _, tc := range logosBswapCases {
			t.Run(fmt.Sprintf("%d/%s", setting, tc.name), func(t *testing.T) {
				base.Debug.LogosOpt = setting
				c := testConfig(t)
				fun := c.Fun("entry", Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("input", tc.inputConst, types.Types[tc.inputKind], 1, nil),
					Valu("result", tc.op, types.Types[tc.outputKind], 0, nil, "input"),
					Ret("mem")))
				kb := knownBitsState{
					entries:         make([]knownBitsEntry, fun.f.NumValues()),
					seenValues:      newBitset(fun.f.NumValues()),
					reachableBlocks: newBitset(fun.f.NumBlocks()),
				}
				v, k := kb.fold(fun.values["result"])
				if setting <= 0 {
					if v != 0 || k != 0 {
						t.Fatalf("disabled byte swap invented facts: value=%x known=%x", v, k)
					}
				} else if k != -1 || v != int64(logosBswapReference(1, tc.width)) {
					t.Fatalf("enabled byte swap lost constant facts: value=%x known=%x", v, k)
				}
				sccp(fun.f)
				CheckFunc(fun.f)
				if got := isConst(fun.values["result"]); got != (setting > 0) {
					t.Fatalf("setting %d: result = %s", setting, fun.values["result"].LongString())
				}
			})
		}
	}
}

func TestLogosBswapEffects(t *testing.T) {
	logosSCCPConfig(t)
	c := testConfig(t)
	fun := c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("ptr", OpArg, c.config.Types.UInt64.PtrTo(), 0, nil),
		Valu("call", OpStaticCall, types.TypeMem, 0, AuxCallLSym("effect"), "mem"),
		Valu("load", OpLoad, c.config.Types.UInt64, 0, nil, "ptr", "call"),
		Valu("result", OpBswap64, c.config.Types.UInt64, 0, nil, "load"),
		Valu("store", OpStore, types.TypeMem, 0, c.config.Types.UInt64, "ptr", "result", "call"),
		Ret("store")))
	CheckFunc(fun.f)
	knownBits(fun.f)
	sccp(fun.f)
	CheckFunc(fun.f)
	if fun.values["result"].Op != OpBswap64 || fun.values["result"].Args[0] != fun.values["load"] {
		t.Fatal("runtime byte swap was replaced by a constant")
	}
	if fun.values["store"].Args[2] != fun.values["call"] || fun.values["call"].Args[0] != fun.values["mem"] {
		t.Fatal("byte-swap folding changed the effect chain")
	}
	for _, name := range []string{"mem", "call", "load", "store"} {
		if possibleConst(fun.values[name]) {
			t.Fatalf("effectful value %s entered SCCP", name)
		}
	}
}
