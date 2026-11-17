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

func TestLogosAutovecFaultGate(t *testing.T) {
	autovecTestSettings(t)
	for _, arch := range []string{"amd64", "arm64"} {
		for _, stackABI := range []bool{false, true} {
			for _, branch := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stackABI%t/branch%t", arch, stackABI, branch), func(t *testing.T) {
					fun := autovecTestLoop(t, arch)
					f := fun.f
					if stackABI {
						f.ABIDefault = f.Config.ABI0
					}
					entry := fun.blocks["entry"]
					var skip *Block
					if branch {
						skip = f.NewBlock(BlockExit)
						skip.SetControl(fun.values["mem"])
						entry.Kind = BlockIf
						entry.SetControl(entry.NewValue0(src.NoXPos, OpArg, f.Config.Types.Bool))
						entry.AddEdgeTo(skip)
					}
					CheckFunc(f)
					autovec(f)
					CheckFunc(f)
					if autovecVectorStores(f) != 1 {
						t.Fatal("fault guard removed the vector path")
					}
					var call, callMem *Value
					for _, block := range f.Blocks {
						for _, v := range block.Values {
							if v.Op == OpStaticLECall {
								t.Fatal("late vectorization emitted an unexpanded call")
							}
							if v.Op == OpStaticCall {
								if call != nil {
									t.Fatal("multiple runtime calls on one loop entry")
								}
								call = v
							}
						}
					}
					if call == nil || call.Aux.(*AuxCall).Fn.Name != "logosAutovecPanicOnFault" {
						t.Fatal("vector entry does not query the current goroutine's fault policy")
					}
					if call.Block == entry || len(call.Block.Preds) != 1 || call.Block.Preds[0].b != entry || entry.Succs[0].b != call.Block {
						t.Fatal("runtime call is not isolated on the original loop-entry edge")
					}
					if branch && (entry.Succs[1].b != skip || skip.Controls[0] != fun.values["mem"]) {
						t.Fatal("fault-policy call changed the unrelated predecessor edge")
					}
					if len(call.Args) != 1 || call.Args[0] != fun.values["mem"] {
						t.Fatal("fault-policy call does not follow the scalar loop's incoming memory")
					}
					aux := call.Aux.(*AuxCall)
					if aux.ABI() != f.ABIDefault || aux.NArgs() != 0 || aux.NResults() != 1 || aux.TypeOfResult(0) != f.Config.Types.Bool || call.AuxInt != aux.ArgWidth() {
						t.Fatal("fault-policy call has the wrong ABI or frame size")
					}
					registers := aux.ABIInfo().OutRegistersUsed()
					if call.Type.NumFields() != registers+1 || call.Type.FieldType(registers) != types.TypeMem {
						t.Fatal("fault-policy call has an unexpanded result tuple")
					}
					for _, block := range f.Blocks {
						for _, v := range block.Values {
							if v.Op == OpSelectN && v.Args[0] == call && v.Type.IsMemory() {
								if callMem != nil || v.AuxInt != int64(registers) {
									t.Fatal("fault-policy call has an invalid memory selector")
								}
								callMem = v
							}
						}
					}
					header := fun.blocks["header"].Preds[0].b
					vectorMem := fun.values["loopmem"].Args[0]
					if callMem == nil || vectorMem.Op != OpPhi || vectorMem.Block != header || vectorMem.Args[0] != callMem {
						t.Fatal("vector prefix and scalar fallback do not receive the call's memory")
					}
					for _, block := range f.Blocks {
						for _, v := range block.Values {
							if v.Op == OpConvert && v.Args[1] != callMem {
								t.Fatal("alias pointer conversion precedes the fault-policy call")
							}
						}
					}
					bound := header.Controls[0].Args[1]
					for _, paniconfault := range []bool{false, true} {
						for _, addresses := range [][2]uint64{{0x1000, 0x1000}, {0x1000, 0x1001}, {0x1000, 0x2000}, {0x2000, 0x1000}} {
							for _, count := range []uint64{0, 1, 3, 4, 5, 8, 9, 17} {
								eval := autovecGuardEvaluator{
									t: t,
									inputs: map[*Value]uint64{
										fun.values["dst"]:   addresses[0],
										fun.values["src"]:   addresses[1],
										fun.values["limit"]: count,
									},
									paniconfault: paniconfault,
								}
								want := count &^ 3
								if paniconfault || addresses[1] == addresses[0]+1 {
									want = 0
								}
								if got := eval.integer(bound); got != want {
									t.Fatalf("paniconfault=%t dst=%x src=%x count=%d: vector bound=%d, want %d", paniconfault, addresses[0], addresses[1], count, got, want)
								}
							}
						}
					}
				})
			}
		}
	}
}

func TestLogosAutovecLeavesNosplitScalar(t *testing.T) {
	autovecTestSettings(t)
	for _, arch := range []string{"amd64", "arm64"} {
		t.Run(arch, func(t *testing.T) {
			fun := autovecTestLoop(t, arch)
			fun.f.NoSplit = true
			before := fun.f.String()
			autovec(fun.f)
			CheckFunc(fun.f)
			if fun.f.String() != before {
				t.Fatal("autovectorization introduced a getg-based call into a nosplit function")
			}
		})
	}
}
