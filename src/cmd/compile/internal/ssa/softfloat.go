// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"math"
)

func softfloat(f *Func) {
	if !f.Config.SoftFloat {
		return
	}
	newInt64 := false

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Type.IsFloat() {
				f.unCache(v)
				switch v.Op {
				case OpPhi, OpLoad, OpArg:
					if v.Type.Size() == 4 {
						v.Type = f.Config.Types.UInt32
					} else {
						v.Type = f.Config.Types.UInt64
					}
				case OpConst32F:
					v.Op = OpConst32
					v.Type = f.Config.Types.UInt32
					v.AuxInt = int64(int32(math.Float32bits(auxTo32F(v.AuxInt))))
				case OpConst64F:
					v.Op = OpConst64
					v.Type = f.Config.Types.UInt64
				case OpNeg32F:
					arg0 := v.Args[0]
					v.reset(OpXor32)
					v.Type = f.Config.Types.UInt32
					v.AddArg(arg0)
					mask := v.Block.NewValue0(v.Pos, OpConst32, v.Type)
					mask.AuxInt = -0x80000000
					v.AddArg(mask)
				case OpNeg64F:
					arg0 := v.Args[0]
					v.reset(OpXor64)
					v.Type = f.Config.Types.UInt64
					v.AddArg(arg0)
					mask := v.Block.NewValue0(v.Pos, OpConst64, v.Type)
					mask.AuxInt = -0x8000000000000000
					v.AddArg(mask)
				case OpRound32F:
					v.Op = OpCopy
					v.Type = f.Config.Types.UInt32
				case OpRound64F:
					v.Op = OpCopy
					v.Type = f.Config.Types.UInt64
				}
				newInt64 = newInt64 || v.Type.Size() == 8
			} else if (v.Op == OpStore || v.Op == OpZero || v.Op == OpMove) && v.Aux.(*types.Type).IsFloat() {
				switch size := v.Aux.(*types.Type).Size(); size {
				case 4:
					v.Aux = f.Config.Types.UInt32
				case 8:
					v.Aux = f.Config.Types.UInt64
					newInt64 = true
				default:
					v.Fatalf("bad float type with size %d", size)
				}
			}
		}
	}

	if newInt64 && f.Config.RegSize == 4 {
		// On 32bit arch, decompose Uint64 introduced in the switch above.
		decomposeBuiltin(f)
		applyRewrite(f, rewriteBlockdec64, rewriteValuedec64, removeDeadValues)
	}

}
