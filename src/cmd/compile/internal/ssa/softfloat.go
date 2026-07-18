// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/ssaop"
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
				f.UnCache(v)
				switch v.Op {
				case ssaop.OpPhi, ssaop.OpLoad, ssaop.OpArg:
					if v.Type.Size() == 4 {
						v.Type = f.Config.Types.UInt32
					} else {
						v.Type = f.Config.Types.UInt64
					}
				case ssaop.OpConst32F:
					v.Op = ssaop.OpConst32
					v.Type = f.Config.Types.UInt32
					v.AuxInt = int64(int32(math.Float32bits(auxTo32F(v.AuxInt))))
				case ssaop.OpConst64F:
					v.Op = ssaop.OpConst64
					v.Type = f.Config.Types.UInt64
				case ssaop.OpNeg32F:
					arg0 := v.Args[0]
					v.Reset(ssaop.OpXor32)
					v.Type = f.Config.Types.UInt32
					v.AddArg(arg0)
					mask := v.Block.NewValue0(v.Pos, ssaop.OpConst32, v.Type)
					mask.AuxInt = -0x80000000
					v.AddArg(mask)
				case ssaop.OpNeg64F:
					arg0 := v.Args[0]
					v.Reset(ssaop.OpXor64)
					v.Type = f.Config.Types.UInt64
					v.AddArg(arg0)
					mask := v.Block.NewValue0(v.Pos, ssaop.OpConst64, v.Type)
					mask.AuxInt = -0x8000000000000000
					v.AddArg(mask)
				case ssaop.OpRound32F:
					v.Op = ssaop.OpCopy
					v.Type = f.Config.Types.UInt32
				case ssaop.OpRound64F:
					v.Op = ssaop.OpCopy
					v.Type = f.Config.Types.UInt64
				}
				newInt64 = newInt64 || v.Type.Size() == 8
			} else if (v.Op == ssaop.OpStore || v.Op == ssaop.OpZero || v.Op == ssaop.OpMove) && v.Aux.(*types.Type).IsFloat() {
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
		applyRewrite(f, rewriteBlockdec64, rewriteValuedec64, RemoveDeadValues)
	}

}
