// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
)

func logosLoopStride(f *Func) int64 {
	if f.Config.arch != "wasm" || base.Debug.LogosLoopStride <= 0 || base.Ctxt.Pkgpath == "runtime" {
		return 0
	}
	return int64(base.Debug.LogosLoopStride)
}

type loopCounter struct {
	phi    *Value
	stride int64
}

func logosHeaderDominates(idom []*Block, h, bb *Block) bool {
	for b := bb; b != nil; b = idom[b.ID] {
		if b == h {
			return true
		}
	}
	return false
}

func logosLoopCounters(f *Func, backEdges []edgeMem, stride int64) map[*Block]*loopCounter {
	if stride <= 0 {
		return nil
	}
	counters := make(map[*Block]*loopCounter, len(backEdges))
	for _, emc := range backEdges {
		h := emc.e.b
		if h == f.Entry || counters[h] != nil {
			continue
		}
		phi := h.NewValue0(h.Pos, OpPhi, f.Config.Types.UInt64)
		for _, p := range h.Preds {
			phi.AddArg(p.b.NewValue0I(p.b.Pos, OpConst64, f.Config.Types.UInt64, 0))
		}
		counters[h] = &loopCounter{phi: phi, stride: stride}
	}
	return counters
}

func logosLoopStrideGate(f *Func, lc *loopCounter, test, sched *Block, i int) {
	phi := lc.phi
	cfg := &f.Config.Types

	test.SetControl(test.NewValue2(test.Pos, OpEq64, cfg.Bool, phi,
		test.NewValue0I(test.Pos, OpConst64, cfg.UInt64, lc.stride-1)))

	inc := test.NewValue2(test.Pos, OpAdd64, cfg.UInt64, phi,
		test.NewValue0I(test.Pos, OpConst64, cfg.UInt64, 1))
	phi.SetArg(i, inc)
	phi.SetArg(len(phi.Args)-1, sched.NewValue0I(sched.Pos, OpConst64, cfg.UInt64, 0))
}
