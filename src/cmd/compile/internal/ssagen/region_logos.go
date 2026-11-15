// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"go/constant"
	"internal/buildcfg"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/sys"
)

type LogosRegionGrowInfo struct {
	H ir.Node
	G ir.Node
}

var LogosRegionAppends sync.Map

func logosRegionBumpBuilder(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	if !buildcfg.Experiment.LogosArena || Arch.LinkArch.Family != sys.AMD64 || base.Flag.Race {
		return s.constNil(types.Types[types.TUNSAFEPTR])
	}
	if len(args) > 0 && args[0].Op == ssa.OpConstNil {
		return s.constNil(types.Types[types.TUNSAFEPTR])
	}
	var size int64
	if len(args) > 2 && (args[2].Op == ssa.OpConst64 || args[2].Op == ssa.OpConst32) {
		size = args[2].AuxInt
	} else if len(n.Args) > 2 && ir.IsConst(n.Args[2], constant.Int) {
		size = ir.Int64Val(n.Args[2])
	} else {
		return s.constNil(types.Types[types.TUNSAFEPTR])
	}
	if size <= 0 || size > LogosArenaBumpMaxBytes {
		return s.constNil(types.Types[types.TUNSAFEPTR])
	}
	rnd := (size + 7) &^ 7
	mem := s.mem()
	bumpTyp := types.NewTuple(types.Types[types.TUNSAFEPTR], types.TypeMem)
	bump := s.newValue3I(ssa.OpLogosRegionBump, bumpTyp, rnd, args[0], args[1], mem)
	ptr := s.newValue1(ssa.OpSelect0, types.Types[types.TUNSAFEPTR], bump)
	s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, bump)
	return ptr
}

func logosRegisterRegion(add func(string, string, intrinsicBuilder, ...*sys.Arch)) {
	add("runtime", "logosRegionBump", logosRegionBumpBuilder, sys.ArchAMD64)
}

func (s *state) logosRegionAppendSlice(n *ir.CallExpr) *LogosRegionGrowInfo {
	v, ok := LogosRegionAppends.Load(n)
	if !ok {
		return nil
	}
	info := v.(LogosRegionGrowInfo)
	return &info
}

func (s *state) logosRegionGrowCall(n *ir.CallExpr) (*obj.LSym, *ssa.Value, *ssa.Value) {
	info := s.logosRegionAppendSlice(n)
	if info == nil {
		return nil, nil, nil
	}
	h := s.expr(info.H)
	g := s.expr(info.G)
	return typecheck.LookupRuntimeFunc(LogosRegionGrowSliceEntry(n.Type().Elem())), h, g
}

func LogosRegionGrowSliceEntry(elem *types.Type) string {
	if elem.HasPointers() {
		return "logosRegionGrowSliceScan"
	}
	return "logosRegionGrowSliceNoscan"
}
