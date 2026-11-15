// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"testing"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func TestLogosRegionRecordStoresSkipsWithoutRegions(t *testing.T) {
	enableLogosArena(t)
	sig := types.NewSignature(nil, nil, nil)
	fn := ir.NewFunc(src.NoXPos, src.NoXPos, types.LocalPkg.Lookup("testfn"), sig)
	dst := ir.NewNameAt(src.NoXPos, types.LocalPkg.Lookup("dst"), types.Types[types.TINT])
	dst.Class = ir.PAUTO
	dst.Curfn = fn
	srcName := ir.NewNameAt(src.NoXPos, types.LocalPkg.Lookup("src"), types.Types[types.TINT])
	srcName.Class = ir.PAUTO
	srcName.SetType(types.Types[types.TINT])
	srcName.Curfn = fn
	fn.Dcl = []*ir.Name{dst, srcName}
	var b batch
	b.logosRegionInit()
	b.initFunc(fn)
	e := b.with(fn)
	ks := e.addrs([]ir.Node{dst})
	e.logosRegionRecordStores([]ir.Node{dst}, []ir.Node{srcName}, ks)
	if len(b.logosRegion.flowDesc) != 0 {
		t.Fatalf("flowDesc has %d entries without regions", len(b.logosRegion.flowDesc))
	}
}
