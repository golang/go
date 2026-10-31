// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"fmt"
	"testing"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/src"
)

func BenchmarkMemcombineSharedOR(b *testing.B) {
	for _, n := range []int{8, 16, 22} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			c := testConfig(b)
			f := c.config.NewFunc(c.Frontend(), new(ssa.Cache))
			entry := f.NewBlock(block.BlockExit)
			f.Entry = entry
			typ := c.config.Types.UInt64
			x := entry.NewValue0(src.NoXPos, ssaop.OpArg, typ)
			y := entry.NewValue0(src.NoXPos, ssaop.OpArg, typ)
			v := entry.NewValue2(src.NoXPos, ssaop.OpOr64, typ, x, y)
			// Each level uses the previous result through two parents.
			// Expanding this graph as a tree takes exponential work.
			for range n {
				a := entry.NewValue2(src.NoXPos, ssaop.OpOr64, typ, v, x)
				b := entry.NewValue2(src.NoXPos, ssaop.OpOr64, typ, v, y)
				v = entry.NewValue2(src.NoXPos, ssaop.OpOr64, typ, a, b)
			}
			b.ReportAllocs()
			for b.Loop() {
				memcombineLoads(f, nil)
			}
		})
	}
}
