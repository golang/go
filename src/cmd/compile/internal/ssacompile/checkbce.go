// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

// checkbce prints all bounds checks that are present in the function.
// Useful to find regressions. checkbce is only activated when with
// corresponding debug options, so it's off by default.
// See test/checkbce.go
func checkbce(f *ssacore.Func) {
	if f.Pass.Debug <= 0 && !logopt.Enabled() {
		return
	}

	for _, b := range f.Blocks {
		if b.Kind == block.BlockInvalid {
			continue
		}
		for _, v := range b.Values {
			if v.Op == ssaop.OpIsInBounds || v.Op == ssaop.OpIsSliceInBounds {
				if f.Pass.Debug > 0 {
					f.Warnl(v.Pos, "Found %v", v.Op)
				}
				if logopt.Enabled() {
					if v.Op == ssaop.OpIsInBounds {
						logopt.LogOpt(v.Pos, "isInBounds", "checkbce", f.Name)
					}
					if v.Op == ssaop.OpIsSliceInBounds {
						logopt.LogOpt(v.Pos, "isSliceInBounds", "checkbce", f.Name)
					}
				}
			}
		}
	}
}
