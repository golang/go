// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/logopt"

// checkbce prints all bounds checks that are present in the function.
// Useful to find regressions. checkbce is only activated when with
// corresponding debug options, so it's off by default.
// See test/checkbce.go
func checkbce(f *Func) {
	if f.pass.debug <= 0 && !logopt.Enabled() {
		return
	}

	for _, b := range f.Blocks {
		if b.Kind == BlockInvalid {
			continue
		}
		for _, v := range b.Values {
			if v.Op == OpIsInBounds || v.Op == OpIsSliceInBounds {
				if f.pass.debug > 0 {
					f.Warnl(v.Pos, "Found %v", v.Op)
				}
				if logopt.Enabled() {
					if v.Op == OpIsInBounds {
						logopt.LogOpt(v.Pos, "isInBounds", "checkbce", f.Name)
					}
					if v.Op == OpIsSliceInBounds {
						logopt.LogOpt(v.Pos, "isSliceInBounds", "checkbce", f.Name)
					}
				}
			}
		}
	}
}
