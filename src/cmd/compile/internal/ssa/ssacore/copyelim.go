// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import "cmd/compile/internal/ssa/ssaop"

// PhiElimValue tries to convert the phi v to a copy.
func PhiElimValue(v *Value) bool {
	if v.Op != ssaop.OpPhi {
		return false
	}

	// If there are two distinct args of v which
	// are not v itself, then the phi must remain.
	// Otherwise, we can replace it with a copy.
	var w *Value
	for _, x := range v.Args {
		if x == v {
			continue
		}
		if x == w {
			continue
		}
		if w != nil {
			return false
		}
		w = x
	}

	if w == nil {
		// v references only itself. It must be in
		// a dead code loop. Don't bother modifying it.
		return false
	}
	v.Op = ssaop.OpCopy
	v.SetArgs1(w)
	f := v.Block.Func
	if f.Pass.Debug > 0 {
		f.Warnl(v.Pos, "eliminated phi")
	}
	return true
}
