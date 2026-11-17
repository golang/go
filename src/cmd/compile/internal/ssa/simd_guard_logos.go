// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// usesSIMD reports whether v defines or consumes vector or mask state.
// Such an instruction may need a CPU feature guard even when its result
// is a scalar, as with a lane extraction or mask conversion.
func usesSIMD(v *Value) bool {
	if v.Type.IsSIMD() {
		return true
	}
	for _, a := range v.Args {
		if a.Type.IsSIMD() {
			return true
		}
	}
	return false
}
