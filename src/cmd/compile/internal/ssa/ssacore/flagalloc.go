// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import "cmd/compile/internal/ssa/ssaop"

func (v *Value) ClobbersFlags() bool {
	if ssaop.OpcodeTable[v.Op].ClobberFlags {
		return true
	}
	if v.Type.IsTuple() && (v.Type.FieldType(0).IsFlags() || v.Type.FieldType(1).IsFlags()) {
		// This case handles the possibility where a flag value is generated but never used.
		// In that case, there's no corresponding Select to overwrite the flags value,
		// so we must consider flags clobbered by the tuple-generating instruction.
		return true
	}
	return false
}
