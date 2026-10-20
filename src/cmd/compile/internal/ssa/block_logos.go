// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// MarkColdBranch marks the branch into b unlikely when b has exactly one
// predecessor ending in a two-way branch with no explicit prediction. The
// likely successor is always the other one, so layout places the cold
// successor last. It reports whether a mark was set and never overrides
// an explicit prediction.
func MarkColdBranch(b *Block) bool {
	if len(b.Preds) != 1 {
		return false
	}
	p := b.Preds[0].b
	if len(p.Succs) != 2 || p.Likely != BranchUnknown {
		return false
	}
	if p.Succs[0].b == b {
		p.Likely = BranchUnlikely
		return true
	}
	if p.Succs[1].b == b {
		p.Likely = BranchLikely
		return true
	}
	return false
}
