// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

func logosPairHoistEdge(b, succ *Block) bool {
	return len(b.Succs) == 1 && len(succ.Preds) == 1
}
