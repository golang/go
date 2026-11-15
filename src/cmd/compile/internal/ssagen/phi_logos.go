// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import "cmd/compile/internal/ssa"

func (s *phiState) computeMinSuccessorLevels() {
	if len(s.f.Blocks) > 1<<31-1 {
		s.s.Fatalf("too many blocks for phi dominator levels")
	}
	noSuccessorLevel := int32(len(s.f.Blocks))
	s.minSL = make([]int32, s.f.NumBlocks())
	for _, b := range s.f.Blocks {
		minLevel := noSuccessorLevel
		for _, e := range b.Succs {
			minLevel = min(minLevel, s.level[e.Block().ID])
		}
		s.minSL[b.ID] = minLevel
	}
	for _, b := range s.f.Postorder() {
		if p := s.idom[b.ID]; p != nil {
			s.minSL[p.ID] = min(s.minSL[p.ID], s.minSL[b.ID])
		}
	}
}

func (s *phiState) logosPrunePhi(b *ssa.Block, currentRootLevel int32) bool {
	return s.minSL[b.ID] > currentRootLevel
}
