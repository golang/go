// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// findScope returns the most specific scope containing pos.
func findScope(pos src.XPos, scopes []dwarf.Scope) int32 {
	if !pos.IsKnown() {
		return 0
	}
	for i := len(scopes) - 1; i > 0; i-- {
		if pos.After(scopes[i].Start) && pos.Before(scopes[i].End) {
			return int32(i)
		}
	}
	return 0
}

type scopedProg struct {
	p     *obj.Prog
	scope int32
}

// scopeRanges calculates scope ranges for symbol fnsym.
func scopeRanges(fnsym *obj.LSym, scopes []dwarf.Scope) {
	var sp []scopedProg

	for p := fnsym.Text; p != nil; p = p.Link {
		sp = append(sp, scopedProg{p, -1})
	}

	for scopeID := int32(len(scopes) - 1); scopeID >= 0; scopeID-- {
		if scope := &scopes[scopeID]; scope.Start.IsKnown() && scope.End.IsKnown() {
			scopeProgs(sp, scopeID, scope.Start, scope.End)
		}
	}

	scopedProgsToRanges(sp, scopes, fnsym.Size)

	// Propagate scope's pc ranges to parent
	for i := len(scopes) - 1; i > 0; i-- {
		cur := &scopes[i]
		if scopes[i].Parent != 0 {
			parent := &scopes[scopes[i].Parent]
			parent.UnifyRanges(cur)
		}
	}
}

// scopeProgs marks all scopedProgs between start and end that don't already
// belong to a scope as belonging to scopeId.
func scopeProgs(sp []scopedProg, scopeId int32, start, end src.XPos) {
	for i := range sp {
		if sp[i].scope >= 0 {
			continue
		}
		if pos := sp[i].p.Pos; pos.After(start) && pos.Before(end) {
			sp[i].scope = scopeId
		}
	}
}

// scopedProgsToRanges scans sp and collects in the Ranges field of each
// scope the start and end instruction of the scope.
func scopedProgsToRanges(sp []scopedProg, scopes []dwarf.Scope, symSize int64) {
	var curscope int32 = -1
	for i := range sp {
		if sp[i].scope == curscope {
			continue
		}
		if curscope >= 0 {
			curranges := scopes[curscope].Ranges
			curranges[len(curranges)-1].End = sp[i].p.Pc
		}
		curscope = sp[i].scope
		if curscope >= 0 {
			scopes[curscope].Ranges = append(scopes[curscope].Ranges, dwarf.Range{Start: sp[i].p.Pc, End: -1})
		}
	}
	if curscope >= 0 {
		curranges := scopes[curscope].Ranges
		curranges[len(curranges)-1].End = symSize
	}
}
