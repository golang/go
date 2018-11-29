// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/src"
	"sort"
)

// See golang.org/issue/20390.
func xposBefore(p, q src.XPos) bool {
	return Ctxt.PosTable.Pos(p).Before(Ctxt.PosTable.Pos(q))
}

func findScope(marks []Mark, pos src.XPos) ScopeID {
	i := sort.Search(len(marks), func(i int) bool {
		return xposBefore(pos, marks[i].Pos)
	})
	if i == 0 {
		return 0
	}
	return marks[i-1].Scope
}

func assembleScopes(fnsym *obj.LSym, fn *Node, dwarfVars []*dwarf.Var, varScopes []ScopeID) []dwarf.Scope {
	// Initialize the DWARF scope tree based on lexical scopes.
	dwarfScopes := make([]dwarf.Scope, 1+len(fn.Func.Parents))
	for i, parent := range fn.Func.Parents {
		dwarfScopes[i+1].Parent = int32(parent)
	}

	scopeVariables(dwarfVars, varScopes, dwarfScopes)
	scopePCs(fnsym, fn.Func.Marks, dwarfScopes)
	return compactScopes(dwarfScopes)
}

// scopeVariables assigns DWARF variable records to their scopes.
func scopeVariables(dwarfVars []*dwarf.Var, varScopes []ScopeID, dwarfScopes []dwarf.Scope) {
	sort.Stable(varsByScopeAndOffset{dwarfVars, varScopes})

	i0 := 0
	for i := range dwarfVars {
		if varScopes[i] == varScopes[i0] {
			continue
		}
		dwarfScopes[varScopes[i0]].Vars = dwarfVars[i0:i]
		i0 = i
	}
	if i0 < len(dwarfVars) {
		dwarfScopes[varScopes[i0]].Vars = dwarfVars[i0:]
	}
}

// A scopedPCs represents a non-empty half-open interval of PCs that
// share a common source position.
type scopedPCs struct {
	start, end int64
	pos        src.XPos
	scope      ScopeID
}

// scopePCs assigns PC ranges to their scopes.
func scopePCs(fnsym *obj.LSym, marks []Mark, dwarfScopes []dwarf.Scope) {
	// If there aren't any child scopes (in particular, when scope
	// tracking is disabled), we can skip a whole lot of work.
	if len(marks) == 0 {
		return
	}

	// Break function text into scopedPCs.
	var pcs []scopedPCs
	p0 := fnsym.Func.Text
	for p := fnsym.Func.Text; p != nil; p = p.Link {
		if p.Pos == p0.Pos {
			continue
		}
		if p0.Pc < p.Pc {
			pcs = append(pcs, scopedPCs{start: p0.Pc, end: p.Pc, pos: p0.Pos})
		}
		p0 = p
	}
	if p0.Pc < fnsym.Size {
		pcs = append(pcs, scopedPCs{start: p0.Pc, end: fnsym.Size, pos: p0.Pos})
	}

	// Assign scopes to each chunk of instructions.
	for i := range pcs {
		pcs[i].scope = findScope(marks, pcs[i].pos)
	}

	// Create sorted PC ranges for each DWARF scope.
	for _, pc := range pcs {
		r := &dwarfScopes[pc.scope].Ranges
		if i := len(*r); i > 0 && (*r)[i-1].End == pc.start {
			(*r)[i-1].End = pc.end
		} else {
			*r = append(*r, dwarf.Range{Start: pc.start, End: pc.end})
		}
	}
}

func compactScopes(dwarfScopes []dwarf.Scope) []dwarf.Scope {
	// Reverse pass to propagate PC ranges to parent scopes.
	for i := len(dwarfScopes) - 1; i > 0; i-- {
		s := &dwarfScopes[i]
		dwarfScopes[s.Parent].UnifyRanges(s)
	}

	return dwarfScopes
}

type varsByScopeAndOffset struct {
	vars   []*dwarf.Var
	scopes []ScopeID
}

func (v varsByScopeAndOffset) Len() int {
	return len(v.vars)
}

func (v varsByScopeAndOffset) Less(i, j int) bool {
	if v.scopes[i] != v.scopes[j] {
		return v.scopes[i] < v.scopes[j]
	}
	return v.vars[i].StackOffset < v.vars[j].StackOffset
}

func (v varsByScopeAndOffset) Swap(i, j int) {
	v.vars[i], v.vars[j] = v.vars[j], v.vars[i]
	v.scopes[i], v.scopes[j] = v.scopes[j], v.scopes[i]
}
