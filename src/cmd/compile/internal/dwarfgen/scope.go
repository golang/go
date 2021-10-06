// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarfgen

import (
	"sort"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// See golang.org/issue/20390.
func xposBefore(p, q src.XPos) bool {
	return base.Ctxt.PosTable.Pos(p).Before(base.Ctxt.PosTable.Pos(q))
}

func findScope(marks []ir.Mark, pos src.XPos) ir.ScopeID {
	i := sort.Search(len(marks), func(i int) bool {
		return xposBefore(pos, marks[i].Pos)
	})
	if i == 0 {
		return 0
	}
	return marks[i-1].Scope
}

func assembleScopes(fnsym *obj.LSym, fn *ir.Func, dwarfVars []*dwarf.Var, varScopes []ir.ScopeID) []dwarf.Scope {
	// Initialize the DWARF scope tree based on lexical scopes.
	dwarfScopes := make([]dwarf.Scope, 1+len(fn.Parents))
	for i, parent := range fn.Parents {
		dwarfScopes[i+1].Parent = int32(parent)
	}

	scopeVariables(dwarfVars, varScopes, dwarfScopes, fnsym.ABI() != obj.ABI0)
	if fnsym.Func().Text != nil {
		scopePCs(fnsym, fn.Marks, dwarfScopes)
	}
	return compactScopes(dwarfScopes)
}

// scopeVariables assigns DWARF variable records to their scopes.
func scopeVariables(dwarfVars []*dwarf.Var, varScopes []ir.ScopeID, dwarfScopes []dwarf.Scope, regabi bool) {
	if regabi {
		sort.Stable(varsByScope{dwarfVars, varScopes})
	} else {
		sort.Stable(varsByScopeAndOffset{dwarfVars, varScopes})
	}

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

// scopePCs assigns PC ranges to their scopes.
func scopePCs(fnsym *obj.LSym, marks []ir.Mark, dwarfScopes []dwarf.Scope) {
	// If there aren't any child scopes (in particular, when scope
	// tracking is disabled), we can skip a whole lot of work.
	if len(marks) == 0 {
		return
	}
	p0 := fnsym.Func().Text
	scope := findScope(marks, p0.Pos)
	for p := p0; p != nil; p = p.Link {
		if p.Pos == p0.Pos {
			continue
		}
		dwarfScopes[scope].AppendRange(dwarf.Range{Start: p0.Pc, End: p.Pc})
		p0 = p
		scope = findScope(marks, p0.Pos)
	}
	if p0.Pc < fnsym.Size {
		dwarfScopes[scope].AppendRange(dwarf.Range{Start: p0.Pc, End: fnsym.Size})
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
	scopes []ir.ScopeID
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

type varsByScope struct {
	vars   []*dwarf.Var
	scopes []ir.ScopeID
}

func (v varsByScope) Len() int {
	return len(v.vars)
}

func (v varsByScope) Less(i, j int) bool {
	return v.scopes[i] < v.scopes[j]
}

func (v varsByScope) Swap(i, j int) {
	v.vars[i], v.vars[j] = v.vars[j], v.vars[i]
	v.scopes[i], v.scopes[j] = v.scopes[j], v.scopes[i]
}
