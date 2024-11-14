// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarfgen

import (
	"fmt"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// To identify variables by original source position.
type varPos struct {
	DeclName string
	DeclFile string
	DeclLine uint
	DeclCol  uint
}

// This is the main entry point for collection of raw material to
// drive generation of DWARF "inlined subroutine" DIEs. See proposal
// 22080 for more details and background info.
func assembleInlines(fnsym *obj.LSym, dwVars []*dwarf.Var) dwarf.InlCalls {
	var inlcalls dwarf.InlCalls

	if base.Debug.DwarfInl != 0 {
		base.Ctxt.Logf("assembling DWARF inlined routine info for %v\n", fnsym.Name)
	}

	// This maps inline index (from Ctxt.InlTree) to index in inlcalls.Calls
	imap := make(map[int]int)

	// Walk progs to build up the InlCalls data structure
	var prevpos src.XPos
	for p := fnsym.Func().Text; p != nil; p = p.Link {
		if p.Pos == prevpos {
			continue
		}
		ii := posInlIndex(p.Pos)
		if ii >= 0 {
			insertInlCall(&inlcalls, ii, imap)
		}
		prevpos = p.Pos
	}

	// This is used to partition DWARF vars by inline index. Vars not
	// produced by the inliner will wind up in the vmap[0] entry.
	vmap := make(map[int32][]*dwarf.Var)

	// Now walk the dwarf vars and partition them based on whether they
	// were produced by the inliner (dwv.InlIndex > 0) or were original
	// vars/params from the function (dwv.InlIndex == 0).
	for _, dwv := range dwVars {

		vmap[dwv.InlIndex] = append(vmap[dwv.InlIndex], dwv)

		// Zero index => var was not produced by an inline
		if dwv.InlIndex == 0 {
			continue
		}

		// Look up index in our map, then tack the var in question
		// onto the vars list for the correct inlined call.
		ii := int(dwv.InlIndex) - 1
		idx, ok := imap[ii]
		if !ok {
			// We can occasionally encounter a var produced by the
			// inliner for which there is no remaining prog; add a new
			// entry to the call list in this scenario.
			idx = insertInlCall(&inlcalls, ii, imap)
		}
		inlcalls.Calls[idx].InlVars =
			append(inlcalls.Calls[idx].InlVars, dwv)
	}

	// Post process the map above to assign child indices to vars.
	//
	// A given variable is treated differently depending on whether it
	// is part of the top-level function (ii == 0) or if it was
	// produced as a result of an inline (ii != 0).
	//
	// If a variable was not produced by an inline and its containing
	// function was not inlined, then we just assign an ordering of
	// based on variable name.
	//
	// If a variable was not produced by an inline and its containing
	// function was inlined, then we need to assign a child index
	// based on the order of vars in the abstract function (in
	// addition, those vars that don't appear in the abstract
	// function, such as "~r1", are flagged as such).
	//
	// If a variable was produced by an inline, then we locate it in
	// the pre-inlining decls for the target function and assign child
	// index accordingly.
	for ii, sl := range vmap {
		var m map[varPos]int
		if ii == 0 {
			if !fnsym.WasInlined() {
				for j, v := range sl {
					v.ChildIndex = int32(j)
				}
				continue
			}
			m = makePreinlineDclMap(fnsym)
		} else {
			ifnlsym := base.Ctxt.InlTree.InlinedFunction(int(ii - 1))
			m = makePreinlineDclMap(ifnlsym)
		}

		// Here we assign child indices to variables based on
		// pre-inlined decls, and set the "IsInAbstract" flag
		// appropriately. In addition: parameter and local variable
		// names are given "middle dot" version numbers as part of the
		// writing them out to export data (see issue 4326). If DWARF
		// inlined routine generation is turned on, we want to undo
		// this versioning, since DWARF variables in question will be
		// parented by the inlined routine and not the top-level
		// caller.
		synthCount := len(m)
		for _, v := range sl {
			vp := varPos{
				DeclName: v.Name,
				DeclFile: v.DeclFile,
				DeclLine: v.DeclLine,
				DeclCol:  v.DeclCol,
			}
			synthesized := strings.HasPrefix(v.Name, "~") || v.Name == "_"
			if idx, found := m[vp]; found {
				v.ChildIndex = int32(idx)
				v.IsInAbstract = !synthesized
			} else {
				// Variable can't be found in the pre-inline dcl list.
				// In the top-level case (ii=0) this can happen
				// because a composite variable was split into pieces,
				// and we're looking at a piece. We can also see
				// return temps (~r%d) that were created during
				// lowering, or unnamed params ("_").
				v.ChildIndex = int32(synthCount)
				synthCount++
			}
		}
	}

	// Make a second pass through the progs to compute PC ranges for
	// the various inlined calls.
	start := int64(-1)
	curii := -1
	var prevp *obj.Prog
	for p := fnsym.Func().Text; p != nil; prevp, p = p, p.Link {
		if prevp != nil && p.Pos == prevp.Pos {
			continue
		}
		ii := posInlIndex(p.Pos)
		if ii == curii {
			continue
		}
		// Close out the current range
		if start != -1 {
			addRange(inlcalls.Calls, start, p.Pc, curii, imap)
		}
		// Begin new range
		start = p.Pc
		curii = ii
	}
	if start != -1 {
		addRange(inlcalls.Calls, start, fnsym.Size, curii, imap)
	}

	// Issue 33188: if II foo is a child of II bar, then ensure that
	// bar's ranges include the ranges of foo (the loop above will produce
	// disjoint ranges).
	for k, c := range inlcalls.Calls {
		if c.Root {
			unifyCallRanges(inlcalls, k)
		}
	}

	// Debugging
	if base.Debug.DwarfInl != 0 {
		dumpInlCalls(inlcalls)
		dumpInlVars(dwVars)
	}

	// Perform a consistency check on inlined routine PC ranges
	// produced by unifyCallRanges above. In particular, complain in
	// cases where you have A -> B -> C (e.g. C is inlined into B, and
	// B is inlined into A) and the ranges for B are not enclosed
	// within the ranges for A, or C within B.
	for k, c := range inlcalls.Calls {
		if c.Root {
			checkInlCall(fnsym.Name, inlcalls, fnsym.Size, k, -1)
		}
	}

	return inlcalls
}

// Secondary hook for DWARF inlined subroutine generation. This is called
// late in the compilation when it is determined that we need an
// abstract function DIE for an inlined routine imported from a
// previously compiled package.
func AbstractFunc(fn *obj.LSym) {
	ifn := base.Ctxt.DwFixups.GetPrecursorFunc(fn)
	if ifn == nil {
		base.Ctxt.Diag("failed to locate precursor fn for %v", fn)
		return
	}
	_ = ifn.(*ir.Func)
	if base.Debug.DwarfInl != 0 {
		base.Ctxt.Logf("DwarfAbstractFunc(%v)\n", fn.Name)
	}
	base.Ctxt.DwarfAbstractFunc(ifn, fn)
}

// Given a function that was inlined as part of the compilation, dig
// up the pre-inlining DCL list for the function and create a map that
// supports lookup of pre-inline dcl index, based on variable
// position/name. NB: the recipe for computing variable pos/file/line
// needs to be kept in sync with the similar code in gc.createSimpleVars
// and related functions.
func makePreinlineDclMap(fnsym *obj.LSym) map[varPos]int {
	dcl := preInliningDcls(fnsym)
	m := make(map[varPos]int)
	for i, n := range dcl {
		pos := base.Ctxt.InnermostPos(n.Pos())
		vp := varPos{
			DeclName: n.Sym().Name,
			DeclFile: pos.RelFilename(),
			DeclLine: pos.RelLine(),
			DeclCol:  pos.RelCol(),
		}
		if _, found := m[vp]; found {
			// We can see collisions (variables with the same name/file/line/col) in obfuscated or machine-generated code -- see issue 44378 for an example. Skip duplicates in such cases, since it is unlikely that a human will be debugging such code.
			continue
		}
		m[vp] = i
	}
	return m
}

func insertInlCall(dwcalls *dwarf.InlCalls, inlIdx int, imap map[int]int) int {
	callIdx, found := imap[inlIdx]
	if found {
		return callIdx
	}

	// Haven't seen this inline yet. Visit parent of inline if there
	// is one. We do this first so that parents appear before their
	// children in the resulting table.
	parCallIdx := -1
	parInlIdx := base.Ctxt.InlTree.Parent(inlIdx)
	if parInlIdx >= 0 {
		parCallIdx = insertInlCall(dwcalls, parInlIdx, imap)
	}

	// Create new entry for this inline
	inlinedFn := base.Ctxt.InlTree.InlinedFunction(inlIdx)
	callXPos := base.Ctxt.InlTree.CallPos(inlIdx)
	callPos := base.Ctxt.InnermostPos(callXPos)
	absFnSym := base.Ctxt.DwFixups.AbsFuncDwarfSym(inlinedFn)
	ic := dwarf.InlCall{
		InlIndex:  inlIdx,
		CallPos:   callPos,
		AbsFunSym: absFnSym,
		Root:      parCallIdx == -1,
	}
	dwcalls.Calls = append(dwcalls.Calls, ic)
	callIdx = len(dwcalls.Calls) - 1
	imap[inlIdx] = callIdx

	if parCallIdx != -1 {
		// Add this inline to parent's child list
		dwcalls.Calls[parCallIdx].Children = append(dwcalls.Calls[parCallIdx].Children, callIdx)
	}

	return callIdx
}

// Given a src.XPos, return its associated inlining index if it
// corresponds to something created as a result of an inline, or -1 if
// there is no inline info. Note that the index returned will refer to
// the deepest call in the inlined stack, e.g. if you have "A calls B
// calls C calls D" and all three callees are inlined (B, C, and D),
// the index for a node from the inlined body of D will refer to the
// call to D from C. Whew.
func posInlIndex(xpos src.XPos) int {
	pos := base.Ctxt.PosTable.Pos(xpos)
	if b := pos.Base(); b != nil {
		ii := b.InliningIndex()
		if ii >= 0 {
			return ii
		}
	}
	return -1
}

func addRange(calls []dwarf.InlCall, start, end int64, ii int, imap map[int]int) {
	if start == -1 {
		panic("bad range start")
	}
	if end == -1 {
		panic("bad range end")
	}
	if ii == -1 {
		return
	}
	if start == end {
		return
	}
	// Append range to correct inlined call
	callIdx, found := imap[ii]
	if !found {
		base.Fatalf("can't find inlIndex %d in imap for prog at %d\n", ii, start)
	}
	call := &calls[callIdx]
	call.Ranges = append(call.Ranges, dwarf.Range{Start: start, End: end})
}

func dumpInlCall(inlcalls dwarf.InlCalls, idx, ilevel int) {
	for i := 0; i < ilevel; i++ {
		base.Ctxt.Logf("  ")
	}
	ic := inlcalls.Calls[idx]
	callee := base.Ctxt.InlTree.InlinedFunction(ic.InlIndex)
	base.Ctxt.Logf("  %d: II:%d (%s) V: (", idx, ic.InlIndex, callee.Name)
	for _, f := range ic.InlVars {
		base.Ctxt.Logf(" %v", f.Name)
	}
	base.Ctxt.Logf(" ) C: (")
	for _, k := range ic.Children {
		base.Ctxt.Logf(" %v", k)
	}
	base.Ctxt.Logf(" ) R:")
	for _, r := range ic.Ranges {
		base.Ctxt.Logf(" [%d,%d)", r.Start, r.End)
	}
	base.Ctxt.Logf("\n")
	for _, k := range ic.Children {
		dumpInlCall(inlcalls, k, ilevel+1)
	}

}

func dumpInlCalls(inlcalls dwarf.InlCalls) {
	for k, c := range inlcalls.Calls {
		if c.Root {
			dumpInlCall(inlcalls, k, 0)
		}
	}
}

func dumpInlVars(dwvars []*dwarf.Var) {
	for i, dwv := range dwvars {
		typ := "local"
		if dwv.Tag == dwarf.DW_TAG_formal_parameter {
			typ = "param"
		}
		ia := 0
		if dwv.IsInAbstract {
			ia = 1
		}
		base.Ctxt.Logf("V%d: %s CI:%d II:%d IA:%d %s\n", i, dwv.Name, dwv.ChildIndex, dwv.InlIndex-1, ia, typ)
	}
}

func rangesContains(par []dwarf.Range, rng dwarf.Range) (bool, string) {
	for _, r := range par {
		if rng.Start >= r.Start && rng.End <= r.End {
			return true, ""
		}
	}
	msg := fmt.Sprintf("range [%d,%d) not contained in {", rng.Start, rng.End)
	for _, r := range par {
		msg += fmt.Sprintf(" [%d,%d)", r.Start, r.End)
	}
	msg += " }"
	return false, msg
}

func rangesContainsAll(parent, child []dwarf.Range) (bool, string) {
	for _, r := range child {
		c, m := rangesContains(parent, r)
		if !c {
			return false, m
		}
	}
	return true, ""
}

// checkInlCall verifies that the PC ranges for inline info 'idx' are
// enclosed/contained within the ranges of its parent inline (or if
// this is a root/toplevel inline, checks that the ranges fall within
// the extent of the top level function). A panic is issued if a
// malformed range is found.
func checkInlCall(funcName string, inlCalls dwarf.InlCalls, funcSize int64, idx, parentIdx int) {

	// Callee
	ic := inlCalls.Calls[idx]
	callee := base.Ctxt.InlTree.InlinedFunction(ic.InlIndex).Name
	calleeRanges := ic.Ranges

	// Caller
	caller := funcName
	parentRanges := []dwarf.Range{{Start: int64(0), End: funcSize}}
	if parentIdx != -1 {
		pic := inlCalls.Calls[parentIdx]
		caller = base.Ctxt.InlTree.InlinedFunction(pic.InlIndex).Name
		parentRanges = pic.Ranges
	}

	// Callee ranges contained in caller ranges?
	c, m := rangesContainsAll(parentRanges, calleeRanges)
	if !c {
		base.Fatalf("** malformed inlined routine range in %s: caller %s callee %s II=%d %s\n", funcName, caller, callee, idx, m)
	}

	// Now visit kids
	for _, k := range ic.Children {
		checkInlCall(funcName, inlCalls, funcSize, k, idx)
	}
}

// unifyCallRanges ensures that the ranges for a given inline
// transitively include all of the ranges for its child inlines.
func unifyCallRanges(inlcalls dwarf.InlCalls, idx int) {
	ic := &inlcalls.Calls[idx]
	for _, childIdx := range ic.Children {
		// First make sure child ranges are unified.
		unifyCallRanges(inlcalls, childIdx)

		// Then merge child ranges into ranges for this inline.
		cic := inlcalls.Calls[childIdx]
		ic.Ranges = dwarf.MergeRanges(ic.Ranges, cic.Ranges)
	}
}
