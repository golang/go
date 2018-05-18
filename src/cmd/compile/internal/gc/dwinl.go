// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/src"
	"sort"
	"strings"
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

	if Debug_gendwarfinl != 0 {
		Ctxt.Logf("assembling DWARF inlined routine info for %v\n", fnsym.Name)
	}

	// This maps inline index (from Ctxt.InlTree) to index in inlcalls.Calls
	imap := make(map[int]int)

	// Walk progs to build up the InlCalls data structure
	var prevpos src.XPos
	for p := fnsym.Func.Text; p != nil; p = p.Link {
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
		sort.Sort(byClassThenName(sl))
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
			ifnlsym := Ctxt.InlTree.InlinedFunction(int(ii - 1))
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
			canonName := unversion(v.Name)
			vp := varPos{
				DeclName: canonName,
				DeclFile: v.DeclFile,
				DeclLine: v.DeclLine,
				DeclCol:  v.DeclCol,
			}
			synthesized := strings.HasPrefix(v.Name, "~r") || canonName == "_"
			if idx, found := m[vp]; found {
				v.ChildIndex = int32(idx)
				v.IsInAbstract = !synthesized
				v.Name = canonName
			} else {
				// Variable can't be found in the pre-inline dcl list.
				// In the top-level case (ii=0) this can happen
				// because a composite variable was split into pieces,
				// and we're looking at a piece. We can also see
				// return temps (~r%d) that were created during
				// lowering, or unnamed params ("_").
				v.ChildIndex = int32(synthCount)
				synthCount += 1
			}
		}
	}

	// Make a second pass through the progs to compute PC ranges for
	// the various inlined calls.
	curii := -1
	var crange *dwarf.Range
	var prevp *obj.Prog
	for p := fnsym.Func.Text; p != nil; prevp, p = p, p.Link {
		if prevp != nil && p.Pos == prevp.Pos {
			continue
		}
		ii := posInlIndex(p.Pos)
		if ii == curii {
			continue
		} else {
			// Close out the current range
			endRange(crange, p)

			// Begin new range
			crange = beginRange(inlcalls.Calls, p, ii, imap)
			curii = ii
		}
	}
	if crange != nil {
		crange.End = fnsym.Size
	}

	// Debugging
	if Debug_gendwarfinl != 0 {
		dumpInlCalls(inlcalls)
		dumpInlVars(dwVars)
	}

	return inlcalls
}

// Secondary hook for DWARF inlined subroutine generation. This is called
// late in the compilation when it is determined that we need an
// abstract function DIE for an inlined routine imported from a
// previously compiled package.
func genAbstractFunc(fn *obj.LSym) {
	ifn := Ctxt.DwFixups.GetPrecursorFunc(fn)
	if ifn == nil {
		Ctxt.Diag("failed to locate precursor fn for %v", fn)
		return
	}
	if Debug_gendwarfinl != 0 {
		Ctxt.Logf("DwarfAbstractFunc(%v)\n", fn.Name)
	}
	Ctxt.DwarfAbstractFunc(ifn, fn, myimportpath)
}

// Undo any versioning performed when a name was written
// out as part of export data.
func unversion(name string) string {
	if i := strings.Index(name, "·"); i > 0 {
		name = name[:i]
	}
	return name
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
		pos := Ctxt.InnermostPos(n.Pos)
		vp := varPos{
			DeclName: unversion(n.Sym.Name),
			DeclFile: pos.RelFilename(),
			DeclLine: pos.RelLine(),
			DeclCol:  pos.Col(),
		}
		if _, found := m[vp]; found {
			Fatalf("child dcl collision on symbol %s within %v\n", n.Sym.Name, fnsym.Name)
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
	parInlIdx := Ctxt.InlTree.Parent(inlIdx)
	if parInlIdx >= 0 {
		parCallIdx = insertInlCall(dwcalls, parInlIdx, imap)
	}

	// Create new entry for this inline
	inlinedFn := Ctxt.InlTree.InlinedFunction(inlIdx)
	callXPos := Ctxt.InlTree.CallPos(inlIdx)
	absFnSym := Ctxt.DwFixups.AbsFuncDwarfSym(inlinedFn)
	pb := Ctxt.PosTable.Pos(callXPos).Base()
	callFileSym := Ctxt.Lookup(pb.SymFilename())
	ic := dwarf.InlCall{
		InlIndex:  inlIdx,
		CallFile:  callFileSym,
		CallLine:  uint32(callXPos.Line()),
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
	pos := Ctxt.PosTable.Pos(xpos)
	if b := pos.Base(); b != nil {
		ii := b.InliningIndex()
		if ii >= 0 {
			return ii
		}
	}
	return -1
}

func endRange(crange *dwarf.Range, p *obj.Prog) {
	if crange == nil {
		return
	}
	crange.End = p.Pc
}

func beginRange(calls []dwarf.InlCall, p *obj.Prog, ii int, imap map[int]int) *dwarf.Range {
	if ii == -1 {
		return nil
	}
	callIdx, found := imap[ii]
	if !found {
		Fatalf("internal error: can't find inlIndex %d in imap for prog at %d\n", ii, p.Pc)
	}
	call := &calls[callIdx]

	// Set up range and append to correct inlined call
	call.Ranges = append(call.Ranges, dwarf.Range{Start: p.Pc, End: -1})
	return &call.Ranges[len(call.Ranges)-1]
}

func cmpDwarfVar(a, b *dwarf.Var) bool {
	// named before artificial
	aart := 0
	if strings.HasPrefix(a.Name, "~r") {
		aart = 1
	}
	bart := 0
	if strings.HasPrefix(b.Name, "~r") {
		bart = 1
	}
	if aart != bart {
		return aart < bart
	}

	// otherwise sort by name
	return a.Name < b.Name
}

// byClassThenName implements sort.Interface for []*dwarf.Var using cmpDwarfVar.
type byClassThenName []*dwarf.Var

func (s byClassThenName) Len() int           { return len(s) }
func (s byClassThenName) Less(i, j int) bool { return cmpDwarfVar(s[i], s[j]) }
func (s byClassThenName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func dumpInlCall(inlcalls dwarf.InlCalls, idx, ilevel int) {
	for i := 0; i < ilevel; i++ {
		Ctxt.Logf("  ")
	}
	ic := inlcalls.Calls[idx]
	callee := Ctxt.InlTree.InlinedFunction(ic.InlIndex)
	Ctxt.Logf("  %d: II:%d (%s) V: (", idx, ic.InlIndex, callee.Name)
	for _, f := range ic.InlVars {
		Ctxt.Logf(" %v", f.Name)
	}
	Ctxt.Logf(" ) C: (")
	for _, k := range ic.Children {
		Ctxt.Logf(" %v", k)
	}
	Ctxt.Logf(" ) R:")
	for _, r := range ic.Ranges {
		Ctxt.Logf(" [%d,%d)", r.Start, r.End)
	}
	Ctxt.Logf("\n")
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
		if dwv.Abbrev == dwarf.DW_ABRV_PARAM_LOCLIST || dwv.Abbrev == dwarf.DW_ABRV_PARAM {
			typ = "param"
		}
		ia := 0
		if dwv.IsInAbstract {
			ia = 1
		}
		Ctxt.Logf("V%d: %s CI:%d II:%d IA:%d %s\n", i, dwv.Name, dwv.ChildIndex, dwv.InlIndex-1, ia, typ)
	}
}
