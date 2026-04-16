// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarfgen

import (
	"bytes"
	"flag"
	"fmt"
	"internal/buildcfg"
	"slices"
	"sort"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

func Info(ctxt *obj.Link, fnsym *obj.LSym, infosym *obj.LSym, curfn obj.Func) (scopes []dwarf.Scope, inlcalls dwarf.InlCalls) {
	fn := curfn.(*ir.Func)

	if fn.Nname != nil {
		expect := fn.Linksym()
		if fnsym.ABI() == obj.ABI0 {
			expect = fn.LinksymABI(obj.ABI0)
		}
		if fnsym != expect {
			base.Fatalf("unexpected fnsym: %v != %v", fnsym, expect)
		}
	}

	// Back when there were two different *Funcs for a function, this code
	// was not consistent about whether a particular *Node being processed
	// was an ODCLFUNC or ONAME node. Partly this is because inlined function
	// bodies have no ODCLFUNC node, which was it's own inconsistency.
	// In any event, the handling of the two different nodes for DWARF purposes
	// was subtly different, likely in unintended ways. CL 272253 merged the
	// two nodes' Func fields, so that code sees the same *Func whether it is
	// holding the ODCLFUNC or the ONAME. This resulted in changes in the
	// DWARF output. To preserve the existing DWARF output and leave an
	// intentional change for a future CL, this code does the following when
	// fn.Op == ONAME:
	//
	// 1. Disallow use of createComplexVars in createDwarfVars.
	//    It was not possible to reach that code for an ONAME before,
	//    because the DebugInfo was set only on the ODCLFUNC Func.
	//    Calling into it in the ONAME case causes an index out of bounds panic.
	//
	// 2. Do not populate apdecls. fn.Func.Dcl was in the ODCLFUNC Func,
	//    not the ONAME Func. Populating apdecls for the ONAME case results
	//    in selected being populated after createSimpleVars is called in
	//    createDwarfVars, and then that causes the loop to skip all the entries
	//    in dcl, meaning that the RecordAutoType calls don't happen.
	//
	// These two adjustments keep toolstash -cmp working for now.
	// Deciding the right answer is, as they say, future work.
	//
	// We can tell the difference between the old ODCLFUNC and ONAME
	// cases by looking at the infosym.Name. If it's empty, DebugInfo is
	// being called from (*obj.Link).populateDWARF, which used to use
	// the ODCLFUNC. If it's non-empty (the name will end in $abstract),
	// DebugInfo is being called from (*obj.Link).DwarfAbstractFunc,
	// which used to use the ONAME form.
	isODCLFUNC := infosym.Name == ""

	var apdecls []*ir.Name
	// Populate decls for fn.
	if isODCLFUNC {
		for _, n := range fn.Dcl {
			if n.Op() != ir.ONAME { // might be OTYPE or OLITERAL
				continue
			}
			switch n.Class {
			case ir.PAUTO:
				if !n.Used() {
					// Text == nil -> generating abstract function
					if fnsym.Func().Text != nil {
						base.Fatalf("debuginfo unused node (AllocFrame should truncate fn.Func.Dcl)")
					}
					continue
				}
			case ir.PPARAM, ir.PPARAMOUT:
			default:
				continue
			}
			if !shouldEmitDwarfVar(n) {
				continue
			}
			apdecls = append(apdecls, n)
			if n.Type().Kind() == types.TSSA {
				// Can happen for TypeInt128 types. This only happens for
				// spill locations, so not a huge deal.
				continue
			}
			fnsym.Func().RecordAutoType(reflectdata.TypeLinksym(n.Type()))
		}
	}

	var closureVars map[*ir.Name]int64
	if fn.Needctxt() {
		closureVars = make(map[*ir.Name]int64)
		csiter := typecheck.NewClosureStructIter(fn.ClosureVars)
		for {
			n, _, offset := csiter.Next()
			if n == nil {
				break
			}
			closureVars[n] = offset
			if n.Heapaddr != nil {
				closureVars[n.Heapaddr] = offset
			}
		}
	}

	decls, dwarfVars := createDwarfVars(fnsym, isODCLFUNC, fn, apdecls, closureVars)

	// For each type referenced by the functions auto vars but not
	// already referenced by a dwarf var, attach an R_USETYPE relocation to
	// the function symbol to insure that the type included in DWARF
	// processing during linking.
	// Do the same with R_USEIFACE relocations from the function symbol for the
	// same reason.
	// All these R_USETYPE relocations are only looked at if the function
	// survives deadcode elimination in the linker.
	typesyms := []*obj.LSym{}
	for t := range fnsym.Func().Autot {
		typesyms = append(typesyms, t)
	}
	for i := range fnsym.R {
		if fnsym.R[i].Type == objabi.R_USEIFACE && !strings.HasPrefix(fnsym.R[i].Sym.Name, "go:itab.") {
			// Types referenced through itab will be referenced from somewhere else
			typesyms = append(typesyms, fnsym.R[i].Sym)
		}
	}
	slices.SortFunc(typesyms, func(a, b *obj.LSym) int {
		return strings.Compare(a.Name, b.Name)
	})
	var lastsym *obj.LSym
	for _, sym := range typesyms {
		if sym == lastsym {
			continue
		}
		lastsym = sym
		infosym.AddRel(ctxt, obj.Reloc{Type: objabi.R_USETYPE, Sym: sym})
	}
	fnsym.Func().Autot = nil

	var varScopes []ir.ScopeID
	for _, decl := range decls {
		pos := declPos(decl)
		varScopes = append(varScopes, findScope(fn.Marks, pos))
	}

	scopes = assembleScopes(fnsym, fn, dwarfVars, varScopes)
	if base.Flag.GenDwarfInl > 0 {
		inlcalls = assembleInlines(fnsym, dwarfVars)
	}
	return scopes, inlcalls
}

func declPos(decl *ir.Name) src.XPos {
	return decl.Canonical().Pos()
}

// createDwarfVars process fn, returning a list of DWARF variables and the
// Nodes they represent.
func createDwarfVars(fnsym *obj.LSym, complexOK bool, fn *ir.Func, apDecls []*ir.Name, closureVars map[*ir.Name]int64) ([]*ir.Name, []*dwarf.Var) {
	// Collect a raw list of DWARF vars.
	var vars []*dwarf.Var
	var decls []*ir.Name

	// Build a VarID lookup map for SSA debug info if available.
	var debug *ssa.FuncDebug
	var varIDMap map[*ir.Name]ssa.VarID
	if fn.DebugInfo != nil {
		debug = fn.DebugInfo.(*ssa.FuncDebug)
		varIDMap = make(map[*ir.Name]ssa.VarID, len(debug.Vars))
		for i, n := range debug.Vars {
			varIDMap[n] = ssa.VarID(i)
		}
	}
	canUseComplex := complexOK && debug != nil

	// markVarSeen marks a variable and all its associated slot names as seen.
	// This is needed because decomposed variables may have slots whose ir.Name
	// differs from the variable itself (e.g., PAUTO vs PPARAMOUT for the same
	// logical variable). Without this, the dcl loop could create duplicate
	// conservative entries for names that are already covered by a complex var.
	seen := make(map[*ir.Name]bool)
	markVarSeen := func(n *ir.Name, varID ssa.VarID) {
		seen[n] = true
		if debug != nil && int(varID) < len(debug.VarSlots) {
			for _, slot := range debug.VarSlots[varID] {
				seen[debug.Slots[slot].N] = true
			}
		}
	}

	// Unified loop: for each variable in apDecls, try createComplexVar
	// (SSA debug info) first, then fall back to createSimpleVar.
	for _, n := range apDecls {
		if !shouldEmitDwarfVar(n) {
			continue
		}
		if canUseComplex {
			if vid, ok := varIDMap[n]; ok {
				if dvar := createComplexVar(fnsym, fn, vid, closureVars); dvar != nil {
					decls = append(decls, n)
					vars = append(vars, dvar)
					markVarSeen(n, vid)
					continue
				}
			}
		}
		seen[n] = true
		decls = append(decls, n)
		vars = append(vars, createSimpleVar(fnsym, n, closureVars))
	}

	// Add SSA-tracked vars not in apDecls.
	if canUseComplex {
		for i, n := range debug.Vars {
			if seen[n] {
				continue
			}
			if !shouldEmitDwarfVar(n) {
				continue
			}
			if dvar := createComplexVar(fnsym, fn, ssa.VarID(i), closureVars); dvar != nil {
				decls = append(decls, n)
				vars = append(vars, dvar)
				markVarSeen(n, ssa.VarID(i))
			}
		}
	}

	// Recover zero-sized variables eliminated by the stackframe pass.
	if debug != nil {
		for _, n := range debug.OptDcl {
			if seen[n] {
				continue
			}
			if n.Class != ir.PAUTO {
				continue
			}
			types.CalcSize(n.Type())
			if n.Type().Size() == 0 {
				decls = append(decls, n)
				vars = append(vars, createSimpleVar(fnsym, n, closureVars))
				vars[len(vars)-1].StackOffset = 0
				fnsym.Func().RecordAutoType(reflectdata.TypeLinksym(n.Type()))
				seen[n] = true
			}
		}
	}

	// For inlined functions or functions with register output params,
	// collect additional declarations that may not be in apDecls.
	dcl := apDecls
	if fnsym.WasInlined() {
		dcl = preInliningDcls(fnsym)
	} else if debug != nil {
		// The backend's stackframe pass prunes away entries from the
		// fn's Dcl list, including PARAMOUT nodes that correspond to
		// output params passed in registers. Add back in these
		// entries here so that we can process them properly during
		// DWARF-gen. See issue 48573 for more details.
		for _, n := range debug.RegOutputParams {
			if !ssa.IsVarWantedForDebug(n) {
				continue
			}
			if n.Class != ir.PPARAMOUT || !n.IsOutputParamInRegisters() {
				base.Fatalf("invalid ir.Name on debugInfo.RegOutputParams list")
			}
			dcl = append(dcl, n)
		}
	}

	// Process remaining variables not yet handled. For each variable,
	// try createComplexVar first, then fall back to createSimpleVar
	// for non-SSA-able params, or createConservativeVar for the rest.
	for _, n := range dcl {
		if seen[n] {
			continue
		}
		if !shouldEmitDwarfVar(n) {
			continue
		}
		seen[n] = true
		if canUseComplex {
			if vid, ok := varIDMap[n]; ok {
				if dvar := createComplexVar(fnsym, fn, vid, closureVars); dvar != nil {
					decls = append(decls, n)
					vars = append(vars, dvar)
					continue
				}
			}
		}
		if n.Class == ir.PPARAM && !ssa.CanSSA(n.Type()) {
			decls = append(decls, n)
			vars = append(vars, createSimpleVar(fnsym, n, closureVars))
			continue
		}
		decls = append(decls, n)
		vars = append(vars, createConservativeVar(fnsym, fn, n, closureVars))
	}

	// Sort decls and vars.
	sortDeclsAndVars(fn, decls, vars)

	return decls, vars
}

// createConservativeVar creates a DWARF variable with a conservative location
// description. This is used for variables that were optimized away or otherwise
// don't have precise location info. The intent is to communicate that "yes,
// there is a variable named X in this function, but no, I don't have enough
// information to reliably report its contents."
// For heap-escaped variables, a location list is created that describes
// dereferencing the pointer at the stack offset.
func createConservativeVar(fnsym *obj.LSym, fn *ir.Func, n *ir.Name, closureVars map[*ir.Name]int64) *dwarf.Var {
	typename := dwarf.InfoPrefix + types.TypeSymName(n.Type())
	tag := dwarf.DW_TAG_variable
	isReturnValue := (n.Class == ir.PPARAMOUT)
	if n.Class == ir.PPARAM || n.Class == ir.PPARAMOUT {
		tag = dwarf.DW_TAG_formal_parameter
	}
	inlIndex := 0
	if base.Flag.GenDwarfInl > 1 {
		if n.InlFormal() || n.InlLocal() {
			inlIndex = posInlIndex(n.Pos()) + 1
			if n.InlFormal() {
				tag = dwarf.DW_TAG_formal_parameter
			}
		}
	}
	declpos := base.Ctxt.InnermostPos(n.Pos())
	dvar := &dwarf.Var{
		Name:          n.Sym().Name,
		IsReturnValue: isReturnValue,
		Tag:           tag,
		WithLoclist:   true,
		StackOffset:   int32(n.FrameOffset()),
		Type:          base.Ctxt.Lookup(typename),
		DeclFile:      declpos.RelFilename(),
		DeclLine:      declpos.RelLine(),
		DeclCol:       declpos.RelCol(),
		InlIndex:      int32(inlIndex),
		ChildIndex:    -1,
		DictIndex:     n.DictIndex,
		ClosureOffset: closureOffset(n, closureVars),
	}
	if n.Esc() == ir.EscHeap {
		if n.Heapaddr == nil {
			base.Fatalf("invalid heap allocated var without Heapaddr")
		}
		debug := fn.DebugInfo.(*ssa.FuncDebug)
		list := createHeapDerefLocationList(n, debug.EntryID)
		dvar.PutLocationList = func(listSym, startPC dwarf.Sym) {
			debug.PutLocationList(list, base.Ctxt, listSym.(*obj.LSym), startPC.(*obj.LSym))
		}
	}
	// Record go type to ensure that it gets emitted by the linker.
	fnsym.Func().RecordAutoType(reflectdata.TypeLinksym(n.Type()))
	return dvar
}

// sortDeclsAndVars sorts the decl and dwarf var lists according to
// parameter declaration order, so as to insure that when a subprogram
// DIE is emitted, its parameter children appear in declaration order.
// Prior to the advent of the register ABI, sorting by frame offset
// would achieve this; with the register we now need to go back to the
// original function signature.
func sortDeclsAndVars(fn *ir.Func, decls []*ir.Name, vars []*dwarf.Var) {
	paramOrder := make(map[*ir.Name]int)
	idx := 1
	for _, f := range fn.Type().RecvParamsResults() {
		if n, ok := f.Nname.(*ir.Name); ok {
			paramOrder[n] = idx
			idx++
		}
	}
	sort.Stable(varsAndDecls{decls, vars, paramOrder})
}

type varsAndDecls struct {
	decls      []*ir.Name
	vars       []*dwarf.Var
	paramOrder map[*ir.Name]int
}

func (v varsAndDecls) Len() int {
	return len(v.decls)
}

func (v varsAndDecls) Less(i, j int) bool {
	nameLT := func(ni, nj *ir.Name) bool {
		oi, foundi := v.paramOrder[ni]
		oj, foundj := v.paramOrder[nj]
		if foundi {
			if foundj {
				return oi < oj
			} else {
				return true
			}
		}
		return false
	}
	return nameLT(v.decls[i], v.decls[j])
}

func (v varsAndDecls) Swap(i, j int) {
	v.vars[i], v.vars[j] = v.vars[j], v.vars[i]
	v.decls[i], v.decls[j] = v.decls[j], v.decls[i]
}

// Given a function that was inlined at some point during the
// compilation, return a sorted list of nodes corresponding to the
// autos/locals in that function prior to inlining. If this is a
// function that is not local to the package being compiled, then the
// names of the variables may have been "versioned" to avoid conflicts
// with local vars; disregard this versioning when sorting.
func preInliningDcls(fnsym *obj.LSym) []*ir.Name {
	fn := base.Ctxt.DwFixups.GetPrecursorFunc(fnsym).(*ir.Func)
	var rdcl []*ir.Name
	for _, n := range fn.Inl.Dcl {
		if n.Sym().Name[0] == '.' || !shouldEmitDwarfVarSafe(n) {
			continue
		}
		rdcl = append(rdcl, n)
	}
	return rdcl
}

func createSimpleVar(fnsym *obj.LSym, n *ir.Name, closureVars map[*ir.Name]int64) *dwarf.Var {
	var tag int
	var offs int64

	localAutoOffset := func() int64 {
		offs = n.FrameOffset()
		if base.Ctxt.Arch.FixedFrameSize == 0 {
			offs -= int64(types.PtrSize)
		}
		if buildcfg.FramePointerEnabled {
			offs -= int64(types.PtrSize)
		}
		return offs
	}

	switch n.Class {
	case ir.PAUTO:
		offs = localAutoOffset()
		tag = dwarf.DW_TAG_variable
	case ir.PPARAM, ir.PPARAMOUT:
		tag = dwarf.DW_TAG_formal_parameter
		if n.IsOutputParamInRegisters() {
			offs = localAutoOffset()
		} else {
			offs = n.FrameOffset() + base.Ctxt.Arch.FixedFrameSize
		}

	default:
		base.Fatalf("createSimpleVar unexpected class %v for node %v", n.Class, n)
	}

	typename := dwarf.InfoPrefix + types.TypeSymName(n.Type())
	delete(fnsym.Func().Autot, reflectdata.TypeLinksym(n.Type()))
	inlIndex := 0
	if base.Flag.GenDwarfInl > 1 {
		if n.InlFormal() || n.InlLocal() {
			inlIndex = posInlIndex(n.Pos()) + 1
			if n.InlFormal() {
				tag = dwarf.DW_TAG_formal_parameter
			}
		}
	}
	declpos := base.Ctxt.InnermostPos(declPos(n))
	return &dwarf.Var{
		Name:          n.Sym().Name,
		IsReturnValue: n.Class == ir.PPARAMOUT,
		IsInlFormal:   n.InlFormal(),
		Tag:           tag,
		StackOffset:   int32(offs),
		Type:          base.Ctxt.Lookup(typename),
		DeclFile:      declpos.RelFilename(),
		DeclLine:      declpos.RelLine(),
		DeclCol:       declpos.RelCol(),
		InlIndex:      int32(inlIndex),
		ChildIndex:    -1,
		DictIndex:     n.DictIndex,
		ClosureOffset: closureOffset(n, closureVars),
	}
}

// createComplexVar builds a single DWARF variable entry and location list.
func createComplexVar(fnsym *obj.LSym, fn *ir.Func, varID ssa.VarID, closureVars map[*ir.Name]int64) *dwarf.Var {
	debug := fn.DebugInfo.(*ssa.FuncDebug)
	n := debug.Vars[varID]

	var tag int
	switch n.Class {
	case ir.PAUTO:
		tag = dwarf.DW_TAG_variable
	case ir.PPARAM, ir.PPARAMOUT:
		tag = dwarf.DW_TAG_formal_parameter
	default:
		return nil
	}

	gotype := reflectdata.TypeLinksym(n.Type())
	delete(fnsym.Func().Autot, gotype)
	typename := dwarf.InfoPrefix + gotype.Name[len("type:"):]
	inlIndex := 0
	if base.Flag.GenDwarfInl > 1 {
		if n.InlFormal() || n.InlLocal() {
			inlIndex = posInlIndex(n.Pos()) + 1
			if n.InlFormal() {
				tag = dwarf.DW_TAG_formal_parameter
			}
		}
	}
	declpos := base.Ctxt.InnermostPos(n.Pos())
	dvar := &dwarf.Var{
		Name:          n.Sym().Name,
		IsReturnValue: n.Class == ir.PPARAMOUT,
		IsInlFormal:   n.InlFormal(),
		Tag:           tag,
		WithLoclist:   true,
		Type:          base.Ctxt.Lookup(typename),
		// The stack offset is used as a sorting key, so for decomposed
		// variables just give it the first one. It's not used otherwise.
		// This won't work well if the first slot hasn't been assigned a stack
		// location, but it's not obvious how to do better.
		StackOffset:   ssagen.StackOffset(debug.Slots[debug.VarSlots[varID][0]]),
		DeclFile:      declpos.RelFilename(),
		DeclLine:      declpos.RelLine(),
		DeclCol:       declpos.RelCol(),
		InlIndex:      int32(inlIndex),
		ChildIndex:    -1,
		DictIndex:     n.DictIndex,
		ClosureOffset: closureOffset(n, closureVars),
	}
	list := debug.LocationLists[varID]
	if len(list) != 0 {
		dvar.PutLocationList = func(listSym, startPC dwarf.Sym) {
			debug.PutLocationList(list, base.Ctxt, listSym.(*obj.LSym), startPC.(*obj.LSym))
		}
	}
	return dvar
}

// createHeapDerefLocationList creates a location list for a heap-escaped variable
// that describes "dereference pointer at stack offset"
func createHeapDerefLocationList(n *ir.Name, entryID ssa.ID) []ssa.LocListEntry {
	// Get the stack offset where the heap pointer is stored
	heapPtrOffset := n.Heapaddr.FrameOffset()
	if base.Ctxt.Arch.FixedFrameSize == 0 {
		heapPtrOffset -= int64(types.PtrSize)
	}
	if buildcfg.FramePointerEnabled {
		heapPtrOffset -= int64(types.PtrSize)
	}

	// Create a location expression: DW_OP_fbreg <offset> DW_OP_deref
	var expr []byte
	expr = append(expr, dwarf.DW_OP_fbreg)
	expr = dwarf.AppendSleb128(expr, heapPtrOffset)
	expr = append(expr, dwarf.DW_OP_deref)

	return []ssa.LocListEntry{{
		StartBlock: entryID,
		StartValue: ssa.BlockStart.ID,
		EndBlock:   entryID,
		EndValue:   ssa.FuncEnd.ID,
		Expr:       expr,
	}}
}

// RecordFlags records the specified command-line flags to be placed
// in the DWARF info.
func RecordFlags(flags ...string) {
	if base.Ctxt.Pkgpath == "" {
		base.Fatalf("missing pkgpath")
	}

	type BoolFlag interface {
		IsBoolFlag() bool
	}
	type CountFlag interface {
		IsCountFlag() bool
	}
	var cmd bytes.Buffer
	for _, name := range flags {
		f := flag.Lookup(name)
		if f == nil {
			continue
		}
		getter := f.Value.(flag.Getter)
		if getter.String() == f.DefValue {
			// Flag has default value, so omit it.
			continue
		}
		if bf, ok := f.Value.(BoolFlag); ok && bf.IsBoolFlag() {
			val, ok := getter.Get().(bool)
			if ok && val {
				fmt.Fprintf(&cmd, " -%s", f.Name)
				continue
			}
		}
		if cf, ok := f.Value.(CountFlag); ok && cf.IsCountFlag() {
			val, ok := getter.Get().(int)
			if ok && val == 1 {
				fmt.Fprintf(&cmd, " -%s", f.Name)
				continue
			}
		}
		fmt.Fprintf(&cmd, " -%s=%v", f.Name, getter.Get())
	}

	// Adds flag to producer string signaling whether regabi is turned on or
	// off.
	// Once regabi is turned on across the board and the relative GOEXPERIMENT
	// knobs no longer exist this code should be removed.
	if buildcfg.Experiment.RegabiArgs {
		cmd.Write([]byte(" regabi"))
	}

	if cmd.Len() == 0 {
		return
	}
	s := base.Ctxt.Lookup(dwarf.CUInfoPrefix + "producer." + base.Ctxt.Pkgpath)
	s.Type = objabi.SDWARFCUINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	base.Ctxt.Data = append(base.Ctxt.Data, s)
	s.P = cmd.Bytes()[1:]
}

// RecordPackageName records the name of the package being
// compiled, so that the linker can save it in the compile unit's DIE.
func RecordPackageName() {
	s := base.Ctxt.Lookup(dwarf.CUInfoPrefix + "packagename." + base.Ctxt.Pkgpath)
	s.Type = objabi.SDWARFCUINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	base.Ctxt.Data = append(base.Ctxt.Data, s)
	s.P = []byte(types.LocalPkg.Name)
}

// shouldEmitDwarfVar reports whether n should have a DWARF variable entry.
// This consolidates filtering that was previously spread across IR (AutoTemp),
// SSA (IsVarWantedForDebug), and dwarfgen (symbol name checks).
func shouldEmitDwarfVar(n *ir.Name) bool {
	if ir.IsAutoTmp(n) {
		return false
	}
	return shouldEmitDwarfVarSafe(n)
}

// shouldEmitDwarfVarSafe is like shouldEmitDwarfVar but omits the ir.IsAutoTmp
// check, making it safe to call during parallel compilation on shared ir.Name
// nodes (e.g., in preInliningDcls). ir.IsAutoTmp reads the mutable flags bitset,
// which can race with other goroutines writing different flags during compilation.
// Auto temps have names starting with "." so callers must filter those separately.
func shouldEmitDwarfVarSafe(n *ir.Name) bool {
	if !ssa.IsVarWantedForDebug(n) {
		return false
	}
	if n.Sym().Name == "_" {
		return false
	}
	if n.Type().IsUntyped() {
		return false
	}
	return true
}

func closureOffset(n *ir.Name, closureVars map[*ir.Name]int64) int64 {
	return closureVars[n]
}
