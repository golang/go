// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarfgen

import (
	"bytes"
	"flag"
	"fmt"
	"internal/buildcfg"
	"sort"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

func Info(fnsym *obj.LSym, infosym *obj.LSym, curfn obj.Func) (scopes []dwarf.Scope, inlcalls dwarf.InlCalls) {
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
			apdecls = append(apdecls, n)
			if n.Type().Kind() == types.TSSA {
				// Can happen for TypeInt128 types. This only happens for
				// spill locations, so not a huge deal.
				continue
			}
			fnsym.Func().RecordAutoType(reflectdata.TypeLinksym(n.Type()))
		}
	}

	decls, dwarfVars := createDwarfVars(fnsym, isODCLFUNC, fn, apdecls)

	// For each type referenced by the functions auto vars but not
	// already referenced by a dwarf var, attach an R_USETYPE relocation to
	// the function symbol to insure that the type included in DWARF
	// processing during linking.
	typesyms := []*obj.LSym{}
	for t := range fnsym.Func().Autot {
		typesyms = append(typesyms, t)
	}
	sort.Sort(obj.BySymName(typesyms))
	for _, sym := range typesyms {
		r := obj.Addrel(infosym)
		r.Sym = sym
		r.Type = objabi.R_USETYPE
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
func createDwarfVars(fnsym *obj.LSym, complexOK bool, fn *ir.Func, apDecls []*ir.Name) ([]*ir.Name, []*dwarf.Var) {
	// Collect a raw list of DWARF vars.
	var vars []*dwarf.Var
	var decls []*ir.Name
	var selected ir.NameSet

	if base.Ctxt.Flag_locationlists && base.Ctxt.Flag_optimize && fn.DebugInfo != nil && complexOK {
		decls, vars, selected = createComplexVars(fnsym, fn)
	} else if fn.ABI == obj.ABIInternal && base.Flag.N != 0 && complexOK {
		decls, vars, selected = createABIVars(fnsym, fn, apDecls)
	} else {
		decls, vars, selected = createSimpleVars(fnsym, apDecls)
	}
	if fn.DebugInfo != nil {
		// Recover zero sized variables eliminated by the stackframe pass
		for _, n := range fn.DebugInfo.(*ssa.FuncDebug).OptDcl {
			if n.Class != ir.PAUTO {
				continue
			}
			types.CalcSize(n.Type())
			if n.Type().Size() == 0 {
				decls = append(decls, n)
				vars = append(vars, createSimpleVar(fnsym, n))
				vars[len(vars)-1].StackOffset = 0
				fnsym.Func().RecordAutoType(reflectdata.TypeLinksym(n.Type()))
			}
		}
	}

	dcl := apDecls
	if fnsym.WasInlined() {
		dcl = preInliningDcls(fnsym)
	} else {
		// The backend's stackframe pass prunes away entries from the
		// fn's Dcl list, including PARAMOUT nodes that correspond to
		// output params passed in registers. Add back in these
		// entries here so that we can process them properly during
		// DWARF-gen. See issue 48573 for more details.
		debugInfo := fn.DebugInfo.(*ssa.FuncDebug)
		for _, n := range debugInfo.RegOutputParams {
			if n.Class != ir.PPARAMOUT || !n.IsOutputParamInRegisters() {
				panic("invalid ir.Name on debugInfo.RegOutputParams list")
			}
			dcl = append(dcl, n)
		}
	}

	// If optimization is enabled, the list above will typically be
	// missing some of the original pre-optimization variables in the
	// function (they may have been promoted to registers, folded into
	// constants, dead-coded away, etc).  Input arguments not eligible
	// for SSA optimization are also missing.  Here we add back in entries
	// for selected missing vars. Note that the recipe below creates a
	// conservative location. The idea here is that we want to
	// communicate to the user that "yes, there is a variable named X
	// in this function, but no, I don't have enough information to
	// reliably report its contents."
	// For non-SSA-able arguments, however, the correct information
	// is known -- they have a single home on the stack.
	for _, n := range dcl {
		if selected.Has(n) {
			continue
		}
		c := n.Sym().Name[0]
		if c == '.' || n.Type().IsUntyped() {
			continue
		}
		if n.Class == ir.PPARAM && !ssa.CanSSA(n.Type()) {
			// SSA-able args get location lists, and may move in and
			// out of registers, so those are handled elsewhere.
			// Autos and named output params seem to get handled
			// with VARDEF, which creates location lists.
			// Args not of SSA-able type are treated here; they
			// are homed on the stack in a single place for the
			// entire call.
			vars = append(vars, createSimpleVar(fnsym, n))
			decls = append(decls, n)
			continue
		}
		typename := dwarf.InfoPrefix + types.TypeSymName(n.Type())
		decls = append(decls, n)
		tag := dwarf.DW_TAG_variable
		isReturnValue := (n.Class == ir.PPARAMOUT)
		if n.Class == ir.PPARAM || n.Class == ir.PPARAMOUT {
			tag = dwarf.DW_TAG_formal_parameter
		}
		if n.Esc() == ir.EscHeap {
			// The variable in question has been promoted to the heap.
			// Its address is in n.Heapaddr.
			// TODO(thanm): generate a better location expression
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
		vars = append(vars, &dwarf.Var{
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
		})
		// Record go type of to insure that it gets emitted by the linker.
		fnsym.Func().RecordAutoType(reflectdata.TypeLinksym(n.Type()))
	}

	// Sort decls and vars.
	sortDeclsAndVars(fn, decls, vars)

	return decls, vars
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
		c := n.Sym().Name[0]
		// Avoid reporting "_" parameters, since if there are more than
		// one, it can result in a collision later on, as in #23179.
		if n.Sym().Name == "_" || c == '.' || n.Type().IsUntyped() {
			continue
		}
		rdcl = append(rdcl, n)
	}
	return rdcl
}

// createSimpleVars creates a DWARF entry for every variable declared in the
// function, claiming that they are permanently on the stack.
func createSimpleVars(fnsym *obj.LSym, apDecls []*ir.Name) ([]*ir.Name, []*dwarf.Var, ir.NameSet) {
	var vars []*dwarf.Var
	var decls []*ir.Name
	var selected ir.NameSet
	for _, n := range apDecls {
		if ir.IsAutoTmp(n) {
			continue
		}

		decls = append(decls, n)
		vars = append(vars, createSimpleVar(fnsym, n))
		selected.Add(n)
	}
	return decls, vars, selected
}

func createSimpleVar(fnsym *obj.LSym, n *ir.Name) *dwarf.Var {
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
	}
}

// createABIVars creates DWARF variables for functions in which the
// register ABI is enabled but optimization is turned off. It uses a
// hybrid approach in which register-resident input params are
// captured with location lists, and all other vars use the "simple"
// strategy.
func createABIVars(fnsym *obj.LSym, fn *ir.Func, apDecls []*ir.Name) ([]*ir.Name, []*dwarf.Var, ir.NameSet) {

	// Invoke createComplexVars to generate dwarf vars for input parameters
	// that are register-allocated according to the ABI rules.
	decls, vars, selected := createComplexVars(fnsym, fn)

	// Now fill in the remainder of the variables: input parameters
	// that are not register-resident, output parameters, and local
	// variables.
	for _, n := range apDecls {
		if ir.IsAutoTmp(n) {
			continue
		}
		if _, ok := selected[n]; ok {
			// already handled
			continue
		}

		decls = append(decls, n)
		vars = append(vars, createSimpleVar(fnsym, n))
		selected.Add(n)
	}

	return decls, vars, selected
}

// createComplexVars creates recomposed DWARF vars with location lists,
// suitable for describing optimized code.
func createComplexVars(fnsym *obj.LSym, fn *ir.Func) ([]*ir.Name, []*dwarf.Var, ir.NameSet) {
	debugInfo := fn.DebugInfo.(*ssa.FuncDebug)

	// Produce a DWARF variable entry for each user variable.
	var decls []*ir.Name
	var vars []*dwarf.Var
	var ssaVars ir.NameSet

	for varID, dvar := range debugInfo.Vars {
		n := dvar
		ssaVars.Add(n)
		for _, slot := range debugInfo.VarSlots[varID] {
			ssaVars.Add(debugInfo.Slots[slot].N)
		}

		if dvar := createComplexVar(fnsym, fn, ssa.VarID(varID)); dvar != nil {
			decls = append(decls, n)
			vars = append(vars, dvar)
		}
	}

	return decls, vars, ssaVars
}

// createComplexVar builds a single DWARF variable entry and location list.
func createComplexVar(fnsym *obj.LSym, fn *ir.Func, varID ssa.VarID) *dwarf.Var {
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
		StackOffset: ssagen.StackOffset(debug.Slots[debug.VarSlots[varID][0]]),
		DeclFile:    declpos.RelFilename(),
		DeclLine:    declpos.RelLine(),
		DeclCol:     declpos.RelCol(),
		InlIndex:    int32(inlIndex),
		ChildIndex:  -1,
		DictIndex:   n.DictIndex,
	}
	list := debug.LocationLists[varID]
	if len(list) != 0 {
		dvar.PutLocationList = func(listSym, startPC dwarf.Sym) {
			debug.PutLocationList(list, base.Ctxt, listSym.(*obj.LSym), startPC.(*obj.LSym))
		}
	}
	return dvar
}

// RecordFlags records the specified command-line flags to be placed
// in the DWARF info.
func RecordFlags(flags ...string) {
	if base.Ctxt.Pkgpath == "" {
		panic("missing pkgpath")
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
