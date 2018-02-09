// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"
)

// "Portable" code generation.

var (
	nBackendWorkers int     // number of concurrent backend workers, set by a compiler flag
	compilequeue    []*Node // functions waiting to be compiled
)

func emitptrargsmap() {
	if Curfn.funcname() == "_" {
		return
	}
	sym := lookup(fmt.Sprintf("%s.args_stackmap", Curfn.funcname()))
	lsym := sym.Linksym()

	nptr := int(Curfn.Type.ArgWidth() / int64(Widthptr))
	bv := bvalloc(int32(nptr) * 2)
	nbitmap := 1
	if Curfn.Type.NumResults() > 0 {
		nbitmap = 2
	}
	off := duint32(lsym, 0, uint32(nbitmap))
	off = duint32(lsym, off, uint32(bv.n))

	if Curfn.IsMethod() {
		onebitwalktype1(Curfn.Type.Recvs(), 0, bv)
	}
	if Curfn.Type.NumParams() > 0 {
		onebitwalktype1(Curfn.Type.Params(), 0, bv)
	}
	off = dbvec(lsym, off, bv)

	if Curfn.Type.NumResults() > 0 {
		onebitwalktype1(Curfn.Type.Results(), 0, bv)
		off = dbvec(lsym, off, bv)
	}

	ggloblsym(lsym, int32(off), obj.RODATA|obj.LOCAL)
}

// cmpstackvarlt reports whether the stack variable a sorts before b.
//
// Sort the list of stack variables. Autos after anything else,
// within autos, unused after used, within used, things with
// pointers first, zeroed things first, and then decreasing size.
// Because autos are laid out in decreasing addresses
// on the stack, pointers first, zeroed things first and decreasing size
// really means, in memory, things with pointers needing zeroing at
// the top of the stack and increasing in size.
// Non-autos sort on offset.
func cmpstackvarlt(a, b *Node) bool {
	if (a.Class() == PAUTO) != (b.Class() == PAUTO) {
		return b.Class() == PAUTO
	}

	if a.Class() != PAUTO {
		return a.Xoffset < b.Xoffset
	}

	if a.Name.Used() != b.Name.Used() {
		return a.Name.Used()
	}

	ap := types.Haspointers(a.Type)
	bp := types.Haspointers(b.Type)
	if ap != bp {
		return ap
	}

	ap = a.Name.Needzero()
	bp = b.Name.Needzero()
	if ap != bp {
		return ap
	}

	if a.Type.Width != b.Type.Width {
		return a.Type.Width > b.Type.Width
	}

	return a.Sym.Name < b.Sym.Name
}

// byStackvar implements sort.Interface for []*Node using cmpstackvarlt.
type byStackVar []*Node

func (s byStackVar) Len() int           { return len(s) }
func (s byStackVar) Less(i, j int) bool { return cmpstackvarlt(s[i], s[j]) }
func (s byStackVar) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func (s *ssafn) AllocFrame(f *ssa.Func) {
	s.stksize = 0
	s.stkptrsize = 0
	fn := s.curfn.Func

	// Mark the PAUTO's unused.
	for _, ln := range fn.Dcl {
		if ln.Class() == PAUTO {
			ln.Name.SetUsed(false)
		}
	}

	for _, l := range f.RegAlloc {
		if ls, ok := l.(ssa.LocalSlot); ok {
			ls.N.(*Node).Name.SetUsed(true)
		}
	}

	scratchUsed := false
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if n, ok := v.Aux.(*Node); ok {
				switch n.Class() {
				case PPARAM, PPARAMOUT:
					// Don't modify nodfp; it is a global.
					if n != nodfp {
						n.Name.SetUsed(true)
					}
				case PAUTO:
					n.Name.SetUsed(true)
				}
			}
			if !scratchUsed {
				scratchUsed = v.Op.UsesScratch()
			}

		}
	}

	if f.Config.NeedsFpScratch && scratchUsed {
		s.scratchFpMem = tempAt(src.NoXPos, s.curfn, types.Types[TUINT64])
	}

	sort.Sort(byStackVar(fn.Dcl))

	// Reassign stack offsets of the locals that are used.
	for i, n := range fn.Dcl {
		if n.Op != ONAME || n.Class() != PAUTO {
			continue
		}
		if !n.Name.Used() {
			fn.Dcl = fn.Dcl[:i]
			break
		}

		dowidth(n.Type)
		w := n.Type.Width
		if w >= thearch.MAXWIDTH || w < 0 {
			Fatalf("bad width")
		}
		s.stksize += w
		s.stksize = Rnd(s.stksize, int64(n.Type.Align))
		if types.Haspointers(n.Type) {
			s.stkptrsize = s.stksize
		}
		if thearch.LinkArch.InFamily(sys.MIPS, sys.MIPS64, sys.ARM, sys.ARM64, sys.PPC64, sys.S390X) {
			s.stksize = Rnd(s.stksize, int64(Widthptr))
		}
		n.Xoffset = -s.stksize
	}

	s.stksize = Rnd(s.stksize, int64(Widthreg))
	s.stkptrsize = Rnd(s.stkptrsize, int64(Widthreg))
}

func compile(fn *Node) {
	Curfn = fn
	dowidth(fn.Type)

	if fn.Nbody.Len() == 0 {
		emitptrargsmap()
		return
	}

	saveerrors()

	order(fn)
	if nerrors != 0 {
		return
	}

	walk(fn)
	if nerrors != 0 {
		return
	}
	if instrumenting {
		instrument(fn)
	}

	// From this point, there should be no uses of Curfn. Enforce that.
	Curfn = nil

	// Set up the function's LSym early to avoid data races with the assemblers.
	fn.Func.initLSym()

	if compilenow() {
		compileSSA(fn, 0)
	} else {
		compilequeue = append(compilequeue, fn)
	}
}

// compilenow reports whether to compile immediately.
// If functions are not compiled immediately,
// they are enqueued in compilequeue,
// which is drained by compileFunctions.
func compilenow() bool {
	return nBackendWorkers == 1 && Debug_compilelater == 0
}

const maxStackSize = 1 << 30

// compileSSA builds an SSA backend function,
// uses it to generate a plist,
// and flushes that plist to machine code.
// worker indicates which of the backend workers is doing the processing.
func compileSSA(fn *Node, worker int) {
	f := buildssa(fn, worker)
	if f.Frontend().(*ssafn).stksize >= maxStackSize {
		largeStackFramesMu.Lock()
		largeStackFrames = append(largeStackFrames, fn.Pos)
		largeStackFramesMu.Unlock()
		return
	}
	pp := newProgs(fn, worker)
	genssa(f, pp)
	pp.Flush()
	// fieldtrack must be called after pp.Flush. See issue 20014.
	fieldtrack(pp.Text.From.Sym, fn.Func.FieldTrack)
	pp.Free()
}

func init() {
	if raceEnabled {
		rand.Seed(time.Now().UnixNano())
	}
}

// compileFunctions compiles all functions in compilequeue.
// It fans out nBackendWorkers to do the work
// and waits for them to complete.
func compileFunctions() {
	if len(compilequeue) != 0 {
		sizeCalculationDisabled = true // not safe to calculate sizes concurrently
		if raceEnabled {
			// Randomize compilation order to try to shake out races.
			tmp := make([]*Node, len(compilequeue))
			perm := rand.Perm(len(compilequeue))
			for i, v := range perm {
				tmp[v] = compilequeue[i]
			}
			copy(compilequeue, tmp)
		} else {
			// Compile the longest functions first,
			// since they're most likely to be the slowest.
			// This helps avoid stragglers.
			obj.SortSlice(compilequeue, func(i, j int) bool {
				return compilequeue[i].Nbody.Len() > compilequeue[j].Nbody.Len()
			})
		}
		var wg sync.WaitGroup
		Ctxt.InParallel = true
		c := make(chan *Node, nBackendWorkers)
		for i := 0; i < nBackendWorkers; i++ {
			wg.Add(1)
			go func(worker int) {
				for fn := range c {
					compileSSA(fn, worker)
				}
				wg.Done()
			}(i)
		}
		for _, fn := range compilequeue {
			c <- fn
		}
		close(c)
		compilequeue = nil
		wg.Wait()
		Ctxt.InParallel = false
		sizeCalculationDisabled = false
	}
}

func debuginfo(fnsym *obj.LSym, curfn interface{}) ([]dwarf.Scope, dwarf.InlCalls) {
	fn := curfn.(*Node)
	debugInfo := fn.Func.DebugInfo
	fn.Func.DebugInfo = nil
	if fn.Func.Nname != nil {
		if expect := fn.Func.Nname.Sym.Linksym(); fnsym != expect {
			Fatalf("unexpected fnsym: %v != %v", fnsym, expect)
		}
	}

	var automDecls []*Node
	// Populate Automs for fn.
	for _, n := range fn.Func.Dcl {
		if n.Op != ONAME { // might be OTYPE or OLITERAL
			continue
		}
		var name obj.AddrName
		switch n.Class() {
		case PAUTO:
			if !n.Name.Used() {
				// Text == nil -> generating abstract function
				if fnsym.Func.Text != nil {
					Fatalf("debuginfo unused node (AllocFrame should truncate fn.Func.Dcl)")
				}
				continue
			}
			name = obj.NAME_AUTO
		case PPARAM, PPARAMOUT:
			name = obj.NAME_PARAM
		default:
			continue
		}
		automDecls = append(automDecls, n)
		gotype := ngotype(n).Linksym()
		fnsym.Func.Autom = append(fnsym.Func.Autom, &obj.Auto{
			Asym:    Ctxt.Lookup(n.Sym.Name),
			Aoffset: int32(n.Xoffset),
			Name:    name,
			Gotype:  gotype,
		})
	}

	decls, dwarfVars := createDwarfVars(fnsym, debugInfo, automDecls)

	var varScopes []ScopeID
	for _, decl := range decls {
		pos := decl.Pos
		if decl.Name.Defn != nil && (decl.Name.Captured() || decl.Name.Byval()) {
			// It's not clear which position is correct for captured variables here:
			// * decl.Pos is the wrong position for captured variables, in the inner
			//   function, but it is the right position in the outer function.
			// * decl.Name.Defn is nil for captured variables that were arguments
			//   on the outer function, however the decl.Pos for those seems to be
			//   correct.
			// * decl.Name.Defn is the "wrong" thing for variables declared in the
			//   header of a type switch, it's their position in the header, rather
			//   than the position of the case statement. In principle this is the
			//   right thing, but here we prefer the latter because it makes each
			//   instance of the header variable local to the lexical block of its
			//   case statement.
			// This code is probably wrong for type switch variables that are also
			// captured.
			pos = decl.Name.Defn.Pos
		}
		varScopes = append(varScopes, findScope(fn.Func.Marks, pos))
	}

	scopes := assembleScopes(fnsym, fn, dwarfVars, varScopes)
	var inlcalls dwarf.InlCalls
	if genDwarfInline > 0 {
		inlcalls = assembleInlines(fnsym, fn, dwarfVars)
	}
	return scopes, inlcalls
}

// createSimpleVars creates a DWARF entry for every variable declared in the
// function, claiming that they are permanently on the stack.
func createSimpleVars(automDecls []*Node) ([]*Node, []*dwarf.Var, map[*Node]bool) {
	var vars []*dwarf.Var
	var decls []*Node
	selected := make(map[*Node]bool)
	for _, n := range automDecls {
		if n.IsAutoTmp() {
			continue
		}
		var abbrev int
		offs := n.Xoffset

		switch n.Class() {
		case PAUTO:
			abbrev = dwarf.DW_ABRV_AUTO
			if Ctxt.FixedFrameSize() == 0 {
				offs -= int64(Widthptr)
			}
			if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
				offs -= int64(Widthptr)
			}

		case PPARAM, PPARAMOUT:
			abbrev = dwarf.DW_ABRV_PARAM
			offs += Ctxt.FixedFrameSize()
		default:
			Fatalf("createSimpleVars unexpected type %v for node %v", n.Class(), n)
		}

		selected[n] = true
		typename := dwarf.InfoPrefix + typesymname(n.Type)
		decls = append(decls, n)
		inlIndex := 0
		if genDwarfInline > 1 {
			if n.InlFormal() || n.InlLocal() {
				inlIndex = posInlIndex(n.Pos) + 1
				if n.InlFormal() {
					abbrev = dwarf.DW_ABRV_PARAM
				}
			}
		}
		declpos := Ctxt.InnermostPos(n.Pos)
		vars = append(vars, &dwarf.Var{
			Name:          n.Sym.Name,
			IsReturnValue: n.Class() == PPARAMOUT,
			IsInlFormal:   n.InlFormal(),
			Abbrev:        abbrev,
			StackOffset:   int32(offs),
			Type:          Ctxt.Lookup(typename),
			DeclFile:      declpos.Base().SymFilename(),
			DeclLine:      declpos.Line(),
			DeclCol:       declpos.Col(),
			InlIndex:      int32(inlIndex),
			ChildIndex:    -1,
		})
	}
	return decls, vars, selected
}

type varPart struct {
	varOffset int64
	slot      ssa.SlotID
}

func createComplexVars(fnsym *obj.LSym, debugInfo *ssa.FuncDebug, automDecls []*Node) ([]*Node, []*dwarf.Var, map[*Node]bool) {
	for _, blockDebug := range debugInfo.Blocks {
		for _, locList := range blockDebug.Variables {
			for _, loc := range locList.Locations {
				if loc.StartProg != nil {
					loc.StartPC = loc.StartProg.Pc
				}
				if loc.EndProg != nil {
					loc.EndPC = loc.EndProg.Pc
				} else {
					loc.EndPC = fnsym.Size
				}
				if Debug_locationlist == 0 {
					loc.EndProg = nil
					loc.StartProg = nil
				}
			}
		}
	}

	// Group SSA variables by the user variable they were decomposed from.
	varParts := map[*Node][]varPart{}
	ssaVars := make(map[*Node]bool)
	for slotID, slot := range debugInfo.VarSlots {
		for slot.SplitOf != nil {
			slot = slot.SplitOf
		}
		n := slot.N.(*Node)
		ssaVars[n] = true
		varParts[n] = append(varParts[n], varPart{varOffset(slot), ssa.SlotID(slotID)})
	}

	// Produce a DWARF variable entry for each user variable.
	// Don't iterate over the map -- that's nondeterministic, and
	// createComplexVar has side effects. Instead, go by slot.
	var decls []*Node
	var vars []*dwarf.Var
	for _, slot := range debugInfo.VarSlots {
		for slot.SplitOf != nil {
			slot = slot.SplitOf
		}
		n := slot.N.(*Node)
		parts := varParts[n]
		if parts == nil {
			continue
		}
		// Don't work on this variable again, no matter how many slots it has.
		delete(varParts, n)

		// Get the order the parts need to be in to represent the memory
		// of the decomposed user variable.
		sort.Sort(partsByVarOffset(parts))

		if dvar := createComplexVar(debugInfo, n, parts); dvar != nil {
			decls = append(decls, n)
			vars = append(vars, dvar)
		}
	}

	return decls, vars, ssaVars
}

func createDwarfVars(fnsym *obj.LSym, debugInfo *ssa.FuncDebug, automDecls []*Node) ([]*Node, []*dwarf.Var) {
	// Collect a raw list of DWARF vars.
	var vars []*dwarf.Var
	var decls []*Node
	var selected map[*Node]bool
	if Ctxt.Flag_locationlists && Ctxt.Flag_optimize && debugInfo != nil {
		decls, vars, selected = createComplexVars(fnsym, debugInfo, automDecls)
	} else {
		decls, vars, selected = createSimpleVars(automDecls)
	}

	var dcl []*Node
	if fnsym.WasInlined() {
		dcl = preInliningDcls(fnsym)
	} else {
		dcl = automDecls
	}

	// If optimization is enabled, the list above will typically be
	// missing some of the original pre-optimization variables in the
	// function (they may have been promoted to registers, folded into
	// constants, dead-coded away, etc). Here we add back in entries
	// for selected missing vars. Note that the recipe below creates a
	// conservative location. The idea here is that we want to
	// communicate to the user that "yes, there is a variable named X
	// in this function, but no, I don't have enough information to
	// reliably report its contents."
	for _, n := range dcl {
		if _, found := selected[n]; found {
			continue
		}
		c := n.Sym.Name[0]
		if c == '.' || n.Type.IsUntyped() {
			continue
		}
		typename := dwarf.InfoPrefix + typesymname(n.Type)
		decls = append(decls, n)
		abbrev := dwarf.DW_ABRV_AUTO_LOCLIST
		if n.Class() == PPARAM || n.Class() == PPARAMOUT {
			abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
		}
		inlIndex := 0
		if genDwarfInline > 1 {
			if n.InlFormal() || n.InlLocal() {
				inlIndex = posInlIndex(n.Pos) + 1
				if n.InlFormal() {
					abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
				}
			}
		}
		declpos := Ctxt.InnermostPos(n.Pos)
		vars = append(vars, &dwarf.Var{
			Name:          n.Sym.Name,
			IsReturnValue: n.Class() == PPARAMOUT,
			Abbrev:        abbrev,
			StackOffset:   int32(n.Xoffset),
			Type:          Ctxt.Lookup(typename),
			DeclFile:      declpos.Base().SymFilename(),
			DeclLine:      declpos.Line(),
			DeclCol:       declpos.Col(),
			InlIndex:      int32(inlIndex),
			ChildIndex:    -1,
		})
		// Append a "deleted auto" entry to the autom list so as to
		// insure that the type in question is picked up by the linker.
		// See issue 22941.
		gotype := ngotype(n).Linksym()
		fnsym.Func.Autom = append(fnsym.Func.Autom, &obj.Auto{
			Asym:    Ctxt.Lookup(n.Sym.Name),
			Aoffset: int32(-1),
			Name:    obj.NAME_DELETED_AUTO,
			Gotype:  gotype,
		})

	}

	return decls, vars
}

// Given a function that was inlined at some point during the
// compilation, return a sorted list of nodes corresponding to the
// autos/locals in that function prior to inlining. If this is a
// function that is not local to the package being compiled, then the
// names of the variables may have been "versioned" to avoid conflicts
// with local vars; disregard this versioning when sorting.
func preInliningDcls(fnsym *obj.LSym) []*Node {
	fn := Ctxt.DwFixups.GetPrecursorFunc(fnsym).(*Node)
	var dcl, rdcl []*Node
	if fn.Name.Defn != nil {
		dcl = fn.Func.Inldcl.Slice() // local function
	} else {
		dcl = fn.Func.Dcl // imported function
	}
	for _, n := range dcl {
		c := n.Sym.Name[0]
		// Avoid reporting "_" parameters, since if there are more than
		// one, it can result in a collision later on, as in #23179.
		if unversion(n.Sym.Name) == "_" || c == '.' || n.Type.IsUntyped() {
			continue
		}
		rdcl = append(rdcl, n)
	}
	sort.Sort(byNodeName(rdcl))
	return rdcl
}

func cmpNodeName(a, b *Node) bool {
	aart := 0
	if strings.HasPrefix(a.Sym.Name, "~") {
		aart = 1
	}
	bart := 0
	if strings.HasPrefix(b.Sym.Name, "~") {
		bart = 1
	}
	if aart != bart {
		return aart < bart
	}

	aname := unversion(a.Sym.Name)
	bname := unversion(b.Sym.Name)
	return aname < bname
}

// byNodeName implements sort.Interface for []*Node using cmpNodeName.
type byNodeName []*Node

func (s byNodeName) Len() int           { return len(s) }
func (s byNodeName) Less(i, j int) bool { return cmpNodeName(s[i], s[j]) }
func (s byNodeName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// varOffset returns the offset of slot within the user variable it was
// decomposed from. This has nothing to do with its stack offset.
func varOffset(slot *ssa.LocalSlot) int64 {
	offset := slot.Off
	for ; slot.SplitOf != nil; slot = slot.SplitOf {
		offset += slot.SplitOffset
	}
	return offset
}

type partsByVarOffset []varPart

func (a partsByVarOffset) Len() int           { return len(a) }
func (a partsByVarOffset) Less(i, j int) bool { return a[i].varOffset < a[j].varOffset }
func (a partsByVarOffset) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// stackOffset returns the stack location of a LocalSlot relative to the
// stack pointer, suitable for use in a DWARF location entry. This has nothing
// to do with its offset in the user variable.
func stackOffset(slot *ssa.LocalSlot) int32 {
	n := slot.N.(*Node)
	var base int64
	switch n.Class() {
	case PAUTO:
		if Ctxt.FixedFrameSize() == 0 {
			base -= int64(Widthptr)
		}
		if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
			base -= int64(Widthptr)
		}
	case PPARAM, PPARAMOUT:
		base += Ctxt.FixedFrameSize()
	}
	return int32(base + n.Xoffset + slot.Off)
}

// createComplexVar builds a DWARF variable entry and location list representing n.
func createComplexVar(debugInfo *ssa.FuncDebug, n *Node, parts []varPart) *dwarf.Var {
	slots := debugInfo.Slots
	var offs int64 // base stack offset for this kind of variable
	var abbrev int
	switch n.Class() {
	case PAUTO:
		abbrev = dwarf.DW_ABRV_AUTO_LOCLIST
		if Ctxt.FixedFrameSize() == 0 {
			offs -= int64(Widthptr)
		}
		if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
			offs -= int64(Widthptr)
		}

	case PPARAM, PPARAMOUT:
		abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
		offs += Ctxt.FixedFrameSize()
	default:
		return nil
	}

	gotype := ngotype(n).Linksym()
	typename := dwarf.InfoPrefix + gotype.Name[len("type."):]
	inlIndex := 0
	if genDwarfInline > 1 {
		if n.InlFormal() || n.InlLocal() {
			inlIndex = posInlIndex(n.Pos) + 1
			if n.InlFormal() {
				abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
			}
		}
	}
	declpos := Ctxt.InnermostPos(n.Pos)
	dvar := &dwarf.Var{
		Name:          n.Sym.Name,
		IsReturnValue: n.Class() == PPARAMOUT,
		IsInlFormal:   n.InlFormal(),
		Abbrev:        abbrev,
		Type:          Ctxt.Lookup(typename),
		// The stack offset is used as a sorting key, so for decomposed
		// variables just give it the lowest one. It's not used otherwise.
		// This won't work well if the first slot hasn't been assigned a stack
		// location, but it's not obvious how to do better.
		StackOffset: int32(stackOffset(slots[parts[0].slot])),
		DeclFile:    declpos.Base().SymFilename(),
		DeclLine:    declpos.Line(),
		DeclCol:     declpos.Col(),
		InlIndex:    int32(inlIndex),
		ChildIndex:  -1,
	}

	if Debug_locationlist != 0 {
		Ctxt.Logf("Building location list for %+v. Parts:\n", n)
		for _, part := range parts {
			Ctxt.Logf("\t%v => %v\n", debugInfo.Slots[part.slot], debugInfo.SlotLocsString(part.slot))
		}
	}

	// Given a variable that's been decomposed into multiple parts,
	// its location list may need a new entry after the beginning or
	// end of every location entry for each of its parts. For example:
	//
	// [variable]    [pc range]
	// string.ptr    |----|-----|    |----|
	// string.len    |------------|  |--|
	// ... needs a location list like:
	// string        |----|-----|-|  |--|-|
	//
	// Note that location entries may or may not line up with each other,
	// and some of the result will only have one or the other part.
	//
	// To build the resulting list:
	// - keep a "current" pointer for each part
	// - find the next transition point
	// - advance the current pointer for each part up to that transition point
	// - build the piece for the range between that transition point and the next
	// - repeat

	type locID struct {
		block int
		loc   int
	}
	findLoc := func(part varPart, id locID) *ssa.VarLoc {
		if id.block >= len(debugInfo.Blocks) {
			return nil
		}
		return debugInfo.Blocks[id.block].Variables[part.slot].Locations[id.loc]
	}
	nextLoc := func(part varPart, id locID) (locID, *ssa.VarLoc) {
		// Check if there's another loc in this block
		id.loc++
		if b := debugInfo.Blocks[id.block]; b != nil && id.loc < len(b.Variables[part.slot].Locations) {
			return id, findLoc(part, id)
		}
		// Find the next block that has a loc for this part.
		id.loc = 0
		id.block++
		for ; id.block < len(debugInfo.Blocks); id.block++ {
			if b := debugInfo.Blocks[id.block]; b != nil && len(b.Variables[part.slot].Locations) != 0 {
				return id, findLoc(part, id)
			}
		}
		return id, nil
	}
	curLoc := make([]locID, len(slots))
	// Position each pointer at the first entry for its slot.
	for _, part := range parts {
		if b := debugInfo.Blocks[0]; b != nil && len(b.Variables[part.slot].Locations) != 0 {
			// Block 0 has an entry; no need to advance.
			continue
		}
		curLoc[part.slot], _ = nextLoc(part, curLoc[part.slot])
	}

	// findBoundaryAfter finds the next beginning or end of a piece after currentPC.
	findBoundaryAfter := func(currentPC int64) int64 {
		min := int64(math.MaxInt64)
		for _, part := range parts {
			// For each part, find the first PC greater than current. Doesn't
			// matter if it's a start or an end, since we're looking for any boundary.
			// If it's the new winner, save it.
		onePart:
			for i, loc := curLoc[part.slot], findLoc(part, curLoc[part.slot]); loc != nil; i, loc = nextLoc(part, i) {
				for _, pc := range [2]int64{loc.StartPC, loc.EndPC} {
					if pc > currentPC {
						if pc < min {
							min = pc
						}
						break onePart
					}
				}
			}
		}
		return min
	}
	var start int64
	end := findBoundaryAfter(0)
	for {
		// Advance to the next chunk.
		start = end
		end = findBoundaryAfter(start)
		if end == math.MaxInt64 {
			break
		}

		dloc := dwarf.Location{StartPC: start, EndPC: end}
		if Debug_locationlist != 0 {
			Ctxt.Logf("Processing range %x -> %x\n", start, end)
		}

		// Advance curLoc to the last location that starts before/at start.
		// After this loop, if there's a location that covers [start, end), it will be current.
		// Otherwise the current piece will be too early.
		for _, part := range parts {
			choice := locID{-1, -1}
			for i, loc := curLoc[part.slot], findLoc(part, curLoc[part.slot]); loc != nil; i, loc = nextLoc(part, i) {
				if loc.StartPC > start {
					break //overshot
				}
				choice = i // best yet
			}
			if choice.block != -1 {
				curLoc[part.slot] = choice
			}
			if Debug_locationlist != 0 {
				Ctxt.Logf("\t %v => %v", slots[part.slot], curLoc[part.slot])
			}
		}
		if Debug_locationlist != 0 {
			Ctxt.Logf("\n")
		}
		// Assemble the location list entry for this chunk.
		present := 0
		for _, part := range parts {
			dpiece := dwarf.Piece{
				Length: slots[part.slot].Type.Size(),
			}
			loc := findLoc(part, curLoc[part.slot])
			if loc == nil || start >= loc.EndPC || end <= loc.StartPC {
				if Debug_locationlist != 0 {
					Ctxt.Logf("\t%v: missing", slots[part.slot])
				}
				dpiece.Missing = true
				dloc.Pieces = append(dloc.Pieces, dpiece)
				continue
			}
			present++
			if Debug_locationlist != 0 {
				Ctxt.Logf("\t%v: %v", slots[part.slot], debugInfo.Blocks[curLoc[part.slot].block].LocString(loc))
			}
			if loc.OnStack {
				dpiece.OnStack = true
				dpiece.StackOffset = stackOffset(slots[loc.StackLocation])
			} else {
				for reg := 0; reg < len(debugInfo.Registers); reg++ {
					if loc.Registers&(1<<uint8(reg)) != 0 {
						dpiece.RegNum = Ctxt.Arch.DWARFRegisters[debugInfo.Registers[reg].ObjNum()]
					}
				}
			}
			dloc.Pieces = append(dloc.Pieces, dpiece)
		}
		if present == 0 {
			if Debug_locationlist != 0 {
				Ctxt.Logf(" -> totally missing\n")
			}
			continue
		}
		// Extend the previous entry if possible.
		if len(dvar.LocationList) > 0 {
			prev := &dvar.LocationList[len(dvar.LocationList)-1]
			if prev.EndPC == dloc.StartPC && len(prev.Pieces) == len(dloc.Pieces) {
				equal := true
				for i := range prev.Pieces {
					if prev.Pieces[i] != dloc.Pieces[i] {
						equal = false
					}
				}
				if equal {
					prev.EndPC = end
					if Debug_locationlist != 0 {
						Ctxt.Logf("-> merged with previous, now %#v\n", prev)
					}
					continue
				}
			}
		}
		dvar.LocationList = append(dvar.LocationList, dloc)
		if Debug_locationlist != 0 {
			Ctxt.Logf("-> added: %#v\n", dloc)
		}
	}
	return dvar
}

// fieldtrack adds R_USEFIELD relocations to fnsym to record any
// struct fields that it used.
func fieldtrack(fnsym *obj.LSym, tracked map[*types.Sym]struct{}) {
	if fnsym == nil {
		return
	}
	if objabi.Fieldtrack_enabled == 0 || len(tracked) == 0 {
		return
	}

	trackSyms := make([]*types.Sym, 0, len(tracked))
	for sym := range tracked {
		trackSyms = append(trackSyms, sym)
	}
	sort.Sort(symByName(trackSyms))
	for _, sym := range trackSyms {
		r := obj.Addrel(fnsym)
		r.Sym = sym.Linksym()
		r.Type = objabi.R_USEFIELD
	}
}

type symByName []*types.Sym

func (a symByName) Len() int           { return len(a) }
func (a symByName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a symByName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
