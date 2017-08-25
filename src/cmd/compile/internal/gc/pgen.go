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
	if Curfn.Type.Results().NumFields() > 0 {
		nbitmap = 2
	}
	off := duint32(lsym, 0, uint32(nbitmap))
	off = duint32(lsym, off, uint32(bv.n))
	var xoffset int64
	if Curfn.IsMethod() {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Recvs(), &xoffset, bv)
	}

	if Curfn.Type.Params().NumFields() > 0 {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Params(), &xoffset, bv)
	}

	off = dbvec(lsym, off, bv)
	if Curfn.Type.Results().NumFields() > 0 {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Results(), &xoffset, bv)
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
			switch a := v.Aux.(type) {
			case *ssa.ArgSymbol:
				n := a.Node.(*Node)
				// Don't modify nodfp; it is a global.
				if n != nodfp {
					n.Name.SetUsed(true)
				}
			case *ssa.AutoSymbol:
				a.Node.(*Node).Name.SetUsed(true)
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

const maxStackSize = 1 << 31

// compileSSA builds an SSA backend function,
// uses it to generate a plist,
// and flushes that plist to machine code.
// worker indicates which of the backend workers is doing the processing.
func compileSSA(fn *Node, worker int) {
	ssafn := buildssa(fn, worker)
	pp := newProgs(fn, worker)
	genssa(ssafn, pp)
	if pp.Text.To.Offset < maxStackSize {
		pp.Flush()
	} else {
		largeStackFramesMu.Lock()
		largeStackFrames = append(largeStackFrames, fn.Pos)
		largeStackFramesMu.Unlock()
	}
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
		sizeCalculationDisabled = false
	}
}

func debuginfo(fnsym *obj.LSym, curfn interface{}) []dwarf.Scope {
	fn := curfn.(*Node)
	debugInfo := fn.Func.DebugInfo
	fn.Func.DebugInfo = nil
	if expect := fn.Func.Nname.Sym.Linksym(); fnsym != expect {
		Fatalf("unexpected fnsym: %v != %v", fnsym, expect)
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
				Fatalf("debuginfo unused node (AllocFrame should truncate fn.Func.Dcl)")
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

	var dwarfVars []*dwarf.Var
	var decls []*Node
	if Ctxt.Flag_locationlists && Ctxt.Flag_optimize {
		decls, dwarfVars = createComplexVars(fn, debugInfo)
	} else {
		decls, dwarfVars = createSimpleVars(automDecls)
	}

	var varScopes []ScopeID
	for _, decl := range decls {
		var scope ScopeID
		if !decl.Name.Captured() && !decl.Name.Byval() {
			// n.Pos of captured variables is their first
			// use in the closure but they should always
			// be assigned to scope 0 instead.
			// TODO(mdempsky): Verify this.
			scope = findScope(fn.Func.Marks, decl.Pos)
		}
		varScopes = append(varScopes, scope)
	}
	return assembleScopes(fnsym, fn, dwarfVars, varScopes)
}

// createSimpleVars creates a DWARF entry for every variable declared in the
// function, claiming that they are permanently on the stack.
func createSimpleVars(automDecls []*Node) ([]*Node, []*dwarf.Var) {
	var vars []*dwarf.Var
	var decls []*Node
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

		typename := dwarf.InfoPrefix + typesymname(n.Type)
		decls = append(decls, n)
		vars = append(vars, &dwarf.Var{
			Name:        n.Sym.Name,
			Abbrev:      abbrev,
			StackOffset: int32(offs),
			Type:        Ctxt.Lookup(typename),
			DeclLine:    n.Pos.Line(),
		})
	}
	return decls, vars
}

type varPart struct {
	varOffset int64
	slot      ssa.SlotID
	locs      ssa.VarLocList
}

func createComplexVars(fn *Node, debugInfo *ssa.FuncDebug) ([]*Node, []*dwarf.Var) {
	for _, locList := range debugInfo.Variables {
		for _, loc := range locList.Locations {
			if loc.StartProg != nil {
				loc.StartPC = loc.StartProg.Pc
			}
			if loc.EndProg != nil {
				loc.EndPC = loc.EndProg.Pc
			}
			if Debug_locationlist == 0 {
				loc.EndProg = nil
				loc.StartProg = nil
			}
		}
	}

	// Group SSA variables by the user variable they were decomposed from.
	varParts := map[*Node][]varPart{}
	for slotID, locList := range debugInfo.Variables {
		if len(locList.Locations) == 0 {
			continue
		}
		slot := debugInfo.Slots[slotID]
		for slot.SplitOf != nil {
			slot = slot.SplitOf
		}
		n := slot.N.(*Node)
		varParts[n] = append(varParts[n], varPart{varOffset(slot), ssa.SlotID(slotID), locList})
	}

	// Produce a DWARF variable entry for each user variable.
	// Don't iterate over the map -- that's nondeterministic, and
	// createComplexVar has side effects. Instead, go by slot.
	var decls []*Node
	var vars []*dwarf.Var
	for _, slot := range debugInfo.Slots {
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
	return decls, vars
}

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
	// The stack offset is used as a sorting key, so for decomposed
	// variables just give it the lowest one. It's not used otherwise.
	stackOffset := debugInfo.Slots[parts[0].slot].N.(*Node).Xoffset + offs
	dvar := &dwarf.Var{
		Name:        n.Sym.Name,
		Abbrev:      abbrev,
		Type:        Ctxt.Lookup(typename),
		StackOffset: int32(stackOffset),
		DeclLine:    n.Pos.Line(),
	}

	if Debug_locationlist != 0 {
		Ctxt.Logf("Building location list for %+v. Parts:\n", n)
		for _, part := range parts {
			Ctxt.Logf("\t%v => %v\n", debugInfo.Slots[part.slot], part.locs)
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

	curLoc := make([]int, len(slots))

	// findBoundaryAfter finds the next beginning or end of a piece after currentPC.
	findBoundaryAfter := func(currentPC int64) int64 {
		min := int64(math.MaxInt64)
		for slot, part := range parts {
			// For each part, find the first PC greater than current. Doesn't
			// matter if it's a start or an end, since we're looking for any boundary.
			// If it's the new winner, save it.
		onePart:
			for i := curLoc[slot]; i < len(part.locs.Locations); i++ {
				for _, pc := range [2]int64{part.locs.Locations[i].StartPC, part.locs.Locations[i].EndPC} {
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
			choice := -1
			for i := curLoc[part.slot]; i < len(part.locs.Locations); i++ {
				if part.locs.Locations[i].StartPC > start {
					break //overshot
				}
				choice = i // best yet
			}
			if choice != -1 {
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
			locIdx := curLoc[part.slot]
			if locIdx >= len(part.locs.Locations) ||
				start >= part.locs.Locations[locIdx].EndPC ||
				end <= part.locs.Locations[locIdx].StartPC {
				if Debug_locationlist != 0 {
					Ctxt.Logf("\t%v: missing", slots[part.slot])
				}
				dpiece.Missing = true
				dloc.Pieces = append(dloc.Pieces, dpiece)
				continue
			}
			present++
			loc := part.locs.Locations[locIdx]
			if Debug_locationlist != 0 {
				Ctxt.Logf("\t%v: %v", slots[part.slot], loc)
			}
			if loc.OnStack {
				dpiece.OnStack = true
				dpiece.StackOffset = int32(offs + slots[part.slot].Off + slots[part.slot].N.(*Node).Xoffset)
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
