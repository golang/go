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
	"internal/race"
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

func emitptrargsmap(fn *Node) {
	if fn.funcname() == "_" || fn.Func.Nname.Sym.Linkname != "" {
		return
	}
	lsym := Ctxt.Lookup(fn.Func.lsym.Name + ".args_stackmap")

	nptr := int(fn.Type.ArgWidth() / int64(Widthptr))
	bv := bvalloc(int32(nptr) * 2)
	nbitmap := 1
	if fn.Type.NumResults() > 0 {
		nbitmap = 2
	}
	off := duint32(lsym, 0, uint32(nbitmap))
	off = duint32(lsym, off, uint32(bv.n))

	if fn.IsMethod() {
		onebitwalktype1(fn.Type.Recvs(), 0, bv)
	}
	if fn.Type.NumParams() > 0 {
		onebitwalktype1(fn.Type.Params(), 0, bv)
	}
	off = dbvec(lsym, off, bv)

	if fn.Type.NumResults() > 0 {
		onebitwalktype1(fn.Type.Results(), 0, bv)
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

	ap := a.Type.HasPointers()
	bp := b.Type.HasPointers()
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
	lastHasPtr := false
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
		if w == 0 && lastHasPtr {
			// Pad between a pointer-containing object and a zero-sized object.
			// This prevents a pointer to the zero-sized object from being interpreted
			// as a pointer to the pointer-containing object (and causing it
			// to be scanned when it shouldn't be). See issue 24993.
			w = 1
		}
		s.stksize += w
		s.stksize = Rnd(s.stksize, int64(n.Type.Align))
		if n.Type.HasPointers() {
			s.stkptrsize = s.stksize
			lastHasPtr = true
		} else {
			lastHasPtr = false
		}
		if thearch.LinkArch.InFamily(sys.MIPS, sys.MIPS64, sys.ARM, sys.ARM64, sys.PPC64, sys.S390X) {
			s.stksize = Rnd(s.stksize, int64(Widthptr))
		}
		n.Xoffset = -s.stksize
	}

	s.stksize = Rnd(s.stksize, int64(Widthreg))
	s.stkptrsize = Rnd(s.stkptrsize, int64(Widthreg))
}

func funccompile(fn *Node) {
	if Curfn != nil {
		Fatalf("funccompile %v inside %v", fn.Func.Nname.Sym, Curfn.Func.Nname.Sym)
	}

	if fn.Type == nil {
		if nerrors == 0 {
			Fatalf("funccompile missing type")
		}
		return
	}

	// assign parameter offsets
	dowidth(fn.Type)

	if fn.Nbody.Len() == 0 {
		// Initialize ABI wrappers if necessary.
		fn.Func.initLSym(false)
		emitptrargsmap(fn)
		return
	}

	dclcontext = PAUTO
	Curfn = fn

	compile(fn)

	Curfn = nil
	dclcontext = PEXTERN
}

func compile(fn *Node) {
	saveerrors()

	order(fn)
	if nerrors != 0 {
		return
	}

	// Set up the function's LSym early to avoid data races with the assemblers.
	// Do this before walk, as walk needs the LSym to set attributes/relocations
	// (e.g. in markTypeUsedInInterface).
	fn.Func.initLSym(true)

	walk(fn)
	if nerrors != 0 {
		return
	}
	if instrumenting {
		instrument(fn)
	}

	// From this point, there should be no uses of Curfn. Enforce that.
	Curfn = nil

	if fn.funcname() == "_" {
		// We don't need to generate code for this function, just report errors in its body.
		// At this point we've generated any errors needed.
		// (Beyond here we generate only non-spec errors, like "stack frame too large".)
		// See issue 29870.
		return
	}

	// Make sure type syms are declared for all types that might
	// be types of stack objects. We need to do this here
	// because symbols must be allocated before the parallel
	// phase of the compiler.
	for _, n := range fn.Func.Dcl {
		switch n.Class() {
		case PPARAM, PPARAMOUT, PAUTO:
			if livenessShouldTrack(n) && n.Name.Addrtaken() {
				dtypesym(n.Type)
				// Also make sure we allocate a linker symbol
				// for the stack object data, for the same reason.
				if fn.Func.lsym.Func().StackObjects == nil {
					fn.Func.lsym.Func().StackObjects = Ctxt.Lookup(fn.Func.lsym.Name + ".stkobj")
				}
			}
		}
	}

	if compilenow(fn) {
		compileSSA(fn, 0)
	} else {
		compilequeue = append(compilequeue, fn)
	}
}

// compilenow reports whether to compile immediately.
// If functions are not compiled immediately,
// they are enqueued in compilequeue,
// which is drained by compileFunctions.
func compilenow(fn *Node) bool {
	// Issue 38068: if this function is a method AND an inline
	// candidate AND was not inlined (yet), put it onto the compile
	// queue instead of compiling it immediately. This is in case we
	// wind up inlining it into a method wrapper that is generated by
	// compiling a function later on in the xtop list.
	if fn.IsMethod() && isInlinableButNotInlined(fn) {
		return false
	}
	return nBackendWorkers == 1 && Debug_compilelater == 0
}

// isInlinableButNotInlined returns true if 'fn' was marked as an
// inline candidate but then never inlined (presumably because we
// found no call sites).
func isInlinableButNotInlined(fn *Node) bool {
	if fn.Func.Nname.Func.Inl == nil {
		return false
	}
	if fn.Sym == nil {
		return true
	}
	return !fn.Sym.Linksym().WasInlined()
}

const maxStackSize = 1 << 30

// compileSSA builds an SSA backend function,
// uses it to generate a plist,
// and flushes that plist to machine code.
// worker indicates which of the backend workers is doing the processing.
func compileSSA(fn *Node, worker int) {
	f := buildssa(fn, worker)
	// Note: check arg size to fix issue 25507.
	if f.Frontend().(*ssafn).stksize >= maxStackSize || fn.Type.ArgWidth() >= maxStackSize {
		largeStackFramesMu.Lock()
		largeStackFrames = append(largeStackFrames, largeStack{locals: f.Frontend().(*ssafn).stksize, args: fn.Type.ArgWidth(), pos: fn.Pos})
		largeStackFramesMu.Unlock()
		return
	}
	pp := newProgs(fn, worker)
	defer pp.Free()
	genssa(f, pp)
	// Check frame size again.
	// The check above included only the space needed for local variables.
	// After genssa, the space needed includes local variables and the callee arg region.
	// We must do this check prior to calling pp.Flush.
	// If there are any oversized stack frames,
	// the assembler may emit inscrutable complaints about invalid instructions.
	if pp.Text.To.Offset >= maxStackSize {
		largeStackFramesMu.Lock()
		locals := f.Frontend().(*ssafn).stksize
		largeStackFrames = append(largeStackFrames, largeStack{locals: locals, args: fn.Type.ArgWidth(), callee: pp.Text.To.Offset - locals, pos: fn.Pos})
		largeStackFramesMu.Unlock()
		return
	}

	pp.Flush() // assemble, fill in boilerplate, etc.
	// fieldtrack must be called after pp.Flush. See issue 20014.
	fieldtrack(pp.Text.From.Sym, fn.Func.FieldTrack)
}

func init() {
	if race.Enabled {
		rand.Seed(time.Now().UnixNano())
	}
}

// compileFunctions compiles all functions in compilequeue.
// It fans out nBackendWorkers to do the work
// and waits for them to complete.
func compileFunctions() {
	if len(compilequeue) != 0 {
		sizeCalculationDisabled = true // not safe to calculate sizes concurrently
		if race.Enabled {
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
			sort.Slice(compilequeue, func(i, j int) bool {
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

func debuginfo(fnsym *obj.LSym, infosym *obj.LSym, curfn interface{}) ([]dwarf.Scope, dwarf.InlCalls) {
	fn := curfn.(*Node)
	if fn.Func.Nname != nil {
		if expect := fn.Func.Nname.Sym.Linksym(); fnsym != expect {
			Fatalf("unexpected fnsym: %v != %v", fnsym, expect)
		}
	}

	var apdecls []*Node
	// Populate decls for fn.
	for _, n := range fn.Func.Dcl {
		if n.Op != ONAME { // might be OTYPE or OLITERAL
			continue
		}
		switch n.Class() {
		case PAUTO:
			if !n.Name.Used() {
				// Text == nil -> generating abstract function
				if fnsym.Func().Text != nil {
					Fatalf("debuginfo unused node (AllocFrame should truncate fn.Func.Dcl)")
				}
				continue
			}
		case PPARAM, PPARAMOUT:
		default:
			continue
		}
		apdecls = append(apdecls, n)
		fnsym.Func().RecordAutoType(ngotype(n).Linksym())
	}

	decls, dwarfVars := createDwarfVars(fnsym, fn.Func, apdecls)

	// For each type referenced by the functions auto vars but not
	// already referenced by a dwarf var, attach a dummy relocation to
	// the function symbol to insure that the type included in DWARF
	// processing during linking.
	typesyms := []*obj.LSym{}
	for t, _ := range fnsym.Func().Autot {
		typesyms = append(typesyms, t)
	}
	sort.Sort(obj.BySymName(typesyms))
	for _, sym := range typesyms {
		r := obj.Addrel(infosym)
		r.Sym = sym
		r.Type = objabi.R_USETYPE
	}
	fnsym.Func().Autot = nil

	var varScopes []ScopeID
	for _, decl := range decls {
		pos := declPos(decl)
		varScopes = append(varScopes, findScope(fn.Func.Marks, pos))
	}

	scopes := assembleScopes(fnsym, fn, dwarfVars, varScopes)
	var inlcalls dwarf.InlCalls
	if genDwarfInline > 0 {
		inlcalls = assembleInlines(fnsym, dwarfVars)
	}
	return scopes, inlcalls
}

func declPos(decl *Node) src.XPos {
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
		return decl.Name.Defn.Pos
	}
	return decl.Pos
}

// createSimpleVars creates a DWARF entry for every variable declared in the
// function, claiming that they are permanently on the stack.
func createSimpleVars(fnsym *obj.LSym, apDecls []*Node) ([]*Node, []*dwarf.Var, map[*Node]bool) {
	var vars []*dwarf.Var
	var decls []*Node
	selected := make(map[*Node]bool)
	for _, n := range apDecls {
		if n.IsAutoTmp() {
			continue
		}

		decls = append(decls, n)
		vars = append(vars, createSimpleVar(fnsym, n))
		selected[n] = true
	}
	return decls, vars, selected
}

func createSimpleVar(fnsym *obj.LSym, n *Node) *dwarf.Var {
	var abbrev int
	offs := n.Xoffset

	switch n.Class() {
	case PAUTO:
		abbrev = dwarf.DW_ABRV_AUTO
		if Ctxt.FixedFrameSize() == 0 {
			offs -= int64(Widthptr)
		}
		if objabi.Framepointer_enabled || objabi.GOARCH == "arm64" {
			// There is a word space for FP on ARM64 even if the frame pointer is disabled
			offs -= int64(Widthptr)
		}

	case PPARAM, PPARAMOUT:
		abbrev = dwarf.DW_ABRV_PARAM
		offs += Ctxt.FixedFrameSize()
	default:
		Fatalf("createSimpleVar unexpected class %v for node %v", n.Class(), n)
	}

	typename := dwarf.InfoPrefix + typesymname(n.Type)
	delete(fnsym.Func().Autot, ngotype(n).Linksym())
	inlIndex := 0
	if genDwarfInline > 1 {
		if n.Name.InlFormal() || n.Name.InlLocal() {
			inlIndex = posInlIndex(n.Pos) + 1
			if n.Name.InlFormal() {
				abbrev = dwarf.DW_ABRV_PARAM
			}
		}
	}
	declpos := Ctxt.InnermostPos(declPos(n))
	return &dwarf.Var{
		Name:          n.Sym.Name,
		IsReturnValue: n.Class() == PPARAMOUT,
		IsInlFormal:   n.Name.InlFormal(),
		Abbrev:        abbrev,
		StackOffset:   int32(offs),
		Type:          Ctxt.Lookup(typename),
		DeclFile:      declpos.RelFilename(),
		DeclLine:      declpos.RelLine(),
		DeclCol:       declpos.Col(),
		InlIndex:      int32(inlIndex),
		ChildIndex:    -1,
	}
}

// createComplexVars creates recomposed DWARF vars with location lists,
// suitable for describing optimized code.
func createComplexVars(fnsym *obj.LSym, fn *Func) ([]*Node, []*dwarf.Var, map[*Node]bool) {
	debugInfo := fn.DebugInfo

	// Produce a DWARF variable entry for each user variable.
	var decls []*Node
	var vars []*dwarf.Var
	ssaVars := make(map[*Node]bool)

	for varID, dvar := range debugInfo.Vars {
		n := dvar.(*Node)
		ssaVars[n] = true
		for _, slot := range debugInfo.VarSlots[varID] {
			ssaVars[debugInfo.Slots[slot].N.(*Node)] = true
		}

		if dvar := createComplexVar(fnsym, fn, ssa.VarID(varID)); dvar != nil {
			decls = append(decls, n)
			vars = append(vars, dvar)
		}
	}

	return decls, vars, ssaVars
}

// createDwarfVars process fn, returning a list of DWARF variables and the
// Nodes they represent.
func createDwarfVars(fnsym *obj.LSym, fn *Func, apDecls []*Node) ([]*Node, []*dwarf.Var) {
	// Collect a raw list of DWARF vars.
	var vars []*dwarf.Var
	var decls []*Node
	var selected map[*Node]bool
	if Ctxt.Flag_locationlists && Ctxt.Flag_optimize && fn.DebugInfo != nil {
		decls, vars, selected = createComplexVars(fnsym, fn)
	} else {
		decls, vars, selected = createSimpleVars(fnsym, apDecls)
	}

	dcl := apDecls
	if fnsym.WasInlined() {
		dcl = preInliningDcls(fnsym)
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
		if _, found := selected[n]; found {
			continue
		}
		c := n.Sym.Name[0]
		if c == '.' || n.Type.IsUntyped() {
			continue
		}
		if n.Class() == PPARAM && !canSSAType(n.Type) {
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
		typename := dwarf.InfoPrefix + typesymname(n.Type)
		decls = append(decls, n)
		abbrev := dwarf.DW_ABRV_AUTO_LOCLIST
		isReturnValue := (n.Class() == PPARAMOUT)
		if n.Class() == PPARAM || n.Class() == PPARAMOUT {
			abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
		} else if n.Class() == PAUTOHEAP {
			// If dcl in question has been promoted to heap, do a bit
			// of extra work to recover original class (auto or param);
			// see issue 30908. This insures that we get the proper
			// signature in the abstract function DIE, but leaves a
			// misleading location for the param (we want pointer-to-heap
			// and not stack).
			// TODO(thanm): generate a better location expression
			stackcopy := n.Name.Param.Stackcopy
			if stackcopy != nil && (stackcopy.Class() == PPARAM || stackcopy.Class() == PPARAMOUT) {
				abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
				isReturnValue = (stackcopy.Class() == PPARAMOUT)
			}
		}
		inlIndex := 0
		if genDwarfInline > 1 {
			if n.Name.InlFormal() || n.Name.InlLocal() {
				inlIndex = posInlIndex(n.Pos) + 1
				if n.Name.InlFormal() {
					abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
				}
			}
		}
		declpos := Ctxt.InnermostPos(n.Pos)
		vars = append(vars, &dwarf.Var{
			Name:          n.Sym.Name,
			IsReturnValue: isReturnValue,
			Abbrev:        abbrev,
			StackOffset:   int32(n.Xoffset),
			Type:          Ctxt.Lookup(typename),
			DeclFile:      declpos.RelFilename(),
			DeclLine:      declpos.RelLine(),
			DeclCol:       declpos.Col(),
			InlIndex:      int32(inlIndex),
			ChildIndex:    -1,
		})
		// Record go type of to insure that it gets emitted by the linker.
		fnsym.Func().RecordAutoType(ngotype(n).Linksym())
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
	var rdcl []*Node
	for _, n := range fn.Func.Inl.Dcl {
		c := n.Sym.Name[0]
		// Avoid reporting "_" parameters, since if there are more than
		// one, it can result in a collision later on, as in #23179.
		if unversion(n.Sym.Name) == "_" || c == '.' || n.Type.IsUntyped() {
			continue
		}
		rdcl = append(rdcl, n)
	}
	return rdcl
}

// stackOffset returns the stack location of a LocalSlot relative to the
// stack pointer, suitable for use in a DWARF location entry. This has nothing
// to do with its offset in the user variable.
func stackOffset(slot ssa.LocalSlot) int32 {
	n := slot.N.(*Node)
	var base int64
	switch n.Class() {
	case PAUTO:
		if Ctxt.FixedFrameSize() == 0 {
			base -= int64(Widthptr)
		}
		if objabi.Framepointer_enabled || objabi.GOARCH == "arm64" {
			// There is a word space for FP on ARM64 even if the frame pointer is disabled
			base -= int64(Widthptr)
		}
	case PPARAM, PPARAMOUT:
		base += Ctxt.FixedFrameSize()
	}
	return int32(base + n.Xoffset + slot.Off)
}

// createComplexVar builds a single DWARF variable entry and location list.
func createComplexVar(fnsym *obj.LSym, fn *Func, varID ssa.VarID) *dwarf.Var {
	debug := fn.DebugInfo
	n := debug.Vars[varID].(*Node)

	var abbrev int
	switch n.Class() {
	case PAUTO:
		abbrev = dwarf.DW_ABRV_AUTO_LOCLIST
	case PPARAM, PPARAMOUT:
		abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
	default:
		return nil
	}

	gotype := ngotype(n).Linksym()
	delete(fnsym.Func().Autot, gotype)
	typename := dwarf.InfoPrefix + gotype.Name[len("type."):]
	inlIndex := 0
	if genDwarfInline > 1 {
		if n.Name.InlFormal() || n.Name.InlLocal() {
			inlIndex = posInlIndex(n.Pos) + 1
			if n.Name.InlFormal() {
				abbrev = dwarf.DW_ABRV_PARAM_LOCLIST
			}
		}
	}
	declpos := Ctxt.InnermostPos(n.Pos)
	dvar := &dwarf.Var{
		Name:          n.Sym.Name,
		IsReturnValue: n.Class() == PPARAMOUT,
		IsInlFormal:   n.Name.InlFormal(),
		Abbrev:        abbrev,
		Type:          Ctxt.Lookup(typename),
		// The stack offset is used as a sorting key, so for decomposed
		// variables just give it the first one. It's not used otherwise.
		// This won't work well if the first slot hasn't been assigned a stack
		// location, but it's not obvious how to do better.
		StackOffset: stackOffset(debug.Slots[debug.VarSlots[varID][0]]),
		DeclFile:    declpos.RelFilename(),
		DeclLine:    declpos.RelLine(),
		DeclCol:     declpos.Col(),
		InlIndex:    int32(inlIndex),
		ChildIndex:  -1,
	}
	list := debug.LocationLists[varID]
	if len(list) != 0 {
		dvar.PutLocationList = func(listSym, startPC dwarf.Sym) {
			debug.PutLocationList(list, Ctxt, listSym.(*obj.LSym), startPC.(*obj.LSym))
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
