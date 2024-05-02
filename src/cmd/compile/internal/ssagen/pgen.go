// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"fmt"
	"internal/buildcfg"
	"os"
	"sort"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/liveness"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

// cmpstackvarlt reports whether the stack variable a sorts before b.
func cmpstackvarlt(a, b *ir.Name, mls *liveness.MergeLocalsState) bool {
	// Sort non-autos before autos.
	if needAlloc(a) != needAlloc(b) {
		return needAlloc(b)
	}

	// If both are non-auto (e.g., parameters, results), then sort by
	// frame offset (defined by ABI).
	if !needAlloc(a) {
		return a.FrameOffset() < b.FrameOffset()
	}

	// From here on, a and b are both autos (i.e., local variables).

	// Sort followers after leaders, if mls != nil
	if mls != nil {
		aFollow := mls.Subsumed(a)
		bFollow := mls.Subsumed(b)
		if aFollow != bFollow {
			return bFollow
		}
	}

	// Sort used before unused (so AllocFrame can truncate unused
	// variables).
	if a.Used() != b.Used() {
		return a.Used()
	}

	// Sort pointer-typed before non-pointer types.
	// Keeps the stack's GC bitmap compact.
	ap := a.Type().HasPointers()
	bp := b.Type().HasPointers()
	if ap != bp {
		return ap
	}

	// Group variables that need zeroing, so we can efficiently zero
	// them altogether.
	ap = a.Needzero()
	bp = b.Needzero()
	if ap != bp {
		return ap
	}

	// Sort variables in descending alignment order, so we can optimally
	// pack variables into the frame.
	if a.Type().Alignment() != b.Type().Alignment() {
		return a.Type().Alignment() > b.Type().Alignment()
	}

	// Sort normal variables before open-coded-defer slots, so that the
	// latter are grouped together and near the top of the frame (to
	// minimize varint encoding of their varp offset).
	if a.OpenDeferSlot() != b.OpenDeferSlot() {
		return a.OpenDeferSlot()
	}

	// If a and b are both open-coded defer slots, then order them by
	// index in descending order, so they'll be laid out in the frame in
	// ascending order.
	//
	// Their index was saved in FrameOffset in state.openDeferSave.
	if a.OpenDeferSlot() {
		return a.FrameOffset() > b.FrameOffset()
	}

	// Tie breaker for stable results.
	return a.Sym().Name < b.Sym().Name
}

// needAlloc reports whether n is within the current frame, for which we need to
// allocate space. In particular, it excludes arguments and results, which are in
// the callers frame.
func needAlloc(n *ir.Name) bool {
	if n.Op() != ir.ONAME {
		base.FatalfAt(n.Pos(), "%v has unexpected Op %v", n, n.Op())
	}

	switch n.Class {
	case ir.PAUTO:
		return true
	case ir.PPARAM:
		return false
	case ir.PPARAMOUT:
		return n.IsOutputParamInRegisters()

	default:
		base.FatalfAt(n.Pos(), "%v has unexpected Class %v", n, n.Class)
		return false
	}
}

func (s *ssafn) AllocFrame(f *ssa.Func) {
	s.stksize = 0
	s.stkptrsize = 0
	s.stkalign = int64(types.RegSize)
	fn := s.curfn

	// Mark the PAUTO's unused.
	for _, ln := range fn.Dcl {
		if ln.OpenDeferSlot() {
			// Open-coded defer slots have indices that were assigned
			// upfront during SSA construction, but the defer statement can
			// later get removed during deadcode elimination (#61895). To
			// keep their relative offsets correct, treat them all as used.
			continue
		}

		if needAlloc(ln) {
			ln.SetUsed(false)
		}
	}

	for _, l := range f.RegAlloc {
		if ls, ok := l.(ssa.LocalSlot); ok {
			ls.N.SetUsed(true)
		}
	}

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if n, ok := v.Aux.(*ir.Name); ok {
				switch n.Class {
				case ir.PPARAMOUT:
					if n.IsOutputParamInRegisters() && v.Op == ssa.OpVarDef {
						// ignore VarDef, look for "real" uses.
						// TODO: maybe do this for PAUTO as well?
						continue
					}
					fallthrough
				case ir.PPARAM, ir.PAUTO:
					n.SetUsed(true)
				}
			}
		}
	}

	var mls *liveness.MergeLocalsState
	var leaders map[*ir.Name]int64
	if base.Debug.MergeLocals != 0 {
		mls = liveness.MergeLocals(fn, f)
		if base.Debug.MergeLocalsTrace > 0 && mls != nil {
			savedNP, savedP := mls.EstSavings()
			fmt.Fprintf(os.Stderr, "%s: %d bytes of stack space saved via stack slot merging (%d nonpointer %d pointer)\n", ir.FuncName(fn), savedNP+savedP, savedNP, savedP)
			if base.Debug.MergeLocalsTrace > 1 {
				fmt.Fprintf(os.Stderr, "=-= merge locals state for %v:\n%v",
					fn, mls)
			}
		}
		leaders = make(map[*ir.Name]int64)
	}

	// Use sort.SliceStable instead of sort.Slice so stack layout (and thus
	// compiler output) is less sensitive to frontend changes that
	// introduce or remove unused variables.
	sort.SliceStable(fn.Dcl, func(i, j int) bool {
		return cmpstackvarlt(fn.Dcl[i], fn.Dcl[j], mls)
	})

	if mls != nil {
		// Rewrite fn.Dcl to reposition followers (subsumed vars) to
		// be immediately following the leader var in their partition.
		followers := []*ir.Name{}
		newdcl := make([]*ir.Name, 0, len(fn.Dcl))
		for i := 0; i < len(fn.Dcl); i++ {
			n := fn.Dcl[i]
			if mls.Subsumed(n) {
				continue
			}
			newdcl = append(newdcl, n)
			if mls.IsLeader(n) {
				followers = mls.Followers(n, followers)
				// position followers immediately after leader
				newdcl = append(newdcl, followers...)
			}
		}
		fn.Dcl = newdcl
	}

	if base.Debug.MergeLocalsTrace > 1 && mls != nil {
		fmt.Fprintf(os.Stderr, "=-= sorted DCL for %v:\n", fn)
		for i, v := range fn.Dcl {
			if !ssa.IsMergeCandidate(v) {
				continue
			}
			fmt.Fprintf(os.Stderr, " %d: %q isleader=%v subsumed=%v used=%v sz=%d align=%d t=%s\n", i, v.Sym().Name, mls.IsLeader(v), mls.Subsumed(v), v.Used(), v.Type().Size(), v.Type().Alignment(), v.Type().String())
		}
	}

	// Reassign stack offsets of the locals that are used.
	lastHasPtr := false
	for i, n := range fn.Dcl {
		if n.Op() != ir.ONAME || n.Class != ir.PAUTO && !(n.Class == ir.PPARAMOUT && n.IsOutputParamInRegisters()) {
			// i.e., stack assign if AUTO, or if PARAMOUT in registers (which has no predefined spill locations)
			continue
		}
		if mls != nil && mls.Subsumed(n) {
			continue
		}
		if !n.Used() {
			fn.DebugInfo.(*ssa.FuncDebug).OptDcl = fn.Dcl[i:]
			fn.Dcl = fn.Dcl[:i]
			break
		}
		types.CalcSize(n.Type())
		w := n.Type().Size()
		if w >= types.MaxWidth || w < 0 {
			base.Fatalf("bad width")
		}
		if w == 0 && lastHasPtr {
			// Pad between a pointer-containing object and a zero-sized object.
			// This prevents a pointer to the zero-sized object from being interpreted
			// as a pointer to the pointer-containing object (and causing it
			// to be scanned when it shouldn't be). See issue 24993.
			w = 1
		}
		s.stksize += w
		s.stksize = types.RoundUp(s.stksize, n.Type().Alignment())
		if n.Type().Alignment() > int64(types.RegSize) {
			s.stkalign = n.Type().Alignment()
		}
		if n.Type().HasPointers() {
			s.stkptrsize = s.stksize
			lastHasPtr = true
		} else {
			lastHasPtr = false
		}
		n.SetFrameOffset(-s.stksize)
		if mls != nil && mls.IsLeader(n) {
			leaders[n] = -s.stksize
		}
	}

	if mls != nil {
		// Update offsets of followers (subsumed vars) to be the
		// same as the leader var in their partition.
		for i := 0; i < len(fn.Dcl); i++ {
			n := fn.Dcl[i]
			if !mls.Subsumed(n) {
				continue
			}
			leader := mls.Leader(n)
			off, ok := leaders[leader]
			if !ok {
				panic("internal error missing leader")
			}
			// Set the stack offset this subsumed (followed) var
			// to be the same as the leader.
			n.SetFrameOffset(off)
		}

		if base.Debug.MergeLocalsTrace > 1 {
			fmt.Fprintf(os.Stderr, "=-= stack layout for %v:\n", fn)
			for i, v := range fn.Dcl {
				if v.Op() != ir.ONAME || (v.Class != ir.PAUTO && !(v.Class == ir.PPARAMOUT && v.IsOutputParamInRegisters())) {
					continue
				}
				fmt.Fprintf(os.Stderr, " %d: %q frameoff %d isleader=%v subsumed=%v sz=%d align=%d t=%s\n", i, v.Sym().Name, v.FrameOffset(), mls.IsLeader(v), mls.Subsumed(v), v.Type().Size(), v.Type().Alignment(), v.Type().String())
			}
		}
	}

	s.stksize = types.RoundUp(s.stksize, s.stkalign)
	s.stkptrsize = types.RoundUp(s.stkptrsize, s.stkalign)
}

const maxStackSize = 1 << 30

// Compile builds an SSA backend function,
// uses it to generate a plist,
// and flushes that plist to machine code.
// worker indicates which of the backend workers is doing the processing.
func Compile(fn *ir.Func, worker int) {
	f := buildssa(fn, worker)
	// Note: check arg size to fix issue 25507.
	if f.Frontend().(*ssafn).stksize >= maxStackSize || f.OwnAux.ArgWidth() >= maxStackSize {
		largeStackFramesMu.Lock()
		largeStackFrames = append(largeStackFrames, largeStack{locals: f.Frontend().(*ssafn).stksize, args: f.OwnAux.ArgWidth(), pos: fn.Pos()})
		largeStackFramesMu.Unlock()
		return
	}
	pp := objw.NewProgs(fn, worker)
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
		largeStackFrames = append(largeStackFrames, largeStack{locals: locals, args: f.OwnAux.ArgWidth(), callee: pp.Text.To.Offset - locals, pos: fn.Pos()})
		largeStackFramesMu.Unlock()
		return
	}

	pp.Flush() // assemble, fill in boilerplate, etc.

	// If we're compiling the package init function, search for any
	// relocations that target global map init outline functions and
	// turn them into weak relocs.
	if fn.IsPackageInit() && base.Debug.WrapGlobalMapCtl != 1 {
		weakenGlobalMapInitRelocs(fn)
	}

	// fieldtrack must be called after pp.Flush. See issue 20014.
	fieldtrack(pp.Text.From.Sym, fn.FieldTrack)
}

// globalMapInitLsyms records the LSym of each map.init.NNN outlined
// map initializer function created by the compiler.
var globalMapInitLsyms map[*obj.LSym]struct{}

// RegisterMapInitLsym records "s" in the set of outlined map initializer
// functions.
func RegisterMapInitLsym(s *obj.LSym) {
	if globalMapInitLsyms == nil {
		globalMapInitLsyms = make(map[*obj.LSym]struct{})
	}
	globalMapInitLsyms[s] = struct{}{}
}

// weakenGlobalMapInitRelocs walks through all of the relocations on a
// given a package init function "fn" and looks for relocs that target
// outlined global map initializer functions; if it finds any such
// relocs, it flags them as R_WEAK.
func weakenGlobalMapInitRelocs(fn *ir.Func) {
	if globalMapInitLsyms == nil {
		return
	}
	for i := range fn.LSym.R {
		tgt := fn.LSym.R[i].Sym
		if tgt == nil {
			continue
		}
		if _, ok := globalMapInitLsyms[tgt]; !ok {
			continue
		}
		if base.Debug.WrapGlobalMapDbg > 1 {
			fmt.Fprintf(os.Stderr, "=-= weakify fn %v reloc %d %+v\n", fn, i,
				fn.LSym.R[i])
		}
		// set the R_WEAK bit, leave rest of reloc type intact
		fn.LSym.R[i].Type |= objabi.R_WEAK
	}
}

// StackOffset returns the stack location of a LocalSlot relative to the
// stack pointer, suitable for use in a DWARF location entry. This has nothing
// to do with its offset in the user variable.
func StackOffset(slot ssa.LocalSlot) int32 {
	n := slot.N
	var off int64
	switch n.Class {
	case ir.PPARAM, ir.PPARAMOUT:
		if !n.IsOutputParamInRegisters() {
			off = n.FrameOffset() + base.Ctxt.Arch.FixedFrameSize
			break
		}
		fallthrough // PPARAMOUT in registers allocates like an AUTO
	case ir.PAUTO:
		off = n.FrameOffset()
		if base.Ctxt.Arch.FixedFrameSize == 0 {
			off -= int64(types.PtrSize)
		}
		if buildcfg.FramePointerEnabled {
			off -= int64(types.PtrSize)
		}
	}
	return int32(off + slot.Off)
}

// fieldtrack adds R_USEFIELD relocations to fnsym to record any
// struct fields that it used.
func fieldtrack(fnsym *obj.LSym, tracked map[*obj.LSym]struct{}) {
	if fnsym == nil {
		return
	}
	if !buildcfg.Experiment.FieldTrack || len(tracked) == 0 {
		return
	}

	trackSyms := make([]*obj.LSym, 0, len(tracked))
	for sym := range tracked {
		trackSyms = append(trackSyms, sym)
	}
	sort.Slice(trackSyms, func(i, j int) bool { return trackSyms[i].Name < trackSyms[j].Name })
	for _, sym := range trackSyms {
		r := obj.Addrel(fnsym)
		r.Sym = sym
		r.Type = objabi.R_USEFIELD
	}
}

// largeStack is info about a function whose stack frame is too large (rare).
type largeStack struct {
	locals int64
	args   int64
	callee int64
	pos    src.XPos
}

var (
	largeStackFramesMu sync.Mutex // protects largeStackFrames
	largeStackFrames   []largeStack
)

func CheckLargeStacks() {
	// Check whether any of the functions we have compiled have gigantic stack frames.
	sort.Slice(largeStackFrames, func(i, j int) bool {
		return largeStackFrames[i].pos.Before(largeStackFrames[j].pos)
	})
	for _, large := range largeStackFrames {
		if large.callee != 0 {
			base.ErrorfAt(large.pos, 0, "stack frame too large (>1GB): %d MB locals + %d MB args + %d MB callee", large.locals>>20, large.args>>20, large.callee>>20)
		} else {
			base.ErrorfAt(large.pos, 0, "stack frame too large (>1GB): %d MB locals + %d MB args", large.locals>>20, large.args>>20)
		}
	}
}
