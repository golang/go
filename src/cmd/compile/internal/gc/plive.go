// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector liveness bitmap generation.

// The command line flag -live causes this code to print debug information.
// The levels are:
//
//	-live (aka -live=1): print liveness lists as code warnings at safe points
//	-live=2: print an assembly listing with liveness annotations
//
// Each level includes the earlier output as well.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"crypto/md5"
	"crypto/sha1"
	"fmt"
	"os"
	"strings"
)

// TODO(mdempsky): Update to reference OpVar{Def,Kill,Live} instead.

// VARDEF is an annotation for the liveness analysis, marking a place
// where a complete initialization (definition) of a variable begins.
// Since the liveness analysis can see initialization of single-word
// variables quite easy, gvardef is usually only called for multi-word
// or 'fat' variables, those satisfying isfat(n->type).
// However, gvardef is also called when a non-fat variable is initialized
// via a block move; the only time this happens is when you have
//	return f()
// for a function with multiple return values exactly matching the return
// types of the current function.
//
// A 'VARDEF x' annotation in the instruction stream tells the liveness
// analysis to behave as though the variable x is being initialized at that
// point in the instruction stream. The VARDEF must appear before the
// actual (multi-instruction) initialization, and it must also appear after
// any uses of the previous value, if any. For example, if compiling:
//
//	x = x[1:]
//
// it is important to generate code like:
//
//	base, len, cap = pieces of x[1:]
//	VARDEF x
//	x = {base, len, cap}
//
// If instead the generated code looked like:
//
//	VARDEF x
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//
// then the liveness analysis would decide the previous value of x was
// unnecessary even though it is about to be used by the x[1:] computation.
// Similarly, if the generated code looked like:
//
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//	VARDEF x
//
// then the liveness analysis will not preserve the new value of x, because
// the VARDEF appears to have "overwritten" it.
//
// VARDEF is a bit of a kludge to work around the fact that the instruction
// stream is working on single-word values but the liveness analysis
// wants to work on individual variables, which might be multi-word
// aggregates. It might make sense at some point to look into letting
// the liveness analysis work on single-word values as well, although
// there are complications around interface values, slices, and strings,
// all of which cannot be treated as individual words.
//
// VARKILL is the opposite of VARDEF: it marks a value as no longer needed,
// even if its address has been taken. That is, a VARKILL annotation asserts
// that its argument is certainly dead, for use when the liveness analysis
// would not otherwise be able to deduce that fact.

// BlockEffects summarizes the liveness effects on an SSA block.
type BlockEffects struct {
	lastbitmapindex int // for livenessepilogue

	// Computed during livenessprologue using only the content of
	// individual blocks:
	//
	//	uevar: upward exposed variables (used before set in block)
	//	varkill: killed variables (set in block)
	//	avarinit: addrtaken variables set or used (proof of initialization)
	uevar    bvec
	varkill  bvec
	avarinit bvec

	// Computed during livenesssolve using control flow information:
	//
	//	livein: variables live at block entry
	//	liveout: variables live at block exit
	//	avarinitany: addrtaken variables possibly initialized at block exit
	//		(initialized in block or at exit from any predecessor block)
	//	avarinitall: addrtaken variables certainly initialized at block exit
	//		(initialized in block or at exit from all predecessor blocks)
	livein      bvec
	liveout     bvec
	avarinitany bvec
	avarinitall bvec
}

// A collection of global state used by liveness analysis.
type Liveness struct {
	fn         *Node
	f          *ssa.Func
	vars       []*Node
	idx        map[*Node]int32
	stkptrsize int64

	be []BlockEffects

	// stackMapIndex maps from safe points (i.e., CALLs) to their
	// index within the stack maps.
	stackMapIndex map[*ssa.Value]int

	// An array with a bit vector for each safe point tracking live variables.
	livevars []bvec

	cache progeffectscache
}

type progeffectscache struct {
	textavarinit []int32
	retuevar     []int32
	tailuevar    []int32
	initialized  bool
}

// livenessShouldTrack reports whether the liveness analysis
// should track the variable n.
// We don't care about variables that have no pointers,
// nor do we care about non-local variables,
// nor do we care about empty structs (handled by the pointer check),
// nor do we care about the fake PAUTOHEAP variables.
func livenessShouldTrack(n *Node) bool {
	return n.Op == ONAME && (n.Class() == PAUTO || n.Class() == PPARAM || n.Class() == PPARAMOUT) && types.Haspointers(n.Type)
}

// getvariables returns the list of on-stack variables that we need to track
// and a map for looking up indices by *Node.
func getvariables(fn *Node) ([]*Node, map[*Node]int32) {
	var vars []*Node
	for _, n := range fn.Func.Dcl {
		if livenessShouldTrack(n) {
			vars = append(vars, n)
		}
	}
	idx := make(map[*Node]int32, len(vars))
	for i, n := range vars {
		idx[n] = int32(i)
	}
	return vars, idx
}

func (lv *Liveness) initcache() {
	if lv.cache.initialized {
		Fatalf("liveness cache initialized twice")
		return
	}
	lv.cache.initialized = true

	for i, node := range lv.vars {
		switch node.Class() {
		case PPARAM:
			// A return instruction with a p.to is a tail return, which brings
			// the stack pointer back up (if it ever went down) and then jumps
			// to a new function entirely. That form of instruction must read
			// all the parameters for correctness, and similarly it must not
			// read the out arguments - they won't be set until the new
			// function runs.

			lv.cache.tailuevar = append(lv.cache.tailuevar, int32(i))

			if node.Addrtaken() {
				lv.cache.textavarinit = append(lv.cache.textavarinit, int32(i))
			}

		case PPARAMOUT:
			// If the result had its address taken, it is being tracked
			// by the avarinit code, which does not use uevar.
			// If we added it to uevar too, we'd not see any kill
			// and decide that the variable was live entry, which it is not.
			// So only use uevar in the non-addrtaken case.
			// The p.to.type == obj.TYPE_NONE limits the bvset to
			// non-tail-call return instructions; see note below for details.
			if !node.Addrtaken() {
				lv.cache.retuevar = append(lv.cache.retuevar, int32(i))
			}
		}
	}
}

// A liveEffect is a set of flags that describe an instruction's
// liveness effects on a variable.
//
// The possible flags are:
//	uevar - used by the instruction
//	varkill - killed by the instruction
//		for variables without address taken, means variable was set
//		for variables with address taken, means variable was marked dead
//	avarinit - initialized or referred to by the instruction,
//		only for variables with address taken but not escaping to heap
//
// The avarinit output serves as a signal that the data has been
// initialized, because any use of a variable must come after its
// initialization.
type liveEffect int

const (
	uevar liveEffect = 1 << iota
	varkill
	avarinit
)

// valueEffects returns the index of a variable in lv.vars and the
// liveness effects v has on that variable.
// If v does not affect any tracked variables, it returns -1, 0.
func (lv *Liveness) valueEffects(v *ssa.Value) (int32, liveEffect) {
	n, e := affectedNode(v)
	if e == 0 || n == nil || n.Op != ONAME { // cheapest checks first
		return -1, 0
	}

	// AllocFrame has dropped unused variables from
	// lv.fn.Func.Dcl, but they might still be referenced by
	// OpVarFoo pseudo-ops. Ignore them to prevent "lost track of
	// variable" ICEs (issue 19632).
	switch v.Op {
	case ssa.OpVarDef, ssa.OpVarKill, ssa.OpVarLive, ssa.OpKeepAlive:
		if !n.Name.Used() {
			return -1, 0
		}
	}

	var effect liveEffect
	if n.Addrtaken() {
		if v.Op != ssa.OpVarKill {
			effect |= avarinit
		}
		if v.Op == ssa.OpVarDef || v.Op == ssa.OpVarKill {
			effect |= varkill
		}
	} else {
		// Read is a read, obviously.
		// Addr by itself is also implicitly a read.
		//
		// Addr|Write means that the address is being taken
		// but only so that the instruction can write to the value.
		// It is not a read.

		if e&ssa.SymRead != 0 || e&(ssa.SymAddr|ssa.SymWrite) == ssa.SymAddr {
			effect |= uevar
		}
		if e&ssa.SymWrite != 0 && (!isfat(n.Type) || v.Op == ssa.OpVarDef) {
			effect |= varkill
		}
	}

	if effect == 0 {
		return -1, 0
	}

	if pos, ok := lv.idx[n]; ok {
		return pos, effect
	}
	return -1, 0
}

// affectedNode returns the *Node affected by v
func affectedNode(v *ssa.Value) (*Node, ssa.SymEffect) {
	// Special cases.
	switch v.Op {
	case ssa.OpLoadReg:
		n, _ := AutoVar(v.Args[0])
		return n, ssa.SymRead
	case ssa.OpStoreReg:
		n, _ := AutoVar(v)
		return n, ssa.SymWrite

	case ssa.OpVarLive:
		return v.Aux.(*Node), ssa.SymRead
	case ssa.OpVarDef, ssa.OpVarKill:
		return v.Aux.(*Node), ssa.SymWrite
	case ssa.OpKeepAlive:
		n, _ := AutoVar(v.Args[0])
		return n, ssa.SymRead
	}

	e := v.Op.SymEffect()
	if e == 0 {
		return nil, 0
	}

	var n *Node
	switch a := v.Aux.(type) {
	case nil, *ssa.ExternSymbol:
		// ok, but no node
	case *ssa.ArgSymbol:
		n = a.Node.(*Node)
	case *ssa.AutoSymbol:
		n = a.Node.(*Node)
	default:
		Fatalf("weird aux: %s", v.LongString())
	}

	return n, e
}

// Constructs a new liveness structure used to hold the global state of the
// liveness computation. The cfg argument is a slice of *BasicBlocks and the
// vars argument is a slice of *Nodes.
func newliveness(fn *Node, f *ssa.Func, vars []*Node, idx map[*Node]int32, stkptrsize int64) *Liveness {
	lv := &Liveness{
		fn:         fn,
		f:          f,
		vars:       vars,
		idx:        idx,
		stkptrsize: stkptrsize,
		be:         make([]BlockEffects, f.NumBlocks()),
	}

	nblocks := int32(len(f.Blocks))
	nvars := int32(len(vars))
	bulk := bvbulkalloc(nvars, nblocks*7)
	for _, b := range f.Blocks {
		be := lv.blockEffects(b)

		be.uevar = bulk.next()
		be.varkill = bulk.next()
		be.livein = bulk.next()
		be.liveout = bulk.next()
		be.avarinit = bulk.next()
		be.avarinitany = bulk.next()
		be.avarinitall = bulk.next()
	}
	return lv
}

func (lv *Liveness) blockEffects(b *ssa.Block) *BlockEffects {
	return &lv.be[b.ID]
}

// NOTE: The bitmap for a specific type t should be cached in t after the first run
// and then simply copied into bv at the correct offset on future calls with
// the same type t. On https://rsc.googlecode.com/hg/testdata/slow.go, onebitwalktype1
// accounts for 40% of the 6g execution time.
func onebitwalktype1(t *types.Type, xoffset *int64, bv bvec) {
	if t.Align > 0 && *xoffset&int64(t.Align-1) != 0 {
		Fatalf("onebitwalktype1: invalid initial alignment, %v", t)
	}

	switch t.Etype {
	case TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TINT,
		TUINT,
		TUINTPTR,
		TBOOL,
		TFLOAT32,
		TFLOAT64,
		TCOMPLEX64,
		TCOMPLEX128:
		*xoffset += t.Width

	case TPTR32,
		TPTR64,
		TUNSAFEPTR,
		TFUNC,
		TCHAN,
		TMAP:
		if *xoffset&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bv.Set(int32(*xoffset / int64(Widthptr))) // pointer
		*xoffset += t.Width

	case TSTRING:
		// struct { byte *str; intgo len; }
		if *xoffset&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bv.Set(int32(*xoffset / int64(Widthptr))) //pointer in first slot
		*xoffset += t.Width

	case TINTER:
		// struct { Itab *tab;	void *data; }
		// or, when isnilinter(t)==true:
		// struct { Type *type; void *data; }
		if *xoffset&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bv.Set(int32(*xoffset / int64(Widthptr)))   // pointer in first slot
		bv.Set(int32(*xoffset/int64(Widthptr) + 1)) // pointer in second slot
		*xoffset += t.Width

	case TSLICE:
		// struct { byte *array; uintgo len; uintgo cap; }
		if *xoffset&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid TARRAY alignment, %v", t)
		}
		bv.Set(int32(*xoffset / int64(Widthptr))) // pointer in first slot (BitsPointer)
		*xoffset += t.Width

	case TARRAY:
		for i := int64(0); i < t.NumElem(); i++ {
			onebitwalktype1(t.Elem(), xoffset, bv)
		}

	case TSTRUCT:
		var o int64
		for _, t1 := range t.Fields().Slice() {
			fieldoffset := t1.Offset
			*xoffset += fieldoffset - o
			onebitwalktype1(t1.Type, xoffset, bv)
			o = fieldoffset + t1.Type.Width
		}

		*xoffset += t.Width - o

	default:
		Fatalf("onebitwalktype1: unexpected type, %v", t)
	}
}

// Returns the number of words of local variables.
func localswords(lv *Liveness) int32 {
	return int32(lv.stkptrsize / int64(Widthptr))
}

// Returns the number of words of in and out arguments.
func argswords(lv *Liveness) int32 {
	return int32(lv.fn.Type.ArgWidth() / int64(Widthptr))
}

// Generates live pointer value maps for arguments and local variables. The
// this argument and the in arguments are always assumed live. The vars
// argument is a slice of *Nodes.
func onebitlivepointermap(lv *Liveness, liveout bvec, vars []*Node, args bvec, locals bvec) {
	var xoffset int64

	for i := int32(0); ; i++ {
		i = liveout.Next(i)
		if i < 0 {
			break
		}
		node := vars[i]
		switch node.Class() {
		case PAUTO:
			xoffset = node.Xoffset + lv.stkptrsize
			onebitwalktype1(node.Type, &xoffset, locals)

		case PPARAM, PPARAMOUT:
			xoffset = node.Xoffset
			onebitwalktype1(node.Type, &xoffset, args)
		}
	}
}

// Returns true for instructions that are safe points that must be annotated
// with liveness information.
func issafepoint(v *ssa.Value) bool {
	return v.Op.IsCall()
}

// Initializes the sets for solving the live variables. Visits all the
// instructions in each basic block to summarizes the information at each basic
// block
func livenessprologue(lv *Liveness) {
	lv.initcache()

	for _, b := range lv.f.Blocks {
		be := lv.blockEffects(b)

		// Walk the block instructions backward and update the block
		// effects with the each prog effects.
		for j := len(b.Values) - 1; j >= 0; j-- {
			pos, e := lv.valueEffects(b.Values[j])
			if e&varkill != 0 {
				be.varkill.Set(pos)
				be.uevar.Unset(pos)
			}
			if e&uevar != 0 {
				be.uevar.Set(pos)
			}
		}

		// Walk the block instructions forward to update avarinit bits.
		// avarinit describes the effect at the end of the block, not the beginning.
		for j := 0; j < len(b.Values); j++ {
			pos, e := lv.valueEffects(b.Values[j])
			if e&varkill != 0 {
				be.avarinit.Unset(pos)
			}
			if e&avarinit != 0 {
				be.avarinit.Set(pos)
			}
		}
	}
}

// Solve the liveness dataflow equations.
func livenesssolve(lv *Liveness) {
	// These temporary bitvectors exist to avoid successive allocations and
	// frees within the loop.
	newlivein := bvalloc(int32(len(lv.vars)))
	newliveout := bvalloc(int32(len(lv.vars)))
	any := bvalloc(int32(len(lv.vars)))
	all := bvalloc(int32(len(lv.vars)))

	// Push avarinitall, avarinitany forward.
	// avarinitall says the addressed var is initialized along all paths reaching the block exit.
	// avarinitany says the addressed var is initialized along some path reaching the block exit.
	for _, b := range lv.f.Blocks {
		be := lv.blockEffects(b)
		if b == lv.f.Entry {
			be.avarinitall.Copy(be.avarinit)
		} else {
			be.avarinitall.Clear()
			be.avarinitall.Not()
		}
		be.avarinitany.Copy(be.avarinit)
	}

	// Walk blocks in the general direction of propagation (RPO
	// for avarinit{any,all}, and PO for live{in,out}). This
	// improves convergence.
	po := lv.f.Postorder()

	for change := true; change; {
		change = false
		for i := len(po) - 1; i >= 0; i-- {
			b := po[i]
			be := lv.blockEffects(b)
			lv.avarinitanyall(b, any, all)

			any.AndNot(any, be.varkill)
			all.AndNot(all, be.varkill)
			any.Or(any, be.avarinit)
			all.Or(all, be.avarinit)
			if !any.Eq(be.avarinitany) {
				change = true
				be.avarinitany.Copy(any)
			}

			if !all.Eq(be.avarinitall) {
				change = true
				be.avarinitall.Copy(all)
			}
		}
	}

	// Iterate through the blocks in reverse round-robin fashion. A work
	// queue might be slightly faster. As is, the number of iterations is
	// so low that it hardly seems to be worth the complexity.

	for change := true; change; {
		change = false
		for _, b := range po {
			be := lv.blockEffects(b)

			newliveout.Clear()
			switch b.Kind {
			case ssa.BlockRet:
				for _, pos := range lv.cache.retuevar {
					newliveout.Set(pos)
				}
			case ssa.BlockRetJmp:
				for _, pos := range lv.cache.tailuevar {
					newliveout.Set(pos)
				}
			case ssa.BlockExit:
				// nothing to do
			default:
				// A variable is live on output from this block
				// if it is live on input to some successor.
				//
				// out[b] = \bigcup_{s \in succ[b]} in[s]
				newliveout.Copy(lv.blockEffects(b.Succs[0].Block()).livein)
				for _, succ := range b.Succs[1:] {
					newliveout.Or(newliveout, lv.blockEffects(succ.Block()).livein)
				}
			}

			if !be.liveout.Eq(newliveout) {
				change = true
				be.liveout.Copy(newliveout)
			}

			// A variable is live on input to this block
			// if it is live on output from this block and
			// not set by the code in this block.
			//
			// in[b] = uevar[b] \cup (out[b] \setminus varkill[b])
			newlivein.AndNot(be.liveout, be.varkill)
			be.livein.Or(newlivein, be.uevar)
		}
	}
}

// Visits all instructions in a basic block and computes a bit vector of live
// variables at each safe point locations.
func livenessepilogue(lv *Liveness) {
	nvars := int32(len(lv.vars))
	liveout := bvalloc(nvars)
	any := bvalloc(nvars)
	all := bvalloc(nvars)
	livedefer := bvalloc(nvars) // always-live variables

	// If there is a defer (that could recover), then all output
	// parameters are live all the time.  In addition, any locals
	// that are pointers to heap-allocated output parameters are
	// also always live (post-deferreturn code needs these
	// pointers to copy values back to the stack).
	// TODO: if the output parameter is heap-allocated, then we
	// don't need to keep the stack copy live?
	if lv.fn.Func.HasDefer() {
		for i, n := range lv.vars {
			if n.Class() == PPARAMOUT {
				if n.IsOutputParamHeapAddr() {
					// Just to be paranoid.  Heap addresses are PAUTOs.
					Fatalf("variable %v both output param and heap output param", n)
				}
				if n.Name.Param.Heapaddr != nil {
					// If this variable moved to the heap, then
					// its stack copy is not live.
					continue
				}
				// Note: zeroing is handled by zeroResults in walk.go.
				livedefer.Set(int32(i))
			}
			if n.IsOutputParamHeapAddr() {
				n.Name.SetNeedzero(true)
				livedefer.Set(int32(i))
			}
		}
	}

	{
		// Reserve an entry for function entry.
		live := bvalloc(nvars)
		for _, pos := range lv.cache.textavarinit {
			live.Set(pos)
		}
		lv.livevars = append(lv.livevars, live)
	}

	for _, b := range lv.f.Blocks {
		be := lv.blockEffects(b)

		// Compute avarinitany and avarinitall for entry to block.
		// This duplicates information known during livenesssolve
		// but avoids storing two more vectors for each block.
		lv.avarinitanyall(b, any, all)

		// Walk forward through the basic block instructions and
		// allocate liveness maps for those instructions that need them.
		// Seed the maps with information about the addrtaken variables.
		for _, v := range b.Values {
			pos, e := lv.valueEffects(v)
			if e&varkill != 0 {
				any.Unset(pos)
				all.Unset(pos)
			}
			if e&avarinit != 0 {
				any.Set(pos)
				all.Set(pos)
			}

			if !issafepoint(v) {
				continue
			}

			// Annotate ambiguously live variables so that they can
			// be zeroed at function entry and at VARKILL points.
			// liveout is dead here and used as a temporary.
			liveout.AndNot(any, all)
			if !liveout.IsEmpty() {
				for pos := int32(0); pos < liveout.n; pos++ {
					if !liveout.Get(pos) {
						continue
					}
					all.Set(pos) // silence future warnings in this block
					n := lv.vars[pos]
					if !n.Name.Needzero() {
						n.Name.SetNeedzero(true)
						if debuglive >= 1 {
							Warnl(v.Pos, "%v: %L is ambiguously live", lv.fn.Func.Nname, n)
						}
					}
				}
			}

			// Live stuff first.
			live := bvalloc(nvars)
			live.Copy(any)
			lv.livevars = append(lv.livevars, live)
		}

		be.lastbitmapindex = len(lv.livevars) - 1
	}

	for _, b := range lv.f.Blocks {
		be := lv.blockEffects(b)

		// walk backward, emit pcdata and populate the maps
		index := int32(be.lastbitmapindex)
		if index < 0 {
			// the first block we encounter should have the ATEXT so
			// at no point should pos ever be less than zero.
			Fatalf("livenessepilogue")
		}

		liveout.Copy(be.liveout)
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]

			if issafepoint(v) {
				// Found an interesting instruction, record the
				// corresponding liveness information.

				live := lv.livevars[index]
				live.Or(live, liveout)
				live.Or(live, livedefer) // only for non-entry safe points
				index--
			}

			// Update liveness information.
			pos, e := lv.valueEffects(v)
			if e&varkill != 0 {
				liveout.Unset(pos)
			}
			if e&uevar != 0 {
				liveout.Set(pos)
			}
		}

		if b == lv.f.Entry {
			if index != 0 {
				Fatalf("bad index for entry point: %v", index)
			}

			// Record live variables.
			live := lv.livevars[index]
			live.Or(live, liveout)
		}
	}

	// Useful sanity check: on entry to the function,
	// the only things that can possibly be live are the
	// input parameters.
	for j, n := range lv.vars {
		if n.Class() != PPARAM && lv.livevars[0].Get(int32(j)) {
			Fatalf("internal error: %v %L recorded as live on entry", lv.fn.Func.Nname, n)
		}
	}
}

func (lv *Liveness) clobber() {
	// The clobberdead experiment inserts code to clobber all the dead variables (locals and args)
	// before and after every safepoint. This experiment is useful for debugging the generation
	// of live pointer bitmaps.
	if objabi.Clobberdead_enabled == 0 {
		return
	}
	var varSize int64
	for _, n := range lv.vars {
		varSize += n.Type.Size()
	}
	if len(lv.livevars) > 1000 || varSize > 10000 {
		// Be careful to avoid doing too much work.
		// Bail if >1000 safepoints or >10000 bytes of variables.
		// Otherwise, giant functions make this experiment generate too much code.
		return
	}
	if h := os.Getenv("GOCLOBBERDEADHASH"); h != "" {
		// Clobber only functions where the hash of the function name matches a pattern.
		// Useful for binary searching for a miscompiled function.
		hstr := ""
		for _, b := range sha1.Sum([]byte(lv.fn.funcname())) {
			hstr += fmt.Sprintf("%08b", b)
		}
		if !strings.HasSuffix(hstr, h) {
			return
		}
		fmt.Printf("\t\t\tCLOBBERDEAD %s\n", lv.fn.funcname())
	}
	if lv.f.Name == "forkAndExecInChild" {
		// forkAndExecInChild calls vfork (on linux/amd64, anyway).
		// The code we add here clobbers parts of the stack in the child.
		// When the parent resumes, it is using the same stack frame. But the
		// child has clobbered stack variables that the parent needs. Boom!
		// In particular, the sys argument gets clobbered.
		// Note to self: GOCLOBBERDEADHASH=011100101110
		return
	}

	var oldSched []*ssa.Value
	for _, b := range lv.f.Blocks {
		// Copy block's values to a temporary.
		oldSched = append(oldSched[:0], b.Values...)
		b.Values = b.Values[:0]

		// Clobber all dead variables at entry.
		if b == lv.f.Entry {
			for len(oldSched) > 0 && len(oldSched[0].Args) == 0 {
				// Skip argless ops. We need to skip at least
				// the lowered ClosurePtr op, because it
				// really wants to be first. This will also
				// skip ops like InitMem and SP, which are ok.
				b.Values = append(b.Values, oldSched[0])
				oldSched = oldSched[1:]
			}
			clobber(lv, b, lv.livevars[0])
		}

		// Copy values into schedule, adding clobbering around safepoints.
		for _, v := range oldSched {
			if !issafepoint(v) {
				b.Values = append(b.Values, v)
				continue
			}
			before := true
			if v.Op.IsCall() && v.Aux != nil && v.Aux.(*obj.LSym) == typedmemmove {
				// Can't put clobber code before the call to typedmemmove.
				// The variable to-be-copied is marked as dead
				// at the callsite. That is ok, though, as typedmemmove
				// is marked as nosplit, and the first thing it does
				// is to call memmove (also nosplit), after which
				// the source value is dead.
				// See issue 16026.
				before = false
			}
			if before {
				clobber(lv, b, lv.livevars[lv.stackMapIndex[v]])
			}
			b.Values = append(b.Values, v)
			clobber(lv, b, lv.livevars[lv.stackMapIndex[v]])
		}
	}
}

// clobber generates code to clobber all dead variables (those not marked in live).
// Clobbering instructions are added to the end of b.Values.
func clobber(lv *Liveness, b *ssa.Block, live bvec) {
	for i, n := range lv.vars {
		if !live.Get(int32(i)) {
			clobberVar(b, n)
		}
	}
}

// clobberVar generates code to trash the pointers in v.
// Clobbering instructions are added to the end of b.Values.
func clobberVar(b *ssa.Block, v *Node) {
	clobberWalk(b, v, 0, v.Type)
}

// b = block to which we append instructions
// v = variable
// offset = offset of (sub-portion of) variable to clobber (in bytes)
// t = type of sub-portion of v.
func clobberWalk(b *ssa.Block, v *Node, offset int64, t *types.Type) {
	if !types.Haspointers(t) {
		return
	}
	switch t.Etype {
	case TPTR32,
		TPTR64,
		TUNSAFEPTR,
		TFUNC,
		TCHAN,
		TMAP:
		clobberPtr(b, v, offset)

	case TSTRING:
		// struct { byte *str; int len; }
		clobberPtr(b, v, offset)

	case TINTER:
		// struct { Itab *tab; void *data; }
		// or, when isnilinter(t)==true:
		// struct { Type *type; void *data; }
		clobberPtr(b, v, offset)
		clobberPtr(b, v, offset+int64(Widthptr))

	case TSLICE:
		// struct { byte *array; int len; int cap; }
		clobberPtr(b, v, offset)

	case TARRAY:
		for i := int64(0); i < t.NumElem(); i++ {
			clobberWalk(b, v, offset+i*t.Elem().Size(), t.Elem())
		}

	case TSTRUCT:
		for _, t1 := range t.Fields().Slice() {
			clobberWalk(b, v, offset+t1.Offset, t1.Type)
		}

	default:
		Fatalf("clobberWalk: unexpected type, %v", t)
	}
}

// clobberPtr generates a clobber of the pointer at offset offset in v.
// The clobber instruction is added at the end of b.
func clobberPtr(b *ssa.Block, v *Node, offset int64) {
	var aux interface{}
	if v.Class() == PAUTO {
		aux = &ssa.AutoSymbol{Node: v}
	} else {
		aux = &ssa.ArgSymbol{Node: v}
	}
	b.NewValue0IA(src.NoXPos, ssa.OpClobber, types.TypeVoid, offset, aux)
}

func (lv *Liveness) avarinitanyall(b *ssa.Block, any, all bvec) {
	if len(b.Preds) == 0 {
		any.Clear()
		all.Clear()
		for _, pos := range lv.cache.textavarinit {
			any.Set(pos)
			all.Set(pos)
		}
		return
	}

	be := lv.blockEffects(b.Preds[0].Block())
	any.Copy(be.avarinitany)
	all.Copy(be.avarinitall)

	for _, pred := range b.Preds[1:] {
		be := lv.blockEffects(pred.Block())
		any.Or(any, be.avarinitany)
		all.And(all, be.avarinitall)
	}
}

// FNV-1 hash function constants.
const (
	H0 = 2166136261
	Hp = 16777619
)

func hashbitmap(h uint32, bv bvec) uint32 {
	n := int((bv.n + 31) / 32)
	for i := 0; i < n; i++ {
		w := bv.b[i]
		h = (h * Hp) ^ (w & 0xff)
		h = (h * Hp) ^ ((w >> 8) & 0xff)
		h = (h * Hp) ^ ((w >> 16) & 0xff)
		h = (h * Hp) ^ ((w >> 24) & 0xff)
	}

	return h
}

// Compact liveness information by coalescing identical per-call-site bitmaps.
// The merging only happens for a single function, not across the entire binary.
//
// There are actually two lists of bitmaps, one list for the local variables and one
// list for the function arguments. Both lists are indexed by the same PCDATA
// index, so the corresponding pairs must be considered together when
// merging duplicates. The argument bitmaps change much less often during
// function execution than the local variable bitmaps, so it is possible that
// we could introduce a separate PCDATA index for arguments vs locals and
// then compact the set of argument bitmaps separately from the set of
// local variable bitmaps. As of 2014-04-02, doing this to the godoc binary
// is actually a net loss: we save about 50k of argument bitmaps but the new
// PCDATA tables cost about 100k. So for now we keep using a single index for
// both bitmap lists.
func livenesscompact(lv *Liveness) {
	// Linear probing hash table of bitmaps seen so far.
	// The hash table has 4n entries to keep the linear
	// scan short. An entry of -1 indicates an empty slot.
	n := len(lv.livevars)

	tablesize := 4 * n
	table := make([]int, tablesize)
	for i := range table {
		table[i] = -1
	}

	// remap[i] = the new index of the old bit vector #i.
	remap := make([]int, n)
	for i := range remap {
		remap[i] = -1
	}
	uniq := 0 // unique tables found so far

	// Consider bit vectors in turn.
	// If new, assign next number using uniq,
	// record in remap, record in lv.livevars
	// under the new index, and add entry to hash table.
	// If already seen, record earlier index in remap.
Outer:
	for i, live := range lv.livevars {
		h := hashbitmap(H0, live) % uint32(tablesize)

		for {
			j := table[h]
			if j < 0 {
				break
			}
			jlive := lv.livevars[j]
			if live.Eq(jlive) {
				remap[i] = j
				continue Outer
			}

			h++
			if h == uint32(tablesize) {
				h = 0
			}
		}

		table[h] = uniq
		remap[i] = uniq
		lv.livevars[uniq] = live
		uniq++
	}

	// We've already reordered lv.livevars[0:uniq]. Clear the
	// pointers later in the array so they can be GC'd.
	tail := lv.livevars[uniq:]
	for i := range tail { // memclr loop pattern
		tail[i] = bvec{}
	}
	lv.livevars = lv.livevars[:uniq]

	// Rewrite PCDATA instructions to use new numbering.
	lv.showlive(nil, lv.livevars[0])
	pos := 1
	lv.stackMapIndex = make(map[*ssa.Value]int)
	for _, b := range lv.f.Blocks {
		for _, v := range b.Values {
			if issafepoint(v) {
				lv.showlive(v, lv.livevars[remap[pos]])
				lv.stackMapIndex[v] = int(remap[pos])
				pos++
			}
		}
	}
}

func (lv *Liveness) showlive(v *ssa.Value, live bvec) {
	if debuglive == 0 || lv.fn.funcname() == "init" || strings.HasPrefix(lv.fn.funcname(), ".") {
		return
	}
	if live.IsEmpty() {
		return
	}

	pos := lv.fn.Func.Nname.Pos
	if v != nil {
		pos = v.Pos
	}

	s := "live at "
	if v == nil {
		s += fmt.Sprintf("entry to %s:", lv.fn.funcname())
	} else if sym, ok := v.Aux.(*obj.LSym); ok {
		fn := sym.Name
		if pos := strings.Index(fn, "."); pos >= 0 {
			fn = fn[pos+1:]
		}
		s += fmt.Sprintf("call to %s:", fn)
	} else {
		s += "indirect call:"
	}

	for j, n := range lv.vars {
		if live.Get(int32(j)) {
			s += fmt.Sprintf(" %v", n)
		}
	}

	Warnl(pos, s)
}

func (lv *Liveness) printbvec(printed bool, name string, live bvec) bool {
	started := false
	for i, n := range lv.vars {
		if !live.Get(int32(i)) {
			continue
		}
		if !started {
			if !printed {
				fmt.Printf("\t")
			} else {
				fmt.Printf(" ")
			}
			started = true
			printed = true
			fmt.Printf("%s=", name)
		} else {
			fmt.Printf(",")
		}

		fmt.Printf("%s", n.Sym.Name)
	}
	return printed
}

// printeffect is like printbvec, but for a single variable.
func (lv *Liveness) printeffect(printed bool, name string, pos int32, x bool) bool {
	if !x {
		return printed
	}
	if !printed {
		fmt.Printf("\t")
	} else {
		fmt.Printf(" ")
	}
	fmt.Printf("%s=%s", name, lv.vars[pos].Sym.Name)
	return true
}

// Prints the computed liveness information and inputs, for debugging.
// This format synthesizes the information used during the multiple passes
// into a single presentation.
func livenessprintdebug(lv *Liveness) {
	fmt.Printf("liveness: %s\n", lv.fn.funcname())

	pcdata := 0
	for i, b := range lv.f.Blocks {
		if i > 0 {
			fmt.Printf("\n")
		}

		// bb#0 pred=1,2 succ=3,4
		fmt.Printf("bb#%d pred=", b.ID)
		for j, pred := range b.Preds {
			if j > 0 {
				fmt.Printf(",")
			}
			fmt.Printf("%d", pred.Block().ID)
		}
		fmt.Printf(" succ=")
		for j, succ := range b.Succs {
			if j > 0 {
				fmt.Printf(",")
			}
			fmt.Printf("%d", succ.Block().ID)
		}
		fmt.Printf("\n")

		be := lv.blockEffects(b)

		// initial settings
		printed := false
		printed = lv.printbvec(printed, "uevar", be.uevar)
		printed = lv.printbvec(printed, "livein", be.livein)
		if printed {
			fmt.Printf("\n")
		}

		// program listing, with individual effects listed

		if b == lv.f.Entry {
			live := lv.livevars[pcdata]
			fmt.Printf("(%s) function entry\n", linestr(lv.fn.Func.Nname.Pos))
			fmt.Printf("\tlive=")
			printed = false
			for j, n := range lv.vars {
				if !live.Get(int32(j)) {
					continue
				}
				if printed {
					fmt.Printf(",")
				}
				fmt.Printf("%v", n)
				printed = true
			}
			fmt.Printf("\n")
		}

		for _, v := range b.Values {
			fmt.Printf("(%s) %v\n", linestr(v.Pos), v.LongString())

			if pos, ok := lv.stackMapIndex[v]; ok {
				pcdata = pos
			}

			pos, effect := lv.valueEffects(v)
			printed = false
			printed = lv.printeffect(printed, "uevar", pos, effect&uevar != 0)
			printed = lv.printeffect(printed, "varkill", pos, effect&varkill != 0)
			printed = lv.printeffect(printed, "avarinit", pos, effect&avarinit != 0)
			if printed {
				fmt.Printf("\n")
			}

			if !issafepoint(v) {
				continue
			}

			live := lv.livevars[pcdata]
			fmt.Printf("\tlive=")
			printed = false
			for j, n := range lv.vars {
				if !live.Get(int32(j)) {
					continue
				}
				if printed {
					fmt.Printf(",")
				}
				fmt.Printf("%v", n)
				printed = true
			}
			fmt.Printf("\n")
		}

		// bb bitsets
		fmt.Printf("end\n")
		printed = false
		printed = lv.printbvec(printed, "varkill", be.varkill)
		printed = lv.printbvec(printed, "liveout", be.liveout)
		printed = lv.printbvec(printed, "avarinit", be.avarinit)
		printed = lv.printbvec(printed, "avarinitany", be.avarinitany)
		printed = lv.printbvec(printed, "avarinitall", be.avarinitall)
		if printed {
			fmt.Printf("\n")
		}
	}

	fmt.Printf("\n")
}

// Dumps a slice of bitmaps to a symbol as a sequence of uint32 values. The
// first word dumped is the total number of bitmaps. The second word is the
// length of the bitmaps. All bitmaps are assumed to be of equal length. The
// remaining bytes are the raw bitmaps.
func livenessemit(lv *Liveness, argssym, livesym *obj.LSym) {
	args := bvalloc(argswords(lv))
	aoff := duint32(argssym, 0, uint32(len(lv.livevars))) // number of bitmaps
	aoff = duint32(argssym, aoff, uint32(args.n))         // number of bits in each bitmap

	locals := bvalloc(localswords(lv))
	loff := duint32(livesym, 0, uint32(len(lv.livevars))) // number of bitmaps
	loff = duint32(livesym, loff, uint32(locals.n))       // number of bits in each bitmap

	for _, live := range lv.livevars {
		args.Clear()
		locals.Clear()

		onebitlivepointermap(lv, live, lv.vars, args, locals)

		aoff = dbvec(argssym, aoff, args)
		loff = dbvec(livesym, loff, locals)
	}

	// Give these LSyms content-addressable names,
	// so that they can be de-duplicated.
	// This provides significant binary size savings.
	// It is safe to rename these LSyms because
	// they are tracked separately from ctxt.hash.
	argssym.Name = fmt.Sprintf("gclocals·%x", md5.Sum(argssym.P))
	livesym.Name = fmt.Sprintf("gclocals·%x", md5.Sum(livesym.P))
}

// Entry pointer for liveness analysis. Solves for the liveness of
// pointer variables in the function and emits a runtime data
// structure read by the garbage collector.
// Returns a map from GC safe points to their corresponding stack map index.
func liveness(e *ssafn, f *ssa.Func) map[*ssa.Value]int {
	// Construct the global liveness state.
	vars, idx := getvariables(e.curfn)
	lv := newliveness(e.curfn, f, vars, idx, e.stkptrsize)

	// Run the dataflow framework.
	livenessprologue(lv)
	livenesssolve(lv)
	livenessepilogue(lv)
	livenesscompact(lv)
	lv.clobber()
	if debuglive >= 2 {
		livenessprintdebug(lv)
	}

	// Emit the live pointer map data structures
	if ls := e.curfn.Func.lsym; ls != nil {
		livenessemit(lv, &ls.Func.GCArgs, &ls.Func.GCLocals)
	}
	return lv.stackMapIndex
}
