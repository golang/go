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
	"crypto/md5"
	"fmt"
	"strings"
)

// go115ReduceLiveness disables register maps and only produces stack
// maps at call sites.
//
// In Go 1.15, we changed debug call injection to use conservative
// scanning instead of precise pointer maps, so these are no longer
// necessary.
//
// Keep in sync with runtime/preempt.go:go115ReduceLiveness.
const go115ReduceLiveness = true

// OpVarDef is an annotation for the liveness analysis, marking a place
// where a complete initialization (definition) of a variable begins.
// Since the liveness analysis can see initialization of single-word
// variables quite easy, OpVarDef is only needed for multi-word
// variables satisfying isfat(n.Type). For simplicity though, buildssa
// emits OpVarDef regardless of variable width.
//
// An 'OpVarDef x' annotation in the instruction stream tells the liveness
// analysis to behave as though the variable x is being initialized at that
// point in the instruction stream. The OpVarDef must appear before the
// actual (multi-instruction) initialization, and it must also appear after
// any uses of the previous value, if any. For example, if compiling:
//
//	x = x[1:]
//
// it is important to generate code like:
//
//	base, len, cap = pieces of x[1:]
//	OpVarDef x
//	x = {base, len, cap}
//
// If instead the generated code looked like:
//
//	OpVarDef x
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//
// then the liveness analysis would decide the previous value of x was
// unnecessary even though it is about to be used by the x[1:] computation.
// Similarly, if the generated code looked like:
//
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//	OpVarDef x
//
// then the liveness analysis will not preserve the new value of x, because
// the OpVarDef appears to have "overwritten" it.
//
// OpVarDef is a bit of a kludge to work around the fact that the instruction
// stream is working on single-word values but the liveness analysis
// wants to work on individual variables, which might be multi-word
// aggregates. It might make sense at some point to look into letting
// the liveness analysis work on single-word values as well, although
// there are complications around interface values, slices, and strings,
// all of which cannot be treated as individual words.
//
// OpVarKill is the opposite of OpVarDef: it marks a value as no longer needed,
// even if its address has been taken. That is, an OpVarKill annotation asserts
// that its argument is certainly dead, for use when the liveness analysis
// would not otherwise be able to deduce that fact.

// TODO: get rid of OpVarKill here. It's useful for stack frame allocation
// so the compiler can allocate two temps to the same location. Here it's now
// useless, since the implementation of stack objects.

// BlockEffects summarizes the liveness effects on an SSA block.
type BlockEffects struct {
	// Computed during Liveness.prologue using only the content of
	// individual blocks:
	//
	//	uevar: upward exposed variables (used before set in block)
	//	varkill: killed variables (set in block)
	uevar   varRegVec
	varkill varRegVec

	// Computed during Liveness.solve using control flow information:
	//
	//	livein: variables live at block entry
	//	liveout: variables live at block exit
	livein  varRegVec
	liveout varRegVec
}

// A collection of global state used by liveness analysis.
type Liveness struct {
	fn         *Node
	f          *ssa.Func
	vars       []*Node
	idx        map[*Node]int32
	stkptrsize int64

	be []BlockEffects

	// allUnsafe indicates that all points in this function are
	// unsafe-points.
	allUnsafe bool
	// unsafePoints bit i is set if Value ID i is an unsafe-point
	// (preemption is not allowed). Only valid if !allUnsafe.
	unsafePoints bvec

	// An array with a bit vector for each safe point in the
	// current Block during Liveness.epilogue. Indexed in Value
	// order for that block. Additionally, for the entry block
	// livevars[0] is the entry bitmap. Liveness.compact moves
	// these to stackMaps and regMaps.
	livevars []varRegVec

	// livenessMap maps from safe points (i.e., CALLs) to their
	// liveness map indexes.
	livenessMap LivenessMap
	stackMapSet bvecSet
	stackMaps   []bvec
	regMapSet   map[liveRegMask]int
	regMaps     []liveRegMask

	cache progeffectscache

	// These are only populated if open-coded defers are being used.
	// List of vars/stack slots storing defer args
	openDeferVars []openDeferVarInfo
	// Map from defer arg OpVarDef to the block where the OpVarDef occurs.
	openDeferVardefToBlockMap map[*Node]*ssa.Block
	// Map of blocks that cannot reach a return or exit (panic)
	nonReturnBlocks map[*ssa.Block]bool
}

type openDeferVarInfo struct {
	n         *Node // Var/stack slot storing a defer arg
	varsIndex int   // Index of variable in lv.vars
}

// LivenessMap maps from *ssa.Value to LivenessIndex.
type LivenessMap struct {
	vals map[ssa.ID]LivenessIndex
}

func (m *LivenessMap) reset() {
	if m.vals == nil {
		m.vals = make(map[ssa.ID]LivenessIndex)
	} else {
		for k := range m.vals {
			delete(m.vals, k)
		}
	}
}

func (m *LivenessMap) set(v *ssa.Value, i LivenessIndex) {
	m.vals[v.ID] = i
}

func (m LivenessMap) Get(v *ssa.Value) LivenessIndex {
	if !go115ReduceLiveness {
		// All safe-points are in the map, so if v isn't in
		// the map, it's an unsafe-point.
		if idx, ok := m.vals[v.ID]; ok {
			return idx
		}
		return LivenessInvalid
	}

	// If v isn't in the map, then it's a "don't care" and not an
	// unsafe-point.
	if idx, ok := m.vals[v.ID]; ok {
		return idx
	}
	return LivenessIndex{StackMapDontCare, StackMapDontCare, false}
}

// LivenessIndex stores the liveness map information for a Value.
type LivenessIndex struct {
	stackMapIndex int
	regMapIndex   int // only for !go115ReduceLiveness

	// isUnsafePoint indicates that this is an unsafe-point.
	//
	// Note that it's possible for a call Value to have a stack
	// map while also being an unsafe-point. This means it cannot
	// be preempted at this instruction, but that a preemption or
	// stack growth may happen in the called function.
	isUnsafePoint bool
}

// LivenessInvalid indicates an unsafe point with no stack map.
var LivenessInvalid = LivenessIndex{StackMapDontCare, StackMapDontCare, true} // only for !go115ReduceLiveness

// StackMapDontCare indicates that the stack map index at a Value
// doesn't matter.
//
// This is a sentinel value that should never be emitted to the PCDATA
// stream. We use -1000 because that's obviously never a valid stack
// index (but -1 is).
const StackMapDontCare = -1000

func (idx LivenessIndex) StackMapValid() bool {
	return idx.stackMapIndex != StackMapDontCare
}

func (idx LivenessIndex) RegMapValid() bool {
	return idx.regMapIndex != StackMapDontCare
}

type progeffectscache struct {
	retuevar    []int32
	tailuevar   []int32
	initialized bool
}

// varRegVec contains liveness bitmaps for variables and registers.
type varRegVec struct {
	vars bvec
	regs liveRegMask
}

func (v *varRegVec) Eq(v2 varRegVec) bool {
	return v.vars.Eq(v2.vars) && v.regs == v2.regs
}

func (v *varRegVec) Copy(v2 varRegVec) {
	v.vars.Copy(v2.vars)
	v.regs = v2.regs
}

func (v *varRegVec) Clear() {
	v.vars.Clear()
	v.regs = 0
}

func (v *varRegVec) Or(v1, v2 varRegVec) {
	v.vars.Or(v1.vars, v2.vars)
	v.regs = v1.regs | v2.regs
}

func (v *varRegVec) AndNot(v1, v2 varRegVec) {
	v.vars.AndNot(v1.vars, v2.vars)
	v.regs = v1.regs &^ v2.regs
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

		case PPARAMOUT:
			// All results are live at every return point.
			// Note that this point is after escaping return values
			// are copied back to the stack using their PAUTOHEAP references.
			lv.cache.retuevar = append(lv.cache.retuevar, int32(i))
		}
	}
}

// A liveEffect is a set of flags that describe an instruction's
// liveness effects on a variable.
//
// The possible flags are:
//	uevar - used by the instruction
//	varkill - killed by the instruction (set)
// A kill happens after the use (for an instruction that updates a value, for example).
type liveEffect int

const (
	uevar liveEffect = 1 << iota
	varkill
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
	// Read is a read, obviously.
	//
	// Addr is a read also, as any subsequent holder of the pointer must be able
	// to see all the values (including initialization) written so far.
	// This also prevents a variable from "coming back from the dead" and presenting
	// stale pointers to the garbage collector. See issue 28445.
	if e&(ssa.SymRead|ssa.SymAddr) != 0 {
		effect |= uevar
	}
	if e&ssa.SymWrite != 0 && (!isfat(n.Type) || v.Op == ssa.OpVarDef) {
		effect |= varkill
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

	switch a := v.Aux.(type) {
	case nil, *obj.LSym:
		// ok, but no node
		return nil, e
	case *Node:
		return a, e
	default:
		Fatalf("weird aux: %s", v.LongString())
		return nil, e
	}
}

// regEffects returns the registers affected by v.
func (lv *Liveness) regEffects(v *ssa.Value) (uevar, kill liveRegMask) {
	if go115ReduceLiveness {
		return 0, 0
	}
	if v.Op == ssa.OpPhi {
		// All phi node arguments must come from the same
		// register and the result must also go to that
		// register, so there's no overall effect.
		return 0, 0
	}
	addLocs := func(mask liveRegMask, v *ssa.Value, ptrOnly bool) liveRegMask {
		if int(v.ID) >= len(lv.f.RegAlloc) {
			// v has no allocated registers.
			return mask
		}
		loc := lv.f.RegAlloc[v.ID]
		if loc == nil {
			// v has no allocated registers.
			return mask
		}
		if v.Op == ssa.OpGetG {
			// GetG represents the G register, which is a
			// pointer, but not a valid GC register. The
			// current G is always reachable, so it's okay
			// to ignore this register.
			return mask
		}

		// Collect registers and types from v's location.
		var regs [2]*ssa.Register
		nreg := 0
		switch loc := loc.(type) {
		case ssa.LocalSlot:
			return mask
		case *ssa.Register:
			if ptrOnly && !v.Type.HasHeapPointer() {
				return mask
			}
			regs[0] = loc
			nreg = 1
		case ssa.LocPair:
			// The value will have TTUPLE type, and the
			// children are nil or *ssa.Register.
			if v.Type.Etype != types.TTUPLE {
				v.Fatalf("location pair %s has non-tuple type %v", loc, v.Type)
			}
			for i, loc1 := range &loc {
				if loc1 == nil {
					continue
				}
				if ptrOnly && !v.Type.FieldType(i).HasHeapPointer() {
					continue
				}
				regs[nreg] = loc1.(*ssa.Register)
				nreg++
			}
		default:
			v.Fatalf("weird RegAlloc location: %s (%T)", loc, loc)
		}

		// Add register locations to vars.
		for _, reg := range regs[:nreg] {
			if reg.GCNum() == -1 {
				if ptrOnly {
					v.Fatalf("pointer in non-pointer register %v", reg)
				} else {
					continue
				}
			}
			mask |= 1 << uint(reg.GCNum())
		}
		return mask
	}

	// v clobbers all registers it writes to (whether or not the
	// write is pointer-typed).
	kill = addLocs(0, v, false)
	for _, arg := range v.Args {
		// v uses all registers is reads from, but we only
		// care about marking those containing pointers.
		uevar = addLocs(uevar, arg, true)
	}
	return uevar, kill
}

type liveRegMask uint32 // only if !go115ReduceLiveness

func (m liveRegMask) niceString(config *ssa.Config) string {
	if m == 0 {
		return "<none>"
	}
	str := ""
	for i, reg := range config.GCRegMap {
		if m&(1<<uint(i)) != 0 {
			if str != "" {
				str += ","
			}
			str += reg.String()
		}
	}
	return str
}

type livenessFuncCache struct {
	be          []BlockEffects
	livenessMap LivenessMap
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

		regMapSet: make(map[liveRegMask]int),
	}

	// Significant sources of allocation are kept in the ssa.Cache
	// and reused. Surprisingly, the bit vectors themselves aren't
	// a major source of allocation, but the liveness maps are.
	if lc, _ := f.Cache.Liveness.(*livenessFuncCache); lc == nil {
		// Prep the cache so liveness can fill it later.
		f.Cache.Liveness = new(livenessFuncCache)
	} else {
		if cap(lc.be) >= f.NumBlocks() {
			lv.be = lc.be[:f.NumBlocks()]
		}
		lv.livenessMap = LivenessMap{lc.livenessMap.vals}
		lc.livenessMap.vals = nil
	}
	if lv.be == nil {
		lv.be = make([]BlockEffects, f.NumBlocks())
	}

	nblocks := int32(len(f.Blocks))
	nvars := int32(len(vars))
	bulk := bvbulkalloc(nvars, nblocks*7)
	for _, b := range f.Blocks {
		be := lv.blockEffects(b)

		be.uevar = varRegVec{vars: bulk.next()}
		be.varkill = varRegVec{vars: bulk.next()}
		be.livein = varRegVec{vars: bulk.next()}
		be.liveout = varRegVec{vars: bulk.next()}
	}
	lv.livenessMap.reset()

	lv.markUnsafePoints()
	return lv
}

func (lv *Liveness) blockEffects(b *ssa.Block) *BlockEffects {
	return &lv.be[b.ID]
}

// NOTE: The bitmap for a specific type t could be cached in t after
// the first run and then simply copied into bv at the correct offset
// on future calls with the same type t.
func onebitwalktype1(t *types.Type, off int64, bv bvec) {
	if t.Align > 0 && off&int64(t.Align-1) != 0 {
		Fatalf("onebitwalktype1: invalid initial alignment: type %v has alignment %d, but offset is %v", t, t.Align, off)
	}

	switch t.Etype {
	case TINT8, TUINT8, TINT16, TUINT16,
		TINT32, TUINT32, TINT64, TUINT64,
		TINT, TUINT, TUINTPTR, TBOOL,
		TFLOAT32, TFLOAT64, TCOMPLEX64, TCOMPLEX128:

	case TPTR, TUNSAFEPTR, TFUNC, TCHAN, TMAP:
		if off&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bv.Set(int32(off / int64(Widthptr))) // pointer

	case TSTRING:
		// struct { byte *str; intgo len; }
		if off&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bv.Set(int32(off / int64(Widthptr))) //pointer in first slot

	case TINTER:
		// struct { Itab *tab;	void *data; }
		// or, when isnilinter(t)==true:
		// struct { Type *type; void *data; }
		if off&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		// The first word of an interface is a pointer, but we don't
		// treat it as such.
		// 1. If it is a non-empty interface, the pointer points to an itab
		//    which is always in persistentalloc space.
		// 2. If it is an empty interface, the pointer points to a _type.
		//   a. If it is a compile-time-allocated type, it points into
		//      the read-only data section.
		//   b. If it is a reflect-allocated type, it points into the Go heap.
		//      Reflect is responsible for keeping a reference to
		//      the underlying type so it won't be GCd.
		// If we ever have a moving GC, we need to change this for 2b (as
		// well as scan itabs to update their itab._type fields).
		bv.Set(int32(off/int64(Widthptr) + 1)) // pointer in second slot

	case TSLICE:
		// struct { byte *array; uintgo len; uintgo cap; }
		if off&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid TARRAY alignment, %v", t)
		}
		bv.Set(int32(off / int64(Widthptr))) // pointer in first slot (BitsPointer)

	case TARRAY:
		elt := t.Elem()
		if elt.Width == 0 {
			// Short-circuit for #20739.
			break
		}
		for i := int64(0); i < t.NumElem(); i++ {
			onebitwalktype1(elt, off, bv)
			off += elt.Width
		}

	case TSTRUCT:
		for _, f := range t.Fields().Slice() {
			onebitwalktype1(f.Type, off+f.Offset, bv)
		}

	default:
		Fatalf("onebitwalktype1: unexpected type, %v", t)
	}
}

// usedRegs returns the maximum width of the live register map.
func (lv *Liveness) usedRegs() int32 {
	var any liveRegMask
	for _, live := range lv.regMaps {
		any |= live
	}
	i := int32(0)
	for any != 0 {
		any >>= 1
		i++
	}
	return i
}

// Generates live pointer value maps for arguments and local variables. The
// this argument and the in arguments are always assumed live. The vars
// argument is a slice of *Nodes.
func (lv *Liveness) pointerMap(liveout bvec, vars []*Node, args, locals bvec) {
	for i := int32(0); ; i++ {
		i = liveout.Next(i)
		if i < 0 {
			break
		}
		node := vars[i]
		switch node.Class() {
		case PAUTO:
			onebitwalktype1(node.Type, node.Xoffset+lv.stkptrsize, locals)

		case PPARAM, PPARAMOUT:
			onebitwalktype1(node.Type, node.Xoffset, args)
		}
	}
}

// allUnsafe indicates that all points in this function are
// unsafe-points.
func allUnsafe(f *ssa.Func) bool {
	// The runtime assumes the only safe-points are function
	// prologues (because that's how it used to be). We could and
	// should improve that, but for now keep consider all points
	// in the runtime unsafe. obj will add prologues and their
	// safe-points.
	//
	// go:nosplit functions are similar. Since safe points used to
	// be coupled with stack checks, go:nosplit often actually
	// means "no safe points in this function".
	return compiling_runtime || f.NoSplit
}

// markUnsafePoints finds unsafe points and computes lv.unsafePoints.
func (lv *Liveness) markUnsafePoints() {
	if allUnsafe(lv.f) {
		// No complex analysis necessary.
		lv.allUnsafe = true
		return
	}

	lv.unsafePoints = bvalloc(int32(lv.f.NumValues()))

	// Mark architecture-specific unsafe points.
	for _, b := range lv.f.Blocks {
		for _, v := range b.Values {
			if v.Op.UnsafePoint() {
				lv.unsafePoints.Set(int32(v.ID))
			}
		}
	}

	// Mark write barrier unsafe points.
	for _, wbBlock := range lv.f.WBLoads {
		if wbBlock.Kind == ssa.BlockPlain && len(wbBlock.Values) == 0 {
			// The write barrier block was optimized away
			// but we haven't done dead block elimination.
			// (This can happen in -N mode.)
			continue
		}
		// Check that we have the expected diamond shape.
		if len(wbBlock.Succs) != 2 {
			lv.f.Fatalf("expected branch at write barrier block %v", wbBlock)
		}
		s0, s1 := wbBlock.Succs[0].Block(), wbBlock.Succs[1].Block()
		if s0 == s1 {
			// There's no difference between write barrier on and off.
			// Thus there's no unsafe locations. See issue 26024.
			continue
		}
		if s0.Kind != ssa.BlockPlain || s1.Kind != ssa.BlockPlain {
			lv.f.Fatalf("expected successors of write barrier block %v to be plain", wbBlock)
		}
		if s0.Succs[0].Block() != s1.Succs[0].Block() {
			lv.f.Fatalf("expected successors of write barrier block %v to converge", wbBlock)
		}

		// Flow backwards from the control value to find the
		// flag load. We don't know what lowered ops we're
		// looking for, but all current arches produce a
		// single op that does the memory load from the flag
		// address, so we look for that.
		var load *ssa.Value
		v := wbBlock.Controls[0]
		for {
			if sym, ok := v.Aux.(*obj.LSym); ok && sym == writeBarrier {
				load = v
				break
			}
			switch v.Op {
			case ssa.Op386TESTL:
				// 386 lowers Neq32 to (TESTL cond cond),
				if v.Args[0] == v.Args[1] {
					v = v.Args[0]
					continue
				}
			case ssa.Op386MOVLload, ssa.OpARM64MOVWUload, ssa.OpPPC64MOVWZload, ssa.OpWasmI64Load32U:
				// Args[0] is the address of the write
				// barrier control. Ignore Args[1],
				// which is the mem operand.
				// TODO: Just ignore mem operands?
				v = v.Args[0]
				continue
			}
			// Common case: just flow backwards.
			if len(v.Args) != 1 {
				v.Fatalf("write barrier control value has more than one argument: %s", v.LongString())
			}
			v = v.Args[0]
		}

		// Mark everything after the load unsafe.
		found := false
		for _, v := range wbBlock.Values {
			found = found || v == load
			if found {
				lv.unsafePoints.Set(int32(v.ID))
			}
		}

		// Mark the two successor blocks unsafe. These come
		// back together immediately after the direct write in
		// one successor and the last write barrier call in
		// the other, so there's no need to be more precise.
		for _, succ := range wbBlock.Succs {
			for _, v := range succ.Block().Values {
				lv.unsafePoints.Set(int32(v.ID))
			}
		}
	}

	// Find uintptr -> unsafe.Pointer conversions and flood
	// unsafeness back to a call (which is always a safe point).
	//
	// Looking for the uintptr -> unsafe.Pointer conversion has a
	// few advantages over looking for unsafe.Pointer -> uintptr
	// conversions:
	//
	// 1. We avoid needlessly blocking safe-points for
	// unsafe.Pointer -> uintptr conversions that never go back to
	// a Pointer.
	//
	// 2. We don't have to detect calls to reflect.Value.Pointer,
	// reflect.Value.UnsafeAddr, and reflect.Value.InterfaceData,
	// which are implicit unsafe.Pointer -> uintptr conversions.
	// We can't even reliably detect this if there's an indirect
	// call to one of these methods.
	//
	// TODO: For trivial unsafe.Pointer arithmetic, it would be
	// nice to only flood as far as the unsafe.Pointer -> uintptr
	// conversion, but it's hard to know which argument of an Add
	// or Sub to follow.
	var flooded bvec
	var flood func(b *ssa.Block, vi int)
	flood = func(b *ssa.Block, vi int) {
		if flooded.n == 0 {
			flooded = bvalloc(int32(lv.f.NumBlocks()))
		}
		if flooded.Get(int32(b.ID)) {
			return
		}
		for i := vi - 1; i >= 0; i-- {
			v := b.Values[i]
			if v.Op.IsCall() {
				// Uintptrs must not contain live
				// pointers across calls, so stop
				// flooding.
				return
			}
			lv.unsafePoints.Set(int32(v.ID))
		}
		if vi == len(b.Values) {
			// We marked all values in this block, so no
			// need to flood this block again.
			flooded.Set(int32(b.ID))
		}
		for _, pred := range b.Preds {
			flood(pred.Block(), len(pred.Block().Values))
		}
	}
	for _, b := range lv.f.Blocks {
		for i, v := range b.Values {
			if !(v.Op == ssa.OpConvert && v.Type.IsPtrShaped()) {
				continue
			}
			// Flood the unsafe-ness of this backwards
			// until we hit a call.
			flood(b, i+1)
		}
	}
}

// Returns true for instructions that must have a stack map.
//
// This does not necessarily mean the instruction is a safe-point. In
// particular, call Values can have a stack map in case the callee
// grows the stack, but not themselves be a safe-point.
func (lv *Liveness) hasStackMap(v *ssa.Value) bool {
	// The runtime only has safe-points in function prologues, so
	// we only need stack maps at call sites. go:nosplit functions
	// are similar.
	if go115ReduceLiveness || compiling_runtime || lv.f.NoSplit {
		if !v.Op.IsCall() {
			return false
		}
		// typedmemclr and typedmemmove are write barriers and
		// deeply non-preemptible. They are unsafe points and
		// hence should not have liveness maps.
		if sym, _ := v.Aux.(*obj.LSym); sym == typedmemclr || sym == typedmemmove {
			return false
		}
		return true
	}

	switch v.Op {
	case ssa.OpInitMem, ssa.OpArg, ssa.OpSP, ssa.OpSB,
		ssa.OpSelect0, ssa.OpSelect1, ssa.OpGetG,
		ssa.OpVarDef, ssa.OpVarLive, ssa.OpKeepAlive,
		ssa.OpPhi:
		// These don't produce code (see genssa).
		return false
	}
	return !lv.unsafePoints.Get(int32(v.ID))
}

// Initializes the sets for solving the live variables. Visits all the
// instructions in each basic block to summarizes the information at each basic
// block
func (lv *Liveness) prologue() {
	lv.initcache()

	if lv.fn.Func.HasDefer() && !lv.fn.Func.OpenCodedDeferDisallowed() {
		lv.openDeferVardefToBlockMap = make(map[*Node]*ssa.Block)
		for i, n := range lv.vars {
			if n.Name.OpenDeferSlot() {
				lv.openDeferVars = append(lv.openDeferVars, openDeferVarInfo{n: n, varsIndex: i})
			}
		}

		// Find any blocks that cannot reach a return or a BlockExit
		// (panic) -- these must be because of an infinite loop.
		reachesRet := make(map[ssa.ID]bool)
		blockList := make([]*ssa.Block, 0, 256)

		for _, b := range lv.f.Blocks {
			if b.Kind == ssa.BlockRet || b.Kind == ssa.BlockRetJmp || b.Kind == ssa.BlockExit {
				blockList = append(blockList, b)
			}
		}

		for len(blockList) > 0 {
			b := blockList[0]
			blockList = blockList[1:]
			if reachesRet[b.ID] {
				continue
			}
			reachesRet[b.ID] = true
			for _, e := range b.Preds {
				blockList = append(blockList, e.Block())
			}
		}

		lv.nonReturnBlocks = make(map[*ssa.Block]bool)
		for _, b := range lv.f.Blocks {
			if !reachesRet[b.ID] {
				lv.nonReturnBlocks[b] = true
				//fmt.Println("No reach ret", lv.f.Name, b.ID, b.Kind)
			}
		}
	}

	for _, b := range lv.f.Blocks {
		be := lv.blockEffects(b)

		// Walk the block instructions backward and update the block
		// effects with the each prog effects.
		for j := len(b.Values) - 1; j >= 0; j-- {
			if b.Values[j].Op == ssa.OpVarDef {
				n := b.Values[j].Aux.(*Node)
				if n.Name.OpenDeferSlot() {
					lv.openDeferVardefToBlockMap[n] = b
				}
			}
			pos, e := lv.valueEffects(b.Values[j])
			regUevar, regKill := lv.regEffects(b.Values[j])
			if e&varkill != 0 {
				be.varkill.vars.Set(pos)
				be.uevar.vars.Unset(pos)
			}
			be.varkill.regs |= regKill
			be.uevar.regs &^= regKill
			if e&uevar != 0 {
				be.uevar.vars.Set(pos)
			}
			be.uevar.regs |= regUevar
		}
	}
}

// markDeferVarsLive marks each variable storing an open-coded defer arg as
// specially live in block b if the variable definition dominates block b.
func (lv *Liveness) markDeferVarsLive(b *ssa.Block, newliveout *varRegVec) {
	// Only force computation of dominators if we have a block where we need
	// to specially mark defer args live.
	sdom := lv.f.Sdom()
	for _, info := range lv.openDeferVars {
		defB := lv.openDeferVardefToBlockMap[info.n]
		if sdom.IsAncestorEq(defB, b) {
			newliveout.vars.Set(int32(info.varsIndex))
		}
	}
}

// Solve the liveness dataflow equations.
func (lv *Liveness) solve() {
	// These temporary bitvectors exist to avoid successive allocations and
	// frees within the loop.
	nvars := int32(len(lv.vars))
	newlivein := varRegVec{vars: bvalloc(nvars)}
	newliveout := varRegVec{vars: bvalloc(nvars)}

	// Walk blocks in postorder ordering. This improves convergence.
	po := lv.f.Postorder()

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
					newliveout.vars.Set(pos)
				}
			case ssa.BlockRetJmp:
				for _, pos := range lv.cache.tailuevar {
					newliveout.vars.Set(pos)
				}
			case ssa.BlockExit:
				// panic exit - nothing to do
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

			if lv.fn.Func.HasDefer() && !lv.fn.Func.OpenCodedDeferDisallowed() &&
				(b.Kind == ssa.BlockExit || lv.nonReturnBlocks[b]) {
				// Open-coded defer args slots must be live
				// everywhere in a function, since a panic can
				// occur (almost) anywhere. Force all appropriate
				// defer arg slots to be live in BlockExit (panic)
				// blocks and in blocks that do not reach a return
				// (because of infinite loop).
				//
				// We are assuming that the defer exit code at
				// BlockReturn/BlockReturnJmp accesses all of the
				// defer args (with pointers), and so keeps them
				// live. This analysis may have to be adjusted if
				// that changes (because of optimizations).
				lv.markDeferVarsLive(b, &newliveout)
			}

			if !be.liveout.Eq(newliveout) {
				change = true
				be.liveout.Copy(newliveout)
			}

			// A variable is live on input to this block
			// if it is used by this block, or live on output from this block and
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
func (lv *Liveness) epilogue() {
	nvars := int32(len(lv.vars))
	liveout := varRegVec{vars: bvalloc(nvars)}
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
				if n.Name.IsOutputParamHeapAddr() {
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
			if n.Name.IsOutputParamHeapAddr() {
				// This variable will be overwritten early in the function
				// prologue (from the result of a mallocgc) but we need to
				// zero it in case that malloc causes a stack scan.
				n.Name.SetNeedzero(true)
				livedefer.Set(int32(i))
			}
		}
	}

	// We must analyze the entry block first. The runtime assumes
	// the function entry map is index 0. Conveniently, layout
	// already ensured that the entry block is first.
	if lv.f.Entry != lv.f.Blocks[0] {
		lv.f.Fatalf("entry block must be first")
	}

	{
		// Reserve an entry for function entry.
		live := bvalloc(nvars)
		lv.livevars = append(lv.livevars, varRegVec{vars: live})
	}

	for _, b := range lv.f.Blocks {
		be := lv.blockEffects(b)
		firstBitmapIndex := len(lv.livevars)

		// Walk forward through the basic block instructions and
		// allocate liveness maps for those instructions that need them.
		for _, v := range b.Values {
			if !lv.hasStackMap(v) {
				continue
			}

			live := bvalloc(nvars)
			lv.livevars = append(lv.livevars, varRegVec{vars: live})
		}

		// walk backward, construct maps at each safe point
		index := int32(len(lv.livevars) - 1)

		liveout.Copy(be.liveout)
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]

			if lv.hasStackMap(v) {
				// Found an interesting instruction, record the
				// corresponding liveness information.

				live := &lv.livevars[index]
				live.Or(*live, liveout)
				live.vars.Or(live.vars, livedefer) // only for non-entry safe points
				index--
			}

			// Update liveness information.
			pos, e := lv.valueEffects(v)
			regUevar, regKill := lv.regEffects(v)
			if e&varkill != 0 {
				liveout.vars.Unset(pos)
			}
			liveout.regs &^= regKill
			if e&uevar != 0 {
				liveout.vars.Set(pos)
			}
			liveout.regs |= regUevar
		}

		if b == lv.f.Entry {
			if index != 0 {
				Fatalf("bad index for entry point: %v", index)
			}

			// Check to make sure only input variables are live.
			for i, n := range lv.vars {
				if !liveout.vars.Get(int32(i)) {
					continue
				}
				if n.Class() == PPARAM {
					continue // ok
				}
				Fatalf("bad live variable at entry of %v: %L", lv.fn.Func.Nname, n)
			}

			// Record live variables.
			live := &lv.livevars[index]
			live.Or(*live, liveout)
		}

		// Check that no registers are live across calls.
		// For closure calls, the CALLclosure is the last use
		// of the context register, so it's dead after the call.
		index = int32(firstBitmapIndex)
		for _, v := range b.Values {
			if lv.hasStackMap(v) {
				live := lv.livevars[index]
				if v.Op.IsCall() && live.regs != 0 {
					lv.printDebug()
					v.Fatalf("%v register %s recorded as live at call", lv.fn.Func.Nname, live.regs.niceString(lv.f.Config))
				}
				index++
			}
		}

		// The liveness maps for this block are now complete. Compact them.
		lv.compact(b)
	}

	// Done compacting. Throw out the stack map set.
	lv.stackMaps = lv.stackMapSet.extractUniqe()
	lv.stackMapSet = bvecSet{}

	// Useful sanity check: on entry to the function,
	// the only things that can possibly be live are the
	// input parameters.
	for j, n := range lv.vars {
		if n.Class() != PPARAM && lv.stackMaps[0].Get(int32(j)) {
			lv.f.Fatalf("%v %L recorded as live on entry", lv.fn.Func.Nname, n)
		}
	}
	if !go115ReduceLiveness {
		// Check that no registers are live at function entry.
		// The context register, if any, comes from a
		// LoweredGetClosurePtr operation first thing in the function,
		// so it doesn't appear live at entry.
		if regs := lv.regMaps[0]; regs != 0 {
			lv.printDebug()
			lv.f.Fatalf("%v register %s recorded as live on entry", lv.fn.Func.Nname, regs.niceString(lv.f.Config))
		}
	}
}

// Compact coalesces identical bitmaps from lv.livevars into the sets
// lv.stackMapSet and lv.regMaps.
//
// Compact clears lv.livevars.
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
func (lv *Liveness) compact(b *ssa.Block) {
	add := func(live varRegVec, isUnsafePoint bool) LivenessIndex { // only if !go115ReduceLiveness
		// Deduplicate the stack map.
		stackIndex := lv.stackMapSet.add(live.vars)
		// Deduplicate the register map.
		regIndex, ok := lv.regMapSet[live.regs]
		if !ok {
			regIndex = len(lv.regMapSet)
			lv.regMapSet[live.regs] = regIndex
			lv.regMaps = append(lv.regMaps, live.regs)
		}
		return LivenessIndex{stackIndex, regIndex, isUnsafePoint}
	}
	pos := 0
	if b == lv.f.Entry {
		// Handle entry stack map.
		if !go115ReduceLiveness {
			add(lv.livevars[0], false)
		} else {
			lv.stackMapSet.add(lv.livevars[0].vars)
		}
		pos++
	}
	for _, v := range b.Values {
		if go115ReduceLiveness {
			hasStackMap := lv.hasStackMap(v)
			isUnsafePoint := lv.allUnsafe || lv.unsafePoints.Get(int32(v.ID))
			idx := LivenessIndex{StackMapDontCare, StackMapDontCare, isUnsafePoint}
			if hasStackMap {
				idx.stackMapIndex = lv.stackMapSet.add(lv.livevars[pos].vars)
				pos++
			}
			if hasStackMap || isUnsafePoint {
				lv.livenessMap.set(v, idx)
			}
		} else if lv.hasStackMap(v) {
			isUnsafePoint := lv.allUnsafe || lv.unsafePoints.Get(int32(v.ID))
			lv.livenessMap.set(v, add(lv.livevars[pos], isUnsafePoint))
			pos++
		}
	}

	// Reset livevars.
	lv.livevars = lv.livevars[:0]
}

func (lv *Liveness) showlive(v *ssa.Value, live bvec) {
	if debuglive == 0 || lv.fn.funcname() == "init" || strings.HasPrefix(lv.fn.funcname(), ".") {
		return
	}
	if !(v == nil || v.Op.IsCall()) {
		// Historically we only printed this information at
		// calls. Keep doing so.
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

func (lv *Liveness) printbvec(printed bool, name string, live varRegVec) bool {
	if live.vars.IsEmpty() && live.regs == 0 {
		return printed
	}

	if !printed {
		fmt.Printf("\t")
	} else {
		fmt.Printf(" ")
	}
	fmt.Printf("%s=", name)

	comma := ""
	for i, n := range lv.vars {
		if !live.vars.Get(int32(i)) {
			continue
		}
		fmt.Printf("%s%s", comma, n.Sym.Name)
		comma = ","
	}
	fmt.Printf("%s%s", comma, live.regs.niceString(lv.f.Config))
	return true
}

// printeffect is like printbvec, but for valueEffects and regEffects.
func (lv *Liveness) printeffect(printed bool, name string, pos int32, x bool, regMask liveRegMask) bool {
	if !x && regMask == 0 {
		return printed
	}
	if !printed {
		fmt.Printf("\t")
	} else {
		fmt.Printf(" ")
	}
	fmt.Printf("%s=", name)
	if x {
		fmt.Printf("%s", lv.vars[pos].Sym.Name)
	}
	for j, reg := range lv.f.Config.GCRegMap {
		if regMask&(1<<uint(j)) != 0 {
			if x {
				fmt.Printf(",")
			}
			x = true
			fmt.Printf("%v", reg)
		}
	}
	return true
}

// Prints the computed liveness information and inputs, for debugging.
// This format synthesizes the information used during the multiple passes
// into a single presentation.
func (lv *Liveness) printDebug() {
	fmt.Printf("liveness: %s\n", lv.fn.funcname())

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
			live := lv.stackMaps[0]
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

			pcdata := lv.livenessMap.Get(v)

			pos, effect := lv.valueEffects(v)
			regUevar, regKill := lv.regEffects(v)
			printed = false
			printed = lv.printeffect(printed, "uevar", pos, effect&uevar != 0, regUevar)
			printed = lv.printeffect(printed, "varkill", pos, effect&varkill != 0, regKill)
			if printed {
				fmt.Printf("\n")
			}

			if pcdata.StackMapValid() || pcdata.RegMapValid() {
				fmt.Printf("\tlive=")
				printed = false
				if pcdata.StackMapValid() {
					live := lv.stackMaps[pcdata.stackMapIndex]
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
				}
				if pcdata.RegMapValid() { // only if !go115ReduceLiveness
					regLive := lv.regMaps[pcdata.regMapIndex]
					if regLive != 0 {
						if printed {
							fmt.Printf(",")
						}
						fmt.Printf("%s", regLive.niceString(lv.f.Config))
						printed = true
					}
				}
				fmt.Printf("\n")
			}

			if pcdata.isUnsafePoint {
				fmt.Printf("\tunsafe-point\n")
			}
		}

		// bb bitsets
		fmt.Printf("end\n")
		printed = false
		printed = lv.printbvec(printed, "varkill", be.varkill)
		printed = lv.printbvec(printed, "liveout", be.liveout)
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
func (lv *Liveness) emit() (argsSym, liveSym, regsSym *obj.LSym) {
	// Size args bitmaps to be just large enough to hold the largest pointer.
	// First, find the largest Xoffset node we care about.
	// (Nodes without pointers aren't in lv.vars; see livenessShouldTrack.)
	var maxArgNode *Node
	for _, n := range lv.vars {
		switch n.Class() {
		case PPARAM, PPARAMOUT:
			if maxArgNode == nil || n.Xoffset > maxArgNode.Xoffset {
				maxArgNode = n
			}
		}
	}
	// Next, find the offset of the largest pointer in the largest node.
	var maxArgs int64
	if maxArgNode != nil {
		maxArgs = maxArgNode.Xoffset + typeptrdata(maxArgNode.Type)
	}

	// Size locals bitmaps to be stkptrsize sized.
	// We cannot shrink them to only hold the largest pointer,
	// because their size is used to calculate the beginning
	// of the local variables frame.
	// Further discussion in https://golang.org/cl/104175.
	// TODO: consider trimming leading zeros.
	// This would require shifting all bitmaps.
	maxLocals := lv.stkptrsize

	// Temporary symbols for encoding bitmaps.
	var argsSymTmp, liveSymTmp, regsSymTmp obj.LSym

	args := bvalloc(int32(maxArgs / int64(Widthptr)))
	aoff := duint32(&argsSymTmp, 0, uint32(len(lv.stackMaps))) // number of bitmaps
	aoff = duint32(&argsSymTmp, aoff, uint32(args.n))          // number of bits in each bitmap

	locals := bvalloc(int32(maxLocals / int64(Widthptr)))
	loff := duint32(&liveSymTmp, 0, uint32(len(lv.stackMaps))) // number of bitmaps
	loff = duint32(&liveSymTmp, loff, uint32(locals.n))        // number of bits in each bitmap

	for _, live := range lv.stackMaps {
		args.Clear()
		locals.Clear()

		lv.pointerMap(live, lv.vars, args, locals)

		aoff = dbvec(&argsSymTmp, aoff, args)
		loff = dbvec(&liveSymTmp, loff, locals)
	}

	if !go115ReduceLiveness {
		regs := bvalloc(lv.usedRegs())
		roff := duint32(&regsSymTmp, 0, uint32(len(lv.regMaps))) // number of bitmaps
		roff = duint32(&regsSymTmp, roff, uint32(regs.n))        // number of bits in each bitmap
		if regs.n > 32 {
			// Our uint32 conversion below won't work.
			Fatalf("GP registers overflow uint32")
		}

		if regs.n > 0 {
			for _, live := range lv.regMaps {
				regs.Clear()
				regs.b[0] = uint32(live)
				roff = dbvec(&regsSymTmp, roff, regs)
			}
		}
	}

	// Give these LSyms content-addressable names,
	// so that they can be de-duplicated.
	// This provides significant binary size savings.
	//
	// These symbols will be added to Ctxt.Data by addGCLocals
	// after parallel compilation is done.
	makeSym := func(tmpSym *obj.LSym) *obj.LSym {
		return Ctxt.LookupInit(fmt.Sprintf("gclocals·%x", md5.Sum(tmpSym.P)), func(lsym *obj.LSym) {
			lsym.P = tmpSym.P
		})
	}
	if !go115ReduceLiveness {
		return makeSym(&argsSymTmp), makeSym(&liveSymTmp), makeSym(&regsSymTmp)
	}
	// TODO(go115ReduceLiveness): Remove regsSym result
	return makeSym(&argsSymTmp), makeSym(&liveSymTmp), nil
}

// Entry pointer for liveness analysis. Solves for the liveness of
// pointer variables in the function and emits a runtime data
// structure read by the garbage collector.
// Returns a map from GC safe points to their corresponding stack map index.
func liveness(e *ssafn, f *ssa.Func, pp *Progs) LivenessMap {
	// Construct the global liveness state.
	vars, idx := getvariables(e.curfn)
	lv := newliveness(e.curfn, f, vars, idx, e.stkptrsize)

	// Run the dataflow framework.
	lv.prologue()
	lv.solve()
	lv.epilogue()
	if debuglive > 0 {
		lv.showlive(nil, lv.stackMaps[0])
		for _, b := range f.Blocks {
			for _, val := range b.Values {
				if idx := lv.livenessMap.Get(val); idx.StackMapValid() {
					lv.showlive(val, lv.stackMaps[idx.stackMapIndex])
				}
			}
		}
	}
	if debuglive >= 2 {
		lv.printDebug()
	}

	// Update the function cache.
	{
		cache := f.Cache.Liveness.(*livenessFuncCache)
		if cap(lv.be) < 2000 { // Threshold from ssa.Cache slices.
			for i := range lv.be {
				lv.be[i] = BlockEffects{}
			}
			cache.be = lv.be
		}
		if len(lv.livenessMap.vals) < 2000 {
			cache.livenessMap = lv.livenessMap
		}
	}

	// Emit the live pointer map data structures
	ls := e.curfn.Func.lsym
	ls.Func.GCArgs, ls.Func.GCLocals, ls.Func.GCRegs = lv.emit()

	p := pp.Prog(obj.AFUNCDATA)
	Addrconst(&p.From, objabi.FUNCDATA_ArgsPointerMaps)
	p.To.Type = obj.TYPE_MEM
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = ls.Func.GCArgs

	p = pp.Prog(obj.AFUNCDATA)
	Addrconst(&p.From, objabi.FUNCDATA_LocalsPointerMaps)
	p.To.Type = obj.TYPE_MEM
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = ls.Func.GCLocals

	if !go115ReduceLiveness {
		p = pp.Prog(obj.AFUNCDATA)
		Addrconst(&p.From, objabi.FUNCDATA_RegPointerMaps)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ls.Func.GCRegs
	}

	return lv.livenessMap
}

// isfat reports whether a variable of type t needs multiple assignments to initialize.
// For example:
//
// 	type T struct { x, y int }
// 	x := T{x: 0, y: 1}
//
// Then we need:
//
// 	var t T
// 	t.x = 0
// 	t.y = 1
//
// to fully initialize t.
func isfat(t *types.Type) bool {
	if t != nil {
		switch t.Etype {
		case TSLICE, TSTRING,
			TINTER: // maybe remove later
			return true
		case TARRAY:
			// Array of 1 element, check if element is fat
			if t.NumElem() == 1 {
				return isfat(t.Elem())
			}
			return true
		case TSTRUCT:
			// Struct with 1 field, check if field is fat
			if t.NumFields() == 1 {
				return isfat(t.Field(0).Type)
			}
			return true
		}
	}

	return false
}
