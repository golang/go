// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

// This file contains the mem2reg pass.
// mem2reg promotes memory operations to register operations.
// This is a classic compiler optimization that can significantly
// improve the performance of generated code by reducing the number
// of memory accesses.
//
// The algorithm identifies memory stores (OpStore) to stack slots
// that are followed by loads (OpLoad) from the same stack slot,
// with no intervening stores to that slot. The loads are then
// replaced with the value that was stored.

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"internal/buildcfg"
	"slices"
)

type useType int

const (
	useLoad   useType = 1 << iota // used as the address to load from
	useStore                      // used as the address to store to
	useOffset                     // pointer arithmetic (OpOffPtr)
	useCopy                       // uses propagated through a copy
	useZero                       // used in OpZero
	useMove                       // used in OpMove
	useOther                      // includes escaping uses and OpPtrIndex
)

// useTable wraps the shared reverse-use table (see uses.go) so that
// queries about values created after the table was built - their IDs
// are out of range - report no uses instead of panicking. The mem2reg
// and decomposeAddr passes ask about such values after their rewrites.
type useTable struct {
	useInfo
}

// of returns the uses of v recorded in u, or nil if v postdates the table.
func (u *useTable) of(v *ssa.Value) []*ssa.Value {
	if int(v.ID) >= len(u.starts) {
		return nil
	}
	return u.get(v)
}

// classifyUses analyzes how a pointer is used and returns a bitmask of use types.
func classifyUses(v *ssa.Value, uses *useTable) useType {
	q := []*ssa.Value{v}
	var u useType
	for len(q) > 0 {
		curr := q[len(q)-1]
		q = q[:len(q)-1]
		for _, use := range uses.of(curr) {
			switch use.Op {
			case ssaop.OpCopy:
				u |= useCopy
				q = append(q, use)
			case ssaop.OpOffPtr:
				u |= useOffset
				q = append(q, use)
			case ssaop.OpLoad:
				u |= useLoad
			case ssaop.OpStore:
				if curr == use.Args[1] {
					// Storing the address as value.
					u |= useOther
				} else {
					u |= useStore
				}
			case ssaop.OpZero:
				u |= useZero
			case ssaop.OpMove:
				u |= useMove
			default:
				// TODO: handle more cases?
				// e.g. AddPtr, SubPtr, PtrIndex, NilCheck, EqPtr, NeqPtr
				u |= useOther
			}
		}
	}
	return u
}

// variableDemographic summarizes how the address of an auto variable is
// used, over all LocalAddr uses of that variable.
type variableDemographic struct {
	// ls is true when the address of this variable is only used by Load
	// and Store, and no VarLive is applied on the name.
	ls bool
	// lszmco is true when the address of this variable is only used by
	// Load, Store, Zero, Move, Copy and OffPtr, and no VarLive is
	// applied on the name.
	lszmco  bool
	varDefs []*ssa.Value // All the VarDefs on this name.
}

// variableDemographics analyzes f and returns the variable demographics data of f.
//
//	demographics is the map from names to their demographics.
//	localAddrs is the LocalAddr ssa values of the names in demographics.
//	u tracks the uses of the values in f (see uses.go); it is nil, and
//	nothing is allocated, when the function has no candidate variables.
//
// The caller must free u (u.free(f)) when non-nil.
func variableDemographics(f *ssa.Func) (
	demographics map[ssa.Aux]*variableDemographic,
	localAddrs []*ssa.Value,
	u *useTable) {
	// Fast path: most functions have no candidate LocalAddr at all, and
	// for them the pass chain has nothing to do. Detect that with a scan
	// that allocates nothing, so those functions pay (almost) nothing.
	found := false
	for _, b := range f.Blocks {
		for _, c := range b.ControlValues() {
			if c.Op == ssaop.OpOffPtr || c.Op == ssaop.OpLocalAddr {
				f.Fatalf("unexpected pointer value in block control")
			}
		}
		for _, v := range b.Values {
			if v.Op == ssaop.OpLocalAddr {
				if n := v.Aux.(*ir.Name); n.Class == ir.PAUTO || isABIInternalParam(f, n) {
					found = true
					break
				}
			}
		}
		if found {
			break
		}
	}
	if !found {
		return nil, nil, nil
	}

	// Build the reverse-use table using the shared cache-backed
	// infrastructure from uses.go.
	u = &useTable{uses(f)}
	varLives := map[ssa.Aux]bool{}
	demographics = make(map[ssa.Aux]*variableDemographic)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case ssaop.OpVarLive:
				varLives[v.Aux] = true
			case ssaop.OpVarDef:
				d := demographics[v.Aux]
				if d == nil {
					d = &variableDemographic{ls: true, lszmco: true}
					demographics[v.Aux] = d
				}
				d.varDefs = append(d.varDefs, v)
			}
		}
	}
	// Compute the use pattern of variables and the memory access demographic patterns.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == ssaop.OpLocalAddr {
				if n := v.Aux.(*ir.Name); n.Class == ir.PAUTO || isABIInternalParam(f, n) {
					ut := classifyUses(v, u)
					d := demographics[n]
					if d == nil {
						d = &variableDemographic{ls: true, lszmco: true}
						demographics[n] = d
					}
					if ut&^(useLoad|useStore) != 0 {
						d.ls = false
					}
					if ut&useOther != 0 {
						d.lszmco = false
					}
					localAddrs = append(localAddrs, v)
				}
			}
		}
	}
	// Names that have a VarLive applied are expected by the runtime to stay
	// in memory (e.g. to be kept alive across a call for the GC), so they do
	// not qualify for promotion.
	for n := range varLives {
		if d := demographics[n]; d != nil {
			d.ls = false
			d.lszmco = false
		}
	}
	return
}

var reinterpretOpMap = map[[2]types.Kind]ssaop.Op{
	{types.TUINT32, types.TFLOAT32}: ssaop.OpI32AsF32,
	{types.TINT32, types.TFLOAT32}:  ssaop.OpI32AsF32,
	{types.TFLOAT32, types.TUINT32}: ssaop.OpF32AsI32,
	{types.TFLOAT32, types.TINT32}:  ssaop.OpF32AsI32,
	{types.TUINT64, types.TFLOAT64}: ssaop.OpI64AsF64,
	{types.TINT64, types.TFLOAT64}:  ssaop.OpI64AsF64,
	{types.TFLOAT64, types.TUINT64}: ssaop.OpF64AsI64,
	{types.TFLOAT64, types.TINT64}:  ssaop.OpF64AsI64,
}

// reinterpretOp returns a t1 => t2 reinterpret op if available in [reinterpretOpMap].
// Otherwise it returns OpInvalid.
func reinterpretOp(t1, t2 *types.Type) ssaop.Op {
	if buildcfg.GOARCH != "amd64" {
		// Currently only amd64 has the proper lowering rules for these ops.
		return ssaop.OpInvalid
	}
	if op, ok := reinterpretOpMap[[2]types.Kind{t1.Kind(), t2.Kind()}]; ok {
		return op
	}

	return ssaop.OpInvalid
}

// copyCompatibleType reports whether a value of type t1 can directly stand in
// for a value of type t2 (same size and compatible representation).
func copyCompatibleType(t1, t2 *types.Type) bool {
	if t1.Size() != t2.Size() {
		return false
	}
	if t1.IsInteger() {
		return t2.IsInteger()
	}
	if ssa.IsPtr(t1) {
		return ssa.IsPtr(t2)
	}
	return t1.Compare(t2) == types.CMPeq
}

// sortLoadStores sorts loadStores (Loads and Stores in block bb) into
// program order, i.e. the order given by bb's store chain. This is needed
// because there is no guarantee of their order in b.Values before the
// schedule pass.
//
// memoryOrders is a per-block cache, populated on demand:
// memoryOrders[b.ID][v.ID] is the position of the memory value v in b's
// store chain, in even increments. Position 0 is the memory coming into
// the block (InitMem, a memory Phi, or any memory defined outside bb);
// each memory-producing value in bb (a Store, but also any other memory
// generator such as a call) gets the position of its memory arg plus
// two. A Load sorts at the position of its memory arg plus one - odd,
// so after the value that produced that memory and before whatever
// consumes it, which orders Stores before Loads of their output memory
// with a plain integer comparison.
func sortLoadStores(bb *ssa.Block, loadStores []*ssa.Value, memoryOrders map[ssa.ID]map[ssa.ID]int) []*ssa.Value {
	memOrder, ok := memoryOrders[bb.ID]
	if !ok {
		memOrder = make(map[ssa.ID]int)
		var computeDepth func(v *ssa.Value) int
		computeDepth = func(v *ssa.Value) int {
			if d, ok := memOrder[v.ID]; ok {
				return d
			}
			// Starting point
			if v.Block != bb || v.Op == ssaop.OpInitMem || v.Op == ssaop.OpPhi {
				memOrder[v.ID] = 0
				return 0
			}
			// Search backwards
			d := computeDepth(v.MemoryArg()) + 2
			memOrder[v.ID] = d
			return d
		}
		for _, v := range bb.Values {
			if v.Type.IsMemory() {
				computeDepth(v)
			}
		}
		memoryOrders[bb.ID] = memOrder
	}
	key := func(v *ssa.Value) int {
		if v.Op == ssaop.OpLoad {
			return memOrder[v.MemoryArg().ID] + 1
		}
		return memOrder[v.ID]
	}
	slices.SortFunc(loadStores, func(a, b *ssa.Value) int {
		return key(a) - key(b)
	})
	return loadStores
}

func mem2reg(f *ssa.Func) {
	changed := false
	if base.Flag.N != 0 {
		return
	}
	st := f.NewStats("mem2reg")

	// Get demographics
	demographics, localAddrs, uses := variableDemographics(f)
	if uses == nil {
		// No candidate variables; nothing to do.
		return
	}
	defer uses.free(f)
	memoryOrders := make(map[ssa.ID]map[ssa.ID]int)

	// First, we need to group all LocalAddr/Addrs that point to the same variable name
	// varGrouped[n] = all localAddrs that have n as their Aux field and have only load and stores.
	varGrouped := make(map[*ir.Name][]*ssa.Value)
	for _, v := range localAddrs {
		// TODO: mem2reg currently only supports load and stores on the LocalAddr directly.
		// However as a small step further it could also handle copy-derived addresses:
		// OffPtr [0] LocalAddr
		// Copy    LocalAddr
		if d, ok := demographics[v.Aux]; !ok || !d.ls {
			continue
		}
		n := v.Aux.(*ir.Name)
		if n.Class != ir.PAUTO {
			// Not handling params right now.
			continue
		}
		varGrouped[n] = append(varGrouped[n], v)
	}
	namesOrdered := []*ir.Name{}
	for n, vag := range varGrouped {
		namesOrdered = append(namesOrdered, n)
		slices.SortFunc(vag, func(a, b *ssa.Value) int { return int(a.ID - b.ID) })
	}
	slices.SortFunc(namesOrdered, func(a, b *ir.Name) int {
		return int(varGrouped[a][0].ID - varGrouped[b][0].ID)
	})
	// Utility functions
	removeStore := func(v *ssa.Value) {
		changed = true
		v.SetArgs1(v.MemoryArg())
		v.Aux = nil
		v.AuxInt = 0
		v.Op = ssaop.OpCopy
	}
	type loadCandidate struct {
		l *ssa.Value // The load
		v *ssa.Value // The value to replace the load
		// if v needs to be reinterpreted to match l's type, the op required. OpInvalid otherwise.
		// if l.Type != v.Type, this op must not be OpInvalid.
		reinterpret ssaop.Op
	}
	replaceLoad := func(lc loadCandidate) {
		changed = true
		if lc.reinterpret == ssaop.OpInvalid {
			if !copyCompatibleType(lc.l.Type, lc.v.Type) {
				f.Fatalf("mem2reg: load is being replaced by a value of an incompatible type")
			}
			lc.l.SetArgs1(lc.v)
		} else {
			// TODO: if we want to be more cautious, check that the reinterpret op will
			// actually make the type right.
			lc.l.SetArgs1(lc.l.Block.NewValue1(lc.l.Pos, lc.reinterpret, lc.l.Type, lc.v))
		}
		lc.l.Aux = nil
		lc.l.AuxInt = 0
		lc.l.Op = ssaop.OpCopy
	}
	// Now we should start the promotion
	// Simple case one - all uses of v is within the same block.
	storeCands := []*ssa.Value{} // These stores are to be overwritten
	loadCands := []loadCandidate{}
	vaUses := []*ssa.Value{}
NextVar:
	for _, n := range namesOrdered {
		vag := varGrouped[n]
		var block *ssa.Block
		for _, va := range vag {
			// Walk through all uses of v and check if they are within the same block
			for _, use := range uses.of(va) {
				if block == nil {
					block = use.Block
				}
				if use.Block != block {
					// Across control flow, needs DF and Phi.
					continue NextVar
				}
			}
		}
		// The uses are all within the same block and are loads/stores.
		storeCands = storeCands[:0]
		loadCands = loadCands[:0]
		vaUses = vaUses[:0]
		for _, va := range vag {
			vaUses = append(vaUses, uses.of(va)...)
		}
		vaUses = sortLoadStores(block, vaUses, memoryOrders)
		var curV *ssa.Value // The current value of n.
		for _, v := range vaUses {
			switch v.Op {
			case ssaop.OpLoad:
				if curV == nil {
					f.Fatalf("mem2reg sees a load from an auto variable before any store")
				}
				reinter := ssaop.OpInvalid
				if !copyCompatibleType(v.Type, curV.Type) {
					reinter = reinterpretOp(curV.Type, v.Type)
					if reinter == ssaop.OpInvalid {
						// Not something we can optimize, bailout the variable and
						// continue to the next variable.
						st.Record("incompatible types in single block case", 1)
						delete(varGrouped, n)
						continue NextVar
					}
				}
				loadCands = append(loadCands, loadCandidate{
					l:           v,
					v:           curV,
					reinterpret: reinter,
				})
			case ssaop.OpStore:
				curV = v.Args[1]
				storeCands = append(storeCands, v)
			default:
				f.Fatalf("should only has load or store uses")
			}
		}
		// Do the replacement.
		for _, v := range storeCands {
			// Remove the stores.
			removeStore(v)
		}
		for _, v := range demographics[n].varDefs {
			// Also remove the VarDefs.
			removeStore(v)
		}
		for _, lc := range loadCands {
			// Replace the loads with the corresponding value.
			replaceLoad(lc)
		}
		// This var is done, remove it from varGrouped.
		if f.Pass.Debug > 1 {
			f.Warnl(n.Pos(), "promoted %v in single block case", n)
		}
		delete(varGrouped, n)
		st.Record("promoted variable in single block case", 1)
	}

	// TODO: the rest of variables needs analysis across basic blocks, implement this.
	if changed {
		deadcode(f)
	}
}
