// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

// This file contains the decomposeAddr pass.
// decomposeAddr decomposes aggregate variables
// (structs and arrays) into separate scalar variables. This allows the subsequent
// mem2reg pass to promote these scalars to ssa values, improving
// performance by reducing memory traffic. This is assuming the regalloc
// can make good decisions under the increased pressure of more ssa values.
//
// This pass is also known as SROA (scalar replacement of aggregates) in other compilers.
//
// Overview:
// 1. Candidate identification: find local variables that don't qualify for mem2reg and
//    are not special. Special variables are NonMergeable or OpenDeferSlot.
// 2. Partitioning: determine if the aggregate's fields are accessed in a way that
//    permits decomposition (e.g., aligned with leaf field boundaries).
// 3. Breakdown of block operations: replace OpZero and OpMove on aggregates with
//    field-by-field scalar operations.
// 4. Rewriting: create new individual local variables for each leaf field and
//    rewrite all loads, stores, and VarDefs to use the new variables.
//
// Stability constraints:
// - OpVarDef is only emitted for new scalars if they:
//		- have pointers: the GC needs to know the live ranges of these variables.
//		- are merge candidates: mergelocals need to know the live ranges of these variables.
// - Aggregates that are too large are skipped to prevent compiler hangs or excessive
//   memory usage.

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"fmt"
	"slices"
)

type leafField struct {
	off int64
	typ *types.Type
}

type decomposeCandidate struct {
	n              *ir.Name
	leaves         []leafField
	leavesByOff    map[int64]leafField
	localAddrs     []*ssa.Value // all the LocalAddrs of this candidate.
	unDecomposable bool
}

var decomposeAddrDummyValue = &ssa.Value{}

// zeroValueForType returns a zero-valued SSA constant for the given type.
// Returns nil for unsupported types.
//
// f may be nil, in which case no constant is materialized: the function
// returns the non-nil decomposeAddrDummyValue for supported types (and nil,
// as usual, for unsupported ones). This mode is used during candidate
// selection to ask "could we build a zero value for this type?" without
// mutating the function; the dummy value itself must never be used as an
// SSA value.
func zeroValueForType(f *ssa.Func, t *types.Type) *ssa.Value {
	switch {
	case t.IsBoolean():
		if f == nil {
			return decomposeAddrDummyValue
		}
		return f.ConstBool(t, false)
	case t.IsInteger():
		if f == nil {
			return decomposeAddrDummyValue
		}
		switch t.Size() {
		case 1:
			return f.ConstInt8(t, 0)
		case 2:
			return f.ConstInt16(t, 0)
		case 4:
			return f.ConstInt32(t, 0)
		default:
			return f.ConstInt64(t, 0)
		}
	case t.IsFloat():
		if f == nil {
			return decomposeAddrDummyValue
		}
		switch t.Size() {
		case 4:
			return f.ConstFloat32(t, 0)
		default:
			return f.ConstFloat64(t, 0)
		}
	case t.IsPtrShaped():
		if f == nil {
			return decomposeAddrDummyValue
		}
		return f.ConstNil(t)
	case t.IsString():
		if f == nil {
			return decomposeAddrDummyValue
		}
		return f.ConstEmptyString(t)
	case t.IsSlice():
		if f == nil {
			return decomposeAddrDummyValue
		}
		return f.ConstSlice(t)
	case t.IsInterface():
		if f == nil {
			return decomposeAddrDummyValue
		}
		return f.ConstInterface(t)
	case t.IsSIMD():
		if f == nil {
			return decomposeAddrDummyValue
		}
		return f.ConstSIMD(t)
	default:
		// Unsupported types.
		// TODO: sometimes composite types like [1]uint64
		// can also get passed down here, maybe make this function aware of them as well?
		return nil
	}
}

// decomposeAddr decomposes aggregate variables
// (structs and arrays) accessed via OffPtr
// into separate scalar variables, allowing mem2reg to promote them.
func decomposeAddr(f *ssa.Func) {
	if base.Flag.N != 0 {
		return
	}

	// Get demographics up front: for a function with no candidate
	// LocalAddr (the common case) this allocates nothing and the pass
	// exits before setting up any of its machinery below.
	demographics, localAddrs, uses := variableDemographics(f)
	if uses == nil {
		return
	}
	defer func() { uses.free(f) }()

	st := f.NewStats("decomposeAddr")

	// maxPartitions limits the number of fields an aggregate can be decomposed into.
	// This prevents OOM and hangs on extremely large arrays or deeply nested structs.
	// Tunable for experiments via -d=ssa/decompose_addr/@maxpartitions=<n>.
	maxPartitions := f.Pass.Val("maxpartitions", 64)

	// maxActivePartitions limits the number of *active* fields an aggregate
	// can be decomposed into. While maxPartitions bounds how many leaf fields
	// an aggregate may have at all (limiting the work this pass does per
	// variable), maxActivePartitions bounds how many of those leaves are
	// actually read (limiting the register pressure the promotion creates):
	// a leaf is "active" if some Load reads it, and only active leaves become
	// live SSA values after the follow-up mem2reg promotion. For example, for
	//
	//	var a [8]int
	//	a[1] = x
	//	a[2] = y
	//	use(a[1])
	//
	// the aggregate has 8 leaves (all within maxPartitions), but only a[1] is
	// active. Leaves that are only stored to die immediately after promotion,
	// so they add no register pressure.
	// This is a parameter that controls the trade-off of less memory accesses
	// and higher register pressure for mem2reg.
	// If we set this number too high, the register allocation might behave suboptimally
	// and the performance might even regress.
	// Tunable for experiments via -d=ssa/decompose_addr/@maxactivepartitions=<n>.
	maxActivePartitions := f.Pass.Val("maxactivepartitions", 4)

	type partitionStatus int
	const (
		partSuccess partitionStatus = iota
		partTooMany                 // too many partitions, above maxPartitions limit
		partLarge                   // leaf field too large to be stored in one register
	)

	// getLeafFields recursively finds all scalar fields of a type with their given offsets.
	var getLeafFields func(t *types.Type, base int64) (leaves []leafField, status partitionStatus)
	getLeafFields = func(t *types.Type, base int64) (leaves []leafField, status partitionStatus) {
		// checkField updates leaves by looking into a composite type field (or element) of
		// type t2 at offset. It returns the status code.
		checkField := func(t2 *types.Type, offset int64) partitionStatus {
			ls, s := getLeafFields(t2, offset)
			if s != partSuccess {
				return s
			}
			if int64(len(leaves)+len(ls)) > maxPartitions {
				return partTooMany
			}
			leaves = append(leaves, ls...)
			return partSuccess
		}
		if t.IsStruct() && !t.IsSIMD() {
			for i := range t.NumFields() {
				fld := t.Field(i)
				if s := checkField(fld.Type, base+fld.Offset); s != partSuccess {
					return nil, s
				}
			}
		} else if t.IsArray() {
			// Limit decomposition of large arrays.
			if t.NumElem() > maxPartitions {
				return nil, partTooMany
			}
			for i := range t.NumElem() {
				if s := checkField(t.Elem(), base+int64(i)*t.Elem().Size()); s != partSuccess {
					return nil, s
				}
			}
		} else if t.IsString() {
			// Strings split into a pointer and a length word, mirroring
			// (*Func).SplitString. By the time this pass runs, decompose
			// builtin has already rewritten string-typed accesses into
			// word-level ones, so accesses line up with these leaves.
			ptrType := f.Config.Types.BytePtr
			lenType := types.Types[types.TINT]
			if s := checkField(ptrType, base); s != partSuccess {
				return nil, s
			}
			if s := checkField(lenType, base+ptrType.Size()); s != partSuccess {
				return nil, s
			}
		} else if t.IsSlice() {
			// Mirrors (*Func).SplitSlice.
			ptrType := types.NewPtr(t.Elem())
			lenType := types.Types[types.TINT]
			if s := checkField(ptrType, base); s != partSuccess {
				return nil, s
			}
			if s := checkField(lenType, base+ptrType.Size()); s != partSuccess {
				return nil, s
			}
			if s := checkField(lenType, base+ptrType.Size()+lenType.Size()); s != partSuccess {
				return nil, s
			}
		} else if t.IsInterface() {
			// Mirrors (*Func).SplitInterface: the itab/type word is a
			// uintptr (see the comment in typebits.Set), the data word is
			// a scannable pointer.
			itabType := types.Types[types.TUINTPTR]
			dataType := f.Config.Types.BytePtr
			if s := checkField(itabType, base); s != partSuccess {
				return nil, s
			}
			if s := checkField(dataType, base+itabType.Size()); s != partSuccess {
				return nil, s
			}
		} else if t.IsComplex() {
			// Mirrors (*Func).SplitComplex.
			elemType := types.Types[types.TFLOAT32]
			if t.Size() == 16 {
				elemType = types.Types[types.TFLOAT64]
			}
			if s := checkField(elemType, base); s != partSuccess {
				return nil, s
			}
			if s := checkField(elemType, base+elemType.Size()); s != partSuccess {
				return nil, s
			}
		} else {
			// Leaf field
			if !t.IsSIMD() && t.Size() > f.Config.RegSize {
				// This leaf field is too large to be stored in one register.
				// This can still trigger for e.g. 8-byte scalars on 32-bit
				// platforms; wide builtin composites (strings, slices,
				// interfaces, complexes) are split into words above.
				return nil, partLarge
			}
			if t.Size() == 0 {
				// The only zero-sized types are (arrays of) empty structs
				// and zero-length arrays, which are both handled by the
				// composite cases above, so a zero-sized leaf is impossible.
				f.Fatalf("zero-sized leaf field of type %v in decomposeAddr", t)
			}
			return []leafField{{base, t}}, partSuccess
		}
		return leaves, partSuccess
	}

	// 1. Collect initial candidates.
	candidates := make(map[*ir.Name]*decomposeCandidate)
	// Different ir.Name can have the same name string.
	// We need to give them different split names to distinguish them.
	stringToNames := make(map[string][]*ir.Name)
	localAddrsByName := make(map[*ir.Name][]*ssa.Value)
	seenNames := make(map[*ir.Name]bool)
	for _, v := range localAddrs {
		n := v.Aux.(*ir.Name)
		if !seenNames[n] {
			stringToNames[n.Sym().Name] = append(stringToNames[n.Sym().Name], n)
			seenNames[n] = true
		}
		localAddrsByName[n] = append(localAddrsByName[n], v)
	}

	namesOrdered := []*ir.Name{}
	for n, las := range localAddrsByName {
		namesOrdered = append(namesOrdered, n)
		slices.SortFunc(las, func(a, b *ssa.Value) int { return int(a.ID - b.ID) })
	}
	slices.SortFunc(namesOrdered, func(a, b *ir.Name) int {
		return int(localAddrsByName[a][0].ID - localAddrsByName[b][0].ID)
	})
NameLoop:
	for _, n := range namesOrdered {
		if dg, ok := demographics[n]; !ok || !dg.lszmco {
			st.Record("non qualifying variable", 1)
			continue
		}
		// Skip special variables
		if n.Class != ir.PAUTO {
			st.Record("params", 1)
			continue
		}
		t := n.Type()
		splittable := (t.IsStruct() && !t.IsSIMD()) || t.IsArray() ||
			t.IsString() || t.IsSlice() || t.IsInterface() || t.IsComplex()
		if !splittable {
			// Bare multi-word builtin composites (strings, slices,
			// interfaces, complexes) split into word leaves just like
			// aggregate fields of those types do; everything else that
			// is not a struct or array stays with mem2reg alone.
			st.Record("non composite", 1)
			continue
		}
		if n.NonMergeable() || n.OpenDeferSlot() {
			// Non mergeable variables and open defer slot are special,
			// the compiler and the runtime expects them to be in the memory.
			// Keep them in memory.
			st.Record("non mergeable or open defer slot", 1)
			continue
		}
		leaves, status := getLeafFields(n.Type(), 0)
		if status != partSuccess {
			switch status {
			case partTooMany:
				st.Record("too many partitions", 1)
			case partLarge:
				st.Record("large leaf field", 1)
			default:
				f.Fatalf("unknown leaf status in decomposeAddr")
			}
			continue
		}
		if len(leaves) == 0 {
			st.Record("no partition", 1)
			continue
		}
		leavesByOff := make(map[int64]leafField, len(leaves))
		for _, lf := range leaves {
			if _, ok := leavesByOff[lf.off]; ok {
				// Go does not have union types, is this possible?
				// Should we fatal out here instead?
				st.Record("duplicate leaf field offset", 1)
				continue NameLoop
			}
			leavesByOff[lf.off] = lf
		}

		ok := true
		for _, lf := range leaves {
			if zeroValueForType(nil, lf.typ) == nil {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}

		candidates[n] = &decomposeCandidate{n: n, leaves: leaves, localAddrs: localAddrsByName[n], leavesByOff: leavesByOff}
	}

	if len(candidates) == 0 {
		return
	}

	// Find entry memory and SP.
	var entryMem, entrySP *ssa.Value
	for _, v := range f.Entry.Values {
		if v.Op == ssaop.OpSP {
			entrySP = v
		}
		if v.Op == ssaop.OpInitMem {
			entryMem = v
		}
		if entryMem != nil && entrySP != nil {
			break
		}
	}
	if entryMem == nil || entrySP == nil {
		// A function only reaches this point with a candidate variable, whose loads,
		// stores and LocalAddrs all take a memory operand, and every memory chain
		// roots at InitMem through the entry block.
		f.Fatalf("entry mem or sp is not found but there are local variables")
	}

	// isDerivedFrom checks if v is an address that's some offset from n.
	isDerivedFrom := func(v *ssa.Value, n *ir.Name) bool {
		for {
			if v.Op == ssaop.OpLocalAddr && v.Aux == n {
				return true
			}
			if v.Op != ssaop.OpOffPtr && v.Op != ssaop.OpCopy {
				return false
			}
			v = v.Args[0]
		}
	}

	// localAddrBase returns the LocalAddr that v is derived from through
	// OffPtrs and Copies, or nil if v is not derived from one.
	localAddrBase := func(v *ssa.Value) *ssa.Value {
		for {
			switch v.Op {
			case ssaop.OpLocalAddr:
				return v
			case ssaop.OpOffPtr, ssaop.OpCopy:
				v = v.Args[0]
			default:
				return nil
			}
		}
	}

	// findOffset returns the offset of v from its LocalAddr base.
	findOffset := func(ptr *ssa.Value) int64 {
		var offset int64
		for {
			switch ptr.Op {
			case ssaop.OpLocalAddr:
				return offset
			case ssaop.OpOffPtr:
				offset += ptr.AuxInt
				ptr = ptr.Args[0]
			case ssaop.OpCopy:
				ptr = ptr.Args[0]
			default:
				// Should not happen.
				f.Fatalf("not derived from local address: %v", ptr)
				return -1
			}
		}
	}

	matchToLeaf := func(c *decomposeCandidate, v *ssa.Value, offset int64) (leafField, bool) {
		var sz int64
		if v.Op == ssaop.OpLoad {
			sz = v.Type.Size()
		} else if v.Op == ssaop.OpStore {
			sz = ssa.AuxToType(v.Aux).Size()
		} else {
			return leafField{}, false
		}
		if lf, ok := c.leavesByOff[offset]; ok {
			// Access must exactly match a leaf field's offset and size.
			if lf.typ.Size() == sz {
				// found a match
				return lf, true
			}
		}
		return leafField{}, false
	}

	// movedLeaves returns the leaves of c fully covered by the byte range
	// [off, off+size), with offsets relative to off, in increasing offset
	// order. ok is false if the range partially covers some leaf of c.
	movedLeaves := func(c *decomposeCandidate, off, size int64) (leaves []leafField, ok bool) {
		end := off + size
		for _, lf := range c.leaves {
			lend := lf.off + lf.typ.Size()
			if lf.off >= end || lend <= off {
				continue // not touched
			}
			if lf.off < off || lend > end {
				return nil, false // partially covered
			}
			leaves = append(leaves, leafField{lf.off - off, lf.typ})
		}
		return leaves, true
	}

	// 2. Check partition access alignment, and
	// 3. Breakdown OpZero and OpMove if check pass.
	// We convert aggregate-level zeros and moves into field-level stores and loads/stores.
	// This exposes the underlying scalar access patterns to the promotion logic.
	// We need to keep track of them to not process them twice, i.e. a Move could be traversed
	// twice from source and destination.
	brokenDown := make(map[*ssa.Value]bool)
	// Note: the values created while breaking down a Zero/Move are not added
	// to the uses map, so when a later candidate reaches the same (already
	// brokenDown) Zero/Move it does not see the replacement scalar accesses
	// either. That is fine for correctness: the replacements are constructed
	// leaf-aligned for both operands, and the recomputation of the uses map
	// after this loop lets the final promotion pick them up.
	// usedLeaf records the leaves that scalar Loads read, it's use to abort decomposition when it's
	// larger than maxActivePartitions.
	changed := false
	var mutations []func()
NamesLoop:
	for _, n := range namesOrdered {
		usedLeaf := make(map[int64]struct{})
		c, ok := candidates[n]
		if !ok || c.unDecomposable {
			// Already ruled out (possibly by an earlier candidate that
			// shares a Move with n): nothing to break down for it, and
			// it must not rule out further partners on its own account.
			continue
		}
		mutations = mutations[:0]
		// Partners of n's Moves whose leaves would be straddled by n's
		// breakdown. They are ruled out together with the mutations, so
		// that if n gives up later in this iteration its partners are
		// left alone; one of them may then break the Move down by its
		// own leaves instead.
		var partnerDisq []*decomposeCandidate
		// This is to keep track of the values that are ready to be processed in a
		// batched way, it will be commited to brokenDown entirely or not at all.
		localBrokenDown := make(map[*ssa.Value]bool)
		derived := []*ssa.Value{} // all the addresses that are derived from the current name.
		var collectDerived func(v *ssa.Value)
		collectDerived = func(v *ssa.Value) {
			derived = append(derived, v)
			for _, use := range uses.of(v) {
				// OffPtr and Copy can both be used to derive a new address
				// from a name. The uses table predates this loop, so a
				// Zero or Move that an earlier candidate already broke down
				// still shows up here, reset to a Copy of memory; only a
				// pointer-shaped Copy is an address.
				if use.Op == ssaop.OpOffPtr || (use.Op == ssaop.OpCopy && use.Type.IsPtrShaped()) {
					collectDerived(use)
				}
			}
		}
		for _, vAddr := range c.localAddrs {
			collectDerived(vAddr)
		}
		slices.SortFunc(derived, func(a, b *ssa.Value) int { return int(a.ID - b.ID) })

		for _, d := range derived {
			for _, use := range uses.of(d) {
				if brokenDown[use] || localBrokenDown[use] {
					continue
				}
				switch use.Op {
				case ssaop.OpOffPtr, ssaop.OpCopy:
					// Handled by derived.
				case ssaop.OpLoad, ssaop.OpStore:
					leaf, ok := matchToLeaf(c, use, findOffset(use.Args[0]))
					if !ok {
						// we can't find a leaf node that matches
						if f.Pass.Debug > 1 {
							f.Warnl(n.Pos(), "decomposeAddr: %v has a load or store not on leaf level", n)
						}
						st.Record("load or store not on leaf level", 1)
						c.unDecomposable = true
						continue NamesLoop
					}
					if use.Op == ssaop.OpLoad {
						// Only a scalar Load marks a leaf active.
						if _, ok := usedLeaf[leaf.off]; !ok {
							usedLeaf[leaf.off] = struct{}{}
							if int64(len(usedLeaf)) > maxActivePartitions {
								st.Record("too many active partitions", 1)
								c.unDecomposable = true
								continue NamesLoop
							}
						}
					}
				case ssaop.OpZero:
					off := findOffset(use.Args[0])
					size := use.AuxInt
					end := off + size

					if off < 0 || end > n.Type().Size() {
						st.Record("zero range out of bounds", 1)
						c.unDecomposable = true
						continue NamesLoop
					}

					// Check if all fields within the zero are fully covered.
					// Partial zeros (where a leaf field is only partly zeroed)
					// are not currently supported.
					for _, lf := range c.leaves {
						lend := lf.off + lf.typ.Size()
						if lf.off < end && lend > off { // touches
							if lf.off < off || lend > end {
								st.Record("partially zeroed", 1)
								c.unDecomposable = true
								continue NamesLoop
							}
						}
					}

					memIn := use.MemoryArg()
					curM := memIn
					for _, lf := range c.leaves {
						if lf.off >= off && lf.off+lf.typ.Size() <= end {
							mutations = append(mutations, func() {
								newPtr := use.Block.NewValue1I(use.Pos, ssaop.OpOffPtr, lf.typ.PtrTo(), lf.off-off, use.Args[0])
								curM = use.Block.NewValue3A(use.Pos, ssaop.OpStore, types.TypeMem, lf.typ, newPtr, zeroValueForType(f, lf.typ), curM)
							})
						}
					}
					mutations = append(mutations, func() {
						use.Reset(ssaop.OpCopy)
						use.Aux = nil
						use.SetArgs1(curM)
					})
					localBrokenDown[use] = true
				case ssaop.OpMove:
					isDst := isDerivedFrom(use.Args[0], n)
					isSrc := isDerivedFrom(use.Args[1], n)
					if !isDst && !isSrc {
						f.Fatalf("non of move's operands is derived from local address: %v", use)
					}
					// If the Move's other operand is derived from a candidate (either another
					// candidate, or n itself in a self-move), breaking this Move down by
					// n's leaves creates accesses on that operand at n-shaped offsets.
					// That is fine when the covered leaf shapes agree (the common same-type copy),
					// but via an unsafe.Pointer cast the two sides can be shaped differently,
					// and the emitted accesses would straddle the partner's or n's own leaves.
					nSide := use.Args[0]
					oSide := use.Args[1]
					if !isDst {
						nSide = use.Args[1]
						oSide = use.Args[0]
					}
					if base := localAddrBase(oSide); base != nil {
						on := base.Aux.(*ir.Name)
						if oc, ok := candidates[on]; ok && !oc.unDecomposable {
							// The breakdown is only correct for both sides if the
							// Move covers exactly the same set of leaves (same
							// relative offsets and sizes) on both of them, and
							// straddles no leaf on either side. Otherwise, e.g. if
							// one side has padding where the other has a leaf, a
							// breakdown by one side's leaves skips some leaves of
							// the other side.
							nLeaves, nOk := movedLeaves(c, findOffset(nSide), use.AuxInt)
							oLeaves, oOk := movedLeaves(oc, findOffset(oSide), use.AuxInt)
							compatible := nOk && oOk && len(nLeaves) == len(oLeaves)
							for i := 0; compatible && i < len(nLeaves); i++ {
								if nLeaves[i].off != oLeaves[i].off || nLeaves[i].typ.Size() != oLeaves[i].typ.Size() {
									compatible = false
								}
							}
							if !compatible {
								if oc == c {
									st.Record("self move with straddled leaves", 1)
									c.unDecomposable = true
									continue NamesLoop
								}
								partnerDisq = append(partnerDisq, oc)
							}
						}
					}

					moved := []int64{}
					if isDst {
						moved = append(moved, findOffset(use.Args[0]))
					}
					if isSrc {
						moved = append(moved, findOffset(use.Args[1]))
					}
					if len(moved) == 2 {
						// Overlapping moves are still possible at this point,
						// if below the threshold specified in IsInlinableMemmove.
						// Bailout them here too.
						if moved[1] < moved[0]+use.AuxInt && moved[0] < moved[1]+use.AuxInt {
							st.Record("overlapping move", 1)
							c.unDecomposable = true
							continue NamesLoop
						}
					}
					size := use.AuxInt

					// Check bounds and partial moves for all operands derived from n.
					for _, m := range moved {
						end := m + size
						if m < 0 || end > n.Type().Size() {
							st.Record("move range out of bounds", 1)
							c.unDecomposable = true
							continue NamesLoop
						}
						for _, lf := range c.leaves {
							lend := lf.off + lf.typ.Size()
							if lf.off < end && lend > m { // touches
								if lf.off < m || lend > end {
									st.Record("partially moved", 1)
									c.unDecomposable = true
									continue NamesLoop
								}
							}
						}
					}
					// The breakdown below emits a Load of every leaf the
					// Move covers, but each of those Loads is consumed at
					// once by the paired Store into the destination: after
					// promotion the value is a register briefly on its
					// way to memory, not a live leaf. So a Move does not
					// mark leaves active; only scalar Loads do (see the
					// OpLoad case). Counting them would rule out every
					// whole-aggregate copy of a variable with more leaves
					// than maxActivePartitions, which is most of the
					// candidates in practice.

					memIn := use.MemoryArg()
					curM := memIn
					for _, lf := range c.leaves {
						// Take the leader offset from the moved slice.
						offInMove := lf.off - moved[0]
						if offInMove >= 0 && offInMove+lf.typ.Size() <= size {
							mutations = append(mutations, func() {
								// Emit scalar load/store for each field.
								newSrcPtr := use.Block.NewValue1I(use.Pos, ssaop.OpOffPtr, lf.typ.PtrTo(), offInMove, use.Args[1])
								newDstPtr := use.Block.NewValue1I(use.Pos, ssaop.OpOffPtr, lf.typ.PtrTo(), offInMove, use.Args[0])
								val := use.Block.NewValue2(use.Pos, ssaop.OpLoad, lf.typ, newSrcPtr, memIn)
								curM = use.Block.NewValue3A(use.Pos, ssaop.OpStore, types.TypeMem, lf.typ, newDstPtr, val, curM)
							})
						}
					}
					mutations = append(mutations, func() {
						use.Reset(ssaop.OpCopy)
						use.Aux = nil
						use.SetArgs1(curM)
					})
					localBrokenDown[use] = true
				}
			}
		}
		for _, m := range mutations {
			m()
		}
		for _, oc := range partnerDisq {
			if oc.unDecomposable {
				continue
			}
			if f.Pass.Debug > 1 {
				f.Warnl(oc.n.Pos(), "decomposeAddr: %v has leaves straddled by a move of %v; disqualified", oc.n, n)
			}
			st.Record("move partner with straddled leaves", 1)
			oc.unDecomposable = true
		}
		for u := range localBrokenDown {
			brokenDown[u] = true
		}
		if len(mutations) > 0 {
			changed = true
		}
	}

	if changed {
		// The Zero/Move breakdown above introduced new values (OffPtrs off
		// the candidates' LocalAddrs, and scalar Loads/Stores through them)
		// that are not in the demographics/uses computed at the top of this
		// pass. Recompute them so that the final promotion below sees, and
		// rewrites, those new accesses too.
		uses.free(f)
		demographics, localAddrs, uses = variableDemographics(f)
		if uses == nil {
			// Cannot happen (the candidates' LocalAddrs still exist), but
			// keep the invariant that a nil table means nothing to do.
			return
		}
	}

	// 4. Final Promotion.
	// For each valid candidate, create new individual variables for each field
	// and update all usages.
	for _, n := range namesOrdered {
		c, ok := candidates[n]
		if !ok || c.unDecomposable {
			continue
		}
		if dg, ok := demographics[n]; !ok || !dg.lszmco {
			// The transformation should not have caused the variable
			// unknown uses.
			f.Fatalf("zero and move breakdown disqualify %v from decomposeAddr", n)
		}

		type pAccess struct {
			a   *ssa.Value
			off int64
		}
		var accesses []pAccess
		var collectSimpleUses func(v *ssa.Value, offset int64)
		collectSimpleUses = func(v *ssa.Value, offset int64) {
			for _, use := range uses.of(v) {
				switch use.Op {
				case ssaop.OpOffPtr:
					collectSimpleUses(use, offset+use.AuxInt)
				case ssaop.OpLoad, ssaop.OpStore:
					accesses = append(accesses, pAccess{use, offset})
				case ssaop.OpCopy:
					collectSimpleUses(use, offset)
				default:
					f.Fatalf("unhandled uses: %s", use.Op.String())
				}
			}
		}
		for _, vAddr := range c.localAddrs {
			collectSimpleUses(vAddr, 0)
		}
		if len(accesses) == 0 {
			st.Record("no accesses", 1)
			continue
		}

		type partition struct {
			offset  int64
			typ     *types.Type
			newAddr *ssa.Value
			newName *ir.Name
		}
		partitions := make(map[int64]*partition)
		partitionsValid := true
		for _, a := range accesses {
			if lf, ok := matchToLeaf(c, a.a, a.off); ok {
				partitions[a.off] = &partition{offset: a.off, typ: lf.typ}
				continue
			}
			// If we didn't find a matching leaf field, we cannot promote this variable.
			partitionsValid = false
			break
		}
		if !partitionsValid || len(partitions) == 0 {
			f.Fatalf("not all accesses are on leaf level")
		}

		if f.Pass.Debug > 0 {
			f.Warnl(n.Pos(), "promoted %v", n)
		}
		st.Record("promoted variable", 1)

		// Deterministically create new variables for each used partition.
		var partOffsets []int64
		for off := range partitions {
			partOffsets = append(partOffsets, off)
		}
		slices.Sort(partOffsets)

		// Different variables can have the same names, we need to distinguish them
		// if that's the case.
		nameIdx := -1
		ns := stringToNames[n.Sym().Name]
		if len(ns) > 1 {
			for i, nn := range ns {
				if nn == n {
					nameIdx = i
					break
				}
			}
		}

		for _, off := range partOffsets {
			p := partitions[off]
			// TODO: Ideally the names should be the name you would use in Go for that part,
			// to the extent that we can do that. Like myVariable.window.menu[3].text, or something.
			// Just using offsets to distinguish is fine for now.
			var newSymName string
			if nameIdx >= 0 {
				newSymName = fmt.Sprintf("%s#%d.part.%d", n.Sym().Name, nameIdx, p.offset)
			} else {
				newSymName = fmt.Sprintf("%s.part.%d", n.Sym().Name, p.offset)
			}
			newSym := &types.Sym{Name: newSymName, Pkg: n.Sym().Pkg}
			// TODO: should we use the existing stack slot for the old variable instead?
			// That might be more work.
			newName := n.Curfn.NewLocal(n.Pos(), newSym, p.typ)
			newName.SetUsed(true)
			// Parts of a compiler-generated temporary are temporaries too.
			newName.SetAutoTemp(n.AutoTemp())
			p.newName = newName
			// New LocalAddr at the entry block for the new scalar variable.
			p.newAddr = f.Entry.NewValue2A(n.Pos(), ssaop.OpLocalAddr, types.NewPtr(p.typ), newName, entrySP, entryMem)
		}

		// Rewrite loads and stores to use the new scalar addresses.
		for _, a := range accesses {
			p := partitions[a.off]
			a.a.SetArg(0, p.newAddr)
		}

		// Rewrite VarDef annotations to def the new names.
		// Note: We only emit VarDef if the new variable contains pointers
		// or is large enough to be a merge candidate, satisfying SSA checker.
		dg, ok := demographics[n]
		if !ok {
			f.Fatalf("decomposeAddr candidate variable unseen: %v", dg)
		}
		for _, v := range dg.varDefs {
			curM := v.MemoryArg()
			for _, off := range partOffsets {
				p := partitions[off]
				if p.newName.Type().HasPointers() || ssa.IsMergeCandidate(p.newName) {
					curM = v.Block.NewValue1A(v.Pos, ssaop.OpVarDef, types.TypeMem, p.newName, curM)
				}
			}
			v.Reset(ssaop.OpCopy)
			v.Aux = nil
			v.SetArgs1(curM)
		}
		changed = true
	}

	if changed {
		deadcode(f)
	}
}
