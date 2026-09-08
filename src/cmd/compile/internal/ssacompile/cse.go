// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmp"
	"fmt"
	"slices"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// cse does common-subexpression elimination on the Function.
// Values are just relinked, nothing is deleted. A subsequent deadcode
// pass is required to actually remove duplicate expressions.
func cse(f *ssa.Func) {
	// Two values are equivalent if they satisfy the following definition:
	// equivalent(v, w):
	//   v.op == w.op
	//   v.type == w.type
	//   v.aux == w.aux
	//   v.auxint == w.auxint
	//   len(v.args) == len(w.args)
	//   v.block == w.block if v.op == OpPhi
	//   equivalent(v.args[i], w.args[i]) for i in 0..len(v.args)-1

	// The algorithm searches for a partition of f's values into
	// equivalence classes using the above definition.
	// It starts with a coarse partition and iteratively refines it
	// until it reaches a fixed point.

	// Make initial coarse partitions by using a subset of the conditions above.
	a := f.Cache.AllocValueSlice(f.NumValues())
	defer func() { f.Cache.FreeValueSlice(a) }() // inside closure to use final value of a
	a = a[:0]
	o := f.Cache.AllocInt32Slice(f.NumValues()) // the ordering score for stores
	defer func() { f.Cache.FreeInt32Slice(o) }()
	if f.Auxmap == nil {
		f.Auxmap = ssa.AuxMap{}
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Type.IsMemory() {
				continue // memory values can never cse
			}
			if f.Auxmap[v.Aux] == 0 {
				f.Auxmap[v.Aux] = int32(len(f.Auxmap)) + 1
			}
			a = append(a, v)
		}
	}
	partition := partitionValues(a, f.Auxmap)

	// map from value id back to eqclass id
	valueEqClass := f.Cache.AllocIDSlice(f.NumValues())
	defer f.Cache.FreeIDSlice(valueEqClass)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			// Use negative equivalence class #s for unique values.
			valueEqClass[v.ID] = -v.ID
		}
	}
	var pNum ssa.ID = 1
	for _, e := range partition {
		if f.Pass.Debug > 1 && len(e) > 500 {
			fmt.Printf("CSE.large partition (%d): ", len(e))
			for j := 0; j < 3; j++ {
				fmt.Printf("%s ", e[j].LongString())
			}
			fmt.Println()
		}

		for _, v := range e {
			valueEqClass[v.ID] = pNum
		}
		if f.Pass.Debug > 2 && len(e) > 1 {
			fmt.Printf("CSE.partition #%d:", pNum)
			for _, v := range e {
				fmt.Printf(" %s", v.String())
			}
			fmt.Printf("\n")
		}
		pNum++
	}

	// Keep a table to remap memory operand of any memory user which does not have a memory result (such as a regular load),
	// to some dominating memory operation, skipping the memory defs that do not alias with it.
	memTable := f.Cache.AllocInt32Slice(f.NumValues())
	defer f.Cache.FreeInt32Slice(memTable)

	// Split equivalence classes at points where they have
	// non-equivalent arguments.
	//
	// Every class here is uniform: its members share one tuple of argument
	// classes. A member only stops fitting once one of its own arguments moves
	// to a different class, so the values that moved drive the work and a visit
	// costs the number of values that actually changed rather than the size of
	// the class they sit in. Rescanning whole classes is what made this
	// quadratic: a chain of n dependent values splits one value per pass and
	// re-sorted the remaining n on every one of them, which is n²/2 comparisons
	// for a single function.
	//
	// Memory arguments take no part. Memory values are left out of the
	// partition above and keep their unique negative class for the whole pass,
	// so a memory argument is never the argument that moved.

	// users gives, for each value, the values that take it as an argument.
	// Nothing below changes which values a value reads, so it stays valid
	// until the substitutions are applied.
	users := uses(f)
	defer users.free(f)

	// classIdx is the partition slot holding a value's class, or -1 once the
	// value stands alone. posInClass is its index inside that slot, which is
	// what lets a value be pulled out of its class without walking it.
	classIdx := f.Cache.AllocInt32Slice(f.NumValues())
	defer f.Cache.FreeInt32Slice(classIdx)
	posInClass := f.Cache.AllocInt32Slice(f.NumValues())
	defer f.Cache.FreeInt32Slice(posInClass)
	for i := range classIdx {
		classIdx[i] = -1
	}
	for i, e := range partition {
		for j, v := range e {
			classIdx[v.ID] = int32(i)
			posInClass[v.ID] = int32(j)
		}
	}

	// keyOf writes v's argument classes into buf. Commutative operands are put
	// in class order first so that the two orders produce one key. The second
	// argument of OpLocalAddr is left out, matching the equivalence the split
	// below has always used.
	keyOf := func(buf []ssa.ID, v *ssa.Value) []ssa.ID {
		if ssaop.OpcodeTable[v.Op].Commutative {
			if valueEqClass[v.Args[0].ID] > valueEqClass[v.Args[1].ID] {
				v.Args[0], v.Args[1] = v.Args[1], v.Args[0]
			}
		}
		buf = buf[:0]
		_, idxMem, _, _ := isMemUser(v)
		for k, a := range v.Args {
			if v.Op == ssaop.OpLocalAddr && k == 1 {
				continue
			}
			if k == idxMem {
				id, _ := getEffectiveMemoryArg(memTable, v)
				buf = append(buf, valueEqClass[id])
				continue
			}
			buf = append(buf, valueEqClass[a.ID])
		}
		return buf
	}
	cmpKeys := func(x, y []ssa.ID) int {
		for i := range x {
			if x[i] != y[i] {
				if x[i] < y[i] {
					return -1
				}
				return +1
			}
		}
		return 0
	}

	// pending holds the values to look at in the next round, and queued marks
	// the ones already on it. Every value starts there.
	queued := make([]bool, f.NumValues())
	pending := make([]*ssa.Value, 0, len(a))
	for _, e := range partition {
		for _, v := range e {
			pending = append(pending, v)
			queued[v.ID] = true
		}
	}

	// markMoved records that v left its class, so everything reading v has to
	// be looked at again.
	markMoved := func(v *ssa.Value) {
		for _, u := range users.get(v) {
			if queued[u.ID] || classIdx[u.ID] < 0 {
				continue
			}
			queued[u.ID] = true
			pending = append(pending, u)
		}
	}

	// install turns g into a class of its own, or into a standalone value.
	install := func(g eqclass) {
		if len(g) == 1 {
			valueEqClass[g[0].ID] = -g[0].ID
			classIdx[g[0].ID] = -1
			return
		}
		ni := int32(len(partition))
		for j, v := range g {
			valueEqClass[v.ID] = pNum
			classIdx[v.ID] = ni
			posInClass[v.ID] = int32(j)
		}
		pNum++
		partition = append(partition, g)
	}

	var keyA, keyB []ssa.ID
	var runs []int
	var dirty []*ssa.Value
	for len(pending) > 0 {
		dirty, pending = pending, dirty[:0]
		// Group this round's changed values by the class they sit in, so each
		// class is handled once.
		slices.SortFunc(dirty, func(v, w *ssa.Value) int {
			return cmp.Compare(classIdx[v.ID], classIdx[w.ID])
		})
		for _, v := range dirty {
			queued[v.ID] = false
		}
		for s := 0; s < len(dirty); {
			ci := classIdx[dirty[s].ID]
			t := s
			for t < len(dirty) && classIdx[dirty[t].ID] == ci {
				t++
			}
			ds := dirty[s:t]
			s = t
			if ci < 0 {
				continue
			}
			e := partition[ci]
			if len(e) < 2 {
				continue
			}

			// Every member of ds leaves. In the first round ds is the whole
			// class. After that a value is queued only because one of its
			// arguments moved, and a move gives the argument a class number no
			// key has held before. A key can only pick that number up by reading
			// a value that moved with it, and every such reader was queued too,
			// so no member outside ds can match a member inside it.
			//
			// Push them to the tail of the class in place, so the class that
			// stays and the group that leaves are both contiguous and no new
			// backing array is needed.
			last := len(e)
			for _, v := range ds {
				p := posInClass[v.ID]
				last--
				if int(p) != last {
					e[p], e[last] = e[last], e[p]
					posInClass[e[p].ID] = p
					posInClass[e[last].ID] = int32(last)
				}
			}
			leaving := e[last:]
			staying := e[:last]

			// Cut the movers into runs of equal key.
			slices.SortFunc(leaving, func(v, w *ssa.Value) int {
				return cmpKeys(keyOf(keyA[:0], v), keyOf(keyB[:0], w))
			})
			runs = runs[:0]
			runs = append(runs, 0)
			keyA = keyOf(keyA, leaving[0])
			for j := 1; j < len(leaving); j++ {
				keyB = keyOf(keyB, leaving[j])
				if cmpKeys(keyA, keyB) != 0 {
					runs = append(runs, j)
					keyA = append(keyA[:0], keyB...)
				}
			}
			runs = append(runs, len(leaving))

			// One group has to inherit the class number, otherwise every member
			// counts as moved and everything reading them is queued again,
			// which is the rescan this pass is trying to avoid. The members
			// that stayed inherit it when there are any; otherwise the largest
			// group of movers does. That is the smaller-half rule.
			keep := -1
			if len(staying) >= 2 {
				partition[ci] = staying
			} else {
				if len(staying) == 1 {
					// Nothing keys a standalone value again, so put its commutative
					// operands in class order now, as every other value's are.
					keyA = keyOf(keyA, staying[0])
					valueEqClass[staying[0].ID] = -staying[0].ID
					classIdx[staying[0].ID] = -1
					markMoved(staying[0])
				}
				partition[ci] = nil
				best := 0
				for r := 1; r < len(runs)-1; r++ {
					if runs[r+1]-runs[r] > runs[best+1]-runs[best] {
						best = r
					}
				}
				if runs[best+1]-runs[best] >= 2 {
					keep = best
					g := leaving[runs[best]:runs[best+1]]
					partition[ci] = g
					for j, v := range g {
						classIdx[v.ID] = ci
						posInClass[v.ID] = int32(j)
					}
				}
			}
			for r := 0; r < len(runs)-1; r++ {
				if r == keep {
					continue
				}
				g := leaving[runs[r]:runs[r+1]]
				for _, v := range g {
					markMoved(v)
				}
				install(g)
			}
		}
	}

	// Drop the slots left behind by classes that dissolved, restoring the
	// invariant that every class in partition holds two or more values.
	partition = slices.DeleteFunc(partition, func(e eqclass) bool { return len(e) < 2 })

	sdom := f.Sdom()

	// Compute substitutions we would like to do. We substitute v for w
	// if v and w are in the same equivalence class and v dominates w.
	rewrite := f.Cache.AllocValueSlice(f.NumValues())
	defer f.Cache.FreeValueSlice(rewrite)
	for _, e := range partition {
		slices.SortFunc(e, func(v, w *ssa.Value) int {
			if c := cmp.Compare(sdom.DomOrder(v.Block), sdom.DomOrder(w.Block)); c != 0 {
				return c
			}
			if _, _, _, ok := isMemUser(v); ok {
				// Additional ordering among the memory users within one block: prefer the earliest
				// possible value among the set of equivalent values, that is the one with the lowest
				// skip count (lowest number of memory defs skipped until their common def).
				_, vSkips := getEffectiveMemoryArg(memTable, v)
				_, wSkips := getEffectiveMemoryArg(memTable, w)
				if c := cmp.Compare(vSkips, wSkips); c != 0 {
					return c
				}
			}
			if v.Op == ssaop.OpLocalAddr {
				// compare the memory args for OpLocalAddrs in the same block
				vm := v.Args[1]
				wm := w.Args[1]
				if vm == wm {
					return cmp.Compare(v.ID, w.ID)
				}
				// if the two OpLocalAddrs are in the same block, and one's memory
				// arg also in the same block, but the other one's memory arg not,
				// the latter must be in an ancestor block
				if vm.Block != v.Block {
					return -1
				}
				if wm.Block != w.Block {
					return +1
				}
				// use store order if the memory args are in the same block
				vs := storeOrdering(vm, o)
				ws := storeOrdering(wm, o)
				if vs <= 0 {
					f.Fatalf("unable to determine the order of %s", vm.LongString())
				}
				if ws <= 0 {
					f.Fatalf("unable to determine the order of %s", wm.LongString())
				}
				return cmp.Compare(vs, ws)
			}
			vStmt := v.Pos.IsStmt() == src.PosIsStmt
			wStmt := w.Pos.IsStmt() == src.PosIsStmt
			if vStmt != wStmt {
				if vStmt {
					return -1
				}
				return +1
			}
			// Sort by value ID last, so which of several equal values in a
			// block is kept does not depend on the order of the class.
			return cmp.Compare(v.ID, w.ID)
		})

		for i := 0; i < len(e)-1; i++ {
			// e is sorted by domorder, so a maximal dominant element is first in the slice
			v := e[i]
			if v == nil {
				continue
			}

			e[i] = nil
			// Replace all elements of e which v dominates
			for j := i + 1; j < len(e); j++ {
				w := e[j]
				if w == nil {
					continue
				}
				if sdom.IsAncestorEq(v.Block, w.Block) {
					rewrite[w.ID] = v
					e[j] = nil
				} else {
					// e is sorted by domorder, so v.Block doesn't dominate any subsequent blocks in e
					break
				}
			}
		}
	}

	rewrites := int64(0)

	// Apply substitutions
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, w := range v.Args {
				if x := rewrite[w.ID]; x != nil {
					if w.Pos.IsStmt() == src.PosIsStmt && w.Op != ssaop.OpNilCheck {
						// about to lose a statement marker, w
						// w is an input to v; if they're in the same block
						// and the same line, v is a good-enough new statement boundary.
						if w.Block == v.Block && w.Pos.Line() == v.Pos.Line() {
							v.Pos = v.Pos.WithIsStmt()
							w.Pos = w.Pos.WithNotStmt()
						} // TODO and if this fails?
					}
					v.SetArg(i, x)
					rewrites++
				}
			}
		}
		for i, v := range b.ControlValues() {
			if x := rewrite[v.ID]; x != nil {
				if v.Op == ssaop.OpNilCheck {
					// nilcheck pass will remove the nil checks and log
					// them appropriately, so don't mess with them here.
					continue
				}
				b.ReplaceControl(i, x)
			}
		}
	}

	if f.Pass.Stats > 0 {
		f.LogStat("CSE REWRITES", rewrites)
	}

	// Annotate HTML dumps with each memory user's effective memory arg.
	f.HTMLWriter.DebugInfo(func(v *ssa.Value) string {
		_, idxMem, _, ok := isMemUser(v)
		if !ok {
			return ""
		}
		memID, skips := getEffectiveMemoryArg(memTable, v)
		if memID == v.Args[idxMem].ID {
			return ""
		}
		return fmt.Sprintf("effmem %s (skips %d)", ssa.ValueHTML(memID), skips)
	})
}

// storeOrdering computes the order for stores by iterate over the store
// chain, assigns a score to each store. The scores only make sense for
// stores within the same block, and the first store by store order has
// the lowest score. The cache was used to ensure only compute once.
func storeOrdering(v *ssa.Value, cache []int32) int32 {
	const minScore int32 = 1
	score := minScore
	w := v
	for {
		if s := cache[w.ID]; s >= minScore {
			score += s
			break
		}
		if w.Op == ssaop.OpPhi || w.Op == ssaop.OpInitMem {
			break
		}
		a := w.MemoryArg()
		if a.Block != w.Block {
			break
		}
		w = a
		score++
	}
	w = v
	for cache[w.ID] == 0 {
		cache[w.ID] = score
		if score == minScore {
			break
		}
		w = w.MemoryArg()
		score--
	}
	return cache[v.ID]
}

// An eqclass approximates an equivalence class. During the
// algorithm it may represent the union of several of the
// final equivalence classes.
type eqclass []*ssa.Value

// partitionValues partitions the values into equivalence classes
// based on having all the following features match:
//   - opcode
//   - type
//   - auxint
//   - aux
//   - nargs
//   - block # if a phi op
//   - first two arg's opcodes and auxint
//   - NOT first two arg's aux; that can break CSE.
//
// partitionValues returns a list of equivalence classes, each
// being a sorted by ID list of *Values. The eqclass slices are
// backed by the same storage as the input slice.
// Equivalence classes of size 1 are ignored.
func partitionValues(a []*ssa.Value, auxIDs ssa.AuxMap) []eqclass {
	slices.SortFunc(a, func(v, w *ssa.Value) int {
		switch cmpVal(v, w, auxIDs) {
		case types.CMPlt:
			return -1
		case types.CMPgt:
			return +1
		default:
			// Sort by value ID last to keep the sort result deterministic.
			return cmp.Compare(v.ID, w.ID)
		}
	})

	var partition []eqclass
	for len(a) > 0 {
		v := a[0]
		j := 1
		for ; j < len(a); j++ {
			w := a[j]
			if cmpVal(v, w, auxIDs) != types.CMPeq {
				break
			}
		}
		if j > 1 {
			partition = append(partition, a[:j])
		}
		a = a[j:]
	}

	return partition
}
func lt2Cmp(isLt bool) types.Cmp {
	if isLt {
		return types.CMPlt
	}
	return types.CMPgt
}

func cmpVal(v, w *ssa.Value, auxIDs ssa.AuxMap) types.Cmp {
	// Try to order these comparison by cost (cheaper first)
	if v.Op != w.Op {
		return lt2Cmp(v.Op < w.Op)
	}
	if v.AuxInt != w.AuxInt {
		return lt2Cmp(v.AuxInt < w.AuxInt)
	}
	if len(v.Args) != len(w.Args) {
		return lt2Cmp(len(v.Args) < len(w.Args))
	}
	if v.Op == ssaop.OpPhi && v.Block != w.Block {
		return lt2Cmp(v.Block.ID < w.Block.ID)
	}
	if v.Type.IsMemory() {
		// We will never be able to CSE two values
		// that generate memory.
		return lt2Cmp(v.ID < w.ID)
	}
	// OpSelect is a pseudo-op. We need to be more aggressive
	// regarding CSE to keep multiple OpSelect's of the same
	// argument from existing.
	if v.Op != ssaop.OpSelect0 && v.Op != ssaop.OpSelect1 && v.Op != ssaop.OpSelectN {
		if tc := v.Type.Compare(w.Type); tc != types.CMPeq {
			return tc
		}
	}

	if v.Aux != w.Aux {
		if v.Aux == nil {
			return types.CMPlt
		}
		if w.Aux == nil {
			return types.CMPgt
		}
		return lt2Cmp(auxIDs[v.Aux] < auxIDs[w.Aux])
	}

	return types.CMPeq
}

// Query if the given instruction only uses "memory" argument and we may try to skip some memory "defs" if they do not alias with its address.
// Return index of pointer argument, index of "memory" argument, the access width and true on such instructions, otherwise return (-1, -1, 0, false).
func isMemUser(v *ssa.Value) (int, int, int64, bool) {
	switch v.Op {
	case ssaop.OpLoad:
		return 0, 1, v.Type.Size(), true
	case ssaop.OpNilCheck:
		return 0, 1, 0, true
	default:
		return -1, -1, 0, false
	}
}

// Query if the given "memory"-defining instruction's memory destination can be analyzed for aliasing with a memory "user" instructions.
// Return index of pointer argument, index of "memory" argument, the access width and true on such instructions, otherwise return (-1, -1, 0, false).
// If the access width is 0, the pointer index may be -1 (no pointer operand is needed).
func isMemDef(v *ssa.Value) (int, int, int64, bool) {
	switch v.Op {
	case ssaop.OpStore:
		return 0, 2, ssa.AuxToType(v.Aux).Size(), true
	case ssaop.OpVarDef:
		return -1, 0, 0, true
	case ssaop.OpZero:
		return 0, 1, v.AuxInt, true
	default:
		return -1, -1, 0, false
	}
}

// Mem table keeps memTableSkipBits lower bits to store the number of skips of "memory" operand
// and the rest to store the ID of the destination "memory"-producing instruction.
const memTableSkipBits = 8

// The maximum ID value we are able to store in the memTable, otherwise fall back to v.ID
const maxId = ssa.ID(1<<(31-memTableSkipBits)) - 1

// Return the first possibly-aliased store along the memory chain starting at v's memory argument and the number of not-aliased stores skipped.
func getEffectiveMemoryArg(memTable []int32, v *ssa.Value) (ssa.ID, uint32) {
	if code := uint32(memTable[v.ID]); code != 0 {
		return ssa.ID(code >> memTableSkipBits), code & ((1 << memTableSkipBits) - 1)
	}
	if idxPtr, idxMem, width, ok := isMemUser(v); ok {
		// TODO: We could early return some predefined value if width==0
		memId := v.Args[idxMem].ID
		if memId > maxId {
			return memId, 0
		}
		mem, skips := skipDisjointMemDefs(v, idxPtr, idxMem, width)
		if mem.ID <= maxId {
			memId = mem.ID
		} else {
			skips = 0 // avoid the skip
		}
		memTable[v.ID] = int32(memId<<memTableSkipBits) | int32(skips)
		return memId, skips
	} else {
		v.Block.Func.Fatalf("expected memory user instruction: %v", v.LongString())
	}
	return 0, 0
}

// Find a memory def that's not trivially disjoint with the user instruction, count the number
// of "skips" along the path. Return the corresponding memory def's value and the number of skips.
func skipDisjointMemDefs(user *ssa.Value, idxUserPtr, idxUserMem int, useWidth int64) (*ssa.Value, uint32) {
	usePtr, mem := user.Args[idxUserPtr], user.Args[idxUserMem]
	const maxSkips = (1 << memTableSkipBits) - 1
	var skips uint32
	for skips = 0; skips < maxSkips; skips++ {
		if idxPtr, idxMem, width, ok := isMemDef(mem); ok {
			if mem.Args[idxMem].Uses > 50 {
				// Skipping a memory def with a lot of uses may potentially increase register pressure.
				break
			}
			if width == 0 {
				mem = mem.Args[idxMem]
				continue
			}
			defPtr := mem.Args[idxPtr]
			if ssa.Disjoint1(defPtr, width, usePtr, useWidth) {
				mem = mem.Args[idxMem]
				continue
			}
		}
		break
	}
	return mem, skips
}
