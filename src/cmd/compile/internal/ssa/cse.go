// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"cmp"
	"fmt"
	"slices"
)

// cse does common-subexpression elimination on the Function.
// Values are just relinked, nothing is deleted. A subsequent deadcode
// pass is required to actually remove duplicate expressions.
func cse(f *Func) {
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
	a := f.Cache.allocValueSlice(f.NumValues())
	defer func() { f.Cache.freeValueSlice(a) }() // inside closure to use final value of a
	a = a[:0]
	o := f.Cache.allocInt32Slice(f.NumValues()) // the ordering score for stores
	defer func() { f.Cache.freeInt32Slice(o) }()
	if f.auxmap == nil {
		f.auxmap = auxmap{}
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Type.IsMemory() {
				continue // memory values can never cse
			}
			if f.auxmap[v.Aux] == 0 {
				f.auxmap[v.Aux] = int32(len(f.auxmap)) + 1
			}
			a = append(a, v)
		}
	}
	partition := partitionValues(a, f.auxmap)

	// map from value id back to eqclass id
	valueEqClass := f.Cache.allocIDSlice(f.NumValues())
	defer f.Cache.freeIDSlice(valueEqClass)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			// Use negative equivalence class #s for unique values.
			valueEqClass[v.ID] = -v.ID
		}
	}
	var pNum ID = 1
	for _, e := range partition {
		if f.pass.debug > 1 && len(e) > 500 {
			fmt.Printf("CSE.large partition (%d): ", len(e))
			for j := 0; j < 3; j++ {
				fmt.Printf("%s ", e[j].LongString())
			}
			fmt.Println()
		}

		for _, v := range e {
			valueEqClass[v.ID] = pNum
		}
		if f.pass.debug > 2 && len(e) > 1 {
			fmt.Printf("CSE.partition #%d:", pNum)
			for _, v := range e {
				fmt.Printf(" %s", v.String())
			}
			fmt.Printf("\n")
		}
		pNum++
	}

	// Split equivalence classes at points where they have
	// non-equivalent arguments.  Repeat until we can't find any
	// more splits.
	var splitPoints []int
	for {
		changed := false

		// partition can grow in the loop. By not using a range loop here,
		// we process new additions as they arrive, avoiding O(n^2) behavior.
		for i := 0; i < len(partition); i++ {
			e := partition[i]

			if opcodeTable[e[0].Op].commutative {
				// Order the first two args before comparison.
				for _, v := range e {
					if valueEqClass[v.Args[0].ID] > valueEqClass[v.Args[1].ID] {
						v.Args[0], v.Args[1] = v.Args[1], v.Args[0]
					}
				}
			}

			// Sort by eq class of arguments.
			slices.SortFunc(e, func(v, w *Value) int {
				for i, a := range v.Args {
					b := w.Args[i]
					if valueEqClass[a.ID] < valueEqClass[b.ID] {
						return -1
					}
					if valueEqClass[a.ID] > valueEqClass[b.ID] {
						return +1
					}
				}
				return 0
			})

			// Find split points.
			splitPoints = append(splitPoints[:0], 0)
			for j := 1; j < len(e); j++ {
				v, w := e[j-1], e[j]
				// Note: commutative args already correctly ordered by byArgClass.
				eqArgs := true
				for k, a := range v.Args {
					if v.Op == OpLocalAddr && k == 1 {
						continue
					}
					b := w.Args[k]
					if valueEqClass[a.ID] != valueEqClass[b.ID] {
						eqArgs = false
						break
					}
				}
				if !eqArgs {
					splitPoints = append(splitPoints, j)
				}
			}
			if len(splitPoints) == 1 {
				continue // no splits, leave equivalence class alone.
			}

			// Move another equivalence class down in place of e.
			partition[i] = partition[len(partition)-1]
			partition = partition[:len(partition)-1]
			i--

			// Add new equivalence classes for the parts of e we found.
			splitPoints = append(splitPoints, len(e))
			for j := 0; j < len(splitPoints)-1; j++ {
				f := e[splitPoints[j]:splitPoints[j+1]]
				if len(f) == 1 {
					// Don't add singletons.
					valueEqClass[f[0].ID] = -f[0].ID
					continue
				}
				for _, v := range f {
					valueEqClass[v.ID] = pNum
				}
				pNum++
				partition = append(partition, f)
			}
			changed = true
		}

		if !changed {
			break
		}
	}

	sdom := f.Sdom()

	// Compute substitutions we would like to do. We substitute v for w
	// if v and w are in the same equivalence class and v dominates w.
	rewrite := f.Cache.allocValueSlice(f.NumValues())
	defer f.Cache.freeValueSlice(rewrite)
	for _, e := range partition {
		slices.SortFunc(e, func(v, w *Value) int {
			c := cmp.Compare(sdom.domorder(v.Block), sdom.domorder(w.Block))
			if c != 0 {
				return c
			}
			if v.Op == OpLocalAddr {
				// compare the memory args for OpLocalAddrs in the same block
				vm := v.Args[1]
				wm := w.Args[1]
				if vm == wm {
					return 0
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
			return 0
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
					if w.Pos.IsStmt() == src.PosIsStmt {
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
				if v.Op == OpNilCheck {
					// nilcheck pass will remove the nil checks and log
					// them appropriately, so don't mess with them here.
					continue
				}
				b.ReplaceControl(i, x)
			}
		}
	}

	if f.pass.stats > 0 {
		f.LogStat("CSE REWRITES", rewrites)
	}
}

// storeOrdering computes the order for stores by iterate over the store
// chain, assigns a score to each store. The scores only make sense for
// stores within the same block, and the first store by store order has
// the lowest score. The cache was used to ensure only compute once.
func storeOrdering(v *Value, cache []int32) int32 {
	const minScore int32 = 1
	score := minScore
	w := v
	for {
		if s := cache[w.ID]; s >= minScore {
			score += s
			break
		}
		if w.Op == OpPhi || w.Op == OpInitMem {
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
type eqclass []*Value

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
func partitionValues(a []*Value, auxIDs auxmap) []eqclass {
	slices.SortFunc(a, func(v, w *Value) int {
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

type auxmap map[Aux]int32

func cmpVal(v, w *Value, auxIDs auxmap) types.Cmp {
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
	if v.Op == OpPhi && v.Block != w.Block {
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
	if v.Op != OpSelect0 && v.Op != OpSelect1 && v.Op != OpSelectN {
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
