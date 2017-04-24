// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"sort"
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
	a := make([]*Value, 0, f.NumValues())
	auxIDs := auxmap{}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if auxIDs[v.Aux] == 0 {
				auxIDs[v.Aux] = int32(len(auxIDs)) + 1
			}
			if v.Type.IsMemory() {
				continue // memory values can never cse
			}
			if opcodeTable[v.Op].commutative && len(v.Args) == 2 && v.Args[1].ID < v.Args[0].ID {
				// Order the arguments of binary commutative operations.
				v.Args[0], v.Args[1] = v.Args[1], v.Args[0]
			}
			a = append(a, v)
		}
	}
	partition := partitionValues(a, auxIDs)

	// map from value id back to eqclass id
	valueEqClass := make([]ID, f.NumValues())
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
	byArgClass := new(partitionByArgClass) // reuseable partitionByArgClass to reduce allocations
	for {
		changed := false

		// partition can grow in the loop. By not using a range loop here,
		// we process new additions as they arrive, avoiding O(n^2) behavior.
		for i := 0; i < len(partition); i++ {
			e := partition[i]

			// Sort by eq class of arguments.
			byArgClass.a = e
			byArgClass.eqClass = valueEqClass
			sort.Sort(byArgClass)

			// Find split points.
			splitPoints = append(splitPoints[:0], 0)
			for j := 1; j < len(e); j++ {
				v, w := e[j-1], e[j]
				eqArgs := true
				for k, a := range v.Args {
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

	sdom := f.sdom()

	// Compute substitutions we would like to do. We substitute v for w
	// if v and w are in the same equivalence class and v dominates w.
	rewrite := make([]*Value, f.NumValues())
	byDom := new(partitionByDom) // reusable partitionByDom to reduce allocs
	for _, e := range partition {
		byDom.a = e
		byDom.sdom = sdom
		sort.Sort(byDom)
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
				if sdom.isAncestorEq(v.Block, w.Block) {
					rewrite[w.ID] = v
					e[j] = nil
				} else {
					// e is sorted by domorder, so v.Block doesn't dominate any subsequent blocks in e
					break
				}
			}
		}
	}

	// if we rewrite a tuple generator to a new one in a different block,
	// copy its selectors to the new generator's block, so tuple generator
	// and selectors stay together.
	// be careful not to copy same selectors more than once (issue 16741).
	copiedSelects := make(map[ID][]*Value)
	for _, b := range f.Blocks {
	out:
		for _, v := range b.Values {
			// New values are created when selectors are copied to
			// a new block. We can safely ignore those new values,
			// since they have already been copied (issue 17918).
			if int(v.ID) >= len(rewrite) || rewrite[v.ID] != nil {
				continue
			}
			if v.Op != OpSelect0 && v.Op != OpSelect1 {
				continue
			}
			if !v.Args[0].Type.IsTuple() {
				f.Fatalf("arg of tuple selector %s is not a tuple: %s", v.String(), v.Args[0].LongString())
			}
			t := rewrite[v.Args[0].ID]
			if t != nil && t.Block != b {
				// v.Args[0] is tuple generator, CSE'd into a different block as t, v is left behind
				for _, c := range copiedSelects[t.ID] {
					if v.Op == c.Op {
						// an equivalent selector is already copied
						rewrite[v.ID] = c
						continue out
					}
				}
				c := v.copyInto(t.Block)
				rewrite[v.ID] = c
				copiedSelects[t.ID] = append(copiedSelects[t.ID], c)
			}
		}
	}

	rewrites := int64(0)

	// Apply substitutions
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, w := range v.Args {
				if x := rewrite[w.ID]; x != nil {
					v.SetArg(i, x)
					rewrites++
				}
			}
		}
		if v := b.Control; v != nil {
			if x := rewrite[v.ID]; x != nil {
				if v.Op == OpNilCheck {
					// nilcheck pass will remove the nil checks and log
					// them appropriately, so don't mess with them here.
					continue
				}
				b.SetControl(x)
			}
		}
	}
	if f.pass.stats > 0 {
		f.LogStat("CSE REWRITES", rewrites)
	}
}

// An eqclass approximates an equivalence class. During the
// algorithm it may represent the union of several of the
// final equivalence classes.
type eqclass []*Value

// partitionValues partitions the values into equivalence classes
// based on having all the following features match:
//  - opcode
//  - type
//  - auxint
//  - aux
//  - nargs
//  - block # if a phi op
//  - first two arg's opcodes and auxint
//  - NOT first two arg's aux; that can break CSE.
// partitionValues returns a list of equivalence classes, each
// being a sorted by ID list of *Values. The eqclass slices are
// backed by the same storage as the input slice.
// Equivalence classes of size 1 are ignored.
func partitionValues(a []*Value, auxIDs auxmap) []eqclass {
	sort.Sort(sortvalues{a, auxIDs})

	var partition []eqclass
	for len(a) > 0 {
		v := a[0]
		j := 1
		for ; j < len(a); j++ {
			w := a[j]
			if cmpVal(v, w, auxIDs) != CMPeq {
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
func lt2Cmp(isLt bool) Cmp {
	if isLt {
		return CMPlt
	}
	return CMPgt
}

type auxmap map[interface{}]int32

func cmpVal(v, w *Value, auxIDs auxmap) Cmp {
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
	// OpSelect is a pseudo-op. We need to be more agressive
	// regarding CSE to keep multiple OpSelect's of the same
	// argument from existing.
	if v.Op != OpSelect0 && v.Op != OpSelect1 {
		if tc := v.Type.Compare(w.Type); tc != CMPeq {
			return tc
		}
	}

	if v.Aux != w.Aux {
		if v.Aux == nil {
			return CMPlt
		}
		if w.Aux == nil {
			return CMPgt
		}
		return lt2Cmp(auxIDs[v.Aux] < auxIDs[w.Aux])
	}

	return CMPeq
}

// Sort values to make the initial partition.
type sortvalues struct {
	a      []*Value // array of values
	auxIDs auxmap   // aux -> aux ID map
}

func (sv sortvalues) Len() int      { return len(sv.a) }
func (sv sortvalues) Swap(i, j int) { sv.a[i], sv.a[j] = sv.a[j], sv.a[i] }
func (sv sortvalues) Less(i, j int) bool {
	v := sv.a[i]
	w := sv.a[j]
	if cmp := cmpVal(v, w, sv.auxIDs); cmp != CMPeq {
		return cmp == CMPlt
	}

	// Sort by value ID last to keep the sort result deterministic.
	return v.ID < w.ID
}

type partitionByDom struct {
	a    []*Value // array of values
	sdom SparseTree
}

func (sv partitionByDom) Len() int      { return len(sv.a) }
func (sv partitionByDom) Swap(i, j int) { sv.a[i], sv.a[j] = sv.a[j], sv.a[i] }
func (sv partitionByDom) Less(i, j int) bool {
	v := sv.a[i]
	w := sv.a[j]
	return sv.sdom.domorder(v.Block) < sv.sdom.domorder(w.Block)
}

type partitionByArgClass struct {
	a       []*Value // array of values
	eqClass []ID     // equivalence class IDs of values
}

func (sv partitionByArgClass) Len() int      { return len(sv.a) }
func (sv partitionByArgClass) Swap(i, j int) { sv.a[i], sv.a[j] = sv.a[j], sv.a[i] }
func (sv partitionByArgClass) Less(i, j int) bool {
	v := sv.a[i]
	w := sv.a[j]
	for i, a := range v.Args {
		b := w.Args[i]
		if sv.eqClass[a.ID] < sv.eqClass[b.ID] {
			return true
		}
		if sv.eqClass[a.ID] > sv.eqClass[b.ID] {
			return false
		}
	}
	return false
}
