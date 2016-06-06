// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"sort"
)

const (
	cmpDepth = 4
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
	for i, e := range partition {
		if f.pass.debug > 1 && len(e) > 500 {
			fmt.Printf("CSE.large partition (%d): ", len(e))
			for j := 0; j < 3; j++ {
				fmt.Printf("%s ", e[j].LongString())
			}
			fmt.Println()
		}

		for _, v := range e {
			valueEqClass[v.ID] = ID(i)
		}
		if f.pass.debug > 2 && len(e) > 1 {
			fmt.Printf("CSE.partition #%d:", i)
			for _, v := range e {
				fmt.Printf(" %s", v.String())
			}
			fmt.Printf("\n")
		}
	}

	// Find an equivalence class where some members of the class have
	// non-equivalent arguments. Split the equivalence class appropriately.
	// Repeat until we can't find any more splits.
	for {
		changed := false

		// partition can grow in the loop. By not using a range loop here,
		// we process new additions as they arrive, avoiding O(n^2) behavior.
		for i := 0; i < len(partition); i++ {
			e := partition[i]
			v := e[0]
			// all values in this equiv class that are not equivalent to v get moved
			// into another equiv class.
			// To avoid allocating while building that equivalence class,
			// move the values equivalent to v to the beginning of e
			// and other values to the end of e.
			allvals := e
		eqloop:
			for j := 1; j < len(e); {
				w := e[j]
				equivalent := true
				for i := 0; i < len(v.Args); i++ {
					if valueEqClass[v.Args[i].ID] != valueEqClass[w.Args[i].ID] {
						equivalent = false
						break
					}
				}
				if !equivalent || v.Type.Compare(w.Type) != CMPeq {
					// w is not equivalent to v.
					// move it to the end and shrink e.
					e[j], e[len(e)-1] = e[len(e)-1], e[j]
					e = e[:len(e)-1]
					valueEqClass[w.ID] = ID(len(partition))
					changed = true
					continue eqloop
				}
				// v and w are equivalent. Keep w in e.
				j++
			}
			partition[i] = e
			if len(e) < len(allvals) {
				partition = append(partition, allvals[len(e):])
			}
		}

		if !changed {
			break
		}
	}

	// Dominator tree (f.sdom) is computed by the generic domtree pass.

	// Compute substitutions we would like to do. We substitute v for w
	// if v and w are in the same equivalence class and v dominates w.
	rewrite := make([]*Value, f.NumValues())
	for _, e := range partition {
		sort.Sort(partitionByDom{e, f.sdom})
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
				if f.sdom.isAncestorEq(v.Block, w.Block) {
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
			if cmpVal(v, w, auxIDs, cmpDepth) != CMPeq {
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

func cmpVal(v, w *Value, auxIDs auxmap, depth int) Cmp {
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

	if tc := v.Type.Compare(w.Type); tc != CMPeq {
		return tc
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

	if depth > 0 {
		for i := range v.Args {
			if v.Args[i] == w.Args[i] {
				// skip comparing equal args
				continue
			}
			if ac := cmpVal(v.Args[i], w.Args[i], auxIDs, depth-1); ac != CMPeq {
				return ac
			}
		}
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
	if cmp := cmpVal(v, w, sv.auxIDs, cmpDepth); cmp != CMPeq {
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
