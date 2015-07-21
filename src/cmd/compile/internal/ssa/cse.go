// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "sort"

// cse does common-subexpression elimination on the Function.
// Values are just relinked, nothing is deleted.  A subsequent deadcode
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

	// Make initial partition based on opcode/type-name/aux/auxint/nargs/phi-block
	type key struct {
		op     Op
		typ    string
		aux    interface{}
		auxint int64
		nargs  int
		block  ID // block id for phi vars, -1 otherwise
	}
	m := map[key]eqclass{}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			bid := ID(-1)
			if v.Op == OpPhi {
				bid = b.ID
			}
			k := key{v.Op, v.Type.String(), v.Aux, v.AuxInt, len(v.Args), bid}
			m[k] = append(m[k], v)
		}
	}

	// A partition is a set of disjoint eqclasses.
	var partition []eqclass
	for _, v := range m {
		partition = append(partition, v)
	}
	// TODO: Sort partition here for perfect reproducibility?
	// Sort by what? Partition size?
	// (Could that improve efficiency by discovering splits earlier?)

	// map from value id back to eqclass id
	valueEqClass := make([]int, f.NumValues())
	for i, e := range partition {
		for _, v := range e {
			valueEqClass[v.ID] = i
		}
	}

	// Find an equivalence class where some members of the class have
	// non-equivalent arguments.  Split the equivalence class appropriately.
	// Repeat until we can't find any more splits.
	for {
		changed := false

		for i, e := range partition {
			v := e[0]
			// all values in this equiv class that are not equivalent to v get moved
			// into another equiv class q.
			var q eqclass
		eqloop:
			for j := 1; j < len(e); {
				w := e[j]
				for i := 0; i < len(v.Args); i++ {
					if valueEqClass[v.Args[i].ID] != valueEqClass[w.Args[i].ID] || !v.Type.Equal(w.Type) {
						// w is not equivalent to v.
						// remove w from e
						e, e[j] = e[:len(e)-1], e[len(e)-1]
						// add w to q
						q = append(q, w)
						valueEqClass[w.ID] = len(partition)
						changed = true
						continue eqloop
					}
				}
				// v and w are equivalent.  Keep w in e.
				j++
			}
			partition[i] = e
			if q != nil {
				partition = append(partition, q)
			}
		}

		if !changed {
			break
		}
	}

	// Compute dominator tree
	idom := dominators(f)

	// Compute substitutions we would like to do.  We substitute v for w
	// if v and w are in the same equivalence class and v dominates w.
	rewrite := make([]*Value, f.NumValues())
	for _, e := range partition {
		sort.Sort(e) // ensure deterministic ordering
		for len(e) > 1 {
			// Find a maximal dominant element in e
			v := e[0]
			for _, w := range e[1:] {
				if dom(w.Block, v.Block, idom) {
					v = w
				}
			}

			// Replace all elements of e which v dominates
			for i := 0; i < len(e); {
				w := e[i]
				if w == v {
					e, e[i] = e[:len(e)-1], e[len(e)-1]
				} else if dom(v.Block, w.Block, idom) {
					rewrite[w.ID] = v
					e, e[i] = e[:len(e)-1], e[len(e)-1]
				} else {
					i++
				}
			}
			// TODO(khr): if value is a control value, do we need to keep it block-local?
		}
	}

	// Apply substitutions
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, w := range v.Args {
				if x := rewrite[w.ID]; x != nil {
					v.SetArg(i, x)
				}
			}
		}
	}
}

// returns true if b dominates c.
// TODO(khr): faster
func dom(b, c *Block, idom []*Block) bool {
	// Walk up from c in the dominator tree looking for b.
	for c != nil {
		if c == b {
			return true
		}
		c = idom[c.ID]
	}
	// Reached the entry block, never saw b.
	return false
}

// An eqclass approximates an equivalence class.  During the
// algorithm it may represent the union of several of the
// final equivalence classes.
type eqclass []*Value

// Sort an equivalence class by value ID.
func (e eqclass) Len() int           { return len(e) }
func (e eqclass) Swap(i, j int)      { e[i], e[j] = e[j], e[i] }
func (e eqclass) Less(i, j int) bool { return e[i].ID < e[j].ID }
