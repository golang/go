// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"os"
)

// If true, check poset integrity after every mutation
var debugPoset = false

const uintSize = 32 << (^uint(0) >> 63) // 32 or 64

// bitset is a bit array for dense indexes.
type bitset []uint

func newBitset(n int) bitset {
	return make(bitset, (n+uintSize-1)/uintSize)
}

func (bs bitset) Reset() {
	for i := range bs {
		bs[i] = 0
	}
}

func (bs bitset) Set(idx uint32) {
	bs[idx/uintSize] |= 1 << (idx % uintSize)
}

func (bs bitset) Clear(idx uint32) {
	bs[idx/uintSize] &^= 1 << (idx % uintSize)
}

func (bs bitset) Test(idx uint32) bool {
	return bs[idx/uintSize]&(1<<(idx%uintSize)) != 0
}

type undoType uint8

const (
	undoInvalid     undoType = iota
	undoCheckpoint           // a checkpoint to group undo passes
	undoSetChl               // change back left child of undo.idx to undo.edge
	undoSetChr               // change back right child of undo.idx to undo.edge
	undoNonEqual             // forget that SSA value undo.ID is non-equal to undo.idx (another ID)
	undoNewNode              // remove new node created for SSA value undo.ID
	undoNewConstant          // remove the constant node idx from the constants map
	undoAliasNode            // unalias SSA value undo.ID so that it points back to node index undo.idx
	undoNewRoot              // remove node undo.idx from root list
	undoChangeRoot           // remove node undo.idx from root list, and put back undo.edge.Target instead
	undoMergeRoot            // remove node undo.idx from root list, and put back its children instead
)

// posetUndo represents an undo pass to be performed.
// It's an union of fields that can be used to store information,
// and typ is the discriminant, that specifies which kind
// of operation must be performed. Not all fields are always used.
type posetUndo struct {
	typ  undoType
	idx  uint32
	ID   ID
	edge posetEdge
}

const (
	// Make poset handle constants as unsigned numbers.
	posetFlagUnsigned = 1 << iota
)

// A poset edge. The zero value is the null/empty edge.
// Packs target node index (31 bits) and strict flag (1 bit).
type posetEdge uint32

func newedge(t uint32, strict bool) posetEdge {
	s := uint32(0)
	if strict {
		s = 1
	}
	return posetEdge(t<<1 | s)
}
func (e posetEdge) Target() uint32 { return uint32(e) >> 1 }
func (e posetEdge) Strict() bool   { return uint32(e)&1 != 0 }
func (e posetEdge) String() string {
	s := fmt.Sprint(e.Target())
	if e.Strict() {
		s += "*"
	}
	return s
}

// posetNode is a node of a DAG within the poset.
type posetNode struct {
	l, r posetEdge
}

// poset is a union-find data structure that can represent a partially ordered set
// of SSA values. Given a binary relation that creates a partial order (eg: '<'),
// clients can record relations between SSA values using SetOrder, and later
// check relations (in the transitive closure) with Ordered. For instance,
// if SetOrder is called to record that A<B and B<C, Ordered will later confirm
// that A<C.
//
// It is possible to record equality relations between SSA values with SetEqual and check
// equality with Equal. Equality propagates into the transitive closure for the partial
// order so that if we know that A<B<C and later learn that A==D, Ordered will return
// true for D<C.
//
// It is also possible to record inequality relations between nodes with SetNonEqual;
// non-equality relations are not transitive, but they can still be useful: for instance
// if we know that A<=B and later we learn that A!=B, we can deduce that A<B.
// NonEqual can be used to check whether it is known that the nodes are different, either
// because SetNonEqual was called before, or because we know that they are strictly ordered.
//
// poset will refuse to record new relations that contradict existing relations:
// for instance if A<B<C, calling SetOrder for C<A will fail returning false; also
// calling SetEqual for C==A will fail.
//
// poset is implemented as a forest of DAGs; in each DAG, if there is a path (directed)
// from node A to B, it means that A<B (or A<=B). Equality is represented by mapping
// two SSA values to the same DAG node; when a new equality relation is recorded
// between two existing nodes,the nodes are merged, adjusting incoming and outgoing edges.
//
// Constants are specially treated. When a constant is added to the poset, it is
// immediately linked to other constants already present; so for instance if the
// poset knows that x<=3, and then x is tested against 5, 5 is first added and linked
// 3 (using 3<5), so that the poset knows that x<=3<5; at that point, it is able
// to answer x<5 correctly. This means that all constants are always within the same
// DAG; as an implementation detail, we enfoce that the DAG containtining the constants
// is always the first in the forest.
//
// poset is designed to be memory efficient and do little allocations during normal usage.
// Most internal data structures are pre-allocated and flat, so for instance adding a
// new relation does not cause any allocation. For performance reasons,
// each node has only up to two outgoing edges (like a binary tree), so intermediate
// "extra" nodes are required to represent more than two relations. For instance,
// to record that A<I, A<J, A<K (with no known relation between I,J,K), we create the
// following DAG:
//
//	  A
//	 / \
//	I  extra
//	    /  \
//	   J    K
type poset struct {
	lastidx   uint32            // last generated dense index
	flags     uint8             // internal flags
	values    map[ID]uint32     // map SSA values to dense indexes
	constants map[int64]uint32  // record SSA constants together with their value
	nodes     []posetNode       // nodes (in all DAGs)
	roots     []uint32          // list of root nodes (forest)
	noneq     map[uint32]bitset // non-equal relations
	undo      []posetUndo       // undo chain
}

func newPoset() *poset {
	return &poset{
		values:    make(map[ID]uint32),
		constants: make(map[int64]uint32, 8),
		nodes:     make([]posetNode, 1, 16),
		roots:     make([]uint32, 0, 4),
		noneq:     make(map[uint32]bitset),
		undo:      make([]posetUndo, 0, 4),
	}
}

func (po *poset) SetUnsigned(uns bool) {
	if uns {
		po.flags |= posetFlagUnsigned
	} else {
		po.flags &^= posetFlagUnsigned
	}
}

// Handle children
func (po *poset) setchl(i uint32, l posetEdge) { po.nodes[i].l = l }
func (po *poset) setchr(i uint32, r posetEdge) { po.nodes[i].r = r }
func (po *poset) chl(i uint32) uint32          { return po.nodes[i].l.Target() }
func (po *poset) chr(i uint32) uint32          { return po.nodes[i].r.Target() }
func (po *poset) children(i uint32) (posetEdge, posetEdge) {
	return po.nodes[i].l, po.nodes[i].r
}

// upush records a new undo step. It can be used for simple
// undo passes that record up to one index and one edge.
func (po *poset) upush(typ undoType, p uint32, e posetEdge) {
	po.undo = append(po.undo, posetUndo{typ: typ, idx: p, edge: e})
}

// upushnew pushes an undo pass for a new node
func (po *poset) upushnew(id ID, idx uint32) {
	po.undo = append(po.undo, posetUndo{typ: undoNewNode, ID: id, idx: idx})
}

// upushneq pushes a new undo pass for a nonequal relation
func (po *poset) upushneq(idx1 uint32, idx2 uint32) {
	po.undo = append(po.undo, posetUndo{typ: undoNonEqual, ID: ID(idx1), idx: idx2})
}

// upushalias pushes a new undo pass for aliasing two nodes
func (po *poset) upushalias(id ID, i2 uint32) {
	po.undo = append(po.undo, posetUndo{typ: undoAliasNode, ID: id, idx: i2})
}

// upushconst pushes a new undo pass for a new constant
func (po *poset) upushconst(idx uint32, old uint32) {
	po.undo = append(po.undo, posetUndo{typ: undoNewConstant, idx: idx, ID: ID(old)})
}

// addchild adds i2 as direct child of i1.
func (po *poset) addchild(i1, i2 uint32, strict bool) {
	i1l, i1r := po.children(i1)
	e2 := newedge(i2, strict)

	if i1l == 0 {
		po.setchl(i1, e2)
		po.upush(undoSetChl, i1, 0)
	} else if i1r == 0 {
		po.setchr(i1, e2)
		po.upush(undoSetChr, i1, 0)
	} else {
		// If n1 already has two children, add an intermediate extra
		// node to record the relation correctly (without relating
		// n2 to other existing nodes). Use a non-deterministic value
		// to decide whether to append on the left or the right, to avoid
		// creating degenerated chains.
		//
		//      n1
		//     /  \
		//   i1l  extra
		//        /   \
		//      i1r   n2
		//
		extra := po.newnode(nil)
		if (i1^i2)&1 != 0 { // non-deterministic
			po.setchl(extra, i1r)
			po.setchr(extra, e2)
			po.setchr(i1, newedge(extra, false))
			po.upush(undoSetChr, i1, i1r)
		} else {
			po.setchl(extra, i1l)
			po.setchr(extra, e2)
			po.setchl(i1, newedge(extra, false))
			po.upush(undoSetChl, i1, i1l)
		}
	}
}

// newnode allocates a new node bound to SSA value n.
// If n is nil, this is an extra node (= only used internally).
func (po *poset) newnode(n *Value) uint32 {
	i := po.lastidx + 1
	po.lastidx++
	po.nodes = append(po.nodes, posetNode{})
	if n != nil {
		if po.values[n.ID] != 0 {
			panic("newnode for Value already inserted")
		}
		po.values[n.ID] = i
		po.upushnew(n.ID, i)
	} else {
		po.upushnew(0, i)
	}
	return i
}

// lookup searches for a SSA value into the forest of DAGS, and return its node.
// Constants are materialized on the fly during lookup.
func (po *poset) lookup(n *Value) (uint32, bool) {
	i, f := po.values[n.ID]
	if !f && n.isGenericIntConst() {
		po.newconst(n)
		i, f = po.values[n.ID]
	}
	return i, f
}

// newconst creates a node for a constant. It links it to other constants, so
// that n<=5 is detected true when n<=3 is known to be true.
// TODO: this is O(N), fix it.
func (po *poset) newconst(n *Value) {
	if !n.isGenericIntConst() {
		panic("newconst on non-constant")
	}

	// If the same constant is already present in the poset through a different
	// Value, just alias to it without allocating a new node.
	val := n.AuxInt
	if po.flags&posetFlagUnsigned != 0 {
		val = int64(n.AuxUnsigned())
	}
	if c, found := po.constants[val]; found {
		po.values[n.ID] = c
		po.upushalias(n.ID, 0)
		return
	}

	// Create the new node for this constant
	i := po.newnode(n)

	// If this is the first constant, put it as a new root, as
	// we can't record an existing connection so we don't have
	// a specific DAG to add it to. Notice that we want all
	// constants to be in root #0, so make sure the new root
	// goes there.
	if len(po.constants) == 0 {
		idx := len(po.roots)
		po.roots = append(po.roots, i)
		po.roots[0], po.roots[idx] = po.roots[idx], po.roots[0]
		po.upush(undoNewRoot, i, 0)
		po.constants[val] = i
		po.upushconst(i, 0)
		return
	}

	// Find the lower and upper bound among existing constants. That is,
	// find the higher constant that is lower than the one that we're adding,
	// and the lower constant that is higher.
	// The loop is duplicated to handle signed and unsigned comparison,
	// depending on how the poset was configured.
	var lowerptr, higherptr uint32

	if po.flags&posetFlagUnsigned != 0 {
		var lower, higher uint64
		val1 := n.AuxUnsigned()
		for val2, ptr := range po.constants {
			val2 := uint64(val2)
			if val1 == val2 {
				panic("unreachable")
			}
			if val2 < val1 && (lowerptr == 0 || val2 > lower) {
				lower = val2
				lowerptr = ptr
			} else if val2 > val1 && (higherptr == 0 || val2 < higher) {
				higher = val2
				higherptr = ptr
			}
		}
	} else {
		var lower, higher int64
		val1 := n.AuxInt
		for val2, ptr := range po.constants {
			if val1 == val2 {
				panic("unreachable")
			}
			if val2 < val1 && (lowerptr == 0 || val2 > lower) {
				lower = val2
				lowerptr = ptr
			} else if val2 > val1 && (higherptr == 0 || val2 < higher) {
				higher = val2
				higherptr = ptr
			}
		}
	}

	if lowerptr == 0 && higherptr == 0 {
		// This should not happen, as at least one
		// other constant must exist if we get here.
		panic("no constant found")
	}

	// Create the new node and connect it to the bounds, so that
	// lower < n < higher. We could have found both bounds or only one
	// of them, depending on what other constants are present in the poset.
	// Notice that we always link constants together, so they
	// are always part of the same DAG.
	switch {
	case lowerptr != 0 && higherptr != 0:
		// Both bounds are present, record lower < n < higher.
		po.addchild(lowerptr, i, true)
		po.addchild(i, higherptr, true)

	case lowerptr != 0:
		// Lower bound only, record lower < n.
		po.addchild(lowerptr, i, true)

	case higherptr != 0:
		// Higher bound only. To record n < higher, we need
		// an extra root:
		//
		//        extra
		//        /   \
		//      root   \
		//       /      n
		//     ....    /
		//       \    /
		//       higher
		//
		i2 := higherptr
		r2 := po.findroot(i2)
		if r2 != po.roots[0] { // all constants should be in root #0
			panic("constant not in root #0")
		}
		extra := po.newnode(nil)
		po.changeroot(r2, extra)
		po.upush(undoChangeRoot, extra, newedge(r2, false))
		po.addchild(extra, r2, false)
		po.addchild(extra, i, false)
		po.addchild(i, i2, true)
	}

	po.constants[val] = i
	po.upushconst(i, 0)
}

// aliasnewnode records that a single node n2 (not in the poset yet) is an alias
// of the master node n1.
func (po *poset) aliasnewnode(n1, n2 *Value) {
	i1, i2 := po.values[n1.ID], po.values[n2.ID]
	if i1 == 0 || i2 != 0 {
		panic("aliasnewnode invalid arguments")
	}

	po.values[n2.ID] = i1
	po.upushalias(n2.ID, 0)
}

// aliasnodes records that all the nodes i2s are aliases of a single master node n1.
// aliasnodes takes care of rearranging the DAG, changing references of parent/children
// of nodes in i2s, so that they point to n1 instead.
// Complexity is O(n) (with n being the total number of nodes in the poset, not just
// the number of nodes being aliased).
func (po *poset) aliasnodes(n1 *Value, i2s bitset) {
	i1 := po.values[n1.ID]
	if i1 == 0 {
		panic("aliasnode for non-existing node")
	}
	if i2s.Test(i1) {
		panic("aliasnode i2s contains n1 node")
	}

	// Go through all the nodes to adjust parent/chidlren of nodes in i2s
	for idx, n := range po.nodes {
		// Do not touch i1 itself, otherwise we can create useless self-loops
		if uint32(idx) == i1 {
			continue
		}
		l, r := n.l, n.r

		// Rename all references to i2s into i1
		if i2s.Test(l.Target()) {
			po.setchl(uint32(idx), newedge(i1, l.Strict()))
			po.upush(undoSetChl, uint32(idx), l)
		}
		if i2s.Test(r.Target()) {
			po.setchr(uint32(idx), newedge(i1, r.Strict()))
			po.upush(undoSetChr, uint32(idx), r)
		}

		// Connect all chidren of i2s to i1 (unless those children
		// are in i2s as well, in which case it would be useless)
		if i2s.Test(uint32(idx)) {
			if l != 0 && !i2s.Test(l.Target()) {
				po.addchild(i1, l.Target(), l.Strict())
			}
			if r != 0 && !i2s.Test(r.Target()) {
				po.addchild(i1, r.Target(), r.Strict())
			}
			po.setchl(uint32(idx), 0)
			po.setchr(uint32(idx), 0)
			po.upush(undoSetChl, uint32(idx), l)
			po.upush(undoSetChr, uint32(idx), r)
		}
	}

	// Reassign all existing IDs that point to i2 to i1.
	// This includes n2.ID.
	for k, v := range po.values {
		if i2s.Test(v) {
			po.values[k] = i1
			po.upushalias(k, v)
		}
	}

	// If one of the aliased nodes is a constant, then make sure
	// po.constants is updated to point to the master node.
	for val, idx := range po.constants {
		if i2s.Test(idx) {
			po.constants[val] = i1
			po.upushconst(i1, idx)
		}
	}
}

func (po *poset) isroot(r uint32) bool {
	for i := range po.roots {
		if po.roots[i] == r {
			return true
		}
	}
	return false
}

func (po *poset) changeroot(oldr, newr uint32) {
	for i := range po.roots {
		if po.roots[i] == oldr {
			po.roots[i] = newr
			return
		}
	}
	panic("changeroot on non-root")
}

func (po *poset) removeroot(r uint32) {
	for i := range po.roots {
		if po.roots[i] == r {
			po.roots = append(po.roots[:i], po.roots[i+1:]...)
			return
		}
	}
	panic("removeroot on non-root")
}

// dfs performs a depth-first search within the DAG whose root is r.
// f is the visit function called for each node; if it returns true,
// the search is aborted and true is returned. The root node is
// visited too.
// If strict, ignore edges across a path until at least one
// strict edge is found. For instance, for a chain A<=B<=C<D<=E<F,
// a strict walk visits D,E,F.
// If the visit ends, false is returned.
func (po *poset) dfs(r uint32, strict bool, f func(i uint32) bool) bool {
	closed := newBitset(int(po.lastidx + 1))
	open := make([]uint32, 1, 64)
	open[0] = r

	if strict {
		// Do a first DFS; walk all paths and stop when we find a strict
		// edge, building a "next" list of nodes reachable through strict
		// edges. This will be the bootstrap open list for the real DFS.
		next := make([]uint32, 0, 64)

		for len(open) > 0 {
			i := open[len(open)-1]
			open = open[:len(open)-1]

			// Don't visit the same node twice. Notice that all nodes
			// across non-strict paths are still visited at least once, so
			// a non-strict path can never obscure a strict path to the
			// same node.
			if !closed.Test(i) {
				closed.Set(i)

				l, r := po.children(i)
				if l != 0 {
					if l.Strict() {
						next = append(next, l.Target())
					} else {
						open = append(open, l.Target())
					}
				}
				if r != 0 {
					if r.Strict() {
						next = append(next, r.Target())
					} else {
						open = append(open, r.Target())
					}
				}
			}
		}
		open = next
		closed.Reset()
	}

	for len(open) > 0 {
		i := open[len(open)-1]
		open = open[:len(open)-1]

		if !closed.Test(i) {
			if f(i) {
				return true
			}
			closed.Set(i)
			l, r := po.children(i)
			if l != 0 {
				open = append(open, l.Target())
			}
			if r != 0 {
				open = append(open, r.Target())
			}
		}
	}
	return false
}

// Returns true if there is a path from i1 to i2.
// If strict ==  true: if the function returns true, then i1 <  i2.
// If strict == false: if the function returns true, then i1 <= i2.
// If the function returns false, no relation is known.
func (po *poset) reaches(i1, i2 uint32, strict bool) bool {
	return po.dfs(i1, strict, func(n uint32) bool {
		return n == i2
	})
}

// findroot finds i's root, that is which DAG contains i.
// Returns the root; if i is itself a root, it is returned.
// Panic if i is not in any DAG.
func (po *poset) findroot(i uint32) uint32 {
	// TODO(rasky): if needed, a way to speed up this search is
	// storing a bitset for each root using it as a mini bloom filter
	// of nodes present under that root.
	for _, r := range po.roots {
		if po.reaches(r, i, false) {
			return r
		}
	}
	panic("findroot didn't find any root")
}

// mergeroot merges two DAGs into one DAG by creating a new extra root
func (po *poset) mergeroot(r1, r2 uint32) uint32 {
	// Root #0 is special as it contains all constants. Since mergeroot
	// discards r2 as root and keeps r1, make sure that r2 is not root #0,
	// otherwise constants would move to a different root.
	if r2 == po.roots[0] {
		r1, r2 = r2, r1
	}
	r := po.newnode(nil)
	po.setchl(r, newedge(r1, false))
	po.setchr(r, newedge(r2, false))
	po.changeroot(r1, r)
	po.removeroot(r2)
	po.upush(undoMergeRoot, r, 0)
	return r
}

// collapsepath marks n1 and n2 as equal and collapses as equal all
// nodes across all paths between n1 and n2. If a strict edge is
// found, the function does not modify the DAG and returns false.
// Complexity is O(n).
func (po *poset) collapsepath(n1, n2 *Value) bool {
	i1, i2 := po.values[n1.ID], po.values[n2.ID]
	if po.reaches(i1, i2, true) {
		return false
	}

	// Find all the paths from i1 to i2
	paths := po.findpaths(i1, i2)
	// Mark all nodes in all the paths as aliases of n1
	// (excluding n1 itself)
	paths.Clear(i1)
	po.aliasnodes(n1, paths)
	return true
}

// findpaths is a recursive function that calculates all paths from cur to dst
// and return them as a bitset (the index of a node is set in the bitset if
// that node is on at least one path from cur to dst).
// We do a DFS from cur (stopping going deep any time we reach dst, if ever),
// and mark as part of the paths any node that has a children which is already
// part of the path (or is dst itself).
func (po *poset) findpaths(cur, dst uint32) bitset {
	seen := newBitset(int(po.lastidx + 1))
	path := newBitset(int(po.lastidx + 1))
	path.Set(dst)
	po.findpaths1(cur, dst, seen, path)
	return path
}

func (po *poset) findpaths1(cur, dst uint32, seen bitset, path bitset) {
	if cur == dst {
		return
	}
	seen.Set(cur)
	l, r := po.chl(cur), po.chr(cur)
	if !seen.Test(l) {
		po.findpaths1(l, dst, seen, path)
	}
	if !seen.Test(r) {
		po.findpaths1(r, dst, seen, path)
	}
	if path.Test(l) || path.Test(r) {
		path.Set(cur)
	}
}

// Check whether it is recorded that i1!=i2
func (po *poset) isnoneq(i1, i2 uint32) bool {
	if i1 == i2 {
		return false
	}
	if i1 < i2 {
		i1, i2 = i2, i1
	}

	// Check if we recorded a non-equal relation before
	if bs, ok := po.noneq[i1]; ok && bs.Test(i2) {
		return true
	}
	return false
}

// Record that i1!=i2
func (po *poset) setnoneq(n1, n2 *Value) {
	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)

	// If any of the nodes do not exist in the poset, allocate them. Since
	// we don't know any relation (in the partial order) about them, they must
	// become independent roots.
	if !f1 {
		i1 = po.newnode(n1)
		po.roots = append(po.roots, i1)
		po.upush(undoNewRoot, i1, 0)
	}
	if !f2 {
		i2 = po.newnode(n2)
		po.roots = append(po.roots, i2)
		po.upush(undoNewRoot, i2, 0)
	}

	if i1 == i2 {
		panic("setnoneq on same node")
	}
	if i1 < i2 {
		i1, i2 = i2, i1
	}
	bs := po.noneq[i1]
	if bs == nil {
		// Given that we record non-equality relations using the
		// higher index as a key, the bitsize will never change size.
		// TODO(rasky): if memory is a problem, consider allocating
		// a small bitset and lazily grow it when higher indices arrive.
		bs = newBitset(int(i1))
		po.noneq[i1] = bs
	} else if bs.Test(i2) {
		// Already recorded
		return
	}
	bs.Set(i2)
	po.upushneq(i1, i2)
}

// CheckIntegrity verifies internal integrity of a poset. It is intended
// for debugging purposes.
func (po *poset) CheckIntegrity() {
	// Record which index is a constant
	constants := newBitset(int(po.lastidx + 1))
	for _, c := range po.constants {
		constants.Set(c)
	}

	// Verify that each node appears in a single DAG, and that
	// all constants are within the first DAG
	seen := newBitset(int(po.lastidx + 1))
	for ridx, r := range po.roots {
		if r == 0 {
			panic("empty root")
		}

		po.dfs(r, false, func(i uint32) bool {
			if seen.Test(i) {
				panic("duplicate node")
			}
			seen.Set(i)
			if constants.Test(i) {
				if ridx != 0 {
					panic("constants not in the first DAG")
				}
			}
			return false
		})
	}

	// Verify that values contain the minimum set
	for id, idx := range po.values {
		if !seen.Test(idx) {
			panic(fmt.Errorf("spurious value [%d]=%d", id, idx))
		}
	}

	// Verify that only existing nodes have non-zero children
	for i, n := range po.nodes {
		if n.l|n.r != 0 {
			if !seen.Test(uint32(i)) {
				panic(fmt.Errorf("children of unknown node %d->%v", i, n))
			}
			if n.l.Target() == uint32(i) || n.r.Target() == uint32(i) {
				panic(fmt.Errorf("self-loop on node %d", i))
			}
		}
	}
}

// CheckEmpty checks that a poset is completely empty.
// It can be used for debugging purposes, as a poset is supposed to
// be empty after it's fully rolled back through Undo.
func (po *poset) CheckEmpty() error {
	if len(po.nodes) != 1 {
		return fmt.Errorf("non-empty nodes list: %v", po.nodes)
	}
	if len(po.values) != 0 {
		return fmt.Errorf("non-empty value map: %v", po.values)
	}
	if len(po.roots) != 0 {
		return fmt.Errorf("non-empty root list: %v", po.roots)
	}
	if len(po.constants) != 0 {
		return fmt.Errorf("non-empty constants: %v", po.constants)
	}
	if len(po.undo) != 0 {
		return fmt.Errorf("non-empty undo list: %v", po.undo)
	}
	if po.lastidx != 0 {
		return fmt.Errorf("lastidx index is not zero: %v", po.lastidx)
	}
	for _, bs := range po.noneq {
		for _, x := range bs {
			if x != 0 {
				return fmt.Errorf("non-empty noneq map")
			}
		}
	}
	return nil
}

// DotDump dumps the poset in graphviz format to file fn, with the specified title.
func (po *poset) DotDump(fn string, title string) error {
	f, err := os.Create(fn)
	if err != nil {
		return err
	}
	defer f.Close()

	// Create reverse index mapping (taking aliases into account)
	names := make(map[uint32]string)
	for id, i := range po.values {
		s := names[i]
		if s == "" {
			s = fmt.Sprintf("v%d", id)
		} else {
			s += fmt.Sprintf(", v%d", id)
		}
		names[i] = s
	}

	// Create reverse constant mapping
	consts := make(map[uint32]int64)
	for val, idx := range po.constants {
		consts[idx] = val
	}

	fmt.Fprintf(f, "digraph poset {\n")
	fmt.Fprintf(f, "\tedge [ fontsize=10 ]\n")
	for ridx, r := range po.roots {
		fmt.Fprintf(f, "\tsubgraph root%d {\n", ridx)
		po.dfs(r, false, func(i uint32) bool {
			if val, ok := consts[i]; ok {
				// Constant
				var vals string
				if po.flags&posetFlagUnsigned != 0 {
					vals = fmt.Sprint(uint64(val))
				} else {
					vals = fmt.Sprint(int64(val))
				}
				fmt.Fprintf(f, "\t\tnode%d [shape=box style=filled fillcolor=cadetblue1 label=<%s <font point-size=\"6\">%s [%d]</font>>]\n",
					i, vals, names[i], i)
			} else {
				// Normal SSA value
				fmt.Fprintf(f, "\t\tnode%d [label=<%s <font point-size=\"6\">[%d]</font>>]\n", i, names[i], i)
			}
			chl, chr := po.children(i)
			for _, ch := range []posetEdge{chl, chr} {
				if ch != 0 {
					if ch.Strict() {
						fmt.Fprintf(f, "\t\tnode%d -> node%d [label=\" <\" color=\"red\"]\n", i, ch.Target())
					} else {
						fmt.Fprintf(f, "\t\tnode%d -> node%d [label=\" <=\" color=\"green\"]\n", i, ch.Target())
					}
				}
			}
			return false
		})
		fmt.Fprintf(f, "\t}\n")
	}
	fmt.Fprintf(f, "\tlabelloc=\"t\"\n")
	fmt.Fprintf(f, "\tlabeldistance=\"3.0\"\n")
	fmt.Fprintf(f, "\tlabel=%q\n", title)
	fmt.Fprintf(f, "}\n")
	return nil
}

// Ordered reports whether n1<n2. It returns false either when it is
// certain that n1<n2 is false, or if there is not enough information
// to tell.
// Complexity is O(n).
func (po *poset) Ordered(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call Ordered with n1==n2")
	}

	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)
	if !f1 || !f2 {
		return false
	}

	return i1 != i2 && po.reaches(i1, i2, true)
}

// OrderedOrEqual reports whether n1<=n2. It returns false either when it is
// certain that n1<=n2 is false, or if there is not enough information
// to tell.
// Complexity is O(n).
func (po *poset) OrderedOrEqual(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call Ordered with n1==n2")
	}

	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)
	if !f1 || !f2 {
		return false
	}

	return i1 == i2 || po.reaches(i1, i2, false)
}

// Equal reports whether n1==n2. It returns false either when it is
// certain that n1==n2 is false, or if there is not enough information
// to tell.
// Complexity is O(1).
func (po *poset) Equal(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call Equal with n1==n2")
	}

	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)
	return f1 && f2 && i1 == i2
}

// NonEqual reports whether n1!=n2. It returns false either when it is
// certain that n1!=n2 is false, or if there is not enough information
// to tell.
// Complexity is O(n) (because it internally calls Ordered to see if we
// can infer n1!=n2 from n1<n2 or n2<n1).
func (po *poset) NonEqual(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call NonEqual with n1==n2")
	}

	// If we never saw the nodes before, we don't
	// have a recorded non-equality.
	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)
	if !f1 || !f2 {
		return false
	}

	// Check if we recored inequality
	if po.isnoneq(i1, i2) {
		return true
	}

	// Check if n1<n2 or n2<n1, in which case we can infer that n1!=n2
	if po.Ordered(n1, n2) || po.Ordered(n2, n1) {
		return true
	}

	return false
}

// setOrder records that n1<n2 or n1<=n2 (depending on strict). Returns false
// if this is a contradiction.
// Implements SetOrder() and SetOrderOrEqual()
func (po *poset) setOrder(n1, n2 *Value, strict bool) bool {
	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)

	switch {
	case !f1 && !f2:
		// Neither n1 nor n2 are in the poset, so they are not related
		// in any way to existing nodes.
		// Create a new DAG to record the relation.
		i1, i2 = po.newnode(n1), po.newnode(n2)
		po.roots = append(po.roots, i1)
		po.upush(undoNewRoot, i1, 0)
		po.addchild(i1, i2, strict)

	case f1 && !f2:
		// n1 is in one of the DAGs, while n2 is not. Add n2 as children
		// of n1.
		i2 = po.newnode(n2)
		po.addchild(i1, i2, strict)

	case !f1 && f2:
		// n1 is not in any DAG but n2 is. If n2 is a root, we can put
		// n1 in its place as a root; otherwise, we need to create a new
		// extra root to record the relation.
		i1 = po.newnode(n1)

		if po.isroot(i2) {
			po.changeroot(i2, i1)
			po.upush(undoChangeRoot, i1, newedge(i2, strict))
			po.addchild(i1, i2, strict)
			return true
		}

		// Search for i2's root; this requires a O(n) search on all
		// DAGs
		r := po.findroot(i2)

		// Re-parent as follows:
		//
		//                  extra
		//     r            /   \
		//      \   ===>   r    i1
		//      i2          \   /
		//                    i2
		//
		extra := po.newnode(nil)
		po.changeroot(r, extra)
		po.upush(undoChangeRoot, extra, newedge(r, false))
		po.addchild(extra, r, false)
		po.addchild(extra, i1, false)
		po.addchild(i1, i2, strict)

	case f1 && f2:
		// If the nodes are aliased, fail only if we're setting a strict order
		// (that is, we cannot set n1<n2 if n1==n2).
		if i1 == i2 {
			return !strict
		}

		// If we are trying to record n1<=n2 but we learned that n1!=n2,
		// record n1<n2, as it provides more information.
		if !strict && po.isnoneq(i1, i2) {
			strict = true
		}

		// Both n1 and n2 are in the poset. This is the complex part of the algorithm
		// as we need to find many different cases and DAG shapes.

		// Check if n1 somehow reaches n2
		if po.reaches(i1, i2, false) {
			// This is the table of all cases we need to handle:
			//
			//      DAG          New      Action
			//      ---------------------------------------------------
			// #1:  N1<=X<=N2 |  N1<=N2 | do nothing
			// #2:  N1<=X<=N2 |  N1<N2  | add strict edge (N1<N2)
			// #3:  N1<X<N2   |  N1<=N2 | do nothing (we already know more)
			// #4:  N1<X<N2   |  N1<N2  | do nothing

			// Check if we're in case #2
			if strict && !po.reaches(i1, i2, true) {
				po.addchild(i1, i2, true)
				return true
			}

			// Case #1, #3 o #4: nothing to do
			return true
		}

		// Check if n2 somehow reaches n1
		if po.reaches(i2, i1, false) {
			// This is the table of all cases we need to handle:
			//
			//      DAG           New      Action
			//      ---------------------------------------------------
			// #5:  N2<=X<=N1  |  N1<=N2 | collapse path (learn that N1=X=N2)
			// #6:  N2<=X<=N1  |  N1<N2  | contradiction
			// #7:  N2<X<N1    |  N1<=N2 | contradiction in the path
			// #8:  N2<X<N1    |  N1<N2  | contradiction

			if strict {
				// Cases #6 and #8: contradiction
				return false
			}

			// We're in case #5 or #7. Try to collapse path, and that will
			// fail if it realizes that we are in case #7.
			return po.collapsepath(n2, n1)
		}

		// We don't know of any existing relation between n1 and n2. They could
		// be part of the same DAG or not.
		// Find their roots to check whether they are in the same DAG.
		r1, r2 := po.findroot(i1), po.findroot(i2)
		if r1 != r2 {
			// We need to merge the two DAGs to record a relation between the nodes
			po.mergeroot(r1, r2)
		}

		// Connect n1 and n2
		po.addchild(i1, i2, strict)
	}

	return true
}

// SetOrder records that n1<n2. Returns false if this is a contradiction
// Complexity is O(1) if n2 was never seen before, or O(n) otherwise.
func (po *poset) SetOrder(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call SetOrder with n1==n2")
	}
	return po.setOrder(n1, n2, true)
}

// SetOrderOrEqual records that n1<=n2. Returns false if this is a contradiction
// Complexity is O(1) if n2 was never seen before, or O(n) otherwise.
func (po *poset) SetOrderOrEqual(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call SetOrder with n1==n2")
	}
	return po.setOrder(n1, n2, false)
}

// SetEqual records that n1==n2. Returns false if this is a contradiction
// (that is, if it is already recorded that n1<n2 or n2<n1).
// Complexity is O(1) if n2 was never seen before, or O(n) otherwise.
func (po *poset) SetEqual(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call Add with n1==n2")
	}

	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)

	switch {
	case !f1 && !f2:
		i1 = po.newnode(n1)
		po.roots = append(po.roots, i1)
		po.upush(undoNewRoot, i1, 0)
		po.aliasnewnode(n1, n2)
	case f1 && !f2:
		po.aliasnewnode(n1, n2)
	case !f1 && f2:
		po.aliasnewnode(n2, n1)
	case f1 && f2:
		if i1 == i2 {
			// Already aliased, ignore
			return true
		}

		// If we recorded that n1!=n2, this is a contradiction.
		if po.isnoneq(i1, i2) {
			return false
		}

		// If we already knew that n1<=n2, we can collapse the path to
		// record n1==n2 (and viceversa).
		if po.reaches(i1, i2, false) {
			return po.collapsepath(n1, n2)
		}
		if po.reaches(i2, i1, false) {
			return po.collapsepath(n2, n1)
		}

		r1 := po.findroot(i1)
		r2 := po.findroot(i2)
		if r1 != r2 {
			// Merge the two DAGs so we can record relations between the nodes
			po.mergeroot(r1, r2)
		}

		// Set n2 as alias of n1. This will also update all the references
		// to n2 to become references to n1
		i2s := newBitset(int(po.lastidx) + 1)
		i2s.Set(i2)
		po.aliasnodes(n1, i2s)
	}
	return true
}

// SetNonEqual records that n1!=n2. Returns false if this is a contradiction
// (that is, if it is already recorded that n1==n2).
// Complexity is O(n).
func (po *poset) SetNonEqual(n1, n2 *Value) bool {
	if debugPoset {
		defer po.CheckIntegrity()
	}
	if n1.ID == n2.ID {
		panic("should not call SetNonEqual with n1==n2")
	}

	// Check whether the nodes are already in the poset
	i1, f1 := po.lookup(n1)
	i2, f2 := po.lookup(n2)

	// If either node wasn't present, we just record the new relation
	// and exit.
	if !f1 || !f2 {
		po.setnoneq(n1, n2)
		return true
	}

	// See if we already know this, in which case there's nothing to do.
	if po.isnoneq(i1, i2) {
		return true
	}

	// Check if we're contradicting an existing equality relation
	if po.Equal(n1, n2) {
		return false
	}

	// Record non-equality
	po.setnoneq(n1, n2)

	// If we know that i1<=i2 but not i1<i2, learn that as we
	// now know that they are not equal. Do the same for i2<=i1.
	// Do this check only if both nodes were already in the DAG,
	// otherwise there cannot be an existing relation.
	if po.reaches(i1, i2, false) && !po.reaches(i1, i2, true) {
		po.addchild(i1, i2, true)
	}
	if po.reaches(i2, i1, false) && !po.reaches(i2, i1, true) {
		po.addchild(i2, i1, true)
	}

	return true
}

// Checkpoint saves the current state of the DAG so that it's possible
// to later undo this state.
// Complexity is O(1).
func (po *poset) Checkpoint() {
	po.undo = append(po.undo, posetUndo{typ: undoCheckpoint})
}

// Undo restores the state of the poset to the previous checkpoint.
// Complexity depends on the type of operations that were performed
// since the last checkpoint; each Set* operation creates an undo
// pass which Undo has to revert with a worst-case complexity of O(n).
func (po *poset) Undo() {
	if len(po.undo) == 0 {
		panic("empty undo stack")
	}
	if debugPoset {
		defer po.CheckIntegrity()
	}

	for len(po.undo) > 0 {
		pass := po.undo[len(po.undo)-1]
		po.undo = po.undo[:len(po.undo)-1]

		switch pass.typ {
		case undoCheckpoint:
			return

		case undoSetChl:
			po.setchl(pass.idx, pass.edge)

		case undoSetChr:
			po.setchr(pass.idx, pass.edge)

		case undoNonEqual:
			po.noneq[uint32(pass.ID)].Clear(pass.idx)

		case undoNewNode:
			if pass.idx != po.lastidx {
				panic("invalid newnode index")
			}
			if pass.ID != 0 {
				if po.values[pass.ID] != pass.idx {
					panic("invalid newnode undo pass")
				}
				delete(po.values, pass.ID)
			}
			po.setchl(pass.idx, 0)
			po.setchr(pass.idx, 0)
			po.nodes = po.nodes[:pass.idx]
			po.lastidx--

		case undoNewConstant:
			// FIXME: remove this O(n) loop
			var val int64
			var i uint32
			for val, i = range po.constants {
				if i == pass.idx {
					break
				}
			}
			if i != pass.idx {
				panic("constant not found in undo pass")
			}
			if pass.ID == 0 {
				delete(po.constants, val)
			} else {
				// Restore previous index as constant node
				// (also restoring the invariant on correct bounds)
				oldidx := uint32(pass.ID)
				po.constants[val] = oldidx
			}

		case undoAliasNode:
			ID, prev := pass.ID, pass.idx
			cur := po.values[ID]
			if prev == 0 {
				// Born as an alias, die as an alias
				delete(po.values, ID)
			} else {
				if cur == prev {
					panic("invalid aliasnode undo pass")
				}
				// Give it back previous value
				po.values[ID] = prev
			}

		case undoNewRoot:
			i := pass.idx
			l, r := po.children(i)
			if l|r != 0 {
				panic("non-empty root in undo newroot")
			}
			po.removeroot(i)

		case undoChangeRoot:
			i := pass.idx
			l, r := po.children(i)
			if l|r != 0 {
				panic("non-empty root in undo changeroot")
			}
			po.changeroot(i, pass.edge.Target())

		case undoMergeRoot:
			i := pass.idx
			l, r := po.children(i)
			po.changeroot(i, l.Target())
			po.roots = append(po.roots, r.Target())

		default:
			panic(pass.typ)
		}
	}

	if debugPoset && po.CheckEmpty() != nil {
		panic("poset not empty at the end of undo")
	}
}
