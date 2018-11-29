// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Page heap.
//
// See malloc.go for the general overview.
//
// Large spans are the subject of this file. Spans consisting of less than
// _MaxMHeapLists are held in lists of like sized spans. Larger spans
// are held in a treap. See https://en.wikipedia.org/wiki/Treap or
// https://faculty.washington.edu/aragon/pubs/rst89.pdf for an overview.
// sema.go also holds an implementation of a treap.
//
// Each treapNode holds a single span. The treap is sorted by page size
// and for spans of the same size a secondary sort based on start address
// is done.
// Spans are returned based on a best fit algorithm and for spans of the same
// size the one at the lowest address is selected.
//
// The primary routines are
// insert: adds a span to the treap
// remove: removes the span from that treap that best fits the required size
// removeSpan: which removes a specific span from the treap
//
// _mheap.lock must be held when manipulating this data structure.

package runtime

import (
	"unsafe"
)

//go:notinheap
type mTreap struct {
	treap *treapNode
}

//go:notinheap
type treapNode struct {
	right     *treapNode // all treapNodes > this treap node
	left      *treapNode // all treapNodes < this treap node
	parent    *treapNode // direct parent of this node, nil if root
	npagesKey uintptr    // number of pages in spanKey, used as primary sort key
	spanKey   *mspan     // span of size npagesKey, used as secondary sort key
	priority  uint32     // random number used by treap algorithm to keep tree probabilistically balanced
}

func (t *treapNode) pred() *treapNode {
	if t.left != nil {
		// If it has a left child, its predecessor will be
		// its right most left (grand)child.
		t = t.left
		for t.right != nil {
			t = t.right
		}
		return t
	}
	// If it has no left child, its predecessor will be
	// the first grandparent who's right child is its
	// ancestor.
	//
	// We compute this by walking up the treap until the
	// current node's parent is its parent's right child.
	//
	// If we find at any point walking up the treap
	// that the current node doesn't have a parent,
	// we've hit the root. This means that t is already
	// the left-most node in the treap and therefore
	// has no predecessor.
	for t.parent != nil && t.parent.right != t {
		if t.parent.left != t {
			println("runtime: predecessor t=", t, "t.spanKey=", t.spanKey)
			throw("node is not its parent's child")
		}
		t = t.parent
	}
	return t.parent
}

func (t *treapNode) succ() *treapNode {
	if t.right != nil {
		// If it has a right child, its successor will be
		// its left-most right (grand)child.
		t = t.right
		for t.left != nil {
			t = t.left
		}
		return t
	}
	// See pred.
	for t.parent != nil && t.parent.left != t {
		if t.parent.right != t {
			println("runtime: predecessor t=", t, "t.spanKey=", t.spanKey)
			throw("node is not its parent's child")
		}
		t = t.parent
	}
	return t.parent
}

// isSpanInTreap is handy for debugging. One should hold the heap lock, usually
// mheap_.lock().
func (t *treapNode) isSpanInTreap(s *mspan) bool {
	if t == nil {
		return false
	}
	return t.spanKey == s || t.left.isSpanInTreap(s) || t.right.isSpanInTreap(s)
}

// walkTreap is handy for debugging.
// Starting at some treapnode t, for example the root, do a depth first preorder walk of
// the tree executing fn at each treap node. One should hold the heap lock, usually
// mheap_.lock().
func (t *treapNode) walkTreap(fn func(tn *treapNode)) {
	if t == nil {
		return
	}
	fn(t)
	t.left.walkTreap(fn)
	t.right.walkTreap(fn)
}

// checkTreapNode when used in conjunction with walkTreap can usually detect a
// poorly formed treap.
func checkTreapNode(t *treapNode) {
	// lessThan is used to order the treap.
	// npagesKey and npages are the primary keys.
	// spanKey and span are the secondary keys.
	// span == nil (0) will always be lessThan all
	// spans of the same size.
	lessThan := func(npages uintptr, s *mspan) bool {
		if t.npagesKey != npages {
			return t.npagesKey < npages
		}
		// t.npagesKey == npages
		return uintptr(unsafe.Pointer(t.spanKey)) < uintptr(unsafe.Pointer(s))
	}

	if t == nil {
		return
	}
	if t.spanKey.npages != t.npagesKey || t.spanKey.next != nil {
		println("runtime: checkTreapNode treapNode t=", t, "     t.npagesKey=", t.npagesKey,
			"t.spanKey.npages=", t.spanKey.npages)
		throw("why does span.npages and treap.ngagesKey do not match?")
	}
	if t.left != nil && lessThan(t.left.npagesKey, t.left.spanKey) {
		throw("t.lessThan(t.left.npagesKey, t.left.spanKey) is not false")
	}
	if t.right != nil && !lessThan(t.right.npagesKey, t.right.spanKey) {
		throw("!t.lessThan(t.left.npagesKey, t.left.spanKey) is not false")
	}
}

// insert adds span to the large span treap.
func (root *mTreap) insert(span *mspan) {
	npages := span.npages
	var last *treapNode
	pt := &root.treap
	for t := *pt; t != nil; t = *pt {
		last = t
		if t.npagesKey < npages {
			pt = &t.right
		} else if t.npagesKey > npages {
			pt = &t.left
		} else if uintptr(unsafe.Pointer(t.spanKey)) < uintptr(unsafe.Pointer(span)) {
			// t.npagesKey == npages, so sort on span addresses.
			pt = &t.right
		} else if uintptr(unsafe.Pointer(t.spanKey)) > uintptr(unsafe.Pointer(span)) {
			pt = &t.left
		} else {
			throw("inserting span already in treap")
		}
	}

	// Add t as new leaf in tree of span size and unique addrs.
	// The balanced tree is a treap using priority as the random heap priority.
	// That is, it is a binary tree ordered according to the npagesKey,
	// but then among the space of possible binary trees respecting those
	// npagesKeys, it is kept balanced on average by maintaining a heap ordering
	// on the priority: s.priority <= both s.right.priority and s.right.priority.
	// https://en.wikipedia.org/wiki/Treap
	// https://faculty.washington.edu/aragon/pubs/rst89.pdf

	t := (*treapNode)(mheap_.treapalloc.alloc())
	t.npagesKey = span.npages
	t.priority = fastrand()
	t.spanKey = span
	t.parent = last
	*pt = t // t now at a leaf.
	// Rotate up into tree according to priority.
	for t.parent != nil && t.parent.priority > t.priority {
		if t != nil && t.spanKey.npages != t.npagesKey {
			println("runtime: insert t=", t, "t.npagesKey=", t.npagesKey)
			println("runtime:      t.spanKey=", t.spanKey, "t.spanKey.npages=", t.spanKey.npages)
			throw("span and treap sizes do not match?")
		}
		if t.parent.left == t {
			root.rotateRight(t.parent)
		} else {
			if t.parent.right != t {
				throw("treap insert finds a broken treap")
			}
			root.rotateLeft(t.parent)
		}
	}
}

func (root *mTreap) removeNode(t *treapNode) {
	if t.spanKey.npages != t.npagesKey {
		throw("span and treap node npages do not match")
	}
	// Rotate t down to be leaf of tree for removal, respecting priorities.
	for t.right != nil || t.left != nil {
		if t.right == nil || t.left != nil && t.left.priority < t.right.priority {
			root.rotateRight(t)
		} else {
			root.rotateLeft(t)
		}
	}
	// Remove t, now a leaf.
	if t.parent != nil {
		if t.parent.left == t {
			t.parent.left = nil
		} else {
			t.parent.right = nil
		}
	} else {
		root.treap = nil
	}
	// Return the found treapNode's span after freeing the treapNode.
	mheap_.treapalloc.free(unsafe.Pointer(t))
}

// remove searches for, finds, removes from the treap, and returns the smallest
// span that can hold npages. If no span has at least npages return nil.
// This is slightly more complicated than a simple binary tree search
// since if an exact match is not found the next larger node is
// returned.
// If the last node inspected > npagesKey not holding
// a left node (a smaller npages) is the "best fit" node.
func (root *mTreap) remove(npages uintptr) *mspan {
	t := root.treap
	for t != nil {
		if t.spanKey == nil {
			throw("treap node with nil spanKey found")
		}
		if t.npagesKey < npages {
			t = t.right
		} else if t.left != nil && t.left.npagesKey >= npages {
			t = t.left
		} else {
			result := t.spanKey
			root.removeNode(t)
			return result
		}
	}
	return nil
}

// removeSpan searches for, finds, deletes span along with
// the associated treap node. If the span is not in the treap
// then t will eventually be set to nil and the t.spanKey
// will throw.
func (root *mTreap) removeSpan(span *mspan) {
	npages := span.npages
	t := root.treap
	for t.spanKey != span {
		if t.npagesKey < npages {
			t = t.right
		} else if t.npagesKey > npages {
			t = t.left
		} else if uintptr(unsafe.Pointer(t.spanKey)) < uintptr(unsafe.Pointer(span)) {
			t = t.right
		} else if uintptr(unsafe.Pointer(t.spanKey)) > uintptr(unsafe.Pointer(span)) {
			t = t.left
		}
	}
	root.removeNode(t)
}

// rotateLeft rotates the tree rooted at node x.
// turning (x a (y b c)) into (y (x a b) c).
func (root *mTreap) rotateLeft(x *treapNode) {
	// p -> (x a (y b c))
	p := x.parent
	a, y := x.left, x.right
	b, c := y.left, y.right

	y.left = x
	x.parent = y
	y.right = c
	if c != nil {
		c.parent = y
	}
	x.left = a
	if a != nil {
		a.parent = x
	}
	x.right = b
	if b != nil {
		b.parent = x
	}

	y.parent = p
	if p == nil {
		root.treap = y
	} else if p.left == x {
		p.left = y
	} else {
		if p.right != x {
			throw("large span treap rotateLeft")
		}
		p.right = y
	}
}

// rotateRight rotates the tree rooted at node y.
// turning (y (x a b) c) into (x a (y b c)).
func (root *mTreap) rotateRight(y *treapNode) {
	// p -> (y (x a b) c)
	p := y.parent
	x, c := y.left, y.right
	a, b := x.left, x.right

	x.left = a
	if a != nil {
		a.parent = x
	}
	x.right = y
	y.parent = x
	y.left = b
	if b != nil {
		b.parent = y
	}
	y.right = c
	if c != nil {
		c.parent = y
	}

	x.parent = p
	if p == nil {
		root.treap = x
	} else if p.left == y {
		p.left = x
	} else {
		if p.right != y {
			throw("large span treap rotateRight")
		}
		p.right = x
	}
}
