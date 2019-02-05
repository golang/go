// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Page heap.
//
// See malloc.go for the general overview.
//
// Allocation policy is the subject of this file. All free spans live in
// a treap for most of their time being free. See
// https://en.wikipedia.org/wiki/Treap or
// https://faculty.washington.edu/aragon/pubs/rst89.pdf for an overview.
// sema.go also holds an implementation of a treap.
//
// Each treapNode holds a single span. The treap is sorted by base address
// and each span necessarily has a unique base address.
// Spans are returned based on a first-fit algorithm, acquiring the span
// with the lowest base address which still satisfies the request.
//
// The first-fit algorithm is possible due to an augmentation of each
// treapNode to maintain the size of the largest span in the subtree rooted
// at that treapNode. Below we refer to this invariant as the maxPages
// invariant.
//
// The primary routines are
// insert: adds a span to the treap
// remove: removes the span from that treap that best fits the required size
// removeSpan: which removes a specific span from the treap
//
// mheap_.lock must be held when manipulating this data structure.

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
	right    *treapNode // all treapNodes > this treap node
	left     *treapNode // all treapNodes < this treap node
	parent   *treapNode // direct parent of this node, nil if root
	key      uintptr    // base address of the span, used as primary sort key
	span     *mspan     // span at base address key
	maxPages uintptr    // the maximum size of any span in this subtree, including the root
	priority uint32     // random number used by treap algorithm to keep tree probabilistically balanced
}

// recomputeMaxPages is a helper method which has a node
// recompute its own maxPages value by looking at its own
// span's length as well as the maxPages value of its
// direct children.
func (t *treapNode) recomputeMaxPages() {
	t.maxPages = t.span.npages
	if t.left != nil && t.maxPages < t.left.maxPages {
		t.maxPages = t.left.maxPages
	}
	if t.right != nil && t.maxPages < t.right.maxPages {
		t.maxPages = t.right.maxPages
	}
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
			println("runtime: predecessor t=", t, "t.span=", t.span)
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
			println("runtime: predecessor t=", t, "t.span=", t.span)
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
	return t.span == s || t.left.isSpanInTreap(s) || t.right.isSpanInTreap(s)
}

// walkTreap is handy for debugging and testing.
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
	if t == nil {
		return
	}
	if t.span.next != nil || t.span.prev != nil || t.span.list != nil {
		throw("span may be on an mSpanList while simultaneously in the treap")
	}
	if t.span.base() != t.key {
		println("runtime: checkTreapNode treapNode t=", t, "     t.key=", t.key,
			"t.span.base()=", t.span.base())
		throw("why does span.base() and treap.key do not match?")
	}
	if t.left != nil && t.key < t.left.key {
		throw("found out-of-order spans in treap (left child has greater base address)")
	}
	if t.right != nil && t.key > t.right.key {
		throw("found out-of-order spans in treap (right child has lesser base address)")
	}
}

// validateMaxPages is handy for debugging and testing.
// It ensures that the maxPages field is appropriately maintained throughout
// the treap by walking the treap in a post-order manner.
func (t *treapNode) validateMaxPages() uintptr {
	if t == nil {
		return 0
	}
	leftMax := t.left.validateMaxPages()
	rightMax := t.right.validateMaxPages()
	max := t.span.npages
	if leftMax > max {
		max = leftMax
	}
	if rightMax > max {
		max = rightMax
	}
	if max != t.maxPages {
		println("runtime: t.maxPages=", t.maxPages, "want=", max)
		throw("maxPages invariant violated in treap")
	}
	return max
}

// treapIter is a bidirectional iterator type which may be used to iterate over a
// an mTreap in-order forwards (increasing order) or backwards (decreasing order).
// Its purpose is to hide details about the treap from users when trying to iterate
// over it.
//
// To create iterators over the treap, call start or end on an mTreap.
type treapIter struct {
	t *treapNode
}

// span returns the span at the current position in the treap.
// If the treap is not valid, span will panic.
func (i *treapIter) span() *mspan {
	return i.t.span
}

// valid returns whether the iterator represents a valid position
// in the mTreap.
func (i *treapIter) valid() bool {
	return i.t != nil
}

// next moves the iterator forward by one. Once the iterator
// ceases to be valid, calling next will panic.
func (i treapIter) next() treapIter {
	i.t = i.t.succ()
	return i
}

// prev moves the iterator backwards by one. Once the iterator
// ceases to be valid, calling prev will panic.
func (i treapIter) prev() treapIter {
	i.t = i.t.pred()
	return i
}

// start returns an iterator which points to the start of the treap (the
// left-most node in the treap).
func (root *mTreap) start() treapIter {
	t := root.treap
	if t == nil {
		return treapIter{}
	}
	for t.left != nil {
		t = t.left
	}
	return treapIter{t: t}
}

// end returns an iterator which points to the end of the treap (the
// right-most node in the treap).
func (root *mTreap) end() treapIter {
	t := root.treap
	if t == nil {
		return treapIter{}
	}
	for t.right != nil {
		t = t.right
	}
	return treapIter{t: t}
}

// insert adds span to the large span treap.
func (root *mTreap) insert(span *mspan) {
	base := span.base()
	var last *treapNode
	pt := &root.treap
	for t := *pt; t != nil; t = *pt {
		last = t
		if t.key < base {
			pt = &t.right
		} else if t.key > base {
			pt = &t.left
		} else {
			throw("inserting span already in treap")
		}
	}

	// Add t as new leaf in tree of span size and unique addrs.
	// The balanced tree is a treap using priority as the random heap priority.
	// That is, it is a binary tree ordered according to the key,
	// but then among the space of possible binary trees respecting those
	// keys, it is kept balanced on average by maintaining a heap ordering
	// on the priority: s.priority <= both s.right.priority and s.right.priority.
	// https://en.wikipedia.org/wiki/Treap
	// https://faculty.washington.edu/aragon/pubs/rst89.pdf

	t := (*treapNode)(mheap_.treapalloc.alloc())
	t.key = span.base()
	t.priority = fastrand()
	t.span = span
	t.maxPages = span.npages
	t.parent = last
	*pt = t // t now at a leaf.

	// Update the tree to maintain the maxPages invariant.
	i := t
	for i.parent != nil {
		if i.parent.maxPages < i.maxPages {
			i.parent.maxPages = i.maxPages
		} else {
			break
		}
		i = i.parent
	}

	// Rotate up into tree according to priority.
	for t.parent != nil && t.parent.priority > t.priority {
		if t != nil && t.span.base() != t.key {
			println("runtime: insert t=", t, "t.key=", t.key)
			println("runtime:      t.span=", t.span, "t.span.base()=", t.span.base())
			throw("span and treap node base addresses do not match")
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
	if t.span.base() != t.key {
		throw("span and treap node base addresses do not match")
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
		p := t.parent
		if p.left == t {
			p.left = nil
		} else {
			p.right = nil
		}
		// Walk up the tree updating maxPages values until
		// it no longer changes, since the just-removed node
		// could have contained the biggest span in any subtree
		// up to the root.
		for p != nil {
			m := p.maxPages
			p.recomputeMaxPages()
			if p.maxPages == m {
				break
			}
			p = p.parent
		}
	} else {
		root.treap = nil
	}
	// Return the found treapNode's span after freeing the treapNode.
	mheap_.treapalloc.free(unsafe.Pointer(t))
}

// find searches for, finds, and returns the treap iterator representing the
// position of the span with the smallest base address which is at least npages
// in size. If no span has at least npages it returns an invalid iterator.
//
// This algorithm is as follows:
// * If there's a left child and its subtree can satisfy this allocation,
//   continue down that subtree.
// * If there's no such left child, check if the root of this subtree can
//   satisfy the allocation. If so, we're done.
// * If the root cannot satisfy the allocation either, continue down the
//   right subtree if able.
// * Else, break and report that we cannot satisfy the allocation.
//
// The preference for left, then current, then right, results in us getting
// the left-most node which will contain the span with the lowest base
// address.
//
// Note that if a request cannot be satisfied the fourth case will be
// reached immediately at the root, since neither the left subtree nor
// the right subtree will have a sufficient maxPages, whilst the root
// node is also unable to satisfy it.
func (root *mTreap) find(npages uintptr) treapIter {
	t := root.treap
	for t != nil {
		if t.span == nil {
			throw("treap node with nil span found")
		}
		// Iterate over the treap trying to go as far left
		// as possible while simultaneously ensuring that the
		// subtrees we choose always have a span which can
		// satisfy the allocation.
		if t.left != nil && t.left.maxPages >= npages {
			t = t.left
		} else if t.span.npages >= npages {
			// Before going right, if this span can satisfy the
			// request, stop here.
			break
		} else if t.right != nil && t.right.maxPages >= npages {
			t = t.right
		} else {
			t = nil
		}
	}
	return treapIter{t}
}

// removeSpan searches for, finds, deletes span along with
// the associated treap node. If the span is not in the treap
// then t will eventually be set to nil and the t.span
// will throw.
func (root *mTreap) removeSpan(span *mspan) {
	base := span.base()
	t := root.treap
	for t.span != span {
		if t.key < base {
			t = t.right
		} else if t.key > base {
			t = t.left
		}
	}
	root.removeNode(t)
}

// erase removes the element referred to by the current position of the
// iterator. This operation consumes the given iterator, so it should no
// longer be used. It is up to the caller to get the next or previous
// iterator before calling erase, if need be.
func (root *mTreap) erase(i treapIter) {
	root.removeNode(i.t)
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

	// Recomputing maxPages for x and y is sufficient
	// for maintaining the maxPages invariant.
	x.recomputeMaxPages()
	y.recomputeMaxPages()
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

	// Recomputing maxPages for x and y is sufficient
	// for maintaining the maxPages invariant.
	y.recomputeMaxPages()
	x.recomputeMaxPages()
}
