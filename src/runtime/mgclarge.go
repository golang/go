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
// Whenever a pointer to a span which is owned by the treap is acquired, that
// span must not be mutated. To mutate a span in the treap, remove it first.
//
// mheap_.lock must be held when manipulating this data structure.

package runtime

import (
	"unsafe"
)

//go:notinheap
type mTreap struct {
	treap           *treapNode
	unscavHugePages uintptr // number of unscavenged huge pages in the treap
}

//go:notinheap
type treapNode struct {
	right    *treapNode      // all treapNodes > this treap node
	left     *treapNode      // all treapNodes < this treap node
	parent   *treapNode      // direct parent of this node, nil if root
	key      uintptr         // base address of the span, used as primary sort key
	span     *mspan          // span at base address key
	maxPages uintptr         // the maximum size of any span in this subtree, including the root
	priority uint32          // random number used by treap algorithm to keep tree probabilistically balanced
	types    treapIterFilter // the types of spans available in this subtree
}

// updateInvariants is a helper method which has a node recompute its own
// maxPages and types values by looking at its own span as well as the
// values of its direct children.
//
// Returns true if anything changed.
func (t *treapNode) updateInvariants() bool {
	m, i := t.maxPages, t.types
	t.maxPages = t.span.npages
	t.types = t.span.treapFilter()
	if t.left != nil {
		t.types |= t.left.types
		if t.maxPages < t.left.maxPages {
			t.maxPages = t.left.maxPages
		}
	}
	if t.right != nil {
		t.types |= t.right.types
		if t.maxPages < t.right.maxPages {
			t.maxPages = t.right.maxPages
		}
	}
	return m != t.maxPages || i != t.types
}

// findMinimal finds the minimal (lowest base addressed) node in the treap
// which matches the criteria set out by the filter f and returns nil if
// none exists.
//
// This algorithm is functionally the same as (*mTreap).find, so see that
// method for more details.
func (t *treapNode) findMinimal(f treapIterFilter) *treapNode {
	if t == nil || !f.matches(t.types) {
		return nil
	}
	for t != nil {
		if t.left != nil && f.matches(t.left.types) {
			t = t.left
		} else if f.matches(t.span.treapFilter()) {
			break
		} else if t.right != nil && f.matches(t.right.types) {
			t = t.right
		} else {
			println("runtime: f=", f)
			throw("failed to find minimal node matching filter")
		}
	}
	return t
}

// findMaximal finds the maximal (highest base addressed) node in the treap
// which matches the criteria set out by the filter f and returns nil if
// none exists.
//
// This algorithm is the logical inversion of findMinimal and just changes
// the order of the left and right tests.
func (t *treapNode) findMaximal(f treapIterFilter) *treapNode {
	if t == nil || !f.matches(t.types) {
		return nil
	}
	for t != nil {
		if t.right != nil && f.matches(t.right.types) {
			t = t.right
		} else if f.matches(t.span.treapFilter()) {
			break
		} else if t.left != nil && f.matches(t.left.types) {
			t = t.left
		} else {
			println("runtime: f=", f)
			throw("failed to find minimal node matching filter")
		}
	}
	return t
}

// pred returns the predecessor of t in the treap subject to the criteria
// specified by the filter f. Returns nil if no such predecessor exists.
func (t *treapNode) pred(f treapIterFilter) *treapNode {
	if t.left != nil && f.matches(t.left.types) {
		// The node has a left subtree which contains at least one matching
		// node, find the maximal matching node in that subtree.
		return t.left.findMaximal(f)
	}
	// Lacking a left subtree, look to the parents.
	p := t // previous node
	t = t.parent
	for t != nil {
		// Walk up the tree until we find a node that has a left subtree
		// that we haven't already visited.
		if t.right == p {
			if f.matches(t.span.treapFilter()) {
				// If this node matches, then it's guaranteed to be the
				// predecessor since everything to its left is strictly
				// greater.
				return t
			} else if t.left != nil && f.matches(t.left.types) {
				// Failing the root of this subtree, if its left subtree has
				// something, that's where we'll find our predecessor.
				return t.left.findMaximal(f)
			}
		}
		p = t
		t = t.parent
	}
	// If the parent is nil, then we've hit the root without finding
	// a suitable left subtree containing the node (and the predecessor
	// wasn't on the path). Thus, there's no predecessor, so just return
	// nil.
	return nil
}

// succ returns the successor of t in the treap subject to the criteria
// specified by the filter f. Returns nil if no such successor exists.
func (t *treapNode) succ(f treapIterFilter) *treapNode {
	// See pred. This method is just the logical inversion of it.
	if t.right != nil && f.matches(t.right.types) {
		return t.right.findMinimal(f)
	}
	p := t
	t = t.parent
	for t != nil {
		if t.left == p {
			if f.matches(t.span.treapFilter()) {
				return t
			} else if t.right != nil && f.matches(t.right.types) {
				return t.right.findMinimal(f)
			}
		}
		p = t
		t = t.parent
	}
	return nil
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

// validateInvariants is handy for debugging and testing.
// It ensures that the various invariants on each treap node are
// appropriately maintained throughout the treap by walking the
// treap in a post-order manner.
func (t *treapNode) validateInvariants() (uintptr, treapIterFilter) {
	if t == nil {
		return 0, 0
	}
	leftMax, leftTypes := t.left.validateInvariants()
	rightMax, rightTypes := t.right.validateInvariants()
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
	typ := t.span.treapFilter() | leftTypes | rightTypes
	if typ != t.types {
		println("runtime: t.types=", t.types, "want=", typ)
		throw("types invariant violated in treap")
	}
	return max, typ
}

// treapIterType represents the type of iteration to perform
// over the treap. Each different flag is represented by a bit
// in the type, and types may be combined together by a bitwise
// or operation.
//
// Note that only 5 bits are available for treapIterType, do not
// use the 3 higher-order bits. This constraint is to allow for
// expansion into a treapIterFilter, which is a uint32.
type treapIterType uint8

const (
	treapIterScav treapIterType = 1 << iota // scavenged spans
	treapIterHuge                           // spans containing at least one huge page
	treapIterBits = iota
)

// treapIterFilter is a bitwise filter of different spans by binary
// properties. Each bit of a treapIterFilter represents a unique
// combination of bits set in a treapIterType, in other words, it
// represents the power set of a treapIterType.
//
// The purpose of this representation is to allow the existence of
// a specific span type to bubble up in the treap (see the types
// field on treapNode).
//
// More specifically, any treapIterType may be transformed into a
// treapIterFilter for a specific combination of flags via the
// following operation: 1 << (0x1f&treapIterType).
type treapIterFilter uint32

// treapFilterAll represents the filter which allows all spans.
const treapFilterAll = ^treapIterFilter(0)

// treapFilter creates a new treapIterFilter from two treapIterTypes.
// mask represents a bitmask for which flags we should check against
// and match for the expected result after applying the mask.
func treapFilter(mask, match treapIterType) treapIterFilter {
	allow := treapIterFilter(0)
	for i := treapIterType(0); i < 1<<treapIterBits; i++ {
		if mask&i == match {
			allow |= 1 << i
		}
	}
	return allow
}

// matches returns true if m and f intersect.
func (f treapIterFilter) matches(m treapIterFilter) bool {
	return f&m != 0
}

// treapFilter returns the treapIterFilter exactly matching this span,
// i.e. popcount(result) == 1.
func (s *mspan) treapFilter() treapIterFilter {
	have := treapIterType(0)
	if s.scavenged {
		have |= treapIterScav
	}
	if s.hugePages() > 0 {
		have |= treapIterHuge
	}
	return treapIterFilter(uint32(1) << (0x1f & have))
}

// treapIter is a bidirectional iterator type which may be used to iterate over a
// an mTreap in-order forwards (increasing order) or backwards (decreasing order).
// Its purpose is to hide details about the treap from users when trying to iterate
// over it.
//
// To create iterators over the treap, call start or end on an mTreap.
type treapIter struct {
	f treapIterFilter
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
	i.t = i.t.succ(i.f)
	return i
}

// prev moves the iterator backwards by one. Once the iterator
// ceases to be valid, calling prev will panic.
func (i treapIter) prev() treapIter {
	i.t = i.t.pred(i.f)
	return i
}

// start returns an iterator which points to the start of the treap (the
// left-most node in the treap) subject to mask and match constraints.
func (root *mTreap) start(mask, match treapIterType) treapIter {
	f := treapFilter(mask, match)
	return treapIter{f, root.treap.findMinimal(f)}
}

// end returns an iterator which points to the end of the treap (the
// right-most node in the treap) subject to mask and match constraints.
func (root *mTreap) end(mask, match treapIterType) treapIter {
	f := treapFilter(mask, match)
	return treapIter{f, root.treap.findMaximal(f)}
}

// mutate allows one to mutate the span without removing it from the treap via a
// callback. The span's base and size are allowed to change as long as the span
// remains in the same order relative to its predecessor and successor.
//
// Note however that any operation that causes a treap rebalancing inside of fn
// is strictly forbidden, as that may cause treap node metadata to go
// out-of-sync.
func (root *mTreap) mutate(i treapIter, fn func(span *mspan)) {
	s := i.span()
	// Save some state about the span for later inspection.
	hpages := s.hugePages()
	scavenged := s.scavenged
	// Call the mutator.
	fn(s)
	// Update unscavHugePages appropriately.
	if !scavenged {
		mheap_.free.unscavHugePages -= hpages
	}
	if !s.scavenged {
		mheap_.free.unscavHugePages += s.hugePages()
	}
	// Update the key in case the base changed.
	i.t.key = s.base()
	// Updating invariants up the tree needs to happen if
	// anything changed at all, so just go ahead and do it
	// unconditionally.
	//
	// If it turns out nothing changed, it'll exit quickly.
	t := i.t
	for t != nil && t.updateInvariants() {
		t = t.parent
	}
}

// insert adds span to the large span treap.
func (root *mTreap) insert(span *mspan) {
	if !span.scavenged {
		root.unscavHugePages += span.hugePages()
	}
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
	t.types = span.treapFilter()
	t.parent = last
	*pt = t // t now at a leaf.

	// Update the tree to maintain the various invariants.
	i := t
	for i.parent != nil && i.parent.updateInvariants() {
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
	if !t.span.scavenged {
		root.unscavHugePages -= t.span.hugePages()
	}
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
		// Walk up the tree updating invariants until no updates occur.
		for p != nil && p.updateInvariants() {
			p = p.parent
		}
	} else {
		root.treap = nil
	}
	// Return the found treapNode's span after freeing the treapNode.
	mheap_.treapalloc.free(unsafe.Pointer(t))
}

// find searches for, finds, and returns the treap iterator over all spans
// representing the position of the span with the smallest base address which is
// at least npages in size. If no span has at least npages it returns an invalid
// iterator.
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
	return treapIter{treapFilterAll, t}
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

	x.updateInvariants()
	y.updateInvariants()
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

	y.updateInvariants()
	x.updateInvariants()
}
