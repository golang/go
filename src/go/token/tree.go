// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

// tree is a self-balancing AVL tree; see
// Lewis & Denenberg, Data Structures and Their Algorithms.
//
// An AVL tree is a binary tree in which the difference between the
// heights of a node's two subtrees--the node's "balance factor"--is
// at most one. It is more strictly balanced than a red/black tree,
// and thus favors lookups at the expense of updates, which is the
// appropriate trade-off for FileSet.
//
// Insertion at a node may cause its ancestors' balance factors to
// temporarily reach ±2, requiring rebalancing of each such ancestor
// by a rotation.
//
// Each key is the pos-end range of a single File.
// All Files in the tree must have disjoint ranges.
//
// The implementation is simplified from Russ Cox's github.com/rsc/omap.

import (
	"fmt"
	"iter"
)

// A tree is a tree-based ordered map:
// each value is a *File, keyed by its Pos range.
// All map entries cover disjoint ranges.
//
// The zero value of tree is an empty map ready to use.
type tree struct {
	root *node
}

type node struct {
	// We use the notation (parent left right) in many comments.
	parent  *node
	left    *node
	right   *node
	file    *File
	key     key   // = file.key(), but improves locality (25% faster)
	balance int32 // at most ±2
	height  int32
}

// A key represents the Pos range of a File.
type key struct{ start, end int }

func (f *File) key() key {
	return key{f.base, f.base + f.size}
}

// compareKey reports whether x is before y (-1),
// after y (+1), or overlapping y (0).
// This is a total order so long as all
// files in the tree have disjoint ranges.
//
// All files are separated by at least one unit.
// This allows us to use strict < comparisons.
// Use key{p, p} to search for a zero-width position
// even at the start or end of a file.
func compareKey(x, y key) int {
	switch {
	case x.end < y.start:
		return -1
	case y.end < x.start:
		return +1
	}
	return 0
}

// check asserts that each node's height, subtree, and parent link is
// correct.
func (n *node) check(parent *node) {
	const debugging = false
	if debugging {
		if n == nil {
			return
		}
		if n.parent != parent {
			panic("bad parent")
		}
		n.left.check(n)
		n.right.check(n)
		n.checkBalance()
	}
}

func (n *node) checkBalance() {
	lheight, rheight := n.left.safeHeight(), n.right.safeHeight()
	balance := rheight - lheight
	if balance != n.balance {
		panic("bad node.balance")
	}
	if !(-2 <= balance && balance <= +2) {
		panic(fmt.Sprintf("node.balance out of range: %d", balance))
	}
	h := 1 + max(lheight, rheight)
	if h != n.height {
		panic("bad node.height")
	}
}

// locate returns a pointer to the variable that holds the node
// identified by k, along with its parent, if any. If the key is not
// present, it returns a pointer to the node where the key should be
// inserted by a subsequent call to [tree.set].
func (t *tree) locate(k key) (pos **node, parent *node) {
	pos, x := &t.root, t.root
	for x != nil {
		sign := compareKey(k, x.key)
		if sign < 0 {
			pos, x, parent = &x.left, x.left, x
		} else if sign > 0 {
			pos, x, parent = &x.right, x.right, x
		} else {
			break
		}
	}
	return pos, parent
}

// all returns an iterator over the tree t.
// If t is modified during the iteration,
// some files may not be visited.
// No file will be visited multiple times.
func (t *tree) all() iter.Seq[*File] {
	return func(yield func(*File) bool) {
		if t == nil {
			return
		}
		x := t.root
		if x != nil {
			for x.left != nil {
				x = x.left
			}
		}
		for x != nil && yield(x.file) {
			if x.height >= 0 {
				// still in tree
				x = x.next()
			} else {
				// deleted
				x = t.nextAfter(t.locate(x.key))
			}
		}
	}
}

// nextAfter returns the node in the key sequence following
// (pos, parent), a result pair from [tree.locate].
func (t *tree) nextAfter(pos **node, parent *node) *node {
	switch {
	case *pos != nil:
		return (*pos).next()
	case parent == nil:
		return nil
	case pos == &parent.left:
		return parent
	default:
		return parent.next()
	}
}

func (x *node) next() *node {
	if x.right == nil {
		for x.parent != nil && x.parent.right == x {
			x = x.parent
		}
		return x.parent
	}
	x = x.right
	for x.left != nil {
		x = x.left
	}
	return x
}

func (t *tree) setRoot(x *node) {
	t.root = x
	if x != nil {
		x.parent = nil
	}
}

func (x *node) setLeft(y *node) {
	x.left = y
	if y != nil {
		y.parent = x
	}
}

func (x *node) setRight(y *node) {
	x.right = y
	if y != nil {
		y.parent = x
	}
}

func (n *node) safeHeight() int32 {
	if n == nil {
		return -1
	}
	return n.height
}

func (n *node) update() {
	lheight, rheight := n.left.safeHeight(), n.right.safeHeight()
	n.height = max(lheight, rheight) + 1
	n.balance = rheight - lheight
}

func (t *tree) replaceChild(parent, old, new *node) {
	switch {
	case parent == nil:
		if t.root != old {
			panic("corrupt tree")
		}
		t.setRoot(new)
	case parent.left == old:
		parent.setLeft(new)
	case parent.right == old:
		parent.setRight(new)
	default:
		panic("corrupt tree")
	}
}

// rebalanceUp visits each excessively unbalanced ancestor
// of x, restoring balance by rotating it.
//
// x is a node that has just been mutated, and so the height and
// balance of x and its ancestors may be stale, but the children of x
// must be in a valid state.
func (t *tree) rebalanceUp(x *node) {
	for x != nil {
		h := x.height
		x.update()
		switch x.balance {
		case -2:
			if x.left.balance == 1 {
				t.rotateLeft(x.left)
			}
			x = t.rotateRight(x)

		case +2:
			if x.right.balance == -1 {
				t.rotateRight(x.right)
			}
			x = t.rotateLeft(x)
		}
		if x.height == h {
			// x's height has not changed, so the height
			// and balance of its ancestors have not changed;
			// no further rebalancing is required.
			return
		}
		x = x.parent
	}
}

// rotateRight rotates the subtree rooted at node y.
// turning (y (x a b) c) into (x a (y b c)).
func (t *tree) rotateRight(y *node) *node {
	// p -> (y (x a b) c)
	p := y.parent
	x := y.left
	b := x.right

	x.checkBalance()
	y.checkBalance()

	x.setRight(y)
	y.setLeft(b)
	t.replaceChild(p, y, x)

	y.update()
	x.update()
	return x
}

// rotateLeft rotates the subtree rooted at node x.
// turning (x a (y b c)) into (y (x a b) c).
func (t *tree) rotateLeft(x *node) *node {
	// p -> (x a (y b c))
	p := x.parent
	y := x.right
	b := y.left

	x.checkBalance()
	y.checkBalance()

	y.setLeft(x)
	x.setRight(b)
	t.replaceChild(p, x, y)

	x.update()
	y.update()
	return y
}

// add inserts file into the tree, if not present.
// It panics if file overlaps with another.
func (t *tree) add(file *File) {
	pos, parent := t.locate(file.key())
	if *pos == nil {
		t.set(file, pos, parent) // missing; insert
		return
	}
	if prev := (*pos).file; prev != file {
		panic(fmt.Sprintf("file %s (%d-%d) overlaps with file %s (%d-%d)",
			prev.Name(), prev.Base(), prev.Base()+prev.Size(),
			file.Name(), file.Base(), file.Base()+file.Size()))
	}
}

// set updates the existing node at (pos, parent) if present, or
// inserts a new node if not, so that it refers to file.
func (t *tree) set(file *File, pos **node, parent *node) {
	if x := *pos; x != nil {
		// This code path isn't currently needed
		// because FileSet never updates an existing entry.
		// Remove this assertion if things change.
		if true { // defeat vet's unreachable pass
			panic("unreachable according to current FileSet requirements")
		}
		x.file = file
		return
	}
	x := &node{file: file, key: file.key(), parent: parent, height: -1}
	*pos = x
	t.rebalanceUp(x)
}

// delete deletes the node at pos.
func (t *tree) delete(pos **node) {
	t.root.check(nil)

	x := *pos
	switch {
	case x == nil:
		// This code path isn't currently needed because FileSet
		// only calls delete after a positive locate.
		// Remove this assertion if things change.
		if true { // defeat vet's unreachable pass
			panic("unreachable according to current FileSet requirements")
		}
		return

	case x.left == nil:
		if *pos = x.right; *pos != nil {
			(*pos).parent = x.parent
		}
		t.rebalanceUp(x.parent)

	case x.right == nil:
		*pos = x.left
		x.left.parent = x.parent
		t.rebalanceUp(x.parent)

	default:
		t.deleteSwap(pos)
	}

	x.balance = -100
	x.parent = nil
	x.left = nil
	x.right = nil
	x.height = -1
	t.root.check(nil)
}

// deleteSwap deletes a node that has two children by replacing
// it by its in-order successor, then triggers a rebalance.
func (t *tree) deleteSwap(pos **node) {
	x := *pos
	z := t.deleteMin(&x.right)

	*pos = z
	unbalanced := z.parent // lowest potentially unbalanced node
	if unbalanced == x {
		unbalanced = z // (x a (z nil b)) -> (z a b)
	}
	z.parent = x.parent
	z.height = x.height
	z.balance = x.balance
	z.setLeft(x.left)
	z.setRight(x.right)

	t.rebalanceUp(unbalanced)
}

// deleteMin updates the subtree rooted at *zpos to delete its minimum
// (leftmost) element, which may be *zpos itself. It returns the
// deleted node.
func (t *tree) deleteMin(zpos **node) (z *node) {
	for (*zpos).left != nil {
		zpos = &(*zpos).left
	}
	z = *zpos
	*zpos = z.right
	if *zpos != nil {
		(*zpos).parent = z.parent
	}
	return z
}
