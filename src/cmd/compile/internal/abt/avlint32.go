// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abt

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	LEAF_HEIGHT = 1
	ZERO_HEIGHT = 0
	NOT_KEY32   = int32(-0x80000000)
)

// T is the exported applicative balanced tree data type.
// A T can be used as a value; updates to one copy of the value
// do not change other copies.
type T[D comparable] struct {
	root *node32[D]
	size int
}

// node32 is the internal tree node data type
type node32[D comparable] struct {
	// Standard conventions hold for left = smaller, right = larger
	left, right *node32[D]
	data        D
	key         int32
	height_     int8
}

func makeNode[D comparable](key int32) *node32[D] {
	return &node32[D]{key: key, height_: LEAF_HEIGHT}
}

// IsEmpty returns true iff t is empty.
func (t *T[D]) IsEmpty() bool {
	return t.root == nil
}

// IsSingle returns true iff t is a singleton (leaf).
func (t *T[D]) IsSingle() bool {
	return t.root != nil && t.root.isLeaf()
}

// VisitInOrder applies f to the key and data pairs in t,
// with keys ordered from smallest to largest.
func (t *T[D]) VisitInOrder(f func(int32, D)) {
	if t.root == nil {
		return
	}
	t.root.visitInOrder(f)
}

func (n *node32[D]) nilOrData() (d D) {
	if n != nil {
		d = n.data
	}
	return
}

func (n *node32[D]) nilOrKeyAndData() (k int32, d D) {
	if n == nil {
		k = NOT_KEY32
	} else {
		k = n.key
		d = n.data
	}
	return
}

func (n *node32[D]) height() int8 {
	if n == nil {
		return 0
	}
	return n.height_
}

// Find returns the data associated with x in the tree, or
// the zero value of D if x is not in the tree.
func (t *T[D]) Find(x int32) D {
	return t.root.find(x).nilOrData()
}

// Insert either adds x to the tree if x was not previously
// a key in the tree, or updates the data for x in the tree if
// x was already a key in the tree.  The previous data associated
// with x is returned, and is the zero value of D if x was not previously a
// key in the tree.
func (t *T[D]) Insert(x int32, data D) D {
	if x == NOT_KEY32 {
		panic("Cannot use sentinel value -0x80000000 as key")
	}
	n := t.root
	var newroot *node32[D]
	var o *node32[D]
	if n == nil {
		n = makeNode[D](x)
		newroot = n
	} else {
		newroot, n, o = n.aInsert(x)
	}
	var r D
	if o != nil {
		r = o.data
	} else {
		t.size++
	}
	n.data = data
	t.root = newroot
	return r
}

func (t *T[D]) Copy() *T[D] {
	u := *t
	return &u
}

func (t *T[D]) Delete(x int32) (r D) {
	n := t.root
	if n == nil {
		return
	}
	d, s := n.aDelete(x)
	if d == nil {
		return
	}
	t.root = s
	t.size--
	return d.data
}

func (t *T[D]) DeleteMin() (k int32, r D) {
	n := t.root
	if n == nil {
		return NOT_KEY32, r
	}
	d, s := n.aDeleteMin()
	if d == nil {
		return NOT_KEY32, r
	}
	t.root = s
	t.size--
	return d.key, d.data
}

func (t *T[D]) DeleteMax() (k int32, r D) {
	n := t.root
	if n == nil {
		return NOT_KEY32, r
	}
	d, s := n.aDeleteMax()
	if d == nil {
		return NOT_KEY32, r
	}
	t.root = s
	t.size--
	return d.key, d.data
}

func (t *T[D]) Size() int {
	return t.size
}

// Intersection returns the intersection of t and u, where the result
// data for any common keys is given by f(t's data, u's data) -- f need
// not be symmetric.  If f returns the zero value of D, then the key and data are not
// added to the result.  If f itself is nil, then whatever value was
// already present in the smaller set is used.
func (t *T[D]) Intersection(u *T[D], f func(x, y D) D) *T[D] {
	if t.Size() == 0 || u.Size() == 0 {
		return &T[D]{}
	}

	var zero D

	// For faster execution and less allocation, prefer t smaller, iterate over t.
	if t.Size() <= u.Size() {
		v := t.Copy()
		for it := t.Iterator(); !it.Done(); {
			k, d := it.Next()
			e := u.Find(k)
			if e == zero {
				v.Delete(k)
				continue
			}
			if f == nil {
				continue
			}
			if c := f(d, e); c != d {
				if c == zero {
					v.Delete(k)
				} else {
					v.Insert(k, c)
				}
			}
		}
		return v
	}
	v := u.Copy()
	for it := u.Iterator(); !it.Done(); {
		k, e := it.Next()
		d := t.Find(k)
		if d == zero {
			v.Delete(k)
			continue
		}
		if f == nil {
			continue
		}
		if c := f(d, e); c != d {
			if c == zero {
				v.Delete(k)
			} else {
				v.Insert(k, c)
			}
		}
	}

	return v
}

// Union returns the union of t and u, where the result data for any common keys
// is given by f(t's data, u's data) -- f need not be symmetric.  If f returns the zero value of D,
// then the key and data are not added to the result.  If f itself is nil, then
// whatever value was already present in the larger set is used.
func (t *T[D]) Union(u *T[D], f func(x, y D) D) *T[D] {
	if t.Size() == 0 {
		return u
	}
	if u.Size() == 0 {
		return t
	}

	var zero D

	if t.Size() >= u.Size() {
		v := t.Copy()
		for it := u.Iterator(); !it.Done(); {
			k, e := it.Next()
			d := t.Find(k)
			if d == zero {
				v.Insert(k, e)
				continue
			}
			if f == nil {
				continue
			}
			if c := f(d, e); c != d {
				if c == zero {
					v.Delete(k)
				} else {
					v.Insert(k, c)
				}
			}
		}
		return v
	}

	v := u.Copy()
	for it := t.Iterator(); !it.Done(); {
		k, d := it.Next()
		e := u.Find(k)
		if e == zero {
			v.Insert(k, d)
			continue
		}
		if f == nil {
			continue
		}
		if c := f(d, e); c != d {
			if c == zero {
				v.Delete(k)
			} else {
				v.Insert(k, c)
			}
		}
	}
	return v
}

// Difference returns the difference of t and u, subject to the result
// of f applied to data corresponding to equal keys.  If f returns the zero value of D
// (or if f is nil) then the key+data are excluded, as usual.  If f
// returns not-zero, then that key+data pair is inserted. instead.
func (t *T[D]) Difference(u *T[D], f func(x, y D) D) *T[D] {
	if t.Size() == 0 {
		return &T[D]{}
	}
	if u.Size() == 0 {
		return t
	}
	var zero D
	v := t.Copy()
	for it := t.Iterator(); !it.Done(); {
		k, d := it.Next()
		e := u.Find(k)
		if e != zero {
			if f == nil {
				v.Delete(k)
				continue
			}
			c := f(d, e)
			if c == zero {
				v.Delete(k)
				continue
			}
			if c != d {
				v.Insert(k, c)
			}
		}
	}
	return v
}

func (t *T[D]) Iterator() Iterator[D] {
	return Iterator[D]{it: t.root.iterator()}
}

func (t *T[D]) Equals(u *T[D]) bool {
	if t == u {
		return true
	}
	if t.Size() != u.Size() {
		return false
	}
	return t.root.equals(u.root)
}

func (t *T[D]) String() string {
	var b strings.Builder
	first := true
	for it := t.Iterator(); !it.Done(); {
		k, v := it.Next()
		if first {
			first = false
		} else {
			b.WriteString("; ")
		}
		b.WriteString(strconv.FormatInt(int64(k), 10))
		b.WriteString(":")
		fmt.Fprint(&b, v)
	}
	return b.String()
}

func (t *node32[D]) equals(u *node32[D]) bool {
	if t == u {
		return true
	}
	it, iu := t.iterator(), u.iterator()
	for !it.done() && !iu.done() {
		nt := it.next()
		nu := iu.next()
		if nt == nu {
			continue
		}
		if nt.key != nu.key {
			return false
		}
		if nt.data != nu.data {
			return false
		}
	}
	return it.done() == iu.done()
}

func (t *T[D]) Equiv(u *T[D], eqv func(x, y D) bool) bool {
	if t == u {
		return true
	}
	if t.Size() != u.Size() {
		return false
	}
	return t.root.equiv(u.root, eqv)
}

func (t *node32[D]) equiv(u *node32[D], eqv func(x, y D) bool) bool {
	if t == u {
		return true
	}
	it, iu := t.iterator(), u.iterator()
	for !it.done() && !iu.done() {
		nt := it.next()
		nu := iu.next()
		if nt == nu {
			continue
		}
		if nt.key != nu.key {
			return false
		}
		if !eqv(nt.data, nu.data) {
			return false
		}
	}
	return it.done() == iu.done()
}

type iterator[D comparable] struct {
	parents []*node32[D]
}

type Iterator[D comparable] struct {
	it iterator[D]
}

func (it *Iterator[D]) Next() (k int32, d D) {
	x := it.it.next()
	if x == nil {
		return NOT_KEY32, d
	}
	return x.key, x.data
}

func (it *Iterator[D]) Done() bool {
	return len(it.it.parents) == 0
}

func (t *node32[D]) iterator() iterator[D] {
	if t == nil {
		return iterator[D]{}
	}
	it := iterator[D]{parents: make([]*node32[D], 0, int(t.height()))}
	it.leftmost(t)
	return it
}

func (it *iterator[D]) leftmost(t *node32[D]) {
	for t != nil {
		it.parents = append(it.parents, t)
		t = t.left
	}
}

func (it *iterator[D]) done() bool {
	return len(it.parents) == 0
}

func (it *iterator[D]) next() *node32[D] {
	l := len(it.parents)
	if l == 0 {
		return nil
	}
	x := it.parents[l-1] // return value
	if x.right != nil {
		it.leftmost(x.right)
		return x
	}
	// discard visited top of parents
	l--
	it.parents = it.parents[:l]
	y := x // y is known visited/returned
	for l > 0 && y == it.parents[l-1].right {
		y = it.parents[l-1]
		l--
		it.parents = it.parents[:l]
	}

	return x
}

// Min returns the minimum element of t.
// If t is empty, then (NOT_KEY32, zero) is returned.
func (t *T[D]) Min() (k int32, d D) {
	return t.root.min().nilOrKeyAndData()
}

// Max returns the maximum element of t.
// If t is empty, then (NOT_KEY32, zero) is returned.
func (t *T[D]) Max() (k int32, d D) {
	return t.root.max().nilOrKeyAndData()
}

// Glb returns the greatest-lower-bound-exclusive of x and the associated
// data.  If x has no glb in the tree, then (NOT_KEY32, zero) is returned.
func (t *T[D]) Glb(x int32) (k int32, d D) {
	return t.root.glb(x, false).nilOrKeyAndData()
}

// GlbEq returns the greatest-lower-bound-inclusive of x and the associated
// data.  If x has no glbEQ in the tree, then (NOT_KEY32, zero) is returned.
func (t *T[D]) GlbEq(x int32) (k int32, d D) {
	return t.root.glb(x, true).nilOrKeyAndData()
}

// Lub returns the least-upper-bound-exclusive of x and the associated
// data.  If x has no lub in the tree, then (NOT_KEY32, zero) is returned.
func (t *T[D]) Lub(x int32) (k int32, d D) {
	return t.root.lub(x, false).nilOrKeyAndData()
}

// LubEq returns the least-upper-bound-inclusive of x and the associated
// data.  If x has no lubEq in the tree, then (NOT_KEY32, zero) is returned.
func (t *T[D]) LubEq(x int32) (k int32, d D) {
	return t.root.lub(x, true).nilOrKeyAndData()
}

func (t *node32[D]) isLeaf() bool {
	return t.left == nil && t.right == nil && t.height_ == LEAF_HEIGHT
}

func (t *node32[D]) visitInOrder(f func(int32, D)) {
	if t.left != nil {
		t.left.visitInOrder(f)
	}
	f(t.key, t.data)
	if t.right != nil {
		t.right.visitInOrder(f)
	}
}

func (t *node32[D]) find(key int32) *node32[D] {
	for t != nil {
		if key < t.key {
			t = t.left
		} else if key > t.key {
			t = t.right
		} else {
			return t
		}
	}
	return nil
}

func (t *node32[D]) min() *node32[D] {
	if t == nil {
		return t
	}
	for t.left != nil {
		t = t.left
	}
	return t
}

func (t *node32[D]) max() *node32[D] {
	if t == nil {
		return t
	}
	for t.right != nil {
		t = t.right
	}
	return t
}

func (t *node32[D]) glb(key int32, allow_eq bool) *node32[D] {
	var best *node32[D] = nil
	for t != nil {
		if key <= t.key {
			if allow_eq && key == t.key {
				return t
			}
			// t is too big, glb is to left.
			t = t.left
		} else {
			// t is a lower bound, record it and seek a better one.
			best = t
			t = t.right
		}
	}
	return best
}

func (t *node32[D]) lub(key int32, allow_eq bool) *node32[D] {
	var best *node32[D] = nil
	for t != nil {
		if key >= t.key {
			if allow_eq && key == t.key {
				return t
			}
			// t is too small, lub is to right.
			t = t.right
		} else {
			// t is an upper bound, record it and seek a better one.
			best = t
			t = t.left
		}
	}
	return best
}

func (t *node32[D]) aInsert(x int32) (newroot, newnode, oldnode *node32[D]) {
	// oldnode default of nil is good, others should be assigned.
	if x == t.key {
		oldnode = t
		newt := *t
		newnode = &newt
		newroot = newnode
		return
	}
	if x < t.key {
		if t.left == nil {
			t = t.copy()
			n := makeNode[D](x)
			t.left = n
			newnode = n
			newroot = t
			t.height_ = 2 // was balanced w/ 0, sibling is height 0 or 1
			return
		}
		var new_l *node32[D]
		new_l, newnode, oldnode = t.left.aInsert(x)
		t = t.copy()
		t.left = new_l
		if new_l.height() > 1+t.right.height() {
			newroot = t.aLeftIsHigh(newnode)
		} else {
			t.height_ = 1 + max(t.left.height(), t.right.height())
			newroot = t
		}
	} else { // x > t.key
		if t.right == nil {
			t = t.copy()
			n := makeNode[D](x)
			t.right = n
			newnode = n
			newroot = t
			t.height_ = 2 // was balanced w/ 0, sibling is height 0 or 1
			return
		}
		var new_r *node32[D]
		new_r, newnode, oldnode = t.right.aInsert(x)
		t = t.copy()
		t.right = new_r
		if new_r.height() > 1+t.left.height() {
			newroot = t.aRightIsHigh(newnode)
		} else {
			t.height_ = 1 + max(t.left.height(), t.right.height())
			newroot = t
		}
	}
	return
}

func (t *node32[D]) aDelete(key int32) (deleted, newSubTree *node32[D]) {
	if t == nil {
		return nil, nil
	}

	if key < t.key {
		oh := t.left.height()
		d, tleft := t.left.aDelete(key)
		if tleft == t.left {
			return d, t
		}
		return d, t.copy().aRebalanceAfterLeftDeletion(oh, tleft)
	} else if key > t.key {
		oh := t.right.height()
		d, tright := t.right.aDelete(key)
		if tright == t.right {
			return d, t
		}
		return d, t.copy().aRebalanceAfterRightDeletion(oh, tright)
	}

	if t.height() == LEAF_HEIGHT {
		return t, nil
	}

	// Interior delete by removing left.Max or right.Min,
	// then swapping contents
	if t.left.height() > t.right.height() {
		oh := t.left.height()
		d, tleft := t.left.aDeleteMax()
		r := t
		t = t.copy()
		t.data, t.key = d.data, d.key
		return r, t.aRebalanceAfterLeftDeletion(oh, tleft)
	}

	oh := t.right.height()
	d, tright := t.right.aDeleteMin()
	r := t
	t = t.copy()
	t.data, t.key = d.data, d.key
	return r, t.aRebalanceAfterRightDeletion(oh, tright)
}

func (t *node32[D]) aDeleteMin() (deleted, newSubTree *node32[D]) {
	if t == nil {
		return nil, nil
	}
	if t.left == nil { // leaf or left-most
		return t, t.right
	}
	oh := t.left.height()
	d, tleft := t.left.aDeleteMin()
	if tleft == t.left {
		return d, t
	}
	return d, t.copy().aRebalanceAfterLeftDeletion(oh, tleft)
}

func (t *node32[D]) aDeleteMax() (deleted, newSubTree *node32[D]) {
	if t == nil {
		return nil, nil
	}

	if t.right == nil { // leaf or right-most
		return t, t.left
	}

	oh := t.right.height()
	d, tright := t.right.aDeleteMax()
	if tright == t.right {
		return d, t
	}
	return d, t.copy().aRebalanceAfterRightDeletion(oh, tright)
}

func (t *node32[D]) aRebalanceAfterLeftDeletion(oldLeftHeight int8, tleft *node32[D]) *node32[D] {
	t.left = tleft

	if oldLeftHeight == tleft.height() || oldLeftHeight == t.right.height() {
		// this node is still balanced and its height is unchanged
		return t
	}

	if oldLeftHeight > t.right.height() {
		// left was larger
		t.height_--
		return t
	}

	// left height fell by 1 and it was already less than right height
	t.right = t.right.copy()
	return t.aRightIsHigh(nil)
}

func (t *node32[D]) aRebalanceAfterRightDeletion(oldRightHeight int8, tright *node32[D]) *node32[D] {
	t.right = tright

	if oldRightHeight == tright.height() || oldRightHeight == t.left.height() {
		// this node is still balanced and its height is unchanged
		return t
	}

	if oldRightHeight > t.left.height() {
		// left was larger
		t.height_--
		return t
	}

	// right height fell by 1 and it was already less than left height
	t.left = t.left.copy()
	return t.aLeftIsHigh(nil)
}

// aRightIsHigh does rotations necessary to fix a high right child
// assume that t and t.right are already fresh copies.
func (t *node32[D]) aRightIsHigh(newnode *node32[D]) *node32[D] {
	right := t.right
	if right.right.height() < right.left.height() {
		// double rotation
		if newnode != right.left {
			right.left = right.left.copy()
		}
		t.right = right.leftToRoot()
	}
	t = t.rightToRoot()
	return t
}

// aLeftIsHigh does rotations necessary to fix a high left child
// assume that t and t.left are already fresh copies.
func (t *node32[D]) aLeftIsHigh(newnode *node32[D]) *node32[D] {
	left := t.left
	if left.left.height() < left.right.height() {
		// double rotation
		if newnode != left.right {
			left.right = left.right.copy()
		}
		t.left = left.rightToRoot()
	}
	t = t.leftToRoot()
	return t
}

// rightToRoot does that rotation, modifying t and t.right in the process.
func (t *node32[D]) rightToRoot() *node32[D] {
	//    this
	// left  right
	//      rl   rr
	//
	// becomes
	//
	//       right
	//    this   rr
	// left  rl
	//
	right := t.right
	rl := right.left
	right.left = t
	// parent's child ptr fixed in caller
	t.right = rl
	t.height_ = 1 + max(rl.height(), t.left.height())
	right.height_ = 1 + max(t.height(), right.right.height())
	return right
}

// leftToRoot does that rotation, modifying t and t.left in the process.
func (t *node32[D]) leftToRoot() *node32[D] {
	//     this
	//  left  right
	// ll  lr
	//
	// becomes
	//
	//    left
	//   ll  this
	//      lr  right
	//
	left := t.left
	lr := left.right
	left.right = t
	// parent's child ptr fixed in caller
	t.left = lr
	t.height_ = 1 + max(lr.height(), t.right.height())
	left.height_ = 1 + max(t.height(), left.left.height())
	return left
}

func (t *node32[D]) copy() *node32[D] {
	u := *t
	return &u
}
