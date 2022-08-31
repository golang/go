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
type T struct {
	root *node32
	size int
}

// node32 is the internal tree node data type
type node32 struct {
	// Standard conventions hold for left = smaller, right = larger
	left, right *node32
	data        interface{}
	key         int32
	height_     int8
}

func makeNode(key int32) *node32 {
	return &node32{key: key, height_: LEAF_HEIGHT}
}

// IsSingle returns true iff t is empty.
func (t *T) IsEmpty() bool {
	return t.root == nil
}

// IsSingle returns true iff t is a singleton (leaf).
func (t *T) IsSingle() bool {
	return t.root != nil && t.root.isLeaf()
}

// VisitInOrder applies f to the key and data pairs in t,
// with keys ordered from smallest to largest.
func (t *T) VisitInOrder(f func(int32, interface{})) {
	if t.root == nil {
		return
	}
	t.root.visitInOrder(f)
}

func (n *node32) nilOrData() interface{} {
	if n == nil {
		return nil
	}
	return n.data
}

func (n *node32) nilOrKeyAndData() (k int32, d interface{}) {
	if n == nil {
		k = NOT_KEY32
		d = nil
	} else {
		k = n.key
		d = n.data
	}
	return
}

func (n *node32) height() int8 {
	if n == nil {
		return 0
	}
	return n.height_
}

// Find returns the data associated with x in the tree, or
// nil if x is not in the tree.
func (t *T) Find(x int32) interface{} {
	return t.root.find(x).nilOrData()
}

// Insert either adds x to the tree if x was not previously
// a key in the tree, or updates the data for x in the tree if
// x was already a key in the tree.  The previous data associated
// with x is returned, and is nil if x was not previously a
// key in the tree.
func (t *T) Insert(x int32, data interface{}) interface{} {
	if x == NOT_KEY32 {
		panic("Cannot use sentinel value -0x80000000 as key")
	}
	n := t.root
	var newroot *node32
	var o *node32
	if n == nil {
		n = makeNode(x)
		newroot = n
	} else {
		newroot, n, o = n.aInsert(x)
	}
	var r interface{}
	if o != nil {
		r = o.data
	} else {
		t.size++
	}
	n.data = data
	t.root = newroot
	return r
}

func (t *T) Copy() *T {
	u := *t
	return &u
}

func (t *T) Delete(x int32) interface{} {
	n := t.root
	if n == nil {
		return nil
	}
	d, s := n.aDelete(x)
	if d == nil {
		return nil
	}
	t.root = s
	t.size--
	return d.data
}

func (t *T) DeleteMin() (int32, interface{}) {
	n := t.root
	if n == nil {
		return NOT_KEY32, nil
	}
	d, s := n.aDeleteMin()
	if d == nil {
		return NOT_KEY32, nil
	}
	t.root = s
	t.size--
	return d.key, d.data
}

func (t *T) DeleteMax() (int32, interface{}) {
	n := t.root
	if n == nil {
		return NOT_KEY32, nil
	}
	d, s := n.aDeleteMax()
	if d == nil {
		return NOT_KEY32, nil
	}
	t.root = s
	t.size--
	return d.key, d.data
}

func (t *T) Size() int {
	return t.size
}

// Intersection returns the intersection of t and u, where the result
// data for any common keys is given by f(t's data, u's data) -- f need
// not be symmetric.  If f returns nil, then the key and data are not
// added to the result.  If f itself is nil, then whatever value was
// already present in the smaller set is used.
func (t *T) Intersection(u *T, f func(x, y interface{}) interface{}) *T {
	if t.Size() == 0 || u.Size() == 0 {
		return &T{}
	}

	// For faster execution and less allocation, prefer t smaller, iterate over t.
	if t.Size() <= u.Size() {
		v := t.Copy()
		for it := t.Iterator(); !it.Done(); {
			k, d := it.Next()
			e := u.Find(k)
			if e == nil {
				v.Delete(k)
				continue
			}
			if f == nil {
				continue
			}
			if c := f(d, e); c != d {
				if c == nil {
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
		if d == nil {
			v.Delete(k)
			continue
		}
		if f == nil {
			continue
		}
		if c := f(d, e); c != d {
			if c == nil {
				v.Delete(k)
			} else {
				v.Insert(k, c)
			}
		}
	}

	return v
}

// Union returns the union of t and u, where the result data for any common keys
// is given by f(t's data, u's data) -- f need not be symmetric.  If f returns nil,
// then the key and data are not added to the result.  If f itself is nil, then
// whatever value was already present in the larger set is used.
func (t *T) Union(u *T, f func(x, y interface{}) interface{}) *T {
	if t.Size() == 0 {
		return u
	}
	if u.Size() == 0 {
		return t
	}

	if t.Size() >= u.Size() {
		v := t.Copy()
		for it := u.Iterator(); !it.Done(); {
			k, e := it.Next()
			d := t.Find(k)
			if d == nil {
				v.Insert(k, e)
				continue
			}
			if f == nil {
				continue
			}
			if c := f(d, e); c != d {
				if c == nil {
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
		if e == nil {
			v.Insert(k, d)
			continue
		}
		if f == nil {
			continue
		}
		if c := f(d, e); c != d {
			if c == nil {
				v.Delete(k)
			} else {
				v.Insert(k, c)
			}
		}
	}
	return v
}

// Difference returns the difference of t and u, subject to the result
// of f applied to data corresponding to equal keys.  If f returns nil
// (or if f is nil) then the key+data are excluded, as usual.  If f
// returns not-nil, then that key+data pair is inserted. instead.
func (t *T) Difference(u *T, f func(x, y interface{}) interface{}) *T {
	if t.Size() == 0 {
		return &T{}
	}
	if u.Size() == 0 {
		return t
	}
	v := t.Copy()
	for it := t.Iterator(); !it.Done(); {
		k, d := it.Next()
		e := u.Find(k)
		if e != nil {
			if f == nil {
				v.Delete(k)
				continue
			}
			c := f(d, e)
			if c == nil {
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

func (t *T) Iterator() Iterator {
	return Iterator{it: t.root.iterator()}
}

func (t *T) Equals(u *T) bool {
	if t == u {
		return true
	}
	if t.Size() != u.Size() {
		return false
	}
	return t.root.equals(u.root)
}

func (t *T) String() string {
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
		b.WriteString(fmt.Sprint(v))
	}
	return b.String()
}

func (t *node32) equals(u *node32) bool {
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

func (t *T) Equiv(u *T, eqv func(x, y interface{}) bool) bool {
	if t == u {
		return true
	}
	if t.Size() != u.Size() {
		return false
	}
	return t.root.equiv(u.root, eqv)
}

func (t *node32) equiv(u *node32, eqv func(x, y interface{}) bool) bool {
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

type iterator struct {
	parents []*node32
}

type Iterator struct {
	it iterator
}

func (it *Iterator) Next() (int32, interface{}) {
	x := it.it.next()
	if x == nil {
		return NOT_KEY32, nil
	}
	return x.key, x.data
}

func (it *Iterator) Done() bool {
	return len(it.it.parents) == 0
}

func (t *node32) iterator() iterator {
	if t == nil {
		return iterator{}
	}
	it := iterator{parents: make([]*node32, 0, int(t.height()))}
	it.leftmost(t)
	return it
}

func (it *iterator) leftmost(t *node32) {
	for t != nil {
		it.parents = append(it.parents, t)
		t = t.left
	}
}

func (it *iterator) done() bool {
	return len(it.parents) == 0
}

func (it *iterator) next() *node32 {
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
// If t is empty, then (NOT_KEY32, nil) is returned.
func (t *T) Min() (k int32, d interface{}) {
	return t.root.min().nilOrKeyAndData()
}

// Max returns the maximum element of t.
// If t is empty, then (NOT_KEY32, nil) is returned.
func (t *T) Max() (k int32, d interface{}) {
	return t.root.max().nilOrKeyAndData()
}

// Glb returns the greatest-lower-bound-exclusive of x and the associated
// data.  If x has no glb in the tree, then (NOT_KEY32, nil) is returned.
func (t *T) Glb(x int32) (k int32, d interface{}) {
	return t.root.glb(x, false).nilOrKeyAndData()
}

// GlbEq returns the greatest-lower-bound-inclusive of x and the associated
// data.  If x has no glbEQ in the tree, then (NOT_KEY32, nil) is returned.
func (t *T) GlbEq(x int32) (k int32, d interface{}) {
	return t.root.glb(x, true).nilOrKeyAndData()
}

// Lub returns the least-upper-bound-exclusive of x and the associated
// data.  If x has no lub in the tree, then (NOT_KEY32, nil) is returned.
func (t *T) Lub(x int32) (k int32, d interface{}) {
	return t.root.lub(x, false).nilOrKeyAndData()
}

// LubEq returns the least-upper-bound-inclusive of x and the associated
// data.  If x has no lubEq in the tree, then (NOT_KEY32, nil) is returned.
func (t *T) LubEq(x int32) (k int32, d interface{}) {
	return t.root.lub(x, true).nilOrKeyAndData()
}

func (t *node32) isLeaf() bool {
	return t.left == nil && t.right == nil && t.height_ == LEAF_HEIGHT
}

func (t *node32) visitInOrder(f func(int32, interface{})) {
	if t.left != nil {
		t.left.visitInOrder(f)
	}
	f(t.key, t.data)
	if t.right != nil {
		t.right.visitInOrder(f)
	}
}

func (t *node32) find(key int32) *node32 {
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

func (t *node32) min() *node32 {
	if t == nil {
		return t
	}
	for t.left != nil {
		t = t.left
	}
	return t
}

func (t *node32) max() *node32 {
	if t == nil {
		return t
	}
	for t.right != nil {
		t = t.right
	}
	return t
}

func (t *node32) glb(key int32, allow_eq bool) *node32 {
	var best *node32 = nil
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

func (t *node32) lub(key int32, allow_eq bool) *node32 {
	var best *node32 = nil
	for t != nil {
		if key >= t.key {
			if allow_eq && key == t.key {
				return t
			}
			// t is too small, lub is to right.
			t = t.right
		} else {
			// t is a upper bound, record it and seek a better one.
			best = t
			t = t.left
		}
	}
	return best
}

func (t *node32) aInsert(x int32) (newroot, newnode, oldnode *node32) {
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
			n := makeNode(x)
			t.left = n
			newnode = n
			newroot = t
			t.height_ = 2 // was balanced w/ 0, sibling is height 0 or 1
			return
		}
		var new_l *node32
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
			n := makeNode(x)
			t.right = n
			newnode = n
			newroot = t
			t.height_ = 2 // was balanced w/ 0, sibling is height 0 or 1
			return
		}
		var new_r *node32
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

func (t *node32) aDelete(key int32) (deleted, newSubTree *node32) {
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

func (t *node32) aDeleteMin() (deleted, newSubTree *node32) {
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

func (t *node32) aDeleteMax() (deleted, newSubTree *node32) {
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

func (t *node32) aRebalanceAfterLeftDeletion(oldLeftHeight int8, tleft *node32) *node32 {
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

func (t *node32) aRebalanceAfterRightDeletion(oldRightHeight int8, tright *node32) *node32 {
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
func (t *node32) aRightIsHigh(newnode *node32) *node32 {
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
func (t *node32) aLeftIsHigh(newnode *node32) *node32 {
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
func (t *node32) rightToRoot() *node32 {
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
func (t *node32) leftToRoot() *node32 {
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

func max(a, b int8) int8 {
	if a > b {
		return a
	}
	return b
}

func (t *node32) copy() *node32 {
	u := *t
	return &u
}
