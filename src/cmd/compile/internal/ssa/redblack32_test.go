// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"testing"
)

type sstring string

func (s sstring) String() string {
	return string(s)
}

// wellFormed ensures that a red-black tree meets
// all of its invariants and returns a string identifying
// the first problem encountered. If there is no problem
// then the returned string is empty. The size is also
// returned to allow comparison of calculated tree size
// with expected.
func (t *RBTint32) wellFormed() (s string, i int) {
	if t.root == nil {
		s = ""
		i = 0
		return
	}
	return t.root.wellFormedSubtree(nil, -0x80000000, 0x7fffffff)
}

// wellFormedSubtree ensures that a red-black subtree meets
// all of its invariants and returns a string identifying
// the first problem encountered. If there is no problem
// then the returned string is empty. The size is also
// returned to allow comparison of calculated tree size
// with expected.
func (t *node32) wellFormedSubtree(parent *node32, min, max int32) (s string, i int) {
	i = -1 // initialize to a failing value
	s = "" // s is the reason for failure; empty means okay.

	if t.parent != parent {
		s = "t.parent != parent"
		return
	}

	if min >= t.key {
		s = "min >= t.key"
		return
	}

	if max <= t.key {
		s = "max <= t.key"
		return
	}

	l := t.left
	r := t.right
	if l == nil && r == nil {
		if t.rank != rankLeaf {
			s = "leaf rank wrong"
			return
		}
	}
	if l != nil {
		if t.rank < l.rank {
			s = "t.rank < l.rank"
		} else if t.rank > 1+l.rank {
			s = "t.rank > 1+l.rank"
		} else if t.rank <= l.maxChildRank() {
			s = "t.rank <= l.maxChildRank()"
		} else if t.key <= l.key {
			s = "t.key <= l.key"
		}
		if s != "" {
			return
		}
	} else {
		if t.rank != 1 {
			s = "t w/ left nil has rank != 1"
			return
		}
	}
	if r != nil {
		if t.rank < r.rank {
			s = "t.rank < r.rank"
		} else if t.rank > 1+r.rank {
			s = "t.rank > 1+r.rank"
		} else if t.rank <= r.maxChildRank() {
			s = "t.rank <= r.maxChildRank()"
		} else if t.key >= r.key {
			s = "t.key >= r.key"
		}
		if s != "" {
			return
		}
	} else {
		if t.rank != 1 {
			s = "t w/ right nil has rank != 1"
			return
		}
	}
	ii := 1
	if l != nil {
		res, il := l.wellFormedSubtree(t, min, t.key)
		if res != "" {
			s = "L." + res
			return
		}
		ii += il
	}
	if r != nil {
		res, ir := r.wellFormedSubtree(t, t.key, max)
		if res != "" {
			s = "R." + res
			return
		}
		ii += ir
	}
	i = ii
	return
}

func (t *RBTint32) DebugString() string {
	if t.root == nil {
		return ""
	}
	return t.root.DebugString()
}

// DebugString prints the tree with nested information
// to allow an eyeball check on the tree balance.
func (t *node32) DebugString() string {
	s := ""
	if t.left != nil {
		s += "["
		s += t.left.DebugString()
		s += "]"
	}
	s += fmt.Sprintf("%v=%v:%d", t.key, t.data, t.rank)
	if t.right != nil {
		s += "["
		s += t.right.DebugString()
		s += "]"
	}
	return s
}

func allRBT32Ops(te *testing.T, x []int32) {
	t := &RBTint32{}
	for i, d := range x {
		x[i] = d + d // Double everything for glb/lub testing
	}

	// fmt.Printf("Inserting double of %v", x)
	k := 0
	min := int32(0x7fffffff)
	max := int32(-0x80000000)
	for _, d := range x {
		if d < min {
			min = d
		}

		if d > max {
			max = d
		}

		t.Insert(d, sstring(fmt.Sprintf("%v", d)))
		k++
		s, i := t.wellFormed()
		if i != k {
			te.Errorf("Wrong tree size %v, expected %v for %v", i, k, t.DebugString())
		}
		if s != "" {
			te.Errorf("Tree consistency problem at %v", s)
			return
		}
	}

	oops := false

	for _, d := range x {
		s := fmt.Sprintf("%v", d)
		f := t.Find(d)

		// data
		if s != fmt.Sprintf("%v", f) {
			te.Errorf("s(%v) != f(%v)", s, f)
			oops = true
		}
	}

	if !oops {
		for _, d := range x {
			s := fmt.Sprintf("%v", d)

			kg, g := t.Glb(d + 1)
			kge, ge := t.GlbEq(d)
			kl, l := t.Lub(d - 1)
			kle, le := t.LubEq(d)

			// keys
			if d != kg {
				te.Errorf("d(%v) != kg(%v)", d, kg)
			}
			if d != kl {
				te.Errorf("d(%v) != kl(%v)", d, kl)
			}
			if d != kge {
				te.Errorf("d(%v) != kge(%v)", d, kge)
			}
			if d != kle {
				te.Errorf("d(%v) != kle(%v)", d, kle)
			}
			// data
			if s != fmt.Sprintf("%v", g) {
				te.Errorf("s(%v) != g(%v)", s, g)
			}
			if s != fmt.Sprintf("%v", l) {
				te.Errorf("s(%v) != l(%v)", s, l)
			}
			if s != fmt.Sprintf("%v", ge) {
				te.Errorf("s(%v) != ge(%v)", s, ge)
			}
			if s != fmt.Sprintf("%v", le) {
				te.Errorf("s(%v) != le(%v)", s, le)
			}
		}

		for _, d := range x {
			s := fmt.Sprintf("%v", d)
			kge, ge := t.GlbEq(d + 1)
			kle, le := t.LubEq(d - 1)
			if d != kge {
				te.Errorf("d(%v) != kge(%v)", d, kge)
			}
			if d != kle {
				te.Errorf("d(%v) != kle(%v)", d, kle)
			}
			if s != fmt.Sprintf("%v", ge) {
				te.Errorf("s(%v) != ge(%v)", s, ge)
			}
			if s != fmt.Sprintf("%v", le) {
				te.Errorf("s(%v) != le(%v)", s, le)
			}
		}

		kg, g := t.Glb(min)
		kge, ge := t.GlbEq(min - 1)
		kl, l := t.Lub(max)
		kle, le := t.LubEq(max + 1)
		fmin := t.Find(min - 1)
		fmax := t.Find(min + 11)

		if kg != 0 || kge != 0 || kl != 0 || kle != 0 {
			te.Errorf("Got non-zero-key for missing query")
		}

		if g != nil || ge != nil || l != nil || le != nil || fmin != nil || fmax != nil {
			te.Errorf("Got non-error-data for missing query")
		}

	}
}

func TestAllRBTreeOps(t *testing.T) {
	allRBT32Ops(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	allRBT32Ops(t, []int32{22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 3, 2, 1, 25, 24, 23, 12, 11, 10, 9, 8, 7, 6, 5, 4})
	allRBT32Ops(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	allRBT32Ops(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	allRBT32Ops(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
	allRBT32Ops(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
}
