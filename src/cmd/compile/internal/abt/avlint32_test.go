// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abt

import (
	"fmt"
	"strconv"
	"testing"
)

func makeTree(te *testing.T, x []int32, check bool) (t *T, k int, min, max int32) {
	t = &T{}
	k = 0
	min = int32(0x7fffffff)
	max = int32(-0x80000000)
	history := []*T{}

	for _, d := range x {
		d = d + d // double everything for Glb/Lub testing.

		if check {
			history = append(history, t.Copy())
		}

		t.Insert(d, stringer(fmt.Sprintf("%v", d)))

		k++
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}

		if !check {
			continue
		}

		for j, old := range history {
			s, i := old.wellFormed()
			if s != "" {
				te.Errorf("Old tree consistency problem %v at k=%d, j=%d, old=\n%v, t=\n%v", s, k, j, old.DebugString(), t.DebugString())
				return
			}
			if i != j {
				te.Errorf("Wrong tree size %v, expected %v for old %v", i, j, old.DebugString())
			}
		}
		s, i := t.wellFormed()
		if s != "" {
			te.Errorf("Tree consistency problem at %v", s)
			return
		}
		if i != k {
			te.Errorf("Wrong tree size %v, expected %v for %v", i, k, t.DebugString())
			return
		}
		if t.Size() != k {
			te.Errorf("Wrong t.Size() %v, expected %v for %v", t.Size(), k, t.DebugString())
			return
		}
	}
	return
}

func applicInsert(te *testing.T, x []int32) {
	makeTree(te, x, true)
}

func applicFind(te *testing.T, x []int32) {
	t, _, _, _ := makeTree(te, x, false)

	for _, d := range x {
		d = d + d // double everything for Glb/Lub testing.
		s := fmt.Sprintf("%v", d)
		f := t.Find(d)

		// data
		if s != fmt.Sprint(f) {
			te.Errorf("s(%v) != f(%v)", s, f)
		}
	}
}

func applicBounds(te *testing.T, x []int32) {
	t, _, min, max := makeTree(te, x, false)
	for _, d := range x {
		d = d + d // double everything for Glb/Lub testing.
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
		if s != fmt.Sprint(g) {
			te.Errorf("s(%v) != g(%v)", s, g)
		}
		if s != fmt.Sprint(l) {
			te.Errorf("s(%v) != l(%v)", s, l)
		}
		if s != fmt.Sprint(ge) {
			te.Errorf("s(%v) != ge(%v)", s, ge)
		}
		if s != fmt.Sprint(le) {
			te.Errorf("s(%v) != le(%v)", s, le)
		}
	}

	for _, d := range x {
		d = d + d // double everything for Glb/Lub testing.
		s := fmt.Sprintf("%v", d)
		kge, ge := t.GlbEq(d + 1)
		kle, le := t.LubEq(d - 1)
		if d != kge {
			te.Errorf("d(%v) != kge(%v)", d, kge)
		}
		if d != kle {
			te.Errorf("d(%v) != kle(%v)", d, kle)
		}
		if s != fmt.Sprint(ge) {
			te.Errorf("s(%v) != ge(%v)", s, ge)
		}
		if s != fmt.Sprint(le) {
			te.Errorf("s(%v) != le(%v)", s, le)
		}
	}

	kg, g := t.Glb(min)
	kge, ge := t.GlbEq(min - 1)
	kl, l := t.Lub(max)
	kle, le := t.LubEq(max + 1)
	fmin := t.Find(min - 1)
	fmax := t.Find(max + 1)

	if kg != NOT_KEY32 || kge != NOT_KEY32 || kl != NOT_KEY32 || kle != NOT_KEY32 {
		te.Errorf("Got non-error-key for missing query")
	}

	if g != nil || ge != nil || l != nil || le != nil || fmin != nil || fmax != nil {
		te.Errorf("Got non-error-data for missing query")
	}
}

func applicDeleteMin(te *testing.T, x []int32) {
	t, _, _, _ := makeTree(te, x, false)
	_, size := t.wellFormed()
	history := []*T{}
	for !t.IsEmpty() {
		k, _ := t.Min()
		history = append(history, t.Copy())
		kd, _ := t.DeleteMin()
		if kd != k {
			te.Errorf("Deleted minimum key %v not equal to minimum %v", kd, k)
		}
		for j, old := range history {
			s, i := old.wellFormed()
			if s != "" {
				te.Errorf("Tree consistency problem %s at old after DeleteMin, old=\n%stree=\n%v", s, old.DebugString(), t.DebugString())
				return
			}
			if i != len(x)-j {
				te.Errorf("Wrong old tree size %v, expected %v after DeleteMin, old=\n%vtree\n%v", i, len(x)-j, old.DebugString(), t.DebugString())
				return
			}
		}
		size--
		s, i := t.wellFormed()
		if s != "" {
			te.Errorf("Tree consistency problem at %v after DeleteMin, tree=\n%v", s, t.DebugString())
			return
		}
		if i != size {
			te.Errorf("Wrong tree size %v, expected %v after DeleteMin", i, size)
			return
		}
		if t.Size() != size {
			te.Errorf("Wrong t.Size() %v, expected %v for %v", t.Size(), i, t.DebugString())
			return
		}
	}
}

func applicDeleteMax(te *testing.T, x []int32) {
	t, _, _, _ := makeTree(te, x, false)
	_, size := t.wellFormed()
	history := []*T{}

	for !t.IsEmpty() {
		k, _ := t.Max()
		history = append(history, t.Copy())
		kd, _ := t.DeleteMax()
		if kd != k {
			te.Errorf("Deleted maximum key %v not equal to maximum %v", kd, k)
		}

		for j, old := range history {
			s, i := old.wellFormed()
			if s != "" {
				te.Errorf("Tree consistency problem %s at old after DeleteMin, old=\n%stree=\n%v", s, old.DebugString(), t.DebugString())
				return
			}
			if i != len(x)-j {
				te.Errorf("Wrong old tree size %v, expected %v after DeleteMin, old=\n%vtree\n%v", i, len(x)-j, old.DebugString(), t.DebugString())
				return
			}
		}

		size--
		s, i := t.wellFormed()
		if s != "" {
			te.Errorf("Tree consistency problem at %v after DeleteMax, tree=\n%v", s, t.DebugString())
			return
		}
		if i != size {
			te.Errorf("Wrong tree size %v, expected %v after DeleteMax", i, size)
			return
		}
		if t.Size() != size {
			te.Errorf("Wrong t.Size() %v, expected %v for %v", t.Size(), i, t.DebugString())
			return
		}
	}
}

func applicDelete(te *testing.T, x []int32) {
	t, _, _, _ := makeTree(te, x, false)
	_, size := t.wellFormed()
	history := []*T{}

	missing := t.Delete(11)
	if missing != nil {
		te.Errorf("Returned a value when there should have been none, %v", missing)
		return
	}

	s, i := t.wellFormed()
	if s != "" {
		te.Errorf("Tree consistency problem at %v after delete of missing value, tree=\n%v", s, t.DebugString())
		return
	}
	if size != i {
		te.Errorf("Delete of missing data should not change tree size, expected %d, got %d", size, i)
		return
	}

	for _, d := range x {
		d += d // double
		vWant := fmt.Sprintf("%v", d)
		history = append(history, t.Copy())
		v := t.Delete(d)

		for j, old := range history {
			s, i := old.wellFormed()
			if s != "" {
				te.Errorf("Tree consistency problem %s at old after DeleteMin, old=\n%stree=\n%v", s, old.DebugString(), t.DebugString())
				return
			}
			if i != len(x)-j {
				te.Errorf("Wrong old tree size %v, expected %v after DeleteMin, old=\n%vtree\n%v", i, len(x)-j, old.DebugString(), t.DebugString())
				return
			}
		}

		if v.(*sstring).s != vWant {
			te.Errorf("Deleted %v expected %v but got %v", d, vWant, v)
			return
		}
		size--
		s, i := t.wellFormed()
		if s != "" {
			te.Errorf("Tree consistency problem at %v after Delete %d, tree=\n%v", s, d, t.DebugString())
			return
		}
		if i != size {
			te.Errorf("Wrong tree size %v, expected %v after Delete", i, size)
			return
		}
		if t.Size() != size {
			te.Errorf("Wrong t.Size() %v, expected %v for %v", t.Size(), i, t.DebugString())
			return
		}
	}

}

func applicIterator(te *testing.T, x []int32) {
	t, _, _, _ := makeTree(te, x, false)
	it := t.Iterator()
	for !it.Done() {
		k0, d0 := it.Next()
		k1, d1 := t.DeleteMin()
		if k0 != k1 || d0 != d1 {
			te.Errorf("Iterator and deleteMin mismatch, k0, k1, d0, d1 = %v, %v, %v, %v", k0, k1, d0, d1)
			return
		}
	}
	if t.Size() != 0 {
		te.Errorf("Iterator ended early, remaining tree = \n%s", t.DebugString())
		return
	}
}

func equiv(a, b any) bool {
	sa, sb := a.(*sstring), b.(*sstring)
	return *sa == *sb
}

func applicEquals(te *testing.T, x, y []int32) {
	t, _, _, _ := makeTree(te, x, false)
	u, _, _, _ := makeTree(te, y, false)
	if !t.Equiv(t, equiv) {
		te.Errorf("Equiv failure, t == t, =\n%v", t.DebugString())
		return
	}
	if !t.Equiv(t.Copy(), equiv) {
		te.Errorf("Equiv failure, t == t.Copy(), =\n%v", t.DebugString())
		return
	}
	if !t.Equiv(u, equiv) {
		te.Errorf("Equiv failure, t == u, =\n%v", t.DebugString())
		return
	}
	v := t.Copy()

	v.DeleteMax()
	if t.Equiv(v, equiv) {
		te.Errorf("!Equiv failure, t != v, =\n%v\nand%v\n", t.DebugString(), v.DebugString())
		return
	}

	if v.Equiv(u, equiv) {
		te.Errorf("!Equiv failure, v != u, =\n%v\nand%v\n", v.DebugString(), u.DebugString())
		return
	}

}

func tree(x []int32) *T {
	t := &T{}
	for _, d := range x {
		t.Insert(d, stringer(fmt.Sprintf("%v", d)))
	}
	return t
}

func treePlus1(x []int32) *T {
	t := &T{}
	for _, d := range x {
		t.Insert(d, stringer(fmt.Sprintf("%v", d+1)))
	}
	return t
}
func TestApplicInsert(t *testing.T) {
	applicInsert(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
	applicInsert(t, []int32{1, 2, 3, 4})
	applicInsert(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	applicInsert(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicInsert(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicInsert(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicInsert(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	applicInsert(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}

func TestApplicFind(t *testing.T) {
	applicFind(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
	applicFind(t, []int32{1, 2, 3, 4})
	applicFind(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	applicFind(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicFind(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicFind(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicFind(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	applicFind(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}

func TestBounds(t *testing.T) {
	applicBounds(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
	applicBounds(t, []int32{1, 2, 3, 4})
	applicBounds(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	applicBounds(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicBounds(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicBounds(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicBounds(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	applicBounds(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}
func TestDeleteMin(t *testing.T) {
	applicDeleteMin(t, []int32{1, 2, 3, 4})
	applicDeleteMin(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
	applicDeleteMin(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	applicDeleteMin(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicDeleteMin(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicDeleteMin(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicDeleteMin(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	applicDeleteMin(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}
func TestDeleteMax(t *testing.T) {
	applicDeleteMax(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
	applicDeleteMax(t, []int32{1, 2, 3, 4})
	applicDeleteMax(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	applicDeleteMax(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicDeleteMax(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicDeleteMax(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicDeleteMax(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	applicDeleteMax(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}
func TestDelete(t *testing.T) {
	applicDelete(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
	applicDelete(t, []int32{1, 2, 3, 4})
	applicDelete(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	applicDelete(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicDelete(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicDelete(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicDelete(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	applicDelete(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}
func TestIterator(t *testing.T) {
	applicIterator(t, []int32{1, 2, 3, 4})
	applicIterator(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	applicIterator(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25})
	applicIterator(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicIterator(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicIterator(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicIterator(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24})
	applicIterator(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}
func TestEquals(t *testing.T) {
	applicEquals(t, []int32{1, 2, 3, 4}, []int32{4, 3, 2, 1})

	applicEquals(t, []int32{24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25},
		[]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})
	applicEquals(t, []int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		[]int32{25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	applicEquals(t, []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24},
		[]int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2})
}

func first(x, y any) any {
	return x
}
func second(x, y any) any {
	return y
}
func alwaysNil(x, y any) any {
	return nil
}
func smaller(x, y any) any {
	xi, _ := strconv.Atoi(fmt.Sprint(x))
	yi, _ := strconv.Atoi(fmt.Sprint(y))
	if xi < yi {
		return x
	}
	return y
}
func assert(t *testing.T, expected, got *T, what string) {
	s, _ := got.wellFormed()
	if s != "" {
		t.Errorf("Tree consistency problem %v for 'got' in assert for %s, tree=\n%v", s, what, got.DebugString())
		return
	}

	if !expected.Equiv(got, equiv) {
		t.Errorf("%s fail, expected\n%vgot\n%v\n", what, expected.DebugString(), got.DebugString())
	}
}

func TestSetOps(t *testing.T) {
	A := tree([]int32{1, 2, 3, 4})
	B := tree([]int32{3, 4, 5, 6, 7})

	AIB := tree([]int32{3, 4})
	ADB := tree([]int32{1, 2})
	BDA := tree([]int32{5, 6, 7})
	AUB := tree([]int32{1, 2, 3, 4, 5, 6, 7})
	AXB := tree([]int32{1, 2, 5, 6, 7})

	aib1 := A.Intersection(B, first)
	assert(t, AIB, aib1, "aib1")
	if A.Find(3) != aib1.Find(3) {
		t.Errorf("Failed aliasing/reuse check, A/aib1")
	}
	aib2 := A.Intersection(B, second)
	assert(t, AIB, aib2, "aib2")
	if B.Find(3) != aib2.Find(3) {
		t.Errorf("Failed aliasing/reuse check, B/aib2")
	}
	aib3 := B.Intersection(A, first)
	assert(t, AIB, aib3, "aib3")
	if A.Find(3) != aib3.Find(3) {
		// A is smaller, intersection favors reuse from smaller when function is "first"
		t.Errorf("Failed aliasing/reuse check, A/aib3")
	}
	aib4 := B.Intersection(A, second)
	assert(t, AIB, aib4, "aib4")
	if A.Find(3) != aib4.Find(3) {
		t.Errorf("Failed aliasing/reuse check, A/aib4")
	}

	aub1 := A.Union(B, first)
	assert(t, AUB, aub1, "aub1")
	if B.Find(3) != aub1.Find(3) {
		// B is larger, union favors reuse from larger when function is "first"
		t.Errorf("Failed aliasing/reuse check, A/aub1")
	}
	aub2 := A.Union(B, second)
	assert(t, AUB, aub2, "aub2")
	if B.Find(3) != aub2.Find(3) {
		t.Errorf("Failed aliasing/reuse check, B/aub2")
	}
	aub3 := B.Union(A, first)
	assert(t, AUB, aub3, "aub3")
	if B.Find(3) != aub3.Find(3) {
		t.Errorf("Failed aliasing/reuse check, B/aub3")
	}
	aub4 := B.Union(A, second)
	assert(t, AUB, aub4, "aub4")
	if A.Find(3) != aub4.Find(3) {
		t.Errorf("Failed aliasing/reuse check, A/aub4")
	}

	axb1 := A.Union(B, alwaysNil)
	assert(t, AXB, axb1, "axb1")
	axb2 := B.Union(A, alwaysNil)
	assert(t, AXB, axb2, "axb2")

	adb := A.Difference(B, alwaysNil)
	assert(t, ADB, adb, "adb")
	bda := B.Difference(A, nil)
	assert(t, BDA, bda, "bda")

	Ap1 := treePlus1([]int32{1, 2, 3, 4})

	ada1_1 := A.Difference(Ap1, smaller)
	assert(t, A, ada1_1, "ada1_1")
	ada1_2 := Ap1.Difference(A, smaller)
	assert(t, A, ada1_2, "ada1_2")

}

type sstring struct {
	s string
}

func (s *sstring) String() string {
	return s.s
}

func stringer(s string) any {
	return &sstring{s}
}

// wellFormed ensures that a red-black tree meets
// all of its invariants and returns a string identifying
// the first problem encountered. If there is no problem
// then the returned string is empty. The size is also
// returned to allow comparison of calculated tree size
// with expected.
func (t *T) wellFormed() (s string, i int) {
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
func (t *node32) wellFormedSubtree(parent *node32, keyMin, keyMax int32) (s string, i int) {
	i = -1 // initialize to a failing value
	s = "" // s is the reason for failure; empty means okay.

	if keyMin >= t.key {
		s = " min >= t.key"
		return
	}

	if keyMax <= t.key {
		s = " max <= t.key"
		return
	}

	l := t.left
	r := t.right

	lh := l.height()
	rh := r.height()
	mh := max(lh, rh)
	th := t.height()
	dh := lh - rh
	if dh < 0 {
		dh = -dh
	}
	if dh > 1 {
		s = fmt.Sprintf(" dh > 1, t=%d", t.key)
		return
	}

	if l == nil && r == nil {
		if th != LEAF_HEIGHT {
			s = " leaf height wrong"
			return
		}
	}

	if th != mh+1 {
		s = " th != mh + 1"
		return
	}

	if l != nil {
		if th <= lh {
			s = " t.height <= l.height"
		} else if th > 2+lh {
			s = " t.height > 2+l.height"
		} else if t.key <= l.key {
			s = " t.key <= l.key"
		}
		if s != "" {
			return
		}

	}

	if r != nil {
		if th <= rh {
			s = " t.height <= r.height"
		} else if th > 2+rh {
			s = " t.height > 2+r.height"
		} else if t.key >= r.key {
			s = " t.key >= r.key"
		}
		if s != "" {
			return
		}
	}

	ii := 1
	if l != nil {
		res, il := l.wellFormedSubtree(t, keyMin, t.key)
		if res != "" {
			s = ".L" + res
			return
		}
		ii += il
	}
	if r != nil {
		res, ir := r.wellFormedSubtree(t, t.key, keyMax)
		if res != "" {
			s = ".R" + res
			return
		}
		ii += ir
	}
	i = ii
	return
}

func (t *T) DebugString() string {
	if t.root == nil {
		return ""
	}
	return t.root.DebugString(0)
}

// DebugString prints the tree with nested information
// to allow an eyeball check on the tree balance.
func (t *node32) DebugString(indent int) string {
	s := ""
	if t.left != nil {
		s = s + t.left.DebugString(indent+1)
	}
	for i := 0; i < indent; i++ {
		s = s + "    "
	}
	s = s + fmt.Sprintf("%v=%v:%d\n", t.key, t.data, t.height_)
	if t.right != nil {
		s = s + t.right.DebugString(indent+1)
	}
	return s
}
