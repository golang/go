// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"exp/locale/collate"
	"strconv"
	"testing"
)

type entryTest struct {
	f   func(in []int) (uint32, error)
	arg []int
	val uint32
}

// makeList returns a list of entries of length n+2, with n normal
// entries plus a leading and trailing anchor.
func makeList(n int) []*entry {
	es := make([]*entry, n+2)
	weights := []rawCE{{w: []int{100, 20, 5, 0}}}
	for i := range es {
		runes := []rune{rune(i)}
		es[i] = &entry{
			runes: runes,
			elems: weights,
		}
		weights = nextWeight(collate.Primary, weights)
	}
	for i := 1; i < len(es); i++ {
		es[i-1].next = es[i]
		es[i].prev = es[i-1]
		_, es[i-1].level = compareWeights(es[i-1].elems, es[i].elems)
	}
	es[0].exclude = true
	es[0].logical = firstAnchor
	es[len(es)-1].exclude = true
	es[len(es)-1].logical = lastAnchor
	return es
}

func TestNextIndexed(t *testing.T) {
	const n = 5
	es := makeList(n)
	for i := int64(0); i < 1<<n; i++ {
		mask := strconv.FormatInt(i+(1<<n), 2)
		for i, c := range mask {
			es[i].exclude = c == '1'
		}
		e := es[0]
		for i, c := range mask {
			if c == '0' {
				e, _ = e.nextIndexed()
				if e != es[i] {
					t.Errorf("%d: expected entry %d; found %d", i, es[i].elems, e.elems)
				}
			}
		}
		if e, _ = e.nextIndexed(); e != nil {
			t.Errorf("%d: expected nil entry; found %d", i, e.elems)
		}
	}
}

func TestRemove(t *testing.T) {
	const n = 5
	for i := int64(0); i < 1<<n; i++ {
		es := makeList(n)
		mask := strconv.FormatInt(i+(1<<n), 2)
		for i, c := range mask {
			if c == '0' {
				es[i].remove()
			}
		}
		e := es[0]
		for i, c := range mask {
			if c == '1' {
				if e != es[i] {
					t.Errorf("%d: expected entry %d; found %d", i, es[i].elems, e.elems)
				}
				e, _ = e.nextIndexed()
			}
		}
		if e != nil {
			t.Errorf("%d: expected nil entry; found %d", i, e.elems)
		}
	}
}

// nextPerm generates the next permutation of the array.  The starting
// permutation is assumed to be a list of integers sorted in increasing order.
// It returns false if there are no more permuations left.
func nextPerm(a []int) bool {
	i := len(a) - 2
	for ; i >= 0; i-- {
		if a[i] < a[i+1] {
			break
		}
	}
	if i < 0 {
		return false
	}
	for j := len(a) - 1; j >= i; j-- {
		if a[j] > a[i] {
			a[i], a[j] = a[j], a[i]
			break
		}
	}
	for j := i + 1; j < (len(a)+i+1)/2; j++ {
		a[j], a[len(a)+i-j] = a[len(a)+i-j], a[j]
	}
	return true
}

func TestInsertAfter(t *testing.T) {
	const n = 5
	orig := makeList(n)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i + 1
	}
	for ok := true; ok; ok = nextPerm(perm) {
		es := makeList(n)
		last := es[0]
		for _, i := range perm {
			last.insertAfter(es[i])
			last = es[i]
		}
		for _, e := range es {
			e.elems = es[0].elems
		}
		e := es[0]
		for _, i := range perm {
			e, _ = e.nextIndexed()
			if e.runes[0] != orig[i].runes[0] {
				t.Errorf("%d:%d: expected entry %X; found %X", perm, i, orig[i].runes, e.runes)
				break
			}
		}
	}
}

func TestInsertBefore(t *testing.T) {
	const n = 5
	orig := makeList(n)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i + 1
	}
	for ok := true; ok; ok = nextPerm(perm) {
		es := makeList(n)
		last := es[len(es)-1]
		for _, i := range perm {
			last.insertBefore(es[i])
			last = es[i]
		}
		for _, e := range es {
			e.elems = es[0].elems
		}
		e := es[0]
		for i := n - 1; i >= 0; i-- {
			e, _ = e.nextIndexed()
			if e.runes[0] != rune(perm[i]) {
				t.Errorf("%d:%d: expected entry %X; found %X", perm, i, orig[i].runes, e.runes)
				break
			}
		}
	}
}

type entryLessTest struct {
	a, b *entry
	res  bool
}

var (
	w1 = []rawCE{{w: []int{100, 20, 5, 5}}}
	w2 = []rawCE{{w: []int{101, 20, 5, 5}}}
)

var entryLessTests = []entryLessTest{
	{&entry{str: "a", elems: w1},
		&entry{str: "a", elems: w1},
		false,
	},
	{&entry{str: "a", elems: w1},
		&entry{str: "a", elems: w2},
		true,
	},
	{&entry{str: "a", elems: w1},
		&entry{str: "b", elems: w1},
		true,
	},
	{&entry{str: "a", elems: w2},
		&entry{str: "a", elems: w1},
		false,
	},
	{&entry{str: "c", elems: w1},
		&entry{str: "b", elems: w1},
		false,
	},
	{&entry{str: "a", elems: w1, logical: firstAnchor},
		&entry{str: "a", elems: w1},
		true,
	},
	{&entry{str: "a", elems: w1},
		&entry{str: "b", elems: w1, logical: firstAnchor},
		false,
	},
	{&entry{str: "b", elems: w1},
		&entry{str: "a", elems: w1, logical: lastAnchor},
		true,
	},
	{&entry{str: "a", elems: w1, logical: lastAnchor},
		&entry{str: "c", elems: w1},
		false,
	},
}

func TestEntryLess(t *testing.T) {
	for i, tt := range entryLessTests {
		if res := entryLess(tt.a, tt.b); res != tt.res {
			t.Errorf("%d: was %v; want %v", i, res, tt.res)
		}
	}
}
