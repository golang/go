// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"slices"
	"testing"
)

func TestAll(t *testing.T) {
	for size := 0; size < 10; size++ {
		m := make(map[int]int)
		for i := range size {
			m[i] = i
		}
		cnt := 0
		for i, v := range All(m) {
			v1, ok := m[i]
			if !ok || v != v1 {
				t.Errorf("at iteration %d got %d, %d want %d, %d", cnt, i, v, i, v1)
			}
			cnt++
		}
		if cnt != size {
			t.Errorf("read %d values expected %d", cnt, size)
		}
	}
}

func TestKeys(t *testing.T) {
	for size := 0; size < 10; size++ {
		var want []int
		m := make(map[int]int)
		for i := range size {
			m[i] = i
			want = append(want, i)
		}

		var got1 []int
		for k := range Keys(m) {
			got1 = append(got1, k)
		}
		slices.Sort(got1)
		if !slices.Equal(got1, want) {
			t.Errorf("Keys(%v) = %v, want %v", m, got1, want)
		}
	}
}

func TestValues(t *testing.T) {
	for size := 0; size < 10; size++ {
		var want []int
		m := make(map[int]int)
		for i := range size {
			m[i] = i
			want = append(want, i)
		}

		var got1 []int
		for v := range Values(m) {
			got1 = append(got1, v)
		}
		slices.Sort(got1)
		if !slices.Equal(got1, want) {
			t.Errorf("Values(%v) = %v, want %v", m, got1, want)
		}
	}
}

func testSeq(yield func(int, int) bool) {
	for i := 0; i < 10; i += 2 {
		if !yield(i, i+1) {
			return
		}
	}
}

var testSeqResult = map[int]int{
	0: 1,
	2: 3,
	4: 5,
	6: 7,
	8: 9,
}

func TestInsert(t *testing.T) {
	got := map[int]int{
		1: 1,
		2: 1,
	}
	Insert(got, testSeq)

	want := map[int]int{
		1: 1,
		2: 1,
	}
	for i, v := range testSeqResult {
		want[i] = v
	}

	if !Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCollect(t *testing.T) {
	m := map[int]int{
		0: 1,
		2: 3,
		4: 5,
		6: 7,
		8: 9,
	}
	got := Collect(All(m))
	if !Equal(got, m) {
		t.Errorf("got %v, want %v", got, m)
	}
}
