// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt_test

import (
	"reflect"
	"testing"

	"logos/rt"
)

// TestFreezeSliceRetainsCapTail pins PR2 T_Zc (PR3 UDAS shares the
// mechanism): a preserved capacity must expose copied backing storage,
// not a zeroed tail. Small witness: 3 ints (<64 bytes).
func TestFreezeSliceRetainsCapTail(t *testing.T) {
	s := []int{10, 20, 30}
	view := s[:1]
	got := *rt.Freeze(&view)
	if len(got) != 1 || cap(got) != 3 {
		t.Fatalf("len, cap = %d, %d; want 1, 3", len(got), cap(got))
	}
	if !reflect.DeepEqual(got[:3], []int{10, 20, 30}) {
		t.Fatalf("full-cap view = %v, want [10 20 30]", got[:3])
	}
	empty := s[:0]
	z := *rt.Freeze(&empty)
	if len(z) != 0 || cap(z) != 3 {
		t.Fatalf("empty len, cap = %d, %d; want 0, 3", len(z), cap(z))
	}
	if !reflect.DeepEqual(z[:3], []int{10, 20, 30}) {
		t.Fatalf("empty full-cap view = %v, want [10 20 30]", z[:cap(z)])
	}
	s[1] = 99
	s[2] = 100
	if !reflect.DeepEqual(got[:3], []int{10, 20, 30}) {
		t.Fatalf("frozen tail changed after source write: %v", got[:3])
	}
	if !reflect.DeepEqual(z[:3], []int{10, 20, 30}) {
		t.Fatalf("empty frozen tail changed after source write: %v", z[:cap(z)])
	}
}

// TestFreezeSliceCapIsPartOfIdentity pins PR2 U0CP: headers that share a
// pointer and length but differ in capacity must copy separately.
func TestFreezeSliceCapIsPartOfIdentity(t *testing.T) {
	s := []int{10, 20, 30}
	src := struct {
		A []int
		B []int
	}{s[:1:1], s[:1:3]}
	got := *rt.Freeze(&src)
	if cap(got.A) != 1 || cap(got.B) != 3 {
		t.Fatalf("caps = (%d, %d), want (1, 3)", cap(got.A), cap(got.B))
	}
	if len(got.B) != 1 || !reflect.DeepEqual(got.B[:3], []int{10, 20, 30}) {
		t.Fatalf("B full-cap = %v len %d", got.B[:cap(got.B)], len(got.B))
	}
	rev := struct {
		A []int
		B []int
	}{s[:1:3], s[:1:1]}
	gotRev := *rt.Freeze(&rev)
	if cap(gotRev.A) != 3 || cap(gotRev.B) != 1 {
		t.Fatalf("reversed caps = (%d, %d), want (3, 1)", cap(gotRev.A), cap(gotRev.B))
	}
}

// TestFreezeIdenticalHeadersShareAndCycle pins the positive controls the
// tail/identity fix must not break: identical headers still share one
// copy, and cycles through elements still terminate.
func TestFreezeIdenticalHeadersShareAndCycle(t *testing.T) {
	s := []int{10, 20, 30}
	dup := struct {
		A []int
		B []int
	}{s[:1:3], s[:1:3]}
	gotDup := *rt.Freeze(&dup)
	if cap(gotDup.A) != 3 || cap(gotDup.B) != 3 {
		t.Fatalf("dup caps = (%d, %d), want (3, 3)", cap(gotDup.A), cap(gotDup.B))
	}
	if len(gotDup.A) == 0 || &gotDup.A[0] != &gotDup.B[0] {
		t.Fatal("identical headers diverged; want one shared copy")
	}
	type node struct {
		Next *node
		Vals []int
	}
	n := &node{Vals: []int{1, 2}[:1:2]}
	n.Next = n
	gotN := rt.Freeze(&n)
	if (*gotN).Next != *gotN {
		t.Fatal("self-cycle through struct does not point at the copy")
	}
	if cap((*gotN).Vals) != 2 || !reflect.DeepEqual((*gotN).Vals[:2], []int{1, 2}) {
		t.Fatalf("cycle payload tail = %v cap %d, want [1 2] cap 2", (*gotN).Vals[:cap((*gotN).Vals)], cap((*gotN).Vals))
	}
}
