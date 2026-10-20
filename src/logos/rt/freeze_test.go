// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Attack tests for M5-S6: Freeze deep-copies its argument, preserves
// cycles and sharing, keeps nils nil, and copies unexported fields.
// The .lgo const-view half of the slice is owned by
// test/logos/run/freeze.lgo and test/logos/compile/freeze_const.lgo;
// this file pins the runtime semantics in plain Go.
package rt_test

import (
	"reflect"
	"testing"

	"logos/rt"
)

type freezeChild struct {
	n    int
	note string
	next *freezeChild
}

type freezeConfig struct {
	name   string
	n      int
	tags   []string
	labels map[string]string
	child  *freezeChild
	other  *freezeChild
	any    any
	arr    [2]int
}

func freezeSample() *freezeConfig {
	return &freezeConfig{
		name:   "orig",
		n:      7,
		tags:   []string{"a", "b"},
		labels: map[string]string{"k": "v"},
		child:  &freezeChild{n: 1, note: "c1"},
		other:  &freezeChild{n: 2, note: "c2"},
		any:    &freezeChild{n: 3, note: "c3"},
		arr:    [2]int{4, 5},
	}
}

func TestFreezeNil(t *testing.T) {
	if got := rt.Freeze[int](nil); got != nil {
		t.Fatalf("Freeze(nil) = %v, want nil", got)
	}
}

func TestFreezeScalar(t *testing.T) {
	v := 42
	got := rt.Freeze(&v)
	if got == &v {
		t.Fatal("Freeze returned the input pointer; want a copy")
	}
	if *got != 42 {
		t.Fatalf("Freeze(&42) = %d, want 42", *got)
	}
}

func TestFreezeDeepNoAlias(t *testing.T) {
	src := freezeSample()
	got := rt.Freeze(src)
	if got == src {
		t.Fatal("Freeze returned the input pointer; want a copy")
	}
	if !reflect.DeepEqual(got, src) {
		t.Fatalf("Freeze(src) = %+v, want deep-equal %+v", got, src)
	}
	src.name = "changed"
	src.n = 99
	src.tags[0] = "CHANGED"
	src.labels["k"] = "CHANGED"
	src.labels["new"] = "CHANGED"
	src.child.n = 100
	src.child.note = "CHANGED"
	src.other.n = 200
	src.any.(*freezeChild).n = 300
	src.arr[0] = 400
	if got.name != "orig" || got.n != 7 {
		t.Fatalf("frozen scalars changed: %+v", got)
	}
	if !reflect.DeepEqual(got.tags, []string{"a", "b"}) {
		t.Fatalf("frozen tags changed: %q", got.tags)
	}
	if !reflect.DeepEqual(got.labels, map[string]string{"k": "v"}) {
		t.Fatalf("frozen labels changed: %v", got.labels)
	}
	if got.child.n != 1 || got.child.note != "c1" {
		t.Fatalf("frozen child changed: %+v", got.child)
	}
	if got.other.n != 2 || got.other.note != "c2" {
		t.Fatalf("frozen other changed: %+v", got.other)
	}
	if got.any.(*freezeChild).n != 3 {
		t.Fatalf("frozen any changed: %+v", got.any)
	}
	if got.arr != [2]int{4, 5} {
		t.Fatalf("frozen arr changed: %v", got.arr)
	}
}

func TestFreezeCycle(t *testing.T) {
	a := &freezeChild{n: 1}
	a.next = a
	got := rt.Freeze(a)
	if got == a {
		t.Fatal("Freeze returned the input pointer; want a copy")
	}
	if got.next != got {
		t.Fatal("frozen self-cycle does not point at the copy")
	}
	if got.n != 1 {
		t.Fatalf("frozen cycle payload = %d, want 1", got.n)
	}
}

func TestFreezeSharedAlias(t *testing.T) {
	shared := &freezeChild{n: 5, note: "shared"}
	src := &freezeConfig{child: &freezeChild{}, other: shared}
	src.child.next = shared
	got := rt.Freeze(src)
	if got.other == shared || got.child.next == shared {
		t.Fatal("frozen graph still aliases the source")
	}
	if got.other != got.child.next {
		t.Fatal("frozen graph lost sharing: two views of one child diverged")
	}
	if got.other.n != 5 || got.other.note != "shared" {
		t.Fatalf("frozen shared child = %+v, want {5 shared}", got.other)
	}
}

func TestFreezeNilInner(t *testing.T) {
	src := &freezeConfig{}
	got := rt.Freeze(src)
	if got.tags != nil || got.labels != nil || got.child != nil || got.other != nil || got.any != nil {
		t.Fatalf("frozen zero value grew Nils: %+v", got)
	}
}

func TestFreezeConcurrent(t *testing.T) {
	src := freezeSample()
	done := make(chan *freezeConfig, 8)
	for range 8 {
		go func() {
			done <- rt.Freeze(src)
		}()
	}
	for range 8 {
		got := <-done
		if !reflect.DeepEqual(got, src) {
			t.Fatalf("concurrent Freeze(src) = %+v, want deep-equal %+v", got, src)
		}
	}
}
