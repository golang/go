// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Attack tests for M5-S6: Freeze deep-copies its argument, preserves
// cycles and sharing, keeps nils nil, and copies unexported fields.
// This file pins in plain Go the run-time half of the slice: Freeze for
// Go callers, the FreezeDynamic copier that open-interface payloads
// use, and the typed helpers that generated copiers call, sharing one
// visited state. The generated copiers that every .lgo call lowers to
// are pinned by test/logos/run/freeze_copy.lgo; the const-view half by
// test/logos/run/freeze.lgo and test/logos/compile/freeze_const.lgo.
package rt_test

import (
	"fmt"
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

func TestFreezeMapKeysAndSliceCapacity(t *testing.T) {
	key := &freezeChild{n: 4, note: "key"}
	raw := []int{1, 2, 3}
	payload := &freezeChild{n: 6, note: "iface"}
	src := &struct {
		keys  map[*freezeChild]string
		s     []int
		iface any
	}{
		keys:  map[*freezeChild]string{key: "value"},
		s:     raw[:2],
		iface: payload,
	}
	got := rt.Freeze(src)
	key.n = 9
	raw[2] = 8
	payload.n = 9
	if len(got.keys) != 1 {
		t.Fatalf("frozen map length = %d, want 1", len(got.keys))
	}
	for k, v := range got.keys {
		if k.n != 4 || v != "value" {
			t.Fatalf("frozen map entry = (%d, %q), want (4, value)", k.n, v)
		}
	}
	tail := got.s[:cap(got.s)]
	if tail[2] != 3 {
		t.Fatalf("frozen slice capacity tail = %d, want 3", tail[2])
	}
	if got.iface.(*freezeChild).n != 6 {
		t.Fatalf("frozen iface payload = %d, want 6", got.iface.(*freezeChild).n)
	}
}

type typedNode struct {
	n     int
	next  *typedNode
	vals  []int
	tags  map[string]int
	iface any
}

func copyTypedNode(fr *rt.Freezer, dst, src *typedNode) {
	*dst = *src
	dst.next = rt.FreezePointer(fr, src.next, copyTypedNode)
	dst.vals = rt.FreezeSlice(fr, src.vals, nil)
	dst.tags = rt.FreezeMap(fr, src.tags, nil, nil)
	rt.FreezeDynamic(fr, &dst.iface, &src.iface)
}

func copyTypedNodePtr(fr *rt.Freezer, dst, src **typedNode) {
	*dst = rt.FreezePointer(fr, *src, copyTypedNode)
}

type nodeRef *typedNode

type refPair struct {
	iface any
	ref   nodeRef
}

func copyRefPair(fr *rt.Freezer, dst, src *refPair) {
	*dst = *src
	rt.FreezeDynamic(fr, &dst.iface, &src.iface)
	dst.ref = rt.FreezePointer(fr, src.ref, copyTypedNode)
}

type keyedNodes struct {
	keys  map[*typedNode]string
	first *typedNode
}

func copyKeyedNodes(fr *rt.Freezer, dst, src *keyedNodes) {
	*dst = *src
	dst.keys = rt.FreezeMap(fr, src.keys, copyTypedNodePtr, nil)
	dst.first = rt.FreezePointer(fr, src.first, copyTypedNode)
}

func recoverText(f func()) (text string) {
	defer func() {
		text = fmt.Sprint(recover())
	}()
	f()
	return ""
}

func TestFreezeRootNil(t *testing.T) {
	if got := rt.FreezeRoot[typedNode](nil, copyTypedNode); got != nil {
		t.Fatalf("FreezeRoot(nil, copier) = %v, want nil", got)
	}
	if got := rt.FreezeRoot[int](nil, nil); got != nil {
		t.Fatalf("FreezeRoot(nil, nil) = %v, want nil", got)
	}
	v := 7
	got := rt.FreezeRoot(&v, nil)
	if got == nil || got == &v || *got != 7 {
		t.Fatalf("FreezeRoot(&7, nil) = %p, want a fresh 7", got)
	}
}

func TestFreezeTypedBackEdgesResolveAcrossInterfaces(t *testing.T) {
	src := &typedNode{n: 1}
	src.next = src
	src.iface = src
	got := rt.FreezeRoot(src, copyTypedNode)
	if got == src || got.next != got {
		t.Fatal("typed back edge does not resolve to the copy under construction")
	}
	if p, ok := got.iface.(*typedNode); !ok || p != got {
		t.Fatalf("interface back edge = %v, want the typed copy %p", got.iface, got)
	}
}

func TestFreezeInterfaceEdgesReuseTypedCopies(t *testing.T) {
	raw := []int{1, 2, 3}
	src := &typedNode{vals: raw[:1], tags: map[string]int{"k": 1}}
	src.iface = []any{src.vals, src.tags, raw[:1:1]}
	got := rt.FreezeRoot(src, copyTypedNode)
	raw[0] = 9
	parts := got.iface.([]any)
	same := parts[0].([]int)
	if &same[0] != &got.vals[0] || cap(same) != 3 || same[0] != 1 {
		t.Fatal("interface view of the typed slice header did not share its copy")
	}
	if reflect.ValueOf(parts[1]).UnsafePointer() != reflect.ValueOf(got.tags).UnsafePointer() {
		t.Fatal("interface view of the typed map did not share its copy")
	}
	if narrow := parts[2].([]int); &narrow[0] == &got.vals[0] || cap(narrow) != 1 || narrow[0] != 1 {
		t.Fatal("interface header with another capacity shared the typed copy")
	}
}

func TestFreezeTypedEdgesReuseInterfaceCopies(t *testing.T) {
	shared := &typedNode{n: 5}
	src := &refPair{iface: nodeRef(shared), ref: shared}
	got := rt.FreezeRoot(src, copyRefPair)
	shared.n = 6
	if got.ref == shared || got.ref.n != 5 {
		t.Fatalf("typed named-pointer edge = %+v, want a fresh copy of 5", got.ref)
	}
	if p, ok := got.iface.(nodeRef); !ok || p != got.ref {
		t.Fatalf("typed edge %p did not reuse the interface copy %v", got.ref, got.iface)
	}
}

func TestFreezeMapTypedKeysShareWithPointers(t *testing.T) {
	key := &typedNode{n: 4}
	src := &keyedNodes{keys: map[*typedNode]string{key: "v"}, first: key}
	got := rt.FreezeRoot(src, copyKeyedNodes)
	key.n = 9
	if len(got.keys) != 1 {
		t.Fatalf("frozen map length = %d, want 1", len(got.keys))
	}
	for k, v := range got.keys {
		if k == key || k != got.first || k.n != 4 || v != "v" {
			t.Fatalf("frozen key %p (%d, %q) does not share the pointer copy %p", k, k.n, v, got.first)
		}
	}
}

func TestFreezeSliceTypedHeaders(t *testing.T) {
	s := []int{10, 20, 30}
	var fr rt.Freezer
	a := rt.FreezeSlice(&fr, s[:1:3], nil)
	b := rt.FreezeSlice(&fr, s[:1:3], nil)
	c := rt.FreezeSlice(&fr, s[:1:1], nil)
	s[2] = 99
	if &a[0] != &b[0] {
		t.Fatal("identical headers diverged; want one shared copy")
	}
	if &a[0] == &c[0] || cap(c) != 1 {
		t.Fatal("headers with different capacities shared one copy")
	}
	if !reflect.DeepEqual(a[:3], []int{10, 20, 30}) {
		t.Fatalf("typed full-cap view = %v, want [10 20 30]", a[:3])
	}
	if rt.FreezeSlice[[]int](&fr, nil, nil) != nil || rt.FreezeMap[map[int]int](&fr, nil, nil, nil) != nil {
		t.Fatal("nil slice or map grew a copy")
	}
}

func TestFreezeRejectMatchesDynamicText(t *testing.T) {
	ch := make(chan int)
	dynamic := recoverText(func() { rt.Freeze(&ch) })
	typed := recoverText(func() { rt.FreezeReject("chan") })
	if dynamic != "logos/rt: Freeze cannot isolate chan" || typed != dynamic {
		t.Fatalf("rejection texts = %q (dynamic), %q (typed), want both %q", dynamic, typed, "logos/rt: Freeze cannot isolate chan")
	}
}
