// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

type listedVal struct {
	val int

	aNode runtime.ListNode
	bNode runtime.ListNode
}

func newListedVal(v int) *listedVal {
	return &listedVal{
		val: v,
	}
}

func TestListPush(t *testing.T) {
	var headA runtime.ListHead
	headA.Init(unsafe.Offsetof(listedVal{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedVal)(p)
	if v == nil {
		t.Fatalf("pop got nil want 3")
	}
	if v.val != 3 {
		t.Errorf("pop got %d want 3", v.val)
	}

	p = headA.Pop()
	v = (*listedVal)(p)
	if v == nil {
		t.Fatalf("pop got nil want 2")
	}
	if v.val != 2 {
		t.Errorf("pop got %d want 2", v.val)
	}

	p = headA.Pop()
	v = (*listedVal)(p)
	if v == nil {
		t.Fatalf("pop got nil want 1")
	}
	if v.val != 1 {
		t.Errorf("pop got %d want 1", v.val)
	}

	p = headA.Pop()
	v = (*listedVal)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func wantVal(t *testing.T, v *listedVal, i int) {
	t.Helper()
	if v == nil {
		t.Fatalf("listedVal got nil want %d", i)
	}
	if v.val != i {
		t.Errorf("pop got %d want %d", v.val, i)
	}
}

func TestListRemoveHead(t *testing.T) {
	var headA runtime.ListHead
	headA.Init(unsafe.Offsetof(listedVal{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedVal)(p)
	wantVal(t, v, 2)

	p = headA.Pop()
	v = (*listedVal)(p)
	wantVal(t, v, 1)

	p = headA.Pop()
	v = (*listedVal)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func TestListRemoveMiddle(t *testing.T) {
	var headA runtime.ListHead
	headA.Init(unsafe.Offsetof(listedVal{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(two))

	p := headA.Pop()
	v := (*listedVal)(p)
	wantVal(t, v, 3)

	p = headA.Pop()
	v = (*listedVal)(p)
	wantVal(t, v, 1)

	p = headA.Pop()
	v = (*listedVal)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func TestListRemoveTail(t *testing.T) {
	var headA runtime.ListHead
	headA.Init(unsafe.Offsetof(listedVal{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(one))

	p := headA.Pop()
	v := (*listedVal)(p)
	wantVal(t, v, 3)

	p = headA.Pop()
	v = (*listedVal)(p)
	wantVal(t, v, 2)

	p = headA.Pop()
	v = (*listedVal)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func TestListRemoveAll(t *testing.T) {
	var headA runtime.ListHead
	headA.Init(unsafe.Offsetof(listedVal{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(one))
	headA.Remove(unsafe.Pointer(two))
	headA.Remove(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedVal)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func BenchmarkListPushPop(b *testing.B) {
	var head runtime.ListHead
	head.Init(unsafe.Offsetof(listedVal{}.aNode))

	vals := make([]listedVal, 10000)
	i := 0
	for b.Loop() {
		if i == len(vals) {
			for range len(vals) {
				head.Pop()
			}
			i = 0
		}

		head.Push(unsafe.Pointer(&vals[i]))

		i++
	}
}
