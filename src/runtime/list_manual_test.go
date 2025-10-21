// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/runtime/sys"
	"runtime"
	"testing"
	"unsafe"
)

// The tests in this file are identical to list_test.go, but for the
// manually-managed variants.

type listedValManual struct {
	val int

	aNode runtime.ListNodeManual
	bNode runtime.ListNodeManual
}

func newListedValManual(v int) *listedValManual {
	return &listedValManual{
		val: v,
	}
}

func TestListManualPush(t *testing.T) {
	var headA runtime.ListHeadManual
	headA.Init(unsafe.Offsetof(listedValManual{}.aNode))

	one := newListedValManual(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedValManual(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedValManual(3)
	headA.Push(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedValManual)(p)
	if v == nil {
		t.Fatalf("pop got nil want 3")
	}
	if v.val != 3 {
		t.Errorf("pop got %d want 3", v.val)
	}

	p = headA.Pop()
	v = (*listedValManual)(p)
	if v == nil {
		t.Fatalf("pop got nil want 2")
	}
	if v.val != 2 {
		t.Errorf("pop got %d want 2", v.val)
	}

	p = headA.Pop()
	v = (*listedValManual)(p)
	if v == nil {
		t.Fatalf("pop got nil want 1")
	}
	if v.val != 1 {
		t.Errorf("pop got %d want 1", v.val)
	}

	p = headA.Pop()
	v = (*listedValManual)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}

	runtime.KeepAlive(one)
	runtime.KeepAlive(two)
	runtime.KeepAlive(three)
}

func wantValManual(t *testing.T, v *listedValManual, i int) {
	t.Helper()
	if v == nil {
		t.Fatalf("listedVal got nil want %d", i)
	}
	if v.val != i {
		t.Errorf("pop got %d want %d", v.val, i)
	}
}

func TestListManualRemoveHead(t *testing.T) {
	var headA runtime.ListHeadManual
	headA.Init(unsafe.Offsetof(listedValManual{}.aNode))

	one := newListedValManual(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedValManual(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedValManual(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedValManual)(p)
	wantValManual(t, v, 2)

	p = headA.Pop()
	v = (*listedValManual)(p)
	wantValManual(t, v, 1)

	p = headA.Pop()
	v = (*listedValManual)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}

	runtime.KeepAlive(one)
	runtime.KeepAlive(two)
	runtime.KeepAlive(three)
}

func TestListManualRemoveMiddle(t *testing.T) {
	var headA runtime.ListHeadManual
	headA.Init(unsafe.Offsetof(listedValManual{}.aNode))

	one := newListedValManual(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedValManual(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedValManual(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(two))

	p := headA.Pop()
	v := (*listedValManual)(p)
	wantValManual(t, v, 3)

	p = headA.Pop()
	v = (*listedValManual)(p)
	wantValManual(t, v, 1)

	p = headA.Pop()
	v = (*listedValManual)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}

	runtime.KeepAlive(one)
	runtime.KeepAlive(two)
	runtime.KeepAlive(three)
}

func TestListManualRemoveTail(t *testing.T) {
	var headA runtime.ListHeadManual
	headA.Init(unsafe.Offsetof(listedValManual{}.aNode))

	one := newListedValManual(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedValManual(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedValManual(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(one))

	p := headA.Pop()
	v := (*listedValManual)(p)
	wantValManual(t, v, 3)

	p = headA.Pop()
	v = (*listedValManual)(p)
	wantValManual(t, v, 2)

	p = headA.Pop()
	v = (*listedValManual)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}

	runtime.KeepAlive(one)
	runtime.KeepAlive(two)
	runtime.KeepAlive(three)
}

func TestListManualRemoveAll(t *testing.T) {
	var headA runtime.ListHeadManual
	headA.Init(unsafe.Offsetof(listedValManual{}.aNode))

	one := newListedValManual(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedValManual(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedValManual(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(one))
	headA.Remove(unsafe.Pointer(two))
	headA.Remove(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedValManual)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}

	runtime.KeepAlive(one)
	runtime.KeepAlive(two)
	runtime.KeepAlive(three)
}

// The tests below are identical, but used with a sys.NotInHeap type.

type listedValNIH struct {
	_ sys.NotInHeap
	listedValManual
}

func newListedValNIH(v int) *listedValNIH {
	l := (*listedValNIH)(runtime.PersistentAlloc(unsafe.Sizeof(listedValNIH{}), unsafe.Alignof(listedValNIH{})))
	l.val = v
	return l
}

func newListHeadNIH() *runtime.ListHeadManual {
	return (*runtime.ListHeadManual)(runtime.PersistentAlloc(unsafe.Sizeof(runtime.ListHeadManual{}), unsafe.Alignof(runtime.ListHeadManual{})))
}

func TestListNIHPush(t *testing.T) {
	headA := newListHeadNIH()
	headA.Init(unsafe.Offsetof(listedValNIH{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedValNIH)(p)
	if v == nil {
		t.Fatalf("pop got nil want 3")
	}
	if v.val != 3 {
		t.Errorf("pop got %d want 3", v.val)
	}

	p = headA.Pop()
	v = (*listedValNIH)(p)
	if v == nil {
		t.Fatalf("pop got nil want 2")
	}
	if v.val != 2 {
		t.Errorf("pop got %d want 2", v.val)
	}

	p = headA.Pop()
	v = (*listedValNIH)(p)
	if v == nil {
		t.Fatalf("pop got nil want 1")
	}
	if v.val != 1 {
		t.Errorf("pop got %d want 1", v.val)
	}

	p = headA.Pop()
	v = (*listedValNIH)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func wantValNIH(t *testing.T, v *listedValNIH, i int) {
	t.Helper()
	if v == nil {
		t.Fatalf("listedVal got nil want %d", i)
	}
	if v.val != i {
		t.Errorf("pop got %d want %d", v.val, i)
	}
}

func TestListNIHRemoveHead(t *testing.T) {
	headA := newListHeadNIH()
	headA.Init(unsafe.Offsetof(listedValNIH{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(three))

	p := headA.Pop()
	v := (*listedValNIH)(p)
	wantValNIH(t, v, 2)

	p = headA.Pop()
	v = (*listedValNIH)(p)
	wantValNIH(t, v, 1)

	p = headA.Pop()
	v = (*listedValNIH)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func TestListNIHRemoveMiddle(t *testing.T) {
	headA := newListHeadNIH()
	headA.Init(unsafe.Offsetof(listedValNIH{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(two))

	p := headA.Pop()
	v := (*listedValNIH)(p)
	wantValNIH(t, v, 3)

	p = headA.Pop()
	v = (*listedValNIH)(p)
	wantValNIH(t, v, 1)

	p = headA.Pop()
	v = (*listedValNIH)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func TestListNIHRemoveTail(t *testing.T) {
	headA := newListHeadNIH()
	headA.Init(unsafe.Offsetof(listedValNIH{}.aNode))

	one := newListedVal(1)
	headA.Push(unsafe.Pointer(one))

	two := newListedVal(2)
	headA.Push(unsafe.Pointer(two))

	three := newListedVal(3)
	headA.Push(unsafe.Pointer(three))

	headA.Remove(unsafe.Pointer(one))

	p := headA.Pop()
	v := (*listedValNIH)(p)
	wantValNIH(t, v, 3)

	p = headA.Pop()
	v = (*listedValNIH)(p)
	wantValNIH(t, v, 2)

	p = headA.Pop()
	v = (*listedValNIH)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}

func TestListNIHRemoveAll(t *testing.T) {
	headA := newListHeadNIH()
	headA.Init(unsafe.Offsetof(listedValNIH{}.aNode))

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
	v := (*listedValNIH)(p)
	if v != nil {
		t.Fatalf("pop got %+v want nil", v)
	}
}
