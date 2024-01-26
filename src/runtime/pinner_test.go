// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"time"
	"unsafe"
)

type obj struct {
	x int64
	y int64
	z int64
}

type objWith[T any] struct {
	x int64
	y int64
	z int64
	o T
}

var (
	globalUintptr                uintptr
	globalPtrToObj               = &obj{}
	globalPtrToObjWithPtr        = &objWith[*uintptr]{}
	globalPtrToRuntimeObj        = func() *obj { return &obj{} }()
	globalPtrToRuntimeObjWithPtr = func() *objWith[*uintptr] { return &objWith[*uintptr]{} }()
)

func assertDidPanic(t *testing.T) {
	if recover() == nil {
		t.Fatal("did not panic")
	}
}

func assertCgoCheckPanics(t *testing.T, p any) {
	defer func() {
		if recover() == nil {
			t.Fatal("cgoCheckPointer() did not panic, make sure the tests run with cgocheck=1")
		}
	}()
	runtime.CgoCheckPointer(p, true)
}

func TestPinnerSimple(t *testing.T) {
	var pinner runtime.Pinner
	p := new(obj)
	addr := unsafe.Pointer(p)
	if runtime.IsPinned(addr) {
		t.Fatal("already marked as pinned")
	}
	pinner.Pin(p)
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	if runtime.GetPinCounter(addr) != nil {
		t.Fatal("pin counter should not exist")
	}
	pinner.Unpin()
	if runtime.IsPinned(addr) {
		t.Fatal("still marked as pinned")
	}
}

func TestPinnerPinKeepsAliveAndReleases(t *testing.T) {
	var pinner runtime.Pinner
	p := new(obj)
	done := make(chan struct{})
	runtime.SetFinalizer(p, func(any) {
		done <- struct{}{}
	})
	pinner.Pin(p)
	p = nil
	runtime.GC()
	runtime.GC()
	select {
	case <-done:
		t.Fatal("Pin() didn't keep object alive")
	case <-time.After(time.Millisecond * 10):
		break
	}
	pinner.Unpin()
	runtime.GC()
	runtime.GC()
	select {
	case <-done:
		break
	case <-time.After(time.Second):
		t.Fatal("Unpin() didn't release object")
	}
}

func TestPinnerMultiplePinsSame(t *testing.T) {
	const N = 100
	var pinner runtime.Pinner
	p := new(obj)
	addr := unsafe.Pointer(p)
	if runtime.IsPinned(addr) {
		t.Fatal("already marked as pinned")
	}
	for i := 0; i < N; i++ {
		pinner.Pin(p)
	}
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	if cnt := runtime.GetPinCounter(addr); cnt == nil || *cnt != N-1 {
		t.Fatalf("pin counter incorrect: %d", *cnt)
	}
	pinner.Unpin()
	if runtime.IsPinned(addr) {
		t.Fatal("still marked as pinned")
	}
	if runtime.GetPinCounter(addr) != nil {
		t.Fatal("pin counter was not deleted")
	}
}

func TestPinnerTwoPinner(t *testing.T) {
	var pinner1, pinner2 runtime.Pinner
	p := new(obj)
	addr := unsafe.Pointer(p)
	if runtime.IsPinned(addr) {
		t.Fatal("already marked as pinned")
	}
	pinner1.Pin(p)
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	if runtime.GetPinCounter(addr) != nil {
		t.Fatal("pin counter should not exist")
	}
	pinner2.Pin(p)
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	if cnt := runtime.GetPinCounter(addr); cnt == nil || *cnt != 1 {
		t.Fatalf("pin counter incorrect: %d", *cnt)
	}
	pinner1.Unpin()
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	if runtime.GetPinCounter(addr) != nil {
		t.Fatal("pin counter should not exist")
	}
	pinner2.Unpin()
	if runtime.IsPinned(addr) {
		t.Fatal("still marked as pinned")
	}
	if runtime.GetPinCounter(addr) != nil {
		t.Fatal("pin counter was not deleted")
	}
}

func TestPinnerPinZerosizeObj(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	p := new(struct{})
	pinner.Pin(p)
	if !runtime.IsPinned(unsafe.Pointer(p)) {
		t.Fatal("not marked as pinned")
	}
}

func TestPinnerPinGlobalPtr(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(globalPtrToObj)
	pinner.Pin(globalPtrToObjWithPtr)
	pinner.Pin(globalPtrToRuntimeObj)
	pinner.Pin(globalPtrToRuntimeObjWithPtr)
}

func TestPinnerPinTinyObj(t *testing.T) {
	var pinner runtime.Pinner
	const N = 64
	var addr [N]unsafe.Pointer
	for i := 0; i < N; i++ {
		p := new(bool)
		addr[i] = unsafe.Pointer(p)
		pinner.Pin(p)
		pinner.Pin(p)
		if !runtime.IsPinned(addr[i]) {
			t.Fatalf("not marked as pinned: %d", i)
		}
		if cnt := runtime.GetPinCounter(addr[i]); cnt == nil || *cnt == 0 {
			t.Fatalf("pin counter incorrect: %d, %d", *cnt, i)
		}
	}
	pinner.Unpin()
	for i := 0; i < N; i++ {
		if runtime.IsPinned(addr[i]) {
			t.Fatal("still marked as pinned")
		}
		if runtime.GetPinCounter(addr[i]) != nil {
			t.Fatal("pin counter should not exist")
		}
	}
}

func TestPinnerInterface(t *testing.T) {
	var pinner runtime.Pinner
	o := new(obj)
	ifc := any(o)
	pinner.Pin(&ifc)
	if !runtime.IsPinned(unsafe.Pointer(&ifc)) {
		t.Fatal("not marked as pinned")
	}
	if runtime.IsPinned(unsafe.Pointer(o)) {
		t.Fatal("marked as pinned")
	}
	pinner.Unpin()
	pinner.Pin(ifc)
	if !runtime.IsPinned(unsafe.Pointer(o)) {
		t.Fatal("not marked as pinned")
	}
	if runtime.IsPinned(unsafe.Pointer(&ifc)) {
		t.Fatal("marked as pinned")
	}
	pinner.Unpin()
}

func TestPinnerPinNonPtrPanics(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	var i int
	defer assertDidPanic(t)
	pinner.Pin(i)
}

func TestPinnerReuse(t *testing.T) {
	var pinner runtime.Pinner
	p := new(obj)
	p2 := &p
	assertCgoCheckPanics(t, p2)
	pinner.Pin(p)
	runtime.CgoCheckPointer(p2, true)
	pinner.Unpin()
	assertCgoCheckPanics(t, p2)
	pinner.Pin(p)
	runtime.CgoCheckPointer(p2, true)
	pinner.Unpin()
}

func TestPinnerEmptyUnpin(t *testing.T) {
	var pinner runtime.Pinner
	pinner.Unpin()
	pinner.Unpin()
}

func TestPinnerLeakPanics(t *testing.T) {
	old := runtime.GetPinnerLeakPanic()
	func() {
		defer assertDidPanic(t)
		old()
	}()
	done := make(chan struct{})
	runtime.SetPinnerLeakPanic(func() {
		done <- struct{}{}
	})
	func() {
		var pinner runtime.Pinner
		p := new(obj)
		pinner.Pin(p)
	}()
	runtime.GC()
	runtime.GC()
	select {
	case <-done:
		break
	case <-time.After(time.Second):
		t.Fatal("leak didn't make GC to panic")
	}
	runtime.SetPinnerLeakPanic(old)
}

func TestPinnerCgoCheckPtr2Ptr(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	p := new(obj)
	p2 := &objWith[*obj]{o: p}
	assertCgoCheckPanics(t, p2)
	pinner.Pin(p)
	runtime.CgoCheckPointer(p2, true)
}

func TestPinnerCgoCheckPtr2UnsafePtr(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	p := unsafe.Pointer(new(obj))
	p2 := &objWith[unsafe.Pointer]{o: p}
	assertCgoCheckPanics(t, p2)
	pinner.Pin(p)
	runtime.CgoCheckPointer(p2, true)
}

func TestPinnerCgoCheckPtr2UnknownPtr(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	p := unsafe.Pointer(new(obj))
	p2 := &p
	func() {
		defer assertDidPanic(t)
		runtime.CgoCheckPointer(p2, nil)
	}()
	pinner.Pin(p)
	runtime.CgoCheckPointer(p2, nil)
}

func TestPinnerCgoCheckInterface(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	var ifc any
	var o obj
	ifc = &o
	p := &ifc
	assertCgoCheckPanics(t, p)
	pinner.Pin(&o)
	runtime.CgoCheckPointer(p, true)
}

func TestPinnerCgoCheckSlice(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	sl := []int{1, 2, 3}
	assertCgoCheckPanics(t, &sl)
	pinner.Pin(&sl[0])
	runtime.CgoCheckPointer(&sl, true)
}

func TestPinnerCgoCheckString(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	b := []byte("foobar")
	str := unsafe.String(&b[0], 6)
	assertCgoCheckPanics(t, &str)
	pinner.Pin(&b[0])
	runtime.CgoCheckPointer(&str, true)
}

func TestPinnerCgoCheckPinned2UnpinnedPanics(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	p := new(obj)
	p2 := &objWith[*obj]{o: p}
	assertCgoCheckPanics(t, p2)
	pinner.Pin(p2)
	assertCgoCheckPanics(t, p2)
}

func TestPinnerCgoCheckPtr2Pinned2Unpinned(t *testing.T) {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	p := new(obj)
	p2 := &objWith[*obj]{o: p}
	p3 := &objWith[*objWith[*obj]]{o: p2}
	assertCgoCheckPanics(t, p2)
	assertCgoCheckPanics(t, p3)
	pinner.Pin(p2)
	assertCgoCheckPanics(t, p2)
	assertCgoCheckPanics(t, p3)
	pinner.Pin(p)
	runtime.CgoCheckPointer(p2, true)
	runtime.CgoCheckPointer(p3, true)
}

func BenchmarkPinnerPinUnpinBatch(b *testing.B) {
	const Batch = 1000
	var data [Batch]*obj
	for i := 0; i < Batch; i++ {
		data[i] = new(obj)
	}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		var pinner runtime.Pinner
		for i := 0; i < Batch; i++ {
			pinner.Pin(data[i])
		}
		pinner.Unpin()
	}
}

func BenchmarkPinnerPinUnpinBatchDouble(b *testing.B) {
	const Batch = 1000
	var data [Batch]*obj
	for i := 0; i < Batch; i++ {
		data[i] = new(obj)
	}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		var pinner runtime.Pinner
		for i := 0; i < Batch; i++ {
			pinner.Pin(data[i])
			pinner.Pin(data[i])
		}
		pinner.Unpin()
	}
}

func BenchmarkPinnerPinUnpinBatchTiny(b *testing.B) {
	const Batch = 1000
	var data [Batch]*bool
	for i := 0; i < Batch; i++ {
		data[i] = new(bool)
	}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		var pinner runtime.Pinner
		for i := 0; i < Batch; i++ {
			pinner.Pin(data[i])
		}
		pinner.Unpin()
	}
}

func BenchmarkPinnerPinUnpin(b *testing.B) {
	p := new(obj)
	for n := 0; n < b.N; n++ {
		var pinner runtime.Pinner
		pinner.Pin(p)
		pinner.Unpin()
	}
}

func BenchmarkPinnerPinUnpinTiny(b *testing.B) {
	p := new(bool)
	for n := 0; n < b.N; n++ {
		var pinner runtime.Pinner
		pinner.Pin(p)
		pinner.Unpin()
	}
}

func BenchmarkPinnerPinUnpinDouble(b *testing.B) {
	p := new(obj)
	for n := 0; n < b.N; n++ {
		var pinner runtime.Pinner
		pinner.Pin(p)
		pinner.Pin(p)
		pinner.Unpin()
	}
}

func BenchmarkPinnerPinUnpinParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		p := new(obj)
		for pb.Next() {
			var pinner runtime.Pinner
			pinner.Pin(p)
			pinner.Unpin()
		}
	})
}

func BenchmarkPinnerPinUnpinParallelTiny(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		p := new(bool)
		for pb.Next() {
			var pinner runtime.Pinner
			pinner.Pin(p)
			pinner.Unpin()
		}
	})
}

func BenchmarkPinnerPinUnpinParallelDouble(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		p := new(obj)
		for pb.Next() {
			var pinner runtime.Pinner
			pinner.Pin(p)
			pinner.Pin(p)
			pinner.Unpin()
		}
	})
}

func BenchmarkPinnerIsPinnedOnPinned(b *testing.B) {
	var pinner runtime.Pinner
	ptr := new(obj)
	pinner.Pin(ptr)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		runtime.IsPinned(unsafe.Pointer(ptr))
	}
	pinner.Unpin()
}

func BenchmarkPinnerIsPinnedOnUnpinned(b *testing.B) {
	ptr := new(obj)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		runtime.IsPinned(unsafe.Pointer(ptr))
	}
}

func BenchmarkPinnerIsPinnedOnPinnedParallel(b *testing.B) {
	var pinner runtime.Pinner
	ptr := new(obj)
	pinner.Pin(ptr)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			runtime.IsPinned(unsafe.Pointer(ptr))
		}
	})
	pinner.Unpin()
}

func BenchmarkPinnerIsPinnedOnUnpinnedParallel(b *testing.B) {
	ptr := new(obj)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			runtime.IsPinned(unsafe.Pointer(ptr))
		}
	})
}

// const string data is not in span.
func TestPinnerConstStringData(t *testing.T) {
	var pinner runtime.Pinner
	str := "test-const-string"
	p := unsafe.StringData(str)
	addr := unsafe.Pointer(p)
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	pinner.Pin(p)
	pinner.Unpin()
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
}

func TestPinnerPinString(t *testing.T) {
	var pinner runtime.Pinner
	heapStr := getHeapStr()
	pinner.Pin(heapStr)
	addr := unsafe.Pointer(unsafe.StringData(heapStr))
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	if runtime.GetPinCounter(addr) != nil {
		t.Fatal("pin counter should not exist")
	}
	pinner.Unpin()
	if runtime.IsPinned(addr) {
		t.Fatal("still marked as pinned")
	}
}

func getHeapStr() string {
	return string(byte(fastrand()))
}

func TestPinnerPinSlice(t *testing.T) {
	var pinner runtime.Pinner
	s := make([]*int, 10)
	pinner.Pin(s)
	addr := unsafe.Pointer(unsafe.SliceData(s))
	if !runtime.IsPinned(addr) {
		t.Fatal("not marked as pinned")
	}
	if runtime.GetPinCounter(addr) != nil {
		t.Fatal("pin counter should not exist")
	}
	pinner.Unpin()
	if runtime.IsPinned(addr) {
		t.Fatal("still marked as pinned")
	}
}
