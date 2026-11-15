// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"encoding/binary"
	"runtime"
	"testing"
	"unsafe"
)

type viewasHeader struct {
	Magic   uint32
	Version uint32
	Size    uint64
}

func viewasNative(b []byte) viewasHeader {
	return viewasHeader{
		Magic:   binary.NativeEndian.Uint32(b),
		Version: binary.NativeEndian.Uint32(b[4:]),
		Size:    binary.NativeEndian.Uint64(b[8:]),
	}
}

func viewasHostLittleEndian() bool {
	return binary.NativeEndian.Uint16([]byte{1, 0}) == 1
}

func TestViewAsDirect(t *testing.T) {
	b := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	h := ViewAs[viewasHeader](b)
	if want := viewasNative(b); *h != want {
		t.Fatalf("ViewAs = %+v, want %+v", *h, want)
	}
}

func TestViewAsHostByteOrderValues(t *testing.T) {
	b := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	want := viewasHeader{Magic: 67305985, Version: 134678021, Size: 1157159078456920585}
	if !viewasHostLittleEndian() {
		want = viewasHeader{Magic: 16909060, Version: 84281096, Size: 651345242494996240}
	}
	if h := ViewAs[viewasHeader](b); *h != want {
		t.Fatalf("ViewAs = %+v, want %+v", *h, want)
	}
}

func TestViewAsOddOffset(t *testing.T) {
	b := make([]byte, 17)
	for i := range b {
		b[i] = byte(i + 1)
	}
	h := ViewAs[viewasHeader](b[1:])
	if want := viewasNative(b[1:]); *h != want {
		t.Fatalf("odd-offset ViewAs = %+v, want %+v", *h, want)
	}
}

func TestViewAsShortPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("ViewAs did not panic on short input")
		}
		msg, ok := r.(string)
		if !ok || msg != "logos: ViewAs: need 16 bytes, have 8" {
			t.Fatalf("ViewAs panic = %v, want %q", r, "logos: ViewAs: need 16 bytes, have 8")
		}
	}()
	_ = ViewAs[viewasHeader]([]byte{1, 2, 3, 4, 5, 6, 7, 8})
}

func TestViewAsCopyPath(t *testing.T) {
	b := make([]byte, 17)
	for i := range b {
		b[i] = byte(i + 1)
	}
	want := viewasNative(b[1:])
	h := viewAsCopy[viewasHeader](b[1:])
	if *h != want {
		t.Fatalf("copy-path ViewAs = %+v, want %+v", *h, want)
	}
	for i := range b {
		b[i] = 0
	}
	if *h != want {
		t.Fatalf("copy-path view aliases source: %+v", *h)
	}
}

// SAFETY: the unsafe pointers below are read-only address probes. They are
// converted to uintptr and compared immediately; no pointer is retained,
// dereferenced, or used after the backing slice is dropped.
func TestViewAsCopyAlignedIndependent(t *testing.T) {
	b := make([]byte, 19)
	for i := range b {
		b[i] = byte(i + 1)
	}
	off := b[1:]
	want := viewasNative(off)
	h := viewAsCopy[viewasHeader](off)
	if uintptr(unsafe.Pointer(h))%unsafe.Alignof(viewasHeader{}) != 0 {
		t.Fatalf("copy-path ViewAs misaligned: %p", h)
	}
	start := uintptr(unsafe.Pointer(&off[0]))
	end := start + uintptr(len(off))
	p := uintptr(unsafe.Pointer(h))
	if p >= start && p < end+uintptr(unsafe.Sizeof(viewasHeader{})) {
		t.Fatalf("copy-path ViewAs aliases its input: view %p inside [%x, %x)", h, start, end)
	}
	if d := ViewAs[viewasHeader](off); *h != *d {
		t.Fatalf("copy-path ViewAs = %+v, direct path = %+v", *h, *d)
	}
	off = nil
	b = nil
	runtime.GC()
	if *h != want {
		t.Fatalf("copy-path ViewAs invalid after input dropped: %+v", *h)
	}
	runtime.KeepAlive(h)
}

func TestViewAsKeepsAlive(t *testing.T) {
	v, want := freshViewas()
	runtime.GC()
	g := make([]byte, 1<<20)
	for i := range g {
		g[i] = 0xFF
	}
	runtime.GC()
	runtime.KeepAlive(g)
	if *v != want {
		t.Fatalf("view after GC = %+v, want %+v", *v, want)
	}
}

func freshViewas() (*viewasHeader, viewasHeader) {
	b := []byte{17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	return ViewAs[viewasHeader](b), viewasNative(b)
}

func TestViewAsPanicMentionsSizes(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("ViewAs did not panic on nil input")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("ViewAs panic type = %T, want string", r)
		}
		if msg != "logos: ViewAs: need 16 bytes, have 0" {
			t.Fatalf("ViewAs panic = %q, want %q", msg, "logos: ViewAs: need 16 bytes, have 0")
		}
	}()
	_ = ViewAs[viewasHeader](nil)
}
