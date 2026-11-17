// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"runtime"
	"runtime/debug"
	"testing"
	"unsafe"
)

const (
	logosGCScanBoundsWord       = unsafe.Sizeof(uintptr(0))
	logosGCScanBoundsMaskBits   = 8 * logosGCScanBoundsWord
	logosGCScanBoundsBitmapSpan = logosGCScanBoundsMaskBits * logosGCScanBoundsWord
)

type logosGCScanBoundsLayout struct {
	name     string
	elem     any
	size     uintptr
	pointers []uintptr
}

func logosGCScanBoundsLayouts() []logosGCScanBoundsLayout {
	type iface struct {
		value any
	}
	type irregular3 struct {
		p unsafe.Pointer
		x uintptr
		q unsafe.Pointer
	}
	type irregular5 struct {
		x uintptr
		p unsafe.Pointer
		y [2]uintptr
		q unsafe.Pointer
	}
	type irregular9 struct {
		p    unsafe.Pointer
		x    [6]uintptr
		q    unsafe.Pointer
		tail uintptr
	}
	type boundary struct {
		gap  [logosGCScanBoundsMaskBits - 1]uintptr
		p    unsafe.Pointer
		q    unsafe.Pointer
		tail uintptr
	}
	type sparse struct {
		p    unsafe.Pointer
		gap  [2 * logosGCScanBoundsMaskBits]uintptr
		q    unsafe.Pointer
		tail [3]uintptr
	}

	return []logosGCScanBoundsLayout{
		{
			name:     "Pointer",
			elem:     unsafe.Pointer(nil),
			size:     unsafe.Sizeof(unsafe.Pointer(nil)),
			pointers: []uintptr{0},
		},
		{
			name: "Interface",
			elem: iface{},
			size: unsafe.Sizeof(iface{}),
			// Type metadata is retained separately; only data is traced.
			pointers: []uintptr{logosGCScanBoundsWord},
		},
		{
			name:     "String",
			elem:     "",
			size:     unsafe.Sizeof(""),
			pointers: []uintptr{0},
		},
		{
			name:     "Slice",
			elem:     []byte(nil),
			size:     unsafe.Sizeof([]byte(nil)),
			pointers: []uintptr{0},
		},
		{
			name:     "Irregular3",
			elem:     irregular3{},
			size:     unsafe.Sizeof(irregular3{}),
			pointers: []uintptr{unsafe.Offsetof(irregular3{}.p), unsafe.Offsetof(irregular3{}.q)},
		},
		{
			name:     "Irregular5",
			elem:     irregular5{},
			size:     unsafe.Sizeof(irregular5{}),
			pointers: []uintptr{unsafe.Offsetof(irregular5{}.p), unsafe.Offsetof(irregular5{}.q)},
		},
		{
			name:     "Irregular9",
			elem:     irregular9{},
			size:     unsafe.Sizeof(irregular9{}),
			pointers: []uintptr{unsafe.Offsetof(irregular9{}.p), unsafe.Offsetof(irregular9{}.q)},
		},
		{
			name:     "BitmapBoundary",
			elem:     boundary{},
			size:     unsafe.Sizeof(boundary{}),
			pointers: []uintptr{unsafe.Offsetof(boundary{}.p), unsafe.Offsetof(boundary{}.q)},
		},
		{
			name:     "EmptyMaskWord",
			elem:     sparse{},
			size:     unsafe.Sizeof(sparse{}),
			pointers: []uintptr{unsafe.Offsetof(sparse{}.p), unsafe.Offsetof(sparse{}.q)},
		},
	}
}

// The declared pointer offsets are independent of the runtime bitmap iterator.
// A slot contributes its full word when its start is below the byte limit,
// including when the limit cuts through that word.
func logosGCScanBoundsExtent(layout logosGCScanBoundsLayout, size uintptr) int64 {
	var extent uintptr
	for elem := uintptr(0); elem < size; elem += layout.size {
		for _, offset := range layout.pointers {
			if offset < size-elem {
				extent = elem + offset + logosGCScanBoundsWord
			}
		}
	}
	return int64(extent)
}

func TestLogosGCScanLayoutBounds(t *testing.T) {
	for _, layout := range logosGCScanBoundsLayouts() {
		for _, mode := range []string{"Self", "Nil", "OtherArena"} {
			t.Run(layout.name+"/"+mode, func(t *testing.T) {
				inFreshGoroutine(t, func() {
					defer debug.SetGCPercent(debug.SetGCPercent(-1))
					mark := runtime.LogosArenaSave()
					defer runtime.LogosArenaRestore(mark)

					// Sweep every byte limit, including pointer-word,
					// element, and bitmap-word boundaries and their
					// neighboring bytes. Round the backing buffer up
					// so a partial final pointer word remains readable.
					limit := 3*layout.size + logosGCScanBoundsBitmapSpan + 1
					allocation := (limit + logosGCScanBoundsWord - 1) &^ (logosGCScanBoundsWord - 1)
					p := runtime.LogosArenaAllocScan(layout.elem, allocation)
					if !runtime.LogosArenaInChunk(p) {
						t.Fatal("scan layout buffer fell back to the heap")
					}

					var target unsafe.Pointer
					switch mode {
					case "Self":
						target = p
					case "OtherArena":
						target = runtime.LogosArenaAlloc(byte(0), 64)
						if !runtime.LogosArenaInChunk(target) {
							t.Fatal("other arena target fell back to the heap")
						}
					}

					// Raw scanner fixtures never interpret interface,
					// string, or slice headers as language values.
					// Only declared pointer slots receive pointers;
					// scalar words and alignment padding stay zero.
					for elem := uintptr(0); elem < allocation; elem += layout.size {
						for _, offset := range layout.pointers {
							if offset < allocation-elem {
								*(*unsafe.Pointer)(unsafe.Add(p, elem+offset)) = target
							}
						}
					}

					for size := uintptr(0); size <= limit; size++ {
						got := runtime.LogosArenaScanTypedBlock(layout.elem, p, size)
						want := logosGCScanBoundsExtent(layout, size)
						if got != want {
							t.Fatalf("size %d: scan extent = %d, want %d", size, got, want)
						}
					}
					runtime.KeepAlive(target)
					runtime.KeepAlive(p)
				})
			})
		}
	}
}

func BenchmarkLogosGCScanSmall(b *testing.B) {
	for _, layout := range logosGCScanBoundsLayouts() {
		if layout.size > 3*logosGCScanBoundsWord {
			continue
		}
		for _, mode := range []string{"Self", "Nil", "OtherArena"} {
			b.Run(layout.name+"/"+mode, func(b *testing.B) {
				defer debug.SetGCPercent(debug.SetGCPercent(-1))
				runtime.GC()
				mark := runtime.LogosArenaSave()
				defer runtime.LogosArenaRestore(mark)

				p := runtime.LogosArenaAllocScan(layout.elem, layout.size)
				if !runtime.LogosArenaInChunk(p) {
					b.Fatal("small scan buffer fell back to the heap")
				}

				var other unsafe.Pointer
				if mode == "OtherArena" {
					other = runtime.LogosArenaAlloc(byte(0), 64)
					if !runtime.LogosArenaInChunk(other) {
						b.Fatal("other arena target fell back to the heap")
					}
				}
				for _, offset := range layout.pointers {
					var target unsafe.Pointer
					switch mode {
					case "Self":
						target = unsafe.Add(p, offset)
					case "OtherArena":
						target = other
					}
					*(*unsafe.Pointer)(unsafe.Add(p, offset)) = target
				}

				b.SetBytes(int64(layout.size))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					runtime.LogosArenaScanTypedBlock(layout.elem, p, layout.size)
				}
				b.StopTimer()
				runtime.KeepAlive(other)
				runtime.KeepAlive(p)
			})
		}
	}
}
