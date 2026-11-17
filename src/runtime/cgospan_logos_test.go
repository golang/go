// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

type logosCgoSpanSmall struct {
	data [64]byte
	next *[64]byte
}

// The unpinned pointer precedes the checked address, so a scan that starts
// at the passed address instead of the object base would miss it.
type logosCgoSpanPrefix struct {
	next *[64]byte
	data [64]byte
}

type logosCgoSpanHeader struct {
	data [1024]byte
	next *[64]byte
}

type logosCgoSpanLarge struct {
	data [64 << 10]byte
	next *[64]byte
}

var logosCgoSpanGlobal logosCgoSpanSmall
var logosCgoSpanGlobalBytes [64]byte

func logosCgoSpanReject(t *testing.T, p unsafe.Pointer) {
	t.Helper()
	defer func() {
		value := recover()
		if value == nil {
			t.Error("accepted unknown Go memory containing an unpinned Go pointer")
		} else if !strings.Contains(fmt.Sprint(value), "unpinned Go") {
			t.Errorf("wrong pointer-check panic: %v", value)
		}
	}()
	runtime.CgoCheckPointer(p, nil)
}

func TestLogosCgoSpanObject(t *testing.T) {
	runtime.CgoCheckPointer(unsafe.Pointer(nil), nil)
	// Each nested target exceeds the tiny allocator's block size. Pinning
	// another tiny allocation must not make an unpinned test target pass.
	shapes := []struct {
		name string
		make func() (unsafe.Pointer, *[64]byte, any)
	}{
		{"SmallBase", func() (unsafe.Pointer, *[64]byte, any) {
			value := &logosCgoSpanSmall{next: new([64]byte)}
			return unsafe.Pointer(value), value.next, value
		}},
		{"SmallInterior", func() (unsafe.Pointer, *[64]byte, any) {
			value := &logosCgoSpanSmall{next: new([64]byte)}
			return unsafe.Pointer(&value.data[len(value.data)-1]), value.next, value
		}},
		{"HeaderInterior", func() (unsafe.Pointer, *[64]byte, any) {
			value := &logosCgoSpanHeader{next: new([64]byte)}
			return unsafe.Pointer(&value.data[len(value.data)-1]), value.next, value
		}},
		{"LargeInterior", func() (unsafe.Pointer, *[64]byte, any) {
			value := &logosCgoSpanLarge{next: new([64]byte)}
			return unsafe.Pointer(&value.data[len(value.data)-1]), value.next, value
		}},
		{"PrefixInterior", func() (unsafe.Pointer, *[64]byte, any) {
			value := &logosCgoSpanPrefix{next: new([64]byte)}
			return unsafe.Pointer(&value.data[len(value.data)-1]), value.next, value
		}},
	}
	for _, shape := range shapes {
		t.Run(shape.name, func(t *testing.T) {
			for _, pinned := range []bool{false, true} {
				t.Run(fmt.Sprintf("Pinned=%v", pinned), func(t *testing.T) {
					p, child, root := shape.make()
					if pinned {
						var pin runtime.Pinner
						pin.Pin(child)
						defer pin.Unpin()
						runtime.CgoCheckPointer(p, nil)
					} else {
						logosCgoSpanReject(t, p)
					}
					runtime.KeepAlive(root)
				})
			}
		})
	}
}

func TestLogosCgoSpanGlobals(t *testing.T) {
	runtime.CgoCheckPointer(unsafe.Pointer(&logosCgoSpanGlobalBytes[31]), nil)
	logosCgoSpanGlobal.next = new([64]byte)
	defer func() { logosCgoSpanGlobal.next = nil }()
	var pin runtime.Pinner
	pin.Pin(logosCgoSpanGlobal.next)
	defer pin.Unpin()
	// The unknown extent of a data/bss pointer remains conservative even
	// when this particular global contains only a pinned nested pointer.
	logosCgoSpanReject(t, unsafe.Pointer(&logosCgoSpanGlobal.data[31]))
}
