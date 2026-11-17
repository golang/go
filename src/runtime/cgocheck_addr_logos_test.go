// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/abi"
	"internal/goexperiment"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

type logosCgoAddrPacket struct {
	header [16]byte
	next   *int
}

func TestLogosCgoPointerFreeAddr(t *testing.T) {
	type namedBool bool
	type namedInt int
	var packet logosCgoAddrPacket
	var nilScalar *int
	cases := []struct {
		name    string
		ptr     any
		marker  any
		allowed bool
	}{
		{"scalar", new(int), true, true},
		{"named_scalar", new(namedInt), true, true},
		{"scalar_nil", nilScalar, true, true},
		{"pointer_free_array", new([16]byte), true, true},
		{"empty_struct", new(struct{}), true, true},
		{"pointer_free_field", &packet.header, true, true},
		{"named_bool_marker", new(int), namedBool(true), true},
		{"false_bool_marker", new(int), false, true},
		{"pointer_field", &packet.next, true, false},
		{"whole_packet", &packet, true, false},
		{"pointer_array", new([2]*int), true, false},
		{"slice_header", new([]byte), true, false},
		{"string_header", new(string), true, false},
		{"interface", new(any), true, false},
		{"function", new(func()), true, false},
		{"channel", new(chan int), true, false},
		{"map", new(map[int]int), true, false},
		{"unsafe_pointer", unsafe.Pointer(&packet.header), true, false},
		{"unsafe_nil", unsafe.Pointer(nil), true, false},
		{"slice_marker", new(int), []int{1}, false},
		{"array_marker", new(int), [1]int{1}, false},
		{"array_pointer_marker", new(int), new([1]int), false},
		{"slice_value", []byte{1}, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := runtime.LogosCgoPointerFreeAddr(abi.TypeOf(tc.ptr), abi.TypeOf(tc.marker))
			want := goexperiment.LogosCgoOpt && tc.allowed
			if got != want {
				t.Fatalf("pointer-free address eligibility = %t, want %t (logoscgoopt=%t)", got, want, goexperiment.LogosCgoOpt)
			}
		})
	}
}

func logosCgoAddrMustPanic(t *testing.T, kind string, f func()) {
	t.Helper()
	defer func() {
		p := recover()
		if p == nil {
			t.Fatal("cgo pointer check accepted an unpinned nested pointer; require cgocheck=1")
		}
		err, ok := p.(error)
		if !ok || !strings.Contains(err.Error(), "Go pointer to unpinned Go "+kind) {
			t.Fatalf("unexpected cgo pointer panic: %v", p)
		}
	}()
	f()
}

func TestLogosCgoPointerFreeAddrBoundary(t *testing.T) {
	// The bool marker confines checking to header. A saved pointer must still
	// scan the complete heap allocation and reject next until it is pinned.
	storage := new([32]int)
	packet := &logosCgoAddrPacket{next: &storage[0]}
	runtime.CgoCheckPointer(&packet.header, true)
	runtime.CgoCheckPointer((*int)(nil), true)
	logosCgoAddrMustPanic(t, "pointer", func() {
		runtime.CgoCheckPointer(unsafe.Pointer(&packet.header), nil)
	})
	logosCgoAddrMustPanic(t, "pointer", func() {
		runtime.CgoCheckPointer(unsafe.Pointer(&packet.header), true)
	})
	logosCgoAddrMustPanic(t, "pointer", func() {
		runtime.CgoCheckPointer(&packet, true)
	})

	var pin runtime.Pinner
	pin.Pin(packet.next)
	defer pin.Unpin()
	runtime.GC()
	runtime.CgoCheckPointer(&packet.header, true)
	runtime.CgoCheckPointer(unsafe.Pointer(&packet.header), nil)
	runtime.CgoCheckPointer(&packet.next, true)
	runtime.KeepAlive(packet)
}

func TestLogosCgoPointerFreeAddrContainers(t *testing.T) {
	storage := new([32]int)
	p := &storage[0]
	slice := []*int{p}
	array := [1]*int{p}
	var value any = p
	logosCgoAddrMustPanic(t, "pointer", func() { runtime.CgoCheckPointer(&slice[0], slice) })
	logosCgoAddrMustPanic(t, "pointer", func() { runtime.CgoCheckPointer(&array[0], array) })
	logosCgoAddrMustPanic(t, "pointer", func() { runtime.CgoCheckPointer(&array[0], &array) })
	logosCgoAddrMustPanic(t, "interface", func() { runtime.CgoCheckPointer(&value, true) })
	runtime.KeepAlive(p)
}
