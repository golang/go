// errorcheck -0 -l -d=wb

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"reflect"
	"unsafe"

	reflect2 "reflect"
)

func sink(e interface{})

func a(hdr *reflect.SliceHeader, p *byte) {
	hdr.Data = uintptr(unsafe.Pointer(p)) // ERROR "write barrier"
}

func b(hdr *reflect.StringHeader, p *byte) {
	hdr.Data = uintptr(unsafe.Pointer(p)) // ERROR "write barrier"
}

func c(hdrs *[1]reflect.SliceHeader, p *byte) {
	hdrs[0].Data = uintptr(unsafe.Pointer(p)) // ERROR "write barrier"
}

func d(hdr *struct{ s reflect.StringHeader }, p *byte) {
	hdr.s.Data = uintptr(unsafe.Pointer(p)) // ERROR "write barrier"
}

func e(p *byte) (resHeap, resStack string) {
	sink(&resHeap)

	hdr := (*reflect.StringHeader)(unsafe.Pointer(&resHeap))
	hdr.Data = uintptr(unsafe.Pointer(p)) // ERROR "write barrier"

	// No write barrier for non-escaping stack vars.
	hdr = (*reflect.StringHeader)(unsafe.Pointer(&resStack))
	hdr.Data = uintptr(unsafe.Pointer(p))

	return
}

func f(hdr *reflect2.SliceHeader, p *byte) {
	hdr.Data = uintptr(unsafe.Pointer(p)) // ERROR "write barrier"
}

type SliceHeader struct {
	Data uintptr
}

func g(hdr *SliceHeader, p *byte) {
	// No write barrier for lookalike SliceHeader.
	hdr.Data = uintptr(unsafe.Pointer(p))
}
