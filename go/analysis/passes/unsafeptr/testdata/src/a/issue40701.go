// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"reflect"
	"unsafe"
)

// Explicitly allocating a variable of type reflect.SliceHeader.
func _(p *byte, n int) []byte {
	var sh reflect.SliceHeader
	sh.Data = uintptr(unsafe.Pointer(p))
	sh.Len = n
	sh.Cap = n
	return *(*[]byte)(unsafe.Pointer(&sh)) // want "possible misuse of reflect.SliceHeader"
}

// Implicitly allocating a variable of type reflect.SliceHeader.
func _(p *byte, n int) []byte {
	return *(*[]byte)(unsafe.Pointer(&reflect.SliceHeader{ // want "possible misuse of reflect.SliceHeader"
		Data: uintptr(unsafe.Pointer(p)),
		Len:  n,
		Cap:  n,
	}))
}

// Use reflect.StringHeader as a composite literal value.
func _(p *byte, n int) []byte {
	var res []byte
	*(*reflect.StringHeader)(unsafe.Pointer(&res)) = reflect.StringHeader{ // want "possible misuse of reflect.StringHeader"
		Data: uintptr(unsafe.Pointer(p)),
		Len:  n,
	}
	return res
}

func _() {
	// don't crash when obj.Pkg() == nil
	var err error
	_ = &err
}
