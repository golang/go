// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl

package gc

import (
	"reflect"
	"testing"
	"unsafe"
)

// Assert that the size of important structures do not change unexpectedly.

func TestSizeof(t *testing.T) {
	const _64bit = unsafe.Sizeof(uintptr(0)) == 8

	var tests = []struct {
		val    interface{} // type as a value
		_32bit uintptr     // size on 32bit platforms
		_64bit uintptr     // size on 64bit platforms
	}{
		{Func{}, 92, 160},
		{Name{}, 44, 72},
		{Param{}, 24, 48},
		{Node{}, 92, 144},
		{Sym{}, 60, 112},
		{Type{}, 60, 96},
		{MapType{}, 20, 40},
		{ForwardType{}, 16, 32},
		{FuncType{}, 28, 48},
		{StructType{}, 12, 24},
		{InterType{}, 4, 8},
		{ChanType{}, 8, 16},
		{ArrayType{}, 16, 24},
		{InterMethType{}, 4, 8},
		{DDDFieldType{}, 4, 8},
		{FuncArgsType{}, 4, 8},
		{ChanArgsType{}, 4, 8},
		{PtrType{}, 4, 8},
		{SliceType{}, 4, 8},
	}

	for _, tt := range tests {
		want := tt._32bit
		if _64bit {
			want = tt._64bit
		}
		got := reflect.TypeOf(tt.val).Size()
		if want != got {
			t.Errorf("unsafe.Sizeof(%T) = %d, want %d", tt.val, got, want)
		}
	}
}
