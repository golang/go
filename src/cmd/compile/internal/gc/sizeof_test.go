// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl

package gc

import (
	"cmd/compile/internal/types"
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
		{Func{}, 96, 160},
		{Name{}, 36, 56},
		{Param{}, 28, 56},
		{Node{}, 84, 136},
		// TODO(gri) test the ones below in the types package
		{types.Sym{}, 60, 104},
		{types.Type{}, 52, 88},
		{types.MapType{}, 20, 40},
		{types.ForwardType{}, 20, 32},
		{types.FuncType{}, 28, 48},
		{types.StructType{}, 12, 24},
		{types.InterType{}, 4, 8},
		{types.ChanType{}, 8, 16},
		{types.ArrayType{}, 12, 16},
		{types.DDDFieldType{}, 4, 8},
		{types.FuncArgsType{}, 4, 8},
		{types.ChanArgsType{}, 4, 8},
		{types.PtrType{}, 4, 8},
		{types.SliceType{}, 4, 8},
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
