// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"reflect"
	"runtime"
	"testing"
	"unsafe"
)

// Assert that the size of important structures do not change unexpectedly.

func TestSizeof(t *testing.T) {
	const _64bit = unsafe.Sizeof(uintptr(0)) == 8

	var tests = []struct {
		val     any     // type as a value
		_32bit  uintptr // size on 32bit platforms
		_64bit  uintptr // size on 64bit platforms
		_wasm32 uintptr // size with 32-bit pointers and 64-bit alignment
	}{
		{Sym{}, 32, 64, 32},
		{Type{}, 60, 96, 64},
		{Map{}, 12, 24, 12},
		{Forward{}, 20, 32, 20},
		{Func{}, 32, 56, 32},
		{Struct{}, 12, 24, 12},
		{Interface{}, 0, 0, 0},
		{Chan{}, 8, 16, 8},
		{Array{}, 12, 16, 16},
		{FuncArgs{}, 4, 8, 4},
		{ChanArgs{}, 4, 8, 4},
		{Ptr{}, 4, 8, 4},
		{Slice{}, 4, 8, 4},
	}

	for _, tt := range tests {
		want := tt._32bit
		if runtime.GOARCH == "wasm32" {
			want = tt._wasm32
		} else if _64bit {
			want = tt._64bit
		}
		got := reflect.TypeOf(tt.val).Size()
		if want != got {
			t.Errorf("unsafe.Sizeof(%T) = %d, want %d", tt.val, got, want)
		}
	}
}
