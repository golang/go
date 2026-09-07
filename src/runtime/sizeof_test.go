// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"reflect"
	"runtime"
	"testing"
	"unsafe"
)

// Assert that the size of important structures do not change unexpectedly.
// When a size changes, also report whether the structure moves to a
// different size class, because that changes how much memory each object
// takes.

func TestSizeof(t *testing.T) {
	const _64bit = unsafe.Sizeof(uintptr(0)) == 8
	const xreg = unsafe.Sizeof(runtime.XRegPerG{})         // Varies per architecture
	const valgrind = unsafe.Sizeof(runtime.ValgrindPerG{}) // Non-zero only with the valgrind build tag
	// Non-zero only with GOWASM=stackswitch, where g.wasmCont and the
	// padding after it add 8 bytes.
	const wasmCont = 2 * unsafe.Sizeof(runtime.WasmContPerG{})
	var tests = []struct {
		val     any     // type as a value
		_32bit  uintptr // size on 32bit platforms
		_64bit  uintptr // size on 64bit platforms
		_wasm32 uintptr // size with 32-bit pointers and 64-bit alignment
	}{
		{runtime.G{}, 244 + xreg + valgrind, 376 + xreg + valgrind + wasmCont, 256 + xreg + valgrind + wasmCont}, // g, but exported for testing
		{runtime.WaitingG{}, 68, 112, 72}, // waitingG, but exported for testing
	}

	if xreg > runtime.PtrSize {
		t.Errorf("unsafe.Sizeof(xRegPerG) = %d, want <= %d", xreg, runtime.PtrSize)
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
			gotClass, wantClass := runtime.AllocSize(got), runtime.AllocSize(want)
			if gotClass != wantClass {
				t.Errorf("%T moves from the %d-byte to the %d-byte size class, %+d bytes per object", tt.val, wantClass, gotClass, int(gotClass)-int(wantClass))
			}
		}
	}
}
