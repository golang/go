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

func TestSizeof(t *testing.T) {
	const _64bit = unsafe.Sizeof(uintptr(0)) == 8
	const xreg = unsafe.Sizeof(runtime.XRegPerG{}) // Varies per architecture
	var tests = []struct {
		val    any     // type as a value
		_32bit uintptr // size on 32bit platforms
		_64bit uintptr // size on 64bit platforms
	}{
		{runtime.G{}, 284 + xreg, 448 + xreg}, // g, but exported for testing
		{runtime.Sudog{}, 64, 104},            // sudog, but exported for testing
	}

	if xreg > runtime.PtrSize {
		t.Errorf("unsafe.Sizeof(xRegPerG) = %d, want <= %d", xreg, runtime.PtrSize)
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
