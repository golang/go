// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/goexperiment"
	"reflect"
	"runtime"
	"testing"
	"unsafe"
)

// Assert that the size of important structures do not change unexpectedly.

func TestSizeof(t *testing.T) {
	const _64bit = unsafe.Sizeof(uintptr(0)) == 8

	g32bit := uintptr(264)
	if goexperiment.ExecTracer2 {
		// gTraceState changed from 2 uint64, 1 pointer, 1 bool to 2 uint64, 3 uint32.
		// On 32-bit, that's one extra word.
		g32bit += 4
	}

	var tests = []struct {
		val    any     // type as a value
		_32bit uintptr // size on 32bit platforms
		_64bit uintptr // size on 64bit platforms
	}{
		{runtime.G{}, g32bit, 432}, // g, but exported for testing
		{runtime.Sudog{}, 56, 88},  // sudog, but exported for testing
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
