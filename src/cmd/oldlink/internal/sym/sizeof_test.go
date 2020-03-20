// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import (
	"reflect"
	"testing"
	"unsafe"
)

// Assert that the size of important structures do not change unexpectedly.

func TestSizeof(t *testing.T) {
	const nbit = unsafe.Sizeof(uintptr(0)) * 8
	const _64bit = nbit == 64

	var tests = []struct {
		val    interface{} // type as a value
		_32bit uintptr     // size on 32bit platforms
		_64bit uintptr     // size on 64bit platforms
	}{
		{Symbol{}, 108, 176},
	}

	for _, tt := range tests {
		want := tt._32bit
		if _64bit {
			want = tt._64bit
		}
		got := reflect.TypeOf(tt.val).Size()
		if want != got {
			t.Errorf("%d bit unsafe.Sizeof(%T) = %d, want %d", nbit, tt.val, got, want)
		}
	}
}
