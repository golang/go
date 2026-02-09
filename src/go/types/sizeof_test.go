// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"reflect"
	"testing"
)

// Signal size changes of important structures.
func TestSizeof(t *testing.T) {
	const _64bit = ^uint(0)>>32 != 0

	var tests = []struct {
		val    any     // type as a value
		_32bit uintptr // size on 32bit platforms
		_64bit uintptr // size on 64bit platforms
	}{
		// Types
		{Basic{}, 16, 32},
		{Array{}, 16, 24},
		{Slice{}, 8, 16},
		{Struct{}, 24, 48},
		{Pointer{}, 8, 16},
		{Tuple{}, 12, 24},
		{Signature{}, 28, 56},
		{Union{}, 12, 24},
		{Interface{}, 40, 80},
		{Map{}, 16, 32},
		{Chan{}, 12, 24},
		{Named{}, 68, 128},
		{TypeParam{}, 28, 48},
		{term{}, 12, 24},

		// Objects
		{PkgName{}, 40, 80},
		{Const{}, 44, 88},
		{TypeName{}, 36, 72},
		{Var{}, 44, 88},
		{Func{}, 44, 88},
		{Label{}, 40, 80},
		{Builtin{}, 40, 80},
		{Nil{}, 36, 72},

		// Misc
		{Scope{}, 44, 88},
		{Package{}, 44, 88},
		{_TypeSet{}, 28, 56},
	}
	for _, test := range tests {
		got := reflect.TypeOf(test.val).Size()
		want := test._32bit
		if _64bit {
			want = test._64bit
		}
		if got != want {
			t.Errorf("unsafe.Sizeof(%T) = %d, want %d", test.val, got, want)
		}
	}
}
