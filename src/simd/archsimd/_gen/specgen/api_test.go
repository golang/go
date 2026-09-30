// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"simd/archsimd/_gen/specgen/specexpr"
	"testing"
)

func TestFuncSignature(t *testing.T) {
	int32x4 := specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(128)}
	intType := specexpr.Basic{Base: "int", Bits: 0}

	tests := []struct {
		name string
		fn   Func
		want string
	}{
		{
			name: "method with named receiver, in, and out",
			fn: Func{
				Name: "Add",
				Recv: Arg{Name: "x", Type: int32x4},
				In:   []Arg{{Name: "y", Type: int32x4}},
				Out:  []Arg{{Name: "z", Type: int32x4}},
			},
			want: "func (x Int32x4) Add(y Int32x4) (z Int32x4)",
		},
		{
			name: "method with omitted names",
			fn: Func{
				Name: "Add",
				Recv: Arg{Name: "", Type: int32x4},
				In:   []Arg{{Name: "", Type: int32x4}},
				Out:  []Arg{{Name: "", Type: int32x4}},
			},
			want: "func (Int32x4) Add(Int32x4) Int32x4",
		},
		{
			name: "method with multiple unnamed results",
			fn: Func{
				Name: "DivMod",
				Recv: Arg{Name: "x", Type: int32x4},
				In:   []Arg{{Name: "", Type: int32x4}},
				Out:  []Arg{{Name: "", Type: int32x4}, {Name: "", Type: intType}},
			},
			want: "func (x Int32x4) DivMod(Int32x4) (Int32x4, int)",
		},
		{
			name: "method with mixed named and unnamed",
			fn: Func{
				Name: "DivMod",
				Recv: Arg{Name: "x", Type: int32x4},
				In:   []Arg{{Name: "", Type: int32x4}},
				Out:  []Arg{{Name: "", Type: int32x4}, {Name: "err", Type: intType}},
			},
			want: "func (x Int32x4) DivMod(Int32x4) (_ Int32x4, err int)",
		},
		{
			name: "package function with no receiver and no results",
			fn: Func{
				Name: "Reset",
			},
			want: "func Reset()",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.fn.Signature()
			if got != tc.want {
				t.Errorf("Signature() = %q, want %q", got, tc.want)
			}
		})
	}
}
