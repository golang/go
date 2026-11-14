// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import "testing"

func TestComplexBuiltinRoundsExactComponentsDirectly(t *testing.T) {
	const (
		wantImag64 = complex64(complex(float32(0), 1+0x1p-24+0x1p-80))
		wantReal64 = complex64(complex(1+0x1p-24+0x1p-80, float32(0)))
		want128    = complex(float64(0), 1+0x1p-24+0x1p-80)
	)
	for _, tt := range []struct {
		name   string
		params string
		ptypes []*Type
		typ    string
		expr   string
		call   string
		want   complex128
	}{
		{
			name: "constantImag64",
			typ:  "complex64",
			expr: "complex(float32(0), 1+0x1p-24+0x1p-80)",
			call: "F()",
			want: complex128(wantImag64),
		},
		{
			name: "constantReal64",
			typ:  "complex64",
			expr: "complex(1+0x1p-24+0x1p-80, float32(0))",
			call: "F()",
			want: complex128(wantReal64),
		},
		{
			name:   "parameterReal64",
			params: "x float32",
			ptypes: []*Type{BasicType("float32")},
			typ:    "complex64",
			expr:   "complex(x, 1+0x1p-24+0x1p-80)",
			call:   "F(0)",
			want:   complex128(wantImag64),
		},
		{
			name:   "parameterImag64",
			params: "x float32",
			ptypes: []*Type{BasicType("float32")},
			typ:    "complex64",
			expr:   "complex(1+0x1p-24+0x1p-80, x)",
			call:   "F(0)",
			want:   complex128(wantReal64),
		},
		{
			name: "constant128",
			typ:  "complex128",
			expr: "complex(float64(0), 1+0x1p-24+0x1p-80)",
			call: "F()",
			want: want128,
		},
		{
			name:   "parameter128",
			params: "x float64",
			ptypes: []*Type{BasicType("float64")},
			typ:    "complex128",
			expr:   "complex(x, 1+0x1p-24+0x1p-80)",
			call:   "F(0)",
			want:   want128,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v, _, err := evalConstFunc(t, Config{}, constSpec{
				src:    "package p\nfunc F(" + tt.params + ") " + tt.typ + " { return " + tt.expr + " }",
				ptypes: tt.ptypes,
				rtypes: []*Type{BasicType(tt.typ)},
				name:   "F",
				call:   tt.call,
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := v.Complex(); got != tt.want {
				t.Fatalf("%s = %v, want %v", tt.expr, got, tt.want)
			}
		})
	}
}
