// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import "testing"

func TestSingleRoundingFloatConversions(t *testing.T) {
	const (
		unsigned       = uint64(9223372586610589697)
		signed         = int64(4611686293305294849)
		exact          = 1 + 0x1p-24 + 0x1p-80
		wantUnsigned   = float32(unsigned)
		wantSigned     = float32(signed)
		wantNegative   = float32(-signed)
		wantExact      = float32(exact)
		wantWide       = float64(unsigned)
		wantNarrowWide = float32(float64(unsigned))
	)
	var u uint64 = unsigned
	var s int64 = signed
	for _, tt := range []struct {
		name  string
		typ   string
		param string
		expr  string
		arg   string
		want  float64
	}{
		{"uint64", "float32", "", "float32(uint64(9223372586610589697))", "", float64(wantUnsigned)},
		{"int64", "float32", "", "float32(int64(4611686293305294849))", "", float64(wantSigned)},
		{"negativeInt64", "float32", "", "float32(int64(-4611686293305294849))", "", float64(wantNegative)},
		{"untypedInt", "float32", "", "float32(9223372586610589697)", "", float64(wantUnsigned)},
		{"untypedNegativeInt", "float32", "", "float32(-4611686293305294849)", "", float64(wantNegative)},
		{"exactRational", "float32", "", "float32(1+0x1p-24+0x1p-80)", "", float64(wantExact)},
		{"uint64Parameter", "float32", "uint64", "float32(x)", "uint64(9223372586610589697)", float64(float32(u))},
		{"int64Parameter", "float32", "int64", "float32(x)", "int64(4611686293305294849)", float64(float32(s))},
		{"negativeInt64Parameter", "float32", "int64", "float32(x)", "int64(-4611686293305294849)", float64(float32(-s))},
		{"float64Control", "float64", "", "float64(uint64(9223372586610589697))", "", wantWide},
		{"float64ParameterControl", "float64", "uint64", "float64(x)", "uint64(9223372586610589697)", float64(u)},
		{"typedFloat64Control", "float32", "", "float32(float64(9223372586610589697))", "", float64(wantNarrowWide)},
		{"float64NarrowParameterControl", "float32", "float64", "float32(x)", "float64(9223372586610589697)", float64(float32(float64(u)))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			params := ""
			var ptypes []*Type
			if tt.param != "" {
				params = "x " + tt.param
				ptypes = []*Type{BasicType(tt.param)}
			}
			v, _, err := evalConstFunc(t, Config{}, constSpec{
				src:    "package p\nfunc F(" + params + ") " + tt.typ + " { return " + tt.expr + " }",
				ptypes: ptypes,
				rtypes: []*Type{BasicType(tt.typ)},
				name:   "F",
				call:   "F(" + tt.arg + ")",
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := v.Float64(); got != tt.want {
				t.Fatalf("%s = %.20g, want %.20g", tt.expr, got, tt.want)
			}
		})
	}
}

func TestSingleRoundingComplexConversions(t *testing.T) {
	const (
		unsigned       = uint64(9223372586610589697)
		signed         = int64(4611686293305294849)
		wantUnsigned   = complex64(unsigned)
		wantSigned     = complex64(signed)
		wantNegative   = complex64(-signed)
		wantExact      = complex64(1 + 0x1p-24 + 0x1p-80)
		wantWide       = complex128(unsigned)
		wantNarrowWide = complex64(float64(unsigned))
	)
	var u uint64 = unsigned
	var s int64 = signed
	for _, tt := range []struct {
		name  string
		typ   string
		param string
		expr  string
		arg   string
		want  complex128
	}{
		{"uint64", "complex64", "", "complex64(uint64(9223372586610589697))", "", complex128(wantUnsigned)},
		{"int64", "complex64", "", "complex64(int64(4611686293305294849))", "", complex128(wantSigned)},
		{"negativeInt64", "complex64", "", "complex64(int64(-4611686293305294849))", "", complex128(wantNegative)},
		{"untypedInt", "complex64", "", "complex64(9223372586610589697)", "", complex128(wantUnsigned)},
		{"exactRational", "complex64", "", "complex64(1+0x1p-24+0x1p-80)", "", complex128(wantExact)},
		{"uint64Parameter", "complex64", "uint64", "complex64(x)", "uint64(9223372586610589697)", complex(float64(float32(u)), 0)},
		{"int64Parameter", "complex64", "int64", "complex64(x)", "int64(4611686293305294849)", complex(float64(float32(s)), 0)},
		{"negativeInt64Parameter", "complex64", "int64", "complex64(x)", "int64(-4611686293305294849)", complex(float64(float32(-s)), 0)},
		{"complex128Control", "complex128", "", "complex128(uint64(9223372586610589697))", "", wantWide},
		{"complex128ParameterControl", "complex128", "uint64", "complex128(x)", "uint64(9223372586610589697)", complex(float64(u), 0)},
		{"typedFloat64Control", "complex64", "", "complex64(float64(9223372586610589697))", "", complex128(wantNarrowWide)},
		{"float64ParameterControl", "complex64", "float64", "complex64(x)", "float64(9223372586610589697)", complex(float64(float32(float64(u))), 0)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			params := ""
			var ptypes []*Type
			if tt.param != "" {
				params = "x " + tt.param
				ptypes = []*Type{BasicType(tt.param)}
			}
			v, _, err := evalConstFunc(t, Config{}, constSpec{
				src:    "package p\nfunc F(" + params + ") " + tt.typ + " { return " + tt.expr + " }",
				ptypes: ptypes,
				rtypes: []*Type{BasicType(tt.typ)},
				name:   "F",
				call:   "F(" + tt.arg + ")",
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

func TestSingleRoundingFloat32Arithmetic(t *testing.T) {
	const (
		add = float32(1) + float32(0x1p-24)
		sub = float32(1) - float32(0x1p-25)
		mul = float32(1+0x1p-23) * float32(1-0x1p-23)
		quo = float32(1) / float32(3)
	)
	for _, tt := range []struct {
		name string
		expr string
		want float32
	}{
		{"add", "float32(1) + float32(0x1p-24)", add},
		{"sub", "float32(1) - float32(0x1p-25)", sub},
		{"mul", "float32(1+0x1p-23) * float32(1-0x1p-23)", mul},
		{"quo", "float32(1) / float32(3)", quo},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v, _, err := evalConstFunc(t, Config{}, constSpec{
				src:    "package p\nfunc F() float32 { return " + tt.expr + " }",
				rtypes: []*Type{BasicType("float32")},
				name:   "F",
				call:   "F()",
			})
			if err != nil {
				t.Fatal(err)
			}
			if got, want := v.Float64(), float64(tt.want); got != want {
				t.Fatalf("%s = %.20g, want %.20g", tt.expr, got, want)
			}
		})
	}
}
