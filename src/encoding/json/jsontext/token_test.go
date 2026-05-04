// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"math"
	"reflect"
	"strconv"
	"testing"
)

func TestTokenStringAllocations(t *testing.T) {
	if testing.CoverMode() != "" {
		t.Skip("coverage mode breaks the compiler optimization this depends on")
	}

	tok := rawToken(`"hello"`)
	var m map[string]bool
	got := int(testing.AllocsPerRun(10, func() {
		// This function uses tok.String() is a non-escaping manner
		// (i.e., looking it up in a Go map). It should not allocate.
		if m[tok.String()] {
			panic("never executed")
		}
	}))
	if got > 0 {
		t.Errorf("Token.String allocated %d times, want 0", got)
	}
}

func TestTokenAccessors(t *testing.T) {
	type valueError[T any] struct {
		Value T
		Error error
	}
	type token struct {
		Bool    bool
		String  string
		Float32 valueError[float32]
		Float   valueError[float64]
		Int     valueError[int64]
		Uint    valueError[uint64]
		Kind    Kind
	}
	negZero := math.Copysign(0, -1)
	errRange := &SyntacticError{Err: strconv.ErrRange}
	errSyntax := &SyntacticError{Err: strconv.ErrSyntax}
	f32 := func(f32 float32) valueError[float32] { return valueError[float32]{Value: f32} }
	f32er := func(f32 float32) valueError[float32] { return valueError[float32]{Value: f32, Error: errRange} }
	f64 := func(f64 float64) valueError[float64] { return valueError[float64]{Value: f64} }
	f64er := func(f64 float64) valueError[float64] { return valueError[float64]{Value: f64, Error: errRange} }
	i64 := func(i64 int64) valueError[int64] { return valueError[int64]{Value: i64} }
	i64er := func(i64 int64) valueError[int64] { return valueError[int64]{Value: i64, Error: errRange} }
	i64es := func(i64 int64) valueError[int64] { return valueError[int64]{Value: i64, Error: errSyntax} }
	u64 := func(u64 uint64) valueError[uint64] { return valueError[uint64]{Value: u64} }
	u64er := func(u64 uint64) valueError[uint64] { return valueError[uint64]{Value: u64, Error: errRange} }
	u64es := func(u64 uint64) valueError[uint64] { return valueError[uint64]{Value: u64, Error: errSyntax} }

	tests := []struct {
		in   Token
		want token
	}{
		{Token{}, token{String: "<invalid jsontext.Token>"}},
		{Null, token{String: "null", Kind: 'n'}},
		{False, token{Bool: false, String: "false", Kind: 'f'}},
		{True, token{Bool: true, String: "true", Kind: 't'}},
		{Bool(false), token{Bool: false, String: "false", Kind: 'f'}},
		{Bool(true), token{Bool: true, String: "true", Kind: 't'}},
		{BeginObject, token{String: "{", Kind: '{'}},
		{EndObject, token{String: "}", Kind: '}'}},
		{BeginArray, token{String: "[", Kind: '['}},
		{EndArray, token{String: "]", Kind: ']'}},
		{String(""), token{String: "", Kind: '"'}},
		{String("hello, world!"), token{String: "hello, world!", Kind: '"'}},
		{rawToken(`"hello, world!"`), token{String: "hello, world!", Kind: '"'}},
		{Float32(float32(0)), token{String: "0", Float32: f32(0), Float: f64(0), Int: i64(0), Uint: u64(0), Kind: '0'}},
		{Float32(float32(math.Copysign(0, -1))), token{String: "-0", Float32: f32(float32(negZero)), Float: f64(negZero), Int: i64(0), Uint: u64es(0), Kind: '0'}},
		{Float32(float32(math.NaN())), token{String: "NaN", Float32: f32(float32(math.NaN())), Float: f64(math.NaN()), Kind: '"'}},
		{Float32(float32(math.Inf(+1))), token{String: "Infinity", Float32: f32(float32(math.Inf(+1))), Float: f64(math.Inf(+1)), Kind: '"'}},
		{Float32(float32(math.Inf(-1))), token{String: "-Infinity", Float32: f32(float32(math.Inf(-1))), Float: f64(math.Inf(-1)), Kind: '"'}},
		{Float32(float32(math.Pi)), token{String: "3.1415927", Float32: f32(math.Pi), Float: f64(float64(float32(math.Pi))), Int: i64es(3), Uint: u64es(3), Kind: '0'}},
		{Float32(float32(-1 * math.MaxFloat32)), token{String: "-3.4028235e+38", Float32: f32(float32(-1 * math.MaxFloat32)), Float: f64(-1 * math.MaxFloat32), Int: i64er(minInt64), Uint: u64es(minUint64), Kind: '0'}},
		{Float32(float32(+1 * math.MaxFloat32)), token{String: "3.4028235e+38", Float32: f32(float32(+1 * math.MaxFloat32)), Float: f64(+1 * math.MaxFloat32), Int: i64er(maxInt64), Uint: u64er(maxUint64), Kind: '0'}},
		{Float32(float32(123)), token{String: "123", Float32: f32(123), Float: f64(123), Int: i64(123), Uint: u64(123), Kind: '0'}},
		{Float(0), token{String: "0", Float32: f32(0), Float: f64(0), Int: i64(0), Uint: u64(0), Kind: '0'}},
		{Float(negZero), token{String: "-0", Float32: f32(float32(negZero)), Float: f64(negZero), Int: i64(0), Uint: u64es(0), Kind: '0'}},
		{Float(math.NaN()), token{String: "NaN", Float32: f32(float32(math.NaN())), Float: f64(math.NaN()), Int: i64(0), Uint: u64(0), Kind: '"'}},
		{Float(math.Inf(+1)), token{String: "Infinity", Float32: f32(float32(math.Inf(+1))), Float: f64(math.Inf(+1)), Kind: '"'}},
		{Float(math.Inf(-1)), token{String: "-Infinity", Float32: f32(float32(math.Inf(-1))), Float: f64(math.Inf(-1)), Kind: '"'}},
		{Float(math.Pi), token{String: "3.141592653589793", Float32: f32(math.Pi), Float: f64(math.Pi), Int: i64es(3), Uint: u64es(3), Kind: '0'}},
		{Float(-1 * math.MaxFloat64), token{String: "-1.7976931348623157e+308", Float32: f32er(float32(math.Inf(-1))), Float: f64(-1 * math.MaxFloat64), Int: i64er(minInt64), Uint: u64es(minUint64), Kind: '0'}},
		{Float(+1 * math.MaxFloat64), token{String: "1.7976931348623157e+308", Float32: f32er(float32(math.Inf(+1))), Float: f64(+1 * math.MaxFloat64), Int: i64er(maxInt64), Uint: u64er(maxUint64), Kind: '0'}},
		{Float(123), token{String: "123", Float32: f32(123), Float: f64(123), Int: i64(123), Uint: u64(123), Kind: '0'}},
		{Int(minInt64), token{String: "-9223372036854775808", Float32: f32(minInt64), Float: f64(minInt64), Int: i64(minInt64), Uint: u64es(minUint64), Kind: '0'}},
		{Int(minInt64 + 1), token{String: "-9223372036854775807", Float32: f32(minInt64 + 1), Float: f64(minInt64 + 1), Int: i64(minInt64 + 1), Uint: u64es(minUint64), Kind: '0'}},
		{Int(-1), token{String: "-1", Float32: f32(-1), Float: f64(-1), Int: i64(-1), Uint: u64es(minUint64), Kind: '0'}},
		{Int(0), token{String: "0", Float32: f32(0), Float: f64(0), Int: i64(0), Uint: u64(0), Kind: '0'}},
		{Int(+1), token{String: "1", Float32: f32(+1), Float: f64(+1), Int: i64(+1), Uint: u64(+1), Kind: '0'}},
		{Int(maxInt64 - 1), token{String: "9223372036854775806", Float32: f32(maxInt64 - 1), Float: f64(maxInt64 - 1), Int: i64(maxInt64 - 1), Uint: u64(maxInt64 - 1), Kind: '0'}},
		{Int(maxInt64), token{String: "9223372036854775807", Float32: f32(maxInt64), Float: f64(maxInt64), Int: i64(maxInt64), Uint: u64(maxInt64), Kind: '0'}},
		{Uint(minUint64), token{String: "0", Kind: '0'}},
		{Uint(minUint64 + 1), token{String: "1", Float32: f32(minUint64 + 1), Float: f64(minUint64 + 1), Int: i64(minUint64 + 1), Uint: u64(minUint64 + 1), Kind: '0'}},
		{Uint(maxUint64 - 1), token{String: "18446744073709551614", Float32: f32(maxUint64 - 1), Float: f64(maxUint64 - 1), Int: i64er(maxInt64), Uint: u64(maxUint64 - 1), Kind: '0'}},
		{Uint(maxUint64), token{String: "18446744073709551615", Float32: f32(maxUint64 - 1), Float: f64(maxUint64 - 1), Int: i64er(maxInt64), Uint: u64(maxUint64), Kind: '0'}},
		{rawToken(`-0`), token{String: "-0", Float32: f32(float32(negZero)), Float: f64(negZero), Int: i64(0), Uint: u64es(0), Kind: '0'}},
		{rawToken(`1e1000`), token{String: "1e1000", Float32: f32er(float32(math.Inf(+1))), Float: f64er(float64(math.Inf(+1))), Int: i64es(maxInt64), Uint: u64es(maxUint64), Kind: '0'}},
		{rawToken(`-1e1000`), token{String: "-1e1000", Float32: f32er(float32(math.Inf(-1))), Float: f64er(float64(math.Inf(-1))), Int: i64es(minInt64), Uint: u64es(minUint64), Kind: '0'}},
		{rawToken(`0.1`), token{String: "0.1", Float32: f32(0.1), Float: f64(0.1), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{rawToken(`0.5`), token{String: "0.5", Float32: f32(0.5), Float: f64(0.5), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{rawToken(`0.9`), token{String: "0.9", Float32: f32(0.9), Float: f64(0.9), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{rawToken(`1.0`), token{String: "1.0", Float32: f32(1.0), Float: f64(1.0), Int: i64es(1), Uint: u64es(1), Kind: '0'}},
		{rawToken(`1.1`), token{String: "1.1", Float32: f32(1.1), Float: f64(1.1), Int: i64es(1), Uint: u64es(1), Kind: '0'}},
		{rawToken(`123`), token{String: "123", Float32: f32(123), Float: f64(123), Int: i64(123), Uint: u64(123), Kind: '0'}},
		{rawToken(`-0.1`), token{String: "-0.1", Float32: f32(-0.1), Float: f64(-0.1), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{rawToken(`-0.5`), token{String: "-0.5", Float32: f32(-0.5), Float: f64(-0.5), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{rawToken(`-0.9`), token{String: "-0.9", Float32: f32(-0.9), Float: f64(-0.9), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{rawToken(`-1.0`), token{String: "-1.0", Float32: f32(-1.0), Float: f64(-1.0), Int: i64es(-1), Uint: u64es(0), Kind: '0'}},
		{rawToken(`-1.1`), token{String: "-1.1", Float32: f32(-1.1), Float: f64(-1.1), Int: i64es(-1), Uint: u64es(0), Kind: '0'}},
		{rawToken(`-123`), token{String: "-123", Float32: f32(-123), Float: f64(-123), Int: i64(-123), Uint: u64es(0), Kind: '0'}},
		{rawToken(`99999999999999999999`), token{String: "99999999999999999999", Float32: f32(1e20 - 1), Float: f64(1e20 - 1), Int: i64er(maxInt64), Uint: u64er(maxUint64), Kind: '0'}},
		{rawToken(`-99999999999999999999`), token{String: "-99999999999999999999", Float32: f32(-1e20 - 1), Float: f64(-1e20 - 1), Int: i64er(minInt64), Uint: u64es(minUint64), Kind: '0'}},
		{rawToken(`3.1415927`), token{String: "3.1415927", Float32: f32(math.Pi), Float: f64(3.1415927), Int: i64es(3), Uint: u64es(3), Kind: '0'}},
		{rawToken(`3.141592653589793`), token{String: "3.141592653589793", Float32: f32(math.Pi), Float: f64(math.Pi), Int: i64es(3), Uint: u64es(3), Kind: '0'}},
		{rawToken(`-9223372036854775807`), token{String: "-9223372036854775807", Float32: f32(-1 << 63), Float: f64(-1 << 63), Int: i64(minInt64 + 1), Uint: u64es(minUint64), Kind: '0'}},
		{rawToken(`-9223372036854775808`), token{String: "-9223372036854775808", Float32: f32(-1 << 63), Float: f64(-1 << 63), Int: i64(minInt64), Uint: u64es(minUint64), Kind: '0'}},
		{rawToken(`-9223372036854775809`), token{String: "-9223372036854775809", Float32: f32(-1 << 63), Float: f64(-1 << 63), Int: i64er(minInt64), Uint: u64es(minUint64), Kind: '0'}},
		{rawToken(`9223372036854775806`), token{String: "9223372036854775806", Float32: f32(1 << 63), Float: f64(1 << 63), Int: i64(maxInt64 - 1), Uint: u64(maxInt64 - 1), Kind: '0'}},
		{rawToken(`9223372036854775807`), token{String: "9223372036854775807", Float32: f32(1 << 63), Float: f64(1 << 63), Int: i64(maxInt64), Uint: u64(maxInt64), Kind: '0'}},
		{rawToken(`9223372036854775808`), token{String: "9223372036854775808", Float32: f32(1 << 63), Float: f64(1 << 63), Int: i64er(maxInt64), Uint: u64(maxInt64 + 1), Kind: '0'}},
		{rawToken(`18446744073709551614`), token{String: "18446744073709551614", Float32: f32(1 << 64), Float: f64(1 << 64), Int: i64er(maxInt64), Uint: u64(maxUint64 - 1), Kind: '0'}},
		{rawToken(`18446744073709551615`), token{String: "18446744073709551615", Float32: f32(1 << 64), Float: f64(1 << 64), Int: i64er(maxInt64), Uint: u64(maxUint64), Kind: '0'}},
		{rawToken(`18446744073709551616`), token{String: "18446744073709551616", Float32: f32(1 << 64), Float: f64(1 << 64), Int: i64er(maxInt64), Uint: u64er(maxUint64), Kind: '0'}},

		// NOTE: There exist many raw JSON numbers where:
		//	float32(ParseFloat(s, 32)) != float32(ParseFloat(s, 64))
		// due to issues with double rounding in opposite directions.
		// This suggests the need for a Token.Float32 accessor.
		{rawToken(`9000000000.0000001`), token{String: "9000000000.0000001", Float32: f32(9000000000.0000001), Float: f64(9000000000.0000001), Int: i64es(9e9), Uint: u64es(9e9), Kind: '0'}},
		// NOTE: ±7.038531e-26 is the only 32-bit precision float where:
		//	f != float32(ParseFloat(FormatFloat(f, 32), 64))
		// assuming FormatFloat uses ECMA-262, 10th edition, section 7.1.12.1.
		{rawToken(`7.038531e-26`), token{String: "7.038531e-26", Float32: f32(7.038531e-26), Float: f64(7.038531e-26), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{Float32(7.038531e-26), token{String: "7.038531e-26", Float32: f32(7.038531e-26), Float: f64(7.038530691851209e-26), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
		{Float(7.038531e-26), token{String: "7.038531e-26", Float32: f32(7.0385313e-26), Float: f64(7.038531e-26), Int: i64es(0), Uint: u64es(0), Kind: '0'}},
	}

	for _, tt := range tests {
		t.Run(tt.in.String(), func(t *testing.T) {
			got := token{
				Bool: func() bool {
					defer func() { recover() }()
					return tt.in.Bool()
				}(),
				String: tt.in.String(),
				Float32: func() valueError[float32] {
					defer func() { recover() }()
					f32, err := tt.in.Float32()
					return valueError[float32]{f32, err}
				}(),
				Float: func() valueError[float64] {
					defer func() { recover() }()
					f64, err := tt.in.Float()
					return valueError[float64]{f64, err}
				}(),
				Int: func() valueError[int64] {
					defer func() { recover() }()
					i64, err := tt.in.Int()
					return valueError[int64]{i64, err}
				}(),
				Uint: func() valueError[uint64] {
					defer func() { recover() }()
					u64, err := tt.in.Uint()
					return valueError[uint64]{u64, err}
				}(),
				Kind: tt.in.Kind(),
			}

			if got.Bool != tt.want.Bool {
				t.Errorf("Token(%s).Bool() = %v, want %v", tt.in, got.Bool, tt.want.Bool)
			}
			if got.String != tt.want.String {
				t.Errorf("Token(%s).String() = %v, want %v", tt.in, got.String, tt.want.String)
			}
			if math.Float32bits(got.Float32.Value) != math.Float32bits(tt.want.Float32.Value) || !reflect.DeepEqual(got.Float32.Error, tt.want.Float32.Error) {
				t.Errorf("Token(%s).Float32() = (%v, %v), want (%v, %v)", tt.in, got.Float32.Value, got.Float32.Error, tt.want.Float32.Value, tt.want.Float32.Error)
			}
			if math.Float64bits(got.Float.Value) != math.Float64bits(tt.want.Float.Value) || !reflect.DeepEqual(got.Float.Error, tt.want.Float.Error) {
				t.Errorf("Token(%s).Float() = (%v, %v), want (%v, %v)", tt.in, got.Float.Value, got.Float.Error, tt.want.Float.Value, tt.want.Float.Error)
			}
			if got.Int.Value != tt.want.Int.Value || !reflect.DeepEqual(got.Int.Error, tt.want.Int.Error) {
				t.Errorf("Token(%s).Int() = (%v, %v), want (%v, %v)", tt.in, got.Int.Value, got.Int.Error, tt.want.Int.Value, tt.want.Int.Error)
			}
			if got.Uint.Value != tt.want.Uint.Value || !reflect.DeepEqual(got.Uint.Error, tt.want.Uint.Error) {
				t.Errorf("Token(%s).Uint() = (%v, %v), want (%v, %v)", tt.in, got.Uint.Value, got.Uint.Error, tt.want.Uint.Value, tt.want.Uint.Error)
			}
			if got.Kind != tt.want.Kind {
				t.Errorf("Token(%s).Kind() = %v, want %v", tt.in, got.Kind, tt.want.Kind)
			}
		})
	}
}

func TestTokenClone(t *testing.T) {
	tests := []struct {
		in           Token
		wantExactRaw bool
	}{
		{Token{}, true},
		{Null, true},
		{False, true},
		{True, true},
		{BeginObject, true},
		{EndObject, true},
		{BeginArray, true},
		{EndArray, true},
		{String("hello, world!"), true},
		{rawToken(`"hello, world!"`), false},
		{Float(3.14159), true},
		{rawToken(`3.14159`), false},
	}

	for _, tt := range tests {
		t.Run(tt.in.String(), func(t *testing.T) {
			got := tt.in.Clone()
			if !reflect.DeepEqual(got, tt.in) {
				t.Errorf("Token(%s) == Token(%s).Clone() = false, want true", tt.in, tt.in)
			}
			gotExactRaw := got.raw == tt.in.raw
			if gotExactRaw != tt.wantExactRaw {
				t.Errorf("Token(%s).raw == Token(%s).Clone().raw = %v, want %v", tt.in, tt.in, gotExactRaw, tt.wantExactRaw)
			}
		})
	}
}
