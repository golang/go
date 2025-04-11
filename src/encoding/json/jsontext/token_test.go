// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"math"
	"reflect"
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
	type token struct {
		Bool   bool
		String string
		Float  float64
		Int    int64
		Uint   uint64
		Kind   Kind
	}

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
		{Float(0), token{String: "0", Float: 0, Int: 0, Uint: 0, Kind: '0'}},
		{Float(math.Copysign(0, -1)), token{String: "-0", Float: math.Copysign(0, -1), Int: 0, Uint: 0, Kind: '0'}},
		{Float(math.NaN()), token{String: "NaN", Float: math.NaN(), Int: 0, Uint: 0, Kind: '"'}},
		{Float(math.Inf(+1)), token{String: "Infinity", Float: math.Inf(+1), Kind: '"'}},
		{Float(math.Inf(-1)), token{String: "-Infinity", Float: math.Inf(-1), Kind: '"'}},
		{Int(minInt64), token{String: "-9223372036854775808", Float: minInt64, Int: minInt64, Uint: minUint64, Kind: '0'}},
		{Int(minInt64 + 1), token{String: "-9223372036854775807", Float: minInt64 + 1, Int: minInt64 + 1, Uint: minUint64, Kind: '0'}},
		{Int(-1), token{String: "-1", Float: -1, Int: -1, Uint: minUint64, Kind: '0'}},
		{Int(0), token{String: "0", Float: 0, Int: 0, Uint: 0, Kind: '0'}},
		{Int(+1), token{String: "1", Float: +1, Int: +1, Uint: +1, Kind: '0'}},
		{Int(maxInt64 - 1), token{String: "9223372036854775806", Float: maxInt64 - 1, Int: maxInt64 - 1, Uint: maxInt64 - 1, Kind: '0'}},
		{Int(maxInt64), token{String: "9223372036854775807", Float: maxInt64, Int: maxInt64, Uint: maxInt64, Kind: '0'}},
		{Uint(minUint64), token{String: "0", Kind: '0'}},
		{Uint(minUint64 + 1), token{String: "1", Float: minUint64 + 1, Int: minUint64 + 1, Uint: minUint64 + 1, Kind: '0'}},
		{Uint(maxUint64 - 1), token{String: "18446744073709551614", Float: maxUint64 - 1, Int: maxInt64, Uint: maxUint64 - 1, Kind: '0'}},
		{Uint(maxUint64), token{String: "18446744073709551615", Float: maxUint64, Int: maxInt64, Uint: maxUint64, Kind: '0'}},
		{rawToken(`-0`), token{String: "-0", Float: math.Copysign(0, -1), Int: 0, Uint: 0, Kind: '0'}},
		{rawToken(`1e1000`), token{String: "1e1000", Float: math.MaxFloat64, Int: maxInt64, Uint: maxUint64, Kind: '0'}},
		{rawToken(`-1e1000`), token{String: "-1e1000", Float: -math.MaxFloat64, Int: minInt64, Uint: minUint64, Kind: '0'}},
		{rawToken(`0.1`), token{String: "0.1", Float: 0.1, Int: 0, Uint: 0, Kind: '0'}},
		{rawToken(`0.5`), token{String: "0.5", Float: 0.5, Int: 0, Uint: 0, Kind: '0'}},
		{rawToken(`0.9`), token{String: "0.9", Float: 0.9, Int: 0, Uint: 0, Kind: '0'}},
		{rawToken(`1.1`), token{String: "1.1", Float: 1.1, Int: 1, Uint: 1, Kind: '0'}},
		{rawToken(`-0.1`), token{String: "-0.1", Float: -0.1, Int: 0, Uint: 0, Kind: '0'}},
		{rawToken(`-0.5`), token{String: "-0.5", Float: -0.5, Int: 0, Uint: 0, Kind: '0'}},
		{rawToken(`-0.9`), token{String: "-0.9", Float: -0.9, Int: 0, Uint: 0, Kind: '0'}},
		{rawToken(`-1.1`), token{String: "-1.1", Float: -1.1, Int: -1, Uint: 0, Kind: '0'}},
		{rawToken(`99999999999999999999`), token{String: "99999999999999999999", Float: 1e20 - 1, Int: maxInt64, Uint: maxUint64, Kind: '0'}},
		{rawToken(`-99999999999999999999`), token{String: "-99999999999999999999", Float: -1e20 - 1, Int: minInt64, Uint: minUint64, Kind: '0'}},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got := token{
				Bool: func() bool {
					defer func() { recover() }()
					return tt.in.Bool()
				}(),
				String: tt.in.String(),
				Float: func() float64 {
					defer func() { recover() }()
					return tt.in.Float()
				}(),
				Int: func() int64 {
					defer func() { recover() }()
					return tt.in.Int()
				}(),
				Uint: func() uint64 {
					defer func() { recover() }()
					return tt.in.Uint()
				}(),
				Kind: tt.in.Kind(),
			}

			if got.Bool != tt.want.Bool {
				t.Errorf("Token(%s).Bool() = %v, want %v", tt.in, got.Bool, tt.want.Bool)
			}
			if got.String != tt.want.String {
				t.Errorf("Token(%s).String() = %v, want %v", tt.in, got.String, tt.want.String)
			}
			if math.Float64bits(got.Float) != math.Float64bits(tt.want.Float) {
				t.Errorf("Token(%s).Float() = %v, want %v", tt.in, got.Float, tt.want.Float)
			}
			if got.Int != tt.want.Int {
				t.Errorf("Token(%s).Int() = %v, want %v", tt.in, got.Int, tt.want.Int)
			}
			if got.Uint != tt.want.Uint {
				t.Errorf("Token(%s).Uint() = %v, want %v", tt.in, got.Uint, tt.want.Uint)
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
		t.Run("", func(t *testing.T) {
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
