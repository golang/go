// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"testing"
	"time"
)

type zeroVal struct{ n int }

func (z zeroVal) IsZero() bool { return z.n == 7 }

type zeroPtr struct{ n int }

func (z *zeroPtr) IsZero() bool { return z.n == 7 }

type zeroer interface{ IsZero() bool }

type namedSlice []int

func (s namedSlice) IsZero() bool { return len(s) == 3 }

type plainStruct struct {
	A int
	B string
}

type omitEmptyField[T any] struct {
	F T `json:"f,omitempty"`
}

type omitZeroField[T any] struct {
	F T `json:"f,omitzero"`
}

func omittedByStdEmpty[T any](t *testing.T, v T) bool {
	t.Helper()
	b, err := stdjson.Marshal(omitEmptyField[T]{v})
	if err != nil {
		t.Fatal(err)
	}
	return string(b) == "{}"
}

func omittedByStdZero[T any](t *testing.T, v T) bool {
	t.Helper()
	b, err := stdjson.Marshal(omitZeroField[T]{v})
	if err != nil {
		t.Fatal(err)
	}
	return string(b) == "{}"
}

func checkOmitEmpty[T any](t *testing.T, v T) {
	t.Helper()
	if got, want := OmitEmpty(v), omittedByStdEmpty(t, v); got != want {
		t.Errorf("OmitEmpty(%#v) = %v, encoding/json omitempty = %v", v, got, want)
	}
}

func checkOmitZero[T any](t *testing.T, v T) {
	t.Helper()
	if got, want := OmitZero(v), omittedByStdZero(t, v); got != want {
		t.Errorf("OmitZero(%#v) = %v, encoding/json omitzero = %v", v, got, want)
	}
}

func TestOmitEmptyParity(t *testing.T) {
	var np *int
	one := 1
	var m map[string]int
	checkOmitEmpty(t, 0)
	checkOmitEmpty(t, 5)
	checkOmitEmpty(t, int8(0))
	checkOmitEmpty(t, uint16(3))
	checkOmitEmpty(t, uintptr(0))
	checkOmitEmpty(t, "")
	checkOmitEmpty(t, "x")
	checkOmitEmpty(t, false)
	checkOmitEmpty(t, true)
	checkOmitEmpty(t, 0.0)
	checkOmitEmpty(t, float32(1.5))
	checkOmitEmpty(t, []int(nil))
	checkOmitEmpty(t, []int{})
	checkOmitEmpty(t, []int{1})
	checkOmitEmpty(t, m)
	checkOmitEmpty(t, map[string]int{})
	checkOmitEmpty(t, map[string]int{"a": 1})
	checkOmitEmpty(t, [0]int{})
	checkOmitEmpty(t, [2]int{})
	checkOmitEmpty(t, np)
	checkOmitEmpty(t, &one)
	checkOmitEmpty(t, plainStruct{})
	checkOmitEmpty(t, plainStruct{A: 1})
	checkOmitEmpty(t, time.Time{})
	checkOmitEmpty(t, any(nil))
	checkOmitEmpty(t, namedSlice(nil))
	checkOmitEmpty(t, namedSlice{1, 2, 3})
}

func TestOmitZeroNil(t *testing.T) {
	if !OmitZero(nil) {
		t.Error("OmitZero(nil) = false, want true")
	}
}

func TestOmitZeroParity(t *testing.T) {
	var np *zeroPtr
	var nv *zeroVal
	var ns []int
	checkOmitZero(t, 0)
	checkOmitZero(t, 5)
	checkOmitZero(t, "")
	checkOmitZero(t, "x")
	checkOmitZero(t, 0.0)
	checkOmitZero(t, false)
	checkOmitZero(t, []int(nil))
	checkOmitZero(t, []int{})
	checkOmitZero(t, ns)
	checkOmitZero(t, [2]int{})
	checkOmitZero(t, [2]int{0, 1})
	checkOmitZero(t, plainStruct{})
	checkOmitZero(t, plainStruct{B: "x"})
	checkOmitZero(t, time.Time{})
	checkOmitZero(t, time.Unix(1, 0))
	checkOmitZero(t, zeroVal{})
	checkOmitZero(t, zeroVal{7})
	checkOmitZero(t, zeroPtr{})
	checkOmitZero(t, zeroPtr{7})
	checkOmitZero(t, np)
	checkOmitZero(t, &zeroPtr{})
	checkOmitZero(t, &zeroPtr{7})
	checkOmitZero(t, nv)
	checkOmitZero(t, &zeroVal{7})
	checkOmitZero(t, &zeroVal{1})
	checkOmitZero(t, namedSlice(nil))
	checkOmitZero(t, namedSlice{1, 2, 3})
	checkOmitZero(t, namedSlice{})
	checkOmitZero(t, zeroer(nil))
	checkOmitZero(t, zeroer(zeroVal{7}))
	checkOmitZero(t, zeroer(zeroVal{1}))
	checkOmitZero(t, zeroer(np))
	checkOmitZero(t, zeroer(&zeroPtr{7}))
	checkOmitZero(t, zeroer(&zeroPtr{1}))
}

func TestOmitZeroAddressableTemp(t *testing.T) {
	if !OmitZero(zeroPtr{7}) {
		t.Error("pointer-receiver IsZero not used for a non-pointer value")
	}
	if OmitZero(zeroPtr{1}) {
		t.Error("pointer-receiver IsZero said zero for n=1")
	}
}

func TestMarshalField(t *testing.T) {
	vals := []any{1, "a<b", []int{1, 2}, map[string]int{"a": 1}, plainStruct{1, "x"}, nil, 1.5, time.Unix(0, 0).UTC()}
	for _, v := range vals {
		want, wantErr := stdjson.Marshal(v)
		got, gotErr := MarshalField(v)
		if string(got) != string(want) || (gotErr == nil) != (wantErr == nil) {
			t.Errorf("MarshalField(%#v) = %q, %v; want %q, %v", v, got, gotErr, want, wantErr)
		}
	}
	_, err := MarshalField(make(chan int))
	if err == nil {
		t.Fatal("MarshalField(chan) returned no error")
	}
	if _, ok := err.(*stdjson.UnsupportedTypeError); !ok {
		t.Errorf("MarshalField(chan) error = %T, want *UnsupportedTypeError", err)
	}
}
