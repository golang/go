// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"testing"
	"time"
)

type plainStruct struct {
	A int
	B string
}

type ifaceStruct struct {
	A any
	B [2]string
}

type zeroPtr struct{ n int }

func (z *zeroPtr) IsZero() bool { return z.n == 7 }

type omitZeroField[T any] struct {
	F T `json:"f,omitzero"`
}

func omittedByStdZero[T any](t *testing.T, v T) bool {
	t.Helper()
	b, err := stdjson.Marshal(omitZeroField[T]{v})
	if err != nil {
		t.Fatal(err)
	}
	return string(b) == "{}"
}

func checkIsZero[T comparable](t *testing.T, v T) {
	t.Helper()
	if got, want := IsZero(v), omittedByStdZero(t, v); got != want {
		t.Errorf("IsZero(%#v) = %v, encoding/json omitzero = %v", v, got, want)
	}
}

func TestIsZeroParity(t *testing.T) {
	negZero := 0.0
	negZero = -negZero
	checkIsZero(t, [2]int{})
	checkIsZero(t, [2]int{0, 1})
	checkIsZero(t, [0]int{})
	checkIsZero(t, [1]float64{negZero})
	checkIsZero(t, plainStruct{})
	checkIsZero(t, plainStruct{B: "x"})
	checkIsZero(t, ifaceStruct{})
	checkIsZero(t, ifaceStruct{A: []int{1}})
	checkIsZero(t, ifaceStruct{B: [2]string{"", "x"}})
	checkIsZero(t, struct{ P *int }{})
}

func TestIsNilPointer(t *testing.T) {
	var np *zeroPtr
	var z interface{ IsZero() bool } = np
	if !IsNilPointer(z) {
		t.Error("IsNilPointer(typed nil pointer) = false")
	}
	if IsNilPointer(&zeroPtr{}) {
		t.Error("IsNilPointer(non-nil pointer) = true")
	}
	var m map[string]int
	if IsNilPointer(m) || IsNilPointer(nil) || IsNilPointer(0) {
		t.Error("IsNilPointer reported a nil pointer for a non-pointer value")
	}
	if !omittedByStdZero(t, z) {
		t.Error("encoding/json keeps an omitzero interface holding a nil pointer")
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
