// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"testing"

	. "cmd/compile/internal/types2"
)

func TestInvalidTypeSet(t *testing.T) {
	if !InvalidTypeSetIsEmpty() {
		t.Error("invalidTypeSet is not empty")
	}
}

func TestTypeSetString(t *testing.T) {
	for body, want := range map[string]string{
		"{}":            "𝓤",
		"{int}":         "{int}",
		"{~int}":        "{~int}",
		"{int|string}":  "{int | string}",
		"{int; string}": "∅",

		"{comparable}":              "{comparable}",
		"{comparable; int}":         "{int}",
		"{~int; comparable}":        "{~int}",
		"{int|string; comparable}":  "{int | string}",
		"{comparable; int; string}": "∅",

		"{m()}":                         "{func (p.T).m()}",
		"{m1(); m2() int }":             "{func (p.T).m1(); func (p.T).m2() int}",
		"{error}":                       "{func (error).Error() string}",
		"{m(); comparable}":             "{comparable; func (p.T).m()}",
		"{m1(); comparable; m2() int }": "{comparable; func (p.T).m1(); func (p.T).m2() int}",
		"{comparable; error}":           "{comparable; func (error).Error() string}",

		"{m(); comparable; int|float32|string}": "{func (p.T).m(); int | float32 | string}",
		"{m1(); int; m2(); comparable }":        "{func (p.T).m1(); func (p.T).m2(); int}",

		"{E}; type E interface{}":           "𝓤",
		"{E}; type E interface{int;string}": "∅",
		"{E}; type E interface{comparable}": "{comparable}",
	} {
		// parse and type check
		src := "package p; type T interface" + body
		pkg := mustTypecheck(src, nil, nil)

		// lookup T
		obj := pkg.Scope().Lookup("T")
		if obj == nil {
			t.Fatalf("%s: T not found (invalid test case)", body)
		}
		T, ok := obj.Type().Underlying().(*Interface)
		if !ok {
			t.Fatalf("%s: %v is not an interface (invalid test case)", body, obj)
		}

		// verify test case
		got := TypeSetString(T)
		if got != want {
			t.Errorf("%s: got %s; want %s", body, got, want)
		}
	}
}

// TODO(gri) add more tests
