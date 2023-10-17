// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"strings"
	"testing"
)

func TestInvalidTypeSet(t *testing.T) {
	if !invalidTypeSet.IsEmpty() {
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
		// parse
		errh := func(error) {} // dummy error handler so that parsing continues in presence of errors
		src := "package p; type T interface" + body
		file, err := syntax.Parse(nil, strings.NewReader(src), errh, nil, 0)
		if err != nil {
			t.Fatalf("%s: %v (invalid test case)", body, err)
		}

		// type check
		var conf Config
		pkg, err := conf.Check(file.PkgName.Value, []*syntax.File{file}, nil)
		if err != nil {
			t.Fatalf("%s: %v (invalid test case)", body, err)
		}

		// lookup T
		obj := pkg.scope.Lookup("T")
		if obj == nil {
			t.Fatalf("%s: T not found (invalid test case)", body)
		}
		T, ok := under(obj.Type()).(*Interface)
		if !ok {
			t.Fatalf("%s: %v is not an interface (invalid test case)", body, obj)
		}

		// verify test case
		got := T.typeSet().String()
		if got != want {
			t.Errorf("%s: got %s; want %s", body, got, want)
		}
	}
}

// TODO(gri) add more tests
