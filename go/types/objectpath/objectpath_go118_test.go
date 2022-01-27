// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package objectpath_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/types/objectpath"
)

func TestGenericPaths(t *testing.T) {
	pkgs := map[string]map[string]string{
		"b": {"b.go": `
package b

const C int = 1

type T[TP0 any, TP1 interface{ M0(); M1() }] struct{}

func (T[RP0, RP1]) M() {}

type N int

func (N) M0()
func (N) M1()

type A = T[int, N]

func F[FP0, FP1 any](FP0, FP1) {}
`},
	}
	paths := []pathTest{
		// Good paths
		{"b", "T", "type b.T[TP0 any, TP1 interface{M0(); M1()}] struct{}", ""},
		{"b", "T.O", "type b.T[TP0 any, TP1 interface{M0(); M1()}] struct{}", ""},
		{"b", "T.M0", "func (b.T[RP0, RP1]).M()", ""},
		{"b", "T.T0O", "type parameter TP0 any", ""},
		{"b", "T.T1O", "type parameter TP1 interface{M0(); M1()}", ""},
		{"b", "T.T1CM0", "func (interface).M0()", ""},
		// Obj of an instance is the generic declaration.
		{"b", "A.O", "type b.T[TP0 any, TP1 interface{M0(); M1()}] struct{}", ""},
		{"b", "A.M0", "func (b.T[int, b.N]).M()", ""},

		// Bad paths
		{"b", "N.C", "", "invalid path: ends with 'C', want [AFMO]"},
		{"b", "N.CO", "", "cannot apply 'C' to b.N (got *types.Named, want type parameter)"},
		{"b", "N.T", "", `invalid path: bad numeric operand "" for code 'T'`},
		{"b", "N.T0", "", "tuple index 0 out of range [0-0)"},
		{"b", "T.T2O", "", "tuple index 2 out of range [0-2)"},
		{"b", "T.T1M0", "", "cannot apply 'M' to TP1 (got *types.TypeParam, want interface or named)"},
		{"b", "C.T0", "", "cannot apply 'T' to int (got *types.Basic, want named or signature)"},
	}

	conf := loader.Config{Build: buildutil.FakeContext(pkgs)}
	conf.Import("b")
	prog, err := conf.Load()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range paths {
		if err := testPath(prog, test); err != nil {
			t.Error(err)
		}
	}

	// bad objects
	for _, test := range []struct {
		obj     types.Object
		wantErr string
	}{
		{types.Universe.Lookup("any"), "predeclared type any = interface{} has no path"},
		{types.Universe.Lookup("comparable"), "predeclared type comparable interface{comparable} has no path"},
	} {
		path, err := objectpath.For(test.obj)
		if err == nil {
			t.Errorf("Object(%s) = %q, want error", test.obj, path)
			continue
		}
		if err.Error() != test.wantErr {
			t.Errorf("Object(%s) error was %q, want %q", test.obj, err, test.wantErr)
			continue
		}
	}
}
