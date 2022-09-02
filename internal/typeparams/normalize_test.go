// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"regexp"
	"strings"
	"testing"

	"golang.org/x/tools/internal/typeparams"
	. "golang.org/x/tools/internal/typeparams"
)

func TestStructuralTerms(t *testing.T) {
	if !Enabled {
		t.Skip("typeparams are not enabled")
	}

	// In the following tests, src must define a type T with (at least) one type
	// parameter. We will compute the structural terms of the first type
	// parameter.
	tests := []struct {
		src       string
		want      string
		wantError string
	}{
		{"package emptyinterface0; type T[P interface{}] int", "all", ""},
		{"package emptyinterface1; type T[P interface{ int | interface{} }] int", "all", ""},
		{"package singleton; type T[P interface{ int }] int", "int", ""},
		{"package under; type T[P interface{~int}] int", "~int", ""},
		{"package superset; type T[P interface{ ~int | int }] int", "~int", ""},
		{"package overlap; type T[P interface{ ~int; int }] int", "int", ""},
		{"package emptyintersection; type T[P interface{ ~int; string }] int", "", "empty type set"},

		{"package embedded0; type T[P interface{ I }] int; type I interface { int }", "int", ""},
		{"package embedded1; type T[P interface{ I | string }] int; type I interface{ int | ~string }", "int ?\\| ?~string", ""},
		{"package embedded2; type T[P interface{ I; string }] int; type I interface{ int | ~string }", "string", ""},

		{"package named; type T[P C] int; type C interface{ ~int|int }", "~int", ""},
		{`// package example is taken from the docstring for StructuralTerms
package example

type A interface{ ~string|~[]byte }

type B interface{ int|string }

type C interface { ~string|~int }

type T[P interface{ A|B; C }] int
`, "~string ?\\| ?int", ""},
	}

	for _, test := range tests {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "p.go", test.src, 0)
		if err != nil {
			t.Fatal(err)
		}
		t.Run(f.Name.Name, func(t *testing.T) {
			conf := types.Config{
				Error: func(error) {}, // keep going on errors
			}
			pkg, err := conf.Check("", fset, []*ast.File{f}, nil)
			if err != nil {
				t.Logf("types.Config.Check: %v", err)
				// keep going on type checker errors: we want to assert on behavior of
				// invalid code as well.
			}
			obj := pkg.Scope().Lookup("T")
			if obj == nil {
				t.Fatal("type T not found")
			}
			T := typeparams.ForNamed(obj.Type().(*types.Named)).At(0)
			terms, err := StructuralTerms(T)
			if test.wantError != "" {
				if err == nil {
					t.Fatalf("StructuralTerms(%s): nil error, want %q", T, test.wantError)
				}
				if !strings.Contains(err.Error(), test.wantError) {
					t.Errorf("StructuralTerms(%s): err = %q, want %q", T, err, test.wantError)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var got string
			if len(terms) == 0 {
				got = "all"
			} else {
				qf := types.RelativeTo(pkg)
				got = types.TypeString(NewUnion(terms), qf)
			}
			want := regexp.MustCompile(test.want)
			if !want.MatchString(got) {
				t.Errorf("StructuralTerms(%s) = %q, want %q", T, got, test.want)
			}
		})
	}
}
