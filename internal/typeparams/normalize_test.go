// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"

	. "golang.org/x/tools/internal/typeparams"
)

func TestNormalize(t *testing.T) {
	if !Enabled {
		t.Skip("typeparams are not enabled")
	}
	tests := []struct {
		src       string
		want      string
		wantError string
	}{
		{"package emptyinterface0; type T interface{}", "interface{}", ""},
		{"package emptyinterface1; type T interface{ int | interface{} }", "interface{}", ""},
		{"package singleton; type T interface{ int }", "interface{int}", ""},
		{"package under; type T interface{~int}", "interface{~int}", ""},
		{"package superset; type T interface{ ~int | int }", "interface{~int}", ""},
		{"package overlap; type T interface{ ~int; int }", "interface{int}", ""},
		{"package emptyintersection; type T interface{ ~int; string }", "nil", ""},

		// The type-checker now produces Typ[Invalid] here (golang/go#48819).
		// TODO(rfindley): can we still test the cycle logic?
		// {"package cycle0; type T interface{ T }", "", "cycle"},
		// {"package cycle1; type T interface{ int | T }", "", "cycle"},
		// {"package cycle2; type T interface{ I }; type I interface { T }", "", "cycle"},

		{"package embedded0; type T interface{ I }; type I interface { int }", "interface{int}", ""},
		{"package embedded1; type T interface{ I | string }; type I interface{ int | ~string }", "interface{int|~string}", ""},
		{"package embedded2; type T interface{ I; string }; type I interface{ int | ~string }", "interface{string}", ""},
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
			T := obj.Type().Underlying().(*types.Interface)
			normal, err := NormalizeInterface(T)
			if test.wantError != "" {
				if err == nil {
					t.Fatalf("Normalize(%s): nil error, want %q", T, test.wantError)
				}
				if !strings.Contains(err.Error(), test.wantError) {
					t.Errorf("Normalize(%s): err = %q, want %q", T, err, test.wantError)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			qf := types.RelativeTo(pkg)
			var got string
			if normal == nil {
				got = "nil"
			} else {
				got = types.TypeString(normal, qf)
			}
			if got != test.want {
				t.Errorf("Normalize(%s) = %q, want %q", T, got, test.want)
			}
		})
	}
}
