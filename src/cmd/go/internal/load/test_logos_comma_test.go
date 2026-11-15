// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"go/ast"
	"go/token"
	"testing"
)

func scanCommaFunc(t *testing.T, src string) *ast.FuncDecl {
	t.Helper()
	fset := token.NewFileSet()
	file := scanLogosTestFile(fset, "comma_test.lgo", []byte(src))
	if file == nil {
		t.Fatalf("scanLogosTestFile returned nil for %q", src)
	}
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			return fn
		}
	}
	t.Fatalf("no function declaration found in %q", src)
	return nil
}

func TestLogosScanTrailingComma(t *testing.T) {
	tests := []struct {
		name     string
		src      string
		wantArg  string
		wantName string
	}{
		{
			name:     "test single trailing comma",
			src:      "package p\nfunc TestFoo(t *testing.T,) {}\n",
			wantArg:  "T",
			wantName: "t",
		},
		{
			name:     "test multiline trailing comma",
			src:      "package p\nfunc TestFoo(t *testing.T,\n) {}\n",
			wantArg:  "T",
			wantName: "t",
		},
		{
			name:     "benchmark trailing comma",
			src:      "package p\nfunc BenchmarkFoo(b *testing.B,) {}\n",
			wantArg:  "B",
			wantName: "b",
		},
		{
			name:     "fuzz trailing comma",
			src:      "package p\nfunc FuzzFoo(f *testing.F,) {}\n",
			wantArg:  "F",
			wantName: "f",
		},
		{
			name:     "benchmark multiline trailing comma",
			src:      "package p\nfunc BenchmarkFoo(b *testing.B,\n) {}\n",
			wantArg:  "B",
			wantName: "b",
		},
		{
			name:     "fuzz multiline trailing comma",
			src:      "package p\nfunc FuzzFoo(f *testing.F,\n) {}\n",
			wantArg:  "F",
			wantName: "f",
		},
		{
			name:    "double comma stays BadExpr",
			src:     "package p\nfunc TestFoo(t *testing.T,,) {}\n",
			wantArg: "",
		},
		{
			name:     "no comma ordinary case",
			src:      "package p\nfunc TestFoo(t *testing.T) {}\n",
			wantArg:  "T",
			wantName: "t",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fn := scanCommaFunc(t, tt.src)
			if fn.Type.Params == nil || len(fn.Type.Params.List) != 1 {
				t.Fatalf("%s: got %d param fields, want 1", tt.name, len(fn.Type.Params.List))
			}
			fld := fn.Type.Params.List[0]
			if tt.wantArg == "" {
				if _, ok := fld.Type.(*ast.BadExpr); !ok {
					t.Fatalf("%s: got param type %T, want *ast.BadExpr", tt.name, fld.Type)
				}
				if isTestFunc(fn, "T") {
					t.Fatalf("%s: double-comma signature reports as test func", tt.name)
				}
				return
			}
			star, ok := fld.Type.(*ast.StarExpr)
			if !ok {
				t.Fatalf("%s: got param type %T, want *ast.StarExpr", tt.name, fld.Type)
			}
			sel, ok := star.X.(*ast.SelectorExpr)
			if !ok {
				t.Fatalf("%s: got star operand %T, want *ast.SelectorExpr", tt.name, star.X)
			}
			x, ok := sel.X.(*ast.Ident)
			if !ok || x.Name != "testing" || sel.Sel.Name != tt.wantArg {
				t.Fatalf("%s: got *%v.%v, want *testing.%s", tt.name, sel.X, sel.Sel.Name, tt.wantArg)
			}
			if len(fld.Names) != 1 || fld.Names[0].Name != tt.wantName {
				t.Fatalf("%s: got param names %v, want [%s]", tt.name, fld.Names, tt.wantName)
			}
			if !isTestFunc(fn, tt.wantArg) {
				t.Fatalf("%s: isTestFunc(%q) = false, want true", tt.name, tt.wantArg)
			}
		})
	}
}
