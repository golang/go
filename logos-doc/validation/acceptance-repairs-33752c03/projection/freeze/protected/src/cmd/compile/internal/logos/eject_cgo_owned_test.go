// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestEjectCgoOwnedTypes(t *testing.T) {
	for _, tt := range []struct {
		name    string
		imports string
		elem    string
	}{
		{"original", "import \"C\"\n", "C.int"},
		{"ordinary", "", "_Ctype_int"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			src := []byte("package p\n" + tt.imports + "const values = []" + tt.elem + "{1, 2}\n")
			generated := []byte("package p\ntype _Ctype_int int32\n")
			base := syntax.NewFileBase("source.lgo")
			file := ejectParse(t, base, src, syntax.LogosMode)
			cgo := ejectParse(t, syntax.NewFileBase("_cgo_gotypes.go"), generated, 0)
			files := []*syntax.File{cgo, file}
			info := &types2.Info{
				Types:      make(map[syntax.Expr]types2.TypeAndValue),
				Instances:  make(map[*syntax.Name]types2.Instance),
				Defs:       make(map[*syntax.Name]types2.Object),
				Uses:       make(map[*syntax.Name]types2.Object),
				Implicits:  make(map[syntax.Node]types2.Object),
				Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
				Scopes:     make(map[syntax.Node]*types2.Scope),
			}
			conf := types2.Config{Logos: true, LogosUnions: true}
			types2.LogosEnableCgo(&conf)
			pkg, err := conf.Check("example.com/cgoowned", files, info)
			if err != nil {
				t.Fatal(err)
			}
			_, edits := Lower(pkg, info, files, map[*syntax.File][]byte{file: src, cgo: generated})
			out, err := ApplyEdits(src, base, edits[file])
			if err != nil {
				t.Fatal(err)
			}
			fset := token.NewFileSet()
			parsed, err := parser.ParseFile(fset, "source.go", out, 0)
			if err != nil {
				t.Fatalf("parse ejected source: %v\n%s", err, out)
			}
			var got string
			ast.Inspect(parsed, func(n ast.Node) bool {
				if decl, ok := n.(*ast.GenDecl); ok && decl.Tok != token.VAR {
					return false
				}
				if lit, ok := n.(*ast.CompositeLit); ok {
					var typ bytes.Buffer
					if err := format.Node(&typ, fset, lit.Type); err != nil {
						t.Fatal(err)
					}
					got = typ.String()
				}
				return true
			})
			if want := "[]" + tt.elem; got != want {
				t.Fatalf("ejected constant type = %q, want %q\n%s", got, want, out)
			}
		})
	}
}
