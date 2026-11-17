// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"strings"
	"testing"
)

func TestLogosCgoConstTypes(t *testing.T) {
	for _, inSyntax := range []bool{false, true} {
		mode := "map"
		if inSyntax {
			mode = "syntax"
		}
		for _, tt := range []struct {
			name string
			init string
			kind ctfe.Kind
			elem string
			err  string
		}{
			{"slice", "[]C.int{1, 2}", ctfe.KSlice, "_Ctype_int", ""},
			{"array", "[2]C.word{1, 2}", ctfe.KArray, "_Ctype_word", ""},
			{"local", "[]Local{1, 2}", ctfe.KSlice, "Local", ""},
			{"overflow", "[]C.uchar{256}", ctfe.KSlice, "", "overflows"},
		} {
			t.Run(mode+"/"+tt.name, func(t *testing.T) {
				var files []*syntax.File
				for _, source := range []struct{ name, text string }{
					{"_cgo_gotypes.go", "package p\ntype _Ctype_int int32\ntype _Ctype_word uint64\ntype _Ctype_uchar uint8\n"},
					{"source.lgo", "package p\nimport \"C\"\ntype Local C.int\nconst Values = " + tt.init + "\n"},
				} {
					file, err := syntax.Parse(syntax.NewFileBase(source.name), strings.NewReader(source.text), nil, nil, syntax.LogosMode)
					if err != nil {
						t.Fatal(err)
					}
					files = append(files, file)
				}
				info := &Info{
					StoreTypesInSyntax: inSyntax,
					Defs:               make(map[*syntax.Name]Object),
					Uses:               make(map[*syntax.Name]Object),
				}
				if !inSyntax {
					info.Types = make(map[syntax.Expr]TypeAndValue)
				}
				conf := Config{Logos: true, LogosUnions: true, go115UsesCgo: true}
				pkg, err := conf.Check("example.com/cgoowned", files, info)
				if tt.err != "" {
					if err == nil || !strings.Contains(err.Error(), tt.err) {
						t.Fatalf("check error = %v, want %q", err, tt.err)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				obj := pkg.Scope().Lookup("Values").(*Const)
				v, ok := info.LogosCTFE[obj]
				if !ok || !v.IsComposite() || v.Type().Kind != tt.kind || v.Type().Elem.Named != tt.elem {
					t.Fatalf("constant type = %#v, want %v of %s", v.Type(), tt.kind, tt.elem)
				}
				elems := v.Elems()
				if tt.kind == ctfe.KSlice {
					backing, lo, length, _, ok := v.SliceHeader()
					if !ok {
						t.Fatal("constant slice has no backing")
					}
					elems = backing[lo : lo+length]
				}
				if len(elems) != 2 || elems[0].Int64() != 1 || elems[1].Int64() != 2 {
					t.Fatalf("constant elements = %v, want [1 2]", elems)
				}
			})
		}
	}
}
