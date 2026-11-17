// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

func TestEjectCgoPreamble(t *testing.T) {
	for _, test := range []struct {
		name    string
		imports string
	}{
		{"block", "/*\nstatic int answer(void) { return 42; }\n*/\nimport \"C\"\n"},
		{"line", "// static int answer(void) { return 42; }\n// #cgo nocallback answer\nimport \"C\"\n"},
		{"group", "import (\n/*\nstatic int answer(void) { return 42; }\n*/\n\"C\"\n\"fmt\"\n)\n"},
		{"group_decl", "/* static int answer(void) { return 42; } */\nimport (\n\"C\"\n)\n"},
		{"empty", "import \"C\"\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			src := []byte("package p\n\n" + test.imports + "\nfunc answer() int { return int(C.answer()) }\n")
			base := syntax.NewFileBase("bridge.lgo")
			file := ejectParse(t, base, src, 0)
			out := t.TempDir()
			if err := Eject(out, []*syntax.File{file}, map[*syntax.File][]byte{file: src}, nil); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(filepath.Join(out, "bridge.go"))
			if err != nil {
				t.Fatal(err)
			}
			want := cgoPreamble(t, src)
			for _, formatted := range []bool{false, true} {
				if formatted {
					original, formatErr := format.Source(src)
					if formatErr != nil {
						t.Fatal(formatErr)
					}
					want = cgoPreamble(t, original)
					got, err = format.Source(got)
					if err != nil {
						t.Fatalf("format ejected source: %v\n%s", err, got)
					}
				}
				if preamble := cgoPreamble(t, got); preamble != want {
					t.Errorf("formatted=%v: C preamble changed:\ngot %q\nwant %q\n%s", formatted, preamble, want, got)
				}
				fs := token.NewFileSet()
				parsed, err := parser.ParseFile(fs, "bridge.go", got, 0)
				if err != nil {
					t.Fatal(err)
				}
				fn := parsed.Decls[len(parsed.Decls)-1].(*ast.FuncDecl)
				pos := fs.Position(fn.Pos())
				line := strings.Count(string(src[:strings.Index(string(src), "func answer")]), "\n") + 1
				if pos.Filename != "bridge.lgo" || pos.Line != line {
					t.Errorf("formatted=%v: function position = %v, want bridge.lgo:%d", formatted, pos, line)
				}
			}
		})
	}
}

func cgoPreamble(t *testing.T, src []byte) string {
	file, err := parser.ParseFile(token.NewFileSet(), "bridge.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT {
			continue
		}
		for _, spec := range gen.Specs {
			imp := spec.(*ast.ImportSpec)
			if imp.Path.Value != `"C"` {
				continue
			}
			doc := imp.Doc
			if doc == nil && len(gen.Specs) == 1 {
				doc = gen.Doc
			}
			var text []string
			if doc != nil {
				for _, comment := range doc.List {
					text = append(text, comment.Text)
				}
			}
			return strings.Join(text, "\n")
		}
	}
	t.Fatal("missing C import")
	return ""
}
