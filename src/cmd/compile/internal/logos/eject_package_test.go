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

func TestEjectPackageFilename(t *testing.T) {
	for _, test := range []struct {
		name string
		src  string
	}{
		{"go_file", "package p\n\nfunc answer() {}\n"},
		{"cgo_file", "package p\nimport \"C\"\n\nfunc answer() {}\n"},
		{"leading_comment", "/* Package p\nkeeps its comments. */\npackage p\nimport \"C\"\n\nfunc answer() {}\n"},
		{"split_package", "package\np\nimport \"C\"\n\nfunc answer() {}\n"},
		{"same_line", "package p; import \"C\"; func answer() int { return int(C.int(42)) }\n"},
		{"import_line", "package p\nimport \"C\"; func answer() int { return int(C.int(42)) }\n"},
		{"go_same_line", "package p; func answer() {}\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			src := []byte(test.src)
			file := ejectParse(t, syntax.NewFileBase("bridge.lgo"), src, 0)
			out := t.TempDir()
			if err := Eject(out, []*syntax.File{file}, map[*syntax.File][]byte{file: src}, nil); err != nil {
				t.Fatal(err)
			}
			name := filepath.Join(out, "bridge.go")
			got, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			line := strings.Count(test.src[:strings.Index(test.src, "func answer")], "\n") + 1
			for _, formatted := range []bool{false, true} {
				if formatted {
					got, err = format.Source(got)
					if err != nil {
						t.Fatalf("format ejected source: %v\n%s", err, got)
					}
				}
				fs := token.NewFileSet()
				parsed, err := parser.ParseFile(fs, name, got, 0)
				if err != nil {
					t.Fatal(err)
				}
				// Upstream cgocall reparses every file at this position,
				// including non-C files in the same cgo package.
				if pos := fs.Position(parsed.Pos()); pos.Filename != name {
					t.Errorf("formatted=%v: package file = %s, want %s", formatted, pos.Filename, name)
				}
				fn := parsed.Decls[len(parsed.Decls)-1].(*ast.FuncDecl)
				pos := fs.Position(fn.Pos())
				want := filepath.Join(out, "bridge.lgo")
				if pos.Filename != want || pos.Line != line {
					t.Errorf("formatted=%v: function position = %v, want %s:%d", formatted, pos, want, line)
				}
			}
		})
	}
}
