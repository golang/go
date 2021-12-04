// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"path"
	"path/filepath"
	"testing"
	"time"

	. "go/types"
)

func TestSelf(t *testing.T) {
	fset := token.NewFileSet()
	files, err := pkgFiles(fset, ".", 0)
	if err != nil {
		t.Fatal(err)
	}

	conf := Config{Importer: importer.Default()}
	_, err = conf.Check("go/types", fset, files, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func BenchmarkCheck(b *testing.B) {
	for _, p := range []string{
		"net/http",
		"go/parser",
		"go/constant",
		"runtime",
		filepath.Join("go", "internal", "gcimporter"),
	} {
		b.Run(path.Base(p), func(b *testing.B) {
			path := filepath.Join("..", "..", p)
			for _, ignoreFuncBodies := range []bool{false, true} {
				name := "funcbodies"
				if ignoreFuncBodies {
					name = "nofuncbodies"
				}
				b.Run(name, func(b *testing.B) {
					b.Run("info", func(b *testing.B) {
						runbench(b, path, ignoreFuncBodies, true)
					})
					b.Run("noinfo", func(b *testing.B) {
						runbench(b, path, ignoreFuncBodies, false)
					})
				})
			}
		})
	}
}

func runbench(b *testing.B, path string, ignoreFuncBodies, writeInfo bool) {
	fset := token.NewFileSet()
	files, err := pkgFiles(fset, path, 0)
	if err != nil {
		b.Fatal(err)
	}
	// determine line count
	lines := 0
	fset.Iterate(func(f *token.File) bool {
		lines += f.LineCount()
		return true
	})

	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		conf := Config{
			IgnoreFuncBodies: ignoreFuncBodies,
			Importer:         importer.Default(),
		}
		var info *Info
		if writeInfo {
			info = &Info{
				Types:      make(map[ast.Expr]TypeAndValue),
				Defs:       make(map[*ast.Ident]Object),
				Uses:       make(map[*ast.Ident]Object),
				Implicits:  make(map[ast.Node]Object),
				Selections: make(map[*ast.SelectorExpr]*Selection),
				Scopes:     make(map[ast.Node]*Scope),
			}
		}
		if _, err := conf.Check(path, fset, files, info); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(lines)*float64(b.N)/time.Since(start).Seconds(), "lines/s")
}

func pkgFiles(fset *token.FileSet, path string, mode parser.Mode) ([]*ast.File, error) {
	filenames, err := pkgFilenames(path) // from stdlib_test.go
	if err != nil {
		return nil, err
	}

	var files []*ast.File
	for _, filename := range filenames {
		file, err := parser.ParseFile(fset, filename, nil, mode)
		if err != nil {
			return nil, err
		}
		files = append(files, file)
	}

	return files, nil
}
