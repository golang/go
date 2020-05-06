// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"flag"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"path/filepath"
	"testing"
	"time"

	. "go/types"
)

var benchmark = flag.Bool("b", false, "run benchmarks")

func TestSelf(t *testing.T) {
	fset := token.NewFileSet()
	files, err := pkgFiles(fset, ".")
	if err != nil {
		t.Fatal(err)
	}

	conf := Config{Importer: importer.Default()}
	_, err = conf.Check("go/types", fset, files, nil)
	if err != nil {
		// Importing go/constant doesn't work in the
		// build dashboard environment. Don't report an error
		// for now so that the build remains green.
		// TODO(gri) fix this
		t.Log(err) // replace w/ t.Fatal eventually
		return
	}
}

func TestBenchmark(t *testing.T) {
	if !*benchmark {
		return
	}

	// We're not using testing's benchmarking mechanism directly
	// because we want custom output.

	for _, p := range []string{"types", "constant", filepath.Join("internal", "gcimporter")} {
		path := filepath.Join("..", p)
		runbench(t, path, false)
		runbench(t, path, true)
		fmt.Println()
	}
}

func runbench(t *testing.T, path string, ignoreFuncBodies bool) {
	fset := token.NewFileSet()
	files, err := pkgFiles(fset, path)
	if err != nil {
		t.Fatal(err)
	}

	b := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			conf := Config{IgnoreFuncBodies: ignoreFuncBodies}
			conf.Check(path, fset, files, nil)
		}
	})

	// determine line count
	lines := 0
	fset.Iterate(func(f *token.File) bool {
		lines += f.LineCount()
		return true
	})

	d := time.Duration(b.NsPerOp())
	fmt.Printf(
		"%s: %s for %d lines (%d lines/s), ignoreFuncBodies = %v\n",
		filepath.Base(path), d, lines, int64(float64(lines)/d.Seconds()), ignoreFuncBodies,
	)
}

func pkgFiles(fset *token.FileSet, path string) ([]*ast.File, error) {
	filenames, err := pkgFilenames(path) // from stdlib_test.go
	if err != nil {
		return nil, err
	}

	var files []*ast.File
	for _, filename := range filenames {
		file, err := parser.ParseFile(fset, filename, nil, 0)
		if err != nil {
			return nil, err
		}
		files = append(files, file)
	}

	return files, nil
}
