// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
	"time"
)

func TestSelf(t *testing.T) {
	filenames := pkgfiles(t, ".") // from stdlib_test.go

	// parse package files
	fset := token.NewFileSet()
	var files []*ast.File
	for _, filename := range filenames {
		file, err := parser.ParseFile(fset, filename, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, file)
	}

	_, err := Check("go/types", fset, files)
	if err != nil {
		// Importing go.tools/go/exact doensn't work in the
		// build dashboard environment at the moment. Don't
		// report an error for now so that the build remains
		// green.
		// TODO(gri) fix this
		t.Log(err) // replace w/ t.Fatal eventually
		return
	}

	if testing.Short() {
		return // skip benchmark in short mode
	}

	benchmark(fset, files, false)
	benchmark(fset, files, true)
}

func benchmark(fset *token.FileSet, files []*ast.File, full bool) {
	b := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			conf := Config{IgnoreFuncBodies: !full}
			conf.Check("go/types", fset, files, nil)
		}
	})

	// determine line count
	lineCount := 0
	fset.Iterate(func(f *token.File) bool {
		lineCount += f.LineCount()
		return true
	})

	d := time.Duration(b.NsPerOp())
	fmt.Printf(
		"%s/op, %d lines/s, %d KB/op (%d iterations)\n",
		d,
		int64(float64(lineCount)/d.Seconds()),
		b.AllocedBytesPerOp()>>10,
		b.N,
	)
}
