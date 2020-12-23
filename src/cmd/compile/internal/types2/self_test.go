// UNREVIEWED
// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"flag"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	. "cmd/compile/internal/types2"
)

var benchmark = flag.Bool("b", false, "run benchmarks")

func TestSelf(t *testing.T) {
	files, err := pkgFiles(".")
	if err != nil {
		t.Fatal(err)
	}

	conf := Config{Importer: defaultImporter()}
	_, err = conf.Check("go/types", files, nil)
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
	files, err := pkgFiles(path)
	if err != nil {
		t.Fatal(err)
	}

	b := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			conf := Config{IgnoreFuncBodies: ignoreFuncBodies}
			conf.Check(path, files, nil)
		}
	})

	// determine line count
	var lines uint
	for _, f := range files {
		lines += f.EOF.Line()
	}

	d := time.Duration(b.NsPerOp())
	fmt.Printf(
		"%s: %s for %d lines (%d lines/s), ignoreFuncBodies = %v\n",
		filepath.Base(path), d, lines, int64(float64(lines)/d.Seconds()), ignoreFuncBodies,
	)
}

func pkgFiles(path string) ([]*syntax.File, error) {
	filenames, err := pkgFilenames(path) // from stdlib_test.go
	if err != nil {
		return nil, err
	}

	var files []*syntax.File
	for _, filename := range filenames {
		file, err := syntax.ParseFile(filename, nil, nil, 0)
		if err != nil {
			return nil, err
		}
		files = append(files, file)
	}

	return files, nil
}
