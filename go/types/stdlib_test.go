// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests types.Check by using it to
// typecheck the standard library and tests.

package types

import (
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var verbose = flag.Bool("types.v", false, "verbose mode")

var (
	pkgCount int // number of packages processed
	start    = time.Now()
)

func TestStdlib(t *testing.T) {
	walkDirs(t, filepath.Join(runtime.GOROOT(), "src/pkg"))
	if *verbose {
		fmt.Println(pkgCount, "packages typechecked in", time.Since(start))
	}
}

func TestStdtest(t *testing.T) {
	path := filepath.Join(runtime.GOROOT(), "test")

	files, err := ioutil.ReadDir(path)
	if err != nil {
		t.Fatal(err)
	}

	fset := token.NewFileSet()
	for _, f := range files {
		// filter directory contents
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".go") {
			continue
		}

		// explicitly exclude files that the type-checker still has problems with
		switch f.Name() {
		case "cmplxdivide.go":
			// This test also needs file cmplxdivide1.go; ignore.
			continue
		case "goto.go", "label1.go":
			// TODO(gri) implement missing label checks
			continue
		case "sizeof.go", "switch.go":
			// TODO(gri) tone down duplicate checking in expression switches
			continue
		case "typeswitch2.go":
			// TODO(gri) implement duplicate checking in type switches
			continue
		}

		// parse file
		filename := filepath.Join(path, f.Name())
		// TODO(gri) The parser loses comments when bailing out early,
		//           and then we don't see the errorcheck command for
		//           some files. Use parser.AllErrors for now. Fix this.
		file, err := parser.ParseFile(fset, filename, nil, parser.ParseComments|parser.AllErrors)

		// check per-file instructions
		// For now we only check two cases.
		expectErrors := false
		if len(file.Comments) > 0 {
			if group := file.Comments[0]; len(group.List) > 0 {
				cmd := strings.TrimSpace(group.List[0].Text[2:]) // 2: ignore // or /* of comment
				switch cmd {
				case "skip":
					continue
				case "errorcheck":
					expectErrors = true
				}
			}
		}

		// type-check file if it parsed cleanly
		if err == nil {
			_, err = Check(filename, fset, []*ast.File{file})
		}

		if expectErrors {
			if err == nil {
				t.Errorf("expected errors but found none in %s", filename)
			}
		} else {
			if err != nil {
				t.Error(err)
			}
		}
	}
}

// Package paths of excluded packages.
var excluded = map[string]bool{
	"builtin": true,
}

// typecheck typechecks the given package files.
func typecheck(t *testing.T, path string, filenames []string) {
	fset := token.NewFileSet()

	// parse package files
	var files []*ast.File
	for _, filename := range filenames {
		file, err := parser.ParseFile(fset, filename, nil, parser.AllErrors)
		if err != nil {
			// the parser error may be a list of individual errors; report them all
			if list, ok := err.(scanner.ErrorList); ok {
				for _, err := range list {
					t.Error(err)
				}
				return
			}
			t.Error(err)
			return
		}

		if *verbose {
			if len(files) == 0 {
				fmt.Println("package", file.Name.Name)
			}
			fmt.Println("\t", filename)
		}

		files = append(files, file)
	}

	// typecheck package files
	var conf Config
	conf.Error = func(err error) { t.Error(err) }
	conf.Check(path, fset, files, nil)
	pkgCount++
}

// pkgfiles returns the list of package files for the given directory.
func pkgfiles(t *testing.T, dir string) []string {
	ctxt := build.Default
	ctxt.CgoEnabled = false
	pkg, err := ctxt.ImportDir(dir, 0)
	if err != nil {
		if _, nogo := err.(*build.NoGoError); !nogo {
			t.Error(err)
		}
		return nil
	}
	if excluded[pkg.ImportPath] {
		return nil
	}
	var filenames []string
	for _, name := range pkg.GoFiles {
		filenames = append(filenames, filepath.Join(pkg.Dir, name))
	}
	for _, name := range pkg.TestGoFiles {
		filenames = append(filenames, filepath.Join(pkg.Dir, name))
	}
	return filenames
}

// Note: Could use filepath.Walk instead of walkDirs but that wouldn't
//       necessarily be shorter or clearer after adding the code to
//       terminate early for -short tests.

func walkDirs(t *testing.T, dir string) {
	// limit run time for short tests
	if testing.Short() && time.Since(start) >= 750*time.Millisecond {
		return
	}

	fis, err := ioutil.ReadDir(dir)
	if err != nil {
		t.Error(err)
		return
	}

	// typecheck package in directory
	if files := pkgfiles(t, dir); files != nil {
		typecheck(t, dir, files)
	}

	// traverse subdirectories, but don't walk into testdata
	for _, fi := range fis {
		if fi.IsDir() && fi.Name() != "testdata" {
			walkDirs(t, filepath.Join(dir, fi.Name()))
		}
	}
}
