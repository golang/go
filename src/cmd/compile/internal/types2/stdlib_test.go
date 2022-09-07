// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests types2.Check by using it to
// typecheck the standard library and tests.

package types2_test

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"fmt"
	"go/build"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "cmd/compile/internal/types2"
)

var stdLibImporter = defaultImporter()

func TestStdlib(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	pkgCount := 0
	duration := walkPkgDirs(filepath.Join(testenv.GOROOT(t), "src"), func(dir string, filenames []string) {
		typecheck(t, dir, filenames)
		pkgCount++
	}, t.Error)

	if testing.Verbose() {
		fmt.Println(pkgCount, "packages typechecked in", duration)
	}
}

// firstComment returns the contents of the first non-empty comment in
// the given file, "skip", or the empty string. No matter the present
// comments, if any of them contains a build tag, the result is always
// "skip". Only comments within the first 4K of the file are considered.
// TODO(gri) should only read until we see "package" token.
func firstComment(filename string) (first string) {
	f, err := os.Open(filename)
	if err != nil {
		return ""
	}
	defer f.Close()

	// read at most 4KB
	var buf [4 << 10]byte
	n, _ := f.Read(buf[:])
	src := bytes.NewBuffer(buf[:n])

	// TODO(gri) we need a better way to terminate CommentsDo
	defer func() {
		if p := recover(); p != nil {
			if s, ok := p.(string); ok {
				first = s
			}
		}
	}()

	syntax.CommentsDo(src, func(_, _ uint, text string) {
		if text[0] != '/' {
			return // not a comment
		}

		// extract comment text
		if text[1] == '*' {
			text = text[:len(text)-2]
		}
		text = strings.TrimSpace(text[2:])

		if strings.HasPrefix(text, "+build ") {
			panic("skip")
		}
		if first == "" {
			first = text // text may be "" but that's ok
		}
		// continue as we may still see build tags
	})

	return
}

func testTestDir(t *testing.T, path string, ignore ...string) {
	files, err := os.ReadDir(path)
	if err != nil {
		t.Fatal(err)
	}

	excluded := make(map[string]bool)
	for _, filename := range ignore {
		excluded[filename] = true
	}

	for _, f := range files {
		// filter directory contents
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".go") || excluded[f.Name()] {
			continue
		}

		// get per-file instructions
		expectErrors := false
		filename := filepath.Join(path, f.Name())
		goVersion := ""
		if comment := firstComment(filename); comment != "" {
			fields := strings.Fields(comment)
			switch fields[0] {
			case "skip", "compiledir":
				continue // ignore this file
			case "errorcheck":
				expectErrors = true
				for _, arg := range fields[1:] {
					if arg == "-0" || arg == "-+" || arg == "-std" {
						// Marked explicitly as not expecting errors (-0),
						// or marked as compiling runtime/stdlib, which is only done
						// to trigger runtime/stdlib-only error output.
						// In both cases, the code should typecheck.
						expectErrors = false
						break
					}
					const prefix = "-lang="
					if strings.HasPrefix(arg, prefix) {
						goVersion = arg[len(prefix):]
					}
				}
			}
		}

		// parse and type-check file
		if testing.Verbose() {
			fmt.Println("\t", filename)
		}
		file, err := syntax.ParseFile(filename, nil, nil, 0)
		if err == nil {
			conf := Config{GoVersion: goVersion, Importer: stdLibImporter}
			_, err = conf.Check(filename, []*syntax.File{file}, nil)
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

func TestStdTest(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in short mode")
	}

	testTestDir(t, filepath.Join(testenv.GOROOT(t), "test"),
		"cmplxdivide.go", // also needs file cmplxdivide1.go - ignore
		"directive.go",   // tests compiler rejection of bad directive placement - ignore
		"directive2.go",  // tests compiler rejection of bad directive placement - ignore
		"embedfunc.go",   // tests //go:embed
		"embedvers.go",   // tests //go:embed
		"linkname2.go",   // types2 doesn't check validity of //go:xxx directives
		"linkname3.go",   // types2 doesn't check validity of //go:xxx directives
	)
}

func TestStdFixed(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in short mode")
	}

	testTestDir(t, filepath.Join(testenv.GOROOT(t), "test", "fixedbugs"),
		"bug248.go", "bug302.go", "bug369.go", // complex test instructions - ignore
		"issue6889.go",   // gc-specific test
		"issue11362.go",  // canonical import path check
		"issue16369.go",  // types2 handles this correctly - not an issue
		"issue18459.go",  // types2 doesn't check validity of //go:xxx directives
		"issue18882.go",  // types2 doesn't check validity of //go:xxx directives
		"issue20529.go",  // types2 does not have constraints on stack size
		"issue22200.go",  // types2 does not have constraints on stack size
		"issue22200b.go", // types2 does not have constraints on stack size
		"issue25507.go",  // types2 does not have constraints on stack size
		"issue20780.go",  // types2 does not have constraints on stack size
		"issue42058a.go", // types2 does not have constraints on channel element size
		"issue42058b.go", // types2 does not have constraints on channel element size
		"issue48097.go",  // go/types doesn't check validity of //go:xxx directives, and non-init bodyless function
		"issue48230.go",  // go/types doesn't check validity of //go:xxx directives
		"issue49767.go",  // go/types does not have constraints on channel element size
		"issue49814.go",  // go/types does not have constraints on array size

		// These tests requires runtime/cgo.Incomplete, which is only available on some platforms.
		// However, types2 does not know about build constraints.
		"bug514.go",
		"issue40954.go",
		"issue42032.go",
		"issue42076.go",
		"issue46903.go",
		"issue51733.go",
		"notinheap2.go",
		"notinheap3.go",
	)
}

func TestStdKen(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	testTestDir(t, filepath.Join(testenv.GOROOT(t), "test", "ken"))
}

// Package paths of excluded packages.
var excluded = map[string]bool{
	"builtin": true,

	// See #46027: some imports are missing for this submodule.
	"crypto/internal/edwards25519/field/_asm": true,
}

// typecheck typechecks the given package files.
func typecheck(t *testing.T, path string, filenames []string) {
	// parse package files
	var files []*syntax.File
	for _, filename := range filenames {
		errh := func(err error) { t.Error(err) }
		file, err := syntax.ParseFile(filename, errh, nil, 0)
		if err != nil {
			return
		}

		if testing.Verbose() {
			if len(files) == 0 {
				fmt.Println("package", file.PkgName.Value)
			}
			fmt.Println("\t", filename)
		}

		files = append(files, file)
	}

	// typecheck package files
	conf := Config{
		Error:    func(err error) { t.Error(err) },
		Importer: stdLibImporter,
	}
	info := Info{Uses: make(map[*syntax.Name]Object)}
	conf.Check(path, files, &info)

	// Perform checks of API invariants.

	// All Objects have a package, except predeclared ones.
	errorError := Universe.Lookup("error").Type().Underlying().(*Interface).ExplicitMethod(0) // (error).Error
	for id, obj := range info.Uses {
		predeclared := obj == Universe.Lookup(obj.Name()) || obj == errorError
		if predeclared == (obj.Pkg() != nil) {
			posn := id.Pos()
			if predeclared {
				t.Errorf("%s: predeclared object with package: %s", posn, obj)
			} else {
				t.Errorf("%s: user-defined object without package: %s", posn, obj)
			}
		}
	}
}

// pkgFilenames returns the list of package filenames for the given directory.
func pkgFilenames(dir string) ([]string, error) {
	ctxt := build.Default
	ctxt.CgoEnabled = false
	pkg, err := ctxt.ImportDir(dir, 0)
	if err != nil {
		if _, nogo := err.(*build.NoGoError); nogo {
			return nil, nil // no *.go files, not an error
		}
		return nil, err
	}
	if excluded[pkg.ImportPath] {
		return nil, nil
	}
	var filenames []string
	for _, name := range pkg.GoFiles {
		filenames = append(filenames, filepath.Join(pkg.Dir, name))
	}
	for _, name := range pkg.TestGoFiles {
		filenames = append(filenames, filepath.Join(pkg.Dir, name))
	}
	return filenames, nil
}

func walkPkgDirs(dir string, pkgh func(dir string, filenames []string), errh func(args ...interface{})) time.Duration {
	w := walker{time.Now(), 10 * time.Millisecond, pkgh, errh}
	w.walk(dir)
	return time.Since(w.start)
}

type walker struct {
	start time.Time
	dmax  time.Duration
	pkgh  func(dir string, filenames []string)
	errh  func(args ...interface{})
}

func (w *walker) walk(dir string) {
	// limit run time for short tests
	if testing.Short() && time.Since(w.start) >= w.dmax {
		return
	}

	files, err := os.ReadDir(dir)
	if err != nil {
		w.errh(err)
		return
	}

	// apply pkgh to the files in directory dir
	pkgFiles, err := pkgFilenames(dir)
	if err != nil {
		w.errh(err)
		return
	}
	if pkgFiles != nil {
		w.pkgh(dir, pkgFiles)
	}

	// traverse subdirectories, but don't walk into testdata
	for _, f := range files {
		if f.IsDir() && f.Name() != "testdata" {
			w.walk(filepath.Join(dir, f.Name()))
		}
	}
}
