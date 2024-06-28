// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests types.Check by using it to
// typecheck the standard library and tests.

package types_test

import (
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/scanner"
	"go/token"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	. "go/types"
)

// The cmd/*/internal packages may have been deleted as part of a binary
// release. Import from source instead.
//
// (See https://golang.org/issue/43232 and
// https://github.com/golang/build/blob/df58bbac082bc87c4a3cdfe336d1ffe60bbaa916/cmd/release/release.go#L533-L545.)
//
// Use the same importer for all std lib tests to
// avoid repeated importing of the same packages.
var stdLibImporter = importer.ForCompiler(token.NewFileSet(), "source", nil)

func TestStdlib(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	testenv.MustHaveGoBuild(t)

	// Collect non-test files.
	dirFiles := make(map[string][]string)
	root := filepath.Join(testenv.GOROOT(t), "src")
	walkPkgDirs(root, func(dir string, filenames []string) {
		dirFiles[dir] = filenames
	}, t.Error)

	c := &stdlibChecker{
		dirFiles: dirFiles,
		pkgs:     make(map[string]*futurePackage),
	}

	start := time.Now()

	// Though we read files while parsing, type-checking is otherwise CPU bound.
	//
	// This doesn't achieve great CPU utilization as many packages may block
	// waiting for a common import, but in combination with the non-deterministic
	// map iteration below this should provide decent coverage of concurrent
	// type-checking (see golang/go#47729).
	cpulimit := make(chan struct{}, runtime.GOMAXPROCS(0))
	var wg sync.WaitGroup

	for dir := range dirFiles {
		dir := dir

		cpulimit <- struct{}{}
		wg.Add(1)
		go func() {
			defer func() {
				wg.Done()
				<-cpulimit
			}()

			_, err := c.getDirPackage(dir)
			if err != nil {
				t.Errorf("error checking %s: %v", dir, err)
			}
		}()
	}

	wg.Wait()

	if testing.Verbose() {
		fmt.Println(len(dirFiles), "packages typechecked in", time.Since(start))
	}
}

// stdlibChecker implements concurrent type-checking of the packages defined by
// dirFiles, which must define a closed set of packages (such as GOROOT/src).
type stdlibChecker struct {
	dirFiles map[string][]string // non-test files per directory; must be pre-populated

	mu   sync.Mutex
	pkgs map[string]*futurePackage // future cache of type-checking results
}

// A futurePackage is a future result of type-checking.
type futurePackage struct {
	done chan struct{} // guards pkg and err
	pkg  *Package
	err  error
}

func (c *stdlibChecker) Import(path string) (*Package, error) {
	panic("unimplemented: use ImportFrom")
}

func (c *stdlibChecker) ImportFrom(path, dir string, _ ImportMode) (*Package, error) {
	if path == "unsafe" {
		// unsafe cannot be type checked normally.
		return Unsafe, nil
	}

	p, err := build.Default.Import(path, dir, build.FindOnly)
	if err != nil {
		return nil, err
	}

	pkg, err := c.getDirPackage(p.Dir)
	if pkg != nil {
		// As long as pkg is non-nil, avoid redundant errors related to failed
		// imports. TestStdlib will collect errors once for each package.
		return pkg, nil
	}
	return nil, err
}

// getDirPackage gets the package defined in dir from the future cache.
//
// If this is the first goroutine requesting the package, getDirPackage
// type-checks.
func (c *stdlibChecker) getDirPackage(dir string) (*Package, error) {
	c.mu.Lock()
	fut, ok := c.pkgs[dir]
	if !ok {
		// First request for this package dir; type check.
		fut = &futurePackage{
			done: make(chan struct{}),
		}
		c.pkgs[dir] = fut
		files, ok := c.dirFiles[dir]
		c.mu.Unlock()
		if !ok {
			fut.err = fmt.Errorf("no files for %s", dir)
		} else {
			// Using dir as the package path here may be inconsistent with the behavior
			// of a normal importer, but is sufficient as dir is by construction unique
			// to this package.
			fut.pkg, fut.err = typecheckFiles(dir, files, c)
		}
		close(fut.done)
	} else {
		// Otherwise, await the result.
		c.mu.Unlock()
		<-fut.done
	}
	return fut.pkg, fut.err
}

// firstComment returns the contents of the first non-empty comment in
// the given file, "skip", or the empty string. No matter the present
// comments, if any of them contains a build tag, the result is always
// "skip". Only comments before the "package" token and within the first
// 4K of the file are considered.
func firstComment(filename string) string {
	f, err := os.Open(filename)
	if err != nil {
		return ""
	}
	defer f.Close()

	var src [4 << 10]byte // read at most 4KB
	n, _ := f.Read(src[:])

	var first string
	var s scanner.Scanner
	s.Init(fset.AddFile("", fset.Base(), n), src[:n], nil /* ignore errors */, scanner.ScanComments)
	for {
		_, tok, lit := s.Scan()
		switch tok {
		case token.COMMENT:
			// remove trailing */ of multi-line comment
			if lit[1] == '*' {
				lit = lit[:len(lit)-2]
			}
			contents := strings.TrimSpace(lit[2:])
			if strings.HasPrefix(contents, "go:build ") {
				return "skip"
			}
			if first == "" {
				first = contents // contents may be "" but that's ok
			}
			// continue as we may still see build tags

		case token.PACKAGE, token.EOF:
			return first
		}
	}
}

func testTestDir(t *testing.T, path string, ignore ...string) {
	files, err := os.ReadDir(path)
	if err != nil {
		// cmd/distpack deletes GOROOT/test, so skip the test if it isn't present.
		// cmd/distpack also requires GOROOT/VERSION to exist, so use that to
		// suppress false-positive skips.
		if _, err := os.Stat(filepath.Join(testenv.GOROOT(t), "test")); os.IsNotExist(err) {
			if _, err := os.Stat(filepath.Join(testenv.GOROOT(t), "VERSION")); err == nil {
				t.Skipf("skipping: GOROOT/test not present")
			}
		}
		t.Fatal(err)
	}

	excluded := make(map[string]bool)
	for _, filename := range ignore {
		excluded[filename] = true
	}

	fset := token.NewFileSet()
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
			if strings.Contains(comment, "-goexperiment") {
				continue // ignore this file
			}
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
		file, err := parser.ParseFile(fset, filename, nil, 0)
		if err == nil {
			conf := Config{
				GoVersion: goVersion,
				Importer:  stdLibImporter,
			}
			_, err = conf.Check(filename, fset, []*ast.File{file}, nil)
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
		"linkname2.go",   // go/types doesn't check validity of //go:xxx directives
		"linkname3.go",   // go/types doesn't check validity of //go:xxx directives
	)
}

func TestStdFixed(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in short mode")
	}

	testTestDir(t, filepath.Join(testenv.GOROOT(t), "test", "fixedbugs"),
		"bug248.go", "bug302.go", "bug369.go", // complex test instructions - ignore
		"bug398.go",      // go/types doesn't check for anonymous interface cycles (go.dev/issue/56103)
		"issue6889.go",   // gc-specific test
		"issue11362.go",  // canonical import path check
		"issue16369.go",  // go/types handles this correctly - not an issue
		"issue18459.go",  // go/types doesn't check validity of //go:xxx directives
		"issue18882.go",  // go/types doesn't check validity of //go:xxx directives
		"issue20529.go",  // go/types does not have constraints on stack size
		"issue22200.go",  // go/types does not have constraints on stack size
		"issue22200b.go", // go/types does not have constraints on stack size
		"issue25507.go",  // go/types does not have constraints on stack size
		"issue20780.go",  // go/types does not have constraints on stack size
		"bug251.go",      // go.dev/issue/34333 which was exposed with fix for go.dev/issue/34151
		"issue42058a.go", // go/types does not have constraints on channel element size
		"issue42058b.go", // go/types does not have constraints on channel element size
		"issue48097.go",  // go/types doesn't check validity of //go:xxx directives, and non-init bodyless function
		"issue48230.go",  // go/types doesn't check validity of //go:xxx directives
		"issue49767.go",  // go/types does not have constraints on channel element size
		"issue49814.go",  // go/types does not have constraints on array size
		"issue56103.go",  // anonymous interface cycles; will be a type checker error in 1.22
		"issue52697.go",  // go/types does not have constraints on stack size

		// These tests requires runtime/cgo.Incomplete, which is only available on some platforms.
		// However, go/types does not know about build constraints.
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

	// See go.dev/issue/46027: some imports are missing for this submodule.
	"crypto/internal/bigmod/_asm":             true,
	"crypto/internal/edwards25519/field/_asm": true,
	"crypto/sha256/_asm":                      true,
}

// printPackageMu synchronizes the printing of type-checked package files in
// the typecheckFiles function.
//
// Without synchronization, package files may be interleaved during concurrent
// type-checking.
var printPackageMu sync.Mutex

// typecheckFiles typechecks the given package files.
func typecheckFiles(path string, filenames []string, importer Importer) (*Package, error) {
	fset := token.NewFileSet()

	// Parse package files.
	var files []*ast.File
	for _, filename := range filenames {
		file, err := parser.ParseFile(fset, filename, nil, parser.AllErrors)
		if err != nil {
			return nil, err
		}

		files = append(files, file)
	}

	if testing.Verbose() {
		printPackageMu.Lock()
		fmt.Println("package", files[0].Name.Name)
		for _, filename := range filenames {
			fmt.Println("\t", filename)
		}
		printPackageMu.Unlock()
	}

	// Typecheck package files.
	var errs []error
	conf := Config{
		Error: func(err error) {
			errs = append(errs, err)
		},
		Importer: importer,
	}
	info := Info{Uses: make(map[*ast.Ident]Object)}
	pkg, _ := conf.Check(path, fset, files, &info)
	err := errors.Join(errs...)
	if err != nil {
		return pkg, err
	}

	// Perform checks of API invariants.

	// All Objects have a package, except predeclared ones.
	errorError := Universe.Lookup("error").Type().Underlying().(*Interface).ExplicitMethod(0) // (error).Error
	for id, obj := range info.Uses {
		predeclared := obj == Universe.Lookup(obj.Name()) || obj == errorError
		if predeclared == (obj.Pkg() != nil) {
			posn := fset.Position(id.Pos())
			if predeclared {
				return nil, fmt.Errorf("%s: predeclared object with package: %s", posn, obj)
			} else {
				return nil, fmt.Errorf("%s: user-defined object without package: %s", posn, obj)
			}
		}
	}

	return pkg, nil
}

// pkgFilenames returns the list of package filenames for the given directory.
func pkgFilenames(dir string, includeTest bool) ([]string, error) {
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
	if includeTest {
		for _, name := range pkg.TestGoFiles {
			filenames = append(filenames, filepath.Join(pkg.Dir, name))
		}
	}
	return filenames, nil
}

func walkPkgDirs(dir string, pkgh func(dir string, filenames []string), errh func(args ...any)) {
	w := walker{pkgh, errh}
	w.walk(dir)
}

type walker struct {
	pkgh func(dir string, filenames []string)
	errh func(args ...any)
}

func (w *walker) walk(dir string) {
	files, err := os.ReadDir(dir)
	if err != nil {
		w.errh(err)
		return
	}

	// apply pkgh to the files in directory dir

	// Don't get test files as these packages are imported.
	pkgFiles, err := pkgFilenames(dir, false)
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
