// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a typechecker test harness. The packages specified
// in tests are typechecked. Error messages reported by the typechecker are
// compared against the error messages expected in the test files.
//
// Expected errors are indicated in the test files by putting a comment
// of the form /* ERROR "rx" */ immediately following an offending token.
// The harness will verify that an error matching the regular expression
// rx is reported at that source position. Consecutive comments may be
// used to indicate multiple errors for the same token position.
//
// For instance, the following test file indicates that a "not declared"
// error should be reported for the undeclared variable x:
//
//	package p
//	func f() {
//		_ = x /* ERROR "not declared" */ + 1
//	}

// TODO(gri) Also collect strict mode errors of the form /* STRICT ... */
//           and test against strict mode.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"flag"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	. "cmd/compile/internal/types2"
)

var (
	haltOnError = flag.Bool("halt", false, "halt on error")
	listErrors  = flag.Bool("errlist", false, "list errors")
	testFiles   = flag.String("files", "", "comma-separated list of test files")
	goVersion   = flag.String("lang", "", "Go language version (e.g. \"go1.12\"")
)

func parseFiles(t *testing.T, filenames []string, mode syntax.Mode) ([]*syntax.File, []error) {
	var files []*syntax.File
	var errlist []error
	errh := func(err error) { errlist = append(errlist, err) }
	for _, filename := range filenames {
		file, err := syntax.ParseFile(filename, errh, nil, mode)
		if file == nil {
			t.Fatalf("%s: %s", filename, err)
		}
		files = append(files, file)
	}
	return files, errlist
}

func unpackError(err error) syntax.Error {
	switch err := err.(type) {
	case syntax.Error:
		return err
	case Error:
		return syntax.Error{Pos: err.Pos, Msg: err.Msg}
	default:
		return syntax.Error{Msg: err.Error()}
	}
}

// delta returns the absolute difference between x and y.
func delta(x, y uint) uint {
	switch {
	case x < y:
		return y - x
	case x > y:
		return x - y
	default:
		return 0
	}
}

// goVersionRx matches a Go version string using '_', e.g. "go1_12".
var goVersionRx = regexp.MustCompile(`^go[1-9][0-9]*_(0|[1-9][0-9]*)$`)

// asGoVersion returns a regular Go language version string
// if s is a Go version string using '_' rather than '.' to
// separate the major and minor version numbers (e.g. "go1_12").
// Otherwise it returns the empty string.
func asGoVersion(s string) string {
	if goVersionRx.MatchString(s) {
		return strings.Replace(s, "_", ".", 1)
	}
	return ""
}

func checkFiles(t *testing.T, filenames []string, goVersion string, colDelta uint, trace bool) {
	if len(filenames) == 0 {
		t.Fatal("no source files")
	}

	var mode syntax.Mode
	if strings.HasSuffix(filenames[0], ".go2") {
		mode |= syntax.AllowGenerics
	}
	// parse files and collect parser errors
	files, errlist := parseFiles(t, filenames, mode)

	pkgName := "<no package>"
	if len(files) > 0 {
		pkgName = files[0].PkgName.Value
	}

	// if no Go version is given, consider the package name
	if goVersion == "" {
		goVersion = asGoVersion(pkgName)
	}

	if *listErrors && len(errlist) > 0 {
		t.Errorf("--- %s:", pkgName)
		for _, err := range errlist {
			t.Error(err)
		}
	}

	// typecheck and collect typechecker errors
	var conf Config
	conf.GoVersion = goVersion
	conf.AcceptMethodTypeParams = true
	conf.InferFromConstraints = true
	// special case for importC.src
	if len(filenames) == 1 && strings.HasSuffix(filenames[0], "importC.src") {
		conf.FakeImportC = true
	}
	conf.Trace = trace
	conf.Importer = defaultImporter()
	conf.Error = func(err error) {
		if *haltOnError {
			defer panic(err)
		}
		if *listErrors {
			t.Error(err)
			return
		}
		errlist = append(errlist, err)
	}
	conf.Check(pkgName, files, nil)

	if *listErrors {
		return
	}

	// collect expected errors
	errmap := make(map[string]map[uint][]syntax.Error)
	for _, filename := range filenames {
		f, err := os.Open(filename)
		if err != nil {
			t.Error(err)
			continue
		}
		if m := syntax.ErrorMap(f); len(m) > 0 {
			errmap[filename] = m
		}
		f.Close()
	}

	// match against found errors
	// TODO(gri) sort err list to avoid mismatched when having multiple errors
	for _, err := range errlist {
		got := unpackError(err)

		// find list of errors for the respective error line
		filename := got.Pos.Base().Filename()
		filemap := errmap[filename]
		line := got.Pos.Line()
		var list []syntax.Error
		if filemap != nil {
			list = filemap[line]
		}
		// list may be nil

		// one of errors in list should match the current error
		index := -1 // list index of matching message, if any
		for i, want := range list {
			rx, err := regexp.Compile(want.Msg)
			if err != nil {
				t.Errorf("%s:%d:%d: %v", filename, line, want.Pos.Col(), err)
				continue
			}
			if rx.MatchString(got.Msg) {
				index = i
				break
			}
		}
		if index < 0 {
			t.Errorf("%s: no error expected: %q", got.Pos, got.Msg)
			continue
		}

		// column position must be within expected colDelta
		want := list[index]
		if delta(got.Pos.Col(), want.Pos.Col()) > colDelta {
			t.Errorf("%s: got col = %d; want %d", got.Pos, got.Pos.Col(), want.Pos.Col())
		}

		// eliminate from list
		if n := len(list) - 1; n > 0 {
			// not the last entry - swap in last element and shorten list by 1
			// TODO(gri) avoid changing the order of entries
			list[index] = list[n]
			filemap[line] = list[:n]
		} else {
			// last entry - remove list from filemap
			delete(filemap, line)
		}

		// if filemap is empty, eliminate from errmap
		if len(filemap) == 0 {
			delete(errmap, filename)
		}
	}

	// there should be no expected errors left
	if len(errmap) > 0 {
		t.Errorf("--- %s: unreported errors:", pkgName)
		for filename, filemap := range errmap {
			for line, list := range filemap {
				for _, err := range list {
					t.Errorf("%s:%d:%d: %s", filename, line, err.Pos.Col(), err.Msg)
				}
			}
		}
	}
}

// TestCheck is for manual testing of selected input files, provided with -files.
// The accepted Go language version can be controlled with the -lang flag.
func TestCheck(t *testing.T) {
	if *testFiles == "" {
		return
	}
	testenv.MustHaveGoBuild(t)
	DefPredeclaredTestFuncs()
	checkFiles(t, strings.Split(*testFiles, ","), *goVersion, 0, testing.Verbose())
}

// TODO(gri) go/types has an extra TestLongConstants test

func TestTestdata(t *testing.T)  { DefPredeclaredTestFuncs(); testDir(t, "testdata", 75) } // TODO(gri) narrow column tolerance
func TestExamples(t *testing.T)  { testDir(t, "examples", 0) }
func TestFixedbugs(t *testing.T) { testDir(t, "fixedbugs", 0) }

func testDir(t *testing.T, dir string, colDelta uint) {
	testenv.MustHaveGoBuild(t)

	fis, err := os.ReadDir(dir)
	if err != nil {
		t.Error(err)
		return
	}

	for _, fi := range fis {
		path := filepath.Join(dir, fi.Name())

		// if fi is a directory, its files make up a single package
		var filenames []string
		if fi.IsDir() {
			fis, err := os.ReadDir(path)
			if err != nil {
				t.Error(err)
				continue
			}
			for _, fi := range fis {
				filenames = append(filenames, filepath.Join(path, fi.Name()))
			}
		} else {
			filenames = []string{path}
		}

		t.Run(filepath.Base(path), func(t *testing.T) {
			checkFiles(t, filenames, *goVersion, colDelta, false)
		})
	}
}
