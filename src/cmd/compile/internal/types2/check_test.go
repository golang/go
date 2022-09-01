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

package types2_test

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	. "cmd/compile/internal/types2"
)

var (
	haltOnError  = flag.Bool("halt", false, "halt on error")
	verifyErrors = flag.Bool("verify", false, "verify errors (rather than list them) in TestManual")
	goVersion    = flag.String("lang", "", "Go language version (e.g. \"go1.12\")")
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

// Note: parseFlags is identical to the version in go/types which is
//       why it has a src argument even though here it is always nil.

// parseFlags parses flags from the first line of the given source
// (from src if present, or by reading from the file) if the line
// starts with "//" (line comment) followed by "-" (possibly with
// spaces between). Otherwise the line is ignored.
func parseFlags(filename string, src []byte, flags *flag.FlagSet) error {
	// If there is no src, read from the file.
	const maxLen = 256
	if len(src) == 0 {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}

		var buf [maxLen]byte
		n, err := f.Read(buf[:])
		if err != nil {
			return err
		}
		src = buf[:n]
	}

	// we must have a line comment that starts with a "-"
	const prefix = "//"
	if !bytes.HasPrefix(src, []byte(prefix)) {
		return nil // first line is not a line comment
	}
	src = src[len(prefix):]
	if i := bytes.Index(src, []byte("-")); i < 0 || len(bytes.TrimSpace(src[:i])) != 0 {
		return nil // comment doesn't start with a "-"
	}
	end := bytes.Index(src, []byte("\n"))
	if end < 0 || end > maxLen {
		return fmt.Errorf("flags comment line too long")
	}

	return flags.Parse(strings.Fields(string(src[:end])))
}

func testFiles(t *testing.T, filenames []string, colDelta uint, manual bool) {
	if len(filenames) == 0 {
		t.Fatal("no source files")
	}

	var conf Config
	flags := flag.NewFlagSet("", flag.PanicOnError)
	flags.StringVar(&conf.GoVersion, "lang", "", "")
	flags.BoolVar(&conf.FakeImportC, "fakeImportC", false, "")
	if err := parseFlags(filenames[0], nil, flags); err != nil {
		t.Fatal(err)
	}

	files, errlist := parseFiles(t, filenames, 0)

	pkgName := "<no package>"
	if len(files) > 0 {
		pkgName = files[0].PkgName.Value
	}

	listErrors := manual && !*verifyErrors
	if listErrors && len(errlist) > 0 {
		t.Errorf("--- %s:", pkgName)
		for _, err := range errlist {
			t.Error(err)
		}
	}

	// typecheck and collect typechecker errors
	conf.Trace = manual && testing.Verbose()
	conf.Importer = defaultImporter()
	conf.Error = func(err error) {
		if *haltOnError {
			defer panic(err)
		}
		if listErrors {
			t.Error(err)
			return
		}
		errlist = append(errlist, err)
	}
	conf.Check(pkgName, files, nil)

	if listErrors {
		return
	}

	// sort errlist in source order
	sort.Slice(errlist, func(i, j int) bool {
		pi := unpackError(errlist[i]).Pos
		pj := unpackError(errlist[j]).Pos
		return pi.Cmp(pj) < 0
	})

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
			// not the last entry - slide entries down (don't reorder)
			copy(list[index:], list[index+1:])
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

// TestManual is for manual testing of a package - either provided
// as a list of filenames belonging to the package, or a directory
// name containing the package files - after the test arguments
// (and a separating "--"). For instance, to test the package made
// of the files foo.go and bar.go, use:
//
//	go test -run Manual -- foo.go bar.go
//
// If no source arguments are provided, the file testdata/manual.go
// is used instead.
// Provide the -verify flag to verify errors against ERROR comments
// in the input files rather than having a list of errors reported.
// The accepted Go language version can be controlled with the -lang
// flag.
func TestManual(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	filenames := flag.Args()
	if len(filenames) == 0 {
		filenames = []string{filepath.FromSlash("testdata/manual.go")}
	}

	info, err := os.Stat(filenames[0])
	if err != nil {
		t.Fatalf("TestManual: %v", err)
	}

	DefPredeclaredTestFuncs()
	if info.IsDir() {
		if len(filenames) > 1 {
			t.Fatal("TestManual: must have only one directory argument")
		}
		testDir(t, filenames[0], 0, true)
	} else {
		testFiles(t, filenames, 0, true)
	}
}

// TODO(gri) go/types has extra TestLongConstants and TestIndexRepresentability tests

func TestCheck(t *testing.T) {
	DefPredeclaredTestFuncs()
	testDirFiles(t, "../../../../go/types/testdata/check", 55, false) // TODO(gri) narrow column tolerance
}
func TestSpec(t *testing.T)     { testDirFiles(t, "../../../../go/types/testdata/spec", 0, false) }
func TestExamples(t *testing.T) { testDirFiles(t, "../../../../go/types/testdata/examples", 45, false) } // TODO(gri) narrow column tolerance
func TestFixedbugs(t *testing.T) {
	testDirFiles(t, "../../../../go/types/testdata/fixedbugs", 100, false)
} // TODO(gri) narrow column tolerance

func testDirFiles(t *testing.T, dir string, colDelta uint, manual bool) {
	testenv.MustHaveGoBuild(t)
	dir = filepath.FromSlash(dir)

	fis, err := os.ReadDir(dir)
	if err != nil {
		t.Error(err)
		return
	}

	for _, fi := range fis {
		path := filepath.Join(dir, fi.Name())

		// If fi is a directory, its files make up a single package.
		if fi.IsDir() {
			testDir(t, path, colDelta, manual)
		} else {
			t.Run(filepath.Base(path), func(t *testing.T) {
				testFiles(t, []string{path}, colDelta, manual)
			})
		}
	}
}

func testDir(t *testing.T, dir string, colDelta uint, manual bool) {
	fis, err := os.ReadDir(dir)
	if err != nil {
		t.Error(err)
		return
	}

	var filenames []string
	for _, fi := range fis {
		filenames = append(filenames, filepath.Join(dir, fi.Name()))
	}

	t.Run(filepath.Base(dir), func(t *testing.T) {
		testFiles(t, filenames, colDelta, manual)
	})
}
