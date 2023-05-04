// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a typechecker test harness. The packages specified
// in tests are typechecked. Error messages reported by the typechecker are
// compared against the errors expected in the test files.
//
// Expected errors are indicated in the test files by putting comments
// of the form /* ERROR pattern */ or /* ERRORx pattern */ (or a similar
// //-style line comment) immediately following the tokens where errors
// are reported. There must be exactly one blank before and after the
// ERROR/ERRORx indicator, and the pattern must be a properly quoted Go
// string.
//
// The harness will verify that each ERROR pattern is a substring of the
// error reported at that source position, and that each ERRORx pattern
// is a regular expression matching the respective error.
// Consecutive comments may be used to indicate multiple errors reported
// at the same position.
//
// For instance, the following test source indicates that an "undeclared"
// error should be reported for the undeclared variable x:
//
//	package p
//	func f() {
//		_ = x /* ERROR "undeclared" */ + 1
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
	"strconv"
	"strings"
	"testing"

	. "cmd/compile/internal/types2"
)

var (
	haltOnError  = flag.Bool("halt", false, "halt on error")
	verifyErrors = flag.Bool("verify", false, "verify errors (rather than list them) in TestManual")
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

func unpackError(err error) (syntax.Pos, string) {
	switch err := err.(type) {
	case syntax.Error:
		return err.Pos, err.Msg
	case Error:
		return err.Pos, err.Msg
	default:
		return nopos, err.Error()
	}
}

// absDiff returns the absolute difference between x and y.
func absDiff(x, y uint) uint {
	if x < y {
		return y - x
	}
	return x - y
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

	// collect expected errors
	errmap := make(map[string]map[uint][]syntax.Error)
	for _, filename := range filenames {
		f, err := os.Open(filename)
		if err != nil {
			t.Error(err)
			continue
		}
		if m := syntax.CommentMap(f, regexp.MustCompile("^ ERRORx? ")); len(m) > 0 {
			errmap[filename] = m
		}
		f.Close()
	}

	// match against found errors
	var indices []int // list indices of matching errors, reused for each error
	for _, err := range errlist {
		gotPos, gotMsg := unpackError(err)

		// find list of errors for the respective error line
		filename := gotPos.Base().Filename()
		filemap := errmap[filename]
		line := gotPos.Line()
		var errList []syntax.Error
		if filemap != nil {
			errList = filemap[line]
		}

		// At least one of the errors in errList should match the current error.
		indices = indices[:0]
		for i, want := range errList {
			pattern, substr := strings.CutPrefix(want.Msg, " ERROR ")
			if !substr {
				var found bool
				pattern, found = strings.CutPrefix(want.Msg, " ERRORx ")
				if !found {
					panic("unreachable")
				}
			}
			pattern, err := strconv.Unquote(strings.TrimSpace(pattern))
			if err != nil {
				t.Errorf("%s:%d:%d: %v", filename, line, want.Pos.Col(), err)
				continue
			}
			if substr {
				if !strings.Contains(gotMsg, pattern) {
					continue
				}
			} else {
				rx, err := regexp.Compile(pattern)
				if err != nil {
					t.Errorf("%s:%d:%d: %v", filename, line, want.Pos.Col(), err)
					continue
				}
				if !rx.MatchString(gotMsg) {
					continue
				}
			}
			indices = append(indices, i)
		}
		if len(indices) == 0 {
			t.Errorf("%s: no error expected: %q", gotPos, gotMsg)
			continue
		}
		// len(indices) > 0

		// If there are multiple matching errors, select the one with the closest column position.
		index := -1 // index of matching error
		var delta uint
		for _, i := range indices {
			if d := absDiff(gotPos.Col(), errList[i].Pos.Col()); index < 0 || d < delta {
				index, delta = i, d
			}
		}

		// The closest column position must be within expected colDelta.
		if delta > colDelta {
			t.Errorf("%s: got col = %d; want %d", gotPos, gotPos.Col(), errList[index].Pos.Col())
		}

		// eliminate from errList
		if n := len(errList) - 1; n > 0 {
			// not the last entry - slide entries down (don't reorder)
			copy(errList[index:], errList[index+1:])
			filemap[line] = errList[:n]
		} else {
			// last entry - remove errList from filemap
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
			for line, errList := range filemap {
				for _, err := range errList {
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
	testDirFiles(t, "../../../../internal/types/testdata/check", 50, false) // TODO(gri) narrow column tolerance
}
func TestSpec(t *testing.T) { testDirFiles(t, "../../../../internal/types/testdata/spec", 0, false) }
func TestExamples(t *testing.T) {
	testDirFiles(t, "../../../../internal/types/testdata/examples", 125, false)
} // TODO(gri) narrow column tolerance
func TestFixedbugs(t *testing.T) {
	testDirFiles(t, "../../../../internal/types/testdata/fixedbugs", 100, false)
}                            // TODO(gri) narrow column tolerance
func TestLocal(t *testing.T) { testDirFiles(t, "testdata/local", 0, false) }

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
