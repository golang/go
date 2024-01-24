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
	"internal/buildcfg"
	"internal/testenv"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"

	. "cmd/compile/internal/types2"
)

var (
	haltOnError  = flag.Bool("halt", false, "halt on error")
	verifyErrors = flag.Bool("verify", false, "verify errors (rather than list them) in TestManual")
)

func parseFiles(t *testing.T, filenames []string, srcs [][]byte, mode syntax.Mode) ([]*syntax.File, []error) {
	var files []*syntax.File
	var errlist []error
	errh := func(err error) { errlist = append(errlist, err) }
	for i, filename := range filenames {
		base := syntax.NewFileBase(filename)
		r := bytes.NewReader(srcs[i])
		file, err := syntax.Parse(base, r, errh, nil, mode)
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

// parseFlags parses flags from the first line of the given source if the line
// starts with "//" (line comment) followed by "-" (possibly with spaces
// between). Otherwise the line is ignored.
func parseFlags(src []byte, flags *flag.FlagSet) error {
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
	const maxLen = 256
	if end < 0 || end > maxLen {
		return fmt.Errorf("flags comment line too long")
	}

	return flags.Parse(strings.Fields(string(src[:end])))
}

// testFiles type-checks the package consisting of the given files, and
// compares the resulting errors with the ERROR annotations in the source.
// Except for manual tests, each package is type-checked twice, once without
// use of Alias types, and once with Alias types.
//
// The srcs slice contains the file content for the files named in the
// filenames slice. The colDelta parameter specifies the tolerance for position
// mismatch when comparing errors. The manual parameter specifies whether this
// is a 'manual' test.
//
// If provided, opts may be used to mutate the Config before type-checking.
func testFiles(t *testing.T, filenames []string, srcs [][]byte, colDelta uint, manual bool, opts ...func(*Config)) {
	// Alias types are disabled by default
	testFilesImpl(t, filenames, srcs, colDelta, manual, opts...)
	if !manual {
		t.Setenv("GODEBUG", "gotypesalias=1")
		testFilesImpl(t, filenames, srcs, colDelta, manual, opts...)
	}
}

func testFilesImpl(t *testing.T, filenames []string, srcs [][]byte, colDelta uint, manual bool, opts ...func(*Config)) {
	if len(filenames) == 0 {
		t.Fatal("no source files")
	}

	// parse files
	files, errlist := parseFiles(t, filenames, srcs, 0)
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

	// set up typechecker
	var conf Config
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

	// apply custom configuration
	for _, opt := range opts {
		opt(&conf)
	}

	// apply flag setting (overrides custom configuration)
	var goexperiment, gotypesalias string
	flags := flag.NewFlagSet("", flag.PanicOnError)
	flags.StringVar(&conf.GoVersion, "lang", "", "")
	flags.StringVar(&goexperiment, "goexperiment", "", "")
	flags.BoolVar(&conf.FakeImportC, "fakeImportC", false, "")
	flags.StringVar(&gotypesalias, "gotypesalias", "", "")
	if err := parseFlags(srcs[0], flags); err != nil {
		t.Fatal(err)
	}

	exp, err := buildcfg.ParseGOEXPERIMENT(runtime.GOOS, runtime.GOARCH, goexperiment)
	if err != nil {
		t.Fatal(err)
	}
	old := buildcfg.Experiment
	defer func() {
		buildcfg.Experiment = old
	}()
	buildcfg.Experiment = *exp

	// By default, gotypesalias is not set.
	if gotypesalias != "" {
		t.Setenv("GODEBUG", "gotypesalias="+gotypesalias)
	}

	// Provide Config.Info with all maps so that info recording is tested.
	info := Info{
		Types:        make(map[syntax.Expr]TypeAndValue),
		Instances:    make(map[*syntax.Name]Instance),
		Defs:         make(map[*syntax.Name]Object),
		Uses:         make(map[*syntax.Name]Object),
		Implicits:    make(map[syntax.Node]Object),
		Selections:   make(map[*syntax.SelectorExpr]*Selection),
		Scopes:       make(map[syntax.Node]*Scope),
		FileVersions: make(map[*syntax.PosBase]string),
	}

	// typecheck
	conf.Check(pkgName, files, &info)
	if listErrors {
		return
	}

	// collect expected errors
	errmap := make(map[string]map[uint][]syntax.Error)
	for i, filename := range filenames {
		if m := syntax.CommentMap(bytes.NewReader(srcs[i]), regexp.MustCompile("^ ERRORx? ")); len(m) > 0 {
			errmap[filename] = m
		}
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

// boolFieldAddr(conf, name) returns the address of the boolean field conf.<name>.
// For accessing unexported fields.
func boolFieldAddr(conf *Config, name string) *bool {
	v := reflect.Indirect(reflect.ValueOf(conf))
	return (*bool)(v.FieldByName(name).Addr().UnsafePointer())
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
		testPkg(t, filenames, 0, true)
	}
}

func TestLongConstants(t *testing.T) {
	format := `package longconst; const _ = %s /* ERROR "constant overflow" */; const _ = %s // ERROR "excessively long constant"`
	src := fmt.Sprintf(format, strings.Repeat("1", 9999), strings.Repeat("1", 10001))
	testFiles(t, []string{"longconst.go"}, [][]byte{[]byte(src)}, 0, false)
}

func withSizes(sizes Sizes) func(*Config) {
	return func(cfg *Config) {
		cfg.Sizes = sizes
	}
}

// TestIndexRepresentability tests that constant index operands must
// be representable as int even if they already have a type that can
// represent larger values.
func TestIndexRepresentability(t *testing.T) {
	const src = `package index; var s []byte; var _ = s[int64 /* ERRORx "int64\\(1\\) << 40 \\(.*\\) overflows int" */ (1) << 40]`
	testFiles(t, []string{"index.go"}, [][]byte{[]byte(src)}, 0, false, withSizes(&StdSizes{4, 4}))
}

func TestIssue47243_TypedRHS(t *testing.T) {
	// The RHS of the shift expression below overflows uint on 32bit platforms,
	// but this is OK as it is explicitly typed.
	const src = `package issue47243; var a uint64; var _ = a << uint64(4294967296)` // uint64(1<<32)
	testFiles(t, []string{"p.go"}, [][]byte{[]byte(src)}, 0, false, withSizes(&StdSizes{4, 4}))
}

func TestCheck(t *testing.T) {
	old := buildcfg.Experiment.RangeFunc
	defer func() {
		buildcfg.Experiment.RangeFunc = old
	}()
	buildcfg.Experiment.RangeFunc = true

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
				testPkg(t, []string{path}, colDelta, manual)
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
		testPkg(t, filenames, colDelta, manual)
	})
}

func testPkg(t *testing.T, filenames []string, colDelta uint, manual bool) {
	srcs := make([][]byte, len(filenames))
	for i, filename := range filenames {
		src, err := os.ReadFile(filename)
		if err != nil {
			t.Fatalf("could not read %s: %v", filename, err)
		}
		srcs[i] = src
	}
	testFiles(t, filenames, srcs, colDelta, manual)
}
