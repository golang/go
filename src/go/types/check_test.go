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

package types_test

import (
	"flag"
	"fmt"
	"go/ast"
	"go/importer"
	"go/internal/typeparams"
	"go/parser"
	"go/scanner"
	"go/token"
	"internal/buildcfg"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	. "go/types"
)

var (
	haltOnError  = flag.Bool("halt", false, "halt on error")
	verifyErrors = flag.Bool("verify", false, "verify errors (rather than list them) in TestManual")
	goVersion    = flag.String("lang", "", "Go language version (e.g. \"go1.12\") for TestManual")
)

var fset = token.NewFileSet()

// Positioned errors are of the form filename:line:column: message .
var posMsgRx = regexp.MustCompile(`^(.*:[0-9]+:[0-9]+): *(?s)(.*)`)

// splitError splits an error's error message into a position string
// and the actual error message. If there's no position information,
// pos is the empty string, and msg is the entire error message.
//
func splitError(err error) (pos, msg string) {
	msg = err.Error()
	if m := posMsgRx.FindStringSubmatch(msg); len(m) == 3 {
		pos = m[1]
		msg = m[2]
	}
	return
}

func parseFiles(t *testing.T, filenames []string, srcs [][]byte, mode parser.Mode) ([]*ast.File, []error) {
	var files []*ast.File
	var errlist []error
	for i, filename := range filenames {
		file, err := parser.ParseFile(fset, filename, srcs[i], mode)
		if file == nil {
			t.Fatalf("%s: %s", filename, err)
		}
		files = append(files, file)
		if err != nil {
			if list, _ := err.(scanner.ErrorList); len(list) > 0 {
				for _, err := range list {
					errlist = append(errlist, err)
				}
			} else {
				errlist = append(errlist, err)
			}
		}
	}
	return files, errlist
}

// ERROR comments must start with text `ERROR "rx"` or `ERROR rx` where
// rx is a regular expression that matches the expected error message.
// Space around "rx" or rx is ignored. Use the form `ERROR HERE "rx"`
// for error messages that are located immediately after rather than
// at a token's position.
//
var errRx = regexp.MustCompile(`^ *ERROR *(HERE)? *"?([^"]*)"?`)

// errMap collects the regular expressions of ERROR comments found
// in files and returns them as a map of error positions to error messages.
//
// srcs must be a slice of the same length as files, containing the original
// source for the parsed AST.
func errMap(t *testing.T, files []*ast.File, srcs [][]byte) map[string][]string {
	// map of position strings to lists of error message patterns
	errmap := make(map[string][]string)

	for i, file := range files {
		tok := fset.File(file.Package)
		src := srcs[i]
		var s scanner.Scanner
		s.Init(tok, src, nil, scanner.ScanComments)
		var prev token.Pos // position of last non-comment, non-semicolon token
		var here token.Pos // position immediately after the token at position prev

	scanFile:
		for {
			pos, tok, lit := s.Scan()
			switch tok {
			case token.EOF:
				break scanFile
			case token.COMMENT:
				if lit[1] == '*' {
					lit = lit[:len(lit)-2] // strip trailing */
				}
				if s := errRx.FindStringSubmatch(lit[2:]); len(s) == 3 {
					pos := prev
					if s[1] == "HERE" {
						pos = here
					}
					p := fset.Position(pos).String()
					errmap[p] = append(errmap[p], strings.TrimSpace(s[2]))
				}
			case token.SEMICOLON:
				// ignore automatically inserted semicolon
				if lit == "\n" {
					continue scanFile
				}
				fallthrough
			default:
				prev = pos
				var l int // token length
				if tok.IsLiteral() {
					l = len(lit)
				} else {
					l = len(tok.String())
				}
				here = prev + token.Pos(l)
			}
		}
	}

	return errmap
}

func eliminate(t *testing.T, errmap map[string][]string, errlist []error) {
	for _, err := range errlist {
		pos, gotMsg := splitError(err)
		list := errmap[pos]
		index := -1 // list index of matching message, if any
		// we expect one of the messages in list to match the error at pos
		for i, wantRx := range list {
			rx, err := regexp.Compile(wantRx)
			if err != nil {
				t.Errorf("%s: %v", pos, err)
				continue
			}
			if rx.MatchString(gotMsg) {
				index = i
				break
			}
		}
		if index >= 0 {
			// eliminate from list
			if n := len(list) - 1; n > 0 {
				// not the last entry - swap in last element and shorten list by 1
				list[index] = list[n]
				errmap[pos] = list[:n]
			} else {
				// last entry - remove list from map
				delete(errmap, pos)
			}
		} else {
			t.Errorf("%s: no error expected: %q", pos, gotMsg)
		}
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

// excludedForUnifiedBuild lists files that cannot be tested
// when using the unified build's export data.
// TODO(gri) enable as soon as the unified build supports this.
var excludedForUnifiedBuild = map[string]bool{
	"issue47818.go2": true,
	"issue49705.go2": true,
}

func testFiles(t *testing.T, sizes Sizes, filenames []string, srcs [][]byte, manual bool, imp Importer) {
	if len(filenames) == 0 {
		t.Fatal("no source files")
	}

	if buildcfg.Experiment.Unified {
		for _, f := range filenames {
			if excludedForUnifiedBuild[filepath.Base(f)] {
				t.Logf("%s cannot be tested with unified build - skipped", f)
				return
			}
		}
	}

	if strings.HasSuffix(filenames[0], ".go1") {
		// TODO(rfindley): re-enable this test by using GoVersion.
		t.Skip("type params are enabled")
	}

	mode := parser.AllErrors
	if !strings.HasSuffix(filenames[0], ".go2") && !manual {
		mode |= typeparams.DisallowParsing
	}

	// parse files and collect parser errors
	files, errlist := parseFiles(t, filenames, srcs, mode)

	pkgName := "<no package>"
	if len(files) > 0 {
		pkgName = files[0].Name.Name
	}

	// if no Go version is given, consider the package name
	goVersion := *goVersion
	if goVersion == "" {
		goVersion = asGoVersion(pkgName)
	}

	listErrors := manual && !*verifyErrors
	if listErrors && len(errlist) > 0 {
		t.Errorf("--- %s:", pkgName)
		for _, err := range errlist {
			t.Error(err)
		}
	}

	// typecheck and collect typechecker errors
	var conf Config
	conf.Sizes = sizes
	conf.GoVersion = goVersion

	// special case for importC.src
	if len(filenames) == 1 {
		if strings.HasSuffix(filenames[0], "importC.src") {
			conf.FakeImportC = true
		}
	}

	conf.Importer = imp
	if imp == nil {
		conf.Importer = importer.Default()
	}
	conf.Error = func(err error) {
		if *haltOnError {
			defer panic(err)
		}
		if listErrors {
			t.Error(err)
			return
		}
		// Ignore secondary error messages starting with "\t";
		// they are clarifying messages for a primary error.
		if !strings.Contains(err.Error(), ": \t") {
			errlist = append(errlist, err)
		}
	}
	conf.Check(pkgName, fset, files, nil)

	if listErrors {
		return
	}

	for _, err := range errlist {
		err, ok := err.(Error)
		if !ok {
			continue
		}
		code := readCode(err)
		if code == 0 {
			t.Errorf("missing error code: %v", err)
		}
	}

	// match and eliminate errors;
	// we are expecting the following errors
	errmap := errMap(t, files, srcs)
	eliminate(t, errmap, errlist)

	// there should be no expected errors left
	if len(errmap) > 0 {
		t.Errorf("--- %s: %d source positions with expected (but not reported) errors:", pkgName, len(errmap))
		for pos, list := range errmap {
			for _, rx := range list {
				t.Errorf("%s: %q", pos, rx)
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
// 	go test -run Manual -- foo.go bar.go
//
// If no source arguments are provided, the file testdata/manual.go2
// is used instead.
// Provide the -verify flag to verify errors against ERROR comments
// in the input files rather than having a list of errors reported.
// The accepted Go language version can be controlled with the -lang
// flag.
func TestManual(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	filenames := flag.Args()
	if len(filenames) == 0 {
		filenames = []string{filepath.FromSlash("testdata/manual.go2")}
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
		testDir(t, filenames[0], true)
	} else {
		testPkg(t, filenames, true)
	}
}

func TestLongConstants(t *testing.T) {
	format := "package longconst\n\nconst _ = %s\nconst _ = %s // ERROR excessively long constant"
	src := fmt.Sprintf(format, strings.Repeat("1", 9999), strings.Repeat("1", 10001))
	testFiles(t, nil, []string{"longconst.go"}, [][]byte{[]byte(src)}, false, nil)
}

// TestIndexRepresentability tests that constant index operands must
// be representable as int even if they already have a type that can
// represent larger values.
func TestIndexRepresentability(t *testing.T) {
	const src = "package index\n\nvar s []byte\nvar _ = s[int64 /* ERROR \"int64\\(1\\) << 40 \\(.*\\) overflows int\" */ (1) << 40]"
	testFiles(t, &StdSizes{4, 4}, []string{"index.go"}, [][]byte{[]byte(src)}, false, nil)
}

func TestIssue47243_TypedRHS(t *testing.T) {
	// The RHS of the shift expression below overflows uint on 32bit platforms,
	// but this is OK as it is explicitly typed.
	const src = "package issue47243\n\nvar a uint64; var _ = a << uint64(4294967296)" // uint64(1<<32)
	testFiles(t, &StdSizes{4, 4}, []string{"p.go"}, [][]byte{[]byte(src)}, false, nil)
}

func TestCheck(t *testing.T)    { DefPredeclaredTestFuncs(); testDirFiles(t, "testdata/check", false) }
func TestSpec(t *testing.T)     { DefPredeclaredTestFuncs(); testDirFiles(t, "testdata/spec", false) }
func TestExamples(t *testing.T) { testDirFiles(t, "testdata/examples", false) }
func TestFixedbugs(t *testing.T) {
	DefPredeclaredTestFuncs()
	testDirFiles(t, "testdata/fixedbugs", false)
}

func testDirFiles(t *testing.T, dir string, manual bool) {
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
			testDir(t, path, manual)
		} else {
			t.Run(filepath.Base(path), func(t *testing.T) {
				testPkg(t, []string{path}, manual)
			})
		}
	}
}

func testDir(t *testing.T, dir string, manual bool) {
	testenv.MustHaveGoBuild(t)

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
		testPkg(t, filenames, manual)
	})
}

// TODO(rFindley) reconcile the different test setup in go/types with types2.
func testPkg(t *testing.T, filenames []string, manual bool) {
	srcs := make([][]byte, len(filenames))
	for i, filename := range filenames {
		src, err := os.ReadFile(filename)
		if err != nil {
			t.Fatalf("could not read %s: %v", filename, err)
		}
		srcs[i] = src
	}
	testFiles(t, nil, filenames, srcs, manual, nil)
}
