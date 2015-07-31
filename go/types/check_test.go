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
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"regexp"
	"runtime"
	"strings"
	"testing"

	_ "golang.org/x/tools/go/gcimporter"
	. "golang.org/x/tools/go/types"
)

var (
	listErrors = flag.Bool("list", false, "list errors")
	testFiles  = flag.String("files", "", "space-separated list of test files")
)

// The test filenames do not end in .go so that they are invisible
// to gofmt since they contain comments that must not change their
// positions relative to surrounding tokens.

// Each tests entry is list of files belonging to the same package.
var tests = []struct {
	files string      // blank-separated list of file names
	cond  func() bool // condition under which the test should be run; nil means always
}{
	{"testdata/errors.src", nil},
	{"testdata/importdecl0a.src testdata/importdecl0b.src", nil},
	{"testdata/importdecl1a.src testdata/importdecl1b.src", nil},
	{"testdata/cycles.src", nil},
	{"testdata/cycles1.src", nil},
	{"testdata/cycles2.src", nil},
	{"testdata/cycles3.src", nil},
	{"testdata/cycles4.src", nil},
	{"testdata/init0.src", nil},
	{"testdata/init1.src", nil},
	{"testdata/init2.src", nil},
	{"testdata/decls0.src", nil},
	{"testdata/decls1.src", nil},
	{"testdata/decls2a.src testdata/decls2b.src", nil},
	{"testdata/decls3.src", nil},
	{"testdata/const0.src", nil},
	{"testdata/const1.src", nil},
	{"testdata/constdecl.src", notGo1_4}, // Go 1.4 parser doesn't report certain errors
	{"testdata/vardecl.src", notGo1_4},   // Go 1.4 parser doesn't report certain errors
	{"testdata/expr0.src", nil},
	{"testdata/expr1.src", nil},
	{"testdata/expr2.src", nil},
	{"testdata/expr3.src", notGo1_4}, // Go 1.4 parser doesn't permit omitting key type in map literals
	{"testdata/methodsets.src", nil},
	{"testdata/shifts.src", nil},
	{"testdata/builtins.src", nil},
	{"testdata/conversions.src", nil},
	{"testdata/stmt0.src", nil},
	{"testdata/stmt1.src", nil},
	{"testdata/gotos.src", nil},
	{"testdata/labels.src", nil},
	{"testdata/issues.src", nil},
	{"testdata/blank.src", nil},
}

func notGo1_4() bool {
	return !strings.HasPrefix(runtime.Version(), "go1.4")
}

var fset = token.NewFileSet()

// Positioned errors are of the form filename:line:column: message .
var posMsgRx = regexp.MustCompile(`^(.*:[0-9]+:[0-9]+): *(.*)`)

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

func parseFiles(t *testing.T, filenames string) ([]*ast.File, []error) {
	var files []*ast.File
	var errlist []error
	for _, filename := range strings.Split(filenames, " ") {
		file, err := parser.ParseFile(fset, filename, nil, parser.AllErrors)
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
func errMap(t *testing.T, testname string, files []*ast.File) map[string][]string {
	// map of position strings to lists of error message patterns
	errmap := make(map[string][]string)

	for _, file := range files {
		filename := fset.Position(file.Package).Filename
		src, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Fatalf("%s: could not read %s", testname, filename)
		}

		var s scanner.Scanner
		s.Init(fset.AddFile(filename, -1, len(src)), src, nil, scanner.ScanComments)
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

func checkFiles(t *testing.T, filenames string) {
	// parse files and collect parser errors
	files, errlist := parseFiles(t, filenames)

	pkgName := "<no package>"
	if len(files) > 0 {
		pkgName = files[0].Name.Name
	}

	if *listErrors && len(errlist) > 0 {
		t.Errorf("--- %s:", pkgName)
		for _, err := range errlist {
			t.Error(err)
		}
	}

	// typecheck and collect typechecker errors
	var conf Config
	conf.Error = func(err error) {
		if *listErrors {
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

	if *listErrors {
		return
	}

	// match and eliminate errors;
	// we are expecting the following errors
	errmap := errMap(t, pkgName, files)
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

func TestCheck(t *testing.T) {
	skipSpecialPlatforms(t)

	// Declare builtins for testing.
	DefPredeclaredTestFuncs()

	// If explicit test files are specified, only check those.
	if *testFiles != "" {
		checkFiles(t, *testFiles)
		return
	}

	// Otherwise, run all the tests.
	for _, test := range tests {
		if test.cond == nil || test.cond() {
			checkFiles(t, test.files)
		}
	}
}
