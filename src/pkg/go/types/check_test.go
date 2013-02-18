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

package types

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"os"
	"regexp"
	"testing"
)

var listErrors = flag.Bool("list", false, "list errors")

// The test filenames do not end in .go so that they are invisible
// to gofmt since they contain comments that must not change their
// positions relative to surrounding tokens.

var tests = []struct {
	name  string
	files []string
}{
	{"decls0", []string{"testdata/decls0.src"}},
	{"decls1", []string{"testdata/decls1.src"}},
	{"decls2", []string{"testdata/decls2a.src", "testdata/decls2b.src"}},
	{"decls3", []string{"testdata/decls3.src"}},
	{"const0", []string{"testdata/const0.src"}},
	{"expr0", []string{"testdata/expr0.src"}},
	{"expr1", []string{"testdata/expr1.src"}},
	{"expr2", []string{"testdata/expr2.src"}},
	{"expr3", []string{"testdata/expr3.src"}},
	{"builtins", []string{"testdata/builtins.src"}},
	{"conversions", []string{"testdata/conversions.src"}},
	{"stmt0", []string{"testdata/stmt0.src"}},
}

var fset = token.NewFileSet()

func getFile(filename string) (file *token.File) {
	fset.Iterate(func(f *token.File) bool {
		if f.Name() == filename {
			file = f
			return false // end iteration
		}
		return true
	})
	return file
}

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

func parseFiles(t *testing.T, testname string, filenames []string) ([]*ast.File, []error) {
	var files []*ast.File
	var errlist []error
	for _, filename := range filenames {
		file, err := parser.ParseFile(fset, filename, nil, parser.DeclarationErrors|parser.AllErrors)
		if file == nil {
			t.Fatalf("%s: could not parse file %s", testname, filename)
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

// ERROR comments must be of the form /* ERROR "rx" */ and rx is
// a regular expression that matches the expected error message.
//
var errRx = regexp.MustCompile(`^/\* *ERROR *"([^"]*)" *\*/$`)

// errMap collects the regular expressions of ERROR comments found
// in files and returns them as a map of error positions to error messages.
//
func errMap(t *testing.T, testname string, files []*ast.File) map[string][]string {
	errmap := make(map[string][]string)

	for _, file := range files {
		filename := fset.Position(file.Package).Filename
		src, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Fatalf("%s: could not read %s", testname, filename)
		}

		var s scanner.Scanner
		// file was parsed already - do not add it again to the file
		// set otherwise the position information returned here will
		// not match the position information collected by the parser
		s.Init(getFile(filename), src, nil, scanner.ScanComments)
		var prev string // position string of last non-comment, non-semicolon token

	scanFile:
		for {
			pos, tok, lit := s.Scan()
			switch tok {
			case token.EOF:
				break scanFile
			case token.COMMENT:
				s := errRx.FindStringSubmatch(lit)
				if len(s) == 2 {
					errmap[prev] = append(errmap[prev], string(s[1]))
				}
			case token.SEMICOLON:
				// ignore automatically inserted semicolon
				if lit == "\n" {
					continue scanFile
				}
				fallthrough
			default:
				prev = fset.Position(pos).String()
			}
		}
	}

	return errmap
}

func eliminate(t *testing.T, errmap map[string][]string, errlist []error) {
	for _, err := range errlist {
		pos, msg := splitError(err)
		list := errmap[pos]
		index := -1 // list index of matching message, if any
		// we expect one of the messages in list to match the error at pos
		for i, msg := range list {
			rx, err := regexp.Compile(msg)
			if err != nil {
				t.Errorf("%s: %v", pos, err)
				continue
			}
			if rx.MatchString(msg) {
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
			t.Errorf("%s: no error expected: %q", pos, msg)
		}

	}
}

func checkFiles(t *testing.T, testname string, testfiles []string) {
	// parse files and collect parser errors
	files, errlist := parseFiles(t, testname, testfiles)

	// typecheck and collect typechecker errors
	ctxt := Default
	ctxt.Error = func(err error) { errlist = append(errlist, err) }
	ctxt.Check(fset, files)

	if *listErrors {
		t.Errorf("--- %s: %d errors found:", testname, len(errlist))
		for _, err := range errlist {
			t.Error(err)
		}
		return
	}

	// match and eliminate errors
	// we are expecting the following errors
	// (collect these after parsing the files so that
	// they are found in the file set)
	errmap := errMap(t, testname, files)
	eliminate(t, errmap, errlist)

	// there should be no expected errors left
	if len(errmap) > 0 {
		t.Errorf("--- %s: %d source positions with expected (but not reported) errors:", testname, len(errmap))
		for pos, list := range errmap {
			for _, rx := range list {
				t.Errorf("%s: %q", pos, rx)
			}
		}
	}
}

var testBuiltinsDeclared = false

func TestCheck(t *testing.T) {
	// Declare builtins for testing.
	// Not done in an init func to avoid an init race with
	// the construction of the Universe var.
	if !testBuiltinsDeclared {
		testBuiltinsDeclared = true
		// Pkg == nil for Universe objects
		def(&Func{Name: "assert", Type: &builtin{_Assert, "assert", 1, false, true}})
		def(&Func{Name: "trace", Type: &builtin{_Trace, "trace", 0, true, true}})
	}

	// For easy debugging w/o changing the testing code,
	// if there is a local test file, only test that file.
	const testfile = "testdata/test.go"
	if fi, err := os.Stat(testfile); err == nil && !fi.IsDir() {
		fmt.Printf("WARNING: Testing only %s (remove it to run all tests)\n", testfile)
		checkFiles(t, testfile, []string{testfile})
		return
	}

	// Otherwise, run all the tests.
	for _, test := range tests {
		checkFiles(t, test.name, test.files)
	}
}
