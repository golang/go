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

// The test filenames do not end in .go so that they are invisible
// to gofmt since they contain comments that must not change their
// positions relative to surrounding tokens.

var tests = []struct {
	name  string
	files []string
}{
	{"test0", []string{"testdata/test0.src"}},
}

var fset = token.NewFileSet()

// TODO(gri) This functionality should be in token.Fileset.
func getFile(filename string) *token.File {
	for f := range fset.Files() {
		if f.Name() == filename {
			return f
		}
	}
	return nil
}

// TODO(gri) This functionality should be in token.Fileset.
func getPos(filename string, offset int) token.Pos {
	if f := getFile(filename); f != nil {
		return f.Pos(offset)
	}
	return token.NoPos
}

// TODO(gri) Need to revisit parser interface. We should be able to use parser.ParseFiles
//           or a similar function instead.
func parseFiles(t *testing.T, testname string, filenames []string) (map[string]*ast.File, error) {
	files := make(map[string]*ast.File)
	var errors scanner.ErrorList
	for _, filename := range filenames {
		if _, exists := files[filename]; exists {
			t.Fatalf("%s: duplicate file %s", testname, filename)
		}
		file, err := parser.ParseFile(fset, filename, nil, parser.DeclarationErrors)
		if file == nil {
			t.Fatalf("%s: could not parse file %s", testname, filename)
		}
		files[filename] = file
		if err != nil {
			// if the parser returns a non-scanner.ErrorList error
			// the file couldn't be read in the first place and
			// file == nil; in that case we shouldn't reach here
			errors = append(errors, err.(scanner.ErrorList)...)
		}

	}
	return files, errors
}

// ERROR comments must be of the form /* ERROR "rx" */ and rx is
// a regular expression that matches the expected error message.
//
var errRx = regexp.MustCompile(`^/\* *ERROR *"([^"]*)" *\*/$`)

// expectedErrors collects the regular expressions of ERROR comments found
// in files and returns them as a map of error positions to error messages.
//
func expectedErrors(t *testing.T, testname string, files map[string]*ast.File) map[token.Pos]string {
	errors := make(map[token.Pos]string)
	for filename := range files {
		src, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Fatalf("%s: could not read %s", testname, filename)
		}

		var s scanner.Scanner
		// file was parsed already - do not add it again to the file
		// set otherwise the position information returned here will
		// not match the position information collected by the parser
		s.Init(getFile(filename), src, nil, scanner.ScanComments)
		var prev token.Pos // position of last non-comment, non-semicolon token

	scanFile:
		for {
			pos, tok, lit := s.Scan()
			switch tok {
			case token.EOF:
				break scanFile
			case token.COMMENT:
				s := errRx.FindStringSubmatch(lit)
				if len(s) == 2 {
					errors[prev] = string(s[1])
				}
			case token.SEMICOLON:
				// ignore automatically inserted semicolon
				if lit == "\n" {
					break
				}
				fallthrough
			default:
				prev = pos
			}
		}
	}
	return errors
}

func eliminate(t *testing.T, expected map[token.Pos]string, errors error) {
	if errors == nil {
		return
	}
	for _, error := range errors.(scanner.ErrorList) {
		// error.Pos is a token.Position, but we want
		// a token.Pos so we can do a map lookup
		// TODO(gri) Need to move scanner.Errors over
		//           to use token.Pos and file set info.
		pos := getPos(error.Pos.Filename, error.Pos.Offset)
		if msg, found := expected[pos]; found {
			// we expect a message at pos; check if it matches
			rx, err := regexp.Compile(msg)
			if err != nil {
				t.Errorf("%s: %v", error.Pos, err)
				continue
			}
			if match := rx.MatchString(error.Msg); !match {
				t.Errorf("%s: %q does not match %q", error.Pos, error.Msg, msg)
				continue
			}
			// we have a match - eliminate this error
			delete(expected, pos)
		} else {
			// To keep in mind when analyzing failed test output:
			// If the same error position occurs multiple times in errors,
			// this message will be triggered (because the first error at
			// the position removes this position from the expected errors).
			t.Errorf("%s: no (multiple?) error expected, but found: %s", error.Pos, error.Msg)
		}
	}
}

func check(t *testing.T, testname string, testfiles []string) {
	// TODO(gri) Eventually all these different phases should be
	//           subsumed into a single function call that takes
	//           a set of files and creates a fully resolved and
	//           type-checked AST.

	files, err := parseFiles(t, testname, testfiles)

	// we are expecting the following errors
	// (collect these after parsing the files so that
	// they are found in the file set)
	errors := expectedErrors(t, testname, files)

	// verify errors returned by the parser
	eliminate(t, errors, err)

	// verify errors returned after resolving identifiers
	pkg, err := ast.NewPackage(fset, files, GcImporter, Universe)
	eliminate(t, errors, err)

	// verify errors returned by the typechecker
	_, err = Check(fset, pkg)
	eliminate(t, errors, err)

	// there should be no expected errors left
	if len(errors) > 0 {
		t.Errorf("%s: %d errors not reported:", testname, len(errors))
		for pos, msg := range errors {
			t.Errorf("%s: %s\n", fset.Position(pos), msg)
		}
	}
}

func TestCheck(t *testing.T) {
	// For easy debugging w/o changing the testing code,
	// if there is a local test file, only test that file.
	const testfile = "test.go"
	if fi, err := os.Stat(testfile); err == nil && !fi.IsDir() {
		fmt.Printf("WARNING: Testing only %s (remove it to run all tests)\n", testfile)
		check(t, testfile, []string{testfile})
		return
	}

	// Otherwise, run all the tests.
	for _, test := range tests {
		check(t, test.name, test.files)
	}
}
