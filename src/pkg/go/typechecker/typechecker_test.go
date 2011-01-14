// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a simple typechecker test harness. Packages found
// in the testDir directory are typechecked. Error messages reported by
// the typechecker are compared against the error messages expected for
// the test files.
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
//	package P0
//	func f() {
//		_ = x /* ERROR "not declared" */ + 1
//	}
// 
// If the -pkg flag is set, only packages with package names matching
// the regular expression provided via the flag value are tested.

package typechecker

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
	"sort"
	"strings"
	"testing"
)


const testDir = "./testdata" // location of test packages

var fset = token.NewFileSet()

var (
	pkgPat = flag.String("pkg", ".*", "regular expression to select test packages by package name")
	trace  = flag.Bool("trace", false, "print package names")
)


// ERROR comments must be of the form /* ERROR "rx" */ and rx is
// a regular expression that matches the expected error message.
var errRx = regexp.MustCompile(`^/\* *ERROR *"([^"]*)" *\*/$`)

// expectedErrors collects the regular expressions of ERROR comments
// found in the package files of pkg and returns them in sorted order
// (by filename and position).
func expectedErrors(t *testing.T, pkg *ast.Package) (list scanner.ErrorList) {
	// scan all package files
	for filename := range pkg.Files {
		src, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Fatalf("expectedErrors(%s): %v", pkg.Name, err)
		}

		var s scanner.Scanner
		file := fset.AddFile(filename, fset.Base(), len(src))
		s.Init(file, src, nil, scanner.ScanComments)
		var prev token.Pos // position of last non-comment token
	loop:
		for {
			pos, tok, lit := s.Scan()
			switch tok {
			case token.EOF:
				break loop
			case token.COMMENT:
				s := errRx.FindSubmatch(lit)
				if len(s) == 2 {
					list = append(list, &scanner.Error{fset.Position(prev), string(s[1])})
				}
			default:
				prev = pos
			}
		}
	}
	sort.Sort(list) // multiple files may not be sorted
	return
}


func testFilter(f *os.FileInfo) bool {
	return strings.HasSuffix(f.Name, ".go") && f.Name[0] != '.'
}


func checkError(t *testing.T, expected, found *scanner.Error) {
	rx, err := regexp.Compile(expected.Msg)
	if err != nil {
		t.Errorf("%s: %v", expected.Pos, err)
		return
	}

	match := rx.MatchString(found.Msg)

	if expected.Pos.Offset != found.Pos.Offset {
		if match {
			t.Errorf("%s: expected error should have been at %s", expected.Pos, found.Pos)
		} else {
			t.Errorf("%s: error matching %q expected", expected.Pos, expected.Msg)
			return
		}
	}

	if !match {
		t.Errorf("%s: %q does not match %q", expected.Pos, expected.Msg, found.Msg)
	}
}


func TestTypeCheck(t *testing.T) {
	flag.Parse()
	pkgRx, err := regexp.Compile(*pkgPat)
	if err != nil {
		t.Fatalf("illegal flag value %q: %s", *pkgPat, err)
	}

	pkgs, err := parser.ParseDir(fset, testDir, testFilter, 0)
	if err != nil {
		scanner.PrintError(os.Stderr, err)
		t.Fatalf("packages in %s contain syntax errors", testDir)
	}

	for _, pkg := range pkgs {
		if !pkgRx.MatchString(pkg.Name) {
			continue // only test selected packages
		}

		if *trace {
			fmt.Println(pkg.Name)
		}

		xlist := expectedErrors(t, pkg)
		err := CheckPackage(fset, pkg, nil)
		if err != nil {
			if elist, ok := err.(scanner.ErrorList); ok {
				// verify that errors match
				for i := 0; i < len(xlist) && i < len(elist); i++ {
					checkError(t, xlist[i], elist[i])
				}
				// the correct number or errors must have been found
				if len(xlist) != len(elist) {
					fmt.Fprintf(os.Stderr, "%s\n", pkg.Name)
					scanner.PrintError(os.Stderr, elist)
					fmt.Fprintln(os.Stderr)
					t.Errorf("TypeCheck(%s): %d errors expected but %d reported", pkg.Name, len(xlist), len(elist))
				}
			} else {
				t.Errorf("TypeCheck(%s): %v", pkg.Name, err)
			}
		} else if len(xlist) > 0 {
			t.Errorf("TypeCheck(%s): %d errors expected but 0 reported", pkg.Name, len(xlist))
		}
	}
}
