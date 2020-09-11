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
	"fmt"
	"go/scanner"
	"go/token"
	"internal/testenv"
	"io/ioutil"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	. "cmd/compile/internal/types2"
)

var (
	haltOnError = flag.Bool("halt", false, "halt on error")
	listErrors  = flag.Bool("errlist", false, "list errors")
	testFiles   = flag.String("files", "", "space-separated list of test files")
)

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

func parseFiles(t *testing.T, filenames []string) ([]*syntax.File, []error) {
	var files []*syntax.File
	var errlist []error
	errh := func(err error) { errlist = append(errlist, err) }
	for _, filename := range filenames {
		file, err := syntax.ParseFile(filename, errh, nil, 0)
		if file == nil {
			t.Fatalf("%s: %s", filename, err)
		}
		files = append(files, file)
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
func errMap(t *testing.T, testname string, files []*syntax.File) map[string][]string {
	// map of position strings to lists of error message patterns
	errmap := make(map[string][]string)

	for _, file := range files {
		filename := file.Pos().RelFilename() // TODO(gri) do we need Filename here?
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

func checkFiles(t *testing.T, sources []string, trace bool) {
	// parse files and collect parser errors
	files, errlist := parseFiles(t, sources)

	pkgName := "<no package>"
	if len(files) > 0 {
		pkgName = files[0].PkgName.Value
	}

	if *listErrors && len(errlist) > 0 {
		t.Errorf("--- %s:", pkgName)
		for _, err := range errlist {
			t.Error(err)
		}
	}

	// typecheck and collect typechecker errors
	var conf Config
	conf.AcceptMethodTypeParams = true
	conf.InferFromConstraints = true
	// special case for importC.src
	if len(sources) == 1 && strings.HasSuffix(sources[0], "importC.src") {
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
		// Ignore secondary error messages starting with "\t";
		// they are clarifying messages for a primary error.
		if !strings.Contains(err.Error(), ": \t") {
			errlist = append(errlist, err)
		}
	}
	conf.Check(pkgName, files, nil)

	if *listErrors {
		return
	}

	// match and eliminate errors;
	// we are expecting the following errors
	errmap := errMap(t, pkgName, files)

	// print map entries (keep for debugging)
	if false {
		for key, entries := range errmap {
			fmt.Printf("%s:\n", key)
			for _, e := range entries {
				fmt.Printf("\t%s\n", e)
			}
		}
	}

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

// TestCheck is for manual testing of selected input files, provided with -files.
func TestCheck(t *testing.T) {
	if *testFiles == "" {
		return
	}
	testenv.MustHaveGoBuild(t)
	DefPredeclaredTestFuncs()
	checkFiles(t, strings.Split(*testFiles, " "), testing.Verbose())
}

func TestTestdata(t *testing.T) {
	t.Skip("positions need to be fixed first")
	DefPredeclaredTestFuncs()
	testDir(t, "testdata")
}
func TestExamples(t *testing.T)  { testDir(t, "examples") }
func TestFixedbugs(t *testing.T) { testDir(t, "fixedbugs") }

func testDir(t *testing.T, dir string) {
	testenv.MustHaveGoBuild(t)

	fis, err := ioutil.ReadDir(dir)
	if err != nil {
		t.Error(err)
		return
	}

	for count, fi := range fis {
		path := filepath.Join(dir, fi.Name())

		// if fi is a directory, its files make up a single package
		if fi.IsDir() {
			if testing.Verbose() {
				fmt.Printf("%3d %s\n", count, path)
			}
			fis, err := ioutil.ReadDir(path)
			if err != nil {
				t.Error(err)
				continue
			}
			files := make([]string, len(fis))
			for i, fi := range fis {
				// if fi is a directory, checkFiles below will complain
				files[i] = filepath.Join(path, fi.Name())
				if testing.Verbose() {
					fmt.Printf("\t%s\n", files[i])
				}
			}
			checkFiles(t, files, false)
			continue
		}

		// otherwise, fi is a stand-alone file
		if testing.Verbose() {
			fmt.Printf("%3d %s\n", count, path)
		}
		checkFiles(t, []string{path}, false)
	}
}
