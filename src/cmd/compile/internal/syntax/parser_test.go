// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

var fast = flag.Bool("fast", false, "parse package files in parallel")
var src_ = flag.String("src", "parser.go", "source file to parse")
var verify = flag.Bool("verify", false, "verify idempotent printing")

func TestParse(t *testing.T) {
	ParseFile(*src_, func(err error) { t.Error(err) }, nil, 0)
}

func TestStdLib(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	start := time.Now()

	type parseResult struct {
		filename string
		lines    uint
	}

	results := make(chan parseResult)
	go func() {
		defer close(results)
		for _, dir := range []string{
			runtime.GOROOT(),
		} {
			walkDirs(t, dir, func(filename string) {
				if debug {
					fmt.Printf("parsing %s\n", filename)
				}
				ast, err := ParseFile(filename, nil, nil, 0)
				if err != nil {
					t.Error(err)
					return
				}
				if *verify {
					verifyPrint(filename, ast)
				}
				results <- parseResult{filename, ast.Lines}
			})
		}
	}()

	var count, lines uint
	for res := range results {
		count++
		lines += res.lines
		if testing.Verbose() {
			fmt.Printf("%5d  %s (%d lines)\n", count, res.filename, res.lines)
		}
	}

	dt := time.Since(start)
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	dm := float64(m2.TotalAlloc-m1.TotalAlloc) / 1e6

	fmt.Printf("parsed %d lines (%d files) in %v (%d lines/s)\n", lines, count, dt, int64(float64(lines)/dt.Seconds()))
	fmt.Printf("allocated %.3fMb (%.3fMb/s)\n", dm, dm/dt.Seconds())
}

func walkDirs(t *testing.T, dir string, action func(string)) {
	fis, err := ioutil.ReadDir(dir)
	if err != nil {
		t.Error(err)
		return
	}

	var files, dirs []string
	for _, fi := range fis {
		if fi.Mode().IsRegular() {
			if strings.HasSuffix(fi.Name(), ".go") {
				path := filepath.Join(dir, fi.Name())
				files = append(files, path)
			}
		} else if fi.IsDir() && fi.Name() != "testdata" {
			path := filepath.Join(dir, fi.Name())
			if !strings.HasSuffix(path, "/test") {
				dirs = append(dirs, path)
			}
		}
	}

	if *fast {
		var wg sync.WaitGroup
		wg.Add(len(files))
		for _, filename := range files {
			go func(filename string) {
				defer wg.Done()
				action(filename)
			}(filename)
		}
		wg.Wait()
	} else {
		for _, filename := range files {
			action(filename)
		}
	}

	for _, dir := range dirs {
		walkDirs(t, dir, action)
	}
}

func verifyPrint(filename string, ast1 *File) {
	var buf1 bytes.Buffer
	_, err := Fprint(&buf1, ast1, true)
	if err != nil {
		panic(err)
	}

	ast2, err := Parse(NewFileBase(filename), &buf1, nil, nil, 0)
	if err != nil {
		panic(err)
	}

	var buf2 bytes.Buffer
	_, err = Fprint(&buf2, ast2, true)
	if err != nil {
		panic(err)
	}

	if bytes.Compare(buf1.Bytes(), buf2.Bytes()) != 0 {
		fmt.Printf("--- %s ---\n", filename)
		fmt.Printf("%s\n", buf1.Bytes())
		fmt.Println()

		fmt.Printf("--- %s ---\n", filename)
		fmt.Printf("%s\n", buf2.Bytes())
		fmt.Println()
		panic("not equal")
	}
}

func TestIssue17697(t *testing.T) {
	_, err := Parse(nil, bytes.NewReader(nil), nil, nil, 0) // return with parser error, don't panic
	if err == nil {
		t.Errorf("no error reported")
	}
}

func TestParseFile(t *testing.T) {
	_, err := ParseFile("", nil, nil, 0)
	if err == nil {
		t.Error("missing io error")
	}

	var first error
	_, err = ParseFile("", func(err error) {
		if first == nil {
			first = err
		}
	}, nil, 0)
	if err == nil || first == nil {
		t.Error("missing io error")
	}
	if err != first {
		t.Errorf("got %v; want first error %v", err, first)
	}
}

// Make sure (PosMax + 1) doesn't overflow when converted to default
// type int (when passed as argument to fmt.Sprintf) on 32bit platforms
// (see test cases below).
var tooLarge int = PosMax + 1

func TestLineDirectives(t *testing.T) {
	// valid line directives lead to a syntax error after them
	const valid = "syntax error: package statement must be first"

	for _, test := range []struct {
		src, msg  string
		filename  string
		line, col uint // 0-based
	}{
		// ignored //line directives
		{"//\n", valid, "", 1, 0},            // no directive
		{"//line\n", valid, "", 1, 0},        // missing colon
		{"//line foo\n", valid, "", 1, 0},    // missing colon
		{"  //line foo:\n", valid, "", 1, 0}, // not a line start
		{"//  line foo:\n", valid, "", 1, 0}, // space between // and line

		// invalid //line directives with one colon
		{"//line :\n", "invalid line number: ", "", 0, 8},
		{"//line :x\n", "invalid line number: x", "", 0, 8},
		{"//line foo :\n", "invalid line number: ", "", 0, 12},
		{"//line foo:x\n", "invalid line number: x", "", 0, 11},
		{"//line foo:0\n", "invalid line number: 0", "", 0, 11},
		{"//line foo:1 \n", "invalid line number: 1 ", "", 0, 11},
		{"//line foo:-12\n", "invalid line number: -12", "", 0, 11},
		{"//line C:foo:0\n", "invalid line number: 0", "", 0, 13},
		{fmt.Sprintf("//line foo:%d\n", tooLarge), fmt.Sprintf("invalid line number: %d", tooLarge), "", 0, 11},

		// invalid //line directives with two colons
		{"//line ::\n", "invalid line number: ", "", 0, 9},
		{"//line ::x\n", "invalid line number: x", "", 0, 9},
		{"//line foo::123abc\n", "invalid line number: 123abc", "", 0, 12},
		{"//line foo::0\n", "invalid line number: 0", "", 0, 12},
		{"//line foo:0:1\n", "invalid line number: 0", "", 0, 11},

		{"//line :123:0\n", "invalid column number: 0", "", 0, 12},
		{"//line foo:123:0\n", "invalid column number: 0", "", 0, 15},
		{fmt.Sprintf("//line foo:10:%d\n", tooLarge), fmt.Sprintf("invalid column number: %d", tooLarge), "", 0, 14},

		// effect of valid //line directives on lines
		{"//line foo:123\n   foo", valid, "foo", 123 - linebase, 3},
		{"//line  foo:123\n   foo", valid, " foo", 123 - linebase, 3},
		{"//line foo:123\n//line bar:345\nfoo", valid, "bar", 345 - linebase, 0},
		{"//line C:foo:123\n", valid, "C:foo", 123 - linebase, 0},
		{"//line /src/a/a.go:123\n   foo", valid, "/src/a/a.go", 123 - linebase, 3},
		{"//line :x:1\n", valid, ":x", 1 - linebase, 0},
		{"//line foo ::1\n", valid, "foo :", 1 - linebase, 0},
		{"//line foo:123abc:1\n", valid, "foo:123abc", 0, 0},
		{"//line foo :123:1\n", valid, "foo ", 123 - linebase, 0},
		{"//line ::123\n", valid, ":", 123 - linebase, 0},

		// effect of valid //line directives on columns
		{"//line :x:1:10\n", valid, ":x", 1 - linebase, 10 - colbase},
		{"//line foo ::1:2\n", valid, "foo :", 1 - linebase, 2 - colbase},
		{"//line foo:123abc:1:1000\n", valid, "foo:123abc", 1 - linebase, 1000 - colbase},
		{"//line foo :123:1000\n\n", valid, "foo ", 124 - linebase, 0},
		{"//line ::123:1234\n", valid, ":", 123 - linebase, 1234 - colbase},

		// ignored /*line directives
		{"/**/", valid, "", 0, 4},             // no directive
		{"/*line*/", valid, "", 0, 8},         // missing colon
		{"/*line foo*/", valid, "", 0, 12},    // missing colon
		{"  //line foo:*/", valid, "", 0, 15}, // not a line start
		{"/*  line foo:*/", valid, "", 0, 15}, // space between // and line

		// invalid /*line directives with one colon
		{"/*line :*/", "invalid line number: ", "", 0, 8},
		{"/*line :x*/", "invalid line number: x", "", 0, 8},
		{"/*line foo :*/", "invalid line number: ", "", 0, 12},
		{"/*line foo:x*/", "invalid line number: x", "", 0, 11},
		{"/*line foo:0*/", "invalid line number: 0", "", 0, 11},
		{"/*line foo:1 */", "invalid line number: 1 ", "", 0, 11},
		{"/*line C:foo:0*/", "invalid line number: 0", "", 0, 13},
		{fmt.Sprintf("/*line foo:%d*/", tooLarge), fmt.Sprintf("invalid line number: %d", tooLarge), "", 0, 11},

		// invalid /*line directives with two colons
		{"/*line ::*/", "invalid line number: ", "", 0, 9},
		{"/*line ::x*/", "invalid line number: x", "", 0, 9},
		{"/*line foo::123abc*/", "invalid line number: 123abc", "", 0, 12},
		{"/*line foo::0*/", "invalid line number: 0", "", 0, 12},
		{"/*line foo:0:1*/", "invalid line number: 0", "", 0, 11},

		{"/*line :123:0*/", "invalid column number: 0", "", 0, 12},
		{"/*line foo:123:0*/", "invalid column number: 0", "", 0, 15},
		{fmt.Sprintf("/*line foo:10:%d*/", tooLarge), fmt.Sprintf("invalid column number: %d", tooLarge), "", 0, 14},

		// effect of valid /*line directives on lines
		{"/*line foo:123*/   foo", valid, "foo", 123 - linebase, 3},
		{"/*line foo:123*/\n//line bar:345\nfoo", valid, "bar", 345 - linebase, 0},
		{"/*line C:foo:123*/", valid, "C:foo", 123 - linebase, 0},
		{"/*line /src/a/a.go:123*/   foo", valid, "/src/a/a.go", 123 - linebase, 3},
		{"/*line :x:1*/", valid, ":x", 1 - linebase, 0},
		{"/*line foo ::1*/", valid, "foo :", 1 - linebase, 0},
		{"/*line foo:123abc:1*/", valid, "foo:123abc", 1 - linebase, 0},
		{"/*line foo :123:10*/", valid, "foo ", 123 - linebase, 10 - colbase},
		{"/*line ::123*/", valid, ":", 123 - linebase, 0},

		// effect of valid /*line directives on columns
		{"/*line :x:1:10*/", valid, ":x", 1 - linebase, 10 - colbase},
		{"/*line foo ::1:2*/", valid, "foo :", 1 - linebase, 2 - colbase},
		{"/*line foo:123abc:1:1000*/", valid, "foo:123abc", 1 - linebase, 1000 - colbase},
		{"/*line foo :123:1000*/\n", valid, "foo ", 124 - linebase, 0},
		{"/*line ::123:1234*/", valid, ":", 123 - linebase, 1234 - colbase},
	} {
		_, err := Parse(nil, strings.NewReader(test.src), nil, nil, 0)
		if err == nil {
			t.Errorf("%s: no error reported", test.src)
			continue
		}
		perr, ok := err.(Error)
		if !ok {
			t.Errorf("%s: got %v; want parser error", test.src, err)
			continue
		}
		if msg := perr.Msg; msg != test.msg {
			t.Errorf("%s: got msg = %q; want %q", test.src, msg, test.msg)
		}

		pos := perr.Pos
		if filename := pos.RelFilename(); filename != test.filename {
			t.Errorf("%s: got filename = %q; want %q", test.src, filename, test.filename)
		}
		if line := pos.RelLine(); line != test.line+linebase {
			t.Errorf("%s: got line = %d; want %d", test.src, line, test.line+linebase)
		}
		if col := pos.RelCol(); col != test.col+colbase {
			t.Errorf("%s: got col = %d; want %d", test.src, col, test.col+colbase)
		}
	}
}
