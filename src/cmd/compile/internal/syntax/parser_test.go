// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"cmd/internal/src"
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

	ast2, err := ParseBytes(src.NewFileBase(filename, filename), buf1.Bytes(), nil, nil, 0)
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
	_, err := ParseBytes(nil, nil, nil, nil, 0) // return with parser error, don't panic
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

func TestLineDirectives(t *testing.T) {
	for _, test := range []struct {
		src, msg  string
		filename  string
		line, col uint // 0-based
	}{
		// test validity of //line directive
		{`//line :`, "invalid line number: ", "", 0, 8},
		{`//line :x`, "invalid line number: x", "", 0, 8},
		{`//line foo :`, "invalid line number: ", "", 0, 12},
		{`//line foo:123abc`, "invalid line number: 123abc", "", 0, 11},
		{`/**///line foo:x`, "syntax error: package statement must be first", "", 0, 16}, //line directive not at start of line - ignored
		{`//line foo:0`, "invalid line number: 0", "", 0, 11},
		{fmt.Sprintf(`//line foo:%d`, lineMax+1), fmt.Sprintf("invalid line number: %d", lineMax+1), "", 0, 11},

		// test effect of //line directive on (relative) position information
		{"//line foo:123\n   foo", "syntax error: package statement must be first", "foo", 123 - linebase, 3},
		{"//line foo:123\n//line bar:345\nfoo", "syntax error: package statement must be first", "bar", 345 - linebase, 0},
	} {
		_, err := ParseBytes(nil, []byte(test.src), nil, nil, 0)
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
		if filename := perr.Pos.RelFilename(); filename != test.filename {
			t.Errorf("%s: got filename = %q; want %q", test.src, filename, test.filename)
		}
		if line := perr.Pos.RelLine(); line != test.line+linebase {
			t.Errorf("%s: got line = %d; want %d", test.src, line, test.line+linebase)
		}
		if col := perr.Pos.Col(); col != test.col+colbase {
			t.Errorf("%s: got col = %d; want %d", test.src, col, test.col+colbase)
		}
	}
}
