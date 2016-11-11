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
var src = flag.String("src", "parser.go", "source file to parse")
var verify = flag.Bool("verify", false, "verify idempotent printing")

func TestParse(t *testing.T) {
	_, err := ParseFile(*src, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
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
		lines    int
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

	var count, lines int
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

	ast2, err := ParseBytes(buf1.Bytes(), nil, nil, 0)
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
	_, err := ParseBytes(nil, nil, nil, 0) // return with parser error, don't panic
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
