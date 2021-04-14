// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that batch files are maintained as CRLF files (consistent
// behavior on all operating systems). See golang.org/issue/37791.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func main() {
	// Ensure that the GOROOT/src/all.bat file exists and has strict CRLF line endings.
	enforceBatchStrictCRLF(filepath.Join(runtime.GOROOT(), "src", "all.bat"))

	// Walk the entire Go repository source tree (without GOROOT/pkg),
	// skipping directories that start with "." and named "testdata",
	// and ensure all .bat files found have exact CRLF line endings.
	err := filepath.Walk(runtime.GOROOT(), func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if fi.IsDir() && (strings.HasPrefix(fi.Name(), ".") || fi.Name() == "testdata") {
			return filepath.SkipDir
		}
		if path == filepath.Join(runtime.GOROOT(), "pkg") {
			// GOROOT/pkg is known to contain generated artifacts, not source code.
			// Skip it to avoid false positives. (Also see golang.org/issue/37929.)
			return filepath.SkipDir
		}
		if filepath.Ext(fi.Name()) == ".bat" {
			enforceBatchStrictCRLF(path)
		}
		return nil
	})
	if err != nil {
		log.Fatalln(err)
	}
}

func enforceBatchStrictCRLF(path string) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatalln(err)
	}
	cr, lf := bytes.Count(b, []byte{13}), bytes.Count(b, []byte{10})
	crlf := bytes.Count(b, []byte{13, 10})
	if cr != crlf || lf != crlf {
		if rel, err := filepath.Rel(runtime.GOROOT(), path); err == nil {
			// Make the test failure more readable by showing a path relative to GOROOT.
			path = rel
		}
		fmt.Printf("Windows batch file %s does not use strict CRLF line termination.\n", path)
		fmt.Printf("Please convert it to CRLF before checking it in due to golang.org/issue/37791.\n")
		os.Exit(1)
	}
}
