// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Command bake takes a list of file names and writes a Go source file to
// standard output that declares a map of string constants containing the input files.
//
// For example, the command
// 	bake foo.html bar.txt
// produces a source file in package main that declares the variable bakedFiles
// that is a map with keys "foo.html" and "bar.txt" that contain the contents
// of foo.html and bar.txt.
package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"unicode/utf8"
)

func main() {
	if err := bake(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func bake(files []string) error {
	w := bufio.NewWriter(os.Stdout)
	fmt.Fprintf(w, "%v\n\npackage static\n\n", warning)
	fmt.Fprintf(w, "var Files = map[string]string{\n")
	for _, fn := range files {
		b, err := ioutil.ReadFile(fn)
		if err != nil {
			return err
		}
		if !utf8.Valid(b) {
			return fmt.Errorf("file %s is not valid UTF-8", fn)
		}
		fmt.Fprintf(w, "\t%q: `%s`,\n", filepath.Base(fn), sanitize(b))
	}
	fmt.Fprintln(w, "}")
	return w.Flush()
}

// sanitize prepares a string as a raw string constant.
func sanitize(b []byte) []byte {
	// Replace ` with `+"`"+`
	b = bytes.Replace(b, []byte("`"), []byte("`+\"`\"+`"), -1)

	// Replace BOM with `+"\xEF\xBB\xBF"+`
	// (A BOM is valid UTF-8 but not permitted in Go source files.
	// I wouldn't bother handling this, but for some insane reason
	// jquery.js has a BOM somewhere in the middle.)
	return bytes.Replace(b, []byte("\xEF\xBB\xBF"), []byte("`+\"\\xEF\\xBB\\xBF\"+`"), -1)
}

const warning = "// DO NOT EDIT ** This file was generated with the bake tool ** DO NOT EDIT //"
