// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
The headscan command extracts comment headings from package files;
it is used to detect false positives which may require an adjustment
to the comment formatting heuristics in comment.go.

Usage: headscan [-root root_directory]

By default, the $GOROOT/src directory is scanned.
*/
package main

import (
	"flag"
	"fmt"
	"go/doc"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

var (
	root    = flag.String("root", filepath.Join(runtime.GOROOT(), "src"), "root of filesystem tree to scan")
	verbose = flag.Bool("v", false, "verbose mode")
)

// ToHTML in comment.go assigns a (possibly blank) ID to each heading
var html_h = regexp.MustCompile(`<h3 id="[^"]*">`)

const html_endh = "</h3>\n"

func isGoFile(fi fs.FileInfo) bool {
	return strings.HasSuffix(fi.Name(), ".go") &&
		!strings.HasSuffix(fi.Name(), "_test.go")
}

func appendHeadings(list []string, comment string) []string {
	var buf strings.Builder
	doc.ToHTML(&buf, comment, nil)
	for s := buf.String(); s != ""; {
		loc := html_h.FindStringIndex(s)
		if len(loc) == 0 {
			break
		}
		var inner string
		inner, s, _ = strings.Cut(s[loc[1]:], html_endh)
		list = append(list, inner)
	}
	return list
}

func main() {
	flag.Parse()
	fset := token.NewFileSet()
	nheadings := 0
	err := filepath.WalkDir(*root, func(path string, info fs.DirEntry, err error) error {
		if !info.IsDir() {
			return nil
		}
		pkgs, err := parser.ParseDir(fset, path, isGoFile, parser.ParseComments)
		if err != nil {
			if *verbose {
				fmt.Fprintln(os.Stderr, err)
			}
			return nil
		}
		for _, pkg := range pkgs {
			d := doc.New(pkg, path, doc.Mode(0))
			list := appendHeadings(nil, d.Doc)
			for _, d := range d.Consts {
				list = appendHeadings(list, d.Doc)
			}
			for _, d := range d.Types {
				list = appendHeadings(list, d.Doc)
			}
			for _, d := range d.Vars {
				list = appendHeadings(list, d.Doc)
			}
			for _, d := range d.Funcs {
				list = appendHeadings(list, d.Doc)
			}
			if len(list) > 0 {
				// directories may contain multiple packages;
				// print path and package name
				fmt.Printf("%s (package %s)\n", path, pkg.Name)
				for _, h := range list {
					fmt.Printf("\t%s\n", h)
				}
				nheadings += len(list)
			}
		}
		return nil
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(nheadings, "headings found")
}
