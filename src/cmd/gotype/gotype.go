// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)


var (
	// main operation modes
	pkgName   = flag.String("p", "", "process only those files in package pkgName")
	recursive = flag.Bool("r", false, "recursively process subdirectories")
	verbose   = flag.Bool("v", false, "verbose mode")

	// debugging support
	printTrace = flag.Bool("trace", false, "print parse trace")
	printAST   = flag.Bool("ast", false, "print AST")
)


var (
	fset       = token.NewFileSet()
	exitCode   = 0
	parserMode = parser.DeclarationErrors
)


func usage() {
	fmt.Fprintf(os.Stderr, "usage: gotype [flags] [path ...]\n")
	flag.PrintDefaults()
	os.Exit(2)
}


func processFlags() {
	flag.Usage = usage
	flag.Parse()
	if *printTrace {
		parserMode |= parser.Trace
	}
}


func report(err os.Error) {
	scanner.PrintError(os.Stderr, err)
	exitCode = 2
}


// parseFile returns the AST for the given file.
// The result
func parseFile(filename string) *ast.File {
	if *verbose {
		fmt.Println(filename)
	}

	// get source
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		report(err)
		return nil
	}

	// ignore files with different package name
	if *pkgName != "" {
		file, err := parser.ParseFile(fset, filename, src, parser.PackageClauseOnly)
		if err != nil {
			report(err)
			return nil
		}
		if file.Name.Name != *pkgName {
			if *verbose {
				fmt.Printf("\tignored (package %s)\n", file.Name.Name)
			}
			return nil
		}
	}

	// parse entire file
	file, err := parser.ParseFile(fset, filename, src, parserMode)
	if err != nil {
		report(err)
		return nil
	}
	if *printAST {
		ast.Print(fset, file)
	}

	return file
}


// BUG(gri): At the moment, only single-file scope analysis is performed.

func processPackage(filenames []string) {
	var files []*ast.File
	pkgName := ""
	for _, filename := range filenames {
		file := parseFile(filename)
		if file == nil {
			continue // ignore file
		}
		// package names must match
		// TODO(gri): this check should be moved into a
		//            function making the package below
		if pkgName == "" {
			// first package file
			pkgName = file.Name.Name
		} else {
			if file.Name.Name != pkgName {
				report(os.NewError(fmt.Sprintf("file %q is in package %q not %q", filename, file.Name.Name, pkgName)))
				continue
			}
		}
		files = append(files, file)
	}

	// TODO(gri): make a ast.Package and analyze it
	_ = files
}


func isGoFilename(filename string) bool {
	// ignore non-Go files
	return !strings.HasPrefix(filename, ".") && strings.HasSuffix(filename, ".go")
}


func processDirectory(dirname string) {
	f, err := os.Open(dirname, os.O_RDONLY, 0)
	if err != nil {
		report(err)
		return
	}
	filenames, err := f.Readdirnames(-1)
	f.Close()
	if err != nil {
		report(err)
		// continue since filenames may not be empty
	}
	for i, filename := range filenames {
		filenames[i] = filepath.Join(dirname, filename)
	}
	processFiles(filenames, false)
}


func processFiles(filenames []string, allFiles bool) {
	i := 0
	for _, filename := range filenames {
		switch info, err := os.Stat(filename); {
		case err != nil:
			report(err)
		case info.IsRegular():
			if allFiles || isGoFilename(info.Name) {
				filenames[i] = filename
				i++
			}
		case info.IsDirectory():
			if allFiles || *recursive {
				processDirectory(filename)
			}
		}
	}
	processPackage(filenames[0:i])
}


func main() {
	processFlags()

	if flag.NArg() == 0 {
		processPackage([]string{os.Stdin.Name()})
	} else {
		processFiles(flag.Args(), true)
	}

	os.Exit(exitCode)
}
