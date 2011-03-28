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


var exitCode = 0


func usage() {
	fmt.Fprintf(os.Stderr, "usage: gotype [flags] [path ...]\n")
	flag.PrintDefaults()
	os.Exit(2)
}


func report(err os.Error) {
	scanner.PrintError(os.Stderr, err)
	exitCode = 2
}


// parse returns the AST for the Go source src.
// The filename is for error reporting only.
// The result is nil if there were errors or if
// the file does not belong to the -p package.
func parse(fset *token.FileSet, filename string, src []byte) *ast.File {
	if *verbose {
		fmt.Println(filename)
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
	mode := parser.DeclarationErrors
	if *printTrace {
		mode |= parser.Trace
	}
	file, err := parser.ParseFile(fset, filename, src, mode)
	if err != nil {
		report(err)
		return nil
	}
	if *printAST {
		ast.Print(fset, file)
	}

	return file
}


func parseStdin(fset *token.FileSet) (files map[string]*ast.File) {
	files = make(map[string]*ast.File)
	src, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		report(err)
		return
	}
	const filename = "<standard input>"
	if file := parse(fset, filename, src); file != nil {
		files[filename] = file
	}
	return
}


func parseFiles(fset *token.FileSet, filenames []string) (files map[string]*ast.File) {
	files = make(map[string]*ast.File)
	for _, filename := range filenames {
		src, err := ioutil.ReadFile(filename)
		if err != nil {
			report(err)
			continue
		}
		if file := parse(fset, filename, src); file != nil {
			if files[filename] != nil {
				report(os.ErrorString(fmt.Sprintf("%q: duplicate file", filename)))
				continue
			}
			files[filename] = file
		}
	}
	return
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
	fset := token.NewFileSet()
	processPackage(fset, parseFiles(fset, filenames[0:i]))
}


// TODO(gri) Replace this with a fully functioning importer.
//           For now a dummy importer is set up by gotype_test.go.
var importer ast.Importer

func processPackage(fset *token.FileSet, files map[string]*ast.File) {
	// make a package (resolve all identifiers)
	pkg, err := ast.NewPackage(fset, files, importer, universe)
	if err != nil {
		report(err)
		return
	}
	// TODO(gri): typecheck package
	_ = pkg
}


func main() {
	flag.Usage = usage
	flag.Parse()

	if flag.NArg() == 0 {
		fset := token.NewFileSet()
		processPackage(fset, parseStdin(fset))
	} else {
		processFiles(flag.Args(), true)
	}

	os.Exit(exitCode)
}


// TODO(gri) Move universe and its initialization in to the right package.
var universe *ast.Scope

func define(kind ast.ObjKind, names ...string) {
	for _, name := range names {
		obj := ast.NewObj(kind, name)
		if universe.Insert(obj) != nil {
			panic("gotype internal error: incorrect universe scope")
		}
	}
}


func init() {
	universe = ast.NewScope(nil)

	define(ast.Typ,
		"bool",
		"byte",
		"complex64",
		"complex128",
		"float32",
		"float64",
		"int8",
		"int16",
		"int32",
		"int64",
		"string",
		"uint8",
		"uint16",
		"uint32",
		"uint64",
		"int",
		"uint",
		"uintptr",
	)

	define(ast.Con,
		"true",
		"false",
		"iota",
		"nil",
	)

	define(ast.Fun,
		"append",
		"cap",
		"close",
		"complex",
		"copy",
		"imag",
		"len",
		"make",
		"new",
		"panic",
		"print",
		"println",
		"real",
		"recover",
	)
}
