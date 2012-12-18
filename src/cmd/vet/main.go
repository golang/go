// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Vet is a simple checker for static errors in Go source code.
// See doc.go for more information.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

var verbose = flag.Bool("v", false, "verbose")
var exitCode = 0

// Flags to control which checks to perform
var (
	vetAll             = flag.Bool("all", true, "check everything; disabled if any explicit check is requested")
	vetMethods         = flag.Bool("methods", false, "check that canonically named methods are canonically defined")
	vetPrintf          = flag.Bool("printf", false, "check printf-like invocations")
	vetStructTags      = flag.Bool("structtags", false, "check that struct field tags have canonical format")
	vetUntaggedLiteral = flag.Bool("composites", false, "check that composite literals used type-tagged elements")
	vetRangeLoops      = flag.Bool("rangeloops", false, "check that range loop variables are used correctly")
)

// setExit sets the value for os.Exit when it is called, later.  It
// remembers the highest value.
func setExit(err int) {
	if err > exitCode {
		exitCode = err
	}
}

// Usage is a replacement usage function for the flags package.
func Usage() {
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
	flag.PrintDefaults()
	os.Exit(2)
}

// File is a wrapper for the state of a file used in the parser.
// The parse tree walkers are all methods of this type.
type File struct {
	fset *token.FileSet
	file *ast.File
	b    bytes.Buffer // for use by methods
}

func main() {
	flag.Usage = Usage
	flag.Parse()

	// If a check is named explicitly, turn off the 'all' flag.
	if *vetMethods || *vetPrintf || *vetStructTags || *vetUntaggedLiteral || *vetRangeLoops {
		*vetAll = false
	}

	if *printfuncs != "" {
		for _, name := range strings.Split(*printfuncs, ",") {
			if len(name) == 0 {
				flag.Usage()
			}
			skip := 0
			if colon := strings.LastIndex(name, ":"); colon > 0 {
				var err error
				skip, err = strconv.Atoi(name[colon+1:])
				if err != nil {
					errorf(`illegal format for "Func:N" argument %q; %s`, name, err)
				}
				name = name[:colon]
			}
			name = strings.ToLower(name)
			if name[len(name)-1] == 'f' {
				printfList[name] = skip
			} else {
				printList[name] = skip
			}
		}
	}

	if flag.NArg() == 0 {
		doFile("stdin", os.Stdin)
	} else {
		for _, name := range flag.Args() {
			// Is it a directory?
			if fi, err := os.Stat(name); err == nil && fi.IsDir() {
				walkDir(name)
			} else {
				doFile(name, nil)
			}
		}
	}
	os.Exit(exitCode)
}

// doFile analyzes one file.  If the reader is nil, the source code is read from the
// named file.
func doFile(name string, reader io.Reader) {
	fs := token.NewFileSet()
	parsedFile, err := parser.ParseFile(fs, name, reader, 0)
	if err != nil {
		errorf("%s: %s", name, err)
		return
	}
	file := &File{fset: fs, file: parsedFile}
	file.walkFile(name, parsedFile)
}

func visit(path string, f os.FileInfo, err error) error {
	if err != nil {
		errorf("walk error: %s", err)
		return nil
	}
	if !f.IsDir() && strings.HasSuffix(path, ".go") {
		doFile(path, nil)
	}
	return nil
}

// walkDir recursively walks the tree looking for .go files.
func walkDir(root string) {
	filepath.Walk(root, visit)
}

// error formats the error to standard error, adding program
// identification and a newline
func errorf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "vet: "+format+"\n", args...)
	setExit(2)
}

// Println is fmt.Println guarded by -v.
func Println(args ...interface{}) {
	if !*verbose {
		return
	}
	fmt.Println(args...)
}

// Printf is fmt.Printf guarded by -v.
func Printf(format string, args ...interface{}) {
	if !*verbose {
		return
	}
	fmt.Printf(format+"\n", args...)
}

// Bad reports an error and sets the exit code..
func (f *File) Bad(pos token.Pos, args ...interface{}) {
	f.Warn(pos, args...)
	setExit(1)
}

// Badf reports a formatted error and sets the exit code.
func (f *File) Badf(pos token.Pos, format string, args ...interface{}) {
	f.Warnf(pos, format, args...)
	setExit(1)
}

// Warn reports an error but does not set the exit code.
func (f *File) Warn(pos token.Pos, args ...interface{}) {
	loc := f.fset.Position(pos).String() + ": "
	fmt.Fprint(os.Stderr, loc+fmt.Sprintln(args...))
}

// Warnf reports a formatted error but does not set the exit code.
func (f *File) Warnf(pos token.Pos, format string, args ...interface{}) {
	loc := f.fset.Position(pos).String() + ": "
	fmt.Fprintf(os.Stderr, loc+format+"\n", args...)
}

// walkFile walks the file's tree.
func (f *File) walkFile(name string, file *ast.File) {
	Println("Checking file", name)
	ast.Walk(f, file)
}

// Visit implements the ast.Visitor interface.
func (f *File) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.CallExpr:
		f.walkCallExpr(n)
	case *ast.CompositeLit:
		f.walkCompositeLit(n)
	case *ast.Field:
		f.walkFieldTag(n)
	case *ast.FuncDecl:
		f.walkMethodDecl(n)
	case *ast.InterfaceType:
		f.walkInterfaceType(n)
	case *ast.RangeStmt:
		f.walkRangeStmt(n)
	}
	return f
}

// walkCall walks a call expression.
func (f *File) walkCall(call *ast.CallExpr, name string) {
	f.checkFmtPrintfCall(call, name)
}

// walkCallExpr walks a call expression.
func (f *File) walkCallExpr(call *ast.CallExpr) {
	switch x := call.Fun.(type) {
	case *ast.Ident:
		f.walkCall(call, x.Name)
	case *ast.SelectorExpr:
		f.walkCall(call, x.Sel.Name)
	}
}

// walkCompositeLit walks a composite literal.
func (f *File) walkCompositeLit(c *ast.CompositeLit) {
	f.checkUntaggedLiteral(c)
}

// walkFieldTag walks a struct field tag.
func (f *File) walkFieldTag(field *ast.Field) {
	if field.Tag == nil {
		return
	}
	f.checkCanonicalFieldTag(field)
}

// walkMethodDecl walks the method's signature.
func (f *File) walkMethod(id *ast.Ident, t *ast.FuncType) {
	f.checkCanonicalMethod(id, t)
}

// walkMethodDecl walks the method signature in the declaration.
func (f *File) walkMethodDecl(d *ast.FuncDecl) {
	if d.Recv == nil {
		// not a method
		return
	}
	f.walkMethod(d.Name, d.Type)
}

// walkInterfaceType walks the method signatures of an interface.
func (f *File) walkInterfaceType(t *ast.InterfaceType) {
	for _, field := range t.Methods.List {
		for _, id := range field.Names {
			f.walkMethod(id, field.Type.(*ast.FuncType))
		}
	}
}

// walkRangeStmt walks a range statement.
func (f *File) walkRangeStmt(n *ast.RangeStmt) {
	checkRangeLoop(f, n)
}
