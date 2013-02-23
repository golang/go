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
	"go/build"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

var verbose = flag.Bool("v", false, "verbose")
var exitCode = 0

// Flags to control which checks to perform. "all" is set to true here, and disabled later if
// a flag is set explicitly.
var report = map[string]*bool{
	"all":        flag.Bool("all", true, "check everything; disabled if any explicit check is requested"),
	"atomic":     flag.Bool("atomic", false, "check for common mistaken usages of the sync/atomic package"),
	"buildtags":  flag.Bool("buildtags", false, "check that +build tags are valid"),
	"composites": flag.Bool("composites", false, "check that composite literals used type-tagged elements"),
	"methods":    flag.Bool("methods", false, "check that canonically named methods are canonically defined"),
	"printf":     flag.Bool("printf", false, "check printf-like invocations"),
	"structtags": flag.Bool("structtags", false, "check that struct field tags have canonical format"),
	"rangeloops": flag.Bool("rangeloops", false, "check that range loop variables are used correctly"),
}

// vet tells whether to report errors for the named check, a flag name.
func vet(name string) bool {
	return *report["all"] || *report[name]
}

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
	fmt.Fprintf(os.Stderr, "\tvet [flags] directory...\n")
	fmt.Fprintf(os.Stderr, "\tvet [flags] files... # Must be a single package\n")
	flag.PrintDefaults()
	os.Exit(2)
}

// File is a wrapper for the state of a file used in the parser.
// The parse tree walkers are all methods of this type.
type File struct {
	pkg  *Package
	fset *token.FileSet
	name string
	file *ast.File
	b    bytes.Buffer // for use by methods
}

func main() {
	flag.Usage = Usage
	flag.Parse()

	// If a check is named explicitly, turn off the 'all' flag.
	for name, ptr := range report {
		if name != "all" && *ptr {
			*report["all"] = false
			break
		}
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
		Usage()
	}
	dirs := false
	files := false
	for _, name := range flag.Args() {
		// Is it a directory?
		fi, err := os.Stat(name)
		if err != nil {
			warnf("error walking tree: %s", err)
			continue
		}
		if fi.IsDir() {
			dirs = true
		} else {
			files = true
		}
	}
	if dirs && files {
		Usage()
	}
	if dirs {
		for _, name := range flag.Args() {
			walkDir(name)
		}
		return
	}
	doPackage(flag.Args())
	os.Exit(exitCode)
}

// doPackageDir analyzes the single package found in the directory, if there is one.
func doPackageDir(directory string) {
	pkg, err := build.Default.ImportDir(directory, 0)
	if err != nil {
		// If it's just that there are no go source files, that's fine.
		if _, nogo := err.(*build.NoGoError); nogo {
			return
		}
		// Non-fatal: we are doing a recursive walk and there may be other directories.
		warnf("cannot process directory %s: %s", directory, err)
		return
	}
	names := append(pkg.GoFiles, pkg.CgoFiles...)
	// Prefix file names with directory names.
	if directory != "." {
		for i, name := range names {
			names[i] = filepath.Join(directory, name)
		}
	}
	doPackage(names)
}

type Package struct {
	types map[ast.Expr]types.Type
}

// doPackage analyzes the single package constructed from the named files.
func doPackage(names []string) {
	var files []*File
	var astFiles []*ast.File
	fs := token.NewFileSet()
	for _, name := range names {
		f, err := os.Open(name)
		if err != nil {
			errorf("%s: %s", name, err)
		}
		defer f.Close()
		data, err := ioutil.ReadAll(f)
		if err != nil {
			errorf("%s: %s", name, err)
		}
		checkBuildTag(name, data)
		parsedFile, err := parser.ParseFile(fs, name, bytes.NewReader(data), 0)
		if err != nil {
			errorf("%s: %s", name, err)
		}
		files = append(files, &File{fset: fs, name: name, file: parsedFile})
		astFiles = append(astFiles, parsedFile)
	}
	pkg := new(Package)
	pkg.types = make(map[ast.Expr]types.Type)
	exprFn := func(x ast.Expr, typ types.Type, val interface{}) {
		pkg.types[x] = typ
	}
	context := types.Context{
		Expr: exprFn,
	}
	// Type check the package.
	_, err := context.Check(fs, astFiles)
	if err != nil {
		warnf("%s", err)
	}
	for _, file := range files {
		file.pkg = pkg
		file.walkFile(file.name, file.file)
	}
}

func visit(path string, f os.FileInfo, err error) error {
	if err != nil {
		errorf("walk error: %s", err)
	}
	// One package per directory. Ignore the files themselves.
	if !f.IsDir() {
		return nil
	}
	doPackageDir(path)
	return nil
}

// walkDir recursively walks the tree looking for .go files.
func walkDir(root string) {
	filepath.Walk(root, visit)
}

// errorf formats the error to standard error, adding program
// identification and a newline, and exits.
func errorf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "vet: "+format+"\n", args...)
	os.Exit(2)
}

// warnf formats the error to standard error, adding program
// identification and a newline, but does not exit.
func warnf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "vet: "+format+"\n", args...)
	setExit(1)
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

func (f *File) loc(pos token.Pos) string {
	// Do not print columns. Because the pos often points to the start of an
	// expression instead of the inner part with the actual error, the
	// precision can mislead.
	posn := f.fset.Position(pos)
	return fmt.Sprintf("%s:%d: ", posn.Filename, posn.Line)
}

// Warn reports an error but does not set the exit code.
func (f *File) Warn(pos token.Pos, args ...interface{}) {
	fmt.Fprint(os.Stderr, f.loc(pos)+fmt.Sprintln(args...))
}

// Warnf reports a formatted error but does not set the exit code.
func (f *File) Warnf(pos token.Pos, format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, f.loc(pos)+format+"\n", args...)
}

// walkFile walks the file's tree.
func (f *File) walkFile(name string, file *ast.File) {
	Println("Checking file", name)
	ast.Walk(f, file)
}

// Visit implements the ast.Visitor interface.
func (f *File) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.AssignStmt:
		f.walkAssignStmt(n)
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

// walkAssignStmt walks an assignment statement
func (f *File) walkAssignStmt(stmt *ast.AssignStmt) {
	f.checkAtomicAssignment(stmt)
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

// goFmt returns a string representation of the expression
func (f *File) gofmt(x ast.Expr) string {
	f.b.Reset()
	printer.Fprint(&f.b, f.fset, x)
	return f.b.String()
}
