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

var (
	verbose = flag.Bool("v", false, "verbose")
	tags    = flag.String("tags", "", "comma-separated list of build tags to apply when parsing")
	tagList = []string{} // exploded version of tags flag; set in main
)

var exitCode = 0

// "-all" flag enables all non-experimental checks
var all = triStateFlag("all", unset, "enable all non-experimental checks")

// Flags to control which individual checks to perform.
var report = map[string]*triState{
	// Only unusual checks are written here.
	// Most checks that operate during the AST walk are added by register.
	"asmdecl":   triStateFlag("asmdecl", unset, "check assembly against Go declarations"),
	"buildtags": triStateFlag("buildtags", unset, "check that +build tags are valid"),
}

// experimental records the flags enabling experimental features. These must be
// requested explicitly; they are not enabled by -all.
var experimental = map[string]bool{}

// setTrueCount record how many flags are explicitly set to true.
var setTrueCount int

// dirsRun and filesRun indicate whether the vet is applied to directory or
// file targets. The distinction affects which checks are run.
var dirsRun, filesRun bool

// includesNonTest indicates whether the vet is applied to non-test targets.
// Certain checks are relevant only if they touch both test and non-test files.
var includesNonTest bool

// A triState is a boolean that knows whether it has been set to either true or false.
// It is used to identify if a flag appears; the standard boolean flag cannot
// distinguish missing from unset. It also satisfies flag.Value.
type triState int

const (
	unset triState = iota
	setTrue
	setFalse
)

func triStateFlag(name string, value triState, usage string) *triState {
	flag.Var(&value, name, usage)
	return &value
}

// triState implements flag.Value, flag.Getter, and flag.boolFlag.
// They work like boolean flags: we can say vet -printf as well as vet -printf=true
func (ts *triState) Get() interface{} {
	return *ts == setTrue
}

func (ts triState) isTrue() bool {
	return ts == setTrue
}

func (ts *triState) Set(value string) error {
	b, err := strconv.ParseBool(value)
	if err != nil {
		return err
	}
	if b {
		*ts = setTrue
		setTrueCount++
	} else {
		*ts = setFalse
	}
	return nil
}

func (ts *triState) String() string {
	switch *ts {
	case unset:
		return "true" // An unset flag will be set by -all, so defaults to true.
	case setTrue:
		return "true"
	case setFalse:
		return "false"
	}
	panic("not reached")
}

func (ts triState) IsBoolFlag() bool {
	return true
}

// vet tells whether to report errors for the named check, a flag name.
func vet(name string) bool {
	return report[name].isTrue()
}

// setExit sets the value for os.Exit when it is called, later. It
// remembers the highest value.
func setExit(err int) {
	if err > exitCode {
		exitCode = err
	}
}

var (
	// Each of these vars has a corresponding case in (*File).Visit.
	assignStmt    *ast.AssignStmt
	binaryExpr    *ast.BinaryExpr
	callExpr      *ast.CallExpr
	compositeLit  *ast.CompositeLit
	exprStmt      *ast.ExprStmt
	field         *ast.Field
	funcDecl      *ast.FuncDecl
	funcLit       *ast.FuncLit
	genDecl       *ast.GenDecl
	interfaceType *ast.InterfaceType
	rangeStmt     *ast.RangeStmt
	returnStmt    *ast.ReturnStmt

	// checkers is a two-level map.
	// The outer level is keyed by a nil pointer, one of the AST vars above.
	// The inner level is keyed by checker name.
	checkers = make(map[ast.Node]map[string]func(*File, ast.Node))
)

func register(name, usage string, fn func(*File, ast.Node), types ...ast.Node) {
	report[name] = triStateFlag(name, unset, usage)
	for _, typ := range types {
		m := checkers[typ]
		if m == nil {
			m = make(map[string]func(*File, ast.Node))
			checkers[typ] = m
		}
		m[name] = fn
	}
}

// Usage is a replacement usage function for the flags package.
func Usage() {
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "\tvet [flags] directory...\n")
	fmt.Fprintf(os.Stderr, "\tvet [flags] files... # Must be a single package\n")
	fmt.Fprintf(os.Stderr, "By default, -all is set and all non-experimental checks are run.\n")
	fmt.Fprintf(os.Stderr, "For more information run\n")
	fmt.Fprintf(os.Stderr, "\tgo doc cmd/vet\n\n")
	fmt.Fprintf(os.Stderr, "Flags:\n")
	flag.PrintDefaults()
	os.Exit(2)
}

// File is a wrapper for the state of a file used in the parser.
// The parse tree walkers are all methods of this type.
type File struct {
	pkg     *Package
	fset    *token.FileSet
	name    string
	content []byte
	file    *ast.File
	b       bytes.Buffer // for use by methods

	// Parsed package "foo" when checking package "foo_test"
	basePkg *Package

	// The objects that are receivers of a "String() string" method.
	// This is used by the recursiveStringer method in print.go.
	stringers map[*ast.Object]bool

	// Registered checkers to run.
	checkers map[ast.Node][]func(*File, ast.Node)
}

func main() {
	flag.Usage = Usage
	flag.Parse()

	// If any flag is set, we run only those checks requested.
	// If all flag is set true or if no flags are set true, set all the non-experimental ones
	// not explicitly set (in effect, set the "-all" flag).
	if setTrueCount == 0 || *all == setTrue {
		for name, setting := range report {
			if *setting == unset && !experimental[name] {
				*setting = setTrue
			}
		}
	}

	tagList = strings.Split(*tags, ",")

	initPrintFlags()
	initUnusedFlags()

	if flag.NArg() == 0 {
		Usage()
	}
	for _, name := range flag.Args() {
		// Is it a directory?
		fi, err := os.Stat(name)
		if err != nil {
			warnf("error walking tree: %s", err)
			continue
		}
		if fi.IsDir() {
			dirsRun = true
		} else {
			filesRun = true
			if !strings.HasSuffix(name, "_test.go") {
				includesNonTest = true
			}
		}
	}
	if dirsRun && filesRun {
		Usage()
	}
	if dirsRun {
		for _, name := range flag.Args() {
			walkDir(name)
		}
		os.Exit(exitCode)
	}
	if doPackage(".", flag.Args(), nil) == nil {
		warnf("no files checked")
	}
	os.Exit(exitCode)
}

// prefixDirectory places the directory name on the beginning of each name in the list.
func prefixDirectory(directory string, names []string) {
	if directory != "." {
		for i, name := range names {
			names[i] = filepath.Join(directory, name)
		}
	}
}

// doPackageDir analyzes the single package found in the directory, if there is one,
// plus a test package, if there is one.
func doPackageDir(directory string) {
	context := build.Default
	if len(context.BuildTags) != 0 {
		warnf("build tags %s previously set", context.BuildTags)
	}
	context.BuildTags = append(tagList, context.BuildTags...)

	pkg, err := context.ImportDir(directory, 0)
	if err != nil {
		// If it's just that there are no go source files, that's fine.
		if _, nogo := err.(*build.NoGoError); nogo {
			return
		}
		// Non-fatal: we are doing a recursive walk and there may be other directories.
		warnf("cannot process directory %s: %s", directory, err)
		return
	}
	var names []string
	names = append(names, pkg.GoFiles...)
	names = append(names, pkg.CgoFiles...)
	names = append(names, pkg.TestGoFiles...) // These are also in the "foo" package.
	names = append(names, pkg.SFiles...)
	prefixDirectory(directory, names)
	basePkg := doPackage(directory, names, nil)
	// Is there also a "foo_test" package? If so, do that one as well.
	if len(pkg.XTestGoFiles) > 0 {
		names = pkg.XTestGoFiles
		prefixDirectory(directory, names)
		doPackage(directory, names, basePkg)
	}
}

type Package struct {
	path      string
	defs      map[*ast.Ident]types.Object
	uses      map[*ast.Ident]types.Object
	selectors map[*ast.SelectorExpr]*types.Selection
	types     map[ast.Expr]types.TypeAndValue
	spans     map[types.Object]Span
	files     []*File
	typesPkg  *types.Package
}

// doPackage analyzes the single package constructed from the named files.
// It returns the parsed Package or nil if none of the files have been checked.
func doPackage(directory string, names []string, basePkg *Package) *Package {
	var files []*File
	var astFiles []*ast.File
	fs := token.NewFileSet()
	for _, name := range names {
		data, err := ioutil.ReadFile(name)
		if err != nil {
			// Warn but continue to next package.
			warnf("%s: %s", name, err)
			return nil
		}
		checkBuildTag(name, data)
		var parsedFile *ast.File
		if strings.HasSuffix(name, ".go") {
			parsedFile, err = parser.ParseFile(fs, name, data, 0)
			if err != nil {
				warnf("%s: %s", name, err)
				return nil
			}
			astFiles = append(astFiles, parsedFile)
		}
		files = append(files, &File{fset: fs, content: data, name: name, file: parsedFile})
	}
	if len(astFiles) == 0 {
		return nil
	}
	pkg := new(Package)
	pkg.path = astFiles[0].Name.Name
	pkg.files = files
	// Type check the package.
	err := pkg.check(fs, astFiles)
	if err != nil && *verbose {
		warnf("%s", err)
	}

	// Check.
	chk := make(map[ast.Node][]func(*File, ast.Node))
	for typ, set := range checkers {
		for name, fn := range set {
			if vet(name) {
				chk[typ] = append(chk[typ], fn)
			}
		}
	}
	for _, file := range files {
		file.pkg = pkg
		file.basePkg = basePkg
		file.checkers = chk
		if file.file != nil {
			file.walkFile(file.name, file.file)
		}
	}
	asmCheck(pkg)
	return pkg
}

func visit(path string, f os.FileInfo, err error) error {
	if err != nil {
		warnf("walk error: %s", err)
		return err
	}
	// One package per directory. Ignore the files themselves.
	if !f.IsDir() {
		return nil
	}
	doPackageDir(path)
	return nil
}

func (pkg *Package) hasFileWithSuffix(suffix string) bool {
	for _, f := range pkg.files {
		if strings.HasSuffix(f.name, suffix) {
			return true
		}
	}
	return false
}

// walkDir recursively walks the tree looking for Go packages.
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

// loc returns a formatted representation of the position.
func (f *File) loc(pos token.Pos) string {
	if pos == token.NoPos {
		return ""
	}
	// Do not print columns. Because the pos often points to the start of an
	// expression instead of the inner part with the actual error, the
	// precision can mislead.
	posn := f.fset.Position(pos)
	return fmt.Sprintf("%s:%d", posn.Filename, posn.Line)
}

// Warn reports an error but does not set the exit code.
func (f *File) Warn(pos token.Pos, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s: %s", f.loc(pos), fmt.Sprintln(args...))
}

// Warnf reports a formatted error but does not set the exit code.
func (f *File) Warnf(pos token.Pos, format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s: %s\n", f.loc(pos), fmt.Sprintf(format, args...))
}

// walkFile walks the file's tree.
func (f *File) walkFile(name string, file *ast.File) {
	Println("Checking file", name)
	ast.Walk(f, file)
}

// Visit implements the ast.Visitor interface.
func (f *File) Visit(node ast.Node) ast.Visitor {
	var key ast.Node
	switch node.(type) {
	case *ast.AssignStmt:
		key = assignStmt
	case *ast.BinaryExpr:
		key = binaryExpr
	case *ast.CallExpr:
		key = callExpr
	case *ast.CompositeLit:
		key = compositeLit
	case *ast.ExprStmt:
		key = exprStmt
	case *ast.Field:
		key = field
	case *ast.FuncDecl:
		key = funcDecl
	case *ast.FuncLit:
		key = funcLit
	case *ast.GenDecl:
		key = genDecl
	case *ast.InterfaceType:
		key = interfaceType
	case *ast.RangeStmt:
		key = rangeStmt
	case *ast.ReturnStmt:
		key = returnStmt
	}
	for _, fn := range f.checkers[key] {
		fn(f, node)
	}
	return f
}

// gofmt returns a string representation of the expression.
func (f *File) gofmt(x ast.Expr) string {
	f.b.Reset()
	printer.Fprint(&f.b, f.fset, x)
	return f.b.String()
}
