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
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	_ "code.google.com/p/go.tools/go/gcimporter"
	"code.google.com/p/go.tools/go/types"
)

// TODO: Need a flag to set build tags when parsing the package.

var verbose = flag.Bool("v", false, "verbose")
var strictShadowing = flag.Bool("shadowstrict", false, "whether to be strict about shadowing; can be noisy")
var testFlag = flag.Bool("test", false, "for testing only: sets -all and -shadow")
var exitCode = 0

// "all" is here only for the appearance of backwards compatibility.
// It has no effect; the triState flags do the work.
var all = flag.Bool("all", true, "check everything; disabled if any explicit check is requested")

// Flags to control which individual checks to perform.
var report = map[string]*triState{
	"asmdecl":     triStateFlag("asmdecl", unset, "check assembly against Go declarations"),
	"assign":      triStateFlag("assign", unset, "check for useless assignments"),
	"atomic":      triStateFlag("atomic", unset, "check for common mistaken usages of the sync/atomic package"),
	"buildtags":   triStateFlag("buildtags", unset, "check that +build tags are valid"),
	"composites":  triStateFlag("composites", unset, "check that composite literals used field-keyed elements"),
	"copylocks":   triStateFlag("copylocks", unset, "check that locks are not passed by value"),
	"methods":     triStateFlag("methods", unset, "check that canonically named methods are canonically defined"),
	"nilfunc":     triStateFlag("nilfunc", unset, "check for comparisons between functions and nil"),
	"printf":      triStateFlag("printf", unset, "check printf-like invocations"),
	"rangeloops":  triStateFlag("rangeloops", unset, "check that range loop variables are used correctly"),
	"shadow":      triStateFlag("shadow", unset, "check for shadowed variables (experimental; must be set explicitly)"),
	"structtags":  triStateFlag("structtags", unset, "check that struct field tags have canonical format"),
	"unreachable": triStateFlag("unreachable", unset, "check for unreachable code"),
}

// experimental records the flags enabling experimental features. These must be
// requested explicitly; they are not enabled by -all.
var experimental = map[string]bool{
	"shadow": true,
}

// setTrueCount record how many flags are explicitly set to true.
var setTrueCount int

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
		return "unset"
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
	if *testFlag {
		return true
	}
	return report[name].isTrue()
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
	fmt.Fprintf(os.Stderr, "For more information run\n")
	fmt.Fprintf(os.Stderr, "\tgodoc code.google.com/p/go.tools/cmd/vet\n\n")
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

	// The last "String() string" method receiver we saw while walking.
	// This is used by the recursiveStringer method in print.go.
	lastStringerReceiver *ast.Object
}

func main() {
	flag.Usage = Usage
	flag.Parse()

	// If any flag is set, we run only those checks requested.
	// If no flags are set true, set all the non-experimental ones not explicitly set (in effect, set the "-all" flag).
	if setTrueCount == 0 {
		for name, setting := range report {
			if *setting == unset && !experimental[name] {
				*setting = setTrue
			}
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
	if !doPackage(".", flag.Args()) {
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
	var names []string
	names = append(names, pkg.GoFiles...)
	names = append(names, pkg.CgoFiles...)
	names = append(names, pkg.TestGoFiles...) // These are also in the "foo" package.
	names = append(names, pkg.SFiles...)
	prefixDirectory(directory, names)
	doPackage(directory, names)
	// Is there also a "foo_test" package? If so, do that one as well.
	if len(pkg.XTestGoFiles) > 0 {
		names = pkg.XTestGoFiles
		prefixDirectory(directory, names)
		doPackage(directory, names)
	}
}

type Package struct {
	path     string
	defs     map[*ast.Ident]types.Object
	uses     map[*ast.Ident]types.Object
	types    map[ast.Expr]types.TypeAndValue
	spans    map[types.Object]Span
	files    []*File
	typesPkg *types.Package
}

// doPackage analyzes the single package constructed from the named files.
// It returns whether any files were checked.
func doPackage(directory string, names []string) bool {
	var files []*File
	var astFiles []*ast.File
	fs := token.NewFileSet()
	for _, name := range names {
		f, err := os.Open(name)
		if err != nil {
			// Warn but continue to next package.
			warnf("%s: %s", name, err)
			return false
		}
		defer f.Close()
		data, err := ioutil.ReadAll(f)
		if err != nil {
			warnf("%s: %s", name, err)
			return false
		}
		checkBuildTag(name, data)
		var parsedFile *ast.File
		if strings.HasSuffix(name, ".go") {
			parsedFile, err = parser.ParseFile(fs, name, bytes.NewReader(data), 0)
			if err != nil {
				warnf("%s: %s", name, err)
				return false
			}
			astFiles = append(astFiles, parsedFile)
		}
		files = append(files, &File{fset: fs, content: data, name: name, file: parsedFile})
	}
	if len(astFiles) == 0 {
		return false
	}
	pkg := new(Package)
	pkg.path = astFiles[0].Name.Name
	pkg.files = files
	// Type check the package.
	err := pkg.check(fs, astFiles)
	if err != nil && *verbose {
		warnf("%s", err)
	}
	for _, file := range files {
		file.pkg = pkg
		if file.file != nil {
			file.walkFile(file.name, file.file)
		}
	}
	asmCheck(pkg)
	return true
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
	fmt.Fprint(os.Stderr, f.loc(pos)+": "+fmt.Sprintln(args...))
}

// Warnf reports a formatted error but does not set the exit code.
func (f *File) Warnf(pos token.Pos, format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, f.loc(pos)+": "+format+"\n", args...)
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
	case *ast.BinaryExpr:
		f.walkBinaryExpr(n)
	case *ast.CallExpr:
		f.walkCallExpr(n)
	case *ast.CompositeLit:
		f.walkCompositeLit(n)
	case *ast.Field:
		f.walkFieldTag(n)
	case *ast.FuncDecl:
		f.walkFuncDecl(n)
	case *ast.FuncLit:
		f.walkFuncLit(n)
	case *ast.GenDecl:
		f.walkGenDecl(n)
	case *ast.InterfaceType:
		f.walkInterfaceType(n)
	case *ast.RangeStmt:
		f.walkRangeStmt(n)
	}
	return f
}

// walkAssignStmt walks an assignment statement
func (f *File) walkAssignStmt(stmt *ast.AssignStmt) {
	f.checkAssignStmt(stmt)
	f.checkAtomicAssignment(stmt)
	f.checkShadowAssignment(stmt)
}

func (f *File) walkBinaryExpr(expr *ast.BinaryExpr) {
	f.checkNilFuncComparison(expr)
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
	f.checkUnkeyedLiteral(c)
}

// walkFieldTag walks a struct field tag.
func (f *File) walkFieldTag(field *ast.Field) {
	if field.Tag == nil {
		return
	}
	f.checkCanonicalFieldTag(field)
}

// walkMethod walks the method's signature.
func (f *File) walkMethod(id *ast.Ident, t *ast.FuncType) {
	f.checkCanonicalMethod(id, t)
}

// walkFuncDecl walks a function declaration.
func (f *File) walkFuncDecl(d *ast.FuncDecl) {
	f.checkUnreachable(d.Body)
	if d.Recv != nil {
		f.walkMethod(d.Name, d.Type)
	}
	f.prepStringerReceiver(d)
	f.checkCopyLocks(d)
}

// prepStringerReceiver checks whether the given declaration is a fmt.Stringer
// implementation, and if so sets the File's lastStringerReceiver field to the
// declaration's receiver object.
func (f *File) prepStringerReceiver(d *ast.FuncDecl) {
	if !f.isStringer(d) {
		return
	}
	if l := d.Recv.List; len(l) == 1 {
		if n := l[0].Names; len(n) == 1 {
			f.lastStringerReceiver = n[0].Obj
		}
	}
}

// isStringer returns true if the provided declaration is a "String() string"
// method; an implementation of fmt.Stringer.
func (f *File) isStringer(d *ast.FuncDecl) bool {
	return d.Recv != nil && d.Name.Name == "String" && d.Type.Results != nil &&
		len(d.Type.Params.List) == 0 && len(d.Type.Results.List) == 1 &&
		f.pkg.types[d.Type.Results.List[0].Type].Type == types.Typ[types.String]
}

// walkGenDecl walks a general declaration.
func (f *File) walkGenDecl(d *ast.GenDecl) {
	f.checkShadowDecl(d)
}

// walkFuncLit walks a function literal.
func (f *File) walkFuncLit(x *ast.FuncLit) {
	f.checkUnreachable(x.Body)
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

// gofmt returns a string representation of the expression.
func (f *File) gofmt(x ast.Expr) string {
	f.b.Reset()
	printer.Fprint(&f.b, f.fset, x)
	return f.b.String()
}
