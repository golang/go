// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Govet is a simple checker for static errors in Go source code.
// See doc.go for more information.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path"
	"strconv"
	"strings"
)

var verbose = flag.Bool("v", false, "verbose")
var printfuncs = flag.String("printfuncs", "", "comma-separated list of print function names to check")
var exitCode = 0

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
	file *token.File
}

func main() {
	flag.Usage = Usage
	flag.Parse()

	if *printfuncs != "" {
		for _, name := range strings.Split(*printfuncs, ",", -1) {
			if len(name) == 0 {
				flag.Usage()
			}
			skip := 0
			if colon := strings.LastIndex(name, ":"); colon > 0 {
				var err os.Error
				skip, err = strconv.Atoi(name[colon+1:])
				if err != nil {
					error(`illegal format for "Func:N" argument %q; %s`, name, err)
				}
				name = name[:colon]
			}
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
			if fi, err := os.Stat(name); err == nil && fi.IsDirectory() {
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
		error("%s: %s", name, err)
		return
	}
	file := &File{fs.File(parsedFile.Pos())}
	file.checkFile(name, parsedFile)
}

// Visitor for path.Walk - trivial.  Just calls doFile on each file.
// TODO: if govet becomes richer, might want to process
// a directory (package) at a time.
type V struct{}

func (v V) VisitDir(path string, f *os.FileInfo) bool {
	return true
}

func (v V) VisitFile(path string, f *os.FileInfo) {
	if strings.HasSuffix(path, ".go") {
		doFile(path, nil)
	}
}

// walkDir recursively walks the tree looking for .go files.
func walkDir(root string) {
	errors := make(chan os.Error)
	done := make(chan bool)
	go func() {
		for e := range errors {
			error("walk error: %s", e)
		}
		done <- true
	}()
	path.Walk(root, V{}, errors)
	close(errors)
	<-done
}

// error formats the error to standard error, adding program
// identification and a newline
func error(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "govet: "+format+"\n", args...)
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
	loc := f.file.Position(pos).String() + ": "
	fmt.Fprint(os.Stderr, loc+fmt.Sprintln(args...))
}

// Warnf reports a formatted error but does not set the exit code.
func (f *File) Warnf(pos token.Pos, format string, args ...interface{}) {
	loc := f.file.Position(pos).String() + ": "
	fmt.Fprintf(os.Stderr, loc+format+"\n", args...)
}

// checkFile checks all the top-level declarations in a file.
func (f *File) checkFile(name string, file *ast.File) {
	Println("Checking file", name)
	ast.Walk(f, file)
}

// Visit implements the ast.Visitor interface.
func (f *File) Visit(node ast.Node) ast.Visitor {
	// TODO: could return nil for nodes that cannot contain a CallExpr -
	// will shortcut traversal.  Worthwhile?
	switch n := node.(type) {
	case *ast.CallExpr:
		f.checkCallExpr(n)
	}
	return f
}


// checkCallExpr checks a call expression.
func (f *File) checkCallExpr(call *ast.CallExpr) {
	switch x := call.Fun.(type) {
	case *ast.Ident:
		f.checkCall(call, x.Name)
	case *ast.SelectorExpr:
		f.checkCall(call, x.Sel.Name)
	}
}

// printfList records the formatted-print functions. The value is the location
// of the format parameter.
var printfList = map[string]int{
	"Errorf":  0,
	"Fatalf":  0,
	"Fprintf": 1,
	"Printf":  0,
	"Sprintf": 0,
}

// printList records the unformatted-print functions. The value is the location
// of the first parameter to be printed.
var printList = map[string]int{
	"Error":  0,
	"Fatal":  0,
	"Fprint": 1, "Fprintln": 1,
	"Print": 0, "Println": 0,
	"Sprint": 0, "Sprintln": 0,
}

// checkCall triggers the print-specific checks if the call invokes a print function.
func (f *File) checkCall(call *ast.CallExpr, name string) {
	if skip, ok := printfList[name]; ok {
		f.checkPrintf(call, name, skip)
		return
	}
	if skip, ok := printList[name]; ok {
		f.checkPrint(call, name, skip)
		return
	}
}

// checkPrintf checks a call to a formatted print routine such as Printf.
// The skip argument records how many arguments to ignore; that is,
// call.Args[skip] is (well, should be) the format argument.
func (f *File) checkPrintf(call *ast.CallExpr, name string, skip int) {
	if len(call.Args) <= skip {
		return
	}
	// Common case: literal is first argument.
	arg := call.Args[skip]
	lit, ok := arg.(*ast.BasicLit)
	if !ok {
		// Too hard to check.
		if *verbose {
			f.Warn(call.Pos(), "can't check args for call to", name)
		}
		return
	}
	if lit.Kind == token.STRING {
		if bytes.IndexByte(lit.Value, '%') < 0 {
			if len(call.Args) > skip+1 {
				f.Badf(call.Pos(), "no formatting directive in %s call", name)
			}
			return
		}
	}
	// Hard part: check formats against args.
	// Trivial but useful test: count.
	numPercent := 0
	for i := 0; i < len(lit.Value); i++ {
		if lit.Value[i] == '%' {
			if i+1 < len(lit.Value) && lit.Value[i+1] == '%' {
				// %% doesn't count.
				i++
			} else {
				numPercent++
			}
		}
	}
	expect := len(call.Args) - (skip + 1)
	if numPercent != expect {
		f.Badf(call.Pos(), "wrong number of formatting directives in %s call: %d percent(s) for %d args", name, numPercent, expect)
	}
}

var terminalNewline = []byte(`\n"`) // \n at end of interpreted string

// checkPrint checks a call to an unformatted print routine such as Println.
// The skip argument records how many arguments to ignore; that is,
// call.Args[skip] is the first argument to be printed.
func (f *File) checkPrint(call *ast.CallExpr, name string, skip int) {
	isLn := strings.HasSuffix(name, "ln")
	args := call.Args
	if len(args) <= skip {
		if *verbose && !isLn {
			f.Badf(call.Pos(), "no args in %s call", name)
		}
		return
	}
	arg := args[skip]
	if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
		if bytes.IndexByte(lit.Value, '%') >= 0 {
			f.Badf(call.Pos(), "possible formatting directive in %s call", name)
		}
	}
	if isLn {
		// The last item, if a string, should not have a newline.
		arg = args[len(call.Args)-1]
		if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			if bytes.HasSuffix(lit.Value, terminalNewline) {
				f.Badf(call.Pos(), "%s call ends with newline", name)
			}
		}
	}
}

// This function never executes, but it serves as a simple test for the program.
// Test with govet -printfuncs="Bad:1,Badf:1,Warn:1,Warnf:1" govet.go
func BadFunctionUsedInTests() {
	fmt.Println()                      // niladic call
	fmt.Println("%s", "hi")            // % in call to Println
	fmt.Printf("%s", "hi", 3)          // wrong # percents
	fmt.Printf("%s%%%d", "hi", 3)      // right # percents
	Printf("now is the time", "buddy") // no %s
	f := new(File)
	f.Warn(0, "%s", "hello", 3)  // % in call to added function
	f.Warnf(0, "%s", "hello", 3) // wrong # %s in call to added function
}
