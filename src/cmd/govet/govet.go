// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Govet is a simple checker for static errors in Go source code.
// See doc.go for more information.
package main

import (
	"flag"
	"fmt"
	"io"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"utf8"
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
		for _, name := range strings.Split(*printfuncs, ",") {
			if len(name) == 0 {
				flag.Usage()
			}
			skip := 0
			if colon := strings.LastIndex(name, ":"); colon > 0 {
				var err os.Error
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
		errorf("%s: %s", name, err)
		return
	}
	file := &File{fs.File(parsedFile.Pos())}
	file.checkFile(name, parsedFile)
}

// Visitor for filepath.Walk - trivial.  Just calls doFile on each file.
// TODO: if govet becomes richer, might want to process
// a directory (package) at a time.
type fileVisitor chan os.Error

func (v fileVisitor) VisitDir(path string, f *os.FileInfo) bool {
	return true
}

func (v fileVisitor) VisitFile(path string, f *os.FileInfo) {
	if strings.HasSuffix(path, ".go") {
		doFile(path, nil)
	}
}

func (v fileVisitor) Error(path string, err os.Error) {
	v <- err
}

// walkDir recursively walks the tree looking for .go files.
func walkDir(root string) {
	v := make(fileVisitor)
	done := make(chan bool)
	go func() {
		for e := range v {
			errorf("walk error: %s", e)
		}
		done <- true
	}()
	filepath.Walk(root, v)
	close(v)
	<-done
}

// error formats the error to standard error, adding program
// identification and a newline
func errorf(format string, args ...interface{}) {
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
	switch n := node.(type) {
	case *ast.CallExpr:
		f.checkCallExpr(n)
	case *ast.Field:
		f.checkFieldTag(n)
	}
	return f
}

// checkField checks a struct field tag.
func (f *File) checkFieldTag(field *ast.Field) {
	if field.Tag == nil {
		return
	}

	tag, err := strconv.Unquote(field.Tag.Value)
	if err != nil {
		f.Warnf(field.Pos(), "unable to read struct tag %s", field.Tag.Value)
		return
	}

	// Check tag for validity by appending
	// new key:value to end and checking that
	// the tag parsing code can find it.
	if reflect.StructTag(tag+` _gofix:"_magic"`).Get("_gofix") != "_magic" {
		f.Warnf(field.Pos(), "struct field tag %s not compatible with reflect.StructTag.Get", field.Tag.Value)
		return
	}
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
// of the format parameter. Names are lower-cased so the lookup is
// case insensitive.
var printfList = map[string]int{
	"errorf":  0,
	"fatalf":  0,
	"fprintf": 1,
	"panicf":  0,
	"printf":  0,
	"sprintf": 0,
}

// printList records the unformatted-print functions. The value is the location
// of the first parameter to be printed.  Names are lower-cased so the lookup is
// case insensitive.
var printList = map[string]int{
	"error":  0,
	"fatal":  0,
	"fprint": 1, "fprintln": 1,
	"panic": 0, "panicln": 0,
	"print": 0, "println": 0,
	"sprint": 0, "sprintln": 0,
}

// checkCall triggers the print-specific checks if the call invokes a print function.
func (f *File) checkCall(call *ast.CallExpr, Name string) {
	name := strings.ToLower(Name)
	if skip, ok := printfList[name]; ok {
		f.checkPrintf(call, Name, skip)
		return
	}
	if skip, ok := printList[name]; ok {
		f.checkPrint(call, Name, skip)
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
		if !strings.Contains(lit.Value, "%") {
			if len(call.Args) > skip+1 {
				f.Badf(call.Pos(), "no formatting directive in %s call", name)
			}
			return
		}
	}
	// Hard part: check formats against args.
	// Trivial but useful test: count.
	numArgs := 0
	for i, w := 0, 0; i < len(lit.Value); i += w {
		w = 1
		if lit.Value[i] == '%' {
			nbytes, nargs := parsePrintfVerb(lit.Value[i:])
			w = nbytes
			numArgs += nargs
		}
	}
	expect := len(call.Args) - (skip + 1)
	if numArgs != expect {
		f.Badf(call.Pos(), "wrong number of args in %s call: %d needed but %d args", name, numArgs, expect)
	}
}

// parsePrintfVerb returns the number of bytes and number of arguments
// consumed by the Printf directive that begins s, including its percent sign
// and verb.
func parsePrintfVerb(s string) (nbytes, nargs int) {
	// There's guaranteed a percent sign.
	nbytes = 1
	end := len(s)
	// There may be flags.
FlagLoop:
	for nbytes < end {
		switch s[nbytes] {
		case '#', '0', '+', '-', ' ':
			nbytes++
		default:
			break FlagLoop
		}
	}
	getNum := func() {
		if nbytes < end && s[nbytes] == '*' {
			nbytes++
			nargs++
		} else {
			for nbytes < end && '0' <= s[nbytes] && s[nbytes] <= '9' {
				nbytes++
			}
		}
	}
	// There may be a width.
	getNum()
	// If there's a period, there may be a precision.
	if nbytes < end && s[nbytes] == '.' {
		nbytes++
		getNum()
	}
	// Now a verb.
	c, w := utf8.DecodeRuneInString(s[nbytes:])
	nbytes += w
	if c != '%' {
		nargs++
	}
	return
}

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
		if strings.Contains(lit.Value, "%") {
			f.Badf(call.Pos(), "possible formatting directive in %s call", name)
		}
	}
	if isLn {
		// The last item, if a string, should not have a newline.
		arg = args[len(call.Args)-1]
		if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			if strings.HasSuffix(lit.Value, `\n"`) {
				f.Badf(call.Pos(), "%s call ends with newline", name)
			}
		}
	}
}

// This function never executes, but it serves as a simple test for the program.
// Test with make test.
func BadFunctionUsedInTests() {
	fmt.Println()                      // not an error
	fmt.Println("%s", "hi")            // ERROR "possible formatting directive in Println call"
	fmt.Printf("%s", "hi", 3)          // ERROR "wrong number of args in Printf call"
	fmt.Printf("%s%%%d", "hi", 3)      // correct
	fmt.Printf("%.*d", 3, 3)           // correct
	fmt.Printf("%.*d", 3, 3, 3)        // ERROR "wrong number of args in Printf call"
	printf("now is the time", "buddy") // ERROR "no formatting directive"
	Printf("now is the time", "buddy") // ERROR "no formatting directive"
	Printf("hi")                       // ok
	f := new(File)
	f.Warn(0, "%s", "hello", 3)  // ERROR "possible formatting directive in Warn call"
	f.Warnf(0, "%s", "hello", 3) // ERROR "wrong number of args in Warnf call"
}

type BadTypeUsedInTests struct {
	X int "hello" // ERROR "struct field tag"
}

// printf is used by the test.
func printf(format string, args ...interface{}) {
	panic("don't call - testing only")
}
