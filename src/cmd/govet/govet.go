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
	"go/printer"
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
	fset *token.FileSet
	file *ast.File
	b    bytes.Buffer // for use by methods
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
	file := &File{fset: fs, file: parsedFile}
	file.checkFile(name, parsedFile)
}

func visit(path string, f *os.FileInfo, err os.Error) os.Error {
	if err != nil {
		errorf("walk error: %s", err)
		return nil
	}
	if f.IsRegular() && strings.HasSuffix(path, ".go") {
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
	loc := f.fset.Position(pos).String() + ": "
	fmt.Fprint(os.Stderr, loc+fmt.Sprintln(args...))
}

// Warnf reports a formatted error but does not set the exit code.
func (f *File) Warnf(pos token.Pos, format string, args ...interface{}) {
	loc := f.fset.Position(pos).String() + ": "
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
	case *ast.FuncDecl:
		f.checkMethodDecl(n)
	case *ast.InterfaceType:
		f.checkInterfaceType(n)
	}
	return f
}

// checkMethodDecl checks for canonical method signatures
// in method declarations.
func (f *File) checkMethodDecl(d *ast.FuncDecl) {
	if d.Recv == nil {
		// not a method
		return
	}

	f.checkMethod(d.Name, d.Type)
}

// checkInterfaceType checks for canonical method signatures
// in interface definitions.
func (f *File) checkInterfaceType(t *ast.InterfaceType) {
	for _, field := range t.Methods.List {
		for _, id := range field.Names {
			f.checkMethod(id, field.Type.(*ast.FuncType))
		}
	}
}

type MethodSig struct {
	args    []string
	results []string
}

// canonicalMethods lists the input and output types for Go methods
// that are checked using dynamic interface checks.  Because the
// checks are dynamic, such methods would not cause a compile error
// if they have the wrong signature: instead the dynamic check would
// fail, sometimes mysteriously.  If a method is found with a name listed
// here but not the input/output types listed here, govet complains.
//
// A few of the canonical methods have very common names.
// For example, a type might implement a Scan method that
// has nothing to do with fmt.Scanner, but we still want to check
// the methods that are intended to implement fmt.Scanner.
// To do that, the arguments that have a + prefix are treated as
// signals that the canonical meaning is intended: if a Scan
// method doesn't have a fmt.ScanState as its first argument,
// we let it go.  But if it does have a fmt.ScanState, then the
// rest has to match.
var canonicalMethods = map[string]MethodSig{
	// "Flush": {{}, {"os.Error"}}, // http.Flusher and jpeg.writer conflict
	"Format":        {[]string{"=fmt.State", "int"}, []string{}},                // fmt.Formatter
	"GobDecode":     {[]string{"[]byte"}, []string{"os.Error"}},                 // gob.GobDecoder
	"GobEncode":     {[]string{}, []string{"[]byte", "os.Error"}},               // gob.GobEncoder
	"MarshalJSON":   {[]string{}, []string{"[]byte", "os.Error"}},               // json.Marshaler
	"MarshalXML":    {[]string{}, []string{"[]byte", "os.Error"}},               // xml.Marshaler
	"Peek":          {[]string{"=int"}, []string{"[]byte", "os.Error"}},         // image.reader (matching bufio.Reader)
	"ReadByte":      {[]string{}, []string{"byte", "os.Error"}},                 // io.ByteReader
	"ReadFrom":      {[]string{"=io.Reader"}, []string{"int64", "os.Error"}},    // io.ReaderFrom
	"ReadRune":      {[]string{}, []string{"int", "int", "os.Error"}},           // io.RuneReader
	"Scan":          {[]string{"=fmt.ScanState", "int"}, []string{"os.Error"}},  // fmt.Scanner
	"Seek":          {[]string{"=int64", "int"}, []string{"int64", "os.Error"}}, // io.Seeker
	"UnmarshalJSON": {[]string{"[]byte"}, []string{"os.Error"}},                 // json.Unmarshaler
	"UnreadByte":    {[]string{}, []string{"os.Error"}},
	"UnreadRune":    {[]string{}, []string{"os.Error"}},
	"WriteByte":     {[]string{"byte"}, []string{"os.Error"}},                // jpeg.writer (matching bufio.Writer)
	"WriteTo":       {[]string{"=io.Writer"}, []string{"int64", "os.Error"}}, // io.WriterTo
}

func (f *File) checkMethod(id *ast.Ident, t *ast.FuncType) {
	// Expected input/output.
	expect, ok := canonicalMethods[id.Name]
	if !ok {
		return
	}

	// Actual input/output
	args := typeFlatten(t.Params.List)
	var results []ast.Expr
	if t.Results != nil {
		results = typeFlatten(t.Results.List)
	}

	// Do the =s (if any) all match?
	if !f.matchParams(expect.args, args, "=") || !f.matchParams(expect.results, results, "=") {
		return
	}

	// Everything must match.
	if !f.matchParams(expect.args, args, "") || !f.matchParams(expect.results, results, "") {
		expectFmt := id.Name + "(" + argjoin(expect.args) + ")"
		if len(expect.results) == 1 {
			expectFmt += " " + argjoin(expect.results)
		} else if len(expect.results) > 1 {
			expectFmt += " (" + argjoin(expect.results) + ")"
		}

		f.b.Reset()
		if err := printer.Fprint(&f.b, f.fset, t); err != nil {
			fmt.Fprintf(&f.b, "<%s>", err)
		}
		actual := f.b.String()
		if strings.HasPrefix(actual, "func(") {
			actual = actual[4:]
		}
		actual = id.Name + actual

		f.Warnf(id.Pos(), "method %s should have signature %s", actual, expectFmt)
	}
}

func argjoin(x []string) string {
	y := make([]string, len(x))
	for i, s := range x {
		if s[0] == '=' {
			s = s[1:]
		}
		y[i] = s
	}
	return strings.Join(y, ", ")
}

// Turn parameter list into slice of types
// (in the ast, types are Exprs).
// Have to handle f(int, bool) and f(x, y, z int)
// so not a simple 1-to-1 conversion.
func typeFlatten(l []*ast.Field) []ast.Expr {
	var t []ast.Expr
	for _, f := range l {
		if len(f.Names) == 0 {
			t = append(t, f.Type)
			continue
		}
		for _ = range f.Names {
			t = append(t, f.Type)
		}
	}
	return t
}

// Does each type in expect with the given prefix match the corresponding type in actual?
func (f *File) matchParams(expect []string, actual []ast.Expr, prefix string) bool {
	for i, x := range expect {
		if !strings.HasPrefix(x, prefix) {
			continue
		}
		if i >= len(actual) {
			return false
		}
		if !f.matchParamType(x, actual[i]) {
			return false
		}
	}
	if prefix == "" && len(actual) > len(expect) {
		return false
	}
	return true
}

// Does this one type match?
func (f *File) matchParamType(expect string, actual ast.Expr) bool {
	if strings.HasPrefix(expect, "=") {
		expect = expect[1:]
	}
	// Strip package name if we're in that package.
	if n := len(f.file.Name.Name); len(expect) > n && expect[:n] == f.file.Name.Name && expect[n] == '.' {
		expect = expect[n+1:]
	}

	// Overkill but easy.
	f.b.Reset()
	printer.Fprint(&f.b, f.fset, actual)
	return f.b.String() == expect
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

func (t *BadTypeUsedInTests) Scan(x fmt.ScanState, c byte) { // ERROR "method Scan[(]x fmt.ScanState, c byte[)] should have signature Scan[(]fmt.ScanState, int[)] os.Error"
}

type BadInterfaceUsedInTests interface {
	ReadByte() byte // ERROR "method ReadByte[(][)] byte should have signature ReadByte[(][)] [(]byte, os.Error[)]"
}

// printf is used by the test.
func printf(format string, args ...interface{}) {
	panic("don't call - testing only")
}
