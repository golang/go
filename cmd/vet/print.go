// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the printf-checker.

package main

import (
	"flag"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
	"unicode/utf8"
)

var printfuncs = flag.String("printfuncs", "", "comma-separated list of print function names to check")

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
func (f *File) checkFmtPrintfCall(call *ast.CallExpr, Name string) {
	if !vet("printf") {
		return
	}
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

// literal returns the literal value represented by the expression, or nil if it is not a literal.
func (f *File) literal(value ast.Expr) *ast.BasicLit {
	switch v := value.(type) {
	case *ast.BasicLit:
		return v
	case *ast.ParenExpr:
		return f.literal(v.X)
	case *ast.BinaryExpr:
		if v.Op != token.ADD {
			break
		}
		litX := f.literal(v.X)
		litY := f.literal(v.Y)
		if litX != nil && litY != nil {
			lit := *litX
			x, errX := strconv.Unquote(litX.Value)
			y, errY := strconv.Unquote(litY.Value)
			if errX == nil && errY == nil {
				return &ast.BasicLit{
					ValuePos: lit.ValuePos,
					Kind:     lit.Kind,
					Value:    strconv.Quote(x + y),
				}
			}
		}
	case *ast.Ident:
		// See if it's a constant or initial value (we can't tell the difference).
		if v.Obj == nil || v.Obj.Decl == nil {
			return nil
		}
		valueSpec, ok := v.Obj.Decl.(*ast.ValueSpec)
		if ok && len(valueSpec.Names) == len(valueSpec.Values) {
			// Find the index in the list of names
			var i int
			for i = 0; i < len(valueSpec.Names); i++ {
				if valueSpec.Names[i].Name == v.Name {
					if lit, ok := valueSpec.Values[i].(*ast.BasicLit); ok {
						return lit
					}
					return nil
				}
			}
		}
	}
	return nil
}

// formatState holds the parsed representation of a printf directive such as "%3.*[4]d"
type formatState struct {
	verb    rune   // the format verb: 'd' for "%d"
	flags   []byte // the list of # + etc.
	argNums []int  // the successive argument numbers that are consumed, adjusted to refer to actual arg in call
	nbytes  int    // number of bytes of the format string consumed.
	indexed bool   // whether an indexing expression appears: %[1]d.
}

// checkPrintf checks a call to a formatted print routine such as Printf.
// call.Args[formatIndex] is (well, should be) the format argument.
func (f *File) checkPrintf(call *ast.CallExpr, name string, formatIndex int) {
	if formatIndex >= len(call.Args) {
		return
	}
	lit := f.literal(call.Args[formatIndex])
	if lit == nil {
		if *verbose {
			f.Warn(call.Pos(), "can't check non-literal format in call to", name)
		}
		return
	}
	if lit.Kind != token.STRING {
		f.Badf(call.Pos(), "literal %v not a string in call to", lit.Value, name)
	}
	format, err := strconv.Unquote(lit.Value)
	if err != nil {
		// Shouldn't happen if parser returned no errors, but be safe.
		f.Badf(call.Pos(), "invalid quoted string literal")
	}
	firstArg := formatIndex + 1 // Arguments are immediately after format string.
	if !strings.Contains(format, "%") {
		if len(call.Args) > firstArg {
			f.Badf(call.Pos(), "no formatting directive in %s call", name)
		}
		return
	}
	// Hard part: check formats against args.
	argNum := firstArg
	indexed := false
	for i, w := 0, 0; i < len(format); i += w {
		w = 1
		if format[i] == '%' {
			state := f.parsePrintfVerb(call, format[i:], argNum)
			w = state.nbytes
			if state.indexed {
				indexed = true
			}
			f.checkPrintfArg(call, state)
			if len(state.argNums) > 0 {
				// Continue with the next sequential argument.
				argNum = state.argNums[len(state.argNums)-1] + 1
			}
		}
	}
	// Dotdotdot is hard.
	if call.Ellipsis.IsValid() && argNum >= len(call.Args)-1 {
		return
	}
	// If the arguments were direct indexed, we assume the programmer knows what's up.
	// Otherwise, there should be no leftover arguments.
	if !indexed && argNum != len(call.Args) {
		expect := argNum - firstArg
		numArgs := len(call.Args) - firstArg
		f.Badf(call.Pos(), "wrong number of args for format in %s call: %d needed but %d args", name, expect, numArgs)
	}
}

// parsePrintfVerb returns the verb that begins the format string, along with its flags,
// the number of bytes to advance the format to step past the verb, and number of
// arguments it consumes. It returns a formatState that encodes what the formatting
// directive wants, without looking at the actual arguments present in the call.
func (f *File) parsePrintfVerb(call *ast.CallExpr, format string, argNum int) *formatState {
	// There's guaranteed a percent sign.
	flags := make([]byte, 0, 5)
	nbytes := 1
	end := len(format)
	// There may be flags.
FlagLoop:
	for nbytes < end {
		switch format[nbytes] {
		case '#', '0', '+', '-', ' ':
			flags = append(flags, format[nbytes])
			nbytes++
		default:
			break FlagLoop
		}
	}
	argNums := make([]int, 0, 1)
	getNum := func() {
		if nbytes < end && format[nbytes] == '*' {
			nbytes++
			argNums = append(argNums, argNum)
			argNum++
		} else {
			for nbytes < end && '0' <= format[nbytes] && format[nbytes] <= '9' {
				nbytes++
			}
		}
	}
	// There may be a width.
	getNum()
	// If there's a period, there may be a precision.
	if nbytes < end && format[nbytes] == '.' {
		flags = append(flags, '.') // Treat precision as a flag.
		nbytes++
		getNum()
	}
	// Now a verb.
	verb, w := utf8.DecodeRuneInString(format[nbytes:])
	nbytes += w
	if verb != '%' {
		argNums = append(argNums, argNum)
		argNum++
	}
	return &formatState{
		verb:    verb,
		flags:   flags,
		argNums: argNums,
		nbytes:  nbytes,
		indexed: false, // TODO
	}
}

// printfArgType encodes the types of expressions a printf verb accepts. It is a bitmask.
type printfArgType int

const (
	argBool printfArgType = 1 << iota
	argInt
	argRune
	argString
	argFloat
	argPointer
	anyType printfArgType = ^0
)

type printVerb struct {
	verb  rune   // User may provide verb through Formatter; could be a rune.
	flags string // known flags are all ASCII
	typ   printfArgType
}

// Common flag sets for printf verbs.
const (
	noFlag       = ""
	numFlag      = " -+.0"
	sharpNumFlag = " -+.0#"
	allFlags     = " -+.0#"
)

// printVerbs identifies which flags are known to printf for each verb.
// TODO: A type that implements Formatter may do what it wants, and vet
// will complain incorrectly.
var printVerbs = []printVerb{
	// '-' is a width modifier, always valid.
	// '.' is a precision for float, max width for strings.
	// '+' is required sign for numbers, Go format for %v.
	// '#' is alternate format for several verbs.
	// ' ' is spacer for numbers
	{'%', noFlag, 0},
	{'b', numFlag, argInt | argFloat},
	{'c', "-", argRune | argInt},
	{'d', numFlag, argInt},
	{'e', numFlag, argFloat},
	{'E', numFlag, argFloat},
	{'f', numFlag, argFloat},
	{'F', numFlag, argFloat},
	{'g', numFlag, argFloat},
	{'G', numFlag, argFloat},
	{'o', sharpNumFlag, argInt},
	{'p', "-#", argPointer},
	{'q', " -+.0#", argRune | argInt | argString},
	{'s', " -+.0", argString},
	{'t', "-", argBool},
	{'T', "-", anyType},
	{'U', "-#", argRune | argInt},
	{'v', allFlags, anyType},
	{'x', sharpNumFlag, argRune | argInt | argString},
	{'X', sharpNumFlag, argRune | argInt | argString},
}

// checkPrintfArg compares the formatState to the arguments actually present,
// reporting any discrepancies it can discern. If the final argument is ellipsissed,
// there's little it can do for that.
func (f *File) checkPrintfArg(call *ast.CallExpr, state *formatState) {
	var v printVerb
	found := false
	// Linear scan is fast enough for a small list.
	for _, v = range printVerbs {
		if v.verb == state.verb {
			found = true
			break
		}
	}
	if !found {
		f.Badf(call.Pos(), "unrecognized printf verb %q", state.verb)
		return
	}
	for _, flag := range state.flags {
		if !strings.ContainsRune(v.flags, rune(flag)) {
			f.Badf(call.Pos(), "unrecognized printf flag for verb %q: %q", state.verb, flag)
			return
		}
	}
	// Verb is good. If len(state.argNums)>trueArgs, we have something like %.*s and all
	// but the final arg must be an integer.
	trueArgs := 1
	if state.verb == '%' {
		trueArgs = 0
	}
	nargs := len(state.argNums)
	for i := 0; i < nargs-trueArgs; i++ {
		argNum := state.argNums[i]
		if !f.argCanBeChecked(call, argNum, true, state.verb) {
			continue
		}
		arg := call.Args[argNum]
		if !f.matchArgType(argInt, arg) {
			f.Badf(call.Pos(), "arg %s for * in printf format not of type int", f.gofmt(arg))
		}
	}
	if state.verb == '%' {
		return
	}
	argNum := state.argNums[len(state.argNums)-1]
	if !f.argCanBeChecked(call, argNum, false, state.verb) {
		return
	}
	arg := call.Args[argNum]
	if !f.matchArgType(v.typ, arg) {
		typeString := ""
		if typ := f.pkg.types[arg]; typ != nil {
			typeString = typ.String()
		}
		f.Badf(call.Pos(), "arg %s for printf verb %%%c of wrong type: %s", f.gofmt(arg), state.verb, typeString)
	}
}

// argCanBeChecked reports whether the specified argument is statically present;
// it may be beyond the list of arguments or in a terminal slice... argument, which
// means we can't see it.
func (f *File) argCanBeChecked(call *ast.CallExpr, argNum int, isStar bool, verb rune) bool {
	if argNum < 0 {
		// Shouldn't happen, so catch it with prejudice.
		panic("negative arg num")
	}
	if argNum < len(call.Args)-1 {
		return true // Always OK.
	}
	if call.Ellipsis.IsValid() {
		return false // We just can't tell; there could be many more arguments.
	}
	if argNum < len(call.Args) {
		return true
	}
	if verb == '*' {
		f.Badf(call.Pos(), "argument number %d for * for verb %%%c out of range", argNum, verb)
	} else {
		f.Badf(call.Pos(), "wrong number of args for format in Printf call")
	}
	return false
}

// checkPrint checks a call to an unformatted print routine such as Println.
// call.Args[firstArg] is the first argument to be printed.
func (f *File) checkPrint(call *ast.CallExpr, name string, firstArg int) {
	isLn := strings.HasSuffix(name, "ln")
	isF := strings.HasPrefix(name, "F")
	args := call.Args
	// check for Println(os.Stderr, ...)
	if firstArg == 0 && !isF && len(args) > 0 {
		if sel, ok := args[0].(*ast.SelectorExpr); ok {
			if x, ok := sel.X.(*ast.Ident); ok {
				if x.Name == "os" && strings.HasPrefix(sel.Sel.Name, "Std") {
					f.Badf(call.Pos(), "first argument to %s is %s.%s", name, x.Name, sel.Sel.Name)
				}
			}
		}
	}
	if len(args) <= firstArg {
		// If we have a call to a method called Error that satisfies the Error interface,
		// then it's ok. Otherwise it's something like (*T).Error from the testing package
		// and we need to check it.
		if name == "Error" && f.isErrorMethodCall(call) {
			return
		}
		// If it's an Error call now, it's probably for printing errors.
		if !isLn {
			// Check the signature to be sure: there are niladic functions called "error".
			if firstArg != 0 || f.numArgsInSignature(call) != firstArg {
				f.Badf(call.Pos(), "no args in %s call", name)
			}
		}
		return
	}
	arg := args[firstArg]
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
