// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the printf-checker.

package main

import (
	"bytes"
	"flag"
	"go/ast"
	exact "go/constant"
	"go/token"
	"go/types"
	"strconv"
	"strings"
	"unicode/utf8"
)

var printfuncs = flag.String("printfuncs", "", "comma-separated list of print function names to check")

func init() {
	register("printf",
		"check printf-like invocations",
		checkFmtPrintfCall,
		funcDecl, callExpr)
}

func initPrintFlags() {
	if *printfuncs == "" {
		return
	}
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

// printfList records the formatted-print functions. The value is the location
// of the format parameter. Names are lower-cased so the lookup is
// case insensitive.
var printfList = map[string]int{
	"errorf":  0,
	"fatalf":  0,
	"fprintf": 1,
	"logf":    0,
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
	"log":   0,
	"panic": 0, "panicln": 0,
	"print": 0, "println": 0,
	"sprint": 0, "sprintln": 0,
}

// checkCall triggers the print-specific checks if the call invokes a print function.
func checkFmtPrintfCall(f *File, node ast.Node) {
	if d, ok := node.(*ast.FuncDecl); ok && isStringer(f, d) {
		// Remember we saw this.
		if f.stringers == nil {
			f.stringers = make(map[*ast.Object]bool)
		}
		if l := d.Recv.List; len(l) == 1 {
			if n := l[0].Names; len(n) == 1 {
				f.stringers[n[0].Obj] = true
			}
		}
		return
	}

	call, ok := node.(*ast.CallExpr)
	if !ok {
		return
	}
	var Name string
	switch x := call.Fun.(type) {
	case *ast.Ident:
		Name = x.Name
	case *ast.SelectorExpr:
		Name = x.Sel.Name
	default:
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

// isStringer returns true if the provided declaration is a "String() string"
// method, an implementation of fmt.Stringer.
func isStringer(f *File, d *ast.FuncDecl) bool {
	return d.Recv != nil && d.Name.Name == "String" && d.Type.Results != nil &&
		len(d.Type.Params.List) == 0 && len(d.Type.Results.List) == 1 &&
		f.pkg.types[d.Type.Results.List[0].Type].Type == types.Typ[types.String]
}

// formatState holds the parsed representation of a printf directive such as "%3.*[4]d".
// It is constructed by parsePrintfVerb.
type formatState struct {
	verb     rune   // the format verb: 'd' for "%d"
	format   string // the full format directive from % through verb, "%.3d".
	name     string // Printf, Sprintf etc.
	flags    []byte // the list of # + etc.
	argNums  []int  // the successive argument numbers that are consumed, adjusted to refer to actual arg in call
	indexed  bool   // whether an indexing expression appears: %[1]d.
	firstArg int    // Index of first argument after the format in the Printf call.
	// Used only during parse.
	file         *File
	call         *ast.CallExpr
	argNum       int  // Which argument we're expecting to format now.
	indexPending bool // Whether we have an indexed argument that has not resolved.
	nbytes       int  // number of bytes of the format string consumed.
}

// checkPrintf checks a call to a formatted print routine such as Printf.
// call.Args[formatIndex] is (well, should be) the format argument.
func (f *File) checkPrintf(call *ast.CallExpr, name string, formatIndex int) {
	if formatIndex >= len(call.Args) {
		f.Bad(call.Pos(), "too few arguments in call to", name)
		return
	}
	lit := f.pkg.types[call.Args[formatIndex]].Value
	if lit == nil {
		if *verbose {
			f.Warn(call.Pos(), "can't check non-constant format in call to", name)
		}
		return
	}
	if lit.Kind() != exact.String {
		f.Badf(call.Pos(), "constant %v not a string in call to %s", lit, name)
		return
	}
	format := exact.StringVal(lit)
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
			state := f.parsePrintfVerb(call, name, format[i:], firstArg, argNum)
			if state == nil {
				return
			}
			w = len(state.format)
			if state.indexed {
				indexed = true
			}
			if !f.okPrintfArg(call, state) { // One error per format is enough.
				return
			}
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

// parseFlags accepts any printf flags.
func (s *formatState) parseFlags() {
	for s.nbytes < len(s.format) {
		switch c := s.format[s.nbytes]; c {
		case '#', '0', '+', '-', ' ':
			s.flags = append(s.flags, c)
			s.nbytes++
		default:
			return
		}
	}
}

// scanNum advances through a decimal number if present.
func (s *formatState) scanNum() {
	for ; s.nbytes < len(s.format); s.nbytes++ {
		c := s.format[s.nbytes]
		if c < '0' || '9' < c {
			return
		}
	}
}

// parseIndex scans an index expression. It returns false if there is a syntax error.
func (s *formatState) parseIndex() bool {
	if s.nbytes == len(s.format) || s.format[s.nbytes] != '[' {
		return true
	}
	// Argument index present.
	s.indexed = true
	s.nbytes++ // skip '['
	start := s.nbytes
	s.scanNum()
	if s.nbytes == len(s.format) || s.nbytes == start || s.format[s.nbytes] != ']' {
		s.file.Badf(s.call.Pos(), "illegal syntax for printf argument index")
		return false
	}
	arg32, err := strconv.ParseInt(s.format[start:s.nbytes], 10, 32)
	if err != nil {
		s.file.Badf(s.call.Pos(), "illegal syntax for printf argument index: %s", err)
		return false
	}
	s.nbytes++ // skip ']'
	arg := int(arg32)
	arg += s.firstArg - 1 // We want to zero-index the actual arguments.
	s.argNum = arg
	s.indexPending = true
	return true
}

// parseNum scans a width or precision (or *). It returns false if there's a bad index expression.
func (s *formatState) parseNum() bool {
	if s.nbytes < len(s.format) && s.format[s.nbytes] == '*' {
		if s.indexPending { // Absorb it.
			s.indexPending = false
		}
		s.nbytes++
		s.argNums = append(s.argNums, s.argNum)
		s.argNum++
	} else {
		s.scanNum()
	}
	return true
}

// parsePrecision scans for a precision. It returns false if there's a bad index expression.
func (s *formatState) parsePrecision() bool {
	// If there's a period, there may be a precision.
	if s.nbytes < len(s.format) && s.format[s.nbytes] == '.' {
		s.flags = append(s.flags, '.') // Treat precision as a flag.
		s.nbytes++
		if !s.parseIndex() {
			return false
		}
		if !s.parseNum() {
			return false
		}
	}
	return true
}

// parsePrintfVerb looks the formatting directive that begins the format string
// and returns a formatState that encodes what the directive wants, without looking
// at the actual arguments present in the call. The result is nil if there is an error.
func (f *File) parsePrintfVerb(call *ast.CallExpr, name, format string, firstArg, argNum int) *formatState {
	state := &formatState{
		format:   format,
		name:     name,
		flags:    make([]byte, 0, 5),
		argNum:   argNum,
		argNums:  make([]int, 0, 1),
		nbytes:   1, // There's guaranteed to be a percent sign.
		indexed:  false,
		firstArg: firstArg,
		file:     f,
		call:     call,
	}
	// There may be flags.
	state.parseFlags()
	indexPending := false
	// There may be an index.
	if !state.parseIndex() {
		return nil
	}
	// There may be a width.
	if !state.parseNum() {
		return nil
	}
	// There may be a precision.
	if !state.parsePrecision() {
		return nil
	}
	// Now a verb, possibly prefixed by an index (which we may already have).
	if !indexPending && !state.parseIndex() {
		return nil
	}
	if state.nbytes == len(state.format) {
		f.Badf(call.Pos(), "missing verb at end of format string in %s call", name)
		return nil
	}
	verb, w := utf8.DecodeRuneInString(state.format[state.nbytes:])
	state.verb = verb
	state.nbytes += w
	if verb != '%' {
		state.argNums = append(state.argNums, state.argNum)
	}
	state.format = state.format[:state.nbytes]
	return state
}

// printfArgType encodes the types of expressions a printf verb accepts. It is a bitmask.
type printfArgType int

const (
	argBool printfArgType = 1 << iota
	argInt
	argRune
	argString
	argFloat
	argComplex
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
	{'b', numFlag, argInt | argFloat | argComplex},
	{'c', "-", argRune | argInt},
	{'d', numFlag, argInt},
	{'e', numFlag, argFloat | argComplex},
	{'E', numFlag, argFloat | argComplex},
	{'f', numFlag, argFloat | argComplex},
	{'F', numFlag, argFloat | argComplex},
	{'g', numFlag, argFloat | argComplex},
	{'G', numFlag, argFloat | argComplex},
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

// okPrintfArg compares the formatState to the arguments actually present,
// reporting any discrepancies it can discern. If the final argument is ellipsissed,
// there's little it can do for that.
func (f *File) okPrintfArg(call *ast.CallExpr, state *formatState) (ok bool) {
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
		return false
	}
	for _, flag := range state.flags {
		if !strings.ContainsRune(v.flags, rune(flag)) {
			f.Badf(call.Pos(), "unrecognized printf flag for verb %q: %q", state.verb, flag)
			return false
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
		if !f.argCanBeChecked(call, i, true, state) {
			return
		}
		arg := call.Args[argNum]
		if !f.matchArgType(argInt, nil, arg) {
			f.Badf(call.Pos(), "arg %s for * in printf format not of type int", f.gofmt(arg))
			return false
		}
	}
	if state.verb == '%' {
		return true
	}
	argNum := state.argNums[len(state.argNums)-1]
	if !f.argCanBeChecked(call, len(state.argNums)-1, false, state) {
		return false
	}
	arg := call.Args[argNum]
	if !f.matchArgType(v.typ, nil, arg) {
		typeString := ""
		if typ := f.pkg.types[arg].Type; typ != nil {
			typeString = typ.String()
		}
		f.Badf(call.Pos(), "arg %s for printf verb %%%c of wrong type: %s", f.gofmt(arg), state.verb, typeString)
		return false
	}
	if v.typ&argString != 0 && v.verb != 'T' && !bytes.Contains(state.flags, []byte{'#'}) && f.recursiveStringer(arg) {
		f.Badf(call.Pos(), "arg %s for printf causes recursive call to String method", f.gofmt(arg))
		return false
	}
	return true
}

// recursiveStringer reports whether the provided argument is r or &r for the
// fmt.Stringer receiver identifier r.
func (f *File) recursiveStringer(e ast.Expr) bool {
	if len(f.stringers) == 0 {
		return false
	}
	var obj *ast.Object
	switch e := e.(type) {
	case *ast.Ident:
		obj = e.Obj
	case *ast.UnaryExpr:
		if id, ok := e.X.(*ast.Ident); ok && e.Op == token.AND {
			obj = id.Obj
		}
	}

	// It's unlikely to be a recursive stringer if it has a Format method.
	if typ := f.pkg.types[e].Type; typ != nil {
		// Not a perfect match; see issue 6259.
		if f.hasMethod(typ, "Format") {
			return false
		}
	}

	// We compare the underlying Object, which checks that the identifier
	// is the one we declared as the receiver for the String method in
	// which this printf appears.
	return f.stringers[obj]
}

// argCanBeChecked reports whether the specified argument is statically present;
// it may be beyond the list of arguments or in a terminal slice... argument, which
// means we can't see it.
func (f *File) argCanBeChecked(call *ast.CallExpr, formatArg int, isStar bool, state *formatState) bool {
	argNum := state.argNums[formatArg]
	if argNum < 0 {
		// Shouldn't happen, so catch it with prejudice.
		panic("negative arg num")
	}
	if argNum == 0 {
		f.Badf(call.Pos(), `index value [0] for %s("%s"); indexes start at 1`, state.name, state.format)
		return false
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
	// There are bad indexes in the format or there are fewer arguments than the format needs.
	// This is the argument number relative to the format: Printf("%s", "hi") will give 1 for the "hi".
	arg := argNum - state.firstArg + 1 // People think of arguments as 1-indexed.
	f.Badf(call.Pos(), `missing argument for %s("%s"): format reads arg %d, have only %d args`, state.name, state.format, arg, len(call.Args)-state.firstArg)
	return false
}

// checkPrint checks a call to an unformatted print routine such as Println.
// call.Args[firstArg] is the first argument to be printed.
func (f *File) checkPrint(call *ast.CallExpr, name string, firstArg int) {
	isLn := strings.HasSuffix(name, "ln")
	isF := strings.HasPrefix(name, "F")
	args := call.Args
	if name == "Log" && len(args) > 0 {
		// Special case: Don't complain about math.Log or cmplx.Log.
		// Not strictly necessary because the only complaint likely is for Log("%d")
		// but it feels wrong to check that math.Log is a good print function.
		if sel, ok := args[0].(*ast.SelectorExpr); ok {
			if x, ok := sel.X.(*ast.Ident); ok {
				if x.Name == "math" || x.Name == "cmplx" {
					return
				}
			}
		}
	}
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
	for _, arg := range args {
		if f.recursiveStringer(arg) {
			f.Badf(call.Pos(), "arg %s for print causes recursive call to String method", f.gofmt(arg))
		}
	}
}
