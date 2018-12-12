// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the printf-checker.

package printf

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

func init() {
	Analyzer.Flags.Var(isPrint, "funcs", "comma-separated list of print function names to check")
}

var Analyzer = &analysis.Analyzer{
	Name:      "printf",
	Doc:       doc,
	Requires:  []*analysis.Analyzer{inspect.Analyzer},
	Run:       run,
	FactTypes: []analysis.Fact{new(isWrapper)},
}

const doc = `check consistency of Printf format strings and arguments

The check applies to known functions (for example, those in package fmt)
as well as any detected wrappers of known functions.

A function that wants to avail itself of printf checking but is not
found by this analyzer's heuristics (for example, due to use of
dynamic calls) can insert a bogus call:

	if false {
		_ = fmt.Sprintf(format, args...) // enable printf checking
	}

The -funcs flag specifies a comma-separated list of names of additional
known formatting functions or methods. If the name contains a period,
it must denote a specific function using one of the following forms:

	dir/pkg.Function
	dir/pkg.Type.Method
	(*dir/pkg.Type).Method

Otherwise the name is interpreted as a case-insensitive unqualified
identifier such as "errorf". Either way, if a listed name ends in f, the
function is assumed to be Printf-like, taking a format string before the
argument list. Otherwise it is assumed to be Print-like, taking a list
of arguments with no format string.
`

// isWrapper is a fact indicating that a function is a print or printf wrapper.
type isWrapper struct{ Printf bool }

func (f *isWrapper) AFact() {}

func (f *isWrapper) String() string {
	if f.Printf {
		return "printfWrapper"
	} else {
		return "printWrapper"
	}
}

func run(pass *analysis.Pass) (interface{}, error) {
	findPrintfLike(pass)
	checkCall(pass)
	return nil, nil
}

type printfWrapper struct {
	obj     *types.Func
	fdecl   *ast.FuncDecl
	format  *types.Var
	args    *types.Var
	callers []printfCaller
	failed  bool // if true, not a printf wrapper
}

type printfCaller struct {
	w    *printfWrapper
	call *ast.CallExpr
}

// maybePrintfWrapper decides whether decl (a declared function) may be a wrapper
// around a fmt.Printf or fmt.Print function. If so it returns a printfWrapper
// function describing the declaration. Later processing will analyze the
// graph of potential printf wrappers to pick out the ones that are true wrappers.
// A function may be a Printf or Print wrapper if its last argument is ...interface{}.
// If the next-to-last argument is a string, then this may be a Printf wrapper.
// Otherwise it may be a Print wrapper.
func maybePrintfWrapper(info *types.Info, decl ast.Decl) *printfWrapper {
	// Look for functions with final argument type ...interface{}.
	fdecl, ok := decl.(*ast.FuncDecl)
	if !ok || fdecl.Body == nil {
		return nil
	}
	fn := info.Defs[fdecl.Name].(*types.Func)

	sig := fn.Type().(*types.Signature)
	if !sig.Variadic() {
		return nil // not variadic
	}

	params := sig.Params()
	nparams := params.Len() // variadic => nonzero

	args := params.At(nparams - 1)
	iface, ok := args.Type().(*types.Slice).Elem().(*types.Interface)
	if !ok || !iface.Empty() {
		return nil // final (args) param is not ...interface{}
	}

	// Is second last param 'format string'?
	var format *types.Var
	if nparams >= 2 {
		if p := params.At(nparams - 2); p.Type() == types.Typ[types.String] {
			format = p
		}
	}

	return &printfWrapper{
		obj:    fn,
		fdecl:  fdecl,
		format: format,
		args:   args,
	}
}

// findPrintfLike scans the entire package to find printf-like functions.
func findPrintfLike(pass *analysis.Pass) (interface{}, error) {
	// Gather potential wrappers and call graph between them.
	byObj := make(map[*types.Func]*printfWrapper)
	var wrappers []*printfWrapper
	for _, file := range pass.Files {
		for _, decl := range file.Decls {
			w := maybePrintfWrapper(pass.TypesInfo, decl)
			if w == nil {
				continue
			}
			byObj[w.obj] = w
			wrappers = append(wrappers, w)
		}
	}

	// Walk the graph to figure out which are really printf wrappers.
	for _, w := range wrappers {
		// Scan function for calls that could be to other printf-like functions.
		ast.Inspect(w.fdecl.Body, func(n ast.Node) bool {
			if w.failed {
				return false
			}

			// TODO: Relax these checks; issue 26555.
			if assign, ok := n.(*ast.AssignStmt); ok {
				for _, lhs := range assign.Lhs {
					if match(pass.TypesInfo, lhs, w.format) ||
						match(pass.TypesInfo, lhs, w.args) {
						// Modifies the format
						// string or args in
						// some way, so not a
						// simple wrapper.
						w.failed = true
						return false
					}
				}
			}
			if un, ok := n.(*ast.UnaryExpr); ok && un.Op == token.AND {
				if match(pass.TypesInfo, un.X, w.format) ||
					match(pass.TypesInfo, un.X, w.args) {
					// Taking the address of the
					// format string or args,
					// so not a simple wrapper.
					w.failed = true
					return false
				}
			}

			call, ok := n.(*ast.CallExpr)
			if !ok || len(call.Args) == 0 || !match(pass.TypesInfo, call.Args[len(call.Args)-1], w.args) {
				return true
			}

			fn, kind := printfNameAndKind(pass, call)
			if kind != 0 {
				checkPrintfFwd(pass, w, call, kind)
				return true
			}

			// If the call is to another function in this package,
			// maybe we will find out it is printf-like later.
			// Remember this call for later checking.
			if fn != nil && fn.Pkg() == pass.Pkg && byObj[fn] != nil {
				callee := byObj[fn]
				callee.callers = append(callee.callers, printfCaller{w, call})
			}

			return true
		})
	}
	return nil, nil
}

func match(info *types.Info, arg ast.Expr, param *types.Var) bool {
	id, ok := arg.(*ast.Ident)
	return ok && info.ObjectOf(id) == param
}

const (
	kindPrintf = 1
	kindPrint  = 2
)

// checkPrintfFwd checks that a printf-forwarding wrapper is forwarding correctly.
// It diagnoses writing fmt.Printf(format, args) instead of fmt.Printf(format, args...).
func checkPrintfFwd(pass *analysis.Pass, w *printfWrapper, call *ast.CallExpr, kind int) {
	matched := kind == kindPrint ||
		kind == kindPrintf && len(call.Args) >= 2 && match(pass.TypesInfo, call.Args[len(call.Args)-2], w.format)
	if !matched {
		return
	}

	if !call.Ellipsis.IsValid() {
		typ, ok := pass.TypesInfo.Types[call.Fun].Type.(*types.Signature)
		if !ok {
			return
		}
		if len(call.Args) > typ.Params().Len() {
			// If we're passing more arguments than what the
			// print/printf function can take, adding an ellipsis
			// would break the program. For example:
			//
			//   func foo(arg1 string, arg2 ...interface{} {
			//       fmt.Printf("%s %v", arg1, arg2)
			//   }
			return
		}
		desc := "printf"
		if kind == kindPrint {
			desc = "print"
		}
		pass.Reportf(call.Pos(), "missing ... in args forwarded to %s-like function", desc)
		return
	}
	fn := w.obj
	var fact isWrapper
	if !pass.ImportObjectFact(fn, &fact) {
		fact.Printf = kind == kindPrintf
		pass.ExportObjectFact(fn, &fact)
		for _, caller := range w.callers {
			checkPrintfFwd(pass, caller.w, caller.call, kind)
		}
	}
}

// isPrint records the print functions.
// If a key ends in 'f' then it is assumed to be a formatted print.
//
// Keys are either values returned by (*types.Func).FullName,
// or case-insensitive identifiers such as "errorf".
//
// The -funcs flag adds to this set.
//
// The set below includes facts for many important standard library
// functions, even though the analysis is capable of deducing that, for
// example, fmt.Printf forwards to fmt.Fprintf. We avoid relying on the
// driver applying analyzers to standard packages because "go vet" does
// not do so with gccgo, and nor do some other build systems.
// TODO(adonovan): eliminate the redundant facts once this restriction
// is lifted.
//
var isPrint = stringSet{
	"fmt.Errorf":   true,
	"fmt.Fprint":   true,
	"fmt.Fprintf":  true,
	"fmt.Fprintln": true,
	"fmt.Print":    true,
	"fmt.Printf":   true,
	"fmt.Println":  true,
	"fmt.Sprint":   true,
	"fmt.Sprintf":  true,
	"fmt.Sprintln": true,

	"runtime/trace.Logf": true,

	"log.Print":             true,
	"log.Printf":            true,
	"log.Println":           true,
	"log.Fatal":             true,
	"log.Fatalf":            true,
	"log.Fatalln":           true,
	"log.Panic":             true,
	"log.Panicf":            true,
	"log.Panicln":           true,
	"(*log.Logger).Fatal":   true,
	"(*log.Logger).Fatalf":  true,
	"(*log.Logger).Fatalln": true,
	"(*log.Logger).Panic":   true,
	"(*log.Logger).Panicf":  true,
	"(*log.Logger).Panicln": true,
	"(*log.Logger).Print":   true,
	"(*log.Logger).Printf":  true,
	"(*log.Logger).Println": true,

	"(*testing.common).Error":  true,
	"(*testing.common).Errorf": true,
	"(*testing.common).Fatal":  true,
	"(*testing.common).Fatalf": true,
	"(*testing.common).Log":    true,
	"(*testing.common).Logf":   true,
	"(*testing.common).Skip":   true,
	"(*testing.common).Skipf":  true,
	// *testing.T and B are detected by induction, but testing.TB is
	// an interface and the inference can't follow dynamic calls.
	"(testing.TB).Error":  true,
	"(testing.TB).Errorf": true,
	"(testing.TB).Fatal":  true,
	"(testing.TB).Fatalf": true,
	"(testing.TB).Log":    true,
	"(testing.TB).Logf":   true,
	"(testing.TB).Skip":   true,
	"(testing.TB).Skipf":  true,
}

// formatString returns the format string argument and its index within
// the given printf-like call expression.
//
// The last parameter before variadic arguments is assumed to be
// a format string.
//
// The first string literal or string constant is assumed to be a format string
// if the call's signature cannot be determined.
//
// If it cannot find any format string parameter, it returns ("", -1).
func formatString(pass *analysis.Pass, call *ast.CallExpr) (format string, idx int) {
	typ := pass.TypesInfo.Types[call.Fun].Type
	if typ != nil {
		if sig, ok := typ.(*types.Signature); ok {
			if !sig.Variadic() {
				// Skip checking non-variadic functions.
				return "", -1
			}
			idx := sig.Params().Len() - 2
			if idx < 0 {
				// Skip checking variadic functions without
				// fixed arguments.
				return "", -1
			}
			s, ok := stringConstantArg(pass, call, idx)
			if !ok {
				// The last argument before variadic args isn't a string.
				return "", -1
			}
			return s, idx
		}
	}

	// Cannot determine call's signature. Fall back to scanning for the first
	// string constant in the call.
	for idx := range call.Args {
		if s, ok := stringConstantArg(pass, call, idx); ok {
			return s, idx
		}
		if pass.TypesInfo.Types[call.Args[idx]].Type == types.Typ[types.String] {
			// Skip checking a call with a non-constant format
			// string argument, since its contents are unavailable
			// for validation.
			return "", -1
		}
	}
	return "", -1
}

// stringConstantArg returns call's string constant argument at the index idx.
//
// ("", false) is returned if call's argument at the index idx isn't a string
// constant.
func stringConstantArg(pass *analysis.Pass, call *ast.CallExpr, idx int) (string, bool) {
	if idx >= len(call.Args) {
		return "", false
	}
	arg := call.Args[idx]
	lit := pass.TypesInfo.Types[arg].Value
	if lit != nil && lit.Kind() == constant.String {
		return constant.StringVal(lit), true
	}
	return "", false
}

// checkCall triggers the print-specific checks if the call invokes a print function.
func checkCall(pass *analysis.Pass) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		fn, kind := printfNameAndKind(pass, call)
		switch kind {
		case kindPrintf:
			checkPrintf(pass, call, fn)
		case kindPrint:
			checkPrint(pass, call, fn)
		}
	})
}

func printfNameAndKind(pass *analysis.Pass, call *ast.CallExpr) (fn *types.Func, kind int) {
	fn, _ = typeutil.Callee(pass.TypesInfo, call).(*types.Func)
	if fn == nil {
		return nil, 0
	}

	var fact isWrapper
	if pass.ImportObjectFact(fn, &fact) {
		if fact.Printf {
			return fn, kindPrintf
		} else {
			return fn, kindPrint
		}
	}

	_, ok := isPrint[fn.FullName()]
	if !ok {
		// Next look up just "printf", for use with -printf.funcs.
		_, ok = isPrint[strings.ToLower(fn.Name())]
	}
	if ok {
		if strings.HasSuffix(fn.Name(), "f") {
			kind = kindPrintf
		} else {
			kind = kindPrint
		}
	}
	return fn, kind
}

// isFormatter reports whether t satisfies fmt.Formatter.
// Unlike fmt.Stringer, it's impossible to satisfy fmt.Formatter without importing fmt.
func isFormatter(pass *analysis.Pass, t types.Type) bool {
	for _, imp := range pass.Pkg.Imports() {
		if imp.Path() == "fmt" {
			formatter := imp.Scope().Lookup("Formatter").Type().Underlying().(*types.Interface)
			return types.Implements(t, formatter)
		}
	}
	return false
}

// formatState holds the parsed representation of a printf directive such as "%3.*[4]d".
// It is constructed by parsePrintfVerb.
type formatState struct {
	verb     rune   // the format verb: 'd' for "%d"
	format   string // the full format directive from % through verb, "%.3d".
	name     string // Printf, Sprintf etc.
	flags    []byte // the list of # + etc.
	argNums  []int  // the successive argument numbers that are consumed, adjusted to refer to actual arg in call
	firstArg int    // Index of first argument after the format in the Printf call.
	// Used only during parse.
	pass         *analysis.Pass
	call         *ast.CallExpr
	argNum       int  // Which argument we're expecting to format now.
	hasIndex     bool // Whether the argument is indexed.
	indexPending bool // Whether we have an indexed argument that has not resolved.
	nbytes       int  // number of bytes of the format string consumed.
}

// checkPrintf checks a call to a formatted print routine such as Printf.
func checkPrintf(pass *analysis.Pass, call *ast.CallExpr, fn *types.Func) {
	format, idx := formatString(pass, call)
	if idx < 0 {
		if false {
			pass.Reportf(call.Lparen, "can't check non-constant format in call to %s", fn.Name())
		}
		return
	}

	firstArg := idx + 1 // Arguments are immediately after format string.
	if !strings.Contains(format, "%") {
		if len(call.Args) > firstArg {
			pass.Reportf(call.Lparen, "%s call has arguments but no formatting directives", fn.Name())
		}
		return
	}
	// Hard part: check formats against args.
	argNum := firstArg
	maxArgNum := firstArg
	anyIndex := false
	for i, w := 0, 0; i < len(format); i += w {
		w = 1
		if format[i] != '%' {
			continue
		}
		state := parsePrintfVerb(pass, call, fn.Name(), format[i:], firstArg, argNum)
		if state == nil {
			return
		}
		w = len(state.format)
		if !okPrintfArg(pass, call, state) { // One error per format is enough.
			return
		}
		if state.hasIndex {
			anyIndex = true
		}
		if len(state.argNums) > 0 {
			// Continue with the next sequential argument.
			argNum = state.argNums[len(state.argNums)-1] + 1
		}
		for _, n := range state.argNums {
			if n >= maxArgNum {
				maxArgNum = n + 1
			}
		}
	}
	// Dotdotdot is hard.
	if call.Ellipsis.IsValid() && maxArgNum >= len(call.Args)-1 {
		return
	}
	// If any formats are indexed, extra arguments are ignored.
	if anyIndex {
		return
	}
	// There should be no leftover arguments.
	if maxArgNum != len(call.Args) {
		expect := maxArgNum - firstArg
		numArgs := len(call.Args) - firstArg
		pass.Reportf(call.Pos(), "%s call needs %v but has %v", fn.Name(), count(expect, "arg"), count(numArgs, "arg"))
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
	s.nbytes++ // skip '['
	start := s.nbytes
	s.scanNum()
	ok := true
	if s.nbytes == len(s.format) || s.nbytes == start || s.format[s.nbytes] != ']' {
		ok = false
		s.nbytes = strings.Index(s.format, "]")
		if s.nbytes < 0 {
			s.pass.Reportf(s.call.Pos(), "%s format %s is missing closing ]", s.name, s.format)
			return false
		}
	}
	arg32, err := strconv.ParseInt(s.format[start:s.nbytes], 10, 32)
	if err != nil || !ok || arg32 <= 0 || arg32 > int64(len(s.call.Args)-s.firstArg) {
		s.pass.Reportf(s.call.Pos(), "%s format has invalid argument index [%s]", s.name, s.format[start:s.nbytes])
		return false
	}
	s.nbytes++ // skip ']'
	arg := int(arg32)
	arg += s.firstArg - 1 // We want to zero-index the actual arguments.
	s.argNum = arg
	s.hasIndex = true
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
func parsePrintfVerb(pass *analysis.Pass, call *ast.CallExpr, name, format string, firstArg, argNum int) *formatState {
	state := &formatState{
		format:   format,
		name:     name,
		flags:    make([]byte, 0, 5),
		argNum:   argNum,
		argNums:  make([]int, 0, 1),
		nbytes:   1, // There's guaranteed to be a percent sign.
		firstArg: firstArg,
		pass:     pass,
		call:     call,
	}
	// There may be flags.
	state.parseFlags()
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
	if !state.indexPending && !state.parseIndex() {
		return nil
	}
	if state.nbytes == len(state.format) {
		pass.Reportf(call.Pos(), "%s format %s is missing verb at end of string", name, state.format)
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
var printVerbs = []printVerb{
	// '-' is a width modifier, always valid.
	// '.' is a precision for float, max width for strings.
	// '+' is required sign for numbers, Go format for %v.
	// '#' is alternate format for several verbs.
	// ' ' is spacer for numbers
	{'%', noFlag, 0},
	{'b', numFlag, argInt | argFloat | argComplex | argPointer},
	{'c', "-", argRune | argInt},
	{'d', numFlag, argInt | argPointer},
	{'e', sharpNumFlag, argFloat | argComplex},
	{'E', sharpNumFlag, argFloat | argComplex},
	{'f', sharpNumFlag, argFloat | argComplex},
	{'F', sharpNumFlag, argFloat | argComplex},
	{'g', sharpNumFlag, argFloat | argComplex},
	{'G', sharpNumFlag, argFloat | argComplex},
	{'o', sharpNumFlag, argInt | argPointer},
	{'p', "-#", argPointer},
	{'q', " -+.0#", argRune | argInt | argString},
	{'s', " -+.0", argString},
	{'t', "-", argBool},
	{'T', "-", anyType},
	{'U', "-#", argRune | argInt},
	{'v', allFlags, anyType},
	{'x', sharpNumFlag, argRune | argInt | argString | argPointer},
	{'X', sharpNumFlag, argRune | argInt | argString | argPointer},
}

// okPrintfArg compares the formatState to the arguments actually present,
// reporting any discrepancies it can discern. If the final argument is ellipsissed,
// there's little it can do for that.
func okPrintfArg(pass *analysis.Pass, call *ast.CallExpr, state *formatState) (ok bool) {
	var v printVerb
	found := false
	// Linear scan is fast enough for a small list.
	for _, v = range printVerbs {
		if v.verb == state.verb {
			found = true
			break
		}
	}

	// Does current arg implement fmt.Formatter?
	formatter := false
	if state.argNum < len(call.Args) {
		if tv, ok := pass.TypesInfo.Types[call.Args[state.argNum]]; ok {
			formatter = isFormatter(pass, tv.Type)
		}
	}

	if !formatter {
		if !found {
			pass.Reportf(call.Pos(), "%s format %s has unknown verb %c", state.name, state.format, state.verb)
			return false
		}
		for _, flag := range state.flags {
			// TODO: Disable complaint about '0' for Go 1.10. To be fixed properly in 1.11.
			// See issues 23598 and 23605.
			if flag == '0' {
				continue
			}
			if !strings.ContainsRune(v.flags, rune(flag)) {
				pass.Reportf(call.Pos(), "%s format %s has unrecognized flag %c", state.name, state.format, flag)
				return false
			}
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
		if !argCanBeChecked(pass, call, i, state) {
			return
		}
		arg := call.Args[argNum]
		if !matchArgType(pass, argInt, nil, arg) {
			pass.Reportf(call.Pos(), "%s format %s uses non-int %s as argument of *", state.name, state.format, analysisutil.Format(pass.Fset, arg))
			return false
		}
	}

	if state.verb == '%' || formatter {
		return true
	}
	argNum := state.argNums[len(state.argNums)-1]
	if !argCanBeChecked(pass, call, len(state.argNums)-1, state) {
		return false
	}
	arg := call.Args[argNum]
	if isFunctionValue(pass, arg) && state.verb != 'p' && state.verb != 'T' {
		pass.Reportf(call.Pos(), "%s format %s arg %s is a func value, not called", state.name, state.format, analysisutil.Format(pass.Fset, arg))
		return false
	}
	if !matchArgType(pass, v.typ, nil, arg) {
		typeString := ""
		if typ := pass.TypesInfo.Types[arg].Type; typ != nil {
			typeString = typ.String()
		}
		pass.Reportf(call.Pos(), "%s format %s has arg %s of wrong type %s", state.name, state.format, analysisutil.Format(pass.Fset, arg), typeString)
		return false
	}
	if v.typ&argString != 0 && v.verb != 'T' && !bytes.Contains(state.flags, []byte{'#'}) && recursiveStringer(pass, arg) {
		pass.Reportf(call.Pos(), "%s format %s with arg %s causes recursive String method call", state.name, state.format, analysisutil.Format(pass.Fset, arg))
		return false
	}
	return true
}

// recursiveStringer reports whether the argument e is a potential
// recursive call to stringer, such as t and &t in these examples:
//
// 	func (t *T) String() string { printf("%s",  t) }
// 	func (t  T) String() string { printf("%s",  t) }
// 	func (t  T) String() string { printf("%s", &t) }
//
func recursiveStringer(pass *analysis.Pass, e ast.Expr) bool {
	typ := pass.TypesInfo.Types[e].Type

	// It's unlikely to be a recursive stringer if it has a Format method.
	if isFormatter(pass, typ) {
		return false
	}

	// Does e allow e.String()?
	obj, _, _ := types.LookupFieldOrMethod(typ, false, pass.Pkg, "String")
	stringMethod, ok := obj.(*types.Func)
	if !ok {
		return false
	}

	// Is the expression e within the body of that String method?
	if stringMethod.Pkg() != pass.Pkg || !stringMethod.Scope().Contains(e.Pos()) {
		return false
	}

	// Is it the receiver r, or &r?
	recv := stringMethod.Type().(*types.Signature).Recv()
	if recv == nil {
		return false
	}
	if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.AND {
		e = u.X // strip off & from &r
	}
	if id, ok := e.(*ast.Ident); ok {
		return pass.TypesInfo.Uses[id] == recv
	}
	return false
}

// isFunctionValue reports whether the expression is a function as opposed to a function call.
// It is almost always a mistake to print a function value.
func isFunctionValue(pass *analysis.Pass, e ast.Expr) bool {
	if typ := pass.TypesInfo.Types[e].Type; typ != nil {
		_, ok := typ.(*types.Signature)
		return ok
	}
	return false
}

// argCanBeChecked reports whether the specified argument is statically present;
// it may be beyond the list of arguments or in a terminal slice... argument, which
// means we can't see it.
func argCanBeChecked(pass *analysis.Pass, call *ast.CallExpr, formatArg int, state *formatState) bool {
	argNum := state.argNums[formatArg]
	if argNum <= 0 {
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
	// There are bad indexes in the format or there are fewer arguments than the format needs.
	// This is the argument number relative to the format: Printf("%s", "hi") will give 1 for the "hi".
	arg := argNum - state.firstArg + 1 // People think of arguments as 1-indexed.
	pass.Reportf(call.Pos(), "%s format %s reads arg #%d, but call has %v", state.name, state.format, arg, count(len(call.Args)-state.firstArg, "arg"))
	return false
}

// printFormatRE is the regexp we match and report as a possible format string
// in the first argument to unformatted prints like fmt.Print.
// We exclude the space flag, so that printing a string like "x % y" is not reported as a format.
var printFormatRE = regexp.MustCompile(`%` + flagsRE + numOptRE + `\.?` + numOptRE + indexOptRE + verbRE)

const (
	flagsRE    = `[+\-#]*`
	indexOptRE = `(\[[0-9]+\])?`
	numOptRE   = `([0-9]+|` + indexOptRE + `\*)?`
	verbRE     = `[bcdefgopqstvxEFGTUX]`
)

// checkPrint checks a call to an unformatted print routine such as Println.
func checkPrint(pass *analysis.Pass, call *ast.CallExpr, fn *types.Func) {
	firstArg := 0
	typ := pass.TypesInfo.Types[call.Fun].Type
	if typ == nil {
		// Skip checking functions with unknown type.
		return
	}
	if sig, ok := typ.(*types.Signature); ok {
		if !sig.Variadic() {
			// Skip checking non-variadic functions.
			return
		}
		params := sig.Params()
		firstArg = params.Len() - 1

		typ := params.At(firstArg).Type()
		typ = typ.(*types.Slice).Elem()
		it, ok := typ.(*types.Interface)
		if !ok || !it.Empty() {
			// Skip variadic functions accepting non-interface{} args.
			return
		}
	}
	args := call.Args
	if len(args) <= firstArg {
		// Skip calls without variadic args.
		return
	}
	args = args[firstArg:]

	if firstArg == 0 {
		if sel, ok := call.Args[0].(*ast.SelectorExpr); ok {
			if x, ok := sel.X.(*ast.Ident); ok {
				if x.Name == "os" && strings.HasPrefix(sel.Sel.Name, "Std") {
					pass.Reportf(call.Pos(), "%s does not take io.Writer but has first arg %s", fn.Name(), analysisutil.Format(pass.Fset, call.Args[0]))
				}
			}
		}
	}

	arg := args[0]
	if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
		// Ignore trailing % character in lit.Value.
		// The % in "abc 0.0%" couldn't be a formatting directive.
		s := strings.TrimSuffix(lit.Value, `%"`)
		if strings.Contains(s, "%") {
			m := printFormatRE.FindStringSubmatch(s)
			if m != nil {
				pass.Reportf(call.Pos(), "%s call has possible formatting directive %s", fn.Name(), m[0])
			}
		}
	}
	if strings.HasSuffix(fn.Name(), "ln") {
		// The last item, if a string, should not have a newline.
		arg = args[len(args)-1]
		if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			str, _ := strconv.Unquote(lit.Value)
			if strings.HasSuffix(str, "\n") {
				pass.Reportf(call.Pos(), "%s arg list ends with redundant newline", fn.Name())
			}
		}
	}
	for _, arg := range args {
		if isFunctionValue(pass, arg) {
			pass.Reportf(call.Pos(), "%s arg %s is a func value, not called", fn.Name(), analysisutil.Format(pass.Fset, arg))
		}
		if recursiveStringer(pass, arg) {
			pass.Reportf(call.Pos(), "%s arg %s causes recursive call to String method", fn.Name(), analysisutil.Format(pass.Fset, arg))
		}
	}
}

// count(n, what) returns "1 what" or "N whats"
// (assuming the plural of what is whats).
func count(n int, what string) string {
	if n == 1 {
		return "1 " + what
	}
	return fmt.Sprintf("%d %ss", n, what)
}

// stringSet is a set-of-nonempty-strings-valued flag.
// Note: elements without a '.' get lower-cased.
type stringSet map[string]bool

func (ss stringSet) String() string {
	var list []string
	for name := range ss {
		list = append(list, name)
	}
	sort.Strings(list)
	return strings.Join(list, ",")
}

func (ss stringSet) Set(flag string) error {
	for _, name := range strings.Split(flag, ",") {
		if len(name) == 0 {
			return fmt.Errorf("empty string")
		}
		if !strings.Contains(name, ".") {
			name = strings.ToLower(name)
		}
		ss[name] = true
	}
	return nil
}
