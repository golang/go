// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printf

import (
	"bytes"
	_ "embed"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"reflect"
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
	"golang.org/x/tools/internal/aliases"
	"golang.org/x/tools/internal/typeparams"
)

func init() {
	Analyzer.Flags.Var(isPrint, "funcs", "comma-separated list of print function names to check")
}

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:       "printf",
	Doc:        analysisutil.MustExtractDoc(doc, "printf"),
	URL:        "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/printf",
	Requires:   []*analysis.Analyzer{inspect.Analyzer},
	Run:        run,
	ResultType: reflect.TypeOf((*Result)(nil)),
	FactTypes:  []analysis.Fact{new(isWrapper)},
}

// Kind is a kind of fmt function behavior.
type Kind int

const (
	KindNone   Kind = iota // not a fmt wrapper function
	KindPrint              // function behaves like fmt.Print
	KindPrintf             // function behaves like fmt.Printf
	KindErrorf             // function behaves like fmt.Errorf
)

func (kind Kind) String() string {
	switch kind {
	case KindPrint:
		return "print"
	case KindPrintf:
		return "printf"
	case KindErrorf:
		return "errorf"
	}
	return ""
}

// Result is the printf analyzer's result type. Clients may query the result
// to learn whether a function behaves like fmt.Print or fmt.Printf.
type Result struct {
	funcs map[*types.Func]Kind
}

// Kind reports whether fn behaves like fmt.Print or fmt.Printf.
func (r *Result) Kind(fn *types.Func) Kind {
	_, ok := isPrint[fn.FullName()]
	if !ok {
		// Next look up just "printf", for use with -printf.funcs.
		_, ok = isPrint[strings.ToLower(fn.Name())]
	}
	if ok {
		if strings.HasSuffix(fn.Name(), "f") {
			return KindPrintf
		} else {
			return KindPrint
		}
	}

	return r.funcs[fn]
}

// isWrapper is a fact indicating that a function is a print or printf wrapper.
type isWrapper struct{ Kind Kind }

func (f *isWrapper) AFact() {}

func (f *isWrapper) String() string {
	switch f.Kind {
	case KindPrintf:
		return "printfWrapper"
	case KindPrint:
		return "printWrapper"
	case KindErrorf:
		return "errorfWrapper"
	default:
		return "unknownWrapper"
	}
}

func run(pass *analysis.Pass) (interface{}, error) {
	res := &Result{
		funcs: make(map[*types.Func]Kind),
	}
	findPrintfLike(pass, res)
	checkCall(pass)
	return res, nil
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
	fn, ok := info.Defs[fdecl.Name].(*types.Func)
	// Type information may be incomplete.
	if !ok {
		return nil
	}

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
func findPrintfLike(pass *analysis.Pass, res *Result) (interface{}, error) {
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
				checkPrintfFwd(pass, w, call, kind, res)
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

// checkPrintfFwd checks that a printf-forwarding wrapper is forwarding correctly.
// It diagnoses writing fmt.Printf(format, args) instead of fmt.Printf(format, args...).
func checkPrintfFwd(pass *analysis.Pass, w *printfWrapper, call *ast.CallExpr, kind Kind, res *Result) {
	matched := kind == KindPrint ||
		kind != KindNone && len(call.Args) >= 2 && match(pass.TypesInfo, call.Args[len(call.Args)-2], w.format)
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
			//   func foo(arg1 string, arg2 ...interface{}) {
			//       fmt.Printf("%s %v", arg1, arg2)
			//   }
			return
		}
		desc := "printf"
		if kind == KindPrint {
			desc = "print"
		}
		pass.ReportRangef(call, "missing ... in args forwarded to %s-like function", desc)
		return
	}
	fn := w.obj
	var fact isWrapper
	if !pass.ImportObjectFact(fn, &fact) {
		fact.Kind = kind
		pass.ExportObjectFact(fn, &fact)
		res.funcs[fn] = kind
		for _, caller := range w.callers {
			checkPrintfFwd(pass, caller.w, caller.call, kind, res)
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
var isPrint = stringSet{
	"fmt.Appendf":  true,
	"fmt.Append":   true,
	"fmt.Appendln": true,
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
	return stringConstantExpr(pass, call.Args[idx])
}

// stringConstantExpr returns expression's string constant value.
//
// ("", false) is returned if expression isn't a string
// constant.
func stringConstantExpr(pass *analysis.Pass, expr ast.Expr) (string, bool) {
	lit := pass.TypesInfo.Types[expr].Value
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
		case KindPrintf, KindErrorf:
			checkPrintf(pass, kind, call, fn)
		case KindPrint:
			checkPrint(pass, call, fn)
		}
	})
}

func printfNameAndKind(pass *analysis.Pass, call *ast.CallExpr) (fn *types.Func, kind Kind) {
	fn, _ = typeutil.Callee(pass.TypesInfo, call).(*types.Func)
	if fn == nil {
		return nil, 0
	}

	_, ok := isPrint[fn.FullName()]
	if !ok {
		// Next look up just "printf", for use with -printf.funcs.
		_, ok = isPrint[strings.ToLower(fn.Name())]
	}
	if ok {
		if fn.FullName() == "fmt.Errorf" {
			kind = KindErrorf
		} else if strings.HasSuffix(fn.Name(), "f") {
			kind = KindPrintf
		} else {
			kind = KindPrint
		}
		return fn, kind
	}

	var fact isWrapper
	if pass.ImportObjectFact(fn, &fact) {
		return fn, fact.Kind
	}

	return fn, KindNone
}

// isFormatter reports whether t could satisfy fmt.Formatter.
// The only interface method to look for is "Format(State, rune)".
func isFormatter(typ types.Type) bool {
	// If the type is an interface, the value it holds might satisfy fmt.Formatter.
	if _, ok := typ.Underlying().(*types.Interface); ok {
		// Don't assume type parameters could be formatters. With the greater
		// expressiveness of constraint interface syntax we expect more type safety
		// when using type parameters.
		if !typeparams.IsTypeParam(typ) {
			return true
		}
	}
	obj, _, _ := types.LookupFieldOrMethod(typ, false, nil, "Format")
	fn, ok := obj.(*types.Func)
	if !ok {
		return false
	}
	sig := fn.Type().(*types.Signature)
	return sig.Params().Len() == 2 &&
		sig.Results().Len() == 0 &&
		analysisutil.IsNamedType(sig.Params().At(0).Type(), "fmt", "State") &&
		types.Identical(sig.Params().At(1).Type(), types.Typ[types.Rune])
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
func checkPrintf(pass *analysis.Pass, kind Kind, call *ast.CallExpr, fn *types.Func) {
	format, idx := formatString(pass, call)
	if idx < 0 {
		if false {
			pass.Reportf(call.Lparen, "can't check non-constant format in call to %s", fn.FullName())
		}
		return
	}

	firstArg := idx + 1 // Arguments are immediately after format string.
	if !strings.Contains(format, "%") {
		if len(call.Args) > firstArg {
			pass.Reportf(call.Lparen, "%s call has arguments but no formatting directives", fn.FullName())
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
		state := parsePrintfVerb(pass, call, fn.FullName(), format[i:], firstArg, argNum)
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
		if state.verb == 'w' {
			switch kind {
			case KindNone, KindPrint, KindPrintf:
				pass.Reportf(call.Pos(), "%s does not support error-wrapping directive %%w", state.name)
				return
			}
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
		pass.ReportRangef(call, "%s call needs %v but has %v", fn.FullName(), count(expect, "arg"), count(numArgs, "arg"))
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
		ok = false // syntax error is either missing "]" or invalid index.
		s.nbytes = strings.Index(s.format[start:], "]")
		if s.nbytes < 0 {
			s.pass.ReportRangef(s.call, "%s format %s is missing closing ]", s.name, s.format)
			return false
		}
		s.nbytes = s.nbytes + start
	}
	arg32, err := strconv.ParseInt(s.format[start:s.nbytes], 10, 32)
	if err != nil || !ok || arg32 <= 0 || arg32 > int64(len(s.call.Args)-s.firstArg) {
		s.pass.ReportRangef(s.call, "%s format has invalid argument index [%s]", s.name, s.format[start:s.nbytes])
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
		pass.ReportRangef(call.Fun, "%s format %s is missing verb at end of string", name, state.format)
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
	argError
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
	{'b', sharpNumFlag, argInt | argFloat | argComplex | argPointer},
	{'c', "-", argRune | argInt},
	{'d', numFlag, argInt | argPointer},
	{'e', sharpNumFlag, argFloat | argComplex},
	{'E', sharpNumFlag, argFloat | argComplex},
	{'f', sharpNumFlag, argFloat | argComplex},
	{'F', sharpNumFlag, argFloat | argComplex},
	{'g', sharpNumFlag, argFloat | argComplex},
	{'G', sharpNumFlag, argFloat | argComplex},
	{'o', sharpNumFlag, argInt | argPointer},
	{'O', sharpNumFlag, argInt | argPointer},
	{'p', "-#", argPointer},
	{'q', " -+.0#", argRune | argInt | argString},
	{'s', " -+.0", argString},
	{'t', "-", argBool},
	{'T', "-", anyType},
	{'U', "-#", argRune | argInt},
	{'v', allFlags, anyType},
	{'w', allFlags, argError},
	{'x', sharpNumFlag, argRune | argInt | argString | argPointer | argFloat | argComplex},
	{'X', sharpNumFlag, argRune | argInt | argString | argPointer | argFloat | argComplex},
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

	// Could current arg implement fmt.Formatter?
	// Skip check for the %w verb, which requires an error.
	formatter := false
	if v.typ != argError && state.argNum < len(call.Args) {
		if tv, ok := pass.TypesInfo.Types[call.Args[state.argNum]]; ok {
			formatter = isFormatter(tv.Type)
		}
	}

	if !formatter {
		if !found {
			pass.ReportRangef(call, "%s format %s has unknown verb %c", state.name, state.format, state.verb)
			return false
		}
		for _, flag := range state.flags {
			// TODO: Disable complaint about '0' for Go 1.10. To be fixed properly in 1.11.
			// See issues 23598 and 23605.
			if flag == '0' {
				continue
			}
			if !strings.ContainsRune(v.flags, rune(flag)) {
				pass.ReportRangef(call, "%s format %s has unrecognized flag %c", state.name, state.format, flag)
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
		if reason, ok := matchArgType(pass, argInt, arg); !ok {
			details := ""
			if reason != "" {
				details = " (" + reason + ")"
			}
			pass.ReportRangef(call, "%s format %s uses non-int %s%s as argument of *", state.name, state.format, analysisutil.Format(pass.Fset, arg), details)
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
		pass.ReportRangef(call, "%s format %s arg %s is a func value, not called", state.name, state.format, analysisutil.Format(pass.Fset, arg))
		return false
	}
	if reason, ok := matchArgType(pass, v.typ, arg); !ok {
		typeString := ""
		if typ := pass.TypesInfo.Types[arg].Type; typ != nil {
			typeString = typ.String()
		}
		details := ""
		if reason != "" {
			details = " (" + reason + ")"
		}
		pass.ReportRangef(call, "%s format %s has arg %s of wrong type %s%s", state.name, state.format, analysisutil.Format(pass.Fset, arg), typeString, details)
		return false
	}
	if v.typ&argString != 0 && v.verb != 'T' && !bytes.Contains(state.flags, []byte{'#'}) {
		if methodName, ok := recursiveStringer(pass, arg); ok {
			pass.ReportRangef(call, "%s format %s with arg %s causes recursive %s method call", state.name, state.format, analysisutil.Format(pass.Fset, arg), methodName)
			return false
		}
	}
	return true
}

// recursiveStringer reports whether the argument e is a potential
// recursive call to stringer or is an error, such as t and &t in these examples:
//
//	func (t *T) String() string { printf("%s",  t) }
//	func (t  T) Error() string { printf("%s",  t) }
//	func (t  T) String() string { printf("%s", &t) }
func recursiveStringer(pass *analysis.Pass, e ast.Expr) (string, bool) {
	typ := pass.TypesInfo.Types[e].Type

	// It's unlikely to be a recursive stringer if it has a Format method.
	if isFormatter(typ) {
		return "", false
	}

	// Does e allow e.String() or e.Error()?
	strObj, _, _ := types.LookupFieldOrMethod(typ, false, pass.Pkg, "String")
	strMethod, strOk := strObj.(*types.Func)
	errObj, _, _ := types.LookupFieldOrMethod(typ, false, pass.Pkg, "Error")
	errMethod, errOk := errObj.(*types.Func)
	if !strOk && !errOk {
		return "", false
	}

	// inScope returns true if e is in the scope of f.
	inScope := func(e ast.Expr, f *types.Func) bool {
		return f.Scope() != nil && f.Scope().Contains(e.Pos())
	}

	// Is the expression e within the body of that String or Error method?
	var method *types.Func
	if strOk && strMethod.Pkg() == pass.Pkg && inScope(e, strMethod) {
		method = strMethod
	} else if errOk && errMethod.Pkg() == pass.Pkg && inScope(e, errMethod) {
		method = errMethod
	} else {
		return "", false
	}

	sig := method.Type().(*types.Signature)
	if !isStringer(sig) {
		return "", false
	}

	// Is it the receiver r, or &r?
	if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.AND {
		e = u.X // strip off & from &r
	}
	if id, ok := e.(*ast.Ident); ok {
		if pass.TypesInfo.Uses[id] == sig.Recv() {
			return method.FullName(), true
		}
	}
	return "", false
}

// isStringer reports whether the method signature matches the String() definition in fmt.Stringer.
func isStringer(sig *types.Signature) bool {
	return sig.Params().Len() == 0 &&
		sig.Results().Len() == 1 &&
		sig.Results().At(0).Type() == types.Typ[types.String]
}

// isFunctionValue reports whether the expression is a function as opposed to a function call.
// It is almost always a mistake to print a function value.
func isFunctionValue(pass *analysis.Pass, e ast.Expr) bool {
	if typ := pass.TypesInfo.Types[e].Type; typ != nil {
		// Don't call Underlying: a named func type with a String method is ok.
		// TODO(adonovan): it would be more precise to check isStringer.
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
	pass.ReportRangef(call, "%s format %s reads arg #%d, but call has %v", state.name, state.format, arg, count(len(call.Args)-state.firstArg, "arg"))
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
	if sig, ok := typ.Underlying().(*types.Signature); ok {
		if !sig.Variadic() {
			// Skip checking non-variadic functions.
			return
		}
		params := sig.Params()
		firstArg = params.Len() - 1

		typ := params.At(firstArg).Type()
		typ = typ.(*types.Slice).Elem()
		it, ok := aliases.Unalias(typ).(*types.Interface)
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
					pass.ReportRangef(call, "%s does not take io.Writer but has first arg %s", fn.FullName(), analysisutil.Format(pass.Fset, call.Args[0]))
				}
			}
		}
	}

	arg := args[0]
	if s, ok := stringConstantExpr(pass, arg); ok {
		// Ignore trailing % character
		// The % in "abc 0.0%" couldn't be a formatting directive.
		s = strings.TrimSuffix(s, "%")
		if strings.Contains(s, "%") {
			m := printFormatRE.FindStringSubmatch(s)
			if m != nil {
				pass.ReportRangef(call, "%s call has possible Printf formatting directive %s", fn.FullName(), m[0])
			}
		}
	}
	if strings.HasSuffix(fn.Name(), "ln") {
		// The last item, if a string, should not have a newline.
		arg = args[len(args)-1]
		if s, ok := stringConstantExpr(pass, arg); ok {
			if strings.HasSuffix(s, "\n") {
				pass.ReportRangef(call, "%s arg list ends with redundant newline", fn.FullName())
			}
		}
	}
	for _, arg := range args {
		if isFunctionValue(pass, arg) {
			pass.ReportRangef(call, "%s arg %s is a func value, not called", fn.FullName(), analysisutil.Format(pass.Fset, arg))
		}
		if methodName, ok := recursiveStringer(pass, arg); ok {
			pass.ReportRangef(call, "%s arg %s causes recursive call to %s method", fn.FullName(), analysisutil.Format(pass.Fset, arg), methodName)
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
