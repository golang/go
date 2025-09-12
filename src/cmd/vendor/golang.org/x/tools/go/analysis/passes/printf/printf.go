// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printf

import (
	_ "embed"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"reflect"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/fmtstr"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/versions"
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

func run(pass *analysis.Pass) (any, error) {
	res := &Result{
		funcs: make(map[*types.Func]Kind),
	}
	findPrintfLike(pass, res)
	checkCalls(pass)
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

	// Check final parameter is "args ...interface{}".
	args := params.At(nparams - 1)
	iface, ok := types.Unalias(args.Type().(*types.Slice).Elem()).(*types.Interface)
	if !ok || !iface.Empty() {
		return nil
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
func findPrintfLike(pass *analysis.Pass, res *Result) (any, error) {
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

// formatStringIndex returns the index of the format string (the last
// non-variadic parameter) within the given printf-like call
// expression, or -1 if unknown.
func formatStringIndex(pass *analysis.Pass, call *ast.CallExpr) int {
	typ := pass.TypesInfo.Types[call.Fun].Type
	if typ == nil {
		return -1 // missing type
	}
	sig, ok := typ.(*types.Signature)
	if !ok {
		return -1 // ill-typed
	}
	if !sig.Variadic() {
		// Skip checking non-variadic functions.
		return -1
	}
	idx := sig.Params().Len() - 2
	if idx < 0 {
		// Skip checking variadic functions without
		// fixed arguments.
		return -1
	}
	return idx
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

// checkCalls triggers the print-specific checks for calls that invoke a print
// function.
func checkCalls(pass *analysis.Pass) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.File)(nil),
		(*ast.CallExpr)(nil),
	}

	var fileVersion string // for selectively suppressing checks; "" if unknown.
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		switch n := n.(type) {
		case *ast.File:
			fileVersion = versions.Lang(versions.FileVersion(pass.TypesInfo, n))

		case *ast.CallExpr:
			fn, kind := printfNameAndKind(pass, n)
			switch kind {
			case KindPrintf, KindErrorf:
				checkPrintf(pass, fileVersion, kind, n, fn.FullName())
			case KindPrint:
				checkPrint(pass, n, fn.FullName())
			}
		}
	})
}

func printfNameAndKind(pass *analysis.Pass, call *ast.CallExpr) (fn *types.Func, kind Kind) {
	fn, _ = typeutil.Callee(pass.TypesInfo, call).(*types.Func)
	if fn == nil {
		return nil, 0
	}

	// Facts are associated with generic declarations, not instantiations.
	fn = fn.Origin()

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
		analysisinternal.IsTypeNamed(sig.Params().At(0).Type(), "fmt", "State") &&
		types.Identical(sig.Params().At(1).Type(), types.Typ[types.Rune])
}

// checkPrintf checks a call to a formatted print routine such as Printf.
func checkPrintf(pass *analysis.Pass, fileVersion string, kind Kind, call *ast.CallExpr, name string) {
	idx := formatStringIndex(pass, call)
	if idx < 0 || idx >= len(call.Args) {
		return
	}
	formatArg := call.Args[idx]
	format, ok := stringConstantExpr(pass, formatArg)
	if !ok {
		// Format string argument is non-constant.

		// It is a common mistake to call fmt.Printf(msg) with a
		// non-constant format string and no arguments:
		// if msg contains "%", misformatting occurs.
		// Report the problem and suggest a fix: fmt.Printf("%s", msg).
		//
		// However, as described in golang/go#71485, this analysis can produce a
		// significant number of diagnostics in existing code, and the bugs it
		// finds are sometimes unlikely or inconsequential, and may not be worth
		// fixing for some users. Gating on language version allows us to avoid
		// breaking existing tests and CI scripts.
		if idx == len(call.Args)-1 &&
			fileVersion != "" && // fail open
			versions.AtLeast(fileVersion, "go1.24") {

			pass.Report(analysis.Diagnostic{
				Pos: formatArg.Pos(),
				End: formatArg.End(),
				Message: fmt.Sprintf("non-constant format string in call to %s",
					name),
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: `Insert "%s" format string`,
					TextEdits: []analysis.TextEdit{{
						Pos:     formatArg.Pos(),
						End:     formatArg.Pos(),
						NewText: []byte(`"%s", `),
					}},
				}},
			})
		}
		return
	}

	firstArg := idx + 1 // Arguments are immediately after format string.
	if !strings.Contains(format, "%") {
		if len(call.Args) > firstArg {
			pass.ReportRangef(call.Args[firstArg], "%s call has arguments but no formatting directives", name)
		}
		return
	}

	// Pass the string constant value so
	// fmt.Sprintf("%"+("s"), "hi", 3) can be reported as
	// "fmt.Sprintf call needs 1 arg but has 2 args".
	operations, err := fmtstr.Parse(format, idx)
	if err != nil {
		// All error messages are in predicate form ("call has a problem")
		// so that they may be affixed into a subject ("log.Printf ").
		pass.ReportRangef(formatArg, "%s %s", name, err)
		return
	}

	// index of the highest used index.
	maxArgIndex := firstArg - 1
	anyIndex := false
	// Check formats against args.
	for _, op := range operations {
		if op.Prec.Index != -1 ||
			op.Width.Index != -1 ||
			op.Verb.Index != -1 {
			anyIndex = true
		}
		rng := opRange(formatArg, op)
		if !okPrintfArg(pass, call, rng, &maxArgIndex, firstArg, name, op) {
			// One error per format is enough.
			return
		}
		if op.Verb.Verb == 'w' {
			switch kind {
			case KindNone, KindPrint, KindPrintf:
				pass.ReportRangef(rng, "%s does not support error-wrapping directive %%w", name)
				return
			}
		}
	}
	// Dotdotdot is hard.
	if call.Ellipsis.IsValid() && maxArgIndex >= len(call.Args)-2 {
		return
	}
	// If any formats are indexed, extra arguments are ignored.
	if anyIndex {
		return
	}
	// There should be no leftover arguments.
	if maxArgIndex+1 < len(call.Args) {
		expect := maxArgIndex + 1 - firstArg
		numArgs := len(call.Args) - firstArg
		pass.ReportRangef(call, "%s call needs %v but has %v", name, count(expect, "arg"), count(numArgs, "arg"))
	}
}

// opRange returns the source range for the specified printf operation,
// such as the position of the %v substring of "...%v...".
func opRange(formatArg ast.Expr, op *fmtstr.Operation) analysis.Range {
	if lit, ok := formatArg.(*ast.BasicLit); ok {
		start, end, err := astutil.RangeInStringLiteral(lit, op.Range.Start, op.Range.End)
		if err == nil {
			return analysisinternal.Range(start, end) // position of "%v"
		}
	}
	return formatArg // entire format string
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

// okPrintfArg compares the operation to the arguments actually present,
// reporting any discrepancies it can discern, maxArgIndex was the index of the highest used index.
// If the final argument is ellipsissed, there's little it can do for that.
func okPrintfArg(pass *analysis.Pass, call *ast.CallExpr, rng analysis.Range, maxArgIndex *int, firstArg int, name string, operation *fmtstr.Operation) (ok bool) {
	verb := operation.Verb.Verb
	var v printVerb
	found := false
	// Linear scan is fast enough for a small list.
	for _, v = range printVerbs {
		if v.verb == verb {
			found = true
			break
		}
	}

	// Could verb's arg implement fmt.Formatter?
	// Skip check for the %w verb, which requires an error.
	formatter := false
	if v.typ != argError && operation.Verb.ArgIndex < len(call.Args) {
		if tv, ok := pass.TypesInfo.Types[call.Args[operation.Verb.ArgIndex]]; ok {
			formatter = isFormatter(tv.Type)
		}
	}

	if !formatter {
		if !found {
			pass.ReportRangef(rng, "%s format %s has unknown verb %c", name, operation.Text, verb)
			return false
		}
		for _, flag := range operation.Flags {
			// TODO: Disable complaint about '0' for Go 1.10. To be fixed properly in 1.11.
			// See issues 23598 and 23605.
			if flag == '0' {
				continue
			}
			if !strings.ContainsRune(v.flags, rune(flag)) {
				pass.ReportRangef(rng, "%s format %s has unrecognized flag %c", name, operation.Text, flag)
				return false
			}
		}
	}

	var argIndexes []int
	// First check for *.
	if operation.Width.Dynamic != -1 {
		argIndexes = append(argIndexes, operation.Width.Dynamic)
	}
	if operation.Prec.Dynamic != -1 {
		argIndexes = append(argIndexes, operation.Prec.Dynamic)
	}
	// If len(argIndexes)>0, we have something like %.*s and all
	// indexes in argIndexes must be an integer.
	for _, argIndex := range argIndexes {
		if !argCanBeChecked(pass, call, rng, argIndex, firstArg, operation, name) {
			return
		}
		arg := call.Args[argIndex]
		if reason, ok := matchArgType(pass, argInt, arg); !ok {
			details := ""
			if reason != "" {
				details = " (" + reason + ")"
			}
			pass.ReportRangef(rng, "%s format %s uses non-int %s%s as argument of *", name, operation.Text, analysisinternal.Format(pass.Fset, arg), details)
			return false
		}
	}

	// Collect to update maxArgNum in one loop.
	if operation.Verb.ArgIndex != -1 && verb != '%' {
		argIndexes = append(argIndexes, operation.Verb.ArgIndex)
	}
	for _, index := range argIndexes {
		*maxArgIndex = max(*maxArgIndex, index)
	}

	// Special case for '%', go will print "fmt.Printf("%10.2%%dhello", 4)"
	// as "%4hello", discard any runes between the two '%'s, and treat the verb '%'
	// as an ordinary rune, so early return to skip the type check.
	if verb == '%' || formatter {
		return true
	}

	// Now check verb's type.
	verbArgIndex := operation.Verb.ArgIndex
	if !argCanBeChecked(pass, call, rng, verbArgIndex, firstArg, operation, name) {
		return false
	}
	arg := call.Args[verbArgIndex]
	if isFunctionValue(pass, arg) && verb != 'p' && verb != 'T' {
		pass.ReportRangef(rng, "%s format %s arg %s is a func value, not called", name, operation.Text, analysisinternal.Format(pass.Fset, arg))
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
		pass.ReportRangef(rng, "%s format %s has arg %s of wrong type %s%s", name, operation.Text, analysisinternal.Format(pass.Fset, arg), typeString, details)
		return false
	}
	// Detect recursive formatting via value's String/Error methods.
	// The '#' flag suppresses the methods, except with %x, %X, and %q.
	if v.typ&argString != 0 && v.verb != 'T' && (!strings.Contains(operation.Flags, "#") || strings.ContainsRune("qxX", v.verb)) {
		if methodName, ok := recursiveStringer(pass, arg); ok {
			pass.ReportRangef(rng, "%s format %s with arg %s causes recursive %s method call", name, operation.Text, analysisinternal.Format(pass.Fset, arg), methodName)
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
func argCanBeChecked(pass *analysis.Pass, call *ast.CallExpr, rng analysis.Range, argIndex, firstArg int, operation *fmtstr.Operation, name string) bool {
	if argIndex <= 0 {
		// Shouldn't happen, so catch it with prejudice.
		panic("negative argIndex")
	}
	if argIndex < len(call.Args)-1 {
		return true // Always OK.
	}
	if call.Ellipsis.IsValid() {
		return false // We just can't tell; there could be many more arguments.
	}
	if argIndex < len(call.Args) {
		return true
	}
	// There are bad indexes in the format or there are fewer arguments than the format needs.
	// This is the argument number relative to the format: Printf("%s", "hi") will give 1 for the "hi".
	arg := argIndex - firstArg + 1 // People think of arguments as 1-indexed.
	pass.ReportRangef(rng, "%s format %s reads arg #%d, but call has %v", name, operation.Text, arg, count(len(call.Args)-firstArg, "arg"))
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
func checkPrint(pass *analysis.Pass, call *ast.CallExpr, name string) {
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
		it, ok := types.Unalias(typ).(*types.Interface)
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
					pass.ReportRangef(call, "%s does not take io.Writer but has first arg %s", name, analysisinternal.Format(pass.Fset, call.Args[0]))
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
			for _, m := range printFormatRE.FindAllString(s, -1) {
				// Allow %XX where XX are hex digits,
				// as this is common in URLs.
				if len(m) >= 3 && isHex(m[1]) && isHex(m[2]) {
					continue
				}
				pass.ReportRangef(call, "%s call has possible Printf formatting directive %s", name, m)
				break // report only the first one
			}
		}
	}
	if strings.HasSuffix(name, "ln") {
		// The last item, if a string, should not have a newline.
		arg = args[len(args)-1]
		if s, ok := stringConstantExpr(pass, arg); ok {
			if strings.HasSuffix(s, "\n") {
				pass.ReportRangef(call, "%s arg list ends with redundant newline", name)
			}
		}
	}
	for _, arg := range args {
		if isFunctionValue(pass, arg) {
			pass.ReportRangef(call, "%s arg %s is a func value, not called", name, analysisinternal.Format(pass.Fset, arg))
		}
		if methodName, ok := recursiveStringer(pass, arg); ok {
			pass.ReportRangef(call, "%s arg %s causes recursive call to %s method", name, analysisinternal.Format(pass.Fset, arg), methodName)
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

// isHex reports whether b is a hex digit.
func isHex(b byte) bool {
	return '0' <= b && b <= '9' ||
		'A' <= b && b <= 'F' ||
		'a' <= b && b <= 'f'
}
