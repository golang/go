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
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/fmtstr"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/versions"
	"golang.org/x/tools/refactor/satisfy"
)

func init() {
	Analyzer.Flags.Var(isPrint, "funcs", "comma-separated list of print function names to check")
}

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:       "printf",
	Doc:        analyzerutil.MustExtractDoc(doc, "printf"),
	URL:        "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/printf",
	Requires:   []*analysis.Analyzer{inspect.Analyzer},
	Run:        run,
	ResultType: reflect.TypeFor[*Result](),
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
	return "(none)"
}

// Result is the printf analyzer's result type. Clients may query the result
// to learn whether a function behaves like fmt.Print or fmt.Printf.
type Result struct {
	funcs map[types.Object]Kind
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
		funcs: make(map[types.Object]Kind),
	}
	findPrintLike(pass, res)
	checkCalls(pass, res)
	return res, nil
}

// A wrapper is a candidate print/printf wrapper function.
//
// We represent functions generally as types.Object, not *Func, so
// that we can analyze anonymous functions such as
//
//	printf := func(format string, args ...any) {...},
//
// representing them by the *types.Var symbol for the local variable
// 'printf'.
type wrapper struct {
	obj     types.Object     // *Func or *Var
	curBody inspector.Cursor // for *ast.BlockStmt
	format  *types.Var       // optional "format string" parameter in the Func{Decl,Lit}
	args    *types.Var       // "args ...any" parameter in the Func{Decl,Lit}
	callers []printfCaller
}

// printfCaller is a candidate print{,f} forwarding call from candidate wrapper w.
type printfCaller struct {
	w    *wrapper
	call *ast.CallExpr // forwarding call (nil for implicit interface method -> impl calls)
}

// formatArgsParams returns the "format string" and "args ...any"
// parameters of a potential print or printf wrapper function.
// (The format is nil in the print-like case.)
func formatArgsParams(sig *types.Signature) (format, args *types.Var) {
	if !sig.Variadic() {
		return nil, nil // not variadic
	}

	params := sig.Params()
	nparams := params.Len() // variadic => nonzero

	// Is second last param 'format string'?
	if nparams >= 2 {
		if p := params.At(nparams - 2); p.Type() == types.Typ[types.String] {
			format = p
		}
	}

	// Check final parameter is "args ...any".
	// (variadic => slice)
	args = params.At(nparams - 1)
	iface, ok := types.Unalias(args.Type().(*types.Slice).Elem()).(*types.Interface)
	if !ok || !iface.Empty() {
		return nil, nil
	}

	return format, args
}

// findPrintLike scans the entire package to find print or printf-like functions.
// When it returns, all such functions have been identified.
func findPrintLike(pass *analysis.Pass, res *Result) {
	var (
		inspect = pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		info    = pass.TypesInfo
	)

	// Pass 1: gather candidate wrapper functions (and populate wrappers).
	var (
		wrappers []*wrapper
		byObj    = make(map[types.Object]*wrapper)
	)
	for cur := range inspect.Root().Preorder((*ast.FuncDecl)(nil), (*ast.FuncLit)(nil), (*ast.InterfaceType)(nil)) {

		// addWrapper records that a func (or var representing
		// a FuncLit) is a potential print{,f} wrapper.
		// curBody is its *ast.BlockStmt, if any.
		addWrapper := func(obj types.Object, sig *types.Signature, curBody inspector.Cursor) *wrapper {
			format, args := formatArgsParams(sig)
			if args != nil {
				// obj (the symbol for a function/method, or variable
				// assigned to an anonymous function) is a potential
				// print or printf wrapper.
				//
				// Later processing will analyze the graph of potential
				// wrappers and their function bodies to pick out the
				// ones that are true wrappers.
				w := &wrapper{
					obj:     obj,
					curBody: curBody,
					format:  format, // non-nil => printf
					args:    args,
				}
				byObj[w.obj] = w
				wrappers = append(wrappers, w)
				return w
			}
			return nil
		}

		switch f := cur.Node().(type) {
		case *ast.FuncDecl:
			// named function or method:
			//
			//    func wrapf(format string, args ...any) {...}
			if f.Body != nil {
				fn := info.Defs[f.Name].(*types.Func)
				addWrapper(fn, fn.Signature(), cur.ChildAt(edge.FuncDecl_Body, -1))
			}

		case *ast.FuncLit:
			// anonymous function directly assigned to a variable:
			//
			//    var wrapf = func(format string, args ...any) {...}
			//    wrapf    := func(format string, args ...any) {...}
			//    wrapf     = func(format string, args ...any) {...}
			//
			// The LHS may also be a struct field x.wrapf or
			// an imported var pkg.Wrapf.
			//
			var lhs ast.Expr
			switch ek, idx := cur.ParentEdge(); ek {
			case edge.ValueSpec_Values:
				curName := cur.Parent().ChildAt(edge.ValueSpec_Names, idx)
				lhs = curName.Node().(*ast.Ident)
			case edge.AssignStmt_Rhs:
				curLhs := cur.Parent().ChildAt(edge.AssignStmt_Lhs, idx)
				lhs = curLhs.Node().(ast.Expr)
			}

			var v *types.Var
			switch lhs := lhs.(type) {
			case *ast.Ident:
				// variable: wrapf = func(...)
				v, _ = info.ObjectOf(lhs).(*types.Var)
			case *ast.SelectorExpr:
				if sel, ok := info.Selections[lhs]; ok {
					// struct field: x.wrapf = func(...)
					v = sel.Obj().(*types.Var)
				} else {
					// imported var: pkg.Wrapf = func(...)
					v = info.Uses[lhs.Sel].(*types.Var)
				}
			}
			if v != nil {
				sig := info.TypeOf(f).(*types.Signature)
				curBody := cur.ChildAt(edge.FuncLit_Body, -1)
				addWrapper(v, sig, curBody)
			}

		case *ast.InterfaceType:
			// Induction through interface methods is gated as
			// if it were a go1.26 language feature, to avoid
			// surprises when go test's vet suite gets stricter.
			if analyzerutil.FileUsesGoVersion(pass, astutil.EnclosingFile(cur), versions.Go1_26) {
				for imeth := range info.TypeOf(f).(*types.Interface).Methods() {
					addWrapper(imeth, imeth.Signature(), inspector.Cursor{})
				}
			}
		}
	}

	// impls maps abstract methods to implementations.
	//
	// Interface methods are modelled as if they have a body
	// that calls each implementing method.
	//
	// In the code below, impls maps Logger.Logf to
	// [myLogger.Logf], and if myLogger.Logf is discovered to be
	// printf-like, then so will be Logger.Logf.
	//
	//   type Logger interface {
	// 	Logf(format string, args ...any)
	//   }
	//   type myLogger struct{ ... }
	//   func (myLogger) Logf(format string, args ...any) {...}
	//   var _ Logger = myLogger{}
	impls := methodImplementations(pass)

	// doCall records a call from one wrapper to another.
	doCall := func(w *wrapper, callee types.Object, call *ast.CallExpr) {
		// Call from one wrapper candidate to another?
		// Record the edge so that if callee is found to be
		// a true wrapper, w will be too.
		if w2, ok := byObj[callee]; ok {
			w2.callers = append(w2.callers, printfCaller{w, call})
		}

		// Is the candidate a true wrapper, because it calls
		// a known print{,f}-like function from the allowlist
		// or an imported fact, or another wrapper found
		// to be a true wrapper?
		// If so, convert all w's callers to kind.
		kind := callKind(pass, callee, res)
		if kind != KindNone {
			propagate(pass, w, call, kind, res)
		}
	}

	// Pass 2: scan the body of each wrapper function
	// for calls to other printf-like functions.
	for _, w := range wrappers {

		// An interface method has no body, but acts
		// like an implicit call to each implementing method.
		if w.curBody.Inspector() == nil {
			for impl := range impls[w.obj.(*types.Func)] {
				doCall(w, impl, nil)
			}
			continue // (no body)
		}

		// Process all calls in the wrapper function's body.
	scan:
		for cur := range w.curBody.Preorder(
			(*ast.AssignStmt)(nil),
			(*ast.UnaryExpr)(nil),
			(*ast.CallExpr)(nil),
		) {
			switch n := cur.Node().(type) {

			// Reject tricky cases where the parameters
			// are potentially mutated by AssignStmt or UnaryExpr.
			// (This logic checks for mutation only before the call.)
			// TODO: Relax these checks; issue 26555.

			case *ast.AssignStmt:
				// If the wrapper updates format or args
				// it is not a simple wrapper.
				for _, lhs := range n.Lhs {
					if w.format != nil && match(info, lhs, w.format) ||
						match(info, lhs, w.args) {
						break scan
					}
				}

			case *ast.UnaryExpr:
				// If the wrapper computes &format or &args,
				// it is not a simple wrapper.
				if n.Op == token.AND &&
					(w.format != nil && match(info, n.X, w.format) ||
						match(info, n.X, w.args)) {
					break scan
				}

			case *ast.CallExpr:
				if len(n.Args) > 0 && match(info, n.Args[len(n.Args)-1], w.args) {
					if callee := typeutil.Callee(pass.TypesInfo, n); callee != nil {
						doCall(w, callee, n)
					}
				}
			}
		}
	}
}

// methodImplementations returns the mapping from interface methods
// declared in this package to their corresponding implementing
// methods (which may also be interface methods), according to the set
// of assignments to interface types that appear within this package.
func methodImplementations(pass *analysis.Pass) map[*types.Func]map[*types.Func]bool {
	impls := make(map[*types.Func]map[*types.Func]bool)

	// To find interface/implementation relations,
	// we use the 'satisfy' pass, but proposal #70638
	// provides a better way.
	//
	// This pass over the syntax could be factored out as
	// a separate analysis pass if it is needed by other
	// analyzers.
	var f satisfy.Finder
	f.Find(pass.TypesInfo, pass.Files)
	for assign := range f.Result {
		// Have: LHS = RHS, where LHS is an interface type.
		for imeth := range assign.LHS.Underlying().(*types.Interface).Methods() {
			// Limit to interface methods of current package.
			if imeth.Pkg() != pass.Pkg {
				continue
			}

			if _, args := formatArgsParams(imeth.Signature()); args == nil {
				continue // not print{,f}-like
			}

			// Add implementing method to the set.
			impl, _, _ := types.LookupFieldOrMethod(assign.RHS, false, pass.Pkg, imeth.Name()) // can't fail
			set, ok := impls[imeth]
			if !ok {
				set = make(map[*types.Func]bool)
				impls[imeth] = set
			}
			set[impl.(*types.Func)] = true
		}
	}
	return impls
}

func match(info *types.Info, arg ast.Expr, param *types.Var) bool {
	id, ok := arg.(*ast.Ident)
	return ok && info.ObjectOf(id) == param
}

// propagate propagates changes in wrapper (non-None) kind information backwards
// through through the wrapper.callers graph of well-formed forwarding calls.
func propagate(pass *analysis.Pass, w *wrapper, call *ast.CallExpr, kind Kind, res *Result) {
	// Check correct call forwarding.
	//
	// Interface methods (call==nil) forward
	// correctly by construction.
	if call != nil && !checkForward(pass, w, call, kind) {
		return
	}

	// If the candidate's print{,f} status becomes known,
	// propagate it back to all its so-far known callers.
	if res.funcs[w.obj] != kind {
		res.funcs[w.obj] = kind

		// Export a fact.
		// (This is a no-op for local symbols.)
		// We can't export facts on a symbol of another package,
		// but we can treat the symbol as a wrapper within
		// the current analysis unit.
		if w.obj.Pkg() == pass.Pkg {
			// Facts are associated with origins.
			pass.ExportObjectFact(origin(w.obj), &isWrapper{Kind: kind})
		}

		// Propagate kind back to known callers.
		for _, caller := range w.callers {
			propagate(pass, caller.w, caller.call, kind, res)
		}
	}
}

// checkForward checks whether a call from wrapper w is a well-formed
// forwarding call of the specified (non-None) kind.
//
// If not, it reports a diagnostic that the user wrote
// fmt.Printf(format, args) instead of fmt.Printf(format, args...).
func checkForward(pass *analysis.Pass, w *wrapper, call *ast.CallExpr, kind Kind) bool {
	// Printf/Errorf calls must delegate the format string.
	switch kind {
	case KindPrintf, KindErrorf:
		if len(call.Args) < 2 || !match(pass.TypesInfo, call.Args[len(call.Args)-2], w.format) {
			return false
		}
	}

	// The args... delegation must be variadic.
	// (That args is actually delegated was
	// established before the root call to doCall.)
	if !call.Ellipsis.IsValid() {
		typ, ok := pass.TypesInfo.Types[call.Fun].Type.(*types.Signature)
		if !ok {
			return false
		}
		if len(call.Args) > typ.Params().Len() {
			// If we're passing more arguments than what the
			// print/printf function can take, adding an ellipsis
			// would break the program. For example:
			//
			//   func foo(arg1 string, arg2 ...interface{}) {
			//       fmt.Printf("%s %v", arg1, arg2)
			//   }
			return false
		}
		pass.ReportRangef(call, "missing ... in args forwarded to %s-like function", kind)
		return false
	}

	return true
}

func origin(obj types.Object) types.Object {
	switch obj := obj.(type) {
	case *types.Func:
		return obj.Origin()
	case *types.Var:
		return obj.Origin()
	}
	return obj
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
	"(testing.TB).Error":       true,
	"(testing.TB).Errorf":      true,
	"(testing.TB).Fatal":       true,
	"(testing.TB).Fatalf":      true,
	"(testing.TB).Log":         true,
	"(testing.TB).Logf":        true,
	"(testing.TB).Skip":        true,
	"(testing.TB).Skipf":       true,
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
func checkCalls(pass *analysis.Pass, res *Result) {
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
			if callee := typeutil.Callee(pass.TypesInfo, n); callee != nil {
				kind := callKind(pass, callee, res)
				switch kind {
				case KindPrintf, KindErrorf:
					checkPrintf(pass, fileVersion, kind, n, fullname(callee))
				case KindPrint:
					checkPrint(pass, n, fullname(callee))
				}
			}
		}
	})
}

func fullname(obj types.Object) string {
	if fn, ok := obj.(*types.Func); ok {
		return fn.FullName()
	}
	return obj.Name()
}

// callKind returns the symbol of the called function
// and its print/printf kind, if any.
// (The symbol may be a var for an anonymous function.)
// The result is memoized in res.funcs.
func callKind(pass *analysis.Pass, obj types.Object, res *Result) Kind {
	kind, ok := res.funcs[obj]
	if !ok {
		// cache miss
		_, ok := isPrint[fullname(obj)]
		if !ok {
			// Next look up just "printf", for use with -printf.funcs.
			_, ok = isPrint[strings.ToLower(obj.Name())]
		}
		if ok {
			// well-known printf functions
			if fullname(obj) == "fmt.Errorf" {
				kind = KindErrorf
			} else if strings.HasSuffix(obj.Name(), "f") {
				kind = KindPrintf
			} else {
				kind = KindPrint
			}
		} else {
			// imported wrappers
			// Facts are associated with generic declarations, not instantiations.
			obj = origin(obj)
			var fact isWrapper
			if pass.ImportObjectFact(obj, &fact) {
				kind = fact.Kind
			}
		}
		res.funcs[obj] = kind // cache
	}
	return kind
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
		typesinternal.IsTypeNamed(sig.Params().At(0).Type(), "fmt", "State") &&
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
			versions.AtLeast(fileVersion, versions.Go1_24) {

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
		if !okPrintfArg(pass, fileVersion, call, rng, &maxArgIndex, firstArg, name, op) {
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
		rng, err := astutil.RangeInStringLiteral(lit, op.Range.Start, op.Range.End)
		if err == nil {
			return rng // position of "%v"
		}
	}
	return formatArg // entire format string
}

// printfArgType encodes the types of expressions a printf verb accepts. It is a bitmask.
type printfArgType int

const (
	argBool printfArgType = 1 << iota
	argByte
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
	{'q', " -+.0#", argRune | argInt | argString}, // note: when analyzing go1.26 code, argInt => argByte
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
func okPrintfArg(pass *analysis.Pass, fileVersion string, call *ast.CallExpr, rng analysis.Range, maxArgIndex *int, firstArg int, name string, operation *fmtstr.Operation) (ok bool) {
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

	// When analyzing go1.26 code, rune and byte are the only %q integers (#72850).
	if verb == 'q' &&
		fileVersion != "" && // fail open
		versions.AtLeast(fileVersion, versions.Go1_26) {
		v.typ = argRune | argByte | argString
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
			pass.ReportRangef(rng, "%s format %s uses non-int %s%s as argument of *", name, operation.Text, astutil.Format(pass.Fset, arg), details)
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
		pass.ReportRangef(rng, "%s format %s arg %s is a func value, not called", name, operation.Text, astutil.Format(pass.Fset, arg))
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
		pass.ReportRangef(rng, "%s format %s has arg %s of wrong type %s%s", name, operation.Text, astutil.Format(pass.Fset, arg), typeString, details)
		return false
	}
	// Detect recursive formatting via value's String/Error methods.
	// The '#' flag suppresses the methods, except with %x, %X, and %q.
	if v.typ&argString != 0 && v.verb != 'T' && (!strings.Contains(operation.Flags, "#") || strings.ContainsRune("qxX", v.verb)) {
		if methodName, ok := recursiveStringer(pass, arg); ok {
			pass.ReportRangef(rng, "%s format %s with arg %s causes recursive %s method call", name, operation.Text, astutil.Format(pass.Fset, arg), methodName)
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
					pass.ReportRangef(call, "%s does not take io.Writer but has first arg %s", name, astutil.Format(pass.Fset, call.Args[0]))
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
			pass.ReportRangef(call, "%s arg %s is a func value, not called", name, astutil.Format(pass.Fset, arg))
		}
		if methodName, ok := recursiveStringer(pass, arg); ok {
			pass.ReportRangef(call, "%s arg %s causes recursive call to %s method", name, astutil.Format(pass.Fset, arg), methodName)
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
	for name := range strings.SplitSeq(flag, ",") {
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
