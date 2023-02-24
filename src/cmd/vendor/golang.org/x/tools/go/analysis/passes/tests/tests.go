// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tests defines an Analyzer that checks for common mistaken
// usages of tests and examples.
package tests

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/typeparams"
)

const Doc = `check for common mistaken usages of tests and examples

The tests checker walks Test, Benchmark and Example functions checking
malformed names, wrong signatures and examples documenting non-existent
identifiers.

Please see the documentation for package testing in golang.org/pkg/testing
for the conventions that are enforced for Tests, Benchmarks, and Examples.`

var Analyzer = &analysis.Analyzer{
	Name: "tests",
	Doc:  Doc,
	Run:  run,
}

var acceptedFuzzTypes = []types.Type{
	types.Typ[types.String],
	types.Typ[types.Bool],
	types.Typ[types.Float32],
	types.Typ[types.Float64],
	types.Typ[types.Int],
	types.Typ[types.Int8],
	types.Typ[types.Int16],
	types.Typ[types.Int32],
	types.Typ[types.Int64],
	types.Typ[types.Uint],
	types.Typ[types.Uint8],
	types.Typ[types.Uint16],
	types.Typ[types.Uint32],
	types.Typ[types.Uint64],
	types.NewSlice(types.Universe.Lookup("byte").Type()),
}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, f := range pass.Files {
		if !strings.HasSuffix(pass.Fset.File(f.Pos()).Name(), "_test.go") {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil {
				// Ignore non-functions or functions with receivers.
				continue
			}
			switch {
			case strings.HasPrefix(fn.Name.Name, "Example"):
				checkExampleName(pass, fn)
				checkExampleOutput(pass, fn, f.Comments)
			case strings.HasPrefix(fn.Name.Name, "Test"):
				checkTest(pass, fn, "Test")
			case strings.HasPrefix(fn.Name.Name, "Benchmark"):
				checkTest(pass, fn, "Benchmark")
			}
			// run fuzz tests diagnostics only for 1.18 i.e. when analysisinternal.DiagnoseFuzzTests is turned on.
			if strings.HasPrefix(fn.Name.Name, "Fuzz") && analysisinternal.DiagnoseFuzzTests {
				checkTest(pass, fn, "Fuzz")
				checkFuzz(pass, fn)
			}
		}
	}
	return nil, nil
}

// checkFuzz checks the contents of a fuzz function.
func checkFuzz(pass *analysis.Pass, fn *ast.FuncDecl) {
	params := checkFuzzCall(pass, fn)
	if params != nil {
		checkAddCalls(pass, fn, params)
	}
}

// checkFuzzCall checks the arguments of f.Fuzz() calls:
//
//  1. f.Fuzz() should call a function and it should be of type (*testing.F).Fuzz().
//  2. The called function in f.Fuzz(func(){}) should not return result.
//  3. First argument of func() should be of type *testing.T
//  4. Second argument onwards should be of type []byte, string, bool, byte,
//     rune, float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16,
//     uint32, uint64
//  5. func() must not call any *F methods, e.g. (*F).Log, (*F).Error, (*F).Skip
//     The only *F methods that are allowed in the (*F).Fuzz function are (*F).Failed and (*F).Name.
//
// Returns the list of parameters to the fuzz function, if they are valid fuzz parameters.
func checkFuzzCall(pass *analysis.Pass, fn *ast.FuncDecl) (params *types.Tuple) {
	ast.Inspect(fn, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if ok {
			if !isFuzzTargetDotFuzz(pass, call) {
				return true
			}

			// Only one argument (func) must be passed to (*testing.F).Fuzz.
			if len(call.Args) != 1 {
				return true
			}
			expr := call.Args[0]
			if pass.TypesInfo.Types[expr].Type == nil {
				return true
			}
			t := pass.TypesInfo.Types[expr].Type.Underlying()
			tSign, argOk := t.(*types.Signature)
			// Argument should be a function
			if !argOk {
				pass.ReportRangef(expr, "argument to Fuzz must be a function")
				return false
			}
			// ff Argument function should not return
			if tSign.Results().Len() != 0 {
				pass.ReportRangef(expr, "fuzz target must not return any value")
			}
			// ff Argument function should have 1 or more argument
			if tSign.Params().Len() == 0 {
				pass.ReportRangef(expr, "fuzz target must have 1 or more argument")
				return false
			}
			ok := validateFuzzArgs(pass, tSign.Params(), expr)
			if ok && params == nil {
				params = tSign.Params()
			}
			// Inspect the function that was passed as an argument to make sure that
			// there are no calls to *F methods, except for Name and Failed.
			ast.Inspect(expr, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					if !isFuzzTargetDot(pass, call, "") {
						return true
					}
					if !isFuzzTargetDot(pass, call, "Name") && !isFuzzTargetDot(pass, call, "Failed") {
						pass.ReportRangef(call, "fuzz target must not call any *F methods")
					}
				}
				return true
			})
			// We do not need to look at any calls to f.Fuzz inside of a Fuzz call,
			// since they are not allowed.
			return false
		}
		return true
	})
	return params
}

// checkAddCalls checks that the arguments of f.Add calls have the same number and type of arguments as
// the signature of the function passed to (*testing.F).Fuzz
func checkAddCalls(pass *analysis.Pass, fn *ast.FuncDecl, params *types.Tuple) {
	ast.Inspect(fn, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if ok {
			if !isFuzzTargetDotAdd(pass, call) {
				return true
			}

			// The first argument to function passed to (*testing.F).Fuzz is (*testing.T).
			if len(call.Args) != params.Len()-1 {
				pass.ReportRangef(call, "wrong number of values in call to (*testing.F).Add: %d, fuzz target expects %d", len(call.Args), params.Len()-1)
				return true
			}
			var mismatched []int
			for i, expr := range call.Args {
				if pass.TypesInfo.Types[expr].Type == nil {
					return true
				}
				t := pass.TypesInfo.Types[expr].Type
				if !types.Identical(t, params.At(i+1).Type()) {
					mismatched = append(mismatched, i)
				}
			}
			// If just one of the types is mismatched report for that
			// type only. Otherwise report for the whole call to (*testing.F).Add
			if len(mismatched) == 1 {
				i := mismatched[0]
				expr := call.Args[i]
				t := pass.TypesInfo.Types[expr].Type
				pass.ReportRangef(expr, fmt.Sprintf("mismatched type in call to (*testing.F).Add: %v, fuzz target expects %v", t, params.At(i+1).Type()))
			} else if len(mismatched) > 1 {
				var gotArgs, wantArgs []types.Type
				for i := 0; i < len(call.Args); i++ {
					gotArgs, wantArgs = append(gotArgs, pass.TypesInfo.Types[call.Args[i]].Type), append(wantArgs, params.At(i+1).Type())
				}
				pass.ReportRangef(call, fmt.Sprintf("mismatched types in call to (*testing.F).Add: %v, fuzz target expects %v", gotArgs, wantArgs))
			}
		}
		return true
	})
}

// isFuzzTargetDotFuzz reports whether call is (*testing.F).Fuzz().
func isFuzzTargetDotFuzz(pass *analysis.Pass, call *ast.CallExpr) bool {
	return isFuzzTargetDot(pass, call, "Fuzz")
}

// isFuzzTargetDotAdd reports whether call is (*testing.F).Add().
func isFuzzTargetDotAdd(pass *analysis.Pass, call *ast.CallExpr) bool {
	return isFuzzTargetDot(pass, call, "Add")
}

// isFuzzTargetDot reports whether call is (*testing.F).<name>().
func isFuzzTargetDot(pass *analysis.Pass, call *ast.CallExpr, name string) bool {
	if selExpr, ok := call.Fun.(*ast.SelectorExpr); ok {
		if !isTestingType(pass.TypesInfo.Types[selExpr.X].Type, "F") {
			return false
		}
		if name == "" || selExpr.Sel.Name == name {
			return true
		}
	}
	return false
}

// Validate the arguments of fuzz target.
func validateFuzzArgs(pass *analysis.Pass, params *types.Tuple, expr ast.Expr) bool {
	fLit, isFuncLit := expr.(*ast.FuncLit)
	exprRange := expr
	ok := true
	if !isTestingType(params.At(0).Type(), "T") {
		if isFuncLit {
			exprRange = fLit.Type.Params.List[0].Type
		}
		pass.ReportRangef(exprRange, "the first parameter of a fuzz target must be *testing.T")
		ok = false
	}
	for i := 1; i < params.Len(); i++ {
		if !isAcceptedFuzzType(params.At(i).Type()) {
			if isFuncLit {
				curr := 0
				for _, field := range fLit.Type.Params.List {
					curr += len(field.Names)
					if i < curr {
						exprRange = field.Type
						break
					}
				}
			}
			pass.ReportRangef(exprRange, "fuzzing arguments can only have the following types: "+formatAcceptedFuzzType())
			ok = false
		}
	}
	return ok
}

func isTestingType(typ types.Type, testingType string) bool {
	ptr, ok := typ.(*types.Pointer)
	if !ok {
		return false
	}
	named, ok := ptr.Elem().(*types.Named)
	if !ok {
		return false
	}
	obj := named.Obj()
	// obj.Pkg is nil for the error type.
	return obj != nil && obj.Pkg() != nil && obj.Pkg().Path() == "testing" && obj.Name() == testingType
}

// Validate that fuzz target function's arguments are of accepted types.
func isAcceptedFuzzType(paramType types.Type) bool {
	for _, typ := range acceptedFuzzTypes {
		if types.Identical(typ, paramType) {
			return true
		}
	}
	return false
}

func formatAcceptedFuzzType() string {
	var acceptedFuzzTypesStrings []string
	for _, typ := range acceptedFuzzTypes {
		acceptedFuzzTypesStrings = append(acceptedFuzzTypesStrings, typ.String())
	}
	acceptedFuzzTypesMsg := strings.Join(acceptedFuzzTypesStrings, ", ")
	return acceptedFuzzTypesMsg
}

func isExampleSuffix(s string) bool {
	r, size := utf8.DecodeRuneInString(s)
	return size > 0 && unicode.IsLower(r)
}

func isTestSuffix(name string) bool {
	if len(name) == 0 {
		// "Test" is ok.
		return true
	}
	r, _ := utf8.DecodeRuneInString(name)
	return !unicode.IsLower(r)
}

func isTestParam(typ ast.Expr, wantType string) bool {
	ptr, ok := typ.(*ast.StarExpr)
	if !ok {
		// Not a pointer.
		return false
	}
	// No easy way of making sure it's a *testing.T or *testing.B:
	// ensure the name of the type matches.
	if name, ok := ptr.X.(*ast.Ident); ok {
		return name.Name == wantType
	}
	if sel, ok := ptr.X.(*ast.SelectorExpr); ok {
		return sel.Sel.Name == wantType
	}
	return false
}

func lookup(pkg *types.Package, name string) []types.Object {
	if o := pkg.Scope().Lookup(name); o != nil {
		return []types.Object{o}
	}

	var ret []types.Object
	// Search through the imports to see if any of them define name.
	// It's hard to tell in general which package is being tested, so
	// for the purposes of the analysis, allow the object to appear
	// in any of the imports. This guarantees there are no false positives
	// because the example needs to use the object so it must be defined
	// in the package or one if its imports. On the other hand, false
	// negatives are possible, but should be rare.
	for _, imp := range pkg.Imports() {
		if obj := imp.Scope().Lookup(name); obj != nil {
			ret = append(ret, obj)
		}
	}
	return ret
}

// This pattern is taken from /go/src/go/doc/example.go
var outputRe = regexp.MustCompile(`(?i)^[[:space:]]*(unordered )?output:`)

type commentMetadata struct {
	isOutput bool
	pos      token.Pos
}

func checkExampleOutput(pass *analysis.Pass, fn *ast.FuncDecl, fileComments []*ast.CommentGroup) {
	commentsInExample := []commentMetadata{}
	numOutputs := 0

	// Find the comment blocks that are in the example. These comments are
	// guaranteed to be in order of appearance.
	for _, cg := range fileComments {
		if cg.Pos() < fn.Pos() {
			continue
		} else if cg.End() > fn.End() {
			break
		}

		isOutput := outputRe.MatchString(cg.Text())
		if isOutput {
			numOutputs++
		}

		commentsInExample = append(commentsInExample, commentMetadata{
			isOutput: isOutput,
			pos:      cg.Pos(),
		})
	}

	// Change message based on whether there are multiple output comment blocks.
	msg := "output comment block must be the last comment block"
	if numOutputs > 1 {
		msg = "there can only be one output comment block per example"
	}

	for i, cg := range commentsInExample {
		// Check for output comments that are not the last comment in the example.
		isLast := (i == len(commentsInExample)-1)
		if cg.isOutput && !isLast {
			pass.Report(
				analysis.Diagnostic{
					Pos:     cg.pos,
					Message: msg,
				},
			)
		}
	}
}

func checkExampleName(pass *analysis.Pass, fn *ast.FuncDecl) {
	fnName := fn.Name.Name
	if params := fn.Type.Params; len(params.List) != 0 {
		pass.Reportf(fn.Pos(), "%s should be niladic", fnName)
	}
	if results := fn.Type.Results; results != nil && len(results.List) != 0 {
		pass.Reportf(fn.Pos(), "%s should return nothing", fnName)
	}
	if tparams := typeparams.ForFuncType(fn.Type); tparams != nil && len(tparams.List) > 0 {
		pass.Reportf(fn.Pos(), "%s should not have type params", fnName)
	}

	if fnName == "Example" {
		// Nothing more to do.
		return
	}

	var (
		exName = strings.TrimPrefix(fnName, "Example")
		elems  = strings.SplitN(exName, "_", 3)
		ident  = elems[0]
		objs   = lookup(pass.Pkg, ident)
	)
	if ident != "" && len(objs) == 0 {
		// Check ExampleFoo and ExampleBadFoo.
		pass.Reportf(fn.Pos(), "%s refers to unknown identifier: %s", fnName, ident)
		// Abort since obj is absent and no subsequent checks can be performed.
		return
	}
	if len(elems) < 2 {
		// Nothing more to do.
		return
	}

	if ident == "" {
		// Check Example_suffix and Example_BadSuffix.
		if residual := strings.TrimPrefix(exName, "_"); !isExampleSuffix(residual) {
			pass.Reportf(fn.Pos(), "%s has malformed example suffix: %s", fnName, residual)
		}
		return
	}

	mmbr := elems[1]
	if !isExampleSuffix(mmbr) {
		// Check ExampleFoo_Method and ExampleFoo_BadMethod.
		found := false
		// Check if Foo.Method exists in this package or its imports.
		for _, obj := range objs {
			if obj, _, _ := types.LookupFieldOrMethod(obj.Type(), true, obj.Pkg(), mmbr); obj != nil {
				found = true
				break
			}
		}
		if !found {
			pass.Reportf(fn.Pos(), "%s refers to unknown field or method: %s.%s", fnName, ident, mmbr)
		}
	}
	if len(elems) == 3 && !isExampleSuffix(elems[2]) {
		// Check ExampleFoo_Method_suffix and ExampleFoo_Method_Badsuffix.
		pass.Reportf(fn.Pos(), "%s has malformed example suffix: %s", fnName, elems[2])
	}
}

func checkTest(pass *analysis.Pass, fn *ast.FuncDecl, prefix string) {
	// Want functions with 0 results and 1 parameter.
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 ||
		fn.Type.Params == nil ||
		len(fn.Type.Params.List) != 1 ||
		len(fn.Type.Params.List[0].Names) > 1 {
		return
	}

	// The param must look like a *testing.T or *testing.B.
	if !isTestParam(fn.Type.Params.List[0].Type, prefix[:1]) {
		return
	}

	if tparams := typeparams.ForFuncType(fn.Type); tparams != nil && len(tparams.List) > 0 {
		// Note: cmd/go/internal/load also errors about TestXXX and BenchmarkXXX functions with type parameters.
		// We have currently decided to also warn before compilation/package loading. This can help users in IDEs.
		// TODO(adonovan): use ReportRangef(tparams).
		pass.Reportf(fn.Pos(), "%s has type parameters: it will not be run by go test as a %sXXX function", fn.Name.Name, prefix)
	}

	if !isTestSuffix(fn.Name.Name[len(prefix):]) {
		// TODO(adonovan): use ReportRangef(fn.Name).
		pass.Reportf(fn.Pos(), "%s has malformed name: first letter after '%s' must not be lowercase", fn.Name.Name, prefix)
	}
}
