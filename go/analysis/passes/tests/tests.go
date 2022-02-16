// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tests defines an Analyzer that checks for common mistaken
// usages of tests and examples.
package tests

import (
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
				checkFuzzCall(pass, fn)
			}
		}
	}
	return nil, nil
}

// Check the arguments of f.Fuzz() calls :
// 1. f.Fuzz() should call a function and it should be of type (*testing.F).Fuzz().
// 2. The called function in f.Fuzz(func(){}) should not return result.
// 3. First argument of func() should be of type *testing.T
// 4. Second argument onwards should be of type []byte, string, bool, byte,
//	  rune, float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16,
//	  uint32, uint64
func checkFuzzCall(pass *analysis.Pass, fn *ast.FuncDecl) {
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
				return true
			}
			// ff Argument function should not return
			if tSign.Results().Len() != 0 {
				pass.ReportRangef(expr, "fuzz target must not return any value")
			}
			// ff Argument function should have 1 or more argument
			if tSign.Params().Len() == 0 {
				pass.ReportRangef(expr, "fuzz target must have 1 or more argument")
				return true
			}
			validateFuzzArgs(pass, tSign.Params(), expr)
		}
		return true
	})
}

// isFuzzTargetDotFuzz reports whether call is (*testing.F).Fuzz().
func isFuzzTargetDotFuzz(pass *analysis.Pass, call *ast.CallExpr) bool {
	if selExpr, ok := call.Fun.(*ast.SelectorExpr); ok {
		if !isTestingType(pass.TypesInfo.Types[selExpr.X].Type, "F") {
			return false
		}
		if selExpr.Sel.Name == "Fuzz" {
			return true
		}
	}
	return false
}

// Validate the arguments of fuzz target.
func validateFuzzArgs(pass *analysis.Pass, params *types.Tuple, expr ast.Expr) {
	fLit, isFuncLit := expr.(*ast.FuncLit)
	exprRange := expr
	if !isTestingType(params.At(0).Type(), "T") {
		if isFuncLit {
			exprRange = fLit.Type.Params.List[0].Type
		}
		pass.ReportRangef(exprRange, "the first parameter of a fuzz target must be *testing.T")
	}
	for i := 1; i < params.Len(); i++ {
		if !isAcceptedFuzzType(params.At(i).Type()) {
			if isFuncLit {
				exprRange = fLit.Type.Params.List[i].Type
			}
			pass.ReportRangef(exprRange, "fuzzing arguments can only have the following types: "+formatAcceptedFuzzType())
		}
	}
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
	return named.Obj().Pkg().Path() == "testing" && named.Obj().Name() == testingType
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
		pass.Reportf(fn.Pos(), "%s has type parameters: it will not be run by go test as a %sXXX function", fn.Name.Name, prefix)
	}

	if !isTestSuffix(fn.Name.Name[len(prefix):]) {
		pass.Reportf(fn.Pos(), "%s has malformed name: first letter after '%s' must not be lowercase", fn.Name.Name, prefix)
	}
}
