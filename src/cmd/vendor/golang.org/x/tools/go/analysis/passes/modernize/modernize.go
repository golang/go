// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	_ "embed"
	"go/ast"
	"go/constant"
	"go/format"
	"go/token"
	"go/types"
	"iter"
	"regexp"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal/typeindex"

	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/packagepath"
	"golang.org/x/tools/internal/stdlib"
	"golang.org/x/tools/internal/typesinternal"
)

//go:embed doc.go
var doc string

// Suite lists all modernize analyzers.
var Suite = []*analysis.Analyzer{
	AnyAnalyzer,
	atomicTypesAnalyzer,
	// AppendClippedAnalyzer, // not nil-preserving!
	// BLoopAnalyzer, // may skew benchmark results, see golang/go#74967
	FmtAppendfAnalyzer,
	ForVarAnalyzer,
	MapsLoopAnalyzer,
	MinMaxAnalyzer,
	NewExprAnalyzer,
	OmitZeroAnalyzer,
	plusBuildAnalyzer,
	RangeIntAnalyzer,
	ReflectTypeForAnalyzer,
	SlicesContainsAnalyzer,
	// SlicesDeleteAnalyzer, // not nil-preserving!
	SlicesSortAnalyzer,
	stditeratorsAnalyzer,
	stringscutAnalyzer,
	StringsCutPrefixAnalyzer,
	StringsSeqAnalyzer,
	StringsBuilderAnalyzer,
	TestingContextAnalyzer,
	unsafeFuncsAnalyzer,
	WaitGroupGoAnalyzer,
}

// -- helpers --

// formatExprs formats a comma-separated list of expressions.
func formatExprs(fset *token.FileSet, exprs []ast.Expr) string {
	var buf strings.Builder
	for i, e := range exprs {
		if i > 0 {
			buf.WriteString(",  ")
		}
		format.Node(&buf, fset, e) // ignore errors
	}
	return buf.String()
}

// isZeroIntConst reports whether e is an integer whose value is 0.
func isZeroIntConst(info *types.Info, e ast.Expr) bool {
	return isIntLiteral(info, e, 0)
}

// isIntLiteral reports whether e is an integer with given value.
func isIntLiteral(info *types.Info, e ast.Expr, n int64) bool {
	return info.Types[e].Value == constant.MakeInt64(n)
}

// filesUsingGoVersion returns a cursor for each *ast.File in the inspector
// that uses at least the specified version of Go (e.g. "go1.24").
//
// The pass's analyzer must require [inspect.Analyzer].
//
// TODO(adonovan): opt: eliminate this function, instead following the
// approach of [fmtappendf], which uses typeindex and
// [analyzerutil.FileUsesGoVersion]; see "Tip" documented at the
// latter function for motivation.
func filesUsingGoVersion(pass *analysis.Pass, version string) iter.Seq[inspector.Cursor] {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	return func(yield func(inspector.Cursor) bool) {
		for curFile := range inspect.Root().Children() {
			file := curFile.Node().(*ast.File)
			if analyzerutil.FileUsesGoVersion(pass, file, version) && !yield(curFile) {
				break
			}
		}
	}
}

// within reports whether the current pass is analyzing one of the
// specified standard packages or their dependencies.
func within(pass *analysis.Pass, pkgs ...string) bool {
	path := pass.Pkg.Path()
	return packagepath.IsStdPackage(path) &&
		moreiters.Contains(stdlib.Dependencies(pkgs...), path)
}

// unparenEnclosing removes enclosing parens from cur in
// preparation for a call to [Cursor.ParentEdge].
func unparenEnclosing(cur inspector.Cursor) inspector.Cursor {
	for cur.ParentEdgeKind() == edge.ParenExpr_X {
		cur = cur.Parent()
	}
	return cur
}

var (
	builtinAny     = types.Universe.Lookup("any")
	builtinAppend  = types.Universe.Lookup("append")
	builtinBool    = types.Universe.Lookup("bool")
	builtinInt     = types.Universe.Lookup("int")
	builtinFalse   = types.Universe.Lookup("false")
	builtinLen     = types.Universe.Lookup("len")
	builtinMake    = types.Universe.Lookup("make")
	builtinNew     = types.Universe.Lookup("new")
	builtinNil     = types.Universe.Lookup("nil")
	builtinString  = types.Universe.Lookup("string")
	builtinTrue    = types.Universe.Lookup("true")
	byteSliceType  = types.NewSlice(types.Typ[types.Byte])
	omitemptyRegex = regexp.MustCompile(`(?:^json| json):"[^"]*(,omitempty)(?:"|,[^"]*")\s?`)
)

// lookup returns the symbol denoted by name at the position of the cursor.
func lookup(info *types.Info, cur inspector.Cursor, name string) types.Object {
	scope := typesinternal.EnclosingScope(info, cur)
	_, obj := scope.LookupParent(name, cur.Node().Pos())
	return obj
}

func first[T any](x T, _ any) T { return x }

// freshName returns a fresh name at the given pos and scope based on preferredName.
// It generates a new name using refactor.FreshName only if:
// (a) the preferred name is already defined at definedCur, and
// (b) there are references to it from within usedCur.
// If useAfterPos.IsValid(), the references must be after
// useAfterPos within usedCur in order to warrant a fresh name.
// Otherwise, it returns preferredName, since shadowing is valid in this case.
// (declaredCur and usedCur may be identical in some use cases).
func freshName(info *types.Info, index *typeindex.Index, scope *types.Scope, pos token.Pos, defCur inspector.Cursor, useCur inspector.Cursor, useAfterPos token.Pos, preferredName string) string {
	obj := lookup(info, defCur, preferredName)
	if obj == nil {
		// preferredName has not been declared here.
		return preferredName
	}
	for use := range index.Uses(obj) {
		if useCur.Contains(use) && use.Node().Pos() >= useAfterPos {
			return refactor.FreshName(scope, pos, preferredName)
		}
	}
	// Name is taken but not used in the given block; shadowing is acceptable.
	return preferredName
}

// isLocal reports whether obj is local to some function.
// Precondition: not a struct field or interface method.
func isLocal(obj types.Object) bool {
	// [... 5=stmt 4=func 3=file 2=pkg 1=universe]
	var depth int
	for scope := obj.Parent(); scope != nil; scope = scope.Parent() {
		depth++
	}
	return depth >= 4
}
