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
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal/generated"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/packagepath"
	"golang.org/x/tools/internal/stdlib"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/versions"
)

//go:embed doc.go
var doc string

// Suite lists all modernize analyzers.
var Suite = []*analysis.Analyzer{
	AnyAnalyzer,
	// AppendClippedAnalyzer, // not nil-preserving!
	BLoopAnalyzer,
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
	StringsCutPrefixAnalyzer,
	StringsSeqAnalyzer,
	StringsBuilderAnalyzer,
	TestingContextAnalyzer,
	WaitGroupAnalyzer,
}

// -- helpers --

// skipGenerated decorates pass.Report to suppress diagnostics in generated files.
func skipGenerated(pass *analysis.Pass) {
	report := pass.Report
	pass.Report = func(diag analysis.Diagnostic) {
		generated := pass.ResultOf[generated.Analyzer].(*generated.Result)
		if generated.IsGenerated(diag.Pos) {
			return // skip
		}
		report(diag)
	}
}

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

// isZeroIntLiteral reports whether e is an integer whose value is 0.
func isZeroIntLiteral(info *types.Info, e ast.Expr) bool {
	return isIntLiteral(info, e, 0)
}

// isIntLiteral reports whether e is an integer with given value.
func isIntLiteral(info *types.Info, e ast.Expr, n int64) bool {
	return info.Types[e].Value == constant.MakeInt64(n)
}

// filesUsing returns a cursor for each *ast.File in the inspector
// that uses at least the specified version of Go (e.g. "go1.24").
//
// TODO(adonovan): opt: eliminate this function, instead following the
// approach of [fmtappendf], which uses typeindex and [fileUses].
// See "Tip" at [fileUses] for motivation.
func filesUsing(inspect *inspector.Inspector, info *types.Info, version string) iter.Seq[inspector.Cursor] {
	return func(yield func(inspector.Cursor) bool) {
		for curFile := range inspect.Root().Children() {
			file := curFile.Node().(*ast.File)
			if !versions.Before(info.FileVersions[file], version) && !yield(curFile) {
				break
			}
		}
	}
}

// fileUses reports whether the specified file uses at least the
// specified version of Go (e.g. "go1.24").
//
// Tip: we recommend using this check "late", just before calling
// pass.Report, rather than "early" (when entering each ast.File, or
// each candidate node of interest, during the traversal), because the
// operation is not free, yet is not a highly selective filter: the
// fraction of files that pass most version checks is high and
// increases over time.
func fileUses(info *types.Info, file *ast.File, version string) bool {
	return !versions.Before(info.FileVersions[file], version)
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
	for astutil.IsChildOf(cur, edge.ParenExpr_X) {
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
