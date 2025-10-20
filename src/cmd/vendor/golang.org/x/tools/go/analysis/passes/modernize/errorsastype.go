// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"go/ast"
	"go/token"
	"go/types"

	"fmt"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/analysisinternal/generated"
	typeindexanalyzer "golang.org/x/tools/internal/analysisinternal/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/goplsexport"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

var errorsastypeAnalyzer = &analysis.Analyzer{
	Name:     "errorsastype",
	Doc:      analysisinternal.MustExtractDoc(doc, "errorsastype"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#errorsastype",
	Requires: []*analysis.Analyzer{generated.Analyzer, typeindexanalyzer.Analyzer},
	Run:      errorsastype,
}

func init() {
	// Export to gopls until this is a published modernizer.
	goplsexport.ErrorsAsTypeModernizer = errorsastypeAnalyzer
}

// errorsastype offers a fix to replace error.As with the newer
// errors.AsType[T] following this pattern:
//
//	var myerr *MyErr
//	if errors.As(err, &myerr) { ... }
//
// =>
//
//	if myerr, ok := errors.AsType[*MyErr](err); ok  { ... }
//
// (In principle several of these can then be chained using if/else,
// but we don't attempt that.)
//
// We offer the fix only within an if statement, but not within a
// switch case such as:
//
//	var myerr *MyErr
//	switch {
//	case errors.As(err, &myerr):
//	}
//
// because the transformation in that case would be ungainly.
//
// Note that the cmd/vet suite includes the "errorsas" analyzer, which
// detects actual mistakes in the use of errors.As. This logic does
// not belong in errorsas because the problems it fixes are merely
// stylistic.
//
// TODO(adonovan): support more cases:
//
//   - Negative cases
//     var myerr E
//     if !errors.As(err, &myerr) { ... }
//     =>
//     myerr, ok := errors.AsType[E](err)
//     if !ok { ... }
//
// - if myerr := new(E); errors.As(err, myerr); { ... }
//
// - if errors.As(err, myerr) && othercond { ... }
func errorsastype(pass *analysis.Pass) (any, error) {
	skipGenerated(pass)

	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo
	)

	for curCall := range index.Calls(index.Object("errors", "As")) {
		call := curCall.Node().(*ast.CallExpr)
		if len(call.Args) < 2 {
			continue // spread call: errors.As(pair())
		}

		v, curDeclStmt := canUseErrorsAsType(info, index, curCall)
		if v == nil {
			continue
		}

		file := astutil.EnclosingFile(curDeclStmt)
		if !fileUses(info, file, "go1.26") {
			continue // errors.AsType is too new
		}

		// Locate identifier "As" in errors.As.
		var asIdent *ast.Ident
		switch n := ast.Unparen(call.Fun).(type) {
		case *ast.Ident:
			asIdent = n // "errors" was dot-imported
		case *ast.SelectorExpr:
			asIdent = n.Sel
		default:
			panic("no Ident for errors.As")
		}

		// Format the type as valid Go syntax.
		// TODO(adonovan): fix: FileQualifier needs to respect
		// visibility at the current point, and either fail
		// or edit the imports as needed.
		// TODO(adonovan): fix: TypeString is not a sound way
		// to print types as Go syntax as it does not respect
		// symbol visibility, etc. We need something loosely
		// integrated with FileQualifier that accumulates
		// import edits, and may fail (e.g. for unexported
		// type or field names from other packages).
		// See https://go.dev/issues/75604.
		qual := typesinternal.FileQualifier(file, pass.Pkg)
		errtype := types.TypeString(v.Type(), qual)

		// Choose a name for the "ok" variable.
		okName := "ok"
		if okVar := lookup(info, curCall, "ok"); okVar != nil {
			// The name 'ok' is already declared, but
			// don't choose a fresh name unless okVar
			// is also used within the if-statement.
			curIf := curCall.Parent()
			for curUse := range index.Uses(okVar) {
				if curIf.Contains(curUse) {
					scope := info.Scopes[curIf.Node().(*ast.IfStmt)]
					okName = refactor.FreshName(scope, v.Pos(), "ok")
					break
				}
			}
		}

		pass.Report(analysis.Diagnostic{
			Pos:     call.Fun.Pos(),
			End:     call.Fun.End(),
			Message: fmt.Sprintf("errors.As can be simplified using AsType[%s]", errtype),
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: fmt.Sprintf("Replace errors.As with AsType[%s]", errtype),
				TextEdits: append(
					// delete "var myerr *MyErr"
					refactor.DeleteStmt(pass.Fset.File(call.Fun.Pos()), curDeclStmt),
					// if              errors.As            (err, &myerr)     { ... }
					//    -------------       --------------    -------- ----
					// if myerr, ok := errors.AsType[*MyErr](err        ); ok { ... }
					analysis.TextEdit{
						// insert "myerr, ok := "
						Pos:     call.Pos(),
						End:     call.Pos(),
						NewText: fmt.Appendf(nil, "%s, %s := ", v.Name(), okName),
					},
					analysis.TextEdit{
						// replace As with AsType[T]
						Pos:     asIdent.Pos(),
						End:     asIdent.End(),
						NewText: fmt.Appendf(nil, "AsType[%s]", errtype),
					},
					analysis.TextEdit{
						// delete ", &myerr"
						Pos: call.Args[0].End(),
						End: call.Args[1].End(),
					},
					analysis.TextEdit{
						// insert "; ok"
						Pos:     call.End(),
						End:     call.End(),
						NewText: fmt.Appendf(nil, "; %s", okName),
					},
				),
			}},
		})
	}
	return nil, nil
}

// canUseErrorsAsType reports whether curCall is a call to
// errors.As beneath an if statement, preceded by a
// declaration of the typed error var. The var must not be
// used outside the if statement.
func canUseErrorsAsType(info *types.Info, index *typeindex.Index, curCall inspector.Cursor) (_ *types.Var, _ inspector.Cursor) {
	if !astutil.IsChildOf(curCall, edge.IfStmt_Cond) {
		return // not beneath if statement
	}
	var (
		curIfStmt = curCall.Parent()
		ifStmt    = curIfStmt.Node().(*ast.IfStmt)
	)
	if ifStmt.Init != nil {
		return // if statement already has an init part
	}
	unary, ok := curCall.Node().(*ast.CallExpr).Args[1].(*ast.UnaryExpr)
	if !ok || unary.Op != token.AND {
		return // 2nd arg is not &var
	}
	id, ok := unary.X.(*ast.Ident)
	if !ok {
		return // not a simple ident (local var)
	}
	v := info.Uses[id].(*types.Var)
	curDef, ok := index.Def(v)
	if !ok {
		return // var is not local (e.g. dot-imported)
	}
	// Have: if errors.As(err, &v) { ... }

	// Reject if v is used outside (before or after) the
	// IfStmt, since that will become its new scope.
	for curUse := range index.Uses(v) {
		if !curIfStmt.Contains(curUse) {
			return // v used before/after if statement
		}
	}
	if !astutil.IsChildOf(curDef, edge.ValueSpec_Names) {
		return // v not declared by "var v T"
	}
	var (
		curSpec = curDef.Parent()  // ValueSpec
		curDecl = curSpec.Parent() // GenDecl
		spec    = curSpec.Node().(*ast.ValueSpec)
	)
	if len(spec.Names) != 1 || len(spec.Values) != 0 ||
		len(curDecl.Node().(*ast.GenDecl).Specs) != 1 {
		return // not a simple "var v T" decl
	}

	// Have:
	//   var v *MyErr
	//   ...
	//   if errors.As(err, &v) { ... }
	// with no uses of v outside the IfStmt.
	return v, curDecl.Parent() // DeclStmt
}
