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
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var ErrorsAsTypeAnalyzer = &analysis.Analyzer{
	Name:     "errorsastype",
	Doc:      analyzerutil.MustExtractDoc(doc, "errorsastype"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#errorsastype",
	Requires: []*analysis.Analyzer{typeindexanalyzer.Analyzer},
	Run:      errorsastype,
}

// errorsastype offers a fix to replace errors.As with the newer
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
// For the negated case (!errors.As), we use !ok instead.
//
// Note that the cmd/vet suite includes the "errorsas" analyzer, which
// detects actual mistakes in the use of errors.As. This logic does
// not belong in errorsas because the problems it fixes are merely
// stylistic.
//
// TODO(adonovan): support more cases:
// - if myerr := new(E); errors.As(err, myerr); { ... }
// - if errors.As(err, myerr) && othercond { ... }
func errorsastype(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo
	)

	for curCall := range index.Calls(index.Object("errors", "As")) {
		call := curCall.Node().(*ast.CallExpr)
		if len(call.Args) < 2 {
			continue // spread call: errors.As(pair())
		}

		v, curDeclStmt, curIfStmt := canUseErrorsAsType(info, index, curCall)
		if v == nil {
			continue
		}

		file := astutil.EnclosingFile(curDeclStmt)
		if !analyzerutil.FileUsesGoVersion(pass, file, versions.Go1_26) {
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
		// We generate a new name only if 'ok' is already declared at
		// curCall and it also used within the if-statement.
		ifScope := info.Scopes[curIfStmt.Node().(*ast.IfStmt)]
		negated := curCall.ParentEdgeKind() == edge.UnaryExpr_X // bool => Tok==NOT
		okName := freshName(info, index, ifScope, v.Pos(), curCall, curIfStmt, token.NoPos, "ok")
		// Because we reject any use of v outside the if statement, any use besides
		// the argument in errors.As must lie inside the if statement.
		usesV := moreiters.Len(index.Uses(v)) > 1

		edits := append(
			// delete "var myerr *MyErr"
			refactor.DeleteStmt(pass.Fset.File(call.Fun.Pos()), curDeclStmt),
			// if              errors.As            (err, &myerr)     { ... }
			//    -------------       --------------    -------- ----
			// if myerr, ok := errors.AsType[*MyErr](err        ); ok { ... }
			analysis.TextEdit{
				// Insert "myerr, ok := " if myerr is used inside the if statement.
				// Otherwise insert "_, ok := ".
				Pos:     call.Pos(),
				End:     call.Pos(),
				NewText: fmt.Appendf(nil, "%s, %s := ", cond(usesV, v.Name(), "_"), okName),
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
				// insert "; ok" for errors.AsType or "; !ok" for !errors.AsType
				Pos:     call.End(),
				End:     call.End(),
				NewText: fmt.Appendf(nil, "; %s%s", cond(negated, "!", ""), okName),
			},
		)
		if negated {
			unaryExpr := curCall.Parent().Node().(*ast.UnaryExpr)
			// delete "!"
			edits = append(edits, analysis.TextEdit{
				Pos: unaryExpr.OpPos,
				End: unaryExpr.X.Pos(),
			})
		}

		pass.Report(analysis.Diagnostic{
			Pos:     call.Fun.Pos(),
			End:     call.Fun.End(),
			Message: fmt.Sprintf("errors.As can be simplified using AsType[%s]", errtype),
			SuggestedFixes: []analysis.SuggestedFix{{
				Message:   fmt.Sprintf("Replace errors.As with AsType[%s]", errtype),
				TextEdits: edits,
			}},
		})
	}
	return nil, nil
}

// canUseErrorsAsType reports whether curCall is a call to errors.As beneath an
// if statement, preceded by a declaration of the typed error var. The var must
// not be used outside the if statement.
// If the conditions are met, it returns the error var, the cursor for its
// DeclStmt, and the cursor for the IfStmt that contains the call to errors.As.
// Otherwise it returns a nil error var.
func canUseErrorsAsType(info *types.Info, index *typeindex.Index, curCall inspector.Cursor) (_ *types.Var, curDeclStmt, curIfStmt inspector.Cursor) {
	curCond := curCall
	if curCond.ParentEdgeKind() == edge.UnaryExpr_X { // if !errors.As(err, &v)
		curCond = curCond.Parent()
	}
	if curCond.ParentEdgeKind() != edge.IfStmt_Cond {
		return // not beneath if or unaryexpr
	}
	curIfStmt = curCond.Parent()
	ifStmt := curIfStmt.Node().(*ast.IfStmt)
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
	if curDef.ParentEdgeKind() != edge.ValueSpec_Names {
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
	if curDecl.ParentEdgeKind() != edge.DeclStmt_Decl {
		return // package-level var, not a local declaration statement
	}
	// AsType requires that its type argument implements error.
	// Reject if v does not implement error.
	if !types.AssignableTo(v.Type(), errorType) {
		return
	}

	// Have:
	//   var v *MyErr
	//   ...
	//   if errors.As(err, &v) { ... }
	// with no uses of v outside the IfStmt.
	return v, curDecl.Parent(), curIfStmt
}
