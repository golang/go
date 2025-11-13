// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/analysisinternal/generated"
	typeindexanalyzer "golang.org/x/tools/internal/analysisinternal/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

var SlicesContainsAnalyzer = &analysis.Analyzer{
	Name: "slicescontains",
	Doc:  analysisinternal.MustExtractDoc(doc, "slicescontains"),
	Requires: []*analysis.Analyzer{
		generated.Analyzer,
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: slicescontains,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#slicescontains",
}

// The slicescontains pass identifies loops that can be replaced by a
// call to slices.Contains{,Func}. For example:
//
//	for i, elem := range s {
//		if elem == needle {
//			...
//			break
//		}
//	}
//
// =>
//
//	if slices.Contains(s, needle) { ... }
//
// Variants:
//   - if the if-condition is f(elem), the replacement
//     uses slices.ContainsFunc(s, f).
//   - if the if-body is "return true" and the fallthrough
//     statement is "return false" (or vice versa), the
//     loop becomes "return [!]slices.Contains(...)".
//   - if the if-body is "found = true" and the previous
//     statement is "found = false" (or vice versa), the
//     loop becomes "found = [!]slices.Contains(...)".
//
// It may change cardinality of effects of the "needle" expression.
// (Mostly this appears to be a desirable optimization, avoiding
// redundantly repeated evaluation.)
//
// TODO(adonovan): Add a check that needle/predicate expression from
// if-statement has no effects. Now the program behavior may change.
func slicescontains(pass *analysis.Pass) (any, error) {
	skipGenerated(pass)

	// Skip the analyzer in packages where its
	// fixes would create an import cycle.
	if within(pass, "slices", "runtime") {
		return nil, nil
	}

	var (
		inspect = pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		index   = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info    = pass.TypesInfo
	)

	// check is called for each RangeStmt of this form:
	//   for i, elem := range s { if cond { ... } }
	check := func(file *ast.File, curRange inspector.Cursor) {
		rng := curRange.Node().(*ast.RangeStmt)
		ifStmt := rng.Body.List[0].(*ast.IfStmt)

		// isSliceElem reports whether e denotes the
		// current slice element (elem or s[i]).
		isSliceElem := func(e ast.Expr) bool {
			if rng.Value != nil && astutil.EqualSyntax(e, rng.Value) {
				return true // "elem"
			}
			if x, ok := e.(*ast.IndexExpr); ok &&
				astutil.EqualSyntax(x.X, rng.X) &&
				astutil.EqualSyntax(x.Index, rng.Key) {
				return true // "s[i]"
			}
			return false
		}

		// Examine the condition for one of these forms:
		//
		// - if elem or s[i] == needle  { ... } => Contains
		// - if predicate(s[i] or elem) { ... } => ContainsFunc
		var (
			funcName string   // "Contains" or "ContainsFunc"
			arg2     ast.Expr // second argument to func (needle or predicate)
		)
		switch cond := ifStmt.Cond.(type) {
		case *ast.BinaryExpr:
			if cond.Op == token.EQL {
				var elem ast.Expr
				if isSliceElem(cond.X) {
					funcName = "Contains"
					elem = cond.X
					arg2 = cond.Y // "if elem == needle"
				} else if isSliceElem(cond.Y) {
					funcName = "Contains"
					elem = cond.Y
					arg2 = cond.X // "if needle == elem"
				}

				// Reject if elem and needle have different types.
				if elem != nil {
					tElem := info.TypeOf(elem)
					tNeedle := info.TypeOf(arg2)
					if !types.Identical(tElem, tNeedle) {
						// Avoid ill-typed slices.Contains([]error, any).
						if !types.AssignableTo(tNeedle, tElem) {
							return
						}
						// TODO(adonovan): relax this check to allow
						//   slices.Contains([]error, error(any)),
						// inserting an explicit widening conversion
						// around the needle.
						return
					}
				}
			}

		case *ast.CallExpr:
			if len(cond.Args) == 1 &&
				isSliceElem(cond.Args[0]) &&
				typeutil.Callee(info, cond) != nil { // not a conversion

				// Attempt to get signature
				sig, isSignature := info.TypeOf(cond.Fun).(*types.Signature)
				if isSignature {
					// skip variadic functions
					if sig.Variadic() {
						return
					}

					// Slice element type must match function parameter type.
					var (
						tElem  = typeparams.CoreType(info.TypeOf(rng.X)).(*types.Slice).Elem()
						tParam = sig.Params().At(0).Type()
					)
					if !types.Identical(tElem, tParam) {
						return
					}
				}

				funcName = "ContainsFunc"
				arg2 = cond.Fun // "if predicate(elem)"
			}
		}
		if funcName == "" {
			return // not a candidate for Contains{,Func}
		}

		// body is the "true" body.
		body := ifStmt.Body
		if len(body.List) == 0 {
			// (We could perhaps delete the loop entirely.)
			return
		}

		// Reject if the body, needle or predicate references either range variable.
		usesRangeVar := func(n ast.Node) bool {
			cur, ok := curRange.FindNode(n)
			if !ok {
				panic(fmt.Sprintf("FindNode(%T) failed", n))
			}
			return uses(index, cur, info.Defs[rng.Key.(*ast.Ident)]) ||
				rng.Value != nil && uses(index, cur, info.Defs[rng.Value.(*ast.Ident)])
		}
		if usesRangeVar(body) {
			// Body uses range var "i" or "elem".
			//
			// (The check for "i" could be relaxed when we
			// generalize this to support slices.Index;
			// and the check for "elem" could be relaxed
			// if "elem" can safely be replaced in the
			// body by "needle".)
			return
		}
		if usesRangeVar(arg2) {
			return
		}

		// Prepare slices.Contains{,Func} call.
		prefix, importEdits := refactor.AddImport(info, file, "slices", "slices", funcName, rng.Pos())
		contains := fmt.Sprintf("%s%s(%s, %s)",
			prefix,
			funcName,
			astutil.Format(pass.Fset, rng.X),
			astutil.Format(pass.Fset, arg2))

		report := func(edits []analysis.TextEdit) {
			pass.Report(analysis.Diagnostic{
				Pos:     rng.Pos(),
				End:     rng.End(),
				Message: fmt.Sprintf("Loop can be simplified using slices.%s", funcName),
				SuggestedFixes: []analysis.SuggestedFix{{
					Message:   "Replace loop by call to slices." + funcName,
					TextEdits: append(edits, importEdits...),
				}},
			})
		}

		// Last statement of body must return/break out of the loop.
		//
		// TODO(adonovan): opt:consider avoiding FindNode with new API of form:
		//    curRange.Get(edge.RangeStmt_Body, -1).
		//             Get(edge.BodyStmt_List, 0).
		//             Get(edge.IfStmt_Body)
		curBody, _ := curRange.FindNode(body)
		curLastStmt, _ := curBody.LastChild()

		// Reject if any statement in the body except the
		// last has a free continuation (continue or break)
		// that might affected by melting down the loop.
		//
		// TODO(adonovan): relax check by analyzing branch target.
		for curBodyStmt := range curBody.Children() {
			if curBodyStmt != curLastStmt {
				for range curBodyStmt.Preorder((*ast.BranchStmt)(nil), (*ast.ReturnStmt)(nil)) {
					return
				}
			}
		}

		switch lastStmt := curLastStmt.Node().(type) {
		case *ast.ReturnStmt:
			// Have: for ... range seq { if ... { stmts; return x } }

			// Special case:
			// body={ return true } next="return false"   (or negation)
			// => return [!]slices.Contains(...)
			if curNext, ok := curRange.NextSibling(); ok {
				nextStmt := curNext.Node().(ast.Stmt)
				tval := isReturnTrueOrFalse(info, lastStmt)
				fval := isReturnTrueOrFalse(info, nextStmt)
				if len(body.List) == 1 && tval*fval < 0 {
					//    for ... { if ... { return true/false } }
					// => return [!]slices.Contains(...)
					report([]analysis.TextEdit{
						// Delete the range statement and following space.
						{
							Pos: rng.Pos(),
							End: nextStmt.Pos(),
						},
						// Change return to [!]slices.Contains(...).
						{
							Pos: nextStmt.Pos(),
							End: nextStmt.End(),
							NewText: fmt.Appendf(nil, "return %s%s",
								cond(tval > 0, "", "!"),
								contains),
						},
					})
					return
				}
			}

			// General case:
			// => if slices.Contains(...) { stmts; return x }
			report([]analysis.TextEdit{
				// Replace "for ... { if ... " with "if slices.Contains(...)".
				{
					Pos:     rng.Pos(),
					End:     ifStmt.Body.Pos(),
					NewText: fmt.Appendf(nil, "if %s ", contains),
				},
				// Delete '}' of range statement and preceding space.
				{
					Pos: ifStmt.Body.End(),
					End: rng.End(),
				},
			})
			return

		case *ast.BranchStmt:
			if lastStmt.Tok == token.BREAK && lastStmt.Label == nil { // unlabeled break
				// Have: for ... { if ... { stmts; break } }

				var prevStmt ast.Stmt // previous statement to range (if any)
				if curPrev, ok := curRange.PrevSibling(); ok {
					// If the RangeStmt's previous sibling is a Stmt,
					// the RangeStmt must be among the Body list of
					// a BlockStmt, CauseClause, or CommClause.
					// In all cases, the prevStmt is the immediate
					// predecessor of the RangeStmt during execution.
					//
					// (This is not true for Stmts in general;
					// see [Cursor.Children] and #71074.)
					prevStmt, _ = curPrev.Node().(ast.Stmt)
				}

				// Special case:
				// prev="lhs = false" body={ lhs = true; break }
				// => lhs = slices.Contains(...) (or negation)
				if assign, ok := body.List[0].(*ast.AssignStmt); ok &&
					len(body.List) == 2 &&
					assign.Tok == token.ASSIGN &&
					len(assign.Lhs) == 1 &&
					len(assign.Rhs) == 1 {

					// Have: body={ lhs = rhs; break }

					if prevAssign, ok := prevStmt.(*ast.AssignStmt); ok &&
						len(prevAssign.Lhs) == 1 &&
						len(prevAssign.Rhs) == 1 &&
						astutil.EqualSyntax(prevAssign.Lhs[0], assign.Lhs[0]) &&
						is[*ast.Ident](assign.Rhs[0]) &&
						info.Uses[assign.Rhs[0].(*ast.Ident)] == builtinTrue {

						// Have:
						//    lhs = false
						//    for ... { if ... { lhs = true; break } }
						//  =>
						//    lhs = slices.Contains(...)
						//
						// TODO(adonovan):
						// - support "var lhs bool = false" and variants.
						// - support negation.
						// Both these variants seem quite significant.
						// - allow the break to be omitted.
						report([]analysis.TextEdit{
							// Replace "rhs" of previous assignment by slices.Contains(...)
							{
								Pos:     prevAssign.Rhs[0].Pos(),
								End:     prevAssign.Rhs[0].End(),
								NewText: []byte(contains),
							},
							// Delete the loop and preceding space.
							{
								Pos: prevAssign.Rhs[0].End(),
								End: rng.End(),
							},
						})
						return
					}
				}

				// General case:
				//    for ... { if ...        { stmts; break } }
				// => if slices.Contains(...) { stmts        }
				report([]analysis.TextEdit{
					// Replace "for ... { if ... " with "if slices.Contains(...)".
					{
						Pos:     rng.Pos(),
						End:     ifStmt.Body.Pos(),
						NewText: fmt.Appendf(nil, "if %s ", contains),
					},
					// Delete break statement and preceding space.
					{
						Pos: func() token.Pos {
							if len(body.List) > 1 {
								beforeBreak, _ := curLastStmt.PrevSibling()
								return beforeBreak.Node().End()
							}
							return lastStmt.Pos()
						}(),
						End: lastStmt.End(),
					},
					// Delete '}' of range statement and preceding space.
					{
						Pos: ifStmt.Body.End(),
						End: rng.End(),
					},
				})
				return
			}
		}
	}

	for curFile := range filesUsing(inspect, info, "go1.21") {
		file := curFile.Node().(*ast.File)

		for curRange := range curFile.Preorder((*ast.RangeStmt)(nil)) {
			rng := curRange.Node().(*ast.RangeStmt)

			if is[*ast.Ident](rng.Key) &&
				rng.Tok == token.DEFINE &&
				len(rng.Body.List) == 1 &&
				is[*types.Slice](typeparams.CoreType(info.TypeOf(rng.X))) {

				// Have:
				// - for _, elem := range s { S }
				// - for i       := range s { S }

				if ifStmt, ok := rng.Body.List[0].(*ast.IfStmt); ok &&
					ifStmt.Init == nil && ifStmt.Else == nil {

					// Have: for i, elem := range s { if cond { ... } }
					check(file, curRange)
				}
			}
		}
	}
	return nil, nil
}

// -- helpers --

// isReturnTrueOrFalse returns nonzero if stmt returns true (+1) or false (-1).
func isReturnTrueOrFalse(info *types.Info, stmt ast.Stmt) int {
	if ret, ok := stmt.(*ast.ReturnStmt); ok && len(ret.Results) == 1 {
		if id, ok := ret.Results[0].(*ast.Ident); ok {
			switch info.Uses[id] {
			case builtinTrue:
				return +1
			case builtinFalse:
				return -1
			}
		}
	}
	return 0
}
