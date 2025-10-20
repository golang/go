// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/analysisinternal/generated"
	typeindexanalyzer "golang.org/x/tools/internal/analysisinternal/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

var StringsBuilderAnalyzer = &analysis.Analyzer{
	Name: "stringsbuilder",
	Doc:  analysisinternal.MustExtractDoc(doc, "stringsbuilder"),
	Requires: []*analysis.Analyzer{
		generated.Analyzer,
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: stringsbuilder,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#stringbuilder",
}

// stringsbuilder replaces string += string in a loop by strings.Builder.
func stringsbuilder(pass *analysis.Pass) (any, error) {
	skipGenerated(pass)

	// Skip the analyzer in packages where its
	// fixes would create an import cycle.
	if within(pass, "strings", "runtime") {
		return nil, nil
	}

	var (
		inspect = pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		index   = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
	)

	// Gather all local string variables that appear on the
	// LHS of some string += string assignment.
	candidates := make(map[*types.Var]bool)
	for curAssign := range inspect.Root().Preorder((*ast.AssignStmt)(nil)) {
		assign := curAssign.Node().(*ast.AssignStmt)
		if assign.Tok == token.ADD_ASSIGN && is[*ast.Ident](assign.Lhs[0]) {
			if v, ok := pass.TypesInfo.Uses[assign.Lhs[0].(*ast.Ident)].(*types.Var); ok &&
				!typesinternal.IsPackageLevel(v) && // TODO(adonovan): in go1.25, use v.Kind() == types.LocalVar &&
				types.Identical(v.Type(), builtinString.Type()) {
				candidates[v] = true
			}
		}
	}

	// Now check each candidate variable's decl and uses.
nextcand:
	for v := range candidates {
		var edits []analysis.TextEdit

		// Check declaration of s:
		//
		//    s := expr
		//    var s [string] [= expr]
		//
		// and transform to:
		//
		//    var s strings.Builder; s.WriteString(expr)
		//
		def, ok := index.Def(v)
		if !ok {
			continue
		}
		ek, _ := def.ParentEdge()
		if ek == edge.AssignStmt_Lhs &&
			len(def.Parent().Node().(*ast.AssignStmt).Lhs) == 1 {
			// Have: s := expr
			// => var s strings.Builder; s.WriteString(expr)

			assign := def.Parent().Node().(*ast.AssignStmt)

			// Reject "if s := f(); ..." since in that context
			// we can't replace the assign with two statements.
			switch def.Parent().Parent().Node().(type) {
			case *ast.BlockStmt, *ast.CaseClause, *ast.CommClause:
				// OK: these are the parts of syntax that
				// allow unrestricted statement lists.
			default:
				continue
			}

			// Add strings import.
			prefix, importEdits := refactor.AddImport(
				pass.TypesInfo, astutil.EnclosingFile(def), "strings", "strings", "Builder", v.Pos())
			edits = append(edits, importEdits...)

			if isEmptyString(pass.TypesInfo, assign.Rhs[0]) {
				// s := ""
				// ---------------------
				// var s strings.Builder
				edits = append(edits, analysis.TextEdit{
					Pos:     assign.Pos(),
					End:     assign.End(),
					NewText: fmt.Appendf(nil, "var %[1]s %[2]sBuilder", v.Name(), prefix),
				})

			} else {
				// s :=                                 expr
				// -------------------------------------    -
				// var s strings.Builder; s.WriteString(expr)
				edits = append(edits, []analysis.TextEdit{
					{
						Pos:     assign.Pos(),
						End:     assign.Rhs[0].Pos(),
						NewText: fmt.Appendf(nil, "var %[1]s %[2]sBuilder; %[1]s.WriteString(", v.Name(), prefix),
					},
					{
						Pos:     assign.End(),
						End:     assign.End(),
						NewText: []byte(")"),
					},
				}...)

			}

		} else if ek == edge.ValueSpec_Names &&
			len(def.Parent().Node().(*ast.ValueSpec).Names) == 1 {
			// Have: var s [string] [= expr]
			// => var s strings.Builder; s.WriteString(expr)

			// Add strings import.
			prefix, importEdits := refactor.AddImport(
				pass.TypesInfo, astutil.EnclosingFile(def), "strings", "strings", "Builder", v.Pos())
			edits = append(edits, importEdits...)

			spec := def.Parent().Node().(*ast.ValueSpec)
			decl := def.Parent().Parent().Node().(*ast.GenDecl)

			init := spec.Names[0].End() // start of " = expr"
			if spec.Type != nil {
				init = spec.Type.End()
			}

			// var s [string]
			//      ----------------
			// var s strings.Builder
			edits = append(edits, analysis.TextEdit{
				Pos:     spec.Names[0].End(),
				End:     init,
				NewText: fmt.Appendf(nil, " %sBuilder", prefix),
			})

			if len(spec.Values) > 0 && !isEmptyString(pass.TypesInfo, spec.Values[0]) {
				// =               expr
				// ----------------    -
				// ; s.WriteString(expr)
				edits = append(edits, []analysis.TextEdit{
					{
						Pos:     init,
						End:     spec.Values[0].Pos(),
						NewText: fmt.Appendf(nil, "; %s.WriteString(", v.Name()),
					},
					{
						Pos:     decl.End(),
						End:     decl.End(),
						NewText: []byte(")"),
					},
				}...)
			} else {
				// delete "= expr"
				edits = append(edits, analysis.TextEdit{
					Pos: init,
					End: spec.End(),
				})
			}

		} else {
			continue
		}

		// Check uses of s.
		//
		// - All uses of s except the final one must be of the form
		//
		//    s += expr
		//
		//   Each of these will become s.WriteString(expr).
		//   At least one of them must be in an intervening loop
		//   w.r.t. the declaration of s:
		//
		//    var s string
		//    for ... { s += expr }
		//
		// - The final use of s must be as an rvalue (e.g. use(s), not &s).
		//   This will become s.String().
		//
		//   Perhaps surprisingly, it is fine for there to be an
		//   intervening loop or lambda w.r.t. the declaration of s:
		//
		//    var s strings.Builder
		//    for range kSmall { s.WriteString(expr) }
		//    for range kLarge { use(s.String()) } // called repeatedly
		//
		//   Even though that might cause the s.String() operation to be
		//   executed repeatedly, this is not a deoptimization because,
		//   by design, (*strings.Builder).String does not allocate.
		var (
			numLoopAssigns int             // number of += assignments within a loop
			loopAssign     *ast.AssignStmt // first += assignment within a loop
			seenRvalueUse  bool            // => we've seen the sole final use of s as an rvalue
		)
		for curUse := range index.Uses(v) {
			// Strip enclosing parens around Ident.
			ek, _ := curUse.ParentEdge()
			for ek == edge.ParenExpr_X {
				curUse = curUse.Parent()
				ek, _ = curUse.ParentEdge()
			}

			// The rvalueUse must be the lexically last use.
			if seenRvalueUse {
				continue nextcand
			}

			// intervening reports whether cur has an ancestor of
			// one of the given types that is within the scope of v.
			intervening := func(types ...ast.Node) bool {
				for cur := range curUse.Enclosing(types...) {
					if v.Pos() <= cur.Node().Pos() { // in scope of v
						return true
					}
				}
				return false
			}

			if ek == edge.AssignStmt_Lhs {
				assign := curUse.Parent().Node().(*ast.AssignStmt)
				if assign.Tok != token.ADD_ASSIGN {
					continue nextcand
				}
				// Have: s += expr

				// At least one of the += operations
				// must appear within a loop.
				// relative to the declaration of s.
				if intervening((*ast.ForStmt)(nil), (*ast.RangeStmt)(nil)) {
					numLoopAssigns++
					if loopAssign == nil {
						loopAssign = assign
					}
				}

				// s +=          expr
				//  -------------    -
				// s.WriteString(expr)
				edits = append(edits, []analysis.TextEdit{
					// replace += with .WriteString()
					{
						Pos:     assign.TokPos,
						End:     assign.Rhs[0].Pos(),
						NewText: []byte(".WriteString("),
					},
					// insert ")"
					{
						Pos:     assign.End(),
						End:     assign.End(),
						NewText: []byte(")"),
					},
				}...)

			} else if ek == edge.UnaryExpr_X &&
				curUse.Parent().Node().(*ast.UnaryExpr).Op == token.AND {
				// Have: use(&s)
				continue nextcand // s is used as an lvalue; reject

			} else {
				// The only possible l-value uses of a string variable
				// are assignments (s=expr, s+=expr, etc) and &s.
				// (For strings, we can ignore method calls s.m().)
				// All other uses are r-values.
				seenRvalueUse = true

				edits = append(edits, analysis.TextEdit{
					// insert ".String()"
					Pos:     curUse.Node().End(),
					End:     curUse.Node().End(),
					NewText: []byte(".String()"),
				})
			}
		}
		if !seenRvalueUse {
			continue nextcand // no rvalue use; reject
		}
		if numLoopAssigns == 0 {
			continue nextcand // no += in a loop; reject
		}

		pass.Report(analysis.Diagnostic{
			Pos:     loopAssign.Pos(),
			End:     loopAssign.End(),
			Message: "using string += string in a loop is inefficient",
			SuggestedFixes: []analysis.SuggestedFix{{
				Message:   "Replace string += string with strings.Builder",
				TextEdits: edits,
			}},
		})
	}

	return nil, nil
}

// isEmptyString reports whether e (a string-typed expression) has constant value "".
func isEmptyString(info *types.Info, e ast.Expr) bool {
	tv, ok := info.Types[e]
	return ok && tv.Value != nil && constant.StringVal(tv.Value) == ""
}
