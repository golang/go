// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/goplsexport"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var slicesBackwardAnalyzer = &analysis.Analyzer{
	Name: "slicesbackward",
	Doc:  analyzerutil.MustExtractDoc(doc, "slicesbackward"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: slicesbackward,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#slicesbackward",
}

func init() {
	// Export to gopls until this is a published modernizer.
	goplsexport.SlicesBackwardModernizer = slicesBackwardAnalyzer
}

// slicesbackward offers a fix to replace a manually-written backward loop:
//
//	for i := len(s) - 1; i >= 0; i-- {
//	    use(s[i])
//	}
//
// with a range loop using slices.Backward (added in Go 1.23):
//
//	for _, v := range slices.Backward(s) {
//	    use(v)
//	}
//
// If the loop index is needed beyond just indexing into the slice, both
// the index and value variables are kept:
//
//	for i, v := range slices.Backward(s) { ... }
func slicesbackward(pass *analysis.Pass) (any, error) {
	// Skip packages that are in the slices stdlib dependency tree to
	// avoid import cycles.
	if within(pass, "slices") {
		return nil, nil
	}

	var (
		info  = pass.TypesInfo
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
	)

	for curFile := range filesUsingGoVersion(pass, versions.Go1_23) {
		file := curFile.Node().(*ast.File)

	nextLoop:
		for curLoop := range curFile.Preorder((*ast.ForStmt)(nil)) {
			loop := curLoop.Node().(*ast.ForStmt)

			// Match init:  i := len(s) - 1   or   i = len(s) - 1
			init, ok := loop.Init.(*ast.AssignStmt)
			if !ok || !isSimpleAssign(init) {
				continue
			}
			indexIdent, ok := init.Lhs[0].(*ast.Ident)
			if !ok {
				continue
			}
			indexObj := info.ObjectOf(indexIdent).(*types.Var)

			// RHS must be len(s) - 1.
			binRhs, ok := init.Rhs[0].(*ast.BinaryExpr)
			if !ok || binRhs.Op != token.SUB {
				continue
			}
			if !isIntLiteral(info, binRhs.Y, 1) {
				continue
			}
			lenCall, ok := binRhs.X.(*ast.CallExpr)
			if !ok || typeutil.Callee(info, lenCall) != builtinLen {
				continue
			}
			if len(lenCall.Args) != 1 {
				continue
			}
			sliceExpr := lenCall.Args[0]
			if _, ok := info.TypeOf(sliceExpr).Underlying().(*types.Slice); !ok {
				continue
			}

			// Match cond:  i >= 0
			cond, ok := loop.Cond.(*ast.BinaryExpr)
			if !ok || cond.Op != token.GEQ {
				continue
			}
			if !astutil.EqualSyntax(cond.X, indexIdent) {
				continue
			}
			if !isZeroIntConst(info, cond.Y) {
				continue
			}

			// Match post:  i--
			dec, ok := loop.Post.(*ast.IncDecStmt)
			if !ok || dec.Tok != token.DEC {
				continue
			}
			if !astutil.EqualSyntax(dec.X, indexIdent) {
				continue
			}

			// Check that i is not used as an lvalue in the loop body.
			// If init is = (not :=), i is a pre-existing variable; also
			// check that it is not used as an lvalue outside the loop
			// (e.g. &i before the loop).
			bodyCur := curLoop.Child(loop.Body)
			for curUse := range index.Uses(indexObj) {
				if !isScalarLvalue(info, curUse) {
					continue
				}
				if bodyCur.Contains(curUse) {
					continue nextLoop // i is mutated in loop body
				}
				if init.Tok == token.ASSIGN && !curLoop.Contains(curUse) {
					continue nextLoop // pre-existing i is an lvalue outside the loop
				}
			}

			// Find all uses of i in the loop body. Classify as:
			//   s[i] — pure element accesses that can be replaced by the value var
			//   other — index used for non-indexing purposes
			var (
				// First assignment in the loop body of the form "name := s[i]"; or nil.
				firstSliceIdxAssign *ast.AssignStmt
				// List of s[i] expressions to replace by the value var (excludes firstSliceIdxAssign, which will be entirely removed).
				sliceIdxsReplace []*ast.IndexExpr
				// Total count of s[i] usages.
				sliceIdxs int
				// Non-indexing uses of i.
				otherUses int
			)
			for curUse := range index.Uses(indexObj) {
				if !bodyCur.Contains(curUse) {
					continue
				}
				// Is i in the Index position of an s[i] expression?
				// If so, we also need to check whether s[i] is an lvalue. If we're
				// mutating the slice or taking an element's address, a fix will not
				// be offered.
				if curUse.ParentEdgeKind() == edge.IndexExpr_Index {
					if isScalarLvalue(pass.TypesInfo, curUse.Parent()) {
						continue nextLoop
					}
					idxCur := curUse.Parent()
					idxExpr := idxCur.Node().(*ast.IndexExpr)
					if astutil.EqualSyntax(idxExpr.X, sliceExpr) {
						sliceIdxs++
						// If the current statement is the first in the body of the form
						// "name := s[i]", save it so we can use "name" as the value
						// variable in slices.Backward. We can also remove the entire assign
						// statement.
						if firstSliceIdxAssign == nil && idxCur.ParentEdgeKind() == edge.AssignStmt_Rhs {
							assignStmt := idxCur.Parent().Node().(*ast.AssignStmt)
							if len(assignStmt.Lhs) == 1 && assignStmt.Tok == token.DEFINE {
								// The condition above implies that assignStmt.Lhs[0] is a valid
								// identifier.
								firstSliceIdxAssign = assignStmt
								// We don't need to replace the index expr with the value variable
								// name if we are going to remove the entire assignment.
								continue
							}
						}
						sliceIdxsReplace = append(sliceIdxsReplace, idxExpr)
						continue
					}
				}
				otherUses++
			}

			// Build the suggested fix.
			//
			// for i := len(s) - 1;     i >= 0; i-- { ... s[i] ... }
			//     --------------------------------       ----
			// for _, v := range slices.Backward(s) { ... v    ... }
			sliceStr := astutil.Format(pass.Fset, sliceExpr)
			prefix, edits := refactor.AddImport(info, file, "slices", "slices", "Backward", loop.Pos())
			elemName := chooseValueName(firstSliceIdxAssign, sliceStr)
			elemName = freshName(info, index, info.Scopes[loop], loop.Pos(), bodyCur, bodyCur, token.NoPos, elemName)

			// Replace each s[i] with elemName (except for in the statement of the
			// form "name := s[i]" where we might have gotten elemName from - we will
			// delete this entire statement instead).
			for _, sx := range sliceIdxsReplace {
				edits = append(edits, analysis.TextEdit{
					Pos:     sx.Pos(),
					End:     sx.End(),
					NewText: []byte(elemName),
				})
			}

			if firstSliceIdxAssign != nil {
				edits = append(edits, analysis.TextEdit{
					Pos: firstSliceIdxAssign.Pos(),
					End: firstSliceIdxAssign.End(),
				})
			}

			// Replace the loop header with a range over slices.Backward. In
			// well-typed code, at least one of the index or value variables must be
			// referenced inside the loop body (otherUses + sliceIndexes > 0).
			var vars string
			if otherUses == 0 { // sliceIdxs > 0
				// All uses of i are s[i]; drop the index variable.
				vars = fmt.Sprintf("_, %s", elemName)
			} else if sliceIdxs == 0 { // otherUses > 0
				// Index i is not used in any s[i] expressions; drop the value variable.
				vars = indexIdent.Name
			} else { // otherUses > 0 && sliceIdxs > 0, keep both variables.
				vars = fmt.Sprintf("%s, %s", indexIdent.Name, elemName)
			}
			header := fmt.Sprintf("%s := range %sBackward(%s)", vars, prefix, sliceStr)
			edits = append(edits, analysis.TextEdit{
				Pos:     loop.Init.Pos(),
				End:     loop.Post.End(),
				NewText: []byte(header),
			})

			pass.Report(analysis.Diagnostic{
				Pos:     loop.Init.Pos(),
				End:     loop.Post.End(),
				Message: "backward loop over slice can be modernized using slices.Backward",
				SuggestedFixes: []analysis.SuggestedFix{{
					Message:   fmt.Sprintf("Replace with range slices.Backward(%s)", sliceStr),
					TextEdits: edits,
				}},
			})
		}
	}
	return nil, nil
}

// chooseValueName uses a heuristic to generate a name for the value variable in
// the call to slices.Backward.
func chooseValueName(assign *ast.AssignStmt, sliceStr string) string {
	if assign != nil {
		return assign.Lhs[0].(*ast.Ident).Name
	}
	// Heuristic: remove plural s suffix from slice var
	// if present, otherwise use first letter.
	if token.IsIdentifier(sliceStr) && len(sliceStr) > 1 {
		if single, ok := strings.CutSuffix(sliceStr, "s"); ok {
			return single
		}
		return sliceStr[:1] // first letter (assuming ASCII)
	}
	return "v"
}
