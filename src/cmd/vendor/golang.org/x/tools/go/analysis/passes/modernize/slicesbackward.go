// Copyright 2025 The Go Authors. All rights reserved.
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

var slicesbackwardAnalyzer = &analysis.Analyzer{
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
	goplsexport.SlicesBackwardModernizer = slicesbackwardAnalyzer
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
				sliceIndexes []*ast.IndexExpr
				otherUses    int
			)
			for curUse := range index.Uses(indexObj) {
				if !bodyCur.Contains(curUse) {
					continue
				}
				// Is i in the Index position of an s[i] expression?
				if curUse.ParentEdgeKind() == edge.IndexExpr_Index {
					idxExpr := curUse.Parent().Node().(*ast.IndexExpr)
					if astutil.EqualSyntax(idxExpr.X, sliceExpr) {
						sliceIndexes = append(sliceIndexes, idxExpr)
						continue
					}
				}
				otherUses++
			}

			// Build the suggested fix.
			//
			// for i := len(s) - 1; i >= 0; i-- { ... s[i] ... }
			//     ----------------------------         ----
			//     _, v := range slices.Backward(s)      v
			sliceStr := astutil.Format(pass.Fset, sliceExpr)
			prefix, edits := refactor.AddImport(info, file, "slices", "slices", "Backward", loop.Pos())
			elemName := freshName(info, index, info.Scopes[loop], loop.Pos(), bodyCur, bodyCur, token.NoPos, "v")

			// Replace each s[i] with elemName.
			for _, sx := range sliceIndexes {
				edits = append(edits, analysis.TextEdit{
					Pos:     sx.Pos(),
					End:     sx.End(),
					NewText: []byte(elemName),
				})
			}

			// Replace the loop header with a range over slices.Backward.
			var header string
			if otherUses == 0 && len(sliceIndexes) > 0 {
				// All uses of i are s[i]; drop the index variable.
				header = fmt.Sprintf("_, %s := range %sBackward(%s)",
					elemName, prefix, sliceStr)
			} else {
				// i is used for other purposes; keep both index and value.
				header = fmt.Sprintf("%s, %s := range %sBackward(%s)",
					indexIdent.Name, elemName, prefix, sliceStr)
			}
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
