// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

// This file defines modernizers that use the "maps" package.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/analysisinternal/generated"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

var MapsLoopAnalyzer = &analysis.Analyzer{
	Name: "mapsloop",
	Doc:  analysisinternal.MustExtractDoc(doc, "mapsloop"),
	Requires: []*analysis.Analyzer{
		generated.Analyzer,
		inspect.Analyzer,
	},
	Run: mapsloop,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#mapsloop",
}

// The mapsloop pass offers to simplify a loop of map insertions:
//
//	for k, v := range x {
//		m[k] = v
//	}
//
// by a call to go1.23's maps package. There are four variants, the
// product of two axes: whether the source x is a map or an iter.Seq2,
// and whether the destination m is a newly created map:
//
//	maps.Copy(m, x)		(x is map)
//	maps.Insert(m, x)       (x is iter.Seq2)
//	m = maps.Clone(x)       (x is a non-nil map, m is a new map)
//	m = maps.Collect(x)     (x is iter.Seq2, m is a new map)
//
// A map is newly created if the preceding statement has one of these
// forms, where M is a map type:
//
//	m = make(M)
//	m = M{}
func mapsloop(pass *analysis.Pass) (any, error) {
	skipGenerated(pass)

	// Skip the analyzer in packages where its
	// fixes would create an import cycle.
	if within(pass, "maps", "bytes", "runtime") {
		return nil, nil
	}

	info := pass.TypesInfo

	// check is called for each statement of this form:
	//   for k, v := range x { m[k] = v }
	check := func(file *ast.File, curRange inspector.Cursor, assign *ast.AssignStmt, m, x ast.Expr) {

		// Is x a map or iter.Seq2?
		tx := types.Unalias(info.TypeOf(x))
		var xmap bool
		switch typeparams.CoreType(tx).(type) {
		case *types.Map:
			xmap = true

		case *types.Signature:
			k, v, ok := assignableToIterSeq2(tx)
			if !ok {
				return // a named isomer of Seq2
			}
			xmap = false

			// Record in tx the unnamed map[K]V type
			// derived from the yield function.
			// This is the type of maps.Collect(x).
			tx = types.NewMap(k, v)

		default:
			return // e.g. slice, channel (or no core type!)
		}

		// Is the preceding statement of the form
		//    m = make(M) or M{}
		// and can we replace its RHS with slices.{Clone,Collect}?
		//
		// Beware: if x may be nil, we cannot use Clone as it preserves nilness.
		var mrhs ast.Expr // make(M) or M{}, or nil
		if curPrev, ok := curRange.PrevSibling(); ok {
			if assign, ok := curPrev.Node().(*ast.AssignStmt); ok &&
				len(assign.Lhs) == 1 &&
				len(assign.Rhs) == 1 &&
				astutil.EqualSyntax(assign.Lhs[0], m) {

				// Have: m = rhs; for k, v := range x { m[k] = v }
				var newMap bool
				rhs := assign.Rhs[0]
				switch rhs := ast.Unparen(rhs).(type) {
				case *ast.CallExpr:
					if id, ok := ast.Unparen(rhs.Fun).(*ast.Ident); ok &&
						info.Uses[id] == builtinMake {
						// Have: m = make(...)
						newMap = true
					}
				case *ast.CompositeLit:
					if len(rhs.Elts) == 0 {
						// Have m = M{}
						newMap = true
					}
				}

				// Take care not to change type of m's RHS expression.
				if newMap {
					trhs := info.TypeOf(rhs)

					// Inv: tx is the type of maps.F(x)
					// - maps.Clone(x) has the same type as x.
					// - maps.Collect(x) returns an unnamed map type.

					if assign.Tok == token.DEFINE {
						// DEFINE (:=): we must not
						// change the type of RHS.
						if types.Identical(tx, trhs) {
							mrhs = rhs
						}
					} else {
						// ASSIGN (=): the types of LHS
						// and RHS may differ in namedness.
						if types.AssignableTo(tx, trhs) {
							mrhs = rhs
						}
					}

					// Temporarily disable the transformation to the
					// (nil-preserving) maps.Clone until we can prove
					// that x is non-nil. This is rarely possible,
					// and may require control flow analysis
					// (e.g. a dominating "if len(x)" check).
					// See #71844.
					if xmap {
						mrhs = nil
					}
				}
			}
		}

		// Choose function.
		var funcName string
		if mrhs != nil {
			funcName = cond(xmap, "Clone", "Collect")
		} else {
			funcName = cond(xmap, "Copy", "Insert")
		}

		// Report diagnostic, and suggest fix.
		rng := curRange.Node()
		prefix, importEdits := refactor.AddImport(info, file, "maps", "maps", funcName, rng.Pos())
		var (
			newText    []byte
			start, end token.Pos
		)
		if mrhs != nil {
			// Replace assignment and loop with expression.
			//
			//   m = make(...)
			//   for k, v := range x { /* comments */ m[k] = v }
			//
			//   ->
			//
			//   /* comments */
			//   m = maps.Copy(x)
			curPrev, _ := curRange.PrevSibling()
			start, end = curPrev.Node().Pos(), rng.End()
			newText = fmt.Appendf(nil, "%s%s = %s%s(%s)",
				allComments(file, start, end),
				astutil.Format(pass.Fset, m),
				prefix,
				funcName,
				astutil.Format(pass.Fset, x))
		} else {
			// Replace loop with call statement.
			//
			//   for k, v := range x { /* comments */ m[k] = v }
			//
			//   ->
			//
			//   /* comments */
			//   maps.Copy(m, x)
			start, end = rng.Pos(), rng.End()
			newText = fmt.Appendf(nil, "%s%s%s(%s, %s)",
				allComments(file, start, end),
				prefix,
				funcName,
				astutil.Format(pass.Fset, m),
				astutil.Format(pass.Fset, x))
		}
		pass.Report(analysis.Diagnostic{
			Pos:     assign.Lhs[0].Pos(),
			End:     assign.Lhs[0].End(),
			Message: "Replace m[k]=v loop with maps." + funcName,
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: "Replace m[k]=v loop with maps." + funcName,
				TextEdits: append(importEdits, []analysis.TextEdit{{
					Pos:     start,
					End:     end,
					NewText: newText,
				}}...),
			}},
		})

	}

	// Find all range loops around m[k] = v.
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	for curFile := range filesUsing(inspect, pass.TypesInfo, "go1.23") {
		file := curFile.Node().(*ast.File)

		for curRange := range curFile.Preorder((*ast.RangeStmt)(nil)) {
			rng := curRange.Node().(*ast.RangeStmt)

			if rng.Tok == token.DEFINE &&
				rng.Key != nil &&
				rng.Value != nil &&
				isAssignBlock(rng.Body) {
				// Have: for k, v := range x { lhs = rhs }

				assign := rng.Body.List[0].(*ast.AssignStmt)
				if index, ok := assign.Lhs[0].(*ast.IndexExpr); ok &&
					astutil.EqualSyntax(rng.Key, index.Index) &&
					astutil.EqualSyntax(rng.Value, assign.Rhs[0]) &&
					is[*types.Map](typeparams.CoreType(info.TypeOf(index.X))) &&
					types.Identical(info.TypeOf(index), info.TypeOf(rng.Value)) { // m[k], v

					// Have: for k, v := range x { m[k] = v }
					// where there is no implicit conversion.
					check(file, curRange, assign, index.X, rng.X)
				}
			}
		}
	}
	return nil, nil
}

// assignableToIterSeq2 reports whether t is assignable to
// iter.Seq[K, V] and returns K and V if so.
func assignableToIterSeq2(t types.Type) (k, v types.Type, ok bool) {
	// The only named type assignable to iter.Seq2 is iter.Seq2.
	if is[*types.Named](t) {
		if !typesinternal.IsTypeNamed(t, "iter", "Seq2") {
			return
		}
		t = t.Underlying()
	}

	if t, ok := t.(*types.Signature); ok {
		// func(yield func(K, V) bool)?
		if t.Params().Len() == 1 && t.Results().Len() == 0 {
			if yield, ok := t.Params().At(0).Type().(*types.Signature); ok { // sic, no Underlying/CoreType
				if yield.Params().Len() == 2 &&
					yield.Results().Len() == 1 &&
					types.Identical(yield.Results().At(0).Type(), builtinBool.Type()) {
					return yield.Params().At(0).Type(), yield.Params().At(1).Type(), true
				}
			}
		}
	}
	return
}
