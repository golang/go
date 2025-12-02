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
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var RangeIntAnalyzer = &analysis.Analyzer{
	Name: "rangeint",
	Doc:  analyzerutil.MustExtractDoc(doc, "rangeint"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: rangeint,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#rangeint",
}

// rangeint offers a fix to replace a 3-clause 'for' loop:
//
//	for i := 0; i < limit; i++ {}
//
// by a range loop with an integer operand:
//
//	for i := range limit {}
//
// Variants:
//   - The ':=' may be replaced by '='.
//   - The fix may remove "i :=" if it would become unused.
//
// Restrictions:
//   - The variable i must not be assigned or address-taken within the
//     loop, because a "for range int" loop does not respect assignments
//     to the loop index.
//   - The limit must not be b.N, to avoid redundancy with bloop's fixes.
//
// Caveats:
//
// The fix causes the limit expression to be evaluated exactly once,
// instead of once per iteration. So, to avoid changing the
// cardinality of side effects, the limit expression must not involve
// function calls (e.g. seq.Len()) or channel receives. Moreover, the
// value of the limit expression must be loop invariant, which in
// practice means it must take one of the following forms:
//
//   - a local variable that is assigned only once and not address-taken;
//   - a constant; or
//   - len(s), where s has the above properties.
func rangeint(pass *analysis.Pass) (any, error) {
	var (
		info      = pass.TypesInfo
		typeindex = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
	)

	for curFile := range filesUsingGoVersion(pass, versions.Go1_22) {
	nextLoop:
		for curLoop := range curFile.Preorder((*ast.ForStmt)(nil)) {
			loop := curLoop.Node().(*ast.ForStmt)
			if init, ok := loop.Init.(*ast.AssignStmt); ok &&
				isSimpleAssign(init) &&
				is[*ast.Ident](init.Lhs[0]) &&
				isZeroIntConst(info, init.Rhs[0]) {
				// Have: for i = 0; ... (or i := 0)
				index := init.Lhs[0].(*ast.Ident)

				if compare, ok := loop.Cond.(*ast.BinaryExpr); ok &&
					compare.Op == token.LSS &&
					astutil.EqualSyntax(compare.X, init.Lhs[0]) {
					// Have: for i = 0; i < limit; ... {}

					limit := compare.Y

					// If limit is "len(slice)", simplify it to "slice".
					//
					// (Don't replace "for i := 0; i < len(map); i++"
					// with "for range m" because it's too hard to prove
					// that len(m) is loop-invariant).
					if call, ok := limit.(*ast.CallExpr); ok &&
						typeutil.Callee(info, call) == builtinLen &&
						is[*types.Slice](info.TypeOf(call.Args[0]).Underlying()) {
						limit = call.Args[0]
					}

					// Check the form of limit: must be a constant,
					// or a local var that is not assigned or address-taken.
					limitOK := false
					if info.Types[limit].Value != nil {
						limitOK = true // constant
					} else if id, ok := limit.(*ast.Ident); ok {
						if v, ok := info.Uses[id].(*types.Var); ok &&
							!(v.Exported() && typesinternal.IsPackageLevel(v)) {
							// limit is a local or unexported global var.
							// (An exported global may have uses we can't see.)
							for cur := range typeindex.Uses(v) {
								if isScalarLvalue(info, cur) {
									// Limit var is assigned or address-taken.
									continue nextLoop
								}
							}
							limitOK = true
						}
					}
					if !limitOK {
						continue nextLoop
					}

					if inc, ok := loop.Post.(*ast.IncDecStmt); ok &&
						inc.Tok == token.INC &&
						astutil.EqualSyntax(compare.X, inc.X) {
						// Have: for i = 0; i < limit; i++ {}

						// Find references to i within the loop body.
						v := info.ObjectOf(index).(*types.Var)
						// TODO(adonovan): use go1.25 v.Kind() == types.PackageVar
						if typesinternal.IsPackageLevel(v) {
							continue nextLoop
						}
						used := false
						for curId := range curLoop.Child(loop.Body).Preorder((*ast.Ident)(nil)) {
							id := curId.Node().(*ast.Ident)
							if info.Uses[id] == v {
								used = true

								// Reject if any is an l-value (assigned or address-taken):
								// a "for range int" loop does not respect assignments to
								// the loop variable.
								if isScalarLvalue(info, curId) {
									continue nextLoop
								}
							}
						}

						// If i is no longer used, delete "i := ".
						var edits []analysis.TextEdit
						if !used && init.Tok == token.DEFINE {
							edits = append(edits, analysis.TextEdit{
								Pos: index.Pos(),
								End: init.Rhs[0].Pos(),
							})
						}

						// If i is used after the loop,
						// don't offer a fix, as a range loop
						// leaves i with a different final value (limit-1).
						if init.Tok == token.ASSIGN {
							// Find the nearest ancestor that is not a label.
							// Otherwise, checking for i usage outside of a for
							// loop might not function properly further below.
							// This is because the i usage might be a child of
							// the loop's parent's parent, for example:
							//     var i int
							// Loop:
							//     for i = 0; i < 10; i++ { break loop }
							//     // i is in the sibling of the label, not the loop
							//     fmt.Println(i)
							//
							ancestor := curLoop.Parent()
							for is[*ast.LabeledStmt](ancestor.Node()) {
								ancestor = ancestor.Parent()
							}
							for curId := range ancestor.Preorder((*ast.Ident)(nil)) {
								id := curId.Node().(*ast.Ident)
								if info.Uses[id] == v {
									// Is i used after loop?
									if id.Pos() > loop.End() {
										continue nextLoop
									}
									// Is i used within a defer statement
									// that is within the scope of i?
									//     var i int
									//     defer func() { print(i)}
									//     for i = ... { ... }
									for curDefer := range curId.Enclosing((*ast.DeferStmt)(nil)) {
										if curDefer.Node().Pos() > v.Pos() {
											continue nextLoop
										}
									}
								}
							}
						}

						// If limit is len(slice),
						// simplify "range len(slice)" to "range slice".
						if call, ok := limit.(*ast.CallExpr); ok &&
							typeutil.Callee(info, call) == builtinLen &&
							is[*types.Slice](info.TypeOf(call.Args[0]).Underlying()) {
							limit = call.Args[0]
						}

						// If the limit is a untyped constant of non-integer type,
						// such as "const limit = 1e3", its effective type may
						// differ between the two forms.
						// In a for loop, it must be comparable with int i,
						//    for i := 0; i < limit; i++
						// but in a range loop it would become a float,
						//    for i := range limit {}
						// which is a type error. We need to convert it to int
						// in this case.
						//
						// Unfortunately go/types discards the untyped type
						// (but see Untyped in golang/go#70638) so we must
						// re-type check the expression to detect this case.
						var beforeLimit, afterLimit string
						if v := info.Types[limit].Value; v != nil {
							tVar := info.TypeOf(init.Rhs[0])
							file := curFile.Node().(*ast.File)
							// TODO(mkalil): use a types.Qualifier that respects the existing
							// imports of this file that are visible (not shadowed) at the current position.
							qual := typesinternal.FileQualifier(file, pass.Pkg)
							beforeLimit, afterLimit = fmt.Sprintf("%s(", types.TypeString(tVar, qual)), ")"
							info2 := &types.Info{Types: make(map[ast.Expr]types.TypeAndValue)}
							if types.CheckExpr(pass.Fset, pass.Pkg, limit.Pos(), limit, info2) == nil {
								tLimit := types.Default(info2.TypeOf(limit))
								if types.AssignableTo(tLimit, tVar) {
									beforeLimit, afterLimit = "", ""
								}
							}
						}

						pass.Report(analysis.Diagnostic{
							Pos:     init.Pos(),
							End:     inc.End(),
							Message: "for loop can be modernized using range over int",
							SuggestedFixes: []analysis.SuggestedFix{{
								Message: fmt.Sprintf("Replace for loop with range %s",
									astutil.Format(pass.Fset, limit)),
								TextEdits: append(edits, []analysis.TextEdit{
									// for i := 0; i < limit; i++ {}
									//     -----              ---
									//          -------
									// for i := range  limit      {}

									// Delete init.
									{
										Pos:     init.Rhs[0].Pos(),
										End:     limit.Pos(),
										NewText: []byte("range "),
									},
									// Add "int(" before limit, if needed.
									{
										Pos:     limit.Pos(),
										End:     limit.Pos(),
										NewText: []byte(beforeLimit),
									},
									// Delete inc.
									{
										Pos: limit.End(),
										End: inc.End(),
									},
									// Add ")" after limit, if needed.
									{
										Pos:     limit.End(),
										End:     limit.End(),
										NewText: []byte(afterLimit),
									},
								}...),
							}},
						})
					}
				}
			}
		}
	}
	return nil, nil
}

// isScalarLvalue reports whether the specified identifier is
// address-taken or appears on the left side of an assignment.
//
// This function is valid only for scalars (x = ...),
// not for aggregates (x.a[i] = ...)
func isScalarLvalue(info *types.Info, curId inspector.Cursor) bool {
	// Unfortunately we can't simply use info.Types[e].Assignable()
	// as it is always true for a variable even when that variable is
	// used only as an r-value. So we must inspect enclosing syntax.

	cur := curId

	// Strip enclosing parens.
	ek, _ := cur.ParentEdge()
	for ek == edge.ParenExpr_X {
		cur = cur.Parent()
		ek, _ = cur.ParentEdge()
	}

	switch ek {
	case edge.AssignStmt_Lhs:
		assign := cur.Parent().Node().(*ast.AssignStmt)
		if assign.Tok != token.DEFINE {
			return true // i = j or i += j
		}
		id := curId.Node().(*ast.Ident)
		if v, ok := info.Defs[id]; ok && v.Pos() != id.Pos() {
			return true // reassignment of i (i, j := 1, 2)
		}
	case edge.IncDecStmt_X:
		return true // i++, i--
	case edge.UnaryExpr_X:
		if cur.Parent().Node().(*ast.UnaryExpr).Op == token.AND {
			return true // &i
		}
	}
	return false
}
