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
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/typeparams"
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
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#hdr-Analyzer_rangeint",
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
								if typesinternal.IsAssignedOrAddressTaken(info, cur) {
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

					validIncrement := false
					if inc, ok := loop.Post.(*ast.IncDecStmt); ok &&
						inc.Tok == token.INC &&
						astutil.EqualSyntax(compare.X, inc.X) {
						// Have: i++
						validIncrement = true
					} else if assign, ok := loop.Post.(*ast.AssignStmt); ok &&
						assign.Tok == token.ADD_ASSIGN &&
						len(assign.Rhs) == 1 && isIntLiteral(info, assign.Rhs[0], 1) &&
						len(assign.Lhs) == 1 && astutil.EqualSyntax(compare.X, assign.Lhs[0]) {
						// Have: i += 1
						validIncrement = true
					}

					if validIncrement {
						// Have: for i = 0; i < limit; i++ {}

						// Find references to i within the loop body.
						v := info.ObjectOf(index).(*types.Var)
						switch v.Kind() {
						case types.PackageVar:
							continue nextLoop
						case types.ResultVar:
							// If v is a named result, it is implicitly
							// used after the loop (go.dev/issue/76880).
							continue nextLoop
						}

						used := false
						body := curLoop.Child(loop.Body)
						for curId := range typeindex.Uses(v) {
							if body.Contains(curId) {
								used = true

								// Reject if any use in the body is an l-value
								// (assigned or address-taken): a "for range int" loop
								// does not respect assignments to the loop variable.
								if typesinternal.IsAssignedOrAddressTaken(info, curId) {
									continue nextLoop
								}
							}

							// If i is a pre-existing variable (for i = 0; ...),
							// don't offer a fix if its value after the loop could
							// be observed directly or via aliasing, as a range loop
							// leaves i with a different final value (limit-1) and
							// shares a single variable across iterations (go.dev/issue/77161).
							if init.Tok == token.ASSIGN {
								// Is i used after the loop?
								if curId.Node().Pos() > loop.End() {
									continue nextLoop
								}
								// Is i assigned or address-taken outside the loop?
								//     var i int
								//     p := &i
								//     for i = 0; i < 10; i++ {}
								//     return *p
								if !curLoop.Contains(curId) && typesinternal.IsAssignedOrAddressTaken(info, curId) {
									continue nextLoop
								}
								// Is i captured by a closure within the scope of i?
								//     var i int
								//     f := func() int { return i } // or defer func() { print(i) }()
								//     for i = 0; i < 10; i++ {}
								//     return f()
								for curFunc := range curId.Enclosing((*ast.FuncLit)(nil)) {
									if curFunc.Node().Pos() > v.Pos() {
										continue nextLoop
									}
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

						// The loop index (v) must not be a type parameter constrained by
						// multiple distinct integer types, or a type parameter constrained
						// by non-integer types. Transforming such instances to a range loop
						// would result in a compiler error.
						// See golang/go#78571.
						if ts := typeparams.TypeSetOf(v.Type()); ts.Len() != 0 {
							// From the spec (https://go.dev/ref/spec#For_range):
							// "If the type of the range expression is a type parameter, all
							// types in its type set must have the same underlying type and the
							// range expression must be valid for that type."
							//
							// Check that all terms have the same underlying type, and that it
							// is an integer. (Range over float is not allowed.)
							u := ts.Term(0).Type().Underlying()
							if !isInteger(u) || !moreiters.Every(ts.UnderlyingTypes(), func(t types.Type) bool {
								return types.Identical(u, t)
							}) {
								continue nextLoop
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
						//    for i := 0; i < limit; i++ {}
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
								tLimit := info2.TypeOf(limit)
								// Eliminate conversion when safe.
								//
								// Redundant conversions are not only unsightly but may in some cases cause
								// architecture-specific types (e.g. syscall.Timespec.Nsec) to be inserted
								// into otherwise portable files.
								//
								// The operand must have an integer type (not, say, '1e6')
								// even when assigning to an existing integer variable.
								if isInteger(tLimit) {
									// When declaring a new var from an untyped limit,
									// the limit's default type is what matters.
									if init.Tok != token.ASSIGN {
										tLimit = types.Default(tLimit)
									}
									if types.AssignableTo(tLimit, tVar) {
										beforeLimit, afterLimit = "", ""
									}
								}
							}
						}

						pass.Report(analysis.Diagnostic{
							Pos:     init.Pos(),
							End:     loop.Post.End(),
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
										End: loop.Post.End(),
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
