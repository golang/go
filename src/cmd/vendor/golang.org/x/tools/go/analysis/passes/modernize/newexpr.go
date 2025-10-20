// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	_ "embed"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"fmt"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/astutil"
)

var NewExprAnalyzer = &analysis.Analyzer{
	Name:      "newexpr",
	Doc:       analysisinternal.MustExtractDoc(doc, "newexpr"),
	URL:       "https://pkg.go.dev/golang.org/x/tools/gopls/internal/analysis/modernize#newexpr",
	Requires:  []*analysis.Analyzer{inspect.Analyzer},
	Run:       run,
	FactTypes: []analysis.Fact{&newLike{}},
}

func run(pass *analysis.Pass) (any, error) {
	var (
		inspect = pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		info    = pass.TypesInfo
	)

	// Detect functions that are new-like, i.e. have the form:
	//
	//	func f(x T) *T { return &x }
	//
	// meaning that it is equivalent to new(x), if x has type T.
	for curFuncDecl := range inspect.Root().Preorder((*ast.FuncDecl)(nil)) {
		decl := curFuncDecl.Node().(*ast.FuncDecl)
		fn := info.Defs[decl.Name].(*types.Func)
		if decl.Body != nil && len(decl.Body.List) == 1 {
			if ret, ok := decl.Body.List[0].(*ast.ReturnStmt); ok && len(ret.Results) == 1 {
				if unary, ok := ret.Results[0].(*ast.UnaryExpr); ok && unary.Op == token.AND {
					if id, ok := unary.X.(*ast.Ident); ok {
						if v, ok := info.Uses[id].(*types.Var); ok {
							sig := fn.Signature()
							if sig.Results().Len() == 1 &&
								is[*types.Pointer](sig.Results().At(0).Type()) && // => no iface conversion
								sig.Params().Len() == 1 &&
								sig.Params().At(0) == v {

								// Export a fact for each one.
								pass.ExportObjectFact(fn, &newLike{})

								// Check file version.
								file := astutil.EnclosingFile(curFuncDecl)
								if !fileUses(info, file, "go1.26") {
									continue // new(expr) not available in this file
								}

								var edits []analysis.TextEdit

								// If 'new' is not shadowed, replace func body: &x -> new(x).
								// This makes it safely and cleanly inlinable.
								curRet, _ := curFuncDecl.FindNode(ret)
								if lookup(info, curRet, "new") == builtinNew {
									edits = []analysis.TextEdit{
										// return    &x
										//        ---- -
										// return new(x)
										{
											Pos:     unary.OpPos,
											End:     unary.OpPos + token.Pos(len("&")),
											NewText: []byte("new("),
										},
										{
											Pos:     unary.X.End(),
											End:     unary.X.End(),
											NewText: []byte(")"),
										},
									}
								}

								// Disabled until we resolve https://go.dev/issue/75726
								// (Go version skew between caller and callee in inliner.)
								// TODO(adonovan): fix and reenable.
								//
								// Also, restore these lines to our section of doc.go:
								// 	//go:fix inline
								//	...
								// 	(The directive comment causes the inline analyzer to suggest
								// 	that calls to such functions are inlined.)
								if false {
									// Add a //go:fix inline annotation, if not already present.
									// TODO(adonovan): use ast.ParseDirective when go1.26 is assured.
									if !strings.Contains(decl.Doc.Text(), "go:fix inline") {
										edits = append(edits, analysis.TextEdit{
											Pos:     decl.Pos(),
											End:     decl.Pos(),
											NewText: []byte("//go:fix inline\n"),
										})
									}
								}

								if len(edits) > 0 {
									pass.Report(analysis.Diagnostic{
										Pos:     decl.Name.Pos(),
										End:     decl.Name.End(),
										Message: fmt.Sprintf("%s can be an inlinable wrapper around new(expr)", decl.Name),
										SuggestedFixes: []analysis.SuggestedFix{
											{
												Message:   "Make %s an inlinable wrapper around new(expr)",
												TextEdits: edits,
											},
										},
									})
								}
							}
						}
					}
				}
			}
		}
	}

	// Report and transform calls, when safe.
	// In effect, this is inlining the new-like function
	// even before we have marked the callee with //go:fix inline.
	for curCall := range inspect.Root().Preorder((*ast.CallExpr)(nil)) {
		call := curCall.Node().(*ast.CallExpr)
		var fact newLike
		if fn, ok := typeutil.Callee(info, call).(*types.Func); ok &&
			pass.ImportObjectFact(fn, &fact) {

			// Check file version.
			file := astutil.EnclosingFile(curCall)
			if !fileUses(info, file, "go1.26") {
				continue // new(expr) not available in this file
			}

			// Check new is not shadowed.
			if lookup(info, curCall, "new") != builtinNew {
				continue
			}

			// The return type *T must exactly match the argument type T.
			// (We formulate it this way--not in terms of the parameter
			// type--to support generics.)
			var targ types.Type
			{
				arg := call.Args[0]
				tvarg := info.Types[arg]

				// Constants: we must work around the type checker
				// bug that causes info.Types to wrongly report the
				// "typed" type for an untyped constant.
				// (See "historical reasons" in issue go.dev/issue/70638.)
				//
				// We don't have a reliable way to do this but we can attempt
				// to re-typecheck the constant expression on its own, in
				// the original lexical environment but not as a part of some
				// larger expression that implies a conversion to some "typed" type.
				// (For the genesis of this idea see (*state).arguments
				// in ../../../../internal/refactor/inline/inline.go.)
				if tvarg.Value != nil {
					info2 := &types.Info{Types: make(map[ast.Expr]types.TypeAndValue)}
					if err := types.CheckExpr(token.NewFileSet(), pass.Pkg, token.NoPos, arg, info2); err != nil {
						continue // unexpected error
					}
					tvarg = info2.Types[arg]
				}

				targ = types.Default(tvarg.Type)
			}
			if !types.Identical(types.NewPointer(targ), info.TypeOf(call)) {
				continue
			}

			pass.Report(analysis.Diagnostic{
				Pos:     call.Pos(),
				End:     call.End(),
				Message: fmt.Sprintf("call of %s(x) can be simplified to new(x)", fn.Name()),
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: fmt.Sprintf("Simplify %s(x) to new(x)", fn.Name()),
					TextEdits: []analysis.TextEdit{{
						Pos:     call.Fun.Pos(),
						End:     call.Fun.End(),
						NewText: []byte("new"),
					}},
				}},
			})
		}
	}

	return nil, nil
}

// A newLike fact records that its associated function is "new-like".
type newLike struct{}

func (*newLike) AFact()         {}
func (*newLike) String() string { return "newlike" }
