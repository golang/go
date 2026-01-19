// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/types"
	"slices"
	"strconv"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/versions"
)

// Warning: this analyzer is not safe to enable by default.
var AppendClippedAnalyzer = &analysis.Analyzer{
	Name:     "appendclipped",
	Doc:      analyzerutil.MustExtractDoc(doc, "appendclipped"),
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      appendclipped,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#appendclipped",
}

// The appendclipped pass offers to simplify a tower of append calls:
//
//	append(append(append(base, a...), b..., c...)
//
// with a call to go1.21's slices.Concat(base, a, b, c), or simpler
// replacements such as slices.Clone(a) in degenerate cases.
//
// We offer bytes.Clone in preference to slices.Clone where
// appropriate, if the package already imports "bytes";
// their behaviors are identical.
//
// The base expression must denote a clipped slice (see [isClipped]
// for definition), otherwise the replacement might eliminate intended
// side effects to the base slice's array.
//
// Examples:
//
//	append(append(append(x[:0:0], a...), b...), c...) -> slices.Concat(a, b, c)
//	append(append(slices.Clip(a), b...)               -> slices.Concat(a, b)
//	append([]T{}, a...)                               -> slices.Clone(a)
//	append([]string(nil), os.Environ()...)            -> os.Environ()
//
// The fix does not always preserve nilness the of base slice when the
// addends (a, b, c) are all empty (see #73557).
func appendclipped(pass *analysis.Pass) (any, error) {
	// Skip the analyzer in packages where its
	// fixes would create an import cycle.
	if within(pass, "slices", "bytes", "runtime") {
		return nil, nil
	}

	info := pass.TypesInfo

	// sliceArgs is a non-empty (reversed) list of slices to be concatenated.
	simplifyAppendEllipsis := func(file *ast.File, call *ast.CallExpr, base ast.Expr, sliceArgs []ast.Expr) {
		// Only appends whose base is a clipped slice can be simplified:
		// We must conservatively assume an append to an unclipped slice
		// such as append(y[:0], x...) is intended to have effects on y.
		clipped, empty := clippedSlice(info, base)
		if clipped == nil {
			return
		}

		// If any slice arg has a different type from the base
		// (and thus the result) don't offer a fix, to avoid
		// changing the return type, e.g:
		//
		//     type S []int
		//   - x := append([]int(nil), S{}...) // x : []int
		//   + x := slices.Clone(S{})          // x : S
		//
		// We could do better by inserting an explicit generic
		// instantiation:
		//
		//   x := slices.Clone[[]int](S{})
		//
		// but this is often unnecessary and unwanted, such as
		// when the value is used an in assignment context that
		// provides an explicit type:
		//
		//   var x []int = slices.Clone(S{})
		baseType := info.TypeOf(base)
		for _, arg := range sliceArgs {
			if !types.Identical(info.TypeOf(arg), baseType) {
				return
			}
		}

		// If the (clipped) base is empty, it may be safely ignored.
		// Otherwise treat it (or its unclipped subexpression, if possible)
		// as just another arg (the first) to Concat.
		//
		// TODO(adonovan): not so fast! If all the operands
		// are empty, then the nilness of base matters, because
		// append preserves nilness whereas Concat does not (#73557).
		if !empty {
			sliceArgs = append(sliceArgs, clipped)
		}
		slices.Reverse(sliceArgs)

		// TODO(adonovan): simplify sliceArgs[0] further: slices.Clone(s) -> s

		// Concat of a single (non-trivial) slice degenerates to Clone.
		if len(sliceArgs) == 1 {
			s := sliceArgs[0]

			// Special case for common but redundant clone of os.Environ().
			// append(zerocap, os.Environ()...) -> os.Environ()
			if scall, ok := s.(*ast.CallExpr); ok {
				obj := typeutil.Callee(info, scall)
				if typesinternal.IsFunctionNamed(obj, "os", "Environ") {
					pass.Report(analysis.Diagnostic{
						Pos:     call.Pos(),
						End:     call.End(),
						Message: "Redundant clone of os.Environ()",
						SuggestedFixes: []analysis.SuggestedFix{{
							Message: "Eliminate redundant clone",
							TextEdits: []analysis.TextEdit{{
								Pos:     call.Pos(),
								End:     call.End(),
								NewText: []byte(astutil.Format(pass.Fset, s)),
							}},
						}},
					})
					return
				}
			}

			// If the slice type is []byte, and the file imports
			// "bytes" but not "slices", prefer the (behaviorally
			// identical) bytes.Clone for local consistency.
			// https://go.dev/issue/70815#issuecomment-2671572984
			fileImports := func(path string) bool {
				return slices.ContainsFunc(file.Imports, func(spec *ast.ImportSpec) bool {
					value, _ := strconv.Unquote(spec.Path.Value)
					return value == path
				})
			}
			clonepkg := cond(
				types.Identical(info.TypeOf(call), byteSliceType) &&
					!fileImports("slices") && fileImports("bytes"),
				"bytes",
				"slices")

			// append(zerocap, s...) -> slices.Clone(s) or bytes.Clone(s)
			//
			// This is unsound if s is empty and its nilness
			// differs from zerocap (#73557).
			prefix, importEdits := refactor.AddImport(info, file, clonepkg, clonepkg, "Clone", call.Pos())
			message := fmt.Sprintf("Replace append with %s.Clone", clonepkg)
			pass.Report(analysis.Diagnostic{
				Pos:     call.Pos(),
				End:     call.End(),
				Message: message,
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: message,
					TextEdits: append(importEdits, []analysis.TextEdit{{
						Pos:     call.Pos(),
						End:     call.End(),
						NewText: fmt.Appendf(nil, "%sClone(%s)", prefix, astutil.Format(pass.Fset, s)),
					}}...),
				}},
			})
			return
		}

		// append(append(append(base, a...), b..., c...) -> slices.Concat(base, a, b, c)
		//
		// This is unsound if all slices are empty and base is non-nil (#73557).
		prefix, importEdits := refactor.AddImport(info, file, "slices", "slices", "Concat", call.Pos())
		pass.Report(analysis.Diagnostic{
			Pos:     call.Pos(),
			End:     call.End(),
			Message: "Replace append with slices.Concat",
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: "Replace append with slices.Concat",
				TextEdits: append(importEdits, []analysis.TextEdit{{
					Pos:     call.Pos(),
					End:     call.End(),
					NewText: fmt.Appendf(nil, "%sConcat(%s)", prefix, formatExprs(pass.Fset, sliceArgs)),
				}}...),
			}},
		})
	}

	// Mark nested calls to append so that we don't emit diagnostics for them.
	skip := make(map[*ast.CallExpr]bool)

	// Visit calls of form append(x, y...).
	for curFile := range filesUsingGoVersion(pass, versions.Go1_21) {
		file := curFile.Node().(*ast.File)

		for curCall := range curFile.Preorder((*ast.CallExpr)(nil)) {
			call := curCall.Node().(*ast.CallExpr)
			if skip[call] {
				continue
			}

			// Recursively unwrap ellipsis calls to append, so
			//   append(append(append(base, a...), b..., c...)
			// yields (base, [c b a]).
			base, slices := ast.Expr(call), []ast.Expr(nil) // base case: (call, nil)
		again:
			if call, ok := base.(*ast.CallExpr); ok {
				if id, ok := call.Fun.(*ast.Ident); ok &&
					call.Ellipsis.IsValid() &&
					len(call.Args) == 2 &&
					info.Uses[id] == builtinAppend {

					// Have: append(base, s...)
					base, slices = call.Args[0], append(slices, call.Args[1])
					skip[call] = true
					goto again
				}
			}

			if len(slices) > 0 {
				simplifyAppendEllipsis(file, call, base, slices)
			}
		}
	}
	return nil, nil
}

// clippedSlice returns res != nil if e denotes a slice that is
// definitely clipped, that is, its len(s)==cap(s).
//
// The value of res is either the same as e or is a subexpression of e
// that denotes the same slice but without the clipping operation.
//
// In addition, it reports whether the slice is definitely empty.
//
// Examples of clipped slices:
//
//	x[:0:0]				(empty)
//	[]T(nil)			(empty)
//	Slice{}				(empty)
//	x[:len(x):len(x)]		(nonempty)  res=x
//	x[:k:k]	 	         	(nonempty)
//	slices.Clip(x)			(nonempty)  res=x
//
// TODO(adonovan): Add a check that the expression x has no side effects in
// case x[:len(x):len(x)] -> x. Now the program behavior may change.
func clippedSlice(info *types.Info, e ast.Expr) (res ast.Expr, empty bool) {
	switch e := e.(type) {
	case *ast.SliceExpr:
		// x[:0:0], x[:len(x):len(x)], x[:k:k]
		if e.Slice3 && e.High != nil && e.Max != nil && astutil.EqualSyntax(e.High, e.Max) { // x[:k:k]
			res = e
			empty = isZeroIntConst(info, e.High) // x[:0:0]
			if call, ok := e.High.(*ast.CallExpr); ok &&
				typeutil.Callee(info, call) == builtinLen &&
				astutil.EqualSyntax(call.Args[0], e.X) {
				res = e.X // x[:len(x):len(x)] -> x
			}
			return
		}
		return

	case *ast.CallExpr:
		// []T(nil)?
		if info.Types[e.Fun].IsType() &&
			is[*ast.Ident](e.Args[0]) &&
			info.Uses[e.Args[0].(*ast.Ident)] == builtinNil {
			return e, true
		}

		// slices.Clip(x)?
		obj := typeutil.Callee(info, e)
		if typesinternal.IsFunctionNamed(obj, "slices", "Clip") {
			return e.Args[0], false // slices.Clip(x) -> x
		}

	case *ast.CompositeLit:
		// Slice{}?
		if len(e.Elts) == 0 {
			return e, true
		}
	}
	return nil, false
}
