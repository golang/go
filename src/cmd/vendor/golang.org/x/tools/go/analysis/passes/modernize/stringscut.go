// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"iter"
	"strconv"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/goplsexport"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var stringscutAnalyzer = &analysis.Analyzer{
	Name: "stringscut",
	Doc:  analyzerutil.MustExtractDoc(doc, "stringscut"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: stringscut,
	URL: "https://pkg.go.dev/golang.org/x/tools/gopls/internal/analysis/modernize#stringscut",
}

func init() {
	// Export to gopls until this is a published modernizer.
	goplsexport.StringsCutModernizer = stringscutAnalyzer
}

// stringscut offers a fix to replace an occurrence of strings.Index{,Byte} with
// strings.{Cut,Contains}, and similar fixes for functions in the bytes package.
// Consider some candidate for replacement i := strings.Index(s, substr).
// The following must hold for a replacement to occur:
//
//  1. All instances of i and s must be in one of these forms.
//     Binary expressions:
//     (a): establishing that i < 0: e.g.: i < 0, 0 > i, i == -1, -1 == i
//     (b): establishing that i > -1: e.g.: i >= 0, 0 <= i, i == 0, 0 == i
//
//     Slice expressions:
//     a: s[:i], s[0:i]
//     b: s[i+len(substr):], s[len(substr) + i:], s[i + const], s[k + i] (where k = len(substr))
//
//  2. There can be no uses of s, substr, or i where they are
//     potentially modified (i.e. in assignments, or function calls with unknown side
//     effects).
//
// Then, the replacement involves the following substitutions:
//
//  1. Replace "i := strings.Index(s, substr)" with "before, after, ok := strings.Cut(s, substr)"
//
//  2. Replace instances of binary expressions (a) with !ok and binary expressions (b) with ok.
//
//  3. Replace slice expressions (a) with "before" and slice expressions (b) with after.
//
//  4. The assignments to before, after, and ok may use the blank identifier "_" if they are unused.
//
//     For example:
//
//     i := strings.Index(s, substr)
//     if i >= 0 {
//     use(s[:i], s[i+len(substr):])
//     }
//
//     Would become:
//
//     before, after, ok := strings.Cut(s, substr)
//     if ok {
//     use(before, after)
//     }
//
// If the condition involving `i` establishes that i > -1, then we replace it with
// `if ok“. Variants listed above include i >= 0, i > 0, and i == 0.
// If the condition is negated (e.g. establishes `i < 0`), we use `if !ok` instead.
// If the slices of `s` match `s[:i]` or `s[i+len(substr):]` or their variants listed above,
// then we replace them with before and after.
//
// When the index `i` is used only to check for the presence of the substring or byte slice,
// the suggested fix uses Contains() instead of Cut.
//
// For example:
//
//	i := strings.Index(s, substr)
//	if i >= 0 {
//		return
//	}
//
// Would become:
//
//	found := strings.Contains(s, substr)
//	if found {
//		return
//	}
func stringscut(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo

		stringsIndex     = index.Object("strings", "Index")
		stringsIndexByte = index.Object("strings", "IndexByte")
		bytesIndex       = index.Object("bytes", "Index")
		bytesIndexByte   = index.Object("bytes", "IndexByte")
	)

	for _, obj := range []types.Object{
		stringsIndex,
		stringsIndexByte,
		bytesIndex,
		bytesIndexByte,
	} {
		// (obj may be nil)
	nextcall:
		for curCall := range index.Calls(obj) {
			// Check file version.
			if !analyzerutil.FileUsesGoVersion(pass, astutil.EnclosingFile(curCall), versions.Go1_18) {
				continue // strings.Index not available in this file
			}
			indexCall := curCall.Node().(*ast.CallExpr) // the call to strings.Index, etc.
			obj := typeutil.Callee(info, indexCall)
			if obj == nil {
				continue
			}

			var iIdent *ast.Ident // defining identifier of i var
			switch ek, idx := curCall.ParentEdge(); ek {
			case edge.ValueSpec_Values:
				// Have: var i = strings.Index(...)
				curName := curCall.Parent().ChildAt(edge.ValueSpec_Names, idx)
				iIdent = curName.Node().(*ast.Ident)
			case edge.AssignStmt_Rhs:
				// Have: i := strings.Index(...)
				// (Must be i's definition.)
				curLhs := curCall.Parent().ChildAt(edge.AssignStmt_Lhs, idx)
				iIdent, _ = curLhs.Node().(*ast.Ident) // may be nil
			}

			if iIdent == nil {
				continue
			}
			// Inv: iIdent is i's definition. The following would be skipped: 'var i int; i = strings.Index(...)'
			// Get uses of i.
			iObj := info.ObjectOf(iIdent)
			if iObj == nil {
				continue
			}

			var (
				s      = indexCall.Args[0]
				substr = indexCall.Args[1]
			)

			// Check that there are no statements that alter the value of s
			// or substr after the call to Index().
			if !indexArgValid(info, index, s, indexCall.Pos()) ||
				!indexArgValid(info, index, substr, indexCall.Pos()) {
				continue nextcall
			}

			// Next, examine all uses of i. If the only uses are of the
			// forms mentioned above (e.g. i < 0, i >= 0, s[:i] and s[i +
			// len(substr)]), then we can replace the call to Index()
			// with a call to Cut() and use the returned ok, before,
			// and after variables accordingly.
			lessZero, greaterNegOne, beforeSlice, afterSlice := checkIdxUses(pass.TypesInfo, index.Uses(iObj), s, substr)

			// Either there are no uses of before, after, or ok, or some use
			// of i does not match our criteria - don't suggest a fix.
			if lessZero == nil && greaterNegOne == nil && beforeSlice == nil && afterSlice == nil {
				continue
			}

			// If the only uses are ok and !ok, don't suggest a Cut() fix - these should be using Contains()
			isContains := (len(lessZero) > 0 || len(greaterNegOne) > 0) && len(beforeSlice) == 0 && len(afterSlice) == 0

			scope := iObj.Parent()
			var (
				// TODO(adonovan): avoid FreshName when not needed; see errorsastype.
				okVarName     = refactor.FreshName(scope, iIdent.Pos(), "ok")
				beforeVarName = refactor.FreshName(scope, iIdent.Pos(), "before")
				afterVarName  = refactor.FreshName(scope, iIdent.Pos(), "after")
				foundVarName  = refactor.FreshName(scope, iIdent.Pos(), "found") // for Contains()
			)

			// If there will be no uses of ok, before, or after, use the
			// blank identifier instead.
			if len(lessZero) == 0 && len(greaterNegOne) == 0 {
				okVarName = "_"
			}
			if len(beforeSlice) == 0 {
				beforeVarName = "_"
			}
			if len(afterSlice) == 0 {
				afterVarName = "_"
			}

			var edits []analysis.TextEdit
			replace := func(exprs []ast.Expr, new string) {
				for _, expr := range exprs {
					edits = append(edits, analysis.TextEdit{
						Pos:     expr.Pos(),
						End:     expr.End(),
						NewText: []byte(new),
					})
				}
			}
			// Get the ident for the call to strings.Index, which could just be
			// "Index" if the strings package is dot imported.
			indexCallId := typesinternal.UsedIdent(info, indexCall.Fun)
			replacedFunc := "Cut"
			if isContains {
				replacedFunc = "Contains"
				replace(lessZero, "!"+foundVarName)  // idx < 0   ->  !found
				replace(greaterNegOne, foundVarName) // idx > -1  ->   found

				// Replace the assignment with found, and replace the call to
				// Index or IndexByte with a call to Contains.
				// i     := strings.Index   (...)
				// -----            --------
				// found := strings.Contains(...)
				edits = append(edits, analysis.TextEdit{
					Pos:     iIdent.Pos(),
					End:     iIdent.End(),
					NewText: []byte(foundVarName),
				}, analysis.TextEdit{
					Pos:     indexCallId.Pos(),
					End:     indexCallId.End(),
					NewText: []byte("Contains"),
				})
			} else {
				replace(lessZero, "!"+okVarName)    // idx < 0   ->  !ok
				replace(greaterNegOne, okVarName)   // idx > -1  ->   ok
				replace(beforeSlice, beforeVarName) // s[:idx]   ->   before
				replace(afterSlice, afterVarName)   // s[idx+k:] ->   after

				// Replace the assignment with before, after, ok, and replace
				// the call to Index or IndexByte with a call to Cut.
				// i     			 := strings.Index(...)
				// -----------------            -----
				// before, after, ok := strings.Cut  (...)
				edits = append(edits, analysis.TextEdit{
					Pos:     iIdent.Pos(),
					End:     iIdent.End(),
					NewText: fmt.Appendf(nil, "%s, %s, %s", beforeVarName, afterVarName, okVarName),
				}, analysis.TextEdit{
					Pos:     indexCallId.Pos(),
					End:     indexCallId.End(),
					NewText: []byte("Cut"),
				})
			}

			// Calls to IndexByte have a byte as their second arg, which
			// must be converted to a string or []byte to be a valid arg for Cut/Contains.
			if obj.Name() == "IndexByte" {
				switch obj.Pkg().Name() {
				case "strings":
					searchByteVal := info.Types[substr].Value
					if searchByteVal == nil {
						// substr is a variable, e.g. substr := byte('b')
						// use string(substr)
						edits = append(edits, []analysis.TextEdit{
							{
								Pos:     substr.Pos(),
								NewText: []byte("string("),
							},
							{
								Pos:     substr.End(),
								NewText: []byte(")"),
							},
						}...)
					} else {
						// substr is a byte constant
						val, _ := constant.Int64Val(searchByteVal) // inv: must be a valid byte
						// strings.Cut/Contains requires a string, so convert byte literal to string literal; e.g. 'a' -> "a", 55 -> "7"
						edits = append(edits, analysis.TextEdit{
							Pos:     substr.Pos(),
							End:     substr.End(),
							NewText: strconv.AppendQuote(nil, string(byte(val))),
						})
					}
				case "bytes":
					// bytes.Cut/Contains requires a []byte, so wrap substr in a []byte{}
					edits = append(edits, []analysis.TextEdit{
						{
							Pos:     substr.Pos(),
							NewText: []byte("[]byte{"),
						},
						{
							Pos:     substr.End(),
							NewText: []byte("}"),
						},
					}...)
				}
			}
			pass.Report(analysis.Diagnostic{
				Pos: indexCall.Fun.Pos(),
				End: indexCall.Fun.End(),
				Message: fmt.Sprintf("%s.%s can be simplified using %s.%s",
					obj.Pkg().Name(), obj.Name(), obj.Pkg().Name(), replacedFunc),
				Category: "stringscut",
				SuggestedFixes: []analysis.SuggestedFix{{
					Message:   fmt.Sprintf("Simplify %s.%s call using %s.%s", obj.Pkg().Name(), obj.Name(), obj.Pkg().Name(), replacedFunc),
					TextEdits: edits,
				}},
			})
		}
	}

	return nil, nil
}

// indexArgValid reports whether expr is a valid strings.Index(_, _) arg
// for the transformation. An arg is valid iff it is:
// - constant;
// - a local variable with no modifying uses after the Index() call; or
// - []byte(x) where x is also valid by this definition.
// All other expressions are assumed not referentially transparent,
// so we cannot be sure that all uses are safe to replace.
func indexArgValid(info *types.Info, index *typeindex.Index, expr ast.Expr, afterPos token.Pos) bool {
	tv := info.Types[expr]
	if tv.Value != nil {
		return true // constant
	}
	switch expr := expr.(type) {
	case *ast.CallExpr:
		return types.Identical(tv.Type, byteSliceType) &&
			indexArgValid(info, index, expr.Args[0], afterPos) // check s in []byte(s)
	case *ast.Ident:
		sObj := info.Uses[expr]
		sUses := index.Uses(sObj)
		return !hasModifyingUses(info, sUses, afterPos)
	default:
		// For now, skip instances where s or substr are not
		// identifers, basic lits, or call expressions of the form
		// []byte(s).
		// TODO(mkalil): Handle s and substr being expressions like ptr.field[i].
		// From adonovan: We'd need to analyze s and substr to see
		// whether they are referentially transparent, and if not,
		// analyze all code between declaration and use and see if
		// there are statements or expressions with potential side
		// effects.
		return false
	}
}

// checkIdxUses inspects the uses of i to make sure they match certain criteria that
// allows us to suggest a modernization. If all uses of i, s and substr match
// one of the following four valid formats, it returns a list of occurrences for
// each format. If any of the uses do not match one of the formats, return nil
// for all values, since we should not offer a replacement.
// 1. lessZero - a condition involving i establishing that i is negative (e.g. i < 0, 0 > i, i == -1, -1 == i)
// 2. greaterNegOne - a condition involving i establishing that i is non-negative (e.g. i >= 0, 0 <= i, i == 0, 0 == i)
// 3. beforeSlice - a slice of `s` that matches either s[:i], s[0:i]
// 4. afterSlice - a slice of `s` that matches one of: s[i+len(substr):], s[len(substr) + i:], s[i + const], s[k + i] (where k = len(substr))
func checkIdxUses(info *types.Info, uses iter.Seq[inspector.Cursor], s, substr ast.Expr) (lessZero, greaterNegOne, beforeSlice, afterSlice []ast.Expr) {
	use := func(cur inspector.Cursor) bool {
		ek, _ := cur.ParentEdge()
		n := cur.Parent().Node()
		switch ek {
		case edge.BinaryExpr_X, edge.BinaryExpr_Y:
			check := n.(*ast.BinaryExpr)
			switch checkIdxComparison(info, check) {
			case -1:
				lessZero = append(lessZero, check)
				return true
			case 1:
				greaterNegOne = append(greaterNegOne, check)
				return true
			}
			// Check does not establish that i < 0 or i > -1.
			// Might be part of an outer slice expression like s[i + k]
			// which requires a different check.
			// Check that the thing being sliced is s and that the slice
			// doesn't have a max index.
			if slice, ok := cur.Parent().Parent().Node().(*ast.SliceExpr); ok &&
				sameObject(info, s, slice.X) &&
				slice.Max == nil {
				if isBeforeSlice(info, ek, slice) {
					beforeSlice = append(beforeSlice, slice)
					return true
				} else if isAfterSlice(info, ek, slice, substr) {
					afterSlice = append(afterSlice, slice)
					return true
				}
			}
		case edge.SliceExpr_Low, edge.SliceExpr_High:
			slice := n.(*ast.SliceExpr)
			// Check that the thing being sliced is s and that the slice doesn't
			// have a max index.
			if sameObject(info, s, slice.X) && slice.Max == nil {
				if isBeforeSlice(info, ek, slice) {
					beforeSlice = append(beforeSlice, slice)
					return true
				} else if isAfterSlice(info, ek, slice, substr) {
					afterSlice = append(afterSlice, slice)
					return true
				}
			}
		}
		return false
	}

	for curIdent := range uses {
		if !use(curIdent) {
			return nil, nil, nil, nil
		}
	}
	return lessZero, greaterNegOne, beforeSlice, afterSlice
}

// hasModifyingUses reports whether any of the uses involve potential
// modifications. Uses involving assignments before the "afterPos" won't be
// considered.
func hasModifyingUses(info *types.Info, uses iter.Seq[inspector.Cursor], afterPos token.Pos) bool {
	for curUse := range uses {
		ek, _ := curUse.ParentEdge()
		if ek == edge.AssignStmt_Lhs {
			if curUse.Node().Pos() <= afterPos {
				continue
			}
			assign := curUse.Parent().Node().(*ast.AssignStmt)
			if sameObject(info, assign.Lhs[0], curUse.Node().(*ast.Ident)) {
				// Modifying use because we are reassigning the value of the object.
				return true
			}
		} else if ek == edge.UnaryExpr_X &&
			curUse.Parent().Node().(*ast.UnaryExpr).Op == token.AND {
			// Modifying use because we might be passing the object by reference (an explicit &).
			// We can ignore the case where we have a method call on the expression (which
			// has an implicit &) because we know the type of s and substr are strings
			// which cannot have methods on them.
			return true
		}
	}
	return false
}

// checkIdxComparison reports whether the check establishes that i is negative
// or non-negative. It returns -1 in the first case, 1 in the second, and 0 if
// we can confirm neither condition. We assume that a check passed to
// checkIdxComparison has i as one of its operands.
func checkIdxComparison(info *types.Info, check *ast.BinaryExpr) int {
	// Check establishes that i is negative.
	// e.g.: i < 0, 0 > i, i == -1, -1 == i
	if check.Op == token.LSS && (isNegativeConst(info, check.Y) || isZeroIntConst(info, check.Y)) || //i < (0 or neg)
		check.Op == token.GTR && (isNegativeConst(info, check.X) || isZeroIntConst(info, check.X)) || // (0 or neg) > i
		check.Op == token.LEQ && (isNegativeConst(info, check.Y)) || //i <= (neg)
		check.Op == token.GEQ && (isNegativeConst(info, check.X)) || // (neg) >= i
		check.Op == token.EQL &&
			(isNegativeConst(info, check.X) || isNegativeConst(info, check.Y)) { // i == neg; neg == i
		return -1
	}
	// Check establishes that i is non-negative.
	// e.g.: i >= 0, 0 <= i, i == 0, 0 == i
	if check.Op == token.GTR && (isNonNegativeConst(info, check.Y) || isIntLiteral(info, check.Y, -1)) || // i > (non-neg or -1)
		check.Op == token.LSS && (isNonNegativeConst(info, check.X) || isIntLiteral(info, check.X, -1)) || // (non-neg or -1) < i
		check.Op == token.GEQ && isNonNegativeConst(info, check.Y) || // i >= (non-neg)
		check.Op == token.LEQ && isNonNegativeConst(info, check.X) || // (non-neg) <= i
		check.Op == token.EQL &&
			(isNonNegativeConst(info, check.X) || isNonNegativeConst(info, check.Y)) { // i == non-neg; non-neg == i
		return 1
	}
	return 0
}

// isNegativeConst returns true if the expr is a const int with value < zero.
func isNegativeConst(info *types.Info, expr ast.Expr) bool {
	if tv, ok := info.Types[expr]; ok && tv.Value != nil && tv.Value.Kind() == constant.Int {
		if v, ok := constant.Int64Val(tv.Value); ok {
			return v < 0
		}
	}
	return false
}

// isNonNegativeConst returns true if the expr is a const int with value >= zero.
func isNonNegativeConst(info *types.Info, expr ast.Expr) bool {
	if tv, ok := info.Types[expr]; ok && tv.Value != nil && tv.Value.Kind() == constant.Int {
		if v, ok := constant.Int64Val(tv.Value); ok {
			return v >= 0
		}
	}
	return false
}

// isBeforeSlice reports whether the SliceExpr is of the form s[:i] or s[0:i].
func isBeforeSlice(info *types.Info, ek edge.Kind, slice *ast.SliceExpr) bool {
	return ek == edge.SliceExpr_High && (slice.Low == nil || isZeroIntConst(info, slice.Low))
}

// isAfterSlice reports whether the SliceExpr is of the form s[i+len(substr):],
// or s[i + k:] where k is a const is equal to len(substr).
func isAfterSlice(info *types.Info, ek edge.Kind, slice *ast.SliceExpr, substr ast.Expr) bool {
	lowExpr, ok := slice.Low.(*ast.BinaryExpr)
	if !ok || slice.High != nil {
		return false
	}
	// Returns true if the expression is a call to len(substr).
	isLenCall := func(expr ast.Expr) bool {
		call, ok := expr.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return false
		}
		return sameObject(info, substr, call.Args[0]) && typeutil.Callee(info, call) == builtinLen
	}

	// Handle len([]byte(substr))
	if is[*ast.CallExpr](substr) {
		call := substr.(*ast.CallExpr)
		tv := info.Types[call.Fun]
		if tv.IsType() && types.Identical(tv.Type, byteSliceType) {
			// Only one arg in []byte conversion.
			substr = call.Args[0]
		}
	}
	substrLen := -1
	substrVal := info.Types[substr].Value
	if substrVal != nil {
		switch substrVal.Kind() {
		case constant.String:
			substrLen = len(constant.StringVal(substrVal))
		case constant.Int:
			// constant.Value is a byte literal, e.g. bytes.IndexByte(_, 'a')
			// or a numeric byte literal, e.g. bytes.IndexByte(_, 65)
			substrLen = 1
		}
	}

	switch ek {
	case edge.BinaryExpr_X:
		kVal := info.Types[lowExpr.Y].Value
		if kVal == nil {
			// i + len(substr)
			return lowExpr.Op == token.ADD && isLenCall(lowExpr.Y)
		} else {
			// i + k
			kInt, ok := constant.Int64Val(kVal)
			return ok && substrLen == int(kInt)
		}
	case edge.BinaryExpr_Y:
		kVal := info.Types[lowExpr.X].Value
		if kVal == nil {
			// len(substr) + i
			return lowExpr.Op == token.ADD && isLenCall(lowExpr.X)
		} else {
			// k + i
			kInt, ok := constant.Int64Val(kVal)
			return ok && substrLen == int(kInt)
		}
	}
	return false
}

// sameObject reports whether we know that the expressions resolve to the same object.
func sameObject(info *types.Info, expr1, expr2 ast.Expr) bool {
	if ident1, ok := expr1.(*ast.Ident); ok {
		if ident2, ok := expr2.(*ast.Ident); ok {
			uses1, ok1 := info.Uses[ident1]
			uses2, ok2 := info.Uses[ident2]
			return ok1 && ok2 && uses1 == uses2
		}
	}
	return false
}
