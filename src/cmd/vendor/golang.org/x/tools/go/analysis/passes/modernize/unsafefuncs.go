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
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/goplsexport"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/versions"
)

// TODO(adonovan): also support:
//
// func String(ptr *byte, len IntegerType) string
// func StringData(str string) *byte
// func Slice(ptr *ArbitraryType, len IntegerType) []ArbitraryType
// func SliceData(slice []ArbitraryType) *ArbitraryType

var unsafeFuncsAnalyzer = &analysis.Analyzer{
	Name:     "unsafefuncs",
	Doc:      analyzerutil.MustExtractDoc(doc, "unsafefuncs"),
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      unsafefuncs,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#unsafefuncs",
}

func init() {
	// Export to gopls until this is a published modernizer.
	goplsexport.UnsafeFuncsModernizer = unsafeFuncsAnalyzer
}

func unsafefuncs(pass *analysis.Pass) (any, error) {
	// Short circuit if the package doesn't use unsafe.
	// (In theory one could use some imported alias of unsafe.Pointer,
	// but let's ignore that.)
	if !typesinternal.Imports(pass.Pkg, "unsafe") {
		return nil, nil
	}

	var (
		inspect        = pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		info           = pass.TypesInfo
		tUnsafePointer = types.Typ[types.UnsafePointer]
	)

	isInteger := func(t types.Type) bool {
		basic, ok := t.Underlying().(*types.Basic)
		return ok && basic.Info()&types.IsInteger != 0
	}

	// isConversion reports whether e is a conversion T(x).
	// If so, it returns T and x.
	isConversion := func(curExpr inspector.Cursor) (t types.Type, x inspector.Cursor) {
		e := curExpr.Node().(ast.Expr)
		if conv, ok := ast.Unparen(e).(*ast.CallExpr); ok && len(conv.Args) == 1 {
			if tv := pass.TypesInfo.Types[conv.Fun]; tv.IsType() {
				return tv.Type, curExpr.ChildAt(edge.CallExpr_Args, 0)
			}
		}
		return
	}

	// The general form is where ptr and the result are of type unsafe.Pointer:
	//
	// 	unsafe.Pointer(uintptr(ptr) + uintptr(n))
	// =>
	// 	unsafe.Add(ptr, n)

	// Search for 'unsafe.Pointer(uintptr + uintptr)'
	// where the left operand was converted from a pointer.
	//
	// (Start from sum, not conversion, as it is not
	// uncommon to use a local type alias for unsafe.Pointer.)
	for curSum := range inspect.Root().Preorder((*ast.BinaryExpr)(nil)) {
		if sum, ok := curSum.Node().(*ast.BinaryExpr); ok &&
			sum.Op == token.ADD &&
			types.Identical(info.TypeOf(sum.X), types.Typ[types.Uintptr]) &&
			astutil.IsChildOf(curSum, edge.CallExpr_Args) {
			// Have: sum ≡ T(x:...uintptr... + y:...uintptr...)
			curX := curSum.ChildAt(edge.BinaryExpr_X, -1)
			curY := curSum.ChildAt(edge.BinaryExpr_Y, -1)

			// Is sum converted to unsafe.Pointer?
			curResult := curSum.Parent()
			if t, _ := isConversion(curResult); !(t != nil && types.Identical(t, tUnsafePointer)) {
				continue
			}
			// Have: result ≡ unsafe.Pointer(x:...uintptr... + y:...uintptr...)

			// Is sum.x converted from unsafe.Pointer?
			_, curPtr := isConversion(curX)
			if !astutil.CursorValid(curPtr) {
				continue
			}
			ptr := curPtr.Node().(ast.Expr)
			if !types.Identical(info.TypeOf(ptr), tUnsafePointer) {
				continue
			}
			// Have: result ≡ unsafe.Pointer(x:uintptr(...unsafe.Pointer...) + y:...uintptr...)

			file := astutil.EnclosingFile(curSum)
			if !analyzerutil.FileUsesGoVersion(pass, file, versions.Go1_17) {
				continue // unsafe.Add not available in this file
			}

			// import "unsafe"
			unsafedot, edits := refactor.AddImport(info, file, "unsafe", "unsafe", "Add", sum.Pos())

			// unsafe.Pointer(x + y)
			// ---------------     -
			//                x + y
			edits = append(edits, deleteConv(curResult)...)

			// uintptr   (ptr) + offset
			// -----------   ----      -
			// unsafe.Add(ptr,   offset)
			edits = append(edits, []analysis.TextEdit{
				{
					Pos:     sum.Pos(),
					End:     ptr.Pos(),
					NewText: fmt.Appendf(nil, "%sAdd(", unsafedot),
				},
				{
					Pos:     ptr.End(),
					End:     sum.Y.Pos(),
					NewText: []byte(", "),
				},
				{
					Pos:     sum.Y.End(),
					End:     sum.Y.End(),
					NewText: []byte(")"),
				},
			}...)

			// Variant: sum.y operand was converted from another integer type.
			// Discard the conversion, as Add is generic over integers.
			//
			// e.g. unsafe.Pointer(uintptr(ptr) + uintptr(len(s)))
			//                                    --------      -
			//      unsafe.Add    (        ptr,           len(s))
			if t, _ := isConversion(curY); t != nil && isInteger(t) {
				edits = append(edits, deleteConv(curY)...)
			}

			pass.Report(analysis.Diagnostic{
				Pos:     sum.Pos(),
				End:     sum.End(),
				Message: "pointer + integer can be simplified using unsafe.Add",
				SuggestedFixes: []analysis.SuggestedFix{{
					Message:   "Simplify pointer addition using unsafe.Add",
					TextEdits: edits,
				}},
			})
		}
	}

	return nil, nil
}

// deleteConv returns edits for changing T(x) to x, respecting precedence.
func deleteConv(cur inspector.Cursor) []analysis.TextEdit {
	conv := cur.Node().(*ast.CallExpr)

	usesPrec := func(n ast.Node) bool {
		switch n.(type) {
		case *ast.BinaryExpr, *ast.UnaryExpr:
			return true
		}
		return false
	}

	// Be careful not to change precedence of e.g. T(1+2) * 3.
	// TODO(adonovan): refine this.
	if usesPrec(cur.Parent().Node()) && usesPrec(conv.Args[0]) {
		// T(x+y) * z
		// -
		//  (x+y) * z
		return []analysis.TextEdit{{
			Pos: conv.Fun.Pos(),
			End: conv.Fun.End(),
		}}
	}

	// T(x)
	// -- -
	//   x
	return []analysis.TextEdit{
		{
			Pos: conv.Pos(),
			End: conv.Args[0].Pos(),
		},
		{
			Pos: conv.Args[0].End(),
			End: conv.End(),
		},
	}
}
