// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stringintconv defines an Analyzer that flags type conversions
// from integers to strings.
package stringintconv

import (
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check for string(int) conversions

This checker flags conversions of the form string(x) where x is an integer
(but not byte or rune) type. Such conversions are discouraged because they
return the UTF-8 representation of the Unicode code point x, and not a decimal
string representation of x as one might expect. Furthermore, if x denotes an
invalid code point, the conversion cannot be statically rejected.

For conversions that intend on using the code point, consider replacing them
with string(rune(x)). Otherwise, strconv.Itoa and its equivalents return the
string representation of the value in the desired base.
`

var Analyzer = &analysis.Analyzer{
	Name:     "stringintconv",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func typeName(typ types.Type) string {
	if v, _ := typ.(interface{ Name() string }); v != nil {
		return v.Name()
	}
	if v, _ := typ.(interface{ Obj() *types.TypeName }); v != nil {
		return v.Obj().Name()
	}
	return ""
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)

		// Retrieve target type name.
		var tname *types.TypeName
		switch fun := call.Fun.(type) {
		case *ast.Ident:
			tname, _ = pass.TypesInfo.Uses[fun].(*types.TypeName)
		case *ast.SelectorExpr:
			tname, _ = pass.TypesInfo.Uses[fun.Sel].(*types.TypeName)
		}
		if tname == nil {
			return
		}
		target := tname.Name()

		// Check that target type T in T(v) has an underlying type of string.
		T, _ := tname.Type().Underlying().(*types.Basic)
		if T == nil || T.Kind() != types.String {
			return
		}
		if s := T.Name(); target != s {
			target += " (" + s + ")"
		}

		// Check that type V of v has an underlying integral type that is not byte or rune.
		if len(call.Args) != 1 {
			return
		}
		v := call.Args[0]
		vtyp := pass.TypesInfo.TypeOf(v)
		V, _ := vtyp.Underlying().(*types.Basic)
		if V == nil || V.Info()&types.IsInteger == 0 {
			return
		}
		switch V.Kind() {
		case types.Byte, types.Rune, types.UntypedRune:
			return
		}

		// Retrieve source type name.
		source := typeName(vtyp)
		if source == "" {
			return
		}
		if s := V.Name(); source != s {
			source += " (" + s + ")"
		}
		diag := analysis.Diagnostic{
			Pos:     n.Pos(),
			Message: fmt.Sprintf("conversion from %s to %s yields a string of one rune, not a string of digits (did you mean fmt.Sprint(x)?)", source, target),
			SuggestedFixes: []analysis.SuggestedFix{
				{
					Message: "Did you mean to convert a rune to a string?",
					TextEdits: []analysis.TextEdit{
						{
							Pos:     v.Pos(),
							End:     v.Pos(),
							NewText: []byte("rune("),
						},
						{
							Pos:     v.End(),
							End:     v.End(),
							NewText: []byte(")"),
						},
					},
				},
			},
		}
		pass.Report(diag)
	})
	return nil, nil
}
